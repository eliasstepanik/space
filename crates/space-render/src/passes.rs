//! GPU pipelines and the offscreen frame renderer.
//!
//! Frame order (the acceleration-structure builds run in their own
//! submissions, like `spikes/rq-triangle`, so no barrier subtlety can leak an
//! incomplete BVH into the ray queries):
//!
//! A. build BLAS (device-side) + zero the probe texture
//! B. build TLAS (device-side)
//! C. GI probe compute -> forward render pass (HDR) -> tone map -> readback

use ash::vk;

use crate::alloc::{Buffer, Image, SharedAllocator};
use crate::gpu::{RenderError, RenderOptions};
use crate::params::{self, ForwardParams, GiParams, RqParams, ToneParams};
use crate::scene::Scene;

/// SPIR-V modules compiled by `build.rs` (naga >= 30).
pub(crate) struct Shaders {
    pub rq_triangle: Vec<u32>,
    pub gi_probe: Vec<u32>,
    pub tonemap: Vec<u32>,
    pub forward_vs: Vec<u32>,
    pub forward_fs: Vec<u32>,
}

impl Shaders {
    #[must_use]
    pub fn from_spv() -> Self {
        #[allow(clippy::chunks_exact_to_as_chunks)]
        fn words(bytes: &[u8]) -> Vec<u32> {
            assert_eq!(bytes.len() % 4, 0, "SPIR-V must be whole words");
            bytes
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        }
        Self {
            rq_triangle: words(include_bytes!(concat!(env!("OUT_DIR"), "/rq_triangle.spv"))),
            gi_probe: words(include_bytes!(concat!(env!("OUT_DIR"), "/gi_probe.spv"))),
            tonemap: words(include_bytes!(concat!(env!("OUT_DIR"), "/tonemap.spv"))),
            forward_vs: words(include_bytes!(concat!(env!("OUT_DIR"), "/forward_vs.spv"))),
            forward_fs: words(include_bytes!(concat!(env!("OUT_DIR"), "/forward_fs.spv"))),
        }
    }
}

/// The scene uploaded to the GPU: geometry, instances and the acceleration
/// structures. The BLAS is created as BOTTOM_LEVEL and the TLAS as TOP_LEVEL —
/// hard constraint: only a **descriptor-bound** acceleration structure must be
/// GENERIC or TOP_LEVEL (VUID-...-03579), and only the TLAS is ever bound
/// (the shaders trace the scene, never the single BLAS directly), so each
/// structure is created with its honest type and an accidental descriptor
/// write of the BLAS is a validation error rather than silent UB.
pub(crate) struct SceneGpu {
    pub vertices: Buffer,
    pub indices: Buffer,
    pub instances: Buffer,
    pub instance_count: u32,
    pub mesh_vertex_count: u32,
    pub blas: vk::AccelerationStructureKHR,
    pub tlas: vk::AccelerationStructureKHR,
    pub blas_prim_count: u32,
    pub tlas_prim_count: u32,
    pub blas_scratch_addr: u64,
    pub tlas_scratch_addr: u64,
    _blas_buf: Buffer,
    _tlas_buf: Buffer,
    _blas_scratch: Buffer,
    _tlas_scratch: Buffer,
    _instance_as_buf: Buffer,
    accel: ash::khr::acceleration_structure::Device,
}

/// Long-lived GPU objects: shader modules, pipelines, layouts, the render
/// pass and the sampler.
pub(crate) struct Pipelines {
    modules: Vec<vk::ShaderModule>,
    pub forward_layout: vk::PipelineLayout,
    pub forward_set_layout: vk::DescriptorSetLayout,
    pub gi_layout: vk::PipelineLayout,
    pub gi_set_layout: vk::DescriptorSetLayout,
    pub tone_layout: vk::PipelineLayout,
    pub tone_set_layout: vk::DescriptorSetLayout,
    pub rq_layout: vk::PipelineLayout,
    pub rq_set_layout: vk::DescriptorSetLayout,
    pub render_pass: vk::RenderPass,
    pub graphics: vk::Pipeline,
    pub gi_pipeline: vk::Pipeline,
    pub tone_pipeline: vk::Pipeline,
    pub rq_pipeline: vk::Pipeline,
    pub sampler: vk::Sampler,
}

struct Created<'a> {
    device: &'a ash::Device,
}

impl<'a> Created<'a> {
    fn module(&self, code: &[u32], name: &'static str) -> Result<vk::ShaderModule, RenderError> {
        unsafe {
            self.device
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(code), None)
                .map_err(|e| RenderError::with_name(name, format!("create_shader_module: {e}")))
        }
    }

    fn compute_pipeline(
        &self,
        module: vk::ShaderModule,
        layout: vk::PipelineLayout,
        name: &'static str,
    ) -> Result<vk::Pipeline, RenderError> {
        unsafe {
            self.device
                .create_compute_pipelines(
                    vk::PipelineCache::null(),
                    &[vk::ComputePipelineCreateInfo::default()
                        .stage(
                            vk::PipelineShaderStageCreateInfo::default()
                                .stage(vk::ShaderStageFlags::COMPUTE)
                                .module(module)
                                .name(c"main"),
                        )
                        .layout(layout)],
                    None,
                )
                .map(|mut v| v.remove(0))
                .map_err(|(_, e)| RenderError::with_name(name, format!("compute pipeline: {e}")))
        }
    }
}

fn binding(
    binding: u32,
    ty: vk::DescriptorType,
    count: u32,
    stages: vk::ShaderStageFlags,
) -> vk::DescriptorSetLayoutBinding<'static> {
    vk::DescriptorSetLayoutBinding::default()
        .binding(binding)
        .descriptor_type(ty)
        .descriptor_count(count)
        .stage_flags(stages)
}

impl Pipelines {
    /// Create every shader module, layout and pipeline. `rq` gates the
    /// ray-query pipelines (they are still created when the device lacks ray
    /// query — creation is legal, use is not — but the GI pass is never
    /// dispatched without support).
    pub fn create(device: &ash::Device, shaders: &Shaders) -> Result<Self, RenderError> {
        let c = Created { device };
        let mut modules = Vec::new();
        let m_rq = c.module(&shaders.rq_triangle, "rq_triangle")?;
        let m_gi = c.module(&shaders.gi_probe, "gi_probe")?;
        let m_tone = c.module(&shaders.tonemap, "tonemap")?;
        let m_vs = c.module(&shaders.forward_vs, "forward_vs")?;
        let m_fs = c.module(&shaders.forward_fs, "forward_fs")?;
        modules.extend_from_slice(&[m_rq, m_gi, m_tone, m_vs, m_fs]);

        // ---- descriptor set layouts ------------------------------------------
        let forward_set_layout = unsafe {
            device
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&[
                        binding(
                            0,
                            vk::DescriptorType::UNIFORM_BUFFER,
                            1,
                            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                        ),
                        binding(
                            1,
                            vk::DescriptorType::STORAGE_BUFFER,
                            1,
                            vk::ShaderStageFlags::VERTEX,
                        ),
                        binding(
                            2,
                            vk::DescriptorType::SAMPLED_IMAGE,
                            1,
                            vk::ShaderStageFlags::FRAGMENT,
                        ),
                        binding(
                            3,
                            vk::DescriptorType::SAMPLER,
                            1,
                            vk::ShaderStageFlags::FRAGMENT,
                        ),
                    ]),
                    None,
                )
                .map_err(|e| RenderError::new(format!("forward set layout: {e}")))?
        };
        let gi_set_layout = unsafe {
            device
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&[
                        binding(
                            0,
                            vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
                            1,
                            vk::ShaderStageFlags::COMPUTE,
                        ),
                        binding(
                            1,
                            vk::DescriptorType::UNIFORM_BUFFER,
                            1,
                            vk::ShaderStageFlags::COMPUTE,
                        ),
                        binding(
                            2,
                            vk::DescriptorType::STORAGE_BUFFER,
                            1,
                            vk::ShaderStageFlags::COMPUTE,
                        ),
                        binding(
                            3,
                            vk::DescriptorType::STORAGE_IMAGE,
                            1,
                            vk::ShaderStageFlags::COMPUTE,
                        ),
                    ]),
                    None,
                )
                .map_err(|e| RenderError::new(format!("gi set layout: {e}")))?
        };
        let tone_set_layout = unsafe {
            device
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&[
                        binding(
                            0,
                            vk::DescriptorType::SAMPLED_IMAGE,
                            1,
                            vk::ShaderStageFlags::COMPUTE,
                        ),
                        binding(
                            1,
                            vk::DescriptorType::SAMPLER,
                            1,
                            vk::ShaderStageFlags::COMPUTE,
                        ),
                        binding(
                            2,
                            vk::DescriptorType::STORAGE_IMAGE,
                            1,
                            vk::ShaderStageFlags::COMPUTE,
                        ),
                        binding(
                            3,
                            vk::DescriptorType::UNIFORM_BUFFER,
                            1,
                            vk::ShaderStageFlags::COMPUTE,
                        ),
                    ]),
                    None,
                )
                .map_err(|e| RenderError::new(format!("tone set layout: {e}")))?
        };
        let rq_set_layout = unsafe {
            device
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&[
                        binding(
                            0,
                            vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
                            1,
                            vk::ShaderStageFlags::COMPUTE,
                        ),
                        binding(
                            1,
                            vk::DescriptorType::STORAGE_IMAGE,
                            1,
                            vk::ShaderStageFlags::COMPUTE,
                        ),
                        binding(
                            2,
                            vk::DescriptorType::UNIFORM_BUFFER,
                            1,
                            vk::ShaderStageFlags::COMPUTE,
                        ),
                    ]),
                    None,
                )
                .map_err(|e| RenderError::new(format!("rq set layout: {e}")))?
        };

        let layouts = [
            forward_set_layout,
            gi_set_layout,
            tone_set_layout,
            rq_set_layout,
        ];
        let pl = |layout: &vk::PipelineLayoutCreateInfo<'_>, name: &str| unsafe {
            device
                .create_pipeline_layout(layout, None)
                .map_err(|e| RenderError::new(format!("{name}: {e}")))
        };
        let forward_layout = pl(
            &vk::PipelineLayoutCreateInfo::default()
                .set_layouts(std::slice::from_ref(&forward_set_layout)),
            "forward pipeline layout",
        )?;
        let gi_layout = pl(
            &vk::PipelineLayoutCreateInfo::default()
                .set_layouts(std::slice::from_ref(&gi_set_layout)),
            "gi pipeline layout",
        )?;
        let tone_layout = pl(
            &vk::PipelineLayoutCreateInfo::default()
                .set_layouts(std::slice::from_ref(&tone_set_layout)),
            "tone pipeline layout",
        )?;
        let rq_layout = pl(
            &vk::PipelineLayoutCreateInfo::default()
                .set_layouts(std::slice::from_ref(&rq_set_layout)),
            "rq pipeline layout",
        )?;
        let _ = layouts;

        // ---- render pass: HDR colour + depth ---------------------------------
        let render_pass = unsafe {
            device
                .create_render_pass(
                    &vk::RenderPassCreateInfo::default()
                        .attachments(&[
                            vk::AttachmentDescription::default()
                                .format(vk::Format::R16G16B16A16_SFLOAT)
                                .samples(vk::SampleCountFlags::TYPE_1)
                                .load_op(vk::AttachmentLoadOp::CLEAR)
                                .store_op(vk::AttachmentStoreOp::STORE)
                                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                                .initial_layout(vk::ImageLayout::UNDEFINED)
                                .final_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
                            vk::AttachmentDescription::default()
                                .format(vk::Format::D32_SFLOAT)
                                .samples(vk::SampleCountFlags::TYPE_1)
                                .load_op(vk::AttachmentLoadOp::CLEAR)
                                .store_op(vk::AttachmentStoreOp::DONT_CARE)
                                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                                .initial_layout(vk::ImageLayout::UNDEFINED)
                                // `separateDepthStencilLayouts` is not enabled,
                                // so the combined layout is the only legal one.
                                .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL),
                        ])
                        .subpasses(&[vk::SubpassDescription::default()
                            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                            .color_attachments(&[vk::AttachmentReference::default()
                                .attachment(0)
                                .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)])
                            .depth_stencil_attachment(
                                &vk::AttachmentReference::default()
                                    .attachment(1)
                                    .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL),
                            )])
                        .dependencies(&[vk::SubpassDependency::default()
                            .src_subpass(vk::SUBPASS_EXTERNAL)
                            .dst_subpass(0)
                            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
                            .src_access_mask(vk::AccessFlags::empty())
                            .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
                            .dst_access_mask(
                                vk::AccessFlags::COLOR_ATTACHMENT_READ
                                    | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                            )]),
                    None,
                )
                .map_err(|e| RenderError::new(format!("render pass: {e}")))?
        };

        // ---- graphics pipeline -------------------------------------------------
        let graphics = unsafe {
            device
                .create_graphics_pipelines(
                    vk::PipelineCache::null(),
                    &[vk::GraphicsPipelineCreateInfo::default()
                        .stages(&[
                            vk::PipelineShaderStageCreateInfo::default()
                                .stage(vk::ShaderStageFlags::VERTEX)
                                .module(m_vs)
                                .name(c"vs_main"),
                            vk::PipelineShaderStageCreateInfo::default()
                                .stage(vk::ShaderStageFlags::FRAGMENT)
                                .module(m_fs)
                                .name(c"fs_main"),
                        ])
                        .vertex_input_state(&vk::PipelineVertexInputStateCreateInfo::default())
                        .input_assembly_state(
                            &vk::PipelineInputAssemblyStateCreateInfo::default()
                                .topology(vk::PrimitiveTopology::TRIANGLE_LIST),
                        )
                        .viewport_state(
                            &vk::PipelineViewportStateCreateInfo::default()
                                .viewport_count(1)
                                .scissor_count(1),
                        )
                        .rasterization_state(
                            &vk::PipelineRasterizationStateCreateInfo::default()
                                .depth_clamp_enable(false)
                                .rasterizer_discard_enable(false)
                                .polygon_mode(vk::PolygonMode::FILL)
                                .cull_mode(vk::CullModeFlags::NONE)
                                .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
                                .depth_bias_enable(false)
                                .line_width(1.0),
                        )
                        .multisample_state(
                            &vk::PipelineMultisampleStateCreateInfo::default()
                                .rasterization_samples(vk::SampleCountFlags::TYPE_1),
                        )
                        .depth_stencil_state(
                            &vk::PipelineDepthStencilStateCreateInfo::default()
                                .depth_test_enable(true)
                                .depth_write_enable(true)
                                .depth_compare_op(vk::CompareOp::LESS),
                        )
                        .color_blend_state(
                            &vk::PipelineColorBlendStateCreateInfo::default()
                                .logic_op_enable(false)
                                .logic_op(vk::LogicOp::COPY)
                                .attachments(&[vk::PipelineColorBlendAttachmentState::default()
                                    .blend_enable(false)
                                    .color_write_mask(vk::ColorComponentFlags::RGBA)]),
                        )
                        .dynamic_state(
                            &vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&[
                                vk::DynamicState::VIEWPORT,
                                vk::DynamicState::SCISSOR,
                            ]),
                        )
                        .layout(forward_layout)
                        .render_pass(render_pass)
                        .subpass(0)],
                    None,
                )
                .map(|mut v| v.remove(0))
                .map_err(|(_, e)| RenderError::new(format!("graphics pipeline: {e}")))?
        };

        let gi_pipeline = c.compute_pipeline(m_gi, gi_layout, "gi pipeline")?;
        let tone_pipeline = c.compute_pipeline(m_tone, tone_layout, "tone pipeline")?;
        let rq_pipeline = c.compute_pipeline(m_rq, rq_layout, "rq pipeline")?;

        let sampler = unsafe {
            device
                .create_sampler(
                    &vk::SamplerCreateInfo::default()
                        .mag_filter(vk::Filter::NEAREST)
                        .min_filter(vk::Filter::NEAREST)
                        .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
                        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                        .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE),
                    None,
                )
                .map_err(|e| RenderError::new(format!("sampler: {e}")))?
        };

        Ok(Self {
            modules,
            forward_layout,
            forward_set_layout,
            gi_layout,
            gi_set_layout,
            tone_layout,
            tone_set_layout,
            rq_layout,
            rq_set_layout,
            render_pass,
            graphics,
            gi_pipeline,
            tone_pipeline,
            rq_pipeline,
            sampler,
        })
    }
}

/// Row-major 3x4 identity transform for a TLAS instance.
impl SceneGpu {
    /// Upload the scene's CPU data and build the BLAS/TLAS *inputs* (the
    /// actual device-side builds happen in `render_*`).
    pub fn upload(
        device: &ash::Device,
        accel: &ash::khr::acceleration_structure::Device,
        allocator: &SharedAllocator,
        scene: &Scene,
    ) -> Result<Self, RenderError> {
        let (vertices, indices) = scene.triangle_soup();
        let vertex_bytes: Vec<u8> = vertices
            .iter()
            .flat_map(|v| v.iter().flat_map(|c| c.to_le_bytes()))
            .collect();
        let index_bytes: Vec<u8> = indices.iter().flat_map(|i| i.to_le_bytes()).collect();

        let geom_usage = vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;
        let vertex_buf = Buffer::new(
            device,
            allocator,
            vertex_bytes.len().max(16) as u64,
            geom_usage,
            gpu_allocator::MemoryLocation::CpuToGpu,
            "blas-vertices",
        )?;
        vertex_buf.write(&vertex_bytes)?;
        let index_buf = Buffer::new(
            device,
            allocator,
            index_bytes.len().max(16) as u64,
            geom_usage,
            gpu_allocator::MemoryLocation::CpuToGpu,
            "blas-indices",
        )?;
        index_buf.write(&index_bytes)?;

        // Render-instance storage buffer (one per mesh).
        let inst_bytes: Vec<u8> = scene
            .instances
            .iter()
            .flat_map(|i| params::as_bytes(i).to_vec())
            .collect();
        let instance_buf = Buffer::new(
            device,
            allocator,
            inst_bytes.len().max(64) as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            gpu_allocator::MemoryLocation::CpuToGpu,
            "render-instances",
        )?;
        instance_buf.write(&inst_bytes)?;

        let as_usage = vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;

        // ---- BLAS -------------------------------------------------------------
        let geometry = vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
            .flags(vk::GeometryFlagsKHR::empty())
            .geometry(vk::AccelerationStructureGeometryDataKHR {
                triangles: vk::AccelerationStructureGeometryTrianglesDataKHR::default()
                    .vertex_format(vk::Format::R32G32B32_SFLOAT)
                    .vertex_data(vk::DeviceOrHostAddressConstKHR {
                        device_address: vertex_buf.device_address(),
                    })
                    .vertex_stride(12)
                    .max_vertex((vertices.len().max(1) - 1) as u32)
                    .index_type(vk::IndexType::UINT32)
                    .index_data(vk::DeviceOrHostAddressConstKHR {
                        device_address: index_buf.device_address(),
                    }),
            });
        let blas_build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .geometries(std::slice::from_ref(&geometry));
        let prim_count = (indices.len() / 3) as u32;
        let blas_sizes = build_sizes(accel, &blas_build_info, prim_count);
        let blas_buf = Buffer::new(
            device,
            allocator,
            blas_sizes.acceleration_structure_size.max(16),
            as_usage,
            gpu_allocator::MemoryLocation::GpuOnly,
            "blas",
        )?;
        let blas_scratch = Buffer::new(
            device,
            allocator,
            blas_sizes.build_scratch_size.max(256),
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            gpu_allocator::MemoryLocation::GpuOnly,
            "blas-scratch",
        )?;
        // BOTTOM_LEVEL: the BLAS is never descriptor-bound (only the TLAS
        // is), so it is created with the type it is actually built as.
        let blas = unsafe {
            accel
                .create_acceleration_structure(
                    &vk::AccelerationStructureCreateInfoKHR::default()
                        .buffer(blas_buf.handle)
                        .size(blas_sizes.acceleration_structure_size)
                        .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL),
                    None,
                )
                .map_err(|e| RenderError::new(format!("create BLAS: {e}")))?
        };
        // ---- TLAS ---------------------------------------------------------------
        // One TLAS instance per render instance, all referencing the single BLAS.
        let tlas_instances: Vec<vk::AccelerationStructureInstanceKHR> = scene
            .instances
            .iter()
            .map(|inst| vk::AccelerationStructureInstanceKHR {
                transform: vk::TransformMatrixKHR {
                    matrix: [
                        inst.center[3],
                        0.0,
                        0.0,
                        inst.center[0],
                        0.0,
                        inst.center[3],
                        0.0,
                        inst.center[1],
                        0.0,
                        0.0,
                        inst.center[3],
                        inst.center[2],
                    ],
                },
                instance_custom_index_and_mask: vk::Packed24_8::new(0, 0xff),
                instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(0, 0),
                acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
                    device_handle: blas_buf.device_address(),
                },
            })
            .collect();
        let inst_as_bytes: Vec<u8> = bytemuck_instances(&tlas_instances);
        let instance_as_buf = Buffer::new(
            device,
            allocator,
            inst_as_bytes.len().max(16) as u64,
            geom_usage,
            gpu_allocator::MemoryLocation::CpuToGpu,
            "tlas-instances",
        )?;
        instance_as_buf.write(&inst_as_bytes)?;

        let tlas_geometry = vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(vk::GeometryTypeKHR::INSTANCES)
            .flags(vk::GeometryFlagsKHR::empty())
            .geometry(vk::AccelerationStructureGeometryDataKHR {
                instances: vk::AccelerationStructureGeometryInstancesDataKHR::default()
                    .array_of_pointers(false)
                    .data(vk::DeviceOrHostAddressConstKHR {
                        device_address: instance_as_buf.device_address(),
                    }),
            });
        let tlas_build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .geometries(std::slice::from_ref(&tlas_geometry));
        let tlas_sizes = build_sizes(accel, &tlas_build_info, scene.instances.len() as u32);
        let tlas_buf = Buffer::new(
            device,
            allocator,
            tlas_sizes.acceleration_structure_size.max(16),
            as_usage,
            gpu_allocator::MemoryLocation::GpuOnly,
            "tlas",
        )?;
        let tlas_scratch = Buffer::new(
            device,
            allocator,
            tlas_sizes.build_scratch_size.max(256),
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            gpu_allocator::MemoryLocation::GpuOnly,
            "tlas-scratch",
        )?;
        // TOP_LEVEL: descriptor-bound acceleration structures must be
        // TOP_LEVEL or GENERIC (hard constraint).
        let tlas = unsafe {
            accel
                .create_acceleration_structure(
                    &vk::AccelerationStructureCreateInfoKHR::default()
                        .buffer(tlas_buf.handle)
                        .size(tlas_sizes.acceleration_structure_size)
                        .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL),
                    None,
                )
                .map_err(|e| RenderError::new(format!("create TLAS: {e}")))?
        };
        Ok(Self {
            vertices: vertex_buf,
            indices: index_buf,
            instances: instance_buf,
            instance_count: scene.instances.len() as u32,
            // The draw is instanced with a fixed vertex count; the box vertex
            // shader wraps its face index (`% 6u`) so drawing the sphere's 60
            // vertices draws each box face once and then repeats some, which
            // is harmless overdraw. The count is therefore the largest mesh.
            mesh_vertex_count: scene
                .meshes
                .iter()
                .map(|m| m.len() as u32)
                .max()
                .unwrap_or(0),
            blas,
            tlas,
            blas_prim_count: prim_count,
            tlas_prim_count: scene.instances.len() as u32,
            blas_scratch_addr: blas_scratch.device_address(),
            tlas_scratch_addr: tlas_scratch.device_address(),
            _blas_buf: blas_buf,
            _tlas_buf: tlas_buf,
            _blas_scratch: blas_scratch,
            _tlas_scratch: tlas_scratch,
            _instance_as_buf: instance_as_buf,
            accel: accel.clone(),
        })
    }
}

impl Drop for SceneGpu {
    fn drop(&mut self) {
        unsafe {
            self.accel.destroy_acceleration_structure(self.tlas, None);
            self.accel.destroy_acceleration_structure(self.blas, None);
        }
    }
}

impl SceneGpu {
    /// Device address of the TLAS instance array (the TLAS build input).
    pub fn tlas_instance_addr(&self) -> u64 {
        self._instance_as_buf.device_address()
    }
}

impl Pipelines {
    /// Destroy every long-lived pipeline object. Called by `RenderContext::drop`.
    pub(crate) unsafe fn destroy(&self, device: &ash::Device) {
        for m in &self.modules {
            device.destroy_shader_module(*m, None);
        }
        device.destroy_pipeline_layout(self.forward_layout, None);
        device.destroy_pipeline_layout(self.gi_layout, None);
        device.destroy_pipeline_layout(self.tone_layout, None);
        device.destroy_pipeline_layout(self.rq_layout, None);
        device.destroy_descriptor_set_layout(self.forward_set_layout, None);
        device.destroy_descriptor_set_layout(self.gi_set_layout, None);
        device.destroy_descriptor_set_layout(self.tone_set_layout, None);
        device.destroy_descriptor_set_layout(self.rq_set_layout, None);
        device.destroy_render_pass(self.render_pass, None);
        device.destroy_pipeline(self.graphics, None);
        device.destroy_pipeline(self.gi_pipeline, None);
        device.destroy_pipeline(self.tone_pipeline, None);
        device.destroy_pipeline(self.rq_pipeline, None);
        device.destroy_sampler(self.sampler, None);
    }
}

fn build_sizes(
    accel: &ash::khr::acceleration_structure::Device,
    info: &vk::AccelerationStructureBuildGeometryInfoKHR<'_>,
    prim_count: u32,
) -> vk::AccelerationStructureBuildSizesInfoKHR<'static> {
    let mut sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
    unsafe {
        accel.get_acceleration_structure_build_sizes(
            vk::AccelerationStructureBuildTypeKHR::DEVICE,
            info,
            &[prim_count],
            &mut sizes,
        )
    };
    sizes
}

/// Byte view of a slice of `AccelerationStructureInstanceKHR` (plain POD).
fn bytemuck_instances(instances: &[vk::AccelerationStructureInstanceKHR]) -> Vec<u8> {
    unsafe {
        std::slice::from_raw_parts(
            instances.as_ptr() as *const u8,
            std::mem::size_of_val(instances),
        )
    }
    .to_vec()
}

/// Per-frame GPU resources (images, views, framebuffers, uniform buffers,
/// descriptors, command buffers). Everything is RAII; dropping the frame frees
/// every dedicated allocation.
///
/// This is an opaque handle: it is returned by
/// [`RenderContext::render_frame_gpu`](crate::RenderContext::render_frame_gpu)
/// and consumed by [`Swapchain::present`](crate::Swapchain::present). Its
/// fields are crate-private so callers treat it as a token.
pub struct FrameGpu {
    pub(crate) width: u32,
    pub(crate) height: u32,
    // `hdr`/`depth` are never read directly: they are held so their dedicated
    // allocations outlive the framebuffer and views that reference them.
    #[allow(dead_code)]
    pub(crate) hdr: Image,
    pub(crate) hdr_view: vk::ImageView,
    #[allow(dead_code)]
    pub(crate) depth: Image,
    pub(crate) depth_view: vk::ImageView,
    pub(crate) out: Image,
    pub(crate) out_view: vk::ImageView,
    pub(crate) framebuffer: vk::Framebuffer,
    pub(crate) gi_tex: Option<(Image, vk::ImageView)>,
    pub(crate) gi_extent: (u32, u32),
    pub(crate) dummy_tex: (Image, vk::ImageView),
    pub(crate) zero_buf: Buffer,
    pub(crate) readback: Buffer,
    pub(crate) forward_params: Buffer,
    pub(crate) gi_params: Buffer,
    pub(crate) tone_params: Buffer,
    pub(crate) rq_params: Buffer,
    pub(crate) pool: vk::DescriptorPool,
    pub(crate) forward_set: vk::DescriptorSet,
    pub(crate) gi_set: vk::DescriptorSet,
    pub(crate) tone_set: vk::DescriptorSet,
    pub(crate) rq_set: vk::DescriptorSet,
    pub(crate) cmd_build_blas: vk::CommandBuffer,
    pub(crate) cmd_build_tlas: vk::CommandBuffer,
    pub(crate) cmd_frame: vk::CommandBuffer,
    pub(crate) fence: vk::Fence,
    pub(crate) device: ash::Device,
}

fn aspect_for(format: vk::Format) -> vk::ImageAspectFlags {
    match format {
        vk::Format::D16_UNORM | vk::Format::D32_SFLOAT | vk::Format::X8_D24_UNORM_PACK32 => {
            vk::ImageAspectFlags::DEPTH
        }
        vk::Format::D24_UNORM_S8_UINT | vk::Format::D32_SFLOAT_S8_UINT => {
            vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
        }
        vk::Format::S8_UINT => vk::ImageAspectFlags::STENCIL,
        _ => vk::ImageAspectFlags::COLOR,
    }
}

fn make_image(
    device: &ash::Device,
    allocator: &SharedAllocator,
    width: u32,
    height: u32,
    format: vk::Format,
    usage: vk::ImageUsageFlags,
    name: &'static str,
) -> Result<(Image, vk::ImageView), RenderError> {
    let image = Image::new(
        device,
        allocator,
        &vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(usage),
        gpu_allocator::MemoryLocation::GpuOnly,
        name,
    )?;
    let view = unsafe {
        device
            .create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image.handle)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(format)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: aspect_for(format),
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    }),
                None,
            )
            .map_err(|e| RenderError::with_name(name, format!("create_image_view: {e}")))?
    };
    Ok((image, view))
}

const fn color_range() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn create_frame(
    device: &ash::Device,
    accel: &ash::khr::acceleration_structure::Device,
    allocator: &SharedAllocator,
    pipelines: &Pipelines,
    scene: &SceneGpu,
    queue: vk::Queue,
    cmd_pool: vk::CommandPool,
    width: u32,
    height: u32,
    options: &RenderOptions,
    depth_format: vk::Format,
) -> Result<FrameGpu, RenderError> {
    let _ = accel;
    let hdr_usage = vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED;
    let (hdr, hdr_view) = make_image(
        device,
        allocator,
        width,
        height,
        vk::Format::R16G16B16A16_SFLOAT,
        hdr_usage,
        "hdr-image",
    )?;
    let (depth, depth_view) = make_image(
        device,
        allocator,
        width,
        height,
        depth_format,
        vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
        "depth-image",
    )?;
    let (out, out_view) = make_image(
        device,
        allocator,
        width,
        height,
        vk::Format::R8G8B8A8_UNORM,
        vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
        "out-image",
    )?;
    let framebuffer = unsafe {
        device
            .create_framebuffer(
                &vk::FramebufferCreateInfo::default()
                    .render_pass(pipelines.render_pass)
                    .attachments(&[hdr_view, depth_view])
                    .width(width)
                    .height(height)
                    .layers(1),
                None,
            )
            .map_err(|e| RenderError::new(format!("framebuffer: {e}")))?
    };

    // GI probe texture: grid x (grid*6) rgba16f, plus a 1x1 dummy used when
    // GI is disabled so the forward descriptor is always complete.
    let gi_tex = if options.gi_enabled {
        let q = options.quality;
        let (img, view) = make_image(
            device,
            allocator,
            q.grid,
            q.grid * 6,
            vk::Format::R16G16B16A16_SFLOAT,
            vk::ImageUsageFlags::STORAGE
                | vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::TRANSFER_DST,
            "gi-probe-tex",
        )?;
        Some((img, view))
    } else {
        None
    };
    let gi_extent = if options.gi_enabled {
        (options.quality.grid, options.quality.grid * 6)
    } else {
        (0, 0)
    };
    let (dummy_tex, dummy_view) = make_image(
        device,
        allocator,
        1,
        1,
        vk::Format::R16G16B16A16_SFLOAT,
        vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE,
        "gi-dummy-tex",
    )?;
    let gi_view_active = gi_tex.as_ref().map_or(dummy_view, |(_, v)| *v);

    // Zero buffer used to clear the probe texture deterministically (the
    // mutation test relies on the texture being *zero*, not undefined).
    let zero_buf = Buffer::new(
        device,
        allocator,
        u64::from(options.quality.grid) * u64::from(options.quality.grid) * 6 * 8,
        vk::BufferUsageFlags::TRANSFER_SRC,
        gpu_allocator::MemoryLocation::CpuToGpu,
        "gi-zero",
    )?;

    let readback = Buffer::new(
        device,
        allocator,
        u64::from(width) * u64::from(height) * 4,
        vk::BufferUsageFlags::TRANSFER_DST,
        gpu_allocator::MemoryLocation::GpuToCpu,
        "readback",
    )?;

    let ubo_usage = vk::BufferUsageFlags::UNIFORM_BUFFER;
    let forward_params = Buffer::new(
        device,
        allocator,
        std::mem::size_of::<ForwardParams>() as u64,
        ubo_usage,
        gpu_allocator::MemoryLocation::CpuToGpu,
        "forward-params",
    )?;
    let gi_params = Buffer::new(
        device,
        allocator,
        std::mem::size_of::<GiParams>() as u64,
        ubo_usage,
        gpu_allocator::MemoryLocation::CpuToGpu,
        "gi-params",
    )?;
    let tone_params = Buffer::new(
        device,
        allocator,
        std::mem::size_of::<ToneParams>() as u64,
        ubo_usage,
        gpu_allocator::MemoryLocation::CpuToGpu,
        "tone-params",
    )?;
    let rq_params = Buffer::new(
        device,
        allocator,
        std::mem::size_of::<RqParams>() as u64,
        ubo_usage,
        gpu_allocator::MemoryLocation::CpuToGpu,
        "rq-params",
    )?;

    // ---- descriptors -----------------------------------------------------------
    let pool = unsafe {
        device
            .create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(4)
                    .pool_sizes(&[
                        vk::DescriptorPoolSize {
                            ty: vk::DescriptorType::UNIFORM_BUFFER,
                            descriptor_count: 4,
                        },
                        vk::DescriptorPoolSize {
                            ty: vk::DescriptorType::STORAGE_BUFFER,
                            descriptor_count: 2,
                        },
                        vk::DescriptorPoolSize {
                            ty: vk::DescriptorType::SAMPLED_IMAGE,
                            descriptor_count: 2,
                        },
                        vk::DescriptorPoolSize {
                            ty: vk::DescriptorType::SAMPLER,
                            descriptor_count: 2,
                        },
                        vk::DescriptorPoolSize {
                            ty: vk::DescriptorType::STORAGE_IMAGE,
                            descriptor_count: 3,
                        },
                        vk::DescriptorPoolSize {
                            ty: vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
                            descriptor_count: 2,
                        },
                    ]),
                None,
            )
            .map_err(|e| RenderError::new(format!("descriptor pool: {e}")))?
    };
    let alloc_set = |layout: vk::DescriptorSetLayout| -> Result<vk::DescriptorSet, RenderError> {
        unsafe {
            device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(pool)
                        .set_layouts(std::slice::from_ref(&layout)),
                )
                .map(|mut v| v.remove(0))
                .map_err(|e| RenderError::new(format!("allocate_descriptor_sets: {e}")))
        }
    };
    let forward_set = alloc_set(pipelines.forward_set_layout)?;
    let gi_set = alloc_set(pipelines.gi_set_layout)?;
    let tone_set = alloc_set(pipelines.tone_set_layout)?;
    let rq_set = alloc_set(pipelines.rq_set_layout)?;

    let ub = |buf: &Buffer, size: u64| {
        vk::DescriptorBufferInfo::default()
            .buffer(buf.handle)
            .offset(0)
            .range(size)
    };
    let img_info = |view: vk::ImageView, layout: vk::ImageLayout| {
        vk::DescriptorImageInfo::default()
            .image_view(view)
            .image_layout(layout)
    };
    let mut as_write = vk::WriteDescriptorSetAccelerationStructureKHR::default()
        .acceleration_structures(std::slice::from_ref(&scene.tlas));
    let mut as_write_rq = vk::WriteDescriptorSetAccelerationStructureKHR::default()
        .acceleration_structures(std::slice::from_ref(&scene.tlas));

    unsafe {
        device.update_descriptor_sets(
            &[
                // forward: 0 params, 1 instances, 2 gi tex, 3 sampler
                vk::WriteDescriptorSet::default()
                    .dst_set(forward_set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(&[ub(
                        &forward_params,
                        std::mem::size_of::<ForwardParams>() as u64,
                    )]),
                vk::WriteDescriptorSet::default()
                    .dst_set(forward_set)
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&[ub(
                        &scene.instances,
                        u64::from(scene.instance_count.max(1)) * 64,
                    )]),
                vk::WriteDescriptorSet::default()
                    .dst_set(forward_set)
                    .dst_binding(2)
                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                    .image_info(&[img_info(
                        gi_view_active,
                        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    )]),
                vk::WriteDescriptorSet::default()
                    .dst_set(forward_set)
                    .dst_binding(3)
                    .descriptor_type(vk::DescriptorType::SAMPLER)
                    .image_info(&[vk::DescriptorImageInfo::default().sampler(pipelines.sampler)]),
                // gi: 0 TLAS, 1 params, 2 instances, 3 out
                vk::WriteDescriptorSet::default()
                    .dst_set(gi_set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                    .descriptor_count(1)
                    .dst_array_element(0)
                    .push_next(&mut as_write),
                vk::WriteDescriptorSet::default()
                    .dst_set(gi_set)
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(&[ub(&gi_params, std::mem::size_of::<GiParams>() as u64)]),
                vk::WriteDescriptorSet::default()
                    .dst_set(gi_set)
                    .dst_binding(2)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&[ub(
                        &scene.instances,
                        u64::from(scene.instance_count.max(1)) * 64,
                    )]),
                vk::WriteDescriptorSet::default()
                    .dst_set(gi_set)
                    .dst_binding(3)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(&[img_info(gi_view_active, vk::ImageLayout::GENERAL)]),
                // tone: 0 hdr tex, 1 sampler, 2 out, 3 params
                vk::WriteDescriptorSet::default()
                    .dst_set(tone_set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                    .image_info(&[img_info(
                        hdr_view,
                        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    )]),
                vk::WriteDescriptorSet::default()
                    .dst_set(tone_set)
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::SAMPLER)
                    .image_info(&[vk::DescriptorImageInfo::default().sampler(pipelines.sampler)]),
                vk::WriteDescriptorSet::default()
                    .dst_set(tone_set)
                    .dst_binding(2)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(&[img_info(out_view, vk::ImageLayout::GENERAL)]),
                vk::WriteDescriptorSet::default()
                    .dst_set(tone_set)
                    .dst_binding(3)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(&[ub(&tone_params, std::mem::size_of::<ToneParams>() as u64)]),
                // rq: 0 TLAS, 1 out storage image, 2 params
                vk::WriteDescriptorSet::default()
                    .dst_set(rq_set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                    .descriptor_count(1)
                    .dst_array_element(0)
                    .push_next(&mut as_write_rq),
                vk::WriteDescriptorSet::default()
                    .dst_set(rq_set)
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(&[img_info(out_view, vk::ImageLayout::GENERAL)]),
                vk::WriteDescriptorSet::default()
                    .dst_set(rq_set)
                    .dst_binding(2)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(&[ub(&rq_params, std::mem::size_of::<RqParams>() as u64)]),
            ],
            &[],
        );
    }

    let cmds = unsafe {
        device
            .allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(cmd_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(3),
            )
            .map_err(|e| RenderError::new(format!("allocate_command_buffers: {e}")))?
    };
    let [cmd_build_blas, cmd_build_tlas, cmd_frame] = <[vk::CommandBuffer; 3]>::try_from(cmds)
        .map_err(|v: Vec<vk::CommandBuffer>| {
            RenderError::new(format!("expected 3 cmd buffers, got {}", v.len()))
        })?;
    let fence = unsafe {
        device
            .create_fence(&vk::FenceCreateInfo::default(), None)
            .map_err(|e| RenderError::new(format!("create_fence: {e}")))?
    };
    let _ = queue;

    Ok(FrameGpu {
        width,
        height,
        hdr,
        hdr_view,
        depth,
        depth_view,
        out,
        out_view,
        framebuffer,
        gi_tex,
        gi_extent,
        dummy_tex: (dummy_tex, dummy_view),
        zero_buf,
        readback,
        forward_params,
        gi_params,
        tone_params,
        rq_params,
        pool,
        forward_set,
        gi_set,
        tone_set,
        rq_set,
        cmd_build_blas,
        cmd_build_tlas,
        cmd_frame,
        fence,
        device: device.clone(),
    })
}

impl FrameGpu {
    fn begin(cmd: vk::CommandBuffer, device: &ash::Device) -> Result<(), RenderError> {
        unsafe {
            device
                .begin_command_buffer(
                    cmd,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .map_err(|e| RenderError::new(format!("begin_command_buffer: {e}")))
        }
    }

    fn end(cmd: vk::CommandBuffer, device: &ash::Device) -> Result<(), RenderError> {
        unsafe {
            device
                .end_command_buffer(cmd)
                .map_err(|e| RenderError::new(format!("end_command_buffer: {e}")))
        }
    }

    pub(crate) fn submit(
        device: &ash::Device,
        queue: vk::Queue,
        cmd: vk::CommandBuffer,
        fence: vk::Fence,
    ) -> Result<(), RenderError> {
        let submit = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd));
        unsafe {
            device
                .queue_submit(queue, std::slice::from_ref(&submit), fence)
                .map_err(|e| RenderError::new(format!("queue_submit: {e}")))?;
            device
                .wait_for_fences(std::slice::from_ref(&fence), true, u64::MAX)
                .map_err(|e| RenderError::new(format!("wait_for_fences: {e}")))?;
            device
                .reset_fences(std::slice::from_ref(&fence))
                .map_err(|e| RenderError::new(format!("reset_fences: {e}")))?;
        }
        Ok(())
    }

    /// Submission A: build the BLAS device-side and zero the probe texture.
    /// The BLAS build runs alone (like `spikes/rq-triangle`) so no barrier
    /// subtlety can leak an incomplete BVH into later ray queries.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_build_blas(
        &self,
        device: &ash::Device,
        accel: &ash::khr::acceleration_structure::Device,
        scene: &SceneGpu,
        vertex_addr: u64,
        index_addr: u64,
        max_vertex: u32,
    ) -> Result<(), RenderError> {
        Self::begin(self.cmd_build_blas, device)?;
        let geometry = vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
            .flags(vk::GeometryFlagsKHR::empty())
            .geometry(vk::AccelerationStructureGeometryDataKHR {
                triangles: vk::AccelerationStructureGeometryTrianglesDataKHR::default()
                    .vertex_format(vk::Format::R32G32B32_SFLOAT)
                    .vertex_data(vk::DeviceOrHostAddressConstKHR {
                        device_address: vertex_addr,
                    })
                    .vertex_stride(12)
                    .max_vertex(max_vertex)
                    .index_type(vk::IndexType::UINT32)
                    .index_data(vk::DeviceOrHostAddressConstKHR {
                        device_address: index_addr,
                    }),
            });
        let build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .dst_acceleration_structure(scene.blas)
            .scratch_data(vk::DeviceOrHostAddressKHR {
                device_address: scene.blas_scratch_addr,
            })
            .geometries(std::slice::from_ref(&geometry));
        let ranges = [vk::AccelerationStructureBuildRangeInfoKHR {
            primitive_count: scene.blas_prim_count,
            primitive_offset: 0,
            first_vertex: 0,
            transform_offset: 0,
        }];
        unsafe {
            accel.cmd_build_acceleration_structures(
                self.cmd_build_blas,
                std::slice::from_ref(&build_info),
                &[&ranges],
            );
        }
        // Deterministically zero the probe texture (buffer -> image), so a
        // skipped GI dispatch reads zeros, not undefined memory.
        if let Some((gi_img, _)) = &self.gi_tex {
            unsafe {
                device.cmd_pipeline_barrier(
                    self.cmd_build_blas,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::default(),
                    &[],
                    &[],
                    &[vk::ImageMemoryBarrier::default()
                        .image(gi_img.handle)
                        .old_layout(vk::ImageLayout::UNDEFINED)
                        .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                        .src_access_mask(vk::AccessFlags::empty())
                        .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                        .subresource_range(color_range())],
                );
                device.cmd_copy_buffer_to_image(
                    self.cmd_build_blas,
                    self.zero_buf.handle,
                    gi_img.handle,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[vk::BufferImageCopy::default()
                        .buffer_row_length(0)
                        .buffer_image_height(0)
                        .image_subresource(vk::ImageSubresourceLayers {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            mip_level: 0,
                            base_array_layer: 0,
                            layer_count: 1,
                        })
                        .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                        .image_extent(vk::Extent3D {
                            width: self.gi_extent.0,
                            height: self.gi_extent.1,
                            depth: 1,
                        })],
                );
            }
        }
        Self::end(self.cmd_build_blas, device)
    }

    /// Submission B: build the TLAS device-side (after the BLAS build has
    /// fully completed — its fence is a device-wide sync).
    pub(crate) fn record_build_tlas(
        &self,
        device: &ash::Device,
        accel: &ash::khr::acceleration_structure::Device,
        scene: &SceneGpu,
        instance_addr: u64,
    ) -> Result<(), RenderError> {
        Self::begin(self.cmd_build_tlas, device)?;
        let geometry = vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(vk::GeometryTypeKHR::INSTANCES)
            .flags(vk::GeometryFlagsKHR::empty())
            .geometry(vk::AccelerationStructureGeometryDataKHR {
                instances: vk::AccelerationStructureGeometryInstancesDataKHR::default()
                    .array_of_pointers(false)
                    .data(vk::DeviceOrHostAddressConstKHR {
                        device_address: instance_addr,
                    }),
            });
        let build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .dst_acceleration_structure(scene.tlas)
            .scratch_data(vk::DeviceOrHostAddressKHR {
                device_address: scene.tlas_scratch_addr,
            })
            .geometries(std::slice::from_ref(&geometry));
        let ranges = [vk::AccelerationStructureBuildRangeInfoKHR {
            primitive_count: scene.tlas_prim_count,
            primitive_offset: 0,
            first_vertex: 0,
            transform_offset: 0,
        }];
        unsafe {
            accel.cmd_build_acceleration_structures(
                self.cmd_build_tlas,
                std::slice::from_ref(&build_info),
                &[&ranges],
            );
        }
        Self::end(self.cmd_build_tlas, device)
    }

    /// Submission C: GI probe compute (optional), the forward render pass,
    /// the tone map and the readback copy.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_frame(
        &self,
        device: &ash::Device,
        pipelines: &Pipelines,
        scene: &SceneGpu,
        options: &RenderOptions,
        gi_dispatch: bool,
    ) -> Result<(), RenderError> {
        Self::begin(self.cmd_frame, device)?;
        unsafe {
            // 0. The `out` image is written as a storage image (by the tone map
            //    or by the ray-query probe), so move it out of UNDEFINED first.
            device.cmd_pipeline_barrier(
                self.cmd_frame,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::default(),
                &[],
                &[],
                &[
                    vk::ImageMemoryBarrier::default()
                        .image(self.out.handle)
                        .old_layout(vk::ImageLayout::UNDEFINED)
                        .new_layout(vk::ImageLayout::GENERAL)
                        .src_access_mask(vk::AccessFlags::empty())
                        .dst_access_mask(vk::AccessFlags::SHADER_WRITE)
                        .subresource_range(color_range()),
                    // The GI dummy (bound when GI is off) is sampled by the
                    // forward pass; it never sees another layout transition.
                    vk::ImageMemoryBarrier::default()
                        .image(self.dummy_tex.0.handle)
                        .old_layout(vk::ImageLayout::UNDEFINED)
                        .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                        .src_access_mask(vk::AccessFlags::empty())
                        .dst_access_mask(vk::AccessFlags::SHADER_READ)
                        .subresource_range(color_range()),
                ],
            );

            // 1. GI probe pass: probe texture TRANSFER_DST -> GENERAL, dispatch.
            if let (Some((gi_img, _)), true) = (&self.gi_tex, gi_dispatch) {
                device.cmd_pipeline_barrier(
                    self.cmd_frame,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::default(),
                    &[],
                    &[],
                    &[vk::ImageMemoryBarrier::default()
                        .image(gi_img.handle)
                        .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                        .new_layout(vk::ImageLayout::GENERAL)
                        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                        .dst_access_mask(vk::AccessFlags::SHADER_WRITE)
                        .subresource_range(color_range())],
                );
                device.cmd_bind_pipeline(
                    self.cmd_frame,
                    vk::PipelineBindPoint::COMPUTE,
                    pipelines.gi_pipeline,
                );
                device.cmd_bind_descriptor_sets(
                    self.cmd_frame,
                    vk::PipelineBindPoint::COMPUTE,
                    pipelines.gi_layout,
                    0,
                    std::slice::from_ref(&self.gi_set),
                    &[],
                );
                device.cmd_dispatch(
                    self.cmd_frame,
                    self.gi_extent.0.div_ceil(8),
                    self.gi_extent.1.div_ceil(8),
                    1,
                );
                // Probe texture GENERAL -> SHADER_READ_ONLY for the forward pass.
                device.cmd_pipeline_barrier(
                    self.cmd_frame,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                    vk::DependencyFlags::default(),
                    &[],
                    &[],
                    &[vk::ImageMemoryBarrier::default()
                        .image(gi_img.handle)
                        .old_layout(vk::ImageLayout::GENERAL)
                        .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                        .dst_access_mask(vk::AccessFlags::SHADER_READ)
                        .subresource_range(color_range())],
                );
            } else if let Some((gi_img, _)) = &self.gi_tex {
                // GI texture exists but the dispatch was skipped (mutation):
                // still move it to SHADER_READ_ONLY so the forward pass reads
                // the zeros deterministically written in submission A.
                device.cmd_pipeline_barrier(
                    self.cmd_frame,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                    vk::DependencyFlags::default(),
                    &[],
                    &[],
                    &[vk::ImageMemoryBarrier::default()
                        .image(gi_img.handle)
                        .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                        .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                        .dst_access_mask(vk::AccessFlags::SHADER_READ)
                        .subresource_range(color_range())],
                );
            }

            // 2a. Raw ray-query probe path: one primary ray per pixel into the
            // TLAS, writing the `out` image directly (no forward pass, no tone
            // map). This is the ray-query correctness path.
            if options.ray_probe {
                device.cmd_bind_pipeline(
                    self.cmd_frame,
                    vk::PipelineBindPoint::COMPUTE,
                    pipelines.rq_pipeline,
                );
                device.cmd_bind_descriptor_sets(
                    self.cmd_frame,
                    vk::PipelineBindPoint::COMPUTE,
                    pipelines.rq_layout,
                    0,
                    std::slice::from_ref(&self.rq_set),
                    &[],
                );
                device.cmd_dispatch(
                    self.cmd_frame,
                    self.width.div_ceil(8),
                    self.height.div_ceil(8),
                    1,
                );
                // out: GENERAL -> TRANSFER_SRC, then read back (step 4 below).
                device.cmd_pipeline_barrier(
                    self.cmd_frame,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::default(),
                    &[],
                    &[],
                    &[vk::ImageMemoryBarrier::default()
                        .image(self.out.handle)
                        .old_layout(vk::ImageLayout::GENERAL)
                        .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                        .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                        .subresource_range(color_range())],
                );
                device.cmd_copy_image_to_buffer(
                    self.cmd_frame,
                    self.out.handle,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    self.readback.handle,
                    &[vk::BufferImageCopy::default()
                        .buffer_row_length(0)
                        .buffer_image_height(0)
                        .image_subresource(vk::ImageSubresourceLayers {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            mip_level: 0,
                            base_array_layer: 0,
                            layer_count: 1,
                        })
                        .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                        .image_extent(vk::Extent3D {
                            width: self.width,
                            height: self.height,
                            depth: 1,
                        })],
                );
                return Self::end(self.cmd_frame, device);
            }

            // 2b. Forward render pass (HDR + depth).
            device.cmd_begin_render_pass(
                self.cmd_frame,
                &vk::RenderPassBeginInfo::default()
                    .render_pass(pipelines.render_pass)
                    .framebuffer(self.framebuffer)
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: vk::Extent2D {
                            width: self.width,
                            height: self.height,
                        },
                    })
                    .clear_values(&[
                        vk::ClearValue {
                            color: vk::ClearColorValue {
                                float32: [0.0, 0.0, 0.0, 1.0],
                            },
                        },
                        vk::ClearValue {
                            depth_stencil: vk::ClearDepthStencilValue {
                                depth: 1.0,
                                stencil: 0,
                            },
                        },
                    ]),
                vk::SubpassContents::INLINE,
            );
            if !options.mutate_skip_forward_draw {
                device.cmd_bind_pipeline(
                    self.cmd_frame,
                    vk::PipelineBindPoint::GRAPHICS,
                    pipelines.graphics,
                );
                device.cmd_set_viewport(
                    self.cmd_frame,
                    0,
                    // Positive height: the projection already flips Y
                    // (`mat4_perspective` negates its y row), so no negative
                    // viewport is needed — and a negative viewport height is
                    // not portable without VK_KHR_maintenance5.
                    &[vk::Viewport {
                        x: 0.0,
                        y: 0.0,
                        width: self.width as f32,
                        height: self.height as f32,
                        min_depth: 0.0,
                        max_depth: 1.0,
                    }],
                );
                device.cmd_set_scissor(
                    self.cmd_frame,
                    0,
                    &[vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: vk::Extent2D {
                            width: self.width,
                            height: self.height,
                        },
                    }],
                );
                device.cmd_bind_descriptor_sets(
                    self.cmd_frame,
                    vk::PipelineBindPoint::GRAPHICS,
                    pipelines.forward_layout,
                    0,
                    std::slice::from_ref(&self.forward_set),
                    &[],
                );
                device.cmd_draw(
                    self.cmd_frame,
                    scene.mesh_vertex_count,
                    scene.instance_count,
                    0,
                    0,
                );
            }
            device.cmd_end_render_pass(self.cmd_frame);

            // 3. Tone map. The render pass already transitioned the HDR
            //    attachment to SHADER_READ_ONLY_OPTIMAL as its final layout,
            //    and the end-of-render-pass dependency makes the colour
            //    writes visible to subsequent shader reads.
            device.cmd_pipeline_barrier(
                self.cmd_frame,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::default(),
                &[vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)],
                &[],
                &[],
            );
            device.cmd_bind_pipeline(
                self.cmd_frame,
                vk::PipelineBindPoint::COMPUTE,
                pipelines.tone_pipeline,
            );
            device.cmd_bind_descriptor_sets(
                self.cmd_frame,
                vk::PipelineBindPoint::COMPUTE,
                pipelines.tone_layout,
                0,
                std::slice::from_ref(&self.tone_set),
                &[],
            );
            device.cmd_dispatch(
                self.cmd_frame,
                self.width.div_ceil(8),
                self.height.div_ceil(8),
                1,
            );

            // 4. Readback: out image GENERAL -> TRANSFER_SRC -> buffer.
            device.cmd_pipeline_barrier(
                self.cmd_frame,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::default(),
                &[],
                &[],
                &[vk::ImageMemoryBarrier::default()
                    .image(self.out.handle)
                    .old_layout(vk::ImageLayout::GENERAL)
                    .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                    .subresource_range(color_range())],
            );
            device.cmd_copy_image_to_buffer(
                self.cmd_frame,
                self.out.handle,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                self.readback.handle,
                &[vk::BufferImageCopy::default()
                    .buffer_row_length(0)
                    .buffer_image_height(0)
                    .image_subresource(vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                    .image_extent(vk::Extent3D {
                        width: self.width,
                        height: self.height,
                        depth: 1,
                    })],
            );
        }
        Self::end(self.cmd_frame, device)
    }

    /// Read the rendered frame's pixels back from the GPU.
    pub(crate) fn read_pixels(&self) -> Result<Vec<u8>, RenderError> {
        let mut pixels = vec![0u8; (self.width * self.height * 4) as usize];
        self.readback.read(&mut pixels)?;
        Ok(pixels)
    }
}

impl Drop for FrameGpu {
    fn drop(&mut self) {
        // The device is still alive here (RenderContext drops frames before
        // the device). The `Image`/`Buffer` fields free their dedicated
        // gpu-allocator allocations via their own Drop (hard constraint:
        // gpu-allocator does not own the dedicated VkImage/VkBuffer).
        let d = self.device.clone();
        unsafe {
            device_destroy_views(&d, self);
            d.destroy_fence(self.fence, None);
            d.destroy_descriptor_pool(self.pool, None);
        }
    }
}

unsafe fn device_destroy_views(device: &ash::Device, frame: &mut FrameGpu) {
    device.destroy_image_view(frame.hdr_view, None);
    device.destroy_image_view(frame.depth_view, None);
    device.destroy_image_view(frame.out_view, None);
    if let Some((_, view)) = &frame.gi_tex {
        device.destroy_image_view(*view, None);
    }
    let (_, dummy_view) = frame.dummy_tex;
    device.destroy_image_view(dummy_view, None);
    device.destroy_framebuffer(frame.framebuffer, None);
}
