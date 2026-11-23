//! GPU pipelines and the frame renderer.
//!
//! One frame is one command buffer and one submission:
//!
//! [TLAS build] -> [GI probe compute] -> forward render pass (HDR + material)
//! -> [glass render pass] -> [post compute: SSR + refraction] -> tone map ->
//! [overlay render pass] -> readback copy *or* blit into a swapchain image
//!
//! The bracketed passes only exist when they are needed: the TLAS build for a
//! frame that traces rays (GI, ray-traced shadows or the raw ray probe), the
//! glass and post passes when the quality preset enables SSR or refraction
//! (`RenderFeatures::needs_post`), the overlay pass when the scene's
//! [`Overlay`](crate::Overlay) has something in it.
//!
//! The GPU objects behind a frame ([`FrameGpu`], [`InstanceBuffer`], [`Tlas`],
//! [`ShapeBlas`], the uploaded [`MeshGpu`] models with their textures and
//! acceleration structures) are created once and reused by `RenderContext`
//! for as long as they fit, so rendering a steady stream of frames builds and
//! uploads nothing.

use ash::vk;

use crate::alloc::{Buffer, Image, SharedAllocator};
use crate::gpu::{RenderError, RenderOptions};
use crate::params::{
    self, ForwardParams, GiParams, GpuInstance, GpuPod, OverlayInstance, PostParams, RqParams,
    ToneParams,
};
use crate::quality::FEATURE_REFRACT;
use crate::scene::{cube_mesh, sphere_mesh, SHAPE_BOX};
use crate::texture::Texture;

/// Format of the material buffer: r = metallic, g = roughness, ba = the
/// shading normal (octahedral, signed), so it has to be a float format.
const MATERIAL_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;

/// Format of the specular buffer: rgb = a surface's base colour (with its
/// metalness, from the material buffer, that gives the specular colour F0
/// the post pass tints reflections with, and the diffuse colour it needs to
/// smooth the sky's share of light), a = the share of the environment its
/// mirror direction can see. Four bytes a pixel.
const SPECULAR_FORMAT: vk::Format = vk::Format::R8G8B8A8_UNORM;

/// Format of the tone-mapped output image (what is read back or blitted to
/// the swapchain, and what the overlay is blended onto).
const OUT_FORMAT: vk::Format = vk::Format::R8G8B8A8_UNORM;

/// Format of a model's base-colour texture: sRGB-encoded, so the sampler hands
/// the shader linear values.
const TEXTURE_FORMAT: vk::Format = vk::Format::R8G8B8A8_SRGB;

/// Anisotropic filtering level for model textures (clamped to what the device
/// offers; off when it offers none).
const TEXTURE_ANISOTROPY: f32 = 8.0;

/// SPIR-V modules compiled by `build.rs` (naga >= 30).
pub(crate) struct Shaders {
    pub rq_triangle: Vec<u32>,
    pub gi_probe: Vec<u32>,
    pub tonemap: Vec<u32>,
    pub forward_vs: Vec<u32>,
    pub forward_fs: Vec<u32>,
    pub forward_glass_vs: Vec<u32>,
    pub forward_glass_fs: Vec<u32>,
    pub forward_mesh_vs: Vec<u32>,
    pub forward_mesh_fs: Vec<u32>,
    /// The three fragment shaders again, with ray-traced shadows.
    pub forward_fs_shadow: Vec<u32>,
    pub forward_glass_fs_shadow: Vec<u32>,
    pub forward_mesh_fs_shadow: Vec<u32>,
    /// The background: the scene's environment where nothing was drawn.
    pub forward_sky_vs: Vec<u32>,
    pub forward_sky_fs: Vec<u32>,
    pub post: Vec<u32>,
    /// The post pass with ray queries (glass refracted through its shape).
    pub post_rq: Vec<u32>,
    pub overlay_vs: Vec<u32>,
    pub overlay_fs: Vec<u32>,
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
            forward_glass_vs: words(include_bytes!(concat!(
                env!("OUT_DIR"),
                "/forward_glass_vs.spv"
            ))),
            forward_glass_fs: words(include_bytes!(concat!(
                env!("OUT_DIR"),
                "/forward_glass_fs.spv"
            ))),
            forward_mesh_vs: words(include_bytes!(concat!(
                env!("OUT_DIR"),
                "/forward_mesh_vs.spv"
            ))),
            forward_mesh_fs: words(include_bytes!(concat!(
                env!("OUT_DIR"),
                "/forward_mesh_fs.spv"
            ))),
            forward_fs_shadow: words(include_bytes!(concat!(
                env!("OUT_DIR"),
                "/forward_fs_shadow.spv"
            ))),
            forward_glass_fs_shadow: words(include_bytes!(concat!(
                env!("OUT_DIR"),
                "/forward_glass_fs_shadow.spv"
            ))),
            forward_mesh_fs_shadow: words(include_bytes!(concat!(
                env!("OUT_DIR"),
                "/forward_mesh_fs_shadow.spv"
            ))),
            forward_sky_vs: words(include_bytes!(concat!(
                env!("OUT_DIR"),
                "/forward_sky_vs.spv"
            ))),
            forward_sky_fs: words(include_bytes!(concat!(
                env!("OUT_DIR"),
                "/forward_sky_fs.spv"
            ))),
            post: words(include_bytes!(concat!(env!("OUT_DIR"), "/post.spv"))),
            post_rq: words(include_bytes!(concat!(env!("OUT_DIR"), "/post_rq.spv"))),
            overlay_vs: words(include_bytes!(concat!(env!("OUT_DIR"), "/overlay_vs.spv"))),
            overlay_fs: words(include_bytes!(concat!(env!("OUT_DIR"), "/overlay_fs.spv"))),
        }
    }
}

/// Long-lived GPU objects: shader modules, pipelines, layouts, the render
/// pass and the sampler.
pub(crate) struct Pipelines {
    modules: Vec<vk::ShaderModule>,
    /// Set 0 = [`Pipelines::forward_set_layout`], set 1 =
    /// [`Pipelines::texture_set_layout`] (used by the mesh pipelines only).
    pub forward_layout: vk::PipelineLayout,
    pub forward_set_layout: vk::DescriptorSetLayout,
    /// A mesh part's base-colour texture: 0 sampled image, 1 sampler.
    pub texture_set_layout: vk::DescriptorSetLayout,
    pub gi_layout: vk::PipelineLayout,
    pub gi_set_layout: vk::DescriptorSetLayout,
    pub tone_layout: vk::PipelineLayout,
    pub tone_set_layout: vk::DescriptorSetLayout,
    pub rq_layout: vk::PipelineLayout,
    pub rq_set_layout: vk::DescriptorSetLayout,
    pub post_layout: vk::PipelineLayout,
    pub post_set_layout: vk::DescriptorSetLayout,
    /// Opaque pass: HDR colour + material/normal (rgba16f) + depth.
    pub render_pass: vk::RenderPass,
    /// Glass pass: glass colour (rgba16f) + the opaque pass's depth (LOAD).
    pub render_pass_glass: vk::RenderPass,
    pub graphics: vk::Pipeline,
    pub graphics_glass: vk::Pipeline,
    /// Opaque pass, mesh models: vertex buffers instead of procedural shapes.
    pub graphics_mesh: vk::Pipeline,
    /// The three forward pipelines with ray-traced shadows (their fragment
    /// shaders query the TLAS at forward binding 5). Null handles on a device
    /// without ray query, where shadows are never enabled.
    pub graphics_shadow: vk::Pipeline,
    pub graphics_glass_shadow: vk::Pipeline,
    pub graphics_mesh_shadow: vk::Pipeline,
    /// Opaque pass, last: the environment as the background, on the pixels
    /// whose depth is still the clear value.
    pub graphics_sky: vk::Pipeline,
    pub gi_pipeline: vk::Pipeline,
    pub tone_pipeline: vk::Pipeline,
    pub rq_pipeline: vk::Pipeline,
    pub post_pipeline: vk::Pipeline,
    /// The post pass with ray queries; null on a device without them.
    pub post_pipeline_rq: vk::Pipeline,
    /// Overlay pass: the tone-mapped output image, LOADed and blended onto.
    pub render_pass_overlay: vk::RenderPass,
    pub overlay_layout: vk::PipelineLayout,
    pub overlay_pipeline: vk::Pipeline,
    pub sampler: vk::Sampler,
    /// Model textures: trilinear, repeating, anisotropic where available.
    pub texture_sampler: vk::Sampler,
}

/// What [`Pipelines::create`] needs to know about the device.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PipelineConfig {
    /// The ray-query chain is enabled: create the shadow pipelines.
    pub ray_query: bool,
    /// The device's anisotropy limit, or 0 when the feature is not enabled.
    pub max_anisotropy: f32,
}

/// What a graphics pipeline draws, which decides what feeds its vertex stage
/// and its fixed-function state (see `Created::graphics_pipeline`).
#[derive(Clone, Copy, PartialEq, Eq)]
enum VertexInput {
    /// Nothing: the procedural shapes are generated in the vertex shader.
    Procedural,
    /// [`MeshVertex`](crate::mesh::MeshVertex) buffers: position, normal, uv.
    Mesh,
    /// The background: no vertex input, and only where nothing was drawn —
    /// depth test "equal" against the clear value, no depth write.
    Sky,
    /// One [`OverlayInstance`] per instance: a rectangle and its colour.
    /// Alpha blending, no depth test.
    Overlay,
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

impl Created<'_> {
    /// A graphics pipeline with dynamic viewport/scissor and
    /// `colour_attachments` identical targets. The forward pipelines
    /// ([`VertexInput::Procedural`], [`VertexInput::Mesh`]) test and write
    /// depth and do not blend; the background ([`VertexInput::Sky`]) passes
    /// only at the depth clear value and writes none; the overlay pipeline
    /// ([`VertexInput::Overlay`]) blends with straight alpha and has no depth.
    fn graphics_pipeline(
        &self,
        vs: (vk::ShaderModule, &'static std::ffi::CStr),
        fs: (vk::ShaderModule, &'static std::ffi::CStr),
        layout: vk::PipelineLayout,
        render_pass: vk::RenderPass,
        colour_attachments: usize,
        vertex: VertexInput,
    ) -> Result<vk::Pipeline, RenderError> {
        let overlay = vertex == VertexInput::Overlay;
        let blend = if overlay {
            // Straight alpha over the display-referred image. The target's
            // own alpha is left as the tone map wrote it.
            vk::PipelineColorBlendAttachmentState::default()
                .blend_enable(true)
                .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
                .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                .color_blend_op(vk::BlendOp::ADD)
                .src_alpha_blend_factor(vk::BlendFactor::ZERO)
                .dst_alpha_blend_factor(vk::BlendFactor::ONE)
                .alpha_blend_op(vk::BlendOp::ADD)
                .color_write_mask(vk::ColorComponentFlags::RGBA)
        } else {
            vk::PipelineColorBlendAttachmentState::default()
                .blend_enable(false)
                .color_write_mask(vk::ColorComponentFlags::RGBA)
        };
        let blend_attachments = vec![blend; colour_attachments];
        let attribute =
            |location: u32, format: vk::Format, offset: u32| vk::VertexInputAttributeDescription {
                location,
                binding: 0,
                format,
                offset,
            };
        let mesh_bindings = [vk::VertexInputBindingDescription {
            binding: 0,
            stride: std::mem::size_of::<crate::mesh::MeshVertex>() as u32,
            input_rate: vk::VertexInputRate::VERTEX,
        }];
        // `MeshVertex`: position, normal, uv.
        let mesh_attributes = [
            attribute(0, vk::Format::R32G32B32_SFLOAT, 0),
            attribute(1, vk::Format::R32G32B32_SFLOAT, 12),
            attribute(2, vk::Format::R32G32_SFLOAT, 24),
        ];
        let overlay_bindings = [vk::VertexInputBindingDescription {
            binding: 0,
            stride: std::mem::size_of::<OverlayInstance>() as u32,
            input_rate: vk::VertexInputRate::INSTANCE,
        }];
        // `OverlayInstance`: rect, colour.
        let overlay_attributes = [
            attribute(0, vk::Format::R32G32B32A32_SFLOAT, 0),
            attribute(1, vk::Format::R32G32B32A32_SFLOAT, 16),
        ];
        let sky = vertex == VertexInput::Sky;
        let vertex_input = match vertex {
            VertexInput::Procedural | VertexInput::Sky => {
                vk::PipelineVertexInputStateCreateInfo::default()
            }
            VertexInput::Mesh => vk::PipelineVertexInputStateCreateInfo::default()
                .vertex_binding_descriptions(&mesh_bindings)
                .vertex_attribute_descriptions(&mesh_attributes),
            VertexInput::Overlay => vk::PipelineVertexInputStateCreateInfo::default()
                .vertex_binding_descriptions(&overlay_bindings)
                .vertex_attribute_descriptions(&overlay_attributes),
        };
        unsafe {
            self.device
                .create_graphics_pipelines(
                    vk::PipelineCache::null(),
                    &[vk::GraphicsPipelineCreateInfo::default()
                        .stages(&[
                            vk::PipelineShaderStageCreateInfo::default()
                                .stage(vk::ShaderStageFlags::VERTEX)
                                .module(vs.0)
                                .name(vs.1),
                            vk::PipelineShaderStageCreateInfo::default()
                                .stage(vk::ShaderStageFlags::FRAGMENT)
                                .module(fs.0)
                                .name(fs.1),
                        ])
                        .vertex_input_state(&vertex_input)
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
                                // Both sides of every triangle are drawn;
                                // `side_of` in shaders/forward.wgsl tells
                                // them apart (not by the winding).
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
                                .depth_test_enable(!overlay)
                                .depth_write_enable(!overlay && !sky)
                                // Reversed depth (1 = near, towards 0 = far;
                                // see `mat4_perspective_reversed`). The
                                // background is drawn at exactly 0, the clear
                                // value: it passes where that is still there.
                                .depth_compare_op(if sky {
                                    vk::CompareOp::EQUAL
                                } else {
                                    vk::CompareOp::GREATER
                                }),
                        )
                        .color_blend_state(
                            &vk::PipelineColorBlendStateCreateInfo::default()
                                .logic_op_enable(false)
                                .logic_op(vk::LogicOp::COPY)
                                .attachments(&blend_attachments),
                        )
                        .dynamic_state(
                            &vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&[
                                vk::DynamicState::VIEWPORT,
                                vk::DynamicState::SCISSOR,
                            ]),
                        )
                        .layout(layout)
                        .render_pass(render_pass)
                        .subpass(0)],
                    None,
                )
                .map(|mut v| v.remove(0))
                .map_err(|(_, e)| RenderError::new(format!("graphics pipeline: {e}")))
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
    /// Create every shader module, layout and pipeline. The GI and ray-probe
    /// compute pipelines are created even when the device lacks ray query
    /// (they are never dispatched there); the shadow variants of the forward
    /// pipelines only with `config.ray_query`.
    pub fn create(
        device: &ash::Device,
        shaders: &Shaders,
        config: PipelineConfig,
    ) -> Result<Self, RenderError> {
        let c = Created { device };
        let mut modules = Vec::new();
        let m_rq = c.module(&shaders.rq_triangle, "rq_triangle")?;
        let m_gi = c.module(&shaders.gi_probe, "gi_probe")?;
        let m_tone = c.module(&shaders.tonemap, "tonemap")?;
        let m_vs = c.module(&shaders.forward_vs, "forward_vs")?;
        let m_fs = c.module(&shaders.forward_fs, "forward_fs")?;
        let m_gvs = c.module(&shaders.forward_glass_vs, "forward_glass_vs")?;
        let m_gfs = c.module(&shaders.forward_glass_fs, "forward_glass_fs")?;
        let m_mvs = c.module(&shaders.forward_mesh_vs, "forward_mesh_vs")?;
        let m_mfs = c.module(&shaders.forward_mesh_fs, "forward_mesh_fs")?;
        let m_svs = c.module(&shaders.forward_sky_vs, "forward_sky_vs")?;
        let m_sfs = c.module(&shaders.forward_sky_fs, "forward_sky_fs")?;
        let m_post = c.module(&shaders.post, "post")?;
        let m_ovs = c.module(&shaders.overlay_vs, "overlay_vs")?;
        let m_ofs = c.module(&shaders.overlay_fs, "overlay_fs")?;
        modules.extend_from_slice(&[
            m_rq, m_gi, m_tone, m_vs, m_fs, m_gvs, m_gfs, m_mvs, m_mfs, m_svs, m_sfs, m_post,
            m_ovs, m_ofs,
        ]);
        // The shadowed fragment shaders use ray queries, so their modules only
        // exist where the feature does.
        let shadow_modules = if config.ray_query {
            let shadowed = [
                c.module(&shaders.forward_fs_shadow, "forward_fs_shadow")?,
                c.module(&shaders.forward_glass_fs_shadow, "forward_glass_fs_shadow")?,
                c.module(&shaders.forward_mesh_fs_shadow, "forward_mesh_fs_shadow")?,
            ];
            modules.extend_from_slice(&shadowed);
            Some(shadowed)
        } else {
            None
        };

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
                        // Mesh part records (`vs_mesh` only).
                        binding(
                            4,
                            vk::DescriptorType::STORAGE_BUFFER,
                            1,
                            vk::ShaderStageFlags::VERTEX,
                        ),
                        // The scene's TLAS, for shadow rays (the shadowed
                        // fragment shaders only; left unwritten otherwise).
                        binding(
                            5,
                            vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
                            1,
                            vk::ShaderStageFlags::FRAGMENT,
                        ),
                    ]),
                    None,
                )
                .map_err(|e| RenderError::new(format!("forward set layout: {e}")))?
        };
        let texture_set_layout = unsafe {
            device
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&[
                        binding(
                            0,
                            vk::DescriptorType::SAMPLED_IMAGE,
                            1,
                            vk::ShaderStageFlags::FRAGMENT,
                        ),
                        binding(
                            1,
                            vk::DescriptorType::SAMPLER,
                            1,
                            vk::ShaderStageFlags::FRAGMENT,
                        ),
                    ]),
                    None,
                )
                .map_err(|e| RenderError::new(format!("texture set layout: {e}")))?
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
                        // Mesh part records, for rays that hit a model.
                        binding(
                            4,
                            vk::DescriptorType::STORAGE_BUFFER,
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

        // post: 0 params, 1 scene, 2 depth, 3 material, 4 glass, 5 out,
        // 6 specular; and for the ray-query variant only (left unwritten
        // otherwise): 7 the scene's TLAS, 8 instances, 9 mesh part records
        let post_set_layout = unsafe {
            let sampled = |b| {
                binding(
                    b,
                    vk::DescriptorType::SAMPLED_IMAGE,
                    1,
                    vk::ShaderStageFlags::COMPUTE,
                )
            };
            device
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&[
                        binding(
                            0,
                            vk::DescriptorType::UNIFORM_BUFFER,
                            1,
                            vk::ShaderStageFlags::COMPUTE,
                        ),
                        sampled(1),
                        sampled(2),
                        sampled(3),
                        sampled(4),
                        binding(
                            5,
                            vk::DescriptorType::STORAGE_IMAGE,
                            1,
                            vk::ShaderStageFlags::COMPUTE,
                        ),
                        sampled(6),
                        binding(
                            7,
                            vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
                            1,
                            vk::ShaderStageFlags::COMPUTE,
                        ),
                        binding(
                            8,
                            vk::DescriptorType::STORAGE_BUFFER,
                            1,
                            vk::ShaderStageFlags::COMPUTE,
                        ),
                        binding(
                            9,
                            vk::DescriptorType::STORAGE_BUFFER,
                            1,
                            vk::ShaderStageFlags::COMPUTE,
                        ),
                    ]),
                    None,
                )
                .map_err(|e| RenderError::new(format!("post set layout: {e}")))?
        };
        let pl = |layout: &vk::PipelineLayoutCreateInfo<'_>, name: &str| unsafe {
            device
                .create_pipeline_layout(layout, None)
                .map_err(|e| RenderError::new(format!("{name}: {e}")))
        };
        let forward_layout = pl(
            &vk::PipelineLayoutCreateInfo::default()
                .set_layouts(&[forward_set_layout, texture_set_layout]),
            "forward pipeline layout",
        )?;
        // The overlay's rectangles carry everything: no descriptors at all.
        let overlay_layout = pl(
            &vk::PipelineLayoutCreateInfo::default(),
            "overlay pipeline layout",
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
        let post_layout = pl(
            &vk::PipelineLayoutCreateInfo::default()
                .set_layouts(std::slice::from_ref(&post_set_layout)),
            "post pipeline layout",
        )?;

        // ---- render passes ------------------------------------------------------
        // Opaque pass: attachment 0 = HDR colour, 1 = material + shading normal
        // (rgba16f), 2 = specular colour (rgba8), 3 = depth.
        // Depth is STOREd and ends in a read-only layout so the post pass can
        // sample it.
        let color_ref = |attachment: u32| {
            vk::AttachmentReference::default()
                .attachment(attachment)
                .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        };
        let color_attachment = |format: vk::Format| {
            vk::AttachmentDescription::default()
                .format(format)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .final_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        };
        let fragment_stages = vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
            | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
            | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS;
        let attachment_writes = vk::AccessFlags::COLOR_ATTACHMENT_WRITE
            | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE;
        let attachment_rw = attachment_writes
            | vk::AccessFlags::COLOR_ATTACHMENT_READ
            | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ;
        let render_pass = unsafe {
            let color_refs = [color_ref(0), color_ref(1), color_ref(2)];
            let depth_ref = vk::AttachmentReference::default()
                .attachment(3)
                .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
            device
                .create_render_pass(
                    &vk::RenderPassCreateInfo::default()
                        .attachments(&[
                            color_attachment(vk::Format::R16G16B16A16_SFLOAT),
                            color_attachment(MATERIAL_FORMAT),
                            color_attachment(SPECULAR_FORMAT),
                            vk::AttachmentDescription::default()
                                .format(vk::Format::D32_SFLOAT)
                                .samples(vk::SampleCountFlags::TYPE_1)
                                .load_op(vk::AttachmentLoadOp::CLEAR)
                                .store_op(vk::AttachmentStoreOp::STORE)
                                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                                .initial_layout(vk::ImageLayout::UNDEFINED)
                                // `separateDepthStencilLayouts` is not enabled,
                                // so the combined layouts are the only legal ones.
                                .final_layout(vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL),
                        ])
                        .subpasses(&[vk::SubpassDescription::default()
                            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                            .color_attachments(&color_refs)
                            .depth_stencil_attachment(&depth_ref)])
                        .dependencies(&[
                            vk::SubpassDependency::default()
                                .src_subpass(vk::SUBPASS_EXTERNAL)
                                .dst_subpass(0)
                                .src_stage_mask(fragment_stages)
                                .src_access_mask(vk::AccessFlags::empty())
                                .dst_stage_mask(fragment_stages)
                                .dst_access_mask(attachment_rw),
                            // Attachment writes -> the post / tone compute reads.
                            vk::SubpassDependency::default()
                                .src_subpass(0)
                                .dst_subpass(vk::SUBPASS_EXTERNAL)
                                .src_stage_mask(fragment_stages)
                                .src_access_mask(attachment_writes)
                                .dst_stage_mask(vk::PipelineStageFlags::COMPUTE_SHADER)
                                .dst_access_mask(vk::AccessFlags::SHADER_READ),
                        ]),
                    None,
                )
                .map_err(|e| RenderError::new(format!("render pass: {e}")))?
        };
        // Glass pass: attachment 0 = glass colour (rgb lit, a = transmission),
        // 1 = the opaque pass's material buffer and 2 = its depth, both LOADed:
        // glass is occluded by opaque geometry, and where it is nearest it
        // writes its own depth, material and shading normal over the opaque
        // ones (the post pass refracts through the glass surface's normal).
        let render_pass_glass = unsafe {
            let color_refs = [color_ref(0), color_ref(1)];
            let depth_ref = vk::AttachmentReference::default()
                .attachment(2)
                .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
            device
                .create_render_pass(
                    &vk::RenderPassCreateInfo::default()
                        .attachments(&[
                            color_attachment(vk::Format::R16G16B16A16_SFLOAT),
                            vk::AttachmentDescription::default()
                                .format(MATERIAL_FORMAT)
                                .samples(vk::SampleCountFlags::TYPE_1)
                                .load_op(vk::AttachmentLoadOp::LOAD)
                                .store_op(vk::AttachmentStoreOp::STORE)
                                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                                .initial_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                                .final_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
                            vk::AttachmentDescription::default()
                                .format(vk::Format::D32_SFLOAT)
                                .samples(vk::SampleCountFlags::TYPE_1)
                                .load_op(vk::AttachmentLoadOp::LOAD)
                                .store_op(vk::AttachmentStoreOp::STORE)
                                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                                .initial_layout(vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL)
                                .final_layout(vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL),
                        ])
                        .subpasses(&[vk::SubpassDescription::default()
                            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                            .color_attachments(&color_refs)
                            .depth_stencil_attachment(&depth_ref)])
                        .dependencies(&[
                            vk::SubpassDependency::default()
                                .src_subpass(vk::SUBPASS_EXTERNAL)
                                .dst_subpass(0)
                                .src_stage_mask(fragment_stages)
                                .src_access_mask(attachment_writes)
                                .dst_stage_mask(fragment_stages)
                                .dst_access_mask(attachment_rw),
                            vk::SubpassDependency::default()
                                .src_subpass(0)
                                .dst_subpass(vk::SUBPASS_EXTERNAL)
                                .src_stage_mask(fragment_stages)
                                .src_access_mask(attachment_writes)
                                .dst_stage_mask(vk::PipelineStageFlags::COMPUTE_SHADER)
                                .dst_access_mask(vk::AccessFlags::SHADER_READ),
                        ]),
                    None,
                )
                .map_err(|e| RenderError::new(format!("glass render pass: {e}")))?
        };

        // Overlay pass: the finished `out` image as the only attachment,
        // LOADed, blended onto, and left ready for the readback copy or the
        // swapchain blit.
        let render_pass_overlay = unsafe {
            let color_refs = [color_ref(0)];
            device
                .create_render_pass(
                    &vk::RenderPassCreateInfo::default()
                        .attachments(&[vk::AttachmentDescription::default()
                            .format(OUT_FORMAT)
                            .samples(vk::SampleCountFlags::TYPE_1)
                            .load_op(vk::AttachmentLoadOp::LOAD)
                            .store_op(vk::AttachmentStoreOp::STORE)
                            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                            // As the tone map (a compute pass) left it.
                            .initial_layout(vk::ImageLayout::GENERAL)
                            .final_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)])
                        .subpasses(&[vk::SubpassDescription::default()
                            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                            .color_attachments(&color_refs)])
                        .dependencies(&[
                            // The compute pass's storage writes -> blending.
                            vk::SubpassDependency::default()
                                .src_subpass(vk::SUBPASS_EXTERNAL)
                                .dst_subpass(0)
                                .src_stage_mask(vk::PipelineStageFlags::COMPUTE_SHADER)
                                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                                .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
                                .dst_access_mask(
                                    vk::AccessFlags::COLOR_ATTACHMENT_READ
                                        | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                                ),
                            // Blended pixels -> the copy / blit out of it.
                            vk::SubpassDependency::default()
                                .src_subpass(0)
                                .dst_subpass(vk::SUBPASS_EXTERNAL)
                                .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
                                .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                                .dst_stage_mask(vk::PipelineStageFlags::TRANSFER)
                                .dst_access_mask(vk::AccessFlags::TRANSFER_READ),
                        ]),
                    None,
                )
                .map_err(|e| RenderError::new(format!("overlay render pass: {e}")))?
        };

        // ---- graphics pipelines --------------------------------------------------
        let forward = |vs: (vk::ShaderModule, &'static std::ffi::CStr),
                       fs: (vk::ShaderModule, &'static std::ffi::CStr),
                       pass: vk::RenderPass,
                       vertex: VertexInput| {
            // (The glass pass has no specular buffer: glass is not mirrored
            // by the post pass.)
            let targets = if pass == render_pass { 3 } else { 2 };
            c.graphics_pipeline(vs, fs, forward_layout, pass, targets, vertex)
        };
        let (procedural, mesh) = (VertexInput::Procedural, VertexInput::Mesh);
        let graphics = forward(
            (m_vs, c"vs_main"),
            (m_fs, c"fs_main"),
            render_pass,
            procedural,
        )?;
        let graphics_mesh = forward((m_mvs, c"vs_mesh"), (m_mfs, c"fs_mesh"), render_pass, mesh)?;
        let graphics_sky = forward(
            (m_svs, c"vs_sky"),
            (m_sfs, c"fs_sky"),
            render_pass,
            VertexInput::Sky,
        )?;
        let graphics_glass = forward(
            (m_gvs, c"vs_glass"),
            (m_gfs, c"fs_glass"),
            render_pass_glass,
            procedural,
        )?;
        // The same three with shadow rays in the fragment shader.
        let [graphics_shadow, graphics_glass_shadow, graphics_mesh_shadow] = match shadow_modules {
            Some([fs, glass_fs, mesh_fs]) => [
                forward(
                    (m_vs, c"vs_main"),
                    (fs, c"fs_main"),
                    render_pass,
                    procedural,
                )?,
                forward(
                    (m_gvs, c"vs_glass"),
                    (glass_fs, c"fs_glass"),
                    render_pass_glass,
                    procedural,
                )?,
                forward(
                    (m_mvs, c"vs_mesh"),
                    (mesh_fs, c"fs_mesh"),
                    render_pass,
                    mesh,
                )?,
            ],
            None => [vk::Pipeline::null(); 3],
        };
        let overlay_pipeline = c.graphics_pipeline(
            (m_ovs, c"vs_overlay"),
            (m_ofs, c"fs_overlay"),
            overlay_layout,
            render_pass_overlay,
            1,
            VertexInput::Overlay,
        )?;

        let gi_pipeline = c.compute_pipeline(m_gi, gi_layout, "gi pipeline")?;
        let tone_pipeline = c.compute_pipeline(m_tone, tone_layout, "tone pipeline")?;
        let rq_pipeline = c.compute_pipeline(m_rq, rq_layout, "rq pipeline")?;
        let post_pipeline = c.compute_pipeline(m_post, post_layout, "post pipeline")?;
        // Its ray-query variant only exists where ray queries do.
        let post_pipeline_rq = if config.ray_query {
            let module = c.module(&shaders.post_rq, "post_rq")?;
            modules.push(module);
            c.compute_pipeline(module, post_layout, "post pipeline (ray queries)")?
        } else {
            vk::Pipeline::null()
        };

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
        // Model textures: trilinear over the whole mip chain, repeating, and
        // anisotropic where the device has the feature (it keeps a floor
        // texture sharp at grazing angles).
        let anisotropy = TEXTURE_ANISOTROPY.min(config.max_anisotropy);
        let texture_sampler = unsafe {
            device
                .create_sampler(
                    &vk::SamplerCreateInfo::default()
                        .mag_filter(vk::Filter::LINEAR)
                        .min_filter(vk::Filter::LINEAR)
                        .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                        .address_mode_u(vk::SamplerAddressMode::REPEAT)
                        .address_mode_v(vk::SamplerAddressMode::REPEAT)
                        .address_mode_w(vk::SamplerAddressMode::REPEAT)
                        .anisotropy_enable(anisotropy > 1.0)
                        .max_anisotropy(anisotropy.max(1.0))
                        .min_lod(0.0)
                        .max_lod(vk::LOD_CLAMP_NONE),
                    None,
                )
                .map_err(|e| RenderError::new(format!("texture sampler: {e}")))?
        };

        Ok(Self {
            modules,
            forward_layout,
            forward_set_layout,
            texture_set_layout,
            gi_layout,
            gi_set_layout,
            tone_layout,
            tone_set_layout,
            rq_layout,
            rq_set_layout,
            post_layout,
            post_set_layout,
            render_pass,
            render_pass_glass,
            graphics,
            graphics_glass,
            graphics_mesh,
            graphics_shadow,
            graphics_glass_shadow,
            graphics_mesh_shadow,
            graphics_sky,
            gi_pipeline,
            tone_pipeline,
            rq_pipeline,
            post_pipeline,
            post_pipeline_rq,
            render_pass_overlay,
            overlay_layout,
            overlay_pipeline,
            sampler,
            texture_sampler,
        })
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
        device.destroy_pipeline_layout(self.post_layout, None);
        device.destroy_pipeline_layout(self.overlay_layout, None);
        device.destroy_descriptor_set_layout(self.post_set_layout, None);
        device.destroy_descriptor_set_layout(self.forward_set_layout, None);
        device.destroy_descriptor_set_layout(self.texture_set_layout, None);
        device.destroy_descriptor_set_layout(self.gi_set_layout, None);
        device.destroy_descriptor_set_layout(self.tone_set_layout, None);
        device.destroy_descriptor_set_layout(self.rq_set_layout, None);
        device.destroy_render_pass(self.render_pass, None);
        device.destroy_render_pass(self.render_pass_glass, None);
        device.destroy_render_pass(self.render_pass_overlay, None);
        device.destroy_pipeline(self.graphics, None);
        device.destroy_pipeline(self.graphics_glass, None);
        device.destroy_pipeline(self.graphics_mesh, None);
        // (Null on a device without ray query: destroying null is a no-op.)
        device.destroy_pipeline(self.graphics_shadow, None);
        device.destroy_pipeline(self.graphics_glass_shadow, None);
        device.destroy_pipeline(self.graphics_mesh_shadow, None);
        device.destroy_pipeline(self.graphics_sky, None);
        device.destroy_pipeline(self.overlay_pipeline, None);
        device.destroy_pipeline(self.post_pipeline, None);
        if self.post_pipeline_rq != vk::Pipeline::null() {
            device.destroy_pipeline(self.post_pipeline_rq, None);
        }
        device.destroy_pipeline(self.gi_pipeline, None);
        device.destroy_pipeline(self.tone_pipeline, None);
        device.destroy_pipeline(self.rq_pipeline, None);
        device.destroy_sampler(self.sampler, None);
        device.destroy_sampler(self.texture_sampler, None);
    }
}

// ---- command buffer helpers ----------------------------------------------------

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

fn allocate_command_buffer(
    device: &ash::Device,
    cmd_pool: vk::CommandPool,
) -> Result<vk::CommandBuffer, RenderError> {
    unsafe {
        device
            .allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(cmd_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
            .map_err(|e| RenderError::new(format!("allocate_command_buffers: {e}")))?
            .first()
            .copied()
            .ok_or_else(|| RenderError::new("allocate_command_buffers returned no buffer"))
    }
}

// ---- acceleration structures ---------------------------------------------------
//
// Every render instance is a scaled, translated copy of one of two unit shapes
// (see `shaders/forward.wgsl`), so the ray-traced scene is two bottom-level
// structures — the unit icosphere and the unit cube, built once per context —
// and one top-level structure with an instance per render instance carrying
// that instance's scale and translation.
//
// A mesh model adds a bottom-level structure of its own — built once, the
// first time a frame that traces rays draws the model, with one geometry per
// material part ([`MeshGpu::ensure_blas`]) — and one top-level instance per
// placement, after the render instances ([`Tlas::write_instances`]).
//
// The BLAS is created as BOTTOM_LEVEL and the TLAS as TOP_LEVEL: only a
// **descriptor-bound** acceleration structure must be GENERIC or TOP_LEVEL
// (VUID-...-03579) and only the TLAS is ever bound, so each structure has its
// honest type and an accidental descriptor write of a BLAS is a validation
// error rather than silent UB.

/// Storage and scratch sizes for a build of `info`, whose geometries hold at
/// most `prim_counts` primitives each.
fn build_sizes(
    accel: &ash::khr::acceleration_structure::Device,
    info: &vk::AccelerationStructureBuildGeometryInfoKHR<'_>,
    prim_counts: &[u32],
) -> vk::AccelerationStructureBuildSizesInfoKHR<'static> {
    let mut sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
    unsafe {
        accel.get_acceleration_structure_build_sizes(
            vk::AccelerationStructureBuildTypeKHR::DEVICE,
            info,
            prim_counts,
            &mut sizes,
        )
    };
    sizes
}

fn as_input_usage() -> vk::BufferUsageFlags {
    vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
}

fn as_storage_usage() -> vk::BufferUsageFlags {
    vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
}

fn as_scratch_usage() -> vk::BufferUsageFlags {
    vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
}

/// Triangle-list geometry (a `vec3<f32>` position at the start of every
/// `vertex_stride` bytes, `u32` indices) for a BLAS size query or build.
fn triangles_geometry(
    vertex_addr: u64,
    vertex_stride: u64,
    index_addr: u64,
    max_vertex: u32,
) -> vk::AccelerationStructureGeometryKHR<'static> {
    vk::AccelerationStructureGeometryKHR::default()
        .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
        .flags(vk::GeometryFlagsKHR::empty())
        .geometry(vk::AccelerationStructureGeometryDataKHR {
            triangles: vk::AccelerationStructureGeometryTrianglesDataKHR::default()
                .vertex_format(vk::Format::R32G32B32_SFLOAT)
                .vertex_data(vk::DeviceOrHostAddressConstKHR {
                    device_address: vertex_addr,
                })
                .vertex_stride(vertex_stride)
                .max_vertex(max_vertex)
                .index_type(vk::IndexType::UINT32)
                .index_data(vk::DeviceOrHostAddressConstKHR {
                    device_address: index_addr,
                }),
        })
}

/// Instance-array geometry for a TLAS size query or build.
fn instances_geometry(instance_addr: u64) -> vk::AccelerationStructureGeometryKHR<'static> {
    vk::AccelerationStructureGeometryKHR::default()
        .geometry_type(vk::GeometryTypeKHR::INSTANCES)
        .flags(vk::GeometryFlagsKHR::empty())
        .geometry(vk::AccelerationStructureGeometryDataKHR {
            instances: vk::AccelerationStructureGeometryInstancesDataKHR::default()
                .array_of_pointers(false)
                .data(vk::DeviceOrHostAddressConstKHR {
                    device_address: instance_addr,
                }),
        })
}

/// Bytes of one vertex of the placeholder shapes' BLAS input: a bare position.
const SHAPE_VERTEX_STRIDE: u64 = 12;

/// Record commands with `record`, submit them and wait until they have run
/// (one fenced submission; nothing else may be pending on `queue`'s pool).
fn submit_once(
    device: &ash::Device,
    queue: vk::Queue,
    cmd_pool: vk::CommandPool,
    what: &'static str,
    record: impl FnOnce(vk::CommandBuffer),
) -> Result<(), RenderError> {
    let cmd = allocate_command_buffer(device, cmd_pool)?;
    let run = (|| {
        begin(cmd, device)?;
        record(cmd);
        end(cmd, device)?;
        unsafe {
            let fence = device
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .map_err(|e| RenderError::new(format!("create_fence: {e}")))?;
            let submit = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd));
            let run = device
                .queue_submit(queue, std::slice::from_ref(&submit), fence)
                .map_err(|e| RenderError::with_name(what, format!("submit: {e}")))
                .and_then(|()| {
                    device
                        .wait_for_fences(std::slice::from_ref(&fence), true, u64::MAX)
                        .map_err(|e| RenderError::with_name(what, format!("wait: {e}")))
                });
            if run.is_err() {
                // The fence may still be pending: it cannot be destroyed yet.
                let _ = device.device_wait_idle();
            }
            device.destroy_fence(fence, None);
            run
        }
    })();
    unsafe {
        if run.is_err() {
            // Whatever was submitted must be off the queue before the command
            // buffer (and the caller's staging buffers) go away.
            let _ = device.device_wait_idle();
        }
        device.free_command_buffers(cmd_pool, &[cmd]);
    }
    run
}

/// A bottom-level acceleration structure: one of the unit placeholder shapes,
/// or a mesh model.
struct Blas {
    handle: vk::AccelerationStructureKHR,
    /// The structure's device address: what a TLAS instance references.
    address: u64,
    _storage: Buffer,
    accel: ash::khr::acceleration_structure::Device,
}

impl Drop for Blas {
    fn drop(&mut self) {
        // The structure goes first; its storage buffer is freed by the field drop.
        unsafe { self.accel.destroy_acceleration_structure(self.handle, None) };
    }
}

/// A [`Blas`] plus the build inputs, which only have to live until the
/// device-side build has completed.
struct BlasBuild {
    blas: Blas,
    vertices: Buffer,
    indices: Buffer,
    scratch: Buffer,
    triangles: u32,
    max_vertex: u32,
}

impl BlasBuild {
    fn new(
        device: &ash::Device,
        accel: &ash::khr::acceleration_structure::Device,
        allocator: &SharedAllocator,
        mesh: &[[f32; 3]],
        name: &'static str,
    ) -> Result<Self, RenderError> {
        let vertex_bytes: Vec<u8> = mesh
            .iter()
            .flatten()
            .flat_map(|c| c.to_le_bytes())
            .collect();
        let index_bytes: Vec<u8> = (0..mesh.len() as u32).flat_map(u32::to_le_bytes).collect();
        let vertices = Buffer::new(
            device,
            allocator,
            vertex_bytes.len().max(16) as u64,
            as_input_usage(),
            gpu_allocator::MemoryLocation::CpuToGpu,
            name,
        )?;
        vertices.write(&vertex_bytes)?;
        let indices = Buffer::new(
            device,
            allocator,
            index_bytes.len().max(16) as u64,
            as_input_usage(),
            gpu_allocator::MemoryLocation::CpuToGpu,
            name,
        )?;
        indices.write(&index_bytes)?;

        let triangles = (mesh.len() / 3) as u32;
        let max_vertex = mesh.len().saturating_sub(1) as u32;
        let geometry = triangles_geometry(
            vertices.device_address(),
            SHAPE_VERTEX_STRIDE,
            indices.device_address(),
            max_vertex,
        );
        let info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .geometries(std::slice::from_ref(&geometry));
        let sizes = build_sizes(accel, &info, &[triangles]);
        let storage = Buffer::new(
            device,
            allocator,
            sizes.acceleration_structure_size.max(16),
            as_storage_usage(),
            gpu_allocator::MemoryLocation::GpuOnly,
            name,
        )?;
        let scratch = Buffer::new(
            device,
            allocator,
            sizes.build_scratch_size.max(256),
            as_scratch_usage(),
            gpu_allocator::MemoryLocation::GpuOnly,
            name,
        )?;
        let handle = unsafe {
            accel
                .create_acceleration_structure(
                    &vk::AccelerationStructureCreateInfoKHR::default()
                        .buffer(storage.handle)
                        .size(sizes.acceleration_structure_size)
                        .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL),
                    None,
                )
                .map_err(|e| RenderError::with_name(name, format!("create BLAS: {e}")))?
        };
        let address = unsafe {
            accel.get_acceleration_structure_device_address(
                &vk::AccelerationStructureDeviceAddressInfoKHR::default()
                    .acceleration_structure(handle),
            )
        };
        Ok(Self {
            blas: Blas {
                handle,
                address,
                _storage: storage,
                accel: accel.clone(),
            },
            vertices,
            indices,
            scratch,
            triangles,
            max_vertex,
        })
    }
}

/// Detail level of the ray-traced sphere (320 triangles): round enough that a
/// ray-traced outline matches the rasterised one at ordinary sizes, small
/// enough to stay cheap to trace against thousands of instances.
pub(crate) const BLAS_SPHERE_LEVEL: u32 = 1;

/// The two unit placeholder shapes as bottom-level acceleration structures.
/// They never change, so a context builds them once, the first time a frame
/// needs ray queries.
pub(crate) struct ShapeBlas {
    sphere: Blas,
    cube: Blas,
}

impl ShapeBlas {
    /// Create both structures and build them device-side (one fenced
    /// submission; the caller must not have work pending on `queue`'s pool).
    pub fn build(
        device: &ash::Device,
        accel: &ash::khr::acceleration_structure::Device,
        allocator: &SharedAllocator,
        queue: vk::Queue,
        cmd_pool: vk::CommandPool,
    ) -> Result<Self, RenderError> {
        let sphere = BlasBuild::new(
            device,
            accel,
            allocator,
            &sphere_mesh(BLAS_SPHERE_LEVEL),
            "blas-sphere",
        )?;
        let cube = BlasBuild::new(device, accel, allocator, &cube_mesh(), "blas-cube")?;
        let cmd = allocate_command_buffer(device, cmd_pool)?;
        let built = Self::record_and_run(device, accel, queue, cmd, [&sphere, &cube]);
        unsafe {
            if built.is_err() {
                // Whatever was submitted must be off the queue before the
                // command buffer and the build inputs go away.
                let _ = device.device_wait_idle();
            }
            device.free_command_buffers(cmd_pool, &[cmd]);
        }
        built?;
        Ok(Self {
            sphere: sphere.blas,
            cube: cube.blas,
        })
    }

    fn record_and_run(
        device: &ash::Device,
        accel: &ash::khr::acceleration_structure::Device,
        queue: vk::Queue,
        cmd: vk::CommandBuffer,
        builds: [&BlasBuild; 2],
    ) -> Result<(), RenderError> {
        let geometries = builds.map(|b| {
            triangles_geometry(
                b.vertices.device_address(),
                SHAPE_VERTEX_STRIDE,
                b.indices.device_address(),
                b.max_vertex,
            )
        });
        let infos: Vec<vk::AccelerationStructureBuildGeometryInfoKHR<'_>> = builds
            .iter()
            .zip(&geometries)
            .map(|(b, geometry)| {
                vk::AccelerationStructureBuildGeometryInfoKHR::default()
                    .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
                    .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
                    .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
                    .dst_acceleration_structure(b.blas.handle)
                    .scratch_data(vk::DeviceOrHostAddressKHR {
                        device_address: b.scratch.device_address(),
                    })
                    .geometries(std::slice::from_ref(geometry))
            })
            .collect();
        let ranges = builds.map(|b| {
            [vk::AccelerationStructureBuildRangeInfoKHR {
                primitive_count: b.triangles,
                primitive_offset: 0,
                first_vertex: 0,
                transform_offset: 0,
            }]
        });
        let range_refs: Vec<&[vk::AccelerationStructureBuildRangeInfoKHR]> =
            ranges.iter().map(|r| r.as_slice()).collect();

        begin(cmd, device)?;
        unsafe { accel.cmd_build_acceleration_structures(cmd, &infos, &range_refs) };
        end(cmd, device)?;
        unsafe {
            let fence = device
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .map_err(|e| RenderError::new(format!("create_fence: {e}")))?;
            let submit = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd));
            let run = device
                .queue_submit(queue, std::slice::from_ref(&submit), fence)
                .map_err(|e| RenderError::new(format!("BLAS build submit: {e}")))
                .and_then(|()| {
                    device
                        .wait_for_fences(std::slice::from_ref(&fence), true, u64::MAX)
                        .map_err(|e| RenderError::new(format!("BLAS build wait: {e}")))
                });
            if run.is_err() {
                let _ = device.device_wait_idle();
            }
            device.destroy_fence(fence, None);
            run
        }
    }

    /// Device address of the structure an instance of `shape` references
    /// (same rule as the vertex shader: `SHAPE_BOX` is the cube, anything else
    /// the sphere).
    fn address_for(&self, shape: u32) -> u64 {
        if shape == SHAPE_BOX {
            self.cube.address
        } else {
            self.sphere.address
        }
    }

    /// How shadow rays tell the outside of an instance of `shape` from its
    /// inside: both shapes are closed solids, but the cube's triangles run
    /// clockwise seen from outside and the sphere's counter-clockwise.
    fn flags_for(shape: u32) -> vk::GeometryInstanceFlagsKHR {
        if shape == SHAPE_BOX {
            INSIDE_OUT_FLAGS
        } else {
            SOLID_FLAGS
        }
    }
}

/// Instance mask bit of a shadow caster: the only bit shadow rays look for
/// (`SHADOW_MASK` in shaders/forward.wgsl). Rays with a full mask — GI, the
/// ray probe — see every instance whatever this bit says.
const MASK_SHADOW_CASTER: u8 = 0x01;
/// Instance mask bit of glass (a transmissive instance), and of everything
/// that is not: a ray through glass looks for the far side of the glass with
/// the first and for what is behind it with the second (`MASK_GLASS`,
/// `MASK_OPAQUE` in shaders/post.wgsl).
const MASK_GLASS: u8 = 0x02;
const MASK_OPAQUE: u8 = 0x04;
const MASK_EVERYTHING: u8 = 0xff;

/// Instance flags of a closed solid wound counter-clockwise seen from outside
/// (the unit sphere, and any well-formed closed mesh): with these, the faces a
/// ray *enters* such a solid through are front faces and the ones it leaves
/// through are back faces — which is what shadow rays cull
/// (`RAY_FLAG_CULL_BACK_FACING`, see `light_visible` in shaders/forward.wgsl).
///
/// No flag at all: without `TRIANGLE_FLIP_FACING` a triangle is front-facing
/// for a ray whose origin sees its vertices run counter-clockwise (its
/// geometric normal `(v1 - v0) x (v2 - v0)` points back at the origin), the
/// rasteriser convention used everywhere else here. A solid wound the other
/// way round gets the flip ([`INSIDE_OUT_FLAGS`]). A GPU test pins both:
/// `shadow_rays_ignore_the_faces_they_leave_a_solid_through`.
const SOLID_FLAGS: vk::GeometryInstanceFlagsKHR = vk::GeometryInstanceFlagsKHR::empty();

/// Instance flags of a closed solid wound *clockwise* seen from outside: the
/// unit cube (`cube_mesh` — a unit test in `scene.rs` keeps an eye on both
/// shapes' winding) and any closed mesh with a negative volume.
const INSIDE_OUT_FLAGS: vk::GeometryInstanceFlagsKHR =
    vk::GeometryInstanceFlagsKHR::TRIANGLE_FLIP_FACING;

/// How shadow rays must treat the faces of `mesh`: single-sided like the
/// placeholder shapes when it is closed (inverted when it is wound inside
/// out), two-sided when it is open — an open surface has no inside for a ray
/// to leave, and a floor quad must block light from either side.
fn instance_flags_for(mesh: &crate::mesh::MeshData) -> vk::GeometryInstanceFlagsKHR {
    if !mesh.is_closed() {
        vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE
    } else if mesh.signed_volume() < 0.0 {
        INSIDE_OUT_FLAGS
    } else {
        SOLID_FLAGS
    }
}

/// One placement of a mesh model in the top-level structure.
#[derive(Clone, Copy, Debug)]
pub(crate) struct MeshTlasInstance {
    /// Row-major 3x4 model -> camera-relative world matrix.
    pub transform: [f32; 12],
    /// Device address of the model's BLAS ([`MeshGpu::ensure_blas`]).
    pub blas: u64,
    /// `1 +` the index of the model's first part record in the mesh record
    /// buffer (24 bits). A ray hit adds its geometry index — the BLAS has one
    /// geometry per drawn part — to find the part it hit; 0 is reserved for
    /// the render instances.
    pub custom_index: u32,
    pub flags: vk::GeometryInstanceFlagsKHR,
}

/// Largest [`MeshTlasInstance::custom_index`]: it is a 24-bit field.
pub(crate) const MAX_CUSTOM_INDEX: u32 = (1 << 24) - 1;

/// The top-level acceleration structure over a frame's render instances and
/// model placements. The storage is sized for `capacity` instances and reused:
/// each frame that needs ray queries rewrites the instance records and
/// rebuilds it in place.
pub(crate) struct Tlas {
    pub handle: vk::AccelerationStructureKHR,
    capacity: u32,
    _storage: Buffer,
    scratch: Buffer,
    instances: Buffer,
    accel: ash::khr::acceleration_structure::Device,
}

impl Tlas {
    /// Per-frame rebuilds favour build time over trace time.
    const FLAGS: vk::BuildAccelerationStructureFlagsKHR =
        vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_BUILD;

    pub fn new(
        device: &ash::Device,
        accel: &ash::khr::acceleration_structure::Device,
        allocator: &SharedAllocator,
        capacity: u32,
    ) -> Result<Self, RenderError> {
        let capacity = capacity.max(1);
        let instances = Buffer::new(
            device,
            allocator,
            u64::from(capacity)
                * std::mem::size_of::<vk::AccelerationStructureInstanceKHR>() as u64,
            as_input_usage(),
            gpu_allocator::MemoryLocation::CpuToGpu,
            "tlas-instances",
        )?;
        let geometry = instances_geometry(instances.device_address());
        let info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .flags(Self::FLAGS)
            .geometries(std::slice::from_ref(&geometry));
        let sizes = build_sizes(accel, &info, &[capacity]);
        let storage = Buffer::new(
            device,
            allocator,
            sizes.acceleration_structure_size.max(16),
            as_storage_usage(),
            gpu_allocator::MemoryLocation::GpuOnly,
            "tlas",
        )?;
        let scratch = Buffer::new(
            device,
            allocator,
            sizes.build_scratch_size.max(256),
            as_scratch_usage(),
            gpu_allocator::MemoryLocation::GpuOnly,
            "tlas-scratch",
        )?;
        let handle = unsafe {
            accel
                .create_acceleration_structure(
                    &vk::AccelerationStructureCreateInfoKHR::default()
                        .buffer(storage.handle)
                        .size(sizes.acceleration_structure_size)
                        .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL),
                    None,
                )
                .map_err(|e| RenderError::new(format!("create TLAS: {e}")))?
        };
        Ok(Self {
            handle,
            capacity,
            _storage: storage,
            scratch,
            instances,
            accel: accel.clone(),
        })
    }

    /// How many instances the structure has room for.
    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    /// Write the TLAS instance records and return how many there are.
    ///
    /// First one per render instance: a uniform scale by the instance radius
    /// plus a translation to its camera-relative centre, referencing the BLAS
    /// of its shape. The record's index is the render instance index, which is
    /// what a ray-query hit reports, and its custom index is 0. An instance
    /// for which `casts_shadow` is false is hidden from shadow rays (and from
    /// nothing else).
    ///
    /// Then one per model placement in `meshes`, with the model's own BLAS,
    /// transform and (non-zero) custom index.
    pub fn write_instances(
        &self,
        shapes: &ShapeBlas,
        instances: &[GpuInstance],
        casts_shadow: impl Fn(&GpuInstance) -> bool,
        meshes: &[MeshTlasInstance],
    ) -> Result<u32, RenderError> {
        let total = instances.len() + meshes.len();
        if total > self.capacity as usize {
            return Err(RenderError::new(format!(
                "{total} instances exceed the TLAS capacity {}",
                self.capacity
            )));
        }
        let record = |matrix: [f32; 12],
                      custom_index: u32,
                      mask: u8,
                      flags: vk::GeometryInstanceFlagsKHR,
                      blas: u64| {
            vk::AccelerationStructureInstanceKHR {
                // Row-major 3x4.
                transform: vk::TransformMatrixKHR { matrix },
                instance_custom_index_and_mask: vk::Packed24_8::new(custom_index, mask),
                instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(
                    0,
                    flags.as_raw() as u8,
                ),
                acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
                    device_handle: blas,
                },
            }
        };
        let mut records: Vec<vk::AccelerationStructureInstanceKHR> = Vec::with_capacity(total);
        records.extend(instances.iter().map(|inst| {
            let [x, y, z, scale] = inst.center;
            let mut mask = if casts_shadow(inst) {
                MASK_EVERYTHING
            } else {
                MASK_EVERYTHING & !MASK_SHADOW_CASTER
            };
            mask &= if inst.transmission > 0.0 {
                !MASK_OPAQUE
            } else {
                !MASK_GLASS
            };
            record(
                [
                    scale, 0.0, 0.0, x, //
                    0.0, scale, 0.0, y, //
                    0.0, 0.0, scale, z,
                ],
                0,
                mask,
                ShapeBlas::flags_for(inst.shape),
                shapes.address_for(inst.shape),
            )
        }));
        records.extend(meshes.iter().map(|mesh| {
            record(
                mesh.transform,
                mesh.custom_index.min(MAX_CUSTOM_INDEX),
                MASK_EVERYTHING & !MASK_GLASS,
                mesh.flags,
                mesh.blas,
            )
        }));
        // SAFETY: `AccelerationStructureInstanceKHR` is a `repr(C)` 64-byte
        // record of plain numbers with no padding.
        self.instances
            .write(unsafe { params::slice_bytes(&records) })?;
        Ok(total as u32)
    }

    /// Record the device-side build over the first `count` instance records,
    /// then the barrier that makes the result visible to the ray queries later
    /// in the same command buffer (the GI and ray-probe compute passes, and
    /// the shadow rays of the forward pass's fragment shaders).
    pub fn record_build(&self, device: &ash::Device, cmd: vk::CommandBuffer, count: u32) {
        let geometry = instances_geometry(self.instances.device_address());
        let info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .flags(Self::FLAGS)
            .dst_acceleration_structure(self.handle)
            .scratch_data(vk::DeviceOrHostAddressKHR {
                device_address: self.scratch.device_address(),
            })
            .geometries(std::slice::from_ref(&geometry));
        let ranges = [vk::AccelerationStructureBuildRangeInfoKHR {
            primitive_count: count.min(self.capacity),
            primitive_offset: 0,
            first_vertex: 0,
            transform_offset: 0,
        }];
        unsafe {
            self.accel.cmd_build_acceleration_structures(
                cmd,
                std::slice::from_ref(&info),
                &[&ranges],
            );
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
                vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR)
                    .dst_access_mask(vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR)],
                &[],
                &[],
            );
        }
    }
}

impl Drop for Tlas {
    fn drop(&mut self) {
        unsafe { self.accel.destroy_acceleration_structure(self.handle, None) };
    }
}

/// A CPU-written buffer of `T` records — a storage buffer (`array<Instance>` /
/// `array<MeshInstance>` in the shaders) or the overlay's instance-rate vertex
/// buffer — kept across frames and regrown only when a scene outgrows it.
pub(crate) struct InstanceBuffer<T: GpuPod> {
    pub buffer: Buffer,
    capacity: u32,
    _records: std::marker::PhantomData<fn(T)>,
}

impl<T: GpuPod> InstanceBuffer<T> {
    pub fn new(
        device: &ash::Device,
        allocator: &SharedAllocator,
        capacity: u32,
        usage: vk::BufferUsageFlags,
        name: &'static str,
    ) -> Result<Self, RenderError> {
        let capacity = capacity.max(1);
        let buffer = Buffer::new(
            device,
            allocator,
            u64::from(capacity) * std::mem::size_of::<T>() as u64,
            usage,
            gpu_allocator::MemoryLocation::CpuToGpu,
            name,
        )?;
        Ok(Self {
            buffer,
            capacity,
            _records: std::marker::PhantomData,
        })
    }

    /// How many records the buffer has room for.
    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    /// Upload `records` (at most [`InstanceBuffer::capacity`]).
    pub fn write(&self, records: &[T]) -> Result<(), RenderError> {
        if records.len() > self.capacity as usize {
            return Err(RenderError::new(format!(
                "{} records exceed the buffer capacity {}",
                records.len(),
                self.capacity
            )));
        }
        // SAFETY: `GpuPod` types are `repr(C)` f32/u32 plain data, no padding.
        self.buffer.write(unsafe { params::slice_bytes(records) })
    }
}

/// A model texture on the device: an sRGB image with its full mip chain.
pub(crate) struct TextureGpu {
    // Held so the dedicated allocation outlives the view.
    _image: Image,
    pub view: vk::ImageView,
    device: ash::Device,
}

impl Drop for TextureGpu {
    fn drop(&mut self) {
        // The view goes first; the image is freed by the field drop.
        unsafe { self.device.destroy_image_view(self.view, None) };
    }
}

impl TextureGpu {
    /// Create one sampled image per texture — [`TEXTURE_FORMAT`], every mip
    /// level from the CPU-side chain — and upload them all through one
    /// staging buffer and one fenced submission (nothing else may be pending
    /// on `queue`'s pool). A level larger than `max_dimension` (the device's
    /// 2D image limit) is left out, so an oversized texture loses its
    /// sharpest levels instead of failing.
    pub fn upload_all(
        device: &ash::Device,
        allocator: &SharedAllocator,
        queue: vk::Queue,
        cmd_pool: vk::CommandPool,
        textures: &[&Texture],
        max_dimension: u32,
    ) -> Result<Vec<Self>, RenderError> {
        if textures.is_empty() {
            return Ok(Vec::new());
        }
        let layers = |mip_level: u32| vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level,
            base_array_layer: 0,
            layer_count: 1,
        };
        let mut uploads = Vec::with_capacity(textures.len());
        // Per image: its handle, level count and the copy of every level.
        let mut copies: Vec<(vk::Image, u32, Vec<vk::BufferImageCopy>)> = Vec::new();
        let mut staged: Vec<u8> = Vec::new();
        for texture in textures {
            let chain = texture.mip_chain();
            let levels: Vec<_> = chain
                .iter()
                .filter(|level| level.width.max(level.height) <= max_dimension.max(1))
                .collect();
            // (Never empty: the chain ends in a 1x1 level.)
            let Some(top) = levels.first() else {
                return Err(RenderError::new("a texture has no mip level"));
            };
            let level_count = levels.len() as u32;
            let image = Image::new(
                device,
                allocator,
                &vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(TEXTURE_FORMAT)
                    .extent(vk::Extent3D {
                        width: top.width,
                        height: top.height,
                        depth: 1,
                    })
                    .mip_levels(level_count)
                    .array_layers(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .initial_layout(vk::ImageLayout::UNDEFINED)
                    .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST),
                gpu_allocator::MemoryLocation::GpuOnly,
                "mesh-texture",
            )?;
            let view = unsafe {
                device
                    .create_image_view(
                        &vk::ImageViewCreateInfo::default()
                            .image(image.handle)
                            .view_type(vk::ImageViewType::TYPE_2D)
                            .format(TEXTURE_FORMAT)
                            .subresource_range(vk::ImageSubresourceRange {
                                aspect_mask: vk::ImageAspectFlags::COLOR,
                                base_mip_level: 0,
                                level_count,
                                base_array_layer: 0,
                                layer_count: 1,
                            }),
                        None,
                    )
                    .map_err(|e| {
                        RenderError::with_name("mesh-texture", format!("create_image_view: {e}"))
                    })?
            };
            let regions = levels
                .iter()
                .enumerate()
                .map(|(level, mip)| {
                    // (Every level is whole RGBA8 texels, so offsets stay
                    // multiples of four, as a buffer-to-image copy requires.)
                    let offset = staged.len() as u64;
                    staged.extend_from_slice(&mip.pixels);
                    vk::BufferImageCopy::default()
                        .buffer_offset(offset)
                        .image_subresource(layers(level as u32))
                        .image_extent(vk::Extent3D {
                            width: mip.width,
                            height: mip.height,
                            depth: 1,
                        })
                })
                .collect();
            copies.push((image.handle, level_count, regions));
            // RAII from here: a later failure destroys the view and the image.
            uploads.push(Self {
                _image: image,
                view,
                device: device.clone(),
            });
        }
        let staging = Buffer::new(
            device,
            allocator,
            staged.len() as u64,
            vk::BufferUsageFlags::TRANSFER_SRC,
            gpu_allocator::MemoryLocation::CpuToGpu,
            "texture-staging",
        )?;
        staging.write(&staged)?;
        submit_once(device, queue, cmd_pool, "texture upload", |cmd| unsafe {
            for (image, level_count, regions) in &copies {
                let transition = |from: vk::ImageLayout,
                                  to: vk::ImageLayout,
                                  src: vk::AccessFlags,
                                  dst: vk::AccessFlags| {
                    vk::ImageMemoryBarrier::default()
                        .image(*image)
                        .old_layout(from)
                        .new_layout(to)
                        .src_access_mask(src)
                        .dst_access_mask(dst)
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: 0,
                            level_count: *level_count,
                            base_array_layer: 0,
                            layer_count: 1,
                        })
                };
                device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[transition(
                        vk::ImageLayout::UNDEFINED,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        vk::AccessFlags::empty(),
                        vk::AccessFlags::TRANSFER_WRITE,
                    )],
                );
                device.cmd_copy_buffer_to_image(
                    cmd,
                    staging.handle,
                    *image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    regions,
                );
                // Sampled by the mesh pipeline's fragment shader from here on.
                device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[transition(
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        vk::AccessFlags::TRANSFER_WRITE,
                        vk::AccessFlags::SHADER_READ,
                    )],
                );
            }
        })?;
        Ok(uploads)
    }
}

/// Descriptor sets for model textures: one set of
/// [`Pipelines::texture_set_layout`] (the image and the texture sampler) per
/// view, from a pool of their own that lives exactly as long as they do.
pub(crate) struct TextureSets {
    pool: vk::DescriptorPool,
    pub sets: Vec<vk::DescriptorSet>,
    device: ash::Device,
}

impl Drop for TextureSets {
    fn drop(&mut self) {
        if self.pool != vk::DescriptorPool::null() {
            // Destroying the pool frees its sets.
            unsafe { self.device.destroy_descriptor_pool(self.pool, None) };
        }
    }
}

impl TextureSets {
    pub fn new(
        device: &ash::Device,
        pipelines: &Pipelines,
        views: &[vk::ImageView],
    ) -> Result<Self, RenderError> {
        let mut out = Self {
            pool: vk::DescriptorPool::null(),
            sets: Vec::new(),
            device: device.clone(),
        };
        if views.is_empty() {
            return Ok(out);
        }
        let count = views.len() as u32;
        out.pool = unsafe {
            device
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .max_sets(count)
                        .pool_sizes(&[
                            vk::DescriptorPoolSize {
                                ty: vk::DescriptorType::SAMPLED_IMAGE,
                                descriptor_count: count,
                            },
                            vk::DescriptorPoolSize {
                                ty: vk::DescriptorType::SAMPLER,
                                descriptor_count: count,
                            },
                        ]),
                    None,
                )
                .map_err(|e| RenderError::new(format!("texture descriptor pool: {e}")))?
        };
        let layouts = vec![pipelines.texture_set_layout; views.len()];
        out.sets = unsafe {
            device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(out.pool)
                        .set_layouts(&layouts),
                )
                .map_err(|e| RenderError::new(format!("texture descriptor sets: {e}")))?
        };
        let image_infos: Vec<[vk::DescriptorImageInfo; 1]> = views
            .iter()
            .map(|view| {
                [vk::DescriptorImageInfo::default()
                    .image_view(*view)
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)]
            })
            .collect();
        let sampler_info = [vk::DescriptorImageInfo::default().sampler(pipelines.texture_sampler)];
        let mut writes = Vec::with_capacity(2 * views.len());
        for (set, image_info) in out.sets.iter().zip(&image_infos) {
            writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(*set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                    .image_info(image_info),
            );
            writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(*set)
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::SAMPLER)
                    .image_info(&sampler_info),
            );
        }
        unsafe { device.update_descriptor_sets(&writes, &[]) };
        Ok(out)
    }
}

/// The 1x1 white texture an untextured mesh part samples (white times the
/// base-colour factor is the factor), with its descriptor set. One per
/// context, created with the first model.
pub(crate) struct WhiteTexture {
    // Field order: the set's pool goes before the image it points at.
    sets: TextureSets,
    _texture: Vec<TextureGpu>,
}

impl WhiteTexture {
    pub fn new(
        device: &ash::Device,
        allocator: &SharedAllocator,
        queue: vk::Queue,
        cmd_pool: vk::CommandPool,
        pipelines: &Pipelines,
    ) -> Result<Self, RenderError> {
        let white = Texture::solid([255; 3]);
        let texture = TextureGpu::upload_all(device, allocator, queue, cmd_pool, &[&white], 1)?;
        let views: Vec<vk::ImageView> = texture.iter().map(|t| t.view).collect();
        let sets = TextureSets::new(device, pipelines, &views)?;
        Ok(Self {
            sets,
            _texture: texture,
        })
    }

    /// The descriptor set to bind for an untextured part.
    pub fn set(&self) -> Result<vk::DescriptorSet, RenderError> {
        self.sets
            .sets
            .first()
            .copied()
            .ok_or_else(|| RenderError::new("the white texture has no descriptor set"))
    }
}

/// One drawn part of an uploaded mesh model.
#[derive(Clone, Copy, Debug)]
pub(crate) struct MeshPartGpu {
    /// Which of `MeshData::parts` this is.
    pub part: usize,
    pub first_index: u32,
    pub index_count: u32,
    /// The part's base-colour texture ([`WhiteTexture`]'s for a part without).
    pub texture_set: vk::DescriptorSet,
}

/// A mesh model on the device: vertex and index buffers, its textures, and —
/// once a frame that traces rays has drawn it — its acceleration structure.
pub(crate) struct MeshGpu {
    pub vertices: Buffer,
    pub indices: Buffer,
    /// The parts that have triangles, in mesh order. Draw records and the
    /// BLAS's geometries both follow this order, which is what lets a ray hit
    /// find its part's record (see [`MeshTlasInstance::custom_index`]).
    pub parts: Vec<MeshPartGpu>,
    /// How shadow rays treat the model's faces ([`instance_flags_for`]).
    pub instance_flags: vk::GeometryInstanceFlagsKHR,
    max_vertex: u32,
    blas: Option<Blas>,
    // Field order: the sets' pool goes before the textures they point at.
    _texture_sets: TextureSets,
    _textures: Vec<TextureGpu>,
}

impl MeshGpu {
    /// Is the model an open surface — a sheet, both of whose sides face the
    /// open — rather than a closed solid? (The question shadow rays ask of
    /// it too: see [`instance_flags_for`].)
    pub fn is_sheet(&self) -> bool {
        self.instance_flags
            .contains(vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE)
    }

    /// Upload `mesh` — vertices and indices into device-local buffers (through
    /// staging buffers), every texture with its mip chain — in fenced
    /// submissions (nothing else may be pending on `queue`'s pool). `white` is
    /// the set an untextured part binds; `max_texture_dimension` the device's
    /// 2D image limit.
    #[allow(clippy::too_many_arguments)]
    pub fn upload(
        device: &ash::Device,
        allocator: &SharedAllocator,
        queue: vk::Queue,
        cmd_pool: vk::CommandPool,
        pipelines: &Pipelines,
        mesh: &crate::mesh::MeshData,
        white: vk::DescriptorSet,
        max_texture_dimension: u32,
    ) -> Result<Self, RenderError> {
        // Out-of-range indices would send the GPU (and the BLAS build) past
        // the end of the vertex buffer: refuse a broken mesh here.
        mesh.validate()
            .map_err(|e| RenderError::new(format!("mesh model: {e}")))?;
        // SAFETY: `MeshVertex` is `repr(C)` with eight `f32`s and no padding;
        // `u32` is plain data.
        let (vertex_bytes, index_bytes) = unsafe {
            (
                params::slice_bytes(&mesh.vertices),
                params::slice_bytes(&mesh.indices),
            )
        };
        let drawn: Vec<(usize, &crate::mesh::MeshPart)> = mesh
            .parts
            .iter()
            .enumerate()
            .filter(|(_, part)| part.index_count >= 3)
            .collect();
        if vertex_bytes.is_empty() || drawn.is_empty() {
            return Err(RenderError::new("a mesh model has no triangles"));
        }
        // Both buffers are also the BLAS build's input.
        let staged = |bytes: &[u8], usage: vk::BufferUsageFlags, name: &'static str| {
            let staging = Buffer::new(
                device,
                allocator,
                bytes.len() as u64,
                vk::BufferUsageFlags::TRANSFER_SRC,
                gpu_allocator::MemoryLocation::CpuToGpu,
                name,
            )?;
            staging.write(bytes)?;
            let target = Buffer::new(
                device,
                allocator,
                bytes.len() as u64,
                usage | vk::BufferUsageFlags::TRANSFER_DST | as_input_usage(),
                gpu_allocator::MemoryLocation::GpuOnly,
                name,
            )?;
            Ok::<_, RenderError>((staging, target, bytes.len() as u64))
        };
        let (vertex_staging, vertices, vertex_size) = staged(
            vertex_bytes,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            "mesh-vertices",
        )?;
        let (index_staging, indices, index_size) = staged(
            index_bytes,
            vk::BufferUsageFlags::INDEX_BUFFER,
            "mesh-indices",
        )?;
        submit_once(device, queue, cmd_pool, "mesh upload", |cmd| unsafe {
            for (from, to, size) in [
                (&vertex_staging, &vertices, vertex_size),
                (&index_staging, &indices, index_size),
            ] {
                device.cmd_copy_buffer(
                    cmd,
                    from.handle,
                    to.handle,
                    &[vk::BufferCopy::default().size(size)],
                );
            }
        })?;

        let sources: Vec<&Texture> = mesh.textures.iter().collect();
        let textures = TextureGpu::upload_all(
            device,
            allocator,
            queue,
            cmd_pool,
            &sources,
            max_texture_dimension,
        )?;
        let views: Vec<vk::ImageView> = textures.iter().map(|t| t.view).collect();
        let texture_sets = TextureSets::new(device, pipelines, &views)?;
        let parts = drawn
            .iter()
            .map(|&(part, source)| MeshPartGpu {
                part,
                first_index: source.first_index,
                index_count: source.index_count,
                texture_set: source
                    .material
                    .base_color_texture
                    .and_then(|slot| texture_sets.sets.get(slot as usize))
                    .copied()
                    .unwrap_or(white),
            })
            .collect();
        Ok(Self {
            vertices,
            indices,
            parts,
            instance_flags: instance_flags_for(mesh),
            max_vertex: (mesh.vertices.len() - 1) as u32,
            blas: None,
            _texture_sets: texture_sets,
            _textures: textures,
        })
    }

    /// Has the model's acceleration structure been built yet?
    pub fn has_blas(&self) -> bool {
        self.blas.is_some()
    }

    /// The device address of the model's bottom-level acceleration structure,
    /// building it first if this is the first frame that needs it (one fenced
    /// submission; nothing else may be pending on `queue`'s pool). It has one
    /// geometry per drawn part, in order, all over the shared vertex buffer.
    pub fn ensure_blas(
        &mut self,
        device: &ash::Device,
        accel: &ash::khr::acceleration_structure::Device,
        allocator: &SharedAllocator,
        queue: vk::Queue,
        cmd_pool: vk::CommandPool,
    ) -> Result<u64, RenderError> {
        if let Some(blas) = &self.blas {
            return Ok(blas.address);
        }
        let geometry = triangles_geometry(
            self.vertices.device_address(),
            std::mem::size_of::<crate::mesh::MeshVertex>() as u64,
            self.indices.device_address(),
            self.max_vertex,
        );
        let geometries = vec![geometry; self.parts.len()];
        let mut ranges = Vec::with_capacity(self.parts.len());
        for part in &self.parts {
            // The part's triangles: a byte offset into the index data.
            let offset = part
                .first_index
                .checked_mul(4)
                .ok_or_else(|| RenderError::new("a mesh part starts past 4 GiB of indices"))?;
            ranges.push(vk::AccelerationStructureBuildRangeInfoKHR {
                primitive_count: part.index_count / 3,
                primitive_offset: offset,
                first_vertex: 0,
                transform_offset: 0,
            });
        }
        let counts: Vec<u32> = ranges.iter().map(|r| r.primitive_count).collect();
        let info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .geometries(&geometries);
        let sizes = build_sizes(accel, &info, &counts);
        let storage = Buffer::new(
            device,
            allocator,
            sizes.acceleration_structure_size.max(16),
            as_storage_usage(),
            gpu_allocator::MemoryLocation::GpuOnly,
            "mesh-blas",
        )?;
        // Only needed until the build below has run.
        let scratch = Buffer::new(
            device,
            allocator,
            sizes.build_scratch_size.max(256),
            as_scratch_usage(),
            gpu_allocator::MemoryLocation::GpuOnly,
            "mesh-blas-scratch",
        )?;
        let handle = unsafe {
            accel
                .create_acceleration_structure(
                    &vk::AccelerationStructureCreateInfoKHR::default()
                        .buffer(storage.handle)
                        .size(sizes.acceleration_structure_size)
                        .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL),
                    None,
                )
                .map_err(|e| RenderError::with_name("mesh-blas", format!("create BLAS: {e}")))?
        };
        let address = unsafe {
            accel.get_acceleration_structure_device_address(
                &vk::AccelerationStructureDeviceAddressInfoKHR::default()
                    .acceleration_structure(handle),
            )
        };
        // RAII from here: a failed build destroys the structure again.
        let blas = Blas {
            handle,
            address,
            _storage: storage,
            accel: accel.clone(),
        };
        let info =
            info.dst_acceleration_structure(handle)
                .scratch_data(vk::DeviceOrHostAddressKHR {
                    device_address: scratch.device_address(),
                });
        submit_once(device, queue, cmd_pool, "mesh BLAS build", |cmd| unsafe {
            accel.cmd_build_acceleration_structures(
                cmd,
                std::slice::from_ref(&info),
                &[ranges.as_slice()],
            );
        })?;
        self.blas = Some(blas);
        Ok(address)
    }
}

/// One indexed draw of a mesh part.
#[derive(Clone, Copy, Debug)]
pub(crate) struct MeshDraw {
    pub vertices: vk::Buffer,
    pub indices: vk::Buffer,
    pub first_index: u32,
    pub index_count: u32,
    /// Index of the part's record in the mesh instance buffer.
    pub record: u32,
    /// The part's base-colour texture (descriptor set 1).
    pub texture_set: vk::DescriptorSet,
}

// ---- per-frame resources -------------------------------------------------------

/// Everything that decides which GPU resources a frame needs. A cached
/// [`FrameGpu`] is reused for as long as this is unchanged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct FrameKey {
    pub width: u32,
    pub height: u32,
    /// Probe grid edge length when GI is on.
    pub gi_grid: Option<u32>,
    /// The SSR / refraction post pass runs.
    pub post: bool,
    /// The glass pass (refraction) runs; implies `post`.
    pub glass: bool,
}

impl FrameKey {
    pub fn new(width: u32, height: u32, options: &RenderOptions) -> Self {
        let features = options.preset.features();
        // The post pass never runs for the raw ray-query probe frame.
        let post = features.needs_post() && !options.ray_probe;
        Self {
            width,
            height,
            gi_grid: options.gi_enabled.then_some(options.quality.grid),
            post,
            glass: post && features.flags() & FEATURE_REFRACT != 0,
        }
    }
}

/// The GPU resources one frame renders with: images, views, framebuffers,
/// uniform buffers, descriptor sets, one command buffer and its fence. A
/// context keeps the last one and reuses it while the [`FrameKey`] is
/// unchanged, so a steady stream of frames allocates nothing.
///
/// Everything is RAII; dropping the frame frees every dedicated allocation.
/// The caller must make sure the last submission has completed first
/// ([`FrameGpu::wait`]).
pub(crate) struct FrameGpu {
    width: u32,
    height: u32,
    // `hdr`/`depth`/`material` are never read directly: they are held so their
    // dedicated allocations outlive the framebuffer and views that reference
    // them.
    _hdr: Image,
    hdr_view: vk::ImageView,
    _depth: Image,
    depth_view: vk::ImageView,
    /// Material buffer: r = metallic, g = roughness, ba = shading normal.
    _material: Image,
    material_view: vk::ImageView,
    _specular: Image,
    specular_view: vk::ImageView,
    /// SSR / refraction resources; `None` when the preset needs no post pass.
    post: Option<PostGpu>,
    /// The tone-mapped `rgba8unorm` result.
    out: Image,
    out_view: vk::ImageView,
    framebuffer: vk::Framebuffer,
    /// `out` as the overlay pass's colour attachment.
    overlay_framebuffer: vk::Framebuffer,
    gi_tex: Option<(Image, vk::ImageView)>,
    gi_extent: (u32, u32),
    dummy_tex: (Image, vk::ImageView),
    readback: Buffer,
    pub forward_params: Buffer,
    pub gi_params: Buffer,
    pub tone_params: Buffer,
    pub rq_params: Buffer,
    pool: vk::DescriptorPool,
    forward_set: vk::DescriptorSet,
    gi_set: vk::DescriptorSet,
    tone_set: vk::DescriptorSet,
    rq_set: vk::DescriptorSet,
    cmd: vk::CommandBuffer,
    cmd_pool: vk::CommandPool,
    /// Signalled whenever no submission of `cmd` is pending (created signalled).
    fence: vk::Fence,
    device: ash::Device,
}

/// Per-frame resources that exist only when the post pass runs.
struct PostGpu {
    /// Composited HDR colour the tone map reads instead of the raw HDR target.
    out: Image,
    out_view: vk::ImageView,
    /// Glass colour (rgb lit, a = transmission) + framebuffer; refraction only.
    _glass: Option<Image>,
    glass_view: Option<vk::ImageView>,
    glass_framebuffer: Option<vk::Framebuffer>,
    params: Buffer,
    set: vk::DescriptorSet,
}

/// Where a frame's tone-mapped image goes at the end of its command buffer.
#[derive(Clone, Copy, Debug)]
pub(crate) enum FrameTarget {
    /// Copy it into the CPU-visible readback buffer ([`FrameGpu::read_pixels`]).
    Readback,
    /// Blit it into this acquired swapchain image and leave that image in
    /// `PRESENT_SRC_KHR`. Nothing is read back.
    Present(vk::Image),
}

/// One instanced draw: a contiguous run of instances sharing a vertex count.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct DrawRun {
    pub first_instance: u32,
    pub instance_count: u32,
    /// Vertices per instance.
    pub vertex_count: u32,
    /// The draw's first vertex: the sphere detail level in the top byte (see
    /// `sphere_position` in `shaders/forward.wgsl`).
    pub first_vertex: u32,
}

/// What one frame draws.
pub(crate) struct FrameDraw<'a> {
    /// Instance records the TLAS is built over: every render instance, then
    /// every model placement ([`Tlas::write_instances`]).
    pub tlas_instances: u32,
    /// The instanced draws, together covering every instance once.
    pub runs: &'a [DrawRun],
    /// The mesh model parts (opaque pass only).
    pub meshes: &'a [MeshDraw],
    /// Rebuild this TLAS before any ray query runs. `Some` exactly when the
    /// frame traces rays (GI, shadows or the raw ray probe).
    pub tlas: Option<&'a Tlas>,
    /// Dispatch the GI probe pass (false with GI off or for the mutation test).
    pub gi_dispatch: bool,
    /// Draw with the shadowed forward pipelines, whose fragment shaders trace
    /// a ray to every light. Only ever set together with `tlas`.
    pub shadows: bool,
    /// Run the post pass with ray queries: glass is refracted through its
    /// real shape, and a mirror shows what is off screen. Only ever set
    /// together with `tlas`.
    pub glass_rays: bool,
    /// Draw the scene's environment as the background (after the opaque
    /// geometry, on the pixels it left empty). False for a scene without an
    /// environment or one that only wants its light: nothing is drawn then.
    pub sky: bool,
    /// The overlay's instance buffer and how many rectangles it holds; `None`
    /// for an empty overlay, and then the overlay pass is not recorded at all.
    pub overlay: Option<(vk::Buffer, u32)>,
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

const fn color_layers() -> vk::ImageSubresourceLayers {
    vk::ImageSubresourceLayers {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        mip_level: 0,
        base_array_layer: 0,
        layer_count: 1,
    }
}

/// Objects [`FrameGpu::new`] has created so far that are not RAII: destroyed
/// by `Drop` if a later step fails, so a failed frame creation leaks nothing.
struct Partial<'a> {
    device: &'a ash::Device,
    views: Vec<vk::ImageView>,
    framebuffers: Vec<vk::Framebuffer>,
    pool: vk::DescriptorPool,
    cmd: Option<(vk::CommandPool, vk::CommandBuffer)>,
    armed: bool,
}

impl Drop for Partial<'_> {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        unsafe {
            for fb in &self.framebuffers {
                self.device.destroy_framebuffer(*fb, None);
            }
            for view in &self.views {
                self.device.destroy_image_view(*view, None);
            }
            if self.pool != vk::DescriptorPool::null() {
                self.device.destroy_descriptor_pool(self.pool, None);
            }
            if let Some((pool, cmd)) = self.cmd {
                self.device.free_command_buffers(pool, &[cmd]);
            }
        }
    }
}

impl FrameGpu {
    /// Create every resource a frame of `key` needs. Nothing here depends on
    /// the scene: the scene's buffers are bound by [`FrameGpu::bind_scene`].
    pub fn new(
        device: &ash::Device,
        allocator: &SharedAllocator,
        pipelines: &Pipelines,
        cmd_pool: vk::CommandPool,
        key: FrameKey,
    ) -> Result<Self, RenderError> {
        let FrameKey { width, height, .. } = key;
        let mut partial = Partial {
            device,
            views: Vec::new(),
            framebuffers: Vec::new(),
            pool: vk::DescriptorPool::null(),
            cmd: None,
            armed: true,
        };
        let image = |w: u32,
                     h: u32,
                     format: vk::Format,
                     usage: vk::ImageUsageFlags,
                     name: &'static str,
                     partial: &mut Partial<'_>|
         -> Result<(Image, vk::ImageView), RenderError> {
            let (img, view) = make_image(device, allocator, w, h, format, usage, name)?;
            partial.views.push(view);
            Ok((img, view))
        };
        let framebuffer = |render_pass: vk::RenderPass,
                           attachments: &[vk::ImageView],
                           partial: &mut Partial<'_>|
         -> Result<vk::Framebuffer, RenderError> {
            let fb = unsafe {
                device
                    .create_framebuffer(
                        &vk::FramebufferCreateInfo::default()
                            .render_pass(render_pass)
                            .attachments(attachments)
                            .width(width)
                            .height(height)
                            .layers(1),
                        None,
                    )
                    .map_err(|e| RenderError::new(format!("framebuffer: {e}")))?
            };
            partial.framebuffers.push(fb);
            Ok(fb)
        };

        let (hdr, hdr_view) = image(
            width,
            height,
            vk::Format::R16G16B16A16_SFLOAT,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            "hdr-image",
            &mut partial,
        )?;
        let (depth, depth_view) = image(
            width,
            height,
            vk::Format::D32_SFLOAT,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            "depth-image",
            &mut partial,
        )?;
        let (material, material_view) = image(
            width,
            height,
            MATERIAL_FORMAT,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            "material-image",
            &mut partial,
        )?;
        let (specular, specular_view) = image(
            width,
            height,
            SPECULAR_FORMAT,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            "specular-image",
            &mut partial,
        )?;
        let (out, out_view) = image(
            width,
            height,
            OUT_FORMAT,
            // Written by the tone map (storage), blended onto by the overlay
            // pass (attachment), then copied or blitted out (transfer source).
            vk::ImageUsageFlags::STORAGE
                | vk::ImageUsageFlags::COLOR_ATTACHMENT
                | vk::ImageUsageFlags::TRANSFER_SRC,
            "out-image",
            &mut partial,
        )?;
        let main_framebuffer = framebuffer(
            pipelines.render_pass,
            &[hdr_view, material_view, specular_view, depth_view],
            &mut partial,
        )?;
        let overlay_framebuffer =
            framebuffer(pipelines.render_pass_overlay, &[out_view], &mut partial)?;

        // Post pass resources (SSR + refraction): only when the preset needs
        // them.
        let post_images = if key.post {
            let (img, view) = image(
                width,
                height,
                vk::Format::R16G16B16A16_SFLOAT,
                vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
                "post-image",
                &mut partial,
            )?;
            let glass = if key.glass {
                let (gimg, gview) = image(
                    width,
                    height,
                    vk::Format::R16G16B16A16_SFLOAT,
                    vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
                    "glass-image",
                    &mut partial,
                )?;
                let fb = framebuffer(
                    pipelines.render_pass_glass,
                    &[gview, material_view, depth_view],
                    &mut partial,
                )?;
                Some((gimg, gview, fb))
            } else {
                None
            };
            Some((img, view, glass))
        } else {
            None
        };

        // GI probe texture: grid x (grid*6) rgba16f, plus a 1x1 dummy bound
        // when GI is disabled so the forward descriptor is always complete.
        let gi_tex = match key.gi_grid {
            Some(grid) => Some(image(
                grid,
                grid * 6,
                vk::Format::R16G16B16A16_SFLOAT,
                vk::ImageUsageFlags::STORAGE
                    | vk::ImageUsageFlags::SAMPLED
                    | vk::ImageUsageFlags::TRANSFER_DST,
                "gi-probe-tex",
                &mut partial,
            )?),
            None => None,
        };
        let gi_extent = key.gi_grid.map_or((0, 0), |grid| (grid, grid * 6));
        let (dummy_tex, dummy_view) = image(
            1,
            1,
            vk::Format::R16G16B16A16_SFLOAT,
            vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE,
            "gi-dummy-tex",
            &mut partial,
        )?;
        let gi_view_active = gi_tex.as_ref().map_or(dummy_view, |(_, v)| *v);

        let readback = Buffer::new(
            device,
            allocator,
            u64::from(width) * u64::from(height) * 4,
            vk::BufferUsageFlags::TRANSFER_DST,
            gpu_allocator::MemoryLocation::GpuToCpu,
            "readback",
        )?;

        let uniform = |size: usize, name: &'static str| {
            Buffer::new(
                device,
                allocator,
                size as u64,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                gpu_allocator::MemoryLocation::CpuToGpu,
                name,
            )
        };
        let forward_params = uniform(std::mem::size_of::<ForwardParams>(), "forward-params")?;
        let gi_params = uniform(std::mem::size_of::<GiParams>(), "gi-params")?;
        let tone_params = uniform(std::mem::size_of::<ToneParams>(), "tone-params")?;
        let rq_params = uniform(std::mem::size_of::<RqParams>(), "rq-params")?;
        let post_params = if key.post {
            Some(uniform(std::mem::size_of::<PostParams>(), "post-params")?)
        } else {
            None
        };

        // ---- descriptors -------------------------------------------------------
        let pool = unsafe {
            device
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .max_sets(5)
                        .pool_sizes(&[
                            vk::DescriptorPoolSize {
                                ty: vk::DescriptorType::UNIFORM_BUFFER,
                                descriptor_count: 5,
                            },
                            vk::DescriptorPoolSize {
                                ty: vk::DescriptorType::STORAGE_BUFFER,
                                descriptor_count: 4,
                            },
                            vk::DescriptorPoolSize {
                                ty: vk::DescriptorType::SAMPLED_IMAGE,
                                descriptor_count: 6,
                            },
                            vk::DescriptorPoolSize {
                                ty: vk::DescriptorType::SAMPLER,
                                descriptor_count: 2,
                            },
                            vk::DescriptorPoolSize {
                                ty: vk::DescriptorType::STORAGE_IMAGE,
                                descriptor_count: 4,
                            },
                            vk::DescriptorPoolSize {
                                ty: vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
                                descriptor_count: 3,
                            },
                        ]),
                    None,
                )
                .map_err(|e| RenderError::new(format!("descriptor pool: {e}")))?
        };
        partial.pool = pool;
        let alloc_set =
            |layout: vk::DescriptorSetLayout| -> Result<vk::DescriptorSet, RenderError> {
                unsafe {
                    device
                        .allocate_descriptor_sets(
                            &vk::DescriptorSetAllocateInfo::default()
                                .descriptor_pool(pool)
                                .set_layouts(std::slice::from_ref(&layout)),
                        )
                        .map_err(|e| RenderError::new(format!("allocate_descriptor_sets: {e}")))?
                        .first()
                        .copied()
                        .ok_or_else(|| RenderError::new("allocate_descriptor_sets returned no set"))
                }
            };
        let forward_set = alloc_set(pipelines.forward_set_layout)?;
        let gi_set = alloc_set(pipelines.gi_set_layout)?;
        let tone_set = alloc_set(pipelines.tone_set_layout)?;
        let rq_set = alloc_set(pipelines.rq_set_layout)?;

        let ub = |buf: &Buffer, size: usize| {
            vk::DescriptorBufferInfo::default()
                .buffer(buf.handle)
                .offset(0)
                .range(size as u64)
        };
        let img_info = |view: vk::ImageView, layout: vk::ImageLayout| {
            vk::DescriptorImageInfo::default()
                .image_view(view)
                .image_layout(layout)
        };
        let sampler_info = [vk::DescriptorImageInfo::default().sampler(pipelines.sampler)];

        // Everything except the scene's own buffers (the instance array, the
        // mesh part records and the TLAS, see `bind_scene`).
        unsafe {
            device.update_descriptor_sets(
                &[
                    // forward: 0 params, [1 instances], 2 gi tex, 3 sampler,
                    // [4 mesh records], [5 TLAS]
                    vk::WriteDescriptorSet::default()
                        .dst_set(forward_set)
                        .dst_binding(0)
                        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                        .buffer_info(&[ub(&forward_params, std::mem::size_of::<ForwardParams>())]),
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
                        .image_info(&sampler_info),
                    // gi: [0 TLAS], 1 params, [2 instances], 3 out,
                    // [4 mesh records]
                    vk::WriteDescriptorSet::default()
                        .dst_set(gi_set)
                        .dst_binding(1)
                        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                        .buffer_info(&[ub(&gi_params, std::mem::size_of::<GiParams>())]),
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
                            // With the post pass the tone map reads its output.
                            post_images.as_ref().map_or(hdr_view, |(_, v, _)| *v),
                            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        )]),
                    vk::WriteDescriptorSet::default()
                        .dst_set(tone_set)
                        .dst_binding(1)
                        .descriptor_type(vk::DescriptorType::SAMPLER)
                        .image_info(&sampler_info),
                    vk::WriteDescriptorSet::default()
                        .dst_set(tone_set)
                        .dst_binding(2)
                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                        .image_info(&[img_info(out_view, vk::ImageLayout::GENERAL)]),
                    vk::WriteDescriptorSet::default()
                        .dst_set(tone_set)
                        .dst_binding(3)
                        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                        .buffer_info(&[ub(&tone_params, std::mem::size_of::<ToneParams>())]),
                    // rq: [0 TLAS], 1 out storage image, 2 params
                    vk::WriteDescriptorSet::default()
                        .dst_set(rq_set)
                        .dst_binding(1)
                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                        .image_info(&[img_info(out_view, vk::ImageLayout::GENERAL)]),
                    vk::WriteDescriptorSet::default()
                        .dst_set(rq_set)
                        .dst_binding(2)
                        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                        .buffer_info(&[ub(&rq_params, std::mem::size_of::<RqParams>())]),
                ],
                &[],
            );
        }

        let post = match (post_images, post_params) {
            (Some((out_img, out_view_p, glass)), Some(params_buf)) => {
                let set = alloc_set(pipelines.post_set_layout)?;
                let (glass_img, glass_view, glass_fb) = match glass {
                    Some((i, v, f)) => (Some(i), Some(v), Some(f)),
                    None => (None, None, None),
                };
                // Without refraction the glass binding is a never-read
                // placeholder (the shader only loads it with FEATURE_REFRACT).
                let glass_bound = glass_view.unwrap_or(dummy_view);
                let sampled = |binding: u32, view: vk::ImageView, layout: vk::ImageLayout| {
                    [vk::DescriptorImageInfo::default()
                        .image_view(view)
                        .image_layout(layout)]
                    .map(|info| (binding, info))
                };
                let read_only = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
                // post: 0 params, 1 scene, 2 depth, 3 material, 4 glass, 5 out,
                // 6 specular
                let images = [
                    sampled(1, hdr_view, read_only),
                    sampled(
                        2,
                        depth_view,
                        vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
                    ),
                    sampled(3, material_view, read_only),
                    sampled(4, glass_bound, read_only),
                    sampled(6, specular_view, read_only),
                ];
                let out_info = [img_info(out_view_p, vk::ImageLayout::GENERAL)];
                let params_info = [ub(&params_buf, std::mem::size_of::<PostParams>())];
                let mut writes = vec![vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(&params_info)];
                for [(binding, info)] in &images {
                    writes.push(
                        vk::WriteDescriptorSet::default()
                            .dst_set(set)
                            .dst_binding(*binding)
                            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                            .image_info(std::slice::from_ref(info)),
                    );
                }
                writes.push(
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(5)
                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                        .image_info(&out_info),
                );
                unsafe { device.update_descriptor_sets(&writes, &[]) };
                Some(PostGpu {
                    out: out_img,
                    out_view: out_view_p,
                    _glass: glass_img,
                    glass_view,
                    glass_framebuffer: glass_fb,
                    params: params_buf,
                    set,
                })
            }
            _ => None,
        };

        let cmd = allocate_command_buffer(device, cmd_pool)?;
        partial.cmd = Some((cmd_pool, cmd));
        // Signalled: nothing is pending, so the first `wait` returns at once.
        let fence = unsafe {
            device
                .create_fence(
                    &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                    None,
                )
                .map_err(|e| RenderError::new(format!("create_fence: {e}")))?
        };

        // Everything is owned by the frame from here on.
        partial.armed = false;
        Ok(Self {
            width,
            height,
            _hdr: hdr,
            hdr_view,
            _depth: depth,
            depth_view,
            _material: material,
            material_view,
            _specular: specular,
            specular_view,
            post,
            out,
            out_view,
            framebuffer: main_framebuffer,
            overlay_framebuffer,
            gi_tex,
            gi_extent,
            dummy_tex: (dummy_tex, dummy_view),
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
            cmd,
            cmd_pool,
            fence,
            device: device.clone(),
        })
    }

    /// The post pass's uniform buffer, when this frame has a post pass.
    pub fn post_params(&self) -> Option<&Buffer> {
        self.post.as_ref().map(|p| &p.params)
    }

    /// Point the descriptor sets at the scene's instance buffer, its mesh part
    /// records (the forward pass draws with them, the GI pass looks up what a
    /// ray hit in them) and, for a frame that traces rays, its TLAS. Only
    /// legal while no submission is pending ([`FrameGpu::wait`]).
    pub fn bind_scene(
        &self,
        device: &ash::Device,
        instances: &Buffer,
        mesh_records: &Buffer,
        tlas: Option<vk::AccelerationStructureKHR>,
    ) {
        let instance_info = [vk::DescriptorBufferInfo::default()
            .buffer(instances.handle)
            .offset(0)
            .range(vk::WHOLE_SIZE)];
        let mesh_info = [vk::DescriptorBufferInfo::default()
            .buffer(mesh_records.handle)
            .offset(0)
            .range(vk::WHOLE_SIZE)];
        let instance_write = |set: vk::DescriptorSet, binding: u32| {
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(binding)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&instance_info)
        };
        let structures = [tlas.unwrap_or_default()];
        let mut as_write_gi = vk::WriteDescriptorSetAccelerationStructureKHR::default()
            .acceleration_structures(&structures);
        let mut as_write_rq = vk::WriteDescriptorSetAccelerationStructureKHR::default()
            .acceleration_structures(&structures);
        let mut as_write_forward = vk::WriteDescriptorSetAccelerationStructureKHR::default()
            .acceleration_structures(&structures);
        let mut as_write_post = vk::WriteDescriptorSetAccelerationStructureKHR::default()
            .acceleration_structures(&structures);
        let mesh_write = |set: vk::DescriptorSet| {
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(4)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&mesh_info)
        };
        let mut writes = vec![
            instance_write(self.forward_set, 1),
            instance_write(self.gi_set, 2),
            mesh_write(self.forward_set),
            mesh_write(self.gi_set),
        ];
        // The post pass's ray-query variant looks up what its rays hit.
        if let Some(post) = &self.post {
            writes.push(instance_write(post.set, 8));
            writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(post.set)
                    .dst_binding(9)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&mesh_info),
            );
            if tlas.is_some() {
                writes.push(
                    vk::WriteDescriptorSet::default()
                        .dst_set(post.set)
                        .dst_binding(7)
                        .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                        .descriptor_count(1)
                        .dst_array_element(0)
                        .push_next(&mut as_write_post),
                );
            }
        }
        if tlas.is_some() {
            // (The forward set's TLAS is only read by the shadowed pipelines.)
            for (set, binding, next) in [
                (self.gi_set, 0, &mut as_write_gi),
                (self.rq_set, 0, &mut as_write_rq),
                (self.forward_set, 5, &mut as_write_forward),
            ] {
                writes.push(
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(binding)
                        .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                        .descriptor_count(1)
                        .dst_array_element(0)
                        .push_next(next),
                );
            }
        }
        unsafe { device.update_descriptor_sets(&writes, &[]) };
    }

    /// Block until the last submission of this frame has completed (returns
    /// at once when none is pending). `Err` on timeout or device loss.
    pub fn wait(&self, timeout_ns: u64) -> Result<(), RenderError> {
        unsafe {
            self.device
                .wait_for_fences(std::slice::from_ref(&self.fence), true, timeout_ns)
                .map_err(|e| RenderError::new(format!("wait for the previous frame: {e}")))
        }
    }

    /// Submit the recorded command buffer. `wait` is a semaphore the
    /// submission waits for at the given stage (the swapchain image
    /// acquisition); `signal` one it signals when done (for the present).
    /// The caller must have [`FrameGpu::wait`]ed since the previous submit.
    pub fn submit(
        &self,
        queue: vk::Queue,
        wait: Option<(vk::Semaphore, vk::PipelineStageFlags)>,
        signal: Option<vk::Semaphore>,
    ) -> Result<(), RenderError> {
        let (wait_semaphores, wait_stages): (Vec<_>, Vec<_>) = wait.into_iter().unzip();
        let signal_semaphores: Vec<vk::Semaphore> = signal.into_iter().collect();
        let submit = vk::SubmitInfo::default()
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&wait_stages)
            .command_buffers(std::slice::from_ref(&self.cmd))
            .signal_semaphores(&signal_semaphores);
        unsafe {
            self.device
                .reset_fences(std::slice::from_ref(&self.fence))
                .map_err(|e| RenderError::new(format!("reset_fences: {e}")))?;
            self.device
                .queue_submit(queue, std::slice::from_ref(&submit), self.fence)
                .map_err(|e| RenderError::new(format!("queue_submit: {e}")))
        }
    }

    /// Bind `pipeline`, set the full-frame viewport/scissor and draw every
    /// instance (inside an active render pass).
    fn draw_scene(
        &self,
        device: &ash::Device,
        pipelines: &Pipelines,
        pipeline: vk::Pipeline,
        draw: &FrameDraw<'_>,
    ) {
        unsafe {
            device.cmd_bind_pipeline(self.cmd, vk::PipelineBindPoint::GRAPHICS, pipeline);
            device.cmd_set_viewport(
                self.cmd,
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
            device.cmd_set_scissor(self.cmd, 0, &[self.full_rect()]);
            device.cmd_bind_descriptor_sets(
                self.cmd,
                vk::PipelineBindPoint::GRAPHICS,
                pipelines.forward_layout,
                0,
                std::slice::from_ref(&self.forward_set),
                &[],
            );
            // One instanced draw per run of instances sharing a detail level.
            for run in draw.runs {
                if run.instance_count > 0 && run.vertex_count > 0 {
                    device.cmd_draw(
                        self.cmd,
                        run.vertex_count,
                        run.instance_count,
                        run.first_vertex,
                        run.first_instance,
                    );
                }
            }
        }
    }

    /// Draw the mesh model parts with `pipeline` (a mesh pipeline; inside the
    /// opaque render pass, after [`FrameGpu::draw_scene`] has set viewport,
    /// scissor and descriptor set 0 — the forward pipelines share one layout,
    /// so that set stays bound).
    fn draw_meshes(
        &self,
        device: &ash::Device,
        pipelines: &Pipelines,
        pipeline: vk::Pipeline,
        draw: &FrameDraw<'_>,
    ) {
        if draw.meshes.is_empty() {
            return;
        }
        unsafe {
            device.cmd_bind_pipeline(self.cmd, vk::PipelineBindPoint::GRAPHICS, pipeline);
            // Parts of one model are consecutive: rebind only on a change.
            let mut bound = (vk::Buffer::null(), vk::Buffer::null());
            let mut texture = vk::DescriptorSet::null();
            for part in draw.meshes {
                if (part.vertices, part.indices) != bound {
                    device.cmd_bind_vertex_buffers(self.cmd, 0, &[part.vertices], &[0]);
                    device.cmd_bind_index_buffer(self.cmd, part.indices, 0, vk::IndexType::UINT32);
                    bound = (part.vertices, part.indices);
                }
                if part.texture_set != texture {
                    device.cmd_bind_descriptor_sets(
                        self.cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        pipelines.forward_layout,
                        1,
                        std::slice::from_ref(&part.texture_set),
                        &[],
                    );
                    texture = part.texture_set;
                }
                device.cmd_draw_indexed(
                    self.cmd,
                    part.index_count,
                    1,
                    part.first_index,
                    0,
                    part.record,
                );
            }
        }
    }

    fn full_rect(&self) -> vk::Rect2D {
        vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D {
                width: self.width,
                height: self.height,
            },
        }
    }

    /// Dispatch the bound compute pipeline over every 8x8 tile of the frame.
    unsafe fn dispatch_full_frame(&self, device: &ash::Device) {
        device.cmd_dispatch(self.cmd, self.width.div_ceil(8), self.height.div_ceil(8), 1);
    }

    /// Record the whole frame into the command buffer:
    ///
    /// 0. TLAS rebuild (only for a frame that traces rays) and the probe
    ///    texture clear,
    /// 1. GI probe compute (optional),
    /// 2. the forward render pass (or the raw ray-query probe instead), then
    ///    the glass pass when refraction is on,
    /// 3. the post pass (SSR + refraction, optional) and the tone map,
    /// 4. the overlay pass (only for a non-empty overlay),
    /// 5. the hand-off to `target`.
    ///
    /// Only legal while no submission is pending ([`FrameGpu::wait`]).
    pub fn record(
        &self,
        device: &ash::Device,
        pipelines: &Pipelines,
        draw: &FrameDraw<'_>,
        options: &RenderOptions,
        target: FrameTarget,
    ) -> Result<(), RenderError> {
        let cmd = self.cmd;
        // The pool is RESET_COMMAND_BUFFER, so `begin` resets the buffer.
        begin(cmd, device)?;
        unsafe {
            // 0a. The scene's TLAS, with the barrier that makes it visible to
            //     the compute-shader ray queries below.
            if let Some(tlas) = draw.tlas {
                tlas.record_build(device, cmd, draw.tlas_instances);
            }
            // With shadows the fragment shaders trace a ray to every light;
            // without, these are the pipelines a frame has always used.
            let shadows = draw.shadows && draw.tlas.is_some();
            let (opaque_pipeline, glass_pipeline, mesh_pipeline) = if shadows {
                (
                    pipelines.graphics_shadow,
                    pipelines.graphics_glass_shadow,
                    pipelines.graphics_mesh_shadow,
                )
            } else {
                (
                    pipelines.graphics,
                    pipelines.graphics_glass,
                    pipelines.graphics_mesh,
                )
            };

            // 0b. Deterministically zero the probe texture, so a skipped GI
            //     dispatch reads zeros, not whatever the last frame left.
            if let Some((gi_img, _)) = &self.gi_tex {
                device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
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
                device.cmd_clear_color_image(
                    cmd,
                    gi_img.handle,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &vk::ClearColorValue { float32: [0.0; 4] },
                    &[color_range()],
                );
            }

            // 0c. The `out` image is written as a storage image (by the tone
            //     map or by the ray-query probe), so move it out of UNDEFINED
            //     first (its previous contents are not needed).
            let mut initial_barriers = Vec::with_capacity(3);
            if let Some(post) = &self.post {
                // The post pass writes this as a storage image.
                initial_barriers.push(
                    vk::ImageMemoryBarrier::default()
                        .image(post.out.handle)
                        .old_layout(vk::ImageLayout::UNDEFINED)
                        .new_layout(vk::ImageLayout::GENERAL)
                        .src_access_mask(vk::AccessFlags::empty())
                        .dst_access_mask(vk::AccessFlags::SHADER_WRITE)
                        .subresource_range(color_range()),
                );
            }
            initial_barriers.extend([
                vk::ImageMemoryBarrier::default()
                    .image(self.out.handle)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .src_access_mask(vk::AccessFlags::empty())
                    .dst_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .subresource_range(color_range()),
                // The GI dummy (bound when GI is off) is never actually read;
                // it only has to be in the layout its descriptor names.
                vk::ImageMemoryBarrier::default()
                    .image(self.dummy_tex.0.handle)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .src_access_mask(vk::AccessFlags::empty())
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)
                    .subresource_range(color_range()),
            ]);
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &initial_barriers,
            );

            // 1. GI probe pass: probe texture TRANSFER_DST -> GENERAL, dispatch.
            if let (Some((gi_img, _)), true) = (&self.gi_tex, draw.gi_dispatch) {
                device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
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
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    pipelines.gi_pipeline,
                );
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    pipelines.gi_layout,
                    0,
                    std::slice::from_ref(&self.gi_set),
                    &[],
                );
                device.cmd_dispatch(
                    cmd,
                    self.gi_extent.0.div_ceil(8),
                    self.gi_extent.1.div_ceil(8),
                    1,
                );
                // Probe texture GENERAL -> SHADER_READ_ONLY for the forward pass.
                device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                    vk::DependencyFlags::empty(),
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
                // the zeros written in step 0b.
                device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                    vk::DependencyFlags::empty(),
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
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    pipelines.rq_pipeline,
                );
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    pipelines.rq_layout,
                    0,
                    std::slice::from_ref(&self.rq_set),
                    &[],
                );
                self.dispatch_full_frame(device);
                let overlaid = self.record_overlay(device, pipelines, draw);
                self.record_target(device, target, overlaid);
                return end(cmd, device);
            }

            // 2b. Forward render pass (HDR + material + depth).
            device.cmd_begin_render_pass(
                cmd,
                &vk::RenderPassBeginInfo::default()
                    .render_pass(pipelines.render_pass)
                    .framebuffer(self.framebuffer)
                    .render_area(self.full_rect())
                    .clear_values(&[
                        // Black, and alpha 0: every surface writes alpha 1, so
                        // the post pass can tell "nothing here" (the refracted
                        // view of glass then shows the environment) from a
                        // black surface. Nothing else reads this alpha.
                        vk::ClearValue {
                            color: vk::ClearColorValue {
                                float32: [0.0, 0.0, 0.0, 0.0],
                            },
                        },
                        // Material channel: zero = "no surface / not metallic".
                        vk::ClearValue {
                            color: vk::ClearColorValue {
                                float32: [0.0, 0.0, 0.0, 0.0],
                            },
                        },
                        // Specular colour: nothing there reflects anything.
                        vk::ClearValue {
                            color: vk::ClearColorValue {
                                float32: [0.0, 0.0, 0.0, 0.0],
                            },
                        },
                        // Reversed depth: 0 is "infinitely far".
                        vk::ClearValue {
                            depth_stencil: vk::ClearDepthStencilValue {
                                depth: 0.0,
                                stencil: 0,
                            },
                        },
                    ]),
                vk::SubpassContents::INLINE,
            );
            if !options.mutate_skip_forward_draw {
                self.draw_scene(device, pipelines, opaque_pipeline, draw);
                self.draw_meshes(device, pipelines, mesh_pipeline, draw);
                if draw.sky {
                    // The background, last: one triangle over the frame that
                    // only lands where the depth buffer is still clear.
                    // (Viewport, scissor and descriptor set 0 are as
                    // `draw_scene` left them.)
                    device.cmd_bind_pipeline(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        pipelines.graphics_sky,
                    );
                    device.cmd_draw(cmd, 3, 1, 0, 0);
                }
            }
            device.cmd_end_render_pass(cmd);

            // 2c. Glass pass (refraction only): transmissive instances into
            //     their own colour target, depth-tested against the opaque
            //     depth, so the opaque scene behind them stays in the HDR
            //     target for the post pass to refract.
            if let Some(glass_fb) = self.post.as_ref().and_then(|p| p.glass_framebuffer) {
                device.cmd_begin_render_pass(
                    cmd,
                    &vk::RenderPassBeginInfo::default()
                        .render_pass(pipelines.render_pass_glass)
                        .framebuffer(glass_fb)
                        .render_area(self.full_rect())
                        .clear_values(&[
                            vk::ClearValue {
                                color: vk::ClearColorValue {
                                    float32: [0.0, 0.0, 0.0, 0.0],
                                },
                            },
                            // Unused: the glass pass LOADs the opaque pass's
                            // material buffer and depth.
                            vk::ClearValue {
                                color: vk::ClearColorValue {
                                    float32: [0.0, 0.0, 0.0, 0.0],
                                },
                            },
                            vk::ClearValue {
                                depth_stencil: vk::ClearDepthStencilValue {
                                    depth: 0.0,
                                    stencil: 0,
                                },
                            },
                        ]),
                    vk::SubpassContents::INLINE,
                );
                if !options.mutate_skip_forward_draw {
                    self.draw_scene(device, pipelines, glass_pipeline, draw);
                }
                device.cmd_end_render_pass(cmd);
            }

            // 3. Tone map. The render pass already transitioned the HDR
            //    attachment to SHADER_READ_ONLY_OPTIMAL as its final layout,
            //    and the end-of-render-pass dependency makes the colour
            //    writes visible to subsequent shader reads.
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)],
                &[],
                &[],
            );
            // 3a. Post pass (SSR + refraction): skipped entirely — nothing is
            //     recorded — unless the preset enabled a post feature.
            if let Some(post) = &self.post {
                device.cmd_bind_pipeline(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    if draw.glass_rays && pipelines.post_pipeline_rq != vk::Pipeline::null() {
                        pipelines.post_pipeline_rq
                    } else {
                        pipelines.post_pipeline
                    },
                );
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    pipelines.post_layout,
                    0,
                    std::slice::from_ref(&post.set),
                    &[],
                );
                self.dispatch_full_frame(device);
                // Post output GENERAL -> SHADER_READ_ONLY for the tone map.
                device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[vk::ImageMemoryBarrier::default()
                        .image(post.out.handle)
                        .old_layout(vk::ImageLayout::GENERAL)
                        .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                        .dst_access_mask(vk::AccessFlags::SHADER_READ)
                        .subresource_range(color_range())],
                );
            }
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipelines.tone_pipeline);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                pipelines.tone_layout,
                0,
                std::slice::from_ref(&self.tone_set),
                &[],
            );
            self.dispatch_full_frame(device);

            // 4. The overlay, on top of the finished image.
            let overlaid = self.record_overlay(device, pipelines, draw);

            // 5. Hand the finished image to its target.
            self.record_target(device, target, overlaid);
        }
        end(cmd, device)
    }

    /// Step 4 of [`FrameGpu::record`]: blend the overlay's rectangles onto
    /// `out`. Returns whether the pass was recorded — it is skipped entirely
    /// for an empty overlay — in which case it has also left `out` in
    /// TRANSFER_SRC_OPTIMAL, synchronised for the copy that follows.
    unsafe fn record_overlay(
        &self,
        device: &ash::Device,
        pipelines: &Pipelines,
        draw: &FrameDraw<'_>,
    ) -> bool {
        let Some((instances, count)) = draw.overlay.filter(|(_, count)| *count > 0) else {
            return false;
        };
        let cmd = self.cmd;
        device.cmd_begin_render_pass(
            cmd,
            &vk::RenderPassBeginInfo::default()
                .render_pass(pipelines.render_pass_overlay)
                .framebuffer(self.overlay_framebuffer)
                .render_area(self.full_rect()),
            vk::SubpassContents::INLINE,
        );
        device.cmd_bind_pipeline(
            cmd,
            vk::PipelineBindPoint::GRAPHICS,
            pipelines.overlay_pipeline,
        );
        device.cmd_set_viewport(
            cmd,
            0,
            &[vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: self.width as f32,
                height: self.height as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            }],
        );
        device.cmd_set_scissor(cmd, 0, &[self.full_rect()]);
        device.cmd_bind_vertex_buffers(cmd, 0, &[instances], &[0]);
        // Six vertices (two triangles) per rectangle instance, in list order:
        // later rectangles blend over earlier ones.
        device.cmd_draw(cmd, 6, count, 0, 0);
        device.cmd_end_render_pass(cmd);
        true
    }

    /// Step 5 of [`FrameGpu::record`]: `out` to TRANSFER_SRC (unless the
    /// overlay pass already left it there: `overlaid`), then either the
    /// readback copy or the blit into the swapchain image.
    unsafe fn record_target(&self, device: &ash::Device, target: FrameTarget, overlaid: bool) {
        let cmd = self.cmd;
        if !overlaid {
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
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
        }
        let extent = vk::Extent3D {
            width: self.width,
            height: self.height,
            depth: 1,
        };
        match target {
            FrameTarget::Readback => device.cmd_copy_image_to_buffer(
                cmd,
                self.out.handle,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                self.readback.handle,
                &[vk::BufferImageCopy::default()
                    .buffer_row_length(0)
                    .buffer_image_height(0)
                    .image_subresource(color_layers())
                    .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                    .image_extent(extent)],
            ),
            FrameTarget::Present(image) => {
                // The acquired image's old contents are irrelevant, so
                // UNDEFINED is the right source layout whatever it was left
                // in. The source stage is TRANSFER — the stage the submission
                // waits for the acquire semaphore at — so the transition
                // cannot run before the image has actually been acquired.
                device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[vk::ImageMemoryBarrier::default()
                        .image(image)
                        .old_layout(vk::ImageLayout::UNDEFINED)
                        .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                        .src_access_mask(vk::AccessFlags::empty())
                        .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                        .subresource_range(color_range())],
                );
                let corners = [
                    vk::Offset3D { x: 0, y: 0, z: 0 },
                    vk::Offset3D {
                        x: self.width as i32,
                        y: self.height as i32,
                        z: 1,
                    },
                ];
                // A blit (not a copy): it converts between the offscreen
                // `rgba8unorm` and whatever 8/10-bit UNORM format the surface
                // offers, channel by channel, so BGRA surfaces show the right
                // colours.
                device.cmd_blit_image(
                    cmd,
                    self.out.handle,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[vk::ImageBlit {
                        src_subresource: color_layers(),
                        src_offsets: corners,
                        dst_subresource: color_layers(),
                        dst_offsets: corners,
                    }],
                    vk::Filter::NEAREST,
                );
                // A swapchain image may only be presented from PRESENT_SRC_KHR.
                device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[vk::ImageMemoryBarrier::default()
                        .image(image)
                        .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                        .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                        .dst_access_mask(vk::AccessFlags::empty())
                        .subresource_range(color_range())],
                );
            }
        }
    }

    /// The pixels of the last frame recorded with [`FrameTarget::Readback`]
    /// (RGBA8, row-major, top row first). The submission must have completed.
    pub fn read_pixels(&self) -> Result<Vec<u8>, RenderError> {
        let mut pixels = vec![0u8; self.width as usize * self.height as usize * 4];
        self.readback.read(&mut pixels)?;
        Ok(pixels)
    }
}

impl Drop for FrameGpu {
    fn drop(&mut self) {
        // The device is still alive here (`RenderContext` drops its cached
        // frame before the device) and no submission is pending. The
        // `Image`/`Buffer` fields free their dedicated gpu-allocator
        // allocations via their own Drop after this body (hard constraint:
        // gpu-allocator does not own the dedicated VkImage/VkBuffer).
        let d = &self.device;
        unsafe {
            d.destroy_framebuffer(self.framebuffer, None);
            d.destroy_framebuffer(self.overlay_framebuffer, None);
            d.destroy_image_view(self.hdr_view, None);
            d.destroy_image_view(self.depth_view, None);
            d.destroy_image_view(self.material_view, None);
            d.destroy_image_view(self.specular_view, None);
            d.destroy_image_view(self.out_view, None);
            if let Some(post) = &self.post {
                if let Some(fb) = post.glass_framebuffer {
                    d.destroy_framebuffer(fb, None);
                }
                if let Some(view) = post.glass_view {
                    d.destroy_image_view(view, None);
                }
                d.destroy_image_view(post.out_view, None);
            }
            if let Some((_, view)) = &self.gi_tex {
                d.destroy_image_view(*view, None);
            }
            d.destroy_image_view(self.dummy_tex.1, None);
            d.destroy_fence(self.fence, None);
            d.destroy_descriptor_pool(self.pool, None);
            // Allocated per frame object, so it must be returned to the pool
            // (the pool outlives every frame).
            d.free_command_buffers(self.cmd_pool, &[self.cmd]);
        }
    }
}
