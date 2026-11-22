//! Phase-0 spike 0.3: prove ash + naga-compiled WGSL can run a ray-query
//! compute pass with correct pixel readback. Throwaway; not a workspace member.
//!
//! Pipeline:
//! 1. Headless ash instance + device with the ray-query / acceleration-structure
//!    / deferred-host-operations chain, plus buffer device address (required for
//!    device-side BLAS builds and for feeding the AS into the shader).
//! 2. gpu-allocator owns all device memory.
//! 3. A BLAS is built device-side from a 3-vertex / 1-triangle mesh.
//! 4. A WGSL compute shader (compiled to SPIR-V by naga in build.rs) traces one
//!    ray per pixel and writes hit colour or miss colour.
//! 5. Read back a 64x64 image and assert centre = hit, corners = miss.
//!
//! `--empty-scene` is the mutation check: the BLAS is built with zero
//! primitives, so every ray misses and the pixel assertions must FAIL.

use std::ffi::{c_char, CStr};
use std::sync::atomic::{AtomicU32, Ordering};

use ash::vk;
use gpu_allocator::vulkan::{
    Allocation, AllocationCreateDesc, AllocationScheme, Allocator, AllocatorCreateDesc,
};

const WIDTH: u32 = 64;
const HEIGHT: u32 = 64;

/// Hit colour (opaque red) written where a ray hits the triangle.
const HIT: [u8; 4] = [255, 0, 0, 255];
/// Miss colour (opaque teal) written where a ray misses.
const MISS: [u8; 4] = [0, 128, 255, 255];
/// Per-channel tolerance for the readback comparison.
const TOL: i32 = 8;
/// Number of pixels the triangle is expected to cover, computed offline from
/// the camera/mesh constants below. The assertion is exact: a run that does not
/// produce exactly this many hit pixels has a broken trace, not a lucky one.
const EXPECTED_HITS: usize = 722;
/// Validation layer name (NUL-terminated for the Vulkan API).
const VALIDATION_LAYER: &std::ffi::CStr = c"VK_LAYER_KHRONOS_validation";

/// Uniform buffer layout, mirrors `Params` in the WGSL (std140-ish, 64 bytes).
#[repr(C)]
#[derive(Clone, Copy)]
struct Params {
    hit_color: [f32; 4],
    miss_color: [f32; 4],
    camera_origin: [f32; 4],
    image_size: [f32; 4],
    tan_half_fov: f32,
    _pad: [f32; 3],
}

/// One vertex: vec3 position (matches the BLAS vertex format RGB32_SFLOAT).
#[repr(C)]
#[derive(Clone, Copy)]
struct Vertex {
    pos: [f32; 3],
}

// A large triangle centred on the camera axis, in front of the camera at z=0
// looking down -Z. Its screen projection covers the centre of the image but
// not the corners, so centre pixels hit and corner pixels miss.
const VERTICES: [Vertex; 3] = [
    Vertex { pos: [-0.5, -0.5, -2.0] },
    Vertex { pos: [0.5, -0.5, -2.0] },
    Vertex { pos: [0.0, 0.5, -2.0] },
];
const INDICES: [u32; 3] = [0, 1, 2];

// Camera: origin at origin, 45 deg vertical FOV (tan(22.5) ~= 0.4142).
const CAMERA_ORIGIN: [f32; 3] = [0.0, 0.0, 0.0];
const TAN_HALF_FOV: f32 = 0.414_213_56;

// ---- validation-layer bookkeeping -------------------------------------------
static VALIDATION_MESSAGES: AtomicU32 = AtomicU32::new(0);

unsafe extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _types: vk::DebugUtilsMessageTypeFlagsEXT,
    p_data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user: *mut core::ffi::c_void,
) -> vk::Bool32 {
    if severity == vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE {
        // Info/verbose chatter is not a failure; only warning+ counts.
        return vk::FALSE;
    }
    let msg = if p_data.is_null() {
        String::from("<null callback data>")
    } else {
        let d = unsafe { &*p_data };
        let name = if d.p_message_id_name.is_null() {
            String::new()
        } else {
            unsafe { CStr::from_ptr(d.p_message_id_name) }
                .to_string_lossy()
                .into_owned()
        };
        let msg = if d.p_message.is_null() {
            String::new()
        } else {
            unsafe { CStr::from_ptr(d.p_message) }.to_string_lossy().into_owned()
        };
        format!("[{name}] {msg}")
    };
    VALIDATION_MESSAGES.fetch_add(1, Ordering::SeqCst);
    eprintln!("VALIDATION: {msg}");
    vk::FALSE
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let empty_scene = args.iter().any(|a| a == "--empty-scene");
    let validation = !args.iter().any(|a| a == "--no-validation");
    let dump_path = args
        .iter()
        .position(|a| a == "--dump")
        .and_then(|i| args.get(i + 1))
        .cloned();
    match run(empty_scene, validation, dump_path.as_deref()) {
        Ok(()) => {
            println!("PASS: ray-query compute produced the expected pixels");
            std::process::exit(0);
        }
        Err(err) => {
            println!("FAIL: {err}");
            std::process::exit(1);
        }
    }
}

// ---- core -------------------------------------------------------------------

fn run(empty_scene: bool, validation: bool, dump_path: Option<&str>) -> Result<(), String> {
    let entry = unsafe { ash::Entry::load() }.map_err(|e| format!("vulkan loader load failed: {e}"))?;

    // ---- instance -----------------------------------------------------------
    let instance_exts: Vec<*const c_char> = vec![
        vk::KHR_GET_PHYSICAL_DEVICE_PROPERTIES2_NAME.as_ptr(),
        vk::EXT_DEBUG_UTILS_NAME.as_ptr(),
    ];
    let layers: Vec<*const c_char> = if validation {
        vec![VALIDATION_LAYER.as_ptr()]
    } else {
        Vec::new()
    };
    let app_info = vk::ApplicationInfo::default()
        .application_name(c"rq-triangle")
        .application_version(vk::make_api_version(0, 1, 0, 0))
        .engine_name(c"rq-triangle")
        .engine_version(vk::make_api_version(0, 1, 0, 0))
        .api_version(vk::make_api_version(0, 1, 3, 0));

    let mut messenger_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
        .message_severity(
            vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
        )
        .message_type(vk::DebugUtilsMessageTypeFlagsEXT::GENERAL | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION)
        .pfn_user_callback(Some(debug_callback));

    if validation {
        let available = unsafe { entry.enumerate_instance_layer_properties() }
            .map_err(|e| format!("enumerate_instance_layer_properties: {e}"))?;
        let name = VALIDATION_LAYER.to_string_lossy();
        if !available
            .iter()
            .any(|l| carray_to_string(&l.layer_name) == *name)
        {
            return Err(format!(
                "validation layer {name} not found; set VK_LAYER_PATH or pass --no-validation"
            ));
        }
    }

    let create_info = vk::InstanceCreateInfo::default()
        .application_info(&app_info)
        .enabled_layer_names(&layers)
        .enabled_extension_names(&instance_exts)
        .push_next(&mut messenger_info);
    let instance = unsafe { entry.create_instance(&create_info, None) }
        .map_err(|e| format!("create_instance failed: {e}"))?;
    let debug_utils = ash::ext::debug_utils::Instance::new(&entry, &instance);
    let debug_messenger = unsafe {
        debug_utils
            .create_debug_utils_messenger(&messenger_info, None)
            .map_err(|e| format!("create_debug_utils_messenger failed: {e}"))?
    };

    // ---- physical device (prefer RADV GPU) ----------------------------------
    let pdevices = unsafe { instance.enumerate_physical_devices() }
        .map_err(|e| format!("enumerate_physical_devices: {e}"))?;
    if pdevices.is_empty() {
        return Err("no physical devices".to_string());
    }
    let mut pd = pdevices[0];
    for &p in &pdevices {
        let props = unsafe { instance.get_physical_device_properties(p) };
        let name = carray_to_string(&props.device_name);
        if name.contains("RADV") || name.contains("Radeon") {
            pd = p;
            break;
        }
    }
    let pd_name = {
        let props = unsafe { instance.get_physical_device_properties(pd) };
        carray_to_string(&props.device_name)
    };
    println!("device: {pd_name}");

    // ---- device extensions + features ---------------------------------------
    let required_device_exts: [&CStr; 5] = [
        vk::KHR_RAY_QUERY_NAME.as_ref(),
        vk::KHR_ACCELERATION_STRUCTURE_NAME.as_ref(),
        vk::KHR_DEFERRED_HOST_OPERATIONS_NAME.as_ref(),
        vk::KHR_BUFFER_DEVICE_ADDRESS_NAME.as_ref(),
        vk::KHR_GET_MEMORY_REQUIREMENTS2_NAME.as_ref(),
    ];
    let available: Vec<String> = unsafe {
        instance
            .enumerate_device_extension_properties(pd)
            .map_err(|e| format!("enumerate_device_extension_properties: {e}"))?
    }
    .iter()
    .map(|e| carray_to_string(&e.extension_name))
    .collect();
    for req in &required_device_exts {
        let s = req.to_string_lossy();
        if !available.iter().any(|n| n == &*s) {
            return Err(format!("device extension {s} not available on {pd_name}"));
        }
    }
    let ext_names: Vec<*const c_char> = required_device_exts.iter().map(|c| c.as_ptr()).collect();

    // Query support first (separate chain so the borrows end before we read).
    {
        let mut probe_rq = vk::PhysicalDeviceRayQueryFeaturesKHR::default();
        let mut probe_as = vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default();
        let mut probe_bda = vk::PhysicalDeviceBufferDeviceAddressFeatures::default();
        let mut probe = vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut probe_rq)
            .push_next(&mut probe_as)
            .push_next(&mut probe_bda);
        unsafe { instance.get_physical_device_features2(pd, &mut probe) };
        if probe_rq.ray_query == 0
            || probe_as.acceleration_structure == 0
            || probe_bda.buffer_device_address == 0
        {
            return Err(
                "required ray-query / acceleration-structure / buffer-device-address features                  not supported on {pd_name}"
                    .to_string(),
            );
        }
    }

    let mut rq_features = vk::PhysicalDeviceRayQueryFeaturesKHR::default().ray_query(true);
    let mut as_features =
        vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default().acceleration_structure(true);
    let mut bda_features = vk::PhysicalDeviceBufferDeviceAddressFeatures::default()
        .buffer_device_address(true);
    let mut features2 = vk::PhysicalDeviceFeatures2::default()
        .push_next(&mut rq_features)
        .push_next(&mut as_features)
        .push_next(&mut bda_features);

    // ---- compute queue family -----------------------------------------------
    let queue_props = unsafe { instance.get_physical_device_queue_family_properties(pd) };
    let queue_family = queue_props
        .iter()
        .position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
        .ok_or("no compute-capable queue family")? as u32;

    let priorities = [1.0f32];
    let queue_create = [vk::DeviceQueueCreateInfo::default()
        .queue_family_index(queue_family)
        .queue_priorities(&priorities)];
    let device_create = vk::DeviceCreateInfo::default()
        .queue_create_infos(&queue_create)
        .enabled_extension_names(&ext_names)
        .push_next(&mut features2);
    let device = unsafe { instance.create_device(pd, &device_create, None) }
        .map_err(|e| format!("create_device: {e}"))?;
    let queue = unsafe { device.get_device_queue(queue_family, 0) };

    let accel = ash::khr::acceleration_structure::Device::new(&instance, &device);

    // ---- gpu-allocator ------------------------------------------------------
    let mut allocator = Allocator::new(&AllocatorCreateDesc {
        instance: instance.clone(),
        device: device.clone(),
        physical_device: pd,
        debug_settings: Default::default(),
        buffer_device_address: true,
        allocation_sizes: Default::default(),
    })
    .map_err(|e| format!("gpu-allocator init: {e}"))?;

    // ---- geometry buffers ----------------------------------------------------
    let vertex_bytes: Vec<u8> = bytemuck_vertices();
    let index_bytes: Vec<u8> = bytemuck_indices();
    let prim_count = if empty_scene { 0 } else { INDICES.len() as u32 };

    let vertex_buf = create_buffer(
        &device,
        &mut allocator,
        vertex_bytes.len() as u64,
        vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        gpu_allocator::MemoryLocation::CpuToGpu,
        "vertices",
    )?;
    write_cpu(&device, &vertex_buf, &vertex_bytes)?;

    let index_buf = create_buffer(
        &device,
        &mut allocator,
        index_bytes.len() as u64,
        vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        gpu_allocator::MemoryLocation::CpuToGpu,
        "indices",
    )?;
    write_cpu(&device, &index_buf, &index_bytes)?;

    let vertex_addr = unsafe {
        device.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(vertex_buf.buffer))
    };
    let index_addr = unsafe {
        device.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(index_buf.buffer))
    };

    // ---- BLAS ----------------------------------------------------------------
    let geometry = vk::AccelerationStructureGeometryKHR::default()
        .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
        .flags(vk::GeometryFlagsKHR::empty())
        .geometry(vk::AccelerationStructureGeometryDataKHR {
            triangles: vk::AccelerationStructureGeometryTrianglesDataKHR::default()
                .vertex_format(vk::Format::R32G32B32_SFLOAT)
                .vertex_data(vk::DeviceOrHostAddressConstKHR {
                    device_address: vertex_addr,
                })
                .vertex_stride(std::mem::size_of::<Vertex>() as u64)
                .max_vertex(VERTICES.len() as u32)
                .index_type(vk::IndexType::UINT32)
                .index_data(vk::DeviceOrHostAddressConstKHR {
                    device_address: index_addr,
                }),
        });

    // A scratch buffer is required even for an empty build; keep 256 bytes.
    let mut blas_build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
        .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
        .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
        .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
        .geometries(std::slice::from_ref(&geometry));

    let sizes = {
        let mut s = vk::AccelerationStructureBuildSizesInfoKHR::default();
        unsafe {
            accel.get_acceleration_structure_build_sizes(
                vk::AccelerationStructureBuildTypeKHR::DEVICE,
                &blas_build_info,
                &[prim_count],
                &mut s,
            )
        };
        s
    };
    println!(
        "BLAS sizes: as={}B build_scratch={}B",
        sizes.acceleration_structure_size, sizes.build_scratch_size
    );

    let blas_buf = create_buffer(
        &device,
        &mut allocator,
        sizes.acceleration_structure_size,
        vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        gpu_allocator::MemoryLocation::GpuOnly,
        "blas",
    )?;
    let scratch_buf = create_buffer(
        &device,
        &mut allocator,
        sizes.build_scratch_size.max(256),
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        gpu_allocator::MemoryLocation::GpuOnly,
        "blas-scratch",
    )?;

    let blas = unsafe {
        accel
            .create_acceleration_structure(
                &vk::AccelerationStructureCreateInfoKHR::default()
                    .buffer(blas_buf.buffer)
                    .size(sizes.acceleration_structure_size)
                    // GENERIC: a descriptor-bound acceleration structure must be
                    // TOP_LEVEL or GENERIC (VUID-VkWriteDescriptorSetAccelerationStructureKHR-pAccelerationStructures-03579);
                    // the build below still builds it as a bottom level.
                    .ty(vk::AccelerationStructureTypeKHR::GENERIC),
                None,
            )
            .map_err(|e| format!("create_acceleration_structure: {e}"))?
    };

    blas_build_info.dst_acceleration_structure = blas;
    blas_build_info.scratch_data = vk::DeviceOrHostAddressKHR {
        device_address: unsafe {
            device.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(scratch_buf.buffer))
        },
    };
    let build_range = [vk::AccelerationStructureBuildRangeInfoKHR {
        primitive_count: prim_count,
        primitive_offset: 0,
        first_vertex: 0,
        transform_offset: 0,
    }];

    // ---- params uniform ------------------------------------------------------
    let params = Params {
        hit_color: [
            HIT[0] as f32 / 255.0,
            HIT[1] as f32 / 255.0,
            HIT[2] as f32 / 255.0,
            1.0,
        ],
        miss_color: [
            MISS[0] as f32 / 255.0,
            MISS[1] as f32 / 255.0,
            MISS[2] as f32 / 255.0,
            1.0,
        ],
        camera_origin: [CAMERA_ORIGIN[0], CAMERA_ORIGIN[1], CAMERA_ORIGIN[2], 0.0],
        image_size: [WIDTH as f32, HEIGHT as f32, 0.0, 0.0],
        tan_half_fov: TAN_HALF_FOV,
        _pad: [0.0; 3],
    };
    let params_buf = create_buffer(
        &device,
        &mut allocator,
        std::mem::size_of::<Params>() as u64,
        vk::BufferUsageFlags::UNIFORM_BUFFER,
        gpu_allocator::MemoryLocation::CpuToGpu,
        "params",
    )?;
    write_cpu(
        &device,
        &params_buf,
        unsafe {
            std::slice::from_raw_parts(
                &params as *const Params as *const u8,
                std::mem::size_of::<Params>(),
            )
        },
    )?;

    // ---- output image + readback buffer --------------------------------------
    let image = unsafe {
        device
            .create_image(
                &vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(vk::Format::R8G8B8A8_UNORM)
                    .extent(vk::Extent3D {
                        width: WIDTH,
                        height: HEIGHT,
                        depth: 1,
                    })
                    .mip_levels(1)
                    .array_layers(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::LINEAR)
                    // UNDEFINED: initialLayout must be UNDEFINED/PREINITIALIZED
                    // (VUID-VkImageCreateInfo-initialLayout-00993). The layout is
                    // transitioned explicitly with pipeline barriers below.
                    .initial_layout(vk::ImageLayout::UNDEFINED)
                    .usage(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC),
                None,
            )
            .map_err(|e| format!("create_image: {e}"))?
    };
    let image_mem = allocate_for_image(&device, &mut allocator, image, "out-image")?;
    unsafe { device.bind_image_memory(image, image_mem.memory(), image_mem.offset()) }
        .map_err(|e| format!("bind_image_memory: {e}"))?;
    let image_view = unsafe {
        device
            .create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(vk::Format::R8G8B8A8_UNORM)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    }),
                None,
            )
            .map_err(|e| format!("create_image_view: {e}"))?
    };

    let readback_size = (WIDTH * HEIGHT * 4) as u64;
    let readback_buf = create_buffer(
        &device,
        &mut allocator,
        readback_size,
        vk::BufferUsageFlags::TRANSFER_DST,
        gpu_allocator::MemoryLocation::GpuToCpu,
        "readback",
    )?;

    // ---- descriptors + pipeline ---------------------------------------------
    let set_layout = unsafe {
        device
            .create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&[
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(0)
                        .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE),
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(1)
                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE),
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(2)
                        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE),
                ]),
                None,
            )
            .map_err(|e| format!("create_descriptor_set_layout: {e}"))?
    };
    let pipeline_layout = unsafe {
        device
            .create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().set_layouts(std::slice::from_ref(&set_layout)),
                None,
            )
            .map_err(|e| format!("create_pipeline_layout: {e}"))?
    };

    let spv = include_bytes!(concat!(env!("OUT_DIR"), "/rq_triangle.spv"));
    let shader_module = unsafe {
        device
            .create_shader_module(
                &vk::ShaderModuleCreateInfo::default().code(bytemuck_spv_words(spv)),
                None,
            )
            .map_err(|e| format!("create_shader_module: {e}"))?
    };
    let pipeline = unsafe {
        device
            .create_compute_pipelines(
                vk::PipelineCache::null(),
                &[vk::ComputePipelineCreateInfo::default()
                    .stage(
                        vk::PipelineShaderStageCreateInfo::default()
                            .stage(vk::ShaderStageFlags::COMPUTE)
                            .module(shader_module)
                            .name(c"main"),
                    )
                    .layout(pipeline_layout)],
                None,
            )
            .map_err(|(_, e)| format!("create_compute_pipelines: {e}"))?
            .remove(0)
    };

    let pool = unsafe {
        device
            .create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(1)
                    .pool_sizes(&[
                        vk::DescriptorPoolSize {
                            ty: vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
                            descriptor_count: 1,
                        },
                        vk::DescriptorPoolSize {
                            ty: vk::DescriptorType::STORAGE_IMAGE,
                            descriptor_count: 1,
                        },
                        vk::DescriptorPoolSize {
                            ty: vk::DescriptorType::UNIFORM_BUFFER,
                            descriptor_count: 1,
                        },
                    ]),
                None,
            )
            .map_err(|e| format!("create_descriptor_pool: {e}"))?
    };
    let set = unsafe {
        device
            .allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(pool)
                .set_layouts(std::slice::from_ref(&set_layout)))
            .map_err(|e| format!("allocate_descriptor_sets: {e}"))?
            .remove(0)
    };

    let mut as_write = vk::WriteDescriptorSetAccelerationStructureKHR::default()
        .acceleration_structures(std::slice::from_ref(&blas));
    let image_info = [vk::DescriptorImageInfo::default()
        .image_view(image_view)
        .image_layout(vk::ImageLayout::GENERAL)];
    let buffer_info = [vk::DescriptorBufferInfo::default()
        .buffer(params_buf.buffer)
        .offset(0)
        .range(std::mem::size_of::<Params>() as u64)];
    unsafe {
        device.update_descriptor_sets(
            &[
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                    .descriptor_count(1)
                    .push_next(&mut as_write)
                    .dst_array_element(0),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(&image_info),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(2)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(&buffer_info),
            ],
            &[],
        );
    }

    // ---- command buffer ------------------------------------------------------
    let cmd_pool = unsafe {
        device
            .create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                    .queue_family_index(queue_family),
                None,
            )
            .map_err(|e| format!("create_command_pool: {e}"))?
    };
    let cmd = unsafe {
        device
            .allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(cmd_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
            .map_err(|e| format!("allocate_command_buffers: {e}"))?
            .remove(0)
    };
    let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }
        .map_err(|e| format!("create_fence: {e}"))?;

    unsafe {
        device
            .begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .map_err(|e| format!("begin_command_buffer: {e}"))?;

        // 1. Build the BLAS device-side.
        accel.cmd_build_acceleration_structures(cmd, &[blas_build_info], &[&build_range]);
        device.end_command_buffer(cmd).map_err(|e| format!("end_command_buffer: {e}"))?;
    }
    // Submission 1: BLAS build only, then wait. This fully separates the build
    // from the trace so no barrier subtlety can leak an incomplete BVH into the
    // ray queries.
    let build_submit = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd));
    unsafe { device.queue_submit(queue, std::slice::from_ref(&build_submit), fence) }
        .map_err(|e| format!("queue_submit(build): {e}"))?;
    unsafe { device.wait_for_fences(std::slice::from_ref(&fence), true, u64::MAX) }
        .map_err(|e| format!("wait(build): {e}"))?;
    unsafe { device.reset_fences(std::slice::from_ref(&fence)) }
        .map_err(|e| format!("reset_fences: {e}"))?;

    unsafe {
        device
            .begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .map_err(|e| format!("begin_command_buffer(trace): {e}"))?;

        // 2. Transition the output image Undefined -> General so the compute
        //    storage-image writes are in a legal layout. The BLAS build needs no
        //    barrier here: it completed in the previous submission, and waiting on
        //    its fence is a full device-wide memory sync.
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::default(),
            &[],
            &[],
            &[vk::ImageMemoryBarrier::default()
                .image(image)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::GENERAL)
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::SHADER_WRITE)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })],
        );

        // 3. Trace.
        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
        device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            pipeline_layout,
            0,
            std::slice::from_ref(&set),
            &[],
        );
        device.cmd_dispatch(cmd, WIDTH.div_ceil(8), HEIGHT.div_ceil(8), 1);

        // 4. Transition the output image General -> TransferSrcOptimal for the
        //    readback copy, making the compute writes visible to the transfer.
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::default(),
            &[],
            &[],
            &[vk::ImageMemoryBarrier::default()
                .image(image)
                .old_layout(vk::ImageLayout::GENERAL)
                .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })],
        );
        device.cmd_copy_image_to_buffer(
            cmd,
            image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            readback_buf.buffer,
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
                    width: WIDTH,
                    height: HEIGHT,
                    depth: 1,
                })],
        );
        device.end_command_buffer(cmd).map_err(|e| format!("end_command_buffer: {e}"))?;
    }

    // ---- submit + wait -------------------------------------------------------
    let submit = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd));
    unsafe { device.queue_submit(queue, std::slice::from_ref(&submit), fence) }
        .map_err(|e| format!("queue_submit: {e}"))?;
    unsafe { device.wait_for_fences(std::slice::from_ref(&fence), true, u64::MAX) }
        .map_err(|e| format!("wait_for_fences timed out or failed: {e}"))?;

    // ---- readback + assertions ----------------------------------------------
    let pixels = read_gpu_to_cpu(&device, &readback_buf, readback_size as usize)?;

    let center = [(32, 32), (31, 31), (32, 31), (31, 32), (24, 32), (32, 24)];
    let corners = [(0, 0), (63, 0), (0, 63), (63, 63)];

    if let Some(path) = dump_path {
        write_ppm(path, &pixels, WIDTH, HEIGHT);
    }

    let mut failures: Vec<String> = Vec::new();

    // Whole-image count first: it catches a trace that hits the right pixels by
    // accident (e.g. a query left active, or a triangle covering too much).
    let hits = pixels
        .chunks(4)
        .filter(|p| pixel_close(&[p[0], p[1], p[2], p[3]], &HIT))
        .count();
    if hits != EXPECTED_HITS {
        failures.push(format!(
            "expected {EXPECTED_HITS} hit pixels, got {hits} / {}",
            (WIDTH * HEIGHT) as usize
        ));
    }

    for &(x, y) in &center {
        let got = pixel_at(&pixels, x, y);
        if !pixel_close(&got, &HIT) {
            failures.push(format!("centre pixel ({x},{y}) = {got:?}, expected ~{HIT:?}"));
        }
    }
    for &(x, y) in &corners {
        let got = pixel_at(&pixels, x, y);
        if !pixel_close(&got, &MISS) {
            failures.push(format!("corner pixel ({x},{y}) = {got:?}, expected ~{MISS:?}"));
        }
    }
    if failures.is_empty() {
        println!(
            "pixel assertions: {hits}/{total} pixels == HIT (expected {EXPECTED_HITS}), {} centre pixels == HIT, {} corner pixels == MISS (tol {TOL})",
            center.len(),
            corners.len(),
            total = (WIDTH * HEIGHT) as usize,
        );
    } else {
        let first = first_mismatch(&pixels, &center, &corners);
        if let Some(m) = first {
            println!("  first mismatch {m}");
        }
    }

    // ---- teardown ------------------------------------------------------------
    unsafe {
        device.device_wait_idle().ok();
        device.destroy_fence(fence, None);
        device.destroy_command_pool(cmd_pool, None);
        device.destroy_descriptor_pool(pool, None);
        device.destroy_shader_module(shader_module, None);
        device.destroy_pipeline(pipeline, None);
        device.destroy_pipeline_layout(pipeline_layout, None);
        device.destroy_descriptor_set_layout(set_layout, None);
        device.destroy_image_view(image_view, None);
        device.destroy_image(image, None);
        accel.destroy_acceleration_structure(blas, None);
        if let Err(e) = allocator.free(image_mem) {
            eprintln!("warning: free(out-image): {e}");
        }
        for (a, name) in [
            (vertex_buf, "vertices"),
            (index_buf, "indices"),
            (blas_buf, "blas"),
            (scratch_buf, "blas-scratch"),
            (params_buf, "params"),
            (readback_buf, "readback"),
        ] {
            destroy_buffer(&device, &mut allocator, a, name);
        }
        device.destroy_device(None);
        debug_utils.destroy_debug_utils_messenger(debug_messenger, None);
        instance.destroy_instance(None);
    }
    allocator.report_memory_leaks(log::Level::Warn);

    // ---- verdict --------------------------------------------------------------
    let validation_count = VALIDATION_MESSAGES.load(Ordering::SeqCst);
    println!("validation messages (warning+): {validation_count}");
    if !failures.is_empty() {
        return Err(format!(
            "{} pixel assertion(s) failed: {}",
            failures.len(),
            failures.join("; ")
        ));
    }
    if validation_count > 0 {
        return Err(format!("{validation_count} validation message(s) emitted; treating as failure"));
    }
    Ok(())
}

// ---- helpers -----------------------------------------------------------------

/// Write the RGBA readback buffer out as a binary PPM, for eyeballing.
fn write_ppm(path: &str, pixels: &[u8], width: u32, height: u32) {
    let mut out = format!("P6\n{width} {height}\n255\n").into_bytes();
    for chunk in pixels.chunks(4) {
        out.extend_from_slice(&chunk[..3]);
    }
    if let Err(e) = std::fs::write(path, out) {
        eprintln!("warning: could not write {path}: {e}");
    }
}

/// Locate the first centre/corner pixel that disagrees with the expectation.
fn first_mismatch(pixels: &[u8], center: &[(u32, u32)], corners: &[(u32, u32)]) -> Option<String> {
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let got = pixel_at(pixels, x, y);
            let want = if center.contains(&(x, y)) {
                &HIT
            } else if corners.contains(&(x, y)) {
                &MISS
            } else {
                continue;
            };
            if !pixel_close(&got, want) {
                return Some(format!("at ({x}, {y}): got {got:?}, expected {want:?}"));
            }
        }
    }
    None
}

fn carray_to_string(arr: &[c_char]) -> String {
    let bytes: Vec<u8> = arr
        .iter()
        .map(|&c| c as u8)
        .take_while(|&b| b != 0)
        .collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

fn bytemuck_vertices() -> Vec<u8> {
    unsafe {
        std::slice::from_raw_parts(
            VERTICES.as_ptr() as *const u8,
            VERTICES.len() * std::mem::size_of::<Vertex>(),
        )
    }
    .to_vec()
}

fn bytemuck_indices() -> Vec<u8> {
    unsafe {
        std::slice::from_raw_parts(
            INDICES.as_ptr() as *const u8,
            INDICES.len() * std::mem::size_of::<u32>(),
        )
    }
    .to_vec()
}

fn bytemuck_spv_words(bytes: &[u8]) -> &[u32] {
    assert_eq!(bytes.len() % 4, 0, "SPIR-V must be a whole number of words");
    unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const u32, bytes.len() / 4) }
}

#[allow(clippy::too_many_arguments)]
fn create_buffer(
    device: &ash::Device,
    allocator: &mut Allocator,
    size: u64,
    usage: vk::BufferUsageFlags,
    location: gpu_allocator::MemoryLocation,
    name: &str,
) -> Result<BufferAlloc, String> {
    let buffer = unsafe {
        device
            .create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(size)
                    .usage(usage)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )
            .map_err(|e| format!("create_buffer({name}): {e}"))?
    };
    let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
    let allocation = allocator
        .allocate(&AllocationCreateDesc {
            name,
            requirements,
            location,
            linear: true,
            allocation_scheme: AllocationScheme::DedicatedBuffer(buffer),
        })
        .map_err(|e| format!("allocate({name}): {e}"))?;
    unsafe { device.bind_buffer_memory(buffer, allocation.memory(), allocation.offset()) }
        .map_err(|e| format!("bind_buffer_memory({name}): {e}"))?;
    Ok(BufferAlloc { buffer, allocation })
}

struct BufferAlloc {
    buffer: vk::Buffer,
    allocation: Allocation,
}

fn allocate_for_image(
    device: &ash::Device,
    allocator: &mut Allocator,
    image: vk::Image,
    name: &str,
) -> Result<Allocation, String> {
    let requirements = unsafe { device.get_image_memory_requirements(image) };
    allocator
        .allocate(&AllocationCreateDesc {
            name,
            requirements,
            location: gpu_allocator::MemoryLocation::GpuOnly,
            linear: true,
            allocation_scheme: AllocationScheme::DedicatedImage(image),
        })
        .map_err(|e| format!("allocate({name}): {e}"))
}

/// Copy `data` into a CPU-visible allocation and flush.
/// Destroy a buffer created by [`create_buffer`]: free the allocation first
/// (gpu-allocator does not own the `VkBuffer` we passed to
/// `AllocationScheme::DedicatedBuffer`), then destroy the buffer itself.
fn destroy_buffer(
    device: &ash::Device,
    allocator: &mut Allocator,
    buf: BufferAlloc,
    name: &str,
) {
    if let Err(e) = allocator.free(buf.allocation) {
        eprintln!("warning: free({name}): {e}");
    }
    unsafe { device.destroy_buffer(buf.buffer, None) };
}

fn write_cpu(device: &ash::Device, buf: &BufferAlloc, data: &[u8]) -> Result<(), String> {
    let ptr = buf
        .allocation
        .mapped_ptr()
        .ok_or("write target is not CPU-mapped")?;
    unsafe {
        std::ptr::copy_nonoverlapping(data.as_ptr(), ptr.cast().as_ptr(), data.len());
    }
    flush(device, &buf.allocation, data.len() as u64)
}

fn flush(device: &ash::Device, allocation: &Allocation, size: u64) -> Result<(), String> {
    if !allocation
        .memory_properties()
        .contains(vk::MemoryPropertyFlags::HOST_COHERENT)
    {
        unsafe {
            device.flush_mapped_memory_ranges(&[vk::MappedMemoryRange::default()
                .memory(allocation.memory())
                .offset(allocation.offset())
                .size(size)])
        }
        .map_err(|e| format!("flush_mapped_memory_ranges: {e}"))?;
    }
    Ok(())
}

fn read_gpu_to_cpu(
    device: &ash::Device,
    buf: &BufferAlloc,
    size: usize,
) -> Result<Vec<u8>, String> {
    if !buf
        .allocation
        .memory_properties()
        .contains(vk::MemoryPropertyFlags::HOST_COHERENT)
    {
        unsafe {
            device.invalidate_mapped_memory_ranges(&[vk::MappedMemoryRange::default()
                .memory(buf.allocation.memory())
                .offset(buf.allocation.offset())
                .size(size as u64)])
        }
        .map_err(|e| format!("invalidate_mapped_memory_ranges: {e}"))?;
    }
    let ptr = buf
        .allocation
        .mapped_ptr()
        .ok_or("readback allocation is not CPU-mapped")?;
    let mut out = vec![0u8; size];
    unsafe {
        std::ptr::copy_nonoverlapping(ptr.cast().as_ptr(), out.as_mut_ptr(), size);
    }
    Ok(out)
}

fn pixel_at(pixels: &[u8], x: u32, y: u32) -> [u8; 4] {
    let i = ((y * WIDTH + x) * 4) as usize;
    [pixels[i], pixels[i + 1], pixels[i + 2], pixels[i + 3]]
}

fn pixel_close(got: &[u8; 4], want: &[u8; 4]) -> bool {
    got.iter()
        .zip(want.iter())
        .all(|(g, w)| (*g as i32 - *w as i32).abs() <= TOL)
}
