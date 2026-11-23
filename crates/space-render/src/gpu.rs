//! ash/Vulkan context: instance + device creation and the two ways to render a
//! frame — headless ([`RenderContext::render_offscreen`]: render to an image,
//! read it back) and into a window
//! ([`RenderContext::render_to_swapchain`]: render, blit into the swapchain
//! image, present; nothing is read back).
//!
//! Device selection prefers a device with the full ray-query chain
//! (`VK_KHR_ray_query` + `VK_KHR_acceleration_structure` +
//! `VK_KHR_deferred_host_operations`); a device without it is a clean
//! fallback that renders the forward pass with GI disabled.

use std::ffi::{c_char, CStr};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use ash::vk;
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};
use raw_window_handle::RawDisplayHandle;

use crate::alloc::SharedAllocator;
use crate::environment::Environment;
use crate::mesh::{MeshData, ModelInstance};
use crate::overlay::Overlay;
use crate::params::{self, GpuInstance, MeshInstanceGpu, OverlayInstance};
use crate::passes::{
    DrawRun, FrameDraw, FrameGpu, FrameKey, FrameTarget, InstanceBuffer, MeshDraw, MeshGpu,
    MeshTlasInstance, PipelineConfig, ShapeBlas, Tlas, WhiteTexture, MAX_CUSTOM_INDEX,
};
use crate::quality::{self, QualityPreset};
use crate::scene::{sphere_level_for, sphere_vertex_count, Scene, SHAPE_BOX, SPHERE_LEVELS};
use crate::swapchain::{PresentOutcome, Swapchain, PRESENT_TIMEOUT_NS};

/// Largest frame edge the renderer accepts, in pixels: a sanity bound that
/// every desktop GPU's `maxImageDimension2D` covers.
pub const MAX_FRAME_DIMENSION: u32 = 16_384;

/// Largest GI probe grid edge (the probe texture is `grid x 6*grid`).
pub const MAX_GI_GRID: u32 = 1_024;

/// Validation layer name (NUL-terminated for the Vulkan API).
const VALIDATION_LAYER: &CStr = c"VK_LAYER_KHRONOS_validation";

/// Validation messages (warning or error severity) seen so far, process-wide.
static VALIDATION_MESSAGES: AtomicU32 = AtomicU32::new(0);

unsafe extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _types: vk::DebugUtilsMessageTypeFlagsEXT,
    p_data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user: *mut core::ffi::c_void,
) -> vk::Bool32 {
    if severity == vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE {
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
            unsafe { CStr::from_ptr(d.p_message) }
                .to_string_lossy()
                .into_owned()
        };
        format!("[{name}] {msg}")
    };
    VALIDATION_MESSAGES.fetch_add(1, Ordering::SeqCst);
    eprintln!("VALIDATION: {msg}");
    vk::FALSE
}

/// For GPU tests: fail instead of skipping when no usable Vulkan device is
/// found and `SPACE_REQUIRE_GPU=1` is set.
///
/// The GPU tests skip (loudly, but green) on a machine without a device that
/// has `VK_KHR_acceleration_structure`, so that the suite runs anywhere. On a
/// runner that is *supposed* to have one, that turns "nothing was rendered"
/// into a passing job; setting the variable there makes the skip a failure.
/// Call it on the skip path with the context creation error.
///
/// # Panics
///
/// When `SPACE_REQUIRE_GPU=1`.
pub fn skip_or_require_gpu(error: &RenderError) {
    assert!(
        std::env::var("SPACE_REQUIRE_GPU").as_deref() != Ok("1"),
        "SPACE_REQUIRE_GPU=1 but no usable Vulkan device: {error}"
    );
    eprintln!("SKIP: no usable Vulkan device ({error}); needs VK_KHR_acceleration_structure");
}

/// Validation messages (warning+) recorded by the debug messenger so far.
#[must_use]
pub fn validation_message_count() -> u32 {
    VALIDATION_MESSAGES.load(Ordering::SeqCst)
}

/// GI probe-grid quality knobs. [`Quality::CI`] is the cheap preset that runs
/// fast on lavapipe; [`Quality::DEFAULT`] is the interactive preset.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quality {
    /// Probe grid edge length (the grid is `grid x grid` probes).
    pub grid: u32,
    /// Rays cast per stored direction (6 directions per probe).
    pub rays_per_probe: u32,
    /// Ray max distance in metres.
    pub probe_range: f32,
    /// Probe spacing in metres.
    pub spacing: f32,
}

impl Quality {
    /// Cheap, deterministic preset for CI (lavapipe): 4x4 probes, 8 rays per
    /// direction.
    pub const CI: Self = Self {
        grid: 4,
        rays_per_probe: 8,
        probe_range: 4_000.0,
        spacing: 5.0,
    };
    /// Interactive preset: 16x16 probes, 16 rays per direction.
    pub const DEFAULT: Self = Self {
        grid: 16,
        rays_per_probe: 16,
        probe_range: 20_000.0,
        spacing: 64.0,
    };
}

impl Default for Quality {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// Per-frame render options, including the mutation switches the GPU tests
/// use to prove their pixel assertions bite (each test names its mutation).
#[derive(Clone, Copy, Debug)]
pub struct RenderOptions {
    /// Run the ray-query probe GI pass and feed it into the forward pass.
    pub gi_enabled: bool,
    pub quality: Quality,
    /// Quality tier driving the optional render features (fog, ...); see
    /// [`crate::quality`]. `RenderOptions::default()` uses
    /// [`QualityPreset::Ci`] (all optional features off) so headless/golden
    /// callers stay deterministic; the client opts in to its own preset.
    pub preset: QualityPreset,
    pub exposure: f32,
    /// Multiplier on the preset's fog density (default 1). The presets' fog is
    /// tuned for scenes metres to kilometres across; a scene at astronomical
    /// scale sets this to 0, or every body would be solid fog colour.
    pub fog_scale: f32,
    /// MUTATION SWITCH: skip the forward draw entirely (the render pass is
    /// still begun and ended, so only the draw is removed). Any pixel test
    /// that depends on geometry must fail with this set.
    pub mutate_skip_forward_draw: bool,
    /// MUTATION SWITCH: skip the GI compute dispatch while leaving
    /// `gi_enabled` on, so the forward pass reads a never-written (zero)
    /// probe texture. The GI pixel test must fail with this set.
    pub mutate_skip_gi_dispatch: bool,
    /// MUTATION SWITCH: upload `light_count = 0` while leaving the scene's
    /// lights in place, so the analytic BRDF term is the only thing removed.
    /// The analytic-light pixel test must fail with this set.
    pub mutate_zero_light_count: bool,
    /// Render the raw ray-query probe (one primary ray per pixel against the
    /// TLAS, [`RqParams`] colours) instead of the forward pass. Used by the
    /// ray-query correctness test and by `render_ray_probe`.
    pub ray_probe: bool,
}

impl Default for RenderOptions {
    fn default() -> Self {
        Self {
            gi_enabled: true,
            quality: Quality::DEFAULT,
            preset: QualityPreset::Ci,
            exposure: 1.0,
            fog_scale: 1.0,
            mutate_skip_forward_draw: false,
            mutate_skip_gi_dispatch: false,
            mutate_zero_light_count: false,
            ray_probe: false,
        }
    }
}

/// The pixels + metadata read back from one offscreen frame.
#[derive(Clone, Debug)]
pub struct FrameOutput {
    pub width: u32,
    pub height: u32,
    /// RGBA8, row-major, top row first.
    pub pixels: Vec<u8>,
    /// Validation messages (warning+) recorded during this call.
    pub validation_messages: u32,
    /// Number of render instances drawn.
    pub instance_count: usize,
}

/// What a context has done for the mesh models it has drawn (see
/// [`RenderContext::model_cache_stats`]). The counters only ever grow: a
/// model's buffers, textures and acceleration structure are created once and
/// reused for as long as consecutive frames keep drawing it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ModelCacheStats {
    /// Models currently on the device.
    pub cached: usize,
    /// Models uploaded so far (vertex and index buffers, textures).
    pub uploads: u64,
    /// Bottom-level acceleration structures built so far: one per model, the
    /// first time a frame that traces rays draws it.
    pub blas_builds: u64,
}

/// Ray-query capability of the selected device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RayQuerySupport {
    pub ray_query: bool,
    pub acceleration_structure: bool,
    pub deferred_host_operations: bool,
}

impl RayQuerySupport {
    #[must_use]
    pub fn complete(&self) -> bool {
        self.ray_query && self.acceleration_structure && self.deferred_host_operations
    }
}

/// Options for [`RenderContext::new`].
#[derive(Clone, Debug)]
pub struct ContextOptions {
    /// Request `VK_LAYER_KHRONOS_validation` when the layer is present;
    /// silently degrade when it is not. `None` reads `SPACE_RENDER_VALIDATION`
    /// ("1" enables).
    pub validation: Option<bool>,
    /// Prefer a discrete/ray-query-capable device (default true).
    pub prefer_ray_query: bool,
    /// The display the context will present to, for the windowed path. When
    /// set, the instance enables the surface extensions that display needs
    /// (`VK_KHR_surface` plus `VK_KHR_win32_surface` / `VK_KHR_xlib_surface` /
    /// `VK_KHR_xcb_surface` / `VK_KHR_wayland_surface`, as
    /// `ash_window::enumerate_required_extensions` reports). Without them
    /// `ash_window::create_surface` aborts on a null entry point. `None` (the
    /// default) is the headless path and enables no surface extension.
    pub window_display: Option<RawDisplayHandle>,
}

impl Default for ContextOptions {
    fn default() -> Self {
        Self {
            validation: None,
            prefer_ray_query: true,
            window_display: None,
        }
    }
}

// ---- error type -------------------------------------------------------------

/// Everything that can go wrong while creating a context or rendering a frame.
#[derive(Clone, Debug)]
pub struct RenderError {
    /// Name of the GPU object involved (`"blas"`, `"hdr-image"`, …) when the
    /// failure is tied to one specific allocation.
    pub name: Option<&'static str>,
    pub message: String,
}

impl RenderError {
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            name: None,
            message: message.into(),
        }
    }

    #[must_use]
    pub fn with_name(name: &'static str, message: impl Into<String>) -> Self {
        Self {
            name: Some(name),
            message: message.into(),
        }
    }
}

impl std::fmt::Display for RenderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.name {
            Some(name) => write!(f, "{name}: {}", self.message),
            None => write!(f, "{}", self.message),
        }
    }
}

impl std::error::Error for RenderError {}

/// Copy a Vulkan fixed-size `c_char` array into a `String`.
pub(crate) fn carray_to_string(array: &[std::ffi::c_char]) -> String {
    let bytes: Vec<u8> = array
        .iter()
        .take_while(|&&c| c != 0)
        .map(|&c| c as u8)
        .collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

// ---- context ----------------------------------------------------------------

/// GPU objects kept between frames (see [`crate::passes`]). Rendering a
/// steady stream of frames of one size and feature set allocates nothing: the
/// frame's images, buffers, descriptor sets and command buffer, the instance
/// buffer, the uploaded models (buffers, textures, BLAS) and the acceleration
/// structures are all reused.
#[derive(Default)]
struct RenderCache {
    frame: Option<(FrameKey, FrameGpu)>,
    instances: Option<InstanceBuffer<GpuInstance>>,
    /// One record per drawn mesh part.
    mesh_records: Option<InstanceBuffer<MeshInstanceGpu>>,
    /// Uploaded mesh models, by the address of their shared `MeshData` (the
    /// `Arc` kept alongside pins that address for as long as the entry lives).
    /// An entry holds the model's vertex and index buffers, its textures and,
    /// once a frame that traces rays has drawn it, its BLAS: each is created
    /// once per mesh, never per frame.
    meshes: std::collections::HashMap<usize, (Arc<MeshData>, MeshGpu)>,
    /// How many models have been uploaded, and BLASes built, so far.
    model_uploads: u64,
    blas_builds: u64,
    /// What an untextured mesh part samples; created with the first model.
    white: Option<WhiteTexture>,
    /// The overlay's rectangles (an instance-rate vertex buffer); created
    /// with the first non-empty overlay.
    overlay: Option<InstanceBuffer<OverlayInstance>>,
    /// The unit sphere/cube BLASes; built on the first frame that traces rays.
    shapes: Option<ShapeBlas>,
    tlas: Option<Tlas>,
    /// The frame's instances grouped by detail level (scratch, reused).
    ordered: Vec<GpuInstance>,
}

/// The ash/Vulkan context: instance, device, gpu-allocator, pipelines, the
/// command pool every frame records into and the frame resources it reuses.
///
/// Rendering takes `&self`: the reused resources sit behind a mutex, so one
/// context renders one frame at a time.
///
/// Drop order is explicit: the cached frame resources, then pipelines +
/// command pool, then the allocator, then the device and instance. A
/// [`Swapchain`] created from the context must be dropped before it.
pub struct RenderContext {
    pub(crate) entry: ash::Entry,
    pub(crate) instance: ash::Instance,
    debug: Option<(ash::ext::debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
    pub(crate) physical_device: vk::PhysicalDevice,
    pub(crate) device: ash::Device,
    pub(crate) queue: vk::Queue,
    queue_family: u32,
    pub(crate) accel: ash::khr::acceleration_structure::Device,
    /// `Option` so [`Drop`] can free the allocator *before*
    /// `destroy_device` (`Allocator::drop` calls `vkFreeMemory`).
    pub(crate) allocator: Option<SharedAllocator>,
    pub(crate) pipelines: crate::passes::Pipelines,
    pub(crate) cmd_pool: vk::CommandPool,
    cache: Mutex<RenderCache>,
    device_name: String,
    ray_query: RayQuerySupport,
    /// The device's `maxImageDimension2D`: a texture level larger than this
    /// is not uploaded.
    max_texture_dimension: u32,
    /// True only when `VK_LAYER_KHRONOS_validation` was actually enabled on the
    /// instance (requested *and* installed). Lets a test tell "validated and
    /// clean" apart from "no layer, so nothing to report".
    validation_active: bool,
    /// True when the instance was created with the window-surface extensions
    /// (`ContextOptions::window_display`); `Swapchain::new` refuses otherwise.
    pub(crate) window_surface: bool,
}

impl RenderContext {
    /// Create the instance, pick a device and build every pipeline.
    pub fn new(options: &ContextOptions) -> Result<Self, RenderError> {
        let entry = unsafe { ash::Entry::load() }
            .map_err(|e| RenderError::new(format!("Vulkan loader load failed: {e}")))?;

        // ---- instance ---------------------------------------------------------
        let validation = match options.validation {
            Some(v) => v,
            None => std::env::var("SPACE_RENDER_VALIDATION")
                .map(|v| v == "1")
                .unwrap_or(false),
        };
        let mut instance_exts: Vec<*const c_char> =
            vec![vk::KHR_GET_PHYSICAL_DEVICE_PROPERTIES2_NAME.as_ptr()];
        if let Some(display) = options.window_display {
            // Windowed path: enable the platform surface extensions *at instance
            // creation* (they cannot be added later).
            for &ext in crate::swapchain::WindowHandles::required_instance_extensions(display)? {
                if !instance_exts.contains(&ext) {
                    instance_exts.push(ext);
                }
            }
        }
        let window_surface = options.window_display.is_some();
        let mut layers: Vec<*const c_char> = Vec::new();
        let mut with_debug_utils = false;
        let mut messenger_info = if validation {
            // The layer is requested only when it is actually installed, so a
            // machine without it still renders (just unvalidated).
            let available =
                unsafe { entry.enumerate_instance_layer_properties() }.map_err(|e| {
                    RenderError::new(format!("enumerate_instance_layer_properties: {e}"))
                })?;
            let name = VALIDATION_LAYER.to_string_lossy();
            if available
                .iter()
                .any(|l| carray_to_string(&l.layer_name) == *name)
            {
                instance_exts.push(vk::EXT_DEBUG_UTILS_NAME.as_ptr());
                with_debug_utils = true;
                layers.push(VALIDATION_LAYER.as_ptr());
                Some(
                    vk::DebugUtilsMessengerCreateInfoEXT::default()
                        .message_severity(
                            vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                                | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
                        )
                        .message_type(
                            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION,
                        )
                        .pfn_user_callback(Some(debug_callback)),
                )
            } else {
                log::warn!("validation requested but {name} is not installed; continuing without");
                None
            }
        } else {
            None
        };

        let app_info = vk::ApplicationInfo::default()
            .application_name(c"space-render")
            .application_version(vk::make_api_version(0, 1, 0, 0))
            .engine_name(c"space-render")
            .engine_version(vk::make_api_version(0, 1, 0, 0))
            .api_version(vk::make_api_version(0, 1, 3, 0));
        let mut instance_create = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_layer_names(&layers)
            .enabled_extension_names(&instance_exts);
        if let Some(mi) = &mut messenger_info {
            instance_create = instance_create.push_next(mi);
        }
        let instance = unsafe { entry.create_instance(&instance_create, None) }
            .map_err(|e| RenderError::new(format!("create_instance: {e}")))?;
        let debug = match (&messenger_info, with_debug_utils) {
            (Some(mi), true) => {
                let du = ash::ext::debug_utils::Instance::new(&entry, &instance);
                let messenger = unsafe { du.create_debug_utils_messenger(mi, None) }
                    .map_err(|e| RenderError::new(format!("create_debug_utils_messenger: {e}")))?;
                Some((du, messenger))
            }
            _ => None,
        };

        // ---- physical device ---------------------------------------------------
        let pdevices = unsafe { instance.enumerate_physical_devices() }
            .map_err(|e| RenderError::new(format!("enumerate_physical_devices: {e}")))?;
        if pdevices.is_empty() {
            return Err(RenderError::new("no Vulkan physical devices"));
        }
        let mut best: Option<(vk::PhysicalDevice, String, RayQuerySupport, usize)> = None;
        for &pd in &pdevices {
            let props = unsafe { instance.get_physical_device_properties(pd) };
            let name = carray_to_string(&props.device_name);
            let support = Self::probe_ray_query(&instance, pd);
            let graphics_and_compute = unsafe {
                instance
                    .get_physical_device_queue_family_properties(pd)
                    .iter()
                    .any(|q| {
                        q.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                            && q.queue_flags.contains(vk::QueueFlags::COMPUTE)
                    })
            };
            if !graphics_and_compute {
                log::debug!("skipping {name}: no graphics+compute queue family");
                continue;
            }
            let score = support.complete() as usize
                + (props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU) as usize * 4;
            let better = best.as_ref().is_none_or(|(_, _, _, s)| score > *s);
            if better {
                best = Some((pd, name, support, score));
            }
        }
        let (physical_device, device_name, ray_query_support, _) = best
            .ok_or_else(|| RenderError::new("no physical device with a graphics+compute queue"))?;
        if options.prefer_ray_query && !ray_query_support.complete() {
            log::warn!("{device_name} lacks the ray-query chain; GI will be disabled");
        }
        log::info!(
            "device: {device_name} (ray_query={})",
            ray_query_support.complete()
        );

        // The renderer always builds a BLAS (`SceneGpu::upload`), so it cannot run
        // without `VK_KHR_acceleration_structure`. ash panics (and aborts) when
        // loading that extension's entry points on a device that lacks it, so
        // report the device as unusable instead; callers/tests treat an `Err`
        // from `new` as "no usable Vulkan device" and skip.
        let has_accel_structure =
            ray_query_support.acceleration_structure && ray_query_support.deferred_host_operations;
        if !has_accel_structure {
            return Err(RenderError::new(format!(
                "{device_name} lacks VK_KHR_acceleration_structure (+ deferred_host_operations)"
            )));
        }

        // ---- device extensions + features --------------------------------------
        let mut required: Vec<&CStr> = vec![
            vk::KHR_BUFFER_DEVICE_ADDRESS_NAME,
            vk::KHR_GET_MEMORY_REQUIREMENTS2_NAME,
            vk::KHR_ACCELERATION_STRUCTURE_NAME,
            vk::KHR_DEFERRED_HOST_OPERATIONS_NAME,
        ];
        if ray_query_support.complete() {
            required.push(vk::KHR_RAY_QUERY_NAME.as_ref());
        }
        if window_surface {
            // Without VK_KHR_swapchain enabled on the *device*, the swapchain entry
            // points are not valid for it and `vkCreateSwapchainKHR` fails with
            // VK_ERROR_INITIALIZATION_FAILED (seen on NVIDIA/Windows).
            let available =
                unsafe { instance.enumerate_device_extension_properties(physical_device) }
                    .map_err(|e| {
                        RenderError::new(format!("enumerate_device_extension_properties: {e}"))
                    })?;
            if !available.iter().any(|p| {
                carray_to_string(&p.extension_name) == vk::KHR_SWAPCHAIN_NAME.to_string_lossy()
            }) {
                return Err(RenderError::new(format!(
                    "{device_name} lacks VK_KHR_swapchain, which the windowed path requires"
                )));
            }
            required.push(vk::KHR_SWAPCHAIN_NAME);
        }
        let ext_names: Vec<*const c_char> = required.iter().map(|c| c.as_ptr()).collect();

        // Anisotropic filtering for model textures: an optional core feature,
        // enabled where the device has it (nearly everywhere).
        let (max_anisotropy, max_texture_dimension) = unsafe {
            let limits = instance
                .get_physical_device_properties(physical_device)
                .limits;
            let anisotropy = instance
                .get_physical_device_features(physical_device)
                .sampler_anisotropy
                != 0;
            (
                if anisotropy {
                    limits.max_sampler_anisotropy
                } else {
                    0.0
                },
                limits.max_image_dimension2_d,
            )
        };
        let core_features =
            vk::PhysicalDeviceFeatures::default().sampler_anisotropy(max_anisotropy > 1.0);

        let mut rq_features = vk::PhysicalDeviceRayQueryFeaturesKHR::default().ray_query(true);
        let mut as_features = vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default()
            .acceleration_structure(true);
        let mut bda_features =
            vk::PhysicalDeviceBufferDeviceAddressFeatures::default().buffer_device_address(true);
        let mut features2 = vk::PhysicalDeviceFeatures2::default()
            .features(core_features)
            .push_next(&mut bda_features)
            .push_next(&mut as_features);
        if ray_query_support.complete() {
            features2 = features2.push_next(&mut rq_features);
        }

        let queue_props =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
        let queue_family = queue_props
            .iter()
            .position(|q| {
                q.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                    && q.queue_flags.contains(vk::QueueFlags::COMPUTE)
            })
            .ok_or_else(|| RenderError::new("no graphics+compute queue family"))?
            as u32;
        let priorities = [1.0f32];
        let queue_create = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family)
            .queue_priorities(&priorities)];
        let device_create = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_create)
            .enabled_extension_names(&ext_names)
            .push_next(&mut features2);
        let device = unsafe { instance.create_device(physical_device, &device_create, None) }
            .map_err(|e| RenderError::new(format!("create_device: {e}")))?;
        let queue = unsafe { device.get_device_queue(queue_family, 0) };
        let accel = ash::khr::acceleration_structure::Device::new(&instance, &device);

        let allocator = Arc::new(Mutex::new(
            Allocator::new(&AllocatorCreateDesc {
                instance: instance.clone(),
                device: device.clone(),
                physical_device,
                debug_settings: Default::default(),
                buffer_device_address: true,
                allocation_sizes: Default::default(),
            })
            .map_err(|e| RenderError::new(format!("gpu-allocator init: {e}")))?,
        ));

        let pipelines = crate::passes::Pipelines::create(
            &device,
            &crate::passes::Shaders::from_spv(),
            PipelineConfig {
                ray_query: ray_query_support.complete(),
                max_anisotropy: if max_anisotropy > 1.0 {
                    max_anisotropy
                } else {
                    0.0
                },
            },
        )?;

        let cmd_pool = unsafe {
            device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .queue_family_index(queue_family)
                        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                    None,
                )
                .map_err(|e| RenderError::new(format!("create_command_pool: {e}")))?
        };

        Ok(Self {
            entry,
            instance,
            debug,
            physical_device,
            device,
            queue,
            queue_family,
            accel,
            allocator: Some(allocator),
            pipelines,
            cmd_pool,
            cache: Mutex::new(RenderCache::default()),
            device_name,
            ray_query: ray_query_support,
            max_texture_dimension,
            validation_active: with_debug_utils,
            window_surface,
        })
    }

    /// Which ray-query features and extensions `pd` offers.
    fn probe_ray_query(instance: &ash::Instance, pd: vk::PhysicalDevice) -> RayQuerySupport {
        let available: Vec<String> = unsafe {
            instance
                .enumerate_device_extension_properties(pd)
                .unwrap_or_default()
        }
        .iter()
        .map(|e| carray_to_string(&e.extension_name))
        .collect();
        let has = |name: &CStr| available.iter().any(|n| n == &name.to_string_lossy());
        let mut probe_rq = vk::PhysicalDeviceRayQueryFeaturesKHR::default();
        let mut probe_as = vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default();
        let mut probe_bda = vk::PhysicalDeviceBufferDeviceAddressFeatures::default();
        let mut probe = vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut probe_rq)
            .push_next(&mut probe_as)
            .push_next(&mut probe_bda);
        unsafe { instance.get_physical_device_features2(pd, &mut probe) };
        let _ = probe_bda;
        RayQuerySupport {
            ray_query: has(vk::KHR_RAY_QUERY_NAME) && probe_rq.ray_query != 0,
            acceleration_structure: has(vk::KHR_ACCELERATION_STRUCTURE_NAME)
                && probe_as.acceleration_structure != 0,
            deferred_host_operations: has(vk::KHR_DEFERRED_HOST_OPERATIONS_NAME),
        }
    }

    /// The live allocator (only `None` inside [`Drop`]).
    pub(crate) fn allocator(&self) -> Result<&SharedAllocator, RenderError> {
        self.allocator
            .as_ref()
            .ok_or_else(|| RenderError::new("allocator already freed"))
    }

    /// The queue family index the context renders on (graphics + compute).
    #[must_use]
    pub fn queue_family(&self) -> u32 {
        self.queue_family
    }

    /// Name of the selected physical device.
    #[must_use]
    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    /// Ray-query capability of the selected device.
    #[must_use]
    pub fn ray_query_support(&self) -> RayQuerySupport {
        self.ray_query
    }

    /// True when the `VK_LAYER_KHRONOS_validation` layer is actually enabled on
    /// this context's instance (requested *and* installed). A `false` here means
    /// validation messages cannot be produced, so a `validation_messages == 0`
    /// assertion would be vacuous.
    #[must_use]
    pub fn validation_active(&self) -> bool {
        self.validation_active
    }

    /// True when the ray-query chain is usable, so probe GI can run.
    #[must_use]
    pub fn supports_gi(&self) -> bool {
        self.ray_query.complete()
    }

    /// How many mesh models are on the device, and how many uploads and
    /// acceleration-structure builds it took to get there.
    #[must_use]
    pub fn model_cache_stats(&self) -> ModelCacheStats {
        let cache = self.cache();
        ModelCacheStats {
            cached: cache.meshes.len(),
            uploads: cache.model_uploads,
            blas_builds: cache.blas_builds,
        }
    }

    /// Does a frame of `scene` with `options` trace shadow rays? Only when the
    /// preset has shadows ([`crate::quality`]), the device can trace rays and
    /// the scene has a light to be shadowed from. (The raw ray probe draws no
    /// lit surface at all.) A frame for which this is false runs exactly the
    /// forward shaders it would without the feature.
    #[must_use]
    pub fn shadows_active(&self, scene: &Scene, options: &RenderOptions) -> bool {
        options.preset.features().shadows
            && self.ray_query.complete()
            && !options.ray_probe
            && !scene.lights.is_empty()
    }

    /// Does a frame of `scene` with `options` trace rays through its glass —
    /// refract it through its real shape instead of bending the view once at
    /// its surface? Only when the preset has refraction, the device can
    /// trace rays and there is glass in the scene.
    #[must_use]
    pub fn glass_rays_active(&self, scene: &Scene, options: &RenderOptions) -> bool {
        options.preset.features().refraction
            && self.ray_query.complete()
            && !options.ray_probe
            && scene.instances.iter().any(|i| i.transmission > 0.0)
    }

    /// Does a frame of `scene` with `options` cast sky rays — ask, per pixel,
    /// how much of the environment's light reaches a surface? Only when the
    /// preset has them ([`crate::quality`]), the device can trace rays and
    /// the scene has an environment to keep out.
    #[must_use]
    pub fn sky_occlusion_active(&self, scene: &Scene, options: &RenderOptions) -> bool {
        options.preset.features().sky_rays > 0
            && self.ray_query.complete()
            && !options.ray_probe
            && !scene.environment.is_none()
    }

    /// Render one frame of `scene` offscreen and read the pixels back.
    ///
    /// `options.gi_enabled` is honoured only when the device supports ray
    /// query; otherwise the forward pass renders with GI off.
    pub fn render_offscreen(
        &self,
        scene: &Scene,
        width: u32,
        height: u32,
        options: &RenderOptions,
    ) -> Result<FrameOutput, RenderError> {
        let before = validation_message_count();
        let options = self.checked_options(width, height, options)?;
        let mut cache = self.cache();
        let pixels = self.render_readback(&mut cache, scene, width, height, &options);
        if pixels.is_err() {
            self.discard_frame(&mut cache);
        }
        Ok(FrameOutput {
            width,
            height,
            pixels: pixels?,
            validation_messages: validation_message_count().wrapping_sub(before),
            instance_count: scene.instances.len(),
        })
    }

    /// Render one frame of `scene` at the swapchain's size straight into its
    /// next image and queue it for presentation. Nothing is read back to the
    /// CPU, and the call does not wait for the GPU to finish the frame (the
    /// next call does, so at most one frame is in flight).
    ///
    /// On [`PresentOutcome::OutOfDate`] nothing was presented; on that and on
    /// [`PresentOutcome::Suboptimal`] call [`Swapchain::recreate`] before
    /// rendering again. `swapchain` must have been created from this context.
    pub fn render_to_swapchain(
        &self,
        scene: &Scene,
        swapchain: &Swapchain,
        options: &RenderOptions,
    ) -> Result<PresentOutcome, RenderError> {
        let (width, height) = swapchain.extent();
        let options = self.checked_options(width, height, options)?;
        let mut cache = self.cache();
        let outcome = self.render_present(&mut cache, scene, swapchain, &options);
        if outcome.is_err() {
            self.discard_frame(&mut cache);
        }
        outcome
    }

    fn cache(&self) -> MutexGuard<'_, RenderCache> {
        self.cache.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// The options a `width` x `height` frame actually renders with, or why it
    /// cannot be rendered at all.
    fn checked_options(
        &self,
        width: u32,
        height: u32,
        options: &RenderOptions,
    ) -> Result<RenderOptions, RenderError> {
        let valid = 1..=MAX_FRAME_DIMENSION;
        if !valid.contains(&width) || !valid.contains(&height) {
            return Err(RenderError::new(format!(
                "frame size {width}x{height} is outside 1..={MAX_FRAME_DIMENSION} pixels per side"
            )));
        }
        let mut options = *options;
        if options.ray_probe && !self.ray_query.complete() {
            return Err(RenderError::new(format!(
                "{} lacks VK_KHR_ray_query, which the ray probe needs",
                self.device_name
            )));
        }
        if options.gi_enabled && !self.ray_query.complete() {
            static WARN: std::sync::Once = std::sync::Once::new();
            WARN.call_once(|| {
                log::warn!("GI requested but the device lacks ray query; rendering without GI");
            });
            options.gi_enabled = false;
        }
        if options.gi_enabled && !(1..=MAX_GI_GRID).contains(&options.quality.grid) {
            return Err(RenderError::new(format!(
                "GI probe grid {} is outside 1..={MAX_GI_GRID}",
                options.quality.grid
            )));
        }
        Ok(options)
    }

    fn render_readback(
        &self,
        cache: &mut RenderCache,
        scene: &Scene,
        width: u32,
        height: u32,
        options: &RenderOptions,
    ) -> Result<Vec<u8>, RenderError> {
        self.ensure_frame(cache, FrameKey::new(width, height, options), u64::MAX)?;
        let frame = self.record_frame(cache, scene, options, FrameTarget::Readback)?;
        frame.submit(self.queue, None, None)?;
        frame.wait(u64::MAX)?;
        frame.read_pixels()
    }

    fn render_present(
        &self,
        cache: &mut RenderCache,
        scene: &Scene,
        swapchain: &Swapchain,
        options: &RenderOptions,
    ) -> Result<PresentOutcome, RenderError> {
        let (width, height) = swapchain.extent();
        // This also waits for the previous frame's submission, after which the
        // swapchain's acquire semaphore is free to be signalled again.
        self.ensure_frame(
            cache,
            FrameKey::new(width, height, options),
            PRESENT_TIMEOUT_NS,
        )?;
        let Some((index, suboptimal)) = swapchain.acquire()? else {
            return Ok(PresentOutcome::OutOfDate);
        };
        let target = FrameTarget::Present(swapchain.image(index));
        let frame = self.record_frame(cache, scene, options, target)?;
        frame.submit(
            self.queue,
            // The swapchain image is first touched by the blit, in the
            // TRANSFER stage; everything before it may run before the image
            // has been acquired.
            Some((swapchain.image_ready(), vk::PipelineStageFlags::TRANSFER)),
            Some(swapchain.present_done(index)),
        )?;
        let outcome = swapchain.present(self.queue, index)?;
        Ok(match outcome {
            PresentOutcome::Presented if suboptimal => PresentOutcome::Suboptimal,
            other => other,
        })
    }

    /// Make the cached frame match `key`, waiting (up to `timeout_ns`) for its
    /// previous submission first. On return nothing is pending on the frame,
    /// so its buffers, descriptor sets and command buffer may be rewritten.
    fn ensure_frame(
        &self,
        cache: &mut RenderCache,
        key: FrameKey,
        timeout_ns: u64,
    ) -> Result<(), RenderError> {
        if let Some((have, frame)) = &cache.frame {
            frame.wait(timeout_ns)?;
            if *have == key {
                return Ok(());
            }
        }
        // Free the old frame's memory before allocating its replacement.
        cache.frame = None;
        let frame = FrameGpu::new(
            &self.device,
            self.allocator()?,
            &self.pipelines,
            self.cmd_pool,
            key,
        )?;
        cache.frame = Some((key, frame));
        Ok(())
    }

    /// Drop the cached frame after a failed render: its fence may never be
    /// signalled, so reusing it could wait forever.
    fn discard_frame(&self, cache: &mut RenderCache) {
        unsafe {
            let _ = self.device.device_wait_idle();
        }
        cache.frame = None;
    }

    /// Upload `scene` into the cached buffers, write the frame's uniforms and
    /// record its command buffer. [`RenderContext::ensure_frame`] must have
    /// succeeded for this frame.
    fn record_frame<'c>(
        &self,
        cache: &'c mut RenderCache,
        scene: &Scene,
        options: &RenderOptions,
        target: FrameTarget,
    ) -> Result<&'c FrameGpu, RenderError> {
        let allocator = self.allocator()?;
        let count = u32::try_from(scene.instances.len())
            .map_err(|_| RenderError::new("the scene has more than u32::MAX instances"))?;
        let grown = |n: u32, min: u32| n.max(min).checked_next_power_of_two().unwrap_or(n);
        let Some((width, height)) = cache.frame.as_ref().map(|(k, _)| (k.width, k.height)) else {
            return Err(RenderError::new("no frame resources to record into"));
        };

        if cache
            .instances
            .as_ref()
            .is_none_or(|b| b.capacity() < count)
        {
            cache.instances = None;
            cache.instances = Some(InstanceBuffer::new(
                &self.device,
                allocator,
                grown(count, 64),
                vk::BufferUsageFlags::STORAGE_BUFFER,
                "render-instances",
            )?);
        }

        // Acceleration structures only for a frame that actually traces rays.
        // (Shadow rays and sky rays come out of the same forward shaders.)
        let shadows =
            self.shadows_active(scene, options) || self.sky_occlusion_active(scene, options);
        // The post pass follows rays for glass, and for what a mirror
        // reflects from off screen (which the environment would otherwise
        // shine through).
        let glass_rays = self.glass_rays_active(scene, options)
            || (options.preset.features().ssr && self.sky_occlusion_active(scene, options));
        let traces_rays = options.gi_enabled || options.ray_probe || shadows || glass_rays;

        // Mesh models: upload any model seen for the first time (buffers and
        // textures; its BLAS when a frame first traces rays against it), drop
        // the ones this scene no longer uses (nothing is pending, so that is
        // safe), and lay out one record + one indexed draw per material part
        // and one TLAS instance per placement.
        let in_use: std::collections::HashSet<usize> =
            scene.models.iter().map(|m| mesh_key(&m.mesh)).collect();
        cache.meshes.retain(|key, _| in_use.contains(key));
        let white = if scene.models.is_empty() {
            vk::DescriptorSet::null()
        } else {
            if cache.white.is_none() {
                cache.white = Some(WhiteTexture::new(
                    &self.device,
                    allocator,
                    self.queue,
                    self.cmd_pool,
                    &self.pipelines,
                )?);
            }
            match &cache.white {
                Some(white) => white.set()?,
                None => vk::DescriptorSet::null(),
            }
        };
        let mut mesh_records = Vec::new();
        let mut mesh_draws = Vec::new();
        let mut mesh_placements = Vec::new();
        for model in &scene.models {
            let key = mesh_key(&model.mesh);
            let gpu = match cache.meshes.entry(key) {
                std::collections::hash_map::Entry::Occupied(entry) => &mut entry.into_mut().1,
                std::collections::hash_map::Entry::Vacant(entry) => {
                    let gpu = MeshGpu::upload(
                        &self.device,
                        allocator,
                        self.queue,
                        self.cmd_pool,
                        &self.pipelines,
                        &model.mesh,
                        white,
                        self.max_texture_dimension,
                    )?;
                    cache.model_uploads += 1;
                    &mut entry.insert((Arc::clone(&model.mesh), gpu)).1
                }
            };
            let first_record = mesh_records.len() as u32;
            for part in &gpu.parts {
                let Some(source) = model.mesh.parts.get(part.part) else {
                    continue;
                };
                mesh_draws.push(MeshDraw {
                    vertices: gpu.vertices.handle,
                    indices: gpu.indices.handle,
                    first_index: part.first_index,
                    index_count: part.index_count,
                    record: mesh_records.len() as u32,
                    texture_set: part.texture_set,
                });
                mesh_records.push(mesh_record(model, &source.material, gpu.is_sheet()));
            }
            // A placement the rays can see: its records start at `first_record`
            // and follow the BLAS's geometry order (one per drawn part).
            if traces_rays && is_placeable(model) {
                if first_record >= MAX_CUSTOM_INDEX {
                    return Err(RenderError::new(
                        "the scene has more mesh parts than a ray hit can address",
                    ));
                }
                cache.blas_builds += u64::from(!gpu.has_blas());
                let blas = gpu.ensure_blas(
                    &self.device,
                    &self.accel,
                    allocator,
                    self.queue,
                    self.cmd_pool,
                )?;
                let [x, y, z] = model_rows(model);
                mesh_placements.push(MeshTlasInstance {
                    transform: [
                        x[0], x[1], x[2], x[3], //
                        y[0], y[1], y[2], y[3], //
                        z[0], z[1], z[2], z[3],
                    ],
                    blas,
                    custom_index: first_record + 1,
                    flags: gpu.instance_flags,
                });
            }
        }
        let record_count = mesh_records.len() as u32;
        if cache
            .mesh_records
            .as_ref()
            .is_none_or(|b| b.capacity() < record_count)
        {
            cache.mesh_records = None;
            cache.mesh_records = Some(InstanceBuffer::new(
                &self.device,
                allocator,
                grown(record_count, 16),
                vk::BufferUsageFlags::STORAGE_BUFFER,
                "mesh-records",
            )?);
        }
        let tlas_count = u32::try_from(scene.instances.len() + mesh_placements.len())
            .map_err(|_| RenderError::new("the scene has more than u32::MAX ray-traced objects"))?;
        if traces_rays {
            if cache.shapes.is_none() {
                cache.shapes = Some(ShapeBlas::build(
                    &self.device,
                    &self.accel,
                    allocator,
                    self.queue,
                    self.cmd_pool,
                )?);
            }
            if cache
                .tlas
                .as_ref()
                .is_none_or(|t| t.capacity() < tlas_count)
            {
                cache.tlas = None;
                cache.tlas = Some(Tlas::new(
                    &self.device,
                    &self.accel,
                    allocator,
                    grown(tlas_count, 16),
                )?);
            }
        }

        // The overlay's rectangles, in normalised device coordinates. An
        // empty overlay touches nothing here.
        let overlay_rects = overlay_instances(&scene.overlay, width, height);
        let overlay_count = u32::try_from(overlay_rects.len())
            .map_err(|_| RenderError::new("the overlay has more than u32::MAX rectangles"))?;
        if overlay_count > 0
            && cache
                .overlay
                .as_ref()
                .is_none_or(|b| b.capacity() < overlay_count)
        {
            cache.overlay = None;
            cache.overlay = Some(InstanceBuffer::new(
                &self.device,
                allocator,
                grown(overlay_count, 256),
                vk::BufferUsageFlags::VERTEX_BUFFER,
                "overlay-rects",
            )?);
        }

        // Group the instances by detail level: one instanced draw per level.
        let runs = group_by_detail(scene, height, &mut cache.ordered);

        let cache: &'c RenderCache = cache;
        let (Some((_, frame)), Some(instances), Some(mesh_buffer)) =
            (&cache.frame, &cache.instances, &cache.mesh_records)
        else {
            return Err(RenderError::new("no frame resources to record into"));
        };
        instances.write(&cache.ordered)?;
        mesh_buffer.write(&mesh_records)?;
        let overlay = match &cache.overlay {
            Some(buffer) if overlay_count > 0 => {
                buffer.write(&overlay_rects)?;
                Some((buffer.buffer.handle, overlay_count))
            }
            _ => None,
        };

        let (forward_params, gi_params, tone_params) =
            build_uniforms(scene, width, height, options);
        let tlas = match (traces_rays, &cache.shapes, &cache.tlas) {
            (true, Some(shapes), Some(tlas)) => {
                // Two kinds of instance cast no shadow: glass (a hard black
                // shadow behind a window is worse than none), and a body with
                // a light inside it — it is that light's lamp (the Sun around
                // the solar system's point light), not something in its way.
                let lights = &forward_params.lights[..scene.lights.len().min(8)];
                let casts_shadow = |instance: &GpuInstance| {
                    instance.transmission <= 0.0
                        && !lights
                            .iter()
                            .any(|light| instance_contains(instance, light.position))
                };
                tlas.write_instances(shapes, &cache.ordered, casts_shadow, &mesh_placements)?;
                Some(tlas)
            }
            _ => None,
        };
        frame.bind_scene(
            &self.device,
            &instances.buffer,
            &mesh_buffer.buffer,
            tlas.map(|t| t.handle),
        );

        // Uniform data is written straight into the CPU-mapped uniform
        // buffers; nothing is pending on the frame, so the GPU is not reading
        // them.
        frame
            .forward_params
            .write(params::as_bytes(&forward_params))?;
        if let Some(gi) = gi_params {
            frame.gi_params.write(params::as_bytes(&gi))?;
        }
        frame.tone_params.write(params::as_bytes(&tone_params))?;
        if let Some(post) = frame.post_params() {
            let post_params = build_post_params(&forward_params, width, height, options);
            post.write(params::as_bytes(&post_params))?;
        }
        if options.ray_probe {
            let rq = rq_uniforms(scene, width, height);
            frame.rq_params.write(params::as_bytes(&rq))?;
        }

        frame.record(
            &self.device,
            &self.pipelines,
            &FrameDraw {
                tlas_instances: tlas_count,
                runs: &runs,
                meshes: &mesh_draws,
                tlas,
                gi_dispatch: options.gi_enabled && !options.mutate_skip_gi_dispatch,
                shadows: shadows && tlas.is_some(),
                glass_rays: glass_rays && tlas.is_some(),
                sky: scene.environment.background && !scene.environment.is_none(),
                overlay,
            },
            options,
            target,
        )?;
        Ok(frame)
    }
}

/// Cache key of a shared mesh: the address of its `MeshData`.
fn mesh_key(mesh: &Arc<MeshData>) -> usize {
    Arc::as_ptr(mesh) as usize
}

/// The rows of `model`'s 3x4 model -> camera-relative world matrix.
fn model_rows(model: &ModelInstance) -> [[f32; 4]; 3] {
    [0, 1, 2].map(|r| {
        let [x, y, z] = model.rotation[r];
        [
            x * model.scale,
            y * model.scale,
            z * model.scale,
            model.position[r],
        ]
    })
}

/// Can `model` be an instance of the ray-traced scene? Its transform has to
/// be finite and invertible, which the rasteriser does not care about (a
/// zero-scale model is simply not visible).
fn is_placeable(model: &ModelInstance) -> bool {
    let finite = model
        .rotation
        .iter()
        .flatten()
        .chain(&model.position)
        .all(|v| v.is_finite());
    finite && model.scale.is_finite() && model.scale != 0.0
}

/// The GPU record of one part of `model`: its placement and `material`, and
/// whether its mesh is an open surface (`sheet`) rather than a closed solid.
fn mesh_record(
    model: &ModelInstance,
    material: &crate::mesh::MeshMaterial,
    sheet: bool,
) -> MeshInstanceGpu {
    let [model_x, model_y, model_z] = model_rows(model);
    let [r, g, b] = material.base_color;
    let [er, eg, eb] = material.emissive;
    MeshInstanceGpu {
        model_x,
        model_y,
        model_z,
        albedo: [r, g, b, material.metallic],
        emissive: [er, eg, eb, material.roughness],
        surface: [f32::from(u8::from(sheet)), 0.0, 0.0, 0.0],
    }
}

/// Is the camera-relative point `p` inside `instance` — within its radius for
/// a sphere, within its half-extent on every axis for a box?
fn instance_contains(instance: &GpuInstance, p: [f32; 4]) -> bool {
    let [x, y, z, radius] = instance.center;
    let d = [p[0] - x, p[1] - y, p[2] - z];
    if instance.shape == SHAPE_BOX {
        d.iter().all(|c| c.abs() <= radius.abs())
    } else {
        d[0] * d[0] + d[1] * d[1] + d[2] * d[2] <= radius * radius
    }
}

/// `overlay`'s rectangles for a `width` x `height` frame: pixel coordinates
/// (origin top-left, +y down) to normalised device coordinates, which run
/// from -1 at the left/top edge to +1 at the right/bottom one.
fn overlay_instances(overlay: &Overlay, width: u32, height: u32) -> Vec<OverlayInstance> {
    let (sx, sy) = (2.0 / width.max(1) as f32, 2.0 / height.max(1) as f32);
    overlay
        .rects()
        .iter()
        .map(|r| OverlayInstance {
            rect: [
                r.x * sx - 1.0,
                r.y * sy - 1.0,
                (r.x + r.width) * sx - 1.0,
                (r.y + r.height) * sy - 1.0,
            ],
            color: r.color,
        })
        .collect()
}

/// The sphere detail level an instance is drawn at: from the radius of its
/// silhouette in pixels on a frame `height` pixels tall. Boxes have one mesh.
fn detail_level(instance: &GpuInstance, focal_px: f32) -> u32 {
    if instance.shape == SHAPE_BOX {
        return 0;
    }
    let [x, y, z, radius] = instance.center;
    let distance = (x * x + y * y + z * z).sqrt();
    // Angular radius of a sphere seen from outside is asin(r / d); r / d is
    // the same to first order and saturates (inf) once the camera is inside.
    let radius_px = radius.abs() / (distance - radius.abs()).max(1e-6) * focal_px;
    sphere_level_for(radius_px)
}

/// Copy `scene`'s instances into `ordered`, grouped by detail level (highest
/// first, scene order kept within a level), and return the instanced draw for
/// each level. `ordered` is what gets uploaded: instance indices on the GPU
/// (including ray-query hits) refer to it.
fn group_by_detail(
    scene: &Scene,
    frame_height: u32,
    ordered: &mut Vec<GpuInstance>,
) -> [DrawRun; SPHERE_LEVELS as usize] {
    // Pixels per unit of tan(angle): height / (2 tan(fov_y / 2)).
    let focal_px = frame_height as f32 / (2.0 * (scene.camera.fov_y * 0.5).tan()).max(1e-6);
    let levels: Vec<u32> = scene
        .instances
        .iter()
        .map(|instance| detail_level(instance, focal_px))
        .collect();
    let mut counts = [0u32; SPHERE_LEVELS as usize];
    for &level in &levels {
        counts[level as usize] += 1;
    }
    // Highest detail first: slot `s` holds level `SPHERE_LEVELS - 1 - s`.
    let mut runs = [DrawRun::default(); SPHERE_LEVELS as usize];
    let mut first = 0;
    for (slot, run) in runs.iter_mut().enumerate() {
        let level = SPHERE_LEVELS - 1 - slot as u32;
        *run = DrawRun {
            first_instance: first,
            instance_count: counts[level as usize],
            vertex_count: sphere_vertex_count(level),
            first_vertex: level << 24,
        };
        first += counts[level as usize];
    }
    ordered.clear();
    ordered.resize(scene.instances.len(), GpuInstance::default());
    let mut next = runs.map(|run| run.first_instance as usize);
    for (instance, &level) in scene.instances.iter().zip(&levels) {
        let slot = (SPHERE_LEVELS - 1 - level) as usize;
        ordered[next[slot]] = *instance;
        next[slot] += 1;
    }
    runs
}

impl Drop for RenderContext {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
        }
        // The cached frame resources own buffers, images and a command
        // buffer: they go before the command pool and the allocator.
        drop(std::mem::take(
            self.cache.get_mut().unwrap_or_else(PoisonError::into_inner),
        ));
        unsafe {
            self.device.destroy_command_pool(self.cmd_pool, None);
            self.pipelines.destroy(&self.device);
            if let Some((du, messenger)) = &self.debug {
                du.destroy_debug_utils_messenger(*messenger, None);
            }
            // The allocator must outlive every Buffer/Image (they are gone
            // now) *and* be destroyed before the device: `Allocator::drop`
            // calls `vkFreeMemory`, which is illegal once the device is gone.
            // Taking it out of `self` makes that order explicit rather than
            // relying on field order.
            if let Some(allocator) = self.allocator.take() {
                drop(allocator);
            }
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

/// Build the three uniform blocks for one frame.
///
/// The GI probe grid is centred on the camera in the camera-relative XZ plane
/// at the camera height, so `grid_origin = (-((gx-1)/2) * spacing, 0, -((gy-1)/2)
/// * spacing)`.
fn build_uniforms(
    scene: &Scene,
    width: u32,
    height: u32,
    options: &RenderOptions,
) -> (
    params::ForwardParams,
    Option<params::GiParams>,
    params::ToneParams,
) {
    let cam = &scene.camera;
    let view_proj = scene.view_proj();
    let inv_view_proj = scene.inv_view_proj().unwrap_or(view_proj);
    let tan_half_fov = (cam.fov_y * 0.5).tan();
    let features = options.preset.features();
    let fog_density = features.fog_density * options.fog_scale.max(0.0);
    let mut feature_flags = features.flags();
    // No fog at all (also for a NaN density) clears the feature bit.
    if fog_density.is_nan() || fog_density <= 0.0 {
        feature_flags &= !quality::FEATURE_FOG;
    }

    let mut forward = params::ForwardParams {
        camera: params::Camera {
            view_proj,
            inv_view_proj,
            // Instances are camera-relative, so the camera sits at the origin.
            position: [0.0, 0.0, 0.0, tan_half_fov],
            forward: [cam.forward[0], cam.forward[1], cam.forward[2], cam.near],
        },
        lights: [params::Light::default(); 8],
        light_count: if options.mutate_zero_light_count {
            0
        } else {
            scene.lights.len().min(8) as u32
        },
        gi_enabled: u32::from(options.gi_enabled),
        _pad0: [0.0; 2],
        gi_grid_origin: [0.0; 4],
        gi_spacing: options.quality.spacing,
        gi_grid_size_x: options.quality.grid,
        gi_grid_size_y: options.quality.grid,
        _pad1: 0.0,
        fog: [
            quality::FOG_COLOR[0],
            quality::FOG_COLOR[1],
            quality::FOG_COLOR[2],
            fog_density,
        ],
        features: feature_flags,
        sky_rays: if scene.environment.is_none() {
            0
        } else {
            features.sky_rays
        },
        ssr_max_roughness: features.ssr_max_roughness,
        _pad2: 0,
        environment: environment_gpu(&scene.environment),
    };
    for (slot, light) in scene.lights.iter().take(8).enumerate() {
        // World-space light -> camera-relative f32 offset.
        let d = space_world::WorldPos::delta(cam.position, light.position);
        forward.lights[slot] = params::Light {
            position: [d[0] as f32, d[1] as f32, d[2] as f32, 1.0],
            color: [
                light.color[0],
                light.color[1],
                light.color[2],
                light.intensity,
            ],
        };
    }

    let gi = if options.gi_enabled {
        let q = options.quality;
        let half = (q.grid.saturating_sub(1) as f32) * 0.5 * q.spacing;
        forward.gi_grid_origin = [-half, 0.0, -half, 0.0];
        Some(params::GiParams {
            grid_origin: forward.gi_grid_origin,
            spacing: q.spacing,
            probe_range: q.probe_range,
            grid_size: [q.grid, q.grid],
            rays_per_probe: q.rays_per_probe,
            _pad: [0.0; 3],
            environment: forward.environment,
        })
    } else {
        None
    };

    let tone = params::ToneParams {
        exposure: options.exposure,
        _pad0: [0.0; 3],
        image_size: [width, height],
        _pad1: [0.0; 2],
    };
    (forward, gi, tone)
}

/// Index-of-refraction ratio for light entering glass from air (1 / 1.5).
const REFRACT_ETA: f32 = 1.0 / 1.5;

/// Metres the refracted ray travels behind a transmissive surface before the
/// post pass looks the scene up (the screen-space "thickness" of the glass).
const REFRACT_DISTANCE_M: f32 = 2.0;

/// The [`params::PostParams`] for the SSR / refraction post pass: the same
/// camera matrices as the forward pass, so the pass reconstructs positions and
/// projects rays in exactly the forward pass's depth/projection convention.
fn build_post_params(
    forward: &params::ForwardParams,
    width: u32,
    height: u32,
    options: &RenderOptions,
) -> params::PostParams {
    let features = options.preset.features();
    params::PostParams {
        camera: forward.camera,
        features: features.flags(),
        ssr_steps: features.ssr_steps,
        image_size: [width, height],
        ssr_max_roughness: features.ssr_max_roughness,
        refract_distance: REFRACT_DISTANCE_M,
        refract_eta: REFRACT_ETA,
        fog_density: if forward.features & quality::FEATURE_FOG != 0 {
            forward.fog[3]
        } else {
            0.0
        },
        environment: forward.environment,
        lights: forward.lights,
        light_count: forward.light_count,
        glass_bounces: features.glass_bounces,
        _pad1: [0; 2],
    }
}

/// The scene's environment as the shaders read it: the three colours times
/// the intensity, with the "there is one" flag in `zenith.w` and "it is the
/// painted background" in `horizon.w`. All zero for a scene without an
/// environment ([`Environment::is_none`]), so its shaders skip every
/// environment term after reading that one value.
fn environment_gpu(environment: &Environment) -> params::EnvironmentGpu {
    if environment.is_none() {
        return params::EnvironmentGpu::default();
    }
    let [zenith, horizon, ground] = environment.colors();
    let rgb = |c: [f32; 3], w: f32| [c[0], c[1], c[2], w];
    params::EnvironmentGpu {
        zenith: rgb(zenith, 1.0),
        horizon: rgb(horizon, f32::from(u8::from(environment.background))),
        ground: rgb(ground, 0.0),
    }
}

/// The [`params::RqParams`] for a raw ray-query probe frame: a bright magenta
/// hit colour and a dark blue miss colour so a test can tell hit, miss and
/// "nothing was written" apart, and a camera ray setup that matches the
/// forward pass (pinhole at the origin, looking down -Z).
fn rq_uniforms(scene: &Scene, width: u32, height: u32) -> params::RqParams {
    params::RqParams {
        hit_color: [1.0, 0.0, 1.0, 1.0],
        miss_color: [0.0, 0.0, 0.25, 1.0],
        camera_origin: [0.0, 0.0, 0.0, 1.0],
        image_size: [width as f32, height as f32, 0.0, 0.0],
        tan_half_fov: (scene.camera.fov_y * 0.5).tan(),
        _pad: [0.0; 3],
    }
}
