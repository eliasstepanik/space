//! ash/Vulkan context: instance + device creation and headless offscreen
//! rendering (render to image, read back).
//!
//! Device selection prefers a device with the full ray-query chain
//! (`VK_KHR_ray_query` + `VK_KHR_acceleration_structure` +
//! `VK_KHR_deferred_host_operations`); a device without it is a clean
//! fallback that renders the forward pass with GI disabled.

use std::ffi::{c_char, CStr};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use ash::vk;
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};

use crate::alloc::SharedAllocator;
use crate::params;
use crate::quality::{self, QualityPreset};
use crate::scene::Scene;

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

/// The ash/Vulkan context: instance, device, gpu-allocator, pipelines and the
/// command pool every offscreen frame records into.
///
/// Drop order is explicit (frames are owned by the caller and must be dropped
/// first): pipelines + command pool, then the allocator, then the device and
/// instance.
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
    device_name: String,
    ray_query: RayQuerySupport,
    /// True only when `VK_LAYER_KHRONOS_validation` was actually enabled on the
    /// instance (requested *and* installed). Lets a test tell "validated and
    /// clean" apart from "no layer, so nothing to report".
    validation_active: bool,
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
        let ext_names: Vec<*const c_char> = required.iter().map(|c| c.as_ptr()).collect();

        let mut rq_features = vk::PhysicalDeviceRayQueryFeaturesKHR::default().ray_query(true);
        let mut as_features = vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default()
            .acceleration_structure(true);
        let mut bda_features =
            vk::PhysicalDeviceBufferDeviceAddressFeatures::default().buffer_device_address(true);
        let mut features2 = vk::PhysicalDeviceFeatures2::default()
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

        let pipelines =
            crate::passes::Pipelines::create(&device, &crate::passes::Shaders::from_spv())?;

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
            device_name,
            ray_query: ray_query_support,
            validation_active: with_debug_utils,
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
        let (output, frame) = self.render_frame_gpu(scene, width, height, options)?;
        drop(frame);
        Ok(output)
    }

    /// Same as [`RenderContext::render_offscreen`] but hands back the live frame
    /// so a swapchain path can copy its `out` image before it is destroyed.
    pub fn render_frame_gpu(
        &self,
        scene: &Scene,
        width: u32,
        height: u32,
        options: &RenderOptions,
    ) -> Result<(FrameOutput, crate::passes::FrameGpu), RenderError> {
        let before = validation_message_count();
        let mut options = *options;
        if options.gi_enabled && !self.ray_query.complete() {
            log::warn!("GI requested but the device lacks ray query; rendering without GI");
            options.gi_enabled = false;
        }
        let gi_dispatch = options.gi_enabled && !options.mutate_skip_gi_dispatch;

        let scene_gpu =
            crate::passes::SceneGpu::upload(&self.device, &self.accel, self.allocator()?, scene)?;
        let frame = crate::passes::create_frame(
            &self.device,
            &self.accel,
            self.allocator()?,
            &self.pipelines,
            &scene_gpu,
            self.queue,
            self.cmd_pool,
            width,
            height,
            &options,
            vk::Format::D32_SFLOAT,
        )?;

        // Uniform data is written straight into the CPU-mapped uniform buffers;
        // each submission below is fenced, so the writes are visible in order.
        let (forward_params, gi_params, tone_params) =
            build_uniforms(scene, width, height, &options);
        frame
            .forward_params
            .write(params::as_bytes(&forward_params))?;
        if let Some(gi) = gi_params {
            frame.gi_params.write(params::as_bytes(&gi))?;
        }
        frame.tone_params.write(params::as_bytes(&tone_params))?;
        if let Some(post) = &frame.post {
            let post_params = build_post_params(&forward_params, width, height, &options);
            post.params.write(params::as_bytes(&post_params))?;
        }
        if options.ray_probe {
            let rq = rq_uniforms(scene, width, height);
            frame.rq_params.write(params::as_bytes(&rq))?;
        }

        let (vertices, _indices) = scene.triangle_soup();
        let max_vertex = vertices.len().saturating_sub(1) as u32;
        frame.record_build_blas(
            &self.device,
            &self.accel,
            &scene_gpu,
            scene_gpu.vertices.device_address(),
            scene_gpu.indices.device_address(),
            max_vertex,
        )?;
        crate::passes::FrameGpu::submit(
            &self.device,
            self.queue,
            frame.cmd_build_blas,
            frame.fence,
        )?;

        if self.ray_query.complete() {
            frame.record_build_tlas(
                &self.device,
                &self.accel,
                &scene_gpu,
                scene_gpu.tlas_instance_addr(),
            )?;
            crate::passes::FrameGpu::submit(
                &self.device,
                self.queue,
                frame.cmd_build_tlas,
                frame.fence,
            )?;
        }

        frame.record_frame(
            &self.device,
            &self.pipelines,
            &scene_gpu,
            &options,
            gi_dispatch,
        )?;
        crate::passes::FrameGpu::submit(&self.device, self.queue, frame.cmd_frame, frame.fence)?;

        let pixels = frame.read_pixels()?;
        let output = FrameOutput {
            width,
            height,
            pixels,
            validation_messages: validation_message_count() - before,
            instance_count: scene.instances.len(),
        };
        Ok((output, frame))
    }
}

impl Drop for RenderContext {
    fn drop(&mut self) {
        // Everything the caller allocated (frames, scenes) is dropped before
        // we get here; only long-lived objects remain.
        unsafe {
            let _ = self.device.device_wait_idle();
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
            features.fog_density,
        ],
        features: features.flags(),
        _pad2: [0; 3],
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
        _pad: 0.0,
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
