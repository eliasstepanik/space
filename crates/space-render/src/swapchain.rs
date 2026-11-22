//! Swapchain path: present rendered frames into a portable window.
//!
//! The window itself is owned by the caller (`space-client` uses winit, which
//! is pure Rust); this module only needs the `raw-window-handle` raw handles,
//! so `space-render` never depends on a windowing library.
//!
//! A frame is rendered exactly as in the headless path (so the pixels are
//! identical) and, instead of being read back, blitted into the acquired
//! swapchain image at the end of the same command buffer — see
//! [`RenderContext::render_to_swapchain`].

use ash::vk;
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};

use crate::gpu::{RenderContext, RenderError};

/// Upper bound for every blocking Vulkan wait on the present path (image
/// acquisition, the previous frame's fence), in nanoseconds. Finite so a stuck
/// GPU/compositor surfaces as an error instead of freezing the window's event
/// loop.
pub(crate) const PRESENT_TIMEOUT_NS: u64 = 5_000_000_000;

/// A window a swapchain can be created for.
#[derive(Clone, Copy, Debug)]
pub struct WindowHandles {
    pub display: RawDisplayHandle,
    pub window: RawWindowHandle,
}

impl WindowHandles {
    /// The instance extensions a surface on `display` requires.
    pub fn required_instance_extensions(
        display: RawDisplayHandle,
    ) -> Result<&'static [*const std::ffi::c_char], RenderError> {
        ash_window::enumerate_required_extensions(display)
            .map_err(|e| RenderError::new(format!("enumerate_required_extensions: {e}")))
    }
}

/// How a swapchain paces presentation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SwapchainConfig {
    /// `true` (the default): wait for the display's vertical blank (`FIFO`) —
    /// no tearing, and the GPU idles between refreshes. `false`: present as
    /// fast as frames are produced (`MAILBOX`, else `IMMEDIATE`, else `FIFO`),
    /// for benchmarking.
    pub vsync: bool,
}

impl Default for SwapchainConfig {
    fn default() -> Self {
        Self { vsync: true }
    }
}

/// What became of a frame handed to [`RenderContext::render_to_swapchain`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PresentOutcome {
    /// The frame is on its way to the screen.
    Presented,
    /// The frame was presented, but the swapchain no longer matches the
    /// surface exactly: call [`Swapchain::recreate`] before the next frame.
    Suboptimal,
    /// Nothing was presented: the swapchain no longer matches the surface
    /// (the window was resized). Call [`Swapchain::recreate`], then render
    /// again.
    OutOfDate,
}

/// The surface-dependent half of a [`Swapchain`], replaced on every recreate.
struct Chain {
    handle: vk::SwapchainKHR,
    images: Vec<vk::Image>,
    format: vk::Format,
    extent: vk::Extent2D,
    present_mode: vk::PresentModeKHR,
}

/// A window surface, its swapchain and the semaphores that order image
/// acquisition, the frame's blit and the present.
pub struct Swapchain {
    surface: vk::SurfaceKHR,
    surface_ext: ash::khr::surface::Instance,
    swapchain_ext: ash::khr::swapchain::Device,
    physical_device: vk::PhysicalDevice,
    config: SwapchainConfig,
    chain: Chain,
    /// Signalled when the acquired image may be written. One is enough: a
    /// frame is only acquired after the previous frame's submission, which
    /// waited on it, has completed.
    image_ready: vk::Semaphore,
    /// One per swapchain image, signalled when the blit into that image is
    /// done and waited on by its present. Per image because a present may
    /// still hold its semaphore when the *next* image is being rendered.
    present_done: Vec<vk::Semaphore>,
    device: ash::Device,
}

/// `count` fresh binary semaphores; on failure none are left behind.
fn semaphores(device: &ash::Device, count: usize) -> Result<Vec<vk::Semaphore>, RenderError> {
    let mut made = Vec::with_capacity(count);
    for _ in 0..count {
        match unsafe { device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None) } {
            Ok(s) => made.push(s),
            Err(e) => {
                for s in made {
                    unsafe { device.destroy_semaphore(s, None) };
                }
                return Err(RenderError::new(format!("create_semaphore: {e}")));
            }
        }
    }
    Ok(made)
}

impl Swapchain {
    /// Create a surface for `handles` and a vsynced swapchain sized to
    /// `width` x `height` (clamped to the surface's supported extent).
    pub fn new(
        ctx: &RenderContext,
        handles: WindowHandles,
        width: u32,
        height: u32,
    ) -> Result<Self, RenderError> {
        Self::with_config(ctx, handles, width, height, SwapchainConfig::default())
    }

    /// [`Swapchain::new`] with explicit pacing.
    pub fn with_config(
        ctx: &RenderContext,
        handles: WindowHandles,
        width: u32,
        height: u32,
        config: SwapchainConfig,
    ) -> Result<Self, RenderError> {
        // `ash_window::create_surface` panics (and, in a non-unwinding fn,
        // aborts) when the platform surface extension was not enabled on the
        // instance, so refuse cleanly here instead.
        if !ctx.window_surface {
            return Err(RenderError::new(
                "create_surface: the context was created without a window display \
                 (set ContextOptions::window_display), so no surface extension is enabled",
            ));
        }
        let surface_ext = ash::khr::surface::Instance::new(&ctx.entry, &ctx.instance);
        let swapchain_ext = ash::khr::swapchain::Device::new(&ctx.instance, &ctx.device);
        let surface = unsafe {
            ash_window::create_surface(
                &ctx.entry,
                &ctx.instance,
                handles.display,
                handles.window,
                None,
            )
            .map_err(|e| RenderError::new(format!("create_surface: {e}")))?
        };
        // From here on the surface (and later the chain) must not leak on an
        // error path.
        let created = (|| {
            // The queue must be able to present to this surface.
            let can_present = unsafe {
                surface_ext
                    .get_physical_device_surface_support(
                        ctx.physical_device,
                        ctx.queue_family(),
                        surface,
                    )
                    .unwrap_or(false)
            };
            if !can_present {
                return Err(RenderError::new(
                    "create_swapchain: the render queue family cannot present to this surface",
                ));
            }
            let chain = create_chain(
                &surface_ext,
                &swapchain_ext,
                ctx.physical_device,
                surface,
                config,
                (width, height),
                vk::SwapchainKHR::null(),
            )?
            .ok_or_else(|| {
                // A minimised window reports a 0x0 extent; a swapchain cannot
                // be created for it.
                RenderError::new(
                    "create_swapchain: surface extent is 0x0 (window minimised or not yet laid out)",
                )
            })?;
            // One per image for the presents, plus the acquire semaphore.
            match semaphores(&ctx.device, chain.images.len() + 1) {
                Ok(made) => Ok((chain, made)),
                Err(e) => {
                    unsafe { swapchain_ext.destroy_swapchain(chain.handle, None) };
                    Err(e)
                }
            }
        })();
        let (chain, mut present_done) = match created {
            Ok(parts) => parts,
            Err(e) => {
                unsafe { surface_ext.destroy_surface(surface, None) };
                return Err(e);
            }
        };
        let image_ready = present_done.pop().unwrap_or_default();
        log::info!(
            "swapchain: {}x{} {:?} {:?}, {} images",
            chain.extent.width,
            chain.extent.height,
            chain.format,
            chain.present_mode,
            chain.images.len()
        );
        Ok(Self {
            surface,
            surface_ext,
            swapchain_ext,
            physical_device: ctx.physical_device,
            config,
            chain,
            image_ready,
            present_done,
            device: ctx.device.clone(),
        })
    }

    /// The swapchain's size in pixels.
    #[must_use]
    pub fn extent(&self) -> (u32, u32) {
        (self.chain.extent.width, self.chain.extent.height)
    }

    /// Whether presentation waits for the vertical blank (`FIFO`). False when
    /// vsync was turned off *and* the surface offers an unthrottled mode.
    #[must_use]
    pub fn is_vsynced(&self) -> bool {
        matches!(
            self.chain.present_mode,
            vk::PresentModeKHR::FIFO | vk::PresentModeKHR::FIFO_RELAXED
        )
    }

    /// Rebuild the swapchain for the surface's current size (after a resize,
    /// or when a frame came back [`PresentOutcome::OutOfDate`] /
    /// [`PresentOutcome::Suboptimal`]). `width` x `height` is only used when
    /// the surface does not dictate its own extent.
    ///
    /// Returns `Ok(false)` — and keeps the old swapchain — while the surface
    /// is 0x0 (minimised window): there is nothing to render to until it has
    /// a size again.
    pub fn recreate(&mut self, width: u32, height: u32) -> Result<bool, RenderError> {
        // Nothing may still be rendering into or presenting the old images.
        unsafe {
            self.device
                .device_wait_idle()
                .map_err(|e| RenderError::new(format!("device_wait_idle: {e}")))?;
        }
        let Some(chain) = create_chain(
            &self.surface_ext,
            &self.swapchain_ext,
            self.physical_device,
            self.surface,
            self.config,
            (width, height),
            self.chain.handle,
        )?
        else {
            return Ok(false);
        };
        let retired = std::mem::replace(&mut self.chain, chain);
        unsafe { self.swapchain_ext.destroy_swapchain(retired.handle, None) };
        // Fresh semaphores: an acquire that was never followed by a present
        // (an error mid-frame) would otherwise leave one signalled, and the
        // image count may have changed.
        let mut fresh = semaphores(&self.device, self.chain.images.len() + 1)?;
        unsafe {
            self.device.destroy_semaphore(self.image_ready, None);
            for s in self.present_done.drain(..) {
                self.device.destroy_semaphore(s, None);
            }
        }
        self.image_ready = fresh.pop().unwrap_or_default();
        self.present_done = fresh;
        Ok(true)
    }

    /// Acquire the next image. `Ok(None)` when the swapchain is out of date
    /// (nothing was acquired); otherwise the image index and whether the
    /// swapchain is merely suboptimal (the image *was* acquired and must be
    /// presented).
    pub(crate) fn acquire(&self) -> Result<Option<(u32, bool)>, RenderError> {
        let acquired = unsafe {
            self.swapchain_ext.acquire_next_image(
                self.chain.handle,
                PRESENT_TIMEOUT_NS,
                self.image_ready,
                vk::Fence::null(),
            )
        };
        match acquired {
            Ok((index, suboptimal)) if (index as usize) < self.chain.images.len() => {
                Ok(Some((index, suboptimal)))
            }
            Ok((index, _)) => Err(RenderError::new(format!(
                "acquire_next_image returned image {index} of {}",
                self.chain.images.len()
            ))),
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => Ok(None),
            Err(e) => Err(RenderError::new(format!("acquire_next_image: {e}"))),
        }
    }

    /// The swapchain image at `index` (as returned by [`Swapchain::acquire`]).
    pub(crate) fn image(&self, index: u32) -> vk::Image {
        self.chain.images[index as usize]
    }

    /// The semaphore the acquisition signals: the frame's submission waits on it.
    pub(crate) fn image_ready(&self) -> vk::Semaphore {
        self.image_ready
    }

    /// The semaphore the frame's submission signals for image `index`.
    pub(crate) fn present_done(&self, index: u32) -> vk::Semaphore {
        self.present_done[index as usize]
    }

    /// Queue image `index` for presentation once its `present_done` semaphore
    /// is signalled.
    pub(crate) fn present(
        &self,
        queue: vk::Queue,
        index: u32,
    ) -> Result<PresentOutcome, RenderError> {
        let wait = [self.present_done(index)];
        let swapchains = [self.chain.handle];
        let indices = [index];
        let info = vk::PresentInfoKHR::default()
            .wait_semaphores(&wait)
            .swapchains(&swapchains)
            .image_indices(&indices);
        match unsafe { self.swapchain_ext.queue_present(queue, &info) } {
            Ok(false) => Ok(PresentOutcome::Presented),
            Ok(true) => Ok(PresentOutcome::Suboptimal),
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => Ok(PresentOutcome::OutOfDate),
            Err(e) => Err(RenderError::new(format!("queue_present: {e}"))),
        }
    }
}

/// Create a swapchain for `surface` as it is right now, retiring `old`.
/// `Ok(None)` when the surface is currently 0x0.
fn create_chain(
    surface_ext: &ash::khr::surface::Instance,
    swapchain_ext: &ash::khr::swapchain::Device,
    physical_device: vk::PhysicalDevice,
    surface: vk::SurfaceKHR,
    config: SwapchainConfig,
    (width, height): (u32, u32),
    old: vk::SwapchainKHR,
) -> Result<Option<Chain>, RenderError> {
    let capabilities = unsafe {
        surface_ext
            .get_physical_device_surface_capabilities(physical_device, surface)
            .map_err(|e| RenderError::new(format!("surface capabilities: {e}")))?
    };
    let formats = unsafe {
        surface_ext
            .get_physical_device_surface_formats(physical_device, surface)
            .map_err(|e| RenderError::new(format!("surface formats: {e}")))?
    };
    let present_modes = unsafe {
        surface_ext
            .get_physical_device_surface_present_modes(physical_device, surface)
            .map_err(|e| RenderError::new(format!("surface present modes: {e}")))?
    };

    // FIFO is the one mode every surface must offer.
    let present_mode = if config.vsync {
        vk::PresentModeKHR::FIFO
    } else {
        [vk::PresentModeKHR::MAILBOX, vk::PresentModeKHR::IMMEDIATE]
            .into_iter()
            .find(|mode| present_modes.contains(mode))
            .unwrap_or(vk::PresentModeKHR::FIFO)
    };

    // The frame is tone-mapped and gamma-encoded into an `rgba8unorm` image
    // and *blitted* into the swapchain image. A blit converts between formats
    // channel by channel, so both RGBA and BGRA orders show the right colours;
    // what matters is that the surface format is UNORM (an `_SRGB` format would
    // gamma-encode the already-encoded values a second time). Prefer the
    // 8-bit formats in the standard sRGB colour space, then any colour space,
    // then the 10-bit UNORM formats.
    let unorm = [
        vk::Format::R8G8B8A8_UNORM,
        vk::Format::B8G8R8A8_UNORM,
        vk::Format::A2B10G10R10_UNORM_PACK32,
        vk::Format::A2R10G10B10_UNORM_PACK32,
    ];
    let pick = |srgb_only: bool| {
        unorm.iter().find_map(|wanted| {
            formats.iter().find(|f| {
                f.format == *wanted
                    && (!srgb_only || f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR)
            })
        })
    };
    let surface_format = pick(true)
        .or_else(|| pick(false))
        .ok_or_else(|| RenderError::new("surface offers no 8- or 10-bit UNORM format"))?;

    // `current_extent` is authoritative unless it is the (MAX, MAX) sentinel;
    // otherwise clamp the request into the supported range (tolerating a
    // driver reporting max < min).
    let extent = if capabilities.current_extent.width != u32::MAX
        && capabilities.current_extent.height != u32::MAX
    {
        capabilities.current_extent
    } else {
        let clamp = |value: u32, min: u32, max: u32| value.clamp(min, max.max(min));
        vk::Extent2D {
            width: clamp(
                width,
                capabilities.min_image_extent.width,
                capabilities.max_image_extent.width,
            ),
            height: clamp(
                height,
                capabilities.min_image_extent.height,
                capabilities.max_image_extent.height,
            ),
        }
    };
    if extent.width == 0 || extent.height == 0 {
        return Ok(None);
    }

    // One more than the minimum, so acquiring never has to wait for the
    // presentation engine to release its last image. max_image_count == 0
    // means "no limit".
    let mut image_count = capabilities.min_image_count.saturating_add(1).max(2);
    if capabilities.max_image_count != 0 {
        image_count = image_count.min(capabilities.max_image_count);
    }
    // The blit needs TRANSFER_DST on the swapchain images.
    if !capabilities
        .supported_usage_flags
        .contains(vk::ImageUsageFlags::TRANSFER_DST)
    {
        return Err(RenderError::new(
            "create_swapchain: surface does not support TRANSFER_DST image usage",
        ));
    }
    // composite_alpha must be one of the supported bits.
    let composite_alpha = [
        vk::CompositeAlphaFlagsKHR::OPAQUE,
        vk::CompositeAlphaFlagsKHR::INHERIT,
        vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED,
        vk::CompositeAlphaFlagsKHR::POST_MULTIPLIED,
    ]
    .into_iter()
    .find(|c| capabilities.supported_composite_alpha.contains(*c))
    .unwrap_or(vk::CompositeAlphaFlagsKHR::OPAQUE);

    let handle = unsafe {
        swapchain_ext
            .create_swapchain(
                &vk::SwapchainCreateInfoKHR::default()
                    .surface(surface)
                    .min_image_count(image_count)
                    .image_format(surface_format.format)
                    .image_color_space(surface_format.color_space)
                    .image_extent(extent)
                    .image_array_layers(1)
                    .image_usage(vk::ImageUsageFlags::TRANSFER_DST)
                    .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .pre_transform(capabilities.current_transform)
                    .composite_alpha(composite_alpha)
                    .present_mode(present_mode)
                    .clipped(true)
                    .old_swapchain(old),
                None,
            )
            .map_err(|e| RenderError::new(format!("create_swapchain: {e}")))?
    };
    let images = match unsafe { swapchain_ext.get_swapchain_images(handle) } {
        Ok(images) if !images.is_empty() => images,
        other => {
            unsafe { swapchain_ext.destroy_swapchain(handle, None) };
            return Err(RenderError::new(match other {
                Err(e) => format!("get_swapchain_images: {e}"),
                Ok(_) => "get_swapchain_images: the swapchain has no images".to_owned(),
            }));
        }
    };
    Ok(Some(Chain {
        handle,
        images,
        format: surface_format.format,
        extent,
        present_mode,
    }))
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        // The device and instance must still be alive: drop the swapchain
        // before the `RenderContext` it was created from.
        unsafe {
            let _ = self.device.device_wait_idle();
            for s in &self.present_done {
                self.device.destroy_semaphore(*s, None);
            }
            self.device.destroy_semaphore(self.image_ready, None);
            self.swapchain_ext
                .destroy_swapchain(self.chain.handle, None);
            self.surface_ext.destroy_surface(self.surface, None);
        }
    }
}
