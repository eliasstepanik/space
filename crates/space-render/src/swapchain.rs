//! Swapchain path: present a rendered frame into a portable window.
//!
//! The window itself is owned by the caller (`space-client` uses winit, which
//! is pure Rust); this module only needs the `raw-window-handle` raw handles,
//! so `space-render` never depends on a windowing library.
//!
//! The frame is rendered offscreen exactly as in the headless path (so the
//! pixels are identical) and then copied into the acquired swapchain image.

use ash::vk;
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};

use crate::gpu::{RenderContext, RenderError};
use crate::passes::FrameGpu;

/// A window a swapchain can be created for.
#[derive(Clone, Copy, Debug)]
pub struct WindowHandles {
    pub display: RawDisplayHandle,
    pub window: RawWindowHandle,
}

impl WindowHandles {
    /// The instance extensions a surface on `display` requires.
    ///
    /// # Safety
    /// `display` must be a live display handle for the process (the same
    /// requirement `ash_window::create_surface` has).
    pub fn required_instance_extensions(
        display: RawDisplayHandle,
    ) -> Result<&'static [*const std::ffi::c_char], RenderError> {
        ash_window::enumerate_required_extensions(display)
            .map_err(|e| RenderError::new(format!("enumerate_required_extensions: {e}")))
    }
}

/// A swapchain plus the objects needed to blit a rendered frame into it.
pub struct Swapchain {
    pub(crate) surface: vk::SurfaceKHR,
    surface_ext: ash::khr::surface::Instance,
    swapchain_ext: ash::khr::swapchain::Device,
    pub(crate) handle: vk::SwapchainKHR,
    pub(crate) images: Vec<vk::Image>,
    pub(crate) views: Vec<vk::ImageView>,
    // Recorded for diagnostics. The format *does* matter on the present path:
    // the blit only preserves texel values between format-compatible formats, so
    // the selection below deliberately prefers `R8G8B8A8_UNORM` (matching the
    // offscreen target) over the usual `B8G8R8A8_UNORM` first choice — do not
    // "simplify" it back to BGRA, that swaps red and blue. See the comment at the
    // format selection in [`Swapchain::new`].
    #[allow(dead_code)]
    pub(crate) format: vk::Format,
    pub(crate) extent: vk::Extent2D,
    cmd_pool: vk::CommandPool,
    /// The one blit command buffer, allocated with the swapchain and reset on
    /// every `present` (allocating per frame leaked a command buffer per frame).
    present_cmd: vk::CommandBuffer,
    present_ready: vk::Fence,
    image_ready: vk::Semaphore,
    present_done: vk::Semaphore,
    device: ash::Device,
}

impl Swapchain {
    /// Create a surface for `handles` and a swapchain sized to `width` x
    /// `height` (clamped to the surface's supported extent).
    pub fn new(
        ctx: &RenderContext,
        handles: WindowHandles,
        width: u32,
        height: u32,
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
        let capabilities = unsafe {
            surface_ext
                .get_physical_device_surface_capabilities(ctx.physical_device, surface)
                .map_err(|e| RenderError::new(format!("surface capabilities: {e}")))?
        };
        let formats = unsafe {
            surface_ext
                .get_physical_device_surface_formats(ctx.physical_device, surface)
                .map_err(|e| RenderError::new(format!("surface formats: {e}")))?
        };
        let present_modes = unsafe {
            surface_ext
                .get_physical_device_surface_present_modes(ctx.physical_device, surface)
                .map_err(|e| RenderError::new(format!("surface present modes: {e}")))?
        };
        // Prefer a mailbox (no tearing, no artificial cap); fall back to FIFO.
        let present_mode = if present_modes.contains(&vk::PresentModeKHR::MAILBOX) {
            vk::PresentModeKHR::MAILBOX
        } else {
            vk::PresentModeKHR::FIFO
        };
        // The blit below copies the offscreen `out` image into the swapchain
        // image. `cmd_blit_image` only preserves texel values exactly between
        // *format-compatible* formats (same component count, order and size).
        // The offscreen target is `R8G8B8A8_UNORM` (see `passes::create_frame`),
        // so the swapchain must pick the same format: `B8G8R8A8_UNORM` — the
        // usual first choice — is **not** format-compatible with it (red and
        // blue are swapped), so a blit into it would swap the red and blue
        // channels of every pixel. Prefer the exact match, then any
        // `R8G8B8A8_UNORM` colour space, and only then fall back to a
        // `B8G8R8A8_UNORM` surface format.
        let format = formats
            .iter()
            .find(|f| {
                f.format == vk::Format::R8G8B8A8_UNORM
                    && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            })
            .or_else(|| {
                formats
                    .iter()
                    .find(|f| f.format == vk::Format::R8G8B8A8_UNORM)
            })
            .or_else(|| {
                let bgra = formats
                    .iter()
                    .find(|f| f.format == vk::Format::B8G8R8A8_UNORM);
                if bgra.is_some() {
                    // Last resort: the offscreen target is `R8G8B8A8_UNORM`, so
                    // blitting into a BGRA swapchain is *not* format-compatible
                    // and the presentation engine will show red and blue swapped
                    // on every pixel. Every real ICD offers RGBA, so reaching
                    // this branch means the surface is unusual — make it loud
                    // rather than silently mis-tinted.
                    log::warn!(
                        "surface offers no R8G8B8A8_UNORM format; falling back to B8G8R8A8_UNORM, \
                         which the RGBA blit will channel-swap (red/blue swapped on screen)"
                    );
                }
                bgra
            })
            .ok_or_else(|| RenderError::new("surface offers no usable format"))?;
        // Use the colour space the chosen surface format was reported with.
        let (format, color_space) = (format.format, format.color_space);
        // `current_extent` is authoritative unless it is the (MAX, MAX) sentinel;
        // otherwise clamp the request into the supported range (never below 1x1,
        // and tolerate a driver reporting max < min).
        let extent = if capabilities.current_extent.width != u32::MAX
            && capabilities.current_extent.height != u32::MAX
        {
            capabilities.current_extent
        } else {
            vk::Extent2D {
                width: width.max(1).clamp(
                    capabilities.min_image_extent.width.max(1),
                    capabilities
                        .max_image_extent
                        .width
                        .max(capabilities.min_image_extent.width.max(1)),
                ),
                height: height.max(1).clamp(
                    capabilities.min_image_extent.height.max(1),
                    capabilities
                        .max_image_extent
                        .height
                        .max(capabilities.min_image_extent.height.max(1)),
                ),
            }
        };
        if extent.width == 0 || extent.height == 0 {
            // A minimised window reports a 0x0 current extent; a swapchain cannot
            // be created for it.
            unsafe { surface_ext.destroy_surface(surface, None) };
            return Err(RenderError::new(
                "create_swapchain: surface extent is 0x0 (window minimised or not yet laid out)",
            ));
        }
        // max_image_count == 0 means "no limit".
        let mut image_count = capabilities.min_image_count.max(2);
        if capabilities.max_image_count != 0 {
            image_count = image_count.min(capabilities.max_image_count);
        }
        // The blit needs TRANSFER_DST on the swapchain images.
        if !capabilities
            .supported_usage_flags
            .contains(vk::ImageUsageFlags::TRANSFER_DST)
        {
            unsafe { surface_ext.destroy_surface(surface, None) };
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
            unsafe { surface_ext.destroy_surface(surface, None) };
            return Err(RenderError::new(
                "create_swapchain: the render queue family cannot present to this surface",
            ));
        }
        let swapchain_ext = ash::khr::swapchain::Device::new(&ctx.instance, &ctx.device);
        let handle = unsafe {
            swapchain_ext
                .create_swapchain(
                    &vk::SwapchainCreateInfoKHR::default()
                        .surface(surface)
                        .min_image_count(image_count)
                        .image_format(format)
                        .image_color_space(color_space)
                        .image_extent(extent)
                        .image_array_layers(1)
                        .image_usage(vk::ImageUsageFlags::TRANSFER_DST)
                        .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                        .pre_transform(capabilities.current_transform)
                        .composite_alpha(composite_alpha)
                        .present_mode(present_mode)
                        .clipped(true),
                    None,
                )
                .map_err(|e| RenderError::new(format!("create_swapchain: {e}")))?
        };
        let images = unsafe {
            swapchain_ext
                .get_swapchain_images(handle)
                .map_err(|e| RenderError::new(format!("get_swapchain_images: {e}")))?
        };
        let views: Vec<vk::ImageView> = unsafe {
            let mut v = Vec::with_capacity(images.len());
            for image in &images {
                v.push(
                    ctx.device
                        .create_image_view(
                            &vk::ImageViewCreateInfo::default()
                                .image(*image)
                                .view_type(vk::ImageViewType::TYPE_2D)
                                .format(format)
                                .subresource_range(vk::ImageSubresourceRange {
                                    aspect_mask: vk::ImageAspectFlags::COLOR,
                                    base_mip_level: 0,
                                    level_count: 1,
                                    base_array_layer: 0,
                                    layer_count: 1,
                                }),
                            None,
                        )
                        .map_err(|e| RenderError::new(format!("swapchain image view: {e}")))?,
                );
            }
            v
        };
        let cmd_pool = unsafe {
            ctx.device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .queue_family_index(ctx.queue_family())
                        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                    None,
                )
                .map_err(|e| RenderError::new(format!("swapchain command pool: {e}")))?
        };
        let present_ready = unsafe {
            ctx.device
                // Created SIGNALED: the first `present` waits on this fence before
                // anything has been submitted, so an unsignaled fence would block
                // the main thread forever (the "not responding" freeze).
                .create_fence(
                    &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                    None,
                )
                .map_err(|e| RenderError::new(format!("create_fence: {e}")))?
        };
        // One primary command buffer for the whole lifetime of the swapchain:
        // `present` resets it before recording, and the `present_ready` fence
        // wait at the top of `present` guarantees the previous submit (and so
        // this buffer) has finished before it is reused.
        let present_cmd = unsafe {
            ctx.device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(cmd_pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )
                .map_err(|e| RenderError::new(format!("allocate cmd buffer: {e}")))?[0]
        };
        let image_ready = unsafe {
            ctx.device
                .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                .map_err(|e| RenderError::new(format!("create_semaphore: {e}")))?
        };
        let present_done = unsafe {
            ctx.device
                .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                .map_err(|e| RenderError::new(format!("create_semaphore: {e}")))?
        };
        Ok(Self {
            surface,
            surface_ext,
            swapchain_ext,
            handle,
            images,
            views,
            format,
            extent,
            cmd_pool,
            present_cmd,
            present_ready,
            image_ready,
            present_done,
            device: ctx.device.clone(),
        })
    }

    /// The swapchain's size in pixels.
    #[must_use]
    pub fn extent(&self) -> (u32, u32) {
        (self.extent.width, self.extent.height)
    }

    /// Acquire an image, blit `frame`'s `out` image into it and present.
    ///
    /// `frame` must have been produced by
    /// [`RenderContext::render_frame_gpu`] at the same extent.
    pub fn present(&self, ctx: &RenderContext, frame: &FrameGpu) -> Result<(), RenderError> {
        let (index, suboptimal) = unsafe {
            self.swapchain_ext
                .acquire_next_image2(
                    &vk::AcquireNextImageInfoKHR::default()
                        .swapchain(self.handle)
                        .timeout(u64::MAX)
                        .semaphore(self.image_ready)
                        .fence(vk::Fence::null()),
                )
                .map_err(|e| RenderError::new(format!("acquire_next_image: {e}")))?
        };
        if suboptimal {
            return Err(RenderError::new("swapchain suboptimal: recreate it"));
        }
        unsafe {
            ctx.device
                .wait_for_fences(&[self.present_ready], true, u64::MAX)
                .map_err(|e| RenderError::new(format!("wait present fence: {e}")))?;
            ctx.device
                .reset_fences(&[self.present_ready])
                .map_err(|e| RenderError::new(format!("reset present fence: {e}")))?;
        }
        let cmd = self.present_cmd;
        unsafe {
            // Safe to reset: the `present_ready` wait above proves the previous
            // submit that recorded this buffer has completed.
            ctx.device
                .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())
                .map_err(|e| RenderError::new(format!("reset cmd buffer: {e}")))?;
        }
        let region = vk::ImageBlit {
            src_subresource: vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            },
            src_offsets: [
                vk::Offset3D { x: 0, y: 0, z: 0 },
                vk::Offset3D {
                    x: self.extent.width as i32,
                    y: self.extent.height as i32,
                    z: 1,
                },
            ],
            dst_subresource: vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            },
            dst_offsets: [
                vk::Offset3D { x: 0, y: 0, z: 0 },
                vk::Offset3D {
                    x: self.extent.width as i32,
                    y: self.extent.height as i32,
                    z: 1,
                },
            ],
        };
        unsafe {
            ctx.device
                .begin_command_buffer(
                    cmd,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .map_err(|e| RenderError::new(format!("begin: {e}")))?;
            // The offscreen `out` image is in TRANSFER_SRC_OPTIMAL after the
            // readback copy; the swapchain image is UNDEFINED until we blit.
            ctx.device.cmd_blit_image(
                cmd,
                frame.out.handle,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                self.images[index as usize],
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[region],
                vk::Filter::NEAREST,
            );
            // A swapchain image may only be presented from PRESENT_SRC_KHR, so
            // transition it out of TRANSFER_DST_OPTIMAL (where the blit left it)
            // before `queue_present` waits on `present_done`. Presenting from
            // any other layout is undefined behaviour and validation-error
            // territory; the barrier makes the blit's TRANSFER_WRITE visible to
            // the presentation engine.
            ctx.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[vk::ImageMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::empty())
                    .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                    .image(self.images[index as usize])
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    })],
            );
            ctx.device
                .end_command_buffer(cmd)
                .map_err(|e| RenderError::new(format!("end: {e}")))?;
            let wait = [self.image_ready];
            let signal = [self.present_done];
            let blit = vk::SubmitInfo::default()
                .wait_semaphores(&wait)
                .wait_dst_stage_mask(&[vk::PipelineStageFlags::TRANSFER])
                .command_buffers(std::slice::from_ref(&cmd))
                .signal_semaphores(&signal);
            ctx.device
                .queue_submit(ctx.queue, std::slice::from_ref(&blit), self.present_ready)
                .map_err(|e| RenderError::new(format!("blit submit: {e}")))?;
            let present = vk::PresentInfoKHR::default()
                .wait_semaphores(std::slice::from_ref(&self.present_done))
                .swapchains(std::slice::from_ref(&self.handle))
                .image_indices(std::slice::from_ref(&index));
            let result = self.swapchain_ext.queue_present(ctx.queue, &present);
            ctx.device
                .wait_for_fences(&[self.present_ready], true, u64::MAX)
                .map_err(|e| RenderError::new(format!("wait present fence: {e}")))?;
            match result {
                Ok(false) | Ok(true) => Ok(()),
                Err(e) => Err(RenderError::new(format!("queue_present: {e}"))),
            }
        }
    }
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_semaphore(self.present_done, None);
            self.device.destroy_semaphore(self.image_ready, None);
            self.device.destroy_fence(self.present_ready, None);
            self.device.destroy_command_pool(self.cmd_pool, None);
            for view in &self.views {
                self.device.destroy_image_view(*view, None);
            }
            self.swapchain_ext.destroy_swapchain(self.handle, None);
            self.surface_ext.destroy_surface(self.surface, None);
        }
    }
}
