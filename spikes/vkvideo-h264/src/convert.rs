//! RGBA -> NV12 conversion: a compute pass writing two linear planes, then
//! two buffer->image copies into the multiplanar encode source image.
//!
//! The compute pass runs on a compute-capable queue family: RADV's video
//! encode family advertises *only* `VK_QUEUE_VIDEO_ENCODE_BIT_KHR`.

use anyhow::{Context, Result};
use ash::vk;

use crate::dev::Ctx;
use crate::frames::{HEIGHT, WIDTH};
use crate::vkext;

pub struct Converter {
    pub pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
    pub ds_layout: vk::DescriptorSetLayout,
    pub pool: vk::DescriptorPool,
    pub src: crate::dev::Buf,
    pub y: crate::dev::Buf,
    pub uv: crate::dev::Buf,
    pub cmd: vk::CommandBuffer,
    pub fence: vk::Fence,
    pub set: vk::DescriptorSet,
    pub pool_cmd: vk::CommandPool,
}

const Y_SIZE: vk::DeviceSize = (WIDTH as u64) * (HEIGHT as u64);
const UV_SIZE: vk::DeviceSize = (WIDTH as u64) * (HEIGHT as u64) / 2;
const RGBA_SIZE: vk::DeviceSize = (WIDTH as u64) * (HEIGHT as u64) * 4;

impl Converter {
    pub fn new(ctx: &Ctx, spv: &[u32]) -> Result<Converter> {
        let d = &ctx.device;
        unsafe {
            let shader = d.create_shader_module(
                &vk::ShaderModuleCreateInfo::default().code(spv),
                None,
            )?;

            let bindings = [
                vk::DescriptorSetLayoutBinding::default()
                    .binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE),
                vk::DescriptorSetLayoutBinding::default()
                    .binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE),
                vk::DescriptorSetLayoutBinding::default()
                    .binding(2)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE),
            ];
            let ds_layout = d
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                    None,
                )
                .context("descriptor set layout")?;
            let layout = d
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default().set_layouts(std::slice::from_ref(&ds_layout)),
                    None,
                )
                .context("pipeline layout")?;
            let pipeline = d
                .create_compute_pipelines(
                    vk::PipelineCache::null(),
                    &[vk::ComputePipelineCreateInfo::default()
                        .stage(vk::PipelineShaderStageCreateInfo::default()
                            .stage(vk::ShaderStageFlags::COMPUTE)
                            .module(shader)
                            .name(c"main"))
                        .layout(layout)],
                    None,
                )
                .map_err(|(_, e)| anyhow::anyhow!("create_compute_pipelines: {e:?}"))?[0];
            d.destroy_shader_module(shader, None);

            let pool = d
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .max_sets(1)
                        .pool_sizes(&[vk::DescriptorPoolSize::default()
                            .ty(vk::DescriptorType::STORAGE_BUFFER)
                            .descriptor_count(3)]),
                    None,
                )
                .context("descriptor pool")?;
            let sets = d
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(pool)
                        .set_layouts(std::slice::from_ref(&ds_layout)),
                )
                .context("allocate descriptor set")?;
            let set = sets[0];

            let src = ctx
                .create_buffer(
                    RGBA_SIZE,
                    vk::BufferUsageFlags::STORAGE_BUFFER,
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                )
                .context("rgba staging buffer")?;
            let y = ctx.create_buffer(Y_SIZE, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
                vk::MemoryPropertyFlags::DEVICE_LOCAL)?;
            let uv = ctx.create_buffer(UV_SIZE, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
                vk::MemoryPropertyFlags::DEVICE_LOCAL)?;

            // Named locals: `buffer_info` stores the pointer, so the arrays must
            // outlive `bufs`.
            let bi_src = [vk::DescriptorBufferInfo::default()
                .buffer(src.buffer)
                .offset(0)
                .range(vk::WHOLE_SIZE)];
            let bi_y = [vk::DescriptorBufferInfo::default()
                .buffer(y.buffer)
                .offset(0)
                .range(vk::WHOLE_SIZE)];
            let bi_uv = [vk::DescriptorBufferInfo::default()
                .buffer(uv.buffer)
                .offset(0)
                .range(vk::WHOLE_SIZE)];
            let bufs = [
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&bi_src),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&bi_y),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(2)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&bi_uv),
            ];
            d.update_descriptor_sets(&bufs, &[]);

            let pool_cmd = d.create_command_pool(
                &vk::CommandPoolCreateInfo::default().queue_family_index(ctx.gfx_family),
                None,
            )?;
            let cmd = d
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(pool_cmd)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )
                .context("allocate command buffer")?[0];
            let fence = d.create_fence(&vk::FenceCreateInfo::default(), None)?;

            Ok(Converter {
                pipeline,
                layout,
                ds_layout,
                pool,
                src,
                y,
                uv,
                cmd,
                fence,
                set,
                pool_cmd,
            })
        }
    }

    /// Convert one RGBA frame into the two planes of `dst` (an NV12 image in
    /// `VIDEO_ENCODE_DST/SRC` layout). Returns the GPU time of the pass.
    pub fn convert_frame(&self, ctx: &Ctx, rgba: &[u8], dst: vk::Image) -> Result<()> {
        let d = &ctx.device;
        unsafe {
            // Upload the RGBA bytes.
            let p = d.map_memory(self.src.memory, 0, rgba.len() as u64, vk::MemoryMapFlags::empty())?;
            std::ptr::copy_nonoverlapping(rgba.as_ptr(), p as *mut u8, rgba.len());
            d.unmap_memory(self.src.memory);

            d.reset_fences(&[self.fence])?;
            d.reset_command_buffer(self.cmd, vk::CommandBufferResetFlags::empty())?;
            d.begin_command_buffer(
                self.cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;

            d.cmd_bind_pipeline(
                self.cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline,
            );
            d.cmd_bind_descriptor_sets(
                self.cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.layout,
                0,
                &[self.set],
                &[],
            );
            // One invocation per 4x2 pixel block: 160 word-columns / 16 and
            // 480 rows / (8 * 2) workgroups.
            d.cmd_dispatch(self.cmd, (WIDTH / 4) / 16, HEIGHT / (8 * 2), 1);

            // The compute writes must be visible to the transfer stage.
            let bar = vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                .dst_access_mask(vk::AccessFlags2::TRANSFER_READ);
            d.cmd_pipeline_barrier2(self.cmd, &vk::DependencyInfo::default().memory_barriers(&[bar]));

            // Linear staging buffers -> the two NV12 planes.
            d.cmd_copy_buffer_to_image2(
                self.cmd,
                &vk::CopyBufferToImageInfo2::default()
                    .src_buffer(self.y.buffer)
                    .dst_image(dst)
                    .dst_image_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .regions(&[vk::BufferImageCopy2::default()
                        .buffer_offset(0)
                        .buffer_row_length(WIDTH)
                        .buffer_image_height(HEIGHT)
                        .image_subresource(vk::ImageSubresourceLayers {
                            aspect_mask: vk::ImageAspectFlags::PLANE_0,
                            mip_level: 0,
                            base_array_layer: 0,
                            layer_count: 1,
                        })
                        .image_offset(vk::Offset3D::default())
                        .image_extent(vk::Extent3D {
                            width: WIDTH,
                            height: HEIGHT,
                            depth: 1,
                        })]),
            );
            d.cmd_copy_buffer_to_image2(
                self.cmd,
                &vk::CopyBufferToImageInfo2::default()
                    .src_buffer(self.uv.buffer)
                    .dst_image(dst)
                    .dst_image_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .regions(&[vk::BufferImageCopy2::default()
                        .buffer_offset(0)
                        .buffer_row_length(WIDTH)
                        .buffer_image_height(HEIGHT / 2)
                        .image_subresource(vk::ImageSubresourceLayers {
                            aspect_mask: vk::ImageAspectFlags::PLANE_1,
                            mip_level: 0,
                            base_array_layer: 0,
                            layer_count: 1,
                        })
                        .image_offset(vk::Offset3D::default())
                        .image_extent(vk::Extent3D {
                            width: WIDTH,
                            height: HEIGHT / 2,
                            depth: 1,
                        })]),
            );

            // The encode engine reads the source in VIDEO_ENCODE_SRC layout.
            let to_encode = vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .dst_stage_mask(vkext::PIPELINE_STAGE_2_VIDEO_ENCODE_BIT_KHR)
                .dst_access_mask(vkext::ACCESS_2_VIDEO_ENCODE_READ_BIT_KHR)
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::VIDEO_ENCODE_SRC_KHR)
                .image(dst)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::PLANE_0
                        | vk::ImageAspectFlags::PLANE_1,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });
            d.cmd_pipeline_barrier2(
                self.cmd,
                &vk::DependencyInfo::default().image_memory_barriers(&[to_encode]),
            );

            d.end_command_buffer(self.cmd)?;
            d.queue_submit(
                ctx.gfx_queue,
                &[vk::SubmitInfo::default().command_buffers(&[self.cmd])],
                self.fence,
            )?;
            d.wait_for_fences(&[self.fence], true, u64::MAX)?;
        }
        Ok(())
    }
}
