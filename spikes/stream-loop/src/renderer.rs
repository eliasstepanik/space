//! Minimal ash offscreen renderer: renders a moving coloured quad to an image,
//! then copies it to a host-visible buffer for readback. No window/surface.
//!
//! WGSL is compiled to SPIR-V with the pure-Rust `naga` crate (no glslang/C).

use ash::vk;
use ash::Entry;

pub const WIDTH: u32 = 640;
pub const HEIGHT: u32 = 480;

const QUAD_WGSL: &str = r#"
struct Uni { offset: f32, _pad: vec3<f32> };
@group(0) @binding(0) var<uniform> uni: Uni;

struct VSOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> };

@vertex fn vs(@builtin(vertex_index) vi: u32) -> VSOut {
  var p = array<vec2<f32>,4>(vec2(-1.0,-1.0), vec2(1.0,-1.0), vec2(-1.0,1.0), vec2(1.0,1.0));
  var o: VSOut;
  let c = p[vi];
  o.pos = vec4(c, 0.0, 1.0);
  o.uv  = c * 0.5 + vec2(0.5, 0.5);
  return o;
}

@fragment fn fs(in: VSOut) -> @location(0) vec4<f32> {
  let x = in.uv.x;
  let y = in.uv.y;
  var col = vec3<f32>(0.10 + 0.25 * x, 0.08, 0.28 + 0.25 * y);
  let cx = fract(uni.offset);
  let half = 0.12;
  if (abs(x - cx) < half && abs(y - 0.5) < half) {
    col = vec3(1.0, 0.30, 0.20);
  }
  return vec4(col, 1.0);
}
"#;

fn compile_entry(src: &str, entry: &str, stage: naga::ShaderStage) -> Vec<u32> {
    let module = naga::front::wgsl::parse_str(src)
        .unwrap_or_else(|e| panic!("WGSL parse error: {}", e.emit_to_string(src)));
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::default(),
    );
    let info = validator
        .validate(&module)
        .unwrap_or_else(|e| panic!("WGSL validate error: {e:?}"));
    let opts = naga::back::spv::Options::default();
    let pipeline = naga::back::spv::PipelineOptions {
        shader_stage: stage,
        entry_point: entry.to_string(),
    };
    let mut words = Vec::new();
    let mut writer = naga::back::spv::Writer::new(&opts).unwrap();
    writer
        .write(&module, &info, Some(&pipeline), &None, &mut words)
        .unwrap();
    words
}

pub struct Renderer {
    entry: Entry,
    instance: ash::Instance,
    device: ash::Device,
    queue: vk::Queue,
    queue_family: u32,
    color_image: vk::Image,
    color_mem: vk::DeviceMemory,
    color_view: vk::ImageView,
    readback: vk::Buffer,
    readback_mem: vk::DeviceMemory,
    readback_ptr: *mut u8,
    uniform: vk::Buffer,
    uniform_mem: vk::DeviceMemory,
    uniform_ptr: *mut u8,
    cmd_pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
    render_pass: vk::RenderPass,
    framebuffer: vk::Framebuffer,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    desc_set: vk::DescriptorSet,
    desc_pool: vk::DescriptorPool,
    fence: vk::Fence,
}

impl Renderer {
    pub fn new() -> Self {
        unsafe {
            let entry = Entry::load().expect("vulkan loader");
            let app = vk::ApplicationInfo::default()
                .application_name(c"stream-loop")
                .api_version(vk::make_api_version(0, 1, 1, 0));
            let instance = entry
                .create_instance(&vk::InstanceCreateInfo::default().application_info(&app), None)
                .expect("instance");
            let pdev = instance
                .enumerate_physical_devices()
                .unwrap()
                .into_iter()
                .find(|&p| cstr_name(&instance.get_physical_device_properties(p).device_name).contains("RADV"))
                .or_else(|| instance.enumerate_physical_devices().unwrap().first().copied())
                .expect("no physical device");
            let props = instance.get_physical_device_properties(pdev);
            eprintln!("[host] using device: {}", cstr_name(&props.device_name));
            let families = instance.get_physical_device_queue_family_properties(pdev);
            let qf = families
                .iter()
                .position(|f| f.queue_flags.contains(vk::QueueFlags::GRAPHICS))
                .expect("no graphics queue") as u32;
            let prio = [1.0f32];
            let ci = vk::DeviceQueueCreateInfo::default()
                .queue_family_index(qf)
                .queue_priorities(&prio);
            let device = instance
                .create_device(pdev, &vk::DeviceCreateInfo::default().queue_create_infos(&[ci]), None)
                .expect("device");
            let queue = device.get_device_queue(qf, 0);
            let mem_props = instance.get_physical_device_memory_properties(pdev);
            let mem_index = |mask: u32, want: vk::MemoryPropertyFlags| -> u32 {
                for (i, t) in mem_props.memory_types[..mem_props.memory_type_count as usize]
                    .iter()
                    .enumerate()
                {
                    if mask & (1 << i) != 0 && t.property_flags.contains(want) {
                        return i as u32;
                    }
                }
                panic!("no memory type");
            };
            // Color image (device local, color attachment + transfer src).
            let color_image = device
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
                        .tiling(vk::ImageTiling::OPTIMAL)
                        .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE),
                    None,
                )
                .unwrap();
            let reqs = device.get_image_memory_requirements(color_image);
            let color_mem = device
                .allocate_memory(
                    &vk::MemoryAllocateInfo::default()
                        .allocation_size(reqs.size)
                        .memory_type_index(mem_index(reqs.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)),
                    None,
                )
                .unwrap();
            device.bind_image_memory(color_image, color_mem, 0).unwrap();
            let color_view = device
                .create_image_view(
                    &vk::ImageViewCreateInfo::default()
                        .image(color_image)
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
                .unwrap();
            // Readback buffer (host-visible, coherent).
            let (readback, readback_mem, readback_ptr) = Self::make_host_buffer(
                &device,
                mem_index,
                (WIDTH * HEIGHT * 4) as usize,
                vk::BufferUsageFlags::TRANSFER_DST,
            );
            // Uniform buffer (host-visible) holding a single f32 offset.
            let (uniform, uniform_mem, uniform_ptr) = Self::make_host_buffer(
                &device,
                mem_index,
                16,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
            );
            // Render pass: single color attachment, load clear, store store.
            let color_att = vk::AttachmentDescription::default()
                .format(vk::Format::R8G8B8A8_UNORM)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .final_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
            let att_ref = vk::AttachmentReference::default()
                .attachment(0)
                .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
            let att_refs = &[att_ref];
            let sub = vk::SubpassDescription::default()
                .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                .color_attachments(att_refs);
            let render_pass = device
                .create_render_pass(
                    &vk::RenderPassCreateInfo::default()
                        .attachments(&[color_att])
                        .subpasses(&[sub]),
                    None,
                )
                .unwrap();
            let framebuffer = device
                .create_framebuffer(
                    &vk::FramebufferCreateInfo::default()
                        .render_pass(render_pass)
                        .attachments(&[color_view])
                        .width(WIDTH)
                        .height(HEIGHT)
                        .layers(1),
                    None,
                )
                .unwrap();
            // Descriptor set: one uniform buffer binding.
            let desc_layout = device
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&[
                        vk::DescriptorSetLayoutBinding::default()
                            .binding(0)
                            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                            .descriptor_count(1)
                            .stage_flags(vk::ShaderStageFlags::VERTEX),
                    ]),
                    None,
                )
                .unwrap();
            let pipeline_layout = device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default().set_layouts(&[desc_layout]),
                    None,
                )
                .unwrap();
            let desc_pool = device
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .max_sets(1)
                        .pool_sizes(&[vk::DescriptorPoolSize {
                            ty: vk::DescriptorType::UNIFORM_BUFFER,
                            descriptor_count: 1,
                        }]),
                    None,
                )
                .unwrap();
            let desc_set = device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(desc_pool)
                        .set_layouts(&[desc_layout]),
                )
                .unwrap()[0];
            let buf_info = vk::DescriptorBufferInfo::default()
                .buffer(uniform)
                .offset(0)
                .range(16);
            device.update_descriptor_sets(
                &[vk::WriteDescriptorSet::default()
                    .dst_set(desc_set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(&[buf_info])],
                &[],
            );
            // Shaders (naga -> SPIR-V).
            let vs = compile_entry(QUAD_WGSL, "vs", naga::ShaderStage::Vertex);
            let fs = compile_entry(QUAD_WGSL, "fs", naga::ShaderStage::Fragment);
            let vs_mod = device
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&vs), None)
                .unwrap();
            let fs_mod = device
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&fs), None)
                .unwrap();
            let stages = [
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::VERTEX)
                    .module(vs_mod)
                    .name(c"vs"),
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::FRAGMENT)
                    .module(fs_mod)
                    .name(c"fs"),
            ];
            let vi_ci = vk::PipelineVertexInputStateCreateInfo::default();
            let ia = vk::PipelineInputAssemblyStateCreateInfo::default()
                .topology(vk::PrimitiveTopology::TRIANGLE_STRIP);
            let viewport = vk::Viewport::default()
                .width(WIDTH as f32)
                .height(HEIGHT as f32)
                .max_depth(1.0);
            let scissor = vk::Rect2D::default().extent(vk::Extent2D {
                width: WIDTH,
                height: HEIGHT,
            });
            let viewports = &[viewport];
            let scissors = &[scissor];
            let vp = vk::PipelineViewportStateCreateInfo::default()
                .viewports(viewports)
                .scissors(scissors);
            let rs = vk::PipelineRasterizationStateCreateInfo::default()
                .line_width(1.0)
                .cull_mode(vk::CullModeFlags::NONE)
                .polygon_mode(vk::PolygonMode::FILL);
            let ms = vk::PipelineMultisampleStateCreateInfo::default()
                .rasterization_samples(vk::SampleCountFlags::TYPE_1);
            let cb = vk::PipelineColorBlendAttachmentState::default()
                .color_write_mask(vk::ColorComponentFlags::RGBA);
            let cbs = &[cb];
            let cbci = vk::PipelineColorBlendStateCreateInfo::default().attachments(cbs);
            let dyn_state = vk::PipelineDynamicStateCreateInfo::default()
                .dynamic_states(&[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR]);
            let pipeline = device
                .create_graphics_pipelines(
                    vk::PipelineCache::null(),
                    &[vk::GraphicsPipelineCreateInfo::default()
                        .stages(&stages)
                        .vertex_input_state(&vi_ci)
                        .input_assembly_state(&ia)
                        .viewport_state(&vp)
                        .rasterization_state(&rs)
                        .multisample_state(&ms)
                        .color_blend_state(&cbci)
                        .dynamic_state(&dyn_state)
                        .layout(pipeline_layout)
                        .render_pass(render_pass)
                        .subpass(0)],
                    None,
                )
                .unwrap()[0];
            device.destroy_shader_module(vs_mod, None);
            device.destroy_shader_module(fs_mod, None);
            let cmd_pool = device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .queue_family_index(qf)
                        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                    None,
                )
                .unwrap();
            let cmd = device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(cmd_pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )
                .unwrap()[0];
            let fence = device.create_fence(&vk::FenceCreateInfo::default(), None).unwrap();
            Renderer {
                entry,
                instance,
                device,
                queue,
                queue_family: qf,
                color_image,
                color_mem,
                color_view,
                readback,
                readback_mem,
                readback_ptr,
                uniform,
                uniform_mem,
                uniform_ptr,
                cmd_pool,
                cmd,
                render_pass,
                framebuffer,
                pipeline,
                pipeline_layout,
                desc_set,
                desc_pool,
                fence,
            }
        }
    }

    unsafe fn make_host_buffer(
        device: &ash::Device,
        mem_index: impl Fn(u32, vk::MemoryPropertyFlags) -> u32,
        size: usize,
        usage: vk::BufferUsageFlags,
    ) -> (vk::Buffer, vk::DeviceMemory, *mut u8) {
        let buf = device
            .create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(size as vk::DeviceSize)
                    .usage(usage),
                None,
            )
            .unwrap();
        let reqs = device.get_buffer_memory_requirements(buf);
        let mem = device
            .allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(reqs.size)
                    .memory_type_index(mem_index(
                        reqs.memory_type_bits,
                        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                    )),
                None,
            )
            .unwrap();
        device.bind_buffer_memory(buf, mem, 0).unwrap();
        let ptr = device
            .map_memory(mem, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
            .unwrap() as *mut u8;
        (buf, mem, ptr)
    }

    /// Render one frame with the quad at `offset` (0..1), read it back into the
    /// host buffer and return a copy of the RGBA pixels.
    pub fn render(&self, offset: f32) -> Vec<u8> {
        unsafe {
            std::ptr::write(self.uniform_ptr as *mut f32, offset);
            let dev = &self.device;
            dev.reset_fences(&[self.fence]).unwrap();
            dev.begin_command_buffer(
                self.cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .unwrap();
            dev.cmd_begin_render_pass(
                self.cmd,
                &vk::RenderPassBeginInfo::default()
                    .render_pass(self.render_pass)
                    .framebuffer(self.framebuffer)
                    .render_area(vk::Rect2D::default().extent(vk::Extent2D {
                        width: WIDTH,
                        height: HEIGHT,
                    }))
                    .clear_values(&[vk::ClearValue {
                        color: vk::ClearColorValue {
                            float32: [0.05, 0.05, 0.10, 1.0],
                        },
                    }]),
                vk::SubpassContents::INLINE,
            );
            dev.cmd_bind_pipeline(
                self.cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline,
            );
            dev.cmd_set_viewport(self.cmd, 0, &[vk::Viewport::default()
                .width(WIDTH as f32)
                .height(HEIGHT as f32)
                .max_depth(1.0)]);
            dev.cmd_set_scissor(self.cmd, 0, &[vk::Rect2D::default().extent(vk::Extent2D {
                width: WIDTH,
                height: HEIGHT,
            })]);
            dev.cmd_bind_descriptor_sets(
                self.cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                &[self.desc_set],
                &[],
            );
            dev.cmd_draw(self.cmd, 4, 1, 0, 0);
            dev.cmd_end_render_pass(self.cmd);
            // Copy color image -> readback buffer.
            dev.cmd_copy_image_to_buffer(
                self.cmd,
                self.color_image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                self.readback,
                &[vk::BufferImageCopy::default().image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image_extent(vk::Extent3D {
                    width: WIDTH,
                    height: HEIGHT,
                    depth: 1,
                })],
            );
            dev.end_command_buffer(self.cmd).unwrap();
            dev.queue_submit(
                self.queue,
                &[vk::SubmitInfo::default().command_buffers(&[self.cmd])],
                self.fence,
            )
            .unwrap();
            dev.wait_for_fences(&[self.fence], true, u64::MAX).unwrap();
            let len = (WIDTH as usize) * (HEIGHT as usize) * 4;
            std::slice::from_raw_parts(self.readback_ptr, len).to_vec()
        }
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        unsafe {
            let d = &self.device;
            d.device_wait_idle().ok();
            d.destroy_fence(self.fence, None);
            d.destroy_command_pool(self.cmd_pool, None);
            d.destroy_pipeline(self.pipeline, None);
            d.destroy_pipeline_layout(self.pipeline_layout, None);
            d.destroy_descriptor_pool(self.desc_pool, None);
            d.destroy_framebuffer(self.framebuffer, None);
            d.destroy_render_pass(self.render_pass, None);
            d.destroy_image_view(self.color_view, None);
            d.destroy_image(self.color_image, None);
            d.free_memory(self.color_mem, None);
            d.destroy_buffer(self.readback, None);
            d.destroy_buffer(self.uniform, None);
            d.free_memory(self.readback_mem, None);
            d.free_memory(self.uniform_mem, None);
            d.destroy_device(None);
            self.instance.destroy_instance(None);
            let _ = &self.entry;
            let _ = self.queue_family;
        }
    }
}

fn cstr_name(name: &[std::ffi::c_char]) -> String {
    let bytes: Vec<u8> = name
        .iter()
        .map(|&c| c as u8)
        .take_while(|&b| b != 0)
        .collect();
    String::from_utf8_lossy(&bytes).into_owned()
}
