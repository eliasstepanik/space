//! Instance / device creation, queue discovery and small allocation helpers.

use anyhow::{Context, Result, bail};
use ash::vk;

use crate::vkext;

pub const VIDEO_QUEUE_EXTS: &[&std::ffi::CStr] = &[
    ash::vk::KHR_VIDEO_QUEUE_NAME,
    ash::vk::KHR_VIDEO_ENCODE_QUEUE_NAME,
    ash::vk::KHR_VIDEO_ENCODE_H264_NAME,
    ash::vk::KHR_VIDEO_MAINTENANCE1_NAME,
];

/// `enabled_extension_names` wants `&[*const c_char]`.
fn ext_ptrs() -> Vec<*const std::ffi::c_char> {
    VIDEO_QUEUE_EXTS.iter().map(|c| c.as_ptr()).collect()
}

pub struct Ctx {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub device: ash::Device,
    pub pd: vk::PhysicalDevice,
    pub queue_family: u32,
    pub queue: vk::Queue,
    /// A family that can run compute. RADV's video-encode family advertises
    /// *only* `VK_QUEUE_VIDEO_ENCODE_BIT_KHR`, so the RGBA->NV12 pass cannot
    /// run on it and needs a second queue.
    pub gfx_family: u32,
    pub gfx_queue: vk::Queue,
    pub props: vk::PhysicalDeviceProperties,
}

impl Drop for Ctx {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

/// The device extensions the physical device actually exposes, as names.
pub fn device_extensions(inst: &ash::Instance, pd: vk::PhysicalDevice) -> Result<Vec<String>> {
    let all = unsafe { inst.enumerate_device_extension_properties(pd) }?;
    Ok(all
        .iter()
        .map(|e| unsafe { std::ffi::CStr::from_ptr(e.extension_name.as_ptr()) })
        .map(|s| s.to_string_lossy().into_owned())
        .collect())
}

/// Create the instance and pick a physical device that exposes the video-encode
/// device extensions *and* an H.264 video-encode queue family.
pub fn create(want_device: Option<&str>) -> Result<Ctx> {
    let entry = unsafe { ash::Entry::load() }.context("vulkan loader: is LD_LIBRARY_PATH set?")?;

    // `VK_KHR_video_*` are *device* extensions: nothing is enabled on the
    // instance.
    let app = vk::ApplicationInfo::default()
        .application_name(c"vkvideo-h264")
        .application_version(1)
        .engine_name(c"spike")
        .engine_version(1)
        .api_version(vk::make_api_version(0, 1, 3, 0));
    let inst = unsafe {
        entry.create_instance(
            &vk::InstanceCreateInfo::default().application_info(&app),
            None,
        )
    }?;

    let pds = unsafe { inst.enumerate_physical_devices() }?;
    let mut chosen = None;
    let mut report = Vec::new();
    for pd in pds {
        let name = unsafe { inst.get_physical_device_properties(pd) }.device_name
            [..64]
            .iter()
            .take_while(|b| **b != 0)
            .map(|b| *b as u8)
            .collect::<Vec<_>>();
        let name = String::from_utf8_lossy(&name).into_owned();
        if let Some(w) = want_device {
            if !name.contains(w) {
                continue;
            }
        }
        let exts = device_extensions(&inst, pd).unwrap_or_default();
        let missing: Vec<&str> = VIDEO_QUEUE_EXTS
            .iter()
            .map(|c| c.to_str().unwrap())
            .filter(|n| !exts.contains(&n.to_string()))
            .collect();
        if !missing.is_empty() {
            report.push(format!(
                "{name}: {} device extensions missing ({} advertised): {}",
                missing.len(),
                exts.len(),
                missing.join(", ")
            ));
            continue;
        }
        match find_encode_queue_family(&inst, pd) {
            Ok(qf) => {
                report.push(format!("{name}: encode queue family {qf}"));
                if chosen.is_none() {
                    chosen = Some((pd, qf, name));
                }
            }
            Err(e) => report.push(format!("{name}: {e}")),
        }
    }
    let Some((pd, queue_family, name)) = chosen else {
        bail!(
            "no physical device exposes an H.264 video-encode queue family. Report:\n  {}",
            report.join("\n  ")
        );
    };
    eprintln!("[dev] using {name} (queue family {queue_family})");

    let props = unsafe { inst.get_physical_device_properties(pd) };
    let gfx_family = pick_compute_family(&inst, pd)?;
    let prio = [1.0f32; 1];
    let mut queues = vec![vk::DeviceQueueCreateInfo::default()
        .queue_family_index(queue_family)
        .queue_priorities(&prio)];
    let mut gfx_prio = [1.0f32; 1];
    if gfx_family != queue_family {
        queues.push(
            vk::DeviceQueueCreateInfo::default()
                .queue_family_index(gfx_family)
                .queue_priorities(&mut gfx_prio),
        );
    }
    let mut maintenance1 = vk::PhysicalDeviceVideoMaintenance1FeaturesKHR::default()
        .video_maintenance1(true);
    let mut feats = vk::PhysicalDeviceFeatures2::default().push_next(&mut maintenance1);
    let exts = ext_ptrs();
    let ci = vk::DeviceCreateInfo::default()
        .queue_create_infos(&queues)
        .enabled_extension_names(&exts)
        .push_next(&mut feats);
    let device = unsafe { inst.create_device(pd, &ci, None) }?;
    let queue = unsafe { device.get_device_queue(queue_family, 0) };
    let gfx_queue = unsafe { device.get_device_queue(gfx_family, 0) };

    Ok(Ctx {
        entry,
        instance: inst,
        device,
        pd,
        queue_family,
        queue,
        gfx_family,
        gfx_queue,
        props,
    })
}

/// First queue family with the COMPUTE bit (RADV family 0: GRAPHICS|COMPUTE).
fn pick_compute_family(inst: &ash::Instance, pd: vk::PhysicalDevice) -> Result<u32> {
    let n = unsafe { inst.get_physical_device_queue_family_properties2_len(pd) };
    let props = unsafe {
        let mut v = vec![vk::QueueFamilyProperties2::default(); n];
        inst.get_physical_device_queue_family_properties2(pd, &mut v);
        v
    };
    props
        .iter()
        .enumerate()
        .find(|(_, p)| {
            p.queue_family_properties
                .queue_flags
                .intersects(vk::QueueFlags::COMPUTE)
        })
        .map(|(i, _)| i as u32)
        .context("no compute-capable queue family")
}

/// Find a queue family that both carries `VK_QUEUE_VIDEO_ENCODE_BIT_KHR` and
/// advertises `VK_VIDEO_CODEC_OPERATION_ENCODE_H264_BIT_KHR`.
fn find_encode_queue_family(inst: &ash::Instance, pd: vk::PhysicalDevice) -> Result<u32> {
    let n = unsafe { inst.get_physical_device_queue_family_properties2_len(pd) };
    // The video properties live in `Box`es so the addresses written into
    // `p_next` stay valid while the driver fills them in. The links are made by
    // hand (raw pointers) because `push_next`'s lifetime would keep `video`
    // mutably borrowed for as long as `props` lives.
    let video: Vec<Box<vk::QueueFamilyVideoPropertiesKHR>> = (0..n)
        .map(|_| Box::new(vk::QueueFamilyVideoPropertiesKHR::default()))
        .collect();
    let mut props: Vec<vk::QueueFamilyProperties2> = video
        .iter()
        .map(|v| {
            let mut p = vk::QueueFamilyProperties2::default();
            p.p_next = Box::as_ptr(v) as *mut _;
            p
        })
        .collect();
    unsafe { inst.get_physical_device_queue_family_properties2(pd, &mut props) };

    let mut encode_only = None;
    for (i, p) in props.iter().enumerate() {
        let qf = p.queue_family_properties.queue_flags;
        let ops = video[i].video_codec_operations;
        if qf.intersects(vk::QueueFlags::VIDEO_ENCODE_KHR) {
            if ops.intersects(vkext::VIDEO_CODEC_OPERATION_ENCODE_H264_BIT_KHR) {
                return Ok(i as u32);
            }
            encode_only = Some(format!(
                "family {i} has VIDEO_ENCODE but videoCodecOperations={ops:?}"
            ));
        }
    }
    match encode_only {
        Some(s) => bail!("VIDEO_ENCODE queue found but no H.264 codec op: {s}"),
        None => bail!("no queue family has VK_QUEUE_VIDEO_ENCODE_BIT_KHR ({} families)", n),
    }
}

/// Memory type index that satisfies `reqs` and one of `props`.
pub fn find_memory_type(ctx: &Ctx, reqs: vk::MemoryRequirements, props: vk::MemoryPropertyFlags) -> Result<u32> {
    let mem = unsafe { ctx.instance.get_physical_device_memory_properties(ctx.pd) };
    for i in 0..mem.memory_type_count {
        if reqs.memory_type_bits & (1 << i) != 0
            && mem.memory_types[i as usize].property_flags.contains(props)
        {
            return Ok(i);
        }
    }
    bail!("no memory type for bits={:b} props={props:?}", reqs.memory_type_bits)
}

pub struct Buf {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub size: vk::DeviceSize,
}

impl Ctx {
    pub fn create_buffer(
        &self,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        props: vk::MemoryPropertyFlags,
    ) -> Result<Buf> {
        unsafe {
            let buffer = self.device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(size)
                    .usage(usage),
                None,
            )?;
            let reqs = self.device.get_buffer_memory_requirements(buffer);
            let memory = self.device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(reqs.size)
                    .memory_type_index(find_memory_type(self, reqs, props)?),
                None,
            )?;
            self.device.bind_buffer_memory(buffer, memory, 0)?;
            Ok(Buf {
                buffer,
                memory,
                size: reqs.size,
            })
        }
    }

    pub fn destroy_buffer(&self, b: &Buf) {
        unsafe {
            self.device.destroy_buffer(b.buffer, None);
            self.device.free_memory(b.memory, None);
        }
    }

    /// Host-visible buffer pre-filled with `data`.
    pub fn create_host_buffer(&self, data: &[u8], usage: vk::BufferUsageFlags) -> Result<Buf> {
        let b = self.create_buffer(
            (data.len() as u64).max(4),
            usage | vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        unsafe {
            let p = self.device.map_memory(b.memory, 0, data.len() as u64, vk::MemoryMapFlags::empty())?;
            std::ptr::copy_nonoverlapping(data.as_ptr(), p as *mut u8, data.len());
            self.device.unmap_memory(b.memory);
        }
        Ok(b)
    }
}
