//! Hand-written bindings for the pieces ash 0.38.0 does not expose.
//!
//! ash 0.38.0+1.3.281 generates the *structures* for the `VK_KHR_video_*`
//! extensions and the raw function pointers inside
//! `ash::extensions::khr::video_queue::{Instance, Device}` and
//! `video_encode_queue::{Instance, Device}`, but it generates **no safe wrapper
//! methods** for any of them (there is no `src/extensions/khr/video_*.rs`). So
//! every video entry point has to be called through the raw `fp()` pointers,
//! and a few enum constants are missing entirely and are re-declared here.
//!
//! Field order and types were checked against Khronos Vulkan-Headers v1.3.281
//! (`vulkan_core.h`, `vk_video/vulkan_video_codec_h264std_encode.h`) — the same
//! header set ash 0.38 is generated from — and against the Mesa 25.0.7 RADV
//! sources (`radv_video.c`, `radv_video_enc.c`, `radv_query.c`).

use std::ffi::c_void;

use ash::vk;

/// `VK_VIDEO_CODEC_OPERATION_ENCODE_H264_BIT_KHR`.
///
/// ash 0.38 generates `VideoCodecOperationFlagsKHR` with only a `NONE`
/// constant, so the codec-operation bits must be spelled out by hand.
pub const VIDEO_CODEC_OPERATION_ENCODE_H264_BIT_KHR: vk::VideoCodecOperationFlagsKHR =
    vk::VideoCodecOperationFlagsKHR::from_raw(0x0001_0000);

/// `VK_QUEUE_VIDEO_ENCODE_BIT_KHR`.
pub const QUEUE_VIDEO_ENCODE_BIT_KHR: vk::QueueFlags =
    vk::QueueFlags::from_raw(0x0000_0040);

/// `VK_PIPELINE_STAGE_2_VIDEO_ENCODE_BIT_KHR`.
pub const PIPELINE_STAGE_2_VIDEO_ENCODE_BIT_KHR: vk::PipelineStageFlags2 =
    vk::PipelineStageFlags2::from_raw(0x0000_0000_0800_0000);

/// `VK_ACCESS_2_VIDEO_ENCODE_READ_BIT_KHR`.
pub const ACCESS_2_VIDEO_ENCODE_READ_BIT_KHR: vk::AccessFlags2 =
    vk::AccessFlags2::from_raw(0x0000_0020_0000_0000);

/// `VK_ACCESS_2_VIDEO_ENCODE_WRITE_BIT_KHR`.
pub const ACCESS_2_VIDEO_ENCODE_WRITE_BIT_KHR: vk::AccessFlags2 =
    vk::AccessFlags2::from_raw(0x0000_0040_0000_0000);

/// `VK_VIDEO_SESSION_CREATE_INLINE_QUERIES_BIT_KHR` (from
/// `VK_KHR_video_maintenance1`). ash 0.38 only generates `PROTECTED_CONTENT`
/// for `VkVideoSessionCreateFlagBitsKHR`.
pub const VIDEO_SESSION_CREATE_INLINE_QUERIES_BIT_KHR: vk::VideoSessionCreateFlagsKHR =
    vk::VideoSessionCreateFlagsKHR::from_raw(0b100);

/// `VK_QUERY_TYPE_RESULT_STATUS_ONLY_KHR`.
pub const QUERY_TYPE_RESULT_STATUS_ONLY_KHR: vk::QueryType =
    vk::QueryType::from_raw(1_000_023_000);

/// `VK_QUERY_RESULT_WITH_STATUS_BIT_KHR`. ash 0.38 *does* generate this one as
/// `QueryResultFlags::WITH_STATUS_KHR`; re-exported under the C name so the
/// call sites read like the spec.
pub const QUERY_RESULT_WITH_STATUS_BIT_KHR: vk::QueryResultFlags =
    vk::QueryResultFlags::WITH_STATUS_KHR;

/// `StdVideoH264ProfileIdc` / `LevelIdc` / `PictureType` / `SliceType` values.
pub mod h264std {
    use ash::vk::native::{
        StdVideoH264LevelIdc, StdVideoH264PictureType, StdVideoH264ProfileIdc,
        StdVideoH264SliceType,
    };

    pub const PROFILE_BASELINE: StdVideoH264ProfileIdc = 66;
    pub const PROFILE_MAIN: StdVideoH264ProfileIdc = 77;

    pub const LEVEL_3_1: StdVideoH264LevelIdc = 8;
    pub const LEVEL_4_0: StdVideoH264LevelIdc = 10;

    pub const PIC_P: StdVideoH264PictureType = 0;
    pub const PIC_I: StdVideoH264PictureType = 2;
    pub const PIC_IDR: StdVideoH264PictureType = 5;

    pub const SLICE_P: StdVideoH264SliceType = 0;
    pub const SLICE_I: StdVideoH264SliceType = 2;
}

/// A zeroed value of a `vk::native` (bindgen) H.264 struct.
///
/// ash 0.38 does not derive `Default` for any of the `StdVideo*` types, and
/// their bitfield unions have no `Default` either. Every one of them is plain
/// integer data, so an all-zero bit pattern is a valid value.
pub fn zeroed<T>() -> T {
    unsafe { std::mem::zeroed() }
}

/// Link one Vulkan structure into another's `pNext` list by hand.
///
/// Needed for exactly two links, because ash 0.38.0 declares **no**
/// `ExtendsVideoEncodeRateControlInfoKHR` and **no**
/// `ExtendsVideoEncodeCapabilitiesKHR` trait, so
/// `VkVideoEncodeH264RateControlInfoKHR` cannot be pushed onto
/// `VkVideoEncodeRateControlInfoKHR` and `VkVideoEncodeH264CapabilitiesKHR`
/// cannot be pushed onto `VkVideoEncodeCapabilitiesKHR` with `push_next`.
/// Every other link in this spike uses the generated `push_next`.
///
/// # Safety
/// `T` must be a Vulkan structure with the standard `sType`/`pNext` prefix (all
/// of them have it, at the same offset), and the object behind `tail` must
/// outlive `head`.
pub unsafe fn chain<T>(head: &mut T, tail: *const c_void) {
    let offset = std::mem::offset_of!(vk::VideoEncodeRateControlInfoKHR, p_next);
    let p: *mut T = head;
    *(p.cast::<u8>().add(offset).cast::<*const c_void>()) = tail;
}

/// Thin typed table over the raw `VK_KHR_video_queue` /
/// `VK_KHR_video_encode_queue` entry points, which ash 0.38 leaves as bare
/// function pointers with no wrapper methods.
#[derive(Clone)]
pub struct VideoFns {
    pub get_physical_device_video_capabilities_khr:
        vk::PFN_vkGetPhysicalDeviceVideoCapabilitiesKHR,
    pub get_physical_device_video_format_properties_khr:
        vk::PFN_vkGetPhysicalDeviceVideoFormatPropertiesKHR,
    pub get_physical_device_video_encode_quality_level_properties_khr:
        vk::PFN_vkGetPhysicalDeviceVideoEncodeQualityLevelPropertiesKHR,
    pub create_video_session_khr: vk::PFN_vkCreateVideoSessionKHR,
    pub destroy_video_session_khr: vk::PFN_vkDestroyVideoSessionKHR,
    pub get_video_session_memory_requirements_khr:
        vk::PFN_vkGetVideoSessionMemoryRequirementsKHR,
    pub bind_video_session_memory_khr: vk::PFN_vkBindVideoSessionMemoryKHR,
    pub create_video_session_parameters_khr: vk::PFN_vkCreateVideoSessionParametersKHR,
    pub update_video_session_parameters_khr: vk::PFN_vkUpdateVideoSessionParametersKHR,
    pub destroy_video_session_parameters_khr: vk::PFN_vkDestroyVideoSessionParametersKHR,
    pub cmd_begin_video_coding_khr: vk::PFN_vkCmdBeginVideoCodingKHR,
    pub cmd_end_video_coding_khr: vk::PFN_vkCmdEndVideoCodingKHR,
    pub cmd_control_video_coding_khr: vk::PFN_vkCmdControlVideoCodingKHR,
    pub get_encoded_video_session_parameters_khr:
        vk::PFN_vkGetEncodedVideoSessionParametersKHR,
    pub cmd_encode_video_khr: vk::PFN_vkCmdEncodeVideoKHR,
    /// Names of the entry points the loader returned NULL for.
    pub missing: Vec<&'static str>,
}

impl VideoFns {
    /// Entry points the loader refused to hand out (empty = full support).
    pub fn missing(&self) -> &[&'static str] {
        &self.missing
    }
}

/// Resolve one instance-level entry point, recording a NULL result.
fn inst_pfn(
    entry: &ash::Entry,
    instance: vk::Instance,
    name: &'static str,
    missing: &mut Vec<&'static str>,
) -> *const c_void {
    let cname = std::ffi::CString::new(name).unwrap();
    let p = unsafe { entry.get_instance_proc_addr(instance, cname.as_ptr()) };
    let p = p.map(|f| f as *const c_void).unwrap_or(std::ptr::null());
    if p.is_null() {
        missing.push(name);
    }
    p
}

/// Resolve one device-level entry point, recording a NULL result.
fn dev_pfn(
    instance: &ash::Instance,
    device: vk::Device,
    name: &'static str,
    missing: &mut Vec<&'static str>,
) -> *const c_void {
    let cname = std::ffi::CString::new(name).unwrap();
    let p = unsafe { instance.get_device_proc_addr(device, cname.as_ptr()) };
    let p = p.map(|f| f as *const c_void).unwrap_or(std::ptr::null());
    if p.is_null() {
        missing.push(name);
    }
    p
}

impl VideoFns {
    /// Load every video entry point and report which ones the loader did *not*
    /// hand out. An empty `missing` list means the driver exposes the full set
    /// of entry points this spike needs.
    pub fn load(
        entry: &ash::Entry,
        instance: &ash::Instance,
        device: ash::Device,
    ) -> (VideoFns, Vec<&'static str>) {
        let raw_i = instance.handle();
        let raw_d = device.handle();
        let mut missing: Vec<&'static str> = Vec::new();

        macro_rules! ip {
            ($name:literal) => {{
                let p = inst_pfn(entry, raw_i, $name, &mut missing);
                unsafe { std::mem::transmute::<*const c_void, _>(p) }
            }};
        }
        macro_rules! dp {
            ($name:literal) => {{
                let p = dev_pfn(instance, raw_d, $name, &mut missing);
                unsafe { std::mem::transmute::<*const c_void, _>(p) }
            }};
        }

        let fns = VideoFns {
            get_physical_device_video_capabilities_khr: ip!(
                "vkGetPhysicalDeviceVideoCapabilitiesKHR"
            ),
            get_physical_device_video_format_properties_khr: ip!(
                "vkGetPhysicalDeviceVideoFormatPropertiesKHR"
            ),
            get_physical_device_video_encode_quality_level_properties_khr: ip!(
                "vkGetPhysicalDeviceVideoEncodeQualityLevelPropertiesKHR"
            ),
            create_video_session_khr: dp!("vkCreateVideoSessionKHR"),
            destroy_video_session_khr: dp!("vkDestroyVideoSessionKHR"),
            get_video_session_memory_requirements_khr: dp!(
                "vkGetVideoSessionMemoryRequirementsKHR"
            ),
            bind_video_session_memory_khr: dp!("vkBindVideoSessionMemoryKHR"),
            create_video_session_parameters_khr: dp!("vkCreateVideoSessionParametersKHR"),
            update_video_session_parameters_khr: dp!("vkUpdateVideoSessionParametersKHR"),
            destroy_video_session_parameters_khr: dp!("vkDestroyVideoSessionParametersKHR"),
            cmd_begin_video_coding_khr: dp!("vkCmdBeginVideoCodingKHR"),
            cmd_end_video_coding_khr: dp!("vkCmdEndVideoCodingKHR"),
            cmd_control_video_coding_khr: dp!("vkCmdControlVideoCodingKHR"),
            get_encoded_video_session_parameters_khr: dp!(
                "vkGetEncodedVideoSessionParametersKHR"
            ),
            cmd_encode_video_khr: dp!("vkCmdEncodeVideoKHR"),
            missing,
        };
        let missing = fns.missing.clone();
        (fns, missing)
    }
}

/// Read the driver-generated SPS and PPS NAL units out of a video session
/// parameter set with `vkGetEncodedVideoSessionParametersKHR`.
///
/// Mesa emits each NAL *without* a start code (`vk_video_encode_h264_sps`
/// starts writing at the NAL header byte), so the caller prepends the Annex-B
/// start codes. Returns `(sps_nal, pps_nal)`.
///
/// # Safety
/// `params` must be a live `VkVideoSessionParametersKHR` created for an H.264
/// *encode* session, and `device` must be the device it was created on.
pub unsafe fn read_h264_parameter_sets(
    fns: &VideoFns,
    device: vk::Device,
    params: vk::VideoSessionParametersKHR,
) -> Result<(Vec<u8>, Vec<u8>), vk::Result> {
    // Size query (pData == NULL). The chain is rebuilt per call because
    // `push_next` mutably borrows the pushed struct, and the borrowed struct
    // must outlive the chain head.
    let mut size = 0usize;
    let r = {
        let mut h264_get = vk::VideoEncodeH264SessionParametersGetInfoKHR::default()
            .write_std_sps(true)
            .write_std_pps(true);
        let get = vk::VideoEncodeSessionParametersGetInfoKHR::default()
            .video_session_parameters(params)
            .push_next(&mut h264_get);
        (fns.get_encoded_video_session_parameters_khr)(
            device,
            &get,
            std::ptr::null_mut(),
            &mut size,
            std::ptr::null_mut(),
        )
    };
    if r != vk::Result::SUCCESS {
        return Err(r);
    }
    let cap = size.max(1);

    let mut out = vec![0u8; cap];
    let mut read_one = |want_sps: bool| -> Result<Vec<u8>, vk::Result> {
        let mut h264_get = vk::VideoEncodeH264SessionParametersGetInfoKHR::default()
            .write_std_sps(want_sps)
            .write_std_pps(!want_sps);
        let get = vk::VideoEncodeSessionParametersGetInfoKHR::default()
            .video_session_parameters(params)
            .push_next(&mut h264_get);
        let mut n = out.len();
        let r = (fns.get_encoded_video_session_parameters_khr)(
            device,
            &get,
            std::ptr::null_mut(),
            &mut n,
            out.as_mut_ptr() as *mut c_void,
        );
        if r != vk::Result::SUCCESS {
            return Err(r);
        }
        Ok(out[..n].to_vec())
    };

    let sps = read_one(true)?;
    let pps = read_one(false)?;
    Ok((sps, pps))
}
