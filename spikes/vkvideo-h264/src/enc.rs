//! The Vulkan video H.264 encode session: profile, capabilities, session,
//! session memory and session parameters.
//!
//! Written against the RADV driver's actual expectations, read out of the Mesa
//! 25.0.7 sources (`radv_video.c`, `radv_video_enc.c`, `radv_query.c`):
//!
//! * the encode queue family is a *separate* family advertising only
//!   `VK_QUEUE_VIDEO_ENCODE_BIT_KHR`, so the RGBA->NV12 compute pass runs on a
//!   different queue and the two are separated by a fence;
//! * the source picture must be `VK_FORMAT_G8_B8R8_2PLANE_420_UNORM` (NV12),
//!   which is what `radv_GetPhysicalDeviceVideoFormatPropertiesKHR` returns for
//!   8-bit encode;
//! * the DPB image is indexed per slot via `surf_slice_size`, i.e. one array
//!   layer per DPB slot;
//! * with rate control DISABLED, RADV takes the QP from
//!   `VkVideoEncodeH264NaluSliceInfoKHR::constantQp`;
//! * the bitstream length comes from a `VK_QUERY_TYPE_VIDEO_ENCODE_FEEDBACK_KHR`
//!   query, in the inline-query form enabled by `VK_KHR_video_maintenance1`.

use anyhow::{Context, Result, bail};
use ash::vk;
use std::ffi::c_void;

use crate::dev::Ctx;
use crate::frames::{HEIGHT, WIDTH};
use crate::h264;
use crate::vkext::{self, VideoFns};

pub const QP: i32 = 26;
pub const GOP: u32 = 30;
pub const DPB_SLOTS: u32 = 2;
pub const BITSTREAM_SIZE: vk::DeviceSize = 4 << 20;

/// Owns the whole `VkVideoProfileInfoKHR` `pNext` chain.
///
/// `VkVideoProfileInfoKHR` is stored by value in [`Encoder`] and re-used later
/// for `VkImageCreateInfo`, so its `pNext` pointers must stay valid after the
/// constructor returns. The linked structures therefore live in `Box`es whose
/// addresses are taken once they are in their final home.
pub struct ProfileChain {
    usage: Box<vk::VideoEncodeUsageInfoKHR<'static>>,
    h264: Box<vk::VideoEncodeH264ProfileInfoKHR<'static>>,
    pub profile: vk::VideoProfileInfoKHR<'static>,
}

impl ProfileChain {
    fn new() -> ProfileChain {
        let usage = Box::new(
            vk::VideoEncodeUsageInfoKHR::default()
                .video_usage_hints(vk::VideoEncodeUsageFlagsKHR::STREAMING)
                .video_content_hints(vk::VideoEncodeContentFlagsKHR::RENDERED)
                .tuning_mode(vk::VideoEncodeTuningModeKHR::LOW_LATENCY),
        );
        let h264 = Box::new(
            vk::VideoEncodeH264ProfileInfoKHR::default()
                .std_profile_idc(vkext::h264std::PROFILE_BASELINE),
        );
        let mut profile = vk::VideoProfileInfoKHR::default()
            .video_codec_operation(vkext::VIDEO_CODEC_OPERATION_ENCODE_H264_BIT_KHR)
            .chroma_subsampling(vk::VideoChromaSubsamplingFlagsKHR::TYPE_420)
            .luma_bit_depth(vk::VideoComponentBitDepthFlagsKHR::TYPE_8)
            .chroma_bit_depth(vk::VideoComponentBitDepthFlagsKHR::TYPE_8);
        // profile -> h264 -> usage, all on the heap so the addresses are stable.
        profile.p_next = &*h264 as *const _ as *const c_void;
        let mut out = ProfileChain { usage, h264, profile };
        out.h264.p_next = &*out.usage as *const _ as *const c_void;
        out.profile.p_next = &*out.h264 as *const _ as *const c_void;
        out
    }
}

pub struct Encoder {
    pub fns: VideoFns,
    pub profile: ProfileChain,
    pub session: vk::VideoSessionKHR,
    pub params: vk::VideoSessionParametersKHR,
    pub session_mems: Vec<vk::DeviceMemory>,
    pub picture_format: vk::Format,
    pub caps_report: String,
    pub format_report: String,

    // Filled in by `create_resources`.
    pub query_pool: vk::QueryPool,
    pub src_image: vk::Image,
    pub src_mem: vk::DeviceMemory,
    pub src_view: vk::ImageView,
    pub dpb_image: vk::Image,
    pub dpb_mem: vk::DeviceMemory,
    pub dpb_views: Vec<vk::ImageView>,
    pub bitstream: crate::dev::Buf,
    pub sps: Vec<u8>,
    pub pps: Vec<u8>,
}

impl Encoder {
    pub fn new(ctx: &Ctx) -> Result<Encoder> {
        let (fns, missing) = VideoFns::load(&ctx.entry, &ctx.instance, ctx.device.clone());
        if !missing.is_empty() {
            bail!(
                "the loader returned NULL for {} video entry point(s): {}",
                missing.len(),
                missing.join(", ")
            );
        }

        let raw_pd = ctx.pd;
        let raw_dev = ctx.device.handle();

        // ---- profile -----------------------------------------------------
        let profile = ProfileChain::new();

        // ---- capabilities ------------------------------------------------
        // ash 0.38 declares no `ExtendsVideoEncodeCapabilitiesKHR` trait, so the
        // H.264 capabilities cannot be `push_next`ed onto the encode
        // capabilities: that link is made by hand.
        let mut h264_caps = vk::VideoEncodeH264CapabilitiesKHR::default();
        let mut enc_caps = vk::VideoEncodeCapabilitiesKHR::default();
        unsafe { vkext::chain(&mut enc_caps, &h264_caps as *const _ as *const _) };
        let mut caps = vk::VideoCapabilitiesKHR::default().push_next(&mut enc_caps);
        unsafe {
            (fns.get_physical_device_video_capabilities_khr)(
                raw_pd,
                &profile.profile,
                &mut caps,
            )
        }
        .result()
        .context("vkGetPhysicalDeviceVideoCapabilitiesKHR(H264 encode)")?;

        let caps_report = format!(
            "minCodedExtent={:?} maxCodedExtent={:?} granularity={:?} maxDpbSlots={} \
             maxActiveRefs={} bitstreamAlign(size={} offset={}) rateControlModes={:?} \
             feedback={:?} | h264: maxLevelIdc={} minQp={} maxQp={} maxL0Refs(P)={} \
             maxTemporalLayers={} stdSyntaxFlags={:?}",
            caps.min_coded_extent,
            caps.max_coded_extent,
            caps.picture_access_granularity,
            caps.max_dpb_slots,
            caps.max_active_reference_pictures,
            caps.min_bitstream_buffer_size_alignment,
            caps.min_bitstream_buffer_offset_alignment,
            enc_caps.rate_control_modes,
            enc_caps.supported_encode_feedback_flags,
            h264_caps.max_level_idc,
            h264_caps.min_qp,
            h264_caps.max_qp,
            h264_caps.max_p_picture_l0_reference_count,
            h264_caps.max_temporal_layer_count,
            h264_caps.std_syntax_flags,
        );

        // ---- picture format ---------------------------------------------
        let mut prof_list = vk::VideoProfileListInfoKHR::default()
            .profiles(std::slice::from_ref(&profile.profile));
        let fmt_info = vk::PhysicalDeviceVideoFormatInfoKHR::default()
            .image_usage(
                vk::ImageUsageFlags::VIDEO_ENCODE_SRC_KHR | vk::ImageUsageFlags::TRANSFER_DST,
            )
            .push_next(&mut prof_list);
        let mut n = 0u32;
        unsafe {
            (fns.get_physical_device_video_format_properties_khr)(
                raw_pd,
                &fmt_info,
                &mut n,
                std::ptr::null_mut(),
            )
        }
        .result()?;
        if n == 0 {
            bail!(
                "vkGetPhysicalDeviceVideoFormatPropertiesKHR returned 0 formats for \
                 VIDEO_ENCODE_SRC (driver video_encode_enabled is false?)"
            );
        }
        let mut fmts = vec![vk::VideoFormatPropertiesKHR::default(); n as usize];
        unsafe {
            (fns.get_physical_device_video_format_properties_khr)(
                raw_pd,
                &fmt_info,
                &mut n,
                fmts.as_mut_ptr(),
            )
        }
        .result()?;
        let format_report = fmts
            .iter()
            .map(|f| format!("{:?}", f.format))
            .collect::<Vec<_>>()
            .join(", ");
        let fmt = fmts
            .iter()
            .find(|f| f.format == vk::Format::G8_B8R8_2PLANE_420_UNORM)
            .or_else(|| fmts.first())
            .context("no video encode picture format")?;
        let picture_format = fmt.format;
        eprintln!("[enc] picture format {picture_format:?} (candidates: {format_report})");

        // ---- session -----------------------------------------------------
        let mut h264_sess = vk::VideoEncodeH264SessionCreateInfoKHR::default()
            .use_max_level_idc(true)
            .max_level_idc(vkext::h264std::LEVEL_4_0);
        let sess_ci = vk::VideoSessionCreateInfoKHR::default()
            .queue_family_index(ctx.queue_family)
            .flags(vkext::VIDEO_SESSION_CREATE_INLINE_QUERIES_BIT_KHR)
            .video_profile(&profile.profile)
            .picture_format(picture_format)
            .max_coded_extent(vk::Extent2D {
                width: WIDTH,
                height: HEIGHT,
            })
            .reference_picture_format(picture_format)
            .max_dpb_slots(DPB_SLOTS)
            .max_active_reference_pictures(1)
            .push_next(&mut h264_sess);
        let session = unsafe {
            let mut s = vk::VideoSessionKHR::null();
            (fns.create_video_session_khr)(raw_dev, &sess_ci, std::ptr::null(), &mut s)
                .result()
                .context("vkCreateVideoSessionKHR")?;
            s
        };

        // ---- session memory ---------------------------------------------
        let mut count = 0u32;
        unsafe {
            (fns.get_video_session_memory_requirements_khr)(
                raw_dev,
                session,
                &mut count,
                std::ptr::null_mut(),
            )
        }
        .result()?;
        let mut reqs = vec![
            vk::VideoSessionMemoryRequirementsKHR::default();
            count.max(1) as usize
        ];
        unsafe {
            (fns.get_video_session_memory_requirements_khr)(
                raw_dev,
                session,
                &mut count,
                reqs.as_mut_ptr(),
            )
        }
        .result()?;
        reqs.truncate(count as usize);

        let mut binds = Vec::new();
        let mut session_mems = Vec::new();
        for r in &reqs {
            let mr = r.memory_requirements;
            let ty = crate::dev::find_memory_type(ctx, mr, vk::MemoryPropertyFlags::DEVICE_LOCAL)?;
            unsafe {
                let mem = ctx.device.allocate_memory(
                    &vk::MemoryAllocateInfo::default()
                        .allocation_size(mr.size.max(4096))
                        .memory_type_index(ty),
                    None,
                )?;
                session_mems.push(mem);
                binds.push(
                    vk::BindVideoSessionMemoryInfoKHR::default()
                        .memory_bind_index(r.memory_bind_index)
                        .memory(mem)
                        .memory_size(mr.size),
                );
            }
        }
        unsafe {
            (fns.bind_video_session_memory_khr)(raw_dev, session, binds.len() as u32, binds.as_ptr())
        }
        .result()
        .context("vkBindVideoSessionMemoryKHR")?;
        eprintln!(
            "[enc] session memory: {} bind(s), sizes {:?}",
            binds.len(),
            reqs.iter()
                .map(|r| r.memory_requirements.size)
                .collect::<Vec<_>>()
        );

        // ---- session parameters (SPS/PPS) --------------------------------
        let sps_vui = h264::sps_with_vui(WIDTH, HEIGHT, 1, 30, 1);
        let sps = &sps_vui.sps;
        let pps = h264::pps(QP as i8);
        let add = vk::VideoEncodeH264SessionParametersAddInfoKHR::default()
            .std_sp_ss(std::slice::from_ref(sps))
            .std_pp_ss(std::slice::from_ref(&pps));
        let mut h264_params = vk::VideoEncodeH264SessionParametersCreateInfoKHR::default()
            .max_std_sps_count(1)
            .max_std_pps_count(1)
            .parameters_add_info(&add);
        let params_ci = vk::VideoSessionParametersCreateInfoKHR::default()
            .video_session(session)
            .push_next(&mut h264_params);
        let params = unsafe {
            let mut p = vk::VideoSessionParametersKHR::null();
            (fns.create_video_session_parameters_khr)(raw_dev, &params_ci, std::ptr::null(), &mut p)
                .result()
                .context("vkCreateVideoSessionParametersKHR")?;
            p
        };
        // The driver deep-copies the SPS (including the VUI it points at) into
        // the session parameters, so the VUI only has to outlive the create call.
        drop(sps_vui);

        Ok(Encoder {
            fns,
            profile,
            session,
            params,
            session_mems,
            picture_format,
            caps_report,
            format_report,
            query_pool: vk::QueryPool::null(),
            src_image: vk::Image::null(),
            src_mem: vk::DeviceMemory::null(),
            src_view: vk::ImageView::null(),
            dpb_image: vk::Image::null(),
            dpb_mem: vk::DeviceMemory::null(),
            dpb_views: Vec::new(),
            bitstream: crate::dev::Buf {
                buffer: vk::Buffer::null(),
                memory: vk::DeviceMemory::null(),
                size: 0,
            },
            sps: Vec::new(),
            pps: Vec::new(),
        })
    }
}
