//! The H.264 SPS/PPS templates handed to the Vulkan encode session.
//!
//! Baseline profile, constrained to what RADV's VCN driver reads back: it
//! re-encodes these structures into the bitstream itself
//! (`vk_video_encode_h264_sps`/`_pps`), so they must be a *valid* baseline
//! syntax for a 640x480 4:2:0 8-bit stream.

use ash::vk::native::{
    StdVideoH264PictureParameterSet, StdVideoH264PpsFlags, StdVideoH264SequenceParameterSet,
    StdVideoH264SequenceParameterSetVui, StdVideoH264SpsFlags,
    StdVideoH264ChromaFormatIdc_STD_VIDEO_H264_CHROMA_FORMAT_IDC_420,
    StdVideoH264PocType_STD_VIDEO_H264_POC_TYPE_0,
};

use crate::vkext::{self, h264std};

/// Baseline profile, no CABAC, no 8x8 transform, no weighted prediction.
pub fn sps(width: u32, height: u32, max_num_ref_frames: u8) -> StdVideoH264SequenceParameterSet {
    let mut flags: StdVideoH264SpsFlags = vkext::zeroed();
    flags.set_frame_mbs_only_flag(1);
    flags.set_direct_8x8_inference_flag(1);
    // Constrained baseline: the constraint flags make the stream safe for
    // hardware/browser decoders that refuse the unconstrained variant.
    flags.set_constraint_set1_flag(1);

    let mut sps: StdVideoH264SequenceParameterSet = vkext::zeroed();
    sps.flags = flags;
    sps.profile_idc = h264std::PROFILE_BASELINE;
    sps.level_idc = h264std::LEVEL_4_0;
    sps.chroma_format_idc = StdVideoH264ChromaFormatIdc_STD_VIDEO_H264_CHROMA_FORMAT_IDC_420;
    sps.seq_parameter_set_id = 0;
    sps.bit_depth_luma_minus8 = 0;
    sps.bit_depth_chroma_minus8 = 0;
    sps.log2_max_frame_num_minus4 = 0; // 4 bits of frame_num
    sps.pic_order_cnt_type = StdVideoH264PocType_STD_VIDEO_H264_POC_TYPE_0;
    sps.log2_max_pic_order_cnt_lsb_minus4 = 4; // 8 bits of POC
    sps.max_num_ref_frames = max_num_ref_frames;
    // Coded extent is MB-rounded; the driver pads the picture to it.
    sps.pic_width_in_mbs_minus1 = width.div_ceil(16) - 1;
    sps.pic_height_in_map_units_minus1 = height.div_ceil(16) - 1;
    sps.frame_crop_left_offset = 0;
    sps.frame_crop_right_offset = (sps.pic_width_in_mbs_minus1 + 1) * 16 - width;
    sps.frame_crop_top_offset = 0;
    sps.frame_crop_bottom_offset = (sps.pic_height_in_map_units_minus1 + 1) * 16 - height;
    sps.pOffsetForRefFrame = std::ptr::null();
    sps.pScalingLists = std::ptr::null();
    sps.pSequenceParameterSetVui = std::ptr::null();
    sps
}

/// Owns an SPS together with the VUI it points at.
///
/// The SPS stores a raw pointer to the VUI, so the VUI must live at a stable
/// address: it is heap-allocated in the `Box` and the pointer is taken *after*
/// the struct is in its final home. Returning the two by value would move the
/// VUI and leave the SPS pointing at a dead stack slot.
pub struct SpsVui {
    pub sps: StdVideoH264SequenceParameterSet,
    vui: Box<StdVideoH264SequenceParameterSetVui>,
}

/// Same as [`sps`] but with a VUI carrying the frame rate and the colour
/// description. Chromium does not need it; it is here so the stream is
/// self-describing for the README's ffprobe-style notes.
pub fn sps_with_vui(
    width: u32,
    height: u32,
    max_num_ref_frames: u8,
    fps_num: u32,
    fps_den: u32,
) -> SpsVui {
    let mut sps = sps(width, height, max_num_ref_frames);
    let mut vui = Box::new(vkext::zeroed::<StdVideoH264SequenceParameterSetVui>());
    vui.flags.set_timing_info_present_flag(1);
    vui.flags.set_fixed_frame_rate_flag(1);
    vui.num_units_in_tick = fps_den;
    vui.time_scale = fps_num * 2; // time_scale = 2 * fps * units
    vui.max_num_reorder_frames = 0;
    vui.max_dec_frame_buffering = max_num_ref_frames.max(1);
    // The NV12 source is produced with full-range BT.601 coefficients.
    vui.flags.set_video_signal_type_present_flag(1);
    vui.flags.set_video_full_range_flag(1);
    vui.flags.set_color_description_present_flag(1);
    vui.matrix_coefficients = 5; // BT.601
    vui.colour_primaries = 5;
    vui.transfer_characteristics = 5;
    vui.pHrdParameters = std::ptr::null();
    sps.flags.set_vui_parameters_present_flag(1);
    let out = SpsVui { sps, vui };
    // Point at the heap copy, which does not move when `out` is returned.
    let mut out = out;
    out.sps.pSequenceParameterSetVui = &*out.vui as *const _;
    out
}

pub fn pps(qp: i8) -> StdVideoH264PictureParameterSet {
    let mut flags: StdVideoH264PpsFlags = vkext::zeroed();
    flags.set_entropy_coding_mode_flag(0); // CAVLC
    flags.set_weighted_pred_flag(0);
    flags.set_transform_8x8_mode_flag(0);
    flags.set_deblocking_filter_control_present_flag(1);
    flags.set_constrained_intra_pred_flag(0);

    let mut pps: StdVideoH264PictureParameterSet = vkext::zeroed();
    pps.flags = flags;
    pps.seq_parameter_set_id = 0;
    pps.pic_parameter_set_id = 0;
    pps.num_ref_idx_l0_default_active_minus1 = 0;
    pps.num_ref_idx_l1_default_active_minus1 = 0;
    pps.weighted_bipred_idc = 0;
    // RADV overwrites this with 0 in `radv_video_patch_encode_session_parameters`,
    // so the QP actually used comes from `constantQp` on the slice, not here.
    pps.pic_init_qp_minus26 = qp - 26;
    pps.pic_init_qs_minus26 = 0;
    pps.chroma_qp_index_offset = 0;
    pps.second_chroma_qp_index_offset = 0;
    pps.pScalingLists = std::ptr::null();
    pps
}
