//! Session resources (bitstream buffer, query pool, NV12 images) and the
//! per-frame encode path.

use anyhow::{Context, Result};
use ash::vk::{self, Handle};

use crate::dev::Ctx;
use crate::enc::{BITSTREAM_SIZE, DPB_SLOTS, GOP, QP, Encoder};
use crate::frames::{FRAME_COUNT, HEIGHT, WIDTH};
use crate::vkext;

/// One encoded picture: the raw NAL bytes from the bitstream buffer.
pub struct Encoded {
    pub bytes: Vec<u8>,
    pub is_idr: bool,
}

impl Encoder {
    /// Create the bitstream buffer, the encode-feedback query pool, the NV12
    /// source image and the DPB image (one array layer per DPB slot, which is
    /// how RADV indexes slots: `surf_offset + slot * surf_slice_size`).
    pub fn create_resources(&mut self, ctx: &Ctx) -> Result<()> {
        let d = &ctx.device;
        let fns = &self.fns;
        let raw_dev = d.handle();
        unsafe {
            // HOST_VISIBLE so the spike can read the bitstream back with
            // vkMapMemory instead of a separate staging copy.
            self.bitstream = ctx.create_buffer(
                BITSTREAM_SIZE,
                vk::BufferUsageFlags::VIDEO_ENCODE_DST_KHR,
                vk::MemoryPropertyFlags::DEVICE_LOCAL
                    | vk::MemoryPropertyFlags::HOST_VISIBLE
                    | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?;

            let mut fb = vk::QueryPoolVideoEncodeFeedbackCreateInfoKHR::default()
                .encode_feedback_flags(
                    vk::VideoEncodeFeedbackFlagsKHR::BITSTREAM_BYTES_WRITTEN
                        | vk::VideoEncodeFeedbackFlagsKHR::BITSTREAM_BUFFER_OFFSET,
                );
            let qci = vk::QueryPoolCreateInfo::default()
                .query_type(vk::QueryType::VIDEO_ENCODE_FEEDBACK_KHR)
                .query_count(FRAME_COUNT)
                .push_next(&mut fb);
            self.query_pool = d
                .create_query_pool(&qci, None)
                .context("vkCreateQueryPool(VIDEO_ENCODE_FEEDBACK_KHR)")?;
            d.reset_query_pool(self.query_pool, 0, FRAME_COUNT);

            let (src_image, src_mem) = self.create_nv12_image(
                ctx,
                1,
                vk::ImageUsageFlags::VIDEO_ENCODE_SRC_KHR | vk::ImageUsageFlags::TRANSFER_DST,
            )?;
            self.src_image = src_image;
            self.src_mem = src_mem;
            self.src_view = d.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(src_image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(self.picture_format)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    }),
                None,
            )?;

            let (dpb_image, dpb_mem) = self.create_nv12_image(
                ctx,
                DPB_SLOTS,
                vk::ImageUsageFlags::VIDEO_ENCODE_DPB_KHR,
            )?;
            self.dpb_image = dpb_image;
            self.dpb_mem = dpb_mem;
            for slot in 0..DPB_SLOTS {
                self.dpb_views.push(d.create_image_view(
                    &vk::ImageViewCreateInfo::default()
                        .image(dpb_image)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(self.picture_format)
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: slot,
                            layer_count: 1,
                        }),
                    None,
                )?);
            }

            let (sps, pps) = vkext::read_h264_parameter_sets(fns, raw_dev, self.params)?;
            self.sps = sps;
            self.pps = pps;
            eprintln!(
                "[enc] driver SPS {} B (nalType {}), PPS {} B (nalType {})",
                self.sps.len(),
                self.sps.first().map(|b| b & 0x1f).unwrap_or(0),
                self.pps.len(),
                self.pps.first().map(|b| b & 0x1f).unwrap_or(0),
            );
        }
        Ok(())
    }

    fn create_nv12_image(
        &self,
        ctx: &Ctx,
        layers: u32,
        usage: vk::ImageUsageFlags,
    ) -> Result<(vk::Image, vk::DeviceMemory)> {
        let d = &ctx.device;
        unsafe {
            let mut prof_list =
                vk::VideoProfileListInfoKHR::default().profiles(std::slice::from_ref(&self.profile.profile));
            let ci = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(self.picture_format)
                .extent(vk::Extent3D {
                    width: WIDTH,
                    height: HEIGHT,
                    depth: 1,
                })
                .mip_levels(1)
                .array_layers(layers)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(usage)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .push_next(&mut prof_list);
            let image = d.create_image(&ci, None).with_context(|| {
                format!("vkCreateImage({:?}, layers={layers}, usage={usage:?})", self.picture_format)
            })?;
            let reqs = d.get_image_memory_requirements(image);
            let ty = crate::dev::find_memory_type(ctx, reqs, vk::MemoryPropertyFlags::DEVICE_LOCAL)?;
            let mem = d.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(reqs.size)
                    .memory_type_index(ty),
                None,
            )?;
            d.bind_image_memory(image, mem, 0)?;
            Ok((image, mem))
        }
    }

    /// Record `vkCmdBeginVideoCoding` + `RESET` into `cmd`.
    ///
    /// RADV keeps the session/params on the *command buffer*
    /// (`radv_CmdBeginVideoCodingKHR` only sets `cmd_buffer->video`), so this
    /// must be recorded into every command buffer that encodes. The `RESET`
    /// control sets the session-wide `enc_need_begin` flag, which is what makes
    /// RADV emit the session-init package before the first picture, so it is
    /// recorded once, into the first command buffer.
    ///
    /// Rate control is DISABLED so RADV takes the QP from each slice's
    /// `constantQp`.
    pub fn begin_session(&self, cmd: vk::CommandBuffer, reset: bool) -> Result<()> {
        let fns = &self.fns;
        let raw_cmd = cmd;
        let slots: Vec<vk::VideoReferenceSlotInfoKHR> = (0..DPB_SLOTS)
            .map(|i| vk::VideoReferenceSlotInfoKHR::default().slot_index(i as i32))
            .collect();
        let mut rc_h264 = vk::VideoEncodeH264RateControlInfoKHR::default()
            .gop_frame_count(GOP)
            .idr_period(GOP)
            .consecutive_b_frame_count(0)
            .temporal_layer_count(1);
        // Rate control DISABLED => layer_count must be 0 (ash sets it from the
        // `layers` slice, so no slice is passed at all).
        let mut rc = vk::VideoEncodeRateControlInfoKHR::default()
            .rate_control_mode(vk::VideoEncodeRateControlModeFlagsKHR::DISABLED);
        // ash 0.38 declares no `ExtendsVideoEncodeRateControlInfoKHR` trait, so
        // this one link has to be made by hand.
        unsafe { vkext::chain(&mut rc, &rc_h264 as *const _ as *const _) };
        let begin = vk::VideoBeginCodingInfoKHR::default()
            .video_session(self.session)
            .video_session_parameters(self.params)
            .reference_slots(&slots)
            .push_next(&mut rc);
        let reset_info = vk::VideoCodingControlInfoKHR::default()
            .flags(vk::VideoCodingControlFlagsKHR::RESET);
        unsafe {
            (fns.cmd_begin_video_coding_khr)(raw_cmd, &begin);
            if reset {
                (fns.cmd_control_video_coding_khr)(raw_cmd, &reset_info);
            }
        }
        Ok(())
    }

    pub fn end_session(&self, ctx: &Ctx, cmd: vk::CommandBuffer) {
        let end = vk::VideoEndCodingInfoKHR::default();
        unsafe {
            (self.fns.cmd_end_video_coding_khr)(cmd, &end);
        }
        let _ = ctx;
    }

    /// `vkCmdEncodeVideo` for one picture.
    ///
    /// `slot` receives the reconstruction; P frames reference `ref_slot`.
    ///
    /// Every structure that ends up in a `pNext` list or behind a raw pointer
    /// is a named local: `push_next(&mut { x })` would drop the temporary at the
    /// end of the statement and leave a dangling pointer.
    pub fn encode_frame(
        &self,
        cmd: vk::CommandBuffer,
        index: u32,
        is_idr: bool,
        slot: u32,
        ref_slot: u32,
    ) -> Result<()> {
        let fns = &self.fns;
        let raw_cmd = cmd;
        let coded = vk::Extent2D {
            width: WIDTH,
            height: HEIGHT,
        };
        let src_res = vk::VideoPictureResourceInfoKHR::default()
            .coded_offset(vk::Offset2D::default())
            .coded_extent(coded)
            .base_array_layer(0)
            .image_view_binding(self.src_view);
        let setup_res = vk::VideoPictureResourceInfoKHR::default()
            .coded_offset(vk::Offset2D::default())
            .coded_extent(coded)
            .base_array_layer(slot)
            .image_view_binding(self.dpb_views[slot as usize]);

        let mut setup_ref_flags: vk::native::StdVideoEncodeH264ReferenceInfoFlags = vkext::zeroed();
        setup_ref_flags.set_used_for_long_term_reference(0);
        let setup_std_ref = vk::native::StdVideoEncodeH264ReferenceInfo {
            flags: setup_ref_flags,
            primary_pic_type: if is_idr {
                vkext::h264std::PIC_IDR
            } else {
                vkext::h264std::PIC_P
            },
            FrameNum: index,
            PicOrderCnt: index as i32,
            long_term_pic_num: 0,
            long_term_frame_idx: 0,
            temporal_id: 0,
        };
        let setup_dpb = vk::VideoEncodeH264DpbSlotInfoKHR::default()
            .std_reference_info(&setup_std_ref);
        let mut setup_slot = vk::VideoReferenceSlotInfoKHR::default()
            .slot_index(slot as i32)
            .picture_resource(&setup_res);
        setup_slot.p_next = &setup_dpb as *const _ as *const _;

        // Reference list: the single previous reference, in `ref_slot`.
        let mut ref_list: vk::native::StdVideoEncodeH264ReferenceListsInfo = vkext::zeroed();
        ref_list.num_ref_idx_l0_active_minus1 = 0;
        ref_list.num_ref_idx_l1_active_minus1 = 0;
        ref_list.RefPicList0[0] = ref_slot as u8;
        ref_list.refList0ModOpCount = 0;
        ref_list.refList1ModOpCount = 0;
        ref_list.refPicMarkingOpCount = 0;

        let mut pic_flags: vk::native::StdVideoEncodeH264PictureInfoFlags = vkext::zeroed();
        pic_flags.set_IdrPicFlag(if is_idr { 1 } else { 0 });
        pic_flags.set_is_reference(1);
        pic_flags.set_no_output_of_prior_pics_flag(if is_idr { 1 } else { 0 });
        pic_flags.set_long_term_reference_flag(0);
        pic_flags.set_adaptive_ref_pic_marking_mode_flag(0);
        let std_pic = vk::native::StdVideoEncodeH264PictureInfo {
            flags: pic_flags,
            seq_parameter_set_id: 0,
            pic_parameter_set_id: 0,
            idr_pic_id: if is_idr { (index / 2) as u16 } else { 0 },
            primary_pic_type: if is_idr {
                vkext::h264std::PIC_IDR
            } else {
                vkext::h264std::PIC_P
            },
            frame_num: index,
            PicOrderCnt: index as i32,
            temporal_id: 0,
            reserved1: [0; 3],
            pRefLists: if is_idr {
                std::ptr::null()
            } else {
                &ref_list
            },
        };

        let mut sh_flags: vk::native::StdVideoEncodeH264SliceHeaderFlags = vkext::zeroed();
        sh_flags.set_num_ref_idx_active_override_flag(if is_idr { 0 } else { 1 });
        let slice_header = vk::native::StdVideoEncodeH264SliceHeader {
            flags: sh_flags,
            first_mb_in_slice: 0,
            slice_type: if is_idr {
                vkext::h264std::SLICE_I
            } else {
                vkext::h264std::SLICE_P
            },
            slice_alpha_c0_offset_div2: 0,
            slice_beta_offset_div2: 0,
            slice_qp_delta: 0,
            reserved1: 0,
            cabac_init_idc: 0,
            disable_deblocking_filter_idc: 0,
            pWeightTable: std::ptr::null(),
        };
        let nalu = vk::VideoEncodeH264NaluSliceInfoKHR::default()
            .constant_qp(QP)
            .std_slice_header(&slice_header);
        let mut pic_info = vk::VideoEncodeH264PictureInfoKHR::default()
            .nalu_slice_entries(std::slice::from_ref(&nalu))
            .std_picture_info(&std_pic)
            .generate_prefix_nalu(false);

        // RADV reads the inline query off `pEncodeInfo->pNext`
        // (`radv_vcn_encode_video`), so it goes on the encode info, after the
        // H.264 picture info.
        let inline_q = vk::VideoInlineQueryInfoKHR::default()
            .query_pool(self.query_pool)
            .first_query(index)
            .query_count(1);

        // Reference slot: hoisted so the resources it points at outlive `refs`.
        let mut ref_flags: vk::native::StdVideoEncodeH264ReferenceInfoFlags = vkext::zeroed();
        ref_flags.set_used_for_long_term_reference(0);
        let ref_std_ref = vk::native::StdVideoEncodeH264ReferenceInfo {
            flags: ref_flags,
            primary_pic_type: vkext::h264std::PIC_P,
            FrameNum: index.saturating_sub(1),
            PicOrderCnt: (index as i32) - 1,
            long_term_pic_num: 0,
            long_term_frame_idx: 0,
            temporal_id: 0,
        };
        let ref_dpb =
            vk::VideoEncodeH264DpbSlotInfoKHR::default().std_reference_info(&ref_std_ref);
        let ref_res = vk::VideoPictureResourceInfoKHR::default()
            .coded_offset(vk::Offset2D::default())
            .coded_extent(coded)
            .base_array_layer(ref_slot)
            .image_view_binding(self.dpb_views[ref_slot as usize]);
        let mut ref_slot_info = vk::VideoReferenceSlotInfoKHR::default()
            .slot_index(ref_slot as i32)
            .picture_resource(&ref_res);
        ref_slot_info.p_next = &ref_dpb as *const _ as *const _;
        let refs = [ref_slot_info];

        let mut encode_info = vk::VideoEncodeInfoKHR::default()
            .dst_buffer(self.bitstream.buffer)
            .dst_buffer_offset(0)
            .dst_buffer_range(self.bitstream.size)
            .src_picture_resource(src_res)
            .reference_slots(if is_idr { &[] } else { &refs });
        if !is_idr {
            encode_info.p_setup_reference_slot = &setup_slot;
        }
        unsafe {
            // encode_info -> pic_info -> inline query. ash 0.38 has no
            // `ExtendsVideoEncodeH264PictureInfoKHR` trait for the inline query,
            // so the second link is made by hand.
            encode_info.p_next = &pic_info as *const _ as *const _;
            vkext::chain(&mut pic_info, &inline_q as *const _ as *const _);
            (fns.cmd_encode_video_khr)(raw_cmd, &encode_info);
        }
        Ok(())
    }

    /// Read back the encode-feedback query for `index`.
    ///
    /// RADV copies exactly two `uint32_t`s for a
    /// `VK_QUERY_TYPE_VIDEO_ENCODE_FEEDBACK_KHR` query, in the order the spec
    /// gives the feedback flags: `BITSTREAM_BUFFER_OFFSET` then
    /// `BITSTREAM_BYTES_WRITTEN`. `WITH_STATUS_BIT` appends a third word (1 =
    /// the encode reported success).
    ///
    /// `wait` = false reads the raw pool memory without spinning, so a query the
    /// GPU never wrote can be reported instead of hanging the spike.
    ///
    /// Returns `(offset, bytes_written)`.
    pub fn read_feedback(&self, ctx: &Ctx, index: u32, wait: bool) -> Result<(u32, u32)> {
        let mut out = [0u32; 3];
        let mut flags = vkext::QUERY_RESULT_WITH_STATUS_BIT_KHR;
        if wait {
            flags |= vk::QueryResultFlags::WAIT;
        }
        unsafe {
            ctx.device
                .get_query_pool_results(self.query_pool, index, &mut out, flags)
                .with_context(|| format!("get_query_pool_results(frame {index})"))?;
        }
        let (offset, bytes, status) = (out[0], out[1], out[2]);
        anyhow::ensure!(
            status & 1 != 0,
            "encode feedback for frame {index} reports failure (status={status:#x})"
        );
        Ok((offset, bytes))
    }

    /// Copy `len` bytes at `offset` out of the bitstream buffer.
    pub fn read_bitstream(&self, ctx: &Ctx, offset: u64, len: usize) -> Result<Vec<u8>> {
        let mut v = vec![0u8; len];
        unsafe {
            let p = ctx
                .device
                .map_memory(self.bitstream.memory, offset, len as u64, vk::MemoryMapFlags::empty())?;
            std::ptr::copy_nonoverlapping(p as *const u8, v.as_mut_ptr(), len);
            ctx.device.unmap_memory(self.bitstream.memory);
        }
        Ok(v)
    }

    /// Wrap the driver's SPS/PPS and one slice NAL into an Annex-B access unit.
    pub fn annexb(&self, slice: &[u8], with_params: bool) -> Vec<u8> {
        let mut out = Vec::new();
        let mut push = |out: &mut Vec<u8>, n: &[u8]| {
            if n.is_empty() {
                return;
            }
            if !starts_with_start_code(n) {
                out.extend_from_slice(&[0, 0, 0, 1]);
            }
            out.extend_from_slice(n);
        };
        if with_params {
            push(&mut out, &self.sps);
            push(&mut out, &self.pps);
        }
        push(&mut out, slice);
        out
    }

    pub fn destroy(&self, ctx: &Ctx) {
        let d = &ctx.device;
        let raw_dev = d.handle();
        unsafe {
            let _ = d.device_wait_idle();
            let fns = &self.fns;
            if !self.params.is_null() {
                (fns.destroy_video_session_parameters_khr)(raw_dev, self.params, std::ptr::null());
            }
            if !self.session.is_null() {
                (fns.destroy_video_session_khr)(raw_dev, self.session, std::ptr::null());
            }
            for m in &self.session_mems {
                d.free_memory(*m, None);
            }
            if !self.query_pool.is_null() {
                d.destroy_query_pool(self.query_pool, None);
            }
            if !self.src_view.is_null() {
                d.destroy_image_view(self.src_view, None);
            }
            if !self.src_image.is_null() {
                d.destroy_image(self.src_image, None);
                d.free_memory(self.src_mem, None);
            }
            for v in &self.dpb_views {
                d.destroy_image_view(*v, None);
            }
            if !self.dpb_image.is_null() {
                d.destroy_image(self.dpb_image, None);
                d.free_memory(self.dpb_mem, None);
            }
            ctx.destroy_buffer(&self.bitstream);
        }
    }
}

fn starts_with_start_code(b: &[u8]) -> bool {
    b.starts_with(&[0, 0, 0, 1]) || b.starts_with(&[0, 0, 1])
}
