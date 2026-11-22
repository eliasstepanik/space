//! Phase-0 spike 0.5: prove Vulkan Video H.264 **encode** works from ash on this
//! machine's real GPU (RADV GFX1151), and that the result decodes in Chromium.
//!
//! Pipeline, all on the GPU:
//!
//! 1. N=60 synthetic 640x480 RGBA frames ([`frames::synthetic_frame`]).
//! 2. RGBA -> NV12 in a compute pass on the compute queue ([`convert`]).
//! 3. One `vkCmdEncodeVideoKHR` per picture on the video-encode queue, IDR +
//!    P-only GOP, into a single bitstream buffer ([`enc`], [`encode`]).
//! 4. The picture length comes from the inline `VK_QUERY_TYPE_VIDEO_ENCODE_FEEDBACK_KHR`
//!    query; the SPS/PPS NALs come from `vkGetEncodedVideoSessionParametersKHR`.
//! 5. `out.h264` is Annex-B: SPS, PPS then one slice NAL per access unit.
//!
//! Exit codes: `0` success, `2` the driver/ash does not support H.264 encode
//! here (a valid, reportable result), anything else = unexpected failure.
//!
//! THROWAWAY spike.

mod convert;
mod dev;
mod enc;
mod encode;
mod frames;
mod h264;
mod vkext;

use std::io::Write;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use ash::vk;

use convert::Converter;
use enc::Encoder;
use frames::{FRAME_COUNT, HEIGHT, PSNR_REGION, WIDTH, psnr_region_luma};

/// How long to wait for one encode submission before declaring the VCN stuck.
const FENCE_TIMEOUT_NS: u64 = 15_000_000_000;

/// Written next to `out.h264` so the Chromium decode check has everything it
/// needs (parameter sets, expected sample luma, frame count).
#[derive(serde::Serialize)]
struct Meta {
    width: u32,
    height: u32,
    frames: u32,
    gop: u32,
    qp: i32,
    sps_hex: String,
    pps_hex: String,
    /// Luma of the PSNR sample region for every source frame, 0..255.
    expected_luma: Vec<Vec<f64>>,
    psnr_region: [u32; 4],
    /// NAL byte length of each access unit as reported by the feedback query.
    au_sizes: Vec<usize>,
    encode_ms_per_frame: f64,
    device: String,
    caps_report: String,
    format_report: String,
    entry_points_missing: Vec<&'static str>,
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn main() -> Result<()> {
    let mut out_path = std::path::PathBuf::from("out.h264");
    let mut dump_yuv: Option<std::path::PathBuf> = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        if a == "--out" {
            out_path = args
                .next()
                .map(std::path::PathBuf::from)
                .context("--out needs a path")?;
        } else if a == "--dump-yuv" {
            dump_yuv = Some(
                args.next()
                    .map(std::path::PathBuf::from)
                    .context("--dump-yuv needs a path")?,
            );
        } else {
            eprintln!("usage: vkvideo-h264 [--out <file>] [--dump-yuv <file>]");
            return Ok(());
        }
    }

    // Reference NV12 for the decode/PSNR harness: the CPU conversion of every
    // synthetic frame, concatenated. Needs no GPU.
    if let Some(p) = dump_yuv {
        let mut f = std::fs::File::create(&p)
            .with_context(|| format!("create {}", p.display()))?;
        for i in 0..FRAME_COUNT {
            let nv12 = frames::nv12_from_rgba(&frames::synthetic_frame(i));
            f.write_all(&nv12)?;
        }
        f.flush()?;
        eprintln!(
            "[main] wrote {} ({} frames of {}x{} NV12)",
            p.display(),
            FRAME_COUNT,
            WIDTH,
            HEIGHT
        );
        return Ok(());
    }

    // ---- device / session -------------------------------------------------
    let ctx = match dev::create(None) {
        Ok(c) => c,
        Err(e) => {
            // No encode queue family / missing extension is the interesting
            // negative result: report it with its own exit code.
            eprintln!("[main] H.264 video encode is NOT available here: {e:#}");
            print_exit(2);
        }
    };

    let mut encoder = match Encoder::new(&ctx) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("[main] encode session could not be created: {e:#}");
            print_exit(2);
        }
    };
    eprintln!("[main] caps: {}", encoder.caps_report);
    encoder
        .create_resources(&ctx)
        .context("create encode resources")?;

    // ---- converter --------------------------------------------------------
    let spv = frames::compile_rgba_to_nv12_spv().map_err(|e| anyhow::anyhow!(e))?;
    let converter = Converter::new(&ctx, &spv).context("create RGBA->NV12 converter")?;

    // ---- command buffer for the encode queue ------------------------------
    let enc_pool = unsafe {
        ctx.device.create_command_pool(
            &vk::CommandPoolCreateInfo::default().queue_family_index(ctx.queue_family),
            None,
        )
    }?;
    let enc_cmd = unsafe {
        ctx.device
            .allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(enc_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
            .context("allocate encode command buffer")?
            .remove(0)
    };
    let enc_fence = unsafe { ctx.device.create_fence(&vk::FenceCreateInfo::default(), None) }?;

    // ---- encode -----------------------------------------------------------
    let mut au_bytes: Vec<u8> = Vec::new();
    let mut au_sizes: Vec<usize> = Vec::new();
    let mut expected_luma: Vec<Vec<f64>> = Vec::new();
    let mut total = Instant::now();
    let mut idr_count = 0usize;

    for i in 0..FRAME_COUNT {
        let rgba = frames::synthetic_frame(i);
        expected_luma.push(psnr_region_luma(&rgba));

        // 1) RGBA -> NV12 on the compute queue.
        converter
            .convert_frame(&ctx, &rgba, encoder.src_image)
            .with_context(|| format!("convert frame {i}"))?;

        // 2) encode on the video queue.
        let is_idr = i % enc::GOP == 0;
        if is_idr {
            idr_count += 1;
        }
        let (slot, ref_slot) = if is_idr { (0, 0) } else { (i % 2, (i + 1) % 2) };

        let t = Instant::now();
        unsafe {
            ctx.device.reset_fences(&[enc_fence])?;
            ctx.device
                .reset_command_buffer(enc_cmd, vk::CommandBufferResetFlags::empty())?;
            ctx.device.begin_command_buffer(
                enc_cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            // RADV keeps the session on the command buffer, so begin (plus the
            // session-wide RESET, once) is recorded into every encode cmd buf.
            encoder.begin_session(enc_cmd, i == 0)?;
            if std::env::var_os("SPIKE_NO_ENCODE").is_none() {
                encoder.encode_frame(enc_cmd, i, is_idr, slot, ref_slot)?;
            }
            encoder.end_session(&ctx, enc_cmd);
            ctx.device.end_command_buffer(enc_cmd)?;
            ctx.device.queue_submit(
                ctx.queue,
                &[vk::SubmitInfo::default().command_buffers(&[enc_cmd])],
                enc_fence,
            )?;
            // Bounded: a VCN that never completes (missing firmware) must be
            // reported, not hung on.
            if let Err(r) = ctx.device.wait_for_fences(&[enc_fence], true, FENCE_TIMEOUT_NS) {
                if r == vk::Result::TIMEOUT {
                    bail!(
                        "frame {i}: the encode submission did not complete within {} s \
                         (the VCN ring never signalled; see README)",
                        FENCE_TIMEOUT_NS / 1_000_000_000
                    );
                }
                r.result().context("wait_for_fences")?;
            }
        }
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        eprintln!("[main] frame {i}: submission completed in {ms:.2} ms");
        let (offset, written) = encoder
            .read_feedback(&ctx, i, false)
            .with_context(|| format!("read feedback frame {i}"))?;
        if written == 0 {
            bail!("frame {i}: driver reported 0 bytes written");
        }
        let slice = encoder
            .read_bitstream(&ctx, offset as u64, written as usize)
            .with_context(|| format!("read bitstream frame {i}"))?;
        let au = encoder.annexb(&slice, true);
        au_bytes.extend_from_slice(&au);
        au_sizes.push(au.len());
        if i < 3 || i == FRAME_COUNT - 1 {
            eprintln!(
                "[main] frame {i:2} {} {} B -> au {} B ({ms:.2} ms)",
                if is_idr { "IDR" } else { "P  " },
                written,
                au.len()
            );
        }
    }
    let total_ms = total.elapsed().as_secs_f64() * 1000.0;

    // ---- write the stream -------------------------------------------------
    let mut f = std::fs::File::create(&out_path)
        .with_context(|| format!("create {}", out_path.display()))?;
    f.write_all(&au_bytes)?;
    f.flush()?;
    let per_frame = total_ms / FRAME_COUNT as f64;
    eprintln!(
        "[main] wrote {} ({} B, {idr_count} IDR, {total_ms:.1} ms total, {per_frame:.2} ms/frame)",
        out_path.display(),
        au_bytes.len()
    );

    let meta = Meta {
        width: WIDTH,
        height: HEIGHT,
        frames: FRAME_COUNT,
        gop: enc::GOP,
        qp: enc::QP,
        sps_hex: hex(&encoder.sps),
        pps_hex: hex(&encoder.pps),
        expected_luma,
        psnr_region: [PSNR_REGION.0, PSNR_REGION.1, PSNR_REGION.2, PSNR_REGION.3],
        au_sizes,
        encode_ms_per_frame: per_frame,
        device: device_name(&ctx),
        caps_report: encoder.caps_report.clone(),
        format_report: encoder.format_report.clone(),
        entry_points_missing: encoder.fns.missing().to_vec(),
    };
    let meta_path = out_path.with_extension("json");
    std::fs::write(&meta_path, serde_json::to_vec_pretty(&meta)?)
        .with_context(|| format!("write {}", meta_path.display()))?;
    eprintln!("[main] wrote {}", meta_path.display());

    unsafe {
        let _ = ctx.device.device_wait_idle();
        encoder.destroy(&ctx);
        ctx.device.destroy_fence(enc_fence, None);
        ctx.device.destroy_command_pool(enc_pool, None);
    }
    drop(converter);
    println!("ENCODE_OK");
    Ok(())
}

/// `vkCmdBeginVideoCoding` has to be recorded into the same command buffer as
/// the encodes it governs, so this re-records it per submission.
fn device_name(ctx: &dev::Ctx) -> String {
    let raw: Vec<u8> = ctx
        .props
        .device_name
        .iter()
        .take_while(|b| **b != 0)
        .map(|b| *b as u8)
        .collect();
    String::from_utf8_lossy(&raw).into_owned()
}

fn print_exit(code: i32) -> ! {
    println!("ENCODE_UNAVAILABLE");
    std::process::exit(code)
}
