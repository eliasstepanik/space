//! Synthetic source frames and the RGBA -> NV12 (BT.601) conversion.
//!
//! The conversion runs as a Vulkan compute pass. The shader reads an RGBA8
//! storage buffer and writes two *linear* storage buffers (planar Y and
//! interleaved UV); those are then copied into the two planes of the NV12
//! encode image. Writing the multiplanar image directly from a shader is not
//! possible (no storage image format for NV12 planes), so the linear staging
//! buffers are the practical path.

pub const WIDTH: u32 = 640;
pub const HEIGHT: u32 = 480;
pub const FRAME_COUNT: u32 = 60;

/// Flat-colour rectangle used for the PSNR check: top-left, clear of the
/// moving square's path, so it is identical in every frame.
pub const PSNR_REGION: (u32, u32, u32, u32) = (16, 16, 128, 64);

/// Background colour of the synthetic frames (the PSNR sample colour).
pub const BG: (u8, u8, u8) = (90, 120, 150);

/// WGSL: RGBA8 storage buffer in, planar Y + interleaved UV out (BT.601
/// full-range).
///
/// WGSL has no byte-typed storage buffer, so all three buffers are `array<u32>`
/// and each invocation writes *whole* words (a 4x2 pixel block = two luma
/// words + one UV word). That keeps two invocations from ever sharing a word,
/// which read-modify-write would require.
pub const RGBA_TO_NV12_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> src : array<u32>;
@group(0) @binding(1) var<storage, read_write> yplane : array<u32>;
@group(0) @binding(2) var<storage, read_write> uvplane : array<u32>;

const KR : f32 = 0.299;
const KG : f32 = 0.587;
const KB : f32 = 0.114;

// Unpack one RGBA8 pixel word (little-endian: byte 0 = R) to normalised rgb.
fn rgb(word : u32) -> vec3<f32> {
    let r = f32(word & 0xffu) / 255.0;
    let g = f32((word >> 8u) & 0xffu) / 255.0;
    let b = f32((word >> 16u) & 0xffu) / 255.0;
    return vec3<f32>(r, g, b);
}

fn luma(c : vec3<f32>) -> u32 {
    let y = clamp(KR * c.x + KG * c.y + KB * c.z, 0.0, 1.0);
    return u32(y * 255.0 + 0.5);
}

fn chroma(c : vec3<f32>) -> vec2<u32> {
    let u = clamp(-0.168736 * c.x - 0.331264 * c.y + 0.5 * c.z + 0.5, 0.0, 1.0);
    let v = clamp(0.5 * c.x - 0.418688 * c.y - 0.081312 * c.z + 0.5, 0.0, 1.0);
    return vec2<u32>(u32(u * 255.0 + 0.5), u32(v * 255.0 + 0.5));
}

// One invocation converts a 4x2 pixel block.
@compute @workgroup_size(16, 8)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let xw = gid.x;              // word column: 4 pixels
    let y0 = gid.y * 2u;         // first row
    if (xw >= 160u || y0 >= 480u) { return; }

    var c : array<vec3<f32>, 8>;
    for (var dy : u32 = 0u; dy < 2u; dy = dy + 1u) {
        for (var dx : u32 = 0u; dx < 4u; dx = dx + 1u) {
            c[dy * 4u + dx] = rgb(src[(y0 + dy) * 160u + xw]);
        }
    }

    // Two full luma words, one per row.
    for (var dy : u32 = 0u; dy < 2u; dy = dy + 1u) {
        var w : u32 = 0u;
        for (var dx : u32 = 0u; dx < 4u; dx = dx + 1u) {
            w = w | (luma(c[dy * 4u + dx]) << (8u * dx));
        }
        yplane[(y0 + dy) * 160u + xw] = w;
    }

    // One UV word: two interleaved chroma samples, each the 2x2 average.
    var uw : u32 = 0u;
    for (var k : u32 = 0u; k < 2u; k = k + 1u) {
        let i = k * 2u;
        let avg = (c[i] + c[i + 1u] + c[i + 4u] + c[i + 5u]) * 0.25;
        let uv = chroma(avg);
        uw = uw | (uv.x << (16u * k)) | (uv.y << (8u + 16u * k));
    }
    uvplane[gid.y * 160u + xw] = uw;
}
"#;

/// Generate frame `i` as RGBA bytes (WIDTH*HEIGHT*4): flat background plus a
/// striped square that sweeps across the frame, so consecutive frames differ.
pub fn synthetic_frame(i: u32) -> Vec<u8> {
    let n = (WIDTH as usize) * (HEIGHT as usize);
    let mut px = vec![0u8; n * 4];
    let (br, bg, bb) = BG;
    for p in px.chunks_exact_mut(4) {
        p[0] = br;
        p[1] = bg;
        p[2] = bb;
        p[3] = 255;
    }

    let size = 64i32;
    let travel = (WIDTH as i32 - size).max(1);
    let step = (travel / 30).max(1);
    let x0 = ((i as i32) * step) % travel;
    let y0 = 200 + ((i as i32) % 5) * 8;
    for y in y0..(y0 + size) {
        if y < 0 || y >= HEIGHT as i32 {
            continue;
        }
        for x in x0..(x0 + size) {
            if x < 0 || x >= WIDTH as i32 {
                continue;
            }
            let o = ((y as usize) * (WIDTH as usize) + (x as usize)) * 4;
            let band = (((x - x0) / 8) % 2) == 0;
            px[o] = if band { 240 } else { 30 };
            px[o + 1] = if band { 60 } else { 200 };
            px[o + 2] = if band { 60 } else { 90 };
            px[o + 3] = 255;
        }
    }
    px
}

/// BT.601 full-range NV12 (planar Y then interleaved UV) for one RGBA frame.
///
/// The same maths as the compute shader, on the CPU. Used only to write a
/// reference `.yuv` that an offline encoder (ffmpeg/libx264) can turn into a
/// valid stream, so the decode + PSNR half of the spike can be validated even
/// when the Vulkan encoder is unavailable.
pub fn nv12_from_rgba(rgba: &[u8]) -> Vec<u8> {
    let w = WIDTH as usize;
    let h = HEIGHT as usize;
    let mut out = vec![0u8; w * h + w * h / 2];
    let ylen = w * h;
    for by in (0..h).step_by(2) {
        for bx in (0..w).step_by(2) {
            let mut sr = 0u32;
            let mut sg = 0u32;
            let mut sb = 0u32;
            for dy in 0..2u32 {
                for dx in 0..2u32 {
                    let o = (((by + dy as usize) * w + bx + dx as usize) * 4) as usize;
                    let (r, g, b) = (rgba[o] as u32, rgba[o + 1] as u32, rgba[o + 2] as u32);
                    let y = (0.299 * r as f64 + 0.587 * g as f64 + 0.114 * b as f64)
                        .round()
                        .clamp(0.0, 255.0) as u8;
                    out[(by as u32 + dy) as usize * w + (bx as u32 + dx) as usize] = y;
                    sr += r;
                    sg += g;
                    sb += b;
                }
            }
            let (r, g, b) = (sr as f64 / 4.0, sg as f64 / 4.0, sb as f64 / 4.0);
            let u = (-0.168736 * r - 0.331264 * g + 0.5 * b + 128.0)
                .round()
                .clamp(0.0, 255.0) as u8;
            let v = (0.5 * r - 0.418688 * g - 0.081312 * b + 128.0)
                .round()
                .clamp(0.0, 255.0) as u8;
            let co = (by / 2) * w + bx;
            out[ylen + co] = u;
            out[ylen + co + 1] = v;
        }
    }
    out
}

/// BT.601 luma of the PSNR sample region, from an RGBA frame.
pub fn psnr_region_luma(rgba: &[u8]) -> Vec<f64> {
    let (x0, y0, w, h) = PSNR_REGION;
    let mut out = Vec::with_capacity((w * h) as usize);
    for y in y0..(y0 + h) {
        for x in x0..(x0 + w) {
            let o = ((y as usize) * (WIDTH as usize) + (x as usize)) * 4;
            out.push(
                0.299 * rgba[o] as f64 + 0.587 * rgba[o + 1] as f64 + 0.114 * rgba[o + 2] as f64,
            );
        }
    }
    out
}

/// PSNR (dB) between two equal-length sample vectors over [0, 255].
pub fn psnr_db(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return f64::INFINITY;
    }
    let mut se = 0.0;
    for i in 0..n {
        let d = a[i] - b[i];
        se += d * d;
    }
    let mse = se / n as f64;
    if mse <= 1e-12 {
        return f64::INFINITY;
    }
    10.0 * (255.0 * 255.0 / mse).log10()
}

/// Compile the WGSL above to SPIR-V with naga (pure Rust, no glslang).
pub fn compile_rgba_to_nv12_spv() -> Result<Vec<u32>, String> {
    let module = naga::front::wgsl::parse_str(RGBA_TO_NV12_WGSL)
        .map_err(|e| e.emit_to_string(RGBA_TO_NV12_WGSL))?;
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::default(),
    );
    let info = validator
        .validate(&module)
        .map_err(|e| format!("validate: {e:?}"))?;
    let opts = naga::back::spv::Options::default();
    let pipeline = naga::back::spv::PipelineOptions {
        shader_stage: naga::ShaderStage::Compute,
        entry_point: "main".to_string(),
    };
    let mut words = Vec::new();
    let mut writer = naga::back::spv::Writer::new(&opts).map_err(|e| format!("{e:?}"))?;
    writer
        .write(&module, &info, Some(&pipeline), &None, &mut words)
        .map_err(|e| format!("{e:?}"))?;
    Ok(words)
}
