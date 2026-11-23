//! Textures: the CPU-side image a mesh material samples, its mip chain, the
//! built-in procedural textures and the image decoders the GLB loader uses.
//!
//! A [`Texture`] is 8-bit RGBA, **sRGB-encoded** colour with linear alpha,
//! row-major with the top row first — exactly what a glTF base-colour image
//! decodes to, and what the renderer uploads as `R8G8B8A8_SRGB` with a full
//! mip chain ([`Texture::mip_chain`], box-filtered in linear light).
//!
//! # Decoders
//!
//! [`decode_image`] reads PNG (every colour type and bit depth, palettes,
//! `tRNS`, Adam7 interlacing; the container and filters are implemented here
//! on top of `miniz_oxide`'s inflate) and baseline/progressive JPEG (through
//! `jpeg-decoder`). Like the GLB loader it treats its input as untrusted:
//! dimensions are capped at [`MAX_TEXTURE_DIMENSION`], the caller passes a byte
//! budget that bounds what is allocated, every chunk is bounds- and
//! CRC-checked, and a malformed image is a [`MeshError`], never a panic.

use std::sync::OnceLock;

use crate::mesh::MeshError;

/// Largest texture edge the decoders and [`Texture::from_rgba8`] accept.
pub const MAX_TEXTURE_DIMENSION: u32 = 8192;

fn err<T>(message: impl Into<String>) -> Result<T, MeshError> {
    Err(MeshError(message.into()))
}

/// An 8-bit RGBA image: sRGB-encoded colour, linear alpha, top row first.
#[derive(Clone, PartialEq, Eq)]
pub struct Texture {
    width: u32,
    height: u32,
    pixels: Vec<u8>,
}

impl std::fmt::Debug for Texture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Texture({}x{})", self.width, self.height)
    }
}

/// One level of a mip chain (same pixel format as [`Texture`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MipLevel {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
}

impl Texture {
    /// Wrap `pixels` (RGBA8, `width * height * 4` bytes, top row first).
    pub fn from_rgba8(width: u32, height: u32, pixels: Vec<u8>) -> Result<Self, MeshError> {
        let valid = 1..=MAX_TEXTURE_DIMENSION;
        if !valid.contains(&width) || !valid.contains(&height) {
            return err(format!(
                "texture size {width}x{height} is outside 1..={MAX_TEXTURE_DIMENSION} per side"
            ));
        }
        if pixels.len() != width as usize * height as usize * 4 {
            return err(format!(
                "a {width}x{height} texture needs {} bytes, got {}",
                width as usize * height as usize * 4,
                pixels.len()
            ));
        }
        Ok(Self {
            width,
            height,
            pixels,
        })
    }

    /// A procedural texture: `pixel(x, y)` for every texel of a `size` x
    /// `size` image (`size` is clamped to `1..=MAX_TEXTURE_DIMENSION`).
    fn generate(size: u32, mut pixel: impl FnMut(u32, u32) -> [u8; 4]) -> Self {
        let size = size.clamp(1, MAX_TEXTURE_DIMENSION);
        let mut pixels = Vec::with_capacity(size as usize * size as usize * 4);
        for y in 0..size {
            for x in 0..size {
                pixels.extend_from_slice(&pixel(x, y));
            }
        }
        Self {
            width: size,
            height: size,
            pixels,
        }
    }

    #[must_use]
    pub fn width(&self) -> u32 {
        self.width
    }

    #[must_use]
    pub fn height(&self) -> u32 {
        self.height
    }

    /// RGBA8, row-major, top row first.
    #[must_use]
    pub fn pixels(&self) -> &[u8] {
        &self.pixels
    }

    /// The texel at (`x`, `y`), clamped to the image.
    #[must_use]
    pub fn texel(&self, x: u32, y: u32) -> [u8; 4] {
        let x = x.min(self.width - 1) as usize;
        let y = y.min(self.height - 1) as usize;
        let i = (y * self.width as usize + x) * 4;
        [
            self.pixels[i],
            self.pixels[i + 1],
            self.pixels[i + 2],
            self.pixels[i + 3],
        ]
    }

    /// Levels in the full mip chain: down to 1x1.
    #[must_use]
    pub fn mip_level_count(&self) -> u32 {
        32 - self.width.max(self.height).leading_zeros()
    }

    /// The full mip chain, level 0 (this image) first. Each level halves the
    /// one above (rounding down, never below 1) with a 2x2 box filter applied
    /// in linear light: averaging the sRGB-encoded bytes instead would darken
    /// every fine pattern with distance.
    #[must_use]
    pub fn mip_chain(&self) -> Vec<MipLevel> {
        let mut levels = vec![MipLevel {
            width: self.width,
            height: self.height,
            pixels: self.pixels.clone(),
        }];
        while let Some(last) = levels.last() {
            if last.width == 1 && last.height == 1 {
                break;
            }
            let next = downsample(last);
            levels.push(next);
        }
        levels
    }
}

// ---- colour space --------------------------------------------------------------

/// sRGB byte -> linear value in 0..=1.
fn srgb_decode_table() -> &'static [f32; 256] {
    static TABLE: OnceLock<[f32; 256]> = OnceLock::new();
    TABLE.get_or_init(|| {
        std::array::from_fn(|i| {
            let c = i as f32 / 255.0;
            if c <= 0.040_45 {
                c / 12.92
            } else {
                ((c + 0.055) / 1.055).powf(2.4)
            }
        })
    })
}

/// Steps of the linear -> sRGB table. 4096 keeps the quantisation error under
/// half an 8-bit step everywhere (the curve is steepest at black: 12.92 * 255
/// / 4096 = 0.8 byte per step), so `encode(decode(v)) == v` for every byte.
const ENCODE_STEPS: usize = 4096;

fn srgb_encode_table() -> &'static [u8; ENCODE_STEPS + 1] {
    static TABLE: OnceLock<[u8; ENCODE_STEPS + 1]> = OnceLock::new();
    TABLE.get_or_init(|| {
        std::array::from_fn(|i| {
            let l = i as f32 / ENCODE_STEPS as f32;
            let c = if l <= 0.003_130_8 {
                l * 12.92
            } else {
                1.055 * l.powf(1.0 / 2.4) - 0.055
            };
            (c * 255.0 + 0.5) as u8
        })
    })
}

/// Linear value (clamped to 0..=1) -> sRGB byte.
fn linear_to_srgb8(linear: f32) -> u8 {
    let index = (linear.clamp(0.0, 1.0) * ENCODE_STEPS as f32 + 0.5) as usize;
    srgb_encode_table()[index.min(ENCODE_STEPS)]
}

/// The next mip level below `src`.
fn downsample(src: &MipLevel) -> MipLevel {
    let (sw, sh) = (src.width as usize, src.height as usize);
    let (dw, dh) = ((sw / 2).max(1), (sh / 2).max(1));
    let decode = srgb_decode_table();
    let mut pixels = Vec::with_capacity(dw * dh * 4);
    for y in 0..dh {
        let rows = [(2 * y).min(sh - 1), (2 * y + 1).min(sh - 1)];
        for x in 0..dw {
            let columns = [(2 * x).min(sw - 1), (2 * x + 1).min(sw - 1)];
            let mut sum = [0.0f32; 4];
            for sy in rows {
                for sx in columns {
                    let p = &src.pixels[(sy * sw + sx) * 4..][..4];
                    for c in 0..3 {
                        sum[c] += decode[p[c] as usize];
                    }
                    sum[3] += f32::from(p[3]);
                }
            }
            pixels.extend_from_slice(&[
                linear_to_srgb8(sum[0] * 0.25),
                linear_to_srgb8(sum[1] * 0.25),
                linear_to_srgb8(sum[2] * 0.25),
                (sum[3] * 0.25 + 0.5) as u8,
            ]);
        }
    }
    MipLevel {
        width: dw as u32,
        height: dh as u32,
        pixels,
    }
}

// ---- procedural textures -------------------------------------------------------

/// Integer hash (the same splitmix32 variant as the shaders' `hash32`).
fn hash32(x: u32) -> u32 {
    let mut z = x;
    z = (z ^ (z >> 16)).wrapping_mul(0x21f0_aaad);
    z = (z ^ (z >> 15)).wrapping_mul(0x735a_2d97);
    z ^ (z >> 15)
}

/// A hash of two cell coordinates, as a float in 0..1.
fn hash2(x: u32, y: u32) -> f32 {
    (hash32(x.wrapping_mul(0x9e37_79b9) ^ hash32(y).wrapping_add(0x85eb_ca6b)) >> 8) as f32
        / 16_777_216.0
}

fn shade(color: [u8; 3], factor: f32) -> [u8; 3] {
    color.map(|c| (f32::from(c) * factor).clamp(0.0, 255.0) as u8)
}

fn opaque(color: [u8; 3]) -> [u8; 4] {
    [color[0], color[1], color[2], 255]
}

impl Texture {
    /// One opaque texel of `color` (sRGB bytes).
    #[must_use]
    pub fn solid(color: [u8; 3]) -> Self {
        Self::generate(1, |_, _| opaque(color))
    }

    /// A `cells` x `cells` checkerboard of `a` and `b` (sRGB bytes); the cell
    /// at the UV origin (the image's top-left corner) is `a`.
    #[must_use]
    pub fn checkerboard(size: u32, cells: u32, a: [u8; 3], b: [u8; 3]) -> Self {
        let size = size.clamp(1, MAX_TEXTURE_DIMENSION);
        let cells = u64::from(cells.clamp(1, size));
        Self::generate(size, |x, y| {
            let cx = u64::from(x) * cells / u64::from(size);
            let cy = u64::from(y) * cells / u64::from(size);
            opaque(if (cx + cy).is_multiple_of(2) { a } else { b })
        })
    }

    /// The UV test grid: 8x8 cells whose colour encodes the coordinates — red
    /// grows with `u` (left to right), green with `v` (top to bottom) — with
    /// white cell borders and every cell labelled column letter + row digit
    /// (`A1` at the UV origin, `H8` at (1, 1)). A mirrored or rotated mapping
    /// is obvious at a glance: the labels read backwards or sideways.
    #[must_use]
    pub fn uv_grid(size: u32) -> Self {
        const CELLS: u32 = 8;
        let size = size.clamp(CELLS * 8, MAX_TEXTURE_DIMENSION);
        let cell = size / CELLS;
        let line = (size / 128).max(1);
        let scale = (cell / 16).max(1);
        Self::generate(size, |x, y| {
            let (cx, cy) = ((x / cell).min(CELLS - 1), (y / cell).min(CELLS - 1));
            let (lx, ly) = (x - cx * cell, y - cy * cell);
            if lx < line || ly < line || lx >= cell - line || ly >= cell - line {
                return [235, 235, 235, 255];
            }
            // The label: two glyphs in the cell's top-left corner.
            let (tx, ty) = (lx.wrapping_sub(3 * line), ly.wrapping_sub(3 * line));
            let (gx, gy) = (tx / scale, ty / scale);
            if tx < cell && ty < cell && gy < crate::overlay::GLYPH_HEIGHT {
                let column = gx / crate::overlay::GLYPH_ADVANCE;
                let within = gx % crate::overlay::GLYPH_ADVANCE;
                let glyph = match column {
                    0 => Some((b'A' + cx as u8) as char),
                    1 => Some((b'1' + cy as u8) as char),
                    _ => None,
                };
                if let Some(ch) = glyph {
                    if crate::overlay::glyph_pixel(ch, within, gy) {
                        return [255, 255, 255, 255];
                    }
                }
            }
            let ramp = |i: u32| (40 + i * 195 / (CELLS - 1)) as u8;
            // Alternate cells are a little darker so the grid reads even
            // where the border lines are lost to minification.
            let dim = if (cx + cy).is_multiple_of(2) {
                1.0
            } else {
                0.8
            };
            opaque(shade([ramp(cx), ramp(cy), 150], dim))
        })
    }

    /// Hull plating: staggered rows of panels in slightly varying tints of
    /// `base`, dark seams between them and a rivet in every panel corner.
    /// Tiles seamlessly.
    #[must_use]
    pub fn hull_panels(size: u32, base: [u8; 3]) -> Self {
        const ROWS: u32 = 4;
        const COLUMNS: u32 = 4;
        let size = size.clamp(ROWS * 8, MAX_TEXTURE_DIMENSION);
        let (row_h, panel_w) = (size / ROWS, size / COLUMNS);
        let seam = (size / 128).max(1);
        let rivet = (size / 128).max(1);
        Self::generate(size, |x, y| {
            let row = (y / row_h).min(ROWS - 1);
            // Every other row is shifted half a panel, like brickwork.
            let shifted = (x + (row % 2) * panel_w / 2) % size;
            let column = (shifted / panel_w).min(COLUMNS - 1);
            let (lx, ly) = (shifted - column * panel_w, y - row * row_h);
            // Distance to the nearest panel edge. (The last panel of a row
            // absorbs the remainder when the size is not a multiple of four.)
            let (dx, dy) = (
                lx.min((panel_w - 1).saturating_sub(lx)),
                ly.min((row_h - 1).saturating_sub(ly)),
            );
            if dx < seam || dy < seam {
                return opaque(shade(base, 0.5));
            }
            // The edge next to a seam catches the light.
            if dx < 2 * seam || dy < 2 * seam {
                return opaque(shade(base, 1.06));
            }
            // A small rivet in each corner.
            let (rx, ry) = (dx.abs_diff(5 * rivet), dy.abs_diff(5 * rivet));
            if rx * rx + ry * ry <= rivet * rivet {
                return opaque(shade(base, 0.68));
            }
            let tint = 0.86 + 0.14 * hash2(column, row);
            // Faint horizontal streaks, constant along a scan line.
            let streak = 0.97 + 0.03 * hash2(y, 7919);
            opaque(shade(base, tint * streak))
        })
    }

    /// Brushed metal: fine horizontal streaks of varying brightness around
    /// `base`. Tiles seamlessly.
    #[must_use]
    pub fn brushed_metal(size: u32, base: [u8; 3]) -> Self {
        let size = size.clamp(4, MAX_TEXTURE_DIMENSION);
        Self::generate(size, |x, y| {
            // Each scan line has its own brightness, plus short streaks that
            // run along it (the grain of the brushing).
            let line = hash2(y, 31);
            let grain = hash2(x / 16, y) * 0.5 + hash2((x + 8) % size / 16, y + size) * 0.5;
            opaque(shade(base, 0.82 + 0.16 * line + 0.1 * grain))
        })
    }

    /// Diagonal warning stripes: `stripes` pairs of `a` and `b` across the
    /// image. Tiles seamlessly.
    #[must_use]
    pub fn hazard_stripes(size: u32, stripes: u32, a: [u8; 3], b: [u8; 3]) -> Self {
        let size = size.clamp(2, MAX_TEXTURE_DIMENSION);
        let bands = u64::from(stripes.clamp(1, size / 2)) * 2;
        Self::generate(size, |x, y| {
            let band = u64::from(x + y) * bands / u64::from(size);
            opaque(if band.is_multiple_of(2) { a } else { b })
        })
    }
}

// ---- PNG -----------------------------------------------------------------------

const PNG_SIGNATURE: [u8; 8] = [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];

const CRC_TABLE: [u32; 256] = {
    let mut table = [0u32; 256];
    let mut n = 0;
    while n < 256 {
        let mut c = n as u32;
        let mut k = 0;
        while k < 8 {
            c = if c & 1 != 0 {
                0xedb8_8320 ^ (c >> 1)
            } else {
                c >> 1
            };
            k += 1;
        }
        table[n] = c;
        n += 1;
    }
    table
};

/// The CRC-32 of a PNG chunk (its type bytes followed by its data).
fn crc32(parts: &[&[u8]]) -> u32 {
    let mut crc = 0xffff_ffffu32;
    for part in parts {
        for &byte in *part {
            crc = CRC_TABLE[((crc ^ u32::from(byte)) & 0xff) as usize] ^ (crc >> 8);
        }
    }
    !crc
}

fn be32(bytes: &[u8], offset: usize) -> Option<u32> {
    let b: [u8; 4] = bytes.get(offset..offset.checked_add(4)?)?.try_into().ok()?;
    Some(u32::from_be_bytes(b))
}

/// What an `IHDR` chunk says.
#[derive(Clone, Copy)]
struct PngHeader {
    width: usize,
    height: usize,
    depth: u8,
    color_type: u8,
    interlaced: bool,
}

impl PngHeader {
    fn channels(&self) -> usize {
        match self.color_type {
            2 => 3,
            4 => 2,
            6 => 4,
            _ => 1,
        }
    }

    /// Bytes of one scan line `width` pixels wide (without its filter byte).
    fn row_bytes(&self, width: usize) -> usize {
        (width * self.channels() * self.depth as usize).div_ceil(8)
    }

    /// The image's passes as `(x0, y0, dx, dy, columns, rows)`: one for a
    /// plain image, the seven Adam7 passes (empty ones dropped) otherwise.
    fn passes(&self) -> Vec<[usize; 6]> {
        const ADAM7: [[usize; 4]; 7] = [
            [0, 0, 8, 8],
            [4, 0, 8, 8],
            [0, 4, 4, 8],
            [2, 0, 4, 4],
            [0, 2, 2, 4],
            [1, 0, 2, 2],
            [0, 1, 1, 2],
        ];
        if !self.interlaced {
            return vec![[0, 0, 1, 1, self.width, self.height]];
        }
        ADAM7
            .iter()
            .map(|&[x0, y0, dx, dy]| {
                let columns = (self.width + dx - 1 - x0.min(self.width + dx - 1)) / dx;
                let rows = (self.height + dy - 1 - y0.min(self.height + dy - 1)) / dy;
                [x0, y0, dx, dy, columns, rows]
            })
            .filter(|pass| pass[4] > 0 && pass[5] > 0)
            .collect()
    }
}

/// Undo PNG's per-scan-line filters in place. `raw` is `rows` lines of one
/// filter byte + `row_bytes` data bytes; `bpp` is the bytes per whole pixel.
fn unfilter(raw: &mut [u8], rows: usize, row_bytes: usize, bpp: usize) -> Result<(), MeshError> {
    let stride = row_bytes + 1;
    for y in 0..rows {
        let (above, rest) = raw.split_at_mut(y * stride);
        let previous: Option<&[u8]> =
            (y > 0).then(|| &above[(y - 1) * stride + 1..(y - 1) * stride + 1 + row_bytes]);
        let filter = rest[0];
        let line = &mut rest[1..stride];
        let up = |i: usize| previous.map_or(0, |p| p[i]);
        match filter {
            0 => {}
            1 => {
                for i in bpp..row_bytes {
                    line[i] = line[i].wrapping_add(line[i - bpp]);
                }
            }
            2 => {
                for (i, value) in line.iter_mut().enumerate() {
                    *value = value.wrapping_add(up(i));
                }
            }
            3 => {
                for i in 0..row_bytes {
                    let left = if i >= bpp { line[i - bpp] } else { 0 };
                    let mean = (u16::from(left) + u16::from(up(i))) / 2;
                    line[i] = line[i].wrapping_add(mean as u8);
                }
            }
            4 => {
                for i in 0..row_bytes {
                    let (a, c) = if i >= bpp {
                        (i16::from(line[i - bpp]), i16::from(up(i - bpp)))
                    } else {
                        (0, 0)
                    };
                    let b = i16::from(up(i));
                    // The Paeth predictor: whichever neighbour is closest to
                    // a + b - c, ties going to left, then above.
                    let p = a + b - c;
                    let (pa, pb, pc) = ((p - a).abs(), (p - b).abs(), (p - c).abs());
                    let predicted = if pa <= pb && pa <= pc {
                        a
                    } else if pb <= pc {
                        b
                    } else {
                        c
                    };
                    line[i] = line[i].wrapping_add(predicted as u8);
                }
            }
            other => return err(format!("PNG: unknown scan-line filter {other}")),
        }
    }
    Ok(())
}

/// Sample `index` of an unfiltered scan line at `depth` bits per sample.
fn png_sample(line: &[u8], index: usize, depth: u8) -> u16 {
    match depth {
        8 => u16::from(line[index]),
        16 => u16::from_be_bytes([line[2 * index], line[2 * index + 1]]),
        _ => {
            let depth = usize::from(depth);
            let bit = index * depth;
            let shift = 8 - depth - bit % 8;
            u16::from(line[bit / 8] >> shift) & ((1 << depth) - 1)
        }
    }
}

/// A sample scaled to 8 bits.
fn to_8bit(sample: u16, depth: u8) -> u8 {
    match depth {
        16 => (sample >> 8) as u8,
        8 => sample as u8,
        // 1, 2 and 4 bits: replicate so the maximum maps to 255.
        _ => (u32::from(sample) * 255 / ((1u32 << depth) - 1)) as u8,
    }
}

fn decode_png(bytes: &[u8], budget: usize) -> Result<Texture, MeshError> {
    let truncated = || MeshError("PNG: the file is truncated".into());
    let mut header: Option<PngHeader> = None;
    let mut palette: Vec<[u8; 3]> = Vec::new();
    let mut transparency: Vec<u8> = Vec::new();
    let mut compressed: Vec<u8> = Vec::new();
    let mut offset = PNG_SIGNATURE.len();
    loop {
        let length = be32(bytes, offset).ok_or_else(truncated)? as usize;
        let kind = bytes.get(offset + 4..offset + 8).ok_or_else(truncated)?;
        let end = (offset + 8).checked_add(length).ok_or_else(truncated)?;
        let data = bytes.get(offset + 8..end).ok_or_else(truncated)?;
        let stored = be32(bytes, end).ok_or_else(truncated)?;
        if crc32(&[kind, data]) != stored {
            return err("PNG: a chunk fails its CRC check");
        }
        offset = end + 4;
        if header.is_none() && kind != b"IHDR" {
            return err("PNG: the first chunk is not IHDR");
        }
        match kind {
            b"IHDR" => {
                if header.is_some() || data.len() != 13 {
                    return err("PNG: bad IHDR chunk");
                }
                let (width, height) = (
                    be32(data, 0).unwrap_or(0) as usize,
                    be32(data, 4).unwrap_or(0) as usize,
                );
                let max = MAX_TEXTURE_DIMENSION as usize;
                if width == 0 || height == 0 || width > max || height > max {
                    return err(format!(
                        "PNG: {width}x{height} is outside 1..={max} pixels per side"
                    ));
                }
                if width * height * 4 > budget {
                    return err(format!(
                        "PNG: {width}x{height} exceeds the texture memory budget"
                    ));
                }
                let (depth, color_type) = (data[8], data[9]);
                let depth_ok = match color_type {
                    0 => matches!(depth, 1 | 2 | 4 | 8 | 16),
                    3 => matches!(depth, 1 | 2 | 4 | 8),
                    2 | 4 | 6 => matches!(depth, 8 | 16),
                    _ => false,
                };
                if !depth_ok || data[10] != 0 || data[11] != 0 || data[12] > 1 {
                    return err("PNG: unsupported colour type, bit depth or method");
                }
                header = Some(PngHeader {
                    width,
                    height,
                    depth,
                    color_type,
                    interlaced: data[12] == 1,
                });
            }
            b"PLTE" => {
                if data.len() % 3 != 0 || data.len() > 768 {
                    return err("PNG: bad PLTE chunk");
                }
                palette = data.as_chunks::<3>().0.to_vec();
            }
            b"tRNS" => transparency = data.iter().copied().take(256).collect(),
            b"IDAT" => compressed.extend_from_slice(data),
            b"IEND" => break,
            // A chunk whose first letter is upper case is critical: not
            // understanding it means not understanding the image.
            other if other[0].is_ascii_uppercase() => {
                return err("PNG: unknown critical chunk");
            }
            _ => {}
        }
    }
    let header = header.ok_or_else(|| MeshError("PNG: no IHDR chunk".into()))?;
    if header.color_type == 3 && palette.is_empty() {
        return err("PNG: a palette image without a palette");
    }

    // Inflate to exactly the size the header implies, and no further.
    let passes = header.passes();
    let expected: usize = passes
        .iter()
        .map(|pass| pass[5] * (1 + header.row_bytes(pass[4])))
        .sum();
    let mut raw = miniz_oxide::inflate::decompress_to_vec_zlib_with_limit(&compressed, expected)
        .map_err(|e| MeshError(format!("PNG: bad image data ({:?})", e.status)))?;
    if raw.len() != expected {
        return err("PNG: the image data is shorter than the header says");
    }

    let (depth, channels) = (header.depth, header.channels());
    let bpp = (channels * usize::from(depth)).div_ceil(8);
    // A colour-keyed transparent sample value (grey and RGB images).
    let key = |i: usize| -> Option<u16> {
        let b = transparency.get(2 * i..2 * i + 2)?;
        Some(u16::from_be_bytes([b[0], b[1]]))
    };
    let (grey_key, rgb_key) = (key(0), key(0).zip(key(1)).zip(key(2)));
    let mut pixels = vec![0u8; header.width * header.height * 4];
    let mut start = 0;
    for [x0, y0, dx, dy, columns, rows] in passes {
        let row_bytes = header.row_bytes(columns);
        let stride = row_bytes + 1;
        let pass = &mut raw[start..start + rows * stride];
        start += rows * stride;
        unfilter(pass, rows, row_bytes, bpp)?;
        for row in 0..rows {
            let line = &pass[row * stride + 1..(row + 1) * stride];
            for column in 0..columns {
                let s = |c: usize| png_sample(line, column * channels + c, depth);
                let rgba = match header.color_type {
                    0 => {
                        let g = to_8bit(s(0), depth);
                        let a = if grey_key == Some(s(0)) { 0 } else { 255 };
                        [g, g, g, a]
                    }
                    2 => {
                        let keyed = rgb_key == Some(((s(0), s(1)), s(2)));
                        [
                            to_8bit(s(0), depth),
                            to_8bit(s(1), depth),
                            to_8bit(s(2), depth),
                            if keyed { 0 } else { 255 },
                        ]
                    }
                    3 => {
                        let index = usize::from(s(0));
                        let Some(&[r, g, b]) = palette.get(index) else {
                            return err("PNG: a pixel names a colour past the palette");
                        };
                        [r, g, b, transparency.get(index).copied().unwrap_or(255)]
                    }
                    4 => {
                        let g = to_8bit(s(0), depth);
                        [g, g, g, to_8bit(s(1), depth)]
                    }
                    _ => [
                        to_8bit(s(0), depth),
                        to_8bit(s(1), depth),
                        to_8bit(s(2), depth),
                        to_8bit(s(3), depth),
                    ],
                };
                let (x, y) = (x0 + column * dx, y0 + row * dy);
                pixels[(y * header.width + x) * 4..][..4].copy_from_slice(&rgba);
            }
        }
    }
    Texture::from_rgba8(header.width as u32, header.height as u32, pixels)
}

/// Encode `texture` as an 8-bit RGBA PNG (what [`MeshData::to_glb`] embeds).
///
/// [`MeshData::to_glb`]: crate::mesh::MeshData::to_glb
#[must_use]
pub fn encode_png(texture: &Texture) -> Vec<u8> {
    let (width, height) = (texture.width as usize, texture.height as usize);
    // Every scan line gets filter 0 ("none"): simple, and procedural textures
    // deflate well as they are.
    let mut raw = Vec::with_capacity(height * (width * 4 + 1));
    for row in texture.pixels.chunks_exact(width * 4) {
        raw.push(0);
        raw.extend_from_slice(row);
    }
    let compressed = miniz_oxide::deflate::compress_to_vec_zlib(&raw, 6);

    let mut out = PNG_SIGNATURE.to_vec();
    let mut chunk = |kind: &[u8; 4], data: &[u8]| {
        out.extend_from_slice(&(data.len() as u32).to_be_bytes());
        out.extend_from_slice(kind);
        out.extend_from_slice(data);
        out.extend_from_slice(&crc32(&[kind, data]).to_be_bytes());
    };
    let mut header = Vec::with_capacity(13);
    header.extend_from_slice(&texture.width.to_be_bytes());
    header.extend_from_slice(&texture.height.to_be_bytes());
    // 8 bits per sample, colour type 6 (RGBA), deflate, adaptive filtering,
    // not interlaced.
    header.extend_from_slice(&[8, 6, 0, 0, 0]);
    chunk(b"IHDR", &header);
    // An IDAT chunk's length is a u32: split anything larger.
    for part in compressed.chunks(1 << 30) {
        chunk(b"IDAT", part);
    }
    chunk(b"IEND", &[]);
    out
}

// ---- JPEG ----------------------------------------------------------------------

fn decode_jpeg(bytes: &[u8], budget: usize) -> Result<Texture, MeshError> {
    use jpeg_decoder::PixelFormat;

    let bad = |e: jpeg_decoder::Error| MeshError(format!("JPEG: {e}"));
    let mut decoder = jpeg_decoder::Decoder::new(bytes);
    decoder.read_info().map_err(bad)?;
    let info = decoder
        .info()
        .ok_or_else(|| MeshError("JPEG: no image header".into()))?;
    let (width, height) = (u32::from(info.width), u32::from(info.height));
    let valid = 1..=MAX_TEXTURE_DIMENSION;
    if !valid.contains(&width) || !valid.contains(&height) {
        return err(format!(
            "JPEG: {width}x{height} is outside 1..={MAX_TEXTURE_DIMENSION} pixels per side"
        ));
    }
    let texels = width as usize * height as usize;
    if texels * 4 > budget {
        return err(format!(
            "JPEG: {width}x{height} exceeds the texture memory budget"
        ));
    }
    // The decoder's own output is at most four bytes per pixel.
    decoder.set_max_decoding_buffer_size(texels * 4);
    let data = decoder.decode().map_err(bad)?;
    let bytes_per_texel = match info.pixel_format {
        PixelFormat::L8 => 1,
        PixelFormat::L16 => 2,
        PixelFormat::RGB24 => 3,
        PixelFormat::CMYK32 => 4,
    };
    if data.len() != texels * bytes_per_texel {
        return err("JPEG: the decoder returned an unexpected amount of data");
    }
    let mut pixels = Vec::with_capacity(texels * 4);
    for texel in data.chunks_exact(bytes_per_texel) {
        let rgb = match info.pixel_format {
            PixelFormat::L8 => [texel[0]; 3],
            // 16-bit samples come in native byte order: keep the high byte.
            PixelFormat::L16 => [(u16::from_ne_bytes([texel[0], texel[1]]) >> 8) as u8; 3],
            PixelFormat::RGB24 => [texel[0], texel[1], texel[2]],
            // Ink amounts (the decoder has already undone Adobe's inversion):
            // what is left of white after the colour ink and the black.
            PixelFormat::CMYK32 => {
                let paper = 255 - u16::from(texel[3]);
                let ink = |c: u8| ((255 - u16::from(c)) * paper / 255) as u8;
                [ink(texel[0]), ink(texel[1]), ink(texel[2])]
            }
        };
        pixels.extend_from_slice(&[rgb[0], rgb[1], rgb[2], 255]);
    }
    Texture::from_rgba8(width, height, pixels)
}

/// Decode a PNG or JPEG image (recognised by its first bytes, not by any
/// declared MIME type) into a [`Texture`]. `budget` is the most RGBA8 bytes the
/// decoded image may occupy; a larger image is an error, not an allocation.
pub fn decode_image(bytes: &[u8], budget: usize) -> Result<Texture, MeshError> {
    if bytes.starts_with(&PNG_SIGNATURE) {
        decode_png(bytes, budget)
    } else if bytes.starts_with(&[0xff, 0xd8]) {
        decode_jpeg(bytes, budget)
    } else {
        err("the image is neither PNG nor JPEG")
    }
}

/// A 16x8 baseline JPEG for the tests (written by an independent encoder):
/// the left half (220, 40, 30), the right half (30, 60, 210).
#[cfg(test)]
#[rustfmt::skip]
pub(crate) const TINY_JPEG: [u8; 291] = [
    0xff, 0xd8, 0xff, 0xe0, 0x00, 0x10, 0x4a, 0x46, 0x49, 0x46, 0x00, 0x01, 0x01, 0x00, 0x00, 0x01,
    0x00, 0x01, 0x00, 0x00, 0xff, 0xdb, 0x00, 0x43, 0x00, 0x03, 0x02, 0x02, 0x03, 0x02, 0x02, 0x03,
    0x03, 0x03, 0x03, 0x04, 0x03, 0x03, 0x04, 0x05, 0x08, 0x05, 0x05, 0x04, 0x04, 0x05, 0x0a, 0x07,
    0x07, 0x06, 0x08, 0x0c, 0x0a, 0x0c, 0x0c, 0x0b, 0x0a, 0x0b, 0x0b, 0x0d, 0x0e, 0x12, 0x10, 0x0d,
    0x0e, 0x11, 0x0e, 0x0b, 0x0b, 0x10, 0x16, 0x10, 0x11, 0x13, 0x14, 0x15, 0x15, 0x15, 0x0c, 0x0f,
    0x17, 0x18, 0x16, 0x14, 0x18, 0x12, 0x14, 0x15, 0x14, 0xff, 0xdb, 0x00, 0x43, 0x01, 0x03, 0x04,
    0x04, 0x05, 0x04, 0x05, 0x09, 0x05, 0x05, 0x09, 0x14, 0x0d, 0x0b, 0x0d, 0x14, 0x14, 0x14, 0x14,
    0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14,
    0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14,
    0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0xff, 0xc0,
    0x00, 0x11, 0x08, 0x00, 0x08, 0x00, 0x10, 0x03, 0x01, 0x11, 0x00, 0x02, 0x11, 0x01, 0x03, 0x11,
    0x01, 0xff, 0xc4, 0x00, 0x14, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x07, 0xff, 0xc4, 0x00, 0x14, 0x10, 0x01, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0xc4, 0x00,
    0x16, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x09, 0x07, 0x08, 0xff, 0xc4, 0x00, 0x14, 0x11, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0xda, 0x00, 0x0c, 0x03,
    0x01, 0x00, 0x02, 0x11, 0x03, 0x11, 0x00, 0x3f, 0x00, 0x22, 0x48, 0x9b, 0xcc, 0x78, 0x4c, 0x86,
    0x23, 0xff, 0xd9,
];

#[cfg(test)]
mod tests {
    use super::*;

    /// JPEG images decode (to within the rounding of the inverse DCT), are
    /// held to the same size budget, and a truncated one is never a panic.
    #[test]
    fn jpeg_decodes() {
        let t = decode_image(&TINY_JPEG, usize::MAX).expect("a baseline JPEG");
        assert_eq!((t.width(), t.height()), (16, 8));
        let near = |p: [u8; 4], q: [u8; 3]| (0..3).all(|c| p[c].abs_diff(q[c]) <= 8) && p[3] == 255;
        assert!(near(t.texel(2, 3), [220, 40, 30]), "{:?}", t.texel(2, 3));
        assert!(near(t.texel(12, 3), [30, 60, 210]), "{:?}", t.texel(12, 3));
        assert!(
            decode_image(&TINY_JPEG, 16 * 8 * 4 - 1).is_err(),
            "over budget"
        );
        assert!(decode_image(&TINY_JPEG, 16 * 8 * 4).is_ok());
        for cut in 0..TINY_JPEG.len() {
            let _ = decode_image(&TINY_JPEG[..cut], usize::MAX);
        }
    }

    #[test]
    fn srgb_tables_round_trip_every_byte() {
        let decode = srgb_decode_table();
        for v in 0..=255u8 {
            assert_eq!(linear_to_srgb8(decode[v as usize]), v, "byte {v}");
        }
        assert_eq!(decode[0], 0.0);
        assert!((decode[255] - 1.0).abs() < 1e-6);
        // Mid grey 188 is about 50% linear: the reason mips are not averaged
        // in sRGB space.
        assert!((decode[188] - 0.5).abs() < 0.01);
    }

    /// The chain runs down to 1x1, halving (rounding down) each level; a
    /// uniform image stays exactly uniform; and the filter averages light, not
    /// bytes: a black/white checker becomes sRGB 188 (50% linear), not 128.
    #[test]
    fn mip_chain_halves_to_one_texel_and_filters_in_linear_light() {
        let checker = Texture::checkerboard(8, 8, [255; 3], [0; 3]);
        let chain = checker.mip_chain();
        let sizes: Vec<(u32, u32)> = chain.iter().map(|l| (l.width, l.height)).collect();
        assert_eq!(sizes, [(8, 8), (4, 4), (2, 2), (1, 1)]);
        assert_eq!(checker.mip_level_count(), 4);
        assert_eq!(chain[0].pixels, checker.pixels());
        for level in &chain[1..] {
            assert_eq!(
                level.pixels.len(),
                (level.width * level.height * 4) as usize
            );
            for texel in level.pixels.as_chunks::<4>().0 {
                assert_eq!(*texel, [188, 188, 188, 255], "level {}", level.width);
            }
        }

        // 2x2 cells of 4 texels: level 1 is still a crisp checker.
        let coarse = Texture::checkerboard(8, 2, [200, 30, 30], [20, 20, 220]);
        let level1 = &coarse.mip_chain()[1];
        assert_eq!(level1.pixels[..4], [200, 30, 30, 255]);
        assert_eq!(level1.pixels[2 * 4..3 * 4], [20, 20, 220, 255]);

        // A uniform colour survives every level bit for bit.
        let flat = Texture::from_rgba8(5, 3, [7, 99, 201, 130].repeat(15)).unwrap();
        let chain = flat.mip_chain();
        let sizes: Vec<(u32, u32)> = chain.iter().map(|l| (l.width, l.height)).collect();
        assert_eq!(sizes, [(5, 3), (2, 1), (1, 1)]);
        assert_eq!(flat.mip_level_count(), 3);
        for level in &chain {
            for texel in level.pixels.as_chunks::<4>().0 {
                assert_eq!(*texel, [7, 99, 201, 130]);
            }
        }
    }

    #[test]
    fn texture_sizes_are_validated() {
        assert!(Texture::from_rgba8(2, 2, vec![0; 16]).is_ok());
        assert!(Texture::from_rgba8(2, 2, vec![0; 15]).is_err());
        assert!(Texture::from_rgba8(0, 2, Vec::new()).is_err());
        assert!(Texture::from_rgba8(MAX_TEXTURE_DIMENSION + 1, 1, Vec::new()).is_err());
    }

    #[test]
    fn procedural_textures_have_the_documented_layout() {
        let (white, black) = ([255u8; 3], [0u8; 3]);
        let checker = Texture::checkerboard(64, 4, white, black);
        assert_eq!((checker.width(), checker.height()), (64, 64));
        // Cell (0, 0) at the UV origin is the first colour; neighbours
        // alternate; the diagonal neighbour matches.
        assert_eq!(checker.texel(0, 0), [255, 255, 255, 255]);
        assert_eq!(checker.texel(15, 15), [255, 255, 255, 255]);
        assert_eq!(checker.texel(16, 0), [0, 0, 0, 255]);
        assert_eq!(checker.texel(0, 16), [0, 0, 0, 255]);
        assert_eq!(checker.texel(16, 16), [255, 255, 255, 255]);

        // The UV grid: red grows with u, green with v, and the labels differ
        // from cell to cell (so the image has no mirror symmetry).
        let grid = Texture::uv_grid(256);
        let body = |cx: u32, cy: u32| grid.texel(cx * 32 + 24, cy * 32 + 24);
        assert!(body(7, 0)[0] > body(0, 0)[0] + 100, "red grows with u");
        assert!(body(0, 7)[1] > body(0, 0)[1] + 100, "green grows with v");
        assert!(body(0, 0)[1] < 60 && body(0, 0)[0] < 60);
        let mirrored = (0..256).all(|y| (0..256).all(|x| grid.texel(x, y) == grid.texel(y, x)));
        assert!(!mirrored, "the grid must show its orientation");
        // The label of cell A1 is drawn in white near its top-left corner.
        let label = (0..32)
            .flat_map(|y| (0..32).map(move |x| (x, y)))
            .filter(|&(x, y)| x >= 4 && y >= 4 && grid.texel(x, y) == [255; 4])
            .count();
        assert!(label > 20, "cell A1 has a label ({label} white texels)");

        // Tiling textures are opaque and deterministic.
        for texture in [
            Texture::hull_panels(128, [200, 204, 210]),
            Texture::brushed_metal(128, [180, 182, 186]),
            Texture::hazard_stripes(128, 4, [240, 200, 20], [20, 20, 20]),
        ] {
            assert_eq!((texture.width(), texture.height()), (128, 128));
            assert!(texture
                .pixels()
                .as_chunks::<4>()
                .0
                .iter()
                .all(|p| p[3] == 255));
            let distinct: std::collections::HashSet<[u8; 4]> = texture
                .pixels()
                .as_chunks::<4>()
                .0
                .iter()
                .copied()
                .collect();
            assert!(distinct.len() >= 2, "a pattern, not a flat colour");
        }
        assert_eq!(
            Texture::hull_panels(64, [200; 3]),
            Texture::hull_panels(64, [200; 3])
        );
        assert_eq!(Texture::solid([1, 2, 3]).pixels(), [1, 2, 3, 255]);
    }

    #[test]
    fn png_round_trips_exactly() {
        let texture = Texture::uv_grid(64);
        let png = encode_png(&texture);
        assert_eq!(png[..8], PNG_SIGNATURE);
        let decoded = decode_image(&png, usize::MAX).expect("our own PNG decodes");
        assert_eq!(decoded, texture);
        // A non-square, translucent image too.
        let odd = Texture::from_rgba8(3, 2, (0..24).map(|i| i * 10).collect()).unwrap();
        assert_eq!(decode_image(&encode_png(&odd), usize::MAX).unwrap(), odd);
    }

    /// Build a PNG from raw (already filtered) scan lines.
    fn png_from(
        header: [u8; 5],
        width: u32,
        height: u32,
        extra: &[(&[u8; 4], &[u8])],
        raw: &[u8],
    ) -> Vec<u8> {
        let mut out = PNG_SIGNATURE.to_vec();
        let mut chunk = |kind: &[u8; 4], data: &[u8]| {
            out.extend_from_slice(&(data.len() as u32).to_be_bytes());
            out.extend_from_slice(kind);
            out.extend_from_slice(data);
            out.extend_from_slice(&crc32(&[kind, data]).to_be_bytes());
        };
        let mut ihdr = Vec::new();
        ihdr.extend_from_slice(&width.to_be_bytes());
        ihdr.extend_from_slice(&height.to_be_bytes());
        ihdr.extend_from_slice(&header);
        chunk(b"IHDR", &ihdr);
        for (kind, data) in extra {
            chunk(kind, data);
        }
        chunk(b"IDAT", &miniz_oxide::deflate::compress_to_vec_zlib(raw, 6));
        chunk(b"IEND", &[]);
        out
    }

    /// The colour types, bit depths and filters our own encoder never writes.
    #[test]
    fn png_colour_types_depths_and_filters_decode() {
        // 2x2 RGB8, each line with a different filter: Sub, then Paeth.
        // Line 0 raw: (10,20,30) (15,25,35) -> Sub: 10,20,30, 5,5,5.
        // Line 1 raw: (12,22,32) (20,30,40) -> Paeth against line 0:
        //   first pixel predicts `up`: 2,2,2; second: a=12 b=15 c=10,
        //   p = 17, nearest is b (15): 5; likewise for the other channels.
        let rgb = png_from(
            [8, 2, 0, 0, 0],
            2,
            2,
            &[],
            &[1, 10, 20, 30, 5, 5, 5, 4, 2, 2, 2, 5, 5, 5],
        );
        let t = decode_image(&rgb, usize::MAX).expect("RGB8");
        assert_eq!(
            t.pixels(),
            [10, 20, 30, 255, 15, 25, 35, 255, 12, 22, 32, 255, 20, 30, 40, 255]
        );

        // 4x1 palette image at 2 bits per pixel with a tRNS alpha for entry 1,
        // filter Up on the first line (the line above is all zero).
        let palette: [u8; 9] = [255, 0, 0, 0, 255, 0, 0, 0, 255];
        let indexed = png_from(
            [2, 3, 0, 0, 0],
            4,
            1,
            &[(b"PLTE", &palette), (b"tRNS", &[255, 128])],
            // Indices 0, 1, 2, 1 -> 0b00_01_10_01.
            &[2, 0b0001_1001],
        );
        let t = decode_image(&indexed, usize::MAX).expect("palette");
        assert_eq!(
            t.pixels(),
            [255, 0, 0, 255, 0, 255, 0, 128, 0, 0, 255, 255, 0, 255, 0, 128]
        );

        // 2x1 grey + alpha at 16 bits, filter Average.
        // Raw samples: (0x8000, 0xffff) (0x4000, 0x0000).
        // Average with left (no line above): first pixel as is; second pixel
        // byte i stores raw - floor(left / 2), i.e. 0x40 - 0x40, 0 - 0,
        // 0 - 0x7f and 0 - 0x7f.
        let grey = png_from(
            [16, 4, 0, 0, 0],
            2,
            1,
            &[],
            &[3, 0x80, 0x00, 0xff, 0xff, 0x00, 0x00, 0x81, 0x81],
        );
        let t = decode_image(&grey, usize::MAX).expect("grey + alpha 16");
        assert_eq!(t.pixels(), [128, 128, 128, 255, 64, 64, 64, 0]);

        // 1-bit greyscale with a colour key making black transparent.
        let bits = png_from(
            [1, 0, 0, 0, 0],
            3,
            1,
            &[(b"tRNS", &[0, 0])],
            &[0, 0b1010_0000],
        );
        let t = decode_image(&bits, usize::MAX).expect("1-bit grey");
        assert_eq!(
            t.pixels(),
            [255, 255, 255, 255, 0, 0, 0, 0, 255, 255, 255, 255]
        );
    }

    /// Adam7: a 5x5 greyscale image, every pixel a distinct value, split into
    /// its seven passes by hand.
    #[test]
    fn interlaced_png_decodes() {
        let value = |x: usize, y: usize| (y * 5 + x) as u8 * 10;
        let passes: [[usize; 4]; 7] = [
            [0, 0, 8, 8],
            [4, 0, 8, 8],
            [0, 4, 4, 8],
            [2, 0, 4, 4],
            [0, 2, 2, 4],
            [1, 0, 2, 2],
            [0, 1, 1, 2],
        ];
        let mut raw = Vec::new();
        for [x0, y0, dx, dy] in passes {
            for y in (y0..5).step_by(dy) {
                if x0 >= 5 {
                    continue;
                }
                raw.push(0);
                raw.extend((x0..5).step_by(dx).map(|x| value(x, y)));
            }
        }
        let png = png_from([8, 0, 0, 0, 1], 5, 5, &[], &raw);
        let t = decode_image(&png, usize::MAX).expect("interlaced");
        for y in 0..5 {
            for x in 0..5 {
                assert_eq!(t.texel(x as u32, y as u32)[0], value(x, y), "({x}, {y})");
            }
        }
    }

    /// A malformed image is an error, never a panic or an oversized allocation.
    #[test]
    fn malformed_images_are_rejected() {
        let good = encode_png(&Texture::checkerboard(16, 4, [255; 3], [0; 3]));
        assert!(decode_image(&good, usize::MAX).is_ok());
        // Every prefix, and every single-byte corruption of the header and the
        // start of the data: must return (the CRC catches the corruptions).
        for cut in 0..good.len() {
            assert!(decode_image(&good[..cut], usize::MAX).is_err(), "cut {cut}");
        }
        for i in 8..good.len().min(80) {
            let mut bad = good.clone();
            bad[i] ^= 0x55;
            assert!(decode_image(&bad, usize::MAX).is_err(), "byte {i}");
        }
        // Over budget: 16 * 16 * 4 = 1024 bytes needed.
        assert!(decode_image(&good, 1023).is_err());
        assert!(decode_image(&good, 1024).is_ok());
        // A header that claims a huge image is refused before any allocation.
        let huge = png_from([8, 6, 0, 0, 0], 100_000, 100_000, &[], &[0]);
        assert!(decode_image(&huge, usize::MAX).is_err());
        // Image data shorter or longer than the header implies.
        let short = png_from([8, 0, 0, 0, 0], 4, 4, &[], &[0, 1, 2, 3, 4]);
        assert!(decode_image(&short, usize::MAX).is_err());
        let long = png_from([8, 0, 0, 0, 0], 1, 1, &[], &[0, 1, 0, 2, 0, 3]);
        assert!(decode_image(&long, usize::MAX).is_err());
        // A bad filter byte, a palette index past the palette, no palette.
        let filter = png_from([8, 0, 0, 0, 0], 1, 1, &[], &[9, 1]);
        assert!(decode_image(&filter, usize::MAX).is_err());
        let index = png_from([8, 3, 0, 0, 0], 1, 1, &[(b"PLTE", &[1, 2, 3])], &[0, 5]);
        assert!(decode_image(&index, usize::MAX).is_err());
        let no_palette = png_from([8, 3, 0, 0, 0], 1, 1, &[], &[0, 0]);
        assert!(decode_image(&no_palette, usize::MAX).is_err());
        // Not an image at all, and truncated JPEGs.
        assert!(decode_image(b"GIF89a", usize::MAX).is_err());
        assert!(decode_image(&[], usize::MAX).is_err());
        assert!(decode_image(&[0xff, 0xd8], usize::MAX).is_err());
        assert!(decode_image(&[0xff, 0xd8, 0xff, 0xe0, 0, 16], usize::MAX).is_err());
    }
}
