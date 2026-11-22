//! Pure-Rust JPEG encoding of RGBA frames (wraps `jpeg-encoder`; no C code).
//!
//! This runs once per streamed frame, so it avoids per-frame work it does not
//! need: the downscaled image is written as RGB (alpha is never encoded) into a
//! buffer that is reused from frame to frame, the common half-resolution case
//! has its own tight loop, and the output buffer is sized from the previous
//! frame instead of growing from empty.

use std::cell::RefCell;
use std::fmt;

/// Why a frame could not be encoded.
#[derive(Debug, PartialEq, Eq)]
pub enum JpegError {
    /// `rgba.len() != width * height * 4`, or a zero dimension.
    BadBuffer,
    /// The (downscaled) image does not fit JPEG's 16-bit dimensions.
    TooLarge,
    /// The underlying encoder failed.
    Encode(String),
}

impl fmt::Display for JpegError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadBuffer => f.write_str("RGBA buffer does not match width*height*4"),
            Self::TooLarge => f.write_str("image too large for JPEG"),
            Self::Encode(e) => write!(f, "jpeg encode: {e}"),
        }
    }
}

impl std::error::Error for JpegError {}

/// An encoded frame and the dimensions of the encoded (possibly downscaled)
/// image.
#[derive(Clone, Debug)]
pub struct EncodedJpeg {
    pub bytes: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

/// Working memory an encoder keeps between frames: the downscaled RGB image and
/// how large the last JPEG was (the next one is almost always about the same).
///
/// [`JpegEncoder::encode`] keeps one per thread automatically; use
/// [`JpegEncoder::encode_with`] to own it explicitly.
#[derive(Debug, Default)]
pub struct EncodeScratch {
    rgb: Vec<u8>,
    last_len: usize,
}

thread_local! {
    /// The scratch [`JpegEncoder::encode`] reuses: a render loop encodes every
    /// frame on the same thread, so this removes its per-frame allocations
    /// without changing the call site.
    static SCRATCH: RefCell<EncodeScratch> = RefCell::new(EncodeScratch::default());
}

/// RGBA → JPEG with a configurable quality and integer downscale factor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct JpegEncoder {
    quality: u8,
    downscale: u32,
}

impl Default for JpegEncoder {
    /// Quality 80, half resolution.
    fn default() -> Self {
        Self::new(80, 2)
    }
}

impl JpegEncoder {
    /// `quality` is clamped to `1..=100`; `downscale` (box-filter divisor of
    /// each dimension) to `1..=16`. `downscale == 1` encodes at full size.
    pub fn new(quality: u8, downscale: u32) -> Self {
        Self {
            quality: quality.clamp(1, 100),
            downscale: downscale.clamp(1, 16),
        }
    }

    pub fn quality(&self) -> u8 {
        self.quality
    }

    pub fn downscale(&self) -> u32 {
        self.downscale
    }

    /// Dimensions of the encoded image for a `width`×`height` source.
    pub fn output_size(&self, width: u32, height: u32) -> (u32, u32) {
        (
            width.div_ceil(self.downscale).max(1),
            height.div_ceil(self.downscale).max(1),
        )
    }

    /// Encode a tightly packed RGBA8 image (alpha is ignored).
    ///
    /// Reuses a per-thread [`EncodeScratch`], so calling this every frame from
    /// one thread does not allocate a fresh downscale buffer each time.
    pub fn encode(&self, rgba: &[u8], width: u32, height: u32) -> Result<EncodedJpeg, JpegError> {
        SCRATCH.with(|scratch| self.encode_with(rgba, width, height, &mut scratch.borrow_mut()))
    }

    /// [`JpegEncoder::encode`] with caller-owned working memory.
    pub fn encode_with(
        &self,
        rgba: &[u8],
        width: u32,
        height: u32,
        scratch: &mut EncodeScratch,
    ) -> Result<EncodedJpeg, JpegError> {
        if width == 0
            || height == 0
            || rgba.len() as u64 != u64::from(width) * u64::from(height) * 4
        {
            return Err(JpegError::BadBuffer);
        }
        let (ow, oh) = self.output_size(width, height);
        let (ow16, oh16) = (
            u16::try_from(ow).map_err(|_| JpegError::TooLarge)?,
            u16::try_from(oh).map_err(|_| JpegError::TooLarge)?,
        );
        let (pixels, color): (&[u8], _) = if self.downscale == 1 {
            // No copy at all: the encoder skips the alpha byte itself.
            (rgba, jpeg_encoder::ColorType::Rgba)
        } else {
            box_downscale(
                rgba,
                width as usize,
                height as usize,
                self.downscale as usize,
                &mut scratch.rgb,
            );
            (&scratch.rgb, jpeg_encoder::ColorType::Rgb)
        };
        // Start at the size of the previous frame (plus slack) so the output
        // does not grow by repeated reallocation; a first frame guesses from
        // the pixel count.
        let guess = if scratch.last_len > 0 {
            scratch.last_len + scratch.last_len / 8
        } else {
            pixels.len() / 8
        };
        let mut bytes = Vec::with_capacity(guess + 1024);
        jpeg_encoder::Encoder::new(&mut bytes, self.quality)
            .encode(pixels, ow16, oh16, color)
            .map_err(|e| JpegError::Encode(e.to_string()))?;
        scratch.last_len = bytes.len();
        Ok(EncodedJpeg {
            bytes,
            width: ow,
            height: oh,
        })
    }
}

/// Average `factor`×`factor` blocks of a tightly packed RGBA image into tightly
/// packed **RGB** in `out` (edge blocks average the pixels that exist; alpha is
/// dropped). `out` is cleared first and its allocation reused.
///
/// `rgba.len()` must be `width * height * 4` with both dimensions non-zero and
/// `factor >= 1` ([`JpegEncoder::encode_with`] checks this).
fn box_downscale(rgba: &[u8], width: usize, height: usize, factor: usize, out: &mut Vec<u8>) {
    let (ow, oh) = (width.div_ceil(factor), height.div_ceil(factor));
    out.clear();
    out.resize(ow * oh * 3, 0);
    if factor == 2 {
        halve(rgba, width, out);
    } else {
        box_downscale_generic(rgba, width, height, factor, out);
    }
}

/// The 2×2 case of [`box_downscale`], which is what the stream uses by default:
/// rows are walked two at a time and pixels in fixed-size chunks, so the inner
/// loop has no per-pixel index arithmetic or block-edge clamping.
fn halve(rgba: &[u8], width: usize, out: &mut [u8]) {
    let stride = width * 4;
    let out_stride = width.div_ceil(2) * 3;
    // `chunks` (not `chunks_exact`): an odd height leaves one last single row.
    for (out_row, rows) in out
        .chunks_exact_mut(out_stride)
        .zip(rgba.chunks(stride * 2))
    {
        if rows.len() == stride * 2 {
            let (top, bottom) = rows.split_at(stride);
            halve_two_rows(top, bottom, out_row);
        } else {
            halve_one_row(rows, out_row);
        }
    }
}

/// Average 2×2 blocks of two full rows into one row of RGB pixels (an odd
/// width leaves a 1×2 block for the last output pixel). Halves round up, like
/// the generic filter's `(sum + n / 2) / n`.
fn halve_two_rows(top: &[u8], bottom: &[u8], out: &mut [u8]) {
    // Two RGBA pixels per chunk; the tail is the single pixel of an odd width.
    let (top_pairs, top_last) = top.as_chunks::<8>();
    let (bottom_pairs, bottom_last) = bottom.as_chunks::<8>();
    let (out_pairs, out_last) = out.as_chunks_mut::<3>().0.split_at_mut(top_pairs.len());
    for ((o, t), b) in out_pairs.iter_mut().zip(top_pairs).zip(bottom_pairs) {
        for c in 0..3 {
            let sum = u16::from(t[c]) + u16::from(t[c + 4]) + u16::from(b[c]) + u16::from(b[c + 4]);
            o[c] = ((sum + 2) / 4) as u8;
        }
    }
    if let Some(o) = out_last.first_mut() {
        // `zip` stops after the three colour bytes: alpha is dropped.
        for ((o, t), b) in o.iter_mut().zip(top_last).zip(bottom_last) {
            *o = (u16::from(*t) + u16::from(*b)).div_ceil(2) as u8;
        }
    }
}

/// The last row of an odd-height image: 2×1 blocks (and a final single pixel
/// when the width is odd too).
fn halve_one_row(row: &[u8], out: &mut [u8]) {
    let (row_pairs, row_last) = row.as_chunks::<8>();
    let (out_pairs, out_last) = out.as_chunks_mut::<3>().0.split_at_mut(row_pairs.len());
    for (o, t) in out_pairs.iter_mut().zip(row_pairs) {
        for c in 0..3 {
            o[c] = (u16::from(t[c]) + u16::from(t[c + 4])).div_ceil(2) as u8;
        }
    }
    if let Some(o) = out_last.first_mut() {
        for (o, t) in o.iter_mut().zip(row_last) {
            *o = *t;
        }
    }
}

/// [`box_downscale`] for any factor (the reference the 2×2 path is tested
/// against). `out` is already sized.
fn box_downscale_generic(rgba: &[u8], width: usize, height: usize, factor: usize, out: &mut [u8]) {
    let ow = width.div_ceil(factor);
    for (i, o) in out.as_chunks_mut::<3>().0.iter_mut().enumerate() {
        let (ox, oy) = (i % ow, i / ow);
        let mut sum = [0u32; 3];
        let mut n = 0u32;
        for y in oy * factor..((oy + 1) * factor).min(height) {
            let row = &rgba[y * width * 4..(y + 1) * width * 4];
            let block = &row[ox * factor * 4..((ox + 1) * factor).min(width) * 4];
            for px in block.as_chunks::<4>().0 {
                // `zip` stops after the three colour bytes: alpha is dropped.
                for (s, v) in sum.iter_mut().zip(px) {
                    *s += u32::from(*v);
                }
                n += 1;
            }
        }
        for (o, s) in o.iter_mut().zip(sum) {
            *o = ((s + n / 2) / n) as u8;
        }
    }
}

/// Read the pixel dimensions from a JPEG's SOF marker, or `None` if `jpeg` is
/// not a well-formed JPEG header. Lets tests and consumers check a frame
/// without a full decoder.
pub fn jpeg_dimensions(jpeg: &[u8]) -> Option<(u16, u16)> {
    if jpeg.get(..2)? != [0xff, 0xd8] {
        return None;
    }
    let mut i = 2;
    loop {
        // Skip fill bytes, find the next marker.
        while *jpeg.get(i)? != 0xff {
            i += 1;
        }
        while *jpeg.get(i)? == 0xff {
            i += 1;
        }
        let marker = *jpeg.get(i)?;
        i += 1;
        match marker {
            0x00 | 0x01 | 0xd0..=0xd8 => continue, // no length field
            0xd9 | 0xda => return None,            // EOI/SOS before any SOF
            _ => {}
        }
        let len = usize::from(u16::from_be_bytes([*jpeg.get(i)?, *jpeg.get(i + 1)?]));
        // SOF0..SOF15 except DHT(c4), JPG(c8), DAC(cc).
        if (0xc0..=0xcf).contains(&marker) && !matches!(marker, 0xc4 | 0xc8 | 0xcc) {
            let h = u16::from_be_bytes([*jpeg.get(i + 3)?, *jpeg.get(i + 4)?]);
            let w = u16::from_be_bytes([*jpeg.get(i + 5)?, *jpeg.get(i + 6)?]);
            return Some((w, h));
        }
        i += len;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A smooth-but-not-flat test image.
    fn gradient(w: u32, h: u32) -> Vec<u8> {
        let mut v = Vec::new();
        for y in 0..h {
            for x in 0..w {
                v.extend_from_slice(&[
                    (x * 255 / w) as u8,
                    (y * 255 / h) as u8,
                    ((x * 7 + y * 13) % 256) as u8,
                    255,
                ]);
            }
        }
        v
    }

    /// Deterministic noise: every byte differs from its neighbours, so a wrong
    /// pixel offset or a swapped channel changes the result.
    fn noise(w: usize, h: usize) -> Vec<u8> {
        let mut state = 0x2545_f491u32;
        (0..w * h * 4)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                (state >> 24) as u8
            })
            .collect()
    }

    fn downscaled(rgba: &[u8], w: usize, h: usize, factor: usize) -> Vec<u8> {
        let mut out = vec![0xaa; 7]; // stale contents must not leak through
        box_downscale(rgba, w, h, factor, &mut out);
        out
    }

    #[test]
    fn jpeg_round_trip_dimensions_full_and_half() {
        let img = gradient(64, 48);
        let full = JpegEncoder::new(80, 1).encode(&img, 64, 48).unwrap();
        assert_eq!((full.width, full.height), (64, 48));
        assert_eq!(&full.bytes[..2], [0xff, 0xd8], "SOI");
        assert_eq!(&full.bytes[full.bytes.len() - 2..], [0xff, 0xd9], "EOI");
        assert_eq!(jpeg_dimensions(&full.bytes), Some((64, 48)));

        let half = JpegEncoder::default().encode(&img, 64, 48).unwrap();
        assert_eq!((half.width, half.height), (32, 24));
        assert_eq!(jpeg_dimensions(&half.bytes), Some((32, 24)));
        assert!(half.bytes.len() < full.bytes.len(), "half-res is smaller");
    }

    #[test]
    fn odd_sizes_round_up() {
        let img = gradient(33, 17);
        let enc = JpegEncoder::new(80, 2).encode(&img, 33, 17).unwrap();
        assert_eq!((enc.width, enc.height), (17, 9));
        assert_eq!(jpeg_dimensions(&enc.bytes), Some((17, 9)));
    }

    #[test]
    fn quality_band_orders_output_size() {
        let img = gradient(128, 96);
        let size = |q| {
            JpegEncoder::new(q, 1)
                .encode(&img, 128, 96)
                .unwrap()
                .bytes
                .len()
        };
        let (low, mid, high) = (size(10), size(60), size(95));
        assert!(low < mid && mid < high, "{low} < {mid} < {high}");
        assert_eq!(JpegEncoder::default().quality(), 80);
        assert_eq!(JpegEncoder::new(0, 0), JpegEncoder::new(1, 1), "clamped");
        assert_eq!(JpegEncoder::new(255, 99).quality(), 100);
    }

    #[test]
    fn bad_buffers_are_rejected() {
        let e = JpegEncoder::default();
        assert_eq!(e.encode(&[0; 7], 2, 1).unwrap_err(), JpegError::BadBuffer);
        assert_eq!(e.encode(&[], 0, 0).unwrap_err(), JpegError::BadBuffer);
    }

    #[test]
    fn box_filter_averages_blocks() {
        // 2x2 of (0,0,0,255) and (100,200,40,255) -> one averaged RGB pixel
        // (the output has no alpha channel: JPEG never encodes it).
        let img = [
            0, 0, 0, 255, 100, 200, 40, 255, 0, 0, 0, 255, 100, 200, 40, 255,
        ];
        assert_eq!(downscaled(&img, 2, 2, 2), [50, 100, 20]);
        // The generic path agrees, and rounds half up like the 2x2 path.
        let mut generic = [0u8; 3];
        box_downscale_generic(&img, 2, 2, 2, &mut generic);
        assert_eq!(generic, [50, 100, 20]);
        let img = [1, 2, 3, 0, 2, 2, 4, 0];
        assert_eq!(downscaled(&img, 2, 1, 2), [2, 2, 4], "(1+2+1)/2, (3+4+1)/2");
    }

    /// The specialised 2x2 loop must produce exactly what the generic box
    /// filter does, including the partial blocks along an odd right/bottom edge.
    #[test]
    fn halving_matches_the_generic_box_filter() {
        for (w, h) in [
            (1, 1),
            (2, 1),
            (1, 2),
            (2, 2),
            (3, 3),
            (5, 4),
            (4, 5),
            (7, 1),
            (1, 7),
            (33, 17),
            (64, 48),
        ] {
            let img = noise(w, h);
            let fast = downscaled(&img, w, h, 2);
            let mut reference = vec![0u8; w.div_ceil(2) * h.div_ceil(2) * 3];
            box_downscale_generic(&img, w, h, 2, &mut reference);
            assert_eq!(fast, reference, "{w}x{h}");
        }
    }

    #[test]
    fn other_factors_average_whole_and_partial_blocks() {
        // 4x1 by 3: one full block of three pixels and one partial block.
        let img = [10, 20, 30, 0, 20, 30, 40, 0, 30, 40, 50, 0, 7, 8, 9, 0];
        assert_eq!(downscaled(&img, 4, 1, 3), [20, 30, 40, 7, 8, 9]);
        let img = noise(10, 7);
        assert_eq!(downscaled(&img, 10, 7, 4).len(), 3 * 2 * 3);
        assert_eq!(downscaled(&img, 10, 7, 16).len(), 3);
    }

    /// The scratch is reused across frames (no fresh downscale buffer per
    /// frame) and reusing it never changes the encoded bytes.
    #[test]
    fn scratch_is_reused_and_does_not_change_the_output() {
        let encoder = JpegEncoder::default();
        let (a, b) = (gradient(64, 48), noise(64, 48));
        let mut scratch = EncodeScratch::default();
        let first = encoder.encode_with(&a, 64, 48, &mut scratch).unwrap();
        let buffer = scratch.rgb.as_ptr();
        assert_eq!(scratch.rgb.len(), 32 * 24 * 3, "RGB, not RGBA");
        assert_eq!(scratch.last_len, first.bytes.len());

        let second = encoder.encode_with(&b, 64, 48, &mut scratch).unwrap();
        assert_eq!(scratch.rgb.as_ptr(), buffer, "same allocation reused");
        assert_ne!(first.bytes, second.bytes);

        // A dirty scratch, a fresh one and the per-thread one all agree.
        let again = encoder.encode_with(&a, 64, 48, &mut scratch).unwrap();
        assert_eq!(again.bytes, first.bytes);
        let fresh = encoder
            .encode_with(&a, 64, 48, &mut EncodeScratch::default())
            .unwrap();
        assert_eq!(fresh.bytes, first.bytes);
        assert_eq!(encoder.encode(&a, 64, 48).unwrap().bytes, first.bytes);
        // A smaller frame afterwards must not see stale pixels.
        let small = encoder.encode_with(&a[..16 * 12 * 4], 16, 12, &mut scratch);
        assert_eq!(scratch.rgb.len(), 8 * 6 * 3);
        assert_eq!(jpeg_dimensions(&small.unwrap().bytes), Some((8, 6)));
    }

    /// Full-size encoding reads the RGBA buffer directly; it must match
    /// encoding the same pixels with the alpha bytes changed.
    #[test]
    fn alpha_is_ignored() {
        let opaque = gradient(40, 24);
        let mut clear = opaque.clone();
        for px in clear.as_chunks_mut::<4>().0 {
            px[3] = 0;
        }
        for factor in [1, 2, 3] {
            let encoder = JpegEncoder::new(70, factor);
            assert_eq!(
                encoder.encode(&opaque, 40, 24).unwrap().bytes,
                encoder.encode(&clear, 40, 24).unwrap().bytes,
                "factor {factor}"
            );
        }
    }

    #[test]
    fn dimensions_rejects_garbage() {
        assert_eq!(jpeg_dimensions(&[]), None);
        assert_eq!(jpeg_dimensions(b"not a jpeg"), None);
        assert_eq!(jpeg_dimensions(&[0xff, 0xd8]), None);
    }
}
