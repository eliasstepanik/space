//! Pure-Rust JPEG encoding of RGBA frames (wraps `jpeg-encoder`; no C code).

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
    pub fn encode(&self, rgba: &[u8], width: u32, height: u32) -> Result<EncodedJpeg, JpegError> {
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
        let scaled;
        let pixels: &[u8] = if self.downscale == 1 {
            rgba
        } else {
            scaled = box_downscale(rgba, width, height, self.downscale);
            &scaled
        };
        let mut bytes = Vec::new();
        jpeg_encoder::Encoder::new(&mut bytes, self.quality)
            .encode(pixels, ow16, oh16, jpeg_encoder::ColorType::Rgba)
            .map_err(|e| JpegError::Encode(e.to_string()))?;
        Ok(EncodedJpeg {
            bytes,
            width: ow,
            height: oh,
        })
    }
}

/// Average `factor`×`factor` blocks (edge blocks average what exists).
fn box_downscale(rgba: &[u8], width: u32, height: u32, factor: u32) -> Vec<u8> {
    let (w, h, f) = (width as usize, height as usize, factor as usize);
    let (ow, oh) = (w.div_ceil(f), h.div_ceil(f));
    let mut out = Vec::with_capacity(ow * oh * 4);
    for oy in 0..oh {
        for ox in 0..ow {
            let mut sum = [0u32; 4];
            let mut n = 0u32;
            for y in oy * f..((oy + 1) * f).min(h) {
                for x in ox * f..((ox + 1) * f).min(w) {
                    let i = (y * w + x) * 4;
                    for (s, v) in sum.iter_mut().zip(&rgba[i..i + 4]) {
                        *s += u32::from(*v);
                    }
                    n += 1;
                }
            }
            out.extend(sum.map(|s| ((s + n / 2) / n) as u8));
        }
    }
    out
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
        // 2x2 of (0,0,0,255) and (100,200,40,255) -> one averaged pixel.
        let img = [
            0, 0, 0, 255, 100, 200, 40, 255, 0, 0, 0, 255, 100, 200, 40, 255,
        ];
        assert_eq!(box_downscale(&img, 2, 2, 2), [50, 100, 20, 255]);
    }

    #[test]
    fn dimensions_rejects_garbage() {
        assert_eq!(jpeg_dimensions(&[]), None);
        assert_eq!(jpeg_dimensions(b"not a jpeg"), None);
        assert_eq!(jpeg_dimensions(&[0xff, 0xd8]), None);
    }
}
