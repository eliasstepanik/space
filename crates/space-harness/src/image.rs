//! Turning a [`Frame`] into a file: lossless PNG and (through `space-stream`'s
//! encoder) JPEG.
//!
//! The PNG writer needs no dependency: the image data is put in *stored*
//! deflate blocks, i.e. not compressed at all. A screenshot is therefore about
//! as large as its pixels (`width * height * 4` bytes plus a few per row), which
//! is fine for a test artefact and keeps a codec out of the build. Every PNG
//! reader accepts it.

use std::fmt;

use space_stream::JpegEncoder;

use crate::control::Frame;

/// An image format the harness can produce.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImageFormat {
    /// Lossless RGBA.
    Png,
    /// Lossy RGB (alpha is dropped).
    Jpeg,
}

impl ImageFormat {
    /// The format a file name asks for by its extension (`.png`, `.jpg`,
    /// `.jpeg`, any case), or `None`.
    #[must_use]
    pub fn from_file_name(name: &str) -> Option<Self> {
        let (_, extension) = name.rsplit_once('.')?;
        match extension.to_ascii_lowercase().as_str() {
            "png" => Some(Self::Png),
            "jpg" | "jpeg" => Some(Self::Jpeg),
            _ => None,
        }
    }

    /// The MIME type.
    #[must_use]
    pub const fn content_type(self) -> &'static str {
        match self {
            Self::Png => "image/png",
            Self::Jpeg => "image/jpeg",
        }
    }
}

/// Why an image could not be encoded.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ImageError {
    /// A zero dimension, or the pixel buffer is not `width * height * 4` bytes.
    BadBuffer,
    /// The image is too large for the format.
    TooLarge,
    /// The JPEG encoder failed.
    Encode(String),
}

impl fmt::Display for ImageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadBuffer => f.write_str("the pixel buffer does not match width*height*4"),
            Self::TooLarge => f.write_str("the image is too large for the format"),
            Self::Encode(why) => write!(f, "jpeg encode: {why}"),
        }
    }
}

impl std::error::Error for ImageError {}

/// The PNG file signature.
pub const PNG_SIGNATURE: [u8; 8] = [0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];

/// Largest payload of one stored deflate block.
const STORED_BLOCK: usize = 0xffff;

/// The CRC-32 (IEEE, reflected) lookup table PNG chunks use.
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

/// CRC-32 of `bytes`, continuing from `crc` (start with 0).
#[must_use]
pub fn crc32(crc: u32, bytes: &[u8]) -> u32 {
    let mut c = !crc;
    for &b in bytes {
        c = CRC_TABLE[usize::from((c as u8) ^ b)] ^ (c >> 8);
    }
    !c
}

/// Adler-32 of `bytes`, continuing from `adler` (start with 1).
#[must_use]
pub fn adler32(adler: u32, bytes: &[u8]) -> u32 {
    const MODULUS: u32 = 65_521;
    let (mut a, mut b) = (adler & 0xffff, adler >> 16);
    // 5552 is the most bytes that can be summed before `b` could overflow.
    for chunk in bytes.chunks(5552) {
        for &byte in chunk {
            a += u32::from(byte);
            b += a;
        }
        a %= MODULUS;
        b %= MODULUS;
    }
    (b << 16) | a
}

fn chunk(out: &mut Vec<u8>, kind: [u8; 4], data: &[u8]) {
    // The caller keeps every chunk under 2^31 bytes.
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    out.extend_from_slice(&kind);
    out.extend_from_slice(data);
    out.extend_from_slice(&crc32(crc32(0, &kind), data).to_be_bytes());
}

/// Encode tightly packed RGBA8 (top row first) as a PNG: 8-bit RGBA,
/// non-interlaced, filter 0, uncompressed. The pixels survive exactly.
///
/// # Errors
///
/// [`ImageError::BadBuffer`] when the buffer does not match the dimensions,
/// [`ImageError::TooLarge`] when the image data would not fit one `IDAT` chunk
/// (about 2 GiB).
pub fn encode_png(rgba: &[u8], width: u32, height: u32) -> Result<Vec<u8>, ImageError> {
    let row = u64::from(width) * 4;
    if width == 0 || height == 0 || rgba.len() as u64 != row * u64::from(height) {
        return Err(ImageError::BadBuffer);
    }
    // Each scanline is a filter byte (0 = none) followed by its pixels.
    let raw_len = (row + 1) * u64::from(height);
    let blocks = raw_len.div_ceil(STORED_BLOCK as u64);
    // zlib header (2) + per block header (5) + data + Adler-32 (4).
    let zlib_len = 2 + blocks * 5 + raw_len + 4;
    if width > i32::MAX as u32 || height > i32::MAX as u32 || zlib_len > i32::MAX as u64 {
        return Err(ImageError::TooLarge);
    }
    let (row, raw_len, zlib_len) = (row as usize, raw_len as usize, zlib_len as usize);

    let mut raw = Vec::with_capacity(raw_len);
    for line in rgba.chunks_exact(row) {
        raw.push(0);
        raw.extend_from_slice(line);
    }

    let mut zlib = Vec::with_capacity(zlib_len);
    // CM = 8 (deflate), 32 KiB window; FLG makes the pair a multiple of 31
    // with the "fastest" level hint.
    zlib.extend_from_slice(&[0x78, 0x01]);
    let mut blocks = raw.chunks(STORED_BLOCK).peekable();
    while let Some(block) = blocks.next() {
        let last = blocks.peek().is_none();
        zlib.push(u8::from(last)); // BFINAL, BTYPE = 00 (stored)
        let len = block.len() as u16;
        zlib.extend_from_slice(&len.to_le_bytes());
        zlib.extend_from_slice(&(!len).to_le_bytes());
        zlib.extend_from_slice(block);
    }
    zlib.extend_from_slice(&adler32(1, &raw).to_be_bytes());
    drop(raw);

    let mut out = Vec::with_capacity(PNG_SIGNATURE.len() + 25 + 12 + zlib.len() + 12);
    out.extend_from_slice(&PNG_SIGNATURE);
    let mut header = Vec::with_capacity(13);
    header.extend_from_slice(&width.to_be_bytes());
    header.extend_from_slice(&height.to_be_bytes());
    // Bit depth 8, colour type 6 (RGBA), deflate, adaptive filtering, no interlace.
    header.extend_from_slice(&[8, 6, 0, 0, 0]);
    chunk(&mut out, *b"IHDR", &header);
    chunk(&mut out, *b"IDAT", &zlib);
    chunk(&mut out, *b"IEND", &[]);
    Ok(out)
}

/// Decode a PNG **as [`encode_png`] writes it** back into
/// `(width, height, rgba)`: 8-bit RGBA, not interlaced, filter 0 on every
/// row, stored deflate blocks. Every checksum (chunk CRCs, the zlib Adler-32)
/// is verified. This is how tests check a screenshot pixel for pixel without
/// an image crate; it is not a general PNG reader, and anything outside that
/// subset is an `Err` saying what was found.
///
/// # Errors
///
/// A description of the first thing that is not as expected.
pub fn decode_stored_png(png: &[u8]) -> Result<(u32, u32, Vec<u8>), String> {
    let mut rest = png
        .strip_prefix(&PNG_SIGNATURE)
        .ok_or("not a PNG: wrong signature")?;
    let mut header: Option<(u32, u32)> = None;
    let mut zlib: Vec<u8> = Vec::new();
    let mut ended = false;
    while !rest.is_empty() {
        let (len, kind, tail) = match rest {
            [a, b, c, d, k0, k1, k2, k3, tail @ ..] => (
                u32::from_be_bytes([*a, *b, *c, *d]) as usize,
                [*k0, *k1, *k2, *k3],
                tail,
            ),
            _ => return Err("truncated chunk header".into()),
        };
        if ended {
            return Err("data after IEND".into());
        }
        let data = tail.get(..len).ok_or("truncated chunk data")?;
        let crc: [u8; 4] = tail
            .get(len..len.saturating_add(4))
            .and_then(|c| c.try_into().ok())
            .ok_or("truncated chunk CRC")?;
        if crc32(crc32(0, &kind), data) != u32::from_be_bytes(crc) {
            return Err(format!("bad CRC in {}", String::from_utf8_lossy(&kind)));
        }
        rest = &tail[len + 4..];
        match &kind {
            b"IHDR" => {
                let [w0, w1, w2, w3, h0, h1, h2, h3, 8, 6, 0, 0, 0] = data else {
                    return Err(format!("unsupported IHDR {data:?}"));
                };
                if header.is_some() {
                    return Err("two IHDR chunks".into());
                }
                header = Some((
                    u32::from_be_bytes([*w0, *w1, *w2, *w3]),
                    u32::from_be_bytes([*h0, *h1, *h2, *h3]),
                ));
            }
            b"IDAT" => zlib.extend_from_slice(data),
            b"IEND" => ended = true,
            other => {
                return Err(format!(
                    "unexpected chunk {}",
                    String::from_utf8_lossy(other)
                ))
            }
        }
    }
    let (width, height) = header.ok_or("no IHDR")?;
    if !ended {
        return Err("no IEND".into());
    }

    // The zlib stream: header, stored blocks, Adler-32.
    let mut blocks = zlib
        .strip_prefix(&[0x78, 0x01])
        .ok_or("unexpected zlib header")?;
    let mut raw = Vec::new();
    loop {
        let (head, len, not_len, tail) = match blocks {
            [head, l0, l1, n0, n1, tail @ ..] => (
                *head,
                u16::from_le_bytes([*l0, *l1]),
                u16::from_le_bytes([*n0, *n1]),
                tail,
            ),
            _ => return Err("truncated deflate block".into()),
        };
        if head > 1 || not_len != !len {
            return Err("not a stored deflate block".into());
        }
        let data = tail.get(..usize::from(len)).ok_or("truncated block data")?;
        raw.extend_from_slice(data);
        blocks = &tail[usize::from(len)..];
        if head == 1 {
            break;
        }
    }
    let adler: [u8; 4] = blocks.try_into().map_err(|_| "bad zlib trailer length")?;
    if adler32(1, &raw) != u32::from_be_bytes(adler) {
        return Err("bad Adler-32".into());
    }

    // Scanlines: a zero filter byte, then the row.
    let row = u64::from(width) * 4;
    if width == 0 || height == 0 || raw.len() as u64 != (row + 1) * u64::from(height) {
        return Err(format!(
            "{} bytes of image data do not make {width}x{height} RGBA",
            raw.len()
        ));
    }
    let row = row as usize;
    let mut rgba = Vec::with_capacity(raw.len());
    for line in raw.chunks_exact(row + 1) {
        let [0, pixels @ ..] = line else {
            return Err("a row uses a filter".into());
        };
        rgba.extend_from_slice(pixels);
    }
    Ok((width, height, rgba))
}

/// Encode tightly packed RGBA8 as a baseline JPEG at full size (alpha is
/// ignored). `quality` is clamped to `1..=100`.
///
/// # Errors
///
/// [`ImageError::BadBuffer`], [`ImageError::TooLarge`] (a JPEG dimension is at
/// most 65535) or [`ImageError::Encode`].
pub fn encode_jpeg(
    rgba: &[u8],
    width: u32,
    height: u32,
    quality: u8,
) -> Result<Vec<u8>, ImageError> {
    use space_stream::JpegError;
    match JpegEncoder::new(quality, 1).encode(rgba, width, height) {
        Ok(jpeg) => Ok(jpeg.bytes),
        Err(JpegError::BadBuffer) => Err(ImageError::BadBuffer),
        Err(JpegError::TooLarge) => Err(ImageError::TooLarge),
        Err(JpegError::Encode(why)) => Err(ImageError::Encode(why)),
    }
}

/// Encode `frame` in `format` (`jpeg_quality` only matters for JPEG).
///
/// # Errors
///
/// As [`encode_png`] / [`encode_jpeg`].
pub fn encode_frame(
    frame: &Frame,
    format: ImageFormat,
    jpeg_quality: u8,
) -> Result<Vec<u8>, ImageError> {
    match format {
        ImageFormat::Png => encode_png(&frame.rgba, frame.width, frame.height),
        ImageFormat::Jpeg => encode_jpeg(&frame.rgba, frame.width, frame.height, jpeg_quality),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checksums_match_their_reference_values() {
        // The standard check values of both algorithms.
        assert_eq!(crc32(0, b"123456789"), 0xcbf4_3926);
        assert_eq!(adler32(1, b"Wikipedia"), 0x11e6_0398);
        assert_eq!(crc32(0, b""), 0);
        assert_eq!(adler32(1, b""), 1);
        // Both can be continued.
        assert_eq!(crc32(crc32(0, b"1234"), b"56789"), 0xcbf4_3926);
        assert_eq!(adler32(adler32(1, b"Wiki"), b"pedia"), 0x11e6_0398);
        // Long inputs take the modulus in time: one megabyte of 0xff.
        let big = vec![0xffu8; 1 << 20];
        let (mut a, mut b) = (1u64, 0u64);
        for _ in 0..big.len() {
            a = (a + 0xff) % 65_521;
            b = (b + a) % 65_521;
        }
        assert_eq!(u64::from(adler32(1, &big)), (b << 16) | a);
    }

    /// The smallest image, byte for byte: a known vector, so the container is
    /// right independently of the decoder the other tests use.
    #[test]
    fn a_one_pixel_png_is_exactly_these_bytes() {
        let png = encode_png(&[0x11, 0x22, 0x33, 0x44], 1, 1).unwrap();
        let mut want: Vec<u8> = PNG_SIGNATURE.to_vec();
        // IHDR: 1x1, 8-bit RGBA.
        want.extend_from_slice(&[0, 0, 0, 13]);
        want.extend_from_slice(b"IHDR");
        want.extend_from_slice(&[0, 0, 0, 1, 0, 0, 0, 1, 8, 6, 0, 0, 0]);
        want.extend_from_slice(&[0x1f, 0x15, 0xc4, 0x89]);
        // IDAT: zlib header, one final stored block of 5 bytes (filter byte +
        // pixel), Adler-32 of those 5 bytes.
        want.extend_from_slice(&[0, 0, 0, 16]);
        want.extend_from_slice(b"IDAT");
        let idat = [
            0x78, 0x01, 0x01, 0x05, 0x00, 0xfa, 0xff, 0x00, 0x11, 0x22, 0x33, 0x44, 0x01, 0x59,
            0x00, 0xab,
        ];
        want.extend_from_slice(&idat);
        want.extend_from_slice(&crc32(crc32(0, b"IDAT"), &idat).to_be_bytes());
        want.extend_from_slice(&[0, 0, 0, 0]);
        want.extend_from_slice(b"IEND");
        want.extend_from_slice(&[0xae, 0x42, 0x60, 0x82]);
        assert_eq!(png, want);
    }

    /// Deterministic noise: no two neighbouring bytes are related, so a wrong
    /// offset, stride or block boundary changes the result.
    fn noise(len: usize, mut state: u32) -> Vec<u8> {
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                (state >> 24) as u8
            })
            .collect()
    }

    /// PNG output decodes back to the same pixels, for sizes on both sides of
    /// every boundary the writer has: one row, one column, image data shorter
    /// and longer than one stored block (65535 bytes: a 16383-pixel row with
    /// its filter byte is 65533, so two of them split mid-row), and many
    /// blocks.
    #[test]
    fn png_decodes_back_to_the_same_pixels() {
        for (i, (w, h)) in [
            (1u32, 1u32),
            (1, 7),
            (7, 1),
            (3, 5),
            (64, 48),
            (16_383, 1),
            (16_383, 2),
            (127, 129),
            (640, 360),
        ]
        .into_iter()
        .enumerate()
        {
            let rgba = noise((w * h * 4) as usize, 0x9e37_79b9 ^ i as u32);
            let png = encode_png(&rgba, w, h).unwrap();
            assert_eq!(png[..8], PNG_SIGNATURE);
            let (dw, dh, pixels) =
                decode_stored_png(&png).unwrap_or_else(|e| panic!("{w}x{h}: {e}"));
            assert_eq!((dw, dh), (w, h));
            assert!(pixels == rgba, "{w}x{h}: pixels differ");
            // About as large as the pixels: 5 bytes per block, 1 per row.
            let raw = (w as usize * 4 + 1) * h as usize;
            assert_eq!(
                png.len(),
                8 + 25 + 12 + 2 + raw.div_ceil(65_535) * 5 + raw + 4 + 12
            );
        }
    }

    /// A stored block holds at most 65535 bytes; image data of exactly that
    /// length is one final block, and one byte more starts a second one.
    #[test]
    fn stored_blocks_split_at_exactly_65535_bytes() {
        // A 1-pixel-wide image has 5-byte rows (filter byte + pixel), so
        // 13107 rows are exactly 65535 bytes.
        for (rows, blocks) in [
            (13_106u32, 1usize),
            (13_107, 1),
            (13_108, 2),
            (26_214, 2),
            (26_215, 3),
        ] {
            let rgba = noise(rows as usize * 4, rows);
            let png = encode_png(&rgba, 1, rows).unwrap();
            let raw = rows as usize * 5;
            assert_eq!(
                png.len(),
                8 + 25 + 12 + 2 + blocks * 5 + raw + 4 + 12,
                "{rows} rows"
            );
            let (w, h, pixels) = decode_stored_png(&png).unwrap();
            assert_eq!((w, h), (1, rows));
            assert!(pixels == rgba, "{rows} rows: pixels differ");
        }
    }

    /// The decoder the tests rely on does check what it reads: a flipped bit
    /// anywhere in the file is noticed, and so is a file cut short.
    #[test]
    fn the_test_decoder_notices_corruption() {
        let rgba = noise(5 * 4 * 4, 1);
        let png = encode_png(&rgba, 5, 4).unwrap();
        assert!(decode_stored_png(&png).is_ok());
        for at in 0..png.len() {
            let mut bad = png.clone();
            bad[at] ^= 0x10;
            assert!(
                decode_stored_png(&bad).is_err(),
                "a flip at byte {at} went unnoticed"
            );
        }
        for len in 0..png.len() {
            assert!(decode_stored_png(&png[..len]).is_err(), "cut at {len}");
        }
        let mut longer = png.clone();
        longer.push(0);
        assert!(decode_stored_png(&longer).is_err());
        assert!(decode_stored_png(b"").is_err());
    }

    /// JPEG output is a JFIF file of the frame's size: SOI, a JFIF APP0
    /// segment, the dimensions in the frame header, EOI at the end.
    #[test]
    fn jpeg_is_a_valid_jfif_of_the_right_size() {
        let (w, h) = (64u32, 48u32);
        let rgba: Vec<u8> = (0..w * h)
            .flat_map(|i| [(i % w * 4) as u8, (i / w * 5) as u8, 128, 255])
            .collect();
        let jpeg = encode_jpeg(&rgba, w, h, 85).unwrap();
        assert_eq!(jpeg[..4], [0xff, 0xd8, 0xff, 0xe0], "SOI then APP0");
        assert_eq!(&jpeg[6..11], b"JFIF\0");
        assert_eq!(jpeg[jpeg.len() - 2..], [0xff, 0xd9], "EOI");
        assert_eq!(space_stream::jpeg_dimensions(&jpeg), Some((64, 48)));
        // Quality is honoured (and clamped rather than refused).
        let small = encode_jpeg(&rgba, w, h, 10).unwrap();
        assert!(small.len() < jpeg.len());
        assert!(encode_jpeg(&rgba, w, h, 0).is_ok());
        assert!(encode_jpeg(&rgba, w, h, 255).is_ok());

        let frame = Frame::new(3, w, h, rgba.clone()).unwrap();
        assert_eq!(encode_frame(&frame, ImageFormat::Jpeg, 85).unwrap(), jpeg);
        let png = encode_frame(&frame, ImageFormat::Png, 85).unwrap();
        assert_eq!(decode_stored_png(&png).unwrap(), (w, h, rgba));
    }

    #[test]
    fn mismatched_buffers_are_refused() {
        assert_eq!(encode_png(&[], 0, 0), Err(ImageError::BadBuffer));
        assert_eq!(encode_png(&[0; 4], 0, 1), Err(ImageError::BadBuffer));
        assert_eq!(encode_png(&[0; 15], 2, 2), Err(ImageError::BadBuffer));
        assert_eq!(encode_png(&[0; 17], 2, 2), Err(ImageError::BadBuffer));
        assert_eq!(encode_jpeg(&[0; 15], 2, 2, 80), Err(ImageError::BadBuffer));
        assert!(encode_png(&[0; 16], 2, 2).is_ok());
        assert!(ImageError::BadBuffer.to_string().contains("width*height*4"));
    }

    #[test]
    fn formats_are_told_by_the_extension() {
        for (name, format) in [
            ("a.png", Some(ImageFormat::Png)),
            ("dir/b.PNG", Some(ImageFormat::Png)),
            ("c.jpg", Some(ImageFormat::Jpeg)),
            ("c.d.JPeG", Some(ImageFormat::Jpeg)),
            ("png", None),
            ("x.gif", None),
            ("x.", None),
            ("", None),
        ] {
            assert_eq!(ImageFormat::from_file_name(name), format, "{name:?}");
        }
        assert_eq!(ImageFormat::Png.content_type(), "image/png");
        assert_eq!(ImageFormat::Jpeg.content_type(), "image/jpeg");
    }
}
