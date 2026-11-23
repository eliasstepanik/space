//! Strict, hand-rolled byte codec shared by the physics components' wire forms.
//!
//! Every physics component ([`crate::body`], [`crate::spring`],
//! [`crate::articulation`]) is encoded with these two types:
//!
//! * [`Writer`] appends big-endian fixed-width integers and the raw IEEE-754
//!   bits of every float (the same byte order as [`space_protocol::codec`] and
//!   [`crate::wire`]), so an encoding is lossless and canonical: two equal
//!   values always produce the same bytes, which is what lets
//!   [`crate::SimWorld::state_hash`] hash the encodings directly.
//! * [`Reader`] is the **total** decoder for untrusted input: every read is
//!   bounds checked, every float can be required to be finite and in a range,
//!   every count is checked against a cap *before* anything is allocated, and
//!   [`Reader::finish`] rejects trailing bytes. A decoder built on it can fail,
//!   but it can never panic, loop or allocate in proportion to a length a peer
//!   wrote.

use space_protocol::CodecError;

/// Shorthand for the error every decoder returns.
pub(crate) fn invalid(what: impl Into<String>) -> CodecError {
    CodecError::InvalidPayload(what.into())
}

/// Appends big-endian fields to a byte buffer.
#[derive(Debug, Default)]
pub(crate) struct Writer {
    out: Vec<u8>,
}

impl Writer {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn u8(&mut self, v: u8) {
        self.out.push(v);
    }

    pub(crate) fn u16(&mut self, v: u16) {
        self.out.extend_from_slice(&v.to_be_bytes());
    }

    pub(crate) fn u32(&mut self, v: u32) {
        self.out.extend_from_slice(&v.to_be_bytes());
    }

    pub(crate) fn u64(&mut self, v: u64) {
        self.out.extend_from_slice(&v.to_be_bytes());
    }

    pub(crate) fn i64(&mut self, v: i64) {
        self.out.extend_from_slice(&v.to_be_bytes());
    }

    pub(crate) fn f32(&mut self, v: f32) {
        self.u32(v.to_bits());
    }

    pub(crate) fn f32s(&mut self, v: &[f32]) {
        for &x in v {
            self.f32(x);
        }
    }

    pub(crate) fn bytes(&mut self, v: &[u8]) {
        self.out.extend_from_slice(v);
    }

    pub(crate) fn into_bytes(self) -> Vec<u8> {
        self.out
    }
}

/// Reads big-endian fields from an untrusted byte slice.
#[derive(Debug)]
pub(crate) struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
    what: &'static str,
}

impl<'a> Reader<'a> {
    /// A reader over `buf`; `what` names the component in error messages.
    pub(crate) fn new(buf: &'a [u8], what: &'static str) -> Self {
        Self { buf, pos: 0, what }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], CodecError> {
        let end = self
            .pos
            .checked_add(n)
            .filter(|&end| end <= self.buf.len())
            .ok_or_else(|| {
                invalid(format!(
                    "{}: truncated ({} bytes, needed {} more at offset {})",
                    self.what,
                    self.buf.len(),
                    n,
                    self.pos
                ))
            })?;
        let s = &self.buf[self.pos..end];
        self.pos = end;
        Ok(s)
    }

    pub(crate) fn u8(&mut self) -> Result<u8, CodecError> {
        Ok(self.take(1)?[0])
    }

    pub(crate) fn u16(&mut self) -> Result<u16, CodecError> {
        let b = self.take(2)?;
        Ok(u16::from_be_bytes([b[0], b[1]]))
    }

    pub(crate) fn u32(&mut self) -> Result<u32, CodecError> {
        let b = self.take(4)?;
        Ok(u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
    }

    pub(crate) fn u64(&mut self) -> Result<u64, CodecError> {
        let b = self.take(8)?;
        let mut a = [0u8; 8];
        a.copy_from_slice(b);
        Ok(u64::from_be_bytes(a))
    }

    pub(crate) fn i64(&mut self) -> Result<i64, CodecError> {
        let b = self.take(8)?;
        let mut a = [0u8; 8];
        a.copy_from_slice(b);
        Ok(i64::from_be_bytes(a))
    }

    pub(crate) fn bytes(&mut self, n: usize) -> Result<&'a [u8], CodecError> {
        self.take(n)
    }

    /// A finite `f32` in `min..=max`.
    pub(crate) fn f32_in(&mut self, field: &str, min: f32, max: f32) -> Result<f32, CodecError> {
        let v = f32::from_bits(self.u32()?);
        if !v.is_finite() || v < min || v > max {
            return Err(invalid(format!(
                "{}: {field} = {v} is not a finite value in {min}..={max}",
                self.what
            )));
        }
        Ok(v)
    }

    /// `N` finite `f32`s, each in `min..=max`.
    pub(crate) fn f32s_in<const N: usize>(
        &mut self,
        field: &str,
        min: f32,
        max: f32,
    ) -> Result<[f32; N], CodecError> {
        let mut out = [0.0f32; N];
        for v in &mut out {
            *v = self.f32_in(field, min, max)?;
        }
        Ok(out)
    }

    /// A boolean byte: exactly 0 or 1.
    pub(crate) fn bool(&mut self, field: &str) -> Result<bool, CodecError> {
        match self.u8()? {
            0 => Ok(false),
            1 => Ok(true),
            b => Err(invalid(format!(
                "{}: {field} must be 0 or 1, got {b}",
                self.what
            ))),
        }
    }

    /// A count byte or word that must not exceed `cap`.
    pub(crate) fn count_u8(&mut self, field: &str, cap: usize) -> Result<usize, CodecError> {
        let n = usize::from(self.u8()?);
        if n > cap {
            return Err(invalid(format!(
                "{}: {field} count {n} exceeds the cap {cap}",
                self.what
            )));
        }
        Ok(n)
    }

    /// The version byte, which must be `expected`.
    pub(crate) fn version(&mut self, expected: u8) -> Result<(), CodecError> {
        let v = self.u8()?;
        if v != expected {
            return Err(invalid(format!(
                "{}: wire version {v} is not the supported {expected}",
                self.what
            )));
        }
        Ok(())
    }

    /// The name this reader reports in its errors.
    pub(crate) fn what(&self) -> &'static str {
        self.what
    }

    /// Every byte was consumed.
    pub(crate) fn finish(self) -> Result<(), CodecError> {
        if self.pos != self.buf.len() {
            return Err(invalid(format!(
                "{}: {} trailing bytes",
                self.what,
                self.buf.len() - self.pos
            )));
        }
        Ok(())
    }
}

/// Encode a unit quaternion check: `|q|²` within the same band
/// [`crate::wire`] accepts for a [`crate::ShipState`].
pub(crate) fn check_quat(what: &str, q: [f32; 4]) -> Result<(), CodecError> {
    let n2 = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
    if !(crate::wire::QUAT_NORM2_MIN..=crate::wire::QUAT_NORM2_MAX).contains(&n2) {
        return Err(invalid(format!(
            "{what}: quaternion {q:?} is not unit (|q|² = {n2})"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{Reader, Writer};

    #[test]
    fn round_trip_and_strictness() {
        let mut w = Writer::new();
        w.u8(1);
        w.u16(0xBEEF);
        w.u32(7);
        w.u64(u64::MAX);
        w.i64(-5);
        w.f32(1.5);
        let bytes = w.into_bytes();
        let mut r = Reader::new(&bytes, "test");
        r.version(1).unwrap();
        assert_eq!(r.u16().unwrap(), 0xBEEF);
        assert_eq!(r.u32().unwrap(), 7);
        assert_eq!(r.u64().unwrap(), u64::MAX);
        assert_eq!(r.i64().unwrap(), -5);
        assert_eq!(r.f32_in("x", 0.0, 2.0).unwrap(), 1.5);
        r.finish().unwrap();

        // Truncation, trailing bytes, out-of-range and non-finite floats.
        assert!(Reader::new(&bytes[..3], "t").u32().is_err());
        let mut r = Reader::new(&bytes, "t");
        r.u8().unwrap();
        assert!(r.finish().is_err());
        let nan = f32::NAN.to_bits().to_be_bytes();
        assert!(Reader::new(&nan, "t").f32_in("x", -1.0, 1.0).is_err());
        let big = 3.0f32.to_bits().to_be_bytes();
        assert!(Reader::new(&big, "t").f32_in("x", -1.0, 1.0).is_err());
        assert!(Reader::new(&[2], "t").bool("b").is_err());
        assert!(Reader::new(&[9], "t").count_u8("n", 8).is_err());
        assert!(Reader::new(&[2], "t").version(1).is_err());
    }
}
