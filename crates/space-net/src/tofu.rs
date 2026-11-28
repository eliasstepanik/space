//! Trust on first use: a client that does not (yet) know a cluster's pin.
//!
//! A cluster server presents its leaf **and** the cluster CA's self-signed
//! certificate behind it ([`crate::trust::ClusterTrust::ca_certificate`]). This
//! module's verifier reads the CA's public key off that certificate — the
//! [`ClusterPin`] — and:
//!
//! * with **no expected pin** (first use): checks that the leaf really was
//!   issued by the CA it presented (the WebPKI verifier against a trust store
//!   holding just that CA), accepts, and remembers the presented pin in a
//!   [`TofuProbe`] for the caller to store. A server that presents no CA (no
//!   cluster secret: a self-signed leaf) is accepted **unpinned**, as an
//!   accept-anything dial always was, and the probe reports no pin;
//! * with an **expected pin**: accepts only a chain whose CA is that pin and
//!   whose leaf that CA issued. Anything else — another CA, a self-signed
//!   server — fails the handshake as [`crate::NetError::Untrusted`], and the
//!   probe still says what was presented, so the caller can name both pins.
//!
//! Where pins are kept, and what a user does about a mismatch, is the
//! caller's business (the client's `tofu` module).

use std::sync::{Arc, Mutex, PoisonError};

use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier};
use rustls::client::WebPkiServerVerifier;
use rustls::crypto::CryptoProvider;
use rustls::pki_types::{CertificateDer, ServerName, UnixTime};
use rustls::{
    CertificateError, DigitallySignedStruct, Error as TlsError, RootCertStore, SignatureScheme,
};

use crate::crypto::VERSIONS;
use crate::trust::ClusterPin;
use crate::ALPN;

/// What the verifier of one dial saw. Cheap to clone; all clones share it.
#[derive(Clone, Debug, Default)]
pub struct TofuProbe(Arc<Mutex<Option<Option<ClusterPin>>>>);

impl TofuProbe {
    /// The pin of the CA the server presented, or `None` when the handshake
    /// never reached the certificate or the server presented no CA. After a
    /// **successful** dial without an expected pin this is the pin to
    /// remember; after a failed one with an expected pin it is the pin that
    /// did not match (or `None`: the server presented none).
    #[must_use]
    pub fn presented_pin(&self) -> Option<ClusterPin> {
        (*self.0.lock().unwrap_or_else(PoisonError::into_inner)).flatten()
    }

    fn record(&self, pin: Option<ClusterPin>) {
        *self.0.lock().unwrap_or_else(PoisonError::into_inner) = Some(pin);
    }
}

/// The 10 bytes in front of an Ed25519 public key in a SubjectPublicKeyInfo's
/// value: the algorithm (`1.3.101.112`) and the bit string's header.
const ED25519_SPKI_PREFIX: [u8; 10] = [0x30, 0x05, 0x06, 0x03, 0x2b, 0x65, 0x70, 0x03, 0x21, 0x00];

/// Split one DER element off the front of `input`: its tag, its content and
/// what follows it.
fn tlv(input: &[u8]) -> Option<(u8, &[u8], &[u8])> {
    let (&tag, rest) = input.split_first()?;
    let (&first, rest) = rest.split_first()?;
    let (len, rest) = if first < 0x80 {
        (usize::from(first), rest)
    } else {
        let n = usize::from(first & 0x7f);
        if n == 0 || n > 4 || rest.len() < n {
            return None;
        }
        let len = rest[..n]
            .iter()
            .fold(0usize, |acc, b| (acc << 8) | usize::from(*b));
        (len, &rest[n..])
    };
    (rest.len() >= len).then(|| (tag, &rest[..len], &rest[len..]))
}

/// The Ed25519 public key a certificate certifies, as a pin; `None` when the
/// bytes are not a certificate with such a key.
#[must_use]
fn certified_ed25519_key(der: &[u8]) -> Option<ClusterPin> {
    let (0x30, certificate, _) = tlv(der)? else {
        return None;
    };
    let (0x30, tbs, _) = tlv(certificate)? else {
        return None;
    };
    let (tag, _, after_first) = tlv(tbs)?;
    let mut rest = if tag == 0xa0 { after_first } else { tbs };
    // Serial, signature algorithm, issuer, validity and subject.
    for _ in 0..5 {
        rest = tlv(rest)?.2;
    }
    let (0x30, spki, _) = tlv(rest)? else {
        return None;
    };
    let key = spki.strip_prefix(&ED25519_SPKI_PREFIX)?;
    Some(ClusterPin::from_bytes(key.try_into().ok()?))
}

/// The pin of the CA a chain presents: the first certificate behind the leaf
/// that certifies an Ed25519 key.
fn presented_ca(intermediates: &[CertificateDer<'_>]) -> Option<ClusterPin> {
    intermediates
        .iter()
        .find_map(|cert| certified_ed25519_key(cert))
}

#[derive(Debug)]
struct TofuVerifier {
    expected: Option<ClusterPin>,
    probe: TofuProbe,
    provider: Arc<CryptoProvider>,
}

impl TofuVerifier {
    /// The WebPKI verification of the chain against a store of `pin`'s CA only.
    fn verify_against(
        &self,
        pin: &ClusterPin,
        end_entity: &CertificateDer<'_>,
        intermediates: &[CertificateDer<'_>],
        server_name: &ServerName<'_>,
        now: UnixTime,
    ) -> Result<ServerCertVerified, TlsError> {
        let roots = RootCertStore {
            roots: vec![pin.trust_anchor()],
        };
        let verifier =
            WebPkiServerVerifier::builder_with_provider(Arc::new(roots), self.provider.clone())
                .build()
                .map_err(|e| TlsError::General(e.to_string()))?;
        verifier.verify_server_cert(end_entity, intermediates, server_name, &[], now)
    }
}

impl ServerCertVerifier for TofuVerifier {
    fn verify_server_cert(
        &self,
        end_entity: &CertificateDer<'_>,
        intermediates: &[CertificateDer<'_>],
        server_name: &ServerName<'_>,
        _ocsp_response: &[u8],
        now: UnixTime,
    ) -> Result<ServerCertVerified, TlsError> {
        let presented = presented_ca(intermediates);
        self.probe.record(presented);
        let unknown_issuer = || TlsError::InvalidCertificate(CertificateError::UnknownIssuer);
        match (self.expected, presented) {
            (Some(expected), Some(seen)) if expected == seen => {
                self.verify_against(&expected, end_entity, intermediates, server_name, now)
            }
            (Some(_), _) => Err(unknown_issuer()),
            (None, Some(seen)) => {
                self.verify_against(&seen, end_entity, intermediates, server_name, now)
            }
            // No CA: a server without a cluster secret. Nothing to learn,
            // nothing to check — the accept-anything mode.
            (None, None) => Ok(ServerCertVerified::assertion()),
        }
    }

    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, TlsError> {
        rustls::crypto::verify_tls12_signature(
            message,
            cert,
            dss,
            &self.provider.signature_verification_algorithms,
        )
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, TlsError> {
        rustls::crypto::verify_tls13_signature(
            message,
            cert,
            dss,
            &self.provider.signature_verification_algorithms,
        )
    }

    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        self.provider
            .signature_verification_algorithms
            .supported_schemes()
    }
}

/// The client config of a trust-on-first-use dial and the probe that reports
/// what the server presented (see the [module docs](self)).
///
/// # Panics
///
/// Panics if the ring provider cannot be installed or used for QUIC — it
/// always can, since it is compiled in.
#[must_use]
pub(crate) fn client_config(expected: Option<ClusterPin>) -> (quinn::ClientConfig, TofuProbe) {
    let probe = TofuProbe::default();
    let provider = Arc::new(rustls::crypto::ring::default_provider());
    let verifier = TofuVerifier {
        expected,
        probe: probe.clone(),
        provider: Arc::clone(&provider),
    };
    let mut config = rustls::ClientConfig::builder_with_provider(provider)
        .with_protocol_versions(VERSIONS)
        .expect("ring supports TLS 1.3")
        .dangerous()
        .with_custom_certificate_verifier(Arc::new(verifier))
        .with_no_client_auth();
    config.alpn_protocols = vec![ALPN.to_vec()];
    let quic = quinn::crypto::rustls::QuicClientConfig::try_from(Arc::new(config))
        .expect("the ring provider supplies the QUIC initial cipher suite");
    (quinn::ClientConfig::new(Arc::new(quic)), probe)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trust::ClusterTrust;

    #[test]
    fn the_pin_is_read_off_the_ca_certificate() {
        let trust = ClusterTrust::from_secret("tofu-unit").unwrap();
        assert_eq!(
            certified_ed25519_key(trust.ca_certificate()),
            Some(trust.pin())
        );
    }

    /// The leaf certifies a key of its own, and bytes that are no certificate
    /// certify nothing.
    #[test]
    fn a_leaf_or_garbage_gives_no_other_pin() {
        let trust = ClusterTrust::from_secret("tofu-unit").unwrap();
        let leaf = trust.issue();
        assert_ne!(certified_ed25519_key(&leaf.cert), Some(trust.pin()));
        assert_eq!(certified_ed25519_key(&[]), None);
        assert_eq!(certified_ed25519_key(&[0x30, 0x03, 1, 2, 3]), None);
    }

    #[test]
    fn the_ca_certificate_is_the_same_every_time() {
        let a = ClusterTrust::from_secret("tofu-unit").unwrap();
        let b = ClusterTrust::from_secret("tofu-unit").unwrap();
        assert_eq!(a.ca_certificate(), b.ca_certificate());
        assert_ne!(
            a.ca_certificate(),
            ClusterTrust::from_secret("other").unwrap().ca_certificate()
        );
    }
}
