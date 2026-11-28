//! Cluster certificate trust: who may be dialled.
//!
//! The model:
//!
//! * Every holder of the **cluster secret** derives the same CA key from it
//!   (PBKDF2-HMAC-SHA256 into an Ed25519 seed, see [`ClusterTrust::from_secret`]).
//! * Each process signs its **own leaf** certificate with that CA
//!   ([`ClusterTrust::issue`]), with a fresh key per call; the CA key never
//!   leaves the process and no certificate is shared.
//! * A dialer verifies the server with the standard WebPKI verifier against a
//!   trust store that holds only that CA ([`ClusterPin::client_config`]).
//! * The CA's public key, hex-encoded, is the public **cluster pin**
//!   ([`ClusterPin`]): it is safe to publish, and a dialer that holds it
//!   rejects any certificate not issued by that CA. Shards verify the
//!   orchestrator and each other this way.
//! * A server also presents the CA's self-signed certificate behind its leaf
//!   ([`ClusterTrust::ca_certificate`]), so a player that was never given the
//!   pin can **learn** it on first use and hold the server to it afterwards
//!   ([`crate::tofu`], [`crate::client_endpoint_tofu`]).
//!
//! Endpoints without a secret or pin keep using the self-signed,
//! accept-anything pair of [`crate::crypto`].

use std::fmt;
use std::str::FromStr;
use std::sync::Arc;

use rcgen::{
    BasicConstraints, CertificateParams, DistinguishedName, DnType, ExtendedKeyUsagePurpose, IsCa,
    Issuer, KeyUsagePurpose, PKCS_ED25519,
};
use ring::signature::{Ed25519KeyPair, KeyPair as _};
use rustls::pki_types::{CertificateDer, Der, PrivatePkcs8KeyDer, TrustAnchor};
use rustls::RootCertStore;

use crate::crypto::{quic_server_config, CertPair, VERSIONS};
use crate::ALPN;

/// The common name of the cluster CA.
pub const CLUSTER_CA_NAME: &str = "space cluster CA";

/// PBKDF2 iterations used to derive the CA seed from the cluster secret.
pub const KDF_ITERATIONS: u32 = 100_000;

/// PBKDF2 salt. Part of the derivation: changing it changes every pin.
const KDF_SALT: &[u8] = b"space-cluster-ca-v1";

/// The fixed prefix of a PKCS#8 v1 Ed25519 private key; the 32-byte seed follows.
const PKCS8_ED25519_PREFIX: [u8; 16] = [
    0x30, 0x2e, 0x02, 0x01, 0x00, 0x30, 0x05, 0x06, 0x03, 0x2b, 0x65, 0x70, 0x04, 0x22, 0x04, 0x20,
];

/// The cluster's public identity: the raw Ed25519 public key of its CA.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct ClusterPin([u8; 32]);

impl ClusterPin {
    /// A pin from the 32 bytes of the CA's public key.
    #[must_use]
    pub const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// The 32 bytes of the CA's public key.
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// The trust anchor of the cluster CA, built by hand from the CA's name and
    /// public key (no certificate is needed to dial).
    pub(crate) fn trust_anchor(&self) -> TrustAnchor<'static> {
        // The Name's value (a SET holding the common name), no outer SEQUENCE.
        let mut subject = vec![
            0x31, 0x19, 0x30, 0x17, 0x06, 0x03, 0x55, 0x04, 0x03, 0x0c, 0x10,
        ];
        subject.extend_from_slice(CLUSTER_CA_NAME.as_bytes());
        // The SubjectPublicKeyInfo's value (Ed25519 algorithm + bit string).
        let mut spki = vec![0x30, 0x05, 0x06, 0x03, 0x2b, 0x65, 0x70, 0x03, 0x21, 0x00];
        spki.extend_from_slice(&self.0);
        TrustAnchor {
            subject: Der::from(subject),
            subject_public_key_info: Der::from(spki),
            name_constraints: None,
        }
    }

    /// The rustls client config that accepts only servers whose certificate
    /// this cluster's CA issued: TLS 1.3, the WebPKI verifier, no client
    /// certificate, ALPN [`ALPN`].
    ///
    /// # Panics
    ///
    /// Panics if the ring provider cannot be installed — it always can, since
    /// it is compiled in.
    #[must_use]
    pub fn tls_client_config(&self) -> Arc<rustls::ClientConfig> {
        let roots = RootCertStore {
            roots: vec![self.trust_anchor()],
        };
        let provider = rustls::crypto::ring::default_provider();
        let mut config = rustls::ClientConfig::builder_with_provider(Arc::new(provider))
            .with_protocol_versions(VERSIONS)
            .expect("ring supports TLS 1.3")
            .with_root_certificates(roots)
            .with_no_client_auth();
        config.alpn_protocols = vec![ALPN.to_vec()];
        Arc::new(config)
    }

    /// [`ClusterPin::tls_client_config`] as a [`quinn::ClientConfig`].
    ///
    /// # Panics
    ///
    /// Panics if the rustls config cannot be used as a QUIC client config — it
    /// always can with the ring provider.
    #[must_use]
    pub fn client_config(&self) -> quinn::ClientConfig {
        let quic = quinn::crypto::rustls::QuicClientConfig::try_from(self.tls_client_config())
            .expect("the ring provider supplies the QUIC initial cipher suite");
        quinn::ClientConfig::new(Arc::new(quic))
    }
}

/// Lowercase hex of `bytes`.
fn hex(bytes: &[u8]) -> String {
    use fmt::Write as _;
    bytes
        .iter()
        .fold(String::with_capacity(bytes.len() * 2), |mut s, b| {
            let _ = write!(s, "{b:02x}");
            s
        })
}

impl fmt::Display for ClusterPin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&hex(&self.0))
    }
}

impl fmt::Debug for ClusterPin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ClusterPin({})", hex(&self.0))
    }
}

impl FromStr for ClusterPin {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, String> {
        let s = s.trim();
        let bad = || {
            format!(
                "a cluster pin is exactly 64 hexadecimal characters (the CA's public key), \
                 got {} characters",
                s.chars().count()
            )
        };
        if s.len() != 64 || !s.is_ascii() {
            return Err(bad());
        }
        let mut out = [0u8; 32];
        for (byte, pair) in out.iter_mut().zip(s.as_bytes().chunks(2)) {
            let digit = |c: u8| (c as char).to_digit(16).ok_or_else(bad);
            *byte = u8::try_from(digit(pair[0])? * 16 + digit(pair[1])?).map_err(|_| bad())?;
        }
        Ok(Self(out))
    }
}

/// What a holder of the cluster secret has: the CA key, to issue leaves with.
#[derive(Clone)]
pub struct ClusterTrust {
    ca_key_pkcs8: Vec<u8>,
    pin: ClusterPin,
    /// The CA's self-signed certificate, sent behind every leaf so that a
    /// trust-on-first-use dialer can read the pin ([`crate::tofu`]).
    ca_cert: CertificateDer<'static>,
}

impl ClusterTrust {
    /// Derive the cluster CA from `secret`; `None` for an empty secret.
    ///
    /// # Panics
    ///
    /// Panics if the derived key is not accepted by the compiled-in crypto —
    /// it always is, any 32 bytes are a valid Ed25519 seed.
    #[must_use]
    pub fn from_secret(secret: &str) -> Option<Self> {
        if secret.is_empty() {
            return None;
        }
        let mut seed = [0u8; 32];
        ring::pbkdf2::derive(
            ring::pbkdf2::PBKDF2_HMAC_SHA256,
            std::num::NonZeroU32::new(KDF_ITERATIONS).expect("non-zero"),
            KDF_SALT,
            secret.as_bytes(),
            &mut seed,
        );
        let pair = Ed25519KeyPair::from_seed_unchecked(&seed).expect("any 32 bytes are a seed");
        let mut pin = [0u8; 32];
        pin.copy_from_slice(pair.public_key().as_ref());
        let mut ca_key_pkcs8 = PKCS8_ED25519_PREFIX.to_vec();
        ca_key_pkcs8.extend_from_slice(&seed);
        let mut trust = Self {
            ca_key_pkcs8,
            pin: ClusterPin(pin),
            ca_cert: CertificateDer::from(Vec::new()),
        };
        let ca_cert = Self::ca_params()
            .self_signed(&trust.ca_key())
            .expect("rcgen: self-signing the CA failed");
        trust.ca_cert = CertificateDer::from(ca_cert.der().to_vec());
        Some(trust)
    }

    /// The CA's self-signed certificate (the same bytes on every call). A server
    /// presents it behind its leaf, as the chain's intermediate.
    #[must_use]
    pub fn ca_certificate(&self) -> &CertificateDer<'static> {
        &self.ca_cert
    }

    /// The cluster's public pin.
    #[must_use]
    pub const fn pin(&self) -> ClusterPin {
        self.pin
    }

    /// The CA's certificate parameters (also its issuer description).
    fn ca_params() -> CertificateParams {
        let mut params = CertificateParams::default();
        let mut name = DistinguishedName::new();
        name.push(DnType::CommonName, CLUSTER_CA_NAME);
        params.distinguished_name = name;
        params.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
        params.key_usages = vec![KeyUsagePurpose::KeyCertSign];
        params
    }

    /// The CA key as an rcgen key.
    fn ca_key(&self) -> rcgen::KeyPair {
        rcgen::KeyPair::from_pkcs8_der_and_sign_algo(
            &PrivatePkcs8KeyDer::from(self.ca_key_pkcs8.clone()),
            &PKCS_ED25519,
        )
        .expect("rcgen accepts a PKCS#8 Ed25519 key")
    }

    /// A fresh leaf certificate (new key each call) signed by the cluster CA.
    ///
    /// # Panics
    ///
    /// Panics if the certificate cannot be generated.
    #[must_use]
    pub fn issue(&self) -> CertPair {
        let ca_key = self.ca_key();
        let issuer = Issuer::new(Self::ca_params(), ca_key);
        let leaf_key = rcgen::KeyPair::generate().expect("rcgen: key generation failed");
        let mut params = CertificateParams::new(vec!["space.local".to_owned()])
            .expect("a fixed, valid subject alternative name");
        params.extended_key_usages = vec![ExtendedKeyUsagePurpose::ServerAuth];
        let cert = params
            .signed_by(&leaf_key, &issuer)
            .expect("rcgen: signing the leaf failed");
        CertPair::from_parts(
            CertificateDer::from(cert.der().to_vec()),
            leaf_key.serialize_der(),
            vec![self.ca_cert.clone()],
        )
    }

    /// A [`quinn::ServerConfig`] presenting a fresh leaf of this cluster.
    #[must_use]
    pub fn server_config(&self) -> quinn::ServerConfig {
        quic_server_config(&self.issue())
    }

    /// The matching client config: [`ClusterPin::client_config`] of [`Self::pin`].
    #[must_use]
    pub fn client_config(&self) -> quinn::ClientConfig {
        self.pin().client_config()
    }
}

impl fmt::Debug for ClusterTrust {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ClusterTrust {{ pin: {} }}", self.pin)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SECRET: &str = "known-answer-secret-0123456789";

    /// Same secret, same pin; another secret, another pin; no secret, no trust.
    #[test]
    fn pins_follow_the_secret() {
        let a = ClusterTrust::from_secret("alpha").unwrap();
        assert_eq!(a.pin(), ClusterTrust::from_secret("alpha").unwrap().pin());
        assert_ne!(a.pin(), ClusterTrust::from_secret("beta").unwrap().pin());
        assert!(ClusterTrust::from_secret("").is_none());
    }

    /// Known answer. If this changes, clusters of different builds can no
    /// longer talk to each other: the derivation is part of the wire contract.
    #[test]
    fn the_pin_of_a_known_secret_is_fixed() {
        let pin = ClusterTrust::from_secret(SECRET).unwrap().pin();
        assert_eq!(
            pin.to_string(),
            "6a8985a64573f6abc5cc8f56ed865a329e60474799c5e95b4010895e7fa9d5f7"
        );
    }

    /// Display and FromStr round-trip; wrong length and non-hex are refused.
    #[test]
    fn pin_text_round_trips() {
        let pin = ClusterTrust::from_secret("alpha").unwrap().pin();
        let text = pin.to_string();
        assert_eq!(text.len(), 64);
        assert_eq!(text.parse::<ClusterPin>().unwrap(), pin);
        assert_eq!(
            format!("  {}\n", text.to_uppercase()).parse::<ClusterPin>(),
            Ok(pin)
        );
        assert_eq!(format!("{pin:?}"), format!("ClusterPin({text})"));
        assert!(text[..62].parse::<ClusterPin>().is_err());
        assert!(format!("{text}00").parse::<ClusterPin>().is_err());
        assert!(format!("zz{}", &text[2..]).parse::<ClusterPin>().is_err());
        assert!("".parse::<ClusterPin>().unwrap_err().contains("64"));
    }

    /// The hand-built anchor is what rustls itself derives from a self-signed
    /// certificate of the same CA name and key.
    #[test]
    fn the_hand_built_anchor_matches_rustls() {
        let trust = ClusterTrust::from_secret(SECRET).unwrap();
        let key = trust.ca_key();
        let cert = ClusterTrust::ca_params().self_signed(&key).unwrap();
        let mut store = RootCertStore::empty();
        store.add(cert.der().clone()).unwrap();
        let theirs = &store.roots[0];
        let ours = trust.pin().trust_anchor();
        assert_eq!(&*ours.subject, &*theirs.subject);
        assert_eq!(
            &*ours.subject_public_key_info,
            &*theirs.subject_public_key_info
        );
    }

    /// Debug never prints the secret or the key seed.
    #[test]
    fn debug_leaves_the_secret_and_key_out() {
        let trust = ClusterTrust::from_secret(SECRET).unwrap();
        let shown = format!("{trust:?}");
        assert!(!shown.contains(SECRET));
        assert!(!shown.contains(&hex(&trust.ca_key_pkcs8[16..])));
        assert!(shown.contains(&trust.pin().to_string()));
    }
}
