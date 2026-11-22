//! TLS material: a self-signed certificate and the rustls configs.
//!
//! `space` runs its own trust domain: shards and clients have no CA, so every
//! endpoint presents a **self-signed** certificate generated at start-up with
//! [`rcgen`], and the client accepts the peer's certificate without a CA chain
//! ([`SkipAllVerifier`]). That is deliberate for Phase 2: QUIC still gets an
//! authenticated, encrypted, integrity-protected channel, and the identity check
//! the game actually needs (the player token in [`space_protocol::Hello`])
//! happens one layer up, in [`crate::accept_connection`]. Pinning real
//! certificates is an orchestrator (Phase 3) concern.
//!
//! What "pure Rust" means here, precisely: the whole `space` dependency graph is
//! pure Rust **except for one deliberate exception** — the TLS/QUIC cryptographic
//! backend. The rustls provider in use is
//! [`rustls::crypto::ring`](rustls::crypto::ring), and building it compiles
//! *ring*'s hand-written assembly and C sources at build time. That assembly is
//! used only for the TLS/QUIC crypto (rustls' TLS 1.3 cipher suites and key
//! exchange, plus quinn-proto's QUIC key derivation — HMAC/HKDF/AEAD for the
//! handshake-token and Retry machinery); nothing else in the binary touches C.
//! Everything else in `space` — the wire format, ECS, simulation, rendering — is
//! pure Rust with no C bindings.
//!
//! This exception is intentional and unavoidable for v1: quinn 0.11 hard-gates
//! its rustls integration behind the `rustls-ring` (or `rustls-aws-lc-rs`)
//! feature, and its `configured_provider()` can only ever return the `ring` or
//! `aws-lc-rs` provider — there is no seam to plug in a third one. The two
//! candidate "pure-Rust" providers do not close the gap: `rustls-post-quantum`
//! is built on `aws-lc-rs` (which compiles C/asm, so it is not pure Rust), and
//! `rustls-rustcrypto`'s QUIC support is unimplemented (`todo!()` for header
//! protection), so it cannot do QUIC packet protection at all. Switching away
//! from `ring` would require forking quinn.
//!
//! Decision: keep the `ring` provider for v1. Revisit if quinn gains a
//! pluggable-provider seam and a pure-Rust provider grows working QUIC support.

use std::sync::Arc;

use rcgen::{generate_simple_self_signed, CertifiedKey};
use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier};
use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer, ServerName, UnixTime};
use rustls::{ClientConfig, DigitallySignedStruct, SupportedProtocolVersion};

use crate::ALPN;

/// A self-signed certificate plus the key it was signed with.
///
/// The key is kept as raw PKCS#8 bytes because [`PrivateKeyDer`] is not
/// [`Clone`]; [`CertPair::key_der`] hands out an owned copy on demand.
#[derive(Clone)]
pub struct CertPair {
    /// The DER-encoded certificate (presented on connect).
    pub cert: CertificateDer<'static>,
    key_pkcs8: Vec<u8>,
}

impl CertPair {
    /// Generate a fresh self-signed certificate.
    ///
    /// `dns_names` become the subject alternative names; [`SkipAllVerifier`]
    /// ignores them, but a real deployment would pin them.
    #[must_use]
    pub fn generate(dns_names: &[&str]) -> Self {
        let names: Vec<String> = dns_names.iter().map(|s| (*s).to_string()).collect();
        let CertifiedKey { cert, signing_key } =
            generate_simple_self_signed(names).expect("rcgen: self-signed cert generation failed");
        Self {
            cert: cert.der().clone(),
            key_pkcs8: signing_key.serialize_der(),
        }
    }

    /// The private key, as a fresh [`PrivateKeyDer`].
    #[must_use]
    pub fn key_der(&self) -> PrivateKeyDer<'static> {
        PrivatePkcs8KeyDer::from(self.key_pkcs8.clone()).into()
    }
}

impl std::fmt::Debug for CertPair {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CertPair")
            .field("cert_len", &self.cert.len())
            .finish_non_exhaustive()
    }
}

/// A [`ServerCertVerifier`](rustls::client::danger::ServerCertVerifier) that
/// accepts any certificate (see the [module docs](self) for why).
#[derive(Debug)]
pub struct SkipAllVerifier;

impl ServerCertVerifier for SkipAllVerifier {
    fn verify_server_cert(
        &self,
        _end_entity: &CertificateDer<'_>,
        _intermediates: &[CertificateDer<'_>],
        _server_name: &ServerName<'_>,
        _ocsp_response: &[u8],
        _now: UnixTime,
    ) -> Result<ServerCertVerified, rustls::Error> {
        Ok(ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        _message: &[u8],
        _cert: &CertificateDer<'_>,
        _dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, rustls::Error> {
        Ok(HandshakeSignatureValid::assertion())
    }

    fn verify_tls13_signature(
        &self,
        _message: &[u8],
        _cert: &CertificateDer<'_>,
        _dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, rustls::Error> {
        Ok(HandshakeSignatureValid::assertion())
    }

    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        use rustls::SignatureScheme as S;
        vec![
            S::ECDSA_NISTP256_SHA256,
            S::ECDSA_NISTP384_SHA384,
            S::ED25519,
            S::RSA_PSS_SHA256,
            S::RSA_PKCS1_SHA256,
        ]
    }
}

/// The TLS versions a `space` endpoint speaks (QUIC requires TLS 1.3).
const VERSIONS: &[&SupportedProtocolVersion] = &[&rustls::version::TLS13];

/// Build the rustls **server** config from `cert`.
///
/// Client certificates are not requested (the player token authenticates), and
/// the ALPN is [`ALPN`].
///
/// # Panics
///
/// Panics if the key is not supported by the compiled-in provider — the rcgen
/// key always is.
#[must_use]
pub fn make_tls_server_config(cert: &CertPair) -> Arc<rustls::ServerConfig> {
    let provider = rustls::crypto::ring::default_provider();
    let mut config = rustls::ServerConfig::builder_with_provider(Arc::new(provider))
        .with_protocol_versions(VERSIONS)
        .expect("ring supports TLS 1.3")
        .with_no_client_auth()
        .with_single_cert(vec![cert.cert.clone()], cert.key_der())
        .expect("the rcgen key must be supported by the ring provider");
    config.alpn_protocols = vec![ALPN.to_vec()];
    Arc::new(config)
}

/// Build the rustls **client** config (any server certificate accepted, see
/// [`SkipAllVerifier`]).
///
/// # Panics
///
/// Panics if the ring provider cannot be installed — it always can, since it is
/// compiled in.
#[must_use]
pub fn make_tls_client_config() -> Arc<ClientConfig> {
    let provider = rustls::crypto::ring::default_provider();
    let mut config = ClientConfig::builder_with_provider(Arc::new(provider))
        .with_protocol_versions(VERSIONS)
        .expect("ring supports TLS 1.3")
        .dangerous()
        .with_custom_certificate_verifier(Arc::new(SkipAllVerifier))
        .with_no_client_auth();
    config.alpn_protocols = vec![ALPN.to_vec()];
    Arc::new(config)
}

/// Build a [`quinn::ServerConfig`] from a fresh self-signed certificate.
///
/// Returns the config and the certificate it presents (handy for tests and for
/// logging the fingerprint). The ALPN from [`make_tls_server_config`] survives,
/// because the quinn config wraps that rustls config rather than rebuilding one.
///
/// # Panics
///
/// Panics if the certificate cannot be generated (see [`CertPair::generate`]).
#[must_use]
pub fn make_server_config() -> (quinn::ServerConfig, CertPair) {
    let cert = CertPair::generate(&["space.local"]);
    let quic = quinn::crypto::rustls::QuicServerConfig::try_from(make_tls_server_config(&cert))
        .expect("the ring provider supplies the QUIC initial cipher suite");
    (quinn::ServerConfig::with_crypto(Arc::new(quic)), cert)
}

/// Build a [`quinn::ClientConfig`] that accepts any server certificate.
///
/// # Panics
///
/// Panics if the rustls config cannot be used as a QUIC client config — it
/// always can with the ring provider (the initial cipher suite is present).
#[must_use]
pub fn make_client_config() -> quinn::ClientConfig {
    let quic = quinn::crypto::rustls::QuicClientConfig::try_from(make_tls_client_config())
        .expect("the ring provider supplies the QUIC initial cipher suite");
    quinn::ClientConfig::new(Arc::new(quic))
}
