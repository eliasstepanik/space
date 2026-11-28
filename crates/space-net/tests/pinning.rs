//! Cluster certificate trust over a real quinn endpoint pair on loopback.
//!
//! | test | checks |
//! |------|--------|
//! | [`a_pinned_client_connects_to_a_server_of_its_cluster`] | a pinned dial against a `ClusterTrust` leaf completes |
//! | [`a_pinned_client_refuses_a_self_signed_server`] | a self-signed server is `NetError::Untrusted` |
//! | [`a_pinned_client_refuses_another_clusters_server`] | a leaf of another secret's CA is `NetError::Untrusted` |
//! | [`a_pin_parsed_from_its_text_verifies`] | the Display string of a pin is enough to verify |
//! | [`an_unpinned_client_still_connects_to_a_cluster_server`] | accept-any dialers are unchanged |
//! | `tofu_*` | the trust-on-first-use dial learns the presented CA's pin, holds a server to an expected pin, accepts a CA-less server unpinned |

use std::net::{IpAddr, Ipv4Addr, SocketAddr};

use space_net::{
    accept_connection, client_endpoint_for, client_endpoint_pinned, client_endpoint_tofu, connect,
    hello, make_server_config, server_endpoint, CertPair, ClusterPin, ClusterTrust, NetError,
};
use space_protocol::{Message, ShardId, ShardMap, Welcome};

fn addr() -> SocketAddr {
    SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0)
}

fn trust(secret: &str) -> ClusterTrust {
    ClusterTrust::from_secret(secret).expect("a non-empty secret")
}

/// Serve one accept in the background, answering a hello with a welcome.
fn serve(config: quinn::ServerConfig) -> (SocketAddr, tokio::task::JoinHandle<()>) {
    let server = server_endpoint(addr(), config).expect("bind server");
    let bound = server.local_addr().expect("local addr");
    let task = tokio::spawn(async move {
        if let Ok((conn, _hello)) = accept_connection(server.accept()).await {
            let welcome = Welcome {
                session: 1,
                shard: ShardId::new(1),
                shard_map: ShardMap::new(1, Vec::new()),
            };
            let _ = conn.send_control(&Message::Welcome(welcome)).await;
            // Keep the endpoint alive until the client has read the welcome.
            let _ = conn.recv_control().await;
        }
    });
    (bound, task)
}

async fn dial(bound: SocketAddr, pin: Option<&ClusterPin>) -> Result<(), NetError> {
    let client = client_endpoint_pinned(bound, pin).expect("client endpoint");
    let (conn, _welcome) = connect(&client, bound, "space.local", &hello("pinning")).await?;
    conn.connection().close(0u32.into(), b"done");
    Ok(())
}

/// Catches: a client config that rejects the cluster's own leaves (wrong
/// anchor bytes, wrong server name, missing ServerAuth usage).
#[tokio::test]
async fn a_pinned_client_connects_to_a_server_of_its_cluster() {
    let t = trust("cluster-one");
    let (bound, _task) = serve(t.server_config());
    dial(bound, Some(&t.pin())).await.expect("connect");
}

/// Catches: a pinned client that still accepts any certificate (the verifier
/// left at `SkipAllVerifier`), or a failed verification mapped to `Handshake`.
#[tokio::test]
async fn a_pinned_client_refuses_a_self_signed_server() {
    let t = trust("cluster-one");
    let (config, _cert) = make_server_config();
    let (bound, _task) = serve(config);
    let err = dial(bound, Some(&t.pin())).await.expect_err("must refuse");
    assert!(matches!(err, NetError::Untrusted(_)), "got {err:?}");
}

/// Catches: trust that checks only that a chain exists, not that it ends in
/// this cluster's CA.
#[tokio::test]
async fn a_pinned_client_refuses_another_clusters_server() {
    let t = trust("cluster-one");
    let (bound, _task) = serve(trust("cluster-two").server_config());
    let err = dial(bound, Some(&t.pin())).await.expect_err("must refuse");
    assert!(matches!(err, NetError::Untrusted(_)), "got {err:?}");
}

/// Catches: a Display/FromStr pair that loses or reorders bytes of the pin.
#[tokio::test]
async fn a_pin_parsed_from_its_text_verifies() {
    let t = trust("cluster-one");
    let pin: ClusterPin = t.pin().to_string().parse().expect("parse");
    let (bound, _task) = serve(t.server_config());
    dial(bound, Some(&pin)).await.expect("connect");
}

/// Catches: `client_endpoint_pinned(.., None)` or the shared endpoint helper
/// changing what an unpinned dial accepts.
#[tokio::test]
async fn an_unpinned_client_still_connects_to_a_cluster_server() {
    let t = trust("cluster-one");
    let (bound, _task) = serve(t.server_config());
    let client = client_endpoint_for(bound).expect("client endpoint");
    let (conn, _welcome) = connect(&client, bound, "space.local", &hello("unpinned"))
        .await
        .expect("connect");
    conn.connection().close(0u32.into(), b"done");
}

/// Dial with the trust-on-first-use verifier; the result and what it saw.
async fn dial_tofu(
    bound: SocketAddr,
    expected: Option<ClusterPin>,
) -> (Result<(), NetError>, Option<ClusterPin>) {
    let (client, probe) = client_endpoint_tofu(bound, expected).expect("client endpoint");
    let result = connect(&client, bound, "space.local", &hello("tofu"))
        .await
        .map(|(conn, _)| conn.connection().close(0u32.into(), b"done"));
    (result, probe.presented_pin())
}

/// Catches: a first-use dial that cannot read the pin off the chain.
#[tokio::test]
async fn tofu_learns_the_pin_of_a_cluster_server() {
    let t = trust("cluster-one");
    let (bound, _task) = serve(t.server_config());
    let (result, seen) = dial_tofu(bound, None).await;
    result.expect("connect");
    assert_eq!(seen, Some(t.pin()));
}

/// Catches: an expected pin that is not enforced, or a mismatch that is not
/// `Untrusted` and does not say what was presented.
#[tokio::test]
async fn tofu_holds_a_server_to_the_expected_pin() {
    let t = trust("cluster-one");
    let (bound, _task) = serve(t.server_config());
    let (result, seen) = dial_tofu(bound, Some(t.pin())).await;
    result.expect("the right pin connects");
    assert_eq!(seen, Some(t.pin()));

    let other = trust("cluster-two");
    let (bound, _task) = serve(other.server_config());
    let (result, seen) = dial_tofu(bound, Some(t.pin())).await;
    assert!(matches!(result, Err(NetError::Untrusted(_))), "{result:?}");
    assert_eq!(seen, Some(other.pin()));
}

/// Catches: a self-signed server learned as a pin, or accepted against an
/// expected one.
#[tokio::test]
async fn tofu_takes_a_server_without_a_ca_unpinned_only_when_nothing_is_expected() {
    let (config, _cert) = make_server_config();
    let (bound, _task) = serve(config);
    let (result, seen) = dial_tofu(bound, None).await;
    result.expect("connect unpinned");
    assert_eq!(seen, None);

    let (config, _cert) = make_server_config();
    let (bound, _task) = serve(config);
    let (result, _) = dial_tofu(bound, Some(trust("cluster-one").pin())).await;
    assert!(matches!(result, Err(NetError::Untrusted(_))), "{result:?}");
}

/// A server that presents `chain` and signs with `leaf`'s key: whatever a
/// hostile server could assemble out of public parts.
fn forged_server(
    chain: Vec<rustls::pki_types::CertificateDer<'static>>,
    leaf: &CertPair,
) -> quinn::ServerConfig {
    let provider = rustls::crypto::ring::default_provider();
    let mut config = rustls::ServerConfig::builder_with_provider(std::sync::Arc::new(provider))
        .with_protocol_versions(&[&rustls::version::TLS13])
        .expect("TLS 1.3")
        .with_no_client_auth()
        .with_single_cert(chain, leaf.key_der())
        .expect("a usable key");
    config.alpn_protocols = vec![space_net::ALPN.to_vec()];
    let quic = quinn::crypto::rustls::QuicServerConfig::try_from(std::sync::Arc::new(config))
        .expect("QUIC server config");
    quinn::ServerConfig::with_crypto(std::sync::Arc::new(quic))
}

/// Catches: a first-use verifier that learns a pin from a CA certificate
/// without checking that the leaf was issued by it (`(None, Some(_)) =>
/// accept`): an impostor could copy a real cluster's public CA certificate,
/// put its own leaf in front of it and have the copied pin stored.
#[tokio::test]
async fn tofu_refuses_a_copied_ca_in_front_of_a_foreign_leaf() {
    let t1 = trust("cluster-one");
    let t2 = trust("cluster-two");
    let leaf2 = t2.issue();

    // Another CA's leaf behind the real CA's certificate.
    let chain = vec![leaf2.cert.clone(), t1.ca_certificate().clone()];
    let (bound, _task) = serve(forged_server(chain, &leaf2));
    let (result, _) = dial_tofu(bound, None).await;
    assert!(
        matches!(result, Err(NetError::Untrusted(_))),
        "first use: {result:?}"
    );
    let chain = vec![leaf2.cert.clone(), t1.ca_certificate().clone()];
    let (bound, _task) = serve(forged_server(chain, &leaf2));
    let (result, _) = dial_tofu(bound, Some(t1.pin())).await;
    assert!(
        matches!(result, Err(NetError::Untrusted(_))),
        "pinned: {result:?}"
    );

    // A self-signed leaf behind the real CA's certificate.
    let (_, own) = make_server_config();
    let chain = vec![own.cert.clone(), t1.ca_certificate().clone()];
    let (bound, _task) = serve(forged_server(chain, &own));
    let (result, _) = dial_tofu(bound, None).await;
    assert!(
        matches!(result, Err(NetError::Untrusted(_))),
        "self-signed: {result:?}"
    );
}
