//! Trust on first use, end to end over real QUIC on loopback: the client's
//! dial path (`space_client::tofu`, which `link.rs` dials through) against
//! in-process shards that do and do not hold a cluster secret.
//!
//! The pin store is a file the tests point at with `SPACE_PIN_FILE` — one
//! temporary file per test — and `--pin` is a process-wide setting, so the
//! tests run one at a time (`gate`).
//!
//! | test | checks |
//! |------|--------|
//! | [`the_first_connection_records_the_pin_and_succeeds`] | no entry: accepted, `host:port` -> the cluster's pin is written |
//! | [`the_second_connection_uses_the_recorded_pin`] | the entry is kept and the next dial succeeds |
//! | [`a_server_of_another_ca_is_refused_and_the_store_is_unchanged`] | mismatch: `Untrusted` naming host:port, both pins and `--reset-pins`; file bytes unchanged |
//! | [`a_pin_given_on_the_command_line_skips_the_store`] | `--pin` right: connects, nothing stored; `--pin` wrong: refused |
//! | [`a_server_without_a_cluster_secret_connects_unpinned_and_stores_nothing`] | no CA, no pin: connects, no file |
//! | [`a_stored_pin_refuses_a_server_that_presents_no_ca`] | no downgrade to unpinned |
//! | [`a_corrupt_store_is_treated_as_empty`] | connects by TOFU and writes a fresh file |
//! | [`a_copied_ca_in_front_of_a_foreign_leaf_is_refused_and_nothing_is_stored`] | the leaf must be the presented CA's; the message says so for a pinned CA |
//! | [`the_player_link_dials_through_the_store`] | `ServerLink::join` records the pin too |
//! | [`reset_pins_deletes_the_file_and_exits_zero`] | the binary's `--reset-pins`; `--help` lists both flags |

use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use space_client::link::{ServerLink, SERVER_NAME};
use space_client::tofu::{self, PinStore, PIN_FILE_ENV};
use space_net::{hello, CertPair, ClusterPin, ClusterTrust, NetError};
use space_protocol::ShardId;
use space_server::net::ShardNet;
use space_server::{Shard, ShardConfig};
use space_world::NodeRef;
use tokio::sync::{Mutex as AsyncMutex, MutexGuard};
use tokio::task::JoinHandle;

static GATE: AsyncMutex<()> = AsyncMutex::const_new(());

/// One test's world: the gate, its own pin file, `--pin` unset.
struct Fixture {
    _gate: MutexGuard<'static, ()>,
    dir: PathBuf,
}

impl Fixture {
    async fn new(name: &str) -> Self {
        let gate = GATE.lock().await;
        let dir = std::env::temp_dir().join(format!("space-tofu-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        // The store's directory is created on the first write, as for a user.
        std::env::set_var(PIN_FILE_ENV, dir.join("config").join("pins.json"));
        tofu::set_pin(None);
        Self { _gate: gate, dir }
    }

    fn file(&self) -> PathBuf {
        self.dir.join("config").join("pins.json")
    }

    fn store(&self) -> PinStore {
        PinStore::load(&self.file())
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        tofu::set_pin(None);
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

struct Server {
    net: Arc<ShardNet>,
    task: JoinHandle<()>,
}

impl Server {
    fn addr(&self) -> SocketAddr {
        self.net.local_addr().expect("bound address")
    }

    /// The key the client stores this server's pin under.
    fn key(&self) -> String {
        self.addr().to_string()
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        self.task.abort();
        self.net.close();
    }
}

/// A shard on loopback; with `secret` it is a cluster's, presenting its CA.
fn start(secret: Option<&str>) -> Server {
    let trust = secret.map(|s| ClusterTrust::from_secret(s).expect("a non-empty secret"));
    let net = Arc::new(
        ShardNet::bind_trusted(SocketAddr::from(([127, 0, 0, 1], 0)), trust).expect("bind"),
    );
    let config = ShardConfig::new(
        ShardId::new(1),
        net.local_addr().expect("addr"),
        NodeRef::new(64, [1, 1, 1]),
    );
    let shard = Arc::new(AsyncMutex::new(Shard::new(config)));
    let task = tokio::spawn({
        let net = Arc::clone(&net);
        async move {
            net.serve(shard).await;
        }
    });
    Server { net, task }
}

/// A hostile server: `chain` presented, signed with `leaf`'s key. It only
/// accepts connections; the handshake is what the client refuses. Keep the
/// endpoint alive for the dial.
fn forged(
    chain: Vec<rustls::pki_types::CertificateDer<'static>>,
    leaf: &CertPair,
) -> (quinn::Endpoint, SocketAddr) {
    let provider = rustls::crypto::ring::default_provider();
    let mut config = rustls::ServerConfig::builder_with_provider(Arc::new(provider))
        .with_protocol_versions(&[&rustls::version::TLS13])
        .expect("TLS 1.3")
        .with_no_client_auth()
        .with_single_cert(chain, leaf.key_der())
        .expect("a usable key");
    config.alpn_protocols = vec![space_net::ALPN.to_vec()];
    let quic = quinn::crypto::rustls::QuicServerConfig::try_from(Arc::new(config))
        .expect("QUIC server config");
    let endpoint = space_net::server_endpoint(
        SocketAddr::from(([127, 0, 0, 1], 0)),
        quinn::ServerConfig::with_crypto(Arc::new(quic)),
    )
    .expect("bind");
    let addr = endpoint.local_addr().expect("addr");
    let accepting = endpoint.clone();
    tokio::spawn(async move {
        while let Some(incoming) = accepting.accept().await {
            tokio::spawn(async move {
                let _ = space_net::accept_incoming(incoming).await;
            });
        }
    });
    (endpoint, addr)
}

fn pin_of(secret: &str) -> ClusterPin {
    ClusterTrust::from_secret(secret).expect("secret").pin()
}

/// What the client does at every dial, whatever the shard answers (a welcome,
/// or a redirect: the handshake is what is pinned).
async fn dial(server: &Server) -> Result<(), NetError> {
    dial_at(server.addr()).await
}

async fn dial_at(addr: SocketAddr) -> Result<(), NetError> {
    tofu::join(addr, SERVER_NAME, &hello("tofu-test"))
        .await
        .map(|_| ())
}

#[tokio::test]
async fn the_first_connection_records_the_pin_and_succeeds() {
    let fixture = Fixture::new("first").await;
    let server = start(Some("cluster-one"));
    assert!(fixture.store().is_empty());
    dial(&server).await.expect("first use connects");
    let store = fixture.store();
    assert_eq!(store.len(), 1);
    assert_eq!(store.get(&server.key()), Some(pin_of("cluster-one")));
}

#[tokio::test]
async fn the_second_connection_uses_the_recorded_pin() {
    let fixture = Fixture::new("second").await;
    let server = start(Some("cluster-one"));
    dial(&server).await.expect("first");
    let after_first = std::fs::read(fixture.file()).expect("the store was written");
    dial(&server)
        .await
        .expect("second, held to the recorded pin");
    assert_eq!(std::fs::read(fixture.file()).unwrap(), after_first);
    // It is the recorded pin that is enforced: another server's pin in its
    // place makes the very same dial fail.
    let mut store = fixture.store();
    store.insert(&server.key(), pin_of("cluster-two"));
    store.save(&fixture.file()).unwrap();
    assert!(matches!(dial(&server).await, Err(NetError::Untrusted(_))));
}

#[tokio::test]
async fn a_server_of_another_ca_is_refused_and_the_store_is_unchanged() {
    let fixture = Fixture::new("mismatch").await;
    let server = start(Some("cluster-evil"));
    // What the user trusted for this host:port the first time.
    let mut store = PinStore::default();
    store.insert(&server.key(), pin_of("cluster-one"));
    store.save(&fixture.file()).unwrap();
    let before = std::fs::read(fixture.file()).unwrap();

    let err = dial(&server).await.expect_err("a different CA is refused");
    let NetError::Untrusted(message) = err else {
        panic!("wanted Untrusted, got {err:?}");
    };
    assert!(message.contains(&server.key()), "{message}");
    assert!(
        message.contains(&pin_of("cluster-one").to_string()),
        "{message}"
    );
    assert!(
        message.contains(&pin_of("cluster-evil").to_string()),
        "{message}"
    );
    assert!(message.contains("--reset-pins"), "{message}");
    assert_eq!(std::fs::read(fixture.file()).unwrap(), before);
}

#[tokio::test]
async fn a_pin_given_on_the_command_line_skips_the_store() {
    let fixture = Fixture::new("flag").await;
    let server = start(Some("cluster-one"));

    tofu::set_pin(Some(pin_of("cluster-one")));
    dial(&server).await.expect("--pin <correct> connects");
    assert!(
        !fixture.file().exists(),
        "nothing is stored for a given pin"
    );

    tofu::set_pin(Some(pin_of("cluster-two")));
    let err = dial(&server).await.expect_err("--pin <wrong> is refused");
    let NetError::Untrusted(message) = err else {
        panic!("wanted Untrusted, got {err:?}");
    };
    assert!(
        message.contains(&server.key()) && message.contains("--pin"),
        "{message}"
    );
    assert!(!fixture.file().exists());
}

#[tokio::test]
async fn a_server_without_a_cluster_secret_connects_unpinned_and_stores_nothing() {
    let fixture = Fixture::new("unpinned").await;
    let server = start(None);
    dial(&server).await.expect("no CA, no pin: unpinned");
    dial(&server).await.expect("and again");
    assert!(!fixture.file().exists());
}

#[tokio::test]
async fn a_stored_pin_refuses_a_server_that_presents_no_ca() {
    let fixture = Fixture::new("downgrade").await;
    let server = start(None);
    let mut store = PinStore::default();
    store.insert(&server.key(), pin_of("cluster-one"));
    store.save(&fixture.file()).unwrap();
    let err = dial(&server).await.expect_err("no CA where one was pinned");
    let NetError::Untrusted(message) = err else {
        panic!("wanted Untrusted, got {err:?}");
    };
    assert!(message.contains("none"), "{message}");
}

#[tokio::test]
async fn a_corrupt_store_is_treated_as_empty() {
    let fixture = Fixture::new("corrupt").await;
    let server = start(Some("cluster-one"));
    std::fs::create_dir_all(fixture.file().parent().unwrap()).unwrap();
    std::fs::write(fixture.file(), "{ this is not json").unwrap();
    dial(&server)
        .await
        .expect("a corrupt store is an empty one");
    assert_eq!(
        fixture.store().get(&server.key()),
        Some(pin_of("cluster-one")),
        "the pin learned is written to a fresh file"
    );
    assert_eq!(
        std::fs::read(fixture.file().with_file_name("pins.json.corrupt")).unwrap(),
        b"{ this is not json",
        "the corrupt file is kept, not overwritten"
    );
}

#[tokio::test]
async fn a_copied_ca_in_front_of_a_foreign_leaf_is_refused_and_nothing_is_stored() {
    let fixture = Fixture::new("forged").await;
    let (t1, t2) = (
        ClusterTrust::from_secret("cluster-one").unwrap(),
        ClusterTrust::from_secret("cluster-two").unwrap(),
    );
    let leaf2 = t2.issue();
    let chain = || vec![leaf2.cert.clone(), t1.ca_certificate().clone()];

    // First use: the real CA's certificate is copied, the leaf is not its own.
    let (_endpoint, addr) = forged(chain(), &leaf2);
    let first = dial_at(addr).await;
    assert!(matches!(first, Err(NetError::Untrusted(_))), "{first:?}");
    assert!(
        !fixture.file().exists(),
        "an impostor's pin must not be stored"
    );

    // Holding the real pin: the CA matches, the leaf is not its.
    let mut store = PinStore::default();
    store.insert(&addr.to_string(), t1.pin());
    store.save(&fixture.file()).unwrap();
    let before = std::fs::read(fixture.file()).unwrap();
    let Err(NetError::Untrusted(message)) = dial_at(addr).await else {
        panic!("wanted Untrusted");
    };
    assert!(message.contains("did not issue"), "{message}");
    assert!(
        !message.contains("received"),
        "the pin did match: {message}"
    );
    assert_eq!(std::fs::read(fixture.file()).unwrap(), before);
}

#[tokio::test]
async fn the_player_link_dials_through_the_store() {
    let fixture = Fixture::new("link").await;
    let server = start(Some("cluster-one"));
    // Whether this shard welcomes or redirects is beside the point: the link's
    // dial went through the store.
    let outcome = ServerLink::join(server.addr()).await;
    if let Ok(link) = &outcome {
        link.close();
    }
    assert_eq!(
        fixture.store().get(&server.key()),
        Some(pin_of("cluster-one"))
    );

    let mut store = PinStore::default();
    store.insert(&server.key(), pin_of("cluster-two"));
    store.save(&fixture.file()).unwrap();
    let refused = ServerLink::join(server.addr()).await.expect_err("mismatch");
    assert!(
        format!("{refused:?}").contains("Untrusted"),
        "the link reports the mismatch: {refused:?}"
    );
}

fn run_client(args: &[&str], pin_file: &Path) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_space-client"))
        .args(args)
        .env(PIN_FILE_ENV, pin_file)
        .output()
        .expect("the client binary starts")
}

#[tokio::test]
async fn reset_pins_deletes_the_file_and_exits_zero() {
    let fixture = Fixture::new("reset").await;
    let mut store = PinStore::default();
    store.insert("127.0.0.1:7777", pin_of("cluster-one"));
    store.save(&fixture.file()).unwrap();

    let out = run_client(&["--reset-pins"], &fixture.file());
    assert!(out.status.success(), "{out:?}");
    assert!(String::from_utf8_lossy(&out.stdout).contains("deleted"));
    assert!(!fixture.file().exists());

    let again = run_client(&["--reset-pins"], &fixture.file());
    assert!(again.status.success(), "no file is not an error: {again:?}");

    let help = run_client(&["--help"], &fixture.file());
    let text = String::from_utf8_lossy(&help.stdout);
    assert!(
        text.contains("--reset-pins") && text.contains("--pin"),
        "{text}"
    );
}
