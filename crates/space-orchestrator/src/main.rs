//! `space-orchestrator` — the cluster control plane.
//!
//! Allocates shard ids, decides when a region is too hot or too cold, publishes
//! every ownership change through the epoch protocol, and launches the shards
//! that carry the regions. The binary only turns text into an
//! [`OrchestratorConfig`] and runs the loop; every decision lives in the crate.
//!
//! ```text
//!   space-orchestrator --bind 127.0.0.1:7900 --cluster space-local \
//!                      --state space-orchestrator.id
//!   # in a container: bind the wildcard, tell shards the dialable address
//!   space-orchestrator --bind 0.0.0.0:7900 --control-addr orchestrator:7900
//! ```
//!
//! There is no argument-parsing dependency (mirrors `space-server`): the grammar
//! is a handful of flags, matched by hand against [`std::env::args`].

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use space_orchestrator::config::{LauncherKind, OrchestratorConfig};
use space_orchestrator::control::ControlPlane;
use space_orchestrator::epoch::EpochPublisher;
use space_orchestrator::ids::PersistentShardIdAllocator;
use space_orchestrator::launcher::docker::DockerLauncher;
use space_orchestrator::launcher::local::LocalLauncher;
use space_orchestrator::launcher::Launcher;
use space_orchestrator::orchestrator::Orchestrator;
use space_protocol::shard::{ShardMap, ShardRegion};

/// What the command line said.
#[derive(Debug)]
struct Options {
    config: OrchestratorConfig,
    peer_token: String,
}

const USAGE: &str =
    "usage: space-orchestrator [--bind <addr:port>] [--control-addr <addr:port>] [--cluster <id>] \
                     [--state <path>] [--first-id <n>] [--launcher local|docker] \
                     [--peer-token <secret>]";

impl Options {
    /// Parse `argv` without the program name.
    fn parse(args: &[String]) -> Result<Self, String> {
        let mut config = OrchestratorConfig::default();
        let mut peer_token = String::new();
        let mut i = 0;
        while i < args.len() {
            let flag = args[i].as_str();
            let value = |what: &str| -> Result<&String, String> {
                args.get(i + 1)
                    .ok_or_else(|| format!("{what} needs a value"))
            };
            match flag {
                "-h" | "--help" => return Err(USAGE.to_owned()),
                "--bind" => {
                    config.bind = value("--bind")?
                        .parse::<SocketAddr>()
                        .map_err(|e| e.to_string())?;
                }
                "--control-addr" => {
                    config.control_addr = Some(
                        value("--control-addr")?
                            .parse::<SocketAddr>()
                            .map_err(|e| e.to_string())?,
                    );
                }
                "--cluster" => {
                    config.cluster = value("--cluster")?.clone();
                }
                "--state" => {
                    config.state_path = PathBuf::from(value("--state")?);
                }
                "--first-id" => {
                    config.first_shard_id = value("--first-id")?
                        .parse::<u64>()
                        .map_err(|e| e.to_string())?;
                }
                "--launcher" => {
                    config.launcher = match value("--launcher")?.as_str() {
                        "local" => LauncherKind::Local,
                        "docker" => LauncherKind::Docker,
                        other => return Err(format!("unknown launcher {other}")),
                    };
                }
                "--peer-token" => {
                    peer_token = value("--peer-token")?.clone();
                }
                other => return Err(format!("unknown argument {other}\n{USAGE}")),
            }
            i += if flag == "--help" || flag == "-h" {
                1
            } else {
                2
            };
        }
        Ok(Self { config, peer_token })
    }
}

fn main() -> Result<(), String> {
    let options = Options::parse(&std::env::args().skip(1).collect::<Vec<_>>())?;
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| e.to_string())?;
    runtime.block_on(run(options))
}

/// Build the control plane and run the loop until the endpoint is closed.
async fn run(options: Options) -> Result<(), String> {
    let config = options.config;
    match config.launcher {
        LauncherKind::Local => serve(config, options.peer_token, LocalLauncher::new()).await,
        LauncherKind::Docker => {
            // Deferred (bollard is not in the offline cache): fail fast with the
            // launcher's own error rather than at the first split.
            let launcher = DockerLauncher::new();
            launcher.ensure_available().map_err(|e| e.to_string())?;
            serve(config, options.peer_token, launcher).await
        }
    }
}

/// Open the id counter, build the control plane and run the loop with `launcher`.
async fn serve<L: Launcher + 'static>(
    config: OrchestratorConfig,
    peer_token: String,
    launcher: L,
) -> Result<(), String> {
    // The initial map: one region at the configured root, owned by the first id.
    // The orchestrator allocates that id so a restart resumes above it (H9).
    let mut ids = PersistentShardIdAllocator::open(&config.state_path, config.first_shard_id)
        .map_err(|e| e.to_string())?;
    let root_owner = ids.allocate();
    let map = ShardMap::new(
        1,
        vec![ShardRegion::new(
            config.root_node(),
            root_owner,
            config.bind,
        )],
    );
    let control = ControlPlane::new(config.clone(), ids, EpochPublisher::new(map), peer_token);
    let mut orchestrator = Orchestrator::new(control, Arc::new(launcher));
    println!(
        "space-orchestrator: cluster {:?} on {}, root owned by {root_owner}",
        config.cluster, config.bind
    );
    orchestrator.run().await.map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Result<Options, String> {
        Options::parse(&args.iter().map(|s| (*s).to_owned()).collect::<Vec<_>>())
    }

    #[test]
    fn defaults_match_the_config() {
        let options = parse(&[]).expect("parses");
        assert_eq!(
            options.config.cluster,
            OrchestratorConfig::default().cluster
        );
        assert_eq!(options.config.bind, OrchestratorConfig::default().bind);
        assert_eq!(options.config.control_addr, None);
        assert_eq!(options.config.control_addr(), options.config.bind);
    }

    #[test]
    fn control_addr_is_parsed_and_distinct_from_bind() {
        let options =
            parse(&["--bind", "0.0.0.0:7900", "--control-addr", "10.0.0.2:7900"]).expect("parses");
        assert_eq!(options.config.bind, "0.0.0.0:7900".parse().unwrap());
        assert_eq!(
            options.config.control_addr(),
            "10.0.0.2:7900".parse().unwrap()
        );
        assert!(parse(&["--control-addr", "nope"]).is_err());
    }

    #[tokio::test]
    async fn the_docker_launcher_is_an_unavailable_error_not_a_panic() {
        let options = parse(&["--launcher", "docker"]).expect("parses");
        let err = run(options).await.expect_err("docker is deferred");
        assert!(err.contains("bollard"), "{err}");
    }

    #[test]
    fn flags_override_the_config() {
        let options = parse(&[
            "--bind",
            "127.0.0.1:9000",
            "--cluster",
            "test",
            "--first-id",
            "10",
            "--launcher",
            "local",
            "--peer-token",
            "s3cret",
        ])
        .expect("parses");
        assert_eq!(options.config.bind.port(), 9_000);
        assert_eq!(options.config.cluster, "test");
        assert_eq!(options.config.first_shard_id, 10);
        assert_eq!(options.peer_token, "s3cret");
    }

    #[test]
    fn a_bad_command_line_is_an_error_not_a_panic() {
        for args in [
            &["--bind", "nope"] as &[&str],
            &["--launcher", "kubernetes"],
            &["--nope"],
            &["--first-id", "x"],
        ] {
            assert!(parse(args).is_err(), "{args:?} must be rejected");
        }
        assert!(parse(&["--help"]).is_err(), "--help prints usage and exits");
    }
}
