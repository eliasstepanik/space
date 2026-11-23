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
//!   # with the built-in web UI (off unless asked for)
//!   space-orchestrator --webui 127.0.0.1:7980
//!   # the scene a fresh cluster shows (default: demo-area)
//!   space-orchestrator --scene solar
//! ```
//!
//! There is no argument-parsing dependency (mirrors `space-server`): the grammar
//! is a handful of flags, matched by hand against [`std::env::args`].
//!
//! # The web UI
//!
//! `--webui <addr:port>` serves the operator dashboard
//! ([`space_orchestrator::webui`]). On a loopback address it needs nothing
//! else. On any other address the orchestrator refuses to start without a
//! token, which a browser then asks for (any user name, the token as the
//! password). The token is best given through the environment, so it does not
//! show in the process list:
//!
//! | flag | environment fallback | |
//! |---|---|---|
//! | `--webui <addr:port>` | `SPACE_ORCHESTRATOR_WEBUI` | where to listen; unset = no UI |
//! | `--webui-token <secret>` | `SPACE_ORCHESTRATOR_WEBUI_TOKEN` | the password |
//! | `--webui-token-file <path>` | `SPACE_ORCHESTRATOR_WEBUI_TOKEN_FILE` | read it from a file (a container secret) |
//! | `--webui-host <name>[:<port>]` (repeatable) | `SPACE_ORCHESTRATOR_WEBUI_HOSTS` (comma-separated) | extra `Host` names to answer to, e.g. a proxy's public name; with a port, only on that port. **Needs a token**, whatever the bind address |
//!
//! A flag wins over its environment variable; an empty variable counts as unset.
//!
//! "No token on loopback" holds only while nothing but this machine can reach
//! the port. A UI without a token therefore refuses a public `--webui-host`
//! at startup, answers only to a loopback `Host`, and refuses requests that
//! carry a proxy's headers — but a proxy that adds none and rewrites `Host` to
//! `127.0.0.1` cannot be told from a local browser: **give the UI a token
//! before putting any proxy in front of it** ([`space_orchestrator::webui`],
//! "Without a token").
//!
//! # Limits
//!
//! `--max-regions <n>` (default 64) is the most regions the map may hold; a
//! split that would exceed it is refused, for the autoscaler and the operator
//! alike.
//!
//! # The scene
//!
//! | flag | environment fallback | |
//! |---|---|---|
//! | `--scene <name>` | `SPACE_ORCHESTRATOR_SCENE` | the **initial** scene: `demo-area` (the default), `materials`, `solar` or `galaxy` |
//!
//! The scene is the cluster's: every shard is told it and tells its clients.
//! An operator changes it in the web UI; a scene chosen there is saved in
//! `<state file>.scene` and is what a restarted orchestrator comes back with —
//! `--scene` only says what to show until then (delete the file to make it
//! apply again). An unknown name is an error at startup.

use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use space_orchestrator::boot::boot;
use space_orchestrator::config::{LauncherKind, OrchestratorConfig};
use space_orchestrator::launcher::docker::DockerLauncher;
use space_orchestrator::launcher::local::LocalLauncher;
use space_orchestrator::launcher::Launcher;
use space_orchestrator::orchestrator::Orchestrator;
use space_orchestrator::webui::{WebUi, WebUiConfig};

/// What the command line (and the environment) said.
#[derive(Debug)]
struct Options {
    config: OrchestratorConfig,
    peer_token: String,
    /// The web UI to serve; `None` (the default) serves none.
    webui: Option<WebUiConfig>,
}

const USAGE: &str =
    "usage: space-orchestrator [--bind <addr:port>] [--control-addr <addr:port>] [--cluster <id>] \
                     [--state <path>] [--first-id <n>] [--launcher local|docker] \
                     [--peer-token <secret>] [--scene demo-area|materials|solar|galaxy] \
                     [--max-regions <n>] \
                     [--webui <addr:port>] [--webui-token <secret> | --webui-token-file <path>] \
                     [--webui-host <name>[:<port>]]... \
     (environment fallbacks: SPACE_ORCHESTRATOR_SCENE for --scene; for the --webui flags \
     SPACE_ORCHESTRATOR_WEBUI, SPACE_ORCHESTRATOR_WEBUI_TOKEN, \
     SPACE_ORCHESTRATOR_WEBUI_TOKEN_FILE, SPACE_ORCHESTRATOR_WEBUI_HOSTS)";

/// Environment fallback for `--scene`.
const ENV_SCENE: &str = "SPACE_ORCHESTRATOR_SCENE";

/// Environment fallback for `--webui`.
const ENV_WEBUI: &str = "SPACE_ORCHESTRATOR_WEBUI";
/// Environment fallback for `--webui-token`.
const ENV_WEBUI_TOKEN: &str = "SPACE_ORCHESTRATOR_WEBUI_TOKEN";
/// Environment fallback for `--webui-token-file`.
const ENV_WEBUI_TOKEN_FILE: &str = "SPACE_ORCHESTRATOR_WEBUI_TOKEN_FILE";
/// Environment fallback for `--webui-host` (comma-separated).
const ENV_WEBUI_HOSTS: &str = "SPACE_ORCHESTRATOR_WEBUI_HOSTS";

/// The web UI flags as given, before they are combined with the environment.
#[derive(Debug, Default)]
struct WebUiFlags {
    bind: Option<String>,
    token: Option<String>,
    token_file: Option<PathBuf>,
    hosts: Vec<String>,
}

impl WebUiFlags {
    /// Combine the flags with the environment into a configuration, or `None`
    /// when no address was given anywhere (the UI stays off).
    ///
    /// The "no open bind without a token" rule is [`WebUiConfig::new`]'s; a
    /// configuration that breaks it is an error here, before anything starts.
    fn resolve(self, env: &dyn Fn(&str) -> Option<String>) -> Result<Option<WebUiConfig>, String> {
        let set = |name: &str| env(name).filter(|value| !value.trim().is_empty());
        let Some(bind) = self.bind.or_else(|| set(ENV_WEBUI)) else {
            if self.token.is_some() || self.token_file.is_some() || !self.hosts.is_empty() {
                return Err(
                    "--webui-token, --webui-token-file and --webui-host need --webui <addr:port>"
                        .to_owned(),
                );
            }
            return Ok(None);
        };
        let bind: SocketAddr = bind
            .trim()
            .parse()
            .map_err(|e| format!("web UI address {bind:?}: {e}"))?;
        let token = match (self.token, self.token_file) {
            (Some(_), Some(_)) => {
                return Err("give --webui-token or --webui-token-file, not both".to_owned());
            }
            (Some(token), None) => Some(token),
            (None, Some(path)) => Some(read_token_file(&path)?),
            (None, None) => match (set(ENV_WEBUI_TOKEN), set(ENV_WEBUI_TOKEN_FILE)) {
                (Some(token), _) => Some(token),
                (None, Some(path)) => Some(read_token_file(Path::new(&path))?),
                (None, None) => None,
            },
        };
        let hosts = if self.hosts.is_empty() {
            set(ENV_WEBUI_HOSTS).map_or_else(Vec::new, |names| {
                names
                    .split(',')
                    .map(str::trim)
                    .filter(|name| !name.is_empty())
                    .map(str::to_owned)
                    .collect()
            })
        } else {
            self.hosts
        };
        WebUiConfig::new(bind, token, hosts).map(Some)
    }
}

/// The token in `path`: the file's content without surrounding whitespace.
fn read_token_file(path: &Path) -> Result<String, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("web UI token file {}: {e}", path.display()))?;
    let token = text.trim();
    if token.is_empty() {
        return Err(format!("web UI token file {} is empty", path.display()));
    }
    Ok(token.to_owned())
}

impl Options {
    /// Parse `argv` without the program name. `env` looks a variable up (the
    /// process environment in production, a table in tests).
    fn parse(args: &[String], env: &dyn Fn(&str) -> Option<String>) -> Result<Self, String> {
        let mut config = OrchestratorConfig::default();
        let mut peer_token = String::new();
        let mut webui = WebUiFlags::default();
        let mut scene: Option<String> = None;
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
                "--scene" => {
                    scene = Some(value("--scene")?.clone());
                }
                "--max-regions" => {
                    config.max_regions = value("--max-regions")?
                        .parse::<usize>()
                        .ok()
                        .filter(|limit| *limit >= 1)
                        .ok_or("--max-regions needs a number from 1 up")?;
                }
                "--webui" => {
                    webui.bind = Some(value("--webui")?.clone());
                }
                "--webui-token" => {
                    webui.token = Some(value("--webui-token")?.clone());
                }
                "--webui-token-file" => {
                    webui.token_file = Some(PathBuf::from(value("--webui-token-file")?));
                }
                "--webui-host" => {
                    webui.hosts.push(value("--webui-host")?.clone());
                }
                other => return Err(format!("unknown argument {other}\n{USAGE}")),
            }
            i += if flag == "--help" || flag == "-h" {
                1
            } else {
                2
            };
        }
        // The initial scene: the flag, else the environment, else the default.
        // An unknown name is refused here rather than falling back, so a typo
        // cannot quietly show another world.
        let from_env = || env(ENV_SCENE).filter(|value| !value.trim().is_empty());
        if let Some(name) = scene.or_else(from_env) {
            config.initial_scene = name
                .trim()
                .parse::<space_protocol::Scene>()
                .map_err(|e| format!("--scene ({ENV_SCENE}): {e}"))?;
        }
        Ok(Self {
            config,
            peer_token,
            webui: webui.resolve(env)?,
        })
    }
}

fn main() -> Result<(), String> {
    let started = Instant::now();
    let options = Options::parse(&std::env::args().skip(1).collect::<Vec<_>>(), &|name| {
        std::env::var(name).ok()
    })?;
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| e.to_string())?;
    runtime.block_on(run(options, started))
}

/// Build the control plane and run the loop until the endpoint is closed.
async fn run(options: Options, started: Instant) -> Result<(), String> {
    let Options {
        config,
        peer_token,
        webui,
    } = options;
    let webui = webui.map(|webui| webui.with_started(started));
    match config.launcher {
        LauncherKind::Local => serve(config, peer_token, webui, LocalLauncher::new()).await,
        LauncherKind::Docker => {
            // Deferred (bollard is not in the offline cache): fail fast with the
            // launcher's own error rather than at the first split.
            let launcher = DockerLauncher::new();
            launcher.ensure_available().map_err(|e| e.to_string())?;
            serve(config, peer_token, webui, launcher).await
        }
    }
}

/// Open the id counter, build the control plane and run the loop with `launcher`.
async fn serve<L: Launcher + 'static>(
    config: OrchestratorConfig,
    peer_token: String,
    webui: Option<WebUiConfig>,
    launcher: L,
) -> Result<(), String> {
    // The initial map: one region at the configured root, owned by the first id.
    // The orchestrator allocates that id so a restart resumes above it (H9); the
    // first live shard that asks for the root adopts it (`boot` marks it unclaimed).
    let control = boot(config.clone(), peer_token)?;
    let root_owner = control
        .unclaimed_root_owner()
        .ok_or("boot left no root owner")?;
    let mut orchestrator = Orchestrator::new(control, Arc::new(launcher));
    println!(
        "space-orchestrator: cluster {:?} on {}, root owned by {root_owner}",
        config.cluster, config.bind
    );
    println!("{}", scene_line(&orchestrator.control.scene, &config));
    // The web UI runs in its own task and holds only a handle: it reads the
    // snapshots the loop publishes and offers commands to the loop's queue, so
    // nothing it does (or a client does to it) can hold the loop up. An address
    // that cannot be bound is a startup error, like the control port's.
    let _webui = match webui {
        Some(webui) => {
            let bind = webui.bind();
            let ui = WebUi::bind(webui, orchestrator.attach_ops())
                .await
                .map_err(|e| format!("web UI: cannot listen on {bind}: {e}"))?;
            println!("space-orchestrator: web UI on {}", ui.describe());
            Some(tokio::spawn(ui.serve()))
        }
        None => None,
    };
    orchestrator.run().await.map_err(|e| e.to_string())
}

/// One line for the startup log: the scene the cluster starts with and where
/// it came from.
fn scene_line(
    scene: &space_orchestrator::scene::SceneStore,
    config: &OrchestratorConfig,
) -> String {
    let current = scene.current();
    let file = scene
        .path()
        .map_or_else(|| "nowhere".to_owned(), |path| path.display().to_string());
    if scene.restored() {
        let unused = if scene.scene() == config.initial_scene {
            String::new()
        } else {
            format!(
                " (a saved scene wins over the initial scene {})",
                config.initial_scene
            )
        };
        format!("space-orchestrator: scene {current}, restored from {file}{unused}")
    } else {
        format!(
            "space-orchestrator: scene {current}, the initial scene; a change is saved in {file}"
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Parse with an empty environment.
    fn parse(args: &[&str]) -> Result<Options, String> {
        parse_with(args, &[])
    }

    /// Parse with the given environment variables set.
    fn parse_with(args: &[&str], env: &[(&str, &str)]) -> Result<Options, String> {
        let args: Vec<String> = args.iter().map(|s| (*s).to_owned()).collect();
        Options::parse(&args, &|name| {
            env.iter()
                .find(|(n, _)| *n == name)
                .map(|(_, v)| (*v).to_owned())
        })
    }

    const TOKEN: &str = "a-long-enough-test-token";

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
        assert!(options.webui.is_none(), "the web UI is off by default");
    }

    #[test]
    fn the_region_limit_has_a_default_and_a_flag() {
        assert_eq!(
            parse(&[]).expect("parses").config.max_regions,
            OrchestratorConfig::DEFAULT_MAX_REGIONS
        );
        assert_eq!(
            parse(&["--max-regions", "12"])
                .expect("parses")
                .config
                .max_regions,
            12
        );
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
        let err = run(options, Instant::now())
            .await
            .expect_err("docker is deferred");
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

    /// Mutation this catches: a `--scene` that is parsed and dropped, the
    /// environment winning over the flag, or an unknown name quietly becoming
    /// the default (a typo would show another world on every shard).
    #[test]
    fn the_initial_scene_comes_from_the_flag_then_the_environment_then_the_default() {
        use space_protocol::Scene;
        assert_eq!(
            parse(&[]).expect("parses").config.initial_scene,
            Scene::DemoArea,
            "the demo area is the default"
        );
        for scene in Scene::ALL {
            let by_flag = parse(&["--scene", scene.name()])
                .unwrap_or_else(|err| panic!("{scene} must parse: {err}"));
            assert_eq!(by_flag.config.initial_scene, scene);
            let by_env =
                parse_with(&[], &[("SPACE_ORCHESTRATOR_SCENE", scene.name())]).expect("parses");
            assert_eq!(by_env.config.initial_scene, scene);
        }
        // A flag wins over the environment; an empty variable is unset.
        let both = parse_with(
            &["--scene", "solar"],
            &[("SPACE_ORCHESTRATOR_SCENE", "galaxy")],
        )
        .expect("parses");
        assert_eq!(both.config.initial_scene, Scene::Solar);
        let empty = parse_with(&[], &[("SPACE_ORCHESTRATOR_SCENE", "  ")]).expect("parses");
        assert_eq!(empty.config.initial_scene, Scene::DemoArea);
        let padded = parse_with(&[], &[("SPACE_ORCHESTRATOR_SCENE", " solar\n")]).expect("parses");
        assert_eq!(padded.config.initial_scene, Scene::Solar);

        // An unknown name is an error that lists the catalogue, from either source.
        for bad in ["Solar", "demo", "4"] {
            let err = parse(&["--scene", bad]).expect_err(bad);
            assert!(
                err.contains("demo-area, materials, solar, galaxy"),
                "{bad}: {err}"
            );
            let err = parse_with(&[], &[("SPACE_ORCHESTRATOR_SCENE", bad)]).expect_err(bad);
            assert!(err.contains("SPACE_ORCHESTRATOR_SCENE"), "{bad}: {err}");
        }
        assert!(parse(&["--scene"]).is_err(), "the flag needs a value");
    }

    #[test]
    fn the_startup_line_says_which_scene_and_where_it_came_from() {
        use space_orchestrator::scene::{scene_path, SceneStore};
        use space_protocol::Scene;
        let state = std::env::temp_dir().join(format!(
            "space-orch-main-scene-{}-{}.id",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.subsec_nanos())
                .unwrap_or(0)
        ));
        let file = scene_path(&state);
        let _ = std::fs::remove_file(&file);
        let config = OrchestratorConfig {
            state_path: state,
            initial_scene: Scene::Galaxy,
            ..OrchestratorConfig::default()
        };
        let mut store = SceneStore::open(&file, config.initial_scene);
        let line = scene_line(&store, &config);
        assert!(
            line.contains("scene galaxy (revision 1), the initial scene")
                && line.contains(&file.display().to_string()),
            "{line}"
        );
        store.set(Scene::Solar).expect("saved");
        let restored = SceneStore::open(&file, config.initial_scene);
        let line = scene_line(&restored, &config);
        assert!(
            line.contains("scene solar (revision 2), restored from")
                && line.contains("wins over the initial scene galaxy"),
            "{line}"
        );
        std::fs::remove_file(&file).ok();
    }

    #[test]
    fn a_bad_command_line_is_an_error_not_a_panic() {
        for args in [
            &["--bind", "nope"] as &[&str],
            &["--launcher", "kubernetes"],
            &["--nope"],
            &["--first-id", "x"],
            &["--scene", "nowhere"],
            &["--max-regions", "0"],
            &["--max-regions", "many"],
            &["--max-regions"],
            &["--webui", "nope"],
            &["--webui"],
        ] {
            assert!(parse(args).is_err(), "{args:?} must be rejected");
        }
        assert!(parse(&["--help"]).is_err(), "--help prints usage and exits");
    }

    #[test]
    fn the_web_ui_is_on_only_when_given_an_address() {
        let options = parse(&["--webui", "127.0.0.1:7980"]).expect("parses");
        let webui = options.webui.expect("the UI is on");
        assert_eq!(webui.bind(), "127.0.0.1:7980".parse().unwrap());
        assert!(!webui.token_required(), "loopback needs no token");

        // The environment turns it on too; a flag wins over it.
        let options =
            parse_with(&[], &[("SPACE_ORCHESTRATOR_WEBUI", "127.0.0.1:7981")]).expect("parses");
        assert_eq!(options.webui.expect("on").bind().port(), 7_981);
        let options = parse_with(
            &["--webui", "127.0.0.1:7980"],
            &[("SPACE_ORCHESTRATOR_WEBUI", "127.0.0.1:7981")],
        )
        .expect("parses");
        assert_eq!(options.webui.expect("on").bind().port(), 7_980);

        // An empty variable is unset, and a token alone starts nothing.
        let options = parse_with(
            &[],
            &[
                ("SPACE_ORCHESTRATOR_WEBUI", ""),
                ("SPACE_ORCHESTRATOR_WEBUI_TOKEN", TOKEN),
            ],
        )
        .expect("parses");
        assert!(options.webui.is_none());
        // ...but the flags without an address are a mistake worth reporting.
        assert!(parse(&["--webui-token", TOKEN]).is_err());
        assert!(parse(&["--webui-host", "ops.example.org"]).is_err());
    }

    /// Mutation this catches: starting an unauthenticated control surface on a
    /// reachable address because nobody thought to set a token.
    #[test]
    fn a_non_loopback_web_ui_does_not_start_without_a_token() {
        for open in ["0.0.0.0:7980", "[::]:7980", "192.168.1.5:7980"] {
            let err = parse(&["--webui", open]).expect_err(open);
            assert!(err.contains("without a token"), "{err}");
            let err = parse_with(&[], &[("SPACE_ORCHESTRATOR_WEBUI", open)]).expect_err(open);
            assert!(err.contains("without a token"), "{err}");
            let err = parse(&["--webui", open, "--webui-token", "short"]).expect_err(open);
            assert!(err.contains("too short"), "{err}");

            let by_flag = parse(&["--webui", open, "--webui-token", TOKEN]).expect("with a token");
            assert!(by_flag.webui.expect("on").token_required());
            let by_env = parse_with(
                &["--webui", open],
                &[("SPACE_ORCHESTRATOR_WEBUI_TOKEN", TOKEN)],
            )
            .expect("with a token from the environment");
            assert!(by_env.webui.expect("on").token_required());
        }
    }

    /// Mutation this catches: a loopback bind plus a public name starting
    /// without a token ("no token: loopback only" — behind a proxy that serves
    /// it to everyone).
    #[test]
    fn a_public_host_name_does_not_start_without_a_token() {
        let err = parse(&[
            "--webui",
            "127.0.0.1:7980",
            "--webui-host",
            "ops.example.org",
        ])
        .expect_err("a public name needs a token");
        assert!(err.contains("without a token"), "{err}");
        let err = parse_with(
            &["--webui", "127.0.0.1:7980"],
            &[("SPACE_ORCHESTRATOR_WEBUI_HOSTS", "ops.example.org")],
        )
        .expect_err("from the environment too");
        assert!(err.contains("without a token"), "{err}");

        let ok = parse(&[
            "--webui",
            "127.0.0.1:7980",
            "--webui-host",
            "ops.example.org:8443",
            "--webui-token",
            TOKEN,
        ])
        .expect("with a token, and a pinned port");
        let debug = format!("{:?}", ok.webui.expect("on"));
        assert!(
            debug.contains("ops.example.org") && debug.contains("8443"),
            "{debug}"
        );
    }

    #[test]
    fn the_token_can_come_from_a_file_and_hosts_from_a_list() {
        let path = std::env::temp_dir().join(format!(
            "space-orch-webui-token-{}-{}.txt",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.subsec_nanos())
                .unwrap_or(0)
        ));
        std::fs::write(&path, format!("{TOKEN}\n")).expect("write the token file");
        let file = path.to_str().expect("a UTF-8 temp path");

        let by_flag = parse(&["--webui", "0.0.0.0:7980", "--webui-token-file", file])
            .expect("the file's token is long enough");
        assert!(by_flag.webui.expect("on").token_required());
        let by_env = parse_with(
            &["--webui", "0.0.0.0:7980"],
            &[
                ("SPACE_ORCHESTRATOR_WEBUI_TOKEN_FILE", file),
                (
                    "SPACE_ORCHESTRATOR_WEBUI_HOSTS",
                    "ops.example.org, , Other.Example.Org",
                ),
            ],
        )
        .expect("parses");
        let debug = format!("{:?}", by_env.webui.expect("on"));
        assert!(
            debug.contains("ops.example.org") && debug.contains("other.example.org"),
            "{debug}"
        );
        assert!(!debug.contains(TOKEN), "the token is never printed");

        // Both at once is ambiguous; an empty or missing file is an error.
        assert!(parse(&[
            "--webui",
            "0.0.0.0:7980",
            "--webui-token",
            TOKEN,
            "--webui-token-file",
            file
        ])
        .is_err());
        std::fs::write(&path, "\n").expect("empty the token file");
        let err = parse(&["--webui", "0.0.0.0:7980", "--webui-token-file", file])
            .expect_err("an empty file is no token");
        assert!(err.contains("is empty"), "{err}");
        std::fs::remove_file(&path).ok();
        assert!(parse(&["--webui", "0.0.0.0:7980", "--webui-token-file", file]).is_err());

        // A host that is not a bare name is refused.
        assert!(parse(&[
            "--webui",
            "127.0.0.1:7980",
            "--webui-host",
            "https://ops.example.org"
        ])
        .is_err());
    }
}
