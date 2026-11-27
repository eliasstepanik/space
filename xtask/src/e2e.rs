//! `xtask e2e`: the Docker end-to-end check.
//!
//! Builds the shard image, starts `deploy/docker-compose.yaml` as its own
//! compose project on its own ports, waits for four shard containers, runs the
//! ignored `live_stack` test against it and takes the stack down again. The
//! cluster secret of the run is a throwaway one that is handed to the compose
//! commands through their environment and never printed.

use std::collections::hash_map::RandomState;
use std::ffi::OsString;
use std::hash::{BuildHasher, Hasher};
use std::path::Path;
use std::process::{exit, Command, Stdio};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use super::{os_args, test_target_dir, Runner};

const PROJECT: &str = "space-e2e";
const COMPOSE_FILE: &str = "deploy/docker-compose.yaml";
const SHARD_DOCKERFILE: &str = "deploy/Dockerfile.shard";
const SHARD_IMAGE: &str = "space/shard:e2e";
const ORCHESTRATOR_IMAGE: &str = "space/orchestrator:e2e";
const PLAYER_PORT: u16 = 7877;
const WEBUI_PORT: u16 = 7981;
const SHARD_PORTS: &str = "7970-7999";
const SHARD_NAME_PREFIX: &str = "space-e2e-shard-";
const MIN_SHARDS: usize = 4;
const LIVE_TEST: &str = "a_player_on_the_running_stack_is_handed_to_the_next_shard";
const SERVER_ENV: &str = "SPACE_E2E_SERVER";
const SOCKET: &str = "/var/run/docker.sock";
/// One poll every two seconds: 180 s in all.
const READY_POLLS: u32 = 90;
const POLL_EVERY: Duration = Duration::from_secs(2);

type Env = [(&'static str, String)];

#[derive(clap::Args)]
pub struct E2eArgs {
    /// Print the commands without running them (nothing is built, started or removed)
    #[arg(long)]
    dry_run: bool,
    /// Host the stack's published ports are reached at [default: the host of a tcp://
    /// DOCKER_HOST, else 127.0.0.1]
    #[arg(long, value_name = "HOST")]
    server_host: Option<String>,
}

// ---------------------------------------------------------------------------
// pure parts
// ---------------------------------------------------------------------------

/// Where the published ports of the stack are reached: the host of a `tcp://`
/// `DOCKER_HOST` (the daemon is there), else this machine.
fn server_host(docker_host: Option<&str>) -> String {
    const LOCAL: &str = "127.0.0.1";
    let Some(rest) = docker_host.and_then(|value| value.strip_prefix("tcp://")) else {
        return LOCAL.to_owned();
    };
    let authority = rest.split('/').next().unwrap_or_default();
    let host = match authority.rsplit_once(':') {
        Some((host, port)) if !port.is_empty() && port.bytes().all(|b| b.is_ascii_digit()) => host,
        _ => authority,
    };
    if host.is_empty() {
        LOCAL.to_owned()
    } else {
        host.to_owned()
    }
}

/// 48 hex characters nobody could guess in advance: a hash of per-process
/// random state, the process id and the clock. It protects a stack that lives
/// for a few minutes.
fn throwaway_secret() -> String {
    use sha2::{Digest, Sha256};
    let random = || RandomState::new().build_hasher().finish();
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |since| since.as_nanos());
    let mut hasher = Sha256::new();
    hasher.update(random().to_le_bytes());
    hasher.update(random().to_le_bytes());
    hasher.update(std::process::id().to_le_bytes());
    hasher.update(nanos.to_le_bytes());
    let hex: String = hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    hex[..48].to_owned()
}

/// The variables `deploy/docker-compose.yaml` reads, for this run.
fn compose_env(secret: &str, docker_gid: &str) -> Vec<(&'static str, String)> {
    vec![
        ("SPACE_CLUSTER_SECRET", secret.to_owned()),
        ("DOCKER_GID", docker_gid.to_owned()),
        ("SPACE_CLUSTER", PROJECT.to_owned()),
        ("SPACE_INSTANCE", PROJECT.to_owned()),
        (
            "SPACE_ORCHESTRATOR_CONTAINER",
            format!("{PROJECT}-orchestrator-1"),
        ),
        ("SPACE_PLAYER_PORT", PLAYER_PORT.to_string()),
        ("WEBUI_PORT", WEBUI_PORT.to_string()),
        ("SPACE_SHARD_PORTS", SHARD_PORTS.to_owned()),
        ("SPACE_SHARD_NAME_PREFIX", SHARD_NAME_PREFIX.to_owned()),
        ("SPACE_SHARD_IMAGE", SHARD_IMAGE.to_owned()),
        ("SPACE_ORCHESTRATOR_IMAGE", ORCHESTRATOR_IMAGE.to_owned()),
        ("SPACE_MIN_SHARDS", MIN_SHARDS.to_string()),
        ("SPACE_SCENE", "demo-area".to_owned()),
    ]
}

/// The test binary reads one variable; the cluster secret is not for it.
fn test_run_env(server: &str) -> [(&'static str, String); 1] {
    [(SERVER_ENV, server.to_owned())]
}

/// The output of `stat -c %g`.
fn parse_gid(stdout: &str) -> Option<u32> {
    stdout.trim().parse().ok()
}

fn compose_args(rest: &[&str]) -> Vec<OsString> {
    let mut args = os_args(&["compose", "-p", PROJECT, "-f", COMPOSE_FILE]);
    args.extend(os_args(rest));
    args
}

fn build_args() -> Vec<OsString> {
    os_args(&[
        "build",
        "--file",
        SHARD_DOCKERFILE,
        "--tag",
        SHARD_IMAGE,
        ".",
    ])
}

fn gid_probe_args() -> Vec<OsString> {
    let volume = format!("{SOCKET}:{SOCKET}");
    os_args(&[
        "run",
        "--rm",
        "--entrypoint",
        "stat",
        "--volume",
        &volume,
        SHARD_IMAGE,
        "-c",
        "%g",
        SOCKET,
    ])
}

fn all_shards_args() -> Vec<OsString> {
    let label = format!("label=space.instance={PROJECT}");
    os_args(&[
        "ps",
        "--all",
        "--quiet",
        "--filter",
        &label,
        "--filter",
        "label=space.role=shard",
    ])
}

fn running_shards_args() -> Vec<OsString> {
    let label = format!("label=space.instance={PROJECT}");
    os_args(&[
        "ps",
        "--quiet",
        "--filter",
        &label,
        "--filter",
        "label=space.role=shard",
        "--filter",
        "status=running",
    ])
}

fn test_build_args(dir: &Path) -> Vec<OsString> {
    let mut args = os_args(&["test", "-p", "space-client", "--locked", "--target-dir"]);
    args.push(dir.into());
    args.extend(os_args(&["--test", "live_stack", "--no-run"]));
    args
}

fn test_run_args(dir: &Path) -> Vec<OsString> {
    let mut args = os_args(&["test", "-p", "space-client", "--locked", "--target-dir"]);
    args.push(dir.into());
    args.extend(os_args(&[
        "--test",
        "live_stack",
        "--",
        "--ignored",
        "--exact",
        LIVE_TEST,
        "--nocapture",
    ]));
    args
}

// ---------------------------------------------------------------------------
// running commands
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct Shell {
    dry_run: bool,
}

fn shown(program: &str, args: &[OsString]) -> String {
    let mut text = program.to_owned();
    for arg in args {
        text.push(' ');
        text.push_str(&arg.to_string_lossy());
    }
    text
}

fn code_of(status: std::process::ExitStatus) -> i32 {
    status.code().filter(|code| *code != 0).unwrap_or(1)
}

impl Shell {
    fn announce(self, program: &str, args: &[OsString], echo: bool) {
        let text = shown(program, args);
        if self.dry_run {
            println!("xtask: would run `{text}`");
        } else if echo {
            println!("xtask: running `{text}`");
        }
    }

    /// `None` on success (or in a dry run), else the exit code, never 0. The
    /// environment is not printed: it holds the secret.
    fn status(self, program: &str, args: &[OsString], env: &Env) -> Option<i32> {
        self.announce(program, args, true);
        if self.dry_run {
            return None;
        }
        let result = Command::new(program)
            .args(args)
            .envs(env.iter().map(|(k, v)| (*k, v)))
            .status();
        match result {
            Ok(status) if status.success() => None,
            Ok(status) => {
                eprintln!("xtask: `{}` failed: {status}", shown(program, args));
                Some(code_of(status))
            }
            Err(err) => {
                eprintln!("xtask: cannot run `{program}`: {err}");
                Some(127)
            }
        }
    }

    /// The stdout of the command; stderr is passed on.
    fn capture(
        self,
        program: &str,
        args: &[OsString],
        env: &Env,
        echo: bool,
    ) -> Result<String, i32> {
        self.announce(program, args, echo);
        if self.dry_run {
            return Ok(String::new());
        }
        let result = Command::new(program)
            .args(args)
            .envs(env.iter().map(|(k, v)| (*k, v)))
            .stdin(Stdio::null())
            .stderr(Stdio::inherit())
            .output();
        match result {
            Ok(output) if output.status.success() => {
                Ok(String::from_utf8_lossy(&output.stdout).into_owned())
            }
            Ok(output) => {
                if echo {
                    eprintln!(
                        "xtask: `{}` failed: {}",
                        shown(program, args),
                        output.status
                    );
                }
                Err(code_of(output.status))
            }
            Err(err) => {
                eprintln!("xtask: cannot run `{program}`: {err}");
                Err(127)
            }
        }
    }
}

fn ids(stdout: &str) -> Vec<OsString> {
    stdout
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(OsString::from)
        .collect()
}

fn force_remove(shell: Shell, ids: &[OsString], env: &Env) -> Option<i32> {
    let mut args = os_args(&["rm", "--force"]);
    args.extend(ids.iter().cloned());
    shell.status("docker", &args, env)
}

// ---------------------------------------------------------------------------
// the command
// ---------------------------------------------------------------------------

pub fn run(args: E2eArgs, invoked_from: &Path) -> ! {
    Runner::at_workspace_root(invoked_from, args.dry_run);
    let shell = Shell {
        dry_run: args.dry_run,
    };
    let host = args
        .server_host
        .unwrap_or_else(|| server_host(std::env::var("DOCKER_HOST").ok().as_deref()));
    let server = format!("{host}:{PLAYER_PORT}");
    println!(
        "xtask: e2e: compose project {PROJECT}, player port {host}:{PLAYER_PORT}/udp, shard ports \
         {SHARD_PORTS}/udp, web UI 127.0.0.1:{WEBUI_PORT}"
    );
    println!("xtask: e2e: {SERVER_ENV}={server}");

    // 3. Preflight.
    let version = os_args(&["version", "--format", "{{.Server.Version}}"]);
    let compose_version = os_args(&["compose", "version"]);
    for check in [&version, &compose_version] {
        if shell.capture("docker", check, &[], true).is_err() {
            eprintln!(
                "xtask: e2e needs a Docker daemon and the compose plugin: `{}` failed. Nothing \
                 was checked: this is not a pass.",
                shown("docker", check)
            );
            exit(2);
        }
    }

    // 4. Test build.
    let test_dir = test_target_dir(|var| std::env::var_os(var), invoked_from);
    if let Some(code) = shell.status("cargo", &test_build_args(&test_dir), &[]) {
        exit(code);
    }
    // 5. Image.
    if let Some(code) = shell.status("docker", &build_args(), &[]) {
        exit(code);
    }

    // 6. The group of the Docker socket.
    let gid = docker_gid(shell);

    // 7. This run's compose environment.
    let secret = throwaway_secret();
    let env = compose_env(&secret, &gid);

    // 8. Leftovers of an earlier run.
    let down = compose_args(&["down", "--volumes", "--remove-orphans"]);
    if let Some(code) = shell.status("docker", &down, &env) {
        exit(code);
    }
    let left = match shell.capture("docker", &all_shards_args(), &env, true) {
        Ok(out) => ids(&out),
        Err(code) => {
            eprintln!("xtask: e2e: cannot check for leftover shard containers");
            exit(code);
        }
    };
    if !left.is_empty() {
        if let Some(code) = force_remove(shell, &left, &env) {
            exit(code);
        }
    }

    // 9-11.
    let mut ok = true;
    let up = compose_args(&["up", "--build", "--detach"]);
    if shell.status("docker", &up, &env).is_some() || !wait_for_shards(shell, &env) {
        ok = false;
    } else {
        let test_env = test_run_env(&server);
        if shell
            .status("cargo", &test_run_args(&test_dir), &test_env)
            .is_some()
        {
            ok = false;
        }
    }

    // 12. Why, before anything is removed.
    if !ok {
        show_why(shell, &env);
    }

    // 13. Teardown.
    if shell.status("docker", &down, &env).is_some() {
        ok = false;
    }
    let behind = match shell.capture("docker", &all_shards_args(), &env, true) {
        Ok(out) => ids(&out),
        Err(_) => {
            eprintln!("xtask: e2e: cannot check for leftover shard containers");
            ok = false;
            Vec::new()
        }
    };
    if !behind.is_empty() {
        eprintln!(
            "xtask: e2e: the orchestrator left {} shard container(s) behind on `compose down`",
            behind.len()
        );
        force_remove(shell, &behind, &env);
        ok = false;
    }

    // 14.
    if ok {
        if shell.dry_run {
            println!("xtask: e2e: dry run, nothing was run");
        } else {
            println!(
                "xtask: e2e passed: a player joined through {server} and was handed from one \
                 shard container to the next"
            );
        }
        exit(0);
    }
    eprintln!("xtask: e2e FAILED");
    exit(1)
}

fn docker_gid(shell: Shell) -> String {
    if let Some(value) = std::env::var("DOCKER_GID")
        .ok()
        .filter(|value| !value.is_empty())
    {
        println!("xtask: e2e: DOCKER_GID={value} (from the environment)");
        return value;
    }
    let probed = shell.capture("docker", &gid_probe_args(), &[], true);
    if shell.dry_run {
        return "0".to_owned();
    }
    match probed.ok().as_deref().and_then(parse_gid) {
        Some(gid) => {
            println!("xtask: e2e: DOCKER_GID={gid} (the group of {SOCKET} as the daemon sees it)");
            gid.to_string()
        }
        None => {
            eprintln!("xtask: cannot read the Docker socket's group; set DOCKER_GID and run again");
            exit(1)
        }
    }
}

/// Step 10. In a dry run the command is printed once and nothing is waited for.
fn wait_for_shards(shell: Shell, env: &Env) -> bool {
    let mut running = 0;
    for poll in 0..READY_POLLS {
        let out = shell.capture("docker", &running_shards_args(), env, poll == 0);
        if shell.dry_run {
            return true;
        }
        running = out.map_or(0, |out| ids(&out).len());
        if running >= MIN_SHARDS {
            return true;
        }
        if poll + 1 < READY_POLLS {
            std::thread::sleep(POLL_EVERY);
        }
    }
    let seconds = u64::from(READY_POLLS) * POLL_EVERY.as_secs();
    eprintln!(
        "xtask: e2e: only {running} of {MIN_SHARDS} shard containers were running after \
         {seconds} s"
    );
    false
}

fn show_why(shell: Shell, env: &Env) {
    shell.status(
        "docker",
        &compose_args(&["logs", "--no-color", "--tail", "300"]),
        env,
    );
    let label = format!("label=space.instance={PROJECT}");
    shell.status(
        "docker",
        &os_args(&["ps", "--all", "--filter", &label]),
        env,
    );
    let all = shell
        .capture("docker", &all_shards_args(), env, true)
        .map(|out| ids(&out))
        .unwrap_or_default();
    for id in all {
        let mut args = os_args(&["logs", "--tail", "80"]);
        args.push(id);
        shell.status("docker", &args, env);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text(args: Vec<OsString>) -> String {
        shown("docker", &args)
    }

    #[test]
    fn the_server_host_follows_a_tcp_docker_host() {
        assert_eq!(server_host(None), "127.0.0.1");
        assert_eq!(
            server_host(Some("unix:///var/run/docker.sock")),
            "127.0.0.1"
        );
        assert_eq!(server_host(Some("ssh://me@box")), "127.0.0.1");
        assert_eq!(
            server_host(Some("npipe:////./pipe/docker_engine")),
            "127.0.0.1"
        );
        assert_eq!(server_host(Some("tcp://dind:2375")), "dind");
        assert_eq!(server_host(Some("tcp://10.0.0.5:2376/")), "10.0.0.5");
        assert_eq!(server_host(Some("tcp://dind")), "dind");
        assert_eq!(server_host(Some("tcp://[::1]:2375")), "[::1]");
        assert_eq!(server_host(Some("tcp://")), "127.0.0.1");
    }

    #[test]
    fn the_secret_is_48_hex_characters_and_new_every_time() {
        let (a, b) = (throwaway_secret(), throwaway_secret());
        assert_eq!(a.len(), 48);
        assert!(a.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')));
        assert_ne!(a, b);
    }

    #[test]
    fn every_compose_variable_is_read_by_the_compose_file() {
        let file = include_str!("../../deploy/docker-compose.yaml");
        for (key, _) in compose_env("secret", "991") {
            assert!(file.contains(&format!("${{{key}")), "{key}");
        }
        let env = compose_env("s3cret", "991");
        assert_eq!(env[0], ("SPACE_CLUSTER_SECRET", "s3cret".to_owned()));
        assert_eq!(env[1], ("DOCKER_GID", "991".to_owned()));
        assert_eq!(env.len(), 13);
    }

    #[test]
    fn the_live_test_and_its_variable_are_where_xtask_expects_them() {
        let test = include_str!("../../crates/space-client/tests/live_stack.rs");
        assert!(test.contains(&format!("fn {LIVE_TEST}(")));
        assert!(test.contains(SERVER_ENV));
    }

    #[test]
    fn the_test_run_gets_only_the_server_address() {
        let env = test_run_env("dind:7877");
        assert_eq!(env, [("SPACE_E2E_SERVER", "dind:7877".to_owned())]);
        assert!(env.iter().all(|(key, _)| *key != "SPACE_CLUSTER_SECRET"));
    }

    #[test]
    fn the_group_id_is_the_trimmed_number() {
        assert_eq!(parse_gid("991\n"), Some(991));
        assert_eq!(parse_gid("0"), Some(0));
        assert_eq!(parse_gid(""), None);
        assert_eq!(parse_gid("root"), None);
    }

    #[test]
    fn the_command_lines_are_the_documented_ones() {
        assert_eq!(
            text(compose_args(&["up", "--build", "--detach"])),
            "docker compose -p space-e2e -f deploy/docker-compose.yaml up --build --detach"
        );
        assert_eq!(
            text(build_args()),
            "docker build --file deploy/Dockerfile.shard --tag space/shard:e2e ."
        );
        assert_eq!(
            text(gid_probe_args()),
            "docker run --rm --entrypoint stat --volume /var/run/docker.sock:/var/run/docker.sock \
             space/shard:e2e -c %g /var/run/docker.sock"
        );
        assert_eq!(
            text(all_shards_args()),
            "docker ps --all --quiet --filter label=space.instance=space-e2e --filter \
             label=space.role=shard"
        );
        assert_eq!(
            text(running_shards_args()),
            "docker ps --quiet --filter label=space.instance=space-e2e --filter \
             label=space.role=shard --filter status=running"
        );
        let dir = Path::new("t");
        assert_eq!(
            shown("cargo", &test_build_args(dir)),
            "cargo test -p space-client --locked --target-dir t --test live_stack --no-run"
        );
        assert_eq!(
            shown("cargo", &test_run_args(dir)),
            "cargo test -p space-client --locked --target-dir t --test live_stack -- --ignored \
             --exact a_player_on_the_running_stack_is_handed_to_the_next_shard --nocapture"
        );
    }

    #[test]
    fn container_ids_are_the_non_empty_lines() {
        assert_eq!(ids("a\n\n b \r\n"), [OsString::from("a"), "b".into()]);
        assert!(ids("\n").is_empty());
    }
}
