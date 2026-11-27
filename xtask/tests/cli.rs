use std::path::{Path, PathBuf};
use std::process::Command;

fn xtask() -> Command {
    Command::new(env!("CARGO_BIN_EXE_xtask"))
}

/// The workspace root (the directory above this crate).
fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask lives in the workspace root")
        .to_path_buf()
}

/// Stdout of a successful `xtask <args>` run from `cwd`, with neither target
/// dir variable set unless `target_dir` gives one.
fn dry_run(args: &[&str], cwd: &Path, target_dir: Option<&Path>) -> String {
    let mut command = xtask();
    command
        .args(args)
        .current_dir(cwd)
        .env_remove("CARGO_TARGET_DIR")
        .env_remove("CARGO_BUILD_TARGET_DIR");
    if let Some(dir) = target_dir {
        command.env("CARGO_TARGET_DIR", dir);
    }
    let output = command.output().expect("spawn xtask");
    assert!(
        output.status.success(),
        "`xtask {args:?}` failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout).into_owned()
}

/// The `cargo test` / `cargo nextest` commands of a dry run (which of the two
/// it is depends on whether cargo-nextest is installed).
fn test_commands(stdout: &str) -> Vec<&str> {
    let commands: Vec<&str> = stdout
        .lines()
        .filter(|l| l.contains("`cargo test ") || l.contains("`cargo nextest "))
        .collect();
    assert!(!commands.is_empty(), "no test command in:\n{stdout}");
    commands
}

/// `ci` must do the same thing wherever it is started: from a member's
/// directory `cargo fmt --check` would only check that member and a relative
/// target dir would land inside it, so xtask moves to the workspace root first.
#[test]
fn ci_runs_from_the_workspace_root_whatever_the_cwd() {
    let root = workspace_root();
    let default_dir = Path::new("target").join("xtask-test");
    for cwd in [
        root.clone(),
        root.join("xtask").join("src"),
        root.join("crates"),
        std::env::temp_dir(),
    ] {
        let stdout = dry_run(&["ci", "--dry-run"], &cwd, None);
        let reported = stdout
            .lines()
            .find_map(|l| l.strip_prefix("xtask: workspace root: "))
            .unwrap_or_else(|| panic!("no workspace root line (cwd {cwd:?}):\n{stdout}"));
        assert_eq!(
            Path::new(reported).canonicalize().unwrap(),
            root.canonicalize().unwrap(),
            "started in {cwd:?}"
        );
        assert!(
            stdout.contains("xtask: would run `cargo fmt --check`")
                && stdout.contains("xtask: would run `cargo clippy --workspace"),
            "{stdout}"
        );
        // Relative, hence inside the workspace root xtask moved to.
        for command in test_commands(&stdout) {
            let expected = format!("--target-dir {}", default_dir.display());
            assert!(command.contains(&expected), "(cwd {cwd:?}) {command}");
        }
        assert!(
            !stdout.contains("xtask: running"),
            "--dry-run ran something:\n{stdout}"
        );
    }
}

/// With `CARGO_TARGET_DIR` set (say, to keep build output out of a synced
/// checkout) the test build must go there too, not to `<checkout>/target`.
#[test]
fn ci_builds_tests_inside_cargo_target_dir() {
    let target = std::env::temp_dir().join("xtask cli test").join("tärget");
    let expected = format!("--target-dir {}", target.join("xtask-test").display());
    let cwd = workspace_root().join("xtask");
    let stdout = dry_run(&["ci", "--dry-run"], &cwd, Some(&target));
    for command in test_commands(&stdout) {
        assert!(command.contains(&expected), "{command}\nwanted {expected}");
    }
    // A relative value means "relative to where I ran this", as for cargo.
    let stdout = dry_run(&["ci", "--dry-run"], &cwd, Some(Path::new("out")));
    let expected = format!(
        "--target-dir {}",
        cwd.join("out").join("xtask-test").display()
    );
    for command in test_commands(&stdout) {
        assert!(command.contains(&expected), "{command}\nwanted {expected}");
    }
}

#[test]
fn lint_dry_run_lists_fmt_and_clippy_only() {
    let stdout = dry_run(&["lint", "--dry-run"], &workspace_root(), None);
    let commands: Vec<&str> = stdout
        .lines()
        .filter_map(|l| l.strip_prefix("xtask: would run "))
        .collect();
    assert_eq!(
        commands,
        [
            "`cargo fmt --check`",
            "`cargo clippy --workspace --all-targets --locked -- -D warnings`"
        ]
    );
}

/// `gpu-setup` unpacks Linux packages. Anywhere else it must say so and stop
/// before creating the prefix (it used to create it, then fail on a missing
/// `dpkg-deb`).
#[cfg(not(target_os = "linux"))]
#[test]
fn gpu_setup_refuses_to_install_off_linux() {
    let prefix = std::env::temp_dir().join(format!("xtask gpu-setup {}", std::process::id()));
    let _ = std::fs::remove_dir_all(&prefix);
    let output = xtask()
        .arg("gpu-setup")
        .arg("--prefix")
        .arg(&prefix)
        .output()
        .expect("spawn xtask");
    let created = prefix.exists();
    let _ = std::fs::remove_dir_all(&prefix);
    assert_eq!(output.status.code(), Some(2), "{output:?}");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("only works on Linux"), "{stderr}");
    assert!(!created, "the prefix was created before refusing");
}

/// The exports are meant to be `eval`ed / `source`d, so each value is one
/// quoted shell word even when the prefix has a space in it, and nothing else
/// is printed.
#[test]
fn gpu_setup_print_env_quotes_the_prefix() {
    let output = xtask()
        .args(["gpu-setup", "--print-env", "--prefix"])
        .arg("/opt/my vulkan")
        .output()
        .expect("spawn xtask");
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    let lines: Vec<&str> = stdout.lines().collect();
    assert_eq!(lines.len(), 3, "{stdout}");
    for line in lines {
        let value = line
            .strip_prefix("export ")
            .and_then(|rest| rest.split_once('='))
            .map(|(_, value)| value)
            .unwrap_or_else(|| panic!("not an export line: {line}"));
        assert!(
            value.starts_with('\'') && value.ends_with('\'') && value.contains("my vulkan"),
            "{line}"
        );
    }
}

#[test]
fn stub_subcommands_print_and_exit_2() {
    for sub in ["load", "parity", "web-smoke", "accept"] {
        let output = xtask().arg(sub).output().expect("spawn xtask");
        assert_eq!(output.status.code(), Some(2), "`xtask {sub}` should exit 2");
        assert_eq!(
            String::from_utf8_lossy(&output.stdout).trim(),
            "not implemented",
            "`xtask {sub}` should print the stub message"
        );
    }
}

#[test]
fn gpu_setup_print_env_parses_and_prints_exports() {
    // `--print-env` must not touch the network: it only prints the exports.
    let output = xtask()
        .args(["gpu-setup", "--print-env"])
        .output()
        .expect("spawn xtask");
    assert!(
        output.status.success(),
        "`gpu-setup --print-env` should exit 0"
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("export LD_LIBRARY_PATH="),
        "missing LD_LIBRARY_PATH export in: {stdout}"
    );
    assert!(
        stdout.contains("export VK_DRIVER_FILES=")
            && stdout.contains("radeon_icd.json")
            && stdout.contains("lvp_icd.json"),
        "missing VK_DRIVER_FILES export in: {stdout}"
    );
    assert!(
        stdout.contains("export VK_LAYER_PATH=") && stdout.contains("explicit_layer.d"),
        "missing VK_LAYER_PATH export in: {stdout}"
    );
}

#[test]
fn gpu_setup_flags_are_accepted() {
    let output = xtask()
        .args(["gpu-setup", "--help"])
        .output()
        .expect("spawn xtask");
    assert!(output.status.success());
    let help = String::from_utf8_lossy(&output.stdout);
    for flag in ["--prefix", "--force", "--print-env"] {
        assert!(
            help.contains(flag),
            "`gpu-setup --help` should document {flag}"
        );
    }
}

#[test]
fn viewer_is_listed_and_documents_its_flags() {
    let output = xtask().arg("--help").output().expect("spawn xtask");
    assert!(output.status.success());
    let commands = String::from_utf8_lossy(&output.stdout);
    let listed = commands
        .lines()
        .any(|line| line.trim_start().starts_with("viewer ") && line.contains("browser viewer"));
    assert!(listed, "`xtask --help` should list viewer:\n{commands}");

    let output = xtask()
        .args(["viewer", "--help"])
        .output()
        .expect("spawn xtask");
    assert!(output.status.success());
    let help = String::from_utf8_lossy(&output.stdout);
    for text in [
        "--port <N>",
        "--bind <ADDR>",
        "--allow-remote",
        "--ws <URL>",
        "--no-build",
        "--install-bindgen",
        "--site-only",
        "[default: 8097]",
        "[default: 127.0.0.1]",
        "viewer-site",
        "space-client --stream",
    ] {
        assert!(
            help.contains(text),
            "`viewer --help` should have {text}:\n{help}"
        );
    }
}

/// The viewer server is a development tool. An address other machines can
/// reach is refused, before anything is built, unless `--allow-remote` is
/// given as well; so are arguments that cannot be meant.
#[test]
fn viewer_refuses_a_non_loopback_bind_and_bad_arguments() {
    for bind in ["0.0.0.0", "192.168.1.20", "::"] {
        let output = xtask()
            .args(["viewer", "--no-build", "--site-only", "--bind", bind])
            .current_dir(workspace_root())
            .output()
            .expect("spawn xtask");
        assert_eq!(output.status.code(), Some(2), "--bind {bind}: {output:?}");
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains("not a loopback address") && stderr.contains("--allow-remote"),
            "{stderr}"
        );
        assert!(output.stdout.is_empty(), "{output:?}");
    }
    for bad in [
        &["--bind", "localhost"][..],
        &["--port", "65536"],
        &["--port", "http"],
        &["--ws", "127.0.0.1:8099"],
        &["--ws", "http://127.0.0.1:8099"],
        &["--frobnicate"],
    ] {
        let output = xtask()
            .args(["viewer", "--no-build", "--site-only"])
            .args(bad)
            .output()
            .expect("spawn xtask");
        assert_eq!(output.status.code(), Some(2), "{bad:?}: {output:?}");
        assert!(output.stdout.is_empty(), "{bad:?}: {output:?}");
    }
}

/// Without the `wasm-bindgen` CLI the command stops with status 2 and the one
/// line that installs the version the lockfile asks for. It does not build or
/// write anything first, and it downloads nothing.
#[test]
fn viewer_names_the_install_command_when_wasm_bindgen_is_missing() {
    let root = workspace_root();
    let lockfile = std::fs::read_to_string(root.join("Cargo.lock")).expect("Cargo.lock");
    let mut lines = lockfile.lines().map(str::trim);
    lines
        .find(|line| *line == "name = \"wasm-bindgen\"")
        .expect("wasm-bindgen is in Cargo.lock");
    let version = lines
        .next()
        .and_then(|line| line.strip_prefix("version = \""))
        .and_then(|rest| rest.strip_suffix('"'))
        .expect("a version line after the name");

    // A machine with nothing installed: an empty PATH, home and cargo home.
    let empty = std::env::temp_dir().join(format!("xtask viewer empty {}", std::process::id()));
    let target = empty.join("target");
    std::fs::create_dir_all(&empty).expect("temp dir");
    let output = xtask()
        .args(["viewer", "--no-build", "--site-only"])
        .current_dir(root.join("xtask"))
        .env("PATH", &empty)
        .env("CARGO_HOME", &empty)
        .env("HOME", &empty)
        .env("USERPROFILE", &empty)
        .env("CARGO_TARGET_DIR", &target)
        .env_remove("CARGO_BUILD_TARGET_DIR")
        .env_remove("CARGO_INSTALL_ROOT")
        .output()
        .expect("spawn xtask");
    let wrote = target.exists();
    let _ = std::fs::remove_dir_all(&empty);

    assert_eq!(output.status.code(), Some(2), "{output:?}");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(
        stderr.lines().last(),
        Some(&*format!(
            "cargo install --locked wasm-bindgen-cli --version {version}"
        )),
        "{stderr}"
    );
    assert!(
        stderr.contains("not found") && stderr.contains("--install-bindgen"),
        "{stderr}"
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(!stdout.contains("xtask: running"), "{stdout}");
    assert!(!wrote, "the target directory was created before stopping");
}

/// `xtask e2e --dry-run` from the workspace root, without the variables that
/// change what it prints, plus `vars` and `extra` arguments.
fn e2e_dry_run(vars: &[(&str, &str)], extra: &[&str]) -> String {
    let mut command = xtask();
    command
        .args(["e2e", "--dry-run"])
        .args(extra)
        .current_dir(workspace_root())
        .env_remove("DOCKER_HOST")
        .env_remove("DOCKER_GID")
        .env_remove("CARGO_TARGET_DIR")
        .env_remove("CARGO_BUILD_TARGET_DIR");
    for (key, value) in vars {
        command.env(key, value);
    }
    let output = command.output().expect("spawn xtask");
    assert!(output.status.success(), "{output:?}");
    String::from_utf8_lossy(&output.stdout).into_owned()
}

#[test]
fn e2e_dry_run_lists_every_command_in_order() {
    let stdout = e2e_dry_run(&[], &[]);
    assert!(!stdout.contains("xtask: running"), "{stdout}");
    assert!(!stdout.contains("SPACE_CLUSTER_SECRET"), "{stdout}");
    assert!(
        stdout.contains("SPACE_E2E_SERVER=127.0.0.1:7877"),
        "{stdout}"
    );
    let dir = Path::new("target").join("xtask-test");
    let dir = dir.display();
    let compose = "docker compose -p space-e2e -f deploy/docker-compose.yaml";
    let all = "docker ps --all --quiet --filter label=space.instance=space-e2e --filter \
               label=space.role=shard";
    let expected = [
        "docker version --format {{.Server.Version}}".to_owned(),
        "docker compose version".to_owned(),
        format!(
            "cargo test -p space-client --locked --target-dir {dir} --test live_stack --no-run"
        ),
        "docker build --file deploy/Dockerfile.shard --tag space/shard:e2e .".to_owned(),
        "docker run --rm --entrypoint stat --volume /var/run/docker.sock:/var/run/docker.sock \
         space/shard:e2e -c %g /var/run/docker.sock"
            .to_owned(),
        format!("{compose} down --volumes --remove-orphans"),
        all.to_owned(),
        format!("{compose} up --build --detach"),
        "docker ps --quiet --filter label=space.instance=space-e2e --filter \
         label=space.role=shard --filter status=running"
            .to_owned(),
        format!(
            "cargo test -p space-client --locked --target-dir {dir} --test live_stack -- \
             --ignored --exact a_player_on_the_running_stack_is_handed_to_the_next_shard \
             --nocapture"
        ),
        format!("{compose} down --volumes --remove-orphans"),
        all.to_owned(),
    ];
    let got: Vec<&str> = stdout
        .lines()
        .filter_map(|line| line.strip_prefix("xtask: would run `"))
        .map(|line| line.strip_suffix('`').unwrap_or(line))
        .collect();
    assert_eq!(got, expected, "{stdout}");
}

#[test]
fn e2e_dry_run_follows_docker_host_and_docker_gid() {
    let vars = [("DOCKER_HOST", "tcp://dind:2375"), ("DOCKER_GID", "991")];
    let stdout = e2e_dry_run(&vars, &[]);
    assert!(stdout.contains("SPACE_E2E_SERVER=dind:7877"), "{stdout}");
    assert!(
        stdout.contains("DOCKER_GID=991 (from the environment)"),
        "{stdout}"
    );
    assert!(!stdout.contains("--entrypoint stat"), "{stdout}");
    let stdout = e2e_dry_run(&vars, &["--server-host", "10.1.2.3"]);
    assert!(
        stdout.contains("SPACE_E2E_SERVER=10.1.2.3:7877"),
        "{stdout}"
    );
}

#[test]
fn e2e_without_docker_exits_2_and_says_it_is_not_a_pass() {
    let empty = std::env::temp_dir().join(format!("xtask e2e empty {}", std::process::id()));
    std::fs::create_dir_all(&empty).expect("temp dir");
    let output = xtask()
        .arg("e2e")
        .current_dir(workspace_root())
        .env("PATH", &empty)
        // Windows also looks for `docker.exe` outside PATH (system directories, the
        // application directory), so a runner with Docker installed may still find
        // it. Point it at a daemon address nothing listens on: found or not, there
        // is no working daemon.
        .env("DOCKER_HOST", "tcp://127.0.0.1:1")
        .env_remove("DOCKER_CONTEXT")
        .env_remove("DOCKER_GID")
        .output()
        .expect("spawn xtask");
    let _ = std::fs::remove_dir_all(&empty);
    assert_eq!(output.status.code(), Some(2), "{output:?}");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("needs a Docker daemon") && stderr.contains("not a pass"),
        "{stderr}"
    );
    assert!(!String::from_utf8_lossy(&output.stdout).contains("cargo test"));
}

#[test]
fn e2e_documents_its_flags() {
    let output = xtask()
        .args(["e2e", "--help"])
        .output()
        .expect("spawn xtask");
    assert!(output.status.success(), "{output:?}");
    let help = String::from_utf8_lossy(&output.stdout);
    assert!(
        help.contains("--dry-run") && help.contains("--server-host <HOST>"),
        "{help}"
    );
}

#[test]
fn lint_subcommand_is_accepted() {
    // `xtask lint` shells out to cargo, so only check that the subcommand parses.
    let output = xtask()
        .arg("lint")
        .arg("--help")
        .output()
        .expect("spawn xtask");
    assert!(output.status.success());
}
