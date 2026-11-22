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
    for sub in ["e2e", "load", "parity", "web-smoke", "accept"] {
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
fn lint_subcommand_is_accepted() {
    // `xtask lint` shells out to cargo, so only check that the subcommand parses.
    let output = xtask()
        .arg("lint")
        .arg("--help")
        .output()
        .expect("spawn xtask");
    assert!(output.status.success());
}
