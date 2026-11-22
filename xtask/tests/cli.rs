use std::process::Command;

fn xtask() -> Command {
    Command::new(env!("CARGO_BIN_EXE_xtask"))
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
