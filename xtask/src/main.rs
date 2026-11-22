use std::path::{Path, PathBuf};
use std::process::{exit, Command, Stdio};

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(
    name = "xtask",
    about = "space development tasks",
    version,
    subcommand_required = true,
    arg_required_else_help = true
)]
struct Cli {
    #[command(subcommand)]
    cmd: Task,
}

#[derive(Subcommand)]
enum Task {
    /// fmt + clippy + nextest (falls back to `cargo test` when nextest is absent)
    Ci,
    /// fmt + clippy only
    Lint,
    /// End-to-end tests (implemented in a later phase)
    E2e,
    /// Load benchmarks (implemented in a later phase)
    Load,
    /// Cross-platform parity checks (implemented in a later phase)
    Parity,
    /// Browser smoke test (implemented in a later phase)
    WebSmoke,
    /// Full acceptance run (implemented in a later phase)
    Accept,
    /// Install the userspace Vulkan stack into a writable prefix (spikes/ENVIRONMENT.md)
    GpuSetup {
        /// Install prefix; defaults to $SPACE_VULKAN_PREFIX, else /workspace/.local/vulkan,
        /// else $HOME/.local/vulkan
        #[arg(long, value_name = "PATH")]
        prefix: Option<PathBuf>,
        /// Re-download and re-extract every package even if the prefix looks complete
        #[arg(long)]
        force: bool,
        /// Only print the required export lines and exit 0 (no network, no install)
        #[arg(long)]
        print_env: bool,
    },
}

fn main() {
    match Cli::parse().cmd {
        Task::Ci => {
            lint();
            test();
        }
        Task::Lint => lint(),
        Task::GpuSetup {
            prefix,
            force,
            print_env,
        } => gpu_setup(prefix, force, print_env),
        Task::E2e | Task::Load | Task::Parity | Task::WebSmoke | Task::Accept => stub(),
    }
}

fn lint() {
    run("cargo", &["fmt", "--check"]);
    run(
        "cargo",
        &[
            "clippy",
            "--workspace",
            "--all-targets",
            "--locked",
            "--",
            "-D",
            "warnings",
        ],
    );
}

fn test() {
    // Build tests into a separate target dir so a running `xtask.exe` is
    // never relinked while it is still executing (Windows file locking).
    const TEST_TARGET_DIR: &str = "target/xtask-test";
    if nextest_available() {
        run(
            "cargo",
            &[
                "nextest",
                "run",
                "--workspace",
                "--locked",
                "--target-dir",
                TEST_TARGET_DIR,
            ],
        );
    } else {
        eprintln!("xtask: cargo-nextest not found, falling back to `cargo test`");
        run(
            "cargo",
            &[
                "test",
                "--workspace",
                "--locked",
                "--target-dir",
                TEST_TARGET_DIR,
            ],
        );
    }
}

fn nextest_available() -> bool {
    Command::new("cargo")
        .args(["nextest", "--version"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

fn run(program: &str, args: &[&str]) {
    println!("xtask: running `{program} {}`", args.join(" "));
    let status = Command::new(program)
        .args(args)
        .status()
        .unwrap_or_else(|err| panic!("xtask: failed to spawn `{program}`: {err}"));
    if !status.success() {
        eprintln!("xtask: `{program} {}` failed: {status}", args.join(" "));
        exit(status.code().unwrap_or(1));
    }
}

fn stub() -> ! {
    println!("not implemented");
    exit(2)
}

// ---------------------------------------------------------------------------
// gpu-setup
// ---------------------------------------------------------------------------

/// Debian pool base for the pinned `.deb`s documented in `spikes/ENVIRONMENT.md`.
/// Plain HTTP: the sandbox egress proxy only needs `deb.debian.org` allowed and
/// no TLS stack has to be linked into xtask.
const DEBIAN_POOL: &str = "http://deb.debian.org/debian/pool/main";

/// `(package filename, pool subdirectory, sha256)`. Versions are pinned exactly
/// as in `spikes/ENVIRONMENT.md`: bookworm `libvulkan1`/`libdrm2` plus
/// bookworm-backports Mesa 25.0.7 (knows GFX1151) and `libdrm-amdgpu1`, plus
/// bookworm `vulkan-validationlayers` (the `VK_LAYER_KHRONOS_validation` layer).
///
/// The `sha256` is the hash of the `.deb` as published in the pool; every
/// download is verified against it *before* `dpkg-deb -x` runs, so a truncated,
/// truncated-or-substituted download can never be unpacked into the prefix.
const VULKAN_DEBS: &[(&str, &str, &str)] = &[
    (
        "libvulkan1_1.3.239.0-1_amd64.deb",
        "v/vulkan-loader",
        "622f28b807ba55a1f5ad1622ff83aa2c5b11fedbec074850e04c10a8f28b43d3",
    ),
    (
        "mesa-vulkan-drivers_25.0.7-2~bpo12+1_amd64.deb",
        "m/mesa",
        "3899f1800424c60a05df5d6913076a526ca8fdbabec43672cd397563d1ca3263",
    ),
    (
        "libdrm-amdgpu1_2.4.123-1~bpo12+1_amd64.deb",
        "libd/libdrm",
        "334e2ae389af0e9cb607c2ccfbbf7ae9feb34131ff998550baeffeeae6cda296",
    ),
    (
        "libdrm2_2.4.123-1~bpo12+1_amd64.deb",
        "libd/libdrm",
        "9cc2f341964286295124a32d3625edb76730500625e54a1e682761f883064d7b",
    ),
    (
        "vulkan-validationlayers_1.3.239.0-2_amd64.deb",
        "v/vulkan-validationlayers",
        "b88ccb5c5925bc05d4a6c50f1784fbd4e92a523c4dd5779fb539c128ba5c2526",
    ),
];

fn gpu_setup(prefix_override: Option<PathBuf>, force: bool, print_env: bool) -> ! {
    let prefix = prefix_override.unwrap_or_else(default_prefix);
    let exports = env_exports(&prefix);

    if print_env {
        print!("{exports}");
        exit(0);
    }

    println!("xtask: Vulkan prefix: {}", prefix.display());

    if !force && vulkan_installed(&prefix) {
        println!("xtask: already installed (loader + both ICD manifests present); pass --force to reinstall");
    } else {
        install_vulkan_stack(&prefix);
    }

    let env_sh = prefix.join("env.sh");
    match std::fs::write(&env_sh, &exports) {
        Ok(()) => println!("xtask: wrote {}", env_sh.display()),
        Err(err) => {
            eprintln!("xtask: failed to write {}: {err}", env_sh.display());
            exit(1);
        }
    }
    println!(
        "xtask: export the following (or `source {}`):",
        env_sh.display()
    );
    print!("{exports}");

    // Device summary. Kept as an explicit step after the env is written, and it
    // never silently no-ops: a pruned `spikes/` tree prints a loud warning.
    match probe_devices(&prefix) {
        Probe::Ok => exit(0),
        Probe::Unavailable => exit(1),
        Probe::Skipped => {
            eprintln!(
                "xtask: WARNING: skipped the Vulkan device summary — `spikes/vkprobe` is not present.\n\
                 This is not a pass: the prefix was installed but never probed. Either restore\n\
                 spikes/vkprobe, or run the equivalent check yourself with the exports above\n\
                 (e.g. `vulkaninfo --summary`, or `cd spikes/vkprobe && cargo run --release`)."
            );
            exit(0);
        }
    }
}

/// Outcome of the device-summary step.
enum Probe {
    /// Vulkan is usable in this prefix.
    Ok,
    /// The probe ran and Vulkan is not usable (or the probe itself failed).
    Unavailable,
    /// The probe could not run at all (e.g. `spikes/` was pruned).
    Skipped,
}

/// `$SPACE_VULKAN_PREFIX`, else `/workspace/.local/vulkan` when `/workspace`
/// exists (the sandbox workspace), else `$HOME/.local/vulkan`.
fn default_prefix() -> PathBuf {
    if let Some(p) = std::env::var_os("SPACE_VULKAN_PREFIX") {
        if !p.is_empty() {
            return PathBuf::from(p);
        }
    }
    if Path::new("/workspace").is_dir() {
        return PathBuf::from("/workspace/.local/vulkan");
    }
    match std::env::var_os("HOME") {
        Some(home) if !home.is_empty() => PathBuf::from(home).join(".local/vulkan"),
        _ => PathBuf::from(".local/vulkan"),
    }
}

/// The `export` lines a shell needs to use the prefix.
///
/// `VK_LAYER_PATH` points at the `explicit_layer.d` directory unpacked from the
/// `vulkan-validationlayers` deb. The manifest there names its library as a
/// bare `libVkLayer_khronos_validation.so`, which the loader resolves through
/// the dynamic loader search path — that works because the `.so` lands in the
/// same prefix `usr/lib/x86_64-linux-gnu` already on `LD_LIBRARY_PATH`.
fn env_exports(prefix: &Path) -> String {
    let prefix = prefix.display();
    format!(
        "export LD_LIBRARY_PATH={prefix}/usr/lib/x86_64-linux-gnu\n\
         export VK_DRIVER_FILES={prefix}/usr/share/vulkan/icd.d/radeon_icd.json:{prefix}/usr/share/vulkan/icd.d/lvp_icd.json\n\
         export VK_LAYER_PATH={prefix}/usr/share/vulkan/explicit_layer.d\n"
    )
}

fn icd_dir(prefix: &Path) -> PathBuf {
    prefix.join("usr/share/vulkan/icd.d")
}

fn layer_dir(prefix: &Path) -> PathBuf {
    prefix.join("usr/share/vulkan/explicit_layer.d")
}

/// True when the loader, both ICD manifests and the validation layer are
/// already in the prefix. The layer is part of "installed" so a prefix created
/// before `vulkan-validationlayers` was pinned is refreshed rather than reused
/// without it.
fn vulkan_installed(prefix: &Path) -> bool {
    let loader = prefix.join("usr/lib/x86_64-linux-gnu/libvulkan.so.1");
    let icd = icd_dir(prefix);
    loader.is_file()
        && icd.join("radeon_icd.json").is_file()
        && icd.join("lvp_icd.json").is_file()
        && layer_dir(prefix)
            .join("VkLayer_khronos_validation.json")
            .is_file()
}

fn install_vulkan_stack(prefix: &Path) {
    if let Err(err) = std::fs::create_dir_all(prefix) {
        eprintln!("xtask: cannot create {}: {err}", prefix.display());
        exit(1);
    }
    if Command::new("dpkg-deb")
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_err()
    {
        eprintln!(
            "xtask: `dpkg-deb` not found; it is required to unpack Debian packages.\n  \
             Manual step: install `dpkg` (Debian/Ubuntu), then for each .deb below run\n  \
             dpkg-deb -x <package>.deb {}",
            prefix.display()
        );
        exit(1);
    }

    let tmp = std::env::temp_dir().join(format!("space-vulkan-{}", std::process::id()));
    if let Err(err) = std::fs::create_dir_all(&tmp) {
        eprintln!("xtask: cannot create {}: {err}", tmp.display());
        exit(1);
    }

    for (name, subdir, expected_sha256) in VULKAN_DEBS {
        let url = format!("{DEBIAN_POOL}/{subdir}/{name}");
        let deb = tmp.join(name);
        println!("xtask: downloading {url}");
        if let Err(err) = download(&url, &deb) {
            eprintln!(
                "xtask: download of {name} failed: {err}\n  \
                 Manual step: curl -fsSL -o '{name}' '{url}' && dpkg-deb -x '{name}' '{}'",
                prefix.display()
            );
            let _ = std::fs::remove_dir_all(&tmp);
            exit(1);
        }
        // Verify before unpacking: never hand an unverified archive to dpkg-deb.
        if let Err(err) = verify_deb(&deb, expected_sha256) {
            eprintln!("xtask: {err}");
            let _ = std::fs::remove_dir_all(&tmp);
            exit(1);
        }
        println!("xtask: sha256 OK {name} ({expected_sha256})");
        println!("xtask: extracting {name} -> {}", prefix.display());
        let status = Command::new("dpkg-deb")
            .args([
                "-x",
                &deb.display().to_string(),
                &prefix.display().to_string(),
            ])
            .status();
        match status {
            Ok(s) if s.success() => {}
            Ok(s) => {
                eprintln!("xtask: `dpkg-deb -x {name}` failed: {s}");
                let _ = std::fs::remove_dir_all(&tmp);
                exit(s.code().unwrap_or(1));
            }
            Err(err) => {
                eprintln!("xtask: failed to spawn `dpkg-deb`: {err}");
                let _ = std::fs::remove_dir_all(&tmp);
                exit(1);
            }
        }
    }

    let _ = std::fs::remove_dir_all(&tmp);
}

fn download(url: &str, dest: &Path) -> Result<(), String> {
    let response = match ureq::get(url).call() {
        Ok(response) => response,
        Err(ureq::Error::Status(code, _)) => return Err(format!("HTTP {code}")),
        Err(ureq::Error::Transport(err)) => return Err(format!("transport error: {err}")),
    };
    if response.status() != 200 {
        return Err(format!("HTTP {}", response.status()));
    }
    let mut file = std::fs::File::create(dest).map_err(|err| err.to_string())?;
    std::io::copy(&mut response.into_reader(), &mut file).map_err(|err| err.to_string())?;
    Ok(())
}

/// Lowercase hex SHA-256 of a file, streamed (the .debs are tens of MB).
fn sha256_file(path: &Path) -> Result<String, String> {
    use sha2::{Digest, Sha256};
    let mut file = std::fs::File::open(path).map_err(|err| err.to_string())?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = std::io::Read::read(&mut file, &mut buf).map_err(|err| err.to_string())?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect())
}

/// Check a freshly downloaded `.deb` against its pinned SHA-256 **before** it is
/// unpacked. `Err` describes the mismatch (or the read failure) for the caller.
fn verify_deb(path: &Path, expected_sha256: &str) -> Result<(), String> {
    let actual =
        sha256_file(path).map_err(|err| format!("cannot hash {}: {err}", path.display()))?;
    if actual != expected_sha256 {
        return Err(format!(
            "sha256 MISMATCH for {}\n  \
             expected: {expected_sha256}\n  \
             actual:   {actual}\n  \
             Refusing to unpack. The pool may have moved this version; update the \
             pin in xtask/src/main.rs (VULKAN_DEBS) only after confirming the new \
             hash from the pool yourself.",
            path.display()
        ));
    }
    Ok(())
}

/// Run `spikes/vkprobe` with the prefix env and print its device summary.
/// Returns `Probe::Skipped` (never silently) when the spike is not present.
fn probe_devices(prefix: &Path) -> Probe {
    let spike = Path::new("spikes/vkprobe");
    if !spike.is_dir() {
        return Probe::Skipped;
    }

    let exports = env_exports(prefix);
    // `current_dir` below is the spike dir, so the binary path must be absolute.
    let mut command = match spike.join("target/release/vkprobe").canonicalize() {
        Ok(binary) => Command::new(binary),
        Err(_) => {
            let mut cmd = Command::new("cargo");
            cmd.args(["run", "--release", "--quiet"]);
            cmd
        }
    };
    command
        .current_dir(spike)
        .env(
            "LD_LIBRARY_PATH",
            format!("{}/usr/lib/x86_64-linux-gnu", prefix.display()),
        )
        .env(
            "VK_DRIVER_FILES",
            format!(
                "{}/usr/share/vulkan/icd.d/radeon_icd.json:{}/usr/share/vulkan/icd.d/lvp_icd.json",
                prefix.display(),
                prefix.display()
            ),
        )
        .env("VK_LAYER_PATH", layer_dir(prefix));

    let output = match command.output() {
        Ok(output) => output,
        Err(err) => {
            eprintln!(
                "xtask: failed to run vkprobe: {err}\n  Manual step: cd spikes/vkprobe && cargo run --release\n  (env: {})",
                exports.replace('\n', "; ")
            );
            return Probe::Unavailable;
        }
    };
    if !output.status.success() {
        eprintln!(
            "xtask: vkprobe exited with {}\n{}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        );
        return Probe::Unavailable;
    }

    let report: serde_json::Value = match serde_json::from_slice(&output.stdout) {
        Ok(report) => report,
        Err(err) => {
            eprintln!("xtask: vkprobe output was not valid JSON ({err})");
            return Probe::Unavailable;
        }
    };

    let available = report
        .get("vulkan_available")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    let found = report
        .get("devices_found")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    println!("xtask: vkprobe: vulkan_available={available} devices_found={found}");
    if let Some(devices) = report.get("devices").and_then(serde_json::Value::as_array) {
        for (i, device) in devices.iter().enumerate() {
            let name = device
                .get("device_name")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("<unknown>");
            let api = device
                .get("api_version")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("?");
            let kind = device
                .get("device_type")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("?");
            println!("xtask:   [{i}] {name} ({kind}, api {api})");
        }
    }

    if !available {
        let reason = report
            .get("loader_error")
            .or_else(|| report.get("instance_error"))
            .map(|v| v.to_string())
            .unwrap_or_else(|| "no reason reported".to_string());
        eprintln!("xtask: Vulkan is not usable in this prefix: {reason}");
        return Probe::Unavailable;
    }
    Probe::Ok
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every pinned deb must carry a real SHA-256 (64 lowercase hex chars); a
    /// placeholder would silently disable verification for that package.
    #[test]
    fn vulkan_debs_have_sha256_pins() {
        assert!(
            !VULKAN_DEBS.is_empty(),
            "VULKAN_DEBS must list the pinned packages"
        );
        for (name, subdir, sha) in VULKAN_DEBS {
            assert!(
                sha.len() == 64
                    && sha
                        .chars()
                        .all(|c| c.is_ascii_digit() || ('a'..='f').contains(&c)),
                "{name} has no valid sha256 pin: {sha:?}"
            );
            assert!(
                !subdir.is_empty() && name.ends_with(".deb"),
                "{name} has a malformed pool path {subdir:?}"
            );
        }
        assert!(
            VULKAN_DEBS
                .iter()
                .any(|(name, _, _)| name.starts_with("vulkan-validationlayers_")),
            "the validation-layers deb must be part of the pinned set"
        );
    }

    fn temp_deb(tag: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!("xtask-deb-{tag}-{}", std::process::id()));
        std::fs::write(&path, b"not really a deb").unwrap();
        path
    }

    #[test]
    fn verify_deb_accepts_the_matching_hash() {
        let path = temp_deb("ok");
        let hash = sha256_file(&path).unwrap();
        assert_eq!(verify_deb(&path, &hash), Ok(()));
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn verify_deb_rejects_a_mismatched_hash() {
        let path = temp_deb("bad");
        let err = verify_deb(&path, &"0".repeat(64)).unwrap_err();
        assert!(err.contains("sha256 MISMATCH"), "unexpected error: {err}");
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn verify_deb_reports_a_missing_file() {
        let err = verify_deb(Path::new("/nonexistent/pkg.deb"), &"0".repeat(64)).unwrap_err();
        assert!(err.contains("cannot hash"), "unexpected error: {err}");
    }

    #[test]
    fn env_exports_sets_the_three_vulkan_vars() {
        let exports = env_exports(Path::new("/prefix"));
        for var in ["LD_LIBRARY_PATH", "VK_DRIVER_FILES", "VK_LAYER_PATH"] {
            assert!(
                exports.contains(&format!("export {var}=")),
                "missing {var} in:\n{exports}"
            );
        }
        assert!(
            exports.contains("explicit_layer.d"),
            "VK_LAYER_PATH must point at the unpacked explicit_layer.d dir:\n{exports}"
        );
    }
}
