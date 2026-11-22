use std::ffi::OsString;
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
    Ci {
        /// Print the commands, and the directory they would run in, without running them
        #[arg(long)]
        dry_run: bool,
    },
    /// fmt + clippy only
    Lint {
        /// Print the commands, and the directory they would run in, without running them
        #[arg(long)]
        dry_run: bool,
    },
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
    let cli = Cli::parse();
    // Every task runs from the workspace root (below). What the user typed
    // relative to their own directory — a `--prefix`, a relative
    // `CARGO_TARGET_DIR` — is resolved against that directory first.
    let invoked_from = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    match cli.cmd {
        Task::Ci { dry_run } => {
            let runner = Runner::at_workspace_root(&invoked_from, dry_run);
            let target_dir = test_target_dir(|var| std::env::var_os(var), &invoked_from);
            lint(runner);
            test(runner, &target_dir);
        }
        Task::Lint { dry_run } => lint(Runner::at_workspace_root(&invoked_from, dry_run)),
        Task::GpuSetup {
            prefix,
            force,
            print_env,
        } => {
            let prefix = invoked_from.join(prefix.unwrap_or_else(default_prefix));
            // `spikes/vkprobe` is looked up relative to the workspace root.
            // Nothing is printed here: `--print-env` output gets `eval`ed.
            if let Some(root) = workspace_root(&invoked_from) {
                let _ = std::env::set_current_dir(root);
            }
            gpu_setup(&prefix, force, print_env)
        }
        Task::E2e | Task::Load | Task::Parity | Task::WebSmoke | Task::Accept => stub(),
    }
}

/// The workspace root: the nearest directory at or above `from` that holds the
/// workspace manifest and this crate, else the directory this xtask was built
/// from (the parent of its manifest directory, fixed at compile time), which
/// covers running the binary from outside the checkout.
fn workspace_root(from: &Path) -> Option<PathBuf> {
    let is_root =
        |dir: &&Path| dir.join("Cargo.toml").is_file() && dir.join("xtask/Cargo.toml").is_file();
    from.ancestors()
        .find(is_root)
        .or_else(|| {
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .filter(is_root)
        })
        .map(Path::to_path_buf)
}

/// Where the test build goes: `xtask-test` inside the target directory.
///
/// A *separate* directory so a running `xtask.exe` is never relinked while it
/// is still executing (Windows file locking), but inside whatever target
/// directory is configured: with `CARGO_TARGET_DIR` (or `build.target-dir` via
/// `CARGO_BUILD_TARGET_DIR`) pointing out of the checkout, a hard-coded
/// `target/xtask-test` would still put a full build tree back into it. A
/// relative value means what it means to cargo: relative to where the command
/// was run (`invoked_from`). With neither set it is `target/xtask-test` under
/// the workspace root. `env` is the environment lookup.
fn test_target_dir(env: impl Fn(&str) -> Option<OsString>, invoked_from: &Path) -> PathBuf {
    let configured = ["CARGO_TARGET_DIR", "CARGO_BUILD_TARGET_DIR"]
        .into_iter()
        .filter_map(env)
        .find(|dir| !dir.is_empty());
    let target = match configured {
        // `join` keeps an absolute path as it is.
        Some(dir) => invoked_from.join(dir),
        None => PathBuf::from("target"),
    };
    target.join("xtask-test")
}

/// Packages whose tests create a Vulkan device when one is usable. GPU tests
/// must not run in parallel (skills/space-dev/SKILL.md): on CI they skip, but
/// on a machine with a real GPU a parallel run would start many devices at
/// once. `.config/nextest.toml` puts the same packages in a `max-threads = 1`
/// test group for the nextest path.
const GPU_PACKAGES: [&str; 3] = ["space-render", "space-client", "space-mcp-client"];

fn os_args(args: &[&str]) -> Vec<OsString> {
    args.iter().map(OsString::from).collect()
}

fn lint_commands() -> Vec<Vec<OsString>> {
    vec![
        os_args(&["fmt", "--check"]),
        os_args(&[
            "clippy",
            "--workspace",
            "--all-targets",
            "--locked",
            "--",
            "-D",
            "warnings",
        ]),
    ]
}

/// `cargo nextest run` for the whole workspace. The GPU packages are
/// serialised by the `gpu` test group in `.config/nextest.toml`.
fn nextest_command(target_dir: &Path) -> Vec<OsString> {
    let mut args = os_args(&["nextest", "run", "--workspace", "--locked", "--target-dir"]);
    args.push(target_dir.into());
    args
}

/// The `cargo test` fallback, as two runs: everything except the GPU packages
/// with cargo's normal test parallelism, then the GPU packages with one test
/// thread.
fn cargo_test_commands(target_dir: &Path) -> Vec<Vec<OsString>> {
    let mut rest = os_args(&["test", "--workspace"]);
    let mut gpu = os_args(&["test"]);
    for package in GPU_PACKAGES {
        rest.extend(os_args(&["--exclude", package]));
        gpu.extend(os_args(&["-p", package]));
    }
    for args in [&mut rest, &mut gpu] {
        args.extend(os_args(&["--locked", "--target-dir"]));
        args.push(target_dir.into());
    }
    gpu.extend(os_args(&["--", "--test-threads=1"]));
    vec![rest, gpu]
}

fn lint(runner: Runner) {
    for args in lint_commands() {
        runner.cargo(&args);
    }
}

fn test(runner: Runner, target_dir: &Path) {
    if nextest_available() {
        runner.cargo(&nextest_command(target_dir));
    } else {
        eprintln!("xtask: cargo-nextest not found, falling back to `cargo test`");
        for args in cargo_test_commands(target_dir) {
            runner.cargo(&args);
        }
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

/// Runs (or, for `--dry-run`, only prints) the cargo commands of a task.
#[derive(Clone, Copy)]
struct Runner {
    dry_run: bool,
}

impl Runner {
    /// Change into the workspace root, so that `cargo fmt` checks every member
    /// (from a member's directory it checks only that member) and relative
    /// paths mean the same thing wherever xtask was started.
    fn at_workspace_root(invoked_from: &Path, dry_run: bool) -> Self {
        match workspace_root(invoked_from) {
            Some(root) => {
                if let Err(err) = std::env::set_current_dir(&root) {
                    eprintln!("xtask: cannot enter {}: {err}", root.display());
                    exit(1);
                }
                println!("xtask: workspace root: {}", root.display());
            }
            None => eprintln!(
                "xtask: WARNING: workspace root not found from {}; staying there",
                invoked_from.display()
            ),
        }
        Self { dry_run }
    }

    /// Run `cargo <args>`; exit with its status if it fails. Arguments are
    /// passed as OS strings, so a path with spaces or non-ASCII characters
    /// reaches cargo unchanged.
    fn cargo(self, args: &[OsString]) {
        let mut shown = String::from("cargo");
        for arg in args {
            shown.push(' ');
            shown.push_str(&arg.to_string_lossy());
        }
        if self.dry_run {
            println!("xtask: would run `{shown}`");
            return;
        }
        println!("xtask: running `{shown}`");
        let status = Command::new("cargo")
            .args(args)
            .status()
            .unwrap_or_else(|err| panic!("xtask: failed to spawn `cargo`: {err}"));
        if !status.success() {
            eprintln!("xtask: `{shown}` failed: {status}");
            exit(status.code().unwrap_or(1));
        }
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

fn gpu_setup(prefix: &Path, force: bool, print_env: bool) -> ! {
    let exports = env_exports(prefix);

    if print_env {
        print!("{exports}");
        exit(0);
    }

    // Before anything is created or downloaded.
    if let Some(reason) = unsupported_host(std::env::consts::OS) {
        eprintln!("xtask: {reason}");
        exit(2);
    }

    println!("xtask: Vulkan prefix: {}", prefix.display());

    if !force && vulkan_installed(prefix) {
        println!("xtask: already installed (loader + both ICD manifests present); pass --force to reinstall");
    } else {
        install_vulkan_stack(prefix);
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
    match probe_devices(prefix) {
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

/// Why `gpu-setup` cannot install anything on `os` (`std::env::consts::OS`), or
/// `None` on Linux. It unpacks Debian builds of the Vulkan loader and Mesa
/// (ELF shared objects found through `LD_LIBRARY_PATH`); elsewhere there is
/// nothing it could usefully do, so it says so instead of creating the prefix
/// and then failing on a missing `dpkg-deb`.
fn unsupported_host(os: &str) -> Option<String> {
    (os != "linux").then(|| {
        format!(
            "gpu-setup only works on Linux: it unpacks Debian packages of the Vulkan loader \
             and Mesa, which cannot be used on {os}. Nothing was installed. Here, use the \
             Vulkan driver that comes with the GPU driver (`--print-env` still prints the \
             Linux exports)."
        )
    })
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
///
/// Every value is single-quoted: the lines are `source`d / `eval`ed, and an
/// unquoted prefix with a space in it (a home directory like `/home/Jane Doe`)
/// would split into a second, invalid `export` argument.
fn env_exports(prefix: &Path) -> String {
    let prefix = prefix.display().to_string();
    let lib = sh_quote(&format!("{prefix}/usr/lib/x86_64-linux-gnu"));
    let icds = sh_quote(&format!(
        "{prefix}/usr/share/vulkan/icd.d/radeon_icd.json:{prefix}/usr/share/vulkan/icd.d/lvp_icd.json"
    ));
    let layers = sh_quote(&format!("{prefix}/usr/share/vulkan/explicit_layer.d"));
    format!(
        "export LD_LIBRARY_PATH={lib}\n\
         export VK_DRIVER_FILES={icds}\n\
         export VK_LAYER_PATH={layers}\n"
    )
}

/// `text` as one POSIX-shell word: single-quoted, with each embedded single
/// quote written as `'\''` (close the quotes, an escaped quote, reopen).
fn sh_quote(text: &str) -> String {
    format!("'{}'", text.replace('\'', r"'\''"))
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

    fn args(command: &[OsString]) -> Vec<String> {
        command
            .iter()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect()
    }

    /// The directory above this crate: the workspace root the tests run in.
    fn checkout() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("xtask lives in the workspace root")
            .to_path_buf()
    }

    /// xtask must work from any directory: `cargo fmt --check` only covers the
    /// whole workspace from its root, and a relative target dir would land in
    /// whatever subdirectory the command happened to be run from.
    #[test]
    fn the_workspace_root_is_found_from_anywhere() {
        let root = checkout();
        for from in [
            root.clone(),
            root.join("xtask"),
            root.join("xtask").join("src"),
            root.join("crates").join("space-stream").join("src"),
        ] {
            assert_eq!(workspace_root(&from), Some(root.clone()), "{from:?}");
        }
        // Outside the checkout: where this xtask was built from.
        assert_eq!(workspace_root(&std::env::temp_dir()), Some(root));
    }

    #[test]
    fn tests_build_inside_the_configured_target_dir() {
        let from = Path::new(if cfg!(windows) {
            r"C:\work\space\crates"
        } else {
            "/work/space/crates"
        });
        let absolute = if cfg!(windows) {
            r"D:\cargo target\späce"
        } else {
            "/var/cargo target/späce"
        };
        fn env<'a>(vars: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<OsString> + 'a {
            move |name| {
                let value = vars.iter().find(|(var, _)| *var == name)?.1;
                Some(OsString::from(value))
            }
        }

        // Nothing configured: the historical `target/xtask-test` (relative to
        // the workspace root xtask runs in).
        assert_eq!(
            test_target_dir(env(&[]), from),
            Path::new("target").join("xtask-test")
        );
        // CARGO_TARGET_DIR is respected, spaces and non-ASCII included.
        assert_eq!(
            test_target_dir(env(&[("CARGO_TARGET_DIR", absolute)]), from),
            Path::new(absolute).join("xtask-test")
        );
        // A relative one is relative to where the user ran the command, as it
        // is for cargo itself — not to the workspace root xtask moved to.
        assert_eq!(
            test_target_dir(env(&[("CARGO_TARGET_DIR", "out")]), from),
            from.join("out").join("xtask-test")
        );
        // `build.target-dir` through its environment form, with the plain
        // variable taking precedence and an empty value ignored.
        assert_eq!(
            test_target_dir(env(&[("CARGO_BUILD_TARGET_DIR", "b")]), from),
            from.join("b").join("xtask-test")
        );
        assert_eq!(
            test_target_dir(
                env(&[("CARGO_BUILD_TARGET_DIR", "b"), ("CARGO_TARGET_DIR", "a")]),
                from
            ),
            from.join("a").join("xtask-test")
        );
        assert_eq!(
            test_target_dir(
                env(&[("CARGO_TARGET_DIR", ""), ("CARGO_BUILD_TARGET_DIR", "b")]),
                from
            ),
            from.join("b").join("xtask-test")
        );
    }

    #[test]
    fn nextest_runs_the_whole_workspace_into_the_test_target_dir() {
        let dir = Path::new("some dir").join("xtask-test");
        let command = nextest_command(&dir);
        assert_eq!(
            args(&command),
            [
                "nextest",
                "run",
                "--workspace",
                "--locked",
                "--target-dir",
                &*dir.to_string_lossy()
            ]
        );
        // The path is one argument, not re-split on its space.
        assert_eq!(command.last().map(PathBuf::from), Some(dir));
    }

    /// Without nextest the GPU packages must not be run by a plain
    /// `cargo test --workspace` (many test threads, each creating a Vulkan
    /// device): they are excluded from the parallel run and run on their own
    /// with one test thread.
    #[test]
    fn the_cargo_test_fallback_runs_gpu_packages_single_threaded() {
        let dir = Path::new("t").join("xtask-test");
        let commands = cargo_test_commands(&dir);
        assert_eq!(commands.len(), 2);
        let [rest, gpu] = [args(&commands[0]), args(&commands[1])];

        assert_eq!(rest[..2], ["test", "--workspace"]);
        assert!(!rest.contains(&"--".to_owned()), "parallel run: {rest:?}");
        assert_eq!(gpu[0], "test");
        assert!(!gpu.contains(&"--workspace".to_owned()), "{gpu:?}");
        assert_eq!(gpu[gpu.len() - 2..], ["--", "--test-threads=1"]);
        for package in GPU_PACKAGES {
            let follows = |all: &[String], flag: &str| {
                all.windows(2).any(|w| w[0] == flag && w[1] == package)
            };
            assert!(
                follows(&rest, "--exclude"),
                "{package} not excluded: {rest:?}"
            );
            assert!(follows(&gpu, "-p"), "{package} not selected: {gpu:?}");
        }
        for command in [&rest, &gpu] {
            assert!(command.contains(&"--locked".to_owned()), "{command:?}");
            let at = command.iter().position(|a| a == "--target-dir").unwrap();
            assert_eq!(Path::new(&command[at + 1]), dir, "{command:?}");
            let test_args = command.iter().position(|a| a == "--");
            assert!(test_args.is_none_or(|dashes| at < dashes), "{command:?}");
        }
    }

    /// The nextest path serialises the same packages through a test group;
    /// the list there and `GPU_PACKAGES` here must not drift apart.
    #[test]
    fn nextest_config_serialises_the_same_gpu_packages() {
        let config = include_str!("../../.config/nextest.toml");
        let setting = |key: &str| {
            config
                .lines()
                .map(str::trim)
                .find(|line| !line.starts_with('#') && line.starts_with(key))
                .unwrap_or_else(|| panic!("no `{key}` line in .config/nextest.toml"))
        };
        assert!(setting("gpu").contains("max-threads = 1"));
        assert!(setting("test-group").contains("'gpu'"));
        let filter = setting("filter");
        assert_eq!(filter.matches("package(").count(), GPU_PACKAGES.len());
        for package in GPU_PACKAGES {
            assert!(filter.contains(&format!("package({package})")), "{filter}");
            let manifest = checkout().join("crates").join(package).join("Cargo.toml");
            assert!(manifest.is_file(), "{package} is not a workspace crate");
        }
    }

    #[test]
    fn gpu_setup_is_only_supported_on_linux() {
        assert_eq!(unsupported_host("linux"), None);
        for os in ["windows", "macos", "freebsd"] {
            let reason = unsupported_host(os).expect("unsupported");
            assert!(reason.contains("only works on Linux") && reason.contains(os));
        }
    }

    #[test]
    fn shell_words_are_single_quoted() {
        assert_eq!(sh_quote("/opt/vulkan"), "'/opt/vulkan'");
        assert_eq!(sh_quote("a b$c\"d\\e`f"), "'a b$c\"d\\e`f'");
        assert_eq!(sh_quote("it's"), r"'it'\''s'");
        assert_eq!(sh_quote(""), "''");
    }

    /// What a POSIX shell makes of `word` if it consists only of single-quoted
    /// runs and backslash-escaped characters; `None` if anything is left
    /// unquoted (the shell would split or expand it) or a quote is not closed.
    fn sh_word(word: &str) -> Option<String> {
        let mut text = String::new();
        let mut chars = word.chars();
        while let Some(c) = chars.next() {
            match c {
                '\'' => loop {
                    match chars.next()? {
                        '\'' => break,
                        quoted => text.push(quoted),
                    }
                },
                '\\' => text.push(chars.next()?),
                _ => return None,
            }
        }
        Some(text)
    }

    /// A prefix with a space (or anything else the shell would act on) must
    /// come out of `env.sh` as exactly one literal value per variable.
    #[test]
    fn env_exports_survive_a_prefix_with_spaces_and_quotes() {
        let prefix = "/home/Jane Doe/it's $HOME/vulkan";
        let exports = env_exports(Path::new(prefix));
        let lines: Vec<&str> = exports.lines().collect();
        assert_eq!(
            lines[0],
            r"export LD_LIBRARY_PATH='/home/Jane Doe/it'\''s $HOME/vulkan/usr/lib/x86_64-linux-gnu'"
        );
        let values: Vec<(&str, Option<String>)> = lines
            .iter()
            .map(|line| {
                let (name, value) = line
                    .strip_prefix("export ")
                    .and_then(|rest| rest.split_once('='))
                    .expect("export NAME=value");
                (name, sh_word(value))
            })
            .collect();
        let icd = format!("{prefix}/usr/share/vulkan/icd.d");
        assert_eq!(
            values,
            [
                (
                    "LD_LIBRARY_PATH",
                    Some(format!("{prefix}/usr/lib/x86_64-linux-gnu"))
                ),
                (
                    "VK_DRIVER_FILES",
                    Some(format!("{icd}/radeon_icd.json:{icd}/lvp_icd.json"))
                ),
                (
                    "VK_LAYER_PATH",
                    Some(format!("{prefix}/usr/share/vulkan/explicit_layer.d"))
                ),
            ]
        );
        // The unquoted form this replaces is not one word.
        assert_eq!(sh_word(&format!("{prefix}/usr")), None);
    }

    /// The real thing: source the lines in a POSIX shell and read the
    /// variables back. Always runs on Unix; elsewhere only if an `sh` is on
    /// `PATH` (Git for Windows ships one) — the test above covers the quoting
    /// itself on every platform.
    #[test]
    fn env_exports_round_trip_through_a_shell() {
        let prefix = "/tmp/Jane Doe/it's $HOME `x`/vülkan";
        let file = std::env::temp_dir().join(format!("xtask-env-{}.sh", std::process::id()));
        std::fs::write(&file, env_exports(Path::new(prefix))).unwrap();
        let output = Command::new("sh")
            .arg("-c")
            .arg(r#". "$1" && printf '%s\n%s\n%s\n' "$LD_LIBRARY_PATH" "$VK_DRIVER_FILES" "$VK_LAYER_PATH""#)
            .arg("sh")
            .arg(&file)
            .output();
        std::fs::remove_file(&file).unwrap();
        let output = match output {
            Ok(output) => output,
            Err(err) if cfg!(unix) => panic!("cannot run sh: {err}"),
            Err(err) => {
                eprintln!("SKIP: no `sh` to source env.sh with on this platform ({err})");
                return;
            }
        };
        assert!(output.status.success(), "{output:?}");
        assert_eq!(
            String::from_utf8_lossy(&output.stdout),
            format!(
                "{prefix}/usr/lib/x86_64-linux-gnu\n\
                 {prefix}/usr/share/vulkan/icd.d/radeon_icd.json:{prefix}/usr/share/vulkan/icd.d/lvp_icd.json\n\
                 {prefix}/usr/share/vulkan/explicit_layer.d\n"
            )
        );
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
