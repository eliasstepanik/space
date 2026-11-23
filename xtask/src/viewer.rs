//! `xtask viewer`: build the browser viewer and serve it on this machine.
//!
//! `crates/space-viewer` is a wasm page that shows the pixel stream of a
//! running `space-client --stream ADDR` and sends keyboard/mouse input back.
//! Getting it into a browser takes three steps, done here by one command and
//! the same way on Windows and Linux:
//!
//! 1. `cargo build` the crate for `wasm32-unknown-unknown`;
//! 2. run the `wasm-bindgen` CLI over the result (the JS glue). The CLI must be
//!    exactly the version of the `wasm-bindgen` crate in `Cargo.lock`, or the
//!    glue does not fit the wasm; xtask checks that and never fetches the tool
//!    by itself (`--install-bindgen` runs `cargo install`, nothing else);
//! 3. serve `index.html` and `pkg/` over HTTP (a browser does not load a wasm
//!    module from a `file://` page).
//!
//! The server is a development tool: plain HTTP/1.1, GET and HEAD, the regular
//! files of one directory, loopback unless asked otherwise. It is written with
//! `std::net` so that xtask needs no async runtime and no further dependency.

use std::ffi::OsString;
use std::io::{self, Read, Write};
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, Shutdown, SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{exit, Command, Stdio};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use super::{os_args, target_dir, Runner};

/// Port the page is served on unless `--port` names another.
const DEFAULT_PORT: u16 = 8097;
/// The `space-client --stream` port: where the page looks for the stream when
/// it is given no `?ws=` (`DEFAULT_STREAM_PORT` in `space-viewer`).
const STREAM_PORT: u16 = 8099;
const WASM_TARGET: &str = "wasm32-unknown-unknown";
const VIEWER_PACKAGE: &str = "space-viewer";
/// What cargo names the build output of [`VIEWER_PACKAGE`].
const VIEWER_WASM: &str = "space_viewer.wasm";
/// What `wasm-bindgen --target web` makes of it; `index.html` imports the first.
const BUNDLE_FILES: [&str; 2] = ["space_viewer.js", "space_viewer_bg.wasm"];
/// The crate in `Cargo.lock` and the executable of `wasm-bindgen-cli`.
const BINDGEN: &str = "wasm-bindgen";
/// The site directory inside the target directory.
const SITE_DIR: &str = "viewer-site";
const INDEX: &str = "index.html";

/// Shown under the options of `xtask viewer --help`.
pub const AFTER_HELP: &str = "\
The page shows the pixel stream of a running `space-client --stream ADDR` and sends
keyboard/mouse input back. Without --ws it connects to ws://<page host>:8099:

  cargo run -p space-client -- --stream 127.0.0.1:8099 --server 127.0.0.1:7777
  cargo run -p xtask -- viewer

The site is put together in <target dir>/viewer-site (index.html and pkg/). The
`wasm-bindgen` CLI is looked for on PATH, then in cargo's bin directory, and must be
the version of the `wasm-bindgen` crate in Cargo.lock; if it is not there, the command
prints the `cargo install` line to run and exits with status 2. Ctrl+C stops the server.";

#[derive(clap::Args)]
pub struct ViewerArgs {
    /// Port to serve the page on (0 picks a free one)
    #[arg(long, value_name = "N", default_value_t = DEFAULT_PORT)]
    port: u16,
    /// Address to listen on; anything but a loopback address also needs --allow-remote
    #[arg(long, value_name = "ADDR", default_value_t = IpAddr::V4(Ipv4Addr::LOCALHOST))]
    bind: IpAddr,
    /// Let --bind name a non-loopback address (plain HTTP, no authentication: a dev tool)
    #[arg(long)]
    allow_remote: bool,
    /// Stream address for the page, added to the printed link as `?ws=URL`
    /// [default: ws://<page host>:8099]
    #[arg(long, value_name = "URL", value_parser = ws_url)]
    ws: Option<String>,
    /// Do not run `cargo build`; use the space_viewer.wasm already in the target directory
    #[arg(long)]
    no_build: bool,
    /// Run `cargo install --locked wasm-bindgen-cli --version <lockfile version>` first
    /// if the CLI is missing or a different version
    #[arg(long)]
    install_bindgen: bool,
    /// Build and put the site together, print where it is, and exit without serving
    #[arg(long)]
    site_only: bool,
}

pub fn run(args: ViewerArgs, invoked_from: &Path) -> ! {
    // Refused before anything is built or installed.
    if let Err(reason) = check_bind(args.bind, args.allow_remote) {
        eprintln!("xtask: {reason}");
        exit(2);
    }
    let runner = Runner::at_workspace_root(invoked_from, false);
    // Absolute, because it is handed to cargo, to wasm-bindgen and to the
    // server. `target_dir` is relative to the workspace root (now the current
    // directory) when nothing is configured.
    let target = target_dir(|var| std::env::var_os(var), invoked_from);
    let target = std::path::absolute(&target).unwrap_or(target);

    let locked = std::fs::read_to_string("Cargo.lock")
        .map_err(|err| err.to_string())
        .and_then(|lockfile| locked_version(&lockfile, BINDGEN));
    let version = match locked {
        Ok(version) => version,
        Err(err) => fail(format!(
            "cannot tell which {BINDGEN} the viewer needs from Cargo.lock: {err}"
        )),
    };
    // Before the build: a missing tool is reported at once, not after minutes
    // of compiling.
    let bindgen = require_bindgen(&version, args.install_bindgen, runner);

    if !args.no_build {
        runner.cargo(&build_command(&target));
    }
    let site = assemble_site(&bindgen, &target);
    if args.site_only {
        println!("xtask: viewer site: {}", site.display());
        exit(0);
    }
    serve(
        &site,
        SocketAddr::new(args.bind, args.port),
        args.ws.as_deref(),
    )
}

fn fail(message: impl std::fmt::Display) -> ! {
    eprintln!("xtask: {message}");
    exit(1)
}

/// `--bind` may only name this machine itself unless `--allow-remote` says the
/// page is meant to be reachable from elsewhere.
fn check_bind(bind: IpAddr, allow_remote: bool) -> Result<(), String> {
    if allow_remote || bind.is_loopback() {
        return Ok(());
    }
    Err(format!(
        "--bind {bind} is not a loopback address. The viewer server is a development tool \
         (plain HTTP, no authentication); pass --allow-remote as well to let other machines \
         reach it."
    ))
}

/// `--ws` value: the page hands it to `new WebSocket(..)` as it is.
fn ws_url(value: &str) -> Result<String, String> {
    let rest = ["ws://", "wss://"]
        .into_iter()
        .find_map(|scheme| value.strip_prefix(scheme));
    match rest {
        Some(rest) if !rest.is_empty() => Ok(value.to_owned()),
        _ => Err("expected a WebSocket address such as ws://127.0.0.1:8099".to_owned()),
    }
}

// ---------------------------------------------------------------------------
// build + JS glue
// ---------------------------------------------------------------------------

/// `cargo build` of the viewer for wasm, the command the CI wasm job runs. The
/// target directory is passed explicitly so that the output is where
/// [`assemble_site`] looks for it (a relative `CARGO_TARGET_DIR` would mean
/// something else to a cargo started in the workspace root).
fn build_command(target_dir: &Path) -> Vec<OsString> {
    let mut args = os_args(&[
        "build",
        "--release",
        "--locked",
        "--target",
        WASM_TARGET,
        "-p",
        VIEWER_PACKAGE,
        "--target-dir",
    ]);
    args.push(target_dir.into());
    args
}

/// The version of `package` in the text of a `Cargo.lock`.
///
/// The lockfile is a list of `[[package]]` tables with one `name = ".."` and
/// one `version = ".."` line each; that much is read here, which saves xtask a
/// TOML parser. `str::lines` takes LF and CRLF alike.
fn locked_version(lockfile: &str, package: &str) -> Result<String, String> {
    let mut versions: Vec<&str> = Vec::new();
    let (mut in_package, mut name, mut version) = (false, None, None);
    // The extra header closes the last table.
    for line in lockfile.lines().map(str::trim).chain(["[end]"]) {
        if line.starts_with('[') {
            if let (true, Some(version)) = (name == Some(package), version) {
                if !versions.contains(&version) {
                    versions.push(version);
                }
            }
            in_package = line == "[[package]]";
            (name, version) = (None, None);
        } else if in_package {
            if let Some(value) = toml_string(line, "name") {
                name = Some(value);
            } else if let Some(value) = toml_string(line, "version") {
                version = Some(value);
            }
        }
    }
    match versions[..] {
        [version] => Ok(version.to_owned()),
        [] => Err(format!("no `{package}` package in it")),
        _ => Err(format!(
            "more than one version of `{package}` in it ({})",
            versions.join(", ")
        )),
    }
}

/// The value of a `key = "value"` line.
fn toml_string<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    line.strip_prefix(key)?
        .trim_start()
        .strip_prefix('=')?
        .trim()
        .strip_prefix('"')?
        .strip_suffix('"')
}

fn install_command(version: &str) -> [&str; 5] {
    [
        "install",
        "--locked",
        "wasm-bindgen-cli",
        "--version",
        version,
    ]
}

/// The `wasm-bindgen` executable of exactly `version`, or exit 2 with the
/// command that installs it. With `install` that command is run first.
fn require_bindgen(version: &str, install: bool, runner: Runner) -> PathBuf {
    let env = |var: &str| std::env::var_os(var);
    let problem = match find_bindgen(env, version) {
        Ok(exe) => return exe,
        Err(problem) => problem,
    };
    if install {
        println!("xtask: {problem}");
        runner.cargo(&os_args(&install_command(version)));
        match find_bindgen(env, version) {
            Ok(exe) => return exe,
            Err(problem) => {
                eprintln!("xtask: `cargo install` finished, but {problem}");
                exit(2);
            }
        }
    }
    eprintln!(
        "xtask: {problem}\n\
         xtask: install it with the command below, or pass --install-bindgen to have xtask run it:\n\
         cargo {}",
        install_command(version).join(" ")
    );
    exit(2)
}

fn find_bindgen(env: impl Fn(&str) -> Option<OsString>, version: &str) -> Result<PathBuf, String> {
    let found = bindgen_candidates(env)
        .into_iter()
        .map(|exe| {
            let version = bindgen_version(&exe);
            (exe, version)
        })
        .collect();
    choose_bindgen(found, version)
}

/// Every `wasm-bindgen` executable there is to choose from: the `PATH`
/// directories in order, then the `bin` directory `cargo install` writes to,
/// which need not be on `PATH`: that of `$CARGO_INSTALL_ROOT` if set, and that
/// of `$CARGO_HOME`, by default `~/.cargo`.
fn bindgen_candidates(env: impl Fn(&str) -> Option<OsString>) -> Vec<PathBuf> {
    let set = |var: &str| env(var).filter(|value| !value.is_empty());
    let mut dirs: Vec<PathBuf> = match set("PATH") {
        Some(path) => std::env::split_paths(&path).collect(),
        None => Vec::new(),
    };
    let home_vars: &[&str] = if cfg!(windows) {
        &["USERPROFILE", "HOME"]
    } else {
        &["HOME"]
    };
    let cargo_home = set("CARGO_HOME").map(PathBuf::from).or_else(|| {
        let home = home_vars.iter().find_map(|var| set(var))?;
        Some(PathBuf::from(home).join(".cargo"))
    });
    let install_roots = [set("CARGO_INSTALL_ROOT").map(PathBuf::from), cargo_home];
    dirs.extend(
        install_roots
            .into_iter()
            .flatten()
            .map(|root| root.join("bin")),
    );

    let file_name = format!("{BINDGEN}{}", std::env::consts::EXE_SUFFIX);
    let mut found = Vec::new();
    for dir in dirs {
        let exe = dir.join(&file_name);
        if !dir.as_os_str().is_empty() && exe.is_file() && !found.contains(&exe) {
            found.push(exe);
        }
    }
    found
}

/// What `<exe> --version` says, if it runs and is a wasm-bindgen.
fn bindgen_version(exe: &Path) -> Option<String> {
    let output = Command::new(exe)
        .arg("--version")
        .stdin(Stdio::null())
        .stderr(Stdio::null())
        .output()
        .ok()?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    (output.status.success())
        .then(|| parse_bindgen_version(&stdout).map(str::to_owned))
        .flatten()
}

/// The version in the output of `wasm-bindgen --version` (`wasm-bindgen 0.2.130`).
fn parse_bindgen_version(stdout: &str) -> Option<&str> {
    let mut words = stdout.lines().next()?.split_whitespace();
    (words.next()? == BINDGEN).then(|| words.next()).flatten()
}

/// The first executable whose version is `wanted`; otherwise what is wrong,
/// as the rest of a sentence. `found` pairs each executable with the version
/// it reported (`None`: it did not run).
fn choose_bindgen(found: Vec<(PathBuf, Option<String>)>, wanted: &str) -> Result<PathBuf, String> {
    let needs = format!(
        "the viewer needs the `{BINDGEN}` CLI in version {wanted} (that of the `{BINDGEN}` \
         crate in Cargo.lock)"
    );
    if found.is_empty() {
        return Err(format!(
            "{needs}; it was not found on PATH or in cargo's bin directory"
        ));
    }
    let mut others = Vec::new();
    for (exe, version) in found {
        match version {
            Some(version) if version == wanted => return Ok(exe),
            Some(version) => others.push(format!("{} is version {version}", exe.display())),
            None => others.push(format!("{} did not report a version", exe.display())),
        }
    }
    Err(format!("{needs}, but {}", others.join(", ")))
}

/// Put the site together in `<target dir>/viewer-site`: the page from the
/// crate and `pkg/` from `wasm-bindgen`. Returns the directory.
fn assemble_site(bindgen: &Path, target_dir: &Path) -> PathBuf {
    let wasm = target_dir
        .join(WASM_TARGET)
        .join("release")
        .join(VIEWER_WASM);
    if !wasm.is_file() {
        fail(format!(
            "{} does not exist; run `viewer` without --no-build to build it",
            wasm.display()
        ));
    }
    let site = target_dir.join(SITE_DIR);
    let pkg = site.join("pkg");
    if let Err(err) = std::fs::create_dir_all(&pkg) {
        fail(format!("cannot create {}: {err}", pkg.display()));
    }
    let page = Path::new("crates").join(VIEWER_PACKAGE).join(INDEX);
    if let Err(err) = std::fs::copy(&page, site.join(INDEX)) {
        fail(format!(
            "cannot copy {} to {}: {err}",
            page.display(),
            site.display()
        ));
    }

    // The same call as in crates/space-viewer/README.md. It overwrites the
    // bundle of an earlier run: the file names do not change.
    let shown = format!(
        "{} --target web --out-dir {} {}",
        bindgen.display(),
        pkg.display(),
        wasm.display()
    );
    println!("xtask: running `{shown}`");
    let status = Command::new(bindgen)
        .args(["--target", "web", "--out-dir"])
        .arg(&pkg)
        .arg(&wasm)
        .status();
    match status {
        Ok(status) if status.success() => {}
        Ok(status) => {
            eprintln!("xtask: `{shown}` failed: {status}");
            exit(status.code().unwrap_or(1));
        }
        Err(err) => fail(format!("failed to run {}: {err}", bindgen.display())),
    }
    for file in BUNDLE_FILES {
        if !pkg.join(file).is_file() {
            fail(format!(
                "{BINDGEN} did not write {}",
                pkg.join(file).display()
            ));
        }
    }
    site
}

// ---------------------------------------------------------------------------
// the link
// ---------------------------------------------------------------------------

/// The line that says what to open, and where the page looks for its stream.
fn announcement(addr: SocketAddr, ws: Option<&str>) -> String {
    let host = page_host(addr.ip());
    let page = page_url(&host, addr.port(), ws);
    match ws {
        Some(ws) => format!("viewer: {page}   (stream: {ws})"),
        None => format!(
            "viewer: {page}   (stream: ws://{host}:{STREAM_PORT} — start one with: \
             space-client --stream {host}:{STREAM_PORT} --server <shard>)"
        ),
    }
}

/// The host to put in a link to a server listening on `ip`: the address
/// itself (an IPv6 one in brackets), and loopback for "every address".
fn page_host(ip: IpAddr) -> String {
    match ip {
        IpAddr::V4(ip) if ip.is_unspecified() => Ipv4Addr::LOCALHOST.to_string(),
        IpAddr::V4(ip) => ip.to_string(),
        IpAddr::V6(ip) if ip.is_unspecified() => format!("[{}]", Ipv6Addr::LOCALHOST),
        IpAddr::V6(ip) => format!("[{ip}]"),
    }
}

/// The page's address; with a stream address, as the `?ws=` the page reads.
fn page_url(host: &str, port: u16, ws: Option<&str>) -> String {
    match ws {
        Some(ws) => format!("http://{host}:{port}/?ws={}", percent_encode(ws)),
        None => format!("http://{host}:{port}/"),
    }
}

/// `text` as a URL query value: everything but the unreserved characters of
/// RFC 3986 is `%XX`-escaped, as JavaScript's `encodeURIComponent` would.
fn percent_encode(text: &str) -> String {
    const HEX: &[u8; 16] = b"0123456789ABCDEF";
    let mut encoded = String::with_capacity(text.len());
    for byte in text.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.' | b'_' | b'~') {
            encoded.push(char::from(byte));
        } else {
            encoded.push('%');
            encoded.push(char::from(HEX[usize::from(byte >> 4)]));
            encoded.push(char::from(HEX[usize::from(byte & 0xf)]));
        }
    }
    encoded
}

/// The reverse, for a request path. `None` for a `%` without two hex digits
/// and for bytes that are not UTF-8.
fn percent_decode(text: &str) -> Option<String> {
    let hex = |byte: u8| {
        char::from(byte)
            .to_digit(16)
            .and_then(|d| u8::try_from(d).ok())
    };
    let mut decoded = Vec::with_capacity(text.len());
    let mut bytes = text.bytes();
    while let Some(byte) = bytes.next() {
        if byte == b'%' {
            let (high, low) = (hex(bytes.next()?)?, hex(bytes.next()?)?);
            decoded.push(high << 4 | low);
        } else {
            decoded.push(byte);
        }
    }
    String::from_utf8(decoded).ok()
}

// ---------------------------------------------------------------------------
// the HTTP server
// ---------------------------------------------------------------------------

/// A request head (request line and header fields) may be this long.
const HEAD_LIMIT: usize = 8 * 1024;
/// A client has this long to send its request head, in total.
const HEAD_TIMEOUT: Duration = Duration::from_secs(5);
/// One write of the response may block this long on a client that is not reading.
const WRITE_TIMEOUT: Duration = Duration::from_secs(30);
/// After the response, what the client still sends is read and dropped for at
/// most this long and this many bytes (see [`handle`]).
const DRAIN_TIMEOUT: Duration = Duration::from_millis(500);
const DRAIN_LIMIT: usize = 64 * 1024;
/// Connections served at the same time; further ones wait in the listen queue.
/// A browser opens at most six to one host.
const WORKERS: usize = 16;

/// What is served, and whether each request is reported on stdout.
struct Site {
    /// Canonical path of the site directory.
    root: PathBuf,
    log: bool,
}

/// A line on stdout while serving. Not `println!`, which panics once stdout is
/// gone, and it is gone after `xtask viewer | grep -m1 viewer:`; the page is
/// served all the same.
fn say(line: impl std::fmt::Display) {
    let _ = writeln!(io::stdout(), "{line}");
}

fn serve(site: &Path, addr: SocketAddr, ws: Option<&str>) -> ! {
    let root = match site.canonicalize() {
        Ok(root) => root,
        Err(err) => fail(format!("cannot open {}: {err}", site.display())),
    };
    let listener = match TcpListener::bind(addr) {
        Ok(listener) => listener,
        Err(err) => fail(format!(
            "cannot listen on {addr}: {err} (is a viewer already running there? \
             --port picks another port)"
        )),
    };
    // With `--port 0` this is where the port is learnt.
    let addr = listener.local_addr().unwrap_or(addr);
    let workers = match spawn_workers(listener, Site { root, log: true }) {
        Ok(workers) => workers,
        Err(err) => fail(format!("cannot start the server threads: {err}")),
    };
    say(format_args!(
        "xtask: serving {} (Ctrl+C stops it)",
        site.display()
    ));
    if !addr.ip().is_loopback() {
        say(format_args!(
            "xtask: listening on {addr}: other machines can load the page (plain HTTP, no \
             authentication)"
        ));
    }
    // Last, and only once the server answers: a script can wait for this line.
    say(announcement(addr, ws));
    for worker in workers {
        let _ = worker.join();
    }
    exit(1)
}

/// Start the threads that serve `site` on `listener`; they run until the
/// process ends. Each takes one connection at a time, so [`WORKERS`] is the
/// cap on connections served at once.
fn spawn_workers(listener: TcpListener, site: Site) -> io::Result<Vec<JoinHandle<()>>> {
    let shared = Arc::new((listener, site));
    (0..WORKERS)
        .map(|n| {
            let shared = Arc::clone(&shared);
            std::thread::Builder::new()
                .name(format!("viewer-http-{n}"))
                .spawn(move || loop {
                    let (listener, site) = &*shared;
                    match listener.accept() {
                        Ok((stream, _)) => {
                            // `handle` has nothing in it that panics on input;
                            // should that ever be wrong, the thread stays.
                            let handled =
                                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                    handle(stream, site)
                                }));
                            if handled.is_err() {
                                let _ = writeln!(io::stderr(), "xtask: viewer: a request panicked");
                            }
                        }
                        // Out of sockets, or the client was already gone.
                        Err(_) => std::thread::sleep(Duration::from_millis(50)),
                    }
                })
        })
        .collect()
}

/// One connection: one request, one response, close.
fn handle(mut stream: TcpStream, site: &Site) {
    let _ = stream.set_write_timeout(Some(WRITE_TIMEOUT));
    let deadline = Instant::now() + HEAD_TIMEOUT;
    let head = read_head(&mut stream, |stream| {
        let left = deadline.saturating_duration_since(Instant::now());
        !left.is_zero() && stream.set_read_timeout(Some(left)).is_ok()
    });
    let (request, response) = match head {
        Ok(head) => (shown_request(&head), respond(&head, &site.root)),
        // Nothing was asked (a port probe, a browser's spare connection).
        Err(HeadError::Silent) => return,
        Err(HeadError::Reply(status)) => ("(incomplete request)".to_owned(), status.into()),
    };
    if site.log {
        let code = response.status.code();
        say(format_args!("xtask: viewer: {request} -> {code}"));
    }
    if stream.write_all(&response.to_bytes()).is_err() {
        return;
    }
    // Closing a socket that still has unread input (the rest of an oversized
    // head, the body of a POST) resets the connection, and the client may
    // lose the response over it. So: end our side, then read for a moment.
    let _ = stream.shutdown(Shutdown::Write);
    let deadline = Instant::now() + DRAIN_TIMEOUT;
    let mut buf = [0u8; 4096];
    let mut dropped = 0;
    while dropped < DRAIN_LIMIT {
        let left = deadline.saturating_duration_since(Instant::now());
        if left.is_zero() || stream.set_read_timeout(Some(left)).is_err() {
            break;
        }
        match stream.read(&mut buf) {
            Ok(n) if n > 0 => dropped += n,
            _ => break,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Status {
    Ok,
    BadRequest,
    NotFound,
    MethodNotAllowed,
    RequestTimeout,
    HeadTooLarge,
    VersionNotSupported,
}

impl Status {
    fn code(self) -> u16 {
        match self {
            Status::Ok => 200,
            Status::BadRequest => 400,
            Status::NotFound => 404,
            Status::MethodNotAllowed => 405,
            Status::RequestTimeout => 408,
            Status::HeadTooLarge => 431,
            Status::VersionNotSupported => 505,
        }
    }

    fn reason(self) -> &'static str {
        match self {
            Status::Ok => "OK",
            Status::BadRequest => "Bad Request",
            Status::NotFound => "Not Found",
            Status::MethodNotAllowed => "Method Not Allowed",
            Status::RequestTimeout => "Request Timeout",
            Status::HeadTooLarge => "Request Header Fields Too Large",
            Status::VersionNotSupported => "HTTP Version Not Supported",
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
enum HeadError {
    /// The client sent nothing: close without a response.
    Silent,
    /// The head is unusable: say why and close.
    Reply(Status),
}

/// Read a request head: everything up to the blank line that ends it, at most
/// [`HEAD_LIMIT`] bytes. `arm` runs before every read and returns `false` once
/// the client is out of time (for a socket it sets the read timeout to what is
/// left).
fn read_head<R: Read>(
    reader: &mut R,
    mut arm: impl FnMut(&mut R) -> bool,
) -> Result<Vec<u8>, HeadError> {
    let mut head = Vec::new();
    let mut buf = [0u8; 1024];
    loop {
        if let Some(end) = head_end(&head) {
            head.truncate(end);
            return Ok(head);
        }
        if head.len() >= HEAD_LIMIT {
            return Err(HeadError::Reply(Status::HeadTooLarge));
        }
        let give_up = |status| match head.is_empty() {
            true => HeadError::Silent,
            false => HeadError::Reply(status),
        };
        if !arm(reader) {
            return Err(give_up(Status::RequestTimeout));
        }
        let room = buf.len().min(HEAD_LIMIT - head.len());
        match reader.read(&mut buf[..room]) {
            Ok(0) => return Err(give_up(Status::BadRequest)),
            Ok(n) => head.extend_from_slice(&buf[..n]),
            Err(err) => match err.kind() {
                io::ErrorKind::Interrupted => {}
                io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut => {
                    return Err(give_up(Status::RequestTimeout))
                }
                _ => return Err(HeadError::Silent),
            },
        }
    }
}

/// Where the blank line that ends a request head starts (CRLF line ends, or
/// bare LF ones).
fn head_end(head: &[u8]) -> Option<usize> {
    let find = |end: &[u8]| head.windows(end.len()).position(|window| window == end);
    match (find(b"\r\n\r\n"), find(b"\n\n")) {
        (Some(crlf), Some(lf)) => Some(crlf.min(lf)),
        (crlf, lf) => crlf.or(lf),
    }
}

/// The first line of a request head.
fn request_line(head: &[u8]) -> Result<&str, Status> {
    let line = head.split(|&byte| byte == b'\n').next().unwrap_or(head);
    let line = line.strip_suffix(b"\r").unwrap_or(line);
    std::str::from_utf8(line).map_err(|_| Status::BadRequest)
}

/// The request line for the log: cut short, and with nothing in it that a
/// terminal would act on.
fn shown_request(head: &[u8]) -> String {
    match request_line(head) {
        Ok(line) => line
            .chars()
            .take(120)
            .flat_map(char::escape_default)
            .collect(),
        Err(_) => "(unreadable request)".to_owned(),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Method {
    Get,
    Head,
}

/// `GET /path HTTP/1.1` -> the method and the request target. The target is
/// only known to start with `/` here; [`site_path`] decides what it names.
fn parse_request_line(line: &str) -> Result<(Method, &str), Status> {
    let mut parts = line.split(' ');
    let (Some(method), Some(target), Some(version), None) =
        (parts.next(), parts.next(), parts.next(), parts.next())
    else {
        return Err(Status::BadRequest);
    };
    if !matches!(version, "HTTP/1.1" | "HTTP/1.0") {
        return Err(match version.strip_prefix("HTTP/") {
            Some(number) if number.starts_with(|c: char| c.is_ascii_digit()) => {
                Status::VersionNotSupported
            }
            _ => Status::BadRequest,
        });
    }
    // A method is a token; the two that are served are spelt in capitals.
    let is_token = |text: &str| {
        !text.is_empty()
            && text
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"!#$%&'*+-.^_`|~".contains(&b))
    };
    let method = match method {
        "GET" => Method::Get,
        "HEAD" => Method::Head,
        other if is_token(other) => return Err(Status::MethodNotAllowed),
        _ => return Err(Status::BadRequest),
    };
    // Only the origin form (`/path?query`): not `http://host/path`, not `*`.
    if !target.starts_with('/') {
        return Err(Status::BadRequest);
    }
    Ok((method, target))
}

/// The file a request target names, as a path relative to the site directory.
///
/// Only plain names get through: after percent-decoding, every `/`-separated
/// part must consist of ASCII letters, digits, `-`, `_` and `.`, and neither
/// start nor end with a dot. That leaves no way to write a parent directory
/// (`..`, also as `%2e%2e`), an absolute path (`//etc`, `/C:/`), a backslash,
/// a NUL, a Windows drive, stream (`a.js::$DATA`) or short name (`SPACE_~1.JS`).
/// What a browser would never send is a bad request (400); other names simply
/// do not exist here (404). `/` is the page.
fn site_path(target: &str) -> Result<PathBuf, Status> {
    let path = target.strip_prefix('/').ok_or(Status::BadRequest)?;
    let path = path.split(['?', '#']).next().unwrap_or_default();
    let path = percent_decode(path).ok_or(Status::BadRequest)?;
    if path.is_empty() {
        return Ok(PathBuf::from(INDEX));
    }
    if path
        .chars()
        .any(|c| c.is_control() || matches!(c, '\\' | ':'))
    {
        return Err(Status::BadRequest);
    }
    let mut file = PathBuf::new();
    for part in path.split('/') {
        if matches!(part, "." | "..") {
            return Err(Status::BadRequest);
        }
        let plain = !part.is_empty()
            && !part.starts_with('.')
            && !part.ends_with('.')
            && part
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.'));
        if !plain {
            return Err(Status::NotFound);
        }
        // One part at a time, so that the separator is the platform's.
        file.push(part);
    }
    Ok(file)
}

/// The bytes of the regular file `relative` names inside `root` (a canonical
/// path). Canonicalising follows links, so one that leaves the site directory
/// is not served.
fn read_site_file(root: &Path, relative: &Path) -> Option<Vec<u8>> {
    let path = root.join(relative).canonicalize().ok()?;
    if !path.starts_with(root) {
        return None;
    }
    let mut file = std::fs::File::open(&path).ok()?;
    if !file.metadata().ok()?.is_file() {
        return None;
    }
    // Read whole, so that `Content-Length` is right even if a rebuild replaces
    // the file meanwhile. The bundle is a few hundred kilobytes.
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).ok()?;
    Some(bytes)
}

fn content_type(file: &Path) -> &'static str {
    let extension = file
        .extension()
        .and_then(|extension| extension.to_str())
        .map(str::to_ascii_lowercase);
    match extension.as_deref() {
        Some("html") => "text/html; charset=utf-8",
        Some("js") => "text/javascript; charset=utf-8",
        // Exactly this, or `WebAssembly.instantiateStreaming` refuses it.
        Some("wasm") => "application/wasm",
        Some("css") => "text/css; charset=utf-8",
        Some("json") => "application/json",
        // wasm-bindgen's `.d.ts` files.
        Some("ts" | "txt") => "text/plain; charset=utf-8",
        _ => "application/octet-stream",
    }
}

struct Response {
    status: Status,
    content_type: &'static str,
    body: Vec<u8>,
    /// The answer to a HEAD request: the header fields without the body.
    head_only: bool,
}

impl From<Status> for Response {
    /// An error as a one-line text page.
    fn from(status: Status) -> Self {
        Response {
            status,
            content_type: "text/plain; charset=utf-8",
            body: format!("{} {}\n", status.code(), status.reason()).into_bytes(),
            head_only: false,
        }
    }
}

impl Response {
    /// The response as it goes over the wire, in one piece (two small writes
    /// would wait on each other: Nagle's algorithm and delayed ACKs).
    ///
    /// `Cache-Control: no-cache` makes the browser ask again every time: the
    /// bundle keeps its file names across rebuilds, and a cached copy would
    /// keep an old viewer running.
    fn to_bytes(&self) -> Vec<u8> {
        let mut head = format!(
            "HTTP/1.1 {} {}\r\n\
             Content-Type: {}\r\n\
             Content-Length: {}\r\n\
             Cache-Control: no-cache\r\n\
             X-Content-Type-Options: nosniff\r\n\
             Connection: close\r\n",
            self.status.code(),
            self.status.reason(),
            self.content_type,
            self.body.len()
        );
        if self.status == Status::MethodNotAllowed {
            head.push_str("Allow: GET, HEAD\r\n");
        }
        head.push_str("\r\n");
        let mut bytes = head.into_bytes();
        if !self.head_only {
            bytes.extend_from_slice(&self.body);
        }
        bytes
    }
}

/// The response to a request head, serving the files in `root`.
fn respond(head: &[u8], root: &Path) -> Response {
    let (method, target) = match request_line(head).and_then(parse_request_line) {
        Ok(request) => request,
        Err(status) => return status.into(),
    };
    let file = site_path(target).and_then(|file| {
        let body = read_site_file(root, &file).ok_or(Status::NotFound)?;
        Ok((content_type(&file), body))
    });
    let mut response = match file {
        Ok((content_type, body)) => Response {
            status: Status::Ok,
            content_type,
            body,
            head_only: false,
        },
        Err(status) => status.into(),
    };
    response.head_only = method == Method::Head;
    response
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The directory above this crate: the workspace root.
    fn checkout() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("xtask lives in the workspace root")
            .to_path_buf()
    }

    /// A fresh, empty directory for one test.
    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("xtask-viewer-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    // -- Cargo.lock ---------------------------------------------------------

    const LOCKFILE: &str = r#"# This file is automatically @generated by Cargo.
# It is not intended for manual editing.
version = 4

[[package]]
name = "walkdir"
version = "2.5.0"
source = "registry+https://github.com/rust-lang/crates.io-index"

[[package]]
name = "wasm-bindgen"
version = "0.2.130"
source = "registry+https://github.com/rust-lang/crates.io-index"
checksum = "a0181c14ddd6c7b8b0b0132a5f1ea2f32be0d893cd3ea97b6e3b28886b8a3d1c"
dependencies = [
 "cfg-if",
 "wasm-bindgen-macro",
]

[[package]]
name = "wasm-bindgen-futures"
version = "0.4.80"
dependencies = [
 "wasm-bindgen",
]

[[package]]
name = "web-sys"
version = "0.3.107"
dependencies = [
 "wasm-bindgen",
]
"#;

    #[test]
    fn the_bindgen_version_is_read_from_a_lockfile_with_lf_or_crlf() {
        let lf = LOCKFILE.replace("\r\n", "\n");
        let crlf = lf.replace('\n', "\r\n");
        assert!(!lf.contains('\r') && crlf.contains("\"0.2.130\"\r\n"));
        for lockfile in [&lf, &crlf] {
            assert_eq!(
                locked_version(lockfile, "wasm-bindgen").as_deref(),
                Ok("0.2.130")
            );
            // Not a prefix match, and not the file's own `version = 4`.
            assert_eq!(
                locked_version(lockfile, "wasm-bindgen-futures").as_deref(),
                Ok("0.4.80")
            );
            assert_eq!(
                locked_version(lockfile, "web-sys").as_deref(),
                Ok("0.3.107")
            );
            assert!(locked_version(lockfile, "wasm").is_err());
            assert!(locked_version(lockfile, "cfg-if").is_err());
        }
    }

    #[test]
    fn a_lockfile_without_or_with_two_versions_is_an_error() {
        let err = locked_version("version = 4\n", "wasm-bindgen").unwrap_err();
        assert!(err.contains("no `wasm-bindgen` package"), "{err}");
        assert!(locked_version("", "wasm-bindgen").is_err());

        let entry = |version: &str| {
            format!("[[package]]\nname = \"wasm-bindgen\"\nversion = \"{version}\"\n\n")
        };
        let two = format!("{}{}", entry("0.2.130"), entry("0.2.92"));
        let err = locked_version(&two, "wasm-bindgen").unwrap_err();
        assert!(err.contains("0.2.130") && err.contains("0.2.92"), "{err}");
        // The same version twice (two sources) is one version.
        let twice = format!("{}{}", entry("0.2.130"), entry("0.2.130"));
        assert_eq!(
            locked_version(&twice, "wasm-bindgen").as_deref(),
            Ok("0.2.130")
        );
        // The last table counts although nothing follows it.
        assert_eq!(
            locked_version(
                "[[package]]\nname = \"wasm-bindgen\"\nversion = \"1.2.3\"",
                "wasm-bindgen"
            )
            .as_deref(),
            Ok("1.2.3")
        );
        // Not fooled by a name in another kind of table.
        assert!(locked_version(
            "[metadata]\nname = \"wasm-bindgen\"\nversion = \"9.9.9\"\n",
            "wasm-bindgen"
        )
        .is_err());
    }

    /// The real lockfile has the crate, and the real page loads the bundle
    /// from where `assemble_site` puts it.
    #[test]
    fn the_checkout_matches_what_the_viewer_command_expects() {
        let lockfile = std::fs::read_to_string(checkout().join("Cargo.lock")).unwrap();
        let version = locked_version(&lockfile, BINDGEN).unwrap();
        let numbers: Vec<&str> = version.split('.').collect();
        assert!(
            numbers.len() == 3
                && numbers
                    .iter()
                    .all(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit())),
            "{version}"
        );

        let crate_dir = checkout().join("crates").join(VIEWER_PACKAGE);
        let page = std::fs::read_to_string(crate_dir.join(INDEX)).unwrap();
        assert!(page.contains(&format!("\"./pkg/{}\"", BUNDLE_FILES[0])));
        assert_eq!(content_type(Path::new(BUNDLE_FILES[1])), "application/wasm");
        assert_eq!(
            VIEWER_WASM,
            format!("{}.wasm", VIEWER_PACKAGE.replace('-', "_"))
        );
    }

    // -- wasm-bindgen CLI ---------------------------------------------------

    #[test]
    fn the_build_is_the_locked_release_wasm_build_into_the_target_dir() {
        let dir = Path::new("some dir").join("tärget");
        let args: Vec<String> = build_command(&dir)
            .iter()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect();
        assert_eq!(
            args,
            [
                "build",
                "--release",
                "--locked",
                "--target",
                "wasm32-unknown-unknown",
                "-p",
                "space-viewer",
                "--target-dir",
                &*dir.to_string_lossy()
            ]
        );
    }

    #[test]
    fn the_install_command_names_the_lockfile_version() {
        assert_eq!(
            format!("cargo {}", install_command("0.2.130").join(" ")),
            "cargo install --locked wasm-bindgen-cli --version 0.2.130"
        );
    }

    #[test]
    fn the_cli_version_is_the_second_word() {
        assert_eq!(
            parse_bindgen_version("wasm-bindgen 0.2.130\n"),
            Some("0.2.130")
        );
        assert_eq!(
            parse_bindgen_version("wasm-bindgen 0.2.92 (2bc5c8bf0)\r\n"),
            Some("0.2.92")
        );
        assert_eq!(parse_bindgen_version("wasm-pack 0.13.1\n"), None);
        assert_eq!(parse_bindgen_version("wasm-bindgen\n"), None);
        assert_eq!(parse_bindgen_version(""), None);
    }

    #[test]
    fn only_a_cli_of_the_locked_version_is_used() {
        let found = |versions: &[(&str, Option<&str>)]| -> Vec<(PathBuf, Option<String>)> {
            versions
                .iter()
                .map(|(exe, version)| (PathBuf::from(exe), version.map(str::to_owned)))
                .collect()
        };
        // The first one that fits, wherever it is in the search order.
        assert_eq!(
            choose_bindgen(found(&[("a", Some("0.2.130"))]), "0.2.130"),
            Ok(PathBuf::from("a"))
        );
        assert_eq!(
            choose_bindgen(
                found(&[
                    ("old", Some("0.2.92")),
                    ("broken", None),
                    ("b", Some("0.2.130")),
                    ("c", Some("0.2.130"))
                ]),
                "0.2.130"
            ),
            Ok(PathBuf::from("b"))
        );
        let missing = choose_bindgen(Vec::new(), "0.2.130").unwrap_err();
        assert!(
            missing.contains("0.2.130") && missing.contains("not found"),
            "{missing}"
        );
        let wrong =
            choose_bindgen(found(&[("old", Some("0.2.92")), ("x", None)]), "0.2.130").unwrap_err();
        assert!(
            wrong.contains("version 0.2.130") && wrong.contains("old is version 0.2.92"),
            "{wrong}"
        );
        // Close is not equal.
        assert!(choose_bindgen(found(&[("a", Some("0.2.13"))]), "0.2.130").is_err());
    }

    #[test]
    fn the_cli_is_looked_for_on_path_then_in_cargos_bin_directory() {
        let dir = temp_dir("candidates");
        let name = format!("{BINDGEN}{}", std::env::consts::EXE_SUFFIX);
        let on_path = dir.join("tools dir");
        let home = dir.join("home");
        let cargo_home = dir.join("cärgo");
        let install_root = dir.join("install root");
        for bin in [
            on_path.clone(),
            home.join(".cargo").join("bin"),
            cargo_home.join("bin"),
            install_root.join("bin"),
        ] {
            std::fs::create_dir_all(&bin).unwrap();
            std::fs::write(bin.join(&name), b"").unwrap();
        }
        let path = std::env::join_paths([dir.join("empty"), on_path.clone(), on_path.clone()])
            .expect("PATH");
        let env = |vars: Vec<(&'static str, OsString)>| {
            move |name: &str| {
                let value = vars.iter().find(|(var, _)| *var == name)?;
                Some(value.1.clone())
            }
        };

        // PATH order first (a directory listed twice counts once), then
        // `~/.cargo/bin` although it is not on PATH.
        let vars = vec![
            ("PATH", path.clone()),
            ("HOME", home.clone().into()),
            ("USERPROFILE", home.clone().into()),
        ];
        assert_eq!(
            bindgen_candidates(env(vars)),
            [
                on_path.join(&name),
                home.join(".cargo").join("bin").join(&name)
            ]
        );
        // `CARGO_HOME` is where cargo installs to when it is set, and
        // `CARGO_INSTALL_ROOT` goes before it.
        let mut vars = vec![
            ("PATH", path),
            ("HOME", home.clone().into()),
            ("USERPROFILE", home.clone().into()),
            ("CARGO_HOME", cargo_home.clone().into()),
        ];
        assert_eq!(
            bindgen_candidates(env(vars.clone())),
            [on_path.join(&name), cargo_home.join("bin").join(&name)]
        );
        vars.push(("CARGO_INSTALL_ROOT", install_root.clone().into()));
        assert_eq!(
            bindgen_candidates(env(vars)),
            [
                on_path.join(&name),
                install_root.join("bin").join(&name),
                cargo_home.join("bin").join(&name)
            ]
        );
        // Nothing set, nothing found (and no lookup in the current directory).
        assert!(bindgen_candidates(env(Vec::new())).is_empty());
        assert!(bindgen_candidates(env(vec![("PATH", OsString::new())])).is_empty());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    // -- arguments and the link ---------------------------------------------

    #[test]
    fn a_non_loopback_bind_needs_allow_remote() {
        let ip = |text: &str| text.parse::<IpAddr>().unwrap();
        for loopback in ["127.0.0.1", "127.8.9.10", "::1"] {
            assert_eq!(check_bind(ip(loopback), false), Ok(()), "{loopback}");
        }
        for remote in ["0.0.0.0", "192.168.1.20", "10.0.0.1", "::", "fe80::1"] {
            let reason = check_bind(ip(remote), false).unwrap_err();
            assert!(reason.contains("--allow-remote"), "{reason}");
            assert_eq!(check_bind(ip(remote), true), Ok(()), "{remote}");
        }
    }

    #[test]
    fn the_stream_address_must_be_a_websocket_url() {
        for good in ["ws://127.0.0.1:8099", "wss://example.org/stream"] {
            assert_eq!(ws_url(good).as_deref(), Ok(good));
        }
        for bad in [
            "",
            "127.0.0.1:8099",
            "http://127.0.0.1:8099",
            "ws://",
            "ws:/x",
        ] {
            assert!(ws_url(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn the_ws_address_is_percent_encoded_into_the_link() {
        assert_eq!(page_url("127.0.0.1", 8098, None), "http://127.0.0.1:8098/");
        assert_eq!(
            page_url("127.0.0.1", 8098, Some("ws://127.0.0.1:8099")),
            "http://127.0.0.1:8098/?ws=ws%3A%2F%2F127.0.0.1%3A8099"
        );
        assert_eq!(
            page_url("[::1]", 18098, Some("ws://[::1]:8099")),
            "http://[::1]:18098/?ws=ws%3A%2F%2F%5B%3A%3A1%5D%3A8099"
        );
        // Nothing of the address can end the parameter or start another one.
        let odd = "ws://hö st/a?b=c&d#e+f%20";
        let encoded = percent_encode(odd);
        assert_eq!(
            encoded,
            "ws%3A%2F%2Fh%C3%B6%20st%2Fa%3Fb%3Dc%26d%23e%2Bf%2520"
        );
        assert!(encoded
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"-._~%".contains(&b)));
        // ... and the page gets back what was passed.
        assert_eq!(percent_decode(&encoded).as_deref(), Some(odd));
        assert_eq!(percent_encode("AZaz09-._~"), "AZaz09-._~");
    }

    #[test]
    fn the_announcement_says_what_to_open_and_where_the_stream_is_expected() {
        let addr = |text: &str| text.parse::<SocketAddr>().unwrap();
        assert_eq!(
            announcement(addr("127.0.0.1:8098"), None),
            "viewer: http://127.0.0.1:8098/   (stream: ws://127.0.0.1:8099 — start one with: \
             space-client --stream 127.0.0.1:8099 --server <shard>)"
        );
        assert_eq!(
            announcement(addr("127.0.0.1:18098"), Some("ws://10.0.0.5:9000")),
            "viewer: http://127.0.0.1:18098/?ws=ws%3A%2F%2F10.0.0.5%3A9000   \
             (stream: ws://10.0.0.5:9000)"
        );
        // IPv6 in brackets; "every address" is opened through loopback.
        assert!(announcement(addr("[::1]:8098"), None)
            .starts_with("viewer: http://[::1]:8098/   (stream: ws://[::1]:8099 "));
        assert!(
            announcement(addr("0.0.0.0:8098"), None).starts_with("viewer: http://127.0.0.1:8098/ ")
        );
        assert!(announcement(addr("[::]:8098"), None).starts_with("viewer: http://[::1]:8098/ "));
        assert!(announcement(addr("192.168.1.20:8098"), None)
            .contains("space-client --stream 192.168.1.20:8099 "));
    }

    // -- request parsing ----------------------------------------------------

    #[test]
    fn request_lines_are_parsed_strictly() {
        assert_eq!(parse_request_line("GET / HTTP/1.1"), Ok((Method::Get, "/")));
        assert_eq!(
            parse_request_line("HEAD /pkg/space_viewer.js?x=1 HTTP/1.0"),
            Ok((Method::Head, "/pkg/space_viewer.js?x=1"))
        );
        for other in [
            "POST", "PUT", "DELETE", "OPTIONS", "PATCH", "CONNECT", "get", "M-SEARCH",
        ] {
            assert_eq!(
                parse_request_line(&format!("{other} / HTTP/1.1")),
                Err(Status::MethodNotAllowed),
                "{other}"
            );
        }
        for bad in [
            "",
            "GET",
            "GET /",
            "GET  / HTTP/1.1",
            "GET / HTTP/1.1 ",
            " GET / HTTP/1.1",
            "GET / extra HTTP/1.1",
            "GET\t/\tHTTP/1.1",
            "G\u{0}T / HTTP/1.1",
            "GÉT / HTTP/1.1",
            "(GET) / HTTP/1.1",
            "GET / HTTP",
            "GET / HTTP/x",
            "GET / FTP/1.1",
            "GET / http/1.1",
            // Not the origin form.
            "GET index.html HTTP/1.1",
            "GET http://127.0.0.1:8098/ HTTP/1.1",
            "GET * HTTP/1.1",
            "GET ../secret HTTP/1.1",
        ] {
            assert_eq!(parse_request_line(bad), Err(Status::BadRequest), "{bad:?}");
        }
        for version in ["HTTP/2.0", "HTTP/0.9", "HTTP/1.2", "HTTP/3"] {
            assert_eq!(
                parse_request_line(&format!("GET / {version}")),
                Err(Status::VersionNotSupported),
                "{version}"
            );
        }
    }

    #[test]
    fn the_request_line_is_the_first_line_of_the_head() {
        assert_eq!(
            request_line(b"GET / HTTP/1.1\r\nHost: x\r\nAccept: */*"),
            Ok("GET / HTTP/1.1")
        );
        assert_eq!(
            request_line(b"GET / HTTP/1.1\nHost: x"),
            Ok("GET / HTTP/1.1")
        );
        assert_eq!(request_line(b"GET / HTTP/1.1"), Ok("GET / HTTP/1.1"));
        assert_eq!(request_line(b""), Ok(""));
        assert_eq!(
            request_line(b"GET /\xff HTTP/1.1\r\n"),
            Err(Status::BadRequest)
        );
        // Nothing a terminal would act on reaches the log, and not much of it.
        assert_eq!(
            shown_request(b"GET /\x1b[2J\x07 HTTP/1.1\r\nHost: x"),
            "GET /\\u{1b}[2J\\u{7} HTTP/1.1"
        );
        assert_eq!(shown_request(b"\xff\xfe"), "(unreadable request)");
        assert_eq!(shown_request(&[b'a'; 5000]).len(), 120);
    }

    /// Hands out its bytes a few at a time, like a slow client.
    struct Trickle<'a>(&'a [u8], usize);

    impl Read for Trickle<'_> {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            let n = self.1.min(buf.len()).min(self.0.len());
            buf[..n].copy_from_slice(&self.0[..n]);
            self.0 = &self.0[n..];
            Ok(n)
        }
    }

    #[test]
    fn the_request_head_is_read_up_to_the_blank_line_and_no_further() {
        let read = |bytes: &[u8]| read_head(&mut Trickle(bytes, 7), |_| true);
        assert_eq!(
            read(b"GET / HTTP/1.1\r\nHost: x\r\n\r\nbody that is not read"),
            Ok(b"GET / HTTP/1.1\r\nHost: x".to_vec())
        );
        assert_eq!(read(b"GET / HTTP/1.1\n\n"), Ok(b"GET / HTTP/1.1".to_vec()));
        assert_eq!(
            read(b"GET / HTTP/1.1\r\n\r\n"),
            Ok(b"GET / HTTP/1.1".to_vec())
        );
        assert_eq!(head_end(b"a\r\nb\r\n"), None);
        assert_eq!(head_end(b"a\n\nb\r\n\r\n"), Some(1));
        assert_eq!(head_end(b"a\r\n\r\nb\n\n"), Some(1));

        // The client went away: mid-request that is a bad request, before the
        // first byte it is nothing at all.
        assert_eq!(
            read(b"GET / HTTP/1.1\r\nHost: x\r\n"),
            Err(HeadError::Reply(Status::BadRequest))
        );
        assert_eq!(read(b""), Err(HeadError::Silent));
    }

    #[test]
    fn the_request_head_is_capped_in_size_and_time() {
        let line = b"GET / HTTP/1.1\r\n";
        let head_of = |len: usize| {
            let mut head = line.to_vec();
            head.extend(std::iter::repeat_n(b'a', len - line.len() - 4));
            head.extend_from_slice(b"\r\n\r\n");
            assert_eq!(head.len(), len);
            head
        };
        // Up to the limit, blank line included, it is taken ...
        let head = read_head(&mut &head_of(HEAD_LIMIT)[..], |_| true).unwrap();
        assert_eq!(head.len(), HEAD_LIMIT - 4);
        // ... one byte more is too much, however it arrives, and no more than
        // the limit is ever read.
        for chunk in [1, 1000, usize::MAX] {
            let long = head_of(HEAD_LIMIT + 1);
            let mut client = Trickle(&long, chunk);
            assert_eq!(
                read_head(&mut client, |_| true),
                Err(HeadError::Reply(Status::HeadTooLarge))
            );
            assert_eq!(long.len() - client.0.len(), HEAD_LIMIT);
        }
        let endless = vec![b'a'; 4 * HEAD_LIMIT];
        assert_eq!(
            read_head(&mut &endless[..], |_| true),
            Err(HeadError::Reply(Status::HeadTooLarge))
        );

        // Out of time: with part of a request that is a 408, with none of it
        // the connection is just closed.
        let mut reads = 0;
        let slow = read_head(&mut Trickle(&endless, 1), |_| {
            reads += 1;
            reads <= 3
        });
        assert_eq!(slow, Err(HeadError::Reply(Status::RequestTimeout)));
        assert_eq!(
            read_head(&mut &endless[..], |_| false),
            Err(HeadError::Silent)
        );

        // A socket read that times out is the same thing.
        struct TimesOut(bool);
        impl Read for TimesOut {
            fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
                if std::mem::take(&mut self.0) {
                    buf[0] = b'G';
                    return Ok(1);
                }
                Err(io::ErrorKind::TimedOut.into())
            }
        }
        assert_eq!(
            read_head(&mut TimesOut(true), |_| true),
            Err(HeadError::Reply(Status::RequestTimeout))
        );
        assert_eq!(
            read_head(&mut TimesOut(false), |_| true),
            Err(HeadError::Silent)
        );
    }

    // -- paths --------------------------------------------------------------

    #[test]
    fn plain_paths_name_files_in_the_site() {
        let rel = |parts: &[&str]| parts.iter().collect::<PathBuf>();
        assert_eq!(site_path("/"), Ok(rel(&["index.html"])));
        assert_eq!(
            site_path("/?ws=ws%3A%2F%2F127.0.0.1%3A8099"),
            Ok(rel(&["index.html"]))
        );
        assert_eq!(site_path("/index.html"), Ok(rel(&["index.html"])));
        assert_eq!(
            site_path("/index.html?x=../../y#z"),
            Ok(rel(&["index.html"]))
        );
        assert_eq!(
            site_path("/pkg/space_viewer_bg.wasm"),
            Ok(rel(&["pkg", "space_viewer_bg.wasm"]))
        );
        assert_eq!(
            site_path("/pkg/space_viewer.d.ts"),
            Ok(rel(&["pkg", "space_viewer.d.ts"]))
        );
        // Escapes of plain characters are plain characters.
        assert_eq!(
            site_path("/pkg%2Fspace%5Fviewer%2ejs"),
            Ok(rel(&["pkg", "space_viewer.js"]))
        );
        assert_eq!(site_path("/a-b/C_d.9"), Ok(rel(&["a-b", "C_d.9"])));
    }

    #[test]
    fn paths_that_could_leave_the_site_are_rejected() {
        let bad_request = [
            // Parent and current directory, plain and percent-encoded.
            "/..",
            "/../",
            "/../secret.txt",
            "/pkg/../../secret.txt",
            "/pkg/..",
            "/./index.html",
            "/%2e%2e/secret.txt",
            "/%2E%2E/secret.txt",
            "/.%2e/secret.txt",
            "/%2e%2e%2fsecret.txt",
            "/pkg/%2e%2e/%2e%2e/secret.txt",
            "/pkg%2f..%2f..%2fsecret.txt",
            // Backslashes.
            "/..\\secret.txt",
            "/pkg\\..\\..\\secret.txt",
            "/%5c..%5csecret.txt",
            "/..%5Csecret.txt",
            "/\\\\server\\share\\file",
            // Drives and streams.
            "/C:/Windows/win.ini",
            "/c%3A/Windows/win.ini",
            "/C:%5CWindows%5Cwin.ini",
            "/index.html::$DATA",
            // NUL and other control characters.
            "/index.html\0",
            "/index.html%00",
            "/index.html%00.js",
            "/%00",
            "/a%0d%0aSet-Cookie:x",
            "/a%09b",
            "/a%7fb",
            // Broken escapes, bytes that are not UTF-8.
            "/%",
            "/%2",
            "/%zz",
            "/a%2",
            "/%ff",
            "/%c3",
            // Not a path from the root at all.
            "",
            "index.html",
            "../secret.txt",
            "http://127.0.0.1:8098/index.html",
        ];
        for target in bad_request {
            assert_eq!(site_path(target), Err(Status::BadRequest), "{target:?}");
        }
        let not_found = [
            // Absolute paths: an empty first part.
            "//etc/passwd",
            "/%2Fetc/passwd",
            "//",
            "/pkg//space_viewer.js",
            // Directories.
            "/pkg/",
            "/index.html/",
            // Dot files, trailing dots and spaces, short names, other bytes.
            "/.git",
            "/.env",
            "/pkg/.hidden.js",
            "/...",
            "/index.html.",
            "/index.html%20",
            "/%20",
            "/SPACE_~1.JS",
            "/~",
            "/a+b",
            "/a%25b",
            "/%252e%252e/secret.txt",
            "/ünicode.js",
            "/%C3%BCnicode.js",
            "/a*b",
            "/a|b",
            "/a<b>",
            "/a\"b",
            "/$Recycle.Bin",
        ];
        for target in not_found {
            assert_eq!(site_path(target), Err(Status::NotFound), "{target:?}");
        }
        // Whatever comes back is a relative path of plain names.
        for target in bad_request.iter().chain(&not_found).chain(&["/a/b.c"]) {
            if let Ok(path) = site_path(target) {
                assert!(path
                    .components()
                    .all(|part| matches!(part, std::path::Component::Normal(_))));
                assert_eq!(*target, "/a/b.c");
            }
        }
    }

    #[test]
    fn content_types_follow_the_file_extension() {
        for (file, expected) in [
            ("index.html", "text/html; charset=utf-8"),
            ("pkg/space_viewer.js", "text/javascript; charset=utf-8"),
            ("pkg/space_viewer_bg.wasm", "application/wasm"),
            ("style.css", "text/css; charset=utf-8"),
            ("data.json", "application/json"),
            ("pkg/space_viewer.d.ts", "text/plain; charset=utf-8"),
            ("INDEX.HTML", "text/html; charset=utf-8"),
            ("A.WASM", "application/wasm"),
            ("archive.tar.gz", "application/octet-stream"),
            ("wasm", "application/octet-stream"),
            ("noextension", "application/octet-stream"),
        ] {
            assert_eq!(content_type(Path::new(file)), expected, "{file}");
        }
    }

    // -- the server, over a real socket --------------------------------------

    struct Reply {
        status: u16,
        /// Header fields, names in lower case.
        fields: Vec<(String, String)>,
        body: Vec<u8>,
    }

    impl Reply {
        fn field(&self, name: &str) -> Option<&str> {
            let mut values = self.fields.iter().filter(|(field, _)| field == name);
            let value = values.next()?;
            assert!(values.next().is_none(), "{name} sent twice");
            Some(&value.1)
        }
    }

    /// Send `request` as it is and read until the server closes the connection.
    fn exchange(server: SocketAddr, request: &[u8]) -> Reply {
        let mut stream = TcpStream::connect(server).expect("connect");
        stream
            .set_read_timeout(Some(Duration::from_secs(20)))
            .unwrap();
        stream.write_all(request).expect("send the request");
        let mut raw = Vec::new();
        stream.read_to_end(&mut raw).expect("read the response");
        let split = raw
            .windows(4)
            .position(|window| window == b"\r\n\r\n")
            .unwrap_or_else(|| panic!("no response head in {:?}", String::from_utf8_lossy(&raw)));
        let head = std::str::from_utf8(&raw[..split]).expect("an ASCII response head");
        let mut lines = head.split("\r\n");
        let status_line = lines.next().unwrap();
        let status = status_line
            .strip_prefix("HTTP/1.1 ")
            .and_then(|rest| rest.split(' ').next())
            .and_then(|code| code.parse().ok())
            .unwrap_or_else(|| panic!("bad status line {status_line:?}"));
        let fields = lines
            .map(|line| {
                let (name, value) = line.split_once(": ").expect("a header field");
                (name.to_ascii_lowercase(), value.to_owned())
            })
            .collect();
        Reply {
            status,
            fields,
            body: raw[split + 4..].to_vec(),
        }
    }

    fn get(server: SocketAddr, target: &str) -> Reply {
        exchange(
            server,
            format!("GET {target} HTTP/1.1\r\nHost: {server}\r\n\r\n").as_bytes(),
        )
    }

    /// What every response carries, whatever its status.
    fn assert_common_fields(reply: &Reply, sent: &str) {
        assert_eq!(reply.field("connection"), Some("close"), "{sent}");
        assert_eq!(reply.field("cache-control"), Some("no-cache"), "{sent}");
        assert_eq!(
            reply.field("x-content-type-options"),
            Some("nosniff"),
            "{sent}"
        );
        assert!(reply.field("content-type").is_some(), "{sent}");
        assert!(reply.field("content-length").is_some(), "{sent}");
    }

    #[test]
    fn the_server_serves_the_site_directory_and_nothing_else() {
        const PAGE: &str = "<!doctype html>\n<title>test page</title>\n";
        const SECRET: &str = "outside-the-site-directory";
        // "\0asm" and a version, then bytes that are not text.
        let wasm: Vec<u8> = [0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00]
            .into_iter()
            .chain((0..=255u8).cycle().take(100_000))
            .collect();
        let dir = temp_dir("server");
        let site = dir.join("site dir");
        std::fs::create_dir_all(site.join("pkg")).unwrap();
        std::fs::write(site.join("index.html"), PAGE).unwrap();
        std::fs::write(site.join("pkg").join("app.js"), "export default 1;\n").unwrap();
        std::fs::write(site.join("pkg").join("app_bg.wasm"), &wasm).unwrap();
        std::fs::write(dir.join("secret.txt"), SECRET).unwrap();
        std::fs::write(site.join("pkg").join("inner.txt"), "inside").unwrap();

        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        let server = listener.local_addr().unwrap();
        let root = site.canonicalize().unwrap();
        spawn_workers(listener, Site { root, log: false }).unwrap();

        // The page, as `/` and by name.
        for target in ["/", "/index.html", "/?ws=ws%3A%2F%2F127.0.0.1%3A8099"] {
            let reply = get(server, target);
            assert_eq!(reply.status, 200, "{target}");
            assert_eq!(
                reply.field("content-type"),
                Some("text/html; charset=utf-8")
            );
            assert_eq!(
                reply.field("content-length"),
                Some(&*PAGE.len().to_string())
            );
            assert_eq!(reply.body, PAGE.as_bytes(), "{target}");
            assert_common_fields(&reply, target);
        }

        // The wasm module: its media type, never cached, byte for byte.
        let reply = get(server, "/pkg/app_bg.wasm");
        assert_eq!(reply.status, 200);
        assert_eq!(reply.field("content-type"), Some("application/wasm"));
        assert_eq!(reply.field("cache-control"), Some("no-cache"));
        assert_eq!(
            reply.field("content-length"),
            Some(&*wasm.len().to_string())
        );
        assert!(reply.body == wasm, "the wasm bytes differ");
        assert_common_fields(&reply, "/pkg/app_bg.wasm");

        let reply = get(server, "/pkg/app.js");
        assert_eq!(reply.status, 200);
        assert_eq!(
            reply.field("content-type"),
            Some("text/javascript; charset=utf-8")
        );

        // HEAD: the same header fields, no body.
        let reply = exchange(server, b"HEAD /pkg/app_bg.wasm HTTP/1.1\r\n\r\n");
        assert_eq!(reply.status, 200);
        assert_eq!(reply.field("content-type"), Some("application/wasm"));
        assert_eq!(
            reply.field("content-length"),
            Some(&*wasm.len().to_string())
        );
        assert!(reply.body.is_empty());
        let reply = exchange(server, b"HEAD /nothing.js HTTP/1.0\n\n");
        assert_eq!(reply.status, 404);
        assert!(reply.body.is_empty());

        // Missing files and directories.
        for target in [
            "/missing.js",
            "/pkg/missing.wasm",
            "/pkg",
            "/pkg/",
            "/favicon.ico",
            "/index.html/x",
            // Device names, where the platform has them.
            "/NUL",
            "/CON",
            "/pkg/aux.js",
            "/COM1",
        ] {
            let reply = get(server, target);
            assert_eq!(reply.status, 404, "{target}");
            assert_eq!(reply.body, b"404 Not Found\n", "{target}");
            assert_eq!(
                reply.field("content-type"),
                Some("text/plain; charset=utf-8")
            );
            assert_common_fields(&reply, target);
        }

        // The file next to the site directory is really there ...
        assert_eq!(
            std::fs::read_to_string(site.join("pkg").join("..").join("..").join("secret.txt"))
                .unwrap(),
            SECRET
        );
        // ... and no spelling of the way to it is served.
        for target in [
            "/../secret.txt",
            "/pkg/../../secret.txt",
            "/%2e%2e/secret.txt",
            "/%2E%2E%2Fsecret.txt",
            "/pkg/%2e%2e/%2e%2e/secret.txt",
            "/pkg%2f..%2f..%2fsecret.txt",
            "/..\\secret.txt",
            "/pkg\\..\\..\\secret.txt",
            "/..%5csecret.txt",
            "/%5c..%5csecret.txt",
            "/.%2e/secret.txt",
            "//secret.txt",
            "/%252e%252e/secret.txt",
            "/..%00/secret.txt",
            "/secret.txt",
            // Also not out and back in.
            "/pkg/../index.html",
            "/../site%20dir/index.html",
        ] {
            let reply = get(server, target);
            assert!(
                matches!(reply.status, 400 | 404),
                "{target}: {}",
                reply.status
            );
            let body = String::from_utf8_lossy(&reply.body);
            assert!(
                !body.contains(SECRET) && !body.contains("test page"),
                "{target}"
            );
            assert_common_fields(&reply, target);
        }
        // An absolute path to it, in the request target or after the slash.
        let absolute = dir.join("secret.txt");
        let absolute = absolute.to_string_lossy();
        for target in [
            absolute.to_string(),
            format!("/{absolute}"),
            format!("/{}", absolute.replace('\\', "/")),
            format!("/{}", percent_encode(&absolute)),
        ] {
            let reply = exchange(server, format!("GET {target} HTTP/1.1\r\n\r\n").as_bytes());
            assert!(
                matches!(reply.status, 400 | 404),
                "{target}: {}",
                reply.status
            );
            assert!(!String::from_utf8_lossy(&reply.body).contains(SECRET));
        }

        // Only GET and HEAD.
        for method in ["POST", "PUT", "DELETE", "OPTIONS", "PATCH"] {
            let request =
                format!("{method} /index.html HTTP/1.1\r\nContent-Length: 5\r\n\r\nhello");
            let reply = exchange(server, request.as_bytes());
            assert_eq!(reply.status, 405, "{method}");
            assert_eq!(reply.field("allow"), Some("GET, HEAD"), "{method}");
            assert!(!String::from_utf8_lossy(&reply.body).contains("test page"));
            assert_common_fields(&reply, method);
        }

        // Things that are not requests get an answer, not a dead server.
        for (request, status) in [
            (&b"\r\n\r\n"[..], 400),
            (b"GET\r\n\r\n", 400),
            (b"GET / HTTP/1.1 extra\r\n\r\n", 400),
            (b"GET /\xff\xfe HTTP/1.1\r\n\r\n", 400),
            (b"\x16\x03\x01\x02\x00\x01\x00\x01\xfc\x03\x03\r\n\r\n", 400),
            (b"GET /%zz HTTP/1.1\r\n\r\n", 400),
            (b"GET http://example.org/ HTTP/1.1\r\n\r\n", 400),
            (b"GET / HTTP/2.0\r\n\r\n", 505),
        ] {
            let reply = exchange(server, request);
            assert_eq!(
                reply.status,
                status,
                "{:?}",
                String::from_utf8_lossy(request)
            );
        }
        let mut long = b"GET / HTTP/1.1\r\nX-Padding: ".to_vec();
        long.resize(3 * HEAD_LIMIT, b'a');
        long.extend_from_slice(b"\r\n\r\n");
        let reply = exchange(server, &long);
        assert_eq!(reply.status, 431);
        assert_common_fields(&reply, "a long head");

        // A connection that says nothing is closed without a word, and one
        // that stops mid-request is told so; neither takes the server down.
        let started = Instant::now();
        let mut silent = TcpStream::connect(server).unwrap();
        silent.shutdown(Shutdown::Write).unwrap();
        let mut answer = Vec::new();
        silent.read_to_end(&mut answer).unwrap();
        assert!(answer.is_empty(), "{:?}", String::from_utf8_lossy(&answer));
        assert!(started.elapsed() < HEAD_TIMEOUT);
        let mut partial = TcpStream::connect(server).unwrap();
        partial.write_all(b"GET / HTTP/1.1\r\nHost: x\r\n").unwrap();
        partial.shutdown(Shutdown::Write).unwrap();
        let mut answer = Vec::new();
        partial.read_to_end(&mut answer).unwrap();
        assert!(answer.starts_with(b"HTTP/1.1 400 "));

        // More clients at once than there are worker threads.
        let clients: Vec<_> = (0..3 * WORKERS)
            .map(|_| std::thread::spawn(move || get(server, "/pkg/app_bg.wasm")))
            .collect();
        for client in clients {
            let reply = client.join().expect("a client thread");
            assert_eq!(reply.status, 200);
            assert_eq!(reply.body.len(), wasm.len());
        }
        assert_eq!(get(server, "/").status, 200);

        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A link in the site directory that points out of it is not followed.
    #[cfg(unix)]
    #[test]
    fn a_link_out_of_the_site_directory_is_not_served() {
        let dir = temp_dir("links");
        let site = dir.join("site");
        std::fs::create_dir_all(site.join("pkg")).unwrap();
        std::fs::write(dir.join("secret.txt"), "outside").unwrap();
        std::fs::write(site.join("index.html"), "inside").unwrap();
        std::os::unix::fs::symlink(dir.join("secret.txt"), site.join("leak.txt")).unwrap();
        std::os::unix::fs::symlink(&dir, site.join("up")).unwrap();
        std::os::unix::fs::symlink(site.join("index.html"), site.join("pkg").join("page.html"))
            .unwrap();

        let root = site.canonicalize().unwrap();
        assert_eq!(read_site_file(&root, Path::new("leak.txt")), None);
        assert_eq!(
            read_site_file(&root, &Path::new("up").join("secret.txt")),
            None
        );
        // A link that stays inside is an ordinary file.
        assert_eq!(
            read_site_file(&root, &Path::new("pkg").join("page.html")),
            Some(b"inside".to_vec())
        );
        assert_eq!(
            respond(b"GET /leak.txt HTTP/1.1", &root).status,
            Status::NotFound
        );
        assert_eq!(
            respond(b"GET /up/secret.txt HTTP/1.1", &root).status,
            Status::NotFound
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn only_regular_files_inside_the_root_are_read() {
        let dir = temp_dir("files");
        let site = dir.join("site");
        std::fs::create_dir_all(site.join("pkg")).unwrap();
        std::fs::write(site.join("index.html"), "page").unwrap();
        std::fs::write(dir.join("secret.txt"), "outside").unwrap();
        let root = site.canonicalize().unwrap();

        assert_eq!(
            read_site_file(&root, Path::new("index.html")),
            Some(b"page".to_vec())
        );
        assert_eq!(read_site_file(&root, Path::new("pkg")), None);
        assert_eq!(read_site_file(&root, Path::new("")), None);
        assert_eq!(read_site_file(&root, Path::new("missing")), None);
        // `site_path` never produces these; the check on the canonical path
        // holds on its own all the same.
        assert_eq!(
            read_site_file(&root, &Path::new("..").join("secret.txt")),
            None
        );
        assert_eq!(read_site_file(&root, &dir.join("secret.txt")), None);

        let response = respond(b"HEAD /index.html HTTP/1.1\r\nHost: x", &root);
        assert_eq!(response.status, Status::Ok);
        let bytes = response.to_bytes();
        let text = String::from_utf8_lossy(&bytes);
        assert!(text.starts_with("HTTP/1.1 200 OK\r\n"), "{text}");
        assert!(text.contains("\r\nContent-Length: 4\r\n") && text.ends_with("\r\n\r\n"));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
