//! The harness against the in-memory fake client: try the HTTP API and Lua
//! scripts without a GPU, a server or the real client.
//!
//! ```sh
//! cargo run -p space-harness --example fake -- --http 127.0.0.1:8098
//! curl http://127.0.0.1:8098/                    # the routes
//! curl http://127.0.0.1:8098/state
//! curl -o frame.png http://127.0.0.1:8098/frame.png
//! curl --data-binary @test.lua http://127.0.0.1:8098/script
//!
//! cargo run -p space-harness --example fake -- --script test.lua   # exit code = the script's outcome
//! ```
//!
//! Flags: `--http ADDR` serves the control API (the token, if any, comes from
//! the `SPACE_HARNESS_TOKEN` environment variable, never from the command
//! line); `--script FILE` runs a Lua script and exits with its outcome;
//! `--out-dir DIR` is where screenshots go; `--timeout SECONDS` limits the
//! script. With `--http` and no `--script` it serves until `POST /quit`.
//!
//! This is also the shape of what a real client does with the crate: one
//! control surface, the HTTP API on a tokio runtime, a script on its own
//! thread, and the script's outcome as the process exit code.

use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;

use space_harness::fake::{FakeClient, FakeOptions};
use space_harness::http::{serve, HttpConfig};
use space_harness::script::{run_script, ScriptConfig, StopHandle};

const USAGE: &str = "usage: fake [--http ADDR] [--script FILE] [--out-dir DIR] [--timeout SECONDS]";

fn main() -> ExitCode {
    match run() {
        Ok(code) => code,
        Err(message) => {
            eprintln!("fake: {message}");
            ExitCode::from(2)
        }
    }
}

fn run() -> Result<ExitCode, String> {
    let (mut http, mut script) = (None, None);
    let mut scripts = ScriptConfig {
        on_output: Some(Arc::new(|line: &str| println!("{line}"))),
        ..ScriptConfig::default()
    };
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        let mut value = || args.next().ok_or(format!("{flag} needs a value\n{USAGE}"));
        match flag.as_str() {
            "--http" => http = Some(value()?),
            "--script" => script = Some(value()?),
            "--out-dir" => scripts.out_dir = value()?.into(),
            "--timeout" => {
                let seconds: f64 = value()?.parse().map_err(|e| format!("--timeout: {e}"))?;
                scripts.timeout =
                    Duration::try_from_secs_f64(seconds).map_err(|e| format!("--timeout: {e}"))?;
            }
            _ => return Err(format!("unknown argument {flag:?}\n{USAGE}")),
        }
    }
    if http.is_none() && script.is_none() {
        return Err(USAGE.to_owned());
    }

    let fake = FakeClient::start(FakeOptions::default());
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .map_err(|e| format!("tokio runtime: {e}"))?;

    if let Some(addr) = &http {
        let listener = runtime
            .block_on(tokio::net::TcpListener::bind(addr.as_str()))
            .map_err(|e| format!("cannot listen on {addr}: {e}"))?;
        let bound = listener.local_addr().map_err(|e| e.to_string())?;
        let config = HttpConfig {
            token: std::env::var("SPACE_HARNESS_TOKEN").ok(),
            script: scripts.clone(),
            ..HttpConfig::default()
        };
        eprintln!(
            "fake: control API on http://{bound}/ ({})",
            if config.token.is_some() {
                "bearer token required"
            } else {
                "no token: loopback only"
            }
        );
        let control = fake.control();
        runtime.spawn(async move {
            // `serve` only returns to refuse the listener (no token off loopback).
            if let Err(refused) = serve(listener, control, config).await {
                eprintln!("fake: {refused}");
                std::process::exit(2);
            }
        });
    }

    let Some(path) = script else {
        while !fake.has_quit() {
            std::thread::sleep(Duration::from_millis(50));
        }
        return Ok(ExitCode::SUCCESS);
    };
    let source = std::fs::read_to_string(&path).map_err(|e| format!("{path}: {e}"))?;
    scripts.name = std::path::Path::new(&path)
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default();
    let report = run_script(fake.control(), &source, &scripts, &StopHandle::new());
    for screenshot in &report.screenshots {
        eprintln!("fake: screenshot {screenshot}");
    }
    match &report.error {
        None => eprintln!("fake: {path}: ok ({} ms)", report.elapsed_ms),
        Some(error) => eprintln!("fake: {path}: FAILED ({:?}): {error}", error.kind),
    }
    Ok(ExitCode::from(u8::from(!report.ok)))
}
