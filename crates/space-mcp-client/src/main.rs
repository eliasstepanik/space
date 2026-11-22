//! `space-mcp-client` binary: `--stdio` (default) or `--http <addr>`.
//!
//! ```text
//! space-mcp-client [--stdio | --http 127.0.0.1:8766] [--out-dir DIR]
//! ```
//!
//! `--out-dir` is where `capture_frame` saves a named frame (default: the
//! system temp directory).

#![forbid(unsafe_code)]

use std::path::PathBuf;

use space_mcp_client::{mcp_server, ClientState};

const USAGE: &str = "usage: space-mcp-client [--stdio | --http <addr:port>] [--out-dir <dir>]";

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!("{USAGE}");
        return Ok(());
    }
    let (transport, rest) =
        space_mcp::parse_transport(&args).map_err(|e| format!("{e}\n{USAGE}"))?;
    let mut out_dir = std::env::temp_dir();
    let mut rest = rest.into_iter();
    while let Some(arg) = rest.next() {
        match arg.as_str() {
            "--out-dir" => {
                out_dir = rest
                    .next()
                    .map(PathBuf::from)
                    .ok_or(format!("--out-dir needs a directory\n{USAGE}"))?;
            }
            other => return Err(format!("unknown argument {other:?}\n{USAGE}")),
        }
    }
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| e.to_string())?;
    runtime.block_on(space_mcp::run(
        mcp_server(&ClientState::new(out_dir)),
        transport,
    ))
}
