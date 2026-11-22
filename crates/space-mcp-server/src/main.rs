//! `space-mcp-server` binary: `--stdio` (default) or `--http <addr>`.
//!
//! ```text
//! space-mcp-server [--stdio | --http 127.0.0.1:8765] [--bind 127.0.0.1:0] [--no-tick]
//! ```
//!
//! `--bind` is the shard's QUIC address (default loopback, any port); the bound
//! address is reported on stderr and by the `shard_status` tool.

#![forbid(unsafe_code)]

use std::net::SocketAddr;
use std::sync::Arc;

use space_mcp_server::{mcp_server, ShardHarness};

const USAGE: &str = "usage: space-mcp-server [--stdio | --http <addr:port>] [--bind <addr:port>] [--no-tick]";

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!("{USAGE}");
        return Ok(());
    }
    let (transport, rest) = space_mcp::parse_transport(&args).map_err(|e| format!("{e}\n{USAGE}"))?;
    let mut bind: SocketAddr = ([127, 0, 0, 1], 0).into();
    let mut ticking = true;
    let mut rest = rest.into_iter();
    while let Some(arg) = rest.next() {
        match arg.as_str() {
            "--bind" => {
                let v = rest.next().ok_or(format!("--bind needs an address\n{USAGE}"))?;
                bind = v.parse().map_err(|e| format!("--bind {v:?}: {e}"))?;
            }
            "--no-tick" => ticking = false,
            other => return Err(format!("unknown argument {other:?}\n{USAGE}")),
        }
    }
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| e.to_string())?;
    runtime.block_on(async move {
        let harness = Arc::new(ShardHarness::boot(bind, ticking).await?);
        eprintln!("space-mcp-server: shard {} on quic {}", space_mcp_server::SHARD_ID, harness.addr);
        space_mcp::run(mcp_server(&harness), transport).await
    })
}
