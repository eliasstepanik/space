//! A small Model Context Protocol (MCP) implementation for agentic testing.
//!
//! * [`jsonrpc`] — JSON-RPC 2.0 messages and parsing
//! * [`tool`] — [`Tool`] and [`ToolRegistry`]
//! * [`server`] — [`McpServer`]: `initialize`, `ping`, `tools/list`, `tools/call`
//! * [`http`] — hand-rolled **Streamable HTTP** (HTTP/1.1 + SSE on tokio)
//! * [`stdio`] — newline-delimited JSON-RPC over stdin/stdout
//! * [`base64`] — for returning binary data (frames) inline
//!
//! No HTTP crate and no MCP framework crate: the only dependencies are `serde`,
//! `serde_json` and `tokio`. The binaries (`space-mcp-server`,
//! `space-mcp-client`) register their tools and call [`run`].

#![forbid(unsafe_code)]
#![warn(missing_docs, missing_debug_implementations)]

pub mod base64;
pub mod http;
pub mod jsonrpc;
pub mod server;
pub mod stdio;
pub mod tool;

use std::net::SocketAddr;
use std::sync::Arc;

pub use server::{McpServer, PROTOCOL_VERSION};
pub use tool::{Tool, ToolRegistry};

/// Which transport a binary serves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Transport {
    /// Newline-delimited JSON-RPC on stdin/stdout (the default).
    Stdio,
    /// Streamable HTTP on this address.
    Http(SocketAddr),
}

/// Split `--stdio` / `--http <addr>` off `args`, returning the transport and the
/// arguments that were not consumed (for the binary's own flags).
///
/// # Errors
///
/// A message when `--http` has no valid address or both transports are given.
pub fn parse_transport(args: &[String]) -> Result<(Transport, Vec<String>), String> {
    let mut transport = None;
    let mut rest = Vec::new();
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        let chosen = match arg.as_str() {
            "--stdio" => Transport::Stdio,
            "--http" => {
                let addr = iter
                    .next()
                    .ok_or("--http needs an address, e.g. 127.0.0.1:8765")?;
                Transport::Http(addr.parse().map_err(|e| format!("--http {addr:?}: {e}"))?)
            }
            _ => {
                rest.push(arg.clone());
                continue;
            }
        };
        if transport.replace(chosen).is_some() {
            return Err("choose one of --stdio or --http <addr>".into());
        }
    }
    Ok((transport.unwrap_or(Transport::Stdio), rest))
}

/// Serve `server` on `transport` until stdin closes (stdio) or the process is
/// killed (HTTP). The HTTP listen address is printed to **stderr**.
///
/// # Errors
///
/// A message for a bind or I/O failure.
pub async fn run(server: McpServer, transport: Transport) -> Result<(), String> {
    match transport {
        Transport::Stdio => stdio::run_stdio(&server).await.map_err(|e| e.to_string()),
        Transport::Http(addr) => {
            let listener = tokio::net::TcpListener::bind(addr)
                .await
                .map_err(|e| format!("bind {addr}: {e}"))?;
            let bound = listener.local_addr().map_err(|e| e.to_string())?;
            eprintln!(
                "MCP Streamable HTTP listening on http://{bound}{}",
                http::MCP_PATH
            );
            http::HttpTransport::new(Arc::new(server))
                .serve(listener)
                .await
                .map_err(|e| e.to_string())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(a: &[&str]) -> Vec<String> {
        a.iter().map(|s| (*s).to_owned()).collect()
    }

    #[test]
    fn stdio_is_the_default_and_other_flags_pass_through() {
        assert_eq!(
            parse_transport(&args(&["--foo", "1"])),
            Ok((Transport::Stdio, args(&["--foo", "1"])))
        );
        assert_eq!(
            parse_transport(&args(&["--http", "127.0.0.1:9", "--x"])),
            Ok((
                Transport::Http("127.0.0.1:9".parse().unwrap()),
                args(&["--x"])
            ))
        );
        assert_eq!(
            parse_transport(&args(&["--stdio"])).unwrap().0,
            Transport::Stdio
        );
    }

    #[test]
    fn bad_transport_flags_are_errors() {
        assert!(parse_transport(&args(&["--http"])).is_err());
        assert!(parse_transport(&args(&["--http", "nope"])).is_err());
        assert!(parse_transport(&args(&["--stdio", "--http", "127.0.0.1:1"])).is_err());
    }
}
