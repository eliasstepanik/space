//! The stdio transport: newline-delimited JSON-RPC.
//!
//! Each input line is one message; each response is written as one line. Blank
//! lines are skipped. There are no sessions on stdio — the process *is* the
//! session. stdout carries protocol messages only; log to stderr.

use std::io;

use tokio::io::{AsyncBufRead, AsyncBufReadExt, AsyncWrite, AsyncWriteExt};

use crate::server::McpServer;

/// Read lines from `reader` until EOF, dispatch each through `server`, and write
/// every response as one line to `writer`.
///
/// # Errors
///
/// An I/O error on `reader` or `writer`.
pub async fn serve_stdio<R, W>(server: &McpServer, reader: R, mut writer: W) -> io::Result<()>
where
    R: AsyncBufRead + Unpin,
    W: AsyncWrite + Unpin,
{
    let mut lines = reader.lines();
    while let Some(line) = lines.next_line().await? {
        if line.trim().is_empty() {
            continue;
        }
        if let Some(response) = server.handle_bytes(line.as_bytes()).await {
            let mut out = response.to_json().into_bytes();
            out.push(b'\n');
            writer.write_all(&out).await?;
            writer.flush().await?;
        }
    }
    Ok(())
}

/// [`serve_stdio`] on the process's stdin/stdout.
///
/// # Errors
///
/// An I/O error on the standard streams.
pub async fn run_stdio(server: &McpServer) -> io::Result<()> {
    serve_stdio(
        server,
        tokio::io::BufReader::new(tokio::io::stdin()),
        tokio::io::stdout(),
    )
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool::{Tool, ToolRegistry};
    use serde_json::{json, Value};

    #[tokio::test]
    async fn newline_json_rpc_round_trips() {
        let registry = ToolRegistry::new().with(Tool::new(
            "hello",
            "Say hello.",
            json!({"type": "object"}),
            |_| async { Ok("hello".to_owned()) },
        ));
        let server = McpServer::new("t", "0", registry);
        let input = concat!(
            r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}"#,
            "\n",
            r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#,
            "\n",
            "\n",
            r#"{"jsonrpc":"2.0","id":2,"method":"tools/list"}"#,
            "\n",
            r#"{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"hello"}}"#,
            "\n",
            "garbage\n",
            r#"{"jsonrpc":"2.0","id":4,"method":"ping"}"#,
        );
        let mut out = Vec::new();
        serve_stdio(&server, input.as_bytes(), &mut out)
            .await
            .unwrap();
        let text = String::from_utf8(out).unwrap();
        let lines: Vec<Value> = text
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        // The notification and the blank line produce no output.
        assert_eq!(lines.len(), 5, "{text}");
        assert_eq!(lines[0]["result"]["serverInfo"]["name"], "t");
        assert_eq!(lines[1]["result"]["tools"][0]["name"], "hello");
        assert_eq!(lines[2]["result"]["content"][0]["text"], "hello");
        assert_eq!(lines[3]["error"]["code"], -32700);
        assert_eq!(
            (lines[4]["id"].clone(), lines[4]["result"].clone()),
            (json!(4), json!({}))
        );
    }
}
