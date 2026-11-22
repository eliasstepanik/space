//! MCP over loopback: the real Streamable HTTP transport on a real TCP socket,
//! driving the real shard tools. No browser, nothing beyond 127.0.0.1.

use std::sync::Arc;

use serde_json::{json, Value};
use space_mcp::http::HttpTransport;
use space_mcp::stdio::serve_stdio;
use space_mcp_server::{mcp_server, ShardHarness};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

const TOOLS: [&str; 7] = [
    "shard_status",
    "topology",
    "list_ships",
    "spawn_ship",
    "ship_state",
    "despawn_ship",
    "advance_ticks",
];

/// A parsed HTTP response.
struct Http {
    status: u16,
    headers: Vec<(String, String)>,
    body: String,
}

impl Http {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(n, _)| n.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }

    /// The JSON-RPC message in the body, whether sent as JSON or as one SSE event.
    fn rpc(&self) -> Value {
        let json = match self.header("content-type") {
            Some("text/event-stream") => self
                .body
                .strip_prefix("event: message\ndata: ")
                .and_then(|b| b.strip_suffix("\n\n"))
                .unwrap_or_else(|| panic!("bad SSE framing: {:?}", self.body)),
            _ => &self.body,
        };
        serde_json::from_str(json).unwrap_or_else(|e| panic!("{e}: {json:?}"))
    }
}

async fn post(addr: std::net::SocketAddr, accept: &str, session: Option<&str>, body: &str) -> Http {
    let session = session.map_or(String::new(), |s| format!("Mcp-Session-Id: {s}\r\n"));
    let request = format!(
        "POST /mcp HTTP/1.1\r\nHost: {addr}\r\nContent-Type: application/json\r\nAccept: {accept}\r\n{session}Content-Length: {}\r\n\r\n{body}",
        body.len()
    );
    let mut stream = TcpStream::connect(addr).await.unwrap();
    stream.write_all(request.as_bytes()).await.unwrap();
    let mut raw = Vec::new();
    stream.read_to_end(&mut raw).await.unwrap();
    let text = String::from_utf8(raw).unwrap();
    let (head, body) = text.split_once("\r\n\r\n").expect("a complete response");
    let mut lines = head.split("\r\n");
    let status = lines
        .next()
        .unwrap()
        .split(' ')
        .nth(1)
        .unwrap()
        .parse()
        .unwrap();
    let headers = lines
        .map(|l| {
            let (n, v) = l.split_once(": ").unwrap();
            (n.to_owned(), v.to_owned())
        })
        .collect();
    Http {
        status,
        headers,
        body: body.to_owned(),
    }
}

const BOTH: &str = "application/json, text/event-stream";

fn call(id: u64, name: &str, arguments: &Value) -> String {
    json!({"jsonrpc":"2.0","id":id,"method":"tools/call","params":{"name":name,"arguments":arguments}})
        .to_string()
}

/// The JSON a tool returned as text.
fn tool_json(rpc: &Value) -> Value {
    assert_eq!(rpc["result"]["isError"], false, "{rpc}");
    assert_eq!(rpc["result"]["content"][0]["type"], "text");
    serde_json::from_str(rpc["result"]["content"][0]["text"].as_str().unwrap()).unwrap()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn mcp_over_streamable_http_drives_a_real_shard() {
    let harness = Arc::new(
        ShardHarness::boot(([127, 0, 0, 1], 0).into(), false)
            .await
            .expect("boot shard"),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let transport = HttpTransport::new(Arc::new(mcp_server(&harness)));
    let serving = tokio::spawn(transport.serve(listener));

    // initialize → session id
    let init = post(
        addr,
        BOTH,
        None,
        r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"test","version":"0"}}}"#,
    )
    .await;
    assert_eq!(init.status, 200);
    assert_eq!(init.header("content-type"), Some("application/json"));
    let session = init
        .header("mcp-session-id")
        .expect("Mcp-Session-Id")
        .to_owned();
    let rpc = init.rpc();
    assert_eq!(rpc["result"]["protocolVersion"], "2025-06-18");
    assert_eq!(rpc["result"]["capabilities"], json!({"tools": {}}));
    assert_eq!(rpc["result"]["serverInfo"]["name"], "space-mcp-server");

    // the initialized notification is accepted without a body
    let note = post(
        addr,
        BOTH,
        Some(&session),
        r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#,
    )
    .await;
    assert_eq!((note.status, note.body.as_str()), (202, ""));

    // session enforcement: a missing id is 400, an unknown one 404 (the status
    // that tells a client to re-initialise), both with a JSON-RPC error
    let list = r#"{"jsonrpc":"2.0","id":2,"method":"tools/list"}"#;
    for (bad, status) in [(None, 400), (Some("not-a-session"), 404)] {
        let r = post(addr, BOTH, bad, list).await;
        assert_eq!(r.status, status, "{bad:?}");
        assert_eq!(r.rpc()["error"]["code"], -32600);
    }

    // tools/list
    let listed = post(addr, BOTH, Some(&session), list).await;
    assert_eq!(listed.status, 200);
    let names: Vec<String> = listed.rpc()["result"]["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["name"].as_str().unwrap().to_owned())
        .collect();
    assert_eq!(names, TOOLS);
    for tool in listed.rpc()["result"]["tools"].as_array().unwrap() {
        assert!(!tool["description"].as_str().unwrap().is_empty());
        assert_eq!(tool["inputSchema"]["type"], "object");
    }

    // tools/call on a freshly booted (untouched, not ticking) shard
    let r = post(
        addr,
        BOTH,
        Some(&session),
        &call(3, "shard_status", &json!({})),
    )
    .await;
    assert_eq!(r.status, 200);
    let status = tool_json(&r.rpc());
    assert_eq!(status["shard"], 1);
    assert_eq!(
        (
            status["tick"].clone(),
            status["ships"].clone(),
            status["clients"].clone()
        ),
        (json!(0), json!(0), json!(0))
    );
    assert_eq!(status["quic_endpoint"], harness.addr.to_string());

    // spawn → advance → inspect, through the same session; SSE this time
    let r = post(
        addr,
        "text/event-stream",
        Some(&session),
        &call(4, "spawn_ship", &json!({"offset": [10.0, 0.0, -5.0]})),
    )
    .await;
    assert_eq!(r.header("content-type"), Some("text/event-stream"));
    let spawned = tool_json(&r.rpc());
    let guid = spawned["guid"].as_str().unwrap().to_owned();
    assert_eq!(
        spawned["offset_from_region_centre_m"],
        json!([10.0, 0.0, -5.0])
    );
    assert_eq!(spawned["velocity_mps"], json!([0.0, 0.0, 0.0]));

    let r = post(
        addr,
        BOTH,
        Some(&session),
        &call(5, "advance_ticks", &json!({"ticks": 30})),
    )
    .await;
    assert_eq!(tool_json(&r.rpc())["tick"], 30);
    let r = post(
        addr,
        BOTH,
        Some(&session),
        &call(6, "ship_state", &json!({"guid": guid})),
    )
    .await;
    let state = tool_json(&r.rpc());
    assert_eq!(state["guid"], guid);
    assert_eq!(
        state["offset_from_region_centre_m"],
        json!([10.0, 0.0, -5.0]),
        "an at-rest ship stays put"
    );

    let r = post(
        addr,
        BOTH,
        Some(&session),
        &call(7, "list_ships", &json!({})),
    )
    .await;
    assert_eq!(tool_json(&r.rpc())["count"], 1);
    let r = post(addr, BOTH, Some(&session), &call(8, "topology", &json!({}))).await;
    let topo = tool_json(&r.rpc());
    assert_eq!(topo["regions"][0]["is_this_shard"], true);

    // tool failures are isError results, not protocol errors
    let r = post(
        addr,
        BOTH,
        Some(&session),
        &call(9, "ship_state", &json!({"guid": "1:999"})),
    )
    .await;
    let rpc = r.rpc();
    assert_eq!(r.status, 200);
    assert_eq!(rpc["result"]["isError"], true);
    assert!(rpc["result"]["content"][0]["text"]
        .as_str()
        .unwrap()
        .contains("no ship"));
    for (name, args) in [
        ("spawn_ship", json!({"offset": [9999.0, 0.0, 0.0]})),
        ("spawn_ship", json!({"offset": [1.0]})),
        ("ship_state", json!({"guid": "banana"})),
        ("advance_ticks", json!({"ticks": 0})),
        ("advance_ticks", json!({})),
    ] {
        let r = post(addr, BOTH, Some(&session), &call(10, name, &args)).await;
        assert_eq!(r.rpc()["result"]["isError"], true, "{name} {args}");
    }

    let r = post(
        addr,
        BOTH,
        Some(&session),
        &call(11, "despawn_ship", &json!({"guid": guid})),
    )
    .await;
    assert_eq!(r.rpc()["result"]["isError"], false);
    let r = post(
        addr,
        BOTH,
        Some(&session),
        &call(12, "list_ships", &json!({})),
    )
    .await;
    assert_eq!(tool_json(&r.rpc())["count"], 0);

    // an unknown method is a JSON-RPC error; an unknown tool is invalid params
    let r = post(
        addr,
        BOTH,
        Some(&session),
        r#"{"jsonrpc":"2.0","id":13,"method":"resources/list"}"#,
    )
    .await;
    assert_eq!(r.rpc()["error"]["code"], -32601);
    let r = post(
        addr,
        BOTH,
        Some(&session),
        &call(14, "no_such_tool", &json!({})),
    )
    .await;
    assert_eq!(r.rpc()["error"]["code"], -32602);

    serving.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_stdio_shim_round_trips_newline_json_rpc() {
    let harness = Arc::new(
        ShardHarness::boot(([127, 0, 0, 1], 0).into(), false)
            .await
            .expect("boot shard"),
    );
    let server = mcp_server(&harness);
    let input = [
        r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}"#.to_owned(),
        r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#.to_owned(),
        r#"{"jsonrpc":"2.0","id":2,"method":"tools/list"}"#.to_owned(),
        call(3, "advance_ticks", &json!({"ticks": 5})),
        call(4, "shard_status", &json!({})),
    ]
    .join("\n");
    let mut out = Vec::new();
    serve_stdio(&server, input.as_bytes(), &mut out)
        .await
        .unwrap();
    let text = String::from_utf8(out).unwrap();
    let lines: Vec<Value> = text
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(lines.len(), 4, "the notification gets no reply: {text}");
    assert_eq!(lines[0]["result"]["protocolVersion"], "2025-06-18");
    assert_eq!(
        lines[1]["result"]["tools"].as_array().unwrap().len(),
        TOOLS.len()
    );
    assert_eq!(
        tool_json(&lines[3])["tick"],
        5,
        "stdio shares the shard with the tools"
    );
}

/// `--no-tick`: nothing but `advance_ticks` moves the shard, so `advance_ticks`
/// is also the only thing that can deliver what a tick queued.
///
/// Mutation this catches: stepping with `tick_n` and never flushing. Every tick
/// then left its frames for the joined player in the shard's outbox — the
/// client never saw one (so it never learnt its ship and never sent input), and
/// the outbox grew by a frame per client per tick for as long as the server ran.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn advance_ticks_delivers_each_tick_instead_of_hoarding_frames() {
    let harness = Arc::new(
        ShardHarness::boot(([127, 0, 0, 1], 0).into(), false)
            .await
            .expect("boot shard"),
    );
    let server = mcp_server(&harness);
    // A player joins: the call the QUIC front end makes for a connecting client.
    let welcome = harness
        .shard
        .lock()
        .await
        .join(&space_protocol::Hello {
            protocol: space_protocol::PROTOCOL_VERSION,
            player_token: "player".into(),
        })
        .expect("the shard has room");

    let reply = server
        .handle_bytes(call(1, "advance_ticks", &json!({"ticks": 20})).as_bytes())
        .await
        .expect("a reply")
        .to_value();
    assert_eq!(tool_json(&reply)["tick"], 20);

    let mut shard = harness.shard.lock().await;
    assert!(
        shard.ship_of(welcome.session).is_some(),
        "the player has a ship, so every tick built a frame for it"
    );
    let undelivered = shard.take_outbox();
    assert!(
        undelivered.is_empty(),
        "{} message(s) were left in the outbox: advance_ticks must flush after every tick",
        undelivered.len()
    );
}
