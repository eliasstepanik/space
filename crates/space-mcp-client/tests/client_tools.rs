//! The client MCP tools, driven through the real MCP dispatcher against a real
//! in-process shard (QUIC over loopback). `capture_frame` needs a Vulkan device
//! (lavapipe is fine) and skips cleanly without one.

use std::sync::Arc;
use std::time::Duration;

use serde_json::{json, Value};
use space_mcp::McpServer;
use space_mcp_client::{mcp_server, ClientState};
use space_mcp_server::ShardHarness;
use space_render::{ContextOptions, RenderContext};

fn out_dir(tag: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "space-mcp-client-test-{tag}-{}",
        std::process::id()
    ))
}

/// Call a tool; `Ok(text)` for a normal result, `Err(text)` for `isError`.
async fn call(server: &McpServer, name: &str, arguments: Value) -> Result<String, String> {
    let body = json!({"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":name,"arguments":arguments}});
    let reply = server
        .handle_bytes(body.to_string().as_bytes())
        .await
        .expect("a response")
        .to_value();
    let text = reply["result"]["content"][0]["text"]
        .as_str()
        .unwrap_or_else(|| panic!("no text in {reply}"))
        .to_owned();
    if reply["result"]["isError"] == true {
        Err(text)
    } else {
        Ok(text)
    }
}

async fn call_json(server: &McpServer, name: &str, arguments: Value) -> Value {
    let text = call(server, name, arguments)
        .await
        .unwrap_or_else(|e| panic!("{name} failed: {e}"));
    serde_json::from_str(&text).unwrap_or_else(|_| panic!("{name} returned non-JSON: {text}"))
}

#[tokio::test]
async fn the_client_exposes_its_tools_and_rejects_bad_calls() {
    let server = mcp_server(&ClientState::new(out_dir("list")));
    assert_eq!(
        server.registry().names(),
        [
            "connect",
            "disconnect",
            "client_status",
            "send_input",
            "capture_frame"
        ]
    );
    // Not connected yet.
    assert_eq!(
        call_json(&server, "client_status", json!({})).await["connected"],
        false
    );
    assert!(call(&server, "send_input", json!({"thrust": [0, 0, -1]}))
        .await
        .unwrap_err()
        .contains("not connected"));
    assert!(call(&server, "disconnect", json!({})).await.is_err());
    // Argument validation happens before any side effect.
    for args in [
        json!({"thrust": [0, 0]}),
        json!({"thrust": [0, 0, 2]}),
        json!({"thrust": [0, 0, "x"]}),
        json!({"thrust": [0, 0, 0], "duration_ms": 0}),
        json!({"thrust": [0, 0, 0], "duration_ms": 999_999}),
    ] {
        assert!(
            call(&server, "send_input", args.clone()).await.is_err(),
            "{args}"
        );
    }
    for args in [
        json!({"width": 4}),
        json!({"width": 100_000}),
        json!({"name": "../escape.jpg"}),
        json!({"name": "a/b.jpg"}),
        json!({"name": ".hidden"}),
    ] {
        let e = call(&server, "capture_frame", args.clone())
            .await
            .unwrap_err();
        assert!(e.contains("must be"), "{args}: {e}");
    }
    assert!(call(&server, "connect", json!({})).await.is_err());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn connect_thrust_and_status_drive_a_real_shard() {
    let harness = Arc::new(
        ShardHarness::boot(([127, 0, 0, 1], 0).into(), true)
            .await
            .expect("boot shard"),
    );
    let client = mcp_server(&ClientState::new(out_dir("drive")));
    let shard_tools = space_mcp_server::mcp_server(&harness);

    let joined = call_json(
        &client,
        "connect",
        json!({"addr": harness.addr.to_string()}),
    )
    .await;
    assert_eq!(joined["connected"], true);
    assert!(call(
        &client,
        "connect",
        json!({"addr": harness.addr.to_string()})
    )
    .await
    .unwrap_err()
    .contains("already connected"));

    // Hold forward thrust (-Z); replication brings the ship's motion back.
    call_json(
        &client,
        "send_input",
        json!({"thrust": [0, 0, -1], "duration_ms": 800}),
    )
    .await;
    let mut status = Value::Null;
    for _ in 0..100 {
        status = call_json(&client, "client_status", json!({})).await;
        if status["offset_m"][2].as_f64().unwrap_or(0.0) < -0.05 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(30)).await;
    }
    assert!(
        status["offset_m"][2].as_f64().unwrap() < -0.05,
        "ship never moved: {status}"
    );
    assert!(status["frames"].as_u64().unwrap() > 0);
    let ship = status["ship"].as_str().expect("ship guid").to_owned();

    // The *server* tools see the same authoritative ship, owned by our session.
    let ships = call_json(&shard_tools, "list_ships", json!({})).await;
    let mine = ships["ships"]
        .as_array()
        .unwrap()
        .iter()
        .find(|s| s["guid"] == ship)
        .expect("our ship is on the shard");
    assert_eq!(mine["session"], joined["session"]);
    assert!(mine["offset_from_region_centre_m"][2].as_f64().unwrap() < 0.0);

    call(&client, "disconnect", json!({})).await.unwrap();
    assert_eq!(
        call_json(&client, "client_status", json!({})).await["connected"],
        false
    );
    // The shard drops the player's ship once it notices the departure.
    let mut count = 1;
    for _ in 0..100 {
        count = call_json(&shard_tools, "shard_status", json!({})).await["clients"]
            .as_u64()
            .unwrap();
        if count == 0 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(30)).await;
    }
    assert_eq!(count, 0, "the shard never saw the disconnect");
}

/// Skips (loudly) when no Vulkan device is usable; runs on lavapipe in CI.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn capture_frame_returns_a_jpeg_on_a_gpu() {
    if let Err(e) = RenderContext::new(&ContextOptions::default()) {
        eprintln!(
            "SKIP: no Vulkan device ({}); see spikes/ENVIRONMENT.md",
            e.message
        );
        return;
    }
    let dir = out_dir("capture");
    let client = mcp_server(&ClientState::new(dir.clone()));
    let frame = call_json(
        &client,
        "capture_frame",
        json!({"width": 160, "height": 120, "name": "frame.jpg"}),
    )
    .await;
    assert_eq!(
        (
            frame["format"].clone(),
            frame["width"].clone(),
            frame["height"].clone()
        ),
        (json!("jpeg"), json!(160), json!(120))
    );
    let jpeg = space_mcp::base64::decode(frame["base64"].as_str().unwrap()).expect("valid base64");
    assert_eq!(jpeg.len() as u64, frame["bytes"].as_u64().unwrap());
    assert_eq!(&jpeg[..2], &[0xff, 0xd8], "JPEG SOI");
    assert_eq!(&jpeg[jpeg.len() - 2..], &[0xff, 0xd9], "JPEG EOI");
    assert_eq!(space_stream::jpeg_dimensions(&jpeg), Some((160, 120)));
    // The saved file is byte-identical to the inline copy.
    let saved = std::fs::read(frame["path"].as_str().unwrap()).expect("saved file");
    assert_eq!(saved, jpeg);
    // The frame is a lit sphere on black, not an empty buffer: more than a
    // trivial amount of entropy survives JPEG compression.
    assert!(
        jpeg.len() > 600,
        "suspiciously tiny frame: {} bytes",
        jpeg.len()
    );

    let bare = call_json(
        &client,
        "capture_frame",
        json!({"width": 64, "height": 64, "inline": false}),
    )
    .await;
    assert!(bare.get("base64").is_none() && bare.get("path").is_none());
    let _ = std::fs::remove_dir_all(dir);
}
