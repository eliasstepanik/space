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

/// A `connect` that is still dialling must not make every other tool wait.
///
/// Mutation this catches: holding the link mutex across the dial. Against a
/// shard that does not answer the dial runs to its timeout, and
/// `client_status`, `send_input`, `capture_frame` and `disconnect` — which all
/// take the same mutex — hung behind it for the whole time.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_connect_that_is_still_dialling_does_not_block_the_other_tools() {
    // A bound UDP socket nobody reads: the QUIC handshake gets no answer.
    let silent = std::net::UdpSocket::bind("127.0.0.1:0").expect("bind a silent port");
    let addr = silent.local_addr().expect("silent addr").to_string();
    let client = mcp_server(&ClientState::new(out_dir("dialling")));

    let dialling = {
        let client = client.clone();
        tokio::spawn(async move { call(&client, "connect", json!({"addr": addr})).await })
    };
    // Give the dial time to start (and to take any lock it was going to take).
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert!(!dialling.is_finished(), "the dial is still in flight");

    let prompt = Duration::from_secs(2);
    let status = tokio::time::timeout(prompt, call_json(&client, "client_status", json!({})))
        .await
        .expect("client_status must not wait for a connect in flight");
    assert_eq!(status["connected"], false);
    let refused = tokio::time::timeout(
        prompt,
        call(&client, "send_input", json!({"thrust": [0, 0, -1]})),
    )
    .await
    .expect("send_input must not wait for a connect in flight");
    assert!(refused.unwrap_err().contains("not connected"));
    let refused = tokio::time::timeout(prompt, call(&client, "disconnect", json!({})))
        .await
        .expect("disconnect must not wait for a connect in flight");
    assert!(refused.unwrap_err().contains("not connected"));

    // The dial itself ends in an error, not a hang.
    let outcome = tokio::time::timeout(Duration::from_secs(30), dialling)
        .await
        .expect("the dial gives up")
        .expect("the dial did not panic");
    let err = outcome.expect_err("nobody answered");
    assert!(err.contains("timed out") || err.contains("failed"), "{err}");
}

/// `space-mcp-server --no-tick` with a player: the documented deterministic
/// loop is connect, `advance_ticks`, then look at what replication delivered.
///
/// Mutation this catches: `advance_ticks` stepping without flushing. With no
/// 30 Hz loop nothing else delivers a tick's frames, so the client stayed at
/// zero frames with no ship for ever (and, never learning its ship, never sent
/// an input either).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_client_of_a_no_tick_shard_receives_frames_when_the_shard_is_stepped() {
    let harness = Arc::new(
        ShardHarness::boot(([127, 0, 0, 1], 0).into(), false)
            .await
            .expect("boot shard"),
    );
    let client = mcp_server(&ClientState::new(out_dir("no-tick")));
    let shard_tools = space_mcp_server::mcp_server(&harness);

    let joined = call_json(
        &client,
        "connect",
        json!({"addr": harness.addr.to_string()}),
    )
    .await;
    assert_eq!(joined["connected"], true);
    // Nothing has ticked: no frame can have arrived yet.
    let before = call_json(&client, "client_status", json!({})).await;
    assert_eq!(before["frames"], 0, "{before}");

    let mut status = Value::Null;
    for _ in 0..100 {
        let stepped = call_json(&shard_tools, "advance_ticks", json!({"ticks": 2})).await;
        assert!(stepped["tick"].as_u64().unwrap() >= 2);
        tokio::time::sleep(Duration::from_millis(20)).await;
        status = call_json(&client, "client_status", json!({})).await;
        if status["frames"].as_u64().unwrap_or(0) > 0 && status["ship"].is_string() {
            break;
        }
    }
    assert!(
        status["frames"].as_u64().unwrap() > 0,
        "stepping the shard must deliver frames to its client: {status}"
    );
    assert!(
        status["ship"].is_string(),
        "and the client learns which ship is its own: {status}"
    );
    assert_eq!(status["connected"], true);

    call(&client, "disconnect", json!({})).await.unwrap();
}

/// A named capture is written from the blocking pool, into a directory that is
/// created on demand, and a failure names the file.
#[test]
fn save_capture_creates_the_directory_and_reports_failures() {
    let dir = out_dir("save").join("nested");
    let _ = std::fs::remove_dir_all(&dir);
    let path = dir.join("frame.jpg");
    space_mcp_client::save_capture(&path, b"jpeg bytes").expect("saved");
    assert_eq!(std::fs::read(&path).expect("readable"), b"jpeg bytes");
    // A path whose "directory" is a file cannot be created.
    let blocked = path.join("frame.jpg");
    let err = space_mcp_client::save_capture(&blocked, b"x").expect_err("not a directory");
    assert!(
        err.contains("cannot save") && err.contains("frame.jpg"),
        "{err}"
    );
    let _ = std::fs::remove_dir_all(out_dir("save"));
}
