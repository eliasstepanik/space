use futures_util::{SinkExt, StreamExt};
use space_stream::protocol::{decode_frame, Key, ViewerInput};
use space_stream::{StreamServer, MAX_VIEWERS};
use std::time::{Duration, Instant};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio_tungstenite::tungstenite::Message;

async fn wait_for(mut cond: impl FnMut() -> bool, what: &str) {
    for _ in 0..400 {
        if cond() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    panic!("timeout: {what}");
}

async fn ws(addr: std::net::SocketAddr) -> tokio_tungstenite::WebSocketStream<TcpStream> {
    let tcp = TcpStream::connect(addr).await.unwrap();
    tokio_tungstenite::client_async(format!("ws://{addr}/"), tcp)
        .await
        .unwrap()
        .0
}

#[tokio::test]
async fn oversized_and_garbage_input_do_not_kill_the_server() {
    let (server, client) = StreamServer::bind("127.0.0.1:0").await.unwrap();
    let addr = server.local_addr();
    let mut a = ws(addr).await;
    wait_for(|| client.viewer_count() == 1, "viewer 1").await;
    // 1 MiB binary message: must be rejected by the size cap.
    let _ = a.send(Message::Binary(vec![2u8; 1 << 20].into())).await;
    // connection should be closed by server
    let r = tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            match a.next().await {
                None | Some(Err(_)) | Some(Ok(Message::Close(_))) => return true,
                _ => {}
            }
        }
    })
    .await;
    assert!(r.is_ok(), "oversized message did not close the connection");
    wait_for(|| client.viewer_count() == 0, "slot freed").await;
    assert!(client.poll_input().is_empty());
    // Server still serves a fresh viewer.
    let mut b = ws(addr).await;
    b.send(Message::Binary(
        ViewerInput::Key { key: Key::Up, down: true }.encode().into(),
    ))
    .await
    .unwrap();
    // text, NaN, empty, 64-byte message exactly at cap
    b.send(Message::Text("hello".into())).await.unwrap();
    b.send(Message::Binary(vec![].into())).await.unwrap();
    b.send(Message::Binary(vec![0xff; 64].into())).await.unwrap();
    let mut got = vec![];
    wait_for(
        || {
            got.extend(client.poll_input());
            !got.is_empty() && client.rejected_input() >= 2
        },
        "input after garbage",
    )
    .await;
    println!("rejected={} got={got:?}", client.rejected_input());
}

#[tokio::test]
async fn garbage_before_handshake_and_handshake_timeout() {
    let (server, client) = StreamServer::bind("127.0.0.1:0").await.unwrap();
    let addr = server.local_addr();
    // Not a websocket at all.
    let mut t = TcpStream::connect(addr).await.unwrap();
    t.write_all(b"\x00\x01\x02garbage\r\n\r\n").await.unwrap();
    wait_for(|| client.viewer_count() == 0, "garbage slot freed").await;
    drop(t);
    // Slowloris: connect and say nothing.
    let mut idle = TcpStream::connect(addr).await.unwrap();
    wait_for(|| client.viewer_count() == 1, "idle counted").await;
    let t0 = Instant::now();
    let mut buf = [0u8; 16];
    let n = tokio::time::timeout(Duration::from_secs(10), idle.read(&mut buf))
        .await
        .expect("server never dropped idle peer")
        .unwrap_or(0);
    println!("idle dropped after {:?}, read={n}", t0.elapsed());
    assert!(t0.elapsed() >= Duration::from_secs(4) && t0.elapsed() < Duration::from_secs(8));
    wait_for(|| client.viewer_count() == 0, "idle slot freed").await;
}

#[tokio::test]
async fn viewer_limit_enforced() {
    let (server, client) = StreamServer::bind("127.0.0.1:0").await.unwrap();
    let addr = server.local_addr();
    let mut keep = vec![];
    for _ in 0..MAX_VIEWERS {
        keep.push(ws(addr).await);
    }
    wait_for(|| client.viewer_count() == MAX_VIEWERS, "full").await;
    let tcp = TcpStream::connect(addr).await.unwrap();
    let r = tokio_tungstenite::client_async(format!("ws://{addr}/"), tcp).await;
    assert!(r.is_err(), "9th viewer accepted");
    assert_eq!(client.viewer_count(), MAX_VIEWERS);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn slow_viewer_does_not_block_or_balloon_and_fast_viewer_gets_newest() {
    let (server, client) = StreamServer::bind("127.0.0.1:0").await.unwrap();
    let addr = server.local_addr();
    // Slow viewer: handshake, then never read.
    let slow = ws(addr).await;
    let mut fast = ws(addr).await;
    wait_for(|| client.viewer_count() == 2, "2 viewers").await;
    let t0 = Instant::now();
    let mut last = 0;
    for i in 0..400u32 {
        last = client.push_jpeg(vec![(i & 0xff) as u8; 1 << 20], 0); // 1 MiB each = 400 MiB offered
        if i % 8 == 0 {
            tokio::task::yield_now().await;
        }
    }
    println!("400 pushes of 1MiB took {:?}", t0.elapsed());
    assert!(t0.elapsed() < Duration::from_secs(5), "push_jpeg blocked");
    // fast viewer eventually gets the final seq
    let mut seen_last = false;
    let mut count = 0;
    while !seen_last {
        let m = tokio::time::timeout(Duration::from_secs(10), fast.next())
            .await
            .expect("fast viewer starved")
            .unwrap()
            .unwrap();
        if let Message::Binary(b) = m {
            let (h, _) = decode_frame(&b).unwrap();
            count += 1;
            seen_last = h.seq == last;
        }
    }
    println!("fast viewer saw {count} of 400 frames, final seq reached");
    // The slow viewer still holds its slot (no write timeout).
    tokio::time::sleep(Duration::from_secs(6)).await;
    println!("viewer_count after 6s with stalled viewer = {}", client.viewer_count());
    drop(slow);
}

#[tokio::test]
async fn input_flood_is_bounded() {
    let (server, client) = StreamServer::bind("127.0.0.1:0").await.unwrap();
    let mut a = ws(server.local_addr()).await;
    let m = Message::Binary(ViewerInput::View { yaw: 1.0, pitch: 0.0 }.encode().into());
    for _ in 0..20_000 {
        a.feed(m.clone()).await.unwrap();
    }
    a.flush().await.unwrap();
    tokio::time::sleep(Duration::from_millis(500)).await;
    let n = client.poll_input().len();
    println!("flood: delivered {n} of 20000");
    assert!(n <= 1024);
    // still alive
    a.send(m).await.unwrap();
    wait_for(|| !client.poll_input().is_empty(), "alive after flood").await;
}
