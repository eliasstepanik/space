//! Phase-0 spike 0.4 smoke driver.
//!
//! Starts the host, launches headless Chromium (CDP over loopback), opens the
//! viewer page, waits for the presented-frame counter to advance, sends a key
//! press through the page, then asserts:
//!   * the host log shows the input was applied, and
//!   * the presented frame id keeps advancing after the key press.
//! Reports frames presented, fps, whether the key changed state, and the
//! measured frame-to-browser latency. Exits non-zero on any failed assertion.
//!
//! THROWAWAY spike driver.

use std::io::Read;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use futures_util::{SinkExt, StreamExt};
use serde_json::{json, Value};
use tokio::net::TcpStream;
use tokio_tungstenite::tungstenite::Message;

const CDP_PORT: u16 = 9333;
const HOST: &str = "127.0.0.1";

// --- tiny helpers ----------------------------------------------------------

fn http_get(port: u16, path: &str) -> Option<String> {
    use std::io::Write;
    let addr = format!("{HOST}:{port}");
    let mut stream = std::net::TcpStream::connect(&addr).ok()?;
    stream
        .set_read_timeout(Some(Duration::from_secs(5)))
        .ok()?;
    let req = format!("GET {path} HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\n\r\n");
    stream.write_all(req.as_bytes()).ok()?;
    // Chromium keeps the connection open even with Connection: close, so read
    // headers, honour Content-Length, then stop (do NOT read_to_end).
    let mut buf = Vec::new();
    let mut tmp = [0u8; 4096];
    let header_end;
    loop {
        let n = stream.read(&mut tmp).ok()?;
        if n == 0 {
            header_end = find_double_crlf(&buf)?;
            break;
        }
        buf.extend_from_slice(&tmp[..n]);
        if let Some(pos) = find_double_crlf(&buf) {
            header_end = pos;
            break;
        }
        if buf.len() > 65536 {
            return None;
        }
    }
    let headers = String::from_utf8_lossy(&buf[..header_end]).to_lowercase();
    let content_len = headers
        .lines()
        .find_map(|l| l.strip_prefix("content-length:"))
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(0);
    let mut body = buf[header_end + 4..].to_vec();
    while body.len() < content_len {
        let n = stream.read(&mut tmp).ok()?;
        if n == 0 {
            break;
        }
        body.extend_from_slice(&tmp[..n]);
    }
    Some(String::from_utf8_lossy(&body).into_owned())
}

fn find_double_crlf(buf: &[u8]) -> Option<usize> {
    buf.windows(4).position(|w| w == b"\r\n\r\n")
}

fn wait_host_ready(log: &PathBuf, timeout: Duration) -> bool {
    let start = Instant::now();
    while start.elapsed() < timeout {
        if let Ok(s) = std::fs::read_to_string(log) {
            if s.contains("[host] listening") {
                return true;
            }
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    false
}

fn wait_cdp_ready(timeout: Duration) -> bool {
    let start = Instant::now();
    while start.elapsed() < timeout {
        if http_get(CDP_PORT, "/json/version").is_some() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(200));
    }
    false
}

fn page_ws_url() -> Option<String> {
    let body = http_get(CDP_PORT, "/json")?;
    let v: Value = serde_json::from_str(&body).ok()?;
    let arr = v.as_array()?;
    for t in arr {
        if t["type"].as_str() == Some("page") {
            if let Some(u) = t["webSocketDebuggerUrl"].as_str() {
                return Some(u.to_string());
            }
        }
    }
    None
}

fn page_ws_url_for(substr: &str) -> Option<String> {
    let body = http_get(CDP_PORT, "/json")?;
    let v: Value = serde_json::from_str(&body).ok()?;
    let arr = v.as_array()?;
    for t in arr {
        if t["type"].as_str() == Some("page") {
            if let Some(u) = t["url"].as_str() {
                if u.contains(substr) {
                    if let Some(w) = t["webSocketDebuggerUrl"].as_str() {
                        return Some(w.to_string());
                    }
                }
            }
        }
    }
    None
}

// --- minimal CDP client over WebSocket -------------------------------------

struct Cdp {
    ws: tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<TcpStream>>,
    id: u64,
}

impl Cdp {
    async fn connect(url: &str) -> anyhow::Result<Cdp> {
        let (ws, _) = tokio_tungstenite::connect_async(url).await?;
        Ok(Cdp { ws, id: 0 })
    }

    async fn call(&mut self, method: &str, params: Value) -> anyhow::Result<Value> {
        self.id += 1;
        let req = json!({ "id": self.id, "method": method, "params": params });
        self.ws.send(Message::Text(req.to_string().into())).await?;
        // Read until we get the matching id.
        loop {
            if let Some(Ok(Message::Text(t))) = self.ws.next().await {
                let v: Value = serde_json::from_str(&t)?;
                if v["id"].as_u64() == Some(self.id) {
                    if let Some(err) = v.get("error") {
                        anyhow::bail!("CDP {method} error: {err}");
                    }
                    return Ok(v.get("result").cloned().unwrap_or(Value::Null));
                }
            }
        }
    }

    async fn evaluate(&mut self, expr: &str) -> anyhow::Result<Value> {
        let r = self
            .call(
                "Runtime.evaluate",
                json!({ "expression": expr, "returnByValue": true, "awaitPromise": true }),
            )
            .await?;
        Ok(r["result"]["value"].clone())
    }
}

// --- process management -----------------------------------------------------

fn spawn_host(exe_dir: &PathBuf, log_path: &PathBuf) -> anyhow::Result<Child> {
    let host = exe_dir.join("host");
    let log = std::fs::File::create(log_path)?;
    let log2 = log.try_clone()?;
    let child = Command::new(&host)
        .stdout(Stdio::from(log))
        .stderr(Stdio::from(log2))
        .spawn()
        .map_err(|e| anyhow::anyhow!("failed to spawn {}: {e}", host.display()))?;
    Ok(child)
}

fn spawn_chromium(url: &str, home: &PathBuf, profile: &PathBuf) -> anyhow::Result<Child> {
    let bin = std::env::var("CHROME_BIN").unwrap_or_else(|_| {
        let p = "/usr/lib/chromium/chromium";
        if std::path::Path::new(p).exists() {
            p.to_string()
        } else {
            "chromium".to_string()
        }
    });
    std::fs::create_dir_all(profile)?;
    let clog = std::fs::File::create(std::env::temp_dir().join("stream-loop-chrome.log"))?;
    let child = Command::new(&bin)
        .env("HOME", home)
        .args([
            "--headless=new",
            "--no-sandbox",
            "--disable-gpu",
            "--disable-dev-shm-usage",
            "--disable-crash-reporter",
            "--no-first-run",
            "--no-default-browser-check",
            &format!("--user-data-dir={}", profile.display()),
            &format!("--remote-debugging-port={CDP_PORT}"),
            "--remote-allow-origins=*",
            url,
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::from(clog))
        .spawn()
        .map_err(|e| anyhow::anyhow!("failed to spawn {bin}: {e}"))?;
    Ok(child)
}

async fn read_stream(cdp: &mut Cdp) -> anyhow::Result<Value> {
    let v = cdp.evaluate("JSON.stringify(window.__stream)").await?;
    let s = v.as_str().unwrap_or("{}");
    Ok(serde_json::from_str(s).unwrap_or(Value::Null))
}

// --- main -------------------------------------------------------------------

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let exe_dir = std::env::current_exe()?
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."));
    let crate_dir = std::env::current_dir()?;
    let page_path = crate_dir.join("web/index.html");
    let page_url = format!("file://{}", page_path.display());
    let log_path = std::env::temp_dir().join("stream-loop-host.log");

    let home = std::env::temp_dir().join("stream-loop-home");
    let profile = std::env::temp_dir().join("stream-loop-profile");
    std::fs::create_dir_all(&home)?;

    let mut host = spawn_host(&exe_dir, &log_path)?;
    if !wait_host_ready(&log_path, Duration::from_secs(20)) {
        let _ = host.kill();
        anyhow::bail!("host did not become ready; log:\n{}", read(&log_path));
    }
    println!("[smoke] host ready");

    let mut chrome = spawn_chromium(&page_url, &home, &profile)?;
    if !wait_cdp_ready(Duration::from_secs(30)) {
        let _ = chrome.kill();
        let _ = host.kill();
        anyhow::bail!("CDP did not become ready on :{CDP_PORT}");
    }
    println!("[smoke] CDP ready");

    // Find the viewer page target (retry until it appears).
    let ws_url = {
        let start = Instant::now();
        loop {
            if let Some(u) = page_ws_url_for("index.html") {
                break u;
            }
            if start.elapsed() > Duration::from_secs(15) {
                if let Some(u) = page_ws_url() {
                    break u;
                }
                let _ = chrome.kill();
                let _ = host.kill();
                anyhow::bail!("no page target found");
            }
            tokio::time::sleep(Duration::from_millis(200)).await;
        }
    };
    let mut cdp = Cdp::connect(&ws_url).await?;
    println!("[smoke] attached to page: {ws_url}");

    // Wait for frames to start flowing (presented counter advances).
    let baseline;
    let start = Instant::now();
    loop {
        let s = read_stream(&mut cdp).await?;
        let presented = s["presented"].as_u64().unwrap_or(0);
        if presented >= 5 {
            baseline = presented;
            break;
        }
        if start.elapsed() > Duration::from_secs(15) {
            let _ = chrome.kill();
            let _ = host.kill();
            anyhow::bail!("frames did not start flowing; stream={s}");
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    println!("[smoke] frames flowing (presented={baseline})");

    // Measure over ~10 s: sample fps/latency and confirm the id keeps advancing.
    let measure_start = Instant::now();
    let mut max_latency = 0.0f64;
    let mut sum_latency = 0.0f64;
    let mut lat_n = 0u64;
    let mut max_fps = 0u64;
    let mut last_id = 0u64;
    while measure_start.elapsed() < Duration::from_secs(10) {
        let s = read_stream(&mut cdp).await?;
        let lat = s["latencyMs"].as_f64().unwrap_or(0.0);
        if lat > 0.0 {
            max_latency = max_latency.max(lat);
            sum_latency += lat;
            lat_n += 1;
        }
        max_fps = max_fps.max(s["fps"].as_u64().unwrap_or(0));
        last_id = s["frameId"].as_u64().unwrap_or(0);
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
    let s_before = read_stream(&mut cdp).await?;
    let id_before = s_before["frameId"].as_u64().unwrap_or(0);
    let presented_before = s_before["presented"].as_u64().unwrap_or(0);
    println!("[smoke] before key: frameId={id_before} presented={presented_before} last_id={last_id}");

    // Send a real key press (ArrowUp) through the page via CDP Input domain.
    cdp.call(
        "Input.dispatchKeyEvent",
        json!({ "type": "keyDown", "code": "ArrowUp", "key": "ArrowUp", "windowsVirtualKeyCode": 38, "nativeVirtualKeyCode": 38 }),
    )
    .await?;
    tokio::time::sleep(Duration::from_millis(150)).await;
    cdp.call(
        "Input.dispatchKeyEvent",
        json!({ "type": "keyUp", "code": "ArrowUp", "key": "ArrowUp", "windowsVirtualKeyCode": 38, "nativeVirtualKeyCode": 38 }),
    )
    .await?;
    println!("[smoke] dispatched ArrowUp key press through the page");

    // Give the host a moment to apply the input and keep rendering.
    tokio::time::sleep(Duration::from_millis(1200)).await;
    let s_after = read_stream(&mut cdp).await?;
    let id_after = s_after["frameId"].as_u64().unwrap_or(0);
    let presented_after = s_after["presented"].as_u64().unwrap_or(0);
    let seq_sent = s_after["seq"].as_u64().unwrap_or(0);

    // Check the host log for the applied input.
    let host_log = read(&log_path);
    let input_applied = host_log.contains("input applied")
        && host_log.contains(&format!("seq={seq_sent}"));
    let frames_advanced = id_after > id_before && presented_after > presented_before;

    let avg_latency = if lat_n > 0 {
        sum_latency / lat_n as f64
    } else {
        0.0
    };

    println!("==================== SMOKE RESULT ====================");
    println!("frames presented over ~10s : {}", presented_after - baseline);
    println!("peak measured fps          : {max_fps}");
    println!("key press changed state    : {}", if input_applied { "YES" } else { "NO" });
    println!("frame id kept advancing    : {} (id {id_before} -> {id_after})", if frames_advanced { "YES" } else { "NO" });
    println!("measured latency (avg)     : {avg_latency:.2} ms");
    println!("measured latency (max)     : {max_latency:.2} ms");
    println!("======================================================");

    let _ = chrome.kill();
    let _ = host.kill();

    if !(input_applied && frames_advanced) {
        eprintln!("[smoke] FAIL: input_applied={input_applied} frames_advanced={frames_advanced}");
        eprintln!("--- host log tail ---\n{}", tail(&host_log, 40));
        anyhow::bail!("smoke assertions failed");
    }
    println!("[smoke] PASS");
    Ok(())
}

fn read(path: &PathBuf) -> String {
    std::fs::read_to_string(path).unwrap_or_default()
}

fn tail(s: &str, n: usize) -> String {
    s.lines().rev().take(n).collect::<Vec<_>>().into_iter().rev().collect::<Vec<_>>().join("\n")
}
