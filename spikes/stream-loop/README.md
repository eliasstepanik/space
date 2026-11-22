# Spike 0.4 — end-to-end pixel-streaming loop (THROWAWAY)

Proves the full transport loop for the `space` remote-rendering product:

```
ash offscreen render  ->  pure-Rust JPEG encode  ->  binary WebSocket
        ^                                                        |
        |                                                        v
   JSON key input  <----------------------------------  browser <canvas>
```

A native `host` renders a moving coloured quad offscreen (real AMD iGPU via
RADV in this sandbox), reads the frame back, JPEG-encodes it, and pushes it to
a browser page over a WebSocket. The page draws it on a `<canvas>` and sends
key presses back; the host applies them (ArrowUp speeds the quad up) and keeps
streaming. A `smoke` driver launches headless Chromium over CDP, drives the
page, and asserts the loop works end to end.

This is a **throwaway spike**: it proves the transport, not render quality.

## What it proves

- ash can render offscreen (no window/surface) on the sandbox iGPU and read
  frames back to CPU memory.
- A **pure-Rust** JPEG encoder produces browser-decodable frames with no C
  bindings.
- A binary WebSocket carries frames with latest-frame-wins backpressure.
- Browser key input reaches the render loop and changes rendered state.
- The loop is measurable end to end and the transport adds ~1 ms on loopback
  (see "Measured numbers" for exactly what that number does and does not
  include).

## WebSocket frame protocol

Binary messages, one per frame:

```
[ 8 bytes: frame_id (u64, big-endian) ][ JPEG bytes ... ]
```

Immediately before each binary frame the host also sends a small JSON **text**
message carrying the host wall-clock timestamp, used only for latency
measurement (the binary frame format itself is unchanged):

```
{"type":"meta","id":<frame_id>,"t":<unix_ms>}
```

Input messages (JSON text, browser -> host):

```
{"type":"key","code":"ArrowUp","down":true,"seq":123}
```

`code` is the physical `event.code`. `ArrowUp` down => quad advances faster,
up => slow. `seq` is tracked and logged as the last applied input sequence.

## Backpressure

Latest-frame-wins: the host keeps at most **one** queued frame per socket. If
the socket is behind, the older queued frame is dropped and logged
(`backpressure: dropped frame_id(s) ...`). This keeps latency bounded under load.

The queue is a `tokio::sync::watch` channel, deliberately **not** a
`Notify` + `Mutex<Option<_>>` pair. The first version of this spike used the
latter and had a **lost-wakeup bug**: `notify_one`/`notify_waiters` store no
permit, and `take()` awaited the notification *before* checking the slot, so a
frame pushed between the slot check and the await was only picked up on the
*next* push — a slow-but-alive client could stall indefinitely. `watch` keeps
the newest value in the channel itself, so a consumer that is busy sending
finds the newer frame when it next calls `changed()`. Two unit tests in
`src/bin/host.rs` (`push_before_consumer_waits_is_not_lost`,
`takes_are_sequential_and_latest_wins`) pin this behaviour.

Accepted sockets also get `TCP_NODELAY`, so small frame messages are never
delayed by Nagle coalescing.

## Pure-Rust JPEG crate

`jpeg-encoder = "0.7.1"` (pure Rust, no C bindings). Encoded at quality 70,
RGBA input. Typical frame ~6.7 KB, encode ~3.5 ms at 640x480.

## Shader compilation

WGSL is compiled to SPIR-V with the pure-Rust `naga` crate (`30.0`, features
`wgsl-in` + `spv-out`) — no glslang/C.

## Measured numbers (this sandbox, RADV GFX1151 iGPU, headless Chromium)

From a `smoke` run (re-run after the backpressure fix):

```
frames presented over ~10s : 342
peak measured fps          : 30
key press changed state    : YES
frame id kept advancing    : YES (id 319 -> 360)
measured latency (avg)     : 1.13 ms
measured latency (max)     : 2.00 ms
```

Host per-frame log (typical): `render_ms ~5-12`, `encode_ms ~3.5-4`,
`jpeg_bytes ~6.7-7.2 KB`. The 30 fps cap is the host's `TARGET_FRAME_MS` (33 ms).

**What the ~1 ms figure actually measures.** It is
`browser_arrival - host_produce_time` on the **same machine** (loopback), where
`host_produce_time` is the wall clock stamped *after* the JPEG encode. So it
covers only the WebSocket hop plus the browser's **JPEG decode + `<canvas>`
draw**, and nothing else:

- it **excludes** the host's render (`render_ms`, ~5–12 ms here) and encode
  (`encode_ms`, ~3.5–4 ms) time, which happen *before* the timestamp;
- it **excludes** any real network path (loopback only), the client's own
  presentation/scheduling delay, and everything upstream of the capture;
- the browser clock is `Date.now()`, i.e. **whole-millisecond resolution**, so
  the reported values are quantised to 1 ms steps (an "avg 1.00 ms" is
  consistent with anything in roughly 0.5–1.5 ms).

So ~1 ms is a *transport + decode/draw* datapoint, **not** an end-to-end
capture-to-on-screen latency. Nothing in this spike measures the U10 budget.

**The U10 budget (≤80 ms p95, capture-to-on-screen, 720p60, JPEG) is a
Phase-6 obligation**, not something this spike satisfies. Measuring it requires
timestamps carried in the frame header from the capture point through encode,
transport, decode and on-screen presentation; the Phase-6 frame header must
carry those timestamps so p95 can be computed on the real path.

## How to run

```sh
export LD_LIBRARY_PATH=/workspace/.local/vulkan/usr/lib/x86_64-linux-gnu
export VK_DRIVER_FILES=/workspace/.local/vulkan/usr/share/vulkan/icd.d/radeon_icd.json:/workspace/.local/vulkan/usr/share/vulkan/icd.d/lvp_icd.json

cargo run --release --bin host     # serves ws://127.0.0.1:8099
cargo run --release --bin smoke    # drives host + headless Chromium, exits 0 on pass
```

`smoke` starts the host itself, launches headless Chromium (CDP on loopback
:9333), opens `web/index.html`, waits for frames, sends an ArrowUp through the
page, and asserts the host applied it and frames keep advancing.

### Mutation check

Disabling the input handler (early-return in `apply_input`) makes `smoke`
**FAIL** with `input_applied=false` (exit 1), confirming the test actually
observes the input path.

## Environment notes (sandbox)

- Headless Chromium is the Debian build at `/usr/lib/chromium/chromium`. It
  needs a **writable `HOME`** (crashpad writes its database under `$HOME`);
  the sandbox default `HOME=/home/sandbox` is not writable, so `smoke` sets
  `HOME` to a temp dir. `--remote-allow-origins=*` is required for CDP WS.
- The CDP HTTP endpoint keeps connections open even with `Connection: close`,
  so `smoke` reads by `Content-Length` rather than to EOF.

## What the real implementation must change

- **Viewer**: replace this plain HTML/JS page with the Rust->wasm viewer
  (decision QA). The transport protocol can stay the same.
- **Codec**: **v1 is JPEG-only** (spike 0.5 found the Vulkan H.264 encode path
  unusable on this hardware — VCN hang, see `../vkvideo-h264/README.md`). A
  Vulkan H.264 encode stays an *optional* fast path gated on an end-to-end
  encode probe, so the frame header should already reserve room for
  keyframe/delta flags and the Phase-6 latency timestamps (see "Measured
  numbers").
- **Frame header / latency**: move the host timestamp from the side-channel
  JSON text message into the **binary frame header**, and add the upstream
  capture timestamp, so p95 capture-to-on-screen (U10) is measurable on the
  real path.
- **Auth**: bind with a token / auth handshake. This spike has **no auth**.
- **Binding**: keep the loopback bind (127.0.0.1) for the local host; any
  remote exposure needs TLS (wss) and token auth, not the raw ws here.
- **Backpressure/transport**: real path needs multiple clients, congestion
  control, and frame-type-aware dropping (drop deltas, keep keyframes).
- **Latency**: measure over a real network path, not loopback, and from the
  capture point (not post-encode) — see the U10 note under "Measured numbers".

## Files

- `src/bin/host.rs` — render loop, JPEG encode, WebSocket server, input, backpressure.
- `src/renderer.rs` — minimal ash offscreen renderer (WGSL via naga).
- `src/bin/smoke.rs` — CDP-driven end-to-end test.
- `web/index.html` — minimal viewer page (transport proof only).
- `Cargo.toml` — standalone package (empty `[workspace]`, not a root member).
