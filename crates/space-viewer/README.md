# space-viewer

Thin wasm viewer for `space-client --stream`: draws streamed JPEG frames on a
canvas and sends mouse/keyboard input back. It holds no game state; the player
in the multiplayer sense is the native proxy (`space-client --stream`), which
connects to the server through the normal `space-net` path.

## Build

```sh
rustup target add wasm32-unknown-unknown
cargo build --release --target wasm32-unknown-unknown -p space-viewer

# JS glue (needs wasm-bindgen-cli matching the lockfile, 0.2.130):
cargo install wasm-bindgen-cli --version 0.2.130
wasm-bindgen --target web --out-dir crates/space-viewer/pkg \
    target/wasm32-unknown-unknown/release/space_viewer.wasm
```

## Run

```sh
cargo run -p space-client -- --stream 127.0.0.1:8099 --galaxy
cd crates/space-viewer && python3 -m http.server 8000
# open http://127.0.0.1:8000/index.html
```

The page connects to `ws://<host the page was loaded from>:8099` (so the line
above reaches `ws://127.0.0.1:8099`, and a page served from another machine
reaches the proxy on that machine). Pass `?ws=ws://HOST:PORT` to name another
address.

Controls: WASD / arrows move, Space up, Shift down, left-drag looks around.

## Behaviour

- **Status line.** The element with id `status` shows the connection state
  (connecting, waiting for the first frame, no connection / retrying); it is
  empty while frames are flowing.
- **Reconnect.** When the connection drops or cannot be made, the viewer retries
  after 0.5 s, doubling up to 8 s, until it is back. Keys still held down are
  pressed again on the new connection. An address the browser refuses outright
  (malformed, or `ws://` from an `https://` page) is reported and not retried.
- **Picture size.** The canvas takes the pixel size of the streamed frame and is
  scaled to the window with its aspect ratio kept (letterboxed), so any
  `--size` is shown undistorted. Pointer positions are relative to the picture.
- **Looking around.** A drag is sent as a *relative* rotation; the proxy owns the
  view direction, so reloading the page or a second viewer does not make the
  camera jump.
- **Frames.** One image decode is in flight at a time; frames that arrive
  meanwhile collapse to the newest, so a slow viewer skips frames instead of
  falling behind.

`ws://` only. `wss://` would pull `ring` (the sanctioned C/asm exception) and is
deferred; put a TLS-terminating proxy in front if needed.
