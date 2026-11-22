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
# open http://127.0.0.1:8000/index.html?ws=ws://127.0.0.1:8099
```

Controls: WASD / arrows move, Space up, Shift down, left-drag looks around.

`ws://` only. `wss://` would pull `ring` (the sanctioned C/asm exception) and is
deferred; put a TLS-terminating proxy in front if needed.
