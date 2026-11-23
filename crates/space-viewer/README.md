# space-viewer

Thin wasm viewer for `space-client --stream`: draws streamed JPEG frames on a
canvas and sends mouse/keyboard input back. It holds no game state; the player
in the multiplayer sense is the native proxy (`space-client --stream`), which
connects to the server through the normal `space-net` path.

It is a **local development and testing tool** (people and AI agents use it to
watch and drive the game in a browser); it is not part of any deployment.

## Build and run

```sh
cargo run -p space-client -- --stream 127.0.0.1:8099 --server 127.0.0.1:7777
cargo run -p xtask -- viewer            # builds the wasm + JS glue, serves the page
# open http://127.0.0.1:8097/
```

`xtask viewer` runs the wasm build and `wasm-bindgen`, puts `index.html` and
`pkg/` together in `<target dir>/viewer-site` and serves that on loopback with
`Cache-Control: no-cache`. It needs the `wasm-bindgen` CLI in exactly the
version `Cargo.lock` pins: if it is missing or different the command exits 2
and prints the `cargo install` line to run (`--install-bindgen` runs it).
`--port N`, `--ws URL` (print the link with `?ws=`), `--site-only` (build the
site and exit), `--no-build`.

The page connects to `ws://<host the page was loaded from>:8099` (so the lines
above reach `ws://127.0.0.1:8099`). Pass `?ws=ws://HOST:PORT` to name another
address.

Without `--server` the client shows its connect screen in the page: click the
picture, type the shard's address and press Enter. While playing: WASD / arrows
move, Space up, Shift down, left-drag looks around, `C` camera, `B` brake
assist, `H` HUD, Esc back to the connect screen. Browser shortcuts (F5, F12,
Ctrl/Cmd combinations) stay with the browser.

## Typing

Text reaches the game through an invisible, always-focused text field
(`#space-viewer-input`), so it works from a keyboard (any layout, AltGr, dead
keys), an input method (the committed text), an on-screen keyboard, paste, and
browser automation that inserts text without key events (`Input.insertText`,
`execCommand('insertText')`, or filling that element and dispatching `input`).
Each character is sent once. Inserted `\n` is Enter and `\t` is Tab; pasted
newlines are dropped and a paste is cut at 256 characters. Movement keys and
Enter/Backspace/Delete/caret keys still come from key events; a synthetic
`keydown` alone types nothing, as with any text field.

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
