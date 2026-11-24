//! `space-harness` — drive a running client without a human and without a
//! window.
//!
//! An AI coding agent (or a CI job) cannot look at a window or press keys. This
//! crate gives a client a control surface that can: everything a user can do
//! and see, behind one trait, with two front ends.
//!
//! * [`Control`] ([`control`]) — the trait: input (the same
//!   [`ViewerInput`] the browser viewer sends, so the window, the viewer, the
//!   HTTP API and Lua all feed one input path), connect / disconnect, a state
//!   snapshot ([`ClientState`]), the rendered picture ([`Frame`]), quit.
//! * [`Hub`] ([`hub`]) — the implementation for a render loop: front ends call
//!   it from their threads, the loop drains it once per frame and never waits.
//! * [`script`] — Lua scripts that simulate a user (`client.type(...)`,
//!   `client.wait_until(...)`, `client.screenshot(...)`); a failing `assert`
//!   is a failed test. Sandboxed, time-limited, stoppable.
//! * [`http`] — a small HTTP API on loopback: `GET /state`, `GET /frame.png`,
//!   `POST /input`, `POST /script`, ... (`GET /` lists the routes).
//! * [`image`] — PNG (dependency-free, lossless) and JPEG encoding of a frame.
//! * [`fake`] — an in-memory client for tests of everything above.
//!
//! # Plugging a client in
//!
//! ```no_run
//! use std::sync::Arc;
//! use space_harness::{http, script, Control, Hub};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let hub = Hub::new();
//! let control: Arc<dyn Control> = hub.clone();
//!
//! // The HTTP API (the binary owns the flags; loopback needs no token).
//! let runtime = tokio::runtime::Runtime::new()?;
//! let listener = runtime.block_on(tokio::net::TcpListener::bind("127.0.0.1:8098"))?;
//! runtime.spawn(http::serve(listener, control.clone(), http::HttpConfig::default()));
//!
//! // A test script, on its own thread; its outcome is the exit code.
//! let source = std::fs::read_to_string("test.lua")?;
//! let script = std::thread::spawn(move || {
//!     script::run_script(control, &source, &script::ScriptConfig::default(), &script::StopHandle::new())
//! });
//!
//! // The render loop: see the example in [`hub`]. It ends when the script
//! // (or `POST /quit`) asked for it.
//! # let _ = (hub, script);
//! # Ok(())
//! # }
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs, missing_debug_implementations)]

pub mod control;
pub mod fake;
pub mod http;
pub mod hub;
pub mod image;
pub mod script;

pub use control::{
    key_from_name, key_name, key_names, BodyState, ClientState, Control, ControlError, Frame, Key,
    Screen, SkyBodyState, ViewerInput,
};
pub use hub::{Command, Hub};
pub use script::{
    run_script, ScriptConfig, ScriptError, ScriptErrorKind, ScriptReport, StopHandle,
};
