//! The headless front end: the application rendered offscreen in a loop at a
//! capped rate, with no window.
//!
//! This is what `--headless` runs (for an agent or a CI job that drives the
//! client through the harness) and, with a [`StreamClient`], what
//! `--stream ADDR` runs: each frame is also JPEG-encoded and pushed to the
//! browser viewers, and their input is fed into the application's one input
//! path ([`App::input`]) — on the connect screen their typing edits the
//! address, while playing their keys fly the ship.
//!
//! # Who can end it
//!
//! Nothing that arrives as input: the keys of this loop come from whoever can
//! reach the stream port or the harness, not from the person who started the
//! process, so the application is built for it without
//! [`crate::app::AppConfig::escape_quits`] (the default) and Esc on its
//! connect screen ends nothing. The loop ends when the harness asks for it
//! ([`space_harness::Hub::quit`]), after [`HeadlessConfig::frames`] frames, or
//! when the caller sets `stop`.
//!
//! # Multiplayer seam
//!
//! A browser viewer is not a peer of the authoritative server. The process
//! running this loop is the player ([`crate::link`]); the viewer is only its
//! remote screen and controller.

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use space_render::{RenderContext, RenderError};
use space_stream::protocol::{FLAG_DOWNSCALED, FLAG_KEYFRAME};
use space_stream::{JpegEncoder, StreamClient};

use crate::app::App;

/// The frame rate the loop is capped at unless told otherwise.
pub const DEFAULT_FPS: u32 = 30;

/// Poll interval while nobody is watching (see
/// [`HeadlessConfig::idle_unwatched`]).
const IDLE_POLL: Duration = Duration::from_millis(50);

/// How the headless loop runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeadlessConfig {
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Frames per second at most (0 is taken as 1).
    pub fps: u32,
    /// Stop after this many rendered frames.
    pub frames: Option<u64>,
    /// With a stream: render nothing while no viewer is connected and nobody
    /// has asked the harness for a frame. Set when only viewers can be
    /// watching (no harness API, no script), so an unwatched stream costs no
    /// GPU time; the application still follows the network.
    pub idle_unwatched: bool,
}

impl Default for HeadlessConfig {
    fn default() -> Self {
        Self {
            width: 640,
            height: 480,
            fps: DEFAULT_FPS,
            frames: None,
            idle_unwatched: false,
        }
    }
}

/// What a run of the loop did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HeadlessReport {
    /// Frames rendered.
    pub frames: u64,
    /// Frames pushed to viewers.
    pub streamed: u64,
    /// Vulkan validation messages recorded while rendering.
    pub validation_messages: u64,
}

/// Run `app` offscreen until it wants to quit, `stop` is set or
/// `config.frames` frames were rendered.
///
/// Every rendered frame is published to the harness ([`App::finish_frame`]);
/// with a `stream` it is also sent to the connected viewers, whose input is
/// applied before each frame. The caller still owns `app` afterwards (and
/// shuts it down by dropping it).
///
/// # Errors
///
/// [`RenderError`] when a frame cannot be rendered or encoded.
pub fn run_headless(
    ctx: &RenderContext,
    app: &mut App,
    config: &HeadlessConfig,
    stream: Option<&StreamClient>,
    stop: &AtomicBool,
) -> Result<HeadlessReport, RenderError> {
    app.set_render_device(ctx.device_name());
    let encoder = JpegEncoder::default();
    let mut flags = FLAG_KEYFRAME;
    if encoder.downscale() > 1 {
        flags |= FLAG_DOWNSCALED;
    }
    let budget = Duration::from_secs_f64(1.0 / f64::from(config.fps.max(1)));
    let (width, height) = (config.width, config.height);
    let mut report = HeadlessReport::default();
    let mut had_viewers = false;
    while !stop.load(Ordering::Relaxed) && config.frames.is_none_or(|limit| report.frames < limit) {
        let start = Instant::now();
        let mut viewers = 0;
        if let Some(stream) = stream {
            for event in stream.poll_input() {
                app.input(event);
            }
            viewers = stream.viewer_count();
            // The last viewer left: nobody can send its key-ups any more, so
            // a ship would keep thrusting. (Once, on the way out — the
            // harness may hold keys while no viewer is connected.)
            if viewers == 0 && had_viewers {
                app.release_keys();
            }
            had_viewers = viewers > 0;
        }
        app.update();
        if app.should_quit() {
            break;
        }
        if config.idle_unwatched && stream.is_some() && viewers == 0 && !app.frame_wanted() {
            std::thread::sleep(IDLE_POLL);
            continue;
        }
        let options = app.render_options();
        let out = ctx.render_offscreen(app.scene(width, height), width, height, &options)?;
        report.validation_messages += u64::from(out.validation_messages);
        if let Some(stream) = stream.filter(|_| viewers > 0) {
            let jpeg = encoder
                .encode(&out.pixels, out.width, out.height)
                .map_err(|e| RenderError::new(format!("stream: {e}")))?;
            stream.push_jpeg(jpeg.bytes, flags);
            report.streamed += 1;
        }
        app.finish_frame(out.width, out.height, Some(out.pixels));
        report.frames += 1;
        if let Some(rest) = budget.checked_sub(start.elapsed()) {
            std::thread::sleep(rest);
        }
    }
    Ok(report)
}
