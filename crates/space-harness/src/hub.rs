//! [`Hub`]: the ready-made [`Control`] for a render loop.
//!
//! A render loop cannot be called into from other threads, and must never wait
//! for one. The hub is the meeting point: front ends (a Lua script, the HTTP
//! API) call the [`Control`] side from their own threads; the loop calls the
//! other side once per frame, and every call it makes takes a mutex for a few
//! instructions and returns:
//!
//! ```
//! use space_harness::{ClientState, Command, Frame, Hub};
//! # use space_harness::Control;
//!
//! let hub = Hub::new();
//! // ... hand `hub.clone()` to `space_harness::http::serve` and to scripts ...
//! let mut frame = 0u64;
//! loop {
//!     for command in hub.drain() {
//!         match command {
//!             Command::Input(_input) => { /* the client's one input path */ }
//!             Command::Connect(_address) => { /* start connecting */ }
//!             Command::Disconnect => { /* back to the connect prompt */ }
//!             Command::Quit => {}
//!         }
//!     }
//!     if hub.quit_requested() {
//!         break;
//!     }
//!     frame += 1;
//!     // ... simulate and render frame `frame` ...
//!     if hub.frame_wanted() {
//!         // Only now read the pixels back.
//!         let rgba = vec![0u8; 2 * 2 * 4];
//!         hub.publish_frame(Frame::new(frame, 2, 2, rgba).unwrap());
//!     }
//!     hub.publish_state(ClientState { frame, width: 2, height: 2, ..ClientState::default() });
//!     # hub.quit(); // (the example must end)
//! }
//! hub.close();
//! ```

use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use crate::control::{ClientState, Control, ControlError, Frame, ViewerInput};

/// Requests kept for the render loop at most. A loop that runs drains them
/// every frame; this only bounds what piles up behind one that does not.
pub const MAX_PENDING_COMMANDS: usize = 4096;

/// Longest server address accepted by [`Control::connect`], in bytes (a DNS
/// name is at most 253, plus a port).
pub const MAX_ADDRESS_LEN: usize = 260;

/// One request from a front end, for the render loop to act on.
#[derive(Clone, Debug, PartialEq)]
pub enum Command {
    /// An input event: treat it exactly like the keyboard's and the mouse's.
    Input(ViewerInput),
    /// Connect to this server address (already checked to be non-empty, of
    /// sane length and free of control characters; not that it resolves).
    Connect(String),
    /// Leave the server, back to the connect prompt.
    Disconnect,
    /// Exit. [`Hub::quit_requested`] stays true from here on.
    Quit,
}

#[derive(Debug, Default)]
struct Inner {
    commands: VecDeque<Command>,
    state: ClientState,
    frame: Option<Frame>,
    /// Somebody is waiting for a frame newer than the one kept.
    wanted: bool,
    quit: bool,
    closed: bool,
}

/// The mailbox between a render loop and its remote controls. See the module
/// docs.
#[derive(Debug, Default)]
pub struct Hub {
    inner: Mutex<Inner>,
    /// Signalled when a frame is published and when the hub closes.
    published: Condvar,
}

impl Hub {
    /// A hub with nothing queued, the default state and no frame.
    #[must_use]
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    fn lock(&self) -> MutexGuard<'_, Inner> {
        // Nothing in here can leave the data half-updated, so a panic on
        // another thread while it held the lock is no reason to stop.
        self.inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn push(&self, command: Command) -> Result<(), ControlError> {
        let mut inner = self.lock();
        if inner.closed {
            return Err(ControlError::Closed);
        }
        if inner.commands.len() >= MAX_PENDING_COMMANDS {
            return Err(ControlError::Busy);
        }
        inner.commands.push_back(command);
        Ok(())
    }

    // ------------------------------------------------- the render loop's side

    /// Take every request queued since the last call, oldest first. Call once
    /// per frame, before simulating.
    pub fn drain(&self) -> Vec<Command> {
        self.lock().commands.drain(..).collect()
    }

    /// Whether a quit was requested ([`Control::quit`]).
    pub fn quit_requested(&self) -> bool {
        self.lock().quit
    }

    /// Replace the snapshot [`Control::state`] returns. Call once per frame.
    pub fn publish_state(&self, state: ClientState) {
        self.lock().state = state;
    }

    /// Whether somebody is waiting for a picture. A loop that would otherwise
    /// not read its pixels back (a window presents straight from the GPU) does
    /// so, and calls [`publish_frame`](Self::publish_frame), only when this is
    /// true; a loop that has the pixels anyway can publish every frame.
    pub fn frame_wanted(&self) -> bool {
        self.lock().wanted
    }

    /// Hand over a rendered frame and wake whoever waits for one. The pixels
    /// are not copied.
    pub fn publish_frame(&self, frame: Frame) {
        let mut inner = self.lock();
        inner.frame = Some(frame);
        inner.wanted = false;
        drop(inner);
        self.published.notify_all();
    }

    /// The client is going away: wake every waiter and refuse further requests
    /// ([`ControlError::Closed`]). The last state stays readable.
    pub fn close(&self) {
        let mut inner = self.lock();
        inner.closed = true;
        inner.commands.clear();
        drop(inner);
        self.published.notify_all();
    }

    /// Whether [`close`](Self::close) was called.
    pub fn is_closed(&self) -> bool {
        self.lock().closed
    }
}

impl Control for Hub {
    fn input(&self, input: ViewerInput) -> Result<(), ControlError> {
        self.push(Command::Input(input))
    }

    fn connect(&self, address: &str) -> Result<(), ControlError> {
        let address = address.trim();
        let invalid = |why: &str| Err(ControlError::Invalid(format!("server address: {why}")));
        if address.is_empty() {
            return invalid("empty");
        }
        if address.len() > MAX_ADDRESS_LEN {
            return invalid("too long");
        }
        if address.chars().any(|c| c.is_control() || c.is_whitespace()) {
            return invalid("contains a space or a control character");
        }
        self.push(Command::Connect(address.to_owned()))
    }

    fn disconnect(&self) -> Result<(), ControlError> {
        self.push(Command::Disconnect)
    }

    fn state(&self) -> ClientState {
        self.lock().state.clone()
    }

    fn frame(&self, newer_than: Option<u64>, timeout: Duration) -> Result<Frame, ControlError> {
        let deadline = Instant::now().checked_add(timeout);
        let mut inner = self.lock();
        // The oldest frame number that answers the call, fixed now: "the
        // current picture" must not become a moving target while we wait.
        let floor = match newer_than {
            Some(seq) => seq.saturating_add(1),
            None => inner.state.frame,
        };
        loop {
            if let Some(frame) = inner.frame.as_ref().filter(|f| f.seq >= floor) {
                return Ok(frame.clone());
            }
            if inner.closed {
                return Err(ControlError::Closed);
            }
            let left = match deadline {
                Some(deadline) => deadline.saturating_duration_since(Instant::now()),
                // A timeout too large to add to a clock: wait in long slices.
                None => Duration::from_secs(3600),
            };
            if left.is_zero() {
                return Err(ControlError::Timeout);
            }
            inner.wanted = true;
            inner = self
                .published
                .wait_timeout(inner, left)
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .0;
        }
    }

    fn quit(&self) {
        let mut inner = self.lock();
        if inner.quit || inner.closed {
            return;
        }
        inner.quit = true;
        // The flag is what counts; the command is for loops that match on it,
        // and goes in even when the queue is full.
        inner.commands.push_back(Command::Quit);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::control::Key;

    fn frame(seq: u64) -> Frame {
        Frame::new(seq, 1, 1, vec![seq as u8, 0, 0, 255]).unwrap()
    }

    fn state_at(frame: u64) -> ClientState {
        ClientState {
            frame,
            ..ClientState::default()
        }
    }

    #[test]
    fn requests_reach_the_loop_in_order_and_once() {
        let hub = Hub::new();
        hub.key(Key::Forward, true).unwrap();
        hub.type_text("a\n").unwrap();
        hub.connect("  127.0.0.1:7777 ").unwrap();
        hub.disconnect().unwrap();
        let key = |key, down| Command::Input(ViewerInput::Key { key, down });
        assert_eq!(
            hub.drain(),
            [
                key(Key::Forward, true),
                Command::Input(ViewerInput::Text('a')),
                key(Key::Enter, true),
                key(Key::Enter, false),
                Command::Connect("127.0.0.1:7777".into()),
                Command::Disconnect,
            ]
        );
        assert!(hub.drain().is_empty());
    }

    #[test]
    fn impossible_addresses_are_refused_before_they_reach_the_loop() {
        let hub = Hub::new();
        let long = "a".repeat(MAX_ADDRESS_LEN + 1);
        for bad in [
            "",
            "   ",
            "host name:1",
            "host\n:1",
            "h\u{1b}[2J",
            long.as_str(),
        ] {
            let err = hub.connect(bad).unwrap_err();
            assert!(matches!(err, ControlError::Invalid(_)), "{bad:?}: {err}");
        }
        assert!(hub.drain().is_empty());
        hub.connect(&"a".repeat(MAX_ADDRESS_LEN)).unwrap();
        hub.connect("[::1]:7777").unwrap();
        assert_eq!(hub.drain().len(), 2);
    }

    /// A loop that stopped draining cannot be made to hold an unbounded queue:
    /// input is refused, loudly, so a script fails instead of typing into the
    /// void.
    #[test]
    fn a_stalled_loop_makes_input_busy_instead_of_growing_the_queue() {
        let hub = Hub::new();
        for _ in 0..MAX_PENDING_COMMANDS {
            hub.input(ViewerInput::Text('x')).unwrap();
        }
        assert_eq!(hub.input(ViewerInput::Text('y')), Err(ControlError::Busy));
        assert_eq!(hub.connect("host:1"), Err(ControlError::Busy));
        // Quit still gets through.
        hub.quit();
        hub.quit();
        assert!(hub.quit_requested());
        let drained = hub.drain();
        assert_eq!(drained.len(), MAX_PENDING_COMMANDS + 1);
        assert_eq!(drained.last(), Some(&Command::Quit));
        hub.input(ViewerInput::Text('z')).unwrap();
    }

    #[test]
    fn the_state_is_the_last_one_published() {
        let hub = Hub::new();
        assert_eq!(hub.state(), ClientState::default());
        hub.publish_state(state_at(3));
        hub.publish_state(state_at(4));
        assert_eq!(hub.state().frame, 4);
    }

    #[test]
    fn a_frame_that_is_already_there_is_returned_at_once() {
        let hub = Hub::new();
        hub.publish_frame(frame(5));
        hub.publish_state(state_at(5));
        let started = Instant::now();
        assert_eq!(hub.frame(None, Duration::from_secs(30)).unwrap().seq, 5);
        assert_eq!(hub.frame(Some(4), Duration::from_secs(30)).unwrap().seq, 5);
        assert!(started.elapsed() < Duration::from_secs(5));
        assert!(!hub.frame_wanted(), "nobody had to wait");
    }

    #[test]
    fn waiting_for_a_frame_is_bounded_and_asks_the_loop_for_one() {
        let hub = Hub::new();
        let started = Instant::now();
        assert_eq!(
            hub.frame(None, Duration::from_millis(60)),
            Err(ControlError::Timeout)
        );
        assert!(started.elapsed() >= Duration::from_millis(60));
        assert!(hub.frame_wanted(), "the loop is asked to read pixels back");
        assert_eq!(hub.frame(None, Duration::ZERO), Err(ControlError::Timeout));
        // A frame that is not newer than asked for does not answer the call.
        hub.publish_frame(frame(5));
        assert!(!hub.frame_wanted());
        assert_eq!(
            hub.frame(Some(5), Duration::from_millis(20)),
            Err(ControlError::Timeout)
        );
        assert_eq!(
            hub.frame(Some(u64::MAX), Duration::from_millis(20)),
            Err(ControlError::Timeout)
        );
        // An absurd timeout does not overflow the clock.
        assert_eq!(hub.frame(Some(4), Duration::MAX).unwrap().seq, 5);
    }

    /// "The current picture" from a loop that reads pixels back only on demand:
    /// the frame kept from an earlier request is stale, so the call waits for
    /// the loop to publish a fresh one.
    #[test]
    fn a_stale_frame_is_not_the_current_picture() {
        let hub = Hub::new();
        hub.publish_frame(frame(5));
        hub.publish_state(state_at(90));
        let waiter = {
            let hub = Arc::clone(&hub);
            std::thread::spawn(move || hub.frame(None, Duration::from_secs(20)))
        };
        // The loop: sees the request at its next frame, renders, publishes.
        let started = Instant::now();
        while !hub.frame_wanted() {
            assert!(started.elapsed() < Duration::from_secs(10), "never asked");
            std::thread::sleep(Duration::from_millis(1));
        }
        hub.publish_frame(frame(91));
        hub.publish_state(state_at(91));
        assert_eq!(waiter.join().unwrap().unwrap().seq, 91);
    }

    #[test]
    fn many_waiters_are_all_woken_by_one_frame() {
        let hub = Hub::new();
        let waiters: Vec<_> = (0..8)
            .map(|_| {
                let hub = Arc::clone(&hub);
                std::thread::spawn(move || hub.frame(Some(0), Duration::from_secs(20)))
            })
            .collect();
        let started = Instant::now();
        while !hub.frame_wanted() {
            assert!(started.elapsed() < Duration::from_secs(10), "never asked");
            std::thread::sleep(Duration::from_millis(1));
        }
        hub.publish_frame(frame(1));
        for waiter in waiters {
            let got = waiter.join().unwrap().unwrap();
            assert_eq!(got.seq, 1);
        }
    }

    #[test]
    fn closing_wakes_waiters_and_refuses_requests() {
        let hub = Hub::new();
        hub.publish_state(state_at(2));
        let waiter = {
            let hub = Arc::clone(&hub);
            std::thread::spawn(move || hub.frame(Some(100), Duration::from_secs(20)))
        };
        let started = Instant::now();
        while !hub.frame_wanted() {
            assert!(started.elapsed() < Duration::from_secs(10), "never asked");
            std::thread::sleep(Duration::from_millis(1));
        }
        hub.input(ViewerInput::Text('x')).unwrap();
        hub.close();
        assert_eq!(waiter.join().unwrap(), Err(ControlError::Closed));
        assert!(hub.is_closed());
        assert_eq!(hub.input(ViewerInput::Text('x')), Err(ControlError::Closed));
        assert_eq!(hub.connect("host:1"), Err(ControlError::Closed));
        assert_eq!(hub.disconnect(), Err(ControlError::Closed));
        assert_eq!(
            hub.frame(None, Duration::from_secs(20)),
            Err(ControlError::Closed)
        );
        assert!(hub.drain().is_empty(), "nothing is left for a dead loop");
        hub.quit();
        assert!(hub.drain().is_empty());
        // The last state can still be read.
        assert_eq!(hub.state().frame, 2);
    }
}
