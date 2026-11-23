//! `space-viewer` — the thin wasm browser viewer for `space-stream`.
//!
//! It owns no game state. [`web::start`] opens a WebSocket to a `space-client
//! --stream` proxy, draws the JPEG frames it receives on a `<canvas>`
//! (`createImageBitmap` on a `Blob`), and sends mouse/keyboard input back as
//! [`space_stream::ViewerInput`] messages. When the connection drops it says so
//! in a status line and reconnects with a backoff.
//!
//! # Keyboard and text
//!
//! The viewer only sees pixels: it cannot know whether the proxy is flying a
//! ship or showing a text field (the connect prompt). So it does not decide
//! what a key press means; it reports everything the press could be and the
//! proxy uses what its current screen wants. There are two sources, and each
//! thing has exactly one:
//!
//! * **Key events** (`keydown` / `keyup`, [`key_action`]) are the source of
//!   the *movement keys* — by physical position (`KeyboardEvent.code`: WASD,
//!   arrows, Space, Shift), held with a down and an up — and of the
//!   *text-entry keys* (`KeyboardEvent.key`: Enter, Backspace, Escape, Tab,
//!   Delete, Home, End, the caret arrows), each a stroke. They are **not** a
//!   source of text.
//! * **The text field** ([`TextField`]) is the source of all *text*. The page
//!   keeps an invisible `<textarea>` focused and lets the browser type into
//!   it; whatever ends up in it is sent, one [`ViewerInput::Text`] per
//!   character, and taken out again. Because the browser does the typing, text
//!   arrives the same way whatever produced it: a physical keyboard under any
//!   layout (dead keys and AltGr included), an input method (the committed
//!   text, once the composition ends), a phone's on-screen keyboard, a paste,
//!   or an automation tool that inserts text without any key event (CDP
//!   `Input.insertText`, `document.execCommand('insertText')`, or setting the
//!   field's `value` and dispatching `input`). And because every character is
//!   taken out of the field exactly once, it is sent exactly once.
//!
//! A printable key that is also a movement key (`W`, Space) does both: its
//! keydown sends the movement key and is left to the browser, which types the
//! character into the field. An on-screen keyboard reports most keys as
//! "Unidentified" (key code 229): its Backspace, Delete and Enter then show up
//! as edits of the field and become the same strokes ([`field_messages`]).
//! A key an input method is handling carries the same key code; what it is
//! called is not trusted then, but where it is still is, so the movement keys
//! keep working with an input method switched on ([`ime_key_action`]).
//!
//! The field is kept focused: it takes the focus when the page loads, when
//! the window gets it, on every click or tap, and back whenever it loses it —
//! unless another text control of the page (an embedding page's own input)
//! has it, which is left alone, keys and all.
//!
//! Keys the viewer uses and that type nothing are kept from the browser
//! (`preventDefault`: an arrow key does not scroll, Tab does not move focus,
//! Backspace does not navigate back). Shortcuts are left alone: nothing is
//! sent or prevented while Ctrl or the Meta/Command key is held (Ctrl+R,
//! Ctrl+L, Ctrl+V, Ctrl+W, Cmd+Q), and function keys (F5, F12) are not keys
//! the viewer knows.
//!
//! # Multiplayer seam
//!
//! The viewer never talks to the authoritative game server. Its input goes to the
//! headless native proxy client (`space-client --stream`), which *is* the player
//! as far as the server is concerned: it connects through the ordinary
//! `space-net` client path and injects this viewer's input into its own input
//! state, so it reaches the server over the existing netcode. No new server
//! protocol exists. See the `space-stream` crate docs.
//!
//! The proxy also owns the view direction: the viewer only reports how far the
//! mouse was dragged ([`turn_message`]) and never an absolute angle, so a
//! reload or a second viewer cannot make the camera jump.
//!
//! Transport is plain `ws://` (see `space-stream` for why `wss://` is deferred).
//!
//! The pure logic (input serialisation, the text field's bookkeeping, layout
//! maths, the frame slot, reconnect timing, status text) is at the top of this
//! file and unit-tested on the host; the browser glue is in [`web`].
#![forbid(unsafe_code)]

use space_stream::protocol::{is_typeable, Key, ViewerInput};

/// Radians of view rotation per pixel of mouse drag.
pub const RADIANS_PER_PIXEL: f32 = 0.0025;
/// The port `space-client --stream` is conventionally bound to.
pub const DEFAULT_STREAM_PORT: u16 = 8099;
/// Delay before the first reconnect attempt, in milliseconds.
pub const RECONNECT_BASE_MS: u32 = 500;
/// Longest delay between reconnect attempts, in milliseconds.
pub const RECONNECT_MAX_MS: u32 = 8_000;
/// Most characters of one paste that are sent (the rest is dropped): a paste
/// is meant for a server address, not for a document.
pub const MAX_PASTE_CHARS: usize = 256;
/// Most characters taken from one edit of the text field (one `input` event,
/// one committed composition, one automation insert); the rest is dropped.
pub const MAX_INPUT_CHARS: usize = MAX_PASTE_CHARS;
/// What the text field holds on each side of its caret while it rests: a
/// zero-width space. An on-screen keyboard deletes by asking the field to
/// remove the character next to the caret and sends no usable key event, so
/// there has to be a character there for its Backspace / Delete to show up at
/// all. A zero-width space is one no keyboard types.
pub const FIELD_SENTINEL: char = '\u{200b}';
/// The resting content of the text field: a sentinel, the caret, a sentinel.
pub const FIELD_REST: &str = "\u{200b}\u{200b}";
/// Where the caret rests in [`FIELD_REST`] (in UTF-16 code units).
pub const FIELD_REST_CARET: u32 = 1;

/// Wire message for a DOM key event, or `None` if the key is not mapped.
pub fn key_message(dom_code: &str, down: bool) -> Option<Vec<u8>> {
    let key = Key::from_dom_code(dom_code)?;
    Some(ViewerInput::Key { key, down }.encode())
}

/// Key-up wire messages for every held DOM key code (sent on window blur, when
/// the browser will not deliver the real key-ups).
pub fn release_messages<'a>(held: impl IntoIterator<Item = &'a String>) -> Vec<Vec<u8>> {
    held.into_iter()
        .filter_map(|code| key_message(code, false))
        .collect()
}

/// Key-down wire messages for every held DOM key code (sent when a connection
/// opens: the keys the user is still holding after a reconnect, which the new
/// connection has never heard about and the browser will not report again).
pub fn press_messages<'a>(held: impl IntoIterator<Item = &'a String>) -> Vec<Vec<u8>> {
    held.into_iter()
        .filter_map(|code| key_message(code, true))
        .collect()
}

/// The modifier keys held during a keyboard event (`KeyboardEvent.ctrlKey`,
/// `.altKey`, `.metaKey`). Shift is not here: it only changes which character
/// a key types, and `KeyboardEvent.key` already says which.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Modifiers {
    /// Control.
    pub ctrl: bool,
    /// Alt (Option on a Mac).
    pub alt: bool,
    /// The Windows / Command key.
    pub meta: bool,
}

impl Modifiers {
    /// Whether a key pressed with these modifiers is a shortcut for the
    /// browser or the OS rather than input for the page: Meta/Command with
    /// anything, or Control without Alt. Control *with* Alt is how Windows
    /// reports AltGr, which types characters (`@`, `{`, `\` on many layouts).
    #[must_use]
    pub const fn is_shortcut(self) -> bool {
        self.meta || (self.ctrl && !self.alt)
    }
}

/// The character a key press is expected to type, from `KeyboardEvent.key`,
/// or `None` if it types none: a named key (`"Enter"`, `"Shift"`, `"F5"`,
/// `"Dead"`, `"Process"` — every value of more than one character), a control
/// character, or a shortcut ([`Modifiers::is_shortcut`]).
///
/// This is **not** where text comes from — the browser types into the text
/// field and [`TextField`] reads it there. It only tells [`key_action`] that
/// a key press must be left to the browser, so that it can type.
///
/// Alt alone is ambiguous: on Windows and Linux Alt+letter is a menu
/// accelerator and `key` is still the plain letter, while on a Mac Option
/// composes characters (Option+L is `@` on a German layout). So with Alt
/// alone, an ASCII letter or digit is not expected to type and anything else
/// is.
#[must_use]
pub fn typed_char(key: &str, modifiers: Modifiers) -> Option<char> {
    let mut chars = key.chars();
    let (Some(c), None) = (chars.next(), chars.next()) else {
        return None;
    };
    if !is_typeable(c) || modifiers.is_shortcut() {
        return None;
    }
    if modifiers.alt && !modifiers.ctrl && c.is_ascii_alphanumeric() {
        return None;
    }
    Some(c)
}

/// What to do about one `keydown` / `keyup` event: see [`key_action`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct KeyAction {
    /// Wire messages to send, in order.
    pub messages: Vec<Vec<u8>>,
    /// Call `preventDefault()`: the viewer used the key and it types nothing,
    /// so the browser must not act on it (scroll, move focus, navigate back,
    /// move the text field's caret, delete in it).
    pub prevent_default: bool,
    /// The event's `code` is a movement key that is now held (`Some(true)`) or
    /// released (`Some(false)`): remember it, so it can be released on blur
    /// and pressed again after a reconnect.
    pub held: Option<bool>,
}

/// Whether a `keydown` belongs to an input method rather than to the page:
/// the event says a composition is in progress (`KeyboardEvent.isComposing`),
/// or it carries the "processing" key code 229 that browsers use for every key
/// an input method or an on-screen keyboard handles (Safari reports the Enter
/// that confirms a composition that way, *after* the composition has ended).
/// Such a keydown gets [`ime_key_action`]; what it types arrives through the
/// text field.
#[must_use]
pub const fn is_ime_keydown(is_composing: bool, key_code: u32) -> bool {
    is_composing || key_code == 229
}

/// What to do about a `keydown` that an input method is handling
/// ([`is_ime_keydown`]).
///
/// Its `key` says nothing usable (`"Process"`, `"Unidentified"`, or — Safari —
/// the Enter that has already confirmed a composition), so no stroke is read
/// from it, and it is never prevented: the input method needs it. What it does
/// still say is where the key is. A movement key (by `code`) is held as usual,
/// so that the ship can be flown with an input method switched on; the keyup
/// that releases it is an ordinary one. An on-screen keyboard reports no
/// `code`, so nothing is sent for its keys here.
#[must_use]
pub fn ime_key_action(code: &str, modifiers: Modifiers, repeat: bool) -> KeyAction {
    let mut action = KeyAction::default();
    if repeat || modifiers.is_shortcut() {
        return action;
    }
    if let Some(key) = Key::from_dom_code(code) {
        let down = true;
        action
            .messages
            .push(ViewerInput::Key { key, down }.encode());
        action.held = Some(true);
    }
    action
}

/// Decide what one keyboard event means on the wire.
///
/// `code` and `key` are `KeyboardEvent.code` (the physical key) and
/// `KeyboardEvent.key` (what it types or names under the current layout);
/// `down` is keydown vs keyup and `repeat` is `KeyboardEvent.repeat`.
///
/// A **keydown** sends, in this order, whichever apply:
///
/// 1. the movement key of `code`, down — not for an auto-repeat (a held key
///    is held once);
/// 2. the text-entry key of `key` as a stroke (down, up) — also for an
///    auto-repeat, which is one more stroke.
///
/// It never sends text: the character a key types reaches the proxy through
/// the text field ([`TextField`]), once. For that to happen a key that types
/// ([`typed_char`]) is **not** prevented, even when it is a movement key too
/// (`W`, Space); every other key the viewer uses is.
///
/// A shortcut ([`Modifiers::is_shortcut`]) sends nothing and is not prevented.
/// A **keyup** sends the movement key of `code`, up, whatever the modifiers
/// are by then (a key released while Ctrl is down must still be released), and
/// nothing else: strokes have no up of their own.
#[must_use]
pub fn key_action(
    code: &str,
    key: &str,
    modifiers: Modifiers,
    down: bool,
    repeat: bool,
) -> KeyAction {
    let mut action = KeyAction::default();
    let movement = Key::from_dom_code(code);
    if !down {
        if let Some(key) = movement {
            action
                .messages
                .push(ViewerInput::Key { key, down }.encode());
            action.held = Some(false);
            action.prevent_default = true;
        }
        return action;
    }
    if modifiers.is_shortcut() {
        return action;
    }
    let types = typed_char(key, modifiers).is_some();
    if let Some(key) = movement {
        // Also for auto-repeat: a held arrow key must not scroll. But a key
        // that types is left alone, or its character would never be typed.
        action.prevent_default = !types;
        if !repeat {
            action
                .messages
                .push(ViewerInput::Key { key, down }.encode());
            action.held = Some(true);
        }
    }
    if let Some(key) = Key::from_dom_key(key) {
        // Prevented, so the browser does not also edit the text field (a
        // Backspace would delete in it and be seen a second time there).
        action.prevent_default = true;
        for down in [true, false] {
            action
                .messages
                .push(ViewerInput::Key { key, down }.encode());
        }
    }
    action
}

/// Wire messages for pasted `text`: one `Text` per character, in order.
/// Control characters (line breaks and tabs included: a pasted address must
/// not submit itself) are dropped, and at most [`MAX_PASTE_CHARS`] characters
/// are sent.
#[must_use]
pub fn paste_messages(text: &str) -> Vec<Vec<u8>> {
    text.chars()
        .filter_map(ViewerInput::text)
        .take(MAX_PASTE_CHARS)
        .map(|input| input.encode())
        .collect()
}

/// What kind of edit an `input` event reports (`InputEvent.inputType`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EditKind {
    /// Text was put in: typing, an input method, an automation tool, or an
    /// edit of unknown kind.
    Insert,
    /// Text was put in from the clipboard or by a drop.
    Paste,
    /// Something before the caret was deleted.
    DeleteBackward,
    /// Something after the caret was deleted.
    DeleteForward,
    /// Not input: undo / redo, cut, formatting.
    Ignore,
}

fn edit_kind(input_type: &str) -> EditKind {
    match input_type {
        "historyUndo" | "historyRedo" | "deleteByCut" | "deleteByDrag" => EditKind::Ignore,
        "insertFromPaste" | "insertFromPasteAsQuotation" | "insertFromDrop" | "insertFromYank" => {
            EditKind::Paste
        }
        "deleteContentForward"
        | "deleteWordForward"
        | "deleteSoftLineForward"
        | "deleteHardLineForward" => EditKind::DeleteForward,
        other if other.starts_with("delete") => EditKind::DeleteBackward,
        other if other.starts_with("format") => EditKind::Ignore,
        _ => EditKind::Insert,
    }
}

/// Wire messages for what an edit left in the text field: `value` is the
/// field's content after the edit (it rested on [`FIELD_REST`] before), and
/// `input_type` the `InputEvent.inputType` that reported it (empty when
/// unknown, e.g. for a plain `input` event an automation tool dispatched).
///
/// * Every character that is not a sentinel was put in, wherever it stands:
///   it is sent as `Text`, in order. A line break is the Enter stroke and a
///   tab the Tab stroke (an on-screen keyboard's Enter is an `insertLineBreak`
///   edit; `"host:port\n"` from an automation tool types and submits). Other
///   control characters are dropped. At most [`MAX_INPUT_CHARS`] are taken.
/// * A paste or a drop (`insertFromPaste`, ...) follows the paste rules
///   instead: line breaks and tabs are dropped too, so that a pasted address
///   does not submit itself.
/// * If nothing was put in and a sentinel is missing, something was deleted:
///   that is the Backspace stroke, or the Delete stroke for a forward
///   deletion (`deleteContentForward`, ...). This is how an on-screen
///   keyboard's Backspace arrives. A field that was simply emptied by an edit
///   that is not a deletion (`value = ""` and a plain `input` event) is
///   nothing.
/// * Undo / redo and cut are not input and send nothing.
#[must_use]
pub fn field_messages(value: &str, input_type: &str) -> Vec<Vec<u8>> {
    let kind = edit_kind(input_type);
    if kind == EditKind::Ignore {
        return Vec::new();
    }
    let mut inputs = Vec::new();
    let stroke = |inputs: &mut Vec<ViewerInput>, key: Key| {
        inputs.extend([true, false].map(|down| ViewerInput::Key { key, down }));
    };
    let mut taken = 0;
    for c in value.chars().filter(|c| *c != FIELD_SENTINEL) {
        if taken == MAX_INPUT_CHARS {
            break;
        }
        match c {
            '\n' if kind != EditKind::Paste => stroke(&mut inputs, Key::Enter),
            '\t' if kind != EditKind::Paste => stroke(&mut inputs, Key::Tab),
            c if is_typeable(c) => inputs.push(ViewerInput::Text(c)),
            _ => continue,
        }
        taken += 1;
    }
    if inputs.is_empty() {
        let sentinels = value.chars().filter(|c| *c == FIELD_SENTINEL).count();
        match kind {
            EditKind::DeleteForward if sentinels < 2 => stroke(&mut inputs, Key::Delete),
            EditKind::DeleteBackward if sentinels < 2 => stroke(&mut inputs, Key::Backspace),
            // An edit that says it inserted (a composition an on-screen
            // keyboard used to delete with), or says nothing at all, and that
            // took exactly one sentinel away: the character before the caret
            // was deleted.
            EditKind::Insert if sentinels == 1 => stroke(&mut inputs, Key::Backspace),
            _ => {}
        }
    }
    inputs.iter().map(ViewerInput::encode).collect()
}

/// What the page must do after an event on the text field: see [`TextField`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FieldEdit {
    /// Wire messages to send, in order.
    pub messages: Vec<Vec<u8>>,
    /// Put the field back to rest ([`FIELD_REST`], caret at
    /// [`FIELD_REST_CARET`]): its content was taken.
    pub reset: bool,
}

/// The bookkeeping of the page's invisible text field, the single source of
/// typed text (see the crate docs).
///
/// The field's content is the truth: an edit puts characters in, the viewer
/// sends what is there and puts the field back to rest. A character is taken
/// out exactly once, so it is sent exactly once, however many events
/// (`keydown`, `beforeinput`, `input`, `compositionend`, in whichever order a
/// browser fires them) spoke of it. The one thing to keep track of is whether
/// an input method is composing: until the composition ends the field holds
/// its unfinished text, which is neither sent nor touched.
///
/// The page calls [`input`](Self::input) for every `input` event,
/// [`composition_started`](Self::composition_started) /
/// [`composition_ended`](Self::composition_ended) for those events, and
/// [`flush`](Self::flush) shortly after a composition ended and before it
/// sends a key (so that an Enter cannot overtake the word it confirms).
#[derive(Clone, Debug, Default)]
pub struct TextField {
    composing: bool,
}

impl TextField {
    /// Whether an input method is composing in the field.
    #[must_use]
    pub const fn is_composing(&self) -> bool {
        self.composing
    }

    /// `compositionstart`: the field now holds unfinished text.
    pub fn composition_started(&mut self) {
        self.composing = true;
    }

    /// `compositionend`: the text in the field is final. It is collected by
    /// the `input` event some browsers fire next, or by the next
    /// [`flush`](Self::flush) — not here: the field must not be changed while
    /// the browser is still finishing (or, as with Korean, already starting
    /// the next) composition.
    pub fn composition_ended(&mut self) {
        self.composing = false;
    }

    /// An `input` event on the field: `value` is the field's content now,
    /// `input_type` the event's `inputType` (empty if it has none) and
    /// `event_is_composing` its `isComposing`.
    pub fn input(&mut self, value: &str, input_type: &str, event_is_composing: bool) -> FieldEdit {
        if self.composing || event_is_composing {
            return FieldEdit::default();
        }
        Self::take(value, input_type)
    }

    /// Collect whatever the field holds beyond its resting content, unless an
    /// input method is composing.
    pub fn flush(&mut self, value: &str) -> FieldEdit {
        if self.composing {
            return FieldEdit::default();
        }
        Self::take(value, "")
    }

    fn take(value: &str, input_type: &str) -> FieldEdit {
        if value == FIELD_REST {
            return FieldEdit::default();
        }
        FieldEdit {
            messages: field_messages(value, input_type),
            reset: true,
        }
    }
}

/// Whether an element takes text itself, so that the viewer must not pull the
/// keyboard focus away from it or treat what is typed into it as game input:
/// `tag_name` is `Element.tagName`, `content_editable` is
/// `HTMLElement.isContentEditable`. (The viewer's own text field is not asked
/// about.)
#[must_use]
pub fn is_text_control(tag_name: &str, content_editable: bool) -> bool {
    content_editable
        || ["INPUT", "TEXTAREA", "SELECT"]
            .iter()
            .any(|tag| tag.eq_ignore_ascii_case(tag_name))
}

/// Wire message for a pointer at CSS-pixel `(x, y)` inside a `width`×`height`
/// rectangle, normalised to `0..=1` (clamped). `None` for an empty rect.
pub fn pointer_message(x: f64, y: f64, width: f64, height: f64, buttons: u8) -> Option<Vec<u8>> {
    if !(width > 0.0 && height > 0.0) || !x.is_finite() || !y.is_finite() {
        return None;
    }
    Some(
        ViewerInput::Pointer {
            x: (x / width).clamp(0.0, 1.0) as f32,
            y: (y / height).clamp(0.0, 1.0) as f32,
            buttons,
        }
        .encode(),
    )
}

/// Wire message for a mouse drag of `(dx, dy)` pixels: a *relative* rotation
/// (right = turn right, down = look down). `None` when there is nothing to
/// send (no movement, or a non-finite delta).
pub fn turn_message(dx: f64, dy: f64) -> Option<Vec<u8>> {
    let radians = |pixels: f64| {
        let radians = -(pixels as f32) * RADIANS_PER_PIXEL;
        // An unmoved axis is `-0.0` here; put a plain zero on the wire.
        if radians == 0.0 {
            0.0
        } else {
            radians
        }
    };
    let (dyaw, dpitch) = (radians(dx), radians(dy));
    if !dyaw.is_finite() || !dpitch.is_finite() || (dyaw == 0.0 && dpitch == 0.0) {
        return None;
    }
    Some(ViewerInput::Turn { dyaw, dpitch }.encode())
}

/// Where an `image_w`×`image_h` picture ends up inside a `box_w`×`box_h`
/// element that shows it with `object-fit: contain` (scaled to fit, aspect
/// kept, centred): `(left, top, width, height)` relative to the element.
///
/// The canvas element fills the window while the frame keeps its own aspect
/// ratio, so pointer positions must be measured against this rectangle, not
/// against the element. `None` if any size is zero, negative or not finite.
pub fn contained_rect(
    box_w: f64,
    box_h: f64,
    image_w: f64,
    image_h: f64,
) -> Option<(f64, f64, f64, f64)> {
    let usable = |v: f64| v.is_finite() && v > 0.0;
    if !(usable(box_w) && usable(box_h) && usable(image_w) && usable(image_h)) {
        return None;
    }
    let scale = (box_w / image_w).min(box_h / image_h);
    let (width, height) = (image_w * scale, image_h * scale);
    Some(((box_w - width) / 2.0, (box_h - height) / 2.0, width, height))
}

/// The stream address to use when the page was given none: the proxy on the
/// host the page itself was loaded from (`location.hostname`, which keeps the
/// brackets of an IPv6 literal), port [`DEFAULT_STREAM_PORT`]. Falls back to
/// loopback when the page has no host (`file://`).
pub fn default_stream_url(page_hostname: &str) -> String {
    let host = match page_hostname.trim() {
        "" => "127.0.0.1",
        host => host,
    };
    format!("ws://{host}:{DEFAULT_STREAM_PORT}")
}

/// How long to wait before reconnect attempt number `attempt` (0 = the first
/// one after a drop): doubling from [`RECONNECT_BASE_MS`], capped at
/// [`RECONNECT_MAX_MS`].
pub fn reconnect_delay_ms(attempt: u32) -> u32 {
    let doublings = 1u32.checked_shl(attempt).unwrap_or(u32::MAX);
    RECONNECT_BASE_MS
        .saturating_mul(doublings)
        .min(RECONNECT_MAX_MS)
}

/// The connection state shown in the page's status line.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Link {
    /// A WebSocket is being opened.
    Connecting,
    /// Connected; no frame has been drawn yet.
    Waiting,
    /// Frames are being drawn.
    Live,
    /// The connection is down; the next attempt is in `delay_ms`.
    Retrying { delay_ms: u32 },
    /// The address cannot be opened at all (not retried).
    Failed,
}

/// The text of the status line for `link` (empty while frames are flowing, so
/// nothing covers the picture).
pub fn status_line(link: Link, url: &str) -> String {
    match link {
        Link::Connecting => format!("connecting to {url} ..."),
        Link::Waiting => format!("connected to {url}, waiting for the first frame"),
        Link::Live => String::new(),
        Link::Retrying { delay_ms } => format!(
            "no connection to {url}; retrying in {:.1} s",
            f64::from(delay_ms) / 1000.0
        ),
        Link::Failed => format!("cannot open {url} (expected a ws://host:port address)"),
    }
}

/// What to do when a decode finishes: see [`FrameSlot::settle`].
#[derive(Debug, PartialEq, Eq)]
pub struct Settled<T> {
    /// Draw the image that was just decoded.
    pub draw: bool,
    /// Start decoding this frame next.
    pub next: Option<T>,
}

/// Latest-frame-wins bookkeeping for image decoding.
///
/// Decoding a JPEG is asynchronous. At most **one** decode is in flight; a
/// frame that arrives meanwhile waits in a single slot and replaces whatever
/// was waiting there, so a viewer that decodes slower than frames arrive skips
/// frames instead of piling up decodes (and their memory) without bound.
///
/// `T` is whatever is needed to start a decode (the JPEG bytes).
#[derive(Debug)]
pub struct FrameSlot<T> {
    /// Sequence number of the newest frame accepted on this connection.
    newest: u64,
    /// A decode is in flight.
    decoding: bool,
    /// The decode in flight belongs to a connection that has since ended.
    stale: bool,
    /// The newest frame that arrived while a decode was in flight.
    waiting: Option<T>,
}

impl<T> Default for FrameSlot<T> {
    fn default() -> Self {
        Self {
            newest: 0,
            decoding: false,
            stale: false,
            waiting: None,
        }
    }
}

impl<T> FrameSlot<T> {
    /// A frame with sequence number `seq` arrived. Returns the frame back if
    /// its decode should start now; otherwise it was kept as the waiting frame
    /// (replacing an older one) or, if it is not newer than what was already
    /// accepted, dropped.
    pub fn offer(&mut self, seq: u64, frame: T) -> Option<T> {
        if seq <= self.newest {
            return None;
        }
        self.newest = seq;
        if self.decoding {
            self.waiting = Some(frame);
            None
        } else {
            self.decoding = true;
            Some(frame)
        }
    }

    /// The decode in flight finished (successfully or not). Says whether its
    /// image should be drawn and which frame to decode next, if one waited.
    pub fn settle(&mut self) -> Settled<T> {
        let draw = self.decoding && !self.stale;
        self.stale = false;
        let next = self.waiting.take();
        self.decoding = next.is_some();
        Settled { draw, next }
    }

    /// The connection ended: forget the waiting frame, start sequence numbers
    /// over (a restarted proxy counts from 1 again) and make sure a decode
    /// still in flight is not drawn over the next connection's frames.
    pub fn reset(&mut self) {
        self.newest = 0;
        self.waiting = None;
        self.stale = self.decoding;
    }

    /// Sequence number of the newest frame accepted since the last reset.
    pub fn newest(&self) -> u64 {
        self.newest
    }
}

pub mod web {
    //! Browser glue (WebSocket, canvas, DOM events). Only meaningful in a browser.

    use std::cell::{Cell, RefCell};
    use std::collections::BTreeSet;
    use std::rc::Rc;

    use space_stream::protocol::{decode_frame, FRAME_HEADER_LEN};
    use wasm_bindgen::prelude::*;
    use wasm_bindgen::JsCast;
    use web_sys::{
        BinaryType, Blob, BlobPropertyBag, CanvasRenderingContext2d, ClipboardEvent, Document,
        Element, Event, EventTarget, FocusOptions, HtmlCanvasElement, HtmlElement,
        HtmlTextAreaElement, ImageBitmap, InputEvent, KeyboardEvent, MessageEvent, MouseEvent,
        WebSocket, Window,
    };

    use super::{
        contained_rect, default_stream_url, ime_key_action, is_ime_keydown, is_text_control,
        key_action, paste_messages, pointer_message, press_messages, reconnect_delay_ms,
        release_messages, status_line, turn_message, FieldEdit, FrameSlot, Link, Modifiers,
        TextField, FIELD_REST, FIELD_REST_CARET,
    };

    /// Id of the (optional) element whose text shows the connection status.
    const STATUS_ELEMENT_ID: &str = "status";
    /// Id of the text field the viewer adds to the page (see [`TextField`]):
    /// the element an automation tool can type into or fill.
    pub const TEXT_FIELD_ID: &str = "space-viewer-input";

    /// The text field is there to hold the focus, not to be seen: one
    /// invisible pixel in the corner. It is inside the viewport (a field
    /// outside it makes browsers scroll to it) and its font is 16 px (below
    /// that, iOS zooms the page when a field is focused).
    const TEXT_FIELD_STYLE: &str = "position:fixed;left:0;top:0;width:1px;height:1px;\
        margin:0;padding:0;border:0;outline:none;resize:none;overflow:hidden;opacity:0;\
        color:transparent;background:transparent;caret-color:transparent;font-size:16px;\
        pointer-events:none;z-index:-1";

    /// Nothing is to be corrected, completed or capitalised in it.
    const TEXT_FIELD_ATTRIBUTES: [(&str, &str); 10] = [
        ("id", TEXT_FIELD_ID),
        ("aria-label", "Keyboard input for the game view"),
        ("rows", "1"),
        ("tabindex", "-1"),
        ("autocapitalize", "off"),
        ("autocomplete", "off"),
        ("autocorrect", "off"),
        ("spellcheck", "false"),
        ("enterkeyhint", "send"),
        ("style", TEXT_FIELD_STYLE),
    ];

    /// Everything the event handlers share. It lives as long as the page: the
    /// handlers hold it and it holds the handlers.
    struct Viewer {
        window: Window,
        document: Document,
        canvas: HtmlCanvasElement,
        /// The invisible, focused field the browser types into.
        field: HtmlTextAreaElement,
        /// What is known about the field's state (composing or not).
        text: RefCell<TextField>,
        ctx: CanvasRenderingContext2d,
        status: Option<Element>,
        url: String,
        /// The current connection (`None` between a close and the next attempt).
        socket: RefCell<Option<WebSocket>>,
        /// Frames waiting to be decoded; the payload is the JPEG as a view into
        /// the WebSocket message's `ArrayBuffer`.
        frames: RefCell<FrameSlot<js_sys::Uint8Array>>,
        /// Reconnect attempts since the last successful open.
        attempt: Cell<u32>,
        /// A frame of the current connection has been drawn.
        live: Cell<bool>,
        /// DOM codes of the mapped keys currently held down.
        held: RefCell<BTreeSet<String>>,
        handlers: RefCell<Option<Handlers>>,
    }

    /// The callbacks that are attached again and again (to every new socket,
    /// every decode, every retry timer). They are created once and reused;
    /// nothing is allocated or leaked per frame.
    struct Handlers {
        open: Closure<dyn FnMut()>,
        message: Closure<dyn FnMut(MessageEvent)>,
        close: Closure<dyn FnMut()>,
        decoded: Closure<dyn FnMut(JsValue)>,
        decode_failed: Closure<dyn FnMut(JsValue)>,
        retry: Closure<dyn FnMut()>,
        /// Collects the text field's content shortly after a composition.
        flush: Closure<dyn FnMut()>,
        /// Gives the focus back to the text field after it lost it.
        refocus: Closure<dyn FnMut()>,
    }

    impl Viewer {
        fn set_status(&self, link: Link) {
            if let Some(status) = &self.status {
                status.set_text_content(Some(&status_line(link, &self.url)));
            }
        }

        /// Send one input message if the connection is open (input while
        /// disconnected is dropped; held keys are re-sent on reconnect).
        fn send(&self, msg: &[u8]) {
            if let Some(socket) = self.socket.borrow().as_ref() {
                if socket.ready_state() == WebSocket::OPEN {
                    let _ = socket.send_with_u8_array(msg);
                }
            }
        }

        /// Open a new connection. An `Err` means the address itself was
        /// refused by the browser (malformed, or blocked), which a retry would
        /// not fix.
        fn connect(&self) -> Result<(), JsValue> {
            self.set_status(Link::Connecting);
            let socket = match WebSocket::new(&self.url) {
                Ok(socket) => socket,
                Err(error) => {
                    self.set_status(Link::Failed);
                    return Err(error);
                }
            };
            socket.set_binary_type(BinaryType::Arraybuffer);
            if let Some(handlers) = self.handlers.borrow().as_ref() {
                socket.set_onopen(Some(handlers.open.as_ref().unchecked_ref()));
                socket.set_onmessage(Some(handlers.message.as_ref().unchecked_ref()));
                // A failed connection attempt also ends in a close event, so
                // this one handler covers "never connected" and "dropped".
                socket.set_onclose(Some(handlers.close.as_ref().unchecked_ref()));
            }
            *self.socket.borrow_mut() = Some(socket);
            Ok(())
        }

        fn on_open(&self) {
            self.attempt.set(0);
            self.set_status(Link::Waiting);
            // The new connection knows nothing about keys that were already
            // down, and the browser reports a held key only once.
            for msg in press_messages(self.held.borrow().iter()) {
                self.send(&msg);
            }
        }

        fn on_close(&self) {
            self.socket.borrow_mut().take();
            self.frames.borrow_mut().reset();
            self.live.set(false);
            let attempt = self.attempt.get();
            self.attempt.set(attempt.saturating_add(1));
            let delay_ms = reconnect_delay_ms(attempt);
            self.set_status(Link::Retrying { delay_ms });
            if let Some(handlers) = self.handlers.borrow().as_ref() {
                let _ = self
                    .window
                    .set_timeout_with_callback_and_timeout_and_arguments_0(
                        handlers.retry.as_ref().unchecked_ref(),
                        i32::try_from(delay_ms).unwrap_or(i32::MAX),
                    );
            }
        }

        /// A frame arrived: read its header and queue the JPEG for decoding.
        fn on_message(&self, event: &MessageEvent) {
            let Ok(buffer) = event.data().dyn_into::<js_sys::ArrayBuffer>() else {
                return;
            };
            let header_len = FRAME_HEADER_LEN as u32;
            if buffer.byte_length() < header_len {
                return;
            }
            // Only the header is copied into wasm memory. The JPEG stays where
            // it is (a view into the message's buffer) until it is handed to
            // the decoder, and is never copied at all if it gets skipped.
            let mut head = [0u8; FRAME_HEADER_LEN];
            js_sys::Uint8Array::new_with_byte_offset_and_length(&buffer, 0, header_len)
                .copy_to(&mut head);
            let Ok((header, _)) = decode_frame(&head) else {
                return;
            };
            let jpeg = js_sys::Uint8Array::new_with_byte_offset(&buffer, header_len);
            let start = self.frames.borrow_mut().offer(header.seq, jpeg);
            if let Some(jpeg) = start {
                self.decode(jpeg);
            }
        }

        /// Start decoding `jpeg` (the frame slot already counts it as in
        /// flight). If that is impossible, move on to the frame that waited.
        fn decode(&self, jpeg: js_sys::Uint8Array) {
            let mut next = Some(jpeg);
            while let Some(jpeg) = next {
                if self.begin_decode(&jpeg) {
                    return;
                }
                next = self.frames.borrow_mut().settle().next;
            }
        }

        /// Hand `jpeg` to the browser's image decoder. True if a decode is now
        /// in flight (one of the `decoded` / `decode_failed` handlers will run).
        fn begin_decode(&self, jpeg: &js_sys::Uint8Array) -> bool {
            let handlers = self.handlers.borrow();
            let Some(handlers) = handlers.as_ref() else {
                return false;
            };
            let options = BlobPropertyBag::new();
            options.set_type("image/jpeg");
            let parts = js_sys::Array::of1(jpeg);
            let Ok(blob) = Blob::new_with_u8_array_sequence_and_options(&parts, &options) else {
                return false;
            };
            let Ok(promise) = self.window.create_image_bitmap_with_blob(&blob) else {
                return false;
            };
            let _ = promise.then2(&handlers.decoded, &handlers.decode_failed);
            true
        }

        /// The decode in flight finished; `bitmap` is `None` if it failed.
        fn on_decoded(&self, bitmap: Option<ImageBitmap>) {
            let settled = self.frames.borrow_mut().settle();
            if let Some(bitmap) = bitmap {
                if settled.draw {
                    self.draw(&bitmap);
                }
                bitmap.close();
            }
            if let Some(jpeg) = settled.next {
                self.decode(jpeg);
            }
        }

        /// Draw a decoded frame. The canvas takes the frame's own pixel size
        /// (CSS scales it to the window with the aspect ratio kept), so a
        /// stream that is not 4:3 is not stretched.
        fn draw(&self, bitmap: &ImageBitmap) {
            let (width, height) = (bitmap.width(), bitmap.height());
            if width == 0 || height == 0 {
                return;
            }
            if self.canvas.width() != width || self.canvas.height() != height {
                self.canvas.set_width(width);
                self.canvas.set_height(height);
            }
            let _ = self.ctx.draw_image_with_image_bitmap(bitmap, 0.0, 0.0);
            if !self.live.replace(true) {
                self.set_status(Link::Live);
            }
        }

        /// Run one of the reusable callbacks as soon as the current event is
        /// done with.
        fn soon(&self, pick: impl FnOnce(&Handlers) -> &Closure<dyn FnMut()>) {
            if let Some(handlers) = self.handlers.borrow().as_ref() {
                let _ = self
                    .window
                    .set_timeout_with_callback_and_timeout_and_arguments_0(
                        pick(handlers).as_ref().unchecked_ref(),
                        0,
                    );
            }
        }

        /// Put the text field back to rest: the sentinels, the caret between
        /// them.
        fn rest_field(&self) {
            self.field.set_value(FIELD_REST);
            let _ = self
                .field
                .set_selection_range(FIELD_REST_CARET, FIELD_REST_CARET);
        }

        /// Send what an edit of the text field amounts to, and reset the field
        /// if its content was taken.
        fn apply(&self, edit: &FieldEdit) {
            for msg in &edit.messages {
                self.send(msg);
            }
            if edit.reset {
                self.rest_field();
            }
        }

        /// The browser changed the text field (a key typed, an input method
        /// committed, an on-screen keyboard deleted, a tool inserted text).
        fn on_field_input(&self, event: &Event) {
            // A tool may dispatch a plain `Event`: then nothing is known about
            // the edit but the field's content, which is enough.
            let input = event.dyn_ref::<InputEvent>();
            let composing = input.is_some_and(InputEvent::is_composing);
            let input_type = input.map(InputEvent::input_type).unwrap_or_default();
            let edit = self
                .text
                .borrow_mut()
                .input(&self.field.value(), &input_type, composing);
            self.apply(&edit);
        }

        /// Collect what the text field holds (nothing while composing).
        fn flush_field(&self) {
            let edit = self.text.borrow_mut().flush(&self.field.value());
            self.apply(&edit);
        }

        fn on_composition(&self, started: bool) {
            if started {
                self.text.borrow_mut().composition_started();
            } else {
                self.text.borrow_mut().composition_ended();
                // Not now: see `TextField::composition_ended`.
                self.soon(|handlers| &handlers.flush);
            }
        }

        /// Whether `element` is a control of the page that takes text itself
        /// (not the viewer's own field).
        fn is_foreign_control(&self, element: &Element) -> bool {
            let ours: &Element = &self.field;
            if element == ours {
                return false;
            }
            let editable = element
                .dyn_ref::<HtmlElement>()
                .is_some_and(HtmlElement::is_content_editable);
            is_text_control(&element.tag_name(), editable)
        }

        /// Give the keyboard focus to the text field, unless another text
        /// control of the page has it. Without scrolling: the field is in the
        /// corner, the picture is not.
        fn focus_field(&self) {
            let elsewhere = self
                .document
                .active_element()
                .is_some_and(|active| self.is_foreign_control(&active));
            if elsewhere {
                return;
            }
            let options = FocusOptions::new();
            options.set_prevent_scroll(true);
            let _ = self.field.focus_with_options(&options);
        }

        /// Whether no element of the page has the keyboard focus (it is on the
        /// page itself).
        fn nothing_is_focused(&self) -> bool {
            self.document.active_element().is_none_or(|active| {
                let tag = active.tag_name();
                tag.eq_ignore_ascii_case("body") || tag.eq_ignore_ascii_case("html")
            })
        }

        /// A click or tap anywhere but on another text control: the keyboard
        /// belongs to the game again. (On a phone this is what brings the
        /// on-screen keyboard up: a field may only be focused from a tap.)
        fn on_pointer_up(&self, event: &Event) {
            let target = event.target().and_then(|t| t.dyn_into::<Element>().ok());
            if !target.is_some_and(|target| self.is_foreign_control(&target)) {
                self.focus_field();
            }
        }

        /// The text field lost the focus (a click on the picture moves it to
        /// the page): take it back, if the page still has it at all.
        fn refocus(&self) {
            if self.document.has_focus().unwrap_or(false) {
                self.focus_field();
            }
        }

        fn on_key(&self, event: &KeyboardEvent, down: bool) {
            if down {
                // Typed into a text control of the embedding page: not ours.
                let target = event.target().and_then(|t| t.dyn_into::<Element>().ok());
                if target.is_some_and(|target| self.is_foreign_control(&target)) {
                    return;
                }
            }
            let code = event.code();
            let modifiers = Modifiers {
                ctrl: event.ctrl_key(),
                alt: event.alt_key(),
                meta: event.meta_key(),
            };
            let action = if down && is_ime_keydown(event.is_composing(), event.key_code()) {
                // An input method or an on-screen keyboard is handling this
                // key; what it types arrives through the text field.
                ime_key_action(&code, modifiers, event.repeat())
            } else {
                if down && self.nothing_is_focused() {
                    // The key is about to type: into the text field, then.
                    self.focus_field();
                }
                // Text still in the field was typed before this key: it goes
                // first (an Enter must not overtake the word it confirms).
                self.flush_field();
                key_action(&code, &event.key(), modifiers, down, event.repeat())
            };
            if action.prevent_default {
                event.prevent_default();
            }
            match action.held {
                Some(true) => {
                    self.held.borrow_mut().insert(code);
                }
                Some(false) => {
                    self.held.borrow_mut().remove(&code);
                }
                None => {}
            }
            for msg in &action.messages {
                self.send(msg);
            }
        }

        /// The user pasted (Ctrl+V, the menu, ...): type the clipboard's text.
        /// The paste is prevented, so the browser does not also put it into
        /// the text field. (If the event carries no clipboard data it is left
        /// alone, and the text arrives through the field as a paste edit.)
        fn on_paste(&self, event: &ClipboardEvent) {
            let Some(text) = event
                .clipboard_data()
                .and_then(|data| data.get_data("text/plain").ok())
            else {
                return;
            };
            event.prevent_default();
            // What was typed before the paste goes first.
            self.flush_field();
            for msg in paste_messages(&text) {
                self.send(&msg);
            }
        }

        /// The window lost focus: the browser will not deliver the key-ups.
        fn on_blur(&self) {
            let held = std::mem::take(&mut *self.held.borrow_mut());
            for msg in release_messages(held.iter()) {
                self.send(&msg);
            }
        }

        /// Pointer position always; dragging with the left button also turns
        /// the view.
        fn on_mouse_move(&self, event: &MouseEvent) {
            let buttons = (event.buttons() & 0xff) as u8;
            let element = self.canvas.get_bounding_client_rect();
            let picture = contained_rect(
                element.width(),
                element.height(),
                f64::from(self.canvas.width()),
                f64::from(self.canvas.height()),
            );
            if let Some((left, top, width, height)) = picture {
                let x = f64::from(event.client_x()) - element.left() - left;
                let y = f64::from(event.client_y()) - element.top() - top;
                if let Some(msg) = pointer_message(x, y, width, height, buttons) {
                    self.send(&msg);
                }
            }
            if buttons & 1 != 0 {
                let (dx, dy) = (event.movement_x(), event.movement_y());
                if let Some(msg) = turn_message(f64::from(dx), f64::from(dy)) {
                    self.send(&msg);
                }
            }
        }
    }

    /// The stream address for a page loaded from `page_hostname` that was not
    /// given one explicitly (see [`default_stream_url`]).
    #[wasm_bindgen]
    pub fn default_ws_url(page_hostname: &str) -> String {
        default_stream_url(page_hostname)
    }

    /// Add the viewer's text field to the page.
    fn create_text_field(document: &Document) -> Result<HtmlTextAreaElement, JsValue> {
        let field: HtmlTextAreaElement = document.create_element("textarea")?.dyn_into()?;
        for (name, value) in TEXT_FIELD_ATTRIBUTES {
            field.set_attribute(name, value)?;
        }
        document.body().ok_or("no body")?.append_child(&field)?;
        Ok(field)
    }

    /// Register a page-lifetime listener (never removed).
    fn listen(
        target: &EventTarget,
        event: &str,
        handler: impl FnMut(Event) + 'static,
    ) -> Result<(), JsValue> {
        let handler = Closure::<dyn FnMut(Event)>::new(handler);
        target.add_event_listener_with_callback(event, handler.as_ref().unchecked_ref())?;
        handler.forget();
        Ok(())
    }

    /// Connect the canvas with id `canvas_id` to the stream at `ws_url`
    /// (e.g. `ws://127.0.0.1:8099`) and keep it connected. Call once from JS
    /// after the wasm loads. If the page has an element with id `status`, the
    /// connection state is written into it.
    ///
    /// This also adds an invisible `<textarea id="space-viewer-input">` to the
    /// page and keeps it focused: it is where all typed text comes from (see
    /// the crate docs).
    #[wasm_bindgen]
    pub fn start(canvas_id: &str, ws_url: &str) -> Result<(), JsValue> {
        let window = web_sys::window().ok_or("no window")?;
        let document = window.document().ok_or("no document")?;
        let canvas: HtmlCanvasElement = document
            .get_element_by_id(canvas_id)
            .ok_or("canvas not found")?
            .dyn_into()?;
        let ctx: CanvasRenderingContext2d = canvas
            .get_context("2d")?
            .ok_or("no 2d context")?
            .dyn_into()?;
        let status = document.get_element_by_id(STATUS_ELEMENT_ID);
        let field = create_text_field(&document)?;

        let viewer = Rc::new(Viewer {
            window: window.clone(),
            document,
            canvas: canvas.clone(),
            field,
            text: RefCell::new(TextField::default()),
            ctx,
            status,
            url: ws_url.to_owned(),
            socket: RefCell::new(None),
            frames: RefCell::new(FrameSlot::default()),
            attempt: Cell::new(0),
            live: Cell::new(false),
            held: RefCell::new(BTreeSet::new()),
            handlers: RefCell::new(None),
        });

        let handlers = {
            let v = &viewer;
            Handlers {
                open: Closure::<dyn FnMut()>::new({
                    let v = Rc::clone(v);
                    move || v.on_open()
                }),
                message: Closure::<dyn FnMut(MessageEvent)>::new({
                    let v = Rc::clone(v);
                    move |event: MessageEvent| v.on_message(&event)
                }),
                close: Closure::<dyn FnMut()>::new({
                    let v = Rc::clone(v);
                    move || v.on_close()
                }),
                decoded: Closure::<dyn FnMut(JsValue)>::new({
                    let v = Rc::clone(v);
                    move |bitmap: JsValue| v.on_decoded(bitmap.dyn_into::<ImageBitmap>().ok())
                }),
                decode_failed: Closure::<dyn FnMut(JsValue)>::new({
                    let v = Rc::clone(v);
                    move |_error: JsValue| v.on_decoded(None)
                }),
                retry: Closure::<dyn FnMut()>::new({
                    let v = Rc::clone(v);
                    move || {
                        // A refused address is shown in the status line.
                        let _ = v.connect();
                    }
                }),
                flush: Closure::<dyn FnMut()>::new({
                    let v = Rc::clone(v);
                    move || v.flush_field()
                }),
                refocus: Closure::<dyn FnMut()>::new({
                    let v = Rc::clone(v);
                    move || v.refocus()
                }),
            }
        };
        *viewer.handlers.borrow_mut() = Some(handlers);

        // The text field: the browser types into it, the viewer reads it.
        viewer.rest_field();
        viewer.focus_field();
        {
            let field: &EventTarget = &viewer.field;
            let v = Rc::clone(&viewer);
            listen(field, "input", move |event| v.on_field_input(&event))?;
            for (event, started) in [("compositionstart", true), ("compositionend", false)] {
                let v = Rc::clone(&viewer);
                listen(field, event, move |_| v.on_composition(started))?;
            }
            let v = Rc::clone(&viewer);
            listen(field, "blur", move |_| v.soon(|handlers| &handlers.refocus))?;
            // The field holds nothing worth having on the clipboard.
            for event in ["cut", "copy"] {
                listen(field, event, |e| e.prevent_default())?;
            }
            let page: &EventTarget = &window;
            let v = Rc::clone(&viewer);
            listen(page, "focus", move |_| v.focus_field())?;
            let v = Rc::clone(&viewer);
            listen(page, "pointerup", move |event| v.on_pointer_up(&event))?;
        }

        // Page-lifetime DOM listeners: registered once, never removed.
        for (event, down) in [("keydown", true), ("keyup", false)] {
            let v = Rc::clone(&viewer);
            let on_key = Closure::<dyn FnMut(KeyboardEvent)>::new(move |e: KeyboardEvent| {
                v.on_key(&e, down)
            });
            window.add_event_listener_with_callback(event, on_key.as_ref().unchecked_ref())?;
            on_key.forget();
        }
        {
            let v = Rc::clone(&viewer);
            let on_paste =
                Closure::<dyn FnMut(ClipboardEvent)>::new(move |e: ClipboardEvent| v.on_paste(&e));
            window.add_event_listener_with_callback("paste", on_paste.as_ref().unchecked_ref())?;
            on_paste.forget();
        }
        {
            let v = Rc::clone(&viewer);
            let on_blur = Closure::<dyn FnMut()>::new(move || v.on_blur());
            window.add_event_listener_with_callback("blur", on_blur.as_ref().unchecked_ref())?;
            on_blur.forget();
        }
        {
            let v = Rc::clone(&viewer);
            let on_move =
                Closure::<dyn FnMut(MouseEvent)>::new(move |e: MouseEvent| v.on_mouse_move(&e));
            canvas
                .add_event_listener_with_callback("mousemove", on_move.as_ref().unchecked_ref())?;
            on_move.forget();
        }

        viewer.connect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_messages_use_the_shared_wire_format() {
        // Same vector as space-stream's `viewer_input_wire_bytes_are_stable`.
        assert_eq!(key_message("KeyA", true), Some(vec![2, 3, 1]));
        assert_eq!(key_message("ArrowLeft", false), Some(vec![2, 3, 0]));
        assert_eq!(key_message("F13", true), None);
    }

    fn decoded(messages: &[Vec<u8>]) -> Vec<ViewerInput> {
        messages
            .iter()
            .map(|m| ViewerInput::decode(m).expect("a message the proxy accepts"))
            .collect()
    }

    fn key(key: Key, down: bool) -> ViewerInput {
        ViewerInput::Key { key, down }
    }

    const NONE: Modifiers = Modifiers {
        ctrl: false,
        alt: false,
        meta: false,
    };
    const CTRL: Modifiers = Modifiers {
        ctrl: true,
        alt: false,
        meta: false,
    };
    const ALT: Modifiers = Modifiers {
        ctrl: false,
        alt: true,
        meta: false,
    };
    /// How Windows reports AltGr.
    const ALT_GR: Modifiers = Modifiers {
        ctrl: true,
        alt: true,
        meta: false,
    };
    const META: Modifiers = Modifiers {
        ctrl: false,
        alt: false,
        meta: true,
    };

    fn text(text: &str) -> Vec<ViewerInput> {
        text.chars().map(ViewerInput::Text).collect()
    }

    fn stroke(wire: Key) -> Vec<ViewerInput> {
        vec![key(wire, true), key(wire, false)]
    }

    /// A stand-in for the page: the text field's content as a browser edits
    /// it, and everything the viewer sent.
    ///
    /// The viewer's side (`input`, `flush`, `composition_*`, `keydown`,
    /// `keyup`) does exactly what the glue in `web` does with each event; the
    /// browser's side (`browser_*`, `press`, `hold`) plays the part of the
    /// browser. A test passes when every character and every stroke arrives
    /// once, in order, whatever the browser did to deliver it.
    struct Page {
        text: TextField,
        value: String,
        sent: Vec<ViewerInput>,
    }

    impl Page {
        fn new() -> Self {
            Self {
                text: TextField::default(),
                value: FIELD_REST.to_owned(),
                sent: Vec::new(),
            }
        }

        fn apply(&mut self, edit: &FieldEdit) {
            self.sent.extend(decoded(&edit.messages));
            if edit.reset {
                FIELD_REST.clone_into(&mut self.value);
            }
        }

        /// An `input` event on the field.
        fn input(&mut self, input_type: &str, is_composing: bool) {
            let edit = self.text.input(&self.value, input_type, is_composing);
            self.apply(&edit);
        }

        /// The flush the glue runs before it sends a key and shortly after a
        /// composition ended.
        fn flush(&mut self) {
            let edit = self.text.flush(&self.value);
            self.apply(&edit);
        }

        /// A keydown; returns whether its default action was prevented.
        fn keydown(
            &mut self,
            code: &str,
            dom_key: &str,
            modifiers: Modifiers,
            repeat: bool,
            (is_composing, key_code): (bool, u32),
        ) -> bool {
            let action = if is_ime_keydown(is_composing, key_code) {
                ime_key_action(code, modifiers, repeat)
            } else {
                self.flush();
                key_action(code, dom_key, modifiers, true, repeat)
            };
            self.sent.extend(decoded(&action.messages));
            action.prevent_default
        }

        fn keyup(&mut self, code: &str, dom_key: &str) {
            self.flush();
            let action = key_action(code, dom_key, NONE, false, false);
            self.sent.extend(decoded(&action.messages));
        }

        /// The caret: between the sentinels while the field rests, so just
        /// before the last one.
        fn caret(&self) -> usize {
            self.value.rfind(FIELD_SENTINEL).unwrap_or(self.value.len())
        }

        fn browser_inserts(&mut self, text: &str) {
            let caret = self.caret();
            self.value.insert_str(caret, text);
        }

        fn browser_deletes_backward(&mut self) {
            let caret = self.caret();
            let before = self.value[..caret].chars().next_back();
            let before = before.expect("something before the caret");
            self.value.remove(caret - before.len_utf8());
        }

        fn browser_deletes_forward(&mut self) {
            let caret = self.caret();
            self.value.remove(caret);
        }

        /// A key going down on a physical keyboard, as a browser plays it:
        /// the keydown event and then, unless that was prevented or the key
        /// types nothing, its character put into the focused field.
        fn hold(&mut self, code: &str, dom_key: &str, modifiers: Modifiers, repeat: bool) {
            let prevented = self.keydown(code, dom_key, modifiers, repeat, PHYSICAL);
            let types = dom_key.chars().count() == 1 && !modifiers.is_shortcut();
            if types && !prevented {
                self.browser_inserts(dom_key);
                self.input("insertText", false);
            }
        }

        /// One whole key press: down, (character), up.
        fn press(&mut self, code: &str, dom_key: &str, modifiers: Modifiers) {
            self.hold(code, dom_key, modifiers, false);
            self.keyup(code, dom_key);
        }

        /// What was sent since the last call. The field must be back at rest:
        /// nothing typed may linger in it.
        #[track_caller]
        fn take(&mut self) -> Vec<ViewerInput> {
            assert_eq!(self.value, FIELD_REST, "the field is not back at rest");
            std::mem::take(&mut self.sent)
        }
    }

    /// `(isComposing, keyCode)` of a key event from a physical keyboard.
    const PHYSICAL: (bool, u32) = (false, 65);
    /// ... of every key of an on-screen keyboard, and of a key an input
    /// method is handling.
    const PROCESSING: (bool, u32) = (false, 229);

    /// Source 1, a physical keyboard: a key press types its character exactly
    /// once. The keydown carries the movement key and no text; the character
    /// comes out of the text field the browser typed it into.
    ///
    /// Mutation this catches: sending the character from the keydown as well
    /// (every character twice), and preventing the keydown of a printable
    /// movement key (W and Space would move but never type).
    #[test]
    fn a_key_press_types_its_character_exactly_once() {
        let mut page = Page::new();
        page.press("KeyW", "w", NONE);
        assert_eq!(
            page.take(),
            [
                key(Key::Forward, true),
                ViewerInput::Text('w'),
                key(Key::Forward, false)
            ]
        );
        // The keydown by itself: the movement key, the same bytes as ever,
        // no text, and left to the browser so that it can type.
        let down = key_action("KeyW", "w", NONE, true, false);
        assert_eq!(down.messages, [key_message("KeyW", true).unwrap()]);
        assert!(!down.prevent_default);
        assert_eq!(down.held, Some(true));
        let up = key_action("KeyW", "w", NONE, false, false);
        assert_eq!(decoded(&up.messages), [key(Key::Forward, false)]);
        assert_eq!((up.held, up.prevent_default), (Some(false), true));

        // Held down: the key is held once, the character repeats (as in any
        // text field), and each repeat is typed once.
        page.hold("KeyW", "w", NONE, false);
        page.hold("KeyW", "w", NONE, true);
        page.hold("KeyW", "w", NONE, true);
        page.keyup("KeyW", "w");
        assert_eq!(
            page.take(),
            [
                key(Key::Forward, true),
                ViewerInput::Text('w'),
                ViewerInput::Text('w'),
                ViewerInput::Text('w'),
                key(Key::Forward, false)
            ]
        );
        let repeat = key_action("KeyW", "w", NONE, true, true);
        assert_eq!(repeat, KeyAction::default(), "a repeat is only typed");

        // Space flies up and types a space; Shift flies down, types nothing
        // and is kept from the browser; a digit only types.
        page.press("Space", " ", NONE);
        assert_eq!(
            page.take(),
            [
                key(Key::Up, true),
                ViewerInput::Text(' '),
                key(Key::Up, false)
            ]
        );
        page.press("ShiftLeft", "Shift", NONE);
        assert_eq!(page.take(), [key(Key::Down, true), key(Key::Down, false)]);
        assert!(key_action("ShiftLeft", "Shift", NONE, true, false).prevent_default);
        page.press("Digit7", "7", NONE);
        assert_eq!(page.take(), text("7"));
        let digit = key_action("Digit7", "7", NONE, true, false);
        assert_eq!(digit, KeyAction::default(), "no key message, not prevented");
        // Arrow keys type nothing, so they are kept from the browser.
        assert!(key_action("ArrowUp", "ArrowUp", NONE, true, false).prevent_default);
    }

    /// Characters are whatever the browser types under the user's keyboard
    /// layout, while movement keys stay on their physical positions.
    #[test]
    fn typed_characters_respect_the_keyboard_layout() {
        let mut page = Page::new();
        // German QWERTZ: ':' is Shift+'.', '@' is AltGr+Q (Ctrl+Alt on
        // Windows), Y and Z are swapped.
        page.press("Period", ":", NONE);
        page.press("KeyQ", "@", ALT_GR);
        page.press("KeyY", "z", NONE);
        page.press("Semicolon", "ö", NONE);
        assert_eq!(page.take(), text(":@zö"));
        // French AZERTY: the key at the W position types 'z' (here with
        // Shift) and still flies forward.
        page.press("KeyW", "Z", NONE);
        assert_eq!(
            page.take(),
            [
                key(Key::Forward, true),
                ViewerInput::Text('Z'),
                key(Key::Forward, false)
            ]
        );
        // A dead key: the accent's keydown types nothing and is left alone;
        // the next key's keydown says "e", but what the browser types — and
        // so what is sent — is the composed character.
        let dead = key_action("Equal", "Dead", NONE, true, false);
        assert_eq!(dead, KeyAction::default());
        page.press("Equal", "Dead", NONE);
        assert!(!page.keydown("KeyE", "e", NONE, false, PHYSICAL));
        page.browser_inserts("é");
        page.input("insertText", false);
        page.keyup("KeyE", "e");
        assert_eq!(page.take(), text("é"));

        // Which keys are left to the browser because they type: a Mac's Option
        // key composes characters; Alt+letter elsewhere is a menu accelerator
        // and types nothing, so the movement key under it is prevented as it
        // always was.
        assert_eq!(typed_char("@", ALT), Some('@'));
        assert_eq!(typed_char("€", ALT), Some('€'));
        assert_eq!(typed_char("d", ALT), None);
        assert_eq!(typed_char("7", ALT), None);
        assert_eq!(typed_char("\u{1F680}", NONE), Some('\u{1F680}'));
        let option_l = key_action("KeyL", "@", ALT, true, false);
        assert_eq!(option_l, KeyAction::default(), "left to the browser");
        let alt_d = key_action("KeyD", "d", ALT, true, false);
        assert_eq!(decoded(&alt_d.messages), [key(Key::Right, true)]);
        assert!(alt_d.prevent_default);
        // Named keys, dead keys, IME placeholders and control characters type
        // nothing.
        for named in [
            "Enter",
            "Shift",
            "Control",
            "Alt",
            "AltGraph",
            "CapsLock",
            "F5",
            "Dead",
            "Process",
            "Unidentified",
            "ArrowLeft",
            "Backspace",
            "",
            "ab",
            "\n",
            "\t",
            "\u{7f}",
            "\u{1b}",
        ] {
            assert_eq!(typed_char(named, NONE), None, "{named:?}");
        }
    }

    /// Enter, Backspace and friends are strokes: a down directly followed by
    /// an up, once per keydown (auto-repeat included), and nothing on keyup.
    #[test]
    fn text_entry_keys_are_strokes() {
        for (code, dom_key, wire) in [
            ("Enter", "Enter", Key::Enter),
            ("NumpadEnter", "Enter", Key::Enter),
            ("Backspace", "Backspace", Key::Backspace),
            ("Escape", "Escape", Key::Escape),
            ("Tab", "Tab", Key::Tab),
            ("Delete", "Delete", Key::Delete),
            ("Home", "Home", Key::Home),
            ("End", "End", Key::End),
            // The numeric keypad with NumLock off.
            ("Numpad7", "Home", Key::Home),
        ] {
            for repeat in [false, true] {
                let down = key_action(code, dom_key, NONE, true, repeat);
                assert_eq!(
                    decoded(&down.messages),
                    [key(wire, true), key(wire, false)],
                    "{code} repeat={repeat}"
                );
                assert!(down.prevent_default, "{code}");
                assert_eq!(down.held, None, "a stroke is never held");
            }
            let up = key_action(code, dom_key, NONE, false, false);
            assert_eq!(up, KeyAction::default(), "{code}");
        }
        // The horizontal arrows steer (held) and move the caret (stroke); the
        // vertical ones only steer.
        let left = key_action("ArrowLeft", "ArrowLeft", NONE, true, false);
        assert_eq!(
            decoded(&left.messages),
            [
                key(Key::Left, true),
                key(Key::CaretLeft, true),
                key(Key::CaretLeft, false)
            ]
        );
        assert_eq!(left.held, Some(true));
        let left_repeat = key_action("ArrowLeft", "ArrowLeft", NONE, true, true);
        assert_eq!(
            decoded(&left_repeat.messages),
            [key(Key::CaretLeft, true), key(Key::CaretLeft, false)]
        );
        let left_up = key_action("ArrowLeft", "ArrowLeft", NONE, false, false);
        assert_eq!(decoded(&left_up.messages), [key(Key::Left, false)]);
        let up = key_action("ArrowUp", "ArrowUp", NONE, true, false);
        assert_eq!(decoded(&up.messages), [key(Key::Forward, true)]);
        // 'a' strafes (and types, through the field), but does not move the
        // caret.
        let a = key_action("KeyA", "a", NONE, true, false);
        assert_eq!(decoded(&a.messages), [key(Key::Left, true)]);

        // A stroke is prevented, so the browser does not also edit the text
        // field: one Backspace is one stroke, and the field is untouched.
        let mut page = Page::new();
        page.press("Backspace", "Backspace", NONE);
        page.press("Enter", "Enter", NONE);
        page.press("Delete", "Delete", NONE);
        page.press("Tab", "Tab", NONE);
        assert_eq!(
            page.take(),
            [
                stroke(Key::Backspace),
                stroke(Key::Enter),
                stroke(Key::Delete),
                stroke(Key::Tab)
            ]
            .concat()
        );
    }

    /// Source 3, an on-screen keyboard: every key is reported as
    /// "Unidentified" with key code 229, so nothing can be read from the key
    /// events; the text field's edits are all there is. Typing, Backspace,
    /// Delete and Enter each arrive once.
    ///
    /// Mutation this catches: resting the field empty (nothing for a soft
    /// Backspace to delete, so it never shows up).
    #[test]
    fn an_on_screen_keyboard_types_deletes_and_enters_through_the_field() {
        let mut page = Page::new();
        let soft_key = |page: &mut Page| {
            let prevented = page.keydown("", "Unidentified", NONE, false, PROCESSING);
            assert!(!prevented);
            assert!(page.sent.is_empty(), "the key event itself says nothing");
        };
        for c in ["7", ".", "a"] {
            soft_key(&mut page);
            page.browser_inserts(c);
            page.input("insertText", false);
            page.keyup("", "Unidentified");
            assert_eq!(page.take(), text(c));
        }
        // Backspace, three times: each one finds a sentinel to delete, because
        // the field is put back to rest after each.
        for _ in 0..3 {
            soft_key(&mut page);
            page.browser_deletes_backward();
            page.input("deleteContentBackward", false);
            assert_eq!(page.take(), stroke(Key::Backspace));
        }
        soft_key(&mut page);
        page.browser_deletes_forward();
        page.input("deleteContentForward", false);
        assert_eq!(page.take(), stroke(Key::Delete));
        // Enter is a line break put into the field.
        soft_key(&mut page);
        page.browser_inserts("\n");
        page.input("insertLineBreak", false);
        assert_eq!(page.take(), stroke(Key::Enter));
        // A browser that does not say what kind of edit it was.
        page.browser_deletes_backward();
        page.input("", false);
        assert_eq!(page.take(), stroke(Key::Backspace));

        // A keyboard that does send real key events for these (iOS; Android
        // for Backspace in a field it believes empty): the keydown is the
        // stroke and is prevented, so no edit follows and nothing doubles.
        page.press("", "Backspace", NONE);
        page.press("", "Enter", NONE);
        assert_eq!(
            page.take(),
            [stroke(Key::Backspace), stroke(Key::Enter)].concat()
        );
        assert!(is_ime_keydown(false, 229) && is_ime_keydown(true, 13));
        assert!(!is_ime_keydown(false, 13) && !is_ime_keydown(false, 0));
    }

    /// With an input method switched on, the browser reports every key as
    /// "being processed". The key's position is still known, so the ship can
    /// still be flown; but nothing is read from what the key is called, and
    /// the key is never taken away from the input method.
    #[test]
    fn movement_keys_work_while_an_input_method_is_on() {
        let mut page = Page::new();
        assert!(!page.keydown("KeyW", "Process", NONE, false, PROCESSING));
        assert!(!page.keydown("KeyW", "Process", NONE, true, PROCESSING));
        page.keyup("KeyW", "w");
        assert_eq!(
            page.take(),
            [key(Key::Forward, true), key(Key::Forward, false)]
        );
        let held = ime_key_action("KeyW", NONE, false);
        assert_eq!((held.held, held.prevent_default), (Some(true), false));
        // An arrow steers but is no caret stroke; Safari's late Enter, an
        // on-screen key (no `code`), a repeat and a shortcut are nothing.
        let arrow = ime_key_action("ArrowLeft", NONE, false);
        assert_eq!(decoded(&arrow.messages), [key(Key::Left, true)]);
        assert!(!arrow.prevent_default);
        for (code, modifiers, repeat) in [
            ("Enter", NONE, false),
            ("", NONE, false),
            ("KeyQ", NONE, false),
            ("KeyW", NONE, true),
            ("KeyW", CTRL, false),
            ("KeyW", META, false),
        ] {
            let action = ime_key_action(code, modifiers, repeat);
            assert_eq!(action, KeyAction::default(), "{code:?} {modifiers:?}");
        }
    }

    /// Source 2, an input method: the unfinished composition is neither sent
    /// nor touched; the committed text is sent once — whether the browser
    /// fires its last `input` before `compositionend` (Chrome) or after it
    /// (Firefox, Safari).
    ///
    /// Mutation this catches: sending on every `input` event (the
    /// intermediate text, several times), and sending both at
    /// `compositionend` and at the `input` that follows it.
    #[test]
    fn an_input_method_sends_the_committed_text_once() {
        let composing = (true, 229);
        let resting = |text: &str| format!("{FIELD_SENTINEL}{text}{FIELD_SENTINEL}");

        // Chrome: compositionstart, input..., input, compositionend.
        let mut page = Page::new();
        page.text.composition_started();
        assert!(!page.keydown("KeyN", "Process", NONE, false, composing));
        page.browser_inserts("に");
        page.input("insertCompositionText", true);
        page.value = resting("日本");
        page.input("insertCompositionText", true);
        assert!(page.sent.is_empty(), "nothing while composing");
        assert_eq!(page.value, resting("日本"), "and the field is left alone");
        page.text.composition_ended();
        assert!(page.sent.is_empty(), "collected after the event, not in it");
        page.flush();
        assert_eq!(page.take(), text("日本"));
        page.flush();
        assert!(page.take().is_empty());

        // Firefox: ..., compositionend, input (no longer composing). The
        // flush scheduled at compositionend then finds nothing left.
        page.text.composition_started();
        page.value = resting("日本");
        page.input("insertCompositionText", true);
        page.text.composition_ended();
        page.input("insertCompositionText", false);
        page.flush();
        assert_eq!(page.take(), text("日本"));

        // Safari: the same order, another input type, and then the keydown of
        // the Enter that confirmed the composition, with key code 229: it is
        // not an Enter for the game.
        page.text.composition_started();
        page.value = resting("日本");
        page.input("insertCompositionText", true);
        page.text.composition_ended();
        page.input("insertFromComposition", false);
        assert!(!page.keydown("Enter", "Enter", NONE, false, PROCESSING));
        page.flush();
        assert_eq!(page.take(), text("日本"));
        // The next Enter is a real one.
        page.press("Enter", "Enter", NONE);
        assert_eq!(page.take(), stroke(Key::Enter));

        // Korean: a syllable is committed and the next composition starts in
        // the same breath. The flush scheduled by the first compositionend
        // must not take the field apart under the new composition; both
        // syllables are sent, once and in order, when composing stops.
        page.text.composition_started();
        page.browser_inserts("한");
        page.input("insertCompositionText", true);
        page.text.composition_ended();
        page.text.composition_started();
        page.flush();
        assert!(page.sent.is_empty());
        assert_eq!(
            page.value,
            resting("한"),
            "left alone under the composition"
        );
        page.browser_inserts("글");
        page.input("insertCompositionText", true);
        page.text.composition_ended();
        page.flush();
        assert_eq!(page.take(), text("한글"));

        // A word confirmed with Enter in one go (an on-screen keyboard's
        // Enter): the Enter keydown arrives before the scheduled flush, and
        // still the word goes first.
        page.text.composition_started();
        page.browser_inserts("host");
        page.input("insertCompositionText", true);
        page.text.composition_ended();
        page.press("", "Enter", NONE);
        page.flush();
        assert_eq!(page.take(), [text("host"), stroke(Key::Enter)].concat());

        // A composition that was cancelled leaves nothing.
        page.text.composition_started();
        page.browser_inserts("x");
        page.input("insertCompositionText", true);
        page.value = resting("");
        page.input("insertCompositionText", true);
        page.text.composition_ended();
        page.flush();
        assert!(page.take().is_empty());
        assert!(!page.text.is_composing());
    }

    /// Source 5, an automation tool that inserts text without key events:
    /// the whole string arrives once, in order.
    ///
    /// Mutation this catches: listening for `keydown` only (nothing arrives).
    #[test]
    fn text_inserted_without_key_events_is_typed_once() {
        let mut page = Page::new();
        // CDP `Input.insertText` and `document.execCommand('insertText')`:
        // the text goes in at the caret and one `input` event follows.
        page.browser_inserts("127.0.0.1:7777");
        page.input("insertText", false);
        assert_eq!(page.take(), text("127.0.0.1:7777"));
        // A second event about the same edit (a tool that dispatches `input`
        // again, or `change`) finds nothing new.
        page.input("insertText", false);
        page.input("", false);
        assert!(page.take().is_empty());

        // Setting the value and dispatching a plain `input` event: the field's
        // resting content is replaced, appended to, or prepended to — the
        // text is what is there besides the sentinels.
        for value in [
            "play.example.org:7777".to_owned(),
            format!("{FIELD_REST}play.example.org:7777"),
            format!("play.example.org:7777{FIELD_REST}"),
            format!("{FIELD_SENTINEL}play.example.org:7777"),
        ] {
            page.value = value;
            page.input("", false);
            assert_eq!(page.take(), text("play.example.org:7777"));
        }
        // Emptying the field that way is not a Backspace.
        page.value = String::new();
        page.input("", false);
        assert!(page.take().is_empty());

        // A line break submits (Enter), "\r\n" is one line break, a tab is
        // the Tab key; other control characters never get through.
        page.browser_inserts("[::1]:7777\r\n");
        page.input("insertText", false);
        assert_eq!(
            page.take(),
            [text("[::1]:7777"), stroke(Key::Enter)].concat()
        );
        page.browser_inserts("a\tb\u{1b}[2J\0c\u{7f}");
        page.input("insertText", false);
        assert_eq!(
            page.take(),
            [text("a"), stroke(Key::Tab), text("b[2Jc")].concat()
        );
        // One edit is bounded.
        page.browser_inserts(&"ä".repeat(MAX_INPUT_CHARS * 3));
        page.input("insertText", false);
        assert_eq!(page.take(), text(&"ä".repeat(MAX_INPUT_CHARS)));

        // Typed text and a key right after it keep their order.
        page.browser_inserts("abc");
        page.press("Enter", "Enter", NONE);
        assert_eq!(page.take(), [text("abc"), stroke(Key::Enter)].concat());
    }

    /// Source 4, a paste: the `paste` event is the normal way
    /// ([`paste_messages`], and the event is prevented so the field never sees
    /// it). If the text reaches the field instead, it follows the same rules:
    /// no line break submits, and it is sent once.
    #[test]
    fn a_paste_that_reaches_the_field_follows_the_paste_rules() {
        let mut page = Page::new();
        for input_type in ["insertFromPaste", "insertFromDrop", "insertFromYank"] {
            page.browser_inserts("  host:1\r\n\tnext\u{1b}[2J\0 ö");
            page.input(input_type, false);
            assert_eq!(page.take(), text("  host:1next[2J ö"), "{input_type}");
        }
        page.browser_inserts(&"ä".repeat(MAX_PASTE_CHARS * 4));
        page.input("insertFromPaste", false);
        assert_eq!(page.take().len(), MAX_PASTE_CHARS);
        // What was typed just before a paste goes first (the glue flushes the
        // field before it sends the pasted text).
        page.browser_inserts("ab");
        page.flush();
        page.sent.extend(decoded(&paste_messages("cd")));
        assert_eq!(page.take(), text("abcd"));
    }

    /// What an edit of the field means, case by case.
    #[test]
    fn field_edits_map_to_text_and_strokes() {
        let one = FIELD_SENTINEL.to_string();
        let sent = |value: &str, input_type: &str| decoded(&field_messages(value, input_type));
        // A missing sentinel with nothing put in is a deletion: backward
        // unless the edit says forward.
        for input_type in [
            "deleteContentBackward",
            "deleteWordBackward",
            "deleteSoftLineBackward",
            "deleteHardLineBackward",
            "deleteContent",
            "",
            "insertCompositionText",
            "insertText",
        ] {
            assert_eq!(
                sent(&one, input_type),
                stroke(Key::Backspace),
                "{input_type:?}"
            );
        }
        for input_type in [
            "deleteContentForward",
            "deleteWordForward",
            "deleteSoftLineForward",
            "deleteHardLineForward",
        ] {
            assert_eq!(sent(&one, input_type), stroke(Key::Delete), "{input_type}");
            assert_eq!(sent("", input_type), stroke(Key::Delete), "{input_type}");
        }
        // Both sentinels gone: one stroke if the edit was a deletion (select
        // all + delete), nothing if the field was just emptied.
        assert_eq!(sent("", "deleteContentBackward"), stroke(Key::Backspace));
        assert_eq!(sent("", "deleteWordBackward"), stroke(Key::Backspace));
        for input_type in ["", "insertText", "insertCompositionText", "insertFromPaste"] {
            assert!(sent("", input_type).is_empty(), "{input_type:?}");
        }
        // Text put in is text, even if a sentinel went missing with it (a
        // replaced selection): no stroke is invented.
        assert_eq!(sent("x", "insertText"), text("x"));
        assert_eq!(sent(&format!("{one}x"), "insertReplacementText"), text("x"));
        assert_eq!(sent(&format!("{one}x"), "deleteContentBackward"), text("x"));
        // Not input at all: undo, redo, cut, formatting.
        for input_type in [
            "historyUndo",
            "historyRedo",
            "deleteByCut",
            "deleteByDrag",
            "formatBold",
        ] {
            for value in ["", one.as_str(), "old text", FIELD_REST] {
                assert!(sent(value, input_type).is_empty(), "{input_type} {value:?}");
            }
        }
        // A paste never deletes.
        assert!(sent(&one, "insertFromPaste").is_empty());
        // The field at rest is nothing, whatever the event claims.
        for input_type in ["", "insertText", "deleteContentBackward", "insertLineBreak"] {
            assert!(sent(FIELD_REST, input_type).is_empty(), "{input_type:?}");
        }
        // Every message is one the proxy accepts, and only keys and text.
        let all = field_messages("a\nb\tc\u{200b}\u{1F680}", "insertText");
        assert_eq!(
            decoded(&all),
            [
                text("a"),
                stroke(Key::Enter),
                text("b"),
                stroke(Key::Tab),
                text("c\u{1F680}")
            ]
            .concat()
        );
    }

    /// The bookkeeping itself: nothing is taken (and the field is not reset)
    /// while an input method composes or when the field already rests; what
    /// is not input is still cleared out of the field.
    #[test]
    fn the_field_is_reset_exactly_when_its_content_was_taken() {
        let mut field = TextField::default();
        assert_eq!(
            field.input(FIELD_REST, "insertText", false),
            FieldEdit::default()
        );
        assert_eq!(field.flush(FIELD_REST), FieldEdit::default());

        let typed = format!("{FIELD_SENTINEL}a{FIELD_SENTINEL}");
        // The event says "composing" although no compositionstart was seen.
        assert_eq!(
            field.input(&typed, "insertCompositionText", true),
            FieldEdit::default()
        );
        field.composition_started();
        assert!(field.is_composing());
        assert_eq!(
            field.input(&typed, "insertText", false),
            FieldEdit::default()
        );
        assert_eq!(field.flush(&typed), FieldEdit::default());
        field.composition_ended();
        let taken = field.flush(&typed);
        assert_eq!(decoded(&taken.messages), text("a"));
        assert!(taken.reset);

        // An undo brings old text back into the field: it is not sent, but it
        // is cleared away, or it would be sent with the next key.
        let undone = field.input("old text", "historyUndo", false);
        assert_eq!((undone.messages.len(), undone.reset), (0, true));
    }

    #[test]
    fn other_text_controls_of_the_page_keep_their_keyboard() {
        for tag in ["INPUT", "input", "TEXTAREA", "textarea", "SELECT"] {
            assert!(is_text_control(tag, false), "{tag}");
        }
        for tag in ["BODY", "CANVAS", "DIV", "BUTTON", "A", "HTML", ""] {
            assert!(!is_text_control(tag, false), "{tag}");
            assert!(is_text_control(tag, true), "{tag} contenteditable");
        }
    }

    /// Browser shortcuts stay the browser's: nothing is sent and nothing is
    /// prevented for them. (Preventing Ctrl+V would also suppress the `paste`
    /// event the viewer relies on.)
    #[test]
    fn browser_shortcuts_are_left_alone() {
        for (code, dom_key, modifiers) in [
            ("F5", "F5", NONE),
            ("F12", "F12", NONE),
            ("F11", "F11", NONE),
            ("KeyR", "r", CTRL),
            ("KeyV", "v", CTRL),
            ("KeyC", "c", CTRL),
            ("KeyW", "w", CTRL),
            ("KeyA", "a", CTRL),
            ("KeyS", "s", CTRL),
            ("KeyD", "d", CTRL),
            ("KeyR", "r", META),
            ("KeyQ", "q", META),
            ("KeyW", "w", META),
            ("Enter", "Enter", CTRL),
            ("Tab", "Tab", CTRL),
            ("ArrowLeft", "ArrowLeft", META),
            ("ControlLeft", "Control", CTRL),
            ("MetaLeft", "Meta", META),
            ("Insert", "Insert", NONE),
        ] {
            for repeat in [false, true] {
                let action = key_action(code, dom_key, modifiers, true, repeat);
                assert_eq!(action, KeyAction::default(), "{code} {modifiers:?}");
            }
        }
        // A movement key that was down before Ctrl was pressed is still
        // released when it goes up, or it would stay held.
        let up = key_action("KeyW", "w", CTRL, false, false);
        assert_eq!(decoded(&up.messages), [key(Key::Forward, false)]);
        assert_eq!(up.held, Some(false));
        assert!(CTRL.is_shortcut() && META.is_shortcut());
        assert!(!NONE.is_shortcut() && !ALT.is_shortcut() && !ALT_GR.is_shortcut());
    }

    #[test]
    fn a_paste_is_typed_character_by_character() {
        assert_eq!(
            decoded(&paste_messages("[::1]:7777")),
            "[::1]:7777"
                .chars()
                .map(ViewerInput::Text)
                .collect::<Vec<_>>()
        );
        // Line breaks, tabs and other control characters are dropped: a pasted
        // address does not submit itself, and nothing unprintable gets through.
        assert_eq!(
            decoded(&paste_messages("  host:1\r\n\tnext\u{1b}[2J\0 ö")),
            "  host:1next[2J ö"
                .chars()
                .map(ViewerInput::Text)
                .collect::<Vec<_>>()
        );
        assert!(paste_messages("").is_empty());
        assert!(paste_messages("\n\n").is_empty());
        // A huge clipboard is cut, by characters (not bytes).
        let long = "ä".repeat(MAX_PASTE_CHARS * 4);
        let sent = paste_messages(&long);
        assert_eq!(sent.len(), MAX_PASTE_CHARS);
        assert!(decoded(&sent)
            .iter()
            .all(|input| *input == ViewerInput::Text('ä')));
    }

    #[test]
    fn blur_releases_every_held_key() {
        let held = ["KeyA".to_string(), "Space".to_string()];
        let msgs = release_messages(&held);
        assert_eq!(msgs.len(), 2);
        for m in msgs {
            assert!(matches!(
                ViewerInput::decode(&m),
                Ok(ViewerInput::Key { down: false, .. })
            ));
        }
        assert!(release_messages(&[]).is_empty());
    }

    /// After a reconnect the keys still physically held are pressed again on
    /// the new connection (unmapped codes are skipped).
    #[test]
    fn reconnect_presses_every_held_key_again() {
        let held = ["KeyW".to_string(), "F13".to_string(), "Space".to_string()];
        let msgs: Vec<_> = press_messages(&held)
            .iter()
            .map(|m| ViewerInput::decode(m).unwrap())
            .collect();
        assert_eq!(
            msgs,
            [
                ViewerInput::Key {
                    key: Key::Forward,
                    down: true
                },
                ViewerInput::Key {
                    key: Key::Up,
                    down: true
                },
            ]
        );
        assert!(press_messages(&[]).is_empty());
    }

    #[test]
    fn viewer_messages_decode_with_the_server_decoder() {
        let msg = key_message("Space", true).unwrap();
        assert_eq!(
            ViewerInput::decode(&msg),
            Ok(ViewerInput::Key {
                key: Key::Up,
                down: true
            })
        );
        // A drag of 400 px to the left is +1 rad of yaw: the same bytes as
        // space-stream's `viewer_input_wire_bytes_are_stable` Turn vector.
        let msg = turn_message(-400.0, 0.0).unwrap();
        assert_eq!(msg, [3, 0x3f, 0x80, 0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn pointer_is_normalised_and_clamped() {
        let msg = pointer_message(160.0, 120.0, 640.0, 480.0, 1).unwrap();
        assert_eq!(
            ViewerInput::decode(&msg),
            Ok(ViewerInput::Pointer {
                x: 0.25,
                y: 0.25,
                buttons: 1
            })
        );
        let msg = pointer_message(-5.0, 9999.0, 640.0, 480.0, 0).unwrap();
        assert_eq!(
            ViewerInput::decode(&msg),
            Ok(ViewerInput::Pointer {
                x: 0.0,
                y: 1.0,
                buttons: 0
            })
        );
        assert_eq!(pointer_message(1.0, 1.0, 0.0, 480.0, 0), None);
        assert_eq!(pointer_message(f64::NAN, 1.0, 10.0, 10.0, 0), None);
    }

    /// The `(dyaw, dpitch)` a drag of `(dx, dy)` pixels puts on the wire.
    fn turn(dx: f64, dy: f64) -> (f32, f32) {
        let msg = turn_message(dx, dy).expect("a turn");
        match ViewerInput::decode(&msg) {
            Ok(ViewerInput::Turn { dyaw, dpitch }) => (dyaw, dpitch),
            other => panic!("not a Turn: {other:?}"),
        }
    }

    fn assert_turn(dx: f64, dy: f64, want: (f32, f32)) {
        let got = turn(dx, dy);
        let close = |g: f32, w: f32| (g - w).abs() <= 1e-6 * w.abs().max(1.0);
        assert!(
            close(got.0, want.0) && close(got.1, want.1),
            "drag ({dx}, {dy}): {got:?} != {want:?}"
        );
    }

    /// A drag is sent as a rotation *delta*: it depends only on how far the
    /// mouse moved, never on where earlier drags left the view. (An absolute
    /// angle made the first drag snap the camera to the viewer's own zero.)
    #[test]
    fn a_drag_is_a_relative_turn() {
        // Right turns right (negative yaw), down looks down (negative pitch).
        assert_turn(100.0, 0.0, (-0.25, 0.0));
        assert_turn(0.0, 40.0, (0.0, -0.1));
        assert_turn(-8.0, -4.0, (0.02, 0.01));
        // The same drag gives the same message every time: no state is kept,
        // and nothing is clamped or wrapped here (the proxy owns the limits).
        assert_eq!(turn_message(100.0, 0.0), turn_message(100.0, 0.0));
        let mut total = 0.0;
        for _ in 0..1000 {
            total += turn(4.0, 0.0).0;
        }
        assert!((total + 10.0).abs() < 1e-3, "1000 drags of 4 px: {total}");
        assert_turn(1.0e6, -1.0e6, (-2500.0, 2500.0));
        // An axis that did not move is a plain zero on the wire, not `-0.0`.
        assert_eq!(turn(3.0, 0.0).1.to_bits(), 0);
        assert_eq!(turn(0.0, 3.0).0.to_bits(), 0);
        // Nothing to say.
        assert_eq!(turn_message(0.0, 0.0), None);
        assert_eq!(turn_message(f64::NAN, 1.0), None);
        assert_eq!(turn_message(f64::INFINITY, 0.0), None);
        assert_eq!(turn_message(1.0e300, 0.0), None, "overflows f32");
    }

    fn assert_rect(got: Option<(f64, f64, f64, f64)>, want: (f64, f64, f64, f64)) {
        let got = got.expect("a rectangle");
        for (g, w) in [got.0, got.1, got.2, got.3]
            .into_iter()
            .zip([want.0, want.1, want.2, want.3])
        {
            assert!((g - w).abs() < 1e-9, "{got:?} != {want:?}");
        }
    }

    /// The picture keeps its aspect ratio inside the window-sized canvas
    /// element, whatever shape the window has.
    #[test]
    fn the_picture_is_fitted_into_the_element_with_its_aspect_kept() {
        // Same shape: fills the element.
        assert_rect(
            contained_rect(1280.0, 960.0, 640.0, 480.0),
            (0.0, 0.0, 1280.0, 960.0),
        );
        // Wide window, 16:9 picture limited by the height: bars left and right.
        assert_rect(
            contained_rect(2000.0, 720.0, 640.0, 360.0),
            (360.0, 0.0, 1280.0, 720.0),
        );
        // Tall window: bars above and below.
        assert_rect(
            contained_rect(400.0, 800.0, 640.0, 480.0),
            (0.0, 250.0, 400.0, 300.0),
        );
        // Smaller than the picture: scaled down, still centred.
        assert_rect(
            contained_rect(320.0, 320.0, 640.0, 480.0),
            (0.0, 40.0, 320.0, 240.0),
        );
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert_eq!(contained_rect(bad, 100.0, 640.0, 480.0), None);
            assert_eq!(contained_rect(100.0, 100.0, 640.0, bad), None);
        }
    }

    /// Pointer positions are measured against the picture, not the element:
    /// the picture's corners are (0,0) and (1,1) even with letterbox bars.
    #[test]
    fn pointer_positions_are_relative_to_the_fitted_picture() {
        let (left, top, width, height) = contained_rect(2000.0, 720.0, 640.0, 360.0).unwrap();
        let at = |x: f64, y: f64| match ViewerInput::decode(&pointer_message(
            x - left,
            y - top,
            width,
            height,
            0,
        )?) {
            Ok(ViewerInput::Pointer { x, y, .. }) => Some((x, y)),
            other => panic!("not a Pointer: {other:?}"),
        };
        assert_eq!(at(360.0, 0.0), Some((0.0, 0.0)));
        assert_eq!(at(1640.0, 720.0), Some((1.0, 1.0)));
        assert_eq!(at(1000.0, 360.0), Some((0.5, 0.5)));
        // In the bars: clamped to the picture's edge.
        assert_eq!(at(0.0, 180.0), Some((0.0, 0.25)));
        assert_eq!(at(1999.0, 540.0), Some((1.0, 0.75)));
    }

    #[test]
    fn the_default_stream_is_on_the_host_that_served_the_page() {
        assert_eq!(default_stream_url("example.org"), "ws://example.org:8099");
        assert_eq!(default_stream_url("192.168.1.20"), "ws://192.168.1.20:8099");
        // `location.hostname` keeps the brackets of an IPv6 literal.
        assert_eq!(default_stream_url("[::1]"), "ws://[::1]:8099");
        // No host (file://): the old loopback default.
        assert_eq!(default_stream_url(""), "ws://127.0.0.1:8099");
        assert_eq!(default_stream_url("  "), "ws://127.0.0.1:8099");
    }

    #[test]
    fn reconnect_backs_off_and_is_capped() {
        let delays: Vec<u32> = (0..7).map(reconnect_delay_ms).collect();
        assert_eq!(delays, [500, 1000, 2000, 4000, 8000, 8000, 8000]);
        for attempt in [8, 22, 23, 31, 32, 33, 1000, u32::MAX] {
            assert_eq!(reconnect_delay_ms(attempt), RECONNECT_MAX_MS, "{attempt}");
        }
    }

    #[test]
    fn the_status_line_names_the_address_and_clears_when_live() {
        let url = "ws://host:8099";
        assert_eq!(status_line(Link::Live, url), "", "nothing over the picture");
        for link in [
            Link::Connecting,
            Link::Waiting,
            Link::Retrying { delay_ms: 500 },
            Link::Failed,
        ] {
            let text = status_line(link, url);
            assert!(text.contains(url), "{link:?}: {text}");
        }
        assert!(status_line(Link::Retrying { delay_ms: 500 }, url).contains("0.5 s"));
        assert!(status_line(Link::Retrying { delay_ms: 8000 }, url).contains("8.0 s"));
    }

    fn settled<T>(draw: bool, next: Option<T>) -> Settled<T> {
        Settled { draw, next }
    }

    /// One decode at a time; of the frames that arrive meanwhile only the
    /// newest is kept.
    #[test]
    fn frames_arriving_during_a_decode_collapse_to_the_newest() {
        let mut slot = FrameSlot::default();
        assert_eq!(slot.offer(1, "f1"), Some("f1"), "idle: decode at once");
        assert_eq!(slot.offer(2, "f2"), None, "decode in flight: wait");
        assert_eq!(slot.offer(3, "f3"), None);
        assert_eq!(slot.offer(4, "f4"), None, "f2 and f3 are skipped");
        assert_eq!(slot.newest(), 4);

        assert_eq!(
            slot.settle(),
            settled(true, Some("f4")),
            "draw f1, decode f4"
        );
        assert_eq!(slot.offer(5, "f5"), None, "f4 is in flight now");
        assert_eq!(slot.settle(), settled(true, Some("f5")));
        assert_eq!(slot.settle(), settled(true, None));

        // Idle again: the next frame decodes at once.
        assert_eq!(slot.offer(6, "f6"), Some("f6"));
        assert_eq!(slot.settle(), settled(true, None));
        // A settle with nothing in flight draws nothing.
        assert_eq!(slot.settle(), settled(false, None));
    }

    #[test]
    fn frames_that_are_not_newer_are_dropped() {
        let mut slot = FrameSlot::default();
        assert_eq!(slot.offer(0, "zero"), None, "sequence numbers start at 1");
        assert_eq!(slot.offer(7, "f7"), Some("f7"));
        assert_eq!(slot.offer(7, "again"), None);
        assert_eq!(slot.offer(3, "old"), None);
        assert_eq!(slot.settle(), settled(true, None), "nothing was queued");
    }

    /// A reconnect starts sequence numbers over (a restarted proxy counts from
    /// 1 again; without the reset every new frame would look old and the
    /// picture would stay frozen), and a decode still in flight from the old
    /// connection is not drawn over the new one.
    #[test]
    fn a_reconnect_resets_the_sequence_and_discards_the_old_decode() {
        let mut slot = FrameSlot::default();
        assert_eq!(slot.offer(900, "old-900"), Some("old-900"));
        assert_eq!(slot.offer(901, "old-901"), None);

        slot.reset();
        assert_eq!(slot.newest(), 0);
        // The new connection's first frame is accepted although 1 < 901. The
        // old decode is still in flight, so it waits (still one at a time).
        assert_eq!(slot.offer(1, "new-1"), None);
        assert_eq!(slot.offer(2, "new-2"), None);
        // The old decode finishes: not drawn, the old waiting frame is gone,
        // the newest new frame is next.
        assert_eq!(slot.settle(), settled(false, Some("new-2")));
        assert_eq!(slot.settle(), settled(true, None));

        // A reset while idle leaves nothing to discard.
        slot.reset();
        assert_eq!(slot.offer(1, "again-1"), Some("again-1"));
        assert_eq!(slot.settle(), settled(true, None));
    }
}
