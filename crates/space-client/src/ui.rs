//! What the client draws in 2D: the connect screen and the heads-up display.
//!
//! Both are drawn into a [`space_render::Overlay`] — rectangles and text in
//! the renderer's built-in bitmap font, composited after tone mapping — so
//! they look exactly the same in the window, in a headless frame and in the
//! browser stream. Everything is sized in units of [`ui_scale`] pixels, an
//! integer chosen from the frame size, so the text stays crisp and legible
//! from 320x240 to 4K.

use space_render::overlay::{GLYPH_ADVANCE, GLYPH_HEIGHT, LINE_ADVANCE};
use space_render::Overlay;

type Color = [f32; 4];

const WHITE: Color = [0.93, 0.95, 1.0, 1.0];
const DIM: Color = [0.58, 0.64, 0.76, 1.0];
const FAINT: Color = [0.40, 0.46, 0.58, 1.0];
const ACCENT: Color = [0.30, 0.74, 1.0, 1.0];
const RED: Color = [1.0, 0.42, 0.38, 1.0];
const AMBER: Color = [1.0, 0.80, 0.36, 1.0];
const PANEL: Color = [0.055, 0.075, 0.125, 0.94];
const PANEL_EDGE: Color = [0.20, 0.30, 0.48, 1.0];
const FIELD: Color = [0.02, 0.03, 0.055, 1.0];
const HUD_PANEL: Color = [0.02, 0.03, 0.06, 0.62];

/// The title on the connect screen.
pub const TITLE: &str = "SPACE";

/// The pixel size of one UI unit for a `width` x `height` frame: 1 up to
/// about 960x540, 2 at 1280x720, 3 at 1920x1080, 4 at 2560x1440 and up.
#[must_use]
pub fn ui_scale(width: u32, height: u32) -> u32 {
    (width / 640).min(height / 360).clamp(1, 4)
}

/// The width of `chars` characters of text at `scale`, pixels.
fn text_width(chars: usize, scale: u32) -> f32 {
    let chars = u32::try_from(chars).unwrap_or(u32::MAX);
    (chars.saturating_mul(GLYPH_ADVANCE).saturating_sub(1) * scale) as f32
}

/// How many characters fit into `width` pixels at `scale` (at least 1).
fn columns(width: f32, scale: u32) -> usize {
    let advance = (GLYPH_ADVANCE * scale) as f32;
    // The last glyph needs no spacing after it.
    (((width + scale as f32) / advance).floor().max(1.0)) as usize
}

/// Characters the built-in font cannot draw become `?` (it would draw a
/// placeholder box, which reads as a rendering fault).
fn printable(text: &str) -> String {
    text.chars()
        .map(|c| if (' '..='~').contains(&c) { c } else { '?' })
        .collect()
}

/// Break `text` into lines of at most `columns` characters, at spaces where
/// possible, and at most `max_lines` of them (the last one ends in `...` when
/// text had to be left out).
#[must_use]
pub fn wrap(text: &str, columns: usize, max_lines: usize) -> Vec<String> {
    let columns = columns.max(4);
    let mut lines: Vec<String> = Vec::new();
    let mut line = String::new();
    for word in text.split_whitespace() {
        let mut word: Vec<char> = word.chars().collect();
        loop {
            let used = line.chars().count();
            let space = usize::from(used > 0);
            if used + space + word.len() <= columns {
                if space == 1 {
                    line.push(' ');
                }
                line.extend(word.iter());
                break;
            }
            if used > 0 {
                lines.push(std::mem::take(&mut line));
            } else {
                // A word longer than a line: cut it.
                line.extend(word.drain(..columns));
                lines.push(std::mem::take(&mut line));
            }
        }
    }
    if !line.is_empty() {
        lines.push(line);
    }
    if lines.len() > max_lines.max(1) {
        lines.truncate(max_lines.max(1));
        if let Some(last) = lines.last_mut() {
            let keep = last.chars().count().min(columns.saturating_sub(3));
            *last = last.chars().take(keep).collect::<String>() + "...";
        }
    }
    lines
}

/// What the connect screen shows under the address field.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Status<'a> {
    /// Waiting for the user: the key hint only.
    Idle,
    /// A connection attempt to this address has been running this long.
    Connecting {
        /// Where to.
        address: &'a str,
        /// Seconds since it started.
        seconds: f32,
    },
    /// The last attempt failed (or the server was lost): why.
    Error(&'a str),
    /// The last attempt failed (or the server was lost), and the client is
    /// going to try again by itself.
    Retrying {
        /// Why it failed.
        reason: &'a str,
        /// Seconds until the next attempt.
        seconds: f32,
    },
}

/// The connect screen's content.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConnectView<'a> {
    /// The visible part of the address field ([`crate::connect::Prompt::window`]
    /// is asked for [`field_columns`] characters).
    pub text: &'a str,
    /// The caret's column in `text`.
    pub caret: usize,
    /// The line under the field.
    pub status: Status<'a>,
    /// Whether Esc ends the client in this front end (a window: yes; a
    /// client without one: no). The key hint offers only what is true.
    pub escape_quits: bool,
}

/// The key hint at the foot of the connect screen: what Enter and Esc do
/// while it shows `status`.
///
/// `escape_quits` is the front end's rule
/// ([`crate::app::AppConfig::escape_quits`]). Where no key ends the client —
/// a headless or streaming one, whose keys belong to whoever can reach it —
/// the hint does not say "Esc quits": Esc is named only where it does
/// something (it cancels an attempt, it stops the retrying), and Enter is
/// always the way on.
#[must_use]
pub fn connect_keys(status: Status<'_>, escape_quits: bool) -> &'static str {
    match (status, escape_quits) {
        (Status::Idle, true) => "Enter connects    Esc quits",
        (Status::Idle, false) => "Enter connects",
        (Status::Connecting { .. }, _) => "Esc cancels",
        (Status::Retrying { .. }, _) => "Enter retries now    Esc stops retrying",
        (Status::Error(_), true) => "Enter tries again    Esc quits",
        (Status::Error(_), false) => "Enter tries again",
    }
}

/// The geometry of the connect screen for one frame size.
#[derive(Clone, Copy, Debug, PartialEq)]
struct ConnectLayout {
    /// One UI unit, pixels.
    unit: f32,
    scale: u32,
    field_scale: u32,
    title_scale: u32,
    /// The panel.
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    padding: f32,
}

impl ConnectLayout {
    fn new(width: u32, height: u32) -> Self {
        let scale = ui_scale(width, height);
        let unit = scale as f32;
        let (w, h) = (width as f32, height as f32);
        let panel_w = (316.0 * unit).min(w - 8.0).max(8.0);
        let panel_h = (164.0 * unit).min(h - 8.0).max(8.0);
        Self {
            unit,
            scale,
            field_scale: scale + (scale / 2).max(1),
            title_scale: scale * 3,
            x: ((w - panel_w) * 0.5).floor(),
            y: ((h - panel_h) * 0.5).floor(),
            width: panel_w,
            height: panel_h,
            padding: 14.0 * unit,
        }
    }

    fn inner_width(&self) -> f32 {
        self.width - 2.0 * self.padding
    }

    /// Characters the address field shows.
    fn field_columns(&self) -> usize {
        columns(self.inner_width() - 10.0 * self.unit, self.field_scale)
    }
}

/// How many characters of the address the connect screen's field shows in a
/// `width` x `height` frame.
#[must_use]
pub fn field_columns(width: u32, height: u32) -> usize {
    ConnectLayout::new(width, height).field_columns()
}

/// A calm backdrop for the connect screen: a dark vertical gradient with a
/// sparse, fixed field of faint stars. It covers the whole frame.
fn backdrop(overlay: &mut Overlay, width: u32, height: u32) {
    let (w, h) = (width as f32, height as f32);
    const BANDS: u32 = 24;
    let band = (h / BANDS as f32).ceil().max(1.0);
    for i in 0..BANDS {
        let t = i as f32 / (BANDS - 1) as f32;
        let color = [0.012 + 0.030 * t, 0.018 + 0.050 * t, 0.045 + 0.090 * t, 1.0];
        overlay.rect(0.0, i as f32 * band, w, band, color);
    }
    // The stars: a fixed pseudo-random pattern in normalised coordinates, so
    // it is the same picture at every size.
    let unit = ui_scale(width, height) as f32;
    let mut seed = 0x2545_f491_4f6c_dd1du64;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 40) as f32 / (1u64 << 24) as f32
    };
    for _ in 0..90 {
        let (x, y, glow, big) = (next(), next(), next(), next());
        let size = if big > 0.85 { 2.0 * unit } else { unit };
        overlay.rect(
            (x * w).floor(),
            (y * h).floor(),
            size,
            size,
            [0.75, 0.82, 1.0, 0.12 + 0.5 * glow * glow],
        );
    }
}

/// A rectangle with a one-unit border.
fn framed(overlay: &mut Overlay, rect: [f32; 4], unit: f32, fill: Color, edge: Color) {
    let [x, y, w, h] = rect;
    overlay.rect(x, y, w, h, edge);
    overlay.rect(
        x + unit,
        y + unit,
        w - 2.0 * unit,
        h - 2.0 * unit,
        [fill[0], fill[1], fill[2], 1.0],
    );
}

/// Draw the connect screen over the whole frame.
pub fn draw_connect(overlay: &mut Overlay, width: u32, height: u32, view: &ConnectView<'_>) {
    backdrop(overlay, width, height);
    let l = ConnectLayout::new(width, height);
    let u = l.unit;

    // The panel: a soft shadow, an edge, the body.
    overlay.rect(
        l.x + 3.0 * u,
        l.y + 4.0 * u,
        l.width,
        l.height,
        [0.0, 0.0, 0.0, 0.35],
    );
    overlay.rect(l.x, l.y, l.width, l.height, PANEL_EDGE);
    overlay.rect(
        l.x + u,
        l.y + u,
        l.width - 2.0 * u,
        l.height - 2.0 * u,
        PANEL,
    );

    let left = l.x + l.padding;
    let mut y = l.y + l.padding;

    // Title and what this is.
    overlay.text(left, y, l.title_scale, WHITE, TITLE);
    let title_height = (GLYPH_HEIGHT * l.title_scale) as f32;
    let tagline = "connect to a shard";
    let tagline_x = l.x + l.width - l.padding - text_width(tagline.len(), l.scale);
    overlay.text(
        tagline_x.max(left),
        y + title_height - (GLYPH_HEIGHT * l.scale) as f32,
        l.scale,
        DIM,
        tagline,
    );
    y += title_height + 5.0 * u;
    overlay.rect(left, y, l.inner_width(), u, ACCENT);
    y += 12.0 * u;

    // The address field.
    overlay.text(left, y, l.scale, DIM, "Server address");
    y += (LINE_ADVANCE * l.scale) as f32 + 2.0 * u;
    let field_h = (GLYPH_HEIGHT * l.field_scale) as f32 + 10.0 * u;
    let active = !matches!(view.status, Status::Connecting { .. });
    framed(
        overlay,
        [left, y, l.inner_width(), field_h],
        u,
        FIELD,
        if active { ACCENT } else { PANEL_EDGE },
    );
    let (text_x, text_y) = (left + 5.0 * u, y + 5.0 * u);
    overlay.text(
        text_x,
        text_y,
        l.field_scale,
        if active { WHITE } else { DIM },
        &printable(view.text),
    );
    if active {
        // The caret: a bar in the gap before the character it is in front of.
        let advance = (GLYPH_ADVANCE * l.field_scale) as f32;
        let thick = l.field_scale.div_ceil(2) as f32;
        let gap = l.field_scale as f32;
        let caret_x = text_x + view.caret as f32 * advance - ((gap + thick) * 0.5).floor();
        overlay.rect(
            caret_x.max(left + 2.0 * u),
            text_y - 3.0 * u,
            thick,
            (GLYPH_HEIGHT * l.field_scale) as f32 + 6.0 * u,
            ACCENT,
        );
    }
    y += field_h + 8.0 * u;

    // Under the field: what can be typed, how the attempt is going, or why
    // the last one failed. The keys are at the foot of the panel.
    let (color, text) = match view.status {
        Status::Idle => (
            FAINT,
            format!(
                "host, host:port or [ipv6]:port - the port is {} unless you give one",
                crate::address::DEFAULT_PORT
            ),
        ),
        Status::Connecting { address, seconds } => {
            let dots = ".".repeat(1 + (seconds.max(0.0) * 2.0) as usize % 3);
            (AMBER, format!("Connecting to {}{dots}", printable(address)))
        }
        Status::Error(why) | Status::Retrying { reason: why, .. } => (RED, printable(why)),
    };
    let keys = connect_keys(view.status, view.escape_quits);
    let line = (LINE_ADVANCE * l.scale) as f32 + u;
    // While retrying, the last of the three lines is the countdown.
    let countdown = match view.status {
        Status::Retrying { seconds, .. } => Some(format!(
            "retrying in {:.0} s",
            seconds.max(0.0).ceil().max(1.0)
        )),
        _ => None,
    };
    let room = if countdown.is_some() { 2 } else { 3 };
    let lines = wrap(&text, columns(l.inner_width(), l.scale), room);
    for (i, text) in lines.iter().enumerate() {
        overlay.text(left, y + i as f32 * line, l.scale, color, text);
    }
    if let Some(countdown) = countdown {
        overlay.text(
            left,
            y + lines.len() as f32 * line,
            l.scale,
            AMBER,
            &countdown,
        );
    }
    let foot = l.y + l.height - l.padding - (GLYPH_HEIGHT * l.scale) as f32;
    overlay.text(left, foot, l.scale, DIM, keys);
}

/// What the heads-up display shows.
#[derive(Clone, Debug, PartialEq)]
pub struct HudView<'a> {
    /// The stat lines, top to bottom.
    pub lines: &'a [String],
    /// The key hint under them (drawn smaller and dimmer, wrapped).
    pub hint: &'a str,
}

/// Draw the heads-up display in the top-left corner.
pub fn draw_hud(overlay: &mut Overlay, width: u32, height: u32, view: &HudView<'_>) {
    let scale = ui_scale(width, height);
    let hint_scale = (scale - 1).max(1);
    let u = scale as f32;
    let margin = 6.0 * u;
    let padding = 5.0 * u;
    let room = (width as f32 - 2.0 * (margin + padding)).max(8.0);

    let lines: Vec<String> = view
        .lines
        .iter()
        .flat_map(|line| wrap(&printable(line), columns(room, scale), 2))
        .collect();
    let hints = wrap(&printable(view.hint), columns(room, hint_scale).min(64), 4);
    let longest = |lines: &[String], scale: u32| {
        lines
            .iter()
            .map(|line| text_width(line.chars().count(), scale))
            .fold(0.0f32, f32::max)
    };
    let line_h = (LINE_ADVANCE * scale) as f32;
    let hint_h = (LINE_ADVANCE * hint_scale) as f32;
    let gap = if hints.is_empty() || lines.is_empty() {
        0.0
    } else {
        3.0 * u
    };
    let content_w = longest(&lines, scale).max(longest(&hints, hint_scale));
    let content_h = lines.len() as f32 * line_h + gap + hints.len() as f32 * hint_h;
    if content_w <= 0.0 || content_h <= 0.0 {
        return;
    }
    overlay.rect(
        margin,
        margin,
        content_w + 2.0 * padding,
        content_h + 2.0 * padding - (LINE_ADVANCE - GLYPH_HEIGHT) as f32,
        HUD_PANEL,
    );
    let x = margin + padding;
    let mut y = margin + padding;
    for (i, line) in lines.iter().enumerate() {
        overlay.text(x, y, scale, if i == 0 { WHITE } else { DIM_BRIGHT }, line);
        y += line_h;
    }
    y += gap;
    for hint in &hints {
        overlay.text(x, y, hint_scale, DIM, hint);
        y += hint_h;
    }
}

const DIM_BRIGHT: Color = [0.80, 0.86, 0.96, 1.0];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_scale_follows_the_frame_size() {
        for ((w, h), scale) in [
            ((64, 48), 1),
            ((320, 240), 1),
            ((640, 480), 1),
            ((1280, 720), 2),
            ((1600, 900), 2),
            ((1920, 1080), 3),
            ((2560, 1440), 4),
            ((7680, 4320), 4),
            // A very wide or very tall frame is sized by its short side.
            ((4000, 400), 1),
            ((700, 3000), 1),
        ] {
            assert_eq!(ui_scale(w, h), scale, "{w}x{h}");
        }
    }

    #[test]
    fn text_is_wrapped_at_spaces_and_bounded() {
        assert_eq!(wrap("one two three", 7, 9), ["one two", "three"]);
        assert_eq!(wrap("one two three", 100, 9), ["one two three"]);
        assert_eq!(wrap("", 10, 3), Vec::<String>::new());
        // A word longer than a line is cut, not lost.
        assert_eq!(wrap("abcdefghij kl", 4, 9), ["abcd", "efgh", "ij", "kl"]);
        // Too many lines: the last one says so.
        let cut = wrap("aaaa bbbb cccc dddd", 4, 2);
        assert_eq!(cut.len(), 2);
        assert!(cut[1].ends_with("..."), "{cut:?}");
        for line in wrap(
            "no answer from 203.0.113.250:7777 (is a shard running there?)",
            24,
            3,
        ) {
            assert!(line.chars().count() <= 24, "{line:?}");
        }
    }

    #[test]
    fn the_field_shows_a_whole_address_at_every_size() {
        for (w, h) in [
            (320, 240),
            (640, 480),
            (1280, 720),
            (1920, 1080),
            (3840, 2160),
        ] {
            let columns = field_columns(w, h);
            assert!(
                columns >= "255.255.255.255:65535".len(),
                "{w}x{h}: only {columns} characters"
            );
        }
        // Even a postage stamp has a field.
        assert!(field_columns(16, 16) >= 1);
    }

    /// The connect screen is drawn for every status and every size without
    /// panicking, covers the frame, and shows a caret exactly when the field
    /// can be typed into.
    #[test]
    fn the_connect_screen_draws_at_every_size_and_status() {
        let statuses = [
            Status::Idle,
            Status::Connecting {
                address: "shard.example.org:7777",
                seconds: 1.3,
            },
            Status::Error("no answer from 127.0.0.1:1 (is a shard running there?)"),
            Status::Retrying {
                reason: "no answer from 127.0.0.1:1 (is a shard running there?)",
                seconds: 2.4,
            },
        ];
        for (w, h) in [
            (1, 1),
            (16, 9),
            (160, 120),
            (640, 480),
            (1280, 720),
            (3840, 2160),
        ] {
            for status in statuses {
                for escape_quits in [true, false] {
                    let mut overlay = Overlay::new();
                    draw_connect(
                        &mut overlay,
                        w,
                        h,
                        &ConnectView {
                            text: "127.0.0.1:7777",
                            caret: 14,
                            status,
                            escape_quits,
                        },
                    );
                    assert!(!overlay.is_empty(), "{w}x{h} {status:?}");
                }
            }
        }
        assert_eq!(printable("caf\u{e9} \u{1f680}"), "caf? ?");
    }

    /// The key hint tells the truth for the front end it is drawn in: "Esc
    /// quits" only where Esc quits, and Enter is always offered where Enter
    /// does something.
    #[test]
    fn the_key_hint_offers_esc_only_for_what_esc_does() {
        let connecting = Status::Connecting {
            address: "127.0.0.1:7777",
            seconds: 0.4,
        };
        let error = Status::Error("no answer from 127.0.0.1:7777");
        let retrying = Status::Retrying {
            reason: "no answer from 127.0.0.1:7777",
            seconds: 1.6,
        };
        // A window: Esc quits wherever nothing else is going on.
        assert_eq!(
            connect_keys(Status::Idle, true),
            "Enter connects    Esc quits"
        );
        assert_eq!(connect_keys(error, true), "Enter tries again    Esc quits");
        // No window: no key ends the client, and no hint says one does.
        assert_eq!(connect_keys(Status::Idle, false), "Enter connects");
        assert_eq!(connect_keys(error, false), "Enter tries again");
        for quits in [true, false] {
            // What Esc does in every front end.
            assert_eq!(connect_keys(connecting, quits), "Esc cancels");
            assert_eq!(
                connect_keys(retrying, quits),
                "Enter retries now    Esc stops retrying"
            );
            for status in [Status::Idle, connecting, error, retrying] {
                let keys = connect_keys(status, quits);
                let idle = matches!(status, Status::Idle | Status::Error(_));
                assert_eq!(keys.contains("quit"), quits && idle, "{status:?}: {keys}");
                // Esc is never named without saying what it does, and Enter
                // is the way on from every screen that takes it.
                assert_eq!(
                    keys.contains("Enter"),
                    !matches!(status, Status::Connecting { .. }),
                    "{status:?}: {keys}"
                );
                // It fits the foot of the narrowest panel that shows text.
                let layout = ConnectLayout::new(640, 480);
                assert!(
                    keys.len() <= columns(layout.inner_width(), layout.scale),
                    "{keys:?} is wider than the panel"
                );
            }
        }
    }

    #[test]
    fn the_hud_draws_and_an_empty_one_draws_nothing() {
        let lines = vec![
            "60 fps  16.7 ms".to_owned(),
            "shard 1  handoffs 0".to_owned(),
        ];
        for (w, h) in [(1, 1), (160, 120), (640, 480), (1280, 720)] {
            let mut overlay = Overlay::new();
            draw_hud(
                &mut overlay,
                w,
                h,
                &HudView {
                    lines: &lines,
                    hint: crate::input::KEY_HELP,
                },
            );
            assert!(!overlay.is_empty(), "{w}x{h}");
        }
        let mut overlay = Overlay::new();
        draw_hud(
            &mut overlay,
            640,
            480,
            &HudView {
                lines: &[],
                hint: "",
            },
        );
        assert!(overlay.is_empty());
    }
}
