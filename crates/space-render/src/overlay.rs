//! The 2D overlay: solid rectangles and text in a built-in bitmap font, drawn
//! on top of the finished frame.
//!
//! An [`Overlay`] is a list of rectangles in **output pixels** (origin at the
//! top-left corner, +y down). The renderer draws it after the tone map,
//! alpha-blended straight onto the final 8-bit image, so it is not affected by
//! exposure, fog or lighting and looks the same whether the frame is read back
//! or presented. Colours are display-referred sRGB with straight alpha.
//!
//! Text is drawn with the font in this file: printable ASCII in a 5x7 dot
//! matrix, authored here as plain `const` data (no font file, no third-party
//! glyph data). Every lit dot becomes a rectangle `scale` pixels on a side
//! (runs of dots in a row are merged), which is why a scale is an integer.
//!
//! An empty overlay costs nothing: the renderer does not even begin the
//! overlay pass for it.

/// Width of a glyph in dots.
pub const GLYPH_WIDTH: u32 = 5;
/// Height of a glyph in dots.
pub const GLYPH_HEIGHT: u32 = 7;
/// Dots from one glyph's left edge to the next one's (one dot of spacing).
pub const GLYPH_ADVANCE: u32 = GLYPH_WIDTH + 1;
/// Dots from one line's top to the next one's (two dots of leading).
pub const LINE_ADVANCE: u32 = GLYPH_HEIGHT + 2;

/// One overlay rectangle, in output pixels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct OverlayRect {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    /// Display-referred sRGB, straight alpha.
    pub color: [f32; 4],
}

/// A list of 2D shapes drawn over the finished frame. See the module docs.
#[derive(Clone, Debug, Default)]
pub struct Overlay {
    rects: Vec<OverlayRect>,
}

impl Overlay {
    /// An empty overlay.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// True when nothing would be drawn.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rects.is_empty()
    }

    /// Remove everything.
    pub fn clear(&mut self) {
        self.rects.clear();
    }

    /// Solid rectangle in OUTPUT pixels, origin top-left, +y down. `color` is
    /// display-referred sRGB with straight alpha.
    pub fn rect(&mut self, x: f32, y: f32, width: f32, height: f32, color: [f32; 4]) {
        // Nothing to draw (this also drops NaN sizes), or nothing drawable.
        let finite = [x, y, width, height].iter().all(|v| v.is_finite());
        if !finite || width <= 0.0 || height <= 0.0 {
            return;
        }
        self.rects.push(OverlayRect {
            x,
            y,
            width,
            height,
            color,
        });
    }

    /// Text in the built-in bitmap font; (`x`, `y`) is the top-left of the
    /// first glyph; `scale` is an integer pixel multiple (0 draws like 1);
    /// `'\n'` starts a new line; characters without a glyph draw a placeholder
    /// box.
    pub fn text(&mut self, x: f32, y: f32, scale: u32, color: [f32; 4], text: &str) {
        let scale = scale.max(1) as f32;
        let (mut column, mut line) = (0u32, 0u32);
        for ch in text.chars() {
            if ch == '\n' {
                column = 0;
                line += 1;
                continue;
            }
            let rows = glyph(ch);
            let left = x + (column * GLYPH_ADVANCE) as f32 * scale;
            let top = y + (line * LINE_ADVANCE) as f32 * scale;
            for (row, bits) in rows.iter().enumerate() {
                // One rectangle per horizontal run of lit dots.
                let mut dot = 0;
                while dot < GLYPH_WIDTH {
                    if !row_bit(*bits, dot) {
                        dot += 1;
                        continue;
                    }
                    let start = dot;
                    while dot < GLYPH_WIDTH && row_bit(*bits, dot) {
                        dot += 1;
                    }
                    self.rect(
                        left + start as f32 * scale,
                        top + row as f32 * scale,
                        (dot - start) as f32 * scale,
                        scale,
                        color,
                    );
                }
            }
            column += 1;
        }
    }

    /// Width and height in pixels that `text` would cover at `scale`: from the
    /// left edge of the first glyph to the right edge of the longest line's
    /// last one, and from the top of the first line to the bottom of the last.
    /// The empty string covers nothing.
    #[must_use]
    pub fn text_size(text: &str, scale: u32) -> (f32, f32) {
        if text.is_empty() {
            return (0.0, 0.0);
        }
        let scale = scale.max(1);
        let (mut lines, mut longest) = (0u32, 0u32);
        for line in text.split('\n') {
            lines += 1;
            longest = longest.max(line.chars().count() as u32);
        }
        // The last glyph of a line has no spacing after it, the last line no
        // leading below it.
        let width = (longest * GLYPH_ADVANCE).saturating_sub(1) * scale;
        let height = (lines * LINE_ADVANCE - (LINE_ADVANCE - GLYPH_HEIGHT)) * scale;
        (width as f32, height as f32)
    }

    /// The rectangles, in draw order (later ones on top).
    pub(crate) fn rects(&self) -> &[OverlayRect] {
        &self.rects
    }
}

/// Is dot `x` (0 = leftmost) of a glyph row lit?
fn row_bit(bits: u8, x: u32) -> bool {
    x < GLYPH_WIDTH && bits & (1 << (GLYPH_WIDTH - 1 - x)) != 0
}

/// The seven rows of `ch`'s glyph, top first; in each row bit 4 is the
/// leftmost dot. A character outside printable ASCII gets the placeholder box.
#[must_use]
pub fn glyph(ch: char) -> [u8; 7] {
    match ch {
        ' '..='~' => FONT[ch as usize - 32],
        _ => PLACEHOLDER,
    }
}

/// Is dot (`x`, `y`) of `ch`'s glyph lit? False outside the 5x7 cell.
#[must_use]
pub fn glyph_pixel(ch: char, x: u32, y: u32) -> bool {
    y < GLYPH_HEIGHT && row_bit(glyph(ch)[y as usize], x)
}

/// Drawn for a character the font has no glyph for.
const PLACEHOLDER: [u8; 7] = [
    0b11111, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11111,
];

/// The font: printable ASCII 32..=126, one 5x7 dot matrix per character.
#[rustfmt::skip]
const FONT: [[u8; 7]; 95] = [
    // ' '
    [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000],
    // '!'
    [0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00000, 0b00100],
    // '"'
    [0b01010, 0b01010, 0b01010, 0b00000, 0b00000, 0b00000, 0b00000],
    // '#'
    [0b01010, 0b01010, 0b11111, 0b01010, 0b11111, 0b01010, 0b01010],
    // '$'
    [0b00100, 0b01111, 0b10100, 0b01110, 0b00101, 0b11110, 0b00100],
    // '%'
    [0b11000, 0b11001, 0b00010, 0b00100, 0b01000, 0b10011, 0b00011],
    // '&'
    [0b01100, 0b10010, 0b10100, 0b01000, 0b10101, 0b10010, 0b01101],
    // '\''
    [0b00100, 0b00100, 0b01000, 0b00000, 0b00000, 0b00000, 0b00000],
    // '('
    [0b00010, 0b00100, 0b01000, 0b01000, 0b01000, 0b00100, 0b00010],
    // ')'
    [0b01000, 0b00100, 0b00010, 0b00010, 0b00010, 0b00100, 0b01000],
    // '*'
    [0b00000, 0b00100, 0b10101, 0b01110, 0b10101, 0b00100, 0b00000],
    // '+'
    [0b00000, 0b00100, 0b00100, 0b11111, 0b00100, 0b00100, 0b00000],
    // ','
    [0b00000, 0b00000, 0b00000, 0b00000, 0b01100, 0b00100, 0b01000],
    // '-'
    [0b00000, 0b00000, 0b00000, 0b11111, 0b00000, 0b00000, 0b00000],
    // '.'
    [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b01100, 0b01100],
    // '/'
    [0b00000, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b00000],
    // '0'
    [0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110],
    // '1'
    [0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
    // '2'
    [0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111],
    // '3'
    [0b11111, 0b00010, 0b00100, 0b00010, 0b00001, 0b10001, 0b01110],
    // '4'
    [0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010],
    // '5'
    [0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110],
    // '6'
    [0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110],
    // '7'
    [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000],
    // '8'
    [0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110],
    // '9'
    [0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100],
    // ':'
    [0b00000, 0b01100, 0b01100, 0b00000, 0b01100, 0b01100, 0b00000],
    // ';'
    [0b00000, 0b01100, 0b01100, 0b00000, 0b01100, 0b00100, 0b01000],
    // '<'
    [0b00010, 0b00100, 0b01000, 0b10000, 0b01000, 0b00100, 0b00010],
    // '='
    [0b00000, 0b00000, 0b11111, 0b00000, 0b11111, 0b00000, 0b00000],
    // '>'
    [0b01000, 0b00100, 0b00010, 0b00001, 0b00010, 0b00100, 0b01000],
    // '?'
    [0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b00000, 0b00100],
    // '@'
    [0b01110, 0b10001, 0b00001, 0b01101, 0b10101, 0b10101, 0b01110],
    // 'A'
    [0b01110, 0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001],
    // 'B'
    [0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110],
    // 'C'
    [0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110],
    // 'D'
    [0b11100, 0b10010, 0b10001, 0b10001, 0b10001, 0b10010, 0b11100],
    // 'E'
    [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111],
    // 'F'
    [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000],
    // 'G'
    [0b01110, 0b10001, 0b10000, 0b10111, 0b10001, 0b10001, 0b01111],
    // 'H'
    [0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
    // 'I'
    [0b01110, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
    // 'J'
    [0b00111, 0b00010, 0b00010, 0b00010, 0b00010, 0b10010, 0b01100],
    // 'K'
    [0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001],
    // 'L'
    [0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111],
    // 'M'
    [0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001],
    // 'N'
    [0b10001, 0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001],
    // 'O'
    [0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
    // 'P'
    [0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000],
    // 'Q'
    [0b01110, 0b10001, 0b10001, 0b10001, 0b10101, 0b10010, 0b01101],
    // 'R'
    [0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001],
    // 'S'
    [0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110],
    // 'T'
    [0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100],
    // 'U'
    [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
    // 'V'
    [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100],
    // 'W'
    [0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b10101, 0b01010],
    // 'X'
    [0b10001, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b10001],
    // 'Y'
    [0b10001, 0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100],
    // 'Z'
    [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111],
    // '['
    [0b01110, 0b01000, 0b01000, 0b01000, 0b01000, 0b01000, 0b01110],
    // '\\'
    [0b00000, 0b10000, 0b01000, 0b00100, 0b00010, 0b00001, 0b00000],
    // ']'
    [0b01110, 0b00010, 0b00010, 0b00010, 0b00010, 0b00010, 0b01110],
    // '^'
    [0b00100, 0b01010, 0b10001, 0b00000, 0b00000, 0b00000, 0b00000],
    // '_'
    [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b11111],
    // '`'
    [0b01000, 0b00100, 0b00010, 0b00000, 0b00000, 0b00000, 0b00000],
    // 'a'
    [0b00000, 0b00000, 0b01110, 0b00001, 0b01111, 0b10001, 0b01111],
    // 'b'
    [0b10000, 0b10000, 0b10110, 0b11001, 0b10001, 0b10001, 0b11110],
    // 'c'
    [0b00000, 0b00000, 0b01110, 0b10000, 0b10000, 0b10001, 0b01110],
    // 'd'
    [0b00001, 0b00001, 0b01101, 0b10011, 0b10001, 0b10001, 0b01111],
    // 'e'
    [0b00000, 0b00000, 0b01110, 0b10001, 0b11111, 0b10000, 0b01110],
    // 'f'
    [0b00110, 0b01001, 0b01000, 0b11100, 0b01000, 0b01000, 0b01000],
    // 'g'
    [0b00000, 0b00000, 0b01111, 0b10001, 0b01111, 0b00001, 0b01110],
    // 'h'
    [0b10000, 0b10000, 0b10110, 0b11001, 0b10001, 0b10001, 0b10001],
    // 'i'
    [0b00100, 0b00000, 0b01100, 0b00100, 0b00100, 0b00100, 0b01110],
    // 'j'
    [0b00010, 0b00000, 0b00110, 0b00010, 0b00010, 0b10010, 0b01100],
    // 'k'
    [0b10000, 0b10000, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010],
    // 'l'
    [0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
    // 'm'
    [0b00000, 0b00000, 0b11010, 0b10101, 0b10101, 0b10001, 0b10001],
    // 'n'
    [0b00000, 0b00000, 0b10110, 0b11001, 0b10001, 0b10001, 0b10001],
    // 'o'
    [0b00000, 0b00000, 0b01110, 0b10001, 0b10001, 0b10001, 0b01110],
    // 'p'
    [0b00000, 0b00000, 0b11110, 0b10001, 0b11110, 0b10000, 0b10000],
    // 'q'
    [0b00000, 0b00000, 0b01101, 0b10011, 0b01111, 0b00001, 0b00001],
    // 'r'
    [0b00000, 0b00000, 0b10110, 0b11001, 0b10000, 0b10000, 0b10000],
    // 's'
    [0b00000, 0b00000, 0b01110, 0b10000, 0b01110, 0b00001, 0b11110],
    // 't'
    [0b01000, 0b01000, 0b11100, 0b01000, 0b01000, 0b01001, 0b00110],
    // 'u'
    [0b00000, 0b00000, 0b10001, 0b10001, 0b10001, 0b10011, 0b01101],
    // 'v'
    [0b00000, 0b00000, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100],
    // 'w'
    [0b00000, 0b00000, 0b10001, 0b10001, 0b10101, 0b10101, 0b01010],
    // 'x'
    [0b00000, 0b00000, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001],
    // 'y'
    [0b00000, 0b00000, 0b10001, 0b10001, 0b01111, 0b00001, 0b01110],
    // 'z'
    [0b00000, 0b00000, 0b11111, 0b00010, 0b00100, 0b01000, 0b11111],
    // '{'
    [0b00010, 0b00100, 0b00100, 0b01000, 0b00100, 0b00100, 0b00010],
    // '|'
    [0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100],
    // '}'
    [0b01000, 0b00100, 0b00100, 0b00010, 0b00100, 0b00100, 0b01000],
    // '~'
    [0b00000, 0b00000, 0b01000, 0b10101, 0b00010, 0b00000, 0b00000],
];

#[cfg(test)]
mod tests {
    use super::*;

    const WHITE: [f32; 4] = [1.0; 4];

    /// The pixels `overlay` covers on a `width` x `height` canvas (a software
    /// rasteriser with the GPU's rule: a pixel is covered when its centre is
    /// inside a rectangle).
    fn coverage(overlay: &Overlay, width: usize, height: usize) -> Vec<bool> {
        let mut lit = vec![false; width * height];
        for r in overlay.rects() {
            for y in 0..height {
                for x in 0..width {
                    let (cx, cy) = (x as f32 + 0.5, y as f32 + 0.5);
                    if cx >= r.x && cx < r.x + r.width && cy >= r.y && cy < r.y + r.height {
                        lit[y * width + x] = true;
                    }
                }
            }
        }
        lit
    }

    #[test]
    fn text_size_counts_glyph_cells_without_trailing_spacing() {
        assert_eq!(Overlay::text_size("", 3), (0.0, 0.0));
        assert_eq!(Overlay::text_size("A", 1), (5.0, 7.0));
        assert_eq!(Overlay::text_size("A", 3), (15.0, 21.0));
        // Two glyphs: 5 + 1 + 5 dots.
        assert_eq!(Overlay::text_size("AB", 2), (22.0, 14.0));
        // The longest line counts; two lines are 7 + 2 + 7 dots tall.
        assert_eq!(Overlay::text_size("AB\nABCD", 1), (23.0, 16.0));
        assert_eq!(Overlay::text_size("A\n", 1), (5.0, 16.0));
        // Scale 0 measures (and draws) like scale 1.
        assert_eq!(Overlay::text_size("A", 0), (5.0, 7.0));
        // Every character is one cell, glyph or not.
        assert_eq!(Overlay::text_size("é√", 1), Overlay::text_size("ab", 1));
    }

    /// Text covers exactly the font's lit dots, scaled, at the given origin —
    /// and never anything outside `text_size`.
    #[test]
    fn text_draws_the_font_dots_inside_its_measured_box() {
        let text = "192.168.0.1:7777\nOk?";
        for scale in [1u32, 2, 3] {
            let (ox, oy) = (3usize, 2usize);
            let (w, h) = Overlay::text_size(text, scale);
            let (width, height) = (w as usize + 8, h as usize + 8);
            let mut overlay = Overlay::new();
            assert!(overlay.is_empty());
            overlay.text(ox as f32, oy as f32, scale, WHITE, text);
            assert!(!overlay.is_empty());
            let lit = coverage(&overlay, width, height);
            let s = scale as usize;
            let mut expected = vec![false; width * height];
            for (line_index, line) in text.split('\n').enumerate() {
                for (column, ch) in line.chars().enumerate() {
                    for gy in 0..7 {
                        for gx in 0..5 {
                            if !glyph_pixel(ch, gx as u32, gy as u32) {
                                continue;
                            }
                            let x0 = ox + (column * 6 + gx) * s;
                            let y0 = oy + (line_index * 9 + gy) * s;
                            for y in y0..y0 + s {
                                for x in x0..x0 + s {
                                    expected[y * width + x] = true;
                                }
                            }
                        }
                    }
                }
            }
            assert!(lit == expected, "scale {scale}");
            for y in 0..height {
                for x in 0..width {
                    let inside = x >= ox && x < ox + w as usize && y >= oy && y < oy + h as usize;
                    assert!(inside || !lit[y * width + x], "({x}, {y}) at scale {scale}");
                }
            }
            // The box is tight: something is lit on each of its four edges.
            let column_lit = |x: usize| (0..height).any(|y| lit[y * width + x]);
            let row_lit = |y: usize| (0..width).any(|x| lit[y * width + x]);
            assert!(column_lit(ox) && column_lit(ox + w as usize - 1));
            assert!(row_lit(oy) && row_lit(oy + h as usize - 1));
        }
        let mut overlay = Overlay::new();
        overlay.text(0.0, 0.0, 2, WHITE, "x");
        overlay.clear();
        assert!(overlay.is_empty());
    }

    #[test]
    fn rects_keep_their_order_and_degenerate_ones_are_dropped() {
        let mut overlay = Overlay::new();
        overlay.rect(1.0, 2.0, 3.0, 4.0, [1.0, 0.0, 0.0, 0.5]);
        overlay.rect(0.0, 0.0, 0.0, 4.0, WHITE);
        overlay.rect(0.0, 0.0, 4.0, -1.0, WHITE);
        overlay.rect(f32::NAN, 0.0, 4.0, 1.0, WHITE);
        overlay.rect(0.0, 0.0, f32::INFINITY, 1.0, WHITE);
        overlay.rect(5.0, 6.0, 7.0, 8.0, WHITE);
        let rects = overlay.rects();
        assert_eq!(rects.len(), 2);
        assert_eq!(
            (rects[0].x, rects[0].y, rects[0].width, rects[0].height),
            (1.0, 2.0, 3.0, 4.0)
        );
        assert_eq!(rects[0].color, [1.0, 0.0, 0.0, 0.5]);
        assert_eq!(rects[1].x, 5.0);
    }

    /// Every printable character has a glyph that fits the 5x7 cell, no two
    /// characters share one, only the space is blank, and anything else draws
    /// the placeholder box.
    #[test]
    fn the_font_covers_printable_ascii_with_distinct_glyphs() {
        let mut seen = std::collections::HashMap::new();
        for code in 32u8..=126 {
            let ch = code as char;
            let rows = glyph(ch);
            assert!(rows.iter().all(|r| *r < 32), "{ch:?} is wider than 5 dots");
            let dots: u32 = rows.iter().map(|r| r.count_ones()).sum();
            assert_eq!(dots == 0, ch == ' ', "{ch:?} has {dots} dots");
            assert!(rows != PLACEHOLDER, "{ch:?} is the placeholder");
            if let Some(other) = seen.insert(rows, ch) {
                panic!("{ch:?} and {other:?} have the same glyph");
            }
        }
        for ch in ['\t', '\u{7f}', 'é', '€', '\0'] {
            assert_eq!(glyph(ch), PLACEHOLDER, "{ch:?}");
        }
        // Spot checks of the characters an address is made of.
        assert_eq!(glyph('1')[0], 0b00100);
        assert_eq!(glyph('.')[5..], [0b01100, 0b01100]);
        assert_eq!(glyph(':')[1..3], [0b01100, 0b01100]);
        assert!(glyph_pixel('L', 0, 0) && !glyph_pixel('L', 1, 0));
        assert!(!glyph_pixel('L', 5, 0) && !glyph_pixel('L', 0, 7));
        // Confusable pairs differ: O/0, l/1/I, 5/S, 8/B, 2/Z.
        for (a, b) in [
            ('O', '0'),
            ('l', '1'),
            ('I', 'l'),
            ('5', 'S'),
            ('8', 'B'),
            ('2', 'Z'),
        ] {
            assert_ne!(glyph(a), glyph(b), "{a} vs {b}");
        }
    }
}
