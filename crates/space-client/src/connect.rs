//! The connect screen's state: one text field (the server address) with a
//! caret, and the error line of the last failed attempt.
//!
//! Pure state, no drawing and no network: [`crate::ui`] draws it and
//! [`crate::app::App`] acts on what it returns. Every front end — the window's
//! keyboard, the browser viewer, the harness — edits it through the same two
//! calls, [`Prompt::insert`] and [`Prompt::stroke`].

use space_stream::protocol::is_typeable;
use space_stream::Key;

/// Most characters the address field holds (a DNS name is at most 253, plus a
/// port).
pub const MAX_ADDRESS_CHARS: usize = 260;

/// What a key stroke in the field asks the application to do.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PromptAction {
    /// Enter: connect to the address in the field.
    Submit,
    /// Escape: step back from the connect screen. What that amounts to is
    /// the application's call: it stops a retry loop, and it ends the client
    /// only where the front end lets its user do that
    /// ([`crate::app::AppConfig::escape_quits`]).
    Back,
}

/// The address field and the error line under it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Prompt {
    text: Vec<char>,
    /// Index into `text` the next character is inserted at (`0..=len`).
    caret: usize,
    error: Option<String>,
}

impl Prompt {
    /// A field holding `prefill`, the caret behind it, no error.
    #[must_use]
    pub fn new(prefill: &str) -> Self {
        let mut prompt = Self {
            text: Vec::new(),
            caret: 0,
            error: None,
        };
        prompt.set_text(prefill);
        prompt
    }

    /// What is typed.
    #[must_use]
    pub fn text(&self) -> String {
        self.text.iter().collect()
    }

    /// The caret position, in characters from the start (`0..=len`).
    #[must_use]
    pub fn caret(&self) -> usize {
        self.caret
    }

    /// Number of characters typed.
    #[must_use]
    pub fn len(&self) -> usize {
        self.text.len()
    }

    /// Whether nothing is typed.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.text.is_empty()
    }

    /// The error line, if the last attempt failed and the text has not been
    /// edited since.
    #[must_use]
    pub fn error(&self) -> Option<&str> {
        self.error.as_deref()
    }

    /// Replace the text (characters the field cannot hold are dropped), put
    /// the caret behind it and clear the error.
    pub fn set_text(&mut self, text: &str) {
        self.text = text
            .chars()
            .filter(|c| accepts(*c))
            .take(MAX_ADDRESS_CHARS)
            .collect();
        self.caret = self.text.len();
        self.error = None;
    }

    /// Show `error` under the field. The text and the caret stay as they are.
    pub fn set_error(&mut self, error: impl Into<String>) {
        self.error = Some(error.into());
    }

    /// Remove the error line.
    pub fn clear_error(&mut self) {
        self.error = None;
    }

    /// Type one character at the caret. White space and control characters
    /// are not part of an address and are ignored (so a pasted address may
    /// carry a trailing newline), as is anything typed into a full field.
    /// Returns whether the text changed; a change clears the error line.
    pub fn insert(&mut self, c: char) -> bool {
        if !accepts(c) || self.text.len() >= MAX_ADDRESS_CHARS {
            return false;
        }
        self.text.insert(self.caret, c);
        self.caret += 1;
        self.error = None;
        true
    }

    /// One stroke of a text-entry key (movement keys do nothing here).
    /// Editing clears the error line; moving the caret does not.
    pub fn stroke(&mut self, key: Key) -> Option<PromptAction> {
        match key {
            Key::Enter => return Some(PromptAction::Submit),
            Key::Escape => return Some(PromptAction::Back),
            Key::Backspace if self.caret > 0 => {
                self.caret -= 1;
                self.text.remove(self.caret);
                self.error = None;
            }
            Key::Delete if self.caret < self.text.len() => {
                self.text.remove(self.caret);
                self.error = None;
            }
            Key::CaretLeft => self.caret = self.caret.saturating_sub(1),
            Key::CaretRight => self.caret = (self.caret + 1).min(self.text.len()),
            Key::Home => self.caret = 0,
            Key::End => self.caret = self.text.len(),
            _ => {}
        }
        None
    }

    /// The part of the text a field `columns` characters wide shows, and the
    /// caret's column in it (`0..=columns`, so the caret is always visible; a
    /// caret behind the last visible character sits at the field's right
    /// edge). A text that fits is shown whole; a longer one scrolls with the
    /// caret.
    #[must_use]
    pub fn window(&self, columns: usize) -> (String, usize) {
        let columns = columns.max(1);
        if self.text.len() < columns {
            return (self.text(), self.caret);
        }
        // Keep the caret inside the window, as far right as the text allows:
        // the window ends one cell past the caret when the caret is at the
        // end of the text (room for the caret itself).
        let start = (self.caret + 1).saturating_sub(columns);
        let end = (start + columns).min(self.text.len());
        (self.text[start..end].iter().collect(), self.caret - start)
    }
}

/// Whether the address field takes `c`.
fn accepts(c: char) -> bool {
    is_typeable(c) && !c.is_whitespace()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn typed(prompt: &mut Prompt, text: &str) {
        for c in text.chars() {
            prompt.insert(c);
        }
    }

    #[test]
    fn typing_inserts_at_the_caret() {
        let mut p = Prompt::new("");
        assert!(p.is_empty());
        typed(&mut p, "10.0.0.5");
        assert_eq!((p.text().as_str(), p.caret()), ("10.0.0.5", 8));
        // Go to the front, type there.
        assert_eq!(p.stroke(Key::Home), None);
        typed(&mut p, "x");
        assert_eq!((p.text().as_str(), p.caret()), ("x10.0.0.5", 1));
        p.stroke(Key::End);
        typed(&mut p, ":7");
        assert_eq!(p.text(), "x10.0.0.5:7");
        assert_eq!(p.len(), 11);
    }

    #[test]
    fn the_prefill_is_editable_with_the_caret_behind_it() {
        let mut p = Prompt::new("127.0.0.1:7777");
        assert_eq!(p.caret(), 14);
        p.stroke(Key::Backspace);
        p.stroke(Key::Backspace);
        typed(&mut p, "88");
        assert_eq!(p.text(), "127.0.0.1:7788");
    }

    #[test]
    fn backspace_delete_and_the_caret_keys() {
        let mut p = Prompt::new("abcd");
        p.stroke(Key::CaretLeft);
        p.stroke(Key::CaretLeft);
        assert_eq!(p.caret(), 2);
        p.stroke(Key::Backspace);
        assert_eq!((p.text().as_str(), p.caret()), ("acd", 1));
        p.stroke(Key::Delete);
        assert_eq!((p.text().as_str(), p.caret()), ("ad", 1));
        p.stroke(Key::CaretRight);
        p.stroke(Key::CaretRight);
        p.stroke(Key::CaretRight);
        assert_eq!(p.caret(), 2, "the caret stops at the end");
        p.stroke(Key::Delete);
        assert_eq!(p.text(), "ad", "nothing behind the caret to delete");
        p.stroke(Key::Home);
        p.stroke(Key::Backspace);
        p.stroke(Key::CaretLeft);
        assert_eq!((p.text().as_str(), p.caret()), ("ad", 0));
        // Keys that mean nothing in a text field.
        for key in [Key::Tab, Key::Forward, Key::Left, Key::Up] {
            assert_eq!(p.stroke(key), None);
        }
        assert_eq!((p.text().as_str(), p.caret()), ("ad", 0));
    }

    #[test]
    fn enter_submits_and_escape_steps_back_without_touching_the_text() {
        let mut p = Prompt::new("host:1");
        assert_eq!(p.stroke(Key::Enter), Some(PromptAction::Submit));
        assert_eq!(p.stroke(Key::Escape), Some(PromptAction::Back));
        assert_eq!((p.text().as_str(), p.caret()), ("host:1", 6));
    }

    #[test]
    fn white_space_and_control_characters_are_not_typed() {
        let mut p = Prompt::new("");
        for c in [' ', '\t', '\n', '\r', '\u{1b}', '\0', '\u{7f}', '\u{a0}'] {
            assert!(!p.insert(c), "{c:?}");
        }
        assert!(p.is_empty());
        // A pasted address with stray white space comes out clean.
        typed(&mut p, " 192.168.0.12:7777\r\n");
        assert_eq!(p.text(), "192.168.0.12:7777");
        // Anything printable is taken, also outside ASCII (a name may be
        // typed in any script; whether it resolves is another matter).
        assert!(p.insert('é'));
    }

    #[test]
    fn the_field_is_bounded() {
        let mut p = Prompt::new(&"a".repeat(MAX_ADDRESS_CHARS + 50));
        assert_eq!(p.len(), MAX_ADDRESS_CHARS);
        assert!(!p.insert('b'));
        p.stroke(Key::Backspace);
        assert!(p.insert('b'));
        assert_eq!(p.len(), MAX_ADDRESS_CHARS);
    }

    #[test]
    fn an_error_stays_until_the_text_is_edited() {
        let mut p = Prompt::new("nowhere:1");
        p.set_error("no answer");
        assert_eq!(p.error(), Some("no answer"));
        assert_eq!(p.text(), "nowhere:1", "the typed address is kept");
        // Looking around does not dismiss it...
        for key in [Key::Home, Key::CaretRight, Key::End, Key::Tab] {
            p.stroke(key);
            assert!(p.error().is_some(), "{key:?}");
        }
        // ...a stroke that changes nothing does not either...
        p.stroke(Key::Delete);
        assert!(p.error().is_some());
        // ...editing does.
        p.stroke(Key::Backspace);
        assert_eq!(p.error(), None);
        p.set_error("again");
        p.insert('2');
        assert_eq!(p.error(), None);
        p.set_error("again");
        p.clear_error();
        assert_eq!(p.error(), None);
        p.set_error("again");
        p.set_text("other:2");
        assert_eq!((p.error(), p.caret()), (None, 7));
    }

    #[test]
    fn a_long_text_scrolls_with_the_caret() {
        let mut p = Prompt::new("0123456789");
        // Fits: shown whole.
        assert_eq!(p.window(20), ("0123456789".into(), 10));
        // Too long: the end is shown, the caret behind the last character.
        assert_eq!(p.window(6), ("56789".into(), 5));
        assert_eq!(p.window(10), ("123456789".into(), 9));
        // The caret moves left inside the window first...
        p.stroke(Key::CaretLeft);
        assert_eq!(p.window(6), ("456789".into(), 5));
        // ...and at the start the beginning of the text is shown.
        p.stroke(Key::Home);
        assert_eq!(p.window(6), ("012345".into(), 0));
        // The caret is always inside the window, whatever the width.
        for columns in 0..14 {
            for caret in 0..=10 {
                p.stroke(Key::Home);
                for _ in 0..caret {
                    p.stroke(Key::CaretRight);
                }
                let (shown, at) = p.window(columns);
                assert!(at <= columns.max(1), "{columns} {caret}");
                assert!(shown.chars().count() <= columns.max(1));
            }
        }
    }
}
