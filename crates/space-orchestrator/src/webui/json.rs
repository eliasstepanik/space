//! The snapshot as JSON (`GET /api/state`), for scripts and agents.
//!
//! Written by hand: the crate has no JSON dependency and the document is only
//! ever *written*, never parsed. Every string goes through [`write_string`],
//! which escapes quotes, backslashes and control characters as JSON requires,
//! and additionally `<`, `>` and `&`, so the document stays inert even if it
//! were ever pasted into a page.
//!
//! The document is read-only by construction: it is rendered from a
//! [`Snapshot`], the same copy the page is rendered from. Ticks are the
//! orchestrator's clock ([`crate::config::TICK_HZ`] per second); wherever a
//! tick is given, an `…_s` field next to it gives the same thing in seconds.
//! A shard with no load report has `"load_report": null` — never zeros.

use std::fmt::Write as _;

use space_world::NodeRef;

use super::ServerInfo;
use crate::config::{
    EPOCH_ACK_TIMEOUT_TICKS, EPOCH_LEAD_TICKS, LOAD_REPORT_INTERVAL_TICKS,
    MIN_EPOCH_INTERVAL_TICKS, REPORT_EXPIRY_TICKS, TICK_HZ,
};
use crate::snapshot::{node_path, octant_signs, Snapshot};

/// Append `text` as a JSON string literal.
pub fn write_string(out: &mut String, text: &str) {
    out.push('"');
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if u32::from(c) < 0x20
                || matches!(c, '\u{7f}' | '<' | '>' | '&' | '\u{2028}' | '\u{2029}') =>
            {
                let _ = write!(out, "\\u{:04x}", u32::from(c));
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Append a number; JSON has no NaN or infinity, so those become `null`.
fn write_float(out: &mut String, value: f64) {
    if value.is_finite() {
        let _ = write!(out, "{value}");
    } else {
        out.push_str("null");
    }
}

/// A JSON object being written: keeps track of the commas.
struct Object<'a> {
    out: &'a mut String,
    empty: bool,
}

impl<'a> Object<'a> {
    fn open(out: &'a mut String) -> Self {
        out.push('{');
        Self { out, empty: true }
    }

    fn close(self) {
        self.out.push('}');
    }

    /// Write `"name":` (with the comma it needs) and hand back the buffer for
    /// the value.
    fn key(&mut self, name: &str) -> &mut String {
        if !self.empty {
            self.out.push(',');
        }
        self.empty = false;
        write_string(self.out, name);
        self.out.push(':');
        self.out
    }

    fn string(&mut self, name: &str, value: &str) {
        write_string(self.key(name), value);
    }

    fn display(&mut self, name: &str, value: impl core::fmt::Display) {
        let text = value.to_string();
        self.string(name, &text);
    }

    fn uint(&mut self, name: &str, value: u64) {
        let _ = write!(self.key(name), "{value}");
    }

    fn float(&mut self, name: &str, value: f64) {
        write_float(self.key(name), value);
    }

    fn boolean(&mut self, name: &str, value: bool) {
        self.key(name)
            .push_str(if value { "true" } else { "false" });
    }

    fn null(&mut self, name: &str) {
        self.key(name).push_str("null");
    }

    fn opt_uint(&mut self, name: &str, value: Option<u64>) {
        match value {
            Some(value) => self.uint(name, value),
            None => self.null(name),
        }
    }

    fn opt_string(&mut self, name: &str, value: Option<&str>) {
        match value {
            Some(value) => self.string(name, value),
            None => self.null(name),
        }
    }

    fn opt_display(&mut self, name: &str, value: Option<impl core::fmt::Display>) {
        match value {
            Some(value) => self.display(name, value),
            None => self.null(name),
        }
    }

    fn object(&mut self, name: &str, fill: impl FnOnce(&mut Object<'_>)) {
        let mut inner = Object::open(self.key(name));
        fill(&mut inner);
        inner.close();
    }

    /// `"name": {…}` when `value` is there, `"name": null` when it is not.
    fn opt_object<T>(
        &mut self,
        name: &str,
        value: Option<T>,
        fill: impl FnOnce(&mut Object<'_>, T),
    ) {
        match value {
            Some(value) => self.object(name, |inner| fill(inner, value)),
            None => self.null(name),
        }
    }

    /// `"name": [ … ]`, each item written by `item` into the buffer.
    fn array<T>(
        &mut self,
        name: &str,
        items: impl IntoIterator<Item = T>,
        mut item: impl FnMut(&mut String, T),
    ) {
        let out = self.key(name);
        out.push('[');
        for (n, value) in items.into_iter().enumerate() {
            if n > 0 {
                out.push(',');
            }
            item(out, value);
        }
        out.push(']');
    }

    /// `"name": [ {…}, … ]`.
    fn objects<T>(
        &mut self,
        name: &str,
        items: impl IntoIterator<Item = T>,
        mut fill: impl FnMut(&mut Object<'_>, T),
    ) {
        self.array(name, items, |out, value| {
            let mut inner = Object::open(out);
            fill(&mut inner, value);
            inner.close();
        });
    }

    /// A tick and, next to it, how long ago (or ahead) that is in seconds.
    fn tick(&mut self, name: &str, tick: u64, now: u64) {
        self.uint(name, tick);
        let seconds = (now as f64 - tick as f64) / TICK_HZ as f64;
        self.float(&format!("{name}_age_s"), seconds);
    }
}

/// Ticks as seconds.
fn seconds(ticks: u64) -> f64 {
    ticks as f64 / TICK_HZ as f64
}

fn node(object: &mut Object<'_>, name: &str, node: NodeRef) {
    object.object(name, |o| {
        o.string("path", &node_path(node));
        o.uint("depth", u64::from(node.depth));
        o.array("index", node.index, |out, axis| {
            let _ = write!(out, "{axis}");
        });
        o.float("edge_m", NodeRef::size_m(node.depth));
    });
}

/// The whole state as one JSON document (see the module docs).
///
/// `form_token` is the per-process token a state-changing form post has to
/// carry; it is included so a script can post an action after reading the
/// state (the document is readable only by whoever may post anyway).
#[must_use]
pub fn state_json(snapshot: &Snapshot, info: &ServerInfo, form_token: &str) -> String {
    let now = snapshot.tick;
    let mut out = String::with_capacity(4096);
    let mut root = Object::open(&mut out);

    root.object("orchestrator", |o| {
        o.string("version", info.version);
        o.opt_string("build", info.build);
        o.string("profile", info.profile);
        o.string("platform", &info.platform);
        o.uint("protocol_version", u64::from(info.protocol_version));
        o.string("cluster", &snapshot.cluster);
        o.display("bind", snapshot.bind);
        o.display("control_addr", snapshot.control_addr);
        o.string("launcher", super::launcher_label(snapshot.launcher));
        o.float("uptime_s", info.uptime.as_secs_f64());
        o.uint("now_unix_s", info.now_unix);
        o.uint("tick", now);
        o.uint("tick_hz", TICK_HZ);
        node(o, "root", snapshot.root);
    });

    root.object("webui", |o| {
        o.display("bind", info.bind);
        o.string("auth", if info.token_required { "token" } else { "none" });
        o.string("form_token", form_token);
    });

    // The one address players type (null: the orchestrator serves none), the
    // shard a new player starts on, and what a hello is answered with now.
    root.opt_object("player_port", snapshot.player_port, |o, port| {
        o.display("bind", port.addr);
        o.string(
            "answer",
            if port.sends_to.is_some() {
                "redirect"
            } else {
                "starting"
            },
        );
        o.opt_uint(
            "sends_to_shard",
            port.sends_to.map(|shard| u64::from(shard.0)),
        );
        o.uint("redirected", port.redirected);
        o.uint("told_starting", port.turned_away);
        o.uint("refused_busy", port.refused);
    });

    root.object("map", |o| {
        o.uint("epoch", snapshot.epoch);
        o.uint("max_regions", snapshot.max_regions as u64);
        o.opt_uint(
            "spawn_owner",
            snapshot.spawn_owner.map(|shard| u64::from(shard.0)),
        );
        match snapshot.last_commit_tick {
            Some(tick) => o.tick("last_commit_tick", tick, now),
            None => o.null("last_commit_tick"),
        }
        o.objects("regions", &snapshot.regions, |o, region| {
            node(o, "node", region.node);
            o.uint("shard", u64::from(region.shard.0));
            o.opt_display("endpoint", region.endpoint);
            o.opt_display("client_endpoint", region.client_endpoint);
            o.uint("nesting", u64::from(region.nesting));
            o.opt_string("parent", region.parent.map(node_path).as_deref());
            match region.octant {
                Some(octant) => {
                    o.uint("octant", u64::from(octant));
                    o.string("octant_signs", octant_signs(octant));
                }
                None => o.null("octant"),
            }
            o.array("free_octants", &region.free_octants, |out, octant| {
                let _ = write!(out, "{octant}");
            });
            o.opt_string("split_refused_because", region.split_refusal.as_deref());
            o.opt_string("merge_refused_because", region.merge_refusal.as_deref());
        });
    });

    // The shards the orchestrator runs by itself, and what is still missing.
    root.object("fleet", |o| {
        o.uint("min_shards", snapshot.min_shards as u64);
        o.uint("max_shards", snapshot.max_shards as u64);
        o.uint("starting", snapshot.starting as u64);
        o.opt_string("wanted", snapshot.wanted.map(node_path).as_deref());
        o.boolean("hello_auth_required", snapshot.hello_auth_required);
    });

    root.objects("shards", &snapshot.shards, |o, shard| {
        o.uint("id", u64::from(shard.shard.0));
        o.string("role", shard.role.label());
        o.boolean("connected", shard.connected);
        o.boolean("registered", shard.registered);
        o.boolean("launched", shard.launched);
        o.opt_display("advertised", shard.advertised);
        o.opt_display("client_endpoint", shard.client_endpoint);
        o.array("regions", &shard.regions, |out, region| {
            write_string(out, &node_path(*region));
        });
        o.opt_object("last_heard", shard.last_heard, |o, heard| {
            o.string("what", heard.what.label());
            o.tick("tick", heard.tick, now);
        });
        o.opt_object("load_report", shard.report, |o, report| {
            o.uint("players", u64::from(report.players));
            o.float("tick_p95_ms", f64::from(report.tick_p95_ms));
            o.float("reported_load", f64::from(report.reported_load));
            o.float("load_metric", report.load_metric);
            o.tick("tick", report.tick, now);
        });
        o.opt_object("autoscaler", shard.scale, |o, scale| {
            o.opt_uint("hot_since_tick", scale.hot_since);
            o.opt_uint("cold_since_tick", scale.cold_since);
            o.uint("immune_until_tick", scale.immune_until);
            o.boolean("cooling_down", scale.immune_until > now);
        });
        // What the shard last confirmed of the scene (null: nothing since it
        // registered), and how that compares with the cluster's.
        o.opt_object("scene", shard.scene, |o, confirmed| {
            o.opt_string("name", confirmed.scene.known().map(|scene| scene.name()));
            o.uint("id", u64::from(confirmed.scene.0));
            o.uint("revision", confirmed.revision);
        });
        o.string("scene_sync", snapshot.scene.sync_of(shard.scene).label());
    });

    root.object("scene", |o| {
        let scene = &snapshot.scene;
        o.string("name", scene.scene.name());
        o.uint("id", u64::from(scene.scene.id().0));
        o.uint("revision", scene.revision);
        match scene.changed_tick {
            Some(tick) => o.tick("changed_tick", tick, now),
            None => o.null("changed_tick"),
        }
        o.opt_display("file", scene.path.as_ref().map(|path| path.display()));
        let connected = || snapshot.shards.iter().filter(|shard| shard.connected);
        let ids = |wanted: bool| {
            connected()
                .filter(move |shard| {
                    (scene.sync_of(shard.scene) == crate::snapshot::SceneSync::InSync) == wanted
                })
                .map(|shard| shard.shard.0)
        };
        o.array("confirmed_by", ids(true), |out, id| {
            let _ = write!(out, "{id}");
        });
        o.array("not_confirmed_by", ids(false), |out, id| {
            let _ = write!(out, "{id}");
        });
        // The same in one word and one sentence: is a switch still on its way
        // to a connected shard, and what the page says about it.
        let progress = snapshot.scene_progress();
        o.boolean("pending", !progress.settled());
        o.string("status", &progress.sentence());
        o.objects("catalogue", space_protocol::Scene::ALL, |o, choice| {
            o.string("name", choice.name());
            o.uint("id", u64::from(choice.id().0));
            o.boolean("current", choice == scene.scene);
        });
    });

    root.opt_object(
        "epoch_in_flight",
        snapshot.in_flight.as_ref(),
        |o, flight| {
            o.uint("epoch", flight.epoch);
            o.string("what", &flight.what);
            o.uint("activation_tick", flight.activation_tick);
            o.uint("deadline_tick", flight.deadline_tick);
            o.float(
                "deadline_in_s",
                seconds(flight.deadline_tick.saturating_sub(now)),
            );
            o.objects("recipients", &flight.recipients, |o, (shard, acked)| {
                o.uint("shard", u64::from(shard.0));
                o.boolean("acked", *acked);
            });
            o.objects("changes", &flight.changes, |o, (changed, owner)| {
                node(o, "node", *changed);
                o.uint("new_owner", u64::from(owner.0));
            });
        },
    );

    root.object("autoscaler", |o| {
        let scaler = &snapshot.autoscaler;
        o.boolean("paused", scaler.paused);
        o.uint("busy_until_tick", scaler.busy_until);
        o.boolean("busy", scaler.busy_until > now);
        o.opt_object(
            "last_decision",
            scaler.last_decision.as_ref(),
            |o, decision| {
                o.string("action", &decision.action);
                o.tick("tick", decision.tick, now);
                o.boolean("carried_out", decision.outcome.is_ok());
                let (Ok(detail) | Err(detail)) = &decision.outcome;
                o.string("detail", detail);
            },
        );
        o.object("config", |o| {
            o.float("split_threshold", crate::autoscale::SPLIT_THRESHOLD);
            o.float("merge_threshold", crate::autoscale::MERGE_THRESHOLD);
            o.uint("split_dwell_ticks", crate::autoscale::SPLIT_DWELL_TICKS);
            o.uint("merge_dwell_ticks", crate::autoscale::MERGE_DWELL_TICKS);
            o.uint("cooldown_ticks", crate::autoscale::SCALE_COOLDOWN_TICKS);
            o.float(
                "max_players_per_shard",
                crate::autoscale::MAX_PLAYERS_PER_SHARD,
            );
            o.float("tick_budget_ms", crate::autoscale::TICK_BUDGET_MS);
        });
    });

    root.object("allocator", |o| {
        o.uint("next_shard_id", snapshot.next_shard_id);
        o.uint("first_shard_id", snapshot.first_shard_id);
        o.display("state_path", snapshot.state_path.display());
        o.opt_uint(
            "unclaimed_root_owner",
            snapshot
                .unclaimed_root_owner
                .map(|shard| u64::from(shard.0)),
        );
        o.array(
            "host_port_range",
            [snapshot.host_port_range.0, snapshot.host_port_range.1],
            |out, port| {
                let _ = write!(out, "{port}");
            },
        );
    });

    root.object("timing", |o| {
        o.uint("epoch_lead_ticks", EPOCH_LEAD_TICKS);
        o.uint("epoch_ack_timeout_ticks", EPOCH_ACK_TIMEOUT_TICKS);
        o.uint("min_epoch_interval_ticks", MIN_EPOCH_INTERVAL_TICKS);
        o.uint("load_report_interval_ticks", LOAD_REPORT_INTERVAL_TICKS);
        o.uint("report_expiry_ticks", REPORT_EXPIRY_TICKS);
    });

    root.objects("events", &snapshot.events, |o, event| {
        o.uint("seq", event.seq);
        o.tick("tick", event.tick, now);
        o.uint("first_tick", event.first_tick);
        o.uint("repeats", u64::from(event.repeats));
        o.string("kind", event.kind.label());
        o.string("level", event.severity.label());
        o.string("text", &event.text);
    });

    root.close();
    out.push('\n');
    out
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A minimal JSON well-formedness check (RFC 8259 grammar, no values
    /// built): `Ok(())` when `text` is exactly one JSON value.
    pub(crate) fn check_json(text: &str) -> Result<(), String> {
        let bytes = text.as_bytes();
        let mut at = 0;
        value(bytes, &mut at, 0)?;
        skip_space(bytes, &mut at);
        if at == bytes.len() {
            Ok(())
        } else {
            Err(format!("trailing data at byte {at}"))
        }
    }

    fn skip_space(bytes: &[u8], at: &mut usize) {
        while matches!(bytes.get(*at), Some(b' ' | b'\n' | b'\r' | b'\t')) {
            *at += 1;
        }
    }

    fn expect(bytes: &[u8], at: &mut usize, literal: &[u8]) -> Result<(), String> {
        if bytes.get(*at..*at + literal.len()) == Some(literal) {
            *at += literal.len();
            Ok(())
        } else {
            Err(format!(
                "expected {:?} at byte {at}",
                String::from_utf8_lossy(literal)
            ))
        }
    }

    fn value(bytes: &[u8], at: &mut usize, depth: usize) -> Result<(), String> {
        if depth > 32 {
            return Err("nested too deeply".into());
        }
        skip_space(bytes, at);
        match bytes.get(*at) {
            Some(b'{') => {
                *at += 1;
                skip_space(bytes, at);
                if bytes.get(*at) == Some(&b'}') {
                    *at += 1;
                    return Ok(());
                }
                loop {
                    skip_space(bytes, at);
                    string(bytes, at)?;
                    skip_space(bytes, at);
                    expect(bytes, at, b":")?;
                    value(bytes, at, depth + 1)?;
                    skip_space(bytes, at);
                    match bytes.get(*at) {
                        Some(b',') => *at += 1,
                        Some(b'}') => {
                            *at += 1;
                            return Ok(());
                        }
                        _ => return Err(format!("expected , or }} at byte {at}")),
                    }
                }
            }
            Some(b'[') => {
                *at += 1;
                skip_space(bytes, at);
                if bytes.get(*at) == Some(&b']') {
                    *at += 1;
                    return Ok(());
                }
                loop {
                    value(bytes, at, depth + 1)?;
                    skip_space(bytes, at);
                    match bytes.get(*at) {
                        Some(b',') => *at += 1,
                        Some(b']') => {
                            *at += 1;
                            return Ok(());
                        }
                        _ => return Err(format!("expected , or ] at byte {at}")),
                    }
                }
            }
            Some(b'"') => string(bytes, at),
            Some(b't') => expect(bytes, at, b"true"),
            Some(b'f') => expect(bytes, at, b"false"),
            Some(b'n') => expect(bytes, at, b"null"),
            Some(b'-' | b'0'..=b'9') => number(bytes, at),
            other => Err(format!("unexpected {other:?} at byte {at}")),
        }
    }

    fn string(bytes: &[u8], at: &mut usize) -> Result<(), String> {
        expect(bytes, at, b"\"")?;
        loop {
            match bytes.get(*at) {
                None => return Err("unterminated string".into()),
                Some(b'"') => {
                    *at += 1;
                    return Ok(());
                }
                Some(b'\\') => match bytes.get(*at + 1) {
                    Some(b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't') => *at += 2,
                    Some(b'u') => {
                        let hex = bytes.get(*at + 2..*at + 6).ok_or("short \\u escape")?;
                        if !hex.iter().all(u8::is_ascii_hexdigit) {
                            return Err(format!("bad \\u escape at byte {at}"));
                        }
                        *at += 6;
                    }
                    _ => return Err(format!("bad escape at byte {at}")),
                },
                Some(byte) if *byte < 0x20 => {
                    return Err(format!("raw control character at byte {at}"));
                }
                Some(_) => *at += 1,
            }
        }
    }

    fn digits(bytes: &[u8], at: &mut usize) -> usize {
        let start = *at;
        while matches!(bytes.get(*at), Some(b'0'..=b'9')) {
            *at += 1;
        }
        *at - start
    }

    fn number(bytes: &[u8], at: &mut usize) -> Result<(), String> {
        if bytes.get(*at) == Some(&b'-') {
            *at += 1;
        }
        let leading_zero = bytes.get(*at) == Some(&b'0');
        let whole = digits(bytes, at);
        if whole == 0 || (leading_zero && whole > 1) {
            return Err(format!("bad number at byte {at}"));
        }
        if bytes.get(*at) == Some(&b'.') {
            *at += 1;
            if digits(bytes, at) == 0 {
                return Err(format!("bad fraction at byte {at}"));
            }
        }
        if matches!(bytes.get(*at), Some(b'e' | b'E')) {
            *at += 1;
            if matches!(bytes.get(*at), Some(b'+' | b'-')) {
                *at += 1;
            }
            if digits(bytes, at) == 0 {
                return Err(format!("bad exponent at byte {at}"));
            }
        }
        Ok(())
    }

    #[test]
    fn the_checker_itself_tells_json_from_not_json() {
        for good in [
            "{}",
            "[]",
            "{\"a\":[1,-2.5,3e4,true,false,null,\"x\\n\\u00e9\"],\"b\":{\"c\":{}}}",
            " [ 0 ] ",
        ] {
            check_json(good).unwrap_or_else(|why| panic!("{good}: {why}"));
        }
        for bad in [
            "",
            "{",
            "{a:1}",
            r#"{"a":1,}"#,
            "[1 2]",
            "01",
            "1.",
            "NaN",
            "\"a\nb\"",
            r#""\x""#,
            "{} {}",
        ] {
            assert!(check_json(bad).is_err(), "{bad:?} must be refused");
        }
    }

    #[test]
    fn strings_are_escaped() {
        let mut out = String::new();
        write_string(&mut out, "a\"b\\c\nd\te\u{1}<script>&\u{2028}é");
        assert_eq!(
            out,
            "\"a\\\"b\\\\c\\nd\\te\\u0001\\u003cscript\\u003e\\u0026\\u2028é\""
        );
        check_json(&out).expect("a valid string literal");
    }

    #[test]
    fn a_number_that_json_cannot_express_becomes_null() {
        for (value, want) in [
            (0.8, "0.8"),
            (17_179_869_184.0, "17179869184"),
            (f64::NAN, "null"),
            (f64::INFINITY, "null"),
            (f64::NEG_INFINITY, "null"),
        ] {
            let mut out = String::new();
            write_float(&mut out, value);
            assert_eq!(out, want);
        }
    }

    #[test]
    fn an_object_places_its_commas() {
        let mut out = String::new();
        let mut object = Object::open(&mut out);
        object.uint("a", 1);
        object.null("b");
        object.array("c", [1u8, 2], |out, n| {
            let _ = write!(out, "{n}");
        });
        object.objects("d", [true, false], |o, flag| o.boolean("f", flag));
        object.object("e", |_| {});
        object.opt_object("g", None::<u8>, |_, _| {});
        object.close();
        assert_eq!(
            out,
            r#"{"a":1,"b":null,"c":[1,2],"d":[{"f":true},{"f":false}],"e":{},"g":null}"#
        );
        check_json(&out).expect("well-formed");
    }
}
