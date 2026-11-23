//! The dashboard: one HTML page rendered from a [`Snapshot`].
//!
//! No script, no external resource, no template file: the page is a string
//! built here, with one inline stylesheet. It refreshes itself with a
//! `<meta http-equiv="refresh">` (which a `?refresh=0` link switches off) and
//! acts through plain `<form method="post">` buttons.
//!
//! # No control holds a choice
//!
//! A reload puts every form control back to what the server rendered, and
//! this page reloads itself. So nothing on it is *chosen first and sent
//! later*: no `<select>`, no text field, no checkbox. Every action is one
//! `<button name="…" value="…">` inside a form of hidden fields, and the click
//! carries the whole request. (The scene was once picked in a `<select>`: the
//! auto-refresh put it back to the running scene within three seconds, and
//! the button next to it then posted exactly that — "accepted", and nothing
//! changed.)
//!
//! # Escaping
//!
//! Markup can only enter the page through the private writer's `Html::raw`,
//! which takes a `&'static str` — a literal in this file. Everything that is
//! data (a cluster name, an address, an event text that quotes what a peer
//! sent) goes through `Html::text`, which escapes `& < > " '`
//! ([`push_escaped`]). There is no third way to append to the page, so "is
//! this value escaped?" is answered by the type of the call, not by reviewing
//! each call site.
//!
//! Every link and form target is **relative** (`?refresh=0`, `action/split`,
//! `api/state`), so the page also works when a proxy serves it under a path
//! prefix.

use std::fmt::Write as _;

use space_protocol::shard::ShardId;
use space_world::NodeRef;

use super::ServerInfo;
use crate::autoscale::{
    MAX_PLAYERS_PER_SHARD, MERGE_DWELL_TICKS, MERGE_THRESHOLD, SCALE_COOLDOWN_TICKS,
    SPLIT_DWELL_TICKS, SPLIT_THRESHOLD, TICK_BUDGET_MS,
};
use crate::config::{
    EPOCH_ACK_TIMEOUT_TICKS, EPOCH_LEAD_TICKS, MIN_EPOCH_INTERVAL_TICKS, REPORT_EXPIRY_TICKS,
    TICK_HZ,
};
use crate::events::{Event, EventKind, Severity};
use crate::snapshot::{
    node_path, octant_signs, RegionView, SceneProgress, SceneSync, ShardRole, ShardView, Snapshot,
};

/// Append `text` to `out` with the five HTML-significant characters escaped.
/// Safe for element content and for double- or single-quoted attribute values.
pub fn push_escaped(out: &mut String, text: &str) {
    for c in text.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            c => out.push(c),
        }
    }
}

/// `text` with the five HTML-significant characters escaped.
#[must_use]
pub fn escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    push_escaped(&mut out, text);
    out
}

/// A page being written (see the module docs for why it has exactly two ways
/// to append).
struct Html {
    out: String,
}

/// Escapes whatever is formatted into it.
struct Escaping<'a>(&'a mut String);

impl core::fmt::Write for Escaping<'_> {
    fn write_str(&mut self, text: &str) -> core::fmt::Result {
        push_escaped(self.0, text);
        Ok(())
    }
}

impl Html {
    fn new() -> Self {
        Self {
            out: String::with_capacity(16 * 1024),
        }
    }

    /// Append markup. Only a literal can be passed.
    fn raw(&mut self, markup: &'static str) {
        self.out.push_str(markup);
    }

    /// Append data, escaped.
    fn text(&mut self, value: impl core::fmt::Display) {
        let _ = write!(Escaping(&mut self.out), "{value}");
    }

    /// `<tag>` + escaped `value` + `</tag>` for the few tags used this way.
    fn wrapped(&mut self, open: &'static str, value: impl core::fmt::Display, close: &'static str) {
        self.raw(open);
        self.text(value);
        self.raw(close);
    }

    /// A status pill: `<span class="pill CLASS">label</span>`.
    fn pill(&mut self, class: &'static str, label: impl core::fmt::Display) {
        self.raw("<span class=\"pill ");
        self.raw(class);
        self.raw("\">");
        self.text(label);
        self.raw("</span>");
    }

    /// One `<dt>key</dt><dd>value</dd>` row of a definition list.
    fn fact(&mut self, key: &'static str, value: impl core::fmt::Display) {
        self.raw("<dt>");
        self.raw(key);
        self.raw("</dt><dd>");
        self.text(value);
        self.raw("</dd>\n");
    }

    fn finish(self) -> String {
        self.out
    }
}

/// What the dashboard is rendered from.
#[derive(Debug, Clone, Copy)]
pub struct PageInput<'a> {
    /// The orchestrator's state.
    pub snapshot: &'a Snapshot,
    /// What the web server knows about itself.
    pub info: &'a ServerInfo,
    /// The per-process token every form carries.
    pub form_token: &'a str,
    /// The nonce of the page's stylesheet (it is in the response's policy).
    pub style_nonce: &'a str,
    /// Seconds between automatic reloads; 0 switches them off.
    pub refresh: u32,
    /// The operator request whose outcome is shown on top, if any.
    pub notice: Option<&'a Event>,
    /// The scene request whose outcome is shown on top, if any (it takes the
    /// place of `notice`).
    pub scene_notice: Option<SceneNotice>,
}

/// What the redirect of a scene request asks the dashboard to say on top.
///
/// Only two numbers travel in the URL. Everything the banner *says* — which
/// scene, which revision, who has confirmed and who has not — is read from
/// the snapshot when the page is rendered, so it is true at that moment and
/// stays true through every reload, and a link cannot make it say otherwise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SceneNotice {
    /// The revision the request was about. A banner is only shown for a
    /// revision the cluster has reached.
    pub revision: u64,
    /// The scene asked for was already running: nothing was changed.
    pub unchanged: bool,
}

/// Seconds between reloads while a connected shard has not confirmed the
/// current scene, whatever interval was chosen (also "off"): a page that says
/// "behind" and then never looks again reads as "it did not work".
pub const SCENE_PENDING_REFRESH_SECS: u32 = 1;

/// The reload interval the page really uses: the chosen one, or
/// [`SCENE_PENDING_REFRESH_SECS`] while a scene change is still on its way to
/// a connected shard.
fn effective_refresh(chosen: u32, progress: &SceneProgress) -> u32 {
    if progress.settled() {
        chosen
    } else {
        SCENE_PENDING_REFRESH_SECS
    }
}

/// Ticks as seconds.
fn seconds(ticks: u64) -> f64 {
    ticks as f64 / TICK_HZ as f64
}

/// A length of time for people: `0.4 s`, `42 s`, `3 min 05 s`, `2 h 04 min`,
/// `3 d 02 h`.
#[must_use]
pub fn span(seconds: f64) -> String {
    if !seconds.is_finite() || seconds < 0.0 {
        return "0 s".into();
    }
    if seconds < 10.0 {
        return format!("{seconds:.1} s");
    }
    let whole = seconds as u64;
    match whole {
        0..=59 => format!("{whole} s"),
        60..=3_599 => format!("{} min {:02} s", whole / 60, whole % 60),
        3_600..=86_399 => format!("{} h {:02} min", whole / 3_600, whole % 3_600 / 60),
        _ => format!("{} d {:02} h", whole / 86_400, whole % 86_400 / 3_600),
    }
}

/// How long ago `tick` was, seen from `now`.
fn ago(now: u64, tick: u64) -> String {
    format!("{} ago", span(seconds(now.saturating_sub(tick))))
}

/// A Unix time as `YYYY-MM-DD HH:MM:SS UTC`.
#[must_use]
pub fn clock(unix_seconds: u64) -> String {
    let days = unix_seconds / 86_400;
    let rest = unix_seconds % 86_400;
    // Days since 1970-01-01 to a civil date (Howard Hinnant's algorithm,
    // valid for every date this can be given).
    let z = days + 719_468;
    let era = z / 146_097;
    let day_of_era = z % 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_index = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_index + 2) / 5 + 1;
    let month = if month_index < 10 {
        month_index + 3
    } else {
        month_index - 9
    };
    let year = year_of_era + era * 400 + u64::from(month <= 2);
    format!(
        "{year:04}-{month:02}-{day:02} {:02}:{:02}:{:02} UTC",
        rest / 3_600,
        rest % 3_600 / 60,
        rest % 60
    )
}

/// The wall-clock time of `tick`, reckoned back from now on the tick clock.
fn clock_of(input: &PageInput<'_>, tick: u64) -> String {
    let age = seconds(input.snapshot.tick.saturating_sub(tick)) as u64;
    clock(input.info.now_unix.saturating_sub(age))
}

/// An edge length for people: metres, kilometres, astronomical units or light
/// years, whichever reads best.
#[must_use]
pub fn distance(metres: f64) -> String {
    const AU: f64 = 1.495_978_707e11;
    const LIGHT_YEAR: f64 = 9.460_730_472_580_8e15;
    if !metres.is_finite() {
        return "?".into();
    }
    if metres >= LIGHT_YEAR / 10.0 {
        format!("{:.3} ly", metres / LIGHT_YEAR)
    } else if metres >= AU / 100.0 {
        format!("{:.3} AU", metres / AU)
    } else if metres >= 10_000.0 {
        format!("{:.0} km", metres / 1_000.0)
    } else {
        format!("{metres:.0} m")
    }
}

/// The page's one stylesheet.
const STYLE: &str = r#"
:root{color-scheme:dark;--bg:#0e1116;--panel:#161a21;--line:#272d37;--text:#d6dbe1;--dim:#8a93a0;
--ok:#3fb950;--warn:#d6a01d;--bad:#f0605a;--info:#5aa7f0;--link:#7cbcff;--btn:#222831;--btnline:#3a424e}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--text);
font:14px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif}
code,.mono,button{font-family:ui-monospace,SFMono-Regular,Consolas,"Liberation Mono",Menlo,monospace;font-size:13px}
a{color:var(--link);text-decoration:none}
a:hover{text-decoration:underline}
header{display:flex;flex-wrap:wrap;align-items:center;gap:.45rem .9rem;padding:.6rem 1rem;
background:var(--panel);border-bottom:1px solid var(--line)}
h1{font-size:1rem;margin:0;font-weight:650;letter-spacing:.01em}
.cluster{color:var(--dim)}
.refresh{margin-left:auto;color:var(--dim);white-space:nowrap}
.refresh a,.refresh strong{margin-left:.5rem}
.refresh strong{color:var(--text)}
main{max-width:1320px;margin:0 auto;padding:1rem;display:grid;gap:1rem}
.cards{display:grid;gap:1rem;grid-template-columns:repeat(auto-fit,minmax(280px,1fr))}
.card{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:.75rem .9rem;min-width:0}
.card.flight{border-color:var(--info)}
h2{font-size:.74rem;letter-spacing:.07em;text-transform:uppercase;color:var(--dim);margin:0 0 .55rem;font-weight:650}
h2 .count{color:var(--text);margin-left:.35rem}
dl{display:grid;grid-template-columns:fit-content(48%) minmax(0,1fr);gap:.2rem .9rem;margin:0}
dt{color:var(--dim)}
dd{margin:0;text-align:right;overflow-wrap:break-word}
.flight dl{grid-template-columns:max-content minmax(0,1fr)}
.flight dd{text-align:left}
p{margin:.5rem 0 0}
.note{color:var(--dim);font-size:.86rem}
.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%}
th,td{text-align:left;padding:.36rem .65rem;border-bottom:1px solid var(--line);vertical-align:top;white-space:nowrap}
tr:last-child td{border-bottom:0}
th{color:var(--dim);font-weight:600;font-size:.74rem;text-transform:uppercase;letter-spacing:.05em}
td.wrap{white-space:normal;min-width:15rem;max-width:34rem;overflow-wrap:anywhere}
td.num{text-align:right;font-variant-numeric:tabular-nums}
.pill{display:inline-block;padding:0 .5rem;border:1px solid currentColor;border-radius:999px;
font-size:.76rem;font-weight:600;line-height:1.5;white-space:nowrap}
.ok{color:var(--ok)}.warn{color:var(--warn)}.bad{color:var(--bad)}.info{color:var(--info)}.dim{color:var(--dim)}
.in1{padding-left:1.6rem}.in2{padding-left:2.6rem}.in3{padding-left:3.6rem}.in4{padding-left:4.6rem}
.twig{color:var(--dim);display:inline-block;width:1.2rem}
.sub{color:var(--dim);font-size:.82rem;margin-top:.1rem}
.nested .sub{margin-left:1.2rem}
form{margin:0}
form.inline{display:inline-flex;flex-wrap:wrap;gap:.25rem;align-items:center}
td form.inline{flex-wrap:nowrap}
button{color:var(--text);background:var(--btn);border:1px solid var(--btnline);border-radius:5px;
padding:.12rem .5rem;line-height:1.5;cursor:pointer}
td button{padding:.1rem .42rem}
button:hover{border-color:var(--link)}
button:disabled{color:#5b636e;border-color:var(--line);background:transparent;cursor:not-allowed}
button.taken{text-decoration:line-through}
.why{color:var(--dim);font-size:.82rem;white-space:normal;min-width:11rem;max-width:21rem;margin-top:.2rem}
.notice{border:1px solid currentColor;border-radius:6px;padding:.55rem .9rem;display:flex;
justify-content:space-between;gap:1rem;align-items:baseline}
.notice span{color:var(--text)}
footer{max-width:1320px;margin:0 auto;padding:0 1rem 1.5rem;color:var(--dim);font-size:.84rem}
@media (max-width:640px){main{padding:.6rem;gap:.6rem}.cards{gap:.6rem}th,td{padding:.3rem .45rem}
.refresh{margin-left:0;width:100%}.refresh a:first-of-type,.refresh strong:first-of-type{margin-left:.3rem}}
"#;

/// The refresh intervals offered in the header, in seconds (0 = off).
const REFRESH_CHOICES: [u32; 4] = [0, 1, 3, 10];

/// Render the dashboard.
#[must_use]
pub fn dashboard(input: &PageInput<'_>) -> String {
    let mut h = Html::new();
    h.raw(
        "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n\
         <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n\
         <meta name=\"robots\" content=\"noindex, nofollow\">\n",
    );
    let progress = input.snapshot.scene_progress();
    let refresh = effective_refresh(input.refresh, &progress);
    if refresh > 0 {
        h.wrapped("<meta http-equiv=\"refresh\" content=\"", refresh, "\">\n");
    }
    h.wrapped(
        "<title>",
        &input.snapshot.cluster,
        " · space-orchestrator</title>\n<link rel=\"icon\" href=\"data:,\">\n",
    );
    h.wrapped("<style nonce=\"", input.style_nonce, "\">");
    h.raw(STYLE);
    h.raw("</style>\n</head>\n<body>\n");
    header(&mut h, input, &progress);
    h.raw("<main>\n");
    if !scene_notice(&mut h, input, &progress) {
        notice(&mut h, input);
    }
    cards(&mut h, input);
    in_flight(&mut h, input);
    shards(&mut h, input);
    regions(&mut h, input);
    events(&mut h, input);
    h.raw("</main>\n");
    footer(&mut h, input);
    h.raw("</body>\n</html>\n");
    h.finish()
}

fn header(h: &mut Html, input: &PageInput<'_>, progress: &SceneProgress) {
    let s = input.snapshot;
    h.raw("<header>\n<h1>space-orchestrator</h1>\n");
    h.wrapped(
        "<span class=\"cluster\">cluster <code>",
        &s.cluster,
        "</code></span>\n",
    );
    h.pill("info", format_args!("epoch {}", s.epoch));
    let owners: Vec<&ShardView> = s.shards.iter().filter(|v| !v.regions.is_empty()).collect();
    let up = owners.iter().filter(|v| v.connected).count();
    h.pill(
        if up == owners.len() { "ok" } else { "bad" },
        format_args!("{up} of {} region owners connected", owners.len()),
    );
    if let Some(flight) = &s.in_flight {
        h.pill("info", format_args!("epoch {} in flight", flight.epoch));
    }
    if s.autoscaler.paused {
        h.pill("warn", "autoscaler paused");
    } else {
        h.pill("ok", "autoscaler active");
    }
    h.raw("\n<nav class=\"refresh\">auto-refresh");
    if !progress.settled() {
        // The chosen interval is overridden for now; say so where it is chosen.
        h.wrapped(
            "<strong class=\"warn\">every ",
            SCENE_PENDING_REFRESH_SECS,
            " s while a shard is switching scene</strong> <span>then</span>",
        );
    }
    let mut choices: Vec<u32> = REFRESH_CHOICES.to_vec();
    if !choices.contains(&input.refresh) {
        choices.push(input.refresh);
    }
    for choice in choices {
        let current = choice == input.refresh;
        if current {
            h.raw("<strong>");
        } else {
            h.wrapped("<a href=\"?refresh=", choice, "\">");
        }
        if choice == 0 {
            h.raw("off");
        } else {
            h.text(format_args!("{choice} s"));
        }
        h.raw(if current { "</strong>" } else { "</a>" });
    }
    h.raw("</nav>\n</header>\n");
}

/// The banner for a scene request: how far the scene has got **right now**
/// (see [`SceneNotice`]). Returns whether it was shown.
fn scene_notice(h: &mut Html, input: &PageInput<'_>, progress: &SceneProgress) -> bool {
    let Some(asked) = input.scene_notice else {
        return false;
    };
    // Not for a revision the cluster has not reached: such a link is stale
    // (the orchestrator was restarted) or made up.
    if asked.revision == 0 || asked.revision > progress.revision {
        return false;
    }
    // "Already running" is a statement about one revision: once the scene has
    // been changed again it is not shown as if it were about the new one.
    if asked.unchanged && asked.revision != progress.revision {
        return false;
    }
    let everyone_has_it = progress.settled() && progress.connected() > 0;
    h.raw("<div class=\"notice ");
    h.raw(match (everyone_has_it, asked.unchanged) {
        (true, false) => "ok",
        // Nothing was changed: not the colour of a switch that went through.
        (true, true) => "info",
        (false, _) => "warn",
    });
    h.raw("\"><span>");
    if asked.unchanged {
        h.text(format_args!(
            "Already running {} (revision {}); nothing to do.",
            progress.scene, progress.revision
        ));
        if !everyone_has_it {
            h.text(format_args!(" {}.", progress.sentence()));
        }
    } else {
        h.text(format_args!("{}.", progress.sentence()));
    }
    if !progress.settled() {
        h.wrapped(
            " <span class=\"dim\">(this page reloads every ",
            SCENE_PENDING_REFRESH_SECS,
            " s until every connected shard has confirmed)</span>",
        );
    }
    h.wrapped(
        "</span><a href=\"?refresh=",
        input.refresh,
        "\">dismiss</a></div>\n",
    );
    true
}

fn notice(h: &mut Html, input: &PageInput<'_>) {
    let Some(event) = input.notice else {
        return;
    };
    h.raw("<div class=\"notice ");
    h.raw(match event.severity {
        Severity::Warn => "bad",
        Severity::Good | Severity::Info => "ok",
    });
    h.raw("\"><span>");
    h.text(&event.text);
    h.wrapped(
        " <span class=\"dim\">(",
        ago(input.snapshot.tick, event.tick),
        ")</span>",
    );
    h.wrapped(
        "</span><a href=\"?refresh=",
        input.refresh,
        "\">dismiss</a></div>\n",
    );
}

/// The hidden fields every form carries, after its opening tag.
fn form_open(h: &mut Html, input: &PageInput<'_>, action: &'static str) {
    h.raw("<form class=\"inline\" method=\"post\" action=\"");
    h.raw(action);
    h.wrapped(
        "\"><input type=\"hidden\" name=\"csrf\" value=\"",
        input.form_token,
        "\">",
    );
    h.wrapped(
        "<input type=\"hidden\" name=\"refresh\" value=\"",
        input.refresh,
        "\">",
    );
    h.wrapped(
        "<input type=\"hidden\" name=\"epoch\" value=\"",
        input.snapshot.epoch,
        "\">",
    );
}

fn cards(h: &mut Html, input: &PageInput<'_>) {
    let s = input.snapshot;
    let info = input.info;
    let now = s.tick;
    h.raw("<section class=\"cards\">\n");

    // ---- cluster
    h.raw("<div class=\"card\"><h2>Cluster</h2>\n<dl>\n");
    h.fact("name", &s.cluster);
    h.fact("control endpoint (QUIC)", s.bind);
    if s.control_addr != s.bind {
        h.fact("shards are told to dial", s.control_addr);
    }
    h.fact("launcher", super::launcher_label(s.launcher));
    h.fact(
        "shards",
        format_args!(
            "{} kept running by the orchestrator, at most {}",
            if s.min_shards == 0 {
                "none".to_owned()
            } else {
                s.min_shards.to_string()
            },
            s.max_shards
        ),
    );
    if s.starting > 0 {
        h.fact("starting", format_args!("{} shard(s)", s.starting));
    }
    if let Some(wanted) = s.wanted {
        h.fact(
            "waiting for a shard",
            format_args!("region {}", node_path(wanted)),
        );
    }
    h.fact(
        "shard registration",
        if s.hello_auth_required {
            "cluster secret required"
        } else {
            "no secret: loopback only"
        },
    );
    match info.build {
        Some(build) => h.fact(
            "version",
            format_args!(
                "{} ({build}, {}, {})",
                info.version, info.profile, info.platform
            ),
        ),
        None => h.fact(
            "version",
            format_args!("{} ({}, {})", info.version, info.profile, info.platform),
        ),
    }
    h.fact("wire protocol", info.protocol_version);
    h.fact("uptime", span(info.uptime.as_secs_f64()));
    h.fact("tick", format_args!("{now} at {TICK_HZ} Hz"));
    h.fact(
        "web UI",
        format_args!(
            "{} ({})",
            info.bind,
            if info.token_required {
                "token required"
            } else {
                "no token: loopback only"
            }
        ),
    );
    h.raw("</dl></div>\n");

    // ---- map
    h.raw("<div class=\"card\"><h2>Map</h2>\n<dl>\n");
    h.fact("committed epoch", s.epoch);
    h.fact(
        "regions",
        format_args!("{} of at most {}", s.regions.len(), s.max_regions),
    );
    h.fact(
        "root region",
        format_args!(
            "{} (edge {})",
            node_path(s.root),
            distance(NodeRef::size_m(s.root.depth))
        ),
    );
    match s.last_commit_tick {
        Some(tick) => h.fact(
            "last commit",
            format_args!("tick {tick}, {}", ago(now, tick)),
        ),
        None => h.fact("last commit", "none since start"),
    }
    match &s.in_flight {
        Some(flight) => h.fact("change in flight", format_args!("epoch {}", flight.epoch)),
        None => h.fact("change in flight", "none"),
    }
    h.fact(
        "epoch timing",
        format_args!(
            "ack within {}, lead {}, {} apart",
            span(seconds(EPOCH_ACK_TIMEOUT_TICKS)),
            span(seconds(EPOCH_LEAD_TICKS)),
            span(seconds(MIN_EPOCH_INTERVAL_TICKS))
        ),
    );
    h.raw("</dl>\n<p>");
    form_open(h, input, "action/push-map");
    h.raw(
        "<button title=\"Send the committed map, with every shard's current address, to every \
         shard in it. This is the push that follows each commit; it changes no ownership.\">\
         re-send map to shards</button></form></p></div>\n",
    );

    scene_card(h, input);

    // ---- autoscaler
    let scaler = &s.autoscaler;
    h.raw("<div class=\"card\"><h2>Autoscaler</h2>\n<dl>\n<dt>state</dt><dd>");
    if scaler.paused {
        h.pill("warn", "paused");
    } else {
        h.pill("ok", "active");
    }
    h.raw("</dd>\n");
    h.fact(
        "load",
        format_args!(
            "max(players / {MAX_PLAYERS_PER_SHARD:.0}, tick p95 / {TICK_BUDGET_MS:.1} ms)"
        ),
    );
    h.fact(
        "split",
        format_args!(
            "load ≥ {SPLIT_THRESHOLD:.2} for {}",
            span(seconds(SPLIT_DWELL_TICKS))
        ),
    );
    h.fact(
        "merge",
        format_args!(
            "load ≤ {MERGE_THRESHOLD:.2} for {}, 0 players",
            span(seconds(MERGE_DWELL_TICKS))
        ),
    );
    h.fact("cooldown per shard", span(seconds(SCALE_COOLDOWN_TICKS)));
    if scaler.busy_until > now {
        h.fact(
            "next action",
            format_args!("not before {}", span(seconds(scaler.busy_until - now))),
        );
    } else {
        h.fact("next action", "whenever a dwell completes");
    }
    let fresh = s.shards.iter().filter(|v| v.report.is_some()).count();
    h.fact(
        "fresh load reports",
        format_args!("{fresh} (kept {})", span(seconds(REPORT_EXPIRY_TICKS))),
    );
    h.raw("<dt>last decision</dt><dd>");
    match &scaler.last_decision {
        None => h.raw("<span class=\"dim\">none yet</span>"),
        Some(decision) => {
            h.text(format_args!(
                "{}, {}: ",
                decision.action,
                ago(now, decision.tick)
            ));
            // "Started" is all a decision can say: whether its epoch then
            // committed or aborted is in the event log.
            match &decision.outcome {
                Ok(started) => h.wrapped("<span class=\"info\">started, ", started, "</span>"),
                Err(why) => h.wrapped("<span class=\"warn\">not carried out: ", why, "</span>"),
            }
        }
    }
    h.raw("</dd>\n</dl>\n<p>");
    form_open(h, input, "action/autoscaler");
    if scaler.paused {
        h.raw(
            "<button name=\"state\" value=\"resume\" title=\"Let the autoscaler act again. \
             Every shard needs a fresh, full dwell first.\">resume autoscaler</button>",
        );
    } else {
        h.raw(
            "<button name=\"state\" value=\"pause\" title=\"Stop the autoscaler from splitting \
             or merging until it is resumed. Reports are still received; the buttons on this \
             page still work.\">pause autoscaler</button>",
        );
    }
    h.raw("</form></p>");
    if fresh == 0 {
        h.raw(
            "<p class=\"note\">No shard has a fresh load report, so the autoscaler has nothing \
             to act on and every merge is refused (a merge needs a report saying 0 players).</p>",
        );
    }
    h.raw("</div>\n");

    // ---- ids
    h.raw("<div class=\"card\"><h2>Shard ids</h2>\n<dl>\n");
    h.fact("next id", s.next_shard_id);
    h.fact("first id (configured)", s.first_shard_id);
    h.fact("counter file", s.state_path.display());
    match s.unclaimed_root_owner {
        Some(shard) => h.fact(
            "root owner",
            format_args!("{shard} reserved, no shard has claimed it yet"),
        ),
        None => h.fact("root owner", "claimed"),
    }
    h.fact(
        "host ports for launched shards",
        format_args!("{} to {}", s.host_port_range.0, s.host_port_range.1),
    );
    h.raw("</dl>\n<p class=\"note\">Ids only ever go up and are never reused (H9).</p></div>\n");

    h.raw("</section>\n");
}

/// The Scene card: what the cluster shows, who has confirmed it, and one
/// button per scene of the catalogue that switches every shard to it. The
/// button of the running scene is there too, disabled and marked, so the row
/// is the catalogue and never shifts under the pointer.
fn scene_card(h: &mut Html, input: &PageInput<'_>) {
    let s = input.snapshot;
    let scene = &s.scene;
    h.raw("<div class=\"card\"><h2>Scene</h2>\n<dl>\n<dt>current</dt><dd>");
    h.pill("info", scene.scene.name());
    h.raw("</dd>\n");
    h.fact("revision", scene.revision);
    match scene.changed_tick {
        Some(tick) => h.fact(
            "last change",
            format_args!("tick {tick}, {}", ago(s.tick, tick)),
        ),
        None => h.fact("last change", "none since start"),
    }
    // Who has it: of the shards that hold a control link. While one of them
    // has not confirmed, the card says that a switch is under way and for whom
    // it is waiting, not only how many.
    let progress = s.scene_progress();
    h.raw("<dt>confirmed by</dt><dd>");
    if progress.connected() == 0 {
        h.raw("<span class=\"dim\">no shard is connected</span>");
    } else if progress.settled() {
        h.pill(
            "ok",
            format_args!(
                "{} of {} connected shards",
                progress.confirmed.len(),
                progress.connected()
            ),
        );
    } else {
        h.pill(
            "warn",
            format_args!(
                "switching → {}, {} of {} confirmed",
                progress.scene,
                progress.confirmed.len(),
                progress.connected()
            ),
        );
        h.raw("<div class=\"sub\">waiting for ");
        for (n, shard) in progress.waiting.iter().enumerate() {
            if n > 0 {
                h.raw(", ");
            }
            h.text(shard);
        }
        h.raw("</div>");
    }
    h.raw("</dd>\n");
    match &scene.path {
        Some(path) => h.fact("scene file", path.display()),
        None => h.fact("scene file", "none: not kept across a restart"),
    }
    h.raw("</dl>\n<p>");
    form_open(h, input, "action/scene");
    for choice in space_protocol::Scene::ALL {
        h.wrapped("<button name=\"scene\" value=\"", choice.name(), "\"");
        if choice == scene.scene {
            h.wrapped(
                " disabled title=\"This is the scene that is running: every shard is told to \
                 show it.\">",
                choice.name(),
                " (current)</button>",
            );
        } else {
            h.wrapped(
                " title=\"Save the scene, tell every connected shard and, through the shards, \
                 every client. A shard that registers later is told at once. This is not a map \
                 change: it does not wait for an epoch and does not delay one.\">Switch all \
                 shards to ",
                choice.name(),
                "</button>",
            );
        }
    }
    h.raw(
        "</form></p>\n<p class=\"note\">One click switches; the answer comes when the shards \
         have confirmed. A shard confirms the revision it runs; one that has not confirmed the \
         current revision is told again every second.</p></div>\n",
    );
}

/// How what a shard has confirmed of the scene reads: a pill class and its
/// label.
fn scene_state(s: &Snapshot, view: &ShardView) -> (&'static str, &'static str) {
    match s.scene.sync_of(view.scene) {
        SceneSync::InSync => ("ok", "current"),
        SceneSync::Behind => ("warn", "behind"),
        SceneSync::Differs => ("bad", "not the cluster's"),
        SceneSync::Unconfirmed if view.connected => ("warn", "not confirmed yet"),
        SceneSync::Unconfirmed => ("dim", "not confirmed"),
    }
}

fn in_flight(h: &mut Html, input: &PageInput<'_>) {
    let s = input.snapshot;
    let Some(flight) = &s.in_flight else {
        return;
    };
    h.raw("<section class=\"card flight\"><h2>Epoch change in flight</h2>\n<dl>\n");
    h.fact("epoch", format_args!("{} → {}", s.epoch, flight.epoch));
    h.fact("change", &flight.what);
    h.raw("<dt>acks</dt><dd>");
    let acked = flight.recipients.iter().filter(|(_, acked)| *acked).count();
    h.text(format_args!("{acked} of {}: ", flight.recipients.len()));
    for (shard, acked) in &flight.recipients {
        if *acked {
            h.pill("ok", format_args!("{shard} acked"));
        } else {
            h.pill("warn", format_args!("{shard} waiting"));
        }
        h.raw(" ");
    }
    h.raw("</dd>\n");
    h.fact(
        "activates at",
        format_args!("tick {}", flight.activation_tick),
    );
    h.fact(
        "deadline",
        format_args!(
            "tick {}, in {}",
            flight.deadline_tick,
            span(seconds(flight.deadline_tick.saturating_sub(s.tick)))
        ),
    );
    h.raw(
        "</dl>\n<p class=\"note\">If an ack is still missing at the deadline the change is \
         aborted and the map stays exactly as it is. No other change can start meanwhile.</p>\
         </section>\n",
    );
}

/// How a shard's connection reads: a pill class and its label.
fn link_state(view: &ShardView) -> (&'static str, &'static str) {
    match (view.connected, view.registered, view.role) {
        (true, _, _) => ("ok", "connected"),
        (false, true, _) => ("bad", "lost"),
        (false, false, ShardRole::ReservedRoot) => ("dim", "not claimed yet"),
        (false, false, _) => ("bad", "never registered"),
    }
}

fn shards(h: &mut Html, input: &PageInput<'_>) {
    let s = input.snapshot;
    let now = s.tick;
    h.wrapped(
        "<section class=\"card\"><h2>Shards<span class=\"count\">",
        s.shards.len(),
        "</span></h2>\n<div class=\"scroll\"><table>\n<tr><th>shard</th><th>link</th>\
         <th>address</th><th>region</th><th>scene</th><th>last message</th>\
         <th>load report</th><th>autoscaler</th></tr>\n",
    );
    for view in &s.shards {
        h.wrapped("<tr><td class=\"mono\">", view.shard, "</td><td>");
        let (class, label) = link_state(view);
        h.pill(class, label);
        match view.role {
            ShardRole::Joining => {
                h.raw(" ");
                h.pill("info", "new, epoch pending");
            }
            ShardRole::Dead => {
                h.raw(" ");
                h.pill("bad", "dead, to be replaced");
            }
            ShardRole::Merging => {
                h.raw(" ");
                h.pill("info", "being merged away");
            }
            ShardRole::Unowned => {
                h.raw(" ");
                h.pill("dim", "owns no region");
            }
            ShardRole::Owner | ShardRole::ReservedRoot => {}
        }
        h.raw("</td><td class=\"mono\">");
        match view.advertised {
            Some(address) => h.text(address),
            None => h.raw("<span class=\"dim\">unknown</span>"),
        }
        h.raw("</td><td class=\"mono\">");
        if view.regions.is_empty() {
            h.raw("<span class=\"dim\">none</span>");
        }
        for (n, region) in view.regions.iter().enumerate() {
            if n > 0 {
                h.raw("<br>");
            }
            h.text(node_path(*region));
        }
        h.raw("</td><td>");
        if let Some(confirmed) = view.scene {
            h.text(format_args!(
                "{}, revision {} ",
                confirmed.scene, confirmed.revision
            ));
        }
        let (class, label) = scene_state(s, view);
        h.pill(class, label);
        h.raw("</td><td>");
        match view.last_heard {
            Some(heard) => h.text(format_args!(
                "{}, {} (tick {})",
                heard.what.label(),
                ago(now, heard.tick),
                heard.tick
            )),
            None => h.raw("<span class=\"dim\">never</span>"),
        }
        h.raw("</td><td>");
        match view.report {
            Some(report) => h.text(format_args!(
                "{} player(s), tick p95 {:.1} ms, load {:.2}, {}",
                report.players,
                report.tick_p95_ms,
                report.load_metric,
                ago(now, report.tick)
            )),
            None => h.raw("<span class=\"dim\">no report yet</span>"),
        }
        h.raw("</td><td>");
        match view.scale {
            Some(scale) if scale.immune_until > now => h.text(format_args!(
                "cooling down, {} left",
                span(seconds(scale.immune_until - now))
            )),
            Some(scale) => match (scale.hot_since, scale.cold_since) {
                (Some(since), _) => h.text(format_args!(
                    "hot for {} of {}",
                    span(seconds(now.saturating_sub(since))),
                    span(seconds(SPLIT_DWELL_TICKS))
                )),
                (None, Some(since)) => h.text(format_args!(
                    "cold for {} of {}",
                    span(seconds(now.saturating_sub(since))),
                    span(seconds(MERGE_DWELL_TICKS))
                )),
                (None, None) => h.raw("<span class=\"dim\">within the band</span>"),
            },
            None => h.raw("<span class=\"dim\">not tracked</span>"),
        }
        h.raw("</td></tr>\n");
    }
    h.raw("</table></div></section>\n");
}

/// The owner's connection state, looked up in the shard list.
fn owner_link(s: &Snapshot, shard: ShardId) -> (&'static str, &'static str) {
    s.shards
        .iter()
        .find(|view| view.shard == shard)
        .map_or(("dim", "unknown"), link_state)
}

fn regions(h: &mut Html, input: &PageInput<'_>) {
    let s = input.snapshot;
    h.wrapped(
        "<section class=\"card\"><h2>Map: regions at epoch ",
        s.epoch,
        "",
    );
    h.wrapped(
        "<span class=\"count\">",
        s.regions.len(),
        "</span></h2>\n<div class=\"scroll\"><table>\n<tr><th>region (level:x:y:z)</th>\
         <th>edge</th><th>owner</th><th>split: carve an octant for a new shard</th>\
         <th>merge</th></tr>\n",
    );
    for region in &s.regions {
        region_row(h, input, region);
    }
    h.raw(
        "</table></div>\n<p class=\"note\">A region is an octree node: level 0 is all of space, \
         each level halves the edge. A split carves one octant (bit 0 = +x, bit 1 = +y, \
         bit 2 = +z) out of a region for a newly launched shard; the rest stays with its owner. \
         <q>auto</q> takes the first free octant counting from the one that holds the region's \
         centre, where joining ships spawn. A region can also be carved more than one level \
         down (a single 1 km cell under the root); it is listed under the region around it. A \
         merge gives a carved region back to the owner of the region around it. Both go \
         through the epoch protocol: prepare, ack by every shard of the map, commit, one \
         change at a time.</p></section>\n",
    );
}

fn region_row(h: &mut Html, input: &PageInput<'_>, region: &RegionView) {
    let s = input.snapshot;
    // The region: its address, then the same thing spelled out.
    h.raw("<tr><td class=\"");
    h.raw(match region.nesting {
        0 => "in0",
        1 => "nested in1",
        2 => "nested in2",
        3 => "nested in3",
        _ => "nested in4",
    });
    h.raw("\">");
    if region.nesting > 0 {
        h.raw("<span class=\"twig\">└</span>");
    }
    h.wrapped("<span class=\"mono\">", node_path(region.node), "</span>");
    h.wrapped(
        "<div class=\"sub\">",
        format_args!(
            "level {}, index {}, {}, {}",
            region.node.depth, region.node.index[0], region.node.index[1], region.node.index[2]
        ),
        "",
    );
    match (region.octant, region.parent) {
        (Some(octant), Some(parent)) => h.text(format_args!(
            ": octant {octant} ({}) of {}",
            octant_signs(octant),
            node_path(parent)
        )),
        // Carved more than one level down (a single cell under the root).
        (None, Some(parent)) => h.text(format_args!(
            ": carved out of {}, {} levels up",
            node_path(parent),
            region.node.depth.saturating_sub(parent.depth)
        )),
        (_, None) => {}
    }
    h.raw("</div></td>");
    h.wrapped(
        "<td class=\"num\">",
        distance(NodeRef::size_m(region.node.depth)),
        "</td>",
    );
    // The owner: who, whether it is there, and where the map says it is.
    h.wrapped("<td><span class=\"mono\">", region.shard, "</span> ");
    let (class, label) = owner_link(s, region.shard);
    h.pill(class, label);
    h.raw("<div class=\"sub mono\">");
    match region.endpoint {
        Some(endpoint) => h.text(endpoint),
        None => h.raw("address not known yet"),
    }
    h.raw("</div></td><td>");
    split_cell(h, input, region);
    h.raw("</td><td>");
    merge_cell(h, input, region);
    h.raw("</td></tr>\n");
}

fn split_cell(h: &mut Html, input: &PageInput<'_>, region: &RegionView) {
    let blocked = region.split_refusal.is_some();
    form_open(h, input, "action/split");
    h.wrapped(
        "<input type=\"hidden\" name=\"shard\" value=\"",
        region.shard.0,
        "\">",
    );
    h.raw("<button name=\"octant\" value=\"auto\"");
    if blocked || region.free_octants.is_empty() {
        h.raw(" disabled");
    }
    h.wrapped(
        " title=\"The first free octant, counting from octant ",
        crate::topology::centre_octant(region.node),
        " (the one holding the region's centre).\">auto</button>",
    );
    for octant in 0..8u8 {
        let free = region.free_octants.contains(&octant);
        h.wrapped("<button name=\"octant\" value=\"", octant, "\"");
        if !free {
            h.raw(" class=\"taken\"");
        }
        if blocked || !free {
            h.raw(" disabled");
        }
        h.raw(" title=\"");
        h.text(format_args!("octant {octant}: {}", octant_signs(octant)));
        if !free {
            h.raw(" (already carved out)");
        }
        h.wrapped("\">", octant, "</button>");
    }
    h.raw("</form>");
    if let Some(why) = &region.split_refusal {
        h.wrapped("<div class=\"why\">", why, "</div>");
    }
}

fn merge_cell(h: &mut Html, input: &PageInput<'_>, region: &RegionView) {
    let s = input.snapshot;
    let Some(parent) = region.parent else {
        h.raw("<span class=\"dim\">top-level region</span>");
        return;
    };
    let into = s.regions.iter().find(|r| r.node == parent).map(|r| r.shard);
    form_open(h, input, "action/merge");
    h.wrapped(
        "<input type=\"hidden\" name=\"shard\" value=\"",
        region.shard.0,
        "\">",
    );
    h.raw("<button");
    if region.merge_refusal.is_some() {
        h.raw(" disabled");
    }
    h.raw(
        " title=\"Give this region back to the owner of the region around it and stop this \
         shard, once the epoch has committed.\">merge into ",
    );
    match into {
        Some(shard) => h.text(shard),
        None => h.raw("the region around it"),
    }
    h.raw("</button></form>");
    if let Some(why) = &region.merge_refusal {
        h.wrapped("<div class=\"why\">", why, "</div>");
    }
}

fn events(h: &mut Html, input: &PageInput<'_>) {
    let s = input.snapshot;
    h.wrapped(
        "<section class=\"card\"><h2>Recent events<span class=\"count\">",
        s.events.len(),
        "</span></h2>\n",
    );
    if s.events.is_empty() {
        h.raw("<p class=\"note\">Nothing has happened yet.</p></section>\n");
        return;
    }
    h.raw(
        "<div class=\"scroll\"><table>\n<tr><th>when</th><th>ago</th><th>tick</th><th>about</th>\
         <th>event</th></tr>\n",
    );
    for event in &s.events {
        h.wrapped(
            "<tr><td class=\"mono\">",
            clock_of(input, event.tick),
            "</td>",
        );
        h.wrapped(
            "<td class=\"num\">",
            span(seconds(s.tick.saturating_sub(event.tick))),
            "</td>",
        );
        h.wrapped("<td class=\"num\">", event.tick, "</td><td>");
        h.pill(
            match event.severity {
                Severity::Good => "ok",
                Severity::Warn => "warn",
                Severity::Info => match event.kind {
                    EventKind::Operator => "info",
                    _ => "dim",
                },
            },
            event.kind.label(),
        );
        h.wrapped("</td><td class=\"wrap\">", &event.text, "");
        if event.repeats > 1 {
            h.wrapped(
                " <span class=\"dim\">(",
                format_args!("{} times since tick {}", event.repeats, event.first_tick),
                ")</span>",
            );
        }
        h.raw("</td></tr>\n");
    }
    h.raw(
        "</table></div>\n<p class=\"note\">Newest first. Times are reckoned from the tick clock, \
         so they are approximate. The log keeps the most recent entries only and starts empty \
         when the orchestrator starts.</p></section>\n",
    );
}

fn footer(h: &mut Html, input: &PageInput<'_>) {
    h.wrapped(
        "<footer>space-orchestrator ",
        input.info.version,
        " · the same state as JSON: <a href=\"api/state\">api/state</a> · rendered ",
    );
    h.wrapped("", clock(input.info.now_unix), " at tick ");
    h.wrapped("", input.snapshot.tick, "</footer>\n");
}

/// A small page for a request that was refused (`status`, `message`), with a
/// link back to the dashboard at `home` (a relative URL chosen by the caller).
#[must_use]
pub fn error_page(status: u16, message: &str, style_nonce: &str, home: &'static str) -> String {
    let mut h = Html::new();
    h.raw(
        "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n\
         <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n\
         <meta name=\"robots\" content=\"noindex, nofollow\">\n",
    );
    h.wrapped(
        "<title>",
        status,
        " · space-orchestrator</title>\n<link rel=\"icon\" href=\"data:,\">\n",
    );
    h.wrapped("<style nonce=\"", style_nonce, "\">");
    h.raw(STYLE);
    h.raw("</style>\n</head>\n<body>\n<header><h1>space-orchestrator</h1></header>\n<main>\n");
    h.raw("<section class=\"card\"><h2>");
    h.text(format_args!("{status} {}", super::http::reason(status)));
    h.wrapped("</h2>\n<p>", message, "</p>\n<p><a href=\"");
    h.raw(home);
    h.raw("\">Back to the dashboard</a></p></section>\n</main>\n</body>\n</html>\n");
    h.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_five_significant_characters_are_escaped() {
        assert_eq!(
            escape(r#"<script>alert("x" & 'y')</script>"#),
            "&lt;script&gt;alert(&quot;x&quot; &amp; &#39;y&#39;)&lt;/script&gt;"
        );
        assert_eq!(escape("plain text é ✓"), "plain text é ✓");
        assert_eq!(escape(""), "");
        // Already-escaped text is escaped again: nothing is ever trusted.
        assert_eq!(escape("&amp;"), "&amp;amp;");
    }

    #[test]
    fn everything_formatted_into_the_page_is_escaped() {
        let mut h = Html::new();
        h.text(format_args!("{} & {}", "<a>", 3));
        h.wrapped("<b title=\"", "\" onmouseover=\"x", "\">");
        h.pill("ok", "<i>");
        h.fact("key", "</dd><script>");
        let out = h.finish();
        assert_eq!(
            out,
            "&lt;a&gt; &amp; 3<b title=\"&quot; onmouseover=&quot;x\">\
             <span class=\"pill ok\">&lt;i&gt;</span>\
             <dt>key</dt><dd>&lt;/dd&gt;&lt;script&gt;</dd>\n"
        );
    }

    #[test]
    fn spans_read_like_a_person_would_say_them() {
        assert_eq!(span(0.0), "0.0 s");
        assert_eq!(span(0.44), "0.4 s");
        assert_eq!(span(9.96), "10.0 s");
        assert_eq!(span(10.0), "10 s");
        assert_eq!(span(59.9), "59 s");
        assert_eq!(span(65.0), "1 min 05 s");
        assert_eq!(span(3_600.0), "1 h 00 min");
        assert_eq!(span(7_325.0), "2 h 02 min");
        assert_eq!(span(90_000.0), "1 d 01 h");
        assert_eq!(span(-3.0), "0 s");
        assert_eq!(span(f64::NAN), "0 s");
        assert_eq!(ago(300, 240), "2.0 s ago");
        assert_eq!(
            ago(10, 20),
            "0.0 s ago",
            "a tick in the future is not negative"
        );
    }

    #[test]
    fn a_unix_time_becomes_a_utc_date() {
        assert_eq!(clock(0), "1970-01-01 00:00:00 UTC");
        assert_eq!(clock(951_782_400), "2000-02-29 00:00:00 UTC");
        assert_eq!(clock(1_700_000_000), "2023-11-14 22:13:20 UTC");
        assert_eq!(clock(1_735_689_599), "2024-12-31 23:59:59 UTC");
        assert_eq!(clock(1_735_689_600), "2025-01-01 00:00:00 UTC");
        assert_eq!(clock(4_102_444_800), "2100-01-01 00:00:00 UTC");
    }

    #[test]
    fn distances_pick_a_readable_unit() {
        assert_eq!(distance(1_024.0), "1024 m");
        assert_eq!(distance(1_048_576.0), "1049 km");
        assert_eq!(distance(NodeRef::size_m(40)), "0.115 AU");
        assert_eq!(distance(NodeRef::size_m(20)), "1.904 ly");
        assert_eq!(distance(f64::INFINITY), "?");
    }
}
