//! The scene: which world a cluster shows, and the messages that carry it.
//!
//! A *scene* is what clients draw around the ships a shard simulates (the demo
//! area, the material chart, the solar system, the galaxy). It is **cluster
//! state**, not a client option: the orchestrator owns it, every shard runs the
//! one it was told, and every client draws the one its shard reports — so two
//! players on one shard can never see two different worlds.
//!
//! # The catalogue
//!
//! [`Scene`] lists the scenes this build knows. Each has a **stable numeric wire
//! id** ([`Scene::id`]) and a **stable lower-case name** ([`Scene::name`]):
//!
//! | id | name | |
//! |---|---|---|
//! | 0 | `demo-area` | the default ([`Scene::DEFAULT`]) |
//! | 1 | `materials` | the PBR material chart |
//! | 2 | `solar` | the real-size solar system |
//! | 3 | `galaxy` | the deterministic galaxy |
//!
//! An id or a name is never reused for another scene and never renumbered: a
//! new scene is **appended** with the next free id.
//!
//! The catalogue will grow, so the wire carries a [`SceneId`] — a bare `u16` —
//! and not the enum: **every** id decodes, also one this build has never heard
//! of. A receiver resolves it with [`SceneId::known`]; a client that does not
//! know the scene draws [`SceneId::or_default`] instead and says so in its log.
//!
//! # Revisions
//!
//! A scene travels with a **revision** ([`SceneState`]). The authority that
//! chooses the scene (the orchestrator, or a standalone shard's command line)
//! raises the revision with every change, and a receiver adopts a state only
//! when it [`supersedes`](SceneState::supersedes) the one it holds. Messages may
//! be late, repeated or overtake each other (each rides its own stream), and
//! none of that can roll a scene back. Revision 0 means "nobody has set a scene
//! yet": the default scene applies ([`SceneState::UNSET`]).
//!
//! # The messages
//!
//! | message | direction | when |
//! |---|---|---|
//! | [`Message::SceneSet`](crate::Message::SceneSet) | orchestrator → shard | right behind the `ShardWelcome` on every (re-)registration, and on every change |
//! | [`Message::SceneAck`](crate::Message::SceneAck) | shard → orchestrator | in answer to every `SceneSet`: the state the shard now runs |
//! | [`Message::SceneUpdate`](crate::Message::SceneUpdate) | shard → client | right after the `Welcome` on every join (also the second connection a client opens during a handoff), and on every change |
//!
//! A scene change is **not** an epoch change: the shard map does not move, so
//! none of these messages takes part in, waits for or delays the epoch protocol.

use serde::{Deserialize, Serialize};

use crate::shard::ShardId;

/// A scene this build knows. See the [module docs](self) for the catalogue.
///
/// The enum never crosses the wire; its [`SceneId`] does.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Scene {
    /// The demo area: the scene a fresh cluster shows.
    DemoArea,
    /// The PBR material chart.
    Materials,
    /// The real-size solar system.
    Solar,
    /// The deterministic galaxy.
    Galaxy,
}

impl Scene {
    /// The scene a cluster shows until somebody chooses another one.
    pub const DEFAULT: Self = Self::DemoArea;

    /// Every scene of the catalogue, in id order.
    pub const ALL: [Self; 4] = [Self::DemoArea, Self::Materials, Self::Solar, Self::Galaxy];

    /// The scene's stable wire id.
    #[must_use]
    pub const fn id(self) -> SceneId {
        SceneId(match self {
            Self::DemoArea => 0,
            Self::Materials => 1,
            Self::Solar => 2,
            Self::Galaxy => 3,
        })
    }

    /// The scene's stable lower-case name: what `--scene` takes, what the web
    /// UI posts and what logs print.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::DemoArea => "demo-area",
            Self::Materials => "materials",
            Self::Solar => "solar",
            Self::Galaxy => "galaxy",
        }
    }

    /// The scene with wire id `id`, if this build knows it.
    #[must_use]
    pub fn from_id(id: SceneId) -> Option<Self> {
        Self::ALL.into_iter().find(|scene| scene.id() == id)
    }

    /// The scene named exactly `name`, if this build knows it.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|scene| scene.name() == name)
    }

    /// The catalogue's names as one list, `"demo-area, materials, solar, galaxy"`
    /// (for usage texts and refusals).
    #[must_use]
    pub fn names() -> String {
        Self::ALL.map(Self::name).join(", ")
    }
}

impl core::fmt::Display for Scene {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.name())
    }
}

/// A scene name that is not in the catalogue.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnknownScene {
    /// The name that was given.
    pub name: String,
}

impl core::fmt::Display for UnknownScene {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "there is no scene named {:?}; the scenes are {}",
            self.name,
            Scene::names()
        )
    }
}

impl std::error::Error for UnknownScene {}

impl core::str::FromStr for Scene {
    type Err = UnknownScene;

    fn from_str(name: &str) -> Result<Self, Self::Err> {
        Self::from_name(name).ok_or_else(|| UnknownScene {
            name: name.to_owned(),
        })
    }
}

/// The wire id of a scene.
///
/// Open on purpose: every `u16` is a valid `SceneId`, so a frame naming a scene
/// a newer peer added still decodes. Whether *this build* knows the scene is a
/// separate question, answered by [`SceneId::known`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct SceneId(pub u16);

impl SceneId {
    /// The id of [`Scene::DEFAULT`].
    pub const DEFAULT: Self = Scene::DEFAULT.id();

    /// The scene this id names, or `None` when this build does not know it.
    #[must_use]
    pub fn known(self) -> Option<Scene> {
        Scene::from_id(self)
    }

    /// The scene this id names, or [`Scene::DEFAULT`] when this build does not
    /// know it — what a client draws for a scene it has never heard of.
    #[must_use]
    pub fn or_default(self) -> Scene {
        self.known().unwrap_or(Scene::DEFAULT)
    }
}

impl Default for SceneId {
    fn default() -> Self {
        Self::DEFAULT
    }
}

impl From<Scene> for SceneId {
    fn from(scene: Scene) -> Self {
        scene.id()
    }
}

impl core::fmt::Display for SceneId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.known() {
            Some(scene) => f.write_str(scene.name()),
            None => write!(f, "unknown scene #{}", self.0),
        }
    }
}

/// A scene at a revision: the unit every scene message carries and every
/// participant keeps (see the [module docs](self), "Revisions").
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SceneState {
    /// Which scene.
    pub scene: SceneId,
    /// The revision it was chosen at. Monotonic; 0 means "never set".
    pub revision: u64,
}

impl SceneState {
    /// "Nobody has set a scene yet": the default scene at revision 0, which
    /// every real state supersedes.
    pub const UNSET: Self = Self {
        scene: SceneId::DEFAULT,
        revision: 0,
    };

    /// `scene` at `revision`.
    #[must_use]
    pub const fn new(scene: SceneId, revision: u64) -> Self {
        Self { scene, revision }
    }

    /// Whether an authority has set this state (its revision is not 0).
    #[must_use]
    pub const fn is_set(&self) -> bool {
        self.revision != 0
    }

    /// Whether this state replaces `held`: only a strictly newer revision does,
    /// so a late or repeated message changes nothing.
    #[must_use]
    pub const fn supersedes(&self, held: &Self) -> bool {
        self.revision > held.revision
    }
}

impl Default for SceneState {
    fn default() -> Self {
        Self::UNSET
    }
}

impl core::fmt::Display for SceneState {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{} (revision {})", self.scene, self.revision)
    }
}

/// Shard → orchestrator: the scene this shard runs now.
///
/// Sent in answer to every [`Message::SceneSet`](crate::Message::SceneSet),
/// whether or not it changed anything: `applied` is the shard's state *after*
/// the message, so a shard that already held a newer revision says so instead
/// of confirming the stale one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SceneAck {
    /// The shard confirming.
    pub shard: ShardId,
    /// The scene and revision it runs.
    pub applied: SceneState,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The ids and names are the wire and the command line: they must never
    /// move. Adding a scene appends a row here; nothing above it changes.
    #[test]
    fn the_catalogue_ids_and_names_are_pinned() {
        let rows: Vec<(u16, &str)> = Scene::ALL
            .iter()
            .map(|scene| (scene.id().0, scene.name()))
            .collect();
        assert_eq!(
            rows,
            vec![
                (0, "demo-area"),
                (1, "materials"),
                (2, "solar"),
                (3, "galaxy")
            ]
        );
        assert_eq!(Scene::DEFAULT, Scene::DemoArea);
        assert_eq!(SceneId::DEFAULT, SceneId(0));
        assert_eq!(SceneId::default(), SceneId::DEFAULT);
        assert_eq!(Scene::names(), "demo-area, materials, solar, galaxy");
    }

    #[test]
    fn ids_and_names_are_unique_and_round_trip() {
        for scene in Scene::ALL {
            assert_eq!(Scene::from_id(scene.id()), Some(scene));
            assert_eq!(Scene::from_name(scene.name()), Some(scene));
            assert_eq!(scene.name().parse::<Scene>(), Ok(scene));
            assert_eq!(scene.to_string(), scene.name());
            assert_eq!(scene.id().to_string(), scene.name());
            assert_eq!(SceneId::from(scene), scene.id());
            assert_eq!(scene.name(), scene.name().to_ascii_lowercase());
        }
        let ids: std::collections::BTreeSet<u16> = Scene::ALL.iter().map(|s| s.id().0).collect();
        let names: std::collections::BTreeSet<&str> = Scene::ALL.iter().map(|s| s.name()).collect();
        assert_eq!(ids.len(), Scene::ALL.len());
        assert_eq!(names.len(), Scene::ALL.len());
    }

    #[test]
    fn an_unknown_name_is_refused_and_the_refusal_lists_the_catalogue() {
        for bad in ["", "Solar", "solar ", "demo", "demo_area", "0"] {
            assert_eq!(Scene::from_name(bad), None, "{bad:?}");
            let err = bad.parse::<Scene>().expect_err(bad);
            assert_eq!(err.name, bad);
            assert!(err.to_string().contains(&Scene::names()), "{err}");
        }
    }

    /// A scene a newer peer added is not an error here: it resolves to "not
    /// known", falls back to the default and still prints as something.
    #[test]
    fn an_unknown_id_is_not_known_and_falls_back_to_the_default() {
        for raw in [4u16, 5, 999, u16::MAX] {
            let id = SceneId(raw);
            assert_eq!(id.known(), None, "{raw}");
            assert_eq!(id.or_default(), Scene::DEFAULT);
            assert_eq!(id.to_string(), format!("unknown scene #{raw}"));
        }
        assert_eq!(SceneId(2).or_default(), Scene::Solar);
    }

    #[test]
    fn only_a_strictly_newer_revision_supersedes() {
        let held = SceneState::new(Scene::Solar.id(), 4);
        assert!(SceneState::new(Scene::Galaxy.id(), 5).supersedes(&held));
        assert!(
            !SceneState::new(Scene::Galaxy.id(), 4).supersedes(&held),
            "a repeat of the same revision changes nothing"
        );
        assert!(
            !SceneState::new(Scene::Galaxy.id(), 3).supersedes(&held),
            "a late message cannot roll the scene back"
        );
        // Every real state supersedes "never set"; "never set" supersedes nothing.
        assert!(held.supersedes(&SceneState::UNSET));
        assert!(!SceneState::UNSET.supersedes(&SceneState::UNSET));
        assert!(!SceneState::UNSET.is_set() && held.is_set());
        assert_eq!(SceneState::default(), SceneState::UNSET);
        assert_eq!(SceneState::UNSET.scene, SceneId::DEFAULT);
        assert_eq!(held.to_string(), "solar (revision 4)");
    }
}
