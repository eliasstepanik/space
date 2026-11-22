//! The staged scheduler.

use crate::world::World;

/// A named list of systems run in order.
///
/// A system is a plain `fn(&mut World)`. Stages give the (single-threaded, for
/// v1) scheduler a fixed order — e.g. `physics` before `replication` — and a
/// natural point to flush a [`CommandBuffer`](crate::CommandBuffer).
///
/// ```
/// # use space_ecs::{Stage, World};
/// fn count(w: &mut World, out: &mut usize) { *out += w.len(); }
/// let stage = Stage::new("count");
/// assert_eq!(stage.name(), "count");
/// ```
#[derive(Clone)]
pub struct Stage {
    name: &'static str,
    systems: Vec<fn(&mut World)>,
}

impl std::fmt::Debug for Stage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Stage")
            .field("name", &self.name)
            .field("systems", &self.systems.len())
            .finish()
    }
}

impl Stage {
    /// An empty stage named `name`.
    pub fn new(name: &'static str) -> Self {
        Self {
            name,
            systems: Vec::new(),
        }
    }

    /// The stage's name.
    #[inline]
    pub fn name(&self) -> &'static str {
        self.name
    }

    /// Appends a system, returning `self` for chaining.
    #[must_use]
    #[allow(
        clippy::should_implement_trait,
        reason = "the brief mandates `Stage::add(system_fn)`; this is not arithmetic addition"
    )]
    pub fn add(mut self, system: fn(&mut World)) -> Self {
        self.systems.push(system);
        self
    }

    /// Runs every system in order against `world`.
    pub fn run(&self, world: &mut World) {
        for system in &self.systems {
            system(world);
        }
    }
}
