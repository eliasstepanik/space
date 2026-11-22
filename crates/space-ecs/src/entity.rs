//! Generational entity handles.

use std::fmt;

/// An ECS entity handle: a generational index.
///
/// `index` addresses a slot in [`World::locations`](crate::World), `generation`
/// distinguishes successive lives of that slot. A handle taken before a
/// [`World::despawn`](crate::World::despawn) is rejected afterwards because the
/// slot's generation has moved on, which is what makes stale handles safe
/// (checked in [`World::entity_is_alive`](crate::World::entity_is_alive) and on
/// every accessor).
///
/// [`Display`](fmt::Display) prints `index:generation`.
///
/// ```
/// use space_ecs::Entity;
/// let e = Entity::new(7, 2);
/// assert_eq!(e.index(), 7);
/// assert_eq!(e.generation(), 2);
/// assert_eq!(e.to_string(), "7:2");
/// ```
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Entity {
    index: u32,
    generation: u32,
}

impl Entity {
    /// Builds a handle for slot `index` at `generation`.
    ///
    /// Only the ECS itself (and tests) should need this: real handles come from
    /// [`World::spawn`](crate::World::spawn).
    pub const fn new(index: u32, generation: u32) -> Self {
        Self { index, generation }
    }

    /// Slot index in the world's entity table.
    #[inline]
    pub const fn index(self) -> u32 {
        self.index
    }

    /// Generation of the slot: 0 for the first life of an index, incremented on
    /// every re-use of the slot.
    #[inline]
    pub const fn generation(self) -> u32 {
        self.generation
    }
}

impl fmt::Display for Entity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.index, self.generation)
    }
}
