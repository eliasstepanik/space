//! Deferred structural edits.

use crate::component::OwnedComponent;
use crate::entity::Entity;
use crate::world::World;

/// Why a recorded command was dropped when the buffer was applied.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CommandError {
    /// The target entity was not alive when the command ran.
    DeadEntity,
}

impl core::fmt::Display for CommandError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            CommandError::DeadEntity => write!(f, "target entity is dead"),
        }
    }
}

impl std::error::Error for CommandError {}

/// A recorded deferred operation.
enum Command {
    Spawn(Vec<OwnedComponent>),
    Despawn(Entity),
    Insert(Entity, OwnedComponent),
    Remove(Entity, u16),
}

/// Records structural edits (`spawn`/`despawn`/`insert`/`remove`) and applies
/// them later on [`World::run`], at a stage boundary.
///
/// Commands that reference a dead entity are *dropped and counted*, never a
/// panic — [`World::run`] returns the number of dropped commands. This lets a
/// system queue an edit for an entity another system despawned earlier in the
/// same stage without coordination.
///
/// Component payloads are stored as owned [`OwnedComponent`]s (built from a
/// [`Component`](crate::Component) with
/// [`Component::to_owned_component`](crate::Component::to_owned_component)), so
/// the buffer can outlive the values it was built from.
///
/// ```
/// # use space_ecs::{Component, ComponentDef, CommandBuffer, World};
/// # #[derive(Clone)] struct A(u8); impl ComponentDef for A { const ID: u16 = 1; type Data = Self; }
/// let mut w = World::new();
/// let mut cb = CommandBuffer::new();
/// cb.spawn(vec![Component::new(&A(3)).to_owned_component()]);
/// let dropped = w.run(&mut cb);
/// assert_eq!(dropped, 0);
/// assert_eq!(w.len(), 1);
/// ```
#[derive(Default)]
pub struct CommandBuffer {
    commands: Vec<Command>,
}

impl std::fmt::Debug for CommandBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CommandBuffer")
            .field("len", &self.commands.len())
            .finish()
    }
}

impl CommandBuffer {
    /// An empty buffer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Records a spawn of the given component set.
    pub fn spawn(&mut self, components: Vec<OwnedComponent>) {
        self.commands.push(Command::Spawn(components));
    }

    /// Records a despawn of `entity`.
    pub fn despawn(&mut self, entity: Entity) {
        self.commands.push(Command::Despawn(entity));
    }

    /// Records inserting `component` into `entity`.
    pub fn insert(&mut self, entity: Entity, component: OwnedComponent) {
        self.commands.push(Command::Insert(entity, component));
    }

    /// Records removing component id `id` from `entity`.
    pub fn remove(&mut self, entity: Entity, id: u16) {
        self.commands.push(Command::Remove(entity, id));
    }

    /// Number of recorded commands.
    pub fn len(&self) -> usize {
        self.commands.len()
    }

    /// `true` when nothing is recorded.
    pub fn is_empty(&self) -> bool {
        self.commands.is_empty()
    }

    /// Applies every command in order, clearing the buffer. Returns the count of
    /// dropped commands (dead targets).
    pub(crate) fn apply(&mut self, world: &mut World) -> usize {
        let mut dropped = 0usize;
        for cmd in std::mem::take(&mut self.commands) {
            match cmd {
                Command::Spawn(components) => {
                    let refs: Vec<crate::Component<'_>> =
                        components.iter().map(borrow_owned).collect();
                    world.spawn(&refs);
                }
                Command::Despawn(e) => {
                    if !world.despawn(e) {
                        dropped += 1;
                    }
                }
                Command::Insert(e, c) => {
                    if !world.insert(e, borrow_owned(&c)) {
                        dropped += 1;
                    }
                }
                Command::Remove(e, id) => {
                    if !world.remove_by_id(e, id) {
                        dropped += 1;
                    }
                }
            }
        }
        dropped
    }
}

/// Borrows an [`OwnedComponent`] as a [`Component`](crate::Component).
///
/// The `new_column`/`clone_data` metadata of `Component` is private, so we
/// rebuild it through the public constructor path: `OwnedComponent` already
/// carries the id/type id and payload; we wrap the payload reference.
fn borrow_owned(owned: &OwnedComponent) -> crate::Component<'_> {
    owned.as_component()
}
