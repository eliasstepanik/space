//! Custom archetype ECS for `space` (Phase 1, task 1.2).
//!
//! Hand-written on purpose: `hecs`/`bevy_ecs` are not used, per `docs/plan.md`
//! ("Custom archetype ECS"). The crate compiles with `#![forbid(unsafe_code)]`
//! — type-erased column storage is a `Box<dyn Any + Send>` wrapping a *typed*
//! `Vec<T>`, so every access goes through a checked downcast and there is no
//! `unsafe` at all (layout and rationale in [`storage`]).
//!
//! # Core model
//!
//! * [`Entity`] is a generational index (`index: u32`, `generation: u32`); a
//!   handle is rejected forever after its entity is despawned, because the
//!   slot's generation has moved on.
//! * Components are registered by a compile-time [`ComponentId`] through the
//!   [`ComponentDef`] trait, and [`register_components!`] turns a list of
//!   `(ID, Type)` pairs into the static replication registry
//!   ([`ComponentInfo`]) that later crates (protocol/sim) use to learn which
//!   components exist, how large they are and whether they are replicated.
//! * Entities with the same component set live in one [`Archetype`]: one
//!   `Vec<T>` column per component, so a query walks contiguous SoA rows.
//!   Adding or removing a component migrates the row to the neighbouring
//!   archetype.
//! * [`Query`] gives typed iteration over `&T` / `&mut T` terms plus a
//!   `.changed_since(tick)` filter; every write through [`World::get_mut`] or a
//!   `&mut` query term stamps the row with [`World::tick`].
//! * [`CommandBuffer`] defers structural edits to a stage boundary, [`Stage`]
//!   runs `fn(&mut World)` systems in order.
//!
//! # Parallel-friendliness
//!
//! v1 uses a single-threaded scheduler, but nothing here precludes a parallel
//! one: [`Query::for_each`] hands the closure a *row* of borrowed components,
//! never a borrow of the whole [`World`], so a scheduler can hand each worker a
//! disjoint subset of archetypes (each owning its columns) and join before the
//! next stage. [`ComponentData`] requires `Send` for exactly that reason, and
//! `World` keeps no global state.
//!
//! # Benchmarks
//!
//! Measured on the development machine (`x86_64`, `--release`, `lto = "thin"`);
//! see `tests/acceptance.rs`:
//!
//! | workload (release)                                        | time     |
//! |-----------------------------------------------------------|----------|
//! | 1 000 000 × [`World::spawn`] with (Position, Velocity)     | ~0.3 s   |
//! | 1 000 000 rows iterated, positions summed                 | ~2 ms    |
//!
//! Both benchmark tests assert the brief's budgets (1 s / 10 ms) and are gated
//! to release builds (`#[cfg(not(debug_assertions))]`): at `opt-level = 1` the
//! same work is several times slower, so those budgets are meaningless in a
//! debug build. Run them with `cargo test --release -p space-ecs bench_`.

#![forbid(unsafe_code)]
#![warn(missing_docs, missing_debug_implementations)]

mod archetype;
mod command;
mod component;
mod entity;
mod macros;
mod query;
mod stage;
#[doc(hidden)]
pub mod storage;
mod world;

pub use archetype::Archetype;
pub use command::{CommandBuffer, CommandError};
pub use component::{
    Component, ComponentData, ComponentDef, ComponentId, ComponentInfo, OwnedComponent,
    MAX_COMPONENTS,
};
pub use entity::Entity;
pub use query::{Access, Query, QueryTerms, WithEntity};
pub use stage::Stage;
pub use world::World;
