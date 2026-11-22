//! The query hot path must not touch the heap per row.
//!
//! A simulation step and a replication pass walk every entity through a query
//! every tick, so an allocation *per row* is an allocation per entity per tick
//! (and, for replication, per client). This test counts heap allocations with a
//! counting global allocator and asserts that iterating a query costs the same
//! handful of allocations for 16 rows as for 4096.
//!
//! It lives in its own test binary because a `#[global_allocator]` is
//! process-wide, and it is the only place in `space-ecs` that needs `unsafe`
//! (implementing `GlobalAlloc`); the library itself stays
//! `#![forbid(unsafe_code)]`.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use space_ecs::{Component, ComponentDef, Entity, WithEntity, World};

thread_local! {
    /// Allocations made by the current thread. `const`-initialised and without
    /// a destructor, so reading it never allocates or registers anything,
    /// which is what makes it usable from inside the allocator.
    static ALLOCATIONS: Cell<u64> = const { Cell::new(0) };
}

/// The system allocator, plus a per-thread count of `alloc`/`realloc` calls.
struct Counting;

// SAFETY: every method forwards its arguments unchanged to `System`, which
// upholds the `GlobalAlloc` contract; the only addition is a thread-local
// counter bump that neither allocates nor unwinds.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let _ = ALLOCATIONS.try_with(|n| n.set(n.get() + 1));
        // SAFETY: the caller upholds the contract of `GlobalAlloc::alloc`.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` was returned by `System` (through this type) for `layout`.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let _ = ALLOCATIONS.try_with(|n| n.set(n.get() + 1));
        // SAFETY: `ptr` was returned by `System` (through this type) for
        // `layout`; the caller upholds the rest of the contract of `realloc`.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// Heap allocations the current thread makes while running `f`.
fn allocations_during(f: impl FnOnce()) -> u64 {
    let before = ALLOCATIONS.with(Cell::get);
    f();
    ALLOCATIONS.with(Cell::get) - before
}

#[derive(Clone, Copy)]
struct Position(f32);
#[derive(Clone, Copy)]
struct Velocity(f32);
#[derive(Clone, Copy)]
struct Mass(f32);

impl ComponentDef for Position {
    const ID: u16 = 0;
    type Data = Self;
}
impl ComponentDef for Velocity {
    const ID: u16 = 1;
    type Data = Self;
}
impl ComponentDef for Mass {
    const ID: u16 = 2;
    type Data = Self;
}

/// A world of `rows` entities in two archetypes, half of them written after
/// tick 0 so a `changed_since(0)` query has rows to keep and rows to skip.
fn world_of(rows: u32) -> World {
    let mut w = World::new();
    let mut entities: Vec<Entity> = Vec::new();
    for i in 0..rows {
        let (p, v, m) = (Position(i as f32), Velocity(1.0), Mass(2.0));
        entities.push(if i % 2 == 0 {
            w.spawn(&[Component::new(&p), Component::new(&v)])
        } else {
            w.spawn(&[Component::new(&p), Component::new(&v), Component::new(&m)])
        });
    }
    w.tick();
    for e in entities.iter().step_by(4) {
        w.get_mut::<Velocity>(*e).unwrap().0 += 1.0;
    }
    for e in entities.iter().skip(1).step_by(4) {
        w.get_mut::<Velocity>(*e).unwrap().0 += 1.0;
    }
    w
}

/// Run every per-row query shape the simulation and the replication pass use,
/// and return (rows visited, heap allocations made).
fn walk(w: &mut World) -> (u64, u64) {
    let mut visited = 0u64;
    let mut checksum = 0.0f64;
    let allocations = allocations_during(|| {
        // The simulation's read pass: an entity term over two components.
        {
            let mut q = w.query::<WithEntity<(&Position, &Velocity)>>();
            q.for_each(|((p, v), e)| {
                visited += 1;
                checksum += f64::from(p.0 + v.0) + f64::from(e.index());
            });
        }
        // The replication pass: the same, filtered by change tick.
        {
            let mut q = w
                .query::<WithEntity<(&Position, &Velocity)>>()
                .changed_since(0);
            q.for_each(|((p, _v), e)| {
                visited += 1;
                checksum += f64::from(p.0) + f64::from(e.generation());
            });
        }
        // Three terms, one of them mutable, under the entity term.
        {
            let mut q = w.query::<WithEntity<(&mut Position, &Velocity, &Mass)>>();
            q.for_each(|((p, v, m), _e)| {
                visited += 1;
                p.0 += v.0 * m.0;
            });
        }
        // The lending iterator, two and three terms, with and without a filter.
        {
            let mut q = w.query::<(&mut Velocity, &Position)>();
            while let Some((v, p)) = q.next() {
                visited += 1;
                v.0 += p.0 * 0.0;
            }
        }
        {
            let mut q = w
                .query::<WithEntity<(&Position, &Velocity, &Mass)>>()
                .changed_since(0);
            while let Some(((p, _v, _m), _e)) = q.next() {
                visited += 1;
                checksum += f64::from(p.0);
            }
        }
    });
    assert!(checksum.is_finite());
    (visited, allocations)
}

#[test]
fn query_iteration_does_not_allocate_per_row() {
    let mut small = world_of(16);
    let mut large = world_of(4096);
    let (small_rows, small_allocations) = walk(&mut small);
    let (large_rows, large_allocations) = walk(&mut large);

    // Sanity: the walks really visited rows, 256 times as many in the large
    // world, and the counter is live (building a query's plan allocates).
    assert!(small_rows >= 16, "{small_rows} rows visited");
    assert_eq!(large_rows, small_rows * 256);
    assert!(small_allocations > 0, "the counting allocator saw nothing");

    // What a query allocates is its plan (one small `Vec` per matching
    // archetype) and, for `changed_since`, one mask buffer: a fixed number of
    // allocations however many rows there are. Per-row allocation would add
    // thousands here (the old code made 3-4 per row for every multi-term row).
    assert_eq!(
        small_allocations, large_allocations,
        "allocations must not depend on the row count \
         ({small_rows} rows: {small_allocations}, {large_rows} rows: {large_allocations})"
    );
    assert!(
        large_allocations <= 64,
        "{large_allocations} allocations for five queries over two archetypes"
    );
}
