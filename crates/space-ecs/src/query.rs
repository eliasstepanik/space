//! Typed queries over archetypes.
//!
//! A query is described by a *term type* [`QueryTerms`]: `&T`, `&mut T` and
//! tuples of them, where each `T: `[`ComponentDef`](crate::ComponentDef) so the
//! term knows its [`ComponentId`](crate::ComponentId). [`Query`] borrows a
//! slice of archetypes, resolves each term's id to a column index per
//! archetype, and iterates the rows that carry every term.
//!
//! The closure passed to [`Query::for_each`] receives a *row* of borrowed
//! components ([`QueryTerms::Item`]), never a borrow of the whole
//! [`World`](crate::World) — that is what keeps the API parallel-friendly: a
//! scheduler can hand each worker a disjoint subset of archetypes and run
//! `for_each` on each, then join before the next stage.

use crate::archetype::Archetype;
use crate::component::{ComponentData, ComponentDef, ComponentId};
use crate::entity::Entity;
use crate::storage::Column;

/// Splits `slice` into `N` disjoint mutable references at `indices`, in the
/// order the indices were given.
///
/// Returns `None` if any index is duplicated or out of range.
///
/// This is `slice::get_disjoint_mut` (std, safe, stable since Rust 1.86): no
/// heap allocation. It sits on the per-row path of every multi-term
/// [`QueryTerms::get_row`] and of archetype migration, where the hand-rolled
/// version it replaces allocated three or four `Vec`s per call.
#[inline]
pub(crate) fn get_many_mut<T, const N: usize>(
    slice: &mut [T],
    indices: [usize; N],
) -> Option<[&mut T; N]> {
    slice.get_disjoint_mut(indices).ok()
}

/// Internal abstraction over a single access kind (`&T` or `&mut T`).
///
/// Sealed in practice: games express access by writing `&T` / `&mut T` in the
/// query type, not by implementing this.
pub trait Access {
    /// The component this term accesses.
    type Component: ComponentDef;
    /// The borrow handed to the closure.
    type Ref<'a>;
    /// `true` for `&mut T`.
    const MUTABLE: bool;
    /// A borrowed view of a whole column; the type downcast happens once when
    /// this is produced (see [`Access::column`]), so the row loop only indexes.
    type Slice<'a>;
    /// Downcasts `col` once into a column view.
    fn column<'a>(col: &'a mut Column) -> Self::Slice<'a>;
    /// Shortens a column view's lifetime, so a row loop can hand out one row at
    /// a time without moving the view.
    fn reborrow<'b, 'a>(slice: &'b mut Self::Slice<'a>) -> Self::Slice<'b>
    where
        'a: 'b;
    /// Borrows row `i` out of a column view.
    fn row<'a>(slice: Self::Slice<'a>, i: usize) -> Self::Ref<'a>;
    /// Borrows row `i` out of `col` (single-row convenience; downcasts `col`).
    fn get<'a>(col: &'a mut Column, i: usize) -> Self::Ref<'a> {
        Self::row(Self::column(col), i)
    }
}

impl<T: ComponentDef<Data = T> + ComponentData> Access for &T {
    type Component = T;
    type Ref<'a> = &'a T;
    type Slice<'a> = &'a [T];
    const MUTABLE: bool = false;
    fn column<'a>(col: &'a mut Column) -> Self::Slice<'a> {
        col.slice::<T>()
    }
    fn reborrow<'b, 'a>(slice: &'b mut Self::Slice<'a>) -> Self::Slice<'b>
    where
        'a: 'b,
    {
        *slice
    }
    fn row<'a>(slice: Self::Slice<'a>, i: usize) -> Self::Ref<'a> {
        &slice[i]
    }
}

impl<T: ComponentDef<Data = T> + ComponentData> Access for &mut T {
    type Component = T;
    type Ref<'a> = &'a mut T;
    type Slice<'a> = &'a mut [T];
    const MUTABLE: bool = true;
    fn column<'a>(col: &'a mut Column) -> Self::Slice<'a> {
        col.vec_mut::<T>()
    }
    fn reborrow<'b, 'a>(slice: &'b mut Self::Slice<'a>) -> Self::Slice<'b>
    where
        'a: 'b,
    {
        &mut *slice
    }
    fn row<'a>(slice: Self::Slice<'a>, i: usize) -> Self::Ref<'a> {
        &mut slice[i]
    }
}

/// A query term: `&T`, `&mut T`, [`Entity`] or a tuple of them.
///
/// Implemented by the crate for the supported shapes; games do not implement it
/// themselves (see [`Access`]).
pub trait QueryTerms {
    /// The row-shaped item handed to the closure, borrowing the columns.
    type Item<'a>;
    /// The component ids this term reads/writes, in order.
    fn ids() -> Vec<ComponentId>;
    /// `true` if any sub-term is `&mut` (decides whether rows get tick-stamped).
    const MUTABLE: bool;
    /// Iterates rows `0..len`, skipping rows where `mask` is `Some(false)`.
    ///
    /// The contract every implementation keeps (and [`WithEntity`] relies on):
    /// `f` is called **exactly once per unmasked row, in ascending row order**.
    ///
    /// `entities` is the archetype's row→[`Entity`] map, parallel to the
    /// columns; only [`WithEntity`] reads it.
    fn run<'a, F>(
        cols: &'a mut [Column],
        entities: &'a [Entity],
        idx: &[usize],
        len: usize,
        mask: Option<&[bool]>,
        f: &mut F,
    ) where
        F: for<'x> FnMut(Self::Item<'x>);
    /// Builds the item for a single `row`.
    fn get_row<'a>(
        cols: &'a mut [Column],
        entities: &'a [Entity],
        idx: &[usize],
        row: usize,
    ) -> Self::Item<'a>;
}

impl<T: ComponentDef<Data = T> + ComponentData> QueryTerms for &T {
    type Item<'a> = &'a T;
    const MUTABLE: bool = false;
    fn ids() -> Vec<ComponentId> {
        vec![T::ID]
    }
    fn run<'a, F>(
        cols: &'a mut [Column],
        _entities: &'a [Entity],
        idx: &[usize],
        len: usize,
        mask: Option<&[bool]>,
        f: &mut F,
    ) where
        F: for<'x> FnMut(&'x T),
    {
        let s = cols[idx[0]].slice::<T>();
        for i in 0..len {
            if mask.is_none_or(|m| m[i]) {
                f(&s[i]);
            }
        }
    }
    fn get_row<'a>(
        cols: &'a mut [Column],
        _entities: &'a [Entity],
        idx: &[usize],
        row: usize,
    ) -> Self::Item<'a> {
        &cols[idx[0]].slice::<T>()[row]
    }
}

impl<T: ComponentDef<Data = T> + ComponentData> QueryTerms for &mut T {
    type Item<'a> = &'a mut T;
    const MUTABLE: bool = true;
    fn ids() -> Vec<ComponentId> {
        vec![T::ID]
    }
    fn run<'a, F>(
        cols: &'a mut [Column],
        _entities: &'a [Entity],
        idx: &[usize],
        len: usize,
        mask: Option<&[bool]>,
        f: &mut F,
    ) where
        F: for<'x> FnMut(&'x mut T),
    {
        let v = cols[idx[0]].vec_mut::<T>();
        for i in 0..len {
            if mask.is_none_or(|m| m[i]) {
                f(&mut v[i]);
            }
        }
    }
    fn get_row<'a>(
        cols: &'a mut [Column],
        _entities: &'a [Entity],
        idx: &[usize],
        row: usize,
    ) -> Self::Item<'a> {
        &mut cols[idx[0]].vec_mut::<T>()[row]
    }
}

/// Query term that yields the row's [`Entity`] handle.
///
/// It is a *wrapper*, not an [`Access`]: the entity of a row is archetype
/// metadata (the row→entity map), not a component column, so it cannot be
/// expressed as `&T`. Wrap the term tuple you want the entity alongside:
///
/// ```
/// # use space_ecs::{Component, ComponentDef, Entity, World, WithEntity};
/// # #[derive(Clone)] struct P(f32); impl ComponentDef for P { const ID: u16 = 0; type Data = Self; }
/// let mut w = World::new();
/// let e = w.spawn(&[Component::new(&P(1.0))]);
/// let mut seen = Vec::new();
/// {
///     let mut q = w.query::<WithEntity<&P>>();
///     q.for_each(|(p, entity)| seen.push((entity, p.0)));
/// }
/// assert_eq!(seen, vec![(e, 1.0)]);
/// ```
///
/// This is what lets a delta-replication pass report *which* entity changed:
/// `w.query::<WithEntity<&mut ShipState>>().changed_since(last_acked)` hands
/// each changed row's entity to the encoder.
#[derive(Debug)]
pub struct WithEntity<Q>(core::marker::PhantomData<Q>);

impl<Q: QueryTerms> QueryTerms for WithEntity<Q> {
    type Item<'a> = (Q::Item<'a>, Entity);
    const MUTABLE: bool = Q::MUTABLE;
    fn ids() -> Vec<ComponentId> {
        Q::ids()
    }
    fn run<'a, F>(
        cols: &'a mut [Column],
        entities: &'a [Entity],
        idx: &[usize],
        len: usize,
        mask: Option<&[bool]>,
        f: &mut F,
    ) where
        F: for<'x> FnMut((Q::Item<'x>, Entity)),
    {
        // The entity of a row is archetype metadata, not a column, so `Q::run`
        // cannot hand it out — but it does not have to. Every `run` visits
        // exactly the unmasked rows, in ascending row order, so the same walk
        // over the row indices names the row each item belongs to. That keeps
        // the inner term's own loop (columns borrowed and downcast **once** per
        // archetype) instead of a `Q::get_row` per row, which re-split and
        // re-downcast the columns for every single row.
        let mut rows = (0..len).filter(|&i| mask.is_none_or(|m| m[i]));
        Q::run(cols, entities, idx, len, mask, &mut |item| {
            let row = rows
                .next()
                .expect("QueryTerms::run visits exactly the unmasked rows, in order");
            f((item, entities[row]));
        });
    }
    fn get_row<'a>(
        cols: &'a mut [Column],
        entities: &'a [Entity],
        idx: &[usize],
        row: usize,
    ) -> Self::Item<'a> {
        (Q::get_row(cols, entities, idx, row), entities[row])
    }
}

impl<A: Access, B: Access> QueryTerms for (A, B) {
    type Item<'a> = (A::Ref<'a>, B::Ref<'a>);
    const MUTABLE: bool = A::MUTABLE || B::MUTABLE;
    fn ids() -> Vec<ComponentId> {
        vec![A::Component::ID, B::Component::ID]
    }
    fn run<'a, F>(
        cols: &'a mut [Column],
        _entities: &'a [Entity],
        idx: &[usize],
        len: usize,
        mask: Option<&[bool]>,
        f: &mut F,
    ) where
        F: for<'x> FnMut((A::Ref<'x>, B::Ref<'x>)),
    {
        let [ca, cb] = get_many_mut(cols, [idx[0], idx[1]])
            .expect("query terms must reference distinct components");
        let mut sa = A::column(ca);
        let mut sb = B::column(cb);
        for i in 0..len {
            if mask.is_none_or(|m| m[i]) {
                f((
                    A::row(A::reborrow(&mut sa), i),
                    B::row(B::reborrow(&mut sb), i),
                ));
            }
        }
    }
    fn get_row<'a>(
        cols: &'a mut [Column],
        _entities: &'a [Entity],
        idx: &[usize],
        row: usize,
    ) -> Self::Item<'a> {
        let [ca, cb] = get_many_mut(cols, [idx[0], idx[1]])
            .expect("query terms must reference distinct components");
        (A::get(ca, row), B::get(cb, row))
    }
}

impl<A: Access, B: Access, C: Access> QueryTerms for (A, B, C) {
    type Item<'a> = (A::Ref<'a>, B::Ref<'a>, C::Ref<'a>);
    const MUTABLE: bool = A::MUTABLE || B::MUTABLE || C::MUTABLE;
    fn ids() -> Vec<ComponentId> {
        vec![A::Component::ID, B::Component::ID, C::Component::ID]
    }
    fn run<'a, F>(
        cols: &'a mut [Column],
        _entities: &'a [Entity],
        idx: &[usize],
        len: usize,
        mask: Option<&[bool]>,
        f: &mut F,
    ) where
        F: for<'x> FnMut((A::Ref<'x>, B::Ref<'x>, C::Ref<'x>)),
    {
        let [ca, cb, cc] = get_many_mut(cols, [idx[0], idx[1], idx[2]])
            .expect("query terms must reference distinct components");
        let mut sa = A::column(ca);
        let mut sb = B::column(cb);
        let mut sc = C::column(cc);
        for i in 0..len {
            if mask.is_none_or(|m| m[i]) {
                f((
                    A::row(A::reborrow(&mut sa), i),
                    B::row(B::reborrow(&mut sb), i),
                    C::row(C::reborrow(&mut sc), i),
                ));
            }
        }
    }
    fn get_row<'a>(
        cols: &'a mut [Column],
        _entities: &'a [Entity],
        idx: &[usize],
        row: usize,
    ) -> Self::Item<'a> {
        let [ca, cb, cc] = get_many_mut(cols, [idx[0], idx[1], idx[2]])
            .expect("query terms must reference distinct components");
        (A::get(ca, row), B::get(cb, row), C::get(cc, row))
    }
}

/// A typed, single-threaded iterator over the rows matching term type `Q`.
///
/// Built by [`World::query`](crate::World::query). The query borrows the world's
/// archetype list mutably for its whole lifetime (so `&mut` terms are sound);
/// drop the query before touching the world again.
///
/// # Examples
///
/// ```
/// # use space_ecs::{register_components, ComponentDef, Component, World};
/// # #[derive(Clone)] struct P(f32); #[derive(Clone)] struct V(f32);
/// # impl ComponentDef for P { const ID: u16 = 0; type Data = Self; }
/// # impl ComponentDef for V { const ID: u16 = 1; type Data = Self; }
/// # register_components!((0, P), (1, V));
/// let mut w = World::with_registry(&COMPONENT_REGISTRY);
/// let _e = w.spawn(&[Component::new(&P(1.0)), Component::new(&V(2.0))]);
/// let mut q = w.query::<(&P, &V)>();
/// let mut sum = 0.0;
/// q.for_each(|(p, v)| sum += p.0 + v.0);
/// assert_eq!(sum, 3.0);
/// ```
pub struct Query<'w, Q: QueryTerms> {
    archetypes: &'w mut [Archetype],
    /// Per-archetype column indices for the term's ids; `None` if the archetype
    /// lacks a term (then the archetype is skipped).
    plans: Vec<Option<Vec<usize>>>,
    /// Rows whose change tick must be `> this` to be visited (`None` = all).
    changed_after: Option<u64>,
    /// The world tick stamped onto rows a `&mut` query visits.
    tick: u64,
    /// Scratch for the per-archetype visit mask of a `changed_since` query,
    /// reused across archetypes (and across `for_each` calls) so the filter
    /// costs one buffer per query instead of one allocation per archetype.
    mask: Vec<bool>,
    // lending-iterator cursor
    arch: usize,
    row: usize,
    _q: core::marker::PhantomData<Q>,
}

impl<Q: QueryTerms> std::fmt::Debug for Query<'_, Q> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Query")
            .field("archetypes", &self.archetypes.len())
            .field("changed_after", &self.changed_after)
            .finish()
    }
}

impl<'w, Q: QueryTerms> Query<'w, Q> {
    pub(crate) fn new(archetypes: &'w mut [Archetype], tick: u64) -> Self {
        let ids = Q::ids();
        let plans = archetypes
            .iter()
            .map(|a| {
                let mut idx = Vec::with_capacity(ids.len());
                for &id in &ids {
                    match a.column_of(id) {
                        Some(c) => idx.push(c),
                        None => return None,
                    }
                }
                Some(idx)
            })
            .collect();
        Self {
            archetypes,
            plans,
            changed_after: None,
            tick,
            mask: Vec::new(),
            arch: 0,
            row: 0,
            _q: core::marker::PhantomData,
        }
    }

    /// Keeps only rows whose change tick is greater than `tick`, i.e. written
    /// after `tick`. This is the delta-replication filter: call it once per
    /// network tick with the last acknowledged tick.
    ///
    /// The filter is a statement about the *row* (see
    /// [`World::row_tick`](crate::World::row_tick)): any write to any component
    /// of the entity since `tick` keeps the whole row in the result, which is
    /// what a delta wants — it ships the entity's full replicated component set.
    #[must_use]
    pub fn changed_since(mut self, tick: u64) -> Self {
        self.changed_after = Some(tick);
        self
    }

    /// Calls `f` once per matching row, resetting the cursor first.
    ///
    /// The closure receives a row of borrowed components, not the `World`, so a
    /// parallel scheduler may split the archetype list across workers. If the
    /// query has a `&mut` term, every visited row is stamped with the world's
    /// current tick (this is what makes `changed_since` see the write).
    pub fn for_each<E>(&mut self, mut f: E)
    where
        E: for<'x> FnMut(Q::Item<'x>),
    {
        // Borrow the fields separately: the plans and the mask scratch are read
        // while one archetype is borrowed mutably, so nothing has to be cloned.
        let Self {
            archetypes,
            plans,
            changed_after,
            tick,
            mask: scratch,
            ..
        } = self;
        for (arch, plan) in archetypes.iter_mut().zip(plans.iter()) {
            let Some(plan) = plan.as_deref() else {
                continue;
            };
            let len = arch.len();
            // Build the visit mask from the change filter (the immutable borrow
            // of `ticks` ends before the columns are borrowed mutably).
            let mask: Option<&[bool]> = match *changed_after {
                Some(after) => {
                    scratch.clear();
                    scratch.extend(arch.ticks().iter().map(|&t| t > after));
                    Some(scratch.as_slice())
                }
                None => None,
            };
            let (cols, entities) = arch.columns_and_entities_mut();
            Q::run(cols, entities, plan, len, mask, &mut |item| f(item));
            if Q::MUTABLE {
                for i in 0..len {
                    if mask.is_none_or(|m| m[i]) {
                        arch.set_tick(i, *tick);
                    }
                }
            }
        }
    }

    /// Lending iterator: returns the next matching row, or `None` when
    /// exhausted.
    #[allow(
        clippy::should_implement_trait,
        reason = "a lending iterator cannot implement std::iter::Iterator without unsafe; the brief mandates `next()`"
    )]
    ///
    /// This is a *lending* iterator (the returned item borrows `self`), not
    /// `std::iter::Iterator` — a standard `Iterator` cannot hand out a borrow
    /// tied to `&mut self` without `unsafe`. Drop the returned item before
    /// calling `next` again. A `&mut` query stamps the visited row with the
    /// world's current tick before returning it.
    pub fn next(&mut self) -> Option<Q::Item<'_>> {
        // Find the next matching row first, touching only the cursor; the
        // borrow handed back to the caller is taken once, at the end.
        let (arch_i, row) = loop {
            if self.arch >= self.archetypes.len() {
                return None;
            }
            if self.plans[self.arch].is_none() {
                self.arch += 1;
                self.row = 0;
                continue;
            }
            let ticks = self.archetypes[self.arch].ticks();
            // Advance to the next row passing the change filter.
            if let Some(after) = self.changed_after {
                while self.row < ticks.len() && ticks[self.row] <= after {
                    self.row += 1;
                }
            }
            if self.row >= ticks.len() {
                self.arch += 1;
                self.row = 0;
                continue;
            }
            let found = (self.arch, self.row);
            self.row += 1;
            break found;
        };
        // The plan is borrowed, not cloned: it lives in a different field than
        // the archetype the item borrows from.
        let plan = self.plans[arch_i].as_deref()?;
        let arch = &mut self.archetypes[arch_i];
        if Q::MUTABLE {
            arch.set_tick(row, self.tick);
        }
        let (cols, entities) = arch.columns_and_entities_mut();
        Some(Q::get_row(cols, entities, plan, row))
    }
}

#[cfg(test)]
mod tests {
    use super::get_many_mut;

    /// The references come back in the order the indices were given (a query's
    /// terms are not in column order), and they really are disjoint borrows.
    #[test]
    fn get_many_mut_keeps_the_requested_order() {
        let mut v = [10, 20, 30, 40];
        let [c, a] = get_many_mut(&mut v, [2, 0]).expect("distinct, in range");
        assert_eq!((*c, *a), (30, 10));
        *c += 1;
        *a += 2;
        assert_eq!(v, [12, 20, 31, 40]);

        let [x, y, z] = get_many_mut(&mut v, [3, 1, 2]).expect("distinct, in range");
        std::mem::swap(x, y);
        *z = 0;
        assert_eq!(v, [12, 40, 0, 20]);

        // No indices at all is trivially fine.
        let [] = get_many_mut(&mut v, []).expect("nothing requested");
    }

    /// A duplicated or out-of-range index yields `None`: never two `&mut` to
    /// one element, never a panic.
    #[test]
    fn get_many_mut_rejects_duplicates_and_out_of_range() {
        let mut v = [1, 2, 3];
        assert!(get_many_mut(&mut v, [1, 1]).is_none());
        assert!(get_many_mut(&mut v, [0, 2, 0]).is_none());
        assert!(get_many_mut(&mut v, [0, 3]).is_none());
        assert!(get_many_mut(&mut v, [usize::MAX]).is_none());
        let mut empty: [u8; 0] = [];
        assert!(get_many_mut(&mut empty, [0]).is_none());
        assert_eq!(v, [1, 2, 3]);
    }
}
