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

/// Splits `slice` into `N` disjoint mutable references at `indices`.
///
/// Returns `None` if any index is duplicated or out of range. This is the safe
/// stand-in for the (still unstable in Rust 1.100) `slice::get_many_mut`.
pub(crate) fn get_many_mut<T, const N: usize>(
    slice: &mut [T],
    indices: [usize; N],
) -> Option<[&mut T; N]> {
    let mut order: [(usize, usize); N] = std::array::from_fn(|i| (indices[i], i));
    order.sort_unstable();
    for w in order.windows(2) {
        if w[0].0 == w[1].0 {
            return None;
        }
    }
    if let Some(last) = order.last() {
        if last.0 >= slice.len() {
            return None;
        }
    }
    let sorted: Vec<usize> = order.iter().map(|x| x.0).collect();
    let mut out: Vec<&mut T> = Vec::with_capacity(N);
    take_sorted(slice, &sorted, 0, &mut out);
    // `out[j]` corresponds to `order[j]`; result slot `i` is `out[pos_of[i]]`.
    let mut pos_of = [0usize; N];
    for (j, &(_, orig)) in order.iter().enumerate() {
        pos_of[orig] = j;
    }
    let mut slots: Vec<Option<&mut T>> = out.into_iter().map(Some).collect();
    let res: Vec<&mut T> = (0..N).map(|i| slots[pos_of[i]].take().unwrap()).collect();
    res.try_into().ok()
}

/// Hands out `&mut` at the sorted positions `sorted` (all `>= base`).
fn take_sorted<'a, T>(slice: &'a mut [T], sorted: &[usize], base: usize, out: &mut Vec<&'a mut T>) {
    if let Some((&i, rest)) = sorted.split_first() {
        let (_head, tail) = slice.split_at_mut(i - base);
        let (one, tail) = tail.split_at_mut(1);
        out.push(&mut one[0]);
        take_sorted(tail, rest, i + 1, out);
    }
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
        // The entity of a row is archetype metadata, so the row loop is done
        // here (one `Q::get_row` per visited row) rather than in `Q::run`, which
        // borrows whole columns once. The extra per-row column downcast is the
        // price of the entity term; replication passes are bandwidth-bound, not
        // downcast-bound.
        for i in 0..len {
            if mask.is_none_or(|m| m[i]) {
                f((Q::get_row(&mut *cols, entities, idx, i), entities[i]));
            }
        }
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
        for a in 0..self.archetypes.len() {
            let plan = match self.plans[a].clone() {
                Some(p) => p,
                None => continue,
            };
            let arch = &mut self.archetypes[a];
            let len = arch.len();
            // Build the visit mask from the change filter (immutable borrow of
            // `ticks` ends before the columns are borrowed mutably).
            let mask: Option<Vec<bool>> = self
                .changed_after
                .map(|after| (0..len).map(|i| arch.ticks()[i] > after).collect());
            let (cols, entities) = arch.columns_and_entities_mut();
            Q::run(cols, entities, &plan, len, mask.as_deref(), &mut |item| {
                f(item)
            });
            if Q::MUTABLE {
                let tick = self.tick;
                for i in 0..len {
                    if mask.as_ref().is_none_or(|m| m[i]) {
                        arch.set_tick(i, tick);
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
        loop {
            if self.arch >= self.archetypes.len() {
                return None;
            }
            let plan = match &self.plans[self.arch] {
                Some(p) => p.clone(),
                None => {
                    self.arch += 1;
                    self.row = 0;
                    continue;
                }
            };
            let len = self.archetypes[self.arch].len();
            // Advance to the next row passing the change filter.
            while self.row < len {
                let keep = match self.changed_after {
                    None => true,
                    Some(after) => self.archetypes[self.arch].ticks()[self.row] > after,
                };
                if keep {
                    break;
                }
                self.row += 1;
            }
            if self.row >= len {
                self.arch += 1;
                self.row = 0;
                continue;
            }
            let arch = &mut self.archetypes[self.arch];
            if Q::MUTABLE {
                let tick = self.tick;
                arch.set_tick(self.row, tick);
            }
            let (cols, entities) = arch.columns_and_entities_mut();
            let item = Q::get_row(cols, entities, &plan, self.row);
            self.row += 1;
            return Some(item);
        }
    }
}
