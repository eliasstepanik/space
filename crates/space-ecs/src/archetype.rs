//! Archetypes: a component set plus one [`Column`] per component.

use crate::component::ComponentId;
use crate::entity::Entity;
use crate::storage::Column;

/// One archetype: a sorted component set with a column per component.
///
/// Entities with the identical component set share an archetype, so every row
/// has the same columns and a query over that set is a plain SoA scan. Columns
/// are kept sorted by [`ComponentId`] so a column lookup is a binary search.
///
/// The change-tick column (one `u64` per row, written by every mutation) and
/// the row→[`Entity`] map are internal and are *not* part of
/// the component set.
#[derive(Debug)]
pub struct Archetype {
    /// Sorted, deduplicated component ids of this archetype.
    components: Vec<ComponentId>,
    /// One column per entry of `components`, same order.
    columns: Vec<Column>,
    /// Per-row tick of the last write, parallel to the columns.
    ///
    /// `u64` so it can carry a `space_sim::SimTick` unchanged (see
    /// [`World::tick`](crate::World::tick)).
    ticks: Vec<u64>,
    /// The entity stored in each row (used to fix up locations on swap-remove).
    entities: Vec<Entity>,
}

impl Archetype {
    /// Creates an archetype for `components` with one column per component.
    ///
    /// # Panics
    /// If `columns` does not hold exactly one column per *distinct* component
    /// id. This is checked in every build: an archetype with a surplus column
    /// looks fine until a row is removed, when the never-filled column panics
    /// half-way through the removal.
    pub(crate) fn new(components: &[ComponentId], columns: Vec<Column>) -> Self {
        let mut components = components.to_vec();
        components.sort_unstable();
        components.dedup();
        assert_eq!(
            components.len(),
            columns.len(),
            "an archetype needs exactly one column per distinct component id"
        );
        Self {
            components,
            columns,
            ticks: Vec::new(),
            entities: Vec::new(),
        }
    }

    /// The sorted component set.
    #[inline]
    pub(crate) fn components(&self) -> &[ComponentId] {
        &self.components
    }

    /// Number of rows.
    #[inline]
    pub fn len(&self) -> usize {
        self.entities.len()
    }

    /// `true` when the archetype holds no entities.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entities.is_empty()
    }

    /// Column index for `id`, if present.
    #[inline]
    pub(crate) fn column_of(&self, id: ComponentId) -> Option<usize> {
        self.components.binary_search(&id).ok()
    }

    /// Shared view of a column.
    #[inline]
    pub(crate) fn column(&self, i: usize) -> &Column {
        &self.columns[i]
    }

    /// Mutable view of a column.
    #[inline]
    pub(crate) fn column_mut(&mut self, i: usize) -> &mut Column {
        &mut self.columns[i]
    }

    /// The columns mutably and the row→[`Entity`] map shared, in one borrow.
    ///
    /// Needed by [`WithEntity`](crate::WithEntity), which reads the entity of a
    /// row while borrowing that row's components. Rust allows it because they
    /// are separate fields.
    #[inline]
    pub(crate) fn columns_and_entities_mut(&mut self) -> (&mut [Column], &[Entity]) {
        (&mut self.columns, &self.entities)
    }

    /// Records the meta (entity + tick) for a row that the caller is about to
    /// fill column-by-column. Returns the new row index.
    pub(crate) fn reserve_row(&mut self, entity: Entity, tick: u64) -> usize {
        self.push_meta(entity, tick)
    }

    /// Appends a row for `entity` with the given tick.
    ///
    /// The caller must have already pushed a value onto every column (in column
    /// order); this only records the entity and its tick.
    pub(crate) fn push_meta(&mut self, entity: Entity, tick: u64) -> usize {
        self.entities.push(entity);
        self.ticks.push(tick);
        self.entities.len() - 1
    }

    /// Removes `row` with swap-remove semantics, returning the entity that was
    /// displaced into `row` (if any) so the caller can fix its location.
    pub(crate) fn swap_remove(&mut self, row: usize) -> Option<Entity> {
        for col in &mut self.columns {
            col.remove_row(row);
        }
        self.ticks.swap_remove(row);
        // The value that was last moves into `row`; report it so the caller can
        // point its location at the new row.
        self.entities.swap_remove(row);
        if row < self.entities.len() {
            Some(self.entities[row])
        } else {
            None
        }
    }

    /// Stamps `row` with `tick`.
    #[inline]
    pub(crate) fn set_tick(&mut self, row: usize, tick: u64) {
        self.ticks[row] = tick;
    }

    /// The `ticks` slice (for `changed_since` mask building and
    /// [`World::row_tick`](crate::World::row_tick)).
    #[inline]
    pub(crate) fn ticks(&self) -> &[u64] {
        &self.ticks
    }
}
