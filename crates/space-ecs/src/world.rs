//! The [`World`] container.

use std::any::TypeId;
use std::collections::HashMap;

use crate::archetype::Archetype;
use crate::command::CommandBuffer;
use crate::component::{Component, ComponentData, ComponentDef, ComponentId, ComponentInfo};
use crate::entity::Entity;
use crate::query::Query;
use crate::query::QueryTerms;
use crate::stage::Stage;
use crate::storage::Column;

/// Where an entity currently lives.
#[derive(Clone, Copy, Debug)]
struct Location {
    archetype: usize,
    row: usize,
    generation: u32,
    alive: bool,
}

/// The ECS container: entities, archetypes, the change tick and the systems.
///
/// A `World` owns every entity and component value. It is created empty with
/// [`World::new`] (or [`World::with_registry`] to attach a
/// [`register_components`](crate::register_components) registry for type-mismatch diagnostics).
///
/// # Component storage
///
/// Each distinct component set is an [`Archetype`] with one [`Column`] per
/// component. Spawning inserts into the archetype for the spawn's component
/// set; [`World::insert`] / [`World::remove`] migrate the row to the neighbour
/// archetype.
#[derive(Debug)]
pub struct World {
    /// One slot per entity index ever used; index == `Entity::index()`.
    locations: Vec<Location>,
    /// Free entity indices (reused on spawn, generation already bumped).
    free: Vec<u32>,
    /// The archetypes, in creation order.
    archetypes: Vec<Archetype>,
    /// id -> a freshly-built empty column, so archetypes can be created lazily.
    column_factories: HashMap<ComponentId, fn() -> Column>,
    /// The component registry used for mismatch diagnostics (may be empty).
    registry: &'static [Option<ComponentInfo>; crate::component::MAX_COMPONENTS],
    /// Monotonic change tick, advanced by [`World::tick`].
    ///
    /// It is a `u64` so it can be driven straight from a `space_sim::SimTick`
    /// (also `u64`) without a wrap policy — see [`World::tick`].
    tick: u64,
}

impl Default for World {
    fn default() -> Self {
        Self::new()
    }
}

impl World {
    /// An empty world with no registry attached.
    pub fn new() -> Self {
        Self::with_registry(&[None; crate::component::MAX_COMPONENTS])
    }

    /// An empty world that reports component names from `registry` in
    /// type-mismatch panics. The registry is normally the `COMPONENT_REGISTRY`
    /// built by [`register_components`](crate::register_components).
    ///
    /// The `'static` requirement exists because the world keeps the reference;
    /// a `const`/`static` registry satisfies it. For a locally-scoped registry,
    /// leak it once at startup or use [`World::new`].
    pub fn with_registry(
        registry: &'static [Option<ComponentInfo>; crate::component::MAX_COMPONENTS],
    ) -> Self {
        Self {
            locations: Vec::new(),
            free: Vec::new(),
            archetypes: Vec::new(),
            column_factories: HashMap::new(),
            registry,
            tick: 0,
        }
    }

    /// The current change tick.
    #[inline]
    pub fn current_tick(&self) -> u64 {
        self.tick
    }

    /// Advances the change tick by one and returns the new value. Systems call
    /// this once per simulation step so `changed_since` can distinguish writes
    /// across ticks.
    ///
    /// The tick is a `u64` (not `u32`) so a caller whose simulation clock is a
    /// `u64` (e.g. `space_sim::SimWorld`, whose `SimTick` is `u64`) can drive it
    /// directly with no truncation and no wrap policy.
    ///
    /// # Tick ordering (the contract replication relies on)
    ///
    /// A tick is *open* between two `tick()` calls: every write in that window —
    /// `spawn`, `insert`, `remove`, `get_mut` and `&mut` queries — is stamped
    /// with the tick that `tick()` last returned. The contract a step must keep
    /// is: **do all of the step's writes, then call `tick()` last** (see
    /// `space_sim::SimWorld::step`, which advances the ECS tick at the very end
    /// of the step, together with the `SimTick`).
    ///
    /// The consequence for replication: a pass runs **after** the step that
    /// produced the tick it is reporting on and filters with
    /// `changed_since(last_acked)`. Because the step's `tick()` call is what
    /// closes the window, every write the step made is stamped with the tick the
    /// step produced, and a write that lands *between* two steps is stamped with
    /// the *next* step's tick — a tick no delta has been sent for yet — so it can
    /// never be missed. (Advancing the tick at the *start* of a step, as earlier
    /// revisions did, stamps such a between-steps write with a tick a delta was
    /// already sent for, which loses it forever.)
    #[inline]
    pub fn tick(&mut self) -> u64 {
        self.tick = self.tick.wrapping_add(1);
        self.tick
    }

    /// The tick `entity`'s row was last stamped with, or `None` if the entity is
    /// dead.
    ///
    /// This is the "which entity changed" half of the change-tick model: a
    /// replication pass iterates a `changed_since` query together with the
    /// [`Entity`](crate::Entity) term and reads the stamp back here when it wants
    /// to log or debug *why* a row was picked.
    ///
    /// # Per-row, not per-component
    ///
    /// The stamp lives on the *row*, not on the component: one write to any
    /// component of an entity stamps the whole row, and a `changed_since` query
    /// reports the row (all of its components), not the component that moved.
    /// That is deliberate — a delta always ships an entity's full replicated
    /// component set, so a per-component stamp would only add bookkeeping — and
    /// it means "changed since `t`" is a statement about the entity, not about
    /// one of its fields.
    #[inline]
    pub fn row_tick(&self, entity: Entity) -> Option<u64> {
        let loc = self.location(entity)?;
        Some(self.archetypes[loc.archetype].ticks()[loc.row])
    }

    /// Stamps `entity`'s row with `tick` without touching any component value.
    ///
    /// Returns `false` if the entity is dead. This is the escape hatch for a
    /// system that *did* write through a raw pointer or an outside channel and
    /// needs the row marked; ordinary writes do not need it because
    /// [`World::get_mut`] and every `&mut` query term stamp automatically.
    #[inline]
    pub fn mark_changed(&mut self, entity: Entity) -> bool {
        match self.location(entity) {
            Some(loc) => {
                let tick = self.tick;
                self.archetypes[loc.archetype].set_tick(loc.row, tick);
                true
            }
            None => false,
        }
    }

    /// Stamps `entity`'s row with `tick` explicitly, bypassing the world tick.
    ///
    /// Returns `false` if the entity is dead. Used by a replication pass that
    /// replays a write it knows happened at a specific past tick (e.g. a
    /// handoff that landed a state captured at `activation_tick`).
    pub fn stamp_at(&mut self, entity: Entity, tick: u64) -> bool {
        match self.location(entity) {
            Some(loc) => {
                self.archetypes[loc.archetype].set_tick(loc.row, tick);
                true
            }
            None => false,
        }
    }

    /// Sets the change tick to `tick` outright.
    ///
    /// A world that joins a session mid-run (a handoff receiver, a client
    /// mirror) must start its change tick where the session's clock is, or a
    /// `changed_since` filter would see every row as ancient. Prefer this over
    /// calling [`World::tick`] in a loop.
    #[inline]
    pub fn set_tick(&mut self, tick: u64) {
        self.tick = tick;
    }

    /// Number of live entities.
    pub fn len(&self) -> usize {
        self.locations.iter().filter(|l| l.alive).count()
    }

    /// `true` when no entity is alive.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    // ----------------------------------------------------------------- spawn

    /// Spawns an entity with `components` and returns its handle.
    ///
    /// Duplicate component ids in `components` are a logic error and panic.
    ///
    /// ```
    /// # use space_ecs::{Component, ComponentDef, World};
    /// # #[derive(Clone)] struct A(u8); impl ComponentDef for A { const ID: u16 = 1; type Data = Self; }
    /// let mut w = World::new();
    /// let e = w.spawn(&[Component::new(&A(5))]);
    /// assert_eq!(w.get::<A>(e).unwrap().0, 5);
    /// ```
    pub fn spawn(&mut self, components: &[Component<'_>]) -> Entity {
        let index = match self.free.pop() {
            Some(i) => i,
            None => {
                self.locations.push(Location {
                    archetype: usize::MAX,
                    row: usize::MAX,
                    generation: 0,
                    alive: false,
                });
                (self.locations.len() - 1) as u32
            }
        };
        let generation = self.locations[index as usize].generation;
        let entity = Entity::new(index, generation);

        // Build the sorted component set and register column factories.
        let mut ids: Vec<ComponentId> = Vec::with_capacity(components.len());
        for c in components {
            self.column_factories
                .entry(c.id())
                .or_insert(c.column_factory());
            ids.push(c.id());
        }
        ids.sort_unstable();
        debug_assert!(
            ids.windows(2).all(|w| w[0] != w[1]),
            "spawn called with duplicate component ids"
        );

        let arch = self.archetype_for(&ids);
        let tick = self.tick;
        // Push values in column order, then record the row meta.
        {
            let arch = &mut self.archetypes[arch];
            for (col_i, &id) in arch.components().to_vec().iter().enumerate() {
                let comp = components.iter().find(|c| c.id() == id).unwrap();
                arch.column_mut(col_i).push_dyn(comp.data());
            }
        }
        let row = self.archetypes[arch].reserve_row(entity, tick);
        self.locations[index as usize] = Location {
            archetype: arch,
            row,
            generation,
            alive: true,
        };
        entity
    }

    /// Despawns `entity`. A stale or already-dead handle is ignored (returns
    /// `false`).
    pub fn despawn(&mut self, entity: Entity) -> bool {
        if !self.entity_is_alive(entity) {
            return false;
        }
        let loc = self.locations[entity.index() as usize];
        let arch = &mut self.archetypes[loc.archetype];
        if let Some(moved) = arch.swap_remove(loc.row) {
            // The entity that moved into `loc.row` needs its location fixed.
            let m = &mut self.locations[moved.index() as usize];
            m.row = loc.row;
        }
        self.locations[entity.index() as usize] = Location {
            archetype: usize::MAX,
            row: usize::MAX,
            generation: entity.generation().wrapping_add(1),
            alive: false,
        };
        self.free.push(entity.index());
        true
    }

    /// `true` if `entity` is a live handle (right index *and* generation).
    #[inline]
    pub fn entity_is_alive(&self, entity: Entity) -> bool {
        match self.locations.get(entity.index() as usize) {
            Some(loc) => loc.alive && loc.generation == entity.generation(),
            None => false,
        }
    }

    // ------------------------------------------------------------- accessors

    /// Shared access to `entity`'s `T`, or `None` if the entity is dead or lacks
    /// the component.
    ///
    /// # Panics
    /// In debug builds, if the component id `T::ID` is present but holds a
    /// different type than `T::Data` (a registry mismatch — see the
    /// `roundtrip_mutation` test).
    pub fn get<T: ComponentDef<Data = T> + ComponentData>(&self, entity: Entity) -> Option<&T> {
        let loc = self.location(entity)?;
        let arch = &self.archetypes[loc.archetype];
        let col_i = arch.column_of(T::ID)?;
        let col = arch.column(col_i);
        debug_assert_eq!(
            col.type_id(),
            TypeId::of::<T::Data>(),
            "get::<{}>() on entity {entity}: component id {} holds a different type ({})",
            T::NAME,
            T::ID,
            self.describe_id(T::ID),
        );
        Some(&col.slice::<T::Data>()[loc.row])
    }

    /// Mutable access to `entity`'s `T`, stamping the row with the current tick.
    /// Returns `None` if the entity is dead or lacks the component.
    pub fn get_mut<T: ComponentDef<Data = T> + ComponentData>(
        &mut self,
        entity: Entity,
    ) -> Option<&mut T> {
        let loc = self.location(entity)?;
        // The type-mismatch diagnostic allocates a `String`. `get_mut` is the
        // hottest path in the sim (every physics write goes through it), so the
        // whole check — and its allocation — is compiled out entirely in release
        // via `cfg(debug_assertions)` (a plain `debug_assert_eq!` would still
        // type-check and evaluate the format args). It must run before the
        // `&mut` borrow of `archetypes`: the format args would otherwise conflict
        // with `arch`.
        #[cfg(debug_assertions)]
        let mismatch = self.describe_id(T::ID);
        let arch = &mut self.archetypes[loc.archetype];
        let col_i = arch.column_of(T::ID)?;
        #[cfg(debug_assertions)]
        assert_eq!(
            arch.column(col_i).type_id(),
            TypeId::of::<T::Data>(),
            "get_mut::<{}>() on entity {entity}: component id {} holds a different type ({})",
            T::NAME,
            T::ID,
            mismatch,
        );
        let tick = self.tick;
        arch.set_tick(loc.row, tick);
        Some(&mut arch.column_mut(col_i).vec_mut::<T::Data>()[loc.row])
    }

    /// Non-panicking variant of [`World::get`]: identical behaviour, kept as a
    /// separate name so callers can be explicit about expecting a miss.
    #[inline]
    pub fn try_get<T: ComponentDef<Data = T> + ComponentData>(&self, entity: Entity) -> Option<&T> {
        self.get::<T>(entity)
    }

    /// `true` if `entity` carries `T`.
    pub fn contains<T: ComponentDef>(&self, entity: Entity) -> bool {
        match self.location(entity) {
            Some(loc) => self.archetypes[loc.archetype].column_of(T::ID).is_some(),
            None => false,
        }
    }

    /// Inserts `component` into `entity`, migrating it to a new archetype if it
    /// did not already have that component id. Overwrites the value if it did.
    ///
    /// Returns `false` if the entity is dead.
    pub fn insert(&mut self, entity: Entity, component: Component<'_>) -> bool {
        if !self.entity_is_alive(entity) {
            return false;
        }
        self.column_factories
            .entry(component.id())
            .or_insert(component.column_factory());
        let loc = self.locations[entity.index() as usize];
        let had = self.archetypes[loc.archetype].column_of(component.id());
        if let Some(col_i) = had {
            // Overwrite the value in place and stamp the tick.
            let arch = &mut self.archetypes[loc.archetype];
            arch.column_mut(col_i).set(loc.row, component.data());
            let tick = self.tick;
            arch.set_tick(loc.row, tick);
            return true;
        }
        // Migrate to the archetype with the extra component.
        self.migrate(entity, &[(component.id(), component.data())], None);
        true
    }

    /// Removes `T` from `entity`, migrating it to the neighbour archetype.
    /// Returns `true` if a component was removed.
    pub fn remove<T: ComponentDef>(&mut self, entity: Entity) -> bool {
        if !self.entity_is_alive(entity) {
            return false;
        }
        let loc = self.locations[entity.index() as usize];
        if self.archetypes[loc.archetype].column_of(T::ID).is_none() {
            return false;
        }
        self.migrate(entity, &[], Some(T::ID));
        true
    }

    /// Removes the component with the given id from `entity` (type-erased
    /// variant of [`World::remove`], used by [`CommandBuffer`]).
    ///
    /// Returns `false` if the entity is dead or lacks the id.
    pub fn remove_by_id(&mut self, entity: Entity, id: ComponentId) -> bool {
        if !self.entity_is_alive(entity) {
            return false;
        }
        let loc = self.locations[entity.index() as usize];
        if self.archetypes[loc.archetype].column_of(id).is_none() {
            return false;
        }
        self.migrate(entity, &[], Some(id));
        true
    }

    // --------------------------------------------------------------- queries

    /// Builds a typed query over term type `Q` (e.g. `(&Position, &Velocity)`).
    ///
    /// The returned [`Query`](crate::Query) borrows the world mutably; drop it
    /// before using the world again.
    pub fn query<Q: QueryTerms>(&mut self) -> Query<'_, Q> {
        let tick = self.tick;
        Query::new(&mut self.archetypes, tick)
    }

    // ------------------------------------------------------------- scheduler

    /// Applies every recorded command; returns the number of commands that were
    /// dropped because their target entity was dead (see
    /// [`CommandBuffer`]).
    pub fn run(&mut self, commands: &mut CommandBuffer) -> usize {
        commands.apply(self)
    }

    /// Runs each stage's systems in order.
    pub fn run_stages(&mut self, stages: &[Stage]) {
        for stage in stages {
            stage.run(self);
        }
    }

    // -------------------------------------------------------------- internals

    fn location(&self, entity: Entity) -> Option<Location> {
        if self.entity_is_alive(entity) {
            Some(self.locations[entity.index() as usize])
        } else {
            None
        }
    }

    fn describe_id(&self, id: ComponentId) -> String {
        // Bounds-safe lookup: a `ComponentId` is a `u16` and a hostile wire id
        // can exceed the registry length, so go through `ComponentInfo::get`
        // (which returns `None` past the end) instead of indexing directly.
        match crate::ComponentInfo::get(self.registry, id) {
            Some(info) => info.to_string(),
            None => format!("unregistered id {id}"),
        }
    }

    /// Returns the archetype index for `ids`, creating it (and its columns) if
    /// new. `ids` must be sorted and deduplicated.
    fn archetype_for(&mut self, ids: &[ComponentId]) -> usize {
        // Fast path: existing archetype with exactly this set.
        if let Some(i) = self.archetypes.iter().position(|a| a.components() == ids) {
            return i;
        }
        let columns: Vec<Column> = ids
            .iter()
            .map(|id| {
                let factory = self
                    .column_factories
                    .get(id)
                    .copied()
                    .unwrap_or_else(|| panic!("no column factory for component id {id}"));
                factory()
            })
            .collect();
        let index = self.archetypes.len();
        self.archetypes.push(Archetype::new(ids, columns));
        index
    }

    /// Moves `entity` to a neighbouring archetype.
    ///
    /// * `add`: `(id, value)` pairs to add (empty for a pure remove).
    /// * `remove`: an optional id to drop.
    fn migrate(
        &mut self,
        entity: Entity,
        add: &[(ComponentId, &(dyn std::any::Any + Send))],
        remove: Option<ComponentId>,
    ) {
        let loc = self.locations[entity.index() as usize];
        let src = loc.archetype;
        let mut new_ids: Vec<ComponentId> = self.archetypes[src].components().to_vec();
        if let Some(rm) = remove {
            new_ids.retain(|&x| x != rm);
        }
        for &(a, _) in add {
            if !new_ids.contains(&a) {
                new_ids.push(a);
            }
        }
        new_ids.sort_unstable();
        new_ids.dedup();

        let dst = self.archetype_for(&new_ids);
        if dst == src {
            return;
        }
        // Copy every surviving column's row from src to dst. `src` and `dst`
        // are distinct archetypes, so we borrow them disjointly with
        // `get_many_mut` (the same safe helper queries use for columns).
        let tick = self.tick;
        self.archetypes[dst].reserve_row(entity, tick);
        let dst_ids = self.archetypes[dst].components().to_vec();
        // Pre-compute, for each dst column, the src column index (or None for a
        // newly added component).
        let src_cols: Vec<Option<usize>> = dst_ids
            .iter()
            .map(|&id| self.archetypes[src].column_of(id))
            .collect();
        for (di, &id) in dst_ids.iter().enumerate() {
            match src_cols[di] {
                Some(si) => {
                    let [src_arch, dst_arch] =
                        crate::query::get_many_mut(&mut self.archetypes, [src, dst])
                            .expect("src and dst archetypes are distinct");
                    Column::copy_row(src_arch.column(si), loc.row, dst_arch.column_mut(di));
                }
                None => {
                    // Newly added component: take its value from `add`.
                    let (_, value) = add
                        .iter()
                        .find(|(cid, _)| *cid == id)
                        .unwrap_or_else(|| panic!("missing value for added component id {id}"));
                    self.archetypes[dst].column_mut(di).push_dyn(*value);
                }
            }
        }
        // Remove from the source archetype.
        let dst_row = self.archetypes[dst].len() - 1;
        if let Some(moved) = self.archetypes[src].swap_remove(loc.row) {
            let m = &mut self.locations[moved.index() as usize];
            m.row = loc.row;
        }
        self.locations[entity.index() as usize] = Location {
            archetype: dst,
            row: dst_row,
            generation: entity.generation(),
            alive: true,
        };
    }
}
