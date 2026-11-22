//! Component definitions, the replication registry and the `Component` value.
//!
//! A component type is wired into the ECS by implementing [`ComponentDef`]: a
//! fixed [`ComponentId`] plus the data type stored in the archetype columns.
//! [`register_components`](crate::register_components) then builds the static registry that the world and
//! later crates (protocol/sim) read to learn every component's name, size and
//! whether it is replicated.

use std::any::{Any, TypeId};
use std::fmt;

/// Identifier of a component type inside one game.
///
/// Ids are chosen by the game and must be unique and below [`MAX_COMPONENTS`];
/// [`register_components`](crate::register_components) enforces both at compile time.
pub type ComponentId = u16;

/// Largest number of distinct components the registry can hold.
pub const MAX_COMPONENTS: usize = 4096;

/// Marker for the data stored in an archetype column.
///
/// `Clone` is required because migrating an entity to another archetype copies
/// its row; `'static` + `Send` because columns own their data and a future
/// parallel scheduler must be able to move archetypes between threads.
pub trait ComponentData: Clone + Send + 'static {}
impl<T: Clone + Send + 'static> ComponentData for T {}

/// Associates a component data type with a fixed [`ComponentId`].
///
/// Implement it by hand (no derive in v1) for every component:
///
/// ```
/// use space_ecs::ComponentDef;
///
/// #[derive(Clone, Debug, PartialEq)]
/// struct Health(f32);
///
/// impl ComponentDef for Health {
///     const ID: u16 = 3;
///     type Data = Self;
/// }
///
/// assert_eq!(Health::ID, 3);
/// ```
pub trait ComponentDef {
    /// Fixed id of this component; unique across the game and
    /// `< `[`MAX_COMPONENTS`]`.`
    const ID: ComponentId;
    /// The value stored in archetype columns. Usually `Self`.
    type Data: ComponentData;
    /// Human-readable name used in the registry and in mismatch messages.
    /// [`register_components`](crate::register_components) overrides this with the plain source name; the
    /// default is the empty string (a hand-written impl that wants a name should
    /// set it explicitly).
    const NAME: &'static str = "";
}

/// Static description of one registered component — a replication-registry
/// entry.
///
/// Later crates consume this: `name` and `size` describe the payload,
/// `replicated` says whether it is sent over the network, and `type_id` lets a
/// mismatched access be rejected (see the `roundtrip_mutation` test).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ComponentInfo {
    /// The component's id.
    pub id: ComponentId,
    /// Type name as written in [`register_components`](crate::register_components).
    pub name: &'static str,
    /// [`std::mem::size_of`] of the stored data.
    pub size: usize,
    /// Whether the component is replicated to clients.
    pub replicated: bool,
    /// [`TypeId`] of the stored data, used to reject mismatched access.
    pub type_id: TypeId,
}

impl ComponentInfo {
    /// Builds an entry for `T`. `const` so [`register_components`](crate::register_components) can build
    /// the whole registry at compile time.
    pub const fn new<T: ComponentDef>(name: &'static str, replicated: bool) -> Self {
        Self {
            id: T::ID,
            name,
            size: size_of::<T::Data>(),
            replicated,
            type_id: TypeId::of::<T::Data>(),
        }
    }

    /// The empty registry (all ids unregistered). A crate that never calls
    /// [`register_components`](crate::register_components) uses this; it is a `const`, so there is no
    /// global *mutable* state.
    pub const EMPTY_REGISTRY: [Option<ComponentInfo>; MAX_COMPONENTS] = [None; MAX_COMPONENTS];

    /// Looks up `id` in `registry`.
    ///
    /// Returns `None` for an unregistered id *and* for an id at or beyond
    /// [`MAX_COMPONENTS`] (a `ComponentId` is a `u16`, so a hostile wire id can
    /// exceed the registry length): the lookup never panics on network input.
    #[inline]
    pub fn get(
        registry: &[Option<ComponentInfo>; MAX_COMPONENTS],
        id: ComponentId,
    ) -> Option<Self> {
        registry.get(usize::from(id)).copied().flatten()
    }

    /// `true` if any two entries share an id (used by the macro's compile-time
    /// duplicate check).
    pub const fn has_duplicate_ids(entries: &[Self]) -> bool {
        let mut i = 0;
        while i < entries.len() {
            let mut j = i + 1;
            while j < entries.len() {
                if entries[i].id == entries[j].id {
                    return true;
                }
                j += 1;
            }
            i += 1;
        }
        false
    }

    /// Spreads `entries` into a lookup array indexed by [`ComponentId`].
    ///
    /// # Panics
    /// If an entry's id is `>= `[`MAX_COMPONENTS`]` (also checked by the macro).
    pub const fn build_registry(entries: &[Self]) -> [Option<Self>; MAX_COMPONENTS] {
        let mut out = [None; MAX_COMPONENTS];
        let mut i = 0;
        while i < entries.len() {
            let info = &entries[i];
            out[info.id as usize] = Some(*info);
            i += 1;
        }
        out
    }
}

impl fmt::Display for ComponentInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}(id {}, {} bytes, {})",
            self.name,
            self.id,
            self.size,
            if self.replicated {
                "replicated"
            } else {
                "local"
            }
        )
    }
}

/// A type-erased component *value*, as passed to
/// [`World::spawn`](crate::World::spawn) and [`World::insert`](crate::World::insert).
///
/// It borrows the data, so the value only has to outlive the call:
///
/// ```
/// # use space_ecs::{Component, ComponentDef, World};
/// # #[derive(Clone)] struct Hp(f32);
/// # impl ComponentDef for Hp { const ID: u16 = 100; type Data = Self; }
/// let hp = Hp(90.0);
/// let mut world = World::new();
/// let e = world.spawn(&[Component::new(&hp)]);
/// assert_eq!(world.get::<Hp>(e).unwrap().0, 90.0);
/// ```
pub struct Component<'a> {
    id: ComponentId,
    type_id: TypeId,
    data: &'a (dyn Any + Send),
    // Builds an empty column for this component's data type; captured from the
    // concrete `T` so the world never needs the registry to allocate storage.
    new_column: fn() -> crate::storage::Column,
    // Clones the payload into an owned box; captured from the concrete `T`
    // (`T: Clone`) because `dyn Any` has no `clone_box`.
    clone_data: fn(&(dyn Any + Send)) -> Box<dyn Any + Send>,
}

impl std::fmt::Debug for Component<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Component")
            .field("id", &self.id)
            .field("type_id", &self.type_id)
            .finish()
    }
}

impl<'a> Component<'a> {
    /// Wraps a component value.
    pub fn new<T>(value: &'a T) -> Self
    where
        T: ComponentDef<Data = T> + ComponentData,
    {
        fn mk<T: ComponentDef<Data = T> + ComponentData>() -> crate::storage::Column {
            crate::storage::Column::new::<T>()
        }
        fn clone_data<T: ComponentDef<Data = T> + ComponentData>(
            v: &(dyn Any + Send),
        ) -> Box<dyn Any + Send> {
            Box::new(v.downcast_ref::<T>().unwrap().clone())
        }
        Self {
            id: T::ID,
            type_id: TypeId::of::<T::Data>(),
            data: value,
            new_column: mk::<T>,
            clone_data: clone_data::<T>,
        }
    }

    /// Id of the component.
    #[inline]
    pub fn id(&self) -> ComponentId {
        self.id
    }

    /// [`TypeId`] of the stored data.
    #[inline]
    pub fn type_id(&self) -> TypeId {
        self.type_id
    }

    /// The borrowed payload, type-erased.
    #[inline]
    pub fn data(&self) -> &(dyn Any + Send) {
        self.data
    }

    /// The column factory for this component's data type.
    #[inline]
    pub(crate) fn column_factory(&self) -> fn() -> crate::storage::Column {
        self.new_column
    }

    /// Downcasts the payload to `T`.
    ///
    /// # Panics
    /// If `T::Data` is not the type this component was built from.
    pub fn downcast_ref<T: ComponentDef<Data = T> + 'static>(&self) -> &T {
        self.data.downcast_ref::<T>().unwrap_or_else(|| {
            panic!(
                "component downcast failed: requested {}, component is id {}",
                T::NAME,
                self.id
            )
        })
    }
}

/// An owned, type-erased component value — the form a
/// [`CommandBuffer`](crate::CommandBuffer) stores so deferred edits can outlive
/// the values they were built from.
///
/// Build one with [`Component::to_owned_component`] or [`OwnedComponent::new`].
pub struct OwnedComponent {
    id: ComponentId,
    type_id: TypeId,
    data: Box<dyn Any + Send>,
    new_column: fn() -> crate::storage::Column,
    clone_data: fn(&(dyn Any + Send)) -> Box<dyn Any + Send>,
}

impl std::fmt::Debug for OwnedComponent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OwnedComponent")
            .field("id", &self.id)
            .field("type_id", &self.type_id)
            .finish()
    }
}

impl OwnedComponent {
    /// Wraps an owned value.
    pub fn new<T>(value: T) -> Self
    where
        T: ComponentDef<Data = T> + ComponentData,
    {
        fn mk<T: ComponentDef<Data = T> + ComponentData>() -> crate::storage::Column {
            crate::storage::Column::new::<T>()
        }
        fn clone_data<T: ComponentDef<Data = T> + ComponentData>(
            v: &(dyn Any + Send),
        ) -> Box<dyn Any + Send> {
            Box::new(v.downcast_ref::<T>().unwrap().clone())
        }
        Self {
            id: T::ID,
            type_id: TypeId::of::<T::Data>(),
            data: Box::new(value),
            new_column: mk::<T>,
            clone_data: clone_data::<T>,
        }
    }

    /// Id of the component.
    #[inline]
    pub fn id(&self) -> ComponentId {
        self.id
    }

    /// [`TypeId`] of the stored data.
    #[inline]
    pub fn type_id(&self) -> TypeId {
        self.type_id
    }

    /// The owned payload, type-erased.
    #[inline]
    pub fn data(&self) -> &(dyn Any + Send) {
        &*self.data
    }

    /// Borrows the payload as a [`Component`], for passing to
    /// [`World::spawn`](crate::World::spawn) / [`World::insert`](crate::World::insert).
    pub fn as_component(&self) -> Component<'_> {
        Component {
            id: self.id,
            type_id: self.type_id,
            data: &*self.data,
            new_column: self.new_column,
            clone_data: self.clone_data,
        }
    }
}

impl<'a> Component<'a> {
    /// Clones the borrowed payload into an [`OwnedComponent`].
    pub fn to_owned_component(&self) -> OwnedComponent {
        OwnedComponent {
            id: self.id,
            type_id: self.type_id,
            data: (self.clone_data)(self.data),
            new_column: self.new_column,
            clone_data: self.clone_data,
        }
    }
}
