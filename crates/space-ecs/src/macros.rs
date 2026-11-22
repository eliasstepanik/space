//! The `register_components!` macro (declared here, re-exported at the crate
//! root).

/// Builds the static component registry from a list of `(ID, Type)` pairs.
///
/// Each entry may add `replicated = <bool>` (default `false`). The macro emits,
/// in the current scope:
///
/// * `COMPONENT_REGISTRY: `[`[Option<ComponentInfo>; MAX_COMPONENTS]`] — the
///   replication table indexed by [`ComponentId`](crate::ComponentId), and
/// * a `const` that fails to compile if two entries share an id.
///
/// The registry is what later crates read (name / size / `replicated`) and what
/// [`World::with_registry`](crate::World::with_registry) hands to queries for
/// type-mismatch diagnostics.
///
/// ```
/// use space_ecs::{register_components, ComponentDef, World};
///
/// #[derive(Clone)]
/// struct Position(f32);
/// #[derive(Clone)]
/// struct Velocity(f32);
///
/// impl ComponentDef for Position {
///     const ID: u16 = 0;
///     type Data = Self;
/// }
/// impl ComponentDef for Velocity {
///     const ID: u16 = 1;
///     type Data = Self;
/// }
///
/// register_components!((0, Position, replicated = true), (1, Velocity));
///
/// assert_eq!(COMPONENT_REGISTRY[0].unwrap().name, "Position");
/// assert!(COMPONENT_REGISTRY[0].unwrap().replicated);
/// assert!(!COMPONENT_REGISTRY[1].unwrap().replicated);
///
/// // The registry drives a World's queries.
/// let mut world = World::with_registry(&COMPONENT_REGISTRY);
/// ```
#[macro_export]
macro_rules! register_components {
    ($(( $id:expr, $ty:ty $(, replicated = $rep:expr)? )),* $(,)?) => {
        /// Number of registered components.
        const __COUNT: usize = [$(
            {
                // Forces `$ty` to be used and yields one element per entry.
                let _ = ::core::any::TypeId::of::<<$ty as $crate::ComponentDef>::Data>();
                let _ = $crate::__replicated!($($rep)?);
                1usize
            }
        ),*].len();

        /// Component entries built by [`register_components!`].
        const ENTRIES: [$crate::ComponentInfo; __COUNT] = [$(
            $crate::ComponentInfo::new::<$ty>(
                stringify!($ty),
                $crate::__replicated!($($rep)?),
            ),
        )*];

        // Compile-time validation: duplicate ids are a hard error.
        const _: () = {
            if $crate::ComponentInfo::has_duplicate_ids(&ENTRIES) {
                panic!("register_components!: duplicate component id");
            }
        };

        /// Registry indexed by [`ComponentId`](crate::ComponentId); `None` for
        /// ids that were not registered.
        const COMPONENT_REGISTRY: [
            ::core::option::Option<$crate::ComponentInfo>;
            $crate::MAX_COMPONENTS
        ] = $crate::ComponentInfo::build_registry(&ENTRIES);
    };
}

/// `replicated = ...` helper: expands to the given expression, or `false`.
#[macro_export]
#[doc(hidden)]
macro_rules! __replicated {
    () => {
        false
    };
    ($e:expr) => {
        $e
    };
}
