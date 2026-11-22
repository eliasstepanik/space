//! Type-erased column storage — **no `unsafe`**.
//!
//! An archetype column holds one value per entity row. The natural
//! implementation is a `Vec<T>` per component; the type erasure needed to keep
//! columns of different types in one `Vec` is done with
//! `Box<dyn Any + Send>`, which the standard library implements *safely*:
//! reading a column is a `downcast` that checks the [`TypeId`] instead of
//! assuming it.
//!
//! This is the deliberate design decision of the crate. The brief allows a
//! single `unsafe_storage.rs` module if type-erased storage forced `unsafe`;
//! it does not, so there is no `unsafe` anywhere and the crate keeps
//! `#![forbid(unsafe_code)]`. The cost is one `Any` type check per column
//! access; the benchmarks in `tests/acceptance.rs` (1 M spawns < 1 s, 1 M-row
//! iteration < 10 ms, release) show it is negligible.
//!
//! The per-component operations that must survive type erasure (`push`,
//! `remove_row`, `copy_row`) are stored in [`Column`] as `fn` pointers created
//! by `Column::new` for the concrete type, so the archetype can migrate a
//! row between columns without knowing the type at the call site.

use std::any::{Any, TypeId};

use crate::component::ComponentData;

/// Type-erased "append a copy of `value` to this column".
///
/// # Panics
/// If `value` is not the column's data type.
pub(crate) type PushFn = fn(&mut Column, &(dyn Any + Send));

/// Type-erased "remove row `row` (swap-remove) from this column".
pub(crate) type RemoveFn = fn(&mut Column, usize);

/// Type-erased "copy row `row` of `src` into `dst`".
///
/// # Panics
/// If either column does not hold the concrete type the pointer was made for.
pub(crate) type CopyFn = fn(src: &Column, row: usize, dst: &mut Column);

/// Type-erased "overwrite row `row` of this column with `value`".
///
/// # Panics
/// If `value` is not the column's data type, or `row` is out of range.
pub(crate) type SetFn = fn(&mut Column, usize, &(dyn Any + Send));

/// One SoA column: owned, type-erased storage of one component's values.
///
/// All columns of an archetype have the same number of rows; the length is
/// mirrored here because `dyn Any` cannot report the length of an unknown
/// `Vec<T>`.
#[doc(hidden)]
pub struct Column {
    data: Box<dyn Any + Send>,
    type_id: TypeId,
    len: usize,
    push: PushFn,
    remove: RemoveFn,
    copy: CopyFn,
    set: SetFn,
}

impl std::fmt::Debug for Column {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Column")
            .field("type", &self.type_id)
            .field("len", &self.len)
            .finish()
    }
}

impl Column {
    /// An empty column for component data type `T`.
    pub(crate) fn new<T: ComponentData>() -> Self {
        fn push<T: ComponentData>(col: &mut Column, value: &(dyn Any + Send)) {
            let value = value.downcast_ref::<T>().unwrap_or_else(|| {
                panic!(
                    "column push type mismatch: column holds {}, got {}",
                    std::any::type_name::<T>(),
                    std::any::type_name_of_val(value)
                )
            });
            col.vec_mut::<T>().push(value.clone());
            col.len += 1;
        }
        fn remove<T: ComponentData>(col: &mut Column, row: usize) {
            col.vec_mut::<T>().swap_remove(row);
            col.len -= 1;
        }
        fn copy<T: ComponentData>(src: &Column, row: usize, dst: &mut Column) {
            let value = src.get::<T>(row);
            dst.push_dyn(value);
        }
        fn set<T: ComponentData>(col: &mut Column, row: usize, value: &(dyn Any + Send)) {
            let value = value.downcast_ref::<T>().unwrap_or_else(|| {
                panic!(
                    "column set type mismatch: column holds {}, got {}",
                    std::any::type_name::<T>(),
                    std::any::type_name_of_val(value)
                )
            });
            col.vec_mut::<T>()[row] = value.clone();
        }
        Self {
            data: Box::new(Vec::<T>::new()),
            type_id: TypeId::of::<T>(),
            len: 0,
            push: push::<T>,
            remove: remove::<T>,
            copy: copy::<T>,
            set: set::<T>,
        }
    }

    /// [`TypeId`] of the stored data.
    #[inline]
    pub(crate) fn type_id(&self) -> TypeId {
        self.type_id
    }

    /// Appends a copy of `value`.
    #[inline]
    pub(crate) fn push_dyn(&mut self, value: &(dyn Any + Send)) {
        (self.push)(self, value)
    }

    /// Removes `row` with swap-remove semantics.
    #[inline]
    pub(crate) fn remove_row(&mut self, row: usize) {
        (self.remove)(self, row)
    }

    /// Overwrites the value at `row` in place.
    ///
    /// # Panics
    /// If `value` is not the column's data type, or `row` is out of range.
    #[inline]
    pub(crate) fn set(&mut self, row: usize, value: &(dyn Any + Send)) {
        (self.set)(self, row, value)
    }

    /// Copies `row` of `src` into `dst`; both columns must hold the same type.
    #[inline]
    pub(crate) fn copy_row(src: &Column, row: usize, dst: &mut Column) {
        (src.copy)(src, row, dst)
    }

    /// Shared reference to the value at `row`.
    ///
    /// # Panics
    /// If the column does not hold `T`, or `row` is out of range.
    pub(crate) fn get<T: ComponentData>(&self, row: usize) -> &T {
        self.slice::<T>()
            .get(row)
            .unwrap_or_else(|| panic!("column row {row} out of range (len {})", self.len))
    }

    /// Shared view of the column as a slice.
    ///
    /// # Panics
    /// If the column does not hold `T`.
    pub(crate) fn slice<T: ComponentData>(&self) -> &[T] {
        match self.data.downcast_ref::<Vec<T>>() {
            Some(v) => v,
            None => panic!(
                "column type mismatch: expected {}, found {}",
                std::any::type_name::<T>(),
                self.describe()
            ),
        }
    }

    /// Mutable view of the column.
    ///
    /// # Panics
    /// If the column does not hold `T`.
    pub(crate) fn vec_mut<T: ComponentData>(&mut self) -> &mut Vec<T> {
        if self.data.downcast_ref::<Vec<T>>().is_none() {
            panic!(
                "column type mismatch: expected {}, found {}",
                std::any::type_name::<T>(),
                self.describe()
            );
        }
        // The type check above guarantees the downcast succeeds.
        self.data.downcast_mut::<Vec<T>>().unwrap()
    }

    fn describe(&self) -> String {
        format!("column of TypeId({:?})", self.type_id)
    }
}
