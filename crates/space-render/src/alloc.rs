//! RAII wrappers around gpu-allocator allocations.
//!
//! Hard constraint: gpu-allocator does **not** own a `VkBuffer`/`VkImage`
//! passed to `AllocationScheme::DedicatedBuffer`/`DedicatedImage`, so the
//! `Allocation` must be freed by hand and the Vulkan object destroyed in the
//! right order (free the allocation first, then destroy the buffer). Every
//! allocation in this crate goes through these wrappers, whose `Drop` does
//! exactly that — a leaked dedicated allocation is impossible by construction.
//!
//! The allocator lives behind `Arc<Mutex<..>>` so each wrapper can free itself
//! independently of the owning context's drop order.

use std::sync::{Arc, Mutex};

use ash::vk;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator};

pub(crate) type SharedAllocator = Arc<Mutex<Allocator>>;

/// A `VkBuffer` plus its dedicated gpu-allocator `Allocation`.
pub(crate) struct Buffer {
    pub handle: vk::Buffer,
    /// `Some` while the dedicated allocation is live; `Drop` takes it and frees
    /// it by hand (gpu-allocator does not own the dedicated `VkBuffer`).
    allocation: Option<Allocation>,
    device: ash::Device,
    allocator: SharedAllocator,
    name: &'static str,
}

impl Buffer {
    /// Create a buffer with a dedicated allocation and upload `data` into it
    /// when the memory is CPU-visible (`location` must then be `CpuToGpu`).
    pub fn new(
        device: &ash::Device,
        allocator: &SharedAllocator,
        size: u64,
        usage: vk::BufferUsageFlags,
        location: gpu_allocator::MemoryLocation,
        name: &'static str,
    ) -> Result<Self, crate::RenderError> {
        let handle = unsafe {
            device
                .create_buffer(
                    &vk::BufferCreateInfo::default()
                        .size(size)
                        .usage(usage)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE),
                    None,
                )
                .map_err(|e| crate::RenderError::with_name(name, format!("create_buffer: {e}")))?
        };
        let requirements = unsafe { device.get_buffer_memory_requirements(handle) };
        let allocation = allocator
            .lock()
            .expect("allocator mutex poisoned")
            .allocate(&AllocationCreateDesc {
                name,
                requirements,
                location,
                linear: true,
                allocation_scheme: AllocationScheme::DedicatedBuffer(handle),
            })
            .map_err(|e| crate::RenderError::with_name(name, format!("allocate: {e}")))?;
        unsafe { device.bind_buffer_memory(handle, allocation.memory(), allocation.offset()) }
            .map_err(|e| crate::RenderError::with_name(name, format!("bind_buffer_memory: {e}")))?;
        Ok(Self {
            handle,
            allocation: Some(allocation),
            device: device.clone(),
            allocator: Arc::clone(allocator),
            name,
        })
    }

    fn allocation(&self) -> &Allocation {
        self.allocation.as_ref().expect("allocation already freed")
    }

    /// Device address of the buffer start (requires the buffer to have been
    /// created with `SHADER_DEVICE_ADDRESS`).
    pub fn device_address(&self) -> u64 {
        unsafe {
            self.device.get_buffer_device_address(
                &vk::BufferDeviceAddressInfo::default().buffer(self.handle),
            )
        }
    }

    /// Copy `data` into the (CPU-mapped) allocation and flush if needed.
    pub fn write(&self, data: &[u8]) -> Result<(), crate::RenderError> {
        if data.is_empty() {
            // Nothing to copy, and a zero-sized flush range is not legal.
            return Ok(());
        }
        let alloc = self.allocation();
        let ptr = alloc
            .mapped_ptr()
            .ok_or_else(|| crate::RenderError::with_name(self.name, "not CPU-mapped"))?;
        if data.len() as u64 > alloc.size() {
            return Err(crate::RenderError::with_name(
                self.name,
                format!(
                    "write of {} bytes into a {}-byte allocation",
                    data.len(),
                    alloc.size()
                ),
            ));
        }
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr.cast().as_ptr(), data.len());
        }
        if !alloc
            .memory_properties()
            .contains(vk::MemoryPropertyFlags::HOST_COHERENT)
        {
            unsafe {
                self.device
                    .flush_mapped_memory_ranges(&[vk::MappedMemoryRange::default()
                        .memory(alloc.memory())
                        .offset(alloc.offset())
                        .size(data.len() as u64)])
            }
            .map_err(|e| crate::RenderError::with_name(self.name, format!("flush: {e}")))?;
        }
        Ok(())
    }

    /// Read `data.len()` bytes back from a CPU-visible allocation, invalidating
    /// the mapped range first when it is not host-coherent.
    pub fn read(&self, out: &mut [u8]) -> Result<(), crate::RenderError> {
        if out.is_empty() {
            return Ok(());
        }
        let alloc = self.allocation();
        if out.len() as u64 > alloc.size() {
            return Err(crate::RenderError::with_name(
                self.name,
                format!(
                    "read of {} bytes from a {}-byte allocation",
                    out.len(),
                    alloc.size()
                ),
            ));
        }
        if !alloc
            .memory_properties()
            .contains(vk::MemoryPropertyFlags::HOST_COHERENT)
        {
            unsafe {
                self.device
                    .invalidate_mapped_memory_ranges(&[vk::MappedMemoryRange::default()
                        .memory(alloc.memory())
                        .offset(alloc.offset())
                        .size(out.len() as u64)])
            }
            .map_err(|e| crate::RenderError::with_name(self.name, format!("invalidate: {e}")))?;
        }
        let ptr = alloc
            .mapped_ptr()
            .ok_or_else(|| crate::RenderError::with_name(self.name, "not CPU-mapped"))?;
        unsafe {
            std::ptr::copy_nonoverlapping(ptr.cast().as_ptr(), out.as_mut_ptr(), out.len());
        }
        Ok(())
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        // Order matters: free the dedicated allocation first (it references
        // the buffer), then destroy the buffer itself.
        if let Some(allocation) = self.allocation.take() {
            if let Err(e) = self
                .allocator
                .lock()
                .expect("allocator mutex poisoned")
                .free(allocation)
            {
                log::error!("free({}): {e}", self.name);
            }
        }
        unsafe { self.device.destroy_buffer(self.handle, None) };
    }
}

/// A `VkImage` plus its dedicated gpu-allocator `Allocation`.
pub(crate) struct Image {
    pub handle: vk::Image,
    allocation: Option<Allocation>,
    device: ash::Device,
    allocator: SharedAllocator,
    name: &'static str,
}

impl Image {
    pub fn new(
        device: &ash::Device,
        allocator: &SharedAllocator,
        create_info: &vk::ImageCreateInfo<'_>,
        location: gpu_allocator::MemoryLocation,
        name: &'static str,
    ) -> Result<Self, crate::RenderError> {
        let handle = unsafe {
            device
                .create_image(create_info, None)
                .map_err(|e| crate::RenderError::with_name(name, format!("create_image: {e}")))?
        };
        let requirements = unsafe { device.get_image_memory_requirements(handle) };
        let allocation = allocator
            .lock()
            .expect("allocator mutex poisoned")
            .allocate(&AllocationCreateDesc {
                name,
                requirements,
                location,
                linear: true,
                allocation_scheme: AllocationScheme::DedicatedImage(handle),
            })
            .map_err(|e| crate::RenderError::with_name(name, format!("allocate: {e}")))?;
        unsafe { device.bind_image_memory(handle, allocation.memory(), allocation.offset()) }
            .map_err(|e| crate::RenderError::with_name(name, format!("bind_image_memory: {e}")))?;
        Ok(Self {
            handle,
            allocation: Some(allocation),
            device: device.clone(),
            allocator: Arc::clone(allocator),
            name,
        })
    }
}

impl Drop for Image {
    fn drop(&mut self) {
        if let Some(allocation) = self.allocation.take() {
            if let Err(e) = self
                .allocator
                .lock()
                .expect("allocator mutex poisoned")
                .free(allocation)
            {
                log::error!("free({}): {e}", self.name);
            }
        }
        unsafe { self.device.destroy_image(self.handle, None) };
    }
}
