# Spike 0.3 — ash + naga-compiled WGSL ray-query compute pass

**Status: PASS.** This throwaway spike proves the rendering bet for
`space-render-ash`: a compute-only Vulkan program built with [`ash`], with a
**WGSL** compute shader compiled to SPIR-V by **naga** at build time, can build
a bottom-level acceleration structure and trace rays with `rayQuery*` inside a
compute shader, and the rendered pixels read back correctly.

It is standalone: `Cargo.toml` has an empty `[workspace]` table and it is **not**
a member of the root workspace. Nothing here is intended to be reused as-is —
the point is the recipe and the go/no-go answers below.

```
spikes/rq-triangle/
├── Cargo.toml              # standalone; ash + gpu-allocator + naga (build-dep)
├── build.rs                # WGSL -> SPIR-V via naga at build time
├── shaders/rq_triangle.wgsl
└── src/main.rs             # offscreen ash program, no surface
```

## What it proves

1. **naga compiles WGSL ray-query to valid SPIR-V** (see the verdict below).
2. The full **ray-query extension chain** works on a real AMD iGPU (RADV) and on
   lavapipe: instance + device creation, `VkAccelerationStructureKHR` build,
   scratch-buffer sizing, device-address bindings.
3. A **compute shader can trace rays** with `rayQuery*` and write correct pixels.
4. **`gpu-allocator`** works for the BLAS buffer, the scratch buffer, the
   storage image and the staging readback buffer, with correct `device_address`
   and buffer→image copy semantics.
5. **Validation is clean**: the whole run produces **zero** validation-layer
   messages (the spike treats any message as a hard failure).
6. The pixel assertions are **real**: with an empty scene the test FAILs (see
   the mutation check below), so a passing run is not a vacuous pass.

## The naga ray-query verdict (QC go/no-go)

**GO — with a version constraint.**

| naga | WGSL ray-query compiles? | Verdict |
|------|--------------------------|---------|
| **24.0.0** | **NO — not this shader.** naga 24 has no `wgpu_ray_query` enable extension and **no** `rayQueryConfirmIntersection` / `rayQueryGenerateIntersection` in its WGSL front end at all, so the shader as written fails to parse: `no definition in scope for identifier: 'rayQueryConfirmIntersection'`. Only after deleting both the `enable` line and the `rayQueryConfirmIntersection` call does naga 24 emit SPIR-V (994 words) — i.e. a *different, weaker* shader. | **NOT USABLE.** With the confirm-free variant, an earlier run of this spike observed the centre pixels randomly dropping 1–4 hits per run on RADV (0.5σ–2σ below the expected count) while lavapipe always passed. **The root cause of those dropped hits is UNKNOWN.** The earlier explanation — "`rayQueryTerminate` is a no-op in the SPIR-V backend, so queries are emitted but never terminated, which is UB" — is **unsupported**: this shader never calls `rayQueryTerminate` (the loop drives `rayQueryProceed` to completion, which is what terminates the query), and a SPIR-V diff of the confirm-free variant shows naga 24 and naga 30 emitting the **same** ray-query opcode set (`OpRayQueryInitializeKHR`, `OpRayQueryProceedKHR`, `OpRayQueryGetIntersectionTypeKHR`; no `OpRayQueryTerminateKHR` in either). Whatever caused the dropped hits, it is not explained by a missing terminate, and it was never reproduced on naga 30. |
| **30.0.2** (what this spike uses) | **YES** — 1997 words, full feature set | **USE THIS.** Supports `rayQueryConfirmIntersection`, `rayQueryGenerateIntersection` and a working `rayQueryTerminate`. **10/10** consecutive runs on RADV and **3/3** on lavapipe produced the exact expected pixel counts. |

**Consequence for `space-render-ash`:** the renderer's build dependency is
**`naga >= 30`, declared in `[workspace.dependencies]`** of the root workspace,
so the renderer cannot silently resolve an older naga through a transitive
dependency. CI must **reject naga < 30** — either a `cargo-deny` `ban`/`skip`
entry for `naga < 30`, or a guard step running
`cargo tree -i naga@24` and failing if it resolves anything. Verify locally with
`cargo tree -i naga` (the renderer must resolve to ≥ 30).

WGSL ray-query requires the module-level enable line:

```wgsl
enable wgpu_ray_query;
```

Without it naga 30 rejects the shader with:
`the 'wgpu_ray_query' enable extension is not enabled`.

## How to run

From the repo root, with the Vulkan environment from [`../ENVIRONMENT.md`](../ENVIRONMENT.md):

```bash
export RUSTUP_TOOLCHAIN=stable
export LD_LIBRARY_PATH=/workspace/.local/vulkan/usr/lib/x86_64-linux-gnu
export VK_DRIVER_FILES=/workspace/.local/vulkan/usr/share/vulkan/icd.d/radeon_icd.json
export VK_LAYER_PATH=/workspace/.local/vulkan-layers   # optional: validation layer

cd spikes/rq-triangle
cargo run --release          # PASS = exit 0
```

`VK_DRIVER_FILES` selects the driver: use `radeon_icd.json` for the RADV iGPU
(the acceptance target) or `lvp_icd.json` for lavapipe. Both pass.

**Validation layer.** The sandbox has no packaged `VK_LAYER_KHRONOS_validation`;
this spike uses the one shipped with chromium, copied to a local prefix:

```bash
mkdir -p /workspace/.local/vulkan-layers
cp /usr/lib/chromium/libVkLayer_khronos_validation.so /workspace/.local/vulkan-layers/
# plus the matching VkLayer_khronos_validation.json manifest in the same dir,
# with "library_path" set to "./libVkLayer_khronos_validation.so"
export VK_LAYER_PATH=/workspace/.local/vulkan-layers
```

If no validation layer is present the spike **fails loudly** rather than
silently skipping it; pass `--no-validation` to opt out explicitly.

Useful flags:

| Flag | Effect |
|------|--------|
| `--empty-scene` | **Mutation check.** Builds a BLAS with zero triangles; the test must FAIL with a non-zero exit. |
| `--no-validation` | Skip the validation layer (needed on machines without one). |
| `--dump <path>` | Write the RGBA image as a PPM. |

A passing run ends with:

```
device: AMD Radeon Graphics (RADV GFX1151)
BLAS sizes: as=640B build_scratch=65932B
pixel assertions: 722/4096 pixels == HIT (expected 722), 6 centre pixels == HIT, 4 corner pixels == MISS (tol 8)
validation messages (warning+): 0
PASS: ray-query compute produced the expected pixels
```

and prints nothing from the validation layer. A failing run prints
`FAIL: <reason>` and exits 1.

### Mutation check (proves the assertions bite)

```
$ cargo run --release -- --empty-scene
device: AMD Radeon Graphics (RADV GFX1151)
BLAS sizes: as=256B build_scratch=128B
  first mismatch at (32, 24): got [0, 128, 255, 255], expected [255, 0, 0, 255]
validation messages (warning+): 0
FAIL: 7 pixel assertion(s) failed: expected 722 hit pixels, got 0 / 4096; ...
$ echo $?
1
```

## Exact extensions / features required

**Instance**

- API version requested in `VkApplicationInfo::api_version`: **1.3** (the
  instance must be created with the same or newer API version than anything the
  device needs, otherwise `vkCreateDevice` fails with `ERROR_INCOMPATIBLE_DRIVER`).
- Layers: `VK_LAYER_KHRONOS_validation` (optional, but any message is a failure;
  pass `--no-validation` to skip).
- Extensions:
  - `VK_KHR_get_physical_device_properties2`
  - `VK_EXT_debug_utils` (for the validation sink)

**Physical device selection**

- A queue family with `VK_QUEUE_COMPUTE_BIT` (no surface/graphics needed).
- `VkPhysicalDeviceAccelerationStructureFeaturesKHR::accelerationStructure = TRUE`
- `VkPhysicalDeviceRayQueryFeaturesKHR::rayQuery = TRUE`
- `VkPhysicalDeviceBufferDeviceAddressFeatures::bufferDeviceAddress = TRUE`

**Device**

- Extensions: `VK_KHR_ray_query`, `VK_KHR_acceleration_structure`,
  `VK_KHR_deferred_host_operations`, `VK_KHR_buffer_device_address`,
  `VK_KHR_get_memory_requirements2`.
- Chain `VkPhysicalDeviceRayQueryFeaturesKHR`,
  `VkPhysicalDeviceAccelerationStructureFeaturesKHR` and
  `VkPhysicalDeviceBufferDeviceAddressFeatures` into `VkDeviceCreateInfo`
  (via `VkDeviceCreateInfo::push_next` on a `PhysicalDeviceFeatures2`).

**Shader (WGSL)**

- `enable wgpu_ray_query;` at module scope (required by naga 30).
- Output is `texture_storage_2d<rgba8unorm, write>` (bound as a
  `VK_IMAGE_USAGE_STORAGE_BIT` R8G8B8A8_UNORM image), written with
  `textureStore`.
- `acceleration_structure` bound via a
  `VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR` descriptor.
- Ray-query builtins used: `rayQueryInitialize`, `rayQueryProceed`,
  `rayQueryGetCandidateIntersection` (+ `RAY_QUERY_INTERSECTION_TRIANGLE`),
  `rayQueryConfirmIntersection`, `rayQueryGetCommittedIntersection`. The
  closest-hit loop drives `rayQueryProceed` to completion (which terminates the
  query) rather than calling `rayQueryTerminate` inside the loop.
- The `Params` uniform keeps every member as `vec4` (or scalar + explicit pad)
  so the WGSL and SPIR-V std140 layouts agree exactly — a SPIR-V `vec3` occupies
  16 bytes while a WGSL `vec3` occupies 12, which would silently shift members.
- `build.rs` targets SPIR-V **1.4** (`lang_version: (1, 4)`); naga emits the
  `SPV_KHR_ray_query` capability + extension automatically when it sees the
  ray-query builtins. Validation runs with `ValidationFlags::all()` and
  `Capabilities::all()`.

**Acceleration structure**

- `VkAccelerationStructureCreateInfoKHR::ty` is **`GENERIC`**, not `BOTTOM_LEVEL`.
  A descriptor-bound acceleration structure must be `TOP_LEVEL` or `GENERIC`
  (`VUID-VkWriteDescriptorSetAccelerationStructureKHR-pAccelerationStructures-03579`);
  the build info still builds it as `BOTTOM_LEVEL`.
- One `VkAccelerationStructureGeometryKHR` of type `TRIANGLES`, using
  `vertex_data`/`index_data` device addresses (RGB32_SFLOAT vertices, UINT32
  indices, `index_type = UINT32`, `max_vertex = 2`).
- Build flags: `PREFER_FAST_TRACE`. Sizes come from
  `get_acceleration_structure_build_sizes` with the *same* build info (minus the
  destination/scratch fields). A scratch buffer is required even for an
  empty-primitive build; keep at least 256 bytes.
- Build with `cmd_build_acceleration_structures` in a **separate command-buffer
  submission** from the trace dispatch, waiting on a fence between them. This
  fully separates the build from the trace so no barrier subtlety can leak an
  incomplete BVH into the ray queries.
- Image layout transitions are explicit `cmd_pipeline_barrier` image barriers:
  `UNDEFINED → GENERAL` before the dispatch (COMPUTE_SHADER / SHADER_WRITE) and
  `GENERAL → TRANSFER_SRC_OPTIMAL` before the readback copy (COMPUTE_SHADER →
  TRANSFER, SHADER_WRITE → TRANSFER_READ). `initialLayout` must be `UNDEFINED`
  or `PREINITIALIZED` (`VUID-VkImageCreateInfo-initialLayout-00993`).

**Other gotchas found on the way**

- `GeometryFlagsKHR` has **no** `TRIANGLE_FACING_CULL_DISABLE` — that flag lives
  on `GeometryInstanceFlagsKHR` (top-level) only.
- `PipelineStageFlags::COMPUTE_SHADER`, not `COMPUTE`.
- `ash` 0.38 renamed the property-2 extension constant to
  `VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_NAME`.
- `gpu_allocator::vk::VulkanAllocator::create_buffer` takes
  `vk::MemoryPropertyFlags` as its own argument and does **not** add
  `eDeviceAddress`; the caller must set it.
- With `AllocationScheme::DedicatedBuffer(buffer)` gpu-allocator does **not**
  own the `VkBuffer`: free the allocation, then `destroy_buffer` it yourself, or
  `vkDestroyDevice` reports leaked buffers.
- The BLAS buffer needs `VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR
  | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT`; the geometry buffers need
  `..._BUILD_INPUT_READ_ONLY_BIT_KHR | ..._SHADER_DEVICE_ADDRESS_BIT`.

## Notes for `space-render-ash`

- The whole program is ~900 lines of `ash` with no higher-level wrapper; the
  ray-query-specific parts are under ~150 lines (BLAS build + barrier + the
  `rayQuery` block in WGSL). The rest is boilerplate that a renderer crate will
  need anyway.
- The WGSL ray-query API is close to the WebGPU/Dawn proposal, so shaders written
  for it port cleanly. Use the canonical closest-hit loop
  (`proceed` → `rayQueryConfirmIntersection` on triangle candidates → read the
  committed intersection after the loop) — that is exactly what
  `shaders/rq_triangle.wgsl` does, and it is the pattern naga 30 supports. Do
  **not** call `rayQueryTerminate` inside the loop: driving `proceed` to
  completion is what terminates the query, and leaving a query active at
  invocation exit is undefined behaviour. (naga 24 cannot compile this pattern
  at all — see the verdict table.)
- `gpu-allocator` handles `VK_MEMORY_PROPERTY_DEVICE_ADDRESS_BIT` correctly and
  exposes `Allocation::info().buffer_device_address`.

[`ash`]: https://github.com/MaikKlein/ash
[`../ENVIRONMENT.md`]: ../ENVIRONMENT.md
