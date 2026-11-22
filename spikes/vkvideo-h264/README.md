# Spike 0.5 — Vulkan Video H.264 **encode** from ash on RADV GFX1151

**Verdict: BLOCKED — the VCN encode ring never completes a submission on this
machine.** The H.264 fast-path in the plan cannot ship on this hardware as-is;
**v1 should ship JPEG-only** (or CPU `libx264`/OpenH264) and keep the Vulkan
encode path behind a runtime capability probe that this sandbox fails.

**Cause: VCN hang, cause unconfirmed (application, Mesa or firmware).** Every
Vulkan Video entry point resolves, the encode queue family is found, the session
and parameter sets are created, and the driver even emits valid SPS/PPS NALs —
only the actual `vkCmdEncodeVideoKHR` GPU work hangs. That rules out an ash or
driver-binding *entry-point* gap, but it does **not** prove the application is
correct: the validation-layer rerun below (see "Validation-layer rerun") found a
real application-side spec violation in the session creation path, so an
application bug cannot be excluded as a contributor to the hang. Pinning the
root cause needs kernel-side evidence the sandbox cannot read (`dmesg` is
denied).

---

## What works (verified on the real GPU, `AMD Radeon Graphics (RADV GFX1151)`)

Running `./target/release/vkvideo-h264 --out out.h264` with the env exports from
[`../ENVIRONMENT.md`](../ENVIRONMENT.md):

```
[dev] using AMD Radeon Graphics (RADV GFX1151) (queue family 3)
[enc] picture format G8_B8R8_2PLANE_420_UNORM (candidates: G8_B8R8_2PLANE_420_UNORM)
[enc] session memory: 1 bind(s), sizes [131072]
[main] caps: minCodedExtent=16x16 maxCodedExtent=4096x4096 granularity=16x16
        maxDpbSlots=16 maxActiveRefs=16 bitstreamAlign(size=16 offset=16)
        rateControlModes=DISABLED | CBR | VBR
        feedback=BITSTREAM_BUFFER_OFFSET | BITSTREAM_BYTES_WRITTEN
        h264: maxLevelIdc=0 minQp=0 maxQp=51 maxL0Refs(P)=1 maxTemporalLayers=4
        stdSyntaxFlags=WEIGHTED_BIPRED_IDC_EXPLICIT | ENTROPY_CODING_MODE_FLAG_UNSET
                    | ENTROPY_CODING_MODE_FLAG_SET | CONSTRAINED_INTRA_PRED_FLAG_SET
[enc] driver SPS 28 B (nalType 0), PPS 8 B (nalType 0)
```

Confirmed working end-to-end up to the GPU submit *(these calls return success
without validation enabled; the validation-layer rerun below shows the session
create-info is malformed, so "succeeds" here means "the driver accepted it", not
"it is spec-correct")*:

- **Extensions present & enabled:** `VK_KHR_video_queue`,
  `VK_KHR_video_encode_queue`, `VK_KHR_video_encode_h264` (plus
  `VK_KHR_video_maintenance1` for inline queries).
- **Encode queue family found:** family **3**, via
  `vkGetPhysicalDeviceQueueFamilyProperties2` +
  `VkQueueFamilyVideoPropertiesKHR` (`videoEncodeOperations` advertises
  `VK_VIDEO_CODEC_OPERATION_ENCODE_H264_BIT_KHR`).
- **All 15 video entry points resolve** through the loader (see gap list below —
  none are NULL).
- **`vkGetPhysicalDeviceVideoCapabilitiesKHR`** returns sane H.264 caps (above).
  Note `maxLevelIdc=0` — the driver reports *no* level ceiling, i.e. it lets the
  app pick any level; the spike pins `LEVEL_4_0`.
- **`vkCreateVideoSessionKHR`** succeeds; **`vkGetVideoSessionMemoryRequirementsKHR`**
  reports 1 binding of 131072 B; **`vkBindVideoSessionMemoryKHR`** succeeds.
- **`vkCreateVideoSessionParametersKHR`** + **`vkUpdateVideoSessionParametersKHR`**
  succeed, and **`vkGetEncodedVideoSessionParametersKHR`** returns a real
  **28-byte SPS** and **8-byte PPS** (Annex-B start codes prepended by the spike).
- **RGBA→NV12 compute conversion** (`convert.rs`, naga-compiled WGSL) runs on the
  compute queue and feeds the `G8_B8R8_2PLANE_420_UNORM` source image.

## The blocker

The first `vkQueueSubmit` containing `vkCmdEncodeVideoKHR` **hangs the VCN ring**.
The fence does not signal on the encode; instead the kernel GPU-fault/resets the
VCN engine after ~11 s and the fence is then satisfied by the reset, while the
`VK_QUERY_TYPE_VIDEO_ENCODE_FEEDBACK_KHR` query is left incomplete:

```
[main] frame 0: submission completed in 11098.19 ms
Error: read feedback frame 0
Caused by:
    0: get_query_pool_results(frame 0)
    1: A fence or query has not yet completed
```

Evidence this is a VCN-side hang and not a CPU/record error:

- With the encode command **omitted** (`SPIKE_NO_ENCODE=1`, begin/end session
  only) the same submission completes in **0.03 ms** — so the command buffer,
  session begin/end, barriers and submit path *run* without the encode. That
  shows they are accepted by the driver, **not** that they are spec-correct: no
  encode ever completed here, so nothing in `enc.rs`/`encode.rs` is validated
  end to end (and the validation-layer rerun below finds a real violation in
  the session setup that the no-encode path shares).
- The completion time (~11 s) matches the amdgpu job timeout, and the query
  never reaching `READY` is the signature of a reset VCN job.
- The VCN firmware *is* loaded:
  `/sys/.../0000:f4:00.0/fw_version/vcn_fw_version = 0x09118022`, and
  `vcn_reset_mask = "queue full"`. `vcn_busy_percent` reads 0 after the hang.
- Kernel log is not readable from the sandbox (`dmesg: Operation not permitted`),
  so the exact reset message could not be captured verbatim.

This reproduces identically on every run and on frame 0 (the IDR), so it is not
a DPB/reference-frame or rate-control issue.

## Validation-layer rerun

The original run had **no validation layer enabled**, so nothing checked the
spike's own video calls. Re-running with `VK_LAYER_KHRONOS_validation` forced on
(the spike never requests layers itself, so it is enabled through the loader
env) changes the failure mode and shows the spike is **not** known-clean:

```sh
# Layer from the rq-triangle README recipe (chromium's build, bookworm-compatible):
export VK_LAYER_PATH=/workspace/.local/vulkan-layers
export VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation
./target/release/vkvideo-h264 --out out.h264
```

Result (`artifacts/validation-rerun.log`, exit **139**, SIGSEGV):

```
[dev] using AMD Radeon Graphics (RADV GFX1151) (queue family 3)
[enc] picture format G8_B8R8_2PLANE_420_UNORM (candidates: G8_B8R8_2PLANE_420_UNORM)
Validation Error: [ VUID-VkVideoSessionCreateInfoKHR-pStdHeaderVersion-parameter ] | MessageID = 0xcf94a003
vkCreateVideoSessionKHR(): pCreateInfo->pStdHeaderVersion is NULL.
The Vulkan spec states: pStdHeaderVersion must be a valid pointer to a valid VkExtensionProperties structure
Segmentation fault
```

What this tells us:

- **The layer loads and runs for this binary** (the packaged bookworm build also
  loads but is too old to validate Vulkan-Video-on-1.4 — see below), and it
  immediately reports a **real application-side spec violation**:
  `VkVideoSessionCreateInfoKHR::pStdHeaderVersion` is NULL. ash
  0.38 does expose the field (`p_std_header_version`, added by
  `VK_KHR_video_maintenance1` — the extension the spike enables for inline
  queries), but `src/enc.rs` never sets it, so the raw
  `vkCreateVideoSessionKHR` call receives a create-info with a NULL
  `pStdHeaderVersion`.
- The process then **segfaults immediately after that call** (exit 139). The
  crash happens right where the layer flags the NULL pointer, so it cannot be
  attributed to the driver alone; `gdb` is not installed in the sandbox, so the
  faulting frame could not be identified. Without validation the same call
  *appears* to succeed, which is consistent with the driver accepting a
  malformed create-info and leaving the session unusable — a plausible path
  from an application bug to the VCN hang, though not proof of it.
- **Consequence for the verdict**: "the application is correct and the driver is
  at fault" is **not** established. The blocker statement is therefore *VCN
  hang, cause unconfirmed (application, Mesa or firmware)*, and any Phase-6
  retry of this path must first set `pStdHeaderVersion` (and re-run with
  validation enabled) before blaming the driver.

Two other packaged layers were tried and could not be used for this binary; both
attempts are recorded in `artifacts/`:

| Layer build | Outcome |
|---|---|
| bookworm `vulkan-validationlayers 1.3.239.0-2` (`artifacts/validation-rerun-bookworm.log`) | Loads, but its header knowledge is 1.3.239: it rejects the driver's Vulkan 1.4 video structs in the `vkCreateDevice` pNext chain (`VUID-VkDeviceCreateInfo-pNext-pNext`, unknown `VkStructureType 1000515000`) and the run aborts with `ERROR_VALIDATION_FAILED_EXT` before any encode. Too old to validate Vulkan-Video-on-1.4 usage. |
| trixie `vulkan-validationlayers 1.4.309.0-1` (`artifacts/validation-rerun-trixie.log`) | Will not load at all on bookworm: `GLIBC_2.38' not found (required by libVkLayer_khronos_validation.so)`. The loader skips the layer and the run proceeds unvalidated (same VCN hang), so it validates nothing. |

So the usable layer in this sandbox is the chromium-bundled one (the
`rq-triangle` recipe), and it is the one whose findings are above.

## ash entry-point gap list

ash **0.38.0** (VK headers 1.3.281) generates the *structures* for the
`VK_KHR_video_*` extensions but **no safe wrapper methods** and **no
`Extends*` traits** for the codec-specific chains. Every entry point below was
resolved through the raw `fp()` pointers in
`ash::extensions::khr::video_queue` / `video_encode_queue` (see `src/vkext.rs`).
**None returned NULL** — the loader hands out all of them, so there is no ash
*entry-point* gap blocking this spike:

| Entry point | level | resolved by loader |
|---|---|---|
| `vkGetPhysicalDeviceVideoCapabilitiesKHR` | instance | yes |
| `vkGetPhysicalDeviceVideoFormatPropertiesKHR` | instance | yes |
| `vkGetPhysicalDeviceVideoEncodeQualityLevelPropertiesKHR` | instance | yes |
| `vkCreateVideoSessionKHR` | device | yes |
| `vkDestroyVideoSessionKHR` | device | yes |
| `vkGetVideoSessionMemoryRequirementsKHR` | device | yes |
| `vkBindVideoSessionMemoryKHR` | device | yes |
| `vkCreateVideoSessionParametersKHR` | device | yes |
| `vkUpdateVideoSessionParametersKHR` | device | yes |
| `vkDestroyVideoSessionParametersKHR` | device | yes |
| `vkCmdBeginVideoCodingKHR` | device | yes |
| `vkCmdEndVideoCodingKHR` | device | yes |
| `vkCmdControlVideoCodingKHR` | device | yes |
| `vkGetEncodedVideoSessionParametersKHR` | device | yes |
| `vkCmdEncodeVideoKHR` | device | yes |

The gaps that *do* exist are ergonomic, not functional, and were worked around in
`src/vkext.rs`:

- No safe wrapper methods for any `vk*Video*KHR` entry point → called via raw
  `fp()` pointers.
- No `ExtendsVideoEncodeCapabilitiesKHR` trait → `VkVideoEncodeH264CapabilitiesKHR`
  cannot be `push_next`ed onto `VkVideoEncodeCapabilitiesKHR`; chained by hand
  (`vkext::chain`).
- No `ExtendsVideoEncodeRateControlInfoKHR` trait → same for
  `VkVideoEncodeH264RateControlInfoKHR`.
- Missing enum constants, re-declared by hand:
  `VK_VIDEO_CODEC_OPERATION_ENCODE_H264_BIT_KHR`,
  `VK_QUEUE_VIDEO_ENCODE_BIT_KHR`,
  `VK_PIPELINE_STAGE_2_VIDEO_ENCODE_BIT_KHR`,
  `VK_ACCESS_2_VIDEO_ENCODE_{READ,WRITE}_BIT_KHR`,
  `VK_VIDEO_SESSION_CREATE_INLINE_QUERIES_BIT_KHR`,
  `VK_QUERY_TYPE_RESULT_STATUS_ONLY_KHR`.
- `StdVideo*` (bindgen) types derive no `Default` → zeroed via `mem::zeroed`.

A newer ash (or the `VK_KHR_video_*` wrappers that landed after 0.38) would
remove the hand-written bindings but would **not** change the VCN hang.

## Encode ms/frame

**Not measurable** — no frame ever completes. The only submission timing is the
~11 s VCN reset, which is a failure artifact, not encode throughput.

## Decode path / PSNR

**Not reached** — `out.h264` is never produced, so neither the Chromium
WebCodecs decode check nor the PSNR ≥ 30 dB assertion could run. The harness for
both is present and validated independently:

- `--dump-yuv <file>` writes the CPU-side NV12 reference for all 60 frames
  (verified: 27 648 000 B = 60 × 640×480×1.5), so the PSNR/decode half can be
  exercised against an offline encoder (ffmpeg/libx264) on any machine.
- `ffmpeg` is **not installed** in this sandbox; the intended decode check was
  Chromium WebCodecs over CDP (`/usr/bin/chromium` is present).

## What Phase 6 must do differently

1. **Do not assume Vulkan H.264 encode works because the extensions and queue
   family are advertised.** RADV GFX1151 advertises full H.264 encode support and
   then hangs the VCN ring on the first real encode. Gate the H.264 fast-path on
   an **end-to-end probe** (encode one IDR, read back a non-zero feedback query),
   not on extension/queue-family presence.
2. **Ship JPEG-only in v1** (or CPU `libx264`/OpenH264 as the H.264 fallback).
   The JPEG path needs no VCN and is unaffected by this blocker.
3. Keep the Vulkan encode path as an **optional, runtime-detected** fast-path.
   `convert.rs` (RGBA→NV12 compute) and the Annex-B assembly in `h264.rs` are
   exercised independently of the encoder and can be reused, but **nothing in
   `enc.rs`/`encode.rs` is proven correct** — no encode ever completed, and the
   validation-layer rerun shows the session create-info is malformed
   (`pStdHeaderVersion` is NULL). Any reuse must start by fixing that and
   re-running with validation enabled.
4. If the fast-path is pursued later, first re-test on hardware/firmware where
   the VCN ring actually completes jobs (newer VCN firmware, or a discrete RDNA
   card with working VCN), **with the validation layer enabled**, and capture
   the kernel reset message to confirm the root cause.

## Files

- `src/main.rs` — driver: device/session setup, per-frame convert+encode loop,
  Annex-B `out.h264` + `out.json` (meta) writer, `--dump-yuv` reference mode.
- `src/dev.rs` — instance/device/queue creation, encode queue-family discovery.
- `src/vkext.rs` — hand-written raw bindings + missing enum constants (ash gap).
- `src/enc.rs` — caps query, session + session-parameters create, DPB, SPS/PPS.
- `src/encode.rs` — `vkCmdEncodeVideoKHR` record (IDR + P GOP), feedback query.
- `src/convert.rs` — RGBA→NV12 compute pipeline (naga WGSL→SPIR-V).
- `src/frames.rs` — 60 synthetic 640×480 RGBA frames, PSNR region, CPU NV12.
- `src/h264.rs` — Annex-B start-code assembly.
- `artifacts/validation-rerun.log` — validation-layer run (chromium layer):
  `pStdHeaderVersion` violation + SIGSEGV, exit 139.
- `artifacts/validation-rerun-bookworm.log` — validation-layer run (bookworm
  packaged layer 1.3.239): too old, aborts on the Vulkan 1.4 video pNext chain.
- `artifacts/validation-rerun-trixie.log` — validation-layer run (trixie
  packaged layer 1.4.309): layer will not load on bookworm (GLIBC 2.38), so the
  run proceeds unvalidated.

## Reproduce

```sh
export LD_LIBRARY_PATH=/workspace/.local/vulkan/usr/lib/x86_64-linux-gnu
export VK_DRIVER_FILES=/workspace/.local/vulkan/usr/share/vulkan/icd.d/radeon_icd.json:/workspace/.local/vulkan/usr/share/vulkan/icd.d/lvp_icd.json
cd spikes/vkvideo-h264
cargo run --release -- --out out.h264          # hangs VCN on frame 0 (the blocker)
SPIKE_NO_ENCODE=1 cargo run --release -- --out /tmp/x.h264   # submit path returns in 0.03 ms
cargo run --release -- --dump-yuv /tmp/ref.yuv # CPU NV12 reference, 60 frames

# With the validation layer forced on (the spike does not request layers itself):
export VK_LAYER_PATH=/workspace/.local/vulkan-layers          # see ../rq-triangle/README.md
export VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation
cargo run --release -- --out out.h264          # validation error + SIGSEGV (exit 139)
```
