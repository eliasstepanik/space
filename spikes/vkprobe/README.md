# vkprobe — Phase-0 spike 0.2 (throwaway)

A standalone, headless probe that reports the Vulkan capabilities of the machine
it runs on as JSON. Used to decide CI test tiers and whether the H.264 encode
spike (0.5) is worth running.

This package is **throwaway** and deliberately **not** a member of the root
`space` workspace — its `Cargo.toml` carries an empty `[workspace]` table so it
builds independently.

## How to run

```sh
cd spikes/vkprobe
cargo run --release
```

Pipe through a JSON validator to confirm the output is well-formed:

```sh
cargo run --release 2>/dev/null | python3 -m json.tool
# or: cargo run --release 2>/dev/null | jq .
```

The probe creates a Vulkan instance **without** any surface / window-system
extensions, so it is safe on a machine with no display. It always exits `0` and
always prints valid JSON — even when no Vulkan driver exists (that is a valid,
important result, not an error).

Run it with the default `VK_LOADER_DRIVERS_SELECT` (i.e. no special env) to see
what the loader picks up by default on the host.

## Output fields

Top level:

- `vulkan_available` — `true` if the Vulkan loader (`libvulkan.so.1`) loaded.
  `false` means no loader/driver is present at all.
- `loader_error` — present only when `vulkan_available` is `false`; the loader
  load failure message.
- `instance_created` — `true` if a headless instance was created.
- `instance_error` — present only when instance creation failed (e.g. loader
  present but no usable ICD).
- `ash_version` — the ash version the probe was built against.
- `devices_found` — number of physical devices enumerated.
- `devices` — per-device report (see below). Empty when no device is found.
- `surface.requested` — always `false`: surface extensions were NOT requested,
  so no display is assumed.
- `conclusion` — human-readable summary for the tier decision.

Per device:

- `device_name`, `vendor_id`, `device_id`, `device_type`
  (`discrete_gpu` / `integrated_gpu` / `virtual_gpu` / `cpu` / `other`).
- `api_version` — `major.minor.patch` string; `api_version_raw` is the packed u32.
- `driver_version` — driver-reported version (packed u32).
- `ray_query.feature_supported` — whether the `VK_KHR_ray_query` device feature
  is reported supported (queried via a `PhysicalDeviceFeatures2` pNext chain).
- `ray_query.extensions` — presence of `VK_KHR_ray_query`,
  `VK_KHR_acceleration_structure`, `VK_KHR_deferred_host_operations`,
  `VK_KHR_ray_tracing_pipeline`.
- `rt_tier.ray_query_enableable` — whether the ray-query feature can be enabled
  (the minimal RT tier). Full RT pipelines also need the ray-tracing extensions.
- `video_encode_h264` — `extensions_present` is true only when BOTH
  `VK_KHR_video_encode_queue` and `VK_KHR_video_encode_h264` are present.
  `encode_op_supported` and `video_profiles` are `null` (see Limitations).
- `vulkan_video_decode_h264.present` — informational only.
- `memory.heaps_total_bytes` — sum of all memory heap sizes; `memory.heaps`
  lists each heap.
- `all_device_extensions` — every device extension name the driver advertises.

## Interpretation guide

- `vulkan_available: false` (or `devices_found: 0`) → **no Vulkan device here**.
  RT tests and the H.264 spike cannot run on this machine. (This is *not* the
  case in the current build sandbox — see "Sandbox result" below; it was true
  only before the userspace stack from `../ENVIRONMENT.md` was installed.)
- `ray_query.feature_supported: true` (and the ray-tracing extensions present)
  → **RT tests can run here**.
- `video_encode_h264.extensions_present: true` → **only** that the encode
  extensions are advertised. **Extension presence does NOT imply working
  encode.** Spike 0.5 proved this on this very device: RADV GFX1151 reports
  `extensions_present: true` and an H.264 encode queue family, then hangs the
  VCN ring on the first real `vkCmdEncodeVideoKHR`. Treat this field as "worth
  probing", never as "encode works"; the only trustworthy signal is an
  end-to-end encode probe (encode one IDR, read back a non-zero feedback
  query).

## Limitations

- ash `0.38.0+1.3.281` does **not** expose
  `get_physical_device_video_profile_info_khr`, so the probe cannot enumerate
  `VK_VIDEO_H264_STANDARD_PROFILE_ID` video profiles or confirm encode-op
  support. It degrades gracefully and reports extension presence only, with the
  limitation recorded in `video_encode_h264.limitation`. A newer ash (e.g. the
  `pixelforge-ash` fork at `0.38.1+1.4.352`) exposes these entry points if
  profile-level detail is needed later.

## Sandbox result

`report-sandbox.json` is the committed output from the build sandbox **with the
userspace Vulkan stack from [`../ENVIRONMENT.md`](../ENVIRONMENT.md) installed**.
It shows `vulkan_available: true` with `devices_found: 2`:

- `AMD Radeon Graphics (RADV GFX1151)` — integrated GPU, api 1.4.305,
  `ray_query.feature_supported: true`, `video_encode_h264.extensions_present:
  true` (but see the interpretation guide: presence ≠ working encode — spike 0.5
  found the encode hangs on this device).
- `llvmpipe (LLVM 15.0.6, 256 bits)` — CPU device, api 1.4.305,
  `ray_query.feature_supported: true`, no H.264 encode extensions.

So RT (ray-query) tests can run here on both devices; the H.264 encode spike
runs on the GPU but is blocked by the VCN hang documented in
[`../vkvideo-h264/README.md`](../vkvideo-h264/README.md).
