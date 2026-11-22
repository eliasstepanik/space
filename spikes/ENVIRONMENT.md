# Sandbox Vulkan environment (Phase-0 spike 0.2b)

The agent sandbox has real AMD GPU passthrough: `/dev/dri/renderD128` -> PCI
`f4:00.0` (AMD/ATI 1002:1586, "GFX1151" iGPU), driven by kernel `amdgpu`.
Only the **userspace** Vulkan stack was missing; it is installed without root
into `/workspace/.local/vulkan` (Debian `.deb`s extracted with `dpkg-deb -x`).

## Install

One command (idempotent; prints the exports, writes `<prefix>/env.sh`, then
runs `spikes/vkprobe`; flags: `--prefix <path>`, `--force`, `--print-env`):

```sh
cargo xtask gpu-setup
```

It downloads these pinned `.deb`s from `http://deb.debian.org/debian/pool/main`
and extracts them with `dpkg-deb -x` (no root; `apt-get update` fails because
`/var/lib/apt` is read-only). `deb.debian.org` must be allowed through the
egress proxy. **Every download is verified against a hardcoded SHA-256 in
`xtask/src/main.rs` (`VULKAN_DEBS`) before `dpkg-deb -x` runs**; a mismatch
aborts the install instead of unpacking an unverified archive.

- `v/vulkan-loader/libvulkan1_1.3.239.0-1_amd64.deb`
- `m/mesa/mesa-vulkan-drivers_25.0.7-2~bpo12+1_amd64.deb` — **backports** Mesa
  25.0.7 links `libLLVM-15` (present) and knows GFX1151; bookworm Mesa 22.3.6
  fails with `amdgpu: unknown (family_id, chip_external_rev): (150, 193)`
- `libd/libdrm/libdrm-amdgpu1_2.4.123-1~bpo12+1_amd64.deb`
- `libd/libdrm/libdrm2_2.4.123-1~bpo12+1_amd64.deb`
- `v/vulkan-validationlayers/vulkan-validationlayers_1.3.239.0-2_amd64.deb` —
  the `VK_LAYER_KHRONOS_validation` layer (matches the bookworm loader's
  1.3.239). Note it is **too old to validate Vulkan-Video on a Vulkan 1.4
  driver** (it rejects the driver's 1.4 video structs in `vkCreateDevice`'s
  pNext chain); for the video spikes use the chromium-bundled layer described in
  [`rq-triangle/README.md`](rq-triangle/README.md). Newer Debian builds
  (1.4.309) need `GLIBC_2.38` and will not load on bookworm.

## Env vars and result

```sh
export LD_LIBRARY_PATH=/workspace/.local/vulkan/usr/lib/x86_64-linux-gnu
export VK_DRIVER_FILES=/workspace/.local/vulkan/usr/share/vulkan/icd.d/radeon_icd.json:/workspace/.local/vulkan/usr/share/vulkan/icd.d/lvp_icd.json
export VK_LAYER_PATH=/workspace/.local/vulkan/usr/share/vulkan/explicit_layer.d
```

`radeon_icd.json` = real GPU (RADV); `lvp_icd.json` = lavapipe fallback.
`cd spikes/vkprobe && cargo run --release` -> `vulkan_available: true`, 2
devices: `AMD Radeon Graphics (RADV GFX1151)` (api 1.4.305, ray_query yes,
H.264 encode exts yes) and `llvmpipe` (cpu, ray_query yes, no encode).
