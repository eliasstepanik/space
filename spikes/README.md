# spikes

Throwaway Phase-0 validation spikes. Each spike is a standalone Cargo project
that exists to de-risk one technical assumption; nothing here is production
code and spikes may be deleted once their question is answered.

**All Phase-0 spikes are complete** (see each spike's own README for its
verdict). They are **throwaway**: no workspace crate depends on them, and none
of them may be depended on by workspace code. Their `Cargo.toml`s carry an empty
`[workspace]` table, so they are deliberately **not** members of the root
`space` workspace and are not built by `cargo xtask ci`.

| Spike | Question it answered | Verdict |
| --- | --- | --- |
| [`vkprobe`](vkprobe/) | Vulkan device/feature/extension probing | 2 devices in the sandbox (RADV GFX1151 + lavapipe); extension presence ≠ working encode |
| [`rq-triangle`](rq-triangle/) | ash + naga-compiled WGSL ray-query triangle smoke render | PASS with `naga >= 30` |
| [`stream-loop`](stream-loop/) | end-to-end frame streaming loop latency | PASS (JPEG over WebSocket; the ~1 ms figure is loopback decode+draw only) |
| [`vkvideo-h264`](vkvideo-h264/) | Vulkan Video H.264 encode | BLOCKED — VCN hang, cause unconfirmed; v1 is JPEG-only |
| [`orch-docker`](orch-docker/) | Docker orchestrator shard-map behaviour | Partial — pure helpers unit-tested; the Docker E2E needs a CI home (Phase-3 entry criterion) |

The sandbox Vulkan environment these spikes run against is documented in
[`ENVIRONMENT.md`](ENVIRONMENT.md) and installed with `cargo xtask gpu-setup`.
