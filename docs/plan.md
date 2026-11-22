# space v1 core plan (revision 2)

- One renderer built on ash/Vulkan, hidden behind a `Renderer` trait.
- DDGI probe global illumination via `VK_KHR_ray_query`.
- Browser client = pixel-streamed view of the native client (WebSocket,
  **JPEG only** in v1; Vulkan H.264 encode is an optional fast path, see
  Phase 6).
- Dynamic Docker orchestrator with an epoch-versioned shard map.
- Custom archetype ECS.
- Quinn netcode.
- Two MCP servers over Streamable HTTP.

## Phases

- Phase 0 — risk-gate spikes
- Phase 1 — foundations (world/ecs/protocol/sim)
  - Position model: i64 cell ×3 + f32 local ×3, 1024 m cells (P1).
  - Snapshot transport (P9): a `Snapshot` rides a **reliable, ordered QUIC
    stream**, not an unreliable datagram. At ~158 B per entity a 1k-entity
    interest set is ~158 KB, far past what datagrams carry without fragmentation;
    the per-client byte budget (`SnapshotBuilder`) bounds each frame, and the
    stream's ordering/flow control is what a full state dump needs. Deltas stay
    on the datagram path (small, idempotent, gap-detectable). Phase 2 wires this
    up; the size bound here is the design input, and the datagram path's cap
    should come from `max_datagram_size()` when it is implemented.
- Phase 2 — headless multiplayer + H1–H8
  - Scale targets: ~100 players/shard, ~1k entities in interest range (U9).
- Phase 3 — orchestrator + H9–H15
  - Load metric `max(players/100, tick p95/33.3 ms)`; split at ≥0.8 for 10 s, merge at ≤0.35 for 60 s (U12).
  - ShardId allocation: ids are **monotonic and never reused** (FIX 9). Guid
    uniqueness `(shard, serial)` requires it — a recycled id would alias two
    entities' guids. The orchestrator owns the counter and persists it across
    restarts. If monotonicity proves impractical, add an incarnation field to the
    guid instead (Phase 3 decision).
- Phase 4 — renderer core
  - WGSL→SPIR-V via naga at build time (P2).
- Phase 5 — ray-query GI/SSR/fog/refraction
- Phase 6 — streaming
  - Latency budget ≤80 ms p95 (JPEG 720p60), measured capture-to-on-screen (P10).
  - Codec: **JPEG only** in v1. Vulkan H.264 encode is an **optional fast path**,
    gated on an **end-to-end encode probe** (encode one IDR and read back a
    non-zero feedback query) — never on extension/queue-family presence — and
    deferred after the GFX1151 VCN encode hang found in spike 0.5.
  - wasm thin viewer for QA (QA).
- Phase 7 — MCP
- Phase 8 — packaging + skill

