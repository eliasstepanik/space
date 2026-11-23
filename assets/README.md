# Assets

Binary game assets that ship with the repo. Each asset lives in its own folder
with a machine-readable `manifest.json` and a human-readable `README.md`.

| Asset | Folder | Notes |
|-------|--------|-------|
| IC-4 "Kestrel" patrol interceptor | [`ships/kestrel/`](ships/kestrel/README.md) | full model + LOD1 proxy + Blender source |

## Layout

```
assets/ships/<id>/
  <id>.glb            full model (runtime asset)
  <id>_lod1.glb       low-poly proxy (optional)
  manifest.json       id, axes, bbox, per-file bytes + sha256, counts, rooms
  README.md           spec sheet + rebuild instructions
  build_report.json   full build record (optional)
  source/             .blend + build scripts (NOT in the Docker build context)
  previews/           review renders (NOT in the Docker build context)
```

`<id>` is lowercase `snake_case` (`kestrel`).

## Conventions

- **Format:** glTF 2.0 binary (`.glb`), one embedded buffer, **no external
  `uri`** anywhere. Textures are embedded (PNG/JPEG via `bufferView`).
- **Axes and units:** glTF `+Y` up, right-handed, **forward (nose) = `-Z`**,
  starboard = `+X`, **1 unit = 1 metre**. (Blender sources are Z-up with the
  nose on `+Y`; the exporter converts with `export_yup=True`.)
- **File names:** lowercase, words joined with `_` (`kestrel_lod1.glb`).
- **Node name prefixes:**
  - `EXT_` exterior hull / equipment,
  - `INT_` interior (rooms, bays and their contents),
  - `SOCKET_` attachment-point empties (`SOCKET_<Kind>_<Name>`); the kind is
    also stored in the node's glTF `extras` as `socket_kind`.
- **Extensions:** `extensionsRequired` must be empty. Only
  `KHR_materials_emissive_strength` is used. No Draco, meshopt or other
  compression extensions (loaders stay trivial; git compresses the files anyway).

## Budgets (enforced by the xtask test)

| Limit | Value |
|-------|-------|
| File size | ≤ 45 MB per file |
| Triangles | ≤ 60,000 per glTF mesh (the model is split into many meshes; the whole ship is far larger) |
| Materials | ≤ 16 per GLB |
| Primitive mode | TRIANGLES only |
| Textures | embedded, no compression extensions |

## No Git LFS

Assets are committed as **plain git blobs**. The largest file is ~14 MB, the whole
Kestrel folder is ~23 MB, so LFS would add infrastructure (a server, the CI
runners and the Docker build all need `git lfs`) for little gain. `.gitattributes`
marks `*.glb`, `*.blend` and `*.jpg` as `binary` so git never tries to diff or
merge them. Revisit if a single asset passes ~45 MB or the repo grows noticeably.

Every rebuild of a binary adds a full copy to history: only commit an asset when
it is final, and never commit intermediate exports.

## Validation

`cargo run -p xtask -- ci` (fmt, clippy, tests) runs `xtask/tests/assets.rs`. It:

- parses `ships/<id>/manifest.json`, then checks that each listed file exists
  and that its **size and SHA-256** (streamed) match;
- parses the GLB header (magic `glTF`, version 2, total length, JSON/BIN chunk
  lengths) of both LODs;
- checks the glTF JSON: no `uri` keys, `extensionsRequired` empty, every primitive
  is TRIANGLES, one buffer, the LOD1 proxy has exactly one mesh and one primitive;
- recomputes triangle counts from the accessors and compares them with the
  manifest (and mesh/node/socket/material/image counts and the budgets above).

Run just this test with `cargo test -p xtask --test assets`.

After intentionally changing an asset, update `bytes` and `sha256` in the
manifest (`sha256sum <file>`, `stat -c %s <file>`) in the same commit.

The Docker build context (`.dockerignore`) drops `ships/*/source/` and
`ships/*/previews/` but keeps the GLBs and manifests, because the CI Dockerfile
runs this test. Manifest entries marked `"optional_in_build_context": true`
(the `.blend`) are verified when present and skipped when absent.
