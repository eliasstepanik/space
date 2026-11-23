# IC-4 "Kestrel"

Hard-sci-fi 23 m two-seat patrol interceptor with a complete interior. Machine-readable
record: [`manifest.json`](manifest.json) (validated in CI). Full build record:
[`build_report.json`](build_report.json). Repo-wide conventions:
[`../../README.md`](../../README.md).

## Files

| File | Size | SHA-256 |
|------|------|---------|
| `kestrel.glb` (full model) | 14,862,392 B | `4a063aa5910acac428614299bcb0feafa6e2ab7bc0e91a42e85faa19fa18bdf5` |
| `kestrel_lod1.glb` (proxy) | 1,712,052 B | `bc6c7f157cd39ae9dc52c8f84ec79c0b44ba2c0946990df42e64de118f887fb6` |
| `source/kestrel.blend` | 3,714,591 B | `a86c8b75548608dbbbd5557f21c75aea97a135bb554cad0f601a61b6ea5bc059` |
| `source/scripts/*.py` | | build + validation scripts (below) |
| `previews/01..25_*.jpg` | | 1280×720 review renders (heroes 24–25: 1920×1080) |

## Identity, axes, size

- **Type:** fighter / interceptor, two seats (pilot + engineer), fission reactor and reaction-mass tankage.
- **Axes (GLB):** `+Y` up, right-handed, **nose `-Z`**, starboard `+X`, 1 unit = 1 m.
  (Source `.blend`: Z-up, nose `+Y`; the exporter uses `export_yup=True`.)
- **Bounding box:** 12.64 (X) × 5.6436 (Y, gear deployed) × 23.795 (Z) m.
  glTF min `(-6.32, -3.3, -11.84)`, max `(6.32, 2.3436, 11.955)`. Design envelope was
  12.5 × 5.6 × 23 m (all axes within 10 %).
- **Frame check:** `SOCKET_Camera_PilotEye` is at z = −6.85 (forward of the origin) looking
  along −Z; both main-engine sockets are at z = +11.5 with the exhaust axis `+Z`.

## Model statistics

| | Full (`kestrel.glb`) | LOD1 (`kestrel_lod1.glb`) |
|-|----------------------|---------------------------|
| Triangles (instanced) | 417,546 (exterior 169,210, interior + bays 248,336) | 11,585 |
| Objects in source | 1,072 (1,069 exported as GLB nodes; the 3 preview-rig objects are excluded) | n/a |
| glTF nodes / meshes / primitives | 1,069 / 251 / 578 | 1 / 1 / 1 |
| Materials | 16 | 1 (`M_Kestrel_LOD1`) |
| Embedded images | 6 | 2 (`T_Hull_Normal`, `T_Hull_ORM`) |
| Sockets (`SOCKET_*`) | 41 | none |

glTF checks (all pass, see `build_report.json`): one embedded buffer, no URIs,
`extensionsRequired` empty, `extensionsUsed` only `KHR_materials_emissive_strength`,
all primitives TRIANGLES with NORMAL + TEXCOORD_0, no skins/animations/cameras/lights,
no Draco/meshopt.

The LOD1 proxy is the exterior only: all exterior meshes joined, welded and decimated
to ~12k triangles. The 16 materials are merged into one: per-face base colour in
`COLOR_0` plus the hull ORM and normal textures (UVs preserved).

## Rooms (14 IDs)

Interior volumes, fore to aft. Source: `10_blockout.py` (`INTERIOR`). "Pressurised"
rooms pass the strict enclosure ray check (500/500 rays hit, ≥ 99 % front-facing from
the centre); bays and wells pass their own criteria (see `build_report.json`,
`enclosure`).

| ID | Name | Contents / purpose | Class |
|----|------|--------------------|-------|
| B01 | NoseSensorBay | radar / optical sensor stack behind the flat nose face | unpressurised bay |
| B02 | NoseGearWell | nose gear retracts aft-up under the cockpit | gear well (closed) |
| R01 | Cockpit | tandem seats (seat positions in the Blender source frame: pilot Y 6.75, engineer Y 4.95; in the GLB these are z −6.75 / −4.95), raised under the canopy | pressurised |
| R02 | AvionicsLifeSupport | avionics racks + ECLSS; ladder up into the cockpit | pressurised |
| R03 | AirlockVestibule | suit-up vestibule, starboard of the airlock chamber | pressurised |
| R03a | AirlockChamber (part of R03) | sealed chamber, port hull hatch + ceiling hatch to the dorsal dock | pressurised |
| R04 | CrewQuarters | 2 bunks, galley, hygiene | pressurised |
| R05 | StoresLocker | consumables / spares lockers | pressurised |
| R06 | EngineeringBay | engineer station, reactor + propulsion control; crawlway entry aft-top | pressurised |
| C01 | DorsalCrawlway | 0.9 × 0.9 m crawlway along the dorsal spine, R06 → B07 | pressurised |
| B03 | OrdnanceBay | ventral weapons bay under R03/R04, 2 internal hardpoints | unpressurised bay |
| B04 | MainGearWell_L / MainGearWell_R | main gear wells in the port / starboard sponsons | gear wells (closed) |
| B05 | PropellantTankBay | reaction-mass tankage under the crawlway | unpressurised bay |
| B06 | ReactorCompartment | fission reactor + shadow shield, under the crawlway | unpressurised bay |
| B07 | EngineBay | turbopumps / engine feeds, feeds both nacelles | unpressurised bay |

R03a and the two B04 wells are separate volumes but share an ID with R03 / B04, hence
14 IDs for 16 volumes.

## Sockets (41)

Empties named `SOCKET_<Kind>_<Name>`, with `socket_kind` in the node `extras`. The
`-Z` local axis of each socket is its pointing / attach direction (verified against the
source in `build_report.json`, `socket_orientation`).

| Kind | Count | Sockets |
|------|-------|---------|
| Camera | 2 | `PilotEye`, `EngEye` |
| Seat | 2 | `Pilot`, `Engineer` |
| Engine | 2 | `Main_L`, `Main_R` |
| RCS | 8 | `Quad_{FwdUp,FwdDn,AftUp,AftDn}_{L,R}` |
| Hardpoint | 4 | `1_BayL`, `2_BayR`, `3_SponsonL`, `4_SponsonR` |
| Turret | 1 | `Chin` |
| Dock | 1 | `Dorsal` |
| Hatch | 1 | `Crew` |
| Gear | 3 | `Contact_Nose`, `Contact_Main_L`, `Contact_Main_R` |
| Nav | 3 | `Light_L`, `Light_R`, `Light_Tail` |
| Light | 14 | `Light_<room ID>` for B01 B02 R01–R06 C01 B03–B07 |
| **Total** | **41** | |

## Materials (16)

`M_Decals`, `M_Glass_Canopy`, `M_HullBare_Metal`, `M_HullPaint_Grey`,
`M_HullPaint_White`, `M_Int_Equipment`, `M_Int_Fabric`, `M_Int_Floor`, `M_Int_Panel`,
`M_Lights_Emissive`, `M_MLI_Foil`, `M_Paint_SafetyYellow`, `M_Radiator`,
`M_Rubber_Dark`, `M_Screens_Emissive`, `M_Thermal_Dark`.

(Emissive materials use `KHR_materials_emissive_strength`.)

## Textures (6, embedded, 854,578 B in total)

| Name | Size | Format | Colour space |
|------|------|--------|--------------|
| `T_Decal_Atlas` | 2048×2048 | PNG | sRGB |
| `T_Hull_Normal` | 1024×1024 | JPEG | Non-Color |
| `T_Hull_ORM` | 1024×1024 | JPEG | Non-Color |
| `T_Int_Normal` | 1024×1024 | JPEG | Non-Color |
| `T_Int_ORM` | 1024×1024 | JPEG | Non-Color |
| `T_Screen_Atlas` | 1024×1024 | JPEG | sRGB |

## Previews

| # | File | Shows |
|---|------|-------|
| 01 | `01_blockout_exterior.jpg` | blockout (stage 10), exterior |
| 02 | `02_blockout_top_cutaway.jpg` | blockout, top cutaway with the interior layout |
| 03 | `03_exterior_starboard_fg.jpg` | exterior, starboard-forward view |
| 04 | `04_exterior_port_aft.jpg` | exterior, port-aft view |
| 05 | `05_exterior_top.jpg` | exterior, orthographic top |
| 06 | `06_exterior_bottom.jpg` | exterior, orthographic bottom |
| 07 | `07_exterior_nose_close.jpg` | nose close-up |
| 08 | `08_exterior_engine_close.jpg` | engine / nacelle close-up |
| 09 | `09_room_R01.jpg` | R01 Cockpit |
| 10 | `10_room_R02.jpg` | R02 Avionics / life support |
| 11 | `11_room_R03.jpg` | R03 Airlock vestibule |
| 12 | `12_room_R03a.jpg` | R03a Airlock chamber |
| 13 | `13_room_R04.jpg` | R04 Crew quarters |
| 14 | `14_room_R05.jpg` | R05 Stores locker |
| 15 | `15_room_R06.jpg` | R06 Engineering bay |
| 16 | `16_room_C01.jpg` | C01 Dorsal crawlway |
| 17 | `17_bay_B01.jpg` | B01 Nose sensor bay |
| 18 | `18_bay_B02.jpg` | B02 Nose gear well (near wall cut away) |
| 19 | `19_bay_B03.jpg` | B03 Ordnance bay |
| 20 | `20_bay_B04.jpg` | B04 Main gear well (near wall cut away) |
| 21 | `21_bay_B05.jpg` | B05 Propellant tank bay |
| 22 | `22_bay_B06.jpg` | B06 Reactor compartment |
| 23 | `23_bay_B07.jpg` | B07 Engine bay |
| 24 | `24_hero_port_bow.jpg` | hero shot, port bow |
| 25 | `25_hero_full_deployed.jpg` | hero shot, whole ship, gear deployed |

## Rebuilding

Requires **Blender 5.2.4 LTS** (the version used for the committed files) plus Python
with `numpy` (bundled with Blender). The scripts are in `source/scripts/` and import
`00_common.py` next to them.

**Working directory does not matter:** `00_common.py` derives every path from its own
file location (`SOURCE_DIR = .../source`, `SHIP_DIR = .../kestrel`), so the defaults are
`source/kestrel.blend`, `kestrel.glb`, `kestrel_lod1.glb` and `previews/`, whichever
directory Blender is started from. (Script docstrings mention `../kestrel.blend` as a
relative example; the code does not use it.) Arguments after `--` are `--key value`
pairs. Run from `assets/ships/kestrel/`:

```sh
S=source/scripts
# 1. Rebuild the .blend (each stage re-opens --blend and saves to --out; the defaults
#    are source/kestrel.blend, so the flags can be omitted)
blender -b --factory-startup --python $S/10_blockout.py                         # wipes + creates the blockout
blender -b --factory-startup --python $S/20_exterior.py -- --blend source/kestrel.blend --out source/kestrel.blend
blender -b --factory-startup --python $S/30_interior.py -- --blend source/kestrel.blend --out source/kestrel.blend
blender -b --factory-startup --python $S/31_bays.py     -- --blend source/kestrel.blend --out source/kestrel.blend
blender -b --factory-startup --python $S/40_materials.py -- --blend source/kestrel.blend --out source/kestrel.blend

# 2. Export (the .blend is opened but never saved)
blender -b --factory-startup source/kestrel.blend --python $S/50_export.py -- --what full
blender -b --factory-startup source/kestrel.blend --python $S/50_export.py -- --what lod1 --target 12000

# 3. Validate
blender -b --factory-startup --python $S/51_validate_glb.py -- --glb kestrel.glb
blender -b --factory-startup --python $S/51_validate_glb.py -- --glb kestrel_lod1.glb --lod1
```

To rebuild into a scratch file instead of overwriting the committed one, pass
`--out /tmp/chain.blend` to stage 10 and `--blend /tmp/chain.blend --out /tmp/chain.blend`
to stages 20–40 (this is how the chain was verified).

Optional / reporting scripts: `11`, `21`, `31_interior_previews.py`, `32`, `54` render
the previews (`previews/`, Cycles); `52_reimport_check.py` and `53_enclosure.py` write
JSON reports (`--out file.json`); `55_build_report.py --work DIR` assembles
`build_report.json` from those reports.

**Reproducibility:** the chain is deterministic in content (two full runs gave the same
hash over sorted object names, world matrices and per-mesh triangle counts) but the
`.blend` bytes differ between runs. The committed `.blend` therefore only matches its
manifest hash as committed; after a rebuild update the manifest's `bytes`/`sha256` (and
triangle counts) for every changed file, otherwise `cargo test -p xtask --test assets`
will fail.

The scripts were not run during this repo-integration step (no Blender in CI); the
commands above are taken from the script docstrings and argument handling.
