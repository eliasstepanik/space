//! The physics bodies a shard replicates, as the client draws them.
//!
//! A body arrives as a replicated entity ([`crate::link::Tracked`]) with its
//! physics components: a `Body` (its collision shape), a `Visual` (which
//! model), `Springs`, an `Articulation` with `PartStates` (one pose per rigid
//! part) and a `PosResidual`. This module turns that into render instances:
//!
//! * **Where** ([`pose_at`]): the shard steps at 30 Hz and the client draws
//!   at whatever its display does, so a body is drawn at a render time a
//!   little in the past ([`INTERPOLATION_DELAY`] ticks behind the newest tick
//!   heard) and interpolated between the two samples around it — position
//!   exactly (as a `WorldPos` difference), orientation by normalised lerp.
//!   When the newest sample is older than the render time (a frame is late)
//!   the body is carried on along its velocity, for at most
//!   [`MAX_EXTRAPOLATION`] ticks, and then held. Every part of an articulated
//!   body is drawn at its own pose. A body that a neighbouring shard took
//!   over arrives as its ghost with the same guid and simply goes on; one
//!   that vanishes from the replica for a moment (it changed hands) is held
//!   where it was for [`LINGER`] before it is dropped.
//! * **What** ([`Look`], [`MeshCache`]): the visual id picks the material, the
//!   body's shape gives the geometry ([`crate::meshes::shape_mesh`]) — the
//!   collider and the picture are the same shape. One mesh per (visual,
//!   shape), built once and shared.
//! * **What holds it**: each spring is drawn as a coil of steel beads between
//!   its two attachment points, stretching and squeezing with the spring
//!   ([`coil`]); each joint gets a pin (a ball for a ball joint, a short axle
//!   along a hinge's axis), so a chain reads as one chain and a door hangs on
//!   its hinge.
//!
//! | visual | drawn as |
//! |---|---|
//! | crate | its box, panelled, coloured by size: amber 0.4–0.5 m, orange 0.6 m, teal 0.7 m (the floating ones) |
//! | ball | its sphere, two-tone checkered so a roll shows: red 0.2 m, blue 0.25 m, sand 0.3 m (the planetoid's moons) |
//! | plank | its box, hazard-striped |
//! | door | its box, blue panels |
//! | chain | its capsule, polished steel, with a pin at every joint |
//! | pendulum | its sphere, polished brass |
//! | kestrel | the Kestrel's model |
//! | bowl, ramp, gantry, curb, fulcrum, door post | the static scenery of the showcase ([`crate::demo_area`]) |
//! | planetoid | its sphere, in grey-brown rock with craters ([`rock_texture`]) |
//! | none / unknown, with a body | its shape in plain grey |

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};

use space_protocol::{far, EntityGuid};
use space_render::{
    CameraPose, GpuInstance, MeshData, MeshMaterial, ModelInstance, Scene, Texture, SHAPE_SPHERE,
};
use space_scene::visual;
use space_sim::articulation::{JointKind, JointTarget};
use space_sim::shape::ShapeRef;
use space_sim::spring::SpringTarget;
use space_world::WorldPos;

use crate::link::{Physical, Replica, Sample, Source};
use crate::meshes::{bowl_mesh, paint, quat_matrix, shape_mesh, y_onto};

/// How far behind the newest tick heard bodies are drawn, in ticks: two
/// samples are then nearly always on hand to interpolate between.
pub const INTERPOLATION_DELAY: f64 = 2.0;

/// How long a body is carried on along its velocity past its newest sample,
/// in ticks, before it is held where it got to.
pub const MAX_EXTRAPOLATION: f64 = 3.0;

/// How long a body that is no longer replicated is still drawn where it was
/// last: long enough to cover a handoff between shards (gone from one tick,
/// back as a ghost the next), short enough to go unnoticed at the edge of the
/// interest range.
pub const LINGER: Duration = Duration::from_millis(300);

/// One part's place, as drawn.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pose {
    /// Where the part's origin is (the server's world).
    pub pos: WorldPos,
    /// The part of the position finer than `pos` holds, metres.
    pub residual: [f32; 3],
    /// Orientation `(w, x, y, z)`.
    pub quat: [f32; 4],
    /// Velocity, m/s.
    pub vel: [f32; 3],
}

impl Pose {
    /// The point `local` (the part's frame, metres) in the world: its place
    /// and the residual offset to add to it.
    #[must_use]
    pub fn point(&self, local: [f32; 3]) -> (WorldPos, [f64; 3]) {
        let m = quat_matrix(self.quat);
        let turned = [0, 1, 2].map(|r| f64::from((0..3).map(|k| m[r][k] * local[k]).sum::<f32>()));
        let offset = [0, 1, 2].map(|k| turned[k] + f64::from(self.residual[k]));
        (self.pos, offset)
    }
}

fn pose_of(state: &space_sim::ShipState, residual: Option<&[f32; 3]>) -> Pose {
    Pose {
        pos: state.pos,
        residual: residual.copied().unwrap_or([0.0; 3]),
        quat: state.quat,
        vel: state.vel,
    }
}

fn poses_of(sample: &Sample) -> Vec<Pose> {
    sample
        .parts
        .iter()
        .enumerate()
        .map(|(i, s)| pose_of(s, sample.residual.get(i)))
        .collect()
}

/// Normalised lerp of two unit quaternions, the short way round.
#[must_use]
pub fn nlerp(a: [f32; 4], b: [f32; 4], t: f32) -> [f32; 4] {
    let d: f32 = (0..4).map(|k| a[k] * b[k]).sum();
    let b = if d < 0.0 { b.map(|c| -c) } else { b };
    let q = [0, 1, 2, 3].map(|k| a[k] + (b[k] - a[k]) * t);
    let l = q.iter().map(|c| c * c).sum::<f32>().sqrt();
    if l > 1e-6 && l.is_finite() {
        q.map(|c| c / l)
    } else {
        a
    }
}

/// The poses of every part at render time `t` (ticks, fractional) from a
/// history (oldest first): interpolated between the samples around `t`,
/// carried on along the velocity for at most [`MAX_EXTRAPOLATION`] ticks past
/// the newest, held before the oldest. `None` for an empty history.
#[must_use]
pub fn pose_at(history: &VecDeque<Sample>, t: f64) -> Option<Vec<Pose>> {
    let first = history.front()?;
    let last = history.back()?;
    if t <= first.tick as f64 {
        return Some(poses_of(first));
    }
    if t >= last.tick as f64 {
        let ahead = (t - last.tick as f64).min(MAX_EXTRAPOLATION) * f64::from(space_sim::dt());
        let mut poses = poses_of(last);
        for p in &mut poses {
            p.pos = WorldPos::from_f64_metres_around(p.pos, p.vel.map(|v| f64::from(v) * ahead));
        }
        return Some(poses);
    }
    let i = history
        .iter()
        .rposition(|s| s.tick as f64 <= t)
        .unwrap_or(0);
    let (a, b) = (&history[i], &history[(i + 1).min(history.len() - 1)]);
    if a.parts.len() != b.parts.len() || b.tick <= a.tick {
        return Some(poses_of(b));
    }
    let gap = (b.tick - a.tick) as f64;
    let alpha = (t - a.tick as f64) / gap;
    let (pa, pb) = (poses_of(a), poses_of(b));
    Some(
        pa.iter()
            .zip(&pb)
            .map(|(p, q)| {
                let d = WorldPos::delta(p.pos, q.pos);
                Pose {
                    pos: WorldPos::from_f64_metres_around(p.pos, along(p, q, d, gap, alpha)),
                    residual: [0, 1, 2]
                        .map(|k| p.residual[k] + (q.residual[k] - p.residual[k]) * alpha as f32),
                    quat: nlerp(p.quat, q.quat, alpha as f32),
                    vel: [0, 1, 2].map(|k| p.vel[k] + (q.vel[k] - p.vel[k]) * alpha as f32),
                }
            })
            .collect(),
    )
}

/// The render time of an entity with `history`, where the near range is
/// drawn at `t` (ticks): further in the past the further it is from the own
/// ship at `own` (`space_protocol::far::draw_delay` — far things are sent less
/// often). The distance is taken to where the entity is **drawn**, found in
/// two steps, not to its newest sample: that one moves thirty times a
/// second or less, and a delay that stepped with it would make everything
/// that comes nearer or goes away judder at that rate (0.2 m a step at
/// 40 m/s, seen on a live stack).
#[must_use]
pub fn draw_time(history: &VecDeque<Sample>, t: f64, own: Option<WorldPos>) -> f64 {
    let Some(own) = own else { return t };
    let mut at = t;
    for _ in 0..2 {
        let Some(poses) = pose_at(history, at) else {
            return t;
        };
        let distance = WorldPos::distance(own, poses[0].pos);
        at = t - (far::draw_delay(distance) - far::NEAR_DELAY);
    }
    at
}

/// How far along the way `d` from pose `p` to pose `q`, `gap` ticks apart,
/// a part is at fraction `alpha`, metres.
///
/// Two samples a tick apart: the straight line, as always. Samples further
/// apart (a far thing, sent a few times a second): the cubic through both
/// that leaves and arrives with the **velocities** the samples carry, so a
/// moon on its orbit goes round between two samples instead of cutting the
/// corner and changing direction at each one. The velocities are used only
/// when they tell the same story as the two positions — their mean within a
/// quarter of the chord's velocity. They do not for a body that was moved
/// rather than moving (what an orbiting planet carries is displaced with it,
/// at speeds its own velocity knows nothing of) or that bounced in between;
/// then it is the straight line again.
fn along(p: &Pose, q: &Pose, d: [f64; 3], gap: f64, alpha: f64) -> [f64; 3] {
    let linear = d.map(|c| c * alpha);
    if gap <= 1.0 {
        return linear;
    }
    let dt = gap * f64::from(space_sim::dt());
    let (v0, v1) = (p.vel.map(f64::from), q.vel.map(f64::from));
    let chord = d.map(|c| c / dt);
    let off: f64 = (0..3)
        .map(|k| (v0[k] + v1[k]) * 0.5 - chord[k])
        .map(|c| c * c)
        .sum::<f64>()
        .sqrt();
    let speed = chord.iter().map(|c| c * c).sum::<f64>().sqrt();
    if off > 0.25 * speed + 0.05 {
        return linear;
    }
    let (a2, a3) = (alpha * alpha, alpha * alpha * alpha);
    let (h10, h01, h11) = (a3 - 2.0 * a2 + alpha, 3.0 * a2 - 2.0 * a3, a3 - a2);
    [0, 1, 2].map(|k| h10 * dt * v0[k] + h01 * d[k] + h11 * dt * v1[k])
}

/// How one visual id is painted: a material, a texture and how often the
/// texture repeats.
#[derive(Clone, Debug)]
pub struct Look {
    /// The material.
    pub material: MeshMaterial,
    /// Its base-colour texture.
    pub texture: Option<Texture>,
    /// Metres per texture repeat.
    pub tile: f32,
}

/// The largest half extent (or radius) of a shape: what sizes a look.
fn size_of(shape: &ShapeRef) -> f32 {
    match shape {
        ShapeRef::Ball { radius } => *radius,
        ShapeRef::Cuboid { half_extents } => half_extents.iter().fold(0.0f32, |a, b| a.max(*b)),
        ShapeRef::Capsule {
            half_height,
            radius,
        }
        | ShapeRef::Cylinder {
            half_height,
            radius,
        }
        | ShapeRef::Cone {
            half_height,
            radius,
        } => half_height.max(*radius),
        _ => 1.0,
    }
}

const STRIPES_YELLOW: [u8; 3] = [240, 196, 20];
const STRIPES_BLACK: [u8; 3] = [24, 24, 26];

/// A planetoid's surface: grey-brown rock — two octaves of value noise —
/// pocked with darker craters with pale rims; wraps round (seamless in u).
#[must_use]
pub fn rock_texture() -> Texture {
    const W: u32 = 128;
    const H: u32 = 64;
    let hash = |x: u32, y: u32| -> f32 {
        let mut h = x.wrapping_mul(0x9E37_79B1) ^ y.wrapping_mul(0x85EB_CA77);
        h ^= h >> 15;
        h = h.wrapping_mul(0x2C1B_3C6D);
        h ^= h >> 12;
        (h & 0xFFFF) as f32 / 65535.0
    };
    // Value noise on a lattice of `cells` across, wrapping in x.
    let noise = |x: f32, y: f32, cells: u32| -> f32 {
        let (fx, fy) = (x * cells as f32 / W as f32, y * cells as f32 / W as f32);
        let (x0, y0) = (fx.floor() as u32, fy.floor() as u32);
        let (tx, ty) = (fx - fx.floor(), fy - fy.floor());
        let s = |t: f32| t * t * (3.0 - 2.0 * t);
        let at = |i: u32, j: u32| hash(i % cells, j);
        let a = at(x0, y0) + (at(x0 + 1, y0) - at(x0, y0)) * s(tx);
        let b = at(x0, y0 + 1) + (at(x0 + 1, y0 + 1) - at(x0, y0 + 1)) * s(tx);
        a + (b - a) * s(ty)
    };
    let craters: [(f32, f32, f32); 7] = [
        (20.0, 18.0, 7.0),
        (70.0, 40.0, 9.0),
        (100.0, 14.0, 5.0),
        (44.0, 50.0, 4.0),
        (118.0, 46.0, 6.0),
        (5.0, 42.0, 3.5),
        (84.0, 24.0, 3.0),
    ];
    let mut pixels = Vec::with_capacity((W * H * 4) as usize);
    for y in 0..H {
        for x in 0..W {
            let (fx, fy) = (x as f32, y as f32);
            let mut v = 0.45 * noise(fx, fy, 8) + 0.25 * noise(fx, fy, 32) + 0.45;
            for (cx, cy, r) in craters {
                // Distance round the seam too.
                let dx = (fx - cx).abs().min(W as f32 - (fx - cx).abs());
                let d = (dx * dx + (fy - cy) * (fy - cy)).sqrt() / r;
                if d < 0.85 {
                    v *= 0.62;
                } else if d < 1.15 {
                    v *= 1.18;
                }
            }
            let v = v.clamp(0.0, 1.0);
            let rgb = [v, v * 0.92, v * 0.82].map(|c| (c * 255.0) as u8);
            pixels.extend_from_slice(&[rgb[0], rgb[1], rgb[2], 255]);
        }
    }
    Texture::from_rgba8(W, H, pixels).unwrap_or_else(|_| Texture::solid([150, 140, 125]))
}

/// A crate's face: boards inside a dark frame, with a diagonal brace — grey,
/// for the material's colour to tint. The frame is what shows where one
/// crate of a stack ends and the next begins.
#[must_use]
pub fn crate_texture() -> Texture {
    const N: u32 = 64;
    let mut pixels = Vec::with_capacity((N * N * 4) as usize);
    for y in 0..N {
        for x in 0..N {
            let edge = x.min(y).min(N - 1 - x).min(N - 1 - y);
            let brace = (x as i32 - y as i32).abs() <= 3 && edge >= 5;
            let board_gap = edge >= 5 && (y % 11 == 0);
            let v: u8 = if edge < 5 {
                70
            } else if edge < 6 {
                120
            } else if brace {
                205
            } else if board_gap {
                150
            } else {
                // A little grain along the boards.
                (228 - ((x * 7 + y * 3) % 9) * 3) as u8
            };
            pixels.extend_from_slice(&[v, v, v, 255]);
        }
    }
    Texture::from_rgba8(N, N, pixels).unwrap_or_else(|_| Texture::solid([220, 220, 220]))
}

/// The look of visual `id` on `shape` (see the [module docs](self)).
#[must_use]
pub fn look(id: u32, shape: &ShapeRef) -> Look {
    let size = size_of(shape);
    let panels = |color: [f32; 3]| Look {
        material: paint(color, 0.05, 0.55),
        texture: Some(Texture::hull_panels(128, [236, 238, 242])),
        tile: 2.0 * size,
    };
    let plain = |color: [f32; 3], metallic: f32, roughness: f32| Look {
        material: paint(color, metallic, roughness),
        texture: None,
        tile: 1.0,
    };
    let stripes = |stripes: u32| Look {
        material: paint([1.0; 3], 0.0, 0.6),
        texture: Some(Texture::hazard_stripes(
            128,
            stripes,
            STRIPES_YELLOW,
            STRIPES_BLACK,
        )),
        tile: 0.5,
    };
    match id {
        visual::CRATE => Look {
            material: paint(
                if size > 0.33 {
                    [0.3, 0.8, 0.86]
                } else if size > 0.28 {
                    [1.0, 0.56, 0.2]
                } else {
                    [1.0, 0.78, 0.3]
                },
                0.0,
                0.6,
            ),
            texture: Some(crate_texture()),
            tile: 2.0 * size,
        },
        visual::BALL => {
            let (a, b) = if size < 0.22 {
                ([210, 40, 36], [242, 240, 236])
            } else if size < 0.28 {
                ([40, 86, 210], [242, 240, 236])
            } else {
                ([196, 160, 92], [242, 232, 206])
            };
            Look {
                material: paint([1.0; 3], 0.0, 0.35),
                texture: Some(Texture::checkerboard(64, 4, a, b)),
                tile: 1.0,
            }
        }
        visual::PLANK => stripes(4),
        visual::DOOR => panels([0.32, 0.46, 0.78]),
        visual::CHAIN_LINK => plain([0.78, 0.79, 0.82], 1.0, 0.28),
        visual::PENDULUM_BOB => plain([1.0, 0.74, 0.32], 1.0, 0.22),
        visual::BOWL => Look {
            material: paint([1.0; 3], 0.0, 0.55),
            texture: Some(Texture::checkerboard(
                128,
                4,
                [214, 218, 226],
                [150, 158, 172],
            )),
            tile: 0.5,
        },
        visual::RAMP => Look {
            material: paint([0.36, 0.52, 0.86], 0.4, 0.45),
            texture: Some(Texture::brushed_metal(128, [236, 238, 242])),
            tile: 1.0,
        },
        visual::GANTRY | visual::DOOR_POST => plain([0.22, 0.24, 0.28], 0.8, 0.4),
        visual::CURB => stripes(2),
        visual::FULCRUM => plain([0.16, 0.17, 0.2], 0.2, 0.7),
        visual::PLANETOID => Look {
            material: paint([1.0; 3], 0.0, 0.9),
            texture: Some(rock_texture()),
            tile: 1.0,
        },
        _ => plain([0.6, 0.62, 0.66], 0.0, 0.6),
    }
}

/// The meshes bodies and showcase statics are drawn with: one per (visual,
/// shape), built on first use and shared from then on (the renderer keys its
/// GPU copy by the `Arc`).
#[derive(Debug, Default)]
pub struct MeshCache {
    meshes: HashMap<String, Option<Arc<MeshData>>>,
}

impl MeshCache {
    /// An empty cache.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// How many meshes it holds.
    #[must_use]
    pub fn len(&self) -> usize {
        self.meshes.len()
    }

    /// Whether it holds none.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.meshes.is_empty()
    }

    /// The mesh for visual `id` on `shape`; `None` when there is nothing to
    /// draw (visual "none" on a ship's shape, an unknown asset).
    pub fn mesh(&mut self, id: u32, shape: &ShapeRef) -> Option<Arc<MeshData>> {
        if *shape == ShapeRef::Asset(space_scene::KESTREL) || id == visual::KESTREL {
            // The ship's own model (its collider is its low-detail proxy).
            return Some(crate::kestrel::proxy_mesh());
        }
        let key = format!("{id}/{shape:?}");
        self.meshes
            .entry(key)
            .or_insert_with(|| build_mesh(id, shape).map(Arc::new))
            .clone()
    }
}

fn build_mesh(id: u32, shape: &ShapeRef) -> Option<MeshData> {
    let look = look(id, shape);
    match shape {
        ShapeRef::Asset(asset) if *asset == space_scene::BOWL => {
            let catalog = space_scene::catalog();
            let geometry = catalog.get(*asset)?.mesh();
            bowl_mesh(geometry, 0.06, look.material, look.texture?)
        }
        ShapeRef::Asset(_) => None,
        _ => shape_mesh(shape, look.material, look.texture, look.tile),
    }
}

/// A body as this frame drew it (for the HUD and the harness).
#[derive(Clone, Debug, PartialEq)]
pub struct Drawn {
    /// Its guid.
    pub guid: EntityGuid,
    /// Its visual id.
    pub visual: u32,
    /// Where its root part is, in the scene (around the spawn point).
    pub place: WorldPos,
    /// Its speed, m/s (the root part's).
    pub speed: f32,
    /// How many rigid parts it has.
    pub parts: usize,
    /// Whether it is (at least partly) in the camera's view.
    pub in_view: bool,
    /// Which copy of it the shard sends ([`Source`]).
    pub source: Source,
    /// How far it is from the own ship, metres (what its distance step and
    /// its draw delay follow).
    pub distance: f64,
    /// How old its newest sample is, ticks ([`Replica::sample_age`]).
    pub sample_age: f64,
}

/// When the bodies of a frame are drawn ([`BodyRenderer::draw`]).
#[derive(Clone, Copy)]
pub struct When<'a> {
    /// The render time of the near range, ticks. A body further from the
    /// own ship is drawn further in the past ([`draw_time`]: it is sent less
    /// often).
    pub t: f64,
    /// Where the own ship is drawn, in the server's world (`None`: nothing
    /// to measure a distance from; everything is drawn at `t`).
    pub own: Option<WorldPos>,
    /// `carry(at, place)`: how far a body drawn as at tick `at` at `place`
    /// is to be moved to be where it is at the moment the sky is drawn at (a
    /// body resting on the Moon moves with the Moon, and is drawn a moment
    /// later than the Moon would otherwise put it); `[0; 3]` everywhere
    /// else.
    pub carry: &'a dyn Fn(f64, WorldPos) -> [f64; 3],
}

impl std::fmt::Debug for When<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("When")
            .field("t", &self.t)
            .field("own", &self.own)
            .finish_non_exhaustive()
    }
}

/// What a body was last drawn as, for [`LINGER`].
#[derive(Clone, Debug)]
struct Last {
    physics: Arc<Physical>,
    poses: Vec<Pose>,
    seen: Instant,
    /// Source, distance and sample age as last drawn.
    about: (Source, f64, f64),
}

/// Draws the bodies of a replica into a scene. Keeps the meshes and, for
/// [`LINGER`], the bodies the replica has just stopped reporting.
#[derive(Debug, Default)]
pub struct BodyRenderer {
    meshes: MeshCache,
    last: HashMap<EntityGuid, Last>,
}

/// Where a point of the server's world is drawn: as far from the scene's
/// `spawn` point as it is from the map's `anchor` (see `crate::app`).
fn place_of(spawn: WorldPos, anchor: Option<WorldPos>, pos: WorldPos) -> WorldPos {
    anchor.map_or(spawn, |anchor| {
        WorldPos::from_f64_metres_around(spawn, WorldPos::delta(anchor, pos))
    })
}

/// `place` + `offset` as seen from `camera`, metres.
fn relative(camera: &CameraPose, place: WorldPos, offset: [f64; 3]) -> [f32; 3] {
    let d = WorldPos::delta(camera.position, place);
    [0, 1, 2].map(|k| (d[k] + offset[k]) as f32)
}

/// Whether a sphere at camera-relative `center` of `radius` is in `camera`'s
/// view (a cone around its axis, as wide as the picture's corners).
fn in_view(camera: &CameraPose, center: [f32; 3], radius: f32) -> bool {
    let f = camera.forward;
    let along = center[0] * f[0] + center[1] * f[1] + center[2] * f[2];
    if along < -radius {
        return false;
    }
    let distance = center.iter().map(|c| c * c).sum::<f32>().sqrt();
    if distance <= radius {
        return true;
    }
    let half_v = camera.fov_y * 0.5;
    let half_corner = (half_v.tan() * (1.0 + camera.aspect * camera.aspect).sqrt()).atan();
    let off = (along / distance).clamp(-1.0, 1.0).acos();
    off - (radius / distance).clamp(0.0, 1.0).asin() <= half_corner
}

impl BodyRenderer {
    /// A renderer with empty caches.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The mesh cache (the showcase statics share it).
    pub fn meshes(&mut self) -> &mut MeshCache {
        &mut self.meshes
    }

    /// Forget the bodies held for [`LINGER`] (a scene change, leaving).
    pub fn forget(&mut self) {
        self.last.clear();
    }

    /// Draw every body of `replica` as at the render time `when` says into
    /// `scene` (models for the parts and pins, sphere instances for the
    /// coils), placed around the scene's `spawn` point. Returns what was
    /// drawn.
    pub fn draw(
        &mut self,
        replica: &Replica,
        when: When<'_>,
        spawn: WorldPos,
        now: Instant,
        scene: &mut Scene,
    ) -> Vec<Drawn> {
        let When { t, own, carry } = when;
        let camera = scene.camera;
        let anchor = replica.anchor();
        // Every body's poses first: a spring may end on another body.
        let mut bodies: Vec<(EntityGuid, Arc<Physical>, Vec<Pose>)> = Vec::new();
        for tracked in replica.bodies() {
            let at = draw_time(&tracked.history, t, own);
            let (Some(physics), Some(mut poses)) =
                (tracked.physics.clone(), pose_at(&tracked.history, at))
            else {
                continue;
            };
            for pose in &mut poses {
                let c = carry(at, pose.pos);
                if c != [0.0; 3] {
                    pose.pos = WorldPos::from_f64_metres_around(pose.pos, c);
                }
            }
            self.last.insert(
                tracked.ship.guid,
                Last {
                    physics: Arc::clone(&physics),
                    poses: poses.clone(),
                    seen: now,
                    about: (
                        tracked.source(),
                        replica.distance_to(tracked),
                        replica.sample_age(tracked),
                    ),
                },
            );
            bodies.push((tracked.ship.guid, physics, poses));
        }
        // Bodies the replica no longer reports: held a moment, then dropped.
        self.last
            .retain(|_, last| now.saturating_duration_since(last.seen) < LINGER);
        for (guid, last) in &self.last {
            if !bodies.iter().any(|(g, _, _)| g == guid) {
                bodies.push((*guid, Arc::clone(&last.physics), last.poses.clone()));
            }
        }
        bodies.sort_by_key(|(guid, _, _)| *guid);

        let mut drawn = Vec::with_capacity(bodies.len());
        for (guid, physics, poses) in &bodies {
            let mut seen = false;
            for (part, pose) in poses.iter().enumerate() {
                let (id, shape) = match part {
                    0 => (
                        physics.visual.unwrap_or(visual::NONE),
                        physics.body.as_ref().map(|b| &b.shape),
                    ),
                    i => match physics
                        .articulation
                        .as_ref()
                        .and_then(|a| a.parts.get(i - 1))
                    {
                        Some(p) => (p.visual.0, Some(&p.body.shape)),
                        None => continue,
                    },
                };
                let Some(shape) = shape else { continue };
                let Some(mesh) = self.meshes.mesh(id, shape) else {
                    continue;
                };
                let place = place_of(spawn, anchor, pose.pos);
                let position = relative(&camera, place, pose.residual.map(f64::from));
                seen |= in_view(
                    &camera,
                    position,
                    mesh.radius() + mesh.center().iter().map(|c| c.abs()).sum::<f32>(),
                );
                scene.models.push(ModelInstance {
                    mesh,
                    position,
                    rotation: quat_matrix(pose.quat),
                    scale: 1.0,
                });
            }
            self.pins(physics, poses, spawn, anchor, &camera, scene);
            self.springs(physics, poses, &bodies, spawn, anchor, &camera, scene);
            let root = &poses[0];
            let (source, distance, sample_age) = self
                .last
                .get(guid)
                .map_or((Source::Local, 0.0, 0.0), |last| last.about);
            drawn.push(Drawn {
                guid: *guid,
                visual: physics.visual.unwrap_or(visual::NONE),
                place: place_of(spawn, anchor, root.pos),
                speed: root.vel.iter().map(|v| v * v).sum::<f32>().sqrt(),
                parts: poses.len(),
                in_view: seen,
                source,
                distance,
                sample_age,
            });
        }
        drawn
    }

    /// A pin at every joint: a ball for a ball joint, a short axle along the
    /// hinge for a hinge.
    fn pins(
        &mut self,
        physics: &Physical,
        poses: &[Pose],
        spawn: WorldPos,
        anchor: Option<WorldPos>,
        camera: &CameraPose,
        scene: &mut Scene,
    ) {
        let Some(articulation) = &physics.articulation else {
            return;
        };
        for joint in &articulation.joints {
            let Some(pose) = poses.get(usize::from(joint.a)) else {
                continue;
            };
            let shape = match joint.a {
                0 => physics.body.as_ref().map(|b| &b.shape),
                i => articulation
                    .parts
                    .get(usize::from(i) - 1)
                    .map(|p| &p.body.shape),
            };
            let thickness = shape.map_or(0.05, |s| match s {
                ShapeRef::Capsule { radius, .. } | ShapeRef::Cylinder { radius, .. } => *radius,
                ShapeRef::Cuboid { half_extents } => {
                    half_extents.iter().fold(f32::MAX, |a, b| a.min(*b))
                }
                _ => 0.05,
            });
            let (pos, offset) = pose.point(joint.frame_a.translation);
            let place = place_of(spawn, anchor, pos);
            let position = relative(camera, place, offset);
            let steel = visual::CHAIN_LINK;
            let (pin_shape, rotation) = match joint.kind {
                JointKind::Revolute { .. } | JointKind::Prismatic { .. } => {
                    // The hinge axis: the joint frame's X, in the part's frame,
                    // then in the world.
                    let axis_local = mul3(&quat_matrix(joint.frame_a.rotation), [1.0, 0.0, 0.0]);
                    let half = shape.map_or(0.1, |s| extent_along(s, axis_local)) + 0.02;
                    let axis = mul3(&quat_matrix(pose.quat), axis_local);
                    (
                        ShapeRef::Cylinder {
                            half_height: half,
                            radius: (thickness * 0.9).clamp(0.015, 0.04),
                        },
                        y_onto(axis),
                    )
                }
                _ => (
                    ShapeRef::Ball {
                        radius: (thickness * 1.15).clamp(0.02, 0.08),
                    },
                    space_render::IDENTITY_ROTATION,
                ),
            };
            if let Some(mesh) = self.meshes.mesh(steel, &pin_shape) {
                scene.models.push(ModelInstance {
                    mesh,
                    position,
                    rotation,
                    scale: 1.0,
                });
            }
            // A joint to the world has its mount there too.
            if let JointTarget::World { .. } = joint.b {
                let mount = ShapeRef::Ball {
                    radius: (thickness * 1.4).clamp(0.03, 0.09),
                };
                if matches!(joint.kind, JointKind::Spherical | JointKind::Fixed) {
                    if let Some(mesh) = self.meshes.mesh(visual::GANTRY, &mount) {
                        scene.models.push(ModelInstance {
                            mesh,
                            position,
                            rotation: space_render::IDENTITY_ROTATION,
                            scale: 1.0,
                        });
                    }
                }
            }
        }
    }

    /// Every spring as a coil between its two ends.
    #[allow(clippy::too_many_arguments)]
    fn springs(
        &mut self,
        physics: &Physical,
        poses: &[Pose],
        bodies: &[(EntityGuid, Arc<Physical>, Vec<Pose>)],
        spawn: WorldPos,
        anchor: Option<WorldPos>,
        camera: &CameraPose,
        scene: &mut Scene,
    ) {
        for spring in &physics.springs.0 {
            let Some(pose) = poses.get(usize::from(spring.part)) else {
                continue;
            };
            let (pos, offset) = pose.point(spring.anchor);
            let a = relative(camera, place_of(spawn, anchor, pos), offset);
            let b = match spring.target {
                SpringTarget::World(p) => relative(camera, place_of(spawn, anchor, p), [0.0; 3]),
                SpringTarget::Entity {
                    guid,
                    part,
                    anchor: at,
                } => {
                    let Some(other) = bodies
                        .iter()
                        .find(|(g, _, _)| *g == guid)
                        .and_then(|(_, _, p)| p.get(usize::from(part)))
                    else {
                        continue;
                    };
                    let (pos, offset) = other.point(at);
                    relative(camera, place_of(spawn, anchor, pos), offset)
                }
            };
            scene
                .instances
                .extend(coil(a, b, COIL_RADIUS, COIL_TURNS, COIL_WIRE));
        }
    }
}

fn mul3(m: &[[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    [0, 1, 2].map(|r| (0..3).map(|k| m[r][k] * v[k]).sum())
}

/// How far a shape reaches along the unit `axis` (its frame) from its origin.
fn extent_along(shape: &ShapeRef, axis: [f32; 3]) -> f32 {
    match shape {
        ShapeRef::Cuboid { half_extents } => {
            (0..3).map(|k| (axis[k] * half_extents[k]).abs()).sum()
        }
        ShapeRef::Ball { radius } => *radius,
        other => size_of(other),
    }
}

/// The radius of a drawn spring coil, metres.
pub const COIL_RADIUS: f32 = 0.07;
/// The turns of a drawn spring coil.
pub const COIL_TURNS: u32 = 8;
/// The thickness (radius) of a drawn spring's wire, metres.
pub const COIL_WIRE: f32 = 0.012;
/// The most beads one coil is drawn with.
const MAX_BEADS: usize = 480;

/// A spring from `a` to `b` (camera-relative, metres) as a helix of steel
/// beads — `turns` turns of `radius` in a wire `wire` thick, with a short
/// straight lead at each end — so it visibly stretches and squeezes with the
/// distance between its ends. Close up the beads overlap into a wire; further
/// away they are spaced up to 2 mm per metre of distance apart (about a
/// pixel), so a coil across the room costs a fraction of one in front of the
/// camera.
#[must_use]
pub fn coil(a: [f32; 3], b: [f32; 3], radius: f32, turns: u32, wire: f32) -> Vec<GpuInstance> {
    let d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let length = d.iter().map(|c| c * c).sum::<f32>().sqrt();
    if !(length > 1e-4 && length.is_finite()) {
        return Vec::new();
    }
    let axis = d.map(|c| c / length);
    // Two directions across the axis.
    let helper = if axis[1].abs() < 0.9 {
        [0.0, 1.0, 0.0]
    } else {
        [1.0, 0.0, 0.0]
    };
    let u = {
        let c = [
            axis[1] * helper[2] - axis[2] * helper[1],
            axis[2] * helper[0] - axis[0] * helper[2],
            axis[0] * helper[1] - axis[1] * helper[0],
        ];
        let l = c.iter().map(|v| v * v).sum::<f32>().sqrt();
        c.map(|v| v / l)
    };
    let v = [
        axis[1] * u[2] - axis[2] * u[1],
        axis[2] * u[0] - axis[0] * u[2],
        axis[0] * u[1] - axis[1] * u[0],
    ];
    let lead = (length * 0.08).min(0.15);
    let body = (length - 2.0 * lead).max(0.0);
    let turns = turns.max(1) as f32;
    let per_turn = ((std::f32::consts::TAU * radius).powi(2) + (body / turns).powi(2)).sqrt();
    let middle = [0, 1, 2].map(|k| 0.5 * (a[k] + b[k]));
    let far = middle.iter().map(|c| c * c).sum::<f32>().sqrt();
    let spacing = (wire * 1.1).max(0.002 * far);
    let coil_beads = ((per_turn * turns / spacing).ceil() as usize).clamp(8, MAX_BEADS);
    let lead_beads = ((lead / spacing).ceil() as usize).max(1);
    let bead = |p: [f32; 3]| GpuInstance {
        center: [p[0], p[1], p[2], wire],
        albedo: [0.82, 0.83, 0.86, 1.0],
        emissive: [0.0, 0.0, 0.0, 1.0],
        roughness: 0.3,
        shape: SHAPE_SPHERE,
        transmission: 0.0,
        _pad1: 0.0,
    };
    let at = |s: f32, r: f32, angle: f32| {
        let (sin, cos) = angle.sin_cos();
        [0, 1, 2].map(|k| a[k] + axis[k] * s + (u[k] * cos + v[k] * sin) * r)
    };
    let mut out = Vec::with_capacity(coil_beads + 2 * lead_beads + 2);
    // The leads run from each end out to the coil's radius.
    for i in 0..=lead_beads {
        let f = i as f32 / lead_beads as f32;
        out.push(bead(at(lead * f, radius * f, 0.0)));
    }
    for i in 0..=coil_beads {
        let f = i as f32 / coil_beads as f32;
        out.push(bead(at(
            lead + body * f,
            radius,
            std::f32::consts::TAU * turns * f,
        )));
    }
    let end = std::f32::consts::TAU * turns;
    for i in 0..=lead_beads {
        let f = i as f32 / lead_beads as f32;
        out.push(bead(at(lead + body + lead * f, radius * (1.0 - f), end)));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use space_sim::ShipState;

    fn sample(tick: u64, x: f64, vel: f32) -> Sample {
        Sample {
            tick,
            parts: vec![ShipState {
                vel: [vel, 0.0, 0.0],
                ..ShipState::at_rest(WorldPos::from_f64_metres_around(
                    WorldPos::ZERO,
                    [x, 0.0, 0.0],
                ))
            }],
            residual: vec![[0.0; 3]],
        }
    }

    fn x(poses: &[Pose]) -> f64 {
        WorldPos::delta(WorldPos::ZERO, poses[0].pos)[0]
    }

    /// Between two samples the position is interpolated, past the newest it
    /// is carried on along the velocity for a bounded time, before the
    /// oldest it is held.
    #[test]
    fn poses_interpolate_extrapolate_a_little_and_hold() {
        let history: VecDeque<Sample> = [
            sample(10, 0.0, 3.0),
            sample(11, 0.1, 3.0),
            sample(12, 0.2, 3.0),
        ]
        .into();
        assert!((x(&pose_at(&history, 10.5).unwrap()) - 0.05).abs() < 1e-6);
        assert!((x(&pose_at(&history, 11.75).unwrap()) - 0.175).abs() < 1e-6);
        assert!((x(&pose_at(&history, 9.0).unwrap())).abs() < 1e-9);
        // One tick past the newest: one tick of 3 m/s further.
        let ahead = x(&pose_at(&history, 13.0).unwrap());
        assert!((ahead - 0.3).abs() < 1e-5, "{ahead}");
        // Far past it: only MAX_EXTRAPOLATION ticks' worth.
        let far = x(&pose_at(&history, 100.0).unwrap());
        let cap = 0.2 + 3.0 * MAX_EXTRAPOLATION * f64::from(space_sim::dt());
        assert!((far - cap).abs() < 1e-5, "{far}");
        assert!(pose_at(&VecDeque::new(), 1.0).is_none());
        // Orientation: halfway between two turns about Y is the middle turn.
        let q = nlerp([1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], 0.5);
        assert!(
            (q[0] - q[2]).abs() < 1e-6 && (q[0] - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-6
        );
        // The short way round: a negated quaternion is the same turn.
        let same = nlerp([1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0], 0.5);
        assert!((same[0].abs() - 1.0).abs() < 1e-6);
    }

    /// A coil spans its two ends, keeps its radius and turns, and its beads
    /// crowd together when it is squeezed and spread out when stretched.
    /// A moon on its orbit, sampled once a second: between two samples it
    /// goes round — along the velocities the samples carry — and does not
    /// cut the corner. Samples a tick apart are joined by the straight line
    /// they always were, and so are samples whose velocities do not explain
    /// the way between them.
    ///
    /// Mutation this catches: the straight line for every gap (`along`
    /// returning `linear`): the moon is then up to 47 cm inside its orbit
    /// and turns a corner of 46 degrees at every sample.
    #[test]
    fn sparse_samples_of_an_orbit_are_joined_along_their_velocities() {
        let (radius, omega) = (6.0f64, 0.8f64);
        let at = |tick: u64| {
            let a = omega * tick as f64 * f64::from(space_sim::dt());
            let speed = radius * omega;
            Sample {
                tick,
                parts: vec![ShipState {
                    vel: [(-speed * a.sin()) as f32, 0.0, (speed * a.cos()) as f32],
                    ..ShipState::at_rest(WorldPos::from_f64_metres_around(
                        WorldPos::ZERO,
                        [radius * a.cos(), 0.0, radius * a.sin()],
                    ))
                }],
                residual: vec![[0.0; 3]],
            }
        };
        let off_orbit = |history: &VecDeque<Sample>| {
            let mut worst = 0.0f64;
            let mut t = history[0].tick as f64;
            while t <= history[history.len() - 1].tick as f64 {
                let pose = pose_at(history, t).expect("a pose")[0];
                let d = WorldPos::delta(WorldPos::ZERO, pose.pos);
                let a = omega * t * f64::from(space_sim::dt());
                let e = [d[0] - radius * a.cos(), d[1], d[2] - radius * a.sin()];
                worst = worst.max(e.iter().map(|c| c * c).sum::<f64>().sqrt());
                t += 0.25;
            }
            worst
        };
        // Once a second (the coarsest step): within two centimetres.
        let sparse: VecDeque<Sample> = [0, 30, 60, 90].into_iter().map(at).collect();
        let worst = off_orbit(&sparse);
        assert!(
            worst < 0.02,
            "{worst:.3} m off the orbit at one sample a second"
        );
        // Five times a second: within a tenth of a millimetre.
        let finer: VecDeque<Sample> = (0..10).map(|i| at(i * 6)).collect();
        assert!(off_orbit(&finer) < 1e-4, "{}", off_orbit(&finer));
        // A tick apart: the straight line, to the bit.
        let near: VecDeque<Sample> = [at(0), at(1)].into_iter().collect();
        let mid = pose_at(&near, 0.5).expect("a pose")[0];
        let (a, b) = (near[0].parts[0].pos, near[1].parts[0].pos);
        let half = WorldPos::delta(a, b).map(|c| c * 0.5);
        assert_eq!(mid.pos, WorldPos::from_f64_metres_around(a, half));
        // Moved, not moving: two samples 300 m apart whose velocities say
        // "at rest" (what a planet carries) are joined by the straight line.
        let carried: VecDeque<Sample> = [sample(0, 0.0, 0.0), sample(30, 300.0, 0.0)]
            .into_iter()
            .collect();
        assert!((x(&pose_at(&carried, 7.5).unwrap()) - 75.0).abs() < 1e-6);
    }

    /// Something flying away at 40 m/s is drawn further and further in the
    /// past as it goes (the delay grows with the distance) — smoothly: the
    /// way it is drawn to move from one frame to the next changes by
    /// millimetres, although its samples, and so the newest position heard,
    /// step thirty times a second.
    ///
    /// Mutation this catches: taking the distance to the newest sample
    /// (`history.back()`) instead of to where the entity is drawn: the delay
    /// then steps with every tick and the drawn motion with it, 0.2 m to
    /// and fro at this speed.
    #[test]
    fn the_draw_delay_follows_the_drawn_distance_without_stepping() {
        let per_tick = 40.0 * f64::from(space_sim::dt());
        let own = Some(WorldPos::ZERO);
        let mut drawn: Vec<f64> = Vec::new();
        // Two frames a tick, with the history a client has at each: the
        // samples up to the tick two ahead of the near range's render time.
        for frame in 0..160 {
            let t = 4.0 + f64::from(frame) * 0.5;
            let newest = t.floor() as u64 + 2;
            let history: VecDeque<Sample> = (newest.saturating_sub(9)..=newest)
                .map(|tick| sample(tick, 5.0 + tick as f64 * per_tick, 40.0))
                .collect();
            let at = draw_time(&history, t, own);
            assert!(at <= t && at > t - 6.0, "{at} for {t}");
            drawn.push(x(&pose_at(&history, at).unwrap()));
        }
        let steps: Vec<f64> = drawn.windows(2).map(|w| w[1] - w[0]).collect();
        let worst = steps
            .windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .fold(0.0, f64::max);
        assert!(steps.iter().all(|s| *s > 0.0), "never backwards");
        assert!(
            worst < 0.02,
            "the drawn step changed by {worst:.3} m between two frames"
        );
        // Without an own ship there is nothing to measure from.
        let history: VecDeque<Sample> = [sample(0, 500.0, 0.0)].into_iter().collect();
        assert_eq!(draw_time(&history, 7.0, None), 7.0);
    }

    #[test]
    fn a_coil_runs_from_end_to_end() {
        let a = [0.0, 0.0, 0.0];
        for length in [0.5f32, 1.5, 3.0] {
            let b = [0.0, -length, 0.0];
            let beads = coil(a, b, COIL_RADIUS, COIL_TURNS, COIL_WIRE);
            assert!(
                beads.len() > 50 && beads.len() < MAX_BEADS + 20,
                "{}",
                beads.len()
            );
            let first = beads.first().unwrap().center;
            let last = beads.last().unwrap().center;
            assert!(first[1].abs() < 1e-5 && (last[1] + length).abs() < 1e-4);
            for bead in &beads {
                let r = (bead.center[0].powi(2) + bead.center[2].powi(2)).sqrt();
                assert!(r <= COIL_RADIUS + 1e-4, "{r}");
                assert!(bead.center[1] <= 1e-5 && bead.center[1] >= -length - 1e-4);
                assert_eq!(bead.center[3], COIL_WIRE);
            }
        }
        assert!(coil(a, a, COIL_RADIUS, COIL_TURNS, COIL_WIRE).is_empty());
    }

    /// Every visual of the scene catalogue has a look, and the bodies and
    /// statics of the demo area all get a mesh (built once, then shared).
    #[test]
    fn every_showcase_shape_gets_a_mesh_once() {
        let mut cache = MeshCache::new();
        let mut shapes: Vec<(u32, ShapeRef)> = Vec::new();
        for d in space_scene::body_defs(space_protocol::Scene::DemoArea) {
            shapes.push((d.visual.0, d.body.shape.clone()));
            if let Some((art, _, _)) = &d.articulation {
                for p in &art.parts {
                    shapes.push((p.visual.0, p.body.shape.clone()));
                }
            }
        }
        for s in space_scene::statics(space_protocol::Scene::DemoArea) {
            if s.visual.0 != visual::NONE {
                shapes.push((s.visual.0, s.shape.clone()));
            }
        }
        for (id, shape) in &shapes {
            let mesh = cache
                .mesh(*id, shape)
                .unwrap_or_else(|| panic!("no mesh for {id} {shape:?}"));
            mesh.validate().expect("valid");
            let again = cache.mesh(*id, shape).unwrap();
            assert!(Arc::ptr_eq(&mesh, &again), "built once");
        }
        assert!(cache.len() < shapes.len(), "same shapes share a mesh");
        for id in 0..=15 {
            let _ = look(id, &ShapeRef::Ball { radius: 0.3 });
        }
    }
}
