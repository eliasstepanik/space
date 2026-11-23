// Forward PBR pass: instanced placeholder geometry (sphere or box per LOD
// level) with a metallic-roughness BRDF (Cook-Torrance GGX + Smith visibility),
// analytic point lights, 1-bounce probe GI as indirect diffuse, HDR output.
// Depth test is provided by the render pass depth attachment.
//
// Outputs (opaque pass, `fs_main`): location 0 = HDR colour, with alpha 1 —
// the target is cleared to alpha 0, so alpha says whether any opaque surface
// was drawn at a pixel (`post.wgsl` asks) —
// location 1 = material buffer `rgba16float` (r = metallic, g = roughness,
// ba = the shading normal on the side the fragment is seen from,
// octahedral-encoded) and location 2 = specular buffer `rgba8unorm` (rgb =
// the specular colour F0, a = the share of the environment the mirror
// direction sees), both read by `post.wgsl`. With
// FEATURE_REFRACT, transmissive instances (transmission > 0) are skipped here
// and drawn by the glass pass (`vs_glass`/`fs_glass`): colour into a separate
// attachment (rgb = lit colour, a = transmission), so the opaque scene behind
// the glass survives in the HDR target for refraction, and their own material
// and normal over the opaque ones in the material buffer.
//
// build.rs compiles this file in several variants. Lines between `//#if NAME`
// and `//#endif` (with an optional `//#else`) exist only in the variants that
// define NAME:
//   MESH     the mesh-model entry points `vs_mesh` / `fs_mesh` (vertex buffers,
//            a base-colour texture in descriptor set 1);
//   SHADOWS  ray-traced shadows: one ray query towards each light against the
//            scene's acceleration structure (binding 5);
//   SKY      the background entry points `vs_sky` / `fs_sky`: the scene's
//            environment drawn where nothing else was.
// The variants without SHADOWS contain no ray-query code at all, so a frame
// that casts no shadows runs exactly the shaders it always did.
//#if SHADOWS
enable wgpu_ray_query;
//#endif

struct Camera {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    position: vec4<f32>, // camera-relative world position, w = tan(fov_y / 2)
    forward: vec4<f32>,  // view direction, w = near plane
}

struct Light {
    position: vec4<f32>, // xyz camera-relative, w = 0 when the slot is inactive
    color: vec4<f32>,    // rgb linear radiance, a = intensity
}

struct Params {
    camera: Camera,
    lights: array<Light, 8>,
    light_count: u32,
    gi_enabled: u32,
    _pad0: vec2<f32>,
    gi_grid_origin: vec4<f32>, // camera-relative XZ of probe (0,0)
    gi_spacing: f32,
    gi_grid_size_x: u32,
    gi_grid_size_y: u32,
    _pad1: f32,
    fog: vec4<f32>, // rgb linear fog colour, w = exp-squared density (1/m)
    features: u32,  // FEATURE_* bitmask (mirrors src/quality.rs)
    sky_rays: u32,  // rays per pixel towards the sky (0 = none); SHADOWS variants only
    ssr_max_roughness: f32, // the post pass mirrors metals up to this roughness (with FEATURE_SSR)
    _pad3: u32,
    environment: Environment, // the scene's surroundings (all zero: none)
}

const FEATURE_FOG: u32 = 1u;
const FEATURE_SSR: u32 = 2u;
const FEATURE_REFRACT: u32 = 4u;
// The post pass mirrors what is at least this metallic (`METALLIC_MIN` there).
const MIRROR_METALLIC: f32 = 0.5;

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> instances: array<Instance>;
@group(0) @binding(2) var gi_tex: texture_2d<f32>;
@group(0) @binding(3) var gi_sampler: sampler;
//#if MESH
@group(0) @binding(4) var<storage, read> mesh_instances: array<MeshInstance>;
// The part's base-colour texture (sRGB, mip-mapped; 1x1 white for an
// untextured part) and its trilinear, repeating sampler.
@group(1) @binding(0) var base_color_tex: texture_2d<f32>;
@group(1) @binding(1) var base_color_sampler: sampler;
//#endif
//#if SHADOWS
@group(0) @binding(5) var scene_as: acceleration_structure;
//#endif

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) albedo: vec3<f32>,
    @location(3) material: vec2<f32>, // x = metallic, y = roughness
    @location(4) emissive: vec3<f32>,
    @location(5) @interpolate(flat) transmission: f32,
}

struct FsOut {
    @location(0) color: vec4<f32>,
    // r = metallic, g = roughness, ba = octahedral shading normal
    @location(1) material: vec4<f32>,
    // rgb = base colour, a = share of the environment in the mirror direction
    @location(2) specular: vec4<f32>,
}

// The glass pass has no specular buffer (the post pass refracts glass, it
// does not mirror it).
struct GlassOut {
    @location(0) color: vec4<f32>,
    @location(1) material: vec4<f32>,
}

// What `shade` makes of a surface point.
struct Shaded {
    color: vec3<f32>,
    // rgb = base colour, a = the share of the environment its mirror
    // direction sees (1 in the open).
    specular: vec4<f32>,
    // The colour target's alpha: 1 for a surface — or, where this pixel has
    // counted sky rays, 1.25 + half the share of them that got out, for the
    // post pass to smooth (it takes anything from 1 up as "a surface").
    coverage: f32,
}

// Octahedral encoding of a unit vector into [-1, 1]^2 (Cigolle et al. 2014).
fn oct_encode(n: vec3<f32>) -> vec2<f32> {
    let p = n.xy / max(abs(n.x) + abs(n.y) + abs(n.z), 1e-9);
    if n.z >= 0.0 {
        return p;
    }
    // Fold the lower hemisphere over the diagonals.
    let s = vec2(select(-1.0, 1.0, p.x >= 0.0), select(-1.0, 1.0, p.y >= 0.0));
    return (vec2(1.0) - abs(p.yx)) * s;
}

// The material buffer texel of a surface point seen on `side`.
fn material_out(in: VsOut, side: Side) -> vec4<f32> {
    return vec4(in.material.x, clamp(in.material.y, 0.05, 1.0), oct_encode(side.n));
}

// Which side of its surface a fragment shows. Nothing is culled, so a
// triangle is drawn from behind as well: the inside of a box the camera is
// in, the underside of a one-sided floor. That side is a surface of its own.
struct Side {
    // The unit shading normal, on the side the viewer is on.
    n: vec3<f32>,
    // Does the scene's surroundings (its environment) reach this side? Not
    // the inside of a closed solid: no sky in there.
    open: bool,
}

// The side of the surface at `world`, with vertex normal `normal`, that this
// fragment shows; `sheet` says the surface is an open one (both of whose
// sides are outsides) rather than the skin of a closed solid.
//
// Seen from behind means: the viewer is on the other side of the triangle
// than the one the normal points to. That is asked of the triangle's own
// plane, not of `dot(normal, view)` — on a smooth-shaded shape the
// interpolated normal tips past the silhouette while the triangle still
// faces the viewer — and without relying on the winding, which differs
// between the shapes here (see `cube_mesh` in src/scene.rs): the screen-space
// derivatives of the position span the plane, and because x runs to the right
// of the frame and y down it, their cross product points away from the
// viewer for every fragment there is. A normal on the same side as that is
// one the viewer sees from behind.
//
// (Derivatives: call this from uniform control flow.)
fn side_of(normal: vec3<f32>, world: vec3<f32>, sheet: bool) -> Side {
    let n = normalize(normal);
    let behind = dot(n, cross(dpdx(world), dpdy(world))) > 0.0;
    return Side(select(n, -n, behind), sheet || !behind);
}

// Unit icosphere: 12 vertices, 20 triangles (60 indices). Positions are
// normalised, so the per-vertex normal equals the position.
fn sphere_vertex(index: u32) -> vec3<f32> {
    let phi = 1.618033988749895;
    let inv = 1.0 / sqrt(1.0 + phi * phi);
    var v = array<vec3<f32>, 12>(
        vec3(-inv, phi * inv, 0.0), vec3(inv, phi * inv, 0.0), vec3(-inv, -phi * inv, 0.0), vec3(inv, -phi * inv, 0.0),
        vec3(0.0, -inv, phi * inv), vec3(0.0, inv, phi * inv), vec3(0.0, -inv, -phi * inv), vec3(0.0, inv, -phi * inv),
        vec3(phi * inv, 0.0, -inv), vec3(phi * inv, 0.0, inv), vec3(-phi * inv, 0.0, -inv), vec3(-phi * inv, 0.0, inv),
    );
    return v[index];
}

fn sphere_index(index: u32) -> u32 {
    var f = array<u32, 60>(
        0u, 11u, 5u, 0u, 5u, 1u, 0u, 1u, 7u, 0u, 7u, 10u, 0u, 10u, 11u,
        1u, 5u, 9u, 5u, 11u, 4u, 11u, 10u, 2u, 10u, 7u, 6u, 7u, 1u, 8u,
        3u, 9u, 4u, 3u, 4u, 2u, 3u, 2u, 6u, 3u, 6u, 8u, 3u, 8u, 9u,
        4u, 9u, 5u, 2u, 4u, 11u, 6u, 2u, 10u, 8u, 6u, 7u, 9u, 8u, 1u,
    );
    return f[index];
}

// Sphere detail levels. The instanced draw's first vertex carries the level in
// its top byte (`vertex_index` includes the draw's first vertex), so one
// pipeline draws every level: level L splits each icosahedron edge into 4^L
// segments (1, 4, 16, 64) and pushes the new vertices out onto the unit
// sphere. Level 0 is the plain icosahedron. Mirrored by `sphere_mesh` in
// src/scene.rs.
const LEVEL_SHIFT: u32 = 24u;
const VERTEX_MASK: u32 = 0xffffffu;

fn sphere_position(v: u32, level: u32) -> vec3<f32> {
    if level == 0u {
        return sphere_vertex(sphere_index(v));
    }
    let n = 1u << (2u * level);
    let per_face = n * n;
    let t = v / 3u;
    let face = (t / per_face) % 20u;
    let tri = t % per_face;
    // Row `row` of a face holds 2 * row + 1 triangles, so triangle `tri` is in
    // row floor(sqrt(tri)); the + 0.5 keeps the float sqrt off the integers.
    let row = u32(sqrt(f32(tri) + 0.5));
    let c = tri - row * row;
    let k = c / 2u;
    let corner = v % 3u;
    // Grid point (i, j): i rows down from corner a, j steps along the row.
    var i = row;
    var j = k;
    if (c & 1u) == 0u {
        // "up" triangle: (row, k), (row + 1, k), (row + 1, k + 1)
        if corner >= 1u { i = row + 1u; }
        if corner == 2u { j = k + 1u; }
    } else {
        // "down" triangle: (row, k), (row + 1, k + 1), (row, k + 1)
        if corner == 1u { i = row + 1u; }
        if corner >= 1u { j = k + 1u; }
    }
    let a = sphere_vertex(sphere_index(face * 3u));
    let b = sphere_vertex(sphere_index(face * 3u + 1u));
    let cc = sphere_vertex(sphere_index(face * 3u + 2u));
    let inv_n = 1.0 / f32(n);
    let wa = 1.0 - f32(i) * inv_n;
    let wb = f32(i - j) * inv_n;
    let wc = f32(j) * inv_n;
    return normalize(a * wa + b * wb + cc * wc);
}

// Unit cube: 36 vertices (6 per face: two triangles), each face with its own
// flat normal.
fn box_vertex(index: u32) -> vec3<f32> {
    let face = (index / 6u) % 6u;
    let corner = index % 6u;
    var n = array<vec3<f32>, 6>(
        vec3(1.0, 0.0, 0.0), vec3(-1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0),
        vec3(0.0, -1.0, 0.0), vec3(0.0, 0.0, 1.0), vec3(0.0, 0.0, -1.0),
    );
    let nn = n[face];
    var helper = vec3(1.0, 0.0, 0.0);
    if abs(nn.x) > 0.9 { helper = vec3(0.0, 1.0, 0.0); }
    let t = normalize(cross(nn, helper));
    let b = cross(nn, t);
    var corners = array<vec2<f32>, 6>(
        vec2(-1.0, -1.0), vec2(-1.0, 1.0), vec2(1.0, 1.0),
        vec2(-1.0, -1.0), vec2(1.0, 1.0), vec2(1.0, -1.0),
    );
    let c = corners[corner];
    return nn + t * c.x + b * c.y;
}

fn box_normal(index: u32) -> vec3<f32> {
    var n = array<vec3<f32>, 6>(
        vec3(1.0, 0.0, 0.0), vec3(-1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0),
        vec3(0.0, -1.0, 0.0), vec3(0.0, 0.0, 1.0), vec3(0.0, 0.0, -1.0),
    );
    return n[(index / 6u) % 6u];
}

// `glass_pass` selects which instances this draw keeps: the opaque pass drops
// transmissive instances when refraction is on, the glass pass keeps only
// those. A dropped instance collapses to one point (zero-area triangles
// rasterise nothing).
fn vertex_body(vi: u32, ii: u32, glass_pass: bool) -> VsOut {
    var out: VsOut;
    let inst = instances[ii];
    let level = vi >> LEVEL_SHIFT;
    let v = vi & VERTEX_MASK;
    let glass = inst.transmission > 0.0;
    let refract_on = (params.features & FEATURE_REFRACT) != 0u;
    // A box has 36 vertices; the draw it shares with level-0 spheres has 60,
    // so the surplus collapses like a dropped instance.
    let surplus = inst.shape != 0u && v >= 36u;
    if surplus || (glass_pass && !glass) || (!glass_pass && glass && refract_on) {
        out.world = vec3(0.0);
        out.normal = vec3(0.0, 0.0, 1.0);
        out.albedo = vec3(0.0);
        out.material = vec2(0.0);
        out.emissive = vec3(0.0);
        out.transmission = 0.0;
        out.pos = vec4(0.0, 0.0, 0.0, 1.0);
        return out;
    }
    var local: vec3<f32>;
    var normal: vec3<f32>;
    if inst.shape == 0u {
        // Unit sphere: the position is the normal.
        local = sphere_position(v, level);
        normal = local;
    } else {
        local = box_vertex(v);
        normal = box_normal(v);
    }
    let world = inst.center.xyz + local * inst.center.w;
    out.world = world;
    out.normal = normal;
    out.albedo = inst.albedo.rgb;
    out.material = vec2(inst.albedo.w, inst.roughness);
    out.emissive = inst.emissive.rgb;
    out.transmission = clamp(inst.transmission, 0.0, 1.0);
    out.pos = params.camera.view_proj * vec4(world, 1.0);
    return out;
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VsOut {
    return vertex_body(vi, ii, false);
}

@vertex
fn vs_glass(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VsOut {
    return vertex_body(vi, ii, true);
}

//#if MESH
// Mesh models: real vertex buffers (position, normal, texture coordinate)
// instead of the procedural shapes above. One indexed draw per material part,
// with `instance_index` selecting the part's `MeshInstance` record. The
// rotation in the model matrix is orthonormal with a uniform scale, so it
// transforms normals as well.
struct MeshVsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) albedo: vec3<f32>,   // the base-colour *factor*
    @location(3) material: vec2<f32>, // x = metallic, y = roughness
    @location(4) emissive: vec3<f32>,
    @location(5) uv: vec2<f32>,
    @location(6) @interpolate(flat) sheet: f32, // 1 = an open surface
}

@vertex
fn vs_mesh(
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @builtin(instance_index) ii: u32,
) -> MeshVsOut {
    let m = mesh_instances[ii];
    let p = vec4(position, 1.0);
    let world = vec3(dot(m.model_x, p), dot(m.model_y, p), dot(m.model_z, p));
    var out: MeshVsOut;
    out.world = world;
    out.normal = vec3(dot(m.model_x.xyz, normal), dot(m.model_y.xyz, normal), dot(m.model_z.xyz, normal));
    out.albedo = m.albedo.rgb;
    out.material = vec2(m.albedo.w, m.emissive.w);
    out.emissive = m.emissive.rgb;
    out.uv = uv;
    out.sheet = m.surface.x;
    out.pos = params.camera.view_proj * vec4(world, 1.0);
    return out;
}
//#endif

const PI: f32 = 3.14159265359;

// Cook-Torrance microfacet specular: GGX normal distribution.
fn distribution_ggx(noh: f32, alpha: f32) -> f32 {
    let a2 = alpha * alpha;
    let d = noh * noh * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
}

// Smith height-correlated GGX visibility term (approximation).
fn visibility_smith(nov: f32, nol: f32, alpha: f32) -> f32 {
    let a2 = alpha * alpha;
    let gv = nov * sqrt(nol * nol * (1.0 - a2) + a2);
    let gl = nol * sqrt(nov * nov * (1.0 - a2) + a2);
    return 0.5 / max(gv + gl, 1e-6);
}

fn fresnel_schlick(f0: vec3<f32>, cos_theta: f32) -> vec3<f32> {
    return f0 + (1.0 - f0) * pow(1.0 - cos_theta, 5.0);
}

fn light_radiance(light: Light, world: vec3<f32>, normal: vec3<f32>, view_dir: vec3<f32>,
                  albedo: vec3<f32>, metallic: f32, roughness: f32) -> vec3<f32> {
    let to_light = light.position.xyz - world;
    let dist2 = dot(to_light, to_light);
    let dist = sqrt(dist2);
    let l = to_light / max(dist, 1e-4);
    let n = normalize(normal);
    let v = view_dir;
    let h = normalize(v + l);
    let dl = dot(n, l);
    if dl <= 0.0 {
        return vec3(0.0);
    }
    let ndl = min(dl, 1.0);
    let ndv = min(max(dot(n, v), 1e-4), 1.0);
    let ndh = min(max(dot(n, h), 0.0), 1.0);
    let alpha = roughness * roughness;
    let f0 = mix(vec3(0.04), albedo, metallic);
    let f = fresnel_schlick(f0, max(dot(h, v), 0.0));
    let d = distribution_ggx(ndh, max(alpha, 0.0001));
    let vis = visibility_smith(ndv, ndl, max(alpha, 0.0001));
    let spec = f * (d * vis);
    let kd = (1.0 - f) * (1.0 - metallic);
    let falloff = light.color.a / max(dist2, 1.0);
    return light.color.rgb * falloff * (kd * albedo / PI + spec) * ndl;
}

// Nearest probe radiance for a surface: pick the probe cell containing the
// world XZ position and the texel of the normal's dominant axis (DDGI-lite).
// rgb = the indirect light arriving that way, a = how open to the sky the
// probe is in that direction (the share of its rays that hit nothing).
fn probe_radiance(world: vec3<f32>, normal: vec3<f32>) -> vec4<f32> {
    let gx = f32(params.gi_grid_size_x);
    let gy = f32(params.gi_grid_size_y);
    let ix = clamp(round((world.x - params.gi_grid_origin.x) / params.gi_spacing), 0.0, gx - 1.0);
    let iy = clamp(round((world.z - params.gi_grid_origin.z) / params.gi_spacing), 0.0, gy - 1.0);
    let a = abs(normal);
    var dir_idx: u32 = 2u;
    if a.y >= a.x && a.y >= a.z {
        dir_idx = select(2u, 3u, normal.y < 0.0);
    } else if a.x >= a.z {
        dir_idx = select(0u, 1u, normal.x < 0.0);
    } else {
        dir_idx = select(4u, 5u, normal.z < 0.0);
    }
    let uv = vec2((ix + 0.5) / gx, (iy * 6.0 + f32(dir_idx) + 0.5) / (gy * 6.0));
    return textureSampleLevel(gi_tex, gi_sampler, uv, 0.0);
}

//#if SHADOWS
// Shadow rays. Instance mask bit 0 marks a shadow caster (glass and a body
// with a light inside it are not). Every solid in the acceleration structure
// is single-sided for these rays: the faces a ray would *leave* a closed shape
// through are culled, so a surface point that sits a little inside its own
// ray-traced shape (the rasterised sphere is finer or coarser than the traced
// one; a smooth-shaded mesh near its terminator) is not shadowed by it, while
// anything between the point and the light is still entered from outside.
// Open meshes are flagged two-sided on their instance instead.
const SHADOW_MASK: u32 = 1u;
// Metres the ray starts off the surface (and stops short of the light), per
// metre of distance from the camera plus one: float precision scales with it.
const SHADOW_BIAS: f32 = 5e-4;

// Is the light at `light_pos` unoccluded as seen from the surface point
// `world`? `n` is the unit shading normal, on the light's side.
fn light_visible(world: vec3<f32>, n: vec3<f32>, light_pos: vec3<f32>) -> bool {
    let bias = SHADOW_BIAS * (1.0 + length(world - params.camera.position.xyz));
    let origin = world + n * bias;
    let to_light = light_pos - origin;
    let dist = length(to_light);
    if dist <= 2.0 * bias {
        return true;
    }
    var rq: ray_query;
    rayQueryInitialize(
        &rq,
        scene_as,
        RayDesc(
            RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_TERMINATE_ON_FIRST_HIT | RAY_FLAG_CULL_BACK_FACING,
            SHADOW_MASK,
            0.0,
            dist - bias,
            origin,
            to_light / dist,
        ),
    );
    // Everything is opaque for this ray and the first hit ends it, so there
    // are no candidates to confirm: proceed until the traversal is done.
    loop {
        if !rayQueryProceed(&rq) {
            break;
        }
    }
    return rayQueryGetCommittedIntersection(&rq).kind == RAY_QUERY_INTERSECTION_NONE;
}

// Sky rays: how much of the environment's light gets to a surface. The same
// rules as for shadow rays — what casts a shadow keeps the sky off, a solid
// is not in its own way — with a fixed reach (src/quality.rs has the twin).
const SKY_RANGE: f32 = 60.0;

// Is there nothing between the surface point `world` (unit normal `n`, on the
// viewer's side) and the sky in the unit direction `dir`?
fn sky_visible(world: vec3<f32>, n: vec3<f32>, dir: vec3<f32>) -> bool {
    let bias = SHADOW_BIAS * (1.0 + length(world - params.camera.position.xyz));
    var rq: ray_query;
    rayQueryInitialize(
        &rq,
        scene_as,
        RayDesc(
            RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_TERMINATE_ON_FIRST_HIT | RAY_FLAG_CULL_BACK_FACING,
            SHADOW_MASK,
            0.0,
            SKY_RANGE,
            world + n * bias,
            dir,
        ),
    );
    loop {
        if !rayQueryProceed(&rq) {
            break;
        }
    }
    return rayQueryGetCommittedIntersection(&rq).kind == RAY_QUERY_INTERSECTION_NONE;
}

// The share of the environment's diffuse light that reaches the surface
// point `world` with unit normal `n`: `count` rays over the hemisphere it
// faces, spread so that each stands for the same amount of light (a spiral
// in the cosine-weighted disc), and the share of them that got out.
//
// The directions are fixed in a frame built from the normal alone (the
// branchless one of Duff et al. 2017), turned and shifted by one of sixteen
// amounts that depends on where the pixel is in a 4 x 4 tile of the frame —
// an ordered pattern, not a random number: a pixel gets the same directions
// every frame. Any 4 x 4 block of pixels thus holds sixteen different sets,
// 16 x `count` directions in all, and the post pass, which averages the share
// over such a block of the same surface (`sky_correction` in post.wgsl),
// gets a smooth answer out of `count` rays a pixel.
fn sky_share(world: vec3<f32>, n: vec3<f32>, count: u32, pixel: vec2<f32>) -> f32 {
    var order = array<u32, 16>(0u, 8u, 2u, 10u, 12u, 4u, 14u, 6u, 3u, 11u, 1u, 9u, 15u, 7u, 13u, 5u);
    let k = order[(u32(pixel.y) & 3u) * 4u + (u32(pixel.x) & 3u)];
    let turn = f32(k) * 0.39269908;
    let shift = (f32((k * 7u + 3u) & 15u) + 0.5) / 16.0;
    let s = select(-1.0, 1.0, n.z >= 0.0);
    let a = -1.0 / (s + n.z);
    let b = n.x * n.y * a;
    let t0 = vec3(1.0 + s * n.x * n.x * a, s * b, -s * n.x);
    let t1 = vec3(b, s + n.y * n.y * a, -n.y);
    var open = 0.0;
    for (var i: u32 = 0u; i < count; i = i + 1u) {
        let r = sqrt((f32(i) + shift) / f32(count));
        let phi = f32(i) * 2.3999632 + turn;
        let dir = t0 * (r * cos(phi)) + t1 * (r * sin(phi)) + n * sqrt(1.0 - r * r);
        if sky_visible(world, n, dir) {
            open = open + 1.0;
        }
    }
    return open / f32(count);
}
//#endif

// Lit (and fogged) colour of one surface point, seen on `side`: lights,
// shadows, GI and the environment all see the surface the viewer sees.
// `mirrored` says the post pass may mirror this surface (it is not glass):
// if it will — a smooth metal, with screen-space reflections on — the
// environment's reflection is left to it, which blends it with what it finds
// on screen and weights both alike.
fn shade(in: VsOut, side: Side, mirrored: bool) -> Shaded {
    let n = side.n;
    let v = normalize(params.camera.position.xyz - in.world);
    let metallic = in.material.x;
    let roughness = clamp(in.material.y, 0.05, 1.0);

    var color = in.emissive;
    for (var i: u32 = 0u; i < params.light_count; i = i + 1u) {
//#if SHADOWS
        // The ray is only cast where the light would contribute at all (it is
        // on the surface's side), and the BRDF only counts where it arrives.
        let lit = light_radiance(params.lights[i], in.world, n, v, in.albedo, metallic, roughness);
        if dot(lit, vec3(1.0)) > 0.0 && light_visible(in.world, n, params.lights[i].position.xyz) {
            color = color + lit;
        }
//#else
        color = color + light_radiance(params.lights[i], in.world, n, v, in.albedo, metallic, roughness);
//#endif
    }
    let f0 = mix(vec3(0.04), in.albedo, metallic);
    let by_post = mirrored && (params.features & FEATURE_SSR) != 0u
        && metallic >= MIRROR_METALLIC && roughness <= params.ssr_max_roughness;
    // How much of the environment's light gets to this point: all of it,
    // unless something knows better.
    var sky = 1.0;
    if params.gi_enabled == 1u {
        // Indirect diffuse: probe radiance along the surface normal's dominant
        // axis, Lambertian albedo, dielectric only (metallic is specular-only).
        let gi = probe_radiance(in.world, n);
        color = color + gi.rgb * in.albedo * (1.0 - metallic) / PI;
        // The probes have looked: the nearest one is this open that way.
        sky = gi.a;
    }
    // ... and how much of its reflection: the same for a rough surface, what
    // is in the mirror direction for a smooth one.
    var sky_mirror = sky;
    // (What the post pass is told of the mirror direction, for a surface it
    // mirrors: the one ray's answer, not the counted share.)
    var ahead = sky;
    var coverage = 1.0;
//#if SHADOWS
    if params.sky_rays > 0u && side.open {
        // This pixel can look for itself.
        sky = sky_share(in.world, n, params.sky_rays, in.pos.xy);
        sky_mirror = sky;
        ahead = sky;
        if roughness < 0.6 {
            ahead = f32(sky_visible(in.world, n, reflect(-v, n)));
            sky_mirror = mix(ahead, sky, roughness * roughness);
        }
        if mirrored {
            coverage = 1.25 + 0.5 * sky;
        }
    }
//#endif
    if params.environment.zenith.w > 0.0 && side.open {
        // The environment (a scene without one stops at the test above): its
        // light from the hemisphere the surface faces, on the diffuse colour,
        // and its reflection along the mirror direction, blurred by roughness
        // and weighted by Fresnel. Shadows do not touch either — this is the
        // light that is left where no lamp reaches — and where the post pass
        // finds a screen-space reflection it blends that over this one.
        // The one place it does not reach is the inside of a closed solid.
        color = color + sky * env_ambient(params.environment, n) * in.albedo * (1.0 - metallic);
        if !by_post {
            let reflected = env_reflection(params.environment, reflect(-v, n), roughness);
            color = color
                + sky_mirror * reflected * env_brdf(f0, roughness, clamp(dot(n, v), 0.0, 1.0));
        }
    }
    if (params.features & FEATURE_FOG) != 0u {
        // Exponential-squared fog on camera distance, after lighting and
        // before the tone map: f = 1 - exp(-(d * density)^2). What a surface
        // fades into is what is behind it: the environment along the view
        // ray where that is painted as the background (so far geometry melts
        // into the sky at whatever height it stands, without a seam), the
        // fog colour otherwise.
        let d = length(in.world - params.camera.position.xyz) * params.fog.w;
        let f = 1.0 - exp(-d * d);
        var fog_color = params.fog.rgb;
        if params.environment.horizon.w > 0.0 {
            fog_color = env_radiance(params.environment, -v);
        }
        color = mix(color, fog_color, f);
    }
    // (Alpha 0 is kept for the inside of a solid, which mirrors no
    // surroundings at all; an open surface whose mirror direction is blocked
    // gets the smallest value above it, which the post pass reads as "none of
    // the environment, but ask what is there".)
    return Shaded(
        max(color, vec3(0.0)),
        vec4(clamp(in.albedo, vec3(0.0), vec3(1.0)), select(0.0, max(ahead, 0.008), side.open)),
        coverage,
    );
}

@fragment
fn fs_main(in: VsOut) -> FsOut {
    // (A sphere or box instance is a closed solid.)
    let side = side_of(in.normal, in.world, false);
    let shaded = shade(in, side, true);
    var out: FsOut;
    out.color = vec4(shaded.color, shaded.coverage);
    out.material = material_out(in, side);
    out.specular = shaded.specular;
    return out;
}

@fragment
fn fs_glass(in: VsOut) -> GlassOut {
    let side = side_of(in.normal, in.world, false);
    var out: GlassOut;
    out.color = vec4(shade(in, side, false).color, in.transmission);
    out.material = material_out(in, side);
    return out;
}

//#if SKY
// The background: the scene's environment wherever no geometry was drawn. One
// triangle over the whole frame at the far end of the (reversed) depth range,
// drawn after the opaque geometry with the depth test "equal": it only passes
// where the depth buffer still holds its clear value.
struct SkyOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) ndc: vec2<f32>,
}

@vertex
fn vs_sky(@builtin(vertex_index) vi: u32) -> SkyOut {
    // (-1, -1), (3, -1), (-1, 3): a triangle that covers the clip square.
    let xy = vec2(f32((vi << 1u) & 2u), f32(vi & 2u)) * 2.0 - vec2(1.0);
    var out: SkyOut;
    out.pos = vec4(xy, 0.0, 1.0);
    out.ndc = xy;
    return out;
}

@fragment
fn fs_sky(in: SkyOut) -> FsOut {
    // The view ray through this pixel: unproject any point on it.
    let h = params.camera.inv_view_proj * vec4(in.ndc, 0.5, 1.0);
    let dir = normalize(h.xyz / h.w - params.camera.position.xyz);
    var out: FsOut;
    // Alpha 0, like the clear colour: there is still no surface here.
    out.color = vec4(env_radiance(params.environment, dir), 0.0);
    out.material = vec4(0.0);
    out.specular = vec4(0.0);
    return out;
}
//#endif

//#if MESH
// A mesh fragment is shaded exactly like a procedural one, with the
// base-colour factor multiplied by the part's texture.
@fragment
fn fs_mesh(in: MeshVsOut) -> FsOut {
    // Sampled unconditionally: implicit derivatives need uniform control flow.
    let texel = textureSample(base_color_tex, base_color_sampler, in.uv);
    let side = side_of(in.normal, in.world, in.sheet > 0.5);
    var surface: VsOut;
    surface.pos = in.pos;
    surface.world = in.world;
    surface.normal = in.normal;
    surface.albedo = in.albedo * texel.rgb;
    surface.material = in.material;
    surface.emissive = in.emissive;
    surface.transmission = 0.0;
    let shaded = shade(surface, side, true);
    var out: FsOut;
    out.color = vec4(shaded.color, shaded.coverage);
    out.material = material_out(surface, side);
    out.specular = shaded.specular;
    return out;
}
//#endif
