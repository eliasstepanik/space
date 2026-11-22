// Forward PBR pass: instanced placeholder geometry (sphere or box per LOD
// level) with a metallic-roughness BRDF (Cook-Torrance GGX + Smith visibility),
// analytic point lights, 1-bounce probe GI as indirect diffuse, HDR output.
// Depth test is provided by the render pass depth attachment.
//
// Outputs (opaque pass, `fs_main`): location 0 = HDR colour (alpha 1),
// location 1 = material buffer `rgba16float` (r = metallic, g = roughness,
// ba = the shading normal, octahedral-encoded) read by `post.wgsl`. With
// FEATURE_REFRACT, transmissive instances (transmission > 0) are skipped here
// and drawn by the glass pass (`vs_glass`/`fs_glass`): colour into a separate
// attachment (rgb = lit colour, a = transmission), so the opaque scene behind
// the glass survives in the HDR target for refraction, and their own material
// and normal over the opaque ones in the material buffer.

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
    _pad2: u32,
    _pad3: u32,
    _pad4: u32,
}

const FEATURE_FOG: u32 = 1u;
const FEATURE_REFRACT: u32 = 4u;

// One drawn part of a mesh model (`vs_mesh`): where it is and what it is made
// of. Mirrored by `MeshInstanceGpu` in src/params.rs.
struct MeshInstance {
    model_x: vec4<f32>,  // rows of the 3x4 model -> camera-relative world matrix
    model_y: vec4<f32>,
    model_z: vec4<f32>,
    albedo: vec4<f32>,   // rgb + metallic in .w
    emissive: vec4<f32>, // rgb radiance + roughness in .w
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> instances: array<Instance>;
@group(0) @binding(2) var gi_tex: texture_2d<f32>;
@group(0) @binding(3) var gi_sampler: sampler;
@group(0) @binding(4) var<storage, read> mesh_instances: array<MeshInstance>;

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

// The material buffer texel of a surface point.
fn material_out(in: VsOut) -> vec4<f32> {
    return vec4(in.material.x, clamp(in.material.y, 0.05, 1.0), oct_encode(normalize(in.normal)));
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

// Mesh models: real vertex buffers (position + normal) instead of the
// procedural shapes above. One indexed draw per material part, with
// `instance_index` selecting the part's `MeshInstance` record. The rotation in
// the model matrix is orthonormal with a uniform scale, so it transforms
// normals as well.
@vertex
fn vs_mesh(
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @builtin(instance_index) ii: u32,
) -> VsOut {
    let m = mesh_instances[ii];
    let p = vec4(position, 1.0);
    let world = vec3(dot(m.model_x, p), dot(m.model_y, p), dot(m.model_z, p));
    var out: VsOut;
    out.world = world;
    out.normal = vec3(dot(m.model_x.xyz, normal), dot(m.model_y.xyz, normal), dot(m.model_z.xyz, normal));
    out.albedo = m.albedo.rgb;
    out.material = vec2(m.albedo.w, m.emissive.w);
    out.emissive = m.emissive.rgb;
    out.transmission = 0.0;
    out.pos = params.camera.view_proj * vec4(world, 1.0);
    return out;
}

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
fn probe_radiance(world: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
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
    return textureSampleLevel(gi_tex, gi_sampler, uv, 0.0).rgb;
}

// Lit (and fogged) colour of one surface point.
fn shade(in: VsOut) -> vec3<f32> {
    let n = normalize(in.normal);
    let v = normalize(params.camera.position.xyz - in.world);
    let metallic = in.material.x;
    let roughness = clamp(in.material.y, 0.05, 1.0);

    var color = in.emissive;
    for (var i: u32 = 0u; i < params.light_count; i = i + 1u) {
        color = color + light_radiance(params.lights[i], in.world, n, v, in.albedo, metallic, roughness);
    }
    if params.gi_enabled == 1u {
        // Indirect diffuse: probe radiance along the surface normal's dominant
        // axis, Lambertian albedo, dielectric only (metallic is specular-only).
        let gi = probe_radiance(in.world, n);
        color = color + gi * in.albedo * (1.0 - metallic) / PI;
    }
    if (params.features & FEATURE_FOG) != 0u {
        // Exponential-squared fog on camera distance, after lighting and
        // before the tone map: f = 1 - exp(-(d * density)^2).
        let d = length(in.world - params.camera.position.xyz) * params.fog.w;
        let f = 1.0 - exp(-d * d);
        color = mix(color, params.fog.rgb, f);
    }
    return max(color, vec3(0.0));
}

@fragment
fn fs_main(in: VsOut) -> FsOut {
    var out: FsOut;
    out.color = vec4(shade(in), 1.0);
    out.material = material_out(in);
    return out;
}

@fragment
fn fs_glass(in: VsOut) -> FsOut {
    var out: FsOut;
    out.color = vec4(shade(in), in.transmission);
    out.material = material_out(in);
    return out;
}
