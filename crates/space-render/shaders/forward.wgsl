// Forward PBR pass: instanced placeholder geometry (sphere or box per LOD
// level) with a metallic-roughness BRDF (Cook-Torrance GGX + Smith visibility),
// analytic point lights, 1-bounce probe GI as indirect diffuse, HDR output.
// Depth test is provided by the render pass depth attachment.

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

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> instances: array<Instance>;
@group(0) @binding(2) var gi_tex: texture_2d<f32>;
@group(0) @binding(3) var gi_sampler: sampler;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) albedo: vec3<f32>,
    @location(3) material: vec2<f32>, // x = metallic, y = roughness
    @location(4) emissive: vec3<f32>,
}

// Unit icosphere: 12 vertices, 20 triangles (60 indices). Positions are
// normalised, so the per-vertex normal equals the position.
fn sphere_vertex(index: u32) -> vec3<f32> {
    let phi = 1.618033988749895;
    let inv = 1.0 / sqrt(1.0 + phi * phi);
    var v = array<vec3<f32>, 12>(
        vec3(-inv, phi * inv, 0.0), vec3(inv, phi * inv, 0.0), vec3(-inv, -phi * inv, 0.0), vec3(inv, -phi * inv, 0.0),
        vec3(0.0, -inv, inv), vec3(0.0, inv, inv), vec3(0.0, -inv, -inv), vec3(0.0, inv, -inv),
        vec3(phi * inv, 0.0, -inv), vec3(phi * inv, 0.0, inv), vec3(-phi * inv, 0.0, -inv), vec3(-phi * inv, 0.0, inv),
    );
    return v[index];
}

fn sphere_index(index: u32) -> u32 {
    var f = array<u32, 60>(
        0u, 11u, 5u, 0u, 5u, 1u, 0u, 1u, 7u, 0u, 7u, 10u, 0u, 10u, 11u,
        1u, 5u, 9u, 5u, 11u, 4u, 11u, 10u, 2u, 10u, 7u, 6u, 7u, 1u, 8u,
        3u, 9u, 4u, 3u, 4u, 2u, 3u, 2u, 6u, 3u, 6u, 8u, 3u, 8u, 9u,
        4u, 9u, 5u, 5u, 8u, 1u, 2u, 4u, 11u, 6u, 2u, 7u, 8u, 6u, 10u,
    );
    return f[index];
}

// Unit cube: 36 vertices (6 per face: two triangles), each face with its own
// flat normal.
fn box_vertex(index: u32) -> vec3<f32> {
    // Wrapped so a 60-vertex draw (the sphere count) draws the box's 36
    // vertices and then harmlessly repeats it.
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

@vertex
fn vs_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VsOut {
    var out: VsOut;
    let inst = instances[ii];
    var local: vec3<f32>;
    var normal: vec3<f32>;
    if inst.shape == 0u {
        // Unit icosphere: the position is the normal.
        local = sphere_vertex(sphere_index(vi));
        normal = local;
    } else {
        local = box_vertex(vi);
        normal = box_normal(vi);
    }
    let world = inst.center.xyz + local * inst.center.w;
    out.world = world;
    out.normal = normal;
    out.albedo = inst.albedo.rgb;
    out.material = vec2(inst.albedo.w, inst.roughness);
    out.emissive = inst.emissive.rgb;
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

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
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
    return vec4(max(color, vec3(0.0)), 1.0);
}
