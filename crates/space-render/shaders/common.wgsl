// Shared WGSL fragments, prepended to every shader by build.rs.
//
// Layout rule (hard constraint): a WGSL vec3 takes 16 bytes in SPIR-V, so all
// struct members here are vec4-sized or explicitly padded, and the Rust-side
// structs in src/gpu.rs mirror these offsets exactly (unit-tested in
// src/gpu.rs).

struct Instance {
    center: vec4<f32>,   // camera-relative world centre, w = radius
    albedo: vec4<f32>,   // rgb + metallic in .w
    emissive: vec4<f32>, // rgb radiance, a = GI ray contribution weight
    roughness: f32,
    shape: u32,          // 0 = sphere, 1 = box
    transmission: f32,   // 0 = opaque, > 0 = transmissive (glass) when refraction is on
    _pad1: f32,
}

//#if MESH
// One drawn part of a mesh model: where it is and what it is made of. The
// forward pass draws with it (`vs_mesh`), and a ray that hits a model finds its
// part's record at `instance_custom_data - 1 + geometry_index` (see
// `Tlas::write_instances`). Mirrored by `MeshInstanceGpu` in src/params.rs.
// (Only in the shader variants that deal with mesh models, see build.rs.)
struct MeshInstance {
    model_x: vec4<f32>,  // rows of the 3x4 model -> camera-relative world matrix
    model_y: vec4<f32>,
    model_z: vec4<f32>,
    albedo: vec4<f32>,   // rgb base-colour factor + metallic in .w
    emissive: vec4<f32>, // rgb radiance + roughness in .w
}
//#endif

//#if ENV
// The scene's environment: a sky-and-ground gradient. src/environment.rs has
// the same three formulas on the CPU and says what they mean. Colours are
// linear radiance with the intensity already multiplied in. Mirrored by
// `EnvironmentGpu` in src/params.rs. (Only in the shader variants that shade
// or composite, see build.rs.)
struct Environment {
    zenith: vec4<f32>,  // rgb from straight up; w > 0 when the scene has an environment
    horizon: vec4<f32>, // rgb from the horizon
    ground: vec4<f32>,  // rgb from straight down
}

fn env_blend(env: Environment, up: f32, down: f32) -> vec3<f32> {
    return env.horizon.rgb + (env.zenith.rgb - env.horizon.rgb) * up
        + (env.ground.rgb - env.horizon.rgb) * down;
}

// Radiance arriving from the unit direction `dir`.
fn env_radiance(env: Environment, dir: vec3<f32>) -> vec3<f32> {
    return env_blend(env, max(dir.y, 0.0), max(-dir.y, 0.0));
}

// The cosine-weighted average of max(direction.y, 0) over the hemisphere
// around a unit normal whose y component is `y`: 2/3 facing up, 0 facing down.
fn env_hemisphere(y: f32) -> f32 {
    let c = clamp(y, -1.0, 1.0);
    return 0.2122066 * ((3.14159265 - acos(c)) * c + sqrt(1.0 - c * c));
}

// Ambient light on a surface with unit normal `n` (irradiance over pi).
fn env_ambient(env: Environment, n: vec3<f32>) -> vec3<f32> {
    return env_blend(env, env_hemisphere(n.y), env_hemisphere(-n.y));
}

// The gradient as a surface of `roughness` reflects it along the unit mirror
// direction `r`: the sharp lookup for a mirror, the hemisphere average for a
// fully rough surface.
fn env_reflection(env: Environment, r: vec3<f32>, roughness: f32) -> vec3<f32> {
    let alpha = clamp(roughness * roughness, 0.0, 1.0);
    return env_blend(
        env,
        mix(max(r.y, 0.0), env_hemisphere(r.y), alpha),
        mix(max(-r.y, 0.0), env_hemisphere(-r.y), alpha),
    );
}
//#endif

// Integer hash (splitmix32 variant) for deterministic ray jitter.
fn hash32(x: u32) -> u32 {
    var z = x;
    z = (z ^ (z >> 16u)) * 0x21f0aaad;
    z = (z ^ (z >> 15u)) * 0x735a2d97;
    return z ^ (z >> 15u);
}

// Uniform float in [0, 1) from a hash draw.
fn hashf(x: u32) -> f32 {
    return f32(hash32(x) >> 8) * (1.0 / 16777216.0);
}
