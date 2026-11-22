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
<<<<<<< HEAD
    transmission: f32,   // 0 = opaque, > 0 = transmissive (glass) when refraction is on
=======
    _pad0: f32,
>>>>>>> refs/remotes/origin/main
    _pad1: f32,
}

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
