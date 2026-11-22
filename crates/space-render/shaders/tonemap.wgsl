// Tone map + resolve: HDR rgba16f scene -> rgba8 display image (compute, so it
// works identically offscreen and in the swapchain path). ACES filmic curve.

struct ToneParams {
    exposure: f32,
    // Scalar padding (not vec3): a WGSL vec3 is 16-byte aligned, which would
    // push image_size to offset 32 and disagree with the Rust mirror (offset 16).
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
    image_size: vec2<u32>,
    _pad3: f32,
    _pad4: f32,
}

@group(0) @binding(0) var hdr_tex: texture_2d<f32>;
@group(0) @binding(1) var hdr_sampler: sampler;
@group(0) @binding(2) var out_image: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(3) var<uniform> tone: ToneParams;

// Narkowicz ACES filmic approximation.
fn aces(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((x * (a * x + vec3(b))) / (x * (c * x + vec3(d)) + vec3(e)), vec3(0.0), vec3(1.0));
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= tone.image_size.x || gid.y >= tone.image_size.y {
        return;
    }
    let uv = (vec2<f32>(f32(gid.x), f32(gid.y)) + vec2<f32>(0.5, 0.5))
        / vec2<f32>(f32(tone.image_size.x), f32(tone.image_size.y));
    let hdr = textureSampleLevel(hdr_tex, hdr_sampler, uv, 0.0).rgb;
    let mapped = aces(hdr * tone.exposure);
    // sRGB encode (approximate: gamma 2.2).
    let srgb = pow(mapped, vec3(1.0 / 2.2));
    textureStore(out_image, vec2<i32>(gid.xy), vec4(srgb, 1.0));
}
