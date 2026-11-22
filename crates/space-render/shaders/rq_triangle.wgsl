// Ray-query probe shader: one primary ray per pixel against the TLAS, writing
// a hit or miss colour. Ported from spikes/rq-triangle (proven on RADV GFX1151
// and lavapipe); the scene AS is a TOP_LEVEL built by space-render.
//
// Canonical WGSL ray-query pattern (naga >= 30): drive rayQueryProceed to
// completion, commit triangle candidates with rayQueryConfirmIntersection,
// then read the committed intersection.

enable wgpu_ray_query;

@group(0) @binding(0) var scene: acceleration_structure;
@group(0) @binding(1) var out_image: texture_storage_2d<rgba8unorm, write>;

// All members vec4-aligned so WGSL/SPIR-V layouts agree exactly (a SPIR-V vec3
// occupies 16 bytes while a WGSL vec3 occupies 12). Mirrored by `RqParams` in
// src/gpu.rs: offsets 0/16/32/48/64 -> 80 bytes.
struct Params {
    hit_color: vec4<f32>,
    miss_color: vec4<f32>,
    camera_origin: vec4<f32>,
    image_size: vec4<f32>,
    tan_half_fov: f32,
}

@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= u32(params.image_size.x) || gid.y >= u32(params.image_size.y) {
        return;
    }
    // Pixel centre -> NDC -> view-space ray direction (camera looks down -Z).
    let uv = (vec2<f32>(f32(gid.x), f32(gid.y)) + vec2<f32>(0.5, 0.5)) / params.image_size.xy;
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let dir = normalize(
        vec3<f32>(ndc.x * params.tan_half_fov, ndc.y * params.tan_half_fov, -1.0),
    );

    var rq: ray_query;
    rayQueryInitialize(
        &rq,
        scene,
        RayDesc(RAY_FLAG_NONE, 0xffu, 0.0, 100000.0, params.camera_origin.xyz, dir),
    );

    // Closest-hit style traversal: commit every triangle candidate so the
    // committed intersection ends up being the closest hit.
    loop {
        if !rayQueryProceed(&rq) {
            break;
        }
        if rayQueryGetCandidateIntersection(&rq).kind == RAY_QUERY_INTERSECTION_TRIANGLE {
            rayQueryConfirmIntersection(&rq);
        }
    }

    var color = params.miss_color;
    if rayQueryGetCommittedIntersection(&rq).kind == RAY_QUERY_INTERSECTION_TRIANGLE {
        color = params.hit_color;
    }

    textureStore(out_image, vec2<i32>(gid.xy), color);
}
