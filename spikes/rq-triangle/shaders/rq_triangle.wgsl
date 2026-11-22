// Phase-0 spike 0.3: ray-query compute shader, authored in WGSL and compiled to
// SPIR-V at build time by naga (see build.rs).
//
// For every pixel it traces one primary ray from a pinhole camera into the
// bottom-level acceleration structure and writes the triangle's colour on a hit
// or a distinct background colour on a miss.
//
// Canonical WGSL ray-query pattern (WGSL spec / naga >= 30): drive
// rayQueryProceed to completion, commit triangle candidates with
// rayQueryConfirmIntersection, then read the committed intersection. Driving
// proceed() until it returns false is required: it is what terminates the
// query (leaving a query active at invocation exit is undefined behaviour).

enable wgpu_ray_query;

@group(0) @binding(0) var scene: acceleration_structure;
@group(0) @binding(1) var out_image: texture_storage_2d<rgba8unorm, write>;

// All members are vec4 so the WGSL and SPIR-V layouts agree exactly (a SPIR-V
// vec3 occupies 16 bytes while a WGSL vec3 occupies 12, which would silently
// shift later members). Offsets: 0 hit_color, 16 miss_color, 32 camera_origin,
// 48 image_size, 64 tan_half_fov -> 80 bytes. Mirrored by `Params` in main.rs.
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
        RayDesc(RAY_FLAG_NONE, 0xffu, 0.0, 1000.0, params.camera_origin.xyz, dir),
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
