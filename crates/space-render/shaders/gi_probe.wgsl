// Ray-query probe GI (DDGI-style, 1-bounce diffuse, docs/plan.md Phase 4).
//
// A Gx x Gy grid of probes sits in the camera-relative XZ plane at the camera
// height, spacing `spacing` metres. Each probe stores radiance for the six
// axis directions (+X,-X,+Y,-Y,+Z,-Z) as a (Gx, Gy*6) rgba16f texture. For
// each direction we cast `rays_per_probe` jittered rays into the TLAS and
// average the radiance of the closest hit — that is the 1-bounce indirect
// light the forward pass reads back as indirect diffuse. A surface a ray
// hits sends back what it emits and, in a scene with an environment, the sky
// light it reflects: its base colour times the environment's ambient light
// on it, if the sky is in sight from there (one more ray, along its normal).
//
// A ray that hits nothing has reached the sky. Its radiance is not added to
// the texel — the forward pass already has the environment's light on every
// surface, exactly, and would have it twice — but counted: the texel's alpha
// is the share of its rays that got out, which is how open the probe is in
// that direction. Where the forward pass has no rays of its own to ask, it
// lets that share of the environment's light through.
//
// The closest hit is either a procedural instance (its record is
// `instances[instance_index]`) or a part of a mesh model: a model's TLAS
// instance carries `1 + the index of its first part record` as its custom
// data (0 marks a procedural instance) and its BLAS has one geometry per part,
// so the part's record is `mesh_instances[custom - 1 + geometry_index]`.
//
// Determinism: ray jitter comes from hash32 over (probe, direction, ray)
// indices only, so a given scene produces bit-identical probe data on every
// run and on every device that implements ray query.

enable wgpu_ray_query;

@group(0) @binding(0) var scene: acceleration_structure;

struct GiParams {
    grid_origin: vec4<f32>, // camera-relative XZ of probe (0,0), w unused
    spacing: f32,           // probe spacing in metres
    probe_range: f32,       // ray max distance in metres
    grid_size: vec2<u32>,   // (Gx, Gy)
    rays_per_probe: u32,    // rays per stored direction
    _pad: f32,
    environment: Environment, // the scene's surroundings (all zero: none)
}

@group(0) @binding(1) var<uniform> gi: GiParams;
@group(0) @binding(2) var<storage, read> instances: array<Instance>;
@group(0) @binding(3) var gi_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var<storage, read> mesh_instances: array<MeshInstance>;

// The six stored directions, in texture-row order (row = iy * 6 + dir).
fn direction(index: u32) -> vec3<f32> {
    switch index {
        case 0u: { return vec3(1.0, 0.0, 0.0); }
        case 1u: { return vec3(-1.0, 0.0, 0.0); }
        case 2u: { return vec3(0.0, 1.0, 0.0); }
        case 3u: { return vec3(0.0, -1.0, 0.0); }
        case 4u: { return vec3(0.0, 0.0, 1.0); }
        default: { return vec3(0.0, 0.0, -1.0); }
    }
}

// Orthonormal tangent frame around `d` (helper axis = least-aligned axis, so
// `cross(d, helper)` is never degenerate — for an axis-aligned `d` the most-
// aligned axis would give a zero cross product and a NaN frame).
fn tangent_frame(d: vec3<f32>) -> mat3x3<f32> {
    var helper = vec3(1.0, 0.0, 0.0);
    if abs(d.x) <= abs(d.y) && abs(d.x) <= abs(d.z) { helper = vec3(1.0, 0.0, 0.0); }
    else if abs(d.y) <= abs(d.x) && abs(d.y) <= abs(d.z) { helper = vec3(0.0, 1.0, 0.0); }
    else { helper = vec3(0.0, 0.0, 1.0); }
    let t = normalize(cross(d, helper));
    let b = cross(d, t);
    return mat3x3(t, b, d);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= gi.grid_size.x || gid.y >= gi.grid_size.y {
        return;
    }
    let probe_pos = gi.grid_origin.xyz + vec3(f32(gid.x) * gi.spacing, 0.0, f32(gid.y) * gi.spacing);

    // Declared at function scope: naga emits an invalid OpConstantNull for a
    // `ray_query` object whose declaration is inside a loop body.
    var rq: ray_query;
    var sky_rq: ray_query;
    let has_sky = gi.environment.zenith.w > 0.0;

    for (var dir_idx: u32 = 0u; dir_idx < 6u; dir_idx = dir_idx + 1u) {
        let d = direction(dir_idx);
        let frame = tangent_frame(d);
        var sum = vec3(0.0, 0.0, 0.0);
        var open = 0.0;
        for (var r: u32 = 0u; r < gi.rays_per_probe; r = r + 1u) {
            let seed = ((gid.y * gi.grid_size.x + gid.x) * 6u + dir_idx) * 4096u + r;
            // Uniform disk sample -> cone direction around d (cos spread ~0.5).
            let a = hashf(seed * 2u + 1u) * 6.28318530718;
            let rr = sqrt(hashf(seed * 2u + 2u)) * 0.5;
            let local = vec3(cos(a) * rr, sin(a) * rr, 1.0);
            let dir = normalize(frame * normalize(local));

            rayQueryInitialize(
                &rq,
                scene,
                RayDesc(RAY_FLAG_NONE, 0xffu, 0.01, gi.probe_range, probe_pos + dir * 0.05, dir),
            );
            loop {
                if !rayQueryProceed(&rq) {
                    break;
                }
                if rayQueryGetCandidateIntersection(&rq).kind == RAY_QUERY_INTERSECTION_TRIANGLE {
                    rayQueryConfirmIntersection(&rq);
                }
            }
            let hit = rayQueryGetCommittedIntersection(&rq);
            if hit.kind == RAY_QUERY_INTERSECTION_TRIANGLE {
                // What the surface is made of, and which way it faces: known
                // for a sphere or a box; a mesh is taken to face the ray.
                var diffuse: vec3<f32>;
                var n = -dir;
                let at = probe_pos + dir * (0.05 + hit.t);
                if hit.instance_custom_data == 0u {
                    let inst = instances[hit.instance_index];
                    sum = sum + inst.emissive.rgb * inst.emissive.a;
                    diffuse = inst.albedo.rgb * (1.0 - inst.albedo.w);
                    let local = at - inst.center.xyz;
                    if inst.shape == 0u {
                        n = normalize(local);
                    } else {
                        let a = abs(local);
                        if a.x >= a.y && a.x >= a.z {
                            n = vec3(sign(local.x), 0.0, 0.0);
                        } else if a.y >= a.z {
                            n = vec3(0.0, sign(local.y), 0.0);
                        } else {
                            n = vec3(0.0, 0.0, sign(local.z));
                        }
                    }
                } else {
                    // A mesh part: what it emits (nothing, for most: the ray
                    // is then simply blocked by the model).
                    let part = mesh_instances[hit.instance_custom_data - 1u + hit.geometry_index];
                    sum = sum + part.emissive.rgb;
                    diffuse = part.albedo.rgb * (1.0 - part.albedo.w);
                }
                // The sky light it reflects — seen from outside (the inside
                // of a solid has no sky), and if the sky is in sight from it.
                if has_sky && dot(n, dir) < 0.0 {
                    rayQueryInitialize(
                        &sky_rq,
                        scene,
                        RayDesc(RAY_FLAG_TERMINATE_ON_FIRST_HIT, 0xffu, 0.0, gi.probe_range, at + n * 0.02, n),
                    );
                    loop {
                        if !rayQueryProceed(&sky_rq) {
                            break;
                        }
                        if rayQueryGetCandidateIntersection(&sky_rq).kind == RAY_QUERY_INTERSECTION_TRIANGLE {
                            rayQueryConfirmIntersection(&sky_rq);
                        }
                    }
                    if rayQueryGetCommittedIntersection(&sky_rq).kind == RAY_QUERY_INTERSECTION_NONE {
                        sum = sum + diffuse * env_ambient(gi.environment, n);
                    }
                }
            } else {
                open = open + 1.0;
            }
        }
        let rays = f32(gi.rays_per_probe);
        textureStore(
            gi_out,
            vec2<i32>(i32(gid.x), i32(gid.y) * 6 + i32(dir_idx)),
            vec4(sum / rays, open / rays),
        );
    }
}
