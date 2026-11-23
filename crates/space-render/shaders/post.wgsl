// Post pass: screen-space reflections (SSR) + refraction, a compute pass that
// runs after the forward pass(es) and before the tone map. The host does not
// dispatch it at all unless FEATURE_SSR or FEATURE_REFRACT is set.
//
// Inputs (all read with textureLoad, no samplers):
//   scene_tex    rgba16f  opaque HDR scene (already fogged by the forward pass,
//                          so reflected/refracted samples carry fog with them);
//                          alpha 1 where an opaque surface was drawn, 0 where
//                          there is none (empty, or only the painted background)
//   depth_tex    d32f     depth: nearest surface, glass included. Reversed
//                          (1 at the near plane, towards 0 with distance,
//                          exactly 0 = nothing drawn), the same view_proj as
//                          the forward pass.
//   material_tex rgba16f  r = metallic, g = roughness, ba = the nearest
//                          surface's shading normal (octahedral), glass included
//   glass_tex    rgba16f  rgb = lit glass colour, a = transmission (0 = no
//                          glass at this pixel); read only with FEATURE_REFRACT
// Output: out_image (rgba16f) = composited HDR colour for the tone map.
//
// Everything is in camera-relative world space (camera at the origin). Surface
// normals are the forward pass's interpolated shading normals from the
// material buffer, so a tessellated sphere reflects and refracts as a smooth
// sphere. (Rebuilding them from depth gave one normal per triangle: mirror
// balls showed their facets as shards.)

// Also read: specular_tex rgba8 (binding 6) — rgb = each surface's base
// colour (its specular colour F0 is 4 % grey mixed towards it by the
// metalness), a = the share of the environment its mirror direction sees. A
// surface the pass mirrors (a smooth metal) gets its whole mirror image here
// — what the ray finds on screen, the environment where it finds nothing —
// weighted by that colour, the forward pass having left it out.
//
// build.rs compiles this file twice. The variant with RQ (ray queries; used
// when the frame has glass and the device can trace rays) follows a ray
// through glass for real: see `through_glass`. The plain one bends the view
// once at the glass surface and looks the scene up a fixed distance along.
//#if RQ
enable wgpu_ray_query;
//#endif

struct Camera {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    position: vec4<f32>, // camera-relative world position, w = tan(fov_y / 2)
    forward: vec4<f32>,  // view direction, w = near plane
}

struct PostParams {
    camera: Camera,
    features: u32, // FEATURE_* bitmask (mirrors src/quality.rs)
    ssr_steps: u32,
    image_size: vec2<u32>,
    ssr_max_roughness: f32,
    refract_distance: f32, // metres the refracted ray travels behind the surface
    refract_eta: f32,      // IOR ratio entering the medium (air -> glass = 1 / 1.5)
    fog_density: f32,      // the forward pass's exp-squared fog density (0: none)
    environment: Environment, // the scene's surroundings (all zero: none)
    lights: array<Light, 8>,  // the scene's lights, as in the forward pass
    light_count: u32,
    glass_bounces: u32,       // reflections allowed inside glass before a ray is let out
    _pad1: u32,
    _pad2: u32,
}

struct Light {
    position: vec4<f32>, // xyz camera-relative, w = 0 when the slot is inactive
    color: vec4<f32>,    // rgb linear radiance, a = intensity
}

const FEATURE_SSR: u32 = 2u;
const FEATURE_REFRACT: u32 = 4u;

@group(0) @binding(0) var<uniform> post: PostParams;
@group(0) @binding(1) var scene_tex: texture_2d<f32>;
@group(0) @binding(2) var depth_tex: texture_depth_2d;
@group(0) @binding(3) var material_tex: texture_2d<f32>;
@group(0) @binding(4) var glass_tex: texture_2d<f32>;
@group(0) @binding(5) var out_image: texture_storage_2d<rgba16float, write>;
@group(0) @binding(6) var specular_tex: texture_2d<f32>;
//#if RQ
@group(0) @binding(7) var scene_as: acceleration_structure;
@group(0) @binding(8) var<storage, read> instances: array<Instance>;
@group(0) @binding(9) var<storage, read> mesh_instances: array<MeshInstance>;
//#endif

const METALLIC_MIN: f32 = 0.5;
const REFINE_STEPS: u32 = 6u;
// Refraction: a lookup closer to the frame's edge than this fraction of the
// frame eases back to the unrefracted view (it is about to run out of picture).
const REFRACT_EDGE_FADE: f32 = 0.05;

fn size_i() -> vec2<i32> {
    return vec2<i32>(post.image_size);
}

fn clamp_px(p: vec2<i32>) -> vec2<i32> {
    return clamp(p, vec2<i32>(0), size_i() - vec2<i32>(1));
}

fn load_depth(p: vec2<i32>) -> f32 {
    return textureLoad(depth_tex, clamp_px(p), 0);
}

// Camera-relative world position of pixel `p` at depth `d`.
fn world_at(p: vec2<i32>, d: f32) -> vec3<f32> {
    let uv = (vec2<f32>(p) + vec2<f32>(0.5)) / vec2<f32>(post.image_size);
    let h = post.camera.inv_view_proj * vec4(uv.x * 2.0 - 1.0, uv.y * 2.0 - 1.0, d, 1.0);
    // An empty pixel (d = 0) is infinitely far away: h.w is 0. Keep the result
    // finite (just very far) so a silhouette neighbour cannot inject inf/NaN.
    return h.xyz / max(h.w, 1e-12);
}

// Distance along the view axis (clip w) of the surface stored at NDC `ndc_xy`
// with depth `d`: the w of inv_view_proj * ndc is 1 / clip_w.
fn view_depth(ndc_xy: vec2<f32>, d: f32) -> f32 {
    let h = post.camera.inv_view_proj * vec4(ndc_xy, d, 1.0);
    return 1.0 / max(h.w, 1e-9);
}

// Inverse of `oct_encode` in forward.wgsl.
fn oct_decode(e: vec2<f32>) -> vec3<f32> {
    var n = vec3(e.x, e.y, 1.0 - abs(e.x) - abs(e.y));
    let t = max(-n.z, 0.0);
    n.x = n.x + select(t, -t, n.x >= 0.0);
    n.y = n.y + select(t, -t, n.y >= 0.0);
    return normalize(n);
}

// Shading normal of the surface at pixel `p`, oriented towards the camera.
// `view` is the incoming camera -> surface direction.
fn surface_normal(p: vec2<i32>, view: vec3<f32>) -> vec3<f32> {
    var n = oct_decode(textureLoad(material_tex, p, 0).ba);
    if dot(n, view) > 0.0 {
        n = -n;
    }
    return n;
}

fn pixel_of_uv(uv: vec2<f32>) -> vec2<i32> {
    return clamp_px(vec2<i32>(floor(uv * vec2<f32>(post.image_size))));
}

// Remaining parameter range before the clip-space segment `a -> b` leaves the
// screen (all clip inequalities are linear in t) or crosses the near plane.
fn exit_t(a: vec4<f32>, b: vec4<f32>) -> f32 {
    var t = 1.0;
    let near = post.camera.forward.w;
    var fa = array<f32, 5>(a.w - a.x, a.w + a.x, a.w - a.y, a.w + a.y, a.w - near);
    var fb = array<f32, 5>(b.w - b.x, b.w + b.x, b.w - b.y, b.w + b.y, b.w - near);
    for (var i = 0; i < 5; i = i + 1) {
        if fb[i] < 0.0 {
            t = min(t, max(fa[i], 0.0) / max(fa[i] - fb[i], 1e-9));
        }
    }
    return t;
}

struct SsrResult {
    color: vec3<f32>,
    confidence: f32, // 0 = miss
}

// Clip-space point of the ray segment `a -> b` at screen-space fraction `s`
// (the screen position is linear in `s`; the clip parameter is not, so the
// depth stays perspective-correct): u = s * wa / (wb - s * (wb - wa)).
fn ray_point(a: vec4<f32>, b: vec4<f32>, s: f32) -> vec4<f32> {
    let u = s * a.w / (b.w - s * (b.w - a.w));
    return mix(a, b, u);
}

// Signed depth of the ray point behind the stored surface at its pixel
// (> 0: the ray is behind the surface).
fn ray_diff(h: vec4<f32>) -> f32 {
    let ndc = h.xy / h.w;
    return h.w - view_depth(ndc, load_depth(pixel_of_uv(ndc * 0.5 + vec2(0.5))));
}

fn trace_ssr(pos: vec3<f32>, normal: vec3<f32>, dir: vec3<f32>, roughness: f32) -> SsrResult {
    var res: SsrResult;
    res.color = vec3(0.0);
    res.confidence = 0.0;
    let vp = post.camera.view_proj;
    let w0 = dot(pos - post.camera.position.xyz, post.camera.forward.xyz);
    let origin = pos + normal * (0.02 + 0.002 * w0);
    let reach = 64.0 + 4.0 * w0;
    let h0 = vp * vec4(origin, 1.0);
    let h1 = vp * vec4(origin + dir * reach, 1.0);
    let t_end = exit_t(h0, h1);
    if t_end <= 0.0 {
        return res;
    }
    // The segment from the origin to where the ray leaves the screen.
    let a = h0;
    let b = mix(h0, h1, t_end);
    // Depth tolerance grows with roughness (blurrier lobes accept more slack)
    // and with distance (depth precision).
    let thickness = (0.1 + 2.0 * roughness) * max(1.0, 0.02 * w0);
    let steps = max(post.ssr_steps, 1u);

    var s_prev = 0.0;
    var prev_diff = -1.0;
    for (var i = 1u; i <= steps; i = i + 1u) {
        let s = f32(i) / f32(steps);
        let diff = ray_diff(ray_point(a, b, s));
        if diff > 0.0 && prev_diff <= 0.0 {
            // The ray went behind a surface between s_prev and s: bisect.
            var lo = s_prev;
            var hi = s;
            for (var k = 0u; k < REFINE_STEPS; k = k + 1u) {
                let mid = 0.5 * (lo + hi);
                if ray_diff(ray_point(a, b, mid)) > 0.0 { hi = mid; } else { lo = mid; }
            }
            let hh = ray_point(a, b, hi);
            let uv = (hh.xy / hh.w) * 0.5 + vec2(0.5);
            let px = pixel_of_uv(uv);
            // Accept only a hit that is close to the surface; a ray that slips
            // behind a silhouette to far geometry converges on a big gap.
            if ray_diff(hh) < thickness {
                let edge = min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y));
                let edge_fade = clamp(edge / 0.1, 0.0, 1.0);
                let step_fade = 1.0 - smoothstep(0.75, 1.0, s);
                res.color = min(scene_at(px).rgb, vec3(64.0));
                res.confidence = edge_fade * step_fade;
                return res;
            }
        }
        prev_diff = diff;
        s_prev = s;
    }
    return res;
}

//#if RQ
// ---- glass, ray traced ---------------------------------------------------
//
// A ray through a piece of glass: bent where it enters (Snell's law, index
// 1.5), followed to where it leaves the glass, bent again there, or reflected
// back in when it meets the surface too flatly to get out (total internal
// reflection: up to `glass_bounces` times, then it is let out where it is). A
// ball therefore shows what is far behind it upside down, a slab shows it
// shifted, and a cube's faces mirror on the inside.
//
// Which piece of glass it is in, a ray query against the glass in the
// scene's acceleration structure says; where it leaves is then worked out
// from the sphere or box itself — the traced sphere is a polyhedron, whose
// facets would show in the picture (and whose near side lies *inside* the
// drawn surface, where the ray starts).
//
// What the ray finds when it is out: a second ray query, against everything
// that is not glass. If that point is in the frame — on screen, and nothing
// opaque between it and the camera (a third ray says) — its colour is the
// frame's: shaded in full. If not, it is shaded here, roughly: what it emits,
// its base colour (the factor, no texture) under the environment's ambient
// light (if one ray along its normal finds the sky) and the lights (no
// shadows), by its true normal if it is a sphere or
// a box and, if it is a mesh, as a floor when it is seen from above and as
// if it faced the ray otherwise. And where there is
// nothing at all, the environment along the ray; or black.
//
// Not modelled: the reflection lost at the far surface, colour or absorption
// of the glass, glass seen through glass (the second piece is skipped).
const MASK_GLASS: u32 = 2u;
const MASK_OPAQUE: u32 = 4u;
const GLASS_INDEX: f32 = 1.5;
const RAY_REACH: f32 = 1.0e7;

struct RayHit {
    t: f32, // distance along the ray; negative: nothing hit
    instance: u32,
    custom: u32,
    geometry: u32,
}

// The nearest thing with `mask` along the ray. `inside_too`: whichever way
// it faces (the far side of the glass a ray is in); otherwise only what the
// ray enters, as for shadow rays — a solid is not in its own way.
fn nearest(origin: vec3<f32>, dir: vec3<f32>, mask: u32, reach: f32, inside_too: bool) -> RayHit {
    var rq: ray_query;
    let flags = select(RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_CULL_BACK_FACING, RAY_FLAG_FORCE_OPAQUE, inside_too);
    rayQueryInitialize(&rq, scene_as, RayDesc(flags, mask, 0.0, reach, origin, dir));
    loop {
        if !rayQueryProceed(&rq) {
            break;
        }
    }
    let hit = rayQueryGetCommittedIntersection(&rq);
    if hit.kind == RAY_QUERY_INTERSECTION_NONE {
        return RayHit(-1.0, 0u, 0u, 0u);
    }
    return RayHit(hit.t, hit.instance_index, hit.instance_custom_data, hit.geometry_index);
}

// How far along the ray from `origin`, inside the sphere or box `inst`, its
// surface is. Not positive: the ray is not inside it.
fn way_out(inst: Instance, origin: vec3<f32>, dir: vec3<f32>) -> f32 {
    let oc = origin - inst.center.xyz;
    let r = inst.center.w;
    if inst.shape == 0u {
        let b = dot(oc, dir);
        let disc = b * b - (dot(oc, oc) - r * r);
        if disc < 0.0 {
            return -1.0;
        }
        return sqrt(disc) - b;
    }
    // A box: the nearest of the three faces the ray is heading for.
    let d = select(dir, vec3(1e-9), abs(dir) < vec3(1e-9));
    let far = (sign(d) * r - oc) / d;
    return min(far.x, min(far.y, far.z));
}

// The outward unit normal of the sphere or box `inst` at the point `at` of
// its surface.
fn instance_normal(inst: Instance, at: vec3<f32>) -> vec3<f32> {
    let local = at - inst.center.xyz;
    if inst.shape == 0u {
        return normalize(local);
    }
    let a = abs(local);
    if a.x >= a.y && a.x >= a.z {
        return vec3(sign(local.x), 0.0, 0.0);
    } else if a.y >= a.z {
        return vec3(0.0, sign(local.y), 0.0);
    }
    return vec3(0.0, 0.0, sign(local.z));
}

// A rough shading of the surface point `at` that the ray along `dir` hit
// (see above for what is in it).
fn rough_shade(hit: RayHit, at: vec3<f32>, dir: vec3<f32>) -> vec3<f32> {
    var base: vec3<f32>;
    var metallic: f32;
    var color: vec3<f32>;
    var n = -dir;
    if hit.custom == 0u {
        let inst = instances[hit.instance];
        base = inst.albedo.rgb;
        metallic = inst.albedo.w;
        color = inst.emissive.rgb;
        n = instance_normal(inst, at);
        if dot(n, dir) > 0.0 {
            n = -n;
        }
    } else {
        let part = mesh_instances[hit.custom - 1u + hit.geometry];
        base = part.albedo.rgb;
        metallic = part.albedo.w;
        color = part.emissive.rgb;
        // A mesh's normals are not at hand here. Seen from well above it is
        // most likely a floor; otherwise it is taken to face the ray.
        if dir.y < -0.3 {
            n = vec3(0.0, 1.0, 0.0);
        }
    }
    let diffuse = base * (1.0 - metallic);
    if post.environment.zenith.w > 0.0 {
        // The environment's light, if the sky is in sight from there (one
        // ray along the normal, as far as the forward pass's sky rays look).
        let step = 1e-3 * (1.0 + length(at - post.camera.position.xyz));
        if nearest(at + n * step, n, MASK_OPAQUE, 60.0, false).t < 0.0 {
            color = color + diffuse * env_ambient(post.environment, n)
                + base * metallic * env_reflection(post.environment, reflect(dir, n), 0.6);
        }
    }
    for (var i: u32 = 0u; i < post.light_count; i = i + 1u) {
        let to_light = post.lights[i].position.xyz - at;
        let d2 = dot(to_light, to_light);
        let ndl = max(dot(n, to_light / max(sqrt(d2), 1e-4)), 0.0);
        color = color
            + diffuse * post.lights[i].color.rgb * (post.lights[i].color.a / max(d2, 1.0)) * ndl * 0.31830987;
    }
    return color;
}

// The radiance that arrives along the ray `origin + s * dir` (s > 0) from
// everything that is not glass.
fn behind_glass(origin: vec3<f32>, dir: vec3<f32>) -> vec3<f32> {
    let hit = nearest(origin, dir, MASK_OPAQUE, RAY_REACH, false);
    if hit.t < 0.0 {
        if post.environment.zenith.w > 0.0 {
            return env_radiance(post.environment, dir);
        }
        return vec3(0.0);
    }
    let at = origin + dir * hit.t;
    // In the frame? Then the frame knows best.
    let h = post.camera.view_proj * vec4(at, 1.0);
    if h.w > post.camera.forward.w {
        let uv = (h.xy / h.w) * 0.5 + vec2(0.5);
        if all(uv >= vec2(0.0)) && all(uv < vec2(1.0)) {
            let eye = post.camera.position.xyz;
            let dist = length(at - eye);
            let seen = nearest(eye, (at - eye) / dist, MASK_OPAQUE, dist * 1.01 + 0.05, false);
            if seen.t >= 0.0 && abs(seen.t - dist) <= 0.005 * dist + 0.01 {
                let texel = scene_at(pixel_of_uv(uv));
                if texel.a > 0.0 {
                    return texel.rgb;
                }
            }
        }
    }
    return rough_shade(hit, at, dir);
}

// What is seen through the glass surface at `pos` (unit normal `n`, towards
// the viewer) along the unit view direction `view`.
fn through_glass(pos: vec3<f32>, view: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    let step = 1e-3 * (1.0 + length(pos - post.camera.position.xyz));
    var dir = refract(view, n, 1.0 / GLASS_INDEX);
    var origin = pos - n * step;
    var out_at = origin;
    var out_dir = dir;
    let glass = nearest(origin, dir, MASK_GLASS, RAY_REACH, true);
    if glass.t >= 0.0 && glass.custom == 0u {
        let inst = instances[glass.instance];
        for (var bounce: u32 = 0u; bounce <= post.glass_bounces; bounce = bounce + 1u) {
            let t = way_out(inst, origin, dir);
            if t <= 0.0 {
                break;
            }
            let at = origin + dir * t;
            // (Outwards: the way the ray is going.)
            let m = instance_normal(inst, at);
            out_at = at + m * step;
            let bent = refract(dir, -m, GLASS_INDEX);
            if dot(bent, bent) > 0.0 {
                out_dir = bent;
                break;
            }
            // Too flat to get out: reflected back in.
            dir = reflect(dir, m);
            origin = at - m * step;
            out_dir = dir;
        }
    }
    return behind_glass(out_at, out_dir);
}
//#endif

// ---- the sky's share, smoothed ---------------------------------------------
//
// With sky rays, each pixel of the forward pass has counted how many of its
// few rays reached the sky and lit its surface by that share of the
// environment. A pixel's rays are turned by its place in a 4 x 4 tile of the
// frame (`sky_share` in forward.wgsl), so neighbours on the same surface have
// asked sixteen times as many directions between them; here their answers
// are averaged — over the 4 x 4 block around a pixel, counting only pixels of
// the same surface (the same plane, within a few centimetres, and the same
// normal, within 25 degrees) — and the pixel's light is corrected by the
// difference. No random numbers anywhere: a frame is the same every time,
// and because any 4 x 4 block holds each of the sixteen turns once, the
// average does not crawl when the camera moves.
//
// The share travels in the colour target's alpha: 0 = no surface, 1 = a
// surface, 1.25 + share / 2 = a surface that has counted.
const COUNTED: f32 = 1.2;

fn counted_share(alpha: f32) -> f32 {
    return clamp((alpha - 1.25) * 2.0, 0.0, 1.0);
}

// What to add to the colour of pixel `p` (whose alpha is `alpha`) to give it
// the smoothed share of the environment's light instead of its own count.
fn sky_correction(p: vec2<i32>, alpha: f32) -> vec3<f32> {
    // Under glass the buffers describe the glass, not what is behind it:
    // that keeps its own count.
    if (post.features & FEATURE_REFRACT) != 0u && textureLoad(glass_tex, p, 0).a > 0.0 {
        return vec3(0.0);
    }
    let d = load_depth(p);
    if d <= 0.0 {
        return vec3(0.0);
    }
    let eye = post.camera.position.xyz;
    let pos = world_at(p, d);
    let mat = textureLoad(material_tex, p, 0);
    let n = oct_decode(mat.ba);
    let reach = 0.02 + 0.01 * length(pos - eye);
    var sum = 0.0;
    var count = 0.0;
    for (var dy = -2; dy <= 1; dy = dy + 1) {
        for (var dx = -2; dx <= 1; dx = dx + 1) {
            let q = p + vec2(dx, dy);
            if any(q < vec2<i32>(0)) || any(q >= size_i()) {
                continue;
            }
            let a = textureLoad(scene_tex, q, 0).a;
            if a < COUNTED {
                continue;
            }
            let nq = oct_decode(textureLoad(material_tex, q, 0).ba);
            if dot(nq, n) < 0.9 || abs(dot(n, world_at(q, load_depth(q)) - pos)) > reach {
                continue;
            }
            sum = sum + counted_share(a);
            count = count + 1.0;
        }
    }
    let change = sum / max(count, 1.0) - counted_share(alpha);
    // The environment's light on this surface per unit of share, as the
    // forward pass adds it: the ambient light on its diffuse colour, and (for
    // a surface this pass does not mirror) the part of its reflection that
    // goes by the counted share rather than by the one ray ahead.
    let view = normalize(pos - eye);
    var nv = n;
    if dot(nv, view) > 0.0 {
        nv = -nv;
    }
    let metallic = mat.x;
    let roughness = mat.y;
    let base = textureLoad(specular_tex, p, 0).rgb;
    var light = env_ambient(post.environment, nv) * base * (1.0 - metallic);
    let mirrored = (post.features & FEATURE_SSR) != 0u && metallic >= METALLIC_MIN
        && roughness <= post.ssr_max_roughness;
    if !mirrored {
        let part = select(roughness * roughness, 1.0, roughness >= 0.6);
        light = light + part * env_reflection(post.environment, reflect(view, nv), roughness)
            * env_brdf(mix(vec3(0.04), base, metallic), roughness, clamp(dot(-view, nv), 0.0, 1.0));
    }
    let fog = length(pos - eye) * post.fog_density;
    return light * change * exp(-fog * fog);
}

// The opaque scene at pixel `p`: its colour, with the environment's light
// smoothed, and whether there is a surface (alpha 1) or not (0).
fn scene_at(p: vec2<i32>) -> vec4<f32> {
    let texel = textureLoad(scene_tex, p, 0);
    var color = texel.rgb;
    if texel.a >= COUNTED {
        color = max(color + sky_correction(p, texel.a), vec3(0.0));
    }
    return vec4(color, min(texel.a, 1.0));
}

fn fresnel(f0: f32, cos_theta: f32) -> f32 {
    return f0 + (1.0 - f0) * pow(1.0 - clamp(cos_theta, 0.0, 1.0), 5.0);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= post.image_size.x || gid.y >= post.image_size.y {
        return;
    }
    let p = vec2<i32>(gid.xy);
    var color = scene_at(p).rgb;
    let d = load_depth(p);
    // Reversed depth: 0 is the clear value, i.e. no surface at this pixel.
    if d > 0.0 {
        let pos = world_at(p, d);
        let view = normalize(pos - post.camera.position.xyz);

        var is_glass = false;
        if (post.features & FEATURE_REFRACT) != 0u {
            let glass = textureLoad(glass_tex, p, 0);
            if glass.a > 0.0 {
                is_glass = true;
                // Refraction: continue the refracted ray `refract_distance`
                // behind the surface and look the scene up where it lands.
                // If the frame has nothing to show for that place — it is off
                // screen, behind the camera, nearer than the glass, or no
                // surface was drawn there — the glass shows what is straight
                // behind it instead (always on screen): the view through it
                // loses its bend there, never its content.
                let n = surface_normal(p, view);
//#if RQ
                // (With ray queries none of that guessing: the ray is
                // followed through the glass, see `through_glass`.)
                let bg = through_glass(pos, view, n);
//#else
                let t = refract(view, n, post.refract_eta);
                let q = pos + t * post.refract_distance;
                let hq = post.camera.view_proj * vec4(q, 1.0);
                let straight = scene_at(p);
                var seen = straight;
                if hq.w > post.camera.forward.w {
                    let uvq = (hq.xy / hq.w) * 0.5 + vec2(0.5);
                    if all(uvq >= vec2(0.0)) && all(uvq < vec2(1.0)) {
                        let pq = pixel_of_uv(uvq);
                        // Behind the glass? Another glass pixel always is (the
                        // HDR target holds the opaque scene there); an opaque
                        // pixel must be farther than the glass surface.
                        // (Reversed depth: farther means a smaller value.)
                        let behind = textureLoad(glass_tex, pq, 0).a > 0.0 || load_depth(pq) <= d;
                        if behind {
                            let found = scene_at(pq);
                            // How far to trust it: not at all where no surface
                            // was drawn (alpha 0), and less and less towards
                            // the frame's edge, so that the view eases back to
                            // the straight one instead of cutting to it.
                            let edge = min(min(uvq.x, 1.0 - uvq.x), min(uvq.y, 1.0 - uvq.y));
                            let trust = found.a * min(edge / REFRACT_EDGE_FADE, 1.0);
                            if trust >= 1.0 {
                                seen = found;
                            } else {
                                seen = mix(straight, found, trust);
                            }
                        }
                    }
                }
                var bg = seen.rgb;
                if post.environment.zenith.w > 0.0 {
                    // Still alpha 0: there is no surface straight behind the
                    // glass either. The ray leaves the scene into its
                    // surroundings, which an environment gives a colour along
                    // the refracted direction. (Without one, this is whatever
                    // the frame's empty pixels hold: black.)
                    bg = mix(env_radiance(post.environment, t), seen.rgb, seen.a);
                }
//#endif
                let f = fresnel(0.04, dot(-view, n));
                color = mix(bg, glass.rgb, clamp(f + (1.0 - glass.a), 0.0, 1.0));
            }
        }

        if !is_glass && (post.features & FEATURE_SSR) != 0u {
            let mat = textureLoad(material_tex, p, 0).rg;
            let metallic = mat.x;
            let roughness = mat.y;
            if metallic >= METALLIC_MIN && roughness <= post.ssr_max_roughness {
                // What this surface mirrors: what the ray finds on screen,
                // and where it finds nothing (or the surface is too rough
                // for a sharp image) the environment in that direction, as
                // far as the forward pass found it in sight. The forward
                // pass has left this surface's mirror image to us; whatever
                // it shows is weighted by the surface's own specular colour
                // (the same factor the forward pass uses for the
                // environment), so gold mirrors everything in gold. Fog lies
                // in front of the surface, so it dims the image as well.
                let n = surface_normal(p, view);
                let r = reflect(view, n);
                let hit = trace_ssr(pos, n, r, roughness);
                let lobe = 1.0 - smoothstep(0.5 * post.ssr_max_roughness, 1.2 * post.ssr_max_roughness, roughness);
                let found = clamp(hit.confidence * lobe, 0.0, 1.0);
                let specular = textureLoad(specular_tex, p, 0);
                var mirrored = hit.color * found;
                if post.environment.zenith.w > 0.0 && found < 1.0 {
                    // The environment in the mirror direction, blurred by
                    // the surface's roughness, if it is in sight ...
                    // (specular.a: 0 = the inside of a solid, no
                    // surroundings; 2/255 = an open surface that sees none of
                    // the environment that way; more = the share it sees.)
                    let share = select(specular.a, 0.0, specular.a < 0.012);
                    var rest = share * env_reflection(post.environment, r, roughness);
//#if RQ
                    // ... and with ray queries, for a smooth mirror, what is
                    // really there instead: the ray is followed off screen
                    // and what it hits is shaded as `behind_glass` does it
                    // (the environment where it hits nothing).
                    if specular.a >= 0.004 {
                        let step = 1e-3 * (1.0 + length(pos - post.camera.position.xyz));
                        let sharp = behind_glass(pos + n * step, r);
                        rest = mix(sharp, rest, clamp(roughness * 2.5, 0.0, 1.0));
                    }
//#endif
                    mirrored = mirrored + (1.0 - found) * rest;
                }
                let f0 = mix(vec3(0.04), specular.rgb, metallic);
                let weight = env_brdf(f0, roughness, clamp(dot(-view, n), 0.0, 1.0));
                let fog = length(pos - post.camera.position.xyz) * post.fog_density;
                color = color + mirrored * weight * exp(-fog * fog);
            }
        }
    }
    textureStore(out_image, p, vec4(color, 1.0));
}
