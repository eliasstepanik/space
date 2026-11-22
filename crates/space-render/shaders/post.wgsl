// Post pass: screen-space reflections (SSR) + refraction, a compute pass that
// runs after the forward pass(es) and before the tone map. The host does not
// dispatch it at all unless FEATURE_SSR or FEATURE_REFRACT is set.
//
// Inputs (all read with textureLoad, no samplers):
//   scene_tex    rgba16f  opaque HDR scene (already fogged by the forward pass,
//                          so reflected/refracted samples carry fog with them)
//   depth_tex    d32f     depth: nearest surface, glass included. Vulkan 0..1,
//                          the same view_proj as the forward pass.
//   material_tex rg8      r = metallic, g = roughness (opaque surfaces only)
//   glass_tex    rgba16f  rgb = lit glass colour, a = transmission (0 = no
//                          glass at this pixel); read only with FEATURE_REFRACT
// Output: out_image (rgba16f) = composited HDR colour for the tone map.
//
// Everything is in camera-relative world space (camera at the origin). Surface
// normals are rebuilt from the depth buffer's neighbouring pixels (face
// normals, picking the smaller-delta neighbour on each axis at silhouettes).

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
    _pad0: f32,
}

const FEATURE_SSR: u32 = 2u;
const FEATURE_REFRACT: u32 = 4u;

@group(0) @binding(0) var<uniform> post: PostParams;
@group(0) @binding(1) var scene_tex: texture_2d<f32>;
@group(0) @binding(2) var depth_tex: texture_depth_2d;
@group(0) @binding(3) var material_tex: texture_2d<f32>;
@group(0) @binding(4) var glass_tex: texture_2d<f32>;
@group(0) @binding(5) var out_image: texture_storage_2d<rgba16float, write>;

const METALLIC_MIN: f32 = 0.5;
const REFINE_STEPS: u32 = 6u;

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
    return h.xyz / h.w;
}

// Distance along the view axis (clip w) of the surface stored at NDC `ndc_xy`
// with depth `d`: the w of inv_view_proj * ndc is 1 / clip_w.
fn view_depth(ndc_xy: vec2<f32>, d: f32) -> f32 {
    let h = post.camera.inv_view_proj * vec4(ndc_xy, d, 1.0);
    return 1.0 / max(h.w, 1e-9);
}

fn pick_shorter(a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
    let la = dot(a, a);
    let lb = dot(b, b);
    if la < 1e-12 { return b; }
    if lb < 1e-12 { return a; }
    if lb < la { return b; }
    return a;
}

// Face normal from depth, oriented towards the camera. `view` is the incoming
// camera -> surface direction.
fn surface_normal(p: vec2<i32>, pos: vec3<f32>, view: vec3<f32>) -> vec3<f32> {
    let pr = world_at(p + vec2<i32>(1, 0), load_depth(p + vec2<i32>(1, 0)));
    let pl = world_at(p + vec2<i32>(-1, 0), load_depth(p + vec2<i32>(-1, 0)));
    let pd = world_at(p + vec2<i32>(0, 1), load_depth(p + vec2<i32>(0, 1)));
    let pu = world_at(p + vec2<i32>(0, -1), load_depth(p + vec2<i32>(0, -1)));
    let dx = pick_shorter(pr - pos, pos - pl);
    let dy = pick_shorter(pd - pos, pos - pu);
    var n = normalize(cross(dx, dy));
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
    // Depth tolerance grows with roughness (blurrier lobes accept more slack)
    // and with distance (depth precision).
    let thickness = (0.1 + 2.0 * roughness) * max(1.0, 0.02 * w0);
    let steps = max(post.ssr_steps, 1u);

    var t_prev = 0.0;
    var prev_diff = -1.0;
    for (var i = 1u; i <= steps; i = i + 1u) {
        let t = t_end * f32(i) / f32(steps);
        let h = mix(h0, h1, t);
        let ndc = h.xy / h.w;
        let diff = h.w - view_depth(ndc, load_depth(pixel_of_uv(ndc * 0.5 + vec2(0.5))));
        if diff > 0.0 && prev_diff <= 0.0 {
            // The ray went behind a surface between t_prev and t: bisect.
            var lo = t_prev;
            var hi = t;
            for (var k = 0u; k < REFINE_STEPS; k = k + 1u) {
                let mid = 0.5 * (lo + hi);
                let hm = mix(h0, h1, mid);
                let nm = hm.xy / hm.w;
                let dm = hm.w - view_depth(nm, load_depth(pixel_of_uv(nm * 0.5 + vec2(0.5))));
                if dm > 0.0 { hi = mid; } else { lo = mid; }
            }
            let hh = mix(h0, h1, hi);
            let nh = hh.xy / hh.w;
            let uv = nh * 0.5 + vec2(0.5);
            let px = pixel_of_uv(uv);
            let dh = hh.w - view_depth(nh, load_depth(px));
            if dh < thickness {
                let edge = min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y));
                let edge_fade = clamp(edge / 0.1, 0.0, 1.0);
                let step_fade = 1.0 - smoothstep(0.75, 1.0, f32(i) / f32(steps));
                res.color = min(textureLoad(scene_tex, px, 0).rgb, vec3(64.0));
                res.confidence = edge_fade * step_fade;
                return res;
            }
        }
        prev_diff = diff;
        t_prev = t;
    }
    return res;
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
    var color = textureLoad(scene_tex, p, 0).rgb;
    let d = load_depth(p);
    if d < 1.0 {
        let pos = world_at(p, d);
        let view = normalize(pos - post.camera.position.xyz);

        var is_glass = false;
        if (post.features & FEATURE_REFRACT) != 0u {
            let glass = textureLoad(glass_tex, p, 0);
            if glass.a > 0.0 {
                is_glass = true;
                // Refraction: continue the refracted ray `refract_distance`
                // behind the surface and look the scene up where it lands.
                let n = surface_normal(p, pos, view);
                let t = refract(view, n, post.refract_eta);
                let q = pos + t * post.refract_distance;
                let hq = post.camera.view_proj * vec4(q, 1.0);
                var bg = color; // fall back to the undistorted background
                if hq.w > post.camera.forward.w {
                    let uvq = (hq.xy / hq.w) * 0.5 + vec2(0.5);
                    if all(uvq >= vec2(0.0)) && all(uvq < vec2(1.0)) {
                        let pq = pixel_of_uv(uvq);
                        // Behind the glass? Another glass pixel always is (the
                        // HDR target holds the opaque scene there); an opaque
                        // pixel must be farther than the glass surface.
                        let behind = textureLoad(glass_tex, pq, 0).a > 0.0 || load_depth(pq) >= d;
                        if behind {
                            bg = textureLoad(scene_tex, pq, 0).rgb;
                        }
                    }
                }
                let f = fresnel(0.04, dot(-view, n));
                color = mix(bg, glass.rgb, clamp(f + (1.0 - glass.a), 0.0, 1.0));
            }
        }

        if !is_glass && (post.features & FEATURE_SSR) != 0u {
            let mat = textureLoad(material_tex, p, 0).rg;
            let metallic = mat.x;
            let roughness = mat.y;
            if metallic >= METALLIC_MIN && roughness <= post.ssr_max_roughness {
                let n = surface_normal(p, pos, view);
                let r = reflect(view, n);
                let hit = trace_ssr(pos, n, r, roughness);
                if hit.confidence > 0.0 {
                    let f0 = mix(0.04, 0.9, metallic);
                    let f = fresnel(f0, dot(-view, n));
                    let lobe = 1.0 - smoothstep(0.5 * post.ssr_max_roughness, 1.2 * post.ssr_max_roughness, roughness);
                    let w = clamp(f * hit.confidence * lobe, 0.0, 1.0);
                    // Blend towards the reflection; a miss leaves `color`
                    // untouched (never darkens).
                    color = mix(color, hit.color, w);
                }
            }
        }
    }
    textureStore(out_image, p, vec4(color, 1.0));
}
