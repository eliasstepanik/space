// The 2D overlay: solid, alpha-blended rectangles drawn straight onto the
// tone-mapped `rgba8unorm` output image, after everything else (see
// src/overlay.rs). Text is rectangles too: one per run of lit font dots.
//
// One instance per rectangle (`OverlayInstance` in src/params.rs): its corners
// already in normalised device coordinates, so the shader needs no uniforms,
// and its colour — display-referred sRGB with straight alpha, written to the
// 8-bit target as it is.

struct OverlayOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) color: vec4<f32>,
}

@vertex
fn vs_overlay(
    @builtin(vertex_index) vi: u32,
    @location(0) rect: vec4<f32>,  // x0, y0, x1, y1 (NDC; y = -1 is the top row)
    @location(1) color: vec4<f32>,
) -> OverlayOut {
    // Two triangles over the unit square.
    var corners = array<vec2<f32>, 6>(
        vec2(0.0, 0.0), vec2(1.0, 0.0), vec2(1.0, 1.0),
        vec2(0.0, 0.0), vec2(1.0, 1.0), vec2(0.0, 1.0),
    );
    let c = corners[vi % 6u];
    var out: OverlayOut;
    out.pos = vec4(mix(rect.xy, rect.zw, c), 0.0, 1.0);
    out.color = color;
    return out;
}

@fragment
fn fs_overlay(in: OverlayOut) -> @location(0) vec4<f32> {
    return in.color;
}
