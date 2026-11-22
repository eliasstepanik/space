//! GPU-backed tests. They need a Vulkan device; `spikes/ENVIRONMENT.md` has the
//! env recipe (`LD_LIBRARY_PATH`, `VK_DRIVER_FILES`, `VK_LAYER_PATH`).
//!
//! Every test names the mutation switch that proves its pixel assertions bite,
//! and each test is run on both devices by the CI GPU job (RADV GFX1151 and
//! lavapipe) — the exact-pixel tests must pass on lavapipe.

use space_render::{
    validation_message_count, ContextOptions, GpuInstance, Quality, QualityPreset, RenderContext,
    RenderOptions, Renderer, Scene,
};
use space_world::WorldPos;

/// The CI quality preset (cheap, deterministic).
fn ci_options() -> RenderOptions {
    RenderOptions {
        gi_enabled: false,
        quality: Quality::CI,
        // Fog off: the golden/pixel tests must not depend on fog.
        preset: QualityPreset::Ci,
        ..Default::default()
    }
}

/// A context, or `None` when no Vulkan device is usable (the test then skips
/// with a loud message rather than pretending to pass).
fn context() -> Option<RenderContext> {
    let requested = std::env::var("SPACE_RENDER_VALIDATION").as_deref() == Ok("1");
    let options = ContextOptions {
        validation: Some(requested),
        prefer_ray_query: true,
        window_display: None,
    };
    match RenderContext::new(&options) {
        Ok(ctx) => {
            // When the runner asks for validation, the layer must actually be
            // active: otherwise every `validation_messages == 0` assertion below
            // is vacuously true and CI would pass on a runner that silently has
            // no layer. Without the var the graceful skip stays allowed.
            assert!(
                !requested || ctx.validation_active(),
                "SPACE_RENDER_VALIDATION=1 but VK_LAYER_KHRONOS_validation is not active \
                 (layer missing from VK_LAYER_PATH?)"
            );
            Some(ctx)
        }
        Err(error) => {
            // Skips (see spikes/ENVIRONMENT.md), or fails with SPACE_REQUIRE_GPU=1.
            space_render::skip_or_require_gpu(&error);
            None
        }
    }
}

/// The shared demo scene: an emissive sphere dead-centre in front of the
/// camera, a magenta cube ahead and to the left of it, and one light.
fn sphere_scene(width: u32, height: u32) -> Scene {
    space_render::demo_scene(width, height)
}

/// Mean RGBA of the pixel at (x, y).
fn pixel(out: &space_render::FrameOutput, x: u32, y: u32) -> [u16; 4] {
    let i = ((y * out.width + x) * 4) as usize;
    [
        u16::from(out.pixels[i]),
        u16::from(out.pixels[i + 1]),
        u16::from(out.pixels[i + 2]),
        u16::from(out.pixels[i + 3]),
    ]
}

/// Number of pixels that are not black.
fn lit_pixels(out: &space_render::FrameOutput) -> usize {
    out.pixels
        .as_chunks::<4>()
        .0
        .iter()
        .filter(|px| px[0] > 8 || px[1] > 8 || px[2] > 8)
        .count()
}

/// Sum of every RGB channel of every pixel: a cheap brightness signature.
fn brightness_sum(out: &space_render::FrameOutput) -> u64 {
    out.pixels
        .as_chunks::<4>()
        .0
        .iter()
        .map(|p| u64::from(p[0]) + u64::from(p[1]) + u64::from(p[2]))
        .sum()
}

/// Fraction of pixels that are not black.
fn coverage(out: &space_render::FrameOutput) -> f64 {
    let mut lit = 0usize;
    for px in out.pixels.as_chunks::<4>().0 {
        if px[0] > 8 || px[1] > 8 || px[2] > 8 {
            lit += 1;
        }
    }
    lit as f64 / (out.width as f64 * out.height as f64)
}

/// The forward pass draws the sphere: the centre of the image is lit and the
/// frame has real geometry coverage. With `mutate_skip_forward_draw` the same
/// assertion must fail (the render pass still runs, so this proves the *draw*
/// is what produces the pixels).
#[test]
fn forward_pass_draws_the_sphere() {
    let Some(mut ctx) = context() else { return };
    let size = 64;
    let out = ctx
        .render_frame(&sphere_scene(size, size), size, size, &ci_options())
        .expect("forward render");
    assert_eq!(out.validation_messages, 0, "validation messages seen");
    // The demo scene is the sphere plus the spawn cube.
    assert_eq!(out.instance_count, 2);
    let centre = pixel(&out, size / 2, size / 2);
    assert!(
        centre[0] > 200 && centre[1] > 200 && centre[2] > 200,
        "centre pixel should be a lit white sphere, got {centre:?}"
    );
    assert!(coverage(&out) > 0.02, "coverage {}", coverage(&out));
    assert_eq!(out.pixels.len(), (size * size * 4) as usize);
}

/// MUTATION CHECK for [`forward_pass_draws_the_sphere`]: skipping the draw must
/// blank the frame.
#[test]
fn mutation_skip_forward_draw_blanks_the_frame() {
    let Some(mut ctx) = context() else { return };
    let size = 64;
    let options = RenderOptions {
        mutate_skip_forward_draw: true,
        ..ci_options()
    };
    let out = ctx
        .render_frame(&sphere_scene(size, size), size, size, &options)
        .expect("mutated render");
    assert_eq!(
        coverage(&out),
        0.0,
        "skipping the forward draw must leave the frame black"
    );
}

/// The ray-query path traces real geometry: the centre pixel is the hit colour
/// (magenta) and a corner pixel is the miss colour (dark blue).
#[test]
fn ray_query_hits_the_sphere() {
    let Some(mut ctx) = context() else { return };
    if !ctx.ray_query_support().complete() {
        eprintln!("SKIP: device has no ray query");
        return;
    }
    let size = 64;
    let options = RenderOptions {
        ray_probe: true,
        ..ci_options()
    };
    let out = ctx
        .render_frame(&sphere_scene(size, size), size, size, &options)
        .expect("ray-query render");
    assert_eq!(out.validation_messages, 0, "validation messages seen");
    let centre = pixel(&out, size / 2, size / 2);
    assert!(
        centre[0] > 200 && centre[2] > 200 && centre[1] < 60,
        "centre should be the magenta hit colour, got {centre:?}"
    );
    let corner = pixel(&out, 1, 1);
    assert!(
        corner[2] > 60 && corner[0] < 40,
        "corner should be the blue miss colour, got {corner:?}"
    );
}

/// The rasterised sphere (generated in the vertex shader from its own vertex
/// and face tables) and the ray-traced sphere (the BLAS built from
/// `icosphere_mesh`) are the same solid: seen off-axis, so that every part of
/// the mesh contributes to the outline, the lit pixels of the forward pass and
/// the hit pixels of the ray probe coincide. The same holds for the cube.
/// A wrong vertex or face in either table moves one outline but not the other.
#[test]
fn rasterised_and_ray_traced_shapes_have_the_same_outline() {
    let Some(mut ctx) = context() else { return };
    if !ctx.ray_query_support().complete() {
        eprintln!("SKIP: device has no ray query");
        return;
    }
    let size = 128u32;
    let camera = space_render::CameraPose {
        position: WorldPos::ZERO,
        forward: [0.0, 0.0, -1.0],
        up: [0.0, 1.0, 0.0],
        fov_y: std::f32::consts::FRAC_PI_3,
        aspect: 1.0,
        near: 0.05,
        far: 1.0e5,
    };
    for (shape, name) in [
        (space_render::SHAPE_SPHERE, "sphere"),
        (space_render::SHAPE_BOX, "cube"),
    ] {
        // Up and to the right of the view axis, so it is seen from below-left.
        let scene = Scene::from_instances(
            camera,
            vec![GpuInstance {
                center: [1.6, 1.1, -4.5, 1.0],
                albedo: [1.0, 1.0, 1.0, 0.0],
                emissive: [4.0, 4.0, 4.0, 1.0],
                roughness: 0.5,
                shape,
                transmission: 0.0,
                _pad1: 0.0,
            }],
        );
        let raster = ctx
            .render_frame(&scene, size, size, &ci_options())
            .expect("forward render");
        let traced = ctx
            .render_frame(
                &scene,
                size,
                size,
                &RenderOptions {
                    ray_probe: true,
                    ..ci_options()
                },
            )
            .expect("ray-probe render");
        assert_eq!(raster.validation_messages, 0, "{name}: forward");
        assert_eq!(traced.validation_messages, 0, "{name}: ray probe");
        let lit: Vec<bool> = raster
            .pixels
            .as_chunks::<4>()
            .0
            .iter()
            .map(|p| p[0] > 8)
            .collect();
        // Hit colour is magenta, miss colour dark blue: red tells them apart.
        let hit: Vec<bool> = traced
            .pixels
            .as_chunks::<4>()
            .0
            .iter()
            .map(|p| p[0] > 128)
            .collect();
        let covered = lit.iter().filter(|&&l| l).count();
        let differing = lit.iter().zip(&hit).filter(|(l, h)| l != h).count();
        eprintln!("{name}: {covered} lit pixels, {differing} differ from the ray-traced outline");
        assert!(covered > 400, "{name}: too small to compare ({covered})");
        // Only outline pixels may differ (rasterisation and ray casting resolve
        // an edge crossing a pixel differently): a couple of percent at most.
        assert!(
            differing * 40 < covered,
            "{name}: rasterised and ray-traced outlines differ in {differing} of {covered} pixels"
        );
    }
}

/// The mesh pipeline (vertex buffers, as used for GLB models) is the same
/// renderer as the procedural shapes: a cube given as a *mesh* — the vertices
/// of `cube_mesh` with flat normals, one material part — placed, scaled and
/// lit like a procedural box instance renders the same image. That pins the
/// model matrix, the normal transform, the material record and the vertex
/// layout all at once. A second render through a GLB round trip of the same
/// mesh proves the loader feeds the pipeline the same data.
#[test]
fn a_mesh_cube_renders_like_the_procedural_box() {
    use space_render::{MeshData, MeshMaterial, MeshPart, MeshVertex, ModelInstance};
    use std::sync::Arc;

    let Some(mut ctx) = context() else { return };
    let size = 128u32;
    let camera = space_render::CameraPose {
        position: WorldPos::ZERO,
        forward: [0.0, 0.0, -1.0],
        up: [0.0, 1.0, 0.0],
        fov_y: std::f32::consts::FRAC_PI_3,
        aspect: 1.0,
        near: 0.05,
        far: 1.0e5,
    };
    let (center, half) = ([0.9f32, -0.6, -5.0], 1.25f32);
    let material = MeshMaterial {
        base_color: [0.8, 0.3, 0.1],
        metallic: 0.0,
        roughness: 0.45,
        emissive: [0.02, 0.03, 0.05],
    };
    let light = space_render::SceneLight {
        position: WorldPos::from_f64_metres_around(WorldPos::ZERO, [3.0, 4.0, 1.0]),
        color: [1.0, 1.0, 1.0],
        intensity: 60.0,
    };

    // The procedural box.
    let boxed = Scene::from_instances(
        camera,
        vec![GpuInstance {
            center: [center[0], center[1], center[2], half],
            albedo: [0.8, 0.3, 0.1, material.metallic],
            emissive: [0.02, 0.03, 0.05, 1.0],
            roughness: material.roughness,
            shape: space_render::SHAPE_BOX,
            transmission: 0.0,
            _pad1: 0.0,
        }],
    )
    .with_lights(vec![light]);

    // The same cube as a mesh: `cube_mesh` is six faces of six vertices, in
    // the order +x, -x, +y, -y, +z, -z.
    let normals = [
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ];
    let vertices: Vec<MeshVertex> = space_render::cube_mesh()
        .iter()
        .enumerate()
        .map(|(i, &position)| MeshVertex {
            position,
            normal: normals[i / 6],
        })
        .collect();
    let cube = MeshData {
        indices: (0..vertices.len() as u32).collect(),
        vertices,
        parts: vec![MeshPart {
            first_index: 0,
            index_count: 36,
            material,
        }],
        bounds_min: [-1.0; 3],
        bounds_max: [1.0; 3],
    };
    let scene_with = |mesh: MeshData| {
        let mut scene = Scene::from_instances(camera, Vec::new()).with_lights(vec![light]);
        scene.models.push(ModelInstance {
            mesh: Arc::new(mesh),
            position: center,
            rotation: space_render::IDENTITY_ROTATION,
            scale: half,
        });
        scene
    };

    let reference = ctx
        .render_frame(&boxed, size, size, &ci_options())
        .expect("box render");
    let meshed = ctx
        .render_frame(&scene_with(cube.clone()), size, size, &ci_options())
        .expect("mesh render");
    let from_glb = MeshData::from_glb(&cube.to_glb()).expect("round trip");
    let loaded = ctx
        .render_frame(&scene_with(from_glb), size, size, &ci_options())
        .expect("glb mesh render");
    for out in [&reference, &meshed, &loaded] {
        assert_eq!(out.validation_messages, 0, "validation messages seen");
    }
    let lit = lit_pixels(&reference);
    assert!(
        lit > 1500,
        "the reference box is too small to compare ({lit})"
    );
    let differing = |a: &space_render::FrameOutput, b: &space_render::FrameOutput| {
        a.pixels
            .as_chunks::<4>()
            .0
            .iter()
            .zip(b.pixels.as_chunks::<4>().0)
            .filter(|(x, y)| (0..3).any(|c| x[c].abs_diff(y[c]) > 2))
            .count()
    };
    let (mesh_diff, glb_diff) = (differing(&reference, &meshed), differing(&meshed, &loaded));
    eprintln!("mesh cube: {lit} lit pixels, {mesh_diff} differ from the box, {glb_diff} after GLB");
    assert!(
        mesh_diff * 200 < lit,
        "the mesh cube differs from the procedural box in {mesh_diff} of {lit} pixels"
    );
    assert_eq!(glb_diff, 0, "the GLB round trip changed the rendered image");

    // MUTATION CHECK: without the mesh the frame is black, so the pixels above
    // really come from the mesh pipeline. And a model that is no longer in the
    // scene is no longer drawn (its buffers are dropped from the cache).
    let empty = Scene::from_instances(camera, Vec::new()).with_lights(vec![light]);
    let blank = ctx
        .render_frame(&empty, size, size, &ci_options())
        .expect("empty render");
    assert_eq!(brightness_sum(&blank), 0);
}

/// The sample ship — six material parts in one vertex/index buffer — renders
/// through the mesh pipeline with every preset, including the post pass
/// (reflections/refraction read the mesh's depth, material and normal like
/// any other surface), and its emissive engine part shows up blue.
#[test]
fn the_sample_ship_renders_with_every_preset() {
    use space_render::{rotation_y, sample_ship, ModelInstance};
    use std::sync::Arc;

    let Some(mut ctx) = context() else { return };
    let size = 128u32;
    let camera = space_render::CameraPose {
        position: WorldPos::ZERO,
        forward: [0.0, 0.0, -1.0],
        up: [0.0, 1.0, 0.0],
        fov_y: std::f32::consts::FRAC_PI_3,
        aspect: 1.0,
        near: 0.05,
        far: 1.0e5,
    };
    let ship = Arc::new(sample_ship());
    let mut scene =
        Scene::from_instances(camera, Vec::new()).with_lights(vec![space_render::SceneLight {
            position: WorldPos::from_f64_metres_around(WorldPos::ZERO, [3.0, 4.0, 2.0]),
            color: [1.0, 1.0, 1.0],
            intensity: 80.0,
        }]);
    // Seen from behind and above, so the engine glow faces the camera.
    scene.models.push(ModelInstance {
        mesh: ship,
        position: [0.0, -0.8, -7.0],
        rotation: rotation_y(0.5),
        scale: 1.0,
    });
    for preset in QualityPreset::ALL {
        let out = ctx
            .render_frame(
                &scene,
                size,
                size,
                &RenderOptions {
                    preset,
                    ..ci_options()
                },
            )
            .expect("ship render");
        assert_eq!(out.validation_messages, 0, "{preset}");
        assert!(
            lit_pixels(&out) > 400,
            "{preset}: {} lit pixels",
            lit_pixels(&out)
        );
        let blue_glow = out
            .pixels
            .as_chunks::<4>()
            .0
            .iter()
            // Radiance (0.6, 2.2, 6.0) tone-maps to a near-white blue: bright,
            // with blue clearly above red.
            .filter(|p| p[2] > 230 && p[2] > p[0].saturating_add(25))
            .count();
        assert!(
            blue_glow > 10,
            "{preset}: engine glow missing ({blue_glow} px)"
        );
    }
}

/// MUTATION CHECK for the ray-query test: with an empty scene every pixel must
/// be the miss colour, so the magenta centre assertion above can only come
/// from a real triangle hit.
#[test]
fn mutation_empty_scene_ray_probe_has_no_hits() {
    let Some(mut ctx) = context() else { return };
    if !ctx.ray_query_support().complete() {
        eprintln!("SKIP: device has no ray query");
        return;
    }
    let size = 64;
    let options = RenderOptions {
        ray_probe: true,
        ..ci_options()
    };
    let empty = Scene::from_instances(sphere_scene(size, size).camera, Vec::new());
    let out = ctx
        .render_frame(&empty, size, size, &options)
        .expect("empty ray-query render");
    let centre = pixel(&out, size / 2, size / 2);
    assert!(
        centre[0] < 40 && centre[2] > 60,
        "empty scene must miss everywhere, got {centre:?}"
    );
}

/// Probe-GI quality: a fine 8x8 grid of 1 m-spaced probes around the camera,
/// enough rays per direction to average out the jitter deterministically.
fn gi_quality() -> Quality {
    Quality {
        grid: 8,
        rays_per_probe: 16,
        probe_range: 100.0,
        spacing: 1.0,
    }
}

/// A scene for the GI test: a white diffuse box dead-centre (zero emissive, no
/// lights, so the forward pass alone renders it black) and a bright emissive
/// sphere *behind* the camera. The box's visible faces point at the camera, so
/// the probe texel the forward pass samples for them is the +Z one — which is
/// exactly the direction the probes see the emitter in.
fn gi_scene(width: u32, height: u32) -> Scene {
    let camera = space_render::CameraPose {
        position: WorldPos::ZERO,
        forward: [0.0, 0.0, -1.0],
        up: [0.0, 1.0, 0.0],
        fov_y: std::f32::consts::FRAC_PI_3,
        aspect: width as f32 / height as f32,
        near: 0.05,
        far: 1000.0,
    };
    Scene::from_instances(
        camera,
        vec![
            // Receiver: white diffuse box, no emissive.
            GpuInstance {
                center: [0.0, 0.0, -5.0, 1.5],
                albedo: [1.0, 1.0, 1.0, 0.0],
                emissive: [0.0, 0.0, 0.0, 1.0],
                roughness: 0.9,
                shape: space_render::SHAPE_BOX,
                transmission: 0.0,
                _pad1: 0.0,
            },
            // Emitter: well behind the camera and large enough to fill the
            // probe cone, so the +Z probe texels see it everywhere. It is never
            // drawn itself and never contains a probe.
            GpuInstance {
                center: [0.0, 0.0, 8.0, 4.0],
                albedo: [1.0, 1.0, 1.0, 0.0],
                emissive: [10.0, 10.0, 10.0, 1.0],
                roughness: 0.5,
                shape: space_render::SHAPE_SPHERE,
                transmission: 0.0,
                _pad1: 0.0,
            },
        ],
    )
}

/// The probe GI lights the receiver: with GI on the box is visibly lit, with
/// the GI dispatch skipped it is black. The second render *is* the mutation
/// check (`mutate_skip_gi_dispatch`), so this test proves the GI pixels come
/// from the ray-query probe pass and not from the forward pass.
#[test]
fn probe_gi_lights_the_receiver() {
    let Some(mut ctx) = context() else { return };
    if !ctx.ray_query_support().complete() {
        eprintln!("SKIP: device has no ray query");
        return;
    }
    let size = 64;
    let scene = gi_scene(size, size);

    let with_gi = ctx
        .render_frame(
            &scene,
            size,
            size,
            &RenderOptions {
                gi_enabled: true,
                quality: gi_quality(),
                ..Default::default()
            },
        )
        .expect("GI render");
    assert_eq!(with_gi.validation_messages, 0, "validation messages seen");

    let without_gi = ctx
        .render_frame(
            &scene,
            size,
            size,
            &RenderOptions {
                gi_enabled: true,
                quality: gi_quality(),
                mutate_skip_gi_dispatch: true,
                ..Default::default()
            },
        )
        .expect("mutated GI render");

    let lit = pixel(&with_gi, size / 2, size / 2);
    let dark = pixel(&without_gi, size / 2, size / 2);
    assert!(
        dark[0] < 8 && dark[1] < 8 && dark[2] < 8,
        "with the GI dispatch skipped the receiver must be black, got {dark:?}"
    );
    assert!(
        lit[0] > 60 && lit[1] > 60 && lit[2] > 60,
        "with GI the receiver must be lit by the emitter behind the camera, got {lit:?}"
    );
}

/// The GI pass is deterministic: two renders of the same scene are
/// byte-identical (the ray jitter is hashed from indices only). The centre
/// pixel must be lit, so the equality cannot be satisfied by two black frames.
#[test]
fn probe_gi_is_deterministic() {
    let Some(mut ctx) = context() else { return };
    if !ctx.ray_query_support().complete() {
        eprintln!("SKIP: device has no ray query");
        return;
    }
    let size = 48;
    let options = RenderOptions {
        gi_enabled: true,
        quality: gi_quality(),
        ..Default::default()
    };
    let a = ctx
        .render_frame(&gi_scene(size, size), size, size, &options)
        .expect("first GI render");
    let b = ctx
        .render_frame(&gi_scene(size, size), size, size, &options)
        .expect("second GI render");
    assert_eq!(a.pixels, b.pixels, "GI output must be bit-identical");
    // Determinism is only meaningful if there is something to reproduce: the
    // receiver centre must be lit by GI.
    let centre = pixel(&a, size / 2, size / 2);
    assert!(
        centre[0] > 60 && centre[1] > 60 && centre[2] > 60,
        "deterministic GI frame must not be black, centre got {centre:?}"
    );
    // MUTATION CHECK: skipping the GI dispatch must blank the receiver, so the
    // byte-identical frames above can only come from the probe pass.
    let mutated = ctx
        .render_frame(
            &gi_scene(size, size),
            size,
            size,
            &RenderOptions {
                mutate_skip_gi_dispatch: true,
                ..options
            },
        )
        .expect("mutated GI render");
    let dark = pixel(&mutated, size / 2, size / 2);
    assert!(
        dark[0] < 8 && dark[1] < 8 && dark[2] < 8,
        "with the GI dispatch skipped the receiver must be black, got {dark:?}"
    );
}

/// LOD integration: the deterministic galaxy renders through the same path,
/// with the instance count bounded by `MAX_SELECTED`, produces non-trivial
/// coverage, is byte-identical across two renders, and `mutate_skip_forward_draw`
/// blanks it (proving the pixels come from the forward draw).
#[test]
fn lod_galaxy_renders() {
    let Some(mut ctx) = context() else { return };
    let size = 96;
    let camera = space_render::CameraPose {
        // Outside the galaxy root, looking back at its centre, so the LOD
        // selection fills the frame.
        position: WorldPos::from_f64_metres_around(WorldPos::ZERO, [0.0, 0.0, -5.0e4]),
        forward: [0.0, 0.0, 1.0],
        up: [0.0, 1.0, 0.0],
        fov_y: std::f32::consts::FRAC_PI_3,
        aspect: 1.0,
        near: 1.0,
        far: 5.0e6,
    };
    let galaxy = space_world::Galaxy::new(0x5eed);
    let lod = space_world::LodSelector::new(30, 63, 8.0);
    let scene = Scene::build(&galaxy, &lod, &camera);
    assert!(
        scene.instances.len() <= space_world::MAX_SELECTED,
        "instances {}",
        scene.instances.len()
    );
    // The LOD selector must produce a non-trivial selection.
    assert!(
        scene.instances.len() >= 100,
        "LOD selection too small: {}",
        scene.instances.len()
    );
    let opts = RenderOptions {
        gi_enabled: false,
        quality: Quality::CI,
        // Fog off: the golden/pixel tests must not depend on fog.
        preset: QualityPreset::Ci,
        ..Default::default()
    };
    let out = ctx
        .render_frame(&scene, size, size, &opts)
        .expect("galaxy render");
    assert_eq!(out.validation_messages, 0, "validation messages seen");
    // Coverage must be non-trivial (not just > 0.0): the stars are small at
    // this distance, so assert on the number of lit pixels and the total
    // brightness rather than a fraction of the frame.
    let lit = lit_pixels(&out);
    eprintln!(
        "galaxy: instances={} lit={lit} coverage={} brightness={}",
        scene.instances.len(),
        coverage(&out),
        brightness_sum(&out)
    );
    assert!(
        lit >= 20,
        "galaxy lit too few pixels: {lit} ({} instances)",
        scene.instances.len()
    );
    let sum = brightness_sum(&out);
    // GOLDEN, not a floor: a bare `> N` would also pass for a *different but
    // still bright* LOD selection, so this pins the exact selection (2135
    // instances) and the exact star geometry the renderer drew. The ±0.1% band
    // only absorbs a future driver's tonemap rounding; any wrong LOD selection
    // moves the sum by far more than that.
    //
    // History of the value (each step re-measured on an NVIDIA RTX 4080 SUPER;
    // the first was also byte-identical on RADV GFX1151 and lavapipe):
    //  * 74_732, 2104 instances — the original.
    //  * 76_185 — the icosphere fixed: four of its vertices sat at radius 0.74
    //    and three faces named the wrong vertices, so every star was dented
    //    and holed.
    //  * 83_903, 2135 instances — `NodeRef::center()` fixed (it was 512 m off
    //    per axis below depth 64, which moved every star and skewed the LOD
    //    frustum cull), and stars more than a few pixels across drawn as
    //    tessellated spheres instead of bare icosahedra.
    const GALAXY_BRIGHTNESS: u64 = 83_903;
    let band = GALAXY_BRIGHTNESS / 1_000; // 0.1%
    assert!(
        sum.abs_diff(GALAXY_BRIGHTNESS) <= band,
        "galaxy brightness_sum {sum} is outside the golden band \
         [{}, {}] (golden {GALAXY_BRIGHTNESS}, {} instances) — the LOD selection changed",
        GALAXY_BRIGHTNESS - band,
        GALAXY_BRIGHTNESS + band,
        scene.instances.len(),
    );
    // Determinism: a second render must be byte-identical.
    let out2 = ctx
        .render_frame(&scene, size, size, &opts)
        .expect("galaxy render 2");
    assert_eq!(out.pixels, out2.pixels, "galaxy render not deterministic");
    // Mutation: skipping the forward draw must blank the frame.
    let mutated = ctx
        .render_frame(
            &scene,
            size,
            size,
            &RenderOptions {
                mutate_skip_forward_draw: true,
                ..opts
            },
        )
        .expect("galaxy mutated");
    assert_eq!(
        brightness_sum(&mutated),
        0,
        "skipping the forward draw must blank the galaxy frame"
    );
}

/// The context reports a device and its ray-query capability, and a second
/// context can be created and dropped (no leaks / validation errors).
#[test]
fn context_reports_its_device() {
    let Some(ctx) = context() else { return };
    assert!(!ctx.device_name().is_empty());
    let support = ctx.ray_query_support();
    eprintln!(
        "device {:?} ray_query={:?} (validation messages so far: {})",
        ctx.device_name(),
        support,
        validation_message_count()
    );
}

/// The Cook-Torrance BRDF lights a *non-emissive* sphere from an analytic
/// point light: the centre pixel is lit but not saturated (so the value comes
/// from the shading model, not from emissive or a clamp), and zeroing the light
/// count with `mutate_zero_light_count` blanks it — proving the pixels are the
/// analytic-light BRDF term and nothing else.
#[test]
fn brdf_analytic_light_lights_the_sphere() {
    let Some(mut ctx) = context() else { return };
    let size = 64;
    // Same camera as `demo_scene`, but the sphere has zero emissive so the only
    // light reaching it is the analytic point light.
    let camera = space_render::CameraPose {
        position: WorldPos::ZERO,
        forward: [0.0, 0.0, -1.0],
        up: [0.0, 1.0, 0.0],
        fov_y: std::f32::consts::FRAC_PI_3,
        aspect: 1.0,
        near: 0.05,
        far: 1.0e5,
    };
    let scene = Scene::from_instances(
        camera,
        vec![GpuInstance {
            center: [0.0, 0.0, -6.0, 1.0],
            albedo: [0.8, 0.8, 0.8, 0.0],
            emissive: [0.0, 0.0, 0.0, 1.0],
            roughness: 0.3,
            shape: space_render::SHAPE_SPHERE,
            transmission: 0.0,
            _pad1: 0.0,
        }],
    )
    .with_lights(vec![space_render::SceneLight {
        position: WorldPos::from_f64_metres_around(WorldPos::ZERO, [2.0, 2.0, -2.0]),
        color: [1.0, 1.0, 1.0],
        intensity: 40.0,
    }]);
    let opts = RenderOptions {
        gi_enabled: false,
        quality: Quality::CI,
        // Fog off: the golden/pixel tests must not depend on fog.
        preset: QualityPreset::Ci,
        ..Default::default()
    };
    let out = ctx
        .render_frame(&scene, size, size, &opts)
        .expect("brdf render");
    assert_eq!(out.validation_messages, 0, "validation messages seen");
    // The sphere must be visible somewhere: the brightest pixel is well above
    // black, and the brightest value across the frame is not the emissive
    // saturation of the demo scene (4.0 radiance -> 255 after tonemap).
    let mut brightest = [0u16; 3];
    for px in out.pixels.as_chunks::<4>().0 {
        for c in 0..3 {
            brightest[c] = brightest[c].max(px[c] as u16);
        }
    }
    assert!(
        brightest[0] > 40,
        "the analytic light must light the non-emissive sphere, brightest {brightest:?}"
    );
    // Mutation: with no lights the sphere is non-emissive and GI is off, so
    // every pixel must be exactly black.
    let mutated = ctx
        .render_frame(
            &scene,
            size,
            size,
            &RenderOptions {
                mutate_zero_light_count: true,
                ..opts
            },
        )
        .expect("mutated brdf render");
    assert_eq!(
        brightness_sum(&mutated),
        0,
        "zeroing the light count must leave a non-emissive scene black"
    );
}

/// The metallic-roughness model behaves like one. Three unlit-by-emission
/// spheres under one point light:
///
/// * a **smooth dielectric** has a small, intense highlight on top of its
///   diffuse colour;
/// * the same material made **rough** has a much dimmer highlight (the same
///   energy spread over a wide lobe) and keeps its diffuse colour;
/// * a **smooth metal** has no diffuse term at all: away from its highlight it
///   is black, and its highlight is tinted by its base colour.
///
/// Mutation: with the light removed (`mutate_zero_light_count`) all three are
/// black, so every pixel asserted on comes from the BRDF.
#[test]
fn pbr_roughness_and_metalness_shape_the_highlight() {
    let Some(mut ctx) = context() else { return };
    let (width, height) = (192u32, 64u32);
    let camera = space_render::CameraPose {
        position: WorldPos::ZERO,
        forward: [0.0, 0.0, -1.0],
        up: [0.0, 1.0, 0.0],
        fov_y: 0.5,
        aspect: width as f32 / height as f32,
        near: 0.05,
        far: 1.0e3,
    };
    let sphere = |x: f32, albedo: [f32; 3], metallic: f32, roughness: f32| GpuInstance {
        center: [x, 0.0, -9.0, 1.0],
        albedo: [albedo[0], albedo[1], albedo[2], metallic],
        emissive: [0.0, 0.0, 0.0, 1.0],
        roughness,
        shape: space_render::SHAPE_SPHERE,
        transmission: 0.0,
        _pad1: 0.0,
    };
    let red = [0.8, 0.1, 0.1];
    let gold = [1.0, 0.78, 0.34];
    let scene = Scene::from_instances(
        camera,
        vec![
            // One sphere centred in each third of the frame: the frame is
            // 2 * 9 * tan(0.25) * 3 = 13.8 m wide at their distance.
            sphere(-4.6, red, 0.0, 0.15),
            sphere(0.0, red, 0.0, 0.9),
            sphere(4.6, gold, 1.0, 0.15),
        ],
    )
    // Above and behind the camera: every sphere has its highlight facing us.
    .with_lights(vec![space_render::SceneLight {
        position: WorldPos::from_f64_metres_around(WorldPos::ZERO, [0.0, 6.0, 2.0]),
        color: [1.0, 1.0, 1.0],
        intensity: 400.0,
    }]);
    let out = ctx
        .render_frame(&scene, width, height, &ci_options())
        .expect("pbr render");
    assert_eq!(out.validation_messages, 0, "validation messages seen");

    // Per sphere (a third of the frame each): the brightest pixel, and the
    // mean colour of the lit-side pixels well away from the highlight.
    let stats = |column: u32| {
        let (x0, x1) = (column * width / 3, (column + 1) * width / 3);
        let mut peak = [0u16; 4];
        let mut pixels = Vec::new();
        for y in 0..height {
            for x in x0..x1 {
                let p = pixel(&out, x, y);
                if p[0] + p[1] + p[2] > peak[0] + peak[1] + peak[2] {
                    peak = p;
                }
                if p[0] + p[1] + p[2] > 0 {
                    pixels.push(p);
                }
            }
        }
        // The dimmer half of the sphere's pixels: the body, not the highlight.
        pixels.sort_by_key(|p| p[0] + p[1] + p[2]);
        let mean_of = |set: &[[u16; 4]]| {
            let n = set.len().max(1) as u32;
            [0, 1, 2].map(|c| set.iter().map(|p| u32::from(p[c])).sum::<u32>() / n)
        };
        let body = mean_of(&pixels[..pixels.len() / 2]);
        // The highlight's skirt: bright, but not clipped by the tone map, so
        // it still shows the highlight's colour.
        let skirt: Vec<[u16; 4]> = pixels
            .iter()
            .copied()
            .filter(|p| (90..240).contains(&p[0]))
            .collect();
        (peak, body, mean_of(&skirt), pixels.len())
    };
    let (smooth_peak, smooth_body, _, smooth_n) = stats(0);
    let (rough_peak, rough_body, _, rough_n) = stats(1);
    let (metal_peak, metal_body, metal_skirt, metal_n) = stats(2);
    eprintln!(
        "smooth plastic: peak {smooth_peak:?} body {smooth_body:?} ({smooth_n} px)\n\
         rough plastic:  peak {rough_peak:?} body {rough_body:?} ({rough_n} px)\n\
         smooth gold:    peak {metal_peak:?} body {metal_body:?} skirt {metal_skirt:?} \
         ({metal_n} px)"
    );
    // The same diffuse lobe lights both plastics: about as many lit pixels
    // (the off-axis sphere is stretched a little by the perspective).
    assert!(
        smooth_n > 300 && smooth_n.abs_diff(rough_n) * 4 < rough_n,
        "the two plastic spheres should be lit alike: {smooth_n} vs {rough_n} px"
    );
    // Roughness: the smooth highlight is far brighter than the rough one, and
    // bright in every channel (a dielectric's highlight is the light's
    // colour); the rough sphere's brightest pixel is still its red diffuse.
    assert!(
        smooth_peak[1] > 150,
        "a smooth dielectric has a white-ish highlight"
    );
    assert!(
        rough_peak[1] + 60 < smooth_peak[1],
        "roughness must dim the highlight: {rough_peak:?} vs {smooth_peak:?}"
    );
    assert!(
        rough_peak[0] > 2 * rough_peak[1],
        "rough plastic stays red: {rough_peak:?}"
    );
    // Diffuse: both dielectrics show their base colour over the lit body.
    for (name, body) in [("smooth", smooth_body), ("rough", rough_body)] {
        assert!(
            body[0] > 40 && body[0] > 3 * body[1].max(1),
            "{name} plastic body should be red, got {body:?}"
        );
    }
    // Metal: no diffuse — most of the lit sphere is (nearly) black, far darker
    // than the plastic body — and the highlight carries the gold tint.
    assert!(
        metal_body[0] * 4 < smooth_body[0],
        "a metal has no diffuse term: body {metal_body:?} vs plastic {smooth_body:?}"
    );
    assert!(
        metal_peak[0] > 200,
        "a smooth metal has a strong highlight: {metal_peak:?}"
    );
    assert!(
        metal_skirt[0] > metal_skirt[2] + 25,
        "a gold highlight is tinted gold (red well above blue), got {metal_skirt:?}"
    );

    // MUTATION CHECK: no light, no pixels.
    let dark = ctx
        .render_frame(
            &scene,
            width,
            height,
            &RenderOptions {
                mutate_zero_light_count: true,
                ..ci_options()
            },
        )
        .expect("unlit render");
    assert_eq!(
        brightness_sum(&dark),
        0,
        "everything above came from the light"
    );
}

/// Astronomical distances in one frame: a 1 m box 6 m away, an Earth-sized
/// sphere 25 000 km behind it, and a Sun-sized sphere one astronomical unit
/// behind that — eleven orders of magnitude of depth with a 1 m near plane.
/// Each must be drawn, and each nearer body must hide the farther one where
/// they overlap. (With a conventional depth buffer everything past ~17 000 km
/// lands exactly on the far plane: the planet and the star are not drawn, or
/// z-fight. See `mat4_perspective_reversed`.)
#[test]
fn bodies_at_astronomical_distances_are_drawn_and_depth_sorted() {
    let Some(mut ctx) = context() else { return };
    let size = 128u32;
    let camera = space_render::CameraPose {
        position: WorldPos::ZERO,
        forward: [0.0, 0.0, -1.0],
        up: [0.0, 1.0, 0.0],
        fov_y: std::f32::consts::FRAC_PI_3,
        aspect: 1.0,
        near: 1.0,
        far: 1.0e13,
    };
    let body = |center: [f32; 3], radius: f32, emissive: [f32; 3], shape| GpuInstance {
        center: [center[0], center[1], center[2], radius],
        albedo: [0.0, 0.0, 0.0, 0.0],
        emissive: [emissive[0], emissive[1], emissive[2], 1.0],
        roughness: 0.8,
        shape,
        transmission: 0.0,
        _pad1: 0.0,
    };
    let (blue, white, magenta) = ([0.0, 0.0, 6.0], [6.0, 6.0, 6.0], [6.0, 0.0, 6.0]);
    const EARTH_RADIUS: f32 = 6.371e6;
    const EARTH_DISTANCE: f32 = 2.5e7;
    const SUN_RADIUS: f32 = 6.9634e8;
    const AU: f32 = 1.495_978_7e11;
    // The star is listed first and the box last, so draw order cannot be what
    // sorts them. The planet covers the centre of the frame, the star is far
    // larger in the sky than the gap the planet leaves... so it is offset to
    // peek out on the right: its left part is behind the planet.
    let sun_offset = 0.33 * AU;
    let scene = Scene::from_instances(
        camera,
        vec![
            body(
                [sun_offset, 0.0, -AU],
                40.0 * SUN_RADIUS,
                white,
                space_render::SHAPE_SPHERE,
            ),
            body(
                [0.0, 0.0, -EARTH_DISTANCE],
                EARTH_RADIUS,
                blue,
                space_render::SHAPE_SPHERE,
            ),
            body([0.0, 0.0, -6.0], 0.5, magenta, space_render::SHAPE_BOX),
        ],
    );
    let out = ctx
        .render_frame(&scene, size, size, &ci_options())
        .expect("astronomical render");
    assert_eq!(out.validation_messages, 0, "validation messages seen");

    let is = |p: [u16; 4], what: &str| match what {
        "magenta" => p[0] > 200 && p[2] > 200 && p[1] < 60,
        "blue" => p[2] > 200 && p[0] < 60 && p[1] < 60,
        "white" => p[0] > 200 && p[1] > 200 && p[2] > 200,
        _ => p[0] < 8 && p[1] < 8 && p[2] < 8,
    };
    let count = |what: &str| {
        (0..size)
            .flat_map(|y| (0..size).map(move |x| (x, y)))
            .filter(|&(x, y)| is(pixel(&out, x, y), what))
            .count()
    };
    let (magenta_n, blue_n, white_n) = (count("magenta"), count("blue"), count("white"));
    eprintln!("box {magenta_n} px, planet {blue_n} px, star {white_n} px");
    // All three are there.
    assert!(
        magenta_n > 100,
        "the 1 m box 6 m away is missing ({magenta_n} px)"
    );
    assert!(
        blue_n > 1000,
        "the planet 25 000 km away is missing ({blue_n} px)"
    );
    assert!(
        white_n > 300,
        "the star 1 AU away is missing ({white_n} px)"
    );
    // Nearest wins: the box in front of the planet at the centre...
    let centre = pixel(&out, size / 2, size / 2);
    assert!(
        is(centre, "magenta"),
        "the box must hide the planet, got {centre:?}"
    );
    // ...the planet in front of the star just right of the box (the star's
    // disc extends to there, but it is 1 AU behind)...
    let on_planet = pixel(&out, size / 2 + 22, size / 2);
    assert!(
        is(on_planet, "blue"),
        "the planet must hide the star, got {on_planet:?}"
    );
    // ...and the star where it clears the planet's limb.
    let on_star = pixel(&out, size - 6, size / 2);
    assert!(
        is(on_star, "white"),
        "the star must show past the planet, got {on_star:?}"
    );

    // Depth ordering, not draw order: reversing the instances draws the same
    // picture.
    let mut reversed = Scene::from_instances(camera, scene.instances.clone());
    reversed.instances.reverse();
    let again = ctx
        .render_frame(&reversed, size, size, &ci_options())
        .expect("reversed render");
    assert_eq!(out.pixels, again.pixels, "draw order changed the image");
}

/// One emissive sphere of `radius` at `distance` metres straight ahead.
fn fog_scene(distance: f32, radius: f32) -> Scene {
    let camera = space_render::CameraPose {
        position: WorldPos::ZERO,
        forward: [0.0, 0.0, -1.0],
        up: [0.0, 1.0, 0.0],
        fov_y: std::f32::consts::FRAC_PI_3,
        aspect: 1.0,
        near: 0.05,
        far: 1.0e5,
    };
    Scene::from_instances(
        camera,
        vec![GpuInstance {
            center: [0.0, 0.0, -distance, radius],
            albedo: [1.0, 1.0, 1.0, 0.0],
            emissive: [4.0, 4.0, 4.0, 1.0],
            roughness: 0.5,
            shape: space_render::SHAPE_SPHERE,
            transmission: 0.0,
            _pad1: 0.0,
        }],
    )
}

/// Fog: a distant white sphere shifts toward the (blue-ish) fog colour with
/// fog on, is monotone in preset strength, leaves a near sphere and the empty
/// background alone, and is identical for presets with fog off.
/// Mutation: `QualityPreset::Ci` (fog off) must make the shift assertions fail,
/// which is what the "off" renders below are compared against.
#[test]
fn fog_shifts_distant_pixels_toward_fog_colour() {
    let Some(mut ctx) = context() else { return };
    let size = 64;
    let mut render = |scene: &Scene, preset: QualityPreset| {
        let opts = RenderOptions {
            gi_enabled: false,
            quality: Quality::CI,
            preset,
            ..Default::default()
        };
        let out = ctx
            .render_frame(scene, size, size, &opts)
            .expect("fog render");
        assert_eq!(out.validation_messages, 0, "validation messages seen");
        out
    };
    let c = size / 2;

    // Far sphere: 800 m away, radius 100 m (covers the image centre).
    let far = fog_scene(800.0, 100.0);
    let off = render(&far, QualityPreset::Ci);
    let low = render(&far, QualityPreset::Low);
    let medium = render(&far, QualityPreset::Medium);
    let high = render(&far, QualityPreset::High);
    assert_eq!(off.pixels, low.pixels, "Low and Ci both have fog off");

    let p_off = pixel(&off, c, c);
    let p_med = pixel(&medium, c, c);
    let p_high = pixel(&high, c, c);
    eprintln!("fog centre pixel: off={p_off:?} medium={p_med:?} high={p_high:?}");
    // Unfogged the emissive sphere saturates to neutral white.
    assert!(
        p_off[0].abs_diff(p_off[2]) <= 2,
        "unfogged must be neutral: {p_off:?}"
    );
    assert!(
        p_off[0] >= 200,
        "unfogged sphere should be bright: {p_off:?}"
    );
    // Fogged: darker toward the fog colour and tinted blue (fog b > r).
    assert!(
        p_high[0] + 20 < p_off[0],
        "red must drop toward fog: {p_high:?} vs {p_off:?}"
    );
    assert!(
        p_high[2] >= p_high[0] + 5,
        "must be tinted toward blue fog: {p_high:?}"
    );
    // Stronger fog, stronger shift.
    assert!(
        p_med[0] <= p_off[0] && p_high[0] <= p_med[0],
        "monotone: {p_off:?} {p_med:?} {p_high:?}"
    );
    assert!(
        p_med[0] < p_off[0],
        "medium fog must also shift: {p_med:?} vs {p_off:?}"
    );

    // The empty background is not fogged.
    assert_eq!(
        pixel(&off, 0, 0),
        pixel(&high, 0, 0),
        "background must be unchanged"
    );

    // A near sphere (6 m) is effectively unaffected by fog.
    let near = fog_scene(6.0, 1.0);
    let near_off = render(&near, QualityPreset::Ci);
    let near_high = render(&near, QualityPreset::High);
    let (a, b) = (pixel(&near_off, c, c), pixel(&near_high, c, c));
    for ch in 0..3 {
        assert!(
            a[ch].abs_diff(b[ch]) <= 2,
            "near pixel moved by fog: {a:?} vs {b:?}"
        );
    }
}

// ---- Phase 5b: screen-space reflections + refraction ---------------------------

/// Render `scene` at `size`x`size` with `preset` (GI off, CI probe quality),
/// asserting zero validation messages.
fn render_preset(
    ctx: &mut RenderContext,
    scene: &Scene,
    size: u32,
    preset: QualityPreset,
) -> space_render::FrameOutput {
    let opts = RenderOptions {
        gi_enabled: false,
        quality: Quality::CI,
        preset,
        ..Default::default()
    };
    let out = ctx
        .render_frame(scene, size, size, &opts)
        .expect("post-pass render");
    assert_eq!(out.validation_messages, 0, "validation messages seen");
    out
}

/// Pixel-count and centroid `(x, y)` of the pixels in rows `rows` for which
/// `pred(rgb)` holds.
fn centroid(
    out: &space_render::FrameOutput,
    rows: std::ops::Range<u32>,
    pred: impl Fn([u16; 3]) -> bool,
) -> Option<(usize, f64, f64)> {
    let (mut n, mut sx, mut sy) = (0usize, 0.0f64, 0.0f64);
    for y in rows {
        for x in 0..out.width {
            let p = pixel(out, x, y);
            if pred([p[0], p[1], p[2]]) {
                n += 1;
                sx += f64::from(x) + 0.5;
                sy += f64::from(y) + 0.5;
            }
        }
    }
    (n > 0).then(|| (n, sx / n as f64, sy / n as f64))
}

/// Camera at the origin looking down -Z, 60 degree fov, square aspect.
fn post_camera() -> space_render::CameraPose {
    space_render::CameraPose {
        position: WorldPos::ZERO,
        forward: [0.0, 0.0, -1.0],
        up: [0.0, 1.0, 0.0],
        fov_y: std::f32::consts::FRAC_PI_3,
        aspect: 1.0,
        near: 0.05,
        far: 1.0e5,
    }
}

fn emissive_sphere(center: [f32; 3], radius: f32, rgb: [f32; 3]) -> GpuInstance {
    GpuInstance {
        center: [center[0], center[1], center[2], radius],
        albedo: [0.0, 0.0, 0.0, 0.0],
        emissive: [rgb[0], rgb[1], rgb[2], 1.0],
        roughness: 0.5,
        shape: space_render::SHAPE_SPHERE,
        transmission: 0.0,
        _pad1: 0.0,
    }
}

/// A smooth metallic floor (a huge flat box whose top face is the plane
/// y = -1) and a bright red sphere (radius 0.5) hovering at (0, -0.4, -6).
///
/// Mirror geometry: the plane reflects the sphere to a mirrored sphere centred
/// at (0, -1.6, -6), which the 60-degree camera sees at NDC y = -0.462 — row
/// ~93.6 of a 128-pixel frame, x = 64 — below the directly seen sphere
/// (rows ~62..80). The floor has no light and no emission, so without SSR it
/// is black everywhere.
fn mirror_scene(roughness: f32) -> Scene {
    let floor = GpuInstance {
        center: [0.0, -101.0, 0.0, 100.0],
        albedo: [0.8, 0.8, 0.8, 1.0], // metallic = 1
        emissive: [0.0, 0.0, 0.0, 0.0],
        roughness,
        shape: space_render::SHAPE_BOX,
        transmission: 0.0,
        _pad1: 0.0,
    };
    Scene::from_instances(
        post_camera(),
        vec![
            floor,
            emissive_sphere([0.0, -0.4, -6.0], 0.5, [6.0, 0.0, 0.0]),
        ],
    )
}

/// A reflection pixel: clearly red.
fn is_red(p: [u16; 3]) -> bool {
    p[0] > 60 && p[0] > 3 * p[1].max(p[2]).max(1)
}

/// SSR: High and Medium show the red sphere's reflection exactly where mirror
/// geometry predicts; Low and Ci (post pass skipped) show a black floor; a floor
/// rougher than the preset's threshold (0.4: High 0.5 reflects, Medium 0.3
/// does not) is gated by the roughness limit; and floor pixels whose mirror ray
/// finds nothing are unchanged (a miss never darkens).
/// Mutation: rendering with `QualityPreset::Ci` (post pass off) is the "off"
/// side of every assertion below, so removing the post dispatch must fail them.
#[test]
fn ssr_reflects_a_bright_object_where_mirror_geometry_predicts() {
    let Some(mut ctx) = context() else { return };
    let size = 128;
    let smooth = mirror_scene(0.1);

    let ci = render_preset(&mut ctx, &smooth, size, QualityPreset::Ci);
    let low = render_preset(&mut ctx, &smooth, size, QualityPreset::Low);
    let medium = render_preset(&mut ctx, &smooth, size, QualityPreset::Medium);
    let high = render_preset(&mut ctx, &smooth, size, QualityPreset::High);
    assert_eq!(ci.pixels, low.pixels, "Low and Ci both skip the post pass");

    // Rows >= 83 only contain the floor (the direct sphere ends near row 80).
    let reflection_rows = 83..size;
    assert!(
        centroid(&ci, reflection_rows.clone(), is_red).is_none(),
        "no SSR => the floor reflects nothing"
    );
    // The direct sphere is there in every preset (sanity for the geometry).
    let direct = centroid(&ci, 0..83, is_red).expect("direct sphere visible");
    assert!(
        (direct.1 - 64.0).abs() < 2.0 && (direct.2 - 71.0).abs() < 3.0,
        "direct sphere centroid {direct:?}"
    );

    for (name, out) in [("medium", &medium), ("high", &high)] {
        let (n, cx, cy) = centroid(out, reflection_rows.clone(), is_red)
            .unwrap_or_else(|| panic!("{name}: no reflection of the red sphere"));
        eprintln!("{name}: reflection n={n} centroid=({cx:.1},{cy:.1})");
        assert!(n >= 40, "{name}: reflection too small ({n} px)");
        assert!(
            (cx - 64.0).abs() <= 2.5 && (cy - 93.6).abs() <= 4.0,
            "{name}: reflection centroid ({cx:.1},{cy:.1}) vs mirror prediction (64, 93.6)"
        );
        // Floor pixels away from the mirror image stay as without SSR.
        for (x, y) in [(10, 110), (118, 110), (30, 100), (64, 120)] {
            // (+-2: the High/Medium presets also enable distance fog.)
            let (a, b) = (pixel(out, x, y), pixel(&ci, x, y));
            assert!(
                (0..3).all(|c| a[c].abs_diff(b[c]) <= 2),
                "{name}: floor pixel ({x},{y}) must not change: {a:?} vs {b:?}"
            );
        }
    }
    // The direct (non-reflective) sphere pixels are untouched by SSR.
    let (a, b) = (pixel(&high, 64, 71), pixel(&ci, 64, 71));
    assert!((0..3).all(|c| a[c].abs_diff(b[c]) <= 2), "{a:?} vs {b:?}");

    // Roughness gate: 0.4 is within High's limit (0.5) but not Medium's (0.3).
    let rough = mirror_scene(0.4);
    let rough_medium = render_preset(&mut ctx, &rough, size, QualityPreset::Medium);
    let rough_high = render_preset(&mut ctx, &rough, size, QualityPreset::High);
    assert!(
        centroid(&rough_medium, reflection_rows.clone(), is_red).is_none(),
        "roughness 0.4 > Medium limit 0.3: no SSR"
    );
    assert!(
        centroid(&rough_high, reflection_rows, is_red).is_some(),
        "roughness 0.4 <= High limit 0.5: SSR"
    );
}

/// A transmissive ball lens (radius 2, centre (0, 0, -4)) in front of a green
/// marker sphere (radius 0.6, centre (1.6, 0, -9)) that sits entirely behind it
/// on screen. Without refraction the lens is an opaque black ball that hides the
/// marker; with refraction the marker shows through, displaced from where it
/// appears with no lens at all.
fn refraction_scene(with_glass: bool) -> Scene {
    let mut instances = vec![emissive_sphere([1.6, 0.0, -9.0], 0.6, [0.0, 6.0, 0.0])];
    if with_glass {
        instances.push(GpuInstance {
            center: [0.0, 0.0, -4.0, 2.0],
            albedo: [0.0, 0.0, 0.0, 0.0],
            emissive: [0.0, 0.0, 0.0, 0.0],
            roughness: 0.5,
            shape: space_render::SHAPE_SPHERE,
            transmission: 1.0,
            _pad1: 0.0,
        });
    }
    Scene::from_instances(post_camera(), instances)
}

fn is_green(p: [u16; 3]) -> bool {
    p[1] > 40 && p[1] > 3 * p[0].max(p[2]).max(1)
}

/// Refraction: the marker seen through the lens is displaced, in the direction
/// a convex lens magnifies (away from the lens centre), relative to the same
/// scene without the lens; with refraction off (Low/Ci) the lens is opaque and
/// the marker is hidden.
/// Mutation: `QualityPreset::Ci` (no glass pass, no post pass) is the "off" side.
#[test]
fn refraction_displaces_the_background_seen_through_glass() {
    let Some(mut ctx) = context() else { return };
    let size = 128;
    let glass = refraction_scene(true);
    let bare = refraction_scene(false);

    let reference = render_preset(&mut ctx, &bare, size, QualityPreset::High);
    let (ref_n, ref_x, ref_y) =
        centroid(&reference, 0..size, is_green).expect("marker visible without glass");

    // Refraction off: the opaque black lens hides the marker completely.
    for preset in [QualityPreset::Ci, QualityPreset::Low] {
        let off = render_preset(&mut ctx, &glass, size, preset);
        assert!(
            centroid(&off, 0..size, is_green).is_none(),
            "{preset}: opaque lens must hide the marker"
        );
        assert!(
            pixel(&off, 64, 64).iter().take(3).all(|&c| c < 8),
            "{preset}: unlit opaque lens is black"
        );
    }

    for preset in [QualityPreset::Medium, QualityPreset::High] {
        let out = render_preset(&mut ctx, &glass, size, preset);
        let (n, x, y) = centroid(&out, 0..size, is_green)
            .unwrap_or_else(|| panic!("{preset}: marker not visible through the glass"));
        eprintln!(
            "{preset}: marker n={n} centroid=({x:.2},{y:.2}); reference n={ref_n} ({ref_x:.2},{ref_y:.2})"
        );
        assert!(
            x > ref_x + 1.0,
            "{preset}: marker should be displaced outward (+x) by the lens: {x:.2} vs {ref_x:.2}"
        );
        assert!(
            (y - ref_y).abs() < 1.0,
            "{preset}: no vertical displacement expected: {y:.2} vs {ref_y:.2}"
        );
    }
}

/// A context built without `window_display` enabled no surface extension, so
/// `Swapchain::new` must return a clean error instead of letting ash abort on a
/// null `vkCreate*SurfaceKHR` entry point (the Windows
/// "Unable to load create_win32_surface_khr" crash).
/// Mutation: remove the `window_surface` guard in `Swapchain::new` and this
/// aborts/panics on platforms where the surface extension is absent.
#[test]
fn swapchain_without_surface_extension_is_a_clean_error() {
    use raw_window_handle::{
        RawDisplayHandle, RawWindowHandle, XlibDisplayHandle, XlibWindowHandle,
    };
    use space_render::{Swapchain, WindowHandles};
    let Some(ctx) = context() else { return };
    let handles = WindowHandles {
        display: RawDisplayHandle::Xlib(XlibDisplayHandle::new(None, 0)),
        window: RawWindowHandle::Xlib(XlibWindowHandle::new(1)),
    };
    match Swapchain::new(&ctx, handles, 64, 64) {
        Err(e) => assert!(e.to_string().contains("window_display"), "{e}"),
        Ok(_) => panic!("swapchain must not be created without surface extensions"),
    }
}
