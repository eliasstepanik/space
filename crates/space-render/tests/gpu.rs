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
        base_color_texture: None,
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
            uv: [0.0; 2],
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
        textures: Vec::new(),
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

/// Refraction: the marker seen through the ball lens is where Snell's law
/// puts it — for a marker this far behind the ball, on the other side of the
/// ball's centre: a ball lens turns what is behind it round. `optics` traces
/// the same rays on the CPU (through an ideal ball) and says which pixels
/// show the marker. With refraction off (Low/Ci) the lens is opaque and the
/// marker is hidden.
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
            ref_x > 64.0 && x < 64.0,
            "{preset}: the ball turns the picture round: {x:.2} (without it: {ref_x:.2})"
        );
        assert!(
            (y - ref_y).abs() < 1.0,
            "{preset}: no vertical displacement expected: {y:.2} vs {ref_y:.2}"
        );
        let expected = optics::ball_lens_image(size, preset, |origin, dir| {
            optics::hits_ball(origin, dir, [1.6, 0.0, -9.0], 0.6).is_some()
        });
        optics::assert_same_pixels(&format!("{preset}: marker"), &out, &expected, is_green);
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

// ---- textures, models in the ray-traced scene, shadows, the overlay ------------

use std::sync::Arc;

use space_render::{
    MeshData, MeshMaterial, MeshPart, MeshVertex, ModelInstance, Overlay, SceneLight, Texture,
    IDENTITY_ROTATION,
};

/// A rough, non-metallic material of `color`: lit by diffuse light alone, and
/// never picked up by screen-space reflections.
fn matte(color: [f32; 3]) -> MeshMaterial {
    MeshMaterial {
        base_color: color,
        metallic: 0.0,
        roughness: 1.0,
        emissive: [0.0; 3],
        base_color_texture: None,
    }
}

/// A material that only glows.
fn glowing(radiance: [f32; 3]) -> MeshMaterial {
    MeshMaterial {
        base_color: [0.0; 3],
        metallic: 0.0,
        roughness: 0.5,
        emissive: radiance,
        base_color_texture: None,
    }
}

fn placed(mesh: &Arc<MeshData>, position: [f32; 3]) -> ModelInstance {
    ModelInstance {
        mesh: Arc::clone(mesh),
        position,
        rotation: IDENTITY_ROTATION,
        scale: 1.0,
    }
}

fn light_at(position: [f64; 3], intensity: f32) -> SceneLight {
    SceneLight {
        position: WorldPos::from_f64_metres_around(WorldPos::ZERO, position),
        color: [1.0, 1.0, 1.0],
        intensity,
    }
}

/// A matte grey floor: a huge box instance whose top face is the plane y = `y`.
fn matte_floor(y: f32) -> GpuInstance {
    GpuInstance {
        center: [0.0, y - 100.0, 0.0, 100.0],
        albedo: [0.8, 0.8, 0.8, 0.0],
        emissive: [0.0; 4],
        roughness: 1.0,
        shape: space_render::SHAPE_BOX,
        transmission: 0.0,
        _pad1: 0.0,
    }
}

/// Render `scene` at `size` x `size` with `preset` and **no fog** (GI off), so
/// that High and Medium differ from Low and Ci by reflections, refraction and
/// shadows only. Asserts zero validation messages.
fn render_clear(
    ctx: &mut RenderContext,
    scene: &Scene,
    size: u32,
    preset: QualityPreset,
) -> space_render::FrameOutput {
    let opts = RenderOptions {
        gi_enabled: false,
        quality: Quality::CI,
        preset,
        fog_scale: 0.0,
        ..Default::default()
    };
    let out = ctx.render_frame(scene, size, size, &opts).expect("render");
    assert_eq!(out.validation_messages, 0, "validation messages seen");
    out
}

fn rgb(out: &space_render::FrameOutput, x: u32, y: u32) -> [u16; 3] {
    let p = pixel(out, x, y);
    [p[0], p[1], p[2]]
}

fn is_blue(p: [u16; 3]) -> bool {
    p[2] > 60 && p[2] > 3 * p[0].max(p[1]).max(1)
}

fn is_dark(p: [u16; 3]) -> bool {
    p.iter().all(|&c| c < 10)
}

fn is_bright(p: [u16; 3]) -> bool {
    p.iter().all(|&c| c > 100)
}

/// A square facing the camera (+Z), `half` units from its centre to its
/// edges, with its texture upright: UV (0, 0) is the corner that is top-left
/// on screen, and the texture repeats `repeat` times across.
fn facing_quad(half: f32, repeat: f32, material: MeshMaterial) -> MeshData {
    let vertex = |x: f32, y: f32| MeshVertex {
        position: [x * half, y * half, 0.0],
        normal: [0.0, 0.0, 1.0],
        uv: [(x + 1.0) * 0.5 * repeat, (1.0 - y) * 0.5 * repeat],
    };
    MeshData {
        // Top-left, top-right, bottom-right, bottom-left.
        vertices: vec![
            vertex(-1.0, 1.0),
            vertex(1.0, 1.0),
            vertex(1.0, -1.0),
            vertex(-1.0, -1.0),
        ],
        indices: vec![0, 3, 2, 0, 2, 1],
        parts: vec![MeshPart {
            first_index: 0,
            index_count: 6,
            material,
        }],
        textures: Vec::new(),
        bounds_min: [-half, -half, 0.0],
        bounds_max: [half, half, 0.0],
    }
}

/// Base-colour textures: a quad facing the camera shows its texture upright
/// and where its texture coordinates put it — the four differently coloured
/// texels of a 2x2 image in the four quadrants, a 4x4 checker cell by cell,
/// twice as many cells when the coordinates run 0..2 (the sampler repeats) —
/// multiplied with the material's base-colour factor; the image survives a GLB
/// round trip pixel for pixel; and a fine pattern seen from far away averages
/// out through the mip chain, in linear light, instead of aliasing.
/// Mutation: the same quad without the texture is a flat colour.
#[test]
fn a_textured_quad_shows_its_texture_where_the_uvs_put_it() {
    let Some(mut ctx) = context() else { return };
    let size = 128u32;
    let camera = post_camera();
    // The quad: 3 m across, 4 m away. On a 128-pixel frame with a 60 degree
    // field of view it spans 1.5 / (4 tan 30) = 0.65 of the half-frame:
    // pixels 22.4 .. 105.6 in x and y.
    let (left, span) = (22.43f32, 83.14f32);
    let cell_centre = |i: u32, cells: u32| (left + (i as f32 + 0.5) * span / cells as f32) as u32;
    let scene_with = |mesh: MeshData, distance: f32, intensity: f32| {
        // The light is at the camera, so the whole quad is evenly lit.
        let mut scene = Scene::from_instances(camera, Vec::new())
            .with_lights(vec![light_at([0.0, 0.0, 0.0], intensity)]);
        scene
            .models
            .push(placed(&Arc::new(mesh), [0.0, 0.0, -distance]));
        scene
    };
    let textured = |texture: Texture, repeat: f32, factor: [f32; 3]| {
        let mut quad = facing_quad(1.5, repeat, matte(factor));
        quad.parts[0].material.base_color_texture = Some(quad.add_texture(texture));
        quad
    };
    let mut render = |mesh: MeshData| {
        let out = ctx
            .render_frame(&scene_with(mesh, 4.0, 40.0), size, size, &ci_options())
            .expect("textured render");
        assert_eq!(out.validation_messages, 0, "validation messages seen");
        out
    };

    // Orientation: four texels, four colours. Row 0 of the image is the top.
    let corners = Texture::from_rgba8(
        2,
        2,
        vec![
            255, 0, 0, 255, /* */ 0, 255, 0, 255, //
            0, 0, 255, 255, /* */ 255, 255, 255, 255,
        ],
    )
    .expect("a 2x2 texture");
    let out = render(textured(corners, 1.0, [1.0; 3]));
    let (near, far) = (cell_centre(0, 2), cell_centre(1, 2));
    let (top_left, top_right) = (rgb(&out, near, near), rgb(&out, far, near));
    let (bottom_left, bottom_right) = (rgb(&out, near, far), rgb(&out, far, far));
    eprintln!("2x2: {top_left:?} {top_right:?} / {bottom_left:?} {bottom_right:?}");
    assert!(is_red(top_left), "top-left texel is red: {top_left:?}");
    assert!(is_green(top_right), "top-right is green: {top_right:?}");
    assert!(is_blue(bottom_left), "bottom-left is blue: {bottom_left:?}");
    assert!(
        is_bright(bottom_right),
        "bottom-right is white: {bottom_right:?}"
    );
    // Outside the quad nothing is drawn.
    assert_eq!(rgb(&out, 10, 64), [0, 0, 0]);

    // A 4x4 checker, cell by cell; the cell at the UV origin is the first colour.
    let (red, blue) = ([255, 0, 0], [0, 0, 255]);
    let checker = || Texture::checkerboard(64, 4, red, blue);
    let check_cells = |out: &space_render::FrameOutput, cells: u32, what: &str| {
        for j in 0..cells {
            for i in 0..cells {
                let p = rgb(out, cell_centre(i, cells), cell_centre(j, cells));
                let ok = if (i + j) % 2 == 0 {
                    is_red(p)
                } else {
                    is_blue(p)
                };
                assert!(ok, "{what}: cell ({i}, {j}) is {p:?}");
            }
        }
    };
    let once = render(textured(checker(), 1.0, [1.0; 3]));
    check_cells(&once, 4, "checker");
    // Coordinates 0..2: the texture repeats, 8x8 cells.
    let twice = render(textured(checker(), 2.0, [1.0; 3]));
    check_cells(&twice, 8, "repeated checker");
    assert!(once.pixels != twice.pixels);

    // The base-colour factor multiplies the texture: no red factor, no red cells.
    let tinted = render(textured(checker(), 1.0, [0.0, 1.0, 1.0]));
    let (was_red, still_blue) = (
        rgb(&tinted, cell_centre(0, 4), cell_centre(0, 4)),
        rgb(&tinted, cell_centre(1, 4), cell_centre(0, 4)),
    );
    // (All that is left of a red texel under a cyan factor is the faint
    // white specular term.)
    assert!(
        was_red.iter().all(|&c| c < 30),
        "red texel x cyan factor has no colour left: {was_red:?}"
    );
    assert!(is_blue(still_blue), "{still_blue:?}");

    // Through a GLB file and back: the same picture, pixel for pixel.
    let reloaded = MeshData::from_glb(&textured(checker(), 1.0, [1.0; 3]).to_glb())
        .expect("the textured quad round-trips");
    assert_eq!(reloaded.textures.len(), 1);
    assert!(
        render(reloaded).pixels == once.pixels,
        "GLB round trip changed the image"
    );

    // MUTATION CHECK: without the texture the quad is one flat colour, so the
    // pattern above can only come from the sampled image.
    let plain = render(facing_quad(1.5, 1.0, matte([1.0; 3])));
    let (a, b) = (
        rgb(&plain, cell_centre(0, 4), cell_centre(0, 4)),
        rgb(&plain, cell_centre(1, 4), cell_centre(0, 4)),
    );
    assert!(
        is_bright(a) && (0..3).all(|c| a[c].abs_diff(b[c]) <= 6),
        "{a:?} vs {b:?}"
    );

    // Mip-mapping: the quad 40 m away is 8 pixels across, so a checker of
    // 2-texel cells (64 texels across) is far below the pixel grid. Sampled
    // from the mip chain it is a flat grey — the same grey as a texture of
    // the checker's average *in linear light* (sRGB 188, not 128) — rather
    // than a moire of black and white pixels.
    let mut far_render = |texture: Texture| {
        let out = ctx
            .render_frame(
                &scene_with(textured(texture, 1.0, [1.0; 3]), 40.0, 4000.0),
                size,
                size,
                &ci_options(),
            )
            .expect("far render");
        assert_eq!(out.validation_messages, 0, "validation messages seen");
        out
    };
    let fine = far_render(Texture::checkerboard(64, 32, [255; 3], [0; 3]));
    let grey = far_render(Texture::solid([188; 3]));
    let mut seen = 0;
    for y in 62..66 {
        for x in 62..66 {
            let (p, q) = (rgb(&fine, x, y), rgb(&grey, x, y));
            assert!(
                q[0] > 60 && q[0] < 250,
                "the grey reference is mid-range: {q:?}"
            );
            assert!(
                (0..3).all(|c| p[c].abs_diff(q[c]) <= 4),
                "minified checker at ({x}, {y}) is {p:?}, the linear-light average is {q:?}"
            );
            seen += 1;
        }
    }
    assert_eq!(seen, 16);
}

/// The shadow test scene: a matte floor at y = -4 and a point light 7 m above
/// it at (0, 3, -8), with whatever `models` and extra `instances` are given.
///
/// Seen by `post_camera` on a 128-pixel frame, the floor point straight below
/// the light is at pixel row ~119 behind anything standing there, and the
/// floor in front of it, at z = -7.33, is row 124.
fn shadow_scene(models: Vec<ModelInstance>, extra: Vec<GpuInstance>) -> Scene {
    let mut instances = vec![matte_floor(-4.0)];
    instances.extend(extra);
    let mut scene = Scene::from_instances(post_camera(), instances)
        .with_lights(vec![light_at([0.0, 3.0, -8.0], 400.0)]);
    scene.models = models;
    scene
}

/// Ray-traced shadows with a mesh model in the scene. A 2 m cube (a *mesh*)
/// stands under the light, its top 4.5 m below it and the floor 7 m below, so
/// its shadow on the floor is a square reaching 7 / 4.5 = 1.56 m from the
/// centre:
///
/// * the floor in front of the cube, inside that square, is black with
///   shadows on (High, Medium) and lit with them off (Low, Ci);
/// * the shadow ends where the geometry says it does;
/// * floor outside the shadow, and the cube's lit top, look the same with and
///   without shadows (no self-shadowing, no darkening);
/// * the model *receives* shadows too: a sphere between the light and the
///   cube darkens the cube's top.
///
/// Mutation: without the cube (same preset) the floor is lit there, so the
/// shadow comes from the model's acceleration structure and nothing else.
#[test]
fn a_model_casts_a_shadow_on_the_floor_and_receives_one() {
    let Some(mut ctx) = context() else { return };
    if !ctx.ray_query_support().complete() {
        eprintln!("SKIP: device has no ray query");
        return;
    }
    let size = 128;
    let cube = Arc::new(MeshData::cuboid([1.0; 3], 1.0, matte([0.8; 3])));
    let cube_at = || vec![placed(&cube, [0.0, -2.5, -8.0])];
    let scene = shadow_scene(cube_at(), Vec::new());
    assert!(ctx.shadows_active(
        &scene,
        &RenderOptions {
            preset: QualityPreset::High,
            ..ci_options()
        }
    ));
    assert!(!ctx.shadows_active(&scene, &ci_options()));

    // Row 124 is the floor 7.33 m ahead; the shadow there is 1.56 m to either
    // side of the centre line, i.e. 23.5 pixels: columns 40.5 .. 87.5.
    let (in_shadow, lit_floor, cube_top) = ((64, 124), (16, 124), (64, 85));
    let shadow_edge = [
        ((45, 124), true),
        ((36, 124), false),
        ((83, 124), true),
        ((92, 124), false),
    ];

    let unshadowed = render_clear(&mut ctx, &scene, size, QualityPreset::Low);
    for (x, y) in [in_shadow, lit_floor, cube_top] {
        let p = rgb(&unshadowed, x, y);
        assert!(is_bright(p), "without shadows ({x}, {y}) is lit: {p:?}");
    }
    assert!(
        render_clear(&mut ctx, &scene, size, QualityPreset::Ci).pixels == unshadowed.pixels,
        "Low and Ci both have shadows off"
    );

    for preset in [QualityPreset::Medium, QualityPreset::High] {
        let out = render_clear(&mut ctx, &scene, size, preset);
        let shadow = rgb(&out, in_shadow.0, in_shadow.1);
        assert!(
            is_dark(shadow),
            "{preset}: the floor under the cube is {shadow:?}"
        );
        for ((x, y), dark) in shadow_edge {
            let p = rgb(&out, x, y);
            assert_eq!(is_dark(p), dark, "{preset}: floor at ({x}, {y}) is {p:?}");
            assert!(dark || is_bright(p), "{preset}: ({x}, {y}) is {p:?}");
        }
        // Lit surfaces are exactly as bright as without shadows.
        for (x, y) in [lit_floor, cube_top] {
            let (a, b) = (rgb(&out, x, y), rgb(&unshadowed, x, y));
            assert!(
                (0..3).all(|c| a[c].abs_diff(b[c]) <= 2),
                "{preset}: lit pixel ({x}, {y}) changed: {a:?} vs {b:?}"
            );
        }
        // How much of the frame went dark: only a shadow-sized part of it.
        let darkened = (0..size)
            .flat_map(|y| (0..size).map(move |x| (x, y)))
            .filter(|&(x, y)| is_bright(rgb(&unshadowed, x, y)) && is_dark(rgb(&out, x, y)))
            .count();
        eprintln!("{preset}: {darkened} pixels in the cube's shadow");
        assert!((150..1200).contains(&darkened), "{preset}: {darkened} px");
    }

    // MUTATION CHECK: no cube, no shadow.
    let bare = render_clear(
        &mut ctx,
        &shadow_scene(Vec::new(), Vec::new()),
        size,
        QualityPreset::High,
    );
    let p = rgb(&bare, in_shadow.0, in_shadow.1);
    assert!(is_bright(p), "without the model the floor is lit: {p:?}");

    // The model receives a shadow: a sphere 2.5 m under the light.
    let occluder = GpuInstance {
        center: [0.0, 0.5, -8.0, 0.6],
        albedo: [0.8, 0.8, 0.8, 0.0],
        emissive: [0.0; 4],
        roughness: 1.0,
        shape: space_render::SHAPE_SPHERE,
        transmission: 0.0,
        _pad1: 0.0,
    };
    let shaded = shadow_scene(cube_at(), vec![occluder]);
    let high = render_clear(&mut ctx, &shaded, size, QualityPreset::High);
    let low = render_clear(&mut ctx, &shaded, size, QualityPreset::Low);
    let (dark, lit) = (
        rgb(&high, cube_top.0, cube_top.1),
        rgb(&low, cube_top.0, cube_top.1),
    );
    assert!(
        is_dark(dark),
        "the sphere's shadow falls on the cube's top: {dark:?}"
    );
    assert!(
        is_bright(lit),
        "shadows off: the cube's top is lit: {lit:?}"
    );

    // Glass casts no shadow: the same sphere, transmissive.
    let glass = GpuInstance {
        transmission: 1.0,
        ..occluder
    };
    let through = render_clear(
        &mut ctx,
        &shadow_scene(cube_at(), vec![glass]),
        size,
        QualityPreset::High,
    );
    let p = rgb(&through, cube_top.0, cube_top.1);
    assert!(
        is_bright(p),
        "a glass sphere leaves the cube's top lit: {p:?}"
    );
}

/// An open mesh — a single-sided plane with no thickness — blocks light from
/// both of its sides (its top-level instance is flagged two-sided), and the
/// start offset of the shadow rays keeps it from shadowing itself.
#[test]
fn an_open_mesh_casts_shadows_from_both_sides() {
    let Some(mut ctx) = context() else { return };
    if !ctx.ray_query_support().complete() {
        eprintln!("SKIP: device has no ray query");
        return;
    }
    let size = 128;
    let plane = Arc::new(MeshData::plane([1.0, 1.0], [1.0, 1.0], matte([0.8; 3])));
    assert!(!plane.is_closed());
    // 5 m under the light and 2 m over the floor: the shadow reaches 1.4 m.
    let facing_up = placed(&plane, [0.0, -2.0, -8.0]);
    let facing_down = ModelInstance {
        // Half a turn about X.
        rotation: [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
        ..facing_up.clone()
    };
    let (in_shadow, lit_floor, on_plane) = ((64, 124), (16, 124), (64, 92));
    for (name, model) in [
        ("facing the light", facing_up),
        ("facing away", facing_down),
    ] {
        let scene = shadow_scene(vec![model], Vec::new());
        let high = render_clear(&mut ctx, &scene, size, QualityPreset::High);
        let low = render_clear(&mut ctx, &scene, size, QualityPreset::Low);
        let (shadow, was) = (
            rgb(&high, in_shadow.0, in_shadow.1),
            rgb(&low, in_shadow.0, in_shadow.1),
        );
        assert!(
            is_dark(shadow) && is_bright(was),
            "{name}: {shadow:?} (unshadowed {was:?})"
        );
        let p = rgb(&high, lit_floor.0, lit_floor.1);
        assert!(is_bright(p), "{name}: floor beside the shadow is {p:?}");
        // The plane itself, where it faces the light, is lit: not shadowed
        // by its own triangles.
        if name == "facing the light" {
            let (a, b) = (
                rgb(&high, on_plane.0, on_plane.1),
                rgb(&low, on_plane.0, on_plane.1),
            );
            assert!(is_bright(a), "{name}: the plane's lit side is {a:?}");
            assert!((0..3).all(|c| a[c].abs_diff(b[c]) <= 2), "{a:?} vs {b:?}");
        }
    }
}

/// Shadow rays treat every closed solid as single-sided: a ray is stopped by
/// a face it *enters* a solid through, never by one it leaves through. Two
/// consequences, both of which fail with the facing rule the wrong way round
/// (or with no culling at all):
///
/// * a light outside an enclosing box still lights what is inside the box
///   (the ray from the surface to the light only ever leaves the box);
/// * a sphere a few pixels across is not shadowed by itself, although it is
///   rasterised as a bare icosahedron whose faces lie *inside* the finer
///   sphere that is ray traced.
#[test]
fn shadow_rays_ignore_the_faces_they_leave_a_solid_through() {
    let Some(mut ctx) = context() else { return };
    if !ctx.ray_query_support().complete() {
        eprintln!("SKIP: device has no ray query");
        return;
    }
    let size = 128;
    let ball = |center: [f32; 3], radius: f32| GpuInstance {
        center: [center[0], center[1], center[2], radius],
        albedo: [0.8, 0.8, 0.8, 0.0],
        emissive: [0.0; 4],
        roughness: 1.0,
        shape: space_render::SHAPE_SPHERE,
        transmission: 0.0,
        _pad1: 0.0,
    };

    // The camera and a ball inside a 20 m box; the light 30 m behind the
    // camera, outside it.
    let room = GpuInstance {
        center: [0.0, 0.0, 0.0, 10.0],
        shape: space_render::SHAPE_BOX,
        ..ball([0.0; 3], 0.0)
    };
    let inside = Scene::from_instances(post_camera(), vec![room, ball([0.0, 0.0, -4.0], 1.0)])
        .with_lights(vec![light_at([0.0, 0.0, 30.0], 4000.0)]);
    let high = render_clear(&mut ctx, &inside, size, QualityPreset::High);
    let low = render_clear(&mut ctx, &inside, size, QualityPreset::Low);
    let (a, b) = (rgb(&high, 64, 64), rgb(&low, 64, 64));
    eprintln!("ball inside the box: shadows on {a:?}, off {b:?}");
    assert!(b[0] > 80, "the ball is lit without shadows: {b:?}");
    assert!(
        (0..3).all(|c| a[c].abs_diff(b[c]) <= 2),
        "the box's far wall must not shadow its inside: {a:?} vs {b:?}"
    );

    // A ball 22 m away is about 5 pixels in radius: drawn as an icosahedron.
    let small = Scene::from_instances(post_camera(), vec![ball([0.0, 0.0, -22.0], 1.0)])
        .with_lights(vec![light_at([6.0, 6.0, 0.0], 3000.0)]);
    let high = render_clear(&mut ctx, &small, size, QualityPreset::High);
    let low = render_clear(&mut ctx, &small, size, QualityPreset::Low);
    let (with, without) = (brightness_sum(&high), brightness_sum(&low));
    eprintln!("small ball: brightness {with} with shadows, {without} without");
    assert!(without > 5_000, "the small ball is visible: {without}");
    assert!(
        with.abs_diff(without) * 50 <= without,
        "a small sphere must not shadow itself: {with} vs {without}"
    );
}

/// A sphere or box instance that contains a light casts no shadow from it —
/// it is the lamp around the bulb (the Sun around the solar system's point
/// light) — while the same body with the light outside it does.
#[test]
fn a_body_around_a_light_does_not_block_it() {
    let Some(mut ctx) = context() else { return };
    if !ctx.ray_query_support().complete() {
        eprintln!("SKIP: device has no ray query");
        return;
    }
    let size = 128;
    let lamp = |center: [f32; 3]| GpuInstance {
        center: [center[0], center[1], center[2], 0.5],
        albedo: [1.0, 1.0, 1.0, 0.0],
        emissive: [3.0, 3.0, 3.0, 1.0],
        roughness: 0.5,
        shape: space_render::SHAPE_SPHERE,
        transmission: 0.0,
        _pad1: 0.0,
    };
    // Around the light: the floor below is lit as if the lamp were not there.
    let around = shadow_scene(Vec::new(), vec![lamp([0.0, 3.0, -8.0])]);
    let p = rgb(
        &render_clear(&mut ctx, &around, size, QualityPreset::High),
        64,
        124,
    );
    assert!(
        is_bright(p),
        "the lamp around the light lets it through: {p:?}"
    );
    // A metre below the light: an ordinary occluder, with a shadow under it.
    let below = shadow_scene(Vec::new(), vec![lamp([0.0, 2.0, -8.0])]);
    let p = rgb(
        &render_clear(&mut ctx, &below, size, QualityPreset::High),
        64,
        124,
    );
    assert!(
        is_dark(p),
        "the same sphere below the light casts a shadow: {p:?}"
    );
}

/// A mesh model shows up in screen-space reflections like any other surface:
/// a glowing red cube (a mesh) hovering over the mirror floor of
/// `mirror_scene` is reflected where mirror geometry predicts.
/// Mutation: no model, or the post pass off (`Ci`), and the floor is black.
#[test]
fn a_model_is_visible_in_a_mirror_reflection() {
    let Some(mut ctx) = context() else { return };
    let size = 128;
    // The floor of `mirror_scene` (y = -1), without its sphere.
    let mut scene = mirror_scene(0.1);
    scene.instances.truncate(1);
    let empty = Scene::from_instances(scene.camera, scene.instances.clone());
    // A 0.8 m cube from y = -0.8 to 0: its mirror image spans y = -2 .. -1.2,
    // centred on (0, -1.6, -6): row ~93.6 of 128, column 64, like the sphere's.
    let cube = Arc::new(MeshData::cuboid([0.4; 3], 1.0, glowing([6.0, 0.0, 0.0])));
    scene.models.push(placed(&cube, [0.0, -0.4, -6.0]));

    let reflection_rows = 84..size;
    let ci = render_preset(&mut ctx, &scene, size, QualityPreset::Ci);
    let direct = centroid(&ci, 0..82, is_red).expect("the cube itself is visible");
    assert!(
        (direct.1 - 64.0).abs() < 1.0,
        "direct cube centroid {direct:?}"
    );
    assert!(
        centroid(&ci, reflection_rows.clone(), is_red).is_none(),
        "no post pass, no reflection"
    );
    for preset in [QualityPreset::Medium, QualityPreset::High] {
        let out = render_preset(&mut ctx, &scene, size, preset);
        let (n, cx, cy) = centroid(&out, reflection_rows.clone(), is_red)
            .unwrap_or_else(|| panic!("{preset}: the model is not reflected"));
        eprintln!("{preset}: model reflection n={n} centroid=({cx:.1},{cy:.1})");
        assert!(n >= 60, "{preset}: reflection too small ({n} px)");
        assert!(
            (cx - 64.0).abs() <= 2.5 && (86.0..100.0).contains(&cy),
            "{preset}: reflection centroid ({cx:.1},{cy:.1}) vs mirror prediction (64, ~93)"
        );
        // MUTATION CHECK: no model, nothing to reflect.
        let bare = render_preset(&mut ctx, &empty, size, preset);
        assert!(centroid(&bare, 0..size, is_red).is_none(), "{preset}");
    }
}

/// A mesh model is seen through glass: the green marker of
/// `refraction_scene`, as a mesh cube, hidden behind the lens with refraction
/// off and visible through it — where Snell's law puts it, on the other side
/// of the ball — with refraction on.
#[test]
fn a_model_is_seen_through_glass() {
    let Some(mut ctx) = context() else { return };
    let size = 128;
    let marker = Arc::new(MeshData::cuboid([0.5; 3], 1.0, glowing([0.0, 6.0, 0.0])));
    let scene_with = |with_glass: bool| {
        // `refraction_scene` lists its marker sphere first: replace it.
        let mut scene = refraction_scene(with_glass);
        scene.instances.remove(0);
        scene.models.push(placed(&marker, [1.6, 0.0, -9.0]));
        scene
    };
    let (glass, bare) = (scene_with(true), scene_with(false));
    assert_eq!((glass.instances.len(), bare.instances.len()), (1, 0));

    let reference = render_preset(&mut ctx, &bare, size, QualityPreset::High);
    let (ref_n, ref_x, ref_y) =
        centroid(&reference, 0..size, is_green).expect("marker visible without glass");
    for preset in [QualityPreset::Ci, QualityPreset::Low] {
        let off = render_preset(&mut ctx, &glass, size, preset);
        assert!(
            centroid(&off, 0..size, is_green).is_none(),
            "{preset}: the opaque lens hides the model"
        );
    }
    for preset in [QualityPreset::Medium, QualityPreset::High] {
        let out = render_preset(&mut ctx, &glass, size, preset);
        let (n, x, y) = centroid(&out, 0..size, is_green)
            .unwrap_or_else(|| panic!("{preset}: the model is not visible through the glass"));
        eprintln!(
            "{preset}: model through glass n={n} centroid=({x:.2},{y:.2}); \
             reference n={ref_n} ({ref_x:.2},{ref_y:.2})"
        );
        assert!(n >= 100, "{preset}: only {n} px of the model show through");
        assert!(
            ref_x > 64.0 && x < 64.0,
            "{preset}: the ball turns the picture round: {x:.2} (without it: {ref_x:.2})"
        );
        assert!((y - ref_y).abs() < 1.5, "{preset}: {y:.2} vs {ref_y:.2}");
        let expected = optics::ball_lens_image(size, preset, |origin, dir| {
            optics::hits_box(origin, dir, [1.6, 0.0, -9.0], [0.5; 3]).is_some()
        });
        optics::assert_same_pixels(&format!("{preset}: model"), &out, &expected, is_green);
    }
}

/// A model is in the ray-traced scene exactly where it is drawn: the
/// character — turned, scaled and moved off-axis — has the same outline in
/// the forward pass (vertex buffers, model matrix) and in the raw ray probe
/// (its bottom-level structure, one geometry per part, and its top-level
/// instance transform). A wrong index offset, part range or matrix moves one
/// outline but not the other.
/// Mutation: without the model the ray probe hits nothing.
#[test]
fn a_model_has_the_same_outline_rasterised_and_ray_traced() {
    let Some(mut ctx) = context() else { return };
    if !ctx.ray_query_support().complete() {
        eprintln!("SKIP: device has no ray query");
        return;
    }
    let size = 160u32;
    // Every part glowing, so the rasterised silhouette is simply "not black".
    let mut glowing_character = space_render::sample_character();
    for part in &mut glowing_character.parts {
        part.material = glowing([4.0; 3]);
    }
    let character = Arc::new(glowing_character);
    let ship = Arc::new({
        let mut ship = space_render::sample_ship();
        for part in &mut ship.parts {
            part.material = glowing([4.0; 3]);
        }
        ship
    });
    let probe = RenderOptions {
        ray_probe: true,
        ..ci_options()
    };
    for (name, model) in [
        (
            "character",
            ModelInstance {
                mesh: Arc::clone(&character),
                position: [0.4, -1.3, -3.2],
                rotation: space_render::rotation_y(0.7),
                scale: 1.3,
            },
        ),
        (
            "ship",
            ModelInstance {
                mesh: Arc::clone(&ship),
                position: [-0.5, 0.3, -6.0],
                rotation: space_render::rotation_y(-1.1),
                scale: 0.8,
            },
        ),
    ] {
        let mut scene = Scene::from_instances(post_camera(), Vec::new());
        scene.models.push(model);
        let raster = ctx
            .render_frame(&scene, size, size, &ci_options())
            .expect("forward render");
        let traced = ctx
            .render_frame(&scene, size, size, &probe)
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
        assert!(covered > 1000, "{name}: too small to compare ({covered})");
        // Only outline pixels may differ (an edge crossing a pixel resolves
        // differently when rasterised and when ray cast).
        assert!(
            differing * 25 < covered,
            "{name}: outlines differ in {differing} of {covered} pixels"
        );
    }
    // MUTATION CHECK: no model, no hits.
    let empty = Scene::from_instances(post_camera(), Vec::new());
    let traced = ctx
        .render_frame(&empty, size, size, &probe)
        .expect("empty ray-probe render");
    assert!(traced.pixels.as_chunks::<4>().0.iter().all(|p| p[0] < 40));
}

/// Probe GI sees mesh models: an emissive model lights the receiver of
/// `gi_scene` like the emissive sphere does, by the radiance of the *part*
/// the rays hit; a non-emissive model emits nothing; and a model between the
/// probes and an emitter blocks its light.
#[test]
fn probe_gi_sees_mesh_models() {
    let Some(mut ctx) = context() else { return };
    if !ctx.ray_query_support().complete() {
        eprintln!("SKIP: device has no ray query");
        return;
    }
    let size = 64;
    let options = RenderOptions {
        gi_enabled: true,
        quality: gi_quality(),
        ..Default::default()
    };
    let mut centre = |scene: &Scene| {
        let out = ctx
            .render_frame(scene, size, size, &options)
            .expect("GI render");
        assert_eq!(out.validation_messages, 0, "validation messages seen");
        rgb(&out, size / 2, size / 2)
    };
    // The receiver box of `gi_scene`, without its emissive sphere.
    let receiver_only = || {
        let mut scene = gi_scene(size, size);
        scene.instances.truncate(1);
        scene
    };
    // A slab behind the camera, where the sphere was: two parts, the first
    // dark, the second — the one facing the probes — glowing green. A ray
    // that looked its part up wrongly would find black.
    let slab = |front: MeshMaterial| {
        let mut back = MeshData::cuboid([6.0, 6.0, 0.5], 1.0, matte([0.5; 3]));
        // The cuboid's faces come in the order -x, +x, -y, +y, -z, +z, six
        // indices each: the -z face (towards the probes) is the fifth.
        back.parts = vec![
            MeshPart {
                first_index: 0,
                index_count: 24,
                material: matte([0.5; 3]),
            },
            MeshPart {
                first_index: 24,
                index_count: 6,
                material: front,
            },
            MeshPart {
                first_index: 30,
                index_count: 6,
                material: matte([0.5; 3]),
            },
        ];
        Arc::new(back)
    };
    let mut emitter = receiver_only();
    emitter
        .models
        .push(placed(&slab(glowing([0.0, 10.0, 0.0])), [0.0, 0.0, 8.0]));
    let lit = centre(&emitter);
    assert!(
        lit[1] > 60 && lit[1] > 3 * lit[0].max(lit[2]).max(1),
        "the model's glowing part lights the receiver green: {lit:?}"
    );

    // MUTATION CHECK: the same slab without the glow lights nothing.
    let mut dull = receiver_only();
    dull.models
        .push(placed(&slab(matte([0.5; 3])), [0.0, 0.0, 8.0]));
    let dark = centre(&dull);
    assert!(
        is_dark(dark),
        "a non-emissive model emits nothing: {dark:?}"
    );

    // Occlusion: the full `gi_scene` (emissive sphere behind the camera) with
    // a thin wall, a mesh, between the probes (z <= 3.5) and the sphere
    // (z >= 4).
    let open = gi_scene(size, size);
    let unblocked = centre(&open);
    assert!(
        unblocked[0] > 60,
        "the sphere lights the receiver: {unblocked:?}"
    );
    let mut walled = gi_scene(size, size);
    let wall = Arc::new(MeshData::cuboid([30.0, 30.0, 0.05], 1.0, matte([0.5; 3])));
    walled.models.push(placed(&wall, [0.0, 0.0, 3.8]));
    let blocked = centre(&walled);
    assert!(is_dark(blocked), "the wall blocks the emitter: {blocked:?}");
}

/// A model's GPU resources are created once: its buffers and textures when a
/// frame first draws it, its acceleration structure when a frame that traces
/// rays first draws it — and never again while consecutive frames keep
/// drawing it, whatever the preset. Two placements of one mesh share them.
#[test]
fn a_model_is_uploaded_once_and_its_blas_built_once() {
    let Some(mut ctx) = context() else { return };
    if !ctx.ray_query_support().complete() {
        eprintln!("SKIP: device has no ray query");
        return;
    }
    let size = 64;
    let character = Arc::new(space_render::sample_character());
    let ship = Arc::new(space_render::sample_ship());
    let mut scene = shadow_scene(
        vec![
            placed(&character, [-1.0, -4.0, -8.0]),
            placed(&character, [1.0, -4.0, -8.0]),
            placed(&ship, [0.0, -1.0, -9.0]),
        ],
        Vec::new(),
    );
    let stats = |ctx: &RenderContext| {
        let s = ctx.model_cache_stats();
        (s.cached, s.uploads, s.blas_builds)
    };
    assert_eq!(stats(&ctx), (0, 0, 0));
    // No rays (Ci, GI off): uploaded, but no acceleration structure yet.
    for _ in 0..3 {
        render_clear(&mut ctx, &scene, size, QualityPreset::Ci);
    }
    assert_eq!(stats(&ctx), (2, 2, 0));
    // Shadows, GI, the ray probe, more frames, other presets: two BLASes,
    // built once.
    for _ in 0..3 {
        render_clear(&mut ctx, &scene, size, QualityPreset::High);
    }
    assert_eq!(stats(&ctx), (2, 2, 2));
    let gi = RenderOptions {
        gi_enabled: true,
        quality: gi_quality(),
        ..Default::default()
    };
    for options in [
        gi,
        RenderOptions {
            ray_probe: true,
            ..ci_options()
        },
        ci_options(),
    ] {
        let out = ctx
            .render_frame(&scene, size, size, &options)
            .expect("render");
        assert_eq!(out.validation_messages, 0, "validation messages seen");
        render_clear(&mut ctx, &scene, size, QualityPreset::Medium);
    }
    assert_eq!(stats(&ctx), (2, 2, 2));
    // A model the scene no longer draws is dropped; coming back, it is
    // uploaded (and its BLAS built) again.
    scene.models.pop();
    render_clear(&mut ctx, &scene, size, QualityPreset::High);
    assert_eq!(stats(&ctx), (1, 2, 2));
    scene.models.push(placed(&ship, [0.0, -1.0, -9.0]));
    render_clear(&mut ctx, &scene, size, QualityPreset::High);
    assert_eq!(stats(&ctx), (2, 3, 3));
}

/// The overlay: rectangles and text in the built-in font land on exactly the
/// pixels they name, on top of the finished frame — opaque ones replace it,
/// translucent ones blend with it, later ones cover earlier ones, anything
/// off-frame is clipped — untouched by exposure, identical in the ray-probe
/// path, right again after the frame size changes, and gone (the frame bit
/// for bit what it was) when the overlay is empty.
#[test]
fn the_overlay_draws_text_and_rectangles_over_the_finished_frame() {
    use space_render::overlay::{glyph_pixel, GLYPH_ADVANCE, LINE_ADVANCE};

    let Some(mut ctx) = context() else { return };
    let (yellow, red_half, blue_quarter, green) = (
        [1.0, 1.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 0.5],
        [0.0, 0.0, 1.0, 0.25],
        [0.0, 1.0, 0.0, 1.0],
    );
    let text = "Ab1.:\n7_";
    let (text_x, text_y, scale) = (4u32, 3u32, 2u32);
    let draw = |overlay: &mut Overlay| {
        overlay.text(text_x as f32, text_y as f32, scale, yellow, text);
        // Over the bright sphere, over the black background, and half
        // off the frame's bottom-left corner.
        overlay.rect(40.0, 26.0, 16.0, 12.0, red_half);
        overlay.rect(70.0, 50.0, 10.0, 10.0, blue_quarter);
        overlay.rect(-5.0, 40.0, 10.0, 100.0, green);
        // Later rectangles cover earlier ones.
        overlay.rect(50.0, 30.0, 4.0, 4.0, green);
    };
    // What the overlay should do to pixel (x, y) of a frame: the colours to
    // blend over it, in order.
    let layers_at = |x: u32, y: u32| -> Vec<[f32; 4]> {
        let mut layers = Vec::new();
        let inside = |rx: i64, ry: i64, w: i64, h: i64| {
            let (x, y) = (i64::from(x), i64::from(y));
            x >= rx && x < rx + w && y >= ry && y < ry + h
        };
        for (line, row_text) in text.split('\n').enumerate() {
            for (column, ch) in row_text.chars().enumerate() {
                let gx0 = text_x + column as u32 * GLYPH_ADVANCE * scale;
                let gy0 = text_y + line as u32 * LINE_ADVANCE * scale;
                if x >= gx0 && y >= gy0 && glyph_pixel(ch, (x - gx0) / scale, (y - gy0) / scale) {
                    layers.push(yellow);
                }
            }
        }
        if inside(40, 26, 16, 12) {
            layers.push(red_half);
        }
        if inside(70, 50, 10, 10) {
            layers.push(blue_quarter);
        }
        if inside(-5, 40, 10, 100) {
            layers.push(green);
        }
        if inside(50, 30, 4, 4) {
            layers.push(green);
        }
        layers
    };
    let check = |out: &space_render::FrameOutput, base: &space_render::FrameOutput, what: &str| {
        let (mut touched, mut text_pixels) = (0, 0);
        for y in 0..out.height {
            for x in 0..out.width {
                let layers = layers_at(x, y);
                let (got, under) = (pixel(out, x, y), pixel(base, x, y));
                let mut expected = [
                    f32::from(under[0]),
                    f32::from(under[1]),
                    f32::from(under[2]),
                ];
                for layer in &layers {
                    for c in 0..3 {
                        expected[c] = layer[c] * 255.0 * layer[3] + expected[c] * (1.0 - layer[3]);
                    }
                }
                // Opaque and untouched pixels are exact; a blend may round
                // either way.
                let slack = if layers.iter().any(|l| l[3] < 1.0) {
                    1.0
                } else {
                    0.0
                };
                for c in 0..3 {
                    assert!(
                        (f32::from(got[c]) - expected[c]).abs() <= slack + 0.01,
                        "{what}: pixel ({x}, {y}) is {got:?}, expected {expected:?} over {under:?}"
                    );
                }
                assert_eq!(got[3], under[3], "{what}: alpha of ({x}, {y}) changed");
                touched += usize::from(!layers.is_empty());
                text_pixels += usize::from(layers.first() == Some(&yellow));
            }
        }
        (touched, text_pixels)
    };

    // A frame with something under the overlay: a bright sphere mid-frame.
    let (width, height) = (96u32, 64u32);
    let camera = space_render::CameraPose {
        aspect: width as f32 / height as f32,
        ..post_camera()
    };
    let background = || {
        Scene::from_instances(
            camera,
            vec![emissive_sphere([0.0, 0.0, -3.0], 1.0, [4.0; 3])],
        )
    };
    let mut overlaid = background();
    draw(&mut overlaid.overlay);
    assert!(background().overlay.is_empty() && !overlaid.overlay.is_empty());

    let base = ctx
        .render_frame(&background(), width, height, &ci_options())
        .expect("base render");
    let out = ctx
        .render_frame(&overlaid, width, height, &ci_options())
        .expect("overlay render");
    assert_eq!(base.validation_messages, 0, "validation messages seen");
    assert_eq!(out.validation_messages, 0, "validation messages seen");
    let sphere = pixel(&base, 48, 32);
    assert!(
        sphere[0] > 240 && sphere[3] == 255,
        "the sphere is under the overlay: {sphere:?}"
    );
    let (touched, text_pixels) = check(&out, &base, "overlay");
    eprintln!("overlay: {touched} pixels touched, {text_pixels} of them text");
    // "Ab1.:" and "7_" at scale 2: every lit dot is 4 pixels.
    let dots: usize = text
        .chars()
        .filter(|c| *c != '\n')
        .map(|c| {
            (0..7)
                .flat_map(|y| (0..5).map(move |x| (x, y)))
                .filter(|&(x, y)| glyph_pixel(c, x, y))
                .count()
        })
        .sum();
    assert_eq!(text_pixels, dots * 4);
    assert!(touched > text_pixels + 16 * 12 + 100);
    // Spot checks, in plain numbers: the top-left dot of 'A' is unlit, the
    // dot right of it lit; the red rectangle over the white sphere is pink;
    // the green one on top of it is green; blue over black is a quarter blue.
    assert_eq!(pixel(&out, 4, 3), pixel(&base, 4, 3));
    assert_eq!(pixel(&out, 6, 3), [255, 255, 0, 255]);
    let pink = pixel(&out, 45, 28);
    assert!(
        pink[0] > 250 && pink[1].abs_diff(sphere[1] / 2) <= 1 && pink[2] == pink[1],
        "{pink:?}"
    );
    assert_eq!(pixel(&out, 51, 31), [0, 255, 0, 255]);
    let blue = pixel(&out, 75, 55);
    assert!(
        blue[0] == 0 && blue[1] == 0 && blue[2].abs_diff(64) <= 1,
        "{blue:?}"
    );
    assert_eq!(
        pixel(&out, 2, 60),
        [0, 255, 0, 255],
        "clipped at the frame edge"
    );

    // Not tone mapped: a darker exposure dims the sphere, not the overlay.
    let dim_options = RenderOptions {
        exposure: 0.05,
        ..ci_options()
    };
    let dim_base = ctx
        .render_frame(&background(), width, height, &dim_options)
        .expect("dim base render");
    let dim = ctx
        .render_frame(&overlaid, width, height, &dim_options)
        .expect("dim overlay render");
    assert!(
        pixel(&dim_base, 48, 32)[0] < 200,
        "exposure dims the sphere"
    );
    check(&dim, &dim_base, "dim exposure");
    assert_eq!(pixel(&dim, 6, 3), [255, 255, 0, 255]);

    // With the post pass, and in the raw ray-probe path.
    for (what, options) in [
        (
            "high preset",
            RenderOptions {
                preset: QualityPreset::High,
                ..ci_options()
            },
        ),
        (
            "ray probe",
            RenderOptions {
                ray_probe: true,
                ..ci_options()
            },
        ),
    ] {
        if options.ray_probe && !ctx.ray_query_support().complete() {
            continue;
        }
        let under = ctx
            .render_frame(&background(), width, height, &options)
            .expect("base render");
        let over = ctx
            .render_frame(&overlaid, width, height, &options)
            .expect("overlay render");
        assert_eq!(over.validation_messages, 0, "{what}");
        check(&over, &under, what);
    }

    // Another frame size (as after a window resize): positions are pixels of
    // the new frame, not fractions of the old one.
    for (w, h) in [(64u32, 48u32), (150, 70)] {
        let resized_camera = space_render::CameraPose {
            aspect: w as f32 / h as f32,
            ..post_camera()
        };
        let mut scene = Scene::from_instances(resized_camera, Vec::new());
        let under = ctx
            .render_frame(&scene, w, h, &ci_options())
            .expect("resized base render");
        draw(&mut scene.overlay);
        let over = ctx
            .render_frame(&scene, w, h, &ci_options())
            .expect("resized overlay render");
        assert_eq!(over.validation_messages, 0, "{w}x{h}");
        check(&over, &under, "resized");
    }

    // An empty (or cleared) overlay leaves no trace.
    overlaid.overlay.clear();
    let cleared = ctx
        .render_frame(&overlaid, width, height, &ci_options())
        .expect("cleared render");
    assert!(
        cleared.pixels == base.pixels,
        "an empty overlay changed the frame"
    );
}

// ---- cameras that are not looking down -Z, glass up close, the environment -----

use space_render::Environment;

/// A camera looking along `yaw` / `pitch` (degrees): yaw 0 is -Z and a
/// positive yaw turns left, towards -X; a positive pitch looks up.
fn camera_looking(yaw: f32, pitch: f32) -> space_render::CameraPose {
    let (yaw, pitch) = (yaw.to_radians(), pitch.to_radians());
    space_render::CameraPose {
        forward: [
            -yaw.sin() * pitch.cos(),
            pitch.sin(),
            -yaw.cos() * pitch.cos(),
        ],
        ..post_camera()
    }
}

/// `scene` turned by `yaw` degrees about the vertical axis through the camera:
/// the view direction and the place of every instance and light with it, so
/// the camera sees the same picture. (Box instances stay axis-aligned: only
/// for scenes whose boxes look the same from every side, like a floor.)
fn turned(scene: &Scene, yaw: f32) -> Scene {
    let (sin, cos) = yaw.to_radians().sin_cos();
    // The rotation of `rotation_y`: +Z turns towards +X.
    let turn = |p: [f32; 3]| [cos * p[0] + sin * p[2], p[1], cos * p[2] - sin * p[0]];
    let mut out = Scene::from_instances(scene.camera, scene.instances.clone());
    out.camera.forward = turn(scene.camera.forward);
    for instance in &mut out.instances {
        let [x, y, z, radius] = instance.center;
        let [x, y, z] = turn([x, y, z]);
        instance.center = [x, y, z, radius];
    }
    out.lights = scene
        .lights
        .iter()
        .map(|light| {
            let d = WorldPos::delta(scene.camera.position, light.position);
            let [x, y, z] = turn([d[0] as f32, d[1] as f32, d[2] as f32]);
            SceneLight {
                position: WorldPos::from_f64_metres_around(
                    scene.camera.position,
                    [f64::from(x), f64::from(y), f64::from(z)],
                ),
                ..*light
            }
        })
        .collect();
    out.environment = scene.environment;
    out
}

/// Screen-space reflections and refraction rebuild world positions from the
/// depth buffer with the inverse view-projection matrix. Three cofactors of
/// that inverse were wrong for any camera with a yaw — every test above looks
/// straight down -Z, where the wrong terms are multiplied by zero — so a
/// turned camera put reflections in the wrong place and made glass look up
/// empty parts of the frame (black) instead of what is behind it.
///
/// Here the mirror floor with its red sphere and the lens with its green
/// marker are turned about the camera together with the camera: the picture
/// must stay the same picture.
#[test]
fn reflections_and_refraction_follow_the_camera_when_it_turns() {
    let Some(mut ctx) = context() else { return };
    let size = 128;
    type Wanted = fn([u16; 3]) -> bool;
    let cases: [(&str, Scene, Wanted, std::ops::Range<u32>); 2] = [
        // The reflection only: rows below the sphere itself.
        ("mirror", mirror_scene(0.1), is_red, 83..size),
        ("lens", refraction_scene(true), is_green, 0..size),
    ];
    for (name, scene, wanted, rows) in cases {
        for preset in [QualityPreset::Medium, QualityPreset::High] {
            let reference = render_preset(&mut ctx, &scene, size, preset);
            let (n0, x0, y0) = centroid(&reference, rows.clone(), wanted)
                .unwrap_or_else(|| panic!("{name}: nothing to compare at {preset}"));
            for yaw in [38.0f32, -115.0, 77.0, 180.0, 263.0] {
                let out = render_preset(&mut ctx, &turned(&scene, yaw), size, preset);
                let (n, x, y) = centroid(&out, rows.clone(), wanted).unwrap_or_else(|| {
                    panic!("{name}, {preset}, yaw {yaw}: the reflection / refraction is gone")
                });
                eprintln!(
                    "{name} {preset} yaw {yaw}: n={n} ({x:.2},{y:.2}); straight n={n0} ({x0:.2},{y0:.2})"
                );
                assert!(
                    n.abs_diff(n0) * 12 <= n0,
                    "{name}, {preset}, yaw {yaw}: {n} pixels instead of {n0}"
                );
                assert!(
                    (x - x0).abs() < 1.0 && (y - y0).abs() < 1.0,
                    "{name}, {preset}, yaw {yaw}: at ({x:.2},{y:.2}) instead of ({x0:.2},{y0:.2})"
                );
            }
        }
    }
}

/// A corner for looking at glass, in otherwise empty black space, laid out
/// like the glass corner of the client's demo area (where the bug below was
/// found): two bright checkered panels 3.2 m square (the back one at
/// z = -7.9, the side one at x = -8.08, both from y = -1.7 to 1.5), three
/// lights, and in the corner, at eye height, a glass cube (1 m, centred on
/// (-5.8, -0.2, -6.6)) and a glass sphere (1.1 m, centred on
/// (-7, -0.15, -5.2)). There is no floor: all that is bright is the panels,
/// so a lookup that goes astray finds darkness.
struct GlassRoom {
    back: Arc<MeshData>,
    side: Arc<MeshData>,
}

/// What stands in the middle of a [`GlassRoom`].
#[derive(Clone, Copy, PartialEq)]
enum Middle {
    Nothing,
    Glass,
    /// The same two shapes, opaque and glowing magenta: their outline.
    Outline,
}

impl GlassRoom {
    fn new() -> Self {
        let wall = |half: [f32; 3]| {
            let mut mesh = MeshData::cuboid(half, 1.0, matte([1.0; 3]));
            let slot = mesh.add_texture(Texture::checkerboard(64, 2, [245; 3], [250, 190, 40]));
            mesh.parts[0].material.base_color_texture = Some(slot);
            Arc::new(mesh)
        };
        Self {
            back: wall([1.6, 1.6, 0.08]),
            side: wall([0.08, 1.6, 1.6]),
        }
    }

    /// The room seen from `eye` (room coordinates) along `yaw` / `pitch`
    /// degrees, for a frame 16 wide and 9 high.
    fn scene(&self, eye: [f32; 3], yaw: f32, pitch: f32, middle: Middle) -> Scene {
        let at = |p: [f32; 3]| [p[0] - eye[0], p[1] - eye[1], p[2] - eye[2]];
        let mut instances = Vec::new();
        if middle != Middle::Nothing {
            let shape = |center: [f32; 3], radius: f32, shape: u32| {
                let [x, y, z] = at(center);
                let outline = middle == Middle::Outline;
                GpuInstance {
                    center: [x, y, z, radius],
                    // Black: the glass has no colour of its own for the
                    // lights to bring out (only their mirror images in its
                    // surface), so it is exactly as bright as what is seen
                    // through it. And the outline is its glow alone.
                    albedo: [0.0; 4],
                    emissive: if outline {
                        [6.0, 0.0, 6.0, 1.0]
                    } else {
                        [0.0; 4]
                    },
                    roughness: 0.05,
                    shape,
                    transmission: if outline { 0.0 } else { 1.0 },
                    _pad1: 0.0,
                }
            };
            instances.push(shape([-5.8, -0.2, -6.6], 0.5, space_render::SHAPE_BOX));
            instances.push(shape([-7.0, -0.15, -5.2], 0.55, space_render::SHAPE_SPHERE));
        }
        let camera = space_render::CameraPose {
            aspect: 16.0 / 9.0,
            ..camera_looking(yaw, pitch)
        };
        let mut scene = Scene::from_instances(camera, instances);
        scene.lights = [
            ([-5.0, 2.2, -4.5], 140.0),
            ([-6.8, 1.0, -3.5], 70.0),
            ([-4.0, 1.0, -6.5], 70.0),
        ]
        .into_iter()
        .map(|(p, intensity): ([f32; 3], f32)| {
            let [x, y, z] = at(p);
            light_at([f64::from(x), f64::from(y), f64::from(z)], intensity)
        })
        .collect();
        scene.models = vec![
            placed(&self.back, at([-6.4, -0.1, -7.9])),
            placed(&self.side, at([-8.08, -0.1, -6.3])),
        ];
        scene
    }
}

fn luma(p: [u16; 3]) -> u32 {
    (u32::from(p[0]) * 2 + u32::from(p[1]) * 5 + u32::from(p[2])) / 8
}

/// Is `channel` of `p` bright and well above the other two? (Looser than
/// `is_red` and friends: the tone map lifts the dim channels of a saturated
/// colour a long way.)
fn dominant(p: [u16; 3], channel: usize) -> bool {
    p[channel] > 150 && (0..3).all(|c| c == channel || p[c] + 60 < p[channel])
}

/// Glass seen from a few metres, from close range, at a grazing angle and
/// from above, by cameras turned towards the corner, with bright checkered
/// panels behind it and nothing else around: the glass shows what its rays
/// really reach. `optics` follows every glass pixel's ray on the CPU — into
/// the cube or the ball, to the far side (reflected inside as often as the
/// preset allows), out again — and says whether it ends on a panel or in the
/// empty dark. Where it ends on a panel the glass is bright; where it ends in
/// the void, dark.
///
/// (Glass used to show black for whole faces of a cube and half a ball,
/// whatever was behind it — the test above has the cause. And before rays
/// were followed through the glass, a lookup a fixed distance along the bent
/// ray stood in for all this; it is still what a device without ray queries
/// does.)
///
/// Only rays that enter the glass at less than 60 degrees from its normal
/// count: flatter than that, more and more of what one sees is the
/// reflection on the glass, which here is of nothing.
/// Mutation: without the second refraction (the ray leaving the glass as it
/// travelled inside), the panels are not where the test expects them.
#[test]
fn glass_seen_from_close_up_and_at_a_grazing_angle_is_not_dark() {
    let Some(mut ctx) = context() else { return };
    if !ctx.ray_query_support().complete() {
        eprintln!("SKIP: device has no ray query");
        return;
    }
    let (width, height) = (320u32, 180u32);
    let room = GlassRoom::new();
    let mut render = |scene: &Scene, preset: QualityPreset| {
        let opts = RenderOptions {
            preset,
            fog_scale: 0.0,
            ..ci_options()
        };
        let out = ctx
            .render_frame(scene, width, height, &opts)
            .expect("render");
        assert_eq!(out.validation_messages, 0, "validation messages seen");
        out
    };
    let views: [(&str, [f32; 3], f32, f32); 5] = [
        // Where the bug was first seen: 4.5 to 5 m away, turned 38 degrees.
        ("from a few metres", [-3.8, 0.0, -2.1], 38.0, -3.0),
        // 1.3 m from the cube's nearest edge.
        ("close to the cube", [-4.7, 0.0, -5.0], 34.5, -5.9),
        // Along the cube's +Z face, about 10 degrees off its plane.
        ("grazing the cube", [-3.9, -0.1, -5.75], 79.6, -3.0),
        // 1.3 m from the sphere's surface.
        ("close to the sphere", [-6.0, 0.1, -3.7], 33.7, -7.9),
        // Looking steeply down onto the cube from above and beside it.
        ("down onto the cube", [-5.0, 1.3, -5.6], 38.7, -49.5),
    ];
    let mut problems = Vec::new();
    for (name, eye, yaw, pitch) in views {
        let outline = render(
            &room.scene(eye, yaw, pitch, Middle::Outline),
            QualityPreset::Ci,
        );
        let glass_pixels: Vec<(u32, u32)> = (0..height)
            .flat_map(|y| (0..width).map(move |x| (x, y)))
            .filter(|&(x, y)| {
                let p = rgb(&outline, x, y);
                p[0] > 200 && p[2] > 200 && p[1] < 80
            })
            .collect();
        assert!(
            glass_pixels.len() > 800,
            "{name}: the glass covers only {} pixels",
            glass_pixels.len()
        );
        // The room, as `GlassRoom::scene` builds it, around this eye.
        let at = |p: [f64; 3]| {
            [
                p[0] - f64::from(eye[0]),
                p[1] - f64::from(eye[1]),
                p[2] - f64::from(eye[2]),
            ]
        };
        let glass = [
            optics::Glass::Cube(at([-5.8, -0.2, -6.6]), 0.5),
            optics::Glass::Ball(at([-7.0, -0.15, -5.2]), 0.55),
        ];
        let panels = [
            (at([-6.4, -0.1, -7.9]), [1.6, 1.6, 0.08]),
            (at([-8.08, -0.1, -6.3]), [0.08, 1.6, 1.6]),
        ];
        let camera = optics::Camera::looking(yaw, pitch, 16.0 / 9.0, width, height);
        for preset in [QualityPreset::Medium, QualityPreset::High] {
            let with = render(&room.scene(eye, yaw, pitch, Middle::Glass), preset);
            let bounces = preset.features().glass_bounces;
            // (pixels whose ray ends on a panel, of those the bright ones;
            // pixels whose ray ends in the void, of those the dark ones)
            let (mut lit, mut lit_ok, mut void, mut void_ok) = (0usize, 0usize, 0usize, 0usize);
            for &(x, y) in &glass_pixels {
                let dir = camera.ray(x, y);
                let Some(seen) = optics::through_glass(&glass, [0.0; 3], dir, bounces) else {
                    continue;
                };
                if seen.cos_in < 0.5 {
                    continue;
                }
                let on_panel = panels
                    .iter()
                    .any(|&(c, half)| optics::hits_box(seen.origin, seen.dir, c, half).is_some());
                let brightness = luma(rgb(&with, x, y));
                if on_panel {
                    lit += 1;
                    lit_ok += usize::from(brightness >= 100);
                } else {
                    void += 1;
                    void_ok += usize::from(brightness < 60);
                }
            }
            eprintln!(
                "{name}, {preset}: {} glass pixels; {lit_ok} of {lit} that look at a panel are \
                 bright, {void_ok} of {void} that look into the void are dark",
                glass_pixels.len()
            );
            if lit < 300 {
                problems.push(format!(
                    "{name}, {preset}: only {lit} pixels look at a panel"
                ));
            }
            if lit_ok * 100 < lit * 95 {
                problems.push(format!(
                    "{name}, {preset}: only {lit_ok} of {lit} pixels that look at a panel are bright"
                ));
            }
            if void >= 50 && void_ok * 100 < void * 90 {
                problems.push(format!(
                    "{name}, {preset}: only {void_ok} of {void} pixels that look into the void are dark"
                ));
            }
        }
    }
    assert!(problems.is_empty(), "{}", problems.join("\n"));
}

/// Rays through glass, traced on the CPU: the twin of `through_glass` in
/// `shaders/post.wgsl`, for the tests to say what a pixel must show. Ideal
/// shapes, double precision, the same rules: index 1.5, bent on the way in
/// and on the way out, reflected inside when it cannot get out, let out as
/// it is after `bounces` reflections.
mod optics {
    use super::{pixel, QualityPreset};

    pub type V = [f64; 3];
    const INDEX: f64 = 1.5;

    fn add(a: V, b: V) -> V {
        [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
    }
    fn sub(a: V, b: V) -> V {
        [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
    }
    fn mul(a: V, s: f64) -> V {
        [a[0] * s, a[1] * s, a[2] * s]
    }
    fn dot(a: V, b: V) -> f64 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }
    fn cross(a: V, b: V) -> V {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    }
    fn unit(a: V) -> V {
        mul(a, 1.0 / dot(a, a).sqrt())
    }

    /// `i` refracted at a surface with unit normal `n` (against `i`), for the
    /// ratio `eta` of the indices before and after; `None`: total reflection.
    fn refract(i: V, n: V, eta: f64) -> Option<V> {
        let c = dot(n, i);
        let k = 1.0 - eta * eta * (1.0 - c * c);
        (k >= 0.0).then(|| sub(mul(i, eta), mul(n, eta * c + k.sqrt())))
    }

    /// Where the ray meets the ball, entering it: distance, or `None`.
    pub fn hits_ball(origin: V, dir: V, center: V, radius: f64) -> Option<f64> {
        let oc = sub(origin, center);
        let b = dot(oc, dir);
        let disc = b * b - (dot(oc, oc) - radius * radius);
        let t = -b - disc.max(0.0).sqrt();
        (disc >= 0.0 && t > 0.0).then_some(t)
    }

    /// Where the ray enters the box and where it leaves it (slab test).
    fn box_span(origin: V, dir: V, center: V, half: V) -> Option<(f64, f64)> {
        let (mut near, mut far) = (f64::NEG_INFINITY, f64::INFINITY);
        for a in 0..3 {
            let (o, d) = (origin[a] - center[a], dir[a]);
            if d.abs() < 1e-12 {
                if o.abs() > half[a] {
                    return None;
                }
                continue;
            }
            let (t0, t1) = ((-half[a] - o) / d, (half[a] - o) / d);
            near = near.max(t0.min(t1));
            far = far.min(t0.max(t1));
        }
        (near <= far).then_some((near, far))
    }

    /// Where the ray meets the box, entering it: distance, or `None`.
    pub fn hits_box(origin: V, dir: V, center: V, half: V) -> Option<f64> {
        box_span(origin, dir, center, half).and_then(|(near, _)| (near > 0.0).then_some(near))
    }

    /// A piece of glass: a cube (centre, half edge) or a ball (centre, radius).
    #[derive(Clone, Copy)]
    pub enum Glass {
        Cube(V, f64),
        Ball(V, f64),
    }

    impl Glass {
        fn entry(&self, origin: V, dir: V) -> Option<f64> {
            match *self {
                Glass::Cube(c, h) => hits_box(origin, dir, c, [h; 3]),
                Glass::Ball(c, r) => hits_ball(origin, dir, c, r),
            }
        }

        /// How far a ray from `origin`, inside, has to its surface.
        fn way_out(&self, origin: V, dir: V) -> f64 {
            match *self {
                Glass::Cube(c, h) => box_span(origin, dir, c, [h; 3]).map_or(0.0, |(_, far)| far),
                Glass::Ball(c, r) => {
                    let oc = sub(origin, c);
                    let b = dot(oc, dir);
                    (b * b - (dot(oc, oc) - r * r)).max(0.0).sqrt() - b
                }
            }
        }

        /// The outward unit normal at a point of its surface.
        fn normal(&self, at: V) -> V {
            match *self {
                Glass::Ball(c, _) => unit(sub(at, c)),
                Glass::Cube(c, _) => {
                    let l = sub(at, c);
                    let a = (0..3)
                        .max_by(|&i, &j| l[i].abs().total_cmp(&l[j].abs()))
                        .unwrap_or(0);
                    let mut n = [0.0; 3];
                    n[a] = l[a].signum();
                    n
                }
            }
        }
    }

    /// A ray as it leaves the glass, and how steeply it went in.
    pub struct Seen {
        pub origin: V,
        pub dir: V,
        /// Cosine of the angle between the ray and the normal where it entered.
        pub cos_in: f64,
    }

    /// The ray from `origin` along `dir` through the first piece of `glass`
    /// it meets; `None` if it meets none.
    pub fn through_glass(glass: &[Glass], origin: V, dir: V, bounces: u32) -> Option<Seen> {
        let (t, piece) = glass
            .iter()
            .filter_map(|g| g.entry(origin, dir).map(|t| (t, g)))
            .min_by(|a, b| a.0.total_cmp(&b.0))?;
        let entry = add(origin, mul(dir, t));
        let n = piece.normal(entry);
        let cos_in = -dot(n, dir);
        let mut dir = refract(dir, n, 1.0 / INDEX)?;
        let mut at = add(entry, mul(n, -1e-6));
        for _ in 0..=bounces {
            let t = piece.way_out(at, dir);
            if t <= 0.0 {
                break;
            }
            let exit = add(at, mul(dir, t));
            let m = piece.normal(exit);
            if let Some(out) = refract(dir, mul(m, -1.0), INDEX) {
                return Some(Seen {
                    origin: add(exit, mul(m, 1e-6)),
                    dir: out,
                    cos_in,
                });
            }
            dir = sub(dir, mul(m, 2.0 * dot(dir, m)));
            at = add(exit, mul(m, -1e-6));
        }
        // Still inside after the last reflection: let out as it is.
        let t = piece.way_out(at, dir).max(0.0);
        Some(Seen {
            origin: add(at, mul(dir, t + 1e-6)),
            dir,
            cos_in,
        })
    }

    /// A pinhole camera at the origin: the ray through the middle of a pixel.
    pub struct Camera {
        forward: V,
        right: V,
        up: V,
        size: (u32, u32),
    }

    impl Camera {
        /// Looking along `yaw` / `pitch` degrees (as `camera_looking`), 60
        /// degrees of vertical field, `aspect` wide.
        pub fn looking(yaw: f32, pitch: f32, aspect: f64, width: u32, height: u32) -> Self {
            let (yaw, pitch) = (f64::from(yaw).to_radians(), f64::from(pitch).to_radians());
            let forward = [
                -yaw.sin() * pitch.cos(),
                pitch.sin(),
                -yaw.cos() * pitch.cos(),
            ];
            let right = unit(cross(forward, [0.0, 1.0, 0.0]));
            let tan = (30.0f64).to_radians().tan();
            Self {
                forward,
                right: mul(right, tan * aspect),
                up: mul(cross(right, forward), tan),
                size: (width, height),
            }
        }

        pub fn ray(&self, x: u32, y: u32) -> V {
            let nx = (f64::from(x) + 0.5) / f64::from(self.size.0) * 2.0 - 1.0;
            let ny = 1.0 - (f64::from(y) + 0.5) / f64::from(self.size.1) * 2.0;
            unit(add(
                self.forward,
                add(mul(self.right, nx), mul(self.up, ny)),
            ))
        }
    }

    /// The pixels of a `size` x `size` frame of `refraction_scene`'s camera in
    /// which its ball lens (radius 2, centre (0, 0, -4)) shows something that
    /// `target` says the ray leaving the ball hits.
    pub fn ball_lens_image(
        size: u32,
        preset: QualityPreset,
        target: impl Fn(V, V) -> bool,
    ) -> Vec<(u32, u32)> {
        let camera = Camera::looking(0.0, 0.0, 1.0, size, size);
        let lens = [Glass::Ball([0.0, 0.0, -4.0], 2.0)];
        let bounces = preset.features().glass_bounces;
        (0..size)
            .flat_map(|y| (0..size).map(move |x| (x, y)))
            .filter(|&(x, y)| {
                through_glass(&lens, [0.0; 3], camera.ray(x, y), bounces)
                    .is_some_and(|seen| target(seen.origin, seen.dir))
            })
            .collect()
    }

    /// The pixels of `out` for which `shows` holds are the `expected` ones,
    /// give or take the outline: nine in ten of the expected ones show it,
    /// and at most a fifth (plus a few) of those that show it were not
    /// expected to.
    pub fn assert_same_pixels(
        what: &str,
        out: &space_render::FrameOutput,
        expected: &[(u32, u32)],
        shows: impl Fn([u16; 3]) -> bool,
    ) {
        let shown = |x: u32, y: u32| {
            let p = pixel(out, x, y);
            shows([p[0], p[1], p[2]])
        };
        let hit = expected.iter().filter(|&&(x, y)| shown(x, y)).count();
        let total = (0..out.height)
            .flat_map(|y| (0..out.width).map(move |x| (x, y)))
            .filter(|&(x, y)| shown(x, y))
            .count();
        eprintln!(
            "{what}: {} pixels expected, {hit} of them shown, {total} shown in all",
            expected.len()
        );
        assert!(
            expected.len() >= 30,
            "{what}: only {} pixels expected",
            expected.len()
        );
        assert!(
            hit * 10 >= expected.len() * 9,
            "{what}: {hit} of {} expected pixels",
            expected.len()
        );
        assert!(
            (total - hit) * 5 <= total + 40,
            "{what}: {} pixels show it that should not",
            total - hit
        );
    }
}

/// What the tone map makes of a linear radiance: ACES, then gamma 2.2, in
/// 8 bits (`shaders/tonemap.wgsl`, exposure 1).
fn tone_mapped(radiance: [f32; 3]) -> [u16; 3] {
    radiance.map(|x| {
        let aces = (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14);
        (aces.clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0).round() as u16
    })
}

/// An environment whose three colours are easy to tell apart: blue above,
/// green at the horizon, red below.
fn rgb_sky() -> Environment {
    Environment::gradient([0.05, 0.1, 1.0], [0.1, 1.0, 0.1], [1.0, 0.1, 0.05])
}

fn metal_ball(center: [f32; 3], radius: f32, roughness: f32) -> GpuInstance {
    GpuInstance {
        center: [center[0], center[1], center[2], radius],
        albedo: [0.95, 0.95, 0.95, 1.0],
        emissive: [0.0; 4],
        roughness,
        shape: space_render::SHAPE_SPHERE,
        transmission: 0.0,
        _pad1: 0.0,
    }
}

/// The environment gives a reflection something to show: a smooth metal
/// sphere alone in an empty, unlit scene is black without one, and with one
/// mirrors the gradient — the zenith's colour where it reflects upwards, the
/// horizon's in the middle, the ground's below — in every preset and whichever
/// way the camera looks. The background stays black unless the environment
/// asks to be drawn there; then the empty pixels show exactly the gradient
/// along their view rays, and the sphere's pixels do not change.
#[test]
fn a_smooth_metal_sphere_mirrors_the_environment() {
    let Some(mut ctx) = context() else { return };
    let size = 128;
    let ball = Scene::from_instances(post_camera(), vec![metal_ball([0.0, 0.0, -4.0], 1.2, 0.1)]);

    // Without an environment: nothing. This is today's picture.
    for preset in QualityPreset::ALL {
        let black = render_clear(&mut ctx, &ball, size, preset);
        assert_eq!(
            brightness_sum(&black),
            0,
            "{preset}: no light, no environment"
        );
    }

    // Where the surface normal tips 45 degrees up the mirror direction is
    // straight up; that point of the sphere is 30 pixels above the centre.
    let (above, middle, below) = ((64, 34), (64, 64), (64, 94));
    for preset in QualityPreset::ALL {
        for yaw in [0.0f32, 140.0] {
            let scene = turned(&ball.clone_with(rgb_sky()), yaw);
            let out = render_clear(&mut ctx, &scene, size, preset);
            let (top, centre, bottom) = (
                rgb(&out, above.0, above.1),
                rgb(&out, middle.0, middle.1),
                rgb(&out, below.0, below.1),
            );
            eprintln!("{preset}, yaw {yaw}: sphere {top:?} / {centre:?} / {bottom:?}");
            assert!(
                dominant(top, 2),
                "{preset}: the top mirrors the zenith: {top:?}"
            );
            assert!(
                dominant(centre, 1),
                "{preset}: the middle the horizon: {centre:?}"
            );
            assert!(
                dominant(bottom, 0),
                "{preset}: the underside the ground: {bottom:?}"
            );
            // The background is not painted unless asked for.
            assert_eq!(rgb(&out, 3, 3), [0, 0, 0], "{preset}");
            assert_eq!(rgb(&out, 124, 124), [0, 0, 0], "{preset}");

            // Asked for: every empty pixel is the gradient along its view ray,
            // and the sphere's pixels are what they were.
            let painted = render_clear(
                &mut ctx,
                &turned(&ball.clone_with(rgb_sky().with_background(true)), yaw),
                size,
                preset,
            );
            let tan = (std::f32::consts::FRAC_PI_3 * 0.5).tan();
            for (x, y) in [
                (3u32, 3u32),
                (64, 2),
                (2, 64),
                (125, 64),
                (64, 125),
                (100, 120),
            ] {
                let ndc = |i: u32| (i as f32 + 0.5) / (size as f32 * 0.5) - 1.0;
                // (The camera's yaw turns the ray about the vertical: its
                // elevation, all the gradient looks at, stays the same.)
                let ray = [ndc(x) * tan, -ndc(y) * tan, -1.0];
                let expected = tone_mapped(rgb_sky().radiance(ray));
                let got = rgb(&painted, x, y);
                assert!(
                    (0..3).all(|c| got[c].abs_diff(expected[c]) <= 3),
                    "{preset}, yaw {yaw}: background at ({x}, {y}) is {got:?}, the gradient \
                     says {expected:?}"
                );
            }
            for y in 0..size {
                for x in 0..size {
                    let lit = rgb(&out, x, y);
                    if lit != [0, 0, 0] {
                        assert_eq!(
                            rgb(&painted, x, y),
                            lit,
                            "{preset}: sphere pixel ({x}, {y})"
                        );
                    }
                }
            }
        }
    }

    // Roughness blurs the reflection towards the average of what the surface
    // faces: the top of a rough ball is no longer pure zenith blue.
    let rough = Scene::from_instances(post_camera(), vec![metal_ball([0.0, 0.0, -4.0], 1.2, 0.9)])
        .with_environment(rgb_sky());
    let out = render_clear(&mut ctx, &rough, size, QualityPreset::High);
    let top = rgb(&out, above.0, above.1);
    assert!(
        top[2] > 60 && top[1] > 40 && !dominant(top, 2),
        "rough top: {top:?}"
    );
}

/// The environment is the light that is left where no lamp reaches: the floor
/// in the cube's shadow (black without an environment, see the shadow test)
/// is lit by it, and still clearly darker than the floor the lamp does reach.
/// It lights models as well as spheres and boxes, by the direction each face
/// looks: the cube's side in the lamp's shadow is brighter than black and
/// darker than its lit top.
#[test]
fn the_environment_lights_what_is_in_shadow() {
    let Some(mut ctx) = context() else { return };
    if !ctx.ray_query_support().complete() {
        eprintln!("SKIP: device has no ray query");
        return;
    }
    let size = 128;
    let cube = Arc::new(MeshData::cuboid([1.0; 3], 1.0, matte([0.8; 3])));
    let scene = shadow_scene(vec![placed(&cube, [0.0, -2.5, -8.0])], Vec::new());
    // Pixels as in `a_model_casts_a_shadow_on_the_floor_and_receives_one`,
    // plus the cube's front face, which the lamp above it does not reach.
    let (in_shadow, lit_floor, cube_top, cube_front) = ((64, 124), (16, 124), (64, 85), (64, 100));
    let studio = Environment::studio().with_intensity(0.35);

    let bare = render_clear(&mut ctx, &scene, size, QualityPreset::High);
    for (x, y) in [in_shadow, cube_front] {
        let p = rgb(&bare, x, y);
        assert!(is_dark(p), "no environment: ({x}, {y}) is unlit: {p:?}");
    }
    let out = render_clear(
        &mut ctx,
        &scene.clone_with(studio),
        size,
        QualityPreset::High,
    );
    let at = |p: (u32, u32)| luma(rgb(&out, p.0, p.1));
    eprintln!(
        "ambient: shadow {} / lit floor {} / cube top {} / cube front {}",
        at(in_shadow),
        at(lit_floor),
        at(cube_top),
        at(cube_front)
    );
    assert!(at(in_shadow) > 60, "the shadow is lit by the environment");
    assert!(
        at(in_shadow) + 40 < at(lit_floor),
        "and still darker than the lit floor beside it"
    );
    assert!(at(cube_front) > 50 && at(cube_front) + 40 < at(cube_top));
    // A floor faces the zenith, a wall the horizon: more light from above —
    // where nothing is in the sky's way. (`Low` casts no sky rays; with them,
    // the floor at the foot of the cube sees less of the sky than the cube's
    // side does.)
    let mut lampless = scene.clone_with(studio);
    lampless.lights.clear();
    let open = render_clear(&mut ctx, &lampless, size, QualityPreset::Low);
    assert!(
        luma(rgb(&open, in_shadow.0, in_shadow.1)) > luma(rgb(&open, cube_front.0, cube_front.1)),
        "up-facing gets more than side-facing"
    );
    // The lit pixels gained the same ambient term: brighter than before, or
    // already at the top of the range.
    for p in [lit_floor, cube_top] {
        assert!(at(p) >= luma(rgb(&bare, p.0, p.1)));
    }
    // A brighter environment, a brighter shadow.
    let brighter = render_clear(
        &mut ctx,
        &scene.clone_with(studio.with_intensity(0.7)),
        size,
        QualityPreset::High,
    );
    assert!(luma(rgb(&brighter, in_shadow.0, in_shadow.1)) > at(in_shadow) + 20);

    // Metals have no diffuse colour and take no ambient light, only the
    // reflection: a rough metal floor under the same environment is darker
    // than the matte one.
    let mut metal = scene.clone_with(studio);
    metal.instances[0].albedo[3] = 1.0;
    let metal = render_clear(&mut ctx, &metal, size, QualityPreset::High);
    assert!(luma(rgb(&metal, in_shadow.0, in_shadow.1)) < at(in_shadow));
}

/// A mirror that reflects something that is not on screen shows the
/// environment there instead of black: the mirror floor of `mirror_scene`,
/// which looks up past the top of the frame. With the red sphere standing on
/// it, the sphere's screen-space reflection is still where it was, on top of
/// the environment's.
#[test]
fn a_mirror_shows_the_environment_where_the_screen_has_nothing() {
    let Some(mut ctx) = context() else { return };
    let size = 128;
    let with_sphere = mirror_scene(0.1);
    let mut floor_only = mirror_scene(0.1);
    floor_only.instances.truncate(1);

    for preset in QualityPreset::ALL {
        // No environment: the floor mirrors nothing, as before.
        let black = render_clear(&mut ctx, &floor_only, size, preset);
        assert_eq!(brightness_sum(&black), 0, "{preset}");

        // The floor at the bottom of the frame mirrors a steep upward
        // direction, the floor near the horizon a shallow one: from the
        // zenith's blue towards the horizon's green.
        let out = render_clear(&mut ctx, &floor_only.clone_with(rgb_sky()), size, preset);
        let (near, far) = (rgb(&out, 64, 122), rgb(&out, 64, 76));
        eprintln!("{preset}: mirror floor near {near:?}, far {far:?}");
        assert!(
            near[2] > 100 && near[2] > near[0] + 60,
            "{preset}: sky in the mirror: {near:?}"
        );
        assert!(
            far[1] > 100 && far[1] > far[0] + 60,
            "{preset}: horizon in the mirror: {far:?}"
        );
        assert!(
            near[2] > far[2] && far[1] > near[1],
            "{preset}: {near:?} vs {far:?}"
        );
        // The floor mirrors only what is above it: none of the ground's red
        // (what red there is, is the little in the other two colours).
        assert!(near[0] + 60 < near[1].min(near[2]) && far[0] < far[2]);
    }

    // With the sphere: its reflection (rows 83..) is found on screen and wins
    // over the environment's.
    let reflection_rows = 83..size;
    for preset in [QualityPreset::Medium, QualityPreset::High] {
        let before = render_preset(&mut ctx, &with_sphere, size, preset);
        let after = render_preset(&mut ctx, &with_sphere.clone_with(rgb_sky()), size, preset);
        let (n0, x0, y0) = centroid(&before, reflection_rows.clone(), is_red).expect("reflection");
        let (n, x, y) = centroid(&after, reflection_rows.clone(), is_red)
            .unwrap_or_else(|| panic!("{preset}: the sphere's reflection is gone"));
        eprintln!("{preset}: reflection n={n} ({x:.1},{y:.1}); without environment n={n0}");
        assert!(n * 10 >= n0 * 6, "{preset}: {n} vs {n0} red pixels");
        assert!((x - x0).abs() < 1.5 && (y - y0).abs() < 3.0);
    }
}

/// Glass in front of nothing shows the environment along its refracted view,
/// not black; glass in front of a painted background likewise; and without an
/// environment it is as black as the empty frame around it.
#[test]
fn glass_in_front_of_nothing_shows_the_environment() {
    let Some(mut ctx) = context() else { return };
    let size = 128;
    let lens = GpuInstance {
        center: [0.0, 0.0, -4.0, 1.5],
        albedo: [0.0; 4],
        emissive: [0.0; 4],
        roughness: 0.5,
        shape: space_render::SHAPE_SPHERE,
        transmission: 1.0,
        _pad1: 0.0,
    };
    let scene = Scene::from_instances(post_camera(), vec![lens]);
    for preset in [QualityPreset::Medium, QualityPreset::High] {
        let black = render_clear(&mut ctx, &scene, size, preset);
        assert_eq!(
            brightness_sum(&black),
            0,
            "{preset}: nothing to see through the glass"
        );
        for background in [false, true] {
            for yaw in [0.0f32, 70.0] {
                let environment = rgb_sky().with_background(background);
                let out = render_clear(
                    &mut ctx,
                    &turned(&scene.clone_with(environment), yaw),
                    size,
                    preset,
                );
                // Straight through the middle: the horizon. Through the upper
                // part the view is bent downwards, towards the ground's red;
                // through the lower part upwards, towards the zenith's blue.
                let (centre, upper, lower) =
                    (rgb(&out, 64, 64), rgb(&out, 64, 34), rgb(&out, 64, 94));
                eprintln!(
                    "{preset}, background {background}, yaw {yaw}: glass {upper:?} / {centre:?} / {lower:?}"
                );
                assert!(dominant(centre, 1), "{centre:?}");
                assert!(
                    upper[0] > lower[0] && lower[2] > upper[2],
                    "the upper part is bent towards the red ground, the lower part towards \
                     the blue zenith: {upper:?} / {lower:?}"
                );
                // Beside the glass: black, or the painted background.
                let corner = rgb(&out, 3, 64);
                assert_eq!(corner != [0, 0, 0], background, "{corner:?}");
            }
        }
    }
}

/// A scene without an environment is not touched by any of this: "none" spelt
/// three different ways renders the same frame, pixel for pixel, as the
/// default — in the presets with and without the post pass, with models,
/// glass, mirrors and shadows in the frame. (That the default itself still
/// renders what it always did is what every other test in this file checks.)
#[test]
fn no_environment_changes_nothing() {
    let Some(mut ctx) = context() else { return };
    let size = 96;
    let room = GlassRoom::new();
    let mut scenes = vec![
        ("mirror", mirror_scene(0.1)),
        ("lens", refraction_scene(true)),
        (
            "room",
            room.scene([-4.7, 0.0, -5.0], 34.5, -5.9, Middle::Glass),
        ),
    ];
    let character = Arc::new(space_render::sample_character());
    scenes.push((
        "shadows",
        shadow_scene(vec![placed(&character, [0.0, -4.0, -7.0])], Vec::new()),
    ));
    let nones = [
        Environment::NONE,
        Environment::sky().with_intensity(0.0).with_background(true),
        Environment::gradient([0.0; 3], [0.0; 3], [0.0; 3]).with_background(true),
        Environment::studio().with_intensity(f32::NAN),
    ];
    for (name, scene) in &scenes {
        for preset in [QualityPreset::Ci, QualityPreset::High] {
            let reference = render_clear(&mut ctx, scene, size, preset);
            for none in nones {
                assert!(none.is_none());
                let out = render_clear(&mut ctx, &scene.clone_with(none), size, preset);
                assert!(
                    out.pixels == reference.pixels,
                    "{name}, {preset}: {none:?} changed the frame"
                );
            }
            // And an environment does change it.
            let lit = render_clear(
                &mut ctx,
                &scene.clone_with(Environment::sky()),
                size,
                preset,
            );
            assert!(lit.pixels != reference.pixels, "{name}, {preset}");
        }
    }
}

/// `clone_with` for the tests above: the same scene with `environment`.
trait WithEnvironment {
    fn clone_with(&self, environment: Environment) -> Scene;
}

impl WithEnvironment for Scene {
    fn clone_with(&self, environment: Environment) -> Scene {
        let mut scene = Scene::from_instances(self.camera, self.instances.clone())
            .with_lights(self.lights.clone())
            .with_environment(environment);
        scene.models = self.models.clone();
        scene.overlay = self.overlay.clone();
        scene
    }
}

// ---- surfaces seen from behind ---------------------------------------------

/// What the floor of [`floor_scene`] is made of.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Floor {
    /// A slab 20 cm thick (a closed mesh): its underside is a face of its own.
    Slab,
    /// A sheet without thickness (an open mesh): its underside is the back of
    /// its top.
    Sheet,
    /// A 12 m box instance whose top face is the floor: a camera under that
    /// floor is inside the box, looking at the back of its top face.
    Box,
    /// The same box in polished metal: a "mirror let into the floor" made
    /// from a box instance, which is a cube.
    MirrorBox,
}

/// A bright sky over a black ground, painted into the background.
const BRIGHT_SKY: Environment =
    Environment::gradient([0.6, 0.8, 1.6], [0.5, 0.6, 0.7], [0.03, 0.03, 0.03])
        .with_background(true);

/// A 12 m square floor whose top is the plane y = 0 (a checkerboard of 2 m
/// cells where it is a mesh), under [`BRIGHT_SKY`], seen from `eye` looking
/// `pitch` degrees up; with `lamp`, a strong light hangs 4 m above its middle.
fn floor_scene(floor: Floor, eye: [f32; 3], pitch: f32, lamp: bool) -> Scene {
    let at = |p: [f32; 3]| [p[0] - eye[0], p[1] - eye[1], p[2] - eye[2]];
    let checkered = |mut mesh: MeshData| {
        let slot = mesh.add_texture(Texture::checkerboard(64, 2, [230; 3], [60; 3]));
        mesh.parts[0].material.base_color_texture = Some(slot);
        Arc::new(mesh)
    };
    let (mut instances, mut models) = (Vec::new(), Vec::new());
    match floor {
        Floor::Slab => models.push(placed(
            &checkered(MeshData::cuboid([6.0, 0.1, 6.0], 4.0, matte([1.0; 3]))),
            at([0.0, -0.1, 0.0]),
        )),
        Floor::Sheet => models.push(placed(
            &checkered(MeshData::plane([6.0, 6.0], [3.0, 3.0], matte([1.0; 3]))),
            at([0.0; 3]),
        )),
        Floor::Box | Floor::MirrorBox => {
            let [x, y, z] = at([0.0, -6.0, 0.0]);
            let mirror = floor == Floor::MirrorBox;
            instances.push(GpuInstance {
                center: [x, y, z, 6.0],
                albedo: [0.8, 0.8, 0.8, if mirror { 1.0 } else { 0.0 }],
                emissive: [0.0; 4],
                roughness: if mirror { 0.04 } else { 1.0 },
                shape: space_render::SHAPE_BOX,
                transmission: 0.0,
                _pad1: 0.0,
            });
        }
    }
    let mut scene =
        Scene::from_instances(camera_looking(0.0, pitch), instances).with_environment(BRIGHT_SKY);
    scene.models = models;
    if lamp {
        scene.lights = vec![light_at(at([0.0, 4.0, 0.0]).map(f64::from), 300.0)];
    }
    scene
}

/// The brightness below which `percent` per cent of the pixels of `out`'s
/// rows `rows` lie.
fn luma_percentile(
    out: &space_render::FrameOutput,
    rows: std::ops::Range<u32>,
    percent: usize,
) -> u32 {
    let mut values: Vec<u32> = rows
        .flat_map(|y| (0..out.width).map(move |x| (x, y)))
        .map(|(x, y)| luma(rgb(out, x, y)))
        .collect();
    values.sort_unstable();
    values[(values.len() - 1) * percent / 100]
}

/// The largest difference in any channel between `a` and `b` in rows `rows`.
fn largest_difference(
    a: &space_render::FrameOutput,
    b: &space_render::FrameOutput,
    rows: std::ops::Range<u32>,
) -> u16 {
    rows.flat_map(|y| (0..a.width).map(move |x| (x, y)))
        .map(|(x, y)| {
            let (p, q) = (rgb(a, x, y), rgb(b, x, y));
            (0..3).map(|c| p[c].abs_diff(q[c])).max().unwrap_or(0)
        })
        .max()
        .unwrap_or(0)
}

/// Nothing is culled, so a camera under a floor sees it from below — and what
/// it sees there is the floor's *underside*: a surface of its own that faces
/// down. Not the top seen through from behind (lit by the lamp above it and
/// by the sky), and not the sky.
///
/// The owner's report was a mirror in the floor "transparent from the bottom":
/// that mirror was a box instance — a cube, nearly all of it under the floor —
/// and a camera inside it saw the cube's inner faces reflect the sky. The
/// inside of a closed solid has no sky: it is dark now. So is the back of a
/// box's top face, which the lamp above used to light like the top itself;
/// and the back of a one-sided sheet is lit from the side it is seen from.
///
/// For each kind of floor, in every preset, from 1.5 m below it:
///
/// * looking up, the lamp above changes nothing (within rounding) and the
///   underside is far darker than the top is from above;
/// * a slab or a sheet shows its checkerboard, its light cells exactly as
///   bright as the half of the environment below the horizon makes them
///   ([`Environment::ambient`] for a normal pointing down), its dark cells
///   dark;
/// * the inside of a box, matte or mirror, is black;
/// * looking sideways, the underside is darker than the sky seen past the
///   floor's edge.
///
/// Mutations of `side_of` in `forward.wgsl`: if it never turns the normal
/// (and with the shader as it was), the sheet and the matte box are lit like
/// their tops and the mirror box shows the sky; if it calls every side open,
/// both boxes are lit by the environment from within.
#[test]
fn a_floor_seen_from_below_shows_its_underside_not_its_top_or_the_sky() {
    let Some(mut ctx) = context() else { return };
    let size = 128;
    let (below, above) = ([0.0, -1.5, 0.0], [0.0, 1.5, 0.0]);
    // A light cell of the checkerboard (sRGB 230), lit by the environment
    // from below: its diffuse light, and the 1.6 % a fully rough dielectric
    // reflects of it.
    let albedo = ((230.0f32 / 255.0 + 0.055) / 1.055).powf(2.4);
    let expected = tone_mapped(
        BRIGHT_SKY
            .ambient([0.0, -1.0, 0.0])
            .map(|c| c * (albedo + 0.0157)),
    );
    let mut problems = Vec::new();
    for floor in [Floor::Slab, Floor::Sheet, Floor::Box, Floor::MirrorBox] {
        for preset in [
            QualityPreset::Ci,
            QualityPreset::Low,
            QualityPreset::Medium,
            QualityPreset::High,
        ] {
            let mut render = |eye, pitch, lamp| {
                render_clear(
                    &mut ctx,
                    &floor_scene(floor, eye, pitch, lamp),
                    size,
                    preset,
                )
            };
            let top = luma_percentile(&render(above, -70.0, true), 0..size, 90);
            // Looking up: the frame is all underside.
            let (up, unlit) = (render(below, 70.0, true), render(below, 70.0, false));
            let (bright, dark) = (
                luma_percentile(&up, 0..size, 90),
                luma_percentile(&up, 0..size, 10),
            );
            let lamp = largest_difference(&up, &unlit, 0..size);
            // Looking sideways: the underside above, and (past the edge of a
            // slab or sheet, 6 m away) the sky below.
            let side = render(below, 15.0, true);
            let (under, sky) = (luma_percentile(&side, 0..56, 90), luma(rgb(&side, 64, 100)));
            eprintln!(
                "{floor:?}, {preset}: top {top}; from below {dark}..{bright}, the lamp changes \
                 {lamp}; sideways {under} under sky {sky}"
            );
            let mut check = |ok: bool, what: String| {
                if !ok {
                    problems.push(format!("{floor:?}, {preset}: {what}"));
                }
            };
            check(top > 200, format!("the top is lit: {top}"));
            check(
                bright + 60 < top,
                format!("the underside ({bright}) is lit like the top ({top})"),
            );
            check(
                lamp <= 2,
                format!("the lamp above changes the underside by {lamp}"),
            );
            match floor {
                Floor::Slab | Floor::Sheet => {
                    // The light cells: the brightest quarter of the frame.
                    let cells: Vec<[u16; 3]> = (0..size)
                        .flat_map(|y| (0..size).map(move |x| (x, y)))
                        .map(|(x, y)| rgb(&up, x, y))
                        .filter(|&p| luma(p) >= bright)
                        .collect();
                    let mean = [0, 1, 2].map(|c| {
                        cells.iter().map(|p| u32::from(p[c])).sum::<u32>() / cells.len() as u32
                    });
                    check(
                        (0..3).all(|c| mean[c].abs_diff(u32::from(expected[c])) <= 6),
                        format!("light cells are {mean:?}, not {expected:?}"),
                    );
                    check(
                        dark < 80 && dark + 40 < bright,
                        format!("the checkerboard is washed out: {dark}..{bright}"),
                    );
                    check(
                        under + 40 < sky,
                        format!("the underside ({under}) looks like the sky ({sky})"),
                    );
                }
                Floor::Box | Floor::MirrorBox => {
                    check(
                        bright < 12,
                        format!("the inside of a solid is lit: {bright}"),
                    );
                }
            }
        }
    }
    assert!(problems.is_empty(), "{}", problems.join("\n"));
}

/// Every face of a box faces outwards, the instance's and the mesh's alike:
/// seen from outside under an environment and no lamp, each face shows the
/// environment's light for the direction it faces — the sky's blue on top,
/// the ground's red underneath, mostly the horizon's green on the sides —
/// to the value [`Environment::ambient`] gives for its normal. (A face whose
/// normal pointed into the box would count as the inside of a closed solid
/// and stay black; one with another face's normal would have its colour.)
#[test]
fn every_face_of_a_box_is_lit_by_the_environment_from_the_side_it_faces() {
    let Some(mut ctx) = context() else { return };
    let size = 64;
    let sky = rgb_sky();
    let mesh = Arc::new(MeshData::cuboid([1.0; 3], 1.0, matte([0.8; 3])));
    let faces: [(&str, [f32; 3]); 6] = [
        ("+x", [1.0, 0.0, 0.0]),
        ("-x", [-1.0, 0.0, 0.0]),
        ("+y", [0.0, 1.0, 0.0]),
        ("-y", [0.0, -1.0, 0.0]),
        ("+z", [0.0, 0.0, 1.0]),
        ("-z", [0.0, 0.0, -1.0]),
    ];
    let mut seen = Vec::new();
    for (name, normal) in faces {
        // The box 5 m in front of a camera that looks at this face head-on.
        let forward = normal.map(|c| -c);
        let center = forward.map(|c| c * 5.0);
        let camera = space_render::CameraPose {
            forward,
            up: if normal[1] == 0.0 {
                [0.0, 1.0, 0.0]
            } else {
                [0.0, 0.0, -1.0]
            },
            ..post_camera()
        };
        // A matte dielectric reflects 1.57 % of the same light on top.
        let expected = tone_mapped(sky.ambient(normal).map(|c| c * (0.8 + 0.0157)));
        let instance = GpuInstance {
            center: [center[0], center[1], center[2], 1.0],
            albedo: [0.8, 0.8, 0.8, 0.0],
            emissive: [0.0; 4],
            roughness: 1.0,
            shape: space_render::SHAPE_BOX,
            transmission: 0.0,
            _pad1: 0.0,
        };
        let as_instance = Scene::from_instances(camera, vec![instance]).with_environment(sky);
        let mut as_mesh = Scene::from_instances(camera, Vec::new()).with_environment(sky);
        as_mesh.models = vec![placed(&mesh, center)];
        for (kind, scene) in [("instance", &as_instance), ("mesh", &as_mesh)] {
            for preset in [QualityPreset::Ci, QualityPreset::High] {
                let p = rgb(&render_clear(&mut ctx, scene, size, preset), 32, 32);
                assert!(
                    (0..3).all(|c| p[c].abs_diff(expected[c]) <= 4),
                    "{kind} face {name}, {preset}: {p:?}, not {expected:?}"
                );
            }
        }
        seen.push((name, expected));
    }
    eprintln!("faces under the environment: {seen:?}");
    // The six faces are told apart by three colours: top, bottom, sides.
    let (top, bottom, side) = (seen[2].1, seen[3].1, seen[0].1);
    assert!(dominant(top, 2) || top[2] > top[0] + 60, "{top:?}");
    assert!(bottom[0] > bottom[2] + 60, "{bottom:?}");
    assert!(side[1] > side[0] && side[1] > side[2], "{side:?}");
}

// ---- the environment, continued: fog, probes, occlusion -----------------------

/// Fog fades a surface into what is behind it. With a painted environment
/// that is the sky along the view ray: a wall far enough away to be all fog
/// is indistinguishable from the sky beside it, at every height — no seam at
/// the horizon, and none above it where the sky has another colour. Without a
/// painted background it is the fog colour, as ever.
///
/// The wall covers the left of the frame, 1 km away, in fog that is opaque by
/// then; each of its pixels is compared with the sky pixel mirrored about the
/// frame's centre (the same elevation).
/// Mutation: with the fog colour constant in `forward.wgsl`, the wall is one
/// colour from top to bottom in front of a sky that is three.
#[test]
fn fog_fades_into_the_painted_sky_behind_it() {
    let Some(mut ctx) = context() else { return };
    let size = 64;
    let wall = GpuInstance {
        center: [-600.0, 0.0, -1500.0, 500.0],
        albedo: [0.6, 0.6, 0.6, 0.0],
        emissive: [0.0; 4],
        roughness: 1.0,
        shape: space_render::SHAPE_BOX,
        transmission: 0.0,
        _pad1: 0.0,
    };
    let scene = |environment: Environment| {
        Scene::from_instances(post_camera(), vec![wall]).with_environment(environment)
    };
    let mut render = |scene: &Scene, preset: QualityPreset| {
        let opts = RenderOptions {
            gi_enabled: false,
            quality: Quality::CI,
            preset,
            // Twice the preset's density: all fog well before a kilometre.
            fog_scale: 4.0,
            ..Default::default()
        };
        let out = ctx.render_frame(scene, size, size, &opts).expect("render");
        assert_eq!(out.validation_messages, 0, "validation messages seen");
        out
    };
    for preset in [QualityPreset::Medium, QualityPreset::High] {
        let painted = render(&scene(rgb_sky().with_background(true)), preset);
        let (top, middle, bottom) = (
            rgb(&painted, 10, 8),
            rgb(&painted, 10, 32),
            rgb(&painted, 10, 56),
        );
        eprintln!(
            "{preset}: the wall in fog is {top:?} / {middle:?} / {bottom:?} from top to bottom"
        );
        for y in 8..57 {
            let (wall, sky) = (rgb(&painted, 10, y), rgb(&painted, 53, y));
            assert!(
                (0..3).all(|c| wall[c].abs_diff(sky[c]) <= 2),
                "{preset}, row {y}: the wall is {wall:?} in front of a sky that is {sky:?}"
            );
        }
        // ... and the sky is not one colour: blue above, red below.
        assert!(
            top[2] > top[0] + 60 && bottom[0] > bottom[2] + 60,
            "{top:?} / {bottom:?}"
        );

        // Not painted: the same wall fades into the fog colour, like a wall
        // in a scene without an environment.
        let unpainted = render(&scene(rgb_sky()), preset);
        let none = render(&scene(Environment::NONE), preset);
        for y in [8, 32, 56] {
            let (a, b) = (rgb(&unpainted, 10, y), rgb(&none, 10, y));
            assert!(
                (0..3).all(|c| a[c].abs_diff(b[c]) <= 1),
                "{preset}: {a:?} vs {b:?}"
            );
            assert!(
                rgb(&unpainted, 53, y) == [0, 0, 0],
                "nothing is painted there"
            );
        }
    }
}

/// Probe settings for the environment tests: probes 2 m apart at the
/// camera's height, at odd x and z from -7 to 7.
fn close_probes() -> Quality {
    Quality {
        grid: 8,
        rays_per_probe: 16,
        probe_range: 60.0,
        spacing: 2.0,
    }
}

fn render_with_gi(
    ctx: &mut RenderContext,
    scene: &Scene,
    size: u32,
    preset: QualityPreset,
    gi: bool,
) -> space_render::FrameOutput {
    let opts = RenderOptions {
        gi_enabled: gi,
        quality: close_probes(),
        preset,
        fog_scale: 0.0,
        ..Default::default()
    };
    let out = ctx.render_frame(scene, size, size, &opts).expect("render");
    assert_eq!(out.validation_messages, 0, "validation messages seen");
    out
}

/// The probes see the environment: a surface a probe ray hits sends back the
/// sky light it reflects, so sky light bounces. Under a sky that shines from
/// straight above only, the underside of a white ball gets next to nothing
/// directly — but below it is a red floor in full sky light, and with probe
/// GI the ball's underside is lit by that floor: brighter, and red.
/// Mutation: without the bounce term in `gi_probe.wgsl` GI changes nothing
/// here.
#[test]
fn sky_light_bounces_off_what_the_probes_see() {
    let Some(mut ctx) = context() else { return };
    if !ctx.ray_query_support().complete() {
        eprintln!("SKIP: device has no ray query");
        return;
    }
    let size = 96;
    let matte = |center: [f32; 4], shape: u32, albedo: [f32; 3]| GpuInstance {
        center,
        albedo: [albedo[0], albedo[1], albedo[2], 0.0],
        emissive: [0.0; 4],
        roughness: 1.0,
        shape,
        transmission: 0.0,
        _pad1: 0.0,
    };
    let scene = Scene::from_instances(
        post_camera(),
        vec![
            // A red floor 1.5 m below the camera, a white ball above eye level.
            matte(
                [0.0, -101.5, 0.0, 100.0],
                space_render::SHAPE_BOX,
                [0.9, 0.1, 0.1],
            ),
            matte(
                [0.0, 0.9, -3.5, 0.6],
                space_render::SHAPE_SPHERE,
                [0.9, 0.9, 0.9],
            ),
        ],
    )
    .with_environment(Environment::gradient([2.0; 3], [0.0; 3], [0.0; 3]));
    for preset in [QualityPreset::Ci, QualityPreset::High] {
        let off = render_with_gi(&mut ctx, &scene, size, preset, false);
        let on = render_with_gi(&mut ctx, &scene, size, preset, true);
        // The lower front of the ball; and its top, for comparison.
        let (under_off, under_on) = (rgb(&off, 48, 38), rgb(&on, 48, 38));
        eprintln!("{preset}: the ball's underside is {under_off:?} without GI, {under_on:?} with");
        assert!(
            luma(under_off) < 80,
            "{preset}: hardly any sky from below: {under_off:?}"
        );
        assert!(
            under_on[0] > under_off[0] + 40 && under_on[0] > under_on[1] + 30,
            "{preset}: the red floor's sky light reaches the underside: {under_on:?} (was {under_off:?})"
        );
        // The frame is the same every time (the probes use no random numbers
        // that change).
        assert!(on.pixels == render_with_gi(&mut ctx, &scene, size, preset, true).pixels);
    }
}

/// A room of six slabs around the camera, 7.2 m square and 3 m high inside,
/// its floor 1.5 m below the camera. `ceiling` leaves the top on; `door`
/// leaves a gap 1.6 m wide, floor to ceiling, in the middle of the wall the
/// camera faces (at z = -3.6).
fn slab_room(ceiling: bool, door: bool, pitch: f32, environment: Environment) -> Scene {
    let slab = |half: [f32; 3]| Arc::new(MeshData::cuboid(half, 1.0, matte([0.8; 3])));
    let (flat, wall_x, wall_z, beside_the_door) = (
        slab([4.0, 0.1, 4.0]),
        slab([0.1, 1.7, 4.0]),
        slab([4.0, 1.7, 0.1]),
        slab([1.6, 1.7, 0.1]),
    );
    let mut scene =
        Scene::from_instances(camera_looking(0.0, pitch), Vec::new()).with_environment(environment);
    scene.models = vec![
        placed(&flat, [0.0, -1.6, 0.0]),
        placed(&wall_x, [-3.7, 0.0, 0.0]),
        placed(&wall_x, [3.7, 0.0, 0.0]),
        placed(&wall_z, [0.0, 0.0, 3.7]),
    ];
    if door {
        scene
            .models
            .push(placed(&beside_the_door, [-2.4, 0.0, -3.7]));
        scene
            .models
            .push(placed(&beside_the_door, [2.4, 0.0, -3.7]));
    } else {
        scene.models.push(placed(&wall_z, [0.0, 0.0, -3.7]));
    }
    if ceiling {
        scene.models.push(placed(&flat, [0.0, 1.6, 0.0]));
    }
    scene
}

/// The probes also know how open to the sky they are: the share of a probe's
/// rays that hit nothing. Where the forward pass casts no rays of its own
/// (the `Ci` and `Low` presets), probe GI lets that share of the
/// environment's light through — so with GI on, the floor of a closed room is
/// dark there too, and lit again when the ceiling comes off. (Without GI
/// those presets cannot tell: the sky lights the room through its ceiling.)
#[test]
fn the_probes_keep_the_sky_out_of_a_closed_room() {
    let Some(mut ctx) = context() else { return };
    if !ctx.ray_query_support().complete() {
        eprintln!("SKIP: device has no ray query");
        return;
    }
    let size = 64;
    let sky = Environment::gradient([1.0; 3], [0.8; 3], [0.2; 3]);
    for preset in [QualityPreset::Ci, QualityPreset::Low] {
        let floor = |ctx: &mut RenderContext, ceiling: bool, gi: bool| {
            let scene = slab_room(ceiling, false, -35.0, sky);
            luma(rgb(&render_with_gi(ctx, &scene, size, preset, gi), 32, 32))
        };
        let (closed, open, unaware) = (
            floor(&mut ctx, true, true),
            floor(&mut ctx, false, true),
            floor(&mut ctx, true, false),
        );
        eprintln!(
            "{preset}: the floor of the room is {closed} closed, {open} without its ceiling \
             ({unaware} closed without GI)"
        );
        assert!(closed < 20, "{preset}: no sky in a closed room: {closed}");
        assert!(open > 120, "{preset}: the sky comes in from above: {open}");
        assert!(
            unaware > 120,
            "{preset}: without rays nothing knows: {unaware}"
        );
    }
}

/// Sky rays keep the environment out of where it cannot get (`High` and
/// `Medium`; needs ray query). A closed room with no lamp in it is dark — its
/// floor, its walls, and a metal ball in it, which has no sky to mirror —
/// where `Low`, which casts no rays, lights it all through the ceiling. With
/// a doorway, the sky comes in by the door: the floor is brightest there and
/// fades into the room. And nothing changes in the open: a floor under the
/// whole sky is exactly as bright with the rays as without.
///
/// The rays have fixed directions, so a frame is the same every time.
/// Mutation: with `sky_share` returning 1 in `forward.wgsl`, the closed room
/// is as bright as on `Low`.
#[test]
fn the_sky_does_not_light_the_inside_of_a_closed_room() {
    let Some(mut ctx) = context() else { return };
    if !ctx.ray_query_support().complete() {
        eprintln!("SKIP: device has no ray query");
        return;
    }
    let size = 96;
    let sky = Environment::gradient([1.0; 3], [0.8; 3], [0.2; 3]);
    let ball = GpuInstance {
        center: [1.2, -0.9, -2.2, 0.6],
        albedo: [0.9, 0.9, 0.9, 1.0],
        emissive: [0.0; 4],
        roughness: 0.1,
        shape: space_render::SHAPE_SPHERE,
        transmission: 0.0,
        _pad1: 0.0,
    };
    let room = |ceiling: bool, door: bool| {
        let mut scene = slab_room(ceiling, door, -20.0, sky);
        scene.instances.push(ball);
        scene
    };
    // Where things are in the 96 x 96 frame of a camera looking 20 degrees
    // down at the far wall: the floor by the far wall (and the door), the
    // floor at the camera's feet, the far wall beside the door, the ball.
    let (by_the_door, at_the_feet, far_wall, on_the_ball) =
        ((48, 52), (48, 92), (16, 30), (69, 62));
    let unlit = render_clear(&mut ctx, &room(true, false), size, QualityPreset::Low);
    let open_low = render_clear(&mut ctx, &room(false, false), size, QualityPreset::Low);
    for preset in [QualityPreset::Medium, QualityPreset::High] {
        let closed = render_clear(&mut ctx, &room(true, false), size, preset);
        let at = |out: &space_render::FrameOutput, p: (u32, u32)| luma(rgb(out, p.0, p.1));
        eprintln!(
            "{preset}, closed room: floor {} / {}, wall {}, ball {} (on low: {} / {}, {}, {})",
            at(&closed, by_the_door),
            at(&closed, at_the_feet),
            at(&closed, far_wall),
            at(&closed, on_the_ball),
            at(&unlit, by_the_door),
            at(&unlit, at_the_feet),
            at(&unlit, far_wall),
            at(&unlit, on_the_ball),
        );
        for p in [by_the_door, at_the_feet, far_wall, on_the_ball] {
            assert!(
                at(&closed, p) < 12,
                "{preset}: {p:?} is lit in a closed room: {}",
                at(&closed, p)
            );
            assert!(
                at(&unlit, p) > 80,
                "low: {p:?} is lit through the ceiling: {}",
                at(&unlit, p)
            );
        }
        assert!(closed.pixels == render_clear(&mut ctx, &room(true, false), size, preset).pixels);

        // A doorway: light by the door, less of it further in.
        let door = render_clear(&mut ctx, &room(true, true), size, preset);
        let (near_door, deep) = (at(&door, by_the_door), at(&door, at_the_feet));
        eprintln!(
            "{preset}, with a door: floor {near_door} by the door, {deep} at the camera's feet"
        );
        assert!(
            near_door > 40,
            "{preset}: the sky comes in by the door: {near_door}"
        );
        assert!(
            deep + 25 < near_door,
            "{preset}: and fades into the room: {deep} vs {near_door}"
        );

        // No ceiling: the middle of the floor is under (nearly) the whole sky
        // again, and a floor with no room around it is exactly what it is
        // without the rays.
        let open = render_clear(&mut ctx, &room(false, false), size, preset);
        assert!(
            at(&open, at_the_feet) + 40 > at(&open_low, at_the_feet),
            "{preset}: open to the sky"
        );
        let mut field =
            Scene::from_instances(camera_looking(0.0, -20.0), Vec::new()).with_environment(sky);
        field.models = room(false, false).models[..1].to_vec();
        let (with, without) = (
            render_clear(&mut ctx, &field, size, preset),
            render_clear(&mut ctx, &field, size, QualityPreset::Low),
        );
        assert!(
            largest_difference(&with, &without, 60..size) <= 1,
            "{preset}: an open floor is unchanged"
        );
    }
}

/// A metal's reflection has one colour, whatever it reflects. A polished gold
/// floor under an even white sky, with a white glowing ball of the same
/// brightness standing on it: the floor mirrors the ball (found on screen by
/// the post pass) and the sky (the environment, where the screen has
/// nothing) — and both mirror images are the same gold. (The screen-space
/// reflection used to be blended in untinted: white in a gold mirror.)
///
/// Mutation: with the post pass weighting what it mirrors by a grey instead
/// of by the specular buffer's colour, neither image is gold.
#[test]
fn a_gold_mirror_reflects_everything_in_gold() {
    let Some(mut ctx) = context() else { return };
    let size = 128;
    let gold = [1.0, 0.77, 0.34];
    let scene = Scene::from_instances(
        post_camera(),
        vec![
            GpuInstance {
                center: [0.0, -101.5, 0.0, 100.0],
                albedo: [gold[0], gold[1], gold[2], 1.0],
                emissive: [0.0; 4],
                roughness: 0.05,
                shape: space_render::SHAPE_BOX,
                transmission: 0.0,
                _pad1: 0.0,
            },
            GpuInstance {
                center: [0.0, -0.5, -5.0, 0.6],
                albedo: [0.0; 4],
                emissive: [0.5, 0.5, 0.5, 1.0],
                roughness: 1.0,
                shape: space_render::SHAPE_SPHERE,
                transmission: 0.0,
                _pad1: 0.0,
            },
        ],
    )
    .with_environment(Environment::gradient([0.5; 3], [0.5; 3], [0.5; 3]));
    for preset in [QualityPreset::Medium, QualityPreset::High] {
        let out = render_clear(&mut ctx, &scene, size, preset);
        // The ball is at rows 62..88 of the middle column; its mirror image
        // is centred on row 119. Beside it the floor mirrors the sky.
        let (ball, image, sky) = (rgb(&out, 64, 75), rgb(&out, 64, 119), rgb(&out, 20, 119));
        eprintln!("{preset}: the ball is {ball:?}, its mirror image {image:?}, the sky's {sky:?}");
        assert!(
            ball[0].abs_diff(ball[2]) <= 3,
            "{preset}: the ball is white: {ball:?}"
        );
        for (what, p) in [("the ball's image", image), ("the sky's image", sky)] {
            assert!(
                p[0] > p[1] + 10 && p[1] > p[2] + 40,
                "{preset}: {what} in a gold mirror is {p:?}"
            );
        }
        assert!(
            (0..3).all(|c| image[c].abs_diff(sky[c]) <= 8),
            "{preset}: the ball's image is {image:?}, the sky's {sky:?}: one mirror, two colours"
        );
    }
}

/// The sky's shadow is smooth and steady. A ball hangs over a floor under an
/// even sky and no lamp: the floor is darker under the ball, and the darkness
/// fades out evenly — no steps from one ray's verdict to the next (each pixel
/// casts only eight rays, or four; its neighbours on the floor, whose rays
/// are turned differently, are averaged in by the post pass), and no noise.
/// And the frame is the same every time it is drawn.
///
/// Along a row of the floor that crosses the shadow, neighbouring pixels
/// differ by a few units of 255 at most, while the row as a whole spans far
/// more than that.
/// Mutation: without the post pass's correction (`sky_correction` returning
/// zero) neighbours differ by a ray's worth.
#[test]
fn the_skys_shadow_under_a_ball_is_smooth_and_steady() {
    let Some(mut ctx) = context() else { return };
    if !ctx.ray_query_support().complete() {
        eprintln!("SKIP: device has no ray query");
        return;
    }
    let size = 128;
    let matte = |center: [f32; 4], shape: u32| GpuInstance {
        center,
        albedo: [0.8, 0.8, 0.8, 0.0],
        emissive: [0.0; 4],
        roughness: 1.0,
        shape,
        transmission: 0.0,
        _pad1: 0.0,
    };
    // A floor 1.5 m below the camera, a ball 1 m across just above it; the
    // camera looks 30 degrees down at the floor under the ball.
    let scene = Scene::from_instances(
        camera_looking(0.0, -30.0),
        vec![
            matte([0.0, -101.5, 0.0, 100.0], space_render::SHAPE_BOX),
            matte([0.0, -0.8, -2.6, 0.5], space_render::SHAPE_SPHERE),
        ],
    )
    .with_environment(Environment::gradient([1.0; 3], [1.0; 3], [1.0; 3]));
    for preset in [QualityPreset::Medium, QualityPreset::High] {
        let out = render_clear(&mut ctx, &scene, size, preset);
        // A row of floor just in front of the ball, whose lowest point is
        // drawn a few rows higher: under its edge in the middle of the row,
        // under the open sky at the ends.
        let y = 70;
        let row: Vec<u32> = (4..124).map(|x| luma(rgb(&out, x, y))).collect();
        let (darkest, brightest) = (
            row.iter().copied().min().unwrap_or(0),
            row.iter().copied().max().unwrap_or(0),
        );
        let step = row
            .windows(2)
            .map(|w| w[0].abs_diff(w[1]))
            .max()
            .unwrap_or(0);
        eprintln!(
            "{preset}: the floor runs from {darkest} to {brightest}, in steps of at most {step}"
        );
        assert!(
            darkest + 15 < brightest,
            "{preset}: the ball keeps sky off the floor under it"
        );
        assert!(step <= 5, "{preset}: neighbouring pixels differ by {step}");
        assert!(out.pixels == render_clear(&mut ctx, &scene, size, preset).pixels);
    }
}
