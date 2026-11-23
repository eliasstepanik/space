//! Render a handful of pictures that show the renderer's model features —
//! textures, the placeholder character, the sample ship, ray-traced shadows,
//! a mirror, metal, glass, the environment and the text overlay — as PNG
//! files, to be looked at:
//!
//! ```sh
//! cargo run -p space-render --example demo_shots -- out_dir [--no-environment]
//! ```
//!
//! The scenes stand under an evening sky (an [`Environment`]): it lights what
//! no lamp reaches, shows in metal and glass, and fills the background.
//! `--no-environment` renders the same pictures without one, to compare.
//!
//! It needs a Vulkan device with ray query (the shadows) and prints how long
//! a frame of the main scene takes.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use space_render::texture::encode_png;
use space_render::{
    rotation_y, sample_character, sample_ship, CameraPose, ContextOptions, Environment,
    FrameOutput, GpuInstance, MeshData, MeshMaterial, ModelInstance, Overlay, Quality,
    QualityPreset, RenderContext, RenderOptions, Scene, SceneLight, Texture, IDENTITY_ROTATION,
    SHAPE_BOX, SHAPE_SPHERE,
};
use space_world::WorldPos;

/// An evening sky: dim enough that the lamps still do the lighting, bright
/// enough that a shadow is not a black hole and that polished metal has
/// something to show. Painted into the background as well.
fn evening_sky() -> Environment {
    Environment::sky()
        .with_intensity(0.35)
        .with_background(true)
}

fn main() {
    let mut dir = PathBuf::from("demo_shots");
    let mut environment = evening_sky();
    for arg in std::env::args_os().skip(1) {
        if arg == "--no-environment" {
            environment = Environment::NONE;
        } else {
            dir = PathBuf::from(arg);
        }
    }
    if let Err(e) = std::fs::create_dir_all(&dir) {
        eprintln!("demo_shots: {}: {e}", dir.display());
        std::process::exit(1);
    }
    let ctx = match RenderContext::new(&ContextOptions::default()) {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!("demo_shots: no usable Vulkan device: {e}");
            std::process::exit(1);
        }
    };
    eprintln!(
        "demo_shots: device {:?}, ray query {}",
        ctx.device_name(),
        ctx.ray_query_support().complete()
    );

    let (width, height) = (1280, 720);
    let area = demo_area(width, height).with_environment(environment);
    for (name, preset, gi) in [
        ("demo_area", QualityPreset::High, false),
        ("demo_area_medium", QualityPreset::Medium, false),
        ("demo_area_low", QualityPreset::Low, false),
        ("demo_area_gi", QualityPreset::High, true),
    ] {
        let options = options(preset, gi);
        let out = render(&ctx, &area, width, height, &options);
        save(&dir, name, &out);
        // A steady stream of the same frame: what one costs, read-back included.
        let frames = 40;
        let start = Instant::now();
        for _ in 0..frames {
            render(&ctx, &area, width, height, &options);
        }
        eprintln!(
            "demo_shots: {name}: {:.2} ms per {width}x{height} frame (shadows {})",
            start.elapsed().as_secs_f64() * 1000.0 / f64::from(frames),
            ctx.shadows_active(&area, &options),
        );
    }

    // The same area from close by: what the environment does to metal, a
    // mirror, glass and a corner no lamp reaches.
    let high = options(QualityPreset::High, false);
    for (name, eye, forward) in [
        ("metal_spheres", [6.4, -0.7, -5.6], [0.25, -0.4, -3.3]),
        ("mirror", [3.6, 0.3, -1.6], [0.0, -0.42, -1.0]),
        ("glass_close", [-3.8, 0.0, -2.1], looking(38.0, -3.0)),
        ("shadowed_corner", [-12.4, -0.3, -2.4], [1.2, -0.7, -4.2]),
    ] {
        let scene = seen_from(demo_area(width, height), eye, forward).with_environment(environment);
        save(&dir, name, &render(&ctx, &scene, width, height, &high));
    }

    for (name, scene) in [
        (
            "character_front",
            character_view(std::f32::consts::PI - 0.5),
        ),
        ("character_back", character_view(0.35)),
        (
            "character_side",
            character_view(std::f32::consts::FRAC_PI_2),
        ),
        ("ship", ship_view()),
        ("textures", texture_wall()),
    ] {
        let size = if name == "textures" {
            (1280, 420)
        } else {
            (560, 700)
        };
        let scene = with_aspect(scene, size.0, size.1).with_environment(environment);
        save(&dir, name, &render(&ctx, &scene, size.0, size.1, &high));
    }
    save(
        &dir,
        "font",
        &render(
            &ctx,
            &font_sheet(),
            640,
            400,
            &options(QualityPreset::Ci, false),
        ),
    );
}

fn options(preset: QualityPreset, gi: bool) -> RenderOptions {
    RenderOptions {
        gi_enabled: gi,
        quality: Quality {
            grid: 8,
            rays_per_probe: 16,
            probe_range: 60.0,
            spacing: 2.0,
        },
        preset,
        ..Default::default()
    }
}

fn render(
    ctx: &RenderContext,
    scene: &Scene,
    width: u32,
    height: u32,
    options: &RenderOptions,
) -> FrameOutput {
    let out = ctx
        .render_offscreen(scene, width, height, options)
        .unwrap_or_else(|e| {
            eprintln!("demo_shots: render failed: {e}");
            std::process::exit(1);
        });
    if out.validation_messages != 0 {
        eprintln!(
            "demo_shots: {} validation messages",
            out.validation_messages
        );
    }
    out
}

fn save(dir: &Path, name: &str, out: &FrameOutput) {
    let path = dir.join(format!("{name}.png"));
    let written = Texture::from_rgba8(out.width, out.height, out.pixels.clone())
        .map_err(|e| e.to_string())
        .and_then(|image| std::fs::write(&path, encode_png(&image)).map_err(|e| e.to_string()));
    match written {
        Ok(()) => eprintln!("demo_shots: wrote {}", path.display()),
        Err(e) => {
            eprintln!("demo_shots: {}: {e}", path.display());
            std::process::exit(1);
        }
    }
}

fn camera(forward: [f32; 3], width: u32, height: u32) -> CameraPose {
    let length = forward.iter().map(|c| c * c).sum::<f32>().sqrt();
    CameraPose {
        position: WorldPos::ZERO,
        forward: forward.map(|c| c / length),
        up: [0.0, 1.0, 0.0],
        fov_y: std::f32::consts::FRAC_PI_3,
        aspect: width as f32 / height as f32,
        near: 0.05,
        far: 1.0e3,
    }
}

fn with_aspect(mut scene: Scene, width: u32, height: u32) -> Scene {
    scene.camera.aspect = width as f32 / height as f32;
    scene
}

/// The view direction for `yaw` / `pitch` degrees: yaw 0 looks down -Z and a
/// positive yaw turns left, a positive pitch looks up.
fn looking(yaw: f32, pitch: f32) -> [f32; 3] {
    let (yaw, pitch) = (yaw.to_radians(), pitch.to_radians());
    [
        -yaw.sin() * pitch.cos(),
        pitch.sin(),
        -yaw.cos() * pitch.cos(),
    ]
}

/// `scene` seen from `eye` (metres from where its camera is) along `forward`,
/// without its overlay. Instances and models are camera-relative, so they
/// move the other way; the lights have world positions and stay.
fn seen_from(mut scene: Scene, eye: [f32; 3], forward: [f32; 3]) -> Scene {
    let length = forward.iter().map(|c| c * c).sum::<f32>().sqrt();
    scene.camera.forward = forward.map(|c| c / length);
    scene.camera.position =
        WorldPos::from_f64_metres_around(scene.camera.position, eye.map(f64::from));
    for instance in &mut scene.instances {
        for (c, e) in instance.center.iter_mut().zip(eye) {
            *c -= e;
        }
    }
    for model in &mut scene.models {
        for (c, e) in model.position.iter_mut().zip(eye) {
            *c -= e;
        }
    }
    scene.overlay = Overlay::new();
    scene
}

fn light(position: [f64; 3], color: [f32; 3], intensity: f32) -> SceneLight {
    SceneLight {
        position: WorldPos::from_f64_metres_around(WorldPos::ZERO, position),
        color,
        intensity,
    }
}

fn paint(color: [f32; 3], metallic: f32, roughness: f32) -> MeshMaterial {
    MeshMaterial {
        base_color: color,
        metallic,
        roughness,
        emissive: [0.0; 3],
        base_color_texture: None,
    }
}

/// `mesh` with `texture` on every part.
fn textured(mut mesh: MeshData, texture: Texture) -> Arc<MeshData> {
    let slot = mesh.add_texture(texture);
    for part in &mut mesh.parts {
        part.material.base_color_texture = Some(slot);
    }
    Arc::new(mesh)
}

fn place(mesh: &Arc<MeshData>, position: [f32; 3], turn: f32, scale: f32) -> ModelInstance {
    ModelInstance {
        mesh: Arc::clone(mesh),
        position,
        rotation: if turn == 0.0 {
            IDENTITY_ROTATION
        } else {
            rotation_y(turn)
        },
        scale,
    }
}

fn instance(
    center: [f32; 3],
    radius: f32,
    shape: u32,
    albedo: [f32; 3],
    metallic: f32,
    roughness: f32,
) -> GpuInstance {
    GpuInstance {
        center: [center[0], center[1], center[2], radius],
        albedo: [albedo[0], albedo[1], albedo[2], metallic],
        emissive: [0.0, 0.0, 0.0, 1.0],
        roughness,
        shape,
        transmission: 0.0,
        _pad1: 0.0,
    }
}

/// A checkered floor whose top is the plane y = `y`, centred under (0, z).
fn checker_floor(y: f32, z: f32) -> ModelInstance {
    let floor = textured(
        MeshData::cuboid([14.0, 0.1, 14.0], 2.0, paint([1.0; 3], 0.0, 0.8)),
        Texture::checkerboard(256, 2, [206, 208, 212], [70, 76, 88]),
    );
    place(&floor, [0.0, y - 0.1, z], 0.0, 1.0)
}

/// The demo area: everything at once, seen from eye height.
fn demo_area(width: u32, height: u32) -> Scene {
    const FLOOR: f32 = -1.7;
    let character = Arc::new(sample_character());
    let ship = Arc::new(sample_ship());
    let crate_mesh = textured(
        MeshData::cuboid([0.5; 3], 1.0, paint([1.0; 3], 0.0, 0.7)),
        Texture::uv_grid(256),
    );
    let pillar = textured(
        MeshData::cuboid([0.25, 1.3, 0.25], 0.5, paint([1.0; 3], 0.0, 0.6)),
        Texture::hazard_stripes(128, 2, [240, 196, 20], [24, 24, 26]),
    );

    let mut lamp = instance([2.5, 3.2, -3.5], 0.12, SHAPE_SPHERE, [1.0; 3], 0.0, 0.5);
    lamp.emissive = [6.0, 5.4, 4.2, 1.0];
    let mut glass = instance(
        [-2.6, FLOOR + 0.7, -4.4],
        0.7,
        SHAPE_BOX,
        [0.9, 0.95, 1.0],
        0.0,
        0.05,
    );
    glass.transmission = 1.0;
    // A glass corner: a cube and a ball at eye height in front of two panels.
    let mut glass_cube = instance(
        [-5.8, FLOOR + 1.5, -6.6],
        0.5,
        SHAPE_BOX,
        [0.9, 0.95, 1.0],
        0.0,
        0.05,
    );
    glass_cube.transmission = 1.0;
    let mut glass_ball = instance(
        [-7.0, FLOOR + 1.55, -5.2],
        0.55,
        SHAPE_SPHERE,
        [0.92, 1.0, 0.95],
        0.0,
        0.05,
    );
    glass_ball.transmission = 1.0;
    let back_panel = textured(
        MeshData::cuboid([1.6, 1.6, 0.08], 3.2, paint([1.0; 3], 0.0, 0.7)),
        Texture::uv_grid(512),
    );
    let side_panel = textured(
        MeshData::cuboid([0.08, 1.6, 1.6], 1.6, paint([1.0; 3], 0.0, 0.7)),
        Texture::checkerboard(256, 4, [236, 238, 242], [36, 150, 96]),
    );
    let pedestal = Arc::new(MeshData::cuboid(
        [0.4, 0.5, 0.4],
        1.0,
        paint([0.12, 0.13, 0.15], 0.0, 0.7),
    ));
    // A wall far to the left with both lamps on its other side: behind it is
    // a corner that no lamp reaches.
    let wall = Arc::new(MeshData::cuboid(
        [0.15, 1.4, 2.0],
        1.0,
        paint([0.75, 0.72, 0.68], 0.0, 0.8),
    ));
    let mut instances = vec![
        // A mirror: a polished slab let into the floor (its top 2 cm proud).
        instance(
            [3.6, FLOOR - 1.48, -5.6],
            1.5,
            SHAPE_BOX,
            [0.9, 0.92, 0.95],
            1.0,
            0.04,
        ),
        // A chrome ball, a smooth gold one and a red plastic one.
        instance(
            [5.9, FLOOR + 0.7, -8.8],
            0.7,
            SHAPE_SPHERE,
            [0.95, 0.95, 0.97],
            1.0,
            0.06,
        ),
        instance(
            [7.4, FLOOR + 0.55, -9.0],
            0.55,
            SHAPE_SPHERE,
            [1.0, 0.77, 0.34],
            1.0,
            0.08,
        ),
        instance(
            [0.9, FLOOR + 0.35, -3.9],
            0.35,
            SHAPE_SPHERE,
            [0.8, 0.08, 0.08],
            0.0,
            0.3,
        ),
        glass,
        glass_cube,
        glass_ball,
        // The lamp around the key light: it lets its own light through.
        lamp,
    ];
    // A row of glowing blocks along the back: something for the mirror, the
    // chrome ball and the GI probes to pick up.
    for (i, color) in [
        [3.0, 0.3, 0.2],
        [0.3, 2.6, 0.4],
        [0.3, 0.6, 3.2],
        [2.6, 2.2, 0.6],
    ]
    .into_iter()
    .enumerate()
    {
        let mut block = instance(
            [-6.0 + 4.0 * i as f32, FLOOR + 0.6, -13.0],
            0.6,
            SHAPE_BOX,
            [0.0; 3],
            0.0,
            0.5,
        );
        block.emissive = [color[0], color[1], color[2], 1.0];
        instances.push(block);
    }
    let mut scene = Scene::from_instances(camera([0.0, -0.22, -1.0], width, height), instances)
        .with_lights(vec![
            light([2.5, 3.2, -3.5], [1.0, 0.95, 0.85], 260.0),
            light([-5.0, 2.5, -5.0], [0.6, 0.75, 1.0], 90.0),
        ]);
    scene.models = vec![
        checker_floor(FLOOR, -8.0),
        // Facing the camera, a little to its left; and one walking away.
        place(&character, [-0.5, FLOOR, -5.2], 2.75, 1.0),
        place(&character, [1.2, FLOOR, -7.0], 0.3, 1.0),
        // The ship hovers low over the mirror.
        place(&ship, [3.6, FLOOR + 0.8, -6.2], 0.7, 0.5),
        // Behind the glass block: the UV crate.
        place(&crate_mesh, [-3.2, FLOOR + 0.5, -6.8], 0.4, 1.0),
        place(&pillar, [6.6, FLOOR + 1.3, -10.5], 0.0, 1.0),
        // The glass corner.
        place(&back_panel, [-6.4, FLOOR + 1.6, -7.9], 0.0, 1.0),
        place(&side_panel, [-8.08, FLOOR + 1.6, -6.3], 0.0, 1.0),
        place(&pedestal, [-5.8, FLOOR + 0.5, -6.6], 0.0, 1.0),
        place(&pedestal, [-7.0, FLOOR + 0.5, -5.2], 0.0, 1.0),
        // The corner in shadow, with a crate and somebody standing in it.
        place(&wall, [-10.0, FLOOR + 1.4, -7.0], 0.0, 1.0),
        place(&crate_mesh, [-11.3, FLOOR + 0.5, -7.2], 0.3, 1.0),
        place(&character, [-11.7, FLOOR, -5.6], 2.9, 1.0),
    ];
    let (white, dim) = ([1.0, 1.0, 1.0, 1.0], [0.85, 0.9, 1.0, 0.9]);
    scene
        .overlay
        .rect(12.0, 12.0, 352.0, 76.0, [0.02, 0.03, 0.06, 0.65]);
    scene.overlay.text(24.0, 22.0, 3, white, "space demo area");
    scene
        .overlay
        .text(24.0, 56.0, 2, dim, "connect: 192.168.0.12:7777");
    scene
}

/// The character alone, turned by `turn` (0 = seen from behind).
fn character_view(turn: f32) -> Scene {
    let character = Arc::new(sample_character());
    let mut scene = Scene::from_instances(camera([0.0, -0.05, -1.0], 4, 5), Vec::new())
        .with_lights(vec![
            light([2.0, 2.5, 0.5], [1.0, 0.97, 0.9], 60.0),
            light([-2.5, 1.0, -1.0], [0.6, 0.7, 1.0], 20.0),
        ]);
    scene.models = vec![
        checker_floor(-0.95, -3.0),
        place(&character, [0.0, -0.95, -2.3], turn, 1.0),
    ];
    scene
}

fn ship_view() -> Scene {
    let ship = Arc::new(sample_ship());
    let mut scene =
        Scene::from_instances(camera([0.0, -0.3, -1.0], 4, 5), Vec::new()).with_lights(vec![
            light([3.0, 4.0, 1.0], [1.0, 0.97, 0.9], 160.0),
            light([-3.0, 1.5, -2.0], [0.6, 0.7, 1.0], 40.0),
        ]);
    scene.models = vec![
        checker_floor(-2.4, -6.0),
        place(&ship, [0.0, -1.2, -5.5], 0.9, 1.0),
    ];
    scene
}

/// The five procedural textures side by side, each on a cube facing the
/// camera, so their orientation can be checked: text reads left to right.
fn texture_wall() -> Scene {
    let textures = [
        Texture::checkerboard(256, 4, [230, 60, 50], [40, 60, 200]),
        Texture::uv_grid(512),
        Texture::hull_panels(256, [236, 238, 242]),
        Texture::brushed_metal(256, [200, 202, 206]),
        Texture::hazard_stripes(256, 4, [240, 196, 20], [24, 24, 26]),
    ];
    let mut scene = Scene::from_instances(camera([0.0, 0.0, -1.0], 3, 1), Vec::new())
        .with_lights(vec![light([0.0, 1.0, 2.0], [1.0; 3], 120.0)]);
    for (i, texture) in textures.into_iter().enumerate() {
        // One texture repeat across each 1.6 m face.
        let cube = textured(
            MeshData::cuboid([0.8; 3], 1.6, paint([1.0; 3], 0.0, 0.8)),
            texture,
        );
        let x = (i as f32 - 2.0) * 1.9;
        scene.models.push(place(&cube, [x, 0.0, -4.6], 0.0, 1.0));
    }
    scene
}

/// Every glyph of the overlay font at scales 1 to 4.
fn font_sheet() -> Scene {
    let mut scene = Scene::from_instances(camera([0.0, 0.0, -1.0], 8, 5), Vec::new());
    let mut overlay = Overlay::new();
    overlay.rect(0.0, 0.0, 640.0, 400.0, [0.05, 0.06, 0.09, 1.0]);
    let lines = [
        " !\"#$%&'()*+,-./0123456789:;<=>?",
        "@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_",
        "`abcdefghijklmnopqrstuvwxyz{|}~",
    ];
    let mut y = 10.0;
    for scale in 1..=3u32 {
        for line in lines {
            overlay.text(10.0, y, scale, [1.0, 1.0, 1.0, 1.0], line);
            y += 9.0 * scale as f32 + 2.0;
        }
        y += 8.0;
    }
    overlay.text(
        10.0,
        y,
        2,
        [0.4, 1.0, 0.5, 1.0],
        "Server address: 192.168.0.12:7777\nConnecting... press Enter (caf\u{e9})",
    );
    overlay.text(10.0, y + 50.0, 4, [1.0, 0.8, 0.3, 1.0], "space 0123456789");
    scene.overlay = overlay;
    scene
}
