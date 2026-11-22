use space_render::{
    CameraPose, ContextOptions, FrameOutput, Quality, RenderContext, RenderOptions, Renderer, Scene,
};
use space_world::WorldPos;

fn stats(out: &FrameOutput, size: u32) -> (u64, usize, (u32, u32, [u8; 4])) {
    let mut sum: u64 = 0;
    let mut best = (0u32, 0u32, [0u8; 4]);
    let mut lit = 0usize;
    for y in 0..size {
        for x in 0..size {
            let i = ((y * size + x) * 4) as usize;
            let px = [
                out.pixels[i],
                out.pixels[i + 1],
                out.pixels[i + 2],
                out.pixels[i + 3],
            ];
            let v = px[0] as u32 + px[1] as u32 + px[2] as u32;
            sum += v as u64;
            if px[0] > 8 || px[1] > 8 || px[2] > 8 {
                lit += 1;
            }
            if v > (best.2[0] as u32 + best.2[1] as u32 + best.2[2] as u32) {
                best = (x, y, px);
            }
        }
    }
    (sum, lit, best)
}

fn main() {
    let mut ctx = RenderContext::new(&ContextOptions::default()).unwrap();
    let galaxy = space_world::Galaxy::new(0x5eed);
    let lod = space_world::LodSelector::new(30, 63, 8.0);
    let opts = RenderOptions {
        gi_enabled: false,
        quality: Quality::CI,
        ..Default::default()
    };
    for (dist, fwd, boost) in [
        (5.0e4f64, [0.0f32, 0.0, 1.0], 1.0f32),
        (5.0e4, [0.0, 0.0, 1.0], 2.0),
        (2.0e4, [0.0, 0.0, 1.0], 1.0),
        (2.0e4, [0.0, 0.0, 1.0], 2.0),
        (1.0e4, [0.0, 0.0, 1.0], 1.0),
        (1.0e4, [0.0, 0.0, 1.0], 2.0),
    ] {
        let camera = CameraPose {
            position: WorldPos::from_f64_metres_around(WorldPos::ZERO, [0.0, 0.0, -dist]),
            forward: fwd,
            up: [0.0, 1.0, 0.0],
            fov_y: std::f32::consts::FRAC_PI_3,
            aspect: 1.0,
            near: 1.0,
            far: 5.0e6,
        };
        let mut scene = Scene::build(&galaxy, &lod, &camera);
        for inst in scene.instances.iter_mut() {
            inst.center[3] *= boost;
        }
        let size = 96u32;
        let out = ctx.render_frame(&scene, size, size, &opts).unwrap();
        let (sum, lit, best) = stats(&out, size);
        eprintln!(
            "dist {dist} fwd {fwd:?} boost {boost} inst {} sum {sum} cov {:.5} best {best:?}",
            scene.instances.len(),
            lit as f64 / (size * size) as f64
        );
    }
}
