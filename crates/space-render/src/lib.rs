//! `space-render` — the ash/Vulkan renderer for the `space` engine (Phase 4).
//!
//! One renderer, built directly on [`ash`] with [`gpu_allocator`] for memory,
//! hidden behind the [`Renderer`] trait so a caller (the native client, a
//! headless capture tool, a future pixel-streaming server) never touches Vulkan
//! types. The pipeline is:
//!
//! * a **forward PBR pass** (Cook-Torrance GGX + Smith visibility, analytic
//!   point lights) into an HDR `rgba16f` target,
//! * **ray-query probe GI** (`VK_KHR_ray_query`, DDGI-style, 1-bounce diffuse)
//!   feeding that pass as indirect diffuse,
<<<<<<< HEAD
//! * an optional **post pass** (screen-space reflections + refraction, see
//!   [`quality`]) skipped entirely on the `Ci`/`Low` presets,
=======
>>>>>>> refs/remotes/origin/main
//! * an **ACES tone map** into an 8-bit image,
//! * LOD integration: the scene is built from `space-world`'s deterministic
//!   galaxy through its LOD selector (≤ [`space_world::MAX_SELECTED`] nodes).
//!
//! WGSL is compiled to SPIR-V at build time with naga ≥ 30 (see `build.rs`);
//! the ray-query shaders need `enable wgpu_ray_query;`, which naga 24 cannot
//! even parse — see `spikes/rq-triangle/README.md`.
//!
//! ## Headless use
//!
//! ```no_run
//! use space_render::{ContextOptions, Quality, RenderContext, RenderOptions, Renderer};
//!
//! let mut ctx = RenderContext::new(&ContextOptions::default()).unwrap();
//! let out = ctx
//!     .render_frame(&space_render::demo_scene(64, 64), 64, 64, &RenderOptions {
//!         quality: Quality::CI,
//!         ..Default::default()
//!     })
//!     .unwrap();
//! assert_eq!(out.pixels.len(), 64 * 64 * 4);
//! ```
//!
//! Everything is 100% Rust: `ash` is a safe binding that loads the Vulkan
//! loader at runtime, so no C code is linked into this crate.

pub mod alloc;
pub mod gpu;
pub mod params;
pub mod passes;
<<<<<<< HEAD
pub mod quality;
=======
>>>>>>> refs/remotes/origin/main
pub mod scene;
pub mod swapchain;

pub use gpu::{
    validation_message_count, ContextOptions, FrameOutput, Quality, RayQuerySupport, RenderContext,
    RenderError, RenderOptions,
};
<<<<<<< HEAD
pub use params::{
    Camera, ForwardParams, GiParams, GpuInstance, Light, PostParams, RqParams, ToneParams,
};
pub use quality::{ParsePresetError, QualityPreset, RenderFeatures};
=======
pub use params::{Camera, ForwardParams, GiParams, GpuInstance, Light, RqParams, ToneParams};
>>>>>>> refs/remotes/origin/main
pub use scene::{
    cube_mesh, icosphere_mesh, CameraPose, Scene, SceneLight, BOX_LOD_DEPTH, SHAPE_BOX,
    SHAPE_SPHERE, STAR_RADIUS_M,
};
pub use swapchain::{Swapchain, WindowHandles};

/// The renderer interface the client and any capture tool code against.
///
/// It exists so the swapchain path (winit window) and the headless offscreen
/// path (tests, pixel streaming) are interchangeable, and so a caller can be
/// written without a single Vulkan type in its signature.
pub trait Renderer {
    /// Human-readable name of the backing device.
    fn device_name(&self) -> &str;

    /// Ray-query capability, so a caller can decide whether to ask for GI.
    fn ray_query_support(&self) -> RayQuerySupport;

    /// Render one frame offscreen and read the pixels back (RGBA8, top row
    /// first). `options.gi_enabled` is honoured only when the device supports
    /// ray query.
    fn render_frame(
        &mut self,
        scene: &Scene,
        width: u32,
        height: u32,
        options: &RenderOptions,
    ) -> Result<FrameOutput, RenderError>;
}

impl Renderer for RenderContext {
    fn device_name(&self) -> &str {
        RenderContext::device_name(self)
    }

    fn ray_query_support(&self) -> RayQuerySupport {
        RenderContext::ray_query_support(self)
    }

    fn render_frame(
        &mut self,
        scene: &Scene,
        width: u32,
        height: u32,
        options: &RenderOptions,
    ) -> Result<FrameOutput, RenderError> {
        RenderContext::render_offscreen(self, scene, width, height, options)
    }
}

/// A deterministic one-sphere test scene, centred `distance` metres in front of
/// a camera looking down `-Z`. Shared by the GPU tests and the client's
/// `--demo` mode so both exercise the same pixels.
#[must_use]
pub fn demo_scene(width: u32, height: u32) -> Scene {
    let distance = 6.0;
    let camera = CameraPose {
        position: space_world::WorldPos::ZERO,
        forward: [0.0, 0.0, -1.0],
        up: [0.0, 1.0, 0.0],
        fov_y: std::f32::consts::FRAC_PI_3,
        aspect: width as f32 / height as f32,
        near: 0.05,
        far: 1.0e5,
    };
    Scene::from_instances(
        camera,
        vec![GpuInstance {
            center: [0.0, 0.0, -distance, 1.0],
            albedo: [1.0, 1.0, 1.0, 0.0],
            emissive: [4.0, 4.0, 4.0, 1.0],
            roughness: 0.5,
            shape: crate::scene::SHAPE_SPHERE,
<<<<<<< HEAD
            transmission: 0.0,
=======
            _pad0: 0.0,
>>>>>>> refs/remotes/origin/main
            _pad1: 0.0,
        }],
    )
    .with_lights(vec![SceneLight {
        position: space_world::WorldPos::from_f64_metres_around(
            space_world::WorldPos::ZERO,
            [2.0, 2.0, -2.0],
        ),
        color: [1.0, 1.0, 1.0],
        intensity: 40.0,
    }])
}
