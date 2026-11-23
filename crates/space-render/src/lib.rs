//! `space-render` — the ash/Vulkan renderer for the `space` engine (Phase 4).
//!
//! One renderer, built directly on [`ash`] with [`gpu_allocator`] for memory,
//! hidden behind the [`Renderer`] trait so a caller (the native client, a
//! headless capture tool, a future pixel-streaming server) never touches Vulkan
//! types. The pipeline is:
//!
//! * a **forward PBR pass** (Cook-Torrance GGX + Smith visibility, analytic
//!   point lights) into an HDR `rgba16f` target. Nothing is culled: a
//!   triangle seen from behind is drawn too, and shaded as the surface that
//!   side is — the underside of a sheet, the inside of a box the camera is in
//!   — with its normal turned to the viewer,
//! * optional **ray-traced shadows** from those lights (one ray query per
//!   light in the forward pass's fragment shaders, see [`quality`]),
//! * **ray-query probe GI** (`VK_KHR_ray_query`, DDGI-style, 1-bounce diffuse)
//!   feeding that pass as indirect diffuse — what surfaces emit and, under an
//!   environment, the sky light they reflect,
//! * an optional **post pass** (screen-space reflections, tinted by each
//!   surface's specular colour, + refraction, ray traced through the glass
//!   where the device can; see [`quality`]) skipped entirely on the
//!   `Ci`/`Low` presets,
//! * an optional **environment** ([`Environment`]): a sky-and-ground gradient
//!   that gives every surface ambient light and something to reflect, glass
//!   something to show where there is no geometry, and — if asked — the frame
//!   a background,
//! * an **ACES tone map** into an 8-bit image,
//! * a **2D overlay** ([`Overlay`]): rectangles and text in a built-in bitmap
//!   font, blended over the tone-mapped image,
//! * LOD integration: the scene is built from `space-world`'s deterministic
//!   galaxy through its LOD selector (≤ [`space_world::MAX_SELECTED`] nodes),
//! * **mesh models** ([`mesh`]): triangle meshes loaded from `.glb` files or
//!   built in code, with base-colour [`Texture`]s, drawn through the same
//!   shading as the procedural spheres and boxes and part of the same
//!   ray-traced scene (shadows, GI).
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
pub mod environment;
pub mod gpu;
pub mod mesh;
pub mod overlay;
pub mod params;
pub mod passes;
pub mod quality;
pub mod scene;
pub mod swapchain;
pub mod texture;

pub use environment::Environment;
pub use gpu::{
    skip_or_require_gpu, validation_message_count, ContextOptions, FrameOutput, ModelCacheStats,
    Quality, RayQuerySupport, RenderContext, RenderError, RenderOptions, MAX_FRAME_DIMENSION,
    MAX_GI_GRID,
};
pub use mesh::{
    rotation_y, sample_character, sample_ship, MeshData, MeshError, MeshMaterial, MeshPart,
    MeshVertex, ModelInstance, IDENTITY_ROTATION,
};
pub use overlay::Overlay;
pub use params::{
    Camera, ForwardParams, GiParams, GpuInstance, Light, MeshInstanceGpu, PostParams, RqParams,
    ToneParams,
};
pub use quality::{ParsePresetError, QualityPreset, RenderFeatures};
pub use scene::{
    cube_mesh, icosphere_mesh, CameraPose, Scene, SceneLight, BOX_LOD_DEPTH, SHAPE_BOX,
    SHAPE_SPHERE, STAR_RADIUS_M,
};
pub use swapchain::{PresentOutcome, Swapchain, SwapchainConfig, WindowHandles};
pub use texture::Texture;

/// The renderer interface the client and any capture tool code against.
///
/// It exists so a headless caller (tests, pixel streaming, capture tools) can
/// be written without a single Vulkan type in its signature. The windowed
/// path is [`RenderContext::render_to_swapchain`], which presents instead of
/// reading pixels back.
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

/// A deterministic test scene: one emissive sphere `distance` metres in front
/// of a camera looking down `-Z`, plus one magenta cube ahead and to the left
/// of it. Shared by the GPU tests and the client's `--demo` mode so both
/// exercise the same pixels.
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
    let mut scene = Scene::from_instances(
        camera,
        vec![GpuInstance {
            center: [0.0, 0.0, -distance, 1.0],
            albedo: [1.0, 1.0, 1.0, 0.0],
            emissive: [4.0, 4.0, 4.0, 1.0],
            roughness: 0.5,
            shape: crate::scene::SHAPE_SPHERE,
            transmission: 0.0,
            _pad1: 0.0,
        }],
    );
    // A magenta cube just ahead and to the left of the sphere, so a player at
    // spawn sees a solid box as well as the sphere.
    scene.push_box([-1.5, 0.0, -4.0], 0.6, [1.0, 0.05, 0.8]);
    scene.with_lights(vec![SceneLight {
        position: space_world::WorldPos::from_f64_metres_around(
            space_world::WorldPos::ZERO,
            [2.0, 2.0, -2.0],
        ),
        color: [1.0, 1.0, 1.0],
        intensity: 40.0,
    }])
}
