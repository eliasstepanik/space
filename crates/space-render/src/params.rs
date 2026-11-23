//! CPU-side mirrors of the WGSL uniform/storage structs.
//!
//! Hard constraint: a WGSL `vec3` takes 16 bytes in SPIR-V, so every struct
//! here uses only `vec4`-sized members (or explicit padding) and the unit
//! tests at the bottom assert each struct's size and member offsets match the
//! WGSL layout in `shaders/`.

/// `Params` in `shaders/rq_triangle.wgsl` (80 bytes).
/// Offsets: 0 hit_color, 16 miss_color, 32 camera_origin, 48 image_size,
/// 64 tan_half_fov -> 80.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct RqParams {
    pub hit_color: [f32; 4],
    pub miss_color: [f32; 4],
    pub camera_origin: [f32; 4],
    pub image_size: [f32; 4],
    pub tan_half_fov: f32,
    pub _pad: [f32; 3],
}

/// `GiParams` in `shaders/gi_probe.wgsl` (48 bytes).
///
/// WGSL struct size rule: the size is `round_up(end_of_last_member,
/// alignment_of_struct)`, and a struct's alignment is the largest alignment of
/// any member — here 16, from `grid_origin: vec4<f32>`. The last real member
/// ends at 40, so the block size is `round_up(40, 16) = 48` and the uniform
/// buffer must be bound with range 48, not 40.
/// Offsets: 0 grid_origin, 16 spacing, 20 probe_range, 24 grid_size,
/// 32 rays_per_probe, 36..48 tail padding -> 48.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct GiParams {
    pub grid_origin: [f32; 4],
    pub spacing: f32,
    pub probe_range: f32,
    pub grid_size: [u32; 2],
    pub rays_per_probe: u32,
    /// Explicit tail padding so the Rust mirror is the full WGSL block size
    /// (the shader's own `_pad: f32` plus the 8 bytes the WGSL round-up adds).
    pub _pad: [f32; 3],
}

/// `Instance` in `shaders/common.wgsl` (64 bytes).
/// Offsets: 0 center, 16 albedo, 32 emissive, 48 roughness, 52 shape,
/// 56 transmission -> 64.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct GpuInstance {
    /// Camera-relative world centre, w = radius.
    pub center: [f32; 4],
    /// rgb + metallic in .w.
    pub albedo: [f32; 4],
    /// rgb radiance, a = GI ray contribution weight.
    pub emissive: [f32; 4],
    pub roughness: f32,
    /// 0 = sphere, 1 = box.
    pub shape: u32,
    /// 0 = opaque; > 0 = transmissive ("glass"): with refraction enabled the
    /// surface shows the refracted scene behind it (`1` = fully transmissive,
    /// only Fresnel reflection of the lit colour on top).
    pub transmission: f32,
    pub _pad1: f32,
}

/// `MeshInstance` in `shaders/common.wgsl` (80 bytes): one drawn part of a
/// mesh model. The forward pass draws with it; a GI ray that hits the model
/// reads its part's emission from it.
/// Offsets: 0 model_x, 16 model_y, 32 model_z, 48 albedo, 64 emissive -> 80.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct MeshInstanceGpu {
    /// Rows of the 3x4 model -> camera-relative world matrix.
    pub model_x: [f32; 4],
    pub model_y: [f32; 4],
    pub model_z: [f32; 4],
    /// rgb base-colour factor + metallic in .w.
    pub albedo: [f32; 4],
    /// rgb radiance + roughness in .w.
    pub emissive: [f32; 4],
}

/// One rectangle of the overlay: the instance-rate vertex of `vs_overlay` in
/// `shaders/overlay.wgsl` (32 bytes; attribute 0 at offset 0, 1 at offset 16).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct OverlayInstance {
    /// x0, y0, x1, y1 in normalised device coordinates (y = -1 is the top).
    pub rect: [f32; 4],
    /// Display-referred sRGB, straight alpha.
    pub color: [f32; 4],
}

/// `Environment` in `shaders/common.wgsl` (48 bytes): the scene's sky-and-ground
/// gradient ([`crate::Environment`]) as the shaders read it.
/// Offsets: 0 zenith, 16 horizon, 32 ground -> 48.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct EnvironmentGpu {
    /// rgb radiance from straight up (intensity included); w = 1 when the
    /// scene has an environment and 0 when it has none — the one value a
    /// shader reads to skip everything else.
    pub zenith: [f32; 4],
    /// rgb radiance from the horizon.
    pub horizon: [f32; 4],
    /// rgb radiance from straight down.
    pub ground: [f32; 4],
}

/// `Camera` in `shaders/forward.wgsl` (160 bytes: two `mat4x4` at 64 bytes
/// each plus two `vec4`s).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Camera {
    pub view_proj: [[f32; 4]; 4],
    pub inv_view_proj: [[f32; 4]; 4],
    /// Camera-relative world position, w = tan(fov_y / 2).
    pub position: [f32; 4],
    /// View direction, w = near plane.
    pub forward: [f32; 4],
}

/// `Light` in `shaders/forward.wgsl` (32 bytes; array stride 32).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Light {
    /// xyz camera-relative, w = 0 when the slot is inactive.
    pub position: [f32; 4],
    /// rgb linear radiance, a = intensity.
    pub color: [f32; 4],
}

/// `Params` in `shaders/forward.wgsl` (544 bytes).
/// Offsets: 0 camera(160), 160 lights(8x32=256) -> 416, 416 light_count,
/// 420 gi_enabled, 424 _pad0(8) -> 432, 432 gi_grid_origin(16) -> 448,
/// 448 gi_spacing, 452 gi_grid_size_x, 456 gi_grid_size_y, 460 _pad1 -> 464,
/// 464 fog (rgb colour, w = density), 480 features (bitmask, see
/// [`crate::quality`]), 484..496 padding, 496 environment(48) -> 544.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct ForwardParams {
    pub camera: Camera,
    pub lights: [Light; 8],
    pub light_count: u32,
    pub gi_enabled: u32,
    pub _pad0: [f32; 2],
    pub gi_grid_origin: [f32; 4],
    pub gi_spacing: f32,
    pub gi_grid_size_x: u32,
    pub gi_grid_size_y: u32,
    pub _pad1: f32,
    /// Fog colour (linear rgb) with the exp-squared density in `.w`.
    pub fog: [f32; 4],
    /// `FEATURE_*` bitmask from [`crate::quality`].
    pub features: u32,
    pub _pad2: [u32; 3],
    /// The scene's environment (all zero for none).
    pub environment: EnvironmentGpu,
}

/// `PostParams` in `shaders/post.wgsl` (240 bytes).
/// Offsets: 0 camera(160), 160 features, 164 ssr_steps, 168 image_size(8),
/// 176 ssr_max_roughness, 180 refract_distance, 184 refract_eta, 188 pad -> 192,
/// 192 environment(48) -> 240.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct PostParams {
    pub camera: Camera,
    /// `FEATURE_*` bitmask from [`crate::quality`].
    pub features: u32,
    pub ssr_steps: u32,
    pub image_size: [u32; 2],
    pub ssr_max_roughness: f32,
    /// Virtual thickness (metres) the refracted ray travels behind the surface
    /// before its screen position is looked up.
    pub refract_distance: f32,
    /// Index-of-refraction ratio entering the medium (air -> glass = 1/1.5).
    pub refract_eta: f32,
    pub _pad: f32,
    /// The scene's environment (all zero for none): what glass shows where
    /// its refracted view finds no geometry.
    pub environment: EnvironmentGpu,
}

/// `ToneParams` in `shaders/tonemap.wgsl` (32 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct ToneParams {
    pub exposure: f32,
    pub _pad0: [f32; 3],
    pub image_size: [u32; 2],
    pub _pad1: [f32; 2],
}

/// Minimal byte-view helpers (kept local so the crate needs no `bytemuck`).
pub(crate) mod bytemuck_free {
    /// Byte view of a `#[repr(C)]` plain-old-data value.
    ///
    /// # Safety
    /// The caller must only pass types that are `repr(C)`, `Copy`, contain no
    /// padding-dependent semantics and have no invalid bit patterns — every
    /// struct in this module qualifies (all members are `f32`/`u32`).
    pub unsafe fn pod_bytes<T>(value: &T) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(value as *const T as *const u8, std::mem::size_of::<T>())
        }
    }
}

use bytemuck_free::pod_bytes;

/// Marker for the uniform/storage mirrors in this module: `repr(C)`, made only
/// of `f32`/`u32` members, with no padding bytes — so viewing one as bytes is
/// always sound. Sealed to this module's structs.
pub trait GpuPod: Copy + sealed::Sealed {}

mod sealed {
    pub trait Sealed {}
}

macro_rules! gpu_pod {
    ($($ty:ty),* $(,)?) => {
        $(impl sealed::Sealed for $ty {} impl GpuPod for $ty {})*
    };
}

gpu_pod!(
    RqParams,
    GiParams,
    GpuInstance,
    MeshInstanceGpu,
    OverlayInstance,
    EnvironmentGpu,
    Camera,
    Light,
    ForwardParams,
    PostParams,
    ToneParams,
);

/// Bytes of `value`, for buffer uploads.
pub fn as_bytes<T: GpuPod>(value: &T) -> &[u8] {
    // SAFETY: `GpuPod` types are repr(C) f32/u32 plain data without padding.
    unsafe { pod_bytes(value) }
}

/// Byte view of a whole slice of plain-old-data values.
///
/// # Safety
/// `T` must be `repr(C)` plain data with no padding bytes and no invalid bit
/// patterns (every [`GpuPod`] type, and Vulkan's instance record, qualify).
pub(crate) unsafe fn slice_bytes<T: Copy>(values: &[T]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The WGSL block sizes and member offsets, **generated by `build.rs` from
    // the parsed WGSL itself** (naga's `StructMember::offset` / struct `span`).
    // Asserting against these instead of hand-transcribed literals is what
    // makes WGSL-side layout drift (e.g. a scalar pad reverted to `vec3`) fail
    // a test rather than silently move a member.
    include!(concat!(env!("OUT_DIR"), "/wgsl_layout.rs"));

    /// Asserts `T`'s size and the byte offset of each named member against the
    /// generated WGSL layout constants.
    macro_rules! assert_layout {
        ($ty:ty { $($field:ident => $off:expr),* $(,)? } size = $size:expr) => {{
            const SIZE: usize = std::mem::size_of::<$ty>();
            assert_eq!(SIZE, $size, concat!("size_of ", stringify!($ty)));
            $(
                assert_eq!(
                    std::mem::offset_of!($ty, $field),
                    $off,
                    concat!(stringify!($ty), ".", stringify!($field))
                );
            )*
        }};
    }

    #[test]
    fn rq_params_matches_wgsl() {
        assert_layout!(RqParams {
            hit_color => RQPARAMS_HIT_COLOR,
            miss_color => RQPARAMS_MISS_COLOR,
            camera_origin => RQPARAMS_CAMERA_ORIGIN,
            image_size => RQPARAMS_IMAGE_SIZE,
            tan_half_fov => RQPARAMS_TAN_HALF_FOV,
        } size = RQPARAMS_SIZE);
    }

    #[test]
    fn gi_params_matches_wgsl() {
        assert_layout!(GiParams {
            grid_origin => GIPARAMS_GRID_ORIGIN,
            spacing => GIPARAMS_SPACING,
            probe_range => GIPARAMS_PROBE_RANGE,
            grid_size => GIPARAMS_GRID_SIZE,
            rays_per_probe => GIPARAMS_RAYS_PER_PROBE,
        } size = GIPARAMS_SIZE);
        // The WGSL block rounds up to a multiple of 16 (40 -> 48); the uniform
        // buffer is bound with this size, so pin it explicitly.
        assert_eq!(GIPARAMS_SIZE, 48);
    }

    #[test]
    fn gpu_instance_matches_wgsl() {
        assert_layout!(GpuInstance {
            center => GPUINSTANCE_CENTER,
            albedo => GPUINSTANCE_ALBEDO,
            emissive => GPUINSTANCE_EMISSIVE,
            roughness => GPUINSTANCE_ROUGHNESS,
            shape => GPUINSTANCE_SHAPE,
            transmission => GPUINSTANCE_TRANSMISSION,
        } size = GPUINSTANCE_SIZE);
    }

    #[test]
    fn mesh_instance_matches_wgsl() {
        assert_layout!(MeshInstanceGpu {
            model_x => MESHINSTANCEGPU_MODEL_X,
            model_y => MESHINSTANCEGPU_MODEL_Y,
            model_z => MESHINSTANCEGPU_MODEL_Z,
            albedo => MESHINSTANCEGPU_ALBEDO,
            emissive => MESHINSTANCEGPU_EMISSIVE,
        } size = MESHINSTANCEGPU_SIZE);
    }

    #[test]
    fn forward_params_matches_wgsl() {
        assert_layout!(ForwardParams {
            camera => FORWARDPARAMS_CAMERA,
            lights => FORWARDPARAMS_LIGHTS,
            light_count => FORWARDPARAMS_LIGHT_COUNT,
            gi_enabled => FORWARDPARAMS_GI_ENABLED,
            gi_grid_origin => FORWARDPARAMS_GI_GRID_ORIGIN,
            gi_spacing => FORWARDPARAMS_GI_SPACING,
            gi_grid_size_x => FORWARDPARAMS_GI_GRID_SIZE_X,
            gi_grid_size_y => FORWARDPARAMS_GI_GRID_SIZE_Y,
            fog => FORWARDPARAMS_FOG,
            features => FORWARDPARAMS_FEATURES,
            environment => FORWARDPARAMS_ENVIRONMENT,
        } size = FORWARDPARAMS_SIZE);
        assert_layout!(EnvironmentGpu {
            zenith => ENVIRONMENTGPU_ZENITH,
            horizon => ENVIRONMENTGPU_HORIZON,
            ground => ENVIRONMENTGPU_GROUND,
        } size = ENVIRONMENTGPU_SIZE);
        assert_layout!(Camera {
            view_proj => CAMERA_VIEW_PROJ,
            inv_view_proj => CAMERA_INV_VIEW_PROJ,
            position => CAMERA_POSITION,
            forward => CAMERA_FORWARD,
        } size = CAMERA_SIZE);
        assert_layout!(Light {
            position => LIGHT_POSITION,
            color => LIGHT_COLOR,
        } size = LIGHT_SIZE);
    }

    #[test]
    fn post_params_matches_wgsl() {
        assert_layout!(PostParams {
            camera => POSTPARAMS_CAMERA,
            features => POSTPARAMS_FEATURES,
            ssr_steps => POSTPARAMS_SSR_STEPS,
            image_size => POSTPARAMS_IMAGE_SIZE,
            ssr_max_roughness => POSTPARAMS_SSR_MAX_ROUGHNESS,
            refract_distance => POSTPARAMS_REFRACT_DISTANCE,
            refract_eta => POSTPARAMS_REFRACT_ETA,
            environment => POSTPARAMS_ENVIRONMENT,
        } size = POSTPARAMS_SIZE);
    }

    #[test]
    fn tone_params_matches_wgsl() {
        assert_layout!(ToneParams {
            exposure => TONEPARAMS_EXPOSURE,
            image_size => TONEPARAMS_IMAGE_SIZE,
        } size = TONEPARAMS_SIZE);
    }

    /// Every WGSL member that the Rust mirror models must sit at the generated
    /// offset *and* the whole struct must be the generated block size, so a
    /// struct can never be shorter than the WGSL block it mirrors.
    #[test]
    fn every_mirror_is_at_least_the_wgsl_block_size() {
        assert_eq!(std::mem::size_of::<RqParams>(), RQPARAMS_SIZE);
        assert_eq!(std::mem::size_of::<GiParams>(), GIPARAMS_SIZE);
        assert_eq!(std::mem::size_of::<GpuInstance>(), GPUINSTANCE_SIZE);
        assert_eq!(std::mem::size_of::<ForwardParams>(), FORWARDPARAMS_SIZE);
        assert_eq!(std::mem::size_of::<PostParams>(), POSTPARAMS_SIZE);
        assert_eq!(std::mem::size_of::<ToneParams>(), TONEPARAMS_SIZE);
    }

    /// `as_bytes` exposes every byte of the struct, and each field is readable
    /// at its asserted offset: write a distinct value into every field and read
    /// it back at the offset the WGSL layout says it lives at.
    #[test]
    fn as_bytes_covers_the_whole_struct() {
        let p = RqParams {
            hit_color: [1.0, 2.0, 3.0, 4.0],
            miss_color: [5.0, 6.0, 7.0, 8.0],
            camera_origin: [9.0, 10.0, 11.0, 12.0],
            image_size: [13.0, 14.0, 15.0, 16.0],
            tan_half_fov: 17.0,
            _pad: [18.0, 19.0, 20.0],
        };
        let bytes = as_bytes(&p);
        assert_eq!(bytes.len(), RQPARAMS_SIZE);
        let f32_at = |offset: usize| {
            f32::from_le_bytes(
                bytes[offset..offset + 4]
                    .try_into()
                    .expect("4 bytes of f32"),
            )
        };
        for (i, value) in p.hit_color.iter().enumerate() {
            assert_eq!(f32_at(RQPARAMS_HIT_COLOR + i * 4), *value);
        }
        for (i, value) in p.miss_color.iter().enumerate() {
            assert_eq!(f32_at(RQPARAMS_MISS_COLOR + i * 4), *value);
        }
        for (i, value) in p.camera_origin.iter().enumerate() {
            assert_eq!(f32_at(RQPARAMS_CAMERA_ORIGIN + i * 4), *value);
        }
        for (i, value) in p.image_size.iter().enumerate() {
            assert_eq!(f32_at(RQPARAMS_IMAGE_SIZE + i * 4), *value);
        }
        assert_eq!(f32_at(RQPARAMS_TAN_HALF_FOV), p.tan_half_fov);
        for (i, value) in p._pad.iter().enumerate() {
            assert_eq!(f32_at(RQPARAMS_TAN_HALF_FOV + 4 + i * 4), *value);
        }
        // The last byte of the struct is covered, i.e. nothing is truncated.
        assert_eq!(
            f32_at(RQPARAMS_SIZE - 4),
            p._pad[2],
            "as_bytes must reach the end of the block"
        );
    }
}
