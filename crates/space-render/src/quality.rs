//! Quality presets and the per-preset render-feature set.
//!
//! A [`QualityPreset`] is the one knob a client exposes (`--quality high`).
//! It expands to a [`RenderFeatures`] struct — one field per optional render
//! feature — which `build_uniforms` uploads to the forward pass in
//! `ForwardParams::{fog, features}`.
//!
//! ## Feature matrix
//!
//! | preset | fog | fog density (1/m) | SSR | SSR steps | SSR max roughness | refraction |
//! |--------|-----|-------------------|-----|-----------|-------------------|------------|
//! | High   | on  | 2.0e-3            | on  | 32        | 0.5               | on         |
//! | Medium | on  | 1.0e-3            | on  | 16        | 0.3               | on         |
//! | Low    | off | —                 | off | —         | —                 | off        |
//! | Ci     | off | —                 | off | —         | —                 | off        |
//!
//! `Ci` is the deterministic, cheapest tier: every optional feature is off so
//! golden-pixel tests are independent of feature work. `Low` also has all
//! optional features off. Fog is exponential-squared on distance from the
//! camera and applies to shaded surfaces (after lighting, before tone map);
//! the empty background is not fogged.
//!
//! SSR (screen-space reflections) and refraction are done by the `post.wgsl`
//! compute pass, which runs between the forward pass(es) and the tone map and
//! is **not dispatched at all** unless `FEATURE_SSR` or `FEATURE_REFRACT` is
//! set. Because it runs after the forward pass, fog is already part of the
//! scene colour it reflects/refracts, so fog stays applied to the composited
//! result. Refraction additionally splits transmissive instances
//! (`GpuInstance::transmission > 0`) into a second "glass" draw (only when
//! `FEATURE_REFRACT` is set; otherwise they render as ordinary opaque
//! surfaces) so the post pass can see the opaque scene *behind* the glass.
//!
//! ## Adding a feature
//!
//! 1. add a `bool` (and any level) to [`RenderFeatures`] and set it in each
//!    arm of [`QualityPreset::features`];
//! 2. add a `FEATURE_*` bit here and in `shaders/forward.wgsl`, and set it in
//!    [`RenderFeatures::flags`];
//! 3. extend the matrix above and the matrix test.

use std::fmt;
use std::str::FromStr;

/// Bit in `ForwardParams::features`: distance fog enabled.
pub const FEATURE_FOG: u32 = 1 << 0;

/// Bit in `ForwardParams::features` / `PostParams::features`: screen-space
/// reflections (post pass).
pub const FEATURE_SSR: u32 = 1 << 1;

/// Bit in `ForwardParams::features` / `PostParams::features`: refraction
/// through transmissive instances (glass draw + post pass).
pub const FEATURE_REFRACT: u32 = 1 << 2;

/// Mask of the features that need the post pass.
pub const POST_FEATURES: u32 = FEATURE_SSR | FEATURE_REFRACT;

/// Linear-light fog colour (pre tone map) used when fog is enabled.
pub const FOG_COLOR: [f32; 3] = [0.35, 0.45, 0.60];

/// A named quality tier.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum QualityPreset {
    #[default]
    High,
    Medium,
    Low,
    /// Deterministic, cheapest tier for CI: all optional features off.
    Ci,
}

impl QualityPreset {
    /// Every preset, highest quality first.
    pub const ALL: [Self; 4] = [Self::High, Self::Medium, Self::Low, Self::Ci];

    /// The render features this preset enables.
    #[must_use]
    pub const fn features(self) -> RenderFeatures {
        match self {
            Self::High => RenderFeatures {
                fog: true,
                fog_density: 2.0e-3,
                ssr: true,
                ssr_steps: 32,
                ssr_max_roughness: 0.5,
                refraction: true,
            },
            Self::Medium => RenderFeatures {
                fog: true,
                fog_density: 1.0e-3,
                ssr: true,
                ssr_steps: 16,
                ssr_max_roughness: 0.3,
                refraction: true,
            },
            Self::Low | Self::Ci => RenderFeatures {
                fog: false,
                fog_density: 0.0,
                ssr: false,
                ssr_steps: 0,
                ssr_max_roughness: 0.0,
                refraction: false,
            },
        }
    }

    /// Lower-case name, as accepted by [`FromStr`].
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::High => "high",
            Self::Medium => "medium",
            Self::Low => "low",
            Self::Ci => "ci",
        }
    }
}

impl fmt::Display for QualityPreset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Error for an unrecognised preset name.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParsePresetError(pub String);

impl fmt::Display for ParsePresetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "unknown quality preset {:?} (expected high, medium, low or ci)",
            self.0
        )
    }
}

impl std::error::Error for ParsePresetError {}

impl FromStr for QualityPreset {
    type Err = ParsePresetError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "high" => Ok(Self::High),
            "medium" => Ok(Self::Medium),
            "low" => Ok(Self::Low),
            "ci" => Ok(Self::Ci),
            _ => Err(ParsePresetError(s.to_string())),
        }
    }
}

/// Optional render features a preset can toggle (one field per feature).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RenderFeatures {
    /// Exponential-squared distance fog in the forward pass.
    pub fog: bool,
    /// Fog density in 1/metre (`f = 1 - exp(-(d * density)^2)`); ignored when
    /// `fog` is off.
    pub fog_density: f32,
    /// Screen-space reflections for smooth metals (post pass).
    pub ssr: bool,
    /// Ray-march steps per SSR ray; ignored when `ssr` is off.
    pub ssr_steps: u32,
    /// Largest roughness that still gets SSR; ignored when `ssr` is off.
    pub ssr_max_roughness: f32,
    /// Refraction through transmissive instances (glass draw + post pass).
    pub refraction: bool,
}

impl RenderFeatures {
    /// The `ForwardParams::features` bitmask.
    #[must_use]
    pub const fn flags(self) -> u32 {
        let mut f = 0;
        if self.fog {
            f |= FEATURE_FOG;
        }
        if self.ssr {
            f |= FEATURE_SSR;
        }
        if self.refraction {
            f |= FEATURE_REFRACT;
        }
        f
    }

    /// True when the post pass must be dispatched.
    #[must_use]
    pub const fn needs_post(self) -> bool {
        self.flags() & POST_FEATURES != 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_names_case_insensitively() {
        for p in QualityPreset::ALL {
            assert_eq!(p.name().parse::<QualityPreset>(), Ok(p));
            assert_eq!(p.name().to_uppercase().parse::<QualityPreset>(), Ok(p));
            assert_eq!(p.to_string(), p.name());
        }
        assert!("ultra".parse::<QualityPreset>().is_err());
        assert!("".parse::<QualityPreset>().is_err());
    }

    #[test]
    fn default_is_high() {
        assert_eq!(QualityPreset::default(), QualityPreset::High);
    }

    #[test]
    fn feature_matrix() {
        let fog = |p: QualityPreset| p.features().fog;
        assert!(fog(QualityPreset::High));
        assert!(fog(QualityPreset::Medium));
        assert!(!fog(QualityPreset::Low));
        assert!(!fog(QualityPreset::Ci));
        assert!(
            QualityPreset::High.features().fog_density
                > QualityPreset::Medium.features().fog_density
        );
        assert_eq!(QualityPreset::Ci.features().flags(), 0);
        assert_eq!(QualityPreset::Low.features().flags(), 0);
        assert_eq!(
            QualityPreset::High.features().flags(),
            FEATURE_FOG | FEATURE_SSR | FEATURE_REFRACT
        );
        assert_eq!(
            QualityPreset::Medium.features().flags(),
            FEATURE_FOG | FEATURE_SSR | FEATURE_REFRACT
        );
        // Presets with fog off must not leak a density to the shader.
        assert_eq!(QualityPreset::Ci.features().fog_density, 0.0);

        // SSR / refraction: High and Medium on, Low and Ci off.
        let (high, medium) = (QualityPreset::High.features(), QualityPreset::Medium.features());
        assert!(high.ssr && high.refraction && high.needs_post());
        assert!(medium.ssr && medium.refraction && medium.needs_post());
        assert_eq!((high.ssr_steps, medium.ssr_steps), (32, 16));
        assert_eq!(high.ssr_max_roughness, 0.5);
        assert_eq!(medium.ssr_max_roughness, 0.3);
        for p in [QualityPreset::Low, QualityPreset::Ci] {
            let f = p.features();
            assert!(!f.ssr && !f.refraction && !f.needs_post(), "{p}");
            assert_eq!((f.ssr_steps, f.ssr_max_roughness), (0, 0.0), "{p}");
            assert_eq!(f.flags() & POST_FEATURES, 0, "{p}");
        }
    }
}
