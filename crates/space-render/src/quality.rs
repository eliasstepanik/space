//! Quality presets and the per-preset render-feature set.
//!
//! A [`QualityPreset`] is the one knob a client exposes (`--quality high`).
//! It expands to a [`RenderFeatures`] struct — one field per optional render
//! feature — which `build_uniforms` uploads to the forward pass in
//! `ForwardParams::{fog, features}`.
//!
//! ## Feature matrix
//!
//! | preset | fog | fog density (1/m) |
//! |--------|-----|-------------------|
//! | High   | on  | 2.0e-3            |
//! | Medium | on  | 1.0e-3            |
//! | Low    | off | —                 |
//! | Ci     | off | —                 |
//!
//! `Ci` is the deterministic, cheapest tier: every optional feature is off so
//! golden-pixel tests are independent of feature work. `Low` also has all
//! optional features off. Fog is exponential-squared on distance from the
//! camera and applies to shaded surfaces (after lighting, before tone map);
//! the empty background is not fogged.
//!
//! ## Adding a feature (Phase 5b: SSR, refraction)
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
            },
            Self::Medium => RenderFeatures {
                fog: true,
                fog_density: 1.0e-3,
            },
            Self::Low | Self::Ci => RenderFeatures {
                fog: false,
                fog_density: 0.0,
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
}

impl RenderFeatures {
    /// The `ForwardParams::features` bitmask.
    #[must_use]
    pub const fn flags(self) -> u32 {
        if self.fog {
            FEATURE_FOG
        } else {
            0
        }
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
        assert_eq!(QualityPreset::High.features().flags(), FEATURE_FOG);
        // Presets with fog off must not leak a density to the shader.
        assert_eq!(QualityPreset::Ci.features().fog_density, 0.0);
    }
}
