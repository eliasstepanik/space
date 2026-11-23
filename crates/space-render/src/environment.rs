//! The scene's surroundings: a sky-and-ground colour gradient.
//!
//! Without an [`Environment`] a scene is lit by its point lights (and probe
//! GI) alone: a surface no light reaches is black, and a mirror is black
//! wherever it reflects something that is not on screen. An environment gives
//! every direction a radiance — a vertical gradient from `ground` (straight
//! down) through `horizon` to `zenith` (straight up) — and the renderer uses
//! it four ways:
//!
//! * **ambient light**: every surface gets the gradient's light from the
//!   hemisphere its normal faces, times its base colour (metals have no
//!   diffuse term and get none of this);
//! * **reflections**: every surface reflects the gradient along its mirror
//!   direction, blurred with roughness and weighted by Fresnel — which is also
//!   what remains wherever a screen-space reflection finds nothing to show;
//! * **refraction**: glass with no geometry behind it — neither where its
//!   bent view lands nor straight behind it — shows the gradient along the
//!   bent view instead of black;
//! * **background**, only when [`Environment::background`] is set: the pixels
//!   no geometry covers show the gradient instead of staying black, and
//!   distance fog fades into the gradient behind each pixel instead of into
//!   one fog colour.
//!
//! The one place it does not reach is the inside of a closed solid — a sphere
//! or box instance, or a closed mesh, seen from within (a camera that has
//! flown into it): there is no sky in there, so those faces get neither the
//! ambient light nor the reflection. An open mesh (a sheet) has two outsides
//! and gets both on both. Elsewhere, with ray query, the forward pass's sky
//! rays occlude both terms per pixel (see `quality.rs`), so a room made of
//! separate walls is as dark as its openings make it; without ray query
//! (`Low`, `Ci`) nothing else occludes the environment.
//!
//! [`Environment::default`] is *no environment*: such a scene renders exactly
//! as it would if this module did not exist.
//!
//! The functions here ([`Environment::radiance`], [`Environment::ambient`],
//! [`Environment::reflection`]) are the CPU twins of `env_radiance`,
//! `env_ambient` and `env_reflection` in `shaders/common.wgsl`: the same
//! formulas, so a test can say what colour a pixel must have.

/// A sky-and-ground gradient around the scene. See the [module docs](self).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Environment {
    /// Radiance from straight up, linear rgb.
    pub zenith: [f32; 3],
    /// Radiance from the horizon, linear rgb.
    pub horizon: [f32; 3],
    /// Radiance from straight down (the ground's bounce), linear rgb.
    pub ground: [f32; 3],
    /// Multiplies all three colours. 0 (or less) is no environment at all.
    pub intensity: f32,
    /// Also draw the gradient where no geometry was drawn. Off by default:
    /// a scene can have ambient light and reflections of a sky without
    /// painting one over its own backdrop (a star field, say).
    pub background: bool,
}

impl Default for Environment {
    /// No environment.
    fn default() -> Self {
        Self::NONE
    }
}

impl Environment {
    /// No environment: nothing is added to any pixel.
    pub const NONE: Self = Self {
        zenith: [0.0; 3],
        horizon: [0.0; 3],
        ground: [0.0; 3],
        intensity: 0.0,
        background: false,
    };

    /// A gradient of the given colours (linear rgb radiance) at intensity 1,
    /// not drawn as the background.
    #[must_use]
    pub const fn gradient(zenith: [f32; 3], horizon: [f32; 3], ground: [f32; 3]) -> Self {
        Self {
            zenith,
            horizon,
            ground,
            intensity: 1.0,
            background: false,
        }
    }

    /// A clear daytime sky over neutral ground: blue above, pale at the
    /// horizon, a dim grey-brown bounce from below.
    #[must_use]
    pub const fn sky() -> Self {
        Self::gradient([0.18, 0.36, 0.80], [0.70, 0.78, 0.88], [0.20, 0.18, 0.16])
    }

    /// Even, colourless surroundings (a photographer's light tent): brighter
    /// above than below, no tint. Good for looking at materials.
    #[must_use]
    pub const fn studio() -> Self {
        Self::gradient([0.9, 0.9, 0.9], [0.55, 0.55, 0.55], [0.18, 0.18, 0.18])
    }

    /// The same gradient at `intensity`.
    #[must_use]
    pub const fn with_intensity(mut self, intensity: f32) -> Self {
        self.intensity = intensity;
        self
    }

    /// The same gradient, drawn as the background or not.
    #[must_use]
    pub const fn with_background(mut self, background: bool) -> Self {
        self.background = background;
        self
    }

    /// True when the environment contributes nothing: no (or a negative or
    /// non-finite) intensity, or three black colours. The renderer then skips
    /// every environment term.
    #[must_use]
    pub fn is_none(&self) -> bool {
        let lit = self.colors().iter().flatten().any(|&c| c > 0.0);
        !(self.intensity > 0.0 && self.intensity.is_finite() && lit)
    }

    /// The three colours as the renderer uses them — zenith, horizon, ground,
    /// each times the intensity — with anything negative or non-finite
    /// replaced by 0. All zero when [`Environment::is_none`].
    #[must_use]
    pub fn colors(&self) -> [[f32; 3]; 3] {
        let scale = if self.intensity > 0.0 && self.intensity.is_finite() {
            self.intensity
        } else {
            0.0
        };
        [self.zenith, self.horizon, self.ground].map(|color| {
            color.map(|c| {
                let v = c * scale;
                if v.is_finite() && v > 0.0 {
                    v
                } else {
                    0.0
                }
            })
        })
    }

    /// `horizon + (zenith - horizon) * up + (ground - horizon) * down`.
    fn blend(&self, up: f32, down: f32) -> [f32; 3] {
        let [zenith, horizon, ground] = self.colors();
        [0, 1, 2]
            .map(|c| horizon[c] + (zenith[c] - horizon[c]) * up + (ground[c] - horizon[c]) * down)
    }

    /// The radiance arriving from `direction` (which need not be normalised;
    /// a zero vector reads as the horizon): the horizon colour at elevation
    /// 0, blending linearly in the sine of the elevation to the zenith colour
    /// straight up and to the ground colour straight down.
    #[must_use]
    pub fn radiance(&self, direction: [f32; 3]) -> [f32; 3] {
        let y = unit_y(direction);
        self.blend(y.max(0.0), (-y).max(0.0))
    }

    /// The ambient light on a surface facing `normal`: the gradient averaged
    /// over the hemisphere the surface sees, cosine-weighted (irradiance over
    /// pi; exact for this gradient). A matte surface of base colour `c` shows
    /// `c * ambient`.
    #[must_use]
    pub fn ambient(&self, normal: [f32; 3]) -> [f32; 3] {
        let y = unit_y(normal);
        self.blend(hemisphere(y), hemisphere(-y))
    }

    /// The gradient as a surface of `roughness` reflects it along the mirror
    /// direction `reflected`: [`Environment::radiance`] for a perfect mirror,
    /// widening to the hemisphere average of [`Environment::ambient`] for a
    /// fully rough one (the blend weight is the GGX alpha, roughness squared).
    #[must_use]
    pub fn reflection(&self, reflected: [f32; 3], roughness: f32) -> [f32; 3] {
        let y = unit_y(reflected);
        let alpha = (roughness * roughness).clamp(0.0, 1.0);
        let mix = |sharp: f32, soft: f32| sharp + (soft - sharp) * alpha;
        self.blend(
            mix(y.max(0.0), hemisphere(y)),
            mix((-y).max(0.0), hemisphere(-y)),
        )
    }
}

/// The cosine-weighted average, over the hemisphere around a unit normal
/// whose y component is `y`, of `max(direction.y, 0)`: how much of the
/// gradient's upward half a surface tilted that way receives. 2/3 facing
/// straight up, 2/(3 pi) facing the horizon, 0 facing straight down.
///
/// (The integral of two clamped cosines whose axes are an angle `b` apart is
/// `2/3 ((pi - b) cos b + sin b)`; divided by pi for the cosine weighting.)
fn hemisphere(y: f32) -> f32 {
    let y = y.clamp(-1.0, 1.0);
    let b = y.acos();
    2.0 / (3.0 * std::f32::consts::PI) * ((std::f32::consts::PI - b) * y + (1.0 - y * y).sqrt())
}

/// The y component of `v` normalised (0 for a zero or non-finite vector).
fn unit_y(v: [f32; 3]) -> f32 {
    let length = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if length > 0.0 && length.is_finite() {
        (v[1] / length).clamp(-1.0, 1.0)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SKY: Environment =
        Environment::gradient([0.1, 0.2, 0.9], [0.8, 0.8, 0.8], [0.3, 0.2, 0.1]);

    fn close(a: [f32; 3], b: [f32; 3]) -> bool {
        (0..3).all(|c| (a[c] - b[c]).abs() < 1e-5)
    }

    #[test]
    fn the_default_is_no_environment() {
        assert_eq!(Environment::default(), Environment::NONE);
        assert!(Environment::default().is_none());
        assert!(!Environment::default().background);
        assert_eq!(Environment::NONE.colors(), [[0.0; 3]; 3]);
        // Colours without intensity, intensity without colours, and nonsense
        // are all "none"; the presets are not.
        assert!(SKY.with_intensity(0.0).is_none());
        assert!(SKY.with_intensity(-1.0).is_none());
        assert!(SKY.with_intensity(f32::NAN).is_none());
        assert!(SKY.with_intensity(f32::INFINITY).is_none());
        assert!(Environment::gradient([0.0; 3], [0.0; 3], [0.0; 3]).is_none());
        assert!(Environment::NONE.with_background(true).is_none());
        for preset in [SKY, Environment::sky(), Environment::studio()] {
            assert!(!preset.is_none());
            assert!(!preset.background && preset.with_background(true).background);
        }
        // Negative and non-finite channels are dropped, the rest scaled.
        let odd = Environment::gradient([-1.0, f32::NAN, 0.5], [1.0; 3], [f32::INFINITY, 0.0, 2.0])
            .with_intensity(2.0);
        assert_eq!(odd.colors(), [[0.0, 0.0, 1.0], [2.0; 3], [0.0, 0.0, 4.0]]);
        assert!(odd.radiance([0.3, 0.4, 0.5]).iter().all(|c| c.is_finite()));
    }

    #[test]
    fn radiance_is_the_gradient_by_elevation() {
        assert!(close(SKY.radiance([0.0, 1.0, 0.0]), SKY.zenith));
        assert!(close(SKY.radiance([0.0, -3.0, 0.0]), SKY.ground));
        assert!(close(SKY.radiance([1.0, 0.0, 0.0]), SKY.horizon));
        assert!(close(SKY.radiance([0.0, 0.0, -7.0]), SKY.horizon));
        assert!(close(SKY.radiance([0.0; 3]), SKY.horizon));
        // 30 degrees up: sine 0.5, half way from horizon to zenith.
        let half_up = SKY.radiance([3f32.sqrt(), 1.0, 0.0]);
        assert!(close(half_up, [0.45, 0.5, 0.85]), "{half_up:?}");
        // The intensity scales everything.
        let bright = SKY.with_intensity(3.0).radiance([0.0, 1.0, 0.0]);
        assert!(close(bright, [0.3, 0.6, 2.7]));
    }

    /// The ambient term is the cosine-weighted hemisphere average: a uniform
    /// environment gives its own colour back whatever the normal; a surface
    /// facing up sees two thirds zenith and one third horizon; and a surface
    /// facing down gets no zenith light at all.
    #[test]
    fn ambient_is_the_hemisphere_average() {
        let grey = Environment::gradient([0.4; 3], [0.4; 3], [0.4; 3]);
        for normal in [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.3, -0.8, 0.5]] {
            assert!(close(grey.ambient(normal), [0.4; 3]));
            assert!(close(grey.reflection(normal, 0.6), [0.4; 3]));
        }
        let only_zenith = Environment::gradient([1.0; 3], [0.0; 3], [0.0; 3]);
        assert!(close(only_zenith.ambient([0.0, 1.0, 0.0]), [2.0 / 3.0; 3]));
        assert!(close(only_zenith.ambient([0.0, -1.0, 0.0]), [0.0; 3]));
        let sideways = 2.0 / (3.0 * std::f32::consts::PI);
        assert!(close(only_zenith.ambient([1.0, 0.0, 0.0]), [sideways; 3]));
        // Compare with a numerical integral over the hemisphere of the normal.
        for normal in [
            [0.0f32, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.6, 0.8, 0.0],
            [0.0, -0.6, 0.8],
        ] {
            let (mut sum, mut weight) = ([0.0f64; 3], 0.0f64);
            let steps = 200;
            for i in 0..steps {
                for j in 0..2 * steps {
                    let theta = (f64::from(i) + 0.5) / f64::from(steps) * std::f64::consts::PI;
                    let phi = (f64::from(j) + 0.5) / f64::from(steps) * std::f64::consts::PI;
                    let d = [
                        theta.sin() * phi.cos(),
                        theta.cos(),
                        theta.sin() * phi.sin(),
                    ];
                    let cosine: f64 = (0..3).map(|k| d[k] * f64::from(normal[k])).sum();
                    if cosine <= 0.0 {
                        continue;
                    }
                    let w = cosine * theta.sin();
                    let l = SKY.radiance(d.map(|c| c as f32));
                    for c in 0..3 {
                        sum[c] += f64::from(l[c]) * w;
                    }
                    weight += w;
                }
            }
            let exact = sum.map(|s| (s / weight) as f32);
            let ours = SKY.ambient(normal);
            assert!(
                (0..3).all(|c| (exact[c] - ours[c]).abs() < 2e-3),
                "normal {normal:?}: integral {exact:?}, formula {ours:?}"
            );
        }
    }

    #[test]
    fn reflection_runs_from_the_mirror_lookup_to_the_ambient_average() {
        for direction in [[0.0, 1.0, 0.0], [0.5, 0.5, 0.7], [0.2, -0.9, 0.1]] {
            assert!(close(
                SKY.reflection(direction, 0.0),
                SKY.radiance(direction)
            ));
            assert!(close(
                SKY.reflection(direction, 1.0),
                SKY.ambient(direction)
            ));
            // In between it is between the two, channel by channel.
            let (sharp, soft, mid) = (
                SKY.radiance(direction),
                SKY.ambient(direction),
                SKY.reflection(direction, 0.6),
            );
            for c in 0..3 {
                let (lo, hi) = (sharp[c].min(soft[c]), sharp[c].max(soft[c]));
                assert!(mid[c] >= lo - 1e-6 && mid[c] <= hi + 1e-6);
            }
        }
    }
}
