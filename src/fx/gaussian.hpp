// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 Guangzhou Songzi Intelligent Technology Services Co., Ltd.

/* The separable Gaussian machinery shared by every blur and the camera lens blur.
 *
 * per channel:
 *   sigma <= 0       identity
 *   0 < sigma < 3    FIR: truncate 3, vertical then horizontal, half-sample reflect, float accumulation,
 *                    taps -R..R
 *   sigma >= 3       Young-van Vliet recursive filter: horizontal then vertical, replicated-edge start,
 *                    double or double-float state, float storage of x, u, y
 *   a mixed call runs its FIR channels first, then its recursive channels
 * planes of equal size; out may be in
 *
 * Published: Young and van Vliet, Signal Processing 44, 1995.
 */
#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <vector>

#include "core/render.hpp"
#include "gpu/gpu.hpp"

namespace spk::fx {

using Vec3 = std::array<double, 3>;

inline constexpr double kFirTruncate = 3.0;
inline constexpr double kIirCrossover = 3.0;

enum class Branch : uint8_t { Identity, Fir, Iir };
Branch branch(double sigma);
int fir_radius(double sigma);  // floor(3 sigma + 0.5)
// taps, double then rounded to float; {1} for sigma <= 0.
std::vector<float> fir_taps(double sigma);

struct IirCoefficients {
    double B, B1, B2, B3;
};
IirCoefficients iir_coefficients(double sigma);

// A per-channel Gaussian (any branch mix).
void gaussian(const core::RenderContext& ctx, const gpu::Plane& in, gpu::Plane& out, const Vec3& sigma);

/* MixtureComponent: one term of M(x) = sum_k w_k G_{sigma_k}(x), per channel */
struct MixtureComponent {
    Vec3 weight;
    Vec3 sigma;
};
/* gaussian_mixture: accumulated in list order, the multiply-add fused into each component's last pass
 *   every component is evaluated, even at weight 0
 */
void gaussian_mixture(const core::RenderContext& ctx, const gpu::Plane& in, gpu::Plane& out,
                      std::span<const MixtureComponent> components);

/* exponential_surrogate(weight, lambda): an exponential PSF as three Gaussians
 *   component i: weight_c a_i, sigma r_i lambda_c
 *   a = (0.1633, 0.6496, 0.1870), summing to 0.9999; r = (0.5360, 1.5236, 2.7684)
 */
std::array<MixtureComponent, 3> exponential_surrogate(const Vec3& weight, const Vec3& lambda_px);

// Pointwise helpers (float, scalars rounded double -> float): out = a x + b y; out = x s + t.
void linear_combination(const core::RenderContext& ctx, const Vec3& a, const gpu::Plane& x, const Vec3& b,
                        const gpu::Plane& y, gpu::Plane& out);
void affine(const core::RenderContext& ctx, const gpu::Plane& x, const Vec3& s, const Vec3& t, gpu::Plane& out);

// Camera lens blur: sigma = lens_blur_um / ctx.pitch_um on all channels,
// in place; identity when sigma <= 0.
void lens_blur(const core::RenderContext& ctx, gpu::Plane& exposure, double lens_blur_um);

// CPU copy of gaussian() with the same operation order, for unit tests.
namespace cpu {
void gaussian(const float* in, float* out, uint32_t w, uint32_t h, const Vec3& sigma);
}

}  // namespace spk::fx
