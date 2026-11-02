// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 Guangzhou Songzi Intelligent Technology Services Co., Ltd.

/* fx internal encoding helpers: the Gaussian machinery encoded into a caller's Batch,
 * so a stage can chain several blurs before one commit.
 */
#pragma once

#include <array>
#include <span>

#include <vector>

#include "fx/gaussian.hpp"
#include "fx/grain.hpp"
#include "fx/halation.hpp"
#include "fx_shared.h"

namespace spk::fx::detail {

// The last pass of a blur may accumulate: out = acc + w v.
struct Fuse {
    uint32_t mode = SPK_FX_ACC_NONE;  // SPK_FX_ACC_NONE / _ZERO / _PLANE
    const gpu::Plane* acc = nullptr;  // for _PLANE (may be the destination plane)
    Vec3 weight{1.0, 1.0, 1.0};       // rounded to float when handed to the kernels
};

/* encode_blur: dst := G_sigma(in), or dst := acc + w G_sigma(in) with fuse
 *   dst may be in, and may be fuse.acc
 */
void encode_blur(const core::RenderContext& ctx, gpu::Batch& bt, const gpu::Plane& in, const gpu::Plane& dst,
                 const Vec3& sigma, const Fuse& fuse = {});
// out = mixture(in) accumulated in list order; out may be `in`.
void encode_mixture(const core::RenderContext& ctx, gpu::Batch& bt, const gpu::Plane& in, const gpu::Plane& out,
                    std::span<const MixtureComponent> comps);

// Pointwise kernels (float, per-channel scalars rounded double -> float).
void encode_lincomb(const core::RenderContext& ctx, gpu::Batch& bt, const Vec3& a, const gpu::Plane& x,
                    const Vec3& b, const gpu::Plane* y, const gpu::Plane& out);  // y == nullptr: y = 0
void encode_affine(const core::RenderContext& ctx, gpu::Batch& bt, const gpu::Plane& x, const Vec3& s,
                   const Vec3& t, const gpu::Plane& out);
void encode_mul(const core::RenderContext& ctx, gpu::Batch& bt, const gpu::Plane& x, const gpu::Plane& y,
                const gpu::Plane& out);
void encode_copy(const core::RenderContext& ctx, gpu::Batch& bt, const gpu::Plane& x, const gpu::Plane& out);

inline bool same(const gpu::Plane& a, const gpu::Plane& b) { return a.buf.mtl() == b.buf.mtl(); }

/* HalationPlan: at one pitch, the scatter and bounce or back-reflection mixtures and their mix scalars */
struct HalationPlan {
    bool scatter = false;
    std::vector<MixtureComponent> scatter_comps;
    Vec3 scatter_a{}, scatter_b{};  // raw <- a raw + b M(raw)
    bool bounce = false;
    std::vector<MixtureComponent> bounce_comps;
    Vec3 mix_a{}, mix_b{};  // raw <- a raw + b H(raw)
};
HalationPlan halation_plan(const HalationParams& h, double pitch_um);
// mobile.2: tent variance v(f), a component's factor, the grid factor F.
double tent_variance(int f);
int component_factor(const MixtureComponent& m);
int mobile_grid_factor(std::span<const MixtureComponent> comps);

/* GrainConstants: at one pitch; simple model uses row l = 0 only */
struct GrainConstants {
    bool active = false, sublayers = true;
    double count[3][3] = {}, sat_d[3][3] = {}, dmin_l[3][3] = {};
    double sigma_dye[3][3] = {};  // 0 where <= 0.4
    bool dye = false;
    double micro_sigma = 0.0, micro_blur = 0.0;
    bool micro = false;
    double final_blur = 0.0;  // 0: no final blur
};
GrainConstants grain_constants(const GrainSetup& g, double pitch_um);

}  // namespace spk::fx::detail
