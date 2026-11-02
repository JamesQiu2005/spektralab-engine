// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 Guangzhou Songzi Intelligent Technology Services Co., Ltd.

/* Film grain.
 *
 * sub-layer Poisson model (default) or simple model
 * then dye-cloud blur, micro-structure lognormal field, final 0.65 px blur, strength mix
 * every draw keyed on (seed, pixel index, stream)
 */
#pragma once

#include <array>
#include <memory>

#include "core/render.hpp"
#include "res/profile.hpp"

namespace spk::params {
class ParamSet;
}

namespace spk::fx {

// D_min,c, the grain model's fog density (fixed defaults). The print side's black reference is d^K_c = -kGrainDminDensity[c].
inline constexpr std::array<double, 3> kGrainDminDensity{0.03, 0.03, 0.03};

/* GrainSetup: per-channel and per-sub-layer constants and the kernels' tables; float64, narrowed once
 *   built from density_curves_layers, the normalized curves and the wire fields
 */
struct GrainSetup;

/* build_grain_setup: throws SPK_ERR_USER when grain and sub-layers are on and the profile has no
 *   density_curves_layers
 */
std::shared_ptr<const GrainSetup> build_grain_setup(const res::Profile& film, const params::ParamSet& p);
// The same from plain values (grain_active, grain_sublayers_active, grain_amount).
struct GrainOptions {
    bool active = true;
    bool sublayers = true;
    double amount = 1.0;
};
std::shared_ptr<const GrainSetup> build_grain_setup(const res::Profile& film, const GrainOptions& o);

/* grain(ctx, density, g, film_seed): in place, after the couplers; uses ctx.pitch_um
 *   inactive or amount 0: identity, no draws
 */
void grain(const core::RenderContext& ctx, gpu::Plane& density, const GrainSetup& g, uint32_t film_seed);

}  // namespace spk::fx
