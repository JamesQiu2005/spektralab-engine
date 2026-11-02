// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 Guangzhou Songzi Intelligent Technology Services Co., Ltd.

/* Print glare, in XYZ after the scan and black/white levels. */
#pragma once

#include <array>

#include "core/render.hpp"

namespace spk::fx {

struct GlareParams {
    bool active = true;    // glare_active
    double amount = 1.0;   // glare_amount
    double percent = 0.03; // P
    double roughness = 0.7;
    double blur_px_full = 0.5;  // b_g at the full tier
    std::array<double, 3> tint_xyz{};  // viewing illuminant XYZ normalised to Y = 1
};

/* glare_draws_seed(g) = active and percent * amount > 0; regardless of scan_film */
bool glare_draws_seed(const GlareParams& g);
// In place on XYZ; uses ctx.rho_tier. Identity when !glare_draws_seed.
void glare(const core::RenderContext& ctx, gpu::Plane& xyz, const GlareParams& g, uint32_t print_seed);

}  // namespace spk::fx
