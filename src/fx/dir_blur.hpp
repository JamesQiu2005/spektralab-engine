// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 Guangzhou Songzi Intelligent Technology Services Co., Ltd.

/* DIR-coupler diffusion blur, on the coupler correction plane (log10-exposure units).
 *
 * M := (1 - w_t) G(core_um / p) + exponential_surrogate(w_t, tail_um / p), w_t = 0.06
 * always the reference implementation
 */
#pragma once

#include "core/render.hpp"

namespace spk::fx {

struct DirDiffusion {
    double core_um = 20.0;
    double tail_um = 200.0;
    double tail_weight = 0.06;
};

// In place; uses ctx.pitch_um. Identity when the core size is 0.
void dir_diffusion_blur(const core::RenderContext& ctx, gpu::Plane& correction, const DirDiffusion& d = {});

}  // namespace spk::fx
