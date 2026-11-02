// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 Guangzhou Songzi Intelligent Technology Services Co., Ltd.

/* Highlight headroom and halation.
 *
 * scatter, then bounces or back-reflection; the mobile variant when ctx.variants.mobile_halation
 * HalationParams holds every quantity of the scatter and bounce models
 */
#pragma once

#include "core/render.hpp"
#include "fx/gaussian.hpp"
#include "res/profile.hpp"

namespace spk::params {
class ParamSet;
}

namespace spk::fx {

struct HalationParams {
    // wire
    bool active = true;
    double amount = 1.0;          // halation_amount: area multiplier, sigmas scale by sqrt(amount)
    double scatter_amount = 1.0;  // halation_scatter_amount
    double headroom_ev = 4.0;     // halation_headroom_ev (back-reflection only)
    // fixed defaults
    Vec3 scatter_core_um{2.2, 2.0, 1.6};
    Vec3 scatter_tail_um{9.3, 9.7, 9.1};
    Vec3 scatter_tail_weight{0.78, 0.65, 0.67};
    double scatter_scale = 1.0;
    double halation_scale = 1.0;
    int bounces = 3;
    double bounce_decay = 0.5;
    bool renormalise = true;
    double base_thickness_um = 120.0;
    // preset-resolved
    Vec3 strength{0.05, 0.015, 0.0};
    Vec3 first_sigma_um{65.0, 65.0, 65.0};
    bool back_reflection = false;
};

/* resolve_halation(film, p): wire values from p, then the preset; on every film build */
HalationParams resolve_halation(const res::ProfileInfo& film, const params::ParamSet& p);
// The same preset applied to `h` (wire values already set).
HalationParams resolve_halation_preset(HalationParams h, const res::ProfileInfo& film, bool antihalation_removed);

// Whether the highlight headroom runs: active, back_reflection, headroom_ev > 0, amount > 0.
bool headroom_applies(const HalationParams& h);
// clip = max over every element of `exposure`.
float headroom_clip(const core::RenderContext& ctx, const gpu::Plane& exposure);
/* highlight_headroom(ctx, exposure, h, clip): in place on linear exposure
 *   identity when clip is not finite and positive; gain in spk_fx_halation_headroom
 */
void highlight_headroom(const core::RenderContext& ctx, gpu::Plane& exposure, const HalationParams& h, float clip);
// In place on linear exposure (reference or mobile per ctx.variants); identity when inactive.
void halation(const core::RenderContext& ctx, gpu::Plane& exposure, const HalationParams& h);

}  // namespace spk::fx
