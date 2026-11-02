// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 Guangzhou Songzi Intelligent Technology Services Co., Ltd.

/* Highlight headroom and halation: preset, scatter, bounces or back-reflection, the reference mix and the
 * mobile.2 coarse-grid variant.
 */
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "fx/fx_internal.hpp"
#include "fx/halation.hpp"
#include "params/params.hpp"
#include "util/error.hpp"
#include "util/numeric.hpp"

namespace spk::fx {

using params::P;

HalationParams resolve_halation(const res::ProfileInfo& film, const params::ParamSet& p) {
    HalationParams h;
    h.active = p.b(P::halation_active);
    h.amount = p.f(P::halation_amount);
    h.scatter_amount = p.f(P::halation_scatter_amount);
    h.headroom_ev = p.f(P::halation_headroom_ev);
    return resolve_halation_preset(h, film, p.b(P::antihalation_removed));
}

/* resolve_halation_preset(h, film, antihalation_removed)
 *   support != "film": h unchanged
 *   tag := antihalation_removed ? "no" : film.antihalation
 *   strength := strong (0.015, 0.005, 0) | weak (0.08, 0.02, 0) | no (0.30, 0.03, 0.003)
 *   first sigma := cine 50 um | still 65 um
 *   back_reflection := tag == "no"
 *   an unknown tag or use: the default strength and sigma, no back-reflection
 */
HalationParams resolve_halation_preset(HalationParams h, const res::ProfileInfo& film, bool antihalation_removed) {
    if (film.support != "film") return h;
    const HalationParams defaults;
    h.strength = defaults.strength;
    h.first_sigma_um = defaults.first_sigma_um;
    h.back_reflection = false;
    const std::string tag = antihalation_removed ? std::string("no") : film.antihalation;
    Vec3 strength;
    if (tag == "strong") {
        strength = {0.015, 0.005, 0.0};
    } else if (tag == "weak") {
        strength = {0.08, 0.02, 0.0};
    } else if (tag == "no") {
        strength = {0.30, 0.03, 0.003};
    } else {
        return h;
    }
    if (film.use != "still" && film.use != "cine") return h;
    const double s = film.use == "cine" ? 50.0 : 65.0;
    h.first_sigma_um = {s, s, s};
    h.strength = strength;
    h.back_reflection = tag == "no";
    return h;
}

bool headroom_applies(const HalationParams& h) {
    return h.active && h.back_reflection && h.headroom_ev > 0.0 && h.amount > 0.0;
}

float headroom_clip(const core::RenderContext&, const gpu::Plane& exposure) {
    const float* d = exposure.data();
    const size_t n = exposure.floats();
    float m = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < n; ++i)
        if (d[i] > m) m = d[i];
    return m;
}

void highlight_headroom(const core::RenderContext& ctx, gpu::Plane& exposure, const HalationParams& h, float clip) {
    if (!headroom_applies(h)) return;
    if (!(clip > 0.0f) || !std::isfinite(clip)) return;
    SpkFxHeadroomArgs a{};
    a.n = uint32_t(exposure.pixels());
    a.clip = clip;
    a.headroom_ev = num::f32(h.headroom_ev);
    gpu::Batch bt = ctx.gpu.batch();
    bt.dispatch_1d(ctx.kernel("spk_fx_halation_headroom"), a.n, {gpu::Arg::plane(exposure), gpu::Arg::value(a)});
    bt.commit_and_wait();
}

/* ---- the plan: every mixture and scalar at one pitch ----
 * halation_plan(h, p), p = pixel pitch in um
 *   scatter, when scatter_amount > 0:
 *     M := (1 - w_t) G_core + exponential_surrogate(w_t, tail)
 *     core := 2.2, 2.0, 1.6 um; tail := 9.3, 9.7, 9.1 um; w_t := 0.78, 0.65, 0.67
 *     raw := (1 - amount) raw + amount M(raw)
 *   A_c := float32(strength_c) float32(halation_amount); rho := sqrt(halation_amount)
 *   bounces, not back-reflection, k = 1..3:
 *     weight_k := 0.5^(k-1) / sum_j 0.5^(j-1); sigma_k := first_sigma rho sqrt(k) / p
 *   back-reflection: four fixed (weight, sigma / d) rows; d := 120 um rho / p
 *   mix: raw := raw / (1 + A) + A H(raw) / (1 + A)
 */

namespace detail {

HalationPlan halation_plan(const HalationParams& h, double p) {
    HalationPlan pl;
    if (!h.active) return pl;
    // scatter
    Vec3 core{}, lam{};
    bool any_size = false;
    for (size_t c = 0; c < 3; ++c) {
        core[c] = h.scatter_core_um[c] * h.scatter_scale / p;
        lam[c] = h.scatter_tail_um[c] * h.scatter_scale / p;
        any_size = any_size || core[c] != 0.0 || lam[c] != 0.0;
    }
    if (h.scatter_amount > 0.0 && any_size) {
        pl.scatter = true;
        MixtureComponent c0;
        Vec3 lam1{};
        for (size_t c = 0; c < 3; ++c) {
            c0.weight[c] = 1.0 - h.scatter_tail_weight[c];
            c0.sigma[c] = std::max(core[c], 1e-6);
            lam1[c] = std::max(lam[c], 1e-6);
        }
        pl.scatter_comps.push_back(c0);
        for (const MixtureComponent& m : exponential_surrogate(h.scatter_tail_weight, lam1)) pl.scatter_comps.push_back(m);
        const double sa = h.scatter_amount;
        pl.scatter_a = {1.0 - sa, 1.0 - sa, 1.0 - sa};
        pl.scatter_b = {sa, sa, sa};
    }
    // strength (float product, widened) and radius multiplier
    Vec3 A{};
    bool any_a = false;
    const float famount = num::f32(h.amount);
    for (size_t c = 0; c < 3; ++c) {
        const float prod = num::f32(h.strength[c]) * famount;
        A[c] = double(prod);
        any_a = any_a || A[c] > 0.0;
    }
    const double rho = std::sqrt(std::max(h.amount, 0.0));
    if (!h.back_reflection) {
        Vec3 sh{};
        bool any_sigma = false;
        for (size_t c = 0; c < 3; ++c) {
            sh[c] = h.first_sigma_um[c] * h.halation_scale * rho / p;
            any_sigma = any_sigma || sh[c] > 0.0;
        }
        if (h.bounces >= 1 && any_a && any_sigma) {
            double total = 0.0;
            for (int j = 1; j <= h.bounces; ++j) total += std::pow(h.bounce_decay, double(j - 1));
            for (int k = 1; k <= h.bounces; ++k) {
                MixtureComponent m;
                const double wk = std::pow(h.bounce_decay, double(k - 1)) / total;
                for (size_t c = 0; c < 3; ++c) {
                    m.weight[c] = wk;
                    m.sigma[c] = std::max(sh[c] * std::sqrt(double(k)), 1e-6);
                }
                pl.bounce_comps.push_back(m);
            }
            pl.bounce = true;
        }
    } else {
        static constexpr double kS[4][3] = {{0.184586, 0.178469, 0.170237},
                                            {1.121207, 1.000078, 0.546805},
                                            {1.651048, 1.581389, 0.808738},
                                            {4.677567, 9.448921, 1.340598}};
        static constexpr double kW[4][3] = {{+0.041510, +0.106789, +0.202791},
                                            {-0.500000, -0.500000, +0.156578},
                                            {+1.034500, +1.384389, -0.500000},
                                            {+0.423990, +0.008822, +1.140631}};
        const double dp = h.base_thickness_um * h.halation_scale * rho / p;
        if (any_a && dp > 0.0) {
            for (int k = 0; k < 4; ++k) {
                MixtureComponent m;
                for (size_t c = 0; c < 3; ++c) {
                    m.weight[c] = kW[k][c];
                    m.sigma[c] = std::max(kS[k][c] * dp, 1e-6);
                }
                pl.bounce_comps.push_back(m);
            }
            pl.bounce = true;
        }
    }
    if (pl.bounce) {
        for (size_t c = 0; c < 3; ++c) {
            if (h.renormalise) {
                pl.mix_a[c] = 1.0 / (1.0 + A[c]);
                pl.mix_b[c] = A[c] / (1.0 + A[c]);
            } else {
                pl.mix_a[c] = 1.0;
                pl.mix_b[c] = A[c];
            }
        }
    }
    return pl;
}

/* ---- mobile.2: large components on a coarse grid ----
 * tent_variance(f): variance of the tent w(d) = max(0, 1 - |d| / f) sampled about (f - 1) / 2
 * component_factor(m): s := smallest sigma with a non-zero weight
 *   first f in 8, 4, 2 with sqrt((s^2 - 2 tent_variance(f)) / f^2) >= 3; else 1
 */

double tent_variance(int f) {
    const double c = (f - 1) / 2.0;
    double sw = 0.0, swd = 0.0;
    for (int j = -2 * f; j <= 3 * f; ++j) {
        const double d = double(j) - c;
        const double w = std::max(0.0, 1.0 - std::fabs(d) / f);
        if (w <= 0.0) continue;
        sw += w;
        swd += w * d * d;
    }
    return swd / sw;
}

int component_factor(const MixtureComponent& m) {
    double s = std::numeric_limits<double>::infinity();
    bool any = false;
    for (size_t c = 0; c < 3; ++c) {
        if (m.weight[c] != 0.0) {
            s = std::min(s, m.sigma[c]);
            any = true;
        }
    }
    if (!any) return 1;
    for (int f : {8, 4, 2}) {
        const double v = std::sqrt((s * s - 2.0 * tent_variance(f)) / double(f * f));
        if (v >= 3.0) return f;
    }
    return 1;
}

/* mobile_grid_factor(comps): smallest factor > 1 among the components; 1: no coarse component */
int mobile_grid_factor(std::span<const MixtureComponent> comps) {
    int F = 0;
    for (const MixtureComponent& m : comps) {
        const int f = component_factor(m);
        if (f > 1) F = F == 0 ? f : std::min(F, f);
    }
    return F == 0 ? 1 : F;
}

}  // namespace detail

namespace {

using detail::HalationPlan;

/* mobile_bounce(ctx, bt, raw, pl)
 *   fine := components of factor 1; coarse := the others
 *   F := mobile_grid_factor; v := tent_variance(F)
 *   coarse sigma := sqrt(sigma^2 - 2 v) / F, or 1e-6 when that is not positive
 *   acc_fine := mixture(raw, fine)
 *   down := tent splat of raw onto the ceil(w / F) x ceil(h / F) grid
 *   cmix := mixture(down, coarse)
 *   raw := a raw + b (bilinear up(cmix) + acc_fine)
 */
void mobile_bounce(const core::RenderContext& ctx, gpu::Batch& bt, gpu::Plane& raw, const HalationPlan& pl) {
    std::vector<MixtureComponent> fine, coarse;
    for (const MixtureComponent& m : pl.bounce_comps) {
        (detail::component_factor(m) > 1 ? coarse : fine).push_back(m);
    }
    const int F = detail::mobile_grid_factor(pl.bounce_comps);
    const double v = detail::tent_variance(F);
    for (MixtureComponent& m : coarse) {
        for (size_t c = 0; c < 3; ++c) {
            const double d = m.sigma[c] * m.sigma[c] - 2.0 * v;
            m.sigma[c] = d > 0.0 ? std::sqrt(d) / F : 1e-6;
        }
    }
    gpu::Plane acc_fine;
    if (!fine.empty()) {
        acc_fine = ctx.scratch(raw.w, raw.h);
        detail::encode_mixture(ctx, bt, raw, acc_fine, fine);
    }
    SpkFxTentArgs t{};
    t.w = raw.w;
    t.h = raw.h;
    t.f = uint32_t(F);
    t.cw = (raw.w + t.f - 1) / t.f;
    t.ch = (raw.h + t.f - 1) / t.f;
    t.has_fine = acc_fine ? 1u : 0u;
    for (size_t c = 0; c < 3; ++c) {
        t.a[c] = num::f32(pl.mix_a[c]);
        t.b[c] = num::f32(pl.mix_b[c]);
    }
    gpu::Plane down = ctx.scratch(t.cw, t.ch);
    bt.dispatch_2d(ctx.kernel("spk_fx_halation_tent_down"), t.cw, t.ch,
                   {gpu::Arg::plane(raw), gpu::Arg::plane(down), gpu::Arg::value(t)});
    gpu::Plane cmix = ctx.scratch(t.cw, t.ch);
    detail::encode_mixture(ctx, bt, down, cmix, coarse);
    gpu::Plane out = ctx.scratch(raw.w, raw.h);
    bt.dispatch_2d(ctx.kernel("spk_fx_halation_up_mix"), raw.w, raw.h,
                   {gpu::Arg::plane(raw), gpu::Arg::plane(cmix), gpu::Arg::plane(acc_fine ? acc_fine : raw),
                    gpu::Arg::plane(out), gpu::Arg::value(t)});
    detail::encode_copy(ctx, bt, out, raw);
}

}  // namespace

/* halation(ctx, exposure, h)
 *   pl := halation_plan(h, pitch)
 *   scatter: exposure := a exposure + b M(exposure)
 *   bounce:  mobile with a coarse component: mobile_bounce
 *            else exposure := mix_a exposure + mix_b H(exposure)
 */
void halation(const core::RenderContext& ctx, gpu::Plane& exposure, const HalationParams& h) {
    if (!h.active) return;
    if (!(ctx.pitch_um > 0.0)) fail_internal("fx: halation without a pixel pitch");
    const HalationPlan pl = detail::halation_plan(h, ctx.pitch_um);
    if (!pl.scatter && !pl.bounce) return;
    gpu::Batch bt = ctx.gpu.batch();
    if (pl.scatter) {
        gpu::Plane m = ctx.scratch(exposure.w, exposure.h);
        detail::encode_mixture(ctx, bt, exposure, m, pl.scatter_comps);
        detail::encode_lincomb(ctx, bt, pl.scatter_a, exposure, pl.scatter_b, &m, exposure);
    }
    if (pl.bounce) {
        const bool mobile = ctx.variants.mobile_halation && detail::mobile_grid_factor(pl.bounce_comps) > 1;
        if (mobile) {
            mobile_bounce(ctx, bt, exposure, pl);
        } else {
            gpu::Plane H = ctx.scratch(exposure.w, exposure.h);
            detail::encode_mixture(ctx, bt, exposure, H, pl.bounce_comps);
            detail::encode_lincomb(ctx, bt, pl.mix_a, exposure, pl.mix_b, &H, exposure);
        }
    }
    bt.commit_and_wait();
}

}  // namespace spk::fx
