// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 Guangzhou Songzi Intelligent Technology Services Co., Ltd.

/* Print glare. mobile.1 draws the same values keyed on the pixel index as the reference,
 * so one code path serves both variants.
 */
#include <cmath>

#include "fx/fx_internal.hpp"
#include "fx/glare.hpp"
#include "util/numeric.hpp"

namespace spk::fx {

bool glare_draws_seed(const GlareParams& g) { return g.active && g.percent * g.amount > 0.0; }

namespace {
double glare_sigma(const GlareParams& g, double rho) { return g.blur_px_full * rho * std::sqrt(std::max(g.amount, 0.0)); }
}  // namespace

/* glare(ctx, xyz, g, print_seed)
 *   P := percent amount
 *   F := lognormal(mean P, std roughness P), stream 200, one value per pixel on all channels
 *   sigma := blur_px_full rho_tier sqrt(amount); F := G_sigma(F)
 *   xyz_c := xyz_c + (F / 100) tint_c
 */
void glare(const core::RenderContext& ctx, gpu::Plane& xyz, const GlareParams& g, uint32_t print_seed) {
    if (!glare_draws_seed(g)) return;
    const double pp = g.percent * g.amount;
    SpkFxLognormalArgs la{};
    la.w = xyz.w;
    la.h = xyz.h;
    la.seed = print_seed;
    la.stream_base = SPK_STREAM_GLARE;
    la.same_all = 1;
    la.mean = num::f32(pp);
    la.stddev = num::f32(g.roughness * pp);
    gpu::Batch bt = ctx.gpu.batch();
    gpu::Plane field = ctx.scratch(xyz.w, xyz.h);
    bt.dispatch_2d(ctx.kernel("spk_fx_grain_lognormal"), xyz.w, xyz.h, {gpu::Arg::plane(field), gpu::Arg::value(la)});
    if (g.blur_px_full > 0.0) {
        const double s = glare_sigma(g, ctx.rho_tier);
        detail::encode_blur(ctx, bt, field, field, Vec3{s, s, s});
    }
    SpkFxGlareAddArgs aa{};
    aa.n = uint32_t(xyz.pixels());
    for (size_t c = 0; c < 3; ++c) aa.tint[c] = num::f32(g.tint_xyz[c]);
    bt.dispatch_1d(ctx.kernel("spk_fx_glare_add"), aa.n, {gpu::Arg::plane(xyz), gpu::Arg::plane(field), gpu::Arg::value(aa)});
    bt.commit_and_wait();
}

}  // namespace spk::fx
