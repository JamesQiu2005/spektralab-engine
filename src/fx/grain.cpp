// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 Guangzhou Songzi Intelligent Technology Services Co., Ltd.

/* Film grain: set-up in float64, the per-pixel draws on the GPU (fx_grain.metal), dye clouds,
 * micro-structure, the final blur and the strength mix.
 *
 * draws are keyed on the pixel index, so the reference and mobile.1 variants share this code
 */
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

#include "fx/fx_internal.hpp"
#include "fx/grain.hpp"
#include "params/params.hpp"
#include "util/error.hpp"
#include "util/numeric.hpp"

namespace spk::fx {

namespace {
constexpr double kParticleArea = 0.2;                      // A_p, um^2
constexpr double kChannelScale[3] = {1.6, 1.6, 3.2};       // s_c
constexpr double kSublayerScale[3] = {2.0, 1.0, 0.5};      // s_l (fast, mid, slow)
constexpr double kUniformity[3] = {0.97, 0.99, 0.97};      // u_c
constexpr double kFinalBlurPx = 0.65;                      // beta, plain pixels
constexpr double kDyeBlurUm = 1.0;                         // b_dye
constexpr double kMicroBlurUm = 0.2;                       // mu_b
constexpr double kMicroSigmaNm = 30.0;                     // mu_s
constexpr int kSimpleSublayers = 1;                        // N_sub
}  // namespace

struct GrainSetup {
    bool active = false;
    bool sublayers = true;
    double amount = 1.0;
    bool positive = false;
    // sub-layer model, the parts that do not depend on the pitch
    uint32_t K = 0;
    std::vector<float> axis, inv, layers;  // [k*3+c], [k*3+c], [(k*3+l)*3+c]
    double f[3][3] = {};                   // [l][c]
    double dmin[3][3] = {};
    double sat_d[3][3] = {};
    // simple model
    double simple_d[3] = {};
};

std::shared_ptr<const GrainSetup> build_grain_setup(const res::Profile& film, const params::ParamSet& p) {
    GrainOptions o;
    o.active = p.b(params::P::grain_active);
    o.sublayers = p.b(params::P::grain_sublayers_active);
    o.amount = p.f(params::P::grain_amount);
    return build_grain_setup(film, o);
}

/* build_grain_setup(film, o)
 *   simple model: D_c := dmax_c + Dmin_c
 *   sub-layer model, l = fast, mid, slow:
 *     Dmax[l][c] := nanmax of the layer curve
 *     f[l][c] := Dmax[l][c] / sum_l Dmax[l][c]
 *     dmin[l][c] := f[l][c] Dmin_c
 *     sat_d[l][c] := Dmax[l][c] + dmin[l][c]
 *   tables: axis := normalized curve, negated on a positive; inv := 1 / step, 0 for a zero step; layers
 */
std::shared_ptr<const GrainSetup> build_grain_setup(const res::Profile& film, const GrainOptions& o) {
    auto g = std::make_shared<GrainSetup>();
    g->active = o.active;
    g->sublayers = o.sublayers;
    g->amount = o.amount;
    g->positive = film.info.is_positive();
    const size_t K = film.curve_points();
    if (o.active && o.sublayers && !film.has_layers())
        fail_user("film profile '" + film.id + "' has no density_curves_layers, which grain sublayers need");
    // simple model constants: D_c = Dmax_film,c + D_min,c
    for (size_t c = 0; c < 3; ++c) g->simple_d[c] = film.dmax[c] + kGrainDminDensity[c];
    if (!film.has_layers()) return g;
    g->K = uint32_t(K);
    // 1-2: Dmax_raw, T_c, f, dmin, D
    double dmax_raw[3][3];
    for (size_t l = 0; l < 3; ++l)
        for (size_t c = 0; c < 3; ++c) dmax_raw[l][c] = num::nanmax_strided(film.density_curves_layers.data() + l * 3 + c, K, 9);
    for (size_t c = 0; c < 3; ++c) {
        const double T = (dmax_raw[0][c] + dmax_raw[1][c]) + dmax_raw[2][c];
        for (size_t l = 0; l < 3; ++l) {
            g->f[l][c] = dmax_raw[l][c] / T;
            g->dmin[l][c] = g->f[l][c] * kGrainDminDensity[c];
            g->sat_d[l][c] = dmax_raw[l][c] + g->dmin[l][c];
        }
    }
    // tables: the axis (negated for a positive film), reciprocal steps, layer curves
    std::vector<double> a(K * 3);
    for (size_t k = 0; k < K; ++k)
        for (size_t c = 0; c < 3; ++c) {
            const double v = film.normalized_curves(k, c);
            a[k * 3 + c] = g->positive ? -v : v;
        }
    g->axis.resize(K * 3);
    g->inv.assign(K * 3, 0.0f);
    for (size_t i = 0; i < K * 3; ++i) g->axis[i] = num::f32(a[i]);
    for (size_t k = 0; k + 1 < K; ++k)
        for (size_t c = 0; c < 3; ++c) {
            const double step = a[(k + 1) * 3 + c] - a[k * 3 + c];
            g->inv[k * 3 + c] = step == 0.0 ? 0.0f : num::f32(1.0 / step);
        }
    g->layers.resize(K * 9);
    for (size_t i = 0; i < K * 9; ++i) g->layers[i] = num::f32(film.density_curves_layers[i]);
    return g;
}

namespace detail {

/* grain_constants(g, p), p = pixel pitch in um
 *   sub-layer model:
 *     n[l][c] := p^2 f[l][c] / (0.2 s_c s_l)
 *     sigma_dye[l][c] := 1 * sqrt(sat_d[l][c] / n[l][c]); kept only above 0.4
 *     micro_sigma := 30 * 0.001 / p; on above 0.05
 *     micro_blur := 0.2 / p; final_blur := 0.65
 *   simple model:
 *     n[0][c] := p^2 / (0.2 s_c); sat_d[0][c] := simple D_c; final_blur := 0.65
 */
GrainConstants grain_constants(const GrainSetup& g, double p) {
    GrainConstants k;
    k.active = g.active && g.amount > 0.0;
    k.sublayers = g.sublayers;
    const double p2 = p * p;
    if (g.sublayers) {
        for (size_t l = 0; l < 3; ++l)
            for (size_t c = 0; c < 3; ++c) {
                const double n = (p2 * g.f[l][c]) / ((kParticleArea * kChannelScale[c]) * kSublayerScale[l]);
                k.count[l][c] = n;
                k.sat_d[l][c] = g.sat_d[l][c];
                k.dmin_l[l][c] = g.dmin[l][c];
                const double s = kDyeBlurUm > 0.0 ? kDyeBlurUm * std::sqrt(g.sat_d[l][c] / n) : 0.0;
                k.sigma_dye[l][c] = s > 0.4 ? s : 0.0;
                k.dye = k.dye || k.sigma_dye[l][c] > 0.0;
            }
        k.micro_sigma = kMicroSigmaNm * 0.001 / p;
        k.micro = k.micro_sigma > 0.05;
        k.micro_blur = kMicroBlurUm / p;
        k.final_blur = kFinalBlurPx > 0.0 ? kFinalBlurPx : 0.0;
    } else {
        for (size_t c = 0; c < 3; ++c) {
            double n = p2 / (kParticleArea * kChannelScale[c]);
            if (kSimpleSublayers > 1) n = n / kSimpleSublayers;
            k.count[0][c] = n;
            k.sat_d[0][c] = g.simple_d[c];
        }
        k.final_blur = kFinalBlurPx > 0.4 ? kFinalBlurPx : 0.0;
    }
    return k;
}

}  // namespace detail

/* grain(ctx, density, g, film_seed)
 *   inactive or amount <= 0: return
 *   simple model: G := spk_fx_grain_draw(SIMPLE)
 *   sub-layer model:
 *     no dye clouds: G := draw(SUM)
 *     dye clouds:    v_l := draw(PLANES); G := sum_l G_{sigma_dye[l]}(v_l)
 *     micro: phi := lognormal(mean 1, std micro_sigma), streams 100 + c; blurred above 0.4 px; G *= phi
 *     dye or micro: G -= Dmin
 *   final_blur > 0: G := blur(G)
 *   density := (1 - amount) density + amount G
 */
void grain(const core::RenderContext& ctx, gpu::Plane& density, const GrainSetup& g, uint32_t film_seed) {
    if (!g.active || !(g.amount > 0.0)) return;  // identity, no draws
    if (!(ctx.pitch_um > 0.0)) fail_internal("fx: grain without a pixel pitch");
    const detail::GrainConstants kc = detail::grain_constants(g, ctx.pitch_um);
    const uint32_t w = density.w, h = density.h;

    SpkFxGrainArgs a{};
    a.w = w;
    a.h = h;
    a.seed = film_seed;
    a.positive = g.positive ? 1u : 0u;
    a.n_sub = uint32_t(kSimpleSublayers);
    a.inv_n_sub = num::f32(1.0 / kSimpleSublayers);
    for (size_t c = 0; c < 3; ++c) {
        a.uniformity[c] = num::f32(kUniformity[c]);
        a.dmin_c[c] = num::f32(kGrainDminDensity[c]);
    }
    for (size_t l = 0; l < 3; ++l)
        for (size_t c = 0; c < 3; ++c) {
            a.dmin_l[l * 3 + c] = num::f32(kc.dmin_l[l][c]);
            a.sat_d[l * 3 + c] = num::f32(kc.sat_d[l][c]);
            a.count[l * 3 + c] = num::f32(kc.count[l][c]);
        }

    // tables (sub-layer model); a 1-float dummy otherwise
    gpu::Buffer axis, inv, layers;
    auto upload = [&](const std::vector<float>& v) {
        gpu::Buffer b = ctx.gpu.pool().acquire(std::max<size_t>(v.size(), 1) * sizeof(float));
        if (!v.empty()) std::memcpy(b.contents(), v.data(), v.size() * sizeof(float));
        return b;
    };
    if (g.sublayers) {
        a.k = g.K;
        axis = upload(g.axis);
        inv = upload(g.inv);
        layers = upload(g.layers);
    } else {
        axis = inv = layers = upload({});
    }

    gpu::Batch bt = ctx.gpu.batch();
    gpu::Plane G = ctx.scratch(w, h);
    const Vec3 shift{-kGrainDminDensity[0], -kGrainDminDensity[1], -kGrainDminDensity[2]};
    auto draw = [&](uint32_t mode, uint32_t do_shift, const gpu::Plane& o0, const gpu::Plane& o1, const gpu::Plane& o2) {
        a.mode = mode;
        a.shift = do_shift;
        bt.dispatch_2d(ctx.kernel("spk_fx_grain_draw"), w, h,
                       {gpu::Arg::plane(density), gpu::Arg::buffer(axis), gpu::Arg::buffer(inv), gpu::Arg::buffer(layers),
                        gpu::Arg::plane(o0), gpu::Arg::plane(o1), gpu::Arg::plane(o2), gpu::Arg::value(a)});
    };

    if (!g.sublayers) {
        draw(SPK_FX_GRAIN_SIMPLE, 0, G, G, G);  // G = sum * (1/N_sub) - D_min (one affine)
    } else {
        if (!kc.dye) {
            draw(SPK_FX_GRAIN_SUM, kc.micro ? 0u : 1u, G, G, G);
        } else {
            gpu::Plane v0 = ctx.scratch(w, h), v1 = ctx.scratch(w, h), v2 = ctx.scratch(w, h);
            draw(SPK_FX_GRAIN_PLANES, 0, v0, v1, v2);
            const gpu::Plane* vl[3] = {&v0, &v1, &v2};
            for (size_t l = 0; l < 3; ++l) {
                detail::Fuse f;
                f.mode = l == 0 ? SPK_FX_ACC_ZERO : SPK_FX_ACC_PLANE;
                f.acc = &G;
                f.weight = {1.0, 1.0, 1.0};
                const Vec3 s{kc.sigma_dye[l][0], kc.sigma_dye[l][1], kc.sigma_dye[l][2]};
                detail::encode_blur(ctx, bt, *vl[l], G, s, f);
            }
        }
        if (kc.micro) {
            gpu::Plane phi = ctx.scratch(w, h);
            SpkFxLognormalArgs la{};
            la.w = w;
            la.h = h;
            la.seed = film_seed;
            la.stream_base = SPK_STREAM_MICRO(0);
            la.same_all = 0;
            la.mean = 1.0f;
            la.stddev = num::f32(kc.micro_sigma);
            bt.dispatch_2d(ctx.kernel("spk_fx_grain_lognormal"), w, h, {gpu::Arg::plane(phi), gpu::Arg::value(la)});
            if (kc.micro_blur > 0.4) {
                const double s = kc.micro_blur;
                detail::encode_blur(ctx, bt, phi, phi, Vec3{s, s, s});
            }
            detail::encode_mul(ctx, bt, G, phi, G);
        }
        if (kc.dye || kc.micro) detail::encode_affine(ctx, bt, G, Vec3{1.0, 1.0, 1.0}, shift, G);
    }
    if (kc.final_blur > 0.0) {
        const double b = kc.final_blur;
        detail::encode_blur(ctx, bt, G, G, Vec3{b, b, b});
    }
    // strength mix
    if (g.amount == 1.0) {
        detail::encode_copy(ctx, bt, G, density);
    } else {
        const double k = g.amount;
        detail::encode_lincomb(ctx, bt, Vec3{1.0 - k, 1.0 - k, 1.0 - k}, density, Vec3{k, k, k}, &G, density);
    }
    bt.commit_and_wait();
}

}  // namespace spk::fx
