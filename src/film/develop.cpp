// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 Guangzhou Songzi Intelligent Technology Services Co., Ltd.

/* develop(): tier RGB plane -> developed negative density plane.
 *
 * each stage encodes its own batch and waits, so whole-frame statistics read the shared plane on the CPU
 */
#include <cmath>
#include <cstring>

#include "film/film_internal.hpp"
#include "film_shared.h"
#include "fx/dir_blur.hpp"
#include "fx/gaussian.hpp"
#include "util/numeric.hpp"

namespace spk::film {

using params::P;

namespace {

bool geometry_identity(const params::ParamSet& p) {
    const int64_t n = p.i(P::geometry_quarter_turns);
    const int64_t q = ((n % 4) + 4) % 4;
    return p.f(P::geometry_crop_x) == 0.0 && p.f(P::geometry_crop_y) == 0.0 && p.f(P::geometry_crop_w) == 1.0 &&
           p.f(P::geometry_crop_h) == 1.0 && p.f(P::geometry_rotation_deg) == 0.0 && q == 0 &&
           !p.b(P::geometry_flip_h) && !p.b(P::geometry_flip_v);
}

}  // namespace

/* run_boost(ctx, plane, beta_ev, M): lift highlights above x0, M = frame maximum
 *   M == 0: scale by 0
 *   x0 := min(0.184 * 2^4, M); x0 == M: identity
 *   a := 28^0.7; q := a (1 - x0 / M); den := exp(q) - q - 1; den <= 0: SPK_ERR_GPU
 *   k := (2^beta_ev - 1) / den
 *   v > x0: v := v + k M (exp(z) - z - 1), z = a (v - x0) / M
 */
void run_boost(const core::RenderContext& ctx, gpu::Plane& plane, double beta_ev, float M) {
    if (M == 0.0f) {
        run_scale(ctx, plane, 0.0f);
        return;
    }
    const double Md = double(M);
    const double x0 = std::min(std::max(0.184 * std::pow(2.0, 4.0), 0.0), Md);
    if (x0 == Md) return;
    const double a = std::pow(28.0, 1.0 - 0.3);
    const double q = a * (1.0 - x0 / Md);
    const double den = std::exp(q) - q - 1.0;
    if (!(den > 0.0)) fail_gpu("highlight boost: the frame maximum leaves no range to lift (den <= 0)");
    const double k = (std::pow(2.0, beta_ev) - 1.0) / den;
    SpkFilmBoostArgs args{};
    args.n = uint32_t(plane.floats());
    args.x0 = num::f32(x0);
    args.r = num::f32(1.0 / Md);
    args.a = num::f32(a);
    args.k = num::f32(k * Md);
    gpu::Batch b = ctx.gpu.batch();
    b.dispatch_1d(ctx.kernel("spk_film_boost"), args.n, {gpu::Arg::plane(plane), gpu::Arg::value(args)});
    b.commit_and_wait();
}

uint32_t node_count(const FilmSetup& setup) {
    const FilmSetup::Impl& s = setup.impl();
    const params::ParamSet& b = s.params;
    uint32_t n = 4;  // upsample, exposure compensation, log and curves always run
    if (s.decode) ++n;
    if (!geometry_identity(b)) ++n;
    if (b.b(P::auto_exposure)) ++n;
    if (b.f(P::halation_boost_ev) != 0.0) ++n;
    if (fx::headroom_applies(s.halation)) ++n;
    if (b.f(P::lens_blur_um) > 0.0) ++n;
    if (s.halation.active) ++n;
    if (s.couplers) ++n;
    if (s.grain) ++n;
    return n;
}

/* develop(ctx, setup, args)
 *   plane := tier source, borrowed; a stage that works in place copies it first
 *   if decode:                 plane := run_decode
 *   if geometry not identity:  plane := run_geometry
 *   if auto_exposure and ev:   plane *= 2^ev             -- warm-up has no ev: gain skipped
 *   plane := spk_film_upsample(plane, A, table)          -- RGB -> H
 *   plane *= 2^exposure_compensation_ev
 *   if boost_ev != 0:          run_boost(plane, boost_ev, max(plane))
 *   if headroom applies:       highlight_headroom(plane, max(plane))
 *   if lens_blur_um > 0:       lens_blur(plane)
 *   if halation active:        halation(plane)
 *   refuse positive + scan_film + scanner correction: SPK_ERR_GPU
 *   L := log10(max(plane, 0) + 1e-10)
 *   D := F(L; x, inv, y)
 *   if couplers:
 *     chi := M s(D)            s(D) = D, or dmax - D on a positive
 *     chi := dir_diffusion_blur(chi)
 *     D := F(L - chi; x, inv, y0)
 *   if grain:                  grain(D, film_seed)
 *   return a persistent copy of D
 */
gpu::Plane develop(core::RenderContext& ctx, const FilmSetup& setup, const DevelopArgs& args) {
    const FilmSetup::Impl& s = setup.impl();
    const params::ParamSet& b = s.params;
    const gpu::Plane& src = args.tier_source;
    if (!src || src.w == 0 || src.h == 0) fail_internal("film::develop: empty tier source");
    if (!(ctx.pitch_um > 0.0)) ctx.pitch_um = pitch_um(b.f(P::film_format_mm), std::max(src.w, src.h));

    gpu::Plane plane = src;
    bool owned = false;
    auto make_owned = [&] {
        if (owned) return;
        gpu::Plane copy = ctx.scratch(plane.w, plane.h);
        std::memcpy(copy.data(), plane.data(), plane.bytes());
        plane = copy;
        owned = true;
    };

    if (s.decode) {
        ctx.node("preprocess.decode");
        plane = run_decode(ctx, plane, s.input_space.cctf);
        owned = true;
    }

    if (!geometry_identity(b)) {
        ctx.node("preprocess.geometry");
        plane = run_geometry(ctx, plane, plan_geometry(b, plane.w, plane.h));
        owned = true;
    }

    if (b.b(P::auto_exposure)) {
        ctx.node("filming.auto_exposure");
        if (args.auto_exposure_ev) {
            const float g = num::f32(std::pow(2.0, *args.auto_exposure_ev));
            make_owned();
            run_scale(ctx, plane, g);
        }
    }

    {
        ctx.node("filming.upsample");
        gpu::Plane h = ctx.scratch(plane.w, plane.h);
        SpkFilmUpsampleArgs a{};
        a.n = uint32_t(plane.pixels());
        a.L = s.table->size;
        for (size_t i = 0; i < 9; ++i) a.a[i] = s.input_matrix_f32[i];
        gpu::Batch bt = ctx.gpu.batch();
        bt.dispatch_1d(ctx.kernel("spk_film_upsample"), a.n,
                       {gpu::Arg::plane(plane), gpu::Arg::plane(h), gpu::Arg::buffer(s.table->f32), gpu::Arg::value(a)});
        bt.commit_and_wait();
        plane = h;
        owned = true;
    }

    {
        ctx.node("filming.exposure_compensation");
        const float g = num::f32(std::pow(2.0, b.f(P::exposure_compensation_ev)));
        if (g != 1.0f) run_scale(ctx, plane, g);
    }

    if (b.f(P::halation_boost_ev) != 0.0) {
        ctx.node("filming.boost");
        run_boost(ctx, plane, b.f(P::halation_boost_ev), plane_max(plane));
    }

    if (fx::headroom_applies(s.halation)) {
        ctx.node("filming.headroom");
        fx::highlight_headroom(ctx, plane, s.halation, fx::headroom_clip(ctx, plane));
    }

    if (b.f(P::lens_blur_um) > 0.0) {
        ctx.node("filming.lens_blur");
        fx::lens_blur(ctx, plane, b.f(P::lens_blur_um));
    }

    if (s.halation.active) {
        ctx.node("filming.halation");
        fx::halation(ctx, plane, s.halation);
    }

    {
        ctx.node("filming.log");
        if (s.positive && b.b(P::scan_film) && (b.b(P::scanner_black_correction) || b.b(P::scanner_white_correction)))
            fail_gpu("black/white correction on a directly scanned positive film is not implemented by the native engine");
        SpkFilmLogArgs a{uint32_t(plane.floats())};
        gpu::Batch bt = ctx.gpu.batch();
        bt.dispatch_1d(ctx.kernel("spk_film_log"), a.n, {gpu::Arg::plane(plane), gpu::Arg::value(a)});
        bt.commit_and_wait();
    }

    gpu::Plane log_exposure;
    {
        ctx.node("filming.curves");
        gpu::Plane out = s.couplers ? ctx.scratch(plane.w, plane.h) : plane;
        SpkFilmCurvesArgs a{uint32_t(plane.pixels()), s.curves.K, 0u};
        gpu::Batch bt = ctx.gpu.batch();
        bt.dispatch_1d(ctx.kernel("spk_film_curves"), a.n,
                       {gpu::Arg::plane(plane), gpu::Arg::plane(out), gpu::Arg::buffer(s.curve_buffer), gpu::Arg::value(a)});
        bt.commit_and_wait();
        if (s.couplers) log_exposure = plane;
        plane = out;
    }

    if (s.couplers) {
        ctx.node("filming.couplers");
        gpu::Plane chi = plane;  // the first development, turned into the correction in place
        SpkFilmCouplerArgs a{};
        a.n = uint32_t(chi.pixels());
        a.positive = s.positive ? 1u : 0u;
        for (size_t i = 0; i < 9; ++i) a.m[i] = s.coupler_m_f32[i];
        for (size_t c = 0; c < 3; ++c) a.dmax[c] = s.dmax_f32[c];
        {
            gpu::Batch bt = ctx.gpu.batch();
            bt.dispatch_1d(ctx.kernel("spk_film_coupler_source"), a.n, {gpu::Arg::plane(chi), gpu::Arg::value(a)});
            bt.commit_and_wait();
        }
        fx::dir_diffusion_blur(ctx, chi);
        SpkFilmCouplerApplyArgs c{uint32_t(log_exposure.pixels()), s.curves.K};
        {
            gpu::Batch bt = ctx.gpu.batch();
            bt.dispatch_1d(ctx.kernel("spk_film_coupler_apply"), c.n,
                           {gpu::Arg::plane(log_exposure), gpu::Arg::plane(chi), gpu::Arg::buffer(s.curve_buffer),
                            gpu::Arg::value(c)});
            bt.commit_and_wait();
        }
        plane = log_exposure;
    }

    if (s.grain) {
        ctx.node("filming.grain");
        fx::grain(ctx, plane, *s.grain, args.film_seed);
    }

    gpu::Plane negative = ctx.gpu.persistent_plane(plane.w, plane.h, gpu::MemTag::Negative);
    std::memcpy(negative.data(), plane.data(), plane.bytes());
    return negative;
}

}  // namespace spk::film
