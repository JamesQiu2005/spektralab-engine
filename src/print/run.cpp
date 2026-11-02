// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 Guangzhou Songzi Intelligent Technology Services Co., Ltd.

/* Executing the print side: print_scan_encode() and preview_lut().
 *
 * per-pixel work runs in kernels/print.metal; each stage encodes its own batch and waits
 */
#include <cstring>

#include "fx/glare.hpp"
#include "fx/scanner.hpp"
#include "print/internal.hpp"
#include "util/error.hpp"
#include "util/numeric.hpp"

namespace spk::print {

using params::P;

LiveParams LiveParams::from(const params::ParamSet& p) {
    LiveParams l;
    l.print_exposure = p.f(P::print_exposure);
    l.m_filter_shift = p.f(P::m_filter_shift);
    l.y_filter_shift = p.f(P::y_filter_shift);
    l.preflash_exposure = p.f(P::preflash_exposure);
    l.filter_shift_scale = p.f(P::filter_shift_scale);
    return l;
}

std::optional<std::array<double, 3>> solve_filter_pack(const res::Resources& res, const params::ParamSet& p,
                                                       const res::Profile& film) {
    return res.filters.lookup_with_alias(p.s(P::print_stock), p.s(P::enlarger_illuminant), film.id, film.base_id);
}

namespace {

template <class T>
void dispatch_inplace(const core::RenderContext& ctx, const char* kernel, gpu::Plane& plane, T args,
                      const gpu::Buffer* table = nullptr) {
    args.w = plane.w;
    args.h = plane.h;
    gpu::Kernel k = ctx.kernel(kernel);
    gpu::Batch b = ctx.gpu.batch();
    if (table)
        b.dispatch_2d(k, plane.w, plane.h, {gpu::Arg::plane(plane), gpu::Arg::buffer(*table), gpu::Arg::value(args)});
    else
        b.dispatch_2d(k, plane.w, plane.h, {gpu::Arg::plane(plane), gpu::Arg::value(args)});
    b.commit_and_wait();
}

void spectral(const core::RenderContext& ctx, const gpu::Plane& in, gpu::Plane& out, SpkPrintSpectralArgs args) {
    args.w = in.w;
    args.h = in.h;
    gpu::Kernel k = ctx.kernel("spk_print_spectral");
    gpu::Batch b = ctx.gpu.batch();
    b.dispatch_2d(k, in.w, in.h, {gpu::Arg::plane(in), gpu::Arg::plane(out), gpu::Arg::value(args)});
    b.commit_and_wait();
}

/* encode_into(ctx, plane, args, out)
 *   spk_print_encode into a tightly packed scratch image; copy row by row into out's row pitch
 */
void encode_into(const core::RenderContext& ctx, const gpu::Plane& plane, SpkPrintEncodeArgs args, gpu::OutputImage& out) {
    if (plane.w > out.width || plane.h > out.height) fail_internal("print encode: plane larger than the output image");
    args.w = plane.w;
    args.h = plane.h;
    args.stride = plane.w;
    const size_t row_bytes = size_t(plane.w) * 4 * sizeof(uint16_t);
    gpu::Buffer tmp = ctx.gpu.pool().acquire(row_bytes * plane.h);
    gpu::Kernel k = ctx.kernel("spk_print_encode");
    gpu::Batch b = ctx.gpu.batch();
    b.dispatch_2d(k, plane.w, plane.h, {gpu::Arg::plane(plane), gpu::Arg::buffer(tmp), gpu::Arg::value(args)});
    b.commit_and_wait();
    const uint8_t* src = static_cast<const uint8_t*>(tmp.contents());
    for (uint32_t y = 0; y < plane.h; ++y)
        std::memcpy(out.rgba16 + size_t(y) * out.row_stride_px * 4, src + size_t(y) * row_bytes, row_bytes);
}

// Releases a result texture when a later step of the same call fails.
struct OutputGuard {
    gpu::OutputImage* img;
    ~OutputGuard() {
        if (img && img->texture) gpu::Context::release_texture(img->texture);
    }
};

}  // namespace

uint32_t node_count(const PrintSetup& setup, const core::EffectVariants& v) {
    const PrintSetup::Impl& s = setup.impl();
    const bool printed = !s.scanned;
    uint32_t n = 4;  // scan, to_rgb, gamut and encode always run
    if (printed) n += 3;  // enlarger, gain, paper
    if (levels_active(s)) ++n;
    if (printed && fx::glare_draws_seed(s.glare)) ++n;
    if (!v.print_spatial_omitted) ++n;
    if (printed && s.edr) ++n;
    return n;
}

/* print_scan_encode(ctx, setup, args)
 *   r := compute_run(setup, live); plane := negative; out := fresh output image
 *   printed:
 *     plane := log10 exposure on paper, spectral integral with r.enl
 *     plane := log10(max(10^plane k_gain, 0) + 1e-10)
 *     plane := paper curve lookup
 *   scan: scanned path writes a scratch plane, never the cached negative
 *     plane := XYZ = max(sum_l T(l) wt(l), 0) + 1e-10
 *   levels active: Y' := clamp(m Y + q, 0, 1); XYZ *= Y' / (Y + 1e-10)
 *   printed and glare can draw: glare(plane, print_seed)
 *   plane := to_rgb plane
 *   plane := spk_print_gamut(plane, cmax)
 *   not print_spatial_omitted: scanner_blur_unsharp(plane)
 *   printed and EDR: tone map
 *   encode into out; finish_output
 *   on failure: the output image is released
 */
gpu::OutputImage print_scan_encode(core::RenderContext& ctx, const PrintSetup& setup, const PrintArgs& args) {
    if (!args.negative || args.negative.w == 0 || args.negative.h == 0) fail_internal("print: empty negative");
    const PrintSetup::Impl& s = setup.impl();
    const std::shared_ptr<const PrintRun> run_ptr = compute_run(s, args.live);
    const PrintRun& r = *run_ptr;
    const bool printed = !s.scanned;

    gpu::Plane plane = args.negative;
    gpu::OutputImage out = ctx.gpu.make_output(plane.w, plane.h);
    OutputGuard guard{&out};

    if (printed) {
        ctx.node("printing.enlarger");
        gpu::Plane exposure = ctx.scratch(plane.w, plane.h);
        spectral(ctx, plane, exposure, r.enl);
        plane = exposure;

        ctx.node("printing.gain");
        SpkPrintGainArgs gain{};
        gain.k = r.k_gain;
        dispatch_inplace(ctx, "spk_print_gain", plane, gain);

        ctx.node("printing.paper");
        SpkPrintCurveArgs curve{};
        curve.K = s.curve_k;
        dispatch_inplace(ctx, "spk_print_paper", plane, curve, &s.curve_gpu);
    }

    {
        ctx.node("scanning.scan");
        gpu::Plane scan = s.scanned ? ctx.scratch(plane.w, plane.h) : plane;
        spectral(ctx, plane, scan, s.scan_args);
        plane = scan;
    }

    if (levels_active(s)) {
        ctx.node("scanning.levels");
        SpkPrintLevelsArgs a{};
        a.m = num::f32(r.m);
        a.q = num::f32(r.q);
        dispatch_inplace(ctx, "spk_print_levels", plane, a);
    }

    if (printed && fx::glare_draws_seed(s.glare)) {
        ctx.node("scanning.glare");
        fx::glare(ctx, plane, s.glare, args.print_seed);
    }

    ctx.node("scanning.to_rgb");
    dispatch_inplace(ctx, "spk_print_matrix", plane, s.to_rgb);

    ctx.node("scanning.gamut");
    dispatch_inplace(ctx, "spk_print_gamut", plane, s.cam_args, &s.cmax_gpu);

    if (!ctx.variants.print_spatial_omitted) {
        ctx.node("scanning.scanner");
        fx::scanner_blur_unsharp(ctx, plane, s.scanner);
    }

    if (printed && s.edr) {
        ctx.node("scanning.edr");
        dispatch_inplace(ctx, "spk_print_edr", plane, s.edr_args, &s.edr_gpu);
    }

    ctx.node("scanning.encode");
    encode_into(ctx, plane, s.enc_args, out);

    ctx.gpu.finish_output(out);
    guard.img = nullptr;
    return out;
}

/* preview_lut(ctx, negative, table)
 *   a.lo := table.lo; a.s := f32(1 / (hi - lo)) in float64
 *   spk_print_lut into a packed scratch image; copy row by row; finish_output
 */
gpu::OutputImage preview_lut(core::RenderContext& ctx, const gpu::Plane& negative, const res::PrintLutTable& table) {
    if (!table.table || table.size < 2) fail_internal("preview_lut: empty table");
    const uint32_t S = table.size;
    const size_t n = size_t(S) * S * S * 3;
    gpu::Buffer lut = ctx.gpu.pool().acquire(n * sizeof(float));
    std::memcpy(lut.contents(), table.table, n * sizeof(float));

    SpkPrintLutArgs a{};
    a.w = negative.w;
    a.h = negative.h;
    a.stride = negative.w;
    a.size = S;
    for (int c = 0; c < 3; ++c) {
        a.lo[c] = table.lo[c];
        a.s[c] = num::f32(1.0 / (double(table.hi[c]) - double(table.lo[c])));
    }
    gpu::OutputImage out = ctx.gpu.make_output(negative.w, negative.h);
    OutputGuard guard{&out};
    const size_t row_bytes = size_t(negative.w) * 4 * sizeof(uint16_t);
    gpu::Buffer tmp = ctx.gpu.pool().acquire(row_bytes * negative.h);
    gpu::Kernel k = ctx.kernel("spk_print_lut");
    gpu::Batch b = ctx.gpu.batch();
    b.dispatch_2d(k, negative.w, negative.h,
                  {gpu::Arg::plane(negative), gpu::Arg::buffer(lut), gpu::Arg::buffer(tmp), gpu::Arg::value(a)});
    b.commit_and_wait();
    const uint8_t* src = static_cast<const uint8_t*>(tmp.contents());
    for (uint32_t y = 0; y < negative.h; ++y)
        std::memcpy(out.rgba16 + size_t(y) * out.row_stride_px * 4, src + size_t(y) * row_bytes, row_bytes);
    ctx.gpu.finish_output(out);
    guard.img = nullptr;
    return out;
}

}  // namespace spk::print
