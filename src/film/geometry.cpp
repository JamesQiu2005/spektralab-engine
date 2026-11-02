// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 Guangzhou Songzi Intelligent Technology Services Co., Ltd.

/* Input transfer decode and geometry, plus the small GPU helpers the film stages share. */
#include <cmath>
#include <cstring>

#include "film/film_internal.hpp"
#include "film_shared.h"
#include "util/numeric.hpp"

namespace spk::film {

using params::P;

namespace {

void split(double v, float& hi, float& lo) {
    hi = num::f32(v);
    lo = num::f32(v - double(hi));
}

uint32_t quarter_turns(const params::ParamSet& p) {
    const int64_t n = p.i(P::geometry_quarter_turns);
    return uint32_t(((n % 4) + 4) % 4);
}

}  // namespace

/* plan_geometry(p, SW, SH) -> GeometryPlan: output pixel -> source coordinate, per axis
 *   CW := max(1, round(cw SW)); CH := max(1, round(ch SH))
 *   odd quarter turn: (OW, OH) := (CH, CW); else (CW, CH)
 *   sx := cw SW / CW; sy := ch SH / CH; hx := cw SW / 2; hy := ch SH / 2
 *   cx := (crop_x + cw / 2) SW - 1/2; cy := (crop_y + ch / 2) SH - 1/2
 *   U0(ox) := ox + 1/2, flipped: OW - (ox + 1/2)
 *   V0(oy) := oy + 1/2, flipped: OH - (oy + 1/2)
 *   turn 0: u = U0,    v = V0
 *   turn 1: v = OW-U0, u = V0
 *   turn 2: u = OW-U0, v = OH-V0
 *   turn 3: v = U0,    u = OH-V0
 *   P := sx u - hx; Q := sy v - hy
 *   f_x := cx + (P cos - Q sin); f_y := cy + (P sin + Q cos)
 *   P and Q terms stored per column or per row, on the axis u or v comes from
 */
GeometryPlan plan_geometry(const params::ParamSet& p, uint32_t SW, uint32_t SH) {
    GeometryPlan g;
    g.src_w = SW;
    g.src_h = SH;
    const double cx = p.f(P::geometry_crop_x), cy = p.f(P::geometry_crop_y);
    const double cw = p.f(P::geometry_crop_w), ch = p.f(P::geometry_crop_h);
    const double deg = p.f(P::geometry_rotation_deg);
    const uint32_t q = quarter_turns(p);
    const bool fh = p.b(P::geometry_flip_h), fv = p.b(P::geometry_flip_v);
    g.identity = cx == 0.0 && cy == 0.0 && cw == 1.0 && ch == 1.0 && deg == 0.0 && q == 0 && !fh && !fv;

    const double CW = std::max(1.0, num::round_half_away(cw * double(SW)));
    const double CH = std::max(1.0, num::round_half_away(ch * double(SH)));
    const bool odd = (q & 1u) != 0;
    const double OW = odd ? CH : CW, OH = odd ? CW : CH;
    g.dst_w = uint32_t(OW);
    g.dst_h = uint32_t(OH);
    if (g.identity) return g;

    const double sx = cw * double(SW) / CW, sy = ch * double(SH) / CH;
    const double hx = cw * double(SW) / 2.0, hy = ch * double(SH) / 2.0;
    g.cx = (cx + cw / 2.0) * double(SW) - 0.5;
    g.cy = (cy + ch / 2.0) * double(SH) - 0.5;
    const double pi = 3.14159265358979323846;
    const double theta = deg * pi / 180.0;
    const double ca = std::cos(theta), sa = std::sin(theta);

    auto U0 = [&](uint32_t ox) { double u = double(ox) + 0.5; return fh ? OW - u : u; };
    auto V0 = [&](uint32_t oy) { double v = double(oy) + 0.5; return fv ? OH - v : v; };
    g.col_x.resize(g.dst_w);
    g.col_y.resize(g.dst_w);
    g.row_x.resize(g.dst_h);
    g.row_y.resize(g.dst_h);
    auto put_u = [&](double u, double& tx, double& ty) {
        const double P_ = sx * u - hx;
        tx = P_ * ca;
        ty = P_ * sa;
    };
    auto put_v = [&](double v, double& tx, double& ty) {
        const double Q_ = sy * v - hy;
        tx = -(Q_ * sa);
        ty = Q_ * ca;
    };
    for (uint32_t ox = 0; ox < g.dst_w; ++ox) {
        const double u0 = U0(ox);
        switch (q) {
            case 0: put_u(u0, g.col_x[ox], g.col_y[ox]); break;       // u = U0
            case 1: put_v(OW - u0, g.col_x[ox], g.col_y[ox]); break;  // v = OW - U0
            case 2: put_u(OW - u0, g.col_x[ox], g.col_y[ox]); break;  // u = OW - U0
            default: put_v(u0, g.col_x[ox], g.col_y[ox]); break;      // v = U0
        }
    }
    for (uint32_t oy = 0; oy < g.dst_h; ++oy) {
        const double v0 = V0(oy);
        switch (q) {
            case 0: put_v(v0, g.row_x[oy], g.row_y[oy]); break;       // v = V0
            case 1: put_u(v0, g.row_x[oy], g.row_y[oy]); break;       // u = V0
            case 2: put_v(OH - v0, g.row_x[oy], g.row_y[oy]); break;  // v = OH - V0
            default: put_u(OH - v0, g.row_x[oy], g.row_y[oy]); break; // u = OH - V0
        }
    }
    return g;
}

Size geometry_output_size(const params::ParamSet& p, uint32_t w, uint32_t h) {
    const GeometryPlan g = plan_geometry(p, w, h);
    return {g.dst_w, g.dst_h};
}

/* geometry_cpu(g, src, dst): bilinear at (f_x, f_y) in float64 coordinates; clamped; tests only
 *   f <= 0: i0 = 0, t = 0; f >= n-1: i0 = n-1, t = 0; else i0 = floor(f), t = f - i0
 *   top := a (1-tx) + b tx; bot := d (1-tx) + e tx; out := top (1-ty) + bot ty
 */
void geometry_cpu(const GeometryPlan& g, const float* src, float* dst) {
    if (g.identity) {
        std::memcpy(dst, src, size_t(g.src_w) * g.src_h * 3 * sizeof(float));
        return;
    }
    auto axis = [](double f, uint32_t n, uint32_t& i0, uint32_t& i1, float& t) {
        if (f <= 0.0) {
            i0 = 0;
            t = 0.0f;
        } else if (f >= double(n - 1)) {
            i0 = n - 1;
            t = 0.0f;
        } else {
            const double fl = std::floor(f);
            i0 = uint32_t(fl);
            t = num::f32(f - fl);
            t = std::fmin(std::fmax(t, 0.0f), 1.0f);
        }
        i1 = std::min(i0 + 1, n - 1);
    };
    for (uint32_t oy = 0; oy < g.dst_h; ++oy)
        for (uint32_t ox = 0; ox < g.dst_w; ++ox) {
            const double fx = g.cx + (g.col_x[ox] + g.row_x[oy]);
            const double fy = g.cy + (g.col_y[ox] + g.row_y[oy]);
            uint32_t x0, x1, y0, y1;
            float tx, ty;
            axis(fx, g.src_w, x0, x1, tx);
            axis(fy, g.src_h, y0, y1, ty);
            const float* a = src + 3 * (size_t(y0) * g.src_w + x0);
            const float* b = src + 3 * (size_t(y0) * g.src_w + x1);
            const float* d = src + 3 * (size_t(y1) * g.src_w + x0);
            const float* e = src + 3 * (size_t(y1) * g.src_w + x1);
            float* o = dst + 3 * (size_t(oy) * g.dst_w + ox);
            for (int c = 0; c < 3; ++c) {
                const float top = a[c] * (1.0f - tx) + b[c] * tx;
                const float bot = d[c] * (1.0f - tx) + e[c] * tx;
                o[c] = top * (1.0f - ty) + bot * ty;
            }
        }
}

// ---- GPU helpers ----------------------------------------------------------------------------------
gpu::Plane run_decode(const core::RenderContext& ctx, const gpu::Plane& in, res::CctfMode mode) {
    gpu::Plane out = ctx.scratch(in.w, in.h);
    SpkFilmDecodeArgs a{};
    a.n = uint32_t(in.floats());
    a.mode = uint32_t(mode);
    a.bt709_bp = num::f32(1.099 * std::pow(0.018, 0.45) - 0.099);
    a.bt709_exp = num::f32(1.0 / 0.45);
    gpu::Batch b = ctx.gpu.batch();
    b.dispatch_1d(ctx.kernel("spk_film_decode"), a.n, {gpu::Arg::plane(in), gpu::Arg::plane(out), gpu::Arg::value(a)});
    b.commit_and_wait();
    return out;
}

/* run_geometry(ctx, in, g): split each float64 term into (hi, lo) floats; dispatch spk_film_geometry */
gpu::Plane run_geometry(const core::RenderContext& ctx, const gpu::Plane& in, const GeometryPlan& g) {
    gpu::Plane out = ctx.scratch(g.dst_w, g.dst_h);
    gpu::Buffer col = ctx.gpu.pool().acquire(size_t(g.dst_w) * 4 * sizeof(float));
    gpu::Buffer row = ctx.gpu.pool().acquire(size_t(g.dst_h) * 4 * sizeof(float));
    float* c = col.as<float>();
    float* r = row.as<float>();
    for (uint32_t i = 0; i < g.dst_w; ++i) {
        split(g.col_x[i], c[4 * i], c[4 * i + 1]);
        split(g.col_y[i], c[4 * i + 2], c[4 * i + 3]);
    }
    for (uint32_t i = 0; i < g.dst_h; ++i) {
        split(g.row_x[i], r[4 * i], r[4 * i + 1]);
        split(g.row_y[i], r[4 * i + 2], r[4 * i + 3]);
    }
    SpkFilmGeometryArgs a{};
    a.src_w = g.src_w;
    a.src_h = g.src_h;
    a.dst_w = g.dst_w;
    a.dst_h = g.dst_h;
    split(g.cx, a.cx_hi, a.cx_lo);
    split(g.cy, a.cy_hi, a.cy_lo);
    gpu::Batch b = ctx.gpu.batch();
    b.dispatch_2d(ctx.kernel("spk_film_geometry"), g.dst_w, g.dst_h,
                  {gpu::Arg::plane(in), gpu::Arg::plane(out), gpu::Arg::buffer(col), gpu::Arg::buffer(row),
                   gpu::Arg::value(a)});
    b.commit_and_wait();
    return out;
}

void run_scale(const core::RenderContext& ctx, gpu::Plane& p, float g) {
    SpkFilmScaleArgs a{uint32_t(p.floats()), g};
    gpu::Batch b = ctx.gpu.batch();
    b.dispatch_1d(ctx.kernel("spk_film_scale"), a.n, {gpu::Arg::plane(p), gpu::Arg::value(a)});
    b.commit_and_wait();
}

float plane_max(const gpu::Plane& p) {
    const float* d = p.data();
    const size_t n = p.floats();
    float m = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < n; ++i)
        if (d[i] > m) m = d[i];
    return m;
}

gpu::Plane decode_and_geometry(const core::RenderContext& ctx, const params::ParamSet& p, const res::Resources& res,
                               const gpu::Plane& raw) {
    gpu::Plane cur = raw;
    bool owned = false;
    if (p.b(P::input_cctf_decoding)) {
        auto mode = res::cctf_mode_of(p.s(P::input_color_space));
        if (!mode) fail_user("input_color_space '" + p.s(P::input_color_space) + "' is not a colour space in the constants blob");
        cur = run_decode(ctx, cur, *mode);
        owned = true;
    }
    const GeometryPlan g = plan_geometry(p, cur.w, cur.h);
    if (!g.identity) {
        cur = run_geometry(ctx, cur, g);
        owned = true;
    }
    if (!owned) {
        gpu::Plane copy = ctx.scratch(raw.w, raw.h);
        std::memcpy(copy.data(), raw.data(), raw.bytes());
        cur = copy;
    }
    return cur;
}

}  // namespace spk::film
