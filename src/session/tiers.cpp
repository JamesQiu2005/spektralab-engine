// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 Guangzhou Songzi Intelligent Technology Services Co., Ltd.

/* Tiers and the tier downscale.
 *
 * The GPU path (kernels/session.metal) and downscale_cpu() do the same operations in the same order:
 *   taps float64 -> float32
 *   vertical pass, horizontal pass: mirror edges, acc = fma(w_k, x, acc), ascending k
 *   bilinear blend with fma; index tables in exact integers
 */
#include "session/tiers.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include "session_shared.h"
#include "util/numeric.hpp"

namespace spk::session {

std::optional<Tier> parse_tier(std::string_view s) {
    if (s == "live") return Tier::Live;
    if (s == "preview") return Tier::Preview;
    if (s == "full") return Tier::Full;
    return std::nullopt;
}

const char* tier_name(Tier t) {
    switch (t) {
        case Tier::Live: return "live";
        case Tier::Preview: return "preview";
        case Tier::Full: return "full";
    }
    return "?";
}

uint32_t tier_target_long_edge(Tier t, uint32_t preview_long_edge) {
    switch (t) {
        case Tier::Live: return preview_long_edge;
        case Tier::Preview: return kPreviewLongEdge;
        case Tier::Full: return 0;
    }
    return 0;
}

Size downscale_size(uint32_t w, uint32_t h, uint32_t long_edge) {
    const uint32_t m = std::max(w, h);
    if (long_edge == 0 || m <= long_edge) return {w, h};
    const double s = double(long_edge) / double(m);
    const double fh = std::floor(double(h) * s);
    const double fw = std::floor(double(w) * s);
    return {std::max<uint32_t>(1, uint32_t(fw)), std::max<uint32_t>(1, uint32_t(fh))};
}

namespace {

/* axis_taps(n, m) -> gaussian_kernel_f32(sigma = max(0, (n/m - 1) / 2), truncate = 4)
 *   one tap {1} means r = 0: the axis is skipped
 */
std::vector<float> axis_taps(uint32_t n, uint32_t m) {
    const double f = double(n) / double(m);
    const double sigma = std::max(0.0, (f - 1.0) / 2.0);
    return num::gaussian_kernel_f32(sigma, 4.0);
}

/* axis_samples(n, m): source coordinate (o + 1/2) n / m - 1/2, mirrored, in exact integers
 *   for o in 0..m-1:
 *     num := (2 o + 1) n - m; den := 2 m
 *     if num > den (n - 1): num := 2 den (n - 1) - num
 *     if num < 0: num := -num
 *     i0 := min(floor(num / den), n - 1); i1 := min(i0 + 1, n - 1)
 *     t := float32(num - i0 den) / float32(den)
 */
struct AxisSamples {
    std::vector<uint32_t> idx;  // 2 per output: i0, i1
    std::vector<float> t;
};
AxisSamples axis_samples(uint32_t n, uint32_t m) {
    AxisSamples s;
    s.idx.resize(size_t(2) * m);
    s.t.resize(m);
    const int64_t N = n, M = m;
    const int64_t den = 2 * M;
    for (int64_t o = 0; o < M; ++o) {
        int64_t num = (2 * o + 1) * N - M;
        if (num > den * (N - 1)) num = 2 * den * (N - 1) - num;
        if (num < 0) num = -num;
        int64_t i0 = num / den;  // num >= 0: floor
        if (i0 > N - 1) i0 = N - 1;
        const int64_t i1 = std::min<int64_t>(i0 + 1, N - 1);
        s.idx[size_t(2 * o)] = uint32_t(i0);
        s.idx[size_t(2 * o + 1)] = uint32_t(i1);
        s.t[size_t(o)] = float(num - i0 * den) / float(den);
    }
    return s;
}

void blur_v_cpu(const float* src, float* dst, uint32_t w, uint32_t h, const std::vector<float>& taps) {
    const int r = int(taps.size() / 2);
    for (uint32_t y = 0; y < h; ++y)
        for (uint32_t x = 0; x < w; ++x) {
            float acc[3] = {0.0f, 0.0f, 0.0f};
            for (int k = -r; k <= r; ++k) {
                const int64_t yy = num::mirror_index(int64_t(y) + k, h);
                const float wk = taps[size_t(k + r)];
                const float* p = src + 3 * (size_t(yy) * w + x);
                for (int c = 0; c < 3; ++c) acc[c] = std::fmaf(wk, p[c], acc[c]);
            }
            float* q = dst + 3 * (size_t(y) * w + x);
            for (int c = 0; c < 3; ++c) q[c] = acc[c];
        }
}

void blur_h_cpu(const float* src, float* dst, uint32_t w, uint32_t h, const std::vector<float>& taps) {
    const int r = int(taps.size() / 2);
    for (uint32_t y = 0; y < h; ++y) {
        const float* row = src + 3 * size_t(y) * w;
        for (uint32_t x = 0; x < w; ++x) {
            float acc[3] = {0.0f, 0.0f, 0.0f};
            for (int k = -r; k <= r; ++k) {
                const int64_t xx = num::mirror_index(int64_t(x) + k, w);
                const float wk = taps[size_t(k + r)];
                const float* p = row + 3 * size_t(xx);
                for (int c = 0; c < 3; ++c) acc[c] = std::fmaf(wk, p[c], acc[c]);
            }
            float* q = dst + 3 * (size_t(y) * w + x);
            for (int c = 0; c < 3; ++c) q[c] = acc[c];
        }
    }
}

}  // namespace

/* downscale(): upload taps and index tables; [blur_v] [blur_h] resample; wait */
gpu::Plane downscale(const core::RenderContext& ctx, const gpu::Plane& src, uint32_t long_edge, gpu::MemTag tag) {
    const Size o = downscale_size(src.w, src.h, long_edge);
    if (o.w == src.w && o.h == src.h) return src;

    gpu::Context& g = ctx.gpu;
    const std::vector<float> ty = axis_taps(src.h, o.h);
    const std::vector<float> tx = axis_taps(src.w, o.w);
    const AxisSamples sx = axis_samples(src.w, o.w);
    const AxisSamples sy = axis_samples(src.h, o.h);

    auto upload = [&](const void* p, size_t bytes) {
        gpu::Buffer b = g.pool().acquire(std::max<size_t>(bytes, 16));
        std::memcpy(b.contents(), p, bytes);
        return b;
    };
    gpu::Buffer b_ty = upload(ty.data(), ty.size() * sizeof(float));
    gpu::Buffer b_tx = upload(tx.data(), tx.size() * sizeof(float));
    gpu::Buffer b_xi = upload(sx.idx.data(), sx.idx.size() * sizeof(uint32_t));
    gpu::Buffer b_xt = upload(sx.t.data(), sx.t.size() * sizeof(float));
    gpu::Buffer b_yi = upload(sy.idx.data(), sy.idx.size() * sizeof(uint32_t));
    gpu::Buffer b_yt = upload(sy.t.data(), sy.t.size() * sizeof(float));

    gpu::Plane out = g.persistent_plane(o.w, o.h, tag);
    gpu::Batch batch = g.batch();
    gpu::Plane cur = src;
    gpu::Plane t1, t2;
    if (ty.size() > 1) {
        t1 = g.pool().plane(src.w, src.h);
        SpkSessionBlurArgs a{src.w, src.h, uint32_t(ty.size() / 2), 0};
        batch.dispatch_2d(ctx.kernel("spk_session_blur_v"), src.w, src.h,
                          {gpu::Arg::plane(cur), gpu::Arg::plane(t1), gpu::Arg::buffer(b_ty), gpu::Arg::value(a)});
        cur = t1;
    }
    if (tx.size() > 1) {
        t2 = g.pool().plane(src.w, src.h);
        SpkSessionBlurArgs a{src.w, src.h, uint32_t(tx.size() / 2), 0};
        batch.dispatch_2d(ctx.kernel("spk_session_blur_h"), src.w, src.h,
                          {gpu::Arg::plane(cur), gpu::Arg::plane(t2), gpu::Arg::buffer(b_tx), gpu::Arg::value(a)});
        cur = t2;
    }
    SpkSessionResampleArgs ra{src.w, src.h, o.w, o.h};
    batch.dispatch_2d(ctx.kernel("spk_session_resample"), o.w, o.h,
                      {gpu::Arg::plane(cur), gpu::Arg::plane(out), gpu::Arg::buffer(b_xi), gpu::Arg::buffer(b_xt),
                       gpu::Arg::buffer(b_yi), gpu::Arg::buffer(b_yt), gpu::Arg::value(ra)});
    batch.commit_and_wait();
    return out;
}

/* downscale_cpu(): the CPU copy of downscale(), same operands, same order
 *   per output pixel:
 *     top := fma(tx, b - a, a); bot := fma(tx, e - d, d); out := fma(ty, bot - top, top)
 */
void downscale_cpu(const float* src, uint32_t w, uint32_t h, float* dst, uint32_t ow, uint32_t oh) {
    if (ow == w && oh == h) {
        std::copy(src, src + size_t(3) * w * h, dst);
        return;
    }
    const std::vector<float> ty = axis_taps(h, oh);
    const std::vector<float> tx = axis_taps(w, ow);
    std::vector<float> a(src, src + size_t(3) * w * h), b(a.size());
    if (ty.size() > 1) {
        blur_v_cpu(a.data(), b.data(), w, h, ty);
        a.swap(b);
    }
    if (tx.size() > 1) {
        blur_h_cpu(a.data(), b.data(), w, h, tx);
        a.swap(b);
    }
    const AxisSamples sx = axis_samples(w, ow);
    const AxisSamples sy = axis_samples(h, oh);
    for (uint32_t y = 0; y < oh; ++y) {
        const uint32_t y0 = sy.idx[2 * y], y1 = sy.idx[2 * y + 1];
        const float tyv = sy.t[y];
        for (uint32_t x = 0; x < ow; ++x) {
            const uint32_t x0 = sx.idx[2 * x], x1 = sx.idx[2 * x + 1];
            const float txv = sx.t[x];
            const float* pa = &a[3 * (size_t(y0) * w + x0)];
            const float* pb = &a[3 * (size_t(y0) * w + x1)];
            const float* pd = &a[3 * (size_t(y1) * w + x0)];
            const float* pe = &a[3 * (size_t(y1) * w + x1)];
            float* q = dst + 3 * (size_t(y) * ow + x);
            for (int c = 0; c < 3; ++c) {
                const float top = std::fmaf(txv, pb[c] - pa[c], pa[c]);
                const float bot = std::fmaf(txv, pe[c] - pd[c], pd[c]);
                q[c] = std::fmaf(tyv, bot - top, top);
            }
        }
    }
}

}  // namespace spk::session
