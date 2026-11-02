// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 Guangzhou Songzi Intelligent Technology Services Co., Ltd.

/* Session kernels.
 *
 * tier downscale: vertical then horizontal Gaussian prefilter (mirror edges, ascending k, float32
 *                 accumulation), then bilinear sampling from exact-integer tables
 * source copy of spk_open_device: drop a 4th channel; non-finite -> 0
 *
 * multiply-adds are explicit fma(), so the GPU and session::downscale_cpu (std::fmaf)
 * give the same bits whatever the compiler contracts
 */
#include "spk_common.h"
#include "session_shared.h"

using namespace metal;

// Vertical pass: out(x, y) = sum_{k=-r..r} w[k+r] * in(x, mirror(y+k)).
kernel void spk_session_blur_v(device const float* src [[buffer(0)]],
                               device float* dst [[buffer(1)]],
                               device const float* taps [[buffer(2)]],
                               constant SpkSessionBlurArgs& a [[buffer(3)]],
                               uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= a.w || gid.y >= a.h) return;
    const int r = int(a.radius);
    const int n = int(a.h);
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f;
    for (int k = -r; k <= r; ++k) {
        const int yy = spk::mirror_index(int(gid.y) + k, n);
        const float wk = taps[k + r];
        device const float* p = src + 3u * (uint(yy) * a.w + gid.x);
        acc0 = fma(wk, p[0], acc0);
        acc1 = fma(wk, p[1], acc1);
        acc2 = fma(wk, p[2], acc2);
    }
    device float* q = dst + 3u * (gid.y * a.w + gid.x);
    q[0] = acc0;
    q[1] = acc1;
    q[2] = acc2;
}

// Horizontal pass: out(x, y) = sum_{k=-r..r} w[k+r] * in(mirror(x+k), y).
kernel void spk_session_blur_h(device const float* src [[buffer(0)]],
                               device float* dst [[buffer(1)]],
                               device const float* taps [[buffer(2)]],
                               constant SpkSessionBlurArgs& a [[buffer(3)]],
                               uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= a.w || gid.y >= a.h) return;
    const int r = int(a.radius);
    const int n = int(a.w);
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f;
    device const float* row = src + 3u * (gid.y * a.w);
    for (int k = -r; k <= r; ++k) {
        const int xx = spk::mirror_index(int(gid.x) + k, n);
        const float wk = taps[k + r];
        device const float* p = row + 3u * uint(xx);
        acc0 = fma(wk, p[0], acc0);
        acc1 = fma(wk, p[1], acc1);
        acc2 = fma(wk, p[2], acc2);
    }
    device float* q = dst + 3u * (gid.y * a.w + gid.x);
    q[0] = acc0;
    q[1] = acc1;
    q[2] = acc2;
}

// Bilinear: per axis the host gives (i0, i1, t) per output index.
//   top = a + tx (b - a), bot = d + tx (e - d), out = top + ty (bot - top)
kernel void spk_session_resample(device const float* src [[buffer(0)]],
                                 device float* dst [[buffer(1)]],
                                 device const uint* xi [[buffer(2)]],   // [2*out_w]: i0, i1
                                 device const float* xt [[buffer(3)]],  // [out_w]
                                 device const uint* yi [[buffer(4)]],   // [2*out_h]
                                 device const float* yt [[buffer(5)]],  // [out_h]
                                 constant SpkSessionResampleArgs& a [[buffer(6)]],
                                 uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= a.out_w || gid.y >= a.out_h) return;
    const uint x0 = xi[2u * gid.x], x1 = xi[2u * gid.x + 1u];
    const uint y0 = yi[2u * gid.y], y1 = yi[2u * gid.y + 1u];
    const float tx = xt[gid.x], ty = yt[gid.y];
    device const float* pa = src + 3u * (y0 * a.in_w + x0);
    device const float* pb = src + 3u * (y0 * a.in_w + x1);
    device const float* pd = src + 3u * (y1 * a.in_w + x0);
    device const float* pe = src + 3u * (y1 * a.in_w + x1);
    device float* q = dst + 3u * (gid.y * a.out_w + gid.x);
    for (uint c = 0; c < 3u; ++c) {
        const float top = fma(tx, pb[c] - pa[c], pa[c]);
        const float bot = fma(tx, pe[c] - pd[c], pd[c]);
        q[c] = fma(ty, bot - top, top);
    }
}

// spk_open_device: tightly packed float32 x channels -> the session's 3-channel source plane.
kernel void spk_session_copy_source(device const float* src [[buffer(0)]],
                                    device float* dst [[buffer(1)]],
                                    constant SpkSessionCopyArgs& a [[buffer(2)]],
                                    uint gid [[thread_position_in_grid]]) {
    if (gid >= a.pixels) return;
    device const float* p = src + size_t(gid) * a.channels;
    for (uint c = 0; c < 3u; ++c) {
        const float v = p[c];
        dst[3u * gid + c] = isfinite(v) ? v : 0.0f;
    }
}
