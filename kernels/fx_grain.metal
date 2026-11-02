// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 Guangzhou Songzi Intelligent Technology Services Co., Ltd.

/* Kernels, named spk_fx_grain_<what>: sub-layer and simple grain draws, the lognormal field.
 *
 * every draw is keyed on (seed, pixel index, stream)
 */
#include "spk_common.h"
#include "fx_shared.h"
#include "fx_random.h"

using namespace metal;

// Right-side search on the float axis: the number of entries <= v.
static inline uint search_right(device const float* axis, uint K, uint c, float v) {
    uint lo = 0, hi = K;
    while (lo < hi) {
        const uint mid = lo + (hi - lo) / 2u;
        if (axis[3u * mid + c] <= v) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    return lo;
}

/* spk_fx_grain_draw, per pixel and channel c
 *   SIMPLE:
 *     d := density + Dmin_c
 *     out0 := (sum_l sublayer_draw(d, D_c, n_c, u_c, stream(c, l))) (1/N_sub) - Dmin_c
 *   sub-layer model:
 *     v := positive ? -density : density
 *     d_l := layer curve l at v, on axis[k*3 + c]; tables inv[k*3 + c], layers[(k*3 + l)*3 + c]
 *     v_l := sublayer_draw(d_l + dmin[l][c], D[l][c], n[l][c], u_c, stream(c, l))
 *     PLANES: out0, out1, out2 := v_0, v_1, v_2
 *     SUM:    out0 := ((0 + v_0) + v_1) + v_2, minus Dmin_c when shift
 */
kernel void spk_fx_grain_draw(device const float* density [[buffer(0)]],
                              device const float* axis [[buffer(1)]],
                              device const float* inv [[buffer(2)]],
                              device const float* layers [[buffer(3)]],
                              device float* out0 [[buffer(4)]],
                              device float* out1 [[buffer(5)]],
                              device float* out2 [[buffer(6)]],
                              constant SpkFxGrainArgs& q [[buffer(7)]],
                              uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= q.w || gid.y >= q.h) return;
    const uint pixel = gid.y * q.w + gid.x;
    const uint o = 3u * (gid.y * q.w + gid.x);
    for (uint c = 0; c < 3u; ++c) {
        const float vin = density[o + c];
        if (q.mode == SPK_FX_GRAIN_SIMPLE) {
            const float d = vin + q.dmin_c[c];
            float G = 0.0f;
            for (uint l = 0; l < q.n_sub; ++l) {
                G = G + spkfx::sublayer_draw(d, q.sat_d[c], q.count[c], q.uniformity[c], pixel,
                                             SPK_STREAM_GRAIN(c, l), q.seed);
            }
            out0[o + c] = G * q.inv_n_sub + (-q.dmin_c[c]);
            continue;
        }
        const float v = q.positive ? -vin : vin;
        const uint K = q.k;
        float d[3];
        if (v <= axis[c]) {
            for (uint l = 0; l < 3u; ++l) d[l] = layers[l * 3u + c];
        } else if (v >= axis[3u * (K - 1u) + c]) {
            for (uint l = 0; l < 3u; ++l) d[l] = layers[((K - 1u) * 3u + l) * 3u + c];
        } else {
            const uint idx = search_right(axis, K, c, v);
            const uint lo = idx - 1u;
            const float t = (v - axis[3u * lo + c]) * inv[3u * lo + c];
            for (uint l = 0; l < 3u; ++l) {
                const float y0 = layers[(lo * 3u + l) * 3u + c];
                const float y1 = layers[((lo + 1u) * 3u + l) * 3u + c];
                d[l] = y0 + t * (y1 - y0);
            }
        }
        float val[3];
        for (uint l = 0; l < 3u; ++l) {
            const uint lc = l * 3u + c;
            const float dl = d[l] + q.dmin_l[lc];
            val[l] = spkfx::sublayer_draw(dl, q.sat_d[lc], q.count[lc], q.uniformity[c], pixel,
                                          SPK_STREAM_GRAIN(c, l), q.seed);
        }
        if (q.mode == SPK_FX_GRAIN_PLANES) {
            out0[o + c] = val[0];
            out1[o + c] = val[1];
            out2[o + c] = val[2];
        } else {
            float G = ((0.0f + val[0]) + val[1]) + val[2];
            if (q.shift) G = G * 1.0f + (-q.dmin_c[c]);
            out0[o + c] = G;
        }
    }
}

/* spk_fx_grain_lognormal: one value per pixel and channel on stream_base + c,
 *   or with same_all one value on all channels from stream_base
 */
kernel void spk_fx_grain_lognormal(device float* out [[buffer(0)]],
                                   constant SpkFxLognormalArgs& q [[buffer(1)]],
                                   uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= q.w || gid.y >= q.h) return;
    const uint pixel = gid.y * q.w + gid.x;
    const uint o = 3u * (gid.y * q.w + gid.x);
    if (q.same_all) {
        const float v = spkfx::lognormal(q.mean, q.stddev, pixel, q.stream_base, q.seed);
        out[o] = v;
        out[o + 1] = v;
        out[o + 2] = v;
        return;
    }
    for (uint c = 0; c < 3u; ++c) out[o + c] = spkfx::lognormal(q.mean, q.stddev, pixel, q.stream_base + c, q.seed);
}

// Test kernel (tests/test_fx.cpp): out[i] = Poisson(lam[i]) for pixel i on `stream` with `seed`,
// and out2[i] = the lognormal value (mean, std) of pixel i on `stream`.
struct SpkFxSamplerTestArgs {
    uint n;
    uint stream;
    uint seed;
    float mean;
    float stddev;
};
kernel void spk_fx_grain_test_sampler(device const float* lam [[buffer(0)]],
                                      device float* out [[buffer(1)]],
                                      device float* out2 [[buffer(2)]],
                                      constant SpkFxSamplerTestArgs& q [[buffer(3)]],
                                      uint i [[thread_position_in_grid]]) {
    if (i >= q.n) return;
    out[i] = spkfx::poisson(lam[i], i, q.stream, q.seed);
    out2[i] = spkfx::lognormal(q.mean, q.stddev, i, q.stream, q.seed);
}
