// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 Guangzhou Songzi Intelligent Technology Services Co., Ltd.

/* Kernels, named spk_fx_halation_<what>: highlight headroom and the mobile.2 halation resampling.
 *
 * the scatter, bounce and back-reflection mixtures run on the fx_blur kernels
 */
#include "spk_common.h"
#include "fx_shared.h"

using namespace metal;

static inline float smooth01(float t) {
    const float u = clamp(t, 0.0f, 1.0f);
    return u * u * (3.0f - 2.0f * u);
}

/* spk_fx_halation_headroom: m := max channel; m <= 0: g = 1
 *   s1 := smooth01((log2 m - log2 clip + 1) / 0.75)
 *   s2 := smooth01((log2 m - log2 0.184 - 1.5) / 1)
 *   g := 2^(headroom_ev s1 s2); all channels times g
 */
kernel void spk_fx_halation_headroom(device float* p [[buffer(0)]],
                                     constant SpkFxHeadroomArgs& q [[buffer(1)]],
                                     uint i [[thread_position_in_grid]]) {
    if (i >= q.n) return;
    device float* v = p + 3u * i;
    const float m = max(max(v[0], v[1]), v[2]);
    float g = 1.0f;
    if (m > 0.0f) {
        const float lm = log2(m);
        const float s1 = smooth01(((lm - log2(q.clip)) - (-1.0f)) / 0.75f);
        const float s2 = smooth01(((lm - log2(SPK_MID_GREY)) - 1.5f) / 1.0f);
        g = exp2((q.headroom_ev * s1) * s2);
    }
    v[0] = v[0] * g;
    v[1] = v[1] * g;
    v[2] = v[2] * g;
}

/* spk_fx_halation_tent_down: tent splat
 *   coarse sample (I, J) is centred at full-res (F I + c, F J + c), c = (F - 1) / 2
 *   value := (1 / F^2) sum_y sum_x w(y - cy) w(x - cx) in[clamp y][clamp x]
 *   w(d) := max(0, 1 - |d| / F), over integers within distance < F
 */
kernel void spk_fx_halation_tent_down(device const float* in [[buffer(0)]],
                                      device float* out [[buffer(1)]],
                                      constant SpkFxTentArgs& q [[buffer(2)]],
                                      uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= q.cw || gid.y >= q.ch) return;
    const int F = int(q.f);
    const float fF = float(q.f);
    const float c = (fF - 1.0f) * 0.5f;
    const float cy = fF * float(gid.y) + c;
    const float cx = fF * float(gid.x) + c;
    // integers y with |y - cy| < F: cy is k + c, so y runs from F*gid.y - F + 1 + floor(c) ... ;
    // enumerate a safe superset and test the distance.
    const int y_lo = int(F * int(gid.y)) - F;
    const int y_hi = int(F * int(gid.y)) + 2 * F;
    const int x_lo = int(F * int(gid.x)) - F;
    const int x_hi = int(F * int(gid.x)) + 2 * F;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f;
    for (int y = y_lo; y <= y_hi; ++y) {
        const float dy = abs(float(y) - cy);
        if (!(dy < fF)) continue;
        const float wy = max(0.0f, 1.0f - dy / fF);
        const int yy = clamp(y, 0, int(q.h) - 1);
        device const float* row = in + 3u * uint(yy) * q.w;
        for (int x = x_lo; x <= x_hi; ++x) {
            const float dx = abs(float(x) - cx);
            if (!(dx < fF)) continue;
            const float wx = max(0.0f, 1.0f - dx / fF);
            const int xx = clamp(x, 0, int(q.w) - 1);
            const float wgt = wy * wx;
            device const float* p = row + 3 * xx;
            a0 += wgt * p[0];
            a1 += wgt * p[1];
            a2 += wgt * p[2];
        }
    }
    const float inv = 1.0f / (fF * fF);
    device float* o = out + 3u * (gid.y * q.cw + gid.x);
    o[0] = a0 * inv;
    o[1] = a1 * inv;
    o[2] = a2 * inv;
}

/* spk_fx_halation_up_mix
 *   m := bilinear coarse at u = (x - c) / F, v = (y - c) / F, indices clamped; + fine if present
 *   out := a raw + b m
 */
kernel void spk_fx_halation_up_mix(device const float* raw [[buffer(0)]],
                                   device const float* coarse [[buffer(1)]],
                                   device const float* fine [[buffer(2)]],
                                   device float* out [[buffer(3)]],
                                   constant SpkFxTentArgs& q [[buffer(4)]],
                                   uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= q.w || gid.y >= q.h) return;
    const float fF = float(q.f);
    const float c = (fF - 1.0f) * 0.5f;
    const float u = (float(gid.x) - c) / fF;
    const float v = (float(gid.y) - c) / fF;
    const float fu0 = floor(u), fv0 = floor(v);
    const float tu = u - fu0, tv = v - fv0;
    const int iu0 = int(fu0), iv0 = int(fv0);
    const int cwm = int(q.cw) - 1, chm = int(q.ch) - 1;
    const uint u0 = uint(clamp(iu0, 0, cwm)), u1 = uint(clamp(iu0 + 1, 0, cwm));
    const uint v0 = uint(clamp(iv0, 0, chm)), v1 = uint(clamp(iv0 + 1, 0, chm));
    const uint o = 3u * (gid.y * q.w + gid.x);
    for (uint ch = 0; ch < 3u; ++ch) {
        const float c00 = coarse[3u * (v0 * q.cw + u0) + ch];
        const float c01 = coarse[3u * (v0 * q.cw + u1) + ch];
        const float c10 = coarse[3u * (v1 * q.cw + u0) + ch];
        const float c11 = coarse[3u * (v1 * q.cw + u1) + ch];
        const float top = (1.0f - tu) * c00 + tu * c01;
        const float bot = (1.0f - tu) * c10 + tu * c11;
        float m = (1.0f - tv) * top + tv * bot;
        if (q.has_fine) m = m + fine[o + ch];
        out[o + ch] = q.a[ch] * raw[o + ch] + q.b[ch] * m;
    }
}
