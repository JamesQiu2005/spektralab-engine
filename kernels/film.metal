// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 Guangzhou Songzi Intelligent Technology Services Co., Ltd.

/* Film-side per-pixel kernels, named spk_film_<what>.
 *
 * binary32 under safe math with precise transcendental functions; the host does the float64 set-up
 */
#include "spk_common.h"
#include "film_shared.h"

using namespace metal;

namespace {

// ---- double-float helpers ---------------------------------------------------------------------
inline float2 two_sum(float a, float b) {
    const float s = a + b;
    const float bb = s - a;
    const float e = (a - (s - bb)) + (b - bb);
    return float2(s, e);
}
inline float2 dd_add(float2 a, float2 b) {
    const float2 s = two_sum(a.x, b.x);
    const float e = s.y + (a.y + b.y);
    const float h = s.x + e;
    const float l = e - (h - s.x);
    return float2(h, l);
}

// geometry_axis(f): on the hi part; hi <= 0: i0 = 0, t = 0; hi >= n-1: i0 = n-1, t = 0;
// else i0 = floor(hi), t = clamp((hi - i0) + lo, 0, 1); i1 = min(i0 + 1, n - 1).
inline void geometry_axis(float2 f, uint n, thread uint& i0, thread uint& i1, thread float& t) {
    const float hi = f.x;
    if (hi <= 0.0f) {
        i0 = 0;
        t = 0.0f;
    } else if (hi >= float(n - 1)) {
        i0 = n - 1;
        t = 0.0f;
    } else {
        const float fl = floor(hi);
        i0 = uint(fl);
        t = (hi - fl) + f.y;
        t = fmin(fmax(t, 0.0f), 1.0f);
    }
    i1 = min(i0 + 1, n - 1);
}

// ---- Mitchell-Netravali, B = C = 1/3 ---------------------------------------------
inline float mitchell(float u) {
    const float a = fabs(u);
    if (a < 1.0f) return (1.0f / 6.0f) * (7.0f * a * a * a - 12.0f * a * a + 16.0f / 3.0f);
    if (a < 2.0f) return (1.0f / 6.0f) * ((-7.0f / 3.0f) * a * a * a + 12.0f * a * a - 20.0f * a + 32.0f / 3.0f);
    return 0.0f;
}

// One axis of the lookup: tap indices (mirrored without repeating the end sample) and weights.
inline void mitchell_axis(float t, uint L, thread int* idx, thread float* w) {
    const float last = float(L - 1);
    float u = t * last;
    int base;
    float frac;
    if (u <= 0.0f) u = 0.0f;
    if (u >= last) {
        base = int(L) - 2;
        frac = 1.0f;
    } else {
        base = int(floor(u));
        frac = u - float(base);
    }
    const int n = int(L);
    for (int k = 0; k < 4; ++k) {
        int i = base - 1 + k;
        if (i < 0) i = -i;
        if (i >= n) i = 2 * (n - 1) - i;
        idx[k] = i;
    }
    w[0] = mitchell(frac + 1.0f);
    w[1] = mitchell(frac);
    w[2] = mitchell(frac - 1.0f);
    w[3] = mitchell(frac - 2.0f);
}

/* curve_lookup F(v; x, inv, y)
 *   v <= x[0] -> y[0]; v >= x[K-1] -> y[K-1]
 *   l := search_right(x, v) - 1; y[l] + (v - x[l]) inv[l] (y[l+1] - y[l])
 */
inline float curve_lookup(device const float* x, device const float* inv, device const float* y, uint K, float v) {
    if (v <= x[0]) return y[0];
    if (v >= x[K - 1]) return y[K - 1];
    uint lo = 0, hi = K;
    while (lo < hi) {
        const uint mid = lo + (hi - lo) / 2;
        if (x[mid] <= v) lo = mid + 1;
        else hi = mid;
    }
    // v > x[0] gives lo >= 1 for every finite v; a NaN v ends with lo = 0 and gives NaN below.
    const uint l = lo == 0 ? 0 : lo - 1;
    const float t = (v - x[l]) * inv[l];
    return y[l] + t * (y[l + 1] - y[l]);
}

}  // namespace

/* spk_film_decode, per element
 *   sRGB      v <= 0.040449936 ? v / 12.92 : ((v + 0.055) / 1.055)^2.4
 *   ProPhoto  v < 16/512 ? v / 16 : v^1.8
 *   Adobe     v^(563/256)
 *   BT.709    v < bp ? v / 4.5 : ((v + 0.099) / 1.099)^(1/0.45)
 *   identity  v
 */
kernel void spk_film_decode(device const float* in [[buffer(0)]], device float* out [[buffer(1)]],
                            constant SpkFilmDecodeArgs& q [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= q.n) return;
    const float v = in[gid];
    float r = v;
    switch (q.mode) {
        case SPK_FILM_DECODE_SRGB:
            if (v <= 0.040449936f) {
                r = v / 12.92f;
            } else {
                const float u = (v + 0.055f) / 1.055f;
                r = sign(u) * precise::pow(fabs(u), 2.4f);
            }
            break;
        case SPK_FILM_DECODE_PROPHOTO:
            r = v < 16.0f / 512.0f ? v / 16.0f : precise::pow(v, 1.8f);
            break;
        case SPK_FILM_DECODE_ADOBE:
            r = precise::pow(v, 563.0f / 256.0f);
            break;
        case SPK_FILM_DECODE_BT709:
            if (v < q.bt709_bp) {
                r = v / 4.5f;
            } else {
                const float u = (v + 0.099f) / 1.099f;
                r = sign(u) * precise::pow(fabs(u), q.bt709_exp);
            }
            break;
        default:
            break;
    }
    out[gid] = r;
}

/* spk_film_geometry: bilinear at (f_x, f_y)
 *   f_x := c_x + (col[ox].x + row[oy].x), double-float sums of the host's binary64 terms; f_y likewise
 *   top := a (1-tx) + b tx; bot := d (1-tx) + e tx; out := top (1-ty) + bot ty
 */
kernel void spk_film_geometry(device const float* src [[buffer(0)]], device float* dst [[buffer(1)]],
                              device const float* col [[buffer(2)]], device const float* row [[buffer(3)]],
                              constant SpkFilmGeometryArgs& q [[buffer(4)]], uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= q.dst_w || gid.y >= q.dst_h) return;
    const uint ox = gid.x, oy = gid.y;
    const float2 sx = dd_add(float2(col[4 * ox], col[4 * ox + 1]), float2(row[4 * oy], row[4 * oy + 1]));
    const float2 sy = dd_add(float2(col[4 * ox + 2], col[4 * ox + 3]), float2(row[4 * oy + 2], row[4 * oy + 3]));
    const float2 fx = dd_add(float2(q.cx_hi, q.cx_lo), sx);
    const float2 fy = dd_add(float2(q.cy_hi, q.cy_lo), sy);
    uint x0, x1, y0, y1;
    float tx, ty;
    geometry_axis(fx, q.src_w, x0, x1, tx);
    geometry_axis(fy, q.src_h, y0, y1, ty);
    device const float* pa = src + 3 * (size_t(y0) * q.src_w + x0);
    device const float* pb = src + 3 * (size_t(y0) * q.src_w + x1);
    device const float* pd = src + 3 * (size_t(y1) * q.src_w + x0);
    device const float* pe = src + 3 * (size_t(y1) * q.src_w + x1);
    device float* o = dst + 3 * (size_t(oy) * q.dst_w + ox);
    for (int c = 0; c < 3; ++c) {
        const float top = pa[c] * (1.0f - tx) + pb[c] * tx;
        const float bot = pd[c] * (1.0f - tx) + pe[c] * tx;
        o[c] = top * (1.0f - ty) + bot * ty;
    }
}

// Auto exposure, exposure compensation and the boost's M = 0 case: every element times one f32 gain.
kernel void spk_film_scale(device float* p [[buffer(0)]], constant SpkFilmScaleArgs& q [[buffer(1)]],
                           uint gid [[thread_position_in_grid]]) {
    if (gid >= q.n) return;
    p[gid] = p[gid] * q.g;
}

/* spk_film_upsample: one pixel of RGB -> film-layer exposure H
 *   (X, Y, Z) := A rgb; b := X + Y + Z
 *   x := X / max(b, 1e-10); y := Y / max(b, 1e-10); o := 1 - x
 *   t0 := clamp(o^2, 0, 1); t1 := clamp(y / max(o, 1e-10), 0, 1)
 *   m := bicubic Mitchell-Netravali (B = C = 1/3) of table at (t0, t1), weights normalised
 *   H := b m; b NaN -> 0
 */
kernel void spk_film_upsample(device const float* rgb [[buffer(0)]], device float* h [[buffer(1)]],
                              device const float* table [[buffer(2)]], constant SpkFilmUpsampleArgs& q [[buffer(3)]],
                              uint gid [[thread_position_in_grid]]) {
    if (gid >= q.n) return;
    const float r = rgb[3 * gid], g = rgb[3 * gid + 1], b = rgb[3 * gid + 2];
    const float X = (q.a[0] * r + q.a[1] * g) + q.a[2] * b;
    const float Y = (q.a[3] * r + q.a[4] * g) + q.a[5] * b;
    const float Z = (q.a[6] * r + q.a[7] * g) + q.a[8] * b;
    const float bs = (X + Y) + Z;
    const float d = bs > 1e-10f ? bs : 1e-10f;
    const float x = X / d, y = Y / d, o = 1.0f - x;
    const float t0 = fmin(fmax(o * o, 0.0f), 1.0f);
    const float t1 = fmin(fmax(y / (o > 1e-10f ? o : 1e-10f), 0.0f), 1.0f);
    const float bp = isnan(bs) ? 0.0f : bs;

    int ia[4], ib[4];
    float wa[4], wb[4];
    mitchell_axis(t0, q.L, ia, wa);
    mitchell_axis(t1, q.L, ib, wb);
    float s = 0.0f, a0 = 0.0f, a1 = 0.0f, a2 = 0.0f;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            const float w = wa[i] * wb[j];
            device const float* t = table + 3 * (size_t(ia[i]) * q.L + size_t(ib[j]));
            s += w;
            a0 += w * t[0];
            a1 += w * t[1];
            a2 += w * t[2];
        }
    }
    if (s != 0.0f) {
        a0 = a0 / s;
        a1 = a1 / s;
        a2 = a2 / s;
    }
    h[3 * gid] = bp * a0;
    h[3 * gid + 1] = bp * a1;
    h[3 * gid + 2] = bp * a2;
}

/* spk_film_boost, per element: v > x0: v + k (exp(z) - z - 1), z = a (v - x0) r */
kernel void spk_film_boost(device float* p [[buffer(0)]], constant SpkFilmBoostArgs& q [[buffer(1)]],
                           uint gid [[thread_position_in_grid]]) {
    if (gid >= q.n) return;
    const float v = p[gid];
    if (v <= q.x0) return;
    const float delta = (v - q.x0) * q.r;
    const float z = q.a * delta;
    const float w = (precise::exp(z) - z) - 1.0f;
    p[gid] = v + q.k * w;
}

/* spk_film_log: p := log10(max(p, 0) + 1e-10) */
kernel void spk_film_log(device float* p [[buffer(0)]], constant SpkFilmLogArgs& q [[buffer(1)]],
                         uint gid [[thread_position_in_grid]]) {
    if (gid >= q.n) return;
    p[gid] = precise::log10(fmax(p[gid], 0.0f) + 1e-10f);
}

/* spk_film_curves: out_c := F(in_c; x, inv, y or y0); out may be in */
kernel void spk_film_curves(device const float* in [[buffer(0)]], device float* out [[buffer(1)]],
                            device const float* curves [[buffer(2)]], constant SpkFilmCurvesArgs& q [[buffer(3)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid >= q.n) return;
    const uint K = q.K;
    device const float* x = curves;
    device const float* inv = curves + K;
    device const float* y = curves + 2 * K + (q.table ? 3 * K : 0);
    for (uint c = 0; c < 3; ++c) {
        const float v = in[3 * gid + c];
        out[3 * gid + c] = curve_lookup(x, inv, y + c * K, K, v);
    }
}

/* spk_film_coupler_source, in place: first-development density -> chi
 *   s_c := positive ? dmax_c - D_c : D_c
 *   chi_m := sum_c s_c M[c][m]
 */
kernel void spk_film_coupler_source(device float* p [[buffer(0)]], constant SpkFilmCouplerArgs& q [[buffer(1)]],
                                    uint gid [[thread_position_in_grid]]) {
    if (gid >= q.n) return;
    float s[3];
    for (uint c = 0; c < 3; ++c) {
        const float d = p[3 * gid + c];
        s[c] = q.positive ? q.dmax[c] - d : d;
    }
    for (uint m = 0; m < 3; ++m) p[3 * gid + m] = (s[0] * q.m[m] + s[1] * q.m[3 + m]) + s[2] * q.m[6 + m];
}

/* spk_film_coupler_apply: L (in place) -> D := F(L - blurred chi; x, inv, y0) */
kernel void spk_film_coupler_apply(device float* l [[buffer(0)]], device const float* chi [[buffer(1)]],
                                   device const float* curves [[buffer(2)]],
                                   constant SpkFilmCouplerApplyArgs& q [[buffer(3)]],
                                   uint gid [[thread_position_in_grid]]) {
    if (gid >= q.n) return;
    const uint K = q.K;
    device const float* x = curves;
    device const float* inv = curves + K;
    device const float* y0 = curves + 5 * K;
    for (uint c = 0; c < 3; ++c) {
        const float l0 = l[3 * gid + c] - chi[3 * gid + c];
        l[3 * gid + c] = curve_lookup(x, inv, y0 + c * K, K, l0);
    }
}
