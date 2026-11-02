// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 Guangzhou Songzi Intelligent Technology Services Co., Ltd.

/* Print-side kernels, named spk_print_<what>: enlarger, paper, scan, levels, XYZ -> RGB, CAM16 compression,
 * EDR, transfer function and encode, preview table.
 *
 * float32 under safe math
 * every constant that is not a literal: the host's float64 set-up, narrowed once
 * except CAM16: the kernel gets the output transform's k numbers and computes
 *   c z, 1/(c z), F_L^(1/4), 100/F_L and the chroma constant from them in float32
 * 10^x is evaluated as 2^(x * 3.321928094887362)
 * sums over wavelength run serially from 380 to 780 nm
 */
#include "spk_common.h"
#include "print_shared.h"

using namespace metal;

namespace {

constant float kLog2Of10 = 3.321928094887362f;
constant float kTiny = 1e-10f;

inline float sgn(float v) { return v > 0.0f ? 1.0f : (v < 0.0f ? -1.0f : 0.0f); }
inline float spow(float v, float p) { return sgn(v) * pow(fabs(v), p); }

inline uint pixel_index(uint2 gid, uint w) { return 3u * (gid.y * w + gid.x); }

// knee(d; tau, lambda, p) = tau + (lambda - tau) * [x / (1 + x^p)^(1/p)]: the compressed
// fraction is formed first, then scaled (and, under safe math, fused with the + tau).
inline float knee(float d, float tau, float lim, float p) {
    if (d <= tau) return d;
    const float x = (d - tau) / (lim - tau);
    return tau + (lim - tau) * (x / pow(1.0f + pow(x, p), 1.0f / p));
}

// Post-adaptation non-linearity: P(v) = 400 sgn(v) f / (27.13 + f) + 0.1, f = (F_L |v| / 100)^0.42.
inline float post_adapt(float v, float f_l) {
    const float f = pow((f_l * fabs(v)) / 100.0f, 0.42f);
    return ((400.0f * sgn(v)) * f) / (27.13f + f) + 0.1f;
}

inline float3 mat_apply(constant const float* m, float3 v) {
    return float3((m[0] * v.x + m[1] * v.y) + m[2] * v.z,
                  (m[3] * v.x + m[4] * v.y) + m[5] * v.z,
                  (m[6] * v.x + m[7] * v.y) + m[8] * v.z);
}

// Per-pixel constants computed in float32 from the k numbers:
// c z, 1/(c z), F_L^(1/4), (1.64 - 0.29^n)^0.73 and 100/F_L.
struct Cam16Derived {
    float cz, inv_cz, f_l_qrt, chroma_k, hundred_over_f_l;
};
inline Cam16Derived cam16_derived(constant SpkPrintCam16Args& q) {
    Cam16Derived d;
    d.cz = q.c * q.z;
    d.inv_cz = 1.0f / d.cz;
    d.f_l_qrt = pow(q.f_l, 0.25f);
    d.chroma_k = pow(1.64f - pow(0.29f, q.n), 0.73f);
    d.hundred_over_f_l = 100.0f / q.f_l;
    return d;
}

// Forward CAM16-UCS of XYZ with Y_white = 1. Returns (J', M', h).
inline float3 cam16_forward(float3 xyz, constant SpkPrintCam16Args& q, thread const Cam16Derived& k) {
    const float3 x100 = float3(100.0f * xyz.x, 100.0f * xyz.y, 100.0f * xyz.z);
    const float3 rgb = mat_apply(q.m16, x100);
    const float ra = post_adapt(q.d_rgb[0] * rgb.x, q.f_l);
    const float ga = post_adapt(q.d_rgb[1] * rgb.y, q.f_l);
    const float ba = post_adapt(q.d_rgb[2] * rgb.z, q.f_l);
    const float a = (ra - (12.0f * ga) / 11.0f) + ba / 11.0f;
    const float b = ((ra + ga) - 2.0f * ba) / 9.0f;
    const float h = atan2(b, a);
    const float et = 0.25f * (cos(h + 2.0f) + 3.8f);
    const float A = (((2.0f * ra + ga) + ba / 20.0f) - 0.305f) * q.n_bb;
    const float ratio = A / q.a_w;
    const float J = (100.0f * sgn(ratio)) * pow(fabs(ratio), k.cz);
    const float den = (ra + ga) + (21.0f * ba) / 20.0f;
    const float t = den != 0.0f ? (((((50000.0f / 13.0f) * q.n_c) * q.n_cb) * et) * sqrt(a * a + b * b)) / den : 0.0f;
    const float C = t > 0.0f ? (pow(t, 0.9f) * sqrt(fabs(J) / 100.0f)) * k.chroma_k : 0.0f;
    const float M = C * k.f_l_qrt;
    const float Mp = log(1.0f + 0.0228f * M) / 0.0228f;
    const float Jp = (1.7f * J) / (1.0f + 0.007f * J);
    return float3(Jp, Mp, h);
}

// Inverse CAM16-UCS, per-pixel variant: guarded at |v| >= 400, the 8-digit inverse matrix.
inline float3 cam16_inverse(float Jp, float Mp, float h, constant SpkPrintCam16Args& q, thread const Cam16Derived& k) {
    const float M = (exp(0.0228f * Mp) - 1.0f) / 0.0228f;
    const float J = Jp / (1.7f - 0.007f * Jp);
    const float C = M / k.f_l_qrt;
    const float A = (q.a_w * sgn(J)) * pow(fabs(J / 100.0f), k.inv_cz);
    const float sq = sqrt(fabs(J) / 100.0f);
    const float t = (sq > 0.0f && C > 0.0f) ? pow(C / (sq * k.chroma_k), 1.0f / 0.9f) : 0.0f;
    const float et = 0.25f * (cos(h + 2.0f) + 3.8f);
    const float p2 = A / q.n_bb + 0.305f;
    const float p3 = 21.0f / 20.0f;
    float a = 0.0f, b = 0.0f;
    if (t != 0.0f) {
        const float p1 = ((((50000.0f / 13.0f) * q.n_c) * q.n_cb) * et) / t;
        const float sh = sin(h), ch = cos(h);
        if (fabs(sh) >= fabs(ch)) {
            const float p4 = p1 / sh;
            b = ((p2 * (2.0f + p3)) * (460.0f / 1403.0f)) /
                (((p4 + ((2.0f + p3) * (220.0f / 1403.0f)) * (ch / sh)) - 27.0f / 1403.0f) + p3 * (6300.0f / 1403.0f));
            a = (b * ch) / sh;
        } else {
            const float p5 = p1 / ch;
            a = ((p2 * (2.0f + p3)) * (460.0f / 1403.0f)) /
                ((p5 + (2.0f + p3) * (220.0f / 1403.0f)) - (27.0f / 1403.0f - p3 * (6300.0f / 1403.0f)) * (sh / ch));
            b = (a * sh) / ch;
        }
    }
    const float ra = ((460.0f * p2 + 451.0f * a) + 288.0f * b) / 1403.0f;
    const float ga = ((460.0f * p2 - 891.0f * a) - 261.0f * b) / 1403.0f;
    const float ba = ((460.0f * p2 - 220.0f * a) - 6300.0f * b) / 1403.0f;
    const float xa[3] = {ra, ga, ba};
    float rc[3];
    for (int i = 0; i < 3; ++i) {
        const float v = xa[i] - 0.1f;
        const float av = fabs(v);
        const float beta = av >= 400.0f ? 0.0f : (27.13f * av) / (400.0f - av);
        rc[i] = ((k.hundred_over_f_l * sgn(v)) * pow(beta, 1.0f / 0.42f)) / q.d_rgb[i];
    }
    const float3 x100 = mat_apply(q.m16_inv, float3(rc[0], rc[1], rc[2]));
    return float3(x100.x / 100.0f, x100.y / 100.0f, x100.z / 100.0f);
}

/* cctf_encode(v, mode)
 *   sRGB      v <= 0.0031308 ? 12.92 v : 1.055 v^(1/2.4) - 0.055
 *   ProPhoto  v < 1/512 ? 16 v : v^(1/1.8)
 *   Adobe     v^(256/563)
 *   BT.709    v < 0.018 ? 4.5 v : 1.099 v^0.45 - 0.099
 *   identity  v
 */
inline float cctf_encode(float v, uint mode) {
    switch (mode) {
        case SPK_PRINT_CCTF_SRGB:
            return v <= 0.0031308f ? 12.92f * v : 1.055f * spow(v, 0.41666666666666669f) - 0.055f;
        case SPK_PRINT_CCTF_PROPHOTO:
            return v < 0.001953125f ? 16.0f * v : pow(v, 0.55555555555555558f);
        case SPK_PRINT_CCTF_ADOBE:
            return pow(v, 0.45470692717584371f);  // bare power: a negative channel gives NaN
        case SPK_PRINT_CCTF_BT709:
            return v < 0.018f ? 4.5f * v : 1.099f * spow(v, 0.45f) - 0.099f;
        default:
            return v;
    }
}

// 16-bit encoding: NaN -> 0, else floor(clamp(v, 0, 1) * 65535 + 0.5).
inline ushort encode16(float v) {
    if (isnan(v)) return 0;
    const float c = min(max(v, 0.0f), 1.0f);
    return ushort(floor(c * 65535.0f + 0.5f));
}

}  // namespace

/* spk_print_spectral: enlarger and scan, per pixel
 *   D(l) := sum_c d_c eps_c(l) + beta(l); T(l) := 2^(-D(l) log2 10); R_k := sum_l T(l) wt_k(l), ascending l
 *   SCAN:          out_k := max(R_k, 0) + 1e-10
 *   LOG_EXPOSURE:  out_k := log10(max(R_k g_k + o_k, 0) + 1e-10)
 */
kernel void spk_print_spectral(device const float* in [[buffer(0)]],
                               device float* out [[buffer(1)]],
                               constant SpkPrintSpectralArgs& a [[buffer(2)]],
                               uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= a.w || gid.y >= a.h) return;
    const uint i = pixel_index(gid, a.w);
    const float d0 = in[i], d1 = in[i + 1], d2 = in[i + 2];
    float r0 = 0.0f, r1 = 0.0f, r2 = 0.0f;
    for (uint l = 0; l < SPK_PRINT_NL; ++l) {
        const float D = ((d0 * a.eps[3 * l] + d1 * a.eps[3 * l + 1]) + d2 * a.eps[3 * l + 2]) + a.beta[l];
        const float T = exp2(-D * kLog2Of10);
        r0 += T * a.wt[3 * l];
        r1 += T * a.wt[3 * l + 1];
        r2 += T * a.wt[3 * l + 2];
    }
    if (a.mode == SPK_PRINT_MODE_SCAN) {
        out[i] = max(r0, 0.0f) + kTiny;
        out[i + 1] = max(r1, 0.0f) + kTiny;
        out[i + 2] = max(r2, 0.0f) + kTiny;
    } else {
        out[i] = log10(max(r0 * a.g[0] + a.o[0], 0.0f) + kTiny);
        out[i + 1] = log10(max(r1 * a.g[1] + a.o[1], 0.0f) + kTiny);
        out[i + 2] = log10(max(r2 * a.g[2] + a.o[2], 0.0f) + kTiny);
    }
}

/* spk_print_gain: l := log10(max(2^(l log2 10) k, 0) + 1e-10) */
kernel void spk_print_gain(device float* p [[buffer(0)]],
                           constant SpkPrintGainArgs& a [[buffer(1)]],
                           uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= a.w || gid.y >= a.h) return;
    const uint i = pixel_index(gid, a.w);
    for (uint c = 0; c < 3; ++c) p[i + c] = log10(max(exp2(p[i + c] * kLog2Of10) * a.k, 0.0f) + kTiny);
}

/* spk_print_paper: clamped linear lookup per channel; NaN stays NaN */
kernel void spk_print_paper(device float* p [[buffer(0)]],
                            device const float* tab [[buffer(1)]],
                            constant SpkPrintCurveArgs& a [[buffer(2)]],
                            uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= a.w || gid.y >= a.h) return;
    const uint i = pixel_index(gid, a.w);
    const uint K = a.K;
    device const float* x = tab;
    device const float* r = tab + K;
    device const float* y = tab + 2 * K;
    for (uint c = 0; c < 3; ++c) {
        const float v = p[i + c];
        float out;
        if (isnan(v)) {
            out = v;
        } else if (v <= x[0]) {
            out = y[c];
        } else if (v >= x[K - 1]) {
            out = y[3 * (K - 1) + c];
        } else {
            uint lo = 0, hi = K;
            while (lo < hi) {
                const uint mid = lo + (hi - lo) / 2;
                if (x[mid] <= v) lo = mid + 1; else hi = mid;
            }
            const uint j = lo - 1;
            const float y0 = y[3 * j + c];
            out = y0 + ((v - x[j]) * r[j]) * (y[3 * (j + 1) + c] - y0);
        }
        p[i + c] = out;
    }
}

/* spk_print_levels: Y' := clamp(m Y + q, 0, 1); XYZ := XYZ Y' / (Y + 1e-10) */
kernel void spk_print_levels(device float* p [[buffer(0)]],
                             constant SpkPrintLevelsArgs& a [[buffer(1)]],
                             uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= a.w || gid.y >= a.h) return;
    const uint i = pixel_index(gid, a.w);
    const float Y = p[i + 1];
    const float Yp = min(max(a.m * Y + a.q, 0.0f), 1.0f);
    const float s = Yp / (Y + kTiny);
    p[i] = s * p[i];
    p[i + 1] = s * Y;
    p[i + 2] = s * p[i + 2];
}

/* spk_print_matrix: RGB := M XYZ */
kernel void spk_print_matrix(device float* p [[buffer(0)]],
                             constant SpkPrintMatrixArgs& a [[buffer(1)]],
                             uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= a.w || gid.y >= a.h) return;
    const uint i = pixel_index(gid, a.w);
    const float3 v = mat_apply(a.m, float3(p[i], p[i + 1], p[i + 2]));
    p[i] = v.x;
    p[i + 1] = v.y;
    p[i + 2] = v.z;
}

/* spk_print_gamut: CAM16-UCS gamut compression, per pixel
 *   (J', M', h) := forward(m_to_xyz rgb)
 *   lightness: Ln := J' / J_white; Ln > lc0: J' := knee(Ln; lc0, lc1, lc2) J_white
 *   cs := max(bilinear C_max(J', h), 1e-9), hue wrapping
 *   chroma: d := M' / cs; d > 0: d := knee(d; knee0, knee1, knee2)
 *   rgb := m_to_rgb inverse(J', d cs, h)                 -- h unchanged
 */
kernel void spk_print_gamut(device float* p [[buffer(0)]],
                            device const float* cmax [[buffer(1)]],
                            constant SpkPrintCam16Args& q [[buffer(2)]],
                            uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= q.w || gid.y >= q.h) return;
    const uint i = pixel_index(gid, q.w);
    const float3 xyz = mat_apply(q.m_to_xyz, float3(p[i], p[i + 1], p[i + 2]));
    const Cam16Derived k = cam16_derived(q);
    const float3 f = cam16_forward(xyz, q, k);
    float Jp = f.x;
    const float Mp = f.y;
    const float h = f.z;

    // lightness
    if (q.lc_active != 0u) {
        float Ln = Jp / q.j_white;
        if (Ln > q.lc[0]) {
            Ln = knee(Ln, q.lc[0], q.lc[1], q.lc[2]);
            Jp = Ln * q.j_white;
        }
    }

    // C_max(J', h), bilinear in the 64 x 720 table
    const float Jc = min(max(Jp, q.j_first), q.j_last);
    const float u = ((Jc - q.j_first) / (q.j_last - q.j_first)) * float(q.rows - 1);
    const int l0 = clamp(int(floor(u)), 0, int(q.rows) - 2);
    const float fl = u - float(l0);
    const float v = (h - q.h_first) / q.h_step;
    const float vf = floor(v);
    const int cols = int(q.cols);
    int h0 = int(vf) % cols;
    if (h0 < 0) h0 += cols;
    const int h1 = h0 == cols - 1 ? 0 : h0 + 1;
    const float fh = v - vf;
    const float t00 = cmax[l0 * cols + h0], t01 = cmax[l0 * cols + h1];
    const float t10 = cmax[(l0 + 1) * cols + h0], t11 = cmax[(l0 + 1) * cols + h1];
    const float cm = (1.0f - fl) * ((1.0f - fh) * t00 + fh * t01) + fl * ((1.0f - fh) * t10 + fh * t11);
    const float cs = max(cm, 1e-9f);

    // chroma
    float d = Mp / cs;
    if (d > 0.0f) d = knee(d, q.knee[0], q.knee[1], q.knee[2]);
    const float Mn = d * cs;

    // inverse at the same hue, back to RGB
    const float3 back = cam16_inverse(Jp, Mn, h, q, k);
    const float3 rgb = mat_apply(q.m_to_rgb, back);
    p[i] = rgb.x;
    p[i + 1] = rgb.y;
    p[i + 2] = rgb.z;
}

/* spk_print_edr: tone map on luminance
 *   y := max(r_y . rgb, 0)
 *   y > 0 and outside [toe_y, shoulder_y]:
 *     u := clamp((log2 y - domain_lo) / span, 0, 1) (N - 1)
 *     y' := 2^(lerp of target_log2_y at u)
 *   else y' := y
 *   rgb := rgb y' / (y + 1e-10)
 */
kernel void spk_print_edr(device float* p [[buffer(0)]],
                          device const float* tab [[buffer(1)]],
                          constant SpkPrintEdrArgs& a [[buffer(2)]],
                          uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= a.w || gid.y >= a.h) return;
    const uint i = pixel_index(gid, a.w);
    const float r = p[i], g = p[i + 1], b = p[i + 2];
    const float y = max((a.r_y[0] * r + a.r_y[1] * g) + a.r_y[2] * b, 0.0f);
    float yp = y;
    if (y > 0.0f && (y < a.toe_y || y > a.shoulder_y)) {
        const uint N = max(a.n, 2u);
        const float u = min(max((log2(y) - a.domain_lo) / a.domain_span, 0.0f), 1.0f) * float(N - 1);
        const uint i0 = min(uint(floor(u)), N - 2);
        const float t = u - float(i0);
        yp = exp2((1.0f - t) * tab[i0] + t * tab[i0 + 1]);
    }
    const float den = y + kTiny;
    p[i] = (r * yp) / den;
    p[i + 1] = (g * yp) / den;
    p[i + 2] = (b * yp) / den;
}

/* spk_print_encode: [m_nid rgb, transfer function], then 16-bit; alpha = 65535 */
kernel void spk_print_encode(device const float* p [[buffer(0)]],
                             device ushort* out [[buffer(1)]],
                             constant SpkPrintEncodeArgs& a [[buffer(2)]],
                             uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= a.w || gid.y >= a.h) return;
    const uint i = pixel_index(gid, a.w);
    float3 v = float3(p[i], p[i + 1], p[i + 2]);
    if (a.cctf_active != 0u) {
        const float3 m = mat_apply(a.m_nid, v);
        v = float3(cctf_encode(m.x, a.cctf_mode), cctf_encode(m.y, a.cctf_mode), cctf_encode(m.z, a.cctf_mode));
    }
    const uint o = 4u * (gid.y * a.stride + gid.x);
    out[o] = encode16(v.x);
    out[o + 1] = encode16(v.y);
    out[o + 2] = encode16(v.z);
    out[o + 3] = 65535;
}

/* spk_print_lut: spk_preview_stock_lut, per pixel
 *   c_ch := clamp((D_ch - lo_ch) s_ch, 0, 1) (S - 1)
 *   i0 := floor(c); i1 := min(i0 + 1, S - 1); f := c - i0
 *   interpolate along k, then j, then i; 16-bit encode; alpha = 65535
 */
kernel void spk_print_lut(device const float* neg [[buffer(0)]],
                          device const float* lut [[buffer(1)]],
                          device ushort* out [[buffer(2)]],
                          constant SpkPrintLutArgs& a [[buffer(3)]],
                          uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= a.w || gid.y >= a.h) return;
    const uint i = pixel_index(gid, a.w);
    const uint S = a.size;
    uint i0[3], i1[3];
    float f[3];
    for (uint c = 0; c < 3; ++c) {
        const float u = min(max((neg[i + c] - a.lo[c]) * a.s[c], 0.0f), 1.0f) * float(S - 1);
        const float fl = floor(u);
        i0[c] = uint(fl);
        f[c] = u - fl;
        i1[c] = min(i0[c] + 1, S - 1);
    }
    const uint o = 4u * (gid.y * a.stride + gid.x);
    for (uint ch = 0; ch < 3; ++ch) {
        // along axis 2 (k) first, then axis 1 (j), then axis 0 (i)
        float c[2][2];
        for (uint di = 0; di < 2; ++di) {
            for (uint dj = 0; dj < 2; ++dj) {
                const uint ii = di ? i1[0] : i0[0];
                const uint jj = dj ? i1[1] : i0[1];
                const uint base = (ii * S + jj) * S;
                c[di][dj] = lut[(base + i0[2]) * 3 + ch] * (1.0f - f[2]) + lut[(base + i1[2]) * 3 + ch] * f[2];
            }
        }
        const float v = (c[0][0] * (1.0f - f[1]) + c[0][1] * f[1]) * (1.0f - f[0]) +
                        (c[1][0] * (1.0f - f[1]) + c[1][1] * f[1]) * f[0];
        out[o + ch] = encode16(v);
    }
    out[o + 3] = 65535;
}
