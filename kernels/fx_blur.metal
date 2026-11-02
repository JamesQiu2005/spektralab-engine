// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 Guangzhou Songzi Intelligent Technology Services Co., Ltd.

/* Kernels, named spk_fx_blur_<what>: FIR and recursive Gaussian passes, mixtures, linear combinations.
 *
 * planes: interleaved float32 x 3
 * FIR:       taps -R..R in increasing offset order, float accumulator from 0, half-sample reflected edges,
 *            vertical then horizontal
 * recursive: Young-van Vliet third order, one thread per (line, channel), state in double-float
 *            (hi + lo, about 48 significant bits), float storage of x, u, y
 */
#include "spk_common.h"
#include "fx_shared.h"

using namespace metal;

// ---- FIR ----------------------------------------------------------------------------------------

/* spk_fx_blur_fir_v: o_c := sum_i taps_c[i] in_c(x, reflect(y + i)) */
kernel void spk_fx_blur_fir_v(device const float* in [[buffer(0)]],
                              device float* out [[buffer(1)]],
                              constant SpkFxFirArgs& q [[buffer(2)]],
                              uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= q.w || gid.y >= q.h) return;
    const int r = int(q.r);
    const int h = int(q.h);
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f;
    for (int i = -r; i <= r; ++i) {
        const int yy = spk::reflect_index(int(gid.y) + i, h);
        device const float* p = in + 3u * (uint(yy) * q.w + gid.x);
        const uint t = uint(i + r);
        acc0 += q.taps[t] * p[0];
        acc1 += q.taps[SPK_FX_FIR_STRIDE + t] * p[1];
        acc2 += q.taps[2u * SPK_FX_FIR_STRIDE + t] * p[2];
    }
    device float* o = out + 3u * (gid.y * q.w + gid.x);
    o[0] = acc0;
    o[1] = acc1;
    o[2] = acc2;
}

/* spk_fx_blur_fir_h: v_c := sum_i taps_c[i] in_c(reflect(x + i), y)
 *   ACC_NONE: out := v; ACC_ZERO: out := 0 + w v; ACC_PLANE: out := acc + w v
 */
kernel void spk_fx_blur_fir_h(device const float* in [[buffer(0)]],
                              device float* out [[buffer(1)]],
                              device const float* accp [[buffer(2)]],
                              constant SpkFxFirArgs& q [[buffer(3)]],
                              uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= q.w || gid.y >= q.h) return;
    const int r = int(q.r);
    const int w = int(q.w);
    device const float* row = in + 3u * gid.y * q.w;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f;
    for (int i = -r; i <= r; ++i) {
        const int xx = spk::reflect_index(int(gid.x) + i, w);
        device const float* p = row + 3 * xx;
        const uint t = uint(i + r);
        acc0 += q.taps[t] * p[0];
        acc1 += q.taps[SPK_FX_FIR_STRIDE + t] * p[1];
        acc2 += q.taps[2u * SPK_FX_FIR_STRIDE + t] * p[2];
    }
    const uint o = 3u * (gid.y * q.w + gid.x);
    if (q.acc_mode == SPK_FX_ACC_NONE) {
        out[o] = acc0;
        out[o + 1] = acc1;
        out[o + 2] = acc2;
    } else if (q.acc_mode == SPK_FX_ACC_ZERO) {
        out[o] = 0.0f + q.weight[0] * acc0;
        out[o + 1] = 0.0f + q.weight[1] * acc1;
        out[o + 2] = 0.0f + q.weight[2] * acc2;
    } else {
        const float a0 = accp[o], a1 = accp[o + 1], a2 = accp[o + 2];
        out[o] = a0 + q.weight[0] * acc0;
        out[o + 1] = a1 + q.weight[1] * acc1;
        out[o + 2] = a2 + q.weight[2] * acc2;
    }
}

// ---- double-float arithmetic (error-free transforms; needs fma and no reassociation) ------------

struct DF {
    float hi, lo;
};

static inline DF df_two_sum(float a, float b) {
    const float s = a + b;
    const float bb = s - a;
    const float e = (a - (s - bb)) + (b - bb);
    return {s, e};
}
static inline DF df_quick(float a, float b) {
    const float s = a + b;
    const float e = b - (s - a);
    return {s, e};
}
static inline DF df_add(DF a, DF b) {
    DF s = df_two_sum(a.hi, b.hi);
    DF t = df_two_sum(a.lo, b.lo);
    s.lo += t.hi;
    s = df_quick(s.hi, s.lo);
    s.lo += t.lo;
    return df_quick(s.hi, s.lo);
}
static inline DF df_mul(DF a, DF b) {
    const float p = a.hi * b.hi;
    float e = fma(a.hi, b.hi, -p);
    e = fma(a.hi, b.lo, e);
    e = fma(a.lo, b.hi, e);
    return df_quick(p, e);
}
static inline DF df_mul_f(DF a, float b) {
    const float p = a.hi * b;
    float e = fma(a.hi, b, -p);
    e = fma(a.lo, b, e);
    return df_quick(p, e);
}
static inline float df_to_float(DF a) { return a.hi + a.lo; }

/* iir_line: one line; elements at base + i stride, i = 0..n-1
 *   forward:  w[-1..-3] := x[0]; w[i] := B x[i] + B1 w[i-1] + B2 w[i-2] + B3 w[i-3]; work[i] := float(w[i])
 *   backward: y[n..n+2] := u[n-1]; y[i] := B u[i] + B1 y[i+1] + B2 y[i+2] + B3 y[i+3]
 *   dst[i] := y, or 0 + w y, or acc + w y
 */
static inline void iir_line(device const float* src, device float* work, device float* dst,
                            device const float* accp, uint base, uint stride, uint n, uint c,
                            constant SpkFxIirArgs& q, uint acc_mode) {
    const DF B = {q.hi[4 * c + 0], q.lo[4 * c + 0]};
    const DF B1 = {q.hi[4 * c + 1], q.lo[4 * c + 1]};
    const DF B2 = {q.hi[4 * c + 2], q.lo[4 * c + 2]};
    const DF B3 = {q.hi[4 * c + 3], q.lo[4 * c + 3]};
    // forward: w[-1] = w[-2] = w[-3] = x[0]
    const float x0 = src[base];
    DF w1 = {x0, 0.0f}, w2 = w1, w3 = w1;
    for (uint i = 0; i < n; ++i) {
        const uint o = base + i * stride;
        const float x = src[o];
        DF t = df_mul_f(B, x);
        t = df_add(t, df_mul(B1, w1));
        t = df_add(t, df_mul(B2, w2));
        t = df_add(t, df_mul(B3, w3));
        w3 = w2;
        w2 = w1;
        w1 = t;
        work[o] = df_to_float(t);
    }
    // backward: y[n] = y[n+1] = y[n+2] = u[n-1]
    const float un = work[base + (n - 1) * stride];
    DF y1 = {un, 0.0f}, y2 = y1, y3 = y1;
    const float wgt = q.weight[c];
    for (uint k = n; k-- > 0;) {
        const uint o = base + k * stride;
        const float u = work[o];
        DF t = df_mul_f(B, u);
        t = df_add(t, df_mul(B1, y1));
        t = df_add(t, df_mul(B2, y2));
        t = df_add(t, df_mul(B3, y3));
        y3 = y2;
        y2 = y1;
        y1 = t;
        const float y = df_to_float(t);
        if (acc_mode == SPK_FX_ACC_NONE) {
            dst[o] = y;
        } else if (acc_mode == SPK_FX_ACC_ZERO) {
            dst[o] = 0.0f + wgt * y;
        } else {
            dst[o] = accp[o] + wgt * y;
        }
    }
}

/* spk_fx_blur_iir_h: along rows; thread t = 3 y + c; forward result stored in out; unmasked channel copied */
kernel void spk_fx_blur_iir_h(device const float* in [[buffer(0)]],
                              device float* out [[buffer(1)]],
                              constant SpkFxIirArgs& q [[buffer(2)]],
                              uint t [[thread_position_in_grid]]) {
    if (t >= 3u * q.h) return;
    const uint y = t / 3u, c = t % 3u;
    const uint base = 3u * y * q.w + c;
    if (((q.mask >> c) & 1u) == 0u) {
        for (uint x = 0; x < q.w; ++x) out[base + 3u * x] = in[base + 3u * x];
        return;
    }
    iir_line(in, out, out, out, base, 3u, q.w, c, q, SPK_FX_ACC_NONE);
}

/* spk_fx_blur_iir_v: along columns; thread t = 3 x + c
 *   forward result stored back into in (a scratch plane); result to out, with the optional accumulation
 */
kernel void spk_fx_blur_iir_v(device float* in [[buffer(0)]],
                              device float* out [[buffer(1)]],
                              device const float* accp [[buffer(2)]],
                              constant SpkFxIirArgs& q [[buffer(3)]],
                              uint t [[thread_position_in_grid]]) {
    if (t >= 3u * q.w) return;
    const uint c = t % 3u;
    const uint base = t;
    const uint stride = 3u * q.w;
    if (((q.mask >> c) & 1u) == 0u) {
        for (uint y = 0; y < q.h; ++y) out[base + y * stride] = in[base + y * stride];
        return;
    }
    iir_line(in, in, out, accp, base, stride, q.h, c, q, q.acc_mode);
}

// ---- pointwise helpers --------------------------------------------------------------------------

// out = a x + b y per channel (y reads as 0 when y_zero).
kernel void spk_fx_blur_lincomb(device const float* x [[buffer(0)]],
                                device const float* y [[buffer(1)]],
                                device float* out [[buffer(2)]],
                                constant SpkFxPointArgs& q [[buffer(3)]],
                                uint i [[thread_position_in_grid]]) {
    if (i >= q.n) return;
    for (uint c = 0; c < 3u; ++c) {
        const uint o = 3u * i + c;
        const float yv = q.y_zero ? 0.0f : y[o];
        out[o] = q.a[c] * x[o] + q.b[c] * yv;
    }
}

// out = x s + t per channel.
kernel void spk_fx_blur_affine(device const float* x [[buffer(0)]],
                               device float* out [[buffer(1)]],
                               constant SpkFxPointArgs& q [[buffer(2)]],
                               uint i [[thread_position_in_grid]]) {
    if (i >= q.n) return;
    for (uint c = 0; c < 3u; ++c) {
        const uint o = 3u * i + c;
        out[o] = x[o] * q.a[c] + q.b[c];
    }
}

// out = x * y elementwise.
kernel void spk_fx_blur_mul(device const float* x [[buffer(0)]],
                            device const float* y [[buffer(1)]],
                            device float* out [[buffer(2)]],
                            constant SpkFxPointArgs& q [[buffer(3)]],
                            uint i [[thread_position_in_grid]]) {
    if (i >= q.n) return;
    for (uint c = 0; c < 3u; ++c) {
        const uint o = 3u * i + c;
        out[o] = x[o] * y[o];
    }
}

kernel void spk_fx_blur_copy(device const float* x [[buffer(0)]],
                             device float* out [[buffer(1)]],
                             constant SpkFxPointArgs& q [[buffer(2)]],
                             uint i [[thread_position_in_grid]]) {
    if (i >= q.n) return;
    out[3u * i] = x[3u * i];
    out[3u * i + 1] = x[3u * i + 1];
    out[3u * i + 2] = x[3u * i + 2];
}
