// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 Guangzhou Songzi Intelligent Technology Services Co., Ltd.

/* Random draws shared by the grain and glare kernels: the float32 Poisson sampler and the Box-Muller
 * lognormal value, in the bracketed order written here.
 *
 * requires precise fp32 functions (-fmetal-math-fp32-functions=precise)
 * contraction of a*b + c into an FMA is left at the compiler's safe-mode default
 */
#pragma once

#include "spk_common.h"

namespace spkfx {

constant float kTwoPi = 6.283185307179586f;  // T = f32(2 pi)

// lgamma(x) for x = k + 1.
inline float lgamma_k(float x) {
    if (x == 1.0f || x == 2.0f) return 0.0f;
    float x0 = x;
    int n = 0;
    if (x <= 7.0f) {
        n = int(7.0f - x);
        x0 = x + float(n);
    }
    const float x2 = 1.0f / (x0 * x0);
    float g = 8.417508417508418e-04f;
    g = g * x2 + -5.952380952380952e-04f;
    g = g * x2 + 7.936507936507937e-04f;
    g = g * x2 + -2.777777777777778e-03f;
    g = g * x2 + 8.333333333333333e-02f;
    float gl = ((g / x0 + 0.5f * metal::log(kTwoPi)) + (x0 - 0.5f) * metal::log(x0)) - x0;
    if (x <= 7.0f) {
        for (int i = 0; i < n; ++i) {
            gl = gl - metal::log(x0 - 1.0f);
            x0 = x0 - 1.0f;
        }
    }
    return gl;
}

/* poisson(lam, pixel, stream, seed): a fresh generator of one (pixel, stream) pair
 *   lam >= 10: transformed rejection (Hormann, Insurance: Mathematics and Economics, 1993), at most 64 tries,
 *              then floor(lam)
 *   lam < 10:  multiply uniforms until the product falls to exp(-lam), at most 512 steps
 */
inline float poisson(float lam, uint pixel, uint stream, uint seed) {
    spk::PhiloxStream g;
    g.init(pixel, stream, seed);
    if (lam >= 10.0f) {
        const float slam = metal::sqrt(lam);
        const float loglam = metal::log(lam);
        const float b = 0.931f + (2.53f * slam);
        const float a = -0.059f + (0.02483f * b);
        const float invalpha = 1.1239f + (1.1328f / (b - 3.4f));
        const float vr = 0.9277f - (3.6224f / (b - 2.0f));
        for (int it = 0; it < 64; ++it) {
            const float U = g.next_uniform() - 0.5f;
            const float V = g.next_uniform();
            const float us = 0.5f - metal::abs(U);
            const float k = metal::floor(((((2.0f * a) / us + b) * U + lam) + 0.43f));
            if (us >= 0.07f && V <= vr) return k;
            if (k < 0.0f || (us < 0.013f && V > us)) continue;
            const float L = ((metal::log(V) + metal::log(invalpha)) - metal::log(a / (us * us) + b));
            const float Rt = ((-lam + k * loglam) - lgamma_k(k + 1.0f));
            if (L <= Rt) return k;
        }
        return metal::floor(lam);
    }
    const float E = metal::exp(-lam);
    float X = 0.0f;
    float prod = 1.0f;
    for (int it = 0; it < 512; ++it) {
        prod = prod * g.next_uniform();
        if (prod > E) {
            X = X + 1.0f;
        } else {
            return X;
        }
    }
    return X;
}

/* sublayer_draw(d, D, n, u, ...)
 *   q := clamp(d / D, 1e-6, 0.999999)                    -- a NaN stays NaN and gives 0 draws
 *   sat := 1 - q u 0.999999
 *   lam := (n / sat) q
 *   return Poisson(lam) (D / n) sat
 */
inline float sublayer_draw(float d, float D, float n, float u, uint pixel, uint stream, uint seed) {
    const float eps = 1e-6f;
    const float one_m_eps = 0.999999f;  // f32
    float q = d / D;
    q = q < eps ? eps : (q > one_m_eps ? one_m_eps : q);  // a NaN stays NaN (lambda NaN -> 0 draws)
    const float sat = 1.0f - ((q * u) * one_m_eps);
    const float lam = (n / sat) * q;
    return poisson(lam, pixel, stream, seed) * ((D / n) * sat);
}

/* lognormal(m, s, ...): Box-Muller from words x, y of block 0
 *   z := sqrt(-2 ln u1) cos(2 pi u2)
 *   sig := sqrt(ln(1 + s^2 / m^2)); mu := ln m - sig^2 / 2
 *   return exp(mu + sig z)
 */
inline float lognormal(float m, float s, uint pixel, uint stream, uint seed) {
    spk::PhiloxStream g;
    g.init(pixel, stream, seed);
    const float u1 = g.next_uniform();
    const float u2 = g.next_uniform();
    const float z = metal::sqrt(-2.0f * metal::log(u1)) * metal::cos(kTwoPi * u2);
    const float sig = metal::sqrt(metal::log(1.0f + (s * s) / (m * m)));
    const float mu = metal::log(m) - (0.5f * sig) * sig;
    return metal::exp(mu + sig * z);
}

}  // namespace spkfx
