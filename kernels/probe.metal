// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 Guangzhou Songzi Intelligent Technology Services Co., Ltd.

/* Numeric-mode probe.
 *
 * safe math with FMA contraction: a*b - a*b = fma(a, b, -(a*b)), the rounding error of the product
 * fast math: folds to 0
 */
#include "spk_common.h"

kernel void spk_probe_fma(device const float* a [[buffer(0)]],
                          device const float* b [[buffer(1)]],
                          device float* out [[buffer(2)]],
                          constant uint& n [[buffer(3)]],
                          uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    const float x = a[i];
    const float y = b[i];
    out[i] = x * y - x * y;
}

// Self-test of the shared Philox implementation: out[4i..4i+3] = philox(ctr[i], key[i]).
kernel void spk_probe_philox(device const uint4* ctr [[buffer(0)]],
                             device const uint2* key [[buffer(1)]],
                             device uint4* out [[buffer(2)]],
                             constant uint& n [[buffer(3)]],
                             uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    out[i] = spk::philox4x32_10(ctr[i], key[i]);
}
