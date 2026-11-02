// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 Guangzhou Songzi Intelligent Technology Services Co., Ltd.

/* Kernels, named spk_fx_glare_<what>: glare add.
 *
 * the field is spk_fx_grain_lognormal (stream 200, one value on all channels)
 * its blur runs on the fx_blur kernels
 */
#include "spk_common.h"
#include "fx_shared.h"

// xyz_c <- xyz_c + (F / 100) * I_c
kernel void spk_fx_glare_add(device float* xyz [[buffer(0)]],
                             device const float* field [[buffer(1)]],
                             constant SpkFxGlareAddArgs& q [[buffer(2)]],
                             uint i [[thread_position_in_grid]]) {
    if (i >= q.n) return;
    for (uint c = 0; c < 3u; ++c) {
        const uint o = 3u * i + c;
        xyz[o] = xyz[o] + (field[o] / 100.0f) * q.tint[c];
    }
}
