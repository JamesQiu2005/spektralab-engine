// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 Guangzhou Songzi Intelligent Technology Services Co., Ltd.

/* Shared by every .metal file of spektralab.metallib.
 *
 * kernel names  spk_<module>_<what>
 * planes        interleaved float32 x 3; pixel i at [3i, 3i+1, 3i+2]
 * arguments     bound in order at index 0, 1, 2, ..., exactly as the host's gpu::Arg list
 * structs       passed by value are declared in a header shared with C++ (kernels/spk_shared.h)
 */
#pragma once

#include <metal_stdlib>
#include "spk_shared.h"

namespace spk {

/* ---- Philox4x32-10 ----
 * each of 10 rounds:
 *   c := (hi(M1 c2) ^ c1 ^ k0, lo(M1 c2), hi(M0 c0) ^ c3 ^ k1, lo(M0 c0))
 *   k0 += W0; k1 += W1
 */
inline uint4 philox4x32_10(uint4 c, uint2 k) {
    for (int r = 0; r < 10; ++r) {
        const uint hi0 = metal::mulhi(SPK_PHILOX_M0, c.x);
        const uint lo0 = SPK_PHILOX_M0 * c.x;
        const uint hi1 = metal::mulhi(SPK_PHILOX_M1, c.z);
        const uint lo1 = SPK_PHILOX_M1 * c.z;
        c = uint4(hi1 ^ c.y ^ k.x, lo1, hi0 ^ c.w ^ k.y, lo0);
        k.x += SPK_PHILOX_W0;
        k.y += SPK_PHILOX_W1;
    }
    return c;
}

// u = ((word >> 8) + 0.5) * 2^-24.
inline float philox_uniform(uint word) { return (float(word >> 8) + 0.5f) * 0x1p-24f; }

// One generator per (pixel, stream): counter = (pixel_index, stream, j, 0), key = (seed, 0).
// Words are consumed x, y, z, w, then j increments.
struct PhiloxStream {
    uint pixel;
    uint stream;
    uint seed;
    uint j;
    uint4 block;
    uint used;  // words of `block` already consumed (4 = need a new block)

    inline void init(uint pixel_index, uint stream_id, uint seed_value) {
        pixel = pixel_index;
        stream = stream_id;
        seed = seed_value;
        j = 0;
        used = 4;
    }
    inline uint next_word() {
        if (used == 4) {
            block = philox4x32_10(uint4(pixel, stream, j, 0u), uint2(seed, 0u));
            ++j;
            used = 0;
        }
        const uint w = used == 0 ? block.x : used == 1 ? block.y : used == 2 ? block.z : block.w;
        ++used;
        return w;
    }
    inline float next_uniform() { return philox_uniform(next_word()); }
};

// Half-sample symmetric reflection.
inline int reflect_index(int i, int n) {
    if (n <= 1) return 0;
    const int period = 2 * n;
    int m = i % period;
    if (m < 0) m += period;
    return m < n ? m : period - 1 - m;
}

// Mirror without edge repeat.
inline int mirror_index(int i, int n) {
    if (n <= 1) return 0;
    const int period = 2 * (n - 1);
    int m = i % period;
    if (m < 0) m += period;
    return m < n ? m : period - m;
}

}  // namespace spk
