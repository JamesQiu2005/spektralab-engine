// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 Guangzhou Songzi Intelligent Technology Services Co., Ltd.

/* Philox4x32-10 and the splitmix64 seed sequence, host copies; rules in rng.hpp. */
#include "fx/rng.hpp"

#include "spk_shared.h"

namespace spk::fx {

std::array<uint32_t, 4> philox4x32_10(PhiloxCounter c, PhiloxKey k) {
    for (int r = 0; r < 10; ++r) {
        const uint64_t p0 = uint64_t(SPK_PHILOX_M0) * c[0];
        const uint64_t p1 = uint64_t(SPK_PHILOX_M1) * c[2];
        const uint32_t hi0 = uint32_t(p0 >> 32), lo0 = uint32_t(p0);
        const uint32_t hi1 = uint32_t(p1 >> 32), lo1 = uint32_t(p1);
        c = {hi1 ^ c[1] ^ k[0], lo1, hi0 ^ c[3] ^ k[1], lo0};
        k[0] += SPK_PHILOX_W0;
        k[1] += SPK_PHILOX_W1;
    }
    return c;
}

float philox_uniform(uint32_t word) { return (float(word >> 8) + 0.5f) * 0x1p-24f; }

uint32_t SeedSequence::next() {
    state_ += kGamma;
    uint64_t z = state_;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    z = z ^ (z >> 31);
    return uint32_t(z);
}

}  // namespace spk::fx
