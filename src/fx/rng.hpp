// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 Guangzhou Songzi Intelligent Technology Services Co., Ltd.

/* The random-number contract.
 *
 * host copy of Philox4x32-10; the GPU copy is kernels/spk_common.h; both give identical words
 * Published: Salmon et al., "Parallel random numbers: as easy as 1, 2, 3", SC 2011.
 *
 * SeedSequence: splitmix64, one per pipeline instance
 * Published: Steele, Lea and Flood, "Fast splittable pseudorandom number generators", OOPSLA 2014.
 *   state := 0x243F6A8885A308D3 at every (re)build
 *   draw:   state += 0x9E3779B97F4A7C15; z := state
 *           z := (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
 *           z := (z ^ (z >> 27)) * 0x94D049BB133111EB
 *           z := z ^ (z >> 31); seed := low 32 bits of z
 */
#pragma once

#include <array>
#include <cstdint>

namespace spk::fx {

using PhiloxCounter = std::array<uint32_t, 4>;
using PhiloxKey = std::array<uint32_t, 2>;

// Ten standard Philox rounds.
std::array<uint32_t, 4> philox4x32_10(PhiloxCounter c, PhiloxKey k);
// u = ((word >> 8) + 0.5) * 2^-24
float philox_uniform(uint32_t word);

class SeedSequence {
public:
    static constexpr uint64_t kInitialState = 0x243F6A8885A308D3ull;
    static constexpr uint64_t kGamma = 0x9E3779B97F4A7C15ull;

    uint32_t next();                         // one draw
    void reset() { state_ = kInitialState; } // a pipeline (re)build
    uint64_t state() const { return state_; }

private:
    uint64_t state_ = kInitialState;
};

// Film seed when grain is inactive: 0x5EED; print seed when glare cannot draw: 0.
inline constexpr uint32_t kNoGrainFilmSeed = 0x5EEDu;
inline constexpr uint32_t kNoGlarePrintSeed = 0u;

}  // namespace spk::fx
