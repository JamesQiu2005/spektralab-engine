// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 Guangzhou Songzi Intelligent Technology Services Co., Ltd.

/* Tiers and the tier downscale.
 *
 * tier      long edge L
 * live      preview_long_edge
 * preview   3400
 * full      the frame's own (L = 0)
 */
#pragma once

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string_view>

#include "core/render.hpp"
#include "gpu/gpu.hpp"

namespace spk::session {

enum class Tier : uint8_t { Live = 0, Preview = 1, Full = 2 };
inline constexpr uint32_t kPreviewLongEdge = 3400;
inline constexpr uint32_t kMeterLongEdge = 1600;

std::optional<Tier> parse_tier(std::string_view s);  // "live" | "preview" | "full"
const char* tier_name(Tier t);
// The tier's target long edge; 0 for full (= the frame's own).
uint32_t tier_target_long_edge(Tier t, uint32_t preview_long_edge);

struct Size {
    uint32_t w = 0, h = 0;
};
/* downscale_size(w, h, L)
 *   M := max(w, h); if L == 0 or M <= L: (w, h)
 *   s := L / M (float64)
 *   (max(1, floor(w s)), max(1, floor(h s)))
 */
Size downscale_size(uint32_t w, uint32_t h, uint32_t long_edge);

/* tier_ratio: rho_tier = actual long edge of the tier image / long edge of the frame
 *   never L / M: the floor in downscale_size can give L - 1 (M = 2156, L = 1600 -> 1599)
 */
inline double tier_ratio(const Size& tier_source, const Size& full_source) {
    return double(std::max(tier_source.w, tier_source.h)) / double(std::max(full_source.w, full_source.h));
}

/* downscale(ctx, src, L, tag) -> persistent plane of downscale_size(); src itself when no downscale applies
 *   per axis of input n, output m:
 *     f := n / m; sigma := max(0, (f - 1) / 2); r := floor(4 sigma + 0.5)
 *     w_k := exp(-0.5 (k / sigma)^2), k = -r..r, normalised; float64, stored float32
 *     r == 0: axis skipped
 *   vertical pass, then horizontal: out := sum_k fma(w_k, x[mirror(i + k)], acc), ascending k
 *   bilinear from the index tables of axis_samples
 */
gpu::Plane downscale(const core::RenderContext& ctx, const gpu::Plane& src, uint32_t long_edge,
                     gpu::MemTag tag = gpu::MemTag::TierImage);

// CPU reference with the identical operation order (tests, and a fallback).
void downscale_cpu(const float* src, uint32_t w, uint32_t h, float* dst, uint32_t ow, uint32_t oh);

}  // namespace spk::session
