// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 Guangzhou Songzi Intelligent Technology Services Co., Ltd.

/* Print-side scanner blur and unsharp mask, on linear output RGB after gamut compression.
 *
 * the print module skips this call when ctx.variants.print_spatial_omitted
 */
#pragma once

#include "core/render.hpp"

namespace spk::fx {

struct ScannerSpatial {
    double lens_blur_px = 0.0;  // scanner_lens_blur, plain pixels (not tier-scaled)
    double unsharp_sigma_full = 0.7;  // sigma_u = 0.7 * rho_tier
    double unsharp_amount = 0.7;
};

void scanner_blur_unsharp(const core::RenderContext& ctx, gpu::Plane& rgb, const ScannerSpatial& s);

}  // namespace spk::fx
