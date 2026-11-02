// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 Guangzhou Songzi Intelligent Technology Services Co., Ltd.

/* Print-side scanner blur and unsharp mask.
 *
 * scanner_blur_unsharp(ctx, rgb, s)
 *   lens_blur_px > 0:  rgb := G(lens_blur_px)(rgb)       -- plain pixels, not tier-scaled
 *   su := 0.7 rho_tier; a := 0.7
 *   rgb := (1 + a) rgb - a G(su)(rgb)
 */
#include "fx/fx_internal.hpp"
#include "fx/scanner.hpp"

namespace spk::fx {

void scanner_blur_unsharp(const core::RenderContext& ctx, gpu::Plane& rgb, const ScannerSpatial& s) {
    const double sb = s.lens_blur_px;
    const double su = s.unsharp_sigma_full * ctx.rho_tier;
    const double a = s.unsharp_amount;
    const bool blur = sb > 0.0;
    const bool unsharp = su > 0.0 && a > 0.0;
    if (!blur && !unsharp) return;
    gpu::Batch bt = ctx.gpu.batch();
    if (blur) detail::encode_blur(ctx, bt, rgb, rgb, Vec3{sb, sb, sb});
    if (unsharp) {
        gpu::Plane g = ctx.scratch(rgb.w, rgb.h);
        detail::encode_blur(ctx, bt, rgb, g, Vec3{su, su, su});
        detail::encode_lincomb(ctx, bt, Vec3{1.0 + a, 1.0 + a, 1.0 + a}, rgb, Vec3{-a, -a, -a}, &g, rgb);
    }
    bt.commit_and_wait();
}

}  // namespace spk::fx
