// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 Guangzhou Songzi Intelligent Technology Services Co., Ltd.

/* DIR-coupler diffusion blur; the mixture is in dir_blur.hpp. */
#include <vector>

#include "fx/dir_blur.hpp"
#include "fx/fx_internal.hpp"
#include "util/error.hpp"

namespace spk::fx {

namespace {
std::vector<MixtureComponent> dir_components(double p, const DirDiffusion& d) {
    std::vector<MixtureComponent> comps;
    MixtureComponent core;
    const double s = d.core_um / p;  // no 1e-6 clamp
    for (size_t c = 0; c < 3; ++c) {
        core.weight[c] = 1.0 - d.tail_weight;
        core.sigma[c] = s;
    }
    comps.push_back(core);
    const double lam = d.tail_um / p;
    for (const MixtureComponent& m : exponential_surrogate(Vec3{d.tail_weight, d.tail_weight, d.tail_weight},
                                                           Vec3{lam, lam, lam}))
        comps.push_back(m);
    return comps;
}
}  // namespace

void dir_diffusion_blur(const core::RenderContext& ctx, gpu::Plane& correction, const DirDiffusion& d) {
    if (!(d.core_um > 0.0)) return;
    if (!(ctx.pitch_um > 0.0)) fail_internal("fx: DIR diffusion without a pixel pitch");
    const std::vector<MixtureComponent> comps = dir_components(ctx.pitch_um, d);
    gpu::Batch bt = ctx.gpu.batch();
    detail::encode_mixture(ctx, bt, correction, correction, comps);
    bt.commit_and_wait();
}

}  // namespace spk::fx
