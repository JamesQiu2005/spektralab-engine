// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 Guangzhou Songzi Intelligent Technology Services Co., Ltd.

/* The film side: tier RGB plane -> developed negative density plane.
 *
 * set-up on the CPU in float64, narrowed once to float32
 * develop() on the GPU in float32
 *
 * develop stages, in order; [ ] = conditional
 *   [decode input transfer]
 *   [geometry: crop, rotation, quarter turns, flips]
 *   [auto exposure: x 2^ev]
 *   spectral upsampling: RGB -> film-layer exposure H
 *   exposure compensation: x 2^ev
 *   [highlight boost]
 *   [highlight headroom: back-reflection preset only]
 *   [camera lens blur]
 *   [halation: scatter, then bounces or back-reflection]
 *   log10 exposure
 *   density curves
 *   [DIR couplers: correction, diffusion, second lookup]
 *   [grain]
 */
#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string_view>
#include <vector>

#include "core/render.hpp"
#include "gpu/gpu.hpp"
#include "params/params.hpp"
#include "res/colour.hpp"
#include "res/profile.hpp"
#include "res/resources.hpp"

namespace spk::fx {
struct HalationParams;
struct GrainSetup;
}

namespace spk::film {

/* SpectralTable: L x L x 3 film-layer weights, [i][j][c]; float64 as built, f32 copy for the kernels
 *   shared with print's mid-grey probe
 */
struct SpectralTable {
    uint32_t size = 0;          // L
    std::vector<double> f64;    // L*L*3, [i][j][c]
    gpu::Buffer f32;            // the same narrowed, persistent (MemTag::Table)
};

struct FilmSetupInputs {
    const params::ParamSet& params;
    std::shared_ptr<const res::Profile> film;
    const res::Resources& res;
    gpu::Context& gpu;
    core::Device device = core::Device::Desktop;
};

// Immutable once built; shared by every render of a pipeline instance.
class FilmSetup {
public:
    ~FilmSetup();
    const res::Profile& profile() const;
    const params::ParamSet& params() const;  // the values it was built from
    bool is_positive() const;

    const res::ColourSpace& input_space() const;
    const SpectralTable& spectral_table() const;

    struct Impl;
    const Impl& impl() const { return *p_; }

private:
    friend std::shared_ptr<const FilmSetup> build_film_setup(const FilmSetupInputs&);
    FilmSetup() = default;
    std::unique_ptr<Impl> p_;
};

/* build_film_setup(in): steps on its definition in setup.cpp
 *   throws SPK_ERR_USER: unknown input colour space, unknown reference illuminant, a window of 8 values,
 *                        grain sublayers on without density_curves_layers
 */
std::shared_ptr<const FilmSetup> build_film_setup(const FilmSetupInputs& in);

// ---- geometry --------------------------------------------------------------
struct Size {
    uint32_t w = 0, h = 0;
};
// The tier image's size after crop, rotation, quarter turns and flips.
Size geometry_output_size(const params::ParamSet& p, uint32_t w, uint32_t h);

// ---- the meter --------------------------------------------------------------
struct MeterValues {
    double center_weighted = 0, average = 0, median = 0;
    double balanced = 0, center = 0, protect_highlights = 0, protect_shadows = 0;
    double ev(std::string_view method) const;  // by wire name
};
/* decode_and_geometry(ctx, p, res, raw) -> new plane of geometry_output_size()
 *   input transfer decode, then geometry; the meter reads its image this way
 */
gpu::Plane decode_and_geometry(const core::RenderContext& ctx, const params::ParamSet& p, const res::Resources& res,
                               const gpu::Plane& raw);
/* meter(ctx, p, res, meter_source): meter_source = the raw frame downscaled to a 1600 px long edge,
 *   or the frame itself; before decode and geometry, which this function applies
 */
MeterValues meter(const core::RenderContext& ctx, const params::ParamSet& p, const res::Resources& res,
                  const gpu::Plane& meter_source);

// ---- develop --------------------------------------------------------------------------------
struct DevelopArgs {
    const gpu::Plane& tier_source;            // raw values, uncropped
    std::optional<double> auto_exposure_ev;   // the session's metered EV, nullopt when off
    uint32_t film_seed = 0;                   // drawn by the session (or kNoGrainFilmSeed)
};
/* develop(ctx, setup, args) -> negative: persistent, post-geometry size
 *   ctx.pitch_um and ctx.rho_tier set by the session
 *   throws SPK_ERR_GPU for a positive film scanned directly with black/white correction
 */
gpu::Plane develop(core::RenderContext& ctx, const FilmSetup& setup, const DevelopArgs& args);

/* node_count(setup): the ctx.node() calls develop() makes; the session adds film and print for pct */
uint32_t node_count(const FilmSetup& setup);

/* pitch_um = 1000 * film_format_mm / pre-geometry long edge */
inline double pitch_um(double film_format_mm, uint32_t pre_geometry_long_edge) {
    return 1000.0 * film_format_mm / double(pre_geometry_long_edge);
}

}  // namespace spk::film
