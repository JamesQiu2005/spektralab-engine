// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 Guangzhou Songzi Intelligent Technology Services Co., Ltd.

/* The print side, the scan and the output: negative density -> output RGB -> encoded RGBA16 in a fresh
 * output texture.
 *
 * print_scan_encode stages, in order; [ ] = conditional
 *   [enlarger: spectral integral -> log10 exposure on paper]    printed path
 *   [gain: print exposure]                                      printed path
 *   [paper curves]                                              printed path
 *   scan: spectral integral -> XYZ (of the paper, or of the film under scan_film)
 *   [scanner levels]
 *   [glare]                                                     printed path
 *   XYZ -> output RGB
 *   gamut compression, CAM16-UCS
 *   [scanner blur + unsharp mask]                               not under print_spatial=omitted
 *   [EDR tone map]                                              printed path
 *   transfer function, 16-bit encode
 */
#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

#include "core/render.hpp"
#include "gpu/gpu.hpp"
#include "params/params.hpp"
#include "res/colour.hpp"
#include "res/profile.hpp"
#include "res/resources.hpp"
#include "util/json.hpp"

namespace spk::film {
class FilmSetup;
}
namespace spk::fx {
struct GlareParams;
}

namespace spk::print {

class Cam16Cache;

// The five live fields, written onto a built pipeline without a rebuild.
struct LiveParams {
    double print_exposure = 1.0;
    double m_filter_shift = 0.0;
    double y_filter_shift = 0.0;
    double preflash_exposure = 0.0;
    double filter_shift_scale = 1.0;
    static LiveParams from(const params::ParamSet& p);
    bool operator==(const LiveParams&) const = default;
};

struct PrintSetupInputs {
    const params::ParamSet& params;
    const film::FilmSetup& film;                 // the mid-grey probe uses the film's tables
    std::shared_ptr<const res::Profile> print;   // the paper (or print film)
    const res::Resources& res;
    gpu::Context& gpu;
    Cam16Cache* cam16 = nullptr;  // engine-wide; may be null
    core::Device device = core::Device::Desktop;
    bool print_exposure_compensation = true;  // false under product_defaults
};

/* PrintSetup: non-live set-up, once per pipeline instance; float64, narrowed once
 *   constants that depend on live fields: compute_run(), once per print run
 */
class PrintSetup {
public:
    ~PrintSetup();
    const res::Profile& profile() const;
    const params::ParamSet& params() const;
    const res::ColourSpace& output_space() const;
    // The glare tint and parameters.
    const fx::GlareParams& glare() const;

    struct Impl;
    const Impl& impl() const { return *p_; }

private:
    friend std::shared_ptr<const PrintSetup> build_print_setup(const PrintSetupInputs&);
    PrintSetup() = default;
    std::unique_ptr<Impl> p_;
};

/* build_print_setup(in): steps on its definition in build.cpp
 *   throws SPK_ERR_USER: unknown output space, unresolvable enlarger_illuminant,
 *                        EDR on a paper without edr_tone_map, printed path without a cdfs curve model
 */
std::shared_ptr<const PrintSetup> build_print_setup(const PrintSetupInputs& in);

struct PrintArgs {
    const gpu::Plane& negative;  // the cached negative (never modified)
    LiveParams live;             // the live fields, current values
    uint32_t print_seed = 0;     // drawn by the session (or kNoGlarePrintSeed)
};
/* print_scan_encode(ctx, setup, args) -> fresh output image of the negative's size; texture +1
 *   ctx carries the tier's pitch and rho_tier
 */
gpu::OutputImage print_scan_encode(core::RenderContext& ctx, const PrintSetup& setup, const PrintArgs& args);

/* preview_lut(ctx, negative, table) -> fresh output image
 *   trilinear lookup of each pixel's densities in the table, 16-bit encode; no glare, no spatial effect
 */
gpu::OutputImage preview_lut(core::RenderContext& ctx, const gpu::Plane& negative, const res::PrintLutTable& table);

/* solve_filter_pack: the database row for (paper, illuminant, film), alias fallback included, or nullopt */
std::optional<std::array<double, 3>> solve_filter_pack(const res::Resources& res, const params::ParamSet& p,
                                                       const res::Profile& film);

/* node_count(setup, variants): the ctx.node() calls print_scan_encode() makes */
uint32_t node_count(const PrintSetup& setup, const core::EffectVariants& v);

// ---- spk_output_transform ----

struct GamutOptions {
    std::string algorithm = "cam16ucs";  // or "off"
    std::array<double, 3> knee{0, 1, 6};
    bool lightness_compression_active = true;
    std::array<double, 3> lightness_compression{0.7, 1.0, 2.2};
/* parse(json): keys optional; SPK_ERR_INVALID_ARG unless a JSON object; SPK_ERR_USER for an unknown algorithm
 *   an array that is not 3 numbers is ignored
 */
    static GamutOptions parse(std::string_view json_text);
};

// Per output space CAM16-UCS tables, float64 built, f32 cmax narrowed.
struct Cam16Tables {
    std::string space;
    std::array<double, 9> m_to_xyz{}, m_to_rgb{};
    std::vector<double> k;            // 22 values (without the per-call knee/lc)
    uint32_t cmax_rows = 0, cmax_cols = 0;
    std::vector<double> cmax_f64;
    std::vector<float> cmax_f32;      // engine-owned for the engine's life (out_cmax)
    gpu::Buffer cmax_gpu;             // the same on the GPU
};

class Cam16Cache {
public:
    // Builds on first use (thread-safe); the returned object lives as long as the cache.
    const Cam16Tables& get(const res::Resources& res, std::string_view output_space);
    size_t size() const;

private:
    struct Impl;
    std::shared_ptr<Impl> p_ = make_impl();
    static std::shared_ptr<Impl> make_impl();
};

struct OutputTransformResult {
    json::Value reply;          // the out_json object
    const float* cmax = nullptr;
    uint32_t cmax_count = 0;
};
// Throws spk::Error(SPK_ERR_USER) for an unknown space or algorithm (never a fallback).
OutputTransformResult output_transform(const res::Resources& res, Cam16Cache& cache, std::string_view src,
                                       std::string_view dst, const GamutOptions& gamut);

}  // namespace spk::print
