// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 Guangzhou Songzi Intelligent Technology Services Co., Ltd.

/* Internals of the film module, shared by its translation units and tests/test_film.cpp. */
#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "film/film.hpp"
#include "fx/grain.hpp"
#include "fx/halation.hpp"

namespace spk::film {

inline constexpr uint32_t kTableSize = 192;  // L

// The coupler gammas, rows = donor (R, G, B), columns = receiver.
using Mat3 = res::Mat3;
Mat3 coupler_gammas(const res::ProfileInfo& info);
// M = amount * [I_s s / I_i interlayer], I_s = I_i = 1.
Mat3 coupler_matrix(const res::ProfileInfo& info, double amount);

// The density and pre-inverted curve tables (float32, as uploaded).
struct CurveTables {
    uint32_t K = 0;
    std::vector<float> x, inv;  // K each (inv[K-1] = 0, unused)
    std::vector<float> y, y0;   // 3K each, channel-major [c*K + k]
    std::vector<double> n0;     // N0 as float64, K x 3 row-major (tests)
};
CurveTables build_curve_tables(const res::Profile& film, double gamma, const Mat3& m, bool positive);

// The spectral-upsampling table in float64: T' (L x L x 3, [i][j][c]).
// sensitivity: 81 x 3 linear (effective), illum: normalised I.
std::vector<double> build_spectral_table_f64(const res::Blob& blob, const std::array<double, 81 * 3>& sensitivity,
                                             const std::vector<double>& window_params, const res::Spectrum& illum,
                                             uint32_t* size_out);

// CPU references with the kernels' binary32 operation order (tests).
namespace cpu {
// Spectral upsampling of one pixel.
std::array<float, 3> upsample(const std::array<float, 9>& a, const float* table, uint32_t L, const float rgb[3]);
// The curve lookup.
float curve_lookup(const float* x, const float* inv, const float* y, uint32_t K, float v);
float mitchell(float u);
}  // namespace cpu

// The geometry plan (binary64 constants and per-axis terms).
struct GeometryPlan {
    bool identity = true;
    uint32_t src_w = 0, src_h = 0, dst_w = 0, dst_h = 0;
    double cx = 0, cy = 0;
    // Per output column / row: the binary64 terms whose sum with (cx, cy) gives (f_x, f_y).
    std::vector<double> col_x, col_y, row_x, row_y;
};
GeometryPlan plan_geometry(const params::ParamSet& p, uint32_t w, uint32_t h);
// CPU reference of the whole stage in binary64 coordinates, for tests.
void geometry_cpu(const GeometryPlan& g, const float* src, float* dst);

// The meter values over a meter image (row-major sh x sw, float32 RGB in the input space).
MeterValues meter_values(const float* rgb, uint32_t sw, uint32_t sh, const res::Mat3& rgb_to_xyz);

// GPU helpers (each encodes into its own batch and waits).
gpu::Plane run_decode(const core::RenderContext& ctx, const gpu::Plane& in, res::CctfMode mode);
gpu::Plane run_geometry(const core::RenderContext& ctx, const gpu::Plane& in, const GeometryPlan& g);
void run_scale(const core::RenderContext& ctx, gpu::Plane& p, float g);
// The highlight boost, in place; M is the frame's maximum.
void run_boost(const core::RenderContext& ctx, gpu::Plane& plane, double beta_ev, float M);
float plane_max(const gpu::Plane& p);  // max over every element (CPU, shared storage)

struct FilmSetup::Impl {
    explicit Impl(const params::ParamSet& p) : params(p) {}
    std::shared_ptr<const res::Profile> profile;
    params::ParamSet params;
    bool positive = false;

    res::ColourSpace input_space;
    bool decode = false;  // input_cctf_decoding
    res::Spectrum illuminant{};
    std::array<double, 2> illuminant_xy{};
    std::array<double, 81 * 3> sensitivity{};
    std::shared_ptr<const SpectralTable> table;

    Mat3 input_matrix{};  // A, binary64
    std::array<float, 9> input_matrix_f32{};

    fx::HalationParams halation;
    std::shared_ptr<const fx::GrainSetup> grain;

    bool couplers = false;
    Mat3 coupler_m{};
    std::array<float, 9> coupler_m_f32{};
    std::array<float, 3> dmax_f32{};
    CurveTables curves;
    gpu::Buffer curve_buffer;  // [x K][inv K][y 3K][y0 3K]
};

}  // namespace spk::film
