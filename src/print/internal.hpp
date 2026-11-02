// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 Guangzhou Songzi Intelligent Technology Services Co., Ltd.

/* Print module internals: the float64 set-up math and the PrintSetup / PrintRun state.
 * tests/test_print.cpp reads it.
 */
#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "fx/glare.hpp"
#include "fx/scanner.hpp"
#include "print/print.hpp"
#include "print_shared.h"
#include "res/colour.hpp"
#include "res/profile.hpp"

namespace spk::print {

using Spectrum = res::Spectrum;
using Mat3 = res::Mat3;
using Vec3 = std::array<double, 3>;
using Spectral3 = std::array<double, 81 * 3>;  // [lambda][channel]

inline constexpr double kLog2Of10 = 3.321928094887362;
inline constexpr double kMidGrey = 0.184;

// ---- enlarger ----------------------------------------------------------------
// The fully inserted dichroic transmittances, k = 0 C, 1 M, 2 Y.
std::array<Spectrum, 3> dichroic_filters();
// I(l) = S_lamp(l) T_C(l) T_M(l) T_Y(l), T_k = 1 - (1 - F_k)(1 - 10^(-cc_k/100)).
Spectrum filtered_light(const Spectrum& lamp, const std::array<Spectrum, 3>& filters, const Vec3& cc);

// ---- print exposure ------------------------------------------------------------
// S_c = 10^log_sensitivity, NaN -> 0.
Spectral3 paper_sensitivity(const res::Profile& paper);
// The reciprocal geometric mean of R_c = sum_l [10^(-D) I]_{NaN->0} S_c.
double exposure_factor(const Spectrum& density, const Spectrum& light, const Spectral3& sens);

// The film-model part of the mid-grey probe: the spectral density of grey G.
struct ProbeInputs {
    const res::Profile* film = nullptr;
    const std::vector<double>* table = nullptr;  // L x L x 3 float64 (film::SpectralTable::f64)
    uint32_t table_size = 0;                     // L
    Mat3 srgb_to_xyz{};                          // sRGB RGB->XYZ followed by the CAT16 adaptation
    double gamma = 1.0;                          // density_curve_gamma
};
Mat3 probe_matrix(const res::ColourSpace& srgb, const std::array<double, 2>& xy_reference, const Mat3& cat16);
// Mitchell-Netravali (B = C = 1/3) bicubic sample of the L x L x 3 table at (t0, t1), float64.
Vec3 bicubic_sample(const std::vector<double>& table, uint32_t L, double t0, double t1);
Spectrum probe_density(const ProbeInputs& in, double grey);

// ---- paper curves --------------------------------------------------------------
// The paper curves from density_curves_model, on the profile's own log_exposure axis, K x 3 float64.
// Throws USER without a model.
std::vector<double> paper_model_curves(const res::Profile& paper);

// ---- the scan --------------------------------------------------------------------
struct ScanTables {
    Spectral3 eps{};   // neutralised medium channel_density
    Spectrum beta{};   // neutralised base density
    Spectral3 wt{};    // W^scan_k = S_view x_k / N, neutralised
    Vec3 xyz_illum{};  // Y = 1
};
ScanTables scan_tables(const res::Profile& medium, const Spectrum& view, const std::vector<double>& cmfs);

/* neutralise(eps, beta, wt): zero every wavelength where beta, any eps or any wt is NaN */
void neutralise(Spectral3& eps, Spectrum& beta, Spectral3& wt);

// ---- CAM16-UCS, float64 ----------------------------------------------------------
namespace cam16 {

inline constexpr Mat3 kM16 = {0.401288, 0.650173, -0.051461, -0.250268, 1.204414,
                              0.045854, -0.002079, 0.048952, 0.953127};
inline constexpr Mat3 kM16InvPublished = {1.86206786,  -1.01125463, 0.14918677, 0.38752654, 0.62144744,
                                          -0.00897398, -0.01584150, -0.03412294, 1.04996444};
inline constexpr uint32_t kRows = 64, kCols = 720;
inline constexpr double kJFirst = 1.0, kJLast = 110.0;

struct Constants {
    double L_A = 64.0, Y_b = 20.0, F = 1.0, c = 0.69, N_c = 1.0, Y_w = 100.0;
    double n = 0, k = 0, F_L = 0, N_bb = 0, N_cb = 0, z = 0, D = 0;
    Vec3 rgb_w{}, D_rgb{};
    double A_w = 0;
    double J_white = 0;  // J' of the adopted white (forward of XYZ(wp))
    Mat3 m16 = kM16;
    Mat3 m16_inv{};      // closed-form cofactor inverse of the literal, float64
};
Constants make_constants(const std::array<double, 2>& wp);

double post_adapt(double v, double F_L);
struct Forward {
    double J = 0, C = 0, h = 0, M = 0, Jp = 0, Mp = 0;
};
Forward forward(const Constants& k, const Vec3& xyz);  // Y_white = 1
// The table-construction variant (unguarded, the computed inverse) from (J', a', b').
Vec3 inverse_unguarded(const Constants& k, double Jp, double ap, double bp);
// The 64 x 720 C_max table, float64.
std::vector<double> cmax_table(const Constants& k, const Mat3& m_to_rgb);

}  // namespace cam16

// Builds the per-space tables (constants, cmax f64 + f32, k without the per-call knee entries).
std::unique_ptr<Cam16Tables> build_cam16_tables(const res::Resources& res, std::string_view space);
// The constants of Cam16Tables::k (with the given knee / lightness parameters in place).
std::vector<double> cam16_k(const cam16::Constants& c, const GamutOptions& g);

// ---- set-up state --------------------------------------------------------------------------------------

struct LevelState {
    bool black = false, white = false;
    double B = 0, W = 0;  // decoded scanner levels
};

struct PrintSetup::Impl {
    std::shared_ptr<const params::ParamSet> params;
    std::shared_ptr<const res::Profile> paper;
    const res::Resources* res = nullptr;
    res::ColourSpace out_cs;
    fx::GlareParams glare;
    fx::ScannerSpatial scanner;
    core::Device device = core::Device::Desktop;

    bool scanned = false;  // scan_film
    bool film_positive = false;
    bool paper_positive = false;
    bool compensation = true;  // print exposure compensation (off under product_defaults)

    // enlarger
    Spectrum lamp{};
    std::array<Spectrum, 3> filters{};
    Vec3 neutral{};  // (c, m, y) neutral pack, CC

    // print exposure; the film's unmodified arrays
    Spectral3 paper_sens{};
    Spectral3 film_eps{};
    Spectrum film_beta{};
    Spectrum probe_mid{}, probe_comp{};

    // paper curves: x[K], r[K], y[K][3] float32 on the GPU
    uint32_t curve_k = 0;
    gpu::Buffer curve_gpu;

    // scan and levels
    ScanTables scan;            // float64
    SpkPrintSpectralArgs scan_args{};  // narrowed
    LevelState levels;
    std::array<double, 3> film_white_density{};  // d^W_c = nanmax raw film curve
    std::vector<double> paper_mean_curve;        // abar(i) = nanmean_c raw paper curve
    double paper_mean_base = 0;                  // nanmean base density (paper)

    // XYZ -> RGB, gamut, EDR, CCTF
    SpkPrintMatrixArgs to_rgb{};
    const Cam16Tables* cam = nullptr;
    std::unique_ptr<Cam16Tables> own_cam;  // when no engine cache was given
    SpkPrintCam16Args cam_args{};
    gpu::Buffer cmax_gpu;
    bool edr = false;
    SpkPrintEdrArgs edr_args{};
    gpu::Buffer edr_gpu;
    bool cctf = true;
    SpkPrintEncodeArgs enc_args{};
};

// The live-dependent constants of one print run, computed once per run.
struct PrintRun {
    LiveParams live;
    Spectrum light{};             // I(l), the filtered enlarger light
    SpkPrintSpectralArgs enl{};   // neutralised eps/beta/W, g, o (narrowed)
    Spectral3 eps_n{}, w_n{};     // neutralised, float64
    Spectrum beta_n{};
    Vec3 g{}, o{};
    bool levels_active = false;
    double m = 1, q = 0, y_k = 0, y_w = 1, kappa_bw = 1;
    float k_gain = 1.0f;          // f32(print_exposure * kappa_bw)
};

/* levels_active(s): a correction is on, and the path is printed or scans a positive film
 *   a scanned negative is a pass-through
 */
bool levels_active(const PrintSetup::Impl& s);

// The per-run computation, separated from the GPU for the unit tests.
std::shared_ptr<PrintRun> compute_run(const PrintSetup::Impl& s, const LiveParams& live);

}  // namespace spk::print
