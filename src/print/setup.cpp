// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 Guangzhou Songzi Intelligent Technology Services Co., Ltd.

/* The print side's float64 set-up math and compute_run().
 *
 * compute_run holds the constants that depend on the five live fields, once per print run;
 * build_print_setup() is in build.cpp.
 */
#include <cmath>
#include <cstring>

#include "fx/grain.hpp"
#include "print/internal.hpp"
#include "res/resources.hpp"
#include "util/error.hpp"
#include "util/numeric.hpp"

namespace spk::print {

using params::P;

// ---- the enlarger ---------------------------------------------------------------------

/* dichroic_filters(): fully inserted transmittance, l in nm
 *   F_Y(l) := 1/2 erf((l - 516) / 12) + 1/2
 *   F_M(l) := l <= 550 ? -1/2 erf((l - 500) / 8) + 1/2 : 1/2 erf((l - 610) / 8) + 1/2
 *   F_C(l) := -1/2 erf((l - 607) / 8) + 1/2
 */
std::array<Spectrum, 3> dichroic_filters() {
    std::array<Spectrum, 3> f{};
    for (size_t i = 0; i < 81; ++i) {
        const double l = 380.0 + 5.0 * double(i);
        f[2][i] = 0.5 * std::erf((l - 516.0) / 12.0) + 0.5;  // Y
        f[1][i] = l <= 550.0 ? 0.5 * (-std::erf((l - 500.0) / 8.0)) + 0.5   // M
                             : 0.5 * std::erf((l - 610.0) / 8.0) + 0.5;
        f[0][i] = 0.5 * (-std::erf((l - 607.0) / 8.0)) + 0.5;  // C
    }
    return f;
}

Spectrum filtered_light(const Spectrum& lamp, const std::array<Spectrum, 3>& filters, const Vec3& cc) {
    Vec3 tau{};
    for (int k = 0; k < 3; ++k) tau[k] = std::pow(10.0, -cc[k] / 100.0);
    Spectrum out{};
    for (size_t i = 0; i < 81; ++i) {
        double v = lamp[i];
        for (int k = 0; k < 3; ++k) v = v * (1.0 - (1.0 - filters[k][i]) * (1.0 - tau[k]));
        out[i] = v;
    }
    return out;
}

// ---- print exposure ---------------------------------------------------------------------

Spectral3 paper_sensitivity(const res::Profile& paper) {
    Spectral3 s{};
    for (size_t l = 0; l < 81; ++l)
        for (size_t c = 0; c < 3; ++c) {
            const double v = std::pow(10.0, paper.log_sensitivity(l, c));
            s[3 * l + c] = std::isnan(v) ? 0.0 : v;
        }
    return s;
}

/* exposure_factor(density, light, sens) -> reciprocal geometric mean
 *   L(l) := 10^-density(l) light(l), NaN -> 0
 *   R_c := sum_l L(l) sens_c(l)
 *   1 / exp(1/3 sum_c ln(max(R_c, 1e-10)))
 */
double exposure_factor(const Spectrum& density, const Spectrum& light, const Spectral3& sens) {
    Vec3 R{};
    for (size_t l = 0; l < 81; ++l) {
        double L = std::pow(10.0, -density[l]) * light[l];
        if (std::isnan(L)) L = 0.0;
        for (size_t c = 0; c < 3; ++c) R[c] += L * sens[3 * l + c];
    }
    double s = 0.0;
    for (size_t c = 0; c < 3; ++c) s += std::log(std::max(R[c], 1e-10));
    return 1.0 / std::exp((1.0 / 3.0) * s);
}

/* probe_matrix: von_kries(white of sRGB -> reference white, CAT16) . RGB->XYZ(sRGB) */
Mat3 probe_matrix(const res::ColourSpace& srgb, const std::array<double, 2>& xy_reference, const Mat3& cat16) {
    if (auto a = res::von_kries(srgb.whitepoint, xy_reference, cat16, res::GuardRef::Source))
        return res::mat3_mul(*a, srgb.rgb_to_xyz);
    return srgb.rgb_to_xyz;
}

namespace {

/* mitchell(t), B = C = 1/3, float64
 *   |t| < 1: 1/6 ((12 - 9B - 6C) |t|^3 + (-18 + 12B + 6C) t^2 + (6 - 2B))
 *   |t| < 2: 1/6 ((-B - 6C) |t|^3 + (6B + 30C) t^2 + (-12B - 48C) |t| + (8B + 24C))
 */
double mitchell(double t) {
    constexpr double B = 1.0 / 3.0, C = 1.0 / 3.0;
    const double a = std::fabs(t);
    if (a < 1.0)
        return (1.0 / 6.0) * ((12.0 - 9.0 * B - 6.0 * C) * a * a * a + (-18.0 + 12.0 * B + 6.0 * C) * a * a + (6.0 - 2.0 * B));
    if (a < 2.0)
        return (1.0 / 6.0) * ((-B - 6.0 * C) * a * a * a + (6.0 * B + 30.0 * C) * a * a + (-12.0 * B - 48.0 * C) * a +
                              (8.0 * B + 24.0 * C));
    return 0.0;
}

struct Taps {
    int64_t idx[4];
    double w[4];
};

Taps taps(double t, uint32_t L) {
    double u = t * double(L - 1);
    int64_t base;
    double frac;
    if (u <= 0.0) u = 0.0;
    if (u >= double(L - 1)) {
        base = int64_t(L) - 2;
        frac = 1.0;
    } else {
        base = int64_t(std::floor(u));
        frac = u - double(base);
    }
    Taps r{};
    for (int k = 0; k < 4; ++k) {
        int64_t n = base - 1 + k;
        if (n < 0) n = -n;
        if (n >= int64_t(L)) n = 2 * (int64_t(L) - 1) - n;
        r.idx[k] = n;
        r.w[k] = mitchell(frac + 1.0 - double(k));
    }
    return r;
}

double clip01(double v) { return v < 0.0 ? 0.0 : (v > 1.0 ? 1.0 : v); }

std::vector<double> column(const res::Mat& m, size_t c) {
    std::vector<double> v(m.rows);
    for (size_t i = 0; i < m.rows; ++i) v[i] = m(i, c);
    return v;
}

}  // namespace

Vec3 bicubic_sample(const std::vector<double>& table, uint32_t L, double t0, double t1) {
    const Taps a = taps(t0, L), b = taps(t1, L);
    Vec3 s{};
    double wsum = 0.0;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) {
            const double w = a.w[i] * b.w[j];
            const size_t base = (size_t(a.idx[i]) * L + size_t(b.idx[j])) * 3;
            for (int c = 0; c < 3; ++c) s[c] += w * table[base + c];
            wsum += w;
        }
    if (wsum != 0.0)
        for (double& v : s) v /= wsum;
    return s;
}

/* probe_density(in, grey) -> spectral density of the developed negative of a neutral grey
 *   xyz := matrix (grey, grey, grey); b := X + Y + Z
 *   (t0, t1) := q_map(x, y)
 *   h_c := bicubic_sample(table, t0, t1)_c * b
 *   d_c := interp(log10(h_c + 1e-10); log_exposure / gamma, density_curves_c)
 *   D(l) := base(l) + sum_c d_c channel_density_c(l)
 */
Spectrum probe_density(const ProbeInputs& in, double grey) {
    const res::Profile& film = *in.film;
    const Vec3 xyz = res::mat3_apply(in.srgb_to_xyz, {grey, grey, grey});
    const double b = (xyz[0] + xyz[1]) + xyz[2];
    const double bp = std::max(b, 1e-10);
    const double x = xyz[0] / bp, y = xyz[1] / bp;
    const double t0 = clip01((1.0 - x) * (1.0 - x));
    const double t1 = clip01(y / std::max(1.0 - x, 1e-10));
    const Vec3 s = bicubic_sample(*in.table, in.table_size, t0, t1);
    const double bb = std::isnan(b) ? 0.0 : b;
    std::vector<double> axis(film.log_exposure.size());
    for (size_t k = 0; k < axis.size(); ++k) axis[k] = film.log_exposure[k] / in.gamma;
    Vec3 d{};
    for (size_t c = 0; c < 3; ++c) {
        const double raw = s[c] * bb;
        const std::vector<double> curve = column(film.density_curves, c);
        d[c] = num::interp(std::log10(raw + 1e-10), axis, curve);
    }
    Spectrum D{};
    for (size_t l = 0; l < 81; ++l)
        D[l] = film.base_density[l] + ((d[0] * film.channel_density(l, 0) + d[1] * film.channel_density(l, 1)) +
                                       d[2] * film.channel_density(l, 2));
    return D;
}

// ---- paper curves ---------------------------------------------------------------------------

/* paper_model_curves(paper): density_curves_model must be "cdfs"
 *   s := paper is positive ? -1 : 1
 *   z := s (x - center[c][l]) / sigma[c][l]
 *   y[i][c] := sum_l amplitude[c][l] * 1/2 erfc(-z / sqrt 2)
 */
std::vector<double> paper_model_curves(const res::Profile& paper) {
    if (!paper.density_curves_model || paper.density_curves_model->model_type != "cdfs")
        fail_user("print profile '" + paper.id + "' has no density_curves_model of type cdfs");
    const res::DensityCurvesModel& m = *paper.density_curves_model;
    if (m.amplitudes.rows < 3 || m.centers.rows != m.amplitudes.rows || m.sigmas.rows != m.amplitudes.rows ||
        m.centers.cols != m.amplitudes.cols || m.sigmas.cols != m.amplitudes.cols)
        fail_user("print profile '" + paper.id + "' has a malformed density_curves_model");
    const double s = paper.info.is_positive() ? -1.0 : 1.0;
    const size_t K = paper.log_exposure.size();
    std::vector<double> y(K * 3);
    for (size_t i = 0; i < K; ++i) {
        const double x = paper.log_exposure[i];
        for (size_t c = 0; c < 3; ++c) {
            double v = 0.0;
            for (size_t l = 0; l < m.amplitudes.cols; ++l) {
                const double z = (s * (x - m.centers(c, l))) / m.sigmas(c, l);
                v += m.amplitudes(c, l) * (0.5 * std::erfc(-z / std::sqrt(2.0)));
            }
            y[3 * i + c] = v;
        }
    }
    return y;
}

// ---- the scan -------------------------------------------------------------------------------

void neutralise(Spectral3& eps, Spectrum& beta, Spectral3& wt) {
    for (size_t l = 0; l < 81; ++l) {
        bool bad = std::isnan(beta[l]);
        for (size_t c = 0; c < 3; ++c) bad = bad || std::isnan(eps[3 * l + c]) || std::isnan(wt[3 * l + c]);
        if (!bad) continue;
        beta[l] = 0.0;
        for (size_t c = 0; c < 3; ++c) eps[3 * l + c] = wt[3 * l + c] = 0.0;
    }
}

/* scan_tables(medium, view, cmfs)
 *   N := sum_l view(l) ybar(l)
 *   wt_k(l) := view(l) cmf_k(l) / N
 *   xyz_illum_k := sum_l view(l) cmf_k(l) / N             -- Y = 1
 *   beta, eps := medium's base density, channel density
 *   neutralise(eps, beta, wt)
 */
ScanTables scan_tables(const res::Profile& medium, const Spectrum& view, const std::vector<double>& cmfs) {
    ScanTables t;
    double N = 0.0;
    for (size_t l = 0; l < 81; ++l) N += view[l] * cmfs[3 * l + 1];
    Vec3 sum{};
    for (size_t l = 0; l < 81; ++l)
        for (size_t k = 0; k < 3; ++k) {
            t.wt[3 * l + k] = (view[l] * cmfs[3 * l + k]) / N;
            sum[k] += view[l] * cmfs[3 * l + k];
        }
    for (size_t k = 0; k < 3; ++k) t.xyz_illum[k] = sum[k] / N;
    for (size_t l = 0; l < 81; ++l) {
        t.beta[l] = medium.base_density[l];
        for (size_t c = 0; c < 3; ++c) t.eps[3 * l + c] = medium.channel_density(l, c);
    }
    neutralise(t.eps, t.beta, t.wt);
    return t;
}

PrintSetup::~PrintSetup() = default;
const res::Profile& PrintSetup::profile() const { return *p_->paper; }
const params::ParamSet& PrintSetup::params() const { return *p_->params; }
const res::ColourSpace& PrintSetup::output_space() const { return p_->out_cs; }
const fx::GlareParams& PrintSetup::glare() const { return p_->glare; }

bool levels_active(const PrintSetup::Impl& s) {
    if (!s.levels.black && !s.levels.white) return false;
    if (s.scanned) return s.film_positive;  // a scanned negative is a pass-through
    return true;
}

// ---- the per-run constants -------------------------------------------------------------------------------

namespace {

void narrow(float* dst, const double* src, size_t n) {
    for (size_t i = 0; i < n; ++i) dst[i] = num::f32(src[i]);
}

/* reference_y(s, r, d): the Y a film density triple d prints to
 *   D(l) := beta(l) + sum_c d_c eps_c(l)
 *   R_c := sum_l 2^(-D(l) log2 10) W_c(l)
 *   ell_c := log10(max(R_c g_c + o_c, 0) + 1e-10)
 *   D'_c := interp(ell_c; paper log_exposure, paper density curve)
 *   y := sum_l 2^(-D'(l) log2 10) wt_Y(l), with the paper's narrowed tables
 *   max(y, 0) + 1e-10
 */
double reference_y(const PrintSetup::Impl& s, const PrintRun& r, const Vec3& d) {
    const res::Profile& paper = *s.paper;
    Vec3 ell{};
    {
        Vec3 R{};
        for (size_t l = 0; l < 81; ++l) {
            const double D = r.beta_n[l] + ((d[0] * r.eps_n[3 * l] + d[1] * r.eps_n[3 * l + 1]) + d[2] * r.eps_n[3 * l + 2]);
            const double T = std::exp2(-D * kLog2Of10);
            for (size_t c = 0; c < 3; ++c) R[c] += T * r.w_n[3 * l + c];
        }
        for (size_t c = 0; c < 3; ++c) ell[c] = std::log10(std::max(R[c] * r.g[c] + r.o[c], 0.0) + 1e-10);
    }
    Vec3 Dp{};
    for (size_t c = 0; c < 3; ++c) {
        std::vector<double> curve(paper.density_curves.rows);
        for (size_t i = 0; i < curve.size(); ++i) curve[i] = paper.density_curves(i, c);
        Dp[c] = num::interp(ell[c], paper.log_exposure, curve);
    }
    const SpkPrintSpectralArgs& a = s.scan_args;
    double y = 0.0;
    for (size_t l = 0; l < 81; ++l) {
        const double D = ((Dp[0] * double(a.eps[3 * l]) + Dp[1] * double(a.eps[3 * l + 1])) + Dp[2] * double(a.eps[3 * l + 2])) +
                         double(a.beta[l]);
        y += std::exp2(-D * kLog2Of10) * double(a.wt[3 * l + 1]);
    }
    return std::max(y, 0.0) + 1e-10;
}

}  // namespace

/* compute_run(s, live) -> PrintRun
 *   printed path:
 *     cc := neutral + (0, m_shift, y_shift) scale        -- no cyan shift, no clamping
 *     I(l) := lamp(l) prod_k (1 - (1 - F_k(l)) (1 - 10^(-cc_k / 100)))
 *     W_c(l) := I(l) S_c(l); neutralise(eps, beta, W)
 *     g := exposure_factor(compensation ? probe at grey 2^ev : probe at mid grey, I, S)
 *     preflash > 0: o_c := preflash sum_l (10^-beta(l) I_neutral(l))_{NaN->0} S_c(l)
 *   levels active:
 *     printed, negative paper: y_k := reference_y(-Dmin); y_w := reference_y(film white density)
 *     B := 0.01 / 12.92; W := ((0.98 + 0.055) / 1.055)^2.4
 *     black only: W := y_w; white only: B := y_k
 *     m := (W - B) / (y_w - y_k + 1e-10); q := B - m y_k
 *     printed, negative paper:
 *       g* := (mid - q) / m
 *       e* := interp(-log10(g*) - mean base; mean paper curve, log_exposure)
 *       e0 := interp(-log10(mid) - mean base; mean paper curve, log_exposure)
 *       kappa := 10^(e* - e0)
 *   k_gain := f32(print_exposure kappa)
 *   narrow the enlarger tables
 */
std::shared_ptr<PrintRun> compute_run(const PrintSetup::Impl& s, const LiveParams& live) {
    auto run = std::make_shared<PrintRun>();
    PrintRun& r = *run;
    r.live = live;

    if (!s.scanned) {
        const Vec3 cc = {s.neutral[0], s.neutral[1] + live.m_filter_shift * live.filter_shift_scale,
                         s.neutral[2] + live.y_filter_shift * live.filter_shift_scale};
        r.light = filtered_light(s.lamp, s.filters, cc);

        r.eps_n = s.film_eps;
        r.beta_n = s.film_beta;
        for (size_t l = 0; l < 81; ++l)
            for (size_t c = 0; c < 3; ++c) r.w_n[3 * l + c] = r.light[l] * s.paper_sens[3 * l + c];
        neutralise(r.eps_n, r.beta_n, r.w_n);

        const double phi = s.compensation ? exposure_factor(s.probe_comp, r.light, s.paper_sens)
                                          : exposure_factor(s.probe_mid, r.light, s.paper_sens);
        r.g = {phi, phi, phi};

        if (live.preflash_exposure > 0.0) {
            const Spectrum pre = filtered_light(s.lamp, s.filters, s.neutral);
            Vec3 sum{};
            for (size_t l = 0; l < 81; ++l) {
                double L = std::pow(10.0, -s.film_beta[l]) * pre[l];
                if (std::isnan(L)) L = 0.0;
                for (size_t c = 0; c < 3; ++c) sum[c] += L * s.paper_sens[3 * l + c];
            }
            for (size_t c = 0; c < 3; ++c) r.o[c] = live.preflash_exposure * sum[c];
        }
    }

    r.levels_active = levels_active(s);
    if (r.levels_active) {
        if (!s.scanned && !s.paper_positive) {
            r.y_k = reference_y(s, r, {-fx::kGrainDminDensity[0], -fx::kGrainDminDensity[1], -fx::kGrainDminDensity[2]});
            r.y_w = reference_y(s, r, s.film_white_density);
        }
        double B = s.levels.B, W = s.levels.W;
        if (s.levels.black && !s.levels.white) W = r.y_w;
        if (s.levels.white && !s.levels.black) B = r.y_k;
        r.m = (W - B) / (r.y_w - r.y_k + 1e-10);
        r.q = B - r.m * r.y_k;
        if (!s.scanned && !s.paper_positive) {
            const res::Profile& paper = *s.paper;
            const double g_star = (kMidGrey - r.q) / r.m;
            const double e_star = num::interp(-std::log10(g_star) - s.paper_mean_base, s.paper_mean_curve, paper.log_exposure);
            const double e_0 = num::interp(-std::log10(kMidGrey) - s.paper_mean_base, s.paper_mean_curve, paper.log_exposure);
            r.kappa_bw = std::pow(10.0, e_star - e_0);
        }
    }
    r.k_gain = num::f32(live.print_exposure * r.kappa_bw);

    r.enl.mode = SPK_PRINT_MODE_LOG_EXPOSURE;
    narrow(r.enl.eps, r.eps_n.data(), 243);
    narrow(r.enl.beta, r.beta_n.data(), 81);
    narrow(r.enl.wt, r.w_n.data(), 243);
    narrow(r.enl.g, r.g.data(), 3);
    narrow(r.enl.o, r.o.data(), 3);
    return run;
}

}  // namespace spk::print
