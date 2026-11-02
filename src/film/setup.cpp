// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 Guangzhou Songzi Intelligent Technology Services Co., Ltd.

/* The film set-up: float64 on the CPU, narrowed once to float32 for the kernels.
 *
 * The spectral table is computed for each set-up; it is not cached between pipelines.
 */
#include <cmath>
#include <cstring>
#include <limits>

#include "film/film_internal.hpp"
#include "util/numeric.hpp"

namespace spk::film {

// ---- FilmSetup accessors -----------------------------------------------------------------------
FilmSetup::~FilmSetup() = default;
const res::Profile& FilmSetup::profile() const { return *p_->profile; }
const params::ParamSet& FilmSetup::params() const { return p_->params; }
bool FilmSetup::is_positive() const { return p_->positive; }
const res::ColourSpace& FilmSetup::input_space() const { return p_->input_space; }
const SpectralTable& FilmSetup::spectral_table() const { return *p_->table; }

/* coupler_gammas(info) -> rows = donor (R, G, B), columns = receiver
 *   fixed sets for two slide stocks, for any other positive, and for negatives
 * coupler_matrix(info, amount): M[r][c] := amount gamma[r][c]
 */
Mat3 coupler_gammas(const res::ProfileInfo& info) {
    auto make = [](double sr, double sg, double sb, double rg, double rb, double gr, double gb, double br, double bg) {
        return Mat3{sr, rg, rb, gr, sg, gb, br, bg, sb};
    };
    if (info.stock == "fujifilm_velvia_100") return make(0.108, 0.072, 0.054, 0.108, 0.054, 0.072, 0.054, 0.054, 0.054);
    if (info.stock == "fujifilm_provia_100f") return make(0.156, 0.104, 0.078, 0.156, 0.078, 0.104, 0.078, 0.078, 0.078);
    if (info.is_positive()) return make(0.12, 0.08, 0.06, 0.12, 0.06, 0.08, 0.06, 0.06, 0.06);
    return make(0.336, 0.319, 0.273, 0.353, 0.302, 0.154, 0.353, 0.168, 0.226);
}

Mat3 coupler_matrix(const res::ProfileInfo& info, double amount) {
    constexpr double kSame = 1.0, kInter = 1.0;  // I_s, I_i
    const Mat3 g = coupler_gammas(info);
    Mat3 m{};
    for (size_t r = 0; r < 3; ++r)
        for (size_t c = 0; c < 3; ++c) m[3 * r + c] = amount * ((r == c ? kSame : kInter) * g[3 * r + c]);
    return m;
}

/* build_curve_tables(film, gamma, M, positive)
 *   l_k := log_exposure[k]; N := normalized curves
 *   x[k] := f32(l_k / gamma); inv[k] := f32(1 / (x64[k+1] - x64[k])), 0 for a zero step
 *   y[c K + k] := f32(N_c[k])
 *   pre-inverted curve: the curve that, after the coupler correction, gives N back on a uniform field
 *     sigma_c[k] := positive ? dmax_c - N_c[k] : N_c[k]
 *     l0_m[k] := l_k - sum_c sigma_c[k] M[c][m]
 *     N0_m[k] := interp(l_k; l0_m, N_m); y0[m K + k] := f32(N0_m[k])
 */
CurveTables build_curve_tables(const res::Profile& film, double gamma, const Mat3& m, bool positive) {
    const size_t K = film.curve_points();
    const std::vector<double>& le = film.log_exposure;
    const res::Mat& N = film.normalized_curves;
    CurveTables t;
    t.K = uint32_t(K);
    t.x.resize(K);
    t.inv.assign(K, 0.0f);
    t.y.resize(3 * K);
    t.y0.resize(3 * K);
    t.n0.resize(3 * K);
    for (size_t k = 0; k < K; ++k) {
        t.x[k] = num::f32(le[k] / gamma);
        if (k + 1 < K) {
            const double d = le[k + 1] / gamma - le[k] / gamma;
            t.inv[k] = d == 0.0 ? 0.0f : num::f32(1.0 / d);
        }
        for (size_t c = 0; c < 3; ++c) t.y[c * K + k] = num::f32(N(k, c));
    }
    std::vector<double> xs(K), ys(K);
    for (size_t mm = 0; mm < 3; ++mm) {
        for (size_t k = 0; k < K; ++k) {
            double s = 0.0;
            for (size_t c = 0; c < 3; ++c) {
                const double sigma = positive ? film.dmax[c] - N(k, c) : N(k, c);
                s += sigma * m[3 * c + mm];
            }
            xs[k] = le[k] - s;
            ys[k] = N(k, mm);
        }
        for (size_t k = 0; k < K; ++k) {
            const double v = num::interp(le[k], xs, ys);
            t.n0[3 * k + mm] = v;
            t.y0[mm * K + k] = num::f32(v);
        }
    }
    return t;
}

/* ---- the spectral table ----
 * q_map(x, y): chromaticity -> table coordinates
 *   o := 1 - x; (clip01(o^2), clip01(y / max(o, 1e-10)))
 */
namespace {

constexpr double kGcThreshold = 0.0, kGcLimit = 1.0, kGcPower = 6.0;  // input gamut compression

double clip01(double v) { return v < 0.0 ? 0.0 : (v > 1.0 ? 1.0 : v); }

std::array<double, 2> q_map(double x, double y) {
    const double o = 1.0 - x;
    return {clip01(o * o), clip01(y / std::max(o, 1e-10))};
}

}  // namespace

/* build_spectral_table_f64(blob, S, window, I, size_out) -> T' (L x L x 3)
 *   spectra := hanatos/spectra_lut (L, L, 81)
 *   window of 4 (cuv, suv, cir, sir):
 *     om(l) := 1/2 (1 + erf((l - cuv) / (suv sqrt 2))) * 1/2 (1 - erf((l - cir) / (sir sqrt 2)))
 *     f_c := sum(S_c om I) / sum(S_c I)
 *     W_c := S_c om / f_c                                -- white balance preserved
 *   otherwise W := S; a window of any other non-zero length: SPK_ERR_USER
 *   T[i][j][c] := sum_l spectra[i][j][l] W_c(l), ascending l
 *   input gamut compression baked into the table:
 *     w := xy of I
 *     poly := spectrum locus, 380..700 nm, closed
 *     for each node (i, j):
 *       t0 := i / (L-1); t1 := j / (L-1); p := (1 - sqrt t0, t1 sqrt t0)
 *       d := |p - w|; d < 1e-9: p' := p
 *       else beta := first hit of ray w -> p on poly
 *            r := d / beta; z := r                       -- threshold 0, limit 1, power 6
 *            r' := z / (1 + z^6)^(1/6)
 *            p' := w + (p - w) / d * r' beta
 *       T'[i][j] := bilinear T at q_map(p') (L-1), clamped
 */
std::vector<double> build_spectral_table_f64(const res::Blob& blob, const std::array<double, 81 * 3>& S,
                                             const std::vector<double>& window, const res::Spectrum& I,
                                             uint32_t* size_out) {
    const res::BlobEntry* e = blob.find("hanatos/spectra_lut");
    if (!e || e->ndim != 3 || e->dims[0] != e->dims[1] || e->dims[2] != 81 || e->dims[0] < 2)
        fail_user("film set-up: the constants blob has no usable hanatos/spectra_lut (L x L x 81)");
    const uint32_t L = e->dims[0];
    const std::vector<double> spectra = blob.f64("hanatos/spectra_lut", {L, L, 81});
    if (size_out) *size_out = L;

    std::array<double, 81 * 3> W = S;
    if (!window.empty()) {
        if (window.size() != 4)
            fail_user("film set-up: only the 4-parameter hanatos2025_adaptation_window_params is built");
        const double cuv = window[0], suv = window[1], cir = window[2], sir = window[3];
        const double r2 = std::sqrt(2.0);
        std::array<double, 81> om{};
        for (size_t l = 0; l < 81; ++l) {
            const double lam = 380.0 + 5.0 * double(l);
            om[l] = 0.5 * (1.0 + std::erf((lam - cuv) / (suv * r2))) * (0.5 * (1.0 - std::erf((lam - cir) / (sir * r2))));
        }
        for (size_t c = 0; c < 3; ++c) {
            double num_ = 0.0, den = 0.0;
            for (size_t l = 0; l < 81; ++l) {
                num_ += S[3 * l + c] * om[l] * I[l];
                den += S[3 * l + c] * I[l];
            }
            const double f = num_ / den;
            for (size_t l = 0; l < 81; ++l) W[3 * l + c] = S[3 * l + c] * om[l] / f;
        }
    }

    const size_t nodes = size_t(L) * L;
    std::vector<double> T(nodes * 3);
    for (size_t n = 0; n < nodes; ++n) {
        const double* s = spectra.data() + n * 81;
        double a0 = 0.0, a1 = 0.0, a2 = 0.0;
        for (size_t l = 0; l < 81; ++l) {
            a0 += s[l] * W[3 * l];
            a1 += s[l] * W[3 * l + 1];
            a2 += s[l] * W[3 * l + 2];
        }
        T[3 * n] = a0;
        T[3 * n + 1] = a1;
        T[3 * n + 2] = a2;
    }

    const std::vector<double> cmf = blob.f64("colour/cmfs_1931_2deg");
    if (cmf.size() != 243) fail_user("film set-up: colour/cmfs_1931_2deg must be 81 x 3");
    std::array<double, 2> w{};
    {
        double X = 0, Y = 0, Z = 0;
        for (size_t l = 0; l < 81; ++l) {
            X += I[l] * cmf[3 * l];
            Y += I[l] * cmf[3 * l + 1];
            Z += I[l] * cmf[3 * l + 2];
        }
        const double t = X + Y + Z;
        w = {X / t, Y / t};
    }
    std::vector<std::array<double, 2>> poly;
    for (size_t l = 0; l <= 64; ++l) {  // 380 ... 700 nm
        const double X = cmf[3 * l], Y = cmf[3 * l + 1], Z = cmf[3 * l + 2];
        const double s = std::max(X + Y + Z, 1e-12);
        poly.push_back({X / s, Y / s});
    }
    poly.push_back(poly.front());

    const double last = double(L - 1);
    std::vector<double> Tp(nodes * 3);
    for (uint32_t i = 0; i < L; ++i) {
        for (uint32_t j = 0; j < L; ++j) {
            const double t0 = double(i) / last, t1 = double(j) / last;
            const double sq = std::sqrt(t0);
            const std::array<double, 2> p{1.0 - sq, t1 * sq};
            const double dx = p[0] - w[0], dy = p[1] - w[1];
            const double d = std::sqrt(dx * dx + dy * dy);
            std::array<double, 2> pp = p;
            if (!(d < 1e-9)) {
                const double dm = std::max(d, 1e-12);
                const double ux = dx / dm, uy = dy / dm;
                double beta = std::numeric_limits<double>::infinity();
                for (size_t k = 0; k + 1 < poly.size(); ++k) {
                    const double ex = poly[k + 1][0] - poly[k][0], ey = poly[k + 1][1] - poly[k][1];
                    const double den = ux * ey - uy * ex;
                    if (std::fabs(den) <= 1e-12) continue;
                    const double ox = w[0] - poly[k][0], oy = w[1] - poly[k][1];
                    const double s = (-ox * ey + oy * ex) / den;
                    const double v = (-ox * uy + oy * ux) / den;
                    if (v >= 0.0 && v <= 1.0 && s > 1e-9 && s < beta) beta = s;
                }
                const double r = d / std::max(beta, 1e-12);
                double rp = r;
                if (r > kGcThreshold) {
                    const double z = (r - kGcThreshold) / (kGcLimit - kGcThreshold);
                    rp = kGcThreshold + (kGcLimit - kGcThreshold) * z / std::pow(1.0 + std::pow(z, kGcPower), 1.0 / kGcPower);
                }
                pp = {w[0] + ux * rp * beta, w[1] + uy * rp * beta};
            }
            const std::array<double, 2> tq = q_map(pp[0], pp[1]);
            auto axis = [&](double tv, uint32_t& a0, uint32_t& a1, double& f) {
                double c = tv * last;
                if (!(c > 0.0)) c = 0.0;
                if (c > last) c = last;
                const double fl = std::floor(c);
                a0 = uint32_t(fl);
                a1 = std::min(a0 + 1, L - 1);
                f = c - fl;
            };
            uint32_t i0, i1, j0, j1;
            double fi, fj;
            axis(tq[0], i0, i1, fi);
            axis(tq[1], j0, j1, fj);
            const double* v00 = &T[3 * (size_t(i0) * L + j0)];
            const double* v01 = &T[3 * (size_t(i0) * L + j1)];
            const double* v10 = &T[3 * (size_t(i1) * L + j0)];
            const double* v11 = &T[3 * (size_t(i1) * L + j1)];
            double* o = &Tp[3 * (size_t(i) * L + j)];
            for (size_t c = 0; c < 3; ++c)
                o[c] = v00[c] * (1.0 - fi) * (1.0 - fj) + v01[c] * (1.0 - fi) * fj + v10[c] * fi * (1.0 - fj) +
                       v11[c] * fi * fj;
        }
    }
    return Tp;
}

/* ---- CPU copies of the kernels, binary32, same operation order (tests) ----
 * upsample: see spk_film_upsample. curve_lookup: see film.metal.
 */
namespace cpu {

float mitchell(float u) {
    const float a = std::fabs(u);
    if (a < 1.0f) return (1.0f / 6.0f) * (7.0f * a * a * a - 12.0f * a * a + 16.0f / 3.0f);
    if (a < 2.0f) return (1.0f / 6.0f) * ((-7.0f / 3.0f) * a * a * a + 12.0f * a * a - 20.0f * a + 32.0f / 3.0f);
    return 0.0f;
}

static void mitchell_axis(float t, uint32_t L, int* idx, float* w) {
    const float last = float(L - 1);
    float u = t * last;
    int base;
    float frac;
    if (u <= 0.0f) u = 0.0f;
    if (u >= last) {
        base = int(L) - 2;
        frac = 1.0f;
    } else {
        base = int(std::floor(u));
        frac = u - float(base);
    }
    const int n = int(L);
    for (int k = 0; k < 4; ++k) {
        int i = base - 1 + k;
        if (i < 0) i = -i;
        if (i >= n) i = 2 * (n - 1) - i;
        idx[k] = i;
    }
    w[0] = mitchell(frac + 1.0f);
    w[1] = mitchell(frac);
    w[2] = mitchell(frac - 1.0f);
    w[3] = mitchell(frac - 2.0f);
}

std::array<float, 3> upsample(const std::array<float, 9>& a, const float* table, uint32_t L, const float rgb[3]) {
    const float r = rgb[0], g = rgb[1], b = rgb[2];
    const float X = (a[0] * r + a[1] * g) + a[2] * b;
    const float Y = (a[3] * r + a[4] * g) + a[5] * b;
    const float Z = (a[6] * r + a[7] * g) + a[8] * b;
    const float bs = (X + Y) + Z;
    const float d = bs > 1e-10f ? bs : 1e-10f;
    const float x = X / d, y = Y / d, o = 1.0f - x;
    const float t0 = std::fmin(std::fmax(o * o, 0.0f), 1.0f);
    const float t1 = std::fmin(std::fmax(y / (o > 1e-10f ? o : 1e-10f), 0.0f), 1.0f);
    const float bp = std::isnan(bs) ? 0.0f : bs;
    int ia[4], ib[4];
    float wa[4], wb[4];
    mitchell_axis(t0, L, ia, wa);
    mitchell_axis(t1, L, ib, wb);
    float s = 0.0f, acc[3] = {0.0f, 0.0f, 0.0f};
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) {
            const float w = wa[i] * wb[j];
            const float* t = table + 3 * (size_t(ia[i]) * L + size_t(ib[j]));
            s += w;
            for (int c = 0; c < 3; ++c) acc[c] += w * t[c];
        }
    std::array<float, 3> h{};
    for (int c = 0; c < 3; ++c) h[c] = bp * (s != 0.0f ? acc[c] / s : acc[c]);
    return h;
}

float curve_lookup(const float* x, const float* inv, const float* y, uint32_t K, float v) {
    if (v <= x[0]) return y[0];
    if (v >= x[K - 1]) return y[K - 1];
    uint32_t lo = 0, hi = K;
    while (lo < hi) {
        const uint32_t mid = lo + (hi - lo) / 2;
        if (x[mid] <= v) lo = mid + 1;
        else hi = mid;
    }
    const uint32_t l = lo == 0 ? 0 : lo - 1;
    const float t = (v - x[l]) * inv[l];
    return y[l] + t * (y[l + 1] - y[l]);
}

}  // namespace cpu

/* build_film_setup(in)
 *   input space := colour_space(input_color_space); unknown: SPK_ERR_USER
 *   I := illuminant(reference_illuminant), mean 1; unknown: SPK_ERR_USER
 *   A := von_kries(white of input -> xy of I, CAT16) . RGB->XYZ(input); no product when allclose
 *   S_c(l) := 10^log_sensitivity_c(l), NaN -> 0
 *   table := build_spectral_table_f64(S, window, I); upload f32
 *   halation := resolve_halation(info, p)
 *   grain := build_grain_setup, if grain_active
 *   M := coupler_matrix(info, dir_couplers_amount)
 *   curves := build_curve_tables(film, density_curve_gamma, M, positive)
 *   upload [x K][inv K][y 3K][y0 3K]
 */
std::shared_ptr<const FilmSetup> build_film_setup(const FilmSetupInputs& in) {
    using params::P;
    if (!in.film) fail_internal("build_film_setup: no film profile");
    const res::Profile& film = *in.film;
    const params::ParamSet& p = in.params;
    const res::Blob& blob = *in.res.blob;

    auto impl = std::make_unique<FilmSetup::Impl>(p);
    FilmSetup::Impl& s = *impl;
    s.profile = in.film;
    s.positive = film.info.is_positive();

    const std::string& space = p.s(P::input_color_space);
    auto cs = res::colour_space(blob, space);
    if (!cs) fail_user("input_color_space '" + space + "' is not a colour space in the constants blob");
    s.input_space = *cs;
    s.decode = p.b(P::input_cctf_decoding);

    const std::string& ill = film.info.reference_illuminant;
    auto I = res::illuminant(blob, ill);
    if (!I) fail_user("film '" + film.id + "': unknown reference_illuminant '" + ill + "'");
    s.illuminant = *I;
    s.illuminant_xy = res::chromaticity(blob, s.illuminant);

    auto cat16 = res::cat_matrix(blob, "CAT16");
    if (!cat16) fail_user("film set-up: the constants blob has no colour/cat/CAT16");
    auto C = res::von_kries(s.input_space.whitepoint, s.illuminant_xy, *cat16, res::GuardRef::Source);
    s.input_matrix = C ? res::mat3_mul(*C, s.input_space.rgb_to_xyz) : s.input_space.rgb_to_xyz;
    for (size_t i = 0; i < 9; ++i) s.input_matrix_f32[i] = num::f32(s.input_matrix[i]);

    if (film.log_sensitivity.rows != 81 || film.log_sensitivity.cols != 3)
        fail_user("film '" + film.id + "': log_sensitivity must be 81 x 3");
    for (size_t l = 0; l < 81; ++l)
        for (size_t c = 0; c < 3; ++c) {
            const double v = std::pow(10.0, film.log_sensitivity(l, c));
            s.sensitivity[3 * l + c] = std::isnan(v) ? 0.0 : v;
        }

    auto t = std::make_shared<SpectralTable>();
    t->f64 = build_spectral_table_f64(blob, s.sensitivity, film.adaptation_window_params, s.illuminant, &t->size);
    t->f32 = in.gpu.persistent(t->f64.size() * sizeof(float), gpu::MemTag::Table);
    float* f = t->f32.as<float>();
    for (size_t i = 0; i < t->f64.size(); ++i) f[i] = num::f32(t->f64[i]);
    s.table = t;

    s.halation = fx::resolve_halation(film.info, p);
    if (p.b(P::grain_active)) s.grain = fx::build_grain_setup(film, p);

    if (film.curve_points() < 2) fail_user("film '" + film.id + "': log_exposure needs at least 2 points");
    s.couplers = p.b(P::dir_couplers_active);
    s.coupler_m = coupler_matrix(film.info, p.f(P::dir_couplers_amount));
    for (size_t i = 0; i < 9; ++i) s.coupler_m_f32[i] = num::f32(s.coupler_m[i]);
    for (size_t c = 0; c < 3; ++c) s.dmax_f32[c] = num::f32(film.dmax[c]);
    s.curves = build_curve_tables(film, p.f(P::density_curve_gamma), s.coupler_m, s.positive);
    const size_t K = s.curves.K;
    s.curve_buffer = in.gpu.persistent(8 * K * sizeof(float), gpu::MemTag::Table);
    float* cb = s.curve_buffer.as<float>();
    std::memcpy(cb, s.curves.x.data(), K * sizeof(float));
    std::memcpy(cb + K, s.curves.inv.data(), K * sizeof(float));
    std::memcpy(cb + 2 * K, s.curves.y.data(), 3 * K * sizeof(float));
    std::memcpy(cb + 5 * K, s.curves.y0.data(), 3 * K * sizeof(float));

    std::shared_ptr<FilmSetup> out(new FilmSetup());
    out->p_ = std::move(impl);
    return out;
}

}  // namespace spk::film
