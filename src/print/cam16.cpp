// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 Guangzhou Songzi Intelligent Technology Services Co., Ltd.

/* CAM16-UCS output gamut compression set-up, the per-space cache, GamutOptions::parse and
 * spk_output_transform. Published: Li et al., CAM16, Color Research and Application, 2017.
 *
 * everything here is float64; the C_max table is narrowed to float32 once
 */
#include <cmath>
#include <map>
#include <mutex>

#include "print/internal.hpp"
#include "res/resources.hpp"
#include "util/error.hpp"
#include "util/numeric.hpp"

namespace spk::print {

namespace cam16 {

namespace {
double sgn(double v) { return v > 0.0 ? 1.0 : (v < 0.0 ? -1.0 : 0.0); }
}  // namespace

/* post_adapt(v, F_L) = 400 sgn(v) f / (27.13 + f) + 0.1, f = (F_L |v| / 100)^0.42 */
double post_adapt(double v, double F_L) {
    const double f = std::pow((F_L * std::fabs(v)) / 100.0, 0.42);
    return ((400.0 * sgn(v)) * f) / (27.13 + f) + 0.1;
}

/* make_constants(wp): L_A = 64, Y_b = 20, F = 1, c = 0.69, N_c = 1, Y_w = 100
 *   n := Y_b / Y_w; k := 1 / (5 L_A + 1)
 *   F_L := 0.2 k^4 (5 L_A) + 0.1 (1 - k^4)^2 cbrt(5 L_A)
 *   N_bb := N_cb := 0.725 (1/n)^0.2; z := 1.48 + sqrt n
 *   D := clamp(F (1 - exp((-L_A - 42) / 92) / 3.6), 0, 1)
 *   rgb_w := M16 100 XYZ(wp); D_i := D Y_w / rgb_w_i + 1 - D
 *   A_w := ((2 Ra + Ga + Ba / 20) - 0.305) N_bb, at the adopted white
 *   J_white := forward(XYZ(wp)).Jp
 */
Constants make_constants(const std::array<double, 2>& wp) {
    Constants k;
    k.n = k.Y_b / k.Y_w;
    k.k = 1.0 / (5.0 * k.L_A + 1.0);
    const double k4 = k.k * k.k * k.k * k.k;
    const double la5 = 5.0 * k.L_A;
    k.F_L = (0.2 * k4) * la5 + (0.1 * ((1.0 - k4) * (1.0 - k4))) * std::cbrt(la5);
    k.N_bb = 0.725 * std::pow(1.0 / k.n, 0.2);
    k.N_cb = k.N_bb;
    k.z = 1.48 + std::sqrt(k.n);
    double D = k.F * (1.0 - (1.0 / 3.6) * std::exp((-k.L_A - 42.0) / 92.0));
    k.D = D < 0.0 ? 0.0 : (D > 1.0 ? 1.0 : D);
    const Vec3 xyz = res::xy_to_XYZ(wp);
    const Vec3 xyz_w = {100.0 * xyz[0], 100.0 * xyz[1], 100.0 * xyz[2]};
    k.rgb_w = res::mat3_apply(k.m16, xyz_w);
    for (int i = 0; i < 3; ++i) k.D_rgb[i] = (k.D * k.Y_w) / k.rgb_w[i] + 1.0 - k.D;
    const double ra = post_adapt(k.D_rgb[0] * k.rgb_w[0], k.F_L);
    const double ga = post_adapt(k.D_rgb[1] * k.rgb_w[1], k.F_L);
    const double ba = post_adapt(k.D_rgb[2] * k.rgb_w[2], k.F_L);
    k.A_w = (((2.0 * ra + ga) + ba / 20.0) - 0.305) * k.N_bb;
    k.m16_inv = res::mat3_inverse(k.m16);
    k.J_white = forward(k, xyz).Jp;
    return k;
}

/* forward(k, xyz), Y_white = 1
 *   rgb := M16 100 xyz; (Ra, Ga, Ba) := post_adapt(D_i rgb_i)
 *   a := Ra - 12 Ga / 11 + Ba / 11; b := (Ra + Ga - 2 Ba) / 9; h := atan2(b, a)
 *   et := (cos(h + 2) + 3.8) / 4
 *   A := (2 Ra + Ga + Ba / 20 - 0.305) N_bb
 *   J := 100 sgn(A / A_w) |A / A_w|^(c z)
 *   t := (50000/13 N_c N_cb et sqrt(a^2 + b^2)) / (Ra + Ga + 21 Ba / 20)
 *   C := t^0.9 sqrt(|J| / 100) (1.64 - 0.29^n)^0.73; M := C F_L^(1/4)
 *   M' := ln(1 + 0.0228 M) / 0.0228; J' := 1.7 J / (1 + 0.007 J)
 */
Forward forward(const Constants& k, const Vec3& xyz) {
    const Vec3 x100 = {100.0 * xyz[0], 100.0 * xyz[1], 100.0 * xyz[2]};
    const Vec3 rgb = res::mat3_apply(k.m16, x100);
    const double ra = post_adapt(k.D_rgb[0] * rgb[0], k.F_L);
    const double ga = post_adapt(k.D_rgb[1] * rgb[1], k.F_L);
    const double ba = post_adapt(k.D_rgb[2] * rgb[2], k.F_L);
    const double a = (ra - (12.0 * ga) / 11.0) + ba / 11.0;
    const double b = ((ra + ga) - 2.0 * ba) / 9.0;
    Forward f;
    f.h = std::atan2(b, a);
    const double et = 0.25 * (std::cos(f.h + 2.0) + 3.8);
    const double A = (((2.0 * ra + ga) + ba / 20.0) - 0.305) * k.N_bb;
    const double ratio = A / k.A_w;
    f.J = (100.0 * sgn(ratio)) * std::pow(std::fabs(ratio), k.c * k.z);
    const double den = (ra + ga) + (21.0 * ba) / 20.0;
    const double t = den != 0.0 ? (((((50000.0 / 13.0) * k.N_c) * k.N_cb) * et) * std::sqrt(a * a + b * b)) / den : 0.0;
    f.C = t > 0.0 ? (std::pow(t, 0.9) * std::sqrt(std::fabs(f.J) / 100.0)) * std::pow(1.64 - std::pow(0.29, k.n), 0.73)
                  : 0.0;
    f.M = f.C * std::pow(k.F_L, 0.25);
    f.Mp = std::log(1.0 + 0.0228 * f.M) / 0.0228;
    f.Jp = (1.7 * f.J) / (1.0 + 0.007 * f.J);
    return f;
}

/* inverse_unguarded(k, J', a', b'): the table-construction inverse
 *   no guard at |v| >= 400; uses the computed M16 inverse, not the published 8-digit one
 */
Vec3 inverse_unguarded(const Constants& k, double Jp, double ap, double bp) {
    const double Mp = std::sqrt(ap * ap + bp * bp);
    const double h = std::atan2(bp, ap);
    const double M = (std::exp(0.0228 * Mp) - 1.0) / 0.0228;
    const double J = Jp / (1.7 - 0.007 * Jp);
    const double C = M / std::pow(k.F_L, 0.25);
    const double A = (k.A_w * sgn(J)) * std::pow(std::fabs(J / 100.0), 1.0 / (k.c * k.z));
    const double sq = std::sqrt(std::fabs(J) / 100.0);
    const double t = (sq > 0.0 && C > 0.0) ? std::pow(C / (sq * std::pow(1.64 - std::pow(0.29, k.n), 0.73)), 1.0 / 0.9) : 0.0;
    const double et = 0.25 * (std::cos(h + 2.0) + 3.8);
    const double p2 = A / k.N_bb + 0.305;
    const double p3 = 21.0 / 20.0;
    double a = 0.0, b = 0.0;
    if (t != 0.0) {
        const double p1 = ((((50000.0 / 13.0) * k.N_c) * k.N_cb) * et) / t;
        const double sh = std::sin(h), ch = std::cos(h);
        if (std::fabs(sh) >= std::fabs(ch)) {
            const double p4 = p1 / sh;
            b = ((p2 * (2.0 + p3)) * (460.0 / 1403.0)) /
                (((p4 + ((2.0 + p3) * (220.0 / 1403.0)) * (ch / sh)) - 27.0 / 1403.0) + p3 * (6300.0 / 1403.0));
            a = (b * ch) / sh;
        } else {
            const double p5 = p1 / ch;
            a = ((p2 * (2.0 + p3)) * (460.0 / 1403.0)) /
                ((p5 + (2.0 + p3) * (220.0 / 1403.0)) - (27.0 / 1403.0 - p3 * (6300.0 / 1403.0)) * (sh / ch));
            b = (a * sh) / ch;
        }
    }
    const double xa[3] = {((460.0 * p2 + 451.0 * a) + 288.0 * b) / 1403.0,
                          ((460.0 * p2 - 891.0 * a) - 261.0 * b) / 1403.0,
                          ((460.0 * p2 - 220.0 * a) - 6300.0 * b) / 1403.0};
    Vec3 rc{};
    for (int i = 0; i < 3; ++i) {
        const double v = xa[i] - 0.1;
        const double av = std::fabs(v);
        const double beta = (27.13 * av) / (400.0 - av);  // unguarded: negative past 400
        const double pw = sgn(beta) * std::pow(std::fabs(beta), 1.0 / 0.42);
        rc[i] = (((100.0 / k.F_L) * sgn(v)) * pw) / k.D_rgb[i];
    }
    const Vec3 x100 = res::mat3_apply(k.m16_inv, rc);
    return {x100[0] / 100.0, x100[1] / 100.0, x100[2] / 100.0};
}

/* cmax_table(k, m_to_rgb) -> 64 x 720: rows J', columns h
 *   J'_i := 1 + 109 i / 63; h_j := -pi + 2 pi j / 720
 *   lo := 0; hi := 150
 *   18 bisection steps on chroma:
 *     in := every channel of RGB(J', mid cos h, mid sin h) is in [-1e-6, 1 + 1e-6]
 *     in: lo := mid; else hi := mid
 *   C_max := lo
 */
std::vector<double> cmax_table(const Constants& k, const Mat3& m_to_rgb) {
    std::vector<double> t(size_t(kRows) * kCols);
    for (uint32_t i = 0; i < kRows; ++i) {
        const double Jp = 1.0 + (109.0 * double(i)) / 63.0;
        for (uint32_t j = 0; j < kCols; ++j) {
            const double h = -M_PI + (2.0 * M_PI * double(j)) / 720.0;
            const double ch = std::cos(h), sh = std::sin(h);
            double lo = 0.0, hi = 150.0;
            for (int step = 0; step < 18; ++step) {
                const double mid = (lo + hi) / 2.0;
                const Vec3 xyz = inverse_unguarded(k, Jp, mid * ch, mid * sh);
                const Vec3 rgb = res::mat3_apply(m_to_rgb, xyz);
                bool in = true;
                for (double v : rgb) in = in && (v >= -1e-6 && v <= 1.0 + 1e-6);
                if (in) lo = mid; else hi = mid;
            }
            t[size_t(i) * kCols + j] = lo;
        }
    }
    return t;
}

}  // namespace cam16

/* cam16_k(c, g) -> the 22 numbers of gamut_compress.k
 *   [F_L, N_bb, N_cb, n, z, A_w, c, N_c, J first, J last, h first, h step, knee 0..2, lc 0..2,
 *    J_white, D_R, D_G, D_B]
 *   lightness compression inactive: lc = (0, 1, 1)
 */
std::vector<double> cam16_k(const cam16::Constants& c, const GamutOptions& g) {
    const bool lc = g.lightness_compression_active;
    return {c.F_L,
            c.N_bb,
            c.N_cb,
            c.n,
            c.z,
            c.A_w,
            c.c,
            c.N_c,
            cam16::kJFirst,
            cam16::kJLast,
            -M_PI,
            2.0 * M_PI / 720.0,
            g.knee[0],
            g.knee[1],
            g.knee[2],
            lc ? g.lightness_compression[0] : 0.0,
            lc ? g.lightness_compression[1] : 1.0,
            lc ? g.lightness_compression[2] : 1.0,
            c.J_white,
            c.D_rgb[0],
            c.D_rgb[1],
            c.D_rgb[2]};
}

namespace {

/* resolve_space(res, name): SPK_ERR_USER when unknown, without matrices, or whitepoint y = 0; never a fallback */
res::ColourSpace resolve_space(const res::Resources& res, std::string_view name) {
    if (!res::cctf_mode_of(name)) fail_user("unknown colour space '" + std::string(name) + "'");
    auto cs = res::colour_space(*res.blob, name);
    if (!cs) fail_user("colour space '" + std::string(name) + "' has no baked matrices or whitepoint");
    if (cs->whitepoint[1] == 0.0) fail_user("colour space '" + std::string(name) + "' has a whitepoint with y = 0");
    return *cs;
}

json::Value mat_json(const Mat3& m) {
    json::Array a;
    for (double v : m) a.emplace_back(v);
    return json::Value(std::move(a));
}

}  // namespace

std::unique_ptr<Cam16Tables> build_cam16_tables(const res::Resources& res, std::string_view space) {
    const res::ColourSpace cs = resolve_space(res, space);
    const cam16::Constants c = cam16::make_constants(cs.whitepoint);
    auto t = std::make_unique<Cam16Tables>();
    t->space = cs.name;
    t->m_to_xyz = cs.rgb_to_xyz;
    t->m_to_rgb = cs.xyz_to_rgb;
    t->k = cam16_k(c, GamutOptions{});
    t->cmax_rows = cam16::kRows;
    t->cmax_cols = cam16::kCols;
    t->cmax_f64 = cam16::cmax_table(c, cs.xyz_to_rgb);
    t->cmax_f32.resize(t->cmax_f64.size());
    for (size_t i = 0; i < t->cmax_f64.size(); ++i) t->cmax_f32[i] = num::f32(t->cmax_f64[i]);
    return t;
}

// ---- GamutOptions ------------------------------------------------------------------------------------

GamutOptions GamutOptions::parse(std::string_view text) {
    GamutOptions g;
    if (text.empty()) return g;
    json::Value v;
    try {
        v = json::parse(text);
    } catch (const json::ParseError& e) {
        fail(SPK_ERR_INVALID_ARG, std::string("gamut_compress_json: ") + e.what());
    }
    if (!v.is_object()) fail(SPK_ERR_INVALID_ARG, "gamut_compress_json must be a JSON object");
    if (const json::Value* a = v.find("algorithm")) {
        if (!a->is_string() || (a->as_string() != "cam16ucs" && a->as_string() != "off"))
            fail_user("unknown gamut compression algorithm" +
                      (a->is_string() ? " '" + a->as_string() + "'" : std::string()) + " (expected cam16ucs or off)");
        g.algorithm = a->as_string();
    }
    auto triple = [&](const char* key, std::array<double, 3>& out) {
        const json::Value* a = v.find(key);
        if (!a || !a->is_array() || a->as_array().size() != 3) return;  // not length 3: ignored
        std::array<double, 3> t{};
        for (size_t i = 0; i < 3; ++i) {
            if (!a->as_array()[i].is_number()) return;
            t[i] = a->as_array()[i].as_number();
        }
        out = t;
    };
    triple("knee", g.knee);
    triple("lightness_compression", g.lightness_compression);
    if (const json::Value* b = v.find("lightness_compression_active"); b && b->is_bool())
        g.lightness_compression_active = b->as_bool();
    return g;
}

// ---- Cam16Cache --------------------------------------------------------------------------------------

struct Cam16Cache::Impl {
    std::mutex m;
    std::map<std::string, std::unique_ptr<Cam16Tables>, std::less<>> by_space;
};
std::shared_ptr<Cam16Cache::Impl> Cam16Cache::make_impl() { return std::make_shared<Impl>(); }

const Cam16Tables& Cam16Cache::get(const res::Resources& res, std::string_view output_space) {
    std::lock_guard<std::mutex> g(p_->m);
    auto it = p_->by_space.find(output_space);
    if (it != p_->by_space.end()) return *it->second;
    std::unique_ptr<Cam16Tables> t = build_cam16_tables(res, output_space);
    const Cam16Tables& ref = *t;
    p_->by_space.emplace(std::string(output_space), std::move(t));
    return ref;
}

size_t Cam16Cache::size() const {
    std::lock_guard<std::mutex> g(p_->m);
    return p_->by_space.size();
}

// ---- spk_output_transform ----------------------------------------------------------------------------

/* output_transform(res, cache, src, dst, gamut)
 *   m := XYZ->RGB(dst); if whites differ: m := m . von_kries(white src -> white dst, CAT02)
 *   m := m . RGB->XYZ(src)
 *   t := cache tables of dst
 *   reply: {source_color_space, target_color_space, source_cctf_mode, target_cctf_mode, matrix,
 *           gamut_compress: {algorithm, m_to_xyz, m_to_rgb, k, cmax_rows, cmax_cols,
 *                            lightness_compression_active}}
 *   tables are returned with algorithm "off" too
 */
OutputTransformResult output_transform(const res::Resources& res, Cam16Cache& cache, std::string_view src,
                                       std::string_view dst, const GamutOptions& gamut) {
    const res::ColourSpace s = resolve_space(res, src);
    const res::ColourSpace d = resolve_space(res, dst);
    auto cat02 = res::cat_matrix(*res.blob, "CAT02");
    if (!cat02) fail_user("the constants blob has no colour/cat/CAT02");
    Mat3 m = d.xyz_to_rgb;
    if (auto a = res::von_kries(s.whitepoint, d.whitepoint, *cat02, res::GuardRef::Target)) m = res::mat3_mul(m, *a);
    m = res::mat3_mul(m, s.rgb_to_xyz);

    const Cam16Tables& t = cache.get(res, d.name);
    const cam16::Constants c = cam16::make_constants(d.whitepoint);

    json::Value gc;
    gc["algorithm"] = gamut.algorithm;
    gc["m_to_xyz"] = mat_json(t.m_to_xyz);
    gc["m_to_rgb"] = mat_json(t.m_to_rgb);
    json::Array k;
    for (double v : cam16_k(c, gamut)) k.emplace_back(v);
    gc["k"] = json::Value(std::move(k));
    gc["cmax_rows"] = t.cmax_rows;
    gc["cmax_cols"] = t.cmax_cols;
    gc["lightness_compression_active"] = gamut.lightness_compression_active;

    OutputTransformResult r;
    r.reply["source_color_space"] = s.name;
    r.reply["target_color_space"] = d.name;
    r.reply["source_cctf_mode"] = int(s.cctf);
    r.reply["target_cctf_mode"] = int(d.cctf);
    r.reply["matrix"] = mat_json(m);
    r.reply["gamut_compress"] = std::move(gc);
    r.cmax = t.cmax_f32.data();
    r.cmax_count = uint32_t(t.cmax_f32.size());
    return r;
}

}  // namespace spk::print
