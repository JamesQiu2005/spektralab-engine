// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 Guangzhou Songzi Intelligent Technology Services Co., Ltd.
//
// The film module: set-up maths (coupler presets and matrix, curve tables and their
// pre-inversion, the spectral table and the input matrix), the kernels against CPU references
// (decode, geometry, upsampling, boost, curves), the meter, node counts and the develop chain.
// Usage: test_film <res>   (a resources folder holding the profiles, the constants and the metallib)
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <random>

#include "check.hpp"
#include "film/film_internal.hpp"
#include "film_shared.h"
#include "params/params.hpp"
#include "res/resources.hpp"
#include "util/numeric.hpp"

using namespace spk;
using params::P;

namespace {

std::shared_ptr<gpu::Context> g_gpu;
std::shared_ptr<const res::Resources> g_res;
std::shared_ptr<const params::Schema> g_schema;

params::ParamSet defaults() { return params::ParamSet(g_schema); }

core::RenderContext make_ctx() { return core::RenderContext{.gpu = *g_gpu}; }

gpu::Plane plane_from(const std::vector<float>& v, uint32_t w, uint32_t h) {
    gpu::Plane p = g_gpu->pool().plane(w, h);
    std::memcpy(p.data(), v.data(), p.bytes());
    return p;
}

float rel(float a, float b) {
    if (a == b) return 0.0f;
    return std::fabs(a - b) / std::max(std::fabs(a), std::fabs(b));
}

// ---- set-up maths ------------------------------------------------------------------------------
void test_couplers() {
    res::ProfileInfo neg;
    neg.stock = "kodak_portra_400";
    const film::Mat3 g = film::coupler_gammas(neg);
    // Column sums 0.658 / 0.898 / 0.928.
    CHECK(std::fabs(g[0] + g[3] + g[6] - 0.658) < 1e-12);
    CHECK(std::fabs(g[1] + g[4] + g[7] - 0.898) < 1e-12);
    CHECK(std::fabs(g[2] + g[5] + g[8] - 0.928) < 1e-12);
    CHECK(g[0] == 0.336 && g[1] == 0.353 && g[2] == 0.302 && g[3] == 0.154 && g[5] == 0.353 && g[7] == 0.226);
    res::ProfileInfo pos;
    pos.type = "positive";
    pos.stock = "kodak_ektachrome_100";
    const film::Mat3 gp = film::coupler_gammas(pos);
    CHECK(gp[0] == 0.12 && gp[4] == 0.08 && gp[8] == 0.06 && gp[1] == 0.12 && gp[2] == 0.06 && gp[3] == 0.08);
    pos.stock = "fujifilm_velvia_100";
    CHECK(film::coupler_gammas(pos)[0] == 0.108 && film::coupler_gammas(pos)[8] == 0.054);
    pos.stock = "fujifilm_provia_100f";
    CHECK(film::coupler_gammas(pos)[0] == 0.156 && film::coupler_gammas(pos)[3] == 0.104);
    const film::Mat3 m = film::coupler_matrix(neg, 2.0);
    CHECK(m[1] == 2.0 * 0.353 && m[8] == 2.0 * 0.273);
}

void test_curve_tables() {
    auto film = g_res->profile("kodak_portra_400");
    const size_t K = film->curve_points();
    res::ProfileInfo info = film->info;
    // gamma 2: x = f32(l / 2), inv = f32(1 / (l[k+1]/2 - l[k]/2)).
    film::CurveTables t2 = film::build_curve_tables(*film, 2.0, film::coupler_matrix(info, 1.0), false);
    CHECK(t2.K == K);
    CHECK(t2.x[10] == num::f32(film->log_exposure[10] / 2.0));
    CHECK(t2.inv[10] == num::f32(1.0 / (film->log_exposure[11] / 2.0 - film->log_exposure[10] / 2.0)));
    CHECK(t2.y[K + 7] == num::f32(film->normalized_curves(7, 1)));
    // amount 0: N0 = N exactly.
    film::CurveTables t0 = film::build_curve_tables(*film, 1.0, film::coupler_matrix(info, 0.0), false);
    bool same = true;
    for (size_t i = 0; i < 3 * K; ++i) same = same && t0.y0[i] == t0.y[i];
    CHECK(same);
    // gamma 1, amount 1: on a uniform field the couplers reproduce the measured curve.
    const film::Mat3 m = film::coupler_matrix(info, 1.0);
    film::CurveTables t = film::build_curve_tables(*film, 1.0, m, false);
    float worst = 0.0f;
    for (int i = 0; i <= 600; ++i) {
        const float L = -2.5f + 6.0f * float(i) / 600.0f;
        float d1[3], chi[3];
        for (int c = 0; c < 3; ++c) d1[c] = film::cpu::curve_lookup(t.x.data(), t.inv.data(), t.y.data() + c * K, t.K, L);
        for (int mm = 0; mm < 3; ++mm)
            chi[mm] = (d1[0] * num::f32(m[mm]) + d1[1] * num::f32(m[3 + mm])) + d1[2] * num::f32(m[6 + mm]);
        for (int c = 0; c < 3; ++c) {
            const float d = film::cpu::curve_lookup(t.x.data(), t.inv.data(), t.y0.data() + c * K, t.K, L - chi[c]);
            worst = std::max(worst, std::fabs(d - d1[c]));
        }
    }
    std::printf("couplers uniform-field round trip (gamma 1): max |D - D1| = %.3g\n", worst);
    CHECK(worst < 5e-3f);
    // Lookup ends clamp to the plateaus.
    CHECK(film::cpu::curve_lookup(t.x.data(), t.inv.data(), t.y.data(), t.K, -50.0f) == t.y[0]);
    CHECK(film::cpu::curve_lookup(t.x.data(), t.inv.data(), t.y.data(), t.K, 50.0f) == t.y[K - 1]);
}

std::shared_ptr<const film::FilmSetup> setup_for(const std::string& stock, params::ParamSet p) {
    p.set(P::film_stock, json::Value(stock));
    film::FilmSetupInputs in{p, g_res->profile(stock), *g_res, *g_gpu};
    return film::build_film_setup(in);
}

params::ParamSet plain_params() {
    params::ParamSet p = defaults();
    p.set(P::grain_active, json::Value(false));
    p.set(P::halation_active, json::Value(false));
    p.set(P::auto_exposure, json::Value(false));
    return p;
}

void test_spectral_table() {
    auto s = setup_for("kodak_portra_400", plain_params());
    const film::SpectralTable& t = s->spectral_table();
    CHECK(t.size == 192 && t.f64.size() == 192u * 192u * 3u);
    bool positive = true;
    for (double v : t.f64) positive = positive && v > 0.0 && std::isfinite(v);
    CHECK(positive);
    // Each set-up computes its own table; the same inputs give the same values.
    auto s2 = setup_for("kodak_portra_400", plain_params());
    CHECK(&s2->spectral_table() != &s->spectral_table() && s2->spectral_table().f64 == t.f64);

    // An 18.4 % ProPhoto neutral lands near log10 H = 0.
    struct Ref {
        const char* stock;
        double e[3];
    } refs[] = {{"kodak_portra_400", {0.006, -0.005, -0.037}}, {"fujifilm_velvia_100", {0.010, -0.001, -0.006}}};
    for (const Ref& r : refs) {
        auto f = setup_for(r.stock, plain_params());
        const float rgb[3] = {0.184f, 0.184f, 0.184f};
        const float* tab = f->spectral_table().f32.as<float>();
        std::array<float, 3> H = film::cpu::upsample(f->impl().input_matrix_f32, tab, 192, rgb);
        std::printf("%s: mid grey log10 H = (%+.4f, %+.4f, %+.4f), bilinear estimate (%+.3f, %+.3f, %+.3f)\n",
                    r.stock, std::log10(H[0]), std::log10(H[1]), std::log10(H[2]), r.e[0], r.e[1], r.e[2]);
        for (int c = 0; c < 3; ++c) CHECK(std::fabs(std::log10(H[c]) - r.e[c]) < 0.02);
    }
    // ProPhoto to D55 always adapts: A differs from the stored matrix.
    CHECK(s->impl().input_matrix[0] != s->input_space().rgb_to_xyz[0]);
}

// ---- kernels against CPU references --------------------------------------------------------------
void test_upsample_kernel() {
    auto s = setup_for("kodak_portra_400", plain_params());
    const uint32_t n = 4096;
    std::vector<float> rgb(3 * n);
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> u(-0.2f, 4.0f);
    for (float& v : rgb) v = u(rng);
    const float special[][3] = {{0, 0, 0}, {1e-6f, 1e-6f, 1e-6f}, {-0.1f, -0.1f, -0.1f}, {1, 0, 0}, {0, 1, 0},
                                {0, 0, 1}, {1024, 1024, 1024}, {-1, 2, 0.5f}, {NAN, 0, 0}};
    for (size_t i = 0; i < std::size(special); ++i)
        for (int c = 0; c < 3; ++c) rgb[3 * i + c] = special[i][c];
    gpu::Plane in = plane_from(rgb, n, 1), out = g_gpu->pool().plane(n, 1);
    SpkFilmUpsampleArgs a{};
    a.n = n;
    a.L = 192;
    for (int i = 0; i < 9; ++i) a.a[i] = s->impl().input_matrix_f32[i];
    gpu::Batch b = g_gpu->batch();
    b.dispatch_1d(g_gpu->kernel("spk_film_upsample"), n,
                  {gpu::Arg::plane(in), gpu::Arg::plane(out), gpu::Arg::buffer(s->spectral_table().f32), gpu::Arg::value(a)});
    b.commit_and_wait();
    float worst = 0.0f;
    size_t exact = 0;
    for (uint32_t i = 0; i < n; ++i) {
        std::array<float, 3> h = film::cpu::upsample(s->impl().input_matrix_f32, s->spectral_table().f32.as<float>(), 192,
                                                     &rgb[3 * i]);
        for (int c = 0; c < 3; ++c) {
            const float g = out.data()[3 * i + c];
            worst = std::max(worst, std::fabs(g - h[c]) <= 1e-30f ? 0.0f : rel(g, h[c]));
            exact += g == h[c];
        }
    }
    std::printf("upsample GPU vs CPU: max rel %.3g, %zu of %u exact\n", worst, exact, 3 * n);
    CHECK(worst < 2e-5f);
    // Black and negative-sum input: H <= 0; the NaN pixel gives H = 0.
    CHECK(out.data()[0] == 0.0f && out.data()[6] <= 0.0f && out.data()[3 * 8] == 0.0f);
}

void test_decode_kernel() {
    core::RenderContext ctx = make_ctx();
    std::vector<float> v;
    for (int i = -20; i <= 300; ++i) v.push_back(float(i) / 200.0f);
    v.push_back(0.040449936f);
    v.push_back(std::nextafter(0.040449936f, 1.0f));
    while (v.size() % 3) v.push_back(0.5f);
    const uint32_t w = uint32_t(v.size() / 3);
    gpu::Plane in = plane_from(v, w, 1);
    const res::CctfMode modes[] = {res::CctfMode::SRGB, res::CctfMode::ProPhoto, res::CctfMode::Adobe,
                                   res::CctfMode::BT709, res::CctfMode::Identity};
    for (res::CctfMode m : modes) {
        gpu::Plane out = film::run_decode(ctx, in, m);
        float worst = 0.0f;
        bool nan_ok = true;
        for (size_t i = 0; i < v.size(); ++i) {
            const double x = v[i];
            double r = x;
            switch (m) {
                case res::CctfMode::SRGB: r = x <= double(0.040449936f) ? double(float(x) / 12.92f) : std::pow((x + 0.055) / 1.055, 2.4); break;
                case res::CctfMode::ProPhoto: r = x < 16.0 / 512.0 ? double(float(x) / 16.0f) : std::pow(x, 1.8); break;
                case res::CctfMode::Adobe: r = std::pow(x, 563.0 / 256.0); break;
                case res::CctfMode::BT709: {
                    const double bp = 1.099 * std::pow(0.018, 0.45) - 0.099;
                    r = x < bp ? double(float(x) / 4.5f) : std::pow((x + 0.099) / 1.099, 1.0 / 0.45);
                    break;
                }
                default: break;
            }
            const float g = out.data()[i];
            if (std::isnan(r)) {
                nan_ok = nan_ok && std::isnan(g);
                continue;
            }
            worst = std::max(worst, rel(g, float(r)));
        }
        std::printf("decode mode %d: max rel vs binary64 %.3g\n", int(m), worst);
        CHECK(worst < 1e-6f);
        CHECK(nan_ok);
    }
}

std::vector<float> pattern(uint32_t w, uint32_t h) {
    std::vector<float> v(size_t(w) * h * 3);
    for (uint32_t y = 0; y < h; ++y)
        for (uint32_t x = 0; x < w; ++x)
            for (int c = 0; c < 3; ++c)
                v[3 * (size_t(y) * w + x) + c] = float(x) * 0.01f + float(y) * 0.37f + float(c) * 100.0f +
                                                  std::sin(float(x * 7 + y * 3 + c)) * 0.5f;
    return v;
}

void test_geometry() {
    core::RenderContext ctx = make_ctx();
    params::ParamSet p = defaults();
    CHECK(film::plan_geometry(p, 1600, 1067).identity);
    film::Size sz = film::geometry_output_size(p, 1600, 1067);
    CHECK(sz.w == 1600 && sz.h == 1067);
    p.set(P::geometry_crop_w, json::Value(0.5));
    p.set(P::geometry_crop_h, json::Value(0.6));
    sz = film::geometry_output_size(p, 1600, 1067);
    CHECK(sz.w == 800 && sz.h == 640);  // round = 640
    p.set(P::geometry_quarter_turns, json::Value(3));
    sz = film::geometry_output_size(p, 1600, 1067);
    CHECK(sz.w == 640 && sz.h == 800);

    const uint32_t W = 37, H = 23;
    const std::vector<float> src = pattern(W, H);
    gpu::Plane in = plane_from(src, W, H);

    // Quarter turns and flips are exact copies.
    for (int q = 0; q < 4; ++q)
        for (int f = 0; f < 4; ++f) {
            params::ParamSet g = defaults();
            g.set(P::geometry_quarter_turns, json::Value(q));
            g.set(P::geometry_flip_h, json::Value(bool(f & 1)));
            g.set(P::geometry_flip_v, json::Value(bool(f & 2)));
            film::GeometryPlan plan = film::plan_geometry(g, W, H);
            if (plan.identity) continue;
            gpu::Plane out = film::run_geometry(ctx, in, plan);
            const uint32_t OW = out.w, OH = out.h;
            bool ok = (q % 2 ? (OW == H && OH == W) : (OW == W && OH == H));
            for (uint32_t oy = 0; ok && oy < OH; ++oy)
                for (uint32_t ox = 0; ok && ox < OW; ++ox) {
                    // forward: flip after the clockwise turn, so undo the flip first
                    uint32_t x = (f & 1) ? OW - 1 - ox : ox, y = (f & 2) ? OH - 1 - oy : oy;
                    uint32_t sx = 0, sy = 0;
                    switch (q) {
                        case 0: sx = x; sy = y; break;
                        case 1: sx = y; sy = H - 1 - x; break;
                        case 2: sx = W - 1 - x; sy = H - 1 - y; break;
                        default: sx = W - 1 - y; sy = x; break;
                    }
                    for (int c = 0; c < 3; ++c)
                        ok = ok && out.data()[3 * (size_t(oy) * OW + ox) + c] == src[3 * (size_t(sy) * W + sx) + c];
                }
            CHECK(ok);
        }

    // Rotation, crop and the edge clamp against the binary64 reference.
    struct G {
        double x, y, w, h, deg;
        int q;
        bool fh, fv;
    } cases[] = {{0, 0, 1, 1, 7.5, 0, false, false},     {0.1, 0.2, 0.5, 0.6, 0, 0, false, false},
                 {0, 0, 1, 1, -30, 0, false, false},     {0.05, 0, 0.9, 1, -3, 2, true, false},
                 {0.3, 0.1, 0.8, 0.95, 45, 1, true, true}, {0.9, 0.9, 0.5, 0.5, 12, 3, false, true}};
    for (const G& c : cases) {
        params::ParamSet g = defaults();
        g.set(P::geometry_crop_x, json::Value(c.x));
        g.set(P::geometry_crop_y, json::Value(c.y));
        g.set(P::geometry_crop_w, json::Value(c.w));
        g.set(P::geometry_crop_h, json::Value(c.h));
        g.set(P::geometry_rotation_deg, json::Value(c.deg));
        g.set(P::geometry_quarter_turns, json::Value(c.q));
        g.set(P::geometry_flip_h, json::Value(c.fh));
        g.set(P::geometry_flip_v, json::Value(c.fv));
        film::GeometryPlan plan = film::plan_geometry(g, W, H);
        gpu::Plane out = film::run_geometry(ctx, in, plan);
        std::vector<float> ref(size_t(plan.dst_w) * plan.dst_h * 3);
        film::geometry_cpu(plan, src.data(), ref.data());
        float worst = 0.0f;
        size_t diff = 0;
        for (size_t i = 0; i < ref.size(); ++i) {
            worst = std::max(worst, std::fabs(out.data()[i] - ref[i]));
            diff += out.data()[i] != ref[i];
        }
        std::printf("geometry (%.2f %.2f %.2f %.2f, %g deg, q%d): %ux%u, %zu of %zu differ, max |d| %.3g\n", c.x, c.y,
                    c.w, c.h, c.deg, c.q, plan.dst_w, plan.dst_h, diff, ref.size(), worst);
        CHECK(worst <= 1e-4f);  // values up to ~210: about one binary32 ulp
    }
}

void test_boost() {
    core::RenderContext ctx = make_ctx();
    auto s = [] {
        params::ParamSet p = plain_params();
        p.set(P::halation_boost_ev, json::Value(3.0));
        p.set(P::dir_couplers_active, json::Value(false));
        return setup_for("kodak_portra_400", p);
    }();
    // The boost alone on a plane with a known maximum.
    std::vector<float> v;
    for (int i = 0; i < 300; ++i) v.push_back(0.01f * float(i));
    v.push_back(10.0f);
    while (v.size() % 3) v.push_back(1.0f);
    const uint32_t w = uint32_t(v.size() / 3);
    gpu::Plane pl = plane_from(v, w, 1);
    film::run_boost(ctx, pl, s->params().f(P::halation_boost_ev), film::plane_max(pl));
    float mx = 0.0f;
    bool low_same = true;
    for (size_t i = 0; i < v.size(); ++i) {
        mx = std::max(mx, pl.data()[i]);
        if (v[i] <= 2.944f) low_same = low_same && pl.data()[i] == v[i];
    }
    std::printf("boost 3 EV: max %.7g (want %.7g)\n", mx, 80.0);
    CHECK(std::fabs(mx - 80.0f) < 80.0f * 1e-5f);
    CHECK(low_same);
}

// ---- meter -------------------------------------------------------------------------------------
void test_meter() {
    auto cs = res::colour_space(*g_res->blob, "ProPhoto RGB");
    const uint32_t w = 64, h = 40;
    std::vector<float> grey(size_t(w) * h * 3, 0.184f);
    film::MeterValues m = film::meter_values(grey.data(), w, h, cs->rgb_to_xyz);
    const double rowsum = (cs->rgb_to_xyz[3] + cs->rgb_to_xyz[4]) + cs->rgb_to_xyz[5];
    const double ev0 = -std::log2(rowsum);
    std::printf("meter on 0.184 grey: cw %.3g avg %.3g med %.3g bal %.3g ctr %.3g ph %.3g ps %.3g\n", m.center_weighted,
                m.average, m.median, m.balanced, m.center, m.protect_highlights, m.protect_shadows);
    CHECK(std::fabs(m.average - ev0) < 1e-6 && std::fabs(m.median - ev0) < 1e-6 && std::fabs(m.center_weighted - ev0) < 1e-6);
    CHECK(std::fabs(m.balanced - ev0) < 1e-6 && std::fabs(m.center - ev0) < 1e-6);
    CHECK(m.protect_highlights == m.balanced && m.protect_shadows == m.balanced);
    // All black: legacy meters give 0, intents the floor's EV.
    std::vector<float> black(size_t(w) * h * 3, 0.0f);
    m = film::meter_values(black.data(), w, h, cs->rgb_to_xyz);
    CHECK(m.average == 0.0 && m.median == 0.0 && m.center_weighted == 0.0);
    CHECK(std::fabs(m.balanced - 12.0) < 1e-9);
    // Median of an even count is the mean of the two middle values; a doubling moves average by 1 EV.
    std::vector<float> two = {0.1f, 0.1f, 0.1f, 0.3f, 0.3f, 0.3f};
    m = film::meter_values(two.data(), 2, 1, cs->rgb_to_xyz);
    CHECK(std::fabs(m.median - (-std::log2((0.2 * rowsum) / 0.184))) < 1e-6);
    std::vector<float> dbl = grey;
    for (float& v : dbl) v *= 2.0f;
    CHECK(std::fabs(film::meter_values(dbl.data(), w, h, cs->rgb_to_xyz).average - (ev0 - 1.0)) < 1e-6);
}

// ---- develop chain (pointwise configuration) -----------------------------------------------------
void test_develop() {
    core::RenderContext ctx = make_ctx();
    params::ParamSet p = plain_params();
    p.set(P::dir_couplers_active, json::Value(false));
    p.set(P::exposure_compensation_ev, json::Value(1.0));
    auto s = setup_for("kodak_portra_400", p);
    // Node count: upsample, EC, log, curves only.
    CHECK(film::node_count(*s) == 4);

    const uint32_t w = 24, h = 3;
    std::vector<float> rgb(size_t(w) * h * 3);
    for (uint32_t i = 0; i < w * h; ++i)
        for (int c = 0; c < 3; ++c) rgb[3 * i + c] = 0.184f * std::pow(2.0f, float(int(i % w) - 12)) * (1.0f + 0.1f * c);
    gpu::Plane src = plane_from(rgb, w, h);
    ctx.pitch_um = film::pitch_um(35.0, w);
    gpu::Plane negative = film::develop(ctx, *s, {src, std::nullopt, 0x5EED});
    CHECK(negative.w == w && negative.h == h);
    CHECK(std::memcmp(src.data(), rgb.data(), src.bytes()) == 0);  // the tier source is untouched
    const film::CurveTables& t = s->impl().curves;
    float worst = 0.0f;
    for (uint32_t i = 0; i < w * h; ++i) {
        std::array<float, 3> H = film::cpu::upsample(s->impl().input_matrix_f32, s->spectral_table().f32.as<float>(), 192,
                                                     &rgb[3 * i]);
        for (int c = 0; c < 3; ++c) {
            const float e = H[c] * 2.0f;
            const float L = std::log10(std::fmax(e, 0.0f) + 1e-10f);
            const float d = film::cpu::curve_lookup(t.x.data(), t.inv.data(), t.y.data() + c * t.K, t.K, L);
            worst = std::max(worst, std::fabs(negative.data()[3 * i + c] - d));
        }
    }
    std::printf("develop (pointwise) vs CPU chain: max |dD| %.3g\n", worst);
    CHECK(worst < 1e-4f);

    // With couplers.
    {
        auto sc = setup_for("kodak_portra_400", plain_params());
        CHECK(film::node_count(*sc) == 5);
        gpu::Plane rc = film::develop(ctx, *sc, {src, std::nullopt, 0x5EED});
        CHECK(rc.w == w);
    }

    // A positive film scanned directly with a level correction is refused at the log step.
    {
        params::ParamSet pp = plain_params();
        pp.set(P::dir_couplers_active, json::Value(false));
        pp.set(P::scan_film, json::Value(true));
        pp.set(P::scanner_white_correction, json::Value(true));
        auto sp = setup_for("fujifilm_velvia_100", pp);
        int32_t status = 0;
        std::string msg;
        try {
            film::develop(ctx, *sp, {src, std::nullopt, 0x5EED});
        } catch (const Error& e) {
            status = e.status();
            msg = e.what();
        }
        CHECK(status == SPK_ERR_GPU);
        CHECK_EQ_STR(msg, "black/white correction on a directly scanned positive film is not implemented by the native engine");
        pp.set(P::scanner_white_correction, json::Value(false));
        auto sa = setup_for("fujifilm_velvia_100", pp);
        film::develop(ctx, *sa, {src, std::nullopt, 0x5EED});  // accepted without the corrections
    }

    // Unknown input colour space refuses the build.
    {
        params::ParamSet bad = plain_params();
        bad.set(P::input_color_space, json::Value("Not A Space"));
        int32_t status = 0;
        try {
            setup_for("kodak_portra_400", bad);
        } catch (const Error& e) {
            status = e.status();
            CHECK(std::string(e.what()).find("input_color_space") != std::string::npos);
        }
        CHECK(status == SPK_ERR_USER);
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) return 2;
    try {
        g_res = res::Resources::load(argv[1]);
        g_gpu = gpu::Context::create(nullptr, std::filesystem::path(argv[1]) / "spektralab.metallib");
        g_schema = params::Schema::build();
        test_couplers();
        test_curve_tables();
        test_spectral_table();
        test_upsample_kernel();
        test_decode_kernel();
        test_geometry();
        test_boost();
        test_meter();
        test_develop();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: uncaught exception: %s\n", e.what());
        ++g_failures;
    }
    return finish("test_film");
}
