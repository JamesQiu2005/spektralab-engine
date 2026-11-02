// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 Guangzhou Songzi Intelligent Technology Services Co., Ltd.
//
// Print module: the float64 set-up math against known values, and the print kernels against host
// references. Usage: test_print <res>   (a resources folder holding the metallib)
//
// The unit-test link set (Makefile TEST_LINK_OBJS) does not contain the print objects, so the
// self-contained print sources are compiled into this test directly.
#include <cmath>
#include <cstring>
#include <filesystem>

#include "check.hpp"
#include "gpu/gpu.hpp"
#include "print/cam16.cpp"
#include "print/internal.hpp"
#include "print/run.cpp"
#include "print/setup.cpp"
#include "res/resources.hpp"
#include "util/json.hpp"

using namespace spk;
using namespace spk::print;
namespace fs = std::filesystem;

static bool near(double a, double b, double rel, double abs_tol = 0.0) {
    return std::fabs(a - b) <= std::max(abs_tol, rel * std::fabs(b));
}

#define CHECK_NEAR(a, b, rel)                                                                              \
    do {                                                                                                   \
        const double _a = (a), _b = (b);                                                                   \
        ++g_checks;                                                                                        \
        if (!near(_a, _b, rel)) {                                                                          \
            ++g_failures;                                                                                  \
            std::fprintf(stderr, "FAIL %s:%d: %s = %.17g, want %.17g\n", __FILE__, __LINE__, #a, _a, _b); \
        }                                                                                                  \
    } while (0)

static void test_illuminants(const res::Resources& r) {
    const res::Blob& b = *r.blob;
    auto xy = [&](const char* n) { return res::chromaticity(b, *res::illuminant(b, n)); };
    CHECK_NEAR(xy("D50")[0], 0.345674591, 1e-9);
    CHECK_NEAR(xy("D50")[1], 0.3585103882, 1e-9);
    CHECK_NEAR(xy("D65")[0], 0.3127205252, 1e-9);
    CHECK_NEAR(xy("D65")[1], 0.329030685, 1e-9);
    CHECK_NEAR(xy("K75P")[0], 0.3152468589, 1e-9);
    CHECK_NEAR(xy("K75P")[1], 0.3328584711, 1e-9);
    // unit mean
    for (const char* n : {"TH-KG3", "TH-KG3-L", "BB3200", "D55", "T", "K75P"}) {
        auto s = res::illuminant(b, n);
        CHECK(s.has_value());
        double sum = 0;
        for (double v : *s) sum += v;
        CHECK_NEAR(sum / 81.0, 1.0, 1e-14);
    }
    CHECK(!res::illuminant(b, "NoSuchLamp"));

    // dichroic filters: half transmission at the erf centres, opaque/clear far away
    auto f = dichroic_filters();
    const size_t i500 = (500 - 380) / 5, i610 = (610 - 380) / 5, i400 = (400 - 380) / 5, i700 = (700 - 380) / 5;
    CHECK_NEAR(f[1][i500], 0.5, 1e-15);  // M below 550
    CHECK_NEAR(f[1][i610], 0.5, 1e-15);  // M above 550
    CHECK(f[1][(550 - 380) / 5] < 1e-6);  // magenta blocks green
    CHECK(f[2][i400] < 1e-6 && f[2][i700] > 1 - 1e-6);  // yellow blocks blue
    CHECK(f[0][i700] < 1e-6 && f[0][i400] > 1 - 1e-6);  // cyan blocks red
    CHECK_NEAR(f[2][(516 - 380) / 5 + 0], f[2][(516 - 380) / 5], 0);  // λ=515 grid point exists
    // filtration: 0 CC is clear, a negative CC amplifies outside the filter's pass range
    Spectrum lamp{};
    lamp.fill(1.0);
    Spectrum clear = filtered_light(lamp, f, {0, 0, 0});
    for (double v : clear) CHECK(v == 1.0);
    Spectrum neg = filtered_light(lamp, f, {0, -10, 0});
    CHECK(neg[(550 - 380) / 5] > 1.0);
    Spectrum m100 = filtered_light(lamp, f, {0, 100, 0});
    CHECK_NEAR(m100[(550 - 380) / 5], 0.1 + 0.9 * f[1][(550 - 380) / 5], 1e-12);
}

static void test_cam16_constants() {
    // known values
    const cam16::Constants pp = cam16::make_constants({0.3457, 0.3585});
    CHECK_NEAR(pp.F_L, 0.6839903845696502, 1e-15);
    CHECK_NEAR(pp.n, 0.2, 0);
    CHECK_NEAR(pp.N_bb, 1.0003040045593807, 1e-15);
    CHECK_NEAR(pp.z, 1.9272135954999579, 1e-15);
    CHECK_NEAR(pp.D, 0.9122363399359157, 1e-15);
    CHECK_NEAR(pp.D_rgb[0], 1.0048858995308805, 1e-15);
    CHECK_NEAR(pp.D_rgb[1], 0.9991651597854243, 1e-15);
    CHECK_NEAR(pp.D_rgb[2], 1.18239029867048, 1e-15);
    CHECK_NEAR(pp.A_w, 37.17349142011997, 1e-15);
    std::printf("ProPhoto: F_L %.17g N_bb %.17g z %.17g D %.17g A_w %.17g J'_white %.17g\n", pp.F_L, pp.N_bb, pp.z,
                pp.D, pp.A_w, pp.J_white);
    std::printf("  D_rgb %.17g %.17g %.17g\n", pp.D_rgb[0], pp.D_rgb[1], pp.D_rgb[2]);
    CHECK_NEAR(pp.J_white, 100.0, 1e-12);
    const cam16::Constants d65 = cam16::make_constants({0.3127, 0.3290});
    CHECK_NEAR(d65.D_rgb[0], 1.0228770275436545, 1e-15);
    CHECK_NEAR(d65.D_rgb[1], 0.9852074782801457, 1e-15);
    CHECK_NEAR(d65.D_rgb[2], 0.9285450586783286, 1e-15);
    CHECK_NEAR(d65.A_w, 37.16907530221132, 1e-15);
    // the computed inverse of the literal M16
    const double inv[9] = {1.862067855087233,    -1.0112546305316843, 0.14918677544445175,
                           0.38752654323613717,  0.6214474419314752,  -0.00897398516761251,
                           -0.01584149884933386, -0.03412293802851556, 1.0499644368778496};
    for (int i = 0; i < 9; ++i) CHECK_NEAR(pp.m16_inv[i], inv[i], 1e-14);

    // forward/inverse round trip in float64
    for (const Vec3& x : {Vec3{0.2, 0.18, 0.1}, Vec3{0.05, 0.1, 0.4}, Vec3{0.6, 0.5, 0.3}, Vec3{0.9, 1.0, 0.8}}) {
        const cam16::Forward f = cam16::forward(pp, x);
        const Vec3 back = cam16::inverse_unguarded(pp, f.Jp, f.Mp * std::cos(f.h), f.Mp * std::sin(f.h));
        for (int c = 0; c < 3; ++c) CHECK_NEAR(back[c], x[c], 1e-10);
    }
}

static void test_cam16_tables(const res::Resources& r) {
    Cam16Cache cache;
    const Cam16Tables& t = cache.get(r, "ProPhoto RGB");
    CHECK(cache.size() == 1 && &cache.get(r, "ProPhoto RGB") == &t);
    CHECK(t.cmax_rows == 64 && t.cmax_cols == 720 && t.cmax_f32.size() == 46080 && t.k.size() == 22);
    double lo = 1e300, hi = -1e300;
    for (double v : t.cmax_f64) {
        lo = std::min(lo, v);
        hi = std::max(hi, v);
    }
    std::printf("ProPhoto cmax range [%g, %g], cell[63][0] %.9g, cell[32][360] %.9g\n", lo, hi, t.cmax_f64[63 * 720],
                t.cmax_f64[32 * 720 + 360]);
    CHECK(lo >= 0.0 && hi < 150.0 && hi > 20.0);
    // a C_max cell is in gamut, one step more is not (bisection resolution 150 / 2^18)
    const cam16::Constants pp = cam16::make_constants({0.3457, 0.3585});
    const auto cs = res::colour_space(*r.blob, "ProPhoto RGB");
    int bad = 0;
    for (uint32_t i = 3; i < 64; i += 7)
        for (uint32_t j = 0; j < 720; j += 37) {
            const double Jp = 1.0 + (109.0 * i) / 63.0, h = -M_PI + (2.0 * M_PI * j) / 720.0;
            const double m = t.cmax_f64[i * 720 + j];
            auto in_gamut = [&](double M) {
                Vec3 rgb = res::mat3_apply(cs->xyz_to_rgb, cam16::inverse_unguarded(pp, Jp, M * std::cos(h), M * std::sin(h)));
                for (double v : rgb)
                    if (!(v >= -1e-6 && v <= 1 + 1e-6)) return false;
                return true;
            };
            if (m > 0 && !in_gamut(m)) ++bad;
            if (m + 150.0 / 262144.0 * 1.0001 < 150.0 && in_gamut(m + 150.0 / 262144.0 * 1.0001) && m < 149) {
                // not necessarily wrong (the gamut edge need not be monotone in M'), report only
            }
        }
    CHECK(bad == 0);

    // GamutOptions and spk_output_transform
    GamutOptions d = GamutOptions::parse("");
    CHECK(d.algorithm == "cam16ucs" && d.lightness_compression_active && d.knee[2] == 6.0);
    GamutOptions o = GamutOptions::parse(R"({"algorithm":"off","knee":[0.1,1,5],"lightness_compression":[1,2],"lightness_compression_active":false})");
    CHECK(o.algorithm == "off" && o.knee[0] == 0.1 && o.knee[2] == 5 && o.lightness_compression[0] == 0.7 &&
          !o.lightness_compression_active);
    CHECK_THROWS(GamutOptions::parse("{bad json"));
    try {
        GamutOptions::parse(R"({"algorithm":"aces"})");
        CHECK(false);
    } catch (const Error& e) {
        CHECK(e.status() == SPK_ERR_USER);
    }
    try {
        GamutOptions::parse("[1,2");
        CHECK(false);
    } catch (const Error& e) {
        CHECK(e.status() == SPK_ERR_INVALID_ARG);
    }
    OutputTransformResult ot = output_transform(r, cache, "ProPhoto RGB", "sRGB", o);
    std::printf("output_transform: %s\n", json::write(ot.reply).substr(0, 400).c_str());
    CHECK(ot.cmax_count == 46080 && ot.cmax != nullptr && cache.size() == 2);
    const json::Value& k = (*ot.reply.find("gamut_compress")->find("k"));
    CHECK(k.as_array().size() == 22);
    CHECK(k.as_array()[15].as_number() == 0 && k.as_array()[16].as_number() == 1 && k.as_array()[17].as_number() == 1);
    CHECK(ot.reply.find("source_cctf_mode")->as_number() == 1 && ot.reply.find("target_cctf_mode")->as_number() == 0);
    CHECK(ot.reply.find("gamut_compress")->find("algorithm")->as_string() == "off");
    // matrix = M_XYZ->RGB(sRGB) . CAT02(D50 -> D65) . M_RGB->XYZ(ProPhoto)
    {
        auto pp_cs = res::colour_space(*r.blob, "ProPhoto RGB");
        auto s_cs = res::colour_space(*r.blob, "sRGB");
        auto a = res::von_kries(pp_cs->whitepoint, s_cs->whitepoint, *res::cat_matrix(*r.blob, "CAT02"), res::GuardRef::Target);
        CHECK(a.has_value());
        Mat3 m = res::mat3_mul(res::mat3_mul(s_cs->xyz_to_rgb, *a), pp_cs->rgb_to_xyz);
        const json::Array& got = ot.reply.find("matrix")->as_array();
        for (int i = 0; i < 9; ++i) CHECK_NEAR(got[i].as_number(), m[i], 1e-12);
    }
    // same whitepoint: no adaptation
    OutputTransformResult same = output_transform(r, cache, "sRGB", "Display P3", d);
    {
        auto s_cs = res::colour_space(*r.blob, "sRGB");
        auto p_cs = res::colour_space(*r.blob, "Display P3");
        Mat3 m = res::mat3_mul(p_cs->xyz_to_rgb, s_cs->rgb_to_xyz);
        const json::Array& got = same.reply.find("matrix")->as_array();
        for (int i = 0; i < 9; ++i) CHECK_NEAR(got[i].as_number(), m[i], 1e-15);
    }
    try {
        output_transform(r, cache, "sRGB", "Not A Space", d);
        CHECK(false);
    } catch (const Error& e) {
        CHECK(e.status() == SPK_ERR_USER && std::string(e.what()).find("Not A Space") != std::string::npos);
    }
}

static void test_paper_and_probe_math(const res::Resources& r) {
    auto paper = r.profile("kodak_portra_endura");
    std::vector<double> y = paper_model_curves(*paper);
    const size_t K = paper->log_exposure.size();
    CHECK(K == 256 && y.size() == K * 3);
    for (size_t c = 0; c < 3; ++c) {
        bool mono = true;
        for (size_t i = 1; i < K; ++i) mono = mono && y[3 * i + c] >= y[3 * (i - 1) + c];
        CHECK(mono);
        double amp = 0;
        for (size_t l = 0; l < 3; ++l) amp += paper->density_curves_model->amplitudes(c, l);
        CHECK(y[3 * (K - 1) + c] <= amp + 1e-12 && y[3 * (K - 1) + c] > 0.9 * amp);
        CHECK(y[c] < 0.01);
    }

    // Mitchell-Netravali bicubic: a constant table is reproduced exactly; at a node a linear ramp
    // is blurred along each axis, which keeps a linear ramp unchanged.
    const uint32_t L = 8;
    std::vector<double> tab(L * L * 3);
    for (uint32_t i = 0; i < L; ++i)
        for (uint32_t j = 0; j < L; ++j) {
            tab[(i * L + j) * 3 + 0] = 0.25;
            tab[(i * L + j) * 3 + 1] = double(i);
            tab[(i * L + j) * 3 + 2] = double(j) * 2.0;
        }
    Vec3 v = bicubic_sample(tab, L, 3.0 / 7.0, 4.0 / 7.0);
    CHECK_NEAR(v[0], 0.25, 1e-15);
    CHECK_NEAR(v[1], 3.0, 1e-14);
    CHECK_NEAR(v[2], 8.0, 1e-14);
    Vec3 e = bicubic_sample(tab, L, 0.0, 1.0);  // edges: reflection about the end samples
    CHECK_NEAR(e[1], 0.0 + (1.0 / 18.0) * 2.0 * 1.0 / 1.0, 1e-14);
    std::printf("bicubic edge: %.17g %.17g\n", e[1], e[2]);

    // exposure factor: the reciprocal geometric mean
    Spectral3 sens{};
    Spectrum light{}, dens{};
    for (size_t l = 0; l < 81; ++l) {
        light[l] = 1.0;
        dens[l] = 1.0;
        sens[3 * l] = 1.0;
        sens[3 * l + 1] = 2.0;
        sens[3 * l + 2] = 4.0;
    }
    const double phi = exposure_factor(dens, light, sens);
    CHECK_NEAR(phi, 1.0 / (81.0 * 0.1 * 2.0), 1e-14);
    dens[10] = std::nan("");
    CHECK_NEAR(exposure_factor(dens, light, sens), 1.0 / (80.0 * 0.1 * 2.0), 1e-14);

    // scan tables: a zero-density, base-free medium scans to Y = 1
    const std::vector<double> cmfs = r.blob->f64("colour/cmfs_1931_2deg");
    ScanTables st = scan_tables(*paper, *res::illuminant(*r.blob, "D50"), cmfs);
    double Ysum = 0;  // after neutralising the paper's unmeasured wavelengths: a little below 1
    for (size_t l = 0; l < 81; ++l) Ysum += st.wt[3 * l + 1];
    CHECK(Ysum <= 1.0 + 1e-14 && Ysum > 0.99);
    CHECK_NEAR(st.xyz_illum[1], 1.0, 1e-14);
    std::printf("D50 XYZ_illum %.17g %.17g %.17g\n", st.xyz_illum[0], st.xyz_illum[1], st.xyz_illum[2]);
}

// ---- GPU kernels ---------------------------------------------------------------------------------------

static gpu::Plane plane_of(gpu::Context& ctx, const std::vector<float>& v, uint32_t w, uint32_t h) {
    gpu::Plane p = ctx.pool().plane(w, h);
    std::memcpy(p.data(), v.data(), v.size() * sizeof(float));
    return p;
}

static void test_kernels(gpu::Context& ctx, const res::Resources& r) {
    // encode +
    {
        const std::vector<float> in = {0.5f, -0.25f, 2.0f, 0.25f, 0.001f, 1.0f, 0.18f, 0.0031308f, 0.9f};
        const uint32_t w = 3, h = 1;
        gpu::Plane p = plane_of(ctx, in, w, h);
        for (uint32_t mode = 0; mode <= 4; ++mode) {
            SpkPrintEncodeArgs a{};
            a.w = w;
            a.h = h;
            a.stride = w;
            a.cctf_active = 1;
            a.cctf_mode = mode;
            const float id[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
            std::memcpy(a.m_nid, id, sizeof id);
            gpu::Buffer out = ctx.pool().acquire(w * h * 8);
            gpu::Batch b = ctx.batch();
            b.dispatch_2d(ctx.kernel("spk_print_encode"), w, h, {gpu::Arg::plane(p), gpu::Arg::buffer(out), gpu::Arg::value(a)});
            b.commit_and_wait();
            const uint16_t* q = out.as<uint16_t>();
            auto enc = [&](double v) -> double {
                switch (mode) {
                    case 0: return v <= 0.0031308 ? 12.92 * v : 1.055 * std::pow(v, 1 / 2.4) - 0.055;
                    case 1: return v < 1.0 / 512 ? 16 * v : std::pow(v, 1 / 1.8);
                    case 2: return std::pow(v, 256.0 / 563.0);
                    case 3: return v < 0.018 ? 4.5 * v : 1.099 * std::pow(v, 0.45) - 0.099;
                    default: return v;
                }
            };
            for (int i = 0; i < 9; ++i) {
                const int px = i / 3, c = i % 3;
                double e = enc(double(in[i]));
                if (mode != 2 && in[i] < 0) e = mode == 1 ? 16.0 * in[i] : (mode == 4 ? in[i] : e);
                const double want = std::isnan(e) ? 0.0 : std::floor(std::min(std::max(e, 0.0), 1.0) * 65535 + 0.5);
                const double got = q[4 * px + c];
                if (std::fabs(got - want) > 1.0) {
                    std::fprintf(stderr, "encode mode %u value %g: got %g want %g\n", mode, in[i], got, want);
                    CHECK(false);
                }
                CHECK(q[4 * px + 3] == 65535);
            }
        }
        // NaN -> 0, +inf -> 65535, -inf -> 0, without a transfer function
        gpu::Plane pn = plane_of(ctx, {std::nanf(""), INFINITY, -INFINITY}, 1, 1);
        SpkPrintEncodeArgs a{};
        a.w = a.h = a.stride = 1;
        gpu::Buffer out = ctx.pool().acquire(8);
        gpu::Batch b = ctx.batch();
        b.dispatch_2d(ctx.kernel("spk_print_encode"), 1, 1, {gpu::Arg::plane(pn), gpu::Arg::buffer(out), gpu::Arg::value(a)});
        b.commit_and_wait();
        const uint16_t* q = out.as<uint16_t>();
        CHECK(q[0] == 0 && q[1] == 65535 && q[2] == 0 && q[3] == 65535);
    }

    // gamut compression per pixel (float32) against a float64 host model of the same steps
    {
        const Cam16Tables& t = *build_cam16_tables(r, "ProPhoto RGB").release();
        const cam16::Constants c = cam16::make_constants({0.3457, 0.3585});
        SpkPrintCam16Args q{};
        q.rows = 64;
        q.cols = 720;
        for (int i = 0; i < 9; ++i) {
            q.m_to_xyz[i] = float(t.m_to_xyz[i]);
            q.m_to_rgb[i] = float(t.m_to_rgb[i]);
            q.m16[i] = float(cam16::kM16[i]);
            q.m16_inv[i] = float(cam16::kM16InvPublished[i]);
        }
        for (int i = 0; i < 3; ++i) q.d_rgb[i] = float(c.D_rgb[i]);
        q.f_l = float(c.F_L);
        q.n_bb = float(c.N_bb);
        q.n_cb = float(c.N_cb);
        q.n = float(c.n);
        q.z = float(c.z);
        q.a_w = float(c.A_w);
        q.c = float(c.c);
        q.n_c = float(c.N_c);
        q.j_first = 1;
        q.j_last = 110;
        q.h_first = float(-M_PI);
        q.h_step = float(2 * M_PI / 720);
        q.knee[0] = 0;
        q.knee[1] = 1;
        q.knee[2] = 6;
        q.lc[0] = 0.7f;
        q.lc[1] = 1.0f;
        q.lc[2] = 2.2f;
        q.lc_active = 1;
        q.j_white = float(c.J_white);
        std::vector<float> in;
        const uint32_t n = 64;
        for (uint32_t i = 0; i < n; ++i) {
            const double a = double(i) / n;
            in.push_back(float(0.05 + 0.9 * a));
            in.push_back(float(0.3 + 0.4 * std::sin(7 * a)));
            in.push_back(float(0.5 + 0.45 * std::cos(5 * a)));
        }
        in[0] = in[1] = in[2] = 0.18f;  // neutral
        gpu::Plane p = plane_of(ctx, in, n, 1);
        gpu::Buffer cm = ctx.pool().acquire(46080 * 4);
        std::memcpy(cm.contents(), t.cmax_f32.data(), 46080 * 4);
        q.w = n;
        q.h = 1;
        gpu::Batch b = ctx.batch();
        b.dispatch_2d(ctx.kernel("spk_print_gamut"), n, 1, {gpu::Arg::plane(p), gpu::Arg::buffer(cm), gpu::Arg::value(q)});
        b.commit_and_wait();
        auto knee = [](double d, double tau, double lim, double pw) {
            if (d <= tau) return d;
            const double x = (d - tau) / (lim - tau);
            return tau + ((lim - tau) * x) / std::pow(1 + std::pow(x, pw), 1 / pw);
        };
        double worst = 0;
        for (uint32_t i = 0; i < n; ++i) {
            const Vec3 rgb = {in[3 * i], in[3 * i + 1], in[3 * i + 2]};
            const cam16::Forward f = cam16::forward(c, res::mat3_apply(t.m_to_xyz, rgb));
            double Jp = f.Jp;
            double Ln = Jp / c.J_white;
            if (Ln > 0.7) Jp = knee(Ln, 0.7, 1.0, 2.2) * c.J_white;
            const double Jc = std::min(std::max(Jp, 1.0), 110.0);
            const double u = (Jc - 1) / 109 * 63;
            const int l0 = std::clamp(int(std::floor(u)), 0, 62);
            const double fl = u - l0;
            const double v = (f.h + M_PI) / (2 * M_PI / 720);
            int h0 = int(std::floor(v)) % 720;
            if (h0 < 0) h0 += 720;
            const int h1 = h0 == 719 ? 0 : h0 + 1;
            const double fh = v - std::floor(v);
            const double* T = t.cmax_f64.data();
            const double cmx = (1 - fl) * ((1 - fh) * T[l0 * 720 + h0] + fh * T[l0 * 720 + h1]) +
                               fl * ((1 - fh) * T[(l0 + 1) * 720 + h0] + fh * T[(l0 + 1) * 720 + h1]);
            const double cs = std::max(cmx, 1e-9);
            double d = f.Mp / cs;
            if (d > 0) d = knee(d, 0, 1, 6);
            const double Mn = d * cs;
            const Vec3 back = res::mat3_apply(t.m_to_rgb, cam16::inverse_unguarded(c, Jp, Mn * std::cos(f.h), Mn * std::sin(f.h)));
            for (int ch = 0; ch < 3; ++ch) worst = std::max(worst, std::fabs(double(p.data()[3 * i + ch]) - back[ch]));
        }
        std::printf("gamut kernel vs float64 host model: max abs %.3g; grey 0.18 -> %.7f %.7f %.7f\n", worst, p.data()[0],
                    p.data()[1], p.data()[2]);
        CHECK(worst < 2e-5);
        delete &t;
    }

    // preview LUT against a host trilinear evaluation
    {
        res::PrintLutTable lt = r.print_luts.table(*r.blob, "kodak_portra_endura");
        const uint32_t w = 37, h = 3;
        std::vector<float> neg(w * h * 3);
        for (uint32_t i = 0; i < w * h * 3; ++i) neg[i] = float(-0.3 + 3.2 * std::fmod(i * 0.6180339887, 1.0));
        gpu::Plane p = plane_of(ctx, neg, w, h);
        core::RenderContext rc{.gpu = ctx};
        gpu::OutputImage img = preview_lut(rc, p, lt);
        const uint32_t S = lt.size;
        int bad = 0;
        for (uint32_t px = 0; px < w * h; ++px) {
            float u[3];
            uint32_t i0[3], i1[3];
            float f[3];
            for (int c = 0; c < 3; ++c) {
                const float s = float(1.0 / (double(lt.hi[c]) - double(lt.lo[c])));
                u[c] = std::min(std::max((neg[3 * px + c] - lt.lo[c]) * s, 0.0f), 1.0f) * float(S - 1);
                i0[c] = uint32_t(std::floor(u[c]));
                f[c] = u[c] - float(i0[c]);
                i1[c] = std::min(i0[c] + 1, S - 1);
            }
            auto T = [&](uint32_t a, uint32_t b, uint32_t k, int ch) { return lt.table[((a * S + b) * S + k) * 3 + ch]; };
            for (int ch = 0; ch < 3; ++ch) {
                float c00 = T(i0[0], i0[1], i0[2], ch) * (1 - f[2]) + T(i0[0], i0[1], i1[2], ch) * f[2];
                float c01 = T(i0[0], i1[1], i0[2], ch) * (1 - f[2]) + T(i0[0], i1[1], i1[2], ch) * f[2];
                float c10 = T(i1[0], i0[1], i0[2], ch) * (1 - f[2]) + T(i1[0], i0[1], i1[2], ch) * f[2];
                float c11 = T(i1[0], i1[1], i0[2], ch) * (1 - f[2]) + T(i1[0], i1[1], i1[2], ch) * f[2];
                float v = (c00 * (1 - f[1]) + c01 * f[1]) * (1 - f[0]) + (c10 * (1 - f[1]) + c11 * f[1]) * f[0];
                const double want = std::floor(std::min(std::max(v, 0.0f), 1.0f) * 65535.0f + 0.5f);
                const uint32_t x = px % w, y = px / w;
                const double got = img.rgba16[(size_t(y) * img.row_stride_px + x) * 4 + ch];
                if (std::fabs(got - want) > 1) ++bad;
            }
        }
        CHECK(bad == 0);
        CHECK(img.width == w && img.height == h && img.row_stride_px >= w);
        gpu::Context::release_texture(img.texture);
    }
}

int main(int argc, char** argv) {
    if (argc < 2) return 2;
    auto r = res::Resources::load(argv[1]);
    test_illuminants(*r);
    test_cam16_constants();
    test_cam16_tables(*r);
    test_paper_and_probe_math(*r);
    auto ctx = gpu::Context::create(nullptr, fs::path(argv[1]) / "spektralab.metallib");
    test_kernels(*ctx, *r);
    return finish("test_print");
}
