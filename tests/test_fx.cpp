// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 Guangzhou Songzi Intelligent Technology Services Co., Ltd.
//
// fx: every blur against a CPU float64 reference and against the same-order CPU
// reference; FIR impulse taps; flat-field invariance; mixtures (fused and straddling); halation
// (reference and mobile.2); headroom; the exact Poisson sampler and the lognormal field against a
// host float32 transcription; seeds and streams; grain statistics and strength; glare.
// Usage: test_fx <dir with the metallib> [<res>]   (without <res> the grain tests, which read a film
// profile, are skipped)
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <vector>

#include "check.hpp"
#include "core/render.hpp"
#include "fx/dir_blur.hpp"
#include "fx/fx_internal.hpp"
#include "fx/gaussian.hpp"
#include "fx/glare.hpp"
#include "fx/grain.hpp"
#include "fx/halation.hpp"
#include "fx/rng.hpp"
#include "fx/scanner.hpp"
#include "gpu/gpu.hpp"
#include "res/profile.hpp"
#include "util/numeric.hpp"

using namespace spk;
using fx::Vec3;

namespace {

std::shared_ptr<gpu::Context> g_ctx;

struct Lcg {
    uint64_t s;
    double next() {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        return double(s >> 11) * 0x1p-53;
    }
};

core::RenderContext make_rc(double pitch = 8.68, double rho = 1.0) {
    core::RenderContext rc{.gpu = *g_ctx};
    rc.pitch_um = pitch;
    rc.rho_tier = rho;
    return rc;
}

gpu::Plane upload(const std::vector<float>& v, uint32_t w, uint32_t h) {
    gpu::Plane p = g_ctx->pool().plane(w, h);
    std::memcpy(p.data(), v.data(), v.size() * sizeof(float));
    return p;
}
std::vector<float> download(const gpu::Plane& p) { return std::vector<float>(p.data(), p.data() + p.floats()); }

std::vector<float> test_image(uint32_t w, uint32_t h, uint64_t seed) {
    Lcg r{seed};
    std::vector<float> v(size_t(w) * h * 3);
    for (uint32_t y = 0; y < h; ++y)
        for (uint32_t x = 0; x < w; ++x)
            for (int c = 0; c < 3; ++c) {
                double base = 0.2 + 0.5 * r.next();
                if (x == w / 3 && y == h / 2) base = 40.0;  // an impulse
                if (x > 2 * w / 3) base += 1.5;            // an edge
                v[3 * (size_t(y) * w + x) + size_t(c)] = float(base * (1.0 + 0.3 * c));
            }
    return v;
}

// ---- float64 references (all double) ---------------------------------------------

void fir64(std::vector<double>& p, uint32_t w, uint32_t h, int c, double sigma) {
    const std::vector<double> k = num::gaussian_kernel(sigma, 3.0);
    const int r = int(k.size() / 2);
    std::vector<double> t(p.size());
    for (uint32_t y = 0; y < h; ++y)
        for (uint32_t x = 0; x < w; ++x) {
            double a = 0;
            for (int i = -r; i <= r; ++i) a += k[size_t(i + r)] * p[3 * (size_t(num::reflect_index(int64_t(y) + i, h)) * w + x) + size_t(c)];
            t[3 * (size_t(y) * w + x) + size_t(c)] = a;
        }
    for (uint32_t y = 0; y < h; ++y)
        for (uint32_t x = 0; x < w; ++x) {
            double a = 0;
            for (int i = -r; i <= r; ++i) a += k[size_t(i + r)] * t[3 * (size_t(y) * w + size_t(num::reflect_index(int64_t(x) + i, w))) + size_t(c)];
            p[3 * (size_t(y) * w + x) + size_t(c)] = a;
        }
}

void iir64_line(double* d, size_t n, size_t stride, const fx::IirCoefficients& k) {
    std::vector<double> u(n);
    double w1 = d[0], w2 = w1, w3 = w1;
    for (size_t i = 0; i < n; ++i) {
        double t = ((k.B * d[i * stride] + k.B1 * w1) + k.B2 * w2) + k.B3 * w3;
        w3 = w2, w2 = w1, w1 = t, u[i] = t;
    }
    double y1 = u[n - 1], y2 = y1, y3 = y1;
    for (size_t i = n; i-- > 0;) {
        double t = ((k.B * u[i] + k.B1 * y1) + k.B2 * y2) + k.B3 * y3;
        y3 = y2, y2 = y1, y1 = t, d[i * stride] = t;
    }
}

std::vector<double> gaussian64(const std::vector<double>& in, uint32_t w, uint32_t h, const Vec3& s) {
    std::vector<double> p = in;
    for (int c = 0; c < 3; ++c) {
        const double sg = s[size_t(c)];
        if (fx::branch(sg) == fx::Branch::Fir) fir64(p, w, h, c, sg);
        if (fx::branch(sg) == fx::Branch::Iir) {
            const fx::IirCoefficients k = fx::iir_coefficients(sg);
            for (uint32_t y = 0; y < h; ++y) iir64_line(p.data() + 3 * size_t(y) * w + size_t(c), w, 3, k);
            for (uint32_t x = 0; x < w; ++x) iir64_line(p.data() + 3 * size_t(x) + size_t(c), h, 3 * size_t(w), k);
        }
    }
    return p;
}

std::vector<double> widen(const std::vector<float>& v) { return std::vector<double>(v.begin(), v.end()); }

double max_rel_err(const std::vector<float>& got, const std::vector<double>& ref) {
    double scale = 0, e = 0;
    for (size_t i = 0; i < ref.size(); ++i) scale = std::max(scale, std::fabs(ref[i]));
    for (size_t i = 0; i < ref.size(); ++i) e = std::max(e, std::fabs(double(got[i]) - ref[i]));
    return e / std::max(scale, 1e-30);
}

// distance in float ULPs (same sign assumed; inputs positive here)
int64_t ulp_diff(float a, float b) {
    int32_t ia, ib;
    std::memcpy(&ia, &a, 4);
    std::memcpy(&ib, &b, 4);
    return std::llabs(int64_t(ia) - int64_t(ib));
}

// ---- tests -----------------------------------------------------------------------------------------

void test_constants() {
    for (double s : {3.0, 4.6, 13.9, 127.6, 261.0}) {
        fx::IirCoefficients k = fx::iir_coefficients(s);
        CHECK(std::fabs(k.B + k.B1 + k.B2 + k.B3 - 1.0) < 1e-14);
    }
    CHECK(fx::branch(2.9) == fx::Branch::Fir && fx::branch(3.0) == fx::Branch::Iir && fx::branch(0.0) == fx::Branch::Identity);
    CHECK(fx::fir_radius(2.9) == 9 && fx::fir_radius(0.65) == 2 && fx::fir_radius(0.5) == 2 && fx::fir_radius(1e-6) == 0);
    auto sur = fx::exponential_surrogate({1, 1, 1}, {10, 10, 10});
    CHECK(std::fabs(sur[0].weight[0] + sur[1].weight[0] + sur[2].weight[0] - 0.9999) < 1e-12);
    CHECK(std::fabs(sur[2].sigma[1] - 27.684) < 1e-12);
    // splitmix64 first four draws
    fx::SeedSequence q;
    CHECK(q.next() == 0x4abea221u && q.next() == 0x23148989u && q.next() == 0x609dfe03u && q.next() == 0xdeb12800u);
    auto a = fx::philox4x32_10({0, 200, 0, 0}, {0x5EED, 0});
    CHECK(a[0] == 0x15cecb3fu && a[3] == 0xf78f6c18u);
    // mobile.2 tent variances: (f^2 + 0.5)/6 for even f
    CHECK(std::fabs(fx::detail::tent_variance(2) - 0.75) < 1e-12);
    CHECK(std::fabs(fx::detail::tent_variance(4) - 2.75) < 1e-12);
    CHECK(std::fabs(fx::detail::tent_variance(8) - 10.75) < 1e-12);
    fx::MixtureComponent m{{1, 1, 1}, {24.45, 30, 30}};
    CHECK(fx::detail::component_factor(m) == 8);
    m.sigma = {24.43, 30, 30};
    CHECK(fx::detail::component_factor(m) == 4);
    m.sigma = {6.11, 30, 30};
    CHECK(fx::detail::component_factor(m) == 1);
    m.weight = {0, 1, 1};  // the zero-weight channel does not count
    CHECK(fx::detail::component_factor(m) == 8);
}

void test_blurs() {
    const uint32_t w = 97, h = 61;
    const std::vector<float> img = test_image(w, h, 7);
    const std::vector<Vec3> sigmas = {{0.5, 0.5, 0.5},    {2.9, 2.9, 2.9},  {3.0, 3.0, 3.0},    {1.3, 0.0, 2.2},
                                      {5.0, 1.2, 0.0},    {40, 13, 7},      {130, 130, 130},    {2.99, 3.0, 3.01},
                                      {1e-6, 1e-6, 1e-6}, {0.0, 0.0, 0.0},  {4.6, 4.6, 4.6}};
    core::RenderContext rc = make_rc();
    double worst_fir = 0, worst_iir = 0;
    int64_t worst_ulp = 0;
    for (const Vec3& s : sigmas) {
        gpu::Plane in = upload(img, w, h);
        gpu::Plane out = g_ctx->pool().plane(w, h);
        fx::gaussian(rc, in, out, s);
        const std::vector<float> got = download(out);
        // same-order CPU reference (float storage, double IIR state)
        std::vector<float> same(img.size());
        fx::cpu::gaussian(img.data(), same.data(), w, h, s);
        int64_t mu = 0;
        double scale = 0;
        for (float v : same) scale = std::max(scale, double(std::fabs(v)));
        for (size_t i = 0; i < got.size(); ++i) {
            // ULPs where the value is not tiny relative to the plane
            if (std::fabs(same[i]) > 1e-3 * scale) mu = std::max(mu, ulp_diff(got[i], same[i]));
        }
        worst_ulp = std::max(worst_ulp, mu);
        const double e = max_rel_err(got, gaussian64(widen(img), w, h, s));
        const bool any_iir = fx::branch(s[0]) == fx::Branch::Iir || fx::branch(s[1]) == fx::Branch::Iir ||
                             fx::branch(s[2]) == fx::Branch::Iir;
        (any_iir ? worst_iir : worst_fir) = std::max(any_iir ? worst_iir : worst_fir, e);
        std::printf("  gaussian sigma (%g, %g, %g): vs float64 %.2e, vs same-order CPU max %lld ulp\n", s[0], s[1], s[2], e,
                    (long long)mu);
        CHECK(e < (any_iir ? 1e-5 : 2e-6));
        CHECK(mu <= 4);
        // in place gives the same bytes
        gpu::Plane ip = upload(img, w, h);
        fx::gaussian(rc, ip, ip, s);
        CHECK(std::memcmp(ip.data(), out.data(), out.bytes()) == 0);
    }
    std::printf("  worst: FIR %.2e, IIR %.2e relative to float64; %lld ulp vs same order\n", worst_fir, worst_iir,
                (long long)worst_ulp);
    // sigma 0 / R = 0: exact identity
    {
        gpu::Plane in = upload(img, w, h), out = g_ctx->pool().plane(w, h);
        fx::gaussian(rc, in, out, {0.0, 1e-6, -1.0});
        CHECK(std::memcmp(in.data(), out.data(), in.bytes()) == 0);
    }
    // degenerate lines: 1 x n and n x 1, both branches
    for (auto wh : {std::pair<uint32_t, uint32_t>{1, 40}, {40, 1}, {1, 1}, {2, 3}}) {
        std::vector<float> v = test_image(wh.first, wh.second, 3);
        for (Vec3 s : {Vec3{1.5, 1.5, 1.5}, Vec3{6.0, 6.0, 6.0}}) {
            gpu::Plane in = upload(v, wh.first, wh.second), out = g_ctx->pool().plane(wh.first, wh.second);
            fx::gaussian(rc, in, out, s);
            CHECK(max_rel_err(download(out), gaussian64(widen(v), wh.first, wh.second, s)) < 1e-5);
        }
    }
}

void test_fir_impulse() {
    // A 1-pixel impulse's response equals the taps of exactly.
    const uint32_t w = 41, h = 41;
    std::vector<float> img(size_t(w) * h * 3, 0.0f);
    for (int c = 0; c < 3; ++c) img[3 * (20 * w + 20) + size_t(c)] = 1.0f;
    core::RenderContext rc = make_rc();
    for (double s : {0.5, 1.0, 2.9}) {
        gpu::Plane in = upload(img, w, h), out = g_ctx->pool().plane(w, h);
        fx::gaussian(rc, in, out, {s, s, s});
        const std::vector<float> k = fx::fir_taps(s);
        const int r = int(k.size() / 2);
        bool exact = true;
        for (int dy = -r; dy <= r; ++dy)
            for (int dx = -r; dx <= r; ++dx) {
                const float want = k[size_t(dx + r)] * k[size_t(dy + r)];
                exact = exact && out.px(uint32_t(20 + dx), uint32_t(20 + dy))[0] == want;
            }
        CHECK(exact);
        CHECK(out.px(20 + uint32_t(r) + 1, 20)[0] == 0.0f);
    }
    // sigma 3.0 is IIR: the response reaches beyond R = 9
    gpu::Plane in = upload(img, w, h), out = g_ctx->pool().plane(w, h);
    fx::gaussian(rc, in, out, {3.0, 3.0, 3.0});
    CHECK(out.px(20 + 10, 20)[0] != 0.0f);
    // reflect vs replicate edges: an impulse at the edge
    std::vector<float> e(size_t(w) * h * 3, 0.0f);
    for (int c = 0; c < 3; ++c) e[3 * (20 * w + 0) + size_t(c)] = 1.0f;
    gpu::Plane ei = upload(e, w, h), eo = g_ctx->pool().plane(w, h);
    fx::gaussian(rc, ei, eo, {2.0, 2.0, 2.0});
    const std::vector<float> k = fx::fir_taps(2.0);
    // reflected: x = 0 gets taps at offsets 0 and -1 (mirror of 0): k[r] + k[r+1] along x
    const float want = (0.0f + k[6] * 1.0f);  // vertical centre tap
    const float hx = k[6] * want + k[7] * want;  // left neighbour reflected onto x = 0 ... order -R..R
    (void)hx;
    CHECK(eo.px(0, 20)[0] > eo.px(1, 20)[0]);
}

void test_flat_field() {
    const uint32_t w = 120, h = 90;
    std::vector<float> img(size_t(w) * h * 3, 0.184f);
    core::RenderContext rc = make_rc();
    for (Vec3 s : {Vec3{0.7, 2.9, 2.0}, Vec3{3.0, 15.0, 26.0}, Vec3{127.6, 127.6, 127.6}, Vec3{2.0, 9.0, 0.0}}) {
        gpu::Plane in = upload(img, w, h), out = g_ctx->pool().plane(w, h);
        fx::gaussian(rc, in, out, s);
        double e = 0;
        for (float v : download(out)) e = std::max(e, std::fabs(v / 0.184 - 1.0));
        std::printf("  flat field sigma (%g, %g, %g): max relative deviation %.2e\n", s[0], s[1], s[2], e);
        CHECK(e < 1e-6);
    }
}

void test_mixtures() {
    const uint32_t w = 80, h = 50;
    const std::vector<float> img = test_image(w, h, 11);
    core::RenderContext rc = make_rc();
    const std::vector<std::vector<fx::MixtureComponent>> cases = {
        {{{0.3, 0.3, 0.3}, {0.5, 0.5, 0.5}}, {{0.7, 0.7, 0.7}, {2.0, 2.0, 2.0}}},             // FIR only
        {{{0.5, 0.5, 0.5}, {4.0, 4.0, 4.0}}, {{0.5, 0.5, 0.5}, {9.0, 9.0, 9.0}}},             // IIR only
        {{{0.22, 0.35, 0.33}, {0.51, 0.46, 0.37}}, {{0.13, 0.11, 0.11}, {1.15, 1.2, 1.1}},
         {{0.5, 0.4, 0.4}, {3.3, 3.4, 2.9}}, {{0.15, 0.12, 0.13}, {5.9, 6.2, 5.8}}},            // scatter-like, straddling
        {{{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}}, {{1.0, 1.0, 1.0}, {3.5, 1e-6, 0.0}}},             // identity first, straddle
        {{{-0.5, 0.2, 1.0}, {20, 18, 10}}, {{1.5, 0.8, 0.0}, {30, 28, 15}}},                  // signed weights
    };
    for (const auto& comps : cases) {
        gpu::Plane in = upload(img, w, h), out = g_ctx->pool().plane(w, h);
        fx::gaussian_mixture(rc, in, out, comps);
        std::vector<double> ref(img.size(), 0.0);
        for (const auto& m : comps) {
            std::vector<double> g = gaussian64(widen(img), w, h, m.sigma);
            for (size_t i = 0; i < ref.size(); ++i) ref[i] += m.weight[i % 3] * g[i];
        }
        const double e = max_rel_err(download(out), ref);
        std::printf("  mixture of %zu: vs float64 %.2e\n", comps.size(), e);
        CHECK(e < 1e-5);
        // in place
        gpu::Plane ip = upload(img, w, h);
        fx::gaussian_mixture(rc, ip, ip, comps);
        CHECK(std::memcmp(ip.data(), out.data(), out.bytes()) == 0);
    }
}

void test_halation() {
    // preset
    res::ProfileInfo info;
    info.antihalation = "strong";
    info.use = "cine";
    fx::HalationParams h = fx::resolve_halation_preset(fx::HalationParams{}, info, false);
    CHECK(h.strength[0] == 0.015 && h.first_sigma_um[0] == 50.0 && !h.back_reflection);
    h = fx::resolve_halation_preset(fx::HalationParams{}, info, true);  // CineStill-like
    CHECK(h.strength[0] == 0.30 && h.strength[2] == 0.003 && h.back_reflection);
    info.use = "other";
    h = fx::resolve_halation_preset(fx::HalationParams{}, info, false);
    CHECK(h.strength[0] == 0.05 && h.first_sigma_um[0] == 65.0 && !h.back_reflection);
    info.support = "paper";
    info.use = "still";
    h = fx::resolve_halation_preset(fx::HalationParams{}, info, true);
    CHECK(!h.back_reflection && h.strength[1] == 0.015);

    // flat field: renormalised halation keeps a constant (scatter loses 1e-4 t)
    const uint32_t w = 256, h_ = 192;
    std::vector<float> flat(size_t(w) * h_ * 3, 0.184f);
    for (bool br : {false, true}) {
        fx::HalationParams hp;
        hp.strength = br ? Vec3{0.30, 0.03, 0.003} : Vec3{0.08, 0.02, 0.0};
        hp.back_reflection = br;
        core::RenderContext rc = make_rc(4.34);
        gpu::Plane p = upload(flat, w, h_);
        fx::halation(rc, p, hp);
        double e = 0;
        for (size_t i = 0; i < p.floats(); ++i) {
            const double t = hp.scatter_tail_weight[i % 3];
            e = std::max(e, std::fabs(p.data()[i] / 0.184 - (1.0 - 1e-4 * t)));
        }
        std::printf("  halation flat field (back-reflection %d): max deviation %.2e\n", int(br), e);
        CHECK(e < 2e-6);
    }
    // reference vs float64 on a point target, and mobile.2 vs reference
    std::vector<float> pts(size_t(w) * h_ * 3, 0.184f / 16);
    for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx)
            for (int c = 0; c < 3; ++c) pts[3 * (size_t(96 + dy) * w + 128 + dx) + size_t(c)] = 0.184f * 11.3f;
    for (bool br : {false, true}) {
        fx::HalationParams hp;
        hp.strength = br ? Vec3{0.30, 0.03, 0.003} : Vec3{0.015, 0.005, 0.0};
        hp.back_reflection = br;
        const double pitch = br ? 3.0 : 1.2;  // large sigmas in px so that mobile has coarse components
        core::RenderContext rc = make_rc(pitch);
        const fx::detail::HalationPlan pl = fx::detail::halation_plan(hp, pitch);
        gpu::Plane ref = upload(pts, w, h_);
        fx::halation(rc, ref, hp);
        // float64 transcription
        std::vector<double> x = widen(pts);
        std::vector<double> m(x.size(), 0.0);
        for (const auto& c : pl.scatter_comps) {
            auto g = gaussian64(x, w, h_, c.sigma);
            for (size_t i = 0; i < x.size(); ++i) m[i] += c.weight[i % 3] * g[i];
        }
        for (size_t i = 0; i < x.size(); ++i) x[i] = pl.scatter_a[i % 3] * x[i] + pl.scatter_b[i % 3] * m[i];
        std::fill(m.begin(), m.end(), 0.0);
        for (const auto& c : pl.bounce_comps) {
            auto g = gaussian64(x, w, h_, c.sigma);
            for (size_t i = 0; i < x.size(); ++i) m[i] += c.weight[i % 3] * g[i];
        }
        for (size_t i = 0; i < x.size(); ++i) x[i] = pl.mix_a[i % 3] * x[i] + pl.mix_b[i % 3] * m[i];
        const double e = max_rel_err(download(ref), x);
        std::printf("  halation points (back-reflection %d): vs float64 %.2e\n", int(br), e);
        CHECK(e < 1e-5);

        core::RenderContext rm = make_rc(pitch);
        rm.variants = core::EffectVariants::for_device(core::Device::Phone);
        const int F = fx::detail::mobile_grid_factor(pl.bounce_comps);
        gpu::Plane mob = upload(pts, w, h_);
        fx::halation(rm, mob, hp);
        double num = 0, den = 0, emax = 0, sum_ref = 0, sum_mob = 0;
        for (size_t i = 0; i < mob.floats(); ++i) {
            const double d = double(mob.data()[i]) - ref.data()[i];
            num += d * d;
            den += double(ref.data()[i]) * ref.data()[i];
            emax = std::max(emax, std::fabs(d));
            sum_ref += ref.data()[i];
            sum_mob += mob.data()[i];
        }
        std::printf("  mobile.2 (F = %d): relative L2 %.2e, max abs %.2e, energy ratio %.7f\n", F, std::sqrt(num / den), emax,
                    sum_mob / sum_ref);
        CHECK(F > 1);
        CHECK(std::sqrt(num / den) < 5e-3);
        CHECK(std::fabs(sum_mob / sum_ref - 1.0) < 3e-4);  // sigmas up to 1.5x the frame: edge handling differs
    }
    // small pitch-in-px: mobile = reference byte for byte
    {
        fx::HalationParams hp;
        hp.strength = {0.015, 0.005, 0.0};
        core::RenderContext rc = make_rc(21.875), rm = make_rc(21.875);
        rm.variants = core::EffectVariants::for_device(core::Device::Phone);
        gpu::Plane a = upload(pts, w, h_), b = upload(pts, w, h_);
        fx::halation(rc, a, hp);
        fx::halation(rm, b, hp);
        CHECK(std::memcmp(a.data(), b.data(), a.bytes()) == 0);
    }
    // amount 0: scatter only; inactive: identity
    {
        fx::HalationParams hp;
        hp.amount = 0.0;
        CHECK(!fx::detail::halation_plan(hp, 4.34).bounce && fx::detail::halation_plan(hp, 4.34).scatter);
        hp.active = false;
        core::RenderContext rc = make_rc(4.34);
        gpu::Plane a = upload(pts, w, h_);
        fx::halation(rc, a, hp);
        CHECK(std::memcmp(a.data(), pts.data(), a.bytes()) == 0);
    }
    // headroom against a host transcription
    {
        std::vector<float> v = test_image(64, 32, 5);
        v[3 * 100] = 30.0f;
        gpu::Plane p = upload(v, 64, 32);
        fx::HalationParams hp;
        hp.back_reflection = true;
        core::RenderContext rc = make_rc();
        const float clip = fx::headroom_clip(rc, p);
        CHECK(clip == 40.0f * 1.6f);
        fx::highlight_headroom(rc, p, hp, clip);
        int bad = 0;
        auto smooth = [](float t) {
            float u = std::min(std::max(t, 0.0f), 1.0f);
            return u * u * (3.0f - 2.0f * u);
        };
        for (size_t i = 0; i < p.pixels(); ++i) {
            const float m = std::max(std::max(v[3 * i], v[3 * i + 1]), v[3 * i + 2]);
            const float s1 = smooth(((std::log2(m) - std::log2(clip)) - (-1.0f)) / 0.75f);
            const float s2 = smooth(((std::log2(m) - std::log2(0.184f)) - 1.5f) / 1.0f);
            const float g = std::exp2((4.0f * s1) * s2);
            for (int c = 0; c < 3; ++c) bad += ulp_diff(p.data()[3 * i + size_t(c)], v[3 * i + size_t(c)] * g) > 2;
        }
        CHECK(bad == 0);
        hp.back_reflection = false;  // films with a layer ignore headroom
        gpu::Plane q = upload(v, 64, 32);
        fx::highlight_headroom(rc, q, hp, clip);
        CHECK(std::memcmp(q.data(), v.data(), q.bytes()) == 0);
    }
}

// ---- the sampler: host float32 transcription ------------------------------------

struct HostStream {
    uint32_t pixel, stream, seed, j = 0, used = 4;
    std::array<uint32_t, 4> block{};
    float next() {
        if (used == 4) {
            block = fx::philox4x32_10({pixel, stream, j, 0}, {seed, 0});
            ++j;
            used = 0;
        }
        return fx::philox_uniform(block[used++]);
    }
};
const float kT = float(6.283185307179586);
float host_lgamma(float x) {
    if (x == 1.0f || x == 2.0f) return 0.0f;
    float x0 = x;
    int n = 0;
    if (x <= 7.0f) {
        n = int(7.0f - x);
        x0 = x + float(n);
    }
    const float x2 = 1.0f / (x0 * x0);
    float g = float(8.417508417508418e-04);
    g = g * x2 + float(-5.952380952380952e-04);
    g = g * x2 + float(7.936507936507937e-04);
    g = g * x2 + float(-2.777777777777778e-03);
    g = g * x2 + float(8.333333333333333e-02);
    float gl = ((g / x0 + 0.5f * std::log(kT)) + (x0 - 0.5f) * std::log(x0)) - x0;
    if (x <= 7.0f)
        for (int i = 0; i < n; ++i) {
            gl = gl - std::log(x0 - 1.0f);
            x0 = x0 - 1.0f;
        }
    return gl;
}
float host_poisson(float lam, uint32_t pixel, uint32_t stream, uint32_t seed) {
    HostStream g{pixel, stream, seed};
    if (lam >= 10.0f) {
        const float slam = std::sqrt(lam), loglam = std::log(lam);
        const float b = 0.931f + (2.53f * slam);
        const float a = -0.059f + (0.02483f * b);
        const float invalpha = 1.1239f + (1.1328f / (b - 3.4f));
        const float vr = 0.9277f - (3.6224f / (b - 2.0f));
        for (int it = 0; it < 64; ++it) {
            const float U = g.next() - 0.5f, V = g.next();
            const float us = 0.5f - std::fabs(U);
            const float k = std::floor(((((2.0f * a) / us + b) * U + lam) + 0.43f));
            if (us >= 0.07f && V <= vr) return k;
            if (k < 0.0f || (us < 0.013f && V > us)) continue;
            const float L = ((std::log(V) + std::log(invalpha)) - std::log(a / (us * us) + b));
            const float Rt = ((-lam + k * loglam) - host_lgamma(k + 1.0f));
            if (L <= Rt) return k;
        }
        return std::floor(lam);
    }
    const float E = std::exp(-lam);
    float X = 0.0f, prod = 1.0f;
    for (int it = 0; it < 512; ++it) {
        prod = prod * g.next();
        if (prod > E) X = X + 1.0f;
        else return X;
    }
    return X;
}
float host_lognormal(float m, float s, uint32_t pixel, uint32_t stream, uint32_t seed) {
    HostStream g{pixel, stream, seed};
    const float u1 = g.next(), u2 = g.next();
    const float z = std::sqrt(-2.0f * std::log(u1)) * std::cos(kT * u2);
    const float sig = std::sqrt(std::log(1.0f + (s * s) / (m * m)));
    const float mu = std::log(m) - (0.5f * sig) * sig;
    return std::exp(mu + sig * z);
}

void test_sampler() {
    struct Args {
        uint32_t n, stream, seed;
        float mean, stddev;
    };
    for (float lam : {0.0f, 0.5f, 3.0f, 9.99f, 10.0f, 47.5f, 300.0f, 2500.0f}) {
        const uint32_t n = 200000;
        gpu::Buffer l = g_ctx->pool().acquire(n * 4), o = g_ctx->pool().acquire(n * 4), o2 = g_ctx->pool().acquire(n * 4);
        for (uint32_t i = 0; i < n; ++i) l.as<float>()[i] = lam;
        Args a{n, 11, 0x4abea221u, 0.03f, 0.021f};
        gpu::Batch bt = g_ctx->batch();
        bt.dispatch_1d(g_ctx->kernel("spk_fx_grain_test_sampler"), n,
                       {gpu::Arg::buffer(l), gpu::Arg::buffer(o), gpu::Arg::buffer(o2), gpu::Arg::value(a)});
        bt.commit_and_wait();
        double s1 = 0, s2 = 0, s3 = 0;
        uint32_t mism = 0, lmism = 0;
        int64_t lmax = 0;
        for (uint32_t i = 0; i < n; ++i) {
            const double k = o.as<float>()[i];
            s1 += k;
            if (i < 20000) {
                mism += k != host_poisson(lam, i, 11, a.seed);
                const int64_t ud = ulp_diff(o2.as<float>()[i], host_lognormal(a.mean, a.stddev, i, 11, a.seed));
                lmax = std::max(lmax, ud);
                lmism += ud > 16;  // host libm vs Metal precise log/cos/exp, amplified by exp
            }
        }
        const double mean = s1 / n;
        for (uint32_t i = 0; i < n; ++i) {
            const double d = o.as<float>()[i] - mean;
            s2 += d * d;
            s3 += d * d * d;
        }
        const double var = s2 / n, skew = var > 0 ? (s3 / n) / std::pow(var, 1.5) : 0;
        std::printf("  Poisson(%g): mean %.4f var %.4f skew %.4f (want %.4f); host mismatches %u/20000, lognormal > 16 ulp %u (max %lld)\n",
                    lam, mean, var, skew, lam > 0 ? 1.0 / std::sqrt(lam) : 0.0, mism, lmism, (long long)lmax);
        CHECK(std::fabs(mean - lam) < 5.0 * std::sqrt(std::max(lam, 1e-3f) / n) + 1e-9);
        if (lam > 0) {
            CHECK(std::fabs(var / lam - 1.0) < 0.02);
            CHECK(std::fabs(skew - 1.0 / std::sqrt(lam)) < 0.05);
        }
        // The host transcription rounds every operation; the kernel keeps Metal's safe-mode FMA
        // contraction, so a rare accept/reject decision can differ by a draw.
        CHECK(mism <= 100);
        CHECK(lmism == 0);
    }
    // lognormal field statistics: mean m, CV r
    {
        const uint32_t n = 400000;
        gpu::Buffer l = g_ctx->pool().acquire(n * 4), o = g_ctx->pool().acquire(n * 4), o2 = g_ctx->pool().acquire(n * 4);
        std::memset(l.contents(), 0, n * 4);
        Args a{n, 200, 0x23148989u, 0.03f, 0.021f};
        gpu::Batch bt = g_ctx->batch();
        bt.dispatch_1d(g_ctx->kernel("spk_fx_grain_test_sampler"), n,
                       {gpu::Arg::buffer(l), gpu::Arg::buffer(o), gpu::Arg::buffer(o2), gpu::Arg::value(a)});
        bt.commit_and_wait();
        double s = 0, s2 = 0;
        for (uint32_t i = 0; i < n; ++i) s += o2.as<float>()[i];
        const double m = s / n;
        for (uint32_t i = 0; i < n; ++i) s2 += (o2.as<float>()[i] - m) * (o2.as<float>()[i] - m);
        const double cv = std::sqrt(s2 / n) / m;
        std::printf("  lognormal(0.03, 0.021): mean %.6f, CV %.4f\n", m, cv);
        CHECK(std::fabs(m / 0.03 - 1) < 0.005 && std::fabs(cv - 0.7) < 0.01);
    }
}

void test_grain(const std::filesystem::path& res) {
    auto film = res::load_profile(res, "kodak_portra_400");
    auto g = fx::build_grain_setup(*film, fx::GrainOptions{});
    const double pitch = 8.68;
    const fx::detail::GrainConstants kc = fx::detail::grain_constants(*g, pitch);
    CHECK(!kc.dye && !kc.micro && kc.final_blur == 0.65);
    const uint32_t w = 256, h = 256;
    std::vector<float> taps = fx::fir_taps(0.65);
    double k2 = 0;
    for (float t : taps) k2 += double(t) * t;
    for (float level : {0.05f, 0.3f, 0.8f, 1.5f}) {
        std::vector<float> flat(size_t(w) * h * 3, level);
        gpu::Plane p = upload(flat, w, h);
        core::RenderContext rc = make_rc(pitch);
        fx::grain(rc, p, *g, 0x4abea221u);
        // interior statistics (away from the reflected edge)
        for (int c = 0; c < 3; ++c) {
            double s = 0, s2 = 0;
            size_t n = 0;
            for (uint32_t y = 4; y < h - 4; ++y)
                for (uint32_t x = 4; x < w - 4; ++x) {
                    const double v = p.px(x, y)[c];
                    s += v, s2 += v * v, ++n;
                }
            const double mean = s / n, var = s2 / n - mean * mean;
            // predicted variance: sum_l d_l (D_l/n_l) sat_l times (sum k^2)^2
            double pv = 0, pm = 0;
            {
                const auto& L = film->density_curves_layers;
                const size_t K = film->curve_points();
                // locate on the double axis (interpolation in double for the prediction)
                std::vector<double> ax(K);
                for (size_t k = 0; k < K; ++k) ax[k] = film->normalized_curves(k, size_t(c));
                for (int l = 0; l < 3; ++l) {
                    std::vector<double> yl(K);
                    for (size_t k = 0; k < K; ++k) yl[k] = L[(k * 3 + size_t(l)) * 3 + size_t(c)];
                    const double dl = num::interp(level, ax, yl) + kc.dmin_l[l][c];
                    const double D = kc.sat_d[l][c], nn = kc.count[l][c];
                    const double u = c == 1 ? 0.99 : 0.97;
                    const double q = std::min(std::max(dl / D, 1e-6), 1 - 1e-6);
                    const double sat = 1 - q * u * (1 - 1e-6);
                    pv += dl * (D / nn) * sat;
                    pm += dl - kc.dmin_l[l][c];  // E[G] - D_min = sum of the interpolated layers
                }
            }
            pv *= k2 * k2;
            std::printf("  grain level %.2f ch %d: mean %.5f (in %.5f, layers %.5f), var %.3e (predicted %.3e, ratio %.3f)\n", level, c,
                        mean, double(level), pm, var, pv, var / pv);
            CHECK(std::fabs(mean - pm) < 4.0 * std::sqrt(var / n * 9.0) + 2e-5);
            CHECK(std::fabs(mean - level) < 0.006);  // the layers sum to the curves within 0.005
            CHECK(std::fabs(var / pv - 1.0) < 0.06);
        }
    }
    // amount 0 is byte-equal to off
    {
        std::vector<float> img(size_t(w) * h * 3);
        for (size_t i = 0; i < img.size(); ++i) img[i] = 0.2f + 0.001f * float(i % 997);
        auto g0 = fx::build_grain_setup(*film, fx::GrainOptions{true, true, 0.0});
        gpu::Plane p = upload(img, w, h);
        core::RenderContext rc = make_rc(pitch);
        fx::grain(rc, p, *g0, 1234);
        CHECK(std::memcmp(p.data(), img.data(), p.bytes()) == 0);

        gpu::Plane whole = upload(img, w, h);
        fx::grain(rc, whole, *g, 99);
        // amount 0.5 halves the deviation
        auto gh = fx::build_grain_setup(*film, fx::GrainOptions{true, true, 0.5});
        gpu::Plane ph = upload(img, w, h);
        fx::grain(rc, ph, *gh, 99);
        double dmax = 0;
        for (size_t i = 0; i < img.size(); ++i) {
            const double want = 0.5 * img[i] + 0.5 * whole.data()[i];
            dmax = std::max(dmax, std::fabs(ph.data()[i] - want));
        }
        CHECK(dmax < 1e-6);
    }
    // the simple model; dye clouds and micro-structure wake at small pitches
    {
        auto gs = fx::build_grain_setup(*film, fx::GrainOptions{true, false, 1.0});
        std::vector<float> flat(size_t(64) * 64 * 3, 0.5f);
        gpu::Plane p = upload(flat, 64, 64);
        core::RenderContext rc = make_rc(pitch);
        fx::grain(rc, p, *gs, 7);
        double s = 0;
        for (size_t i = 0; i < p.floats(); ++i) s += p.data()[i];
        CHECK(std::fabs(s / double(p.floats()) - 0.5) < 0.01);
        const fx::detail::GrainConstants kd = fx::detail::grain_constants(*g, 0.4);
        CHECK(kd.dye && kd.micro);
        gpu::Plane q = upload(flat, 64, 64);
        core::RenderContext r4 = make_rc(0.4);
        fx::grain(r4, q, *g, 7);
        double m = 0;
        for (size_t i = 0; i < q.floats(); ++i) m += q.data()[i];
        std::printf("  grain at p = 0.4 um (dye clouds + micro-structure): mean %.5f\n", m / double(q.floats()));
        CHECK(std::fabs(m / double(q.floats()) - 0.5) < 0.01);
    }
    // a film without layers is refused with sub-layers on
    {
        res::Profile nl = *film;
        nl.density_curves_layers.clear();
        CHECK_THROWS(fx::build_grain_setup(nl, fx::GrainOptions{}));
        fx::build_grain_setup(nl, fx::GrainOptions{true, false, 1.0});
        fx::build_grain_setup(nl, fx::GrainOptions{false, true, 1.0});
    }
}

void test_glare() {
    const uint32_t w = 300, h = 200;
    std::vector<float> xyz(size_t(w) * h * 3, 0.5f);
    fx::GlareParams g;
    g.tint_xyz = {0.9642, 1.0, 0.8251};
    for (double amount : {1.0, 10.0, 30.0}) {
        g.amount = amount;
        gpu::Plane p = upload(xyz, w, h);
        core::RenderContext rc = make_rc();
        fx::glare(rc, p, g, 0x23148989u);
        double add[3] = {0, 0, 0};
        bool corr = true;
        for (size_t i = 0; i < p.pixels(); ++i) {
            for (int c = 0; c < 3; ++c) add[c] += p.data()[3 * i + size_t(c)] - 0.5;
            const double ax = p.data()[3 * i] - 0.5, ay = p.data()[3 * i + 1] - 0.5;
            corr = corr && std::fabs(ax / 0.9642 - ay) < 1e-6;
        }
        for (int c = 0; c < 3; ++c) {
            const double m = add[c] / double(p.pixels()), want = 0.03 * amount / 100.0 * g.tint_xyz[size_t(c)];
            CHECK(std::fabs(m / want - 1.0) < 0.02);
        }
        CHECK(corr);
        std::printf("  glare amount %g: mean added Y %.3e (want %.3e)\n", amount, add[1] / double(p.pixels()),
                    0.03 * amount / 100.0);
    }
    g.active = false;
    CHECK(!fx::glare_draws_seed(g));
    gpu::Plane p = upload(xyz, w, h);
    core::RenderContext rc = make_rc();
    fx::glare(rc, p, g, 1);
    CHECK(std::memcmp(p.data(), xyz.data(), p.bytes()) == 0);
}

void test_scanner_dir() {
    const uint32_t w = 90, h = 70;
    const std::vector<float> img = test_image(w, h, 13);
    // scanner: blur then unsharp, against float64
    fx::ScannerSpatial s;
    s.lens_blur_px = 3.0;
    core::RenderContext rc = make_rc(8.68, 0.5);
    gpu::Plane p = upload(img, w, h);
    fx::scanner_blur_unsharp(rc, p, s);
    std::vector<double> x = gaussian64(widen(img), w, h, {3.0, 3.0, 3.0});
    std::vector<double> gx = gaussian64(x, w, h, {0.35, 0.35, 0.35});
    for (size_t i = 0; i < x.size(); ++i) x[i] = 1.7 * x[i] - 0.7 * gx[i];
    const double e = max_rel_err(download(p), x);
    std::printf("  scanner blur 3 px + unsharp at rho 0.5: vs float64 %.2e\n", e);
    CHECK(e < 1e-5);
    // DIR diffusion on a flat field keeps the constant
    std::vector<float> flat(size_t(w) * h * 3, -0.3f);
    gpu::Plane f = upload(flat, w, h);
    core::RenderContext rd = make_rc(4.34);
    fx::dir_diffusion_blur(rd, f, {});
    double dev = 0;
    for (float v : download(f)) dev = std::max(dev, std::fabs(v / -0.3 - (1.0 - 0.06 * 1e-4)));
    std::printf("  DIR flat field: max deviation %.2e\n", dev);
    CHECK(dev < 2e-6);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) return 2;
    g_ctx = gpu::Context::create(nullptr, std::filesystem::path(argv[1]) / "spektralab.metallib");
    test_constants();
    test_blurs();
    test_fir_impulse();
    test_flat_field();
    test_mixtures();
    test_halation();
    test_sampler();
    if (argc >= 3) test_grain(argv[2]);
    else std::printf("  grain tests skipped: no resources folder given\n");
    test_glare();
    test_scanner_dir();
    return finish("test_fx");
}
