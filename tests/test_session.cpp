// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 Guangzhou Songzi Intelligent Technology Services Co., Ltd.
//
// Session pieces that need no engine: tier sizes and the tier ratio, and the progress record's pct.
// Usage: test_session [<dir with the metallib>]
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <random>
#include <vector>

#include "check.hpp"
#include "core/render.hpp"
#include "session/progress.hpp"
#include "session/tiers.hpp"

using namespace spk;

static void test_tier_sizes() {
    using session::downscale_size;
    using session::tier_ratio;
    // The float64 floor can give L - 1: M = 2156 at L = 1600 -> 1599.
    session::Size t = downscale_size(2156, 1437, 1600);
    CHECK(t.w == 1599);
    session::Size full{2156, 1437};
    CHECK(tier_ratio(t, full) == 1599.0 / 2156.0);
    CHECK(tier_ratio(t, full) != 1600.0 / 2156.0);
    CHECK(tier_ratio(full, full) == 1.0);
    // M <= L and the full tier: the frame itself.
    session::Size same = downscale_size(1200, 800, 1600);
    CHECK(same.w == 1200 && same.h == 800);
    session::Size f = downscale_size(6000, 4000, 0);
    CHECK(f.w == 6000 && f.h == 4000);
    // An exact case keeps L.
    session::Size e = downscale_size(3200, 2400, 1600);
    CHECK(e.w == 1600 && e.h == 1200);
    // Portrait: the long edge is H.
    session::Size p = downscale_size(1437, 2156, 1600);
    CHECK(p.h == 1599 && tier_ratio(p, {1437, 2156}) == 1599.0 / 2156.0);
}

static double pct(const session::ProgressRecord& r) { return r.snapshot(nullptr).find("pct")->as_number(); }

static void test_progress() {
    session::ProgressRecord r;
    r.reset("p1", 4, "x");
    CHECK(pct(r) == 0.0);
    r.node("a.b");
    r.node("a.c");
    CHECK(pct(r) == 50.0);
    r.finish(true, std::nullopt);
    CHECK(pct(r) == 100.0);
    r.reset("p2", 4, "x");
    CHECK(pct(r) == 0.0);
}

// The GPU downscale equals the host reference bit for bit, on random data including
// negatives, for factors that use one axis, both axes, and an axis of length 1.
static void test_downscale(gpu::Context& g) {
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> u(-0.5f, 4.0f);
    struct Case {
        uint32_t w, h, L;
    } cases[] = {{1601, 1000, 1600}, {2000, 1333, 1600}, {4096, 2731, 1600}, {4096, 2731, 3400},
                 {300, 1, 100}, {1, 300, 100}, {37, 2000, 64}, {2156, 1437, 1600}};
    core::RenderContext ctx{.gpu = g};
    for (const Case& c : cases) {
        gpu::Plane src = g.persistent_plane(c.w, c.h, gpu::MemTag::Other);
        for (size_t i = 0; i < src.floats(); ++i) src.data()[i] = u(rng);
        gpu::Plane out = session::downscale(ctx, src, c.L);
        session::Size o = session::downscale_size(c.w, c.h, c.L);
        CHECK(out.w == o.w && out.h == o.h);
        std::vector<float> ref(size_t(3) * o.w * o.h);
        session::downscale_cpu(src.data(), c.w, c.h, ref.data(), o.w, o.h);
        CHECK(std::memcmp(ref.data(), out.data(), ref.size() * sizeof(float)) == 0);
    }
    // A flat field stays flat (weights sum to 1 in float64; float32 rounding only).
    gpu::Plane f = g.persistent_plane(1700, 900, gpu::MemTag::Other);
    for (size_t i = 0; i < f.floats(); ++i) f.data()[i] = 0.18f;
    gpu::Plane o = session::downscale(ctx, f, 1600);
    float lo = 1, hi = 0;
    for (size_t i = 0; i < o.floats(); ++i) lo = std::min(lo, o.data()[i]), hi = std::max(hi, o.data()[i]);
    CHECK(std::fabs(lo - 0.18f) < 1e-6f && std::fabs(hi - 0.18f) < 1e-6f);
    // No downscale: the plane itself.
    gpu::Plane same = session::downscale(ctx, f, 2000);
    CHECK(same.buf.contents() == f.buf.contents());
}

int main(int argc, char** argv) {
    test_tier_sizes();
    test_progress();
    if (argc >= 2) {
        auto g = gpu::Context::create(nullptr, std::filesystem::path(argv[1]) / "spektralab.metallib");
        test_downscale(*g);
    }
    return finish("test_session");
}
