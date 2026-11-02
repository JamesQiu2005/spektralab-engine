// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 Guangzhou Songzi Intelligent Technology Services Co., Ltd.
//
// GPU bridge: context, pool accounting, the math-mode probe, GPU Philox == host Philox, output
// texture layout. Usage: test_gpu <dir with the metallib>
#include <cstring>
#include <filesystem>

#include "check.hpp"
#include "fx/rng.hpp"
#include "gpu/gpu.hpp"

using namespace spk;

int main(int argc, char** argv) {
    if (argc < 2) return 2;
    auto ctx = gpu::Context::create(nullptr, std::filesystem::path(argv[1]) / "spektralab.metallib");
    std::printf("device: %s, max 2-D texture %u, linear alignment %zu B\n", ctx->device_name().c_str(),
                ctx->max_texture_dimension_2d(), ctx->linear_texture_alignment_rgba16());
    gpu::ProbeResult p = ctx->probe_math_mode();
    std::printf("probe: %s\n", p.verdict.c_str());
    CHECK(p.ok && p.verdict == gpu::Context::kSafeVerdict);
    CHECK(ctx->pool().stats().total_bytes == 0);  // the probe trims its scratch

    // pool accounting
    {
        gpu::Plane a = ctx->pool().plane(100, 10);
        CHECK(a.bytes() == 12000 && a.buf.size() >= 12000);
        gpu::PoolStats s = ctx->pool().stats();
        CHECK(s.live_bytes == a.buf.size() && s.live_buffers == 1 && s.free_bytes == 0);
    }
    gpu::PoolStats s = ctx->pool().stats();
    CHECK(s.live_bytes == 0 && s.free_buffers == 1 && s.total_bytes == s.free_bytes);
    {
        gpu::Buffer b = ctx->pool().acquire(11000);  // reuses the 12000-byte buffer
        CHECK(ctx->pool().stats().reuses == 1);
    }
    CHECK(ctx->pool().trim() == 12000 && ctx->pool().stats().total_bytes == 0);
    CHECK(ctx->pool().stats().live_underflows == 0);
    {
        gpu::Buffer pb = ctx->persistent(4096, gpu::MemTag::Negative);
        CHECK(ctx->persistent_stats().bytes == 4096);
    }
    CHECK(ctx->persistent_stats().bytes == 0);

    // Philox on the GPU equals the host implementation
    const uint32_t n = 1024;
    gpu::Buffer ctr = ctx->pool().acquire(n * 16), key = ctx->pool().acquire(n * 8), out = ctx->pool().acquire(n * 16);
    uint32_t* c = ctr.as<uint32_t>();
    uint32_t* k = key.as<uint32_t>();
    for (uint32_t i = 0; i < n; ++i) {
        c[4 * i] = i * 2654435761u;
        c[4 * i + 1] = i % 7 == 0 ? 200 : i % 23;
        c[4 * i + 2] = i / 3;
        c[4 * i + 3] = 0;
        k[2 * i] = i == 1 ? 0x5EED : 0x4abea221u ^ i;
        k[2 * i + 1] = 0;
    }
    c[0] = c[1] = c[2] = c[3] = 0;
    k[0] = 0;
    c[4] = 0, c[5] = 200, c[6] = 0, c[7] = 0;
    gpu::Batch bt = ctx->batch();
    bt.dispatch_1d(ctx->kernel("spk_probe_philox"), n, {gpu::Arg::buffer(ctr), gpu::Arg::buffer(key), gpu::Arg::buffer(out), gpu::Arg::value(n)});
    bt.commit_and_wait();
    const uint32_t* o = out.as<uint32_t>();
    int mismatches = 0;
    for (uint32_t i = 0; i < n; ++i) {
        auto h = fx::philox4x32_10({c[4 * i], c[4 * i + 1], c[4 * i + 2], c[4 * i + 3]}, {k[2 * i], k[2 * i + 1]});
        for (int j = 0; j < 4; ++j) mismatches += h[size_t(j)] != o[4 * i + uint32_t(j)];
    }
    CHECK(mismatches == 0);
    CHECK(o[0] == 0x6627e8d5u && o[4] == 0x15cecb3fu);

    // output texture over a fresh linear buffer
    gpu::OutputImage img = ctx->make_output(1001, 3);
    const size_t a = std::max<size_t>(ctx->linear_texture_alignment_rgba16(), 8) / 8;
    CHECK(img.row_stride_px == (1001 + a - 1) / a * a && img.row_stride_px >= 1001);
    CHECK(img.texture && img.rgba16);
    img.rgba16[0] = 65535;
    gpu::OutputImage img2 = ctx->make_output(1001, 3);
    CHECK(img2.rgba16 != img.rgba16);
    gpu::Context::release_texture(img.texture);
    gpu::Context::release_texture(img2.texture);

    // a missing kernel is a GPU error
    CHECK_THROWS(ctx->kernel("spk_no_such_kernel"));
    return finish("test_gpu");
}
