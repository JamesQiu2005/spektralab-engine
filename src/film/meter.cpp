// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 Guangzhou Songzi Intelligent Technology Services Co., Ltd.

/* The auto-exposure meter: float64 on the CPU over the shared plane of the meter image
 * (the raw 1600 px downscale taken through decode and geometry).
 */
#include <algorithm>
#include <cmath>
#include <vector>

#include "film/film_internal.hpp"

namespace spk::film {

using params::P;

double MeterValues::ev(std::string_view m) const {
    if (m == "center_weighted") return center_weighted;
    if (m == "average") return average;
    if (m == "median") return median;
    if (m == "balanced") return balanced;
    if (m == "center") return center;
    if (m == "protect_highlights") return protect_highlights;
    if (m == "protect_shadows") return protect_shadows;
    fail_internal("unknown auto-exposure method '" + std::string(m) + "'");
}

namespace {

constexpr double kMid = 0.184, kSigmaCw = 0.2, kFloor = 0.184 / 4096.0;

double legacy_ev(double e) {
    const double ev = -std::log2(e);
    return std::isfinite(ev) ? ev : 0.0;
}

double clampd(double v, double lo, double hi) { return std::min(std::max(v, lo), hi); }

// Ascending, NaN last (a strict weak order even when a decode produced NaN).
bool nan_last(double a, double b) { return a < b || (!std::isnan(a) && std::isnan(b)); }

}  // namespace

/* meter_values(rgb, sw, sh, rgb_to_xyz)
 *   Y[i] := row 1 of rgb_to_xyz . rgb[i]
 *   weight(x, y) := exp(-(x^2 + y^2) / (2 * 0.2^2))
 *     x := (col / sw - 1/2) sw / max(sw, sh); y := (row / sh - 1/2) sh / max(sw, sh)
 *   mid := 0.184; legacy(e) := -log2(e), 0 if not finite
 *   legacy meters
 *     average          := legacy(mean(Y) / mid)
 *     median           := legacy(median(Y) / mid)
 *     center_weighted  := legacy(weighted mean(Y) / mid)
 *   intent meters, on y := max(Y, mid / 4096)
 *     P(q) := sorted(y)[q (n - 1) / 1000], integer division
 *     in range := P(10) <= y <= P(990)
 *     balanced := -log2(exp(mean(ln y over range)) / mid)
 *     center   := -log2(exp(weighted mean(ln y over range)) / mid)
 *     protect_highlights := clamp(log2(mid 2^2.5 / P(995)), balanced - 3, balanced)
 *     protect_shadows    := clamp(log2(mid 2^-3.5 / P(50)), balanced, balanced + 2)
 */
MeterValues meter_values(const float* rgb, uint32_t sw, uint32_t sh, const res::Mat3& rgb_to_xyz) {
    MeterValues mv;
    const size_t n = size_t(sw) * sh;
    if (n == 0) return mv;
    const double m0 = rgb_to_xyz[3], m1 = rgb_to_xyz[4], m2 = rgb_to_xyz[5];
    std::vector<double> Y(n);
    for (size_t i = 0; i < n; ++i)
        Y[i] = (m0 * double(rgb[3 * i]) + m1 * double(rgb[3 * i + 1])) + m2 * double(rgb[3 * i + 2]);

    const double M = double(std::max(sw, sh));
    std::vector<double> wx(sw), wy(sh);
    for (uint32_t j = 0; j < sw; ++j) wx[j] = (double(j) / double(sw) - 0.5) * (double(sw) / M);
    for (uint32_t i = 0; i < sh; ++i) wy[i] = (double(i) / double(sh) - 0.5) * (double(sh) / M);
    const double two_s2 = 2.0 * (kSigmaCw * kSigmaCw);
    auto weight = [&](size_t k) {
        const double x = wx[k % sw], y = wy[k / sw];
        return std::exp(-(x * x + y * y) / two_s2);
    };

    // legacy meters
    {
        double s = 0.0;
        for (double y : Y) s += y;
        mv.average = legacy_ev((s / double(n)) / kMid);

        std::vector<double> sorted = Y;
        std::sort(sorted.begin(), sorted.end(), nan_last);
        const double med = (n % 2) ? sorted[n / 2] : (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0;
        mv.median = legacy_ev(med / kMid);

        double sw_ = 0.0, swy = 0.0;
        for (size_t k = 0; k < n; ++k) {
            const double w = weight(k);
            swy += w * Y[k];
            sw_ += w;
        }
        mv.center_weighted = legacy_ev((swy / sw_) / kMid);
    }

    // intent meters
    {
        std::vector<double> y(n);
        for (size_t k = 0; k < n; ++k) y[k] = std::max(Y[k], kFloor);
        std::vector<double> sorted = y;
        std::sort(sorted.begin(), sorted.end(), nan_last);
        auto P_ = [&](uint64_t q) { return sorted[size_t(q * uint64_t(n - 1) / 1000u)]; };
        const double p1 = P_(10), p5 = P_(50), p99 = P_(990), p995 = P_(995);
        double sl = 0.0, swl = 0.0, sw_ = 0.0;
        size_t cnt = 0;
        for (size_t k = 0; k < n; ++k) {
            if (!(y[k] >= p1 && y[k] <= p99)) continue;
            const double l = std::log(y[k]);
            const double w = weight(k);
            sl += l;
            swl += w * l;
            sw_ += w;
            ++cnt;
        }
        mv.balanced = -std::log2(std::exp(sl / double(cnt)) / kMid);
        mv.center = -std::log2(std::exp(swl / sw_) / kMid);
        mv.protect_highlights =
            clampd(std::log2(kMid * std::pow(2.0, 2.5) / p995), mv.balanced - 3.0, mv.balanced);
        mv.protect_shadows = clampd(std::log2(kMid * std::pow(2.0, -3.5) / p5), mv.balanced, mv.balanced + 2.0);
    }
    return mv;
}

MeterValues meter(const core::RenderContext& ctx, const params::ParamSet& p, const res::Resources& res,
                  const gpu::Plane& meter_source) {
    const std::string& space = p.s(P::input_color_space);
    auto cs = res::colour_space(*res.blob, space);
    if (!cs) fail_user("input_color_space '" + space + "' is not a colour space in the constants blob");
    gpu::Plane img = decode_and_geometry(ctx, p, res, meter_source);
    return meter_values(img.data(), img.w, img.h, cs->rgb_to_xyz);
}

}  // namespace spk::film
