// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 Guangzhou Songzi Intelligent Technology Services Co., Ltd.

/* Host-side numeric helpers; behaviour stated in numeric.hpp. */
#include "util/numeric.hpp"

#include <cmath>
#include <cstring>
#include <limits>

namespace spk::num {

namespace {
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
}

double round_half_away(double x) { return std::round(x); }

size_t search_right(std::span<const double> x, double v) {
    size_t lo = 0, hi = x.size();
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (x[mid] <= v) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

size_t search_right_f32(std::span<const float> x, float v) {
    size_t lo = 0, hi = x.size();
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (x[mid] <= v) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

double interp(double v, std::span<const double> x, std::span<const double> y) {
    const size_t k = x.size();
    if (std::isnan(v)) return kNaN;
    if (v <= x[0]) return y[0];
    if (v >= x[k - 1]) return y[k - 1];
    size_t lo = search_right(x, v);
    // v > x[0] is not guaranteed to give lo >= 1 on a non-monotone axis; clamp to a valid bracket.
    size_t j = lo == 0 ? 0 : lo - 1;
    if (j + 1 >= k) j = k - 2;
    double dx = x[j + 1] - x[j];
    if (dx == 0.0) return y[j];
    return y[j] + (v - x[j]) / dx * (y[j + 1] - y[j]);
}

double nanmin(std::span<const double> v) { return nanmin_strided(v.data(), v.size(), 1); }
double nanmax(std::span<const double> v) { return nanmax_strided(v.data(), v.size(), 1); }

double nanmin_strided(const double* p, size_t n, size_t stride) {
    double m = kNaN;
    for (size_t i = 0; i < n; ++i) {
        double x = p[i * stride];
        if (std::isnan(x)) continue;
        if (std::isnan(m) || x < m) m = x;
    }
    return m;
}

double nanmax_strided(const double* p, size_t n, size_t stride) {
    double m = kNaN;
    for (size_t i = 0; i < n; ++i) {
        double x = p[i * stride];
        if (std::isnan(x)) continue;
        if (std::isnan(m) || x > m) m = x;
    }
    return m;
}

double nanmean(std::span<const double> v) {
    double s = 0.0;
    size_t n = 0;
    for (double x : v) {
        if (std::isnan(x)) continue;
        s += x;
        ++n;
    }
    return n ? s / double(n) : kNaN;
}

int gaussian_radius(double sigma, double truncate) {
    if (!(sigma > 0.0)) return 0;
    return static_cast<int>(std::floor(truncate * sigma + 0.5));
}

std::vector<double> gaussian_kernel(double sigma, double truncate) {
    const int r = gaussian_radius(sigma, truncate);
    if (r == 0) return {1.0};
    std::vector<double> w(size_t(2 * r + 1));
    double sum = 0.0;
    for (int k = -r; k <= r; ++k) {
        double t = double(k) / sigma;
        double e = std::exp(-0.5 * t * t);
        w[size_t(k + r)] = e;
        sum += e;
    }
    for (double& e : w) e /= sum;
    return w;
}

std::vector<float> gaussian_kernel_f32(double sigma, double truncate) {
    std::vector<double> d = gaussian_kernel(sigma, truncate);
    std::vector<float> f(d.size());
    for (size_t i = 0; i < d.size(); ++i) f[i] = f32(d[i]);
    return f;
}

/* half_to_double(h): sign s, exponent e, mantissa m
 *   e == 0:  m 2^-24                    -- subnormal and zero
 *   e == 31: m ? NaN : infinity
 *   else:    (m | 0x400) 2^(e - 25)
 */
double half_to_double(uint16_t h) {
    const int sign = (h >> 15) & 1;
    const int exp = (h >> 10) & 0x1F;
    const int mant = h & 0x3FF;
    double v;
    if (exp == 0) {
        v = std::ldexp(double(mant), -24);  // subnormal (and zero): mant * 2^-24
    } else if (exp == 31) {
        v = mant ? std::numeric_limits<double>::quiet_NaN() : std::numeric_limits<double>::infinity();
    } else {
        v = std::ldexp(double(mant | 0x400), exp - 25);
    }
    return sign ? -v : v;
}

int64_t mirror_index(int64_t i, int64_t n) {
    if (n <= 1) return 0;
    const int64_t period = 2 * (n - 1);
    int64_t m = i % period;
    if (m < 0) m += period;
    return m < n ? m : period - m;
}

int64_t reflect_index(int64_t i, int64_t n) {
    if (n <= 1) return 0;
    const int64_t period = 2 * n;
    int64_t m = i % period;
    if (m < 0) m += period;
    return m < n ? m : period - 1 - m;
}

}  // namespace spk::num
