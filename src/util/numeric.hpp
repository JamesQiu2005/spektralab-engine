// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 Guangzhou Songzi Intelligent Technology Services Co., Ltd.

/* Host-side numeric helpers with exactly defined behaviour.
 *
 * float64 unless named f32
 * never compiled with fast math or FP contraction: the Makefile passes -ffp-contract=off
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace spk::num {

// Round a double to IEEE binary32 (round-to-nearest-even): written f32(.) in comments.
inline float f32(double x) { return static_cast<float>(x); }

// Round half away from zero (integer parameters; megapixels).
double round_half_away(double x);

/* search_right(x, v)
 *   lo := 0; hi := K
 *   while lo < hi: mid := lo + (hi - lo) / 2; if x[mid] <= v: lo := mid + 1 else hi := mid
 *   return lo   -- entries <= v on a sorted axis; this same bisection on a non-monotone one; NaN v -> 0
 */
size_t search_right(std::span<const double> x, double v);
size_t search_right_f32(std::span<const float> x, float v);

/* interp(v, x, y): clamped piecewise-linear; K >= 2, equal sizes
 *   NaN v -> NaN
 *   v <= x[0] -> y[0]; v >= x[K-1] -> y[K-1]
 *   j := search_right(x, v) - 1
 *   x[j+1] == x[j] -> y[j]
 *   y[j] + (v - x[j]) / (x[j+1] - x[j]) * (y[j+1] - y[j])
 */
double interp(double v, std::span<const double> x, std::span<const double> y);

// NaN-skipping reductions. All-NaN (or empty) input returns NaN.
double nanmin(std::span<const double> v);
double nanmax(std::span<const double> v);
double nanmean(std::span<const double> v);
// Strided variants for a column of a row-major (n x stride) matrix.
double nanmin_strided(const double* p, size_t n, size_t stride);
double nanmax_strided(const double* p, size_t n, size_t stride);

/* gaussian_kernel(sigma, truncate), float64
 *   r := floor(truncate sigma + 0.5)
 *   sigma <= 0 or r == 0 -> {1.0}
 *   w_k := exp(-0.5 (k / sigma)^2), k = -r..r; divide by their sum, accumulated in ascending k
 */
int gaussian_radius(double sigma, double truncate);
std::vector<double> gaussian_kernel(double sigma, double truncate);
// The same taps rounded once to float.
std::vector<float> gaussian_kernel_f32(double sigma, double truncate);

// IEEE binary16 -> double, exact for every value including subnormals, infinities and NaN.
double half_to_double(uint16_t h);

/* mirror_index: tier downscale fold, "c b | a b c d | c b"; period 2(n-1); n == 1 -> 0 */
int64_t mirror_index(int64_t i, int64_t n);
/* reflect_index: FIR blur fold, "d c b a | a b c d"; period 2n */
int64_t reflect_index(int64_t i, int64_t n);

}  // namespace spk::num
