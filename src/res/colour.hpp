// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 Guangzhou Songzi Intelligent Technology Services Co., Ltd.

/* Named colour data from the constants blob: the eight baked colour spaces with their transfer modes,
 * CAT matrices, illuminant SPDs by name. Pure float64 set-up helpers.
 */
#pragma once

#include <array>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "res/blob.hpp"

namespace spk::res {

using Spectrum = std::array<double, 81>;
using Mat3 = std::array<double, 9>;  // row-major, out_i = sum_j m[3i+j] x_j

// Transfer-function modes: 0 sRGB curve, 1 ProPhoto, 2 Adobe RGB,
// 3 BT.709 curve, 4 identity.
enum class CctfMode : int { SRGB = 0, ProPhoto = 1, Adobe = 2, BT709 = 3, Identity = 4 };

struct ColourSpace {
    std::string name;
    CctfMode cctf = CctfMode::Identity;
    std::array<double, 6> primaries{};  // xy of R, G, B
    std::array<double, 2> whitepoint{};
    Mat3 rgb_to_xyz{};  // stored matrices: use these, never re-derive
    Mat3 xyz_to_rgb{};
};

// The eight baked space names, in catalogue order.
std::span<const std::string_view> baked_colour_space_names();
// Transfer mode of a baked name; nullopt for an unknown name.
std::optional<CctfMode> cctf_mode_of(std::string_view name);
// Loads the space from the blob; nullopt if the name is not baked or the blob lacks its entries.
std::optional<ColourSpace> colour_space(const Blob& blob, std::string_view name);

// colour/cat/<name> (CAT02, CAT16, Bradford, Von Kries, XYZ Scaling); nullopt if absent.
std::optional<Mat3> cat_matrix(const Blob& blob, std::string_view name);

/* illuminant(blob, name) -> SPD / mean(SPD) over the 81 samples; nullopt when unknown, never a fallback
 *   BB<kelvin>   blackbody(kelvin)
 *   T            colour/sd_light_source/Incandescent
 *   K75P         colour/sd_light_source/Kinoton 75P
 *   TH-KG3       blackbody(3400) * filters/kg3
 *   TH-KG3-L     blackbody(3400) * filters/kg3 * filters/lens_canon
 *   otherwise    colour/sd_illuminant/<name>, then colour/sd_light_source/<name>
 */
std::optional<Spectrum> illuminant(const Blob& blob, std::string_view name);
bool is_illuminant_name(const Blob& blob, std::string_view name);

/* blackbody(T): s(l) = (c1 / (pi n^2 l^5)) / expm1(c2 / (n l T)), l in metres
 *   c1 = 3.741771e-16, c2 = 1.4388e-2, n = 1; unnormalised
 */
Spectrum blackbody(double kelvin);

// xy chromaticity of an SPD with the baked CIE 1931 2-degree CMFs.
std::array<double, 2> chromaticity(const Blob& blob, const Spectrum& s);

// ---- float64 3x3 helpers shared by film and print ----

// Ordinary 3x3 product, each entry summed over k = 0, 1, 2 in that order.
Mat3 mat3_mul(const Mat3& a, const Mat3& b);
std::array<double, 3> mat3_apply(const Mat3& m, const std::array<double, 3>& v);
// Closed-form adjugate inverse with one division r = 1/det.
Mat3 mat3_inverse(const Mat3& m);
// XYZ(x, y) = (x/y, 1, (1 - x - y)/y).
std::array<double, 3> xy_to_XYZ(const std::array<double, 2>& xy);
// The "allclose" guard: true when |a - b| <= 1e-8 + 1e-5 |b| for both coordinates (b = reference).
bool xy_allclose(const std::array<double, 2>& a, const std::array<double, 2>& b);
/* von_kries(xy_source, xy_target, M_A, b)
 *   if xy_allclose(the other, b): nullopt                -- the caller skips the product
 *   rho_s := M_A XYZ(xy_source); rho_t := M_A XYZ(xy_target)
 *   A := (M_A^-1 diag(rho_t / rho_s)) M_A, left to right
 *   b = the colour space's whitepoint: Source on the film side, Target on the print side
 */
enum class GuardRef { Source, Target };
std::optional<Mat3> von_kries(const std::array<double, 2>& xy_source, const std::array<double, 2>& xy_target,
                              const Mat3& cone, GuardRef b);

}  // namespace spk::res
