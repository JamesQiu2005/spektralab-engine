// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 Guangzhou Songzi Intelligent Technology Services Co., Ltd.

/* Named colour data from the constants blob; rules in colour.hpp. */
#include "res/colour.hpp"

#include <cmath>
#include <cstdlib>
#include <string>
#include <xlocale.h>

namespace spk::res {

namespace {

constexpr std::string_view kNames[] = {"sRGB",         "Display P3",    "ProPhoto RGB", "Adobe RGB (1998)",
                                       "ITU-R BT.709", "ITU-R BT.2020", "ACEScg",       "ACES2065-1"};
constexpr CctfMode kModes[] = {CctfMode::SRGB,  CctfMode::SRGB,  CctfMode::ProPhoto, CctfMode::Adobe,
                               CctfMode::BT709, CctfMode::BT709, CctfMode::Identity, CctfMode::Identity};

std::optional<Spectrum> blob_spectrum(const Blob& blob, const std::string& name) {
    const BlobEntry* e = blob.find(name);
    if (!e || !e->size_ok || e->ndim != 1 || e->dims[0] != 81) return std::nullopt;
    std::vector<double> v = blob.f64(name);
    Spectrum s{};
    for (size_t i = 0; i < 81; ++i) s[i] = v[i];
    return s;
}

template <size_t N>
bool blob_array(const Blob& blob, const std::string& name, std::array<double, N>& out) {
    const BlobEntry* e = blob.find(name);
    if (!e || !e->size_ok || e->element_count() != N) return false;
    std::vector<double> v = blob.f64(name);
    for (size_t i = 0; i < N; ++i) out[i] = v[i];
    return true;
}

Spectrum normalise_mean(Spectrum s) {
    double sum = 0.0;
    for (double x : s) sum += x;
    const double mean = sum / 81.0;
    for (double& x : s) x /= mean;
    return s;
}

/* raw_illuminant(blob, name) -> unnormalised SPD
 *   "BB" + number > 0: blackbody(number)
 *   T, K75P: named light sources
 *   TH-KG3[-L]: blackbody(3400) x filters/kg3 [x filters/lens_canon]
 *   else colour/sd_illuminant/<name>, then colour/sd_light_source/<name>
 */
std::optional<Spectrum> raw_illuminant(const Blob& blob, std::string_view name) {
    if (name.size() > 2 && name.substr(0, 2) == "BB") {
        static locale_t loc = newlocale(LC_ALL_MASK, "C", nullptr);
        std::string num(name.substr(2));
        char* end = nullptr;
        double t = strtod_l(num.c_str(), &end, loc);
        if (end == num.c_str() || !std::isfinite(t) || !(t > 0)) return std::nullopt;
        return blackbody(t);
    }
    if (name == "T") return blob_spectrum(blob, "colour/sd_light_source/Incandescent");
    if (name == "K75P") return blob_spectrum(blob, "colour/sd_light_source/Kinoton 75P");
    if (name == "TH-KG3" || name == "TH-KG3-L") {
        auto kg3 = blob_spectrum(blob, "filters/kg3");
        if (!kg3) return std::nullopt;
        Spectrum s = blackbody(3400.0);
        for (size_t i = 0; i < 81; ++i) s[i] *= (*kg3)[i];
        if (name == "TH-KG3-L") {
            auto lens = blob_spectrum(blob, "filters/lens_canon");
            if (!lens) return std::nullopt;
            for (size_t i = 0; i < 81; ++i) s[i] *= (*lens)[i];
        }
        return s;
    }
    if (auto s = blob_spectrum(blob, "colour/sd_illuminant/" + std::string(name))) return s;
    return blob_spectrum(blob, "colour/sd_light_source/" + std::string(name));
}

}  // namespace

std::span<const std::string_view> baked_colour_space_names() { return kNames; }

std::optional<CctfMode> cctf_mode_of(std::string_view name) {
    for (size_t i = 0; i < 8; ++i)
        if (kNames[i] == name) return kModes[i];
    return std::nullopt;
}

std::optional<ColourSpace> colour_space(const Blob& blob, std::string_view name) {
    auto mode = cctf_mode_of(name);
    if (!mode) return std::nullopt;
    ColourSpace cs;
    cs.name = std::string(name);
    cs.cctf = *mode;
    const std::string p = "colour/cs/" + cs.name + "/";
    if (!blob_array(blob, p + "primaries", cs.primaries) || !blob_array(blob, p + "whitepoint", cs.whitepoint) ||
        !blob_array(blob, p + "matrix_RGB_to_XYZ", cs.rgb_to_xyz) ||
        !blob_array(blob, p + "matrix_XYZ_to_RGB", cs.xyz_to_rgb))
        return std::nullopt;
    return cs;
}

std::optional<Mat3> cat_matrix(const Blob& blob, std::string_view name) {
    Mat3 m{};
    if (!blob_array(blob, "colour/cat/" + std::string(name), m)) return std::nullopt;
    return m;
}

Spectrum blackbody(double kelvin) {
    constexpr double c1 = 3.741771e-16, c2 = 1.4388e-2, n = 1.0;
    Spectrum s{};
    for (size_t i = 0; i < 81; ++i) {
        const double lam = (380.0 + 5.0 * double(i)) * 1e-9;
        s[i] = (c1 / (M_PI * n * n * std::pow(lam, 5))) / std::expm1(c2 / (n * lam * kelvin));
    }
    return s;
}

std::optional<Spectrum> illuminant(const Blob& blob, std::string_view name) {
    auto s = raw_illuminant(blob, name);
    if (!s) return std::nullopt;
    return normalise_mean(*s);
}

bool is_illuminant_name(const Blob& blob, std::string_view name) { return raw_illuminant(blob, name).has_value(); }

/* chromaticity(blob, s): (X, Y, Z) := sum_l s(l) cmf(l); (X, Y) / (X + Y + Z) */
std::array<double, 2> chromaticity(const Blob& blob, const Spectrum& s) {
    std::vector<double> cmf = blob.f64("colour/cmfs_1931_2deg");
    double X = 0, Y = 0, Z = 0;
    for (size_t i = 0; i < 81; ++i) {
        X += s[i] * cmf[3 * i];
        Y += s[i] * cmf[3 * i + 1];
        Z += s[i] * cmf[3 * i + 2];
    }
    const double t = X + Y + Z;
    return {X / t, Y / t};
}

Mat3 mat3_mul(const Mat3& a, const Mat3& b) {
    Mat3 r{};
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 3; ++j) {
            double s = 0.0;
            for (size_t k = 0; k < 3; ++k) s += a[3 * i + k] * b[3 * k + j];
            r[3 * i + j] = s;
        }
    return r;
}

std::array<double, 3> mat3_apply(const Mat3& m, const std::array<double, 3>& v) {
    std::array<double, 3> r{};
    for (size_t i = 0; i < 3; ++i) {
        double s = 0.0;
        for (size_t k = 0; k < 3; ++k) s += m[3 * i + k] * v[k];
        r[i] = s;
    }
    return r;
}

/* mat3_inverse(m): closed-form adjugate; r := 1 / det; every entry cofactor * r */
Mat3 mat3_inverse(const Mat3& m) {
    const double a = m[0], b = m[1], c = m[2], d = m[3], e = m[4], f = m[5], g = m[6], h = m[7], i = m[8];
    const double A0 = e * i - f * h;
    const double B0 = -(d * i - f * g);
    const double C0 = d * h - e * g;
    const double det = a * A0 + b * B0 + c * C0;
    const double r = 1.0 / det;
    return {A0 * r, -(b * i - c * h) * r, (b * f - c * e) * r,
            B0 * r, (a * i - c * g) * r,  -(a * f - c * d) * r,
            C0 * r, -(a * h - b * g) * r, (a * e - b * d) * r};
}

std::array<double, 3> xy_to_XYZ(const std::array<double, 2>& xy) {
    return {xy[0] / xy[1], 1.0, (1.0 - xy[0] - xy[1]) / xy[1]};
}

bool xy_allclose(const std::array<double, 2>& a, const std::array<double, 2>& b) {
    for (size_t k = 0; k < 2; ++k)
        if (!(std::fabs(a[k] - b[k]) <= 1e-8 + 1e-5 * std::fabs(b[k]))) return false;
    return true;
}

std::optional<Mat3> von_kries(const std::array<double, 2>& xy_source, const std::array<double, 2>& xy_target,
                              const Mat3& cone, GuardRef b) {
    const bool close = b == GuardRef::Source ? xy_allclose(xy_target, xy_source) : xy_allclose(xy_source, xy_target);
    if (close) return std::nullopt;
    const std::array<double, 3> rs = mat3_apply(cone, xy_to_XYZ(xy_source));
    const std::array<double, 3> rt = mat3_apply(cone, xy_to_XYZ(xy_target));
    const Mat3 inv = mat3_inverse(cone);
    Mat3 dg{};
    for (size_t k = 0; k < 3; ++k) dg[4 * k] = rt[k] / rs[k];
    return mat3_mul(mat3_mul(inv, dg), cone);
}

}  // namespace spk::res
