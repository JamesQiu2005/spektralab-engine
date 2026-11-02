// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 Guangzhou Songzi Intelligent Technology Services Co., Ltd.

/* Film / paper profile JSON loader; format in docs/RESOURCE_FORMAT.md.
 *
 * load_profile(resources_dir, id)
 *   check id: [A-Za-z0-9_-]{1,96}
 *   doc := parse <resources>/profiles/<id>.json
 *   if doc.derived_from is a JSON string:                   -- an alias, one level only
 *     base := parse profiles/<base>.json; refuse if it is itself an alias
 *     parse base info and data
 *     lay the alias's info keys over, key by key
 *     ignore the alias's data and metadata
 *     stock := alias id; base_id := base id
 *   else: parse info and data
 *   every shape mismatch is a load error; JSON null in a numeric array reads as NaN
 *   type must be "negative" or "positive" (checked again after an alias overlay)
 *   computed: dmin_c := nanmin_k D_c[k]; N_c[k] := D_c[k] - dmin_c; dmax_c := nanmax_k N_c[k]
 *   all failures: spk::Error(SPK_ERR_USER, message naming the id)
 */
#pragma once

#include <array>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace spk::res {

constexpr size_t kSpectralSamples = 81;  // 380..780 nm, 5 nm

// Row-major matrix of doubles.
struct Mat {
    size_t rows = 0, cols = 0;
    std::vector<double> v;
    double operator()(size_t r, size_t c) const { return v[r * cols + c]; }
    double& operator()(size_t r, size_t c) { return v[r * cols + c]; }
    bool empty() const { return v.empty(); }
};

struct ProfileInfo {
    std::string stock;  // forced to the requested (alias) id
    std::string name;   // default = stock
    std::string type = "negative";
    std::string support = "film";
    std::string stage = "filming";
    std::string use = "still";
    std::string antihalation = "weak";
    std::string target_print;  // "" when absent or null
    std::string channel_model = "color";  // read; no behaviour may depend on it
    std::string densitometer = "status_M";
    double log_sensitivity_density_over_min = 0.2;
    std::string reference_illuminant = "D55";
    std::string viewing_illuminant = "D50";

    bool is_positive() const { return type == "positive"; }
    bool is_film() const { return support == "film"; }
};

struct DensityCurvesModel {
    std::string model_type = "cdfs";
    Mat centers, amplitudes, sigmas;  // equal shapes (channels x layers)
};

struct EdrToneMap {
    int version = 1;
    double domain_lo = 0, domain_hi = 0;  // domain_log2_y
    double toe_y = 0, shoulder_y = 0;
    std::vector<double> target_log2_y;  // N >= 2, all finite
    std::string calibration_film;       // optional
    std::string semantic;               // optional
};

struct Profile {
    std::string id;       // requested id (alias id for an alias)
    std::string base_id;  // "" unless an alias
    ProfileInfo info;

    std::vector<double> wavelengths;   // 81, exactly 380..780 step 5
    Mat log_sensitivity;               // 81 x 3, NaN allowed
    Mat channel_density;               // 81 x 3, NaN allowed
    std::vector<double> base_density;  // 81, NaN allowed
    std::optional<std::vector<double>> midscale_neutral_density;  // 81 if present (unused)
    std::vector<double> log_exposure;  // K >= 2
    Mat density_curves;                // K x 3
    // K x 3 x 3 flattened [k][sublayer][channel]: index (k*3 + l)*3 + c. Empty if absent.
    std::vector<double> density_curves_layers;
    std::optional<DensityCurvesModel> density_curves_model;
    std::vector<double> adaptation_window_params;  // 0, 4 or 8 values
    std::optional<Mat> adaptation_surface_params;  // 3 x n; absent or [] -> nullopt (read, not used)
    std::optional<EdrToneMap> edr_tone_map;        // papers only

    // Computed at load
    std::array<double, 3> dmin{};
    Mat normalized_curves;  // K x 3
    std::array<double, 3> dmax{};

    bool is_alias() const { return !base_id.empty(); }
    bool has_layers() const { return !density_curves_layers.empty(); }
    size_t curve_points() const { return log_exposure.size(); }
};

// True for [A-Za-z0-9_-]{1,96}.
bool valid_stock_id(std::string_view id);

std::shared_ptr<const Profile> load_profile(const std::filesystem::path& resources_dir,
                                            std::string_view id);

}  // namespace spk::res
