// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 Guangzhou Songzi Intelligent Technology Services Co., Ltd.

/* The parameter schema, the flat parameter values, and delta validation.
 *
 * Schema    48 rows: name, path, type, range, default, layer, live
 * ParamSet  one current value per row
 * Delta     a validated set of (row, value)
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "util/json.hpp"

namespace spk::params {

enum class Type : uint8_t { Float, Int, Bool, Str };
enum class Layer : uint8_t { Shoot, Print };

const char* type_name(Type t);    // "float", "int", "bool", "str"
const char* layer_name(Layer l);  // "shoot", "print"

// One schema row.
struct FieldDecl {
    std::string name;
    std::string path;  // dotted, informational
    Type type = Type::Float;
    Layer layer = Layer::Shoot;
    bool live = false;
    std::optional<std::pair<double, double>> range;  // closed interval
    json::Value def;                                 // un-digested default
    // Enumerated-string validator: returns an error message ("'<name>' = '<v>' is not ...") or
    // nullopt when valid. Empty function: any string.
    std::function<std::optional<std::string>(std::string_view value)> check_enum;
};

// Field ids, in schema listing order.
enum class P : uint16_t {
    film_stock, print_stock,
    exposure_compensation_ev, auto_exposure, auto_exposure_method, film_format_mm, lens_blur_um,
    halation_active, halation_amount, halation_boost_ev,
    grain_active, grain_sublayers_active,
    dir_couplers_active, dir_couplers_amount, density_curve_gamma,
    input_color_space, input_cctf_decoding,
    geometry_crop_x, geometry_crop_y, geometry_crop_w, geometry_crop_h, geometry_rotation_deg,
    geometry_quarter_turns, geometry_flip_h, geometry_flip_v,
    print_exposure, m_filter_shift, y_filter_shift, c_filter_neutral, m_filter_neutral, y_filter_neutral,
    preflash_exposure, enlarger_illuminant,
    glare_active, scanner_white_correction, scanner_black_correction, scanner_lens_blur,
    output_color_space, output_cctf_encoding, scan_film, extended_dynamic_range,
    halation_scatter_amount, grain_amount, glare_amount, antihalation_removed, filter_shift_scale,
    halation_headroom_ev, preview_long_edge,
    kCoreCount
};
inline constexpr size_t kCoreFieldCount = size_t(P::kCoreCount);  // 48
inline constexpr size_t id(P p) { return size_t(p); }

// The seven auto-exposure method names.
const std::vector<std::string_view>& auto_exposure_methods();

class Schema {
public:
    // The 48 rows, in listing order.
    static const std::vector<FieldDecl>& core_fields();
    static std::shared_ptr<const Schema> build();

    size_t size() const { return fields_.size(); }
    const FieldDecl& field(size_t id) const { return fields_[id]; }
    std::optional<size_t> find(std::string_view name) const;
    // {"fields":[...],"schema_version":1}, computed once.
    const std::string& json_text() const { return json_; }

private:
    std::vector<FieldDecl> fields_;
    std::string json_;
};

// The current value of every field (flat). Values are stored as json::Value of the field's type;
// ints are whole numbers (stored as double).
class ParamSet {
public:
    explicit ParamSet(std::shared_ptr<const Schema> schema);  // schema defaults
    const Schema& schema() const { return *schema_; }
    std::shared_ptr<const Schema> schema_ptr() const { return schema_; }

    double f(P p) const { return values_[id(p)].as_number(); }
    int64_t i(P p) const { return int64_t(values_[id(p)].as_number()); }
    bool b(P p) const { return values_[id(p)].as_bool(); }
    const std::string& s(P p) const { return values_[id(p)].as_string(); }

    const json::Value& get(size_t field_id) const { return values_[field_id]; }
    // No validation: callers pass already-validated values (ints already rounded).
    void set(size_t field_id, json::Value v) { values_[field_id] = std::move(v); }
    void set(P p, json::Value v) { set(id(p), std::move(v)); }

    // The flat params object: one key per schema row.
    json::Value to_json() const;

    bool operator==(const ParamSet& o) const { return values_ == o.values_; }

private:
    std::shared_ptr<const Schema> schema_;
    std::vector<json::Value> values_;
};

// A validated delta.
struct Delta {
    std::vector<std::pair<size_t, json::Value>> fields;  // ascending bytewise name order; ints rounded
    bool has_stock = false;                // film_stock or print_stock present
    bool has_film_stock = false;
    bool has_print_stock = false;
    bool any_shoot = false;                // -> invalidated = "shoot"
    bool any_rebuild = false;              // a non-live field or a stock -> pipeline rebuild
    bool has_antihalation_removed = false; // re-resolve the halation preset
    bool has_preview_long_edge = false;
    std::optional<bool> product_defaults;  // open delta only
    bool empty() const { return fields.empty(); }
};

enum class DeltaContext { Open, SetParams };

/* validate_delta(schema, text, ctx) -> Delta
 *   ctx == Open and text == "": empty delta
 *   parse text; top level must be an object
 *   ctx == Open: remove "product_defaults", must be a bool
 *   for each key in ascending bytewise order; first failure throws SPK_ERR_USER:
 *     name in schema
 *     JSON type matches the row
 *     enumeration of the row, if any
 *     range of the row, NaN refused
 *     int rows: round half away from zero, after the range check
 *   set has_stock, any_shoot, any_rebuild, ...
 */
Delta validate_delta(const Schema& schema, std::string_view text, DeltaContext ctx);

// Applies a validated delta's values onto `p` (no digest, no invalidation).
void apply_delta(ParamSet& p, const Delta& d);

}  // namespace spk::params
