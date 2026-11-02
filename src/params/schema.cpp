// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 Guangzhou Songzi Intelligent Technology Services Co., Ltd.

/* The 48 schema rows, in listing order, and their JSON form.
 *
 * Schema::build()
 *   json := {"fields":[{name,path,type,layer,default,live,range?}...],"schema_version":1}
 */
#include "params/params.hpp"
#include "util/error.hpp"

namespace spk::params {

const char* type_name(Type t) {
    switch (t) {
        case Type::Float: return "float";
        case Type::Int: return "int";
        case Type::Bool: return "bool";
        case Type::Str: return "str";
    }
    return "?";
}

const char* layer_name(Layer l) { return l == Layer::Shoot ? "shoot" : "print"; }

const std::vector<std::string_view>& auto_exposure_methods() {
    static const std::vector<std::string_view> m = {"center_weighted", "average", "median", "balanced",
                                                     "center", "protect_highlights", "protect_shadows"};
    return m;
}

namespace {

using R = std::optional<std::pair<double, double>>;
constexpr auto S = Layer::Shoot;
constexpr auto Pr = Layer::Print;

FieldDecl row(const char* name, const char* path, Type t, R range, json::Value def, Layer l,
              bool live = false) {
    FieldDecl f;
    f.name = name;
    f.path = path;
    f.type = t;
    f.range = range;
    f.def = std::move(def);
    f.layer = l;
    f.live = live;
    return f;
}

std::vector<FieldDecl> make_core() {
    const R none = std::nullopt;
    auto rg = [](double lo, double hi) { return R(std::make_pair(lo, hi)); };
    std::vector<FieldDecl> v = {
        row("film_stock", "film.info.stock", Type::Str, none, "kodak_portra_400", S),
        row("print_stock", "print.info.stock", Type::Str, none, "kodak_portra_endura", Pr),
        row("exposure_compensation_ev", "camera.exposure_compensation_ev", Type::Float, rg(-8, 8), 0, S),
        row("auto_exposure", "camera.auto_exposure", Type::Bool, none, true, S),
        row("auto_exposure_method", "camera.auto_exposure_method", Type::Str, none, "center_weighted", S),
        row("film_format_mm", "camera.film_format_mm", Type::Float, rg(4, 200), 35, S),
        row("lens_blur_um", "camera.lens_blur_um", Type::Float, rg(0, 200), 0, S),
        row("halation_active", "film_render.halation.active", Type::Bool, none, true, S),
        row("halation_amount", "film_render.halation.halation_amount", Type::Float, rg(0, 4), 1, S),
        row("halation_boost_ev", "film_render.halation.boost_ev", Type::Float, rg(0, 8), 0, S),
        row("grain_active", "film_render.grain.active", Type::Bool, none, true, S),
        row("grain_sublayers_active", "film_render.grain.sublayers_active", Type::Bool, none, true, S),
        row("dir_couplers_active", "film_render.dir_couplers.active", Type::Bool, none, true, S),
        row("dir_couplers_amount", "film_render.dir_couplers.amount", Type::Float, rg(0, 4), 1, S),
        row("density_curve_gamma", "film_render.density_curve_gamma", Type::Float, rg(0.2, 4), 1, S),
        row("input_color_space", "io.input_color_space", Type::Str, none, "ProPhoto RGB", S),
        row("input_cctf_decoding", "io.input_cctf_decoding", Type::Bool, none, false, S),
        row("geometry_crop_x", "io.geometry.crop_x", Type::Float, rg(0, 1), 0, S),
        row("geometry_crop_y", "io.geometry.crop_y", Type::Float, rg(0, 1), 0, S),
        row("geometry_crop_w", "io.geometry.crop_w", Type::Float, rg(1e-4, 1), 1, S),
        row("geometry_crop_h", "io.geometry.crop_h", Type::Float, rg(1e-4, 1), 1, S),
        row("geometry_rotation_deg", "io.geometry.rotation_deg", Type::Float, rg(-45, 45), 0, S),
        row("geometry_quarter_turns", "io.geometry.quarter_turns", Type::Int, rg(0, 3), 0, S),
        row("geometry_flip_h", "io.geometry.flip_h", Type::Bool, none, false, S),
        row("geometry_flip_v", "io.geometry.flip_v", Type::Bool, none, false, S),
        row("print_exposure", "enlarger.print_exposure", Type::Float, rg(0.05, 20), 1, Pr, true),
        row("m_filter_shift", "enlarger.m_filter_shift", Type::Float, rg(-1, 1), 0, Pr, true),
        row("y_filter_shift", "enlarger.y_filter_shift", Type::Float, rg(-1, 1), 0, Pr, true),
        row("c_filter_neutral", "enlarger.c_filter_neutral", Type::Float, rg(0, 200), 0, Pr),
        row("m_filter_neutral", "enlarger.m_filter_neutral", Type::Float, rg(0, 200), 65, Pr),
        row("y_filter_neutral", "enlarger.y_filter_neutral", Type::Float, rg(0, 200), 55, Pr),
        row("preflash_exposure", "enlarger.preflash_exposure", Type::Float, rg(0, 1), 0, Pr, true),
        row("enlarger_illuminant", "enlarger.illuminant", Type::Str, none, "TH-KG3", Pr),
        row("glare_active", "print_render.glare.active", Type::Bool, none, true, Pr),
        row("scanner_white_correction", "scanner.white_correction", Type::Bool, none, false, Pr),
        row("scanner_black_correction", "scanner.black_correction", Type::Bool, none, false, Pr),
        row("scanner_lens_blur", "scanner.lens_blur", Type::Float, rg(0, 20), 0, Pr),
        row("output_color_space", "io.output_color_space", Type::Str, none, "ProPhoto RGB", Pr),
        row("output_cctf_encoding", "io.output_cctf_encoding", Type::Bool, none, true, Pr),
        row("scan_film", "io.scan_film", Type::Bool, none, false, Pr),
        row("extended_dynamic_range", "print_render.edr_enabled", Type::Bool, none, false, Pr),
        row("halation_scatter_amount", "film_render.halation.scatter_amount", Type::Float, rg(0, 1), 1, S),
        row("grain_amount", "film_render.grain.amount", Type::Float, rg(0, 2), 1, S),
        row("glare_amount", "print_render.glare.amount", Type::Float, rg(0, 30), 1, Pr),
        row("antihalation_removed", "film_render.halation.antihalation_removed", Type::Bool, none, false, S),
        row("filter_shift_scale", "enlarger.filter_shift_scale", Type::Float, rg(1, 60), 1, Pr, true),
        row("halation_headroom_ev", "film_render.halation.headroom_ev", Type::Float, rg(0, 8), 4, S),
        row("preview_long_edge", "io.preview_long_edge", Type::Int, rg(800, 8192), 1600, Pr),
    };
    // auto_exposure_method's enumeration.
    v[size_t(P::auto_exposure_method)].check_enum = [](std::string_view s) -> std::optional<std::string> {
        for (std::string_view m : auto_exposure_methods())
            if (m == s) return std::nullopt;
        std::string list;
        for (std::string_view m : auto_exposure_methods()) list += (list.empty() ? "" : ", ") + std::string(m);
        return "'auto_exposure_method' = '" + std::string(s) + "' is not one of " + list;
    };
    return v;
}

}  // namespace

const std::vector<FieldDecl>& Schema::core_fields() {
    static const std::vector<FieldDecl> v = make_core();
    return v;
}

std::shared_ptr<const Schema> Schema::build() {
    auto s = std::make_shared<Schema>();
    s->fields_ = core_fields();
    json::Array rows;
    for (const FieldDecl& f : s->fields_) {
        json::Value r;
        r["name"] = f.name;
        r["path"] = f.path;
        r["type"] = type_name(f.type);
        r["layer"] = layer_name(f.layer);
        r["default"] = f.def;
        r["live"] = f.live;
        if (f.range) r["range"] = json::Array{f.range->first, f.range->second};
        rows.push_back(std::move(r));
    }
    json::Value doc;
    doc["schema_version"] = 1;
    doc["fields"] = std::move(rows);
    s->json_ = json::write(doc);
    return s;
}

std::optional<size_t> Schema::find(std::string_view name) const {
    for (size_t i = 0; i < fields_.size(); ++i)
        if (fields_[i].name == name) return i;
    return std::nullopt;
}

ParamSet::ParamSet(std::shared_ptr<const Schema> schema) : schema_(std::move(schema)) {
    values_.reserve(schema_->size());
    for (size_t i = 0; i < schema_->size(); ++i) values_.push_back(schema_->field(i).def);
}

json::Value ParamSet::to_json() const {
    json::Value o = json::Object{};
    for (size_t i = 0; i < values_.size(); ++i) o[schema_->field(i).name] = values_[i];
    return o;
}

}  // namespace spk::params
