// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 Guangzhou Songzi Intelligent Technology Services Co., Ltd.

/* Delta parsing and validation; the steps are on validate_delta in params.hpp.
 *
 * Reader: the strict JSON reader, so a parse error is SPK_ERR_USER.
 * Key order: json::Object is a std::map over std::string, so iteration is bytewise.
 * Type rule: any number for float and int; true or false for bool; a string for str; null never.
 * Range message: numbers in %f form.
 */
#include <cmath>
#include <cstdio>
#include <string>

#include "params/params.hpp"
#include "util/error.hpp"
#include "util/numeric.hpp"

namespace spk::params {

namespace {

std::string fmt_f(double v) {
    char buf[64];
    std::snprintf(buf, sizeof buf, "%f", v);
    return buf;
}

bool type_matches(Type t, const json::Value& v) {
    switch (t) {
        case Type::Float:
        case Type::Int: return v.is_number();
        case Type::Bool: return v.is_bool();
        case Type::Str: return v.is_string();
    }
    return false;
}

}  // namespace

Delta validate_delta(const Schema& schema, std::string_view text, DeltaContext ctx) {
    Delta d;
    // Open: NULL or "" means no delta. spk_set_params: "" does not parse (the delta is required).
    if (text.empty() && ctx == DeltaContext::Open) return d;
    json::Value doc;
    try {
        doc = json::parse(text);
    } catch (const json::ParseError& e) {
        fail_user(e.what());
    }
    if (!doc.is_object()) fail_user("params_delta must be an object");
    json::Object obj = doc.as_object();

    if (ctx == DeltaContext::Open) {
        if (auto it = obj.find("product_defaults"); it != obj.end()) {
            if (!it->second.is_bool()) fail_user("'product_defaults' expects bool");
            d.product_defaults = it->second.as_bool();
            obj.erase(it);
        }
    }

    d.fields.reserve(obj.size());
    for (const auto& [name, value] : obj) {  // ascending bytewise order
        const std::optional<size_t> fid = schema.find(name);
        if (!fid) fail_user("unknown parameter '" + name + "'");
        const FieldDecl& f = schema.field(*fid);
        if (!type_matches(f.type, value)) fail_user("'" + name + "' expects " + type_name(f.type));

        json::Value stored = value;
        if (f.type == Type::Str && f.check_enum) {
            if (std::optional<std::string> msg = f.check_enum(value.as_string())) fail_user(*msg);
        }
        if (f.type == Type::Float || f.type == Type::Int) {
            const double v = value.as_number();
            if (f.range) {
                const auto [lo, hi] = *f.range;
                if (!(v >= lo && v <= hi))
                    fail_user("'" + name + "'=" + fmt_f(v) + " is outside [" + fmt_f(lo) + ", " + fmt_f(hi) + "]");
            } else if (std::isnan(v)) {
                fail_user("'" + name + "' expects " + type_name(f.type) + ", not NaN");
            }
            if (f.type == Type::Int) stored = json::Value(num::round_half_away(v));
        }

        d.fields.emplace_back(*fid, std::move(stored));
        if (name == "film_stock") d.has_film_stock = true;
        if (name == "print_stock") d.has_print_stock = true;
        if (name == "antihalation_removed") d.has_antihalation_removed = true;
        if (name == "preview_long_edge") d.has_preview_long_edge = true;
        if (f.layer == Layer::Shoot) d.any_shoot = true;
        if (!f.live) d.any_rebuild = true;
    }
    d.has_stock = d.has_film_stock || d.has_print_stock;
    if (d.has_stock) d.any_rebuild = true;
    return d;
}

void apply_delta(ParamSet& p, const Delta& d) {
    for (const auto& [fid, v] : d.fields) p.set(fid, v);
}

}  // namespace spk::params
