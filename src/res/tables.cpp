// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 Guangzhou Songzi Intelligent Technology Services Co., Ltd.

/* neutral_print_filters.json, print_luts.json and the blob's print tables; rules in tables.hpp. */
#include "res/tables.hpp"

#include <cmath>
#include <fstream>
#include <sstream>

#include "util/error.hpp"

namespace spk::res {

namespace {

bool read_if_exists(const std::filesystem::path& p, std::string& out) {
    std::error_code ec;
    if (!std::filesystem::exists(p, ec)) return false;
    std::ifstream f(p, std::ios::binary);
    if (!f) fail(SPK_ERR_IO, "cannot read '" + p.string() + "'");
    std::ostringstream ss;
    ss << f.rdbuf();
    out = ss.str();
    return true;
}

}  // namespace

// ---- neutral filters ---------------------------------------------------------------------------

NeutralFilterDb NeutralFilterDb::load(const std::filesystem::path& file) {
    NeutralFilterDb db;
    std::string text;
    if (!read_if_exists(file, text)) return db;
    try {
        db.root_ = json::parse(text);
    } catch (const json::ParseError& e) {
        fail(SPK_ERR_IO, "neutral_print_filters.json: " + std::string(e.what()));
    }
    db.present_ = true;
    return db;
}

std::optional<std::array<double, 3>> NeutralFilterDb::lookup(std::string_view print_stock,
                                                             std::string_view illuminant,
                                                             std::string_view film_stock) const {
    const json::Value* a = root_.find(print_stock);
    if (!a) return std::nullopt;
    const json::Value* b = a->find(illuminant);
    if (!b) return std::nullopt;
    const json::Value* leaf = b->find(film_stock);
    if (!leaf || !leaf->is_array() || leaf->as_array().size() != 3) return std::nullopt;
    std::array<double, 3> out{};
    for (size_t i = 0; i < 3; ++i) {
        const json::Value& v = leaf->as_array()[i];
        if (!v.is_number()) return std::nullopt;
        out[i] = v.as_number();
    }
    return out;
}

std::optional<std::array<double, 3>> NeutralFilterDb::lookup_with_alias(std::string_view print_stock,
                                                                        std::string_view illuminant,
                                                                        std::string_view film_id,
                                                                        std::string_view base_id) const {
    const json::Value* leaf = nullptr;
    if (const json::Value* a = root_.find(print_stock))
        if (const json::Value* b = a->find(illuminant)) leaf = b->find(film_id);
    if (leaf && leaf->is_array()) return lookup(print_stock, illuminant, film_id);
    if (!base_id.empty()) return lookup(print_stock, illuminant, base_id);
    return std::nullopt;
}

// ---- print LUT index -----------------------------------------------------------------------------

/* PrintLutIndex::load(file)
 *   absent: present = false
 *   root must be an object; for each stock: object, string paired_film, lut_size integer in (1, 4096]
 *   declared_pairing, output_color_space, output_cctf_encoding: read when of the right type
 *   any failure: SPK_ERR_IO
 */
PrintLutIndex PrintLutIndex::load(const std::filesystem::path& file) {
    PrintLutIndex idx;
    std::string text;
    if (!read_if_exists(file, text)) return idx;
    json::Value root;
    try {
        root = json::parse(text);
    } catch (const json::ParseError& e) {
        fail(SPK_ERR_IO, "print_luts.json: " + std::string(e.what()));
    }
    if (!root.is_object()) fail(SPK_ERR_IO, "print_luts.json: not an object");
    for (const auto& [stock, v] : root.as_object()) {
        auto bad = [&](const std::string& m) { fail(SPK_ERR_IO, "print_luts.json: '" + stock + "': " + m); };
        if (!v.is_object()) bad("not an object");
        PrintLutEntry e;
        e.stock = stock;
        const json::Value* pf = v.find("paired_film");
        if (!pf || !pf->is_string()) bad("missing paired_film");
        e.paired_film = pf->as_string();
        const json::Value* ls = v.find("lut_size");
        if (!ls || !ls->is_number()) bad("missing lut_size");
        double s = ls->as_number();
        if (!(s > 1.0) || s != std::floor(s) || s > 4096.0) bad("lut_size must be an integer > 1");
        e.lut_size = uint32_t(s);
        if (const json::Value* d = v.find("declared_pairing"); d && d->is_bool()) e.declared_pairing = d->as_bool();
        if (const json::Value* d = v.find("output_color_space"); d && d->is_string()) e.output_color_space = d->as_string();
        if (const json::Value* d = v.find("output_cctf_encoding"); d && d->is_bool()) e.output_cctf_encoding = d->as_bool();
        idx.entries_.emplace(stock, std::move(e));
    }
    idx.present_ = true;
    idx.bytes_ = std::move(text);
    return idx;
}

const PrintLutEntry* PrintLutIndex::find(std::string_view stock) const {
    auto it = entries_.find(stock);
    return it == entries_.end() ? nullptr : &it->second;
}

PrintLutTable PrintLutIndex::table(const Blob& blob, std::string_view stock) const {
    if (!present_) fail_user("no print-preview table index (print_luts.json) in the resources");
    const PrintLutEntry* e = find(stock);
    if (!e) {
        std::string list;
        for (const auto& [k, _] : entries_) list += (list.empty() ? "" : ", ") + k;
        fail_user("no print-preview table for '" + std::string(stock) + "'; available: " + list);
    }
    const std::string tname = "print_lut/" + std::string(stock);
    const std::string aname = "print_lut_axes/" + std::string(stock);
    const BlobEntry* te = blob.find(tname);
    const BlobEntry* ae = blob.find(aname);
    const uint32_t S = e->lut_size;
    if (!te || te->dtype != DType::F32 || te->ndim != 4 || te->dims[0] != S || te->dims[1] != S ||
        te->dims[2] != S || te->dims[3] != 3)
        fail_user("print-preview table '" + tname + "' is missing or has the wrong shape");
    if (!ae || ae->dtype != DType::F32 || ae->ndim != 2 || ae->dims[0] != 3 || ae->dims[1] != S)
        fail_user("print-preview axes '" + aname + "' are missing or have the wrong shape");
    PrintLutTable t;
    t.table = blob.f32_view(tname).data();
    t.size = S;
    std::span<const float> ax = blob.f32_view(aname);
    for (size_t c = 0; c < 3; ++c) {
        t.lo[c] = ax[c * S];
        t.hi[c] = ax[c * S + S - 1];
        if (!(t.hi[c] > t.lo[c])) fail_user("print-preview axes of '" + std::string(stock) + "' are not increasing");
    }
    return t;
}

}  // namespace spk::res
