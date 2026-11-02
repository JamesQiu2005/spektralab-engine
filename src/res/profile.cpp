// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 Guangzhou Songzi Intelligent Technology Services Co., Ltd.
#include "res/profile.hpp"

#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>

#include "util/error.hpp"
#include "util/json.hpp"
#include "util/numeric.hpp"

namespace spk::res {

namespace {

using json::Value;

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

struct Ctx {
    std::string id;  // for messages
    [[noreturn]] void bad(const std::string& what) const {
        fail_user("profile '" + id + "': " + what);
    }
};

std::string read_text(const std::filesystem::path& p, const Ctx& c) {
    std::ifstream f(p, std::ios::binary);
    if (!f) c.bad("cannot read '" + p.string() + "'");
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

Value parse_file(const std::filesystem::path& p, const Ctx& c) {
    std::string text = read_text(p, c);
    try {
        return json::parse(text);
    } catch (const json::ParseError& e) {
        c.bad(e.what());
    }
}

double number_or_nan(const Value& v, const Ctx& c, const char* key) {
    if (v.is_number()) return v.as_number();
    if (v.is_null()) return kNaN;
    c.bad(std::string("'") + key + "' holds a non-numeric element");
}

std::vector<double> vec(const Value& v, size_t n, const Ctx& c, const char* key) {
    if (!v.is_array()) c.bad(std::string("'") + key + "' must be an array");
    const auto& a = v.as_array();
    if (n && a.size() != n)
        c.bad(std::string("'") + key + "' must have " + std::to_string(n) + " values, has " + std::to_string(a.size()));
    std::vector<double> out;
    out.reserve(a.size());
    for (const Value& e : a) out.push_back(number_or_nan(e, c, key));
    return out;
}

// rows == 0: any row count (>= 1).
Mat mat(const Value& v, size_t rows, size_t cols, const Ctx& c, const char* key) {
    if (!v.is_array()) c.bad(std::string("'") + key + "' must be an array");
    const auto& a = v.as_array();
    if (rows && a.size() != rows)
        c.bad(std::string("'") + key + "' must have " + std::to_string(rows) + " rows, has " + std::to_string(a.size()));
    if (a.empty()) c.bad(std::string("'") + key + "' is empty");
    Mat m;
    m.rows = a.size();
    m.cols = cols;
    m.v.reserve(m.rows * cols);
    for (const Value& r : a) {
        if (!r.is_array() || r.as_array().size() != cols)
            c.bad(std::string("'") + key + "' must have rows of " + std::to_string(cols) + " values");
        for (const Value& e : r.as_array()) m.v.push_back(number_or_nan(e, c, key));
    }
    return m;
}

// A matrix whose column count is taken from the first row; every row must match it.
Mat mat_any(const Value& v, const Ctx& c, const char* key) {
    if (!v.is_array() || v.as_array().empty() || !v.as_array()[0].is_array())
        c.bad(std::string("'") + key + "' must be a non-empty array of arrays");
    return mat(v, 0, v.as_array()[0].as_array().size(), c, key);
}

void read_info_strings(const Value& info, ProfileInfo& pi) {
    auto str = [&](const char* k, std::string& dst) {
        if (const Value* v = info.find(k); v && v->is_string()) dst = v->as_string();
    };
    str("name", pi.name);
    str("type", pi.type);
    str("support", pi.support);
    str("stage", pi.stage);
    str("use", pi.use);
    str("antihalation", pi.antihalation);
    str("target_print", pi.target_print);
    str("channel_model", pi.channel_model);
    str("densitometer", pi.densitometer);
    str("reference_illuminant", pi.reference_illuminant);
    str("viewing_illuminant", pi.viewing_illuminant);
    if (const Value* v = info.find("log_sensitivity_density_over_min"); v && v->is_number())
        pi.log_sensitivity_density_over_min = v->as_number();
}

void check_type(const ProfileInfo& pi, const Ctx& c) {
    if (pi.type != "negative" && pi.type != "positive")
        c.bad("info.type '" + pi.type + "' is not 'negative' or 'positive'");
}

EdrToneMap read_edr(const Value& v, const Ctx& c) {
    if (!v.is_object()) c.bad("'edr_tone_map' must be an object");
    EdrToneMap e;
    const Value* ver = v.find("version");
    if (!ver || !ver->is_number() || ver->as_number() != 1.0) c.bad("edr_tone_map.version must be 1");
    e.version = 1;
    const Value* dom = v.find("domain_log2_y");
    if (!dom) c.bad("edr_tone_map.domain_log2_y is missing");
    std::vector<double> d = vec(*dom, 2, c, "edr_tone_map.domain_log2_y");
    e.domain_lo = d[0];
    e.domain_hi = d[1];
    if (!(e.domain_hi > e.domain_lo)) c.bad("edr_tone_map.domain_log2_y needs hi > lo");
    const Value* toe = v.find("toe_y");
    const Value* sh = v.find("shoulder_y");
    if (!toe || !toe->is_number() || !(toe->as_number() > 0)) c.bad("edr_tone_map.toe_y must be > 0");
    if (!sh || !sh->is_number() || !(sh->as_number() > toe->as_number()))
        c.bad("edr_tone_map.shoulder_y must be > toe_y");
    e.toe_y = toe->as_number();
    e.shoulder_y = sh->as_number();
    const Value* t = v.find("target_log2_y");
    if (!t) c.bad("edr_tone_map.target_log2_y is missing");
    e.target_log2_y = vec(*t, 0, c, "edr_tone_map.target_log2_y");
    if (e.target_log2_y.size() < 2) c.bad("edr_tone_map.target_log2_y needs at least 2 values");
    for (double x : e.target_log2_y)
        if (!std::isfinite(x)) c.bad("edr_tone_map.target_log2_y must be finite");
    if (const Value* s = v.find("calibration_film"); s && s->is_string()) e.calibration_film = s->as_string();
    if (const Value* s = v.find("semantic"); s && s->is_string()) e.semantic = s->as_string();
    return e;
}

/* read_data(data, p): required wavelengths, log_sensitivity, channel_density, base_density,
 *                     log_exposure, density_curves; every other key optional, shape-checked when present
 */
void read_data(const Value& data, Profile& p, const Ctx& c) {
    if (!data.is_object()) c.bad("'data' must be an object");
    auto req = [&](const char* k) -> const Value& {
        const Value* v = data.find(k);
        if (!v) c.bad(std::string("data.") + k + " is missing");
        return *v;
    };
    p.wavelengths = vec(req("wavelengths"), kSpectralSamples, c, "wavelengths");
    for (size_t i = 0; i < kSpectralSamples; ++i)
        if (p.wavelengths[i] != 380.0 + 5.0 * double(i)) c.bad("wavelengths must be 380, 385, ..., 780");
    p.log_sensitivity = mat(req("log_sensitivity"), kSpectralSamples, 3, c, "log_sensitivity");
    p.channel_density = mat(req("channel_density"), kSpectralSamples, 3, c, "channel_density");
    p.base_density = vec(req("base_density"), kSpectralSamples, c, "base_density");
    if (const Value* v = data.find("midscale_neutral_density"))
        p.midscale_neutral_density = vec(*v, kSpectralSamples, c, "midscale_neutral_density");
    p.log_exposure = vec(req("log_exposure"), 0, c, "log_exposure");
    const size_t K = p.log_exposure.size();
    if (K < 2) c.bad("log_exposure needs at least 2 points");
    p.density_curves = mat(req("density_curves"), K, 3, c, "density_curves");

    if (const Value* v = data.find("density_curves_layers")) {
        if (!v->is_array() || v->as_array().size() != K) c.bad("density_curves_layers must be K x 3 x 3");
        p.density_curves_layers.reserve(K * 9);
        for (const Value& row : v->as_array()) {
            Mat m = mat(row, 3, 3, c, "density_curves_layers");
            p.density_curves_layers.insert(p.density_curves_layers.end(), m.v.begin(), m.v.end());
        }
    }
    if (const Value* v = data.find("density_curves_model")) {
        if (!v->is_object()) c.bad("density_curves_model must be an object");
        DensityCurvesModel m;
        if (const Value* t = v->find("model_type"); t && t->is_string()) m.model_type = t->as_string();
        auto part = [&](const char* k) {
            const Value* x = v->find(k);
            if (!x) c.bad(std::string("density_curves_model.") + k + " is missing");
            return mat_any(*x, c, k);
        };
        m.centers = part("centers");
        m.amplitudes = part("amplitudes");
        m.sigmas = part("sigmas");
        if (m.amplitudes.rows != m.centers.rows || m.amplitudes.cols != m.centers.cols ||
            m.sigmas.rows != m.centers.rows || m.sigmas.cols != m.centers.cols)
            c.bad("density_curves_model matrices differ in shape");
        p.density_curves_model = std::move(m);
    }
    if (const Value* v = data.find("hanatos2025_adaptation_window_params")) {
        p.adaptation_window_params = vec(*v, 0, c, "hanatos2025_adaptation_window_params");
        size_t n = p.adaptation_window_params.size();
        if (n != 0 && n != 4 && n != 8) c.bad("hanatos2025_adaptation_window_params must have 0, 4 or 8 values");
    }
    if (const Value* v = data.find("hanatos2025_adaptation_surface_params");
        v && !(v->is_array() && v->as_array().empty())) {  // [] on the papers: no surface
        Mat m = mat_any(*v, c, "hanatos2025_adaptation_surface_params");
        if (m.rows != 3) c.bad("hanatos2025_adaptation_surface_params must be 3 x n");
        p.adaptation_surface_params = std::move(m);
    }
    if (const Value* v = data.find("edr_tone_map")) p.edr_tone_map = read_edr(*v, c);
}

void derive(Profile& p) {
    const size_t K = p.curve_points();
    p.normalized_curves = p.density_curves;
    for (size_t ch = 0; ch < 3; ++ch) {
        p.dmin[ch] = num::nanmin_strided(p.density_curves.v.data() + ch, K, 3);
        for (size_t k = 0; k < K; ++k) p.normalized_curves(k, ch) = p.density_curves(k, ch) - p.dmin[ch];
        p.dmax[ch] = num::nanmax_strided(p.normalized_curves.v.data() + ch, K, 3);
    }
}

// Parses a non-alias profile document into p (info + data).
void parse_plain(const Value& doc, Profile& p, const Ctx& c) {
    const Value* info = doc.find("info");
    const Value* data = doc.find("data");
    if (!info) c.bad("'info' is missing");
    if (!info->is_object()) c.bad("'info' must be an object");
    if (!data) c.bad("'data' is missing");
    if (const Value* s = info->find("stock"); s && s->is_string()) p.info.stock = s->as_string();
    read_info_strings(*info, p.info);
    check_type(p.info, c);
    read_data(*data, p, c);
}

}  // namespace

bool valid_stock_id(std::string_view id) {
    if (id.empty() || id.size() > 96) return false;
    for (char ch : id) {
        bool ok = (ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z') || (ch >= '0' && ch <= '9') ||
                  ch == '_' || ch == '-';
        if (!ok) return false;
    }
    return true;
}

/* load_profile: steps in profile.hpp */
std::shared_ptr<const Profile> load_profile(const std::filesystem::path& resources_dir, std::string_view id) {
    Ctx c{std::string(id)};
    if (!valid_stock_id(id)) fail_user("invalid stock id '" + std::string(id) + "'");
    const std::filesystem::path dir = resources_dir / "profiles";
    Value doc = parse_file(dir / (std::string(id) + ".json"), c);
    if (!doc.is_object()) c.bad("the file is not a JSON object");

    auto p = std::make_shared<Profile>();
    p->id = std::string(id);

    const Value* df = doc.find("derived_from");
    if (df && df->is_string()) {
        const std::string base = df->as_string();
        Ctx cb{std::string(id) + "' (alias of '" + base};
        if (!valid_stock_id(base)) cb.bad("invalid base stock id");
        Value bdoc = parse_file(dir / (base + ".json"), cb);
        if (!bdoc.is_object()) cb.bad("the base file is not a JSON object");
        if (const Value* bdf = bdoc.find("derived_from"); bdf && bdf->is_string())
            cb.bad("the base is itself an alias (only one level is allowed)");
        parse_plain(bdoc, *p, cb);
        if (p->info.stock.empty()) p->info.stock = base;
        if (p->info.name.empty()) p->info.name = p->info.stock;  // the base's default name
        if (const Value* ainfo = doc.find("info"); ainfo && ainfo->is_object()) read_info_strings(*ainfo, p->info);
        check_type(p->info, cb);
        p->base_id = base;
        p->info.stock = p->id;
        if (p->info.name.empty()) p->info.name = p->id;
    } else {
        parse_plain(doc, *p, c);
        if (p->info.stock.empty()) p->info.stock = p->id;
        if (p->info.name.empty()) p->info.name = p->info.stock;
    }
    derive(*p);
    return p;
}

}  // namespace spk::res
