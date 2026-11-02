// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 Guangzhou Songzi Intelligent Technology Services Co., Ltd.
//
// params: the 48-row schema and delta validation.
#include <string>

#include "check.hpp"
#include "params/params.hpp"
#include "util/error.hpp"

using namespace spk;
using namespace spk::params;

static std::string refusal(const Schema& s, const std::string& text, DeltaContext c = DeltaContext::SetParams) {
    try {
        validate_delta(s, text, c);
    } catch (const Error& e) {
        return std::to_string(e.status()) + " " + e.what();
    }
    return "accepted";
}

int main() {
    auto s = Schema::build();
    CHECK(s->size() == 48);

    // Accepted deltas and their flags.
    Delta d = validate_delta(*s, "{}", DeltaContext::SetParams);
    CHECK(d.empty() && !d.any_shoot && !d.any_rebuild);
    d = validate_delta(*s, "", DeltaContext::Open);
    CHECK(d.empty());
    d = validate_delta(*s, R"({"print_exposure":2,"m_filter_shift":0.5})", DeltaContext::SetParams);
    CHECK(d.fields.size() == 2 && !d.any_shoot && !d.any_rebuild);
    CHECK(s->field(d.fields[0].first).name == "m_filter_shift");  // ascending bytewise order
    d = validate_delta(*s, R"({"print_exposure":2,"scan_film":true})", DeltaContext::SetParams);
    CHECK(!d.any_shoot && d.any_rebuild);
    d = validate_delta(*s, R"({"exposure_compensation_ev":1})", DeltaContext::SetParams);
    CHECK(d.any_shoot && d.any_rebuild && !d.has_stock);
    d = validate_delta(*s, R"({"print_stock":"kodak_2383"})", DeltaContext::SetParams);
    CHECK(d.has_stock && d.has_print_stock && !d.any_shoot && d.any_rebuild);
    d = validate_delta(*s, R"({"preview_long_edge":2400.5,"geometry_quarter_turns":2.5})", DeltaContext::SetParams);
    CHECK(d.has_preview_long_edge && d.fields.size() == 2);
    CHECK(d.fields[0].second.as_number() == 3.0);     // geometry_quarter_turns: half away from zero
    CHECK(d.fields[1].second.as_number() == 2401.0);  // preview_long_edge
    d = validate_delta(*s, R"({"geometry_quarter_turns":2.6})", DeltaContext::SetParams);
    CHECK(d.fields[0].second.as_number() == 3.0);
    d = validate_delta(*s, R"({"antihalation_removed":true})", DeltaContext::SetParams);
    CHECK(d.has_antihalation_removed && d.any_shoot);
    d = validate_delta(*s, R"({"product_defaults":true,"grain_active":false})", DeltaContext::Open);
    CHECK(d.product_defaults == std::optional<bool>(true) && d.fields.size() == 1);
    d = validate_delta(*s, R"({"print_exposure":0.05})", DeltaContext::SetParams);  // closed interval
    CHECK(d.fields.size() == 1);
    d = validate_delta(*s, R"({"print_exposure":20})", DeltaContext::SetParams);
    CHECK(d.fields.size() == 1);

    // Refusals and their messages.
    const std::string u = std::to_string(SPK_ERR_USER) + " ";
    CHECK_EQ_STR(refusal(*s, "[1]"), u + "params_delta must be an object");
    CHECK_EQ_STR(refusal(*s, R"({"no_such_field":1})"), u + "unknown parameter 'no_such_field'");
    CHECK_EQ_STR(refusal(*s, R"({"product_defaults":true})"), u + "unknown parameter 'product_defaults'");
    CHECK_EQ_STR(refusal(*s, R"({"product_defaults":1})", DeltaContext::Open), u + "'product_defaults' expects bool");
    CHECK_EQ_STR(refusal(*s, R"({"grain_active":"yes"})"), u + "'grain_active' expects bool");
    CHECK_EQ_STR(refusal(*s, R"({"grain_active":1})"), u + "'grain_active' expects bool");
    CHECK_EQ_STR(refusal(*s, R"({"print_exposure":null})"), u + "'print_exposure' expects float");
    CHECK_EQ_STR(refusal(*s, R"({"preview_long_edge":"x"})"), u + "'preview_long_edge' expects int");
    CHECK_EQ_STR(refusal(*s, R"({"film_stock":3})"), u + "'film_stock' expects str");
    CHECK_EQ_STR(refusal(*s, R"({"print_exposure":99.0})"),
                 u + "'print_exposure'=99.000000 is outside [0.050000, 20.000000]");
    CHECK_EQ_STR(refusal(*s, R"({"preview_long_edge":8192.5})"),
                 u + "'preview_long_edge'=8192.500000 is outside [800.000000, 8192.000000]");
    CHECK(refusal(*s, R"({"auto_exposure_method":"spot"})").rfind(u + "'auto_exposure_method' = 'spot' is not", 0) == 0);
    // The first failure in ascending bytewise order wins.
    CHECK_EQ_STR(refusal(*s, R"({"zzz":1,"grain_active":"x"})"), u + "'grain_active' expects bool");
    CHECK_EQ_STR(refusal(*s, R"({"Zed":1,"grain_active":"x"})"), u + "unknown parameter 'Zed'");
    // Strict JSON.
    CHECK(refusal(*s, R"({"print_exposure":NaN})").rfind(u + "JSON parse error", 0) == 0);
    CHECK(refusal(*s, R"({"print_exposure":Infinity})").rfind(u + "JSON parse error", 0) == 0);
    CHECK(refusal(*s, R"({"print_exposure":0x1p0})").rfind(u + "JSON parse error", 0) == 0);
    CHECK(refusal(*s, "").rfind(u + "JSON parse error", 0) == 0);

    // apply_delta writes the rounded values.
    ParamSet p(s);
    apply_delta(p, validate_delta(*s, R"({"geometry_quarter_turns":1.5,"print_exposure":1.25})", DeltaContext::SetParams));
    CHECK(p.i(P::geometry_quarter_turns) == 2 && p.f(P::print_exposure) == 1.25);
    CHECK(p.to_json().as_object().size() == 48);
    return finish("test_params");
}
