// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 Guangzhou Songzi Intelligent Technology Services Co., Ltd.
//
// util (JSON, numeric helpers) and fx/rng known-answer tests.
#include <cmath>
#include <cstring>
#include <limits>

#include "check.hpp"
#include "fx/rng.hpp"
#include "util/json.hpp"
#include "util/numeric.hpp"

using namespace spk;

static void test_number_format() {
    using json::format_number;
    CHECK_EQ_STR(format_number(1600), "1600");
    CHECK_EQ_STR(format_number(0), "0");
    CHECK_EQ_STR(format_number(-0.0), "-0");
    CHECK_EQ_STR(format_number(35), "35");
    CHECK_EQ_STR(format_number(-8), "-8");
    CHECK_EQ_STR(format_number(151301008), "151301008");
    CHECK_EQ_STR(format_number(151.301008), "151.301008");  // %.17g drops trailing zeros
    CHECK_EQ_STR(format_number(0.3), "0.29999999999999999");
    CHECK_EQ_STR(format_number(0.1), "0.10000000000000001");
    CHECK_EQ_STR(format_number(0.5), "0.5");
    CHECK_EQ_STR(format_number(1e-4), "0.0001");
    CHECK_EQ_STR(format_number(1e-5), "1.0000000000000001e-05");
    CHECK_EQ_STR(format_number(1e17), "1e+17");
    CHECK_EQ_STR(format_number(123456789012345678.0), "1.2345678901234568e+17");
    CHECK_EQ_STR(format_number(1e16), "10000000000000000");
    CHECK_EQ_STR(format_number(4294967295.0), "4294967295");
    CHECK_EQ_STR(format_number(std::numeric_limits<double>::infinity()), "null");
    CHECK_EQ_STR(format_number(-std::numeric_limits<double>::infinity()), "null");
    CHECK_EQ_STR(format_number(std::nan("")), "null");
    CHECK_EQ_STR(format_number(5e-324), "4.9406564584124654e-324");
    // Round trip: %.17g always re-reads to the same double.
    const double vals[] = {0.1, 1.0 / 3.0, 2.0 / 3.0, 1e-300, 1.7976931348623157e308, 0.18400000000000000,
                           -65.123456789, 5e-324, 51.568123456789};
    for (double v : vals) {
        json::Value r = json::parse(format_number(v));
        CHECK(r.is_number() && r.as_number() == v);
    }
}

static void test_writer() {
    json::Value o;
    o["zeta"] = 1;
    o["alpha"] = json::Array{1, 2.5, true, nullptr, "x"};
    o["Beta"] = "q\"\\\n\r\t\b\f\x01\x1f\x7f/é";
    o["empty_obj"] = json::Object{};
    o["empty_arr"] = json::Array{};
    o["nan"] = std::nan("");
    CHECK_EQ_STR(json::write(o),
                 "{\"Beta\":\"q\\\"\\\\\\n\\r\\t\\b\\f\\u0001\\u001f\x7f/é\",\"alpha\":[1,2.5,true,null,\"x\"],"
                 "\"empty_arr\":[],\"empty_obj\":{},\"nan\":null,\"zeta\":1}");
    // bytewise key order: upper case before lower case, and UTF-8 bytes (>= 0x80) last
    json::Value k;
    k["b"] = 1;
    k["B"] = 2;
    k["\xc3\xa9"] = 3;
    k["a_"] = 4;
    k["a"] = 5;
    CHECK_EQ_STR(json::write(k), "{\"B\":2,\"a\":5,\"a_\":4,\"b\":1,\"\xc3\xa9\":3}");
    // write -> parse -> write is stable
    std::string once = json::write(o);
    CHECK_EQ_STR(json::write(json::parse(once)), once);
}

static void test_reader() {
    json::Value v = json::parse(" {\"a\": [1, -2.5e3, 0, -0, 1E2, 0.5e-1], \"b\": {\"c\": null}, \"d\": \"\\u00e9\\ud83d\\ude00\\/\"} ");
    CHECK(v.find("a")->as_array().size() == 6);
    CHECK(v.find("a")->as_array()[1].as_number() == -2500);
    CHECK(v.find("a")->as_array()[4].as_number() == 100);
    CHECK(v.find("b")->find("c")->is_null());
    CHECK_EQ_STR(v.find("d")->as_string(), "\xc3\xa9\xf0\x9f\x98\x80/");
    // duplicate key: last wins
    CHECK(json::parse("{\"x\":1,\"x\":2}").find("x")->as_number() == 2);
    // strictness
    const char* bad[] = {"NaN", "nan", "Infinity", "-Infinity", "inf", "0x1p0", "01", "1.", ".5", "+1", "1e", "[1,]",
                         "{\"a\":1,}", "{'a':1}", "[1] x", "\"a\tb\"", "\"\\x\"", "\"\\ud800\"", "\"\xff\"",
                         "\"\xc0\xaf\"", "tru", "", "1e400", "{\"a\" 1}", "[", "\"abc"};
    for (const char* b : bad) {
        bool threw = false;
        try {
            json::parse(b);
        } catch (const json::ParseError&) {
            threw = true;
        }
        if (!threw) std::fprintf(stderr, "  accepted: %s\n", b);
        CHECK(threw);
    }
    CHECK(json::parse("1e-400").as_number() == 0.0);  // underflow rounds to 0
    // byte offset in the message
    try {
        json::parse("{\"a\": nan}");
        CHECK(false);
    } catch (const json::ParseError& e) {
        CHECK(e.offset() == 6);
        CHECK(std::string(e.what()).find("at byte 6") != std::string::npos);
    }
    // depth 64 ok, 65 refused
    std::string d64(64, '['), e64(64, ']');
    CHECK(json::parse(d64 + e64).is_array());
    std::string d65(65, '['), e65(65, ']');
    CHECK_THROWS(json::parse(d65 + e65));
}

static void test_numeric() {
    const double x[] = {0, 1, 2, 3};
    const double y[] = {10, 20, 40, 80};
    CHECK(num::search_right(x, 1.0) == 2);
    CHECK(num::search_right(x, -1.0) == 0);
    CHECK(num::search_right(x, 3.0) == 4);
    CHECK(num::interp(-5, x, y) == 10);
    CHECK(num::interp(5, x, y) == 80);
    CHECK(num::interp(1.5, x, y) == 30);
    CHECK(num::interp(2.0, x, y) == 40);
    const double xd[] = {0, 1, 1, 2};  // zero-width bracket returns y_j
    const double yd[] = {0, 1, 5, 6};
    CHECK(num::interp(1.0, xd, yd) == 5);
    const double nn[] = {std::nan(""), 3, -1, std::nan(""), 7};
    CHECK(num::nanmin(nn) == -1);
    CHECK(num::nanmax(nn) == 7);
    CHECK(num::nanmean(nn) == 3);
    const double allnan[] = {std::nan(""), std::nan("")};
    CHECK(std::isnan(num::nanmin(allnan)) && std::isnan(num::nanmean(allnan)));
    // Gaussian kernels: radius floor(truncate*sigma + 0.5), normalised
    CHECK(num::gaussian_radius(0.65, 3.0) == 2);
    CHECK(num::gaussian_radius(2.9, 3.0) == 9);
    CHECK(num::gaussian_radius(0.5, 3.0) == 2);
    CHECK(num::gaussian_radius(0.1, 4.0) == 0);
    auto k = num::gaussian_kernel(1.0, 4.0);
    CHECK(k.size() == 9);
    double s = 0;
    for (double w : k) s += w;
    CHECK(std::fabs(s - 1.0) < 1e-15);
    CHECK(k[4] > k[3] && k[3] == k[5]);
    CHECK(num::gaussian_kernel(0.0, 3.0).size() == 1);
    // float16
    CHECK(num::half_to_double(0x3C00) == 1.0);
    CHECK(num::half_to_double(0xC000) == -2.0);
    CHECK(num::half_to_double(0x0001) == std::ldexp(1.0, -24));
    CHECK(num::half_to_double(0x03FF) == 1023 * std::ldexp(1.0, -24));
    CHECK(num::half_to_double(0x0400) == std::ldexp(1.0, -14));
    CHECK(num::half_to_double(0x7BFF) == 65504.0);
    CHECK(std::isinf(num::half_to_double(0x7C00)) && num::half_to_double(0xFC00) < 0);
    CHECK(std::isnan(num::half_to_double(0x7E00)));
    CHECK(std::signbit(num::half_to_double(0x8000)));
    // index folds
    CHECK(num::mirror_index(-1, 4) == 1 && num::mirror_index(4, 4) == 2 && num::mirror_index(-3, 4) == 3);
    CHECK(num::mirror_index(5, 1) == 0);
    CHECK(num::reflect_index(-1, 4) == 0 && num::reflect_index(4, 4) == 3 && num::reflect_index(-5, 4) == 3);
    CHECK(num::reflect_index(9, 4) == 1);
    CHECK(num::round_half_away(2.5) == 3 && num::round_half_away(-2.5) == -3 && num::round_half_away(2.4) == 2);
}

static void test_rng() {
    auto a = fx::philox4x32_10({0, 0, 0, 0}, {0, 0});
    CHECK(a[0] == 0x6627e8d5u && a[1] == 0xe169c58du && a[2] == 0xbc57ac4cu && a[3] == 0x9b00dbd8u);
    auto b = fx::philox4x32_10({0, 200, 0, 0}, {0x5EED, 0});
    CHECK(b[0] == 0x15cecb3fu && b[1] == 0xf63dfa39u && b[2] == 0x3ec2dc0eu && b[3] == 0xf78f6c18u);
    fx::SeedSequence s;
    CHECK(s.next() == 0x4abea221u);
    CHECK(s.next() == 0x23148989u);
    CHECK(s.next() == 0x609dfe03u);
    CHECK(s.next() == 0xdeb12800u);
    s.reset();
    CHECK(s.next() == 0x4abea221u);
    CHECK(fx::philox_uniform(0) == 0.5f * 0x1p-24f);
    // Edge: in binary32, (2^24 - 1) + 0.5 rounds (ties to even) to 2^24, so the largest words give
    // exactly 1.0f. Pinned here.
    CHECK(fx::philox_uniform(0xFFFFFEFFu) < 1.0f);
    CHECK(fx::philox_uniform(0xFFFFFFFFu) == 1.0f);
}

int main() {
    test_number_format();
    test_writer();
    test_reader();
    test_numeric();
    test_rng();
    return finish("test_util");
}
