// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 Guangzhou Songzi Intelligent Technology Services Co., Ltd.
//
// Resource readers against real data. Usage: test_res <res> [<res_app>]   (<res_app>: a second
// resources folder holding the alias and black-and-white profiles; those checks are skipped without it)
// Prints every blob entry (name, dtype, shape) and every profile loaded.
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <set>

#include "check.hpp"
#include "res/colour.hpp"
#include "res/resources.hpp"
#include "util/error.hpp"

using namespace spk;
namespace fs = std::filesystem;

static void test_blob(const fs::path& res) {
    auto blob = res::Blob::open(res / "spektralab_constants.bin");
    CHECK(blob->version() == 1);
    CHECK(blob->entries().size() == 71);
    std::printf("constants blob: %zu entries\n", blob->entries().size());
    for (const res::BlobEntry& e : blob->entries()) {
        std::string shape;
        for (uint32_t i = 0; i < e.ndim; ++i) shape += (i ? "x" : "") + std::to_string(e.dims[i]);
        std::printf("  %-48s %-8s %-14s offset %9llu bytes %9llu%s\n", e.name.c_str(), res::dtype_name(e.dtype),
                    shape.c_str(), (unsigned long long)e.offset, (unsigned long long)e.nbytes, e.size_ok ? "" : "  SIZE MISMATCH");
        CHECK(e.size_ok);
        CHECK(e.offset % 8 == 0);
    }
    CHECK(blob->entries().front().offset == 7400);  // 16 + 71 * 104 = 7400
    const res::BlobEntry* lut = blob->find("hanatos/spectra_lut");
    CHECK(lut && lut->dtype == res::DType::F16 && lut->dims[0] == 192 && lut->dims[1] == 192 && lut->dims[2] == 81);
    std::vector<double> v = blob->f64("hanatos/spectra_lut");
    double lo = 1e300, hi = -1e300;
    bool finite = true;
    for (double x : v) {
        lo = std::min(lo, x);
        hi = std::max(hi, x);
        finite &= std::isfinite(x);
    }
    std::printf("  spectra_lut range [%.3g, %.4g]\n", lo, hi);
    CHECK(finite && lo > 0 && lo < 3e-6 && hi > 16 && hi < 16.5);
    std::vector<double> wl = blob->f64("spectral/wavelengths");
    CHECK(wl.size() == 81 && wl[0] == 380 && wl[80] == 780);
    int n_lut = 0;
    for (const res::BlobEntry& e : blob->entries()) n_lut += e.name.rfind("print_lut/", 0) == 0;
    CHECK(n_lut == 8);

    // corrupted copies are refused
    fs::path tmp = fs::path(SPK_TEST_TMP);
    fs::create_directories(tmp);
    std::ifstream in(res / "spektralab_constants.bin", std::ios::binary);
    std::string bytes((std::istreambuf_iterator<char>(in)), {});
    auto write = [&](const std::string& name, const std::string& b) {
        std::ofstream o(tmp / name, std::ios::binary);
        o << b;
        return tmp / name;
    };
    std::string bad_magic = bytes;
    bad_magic[0] = 'X';
    CHECK_THROWS(res::Blob::open(write("bad_magic.bin", bad_magic)));
    std::string bad_ver = bytes;
    bad_ver[4] = 2;
    CHECK_THROWS(res::Blob::open(write("bad_version.bin", bad_ver)));
    CHECK_THROWS(res::Blob::open(write("truncated.bin", bytes.substr(0, bytes.size() / 2))));
    std::string no_req = bytes;  // rename the cmfs entry
    size_t pos = no_req.find("colour/cmfs_lms");
    no_req[pos] = 'X';
    CHECK_THROWS(res::Blob::open(write("missing_required.bin", no_req)));
}

static void check_profile_folder(const fs::path& dir, size_t want) {
    {
        std::set<std::string> ids;
        for (const auto& f : fs::directory_iterator(dir / "profiles"))
            if (f.path().extension() == ".json") ids.insert(f.path().stem().string());
        size_t loaded = 0;
        std::printf("profiles in %s: %zu\n", dir.c_str(), ids.size());
        for (const std::string& id : ids) {
            try {
                auto p = res::load_profile(dir, id);
                ++loaded;
                std::printf("  %-34s %-8s %-6s use=%-5s ah=%-6s K=%zu layers=%d edr=%d window=%zu%s%s\n", id.c_str(),
                            p->info.type.c_str(), p->info.support.c_str(), p->info.use.c_str(),
                            p->info.antihalation.c_str(), p->curve_points(), int(p->has_layers()),
                            int(p->edr_tone_map.has_value()), p->adaptation_window_params.size(),
                            p->is_alias() ? "  alias of " : "", p->base_id.c_str());
                CHECK(p->info.stock == id);
                CHECK(p->log_sensitivity.rows == 81 && p->channel_density.rows == 81);
                for (int c = 0; c < 3; ++c) CHECK(std::isfinite(p->dmax[c]) && p->dmax[c] > 0);
            } catch (const std::exception& e) {
                std::printf("  %-34s LOAD FAILED: %s\n", id.c_str(), e.what());
                CHECK(false);
            }
        }
        CHECK(loaded == ids.size());
        CHECK(loaded == want);
    }
}

static void test_app_profiles(const fs::path& app) {
    check_profile_folder(app, 34u);
    // alias semantics
    auto cs = res::load_profile(app, "cinestill_800t");
    auto base = res::load_profile(app, "kodak_vision3_500t");
    CHECK(cs->is_alias() && cs->base_id == "kodak_vision3_500t" && cs->id == "cinestill_800t");
    CHECK(cs->info.name == "CineStill 800T" && cs->info.use == "still" && cs->info.antihalation == "no");
    CHECK(base->info.use == "cine" && base->info.antihalation == "strong");
    CHECK(cs->info.type == base->info.type && cs->info.reference_illuminant == base->info.reference_illuminant);
    CHECK(cs->density_curves.v.size() == base->density_curves.v.size());
    CHECK(std::memcmp(cs->log_sensitivity.v.data(), base->log_sensitivity.v.data(), 81 * 3 * 8) == 0);
    // B&W product data
    auto tx = res::load_profile(app, "kodak_tri_x_400");
    CHECK(tx->info.channel_model == "bw" && tx->info.target_print == "ilford_multigrade_iv_rc");
    auto mg = res::load_profile(app, "ilford_multigrade_iv_rc");
    CHECK(mg->info.support == "paper" && !mg->edr_tone_map && mg->info.log_sensitivity_density_over_min == 0.6);
}

static void test_profiles(const fs::path& res) {
    check_profile_folder(res, 28u);
    // NaN preserved (Portra 400 channel_density has 22 nulls)
    auto p4 = res::load_profile(res, "kodak_portra_400");
    int nans = 0;
    for (double x : p4->channel_density.v) nans += std::isnan(x);
    CHECK(nans == 22);
    // target_print null on a paper reads as ""
    auto endura = res::load_profile(res, "kodak_portra_endura");
    CHECK(endura->info.target_print.empty() && endura->edr_tone_map && endura->edr_tone_map->target_log2_y.size() == 256);

    // refusals
    CHECK_THROWS(res::load_profile(res, "../resources/profiles/kodak_portra_400"));
    CHECK_THROWS(res::load_profile(res, "no_such_stock"));
    CHECK_THROWS(res::load_profile(res, ""));
    fs::path tmp = fs::path(SPK_TEST_TMP) / "res_bad";
    fs::create_directories(tmp / "profiles");
    std::ifstream in(res / "profiles" / "kodak_portra_400.json");
    std::string text((std::istreambuf_iterator<char>(in)), {});
    auto put = [&](const std::string& id, const std::string& t) { std::ofstream(tmp / "profiles" / (id + ".json")) << t; };
    std::string slide = text;
    slide.replace(slide.find("\"negative\""), 10, "\"slide\"");
    put("slide", slide);
    CHECK_THROWS(res::load_profile(tmp, "slide"));
    std::string wl80 = text;
    size_t w = wl80.find("\"wavelengths\"");
    size_t first = wl80.find("380", w);
    wl80.erase(first, wl80.find(',', first) - first + 1);  // drop the 380 sample
    put("wl80", wl80);
    CHECK_THROWS(res::load_profile(tmp, "wl80"));
    put("chain_a", "{\"derived_from\": \"chain_b\"}");
    put("chain_b", "{\"derived_from\": \"kodak_portra_400\"}");
    put("kodak_portra_400", text);
    CHECK_THROWS(res::load_profile(tmp, "chain_a"));
    put("ok_alias", "{\"derived_from\": \"kodak_portra_400\", \"info\": {\"name\": 7, \"use\": \"cine\"}}");
    auto oa = res::load_profile(tmp, "ok_alias");
    CHECK(oa->info.name == "Kodak Portra 400" && oa->info.use == "cine" && oa->info.stock == "ok_alias");
    put("nonstr_alias", "{\"derived_from\": null, \"info\": {}}");
    CHECK_THROWS(res::load_profile(tmp, "nonstr_alias"));  // not an alias -> needs data
}

static void test_tables(const fs::path& res_dir) {
    auto r = res::Resources::load(res_dir);
    // neutral filter database
    auto hit = r->filters.lookup("kodak_portra_endura", "TH-KG3", "kodak_portra_400");
    CHECK(hit && (*hit)[0] == 0.0 && std::fabs((*hit)[1] - 51.568) < 1e-3 && std::fabs((*hit)[2] - 52.534) < 1e-3);
    CHECK(!r->filters.lookup("kodak_portra_endura", "TH-KG3", "cinestill_800t"));
    CHECK(r->filters.lookup_with_alias("kodak_portra_endura", "TH-KG3", "cinestill_800t", "kodak_vision3_500t") ==
          r->filters.lookup("kodak_portra_endura", "TH-KG3", "kodak_vision3_500t"));
    CHECK(!r->filters.lookup("ilford_multigrade_iv_rc", "TH-KG3", "kodak_tri_x_400"));
    CHECK(!r->filters.lookup("kodak_portra_endura", "D50", "kodak_portra_400"));
    // Alias fallback only when the alias leaf is not an array.
    {
        fs::path f = fs::path(SPK_TEST_TMP) / "nf_alias.json";
        std::ofstream(f, std::ios::binary)
            << "{\"P\":{\"TH-KG3\":{\"alias_x\":[1,2],\"alias_z\":[1,null,3],\"alias_s\":\"x\","
               "\"alias_ok\":[4,5,6],\"base_b\":[7,8,9]}}}";
        auto db = res::NeutralFilterDb::load(f);
        CHECK(!db.lookup_with_alias("P", "TH-KG3", "alias_x", "base_b"));
        CHECK(!db.lookup_with_alias("P", "TH-KG3", "alias_z", "base_b"));
        auto s = db.lookup_with_alias("P", "TH-KG3", "alias_s", "base_b");
        CHECK(s && (*s)[0] == 7.0 && (*s)[2] == 9.0);
        auto m = db.lookup_with_alias("P", "TH-KG3", "alias_missing", "base_b");
        CHECK(m && (*m)[1] == 8.0);
        auto o = db.lookup_with_alias("P", "TH-KG3", "alias_ok", "base_b");
        CHECK(o && (*o)[0] == 4.0);
        CHECK(!db.lookup_with_alias("P", "TH-KG3", "alias_missing", ""));
        CHECK(!db.lookup_with_alias("Q", "TH-KG3", "alias_missing", "base_b"));
    }
    // print LUT index and tables
    CHECK(r->print_luts.present() && r->print_luts.entries().size() == 8);
    std::ifstream in(res_dir / "print_luts.json", std::ios::binary);
    std::string bytes((std::istreambuf_iterator<char>(in)), {});
    CHECK(r->print_luts.catalog_bytes() == bytes);
    for (const auto& [stock, e] : r->print_luts.entries()) {
        res::PrintLutTable t = r->print_luts.table(*r->blob, stock);
        CHECK(t.size == 33 && e.lut_size == 33 && e.output_color_space == "Display P3");
        auto raw = r->blob->bytes("print_lut/" + stock);
        CHECK(std::memcmp(raw.data(), t.table, raw.size()) == 0 && raw.size() == 33u * 33 * 33 * 3 * 4);
    }
    CHECK(r->print_luts.find("kodak_2383")->paired_film == "kodak_vision3_250d");
    try {
        r->print_luts.table(*r->blob, "ilford_multigrade_iv_rc");
        CHECK(false);
    } catch (const Error& e) {
        CHECK(e.status() == SPK_ERR_USER && std::string(e.what()).find("kodak_2383") != std::string::npos);
    }
    // colour data
    for (std::string_view n : res::baked_colour_space_names()) CHECK(res::colour_space(*r->blob, n).has_value());
    CHECK(!res::colour_space(*r->blob, "CIE RGB"));
    auto pp = res::colour_space(*r->blob, "ProPhoto RGB");
    CHECK(pp->rgb_to_xyz[0] == 0.7977 && pp->xyz_to_rgb[0] == 1.3460 && pp->cctf == res::CctfMode::ProPhoto);
    for (const char* n : {"D50", "D55", "D65", "D75", "E", "A", "T", "K75P", "TH-KG3", "TH-KG3-L", "BB3200", "Incandescent",
                          "Kinoton 75P"}) {
        auto s = res::illuminant(*r->blob, n);
        CHECK(s.has_value());
        if (s) {
            double m = 0;
            for (double x : *s) m += x;
            CHECK(std::fabs(m / 81.0 - 1.0) < 1e-12);
        }
    }
    CHECK(!res::illuminant(*r->blob, "D93") && !res::illuminant(*r->blob, "BB") && !res::illuminant(*r->blob, "BBx"));
    // known chromaticities
    auto d50 = res::chromaticity(*r->blob, *res::illuminant(*r->blob, "D50"));
    auto d65 = res::chromaticity(*r->blob, *res::illuminant(*r->blob, "D65"));
    auto k75 = res::chromaticity(*r->blob, *res::illuminant(*r->blob, "K75P"));
    std::printf("xy D50 (%.10f, %.10f) D65 (%.10f, %.10f) K75P (%.10f, %.10f)\n", d50[0], d50[1], d65[0], d65[1], k75[0], k75[1]);
    CHECK(std::fabs(d50[0] - 0.345674591) < 1e-9 && std::fabs(d50[1] - 0.3585103882) < 1e-9);
    CHECK(std::fabs(d65[0] - 0.3127205252) < 1e-9 && std::fabs(d65[1] - 0.329030685) < 1e-9);
    CHECK(std::fabs(k75[0] - 0.3152468589) < 1e-9 && std::fabs(k75[1] - 0.3328584711) < 1e-9);
    // 3x3 helpers
    res::Mat3 inv = res::mat3_inverse(pp->rgb_to_xyz);
    res::Mat3 id = res::mat3_mul(pp->rgb_to_xyz, inv);
    for (int i = 0; i < 9; ++i) CHECK(std::fabs(id[i] - (i % 4 == 0 ? 1.0 : 0.0)) < 1e-12);
    auto cat02 = res::cat_matrix(*r->blob, "CAT02");
    CHECK(cat02 && (*cat02)[0] == 0.7328);
    CHECK(!res::von_kries(d50, d50, *cat02, res::GuardRef::Target));
    auto a = res::von_kries(d50, {0.3457, 0.3585}, *cat02, res::GuardRef::Target);  // applied
    CHECK(a.has_value());
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: test_res <res> [<res_app>]\n");
        return 2;
    }
    test_blob(argv[1]);
    test_profiles(argv[1]);
    test_tables(argv[1]);
    if (argc >= 3) test_app_profiles(argv[2]);
    else std::printf("alias and black-and-white profile checks skipped: no second resources folder given\n");
    return finish("test_res");
}
