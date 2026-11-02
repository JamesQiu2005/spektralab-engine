// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 Guangzhou Songzi Intelligent Technology Services Co., Ltd.

/* neutral_print_filters.json, print_luts.json and the blob's print tables. */
#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>

#include "res/blob.hpp"
#include "util/json.hpp"

namespace spk::res {

// print stock -> enlarger illuminant -> film stock -> [c, m, y] (Kodak CC).
class NeutralFilterDb {
public:
    // Absent file: an empty database (every lookup misses). A file that does not parse throws
    // spk::Error(SPK_ERR_IO) (engine creation fails).
    static NeutralFilterDb load(const std::filesystem::path& file);

    // A hit needs every level present and a leaf that is an array of exactly 3 numbers.
    std::optional<std::array<double, 3>> lookup(std::string_view print_stock, std::string_view illuminant,
                                                std::string_view film_stock) const;
/* lookup_with_alias(print, illuminant, film_id, base_id)
 *   own leaf is an array: lookup(film_id); not 3 numbers -> miss, no fallback
 *   else if base_id non-empty: lookup(base_id)
 *   else miss
 */
    std::optional<std::array<double, 3>> lookup_with_alias(std::string_view print_stock,
                                                           std::string_view illuminant,
                                                           std::string_view film_id,
                                                           std::string_view base_id) const;
    bool present() const { return present_; }

private:
    json::Value root_;
    bool present_ = false;
};

struct PrintLutEntry {
    std::string stock;
    std::string paired_film;
    bool declared_pairing = false;
    uint32_t lut_size = 0;  // S > 1
    std::string output_color_space;
    bool output_cctf_encoding = true;
};

struct PrintLutTable {
    const float* table = nullptr;  // S*S*S*3 float32, row-major [i][j][k][ch], into the blob mapping
    uint32_t size = 0;             // S
    std::array<float, 3> lo{}, hi{};  // per-channel axis endpoints (print_lut_axes/<stock>)
};

class PrintLutIndex {
public:
    // Absent file: present() == false (the LUT entry points refuse; catalog serves "{}").
    // A malformed file (not an object, an entry without a string paired_film, lut_size <= 1 or
    // not an integer) throws spk::Error(SPK_ERR_IO).
    static PrintLutIndex load(const std::filesystem::path& file);

    bool present() const { return present_; }
    // The file's bytes verbatim, or "{}" when absent.
    const std::string& catalog_bytes() const { return bytes_; }
    const std::map<std::string, PrintLutEntry, std::less<>>& entries() const { return entries_; }
    const PrintLutEntry* find(std::string_view stock) const;

/* table(blob, stock) -> bit-exact view into the blob; throws SPK_ERR_USER when:
 *   no index; no entry for stock (message lists the available stocks);
 *   print_lut/<stock> is not float32 (S, S, S, 3); print_lut_axes/<stock> is not float32 (3, S);
 *   hi_c <= lo_c
 */
    PrintLutTable table(const Blob& blob, std::string_view stock) const;

private:
    bool present_ = false;
    std::string bytes_ = "{}";
    std::map<std::string, PrintLutEntry, std::less<>> entries_;
};

}  // namespace spk::res
