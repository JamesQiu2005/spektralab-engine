// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 Guangzhou Songzi Intelligent Technology Services Co., Ltd.

/* Everything an engine loads from its resources directory at creation.
 *
 * <resources>/spektralab_constants.bin     required (Blob)
 * <resources>/neutral_print_filters.json   optional (NeutralFilterDb)
 * <resources>/print_luts.json              optional (PrintLutIndex)
 * <resources>/spektralab.metallib          required: the kernel library
 * <resources>/profiles/<stock>.json        read on demand (load_profile)
 */
#pragma once

#include <filesystem>
#include <memory>
#include <string>

#include "res/blob.hpp"
#include "res/profile.hpp"
#include "res/tables.hpp"

namespace spk::res {

inline constexpr const char* kBlobFile = "spektralab_constants.bin";
inline constexpr const char* kFiltersFile = "neutral_print_filters.json";
inline constexpr const char* kPrintLutsFile = "print_luts.json";
inline constexpr const char* kMetallibFile = "spektralab.metallib";

struct Resources {
    std::filesystem::path dir;
    std::shared_ptr<const Blob> blob;
    NeutralFilterDb filters;
    PrintLutIndex print_luts;
    std::filesystem::path metallib;  // dir / kMetallibFile (existence checked by gpu::Context)

    // Throws spk::Error(SPK_ERR_IO) on any creation failure.
    static std::shared_ptr<const Resources> load(const std::filesystem::path& dir);

    std::shared_ptr<const Profile> profile(std::string_view id) const { return load_profile(dir, id); }
};

}  // namespace spk::res
