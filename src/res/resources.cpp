// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 Guangzhou Songzi Intelligent Technology Services Co., Ltd.
#include "res/resources.hpp"

#include "util/error.hpp"

namespace spk::res {

std::shared_ptr<const Resources> Resources::load(const std::filesystem::path& dir) {
    std::error_code ec;
    if (!std::filesystem::is_directory(dir, ec)) fail(SPK_ERR_IO, "resources directory '" + dir.string() + "' does not exist");
    auto r = std::make_shared<Resources>();
    r->dir = dir;
    r->blob = Blob::open(dir / kBlobFile);
    r->filters = NeutralFilterDb::load(dir / kFiltersFile);
    r->print_luts = PrintLutIndex::load(dir / kPrintLutsFile);
    r->metallib = dir / kMetallibFile;
    return r;
}

}  // namespace spk::res
