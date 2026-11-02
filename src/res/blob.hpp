// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 Guangzhou Songzi Intelligent Technology Services Co., Ltd.

/* Reader of spektralab_constants.bin; format in docs/RESOURCE_FORMAT.md.
 *
 * header, 16 B:   "SPKR", u32 version = 1, u32 count, u32 reserved
 * entry, 104 B:   char[64] name (NUL-padded), u32 dtype, u32 ndim, u32 dims[4],
 *                 u64 offset (8-aligned), u64 nbytes (= prod(dims) * dtype size)
 *
 * memory-mapped read-only: its pages are file-backed for the memory report
 * entries looked up by name
 * values widened to float64 on request; float32 entries also viewable in place
 */
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace spk::res {

enum class DType : uint32_t { F64 = 0, F32 = 1, F16 = 2, I32 = 3, U8 = 4 };
size_t dtype_size(DType t);
const char* dtype_name(DType t);

struct BlobEntry {
    std::string name;
    DType dtype = DType::F64;
    uint32_t ndim = 0;
    std::array<uint32_t, 4> dims{};  // unused trailing dims are 0
    uint64_t offset = 0;
    uint64_t nbytes = 0;
    bool size_ok = true;  // nbytes == prod(dims) * dtype size; a mismatch refuses the read only
    uint64_t element_count() const;
};

class Blob {
public:
    // Throws spk::Error(SPK_ERR_IO, ...) when the file is missing, truncated, has the wrong magic
    // or version, an entry runs past the end, or a required entry (kRequired) is missing or
    // malformed. The caller (engine creation) turns that into a NULL engine.
    static std::shared_ptr<const Blob> open(const std::filesystem::path& file);
    ~Blob();
    Blob(const Blob&) = delete;
    Blob& operator=(const Blob&) = delete;

    uint32_t version() const { return version_; }
    const std::vector<BlobEntry>& entries() const { return order_; }  // in file order
    const BlobEntry* find(std::string_view name) const;               // nullptr if absent
    bool has(std::string_view name) const { return find(name) != nullptr; }

    // Widened copy (every dtype). Throws spk::Error(SPK_ERR_IO) if absent or size-mismatched.
    std::vector<double> f64(std::string_view name) const;
    // Widened copy with a shape check: throws unless the entry's dims equal `shape` exactly.
    std::vector<double> f64(std::string_view name, std::initializer_list<uint32_t> shape) const;
    // Zero-copy view of a float32 entry, valid for the Blob's lifetime. Throws on another dtype.
    std::span<const float> f32_view(std::string_view name) const;
    // Raw bytes of an entry.
    std::span<const uint8_t> bytes(std::string_view name) const;

    size_t file_size() const { return size_; }

    // Entries whose absence refuses engine creation.
    static const std::vector<std::string>& required_entries();

private:
    Blob() = default;
    const BlobEntry& get(std::string_view name) const;

    const uint8_t* data_ = nullptr;
    size_t size_ = 0;
    uint32_t version_ = 0;
    std::vector<BlobEntry> order_;
    std::map<std::string, size_t, std::less<>> by_name_;
};

}  // namespace spk::res
