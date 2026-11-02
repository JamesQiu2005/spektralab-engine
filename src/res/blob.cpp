// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 Guangzhou Songzi Intelligent Technology Services Co., Ltd.
#include "res/blob.hpp"

#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "util/error.hpp"
#include "util/numeric.hpp"

namespace spk::res {

namespace {

constexpr size_t kHeader = 16;
constexpr size_t kEntry = 104;

uint32_t rd_u32(const uint8_t* p) {
    return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
}
uint64_t rd_u64(const uint8_t* p) { return uint64_t(rd_u32(p)) | (uint64_t(rd_u32(p + 4)) << 32); }

[[noreturn]] void io_fail(const std::string& m) { fail(SPK_ERR_IO, "constants blob: " + m); }

}  // namespace

size_t dtype_size(DType t) {
    switch (t) {
        case DType::F64: return 8;
        case DType::F32: return 4;
        case DType::F16: return 2;
        case DType::I32: return 4;
        case DType::U8: return 1;
    }
    return 0;
}

const char* dtype_name(DType t) {
    switch (t) {
        case DType::F64: return "float64";
        case DType::F32: return "float32";
        case DType::F16: return "float16";
        case DType::I32: return "int32";
        case DType::U8: return "uint8";
    }
    return "?";
}

uint64_t BlobEntry::element_count() const {
    uint64_t n = 1;
    for (uint32_t i = 0; i < ndim; ++i) n *= dims[i];
    return n;
}

const std::vector<std::string>& Blob::required_entries() {
    static const std::vector<std::string> names = {
        "spectral/wavelengths",
        "colour/cmfs_1931_2deg",
        "colour/cmfs_lms",
        "colour/matrix_16",
        "colour/mallett_srgb_basis",
        "hanatos/spectra_lut",
    };
    return names;
}

/* Blob::open(file)
 *   map the file; check magic, version == 1, table fits
 *   for each entry: name NUL-terminated, dtype <= 4, ndim in 1..4, unused dims 0,
 *                   offset 8-aligned, data inside the file
 *                   size_ok := prod(dims) * dtype size == nbytes   -- a mismatch refuses only the read
 *   each required entry present and size_ok
 *   spectral/wavelengths: 1-D, 81 long
 *   at least one colour/cat/<name> entry
 *   any failure: SPK_ERR_IO
 */
std::shared_ptr<const Blob> Blob::open(const std::filesystem::path& file) {
    int fd = ::open(file.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) io_fail("cannot open '" + file.string() + "'");
    struct stat st {};
    if (fstat(fd, &st) != 0) { ::close(fd); io_fail("cannot stat '" + file.string() + "'"); }
    const size_t size = size_t(st.st_size);
    if (size < kHeader) { ::close(fd); io_fail("truncated header"); }
    void* m = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (m == MAP_FAILED) io_fail("cannot map '" + file.string() + "'");

    std::shared_ptr<Blob> b(new Blob());
    b->data_ = static_cast<const uint8_t*>(m);
    b->size_ = size;
    const uint8_t* d = b->data_;

    if (std::memcmp(d, "SPKR", 4) != 0) io_fail("wrong magic");
    b->version_ = rd_u32(d + 4);
    if (b->version_ != 1) io_fail("unsupported version " + std::to_string(b->version_));
    const uint32_t count = rd_u32(d + 8);
    if (kHeader + uint64_t(count) * kEntry > size) io_fail("truncated entry table");

    b->order_.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        const uint8_t* e = d + kHeader + size_t(i) * kEntry;
        BlobEntry en;
        size_t len = 0;
        while (len < 64 && e[len] != 0) ++len;
        if (len == 64) io_fail("entry " + std::to_string(i) + ": name not NUL-terminated");
        en.name.assign(reinterpret_cast<const char*>(e), len);
        uint32_t dt = rd_u32(e + 64);
        if (dt > 4) io_fail("entry '" + en.name + "': unknown dtype " + std::to_string(dt));
        en.dtype = static_cast<DType>(dt);
        en.ndim = rd_u32(e + 68);
        if (en.ndim < 1 || en.ndim > 4) io_fail("entry '" + en.name + "': ndim out of 1..4");
        for (int k = 0; k < 4; ++k) en.dims[size_t(k)] = rd_u32(e + 72 + 4 * k);
        for (uint32_t k = en.ndim; k < 4; ++k)
            if (en.dims[k] != 0) io_fail("entry '" + en.name + "': non-zero unused dimension");
        en.offset = rd_u64(e + 88);
        en.nbytes = rd_u64(e + 96);
        if (en.offset % 8 != 0) io_fail("entry '" + en.name + "': offset not 8-aligned");
        if (en.offset > size || en.nbytes > size - en.offset) io_fail("entry '" + en.name + "' runs past the end of the file");
        en.size_ok = en.element_count() * dtype_size(en.dtype) == en.nbytes;
        b->by_name_[en.name] = b->order_.size();
        b->order_.push_back(std::move(en));
    }

    for (const std::string& r : required_entries()) {
        const BlobEntry* en = b->find(r);
        if (!en) io_fail("required entry '" + r + "' is missing");
        if (!en->size_ok) io_fail("required entry '" + r + "' has a size that does not match its shape");
    }
    const BlobEntry* wl = b->find("spectral/wavelengths");
    if (wl->ndim != 1 || wl->dims[0] != 81) io_fail("'spectral/wavelengths' must have 81 samples");
    bool any_cat = false;
    for (const BlobEntry& en : b->order_)
        if (en.name.rfind("colour/cat/", 0) == 0) any_cat = true;
    if (!any_cat) io_fail("no 'colour/cat/*' entry");
    return b;
}

Blob::~Blob() {
    if (data_) munmap(const_cast<uint8_t*>(data_), size_);
}

const BlobEntry* Blob::find(std::string_view name) const {
    auto it = by_name_.find(name);
    return it == by_name_.end() ? nullptr : &order_[it->second];
}

const BlobEntry& Blob::get(std::string_view name) const {
    const BlobEntry* e = find(name);
    if (!e) io_fail("no entry '" + std::string(name) + "'");
    if (!e->size_ok) io_fail("entry '" + std::string(name) + "' has a size that does not match its shape");
    return *e;
}

std::span<const uint8_t> Blob::bytes(std::string_view name) const {
    const BlobEntry& e = get(name);
    return {data_ + e.offset, size_t(e.nbytes)};
}

/* f64(name): little-endian decode to float64; float16 through half_to_double */
std::vector<double> Blob::f64(std::string_view name) const {
    const BlobEntry& e = get(name);
    const uint8_t* p = data_ + e.offset;
    const size_t n = size_t(e.element_count());
    std::vector<double> out(n);
    switch (e.dtype) {
        case DType::F64:
            std::memcpy(out.data(), p, n * 8);
            break;
        case DType::F32:
            for (size_t i = 0; i < n; ++i) {
                float f;
                std::memcpy(&f, p + 4 * i, 4);
                out[i] = double(f);
            }
            break;
        case DType::F16:
            for (size_t i = 0; i < n; ++i) out[i] = num::half_to_double(uint16_t(p[2 * i] | (p[2 * i + 1] << 8)));
            break;
        case DType::I32:
            for (size_t i = 0; i < n; ++i) out[i] = double(int32_t(rd_u32(p + 4 * i)));
            break;
        case DType::U8:
            for (size_t i = 0; i < n; ++i) out[i] = double(p[i]);
            break;
    }
    return out;
}

std::vector<double> Blob::f64(std::string_view name, std::initializer_list<uint32_t> shape) const {
    const BlobEntry& e = get(name);
    bool ok = shape.size() == e.ndim;
    size_t k = 0;
    for (uint32_t s : shape) {
        if (k < 4 && e.dims[k] != s) ok = false;
        ++k;
    }
    if (!ok) io_fail("entry '" + std::string(name) + "' has an unexpected shape");
    return f64(name);
}

std::span<const float> Blob::f32_view(std::string_view name) const {
    const BlobEntry& e = get(name);
    if (e.dtype != DType::F32) io_fail("entry '" + std::string(name) + "' is not float32");
    return {reinterpret_cast<const float*>(data_ + e.offset), size_t(e.element_count())};
}

}  // namespace spk::res
