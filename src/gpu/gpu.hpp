// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 Guangzhou Songzi Intelligent Technology Services Co., Ltd.

/* A thin Metal bridge. Pure C++ API; the implementation is gpu.mm (Objective-C++, ARC).
 * Metal objects cross this header only as opaque void*.
 *
 * Buffer   MTLStorageModeShared: CPU reads and writes contents() once its batch has completed
 *          reference-counted handle; pooled -> returns to its Pool; persistent -> leaves the ledger
 * Batch    keeps every Buffer it binds alive until completed
 * Kernel   looked up by function name in the loaded metallib; pipelines cached
 * failure  spk::Error(SPK_ERR_GPU, ...)
 * threads  Context, Pool and the pipeline cache lock internally; one Batch, one thread
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <initializer_list>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace spk::gpu {

struct BufferImpl;
struct ContextImpl;
struct PoolState;
struct LedgerState;
struct ArgAccess;
class Context;
class Batch;

// What a persistent allocation holds, for the memory report.
enum class MemTag : uint8_t { Source, TierImage, Negative, MeterImage, Table, Other };

class Buffer {
public:
    Buffer() = default;
    explicit operator bool() const { return p_ != nullptr; }
    void* contents() const;  // CPU pointer (shared storage)
    size_t size() const;     // bytes
    void* mtl() const;       // id<MTLBuffer>, not retained
    bool pooled() const;
    template <class T>
    T* as() const { return static_cast<T*>(contents()); }

private:
    friend class Context;
    friend class Batch;
    friend class Pool;
    std::shared_ptr<BufferImpl> p_;
};

// A three-channel interleaved float32 image (R,G,B per pixel), row-major, top row first.
// This is the one plane representation of the engine.
struct Plane {
    Buffer buf;
    uint32_t w = 0, h = 0;
    static constexpr uint32_t kChannels = 3;

    explicit operator bool() const { return static_cast<bool>(buf); }
    size_t pixels() const { return size_t(w) * h; }
    size_t floats() const { return pixels() * kChannels; }
    size_t bytes() const { return floats() * sizeof(float); }
    float* data() const { return buf.as<float>(); }
    float* px(uint32_t x, uint32_t y) const { return data() + (size_t(y) * w + x) * kChannels; }
};

struct PoolStats {
    uint64_t total_bytes = 0, live_bytes = 0, free_bytes = 0, pending_bytes = 0;
    uint64_t buffers = 0, live_buffers = 0, free_buffers = 0, pending_buffers = 0;
    uint64_t frame_high_water_bytes = 0;
    // audit counters; the last three must stay 0
    uint64_t allocations = 0, reuses = 0, persistent_allocations = 0;
    uint64_t reuse_bytes_requested = 0, reuse_bytes_taken = 0, reuse_max_taken = 0,
             reuse_max_taken_for_request = 0, pending_held = 0;
    uint64_t live_underflows = 0, over_releases = 0, reclaim_while_encoding = 0;
};

struct PersistentStats {
    uint64_t bytes = 0, buffers = 0;
    uint64_t file_backed_bytes = 0, file_backed_buffers = 0;
    uint64_t by_tag[6] = {};  // indexed by MemTag
};

/* Pool: scratch buffers
 *   acquire(bytes): the smallest free buffer of size in [bytes, 2 bytes], else allocate
 *   dropping the last handle makes the buffer free again
 */
class Pool {
public:
    Buffer acquire(size_t bytes);
    Plane plane(uint32_t w, uint32_t h);  // contents are undefined
/* trim(keep_bytes): drop free buffers, largest first, until free bytes <= keep_bytes; -> bytes freed */
    size_t trim(size_t keep_bytes = 0);
    void begin_frame();  // resets frame_high_water_bytes to the current live bytes
    PoolStats stats() const;

private:
    friend class Context;
    std::shared_ptr<PoolState> s_;
    void* device_ = nullptr;
};

class Kernel {
public:
    const std::string& name() const { return name_; }
    void* pipeline() const { return pso_; }  // id<MTLComputePipelineState>
    uint32_t thread_execution_width() const { return tew_; }
    uint32_t max_threads_per_group() const { return max_threads_; }
    explicit operator bool() const { return pso_ != nullptr; }

private:
    friend class Context;
    std::string name_;
    void* pso_ = nullptr;  // owned by the Context's cache (lives as long as the Context)
    uint32_t tew_ = 0, max_threads_ = 0;
};

// One kernel argument, bound at index = its position in the argument list.
class Arg {
public:
    static Arg buffer(const Buffer& b, size_t offset = 0) { Arg a; a.kind_ = Kind::Buf; a.buf_ = b; a.off_ = offset; return a; }
    static Arg plane(const Plane& p) { return buffer(p.buf); }
    static Arg bytes(const void* p, size_t n) {
        Arg a;
        a.kind_ = Kind::Bytes;
        a.bytes_.assign(static_cast<const uint8_t*>(p), static_cast<const uint8_t*>(p) + n);
        return a;
    }
    template <class T>
    static Arg value(const T& v) { return bytes(&v, sizeof(T)); }

private:
    friend class Batch;
    friend struct ArgAccess;
    enum class Kind { Buf, Bytes } kind_ = Kind::Bytes;
    Buffer buf_;
    size_t off_ = 0;
    std::vector<uint8_t> bytes_;  // set-bytes payload (<= 4 KiB)
};

// A command buffer with one serial compute encoder. Dispatches run in encode order.
class Batch {
public:
    Batch(Batch&&) noexcept;
    Batch& operator=(Batch&&) noexcept;
    ~Batch();  // an uncommitted batch is committed and waited for (errors swallowed)

    void dispatch_1d(const Kernel& k, uint32_t n, std::initializer_list<Arg> args);
    void dispatch_2d(const Kernel& k, uint32_t w, uint32_t h, std::initializer_list<Arg> args);
    void dispatch_1d(const Kernel& k, uint32_t n, const std::vector<Arg>& args);
    void dispatch_2d(const Kernel& k, uint32_t w, uint32_t h, const std::vector<Arg>& args);
    // Commits and blocks until completion; throws spk::Error(SPK_ERR_GPU) on a command error.
    void commit_and_wait();
    bool empty() const;

    struct Impl;  // opaque (gpu.mm)

private:
    friend class Context;
    Batch() = default;
    std::unique_ptr<Impl> p_;
};

/* OutputImage: an RGBA16Unorm linear texture over a fresh shared buffer
 *   texture is +1 and handed to the caller; rgba16 is the buffer's contents
 */
struct OutputImage {
    void* texture = nullptr;  // id<MTLTexture>, +1
    uint16_t* rgba16 = nullptr;
    uint32_t width = 0, height = 0, row_stride_px = 0;
};

struct ProbeResult {
    bool ok = false;           // true only for the safe verdict
    std::string verdict;       // backend.math_mode string
    std::string detail;        // why it was refused, for the creation error message
};

class Context {
public:
    // device: id<MTLDevice> or nullptr for the system default. Loads `metallib`.
    // Throws spk::Error(SPK_ERR_GPU) when no device, no library, or the library cannot be read.
    static std::shared_ptr<Context> create(void* device, const std::filesystem::path& metallib);
    ~Context();
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;

    void* device() const;  // id<MTLDevice>
    std::string device_name() const;
    // The device's real 2-D texture limit from Apple's feature-set tables.
    uint32_t max_texture_dimension_2d() const;
    // minimumLinearTextureAlignmentForPixelFormat:RGBA16Unorm, in bytes.
    size_t linear_texture_alignment_rgba16() const;

    // Cached compute pipeline for a kernel function; throws spk::Error(SPK_ERR_GPU) if missing.
    Kernel kernel(std::string_view name);

    Batch batch();

    Pool& pool() { return pool_; }
    // Allocations outside the pool (sources, tier images, negatives, tables).
    Buffer persistent(size_t bytes, MemTag tag);
    Plane persistent_plane(uint32_t w, uint32_t h, MemTag tag);
    PersistentStats persistent_stats() const;
    // Records a file-backed region (the mapped constants blob) in the persistent ledger.
    void note_file_backed(size_t bytes);

/* wrap(mtl_buffer): borrowed caller buffer (spk_open_device); not counted, not retained past the handle
 *   the session checks same_device() first
 */
    Buffer wrap(void* mtl_buffer);
    bool same_device(void* mtl_buffer) const;
    static size_t buffer_length(void* mtl_buffer);

    // Allocates the output buffer and texture for one result.
    OutputImage make_output(uint32_t width, uint32_t height);
/* finish_output(out): after the batch that wrote out.rgba16 has completed
 *   device:        no-op, the texture is the buffer
 *   iOS simulator: blit the buffer into the private texture
 */
    void finish_output(const OutputImage& out);
    // Releases a texture handed out by make_output (spk_result_free).
    static void release_texture(void* texture);

    // Runs the numeric-mode probe kernel and checks it against the host.
    ProbeResult probe_math_mode();

    static constexpr const char* kSafeVerdict = "safe (probe: fma contraction present, no reassociation)";

private:
    Context() = default;
    std::unique_ptr<ContextImpl> p_;
    Pool pool_;
    std::shared_ptr<LedgerState> ledger_;
};

}  // namespace spk::gpu
