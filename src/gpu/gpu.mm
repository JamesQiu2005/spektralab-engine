// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 Guangzhou Songzi Intelligent Technology Services Co., Ltd.

/* Objective-C++ implementation of gpu.hpp, compiled with -fobjc-arc.
 *
 * Metal objects live in __strong members of the Impl structs below; ARC releases them.
 * The one manual retain: the output texture handed to the host
 *   +1 through __bridge_retained; released by Context::release_texture with CFRelease.
 */
#include "gpu/gpu.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <TargetConditionals.h>
#include <objc/runtime.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <mutex>

#include "util/error.hpp"

namespace spk::gpu {

#if TARGET_OS_SIMULATOR
// Associated-object keys compare by address: one object, used by both make_output and finish_output.
static const char kRgba16BufferKey = 0;
#endif

// ---- internal state ----------------------------------------------------------------------------

struct LedgerState {
    std::mutex m;
    PersistentStats st;
};

struct PoolState {
    std::mutex m;
    id<MTLDevice> device = nil;
    std::multimap<size_t, id<MTLBuffer>> free;  // size -> buffer
    PoolStats st;
};

struct BufferImpl {
    id<MTLBuffer> buf = nil;
    size_t size = 0;
    std::shared_ptr<PoolState> pool;      // set for pooled buffers
    std::shared_ptr<LedgerState> ledger;  // set for persistent buffers
    MemTag tag = MemTag::Other;

    /* last handle gone:
     *   pooled:     live -= size; back onto the free list
     *   persistent: leave the ledger
     */
    ~BufferImpl() {
        if (pool) {
            std::lock_guard<std::mutex> g(pool->m);
            PoolStats& s = pool->st;
            if (s.live_bytes < size || s.live_buffers == 0) {
                ++s.live_underflows;
            } else {
                s.live_bytes -= size;
                --s.live_buffers;
            }
            pool->free.emplace(size, buf);
            s.free_bytes += size;
            ++s.free_buffers;
        } else if (ledger) {
            std::lock_guard<std::mutex> g(ledger->m);
            PersistentStats& s = ledger->st;
            s.bytes -= size;
            --s.buffers;
            s.by_tag[size_t(tag)] -= size;
        }
    }
};

struct ContextImpl {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLLibrary> library = nil;
    std::mutex m;
    std::map<std::string, id<MTLComputePipelineState>, std::less<>> psos;
};

struct ArgAccess {
    static bool is_buffer(const Arg& a) { return a.kind_ == Arg::Kind::Buf; }
    static const Buffer& buffer(const Arg& a) { return a.buf_; }
    static size_t offset(const Arg& a) { return a.off_; }
    static const std::vector<uint8_t>& bytes(const Arg& a) { return a.bytes_; }
};

namespace {

[[noreturn]] void gpu_fail(const std::string& m) { fail(SPK_ERR_GPU, m); }

std::string ns_error_text(NSError* e) {
    if (!e) return "unknown error";
    const char* s = [[e localizedDescription] UTF8String];
    return s ? std::string(s) : std::string("unknown error");
}

id<MTLLibrary> load_library(id<MTLDevice> dev, const std::filesystem::path& p) {
    std::error_code ec;
    if (!std::filesystem::exists(p, ec)) gpu_fail("kernel library '" + p.string() + "' not found");
    NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:p.c_str()]];
    NSError* err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithURL:url error:&err];
    if (!lib) gpu_fail("cannot load kernel library '" + p.string() + "': " + ns_error_text(err));
    return lib;
}

}  // namespace

// ---- Buffer --------------------------------------------------------------------------------------

void* Buffer::contents() const { return p_ ? [p_->buf contents] : nullptr; }
size_t Buffer::size() const { return p_ ? p_->size : 0; }
void* Buffer::mtl() const { return p_ ? (__bridge void*)p_->buf : nullptr; }
bool Buffer::pooled() const { return p_ && p_->pool; }

// ---- Pool ----------------------------------------------------------------------------------------

Buffer Pool::acquire(size_t bytes) {
    if (!s_) gpu_fail("buffer pool used before the GPU context exists");
    const size_t want = std::max<size_t>(bytes, 16);
    auto impl = std::make_shared<BufferImpl>();
    {
        std::lock_guard<std::mutex> g(s_->m);
        PoolStats& s = s_->st;
        auto it = s_->free.lower_bound(want);
        if (it != s_->free.end() && it->first <= 2 * want) {
            impl->buf = it->second;
            impl->size = it->first;
            s_->free.erase(it);
            s.free_bytes -= impl->size;
            --s.free_buffers;
            ++s.reuses;
            s.reuse_bytes_requested += want;
            s.reuse_bytes_taken += impl->size;
            if (impl->size > s.reuse_max_taken) {
                s.reuse_max_taken = impl->size;
                s.reuse_max_taken_for_request = want;
            }
        } else {
            id<MTLBuffer> b = [s_->device newBufferWithLength:want options:MTLResourceStorageModeShared];
            if (!b) gpu_fail("cannot allocate a " + std::to_string(want) + "-byte scratch buffer");
            impl->buf = b;
            impl->size = want;
            ++s.allocations;
        }
        s.live_bytes += impl->size;
        ++s.live_buffers;
        s.frame_high_water_bytes = std::max(s.frame_high_water_bytes, s.live_bytes);
        impl->pool = s_;
    }
    Buffer out;
    out.p_ = std::move(impl);
    return out;
}

Plane Pool::plane(uint32_t w, uint32_t h) {
    Plane p;
    p.w = w;
    p.h = h;
    p.buf = acquire(size_t(w) * h * Plane::kChannels * sizeof(float));
    return p;
}

size_t Pool::trim(size_t keep_bytes) {
    if (!s_) return 0;
    std::lock_guard<std::mutex> g(s_->m);
    PoolStats& s = s_->st;
    size_t freed = 0;
    // Drop the largest free buffers first until at most keep_bytes of free memory remain.
    while (!s_->free.empty() && s.free_bytes > keep_bytes) {
        auto it = std::prev(s_->free.end());
        freed += it->first;
        s.free_bytes -= it->first;
        --s.free_buffers;
        s_->free.erase(it);  // ARC releases the MTLBuffer
    }
    return freed;
}

void Pool::begin_frame() {
    if (!s_) return;
    std::lock_guard<std::mutex> g(s_->m);
    s_->st.frame_high_water_bytes = s_->st.live_bytes;
}

PoolStats Pool::stats() const {
    if (!s_) return {};
    std::lock_guard<std::mutex> g(s_->m);
    PoolStats s = s_->st;
    s.total_bytes = s.live_bytes + s.free_bytes + s.pending_bytes;
    s.buffers = s.live_buffers + s.free_buffers + s.pending_buffers;
    return s;
}

// ---- Batch ---------------------------------------------------------------------------------------

struct Batch::Impl {
    id<MTLCommandBuffer> cb = nil;
    id<MTLComputeCommandEncoder> enc = nil;
    std::vector<Buffer> keep;
    size_t dispatches = 0;
    bool done = false;
};

Batch::Batch(Batch&&) noexcept = default;
Batch& Batch::operator=(Batch&&) noexcept = default;

Batch::~Batch() {
    if (p_ && !p_->done) {
        try {
            commit_and_wait();
        } catch (...) {
        }
    }
}

bool Batch::empty() const { return !p_ || p_->dispatches == 0; }

namespace {

template <class It>
void bind(id<MTLComputeCommandEncoder> enc, std::vector<Buffer>& keep, It b, It e) {
    NSUInteger i = 0;
    for (It it = b; it != e; ++it, ++i) {
        const Arg& a = *it;
        if (ArgAccess::is_buffer(a)) {
            const Buffer& buf = ArgAccess::buffer(a);
            if (!buf) gpu_fail("kernel argument " + std::to_string(i) + " is a null buffer");
            [enc setBuffer:(__bridge id<MTLBuffer>)buf.mtl() offset:ArgAccess::offset(a) atIndex:i];
            keep.push_back(buf);
        } else {
            const auto& by = ArgAccess::bytes(a);
            if (by.empty() || by.size() > 4096) gpu_fail("kernel argument " + std::to_string(i) + ": set-bytes size must be 1..4096");
            [enc setBytes:by.data() length:by.size() atIndex:i];
        }
    }
}

}  // namespace

/* dispatch_impl(batch, kernel, grid, args)
 *   grid empty: nothing
 *   open the serial compute encoder on first use
 *   bind args at index = position; keep buffers alive until completion
 *   thread group: 1-D min(max, 256) wide; 2-D execution width x min(max / width, 8)
 *   dispatchThreads
 */
static void dispatch_impl(Batch::Impl* p, const Kernel& k, MTLSize grid, const Arg* b, const Arg* e) {
    if (!p || p->done) gpu_fail("dispatch on a committed batch");
    if (!k) gpu_fail("dispatch of an empty kernel");
    if (grid.width == 0 || grid.height == 0) return;
    if (!p->enc) {
        p->enc = [p->cb computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
        if (!p->enc) gpu_fail("cannot create a compute encoder");
    }
    id<MTLComputePipelineState> pso = (__bridge id<MTLComputePipelineState>)k.pipeline();
    [p->enc setComputePipelineState:pso];
    bind(p->enc, p->keep, b, e);
    const NSUInteger tew = std::max<NSUInteger>(1, k.thread_execution_width());
    const NSUInteger maxt = std::max<NSUInteger>(1, k.max_threads_per_group());
    MTLSize tg;
    if (grid.height == 1) {
        tg = MTLSizeMake(std::min<NSUInteger>(maxt, 256), 1, 1);
    } else {
        tg = MTLSizeMake(tew, std::max<NSUInteger>(1, std::min<NSUInteger>(maxt / tew, 8)), 1);
    }
    [p->enc dispatchThreads:grid threadsPerThreadgroup:tg];
    ++p->dispatches;
}

void Batch::dispatch_1d(const Kernel& k, uint32_t n, std::initializer_list<Arg> args) {
    dispatch_impl(p_.get(), k, MTLSizeMake(n, 1, 1), args.begin(), args.end());
}
void Batch::dispatch_2d(const Kernel& k, uint32_t w, uint32_t h, std::initializer_list<Arg> args) {
    dispatch_impl(p_.get(), k, MTLSizeMake(w, h, 1), args.begin(), args.end());
}
void Batch::dispatch_1d(const Kernel& k, uint32_t n, const std::vector<Arg>& args) {
    dispatch_impl(p_.get(), k, MTLSizeMake(n, 1, 1), args.data(), args.data() + args.size());
}
void Batch::dispatch_2d(const Kernel& k, uint32_t w, uint32_t h, const std::vector<Arg>& args) {
    dispatch_impl(p_.get(), k, MTLSizeMake(w, h, 1), args.data(), args.data() + args.size());
}

void Batch::commit_and_wait() {
    if (!p_ || p_->done) return;
    p_->done = true;
    if (p_->enc) [p_->enc endEncoding];
    [p_->cb commit];
    [p_->cb waitUntilCompleted];
    p_->keep.clear();
    if (p_->cb.status == MTLCommandBufferStatusError) gpu_fail("GPU command failed: " + ns_error_text(p_->cb.error));
}

// ---- Context -------------------------------------------------------------------------------------

std::shared_ptr<Context> Context::create(void* device, const std::filesystem::path& metallib) {
    std::shared_ptr<Context> c(new Context());
    c->p_ = std::make_unique<ContextImpl>();
    @autoreleasepool {
        id<MTLDevice> dev = device ? (__bridge id<MTLDevice>)device : MTLCreateSystemDefaultDevice();
        if (!dev) gpu_fail("no Metal device");
        c->p_->device = dev;
        c->p_->queue = [dev newCommandQueue];
        if (!c->p_->queue) gpu_fail("cannot create a Metal command queue");
        c->p_->library = load_library(dev, metallib);
    }
    c->pool_.s_ = std::make_shared<PoolState>();
    c->pool_.s_->device = c->p_->device;
    c->pool_.device_ = (__bridge void*)c->p_->device;
    c->ledger_ = std::make_shared<LedgerState>();
    return c;
}

Context::~Context() = default;

void* Context::device() const { return (__bridge void*)p_->device; }

std::string Context::device_name() const {
    const char* s = [[p_->device name] UTF8String];
    return s ? s : "";
}

/* max_texture_dimension_2d(): 16384 for family Apple3 or later and Mac2; otherwise 8192
 *   Metal has no query for this; values from Apple's Metal feature-set tables
 */
uint32_t Context::max_texture_dimension_2d() const {
    id<MTLDevice> d = p_->device;
    if ([d supportsFamily:MTLGPUFamilyApple3] || [d supportsFamily:MTLGPUFamilyMac2]) return 16384;
    return 8192;
}

size_t Context::linear_texture_alignment_rgba16() const {
    return [p_->device minimumLinearTextureAlignmentForPixelFormat:MTLPixelFormatRGBA16Unorm];
}

Kernel Context::kernel(std::string_view name) {
    std::lock_guard<std::mutex> g(p_->m);
    auto it = p_->psos.find(name);
    if (it == p_->psos.end()) {
        @autoreleasepool {
            NSString* nm = [[NSString alloc] initWithBytes:name.data() length:name.size() encoding:NSUTF8StringEncoding];
            id<MTLFunction> fn = [p_->library newFunctionWithName:nm];
            if (!fn) gpu_fail("kernel '" + std::string(name) + "' is not in the kernel library");
            NSError* err = nil;
            id<MTLComputePipelineState> pso = [p_->device newComputePipelineStateWithFunction:fn error:&err];
            if (!pso) gpu_fail("cannot build kernel '" + std::string(name) + "': " + ns_error_text(err));
            it = p_->psos.emplace(std::string(name), pso).first;
        }
    }
    Kernel k;
    k.name_ = std::string(name);
    k.pso_ = (__bridge void*)it->second;
    k.tew_ = uint32_t([it->second threadExecutionWidth]);
    k.max_threads_ = uint32_t([it->second maxTotalThreadsPerThreadgroup]);
    return k;
}

Batch Context::batch() {
    Batch b;
    b.p_ = std::make_unique<Batch::Impl>();
    b.p_->cb = [p_->queue commandBuffer];
    if (!b.p_->cb) gpu_fail("cannot create a command buffer");
    return b;
}

Buffer Context::persistent(size_t bytes, MemTag tag) {
    const size_t want = std::max<size_t>(bytes, 16);
    id<MTLBuffer> b = [p_->device newBufferWithLength:want options:MTLResourceStorageModeShared];
    if (!b) gpu_fail("cannot allocate a " + std::to_string(want) + "-byte buffer");
    auto impl = std::make_shared<BufferImpl>();
    impl->buf = b;
    impl->size = want;
    impl->ledger = ledger_;
    impl->tag = tag;
    {
        std::lock_guard<std::mutex> g(ledger_->m);
        ledger_->st.bytes += want;
        ++ledger_->st.buffers;
        ledger_->st.by_tag[size_t(tag)] += want;
    }
    {
        std::lock_guard<std::mutex> g(pool_.s_->m);
        ++pool_.s_->st.persistent_allocations;
    }
    Buffer out;
    out.p_ = std::move(impl);
    return out;
}

Plane Context::persistent_plane(uint32_t w, uint32_t h, MemTag tag) {
    Plane p;
    p.w = w;
    p.h = h;
    p.buf = persistent(size_t(w) * h * Plane::kChannels * sizeof(float), tag);
    return p;
}

PersistentStats Context::persistent_stats() const {
    std::lock_guard<std::mutex> g(ledger_->m);
    return ledger_->st;
}

void Context::note_file_backed(size_t bytes) {
    std::lock_guard<std::mutex> g(ledger_->m);
    ledger_->st.bytes += bytes;
    ++ledger_->st.buffers;
    ledger_->st.file_backed_bytes += bytes;
    ++ledger_->st.file_backed_buffers;
    ledger_->st.by_tag[size_t(MemTag::Table)] += bytes;
}

Buffer Context::wrap(void* mtl_buffer) {
    if (!mtl_buffer) gpu_fail("null MTLBuffer");
    auto impl = std::make_shared<BufferImpl>();
    impl->buf = (__bridge id<MTLBuffer>)mtl_buffer;
    impl->size = [impl->buf length];
    Buffer out;
    out.p_ = std::move(impl);
    return out;
}

bool Context::same_device(void* mtl_buffer) const {
    if (!mtl_buffer) return false;
    id<MTLBuffer> b = (__bridge id<MTLBuffer>)mtl_buffer;
    return [b device] == p_->device;
}

size_t Context::buffer_length(void* mtl_buffer) {
    if (!mtl_buffer) return 0;
    return [(__bridge id<MTLBuffer>)mtl_buffer length];
}

/* make_output(width, height)
 *   A := max(alignment(RGBA16Unorm), 8) / 8, in pixels
 *   stride := ceil(width / A) A; buffer := fresh shared, stride * 8 * height bytes
 *   device:    linear texture over the buffer, bytesPerRow = stride * 8
 *   simulator: private texture, the buffer kept alive by association; finish_output blits
 */
OutputImage Context::make_output(uint32_t width, uint32_t height) {
    if (width == 0 || height == 0) gpu_fail("empty output image");
    const size_t align_bytes = std::max<size_t>(linear_texture_alignment_rgba16(), 8);
    const size_t a = align_bytes / 8;  // pixels
    const size_t stride = (size_t(width) + a - 1) / a * a;
    const size_t bytes = stride * 8 * height;
    OutputImage out;
    @autoreleasepool {
        id<MTLBuffer> buf = [p_->device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        if (!buf) gpu_fail("cannot allocate the output buffer");
        MTLTextureDescriptor* d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Unorm
                                                                                     width:width
                                                                                    height:height
                                                                                 mipmapped:NO];
        d.usage = MTLTextureUsageShaderRead;
        id<MTLTexture> tex = nil;
#if TARGET_OS_SIMULATOR
        d.storageMode = MTLStorageModePrivate;
        tex = [p_->device newTextureWithDescriptor:d];
        if (tex) objc_setAssociatedObject(tex, &kRgba16BufferKey, buf, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
#else
        d.storageMode = MTLStorageModeShared;
        tex = [buf newTextureWithDescriptor:d offset:0 bytesPerRow:stride * 8];
#endif
        if (!tex) gpu_fail("cannot create the output texture");
        out.rgba16 = static_cast<uint16_t*>([buf contents]);
        out.texture = (__bridge_retained void*)tex;  // +1 for the host
    }
    out.width = width;
    out.height = height;
    out.row_stride_px = uint32_t(stride);
    return out;
}

void Context::finish_output(const OutputImage& out) {
#if TARGET_OS_SIMULATOR
    @autoreleasepool {
        id<MTLTexture> tex = (__bridge id<MTLTexture>)out.texture;
        id<MTLBuffer> buf = objc_getAssociatedObject(tex, &kRgba16BufferKey);
        if (!tex || !buf) gpu_fail("finish_output: not an engine output texture");
        id<MTLCommandBuffer> cb = [p_->queue commandBuffer];
        id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder];
        [bl copyFromBuffer:buf
                   sourceOffset:0
              sourceBytesPerRow:size_t(out.row_stride_px) * 8
            sourceBytesPerImage:size_t(out.row_stride_px) * 8 * out.height
                     sourceSize:MTLSizeMake(out.width, out.height, 1)
                      toTexture:tex
               destinationSlice:0
               destinationLevel:0
              destinationOrigin:MTLOriginMake(0, 0, 0)];
        [bl endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) gpu_fail("output blit failed: " + ns_error_text(cb.error));
    }
#else
    (void)out;
#endif
}

void Context::release_texture(void* texture) {
    if (texture) CFRelease(texture);
}

/* probe_math_mode(): verify the metallib's math mode on the GPU
 *   a[i] := float32(0.1 + 0.37 i); b[i] := 0.5 a[i] + 1, i = 0..63
 *   out[i] := spk_probe_fma(a, b)                    -- a b - a b on the GPU
 *   host[i] := fmaf(a, b, -(a b))
 *   all out == host: safe                            -- fma contraction, no reassociation
 *   all out == 0:    fast math, refuse
 *   otherwise:       unknown, refuse
 */
ProbeResult Context::probe_math_mode() {
    constexpr uint32_t n = 64;
    ProbeResult r;
    Buffer a = pool_.acquire(n * sizeof(float));
    Buffer b = pool_.acquire(n * sizeof(float));
    Buffer o = pool_.acquire(n * sizeof(float));
    float* pa = a.as<float>();
    float* pb = b.as<float>();
    for (uint32_t i = 0; i < n; ++i) {
        pa[i] = static_cast<float>(0.1 + 0.37 * double(i));
        pb[i] = 0.5f * pa[i] + 1.0f;
    }
    std::memset(o.contents(), 0xFF, n * sizeof(float));
    Kernel k = kernel("spk_probe_fma");
    Batch bt = batch();
    bt.dispatch_1d(k, n, {Arg::buffer(a), Arg::buffer(b), Arg::buffer(o), Arg::value(n)});
    bt.commit_and_wait();
    const float* po = o.as<float>();
    uint32_t zeros = 0, match = 0;
    for (uint32_t i = 0; i < n; ++i) {
        const float host = std::fmaf(pa[i], pb[i], -(pa[i] * pb[i]));
        if (po[i] == 0.0f) ++zeros;
        if (std::memcmp(&po[i], &host, sizeof(float)) == 0 || (po[i] == host)) ++match;
    }
    if (match == n) {
        r.ok = true;
        r.verdict = kSafeVerdict;
    } else if (zeros == n) {
        r.verdict = "fast (probe: a*b-a*b is exactly 0, no fma contraction)";
        r.detail = "the kernels were compiled with fast math; rebuild them with -fmetal-math-mode=safe";
    } else {
        r.verdict = "unknown (probe: fma contraction behaviour not recognised)";
        r.detail = "numeric-mode probe: " + std::to_string(match) + "/64 values match the host fma, " +
                   std::to_string(zeros) + "/64 are zero";
    }
    a = Buffer();
    b = Buffer();
    o = Buffer();
    pool_.trim();
    return r;
}

}  // namespace spk::gpu
