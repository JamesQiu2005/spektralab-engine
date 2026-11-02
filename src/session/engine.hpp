// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 Guangzhou Songzi Intelligent Technology Services Co., Ltd.

/* The engine: resources, GPU, caches, open sessions and engine-owned reply strings.
 *
 * one engine lock guards engine-wide state
 * src/abi casts spk_engine* <-> spk::Engine*; nothing else crosses the ABI
 * all methods report failure by throwing spk::Error
 */
#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include "core/render.hpp"
#include "film/film.hpp"
#include "gpu/gpu.hpp"
#include "params/params.hpp"
#include "print/print.hpp"
#include "res/resources.hpp"
#include "util/json.hpp"

namespace spk {

class Session;

// The pixels handed to spk_open / spk_open_device (borrowed for the call).
struct OpenInput {
    const float* host_data = nullptr;  // spk_open
    void* device_buffer = nullptr;     // spk_open_device: id<MTLBuffer>
    uint32_t width = 0, height = 0, channels = 0;
};

class Engine {
public:
    // Throws spk::Error on every creation failure; the ABI returns NULL.
    static std::unique_ptr<Engine> create(const char* resources_dir, void* mtl_device, int32_t device_type);
    ~Engine();  // releases every open session first
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    // ---- ABI-facing (engine-owned strings stay valid until the next call of the same kind or the engine's end) ----
    const char* capabilities();         // recomputed each call; takes the engine lock
    json::Value capabilities_value();   // the same object (open reply)
    const char* params_schema() const;  // computed once
    const char* memory_report();        // recomputed each call; cannot fail
    size_t trim();                      // spk_engine_trim
    std::string warm_up(const char* film_stock, const char* print_stock);  // reply JSON
    Session* open(const OpenInput& in, std::string_view delta, std::string* out_json);
    void release(Session* s);
    const char* print_lut_catalog() const;
    void print_lut_table(std::string_view stock, const float** out_table, uint32_t* out_size);
    // spk_output_transform; *out_json receives the reply text.
    void output_transform(std::string_view src, std::string_view dst, const char* gamut_json, std::string* out_json,
                          const float** out_cmax, uint32_t* out_cmax_count);

    // ---- for the modules ----
    core::Device device() const { return device_; }
    const core::EffectVariants& variants() const { return variants_; }
    gpu::Context& gpu() { return *gpu_; }
    const res::Resources& resources() const { return *res_; }
    const params::Schema& schema() const { return *schema_; }
    std::shared_ptr<const params::Schema> schema_ptr() const { return schema_; }
    const std::string& math_mode() const { return math_mode_; }
    print::Cam16Cache& cam16_cache() { return cam16_; }
    std::mutex& lock() { return mutex_; }

    // The process-wide id counter shared by session ids (s<N>) and progress ids (p<N>), from 1.
    static uint64_t next_id();
    // spektralab-native <semver> (<YYYY-MM-DD>, math=safe)
    static const char* build_info();

    // ---- for sessions (src/session) ----
    // Frees the idle scratch pool when no session is open or the frame size
    // differs from the one the pool was last sized for. Takes the engine lock.
    void frame_switch_trim(uint32_t w, uint32_t h);
    // A render-type call starts / ends: the in-flight count and the idle clock.
    // render_begin() returns the one-shot memory_pressure_critical flag.
    bool render_begin();
    void render_end();

    // OS memory-pressure and idle-trim handlers; public for the dispatch callbacks.
    void on_memory_pressure(bool critical);
    void on_idle_tick();

private:
    Engine() = default;
    void start_monitors();
    void stop_monitors();

    core::Device device_ = core::Device::Desktop;
    core::EffectVariants variants_;
    std::shared_ptr<const res::Resources> res_;
    std::shared_ptr<gpu::Context> gpu_;
    std::shared_ptr<const params::Schema> schema_;
    std::string math_mode_;
    print::Cam16Cache cam16_;

    mutable std::mutex mutex_;  // the engine lock
    std::vector<std::unique_ptr<Session>> sessions_;
    std::string capabilities_text_;
    std::string memory_report_text_;

    // Scratch given back on a frame switch, OS pressure and 60 s idle.
    uint32_t pool_frame_w_ = 0, pool_frame_h_ = 0;
    struct Monitors;  // GCD sources (engine.cpp)
    std::unique_ptr<Monitors> monitors_;
    std::atomic<int> renders_in_flight_{0};
    std::atomic<int64_t> last_activity_ns_{0};
    std::atomic<bool> idle_trimmed_{false};
    std::atomic<bool> critical_pending_{false};
    std::atomic<uint64_t> warn_events_{0}, critical_events_{0}, idle_trims_{0};
};

}  // namespace spk
