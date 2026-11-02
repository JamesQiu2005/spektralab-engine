// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 Guangzhou Songzi Intelligent Technology Services Co., Ltd.

/* The engine: creation, introspection, the memory report with its pressure and idle-trim sources,
 * warm-up, the session list and the print-preview tables. spk_output_transform is computed in print.
 */
#include "session/engine.hpp"

#include <dispatch/dispatch.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>

#include "fx/glare.hpp"
#include "fx/rng.hpp"
#include "session/session.hpp"
#include "session/tiers.hpp"

#ifndef SPK_BUILD_DATE
#define SPK_BUILD_DATE "unknown"
#endif

namespace spk {

uint64_t Engine::next_id() {
    static std::atomic<uint64_t> counter{0};
    return ++counter;
}

const char* Engine::build_info() {
    static const std::string s = std::string("spektralab-native ") + SPK_VERSION_STRING + " (" + SPK_BUILD_DATE + ", math=safe)";
    return s.c_str();
}

/* Engine::create(resources_dir, mtl_device, device_type)
 *   check arguments
 *   variants := EffectVariants::for_device(device_type)
 *   resources := Resources::load(resources_dir)
 *   gpu := Context::create(mtl_device, resources.metallib)
 *   probe := gpu.probe_math_mode(); refuse unless probe.ok
 *   record the blob's size as file-backed memory
 *   schema := Schema::build()
 *   start monitors
 */
std::unique_ptr<Engine> Engine::create(const char* resources_dir, void* mtl_device, int32_t device_type) {
    if (!resources_dir) fail(SPK_ERR_INVALID_ARG, "resources_dir is NULL");
    if (device_type < 0 || device_type > 2)
        fail(SPK_ERR_INVALID_ARG, "unknown device_type " + std::to_string(device_type) +
                                      "; expected 0 (desktop), 1 (phone) or 2 (tablet)");
    std::unique_ptr<Engine> e(new Engine());
    e->device_ = static_cast<core::Device>(device_type);
    e->variants_ = core::EffectVariants::for_device(e->device_);
    e->res_ = res::Resources::load(resources_dir);
    e->gpu_ = gpu::Context::create(mtl_device, e->res_->metallib);

    gpu::ProbeResult probe = e->gpu_->probe_math_mode();
    if (!probe.ok) fail(SPK_ERR_GPU, "numeric-mode probe refused the kernels: " + probe.verdict + "; " + probe.detail);
    e->math_mode_ = probe.verdict;
    e->gpu_->note_file_backed(e->res_->blob->file_size());

    e->schema_ = params::Schema::build();
    e->start_monitors();
    return e;
}

Engine::~Engine() {
    stop_monitors();
    std::vector<std::unique_ptr<Session>> doomed;
    {
        std::lock_guard<std::mutex> g(mutex_);
        doomed.swap(sessions_);
    }
    doomed.clear();
}

// ---- memory: pressure events and the idle trim -------------------------------------

namespace {
constexpr int64_t kIdleTrimSeconds = 60;
constexpr int64_t kIdleTickSeconds = 5;
int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}
}  // namespace

struct Engine::Monitors {
    dispatch_queue_t queue = nullptr;
    dispatch_source_t pressure = nullptr;
    dispatch_source_t idle = nullptr;
};

/* start_monitors(): one serial queue with two sources
 *   pressure event -> on_memory_pressure(critical?)
 *   timer, every kIdleTickSeconds -> on_idle_tick()
 *   no queue: no monitors; the report says monitor = false
 */
void Engine::start_monitors() {
    last_activity_ns_.store(now_ns());
    auto m = std::make_unique<Monitors>();
    m->queue = dispatch_queue_create("spektralab.memory", DISPATCH_QUEUE_SERIAL);
    if (!m->queue) return;  // no monitor: the report says monitor = false
    m->pressure = dispatch_source_create(DISPATCH_SOURCE_TYPE_MEMORYPRESSURE, 0,
                                         DISPATCH_MEMORYPRESSURE_WARN | DISPATCH_MEMORYPRESSURE_CRITICAL, m->queue);
    if (m->pressure) {
        dispatch_set_context(m->pressure, this);
        dispatch_source_set_event_handler_f(m->pressure, [](void* ctx) {
            Engine* e = static_cast<Engine*>(ctx);
            const unsigned long flags = dispatch_source_get_data(e->monitors_->pressure);
            if (flags & DISPATCH_MEMORYPRESSURE_CRITICAL) e->on_memory_pressure(true);
            else if (flags & DISPATCH_MEMORYPRESSURE_WARN) e->on_memory_pressure(false);
        });
    }
    m->idle = dispatch_source_create(DISPATCH_SOURCE_TYPE_TIMER, 0, 0, m->queue);
    if (m->idle) {
        dispatch_set_context(m->idle, this);
        dispatch_source_set_timer(m->idle, dispatch_time(DISPATCH_TIME_NOW, kIdleTickSeconds * NSEC_PER_SEC),
                                  uint64_t(kIdleTickSeconds) * NSEC_PER_SEC, NSEC_PER_SEC);
        dispatch_source_set_event_handler_f(m->idle, [](void* ctx) { static_cast<Engine*>(ctx)->on_idle_tick(); });
    }
    monitors_ = std::move(m);
    if (monitors_->pressure) dispatch_resume(monitors_->pressure);
    if (monitors_->idle) dispatch_resume(monitors_->idle);
}

void Engine::stop_monitors() {
    if (!monitors_) return;
    Monitors& m = *monitors_;
    if (m.pressure) dispatch_source_cancel(m.pressure);
    if (m.idle) dispatch_source_cancel(m.idle);
    // Handlers run on m.queue: once this empty block has run, none is running and none will start.
    if (m.queue) dispatch_sync_f(m.queue, nullptr, [](void*) {});
    if (m.pressure) dispatch_release(m.pressure);
    if (m.idle) dispatch_release(m.idle);
    if (m.queue) dispatch_release(m.queue);
    monitors_.reset();
}

/* on_memory_pressure(critical)
 *   critical: free all idle scratch; flag the next render
 *   warn:     keep up to this frame's high-water bytes of free scratch, free the rest
 */
void Engine::on_memory_pressure(bool critical) {
    if (critical) {
        ++critical_events_;
        gpu_->pool().trim(0);  // free everything idle
        critical_pending_.store(true);
    } else {
        ++warn_events_;
        // Keep the running frame's own peak, free the rest.
        gpu_->pool().trim(gpu_->pool().stats().frame_high_water_bytes);
    }
}

/* on_idle_tick()
 *   if no render in flight and not yet trimmed and idle >= kIdleTrimSeconds:
 *     mark trimmed; if any scratch is free: free it, count one idle trim
 */
void Engine::on_idle_tick() {
    if (renders_in_flight_.load() != 0 || idle_trimmed_.load()) return;
    if (now_ns() - last_activity_ns_.load() < kIdleTrimSeconds * 1000000000LL) return;
    idle_trimmed_.store(true);
    if (gpu_->pool().stats().free_bytes == 0) return;
    gpu_->pool().trim(0);
    ++idle_trims_;
}

/* render_begin() -> one-shot critical flag
 *   in-flight += 1; touch the idle clock; reset the pool's high-water mark
 */
bool Engine::render_begin() {
    ++renders_in_flight_;
    last_activity_ns_.store(now_ns());
    idle_trimmed_.store(false);
    gpu_->pool().begin_frame();
    return critical_pending_.exchange(false);
}

void Engine::render_end() {
    last_activity_ns_.store(now_ns());
    idle_trimmed_.store(false);
    --renders_in_flight_;
}

/* frame_switch_trim(w, h)
 *   if no session is open or (w, h) differs from the last frame: free idle scratch
 *   remember (w, h)
 */
void Engine::frame_switch_trim(uint32_t w, uint32_t h) {
    std::lock_guard<std::mutex> g(mutex_);
    if (sessions_.empty() || w != pool_frame_w_ || h != pool_frame_h_) gpu_->pool().trim(0);
    pool_frame_w_ = w;
    pool_frame_h_ = h;
}

/* capabilities_value() -> {version, engine, max_pixels, max_mp, max_texture_dimension_2d, tiers,
 *                          transport_version, schema_version, backend}
 *   caller holds the engine lock
 */
json::Value Engine::capabilities_value() {
    json::Value b;
    b["spectral"] = "native";
    b["gpu"] = gpu_->device_name();
    b["gpu_available"] = true;
    b["working_precision"] = "float32";
    b["render_core"] = "native-metal";
    b["host"] = "native";
    b["math_mode"] = math_mode_;
    b["device"] = device_ == core::Device::Desktop ? "desktop" : device_ == core::Device::Phone ? "phone" : "tablet";
    b["effect_versions"] = std::string(variants_.effect_versions());
    b["concurrent"] = true;
    json::Value sc;
    sc["entries"] = uint64_t(sessions_.size());
    sc["max_entries"] = 0;
    sc["enabled"] = false;
    sc["setup_cam16"] = uint64_t(cam16_.size());
    b["session_cache"] = std::move(sc);

    json::Value c;
    c["version"] = build_info();
    c["engine"] = SPK_ENGINE_NAME;
    c["max_pixels"] = uint64_t(SPK_MAX_PIXELS);
    c["max_mp"] = double(SPK_MAX_PIXELS) / 1e6;
    c["max_texture_dimension_2d"] = gpu_->max_texture_dimension_2d();
    json::Value tiers;
    tiers["full"] = nullptr;
    tiers["live"] = 1600;
    tiers["preview"] = 3400;
    c["tiers"] = std::move(tiers);
    c["transport_version"] = SPK_TRANSPORT_VERSION;
    c["schema_version"] = SPK_SCHEMA_VERSION;
    c["backend"] = std::move(b);
    return c;
}

const char* Engine::capabilities() {
    std::lock_guard<std::mutex> g(mutex_);
    capabilities_text_ = json::write(capabilities_value());
    return capabilities_text_.c_str();
}

const char* Engine::params_schema() const { return schema_->json_text().c_str(); }

/* memory_report() -> {total_bytes, pool, persistent, pressure}
 *   total_bytes = pool.total_bytes + persistent.bytes
 *   persistent.source_bytes, cached_negative_bytes, negatives: sums over open sessions
 */
const char* Engine::memory_report() {
    std::lock_guard<std::mutex> g(mutex_);
    const gpu::PoolStats ps = gpu_->pool().stats();
    const gpu::PersistentStats pe = gpu_->persistent_stats();

    json::Value audit;
    audit["allocations"] = ps.allocations;
    audit["reuses"] = ps.reuses;
    audit["persistent_allocations"] = ps.persistent_allocations;
    audit["reuse_bytes_requested"] = ps.reuse_bytes_requested;
    audit["reuse_bytes_taken"] = ps.reuse_bytes_taken;
    audit["reuse_max_taken"] = ps.reuse_max_taken;
    audit["reuse_max_taken_for_request"] = ps.reuse_max_taken_for_request;
    audit["pending_held"] = ps.pending_held;
    audit["live_underflows"] = ps.live_underflows;
    audit["over_releases"] = ps.over_releases;
    audit["reclaim_while_encoding"] = ps.reclaim_while_encoding;

    json::Value pool;
    pool["total_bytes"] = ps.total_bytes;
    pool["live_bytes"] = ps.live_bytes;
    pool["free_bytes"] = ps.free_bytes;
    pool["pending_bytes"] = ps.pending_bytes;
    pool["buffers"] = ps.buffers;
    pool["live_buffers"] = ps.live_buffers;
    pool["free_buffers"] = ps.free_buffers;
    pool["pending_buffers"] = ps.pending_buffers;
    pool["frame_high_water_bytes"] = ps.frame_high_water_bytes;
    pool["audit"] = std::move(audit);

    uint64_t src = 0, neg = 0, nneg = 0;
    json::Array per;
    for (const auto& s : sessions_) {
        json::Value r;
        r["session_id"] = s->id();
        r["source_bytes"] = s->source_bytes();
        r["cached_negative_bytes"] = s->cached_negative_bytes();
        r["negatives"] = s->negatives();
        src += s->source_bytes();
        neg += s->cached_negative_bytes();
        nneg += s->negatives();
        per.push_back(std::move(r));
    }
    json::Value pers;
    pers["bytes"] = pe.bytes;
    pers["buffers"] = pe.buffers;
    pers["file_backed_bytes"] = pe.file_backed_bytes;
    pers["file_backed_buffers"] = pe.file_backed_buffers;
    pers["source_bytes"] = src;
    pers["cached_negative_bytes"] = neg;
    pers["negatives"] = nneg;
    pers["sessions"] = std::move(per);

    json::Value pr;
    const bool monitor = monitors_ && monitors_->pressure;
    pr["monitor"] = monitor;
    pr["warn_events"] = warn_events_.load();
    pr["critical_events"] = critical_events_.load();
    pr["critical_pending"] = critical_pending_.load();
    pr["idle_trim_seconds"] = (monitors_ && monitors_->idle) ? kIdleTrimSeconds : 0;
    pr["idle_trims"] = idle_trims_.load();

    json::Value r;
    r["total_bytes"] = ps.total_bytes + pe.bytes;
    r["pool"] = std::move(pool);
    r["persistent"] = std::move(pers);
    r["pressure"] = std::move(pr);
    memory_report_text_ = json::write(r);
    return memory_report_text_.c_str();
}

size_t Engine::trim() { return gpu_->pool().trim(); }

/* warm_up(film_stock, print_stock) -> reply JSON
 *   p := private ParamSet; seeds := private SeedSequence
 *   step "profiles": load film and paper; neutral pack from the database
 *   step "pipeline": build film and print set-up (fills the output gamut cache)
 *   step "kernels":  src := 64 x 64 of 0.18; pitch as if the long edge were 1600
 *                    develop(src, no meter EV) -> negative; print_scan_encode -> image; release it
 *   each step records {name, ok, ms, error?}; a failed step ends the sequence
 */
std::string Engine::warm_up(const char* film_stock, const char* print_stock) {
    using clock = std::chrono::steady_clock;
    using params::P;
    auto ms_since = [](clock::time_point t0) {
        return std::chrono::duration<double, std::milli>(clock::now() - t0).count();
    };
    auto r2 = [](double ms) { return std::round(ms * 100.0) / 100.0; };
    const std::string film_id = film_stock && *film_stock ? film_stock : "kodak_portra_400";
    const std::string print_id = print_stock && *print_stock ? print_stock : "kodak_portra_endura";
    const clock::time_point t_all = clock::now();

    json::Array steps;
    auto run_step = [&](const char* name, auto&& fn) -> bool {
        const clock::time_point t0 = clock::now();
        json::Value st;
        st["name"] = name;
        bool ok = true;
        try {
            fn();
        } catch (const std::exception& e) {
            ok = false;
            st["error"] = std::string(e.what());
        } catch (...) {
            ok = false;
            st["error"] = "unknown failure";
        }
        st["ok"] = ok;
        st["ms"] = r2(ms_since(t0));
        steps.push_back(std::move(st));
        return ok;
    };

    params::ParamSet p(schema_);
    p.set(P::film_stock, film_id);
    p.set(P::print_stock, print_id);
    std::shared_ptr<const res::Profile> film_profile, print_profile;
    std::shared_ptr<const film::FilmSetup> film_setup;
    std::shared_ptr<const print::PrintSetup> print_setup;

    bool ok = run_step("profiles", [&] {
        film_profile = res_->profile(film_id);
        print_profile = res_->profile(print_id);
        if (auto pack = res_->filters.lookup_with_alias(print_id, p.s(P::enlarger_illuminant), film_profile->id,
                                                        film_profile->base_id)) {
            p.set(P::c_filter_neutral, (*pack)[0]);
            p.set(P::m_filter_neutral, (*pack)[1]);
            p.set(P::y_filter_neutral, (*pack)[2]);
        }
    });
    if (ok)
        ok = run_step("pipeline", [&] {
            film_setup = film::build_film_setup({p, film_profile, *res_, *gpu_, device_});
            print_setup = print::build_print_setup(
                {p, *film_setup, print_profile, *res_, *gpu_, &cam16_, device_, true});
        });
    if (ok)
        run_step("kernels", [&] {
            constexpr uint32_t kSide = 64, kTold = session::kMeterLongEdge;
            gpu::Plane src = gpu_->pool().plane(kSide, kSide);
            std::fill(src.data(), src.data() + src.floats(), 0.18f);
            core::RenderContext ctx{.gpu = *gpu_, .device = device_, .variants = variants_};
            ctx.rho_tier = 1.0;
            ctx.pitch_um = film::pitch_um(p.f(P::film_format_mm), kTold);
            fx::SeedSequence seeds;  // private: never a session's
            const uint32_t film_seed = p.b(P::grain_active) ? seeds.next() : fx::kNoGrainFilmSeed;
            gpu::Plane negative = film::develop(ctx, *film_setup, {src, std::nullopt, film_seed});
            const uint32_t print_seed =
                fx::glare_draws_seed(print_setup->glare()) ? seeds.next() : fx::kNoGlarePrintSeed;
            gpu::OutputImage img =
                print::print_scan_encode(ctx, *print_setup, {negative, print::LiveParams::from(p), print_seed});
            gpu::Context::release_texture(img.texture);
        });

    json::Value r;
    r["already_warm"] = false;
    r["film_stock"] = film_id;
    r["print_stock"] = print_id;
    r["render_core"] = "native-metal";
    r["steps"] = std::move(steps);
    r["total_ms"] = r2(ms_since(t_all));
    return json::write(r);
}

/* open(in, delta, out_json)
 *   Session::open outside the engine lock; then add to the session list under it
 */
Session* Engine::open(const OpenInput& in, std::string_view delta, std::string* out_json) {
    std::unique_ptr<Session> s = Session::open(*this, in, delta, out_json);
    Session* raw = s.get();
    std::lock_guard<std::mutex> g(mutex_);
    sessions_.push_back(std::move(s));
    return raw;
}

/* release(session)
 *   under the lock: remove from the list; phone or tablet and list now empty: free idle scratch
 *   destroy the session outside the lock
 */
void Engine::release(Session* s) {
    std::unique_ptr<Session> keep;
    {
        std::lock_guard<std::mutex> g(mutex_);
        auto it = std::find_if(sessions_.begin(), sessions_.end(), [&](const auto& p) { return p.get() == s; });
        if (it == sessions_.end()) return;
        keep = std::move(*it);
        sessions_.erase(it);
        if (sessions_.empty() && device_ != core::Device::Desktop) gpu_->pool().trim();
    }
    // `keep` is destroyed here, outside the engine lock.
}

const char* Engine::print_lut_catalog() const { return res_->print_luts.catalog_bytes().c_str(); }

void Engine::print_lut_table(std::string_view stock, const float** out_table, uint32_t* out_size) {
    std::lock_guard<std::mutex> g(mutex_);
    res::PrintLutTable t = res_->print_luts.table(*res_->blob, stock);
    *out_table = t.table;
    *out_size = t.size;
}

/* output_transform(src, dst, gamut_json, out_json, out_cmax, out_cmax_count)
 *   parse gamut options outside the lock; compute under it; serialise outside it
 */
void Engine::output_transform(std::string_view src, std::string_view dst, const char* gamut_json,
                              std::string* out_json, const float** out_cmax, uint32_t* out_cmax_count) {
    print::GamutOptions g = print::GamutOptions::parse(gamut_json ? gamut_json : "");
    print::OutputTransformResult r;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        r = print::output_transform(*res_, cam16_, src, dst, g);
    }
    *out_json = json::write(r.reply);
    *out_cmax = r.cmax;
    *out_cmax_count = r.cmax_count;
}

}  // namespace spk
