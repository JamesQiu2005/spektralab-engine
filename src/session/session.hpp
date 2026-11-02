// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 Guangzhou Songzi Intelligent Technology Services Co., Ltd.

/* One open frame.
 *
 * owns    source plane, parameter tree, pipeline instance (FilmSetup + PrintSetup + SeedSequence),
 *         meter, and per tier: tier image, cached negative, applied EV, pitch
 * locks   every public method except cancel() and progress() holds the session lock throughout
 *         cancel() and progress() use the ProgressRecord's own lock
 * errors  all methods throw spk::Error
 */
#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>

#include "film/film.hpp"
#include "fx/rng.hpp"
#include "gpu/gpu.hpp"
#include "params/params.hpp"
#include "print/print.hpp"
#include "session/progress.hpp"
#include "session/tiers.hpp"
#include "spektralab.h"

namespace spk {

class Engine;
struct OpenInput;

// A pipeline instance: rebuilt at open and at every non-live set_params.
struct Pipeline {
    std::shared_ptr<const film::FilmSetup> film;
    std::shared_ptr<const print::PrintSetup> print;
    print::LiveParams live;  // live fields written without a rebuild
    fx::SeedSequence seeds;  // reset on every (re)build
};

struct TierState {
    gpu::Plane source;    // tier source image (raw values, uncropped); empty until first use
    uint32_t source_long_edge = 0;  // max(source.w, source.h): the tier's long edge for the pitch and rho_tier
    gpu::Plane negative;  // cached negative; empty when none
    std::optional<double> applied_ev;
    double pitch_um = 0;
};

class Session {
public:
    // spk_open. Throws on every open failure.
    static std::unique_ptr<Session> open(Engine& engine, const OpenInput& in, std::string_view delta,
                                         std::string* out_json);
    ~Session();

    const std::string& id() const { return id_; }
    Engine& engine() const { return *engine_; }

    std::string set_params(std::string_view delta);  // reply JSON; atomic
    std::string get_params();                         // flat params JSON
    // spk_render / spk_reprint; `out` was zero-filled by the ABI and is written only on success.
    void render(std::string_view tier, bool reprint, spk_result* out);
    std::string solve(std::string_view target);
    void preview_stock_lut(const char* print_stock, const char* tier, spk_result* out, std::string* out_json);

    void cancel(const char* progress_id);               // no session lock
    std::string progress(const char* progress_id);      // no session lock

    // Memory report contributions.
    uint64_t source_bytes() const;
    uint64_t cached_negative_bytes() const;
    uint32_t negatives() const;

private:
    Session() = default;
    friend class Engine;

    // ---- internals; the session lock is held by the caller ----
    // The digest: the neutral filter pack from the database, alias fallback included.
    void digest(params::ParamSet& p, const res::Profile& film) const;
    // A pipeline instance built from `p`: throws on a build refusal.
    Pipeline build_pipeline(const params::ParamSet& p, const std::shared_ptr<const res::Profile>& film,
                            const std::shared_ptr<const res::Profile>& print) const;
    // The context of one render on tier `t` (pitch and rho from the tier image's actual size).
    core::RenderContext make_context(session::Tier t);
    TierState& tier_source(session::Tier t);  // builds the tier image lazily
    const film::MeterValues& ensure_meter_locked();
    std::optional<double> session_ev();  // the EV develop applies (meters when needed); nullopt when off
    // Develops tier t into its cache (the render path): meter, film seed, develop, cache on success
    // only.
    void develop_tier(core::RenderContext& ctx, session::Tier t);
    uint32_t draw_print_seed_locked();
    void drop_negative(TierState& t);
    void drop_all_negatives();
    void update_memory_counters();
    void fill_result(spk_result* out, const gpu::OutputImage& img, double elapsed_ms, int32_t reprint,
                     const std::string& progress_id) const;

    Engine* engine_ = nullptr;
    std::string id_;
    std::mutex mutex_;  // the session lock
    session::ProgressRecord progress_;
    bool product_defaults_ = false;

    gpu::Plane source_;  // full frame, 3 channels, non-finite -> 0
    // Always the current values: a live-only delta writes params_ AND pipeline_.live (= LiveParams::
    // from(*params_)).
    std::unique_ptr<params::ParamSet> params_;
    std::shared_ptr<const res::Profile> film_profile_, print_profile_;
    Pipeline pipeline_;
    std::array<TierState, 3> tiers_;
    gpu::Plane meter_image_;
    std::optional<film::MeterValues> meter_;

    // Read by the memory report under the engine lock while a render may run.
    std::atomic<uint64_t> negative_bytes_{0};
    std::atomic<uint32_t> negative_count_{0};
};

}  // namespace spk
