// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 Guangzhou Songzi Intelligent Technology Services Co., Ltd.

/* One open frame: open, the digest, set_params with its candidate rebuild and invalidation,
 * get_params, render / reprint over the tier cache, solve, the print-preview table render,
 * cancellation and progress.
 */
#include "session/session.hpp"

#include <chrono>
#include <cmath>
#include <cstring>

#include "fx/glare.hpp"
#include "session/engine.hpp"
#include "util/numeric.hpp"
#include "session_shared.h"

namespace spk {

using params::P;
using session::Tier;

namespace {

using clock = std::chrono::steady_clock;
double ms_since(clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(clock::now() - t0).count();
}

constexpr uint64_t kMaxPixels = 151301008ull;  // 14204 x 10652

// The ten meter-key fields (a change to any of them re-arms the meter).
constexpr P kMeterKey[] = {P::input_color_space, P::input_cctf_decoding, P::geometry_crop_x, P::geometry_crop_y,
                           P::geometry_crop_w,   P::geometry_crop_h,     P::geometry_rotation_deg,
                           P::geometry_quarter_turns, P::geometry_flip_h, P::geometry_flip_v};

size_t tier_index(Tier t) { return size_t(t); }

Tier tier_or_throw(std::string_view s) {
    std::optional<Tier> t = session::parse_tier(s);
    if (!t) fail(SPK_ERR_INVALID_ARG, "unknown tier '" + std::string(s) + "'; expected live, preview or full");
    return *t;
}

/* rethrow_as_render_failure(): inside a catch
 *   SPK_ERR_CANCELLED stays; every other failure -> SPK_ERR_GPU, message kept
 */
[[noreturn]] void rethrow_as_render_failure() {
    try {
        throw;
    } catch (const Error& e) {
        if (e.status() == SPK_ERR_CANCELLED) throw;
        fail_gpu(e.what());
    } catch (const std::bad_alloc&) {
        fail_gpu("out of memory");
    } catch (const std::exception& e) {
        fail_gpu(e.what());
    }
}

}  // namespace

// ---- open ------------------------------------------------------------------------------

Session::~Session() = default;

/* digest(p, film): stock-resolved parameters
 *   row := database[print stock][enlarger illuminant][film id, else base id for an alias]
 *   hit: c/m/y_filter_neutral := row; miss: leave them
 *   the halation preset and coupler constants are resolved by the film build, not here
 */
void Session::digest(params::ParamSet& p, const res::Profile& film) const {
    const res::Resources& res = engine_->resources();
    if (auto pack = res.filters.lookup_with_alias(p.s(P::print_stock), p.s(P::enlarger_illuminant), film.id,
                                                  film.base_id)) {
        p.set(P::c_filter_neutral, (*pack)[0]);
        p.set(P::m_filter_neutral, (*pack)[1]);
        p.set(P::y_filter_neutral, (*pack)[2]);
    }
}

/* build_pipeline(p, film, print) -> Pipeline
 *   film set-up; print set-up (exposure compensation = !product_defaults)
 *   live := LiveParams::from(p); seeds := initial state
 *   throws on a build refusal
 */
Pipeline Session::build_pipeline(const params::ParamSet& p, const std::shared_ptr<const res::Profile>& film,
                                 const std::shared_ptr<const res::Profile>& print) const {
    Engine& e = *engine_;
    Pipeline pl;
    pl.film = film::build_film_setup({p, film, e.resources(), e.gpu(), e.device()});
    pl.print = print::build_print_setup({p, *pl.film, print, e.resources(), e.gpu(), &e.cam16_cache(),
                                         e.device(), !product_defaults_});
    pl.live = print::LiveParams::from(p);
    pl.seeds.reset();
    return pl;
}

/* Session::open(engine, in, delta, out_json)
 *   1 check the frame and, for a device buffer, its device and length
 *   2 engine.frame_switch_trim(w, h)
 *   3 validate the delta
 *   4 p := defaults + delta stocks; load profiles; digest; force the working output space;
 *     apply delta; digest
 *   5 copy pixels: non-finite -> 0, 4th channel dropped
 *   6 build the pipeline; id := s<N>
 *   7 reply: {capabilities, detected_input, meta, params, session_id}
 */
std::unique_ptr<Session> Session::open(Engine& engine, const OpenInput& in, std::string_view delta_text,
                                       std::string* out_json) {
    gpu::Context& g = engine.gpu();
    if (in.channels != 3 && in.channels != 4)
        fail_user("input has " + std::to_string(in.channels) + " channels; expected 3 or 4");
    if (in.width == 0 || in.height == 0) fail_user("input has a zero width or height");
    const uint64_t npx = uint64_t(in.width) * in.height;
    if (npx > kMaxPixels)
        fail_user("input of " + std::to_string(in.width) + " x " + std::to_string(in.height) + " = " +
                  std::to_string(npx) + " pixels exceeds the cap of 151301008 pixels (14204 x 10652)");
    const uint32_t long_edge = std::max(in.width, in.height);
    if (long_edge > g.max_texture_dimension_2d())
        fail_user("input long edge " + std::to_string(long_edge) + " exceeds the device's maximum 2-D texture dimension " +
                  std::to_string(g.max_texture_dimension_2d()));
    if (in.device_buffer) {
        if (!g.same_device(in.device_buffer)) fail_user("input buffer belongs to another MTLDevice");
        const size_t need = size_t(npx) * in.channels * sizeof(float);
        if (gpu::Context::buffer_length(in.device_buffer) < need)
            fail_user("input buffer is " + std::to_string(gpu::Context::buffer_length(in.device_buffer)) +
                      " bytes; the frame needs " + std::to_string(need));
    } else if (!in.host_data) {
        fail(SPK_ERR_INVALID_ARG, "input->data is NULL");
    }

    engine.frame_switch_trim(in.width, in.height);

    params::Delta d = params::validate_delta(engine.schema(), delta_text, params::DeltaContext::Open);

    std::unique_ptr<Session> s(new Session());
    s->engine_ = &engine;
    s->product_defaults_ = d.product_defaults.value_or(false);

    auto p = std::make_unique<params::ParamSet>(engine.schema_ptr());
    for (const auto& [fid, v] : d.fields)
        if (fid == params::id(P::film_stock) || fid == params::id(P::print_stock)) p->set(fid, v);
    std::shared_ptr<const res::Profile> film = engine.resources().profile(p->s(P::film_stock));
    std::shared_ptr<const res::Profile> print = engine.resources().profile(p->s(P::print_stock));
    s->digest(*p, *film);
    p->set(P::output_color_space, "ProPhoto RGB");
    p->set(P::output_cctf_encoding, true);
    params::apply_delta(*p, d);
    // Product defaults (print-exposure compensation off) travel as product_defaults_ into
    // every print build.
    s->digest(*p, *film);

    gpu::Plane src = g.persistent_plane(in.width, in.height, gpu::MemTag::Source);
    if (in.device_buffer) {
        gpu::Buffer wrapped = g.wrap(in.device_buffer);
        SpkSessionCopyArgs a{uint32_t(npx), in.channels};
        gpu::Batch b = g.batch();
        b.dispatch_1d(g.kernel("spk_session_copy_source"), uint32_t(npx),
                      {gpu::Arg::buffer(wrapped), gpu::Arg::plane(src), gpu::Arg::value(a)});
        b.commit_and_wait();
    } else {
        const float* q = in.host_data;
        float* o = src.data();
        for (uint64_t i = 0; i < npx; ++i, q += in.channels, o += 3)
            for (int c = 0; c < 3; ++c) o[c] = std::isfinite(q[c]) ? q[c] : 0.0f;
    }

    s->pipeline_ = s->build_pipeline(*p, film, print);
    s->params_ = std::move(p);
    s->film_profile_ = std::move(film);
    s->print_profile_ = std::move(print);
    s->source_ = std::move(src);
    s->id_ = "s" + std::to_string(Engine::next_id());

    if (out_json) {
        json::Value r;
        {
            std::lock_guard<std::mutex> lk(engine.lock());
            r["capabilities"] = engine.capabilities_value();
        }
        json::Value di;
        di["input_cctf_decoding"] = s->params_->b(P::input_cctf_decoding);
        di["input_color_space"] = s->params_->s(P::input_color_space);
        di["input_color_space_source"] = "caller";
        r["detected_input"] = std::move(di);
        json::Value meta;
        meta["height"] = in.height;
        meta["megapixels"] = num::round_half_away(double(npx) / 1e4) / 100.0;
        meta["source"] = "<in-process>";
        meta["width"] = in.width;
        r["meta"] = std::move(meta);
        r["params"] = s->params_->to_json();
        r["session_id"] = s->id_;
        *out_json = json::write(r);
    }
    return s;
}

// ---- set_params / get_params ----------------------------------------------------

/* set_params(delta) -> reply JSON; atomic
 *   validate before the lock; take the lock
 *   cand := params + delta
 *   if stock in delta: film, print := reload; digest(cand)
 *   else: no digest (enlarger_illuminant alone does not re-resolve the pack)
 *   if any_rebuild: pipeline := build_pipeline(cand)      -- throws: nothing has changed
 *   else:           pipeline.live := LiveParams::from(cand)
 *   commit cand and the profiles
 *   any_shoot:                  drop every cached negative
 *   preview_long_edge changed the live tier's size: drop the live tier's whole state
 *   any meter-key field changed: meter := none
 *   reply: {invalidated, params}
 */
std::string Session::set_params(std::string_view delta_text) {
    // Validation runs before the session lock is taken; the schema is immutable.
    const params::Delta d = params::validate_delta(engine_->schema(), delta_text, params::DeltaContext::SetParams);
    std::lock_guard<std::mutex> g(mutex_);

    auto cand = std::make_unique<params::ParamSet>(*params_);
    params::apply_delta(*cand, d);
    std::shared_ptr<const res::Profile> film = film_profile_, print = print_profile_;
    if (d.has_stock) {
        film = engine_->resources().profile(cand->s(P::film_stock));
        print = engine_->resources().profile(cand->s(P::print_stock));
        digest(*cand, *film);
    }
    if (d.any_rebuild) {
        Pipeline pl = build_pipeline(*cand, film, print);  // throws: nothing has changed
        pipeline_ = std::move(pl);
    } else {
        pipeline_.live = print::LiveParams::from(*cand);
    }

    const uint32_t old_live = params_->i(P::preview_long_edge), new_live = cand->i(P::preview_long_edge);
    bool rearm_meter = false;
    for (P k : kMeterKey)
        if (!(params_->get(params::id(k)) == cand->get(params::id(k)))) rearm_meter = true;

    std::unique_ptr<params::ParamSet> old = std::move(params_);
    params_ = std::move(cand);
    film_profile_ = std::move(film);
    print_profile_ = std::move(print);

    if (d.any_shoot) drop_all_negatives();
    if (d.has_preview_long_edge) {
        const session::Size a = session::downscale_size(source_.w, source_.h, old_live);
        const session::Size b = session::downscale_size(source_.w, source_.h, new_live);
        if (a.w != b.w || a.h != b.h) tiers_[tier_index(Tier::Live)] = TierState{};
    }
    if (rearm_meter) meter_.reset();
    update_memory_counters();

    json::Value r;
    r["invalidated"] = d.any_shoot ? "shoot" : "print";
    r["params"] = params_->to_json();
    return json::write(r);
}

std::string Session::get_params() {
    std::lock_guard<std::mutex> g(mutex_);
    return json::write(params_->to_json());
}

// ---- tiers, meter, develop --------------------------------------------------------

/* make_context(t): context of one render on tier t
 *   rho_tier := tier long edge / full long edge
 *   pitch := the cached negative's pitch, else film_format_mm * 1000 / tier long edge
 */
core::RenderContext Session::make_context(Tier t) {
    Engine& e = *engine_;
    TierState& ts = tier_source(t);
    core::RenderContext ctx{.gpu = e.gpu(), .device = e.device(), .variants = e.variants(),
                            .cancel = &progress_.cancel_token(), .progress = &progress_};
    ctx.rho_tier = session::tier_ratio({ts.source.w, ts.source.h}, {source_.w, source_.h});
    ctx.pitch_um = ts.negative ? ts.pitch_um : film::pitch_um(params_->f(P::film_format_mm), ts.source_long_edge);
    return ctx;
}

/* tier_source(t): lazily build the tier image
 *   L := tier target long edge; source := downscale(frame, L)
 */
TierState& Session::tier_source(Tier t) {
    TierState& ts = tiers_[tier_index(t)];
    if (!ts.source) {
        const uint32_t L = session::tier_target_long_edge(t, uint32_t(params_->i(P::preview_long_edge)));
        core::RenderContext ctx{.gpu = engine_->gpu(), .device = engine_->device(), .variants = engine_->variants()};
        ts.source = session::downscale(ctx, source_, L, gpu::MemTag::TierImage);
        ts.source_long_edge = std::max(ts.source.w, ts.source.h);
    }
    return ts;
}

/* ensure_meter_locked(): meter once, at a fixed 1600 px long edge
 *   meter image := live tier image when preview_long_edge == 1600 and it exists, else downscale(frame, 1600)
 *   values := film::meter(params, meter image); kept until a meter-key field changes
 */
const film::MeterValues& Session::ensure_meter_locked() {
    if (meter_) return *meter_;
    Engine& e = *engine_;
    core::RenderContext ctx{.gpu = e.gpu(), .device = e.device(), .variants = e.variants()};
    if (!meter_image_) {
        // The live tier image is the same computation when the live tier's long edge is 1600.
        const TierState& live = tiers_[tier_index(Tier::Live)];
        if (live.source && params_->i(P::preview_long_edge) == session::kMeterLongEdge)
            meter_image_ = live.source;
        else
            meter_image_ = session::downscale(ctx, source_, session::kMeterLongEdge, gpu::MemTag::MeterImage);
    }
    meter_ = film::meter(ctx, *params_, e.resources(), meter_image_);
    return *meter_;
}

std::optional<double> Session::session_ev() {
    if (!params_->b(P::auto_exposure)) return std::nullopt;
    return ensure_meter_locked().ev(params_->s(P::auto_exposure_method));
}

/* develop_tier(ctx, t)
 *   drop the tier's negative
 *   ev := session_ev()                                 -- meters when needed
 *   seed := grain_active ? seeds.next() : 0x5EED
 *   negative := film::develop(tier image, ev, seed); store its pitch and ev
 */
void Session::develop_tier(core::RenderContext& ctx, Tier t) {
    TierState& ts = tier_source(t);
    drop_negative(ts);  // a failed or cancelled develop caches nothing
    const std::optional<double> ev = session_ev();
    const uint32_t seed = params_->b(P::grain_active) ? pipeline_.seeds.next() : fx::kNoGrainFilmSeed;
    ctx.pitch_um = film::pitch_um(params_->f(P::film_format_mm), ts.source_long_edge);
    ts.negative = film::develop(ctx, *pipeline_.film, {ts.source, ev, seed});
    ts.pitch_um = ctx.pitch_um;
    ts.applied_ev = ev;
    update_memory_counters();
}

/* draw_print_seed_locked() -> glare can draw ? seeds.next() : 0
 *   drawn even where the glare stage is skipped (scan_film), so the sequence advances alike
 */
uint32_t Session::draw_print_seed_locked() {
    return fx::glare_draws_seed(pipeline_.print->glare()) ? pipeline_.seeds.next() : fx::kNoGlarePrintSeed;
}

void Session::drop_negative(TierState& t) {
    t.negative = gpu::Plane{};
    t.applied_ev.reset();
    update_memory_counters();
}

void Session::drop_all_negatives() {
    for (TierState& t : tiers_) {
        t.negative = gpu::Plane{};
        t.applied_ev.reset();
    }
    update_memory_counters();
}

void Session::update_memory_counters() {
    uint64_t b = 0;
    uint32_t n = 0;
    for (const TierState& t : tiers_)
        if (t.negative) {
            b += t.negative.buf.size();
            ++n;
        }
    negative_bytes_.store(b);
    negative_count_.store(n);
}

void Session::fill_result(spk_result* out, const gpu::OutputImage& img, double elapsed_ms, int32_t reprint,
                          const std::string& pid) const {
    out->rgba16 = img.rgba16;
    out->texture = img.texture;
    out->width = img.width;
    out->height = img.height;
    out->row_stride_px = img.row_stride_px;
    out->elapsed_ms = elapsed_ms;
    out->reprint = reprint;
    out->negative_was_cached = reprint;
    std::memset(out->progress_id, 0, sizeof out->progress_id);
    std::strncpy(out->progress_id, pid.c_str(), sizeof out->progress_id - 1);
}

// ---- render / reprint ------------------------------------------------------------

/* render(tier, reprint, out)
 *   lock; t0 := now
 *   reuse := reprint and the tier has a negative
 *   total := (reuse ? 0 : film nodes) + print nodes; new progress id; reset the record
 *   flag := engine.render_begin(); record it
 *   try:
 *     ctx := make_context(t)
 *     if !reuse: develop_tier(ctx, t)
 *     print_seed := draw_print_seed_locked()
 *     ctx.pitch := the tier's stored pitch              -- a reprint after a rebuild
 *     img := print_scan_encode(ctx, negative, live, print_seed)
 *     fill out; reprint = negative_was_cached = reuse
 *     progress: done, applied EV
 *   on failure:
 *     if this call developed: drop that negative        -- a reused one stays
 *     progress: not done; rethrow as a render failure
 */
void Session::render(std::string_view tier_name, bool reprint, spk_result* out) {
    std::lock_guard<std::mutex> g(mutex_);
    const clock::time_point t0 = clock::now();
    const Tier t = tier_or_throw(tier_name);
    Engine& e = *engine_;

    // The progress record: the node total is known once develop-or-reuse is decided.
    const bool reuse = reprint && static_cast<bool>(tiers_[tier_index(t)].negative);
    uint32_t total = 0;
    try {
        if (!reuse) total += film::node_count(*pipeline_.film);
        total += print::node_count(*pipeline_.print, e.variants());
    } catch (...) {
        total = 0;  // reported as pct 0 until done; the render itself reports the failure
    }
    const std::string pid = "p" + std::to_string(Engine::next_id());
    progress_.reset(pid, total, std::string(e.variants().effect_versions()));
    progress_.set_memory_pressure_critical(e.render_begin());
    struct EndGuard {
        Engine& e;
        ~EndGuard() { e.render_end(); }
    } end_guard{e};

    TierState* tsp = nullptr;
    bool developed = false;
    try {
        core::RenderContext ctx = make_context(t);
        TierState& ts = tiers_[tier_index(t)];
        tsp = &ts;
        if (!reuse) {
            developed = true;
            develop_tier(ctx, t);
        }
        const uint32_t print_seed = draw_print_seed_locked();
        ctx.pitch_um = ts.pitch_um;  // a reprint after a rebuild: the tier's stored pitch
        gpu::OutputImage img = print::print_scan_encode(ctx, *pipeline_.print, {ts.negative, pipeline_.live, print_seed});
        fill_result(out, img, ms_since(t0), reuse ? 1 : 0, pid);
        progress_.finish(true, ts.applied_ev);
    } catch (...) {
        // A cancelled or failed render caches no negative of its own; a cached one
        // it only reused stays.
        if (tsp && developed) drop_negative(*tsp);
        progress_.finish(false, std::nullopt);
        rethrow_as_render_failure();
    }
}

// ---- solve ----------------------------------------------------------------------------

/* solve(target) -> reply JSON; never changes parameters
 *   exposure: m := ensure_meter_locked()
 *             solved_params.exposure_compensation_ev := m.ev(auto_exposure_method)
 *             exposure_ev_by_method := {balanced, center, protect_highlights, protect_shadows}
 *   pack:     row := database lookup, else the current values
 *             solved_params.c/m/y_filter_neutral := row
 */
std::string Session::solve(std::string_view target) {
    std::lock_guard<std::mutex> g(mutex_);
    const bool exposure = target == "exposure" || target == "both";
    const bool pack = target == "filter_pack" || target == "both";
    if (!exposure && !pack)
        fail_user("unknown solve target '" + std::string(target) + "'; expected exposure, filter_pack or both");

    json::Value solved = json::Object{};
    json::Value r;
    if (exposure) {
        const film::MeterValues* m = nullptr;
        try {
            m = &ensure_meter_locked();
        } catch (...) {
            rethrow_as_render_failure();  // the meter failed: SPK_ERR_GPU
        }
        solved["exposure_compensation_ev"] = m->ev(params_->s(P::auto_exposure_method));
        json::Value by;
        by["balanced"] = m->balanced;
        by["center"] = m->center;
        by["protect_highlights"] = m->protect_highlights;
        by["protect_shadows"] = m->protect_shadows;
        r["exposure_ev_by_method"] = std::move(by);
    }
    if (pack) {
        std::array<double, 3> v{params_->f(P::c_filter_neutral), params_->f(P::m_filter_neutral),
                                params_->f(P::y_filter_neutral)};
        try {
            if (auto hit = print::solve_filter_pack(engine_->resources(), *params_, *film_profile_)) v = *hit;
        } catch (const std::exception& e) {
            fail_internal(std::string("filter resolution failed: ") + e.what());
        }
        solved["c_filter_neutral"] = v[0];
        solved["m_filter_neutral"] = v[1];
        solved["y_filter_neutral"] = v[2];
    }
    r["solved_params"] = std::move(solved);
    return json::write(r);
}

// ---- the print-preview table render ----------------------------------------------

/* preview_stock_lut(print_stock, tier, out, out_json)
 *   lock; stock := print_stock or the session's own
 *   table := print table of stock                      -- SPK_ERR_USER before any progress reset
 *   total := 1 + (no cached negative ? film nodes : 0); new progress id; reset the record
 *   if no cached negative: develop_tier (draws the film seed; no print seed in this call)
 *   img := print::preview_lut(negative, table)
 *   fill out; reprint = negative_was_cached = 1 always
 *   reply: {apply_backend, apply_ms, declared_pairing, lut_source, paired_film, print_stock, tier, warning?}
 *   warning iff film_stock != the table's paired_film
 */
void Session::preview_stock_lut(const char* print_stock, const char* tier_name, spk_result* out,
                                std::string* out_json) {
    std::lock_guard<std::mutex> g(mutex_);
    const clock::time_point t0 = clock::now();
    const Tier t = tier_or_throw(tier_name ? tier_name : "live");
    Engine& e = *engine_;
    const std::string stock = print_stock && *print_stock ? print_stock : params_->s(P::print_stock);
    const res::Resources& res = e.resources();
    const res::PrintLutTable table = res.print_luts.table(*res.blob, stock);  // SPK_ERR_USER: no table
    const res::PrintLutEntry* entry = res.print_luts.find(stock);

    const bool cached = static_cast<bool>(tiers_[tier_index(t)].negative);
    uint32_t total = 1;
    try {
        if (!cached) total += film::node_count(*pipeline_.film);
    } catch (...) {
    }
    const std::string pid = "p" + std::to_string(Engine::next_id());
    progress_.reset(pid, total, std::string(e.variants().effect_versions()));
    progress_.set_memory_pressure_critical(e.render_begin());
    struct EndGuard {
        Engine& e;
        ~EndGuard() { e.render_end(); }
    } end_guard{e};

    TierState& ts = tiers_[tier_index(t)];
    double apply_ms = 0;
    bool developed = false;
    try {
        core::RenderContext ctx = make_context(t);
        if (!ts.negative) {  // exactly as a reprint would
            developed = true;
            develop_tier(ctx, t);
        }
        ctx.pitch_um = ts.pitch_um;
        ctx.node("printing.preview_lut");  // the table pass is one node (the "+1" of the total)
        const clock::time_point ta = clock::now();
        gpu::OutputImage img = print::preview_lut(ctx, ts.negative, table);
        apply_ms = ms_since(ta);
        fill_result(out, img, ms_since(t0), 1, pid);  // always reprint = 1
        progress_.finish(true, ts.applied_ev);
    } catch (...) {
        if (developed) drop_negative(ts);
        progress_.finish(false, std::nullopt);
        rethrow_as_render_failure();
    }

    if (out_json) {
        json::Value r;
        r["apply_backend"] = "native-metal";
        r["apply_ms"] = apply_ms;
        r["declared_pairing"] = entry ? entry->declared_pairing : false;
        r["lut_source"] = "shipped";
        const std::string paired = entry ? entry->paired_film : "";
        r["paired_film"] = paired;
        r["print_stock"] = stock;
        r["tier"] = session::tier_name(t);
        const std::string& film = params_->s(P::film_stock);  // the alias id, not its base
        if (film != paired)
            r["warning"] = "the print table for '" + stock + "' was made for '" + paired + "', not for '" + film + "'";
        *out_json = json::write(r);
    }
}

// ---- control ------------------------------------------------------------------------------------------

void Session::cancel(const char* progress_id) { progress_.cancel(progress_id); }
std::string Session::progress(const char* progress_id) { return json::write(progress_.snapshot(progress_id)); }

uint64_t Session::source_bytes() const { return source_.buf.size(); }
uint64_t Session::cached_negative_bytes() const { return negative_bytes_.load(); }
uint32_t Session::negatives() const { return negative_count_.load(); }

}  // namespace spk
