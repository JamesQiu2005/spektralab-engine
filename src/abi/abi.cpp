// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 Guangzhou Songzi Intelligent Technology Services Co., Ltd.

/* Every exported spk_* symbol.
 *
 * for each entry point:
 *   clear last-error (except spk_capabilities, spk_get_params, spk_cancel, spk_string_free, spk_last_error)
 *   check required pointers -> SPK_ERR_INVALID_ARG
 *   call spk::Engine or spk::Session inside try
 *   catch -> map_exception -> status + last-error
 * nothing below may throw
 */
#include <cstdlib>
#include <cstring>
#include <new>
#include <string>

#include "gpu/gpu.hpp"
#include "session/engine.hpp"
#include "session/session.hpp"
#include "session/tiers.hpp"
#include "spektralab.h"
#include "util/error.hpp"

using spk::Engine;
using spk::Session;
namespace last_error = spk::last_error;

namespace {

Engine* E(spk_engine* e) { return reinterpret_cast<Engine*>(e); }
Session* S(spk_session* s) { return reinterpret_cast<Session*>(s); }

/* map_exception(): inside a catch, rethrow and classify
 *   spk::Error       -> its status
 *   std::bad_alloc   -> SPK_ERR_GPU, "out of memory"
 *   anything else    -> SPK_ERR_INTERNAL
 *   write the message to last-error
 */
int32_t map_exception() noexcept {
    try {
        throw;
    } catch (const spk::Error& e) {
        last_error::set(e.what());
        return e.status();
    } catch (const std::bad_alloc&) {
        last_error::set("out of memory");
        return SPK_ERR_GPU;  // allocation failures surface as SPK_ERR_GPU
    } catch (const std::exception& e) {
        last_error::set(std::string("internal error: ") + e.what());
        return SPK_ERR_INTERNAL;
    } catch (...) {
        last_error::set("internal error: unknown exception");
        return SPK_ERR_INTERNAL;
    }
}

template <class F>
int32_t guarded(F&& f) noexcept {
    try {
        return f();
    } catch (...) {
        return map_exception();
    }
}

int32_t invalid(const char* msg) noexcept {
    last_error::set(msg);
    return SPK_ERR_INVALID_ARG;
}

// A caller-owned copy of `s` for an out_json pointer (freed with spk_string_free).
char* dup_string(const std::string& s) {
    char* p = static_cast<char*>(std::malloc(s.size() + 1));
    if (!p) throw std::bad_alloc();
    std::memcpy(p, s.data(), s.size() + 1);
    return p;
}

void put_json(char** out_json, const std::string& s) {
    if (out_json) *out_json = dup_string(s);
}

void zero_result(spk_result* r) { std::memset(r, 0, sizeof *r); }

/* scrub_result(r): after a failure
 *   release r->texture if one was produced; zero r
 */
void scrub_result(spk_result* r) {
    if (r->texture) spk::gpu::Context::release_texture(r->texture);
    zero_result(r);
}

spk_engine* create(const char* dir, void* device, int32_t type) noexcept {
    last_error::clear();
    try {
        std::unique_ptr<Engine> e = Engine::create(dir, device, type);
        return reinterpret_cast<spk_engine*>(e.release());
    } catch (...) {
        map_exception();
        return nullptr;
    }
}

}  // namespace

extern "C" {

// ---- lifecycle -----------------------------------------------------------------------------

SPK_API spk_engine* spk_engine_create(const char* resources_dir, void* device) {
    return create(resources_dir, device, SPK_DEVICE_DESKTOP);
}

SPK_API spk_engine* spk_engine_create_for_device(const char* resources_dir, void* device, int32_t device_type) {
    return create(resources_dir, device, device_type);
}

SPK_API void spk_engine_destroy(spk_engine* engine) {
    last_error::clear();
    try {
        delete E(engine);
    } catch (...) {
        map_exception();
    }
}

SPK_API size_t spk_engine_trim(spk_engine* engine) {
    last_error::clear();
    if (!engine) return 0;
    try {
        return E(engine)->trim();
    } catch (...) {
        map_exception();
        return 0;
    }
}

SPK_API const char* spk_build_info(void) {
    last_error::clear();
    return Engine::build_info();
}

// ---- introspection ---------------------------------------------------------------------------

SPK_API const char* spk_capabilities(spk_engine* engine) {
    // does not touch last-error
    if (!engine) return "";
    try {
        return E(engine)->capabilities();
    } catch (...) {
        return "";
    }
}

SPK_API const char* spk_params_schema(spk_engine* engine) {
    last_error::clear();
    if (!engine) return "";
    try {
        return E(engine)->params_schema();
    } catch (...) {
        map_exception();
        return "";
    }
}

SPK_API const char* spk_memory_report(spk_engine* engine) {
    last_error::clear();
    if (!engine) return "";
    try {
        return E(engine)->memory_report();
    } catch (...) {
        map_exception();
        return "{}";
    }
}

// ---- warm-up ---------------------------------------------------------------------------------

SPK_API spk_status spk_warm_up(spk_engine* engine, const char* film_stock, const char* print_stock, char** out_json) {
    last_error::clear();
    if (!engine) return invalid("engine is NULL");
    return guarded([&] {
        std::string reply = E(engine)->warm_up(film_stock, print_stock);
        put_json(out_json, reply);
        return SPK_OK;
    });
}

// ---- sessions ---------------------------------------------------------------------------------

SPK_API spk_session* spk_open(spk_engine* engine, const spk_image* input, const char* params_delta_json,
                              char** out_json) {
    last_error::clear();
    if (!engine) { invalid("engine is NULL"); return nullptr; }
    if (!input) { invalid("input is NULL"); return nullptr; }
    if (!input->data) { invalid("input->data is NULL"); return nullptr; }
    try {
        spk::OpenInput in;
        in.host_data = input->data;
        in.width = input->width;
        in.height = input->height;
        in.channels = input->channels;
        std::string reply;
        Session* s = E(engine)->open(in, params_delta_json ? params_delta_json : "", out_json ? &reply : nullptr);
        if (out_json) {
            try {
                *out_json = dup_string(reply);
            } catch (...) {
                E(engine)->release(s);
                throw;
            }
        }
        return reinterpret_cast<spk_session*>(s);
    } catch (...) {
        map_exception();
        return nullptr;
    }
}

SPK_API spk_session* spk_open_device(spk_engine* engine, const spk_device_image* input, const char* params_delta_json,
                                     char** out_json) {
    last_error::clear();
    if (!engine) { invalid("engine is NULL"); return nullptr; }
    if (!input) { invalid("input is NULL"); return nullptr; }
    if (!input->buffer) { invalid("input->buffer is NULL"); return nullptr; }
    try {
        spk::OpenInput in;
        in.device_buffer = input->buffer;
        in.width = input->width;
        in.height = input->height;
        in.channels = input->channels;
        std::string reply;
        Session* s = E(engine)->open(in, params_delta_json ? params_delta_json : "", out_json ? &reply : nullptr);
        if (out_json) {
            try {
                *out_json = dup_string(reply);
            } catch (...) {
                E(engine)->release(s);
                throw;
            }
        }
        return reinterpret_cast<spk_session*>(s);
    } catch (...) {
        map_exception();
        return nullptr;
    }
}

SPK_API spk_status spk_set_params(spk_session* session, const char* params_delta_json, char** out_json) {
    last_error::clear();
    if (!session) return invalid("session is NULL");
    if (!params_delta_json) {
        last_error::set("params_delta_json is NULL");
        return SPK_ERR_USER;
    }
    return guarded([&] {
        std::string reply = S(session)->set_params(params_delta_json);
        put_json(out_json, reply);
        return SPK_OK;
    });
}

SPK_API spk_status spk_get_params(spk_session* session, char** out_json) {
    // does not clear last-error
    if (!session) return invalid("session is NULL");
    return guarded([&] {
        std::string reply = S(session)->get_params();
        put_json(out_json, reply);
        return SPK_OK;
    });
}

SPK_API void spk_session_release(spk_session* session) {
    last_error::clear();
    if (!session) return;
    try {
        Session* s = S(session);
        s->engine().release(s);
    } catch (...) {
        map_exception();
    }
}

// ---- rendering ------------------------------------------------------------------------------

/* render_common(session, tier, out, reprint)
 *   check session, out, tier name
 *   zero out
 *   session->render(tier, reprint, out)
 *   on failure: scrub out
 */
static spk_status render_common(spk_session* session, const char* tier, spk_result* out, bool reprint) {
    last_error::clear();
    if (!session) return invalid("session is NULL");
    if (!out) return invalid("out is NULL");
    if (!tier || !spk::session::parse_tier(tier)) {
        last_error::set(std::string("unknown tier '") + (tier ? tier : "") + "'; expected live, preview or full");
        return SPK_ERR_INVALID_ARG;
    }
    zero_result(out);
    int32_t st = guarded([&] {
        S(session)->render(tier, reprint, out);
        return SPK_OK;
    });
    if (st != SPK_OK) scrub_result(out);
    return st;
}

SPK_API spk_status spk_render(spk_session* session, const char* tier, spk_result* out) {
    return render_common(session, tier, out, false);
}

SPK_API spk_status spk_reprint(spk_session* session, const char* tier, spk_result* out) {
    return render_common(session, tier, out, true);
}

SPK_API void spk_result_free(spk_result* result) {
    last_error::clear();
    if (!result || !result->texture) return;
    spk::gpu::Context::release_texture(result->texture);
    result->texture = nullptr;
    result->rgba16 = nullptr;
}

// ---- solve ------------------------------------------------------------------------------------

SPK_API spk_status spk_solve(spk_session* session, const char* target, char** out_json) {
    last_error::clear();
    if (!session) return invalid("session is NULL");
    return guarded([&] {
        std::string reply = S(session)->solve(target && *target ? target : "both");
        put_json(out_json, reply);
        return SPK_OK;
    });
}

// ---- print-preview tables ---------------------------------------------------------------------

SPK_API const char* spk_print_lut_catalog(spk_engine* engine) {
    last_error::clear();
    if (!engine) return "{}";
    try {
        return E(engine)->print_lut_catalog();
    } catch (...) {
        map_exception();
        return "{}";
    }
}

SPK_API spk_status spk_print_lut_table(spk_engine* engine, const char* print_stock, const float** out_table,
                                       uint32_t* out_size) {
    last_error::clear();
    if (!engine) return invalid("engine is NULL");
    if (!print_stock) return invalid("print_stock is NULL");
    if (!out_table) return invalid("out_table is NULL");
    if (!out_size) return invalid("out_size is NULL");
    return guarded([&] {
        E(engine)->print_lut_table(print_stock, out_table, out_size);
        return SPK_OK;
    });
}

SPK_API spk_status spk_preview_stock_lut(spk_session* session, const char* print_stock, const char* tier,
                                         spk_result* out, char** out_json) {
    last_error::clear();
    if (!session) return invalid("session is NULL");
    if (!out) return invalid("out is NULL");
    const char* t = tier ? tier : "live";
    if (!spk::session::parse_tier(t)) {
        last_error::set(std::string("unknown tier '") + t + "'; expected live, preview or full");
        return SPK_ERR_INVALID_ARG;
    }
    zero_result(out);
    int32_t st = guarded([&] {
        std::string reply;
        S(session)->preview_stock_lut(print_stock, t, out, &reply);
        put_json(out_json, reply);
        return SPK_OK;
    });
    if (st != SPK_OK) scrub_result(out);
    return st;
}

// ---- output transform ---------------------------------------------------------------------------

SPK_API spk_status spk_output_transform(spk_engine* engine, const char* src_cs, const char* dst_cs,
                                        const char* gamut_compress_json, char** out_json, const float** out_cmax,
                                        uint32_t* out_cmax_count) {
    last_error::clear();
    if (!engine) return invalid("engine is NULL");
    if (!src_cs) return invalid("src_cs is NULL");
    if (!dst_cs) return invalid("dst_cs is NULL");
    if (!out_json) return invalid("out_json is NULL");
    if (!out_cmax) return invalid("out_cmax is NULL");
    if (!out_cmax_count) return invalid("out_cmax_count is NULL");
    return guarded([&] {
        std::string reply;
        const float* cmax = nullptr;
        uint32_t n = 0;
        E(engine)->output_transform(src_cs, dst_cs, gamut_compress_json, &reply, &cmax, &n);
        *out_json = dup_string(reply);
        *out_cmax = cmax;
        *out_cmax_count = n;
        return SPK_OK;
    });
}

// ---- control -----------------------------------------------------------------------------------

SPK_API void spk_cancel(spk_session* session, const char* progress_id) {
    if (!session) return;
    try {
        S(session)->cancel(progress_id);
    } catch (...) {
    }
}

SPK_API spk_status spk_progress(spk_session* session, const char* progress_id, char** out_json) {
    last_error::clear();
    if (!session) return invalid("session is NULL");
    return guarded([&] {
        std::string reply = S(session)->progress(progress_id);
        put_json(out_json, reply);
        return SPK_OK;
    });
}

SPK_API void spk_string_free(char* s) { std::free(s); }

SPK_API const char* spk_last_error(spk_engine* /*ignored*/) { return last_error::get(); }

}  // extern "C"
