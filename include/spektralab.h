// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 Guangzhou Songzi Intelligent Technology Services Co., Ltd.

/* spektralab-native: the public C ABI.
 *
 * Every entry point:
 *   linkage     C linkage, default visibility; no other symbol is exported
 *   failure     negative spk_status or NULL handle + message in the thread's last-error slot
 *   exceptions  never cross the ABI
 *   inputs      borrowed for the call; the engine copies what it keeps
 *   out_json    heap string owned by the caller: spk_string_free(*out_json)
 *   const char* return value: engine-owned, never freed; lifetime stated per function
 *   threads     calls on different sessions may run concurrently
 *               calls on one session serialise on its lock
 *               spk_cancel and spk_progress do not take that lock
 *               releasing an object while another thread is inside a call on it: undefined
 */
#ifndef SPEKTRALAB_H
#define SPEKTRALAB_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(SPK_BUILDING_LIBRARY)
#define SPK_API __attribute__((visibility("default")))
#else
#define SPK_API
#endif

/* ---- version numbers: reported again by spk_capabilities and spk_params_schema ------------ */
#define SPK_TRANSPORT_VERSION 1
#define SPK_SCHEMA_VERSION 1
#define SPK_VERSION_MAJOR 1
#define SPK_VERSION_MINOR 0
#define SPK_VERSION_PATCH 0
#define SPK_VERSION_STRING "1.0.0"
#define SPK_ENGINE_NAME "spektralab.native"

/* ---- status codes. There is no other code. -------------------------------------------------
 *   SPK_OK                success
 *   SPK_ERR_INVALID_ARG   required pointer NULL, unknown tier, unparsable gamut JSON
 *   SPK_ERR_USER          unknown or invalid parameter, stock, solve target or progress id;
 *                         a configuration that cannot be built
 *   SPK_ERR_RESOURCE      reserved: this engine does not return it
 *   SPK_ERR_UNSUPPORTED   reserved: this engine does not return it
 *   SPK_ERR_GPU           device, library or dispatch failure; out of memory;
 *                         every render failure that is not a cancellation
 *   SPK_ERR_IO            resource file missing or corrupt (at engine creation: a NULL engine)
 *   SPK_ERR_CANCELLED     the render was cancelled
 *   SPK_ERR_INTERNAL      a bug in the engine
 */
typedef int32_t spk_status;
#define SPK_OK 0
#define SPK_ERR_INVALID_ARG (-1)
#define SPK_ERR_USER (-2)
#define SPK_ERR_RESOURCE (-3)
#define SPK_ERR_UNSUPPORTED (-4)
#define SPK_ERR_GPU (-5)
#define SPK_ERR_IO (-6)
#define SPK_ERR_CANCELLED (-7)
#define SPK_ERR_INTERNAL (-8)

/* ---- device types: fixed for an engine's life ----------------------------------------------
 *   DESKTOP          every effect at its reference implementation
 *   PHONE, TABLET    halation=mobile.2, print_spatial=omitted; last release trims the scratch pool
 */
#define SPK_DEVICE_DESKTOP 0
#define SPK_DEVICE_PHONE 1
#define SPK_DEVICE_TABLET 2

/* ---- limits ------------------------------------------------------------------------------ */
#define SPK_MAX_PIXELS 151301008ULL /* 14204 x 10652 */

/* ---- opaque handles ---------------------------------------------------------------------- */
typedef struct spk_engine spk_engine;
typedef struct spk_session spk_session;

/* ---- structs (LP64 wire layouts) ------------------------------------ */

/* spk_image: caller-owned host image, borrowed for the call. Size 24, alignment 8.
 *   pixel (x, y), channel c = data[(y * width + x) * channels + c]
 *   values in the session's input_color_space
 *   non-finite values -> 0 and a 4th channel dropped when the engine copies them
 */
typedef struct spk_image {
    const float* data;  /* 0: tightly packed float32, row-major, row 0 at the top */
    uint32_t width;     /* 8 */
    uint32_t height;    /* 12 */
    uint32_t channels;  /* 16: 3 or 4 (a 4th channel is discarded) */
    /* 20: padding */
} spk_image;

/* spk_device_image: the same image in GPU-visible memory. Size 24, alignment 8.
 *   buffer must be on the engine's own device, >= width * height * channels * 4 bytes
 *   borrowed for the call; never retained past it
 */
typedef struct spk_device_image {
    void* buffer;       /* 0: id<MTLBuffer> on the engine's own device, borrowed for the call */
    uint32_t width;     /* 8 */
    uint32_t height;    /* 12 */
    uint32_t channels;  /* 16 */
    /* 20: padding */
} spk_device_image;

/* spk_result: one rendered image. Size 88, alignment 8.
 *   zero-filled on entry by every render call, so a failed call leaves it zeroed
 *   pixel (x, y), channel c = rgba16[4 * (y * row_stride_px + x) + c]; alpha = 65535
 *   padding pixels at each row's end are undefined
 *   texture is +1: spk_result_free(result), or adopt the reference, never both
 *   the image outlives its session and its engine
 */
typedef struct spk_result {
    const uint16_t* rgba16;      /* 0: RGBA16 pixels inside the texture's backing buffer */
    void* texture;               /* 8: id<MTLTexture>, handed over +1 */
    uint32_t width;              /* 16 */
    uint32_t height;             /* 20 */
    uint32_t row_stride_px;      /* 24: row pitch in pixels (8 bytes each), >= width */
    /* 28: padding */
    double elapsed_ms;           /* 32 */
    int32_t reprint;             /* 40 */
    int32_t negative_was_cached; /* 44 */
    char progress_id[40];        /* 48: NUL-terminated "p<N>" */
} spk_result;

#ifdef __cplusplus
#define SPK_STATIC_ASSERT(c, m) static_assert(c, m)
#define SPK_ALIGNOF(t) alignof(t)
#else
#define SPK_STATIC_ASSERT(c, m) _Static_assert(c, m)
#define SPK_ALIGNOF(t) _Alignof(t)
#endif

SPK_STATIC_ASSERT(sizeof(void*) == 8, "spektralab requires LP64");
SPK_STATIC_ASSERT(sizeof(spk_image) == 24, "spk_image size");
SPK_STATIC_ASSERT(SPK_ALIGNOF(spk_image) == 8, "spk_image alignment");
SPK_STATIC_ASSERT(offsetof(spk_image, data) == 0, "spk_image.data");
SPK_STATIC_ASSERT(offsetof(spk_image, width) == 8, "spk_image.width");
SPK_STATIC_ASSERT(offsetof(spk_image, height) == 12, "spk_image.height");
SPK_STATIC_ASSERT(offsetof(spk_image, channels) == 16, "spk_image.channels");
SPK_STATIC_ASSERT(sizeof(spk_device_image) == 24, "spk_device_image size");
SPK_STATIC_ASSERT(SPK_ALIGNOF(spk_device_image) == 8, "spk_device_image alignment");
SPK_STATIC_ASSERT(offsetof(spk_device_image, buffer) == 0, "spk_device_image.buffer");
SPK_STATIC_ASSERT(offsetof(spk_device_image, width) == 8, "spk_device_image.width");
SPK_STATIC_ASSERT(offsetof(spk_device_image, height) == 12, "spk_device_image.height");
SPK_STATIC_ASSERT(offsetof(spk_device_image, channels) == 16, "spk_device_image.channels");
SPK_STATIC_ASSERT(sizeof(spk_result) == 88, "spk_result size");
SPK_STATIC_ASSERT(SPK_ALIGNOF(spk_result) == 8, "spk_result alignment");
SPK_STATIC_ASSERT(offsetof(spk_result, rgba16) == 0, "spk_result.rgba16");
SPK_STATIC_ASSERT(offsetof(spk_result, texture) == 8, "spk_result.texture");
SPK_STATIC_ASSERT(offsetof(spk_result, width) == 16, "spk_result.width");
SPK_STATIC_ASSERT(offsetof(spk_result, height) == 20, "spk_result.height");
SPK_STATIC_ASSERT(offsetof(spk_result, row_stride_px) == 24, "spk_result.row_stride_px");
SPK_STATIC_ASSERT(offsetof(spk_result, elapsed_ms) == 32, "spk_result.elapsed_ms");
SPK_STATIC_ASSERT(offsetof(spk_result, reprint) == 40, "spk_result.reprint");
SPK_STATIC_ASSERT(offsetof(spk_result, negative_was_cached) == 44, "spk_result.negative_was_cached");
SPK_STATIC_ASSERT(offsetof(spk_result, progress_id) == 48, "spk_result.progress_id");

/* ==== entry points ========================================================================= */

/* ---- engine lifecycle ---- */

/* spk_engine_create(resources_dir, device) -> engine | NULL
 *   same as spk_engine_create_for_device(resources_dir, device, SPK_DEVICE_DESKTOP)
 *
 * spk_engine_create_for_device(resources_dir, device, device_type) -> engine | NULL
 *   in  resources_dir   required; layout in docs/RESOURCE_FORMAT.md
 *       device          id<MTLDevice>, retained for the engine's life; NULL = system default
 *       device_type     SPK_DEVICE_DESKTOP | _PHONE | _TABLET
 *   clear last-error
 *   load constants blob; neutral_print_filters.json and print_luts.json if present
 *   load spektralab.metallib; create the command queue
 *   run the numeric-mode probe; refuse unless it reports safe math
 *   build the schema; start the memory-pressure and idle-trim monitors
 *   out engine, freed with spk_engine_destroy
 *   NULL on any failure; reason in spk_last_error(NULL)
 */
SPK_API spk_engine* spk_engine_create(const char* resources_dir, void* device);
SPK_API spk_engine* spk_engine_create_for_device(const char* resources_dir, void* device,
                                                 int32_t device_type);
/* spk_engine_destroy(engine)
 *   NULL: no-op
 *   release every open session, then the engine
 *   results already handed out stay valid
 */
SPK_API void spk_engine_destroy(spk_engine* engine);
/* spk_engine_trim(engine) -> bytes freed
 *   NULL: 0
 *   free every idle scratch buffer now; buffers in use are kept; no picture changes
 *   safe while a render runs on another thread
 */
SPK_API size_t spk_engine_trim(spk_engine* engine);
/* spk_build_info() -> static string
 *   "spektralab-native <semver> (<YYYY-MM-DD>, math=safe)"
 */
SPK_API const char* spk_build_info(void);

/* ---- introspection: engine-owned JSON; NULL engine -> "" ----
 *
 * spk_capabilities(engine) -> JSON
 *   recomputed under the engine lock; valid until the next call of this function
 *   does not touch last-error
 */
SPK_API const char* spk_capabilities(spk_engine* engine);
/* spk_params_schema(engine) -> JSON {"fields":[...],"schema_version":1}
 *   computed once; valid for the engine's life
 */
SPK_API const char* spk_params_schema(spk_engine* engine);
/* spk_memory_report(engine) -> JSON
 *   recomputed each call; valid until the next call of this function; cheap enough for a UI timer
 */
SPK_API const char* spk_memory_report(spk_engine* engine);

/* ---- warm-up ----
 *
 * spk_warm_up(engine, film_stock, print_stock, out_json) -> status
 *   film_stock  NULL or "" -> kodak_portra_400; print_stock NULL or "" -> kodak_portra_endura
 *   out_json    may be NULL
 *   step "profiles": load both profiles
 *   step "pipeline": build a private film and print set-up
 *   step "kernels":  develop and print a 64 x 64 frame of 0.18, told it is a 1600 px frame
 *   a failing step is reported in the reply and ends the sequence; later steps do not run
 *   no session and no session's seed sequence is touched
 *   returns SPK_OK | SPK_ERR_INVALID_ARG (NULL engine)
 */
SPK_API spk_status spk_warm_up(spk_engine* engine, const char* film_stock, const char* print_stock,
                               char** out_json);

/* ---- sessions ----
 *
 * spk_open(engine, input, params_delta_json, out_json) -> session | NULL
 *   params_delta_json  NULL or "" = none; may hold "product_defaults": bool besides schema fields
 *   out_json           may be NULL; receives the open reply
 *   check frame: channels 3|4, size > 0, pixels <= SPK_MAX_PIXELS, long edge <= device limit
 *   free idle scratch on a frame switch
 *   validate delta
 *   params := defaults + delta stocks; load both profiles; digest
 *   force output_color_space = "ProPhoto RGB", output_cctf_encoding = true
 *   apply delta; digest again (a database hit overwrites explicit filter values)
 *   copy pixels into the session's own 3-channel plane; non-finite -> 0
 *   build the pipeline
 *   out session, freed with spk_session_release
 *   NULL on failure; reason in spk_last_error(NULL)
 */
SPK_API spk_session* spk_open(spk_engine* engine, const spk_image* input,
                              const char* params_delta_json, char** out_json);
/* spk_open_device(engine, input, params_delta_json, out_json) -> session | NULL
 *   spk_open, with the pixels read from input->buffer
 *   extra failures: buffer NULL, on another device, or shorter than the frame
 */
SPK_API spk_session* spk_open_device(spk_engine* engine, const spk_device_image* input,
                                     const char* params_delta_json, char** out_json);
/* spk_set_params(session, params_delta_json, out_json) -> status
 *   validate delta (before taking the session lock); first failure wins
 *   candidate := params + delta
 *   if the delta names a stock: reload profiles, digest candidate
 *   if the delta names a non-live field or a stock: build a new pipeline from candidate
 *   else: write the live fields onto the running pipeline
 *   commit; any refusal above leaves the session exactly as it was
 *   a shoot field in the delta: drop every tier's cached negative
 *   live tier size changed: drop the live tier's whole state
 *   a meter-key field changed: re-arm the meter
 *   out_json may be NULL; receives {"invalidated":"shoot"|"print","params":{...}}
 *   returns SPK_OK | SPK_ERR_INVALID_ARG (NULL session) | SPK_ERR_USER
 */
SPK_API spk_status spk_set_params(spk_session* session, const char* params_delta_json,
                                  char** out_json);
/* spk_get_params(session, out_json) -> status
 *   out_json may be NULL; receives the flat params object
 *   takes the session lock; does not clear last-error
 *   returns SPK_OK | SPK_ERR_INVALID_ARG (NULL session)
 */
SPK_API spk_status spk_get_params(spk_session* session, char** out_json);
/* spk_session_release(session)
 *   NULL: no-op
 *   free source, tier images, cached negatives; results already handed out stay valid
 *   phone or tablet engine, last open session: trim the scratch pool
 */
SPK_API void spk_session_release(spk_session* session);

/* ---- rendering ----
 *
 * spk_render(session, tier, out) -> status
 *   tier = "live" | "preview" | "full"
 *   discard the tier's cached negative
 *   develop afresh (new film seed when grain is on), cache the negative
 *   print and scan from it (new print seed when glare can draw)
 *   out := RGBA16 image, reprint = 0; caller frees with spk_result_free
 *   returns SPK_OK | SPK_ERR_INVALID_ARG | SPK_ERR_CANCELLED | SPK_ERR_GPU; on failure out is zeroed
 */
SPK_API spk_status spk_render(spk_session* session, const char* tier, spk_result* out);
/* spk_reprint(session, tier, out) -> status
 *   if tier has no cached negative: develop it first, reprint = 0
 *   else: reuse it, no film-side work, no film seed, reprint = 1
 *   print and scan from the negative (new print seed when glare can draw)
 *   out := RGBA16 image, caller frees with spk_result_free
 *   returns SPK_OK | SPK_ERR_INVALID_ARG | SPK_ERR_CANCELLED | SPK_ERR_GPU; on failure out is zeroed
 */
SPK_API spk_status spk_reprint(spk_session* session, const char* tier, spk_result* out);
/* spk_result_free(result)
 *   NULL result or NULL texture: no-op
 *   release the texture and with it the pixels; texture := NULL, rgba16 := NULL
 *   never on a result whose texture the host adopted
 */
SPK_API void spk_result_free(spk_result* result);

/* ---- solve ----
 *
 * spk_solve(session, target, out_json) -> status
 *   target = "exposure" | "filter_pack" | "both"; NULL or "" -> "both"
 *   never changes parameters
 *   exposure:    meter (or reuse) -> solved_params.exposure_compensation_ev, exposure_ev_by_method
 *   filter_pack: database row for (paper, illuminant, film), else current values
 *   out_json may be NULL
 *   returns SPK_OK | SPK_ERR_INVALID_ARG | SPK_ERR_USER (unknown target)
 *         | SPK_ERR_GPU (meter failed) | SPK_ERR_INTERNAL (filter resolution failed)
 */
SPK_API spk_status spk_solve(spk_session* session, const char* target, char** out_json);

/* ---- print-preview tables ----
 *
 * spk_print_lut_catalog(engine) -> bytes of print_luts.json, or "{}" when absent or engine is NULL
 *   engine-owned; valid for the engine's life
 */
SPK_API const char* spk_print_lut_catalog(spk_engine* engine);
/* spk_print_lut_table(engine, print_stock, out_table, out_size) -> status
 *   *out_table := S * S * S * 3 float32, row-major [i][j][k][ch]; engine-owned for the engine's life
 *   *out_size  := S
 *   returns SPK_OK | SPK_ERR_INVALID_ARG (any NULL) | SPK_ERR_USER (no table)
 */
SPK_API spk_status spk_print_lut_table(spk_engine* engine, const char* print_stock,
                                       const float** out_table, uint32_t* out_size);
/* spk_preview_stock_lut(session, print_stock, tier, out, out_json) -> status
 *   print_stock NULL or "" -> the session's own; tier NULL -> "live"; out_json may be NULL
 *   if tier has no cached negative: develop and cache it
 *   per pixel: densities -> trilinear table lookup -> RGBA16
 *   output is in the table's own encoding, not the session's output space
 *   no glare, no spatial effect; no print seed is drawn
 *   out.reprint = out.negative_was_cached = 1, even when a negative had to be developed
 *   returns SPK_OK | SPK_ERR_INVALID_ARG | SPK_ERR_USER (no table) | SPK_ERR_CANCELLED | SPK_ERR_GPU
 */
SPK_API spk_status spk_preview_stock_lut(spk_session* session, const char* print_stock,
                                         const char* tier, spk_result* out, char** out_json);

/* ---- output transform: a fetch, never a render ----
 *
 * spk_output_transform(engine, src_cs, dst_cs, gamut_compress_json, out_json, out_cmax, out_cmax_count)
 *   gamut_compress_json NULL or "" = defaults; everything else required
 *   matrix := XYZ->RGB(dst) . CAT02(white src -> white dst) . RGB->XYZ(src)
 *   *out_json       := matrix, transfer modes, gamut_compress constants
 *   *out_cmax       := rows * cols float32 C_max table of dst; engine-owned for the engine's life
 *   *out_cmax_count := rows * cols
 *   never falls back to another space
 *   returns SPK_OK | SPK_ERR_INVALID_ARG | SPK_ERR_USER (unknown space or algorithm)
 */
SPK_API spk_status spk_output_transform(spk_engine* engine, const char* src_cs, const char* dst_cs,
                                        const char* gamut_compress_json, char** out_json,
                                        const float** out_cmax, uint32_t* out_cmax_count);

/* ---- control ----
 *
 * spk_cancel(session, progress_id)
 *   if progress_id is NULL or the current id: set the current render's cancel flag
 *   checked between pipeline nodes; a cancel before a render resets its record is lost
 *   callable while a render holds the session lock
 */
SPK_API void spk_cancel(spk_session* session, const char* progress_id);
/* spk_progress(session, progress_id, out_json) -> status
 *   progress_id NULL or "" = current; out_json may be NULL
 *   consistent snapshot of the record; safe while a render runs on another thread
 *   returns SPK_OK | SPK_ERR_INVALID_ARG (NULL session) | SPK_ERR_USER (an id that is not current)
 */
SPK_API spk_status spk_progress(spk_session* session, const char* progress_id, char** out_json);
/* spk_string_free(s)
 *   free a string produced through out_json; NULL allowed
 */
SPK_API void spk_string_free(char* s);
/* spk_last_error(engine) -> message
 *   engine ignored, may be NULL; never NULL, "" when empty
 *   per-thread; valid until the next call on this thread that writes the slot
 *   read it immediately after the failing call
 */
SPK_API const char* spk_last_error(spk_engine* engine);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* SPEKTRALAB_H */
