# spektralab-engine: API specification

The contract of the C ABI declared in `include/spektralab.h`. Everything here describes this repository.
The library exports exactly the 26 functions listed in section 1.3 and nothing else.

Notation: `:=` assigns, `->` returns or maps, `--` starts a remark. Keys are shown in the writer's bytewise order.

---

## 1. The C ABI

### 1.1 Conventions

```
linkage    C linkage, default visibility; every other symbol is hidden (src/abi/exports.list)
platform   LP64, little-endian; Apple Metal (MTLDevice, MTLBuffer, MTLTexture cross as void*)
failure    negative spk_status, or a NULL handle, plus a message in the last-error slot
           no exception or signal crosses the ABI
```

**Status codes.** `spk_status` is `int32_t`. There is no other code.

| name | value | meaning |
|---|---|---|
| `SPK_OK` | 0 | success |
| `SPK_ERR_INVALID_ARG` | -1 | a required pointer is NULL, an unknown tier name, malformed gamut JSON |
| `SPK_ERR_USER` | -2 | an unknown or invalid parameter, stock, solve target or progress id; a configuration that cannot be built |
| `SPK_ERR_RESOURCE` | -3 | reserved; not returned |
| `SPK_ERR_UNSUPPORTED` | -4 | reserved; not returned |
| `SPK_ERR_GPU` | -5 | a device, library or dispatch failure; out of memory; every render failure that is not a cancellation |
| `SPK_ERR_IO` | -6 | a resource file is missing or corrupt; at creation this surfaces as a NULL engine; later, only from a print table whose stored size contradicts its shape |
| `SPK_ERR_CANCELLED` | -7 | the render was cancelled |
| `SPK_ERR_INTERNAL` | -8 | a bug in the engine |

**Last error.**

```
one slot per thread
spk_last_error(any) -> that thread's slot; never NULL; "" when empty
valid until the next call on the same thread that writes the slot
on entry, every function clears the slot, except:
  spk_capabilities, spk_get_params, spk_cancel, spk_string_free, spk_last_error
messages are English, for display; do not parse them
```

**Ownership.**

```
caller -> engine    device, input pixels or buffer, strings: borrowed for the call; the engine copies what it keeps
                    exception: the MTLDevice is retained for the engine's life
char** out_json     heap block owned by the caller; free with spk_string_free
const char* return  engine-owned; never free; lifetime per function
spk_result.texture  handed over +1
                    spk_result_free(result), or adopt the reference; never both
                    rgba16 points into its backing memory and lives as long as the texture
                    each render has a fresh buffer; a held result stays valid after its session
                    is released and after its engine is destroyed
spk_print_lut_table, cmax of spk_output_transform: engine-owned, valid for the engine's life
```

**Threads.**

```
engine lock   guards engine-wide state: session list, reply strings, caches
session lock  held for the whole of every call on that session, including a render
calls on different sessions  may run concurrently; capabilities.backend.concurrent = true
calls on one session         serialise
spk_cancel, spk_progress     do not take the session lock; safe during a render
release or destroy while another thread is inside a call on that object: undefined
```

**JSON on the wire.**

```
writer  UTF-8; compact; object keys in ascending bytewise order
        numbers: printf "%.17g" in the C locale, so integers print without a fraction
        non-finite number -> null
        escapes: " \ \n \r \t \b \f; other C0 controls as \u00xx, lower-case hex; other bytes unchanged
reader  strict RFC 8259
        no NaN, Infinity or hex floats; a number overflowing a double is a parse error
        duplicate key: the last one wins
        depth <= 64; only whitespace after the top-level value
        parse errors name the byte offset
        null in a numeric array of a profile reads as NaN; null in a parameter delta is a type error
```

### 1.2 Struct layouts (LP64)

Checked at compile time in the header.

```c
spk_image          /* size 24, align 8: caller-owned, borrowed for the call */
   0  const float* data      /* tightly packed float32, row-major, row 0 at the top, channels interleaved */
   8  uint32_t     width
  12  uint32_t     height
  16  uint32_t     channels  /* 3 or 4; a 4th channel is discarded */
  20  padding

spk_device_image   /* size 24, align 8 */
   0  void*        buffer    /* id<MTLBuffer> on the engine's own device; pixels from byte 0; borrowed */
   8  uint32_t     width
  12  uint32_t     height
  16  uint32_t     channels
  20  padding

spk_result         /* size 88, align 8: zero-filled on entry, so a failed call leaves it zeroed */
   0  const uint16_t* rgba16              /* section 4 */
   8  void*           texture             /* id<MTLTexture>, +1 */
  16  uint32_t        width
  20  uint32_t        height
  24  uint32_t        row_stride_px       /* pixels of 8 bytes each, >= width */
  28  padding
  32  double          elapsed_ms          /* wall time from just after the session lock is taken */
  40  int32_t         reprint             /* 1 if a cached negative was used */
  44  int32_t         negative_was_cached /* same value as reprint */
  48  char[40]        progress_id         /* NUL-terminated "p<N>" */
```

Device types: `SPK_DEVICE_DESKTOP = 0`, `SPK_DEVICE_PHONE = 1`, `SPK_DEVICE_TABLET = 2`.

Versions:

| what | value | also reported by |
|---|---|---|
| `SPK_TRANSPORT_VERSION` | 1 | `capabilities.transport_version` |
| `SPK_SCHEMA_VERSION` | 1 | `capabilities.schema_version`, `params_schema.schema_version` |
| build info | `spektralab-native 1.0.0 (<YYYY-MM-DD>, math=safe)` | `spk_build_info()`, `capabilities.version` |
| engine name | `spektralab.native` | `capabilities.engine` |

### 1.3 Entry points

The 26 exported functions:

```
engine      spk_engine_create  spk_engine_create_for_device  spk_engine_destroy  spk_engine_trim  spk_build_info
introspect  spk_capabilities  spk_params_schema  spk_memory_report
warm-up     spk_warm_up
sessions    spk_open  spk_open_device  spk_set_params  spk_get_params  spk_session_release
render      spk_render  spk_reprint  spk_result_free
solve       spk_solve
tables      spk_print_lut_catalog  spk_print_lut_table  spk_preview_stock_lut
output      spk_output_transform
control     spk_cancel  spk_progress  spk_string_free  spk_last_error
```

#### Engine

```
spk_engine_create(resources_dir, device) -> engine | NULL
  same as spk_engine_create_for_device(resources_dir, device, SPK_DEVICE_DESKTOP)

spk_engine_create_for_device(resources_dir, device, device_type) -> engine | NULL
  resources_dir  required; docs/RESOURCE_FORMAT.md
  device         id<MTLDevice>, retained for the engine's life; NULL = system default
  device_type    0 | 1 | 2; fixed for the engine's life
  fails when
    resources_dir is NULL; device_type is unknown
    the directory or constants blob is missing, truncated, of the wrong magic or version,
      or lacks a required entry
    neutral_print_filters.json exists and does not parse
    print_luts.json exists and is malformed
    no Metal device; spektralab.metallib missing or unloadable
    the numeric-mode probe does not report safe math (section 5)
  on failure: NULL, reason in spk_last_error(NULL)

spk_engine_destroy(engine)
  NULL: no-op
  release every open session, then the engine; results already handed out stay valid

spk_engine_trim(engine) -> size_t bytes freed
  NULL: 0
  free every idle scratch buffer now; no picture changes; safe during a render on another thread

spk_build_info() -> const char*
  static string, valid for the process's life
```

#### Introspection

```
spk_capabilities(engine) -> const char*
  NULL engine: ""
  recomputed under the engine lock; valid until the next spk_capabilities on this engine
  shape: section 2.11

spk_params_schema(engine) -> const char*
  NULL engine: ""
  computed once at creation; valid for the engine's life; shape: section 2.11

spk_memory_report(engine) -> const char*
  NULL engine: ""
  recomputed each call; valid until the next spk_memory_report; cheap; shape: section 2.10
```

#### Warm-up

```
spk_warm_up(engine, film_stock, print_stock, out_json) -> status
  film_stock   NULL or "" -> kodak_portra_400
  print_stock  NULL or "" -> kodak_portra_endura
  out_json     may be NULL
  step "profiles": load both profiles, resolve the neutral pack
  step "pipeline": build film and print set-up (fills the output gamut table of the working space)
  step "kernels":  64 x 64 of 0.18, told it is a 1600 px frame, through develop and print
  a failing step is reported in the reply and ends the sequence; later steps do not run
  no session is touched; no session's seed sequence advances
  returns SPK_OK, or SPK_ERR_INVALID_ARG for a NULL engine
  reply: section 2.10
```

#### Sessions

```
spk_open(engine, input, params_delta_json, out_json) -> session | NULL
  params_delta_json  NULL or "" = none
  out_json           may be NULL
  steps: section 2.2; reply: section 2.5
  fails when
    engine, input or input->data is NULL
    channels not 3 or 4; width or height 0; width * height > 151301008
    long edge > the device's maximum texture dimension
    the delta does not parse, product_defaults is not a bool, or the delta fails validation
    a profile does not load; the parameters cannot build a pipeline (section 2.4)

spk_open_device(engine, input, params_delta_json, out_json) -> session | NULL
  identical to spk_open except how the pixels arrive
  extra failures: buffer NULL; buffer on another MTLDevice;
                  buffer shorter than width * height * channels * 4 bytes

spk_set_params(session, params_delta_json, out_json) -> status
  params_delta_json  required; out_json may be NULL
  semantics: section 2.4; atomic: any refusal leaves the session exactly as it was
  returns SPK_OK
        | SPK_ERR_INVALID_ARG  NULL session
        | SPK_ERR_USER         NULL or unparsable JSON, validation failure, a profile that does
                               not load, a pipeline that cannot be built

spk_get_params(session, out_json) -> status
  out_json may be NULL; receives the flat params object
  takes the session lock; does not clear last-error
  returns SPK_OK | SPK_ERR_INVALID_ARG

spk_session_release(session)
  NULL: no-op
  free source, tier images, cached negatives; results already handed out stay valid
  phone or tablet engine, last open session: trim the scratch pool
```

#### Rendering

```
spk_render(session, tier, out) -> status
  tier = "live" | "preview" | "full"
  develop a fresh negative, cache it, print and scan: section 2.7; out.reprint = 0
  returns SPK_OK
        | SPK_ERR_INVALID_ARG  NULL session or out;
                               "unknown tier '<t>'; expected live, preview or full"
        | SPK_ERR_CANCELLED
        | SPK_ERR_GPU          any other failure

spk_reprint(session, tier, out) -> status
  if tier has no cached negative: develop it first, out.reprint = 0
  else: reuse it, out.reprint = 1
  print and scan from the negative
  returns as spk_render

spk_result_free(result)
  NULL result or NULL texture: no-op
  release the texture and with it the pixels; texture := NULL; rgba16 := NULL
  never on a result whose texture the host adopted
```

#### Solve

```
spk_solve(session, target, out_json) -> status
  target = "exposure" | "filter_pack" | "both"; NULL or "" -> "both"
  out_json may be NULL
  never changes parameters
  exposure:    run or reuse the session's meter
  filter_pack: database row for (print stock, enlarger illuminant, film stock), alias fallback
               included; no row: the session's current values
  returns SPK_OK
        | SPK_ERR_INVALID_ARG  NULL session
        | SPK_ERR_USER         unknown target
        | SPK_ERR_GPU          the meter failed
        | SPK_ERR_INTERNAL     filter resolution failed
  reply:
    { "exposure_ev_by_method": {"balanced": ev, "center": ev,      -- only when exposure was solved
                                "protect_highlights": ev, "protect_shadows": ev},
      "solved_params": {"c_filter_neutral": n, "m_filter_neutral": n, "y_filter_neutral": n,
                                                                   -- only when the pack was solved
                        "exposure_compensation_ev": ev} }          -- meter value of the session's
                                                                   -- auto_exposure_method
```

#### Print-preview tables

```
spk_print_lut_catalog(engine) -> const char*
  the bytes of print_luts.json verbatim; "{}" if absent or the engine is NULL
  engine-owned; valid for the engine's life

spk_print_lut_table(engine, print_stock, out_table, out_size) -> status
  *out_table := S * S * S * 3 float32, row-major [i][j][k][ch]; engine-owned for the engine's life
  *out_size  := S
  returns SPK_OK
        | SPK_ERR_INVALID_ARG  any NULL
        | SPK_ERR_USER         no index, no table (the message lists the available stocks), wrong shape

spk_preview_stock_lut(session, print_stock, tier, out, out_json) -> status
  print_stock  NULL or "" -> the session's own
  tier         NULL -> "live"
  out_json     may be NULL
  negative := the tier's cached one, else develop and cache it (as a reprint would)
  per pixel:  densities -> trilinear lookup (docs/RESOURCE_FORMAT.md) -> RGBA16
  no glare, no spatial effect, no print seed
  encoding: the table's own (catalog output_color_space), not the session's output space
  out.reprint = out.negative_was_cached = 1, always, even when a negative had to be developed
  returns SPK_OK | SPK_ERR_INVALID_ARG | SPK_ERR_USER (no table) | SPK_ERR_CANCELLED | SPK_ERR_GPU
  reply:
    { "apply_backend": "native-metal", "apply_ms": ms, "declared_pairing": bool,
      "lut_source": "shipped", "paired_film": str, "print_stock": str, "tier": str,
      "warning": str }    -- warning iff the session's film_stock != paired_film
```

#### Output transform

A fetch, never a render.

```
spk_output_transform(engine, src_cs, dst_cs, gamut_compress_json, out_json, out_cmax, out_cmax_count) -> status
  src_cs, dst_cs        baked colour-space names; required
  gamut_compress_json   NULL or "" = defaults
  out_json, out_cmax, out_cmax_count  required
  never falls back to another space
  returns SPK_OK
        | SPK_ERR_INVALID_ARG  any required NULL; gamut JSON unparsable or not an object
        | SPK_ERR_USER         unknown space; space without matrices or whitepoint;
                               algorithm other than "cam16ucs" or "off"

  gamut JSON, every key optional:
    { "algorithm": "cam16ucs" | "off",                 -- default "cam16ucs"
      "knee": [threshold, limit, power],               -- default [0, 1, 6]
      "lightness_compression_active": bool,            -- default true
      "lightness_compression": [threshold, limit, power] }  -- default [0.7, 1.0, 2.2]
    an array that is not exactly 3 numbers is ignored

  reply:
    { "gamut_compress": { "algorithm": str,
                          "cmax_cols": 720, "cmax_rows": 64,
                          "k": [22 numbers],
                          "lightness_compression_active": bool,
                          "m_to_rgb": [9 numbers], "m_to_xyz": [9 numbers] },
      "matrix": [9 numbers],
      "source_cctf_mode": int, "source_color_space": str,
      "target_cctf_mode": int, "target_color_space": str }

  matrix, row-major, out_i = sum_j m[3i + j] x_j:
    matrix := XYZ->RGB(dst) . CAT02(white src -> white dst) . RGB->XYZ(src)
    the CAT02 product is skipped when |a - b| <= 1e-8 + 1e-5 |b| for both coordinates, b = dst white
  cctf mode: 0 sRGB curve | 1 ProPhoto | 2 Adobe RGB (1998) | 3 BT.709 curve | 4 identity
  k: [F_L, N_bb, N_cb, n, z, A_w, c, N_c, J first, J last, h first, h step,
      knee0, knee1, knee2, lc0, lc1, lc2, white J', D_R, D_G, D_B]
     lightness compression inactive: (lc0, lc1, lc2) = (0, 1, 1)
  with algorithm "off" the tables are still returned

  *out_cmax := rows * cols float32, row-major (lightness rows, hue columns)
  *out_cmax_count := rows * cols = 46080
  engine-owned for the engine's life
```

#### Control

```
spk_cancel(session, progress_id)
  NULL session: no-op
  if progress_id is NULL or the current id: mark the current render cancelled
  else nothing
  callable from another thread while a render holds the session lock

spk_progress(session, progress_id, out_json) -> status
  progress_id  NULL or "" = the current one; out_json may be NULL
  consistent snapshot; safe during a render on another thread
  returns SPK_OK
        | SPK_ERR_INVALID_ARG  NULL session
        | SPK_ERR_USER         "unknown progress id '<id>'"
  reply: section 2.9

spk_string_free(s)
  free a string produced through out_json; NULL allowed

spk_last_error(engine) -> const char*
  engine ignored, may be NULL; section 1.1
```

---

## 2. Sessions

### 2.1 What an engine holds

```
at creation   constants blob (required)
              neutral_print_filters.json (optional: absent = every lookup misses)
              print_luts.json (optional: absent = the table entry points refuse)
              the kernel library
on demand     profiles/<stock>.json: at open, warm-up and every stock change; read each time
engine cache  CAM16 output tables, keyed by output colour space; a pure function of the key
pipeline      the spectral-upsampling table is computed per pipeline build
```

### 2.2 Opening a frame

```
open(engine, input, delta):
  1 check the frame
      channels 3 | 4; size > 0; width * height <= 151301008 (14204 x 10652)
      long edge <= capabilities.max_texture_dimension_2d
      device buffer: same device, long enough
  2 frame switch: if no session is open or the size differs from the last frame: free idle scratch
  3 parse the delta
      must be a JSON object: "params_delta must be an object"
      remove "product_defaults"; must be a bool; not a schema field; no layer; not echoed
      validate the rest (section 2.4)
  4 parameter tree
      p := schema defaults
      p.film_stock, p.print_stock := delta's, if any
      load both profiles
      digest(p)
      p.output_color_space := "ProPhoto RGB"; p.output_cctf_encoding := true
      apply delta
      digest(p)
  5 copy pixels into the session's own plane
      3 float32 channels, interleaved, row-major, top row first
      a 4th channel dropped; NaN, +inf, -inf -> 0; finite values unchanged
  6 build the pipeline; failure fails the open
  7 reply (section 2.5); id := "s" + next(N)
      N: process-wide counter from 1, shared with progress ids
```

**The digest.**

```
digest(p):
  row := database[print_stock][enlarger_illuminant][film id]
  if that leaf is not an array and the film is an alias: row := ...[base id]
  row is 3 numbers: c/m/y_filter_neutral := row
  else: leave them
```

Consequences:

- At open, the second digest runs after the delta, so explicit `c/m/y_filter_neutral` in an open delta are overwritten whenever the database has the pair. They take effect only for pairs it lacks.
- Film-dependent constants (coupler gammas, halation preset) are resolved by every film set-up build, not by the digest.

**Product defaults.**

```
open delta "product_defaults": true
  -> print-exposure compensation off for the session's life
     on  (default): print exposure compensates, so exposure_compensation_ev only moves the negative
                    along the film curve
     off:           exposure_compensation_ev moves the finished print's brightness
in a spk_set_params delta: an unknown parameter, SPK_ERR_USER
```

### 2.3 Layers: what invalidates what

| layer | meaning | cached state dropped by a change |
|---|---|---|
| `shoot` | camera, exposure, film, geometry, input encoding | every tier's cached negative; tier images are kept |
| `print` | enlarger, paper, scan, output encoding | nothing; the next `spk_reprint` reuses the negatives |

```
film_stock          shoot; reload profiles, digest
print_stock         print; reload profiles, digest; cached negatives kept
                    the digest overrides c/m/y_filter_neutral of the same delta on a database hit
preview_long_edge   print; if the live tier's size changes: drop the live tier's whole state
                    (image, negative, applied EV, pitch); other tiers untouched
meter key           input_color_space, input_cctf_decoding, geometry_crop_x, geometry_crop_y,
                    geometry_crop_w, geometry_crop_h, geometry_rotation_deg, geometry_quarter_turns,
                    geometry_flip_h, geometry_flip_v
                    a change re-arms the meter; nothing else re-meters
live fields         print_exposure, m_filter_shift, y_filter_shift, preflash_exposure, filter_shift_scale
                    written onto the running pipeline without a rebuild
every other field   rebuilds the pipeline; invisible in pixels, except that a rebuild restarts
                    the seed sequence (section 2.8)
invalidated reply   "shoot" if any field of the delta is shoot layer, else "print"
                    {} -> "print", no change
```

### 2.4 Delta semantics

```
set_params(session, delta):
  validate; before the session lock; first failure wins; status SPK_ERR_USER:
    top level is an object            params_delta must be an object
    for each key, ascending bytewise:
      name in schema                  unknown parameter '<name>'
      JSON type matches               '<name>' expects <float|int|bool|str>
                                      float, int: any number; bool: true or false; str: string
      enumeration                     'auto_exposure_method' = '<v>' is not one of <list>
      closed range, NaN refused       '<name>'=<v> is outside [<lo>, <hi>]   (numbers in %f form)
      int: range checked as given, then rounded half away from zero

  lock
  cand := params + delta
  if delta names a stock:
    film, print := reload profiles       -- failure: SPK_ERR_USER, session unchanged
    digest(cand)
  else:
    no digest                            -- enlarger_illuminant alone does not re-resolve the pack
  if delta names a non-live field or a stock:
    pipeline := build(cand)              -- failure: SPK_ERR_USER, nothing has changed
  else:
    pipeline.live := live fields of cand
  commit cand
  invalidate (section 2.3)
  reply {"invalidated": ..., "params": {...}}
```

**Refused at pipeline build** (`SPK_ERR_USER` from `spk_set_params`, NULL from open):

- `input_color_space` or `output_color_space` is not a baked colour space;
- `enlarger_illuminant`, or a profile's `reference_illuminant` or `viewing_illuminant`, cannot be resolved;
- `extended_dynamic_range` true with a print profile that has no `edr_tone_map`;
- `grain_active` and `grain_sublayers_active` true with a film profile lacking `density_curves_layers`;
- a film profile whose `hanatos2025_adaptation_window_params` has 8 values;
- `scan_film` false with a print profile lacking a `density_curves_model` of type `cdfs`.

**Refused when a develop runs, not at build.**

```
positive film and scan_film and (scanner_black_correction or scanner_white_correction):
  spk_open, spk_set_params: accepted
  spk_render, or a reprint or preview that has to develop:
    SPK_ERR_GPU, out zeroed, no negative cached
    "black/white correction on a directly scanned positive film is not implemented by the native engine"
a negative film under the same settings: not refused; a direct scan of a negative is not corrected
```

### 2.5 Reply shapes of open, set_params and get_params

```
spk_open, spk_open_device:
  { "capabilities": {...},                      -- section 2.11, at this moment
    "detected_input": { "input_cctf_decoding": bool, "input_color_space": str,
                        "input_color_space_source": "caller" },
    "meta": { "height": H, "megapixels": round(W * H / 1e4) / 100, "source": "<in-process>", "width": W },
    "params": {...},
    "session_id": "s<N>" }

spk_set_params:
  { "invalidated": "shoot" | "print", "params": {...} }

spk_get_params:
  the flat params object itself

params:
  exactly 48 keys, one per schema row, no nesting
  float, int -> number (an int is a whole number)
  bool -> true | false
  str -> string; a stock field holds the id the session addresses (an alias reads as its own id)

on failure none of the three writes out_json
```

The host reads the resolved `output_color_space` from `params`.

### 2.6 Tiers

| tier | long edge | use |
|---|---|---|
| `live` | `preview_long_edge`, default 1600 | interactive edits |
| `preview` | 3400 | judging grain and halation |
| `full` | the frame's own | export; never a reduced size |

**The tier image** is built on first use from the session's plane, on raw input values, before decoding and before geometry.

```
downscale(frame W x H, long edge L):
  M := max(W, H)
  if L == 0 or M <= L: the frame itself                -- no upscaling
  s := L / M (float64)
  oh := max(1, floor(H s)); ow := max(1, floor(W s))
  per axis of input n, output m:
    f := n / m; sigma := max(0, (f - 1) / 2); r := floor(4 sigma + 0.5)
    w_k := exp(-1/2 (k / sigma)^2), k = -r..r, normalised to sum 1, float64, stored float32
    r == 0: axis skipped
  vertical pass, then horizontal; each sample, all channels with the same weights:
    acc := 0; for k = -r..r ascending: acc := fma(w_k, x[mirror(i + k)], acc)
    mirror: "c b | a b c d | c b", period 2(n - 1); an axis of length 1 reads index 0
  bilinear, per output index o, in exact integers:
    num := (2o + 1) n - m; den := 2m
    if num > den (n - 1): num := 2 den (n - 1) - num
    if num < 0: num := -num
    i0 := floor(num / den); i1 := min(i0 + 1, n - 1); t := float32(num - i0 den) / float32(den)
    top := fma(tx, b - a, a); bot := fma(tx, e - d, d); out := fma(ty, bot - top, top)
```

The floor can give `L - 1`: a frame of long edge 2156 at `L = 1600` gives 1599.

```
output size   tier image size after geometry; quarter turns swap the axes; a crop shrinks it
pixel pitch   pitch_um := film_format_mm * 1000 / long edge of the uncropped tier image
tier ratio    rho_tier := long edge of the tier image / long edge of the frame
meter         once, at a fixed long edge of 1600, regardless of tier and preview_long_edge
              image := the live tier image when preview_long_edge == 1600 and it exists,
                       else downscale(frame, 1600); then decode and geometry
              computed on the first render or solve that needs it
              kept until a meter-key field changes
              not computed for a render with auto_exposure = false
              every tier applies the same EV for the session's auto_exposure_method
```

### 2.7 Render, reprint and the cached negative

Each tier has at most one cached negative: the film-side density plane at the tier's post-geometry size. With it the tier keeps the applied EV and the pitch.

| call | tier has a cached negative | tier has none |
|---|---|---|
| `spk_render` | discard it; develop with a new film seed; cache; print. `reprint = 0` | develop, cache, print. `reprint = 0` |
| `spk_reprint` | print from the cache; no film-side work; no film seed. `reprint = 1` | develop, cache, print. `reprint = 0` |

```
render(session, tier, reprint):
  reuse := reprint and tier has a negative
  new progress id; reset the progress record
  if !reuse:
    drop the negative
    ev := meter value, or none when auto_exposure is off
    film_seed := grain_active ? seeds.next() : 0x5EED
    negative := develop(tier image, ev, film_seed)
  print_seed := glare can draw ? seeds.next() : 0
  image := print_scan_encode(negative, live fields, print_seed, the tier's stored pitch)
  on failure:
    if this call developed: drop that negative         -- a reused one stays
    progress.done := false
```

A reprint after a print-layer rebuild uses the negative made by the previous pipeline.

### 2.8 Seeds and determinism

All randomness (grain and glare) comes from one generator per pipeline instance. A pipeline instance is created at open and at every non-live `spk_set_params`. Warm-up uses a private one.

```
state S0 := 0x243F6A8885A308D3
draw (splitmix64; Steele, Lea and Flood, OOPSLA 2014):
  S := S + 0x9E3779B97F4A7C15 (mod 2^64)
  z := S
  z := (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
  z := (z ^ (z >> 27)) * 0x94D049BB133111EB
  z := z ^ (z >> 31)
  seed := low 32 bits of z
first three seeds after a (re)build: 0x4ABEA221, 0x23148989, 0x609DFE03

film seed    one per develop, if grain_active; else 0x5EED
print seed   one per print run, if glare_active and 0.03 * glare_amount > 0; else 0
             drawn on those conditions alone, also under scan_film, where glare does not act
order        within spk_render: film draw, then print draw
             spk_reprint: print draw only
             spk_preview_stock_lut: a film draw if it develops; no print draw
per pixel    Philox4x32-10 (Salmon et al., SC 2011), keyed on pixel index, stream and seed
             counter = (pixel, stream, j, 0), key = (seed, 0)
```

Consequences:

- Grain and glare both off: a render is a pure function of the parameters and the frame. Render and reprint give byte-identical pictures, and so do two engines.
- Grain or glare on: the realisation depends on the sequence of develops and prints since the pipeline was built. Open, `spk_render("live")`: draw 1 for grain, draw 2 for glare. A following `spk_reprint("live")`: draw 3 for glare, grain unchanged.

### 2.9 Cancellation and progress

```
every render-type call (render, reprint, preview table):
  new progress id "p<N>", copied into spk_result.progress_id
  reset the record; the cancel flag is cleared

cancel flag: checked at each pipeline node
  cancelled render: stops at the next node, finishes encoded GPU work, caches no negative
                    of its own, returns SPK_ERR_CANCELLED with out zeroed
  a cancel arriving before a render resets its record is lost
```

Progress reply:

```
{ "auto_exposure_ev": number | null,    -- EV applied to this render's negative, the cached one's on
                                        -- a reprint; null while running, after a failure, or meter off
  "cancelled": bool,
  "done": bool,                         -- true after a successful render only
  "effect_versions": str,               -- section 2.11; "" before any render
  "memory_pressure_critical": bool,     -- a critical event arrived as this render started; one-shot
  "node_times": {},                     -- always empty
  "pct": number,                        -- done ? 100 : 100 * fired / total
  "progress_id": str,                   -- "" before any render
  "stage": str }                        -- "queued", then a dotted node label; display text, do not parse
```

### 2.10 Warm-up reply and memory report

```
warm-up reply:
  { "already_warm": false,
    "film_stock": str, "print_stock": str,
    "render_core": "native-metal",
    "steps": [ {"error"?: str, "ms": number, "name": "profiles" | "pipeline" | "kernels", "ok": bool} ],
    "total_ms": number }                -- ms rounded to 0.01

scratch memory is given back
  at spk_engine_trim
  at an open with no session open, or of a different frame size
  at the last release (phone and tablet only)
  on an OS memory-pressure event
    warn:     keep up to the running frame's peak of free scratch, free the rest
    critical: free everything idle, flag the next render
  after 60 s with no render in flight (checked every 5 s)
```

Memory report:

```
{ "total_bytes": pool.total_bytes + persistent.bytes,
  "pool": {
    "total_bytes", "live_bytes", "free_bytes", "pending_bytes",       -- total = live + free + pending
    "buffers", "live_buffers", "free_buffers", "pending_buffers",
    "frame_high_water_bytes",                                         -- peak live bytes of this frame
    "audit": { "allocations", "reuses", "persistent_allocations",
               "reuse_bytes_requested", "reuse_bytes_taken",
               "reuse_max_taken", "reuse_max_taken_for_request",
               "pending_held", "live_underflows", "over_releases", "reclaim_while_encoding" } },
  "persistent": {
    "bytes", "buffers",                                               -- everything outside the pool,
                                                                      -- the mapped blob included
    "file_backed_bytes", "file_backed_buffers",                       -- part of bytes
    "source_bytes", "cached_negative_bytes", "negatives",             -- sums over open sessions
    "sessions": [ {"cached_negative_bytes", "negatives", "session_id", "source_bytes"} ] },
  "pressure": { "monitor": bool, "warn_events", "critical_events", "critical_pending": bool,
                "idle_trim_seconds",                                  -- 60, or 0 without a timer
                "idle_trims" } }

not counted: the host's memory, an input buffer passed to spk_open_device, textures already handed over
pending_bytes, pending_buffers, pending_held, over_releases, reclaim_while_encoding: always 0 here
live_underflows: 0 in a correct engine
```

### 2.11 Capabilities and params_schema

```
capabilities:
  { "backend": {
      "concurrent": true,
      "device": "desktop" | "phone" | "tablet",
      "effect_versions": str,
      "gpu": device name,
      "gpu_available": true,
      "host": "native",
      "math_mode": "safe (probe: fma contraction present, no reassociation)",
      "render_core": "native-metal",
      "session_cache": { "enabled": false, "entries": open sessions, "max_entries": 0,
                         "setup_cam16": cached CAM16 tables },
      "spectral": "native",
      "working_precision": "float32" },
    "engine": "spektralab.native",
    "max_mp": 151.301008,
    "max_pixels": 151301008,
    "max_texture_dimension_2d": 16384 | 8192,
    "schema_version": 1,
    "tiers": { "full": null, "live": 1600, "preview": 3400 },   -- live is the default; a session
                                                                -- renders at its preview_long_edge
    "transport_version": 1,
    "version": build info string }

effect_versions: put it in every cache key built from a render
  desktop         halation=reference;grain=reference;glare=reference;print_spatial=reference;dir_couplers=reference
  phone, tablet   halation=mobile.2;grain=mobile.1;glare=mobile.1;print_spatial=omitted;dir_couplers=reference
  print_spatial=omitted: the scanner lens blur and the scan unsharp mask are not part of the render
  mobile.2 halation: large components computed on a coarser grid
  mobile.1 grain and glare: the same values as the reference

params_schema:
  { "fields": [ {"default", "layer": "shoot"|"print", "live": bool, "name",
                 "path",                                -- dotted; informational
                 "range": [lo, hi],                     -- only when the field has one
                 "type": "float"|"int"|"bool"|"str"} ],
    "schema_version": 1 }
  one entry per field, in the order of section 3
  default: the schema's default, before any digest
```

---

## 3. The parameter schema

48 rows, in `params_schema` order. L = layer (S shoot, P print). A name not listed here is refused as an unknown parameter.

### Stocks

| name | path | type | range | default | L | live | meaning |
|---|---|---|---|---|---|---|---|
| `film_stock` | `film.info.stock` | str | profile id `[A-Za-z0-9_-]{1,96}`, checked at load | `kodak_portra_400` | S | | Film profile |
| `print_stock` | `print.info.stock` | str | profile id | `kodak_portra_endura` | P | | Paper or print film profile |

### Camera and exposure

| name | path | type | range | default | L | live | meaning |
|---|---|---|---|---|---|---|---|
| `exposure_compensation_ev` | `camera.exposure_compensation_ev` | float | [-8, 8] | 0 | S | | Exposure compensation, stops |
| `auto_exposure` | `camera.auto_exposure` | bool | | true | S | | Apply the session meter's EV |
| `auto_exposure_method` | `camera.auto_exposure_method` | str | `center_weighted`, `average`, `median`, `balanced`, `center`, `protect_highlights`, `protect_shadows` | `center_weighted` | S | | Which meter value is applied |
| `film_format_mm` | `camera.film_format_mm` | float | [4, 200] | 35 | S | | Long edge of the film frame, mm; sets the pixel pitch |
| `lens_blur_um` | `camera.lens_blur_um` | float | [0, 200] | 0 | S | | Camera lens blur, um on film |

### Halation

| name | path | type | range | default | L | live | meaning |
|---|---|---|---|---|---|---|---|
| `halation_active` | `film_render.halation.active` | bool | | true | S | | Halation and scatter stage on |
| `halation_amount` | `film_render.halation.halation_amount` | float | [0, 4] | 1 | S | | Area multiplier: strength x amount, every sigma x sqrt(amount); 0 = off |
| `halation_boost_ev` | `film_render.halation.boost_ev` | float | [0, 8] | 0 | S | | Highlight boost before halation, stops |

### Grain

| name | path | type | range | default | L | live | meaning |
|---|---|---|---|---|---|---|---|
| `grain_active` | `film_render.grain.active` | bool | | true | S | | Grain on; also conditions the film-seed draw |
| `grain_sublayers_active` | `film_render.grain.sublayers_active` | bool | | true | S | | Per-sublayer grain; needs `density_curves_layers` |

### Development

| name | path | type | range | default | L | live | meaning |
|---|---|---|---|---|---|---|---|
| `dir_couplers_active` | `film_render.dir_couplers.active` | bool | | true | S | | DIR couplers on |
| `dir_couplers_amount` | `film_render.dir_couplers.amount` | float | [0, 4] | 1 | S | | Coupler strength |
| `density_curve_gamma` | `film_render.density_curve_gamma` | float | [0.2, 4] | 1 | S | | Film contrast factor on the density curves |

### Input

| name | path | type | range | default | L | live | meaning |
|---|---|---|---|---|---|---|---|
| `input_color_space` | `io.input_color_space` | str | a baked colour space; else refused at build | `ProPhoto RGB` | S | | Space of the input pixels; meter key |
| `input_cctf_decoding` | `io.input_cctf_decoding` | bool | | false | S | | Decode the input's transfer function first; meter key |

### Geometry (all meter key)

| name | path | type | range | default | L | live | meaning |
|---|---|---|---|---|---|---|---|
| `geometry_crop_x` | `io.geometry.crop_x` | float | [0, 1] | 0 | S | | Crop origin x, fraction of width |
| `geometry_crop_y` | `io.geometry.crop_y` | float | [0, 1] | 0 | S | | Crop origin y, fraction of height |
| `geometry_crop_w` | `io.geometry.crop_w` | float | [1e-4, 1] | 1 | S | | Crop width, fraction |
| `geometry_crop_h` | `io.geometry.crop_h` | float | [1e-4, 1] | 1 | S | | Crop height, fraction |
| `geometry_rotation_deg` | `io.geometry.rotation_deg` | float | [-45, 45] | 0 | S | | Fine rotation, degrees |
| `geometry_quarter_turns` | `io.geometry.quarter_turns` | int | [0, 3] | 0 | S | | Quarter turns |
| `geometry_flip_h` | `io.geometry.flip_h` | bool | | false | S | | Horizontal flip |
| `geometry_flip_v` | `io.geometry.flip_v` | bool | | false | S | | Vertical flip |

### Enlarger

| name | path | type | range | default | L | live | meaning |
|---|---|---|---|---|---|---|---|
| `print_exposure` | `enlarger.print_exposure` | float | [0.05, 20] | 1 | P | yes | Print exposure multiplier |
| `m_filter_shift` | `enlarger.m_filter_shift` | float | [-1, 1] | 0 | P | yes | Magenta: M = neutral + shift x `filter_shift_scale`, in CC |
| `y_filter_shift` | `enlarger.y_filter_shift` | float | [-1, 1] | 0 | P | yes | Yellow, likewise |
| `c_filter_neutral` | `enlarger.c_filter_neutral` | float | [0, 200] | 0 | P | | Neutral cyan filtration, CC; digested |
| `m_filter_neutral` | `enlarger.m_filter_neutral` | float | [0, 200] | 65 | P | | Neutral magenta filtration, CC; digested |
| `y_filter_neutral` | `enlarger.y_filter_neutral` | float | [0, 200] | 55 | P | | Neutral yellow filtration, CC; digested |
| `preflash_exposure` | `enlarger.preflash_exposure` | float | [0, 1] | 0 | P | yes | Preflash exposure |
| `enlarger_illuminant` | `enlarger.illuminant` | str | an illuminant name; else refused at build | `TH-KG3` | P | | Enlarger lamp; no digest on its own |

### Print, scan and output

| name | path | type | range | default | L | live | meaning |
|---|---|---|---|---|---|---|---|
| `glare_active` | `print_render.glare.active` | bool | | true | P | | Print glare on; also conditions the print-seed draw |
| `scanner_white_correction` | `scanner.white_correction` | bool | | false | P | | Scanner white-level correction |
| `scanner_black_correction` | `scanner.black_correction` | bool | | false | P | | Scanner black-level correction |
| `scanner_lens_blur` | `scanner.lens_blur` | float | [0, 20] | 0 | P | | Scanner blur, pixels; part of `print_spatial` |
| `output_color_space` | `io.output_color_space` | str | a baked colour space | `ProPhoto RGB` (forced at open) | P | | The space the render is written in |
| `output_cctf_encoding` | `io.output_cctf_encoding` | bool | | true | P | | Encode with the space's transfer function |
| `scan_film` | `io.scan_film` | bool | | false | P | | Skip the print: scan the negative or slide directly |
| `extended_dynamic_range` | `print_render.edr_enabled` | bool | | false | P | | Apply the paper's EDR tone map before encoding; refused if the paper has none; see section 4 on clipping |

### Further effect controls

| name | path | type | range | default | L | live | meaning |
|---|---|---|---|---|---|---|---|
| `halation_scatter_amount` | `film_render.halation.scatter_amount` | float | [0, 1] | 1 | S | | Scatter mix weight |
| `grain_amount` | `film_render.grain.amount` | float | [0, 2] | 1 | S | | `in + a (grained - in)`; 1 = the model's grain, 0 = none |
| `glare_amount` | `print_render.glare.amount` | float | [0, 30] | 1 | P | | Area multiplier: light x amount, blur x sqrt(amount) |
| `antihalation_removed` | `film_render.halation.antihalation_removed` | bool | | false | S | | Treat the film as having no anti-halation layer: back-reflection preset |
| `filter_shift_scale` | `enlarger.filter_shift_scale` | float | [1, 60] | 1 | P | yes | CC per unit of the two shifts |
| `halation_headroom_ev` | `film_render.halation.headroom_ev` | float | [0, 8] | 4 | S | | Back-reflection preset only: stops of assumed clipped-highlight headroom |
| `preview_long_edge` | `io.preview_long_edge` | int | [800, 8192] | 1600 | P | | Long edge of the `live` tier, px; does not move the meter |

---

## 4. Output format of a render

```
layout    RGBA, 4 x uint16 per pixel (8 bytes), interleaved, row-major, row 0 at the top
          element(x, y, c) := rgba16[4 (y row_stride_px + x) + c]
stride    A := max(minimum linear texture alignment of RGBA16Unorm, 8) / 8    -- queried from the device
          row_stride_px := ceil(width / A) A; padding pixels are undefined
encoding  per colour channel, in float32:
            NaN -> 0
            code := floor(clamp(x, 0, 1) * 65535 + 0.5)
          +inf -> 65535; -inf -> 0; alpha = 65535
          values above 1, which EDR can produce, clip to 65535: no headroom above SDR white
colour    the session's output_color_space (ProPhoto RGB unless the host sets another),
          encoded with that space's transfer function when output_cctf_encoding is true
          leaving the working space is the host's job, with spk_output_transform
          spk_preview_stock_lut: the table's own space and encoding
texture   MTLTexture: 2D, RGBA16Unorm, usage shader-read
          device:        linear texture over the result buffer, shared storage,
                         bytesPerRow = row_stride_px * 8; rgba16 is that buffer's contents
          iOS simulator: private texture filled by a blit, the shared buffer kept alive by
                         association; identical texels
size      the tier's post-geometry size
lifetime  each result has its own buffer
```

---

## 5. Toolchain and numerics

```
kernels     -std=metal3.1 -fmetal-math-mode=safe -fmetal-math-fp32-functions=precise
            no debug info, no embedded sources: the metallib is byte-reproducible
            stamped for the minimum OS: macOS 15.0; iOS 17.0 (air64-apple-ios17.0[-simulator])
host        C++20, -O2, -fno-fast-math, -ffp-contract=off
            hidden visibility; only spk_* exported
targets     macOS 15.0, arm64; iOS 17.0, arm64 device and simulator
arithmetic  GPU pixels: float32
            host set-up (curves, spectra, matrices, tables): float64, narrowed once to float32
            where a kernel needs more than float32: double-float (hi + lo) on error-free FMA transforms
```

**Numeric-mode probe.** Run at engine creation. The engine refuses to start otherwise.

```
for i = 0..63:
  a := float32(0.1 + 0.37 i); b := 0.5 a + 1           -- float32
  gpu[i]  := a * b - a * b, in kernel spk_probe_fma
  host[i] := fmaf(a, b, -(a * b))
all gpu == host:  safe; capabilities.backend.math_mode reports it
all gpu == 0:     fast math; refuse; rebuild with -fmetal-math-mode=safe
otherwise:        unknown contraction behaviour; refuse
```

**Host helpers** (`src/util/numeric.hpp`).

```
search_right(x, v)     bisection; lo := 0, hi := K; while lo < hi: mid := lo + (hi - lo) / 2;
                       x[mid] <= v ? lo := mid + 1 : hi := mid; return lo
                       the same bisection on a non-monotone axis; NaN v -> 0
interp(v, x, y)        clamps to the end values outside the axis; NaN v -> NaN
nanmin, nanmax, nanmean  skip NaN; all-NaN -> NaN
gaussian_kernel(s, t)  radius floor(t s + 0.5); normalised to sum 1; s <= 0 or radius 0 -> {1}
half_to_double(h)      exact for every value, subnormals and specials included
```

Summation order matters beyond the rounding of a contracted multiply-add, so the sources state it where it counts: the tier prefilter accumulates in ascending k; wavelength sums run from 380 to 780 nm.
