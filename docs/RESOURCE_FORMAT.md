# spektralab-engine: resource format

`spk_engine_create` takes one directory. This repository ships none of its data files. This document says what the directory must contain.

```
<resources>/
  spektralab_constants.bin      required   the constants blob
  spektralab.metallib           required   the kernel library
  profiles/<stock>.json         per stock  film and paper profiles, read on demand
  neutral_print_filters.json    optional   the neutral filter database
  print_luts.json               optional   the print-preview table index
```

One `profiles/` directory, one namespace of stock ids.

A failure in a required file gives a NULL engine. A profile failure surfaces at the call that needs the stock: `SPK_ERR_USER` from `spk_set_params`, NULL from `spk_open`, a failed step in the `spk_warm_up` reply.

---

## The metallib

```
name     spektralab.metallib, exactly
source   every kernels/*.metal of this repository
build    make                 build/spektralab.metallib            macOS 15
         make ios             build/ios/spektralab.metallib        iOS 17 device
                              build/ios-sim/spektralab.metallib    iOS 17 simulator
flags    -std=metal3.1 -fmetal-math-mode=safe -fmetal-math-fp32-functions=precise
check    at creation the numeric-mode probe runs spk_probe_fma;
         a library built with fast math is refused (docs/API_SPEC.md, section 5)
```

Copy the library that matches the platform into the resources folder. Kernels are looked up by name (`spk_<module>_<what>`), so a library from another build of different kernels fails at the first dispatch that needs a missing one.

---

## Profiles: `profiles/<stock>.json`

The stock id is the file's base name, `[A-Za-z0-9_-]{1,96}`, checked before any file is opened. A JSON object. Keys honoured at the top level:

```
info          object; required unless an alias
data          object; required unless an alias
derived_from  honoured only as a JSON string: makes the file an alias
anything else, metadata included: ignored
```

Every shape mismatch is a load error, never reshaped. JSON `null` inside a numeric array reads as NaN; NaN is legal where noted, and marks unmeasured samples.

### `info`

Every key optional. A string key is read only when its value is a string; `null` or another type gives the default.

| key | type | default | use |
|---|---|---|---|
| `stock` | str | the requested id | identity; the coupler presets read it |
| `name` | str | = stock | display name; not used by the engine |
| `type` | str | `negative` | `negative` or `positive`; anything else is a load error |
| `support` | str | `film` | the halation preset applies only to `film` |
| `stage` | str | `filming` | read; not used |
| `use` | str | `still` | halation first sigma: `still` 65 um, `cine` 50 um |
| `antihalation` | str | `weak` | halation preset: `strong`, `weak`, `no` (back-reflection model) |
| `target_print` | str | `""` | read; not used |
| `channel_model` | str | `color` | read; no behaviour depends on it |
| `densitometer` | str | `status_M` | read; not used |
| `log_sensitivity_density_over_min` | number | 0.2 | read; not used |
| `reference_illuminant` | str | `D55` | the illuminant the film's sensitivity is balanced under |
| `viewing_illuminant` | str | `D50` | the illuminant a print, or under `scan_film` the film, is viewed and scanned under |

An unknown `antihalation` or `use` value leaves the default halation strength and sigma in place.

### `data`

Spectral arrays have 81 samples, 380..780 nm in 5 nm steps. Shapes are written (rows, columns).

| key | shape | required | meaning |
|---|---|---|---|
| `wavelengths` | (81) | yes | exactly 380, 385, ..., 780 |
| `log_sensitivity` | (81, 3) | yes | log10 spectral sensitivity of the red-, green- and blue-sensitive layers. NaN allowed: reads as zero sensitivity |
| `channel_density` | (81, 3) | yes | spectral density of the cyan, magenta and yellow dyes at unit amount, columns in red, green, blue layer order. NaN allowed |
| `base_density` | (81) | yes | spectral density of the unexposed base. NaN allowed |
| `midscale_neutral_density` | (81) | no | read; not used |
| `log_exposure` | (K), K >= 2 | yes | log10 exposure axis of the characteristic curves |
| `density_curves` | (K, 3) | yes | density per layer at each log exposure |
| `density_curves_layers` | (K, 3, 3) | for grain sublayers | `[k][sublayer][channel]`; sublayers fast, medium, slow |
| `density_curves_model` | object | for a paper on the printed path | below |
| `hanatos2025_adaptation_window_params` | (0) or (4) | no | spectral window for upsampling, below. (8) loads but is refused at pipeline build |
| `hanatos2025_adaptation_surface_params` | (3, n) or `[]` | no | read; not used |
| `edr_tone_map` | object | for `extended_dynamic_range` | below |

**Computed at load**, per channel c, NaN skipped:

```
dmin_c := min_k density_curves[k][c]
N_c[k] := density_curves[k][c] - dmin_c
dmax_c := max_k N_c[k]
```

**`density_curves_model`.** The paper's curves on the printed path are evaluated from this model, not from `density_curves`.

```
{ "model_type": "cdfs",              -- default "cdfs"; any other value is refused on the printed path
  "centers":    [[...], ...],        -- channels x layers; equal shapes, at least 3 rows
  "amplitudes": [[...], ...],
  "sigmas":     [[...], ...] }

y[i][c] := sum_l amplitudes[c][l] * 1/2 erfc(-z / sqrt 2)
z       := s (log_exposure[i] - centers[c][l]) / sigmas[c][l]
s       := type == "positive" ? -1 : 1
```

**Adaptation window**, 4 values `(cuv, suv, cir, sir)` in nm:

```
om(l) := 1/2 (1 + erf((l - cuv) / (suv sqrt 2))) * 1/2 (1 - erf((l - cir) / (sir sqrt 2)))
```

**`edr_tone_map`.** Any violation is a load error.

```
{ "version": 1,                      -- must be 1
  "domain_log2_y": [lo, hi],         -- hi > lo
  "toe_y": number,                   -- > 0
  "shoulder_y": number,              -- > toe_y
  "target_log2_y": [N >= 2 numbers], -- all finite
  "calibration_film": str,           -- optional; read, not used
  "semantic": str }                  -- optional; read, not used
```

### Aliases

```
load(alias id):
  validate alias id; validate base id := derived_from
  parse profiles/<base>.json; it must not have a string derived_from: one level only
  parse the base's info and data, with every shape check
  lay the alias's info over, only for keys present with a value of the right type:
    name type support stage use antihalation target_print channel_model densitometer
    reference_illuminant viewing_illuminant log_sensitivity_density_over_min
  ignore the alias's info.stock, data and metadata
  stock id := alias id; base id kept separately
  type must still be negative or positive
```

The alias id appears in `params.film_stock` and in replies. The neutral-filter lookup tries the alias id first, then the base id.

---

## `neutral_print_filters.json`

```
{ "<print stock>": { "<enlarger illuminant>": { "<film stock>": [c, m, y] } } }
```

The leaf is three numbers: neutral cyan, magenta and yellow filtration in Kodak CC units (density x 100). Absent file: every lookup misses. A file that does not parse refuses engine creation.

```
lookup(print, illuminant, film id, base id):
  leaf := root[print][illuminant][film id]
  leaf is an array: hit if it is exactly 3 numbers, else miss
  else if an alias: root[print][illuminant][base id], same rule
  else miss
hit:  c/m/y_filter_neutral := leaf
miss: the current values stand (schema defaults 0, 65, 55, or the delta's)
```

---

## `print_luts.json` and the print-preview tables

```
{ "<print stock>": { "paired_film": str,            -- required
                     "lut_size": S,                 -- required; integer, 1 < S <= 4096
                     "declared_pairing": bool,      -- optional
                     "output_color_space": str,     -- optional; informs the host
                     "output_cctf_encoding": bool } }  -- optional
```

Absent file: the table entry points refuse and `spk_print_lut_catalog` serves `{}`. A malformed file refuses engine creation. The bytes are served verbatim by `spk_print_lut_catalog`.

The tables live in the blob, float32 only:

```
print_lut/<stock>       (S, S, S, 3)   [i][j][k][ch]: encoded output channel ch at film densities
                                       (axis0[i], axis1[j], axis2[k]); i = red-layer density,
                                       j = green-layer, k = blue-layer
print_lut_axes/<stock>  (3, S)         per-channel density axes; only the end points are read
                                       lo_c := axes[c][0], hi_c := axes[c][S-1]; hi_c > lo_c required
```

Lookup, per pixel, float32:

```
s_ch := float32(1 / (hi_ch - lo_ch))                      -- computed in float64
c_ch := clamp((D_ch - lo_ch) s_ch, 0, 1) (S - 1)
i0 := floor(c); i1 := min(i0 + 1, S - 1); f := c - i0
along k: c_ij := v[i][j][k0] (1 - f_k) + v[i][j][k1] f_k
then j, then i:
  out := (c_00 (1 - f_j) + c_01 f_j) (1 - f_i) + (c_10 (1 - f_j) + c_11 f_j) f_i
```

---

## `spektralab_constants.bin`

Little-endian throughout. Memory-mapped read-only.

**Header, 16 bytes**

| offset | type | value |
|---|---|---|
| 0 | char[4] | magic `SPKR` |
| 4 | u32 | version = 1; any other value refuses creation |
| 8 | u32 | `count`, number of entries |
| 12 | u32 | reserved, not read |

**Entry table.** `count` entries of 104 bytes, from offset 16.

| offset in entry | type | field |
|---|---|---|
| 0 | char[64] | name, UTF-8, NUL-padded, at most 63 bytes used |
| 64 | u32 | dtype: 0 float64, 1 float32, 2 float16 (IEEE 754 binary16), 3 int32, 4 uint8 |
| 68 | u32 | ndim, 1 to 4 |
| 72 | u32[4] | dims; unused trailing dims must be 0 |
| 88 | u64 | offset of the data from the start of the file; 8-byte aligned |
| 96 | u64 | nbytes; should equal prod(dims) x dtype size |

**Data.** C order: row-major, last index fastest.

```
Blob::open(file) refuses creation when:
  the file is missing or shorter than the header; wrong magic or version; the table does not fit
  a name has no NUL; dtype > 4; ndim out of 1..4; an unused dim non-zero
  an offset is not 8-aligned; an entry runs past the end of the file
  a required entry is missing, or its nbytes does not match its shape
  spectral/wavelengths is not 1-D with 81 samples
  there is no colour/cat/<name> entry
any other entry whose nbytes does not match its shape: refused when read
entries are found by name; their order is not significant
values are widened to float64 for set-up; float16 is decoded exactly
float32 print tables are served bit for bit from the mapping
```

### Entries

All spectra on the 81-sample grid. Entries are widened on read, so any numeric dtype works unless stated.

**Required at creation.**

| name | shape | meaning |
|---|---|---|
| `spectral/wavelengths` | (81) | the wavelength grid, nm |
| `colour/cmfs_1931_2deg` | (81, 3) | CIE 1931 2-degree colour-matching functions (CIE 015:2018) |
| `colour/cmfs_lms` | any | LMS cone fundamentals; present and consistent, not used |
| `colour/matrix_16` | any | the CAT16 matrix; present and consistent, not used |
| `colour/mallett_srgb_basis` | any | Mallett and Yuksel sRGB reflectance basis (EGSR 2019); present and consistent, not used |
| `hanatos/spectra_lut` | (L, L, 81), L >= 2 | Irradiance spectra over a grid of chromaticity coordinates, `[i][j][l]`; see below |
| `colour/cat/<name>` | (3, 3) | Chromatic-adaptation cone-response matrices, row-major, LMS = M XYZ; at least one |

```
spectra_lut coordinates, from chromaticity (x, y):
  t0 := clip01((1 - x)^2)
  t1 := clip01(y / max(1 - x, 1e-10))
  node (i, j) sits at (i / (L - 1), j / (L - 1))
```

**Needed when building a pipeline.** A missing entry refuses that build with `SPK_ERR_USER`.

| name | shape | needed for |
|---|---|---|
| `colour/cat/CAT16` | (3, 3) | every film build; the printed path's mid-grey probe |
| `colour/cat/CAT02` | (3, 3) | every print build; `spk_output_transform` |
| `colour/cs/<space>/primaries` | 6 values | the space, as input or output |
| `colour/cs/<space>/whitepoint` | 2 values | likewise; xy of white |
| `colour/cs/<space>/matrix_RGB_to_XYZ` | 9 values | row-major, XYZ = M rgb, white Y = 1; used as stored, never recomputed |
| `colour/cs/<space>/matrix_XYZ_to_RGB` | 9 values | its stored inverse |
| `colour/cs/sRGB/*` as above | | the printed path's mid-grey probe, whatever the session's spaces |
| `colour/sd_illuminant/<name>` | (81) | an illuminant name |
| `colour/sd_light_source/<name>` | (81) | an illuminant name |
| `filters/kg3` | (81) | transmittance of Schott KG3; `TH-KG3`, `TH-KG3-L` |
| `filters/lens_canon` | (81) | transmittance of a camera lens; `TH-KG3-L` |

**Baked colour spaces.** The names accepted for `input_color_space`, `output_color_space`, `src_cs` and `dst_cs`.

| space | transfer function | mode |
|---|---|---|
| `sRGB` | sRGB | 0 |
| `Display P3` | sRGB | 0 |
| `ProPhoto RGB` | ProPhoto | 1 |
| `Adobe RGB (1998)` | Adobe | 2 |
| `ITU-R BT.709` | BT.709 | 3 |
| `ITU-R BT.2020` | BT.709 | 3 |
| `ACEScg` | identity | 4 |
| `ACES2065-1` | identity | 4 |

**Illuminant names.** Accepted for `enlarger_illuminant` and the profile `*_illuminant` keys. Every illuminant is normalised to mean 1 over the 81 samples.

```
BB<kelvin>   a black body: s(l) = (c1 / (pi l^5)) / expm1(c2 / (l T)), l in metres,
             c1 = 3.741771e-16, c2 = 1.4388e-2
T            colour/sd_light_source/Incandescent
K75P         colour/sd_light_source/Kinoton 75P
TH-KG3       BB3400 x filters/kg3
TH-KG3-L     BB3400 x filters/kg3 x filters/lens_canon
<name>       colour/sd_illuminant/<name>, then colour/sd_light_source/<name>
```

The defaults in `info` and the schema name `D50`, `D55` and `TH-KG3`, so a resources folder with default settings needs `colour/sd_illuminant/D50`, `colour/sd_illuminant/D55` and `filters/kg3`.
