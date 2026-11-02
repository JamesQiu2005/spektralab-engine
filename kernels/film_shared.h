// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 Guangzhou Songzi Intelligent Technology Services Co., Ltd.

/* By-value argument structs of kernels/film.metal, shared with src/film.
 *
 * compiles as MSL and as C++20: fixed-width scalars and scalar arrays only, 4-byte alignment
 */
#pragma once

#include "spk_shared.h"

// Input transfer decode modes; the values equal res::CctfMode.
#define SPK_FILM_DECODE_SRGB 0u
#define SPK_FILM_DECODE_PROPHOTO 1u
#define SPK_FILM_DECODE_ADOBE 2u
#define SPK_FILM_DECODE_BT709 3u
#define SPK_FILM_DECODE_IDENTITY 4u

// spk_film_decode: out[i] = decode(in[i]) for i < n (elements, not pixels).
struct SpkFilmDecodeArgs {
    SPK_SHARED_UINT n;
    SPK_SHARED_UINT mode;
    float bt709_bp;   // f32(1.099 * 0.018^0.45 - 0.099), binary64 on the host
    float bt709_exp;  // f32
};

// spk_film_geometry. The binary64 coordinate of output pixel (ox, oy) is
//   f_x = c_x + (col[ox].x + row[oy].x),   f_y = c_y + (col[ox].y + row[oy].y)
// where the host has split each binary64 term into a float pair (hi, lo). The per-pixel sums are
// double-float. col holds 4 floats per output column (x_hi, x_lo, y_hi, y_lo), row 4 per row.
struct SpkFilmGeometryArgs {
    SPK_SHARED_UINT src_w;
    SPK_SHARED_UINT src_h;
    SPK_SHARED_UINT dst_w;
    SPK_SHARED_UINT dst_h;
    float cx_hi;
    float cx_lo;
    float cy_hi;
    float cy_lo;
};

// spk_film_scale: p[i] *= g for i < n (elements).
struct SpkFilmScaleArgs {
    SPK_SHARED_UINT n;
    float g;
};

// spk_film_upsample: RGB plane -> film-layer exposure H.
// a: the input matrix A (f32, row-major); table: L x L x 3 float, [i][j][c].
struct SpkFilmUpsampleArgs {
    SPK_SHARED_UINT n;  // pixels
    SPK_SHARED_UINT L;
    float a[9];
};

// spk_film_boost, per element.
struct SpkFilmBoostArgs {
    SPK_SHARED_UINT n;  // elements
    float x0;           // f32(x0)
    float r;            // f32(1 / M)
    float a;            // f32(a)
    float k;            // f32(k * M)
};

// spk_film_log (kappa = 1): p[i] = log10(max(p[i], 0) + 1e-10).
struct SpkFilmLogArgs {
    SPK_SHARED_UINT n;  // elements
};

// spk_film_curves: per pixel and channel, F(v; x, inv, y). The curve buffer
// is [x: K][inv: K][y: 3K][y0: 3K] floats (y channel-major, y[c*K + k]); `table` selects y or
// the pre-inverted y0.
struct SpkFilmCurvesArgs {
    SPK_SHARED_UINT n;  // pixels
    SPK_SHARED_UINT K;
    SPK_SHARED_UINT table;
};

// spk_film_coupler_source: first-development density -> chi, in place.
struct SpkFilmCouplerArgs {
    SPK_SHARED_UINT n;  // pixels
    SPK_SHARED_UINT positive;
    float m[9];     // M (row = donor, column = receiver), f32
    float dmax[3];  // f32(Dmax_c)
};

// spk_film_coupler_apply: L (in place) <- F(L - chi~; x, inv, y0).
struct SpkFilmCouplerApplyArgs {
    SPK_SHARED_UINT n;  // pixels
    SPK_SHARED_UINT K;
};
