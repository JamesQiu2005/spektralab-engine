// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 Guangzhou Songzi Intelligent Technology Services Co., Ltd.

/* By-value argument structs of the fx kernels, shared by MSL and C++20.
 *
 * only 4-byte scalar members and fixed arrays of them
 */
#pragma once

#include "spk_shared.h"

// FIR: sigma < 3 gives R = floor(3 sigma + 0.5) <= 9, so at most 19 taps per channel.
#define SPK_FX_FIR_MAX_R 9u
#define SPK_FX_FIR_STRIDE 19u

// Accumulation modes of a blur's last pass:
//   NONE: out = v;  ZERO: out = 0 + w v (the first mixture component);  PLANE: out = acc + w v.
#define SPK_FX_ACC_NONE 0u
#define SPK_FX_ACC_ZERO 1u
#define SPK_FX_ACC_PLANE 2u

struct SpkFxFirArgs {
    SPK_SHARED_UINT w;
    SPK_SHARED_UINT h;
    SPK_SHARED_UINT r;         // common radius R (taps -R..R at taps[c*19 + i + R])
    SPK_SHARED_UINT acc_mode;  // horizontal pass only
    float weight[3];
    float taps[3 * 19];
};

struct SpkFxIirArgs {
    SPK_SHARED_UINT w;
    SPK_SHARED_UINT h;
    SPK_SHARED_UINT mask;      // bit c set: channel c is filtered; others are copied in -> out
    SPK_SHARED_UINT acc_mode;  // vertical pass only
    float weight[3];
    // (B, B1, B2, B3) per channel as double-float pairs: value = hi + lo exactly.
    float hi[3 * 4];
    float lo[3 * 4];
};

// Pointwise: lincomb out = a x + b y; affine out = x s + t (a<->s, b<->t); mul out = x y.
struct SpkFxPointArgs {
    SPK_SHARED_UINT n;         // pixels
    SPK_SHARED_UINT y_zero;    // lincomb: 1 -> y reads as 0 (an accumulator that starts at 0)
    float a[3];
    float b[3];
};

// Highlight headroom.
struct SpkFxHeadroomArgs {
    SPK_SHARED_UINT n;
    float clip;
    float headroom_ev;
};

// Mobile halation tent splat / bilinear up + mix.
struct SpkFxTentArgs {
    SPK_SHARED_UINT w;    // full-resolution size
    SPK_SHARED_UINT h;
    SPK_SHARED_UINT cw;   // coarse size
    SPK_SHARED_UINT ch;
    SPK_SHARED_UINT f;    // factor F
    SPK_SHARED_UINT has_fine;
    float a[3];           // mix scalars (up pass)
    float b[3];
};

// Grain draws. Mode values below.
#define SPK_FX_GRAIN_SUM 0u      // sub-layer model, no dye clouds: G = v0 + v1 + v2 (then - D_min if shift)
#define SPK_FX_GRAIN_PLANES 1u   // sub-layer model with dye clouds: value_l into plane l
#define SPK_FX_GRAIN_SIMPLE 2u   // simple model
struct SpkFxGrainArgs {
    SPK_SHARED_UINT w;
    SPK_SHARED_UINT h;
    SPK_SHARED_UINT seed;
    SPK_SHARED_UINT k;         // curve points K
    SPK_SHARED_UINT mode;
    SPK_SHARED_UINT positive;  // sub-layer model: negate the input density
    SPK_SHARED_UINT shift;     // 1: subtract D_min,c (affine x*1 + t) in the same kernel
    SPK_SHARED_UINT n_sub;     // simple model sub-layers
    float inv_n_sub;           // simple model: 1/N_sub
    float dmin_l[9];           // [l*3 + c], f32(dmin[l][c])
    float sat_d[9];            // D[l][c]
    float count[9];            // n[l][c]
    float uniformity[3];
    float dmin_c[3];           // D_min,c (simple model add, final shift t = -D_min,c)
};

// Lognormal field: mean m, std s, stream = stream_base + c (per channel) or
// one stream for all channels (same_all = 1).
struct SpkFxLognormalArgs {
    SPK_SHARED_UINT w;
    SPK_SHARED_UINT h;
    SPK_SHARED_UINT seed;
    SPK_SHARED_UINT stream_base;
    SPK_SHARED_UINT same_all;
    float mean;
    float stddev;
};

// Glare add: xyz_c += (F / 100) * I_c.
struct SpkFxGlareAddArgs {
    SPK_SHARED_UINT n;
    float tint[3];
};
