// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 Guangzhou Songzi Intelligent Technology Services Co., Ltd.

/* By-value argument structs of kernels/print.metal, shared with src/print.
 *
 * compiles as MSL and as C++20: fixed-width scalars and float arrays only, 4-byte alignment
 */
#pragma once

#include "spk_shared.h"

#define SPK_PRINT_NL 81      // wavelength samples, 380..780 nm at 5 nm
#define SPK_PRINT_NL3 243    // 81 x 3, [lambda][channel]

// Kernel modes of spk_print_spectral.
#define SPK_PRINT_MODE_LOG_EXPOSURE 0u    // enlarger: log10(max(R g + o, 0) + 1e-10)
#define SPK_PRINT_MODE_SCAN 1u            // scan: max(sum T W, 0) + 1e-10

// Transfer-function modes, = res::CctfMode.
#define SPK_PRINT_CCTF_SRGB 0u
#define SPK_PRINT_CCTF_PROPHOTO 1u
#define SPK_PRINT_CCTF_ADOBE 2u
#define SPK_PRINT_CCTF_BT709 3u
#define SPK_PRINT_CCTF_IDENTITY 4u

// Enlarger and scan: one spectral integral per pixel,
//   D(l) = sum_c d_c eps[l][c] + beta[l];  T(l) = 2^(-D(l) * log2);  R_k = sum_l T(l) wt[l][k]
// in float32, lambda ascending. The tables are already neutralised (NaN wavelengths zeroed).
struct SpkPrintSpectralArgs {
    SPK_SHARED_UINT w;
    SPK_SHARED_UINT h;
    SPK_SHARED_UINT mode;
    SPK_SHARED_UINT pad0;
    float eps[SPK_PRINT_NL3];
    float beta[SPK_PRINT_NL];
    float wt[SPK_PRINT_NL3];
    float g[3];  // enlarger: normalisation gain; unused by the scan
    float o[3];  // enlarger: pre-flash offset; unused by the scan
};

// gain: l' = log10(max(2^(l log2 10) k, 0) + 1e-10).
struct SpkPrintGainArgs {
    SPK_SHARED_UINT w;
    SPK_SHARED_UINT h;
    float k;
};

// paper: the paper curves; the table buffer holds x[K], r[K], y[K][3] (float32).
struct SpkPrintCurveArgs {
    SPK_SHARED_UINT w;
    SPK_SHARED_UINT h;
    SPK_SHARED_UINT K;
};

// levels: Y' = clamp(m Y + q, 0, 1); XYZ *= Y' / (Y + 1e-10).
struct SpkPrintLevelsArgs {
    SPK_SHARED_UINT w;
    SPK_SHARED_UINT h;
    float m;
    float q;
};

// XYZ to RGB: RGB = M XYZ, M row-major.
struct SpkPrintMatrixArgs {
    SPK_SHARED_UINT w;
    SPK_SHARED_UINT h;
    float m[9];
};

// CAM16-UCS gamut compression. The scalars are the float32-narrowed numbers of the
// spk_output_transform `k` array; the kernel computes c z, 1/(c z), F_L^(1/4), 100/F_L and
// (1.64 - 0.29^n)^0.73 from them in float32.
struct SpkPrintCam16Args {
    SPK_SHARED_UINT w;
    SPK_SHARED_UINT h;
    SPK_SHARED_UINT rows;  // 64 (lightness)
    SPK_SHARED_UINT cols;  // 720 (hue)
    float m_to_xyz[9];
    float m_to_rgb[9];
    float m16[9];      // the literal forward matrix
    float m16_inv[9];  // the published 8-digit inverse (per-pixel variant)
    float d_rgb[3];    // D_R, D_G, D_B
    float f_l;         // F_L
    float n_bb;
    float n_cb;
    float n;           // Y_b / Y_w
    float z;
    float a_w;
    float c;
    float n_c;
    float j_first;     // 1
    float j_last;      // 110
    float h_first;     // -pi
    float h_step;      // 2 pi / 720
    float knee[3];     // (threshold, limit, power) of the chroma knee
    float lc[3];       // lightness knee
    SPK_SHARED_UINT lc_active;
    float j_white;     // J' of the adopted white
};

// EDR: EDR tone map; the table buffer holds target_log2_y[N].
struct SpkPrintEdrArgs {
    SPK_SHARED_UINT w;
    SPK_SHARED_UINT h;
    SPK_SHARED_UINT n;
    float r_y[3];
    float domain_lo;
    float domain_span;  // domain_log2_y[1] - domain_log2_y[0]
    float toe_y;
    float shoulder_y;
};

// encode: optional near-identity matrix + transfer function, then 16-bit RGBA into
// an output-layout buffer (row pitch = stride pixels).
struct SpkPrintEncodeArgs {
    SPK_SHARED_UINT w;
    SPK_SHARED_UINT h;
    SPK_SHARED_UINT stride;
    SPK_SHARED_UINT cctf_active;  // 0: values pass to the 16-bit encode unchanged
    SPK_SHARED_UINT cctf_mode;
    float m_nid[9];
};

// spk_preview_stock_lut: trilinear lookup into an S^3 x 3 table, then 16-bit encode.
struct SpkPrintLutArgs {
    SPK_SHARED_UINT w;
    SPK_SHARED_UINT h;
    SPK_SHARED_UINT stride;
    SPK_SHARED_UINT size;  // S
    float lo[3];
    float s[3];  // float32(1 / (hi - lo)), computed in float64
};
