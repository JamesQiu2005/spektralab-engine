// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 Guangzhou Songzi Intelligent Technology Services Co., Ltd.

/* By-value argument structs of kernels/session.metal; compiles as MSL and as C++20. */
#pragma once

#include "spk_shared.h"

// One separable prefilter pass of the tier downscale.
struct SpkSessionBlurArgs {
    SPK_SHARED_UINT w;       // plane width
    SPK_SHARED_UINT h;       // plane height
    SPK_SHARED_UINT radius;  // r >= 1 (an axis with r = 0 is skipped by the host)
    SPK_SHARED_UINT pad;
};

// The bilinear sampling (index tables precomputed on the host in
// exact integer arithmetic).
struct SpkSessionResampleArgs {
    SPK_SHARED_UINT in_w;
    SPK_SHARED_UINT in_h;
    SPK_SHARED_UINT out_w;
    SPK_SHARED_UINT out_h;
};

// The source copy of spk_open_device: channels 3 or 4.
struct SpkSessionCopyArgs {
    SPK_SHARED_UINT pixels;
    SPK_SHARED_UINT channels;
};
