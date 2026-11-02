// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 Guangzhou Songzi Intelligent Technology Services Co., Ltd.

/* Constants and by-value argument structs shared between the Metal kernels and the C++ host.
 *
 * compiles as MSL and as C++20: fixed-width scalars only, no pointers, no vector types, 4-byte alignment
 */
#pragma once

#ifdef __METAL_VERSION__
#define SPK_SHARED_UINT uint
#else
#include <cstdint>
#define SPK_SHARED_UINT uint32_t
#endif

#define SPK_PHILOX_M0 0xD2511F53u
#define SPK_PHILOX_M1 0xCD9E8D57u
#define SPK_PHILOX_W0 0x9E3779B9u
#define SPK_PHILOX_W1 0xBB67AE85u

// Random streams.
#define SPK_STREAM_GRAIN(c, l) ((c) + 10u * (l))
#define SPK_STREAM_MICRO(c) (100u + (c))
#define SPK_STREAM_GLARE 200u

#define SPK_MID_GREY 0.184f
