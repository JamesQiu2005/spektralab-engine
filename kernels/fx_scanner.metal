// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 Guangzhou Songzi Intelligent Technology Services Co., Ltd.

/* Scanner blur and unsharp mask need no kernels of their own:
 * blur = the shared Gaussian (fx_blur.metal); unsharp combine = spk_fx_blur_lincomb, out = (1 + a) x - a G(x).
 */
#include "spk_common.h"
