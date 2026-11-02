// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 Guangzhou Songzi Intelligent Technology Services Co., Ltd.

/* The render-time context shared by session, film, print and fx. Header-only.
 *
 * RenderContext = GPU + device variant + pixel pitch + tier ratio + cancel flag + progress sink
 */
#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>

#include "gpu/gpu.hpp"
#include "util/error.hpp"

namespace spk::core {

enum class Device : uint8_t { Desktop = 0, Phone = 1, Tablet = 2 };

/* Effect implementations selected by the device type.
 *   Desktop          all false
 *   Phone, Tablet    all true
 */
struct EffectVariants {
    bool mobile_halation = false;        // halation=mobile.2
    bool mobile_grain = false;           // grain=mobile.1 (byte-identical to the reference)
    bool mobile_glare = false;           // glare=mobile.1 (byte-identical to the reference)
    bool print_spatial_omitted = false;  // scanner blur and unsharp mask absent

    static EffectVariants for_device(Device d) {
        EffectVariants v;
        if (d != Device::Desktop) v = {true, true, true, true};
        return v;
    }
    // The exact effect_versions string.
    std::string_view effect_versions() const {
        return print_spatial_omitted
                   ? "halation=mobile.2;grain=mobile.1;glare=mobile.1;print_spatial=omitted;dir_couplers=reference"
                   : "halation=reference;grain=reference;glare=reference;print_spatial=reference;dir_couplers=reference";
    }
};

// Set by spk_cancel from any thread; polled by RenderContext::node().
struct CancelToken {
    std::atomic<bool> cancelled{false};
};

// Receives one call per pipeline node. Implemented by the session; must be cheap and thread-safe
// against concurrent spk_progress readers.
class ProgressSink {
public:
    virtual ~ProgressSink() = default;
    // label: dotted, e.g. "filming.halation"; the phase prefix is one of preprocess, filming,
    // printing, scanning.
    virtual void node(std::string_view label) = 0;
};

struct RenderContext {
    gpu::Context& gpu;
    Device device = Device::Desktop;
    EffectVariants variants;
    CancelToken* cancel = nullptr;     // null: not cancellable (warm-up)
    ProgressSink* progress = nullptr;  // null: no progress reporting

    // Pixel pitch in micrometres and the ratio of this tier's long edge to the full tier's, fixed
    // before the film side runs.
    double pitch_um = 0.0;
    double rho_tier = 1.0;

    /* node(label): called between stages
     *   if the cancel flag is set: throw spk::Error(SPK_ERR_CANCELLED)
     *   report label to the progress sink
     */
    void node(std::string_view label) const {
        if (cancel && cancel->cancelled.load(std::memory_order_relaxed)) fail_cancelled();
        if (progress) progress->node(label);
    }
    gpu::Plane scratch(uint32_t w, uint32_t h) const { return gpu.pool().plane(w, h); }
    gpu::Kernel kernel(std::string_view name) const { return gpu.kernel(name); }
};

}  // namespace spk::core
