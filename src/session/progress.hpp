// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 Guangzhou Songzi Intelligent Technology Services Co., Ltd.

/* The per-session progress record.
 *
 * guarded by its own small mutex, separate from the session lock,
 * so spk_progress and spk_cancel are safe during a render
 */
#pragma once

#include <cstdint>
#include <map>
#include <mutex>
#include <optional>
#include <string>

#include "core/render.hpp"
#include "util/json.hpp"

namespace spk::session {

class ProgressRecord final : public core::ProgressSink {
public:
    // A render-type call starts: reset, new id, stage "queued"; the cancel flag is cleared.
    void reset(std::string progress_id, uint32_t total_nodes, std::string effect_versions);
    void node(std::string_view label) override;
    void finish(bool done, std::optional<double> auto_exposure_ev);
    void set_memory_pressure_critical(bool v);

    // spk_cancel: NULL id or the current id marks the current render cancelled.
    void cancel(const char* progress_id);
    core::CancelToken& cancel_token() { return cancel_; }
    std::string current_id() const;

    // Snapshot reply. Throws spk::Error(SPK_ERR_USER,
    // "unknown progress id '<id>'") for an id that is not the current one.
    json::Value snapshot(const char* progress_id) const;

private:
    mutable std::mutex m_;
    core::CancelToken cancel_;
    std::string id_;
    std::string stage_ = "queued";
    std::string effect_versions_;
    uint32_t fired_ = 0, total_ = 0;
    bool done_ = false;
    bool memory_pressure_critical_ = false;
    std::optional<double> auto_exposure_ev_;
    json::Object node_times_;
};

}  // namespace spk::session
