// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 Guangzhou Songzi Intelligent Technology Services Co., Ltd.
#include "session/progress.hpp"

#include <algorithm>

namespace spk::session {

void ProgressRecord::reset(std::string progress_id, uint32_t total_nodes, std::string effect_versions) {
    std::lock_guard<std::mutex> g(m_);
    id_ = std::move(progress_id);
    total_ = total_nodes;
    fired_ = 0;
    stage_ = "queued";
    done_ = false;
    auto_exposure_ev_.reset();
    node_times_.clear();
    effect_versions_ = std::move(effect_versions);
    cancel_.cancelled.store(false);
}

void ProgressRecord::node(std::string_view label) {
    std::lock_guard<std::mutex> g(m_);
    stage_ = std::string(label);
    ++fired_;
}

void ProgressRecord::finish(bool done, std::optional<double> auto_exposure_ev) {
    std::lock_guard<std::mutex> g(m_);
    done_ = done;
    auto_exposure_ev_ = auto_exposure_ev;
}

void ProgressRecord::set_memory_pressure_critical(bool v) {
    std::lock_guard<std::mutex> g(m_);
    memory_pressure_critical_ = v;
}

void ProgressRecord::cancel(const char* progress_id) {
    std::lock_guard<std::mutex> g(m_);
    if (!progress_id || id_ == progress_id) cancel_.cancelled.store(true);
}

std::string ProgressRecord::current_id() const {
    std::lock_guard<std::mutex> g(m_);
    return id_;
}

/* snapshot(id) -> {auto_exposure_ev, cancelled, done, effect_versions, memory_pressure_critical,
 *                  node_times, pct, progress_id, stage}
 *   id non-empty and not current: throw SPK_ERR_USER
 *   pct := done ? 100 : 100 * min(fired, total) / total; 0 when total is 0
 */
json::Value ProgressRecord::snapshot(const char* progress_id) const {
    std::lock_guard<std::mutex> g(m_);
    if (progress_id && *progress_id && id_ != progress_id)
        fail_user("unknown progress id '" + std::string(progress_id) + "'");
    json::Value r;
    r["auto_exposure_ev"] = auto_exposure_ev_ ? json::Value(*auto_exposure_ev_) : json::Value(nullptr);
    r["cancelled"] = cancel_.cancelled.load();
    r["done"] = done_;
    r["effect_versions"] = effect_versions_;
    r["memory_pressure_critical"] = memory_pressure_critical_;
    r["node_times"] = node_times_;
    const double frac = total_ ? std::min(fired_, total_) / double(total_) : 0.0;
    r["pct"] = done_ ? 100.0 : 100.0 * frac;
    r["progress_id"] = id_;
    r["stage"] = stage_;
    return r;
}

}  // namespace spk::session
