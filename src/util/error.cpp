// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 Guangzhou Songzi Intelligent Technology Services Co., Ltd.
#include "util/error.hpp"

namespace spk::last_error {

namespace {
thread_local std::string g_slot;
}

void set(std::string message) { g_slot = std::move(message); }
void clear() { g_slot.clear(); }
const char* get() { return g_slot.c_str(); }

}  // namespace spk::last_error
