// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 Guangzhou Songzi Intelligent Technology Services Co., Ltd.

/* Internal error type and the thread-local last-error slot.
 *
 * every module: throw spk::Error{status, message}
 * only src/abi catches: exception -> C status + last-error slot
 *   std::bad_alloc  -> SPK_ERR_GPU
 *   std::exception  -> SPK_ERR_INTERNAL
 */
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>

#include "spektralab.h"

namespace spk {

class Error : public std::runtime_error {
public:
    Error(int32_t status, std::string message)
        : std::runtime_error(std::move(message)), status_(status) {}
    int32_t status() const noexcept { return status_; }

private:
    int32_t status_;
};

[[noreturn]] inline void fail(int32_t status, std::string message) {
    throw Error(status, std::move(message));
}

// Shorthand for the common cases.
[[noreturn]] inline void fail_user(std::string m) { fail(SPK_ERR_USER, std::move(m)); }
[[noreturn]] inline void fail_gpu(std::string m) { fail(SPK_ERR_GPU, std::move(m)); }
[[noreturn]] inline void fail_internal(std::string m) { fail(SPK_ERR_INTERNAL, std::move(m)); }
[[noreturn]] inline void fail_unsupported(std::string m) { fail(SPK_ERR_UNSUPPORTED, std::move(m)); }

// Thrown (as Error with SPK_ERR_CANCELLED) by RenderContext::node() when the cancel flag is set.
[[noreturn]] inline void fail_cancelled() { fail(SPK_ERR_CANCELLED, "render cancelled"); }

namespace last_error {
// The calling thread's slot. get() never returns NULL; "" when empty. The pointer stays valid
// until the next set()/clear() on the same thread.
void set(std::string message);
void clear();
const char* get();
}  // namespace last_error

}  // namespace spk
