// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 Guangzhou Songzi Intelligent Technology Services Co., Ltd.
//
// A minimal test harness.
#pragma once

#include <cstdio>
#include <string>

inline int g_failures = 0;
inline int g_checks = 0;

#define CHECK(cond)                                                                    \
    do {                                                                               \
        ++g_checks;                                                                    \
        if (!(cond)) {                                                                 \
            ++g_failures;                                                              \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);       \
        }                                                                              \
    } while (0)

#define CHECK_EQ_STR(a, b)                                                                              \
    do {                                                                                                \
        ++g_checks;                                                                                     \
        const std::string _a = (a), _b = (b);                                                           \
        if (_a != _b) {                                                                                 \
            ++g_failures;                                                                               \
            std::fprintf(stderr, "FAIL %s:%d: %s\n   got: %s\n  want: %s\n", __FILE__, __LINE__, #a, \
                         _a.c_str(), _b.c_str());                                                       \
        }                                                                                               \
    } while (0)

#define CHECK_THROWS(stmt)          \
    do {                            \
        bool _thrown = false;       \
        try {                       \
            stmt;                   \
        } catch (...) {             \
            _thrown = true;         \
        }                           \
        CHECK(_thrown && #stmt);    \
    } while (0)

inline int finish(const char* name) {
    std::printf("%s: %d checks, %d failures\n", name, g_checks, g_failures);
    return g_failures ? 1 : 0;
}
