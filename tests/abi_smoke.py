# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Guangzhou Songzi Intelligent Technology Services Co., Ltd.
"""ABI smoke test.

    python3 tests/abi_smoke.py <libspektralab.dylib> <res> [<res_fast>]

Checks engine creation and the math probe, build info, capabilities and params_schema, the
print-preview tables, NULL-argument statuses, the memory report invariants, device types, and (with
<res_fast>) that a fast-math metallib is refused.
"""
import ctypes
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spk_ctypes import Engine, SpkError  # noqa: E402

dylib, res_core = sys.argv[1], sys.argv[2]
res_fast = sys.argv[3] if len(sys.argv) > 3 else None

fails = []


def check(cond, what):
    print(("ok    " if cond else "FAIL  ") + what)
    if not cond:
        fails.append(what)


e = Engine(dylib, res_core)
lib = e.lib

# ---- build info and capabilities -----------------------------------------------------------------
bi = lib.spk_build_info().decode()
print("build_info:", bi)
check(re.fullmatch(r"spektralab-native (\d+)\.(\d+)\.(\d+) \(\d{4}-\d{2}-\d{2}, math=safe\)", bi) is not None,
      "build string format")
caps = e.capabilities()
print("math_mode:", caps["backend"]["math_mode"])
check(caps["backend"]["math_mode"] == "safe (probe: fma contraction present, no reassociation)", "math probe verdict")
check(caps["version"] == bi, "capabilities.version == spk_build_info()")
backend = {"spectral", "gpu", "gpu_available", "working_precision", "render_core", "host", "math_mode", "device",
           "effect_versions", "concurrent", "session_cache"}
check(set(caps) == {"version", "engine", "max_pixels", "max_mp", "max_texture_dimension_2d", "tiers",
                    "transport_version", "schema_version", "backend"}, "capabilities keys")
check(set(caps["backend"]) == backend, "capabilities.backend keys")
check(caps["tiers"] == {"full": None, "live": 1600, "preview": 3400}, "capabilities.tiers")
check(caps["schema_version"] == 1 and caps["transport_version"] == 1, "capabilities versions")
check(caps["engine"] == "spektralab.native" and caps["max_pixels"] == 151301008, "capabilities.engine, max_pixels")
check(caps["backend"]["effect_versions"] ==
      "halation=reference;grain=reference;glare=reference;print_spatial=reference;dir_couplers=reference",
      "desktop effect_versions")

# ---- params_schema -------------------------------------------------------------------------------
sch = e.params_schema()
check(sch["schema_version"] == 1 and len(sch["fields"]) == 48, "params_schema: 48 rows, schema_version 1")
names = [f["name"] for f in sch["fields"]]
check(len(set(names)) == 48 and names[0] == "film_stock" and names[-1] == "preview_long_edge", "params_schema names, order")
check(all(set(f) >= {"name", "path", "type", "layer", "default", "live"} for f in sch["fields"]), "params_schema row keys")
raw = lib.spk_params_schema(e.h)


def sorted_everywhere(pairs):
    keys = [k.encode() for k, _ in pairs]
    if keys != sorted(keys):
        raise ValueError("keys not in bytewise order: %r" % keys)
    return dict(pairs)


try:
    json.loads(raw, object_pairs_hook=sorted_everywhere)
    ordered = True
except ValueError:
    ordered = False
check(ordered and b": " not in raw and b", " not in raw, "schema JSON compact, keys bytewise-sorted at every level")

# ---- print-preview tables ------------------------------------------------------------------------
cat = e.print_lut_catalog()
check(lib.spk_print_lut_catalog(e.h) == open(Path(res_core) / "print_luts.json", "rb").read(), "catalog bytes verbatim")
for stock in cat:
    t = e.print_lut_table(stock)
    check(np.isfinite(t).all() and t.size == 33 ** 3 * 3, f"print LUT {stock}: 33^3 x 3, finite")
try:
    e.print_lut_table("ilford_multigrade_iv_rc")
    check(False, "unknown LUT stock refused")
except SpkError as err:
    check(err.status == -2 and "kodak_2383" in err.message, "unknown LUT stock: SPK_ERR_USER listing the stocks")

# ---- NULL arguments, last error -------------------------------------------------------------------
check(lib.spk_last_error(None) is not None, "spk_last_error(NULL) is never NULL")
check(lib.spk_capabilities(None) == b"" and lib.spk_params_schema(None) == b"", "NULL engine introspection -> \"\"")
check(lib.spk_print_lut_catalog(None) == b"{}", "NULL engine catalog -> {}")
check(lib.spk_engine_trim(None) == 0, "trim(NULL) == 0")
check(lib.spk_set_params(None, b"{}", None) == -1, "set_params(NULL) -> INVALID_ARG")
check(lib.spk_get_params(None, None) == -1, "get_params(NULL) -> INVALID_ARG")
check(lib.spk_render(None, b"live", None) == -1, "render(NULL) -> INVALID_ARG")
check(lib.spk_solve(None, None, None) == -1, "solve(NULL) -> INVALID_ARG")
check(lib.spk_warm_up(None, None, None, None) == -1, "warm_up(NULL) -> INVALID_ARG")
check(not lib.spk_open(e.h, None, None, None), "open(input NULL) -> NULL")
check(e.error() == "input is NULL", "  ... with a message")
lib.spk_session_release(None)
lib.spk_engine_destroy(None)
lib.spk_result_free(None)
lib.spk_string_free(None)
check(not lib.spk_engine_create(None, None), "engine_create(NULL dir) -> NULL")
check(not lib.spk_engine_create_for_device(res_core.encode(), None, 7), "unknown device type -> NULL")
check("0 (desktop), 1 (phone) or 2 (tablet)" in e.error(), "  ... message names the three device types")
check(not lib.spk_engine_create(b"/nonexistent-spk-res", None), "missing resources dir -> NULL")

# ---- memory report ----------------------------------------------------------------------------------
m = json.loads(lib.spk_memory_report(e.h))
check(m["total_bytes"] == m["pool"]["total_bytes"] + m["persistent"]["bytes"], "memory: total = pool + persistent")
check(all(m["pool"]["audit"][k] == 0 for k in ("live_underflows", "over_releases", "reclaim_while_encoding")),
      "memory: audit invariants")
lib.spk_engine_trim(e.h)
check(json.loads(lib.spk_memory_report(e.h))["pool"]["total_bytes"] == 0, "memory: pool empty after trim")

# ---- a small frame opens -----------------------------------------------------------------------------
img = np.full((8, 8, 3), 0.18, np.float32)
try:
    e.open(img, {}).close()
    check(True, "spk_open works")
except SpkError as err:
    check(False, f"spk_open failed ({err.message})")

# ---- phone / tablet engines -------------------------------------------------------------------------
for dev in ("phone", "tablet"):
    m_ = Engine(dylib, res_core, dev)
    c = m_.capabilities()
    check(c["backend"]["device"] == dev and c["backend"]["effect_versions"] ==
          "halation=mobile.2;grain=mobile.1;glare=mobile.1;print_spatial=omitted;dir_couplers=reference",
          f"{dev} engine effect_versions")
    m_.close()

# ---- fast math is refused ---------------------------------------------------------------------------
if res_fast:
    h = lib.spk_engine_create(res_fast.encode(), None)
    msg = e.error()
    check(not h, "a fast-math metallib refuses engine creation")
    print("      message:", msg)

e.close()
print(f"\nabi_smoke: {len(fails)} failure(s)")
sys.exit(1 if fails else 0)
