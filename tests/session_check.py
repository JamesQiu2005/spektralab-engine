# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Guangzhou Songzi Intelligent Technology Services Co., Ltd.
"""Session behaviour through the C ABI.

    python3 tests/session_check.py <libspektralab.dylib> <res> [--fast]

Checks the reply shapes of open / set_params / get_params / solve / progress / warm-up /
preview_stock_lut, validation and atomic refusal, invalidation and reprint flags, tier sizes, seeds
and determinism, the meter rules, non-finite input, the memory report invariants, and cancellation
from a second thread.
"""
import ctypes
import json
import sys
import threading
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spk_ctypes import Engine, Image, Result, SpkError  # noqa: E402

dylib, res_core = sys.argv[1], sys.argv[2]
fast = "--fast" in sys.argv

fails = []


def check(cond, what):
    print(("ok    " if cond else "FAIL  ") + what)
    if not cond:
        fails.append(what)


e = Engine(dylib, res_core)
lib = e.lib
schema_names = {f["name"] for f in e.params_schema()["fields"]}

DET = {"grain_active": False, "glare_active": False}


def jout(fn, *args):
    out = ctypes.c_char_p()
    st = fn(*args, ctypes.byref(out))
    text = ctypes.string_at(out).decode() if out.value else None
    if out.value:
        lib.spk_string_free(ctypes.cast(out, ctypes.c_void_p))
    return st, (json.loads(text) if text else None)


def progress(s, pid=None):
    st, r = jout(lib.spk_progress, s.h, pid.encode() if pid else None)
    return st, r


def memory():
    return json.loads(lib.spk_memory_report(e.h).decode())


rng = np.random.default_rng(3)
img = (rng.random((224, 768, 3), dtype=np.float32) * 0.5).astype(np.float32)

# ---- open reply ---------------------------------------------------------
s = e.open(img, {})
r = s.reply
check(sorted(r) == ["capabilities", "detected_input", "meta", "params", "session_id"], "open reply: exactly 5 keys")
check(sorted(r["detected_input"]) == ["input_cctf_decoding", "input_color_space", "input_color_space_source"]
      and r["detected_input"]["input_color_space_source"] == "caller", "detected_input keys")
check(r["meta"] == {"height": 224, "megapixels": 0.17, "source": "<in-process>", "width": 768}, "meta (megapixels 0.17)")
check(len(r["params"]) == 48, "params has 48 keys")
check(set(r["params"]) == schema_names, "params keys = the schema's names")
check(r["session_id"].startswith("s") and r["session_id"][1:].isdigit(), "session_id s<N>")
check(sorted(r["capabilities"]) == sorted(e.capabilities()), "capabilities key set")
check(r["params"]["m_filter_neutral"] != 65 and r["params"]["output_color_space"] == "ProPhoto RGB",
      "digest at open (portra 400 / portra endura row), forced working space")
gp = s.get_params()
check(gp == r["params"], "get_params == open params")
s.close()

# Explicit c/m/y in the open delta: overwritten by a database hit, kept on a miss.
s = e.open(img, {"c_filter_neutral": 10, "m_filter_neutral": 60, "y_filter_neutral": 40})
check(s.reply["params"]["m_filter_neutral"] != 60, "open delta pack overwritten on a database hit")
s.close()
s = e.open(img, {"c_filter_neutral": 10, "m_filter_neutral": 60, "y_filter_neutral": 40, "enlarger_illuminant": "D65"})
gp = s.reply["params"]
check(gp["c_filter_neutral"] == 10 and gp["m_filter_neutral"] == 60, "open delta pack kept on a database miss")
s.close()

# Refusals at open (NULL session).
for delta, what in [({"no_such_field": True}, "unknown field"), ({"product_defaults": 1}, "product_defaults not bool"),
                    ({"film_stock": "no_such_film"}, "unknown stock"), ({"output_color_space": "Nope"}, "unbuildable")]:
    try:
        e.open(img, delta)
        check(False, f"open refuses {what}")
    except SpkError as err:
        check(bool(err.message), f"open refuses {what}: {err.message}")
out = ctypes.c_char_p()
im = np.ascontiguousarray(img)
h = lib.spk_open(e.h, ctypes.byref(Image(im.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 768, 224, 3)),
                 b'{"print_exposure":NaN}', ctypes.byref(out))
check(not h and "parse" in e.error(), "strict JSON at open")

# ---- set_params -----------------------------------------------------------
s = e.open(img, {**DET, "auto_exposure": False})
before = s.get_params()
for delta in [{"no_such_field": 1}, {"print_exposure": 99.0}, {"grain_active": "yes"}, {"product_defaults": True},
              {"film_stock": "no_such_film"}, {"output_color_space": "Not A Space"}, {"print_exposure": None},
              {"another_unknown_field": "x"}, {"geometry_quarter_turns": 3.6}]:
    try:
        s.set_params(delta)
        check(False, f"refused: {delta}")
    except SpkError as err:
        named = any(k in err.message for k in delta)
        check(err.status == -2 and s.get_params() == before, f"refused {delta}: -2, unchanged ({err.message})")
rep = s.set_params({})
check(sorted(rep) == ["invalidated", "params"] and rep["invalidated"] == "print" and rep["params"] == before,
      "empty delta: print, nothing changes")
check(s.set_params({"exposure_compensation_ev": 1})["invalidated"] == "shoot", "shoot field -> shoot")
check(s.set_params({"print_exposure": 1.2, "exposure_compensation_ev": 0})["invalidated"] == "shoot", "mixed -> shoot")
check(s.set_params({"geometry_quarter_turns": 2.5})["params"]["geometry_quarter_turns"] == 3, "int rounding 2.5 -> 3")
s.set_params({"geometry_quarter_turns": 0})
rep = s.set_params({"print_stock": "kodak_2383", "m_filter_neutral": 1})
check(rep["params"]["m_filter_neutral"] != 1 and rep["invalidated"] == "print", "print stock change re-digests")
rep = s.set_params({"m_filter_neutral": 7})
check(rep["params"]["m_filter_neutral"] == 7, "explicit pack without a stock change is kept")
rep = s.set_params({"enlarger_illuminant": "D65"})
check(rep["params"]["m_filter_neutral"] == 7, "illuminant change alone does not re-digest")
s.close()

# ---- render / reprint / invalidation -----------------------------------------
big = (rng.random((2731, 4096, 3), dtype=np.float32) * 0.4).astype(np.float32)
s = e.open(big, {**DET, "auto_exposure": False})
sizes = []
for t in ("live", "preview", "full"):
    px, m = s.render(t)
    sizes.append((m["width"], m["height"], m["reprint"]))
check(sizes == [(1600, 1066, False), (3400, 2266, False), (4096, 2731, False)], f"tier sizes {sizes}")
check(np.all(px[..., 3] == 65535), "alpha opaque")
_, m = s.render("live", reprint=True)
check(m["reprint"] and m["negative_was_cached"], "reprint reuses the negative")
s.set_params({"print_exposure": 1.3})
_, m = s.render("live", reprint=True)
check(m["reprint"], "print-layer change keeps the negative")
s.set_params({"scan_film": True})
_, m = s.render("live", reprint=True)
check(m["reprint"], "print-layer rebuild keeps the negative")
s.set_params({"scan_film": False, "lens_blur_um": 1})
_, ml = s.render("live", reprint=True)
_, mp = s.render("preview", reprint=True)
check(not ml["reprint"] and not mp["reprint"], "shoot change drops every negative")
_, ml = s.render("live", reprint=True)
s.set_params({"preview_long_edge": 1200})
_, ml = s.render("live", reprint=True)
_, mp = s.render("preview", reprint=True)
check(not ml["reprint"] and mp["reprint"] and ml["width"] == 1200, "preview_long_edge drops only live")
# memory report while negatives are cached
mr = memory()
check(mr["total_bytes"] == mr["pool"]["total_bytes"] + mr["persistent"]["bytes"], "total = pool + persistent")
pr = mr["persistent"]
check(pr["source_bytes"] + pr["cached_negative_bytes"] <= pr["bytes"] and pr["negatives"] in (0, 1, 2, 3)
      and len(pr["sessions"]) == 1 and pr["sessions"][0]["negatives"] == 2, f"persistent breakdown {pr['negatives']}")
check(sorted(mr["pressure"]) == ["critical_events", "critical_pending", "idle_trim_seconds", "idle_trims", "monitor",
                                 "warn_events"] and mr["pressure"]["idle_trim_seconds"] == 60, "pressure keys")
s.close()
lib.spk_engine_trim(e.h)
mr = memory()
check(mr["pool"]["total_bytes"] == 0 and mr["persistent"]["source_bytes"] == 0, "release + trim: pool empty")
check(mr["pool"]["audit"]["live_underflows"] == 0 and mr["pool"]["audit"]["over_releases"] == 0, "audit counters 0")

# No upscale; preview_long_edge >= frame -> live == full.
s = e.open(img, {**DET, "auto_exposure": False})
a, ma = s.render("live")
b, mb = s.render("full")
check(ma["width"] == 768 and np.array_equal(a, b), "no upscale: live == full")
_, mq = s.render("full", reprint=True)
c, _ = s.render("full")
check(np.array_equal(b, c), "grain/glare off: render twice identical")
s.close()

# ---- seeds ------------------------------------------------------------------------------
def seq(eng, delta, ops):
    ss = eng.open(img, delta)
    outs = []
    for op, arg in ops:
        if op == "set":
            ss.set_params(arg)
        else:
            outs.append(ss.render(arg, reprint=(op == "reprint"))[0])
    ss.close()
    return outs


ops = [("render", "live"), ("reprint", "live"), ("set", {"scanner_lens_blur": 0.0}), ("reprint", "live"), ("render", "live")]
on = {"auto_exposure": False}
A = seq(e, on, ops)
e2 = Engine(dylib, res_core)
B = seq(e2, on, ops)
check(all(np.array_equal(x, y) for x, y in zip(A, B)), "two engines identical at every step (grain+glare on)")
check(not np.array_equal(A[0], A[1]), "reprint re-draws the glare")
check(not np.array_equal(A[0], A[3]), "a second render differs")
G = seq(e, on, [("render", "live"), ("set", {"scanner_lens_blur": 0.0}), ("render", "live")])
check(np.array_equal(G[0], G[1]) and np.array_equal(G[0], A[0]), "a rebuild resets the sequence")
# Live write == rebuild (grain and glare off).
L1 = seq(e, {**DET, "auto_exposure": False}, [("render", "full"), ("set", {"print_exposure": 1.4}), ("reprint", "full")])
L2 = seq(e, {**DET, "auto_exposure": False},
         [("render", "full"), ("set", {"print_exposure": 1.4, "scan_film": False}), ("reprint", "full")])
check(np.array_equal(L1[1], L2[1]), "live write == rebuild")
# Atomic refusal keeps the random sequence.
R1 = seq(e, on, [("render", "live"), ("render", "live")])
ss = e.open(img, on)
x0 = ss.render("live")[0]
try:
    ss.set_params({"output_color_space": "Not A Space", "print_exposure": 2})
except SpkError:
    pass
x1 = ss.render("live")[0]
ss.close()
check(np.array_equal(R1[1], x1), "refused rebuild leaves the sequence untouched")

# ---- meter and progress -------------------------------------------------------------
s = e.open(big[:1700, :2600], {**DET})
evs = []
for t in ("live", "preview", "full"):
    _, m = s.render(t)
    st, pr = progress(s)
    evs.append(pr["auto_exposure_ev"])
check(st == 0 and sorted(pr) == ["auto_exposure_ev", "cancelled", "done", "effect_versions", "memory_pressure_critical",
                                 "node_times", "pct", "progress_id", "stage"], f"progress keys {sorted(pr)}")
check(pr["done"] and pr["pct"] == 100 and not pr["cancelled"] and pr["node_times"] == {}, "progress after a render")
check(evs[0] is not None and evs[0] == evs[1] == evs[2], "meter EV equal across tiers")
st, _ = progress(s, "p999999")
check(st == -2, "unknown progress id -> USER")
s.set_params({"film_stock": "kodak_ektar_100", "exposure_compensation_ev": 1})
s.render("live")
check(progress(s)[1]["auto_exposure_ev"] == evs[0], "film / EV comp do not re-meter")
s.set_params({"geometry_crop_x": 0.2, "geometry_crop_w": 0.5})
s.render("live")
check(progress(s)[1]["auto_exposure_ev"] != evs[0], "a crop re-meters")
s.set_params({"auto_exposure": False})
s.render("live")
check(progress(s)[1]["auto_exposure_ev"] is None, "auto_exposure off -> null")
st, sol = jout(lib.spk_solve, s.h, b"both")
check(st == 0 and sorted(sol) == ["exposure_ev_by_method", "solved_params"]
      and sorted(sol["exposure_ev_by_method"]) == ["balanced", "center", "protect_highlights", "protect_shadows"]
      and sorted(sol["solved_params"]) == ["c_filter_neutral", "exposure_compensation_ev", "m_filter_neutral",
                                           "y_filter_neutral"], "solve(both) reply")
st, sol = jout(lib.spk_solve, s.h, b"filter_pack")
check(st == 0 and list(sol) == ["solved_params"], "solve(filter_pack) reply")
st, _ = jout(lib.spk_solve, s.h, b"nope")
check(st == -2, "unknown solve target -> USER")
s.close()

# ---- non-finite input  -------------------------------------------------------------------------------
bad = img.copy()
bad[0, 0, 0], bad[5, 7, 1], bad[100, 300, 2] = np.nan, np.inf, -np.inf
zero = bad.copy()
zero[~np.isfinite(zero)] = 0
a = seq(e, {**DET, "auto_exposure": False}, [("render", "full")])[0]
s1 = e.open(bad, {**DET, "auto_exposure": False}); x = s1.render("full")[0]; s1.close()
s2 = e.open(zero, {**DET, "auto_exposure": False}); y = s2.render("full")[0]; s2.close()
check(np.array_equal(x, y), "non-finite input renders as zeros")
rgba = np.concatenate([img, np.full(img.shape[:2] + (1,), 7.0, np.float32)], axis=2)
s1 = e.open(rgba, {**DET, "auto_exposure": False}); x = s1.render("full")[0]; s1.close()
check(np.array_equal(x, a), "4th channel ignored")

# ---- warm-up and the preview table -------------------------------------------------------------------------
st, w = jout(lib.spk_warm_up, e.h, None, None)
check(st == 0 and sorted(w) == ["already_warm", "film_stock", "print_stock", "render_core", "steps", "total_ms"]
      and w["film_stock"] == "kodak_portra_400" and not w["already_warm"], f"warm-up reply {w and w['steps']}")
s = e.open(img, {**DET, "auto_exposure": False, "film_stock": "kodak_ektar_100"})
r = Result()
st, rep = jout(lib.spk_preview_stock_lut, s.h, None, None, ctypes.byref(r))
check(st == 0 and r.reprint == 1 and r.negative_was_cached == 1, "preview_stock_lut reports reprint = 1")
check(rep and sorted(rep) == ["apply_backend", "apply_ms", "declared_pairing", "lut_source", "paired_film",
                              "print_stock", "tier", "warning"] and rep["tier"] == "live", f"preview_stock_lut reply {rep}")
lib.spk_result_free(ctypes.byref(r))
st, rep = jout(lib.spk_preview_stock_lut, s.h, b"no_such_paper", b"live", ctypes.byref(r))
check(st == -2 and not r.texture, "preview_stock_lut without a table -> USER, out zeroed")
s.close()

# ---- cancellation from a second thread ----------------------------------------------------------------------
if not fast:
    s = e.open(big, {"auto_exposure": False})
    res_box = {}

    def worker():
        rr = Result()
        res_box["st"] = lib.spk_render(s.h, b"full", ctypes.byref(rr))
        res_box["tex"] = rr.texture
        if rr.texture:
            lib.spk_result_free(ctypes.byref(rr))

    th = threading.Thread(target=worker)
    th.start()
    import time
    while th.is_alive():
        lib.spk_cancel(s.h, None)
        time.sleep(0.0005)
    th.join()
    st, pr = progress(s)
    if res_box["st"] == -7:
        check(not res_box["tex"] and pr["cancelled"] and not pr["done"], "cancelled: out zeroed, progress flags")
        _, m = s.render("full", reprint=True)
        check(not m["reprint"], "after a cancel the reprint re-develops")
    else:
        print("note  the render finished before a cancel landed:", res_box["st"])
    s.close()

print(f"\n{len(fails)} failure(s)")
sys.exit(1 if fails else 0)
