# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Guangzhou Songzi Intelligent Technology Services Co., Ltd.
"""ctypes binding of include/spektralab.h for the Python tests."""
import ctypes as C
import json

import numpy as np

SPK_OK = 0
STATUS_NAMES = {-1: "INVALID_ARG", -2: "USER", -3: "RESOURCE", -4: "UNSUPPORTED", -5: "GPU", -6: "IO",
                -7: "CANCELLED", -8: "INTERNAL"}
DEVICES = {"desktop": 0, "phone": 1, "tablet": 2}
TEXT_OUT = C.POINTER(C.c_char_p)


class Image(C.Structure):
    _fields_ = [("data", C.POINTER(C.c_float)), ("width", C.c_uint32), ("height", C.c_uint32),
                ("channels", C.c_uint32)]


class Result(C.Structure):
    _fields_ = [("rgba16", C.POINTER(C.c_uint16)), ("texture", C.c_void_p), ("width", C.c_uint32),
                ("height", C.c_uint32), ("row_stride_px", C.c_uint32), ("elapsed_ms", C.c_double),
                ("reprint", C.c_int32), ("negative_was_cached", C.c_int32), ("progress_id", C.c_char * 40)]


class SpkError(RuntimeError):
    def __init__(self, status, message):
        super().__init__(f"{STATUS_NAMES.get(status, status)}: {message}")
        self.status = status
        self.message = message


def load(path):
    """Opens the library with every signature the tests use."""
    lib = C.CDLL(str(path))
    H, TXT, ST = C.c_void_p, C.c_char_p, C.c_int32
    RES = C.POINTER(Result)
    signatures = dict(
        spk_engine_create=(H, [TXT, H]),
        spk_engine_create_for_device=(H, [TXT, H, ST]),
        spk_engine_destroy=(None, [H]),
        spk_engine_trim=(C.c_size_t, [H]),
        spk_build_info=(TXT, []),
        spk_capabilities=(TXT, [H]),
        spk_params_schema=(TXT, [H]),
        spk_memory_report=(TXT, [H]),
        spk_warm_up=(ST, [H, TXT, TXT, TEXT_OUT]),
        spk_open=(H, [H, C.POINTER(Image), TXT, TEXT_OUT]),
        spk_set_params=(ST, [H, TXT, TEXT_OUT]),
        spk_get_params=(ST, [H, TEXT_OUT]),
        spk_session_release=(None, [H]),
        spk_render=(ST, [H, TXT, RES]),
        spk_reprint=(ST, [H, TXT, RES]),
        spk_result_free=(None, [RES]),
        spk_solve=(ST, [H, TXT, TEXT_OUT]),
        spk_print_lut_catalog=(TXT, [H]),
        spk_print_lut_table=(ST, [H, TXT, C.POINTER(C.POINTER(C.c_float)), C.POINTER(C.c_uint32)]),
        spk_preview_stock_lut=(ST, [H, TXT, TXT, RES, TEXT_OUT]),
        spk_cancel=(None, [H, TXT]),
        spk_progress=(ST, [H, TXT, TEXT_OUT]),
        spk_string_free=(None, [H]),
        spk_last_error=(TXT, [H]),
    )
    for name, (restype, argtypes) in signatures.items():
        fn = getattr(lib, name)
        fn.restype, fn.argtypes = restype, argtypes
    return lib


def take_text(lib, out):
    """The JSON text of an out_json pointer, freed; None when it is empty."""
    if not out.value:
        return None
    text = C.string_at(out).decode()
    lib.spk_string_free(C.cast(out, C.c_void_p))
    return text


class Engine:
    def __init__(self, library, resources, device="desktop"):
        self.lib = load(library)
        self.h = self.lib.spk_engine_create_for_device(str(resources).encode(), None, DEVICES[device])
        if not self.h:
            raise SpkError(-1, self.error())

    def error(self):
        return (self.lib.spk_last_error(None) or b"").decode(errors="replace")

    def capabilities(self):
        return json.loads(self.lib.spk_capabilities(self.h))

    def params_schema(self):
        return json.loads(self.lib.spk_params_schema(self.h))

    def print_lut_catalog(self):
        return json.loads(self.lib.spk_print_lut_catalog(self.h))

    def print_lut_table(self, stock):
        table, size = C.POINTER(C.c_float)(), C.c_uint32()
        status = self.lib.spk_print_lut_table(self.h, stock.encode(), C.byref(table), C.byref(size))
        if status != SPK_OK:
            raise SpkError(status, self.error())
        return np.ctypeslib.as_array(table, shape=(size.value ** 3 * 3,)).copy()

    def open(self, image, delta=None):
        pixels = np.ascontiguousarray(image, dtype=np.float32)
        height, width, channels = pixels.shape
        input_image = Image(pixels.ctypes.data_as(C.POINTER(C.c_float)), width, height, channels)
        out = C.c_char_p()
        handle = self.lib.spk_open(self.h, C.byref(input_image), json.dumps(delta or {}).encode(), C.byref(out))
        if not handle:
            raise SpkError(-2, self.error())
        return Session(self, handle, json.loads(take_text(self.lib, out)))

    def close(self):
        if self.h:
            self.lib.spk_engine_destroy(self.h)
            self.h = None


class Session:
    def __init__(self, engine, handle, reply):
        self.e, self.h, self.reply = engine, handle, reply

    def _json_call(self, fn, *args):
        out = C.c_char_p()
        status = fn(self.h, *args, C.byref(out))
        if status != SPK_OK:
            raise SpkError(status, self.e.error())
        text = take_text(self.e.lib, out)
        return json.loads(text) if text else {}

    def set_params(self, delta):
        return self._json_call(self.e.lib.spk_set_params, json.dumps(delta).encode())

    def get_params(self):
        return self._json_call(self.e.lib.spk_get_params)

    def render(self, tier="full", reprint=False):
        """Returns (height x width x 4 uint16 pixels, flags)."""
        result = Result()
        fn = self.e.lib.spk_reprint if reprint else self.e.lib.spk_render
        status = fn(self.h, tier.encode(), C.byref(result))
        if status != SPK_OK:
            raise SpkError(status, self.e.error())
        rows = np.ctypeslib.as_array(result.rgba16, shape=(result.height * result.row_stride_px * 4,))
        pixels = rows.reshape(result.height, result.row_stride_px, 4)[:, :result.width].copy()
        flags = {"width": result.width, "height": result.height, "reprint": bool(result.reprint),
                 "negative_was_cached": bool(result.negative_was_cached)}
        self.e.lib.spk_result_free(C.byref(result))
        return pixels, flags

    def close(self):
        if self.h:
            self.e.lib.spk_session_release(self.h)
            self.h = None
