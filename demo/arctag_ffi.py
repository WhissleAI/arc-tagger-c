"""ctypes binding of the arctag C library (include/arctag.h), as used by the web demo.

Only the standard library is needed. The shared library is built by CMake next to the static one (target arctag_shared:
build/libarctag.so, build/libarctag.dylib or build/[Release/]arctag.dll). ctypes releases the GIL during every C call, so
handles can run in worker threads while the asyncio loop keeps serving. A handle is not thread-safe: use it from one thread
at a time.
"""
import ctypes
import json
import os
import sys
from pathlib import Path

FP32, INT8 = 0, 1


class _Options(ctypes.Structure):
    _fields_ = [("size", ctypes.c_size_t), ("precision", ctypes.c_int), ("threads", ctypes.c_int), ("model_file", ctypes.c_char_p)]


def lib_candidates(root):
    """where CMake puts the shared library, newest build layouts first"""
    root = Path(root)
    names = {"win32": ["arctag.dll"], "darwin": ["libarctag.dylib"]}.get(sys.platform, ["libarctag.so"])
    out = []
    for d in ("build", "build/Release", "build/RelWithDebInfo", "build/Debug"):
        out += [root / d / n for n in names]
    return out


def find_lib(root, explicit=None):
    explicit = explicit or os.environ.get("ARCTAG_LIB")
    if explicit:
        return Path(explicit)
    for p in lib_candidates(root):
        if p.is_file():
            return p
    raise FileNotFoundError("libarctag shared library not found (looked in: " + ", ".join(str(p) for p in lib_candidates(root)) +
                            "). Build it with: cmake -S . -B build && cmake --build build -j")


class Lib:
    def __init__(self, path):
        self.path = str(path)
        if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
            os.add_dll_directory(str(Path(path).parent))           # onnxruntime.dll is copied next to the build outputs
        L = self.L = ctypes.CDLL(self.path)
        vp, cp, sz = ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t
        L.arctag_options_init.argtypes = [ctypes.POINTER(_Options)]; L.arctag_options_init.restype = None
        L.arctag_create_ex.argtypes = [cp, ctypes.POINTER(_Options), ctypes.c_char_p, sz]; L.arctag_create_ex.restype = vp
        L.arctag_apply_config.argtypes = [vp, cp]; L.arctag_apply_config.restype = ctypes.c_int
        L.arctag_set_option.argtypes = [vp, cp, cp]; L.arctag_set_option.restype = ctypes.c_int
        for f in ("arctag_tag_text", "arctag_tag_words"):
            getattr(L, f).argtypes = [vp, cp]; getattr(L, f).restype = vp
        L.arctag_push_words.argtypes = [vp, cp]; L.arctag_push_words.restype = ctypes.c_int
        L.arctag_poll.argtypes = [vp]; L.arctag_poll.restype = vp
        L.arctag_info.argtypes = [vp]; L.arctag_info.restype = vp
        L.arctag_reset.argtypes = [vp]; L.arctag_reset.restype = None
        L.arctag_last_error.argtypes = [vp]; L.arctag_last_error.restype = cp
        L.arctag_free_str.argtypes = [vp]; L.arctag_free_str.restype = None
        L.arctag_destroy.argtypes = [vp]; L.arctag_destroy.restype = None
        L.arctag_version.argtypes = []; L.arctag_version.restype = cp

    def version(self):
        return self.L.arctag_version().decode()

    def take(self, p):
        """malloc'd JSON string from the library -> Python object (and free it)"""
        if not p:
            return None
        try:
            return json.loads(ctypes.string_at(p).decode("utf-8", "replace"))
        finally:
            self.L.arctag_free_str(p)


class ArcError(RuntimeError):
    pass


class Handle:
    """one arctag handle (one ONNX Runtime session, one word stream)"""

    def __init__(self, lib, model_dir, precision="fp32", threads=0):
        self.lib, self.L = lib, lib.L
        o = _Options()
        self.L.arctag_options_init(ctypes.byref(o))
        o.precision = INT8 if precision == "int8" else FP32
        o.threads = int(threads or 0)
        err = ctypes.create_string_buffer(512)
        h = self.L.arctag_create_ex(str(model_dir).encode(), ctypes.byref(o), err, len(err))
        if not h:
            raise ArcError(err.value.decode("utf-8", "replace") or "arctag_create_ex failed")
        self.h = h
        self.precision = "int8" if precision == "int8" else "fp32"

    def _err(self, what):
        e = self.L.arctag_last_error(self.h)
        return ArcError(f"{what}: {e.decode('utf-8', 'replace') if e else 'error'}")

    def apply_config(self, cfg):
        if self.L.arctag_apply_config(self.h, json.dumps(cfg).encode()) != 0:
            raise self._err("config")

    def set_option(self, key, value):
        if self.L.arctag_set_option(self.h, key.encode(), str(value).encode()) != 0:
            raise self._err(f"option {key}")

    def tag_text(self, text):
        r = self.lib.take(self.L.arctag_tag_text(self.h, text.encode()))
        if r is None:
            raise self._err("tag_text")
        return r

    def push(self, obj):
        if self.L.arctag_push_words(self.h, json.dumps(obj).encode()) != 0:
            raise self._err("push_words")

    def poll(self):
        r = self.lib.take(self.L.arctag_poll(self.h))
        if r is None:
            raise self._err("poll")
        return r

    def info(self):
        return self.lib.take(self.L.arctag_info(self.h))

    def reset(self):
        self.L.arctag_reset(self.h)

    def close(self):
        if self.h:
            self.L.arctag_destroy(self.h); self.h = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
