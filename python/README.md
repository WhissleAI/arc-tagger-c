# Python reference

| file | what |
|---|---|
| `arctag_ref.py` | the reference the C library is checked against: same features, ONNX graph and T0 decoding (numpy + onnxruntime + tokenizers) |
| `arctag_guard.py` | the kids / game guard layer, a mirror of `src/guard.c` ([docs/KIDS.md](../docs/KIDS.md)) |

```bash
pip install numpy onnxruntime==1.20.1 tokenizers
python tests/make_expected.py model [--precision int8]     # regenerate tests/expected/
```

To call the C library from Python, use the ctypes binding [`demo/arctag_ffi.py`](../demo/arctag_ffi.py) with the
shared library the build makes (`build/libarctag.so` / `.dylib` / `arctag.dll`).
