# int8 vs fp32

The model repo ships two graphs of the same model: `arctag.onnx` (fp32, 34.0 MB) and `arctag_int8.onnx` (int8, 9.4 MB).

`arctag_int8.onnx` is **weight-only int8**:

- every 2-D weight table of at least 4,096 values that feeds a MatMul / Gather is stored as int8, with one scale per column
  (`scale = max |w| / 127` over the column), behind a standard `DequantizeLinear` node;
- 1-D weights, small tables and all activations stay float, so compute is float;
- it is made from `arctag.onnx` with `python tools/quantize_model.py model/arctag.onnx model/arctag_int8.onnx`
  (deterministic: rebuilding gives the same bytes);
- it uses the same thresholds as fp32 (`arctag_meta.json` has no int8-specific thresholds).

| | native backend | ONNX Runtime |
|---|---|---|
| how the int8 tables are used | kept int8 in memory, multiplied with the column scale in the kernels | folded to float tables once, at load |
| memory per handle | 26 MB RSS (fp32: 72 MB) | 118 MB, same as fp32 |
| streaming vs offline | identical | identical |

Load it with any of:

```c
arctag_options o; arctag_options_init(&o); o.precision = ARCTAG_INT8;     /* C API */
arctag_create_ex("model", &o, err, sizeof err);
arctag_set_option(h, "precision", "int8");                               /* or later: reloads the model */
```

- `"precision": "int8"` in a config;
- `--int8` in the CLI tools and the server;
- `ARCTAG_PRECISION=int8` for `arctag_create`;
- `o.model_file` / `--model-file F` for any other `.onnx` file in the model directory.

**Recommendation.** Use int8 with the native backend (the default) for edge devices and many handles per process: it
loads in 68 ms, streams at 2.9 ms per word and needs 26 MB per handle. Use fp32 when you need the reference output.

## ONNX Runtime session settings

The library opens ONNX Runtime sessions with all graph optimisations on and the quantised-graph (QDQ) passes off
(`session.disable_quant_qdq = 1`). With the QDQ passes on, ONNX Runtime would fuse each `DequantizeLinear` + `MatMul`
pair into int8 kernels that also quantise the activations on every call. Scores would then move slightly and depend on
the streaming window. With them off, the tables fold into float once and compute stays in float.

`python/arctag_ref.py` opens its session with the same settings, so the C library under ONNX Runtime matches it bit for
bit on both graphs (`parity_fp32`, `parity_int8`). If you run `arctag_int8.onnx` with ONNX Runtime yourself, use the
same two settings, or at least `GraphOptimizationLevel.ORT_ENABLE_BASIC`. Basic level also folds the tables to float, but
its scores differ from the library's by up to 1e-4, because the library's extended fusions are then missing.

## Agreement with fp32

On the made-up test texts (`int8_agreement`, the 120 parity cases): all items F1 0.987, max |dp| of shared items 0.103,
the same on both backends (pass at F1 >= 0.90 and |dp| <= 0.25).

On the Whissle internal held-out test set (168,781 words, 1,187 windows, each run with its ASR source), at the model's
thresholds:

**Raw outputs.** Max |dp| per output over all words and columns, and on/off flips at the profile thresholds:

| output | max abs dp | flips |
|---|---|---|
| frame (content state) | 0.150 | 9,709 |
| span_beg / span_in / span_end | 0.129 / 0.125 / 0.194 | 339 / 436 / 554 |
| ev_at / ev_imm / ev_rec | 0.075 / 0.051 / 0.056 | 179 / 11 / 286 |
| utt (intent / act / safety / affect) | 0.112 | 11,005 |
| speaker | 0.081 | 851 |
| item | 0.044 | 293 |
| span output (entity / number / PII / profanity / toxicity, type / boundary) | 0.04-0.14 | |
| verifier (ver_m / ver_e, logits) | 0.21 / 0.26 | |

**Decoded output.** F1 between the fp32 and int8 items:

| family | F1 |
|---|---|
| entity spans | 0.985 |
| number spans | 0.988 |
| PII spans | 0.978 |
| profanity spans | 1.000 |
| toxicity spans | 0.918 |
| all spans | 0.986 |
| events | 0.984 |
| intent / act / safety read-outs | 0.992 / 0.996 / 0.980 |
| emotion / arousal / valence / intensity | 0.997 / 0.998 / 0.997 / 0.995 |
| item / content state | 0.998 / 0.988 |

Toxicity spans move the most. The test set has no toxicity labels, so that is disagreement with fp32, not a measured
loss.

## Accuracy on the Whissle internal held-out test set

Scored against the test set's reference labels (1,286 reference spans, exact tag and range; 283 items with intent
labels), full decode at the model's thresholds:

| | fp32 | int8 | delta |
|---|---|---|---|
| span F1, all | 0.544 | 0.543 | -0.1 |
| names / mentions | 0.509 | 0.507 | -0.2 |
| numbers | 0.622 | 0.622 | 0.0 |
| PII | 0.442 | 0.437 | -0.5 (129 reference spans: within noise) |
| intent accuracy | 0.601 | 0.601 | 0.0 |

## Speed and memory

`bench` at 1 thread:

- `examples/data/*.txt`, 600 streamed words, each pushed on its own and polled;
- `stream_ctx` 256;
- batch = `tag_text` per line, 2 passes;
- AMD Threadripper 2970WX, cores shared with other jobs (load average 13-17), nice 19; best of two runs.

Absolute numbers on a quiet machine will be lower; compare the ratios. Streaming is p50 / p95 ms per word.

| backend, model file | file | load | streaming, 1 thread | batch ms/word, 1 thread | peak RSS |
|---|---|---|---|---|---|
| ONNX Runtime, fp32 | 34.0 MB | 935 ms | 185 / 279 | 0.58 | 118 MB |
| ONNX Runtime, int8 | 9.4 MB | 712 ms | 191 / 277 | 0.77 | 118 MB |
| native, fp32 | 34.0 MB | 155 ms | 6.2 / 12.1 | 0.96 | 72 MB |
| **native, int8** | **9.4 MB** | **68 ms** | **2.9 / 7.9** | 0.66 | **26 MB** |

Notes:

- **The native backend uses one thread.**
- **Under ONNX Runtime, streaming cost is set by `stream_ctx`.** Every poll re-runs the last `stream_ctx` words, so a
  per-word poll costs about `stream_ctx` x the batch per-word cost; with `stream_ctx` 96 (`edge_low_latency.json`) it is
  roughly a third of the 256 figure. The native backend's poll costs about one word.
- **Under ONNX Runtime the int8 file runs at fp32 speed and memory**, because its tables become float at load. Its only
  gain there is the smaller download.

## Older dynamic int8 files

Model releases up to v0.4.0 shipped a *dynamic* int8 `arctag_int8.onnx` (ONNX Runtime `quantize_dynamic`, 11.5 MB:
per-channel uint8 weights, activations quantised per call). The library still loads such a file: the native backend
converts it to float tables at load (`weights: "int8-folded"` in `arctag_info`, 52 MB, fp32 speed), and ONNX Runtime
runs its int8 kernels (77 MB). It agrees less well with fp32 (held-out span F1 0.963, toxicity spans 0.824), and under
ONNX Runtime its streaming scores depend slightly on the window. Fetch v0.4.1 or later (`fetch_model.sh` does by
default) for the weight-only file.
