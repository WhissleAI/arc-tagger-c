# int8 vs fp32

Two int8 forms of `arctag.onnx` work with the library, under both backends:

| | weight-only int8 | dynamic int8 |
|---|---|---|
| made with | `python tools/quantize_model.py model/arctag.onnx model/arctag_int8.onnx` | ONNX Runtime `quantize_dynamic` (the `arctag_int8.onnx` the model repo ships today) |
| file | 9.4 MB | 11.5 MB |
| what is int8 | the weight tables (one scale per column); compute is float | the weights, and under ONNX Runtime the activations, quantised per call |
| streaming vs offline | identical, both backends | identical on the native backend; differs slightly under ONNX Runtime |
| agreement with fp32 (`int8_agreement`, span F1 / max \|dp\|) | 0.987 / 0.103, both backends | native 0.981 / 0.123; ONNX Runtime 0.975 / 0.172 |
| agreement with fp32, held-out test set (span F1, ONNX Runtime) | 0.986 | 0.963 |
| native backend | kept int8 in memory: 26 MB RSS, streaming about 2x faster than fp32 | converted to float tables at load: 52 MB RSS, fp32 speed |
| ONNX Runtime | folded to float tables at load (fp32 speed and memory) | int8 kernels: 77 MB RSS |

**Recommendation: weight-only int8.** It is the smallest file, it streams exactly like the offline pass, it agrees best
with fp32 (also on the held-out test set, see below), and with the native backend (the default) it is the fastest and
lightest configuration: 68 ms load, 2.9 ms per streamed word, 26 MB per handle. Keep the dynamic file
only if you run ONNX Runtime and need its lower memory there.

Both are loaded as "int8" (`arctag_int8.onnx` in the model directory), or by name with `model_file`. You choose at load
time, and switching later reloads the model:

```c
arctag_options o; arctag_options_init(&o); o.precision = ARCTAG_INT8;     /* C API */
arctag_create_ex("model", &o, err, sizeof err);
arctag_set_option(h, "precision", "int8");                               /* or later: reloads the model */
o.model_file = "arctag_w8.onnx";                                         /* or any other file in the model directory */
```

You can also choose it with:

- `"precision": "int8"` in a config;
- `--int8` in the CLI tools and the server (`bench --model-file F` for any file);
- `ARCTAG_PRECISION=int8` for `arctag_create`.

The rest of this page, up to [Speed and memory](#speed-and-memory), describes the dynamic int8 file under ONNX
Runtime, with the weight-only file alongside where it was measured.

**Summary (dynamic int8, ONNX Runtime).** int8 is 3x smaller on disk, uses about 40 MB less RAM per handle
and loads faster. It is about as fast as fp32, because most of the compute is activation-by-activation products, which
stay fp32. Decoded output agrees with fp32 on about 96% of spans and 96-99% of utterance read-outs. Accuracy on the
Whissle internal held-out test set is within 1 point. Toxicity spans are the family that moves most.

## Quantisation settings

`arctag_int8.onnx` is made with ONNX Runtime dynamic quantisation (`quantize_dynamic`), applied to every MatMul with a
constant 2-D weight and to the token embedding table:

- per-channel scales;
- uint8 weights;
- activations quantised per call (`DynamicQuantizeLinear` + `MatMulInteger`).

These parts stay fp32:

- the read-out layers (the final layers that produce each output);
- all 1-D weights and the small per-domain tables;
- every activation-by-activation product.

The graph uses only standard ONNX Runtime CPU ops. It was made with onnxruntime 1.31.1's quantiser and runs on the 1.20.1
runtime the library links.

## Choosing the settings

All candidates were measured on the Whissle internal held-out test set (168,781 words, 1,187 windows), at the model's
thresholds, against fp32. Each window is run with its ASR source (human-written text, web text or an ASR). Agreement is
F1 between the fp32 and int8 decoded items (exact tag and word range, or exact tag for read-outs that are on).

| settings | size | span agreement | toxicity spans | PII spans | events | intent | safety read-out |
|---|---|---|---|---|---|---|---|
| per-tensor int8 | 9.7 MB | 0.954 | 0.803 | 0.923 | 0.950 | 0.971 | 0.930 |
| per-channel int8 | 9.8 MB | 0.919 | 0.546 | 0.850 | 0.905 | 0.954 | 0.874 |
| per-channel uint8 | 9.8 MB | 0.960 | 0.816 | 0.931 | 0.955 | 0.977 | 0.951 |
| **per-channel uint8, read-out layers fp32** (shipped) | **11.5 MB** | **0.963** | **0.824** | **0.940** | **0.958** | **0.981** | **0.960** |
| weight-only int8 (`tools/quantize_model.py`), for comparison | 9.4 MB | 0.986 | 0.918 | 0.978 | 0.984 | 0.992 | 0.980 |

What the table shows:

- **Per-channel int8 is the worst setting**, and uint8 fixes it. With signed int8 weights, the u8 x s8 products of ONNX
  Runtime's AVX2 kernels saturate, which pushes the verifier logits off by about 4. Per-channel uint8 avoids this and
  has the smallest errors of the small dynamic graphs.
- **The read-out layers are the sensitive part.** Keeping them fp32 costs 1.7 MB and raises agreement on every family.
  The biggest gains are on safety read-outs (+0.9 pt), PII (+0.8 pt), toxicity (+0.9 pt) and intent (+0.5 pt).
- **The weight-only file agrees best of all**, because its activations stay float.

## Agreement of the shipped int8 (Whissle internal held-out test set)

**Raw outputs.** The table shows the max |dp| per output over all words and columns, and the on/off agreement at the
profile thresholds (every word x valid column).

| output | max abs dp | on/off agreement | flips |
|---|---|---|---|
| frame (content state) | 0.244 | 99.58% | 15,645 |
| span_beg | 0.414 | 99.987% | 768 |
| span_in | 0.389 | 99.981% | 1,332 |
| span_end | 0.637 | 99.981% | 1,294 |
| ev_at | 0.275 | 99.872% | 676 |
| ev_imm | 0.132 | 99.988% | 30 |
| ev_rec | 0.143 | 99.940% | 568 |
| utt (intent / act / safety / affect) | 0.358 | 99.863% | 21,216 |
| speaker | 0.252 | 99.872% | 1,923 |
| item | 0.155 | 99.830% | 1,070 |
| span output (entity / number / PII / profanity, type / boundary) | 0.08-0.42 | | |
| verifier (ver_m / ver_e, logits) | 1.08 / 0.75 | | |

The max |dp| is a worst case at a few words. The mean |dp| is at most 0.002 on every probability output, and about 0.07 on
the two verifier logits.

**Decoded output.** The table shows F1 between the fp32 and int8 items.

| family | fp32 items | dynamic int8 F1 | weight-only int8 F1 |
|---|---|---|---|
| entity spans | 8,383 | 0.955 | 0.985 |
| number spans | 12,774 | 0.973 | 0.988 |
| PII spans | 1,697 | 0.940 | 0.978 |
| profanity spans | 196 | 0.992 | 1.000 |
| toxicity spans | 267 | **0.824** | 0.918 |
| all spans | 23,317 | 0.963 | 0.986 |
| events | 15,598 | 0.958 | 0.984 |
| intent / act / safety read-outs | | 0.981 / 0.992 / 0.960 | 0.992 / 0.996 / 0.980 |
| emotion / arousal / valence / intensity | | 0.997 / 0.995 / 0.992 / 0.992 | 0.997 / 0.998 / 0.997 / 0.995 |
| item / content state | | 0.990 / 0.978 | 0.998 / 0.988 |

The ctest `int8_agreement` checks the same on the made-up test texts: all items have F1 0.981 on the native backend and
0.975 under ONNX Runtime (pass at 0.90), and the max |dp| of shared items is 0.123 / 0.172 (pass at 0.25).

## Accuracy on the Whissle internal held-out test set

Scored against the test set's reference labels:

- 1,286 reference spans, scored as exact tag and range;
- 283 items with intent labels;
- the full decode at the model's thresholds, plus raw runs at 0.5.

| | fp32 | dynamic int8 | delta | weight-only int8 | delta |
|---|---|---|---|---|---|
| span F1, all (decode) | 0.544 | 0.542 | -0.3 | 0.543 | -0.1 |
| names / mentions | 0.509 | 0.508 | -0.1 | 0.507 | -0.2 |
| numbers | 0.622 | 0.615 | **-0.7** | 0.622 | 0.0 |
| PII | 0.442 | 0.439 | -0.3 | 0.437 | -0.5 |
| span F1, all (raw runs at 0.5) | 0.543 | 0.546 | +0.3 | 0.542 | 0.0 |
| intent accuracy | 0.601 | 0.615 | +1.4 | 0.601 | 0.0 |

With dynamic int8, numbers lose the most (-0.7 pt) and intent gains 1.4 pt (4 of 283 items). Against fp32, toxicity
spans move the most, but the test set has no toxicity labels, so that change is disagreement rather than a measured
loss. PII is within noise (129 reference spans). If toxicity matters most to you, run fp32 or the weight-only file for
that stream.

## Speed and memory

`bench` with each backend and model file at 1 thread:

- `examples/data/*.txt`, 600 streamed words, each pushed on its own and polled;
- `stream_ctx` 256;
- batch = `tag_text` per line, 2 passes;
- AMD Threadripper 2970WX, cores shared with other jobs (load average 13-17), nice 19; best of two runs.

Absolute numbers on a quiet machine will be lower. The ratios are what to compare. Streaming is p50 / p95 ms per word.

| backend, model file | file | load | streaming, 1 thread | batch ms/word, 1 thread | peak RSS |
|---|---|---|---|---|---|
| ONNX Runtime, fp32 | 34.0 MB | 935 ms | 185 / 279 | 0.58 | 118 MB |
| ONNX Runtime, dynamic int8 | 11.5 MB | 563 ms | 155 / 177 | 0.45 | 77 MB |
| ONNX Runtime, weight-only int8 | 9.4 MB | 712 ms | 191 / 277 | 0.77 | 118 MB |
| native, fp32 | 34.0 MB | 155 ms | 6.2 / 12.1 | 0.96 | 72 MB |
| native, dynamic int8 | 11.5 MB | 110 ms | 6.4 / 13.4 | 1.14 | 52 MB |
| **native, weight-only int8** | **9.4 MB** | **68 ms** | **2.9 / 7.9** | 0.66 | **26 MB** |

Notes:

- **The native backend uses one thread.**
- **Under ONNX Runtime, streaming cost is set by `stream_ctx`.** Every poll re-runs the last `stream_ctx` words, so a
  per-word poll costs about `stream_ctx` x the batch per-word cost; with `stream_ctx` 96 (`edge_low_latency.json`) it is
  roughly a third of the 256 figure. The native backend's poll costs about one word.
- **Under ONNX Runtime the weight-only file runs at fp32 speed and memory**: its tables become float once, at load.

## Things to know

- **Dynamic int8 streaming scores depend on the window under ONNX Runtime.** Dynamic quantisation computes activation
  scales per call, so the same words can score slightly differently in a 256-word streaming window than in the
  full-text pass. fp32, the weight-only file and the native backend do not have this effect. `parity_int8` therefore
  checks the ONNX Runtime offline output bit-exactly against the int8 Python reference and only counts the streaming
  differences; `native_int8_stream` requires native streaming to equal the native offline pass exactly.
- **Thresholds are shared.** All model files use the same thresholds in `arctag_meta.json`; there are no int8-specific
  thresholds.
