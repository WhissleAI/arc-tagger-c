# int8 vs fp32

`arctag_int8.onnx` is a dynamically quantised copy of `arctag.onnx`. Both graphs use the same decoder, thresholds and
JSON output. You choose one at load time, and switching later reloads the session:

```c
arctag_options o; arctag_options_init(&o); o.precision = ARCTAG_INT8;     /* C API */
arctag_create_ex("model", &o, err, sizeof err);
arctag_set_option(h, "precision", "int8");                               /* or later: reloads the session */
```

You can also choose it with:

- `"precision": "int8"` in a config;
- `--int8` in the CLI tools and the server;
- `ARCTAG_PRECISION=int8` for `arctag_create`.

**Summary.** int8 is 3x smaller on disk, uses about 35 MB less RAM per handle and loads about 20% faster. It is about
as fast as fp32, because most of the compute is activation-by-activation products, which stay fp32. Decoded output
agrees with fp32 on about 96% of spans and 97-99% of utterance read-outs. Accuracy on the Whissle internal held-out test
set is within 1 point. Toxicity spans are the family that moves most.

Use int8 when memory or download size matters (edge devices, many handles per process). Use fp32 when you need the
reference output.

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

All candidates were measured on the Whissle internal held-out test set (168,781 words), at the v0.3.0 thresholds,
against fp32. Agreement is F1 between the fp32 and int8 decoded items (exact tag and word range, or exact tag for
read-outs that are on).

| settings | size | span agreement | toxicity spans | PII spans | events | intent | safety read-out |
|---|---|---|---|---|---|---|---|
| per-tensor int8 | 9.7 MB | 0.944 | 0.813 | 0.928 | 0.954 | 0.971 | 0.954 |
| per-channel int8 | 9.8 MB | 0.908 | 0.600 | 0.853 | 0.924 | 0.954 | 0.924 |
| per-channel uint8 | 9.8 MB | 0.960 | 0.850 | 0.934 | 0.949 | 0.977 | 0.963 |
| **per-channel uint8, read-out layers fp32** (shipped) | **11.5 MB** | **0.962** | **0.864** | **0.935** | **0.960** | **0.981** | **0.970** |

What the table shows:

- **Per-channel int8 is the worst setting**, and uint8 fixes it. With signed int8 weights, the u8 x s8 products of ONNX
  Runtime's AVX2 kernels saturate, which pushes the verifier logits off by more than 3. Per-channel uint8 avoids this
  and has the smallest errors of the small graphs.
- **The read-out layers are the sensitive part.** Keeping them fp32 costs 1.7 MB and raises agreement on every family.
  The biggest gains are on events (+1.1 pt), safety read-outs (+0.6 pt) and toxicity (+1.4 pt).
- Keeping more of the graph fp32 added several MB without a measurable gain.

## Agreement of the shipped int8 (Whissle internal held-out test set, v0.3.0 thresholds)

**Raw outputs.** The table shows the max |dp| per output over all words and columns, and the on/off agreement at the
profile thresholds (every word x valid column).

| output | max abs dp | on/off agreement | flips |
|---|---|---|---|
| frame (content state) | 0.202 | 99.73% | 10,216 |
| span_beg | 0.461 | 99.989% | 708 |
| span_in | 0.576 | 99.980% | 1,376 |
| span_end | 0.624 | 99.982% | 1,266 |
| ev_at | 0.484 | 99.790% | 1,108 |
| ev_imm | 0.132 | 99.974% | 65 |
| ev_rec | 0.179 | 99.975% | 239 |
| utt (intent / act / safety / affect) | 0.277 | 99.856% | 22,276 |
| speaker | 0.171 | 99.901% | 1,487 |
| item | 0.170 | 99.784% | 1,366 |
| span output (5 families, type / boundary) | 0.13-0.38 | | |
| verifier (ver_m / ver_e, logits) | 0.87 / 0.81 | | |

The max |dp| is a worst case at a few words. The mean |dp| is at most 0.002 on every probability output, and about 0.07 on
the two verifier logits.

**Decoded output.** The table shows F1 between the fp32 and int8 items.

| family | fp32 items | F1 |
|---|---|---|
| entity spans | 9,869 | 0.959 |
| number spans | 12,177 | 0.972 |
| PII spans | 2,638 | 0.935 |
| profanity spans | 201 | 0.987 |
| toxicity spans | 146 | **0.864** |
| all spans | 25,031 | 0.962 |
| events | 18,213 | 0.960 |
| intent / act / safety read-outs | | 0.981 / 0.993 / 0.970 |
| emotion / arousal / valence / intensity | | 0.997 / 0.994 / 0.991 / 0.987 |
| item / content state | | 0.989 / 0.984 |

The ctest `int8_agreement` checks the same on the made-up test texts: all items have F1 0.980 (pass at 0.90), and the
max |dp| of shared items is 0.139 (pass at 0.25).

## Accuracy on the Whissle internal held-out test set

Scored against the test set's reference labels:

- 1,286 reference spans, scored as exact tag and range;
- 283 items with intent labels;
- the full decode at the v0.3.0 thresholds, plus raw runs at 0.5.

| | fp32 | int8 | delta |
|---|---|---|---|
| span F1, all (decode) | 0.468 | 0.465 | -0.3 |
| names / mentions | 0.452 | 0.451 | -0.1 |
| numbers | 0.525 | 0.515 | **-1.0** |
| PII | 0.362 | 0.370 | +0.8 |
| span F1, all (raw runs at 0.5) | 0.473 | 0.467 | -0.5 |
| intent accuracy | 0.569 | 0.562 | -0.7 |

Numbers lose the most (-1.0 pt). Against fp32, toxicity spans move the most, but the test set has no toxicity labels,
so that change is disagreement rather than a measured loss. PII is within noise (129 reference spans). If toxicity
matters most to you, run fp32 for that stream.

## Speed and memory

The table compares fp32 and int8 at 1 and 4 threads with `bench`:

- `examples/data/*.txt` (636 words), 1,000 streamed words, each pushed on its own and polled;
- `stream_ctx` 256;
- batch = `tag_text` per line, 3 passes;
- AMD Threadripper 2970WX, shared with other jobs (load average about 18 of 48 threads), nice 19, best of two runs.

Absolute numbers on a quiet machine will be lower. The ratios are what to compare.

| | fp32, 1 thread | int8, 1 thread | fp32, 4 threads | int8, 4 threads |
|---|---|---|---|---|
| model file | 34.0 MB | 11.5 MB | 34.0 MB | 11.5 MB |
| load (create) | 557 ms | 446 ms | 583 ms | 488 ms |
| streaming ms per word, p50 / p95 | 138 / 168 | 137 / 175 | 116 / 151 | 109 / 127 |
| batch ms per word | 0.45 | 0.41 | 0.49 | 0.42 |
| peak RSS | 110 MB | 76 MB | 114 MB | 76 MB |

With `stream_ctx` 96 (the `edge_low_latency.json` setting, 1 thread):

| | fp32 | int8 |
|---|---|---|
| streaming ms per word, p50 / p95 | 40 / 54 | 38 / 53 |
| peak RSS | 97 MB | 63 MB |

Notes:

- **Streaming cost is set by `stream_ctx`.** Every poll re-runs the last `stream_ctx` words, so a per-word poll costs
  about `stream_ctx` x the batch per-word cost. Poll per ASR update rather than per word, or lower `stream_ctx`.
- **Extra threads help little.** The model is small and its runtime is dominated by many small ops.

## Things to know

- **int8 streaming scores depend on the window.** Dynamic quantisation computes activation scales per call, so the
  same words can score slightly differently in a 256-word streaming window than in the full-text pass. fp32 does not
  have this effect: streaming and offline agree within 1e-4. `parity_int8` therefore checks the offline output
  bit-exactly against the int8 Python reference and only counts the streaming differences.
- **Thresholds are shared.** Both graphs use the same thresholds in `arctag_meta.json`; there are no int8-specific
  thresholds.
