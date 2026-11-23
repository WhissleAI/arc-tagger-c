# arc-tagger-c

A small C library, CLI tools and a server for **Arc Tagger**: a text-only streaming tagger (about 10 M parameters)
that tags words from any ASR, or plain text, as they arrive.

## What you get

- **Spans**: entities, numbers (dates, times, money, scores, IDs), PII (name, phone, email, address, card, gov id,
  DOB, handle), profanity and toxicity.
- **Per-word events**: turn yield, interruption, command complete, speaker change, legal rulings, meeting decisions and
  action items, sports plays.
- **Utterance read-outs**: intent, dialogue act, safety (scam, PII leak / request, threat, crisis, child safety,
  jailbreak), affect, speaker role, topic, content state.
- **Redaction**, per-vertical configs, and a kids / game guard layer.
- **CPU only, no dependencies**: the native C backend loads in under 200 ms and streams at a few ms per word.
  ONNX Runtime is an optional second backend.

284 tags across 10 read-outs. English only. Library 0.5.0, model release v0.4.1.

## Quickstart

```bash
git clone https://github.com/WhissleAI/arc-tagger-c && cd arc-tagger-c
./scripts/download_onnxruntime.sh        # ONNX Runtime 1.20.1 (skip it and build with -DARCTAG_WITH_ORT=OFF)
./scripts/fetch_model.sh                 # public model files -> ./model (no token needed)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
ctest --test-dir build --output-on-failure

./build/arctag_cli "Hi, this is Maria Lopez. My number is 555 867 5309."
./build/arctag_cli --redact mask "Email me at jo@example.com or call 415 555 0199."
```

The native backend is the default either way; ONNX Runtime is a second backend. Windows: run
`scripts/download_onnxruntime.ps1`, then the same CMake commands (MSVC).

```c
#include "arctag.h"
char err[256];
arctag_t *h = arctag_create("model", err, sizeof err);
arctag_set_option(h, "domain", "meeting");
char *json = arctag_tag_text(h, "Priya sends the numbers Friday.");
puts(json); arctag_free_str(json);
arctag_destroy(h);
```

## Model files

From the public Hugging Face repo [`WhissleAI/arc-tagger`](https://huggingface.co/WhissleAI/arc-tagger), folder
`onnx/`. `fetch_model.sh` checks every file against `SHA256SUMS`.

| file | what |
|---|---|
| `arctag.onnx` | fp32 graph, 34 MB |
| `arctag_int8.onnx` | weight-only int8, 9.4 MB (see [docs/int8.md](docs/int8.md)) |
| `arctag_meta.json` | read-out columns, thresholds, decoding constants |
| `tokenizer.json` | byte-level BPE tokenizer |

Use the files of model release v0.4.1 with this library; `fetch_model.sh` fetches that tag by default.

## Documentation

| doc | covers |
|---|---|
| [docs/USAGE.md](docs/USAGE.md) | tools, C API, options, configs, redaction, output JSON |
| [docs/MODEL.md](docs/MODEL.md) | tags, how decoding works, limits, parity with the Python reference |
| [docs/int8.md](docs/int8.md) | int8 vs fp32: accuracy, speed, memory |
| [docs/KIDS.md](docs/KIDS.md) | kids / game domain and the guard layer |
| [configs/](configs/README.md) | ready configs per vertical |
| [docs/README.md](docs/README.md) | all docs |

## Deployment

| where | folder | what |
|---|---|---|
| C / C++ apps | [include/arctag.h](include/arctag.h), [examples/](examples/README.md) | the library and example programs |
| Python | [python/](python/README.md), [demo/arctag_ffi.py](demo/arctag_ffi.py) | reference implementation and a ctypes binding |
| Browser demo | [demo/](demo/README.md) | type or speak, watch tags appear word by word |
| Hosted service | [deploy/](deploy/README.md) | `arctag-serve`: HTTP + WebSocket, distroless image |
| Cloud platforms | [deploy/platforms/](deploy/platforms/README.md) | SageMaker / AWS Marketplace, Snowflake, Hugging Face, NVIDIA NIM: status and checklists |
| With speech | [arc_transcriber](https://github.com/WhissleAI/arc_transcriber) | audio in, transcript + tags out (combined image) |

## Licence

Code (including this inference code) and model weights: CC BY-NC 4.0 plus the Whissle terms in [LICENSE](LICENSE),
non-commercial use only. Commercial use needs a separate written licence from Whissle Inc.: ksingla@whissle.ai.
Third-party components keep their own licences: see [NOTICE](NOTICE).

## Links

- Model and model card: https://huggingface.co/WhissleAI/arc-tagger
- Arc Transcriber (speech to tagged text): https://github.com/WhissleAI/arc_transcriber
- SDKs: https://whissle.ai/sdks
