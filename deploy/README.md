# Arc Tagger as a hosted service (`deploy/`)

`arctag-serve` runs the Arc Tagger through this repository's C library, **native backend** (plain C with AVX2+FMA /
NEON kernels: no ONNX Runtime, no Python), behind a small HTTP + WebSocket server written in C. No network access at
run time. One binary, one container image, the same contract on every platform:

| route | what |
|---|---|
| `POST /v1/tags` | text, pre-split words (with times, confidences, speakers) or transcript segments in; spans, events and utterance read-outs out; batches |
| `WS /v1/realtime` | streaming: push words / text, get each update (`arctag_poll`) back; a few ms per word |
| `GET /health`, `GET /v1/health/live`, `GET /v1/health/ready` | liveness / readiness (NIM-style paths) |
| `GET /v1/models` | the loaded model, OpenAI list shape, plus the handle's `arctag_info` |
| `GET /ping`, `POST /invocations`, `GET /execution-parameters` | Amazon SageMaker's container contract (port 8080, `docker run IMAGE serve`) |
| `POST /v1/snowflake/tags` | Snowflake service-function rows (`{"data": [[row, text, domain?, options?], ...]}`) |

The full contract is [`openapi.yaml`](openapi.yaml); the output JSON is the library's ([docs/USAGE.md](../docs/USAGE.md)).
Platform adapters are in [`platforms/`](platforms/). The existing `examples/arctag_server.c` stays as the minimal
one-handle example; `deploy/` is the production server (handle pool, keep-alive, WebSocket, platform routes).

## Quick start

```bash
deploy/fetch_weights.sh                              # public WhissleAI/arc-tagger @ v0.4.1 onnx/ -> deploy/weights/arc-tagger
make -C deploy -j2                                   # deploy/build/arctag-serve
ARCTAG_MODEL_DIR=deploy/weights/arc-tagger deploy/build/arctag-serve
curl -s localhost:8080/v1/tags -H 'Content-Type: application/json' \
     -d '{"text": "Hi, this is Maria Lopez. My number is 555 867 5309.", "domain": "talk"}'
```

### Requests

```jsonc
{"text": "...", "domain": "talk"}                                         // whitespace words; . ? ! end an utterance
{"words": [{"w": "hello", "t0": 0.1, "t1": 0.4, "conf": 0.93, "spk": "A"}, ...]}
{"segments": [{"text": "...", "start": 0.0, "end": 1.8, "speaker": "agent", "words": [...]}]}   // e.g. Arc Transcriber verbose_json
{"inputs": ["text one", {"text": "text two", "domain": "nba"}], "domain": "talk"}             // batch; also "instances" or a bare array
```

Per-request options (top level or in `config`): `domain`, `asr`, `emit`, `tags`, `redact`, `thresholds`, `fine_auto`,
`guard`, `command_context`. They apply to that request only: the handle is put back to the server's baseline after it.
Load-time keys (`precision`, `backend`, `threads`, `max_ctx`, `stream_ctx`) are fixed when the server starts and are
refused per request (400). `text/plain` bodies work too (`?domain=`).

### Realtime (`WS /v1/realtime`)

```
ws://host:8080/v1/realtime?domain=talk
-> {"type":"words","words":["my","name","is"]}   {"type":"words","words":["John","Smith."],"endpoint":true}
-> {"type":"text","text":"call me at 415 555 0199.","endpoint":true}   {"type":"words","words":[...],"from":3}   (revise)
-> {"type":"session.update","session":{"domain":"meeting","redact":"mask"}}   {"type":"reset"}   {"type":"session.close"}
<- session.created, tags.update {n_words, a0, spans (replace yours with end > a0), events, utterances, current}, session.updated, error
```

Each session has its own handle (`ARCTAG_MAX_STREAMS`); with the native backend an update computes only the new words.

### Configuration (environment)

| variable | default | |
|---|---|---|
| `PORT`, `HOST` | `8080`, `0.0.0.0` | |
| `ARCTAG_MODEL_DIR` | first of `/opt/ml/model`, `/repository`, `/opt/arc/model`, `./model` holding `arctag_meta.json` (also in `onnx/` or `arc-tagger/`) | `arctag.onnx` and / or `arctag_int8.onnx`, `arctag_meta.json`, `tokenizer.json` |
| `ARCTAG_PRECISION` | `fp32` | `int8`: 9.4 MB graph, 26 MB per handle, about 2x faster streaming; decisions close to fp32 (docs/int8.md) |
| `ARCTAG_WORKERS` | 2 | handles for request / response calls (one thread each) |
| `ARCTAG_MAX_STREAMS` | 16 | concurrent realtime sessions |
| `ARCTAG_DOMAIN` | `auto` | default domain |
| `ARCTAG_CONFIG` | | a `configs/NAME.json` applied to every handle (a vertical: contact_center, kids_game, ...) |
| `ARCTAG_MODEL_ID` | `arc-tagger` | |
| `ARC_QUEUE_TIMEOUT_S`, `ARC_MAX_BODY_MB`, `ARC_MAX_CONNECTIONS`, `ARC_ACCESS_LOG` | 60, 100, 64, 1 | |

## Images

```bash
docker buildx build --platform linux/amd64,linux/arm64 -f deploy/Dockerfile --target runtime -t arc-tagger:0.1.0 .   # weights mounted
deploy/fetch_weights.sh && docker buildx build --platform linux/amd64,linux/arm64 -f deploy/Dockerfile --target bundled -t arc-tagger:0.1.0-v0.4.1 .
```

- Multi-stage: `debian:bookworm-slim` builds; the runtime is `gcr.io/distroless/cc-debian12:nonroot` (glibc only: no
  shell, no package manager), uid 65532, read-only weights owned by root; `HEALTHCHECK` is the binary itself
  (`serve --healthcheck`).
- amd64 and arm64 from one build. x86-64 hosts need AVX2 + FMA (the native backend checks at load and says so); every
  current cloud instance family has them.
- **Pro tier later**: the image serves whatever model folder it is given. A Pro image is the same Dockerfile with
  `--build-arg WEIGHTS=<pro folder> --build-arg MODEL_ID=arc-tagger-pro`, or the runtime image with other weights mounted;
  the code does not change. Private weights then stay in private registries / model packages.
- Combined with the transcriber (audio in, transcript + tags out): `WhissleAI/arc_transcriber` `deploy/Dockerfile.combined`
  builds this repository's library and `deploy/server/tagsrv.c` into its binary.

## GPU: there is no GPU path (and so no CUDA image)

The native backend (`src/native.c`) is plain C: AVX2+FMA or NEON kernels, Accelerate on macOS, one thread per handle.
The optional ONNX Runtime backend creates its session with the CPU provider only. There is no CUDA / GPU code in the
repository. The model is about 10 M parameters and streams at a few ms per word on one CPU core, so a GPU would not pay
for itself here; for NVIDIA NIM this means the tagger is not a NIM candidate on its own (see
[`platforms/nim/`](platforms/nim/README.md)).

## Model card (draft)

**Model**: Arc Tagger v0.4.1 (`WhissleAI/arc-tagger`, `onnx/`): a text-only streaming tagger of about 10 M parameters,
fp32 graph 34 MB or weight-only int8 9.4 MB; T0 (zero-lag) decoding. Served by arctag 0.5.0, native backend.

**Does** (English): spans for entities (person, organization, place, team, product, legal reference, ...), numbers
(date, time, money, count, score, phone / ID digits, ...), PII (name, phone, email, address, card, gov id, DOB, handle),
profanity and toxicity; 28 per-word events (turn yield, interruption, command complete, speaker change, legal
objections and rulings, meeting decisions and action items, sports plays); utterance read-outs for intent, dialogue
act, safety (scam, PII leak / request, threat, crisis, child safety, solicitation, toxicity, jailbreak), emotion,
arousal, valence, intensity, speaker role (needs speaker ids), item / topic and content state; redaction; per-vertical
configs; a kids / game guard layer.

**Does not**: other languages; custom name lists (the session-KB pointer is not in the C library); the T1 / T2
delayed tiers; automatic choice of the top-level domain (set `domain`; only the fine domain inside sports / court is
automatic); audio (pair it with Arc Transcriber or any ASR).

**Checked against the Python reference** (README "Parity"): native backend vs the Python reference on 120 cases: 563
spans with every tag, range and on/off decision identical, max score difference 4.6e-5; ONNX Runtime backend
bit-equal; streaming equals the offline pass. This service's own parity run (below): 122 cases (tests/cases.py's 16
texts x 7 domains plus examples/data), 654 reference spans, 0 differing cases, max score difference 1.0e-4.

**Speed and memory** (README "Backends", 1 thread on a shared 24-core Threadripper 2970WX at load 13-17, nice 19):
native fp32 loads in 155 ms, streams at 6.2 / 12.1 ms per word (p50 / p95), 72 MB peak RSS; native int8 68 ms, 2.9 / 7.9
ms per word, 26 MB.

**Quality** (README "Precision"): on the Whissle internal held-out test set, span F1 against the labels 0.544 (fp32) and
0.543 (int8); int8 agrees with fp32 on 98.6 % of spans and 98-99 % of events, intents and safety read-outs.

**Licence**: code and weights CC BY-NC 4.0 with the Whissle terms ([LICENSE](../LICENSE)); cJSON is MIT. Commercial use,
including a paid marketplace listing, needs Whissle's commercial licence: the listing's EULA must be Whissle's
commercial terms, not CC BY-NC.

## Conformance

```bash
deploy/conformance/run.sh deploy/weights/arc-tagger --python /path/to/python-with-numpy-onnxruntime-tokenizers
```

It builds the binary (`nice -n 19`, `make -j2`), starts it, and checks every route, each request shape (text, words,
segments, batches, text/plain), per-request options and that they do not leak into the next request, the error contract,
SageMaker `/invocations` (JSON, batch, JSON lines), Snowflake rows, realtime (words streamed one at a time give the
same spans as one offline call), concurrency, and **parity with `python/arctag_ref.py`** (numpy + onnxruntime 1.20.1 +
tokenizers) on the same texts. With a running Docker daemon it repeats everything against the image, after checking
that the image starts and is ready with `--network none`; it never starts Docker itself.
