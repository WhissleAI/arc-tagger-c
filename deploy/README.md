# Arc Tagger as a hosted service (`deploy/`)

`arctag-serve` runs Arc Tagger through this repo's library (**native backend**: plain C, no ONNX Runtime, no Python)
behind a small HTTP + WebSocket server in C. No network access at run time. One binary, one image, the same contract
on every platform.

| route | what |
|---|---|
| `POST /v1/tags` | text, words (with times, confidences, speakers) or transcript segments in; spans, events, read-outs out; batches |
| `WS /v1/realtime` | push words or text, get each update back; a few ms per word |
| `GET /health`, `/v1/health/live`, `/v1/health/ready` | liveness and readiness |
| `GET /v1/models` | the loaded model (OpenAI list shape) plus `arctag_info` |
| `GET /ping`, `POST /invocations`, `GET /execution-parameters` | Amazon SageMaker's container contract |
| `POST /v1/snowflake/tags` | Snowflake service-function rows |

Contract: [`openapi.yaml`](openapi.yaml). Output JSON: [docs/USAGE.md](../docs/USAGE.md#output-json). Platform guides
and marketplace status: [`platforms/`](platforms/README.md). `examples/arctag_server.c` stays as the minimal
one-handle example; this is the production server.

## Quick start

```bash
deploy/fetch_weights.sh                              # WhissleAI/arc-tagger @ v0.4.1 -> deploy/weights/arc-tagger
make -C deploy -j2                                   # deploy/build/arctag-serve
ARCTAG_MODEL_DIR=deploy/weights/arc-tagger deploy/build/arctag-serve
curl -s localhost:8080/v1/tags -H 'Content-Type: application/json' \
     -d '{"text": "Hi, this is Maria Lopez. My number is 555 867 5309.", "domain": "talk"}'
```

## Requests

```jsonc
{"text": "...", "domain": "talk"}                                          // . ? ! end an utterance
{"words": [{"w": "hello", "t0": 0.1, "t1": 0.4, "conf": 0.93, "spk": "A"}, ...]}
{"segments": [{"text": "...", "start": 0.0, "end": 1.8, "speaker": "agent", "words": [...]}]}   // e.g. Arc Transcriber verbose_json
{"inputs": ["text one", {"text": "text two", "domain": "nba"}], "domain": "talk"}             // batch; also "instances"
```

Per-request options (top level or in `config`): `domain`, `asr`, `emit`, `tags`, `redact`, `thresholds`, `fine_auto`,
`guard`, `command_context`. They apply to that request only. Load-time keys (`precision`, `backend`, `threads`,
`max_ctx`, `stream_ctx`) are fixed at start and refused per request (400). `text/plain` bodies work too (`?domain=`).

## Realtime

```
ws://host:8080/v1/realtime?domain=talk
-> {"type":"words","words":["my","name","is"]}   {"type":"words","words":["John","Smith."],"endpoint":true}
-> {"type":"text","text":"call me at 415 555 0199.","endpoint":true}   {"type":"words","words":[...],"from":3}
-> {"type":"session.update","session":{"domain":"meeting","redact":"mask"}}   {"type":"reset"}   {"type":"session.close"}
<- session.created, tags.update {n_words, a0, spans, events, utterances, current}, session.updated, error
```

Each session has its own handle; an update computes only the new words.

## Configuration

| variable | default | |
|---|---|---|
| `PORT`, `HOST` | `8080`, `0.0.0.0` | |
| `ARCTAG_MODEL_DIR` | first of `/opt/ml/model`, `/repository`, `/opt/arc/model`, `./model` (also their `onnx/` or `arc-tagger/`) | folder with `arctag_meta.json` |
| `ARCTAG_PRECISION` | `fp32` | `int8`: 26 MB per handle, about 2x faster streaming ([int8.md](../docs/int8.md)) |
| `ARCTAG_WORKERS` | 2 | handles for request / response calls |
| `ARCTAG_MAX_STREAMS` | 16 | concurrent realtime sessions |
| `ARCTAG_DOMAIN`, `ARCTAG_CONFIG` | `auto`, | default domain; a `configs/NAME.json` for every handle |
| `ARCTAG_MODEL_ID` | `arc-tagger` | |
| `ARC_QUEUE_TIMEOUT_S`, `ARC_MAX_BODY_MB`, `ARC_MAX_CONNECTIONS`, `ARC_ACCESS_LOG` | 60, 100, 64, 1 | |

## Images

```bash
docker buildx build --platform linux/amd64,linux/arm64 -f deploy/Dockerfile --target runtime -t arc-tagger:0.1.0 .     # weights mounted
deploy/fetch_weights.sh && docker buildx build --platform linux/amd64,linux/arm64 -f deploy/Dockerfile --target bundled -t arc-tagger:0.1.0-v0.4.1 .
```

- Runtime base: `gcr.io/distroless/cc-debian12:nonroot` (no shell, no package manager), uid 65532, read-only weights;
  `HEALTHCHECK` is the binary itself.
- amd64 and arm64 from one build. x86-64 hosts need AVX2 + FMA (checked at load).
- **CPU only**: no GPU code; the model is too small for a GPU to pay off ([platforms/nim/](platforms/nim/README.md)).
- Other weights in the same folder format work with the same image (`--build-arg WEIGHTS=... --build-arg MODEL_ID=...`).
- Audio in, transcript + tags out: `WhissleAI/arc_transcriber` `deploy/Dockerfile.combined` links this library and
  `deploy/server/tagsrv.c` into its server.

## Conformance

```bash
deploy/conformance/run.sh deploy/weights/arc-tagger --python /path/to/python-with-numpy-onnxruntime-tokenizers
```

It builds and starts the server and checks every route and request shape, that per-request options do not leak, the
error contract, SageMaker and Snowflake shapes, realtime (streamed words give the same spans as one offline call),
concurrency, and **parity with `python/arctag_ref.py`** (last run: 122 cases, 654 spans, 0 differing cases, max score
difference 1.0e-4). With a running Docker daemon it repeats the run against the image, after checking it starts with
`--network none`.

## Folders

| folder | what |
|---|---|
| [`server/`](server/README.md) | the server's C source |
| [`conformance/`](conformance/README.md) | the conformance run |
| [`platforms/`](platforms/README.md) | Hugging Face, SageMaker / AWS Marketplace, Snowflake, NVIDIA NIM |
