# Conformance run

Builds `arctag-serve`, starts it, and checks every route against the contract and against `python/arctag_ref.py` on
the same texts. What it covers: [deploy/README.md](../README.md#conformance).

```bash
deploy/conformance/run.sh MODEL_DIR [--python PY] [--skip-reference] [--no-docker]
```

- `MODEL_DIR`: the model folder (`deploy/fetch_weights.sh` or `scripts/fetch_model.sh`).
- `PY`: a Python with numpy, onnxruntime 1.20.1 and tokenizers, for the reference.
- With a running Docker daemon it repeats the run against the image; it never starts Docker itself.

| file | what |
|---|---|
| `run.sh` | build, start, run, stop (native, then Docker) |
| `conformance.py` | the checks |
| `http_client.py` | a small HTTP / WebSocket client with no dependencies (shared with arc_transcriber) |
