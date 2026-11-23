# Examples

Small programs built with the library (`cmake --build build`); each takes `-m MODEL_DIR`, `-c CONFIG.json`, `--int8`
and `--backend`. Commands: [docs/USAGE.md](../docs/USAGE.md#command-line-tools).

| file | shows |
|---|---|
| `arctag_cli.c` | tag one text or word array; `--stream` for push / poll |
| `batch_tag.c` | JSON Lines batch tagging |
| `stream_asr.c` | ASR interim / final / endpoint messages with push and poll |
| `redact.c` | redaction |
| `vertical_run.c` | run one vertical config over an input |
| `compare_precision.c` | fp32 vs int8 side by side |
| `bench.c` | load time, latency, memory |
| `arctag_server.c` | minimal one-stream JSON server (production: [deploy/](../deploy/README.md)) |
| `data/` | made-up inputs |

```bash
./build/vertical_run configs/contact_center.json examples/data/call_center.txt
./build/stream_asr -c configs/voice_assistant.json examples/data/asr_stream.jsonl
```
