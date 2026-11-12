# Arc Tagger web demo

A browser demo of Arc Tagger that runs on your machine and is backed by this repo's C library. Type a chat message and
watch the tags appear word by word, before you press Enter:

- PII, profanity and toxicity spans;
- entities and numbers;
- intent, dialogue act and utterance-safety lanes;
- events;
- redaction.

With a Deepgram key, the Live tab does the same for your microphone or a browser tab.

## Requirements

- What the library needs: CMake, a C compiler and the prebuilt ONNX Runtime (`scripts/download_onnxruntime.sh`).
- The model files in `./model`, from `scripts/fetch_model.sh` (needs `HF_TOKEN`, see the [main README](../README.md#model-files-and-the-hugging-face-token)).
- Python 3.9 or newer, plus one package: `websockets` (pinned in [`requirements.txt`](requirements.txt)).
- Optional: a [Deepgram](https://deepgram.com) API key for the Live tab.

## Run

```bash
./scripts/download_onnxruntime.sh                       # once
HF_TOKEN=hf_... ./scripts/fetch_model.sh                # once: ./model
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
python3 -m pip install -r demo/requirements.txt         # once
python3 demo/server.py                                  # http://localhost:8080
```

`demo/run.sh` does all of this in one go. It skips the steps that are already done, and it fetches the model only when
`./model` is empty and `HF_TOKEN` is set. Arguments after `run.sh` go to the server, for example
`demo/run.sh --int8 --port 9000`.

The build makes the static library and a shared copy for language bindings: `build/libarctag.so` (Linux),
`build/libarctag.dylib` (macOS) or `build/[Release/]arctag.dll` (Windows). The server loads the shared copy with Python
`ctypes`. To leave it out, configure with `-DARCTAG_BINDINGS_LIB=OFF`.

### Server flags

| flag | default | meaning |
|---|---|---|
| `--port` | `8080` (`$ARCTAG_DEMO_PORT`) | port; `0` picks a free one |
| `--host` | `127.0.0.1` (`$ARCTAG_DEMO_HOST`) | bind address; use `0.0.0.0` to serve your LAN, with care: there is no login |
| `--int8` | off | default precision int8 (`model/arctag_int8.onnx`); the page can still switch per session |
| `--model` | `./model` (`$ARCTAG_MODEL_DIR`) | model directory |
| `--configs` | `./configs` | vertical configs: every `*.json` is offered as a vertical |
| `--lib` | `build/` (`$ARCTAG_LIB`) | path of the shared library |
| `--threads` | `2` (`$ARCTAG_THREADS`) | ONNX Runtime threads per session |
| `--max-sessions` | `8` | concurrent sessions; each holds one model session (about 110 MB fp32, 75 MB int8) |

Environment: `DEEPGRAM_API_KEY` turns on the Live tab.

## The page

**Type** is the main tab and opens first. It is a chat box. The draft streams to the server on every word boundary and
after a 250 ms typing pause. Each draft replaces the open message's words in the library's stream
(`arctag_push_words` with `from`), and one `arctag_poll` follows. The page then shows:

- per word: provisional **T0** spans as soon as the word arrives, then confirmed **T1** spans. T1 means the span was closed
  by the next completed word, or by Enter;
- lanes under the words: intent, dialogue act and utterance safety at each word;
- alarm chips while you type, with the time from key press to alarm;
- a Tags panel: intent, act, entities, numbers, PII, profanity, toxicity, safety and events;
- per sent message, Arc's alarm (the word it fired on, and how long before Enter) next to the end-of-turn verdict a
  turn-based check would give;
- a latency line, with update round trip and tagger time per update;
- **Long text**, which streams a pasted or loaded text (one line = one utterance, a `Name:` prefix = a speaker) at 40, 150
  or 1500 words per minute, with a timeline and session context. **Load example** loads a made-up support chat.

The controls:

- **Setting** is the vertical. It lists every `configs/*.json` plus Auto, the generic profile with no config. Each config
  sets the domain, the emitted sections, the threshold overrides and the redaction policy.
- **Redaction** turns on masking with the config's policy, or with mask-all-private-PII when the config has none.
- **Precision** picks fp32 or int8.
- **Recording** (also `?video=1`) gives a clean 1920x1080 layout for screen recording.
- The **example row** types 36 made-up examples across 7 verticals at a human pace.

**Live** needs `DEEPGRAM_API_KEY` in the server's environment. Without it, the Mic and Start tab buttons are disabled and
a note says why. With it, the page sends 16 kHz PCM from your microphone (**Mic**) or a shared Chrome tab
(**Start tab**) to the server, which streams it to Deepgram nova-3 with diarization. The words go through the same push /
poll path.

The transcript shows turn cards with speaker, role and act, per-word overlays, lanes and event markers. Side panels
depend on the vertical: intents, commitments, escalation, the redaction log, speakers, objections, scores and others.
**Redacted output** shows exactly what a downstream consumer gets. **Bleep tab audio** plays the shared tab a little
delayed, with every masked word bleeped. The key never leaves the server.

**Developer** opens a drawer with the live JSON message stream and a copyable sample payload.

## How it works

```
browser ──WebSocket /ws──> demo/server.py ──ctypes──> libarctag (arctag_push_words / arctag_poll) ──> ONNX Runtime
             (JSON)            │  one handle per session, run in worker threads (the GIL is released during C calls)
                               └─ Live tab only: audio ──> Deepgram nova-3 ──> words
```

- `server.py` is one asyncio process built on `websockets`. It serves `static/`, the WebSocket and `GET /api/health` /
  `/api/profiles`.
- Every browser session gets its own arctag handle, configured from the vertical's config. Handles are pooled and reused.
- Drafts that arrive while the model runs are coalesced into one poll.
- Per update, the server sends:
  - the changed words (diffed);
  - T0 and T1 spans;
  - new events;
  - lane points, only where a value changes;
  - the running and closed-utterance read-outs;
  - one turn per utterance;
  - the safety tags that are on (Type tab).
- Thresholds, span decoding, events, read-outs and the fine domain (inside sports and court) come from the library.
- Redaction uses the library's categories and policy. The per-word masks are checked against the library's own
  `redacted_text` on every poll, and the result is reported as `stats.redact_mismatch` (expected 0). A masked word stays
  masked while it is unchanged (zero-lag hold).
- `arctag_ffi.py` is a small stand-alone ctypes binding of `include/arctag.h`, which you can reuse.

Privacy: typed text and transcripts live in memory for the session only. They are never written to disk, and the server
log holds only session starts and ends, counters and error types.

## Tests

`tests/demo_smoke.py` runs as the ctest `demo_smoke`. It is skipped when `websockets` or the model is missing. The test:

1. starts the server on a free port;
2. types two made-up messages word by word, partial word first, as the page does;
3. checks that spans, lanes, closed-utterance read-outs and typed alarms come back;
4. checks redaction: placeholders only, no masked word in clear, agreement with the library;
5. checks timing: every span shows up on its first word or at most one word later, and is confirmed at most one word
   after its last word;
6. repeats with int8 when that graph is present;
7. checks that the server log holds no typed text.

```bash
python3 tests/demo_smoke.py -v
ctest --test-dir build -R demo_smoke --output-on-failure
```

## Limits

- No T2 tier, no session name list (KB) and no automatic top-level domain detection: the library does not implement them.
  "Auto" runs the generic profile.
- The Type tab has no separate guard classifier. For contrast, the right column shows Arc's own end-of-turn read-out.
- Live ASR is Deepgram nova-3 only, and needs your own key and network access.
- English only.
