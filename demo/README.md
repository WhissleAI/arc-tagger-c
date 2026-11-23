# Arc Tagger web demo

A browser demo backed by this repo's C library, running on your machine. Type a chat message and watch PII,
profanity, toxicity, entities, numbers, intent / act / safety lanes, events and redaction appear word by word. With a
Deepgram key, the Live tab does the same for your microphone or a browser tab.

## Run

```bash
demo/run.sh                  # builds if needed, fetches ./model if empty, serves http://localhost:8080
demo/run.sh --int8 --port 9000
```

By hand: build the library ([README](../README.md#quickstart)), `python3 -m pip install -r demo/requirements.txt`
(one package: `websockets`), then `python3 demo/server.py`. Needs Python 3.9+.

The server loads the shared library the build makes (`build/libarctag.so`, `.dylib` or `arctag.dll`) through
`ctypes`; `-DARCTAG_BINDINGS_LIB=OFF` leaves it out.

| flag | default | meaning |
|---|---|---|
| `--port` | `8080` (`$ARCTAG_DEMO_PORT`) | `0` picks a free port |
| `--host` | `127.0.0.1` (`$ARCTAG_DEMO_HOST`) | `0.0.0.0` serves your LAN: there is no login |
| `--int8` | off | default precision int8; the page can switch per session |
| `--model` | `./model` (`$ARCTAG_MODEL_DIR`) | model directory |
| `--configs` | `./configs` | every `*.json` is offered as a vertical |
| `--lib` | `build/` | path of the shared library |
| `--threads` | `2` (`$ARCTAG_THREADS`) | ONNX Runtime threads per session (ONNX Runtime backend only) |
| `--max-sessions` | `8` | concurrent sessions, one library handle each |

`DEEPGRAM_API_KEY` in the server's environment turns on the Live tab. The key never leaves the server.

## The page

- **Type** (opens first): a chat box. Drafts stream to the server on every word boundary and after a 250 ms pause.
  You see provisional T0 spans per word, then confirmed T1 spans; intent, act and safety lanes; alarm chips with the
  time from key press to alarm; a tags panel; and per-update latency. **Long text** streams a pasted text at 40, 150
  or 1500 words per minute.
- **Live**: microphone (**Mic**) or a shared Chrome tab (**Start tab**) to Deepgram nova-3 with diarization; the words
  go through the same push / poll path. Turn cards show speaker, role and act; side panels depend on the vertical.
  **Bleep tab audio** replays the tab with masked words bleeped.
- **Controls**: **Setting** (a vertical from `configs/`, or Auto), **Redaction**, **Precision**, **Recording**
  (`?video=1`, a clean 1920x1080 layout), and an example row that types 36 made-up examples across 7 verticals.
- **Developer**: the live JSON message stream and a sample payload.

## How it works

```
browser --WebSocket /ws--> demo/server.py --ctypes--> libarctag (arctag_push_words / arctag_poll)
                              one handle per session, in worker threads; Live tab: audio -> Deepgram -> words
```

- One asyncio process on `websockets`; serves `static/`, the WebSocket, `GET /api/health` and `/api/profiles`.
- Drafts that arrive while the model runs are coalesced into one poll; updates carry only changed words and new items.
- Thresholds, decoding, events, read-outs, redaction and the fine domain all come from the library. Per-word masks are
  checked against the library's `redacted_text` on every poll (`stats.redact_mismatch`, expected 0).
- `arctag_ffi.py` is a stand-alone ctypes binding of `include/arctag.h` you can reuse.
- Privacy: text and transcripts live in memory for the session only; the log holds session starts and ends, counters
  and error types.

## Tests

`tests/demo_smoke.py` (ctest `demo_smoke`; skipped without `websockets` or the model) types made-up messages word by
word and checks spans, lanes, read-outs, alarms, redaction, timing (every span within one word), int8, and that the
log holds no typed text.

```bash
ctest --test-dir build -R demo_smoke --output-on-failure
```

## Limits

- No T2 tier, no session name list, no automatic top-level domain ("Auto" is the generic profile).
- Live ASR is Deepgram nova-3 only, with your own key and network access.
- English only.
