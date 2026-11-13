# arctag: C API for the Arc Tagger

A small C library, with a CLI, examples and a tiny JSON/HTTP server, around **Arc Tagger v0.4.0**. Arc Tagger is a
text-only streaming tagger with about 10 M parameters. Give it words from any ASR, or plain text, and it returns:

- **spans**:
  - entities: person, organization, place, team, product, legal reference, ...;
  - numbers: date, time, money, count, score, phone / ID digits, ...;
  - PII: name, phone, email, address, card, gov id, DOB, handle;
  - profanity and toxicity.
- **events**, per word: turn yield, interruption, command complete, speaker change, story start, legal objection and
  rulings, meeting decision and action item, sports plays and scores.
- **utterance read-outs**:
  - intent;
  - dialogue act;
  - safety: scam, PII leak / request, threat, crisis, child safety, solicitation, toxicity, jailbreak attempt;
  - affect: emotion, arousal, valence, intensity;
  - speaker role, which needs speaker ids;
  - item / topic;
  - content state: mode, phase, game state.

It runs on CPU with one of two backends: a **native** backend in plain C (the default: no dependencies, loads in well under
200 ms, streams at a few ms per word) or **ONNX Runtime**. Both read the same model files: `arctag.onnx` (fp32, 34 MB)
and `arctag_int8.onnx` (int8, 11.5 MB), and you choose one at load time. Decoding follows the T0 (zero-lag) path of the
Arc Tagger v0.4.0 Python package. The output matches the package's tags and on/off decisions; with ONNX Runtime it
matches this repo's Python reference bit for bit, and the native backend makes the same decisions (see [Parity](#parity)).

## Quickstart

```bash
git clone git@github.com:WhissleAI/arc-tagger-c.git && cd arc-tagger-c
./scripts/download_onnxruntime.sh              # prebuilt ONNX Runtime 1.20.1 -> third_party/onnxruntime (optional, see Backends)
export HF_TOKEN=hf_xxx                         # read token with access to the private model repo
./scripts/fetch_model.sh                       # -> ./model (arctag.onnx, arctag_int8.onnx, arctag_meta.json, tokenizer.json)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
ctest --test-dir build --output-on-failure

./build/arctag_cli "Hi, this is Maria Lopez. My number is 555 867 5309."
./build/arctag_cli --int8 -d meeting "Priya will send the final numbers by Friday."
./build/arctag_cli --backend onnxruntime "Same text, run by ONNX Runtime."
./build/arctag_cli --redact mask "Email me at jo@example.com or call 415 555 0199."
./build/vertical_run configs/contact_center.json examples/data/call_center.txt
./build/stream_asr -c configs/voice_assistant.json examples/data/asr_stream.jsonl
./build/arctag_server -c configs/trust_safety_gaming.json --port 8090
```

```bash
curl -s localhost:8090/tag -d 'Can you call me back at 415 555 0199 tomorrow?'
curl -s localhost:8090/tag -d '{"text": "The Lakers are up 98 to 95.", "domain": "nba"}'
curl -s localhost:8090/push -d '{"words": ["hello", "my", "name", "is", "John", "Smith."], "endpoint": true}'
curl -s localhost:8090/poll
```

**Windows:** run `scripts/download_onnxruntime.ps1` in PowerShell, then the same CMake commands (MSVC). The build
copies `onnxruntime.dll` next to the executables. For the model, run `scripts/fetch_model.sh` from Git Bash.

**macOS:** the download script picks `osx-arm64` or `osx-x86_64`. You can also pass one explicitly, for example
`./scripts/download_onnxruntime.sh osx-universal2`.

[docs/USAGE.md](docs/USAGE.md) covers every tool, the config keys and the full output schema. [docs/int8.md](docs/int8.md)
covers the int8 variant.

## Model files and the Hugging Face token

The model is in the **private** Hugging Face repo `WhissleAI/arc-tagger`. The C library needs only its `onnx/` folder:

| file | what |
|---|---|
| `onnx/arctag.onnx` | fp32 full-context T0 forward (opset 17, dynamic word axis, up to 2048 words per pass), probabilities out |
| `onnx/arctag_int8.onnx` | the same graph with dynamic int8 weights (optional, see [Precision](#precision-fp32-and-int8)) |
| `onnx/arctag_meta.json` | read-out columns, per-profile thresholds, span-head families, event tiers, fine-domain cues, feature buckets |
| `onnx/tokenizer.json` | byte-level BPE (8k) tokenizer, read directly by the C tokenizer |
| `onnx/SHA256SUMS` | checksums, verified by `fetch_model.sh` |

To fetch them:

1. Get read access to the repo, then create a **read** token at <https://huggingface.co/settings/tokens>.
2. Run `export HF_TOKEN=hf_...` and `./scripts/fetch_model.sh [DEST]`.

`fetch_model.sh` reads these environment variables:

| variable | default | meaning |
|---|---|---|
| `HF_REPO` | `WhissleAI/arc-tagger` | repo |
| `HF_REVISION` | `v0.4.0` | tag, branch or commit |
| `ARCTAG_FETCH` | `all` | `fp32`, `int8` or `all` |
| `ARCTAG_INT8_URL` | | another location for the int8 graph |

If the int8 graph can't be found, the script says so and installs the fp32 files. The int8 tests then report as
skipped.

## API (`include/arctag.h`)

```c
#include "arctag.h"

char err[256];
arctag_options o; arctag_options_init(&o);
o.precision = ARCTAG_INT8;                                       /* default ARCTAG_FP32 */
o.threads = 1;                                                   /* default: $ARCTAG_THREADS or ORT's choice */
arctag_t *h = arctag_create_ex("model", &o, err, sizeof err);   /* NULL on error, message in err */
/* or: h = arctag_create_from_config("model", "configs/meetings.json", err, sizeof err); */
arctag_set_option(h, "domain", "meeting");

char *json = arctag_tag_text(h, "Okay team, let's start. Priya sends the numbers Friday.");
puts(json);
arctag_free_str(json);

/* streaming: push words as they arrive, poll whenever you want fresh output */
arctag_push_words(h, "[\"my\", \"name\", \"is\"]");
arctag_push_words(h, "{\"words\": [\"John\", \"Smith.\"], \"endpoint\": true}");
char *upd = arctag_poll(h);   /* spans in the recent window, new events, closed utterances, running read-out */
arctag_free_str(upd);

arctag_destroy(h);
```

| function | |
|---|---|
| `arctag_create(model_dir, err, errlen)` | load with defaults (fp32, unless `$ARCTAG_PRECISION=int8`) |
| `arctag_create_ex(model_dir, const arctag_options*, err, errlen)` | load with options: `precision`, `threads`, `model_file` |
| `arctag_create_from_config(model_dir, config_path, err, errlen)` | load and apply a config file (`configs/*.json`) |
| `arctag_apply_config(h, json)` | apply a config given as a JSON string |
| `arctag_set_option(h, key, value)` | one option (see below) |
| `arctag_tag_text(h, text)` | tag a whole text (whitespace words; `. ? !` ends an utterance) |
| `arctag_tag_words(h, words_json)` | the same for pre-split words (`"eou": true` on a word ends an utterance) |
| `arctag_push_words(h, json)` | streaming input: append, revise from index `from`, `endpoint` |
| `arctag_poll(h)` | bring the model up to date with the stream (at most the newest `stream_ctx` words) and return the update |
| `arctag_reset(h)`, `arctag_info(h)`, `arctag_last_error(h)` | forget the stream; describe the handle; last error |
| `arctag_free_str(s)`, `arctag_destroy(h)`, `arctag_version()` | |

Every returned string is malloc'd JSON, which you release with `arctag_free_str`. A handle is not thread-safe, so use
one per stream or thread. Each handle holds its own copy of the model: with the native backend about 72 MB peak RSS with
fp32 and 26 MB with weight-only int8; with ONNX Runtime about 118 MB with fp32 and 77 MB with dynamic int8 (see
[Backends](#backends)). Under ONNX Runtime a full 2048-word pass peaks near 1 GB, because memory grows faster than
linearly with the window, so lower `max_ctx` to reduce that.

**Options** (`arctag_set_option`, or the same keys in a config):

| key | values | default |
|---|---|---|
| `domain` | `auto`, `talk`, `meeting`, `court`, `assistant`, `sports`, `news`; a fine domain (`nba` `nhl` `nfl` `soccer` `cricket` `court_appellate` `court_trial` `assistant_noisy` `stream_chat`); or `ui:fine` | `auto` |
| `asr` | where the words come from: `unknown`, `human` (human-written text), `stock_rnnt`, `arc_transcriber`, `nemotron35`, `deepgram`, `final_only`, `web` | `unknown` |
| `backend` | `native` / `onnxruntime` (reloads the model) | `native` (or `$ARCTAG_BACKEND`) |
| `precision` | `fp32` / `int8` (reloads the model) | `fp32` |
| `threads` | ONNX Runtime intra-op threads (reloads the session); the native backend uses one thread | `$ARCTAG_THREADS` or 0 |
| `fine_auto` | `on` / `off`: inside `sports` and `court`, pick the fine domain from lexicon cues the way the package does | `on` |
| `emit` | output sections: `entity number pii profanity toxicity events intent act safety emotion arousal valence intensity role item state` | all |
| `tags` | tag prefixes to keep | all |
| `redact` | `off`, `mask`, or a policy such as `phone=mask,name=mask,contact=off` | `off` |
| `thr:PATTERN` | threshold override: a tag, `prefix*`, `...@readout`, or `head:FAMILY`; `""` removes it | |
| `thr_reset` | drops all overrides | |
| `max_ctx` | words per model pass in `tag_text` / `tag_words`; longer texts are cut into independent chunks | 2048 |
| `stream_ctx` | words of context the stream keeps (ONNX Runtime re-runs them on each `arctag_poll`) | 256 |
| `stream_state` | native: `on` carries the state from poll to poll; `off` re-runs the `stream_ctx` window like ONNX Runtime (same output, slower) | `on` |

The domain is a model input: it sets which tags are valid and selects the thresholds. `auto` runs the model's
`unknown` domain, where every tag is valid, so sports events or items can fire on unrelated text. Set the domain when
you know it, or use `emit` to drop sections you don't need.

**Word objects** (`push_words`, `tag_words`) are a string or
`{"w": "word", "t0": 1.20, "t1": 1.45, "conf": 0.93, "spk": "A", "final": true, "eou": false}`. Only `w` is required.
Aliases are accepted: `word` / `text`, `start` / `end`, `confidence`, and `speaker` (string or number). Timings in
seconds, confidence and speaker ids improve the read-outs, and speaker roles need `spk`.

## Output (short form)

```jsonc
{"profile": "talk:talk", "n_words": 31, "words": [...],
 "spans": [{"type": "safety.pii.phone_number", "family": "pii", "start": 9, "end": 12, "text": "555 867 5309",
            "score": 0.99998, "thr": 0.3, "src": "head", "alt": {"tag": "safety.pii.gov_id", "p": 3e-06}, "ver": 0.70}],
 "events": [{"type": "discourse.turn_yield", "q": "ev_at", "word": 30, "score": 0.91, "thr": 0.6, "text": "ship?"}],
 "utterances": [{"start": 0, "end": 9, "text": "...",
   "intent": [{"tag": "intent.report_inform", "p": 0.71, "thr": 0.5, "on": true}, ...],
   "act": [...], "safety": [...], "emotion": [...], "arousal": [...], "valence": [...], "intensity": [...],
   "role": [...], "item": [...], "state": {"content.mode": [...], ...}}],
 "redacted_text": "..."}                                  // when redaction is on
```

Spans cover the word range `[start, end)`. Each read-out entry is `{"tag", "p", "thr", "on"}`, where `on = p >= thr`
for the active profile and its overrides. `arctag_poll` returns
`{"n_words", "a0", "profile", "spans", "events", "utterances", "current"}`: replace your spans with `end > a0` by the
poll's spans. [docs/USAGE.md](docs/USAGE.md#output-json) has every field.

## Tags

| family | tags |
|---|---|
| entity spans | `entity.mention.{person,team,organization,place,work_product,event_name,legal_ref,game_entity,game_map,term}` |
| number spans | `entity.number.{date,time_of_day,duration,clock,money,percent,count,ordinal,score,stat,measure,ratio_odds,field_position,jersey_number,ref_number,id_pii}` and `entity.numeral` (any number; overlaps the subtypes) |
| PII spans | `safety.pii.{private_name,phone_number,email,street_address,card_account_number,gov_id,url_handle,date_of_birth,public_contact}` |
| profanity / toxicity spans | `safety.profanity.{mild,strong}`, `safety.toxicity.*` (5) |
| events (28) | `event.sports.*` (18), `event.legal.{objection,ruling_sustained,ruling_overruled}.recent`, `event.meeting.{decision,action_item}.recent`, `discourse.{turn_yield,interruption}`, `discourse.assistant.command_complete`, `speaker.change`, `content.story_start` |
| intent | `intent.*` (26 general intents), `discourse.intent.*` (60 assistant intents) |
| dialogue act | `discourse.act.*` (7), `discourse.answer`, `discourse.backchannel`, `discourse.assistant.*` |
| safety | `safety.scam.*` (6), `safety.pii.{leak,request}`, `safety.threat.*`, `safety.crisis.*`, `safety.child.*`, `safety.solicitation.*`, `safety.toxicity.*`, `safety.jailbreak.attempt` |
| affect | `affect.emotion.*` (5), `affect.arousal.{low,medium,high}`, `affect.valence.{neg,neu,pos,mixed}`, `affect.intensity.{1,2,3}` |
| speaker role | `speaker.role.*` (19: play-by-play, analyst, host, judge, counsel, witness, agent, customer, device user, ...) |
| item | `item.case_area.*`, `item.esports_title.*` |
| content state | `content.{mode,subject,legal_phase,meeting_phase,game_state,sports_focus,breaking_news,...}`, `scene.*`, `meta.*` |

`arctag_meta.json` (`readouts`) lists every column.

## Backends

| | native (default) | ONNX Runtime |
|---|---|---|
| dependencies | none (libc, libm) | ONNX Runtime 1.20.1 (`scripts/download_onnxruntime.sh`) |
| streaming | each poll computes only the words added or revised since the last poll | each poll re-runs the last `stream_ctx` words |
| threads | one per handle | `threads` intra-op threads |
| output | same decisions as the Python reference; scores within 1e-4 | bit-equal to the Python reference |

Choose one with any of:

- `arctag_options.backend = ARCTAG_BACKEND_NATIVE` / `ARCTAG_BACKEND_ONNXRUNTIME` for `arctag_create_ex`;
- `arctag_set_option(h, "backend", "native" | "onnxruntime")` (reloads the model) or `"backend"` in a config;
- `--backend native|onnxruntime` in the CLI tools, the examples and the server;
- `ARCTAG_BACKEND=onnxruntime` in the environment (used when the options leave the backend at its default).

`-DARCTAG_WITH_ORT=OFF` builds the library, tools and tests without ONNX Runtime (native only).

Measured with `bench` (`examples/data/*.txt`, 600 streamed words each pushed on its own and polled, `stream_ctx` 256;
batch = `tag_text` per line, 2 passes) at 1 thread on a 24-core Threadripper 2970WX whose cores were shared with other
jobs (load average 13-17), nice 19, best of two runs. Streaming p50 / p95 ms per word; batch ms per word. Absolute
numbers on a quiet machine will be lower; compare the ratios.

| backend, model file | file | load | streaming | batch | peak RSS |
|---|---|---|---|---|---|
| ONNX Runtime, fp32 | 34.0 MB | 935 ms | 185 / 279 | 0.58 | 118 MB |
| ONNX Runtime, int8 (dynamic) | 11.5 MB | 563 ms | 155 / 177 | 0.45 | 77 MB |
| ONNX Runtime, int8 (weight-only) | 9.4 MB | 712 ms | 191 / 277 | 0.77 | 118 MB |
| **native, fp32** | 34.0 MB | **155 ms** | **6.2 / 12.1** | 0.96 | 72 MB |
| native, int8 (dynamic) | 11.5 MB | 110 ms | 6.4 / 13.4 | 1.14 | 52 MB |
| **native, int8 (weight-only)** | **9.4 MB** | **68 ms** | **2.9 / 7.9** | 0.66 | **26 MB** |

The native backend streams about 30x faster, because a poll costs one word instead of `stream_ctx` words; it also loads
about 6x faster and uses less memory. For one-shot `tag_text` ONNX Runtime is faster (0.58 vs 0.96 ms per word).
The weight-only int8 file (made with `tools/quantize_model.py`) is the fastest and smallest under the native backend; see
[Precision](#precision-fp32-and-int8).

## Precision: fp32 and int8

Two int8 files work with both backends:

- **weight-only int8** (`python tools/quantize_model.py model/arctag.onnx model/arctag_int8.onnx`): int8 weight tables,
  float compute. 9.4 MB. Streaming equals the offline pass exactly. **Recommended**, especially with the native backend
  (26 MB RSS, about 2x faster streaming than fp32), and it agrees best with fp32.
- **dynamic int8** (the `arctag_int8.onnx` the model repo ships today): activations are quantised per call under ONNX
  Runtime. 11.5 MB. The native backend converts it to float tables at load.

Agreement of the int8 files with fp32 (`int8_agreement`, the 120 parity cases):

| | span F1 vs fp32 | max \|p diff\| |
|---|---|---|
| weight-only int8, either backend | 0.987 | 0.103 |
| dynamic int8, native | 0.981 | 0.123 |
| dynamic int8, ONNX Runtime | 0.975 | 0.172 |

On the Whissle internal held-out test set the dynamic int8 file under ONNX Runtime agrees with fp32 on 96% of spans
(toxicity 82%, PII 94%, numbers 97%) and on 96-98% of events, intents and safety read-outs; span F1 against the test
labels is within 0.3 pt of fp32 (0.542 vs 0.544). The weight-only file agrees on 98.6% of spans (toxicity 92%) and
matches fp32 span F1 within 0.1 pt and intent accuracy exactly.

Use int8 for edge devices and many handles per process, and fp32 when you need the reference output. Under ONNX
Runtime, dynamic int8 streaming scores depend slightly on the window, because activations are quantised per call; the
weight-only file and the native backend do not have this effect. [docs/int8.md](docs/int8.md) has the details.

## Configs and verticals

Each `configs/*.json` sets these for one use case:

- domain;
- precision and threads;
- the sections to emit;
- threshold overrides;
- redaction;
- streaming or batch.

Load one with `-c`, `arctag_create_from_config` or `POST /config`.

| config | for | key settings |
|---|---|---|
| [`voice_assistant`](configs/voice_assistant.json) | assistants, voice commands | assistant domain, streaming ctx 128, intents / acts / slots, jailbreak flag lowered to 0.45 |
| [`contact_center`](configs/contact_center.json) | call centers, agent assist | talk domain, streaming, roles, sentiment, scam / PII-request / escalation lowered for recall, PII and profanity masked |
| [`trust_safety_gaming`](configs/trust_safety_gaming.json) | chat moderation, gaming | stream_chat domain, toxicity / threat / child-safety / self-harm review triggers, affect, PII and toxicity masked |
| [`legal`](configs/legal.json) | hearings, depositions | court_trial domain (fine_auto), objection / ruling events at 0.4, legal references, court roles, legal phase |
| [`meetings`](configs/meetings.json) | meeting notes | meeting domain, batch, decisions and action items lowered, assign-task intent, meeting phase |
| [`sports`](configs/sports.json) | live commentary | sports domain (fine_auto picks the league), streaming, play / score events, profanity masked |
| [`news`](configs/news.json) | news and talk shows | news domain, batch, entities, story starts, roles, breaking-news state |
| [`pii_redaction`](configs/pii_redaction.json) | generic redaction | PII + utterance safety only, mask every private PII category |
| [`edge_low_latency`](configs/edge_low_latency.json) | devices, low latency | **int8, 1 thread**, streaming ctx 96 |

The examples show how each API is used:

| example | shows |
|---|---|
| `examples/batch_tag.c` | JSONL batch tagging |
| `examples/stream_asr.c` | ASR interim / final / endpoint with push and poll |
| `examples/redact.c` | redaction |
| `examples/vertical_run.c` | per-vertical config runner |
| `examples/compare_precision.c` | fp32 vs int8 side by side |
| `examples/bench.c` | latency and memory |

Made-up inputs are in `examples/data/`. [docs/USAGE.md](docs/USAGE.md) has the commands.

## Web demo

`demo/` is a browser demo backed by this library. Type a message and watch spans, intent / act / safety lanes and
redaction appear word by word. With `DEEPGRAM_API_KEY` set, the Live tab does the same for your microphone.

```bash
demo/run.sh            # builds if needed, then serves http://localhost:8080 (needs ./model and Python 3 + websockets)
demo/run.sh --int8     # int8 by default; the page can switch precision per session
```

The server (`demo/server.py`, one pinned dependency: `websockets`) loads the shared library that the build makes next
to the static one (`build/libarctag.so` / `.dylib` / `arctag.dll`; turn it off with `-DARCTAG_BINDINGS_LIB=OFF`)
through `ctypes`. [demo/README.md](demo/README.md) covers the flags, the tabs and the protocol.

## Parity

| check | result |
|---|---|
| Python reference (fp32) vs the Arc Tagger v0.4.0 Python package, 16 texts x 7 domains | 526 spans, 294 events, 280 utterances and 2,728 read-out rows: same tags, ranges and on/off decisions, same top affect, same redacted text and fine domain; max p diff 5e-4 (the package rounds to 1e-3). One listing difference: an off safety row whose p sits on the listing floor (0.5 x thr = 0.01; reference p 0.009994) is listed by the package only |
| C tokenizer vs HF `tokenizers`, 37 strings (punctuation, numbers, emoji, CJK, Cyrillic, Arabic digits, special tokens) | identical ids |
| C library vs Python reference, fp32, 120 cases (8 with configs: emit, tags, redaction, overrides) | 563 spans, all fields identical, scores bit-equal |
| C library vs Python reference, int8 | 564 spans, all fields identical, scores bit-equal |
| C streaming (push per utterance + poll) vs `arctag_tag_text`, fp32 | identical within 1e-4 |
| native backend vs Python reference, fp32, 120 cases | 563 spans, every tag, range and on/off decision identical; max score difference 4.6e-5 |
| native streaming vs native `arctag_tag_text`, fp32 and int8 | identical (scores bit-equal) |
| native backend vs ONNX Runtime, `tag_text` on `examples/data` | max score difference 1.1e-4 |

The two backends round differently (ONNX Runtime has its own float32 sigmoid and reduction order), so their scores
differ in the last digits. The bit-equal and 1e-4 checks apply to ONNX Runtime; the native backend is checked at 5e-4
with identical decisions.

`ctest` covers:

- tokenizer;
- `parity_fp32`, `parity_int8` (ONNX Runtime), `parity_native_fp32`, `native_int8_stream`;
- `int8_agreement`, `int8_agreement_ort` (F1 >= 0.90, max |dp| <= 0.25);
- `native_stream` (streaming with revisions and speaker changes equals a fresh pass, long sessions, both backends agree,
  truncated / corrupted model files are refused);
- `config` (all `configs/*.json`, option errors, filters, redaction, overrides);
- example smoke tests;
- `demo_smoke` (the web demo, word by word, when Python 3 and `websockets` are installed).

`-DARCTAG_PARITY_EXACT=ON` requires bit-equal scores. This holds on x86-64 with the bundled ORT 1.20.1, and the
default tolerance is 1e-4. `-DARCTAG_SANITIZE=ON` builds with ASAN + UBSan; CI runs the suite under it with leak
checking.

## How it works

`text -> words -> utterances -> per-word input features -> model (native or ONNX Runtime) -> per-word / per-utterance probabilities -> decode`

- **Inputs** (`src/arctag.c`), per word:
  - up to 8 BPE pieces (`" " + word`: the first 7 and the last);
  - digit flags;
  - gap / duration / confidence buckets;
  - final / interim;
  - speaker slot and speaker change;
  - ASR source;
  - utterance starts;
  - the domain id.
- **Outputs** of the graph: per-word probabilities (content state, span begin / inside / end, events), per-family span
  type and boundary probabilities, a span verifier score, and per-utterance, per-speaker and per-item probabilities.
- **Decoding** (the T0, zero-lag tier):
  - **Span families** marked `use` in `arctag_meta.json` (entity, number, PII, profanity) are decoded as greedy BIO on
    `1 - P(O) >= thr`, and each span is typed by the argmax of its summed log type-probabilities.
  - **Other span tags** (toxicity among them) use runs of `span_in >= thr`, split at `span_beg` and at utterance starts.
  - **Events** fire at `p >= thr` (T0 display tier only) and are deduplicated within 3 words.
  - **Read-outs** come from the utterance outputs at the utterance's last word. The affect read-out is the top-1 of each
    group.
  - **Fine domain**: inside `sports` / `court`, `fine_auto` re-checks lexicon cues every 25 words and switches the fine
    domain, re-decoding the recent spans the way the Python package does.
- **Thresholds**: the model release's thresholds per profile (display, events, span families), with display floors on top:
  spans >= 0.3, and toxicity >= 0.5 outside the safety domains.

## Limits

- `tag_text` is offline over at most `max_ctx` (2048) words per pass. Longer texts are split into independent chunks.
- Under ONNX Runtime, `arctag_poll` re-runs the whole `stream_ctx` window every time. Poll per ASR update rather than
  per word, or lower `stream_ctx`, or use the native backend, which carries its state from poll to poll.
- Not included:
  - the session-KB pointer (custom name lists);
  - the T1 / T2 delayed tiers;
  - automatic selection of the top-level domain (only the fine domain inside `sports` / `court` is automatic).
- English only.

## Layout

```
include/arctag.h        public API
src/arctag.c            features, backends, decoding, configs, redaction, JSON
src/native.c            native backend (plain C; AVX2 / NEON kernels where available)
src/onnx_file.c         reads the model tables from the .onnx files for the native backend
src/tokenizer.c         byte-level BPE reading tokenizer.json (+ src/unicode_tables.h from tools/gen_unicode.py)
configs/                vertical configs (JSON)
demo/                   web demo: server.py (asyncio + ctypes), arctag_ffi.py (ctypes binding), static/ (page)
examples/               arctag_cli, arctag_server, batch_tag, stream_asr, redact, vertical_run, compare_precision, bench; data/
tests/                  test_tokenizer, test_parity, test_config, test_int8_agreement, test_native, demo_smoke.py, expected/ (made-up texts), make_expected.py
python/arctag_ref.py    Python reference (numpy + onnxruntime + tokenizers)
docs/                   USAGE.md, int8.md
scripts/                download_onnxruntime.sh/.ps1, fetch_model.sh
tools/quantize_model.py weight-only int8 storage for arctag.onnx (smaller file, float compute, streaming-exact)
third_party/cjson       cJSON 1.7.19 (MIT)
```

Regenerate the references with `pip install numpy onnxruntime==1.20.1 tokenizers`, then
`python tests/make_expected.py model [--precision int8]`.

## Licence

Code (including this inference code) and model weights: CC BY-NC 4.0 plus the Whissle terms in [LICENSE](LICENSE),
non-commercial use only. Commercial use needs a separate written licence from Whissle Inc.: ksingla@whissle.ai.
Third-party components keep their own licences: see [NOTICE](NOTICE).
