# arctag: C API for the Arc Tagger

A small C library (plus a CLI and a tiny JSON/HTTP server) around **Arc Tagger**: a ~9.8M-parameter, text-only
streaming tagger. Give it words from any ASR or plain text and it returns:

- **spans**: entities (person, organization, place, team, product, ...), numbers (date, time, money, count, score,
  phone/ID digits, ...), PII (name, phone, email, address, card, gov id, DOB, ...), profanity and toxicity.
- **events**: per-word events such as turn yield, interruption, command complete, legal objection, meeting decision or
  action item, and sports scores.
- **utterance read-outs**: intent, dialogue act, safety (scam, PII leak, toxicity, ...), emotion and arousal, speaker
  role (needs speaker ids), item/topic, and content state (mode, phase, game state).

The library is plain C with its own forward pass of the model: no ONNX Runtime at build or run time. Weights are a
memory-mapped file, `arctag.atw` (int8: 9 MB, fp32: 34 MB), exported from the model's ONNX graph. It uses one thread, about
15 MB, and about 1 ms per new word however long the stream has run. Decoding (thresholds, the span head, events and read-outs)
follows the T0 (zero-lag) path of the Arc Tagger demo engine, and the output matches the ONNX graph (see [Parity](#parity)).

## Quickstart

```bash
git clone git@github.com:WhissleAI/arc-tagger-c.git && cd arc-tagger-c
export HF_TOKEN=hf_xxx                         # read token with access to the private model repo
./scripts/fetch_model.sh                       # -> ./model (arctag.onnx, arctag_meta.json, tokenizer.json)
pip install numpy onnx onnxruntime tokenizers  # only to export; the library does not use them
python3 tools/export_model.py model --weights int8 --out model-int8   # -> arctag.atw, with parity against the ONNX
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./scripts/test_native.sh model --quick         # every test, in about 2 minutes (see Parity)

./build/arctag_cli -m model-int8 "Hi, this is Maria Lopez. My number is 555 867 5309."
./build/arctag_cli -m model-int8 -d meeting "Priya will send the final numbers by Friday."
./build/arctag_cli -m model-int8 --stream "words arrive one at a time. like this!"     # push/poll demo
./build/arctag_server -m model-int8 --port 8090     # HTTP: POST /tag, POST /push, GET /poll, POST /reset, GET /health
```

```bash
curl -s localhost:8090/tag -d 'Can you call me back at 415 555 0199 tomorrow?'
curl -s localhost:8090/tag -d '{"text": "The Lakers are up 98 to 95.", "domain": "nba"}'
curl -s localhost:8090/push -d '{"words": ["hello", "my", "name", "is", "John", "Smith."], "endpoint": true}'
curl -s localhost:8090/poll
```

**macOS:** Apple platforms use Accelerate, so link `-framework Accelerate`. A universal library needs no per-architecture flags:
`cmake -S . -B build -DCMAKE_OSX_ARCHITECTURES="arm64;x86_64"`. See [Using it in an app](#using-it-in-an-app).

**Other platforms:** Linux and other POSIX systems with GCC or Clang use the built-in matmul kernels (NEON on arm64, AVX2 on
x86_64, checked when the model loads). MSVC has no build of the native runtime: CMake falls back to the ONNX Runtime
implementation in `ort/` (run `scripts/download_onnxruntime.ps1` first, and use an `arctag.onnx` model directory).

## Model files and the Hugging Face token

The model lives in the **private** Hugging Face repo [`ksingla025/arc-tagger`](https://huggingface.co/ksingla025/arc-tagger).
Exporting needs only the `onnx/` folder:

| file | what |
|---|---|
| `onnx/arctag.onnx` | full-context T0 forward (opset 17, dynamic word axis, up to 2048 words per pass), probabilities out. `tools/export_model.py` turns it into `arctag.atw` |
| `onnx/arctag_meta.json` | read-out columns, per-domain validity and display thresholds, span-head families, feature buckets |
| `onnx/tokenizer.json` | byte-level BPE (8k) tokenizer, which the C tokenizer reads directly |
| `onnx/SHA256SUMS` | checksums verified by `fetch_model.sh` |

1. Get read access to `ksingla025/arc-tagger` (ask the owner).
2. Create a **read** token at <https://huggingface.co/settings/tokens>.
3. `export HF_TOKEN=hf_...` and run `./scripts/fetch_model.sh [DEST]`. `HF_REPO` and `HF_REVISION` override the repo and
   revision (default `main`; `HF_REVISION=2ec982888e1b51f651db06c23afdc2450668ccab` pins the first `onnx/` upload).

Without the script:
`huggingface-cli download ksingla025/arc-tagger --include "onnx/*" --local-dir . && mv onnx model`.

The rest of that repo is the PyTorch checkpoint and the Python package (`arctag/`, `example.py`). This C repo does not
need it.

### Exporting a checkpoint

One command, for a new checkpoint as for this one:

```bash
python3 tools/export_model.py NEW_ONNX_DIR --out model-int8 --weights int8    # or --weights fp32
```

`NEW_ONNX_DIR` holds `arctag.onnx`, `arctag_meta.json` and `tokenizer.json`. Nothing is written to `--out` unless all of this
passes; otherwise the exit status is non-zero, with the reason on stderr:

1. **Shape of the graph.** Its inputs and outputs, every parameter's name and shape, the trunk's split layout and the count of
   architecture operators must match what `src/atn.c` implements. Read-out and span-head column counts and the number of domains
   may change with the tag set. A new layer or a changed width fails here and names what differs.
2. **Nothing dropped.** A parameter the runtime does not use is an error.
3. **No generated names.** Values the graph exporter left unnamed (attention score biases, RoPE frequencies, the mention prior,
   span-head slices) are found by walking from named parameters, never by names like `onnx::Mul_3635`.
4. **Parity.** The native forward pass, compiled from this checkout, runs on the new weights next to ONNX Runtime on the same graph:
   31 texts, up to 1024 words. Probabilities must agree within 2e-4 and the verifier logits within 5e-4 (`--tol`, `--tol-logit`).

`tests/test_export.py` checks each of these on altered graphs. If step 1 fails on a checkpoint that is meant to be different, the
forward pass in `src/atn.c` has to change with it.

`arctag.atw` holds the tables as the ONNX stores them, so an int8 export matches an int8 ONNX: int8 itself moves scores from fp32
by up to about 0.1 on a tag, and that is the quantization, not the runtime. Both kinds load the same way.

## API (`include/arctag.h`)

```c
#include "arctag.h"

char err[256];
arctag_t *h = arctag_create("model", err, sizeof err);          /* NULL on error, message in err */
arctag_set_option(h, "domain", "meeting");                      /* optional */

char *json = arctag_tag_text(h, "Okay team, let's start. Priya sends the numbers Friday.");
puts(json);
arctag_free_str(json);

/* streaming: push words as they arrive, poll whenever you want fresh output */
arctag_push_words(h, "[\"my\", \"name\", \"is\"]");
arctag_push_words(h, "{\"words\": [\"John\", \"Smith.\"], \"endpoint\": true}");
char *upd = arctag_poll(h);  /* spans in the recent window, new events, closed utterances, running read-out */
arctag_free_str(upd);

arctag_destroy(h);
```

| function | |
|---|---|
| `arctag_t *arctag_create(const char *model_dir, char *err, size_t errlen)` | load `arctag.atw`, `arctag_meta.json`, `tokenizer.json`. A corrupt, truncated or mismatched file returns NULL and a message |
| `int arctag_set_option(arctag_t*, const char *key, const char *value)` | `domain`, `asr`, `threads`, `max_ctx`, `stream_ctx` (see below) |
| `char *arctag_tag_text(arctag_t*, const char *text)` | tag a whole text (whitespace words; `. ? !` ends an utterance) |
| `char *arctag_tag_words(arctag_t*, const char *words_json)` | same, for pre-split words (`"eou": true` on a word ends an utterance) |
| `int arctag_push_words(arctag_t*, const char *words_json)` | streaming input (append, revise from index `from`, `endpoint`) |
| `char *arctag_poll(arctag_t*)` | run the words not run yet and return the update |
| `void arctag_reset(arctag_t*)` | forget the stream |
| `const char *arctag_last_error(const arctag_t*)` | last error message |
| `void arctag_free_str(char*)`, `void arctag_destroy(arctag_t*)`, `const char *arctag_version(void)` | |

A handle is not thread-safe, but handles are independent: any number can run at once on different threads (tested under
ThreadSanitizer). Each maps the weights read-only, so the operating system shares them between handles and processes.

**Options**

| key | values | default |
|---|---|---|
| `domain` | `auto`, `talk`, `meeting`, `court`, `assistant`, `sports`, `news`, or a fine domain (`nba` `nhl` `nfl` `soccer` `cricket` `court_appellate` `court_trial` `assistant_noisy` `stream_chat`), or a profile name `ui:fine` from `arctag_meta.json` | `auto` |
| `asr` | where the words come from, one of `arctag_meta.json` `asr_sources` (`unknown`, `gold` for human text, `final_only`, `web`, ...) | `default_asr` in the meta file (the setting the thresholds were fitted with) |
| `threads` | accepted for compatibility and ignored, like `$ARCTAG_THREADS`. Every call runs on its caller's thread, alone. Use one handle per thread for parallelism | |
| `max_ctx` | words per model pass in `tag_text` / `tag_words`; longer texts are cut into independent chunks | 2048 |
| `stream_ctx` | the stream's attention window in words, 16..2048. See [Streaming](#streaming) | 256 |

The domain conditions the model (FiLM, adapters and tag validity per domain) and selects the per-domain thresholds. `auto`
runs the model's `unknown` domain, where every tag is valid, so sports events can fire on non-sports text. Set the
domain when you know it.

**Word objects** (`push_words`, `tag_words`): a string, or
`{"w": "word", "t0": 1.20, "t1": 1.45, "conf": 0.93, "spk": "A", "final": true, "eou": false}`. Only `w` is required.
The aliases `word`/`text`, `start`/`end`, `confidence` and `speaker` (string or number) are also accepted. Timings
(seconds), confidence and speaker ids improve the read-outs when you have them. Speaker roles need `spk`.

## Output JSON

`arctag_tag_text` / `arctag_tag_words`:

```jsonc
{
  "profile": "talk:talk",                 // domain profile used
  "n_words": 31,
  "words": ["Hi,", "this", "is", "Maria", "Lopez", ...],
  "spans": [                               // sorted by (start, -end); start/end = word indices, end exclusive
    {"type": "entity.mention.person", "family": "entity", "start": 3, "end": 5, "text": "Maria Lopez",
     "score": 0.991061, "thr": 0.5, "src": "head",          // src: "head" (v2 span head) | "span_in" (per-tag runs)
     "alt": {"tag": "entity.mention.organization", "p": 0.004},   // runner-up type (head spans, merged spans)
     "ver": 0.997},                                         // span verification score (entity / number / PII tags)
    {"type": "safety.pii.phone_number", "family": "pii", "start": 9, "end": 12, "text": "555 867 5309", "score": 0.99998, ...}
  ],
  "events": [                              // T0 events, de-duplicated within 3 words per tag
    {"type": "discourse.turn_yield", "q": "ev_at", "word": 30, "score": 0.91, "thr": 0.42, "text": "ship?"}
  ],
  "utterances": [                          // one per utterance, read at its last word
    {"start": 0, "end": 9, "text": "Hi, this is Maria Lopez calling about my order.",
     "intent":  [{"tag": "intent.report_inform", "p": 0.21, "thr": 0.45, "on": false}, ...],   // top 4
     "act":     [...],                     // top 3 discourse.act.* + discourse.answer + backchannel + assistant.*
     "safety":  [...],                     // safety.* with p >= thr / 2 (top 6); "on" = above threshold
     "emotion": [...], "arousal": [...],   // affect.emotion.* / affect.arousal.* (top 3)
     "role":    [...],                     // speaker.role.* (only when the words carry speaker ids)
     "item":    [...],                     // item.* (session-level topic / title)
     "state":   {"content.mode": [...], "content.meeting_phase": [...], ...}}
  ]
}
```

Each read-out entry is `{"tag", "p", "thr", "on"}`, where `on = p >= thr` and `thr` is the display threshold for the
current domain. Scores are rounded to 6 decimals.

`arctag_poll`:

```jsonc
{"n_words": 120, "a0": 24,           // spans are re-decoded over the last 96 words: drop your spans with end > a0, add these
 "spans": [...], "events": [...],    // events: only new ones since the last poll
 "utterances": [...],                // utterances closed (endpoint) since the last poll, with "final": true
 "current": {"start": 110, "end": 120, "final": false, "intent": [...], ...}}   // running read-out of the open utterance, or null
```

Revisions: `{"from": k, "words": [...]}` replaces `words[k:]`. Use it when the ASR changes an interim hypothesis.
Events that were already returned are not taken back.

## Streaming

`arctag_poll` does not re-run a window. The model state stays between polls, each new word costs one recurrent step, and a poll
only runs the words it has not seen.

**`stream_ctx` is an attention window, not a re-run window.** This is a change from the ONNX Runtime implementation, where it was
the number of words re-run by each poll, with the model state restarted at the start of them.

- The global attention layers and the mention memory look at the newest `stream_ctx` words. The local attention always looks
  at 32. The recurrent trunk state and the pooled utterance, speaker and item read-outs carry the whole session.
- Up to `stream_ctx` words, streaming and `arctag_tag_text` give the same result (test: 0 mismatches at a tolerance of 1e-3).
- Past it they differ, because `tag_text` attends to all the words of its pass (up to `max_ctx`, 2048). Measured on the newest
  96 rows of a 1024-word text with a window of 256: up to 0.11 on a tag, mean 8.6e-4. The old re-run differed from full context by
  up to 0.31 at 512 words, so this is closer as well as cheaper. A larger `stream_ctx` costs memory (about 2 MB per 256 words), and
  time per word grows slowly with it.
- Changing `stream_ctx` mid-session replays the held words with the new window. Events already returned are not repeated.

**Revisions.** Replacing words re-runs from the nearest snapshot, about every 16 words. Within the last 48 words this is exact
and costs about 1 ms per word re-run. Deeper revisions replay all held words from a fresh state, which takes up to a few hundred ms.
That is exact while the session is still fully held, and approximate after it.

**Long sessions.** Word indices count the whole session and never reset, but only a bounded part of it is kept, so memory plateaus:

- Held: the newest `stream_ctx` + 272 to 336 words (528-592 by default), and 208 to 272 rows of model output. Tested at 250,000 words:
  flat memory and cost per word.
- `from` older than the oldest held word: `arctag_push_words` returns -1 with a message and changes nothing.
- An utterance longer than the held words reports its true `start`, but its `text` holds only the held part. Its read-outs still cover
  all of it.
- More than 65,536 words waiting for `arctag_poll`: `arctag_push_words` returns -1 and changes nothing. A large backlog below that
  limit is fine: a poll works through it in chunks, reporting all its events and utterances, using no more row memory than usual.
- A bad word anywhere in a push makes the whole push fail and changes nothing.

**Speaker labels (the `spksrc` input).** The ONNX graph has a "this session has speaker ids" flag on every word, and the reference
sets it for all the words of a pass if any word has a label. In a stream that would make earlier scores depend on later words, and the
first label would force a replay. So here:

- In a stream the flag turns on at the first labelled word and stays on, without a replay. A word's scores never depend on later
  words, and what was reported for earlier words stands. A revision that removes the first label turns it back off from there.
- `arctag_tag_text` and `arctag_tag_words` keep the graph's definition: all the words of a pass.
- With labels from the first word, the most common case, the two agree exactly. If labels first appear mid-session they differ a
  little. In one test (first label at word 60 of 120), all 26 spans had the same type and range, and the largest score difference was 0.052.

## Tag families (ontology r8 + contact-centre roles)

- `entity.mention.{person,team,organization,place,work_product,event_name,legal_ref,game_entity,game_map,term}`
- `entity.number.{date,time_of_day,duration,clock,money,percent,count,ordinal,score,stat,measure,ratio_odds,field_position,jersey_number,ref_number,id_pii}`
  and `entity.numeral` (any number, overlaps the subtypes)
- `safety.pii.{private_name,phone_number,email,street_address,card_account_number,gov_id,url_handle,date_of_birth,public_contact}`,
  `safety.profanity.{mild,strong}`, `safety.toxicity.*`
- `discourse.act.*`, `discourse.intent.*`, `intent.*`, `safety.*` (utterance level), `affect.emotion.*`,
  `affect.arousal.*`, `speaker.role.*`, `item.*`, `content.*`, `event.*`

`arctag_meta.json` (`readouts`) lists every column.

## Python reference

`python/arctag_ref.py` is the reference implementation, written with numpy, onnxruntime and tokenizers. It produces
the same features, uses the same graph and decodes the same way:

```bash
pip install numpy onnxruntime tokenizers
python python/arctag_ref.py model "some text" [domain]
python tests/make_expected.py model        # regenerate tests/expected/*.json
```

## Parity

| check | result |
|---|---|
| Native forward pass vs ONNX Runtime on the matching graph, every output, 31 texts up to 1024 words, 1,056,299 values | fp32: max 4.8e-5. int8: max 4.9e-5. Verifier logits: max 6.0e-5 and 6.8e-5. Same with the built-in matmul, plain C and AVX2 kernels (max 4.9e-5 and 7.1e-5) |
| ONNX graph vs the PyTorch streaming engine (`fast_arc.FastArc`, which the demo uses), all output probabilities, 5 made-up texts | max abs diff 2.3e-5 (1.2e-6 on the speaker read-out with speaker ids) |
| ONNX graph vs the PyTorch re-implementation it was exported from, 1200-word input | max abs diff 1.6e-4 |
| Python reference decoder vs the demo engine (`ArcSession`, T0), 5 texts x 3 domains | spans, events and read-out tags identical |
| C tokenizer vs HF `tokenizers`, 37 strings (punctuation, numbers, emoji, CJK, Cyrillic, Arabic digits, special tokens) | identical ids |
| C library vs Python reference, 5 texts x 5 domains (154 spans), fp32 | spans, events and read-outs identical; max score diff 5.0e-5 |
| Word-by-word and batched streaming vs `arctag_tag_words`, inside the window, plain, timed and with speakers | identical (tolerance 1e-3) |
| Revisions at depths 1 to 240, undone, vs a stream that never saw them | identical (tolerance 1e-3) |

`./scripts/test_native.sh MODEL_DIR` runs all of these with the tests below. It takes about 10 minutes (`--quick`: about 2).

| test | what it does |
|---|---|
| `tools/export_model.py` + `tests/test_native_parity.c` | tensor parity against the matching ONNX, fp32 and int8; rewinds and one-word steps |
| `tests/test_export.py` | a new checkpoint exports; a changed width, a new parameter, a computed constant, a missing operator, or changed arithmetic each fail loudly |
| `tests/test_parity.c`, `tests/test_tokenizer.c` | public API vs the Python reference |
| `tests/test_stream.c` | invariance, poll cadence, revisions, revisions too old, late speaker labels, options |
| `tests/test_threads.c` under TSan | handles on many threads, a shared model, create/destroy races: same bytes as one thread |
| `tests/test_corrupt.c` under ASan + UBSan | truncated, padded or bit-flipped files; hostile table entries and index values with valid checksums (whatever still loads must run); mutated meta and tokenizer files |
| `tests/fuzz_inputs.c` under ASan + UBSan | mutated text and JSON through every call, in random order. Output must always be valid JSON |
| `tests/test_oom.c` under ASan + UBSan | every allocation in every call made to fail once |
| `tests/test_soak.c` | 200,000 words with revisions: memory and cost per word must stay flat |

`tests/perf/` has the benchmark (`bench.c`, which also links against `ort/`) and a cold-load measure (`cold_load.c`).

## Performance

Measured on an M3 Pro with `tests/perf/bench.c`, one thread, one measurement per process, medians over 2 or 3 rounds. The machine was busy
(load average 4.6 to 8.1), and all columns were measured in rotation under that load. Compare them with each other rather than taking them as best case.

| measure | ONNX Runtime, fp32 | native, int8 | native, fp32 | native, int8, no Accelerate |
|---|---|---|---|---|
| Whole text at 8 / 32 / 150 / 512 words, ms | 2.2 / 7.3 / 37 / 183 | 1.3 / 2.4 / 14 / 57 | 1.1 / 2.6 / 13 / 55 | 3.5 / 11 / 63 / 179 |
| Added memory at 150 / 512 words, MB resident | 137 / 193 | 14.6 / 17.1 | 32.9 / 35.3 | 14.4 / 16.8 |
| of which dirty (the rest is the weights file, which the system can drop) | 128 / 185 | 7.6 / 10.1 | 7.5 / 9.8 | 7.4 / 9.9 |
| Streaming, per new word at 8 / 256 words, ms | 2.3 / 61 | 0.55 / 0.54 | 1.0 / 0.67 | 0.57 / 0.56 |
| Streaming, per new word at 2047 words, ms | not run | 0.56 | 0.60 | 0.53 |
| Revising the last 3 words, ms | 57-68 (at 400 words) | 1.8-1.9 | 1.6-2.1 | 5.6-7.0 |
| Load, warm file cache, ms | 243-276 | 7-11 | 12-24 | 7-8 |

Load with the weights out of the file cache (`tests/perf/cold_load.c`; `mincore` confirms 0% resident first): 8.6-9.8 ms for int8, then
1.8-2.9 ms for the first short text. Loading reads the whole weights file once to check its checksum, through the file cache, not into
the process. A 250,000-word stream with a revision every 10 words stayed at 10.2 MB dirty and 20.7 MB resident, with cost per word flat
at about 0.7-0.85 ms.

An x86_64 build under Rosetta (AVX2 advertised) ran at 6.3-7.1 / 19-21 / 99-180 / 366-383 ms and 1.9 ms per new word. That is translated code
with Accelerate's x86_64 slice, so it says nothing about a real Intel Mac, which has not been measured.

## Using it in an app

The native sources are a small, self-contained set. Vendor `include/arctag.h`, all of `src/`, and `third_party/cjson/`:

| file | what |
|---|---|
| `src/arctag.c` | public API: features, stream, decoding, JSON |
| `src/atn.c`, `src/atn.h` | forward pass and weights loader |
| `src/tokenizer.c`, `src/tokenizer.h`, `src/unicode_tables.h` | BPE tokenizer |
| `third_party/cjson/cJSON.c`, `cJSON.h` | JSON (the library sets cJSON's allocation hooks once, so check before sharing this copy with other code) |

Header search paths: `include`, `src`, `third_party/cjson`. Compile with `-O2`; debug builds are far slower. On Apple platforms link
`-framework Accelerate`. No other flags: AVX2 on x86_64 is enabled by a pragma around the kernels, so the same flags build arm64,
x86_64 and universal. Nothing needs `onnxruntime`, and a model directory needs only `arctag.atw`, `arctag_meta.json` and
`tokenizer.json`. The header is unchanged, so a caller only relinks and ships the new weights file.

x86_64 needs AVX2 and FMA, checked when the model loads: without them `arctag_create` returns an error. All Intel Macs that run
current macOS have them. Under Rosetta they are only advertised if the process starts with `ROSETTA_ADVERTISE_AVX=1`, so a universal
app forced into Rosetta gets that error unless it sets this.

## How it works

`text -> words -> utterances -> per-word features -> forward pass (src/atn.c) -> decode`

- **Features** (`src/arctag.c: word_feat`, `word_input`): up to 8 BPE pieces per word (`" " + word`; the first 7 and the last), digit
  flags, buckets for gap, duration and confidence, final/interim, speaker slot and change, ASR source, and utterance
  starts.
- **Model**: word encoder (2 local-attention layers, 32-word window), then a trunk of 6 Mamba-3 blocks. The SSM
  recurrence is unrolled in closed form in the ONNX graph, so it has no loops; `src/atn.c` runs it as the recurrence it is. Global attention after blocks 1 and 3, then a
  domain adapter. Read-outs: MemRead (earlier-mention memory), word read-outs, the span-begin read-out, a 5-family
  span head, and pooled utterance, speaker and item read-outs.
- **Decoding** (the demo engine's T0 path):
  - Span-head families are decoded greedily as BIO on `1 - P(O) >= thr`. Each span gets one type: the argmax of the
    summed log type-probabilities over the span.
  - Other span tags use runs of `span_in >= thr`, split at `span_beg` and at utterance starts. Overlapping spans of
    one family are merged.
  - Events fire at `p >= thr` and are de-duplicated within 3 words.
  - Read-outs come from the pooled utterance state at the utterance's last word.
- **Thresholds**: the T0 display refit for step 88000 (precision-first, fitted on the cleaned dev set), plus the events
  refit, with step 84000 as the fallback. On top come the demo's display clamps: spans >= .3, toxicity >= .5 outside
  the safety domains, events >= .3, utterance safety >= .4.

## Limits

- `tag_text` is offline over at most `max_ctx` (2048) words per pass. Longer texts are split into independent chunks.
- Weights are memory-mapped: changing or truncating `arctag.atw` while a handle is open can crash the process, as with any
  mapped file. Replace it by renaming a new file over it. Damage before loading is caught by the checks at load.
- Not included: the session-KB pointer (custom name lists), the T1 and T2 delayed tiers, and the demo's automatic
  domain detection.
- English only.

## Layout

```
include/arctag.h      public API
src/arctag.c          features, stream, decoding, JSON
src/atn.c             forward pass and weights loader (+ src/atn.h)
src/tokenizer.c       byte-level BPE reading tokenizer.json (+ src/unicode_tables.h from tools/gen_unicode.py)
ort/arctag_ort.c      the ONNX Runtime implementation of the same API, for comparison (cmake -DARCTAG_BUILD_ORT=ON)
examples/             arctag_cli.c, arctag_server.c
tests/                native tests (see Parity), expected/ (made-up texts only), make_expected.py, perf/
tools/                export_model.py, make_refs.py, quantize_model.py, gen_unicode.py
python/arctag_ref.py  Python reference (uses ONNX Runtime)
scripts/              test_native.sh, fetch_model.sh, download_onnxruntime.sh/.ps1 (only for ort/)
third_party/cjson     cJSON 1.7.19 (MIT), with its parse-error position made per thread (one marked line)
```

## Licence

Code (including this inference code and the native runtime) and model weights, in any form including an exported `arctag.atw`:
CC BY-NC 4.0 plus the Whissle terms in [LICENSE](LICENSE), non-commercial use only. Commercial use needs a separate written licence
from Whissle Inc.: ksingla@whissle.ai. Third-party components keep their own licences: see [NOTICE](NOTICE). The native library
contains only cJSON from third parties and links Apple's Accelerate on Apple platforms. ONNX Runtime is used only by `ort/`, the export
tool and the Python reference, and is not part of a native build.
