# arctag: C API for the Arc Tagger

A small C library (plus a CLI and a tiny JSON/HTTP server) around **Arc Tagger**: a ~9.8M-parameter, text-only
streaming tagger. Give it words from any ASR or plain text and it returns:

- **spans**: entities (person, organization, place, team, product, ...), numbers (date, time, money, count, score,
  phone/ID digits, ...), PII (name, phone, email, address, card, gov id, DOB, ...), profanity and toxicity.
- **events**: per-word events such as turn yield, interruption, command complete, legal objection, meeting decision or
  action item, and sports scores.
- **utterance read-outs**: intent, dialogue act, safety (scam, PII leak, toxicity, ...), emotion and arousal, speaker
  role (needs speaker ids), item/topic, and content state (mode, phase, game state).

The model is a single ONNX graph (`arctag.onnx`, fp32, 34 MB) that runs on CPU with ONNX Runtime. Decoding (thresholds,
the span head, events and read-outs) follows the zero-lag decoding of the Arc Tagger package, and the output
matches it exactly (see [Parity](#parity)).

## Quickstart

```bash
git clone git@github.com:WhissleAI/arc-tagger-c.git && cd arc-tagger-c
./scripts/download_onnxruntime.sh              # prebuilt ONNX Runtime 1.20.1 -> third_party/onnxruntime
export HF_TOKEN=hf_xxx                         # read token with access to the private model repo
./scripts/fetch_model.sh                       # -> ./model (arctag.onnx, arctag_meta.json, tokenizer.json)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
ctest --test-dir build --output-on-failure     # tokenizer + C-vs-Python parity

./build/arctag_cli -m model "Hi, this is Maria Lopez. My number is 555 867 5309."
./build/arctag_cli -m model -d meeting "Priya will send the final numbers by Friday."
./build/arctag_cli -m model --stream "words arrive one at a time. like this!"     # push/poll demo
./build/arctag_server -m model --port 8090     # HTTP: POST /tag, POST /push, GET /poll, POST /reset, GET /health
```

```bash
curl -s localhost:8090/tag -d 'Can you call me back at 415 555 0199 tomorrow?'
curl -s localhost:8090/tag -d '{"text": "The Lakers are up 98 to 95.", "domain": "nba"}'
curl -s localhost:8090/push -d '{"words": ["hello", "my", "name", "is", "John", "Smith."], "endpoint": true}'
curl -s localhost:8090/poll
```

**Windows:** run `scripts/download_onnxruntime.ps1` in PowerShell, then use the same CMake commands (MSVC). The build
copies `onnxruntime.dll` next to the executables. For the model, run `scripts/fetch_model.sh` from Git Bash, or download
the four files under `onnx/` in the Hugging Face repo by hand.

**macOS:** the download script picks `osx-arm64` or `osx-x86_64` (or pass one explicitly, e.g.
`./scripts/download_onnxruntime.sh osx-universal2`).

## Model files and the Hugging Face token

The model lives in the **private** Hugging Face repo [`ksingla025/arc-tagger`](https://huggingface.co/ksingla025/arc-tagger).
The C library only needs the `onnx/` folder:

| file | what |
|---|---|
| `onnx/arctag.onnx` | full-context T0 forward (opset 17, dynamic word axis, up to 2048 words per pass), probabilities out |
| `onnx/arctag_meta.json` | read-out columns, per-domain validity and display thresholds, span-head families, feature buckets |
| `onnx/tokenizer.json` | byte-level BPE (8k) tokenizer, which the C tokenizer reads directly |
| `onnx/SHA256SUMS` | checksums verified by `fetch_model.sh` |

1. Get read access to `ksingla025/arc-tagger` (ask the owner).
2. Create a **read** token at <https://huggingface.co/settings/tokens>.
3. `export HF_TOKEN=hf_...` and run `./scripts/fetch_model.sh [DEST]`. `HF_REPO` and `HF_REVISION` override the repo and
   revision (default `main`; `HF_REVISION=2ec982888e1b51f651db06c23afdc2450668ccab` pins the first `onnx/` upload).

Without the script:
`huggingface-cli download ksingla025/arc-tagger --include "onnx/*" --local-dir . && mv onnx model`.

The rest of that repo is the Python package (`arctag/`, `example.py`). This C repo does not need it.

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
| `arctag_t *arctag_create(const char *model_dir, char *err, size_t errlen)` | load `arctag.onnx`, `arctag_meta.json`, `tokenizer.json` |
| `int arctag_set_option(arctag_t*, const char *key, const char *value)` | `domain`, `asr`, `threads`, `max_ctx`, `stream_ctx` (see below) |
| `char *arctag_tag_text(arctag_t*, const char *text)` | tag a whole text (whitespace words; `. ? !` ends an utterance) |
| `char *arctag_tag_words(arctag_t*, const char *words_json)` | same, for pre-split words (`"eou": true` on a word ends an utterance) |
| `int arctag_push_words(arctag_t*, const char *words_json)` | streaming input (append, revise from index `from`, `endpoint`) |
| `char *arctag_poll(arctag_t*)` | run the model on the newest `stream_ctx` words and return the update |
| `void arctag_reset(arctag_t*)` | forget the stream |
| `const char *arctag_last_error(const arctag_t*)` | last error message |
| `void arctag_free_str(char*)`, `void arctag_destroy(arctag_t*)`, `const char *arctag_version(void)` | |

A handle is not thread-safe. Use one handle per thread; each handle holds its own ONNX Runtime session. Memory is about
90 MB RSS for short inputs. Memory grows with the window, so a full 2048-word pass peaks near
1 GB; lower `max_ctx` to reduce it. `ARCTAG_THREADS=n` sets the default number of ONNX Runtime threads.

**Options**

| key | values | default |
|---|---|---|
| `domain` | `auto`, `talk`, `meeting`, `court`, `assistant`, `sports`, `news`, or a fine domain (`nba` `nhl` `nfl` `soccer` `cricket` `court_appellate` `court_trial` `assistant_noisy` `stream_chat`), or a profile name `ui:fine` from `arctag_meta.json` | `auto` |
| `asr` | where the words come from, one of `arctag_meta.json` `asr_sources` (`unknown`, `gold` for human text, `final_only`, `web`, ...) | `default_asr` in the meta file (the setting the thresholds were fitted with) |
| `threads` | ONNX Runtime intra-op threads (0 = ORT decides) | `$ARCTAG_THREADS` or 0 |
| `max_ctx` | words per model pass in `tag_text` / `tag_words`; longer texts are cut into independent chunks | 2048 |
| `stream_ctx` | words of context re-run by each `arctag_poll` | 256 |

The domain conditions the model (and tag validity per domain) and selects the per-domain thresholds. `auto`
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
| Python reference decoder vs the Arc Tagger package, 5 texts x 3 domains | spans, events and read-out tags identical |
| C tokenizer vs HF `tokenizers`, 37 strings (punctuation, numbers, emoji, CJK, Cyrillic, Arabic digits, special tokens) | identical ids |
| C library vs Python reference, 5 texts x 5 domains (154 spans) | spans, events and read-outs identical; max score diff 1.6e-5 |
| C streaming (push per utterance + poll) vs `arctag_tag_text` | identical |

## How it works

`text -> words -> utterances -> per-word features -> arctag.onnx -> decode`

- **Features** (`src/arctag.c: run_model`): up to 8 BPE pieces per word (`" " + word`; the first 7 and the last), digit
  flags, buckets for gap, duration and confidence, final/interim, speaker slot and change, ASR source, and utterance
  starts.
- **Model**: `arctag.onnx`, a loop-free graph that returns per-word, span and pooled utterance, speaker and item
  read-outs for the whole window.
- **Decoding** (the same as the Arc Tagger package):
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
- `arctag_poll` re-runs the whole `stream_ctx` window every time. It does not carry state forward. Rough cost per poll
  on a laptop CPU: 15 ms for 32 words, 70 ms for 150, 450 ms for 512. Poll per ASR update rather than per word, or lower
  `stream_ctx`.
- Not included: the session-KB pointer (custom name lists), the T1 and T2 delayed tiers, and the demo's automatic
  domain detection.
- English only.

## Layout

```
include/arctag.h      public API
src/arctag.c          features, ONNX Runtime session, decoding, JSON
src/tokenizer.c       byte-level BPE reading tokenizer.json (+ src/unicode_tables.h from tools/gen_unicode.py)
examples/             arctag_cli.c, arctag_server.c
tests/                test_tokenizer.c, test_parity.c, expected/ (made-up texts only), make_expected.py
python/arctag_ref.py  Python reference
scripts/              download_onnxruntime.sh/.ps1, fetch_model.sh
third_party/cjson     cJSON 1.7.19 (MIT)
```

## Licence

Code (including this inference code) and model weights: CC BY-NC 4.0 plus the Whissle terms in [LICENSE](LICENSE),
non-commercial use only. Commercial use needs a separate written licence from Whissle Inc.: ksingla@whissle.ai.
Third-party components keep their own licences: see [NOTICE](NOTICE).
