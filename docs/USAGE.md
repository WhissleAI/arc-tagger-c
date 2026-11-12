# Using arctag

Commands, configs and the output JSON of the C library. The API is in [`include/arctag.h`](../include/arctag.h), and the
int8 variant is covered in [int8.md](int8.md).

## Build and model

```sh
./scripts/download_onnxruntime.sh                  # prebuilt ONNX Runtime 1.20.1 -> third_party/onnxruntime
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
# or, native backend only (no ONNX Runtime needed):
# cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DARCTAG_WITH_ORT=OFF && cmake --build build -j
HF_TOKEN=hf_... ./scripts/fetch_model.sh           # -> model/ (arctag.onnx, arctag_int8.onnx, arctag_meta.json, tokenizer.json)
ctest --test-dir build --output-on-failure
```

`fetch_model.sh` reads these environment variables:

| variable | default | meaning |
|---|---|---|
| `HF_REPO` | `WhissleAI/arc-tagger` | Hugging Face repo |
| `HF_REVISION` | `v0.3.0` | tag, branch or commit |
| `ARCTAG_FETCH` | `all` | `fp32`, `int8` or `all` |
| `ARCTAG_INT8_URL` | | where to get `arctag_int8.onnx` if the HF repo does not have it |

All files are checked against `SHA256SUMS`.

A model directory holds:

- `arctag.onnx` (fp32);
- `arctag_int8.onnx` (int8, optional);
- `arctag_meta.json` (ontology read-outs, per-domain thresholds, decoding constants);
- `tokenizer.json`.

## Backends

Two backends run the model from the same files:

- `native` (default): plain C, one thread per handle. Streaming polls compute only the words added or revised since the
  last poll, so a per-word poll costs about 1-3 ms. Loads in 40-70 ms.
- `onnxruntime`: ONNX Runtime 1.20.1, `threads` intra-op threads. Bit-equal to the Python reference. Each poll re-runs
  the last `stream_ctx` words (about 100 ms per poll at 256).

Both make the same decisions; scores differ by at most a few 1e-4. Choose one with:

| where | how |
|---|---|
| C, at load | `arctag_options o; arctag_options_init(&o); o.backend = ARCTAG_BACKEND_ONNXRUNTIME;` then `arctag_create_ex` |
| C, later | `arctag_set_option(h, "backend", "onnxruntime")` (reloads the model; the stream is kept) |
| config | `"backend": "native"` or `"onnxruntime"` |
| tools and server | `--backend native` / `--backend onnxruntime` |
| environment | `ARCTAG_BACKEND=onnxruntime`, used when the options leave the backend at its default |

`arctag_info` reports `backend`, `weights` (`fp32`, `int8`, or `int8-folded` when the native backend converted a
dynamically quantised file to float at load) and, for the native backend, `kernels` (`avx2`, `neon` or `scalar`) and
`stream_state`.

The native backend checks the model file when it loads it (every table's shape and every index range) and refuses a
truncated or corrupted file with a message, without crashing. AVX2 kernels are used only when the CPU has them.

## Command-line tools

All tools take `-m MODEL_DIR` (default `$ARCTAG_MODEL_DIR` or `model`), `-c CONFIG.json`, `--int8` / `--fp32` and
`--backend native|onnxruntime`.
Options given on the command line are applied after the config.

| tool | what it does | example |
|---|---|---|
| `arctag_cli` | tags one text or word array and prints the JSON | `arctag_cli -d meeting "Let's ship it Friday. Dana owns the release notes."` |
| `arctag_cli --stream` | feeds words one at a time through push / poll and prints every poll | `echo "goal for Arsenal!" \| arctag_cli -d soccer --stream` |
| `batch_tag` | one document per line in, one JSON result per line out (JSON Lines), plus ms/word on stderr | `batch_tag -c configs/news.json examples/data/news.txt > out.jsonl` |
| `stream_asr` | drives the streaming API from an ASR message stream (interim / final / endpoint) | `stream_asr -c configs/contact_center.json examples/data/asr_stream.jsonl` |
| `redact` | PII / profanity redaction, line by line | `redact --policy phone=mask,email=mask,name=mask examples/data/pii.txt` |
| `vertical_run` | runs one vertical config over an input and prints a report | `vertical_run configs/legal.json examples/data/court.txt` |
| `compare_precision` | fp32 and int8 side by side: load time, ms/word, items that differ | `compare_precision examples/data/docs.txt` |
| `bench` | load time, streaming per-word latency p50 / p95, batch ms/word, peak RSS (`--model-file F` for another .onnx, `--stream-state on\|off`) | `bench --backend native --int8 -t 1 examples/data/*.txt` |
| `arctag_server` | JSON over stdin/stdout or HTTP, one stream per server | `arctag_server -c configs/voice_assistant.json --port 8080` |

`arctag_cli` flags:

- `-d DOMAIN`, `-a ASR`, `-t THREADS`;
- `--redact POLICY`, `--emit SECTIONS`, `--tags PREFIXES`;
- `--thr PATTERN=VALUE` (repeatable);
- `--words` (the input is a JSON word array);
- `--info` (print the handle description).

`arctag_server` endpoints:

| endpoint | body / response |
|---|---|
| `POST /tag` | raw text, or `{"text", "domain"}` |
| `POST /push` | push-words JSON |
| `GET` or `POST /poll` | the poll result |
| `POST /reset` | resets the stream |
| `POST /config` | a config object |
| `GET /info` | the handle description |
| `GET /health` | health check |

With `--stdin` the server reads one JSON request per line (`{"op": "tag" | "tag_words" | "push" | "poll" | "reset" |
"set" | "config" | "info", ...}`).

## C API in five calls

```c
arctag_options o; arctag_options_init(&o);
o.precision = ARCTAG_INT8;                                      /* or leave the defaults: fp32, native backend */
arctag_t *h = arctag_create_ex("model", &o, err, sizeof err);    /* or arctag_create_from_config("model", "configs/legal.json", ...) */
arctag_set_option(h, "domain", "meeting");
char *json = arctag_tag_text(h, "Dana will send the notes by Friday.");
/* ... */ arctag_free_str(json); arctag_destroy(h);
```

Streaming looks like this:

```c
arctag_push_words(h, "[{\"w\":\"call\",\"t0\":0.1,\"t1\":0.3,\"spk\":\"A\"}, \"me\"]");      /* append */
arctag_push_words(h, "{\"words\":[\"tomorrow.\"],\"from\":2,\"endpoint\":true}");        /* revise words[2:], close the utterance */
char *p = arctag_poll(h);   /* spans with end > a0 replace yours; new events; utterances closed since the last poll */
```

A handle is not thread-safe. Use one handle per stream or thread. Handles share nothing, so for N streams load N handles.
A native handle with the weight-only int8 file costs about 26 MB, with fp32 about 72 MB.

## Configs

A config is a JSON object. A file under `configs/` can be loaded with `arctag_create_from_config`, `-c` or
`POST /config`, and a JSON string can be applied with `arctag_apply_config`. Unknown keys are an error. Keys starting
with `_`, and `name`, `use_case`, `description`, `vertical`, `notes`, are ignored.

| key | value |
|---|---|
| `model_dir` | model directory relative to the config file (used when `model_dir` is NULL) |
| `backend` | `"native"` (default) / `"onnxruntime"` |
| `precision` | `"fp32"` / `"int8"` |
| `threads` | ONNX Runtime intra-op threads (0 = ORT default); the native backend uses one thread |
| `mode` | `"batch"` / `"streaming"`: informational for the library, read by `vertical_run` to pick the API |
| `domain` | a profile (`auto talk meeting court assistant sports news`), a fine domain (`nba nhl nfl soccer cricket court_appellate court_trial assistant_noisy stream_chat`) or `"ui:fine"` |
| `asr` | where the words come from (`unknown human stock_rnnt arc_transcriber nemotron35 deepgram final_only web`) |
| `fine_auto` | `true` (default): inside `sports` / `court`, switch the fine domain from lexicon cues the way the Python package does |
| `max_ctx`, `stream_ctx` | words per model pass for `tag_text` (2048), and words of context the stream keeps (256; ONNX Runtime re-runs them per poll) |
| `stream_state` | native backend: `true` (default) carries the state from poll to poll; `false` re-runs the window on every poll (same output, slower) |
| `emit` | output sections to keep: `entity number pii profanity toxicity events intent act safety emotion arousal valence intensity role item state` (`[]` = all) |
| `tags` | tag prefixes to keep (`[]` = all) |
| `thresholds` | `{PATTERN: value}` overrides, applied on top of the domain's thresholds (see below); replaces earlier overrides |
| `redact` | `"off"`, `"mask"` (all PII categories except public `contact`, plus profanity / toxicity) or `{"mode": "mask", "policy": {CATEGORY: "mask" / "tag" / "off"}}` |

Threshold patterns can take these forms:

- an exact tag, such as `safety.pii.request`;
- a prefix ending in `*`, such as `event.sports.*`;
- any of the above with `@readout` added to limit it to one read-out (`frame span_beg span_in span_end ev_at ev_imm
  ev_rec utt speaker item`), such as `discourse.intent.*@utt`;
- `head:FAMILY`, the score threshold of a span-head family (`entity.number entity.mention safety.pii safety.profanity
  safety.toxicity`).

A family wildcard such as `safety.pii.*` also sets that family's span-head threshold. A value above 1 turns the tag off,
and a value of 0 makes it always on.

Redaction categories:

| category | covers |
|---|---|
| `phone` | `safety.pii.phone_number` |
| `email` | `safety.pii.email` |
| `address` | `safety.pii.street_address` |
| `card` | `safety.pii.card_account_number` |
| `gov_id` | `safety.pii.gov_id` |
| `dob` | `safety.pii.date_of_birth` |
| `name` | `safety.pii.private_name` |
| `handle` | `safety.pii.url_handle` |
| `contact` | `safety.pii.public_contact` |
| `profanity` | `safety.profanity.*` |
| `toxicity` | `safety.toxicity.*` |

A masked span becomes `[PHONE]`, `[EMAIL]`, `[ADDRESS]`, `[CARD]`, `[GOV_ID]`, `[DOB]`, `[NAME]`, `[HANDLE]`,
`[CONTACT]`, `[PROFANITY]` or `[TOXIC]`.

### Shipped configs

| config | vertical | precision / threads | mode | domain | emits | overrides | redaction |
|---|---|---|---|---|---|---|---|
| `voice_assistant.json` | voice assistant | fp32 / 2 | streaming, ctx 128 | assistant | entity number pii events intent act safety | `safety.jailbreak.attempt` 0.45 | off |
| `contact_center.json` | contact center | fp32 / 2 | streaming, ctx 256 | talk | pii entity number profanity toxicity events intent act safety emotion valence role | `safety.scam.*` 0.4, `safety.pii.request` 0.45, `safety.threat.escalation` 0.4 | mask PII + profanity, tag contacts / toxicity |
| `trust_safety_gaming.json` | trust and safety, gaming chat | fp32 / 2 | streaming, ctx 128 | stream_chat | profanity toxicity pii entity safety intent emotion arousal valence intensity | `safety.toxicity.threat_violence` 0.4, `safety.threat.escalation` 0.35, `safety.child.*` 0.3, `safety.crisis.self_harm` 0.3, `safety.solicitation.dating_romantic` 0.4 | mask profanity, toxicity, private PII |
| `legal.json` | legal / court | fp32 / 4 | batch | court_trial (fine_auto may switch to appellate) | entity number pii events intent act role item state | `event.legal.*` 0.4, `entity.mention.legal_ref` 0.5 | off |
| `meetings.json` | meetings | fp32 / 4 | batch | meeting | entity number pii events intent act role state | `event.meeting.action_item.recent` 0.35, `event.meeting.decision.recent` 0.4, `intent.assign_task` 0.4 | off |
| `sports.json` | sports commentary | fp32 / 2 | streaming, ctx 256 | sports (fine_auto: nba nhl nfl soccer cricket) | entity number profanity toxicity events intent state | `event.sports.*` 0.5 | mask profanity |
| `news.json` | news / talk | fp32 / 2 | batch | news | entity number profanity toxicity events intent role state | none | off |
| `pii_redaction.json` | generic PII redaction | fp32 / 4 | batch | auto | pii safety | none (PII head already at the 0.3 floor) | mask (default policy) |
| `edge_low_latency.json` | low-latency edge | **int8 / 1** | streaming, ctx 96 | auto | entity number pii profanity toxicity events intent act safety | none | off |

Each config's `_notes` explains its choices. Copy one and edit it for your own use: change the domain, drop sections you
don't need with `emit`, and move a threshold down for recall or up for precision.

## Output JSON

`arctag_tag_text` / `arctag_tag_words` return:

```jsonc
{
  "profile": "meeting:meeting",          // ui:fine profile used (fine_auto may switch it inside sports / court)
  "n_words": 13,
  "words": ["Call", "me", ...],
  "spans": [{
     "type": "safety.pii.phone_number",  // ontology tag
     "family": "pii",                    // entity | number | pii | profanity | toxicity
     "start": 3, "end": 6,               // word range [start, end)
     "text": "555 123 4567",
     "score": 0.999998, "thr": 0.6,      // decision score and the threshold it passed
     "src": "head",                      // head = span-head decode; span_in = per-word run decode
     "alt": {"tag": "safety.pii.gov_id", "p": 3e-06},   // runner-up type (head spans)
     "ver": 0.69                         // span verifier probability
  }],
  "events": [{"type": "event.meeting.action_item.recent", "q": "ev_rec", "word": 9, "score": 0.81, "thr": 0.35, "text": "Friday."}],
  "utterances": [{
     "start": 0, "end": 7, "text": "...",
     "intent":    [{"tag": "intent.request_action", "p": 0.88, "thr": 0.5, "on": true}, ...],
     "act":       [...],                 // discourse.act.*
     "safety":    [...],                 // safety.* utterance flags (scam, pii.leak / request, threat, crisis, child, jailbreak, ...)
     "emotion":   [...], "arousal": [...], "valence": [...], "intensity": [...],   // affect groups, best first (top-1 is the read-out)
     "role":      [...],                 // speaker.role.* of the utterance's speaker (needs spk on the words)
     "item":      [...],                 // item.* (case area, esports title)
     "state":     {"content.mode": [...], "content.game_state": [...], ...}       // content state groups at the utterance end
  }],
  "redacted_text": "Call me at [PHONE] tomorrow. ..."   // only when redaction is on
}
```

Read-out lists are sorted best first. `on` means `p >= thr` for the active profile, after any overrides. Utterances end
at a word ending in `.`, `?` or `!`, at a per-word `"eou": true`, or at a streaming endpoint. `emit` and `tags` remove
sections and entries from the output. They never change what is computed.

`arctag_poll` returns:

```jsonc
{"n_words": 42, "a0": 30, "profile": "sports:soccer",
 "spans": [...],          // spans with end > a0: drop yours with end > a0 and append these
 "events": [...],         // events not reported before
 "utterances": [...],     // utterances closed since the last poll
 "current": {...} | null, // running read-out of the open utterance
 "redacted_text": "..."}  // words[a0:] redacted (redaction on)
```

`arctag_info` returns:

```jsonc
{"version": "0.3.0", "precision": "int8", "model_file": "model/arctag_int8.onnx", "profile": "auto:unknown", "asr": "unknown",
 "threads": 1, "max_ctx": 2048, "stream_ctx": 96, "fine_auto": true, "emit": [...], "tags": [], "redact": "off", "thresholds": {}}
```

## Tips

- **Pick a domain.** `auto` uses the generic thresholds. On off-domain text the domain-specific heads (sports events,
  items) can still fire. Either set the domain or use `emit` to drop those sections.
- **Speaker ids** (`"spk"` on words) enable speaker-change features and speaker roles.
- **ASR source.** Set `asr` to the recogniser that produced the words when it is one of the listed sources. The model
  learnt each one's punctuation and casing habits.
- **Streaming cost.** With the native backend a poll costs about one word's compute (more after a revision of earlier
  words), so per-word polling is fine. Under ONNX Runtime each poll re-runs the last `stream_ctx` words: a smaller value
  lowers the per-word cost and shortens the context (`edge_low_latency.json` uses 96), and polling every N words instead
  of every word also helps.

## Upgrading from library 0.1.0

The API is source compatible: every change is an addition. The output changes in these ways:

- every utterance has `valence` and `intensity` lists;
- poll results carry `profile`;
- events are filtered to the package's display tier;
- the 0.1.0 display floors for events (0.3) and utterance safety (0.4) are gone, as in the package;
- the default `asr` is `unknown`;
- `fine_auto` is on by default, so `profile` can change inside `sports` / `court`.

0.1.0 model files do not load: use the v0.3.0 `onnx/` files.
