# Using arctag

Tools, the C API, configs and the output JSON. The header is [`include/arctag.h`](../include/arctag.h).

## Build and model

```sh
./scripts/download_onnxruntime.sh        # ONNX Runtime 1.20.1 -> third_party/onnxruntime
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
# native backend only: add -DARCTAG_WITH_ORT=OFF and skip the download
./scripts/fetch_model.sh                 # -> model/
ctest --test-dir build --output-on-failure
```

macOS: the download script picks `osx-arm64` or `osx-x86_64` (or pass one, e.g. `osx-universal2`). Windows: use
`scripts/download_onnxruntime.ps1`; the build copies `onnxruntime.dll` next to the executables.

`fetch_model.sh [DEST]` reads:

| variable | default | meaning |
|---|---|---|
| `HF_REPO` | `WhissleAI/arc-tagger` | Hugging Face repo |
| `HF_REVISION` | `v0.4.1` | tag, branch or commit |
| `HF_TOKEN` | | optional (for a private mirror) |
| `ARCTAG_FETCH` | `all` | `fp32`, `int8` or `all` |
| `ARCTAG_INT8_URL` | | another location for the int8 graph |

If the int8 graph is missing, the script installs the fp32 files and the int8 tests report as skipped.

**Versions.** Library 0.5.0 reads model release v0.4.1 (`fetch_model.sh` default). Files from another release can fail
to load or decode differently, because read-out columns and thresholds live in `arctag_meta.json`.

## Backends

| | native (default) | ONNX Runtime |
|---|---|---|
| dependencies | none (libc, libm) | ONNX Runtime 1.20.1 |
| streaming | each poll computes only new or revised words | each poll re-runs the last `stream_ctx` words |
| threads | one per handle | `threads` intra-op threads |
| output | same decisions as the Python reference; scores within 1e-4 | bit-equal to the Python reference |

Pick one with `arctag_options.backend`, `arctag_set_option(h, "backend", ...)` (reloads the model), `"backend"` in a
config, `--backend native|onnxruntime` on the tools, or `ARCTAG_BACKEND=onnxruntime`. The native backend checks the
model file at load and refuses a truncated or corrupted one with a message. Speed and memory: [int8.md](int8.md#speed-and-memory).

## Command-line tools

All tools take `-m MODEL_DIR` (default `$ARCTAG_MODEL_DIR` or `model`), `-c CONFIG.json`, `--int8` / `--fp32` and
`--backend`. Command-line options apply after the config.

| tool | does | example |
|---|---|---|
| `arctag_cli` | tags one text or word array | `arctag_cli -d meeting "Dana owns the release notes."` |
| `arctag_cli --stream` | feeds words one at a time through push / poll | `echo "goal for Arsenal!" \| arctag_cli -d soccer --stream` |
| `batch_tag` | one document per line in, JSON Lines out | `batch_tag -c configs/news.json examples/data/news.txt` |
| `stream_asr` | drives the streaming API from ASR messages | `stream_asr -c configs/contact_center.json examples/data/asr_stream.jsonl` |
| `redact` | redaction, line by line | `redact --policy phone=mask,email=mask examples/data/pii.txt` |
| `vertical_run` | runs one config over an input and prints a report | `vertical_run configs/legal.json examples/data/court.txt` |
| `compare_precision` | fp32 and int8 side by side | `compare_precision examples/data/docs.txt` |
| `bench` | load time, per-word latency, batch speed, peak RSS | `bench --backend native --int8 -t 1 examples/data/*.txt` |
| `arctag_server` | minimal JSON server, one stream | `arctag_server -c configs/voice_assistant.json --port 8090` |

`arctag_cli` flags: `-d DOMAIN`, `-a ASR`, `-t THREADS`, `--redact POLICY`, `--emit SECTIONS`, `--tags PREFIXES`,
`--thr PATTERN=VALUE` (repeatable), `--words` (input is a JSON word array), `--info`.

`arctag_server` routes: `POST /tag`, `POST /push`, `GET|POST /poll`, `POST /reset`, `POST /config`, `GET /info`,
`GET /health`; `--stdin` reads one JSON request per line. For production use [deploy/](../deploy/README.md).

## C API

```c
arctag_options o; arctag_options_init(&o);
o.precision = ARCTAG_INT8;                                     /* default fp32, native backend */
arctag_t *h = arctag_create_ex("model", &o, err, sizeof err);   /* or arctag_create_from_config(...) */
arctag_set_option(h, "domain", "meeting");
char *json = arctag_tag_text(h, "Dana will send the notes by Friday.");
arctag_free_str(json);

arctag_push_words(h, "[{\"w\":\"call\",\"t0\":0.1,\"t1\":0.3,\"spk\":\"A\"}, \"me\"]");    /* append */
arctag_push_words(h, "{\"words\":[\"tomorrow.\"],\"from\":2,\"endpoint\":true}");      /* revise words[2:], close */
char *p = arctag_poll(h);   /* spans with end > a0 replace yours; new events; closed utterances */
arctag_free_str(p);
arctag_destroy(h);
```

| function | |
|---|---|
| `arctag_create`, `arctag_create_ex`, `arctag_create_from_config` | load (defaults / options / a config file) |
| `arctag_apply_config`, `arctag_set_option` | apply a JSON config string; set one option |
| `arctag_tag_text`, `arctag_tag_words` | tag a whole text (`. ? !` end an utterance) or pre-split words |
| `arctag_push_words`, `arctag_poll` | streaming: append or revise (`from`), `endpoint`; get the update |
| `arctag_reset`, `arctag_info`, `arctag_last_error`, `arctag_version` | |
| `arctag_free_str`, `arctag_destroy` | every returned string is malloc'd JSON |

A handle is not thread-safe: use one per stream or thread. Native handles cost about 26 MB (int8) or 72 MB (fp32);
ONNX Runtime handles about 118 MB, and a full 2048-word pass peaks near 1 GB (lower `max_ctx`).

**Words** are a string or `{"w", "t0", "t1", "conf", "spk", "final", "eou"}` (only `w` required; `word` / `text`,
`start` / `end`, `confidence`, `speaker` also accepted). Times, confidences and speaker ids improve the read-outs;
speaker roles need `spk`.

## Options and configs

Options go through `arctag_set_option` or the same keys in a config (JSON; files in [configs/](../configs/README.md)).
Unknown keys are an error; keys starting with `_`, and `name`, `use_case`, `description`, `vertical`, `notes`, are
ignored.

| key | values | default |
|---|---|---|
| `domain` | `auto`, `talk`, `meeting`, `court`, `assistant`, `sports`, `news`, `kids`; a fine domain (`nba` `nhl` `nfl` `soccer` `cricket` `court_appellate` `court_trial` `assistant_noisy` `stream_chat`); or `ui:fine` | `auto` |
| `asr` | `unknown`, `human`, `stock_rnnt`, `arc_transcriber`, `nemotron35`, `deepgram`, `final_only`, `web` | `unknown` |
| `backend`, `precision`, `threads` | `native` / `onnxruntime`; `fp32` / `int8`; ONNX Runtime threads (all reload the model) | `native`, `fp32`, 0 |
| `fine_auto` | inside `sports` / `court`, pick the fine domain from lexicon cues | on |
| `emit` | sections: `entity number pii profanity toxicity events intent act safety emotion arousal valence intensity role item state` | all |
| `tags` | tag prefixes to keep | all |
| `redact` | `off`, `mask`, or a policy (below) | `off` |
| `thresholds` / `thr:PATTERN` | threshold overrides (below); `thr_reset` drops them | |
| `max_ctx`, `stream_ctx` | words per `tag_text` pass; words of context a stream keeps | 2048, 256 |
| `stream_state` | native: carry state between polls (`off` re-runs the window: same output, slower) | on |
| `guard`, `command_context` | kids guard layer ([KIDS.md](KIDS.md)) | `auto`, `off` |
| `model_dir`, `mode` | config only: model directory relative to the config; `batch` / `streaming` (read by `vertical_run`) | |

The domain is a model input: it sets which tags are valid and selects thresholds. `auto` runs the generic profile,
where sports events or items can fire on unrelated text. Set the domain when you know it, or drop sections with `emit`.

**Threshold patterns**: an exact tag (`safety.pii.request`), a prefix (`event.sports.*`), either with `@readout`
(`discourse.intent.*@utt`), or `head:FAMILY` for a span-head family. A value above 1 turns a tag off; 0 makes it
always on.

**Redaction**: `"mask"` masks all PII except public contacts, plus profanity and toxicity. A policy sets each category
to `mask`, `tag` or `off`, e.g. `phone=mask,name=mask,contact=off` or `{"mode": "mask", "policy": {...}}`.

| category | tag | mask |
|---|---|---|
| `phone`, `email`, `address` | `safety.pii.phone_number`, `.email`, `.street_address` | `[PHONE]`, `[EMAIL]`, `[ADDRESS]` |
| `card`, `gov_id`, `dob` | `safety.pii.card_account_number`, `.gov_id`, `.date_of_birth` | `[CARD]`, `[GOV_ID]`, `[DOB]` |
| `name`, `handle`, `contact` | `safety.pii.private_name`, `.url_handle`, `.public_contact` | `[NAME]`, `[HANDLE]`, `[CONTACT]` |
| `profanity`, `toxicity` | `safety.profanity.*`, `safety.toxicity.*` | `[PROFANITY]`, `[TOXIC]` |

## Output JSON

`arctag_tag_text` / `arctag_tag_words` return:

```jsonc
{
  "profile": "meeting:meeting",          // profile used (fine_auto may switch it inside sports / court)
  "n_words": 13,
  "words": ["Call", "me", ...],
  "spans": [{
     "type": "safety.pii.phone_number", "family": "pii",   // family: entity | number | pii | profanity | toxicity
     "start": 3, "end": 6, "text": "555 123 4567",         // word range [start, end)
     "score": 0.999998, "thr": 0.6,
     "src": "head",                      // head = span-head decode; span_in = per-word run decode
     "alt": {"tag": "safety.pii.gov_id", "p": 3e-06},       // runner-up type
     "ver": 0.69                         // span verifier probability
  }],
  "events": [{"type": "event.meeting.action_item.recent", "q": "ev_rec", "word": 9, "score": 0.81, "thr": 0.35, "text": "Friday."}],
  "utterances": [{
     "start": 0, "end": 7, "text": "...",
     "intent": [{"tag": "intent.request_action", "p": 0.88, "thr": 0.5, "on": true}, ...],
     "act": [...], "safety": [...],
     "emotion": [...], "arousal": [...], "valence": [...], "intensity": [...],   // top-1 is the read-out
     "role": [...], "item": [...],
     "state": {"content.mode": [...], "content.game_state": [...], ...}
  }],
  "redacted_text": "Call me at [PHONE] tomorrow. ..."    // only when redaction is on
}
```

Read-out lists are sorted best first; `on` means `p >= thr` for the active profile and overrides. Utterances end at
`.`, `?` or `!`, at `"eou": true`, or at a streaming endpoint. `emit` and `tags` only remove output; they never change
what is computed.

`arctag_poll` returns `{"n_words", "a0", "profile", "spans", "events", "utterances", "current", "redacted_text"}`:
drop your spans with `end > a0` and append these; `events` are new ones; `utterances` closed since the last poll;
`current` is the open utterance's running read-out.

`arctag_info` returns the handle's settings: `version`, `precision`, `model_file`, `profile`, `backend`, `weights`,
`kernels` (native: `avx2`, `neon` or `scalar`), `threads`, `max_ctx`, `stream_ctx`, `emit`, `redact`, `thresholds`, ...

## Tips

- **Pick a domain**, or use `emit` to drop sections that do not apply.
- **Speaker ids** (`spk`) enable speaker-change features and roles.
- **ASR source**: set `asr` when the words come from one of the listed recognisers.
- **Streaming cost**: with the native backend, per-word polling is fine. Under ONNX Runtime each poll re-runs
  `stream_ctx` words: lower it or poll every few words.
