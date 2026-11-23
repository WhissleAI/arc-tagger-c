# The model: tags, decoding, parity

Arc Tagger v0.4.1 has 284 tags across 10 read-outs (365 `tag@readout` columns); `arctag_meta.json` (`readouts`) lists
every column. The model card is on [Hugging Face](https://huggingface.co/WhissleAI/arc-tagger).

## Tags

| family | tags |
|---|---|
| entity spans | `entity.mention.{person,team,organization,place,work_product,event_name,legal_ref,game_entity,game_map,term}` |
| number spans | `entity.number.{date,time_of_day,duration,clock,money,percent,count,ordinal,score,stat,measure,ratio_odds,field_position,jersey_number,ref_number,id_pii}`, `entity.numeral` |
| PII spans | `safety.pii.{private_name,phone_number,email,street_address,card_account_number,gov_id,url_handle,date_of_birth,public_contact}` |
| profanity / toxicity spans | `safety.profanity.{mild,strong}`, `safety.toxicity.*` (5) |
| events (28) | `event.sports.*` (18), `event.legal.*` (3), `event.meeting.{decision,action_item}.recent`, `discourse.{turn_yield,interruption}`, `discourse.assistant.command_complete`, `speaker.change`, `content.story_start` |
| intent | `intent.*` (26 general), `discourse.intent.*` (60 assistant) |
| dialogue act | `discourse.act.*` (7), `discourse.answer`, `discourse.backchannel`, `discourse.assistant.*` |
| safety | `safety.scam.*` (6), `safety.pii.{leak,request}`, `safety.threat.*`, `safety.crisis.*`, `safety.child.*`, `safety.solicitation.*`, `safety.toxicity.*`, `safety.jailbreak.attempt` |
| affect | `affect.emotion.*` (5), `affect.arousal.*` (3), `affect.valence.*` (4), `affect.intensity.*` (3) |
| speaker role | `speaker.role.*` (19: play-by-play, analyst, host, judge, counsel, witness, agent, customer, ...) |
| item | `item.case_area.*`, `item.esports_title.*` |
| content state | `content.*` (mode, subject, legal phase, meeting phase, game state, sports focus, ...), `scene.*`, `meta.*` |

## How it works

`text -> words -> utterances -> per-word features -> model -> per-word and per-utterance probabilities -> decode`

- **Inputs** per word (`src/arctag.c`): up to 8 BPE pieces, digit flags, gap / duration / confidence buckets,
  final / interim, speaker slot and change, ASR source, utterance starts, the domain id.
- **Decoding** follows the T0 (zero-lag) path of the Arc Tagger Python package:
  - entity, number, PII and profanity spans: greedy BIO on `1 - P(O) >= thr`, typed by the best summed type score;
  - other span tags (toxicity): runs of `span_in >= thr`, split at span starts and utterance starts;
  - events fire at `p >= thr` and are deduplicated within 3 words;
  - read-outs come from the utterance's last word; affect is the top-1 of each group;
  - inside `sports` / `court`, `fine_auto` re-checks lexicon cues every 25 words and switches the fine domain.
- **Thresholds**: the model release's per-profile thresholds, with display floors (spans >= 0.3; toxicity >= 0.5
  outside the safety domains).

## Limits

- English only. Text in: pair it with an ASR (or [Arc Transcriber](https://github.com/WhissleAI/arc_transcriber)) for audio.
- `tag_text` covers at most `max_ctx` (2048) words per pass; longer texts are split into independent chunks.
- Not in the C library: custom name lists (session KB), the T1 / T2 delayed tiers, automatic choice of the top-level
  domain (only the fine domain inside `sports` / `court` is automatic).

## Parity

| check | result |
|---|---|
| Python reference vs the Arc Tagger v0.4.1 package, 16 texts x 7 domains | same tags, ranges, on/off decisions and redacted text; max p diff 5e-4 (the package rounds to 1e-3) |
| C tokenizer vs HF `tokenizers`, 37 strings | identical ids |
| ONNX Runtime backend vs Python reference, 120 cases, fp32 and int8 | all fields identical, scores bit-equal |
| native backend vs Python reference, fp32, 120 cases | every tag, range and decision identical; max score diff 4.6e-5 |
| streaming vs `arctag_tag_text` | identical (native: bit-equal; ONNX Runtime: within 1e-4) |
| native vs ONNX Runtime on `examples/data` | max score diff 1.1e-4 |

The two backends round differently, so their scores differ in the last digits; decisions are the same.

`ctest` runs the tokenizer, parity (fp32, int8, native), int8 agreement, native streaming (revisions, speaker
changes, long sessions, corrupted files refused), configs, kids guard, example smoke tests and the web demo smoke test.
`-DARCTAG_PARITY_EXACT=ON` requires bit-equal scores; `-DARCTAG_SANITIZE=ON` builds with ASan + UBSan (CI runs it).

Regenerate the references with `pip install numpy onnxruntime==1.20.1 tokenizers` and
`python tests/make_expected.py model [--precision int8]`.
