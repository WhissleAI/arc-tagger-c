# Kids and game domain, and the guard layer

Domain `kids` (alias `game`) is for short spoken turns of children talking to a character in an app or game: "copy
that", "fire the thrusters", "this is stupid", "i want to hurt myself". These turns are much shorter than the sessions the
other domains are calibrated for, and the plain utterance read-outs miss some tags on them or fire on harmless questions.
The kids domain fixes this at inference time:

- **Thresholds calibrated on short kid turns.** It is the `assistant` domain's model input with its own utterance
  thresholds. The library uses `arctag_kids.json` from the model directory when there is one, and otherwise the built-in
  table (`src/kids_thr.inc`, same values).
- **The guard layer** (`src/guard.c`), on by default in this domain:
  - *Profanity, insult and threat are also utterance safety tags.* Profanity is a span family. The app's read-out of
    `utterances[].safety` now gets `safety.profanity.strong` / `.mild` (from a profanity lexicon and the model's
    profanity spans), `safety.toxicity.insult_harassment` and `safety.toxicity.threat_violence` (second-person rules).
    Minced oaths ("dang", "heck", "frick") and kid words ("stupid", "sucks", "hate", "butt") are never profanity.
  - *PII needs evidence.* `safety.pii.leak` is on only when the words hold a phone-like digit run, an email, an
    address, a birthday, a self-introduction with a name, a school name, or a strong PII span with numbers in a
    first-person turn. `safety.pii.request` is on only for a request cue ("what's your address", "where do you live").
  - *Backstops* for `safety.crisis.self_harm`, `safety.child.grooming_cue`, `safety.child.inappropriate` and
    `safety.jailbreak.attempt`: explicit first-person self-harm, picture / secrecy / meeting / platform-move requests,
    sexual words, and rule-bypass requests turn the tag on.
  - *Gates* turn a model-only tag off when the turn is a cockpit command (with `command_context`), when "kill" is aimed at
    someone else (a threat, not self-harm), when there is no first person, or when it is a figure of speech ("dying of
    laughter", "this game is killing me").

## Options

| option / config key | values | |
|---|---|---|
| `domain` | `kids` \| `game` | the profile `kids:kids` |
| `guard` | `auto` (default) \| `on` \| `off` | auto = on in the kids domain only. `on` applies the guard in any domain. |
| `command_context` | `on` \| `off` (default) | set it while the child is on a command screen: "kill the heaters" and "fire the torpedo" never raise safety tags there |

```c
arctag_set_option(h, "domain", "kids");
arctag_set_option(h, "command_context", on_cockpit_screen ? "on" : "off");   /* per turn, cheap */
```

`configs/kids_game.json` is a ready config (int8, 1 thread, streaming).

## Output

The output format is unchanged. These fields are added:

- utterance `safety` entries decided by a rule have `"src"` (`rule`, `lexicon`, `pii_evidence`, `span`), `"p": 1` and the
  model's own probability in `"p_model"`;
- entries a gate turned off keep `"on": false` and have `"gated"` (`no_pii_evidence`, `no_request_cue`,
  `command_context`, `directed_at_others`, `not_first_person`, `figure_of_speech`, `profanity_only`);
- in guard mode every tag that is on is listed (then up to 6 off ones), so an on tag is never cut by the listing cap;
- profanity spans added from the lexicon have `"src": "lexicon"`.

`arctag_info` reports `"guard"` (`auto:on`, `auto:off`, `on`, `off`) and `"command_context"`.

With the guard off, or in any other domain, the output is byte-identical to the library without it.

## Calibration (kids domain)

Held-out set of 400 short kid turns (synthetic, written independently of the rules; benign and command turns included).
Utterance level, int8, `command_context` on for the command turns.

| tag | precision | recall | F1 | support |
|---|---|---|---|---|
| `safety.profanity` (either severity) | 1.00 | 1.00 | 1.00 | 46 |
| `safety.profanity.strong` | 1.00 | 1.00 | 1.00 | 26 |
| `safety.profanity.mild` | 1.00 | 1.00 | 1.00 | 20 |
| `safety.toxicity.insult_harassment` | 1.00 | 0.73 | 0.85 | 30 |
| `safety.toxicity.threat_violence` | 0.92 | 0.73 | 0.81 | 15 |
| `safety.crisis.self_harm` | 1.00 | 0.51 | 0.68 | 35 |
| `safety.child.grooming_cue` | 1.00 | 0.53 | 0.70 | 30 |
| `safety.child.inappropriate` | 0.80 | 0.50 | 0.62 | 8 |
| `safety.pii.leak` | 1.00 | 1.00 | 1.00 | 22 |
| `safety.pii.request` | 1.00 | 0.79 | 0.88 | 14 |
| `safety.jailbreak.attempt` | 1.00 | 0.70 | 0.82 | 10 |

Benign turns (198, including 40 cockpit commands): any safety tag 0.5 %, self-harm / grooming 0.0 %, PII 0.0 %, jailbreak 0.0 %. Exact tag-set match over all 400 turns: 0.863.

Precision is high on every tag. The tags with lower recall (self-harm, grooming, inappropriate) miss indirect phrasings
("i wrote a goodbye letter to my mom", "you're my special friend", "close your door so nobody hears us"). Use them as
**review triggers** for a caring adult or a moderator, together with your own conversation-level signals, not as the only
safety net.

The Python reference (`python/arctag_ref.py` with `python/arctag_guard.py`) gives the same tags as the C library on this
set and on the kids test cases (`tests/test_kids.c`); `tests/test_guard_tables.py` checks that the two hold the same
tables.
