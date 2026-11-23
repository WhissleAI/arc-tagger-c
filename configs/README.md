# Configs

One JSON file per use case. Each sets the domain, precision and threads, the sections to emit, threshold overrides,
redaction, and batch or streaming. Load one with `-c FILE`, `arctag_create_from_config`, `ARCTAG_CONFIG` (server) or
`POST /config`. Keys: [docs/USAGE.md](../docs/USAGE.md#options-and-configs).

| config | for | precision / threads | mode | domain | overrides and redaction |
|---|---|---|---|---|---|
| [`voice_assistant`](voice_assistant.json) | assistants, voice commands | fp32 / 2 | streaming, ctx 128 | assistant | jailbreak flag at 0.45 |
| [`contact_center`](contact_center.json) | call centers, agent assist | fp32 / 2 | streaming, ctx 256 | talk | scam, PII request, escalation lowered; PII and profanity masked |
| [`trust_safety_gaming`](trust_safety_gaming.json) | chat moderation, gaming | fp32 / 2 | streaming, ctx 128 | stream_chat | threat, child safety, self-harm lowered; profanity, toxicity, PII masked |
| [`legal`](legal.json) | hearings, depositions | fp32 / 4 | batch | court_trial (fine_auto) | objection / ruling events at 0.4 |
| [`meetings`](meetings.json) | meeting notes | fp32 / 4 | batch | meeting | decisions and action items lowered |
| [`sports`](sports.json) | live commentary | fp32 / 2 | streaming, ctx 256 | sports (fine_auto picks the league) | play / score events at 0.5; profanity masked |
| [`news`](news.json) | news and talk shows | fp32 / 2 | batch | news | none |
| [`pii_redaction`](pii_redaction.json) | generic redaction | fp32 / 4 | batch | auto | every private PII category masked |
| [`edge_low_latency`](edge_low_latency.json) | devices, low latency | int8 / 1 | streaming, ctx 96 | auto | none |
| [`kids_game`](kids_game.json) | kids apps and games | int8 / 1 | streaming, ctx 64 | kids, guard on | kids thresholds ([docs/KIDS.md](../docs/KIDS.md)) |

Each file's `_notes` explains its choices. Copy one and edit it: change the domain, drop sections with `emit`, move a
threshold down for recall or up for precision.
