# Platforms and marketplace listings

One image, one contract ([`../openapi.yaml`](../openapi.yaml)). Platforms differ only in where the weights come from,
which health route they probe, and how requests reach the container. The server never reaches the network, so every
platform's network isolation works as is.

This page is the single place for marketplace status and listing checklists. Each folder has the how-to. For audio
in, transcript + tags out, list the combined image from
[arc_transcriber](https://github.com/WhissleAI/arc_transcriber/blob/main/deploy/platforms/README.md) instead.

## Status

| platform | guide | weights | health | status |
|---|---|---|---|---|
| Hugging Face Inference Endpoints | [huggingface/](huggingface/README.md) | `WhissleAI/arc-tagger` at `/repository` | `/v1/health/ready` | ready to try |
| Amazon SageMaker / AWS Marketplace | [sagemaker/](sagemaker/README.md) | `model.tar.gz` or bundled | `/ping` | model package and test endpoint templates; not yet validated in an AWS account; not listed |
| Snowflake (SPCS) | [snowflake/](snowflake/README.md) | bundled | `/v1/health/ready` | service ready to try (amd64); Marketplace Native App: the combined one in arc_transcriber; not listed |
| NVIDIA NIM | [nim/](nim/README.md) | | | **blocked: CPU-only** (no GPU path) |

## Checklist for every paid listing

- [ ] **Licence**: code and weights are CC BY-NC 4.0. A paid or commercial listing needs Whissle's commercial EULA.
- [ ] **Image** in the platform's registry, immutable version tags, scanned (critical / high findings cleared),
      rebuilt on base-image security updates.
- [ ] **No network at run time**: the conformance run checks readiness under `--network none`.
- [ ] **Listing copy**: description and model card (below), usage (the platform guide), sample input and output,
      support contact, limits (English only, text in).

## AWS Marketplace (SageMaker model package)

- [ ] Image in ECR in the seller account, in each listed region.
- [ ] Model package from [`sagemaker/model-package.json`](sagemaker/model-package.json): content types
      `application/json`, `application/jsonlines`, `text/plain`; CPU instance types; `CertifyForMarketplace`; a
      validation batch transform on sample JSON Lines in S3.
- [ ] Run the test endpoint and the validation transform in an AWS account before submitting.
- [ ] Price per instance-hour in the listing; SageMaker meters, the container does not.
- [ ] A buyer notebook (the transcriber's [sample notebook](https://github.com/WhissleAI/arc_transcriber/blob/main/deploy/platforms/sagemaker/sample_notebook.ipynb) covers the combined product).

## Hugging Face

- [ ] Check WebSocket support on the endpoint type before advertising `/v1/realtime`.
- [ ] Check `endpoint.json` field names against the current Inference Endpoints API.

## NVIDIA NIM

Blocked. NIMs are GPU containers and Arc Tagger is CPU-only. See [nim/](nim/README.md).

## Listing copy (model card)

**Model**: Arc Tagger v0.4.1 (`WhissleAI/arc-tagger`, `onnx/`), a text-only streaming tagger of about 10 M parameters
(fp32 34 MB, int8 9.4 MB). Served by arctag 0.5.0, native backend.

**Does** (English): spans for entities, numbers, PII, profanity and toxicity; 28 per-word events; utterance read-outs
for intent, dialogue act, safety, affect, speaker role (needs speaker ids), topic and content state; redaction;
per-vertical configs; a kids / game guard layer. 284 tags in all.

**Does not**: other languages; custom name lists; the T1 / T2 delayed tiers; automatic choice of the top-level domain;
audio (pair it with Arc Transcriber or any ASR).

**Checked**: native backend vs the Python reference on 120 cases, every tag, range and decision identical (max score
difference 4.6e-5); the server's conformance run checks parity again (122 cases, 0 differing).

**Speed and memory** (1 thread, shared machine): native int8 loads in 68 ms, streams at 2.9 / 7.9 ms per word
(p50 / p95), 26 MB per handle; fp32 155 ms, 6.2 / 12.1 ms, 72 MB.

**Quality**: on the Whissle internal held-out test set, span F1 0.544 (fp32) and 0.543 (int8). See the
[model card](https://huggingface.co/WhissleAI/arc-tagger).

**Licence**: CC BY-NC 4.0 with the Whissle terms ([LICENSE](../../LICENSE)); cJSON is MIT. Commercial use:
ksingla@whissle.ai.
