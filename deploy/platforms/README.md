# Platform adapters

One image, one contract ([`../openapi.yaml`](../openapi.yaml)); platforms differ only in where the weights come from,
which port and health route they probe, and how requests reach the container.

| platform | weights | port / health | requests | status |
|---|---|---|---|---|
| [Hugging Face Inference Endpoints](huggingface/README.md) | `WhissleAI/arc-tagger` mounted at `/repository` (found in `/repository/onnx`) | 8080, `/v1/health/ready` | `/v1/tags`, `/v1/realtime` | ready to try |
| [Amazon SageMaker / AWS Marketplace](sagemaker/README.md) | `model.tar.gz` at `/opt/ml/model`, or bundled | 8080, `/ping` | `/invocations` (real-time, batch transform) | ready to try; Marketplace needs the checklist |
| [Snowflake SPCS](snowflake/README.md) | bundled | 8080, `/v1/health/ready` | service functions -> `/v1/snowflake/tags` | ready to try (amd64) |
| [NVIDIA NIM](nim/README.md) | | NIM-style health / models routes exist | | not a candidate: CPU model, no GPU path |

The server never reaches the network, so every platform's network isolation works as is.
