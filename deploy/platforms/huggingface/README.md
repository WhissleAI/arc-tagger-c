# Hugging Face Inference Endpoints

The endpoint downloads the selected model repository to `/repository`; arctag-serve finds the model in
`/repository/onnx` (the layout of `WhissleAI/arc-tagger`), so the **runtime** image needs no baked weights.

1. `docker buildx build --platform linux/amd64 -f deploy/Dockerfile --target runtime -t <registry>/arc-tagger:0.1.0 --push .`
2. Create the endpoint (UI "Custom container", or the API with [`endpoint.json`](endpoint.json)):

| setting | value |
|---|---|
| model repository | `WhissleAI/arc-tagger`, revision `v0.4.1` |
| container | custom, `<registry>/arc-tagger:0.1.0`, port `8080`, health route `/v1/health/ready` |
| hardware | CPU (1-2 vCPU per replica: one handle uses one core) |
| environment | `ARCTAG_WORKERS=2`, optional `ARCTAG_PRECISION=int8`, `ARCTAG_DOMAIN`, `ARCTAG_CONFIG` |

3. `curl -s https://<endpoint>.endpoints.huggingface.cloud/v1/tags -H "Authorization: Bearer $HF_TOKEN" -H 'Content-Type: application/json' -d '{"text": "..."}'`

Open items (WebSocket support, `endpoint.json` field names): [platforms/README.md](../README.md#hugging-face).
