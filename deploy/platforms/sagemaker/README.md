# Amazon SageMaker and AWS Marketplace

arctag-serve implements SageMaker's inference container contract: `docker run IMAGE serve`, port 8080, `GET /ping`,
`POST /invocations`, `GET /execution-parameters`; model from `/opt/ml/model`; no network calls, so it runs with
`EnableNetworkIsolation: true` (required for Marketplace model packages).

## Model artifacts

```bash
deploy/fetch_weights.sh                                   # deploy/weights/arc-tagger (public, checksummed)
tar -czf model.tar.gz -C deploy/weights/arc-tagger .      # files at the root of /opt/ml/model (an onnx/ subfolder also works)
aws s3 cp model.tar.gz s3://<bucket>/arc-tagger/v0.4.1/model.tar.gz
```

Or bundle them (`--target bundled`) and leave `ModelDataUrl` out.

## Test endpoint

[`endpoint.cfn.yaml`](endpoint.cfn.yaml), then:

```bash
aws sagemaker-runtime invoke-endpoint --endpoint-name arc-tagger-test --content-type application/json \
  --body '{"text": "Hi, this is Maria Lopez. My number is 555 867 5309.", "domain": "talk"}' --cli-binary-format raw-in-base64-out out.json
```

`/invocations`: `application/json` (one request or `{"instances": [...]}`), `application/jsonlines` (one request per
line), `text/plain`. Audio is refused with 415 (use Arc Transcriber or the combined image).

## Batch transform

```bash
aws sagemaker create-transform-job --transform-job-name arc-tagger-batch-1 --model-name arc-tagger \
  --transform-input DataSource={S3DataSource={S3DataType=S3Prefix,S3Uri=s3://<bucket>/transcripts/}},ContentType=application/jsonlines,SplitType=Line \
  --transform-output S3OutputPath=s3://<bucket>/transcripts-tagged/,Accept=application/jsonlines,AssembleWith=Line \
  --transform-resources InstanceType=ml.c6i.xlarge,InstanceCount=1 --batch-strategy MultiRecord --max-payload-in-mb 6
```

Each input line `{"text": ...}` / `{"words": [...]}` / `{"segments": [...]}` gives one output line, in order.

## AWS Marketplace model package checklist

- [ ] **Licence**: code and weights are CC BY-NC 4.0; a paid listing needs Whissle's commercial EULA.
- [ ] **Image in ECR** (seller account, each listed region), immutable version tags.
- [ ] **Image scan**: ECR enhanced scanning; clear critical / high findings (distroless runtime keeps this short);
      rebuild on base-image updates.
- [ ] **No network at run time**: verified by conformance `run.sh` (`--network none` readiness).
- [ ] **Model package** from [`model-package.json`](model-package.json): content types `application/json`,
      `application/jsonlines`, `text/plain`; CPU instance types; `CertifyForMarketplace`; a validation batch transform on
      sample JSON lines in S3.
- [ ] **Metering by instance-hour**: hourly price per instance type in the listing; SageMaker meters, the container does
      nothing (and cannot reach a metering API under isolation).
- [ ] **Listing content**: model card (`../../README.md`), usage (this file), sample input / output, a notebook,
      support contact, limits (English only, text in).
- [ ] **Pro tier later**: a second model package from the same Dockerfile with the Pro weights and `MODEL_ID`.
