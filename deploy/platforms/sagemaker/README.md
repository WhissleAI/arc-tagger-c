# Amazon SageMaker and AWS Marketplace

arctag-serve implements SageMaker's inference container contract: `docker run IMAGE serve`, port 8080, `GET /ping`,
`POST /invocations`, `GET /execution-parameters`; model from `/opt/ml/model`; no network calls, so it runs with
`EnableNetworkIsolation: true` (required for Marketplace).

| file | what |
|---|---|
| [`endpoint.cfn.yaml`](endpoint.cfn.yaml) | test endpoint: model (network isolation), endpoint config, endpoint |
| [`model-package.json`](model-package.json) | `create-model-package` input for the Marketplace |

Marketplace checklist and status: [platforms/README.md](../README.md#aws-marketplace-sagemaker-model-package).

## Model artifacts

```bash
deploy/fetch_weights.sh                                   # deploy/weights/arc-tagger (public, checksummed)
tar -czf model.tar.gz -C deploy/weights/arc-tagger .      # files at the root (an onnx/ subfolder also works)
aws s3 cp model.tar.gz s3://<bucket>/arc-tagger/v0.4.1/model.tar.gz
```

Or bundle them (`--target bundled`) and leave `ModelDataUrl` out.

## Test endpoint

Deploy [`endpoint.cfn.yaml`](endpoint.cfn.yaml), then:

```bash
aws sagemaker-runtime invoke-endpoint --endpoint-name arc-tagger-test --content-type application/json \
  --body '{"text": "Hi, this is Maria Lopez. My number is 555 867 5309.", "domain": "talk"}' --cli-binary-format raw-in-base64-out out.json
```

`/invocations` accepts `application/json` (one request or `{"instances": [...]}`), `application/jsonlines` (one request
per line) and `text/plain`. Audio is refused with 415: use Arc Transcriber or its combined image.

## Batch transform

```bash
aws sagemaker create-transform-job --transform-job-name arc-tagger-batch-1 --model-name arc-tagger \
  --transform-input DataSource={S3DataSource={S3DataType=S3Prefix,S3Uri=s3://<bucket>/transcripts/}},ContentType=application/jsonlines,SplitType=Line \
  --transform-output S3OutputPath=s3://<bucket>/transcripts-tagged/,Accept=application/jsonlines,AssembleWith=Line \
  --transform-resources InstanceType=ml.c6i.xlarge,InstanceCount=1 --batch-strategy MultiRecord --max-payload-in-mb 6
```

Each input line (`{"text": ...}`, `{"words": [...]}` or `{"segments": [...]}`) gives one output line, in order.
