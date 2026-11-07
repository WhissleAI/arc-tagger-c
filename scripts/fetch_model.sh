#!/usr/bin/env bash
# Download the C-library model files (onnx/ folder of the private Hugging Face repo) into ./model (or $1).
#   HF_TOKEN=hf_... scripts/fetch_model.sh [DEST_DIR]
# Env: HF_REPO (default ksingla025/arc-tagger), HF_REVISION (default main).
set -euo pipefail
REPO="${HF_REPO:-ksingla025/arc-tagger}"
REV="${HF_REVISION:-main}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${1:-$ROOT/model}"
if [ -z "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN is not set (the model repo is private). Create a read token at https://huggingface.co/settings/tokens" >&2
  exit 1
fi
mkdir -p "$DEST"
for f in arctag.onnx arctag_meta.json tokenizer.json SHA256SUMS; do
  url="https://huggingface.co/${REPO}/resolve/${REV}/onnx/${f}"
  echo "fetching onnx/$f"
  curl -fL --retry 3 -H "Authorization: Bearer ${HF_TOKEN}" -o "$DEST/$f.part" "$url"
  mv "$DEST/$f.part" "$DEST/$f"
done
cd "$DEST"
if command -v sha256sum >/dev/null; then sha256sum -c SHA256SUMS; else shasum -a 256 -c SHA256SUMS; fi
echo "model files in $DEST"
