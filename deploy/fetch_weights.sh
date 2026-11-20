#!/usr/bin/env bash
# Fetch the PUBLIC Arc Tagger model files an image bundles into deploy/weights/arc-tagger (git-ignored), checked against
# the model repo's SHA256SUMS. Build time only: the server never downloads anything.
#
#   deploy/fetch_weights.sh            (WhissleAI/arc-tagger @ v0.4.1, onnx/: fp32 + int8 graphs, meta, tokenizer; 44 MB)
# Env: ARCTAG_REVISION (v0.4.1), DEST (deploy/weights), HF_TOKEN (optional: the repo is public).
# Part of arc-tagger-c (Whissle Inc.), CC BY-NC 4.0 with the Whissle terms; see LICENSE.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
DEST="${DEST:-$HERE/weights}/arc-tagger"
REV="${ARCTAG_REVISION:-v0.4.1}"
AUTH=()
[ -n "${HF_TOKEN:-}" ] && AUTH=(-H "Authorization: Bearer ${HF_TOKEN}")
mkdir -p "$DEST"
for f in SHA256SUMS arctag_meta.json tokenizer.json arctag.onnx arctag_int8.onnx; do
  echo "fetching onnx/$f"
  curl -fsSL --retry 3 ${AUTH[@]+"${AUTH[@]}"} -o "$DEST/$f.part" "https://huggingface.co/WhissleAI/arc-tagger/resolve/$REV/onnx/$f"
  mv "$DEST/$f.part" "$DEST/$f"
done
cd "$DEST"
if command -v sha256sum >/dev/null; then sha256sum -c SHA256SUMS; else shasum -a 256 -c SHA256SUMS; fi
du -sh "$DEST"
