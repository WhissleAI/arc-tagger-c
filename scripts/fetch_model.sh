#!/usr/bin/env bash
# Download the C-library model files of Arc Tagger v0.3.0 (onnx/ folder of the private Hugging Face repo) into ./model (or $1).
#   HF_TOKEN=hf_... scripts/fetch_model.sh [DEST_DIR]
# Env:
#   HF_REPO        default WhissleAI/arc-tagger
#   HF_REVISION    default main (a tag, branch or commit of the model repo; pin it to the v0.3.0 upload for reproducible tests)
#   ARCTAG_FETCH   fp32 | int8 | all (default all): which ONNX graphs to fetch
#   ARCTAG_INT8_URL  optional URL of arctag_int8.onnx when it is not in the HF repo (e.g. a GitHub release asset;
#                    a GitHub URL is fetched with $GITHUB_TOKEN when set)
# The int8 graph is optional: if it can be found nowhere, the script says so and the fp32 files are still installed.
set -euo pipefail
REPO="${HF_REPO:-WhissleAI/arc-tagger}"
REV="${HF_REVISION:-main}"
WHICH="${ARCTAG_FETCH:-all}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${1:-$ROOT/model}"
if [ -z "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN is not set (the model repo is private). Create a read token at https://huggingface.co/settings/tokens" >&2
  exit 1
fi
case "$WHICH" in fp32|int8|all) ;; *) echo "ARCTAG_FETCH must be fp32, int8 or all" >&2; exit 2;; esac
mkdir -p "$DEST"

hf_get() {  # hf_get FILE -> 0 on success
  local url="https://huggingface.co/${REPO}/resolve/${REV}/onnx/$1"
  curl -fsSL --retry 3 -H "Authorization: Bearer ${HF_TOKEN}" -o "$DEST/$1.part" "$url" && mv "$DEST/$1.part" "$DEST/$1"
}

files=(arctag_meta.json tokenizer.json SHA256SUMS)
[ "$WHICH" != int8 ] && files+=(arctag.onnx)
for f in "${files[@]}"; do
  echo "fetching onnx/$f"
  hf_get "$f" || { rm -f "$DEST/$f.part"; echo "failed to fetch onnx/$f from ${REPO}@${REV}" >&2; exit 1; }
done

if [ "$WHICH" != fp32 ]; then
  echo "fetching onnx/arctag_int8.onnx"
  if ! hf_get arctag_int8.onnx; then
    rm -f "$DEST/arctag_int8.onnx.part"
    if [ -n "${ARCTAG_INT8_URL:-}" ]; then
      echo "not in ${REPO}@${REV}; fetching ${ARCTAG_INT8_URL}"
      auth=()
      case "$ARCTAG_INT8_URL" in *github.com*) [ -n "${GITHUB_TOKEN:-}" ] && auth=(-H "Authorization: Bearer ${GITHUB_TOKEN}" -H "Accept: application/octet-stream");; esac
      curl -fsSL --retry 3 "${auth[@]}" -o "$DEST/arctag_int8.onnx.part" "$ARCTAG_INT8_URL" && mv "$DEST/arctag_int8.onnx.part" "$DEST/arctag_int8.onnx" \
        || { rm -f "$DEST/arctag_int8.onnx.part"; echo "failed to fetch $ARCTAG_INT8_URL" >&2; [ "$WHICH" = int8 ] && exit 1; }
    else
      echo "arctag_int8.onnx is not in ${REPO}@${REV} (set ARCTAG_INT8_URL to fetch it from elsewhere); int8 precision will be unavailable" >&2
      [ "$WHICH" = int8 ] && exit 1
    fi
  fi
fi

cd "$DEST"
# check every file that is present (SHA256SUMS lists both graphs)
if command -v sha256sum >/dev/null; then sha256sum -c --ignore-missing SHA256SUMS; else shasum -a 256 -c --ignore-missing SHA256SUMS; fi
echo "model files in $DEST: $(ls arctag*.onnx 2>/dev/null | tr '\n' ' ')"
