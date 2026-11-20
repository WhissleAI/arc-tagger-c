#!/usr/bin/env bash
# Build arctag-serve natively, start it, run conformance.py (routes + realtime + parity with python/arctag_ref.py), stop
# it. Then the same against the Docker image when a Docker daemon is already running (it never starts Docker Desktop).
#
#   deploy/conformance/run.sh MODEL_DIR [--python PY] [--skip-reference] [--no-docker]
#
# MODEL_DIR: the model folder (arctag.onnx, arctag_meta.json, tokenizer.json; deploy/fetch_weights.sh or
# scripts/fetch_model.sh). PY: an interpreter with numpy, onnxruntime==1.20.1, tokenizers for the reference.
# Builds run under nice -n 19 with make -j2.
# Part of arc-tagger-c (Whissle Inc.), CC BY-NC 4.0 with the Whissle terms; see LICENSE.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
MODEL="${1:?usage: run.sh MODEL_DIR [--python PY] [--skip-reference] [--no-docker]}"
MODEL="$(cd "$MODEL" && pwd)"
shift
PY=python3
REF=()
DOCKER=1
while [ $# -gt 0 ]; do
  case "$1" in
    --python) PY="$2"; shift 2;;
    --skip-reference) REF=(--skip-reference); shift;;
    --no-docker) DOCKER=0; shift;;
    *) echo "unknown argument $1" >&2; exit 2;;
  esac
done
OUT="${CONFORMANCE_OUT:-$ROOT/deploy/build/conformance}"
mkdir -p "$OUT"
PORT="${PORT:-18$(( RANDOM % 900 + 100 ))}"
URL="http://127.0.0.1:$PORT"
PID=""
CID=""
cleanup() { [ -n "$PID" ] && kill "$PID" 2>/dev/null || true; [ -n "$CID" ] && docker rm -f "$CID" >/dev/null 2>&1 || true; }
trap cleanup EXIT
wait_ready() {
  for _ in $(seq 1 120); do
    if curl -fs "$1/v1/health/ready" >/dev/null 2>&1; then return 0; fi
    sleep 0.5
  done
  echo "server at $1 did not become ready" >&2
  return 1
}

echo "== native build"
nice -n 19 make -C "$ROOT/deploy" -j2 >"$OUT/build-native.log" 2>&1 || { tail -30 "$OUT/build-native.log"; exit 1; }
echo "== native server on $URL"
ARCTAG_MODEL_DIR="$MODEL" PORT="$PORT" HOST=127.0.0.1 ARCTAG_WORKERS=2 ARC_ACCESS_LOG=0 nice -n 19 "$ROOT/deploy/build/arctag-serve" \
  >"$OUT/server-native.log" 2>&1 &
PID=$!
wait_ready "$URL"
rc=0
nice -n 19 "$PY" "$HERE/conformance.py" --url "$URL" --model-dir "$MODEL" --python "$PY" ${REF[@]+"${REF[@]}"} \
  --report "$OUT/report-native.json" || rc=$?
kill "$PID" 2>/dev/null || true
wait "$PID" 2>/dev/null || true
PID=""

if [ "$DOCKER" = 1 ] && command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  echo "== docker build"
  TAG=arc-tagger:conformance
  nice -n 19 docker build -q -f "$ROOT/deploy/Dockerfile" --target runtime -t "$TAG" "$ROOT" >"$OUT/build-docker.log" 2>&1 \
    || { tail -30 "$OUT/build-docker.log"; exit 1; }
  # network isolation (SageMaker): it must start and be ready with no network at all
  CID=$(docker run -d --network none -v "$MODEL:/opt/ml/model:ro" "$TAG" serve)
  ok=0
  for _ in $(seq 1 60); do
    if docker exec "$CID" /opt/arc/bin/serve --healthcheck >/dev/null 2>&1; then ok=1; break; fi
    sleep 0.5
  done
  docker rm -f "$CID" >/dev/null; CID=""
  [ "$ok" = 1 ] && echo "PASS  docker: starts and is ready with --network none" || { echo "FAIL  docker: not ready with --network none"; rc=1; }
  CID=$(docker run -d -p "127.0.0.1:$PORT:8080" -v "$MODEL:/opt/ml/model:ro" "$TAG" serve)
  wait_ready "$URL"
  nice -n 19 "$PY" "$HERE/conformance.py" --url "$URL" --model-dir "$MODEL" --python "$PY" ${REF[@]+"${REF[@]}"} \
    --report "$OUT/report-docker.json" || rc=$?
  docker rm -f "$CID" >/dev/null; CID=""
else
  echo "== docker: no running Docker daemon (or --no-docker): the image is UNTESTED in this run"
fi
exit $rc
