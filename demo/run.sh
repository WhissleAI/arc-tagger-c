#!/usr/bin/env bash
# Arc Tagger web demo in one command: builds the library (shared copy included), checks the model and the one Python package,
# then serves http://localhost:8080 (extra arguments go to demo/server.py, e.g. --int8 or --port 9000).
#   HF_TOKEN=hf_... demo/run.sh          (the token is only needed once, to fetch the model into ./model)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python3}"
[ -d third_party/onnxruntime ] || ./scripts/download_onnxruntime.sh
if [ ! -f model/arctag.onnx ]; then
  if [ -n "${HF_TOKEN:-}" ]; then
    ./scripts/fetch_model.sh
  else
    echo "No model in ./model. Get a read token for WhissleAI/arc-tagger, then: export HF_TOKEN=hf_... && scripts/fetch_model.sh" >&2
    exit 1
  fi
fi
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release > /dev/null
cmake --build build -j 4
if ! "$PY" -c "import websockets" 2> /dev/null; then
  echo "The demo needs one Python package: $PY -m pip install -r demo/requirements.txt" >&2
  exit 1
fi
exec "$PY" demo/server.py "$@"
