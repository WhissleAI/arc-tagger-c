#!/usr/bin/env bash
# Arc Tagger web demo in one command: builds the library (shared copy included), checks the model and the one Python package,
# then serves http://localhost:8080 (extra arguments go to demo/server.py, e.g. --int8 or --port 9000).
#   demo/run.sh                          (fetches the model into ./model the first time)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python3}"
[ -d third_party/onnxruntime ] || ./scripts/download_onnxruntime.sh
[ -f model/arctag.onnx ] || ./scripts/fetch_model.sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release > /dev/null
cmake --build build -j 4
if ! "$PY" -c "import websockets" 2> /dev/null; then
  echo "The demo needs one Python package: $PY -m pip install -r demo/requirements.txt" >&2
  exit 1
fi
exec "$PY" demo/server.py "$@"
