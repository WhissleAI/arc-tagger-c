#!/bin/bash
# Every native runtime test, one command:
#
#   ./scripts/test_native.sh MODEL_DIR            # the full suite (about 9 minutes on a busy M3 Pro)
#   ./scripts/test_native.sh MODEL_DIR --quick    # the same with small fuzz and soak counts
#
# MODEL_DIR holds an fp32 arctag.onnx, arctag_meta.json and tokenizer.json (scripts/fetch_model.sh). It is only read. Exports go to
# BUILD (default ./build-test, about 100 MB; safe to delete), so this needs python3 with numpy, onnx, onnxruntime and tokenizers.
#
# Steps: export fp32 and int8 (each export includes tensor parity against the matching ONNX) -> kernels without Accelerate -> public
# API vs the Python reference -> streaming invariance and revisions -> threads under TSan -> corrupt files and input fuzz under
# ASan + UBSan -> soak. Exit status: non-zero on the first failure.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD
MODEL=${1:?usage: test_native.sh MODEL_DIR [--quick]}
QUICK=${2:-}
B=${BUILD:-$ROOT/build-test}
mkdir -p "$B"
CORRUPT=2000; FUZZ=4000; SOAK=200000; THREADS=8; ROUNDS=4
if [ "$QUICK" = "--quick" ]; then CORRUPT=100; FUZZ=300; SOAK=5000; THREADS=4; ROUNDS=1; fi

CC=${CC:-cc}
INC="-Iinclude -Isrc -Ithird_party/cjson -Itests"
LIB="src/arctag.c src/atn.c src/tokenizer.c third_party/cjson/cJSON.c"
FW=""; [ "$(uname)" = Darwin ] && FW="-framework Accelerate"
OPT="-std=gnu99 -O2 -Wall -Wno-unused-function"
step() { printf '\n=== %s\n' "$*"; }
build() { local name=$1 src=$2; shift 2; $CC $OPT $INC $LIB "$src" $FW -lm "$@" -o "$B/$name"; }

step "export and tensor parity against the matching ONNX"
python3 tools/export_model.py "$MODEL" --out "$B/m_f32" --weights fp32 --keep-onnx | tail -3
python3 tools/export_model.py "$MODEL" --out "$B/m_i8" --weights int8 --keep-onnx | tail -3
for k in f32 i8; do python3 tools/make_refs.py "$B/m_$k" "$B/refs_$k.bin"; done

step "export gates: a new checkpoint passes, altered graphs fail loudly"
python3 tests/test_export.py "$MODEL" "$B/tmp_export"

step "other kernels on the same references: built-in matmul, plain C"
for flags in "-DATN_NO_ACCELERATE" "-DATN_NO_ACCELERATE -DATN_SCALAR"; do
    $CC $OPT -Isrc src/atn.c tests/test_native_parity.c -lm $flags -o "$B/parity_alt"
    for k in f32 i8; do "$B/parity_alt" "$B/m_$k/arctag.atw" "$B/refs_$k.bin" | tail -2; done
done

step "public API vs the Python reference (fp32; tests/expected/reference.json comes from the fp32 graph)"
build test_parity tests/test_parity.c
"$B/test_parity" "$B/m_f32" tests/expected/reference.json
build test_tokenizer tests/test_tokenizer.c
"$B/test_tokenizer" "$B/m_f32/tokenizer.json" tests/expected/tokens.json

step "streaming invariance, cadence, revisions, speakers, options"
build test_stream tests/test_stream.c
"$B/test_stream" "$B/m_f32" tests/expected/reference.json
"$B/test_stream" "$B/m_i8" tests/expected/reference.json

step "multi-thread, ThreadSanitizer"
build test_threads_tsan tests/test_threads.c -fsanitize=thread -g
TSAN_OPTIONS="halt_on_error=1" "$B/test_threads_tsan" "$B/m_i8" "$THREADS" "$ROUNDS"

step "corrupt model files, AddressSanitizer + UndefinedBehaviorSanitizer"
SAN="-fsanitize=address,undefined -fno-sanitize-recover=undefined -fno-omit-frame-pointer -g"
build test_corrupt_asan tests/test_corrupt.c $SAN
"$B/test_corrupt_asan" "$B/m_i8" "$B/tmp_corrupt" "$CORRUPT"

step "text and JSON input fuzz, ASan + UBSan (kernels: this platform's and plain C)"
build fuzz_inputs_asan tests/fuzz_inputs.c $SAN
"$B/fuzz_inputs_asan" "$B/m_i8" "$FUZZ" 1
"$B/fuzz_inputs_asan" "$B/m_f32" "$FUZZ" 2
build fuzz_inputs_scalar tests/fuzz_inputs.c $SAN -DATN_NO_ACCELERATE -DATN_SCALAR
"$B/fuzz_inputs_scalar" "$B/m_i8" $((FUZZ / 4)) 3

step "streaming gates and a short soak, ASan + UBSan: the windows moving, revisions, compaction"
build test_stream_asan tests/test_stream.c $SAN
"$B/test_stream_asan" "$B/m_i8" tests/expected/reference.json | tail -1
build test_soak_asan tests/test_soak.c $SAN
"$B/test_soak_asan" "$B/m_i8" $((SOAK / 10)) | tail -1

step "out of memory: every allocation made to fail once, ASan + UBSan"
$CC $OPT $INC -include tests/fail_alloc.h $LIB tests/test_oom.c $FW -lm $SAN -o "$B/test_oom"
"$B/test_oom" "$B/m_i8"

step "soak: $SOAK words"
build test_soak tests/test_soak.c
"$B/test_soak" "$B/m_i8" "$SOAK"

step "all native tests passed"
