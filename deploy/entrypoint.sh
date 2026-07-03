#!/usr/bin/env bash
# Air-gapped entrypoint: verify the vendored weights against the pinned
# manifest, start llama-server, then serve or batch-clean.
#
#   cleaner-entrypoint serve                        keep the model server up
#   cleaner-entrypoint batch <in.jsonl> [out.jsonl] clean a mounted file
#
# batch binds the model server to loopback only and is meant to run with
# `--network none`: records in and out move exclusively over mounted volumes.
set -euo pipefail

APP=/app/cleaner
MODEL_DIR=$APP/models
LLAMA_BIN="${LLAMA_BIN:-/app/llama-server}"
PORT="${PORT:-8080}"
ALIAS="${ALIAS:-qwen3-0.6b-cleaner}"
MODE="${1:-serve}"

# 1. never serve weights that don't match the manifest pinned in git
cd "$MODEL_DIR"
sha256sum -c MANIFEST.sha256
MODEL_FILE=$(ls "$MODEL_DIR"/*.gguf | head -1)

# 2. model server: loopback for batch; all interfaces for serve (the operator
#    decides what, if anything, is reachable via the container network)
HOST=127.0.0.1
[ "$MODE" = "serve" ] && HOST=0.0.0.0
"$LLAMA_BIN" -m "$MODEL_FILE" --host "$HOST" --port "$PORT" --alias "$ALIAS" &
SERVER_PID=$!

# 3. wait for the server to come up (no curl needed: stdlib only)
for _ in $(seq 1 120); do
    python3 - "$PORT" <<'PY' && break
import sys, urllib.request
try:
    urllib.request.urlopen(f"http://127.0.0.1:{sys.argv[1]}/health", timeout=2)
except Exception:
    raise SystemExit(1)
PY
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "llama-server died" >&2; exit 1; }
    sleep 1
done

cd "$APP"
case "$MODE" in
    batch)
        IN="${2:?usage: batch <in.jsonl> [out.jsonl]}"
        python3 runtime/clean.py --live --port "$PORT" \
            --batch "$IN" ${3:+--out "$3"} \
            --audit-dir /data/audit --model-file "$MODEL_FILE"
        ;;
    serve)
        wait "$SERVER_PID"
        ;;
    *)
        echo "unknown mode '$MODE' (use: serve | batch <in> [out])" >&2
        exit 2
        ;;
esac
