#!/bin/bash
# Start llama-server (loopback only) if needed, then the Gradio app. Extra args go to app.py (--lan, --share --auth u:p).
set -euo pipefail
cd "$(dirname "$0")"
if ! curl -s 127.0.0.1:8080/health | grep -q ok; then
  # --no-mmproj-offload: the vision projector returns NaN on CUDA (llama.cpp 6e60f356 and 0bc845d3), so it runs on CPU
  (cd jev-omni-q4 && nohup ../${LLAMA_DIR:-llama.cpp}/build/bin/llama-server -m Jev-Omni-Unified-Q4_K_M.gguf \
    --mmproj mmproj-jev-omni.gguf --no-mmproj-offload --embedding --pooling none --host 127.0.0.1 --port 8080 \
    --ctx-size 8192 --parallel 1 --batch-size 2048 --ubatch-size 2048 --threads 8 --no-warmup -ngl 99 > ../server.log 2>&1 &)
  for i in $(seq 1 180); do curl -s 127.0.0.1:8080/health | grep -q ok && break; sleep 1; done
  curl -s 127.0.0.1:8080/health | grep -q ok || { echo "llama-server failed, see server.log"; tail -5 server.log; exit 1; }
fi
exec .venv/bin/python app.py "$@"
