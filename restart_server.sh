#!/bin/bash
# usage: ./restart_server.sh [extra llama-server args...]
cd "$(dirname "$0")"
pkill -x llama-server ; sleep 2
cd jev-omni-q4
nohup ../${LLAMA_DIR:-llama.cpp}/build/bin/llama-server -m Jev-Omni-Unified-Q4_K_M.gguf --mmproj mmproj-jev-omni.gguf --embedding --pooling none --host 127.0.0.1 --port 8080 --ctx-size 8192 --parallel 1 --batch-size 2048 --ubatch-size 2048 --threads 8 --no-warmup -ngl 99 "$@" > ../server.log 2>&1 &
for i in $(seq 1 120); do curl -s 127.0.0.1:8080/health | grep -q ok && echo ready && exit 0; sleep 1; done; echo TIMEOUT; tail -5 ../server.log
