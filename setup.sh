#!/bin/bash
# One-time setup: Python venv, llama.cpp (CUDA) build, model download + SHA-256 check.
set -euo pipefail
cd "$(dirname "$0")"
LLAMA_REV=0bc845d356f437d5ce4fe975c36428f7522829cb   # tested; pinned README rev 6e60f356 also works (24% slower)

for c in nvidia-smi nvcc cmake git python3; do command -v $c >/dev/null || { echo "missing: $c"; exit 1; }; done

[ -d .venv ] || python3 -m venv .venv
.venv/bin/pip install -q -U pip -r requirements.txt

if [ ! -x llama.cpp/build/bin/llama-server ]; then
  [ -d llama.cpp ] || git clone https://github.com/ggml-org/llama.cpp
  git -C llama.cpp checkout -q $LLAMA_REV
  cmake -S llama.cpp -B llama.cpp/build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native -DCMAKE_BUILD_TYPE=Release
  cmake --build llama.cpp/build --target llama-server -j"$(nproc)"
fi

.venv/bin/hf download Reza2kn/Jev-Omni-Q4_K_M-GGUF --local-dir jev-omni-q4
(cd jev-omni-q4 && sha256sum -c ../sha256.expected)
echo "setup done. next: ./run.sh"
