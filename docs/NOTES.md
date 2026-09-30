# Jev-Omni 파일럿 진행 기록 (2026-09-29)

- OS: Ubuntu 26.04.1 (네이티브, WSL 아님) / RTX 4070 Ti 12GB, Driver 595.91.07, CUDA 13.3 (nvcc) / 디스크 여유 499G
- Python: 3.14.4, venv `.venv` (huggingface_hub, numpy, pillow). 시스템 `hf`는 huggingface_hub 누락으로 깨져 있어 venv 사용
- llama.cpp: 검증 리비전 `6e60f35608ec6918b44a9839c0c433687165f086` 그대로 빌드 성공 (GGML_CUDA=ON, arch 89, gcc 15.2)
- 모델 SHA-256: 3종 모두 README 값과 일치 (`sha256.expected`)
- 서버: README 옵션 + `-ngl 99 --no-mmproj-offload`, 127.0.0.1:8080 (`restart_server.sh`)
- VRAM: llama-server 10,444 MiB (ctx 8192; README 9,256 MiB는 ctx 2048 기준), GPU 전체 11,287/12,282 MiB

## 이슈: mmproj CUDA 실행 시 NaN
- 증상: `--image` 판정 시 확률 NaN. 이후 텍스트 판정까지 NaN (서버 재시작 전까지 지속)
- 격리: `-fa off` → 여전히 NaN / `--no-mmproj-offload` → 정상
- 결론: 이 리비전+CUDA에서 비전 인코더(mmproj) GPU 경로가 NaN 생성. 인코더만 CPU로, LLM은 GPU 유지
- eval_anomaly.py는 NaN 점수 발견 시 즉시 중단

## 동작 확인
- 텍스트(README 예시): No 0.9984 / Yes 0.0016, 0.23s
- 이미지(vscode 아이콘 RGB): Illustration 0.9955, 0.57~0.69s (3회 동일 값)

## 어댑터 출력 형식 (코드 확인)
- `decide()` → `{prediction, prediction_index, confidence, probabilities: {option: p}}`, softmax over options
- 헤드 가중치는 선택지 **위치** 기반 (`linear.weight[:n]`) → 선택지 순서 고정: [Normal, Anomalous]

## 추가 실험 (같은 날)
- screw 튜닝: dev/holdout 분할, v1(결함 유형 명시) holdout AUROC 0.901→0.956, 점수가 1 근처로 쏠림
- 최신 llama.cpp 0bc845d3: GPU mmproj NaN 동일, CPU mmproj 점수 bit 동일, 0.468→0.356 s/img
- restart_server.sh: LLAMA_DIR 환경변수로 빌드 선택
