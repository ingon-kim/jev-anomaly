# Jev-Omni Q4_K_M 이미지 이상 탐지 파일럿 리포트

- 일자: 2026-09-29
- 모델: `Reza2kn/Jev-Omni-Q4_K_M-GGUF` (SHA-256 3종 README 값과 일치, 파일 무수정)
- 데이터: MVTec AD test split — bottle / screw / carpet (HF 미러 `TheoM55/mvtec_anomaly_detection`, 공식 배포본과 장수 동일). 라이선스 CC BY-NC-SA 4.0 (비상업)
- 방식: zero-shot, 학습·튜닝 없음. 선택지 `[Normal, Anomalous]`, 이상 점수 = `P(Anomalous)`

## 1. 환경

| 항목 | 값 |
|---|---|
| OS / GPU | Ubuntu 26.04.1 (네이티브) / RTX 4070 Ti 12GB, Driver 595.91.07 |
| CUDA / 컴파일러 | nvcc 13.3 / gcc 15.2 |
| llama.cpp | **검증 리비전 `6e60f35608ec6918b44a9839c0c433687165f086`** (최신 빌드 불필요), `GGML_CUDA=ON`, arch 89 |
| 서버 옵션 | README 옵션 그대로 + `-ngl 99 --no-mmproj-offload`, `127.0.0.1:8080` 바인딩 확인 |
| VRAM (llama-server) | 10,444 MiB 기동 직후 / 10,482 MiB 평가 중 (GPU 전체 약 11.3/12.3 GB) |

- VRAM이 README(9,256 MiB)보다 큰 이유: README 수치는 ctx 2048, 본 실행은 README 명령의 ctx 8192. 여유가 필요하면 `--ctx-size 2048`로 약 1GB 절감 가능 (이미지 1장 프롬프트 ≈ 510 토큰)

### 이슈: mmproj(비전 인코더) CUDA 실행 시 NaN

- 증상: 기본 설정(`-ngl 99`, mmproj GPU)에서 이미지 판정 확률이 NaN. 이후 **텍스트 판정까지 NaN**으로 오염되고 서버 재시작 전까지 지속
- 격리: RGBA/RGB·크기 무관 / `-fa off` → 여전히 NaN / `--no-mmproj-offload` → 정상
- 조치: 비전 인코더만 CPU, LLM 본체는 GPU 전체 오프로드 유지. `eval_anomaly.py`는 NaN 점수 발견 시 즉시 중단
- 후속 확인 결과: 최신 llama.cpp(`0bc845d3`)에서도 동일하게 NaN 재현 → 5.2절

## 2. 동작 확인

| 테스트 | 결과 |
|---|---|
| README 텍스트 예시 (회의 시작?) | No 0.9984 / Yes 0.0016, 0.23 s |
| 이미지 (VS Code 아이콘) Photograph vs Illustration | Illustration 0.9955, 0.57~0.69 s, 3회 동일 값 (결정적) |

어댑터 출력 형식 (코드 확인): `{prediction, prediction_index, confidence, probabilities: {option: p}}`, 선택지에 대한 softmax. 판정 헤드는 선택지 **위치** 기반 가중치(`linear.weight[:n]`)라 선택지 순서를 고정함.

## 3. 결과

프롬프트 (카테고리명만 치환, 튜닝 없음):
- state: `The image shows a {cat} photographed for industrial visual inspection.`
- question: `Is the object in the image normal, or does it have a defect or anomaly?`

| 카테고리 | N (정상/이상) | **AUROC** | Acc@0.5 | Prec@0.5 | Rec@0.5 | F1@0.5 | F1 최적 임계값 → F1 | 장당 시간 |
|---|---|---:|---:|---:|---:|---:|---|---:|
| bottle | 83 (20/63) | **0.917** | 0.819 | 0.887 | 0.873 | 0.880 | 0.625 → 0.904 | 0.46 s |
| screw | 160 (41/119) | **0.877** | 0.506 | 1.000 | 0.336 | 0.503 | 0.213 → 0.895 | 0.50 s |
| carpet | 117 (28/89) | **1.000** | 1.000 | 1.000 | 1.000 | 1.000 | 0.698 → 1.000 | 0.53 s |

- 장당 시간은 어댑터 호출 end-to-end (이미지 인코딩 CPU 포함). 3카테고리 360장 총 약 3분
- **0.5 임계값은 보정되지 않음**: screw는 정상 평균 0.22 / 이상 평균 0.46이라 0.5에서 재현율 0.34. 카테고리별 임계값 설정 필수
- F1 최적 임계값은 test 세트 자체에서 고른 값 → 낙관적 추정. 실사용 시 별도 검증셋으로 결정할 것

### 결함 유형별 평균 P(Anomalous)

| 카테고리 | good | 결함 유형별 |
|---|---:|---|
| bottle | 0.444 | broken_large 0.868, broken_small 0.906, **contamination 0.625** |
| screw | 0.220 | **scratch_head 0.296**, manipulated_front 0.471, scratch_neck 0.474, thread_side 0.482, thread_top 0.582 |
| carpet | 0.040 | color 0.975, cut 0.974, hole 0.969, metal_contamination 0.952, thread 0.954 |

### 오판 사례 상위 5개 (카테고리별 |label − score| 최대)

| 카테고리 | 파일 | 정답 | 점수 |
|---|---|---|---:|
| bottle | contamination_016.png | 이상 | 0.311 |
| bottle | contamination_020.png | 이상 | 0.361 |
| bottle | good_008.png | 정상 | 0.610 |
| bottle | good_012.png | 정상 | 0.600 |
| bottle | contamination_011.png | 이상 | 0.413 |
| screw | scratch_head_004.png | 이상 | 0.152 |
| screw | scratch_head_000.png | 이상 | 0.155 |
| screw | manipulated_front_018.png | 이상 | 0.162 |
| screw | thread_side_021.png | 이상 | 0.187 |
| screw | thread_side_000.png | 이상 | 0.191 |
| carpet | (@0.5 오판 없음) 최저 이상 점수 thread_014.png | 이상 | 0.698 |

전체 이미지별 점수: `results/{bottle,screw,carpet}.csv`

**경향**: 크고 뚜렷한 결함(파손, 카펫 구멍·오염)은 잘 잡음. 작은 표면 결함(나사 머리 긁힘, 결함 면적 약 0.3%)과 대비가 약한 결함(병 바닥이 살짝 탁한 오염, 면적은 약 15%로 큼)은 정상과 점수가 겹침.

## 4. 권고

| 카테고리 | AUROC | 기준상 권고 |
|---|---:|---|
| bottle | 0.917 | ≥0.90 → 단독 사용 가능 |
| screw | 0.877 (프롬프트 튜닝 후 holdout 0.956, 5.1절) | 0.75~0.90 → **PatchCore 등 전용 모델 + Jev-Omni 2차 판정** (튜닝 후 단독 구간이지만 임계값 여유가 좁음) |
| carpet | 1.000 | ≥0.90 → 단독 사용 가능 |

**종합 권고: 전용 모델(PatchCore 등) 1차 + Jev-Omni 2차 판정 구성.**
- 기준상 2/3 카테고리가 "단독 사용" 구간이지만, 미세 결함 카테고리(screw)가 0.88에 그치고 bottle도 0.92로 경계선. 전용 모델은 MVTec AD image AUROC가 통상 0.98~0.99대로 보고됨
- Jev-Omni는 학습 데이터 없이 텍스트 프롬프트로 판정 기준을 바꿀 수 있다는 점이 강점 → 신규 품목 초기 선별, 전용 모델 경계 사례의 2차 판정, 의미적 이상(부품 누락·잘못된 조립 등)에 적합
- 단독 사용 시에도 카테고리별 임계값 보정은 필수 (0.5 기본값은 screw에서 실패)

### 한계
- 표본이 작음 (카테고리당 83~160장, 정상 20~41장). 1~2장만 바뀌어도 AUROC 수 %p 변동
- 3절 수치는 프롬프트 1종(튜닝 없음) 결과. 튜닝은 screw만 수행 (5.1절), bottle은 미수행
- 데이터는 HF 서드파티 미러 (장수는 공식과 일치하나 공식 파일 해시 대조는 불가)
- README 경고대로 Q4 확률은 FP32 원본과 보정이 다를 수 있음

## 5. 추가 실험

### 5.1 screw 프롬프트 튜닝

과적합 방지를 위해 screw test를 파일명 정렬 후 교대로 dev(정상 21/이상 60)와 holdout(정상 20/이상 59)으로 분할했다. 프롬프트는 dev에서만 고르고, holdout에서 최종 확인했다. 변형 5종은 `tune_prompts.tsv`에 있다.

| 프롬프트 | dev AUROC | holdout AUROC |
|---|---:|---:|
| 기준 (카테고리명만) | 0.862 | 0.901 |
| **v1: 결함 유형 명시** | **0.960** | **0.956** |
| v2: 정상 형태 서술 | 0.889 | – |
| v3: 정상 형태와 결함 유형 서술 | 0.912 | – |
| v4: 부위별 확인 지시 | 0.926 | – |
| v5: 엄격한 검사관 역할 | 0.920 | – |

v1 프롬프트:
- state: `The image shows a metal screw photographed from above for industrial visual inspection. Defects include scratches on the head, damaged or deformed threads, and a manipulated tip.`
- question: `Does this screw have any scratch, thread damage, or deformation?`

holdout 결과:
- AUROC가 0.901에서 **0.956**으로 올랐다. 권고 구간 기준으로 "전용 모델 + 2차"에서 "단독 사용"으로 바뀐다.
- dev에서 고른 임계값 0.983을 holdout에 적용하면 Acc 0.848, Prec 0.873, Rec 0.932, F1 0.902이다.
- **주의:** 점수가 전부 1 근처로 쏠린다(holdout 정상 평균 0.974, 이상 평균 0.990). 순위 성능은 좋아졌지만 임계값 여유가 약 0.02에 불과하다. 조명이나 촬영 조건이 조금만 바뀌어도 임계값이 흔들릴 위험이 크다. 운영 전에 별도 검증셋으로 임계값을 다시 보정해야 하고, 로짓 기준 임계값이나 온도 스케일링도 검토할 만하다.
- 프롬프트 5종을 골라 본 결과라서 holdout 수치도 표본(79장)의 한계가 있다.

### 5.2 최신 llama.cpp (`0bc845d356f437d5ce4fe975c36428f7522829cb`, 2026-09-29)

| 항목 | 검증 리비전 `6e60f356` | 최신 `0bc845d3` |
|---|---|---|
| 빌드 | 성공 | 성공 |
| mmproj GPU (`--no-mmproj-offload` 없음) | NaN, 이후 서버 오염 | **동일하게 NaN**, 서버 오염 |
| mmproj CPU, bottle 점수 | 기준 | 전 이미지 점수 차이 0 (bit 동일), AUROC 0.917 |
| 장당 시간 (bottle, 동일 조건 연속 측정) | 0.468 s | **0.356 s (약 24% 단축)** |
| VRAM (llama-server) | 10,444 MiB | 10,662 MiB (GPU mmproj 로드 상태 측정치) |

결론:
- GPU 인코더 NaN은 최신 빌드에서도 해결되지 않았다. `--no-mmproj-offload`가 여전히 필수이다.
- 최신 빌드는 결과가 동일하고 약 24% 빠르므로 교체할 만하다. 실행 방법은 `LLAMA_DIR=llama.cpp-latest ./restart_server.sh --no-mmproj-offload`이다.
- 현재 기동 중인 서버는 검증 리비전(`--no-mmproj-offload`)으로 되돌려 두었다.

## 6. 1·2단계 개선 실험 (screw, bottle)

프로토콜: test split을 dev/holdout 절반으로 나눔(파일명 정렬 후 교대). 프롬프트·임계값·분류기는 dev에서만 정하고 **holdout 결과로 비교**. 서버는 최신 llama.cpp `0bc845d3`(CPU mmproj)로 실행했고 점수는 검증 리비전과 비트 단위로 동일. 스크립트는 `stage12.py`, `stage2.py`, 결과 CSV는 `results/stage12/`.

| 방식 | screw dev | screw **holdout** | bottle dev | bottle **holdout** | 장당 시간 |
|---|---:|---:|---:|---:|---:|
| S0 기본 질문 | 0.862 | 0.901 | 0.863 | **0.965** | 0.35~0.38 s |
| P1 결함 종류 명시 질문 | 0.960 | 0.956 | 0.881 | 0.935 | 0.37~0.41 s |
| **U 물체 영역 자르기 + 1584px 확대 (+P1)** | 0.971 | **0.981** | 0.919 | 0.945 | 0.75~0.78 s |
| T 2×2 타일 최댓값 (+P1) | 0.886 | 0.909 | 0.981 | 0.942 | 2.0~2.1 s |
| S 보기 순서 바꿔 두 번 평균 (+P1) | 0.963 | 0.959 | 0.888 | 0.926 | 0.72~0.78 s |
| M 다지선다 (정상 + 결함 종류) | 0.695 | 0.814 | 0.813 | 0.894 | 0.38~0.42 s |
| L hidden state + 로지스틱 회귀 (dev 학습) | 0.952 | 0.952 | 0.897 | 0.929 | +0 (특징 재사용) |
| L+ 위와 같음 + train 정상 이미지 추가 | 0.958 | 0.956 | 0.913 | 0.926 | +0 |
| K 정상만 사용한 kNN (train/good 대비 코사인 거리) | 0.771 | 0.816 | 0.903 | 0.871 | +0 |

- **U가 screw에서 가장 좋았다**(0.901 → 0.981). 나사를 잘라 확대하면 긁힘이 더 많은 토큰에 걸린다. holdout 재현율 1.0, 정밀도 0.94.
- **bottle은 어떤 방식도 기본 질문을 확실히 넘지 못했다.** dev/holdout 정상이 각 10장이라 dev와 holdout 차이(예: 기본 0.863 vs 0.965)가 방식 간 차이만큼 크다. 우열 판단 불가.
- **다지선다(M)는 실패**: 결함 종류 정답률이 holdout 기준 screw 25%, bottle 48%. screw는 160장 중 153장을 마지막 보기(끝 변형)로 골랐고, bottle은 83장 모두 결함 보기 중 하나를 골라 '정상'을 한 번도 고르지 않았다.
- **로지스틱 회귀(L)**: AUROC는 최고 프롬프트 방식과 비슷하지만 점수가 0~1에 고르게 퍼진다(프롬프트 방식은 0.95~0.99에 몰림). 기준선을 잡기 쉬워진다는 점이 이점이다.
- **정상만 쓰는 kNN(K)**은 PatchCore처럼 쓰기에 부족했다.
- **실험 중 발견한 편향 (수정 완료)**: 처음 U와 T는 자르기 창이 이미지 밖으로 나가면 검은색으로 채웠다. bottle에서 이 검은 테두리 비율이 정상 19%, 불량 4%로 달랐고, 그 결과 bottle U가 AUROC 1.000이라는 **허위 성능**을 냈다. 창을 이미지 안으로 제한해 다시 측정한 값이 위 표다. 폐기한 결과는 `results/stage12/_padded_v1/`에 있다.

### 앱 반영

- `app.py`(Gradio): 정밀 모드(U), 빠름 모드(S0 또는 P1 질문), 품목 프리셋, 기준선 조절.
- 프리셋 기준선은 dev에서 F1이 최적인 값이다. screw 정밀 0.987, bottle 정밀 0.907. carpet 정밀 모드는 검증하지 않았다.
- 배포: `./run.sh`는 로컬 전용, `--lan`은 사내망, `--share --auth`는 임시 공개. llama-server는 항상 loopback.

## 7. 재현

```bash
cd ~/jev-anomaly
./restart_server.sh --no-mmproj-offload      # 반드시 이 옵션 (NaN 이슈)
.venv/bin/python eval_anomaly.py --data data/screw \
  --state "The image shows a screw photographed for industrial visual inspection." \
  --out results/screw.csv
.venv/bin/python test_eval_anomaly.py        # 지표 함수 자체 검증
```

파일: `eval_anomaly.py`, `test_eval_anomaly.py`, `restart_server.sh`, `docs/NOTES.md`(진행 기록), `results/`, `data/<cat>/{normal,anomaly}` (mvtec_raw 심볼릭 링크)
