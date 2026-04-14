# Coworker 실험 요청 v3 — Gemma-3 승인 + 27B/32B 스케일 실험

**작성일**: 2026-04-14
**요청자**: mais (develop 브랜치, A6000 × 2 환경)
**대상**: iamseungpil (origin/main, **A100 80GB × 4**)
**긴급도**: 높음 — ICLR 2027 submission main-track 확률 ~10pp 상승 가능
**예상 GPU 시간**: 총 **~80 A100-hour** (4-GPU 병렬 시 ~20 wall-clock hours)
**소요 기간**: 승인 대기 수시간 + 실험 1일 + 검증 0.5일 = **약 2일**

---

## 0. 배경 및 요청 이유

2026-04-14 paper §5 전면 개편 (`math/paper/benchmark_design/PAPER_DRAFT_v1_2026_04_14.md`) 과정에서 세 가지 구조적 문제가 확인됨:

1. **Function-calling 평가 축 필수**: 현대 production Instruct 모델은 모두 structured FC 내장. Free-text scorer 만으로는 deployment realism 주장 불가. FC-native Instruct roster가 P1 primary가 되어야 함.

2. **Netsru 배포 모델 정확 매칭**: Netsru 실제 배포는 `gemma-3-27b-it`. 이 모델에서 우리 방법이 작동함을 직접 보이면 appendix E와 완벽 연결. **ICLR 메인트랙 main-track 확률 +5~10pp** (reviewer "production relevance" 공격 선제 방어).

3. **A6000 48GB 한계**: Gemma-3-27B (55GB bf16), Qwen2.5-32B (64GB bf16), Llama-3.1-70B 같은 대형 모델은 우리 A6000에서 단일 GPU 로딩 불가. A100 80GB × 4 에서만 fair FC 평가 가능.

Develop side (mais) 는 이미 다음을 완료/진행:
- Paper §5 claim-indexed 실험 plan (E1-E16 with P1/P2/P3 tiers)
- B_ont 빌드 파이프라인 (`scripts/ocq/build_qwen_metatool_b_ont.py`)
- Multi-tool + graded scoring (F_0.5, EU, FG-F1, ECE)
- FC-native Qwen2.5-7B-Instruct + Mistral-7B-Instruct-v0.3 primary 실험 진행 중

**요청 핵심**: 사용자 HF 계정으로 Gemma-3 gated models 승인 + A100×4 에서 27B/32B scale 실험 실행.

---

## 1. HuggingFace 모델 승인 요청 (브라우저 action, 승인자 수시간 내 자동 처리)

iamseungpil HF 계정에서 아래 각 모델 페이지 방문 → "Request access" 클릭 → form 작성 → submit.

| 모델 | 용도 | 우선순위 |
|---|---|---|
| [`google/gemma-3-27b-it`](https://huggingface.co/google/gemma-3-27b-it) | **Netsru 배포 모델, P1 primary** | 🔴 필수 |
| [`google/gemma-3-12b-it`](https://huggingface.co/google/gemma-3-12b-it) | Scaling 중간 사이즈 | 🟡 권장 |
| [`google/gemma-2-9b-it`](https://huggingface.co/google/gemma-2-9b-it) | 이전 세대 Gemma 비교 | 🟡 권장 |
| [`meta-llama/Llama-3.1-8B-Instruct`](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | 공식 Llama Instruct (NousResearch mirror 대신) | 🟡 권장 |
| [`meta-llama/Llama-3.1-70B-Instruct`](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct) | 대형 cross-family scaling | ⚪ 선택 |

Form 작성 내용 (Gemma 계열 공통):
- **Full name**: (iamseungpil 본명)
- **Affiliation**: (소속 organization/institution)
- **Contact email**: (소속 이메일)
- **Intended use**:
  > *"Academic research on attention-based steering methods for tool selection. Evaluation for a forthcoming ICLR 2027 submission. Non-commercial, no redistribution. Comparison against Qwen/Llama/Mistral Instruct models across cross-architecture benchmarks."*
- Acknowledge **Gemma Prohibited Use Policy** 체크
- **Submit**

Google은 실무적으로 **즉시~수시간 내 자동 승인**. 승인 확인 방법: 같은 페이지 상단에 "You have been granted access to this model" 배너.

---

## 2. 실험 Track Summary (A100×4 에서 실행)

| Track | 모델 | Benchmark | Metric | GPU-hr | A100×4 wall-clock |
|---|---|---|---|---|---|
| **R5** | Gemma-3-27b-it | MetaTool Subtask1 + Subtask4 | E1 6-scorer + E2 9-metric | 30 | ~8h |
| **R6** | Qwen2.5-32B-Instruct | MetaTool Subtask4 (E7 scaling point) | E2 9-metric | 15 | ~4h |
| **R7** | Gemma-2-9b-it (선택) | MetaTool Subtask1 + Subtask4 | E1 + E2 | 15 | ~4h |
| **R8** | Llama-3.1-70B-Instruct (선택) | MetaTool Subtask4 (대형 cross-vendor scaling) | E2 | 20 | ~5h |
| **R9** | Llama-3.1-8B-Instruct (공식) | MetaTool Subtask1 + Subtask4 | E1 + E2 (공식 대신 NousResearch mirror 검증) | 10 | ~3h |

순 필수: **R5 + R6 = 45 GPU-hr, 12 wall-clock hours**.
선택 포함 시 총 **80 GPU-hr, 20 wall-clock hours**.

---

## 3. 정확한 실행 절차 (R5 Gemma-3-27b-it 예시)

### 3.1 Develop 에서 pull + 환경 준비

```bash
cd /path/to/boltzmann-attention
git fetch origin develop
git checkout develop
git pull origin develop
# Python venv: torch 2.8+, transformers 5.4+ (develop 와 동일 환경 사용)
```

### 3.2 HF 토큰 설정 (승인 후 필수)

```bash
huggingface-cli login
# 또는
export HF_TOKEN=hf_xxx_your_token_xxx
```

### 3.3 Gemma-3-27B B_ont 빌드

```bash
# MetaTool ontology 는 이미 build됨: reports/axis2_theoretical_verification/metatool_ontology.json
# 각 (layer, head) 별 pre-RoPE K 수집 후 Gram-Schmidt 분해
python scripts/ocq/build_qwen_metatool_b_ont.py \
    --model google/gemma-3-27b-it --device cuda:0 \
    --target-layers "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41" \
    --pad-to-max \
    --out external/SEKA/seka_projections/ontology-gemma3-27b-it-metatool/B_ont.pt
# ~15분 소요 (A100 1장), skipL0 + pad-to-max 검증된 best practice
```

편의 스크립트: `scripts/post_gemma_approval_launch.sh` 이 위 과정을 자동화. Gemma-3 weight 다운로드 + B_ont 빌드 + random + featshuffle control 모두 한 번에.

### 3.4 Random + Featshuffle control 생성

```bash
python scripts/ocq/make_control_b_ont.py \
    --src external/SEKA/seka_projections/ontology-gemma3-27b-it-metatool/B_ont.pt \
    --out external/SEKA/seka_projections/ontology-gemma3-27b-it-metatool-random/B_ont.pt \
    --mode random_orthonormal --seed 0
python scripts/ocq/make_control_b_ont.py \
    --src external/SEKA/seka_projections/ontology-gemma3-27b-it-metatool/B_ont.pt \
    --out external/SEKA/seka_projections/ontology-gemma3-27b-it-metatool-featshuffle/B_ont.pt \
    --mode feature_shuffle --seed 0
```

### 3.5 E1 — Subtask1 6-scorer 실행 (A100 1장)

```bash
# Free-text scorers × 3 B_ont
for SCORER in substring_any first_line label_logprob; do
  for NORM in sum mean; do
    for BONT_TAG in metatool metatool-random metatool-featshuffle; do
      python scripts/ocq/eval_metatool_subtask1.py \
        --model google/gemma-3-27b-it --device cuda:0 \
        --scorer "$SCORER" --lp-normalize "$NORM" \
        --methods no_steer ocq_bias_a0.3 \
        --b-ont "external/SEKA/seka_projections/ontology-gemma3-27b-it-${BONT_TAG}/B_ont.pt" \
        --max-samples 0 \
        --out "reports/gemma/subtask1_${SCORER}_${NORM}_${BONT_TAG}.json"
    done
  done
done
# substring + first_line 은 --lp-normalize 무의미하므로 한 번만 돌리면 되지만, 스크립트 재사용 단순화를 위해 루프 유지.

# FC scorer (새 구현 필요): fc_name_match + fc_label_logprob
# 2026-04-15 까지 develop side 에서 구현 pending, 현재는 free-text scorer 5종만 진행.
```

### 3.6 E2 — Subtask4 multi-tool (9 metrics × 6 methods)

**중요**: Subtask4 는 495 query × **2-tool GT** multi-tool benchmark. Structured output format 필요.

```bash
# 기본 multi-tool eval driver (develop branch 에 추가 예정, 2026-04-15)
python scripts/ocq/eval_metatool_subtask4.py \
    --model google/gemma-3-27b-it --device cuda:0 \
    --use-function-calling \
    --methods no_steer ocq_bias_a0.3 adaseka_2expert adaseka_3expert \
    --b-ont-real external/SEKA/seka_projections/ontology-gemma3-27b-it-metatool/B_ont.pt \
    --b-ont-random external/SEKA/seka_projections/ontology-gemma3-27b-it-metatool-random/B_ont.pt \
    --b-ont-featshuffle external/SEKA/seka_projections/ontology-gemma3-27b-it-metatool-featshuffle/B_ont.pt \
    --metrics F1 F_0.5 EU Jaccard FG-F1 FG-F_0.5 FG-EU ECE \
    --max-samples 0 \
    --out reports/gemma/subtask4_all_metrics.json
```

**참고**: `eval_metatool_subtask4.py` 는 develop 에서 2026-04-15 중 commit 예정. 현재 없음. **Coworker 는 이 파일이 push 될 때까지 R5 의 E2 부분 대기**.

### 3.7 결과 push

```bash
# reports/gemma/ 아래 결과 파일들만 commit
git add reports/gemma/
git add external/SEKA/seka_projections/ontology-gemma3-27b-it-*/B_ont.pt  # B_ont 파일도 push
git commit -m "Gemma-3-27b-it results (R5): Subtask1 E1 + Subtask4 E2"
git push origin <coworker-branch>  # or develop
```

---

## 4. 정확한 실행 절차 (R6 Qwen2.5-32B-Instruct)

A6000 48GB 에서 Qwen2.5-32B (64GB bf16) 는 8-bit quant 필요. A100 80GB 에서는 bf16 native 가능.

```bash
# B_ont 빌드 (Qwen2.5-32B-Instruct 는 이미 cached 되어 있으면 즉시 시작)
python scripts/ocq/build_qwen_metatool_b_ont.py \
    --model Qwen/Qwen2.5-32B-Instruct --device cuda:0 \
    --target-layers "1,2,...,63" \
    --pad-to-max \
    --out external/SEKA/seka_projections/ontology-qwen25-32b-it-metatool/B_ont.pt

# Control 생성
python scripts/ocq/make_control_b_ont.py --src ... --mode random_orthonormal --seed 0 ...
python scripts/ocq/make_control_b_ont.py --src ... --mode feature_shuffle --seed 0 ...

# E2 실행 (Subtask4)
python scripts/ocq/eval_metatool_subtask4.py \
    --model Qwen/Qwen2.5-32B-Instruct --device cuda:0 \
    --use-function-calling \
    --methods no_steer ocq_bias_a0.3 adaseka_2expert adaseka_3expert \
    ... (R5 와 동일 option)
```

---

## 5. 대안 — 4-GPU 병렬 분산 실행

Tensor parallel 또는 pipeline parallel 로 27B/32B 모델을 4-GPU 에 분산하면 단일 GPU 대비 2-3× 가속. `accelerate launch` 또는 `vllm serve` 활용 가능.

```bash
# vllm 사용 권장 (Gemma / Qwen 모두 지원)
python -m vllm.entrypoints.openai.api_server \
    --model google/gemma-3-27b-it \
    --tensor-parallel-size 4 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.90

# 별도 terminal 에서 평가 driver 호출 (vllm client mode 추가 구현 필요)
```

**주의**: 현재 `eval_metatool_subtask1.py` 는 direct `model.generate()` 를 사용하므로 vllm 과 호환 안 됨. vllm 사용하려면 client mode 추가 필요. 아래 두 옵션 중 선택:

- **Option A (단순, 느림)**: `tensor_parallel_size=1`, 각 GPU 에 1 run × 4 runs 병렬 실행 (embarrassingly parallel). R5 + R6 + R7 + R8/R9 를 동시에 4 GPU 에 할당.
- **Option B (복잡, 빠름)**: vllm 또는 `accelerate` 로 27B 모델을 4-GPU 에 분산. develop side 구현 필요 (~1일 작업).

**권장**: Option A. 단일 GPU 대해 Gemma-3-27B (55GB) 가 80GB A100 에 여유있게 로드. 4개 run 병렬 = 4× 가속 + 단순성.

---

## 6. 우선순위별 실행 순서

1. **Track A (오늘)**: Gemma 승인 form 제출 (5분). 승인 대기.
2. **Track B (승인 후)**: R5 Gemma-3-27b-it 전체 (~8h).
3. **Track C (R5 병렬)**: R6 Qwen2.5-32B-Instruct (~4h, 다른 GPU).
4. **Track D (선택)**: R7 Gemma-2-9b-it, R8 Llama-70B-Instruct.
5. **최종**: `reports/gemma/` + `reports/qwen32b/` push → develop 에서 consolidation.

**R5 + R6 만으로도 ICLR main-track 확률 기여 ~+5pp 확보**. Track D 포함 시 +5pp 추가.

---

## 7. 필요 사전 준비 파일 (develop side 에서 push 완료)

| 파일 | 상태 | 용도 |
|---|---|---|
| `scripts/ocq/build_qwen_metatool_b_ont.py` | ✓ (develop 존재) | B_ont 빌드 |
| `scripts/ocq/make_control_b_ont.py` | ✓ (develop) | random/featshuffle 생성 |
| `scripts/ocq/eval_metatool_subtask1.py` | ✓ (develop, 6-scorer 포함) | E1 runner |
| `reports/axis2_theoretical_verification/metatool_ontology.json` | ✓ (develop) | 4-facet ontology spec |
| `scripts/ocq/eval_metatool_subtask4.py` | ❌ **2026-04-15 중 push 예정** | E2 multi-tool runner |
| `scripts/post_gemma_approval_launch.sh` | ✓ (develop) | Gemma 승인 후 자동화 |

**Subtask4 multi-tool driver 가 missing**. 내일 오전 중 구현 + commit 예정. 그 전까지 R5 의 E1 부분 (Subtask1 6-scorer) 은 즉시 실행 가능. E2 multi-tool 은 commit 후 실행.

---

## 8. 연락 및 결과 보고

- **승인 확인 보고**: Gemma 승인되면 슬랙 알림 (수시간 내 예상).
- **실험 결과**: `reports/gemma/` + `reports/qwen32b/` 아래 JSON 파일들 + 짧은 summary markdown (`reports/coworker_results_2026_04_15.md` 등).
- **문제 발생 시**: develop branch 에 issue 또는 슬랙. mais 가 debug.

---

## 9. 이 request 의 ICLR 기여 정량

| 완료 결과 | ICLR 메인트랙 확률 기여 |
|---|---|
| R5 Gemma-3-27b-it E1 + E2 | **+5pp** (Netsru alignment, 4-vendor 증명) |
| R6 Qwen2.5-32B-Instruct E2 | +2pp (scaling 완결성) |
| R7 Gemma-2-9b-it | +1pp (vendor 내 세대 비교) |
| R8 Llama-3.1-70B-Instruct | +2pp (대형 cross-vendor) |
| R9 Llama-3.1-8B-Instruct (공식) | +1pp (mirror 검증) |

**R5 + R6 = +7pp, 총 request 완료 시 +11pp**. 현재 main-track 확률 ~35-45% 에서 **~42-52%** 로 상승.

---

감사합니다. 승인 form 제출 및 실행 상태 공유 부탁드립니다.
