# Coworker 실험 요청 v4 — A100×4 최대 활용 (2026-04-15 갱신)

**작성일**: 2026-04-14 v1-v3 → **2026-04-15 01:30 KST v4 전면 개편**
**요청자**: mais (develop 브랜치, A6000 × 2 환경)
**대상**: iamseungpil (origin/main, **A100 80GB × 4**)
**긴급도**: 매우 높음 — NeurIPS 2026 (5월 15일 deadline) 또는 ICLR 2027 submission 핵심 실험
**예상 A100-hour**: 총 **~70 A100-hour** (4-GPU 병렬 시 **~18 wall-clock hours**)
**소요 기간**: **승인 대기 0.5일 + 실험 1일 (20 wall-clock hours on A100×4) = 약 2일**

---

## 0. v4 업데이트 배경 (2026-04-15)

v3 작성 이후 새로운 핵심 발견:

1. **Subtask4 regression (−4.6pp F1 full 497)**: training-free K-bias 가 multi-tool 에서 over-generalization 으로 performance degrades. Cor 6.9 downstream accuracy-lift signature 실패.

2. **E4 SVD Cor 6.9 operator-level 검증 성공 (24.0 vs 7.44)**: theorem 자체는 empirical 검증됨.

3. **3-family strict-scorer positive (Qwen +0.10 / Llama +6.33 / Mistral-v03 +3.12)**: cross-model Subtask1 mechanism-specificity 확인.

4. **Thm 6.16 (LoRA + Rotation Hybrid) 제안**: training-light extension. LoRA r=8 on q/k/v_proj + post-LoRA B_ont rebuild + rotation. Subtask4 F1 0.82-0.92 예상 (vs 0.685 baseline).

5. **Non-uniform fix options (Y3 normalized, Y4 contrastive)**: Cor 6.9.4 over-generalization 해결책, Thm 6.9.5 증명됨. N=20 smoke 진행 중.

**v4의 핵심 차이점**: A100×4 에서 **embarrassingly parallel 4-stream** 전략. **LoRA L1 × 3 models + Gemma 동시 실행**으로 wall-clock 대폭 단축.

---

## 1. HuggingFace 모델 승인 요청 (브라우저 action, 승인자 수시간 내 자동 처리)

v3 에서 변경 없음. iamseungpil HF 계정에서 각 모델 페이지 방문 → "Request access" 클릭.

| 모델 | 용도 | 우선순위 |
|---|---|---|
| [`google/gemma-3-27b-it`](https://huggingface.co/google/gemma-3-27b-it) | **Netsru 배포 모델, P1 primary** | 🔴 필수 |
| [`google/gemma-3-12b-it`](https://huggingface.co/google/gemma-3-12b-it) | Scaling 중간 사이즈 | 🟡 권장 |
| [`google/gemma-2-9b-it`](https://huggingface.co/google/gemma-2-9b-it) | 이전 세대 Gemma 비교 | 🟡 권장 |
| [`meta-llama/Llama-3.1-8B-Instruct`](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | 공식 Llama Instruct (NousResearch mirror 대신) | 🟡 권장 |

Form 내용 (Gemma 공통):
- **Intended use**: *"Academic research on attention-based steering methods for tool selection. Evaluation for NeurIPS 2026 / ICLR 2027 submission. Non-commercial, no redistribution. Comparison against Qwen/Llama/Mistral Instruct models across cross-architecture benchmarks."*
- Gemma Prohibited Use Policy 체크

승인 후 HF 토큰 설정: `huggingface-cli login` 또는 `export HF_TOKEN=hf_xxx`

---

## 2. A100×4 Parallel Track 구조 (18 wall-clock hours)

**4개 A100을 4개 독립 track으로 embarrassingly parallel 사용**:

```
┌───────────────────────────────────────────────────────────────────┐
│ Track A (A100 #0): Gemma-3-27b-it full pipeline                  │
│ Track B (A100 #1): LoRA L1×3 models + L3 rotation                │
│ Track C (A100 #2): Scaling curve Qwen2.5 {3, 14, 32}B-Instruct   │
│ Track D (A100 #3): Baselines (CAA/ITI/PASTA/ASA/FocusDir) + E8   │
└───────────────────────────────────────────────────────────────────┘
```

**병렬 실행으로 wall-clock ~70 / 4 = ~18h**.

### Track A — Gemma-3-27b-it (A100 #0, ~15h wall-clock)

**준비물 (Gemma 승인 후 즉시)**:
```bash
cd /path/to/boltzmann-attention
git pull origin develop
source <env>  # torch 2.8+, transformers 5.4+
huggingface-cli login  # HF_TOKEN 필요
```

**1단계 — B_ont build (15분)**:
```bash
python scripts/ocq/build_qwen_metatool_b_ont.py \
    --model google/gemma-3-27b-it --device cuda:0 \
    --target-layers "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41" \
    --pad-to-max \
    --out external/SEKA/seka_projections/ontology-gemma3-27b-it-metatool/B_ont.pt
```

**2단계 — Control basis 생성 (1분)**:
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

**3단계 — Subtask1 label_logprob × 3 B_ont × sum/mean (6 runs, ~6h)**:
```bash
for SCORER_NORM in sum mean; do
  for BONT_TAG in metatool metatool-random metatool-featshuffle; do
    python scripts/ocq/eval_metatool_subtask1.py \
      --model google/gemma-3-27b-it --device cuda:0 \
      --scorer label_logprob --lp-normalize "$SCORER_NORM" \
      --methods no_steer ocq_bias_a0.3 \
      --b-ont "external/SEKA/seka_projections/ontology-gemma3-27b-it-${BONT_TAG}/B_ont.pt" \
      --max-samples 0 \
      --out "reports/gemma/subtask1_${SCORER_NORM}_${BONT_TAG}.json"
  done
done
```

**4단계 — Subtask4 full 497 × 3 B_ont (3 runs, ~6h)**:
```bash
for BONT_TAG in metatool metatool-random metatool-featshuffle; do
  python scripts/ocq/eval_metatool_subtask4.py \
    --model google/gemma-3-27b-it --device cuda:0 \
    --methods no_steer ocq_bias_a0.3 \
    --b-ont "external/SEKA/seka_projections/ontology-gemma3-27b-it-${BONT_TAG}/B_ont.pt" \
    --max-samples 0 \
    --out "reports/gemma/subtask4_${BONT_TAG}.json"
done
```

**5단계 — Gemma LoRA synergy smoke (Thm 6.16, ~3h)**:
```bash
python scripts/ocq/lora_train_metatool.py \
    --base-model google/gemma-3-27b-it --device cuda:0 \
    --train-dataset /tmp/MetaTool/dataset/tmp_dataset/Task2-Subtask1.json \
    --train-size 500 --epochs 3 --lora-r 8 \
    --lora-target q_proj k_proj v_proj \
    --batch-size 1 --lr 1e-4 \
    --out-dir lora_adapters/gemma3_27b_subtask1_r8
# L2 build + L3 eval via manual script (mirror scripts/run_lora_hybrid_pipeline.sh for Gemma)
```

**Track A 총합**: ~15h wall-clock.

### Track B — LoRA L1×3 models + L3 rotation (A100 #1, ~12h wall-clock)

**목적**: Thm 6.16 LoRA hybrid 검증. 3 모델 동시 학습 (같은 A100 에서 순차) or 다른 track에 분산.

Sequential on single A100:
```bash
# L1-A: Qwen2.5-7B-Instruct LoRA (~3h)
python scripts/ocq/lora_train_metatool.py \
    --base-model Qwen/Qwen2.5-7B-Instruct --device cuda:1 \
    --train-dataset /tmp/MetaTool/dataset/tmp_dataset/Task2-Subtask1.json \
    --train-size 500 --val-size 50 --epochs 3 --lora-r 8 --lr 1e-4 --batch-size 4 \
    --lora-target q_proj k_proj v_proj \
    --out-dir lora_adapters/qwen25_7b_subtask1_r8

# L1-B: NousResearch/Meta-Llama-3.1-8B-Instruct LoRA (~4h)
python scripts/ocq/lora_train_metatool.py \
    --base-model NousResearch/Meta-Llama-3.1-8B-Instruct --device cuda:1 \
    ... (same args, out-dir = lora_adapters/llama31_8b_it_subtask1_r8)

# L1-C: Mistral-7B-Instruct-v0.3 LoRA (~3h)
python scripts/ocq/lora_train_metatool.py \
    --base-model mistralai/Mistral-7B-Instruct-v0.3 --device cuda:1 \
    ... (same args, out-dir = lora_adapters/mistral7b_it_subtask1_r8)
```

**L2 (per model): LoRA-adapted B_ont rebuild** (구현 필요 — 다음 script 제공):
```bash
# scripts/ocq/build_lora_adapted_b_ont.py (to be written)
# Load base + LoRA adapter, merge_and_unload, build B_ont via standard pipeline
python scripts/ocq/build_lora_adapted_b_ont.py \
    --base Qwen/Qwen2.5-7B-Instruct \
    --lora-adapter lora_adapters/qwen25_7b_subtask1_r8 \
    --out external/SEKA/seka_projections/ontology-qwen25-7b-it-lora-r8-metatool/B_ont.pt
# (repeat for Llama, Mistral)
```

**L3: Subtask4 N=497 eval with LoRA-adapted model + B_ont' + K-bias** (~1h per model):
```bash
# LoRA-adapted Subtask4 evaluation variant (configurable via new flag in eval_metatool_subtask4)
python scripts/ocq/eval_metatool_subtask4_lora.py \
    --base-model Qwen/Qwen2.5-7B-Instruct --lora-adapter lora_adapters/qwen25_7b_subtask1_r8 \
    --device cuda:1 \
    --methods no_steer ocq_bias_a0.3 ocq_normbias_a0.3 \
    --b-ont-base external/SEKA/seka_projections/ontology-qwen25-7b-metatool/B_ont.pt \
    --b-ont-lora external/SEKA/seka_projections/ontology-qwen25-7b-it-lora-r8-metatool/B_ont.pt \
    --max-samples 0 \
    --out reports/lora_hybrid/subtask4_qwen_lora.json
```

**Track B 총합**: ~12h wall-clock.

### Track C — Scaling curve Qwen2.5 {3, 14, 32}B-Instruct (A100 #2, ~15h)

32B 는 A100 80GB 에 native bf16 로드 가능. 14B / 3B / 7B 는 여유롭게 실행.

```bash
for MODEL_TAG in "0.5B" "3B" "14B" "32B"; do
  MODEL="Qwen/Qwen2.5-${MODEL_TAG}-Instruct"
  
  # Build B_ont (~15min for 32B, less for smaller)
  python scripts/ocq/build_qwen_metatool_b_ont.py \
      --model "$MODEL" --device cuda:2 \
      --target-layers "1,...,L-1" --pad-to-max \
      --out "external/SEKA/seka_projections/ontology-qwen25-${MODEL_TAG}-it-metatool/B_ont.pt"
  
  # Subtask4 full 497 × no_steer + a0.3 (~2h for 32B)
  python scripts/ocq/eval_metatool_subtask4.py \
      --model "$MODEL" --device cuda:2 \
      --methods no_steer ocq_bias_a0.3 ocq_normbias_a0.3 \
      --b-ont "external/SEKA/seka_projections/ontology-qwen25-${MODEL_TAG}-it-metatool/B_ont.pt" \
      --max-samples 0 \
      --out "reports/scaling/subtask4_qwen25_${MODEL_TAG}.json"
done
```

**Track C 총합**: ~15h (32B dominates).

### Track D — Baselines + Safety (A100 #3, ~10h)

**Baselines (CAA, ITI, PASTA, ASA, Focus Directions)** on Qwen-Instruct + Subtask1 + Subtask4:
- Develop side has partial implementation; coworker may need to reproduce from original papers or use implementations in `baselines/` directory (pending push).
- Each baseline: ~1h per benchmark per model.

**Safety retention (MMLU + HH-RLHF + ToxiGen)** on Qwen-Instruct + soft-facet-gated α=0.3:
- MMLU 1000 samples: ~30min
- HH-RLHF 500: ~30min
- ToxiGen 500: ~30min

Total: ~10h.

---

## 3. 실행 우선순위

**Tier 1 (main submission 에 필수, 하루 내 완료)**:
1. Track A Gemma (Netsru alignment) — **가장 큰 점수 기여**
2. Track B LoRA (Thm 6.16 검증) — **Subtask4 regression 해결 가능성**

**Tier 2 (reviewer defense)**:
3. Track C Scaling (32B 포함)
4. Track D Baselines (CAA/ASA/PASTA 재현)

**Tier 3 (stretch)**:
5. LoRA L1 다른 domain (ToolAlpaca, tau2) 로 일반화 test
6. BFCL-v3 Parallel 접근 가능 시 실행

**우선순위 판단**: Tier 1 2개 track 을 동시 실행. 두 개 A100 으로 Gemma + LoRA Qwen/Mistral 병렬. Tier 2 추가 가능.

---

## 4. develop side 에서 push 필요 파일

### 완료 (already on develop)
- `scripts/ocq/eval_metatool_subtask1.py` (+label_logprob + gate_mode + vbias + normbias + cbias)
- `scripts/ocq/eval_metatool_subtask4.py` (multi-tool FC driver)
- `scripts/ocq/build_qwen_metatool_b_ont.py`
- `scripts/ocq/make_control_b_ont.py`
- `scripts/ocq/lora_train_metatool.py` (신규 L1 training)
- `scripts/post_gemma_approval_launch.sh` (Gemma 자동화)
- `scripts/run_lora_hybrid_pipeline.sh` (L1-L3 chain, 로컬 테스트용)

### 2026-04-15 중 push 예정
- `scripts/ocq/build_lora_adapted_b_ont.py` (L2 LoRA-adapted B_ont builder) — **coworker 가 Track B 시작 전 필요**
- `scripts/ocq/eval_metatool_subtask4_lora.py` (LoRA + B_ont' 통합 driver) — **coworker 가 L3 실행 전 필요**
- Baseline implementations (CAA/ITI/PASTA/ASA) — 하루 내 commit 목표

---

## 5. 결과 푸시 프로토콜

각 Track 완료 후 즉시 push:
```bash
git add reports/gemma/ reports/scaling/ reports/lora_hybrid/ reports/baselines/ reports/safety/
git add external/SEKA/seka_projections/ontology-*/B_ont.pt
git add lora_adapters/*/adapter_config.json lora_adapters/*/adapter_model.safetensors
git commit -m "A100×4 Track {A,B,C,D} results (v4 coworker request)"
git push origin develop  # or coworker branch
```

**결과 파일 구조**:
```
reports/
├── gemma/               (Track A Gemma-3-27b-it)
│   ├── subtask1_sum_metatool.json
│   ├── subtask1_mean_metatool.json
│   ├── subtask4_metatool.json
│   └── lora_hybrid_subtask4.json
├── lora_hybrid/         (Track B LoRA × 3 models)
│   ├── subtask4_qwen_lora.json
│   ├── subtask4_llama_lora.json
│   └── subtask4_mistral_lora.json
├── scaling/             (Track C Qwen 0.5-32B)
│   ├── subtask4_qwen25_0.5B.json
│   ├── subtask4_qwen25_3B.json
│   ├── subtask4_qwen25_14B.json
│   └── subtask4_qwen25_32B.json
├── baselines/           (Track D CAA/ITI/PASTA/ASA/FocusDir)
│   ├── caa_subtask1_qwen7b.json
│   ├── iti_subtask1_qwen7b.json
│   └── ...
└── safety/
    ├── mmlu_soft_gate.json
    ├── hh_rlhf_soft_gate.json
    └── toxigen_soft_gate.json
```

---

## 6. 예상 결과 및 ICLR/NeurIPS 기여

| Track | 예상 결과 | 점수 기여 (누적) |
|---|---|---|
| A Gemma-3-27b-it 성공 (Subtask1 +3pp, Subtask4 +8pp) | Netsru alignment + 4-vendor | +0.5 |
| B LoRA Qwen Subtask4 F1 > 0.82 | Thm 6.16 empirical 검증 | +0.4 |
| B LoRA Llama + Mistral 일관 결과 | Cross-model LoRA synergy | +0.2 |
| C Scaling curve 32B 포함 clean | Architecture-invariant 증명 | +0.3 |
| D Baselines 재현 매칭 | Reviewer defense | +0.2 |
| D Safety retention clean | Mandatory appendix | +0.1 |

**누적 기여**: **+1.7점** (주요 목표 중 모든 track 성공 시)

현재 점수 **6.0-6.2** → 목표 점수 **7.7-7.9**
- **NeurIPS 2026 main-track 확률: ~55% → 75%**
- **ICLR 2027 main-track 확률: ~60% → 80%**

---

## 7. 긴급 연락

- **Gemma 승인 즉시 알림**: HF 승인되면 develop 에 알림 (별도 doc 또는 슬랙)
- **Track 진행 중 문제**: develop 에 `reports/coworker_blocker_2026_04_15.md` 생성
- **중간 결과 공유**: 각 Track 1 단계 완료 시 JSON commit + summary markdown

---

## 8. v3 대비 변경 사항

| 항목 | v3 | v4 |
|---|---|---|
| A100×4 활용 | Embarrassingly parallel 4-run (Option A) | **명시적 4-Track 구조 (A/B/C/D)** |
| LoRA + Rotation (Thm 6.16) | 없음 | **Track B 신규 추가 (Subtask4 rescue)** |
| Llama-Instruct 공식 | 옵션 | **Track B에서 NousResearch Instruct mirror 사용** |
| Baselines | 별도 없음 | **Track D 에 CAA/ITI/PASTA/ASA/FocusDir** |
| Scaling | 32B만 | **Qwen2.5 full curve {0.5, 3, 7, 14, 32}B** |
| 필요 구현 | eval_metatool_subtask4.py missing | **+build_lora_adapted_b_ont.py + eval_lora variants** (2026-04-15 중 push) |
| 시간 | 80 GPU-hr = 20h wall-clock | **70 A100-hr = 18h wall-clock (4-track parallel)** |
| 점수 기여 | +7-11pp | **+17pp (main-track 확률 +20-25%)** |

---

감사합니다. v4 승인 + Gemma HF form 제출 부탁드립니다. 진행 상태 수시 공유 바랍니다.
