# Coworker 실험 요청 — Cross-model MetaTool Subtask1 검증

**작성일**: 2026-04-10
**요청자**: mais (develop 브랜치)
**대상**: iamseungpil (origin/main, A100 80GB × 4)
**긴급도**: 높음 — Phase B paper direction 결정이 이 결과에 의존
**예상 GPU 시간**: 5-6시간 (4장 병렬 활용 시)

---

## 1. 배경 (30초 요약)

2026-04-09 저녁, Qwen2.5-7B / MetaTool Subtask1 (995 queries) 전체에서 K-bias α sweep kill-switch PASS:

| Method | top-1 | Δ vs no_steer |
|---|---|---|
| no_steer | 75.58% (752/995) | baseline |
| ocq_bias α=0.20 | 84.02% | +8.44 pp |
| ocq_bias α=0.25 | 65.23% | **−10.35 pp** (dip) |
| ocq_bias α=0.30 | **86.73%** | **+11.15 pp** ✅ peak |
| ocq_bias α=0.35 | 83.12% | +7.54 pp |
| ocq_bias α=0.40 | 73.37% | −2.21 pp |

**문제점 2가지**:
1. Qwen 단일 모델만 검증됨. Llama / Mistral이 재현되지 않으면 paper는 "Qwen-specific observation"으로 격하.
2. α=0.25 dip은 이론적으로 설명되지 않음 (mais 쪽에서 dip 분석 병행 중).

**그래서 필요한 것**: Llama-3.1-8B, Mistral-7B-v0.3 에서 동일 α sweep. **이것이 Week 1 kill-switch stage 2**이며, 이 결과로 paper direction 이 확정됩니다.

Phase 1.x에서 **Mistral은 이미 한 번 negative**였음 (memory/phase1_3_ontology_beats_seka.md 참조). Catalog-ontology 방식이 그 문제를 해결하는지 이번 실험이 결정적 증거가 됨.

---

## 2. 요청 실험 (2개 모델, 완전 독립)

### Task A: Llama-3.1-8B B_ont build + α sweep

**Step A.1 — B_ont build** (A100 1장, ~10분)

```bash
source /home/woori/workspace_common/CDP/poc/set.env && \
python scripts/ocq/build_qwen_metatool_b_ont.py \
    --model meta-llama/Meta-Llama-3.1-8B \
    --dataset /tmp/MetaTool/dataset/tmp_dataset/Task2-Subtask1.json \
    --metatool-ontology reports/axis2_theoretical_verification/metatool_ontology.json \
    --out external/SEKA/seka_projections/ontology-llama31-8b-metatool/B_ont.pt \
    --device cuda:0
```

**주의**: 현재 `build_qwen_metatool_b_ont.py`는 Qwen hardcoded 여부 확인 필요. 스크립트가 `--model` 인자를 안 받으면 Llama/Mistral용으로 복제해서 수정 필요 (Qwen 전용 로직이 있다면 알려주세요 — mais가 일반화 패치 작성).

**Step A.2 — α sweep eval** (A100 1장 순차 또는 4장 병렬, ~30분)

```bash
source /home/woori/workspace_common/CDP/poc/set.env && \
python scripts/ocq/eval_metatool_subtask1.py \
    --model meta-llama/Meta-Llama-3.1-8B \
    --device cuda:0 \
    --methods no_steer ocq_bias_a0.2 ocq_bias_a0.25 ocq_bias_a0.3 ocq_bias_a0.35 ocq_bias_a0.4 \
    --b-ont external/SEKA/seka_projections/ontology-llama31-8b-metatool/B_ont.pt \
    --out /tmp/metatool_FULL995_llama31_8b_alpha_sweep.json
```

### Task B: Mistral-7B-v0.3 B_ont build + α sweep

위와 동일, model_name만 `mistralai/Mistral-7B-v0.3` 으로 교체. Output path `ontology-mistral-7b-v03-metatool/` 및 `/tmp/metatool_FULL995_mistral_7b_v03_alpha_sweep.json`.

### 4장 병렬 활용 팁

두 Task가 완전 독립이므로:
- cuda:0 → Llama B_ont build + eval
- cuda:1 → Mistral B_ont build + eval
- cuda:2, cuda:3 → 각 모델의 추가 α 값 병렬 (예: α=0.45, 0.15) 또는 다른 시드로 variance 측정

이러면 총 소요 시간이 ~3시간 이내로 축소됩니다.

---

## 3. 판정 기준 (결과 해석)

결과 JSON 받으면 mais가 분석하겠지만, coworker 본인도 미리 판정 가능하도록:

### PASS 조건 (각 모델 독립)
- no_steer 대비 **α=0.2, 0.3, 0.35 중 최소 1개**에서 **≥ +5pp lift** 가 나오면 해당 모델 PASS

### 전체 판정
- Llama PASS + Mistral PASS → **Phase B paper direction 확정**, X+Y+Z 10일 plan 정당화
- Llama PASS + Mistral FAIL → "Qwen+Llama-family observation"으로 scope 축소, Mistral 별도 섹션
- Llama FAIL + Mistral PASS → 특이한 결과, 조사 필요
- 둘 다 FAIL → **Qwen-specific scoping**, 논문 ambition 축소 (workshop/short)

### 추가 관찰 포인트
- **α curve 형태**: Qwen에서 α=0.25 dip이 있었음. Llama/Mistral에서도 dip이 있는지, 있다면 어느 α에서? 동일 위치(0.25)면 **구조적 현상**, 다른 위치면 **model-specific calibration artifact**.
- **no_steer baseline**: Qwen 75.58%. Llama/Mistral baseline 자체도 중요한 수치 (7B-8B 모델의 MetaTool 성능 기준선).

---

## 4. Deliverables

Coworker가 mais 쪽에 돌려줄 것:

1. `/tmp/metatool_FULL995_llama31_8b_alpha_sweep.json`
2. `/tmp/metatool_FULL995_mistral_7b_v03_alpha_sweep.json`
3. B_ont 파일 2개 (`external/SEKA/seka_projections/` 아래)
4. 각 build 로그에서 `r_per_pair`, `r_median`, `n_skipped` 숫자 (Qwen은 r_median=28 이었음, cross-model 비교용)
5. 간단한 메모: 어떤 문제가 있었다면 (OOM, tokenizer issue 등)

---

## 5. 참고 파일 (코드 + 결과 재현)

**코드** (develop branch에 있음, origin/main과 다를 수 있음 — 필요 시 cherry-pick 또는 branch switch):
- `scripts/ocq/build_qwen_metatool_b_ont.py` — B_ont builder
- `scripts/ocq/build_metatool_ontology.py` — ontology facet extraction (catalog → facet sentences)
- `scripts/ocq/eval_metatool_subtask1.py` — eval driver
- `reports/axis2_theoretical_verification/metatool_ontology.json` — 4 facet 정의 (pre-built)

**Qwen 원본 결과** (비교용):
- `/tmp/metatool_FULL995_alpha_sweep_cuda0.json`
- `/tmp/metatool_FULL995_ablations_cuda1.json`

**관련 메모리**:
- `memory/metatool_subtask1_first_signal_2026_04_09.md` — 원래 smoke 결과 (50 샘플 기준)
- `memory/phase1_3_ontology_beats_seka.md` — Mistral negative 이력
- `memory/phase_b_tool_selection_plan.md` — 전체 Phase B 계획
- `memory/session_failure_mode_2026_04_10.md` — 오늘 추가된 feedback (speculative plan stacking 주의)

**원본 dataset**:
- `/tmp/MetaTool/dataset/tmp_dataset/Task2-Subtask1.json` (995 queries, 199 unique tools)

---

## 6. 왜 이것이 우선인가 (session context)

2026-04-10 세션에서 Qwen 결과 기반으로 X+Y+Z 3-axis paper plan(10일)이 설계되었으나, 모두 **"Qwen 결과가 다른 모델로 transfer된다"는 가정**에 의존. 이 가정을 검증하지 않고 10일 투자는 부적절하다는 판단.

`memory/session_failure_mode_2026_04_10.md` 의 원칙: "Each stage must empirically gate the next." Cross-model 검증이 gate.

이 결과가 나오기 전까지 mais 쪽은 다음 작업에만 집중:
1. α=0.25 dip 원인 분석 (현재 실행 중)
2. Self-gated C prototype 설계 (구현은 gate 통과 후)
3. 문서 정리 (PHASE_B_PAPER_PLAN_v1 retraction section 초안)

Cross-model 결과가 도착하면 즉시 판정하고 다음 단계 결정합니다.

---

**질문이나 blocker 있으면 바로 알려주세요.** 특히 build 스크립트가 Qwen hardcoded인지, A100 4장의 memory 여유가 충분한지 (7B-8B 모델은 1장으로 충분할 것이나 확인 필요).
