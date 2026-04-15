# Paper Framing Debate — 2026-04-16 새벽

**대상**: Coworker (A100×4 보유)
**목적**: Q-only pivot vs 현재 Q+K pair framing 결정을 위한 의견 수렴
**Deadline context**: NeurIPS 2026 sprint D-28/29, paper draft at locked 6.20

---

## 1. Core question

> **"K-bias (SEKA 와 동일 form) 를 main accuracy contribution 으로 유지할 것인가, Q-side 를 primary novel contribution 으로 승격할 것인가?"**

User 의 전략적 관찰:
> "K-bias 가 SEKA 랑 같은 거니까. Q only 는 우리 제안이니까."

핵심: SEKA $k' = k + gPk$ 와 우리 K-bias $K + \alpha BB^\top K$ 는 구조적으로 같은 form. Paper §2.1 에서 이미 closest prior art 로 인정.

---

## 2. 오늘 밤샘 (04-16 00:00–02:00) 핵심 empirical findings

### Finding 1 — Thm 6.17 V-inert 는 K-basis-on-V artifact (확정)

**진단 실험** (`/tmp/basis_overlap.py`, `/tmp/vbias_basis_ab_test.py`):

| Metric | Qwen | Llama | 의미 |
|---|---|---|---|
| V-K subspace top principal SV | 0.83 | 0.79 | primary direction 만 overlap (language-mean 추정) |
| V-K per-column mean cosine | 0.07 | 0.07 | facet-specific directions **near-orthogonal** |
| V-bias K-basis effect \|Δlog_p\| | 0.05 | — | **near no-op** |
| V-bias V-basis effect \|Δlog_p\| | **0.22** | — | **4.4× K-basis** |
| V-bias random-basis effect \|Δlog_p\| | 0.10 | — | K-basis 보다도 2× 큼 |

**결론**: 현재 paper 의 Thm 6.17 (ii)(d) V-inert/V·K-destructive 주장은 "K-basis 를 V-space 에 적용한 것이 language-mean 방향만 증폭" 때문에 발생한 구현 아티팩트.

### Finding 2 — Thm 6.21 log-p 예측 반증

**실험** (`/tmp/measure_logp_curve_v3.py`, Qwen Subtask1 N=100, autoregressive full-target):

| α | mean sum_logp | Δ | first_argmax_acc |
|---|---|---|---|
| 0.000 | −3.59 | 0 | 0.46 |
| 0.025 | −3.68 | −0.09 | 0.46 |
| 0.100 | −3.86 | −0.27 | 0.46 |
| 0.300 | −3.90 | −0.30 | 0.47 |
| 0.500 | −3.76 | −0.17 | 0.48 |

**Paper §519 claim** (현재): "log-p monotonically increasing by +1.24 nats over [0, 0.3]"  
**실측 v3**: 단조 **감소** by −0.30 nats (α=0.3). **방향 완전 반전**.

V1 script 의 argmax_acc=0 (null measurement) 버그 수정 후 진짜 패턴은 "log-p 감소 + argmax_acc 유지" — Thm 6.21 의 G_K > 0 예측은 log-p objective 에서 empirically 성립 안 함.

### Finding 3 — Scaling law 재발견 (α_opt 모델별 예측)

3-model 측정 (Qwen/Llama/Mistral L=13/15/15):

| 모델 | ε_q | Var_s V | gain_H-cat | 실측 α_opt |
|---|---|---|---|---|
| Qwen | 0.192 | 0.406 | 2.48 | ~0.05 |
| Llama | 0.151 | 0.070 | 2.82 | ~0.30 |
| Mistral | 0.310 | 0.179 | 2.00 | ~0 |

**제안된 scaling law** (3-point fit):
$$\alpha_\mathrm{opt}(\theta, \tau) \approx C(\tau) \cdot \frac{\mathrm{gain}_{H\text{-cat}}(\theta) - 2.0}{\sqrt{\varepsilon_q^\tau \cdot \mathrm{Var}_s V}}$$

Mistral = 0 예측 정확 (gain 경계 효과), Llama/Qwen ratio 4.77 vs 실측 6.0 (21% 오차).

### Finding 4 — Statistical hygiene 문제

Qwen Subtask4 QKV microsweep full 497:
- Q-only β=−0.1: **+1.64pp F1**
- Q+K pair α=0.025, β=−0.1: **+1.95pp F1**
- Δ = +0.31pp
- SE (N=497, F1) ≈ 2.2pp

**Q+K pair 주장은 통계적으로 Q-only 와 구분 안 됨.** 현재 paper §3.6.1 (iii) "strongest verified pair" claim 은 within sampling noise.

---

## 3. 세 가지 framing 옵션

### Approach 1 — **Status quo** (현재 paper body)
- Main: Thm 6.17 (iii) Q+K small-α pair + K dual-role (stability + accuracy)
- 12 contributions in §1.1
- V-inert + V·K destructive + Thm 6.21 α_opt + unified Pareto 복잡 narrative

### Approach 2 — **Full Q-only pivot** (실험자 2 제안)
- Main: Thm 6.17 단순화 → Q-coverage step-adaptive only
- K = stability diagnostic + compression 전용 (accuracy role 제거)
- Thm 6.17 (ii)(d) V-inert/trio 주장 appendix 이동
- Thm 6.19 unified: Q-steering + K-stability-diagnostic + K-compression 3-role clean separation
- 12 → 6 contributions
- Writing cost: **3–4 시간**

### Hybrid — **Headline-only swap** (실험자 1 제안)
- Body 유지, Abstract + §1.1 headline 만 수정
- "Q-side family (+1.6 ~ +3.7pp on shared $B_\mathrm{ont}$)" 을 main 으로 승격
- §3.6.1 (iii) "best pair" 주장에 **SE disclaimer** 1 줄 추가 (statistical honesty)
- Thm 6.17 (ii)(d) 본문 유지 + V-basis test 결과 pending 명시
- Writing cost: **30 분**

---

## 4. 리뷰어 점수 추정 (교정됨, 두 실험자 합의)

| 축 | Approach 1 | Hybrid | Approach 2 |
|---|---|---|---|
| Soundness | 3.2 | 3.3 | 3.3 |
| Presentation | 3.0 | 3.3 | 3.4 |
| Contribution | 3.0 | 3.1 | 3.1 |
| **Mean** | **6.2** | **6.5–6.6** | **6.7–6.9** |
| **Accept prob** | 45% | 55% | 58% |
| **Writing cost** | 0 | 30 min | 3–4 h |
| **Reframe fatigue** | 0 | 0 | 4th reframe risk |
| **Robustness to SEKA unfavorable** | weak | medium | strong |
| **ROI (score/hour)** | — | +0.48/h | +0.091/h |

**Hybrid 가 full pivot 대비 5× ROI 우위.**

### 조건부 Expected Value (pending 결과 marginalized)
- Approach 1: 6.07
- Hybrid: 6.31 (+0.24)
- Full Pivot: 6.63 (+0.32 over Hybrid)

---

## 5. 두 실험자 의견 요약

### 실험자 1 (hybrid + caution)
**주장**: Pivot 마진은 +0.2–0.3, full pivot 는 ROI 낮음. 30분 hybrid 로 거의 같은 효과. Reframe fatigue (4th iteration) risk.
**핵심 지적**: 내 직전 분석에서 **soft-routed Q-side +3.7pp F1 / +4.8pp Exact 누락** + Thm 6.20 누락. Approach 1 점수를 7.2 → 5.0 으로 내린 것은 "pivot advocacy 왜곡".

### 실험자 2 (full pivot + statistical honesty)
**주장**: Statistical hygiene (Q+K pair within SE) + novelty focus (Q-side prior art 공백) + robustness to pending → full pivot 권장.
**Self-criticism**: "empirical reality 덮는 pattern" 자인. Pivot 은 "인정 못한 statistical reality 수용".

### 두 의견 공통점
- +1.95pp "best pair" 는 within SE (실험자 2 가 critical, 실험자 1 도 유효성 인정)
- Q-coverage step-adaptive 는 genuinely novel (Focus Directions 조차 stationary)
- Full pivot 의 structural benefit 는 실재 (V-inert, Thm 6.21 mismatch 해소)

### 두 의견 차이
- **실험자 1**: 현재 paper 에 이미 Q-side 3 mechanism 들어있음 → hybrid 로 충분
- **실험자 2**: 본문 정리도 필요 → full pivot

---

## 6. 종합 권고 (adaptive 3-step)

### Step 1 (즉시, 30분) — Hybrid headline swap
1. Abstract 에 Q-side family (+1.6~+3.7pp) headline 승격
2. §1.1 Thm 6.20 contribution 복원 (누락됨)
3. §3.6.1 (iii) "best pair" 에 SE=2.2pp within-noise disclaimer 추가
4. §1.1 item 7 main contribution 위치로 이동

### Step 2 (병렬, 30분) — Mistral Q-only smoke test
`CUDA_VISIBLE_DEVICES=1 python eval_metatool_subtask4.py --methods ocq_qbias_b-0.1 --max-samples 100` — Q-coverage 의 3-family 보편성 검증.

### Step 3 (3–4시간 후, 조건부) — SEKA + V-basis 결과 보고 full pivot 판단
**Trigger for full pivot**:
- SEKA 결과 우리 열세 (Llama Subtask4 F1 gap < 0)  
  **AND**
- Qwen mixed-basis V-basis test 에서 V+Q < Q-only (V-rescue 실패)

이 양 조건 동시 발생 시 (확률 ~25%) full pivot 이 structural 필요. 아니면 Hybrid 유지.

---

## 7. Coworker 에게 묻는 세 가지

1. **이 framing 결정에 대한 독립 의견**: Hybrid 충분? Full pivot 필요? 아니면 다른 제3안?
2. **SEKA / AdaSEKA 비교 결과 (A100 track)**: 진행 상황 및 예상 F1 gap?
3. **Soft-routed Q-side (AdaSEKA-proxy +3.7pp) 를 Q-side novel contribution 으로 positioning 하는 것 동의?** 이것이 본래 AdaSEKA proxy 였는데 proxy label 을 제거하고 soft M-of-2 Q-side routing 우리 method 로 재framing 적절한지.

---

## 8. 현재 진행 중 long jobs

| GPU | Job | ETA |
|---|---|---|
| 0 | Qwen mixed-basis QKV full 497 sweep | ~2h |
| 0 | Llama Subtask4 K-sweep (4 α, 이미 진행 중이었음) | ~1–2h |
| 0 | Llama Subtask1 α-sweep (6 α, 이미 진행 중) | ~2h |
| 1 | SEKA full Llama Subtask4 eval (no_steer + amp {1,2,5}) | ~2–3h |

모든 결과 04-16 아침 (~08:00 KST) 이전 도달 예상. 그 때 Step 3 결정 진행.

---

## 9. Raw files 참조

- Finding 1 (V-K overlap): `reports/basis_overlap_2026_04_16/{qwen,llama}_VK_overlap.json`
- Finding 1 (V-bias AB): `reports/vbias_ab_test_2026_04_16/qwen_vbias_ab.json`
- Finding 2 (Thm 6.21 v3 logp): `reports/logp_curve_2026_04_16/qwen_st1_logp_curve_v3.json`
- Finding 3 (scaling law): `reports/scaling_law_2026_04_16/{qwen_L13,llama_L15,mistral_L15}.json`
- Finding 4 (QKV microsweep): `reports/qkv_alpha_microsweep_2026_04_15/full497_alpha_microsweep.json`

---

**Document prepared**: 2026-04-16 02:30 KST  
**Next decision point**: 2026-04-16 ~08:00 KST (SEKA + V-basis 결과 + coworker feedback 합쳐서)
