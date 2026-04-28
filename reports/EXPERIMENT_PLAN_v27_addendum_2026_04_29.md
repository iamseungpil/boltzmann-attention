# EXPERIMENT_PLAN_v27 — Addendum (2026-04-29 evening)

> Append to: `reports/EXPERIMENT_PLAN_v27_rank_bounded_replaceability_2026_04_28.md`
> Driven by NeurIPS-reviewer simulation (W1, W2, W3 fixes)

## 새로 추가되는 실험: E7, E8, E9

### E7. Multi-tool selection F1 (W1 해결)

**의도.** "98.4% next-token top-1"는 next-token argmax 동의일 뿐 실제 도구 선택 정확도가 아니다. 본 실험은 prompt-제거 + 정적 + Q-bias 결합 개입 하에 *실제 generation*을 돌려 **다중 도구 호출 F1**을 측정한다.

**가설 (반증 가능).**
- H7a: 정적 + Q-bias($\beta{=}{+}4.0$) hybrid가 Qwen MetaTool ST4 F1을 noprompt 베이스라인의 ≤0.10에서 full-prompt의 ≥0.50으로 끌어올린다.
- H7b: top-1 next-token 일치율 0.984가 multi-step generation에서 *대부분 유지*된다 (F1 ≥ 0.80 × full_prompt_F1).

**검증 방법.**
- 모델: Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct.
- Task: MetaTool Subtask 4 (N=128 권장, 시간 허락하면 N=497 전체).
- 조건 (4):
  1. `full_prompt` — system + user, no intervention (anchor)
  2. `noprompt` — user only, no intervention (bottom)
  3. `static_only` — user only + V_1 V_1^T φ_mean injection at o_proj
  4. `hybrid_b{β}` — static + β·V_8V_8^T·Q at q_proj. Qwen β=+4.0, Llama β=-2.0 / +4.0.
- **지표 (`eval_metatool_subtask4.py` 재사용)**:
  - **F1**: 가중 평균 정밀-재현
  - **F_{0.5}**: 정밀도 가중치 (실수 비용 강조)
  - **EU(α=1, β=2, γ=1)**: Expected Utility (도구 누락 페널티)
  - **Jaccard**: 집합 유사도
  - **Exact-set**: 정확히 GT 집합 일치 비율
- 생성 설정: max_new_tokens=256, do_sample=False, temperature=1.0 (greedy).

**Pass criterion.**
- H7a: hybrid F1 ≥ 0.50 (Qwen). 실패 시 noprompt baseline에 가까우면 W1 잔존, paper §5에 명시.
- H7b: hybrid F1 ≥ 0.80 × full F1.

**리소스.** A6000 × 1 × ~30분 (256 토큰 generation × 128 쿼리 × 4 조건). 양 모델 병렬.

**스크립트 (신규).** `scripts/rank_replaceability/intervention_metatool_eval.py` (`eval_metatool_subtask4.py`의 prompt builder + parser 재사용).

---

### E8. Llama Q-self basis 검증 (W2 가설 1 해결)

**의도.** Llama가 Qwen 대비 부분 회복인 이유 — 본 paper §6.2에서 세 가설 적시:
1. **선형 projector 기저 불일치**: V_k가 Φ_P SVD에서 추출, 실제 Q-Jacobian과 정렬 안 됨
2. GQA 헤드 그룹화 효과
3. 잔차-흐름 기여 지배

E8은 가설 1만 검증한다 (가장 빠르고 이론에 직접 닿음).

**가설 (반증 가능).**
- H8: V_k를 Φ_P 대신 *Q activations 자체*의 SVD에서 추출하면 Llama의 Q-bias 응답이 Qwen 수준으로 강화된다 (top-1 ≥ 0.50).

**검증 방법.**
- noprompt 입력 위에서 forward pass, 각 (layer, head)의 Q activations 수집 (마지막 위치).
- 행렬 $M_Q^{(\ell,h)} \in \R^{N\times d_h}$, SVD, 상위 k vectors → V_k^{Q-self}.
- qbias_hybrid_eval.py에 `--basis-mode q_self` 추가, V_k^{Q-self}로 Q-bias projector 구성.
- Llama-3.1-8B, MetaTool ST4, N=128, β ∈ {-4, -2, -1, 0, 1, 2, 4}.

**Pass criterion.**
- top-1 ≥ 0.30 (Qwen 수준 0.984은 못 미쳐도, 현 0.10 대비 3배 이상 상승)
- KL ≤ 5.0 (Qwen 수준 ~3.0에 근접)

**Outcome 매트릭스.**
| 결과 | 해석 |
|---|---|
| Pass | 가설 1 확인. Theorem 2의 V_k는 *함수* 의존적, Φ_P SVD가 항상 최선 아님 |
| Fail (변화 없음) | 가설 1 기각. 가설 2 (GQA) 또는 3 (residual stream) 후속 |
| Marginal (+5pp) | 가설 1 partial. 가설 2/3 동시 작용 가능 |

**리소스.** A6000 × 1 × ~5분.

**스크립트.** `qbias_hybrid_eval.py` 확장 (`--basis-mode q_self` 플래그 추가).

---

### E9. Production-scale prefix r* (W3 해결)

**의도.** 현재 측정 prefix는 93--189 토큰. 실제 production 에이전틱 하니스(Anthropic Tool-Use, Graphify)는 2--8K 토큰. 이 regime에서 r*가 폭발하는지 확인.

**가설 (반증 가능).**
- H9a: prefix 길이를 ~2K 토큰으로 늘려도 r*(0.95) 평균 ≤ 5 유지 (현재 2.25 대비 2x 증가 허용).
- H9b: r*가 prefix 길이의 *sub-linear* 함수 ( r* = O(L^{0.3})에 가까움).

**검증 방법.**
- `eval_tau2_bench.py`의 `build_tools_json()`을 사용하여 도구당 (description + 매개변수 schema) JSON 형식으로 prefix 구성.
- 도메인별 token 수 (대략):
  - Retail: 15 도구 × ~150 토큰 = ~2.3K
  - Telecom: 35 도구 × ~150 = ~5.3K
  - Airline: 12 도구 × ~150 = ~1.8K
- E1 측정 재실행, prefix_mode="real_full_schema".
- 비교: 현재 prefix (이름만) → 새 prefix (전체 schema).
- 모델: Qwen2.5-7B (primary), Llama-3.1-8B (secondary).

**Pass criterion.**
- r*(0.95) 평균 ≤ 5 (모든 (모델, 도메인) 셀)
- 격차 (full schema - names only) ≤ +2.0 평균.

**Outcome 매트릭스.**
| 결과 | 함의 |
|---|---|
| r* ≤ 5 | Theorem 1 conclusion robust to prefix length, paper claim 강화 |
| r* in [5, 15] | Static 충분 조건 borderline; rank-16 정도 intervention 필요 |
| r* > 15 | Long-prefix regime은 정적으로 부족, query-conditional이 필수 — paper의 claim 약화, scope 제한 |

**리소스.** A6000 × 1 × 6 셀 (3 도메인 × 2 모델) × ~3분 (긴 prefix로 forward 시간 증가) = ~20분.

**스크립트.** `measure_phi_rank.py`의 tau2 loader에 `--tool-schema-mode {names, full}` 추가.

---

## 통합 결정 트리 (E7-E9 결과 반영)

```
E9 (production prefix)
 ├─ r* ≤ 5: Theorem 1 generalizes. Paper claim 유지.
 ├─ r* ∈ [5, 15]: scope을 1-2K 토큰 prefix로 한정. Paper claim 약화 명시.
 └─ r* > 15: 정적 paradigm 잘못. 다른 framing 필요.

E7 (multi-tool F1)
 ├─ Qwen hybrid F1 ≥ 0.50: W1 해결. NeurIPS 점수 6-7 가능.
 ├─ F1 ∈ [0.20, 0.50]: partial. Paper에 "next-token vs F1 격차" 명시.
 └─ F1 < 0.20: hybrid가 first-token만 일치, generation이 발산. 
     다른 intervention 또는 generation-aware loss 필요.

E8 (Llama Q-self basis)
 ├─ Llama hybrid top-1 ≥ 0.30: Theorem 2 V_k 의존성 확인. 
     Paper "Llama partial → Llama works with proper basis"로 격상.
 ├─ top-1 ∈ [0.10, 0.30]: 가설 1 partial. 가설 2/3 후속 필요.
 └─ top-1 < 0.10: 가설 1 기각. Paper §6.2의 다른 가설로 이동.
```

---

## NeurIPS 점수 변화 예측

| 시나리오 | E7 결과 | E8 결과 | E9 결과 | 예상 NeurIPS 점수 |
|---|---|---|---|:-:|
| 현재 | — | — | — | 5 (borderline reject) |
| Best case | F1 ≥ 0.50 | top-1 ≥ 0.30 | r* ≤ 5 | **7** (accept) |
| Middling | F1 ∈ [0.30, 0.50] | top-1 ∈ [0.15, 0.30] | r* ∈ [5, 10] | 6 (marginal accept) |
| Worst case | F1 < 0.20 | 변화 없음 | r* > 15 | 4 (clear reject, framing 재검토) |

---

## 우선순위 + 일정

| 우선순위 | 실험 | 시간 | GPU |
|:-:|---|:-:|---|
| 1 | E9 (가장 cheap, framing 영향 최대) | ~20분 | A6000 ×2 |
| 2 | E8 (Llama 가설 분리) | ~10분 | A6000 ×1 |
| 3 | E7 (multi-tool F1, 가장 무거움) | ~60분 | A6000 ×2 |

E9를 먼저 실행해서 paper claim의 *가장 큰 risk*를 검증하고, 동시에 E8 실행. 마지막으로 E7 (가장 무거움). 총 estimate: ~90분 wallclock.
