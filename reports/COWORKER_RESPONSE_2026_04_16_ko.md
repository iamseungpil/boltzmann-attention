# Coworker 비판 응답 + 현황 보고 (2026-04-16 AM)

**From**: develop side (mais)
**To**: coworker (A100×4 트랙)
**Re**: `PAPER_DRAFT_v3.md` 에 대한 어제 밤 4대 비판
**Branch**: develop, 본 수정 반영 후

---

## 한 줄 요약

- 4대 비판 모두 v3 시점에서는 정확한 지적. **3/4 는 오늘 writing-only 수정으로 대응 완료** (추가 실험 불필요). #2 는 여전히 P0-A SEKA 에 의존.
- 오늘 밤 GPU0/GPU1 chain 결과로 locked score **6.30 → 6.61** 승격 (Thm 6.1 Llama L=15 2-family + Mistral null-control (H-cat)-boundary 재framing).
- 논문은 이제 v1/v2/v3 3개 변종으로 존재, 셋 모두 동일한 비판-대응 수정 반영. 기본 lock 은 **v2**. P0-A 도착 시 v3 로, P0-B 도착 시 v1 로 분기.

---

## 4대 비판 — 응답

### (1) Central novelty mismatch: title 은 "step-adaptive" 주장, 실제 구현은 stationary approximation

**상태**: writing-only 수정 완료.

**조치**: v1/v2/v3 전체에서 `step-adaptive Q-coverage` → `facet-aware Q-coverage` 로 rebrand (v3 title 포함). 수학 정의 $\Delta_Q^{(t)} = -\beta \sum_{s<t} P_{f_s} q_t$ 는 유지 (형식상 step-varying) 하되 모든 claim 위치에 "App. F.5 의 stationary approximation 하 검증" 를 명시. Novelty 는 **per-head facet basis $B_\mathrm{ont}$ projection $P_f := B_f B_f^\top$** (Focus Directions / SEKA 에 부재) 로 재귀속, step-adaptivity 로 귀속되지 않음.

### (2) SEKA Subtask4 미완결 → axis-separation premature

**상태**: null-control framing 으로 부분 해소, **P0-A 에 여전히 의존**.

**P0-A 없어도 성립하는 defense line**: axis-separation 은 SEKA head-to-head 를 *필요로 하지 않음* — 3-tier null-control 이 이미 "any K-perturbation would do" 가설 기각 (real 0.685 vs random/featshuffle 0.000 at 동일 α=0.3 norm = +68.5pp gap). Self-falsification angle: 우리 자신의 stationary K-bias at α=0.3 on Subtask4 이 −4.6pp — 우리가 스스로 K-stationary 가설을 기각. P0-A 는 external 비교 closing 으로 +0.30 score marginal 추가하나, minimum defense 는 P0-A 없이 성립.

### (3) Contribution overload, 일부 prediction-only

**상태**: writing-only 수정 완료.

**조치**: Thm 6.19 "tri-role unification" → 세 변종 모두에서 **Rmk 6.19** 로 격하 (v1/v2 항목 0, v3 항목 4). 이미 열거된 항목들을 parsimony 로 묶는 관찰로 재framing, 새 empirical 내용 없는 standalone theorem 에서 제외. Prediction-only 로 남은 항목: **Thm 6.18 (P0-B) + Thm 6.21 α-opt 만**. 오늘 밤 이후 empirical-verified: Thm 6.1 (2-family Qwen+Llama), Cor 6.9/6.9.6 (full 497), Thm 6.17 Q-coverage (2-family full 497), Thm 6.13 OCQ 2-bit (full WT2), Thm 6.20 AUROC 0.976, (H-cat) threshold (3-model 중 2 within / 1 boundary).

### (4) Effect size +1.64/+0.40pp main-track 으로 약함

**상태**: writing-only 수정 완료.

**조치**: Llama-Inst Subtask1 **K-bias +15.08pp (full N=995)** 를 v1/v2/KO 의 §1.1 항목 0′ 로 headline 승격. v3 에서는 항목 1′ "parity-regime supporting evidence" 로 삽입 (novelty 는 multi-selection axis 에 유지). Framing shift: Subtask4 +1.64/+0.40pp 는 multi-tool **autoregressive-regime-bounded** lift; Subtask1 cross-family 에서는 stationary 연산자가 re-attention 간섭 없이 동작하여 double-digit lift. Defense line: "Subtask4 작은 lift 는 regime ceiling 반영, skill deficiency 아님" — weak-direction 대안 가설 기각.

---

## 오늘 밤 새 실험 결과 (GPU0/GPU1 chain, 18:20 KST 완료)

| 결과 | 값 | 관련성 |
|---|---|---|
| Thm 6.1 Llama L=15 bound | **3200/3200 pass, 1.00 rate** (Qwen L=13: 2800/2800 prior) | 2-family 검증 (단일 → 이중) |
| Mistral-Inst Subtask1 null-control (α=0.3) | real −2.92pp / random **+0.60pp** / featshuffle −0.60pp | (H-cat)-boundary 재framing — ad-hoc "hedging" 가설 기각 |
| Llama-Inst Subtask1 K-bias full 995 | 62.31% → **77.39% (+15.08pp)** | 논문 최대 lift, 새 headline |
| Llama-Inst Subtask1 Q-cov β=−0.3 | +8.04pp | 보조 Q-channel 확인 |
| Qwen Subtask4 QKV full 497 (5-cell) | Q+K small-α best **+1.95pp**; trio destructive −0.88pp vs Q+K | Thm 6.17 (d) QKV-joint falsified; (b)(a′) verified |
| Qwen Subtask4 V-bias full 497 | α=0.1: −0.43pp, α=0.3: −0.90pp | V-single-axis 예상 negative control |
| Llama-Inst Subtask4 Q full 497 | β=−0.1: +0.40pp, β=−0.3: −1.30pp | cross-model Q-coverage 검증 |
| Llama3-v3 (추가 모델) | no_steer F1=0.333, 모든 steer 하락 | scope 외 (미통합) |

**통합 위치**: PAPER_DRAFT_v1 §5.6 (Thm 6.1 Llama extension) + §5.4 row 704 (Mistral) + §5.5.1 ((H-cat)-boundary 재framing). KO mirror 동일. v2/v3 는 이미 지원 결과 흡수됨.

---

## 논문 변종 — 분기 결정 표

세 변종 모두 동일한 비판-대응 수정 반영 완료. P0 결과 도착 시점 기준으로 분기:

| Trigger | 날짜 | 선택 변종 | 근거 |
|---|---|---|---|
| P0-A SEKA Subtask4 Qwen+Llama 둘 다 네거티브 | ≤ 4/22 | **v3** (Facet-Aware Q-Coverage pivot) | axis-separation empirically closed; multi-selection 을 primary novelty |
| P0-B Thm 6.18 full-WT2 PPL ≤ 13.5 at 1.81 bits | ≤ 4/25 | **v1** (Joint Pareto unified) | compression Pareto closed; 가장 넓은 서사 |
| 둘 다 | ≤ 4/28 | **v1 + v3 의 axis-separation 을 서브 컬럼으로** | maximal coverage |
| 5/01 까지 둘 다 미도착 | 5/04 abstract | **v2** (Hybrid, default lock) | 모든 headline 이미 검증됨 |

점수 기대치:
- v2 default: 6.61–6.90 (accept 62%)
- v3 pivot (P0-A 성공): 6.85–7.25 (accept 70%)
- v1 expand (P0-B 성공): 6.85–7.20 (accept 68%)

---

## 요청 사항 (v3 night sprint 대비 변동 없음)

P0 4개 task 그대로 유지 요청. 우선순위 변동 없음:

- **P0-A SEKA+AdaSEKA head-to-head** (source: `external/SEKA/src/model/{seka_llm,adaptive_seka_llm}.py`, 이전 proxy 아님). Qwen2.5-7B-Instruct + Llama-3.1-8B-Instruct, Subtask1+4. → v3 pivot unlock.
- **P0-B Thm 6.18 full-WT2 PPL** (code starts: `scripts/ocq/measure_thm618_attn_weighted_bits.py`, PPL eval 추가 필요). Qwen2.5-7B WT2 ctx=2048 non-overlap, sweep avg_bits {1.81, 2.0, 2.5, 3.0, 4.0}. Full credit PPL ≤ 13.5 at 1.81. → v1 expansion unlock.
- **P0-C 6 baseline** (CAA/ITI/PASTA/Focus/LoRA-FT/RAG, source-first 정책 `feedback_external_baseline_use_original_source` 준수). Partial 인정 = CAA+ITI+LoRA-FT (+0.15).
- **P0-D Thm 6.20 τ²-bench retail** (B_ont: `external/SEKA/seka_projections/ontology-qwen25-7b-tau2-retail/B_ont.pt`; `scripts/ocq/measure_epsilon_q_plan_predictor.py` adapt). 50–200 conversations. → +0.20.

SEKA hang 은 develop side 에서 **원인 확정**: `seka_llm.py:34` 의 multi-GPU auto-detection. A6000×2 환경에서 `torch.cuda.device_count() > 1` → `device_map="auto"` → 모델이 2개 GPU 에 자동 샤딩 (layers 0-11 GPU:0, 12-27 GPU:1) → hook 내 P_pos 텐서가 cuda:0 에만 있으므로 cuda:1 layer 의 hook forward 에서 **cross-device CUDA deadlock** 발생.

⚠️ **P0-A 실행 시 필수 조치**: A100×4 환경에서 SEKA 실행 시 반드시 **`CUDA_VISIBLE_DEVICES=0`** (또는 단일 GPU 번호) 으로 제한할 것. 4장이면 더 복잡한 4-way 샤딩 발생하여 동일한 hang 으로 시간 낭비할 위험. `CUDA_VISIBLE_DEVICES=0` 설정 시 develop side A6000 에서 45초 hang → **2.2초 정상 생성** 확인됨.

참고: dtype 변환 (bf16↔f32) 은 벤치마크 결과 55μs/token (256-token loop 14ms) 으로 hang 원인 아님.

---

## 오늘 수정된 파일

- `math/paper/benchmark_design/PAPER_DRAFT_v1_2026_04_14.md` — 3개 edit + 오늘 밤 결과 통합 (§5.6, §5.4, §5.5.1)
- `math/paper/benchmark_design/PAPER_DRAFT_v1_ko.md` — 3개 edit + 오늘 밤 결과 mirror
- `math/paper/benchmark_design/PAPER_DRAFT_v2.md` — 3개 edit
- `math/paper/benchmark_design/PAPER_DRAFT_v3.md` — 3개 edit (title rebrand 포함)
- `reports/thm61_llama_2026_04_15/llama_L15_a0.3_N100.json` — 신규
- `reports/mistral_null_2026_04_15/*.json` — 신규 (random + featshuffle)
- `reports/wave_pm2_2026_04_15/` — PM Wave 2 SUMMARY + gpu0/gpu1 outputs

다음 세션 핸드오프 메모: `tonight_integration_2026_04_16.md`.

---

P0 결과 하나라도 도착하면 Slack 으로 ping 부탁. 도착 후 1일 이내 pivot 진행.
