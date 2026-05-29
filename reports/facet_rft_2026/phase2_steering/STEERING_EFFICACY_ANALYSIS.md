# Phase 2a Steering 효능 분석 — 왜 +1.5%p에 그쳤나?

**작성**: 2026-05-29 (Phase 2a α 그리드 중간)
**상태**: 살아있는 문서 — 다음 세션 검토/실행용
**Owner branch**: `facet-rft-2026`

---

## 0. 문제 정의

T1 residual-stream steering (mean-of-differences 벡터, `validates` relation)을 적용한 결과:

| 모델 | baseline (α=0) | peak (α=0.5) | lift |
|---|---|---|---|
| Qwen2.5-7B | 0.1765 | 0.1917 | **+1.5%p** |
| Hermes-3-8B | 0.1237 | (0.1286, 데이터 불량) | **~+0.5~1.1%p** |

**기대 대비 격차**: steering 논문(CAA, ITI 등)은 종종 10~40%p 향상을 보고. 우리는 +1.5%p.
이 문서는 격차의 원인을 7개 가정(H1–H7)으로 분해하고, H7은 기존 데이터로 검증한 결과를 포함한다.

---

## 1. 배경: steering 논문이 큰 수치를 내는 조건 vs 우리 과제

| 항목 | 기존 steering 큰 향상 사례 | 우리 (τ²-bench telecom) |
|---|---|---|
| 과제 유형 | 단일 축 행동 (truthfulness, refusal, sycophancy) | 멀티턴 agentic 성공 |
| 성공 구조 | 1개 방향이 전체 목표 포착 | 다인자: 의도×도구선택×순서×검증×종료 |
| 실패 원인 | "능력 있으나 잘못 적용" | capability 실패 + behavioral 실패 혼재 |
| 측정 | 단일 응답 정확도 | pass^1 (전체 멀티턴 trajectory 성공) |
| 대표 결과 | ITI TruthfulQA, CAA refusal: +10~40%p | — |

→ **구조적으로 우리 과제는 단일 벡터 steering이 큰 효과를 내기 어려운 setting**. 이게 가정들의 출발점.

---

## 2. 7개 가정 (H1–H7)

### H1. 단일 relation steering은 다단계 과제에 너무 좁다 — ★최유력
- **주장**: `validates`는 42 relation 중 1개. 검증 행동만 교정. 과제 실패는 다양한 원인(도구 오선택, 순서 오류, 조기 종료, over-diagnosis)에서 발생 → 1 relation은 최대 1 실패 모드만 교정.
- **예측**: relation별 효과 크게 다름. `achieves_goal`, `terminate_when_done`, `plan_committed_to_goal` 류가 `validates`보다 클 수 있음.
- **검증 실험**: Phase 2b relation sweep (5 relations × α=0.5 × layers 고정). **스크립트 준비됨** (`_phase2b_qwen_dual_gpu.sh`).
- **상태**: 미검증.

### H2. Capability ceiling — 능력 부족 실패는 steering 불가 — ★최유력
- **주장**: Qwen-7B baseline 0.18 = telecom이 7B에게 본질적으로 어려움. 실패의 상당수가 capability 실패(멀티스텝 추론 자체 불가)이지 behavioral 실패가 아님. Steering은 행동 nudge일 뿐 없는 능력 추가 못함.
- **예측**: 큰 모델(32B)일수록 steering 효과 ↑ (capability 충분 → behavioral 교정 여지 ↑).
- **검증 실험**: 32B 모델 same steering. 효과 ↑면 H2 지지. **단 strategic gating: Phase 2-4 양성 후에만 32B**.
- **상태**: 미검증. (gating에 묶임)

### H3. Mean-of-differences 벡터가 약한 추정자 (distribution shift) — ★유력
- **주장**: contrast_pairs_v3는 **짧은 합성 문장** (1608 pairs 추출이 23~51초 = 매우 짧은 텍스트). Probing은 고립 문장에서 했지만 steering은 멀티턴+system prompt+tools+히스토리 맥락에서 적용. 분포 차이로 "validates 방향" 벡터가 agentic 맥락에선 다른 방향일 수 있음.
- **예측**: 실제 agent trajectory에서 추출한 벡터가 더 강한 효과.
- **검증 실험**: trajectory 기반 벡터 재추출 (성공 trajectory의 hidden state vs 실패 trajectory). **신규 스크립트 필요**.
- **상태**: 미검증.

### H4. Constant steering (모든 토큰·위치) = 부작용 동반 — ★유력 (H7로 강하게 지지됨)
- **주장**: α·v를 매 forward·매 토큰 위치(프롬프트 인코딩 포함)에 더함. 어떤 맥락엔 도움, 어떤 맥락엔 해 → 상쇄 → 작은 net + U-shape (α 크면 부작용 지배).
- **예측**: context-gating (특정 토큰/단계에만 적용) 하면 효과 ↑, 부작용 ↓.
- **검증 실험**: context-gate vs raw add 비교. Phase 0 설계엔 "context-gate (Default) / Gram-Schmidt (Fallback)" 있으나 **현재 raw add로 미구현**.
- **상태**: **H7에서 간접 지지 확인** (per-task 큰 swing ±0.5 상쇄 — §3 참조). 구현 필요.

### H5. Layer 선택 미최적 — ☆Phase 2b로 검증 예정
- **주장**: layers 12,13,14 (중간) = probing PCLI 분리도 높은 층. 하지만 분리도 ≠ 인과적 steering 효과. 출력 행동은 후반 층(20+)이 더 직접 제어할 수 있음.
- **예측**: layer sweep에서 다른 층이 더 강할 수 있음.
- **검증 실험**: Phase 2b layer sweep ([2,3,4], [7,8,9], [12,13,14], [17,18,19], [22,23,24] × α=0.5). **스크립트 준비됨**.
- **상태**: 미검증.

### H6. relation 선택 자체가 과제 성공과 약하게 연결 — ☆
- **주장**: `validates`는 top-PCLI(분리도) 기준 선택이지, 과제 성공과 인과적으로 가장 연결된 relation 기준이 아님. 분리 잘 되는 relation ≠ 성공에 중요한 relation.
- **예측**: 과제 성공 상관 기준으로 relation 재선택하면 효과 ↑.
- **검증 실험**: 각 relation 벡터를 성공/실패 trajectory에 투영, 성공 상관 높은 relation 선택 후 steering. **신규 분석 필요**.
- **상태**: 미검증. (H1 relation sweep과 부분 중첩)

### H7. pass^1 metric coarse + 작은 N → subset 효과 희석 — ☆부분 확인됨
- **주장**: N=30 tasks × 4 trials, binary pass. 특정 task 유형엔 큰 효과지만 평균에서 희석.
- **예측**: 도메인/난이도별 분해하면 일부 subset에서 큰 효과.
- **검증**: 기존 데이터로 즉시 분석 가능 → **완료 (§3)**.
- **상태**: **부분 확인** — Qwen Easy persona +10%p (단 n=40 CI overlap).

---

## 3. H7 분석 결과 (기존 데이터, α=0 vs α=0.5)

### Qwen (clean data: α=0 N=119, α=0.5 N=120)

**도메인별** (균일, 큰 subset 없음):
| 도메인 | α=0 | α=0.5 | Δ |
|---|---|---|---|
| mms_issue | 0.077 | 0.100 | +2.3%p |
| mobile_data_issue | 0.175 | 0.175 | 0 |
| service_issue | 0.275 | 0.300 | +2.5%p |

**PERSONA별** (← 큰 신호):
| PERSONA | α=0 | α=0.5 | Δ |
|---|---|---|---|
| **Easy** | 0.100 | **0.200** | **+10.0%p** ⭐ |
| Hard | 0.179 | 0.175 | −0.4%p |
| None | 0.250 | 0.200 | −5.0%p |

**per-task pass-rate flips** (4 trials each, |Δ|≥0.25):
```
+0.50  service: airplane_mode_on|lock_sim_card_pin|overdue_bill_suspension
+0.25  service: break_apn_settings|lock_sim_card_pin|overdue_bill_suspension
+0.25  service: break_apn_settings|contract_end_suspension|lock_sim_card_pin
+0.25  mms:     break_app_storage_permission|data_usage_exceeded
−0.25  service: contract_end_suspension|unseat_sim_card
−0.50  service: airplane_mode_on|break_apn_settings|lock_sim_card_pin|overdue  ← 큰 손실
```

### Hermes (⚠️ α=0.5 데이터 불량: 119 done 중 49 infra_error, 70 valid만)
| 분해 | α=0 | α=0.5 (불량) | Δ |
|---|---|---|---|
| OVERALL | 0.1237 | 0.1286 | +0.5%p |
| mobile_data | 0.184 | 0.207 | +2.3%p |
| None persona | 0.182 | 0.227 | +4.5%p |

→ **Hermes α=0.5 재실행 필요** (49 infra_error로 신뢰 불가).

### H7의 두 가지 발견

**발견 1: 집계가 subset 효과를 희석** (H7 부분 확인)
- Qwen Easy persona +10%p가 집계 +1.5%p에 묻힘.
- 단 n=40/bucket, Wilson CI overlap → 신호일 뿐 통계적 유의 아님.

**발견 2 (더 중요): per-task 큰 swing 상쇄 → H4 강력 지지**
- Steering은 균일한 작은 nudge가 아니라 **큰 양/음 효과의 합** (일부 +0.5, 일부 −0.5).
- net +1.5%p = (큰 gains) − (큰 losses).
- **함의**: context-gating으로 손실 맥락을 제거하면 net 효과가 훨씬 커질 수 있음 → H4 수정이 직접 대응.

---

## 4. 종합 판단

**가장 본질적 (구조적, 수정 어려움)**:
- H1 (단일 relation = 부분 교정) + H2 (capability ceiling)
- 이 둘이 맞으면 **7B에서 +1.5%p는 "정상"**. 큰 향상은 (a) 다중 relation 조합 또는 (b) capability 충분한 큰 모델에서 기대.

**방법론적 (수정 가능, 효과 키울 여지)**:
- H4 (context-gate) — H7 발견 2가 직접 지지. **최우선 개선 후보**.
- H3 (trajectory 벡터) — 분포 정합으로 벡터 품질 ↑.
- H5/H6 (layer/relation 선택) — Phase 2b로 탐색.

**현재 Phase 3 gating 상태**: lift ≥ +3%p in ≥2 모델 기준. 현재 Qwen +1.5%p, Hermes 미확정 → **기준 미달**. Phase 2b/H4 개선 권장.

---

## 5. 가정-실험 매핑 (실행 우선순위)

| 우선 | 가정 | 실험 | 준비 상태 | 예상 효과 |
|---|---|---|---|---|
| **1** | H4 | context-gate steering 구현 + raw add 비교 | 신규 (Phase 0 설계 존재) | 효과 ↑↑ (H7 직접 지지) |
| **2** | H1+H5 | Phase 2b relation sweep + layer sweep | `_phase2b_qwen_dual_gpu.sh` 준비됨 | 효과 ? (relation별 큰 차이 가능) |
| **3** | H7 | Hermes α=0.5 재실행 (49 infra 제거) + 도메인 재분석 | 즉시 가능 | 검증 보강 |
| 4 | H3 | trajectory 기반 벡터 재추출 | 신규 스크립트 | 효과 ↑ (분포 정합) |
| 5 | H6 | 성공 상관 기준 relation 재선택 | 신규 분석 | H1과 중첩 |
| 6 | H2 | 32B same steering | strategic gating (Phase 2-4 후) | 효과 ↑면 capability ceiling 입증 |

---

## 6. 다음 세션 실행 권장 순서

1. **Hermes α=0.5 재실행** (49 infra_error로 불량) — 깨끗한 α=0.5 데이터 확보 후 H7 Hermes 재분석.
2. **전체 α 그리드 종료 대기** (Qwen ext α=0.4/0.6 ~13:40, Hermes 본+ext ~18:50 KST).
3. **best α 확정** + 8-alpha U-curve 작성 (sanity + 7 alphas × 2 models).
4. **방향 결정**:
   - **H4 우선 (권장)**: context-gate steering 구현. 토큰/단계별 gating으로 손실 맥락 제거. H7 발견 2가 가장 직접적 근거.
   - **H1/H5 병행**: Phase 2b relation/layer sweep (스크립트 준비됨, GPU 비면 즉시).
5. **통계 보강**: N=30 trials=4 → CI overlap. 유의성 필요시 N=60 (60 tasks 또는 trials=8).

---

## 7. 핵심 수치 요약 (빠른 참조)

```
Qwen U-curve (clean):
  α=0.0  0.1765  (baseline)
  α=0.1  0.1750  (-0.15%p)
  α=0.3  0.1667  (-1.0%p)
  α=0.5  0.1917  (+1.5%p) ← peak
  α=1.0  0.1250  (-5.15%p)
  α=2.0  0.1356  (-4.09%p)
  [α=0.4, 0.6 진행 중 — peak 해상도 보강]

Hermes U-curve (α=0.5 불량):
  α=0.0  0.1237  (baseline)
  α=0.1  0.1083  (-1.5%p)
  α=0.3  0.1083  (-1.5%p)
  α=0.5  (재실행 필요)
  [α=1.0, 2.0, 0.4, 0.6 대기]

H7 핵심: Qwen Easy persona +10%p (희석됨), per-task ±0.5 swing 상쇄 → context-gate(H4) 근거
```

---

## 부록: 가정 검증을 위한 코드 위치

- 도메인/persona/task 분해 분석: 본 분석에 사용한 Python 스니펫 (per-bucket pass^1 + per-task flip)
- Phase 2b relation/layer sweep: `_phase2b_qwen_dual_gpu.sh`
- context-gate 구현 지점: `_steering_vllm_server.py`의 `patched_forward` — 현재 모든 위치에 더함. 토큰 조건/단계 조건 추가 필요.
- trajectory 벡터 추출: `_extract_steering_vectors.py` 변형 (contrast pair 대신 success/fail trajectory hidden state).
