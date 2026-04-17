# 실험 계획서 v1 — 논문 제출까지 전체 실험 로드맵

> **목적**: coworker 간 실험 협의용. 모든 실험의 목적, 조건, 순서를 명시.
> **날짜**: 2026-04-17 (update 2: Q-sign test intermediate + ToolBench scaling 추가)
> **타겟**: ICLR 2027 (Sep 2026 submission deadline 추정)
> **현재 상태**: τ²-bench 4 도메인 완료, Banking multipass_ladapt +5.64pp 발견. Q+ (β>0) Retail/Telecom 대폭 Lift 징후 (관측 중). ToolBench 스케일링 실험 준비 완료.

---

## 0. 최근 발견 요약 (2026-04-17 update 2)

### 0.1 새롭게 검증된 결과

| 벤치마크 | N | 방법 | F1 | Δ baseline | 상태 |
|----------|---|------|-----|-----------|------|
| τ² Retail | 114 | Q-only β=-0.03 | 0.5190 | **+5.11pp** | ✅ 재현됨 |
| τ² Retail | 114 | ladapt K+Q β=-0.03 | 0.4829 | +1.50pp | ✅ |
| τ² Telecom | 200 | Q-only β=-0.03 | 0.4349 | **+18.37pp** | ✅ 재현됨 |
| τ² Airline | 50 | ladapt K+Q β=-0.03 | 0.4875 | **+3.80pp** | ✅ |
| τ² Banking non-meta | 13 | **multipass_ladapt** k0.05_q-0.03 | **0.8128** | **+5.64pp** | ✅ **신규 SOTA** |
| τ² Banking non-meta | 13 | single-pass ladapt | 0.7564 | 0.00pp | baseline 수준 |
| τ² Banking 전체 | 97 | — | — | -5.99pp | meta-tool scope 밖 |
| MetaTool ST4 | 497 | multipass iterative_kq | 0.7635 | **+2.18pp** | ✅ |

### 0.2 관측 중 (intermediate, 완료 전)

| 벤치마크 | 방법 | 진행 | F1 | Δ baseline | 비고 |
|----------|------|------|-----|-----------|------|
| τ² Retail | **ocq_qbias_b+0.03** (Q 증폭) | 20/114 | 0.6662 | **+19.83pp** | ⚠️ handoff 예측 반대 방향 |
| τ² Telecom | **ocq_qbias_b+0.03** (Q 증폭) | 130/200 | 0.4376 | **+18.64pp** | ⚠️ Q-와 유사 크기 |

### 0.3 실험 우선순위 변경

- **P0 (긴급)**: Q-sign 확정 — retail/telecom Q+0.03 완전 실행 종료 후, airline/MetaTool에서도 Q+ 테스트
- **P0 (긴급)**: Multipass 교차 검증 — retail/telecom/airline에서도 `multipass_*` 테스트 (Banking에서 유일 승자였음)
- **P1 (신규)**: ToolBench 스케일링 실험 (Phase 2.7, 본 문서 새 섹션) — "K 값이 카탈로그 크기에 반비례" 가설 직접 검증
- P2: Llama cross-model (원래 Phase 1B)

---

---

## 1. 현재까지 확인된 것 (요약)

### 1.1 작동하는 것

| 방법 | 벤치마크 | N | F1 | Δ | 상태 |
|------|---------|---|-----|---|------|
| Q-only 전체 B_ont (β=-0.03) | MetaTool ST4, Qwen | 497 | 0.754 | **+2.30pp** | ✅ 재현됨 |
| layer-adaptive K early + Q (α=0.05, β=-0.05) | MetaTool ST4, Qwen | 497 | 0.751 | **+2.08pp** | ✅ |
| single-pass ladapt K+Q (α=0.05, β=-0.03) | MetaTool ST4, Qwen | 497 | 0.745 | **+1.40pp** | ✅ |
| multipass P_emitted + K early | MetaTool ST4, Qwen | 50 | 0.714 | +1.7pp | smoke만 |

### 1.2 실패한 것

| 방법 | 결과 | 원인 |
|------|------|------|
| K-only 전 레이어 (α=0.3) | -4.57pp | 후반 레이어에서 출력 수렴 파괴 |
| Q-only P_emitted iterative (v2) | +0.00pp | P_emitted=0 조건에서 hook이 no-op |
| stop-at-tool iterative | -23pp | 모델 context 깨짐, 2번째 도구 생성 불가 |
| SEKA amp=2.0 | -64pp | K-only stationary, 멀티 도구 구조적 한계 |

### 1.3 아직 모르는 것

- Llama-3.1-8B에서 layer-adaptive가 작동하는가?
- N=497 multipass P_emitted가 sign flip 없는가?
- MetaTool 이외 벤치마크에서 재현되는가?
- facet-weighted nDCG가 F1과 순위 역전을 보이는가?

---

## 2. 실험 축 (Experimental Axes)

논문에서 보여줘야 할 축은 5개:

```
축 1: 방법 비교 (Method)
  - no_steer (baseline)
  - K-only (v1)
  - Q-only 전체 B_ont (v0, beta_sweep 방식)
  - Q-only P_emitted multipass (이론적으로 정확한 방식)
  - Layer-Adaptive K+Q (v5, 최적)
  - SEKA (외부 baseline)
  
축 2: 모델 (Model)
  - Qwen2.5-7B-Instruct (주력)
  - Llama-3.1-8B-Instruct (cross-model)
  
축 3: 벤치마크 (Benchmark)
  - MetaTool Subtask4 (15 도구, 497 쿼리) — ablation용
  - Seal-Tools (4,076 도구, 10K 멀티도구 쿼리) — 핵심 실험
  - xLAM 60K 또는 ToolACE (3.7K~26K 도구) — 규모 검증 (선택)
  
축 4: 하이퍼파라미터 (Hyperparameter)
  - α_K: 0, 0.03, 0.05, 0.1
  - β_Q: 0, -0.03, -0.05, -0.1
  - layer boundary: L/4, L/3, L/2
  
축 5: 메트릭 (Metric)
  - F1, Exact Match, Facet-Weighted nDCG (구현 필요)
```

---

## 3. 실험 단계 (Phase)

### Phase 0: 현재 진행 중 (2026-04-17)

**목표**: MetaTool에서 최적 조합 확정

| ID | 실험 | 모델 | N | GPU | 상태 |
|----|------|------|---|-----|------|
| P0-1 | multipass_kq k_early α=0.05 β=-0.05 | Qwen | 497 | GPU0 | 🔄 실행 중 |
| P0-2 | single-pass ladapt K+Q β=-0.03 | Qwen | 497 | — | ✅ +1.40pp |
| P0-3 | single-pass Q-only β=-0.03 | Qwen | 497 | — | ✅ +2.30pp |

**P0 완료 기준**: iterative_kq N=497 sign flip 여부 확인

---

### Phase 1: MetaTool 완전 비교표 (2026-04-18~19)

**목표**: 논문 Table 1 — 모든 방법 × 2개 모델 × MetaTool

#### 1A. Qwen 완전 비교 (1일)

| ID | 방법 | Q 빼기 | K | 레이어 | α | β | 예상 시간 |
|----|------|--------|---|--------|---|---|-----------|
| 1A-1 | no_steer | — | — | — | 0 | 0 | 8분 |
| 1A-2 | K-only 전 레이어 | — | ✓ | 전체 | 0.05 | 0 | 10분 |
| 1A-3 | K-only 초기1/4 | — | ✓ | 0-6 | 0.05 | 0 | 10분 |
| 1A-4 | Q-only 전체 B_ont | 전체 B_ont | — | 전체 | 0 | -0.03 | 12분 |
| 1A-5 | Q-only P_emitted multipass | P_emitted | — | 전체 | 0 | -0.05 | 25분 |
| 1A-6 | **ladapt K early + Q (single-pass)** | 전체 B_ont | ✓ | K:0-6, Q:7-27 | 0.05 | -0.03 | 12분 |
| 1A-7 | **multipass P_emitted + K early** | P_emitted | ✓ | K:0-6, Q:전체 | 0.05 | -0.05 | 25분 |
| 1A-8 | SEKA amp=0.5 | — | SEKA | 전체 | — | — | 10분 |
| 1A-9 | SEKA amp=1.0 | — | SEKA | 전체 | — | — | 10분 |

**총 GPU 시간**: ~2시간 (Qwen, 1 GPU)

#### 1B. Llama Cross-Model 검증 (1일)

| ID | 방법 | 비고 |
|----|------|------|
| 1B-1 | no_steer | Llama baseline |
| 1B-2 | Q-only 전체 B_ont β=-0.03 | Q-only 재현 |
| 1B-3 | K-only 초기1/4 α=0.05 | K early 효과 |
| 1B-4 | ladapt K+Q (α=0.05, β=-0.03) | 최적 조합 |
| 1B-5 | multipass P_emitted + K early | 이론 검증 |
| 1B-6 | SEKA amp=0.5 | 외부 baseline |

**필요 사항**: Llama용 B_ont 빌드 (build_llama_metatool_b_ont.py 있는지 확인 필요)
**총 GPU 시간**: ~2시간 (Llama, 1 GPU)

#### 1C. 하이퍼파라미터 sweep (선택, 1일)

ladapt 최적 조합에서:
- α sweep: {0.01, 0.03, 0.05, 0.1} × β sweep: {-0.01, -0.03, -0.05, -0.1}
- layer boundary sweep: {L/6, L/4, L/3, L/2}
- **총 GPU 시간**: ~4시간 (16 조합 × 15분)

---

### Phase 2: Seal-Tools 벤치마크 (2026-04-20~22)

**목표**: 4,076 도구 × 10K 멀티도구 쿼리에서 우리 방법 검증

#### 2A. 데이터 준비 (0.5일)

| 작업 | 내용 |
|------|------|
| 다운로드 | `git clone https://github.com/fairyshine/Seal-Tools` |
| 데이터 분석 | 도구 카테고리 구조 파악, 멀티도구 쿼리 추출 |
| B_ont 빌드 | Seal-Tools 146분야 → 4-facet 온톨로지 매핑 → B_ont 구축 |
| 평가 스크립트 | eval_seal_tools.py 작성 (MetaTool 파이프라인 기반) |

**핵심 설계 결정**:
- Seal-Tools의 146분야를 domain facet으로 매핑
- 도구 function type을 function_action facet으로
- 도구 input/output type을 io_type facet으로
- 도구 카테고리를 tool_category facet으로

#### 2B. Seal-Tools 실험 (1.5일)

| ID | 방법 | 모델 | N | 비고 |
|----|------|------|---|------|
| 2B-1 | no_steer | Qwen | 500 (smoke) → 전체 | baseline |
| 2B-2 | Q-only 전체 B_ont | Qwen | 전체 | best known method |
| 2B-3 | ladapt K+Q | Qwen | 전체 | 최적 조합 |
| 2B-4 | multipass P_emitted + K early | Qwen | 전체 | 이론 검증 |
| 2B-5 | no_steer | Llama | 전체 | cross-model |
| 2B-6 | ladapt K+Q | Llama | 전체 | cross-model |

**총 GPU 시간**: ~8시간 (4,076 도구라 추론 시간 증가 예상)

#### 2C. 규모 분석 (0.5일)

| 실험 | 내용 |
|------|------|
| 도구 수 scaling | 50, 200, 500, 1000, 4076 도구로 subset 생성, F1 vs 도구 수 곡선 |
| facet 수 scaling | 2, 3, 4 facet 조합으로 B_ont 변형, F1 vs facet 수 |

---

### Phase 2.7 (신규): ToolBench 스케일링 실험 (2026-04-17~19)

**동기**: handoff §7 "K 값이 도구 카탈로그 크기와 태스크 horizon 길이에 **반비례**" 가설은 현재까지 실측 근거 없음. 기존 벤치마크 (MetaTool 15, τ² 5-17) 는 모두 작은 카탈로그. 카탈로그 크기 축을 실제로 sweep 해야 가설 검증 가능.

**데이터**: StableToolBench solvable_queries (`external/StableToolBench/solvable_queries/test_instruction/`)
- 6 subset × 61~163 쿼리 = 총 765 쿼리
- 각 쿼리는 자체 `api_list` (4~13 API) + `relevant APIs` (GT, 2~6 개)
- 전 subset 내 unique `(tool, api)` pair = 814 (distractor pool)
- 47 카테고리 분포

**B_ont 빌드**: `scripts/ocq/build_toolbench_ontology.py` (완료) → `reports/axis2_theoretical_verification/toolbench_ontology.json` (46 카테고리, 평균 28 문장/카테고리) → 기존 `build_qwen_metatool_b_ont.py`로 per-(L, h) basis 빌드. Output: `external/SEKA/seka_projections/ontology-qwen25-7b-toolbench/B_ont.pt`.

**평가 스크립트**: `scripts/ocq/eval_toolbench.py` (완료). 핵심 매커닉:
1. 각 쿼리의 native `api_list` + N 개 distractor (크로스 쿼리 pool에서 샘플링) 을 합쳐 tools_json 구성
2. tokenizer chat template로 prompt 빌드 → greedy generate → JSON `{"name": ...}` 추출
3. display_id = `"TOOLNAME__APINAME"` (정규화된 유효 식별자)
4. F1, Exact, Recall, Precision, GT_subset, nDCG 집계

**실험 축**:

| 축 | 값 |
|----|----|
| 카탈로그 크기 (distractors) | 0, 50, 200, 500 |
| 방법 | no_steer, ocq_qbias_b-0.03, ocq_qbias_b+0.03, ocq_ladapt_k0.05_q-0.03 |
| Subset | G1_instruction (163), G2_instruction (106), G3_instruction (61) |

**예상 소요시간** (Qwen2.5-7B, A6000, max_new_tokens=256):
- subset 당 쿼리 × 4 distractor × 4 method = 16 runs
- 각 run 평균 ≈ 쿼리수 × 5s = ~10–15 분
- subset 당 ≈ 3–4 시간
- 3 subset 총 ≈ 9–12 시간 (GPU 1개 기준)

**핵심 가설**:

| ID | 가설 | 예측 (distractors 0 → 500) |
|----|------|---------------------------|
| H-TB-1 | K-only 효과는 카탈로그 확대에 따라 **감소** | α=0.05 lift: +5pp → +1pp → 0 → -2pp |
| H-TB-2 | Q-only (β sign 상관없이) 효과는 카탈로그 확대에 따라 **증가 또는 유지** | +3pp → +3pp → +4pp → +5pp |
| H-TB-3 | ladapt는 Q-only와 유사 수준 (K 기여가 줄어서) | Q-only와 차이 줄어듦 |
| H-TB-4 | Q-sign (양/음) 의 상대 우위가 카탈로그 크기로 바뀔 수 있음 | β+ 유리 → β- 유리 flip 혹은 역전 |

**실행 스크립트**: `scripts/ocq/run_toolbench_sweep.sh` (subset 루프, 4 distractor 레벨)

**담당**: Claude (GPU 실험). 결과 해석은 coworker 협의.

---

### Phase 2.8 (신규): Multipass Cross-Benchmark 검증 (2026-04-17~18)

**동기**: Banking non-meta에서 **단일 pass ladapt는 baseline과 동일 (0.0pp)** 인데 **multipass ladapt는 +5.64pp**. MetaTool에서도 multipass iterative_kq = +2.18pp로 최고. Multipass가 cross-domain 공통 이점일 가능성 시사.

**실험**:

| ID | 벤치마크 | 방법 | 비교 대상 | 예상 시간 |
|----|---------|------|-----------|----------|
| 2.8-1 | τ² Retail (114) | multipass_ocq_qbias_b-0.03 | single-pass Q-0.03 (+5.11pp) | 30분 |
| 2.8-2 | τ² Retail (114) | multipass_ocq_ladapt_k0.05_q-0.03 | single-pass ladapt (+1.50pp) | 30분 |
| 2.8-3 | τ² Telecom (200) | multipass_ocq_qbias_b-0.03 | single-pass Q-0.03 (+18.37pp) | 30분 |
| 2.8-4 | τ² Airline (50) | multipass_ocq_ladapt_k0.05_q-0.03 | single-pass ladapt (+3.80pp) | 15분 |

**가설**: multipass 가 retail/telecom/airline 에서도 +1~5pp 추가 lift를 주면 논문 main contribution 이 될 수 있음.

**담당**: Claude (GPU 실험).

---

### Phase 3: 메트릭 구현 + 재평가 (2026-04-23~24)

**목표**: FW-nDCG 구현 후 전 실험 결과 재평가

| 작업 | 내용 |
|------|------|
| FW-nDCG 구현 | compute_metrics()에 facet-weighted nDCG 추가 |
| 재평가 | Phase 1~2의 모든 JSON에 nDCG 추가 계산 (forward pass 불필요, 저장된 pred에서 계산) |
| 순위 역전 분석 | F1 ranking vs nDCG ranking 비교 → 어떤 케이스에서 역전이 발생하는지 |
| 논문 Table | F1 + Exact + nDCG 3열 비교표 |

---

### Phase 4: 추가 벤치마크 (선택, 2026-04-25~27)

**조건**: Phase 2까지 결과가 긍정적일 때만 진행

| ID | 벤치마크 | 목적 | GPU 시간 |
|----|---------|------|----------|
| 4A | xLAM 60K (HF) | 3.7K 도구 규모 재현 | ~4시간 |
| 4B | ToolACE (HF, ICLR 2025) | 26K 도구 극한 규모 | ~8시간 |
| 4C | BFCL parallel subset | 커뮤니티 표준 벤치마크 | ~2시간 |

---

### Phase 5: 논문 작성 + 정리 (2026-04-28~05-04)

| 작업 | 내용 |
|------|------|
| PAPER_DRAFT_v4.md 최종 업데이트 | 모든 실험 결과 반영 |
| 영문 논문 초안 | ICLR 2027 format, 8+α 페이지 |
| Figure 제작 | 레이어별 MSE U-shape, F1 비교 bar chart, scaling curve |
| 수학 증명 정리 | THEOREM_SUPPLEMENTS 내용을 Appendix로 |
| 코드 정리 | reproduce scripts + README |

---

## 4. GPU 자원 계획

```
가용 GPU: A6000 × 2 (48GB each)
모델 로딩: Qwen 7B = ~15GB, Llama 8B = ~17GB

Phase 0: 1 GPU, 0.5일 (진행 중)
Phase 1: 2 GPU 병렬 (Qwen GPU0 + Llama GPU1), 1일
Phase 2: 2 GPU 병렬, 2일
Phase 3: CPU만 (재계산), 1일
Phase 4: 2 GPU 병렬, 2일 (선택)
Phase 5: CPU만 (작성), 5일

총 GPU 일수: ~5일 (Phase 4 포함 시 7일)
총 일정: ~18일 (2026-04-17 ~ 2026-05-04)
```

---

## 4B. Phase 2.5 추가 실험 — Layer Sweep (coworker 요청)

**목적**: ladapt에서 K/Q 레이어 분배가 F1에 미치는 영향 측정. 현재는 k_boundary_frac=0.25 고정만 테스트.

**실험 대상 (retail 도메인 기준, L=28)**:

| 실험 ID | K 적용 레이어 | Q 적용 레이어 | 목적 |
|---------|-------------|-------------|------|
| LS-1 | 0 ~ L/7 (L0-L3) | L/7 ~ L (L4-L27) | K 더 좁게 |
| LS-2 | 0 ~ L/4 (L0-L6) | L/4 ~ L (L7-L27) | 현재 baseline |
| LS-3 | 0 ~ L/3 (L0-L9) | L/3 ~ L (L10-L27) | K 더 넓게 |
| LS-4 | 0 ~ L/2 (L0-L13) | L/2 ~ L (L14-L27) | K 반 |
| LS-5 | 0 ~ L/4 (L0-L6) | **전체 (L0-L27)** | K+Q 초기 겹침 |
| LS-6 | 0 ~ L/4 (L0-L6) | 후반 1/4 (L21-L27) | Q 후반만 |

**파라미터**: α=0.05, β=-0.03, β=-0.05 두 값 테스트

**벤치마크**:
- MetaTool Subtask4 N=497 (통제 실험, 빠름)
- τ²-bench retail N=114 (실전 검증)

**예상 GPU 시간**: 6 configs × 2 β × 2 benchmarks = 24 runs × 15분 ≈ 6시간

**구현**: `eval_tau2_bench.py`에 `ocq_ladapt_k<α>_q<β>_f<k_frac>` 포맷 이미 지원됨 (k_frac 파라미터만 변경). `q_late_only` 모드 추가 필요 (scripts/ocq/eval_subtask4_dynamic_qk_v2.py에 이미 구현됨).

**담당**: **coworker (승필)** — Claude는 Phase 1-2 핵심 실험에 집중

---

## 5. 작업 분배 (제안)

| 작업 | 담당 | 비고 |
|------|------|------|
| Phase 0~1: MetaTool 완전 비교 | Claude (자동) | GPU 실험 실행 |
| Phase 2: τ²-bench 전 도메인 | Claude (자동) | 진행 중 |
| **Phase 2.5: Layer Sweep** | **coworker (승필)** | **신규 추가** |
| Phase 2A: Seal-Tools 데이터 분석 + B_ont 설계 | **협의 필요** | 146분야 → 4-facet 매핑 검증 |
| Phase 2B: Seal-Tools 실험 | Claude (자동) | GPU 실험 실행 |
| Phase 3: FW-nDCG 설계 | **협의 필요** | facet 가중치 정의 검증 |
| Phase 5: 논문 한국어 초안 | Claude | |
| Phase 5: 논문 영문 + LaTeX | **coworker** (main에서 작업 중) | |
| 수학 증명 리뷰 | **협의 필요** | Thm 6.17' 수정사항 검증 |

---

## 6. Kill Switch 기준

각 Phase에서 진행/중단 결정 기준:

| Phase | 진행 기준 | 중단 시 대안 |
|-------|----------|-------------|
| Phase 1 | Qwen에서 최소 1개 방법이 +1.5pp 이상 | MetaTool 음수면 ablation-only 논문으로 전환 |
| Phase 1B | Llama에서 같은 방향 (양수) | Llama 음수면 Qwen single-model 논문 |
| Phase 2 | Seal-Tools에서 최소 +1pp 또는 방향 일관 | Seal-Tools 실패 시 MetaTool-only + "future work" |
| Phase 3 | nDCG가 F1과 다른 insight 제공 | nDCG가 F1과 동일하면 F1만 보고 |

---

## 7. 이전 세션에서 coworker가 언급한 실험들 (승필)

> "qwen full-497 small-alpha Q+K: 최고 0.7529 F1"

이 결과는 `reports/qkv_alpha_microsweep_2026_04_15/full497_alpha_microsweep.json`에 있음.
→ Phase 1A에서 동일 조건 재현 포함 (α sweep 포함)

> "Llama full-497에서 K-bias는 붕괴, Q-bias만 약한 양수"

→ Phase 1B에서 Llama layer-adaptive 검증 (K early면 붕괴 안 할 수 있음)

---

## 8. 논의 필요 사항 (Coworker에게)

1. **Seal-Tools의 facet 매핑**: 146분야를 우리 4-facet 중 어디에 대응시킬지 — domain? tool_category?
2. **B_ont 빌드 방법**: Seal-Tools에 맞는 B_ont를 어떻게 빌드할지 — MetaTool 방식 그대로? 아니면 Seal-Tools 카테고리에서 직접?
3. **Llama B_ont**: 이미 빌드되어 있는가? 없으면 빌드 우선순위는?
4. **논문 프레이밍**: "Layer-Adaptive Q+K"가 메인 contribution인가, 아니면 "ontology-based attention regularization"이 메인인가?
5. **nDCG relevance score**: facet 일치 개수 vs facet 에너지(ε_f) 가중 — 어느 쪽이 현장에 맞는가?
6. **코드 정리 범위**: 실험 스크립트 전부 정리할지, 핵심만 정리할지

---

## 10. 가설 상태표 (2026-04-17 update 2)

각 가설의 최신 증거 상태. ✅ 지지, ❌ 반박 or 수정, 🔶 부분 지지/미완, ❓ 미검증.

| ID | 가설 | 상태 | 증거/비고 |
|----|------|------|-----------|
| **H-A-1** | Q-only가 Q+K보다 유리 (Long-horizon) | 🔶 부분 | Telecom (12 액션) Q-only +18.37pp > ladapt, but Q+ 도 비슷한 lift |
| **H-A-2** | K의 정확도 보조는 Short-horizon(3-5)에서만 유효 | ✅ | Airline (short) ladapt +3.80pp > Q-only |
| **H-B** | **Q 증폭 (β > 0) 은 harmful** | ❌ **반박 징후** | Retail Q+0.03 F1=0.6662 (+19.83pp, 20/114 intermediate); Telecom Q+0.03 (+18.64pp, 130/200) — full run 확정 대기 |
| **H-C** | Banking 순위: ladapt β=-0.05 > β=-0.03 > multipass > AdaSEKA > K-only | ❌ **반박** | 완료 결과: **multipass_ladapt** (+5.64pp) 만 승자. 단일 pass ladapt = 0pp = baseline |
| **H-D** | Banking meta-tool (discoverable/agent) 은 static ontology scope 밖 | ✅ | 전체 97 task -5.99pp, non-meta 13 만 분리하면 +5.64pp |
| **H-E-1** | "K의 값은 도구 카탈로그 크기에 반비례" | ❓ **ToolBench 실험 대기** | Phase 2.7 에서 직접 sweep 예정 |
| **H-E-2** | "K의 값은 horizon 길이에 반비례" | 🔶 부분 | Telecom(12)에서 ladapt vs Q-only 차이 작음. 그러나 Q+의 lift가 H-B와 섞여 노이즈 |

### 10.1 새로운 (추가된) 가설

| ID | 신규 가설 | 검증 계획 |
|----|-----------|-----------|
| **H-NEW-1** | Q-sign 은 도메인의 ontology alignment 품질에 의존. 잘 맞은 B_ont 에서는 Q+ amplify 가 선호될 수 있음 | Q+ 와 Q- 를 airline, MetaTool, ToolBench 에서 교차 측정. Per-query GT F1 vs query-ontology similarity 상관 분석 |
| **H-NEW-2** | **Multipass 가 cross-domain 공통 이점**. 단일 attempt miss 를 재시도로 복구 | Phase 2.8: retail/telecom/airline multipass 적용 |
| **H-NEW-3** | Ontology = "semantic attractor". Q+ 는 tool-schema attention 강화 → 정답 tool 선택률 상승 | Per-token attention weight 시각화 (도메인별). Q+ 에서 정답 tool 에 대한 attention 변화 측정 |
| **H-NEW-4** | **Single-pass ladapt 에서 K hook + Q hook 간 GQA interaction** 이 Banking에서 효과 상쇄 | K-only, Q-only, ladapt 를 동일 설정에서 비교. multipass 로 복구되는 이유가 GQA 평균화 때문이라면 K-only를 multipass 로 감싸도 유사 복구될 것 |
| **H-NEW-5** | ToolBench scaling 에서 K-only 는 감소, Q-only 는 유지/증가 | Phase 2.7 의 distractor sweep 에서 직접 측정 |

### 10.2 Coworker 협의 필요 질문

1. **Q+ amplify 결과가 확정되면** 논문 프레이밍은 "subtraction as coverage regularization" → "signed ontology steering" 으로 재구성해야 함. 기존 Thm 6.17' 서술 수정 여부?
2. **Multipass 메커니즘의 이론적 해석**: 단순 재시도? generation distribution 의 mode-covering? Thm 6.18 (attn-weighted bit alloc) 와 어떤 관계?
3. ToolBench scaling 실험의 distractor 샘플링 방법: uniform vs 같은 카테고리 oversample vs adversarial (GT에 가까운 tool들)? 현재는 uniform.
4. FW-nDCG 는 계획대로 Phase 3 진행? 아니면 ToolBench 스케일링 결과 보고 우선순위 조정?

---

## 9. 일정 타임라인

```
2026-04-17 (오늘)  Phase 0 완료 + 이 계획서 협의
2026-04-18~19      Phase 1: MetaTool 완전 비교 (Qwen + Llama)
2026-04-20~22      Phase 2: Seal-Tools 벤치마크
2026-04-23~24      Phase 3: FW-nDCG 메트릭
2026-04-25~27      Phase 4: 추가 벤치마크 (선택)
2026-04-28~05-04   Phase 5: 논문 작성
2026-05-04         논문 초안 완성
2026-05-15 (목표)  최종 제출
```
