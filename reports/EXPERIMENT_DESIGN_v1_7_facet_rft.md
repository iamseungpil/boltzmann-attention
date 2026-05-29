# 실험 설계서: 온톨로지 기반 그래프 구조 주입을 통한 Training-Free 다단계 도구 계획 개선

**버전**: v1.23  
**작성일**: 2026-05-26  
**최종 갱신**: 2026-05-29 (v1.21: Phase 2a/2c/2d steering 실측 — 상수 single-relation steering null, steering↔RFT class-hierarchy, LoRA-RFT 피벗; v1.22: cross-domain 전이 축(4도메인 telecom/retail/airline/banking) 중심 격상 + 합성-온톨로지 북극성; v1.23: facet-guided distillation lever. 상세: phase2_steering/STEERING_CONTROL_DESIGN.md §1-11)  
**목표 학회**: NeurIPS 2026 / ICLR 2027  

---

## 1. 연구 배경 및 문제 정의

### 1.1 핵심 문제

기업 환경의 AI 에이전트는 수십~수백 개의 도구(API)를 조합하여 다단계 업무 계획을 수립해야 한다. 특허(OISA v5)에서 정의한 바와 같이, 45개 도구 기준으로 가능한 계획 경로는 85,000개 이상이다. 이 탐색 공간에서 올바른 계획(도구 호출 순서, 전제조건 충족, 상호 배타 관계 준수)을 수립하는 것이 핵심 과제다.

### 1.2 기존 접근의 한계

| 접근 방식 | 한계 |
|---|---|
| **Channel A** (프롬프트 포맷 변형) | 모델 학습 포맷 편향과 confound됨. F1은 순서 오류를 측정 못함 |
| **GraphRAG 텍스트 직렬화** | Graph → Text 변환 시 위상 정보 손실. Attention 희석 |
| **GMT (Cross-Attention)** | KGC 태스크 특화. 계획 수립 적용 선례 없음. 학습 필요 |
| **GAP / Routine** | SFT + RL 또는 Fine-tuning 필수. 새 도메인 적응 어려움 |
| **KnowAgent** | 행동 지식을 텍스트로 직렬화. 구조적 제약 강제 불가 |

### 1.3 연구 기회

아래 두 트렌드가 교차하는 지점이 미개척 상태:

```
[Graph/KG → LLM 구조 주입 연구]     [Vector Steering → 추론 개선 연구]
  GMT, FLAME, K-BERT 등               CoT Steering, VSPO, Bias-Only 등
              ↓                                    ↓
         ┌──────────────────────────────────────────┐
         │  온톨로지 관계 → LLM 내부 표현 주입 →      │
         │  Test-Time 다단계 계획 정확도 향상          │
         │  (선례 없음 = 연구 gap)                    │
         └──────────────────────────────────────────┘
```

---

## 2. 연구 가설

### 주 가설 (H_main)

> **기업 도구 온톨로지에서 추출한 관계 구조(precondition / postcondition / workflow_role / mutex)를 LLM의 내부 표현에 직접 주입하면, LLM 재학습 없이 다단계 도구 계획의 정확도(pass^1)가 유의미하게 향상된다.**

### 세부 가설

| 가설 | 내용 | 검증 방법 |
|---|---|---|
| **H1** | Cross-Attention 주입이 텍스트 직렬화보다 도구 의존성 제약 준수율이 높다 | FlowBench: 포맷별 비교 |
| **H2** | 온톨로지 구조 주입이 새 도메인에 대한 zero-shot 일반화를 개선한다 | τ²-bench: 도메인 교차 평가 |
| **H3** | 병렬/직렬 도구 의존성 그래프 주입이 계획 실행 효율을 개선한다 | TPS-Bench: 완료율 + 실행시간 |
| **H4** | Training-free 방법이 cold-start 상황(학습 데이터 0)에서 fine-tuned 방법보다 비용 효율적이다 | Routine 벤치마크: 학습량 대비 성능 |

---

## 3. 제안 방법론

### 3.1 전체 아키텍처

```
[오프라인 단계: 온톨로지 구축 및 벡터 추출]

Tool Schema (YAML/JSON)
        │
        ▼
  [AFOD: 온톨로지 관계 자동 발견]
        │
        │  [방향성 관계 → A6 Rotation]
        ├─ precedes(A, B):          A는 B보다 먼저 호출
        ├─ requires(A, B):          A 호출 전 B가 성공해야 함
        ├─ enables(A, B):           A 후 B가 가능해짐
        ├─ parameter_feeds(A, B, p): A 출력이 B 입력
        ├─ validates(A, B):         A가 B 결과를 검증
        ├─ retry_after_fail(A, B):  A 실패 후 B → 재시도
        ├─ compensates(A, B):       A가 B 효과를 역전
        │  [대칭/범주형 관계 → T1 Additive]
        ├─ mutex(A, B):             동시 호출 불가
        ├─ parallel_safe(A, B):     순서 무관, 병렬 가능
        ├─ conditional_on(A, B, c): 조건부 호출
        ├─ precondition_state(A, P): 호출 전 상태 요구
        └─ workflow_role(A):        prerequisite/main/validation/cleanup
        │
        ▼
  [관계별 Contrast Pair 생성]
  (올바른 순서 예시 N개 vs 위반 예시 N개)
        │
        ▼
  [Steering Vector 추출]  +  [Cross-Attention Module 학습]
  v_precedes ∈ ℝ^d           K_onto, V_onto ∈ ℝ^{m×d}
  v_requires ∈ ℝ^d
  v_mutex    ∈ ℝ^d
        │
        ▼
  [Ontology Vector Bank 저장]


[온라인 단계: Test-Time 계획 생성]

Query (사용자 요청)
        │
        ▼
  [관련 온톨로지 서브그래프 검색]
  (관련 도구 집합 → 해당 관계 추출)
        │
        ├─────────────────────────────────────┐
        ▼                                     ▼
  [Method A: Cross-Attention 주입]    [Method B: Vector Steering]
  h'_t = h_t + Attn(Q, K_onto, V_onto)  h'_t = h_t + Σ αᵢ · vᵢ
        │                                     │
        └─────────────┬───────────────────────┘
                      ▼
              [LLM 계획 생성]
              (Qwen2.5-7B, LLaMA-3 등 오픈소스)
                      │
                      ▼
              [τ²-bench Simulator]
              DB 최종 상태 해시 비교
                      │
                      ▼
                  pass^1 / pass^5
```

### 3.2 온톨로지 설계 — v2 확장 (5종 → 12종)

#### 3.2.1 기존 facet과의 차이

```
기존 (action, domain) facet:            새 온톨로지 관계:
  action = "modify"                       precedes(verify_identity,
  domain = "account"                               cancel_subscription)
                                          requires(calculate_penalty,
  → 개별 도구의 레이블                             get_account_status)
  → 관계 구조 없음                          mutex(apply_discount,
  → discriminating power 낮음                      apply_penalty)
                                          workflow_role(verify_identity)
                                                    = prerequisite
                                          → 도구 간 관계 구조
                                          → 계획 순서에 직접 영향
```

#### 3.2.2 온톨로지 서베이 기반 관계 확장 (42종 최종)

서베이 대상: Routine(2025), GAP(NeurIPS 2025), PDDL 고전 계획, BPMN/Petri Net 프로세스마이닝, KnowAgent(NAACL 2025), GoT(Besta et al. 2023), ToT(Yao et al. 2023), HTN(고전 AI 계획), GoalAct(arXiv 2504.16563)

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Group A–F: 기본 27종 (v1.6에서 확정, Phase 0 v3 probing 대상)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  [A] precedes, requires, enables, mutex, workflow_role     (기초 5종)
  [B] parameter_feeds, conditional_on, validates,
      retry_after_fail, compensates, parallel_safe,
      precondition_state                                    (서베이 7종)
  [C] causal_link, directly_follows                        (인과 2종)
  [D] error_fallback, tool_subsumes                        (복구/추상화 2종)
  [E] and_join, state_transition, exclusive_choice,
      effect_state                                         (프로세스 4종)
  [F] domain_category, checkpoint, idempotent, reversible,
      mandatory_in_flow, optional_in_flow, loop_capable    (속성 7종)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Group G: Graph/Tree of Thoughts + Harness Engineering (6종)
  출처: GoT(Besta et al. 2023), ToT(Yao et al. 2023)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  fan_out(src, [t1,t2,...])     GoT: 후보 생성(Generate)
  pruned_by(tool, scorer)       GoT: 후보 가지치기(Prune)
  scored_preference(A, B, ctx)  GoT: 품질 기반 선호 (Score/Keep)
  backtrack_to(dead_end, restore_point)
                                ToT: 상태 복원(Backtrack)
                                ※ 도구 대체(T1 retry_after_fail)와 다름:
                                   dead_end는 도구, restore_point는 상태
  observation_triggers(obs, tool)
                                Harness: 반응적 obs→도구 매핑
  guardrail(forbidden_tool)     Harness: 호출 금지 제약

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Group H: HTN 계층적 태스크 네트워크 (4종)
  출처: 고전 HTN AI 계획 (Erol et al. 1994 계통)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  decomposes_into(goal, [tool_set])
                                상위 목표 → 도구 집합으로 분해
  subtask_of(tool, goal)        도구 → 목표 포함 관계 (도메인 카테고리)
  achieves_goal(tool, goal)     도구 호출이 목표를 달성
  refines(abstract_act, concrete_tool, ctx)
                                추상 행동 → 구체 도구로 정제

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Group I: GoalAct 기반 목표-계획 생명주기 (5종)
  출처: GoalAct arXiv 2504.16563 (G_t = π(Q|T|S_t))
  ※ "주기적 목표 환기"가 아님 — 실행 이력 S_t 기반 연속적 전역 플랜 재작성
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  plan_step_precedes(step_i, step_j)
                                GoalAct 플랜 내 추상 단계 순서
                                (도구보다 상위 추상 레벨)
  plan_step_skill(step, skill_type)
                                step → {searching, coding, writing, finish}
                                4종 skill type으로 분류 (GoalAct §3.1)
  plan_revised_to(obs_trigger, old_step, new_step)
                                S_t 기반 플랜 스텝 교체
                                (GoalAct의 핵심 메커니즘: G_t 재작성)
  step_realizes_tool(step, tool)
                                추상 계획 단계 → 구체 도구 (수직 브릿지)
  plan_committed_to_goal(step, goal)
                                플랜 스텝이 목표에 커밋

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
어휘 확장 (Group I 지원)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  GOAL_VOCAB (16종):
    L3: restore_mobile_service, resolve_account_issue,
        configure_advanced_features
    L2: restore_suspended_service, restore_data_connectivity,
        enable_international_access, resolve_billing_issue,
        restore_app_functionality, fix_network_configuration,
        fix_device_state, restore_direct_connection,
        fix_sim_connectivity, escalate_unresolved_issue,
        enable_wifi_calling, initiate_billing_resolution,
        manage_data_usage

  PLAN_STEP_VOCAB (12종):
    identify_root_cause, apply_targeted_fix, verify_resolution,
    close_or_escalate, gather_account_context, check_system_status,
    attempt_quick_fix, escalate_or_document, confirm_customer_outcome,
    apply_policy_action, sequence_tool_calls, monitor_after_fix

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
경험적 빈도 (τ²-bench 로그 실측, 관계 강도로 활용)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  directly_follows(A, B, freq)  예: reset_apn→reboot (1036회)
  → precedes 관계의 강도 가중치 α_rel로 사용
  → directly_follows ⊆ precedes이므로 별도 관계가 아닌 속성으로 처리
```

#### 3.2.3 관계 유형별 기하학적 분류 (42종 전체, 방법 배정 근거)

```
방향성(Directional) → A6 Per-head Rotation 후보:
  precedes, requires, enables, parameter_feeds,
  validates, retry_after_fail, compensates,
  causal_link, directly_follows,
  fan_out, pruned_by, backtrack_to,
  decomposes_into,
  plan_step_precedes, step_realizes_tool, plan_committed_to_goal
  근거: R(A,B) ≠ R(B,A) — 비대칭성이 방향 벡터로 표현됨
        SO(d_head) 회전이 cosine 기하에서 방향만 바꿈

대칭(Symmetric) → T1 Additive 후보:
  mutex, parallel_safe, and_join
  근거: R(A,B) = R(B,A) — 쌍방향 제약

조건부(Conditional) → T1 Additive 또는 혼합:
  conditional_on, scored_preference, refines, plan_revised_to
  근거: 조건 c의 크기가 개입 강도를 결정
        magnitude shift가 조건부 활성화에 자연스러움

범주형(Categorical) → T1 Additive:
  workflow_role, domain_category, checkpoint, idempotent,
  reversible, mandatory_in_flow, optional_in_flow, loop_capable,
  precondition_state, effect_state, state_transition,
  exclusive_choice, error_fallback, tool_subsumes,
  observation_triggers, guardrail,
  subtask_of, achieves_goal,
  plan_step_skill
  근거: 단항 속성 또는 유형 레이블 — 크기 shift가 존재/부재 표현에 적합

→ Phase 0 v3 probing에서 42종 전체의 기하학 예측 검증 (A7, §4.3 참조)
```

### 3.3 구현 방법 비교

#### 3.3.1 기본 lever 비교

| 구성 | 방법 A: Cross-Attention | 방법 B: Vector Steering | 방법 C: 하이브리드 |
|---|---|---|---|
| **원리** | 온톨로지 노드를 K,V로 변환, 매 레이어 조회 | 관계 벡터를 residual stream에 가산 | A + B 결합 |
| **주입 위치** | Transformer 중간 레이어 cross-attn | Residual stream (h_t + αv) | 레이어별 차등 |
| **추가 학습** | Cross-attn 모듈만 경량 학습 (LoRA) | 없음 (pure test-time) | Cross-attn LoRA만 |
| **동적 그래프** | 가능 (K,V 교체) | 벡터 재계산 필요 | 가능 |
| **표현력** | 높음 (노드별 선택적 attention) | 중간 (벡터 합으로 압축) | 높음 |
| **구현 복잡도** | 중간 | 낮음 | 높음 |
| **선행 연구** | GMT (KGC), K-BERT | RepE, CAA, Bias-Only Steering | 미존재 |

#### 3.3.2 합성 방법 — 3 lever (T2 × T4 × T1) 조합 (2026-05-27 v1.11 추가)

3 lever는 직교적 (서로 다른 메커니즘, 합성 가능):
- **T2** = 새 architectural module (cross-attn block 삽입)
- **T4** = weight 자체 변형 (RFT로 LoRA delta update)
- **T1** = runtime additive intervention (inference-time hidden state hook)

따라서 8가지 조합 가능 (2³ ablation matrix):

| 조건 | T2 | T4 | T1 | 명명 | 학습 | Phase 시점 |
|---|---|---|---|---|---|---|
| B0 | ❌ | ❌ | ❌ | Vanilla | 없음 | Phase 1 |
| **T1@base** | ❌ | ❌ | ✅ | Steering-only | 없음 | Phase 2a |
| T2 | ✅ | ❌ | ❌ | Cross-Attn LoRA | LoRA | Phase 3 |
| **T3 = T2 + T1@T2** | ✅ | ❌ | ✅ | Hybrid | LoRA | Phase 2b |
| **T4-RFT** | ❌ | ✅ | ❌ | RFT-only | RFT | Phase 4 |
| **T5 = T4 + T1@T4** | ❌ | ✅ | ✅ | RFT + Steering (사용자 thesis) | RFT | Phase 2c (조건부) |
| T2+T4 | ✅ | ✅ | ❌ | LoRA + RFT | LoRA→RFT | Phase 5 옵션 |
| **T6 = T2 + T4 + T1@(T2+T4)** | ✅ | ✅ | ✅ | Triple | LoRA→RFT | Phase 5 (조건부) |

**Lever 직교성 매트릭스** — 합성 시 충돌 없음:

|  | T2 (architecture) | T4 (weight) | T1 (runtime) |
|---|---|---|---|
| 메커니즘 | 새 모듈 추가 | 기존 weight ↔ LoRA delta | inference hook |
| 작동 시점 | 모든 forward | 모든 forward | 선택된 layer forward |
| 도메인 전환 비용 | LoRA 재학습 | RFT 재학습 | vector 재계산 |
| 학습 데이터 | 74 train task | rollout + verifier | 200 contrast pair |
| 합성 위험 | RFT가 cross-attn overwrite | T1이 학습 분포 벗어남 | (낮음) |

**Compositional hypothesis** (사용자 thesis, 명제 C'):

```
LRH 관점:
  Steering lift ΔT1 ∝ |c_i|  (해당 방향이 weight에 분리된 정도)
  ⇒ ΔT1@base < ΔT1@T2 < ΔT1@T4 < ΔT1@(T2+T4)
  
이유: T2/T4 학습이 representation separability를 강화
      → 동일 α로 steering 가산해도 lift 폭 ↑
      → Phase 2c/2d (T1@T4, T1@(T2+T4))가 main contribution 후보
```

**선행 연구 (composition)**:
- Persona Vector Distillation (Anthropic 2025-12): RLHF + steering 부분 시도, persona 영역
- Task Vectors (Ilharco et al. 2023): fine-tuned − base = task vector. 우리는 *관계별 분리 vector*
- T2+T4+T1 triple composition with multi-step ontology: **prior 없음**, 우리가 처음

### 3.4 수학적 프레임워크: 개입 공간의 분류

#### 3.4.1 프롬프트-동치 정리와 한계

```
정리 (부분 동치):
  프롬프트 P를 LLM에 입력하면
    C(P) = { K^(l,h)(P), V^(l,h)(P) }_{l,h}  ← KV cache 집합
  이 생성된다.
  이 캐시를 그대로 주입하면 P를 실제로 넣은 것과 100% 동치.

한계:
  T1 (Additive h' = h + αv) 과 A6 (Rotation h' = R(θ)h) 은
  Q-side 개입 — 생성 토큰의 쿼리 표현을 수정한다.
  프롬프트가 attention의 K/V를 결정하는 것과는 다른 경로.
  완전한 100% 동치를 달성하려면 Q-side + KV-side 모두 제어 필요.

압축 비율:
  n 토큰 프롬프트 → n×L×(2×d_head×H) 차원의 KV
  steering vector → d_model 차원 1개
  → Steering은 KV cache의 극도로 압축된 표현
  → 선형 표현 가설(LRH)이 성립할 때만 손실 없는 압축 가능
```

#### 3.4.2 세 가지 개입 공간

```
  ┌─────────────────────────────────────────────────────────┐
  │           LLM Attention 메커니즘                         │
  │                                                          │
  │  입력 토큰  →  [임베딩]  →  hidden state h_t             │
  │                                  │                       │
  │                         ┌────────┴────────┐             │
  │                         ▼                 ▼             │
  │    T1/A6 개입 →  [Q 프로젝션]    [K 프로젝션] [V 프로젝션]│
  │                         │                 │         │    │
  │                     Q = Wq·h'     K = Wk·h   V = Wv·h  │
  │                         │                 │         │    │
  │                         └────────┬────────┘         │    │
  │                                  ▼                  │    │
  │                      Attention(Q, K+ΔK, V+ΔV)      │    │
  │                                  ↑                  ↑    │
  │                           A8 (KV Cache Steering 주입)    │
  └─────────────────────────────────────────────────────────┘

  T1 (Additive):     h' = h + αv               ← Q-side
  A6 (Rotation):     h' = R(θ)h                ← Q-side
  A8 (KV Steering):  K' = K + c^k·S^k,         ← KV-side
                     V' = V + c^v·S^v

  세 방법은 서로 직교(orthogonal)하며 조합 가능.
  T1/A6 = 쿼리가 무엇에 주목하는가를 제어
  A8     = 무엇에 주목당하는가(컨텍스트 표현)를 제어
```

#### 3.4.3 KV Cache Steering (A8) 이론적 위치

```
출처: "KV Cache Steering for Controlling Frozen LLMs"
      arXiv 2507.08799 (Belitsky et al. 2025)

핵심 메커니즘:
  1. 대조쌍 (p⁺: CoT 포함, p⁻: 답만) 에서 스티어링 벡터 추출
     S^k_l = (1/N) Σ [f_l(p⁺) − f_l(p⁻)]    ← gradient 불필요
  2. Prefill 직후 KV cache에 단 1회 가산 주입
     K'_l = K_l + c^k · S^k_l
     V'_l = V_l + c^v · S^v_l

Activation Steering 대비 핵심 장점:
  - 증폭(Amplification) 없음:
      Activation steering은 매 생성 스텝마다 적용 → 레이어 방향+
      토큰 방향으로 복리 누적 → oversteering, 불안정
      KV steering은 prefill 후 1회 고정 → 누적 없음

실험 결과:
  Llama-3.1-70B: GPQA Diamond +4.6%, MATH +7.4%
  Latency overhead: ≈0 (10 ms/tok, activation steering 15 ms/tok)

우리 연구에의 적용:
  - generate_contrast_pairs_v3.py의 (p⁺, p⁻) 쌍이
    A8용 스티어링 벡터 추출의 직접 재료
  - 42종 관계 각각에 대해 S^k_l, S^v_l 추출 가능
  - Phase 0 probing 후 A8 추가 실험으로 자연스럽게 연결
```

### 3.5 PCLI — Probing-Calibrated Layerwise Intervention

#### 3.5.1 설계 원리

Trade-off(안정성 ↔ 효과)가 대립이 아닌 이유:

```
Linear Representation Hypothesis:
  τ(probing acc) → 1.0:  관계가 선형 분리 가능 → 벡터 v 잘 정의됨
                           → 작은 α로도 충분히 효과적
  τ → 0.5:               분리 불가 → v가 노이즈
                           → 큰 α 필요 → 필연적으로 불안정

따라서: amplification_risk ∝ 1/τ
τ가 높은 관계에서는 안정성과 효과가 함께 최대화된다.
Phase 0 screening (τ ≥ 0.70)이 이 원리의 구체적 실현.
```

세 가지 증폭 경로와 차단 방법:

```
  경로                      발생 조건               차단 방법
  ──────────────────────────────────────────────────────────
  수직 (레이어 관통)        residual h 개입          peak 레이어 1회만 개입
  수평 (시간축 누적)        K, V 캐시 오염           Q-only (K,V 미수정)
  복합 (수직+수평)          전 레이어 매 스텝 개입    A8 (prefill 1회 고정)
```

#### 3.5.2 계수 자동 교정 수식

```
α_r = BASE_ALPHA × tau_factor(τ_r) / settling(L_peak_r)

  tau_factor(τ)   = max(0.10, (1-τ) / (1-τ_skip))
                    τ=0.70 → 1.0   (기준)
                    τ=0.85 → 0.50  (강한 인코딩 = 절반)
                    τ=1.00 → 0     (완전 분리 = α≈0)

  settling(L)     = max(0.10, (N_layers-1-L) / (N_layers-1))
                    L=02  → 0.929  (26개 settling layer)
                    L=18  → 0.357  (10개)
                    L=28  → 0.036  (0개에 가까움)

예시 (BASE_ALPHA=0.30):
  ORDERING  τ=0.877 L=02 → α = 0.30 × 0.41 / 0.93 = 0.13  (작고 안정)
  FIRST_TOOL τ=0.836 L=18 → α = 0.30 × 0.55 / 0.36 = 0.46  (중간)
```

#### 3.5.3 방법 선택 결정 트리

```
τ < 0.70?
  └─ YES → skip (노이즈 개입 방지)
  └─ NO  ──→ pattern?

              flat (std < 0.03)?
                └─ YES → A8 (KV Cache Steering, 증폭 없음)

              early_peak (L ≤ 5)?
                └─ directional → A6_peak @L_peak  ★ 최적 구역
                └─ otherwise   → T1_qonly @L_peak

              mid_late (L ≥ 10)?
                └─ directional + τ≥0.85 → T1_qonly @L_peak
                └─ directional + τ<0.85 → A8 (안정성 우선)
                └─ symmetric/categorical + τ≥0.85 → T1_qonly @L_peak
                └─ symmetric/categorical + τ<0.85 → A8
```

★ **early_peak + directional + τ≥0.85** = 안정성·효과 모두 최대
- 작은 α 로 충분 (τ 높음)
- 개입 후 ~26 레이어 자연 정착 (settling)
- 수직 증폭이 있어도 모델 natural dynamics가 흡수

#### 3.5.4 개입 레이어 분리 원칙

```
각 관계가 서로 다른 L_peak를 사용하면 개입이 물리적으로 독립:

  관계 A (ORDERING)  → 개입 @L02
  관계 B (FIRST_TOOL)→ 개입 @L18
  → 겹치지 않음 → 상호 간섭 없음

  같은 레이어에 복수 관계가 배정되면 충돌 위험:
    h'_L = h + α₁v₁ + α₂v₂
    → 두 벡터 합 방향 보장 없음
    → check_results_v3.py의 충돌 분석으로 사전 탐지
```

#### 3.5.5 자동화 파이프라인

```
Phase 0 v3 probing 완료
        │
        ▼
check_results_v3.py 실행
        │
        ├─ 기존: 관계별 acc/auc/go-nogo 테이블
        ├─ 신규: curve pattern 분류 (early_peak/flat/mid_late)
        ├─ 신규: α 자동 교정
        ├─ 신규: 방법 선택 (A6_peak/T1_qonly/A8/skip)
        ├─ 신규: 레이어 충돌 분석
        └─ 신규: intervention_map.json 저장
                 │
                 ▼
        Phase 2 구현에서 직접 로드
        → 개입 스케줄 자동 구성 (for rel, spec in imap.items())
```

**intervention_map.json 구조:**
```json
{
  "metadata": { "pcli_version": "1.0", "base_alpha": 0.30, ... },
  "intervention_map": {
    "precedes": {
      "geometry": "directional", "tau": 0.877, "best_layer": 2,
      "pattern": "early_peak",  "method": "A6_peak",
      "alpha": 0.132,           "go_nogo": true
    },
    "mutex": {
      "pattern": "flat", "method": "A8", "alpha": null, ...
    },
    ...
  }
}
```

---

## 4. 실험 조건 (Conditions)

### 4.1 베이스라인 (비교군)

| 조건명 | 설명 | 카테고리 |
|---|---|---|
| **B0: Vanilla LLM** | 온톨로지 정보 없음, 순수 LLM | 하한선 |
| **B1: ReAct** | Reasoning + Acting 교차, 온톨로지 없음 | 표준 에이전트 |
| **B2: Text Serialization** | 온톨로지 → 텍스트 → 프롬프트 주입 (GraphRAG 방식) | 현재 주류 |
| **B3: KnowAgent** | 행동 지식 텍스트 직렬화 + CoT (NAACL 2025) | 직접 비교 |
| **B4: Routine (fine-tuned)** | 구조화 스크립트 + fine-tuning (상한선 참조) | 상한선 |
| **B5: GAP (SFT+RL)** | 그래프 계획 + SFT+RL (상한선 참조) | 상한선 |

### 4.2 제안 방법 (Treatment)

**Notation 갱신** (v1.11): T1 steering은 *어느 모델에서 추출했는지*에 따라 다른 효과 → `T1@<model>` 표기. T4는 기존 LATS 의미 + 신규 RFT 의미로 분기.

| 조건명 | 설명 | 학습 | 검증 시점 |
|---|---|---|---|
| **T1@base** | 온톨로지 관계 → steering vector (base 모델에서 추출), residual에 가산. 완전 training-free | 없음 | Phase 2a |
| **T2: Cross-Attn (LoRA)** | 온톨로지 노드 → cross-attention 주입, LoRA만 학습 | LoRA SFT | Phase 3 |
| **T3 = T2 + T1@T2** | T2-tuned 모델에서 v_relation 재추출 + steering. Hybrid | LoRA SFT | Phase 2b |
| **T4-LATS: Cross-Attn + LATS** | T2 + LATS tree search (τ²-bench simulator reward 활용, training-free reward) | LoRA만 | Phase 4 path α |
| **T4-RFT: facet-aware RFT** | 온톨로지 verifier 기반 GRPO/rejection-SFT로 모델 weight 자체 업데이트 | RFT | Phase 4 path β |
| **T5 = T4-RFT + T1@T4** | RFT 후 모델에서 v_relation 재추출 + steering (사용자 thesis, 명제 C') | RFT | Phase 2c (조건부) |
| **T6 = T2 + T4-RFT + T1@(T2+T4)** | Triple: cross-attn LoRA + RFT + steering 합성. Pareto upper bound | LoRA→RFT | Phase 5 (조건부) |

**Steering vector 추출 모델 명시** (T1 variants):

| Variant | 추출 source | 합성 |
|---|---|---|
| `T1@base` | Qwen2.5-7B-Instruct 그대로 | Phase 2a 단독 |
| `T1@T2` | T2 LoRA SFT 후 모델 | T3 (Phase 2b) |
| `T1@T4` | T4-RFT 후 모델 | T5 (Phase 2c) |
| `T1@(T2+T4)` | T2 LoRA SFT → T4-RFT 후 모델 | T6 (Phase 5) |

각 variant는 contrast pair에서 별도 추출. 직접 사용 불가 (모델 분포가 다름).

### 4.3 Ablation 조건

| 조건명 | 목적 |
|---|---|
| **A1: T2 - precedes only** | 어떤 관계 유형이 가장 중요한가? |
| **A2: T2 - mutex only** | |
| **A3: T2 - all relations** | |
| **A4: T2 - random vectors** | 온톨로지 의미가 있어야 하는가, 구조만으로 충분한가? |
| **A5: T2 - different layers** | 어느 레이어 주입이 최적인가? |
| **A6: Per-head Rotation (SO(d_head))** | Additive residual 대비 per-head rotation이 효과적인가? (ORDERING 특화) |

| **A7: Method × Relation Cross** | T1 vs A6를 42종 관계 유형별로 분리 비교 → 기하학 예측 검증 |
| **A8: KV Cache Steering** | KV-side 개입(arXiv 2507.08799 방법)을 42종 관계에 적용 → T1/A6(Q-side)와 비교 |

#### A7 상세 — T1 vs A6 실증 비교 설계 (42종 관계)

```
2 × N Factorial (방법 × 관계 유형):

  방법:    {T1 (Additive),   A6 (Per-head Rotation)}
  관계:    42종 전체 (§3.2.2 Group A–I)
           Phase 0 v3 probing 결과에서 best_acc ≥ 0.70인 관계 우선

측정:
  - Constraint Violation Rate (CVR) — 관계 유형별
  - pass^1 기여도 분해 (ablation per relation)
  - Layer probing accuracy (개입 전/후 비교)

이론적 예측 (검증 대상, 기하학적 분류에 따라):
  ┌──────────────────────────────────────────────────────────┐
  │ 관계 그룹         예측          근거                      │
  │ ─────────────────────────────────────────────────────── │
  │ [방향성]                                                  │
  │ precedes          A6 > T1       antisymmetric            │
  │ requires/enables  A6 ≥ T1       causal direction         │
  │ parameter_feeds   A6 ≥ T1       data flow dir.           │
  │ validates         A6 ≥ T1       verification dir         │
  │ retry_after_fail  A6 ≥ T1       fail→fix→retry           │
  │ fan_out/pruned_by A6 ≥ T1       GoT graph direction      │
  │ backtrack_to      A6 > T1       dead_end→restore dir.    │
  │ decomposes_into   A6 ≥ T1       goal→tools direction     │
  │ plan_step_precedes A6 > T1      abstract step ordering   │
  │ step_realizes_tool A6 ≥ T1      abstract→concrete dir.   │
  │ ─────────────────────────────────────────────────────── │
  │ [대칭/범주형]                                             │
  │ mutex             T1 ≥ A6       symmetric                │
  │ workflow_role     T1 > A6       unary, no dir.           │
  │ parallel_safe     T1 ≥ A6       symmetric                │
  │ conditional_on    T1 ≥ A6       cond. magnitude          │
  │ observation_triggers T1 ≥ A6    categorical mapping      │
  │ guardrail         T1 > A6       prohibition flag         │
  │ subtask_of        T1 ≥ A6       membership               │
  │ achieves_goal     T1 ≥ A6       categorical              │
  │ plan_step_skill   T1 > A6       type label               │
  └──────────────────────────────────────────────────────────┘

성공 기준: ≥2/3 관계에서 예측 방향 일치 (방향성 그룹, 대칭/범주형 그룹 각각)
  → 성공 시: Method Router (관계 유형 → 방법 자동 선택) 논문 기여로 추가
  → 실패 시: 단일 방법 선택 + ablation으로 기술
```

#### A8 상세 — KV Cache Steering (Q-side vs KV-side 비교)

```
배경:
  "KV Cache Steering for Controlling Frozen LLMs"
  arXiv 2507.08799 (Belitsky et al. 2025)

  T1/A6: Q-side 개입 (hidden state 수정 → 쿼리 방향 변경)
  A8:    KV-side 개입 (K,V cache 직접 수정 → 주목당하는 컨텍스트 변경)

설계:
  1. Phase 0 v3에서 생성한 contrast_pairs_v3.json 재사용
     (p⁺, p⁻) 쌍 → Mean-of-Differences → S^k_l, S^v_l 추출
     gradient 없음, 순수 분석적 계산

  2. 적용:
     K'_l = K_l + c^k · S^k_l    (모든 레이어)
     V'_l = V_l + c^v · S^v_l
     c^k ∈ {0.05, 0.1, 0.2, 0.4}, c^v ∈ {0.5, 1.0, 3.0, 5.0}

  3. 측정:
     - pass^1 on τ²-bench telecom (T1/A6와 동일 조건)
     - Amplification 분석: 레이어별 activation 분포 변화
     - 계수 민감도 곡선 (hyperparameter robustness)

비교 프레임:
  ┌──────────────────────────────────────────────────────┐
  │          T1       A6        A8        비고            │
  │ 개입위치  Q-side  Q-side    KV-side                   │
  │ 연산방식  가산    회전      가산                       │
  │ 증폭위험  있음    있음      없음 (1회 고정)            │
  │ 추가학습  없음    없음      없음                       │
  │ 조합가능  T1+A8  A6+A8    T1+A6+A8                   │
  └──────────────────────────────────────────────────────┘

시점: Phase 2 완료 후 추가 실험 (Phase 0 인프라 재사용으로 비용 최소)
```

#### A6 상세 — Per-head Rotation 설계

```
이론적 근거 (Phase 0 v2 + Lie Group 분석):
  - ORDERING 정보가 L02-L05에 집중 → 초기 attention head가 순서 geometry 내재
  - Additive (T1): h' = h + αv
      → 노름 변화 + 방향 편향 → FIRST_TOOL/NEXT_TOOL (크기 있는 shift)에 적합
  - Rotation (A6): h' = R(θ_rel) h,  R ∈ SO(d_head)
      → 노름 보존, cosine similarity geometry만 변경
      → ORDERING (A before B vs B before A, 방향만 다름)에 적합

구현:
  def apply_rotation_hook(hidden, layer_idx, head_idx, R_ontology):
      d_head = hidden.shape[-1] // n_heads
      h_head = hidden[:, :, head_idx*d_head:(head_idx+1)*d_head]
      # R_ontology ∈ SO(d_head): relation별 다른 rotation matrix
      rotated = torch.einsum('bsd,dd->bsd', h_head, R_ontology)
      hidden[:, :, head_idx*d_head:(head_idx+1)*d_head] = rotated
      return hidden

SEKA rotation 재활용 전략:
  1. SEKA base rotation (R_seka) → warm-start initialization
  2. SEKA 효과적 (layer, head) 쌍 ∩ Phase 0 ORDERING peak (L02-L05) → 탐색 공간 제한
  3. relation별 small Givens rotation 조정 → 학습 없이 구성 가능
```

---

## 5. 벤치마크 전체 목록 및 우선순위

### 5.0 선정 기준

```
필수 조건:
  ✅ 다단계 도구 호출 (single-turn 제외)
  ✅ 실행 기반 평가 (텍스트 매칭만 하는 벤치마크 낮은 우선순위)
  ✅ 도구 간 의존성/순서 제약이 있어야 함
  ✅ 공개 데이터셋 (재현 가능)

우선순위 가중 요소:
  + 기업/서비스 도메인 (우리 타겟 도메인)
  + 구조 주입 효과를 직접 측정 가능
  + 2024-2025 최신 (심사 시점에 신선도)
  + 이미 설치/접근 가능
  - 학습 데이터 필요 (우리는 training-free)
  - 도구 의존성 구조가 없음 (우리 방법의 강점 측정 불가)
```

---

### 5.1 Priority 1 — ★★★★★ (필수 주 벤치마크)

#### B-01. τ²-bench

```
출처:       Yao et al. 2024 (tau2-bench)
도메인:     Retail (N=114) / Telecom (N=256) / Airline (N=50)
태스크:     고객 서비스 에이전트: 유저 멀티턴 대화 + DB 도구 호출
평가:       pass^1, pass^5 — DB 최종 상태 해시 비교 (실행 기반)
도구 수:    도메인별 15-30개, 의존성 암묵적
특징:       실제 DB simulator 내장, 멀티도메인, 정책 준수 평가 포함
이미 설치:  ✅ ~/workspace_common/boltzmann-attention/external/tau2-bench
D0 결과:    ✅ Telecom S1 (facet_full +18%p), Retail S2, Airline 소규모
활용:       주 정확도 검증 (H_main, H2), 도메인 일반화 실험

선정 이유:
  - 유일하게 설치+결과 존재 → 즉시 비교 가능
  - pass^k = 순서 오류까지 잡는 엄밀한 메트릭
  - 멀티도메인 → cross-domain 일반화 실험 자연스럽게 내장
```

#### B-02. TPS-Bench

```
출처:       arxiv 2511.01527 (2025)
도메인:     웹검색, 지도, 캘린더, 날씨 등 MCP 도구 수백 개
태스크:     병렬/직렬 의존성을 가진 복합 태스크
난이도:     Easy (≤5 서브태스크) / Hard (≤50 서브태스크)
평가:       Task Completion Rate + Execution Time
현재 SOTA:  GLM-4.5 64.72%, GPT-4o 45.08%
특징:       도구 스케줄링(병렬화) 자체가 평가 대상
활용:       도구 의존성 그래프 주입 효과 검증 (H3), 병렬화 효율

선정 이유:
  - precedes/enables 관계가 직접 성능에 영향
  - Hard 난이도 = 개선 여지 크고 우리 방법이 두드러짐
  - 2025 최신 → 심사 시점 신선도
```

#### B-03. FlowBench

```
출처:       EMNLP 2024 (arxiv 2406.14884)
도메인:     6개 도메인, 22개 역할, 51개 시나리오
태스크:     워크플로우 가이드 계획 수립
평가:       단계별 정확도 + 최종 완료율
워크플로우 포맷: 텍스트 / 코드 / 플로우차트 모두 포함
현재 SOTA:  GPT-4o도 "만족스럽지 못한 결과"
활용:       텍스트 직렬화 vs 구조 주입 직접 비교 (H1)

선정 이유:
  - 워크플로우 포맷 다양성 = 우리의 구조 주입 효과를 고립 측정 가능
  - 기존 방법 성능이 낮음 → 개선 여지 충분
  - 텍스트 직렬화(B2)와 우리 방법(T2) 직접 비교에 최적
```

---

### 5.2 Priority 2 — ★★★★☆ (중요 보조 벤치마크)

#### B-04. AgentArch (ServiceNow)

```
출처:       arxiv 2509.10769, ServiceNow Research (2025)
도메인:     기업 워크플로우 (Time-off 관리, Customer Routing)
태스크:     실제 ServiceNow 환경 내 에이전트 아키텍처 비교
평가:       태스크 완료율
현재 결과:  GPT-4.1 70.8% (단순) / 35.3% (복잡)
특징:       18가지 에이전트 아키텍처 비교, GitHub 공개
활용:       기업 환경 현실성 검증, 아키텍처 비교

선정 이유:
  - 기업 워크플로우 = 우리 온톨로지 타겟 도메인과 직접 대응
  - 복잡 태스크 35.3% → 개선 여지 큼
  - 오픈소스 코드 → 우리 방법 통합 가능
```

#### B-05. ToolPRMBench

```
출처:       arxiv 2601.12294 (2025)
태스크:     도구 사용 에이전트의 단계별(step-level) 평가
평가:       Process Reward Model 정확도, 각 스텝 올바름 여부
특징:       최초의 interactive agent 단계별 벤치마크
활용:       온톨로지 제약이 어느 단계에서 효과적인지 세분화 분석

선정 이유:
  - 단계별 분석 → "어느 스텝에서 precedes 위반이 발생하는가" 측정 가능
  - PRM 연구와 연계 가능 (향후 확장)
```

#### B-06. ToolComp

```
출처:       arxiv 2501.01290 (2025)
태스크:     485개 의존적 다중 도구 사용 태스크
평가:       최종 정답 + 중간 단계 프로세스 감독 (human-verified)
특징:       PRM이 outcome-only 대비 우수함 실증
활용:       의존성 있는 도구 사용 정확도 + 중간 단계 품질

선정 이유:
  - 도구 의존성이 핵심 (우리의 precedes/requires와 직접 대응)
  - 프로세스 감독 레이블 → 단계별 오류 분석 가능
```

#### B-07. GAP Benchmark (MHQA)

```
출처:       NeurIPS 2025 (arxiv 2510.25320)
도메인:     Multi-Hop Question Answering
태스크:     의존성 그래프 기반 병렬/직렬 도구 실행
평가:       정답 정확도 + 도구 실행 효율
현재 SOTA:  GAP (SFT+RL) >> ReAct
활용:       GAP과 직접 비교 (training-free vs SFT+RL)

선정 이유:
  - 동일 벤치마크에서 training-free 방법이 SFT+RL 대비 얼마나 차이나는지 측정
  - 상한선과의 gap 명확화
```

#### B-08. KnowAgent Benchmark (HotpotQA + ALFWorld)

```
출처:       NAACL 2025 (arxiv 2403.03101)
도메인:     HotpotQA (멀티홉 QA) + ALFWorld (가정 환경 계획)
평가:       EM, F1, 성공률
활용:       KnowAgent (텍스트 직렬화) vs 우리 방법(구조 주입) 직접 비교

선정 이유:
  - 가장 직접적인 비교 대상 (같은 문제, 다른 주입 방법)
  - NAACL 2025 발표 → 심사위원 인지도 높음
```

---

### 5.3 Priority 3 — ★★★☆☆ (분석 확장용)

#### B-09. AgentProcessBench

```
출처:       arxiv 2603.14465 (2025)
태스크:     도구 사용 에이전트 스텝별 프로세스 품질 진단
평가:       단계 수준 오류 분류 및 진단
활용:       온톨로지 위반 오류 유형 분류 (오류 분석 섹션)
```

#### B-10. ToolBench 2.0

```
출처:       2025
도메인:     장거리 멀티스텝 태스크
평가:       장기 목표 달성 메트릭
활용:       long-horizon 계획에서 구조 주입 지속 효과
```

#### B-11. StableToolBench

```
출처:       THUNLP, GitHub 공개
도메인:     16,000+ RapidAPI
평가:       Pass Rate, Win Rate (GPT-4 judge)
특징:       Virtual API server로 안정성 해결
활용:       대규모 도구 풀에서 검색 정확도 비교

주의:       도구 수 너무 많아 의존성 구조 측정 어려움
```

#### B-12. ASTRA-bench

```
출처:       arxiv 2603.01357 (2025)
태스크:     개인화된 사용자 컨텍스트 기반 도구 계획
평가:       추론 정확도 + 행동 계획 품질
활용:       사용자 컨텍스트 의존 계획 (온톨로지 확장 방향)
```

#### B-13. Blocksworld with MCP

```
출처:       arxiv 2512.03955 (2025)
태스크:     고전 AI 계획 문제 (Blocksworld) + MCP 도구
평가:       계획 정확도
특징:       precedes 관계가 완전히 명시적 → ablation에 적합
활용:       순수 순서 제약 환경에서 방법 검증 (단위 테스트)
```

#### B-14. Ego-Graph Tool Retrieval Benchmark

```
출처:       IJCNLP 2025 (arxiv 2508.05888)
태스크:     기업 태스크 계획을 위한 도구 검색
평가:       CompleteRecall (도구 포함율)
현재 SOTA:  91.85% (Ego-Graph) vs 89.26% (비KG 베이스라인)
활용:       도구 검색 단계에서 온톨로지의 기여도 측정
```

---

### 5.4 Priority 4 — ★★☆☆☆ (상한선/범용 참고용)

| 벤치마크 | 출처 | 평가 방식 | 우리 연구 역할 | 주의사항 |
|---|---|---|---|---|
| **AppWorld** | Trivedi et al. 2024 | API 실행 완료율 | 멀티앱 의존성 참고 | 코드 실행 중심 |
| **WorkArena++** | ServiceNow 2024 | 웹 태스크 완료율 | 기업 웹 워크플로우 | BrowserGym 환경 필요 |
| **WebArena** | 다수 2023 | 웹 태스크 완료율 | 웹 에이전트 상한선 | 웹 특화 |
| **OSWorld** | 2024 | GUI 조작 완료율 | 컴퓨터 사용 상한선 | GUI 중심, 도구 호출 아님 |
| **GAIA** | 2023-2024 | 정확도 | 범용 어시스턴트 상한선 | 멀티모달 포함 |
| **ALFWorld** | 2021 | 성공률 | KnowAgent 비교 환경 | 가정 환경, 오래된 벤치마크 |
| **SWE-bench** | 2024 | 코드 패치 정확도 | 코딩 에이전트 참고 | 코딩 특화, 관련성 낮음 |
| **MCP-Bench** | arxiv 2508.20453 | 완료율 | MCP 도구 사용 참고 | 2025 최신 |
| **MCPVerse** | arxiv 2508.16260 | 완료율 | 대규모 실세계 도구 | 1M 컨텍스트 환경 |

---

### 5.5 벤치마크 우선순위 요약표

| 순위 | 벤치마크 | 별점 | 논문 역할 | 구현 난이도 | 비고 |
|---|---|---|---|---|---|
| 1 | **τ²-bench** | ★★★★★ | 주 결과 | 낮음 (설치됨) | D0 결과 있음 |
| 2 | **TPS-Bench** | ★★★★★ | 주 결과 (의존성) | 중간 | 2025 최신 |
| 3 | **FlowBench** | ★★★★★ | 직접 비교 (포맷 효과) | 중간 | EMNLP 2024 |
| 4 | **AgentArch** | ★★★★☆ | 기업 현실성 | 중간 | GitHub 공개 |
| 5 | **ToolPRMBench** | ★★★★☆ | 단계별 분석 | 중간 | 2025 |
| 6 | **ToolComp** | ★★★★☆ | 의존성 도구 사용 | 중간 | PRM 연계 |
| 7 | **GAP Benchmark** | ★★★★☆ | 상한선 비교 | 높음 (재현) | NeurIPS 2025 |
| 8 | **KnowAgent BM** | ★★★★☆ | 직접 비교 | 낮음 | NAACL 2025 |
| 9 | **AgentProcessBench** | ★★★☆☆ | 오류 분석 | 중간 | 2025 |
| 10 | **ToolBench 2.0** | ★★★☆☆ | 장거리 계획 | 중간 | |
| 11 | **StableToolBench** | ★★★☆☆ | 대규모 도구 검색 | 낮음 | GitHub 공개 |
| 12 | **ASTRA-bench** | ★★★☆☆ | 컨텍스트 계획 | 중간 | 2025 |
| 13 | **Blocksworld MCP** | ★★★☆☆ | 순서 제약 단위 테스트 | 낮음 | |
| 14 | **Ego-Graph BM** | ★★★☆☆ | 도구 검색 | 중간 | IJCNLP 2025 |
| 15 | **AppWorld** | ★★☆☆☆ | 멀티앱 참고 | 높음 | |
| 16 | **WorkArena++** | ★★☆☆☆ | 기업 웹 참고 | 높음 | |
| 17 | **WebArena** | ★★☆☆☆ | 웹 상한선 | 높음 | |
| 18 | **OSWorld** | ★★☆☆☆ | 컴퓨터 사용 참고 | 높음 | GUI 환경 |
| 19 | **GAIA** | ★★☆☆☆ | 범용 상한선 | 중간 | |
| 20 | **MCP-Bench** | ★★☆☆☆ | MCP 참고 | 중간 | 2025 |
| 21 | **ALFWorld** | ★★☆☆☆ | KnowAgent 환경 | 낮음 | 구형 |
| 22 | **MCPVerse** | ★★☆☆☆ | 대규모 참고 | 높음 | 1M 컨텍스트 |
| 23 | **SWE-bench** | ★☆☆☆☆ | 코딩 참고만 | 중간 | 관련성 낮음 |

---

### 5.6 논문 단계별 벤치마크 운용 계획

```
논문 제출 최소 요건 (Phase 3 완료 시):
  ✅ τ²-bench (3 domains)
  ✅ FlowBench
  ✅ KnowAgent Benchmark (HotpotQA + ALFWorld)
  → 주장 1 (cross-attention > 텍스트 직렬화) 검증 가능

논문 강화 요건 (Phase 4 완료 시):
  ✅ 위 + TPS-Bench (Easy + Hard)
  ✅ AgentArch
  → 주장 2 (병렬 의존성), 주장 3 (기업 현실성) 추가

논문 완성 요건 (Phase 5 완료 시):
  ✅ 위 + GAP Benchmark (도메인 일반화)
  ✅ ToolPRMBench (단계별 분석)
  → Appendix: 오류 유형 분석, cross-domain 일반화
```

---

### 5.7 평가 지표 전체

| 지표 | 계산 방법 | 적용 벤치마크 | 측정 대상 |
|---|---|---|---|
| **pass^1** | 1회 시도 DB 상태 해시 일치율 | τ²-bench | 계획 정확도 (주) |
| **pass^5** | 5회 중 1회 이상 성공률 | τ²-bench | 계획 잠재력 |
| **Task Completion Rate** | 태스크 완료 수 / 전체 | TPS, AgentArch, FlowBench | 태스크 완료 |
| **Execution Efficiency** | 최적 스텝 수 / 실제 스텝 수 | TPS-Bench | 병렬화 효율 |
| **Constraint Violation Rate** | 온톨로지 제약 위반 / 전체 스텝 | τ²-bench, TPS | 제약 준수도 |
| **Step Accuracy** | 단계별 정답 스텝 비율 | ToolPRMBench, FlowBench | 중간 과정 품질 |
| **CompleteRecall** | 정답 도구 포함율 | Ego-Graph BM, τ²-bench | 도구 검색 |
| **Cross-Domain Drop** | (학습 도메인 pass^1) - (새 도메인 pass^1) | τ²-bench | 일반화 |
| **Token Cost** | 입력 토큰 수 | 전체 | 효율성 |
| **Latency** | 추론 시간 (ms/task) | 전체 | 실용성 |
| **EM / F1** | 정확 일치 / 부분 일치 | HotpotQA (KnowAgent) | QA 정확도 |

#### 5.7.1 Enterprise Viability Metrics (★ v1.19 신규)

Pass^1 단독으로는 *모델 실용성* 평가 부족. Phase 1 v2 실측에서 **모델별 *실패 모드*가 질적으로 다름**:

| 지표 | 정의 | 측정 의미 |
|---|---|---|
| **ATR (Average Time to Resolution)** | 성공 sim의 avg wallclock duration | 실용 latency (낮을수록 ↑) |
| **ATR p95** | 95th percentile (꼬리) | worst-case latency |
| **max_steps_rate** | max_steps 종료 비율 | **과진단 trap** (capability 충분하나 종료 못함) |
| **quick_failure_rate** | <10 msg + reward=0 비율 | **즉시 포기형** (capability 부족) |
| **over_diagnosis_rate** | >100 msg + reward=0 비율 | 오래 시도 후 실패 |
| **productive_resolution_rate** | <30 msg + reward=1 비율 | 빠르고 정확한 해결 |
| **tool_calls_per_success** | 성공 sim의 avg tool 호출 수 | 도구 사용 효율 |
| **infra_error_rate** | infrastructure_error + too_many_errors | 안정성 |
| **EVS (Enterprise Viability Score)** | (pass^1 + 0.5·productive_rate) / (1 + max_steps_rate + 0.5·over_diag_rate + infra_rate) | **종합 실용 지표** |

#### 5.7.2 실측 (Phase 1 v2 telecom base)

| Setup | n | pass^1 | ATR(s) | max_S% | quick% | over_diag% | prod% | infra% | **EVS** |
|---|---|---|---|---|---|---|---|---|---|
| **Hermes-3 conc=3** | 42 | 0.119 | 63 | 19.0% | 2.4% | 23.8% | 7.1% | 2.4% | **0.116** ★ |
| Hermes-3 conc=4 | 23 | 0.087 | 18 | 21.7% | 0% | 21.7% | 8.7% | 13.0% | 0.090 |
| Qwen v1 B0 (16K) | 456 | 0.044 | 70 | 33.6% | 7.7% | 34.2% | 0.7% | 10.7% | 0.029 |
| Qwen v2 B1 (ReAct) | 456 | 0.035 | 90 | 56.6% | 0% | 57.5% | 0.7% | 1.5% | 0.021 |
| Qwen v2 B0 (32K) | 456 | 0.029 | 76 | 52.4% | 0% | 53.9% | 0.9% | 1.5% | 0.018 |
| Llama+chat-template | 456 | 0.002 | 27 | 0% | **91.7%** | 0% | 0.2% | 27.9% | 0.003 |

#### 5.7.3 핵심 패턴 발견 — 모델별 *실패 모드 분기*

```
실패 모드 trichotomy:

  Quick-failure (Llama-3.1-8B 91.7%)
    └ 즉시 transfer_to_human → enterprise 불가능
    └ 원인: native FC tuning 약함 (BFCL 70%)
    
  Over-diagnosis (Qwen2.5-7B 54-58%, ReAct 더 심각)
    └ 200 step trap: capability 충분하나 *종료 인지* 부족
    └ 원인: native FC OK but planning/termination 약함
    
  Balanced (Hermes-3 19% max_S, 7% prod)
    └ 진단 시도 + 짧게 끝낼 때 끝냄
    └ 원인: explicit FC fine-tune + 적절한 termination
```

#### 5.7.4 함의 — 우리 ontology 개입의 *부가 contribution*

```
Pass^1 lift 외에:

  Phase 2 (T1/A6/A8 steering) 측정 시 EVS도 보고:
    - achieves_goal, plan_committed_to_goal steering이 *종료 신호 강화* 가능
    - max_steps_rate 감소 = over-diagnosis trap 해소
    - productive_rate 증가 = enterprise 사용 가능

  Phase 4 (RFT) 측정:
    - facet-aware reward에 *종료 보너스* 포함 (turn 수에 inverse weight)
    - 결과 모델은 *짧고 정확한 해결* 학습
    
우리 main contribution은:
  (1) pass^1 lift (정확성)
  (2) EVS lift (실용성) ← v1.19 신규 주장
  
Hermes-3 baseline이 *이미 EVS 0.116* → 우리 개입이 *그 위에서 추가 lift*  
하면 enterprise-grade agent 영역 진입 가능.
```

---

## 6. 실험 모델 전체 목록 및 우선순위

### 6.0 모델 선정 원칙

```
필수 조건:
  ✅ 오픈소스 (내부 activation 접근 가능)
     → Vector Steering / Cross-Attention 주입에 필수
  ✅ Instruction-following 능력 (계획 수립 태스크)
  ✅ Function Calling 지원 (도구 호출 형식)

우선순위 가중 요소:
  + 이전 실험 사용 모델 (D0, Channel A 결과 연속성)
  + 7B 계열 (GPU 메모리 효율, 반복 실험 가능)
  + Function Calling 특화 학습 여부
  + Reasoning 능력 (Steering 연구와 연계)
  - API 전용 모델 (activation 접근 불가) → 상한선 참조만
```

---

### 6.1 Tier 1 — 핵심 실험 모델 (모든 Phase 적용)

| 모델 ID | 파라미터 | 이전 실험 | Function Calling | 선정 이유 |
|---|---|---|---|---|
| **Qwen/Qwen2.5-7B-Instruct** | 7B | ✅ D0 완료 | ✅ 특화 | D0 baseline 존재, 내부 구조 파악됨 |
| **meta-llama/Llama-3.1-8B-Instruct** | 8B | △ Channel A | ✅ 지원 | 학습 포맷 대조군 (FC 특화 안됨) |

```
Qwen2.5-7B 선정 근거:
  - D0 실험 (retail/telecom/airline) 완료 → 직접 비교 가능
  - Function Calling 특화 학습 → FC 학습 편향 확인됨 (이전 실험)
  - 7B = GPU 단일 카드에서 steering 실험 반복 가능
  - 한국어/기업 도메인 강점

LLaMA-3.1-8B 선정 근거:
  - FC 특화 학습 없음 → Qwen과 학습 포맷 편향 대조 가능
  - Channel A 실험에서 이미 비교군으로 사용
  - 오픈소스 표준 모델, 재현성 높음
```

---

### 6.2 Tier 2 — 스케일 효과 및 추론 모델 (Phase 3-4)

| 모델 ID | 파라미터 | 카테고리 | 실험 목적 |
|---|---|---|---|
| **Qwen/Qwen2.5-14B-Instruct** | 14B | 스케일업 | 7B → 14B 스케일에서 주입 효과 변화 |
| **Qwen/Qwen2.5-7B-Instruct-FC** | 7B | FC 특화 | Function Calling 특화 버전 비교 |
| **deepseek-ai/DeepSeek-R1-Distill-Qwen-7B** | 7B | 추론 특화 | Reasoning model에서 steering 효과 (ICLR 2025 workshop 연구 참조) |
| **deepseek-ai/DeepSeek-R1-Distill-Llama-8B** | 8B | 추론 특화 | DeepSeek-R1 계열 LLaMA 베이스 |
| **mistralai/Mistral-7B-Instruct-v0.3** | 7B | 비교군 | 유럽계 모델, 다른 학습 분포 |

```
DeepSeek-R1-Distill 선정 근거:
  - ICLR 2025 Workshop: R1-Distill 모델에서 steering vector로
    reasoning 행동(backtracking 등) 제어 가능 확인됨
  - 추론 특화 모델에서 온톨로지 steering이 더 효과적인지 비교
  - 7B-8B 계열 → GPU 메모리 제약 없음
```

---

### 6.3 Tier 3 — 대형 모델 (Phase 5, 상한선 탐색)

| 모델 ID | 파라미터 | 실험 목적 | 비고 |
|---|---|---|---|
| **Qwen/Qwen2.5-72B-Instruct** | 72B | 스케일 상한선 | multi-GPU 필요 |
| **meta-llama/Llama-3.1-70B-Instruct** | 70B | 대형 비교군 | multi-GPU 필요 |
| **meta-llama/Llama-3.3-70B-Instruct** | 70B | 최신 LLaMA | multi-GPU 필요 |
| **Qwen/Qwen2.5-32B-Instruct** | 32B | 중간 스케일 | single GPU (A100 80G) |

---

### 6.4 Tier 4 — API 모델 (상한선 참조 전용, activation 접근 불가)

| 모델 | 제공사 | 역할 | 비고 |
|---|---|---|---|
| **GPT-4o** | OpenAI | 상한선 | Routine 96% 달성 모델 |
| **Claude-3.5-Sonnet** | Anthropic | 상한선 | AgentArch 비교용 |
| **Gemini-1.5-Pro** | Google | 상한선 | 참고용 |

```
주의: API 모델은 activation 접근 불가
  → Steering / Cross-Attention 주입 실험 불가
  → "Routine/GAP fine-tuned 방법의 상한선" 참조에만 사용
  → 논문에서 "우리 training-free 방법 vs API 상한선" 비교표에 포함
```

---

### 6.5 모델별 실험 범위 매트릭스

```
                    Phase0  Phase1  Phase2  Phase3  Phase4  Phase5
                    (Probe) (Base)  (Steer) (X-Att) (Hybrid)(Gen)
─────────────────────────────────────────────────────────────────
Qwen2.5-7B          ✅      ✅      ✅      ✅      ✅      ✅   ← 주 모델
LLaMA-3.1-8B        ✅      ✅      ✅      ✅      △       △
Qwen2.5-14B         △       △       △      ✅      △       △   ← 스케일
DeepSeek-R1-Qwen7B  △       ✅      ✅      △       △       △   ← 추론
DeepSeek-R1-Llama8B △       ✅      △       △       △       △
Mistral-7B          ✗       ✅      △       △       ✗       ✗   ← 비교
Qwen2.5-32B         ✗       △       ✗      △       ✗       ✗   ← 스케일
Qwen2.5-72B         ✗       ✗       ✗      △       ✗       ✗   ← 상한선
GPT-4o (API)        ✗       ✅      ✗       ✗       ✗      ✗   ← 상한선

✅ 필수   △ 선택/여건 되면   ✗ 미포함
```

---

### 6.6 GPU 메모리 요구사항

| 모델 | 추론 VRAM | Steering 실험 | Cross-Attn LoRA | 비고 |
|---|---|---|---|---|
| Qwen2.5-7B | ~14GB | ~16GB | ~20GB | A100 40G 1장 가능 |
| LLaMA-3.1-8B | ~16GB | ~18GB | ~22GB | A100 40G 1장 가능 |
| DeepSeek-R1-7B | ~14GB | ~16GB | ~20GB | A100 40G 1장 가능 |
| Mistral-7B | ~14GB | ~16GB | ~20GB | A100 40G 1장 가능 |
| Qwen2.5-14B | ~28GB | ~32GB | ~40GB | A100 40G 1장 가능 |
| Qwen2.5-32B | ~64GB | ~72GB | ~80GB | A100 80G 1장 |
| Qwen2.5-72B | ~144GB | — | — | A100 80G 2장 |

```
현재 서버 환경 (woori@61.33.35.153):
  GPU: cuda:0 (모델 스펙 확인 필요)
  Tier 1-2 모델: 단일 GPU에서 실험 가능 (추정)
  Tier 3 대형 모델: 서버 스펙 확인 후 결정
```

---

### 6.7 이전 실험과의 연속성

```
D0 실험 결과 (Channel A, 프롬프트 포맷):
  모델: Qwen2.5-7B-Instruct
  결과: Telecom facet_full F1=0.361 >> nl_full=0.181 (S1)
        Retail facet_full ≈ nl_full (S2 중립)
        Airline facet_full > nl_full (+4%p, CI 중첩)

새 실험과의 연결:
  - Qwen2.5-7B D0 결과 = Channel A 진단 (Baseline B0/B2 참조)
  - LLaMA vs Qwen 비교 = FC 학습 편향 confound 분리
  - DeepSeek-R1 추가 = 추론 특화 모델에서 steering 효과 검증
  - 메트릭 변경: F1(도구 집합) → pass^1(DB 실행 결과)
```

---

## 7. 실험 단계별 계획

### Phase 0: 사전 검증 (진행 중)

```
목표: 42종 온톨로지 관계가 LLM activation 공간에 실제로 존재하는가?

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
v1 → v2 설계 수정 (confound 발견) ✅ 완료
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
v1 문제: "first A then B" vs "first B then A" 템플릿
  → 어순 자체가 label과 상관됨
  → L01에서 acc=100% (표면적 텍스트 패턴만 학습, 무의미)

v2 재설계: 시나리오 기반 묵시적 쌍 (세 유형)
  FIRST_TOOL: "[task] → 다음 호출: {correct_tool/wrong_tool}"
  NEXT_TOOL:  "agent called {prev}. Next: {correct/wrong}"
  ORDERING:   "[{tool1}]...[{tool2}] 동일 도구, 순서만 다름

  → 텍스트 표면 형식이 pos/neg 동일 (어순 편향 제거)
  → 1440 pairs (444 first + 800 next + 196 ordering)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
v2 결과 (Qwen2.5-7B, τ²-bench telecom, 임계값 0.70) ✅ PASS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FIRST_TOOL:  peak L18,  acc=0.836, AUC=?   ✓ PASS
NEXT_TOOL:   plateau L01-L28 ~0.88          ✓ PASS  (부분 confound 잔존 의심)
ORDERING:    peak L02,  acc=0.877, AUC=0.946  ✓ PASS
             └→ L28까지 하강 (acc=0.770 유지)

결과 위치:  reports/facet_rft_2026/phase0_probing/
  - probing_results_v2_telecom.json
  - probing_curve_v2_telecom.png

─── 핵심 발견 ───────────────────────────────────────
ORDERING 정보는 초기 레이어(L02-L05)에 집중되고
후기 레이어(L20-L28)에서 하강한다.
→ 초기 attention head가 tool ordering geometry를 내재함
→ 이 정보는 후기 레이어에서 추상화·소실됨
────────────────────────────────────────────────────

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
v3: 42종 관계 전체 probing ⏳ 실행 대기 중
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

코드:
  tau2_telecom_ontology.py     (v5, 42종 관계)
  generate_contrast_pairs_v3.py (42종 × 12~25 쌍 생성)
  probe_ontology_v3.py          (A7 예측 검증 포함)
  run_probe_v3.sh               (실행 스크립트)
  check_results_v3.py           (결과 확인)

실행 위치: /home/woori/workspace_common/boltzmann-attention-pi
예상 출력:
  reports/facet_rft_2026/phase0_probing/
    contrast_pairs_v3.json
    probing_results_v3_telecom.json
    probe_v3_run.log

v3 결과 확인 후 업데이트 예정:
  - 42종 관계별 best_layer, best_acc, best_auc, go_nogo_pass
  - A7 예측 검증: 방향성 관계(A6 후보) vs 대칭/범주형 관계(T1 후보) acc 비교
  - 개입 레이어 전략 refinement

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
개입 레이어 전략 (v2 기반 잠정 확정, v3 결과로 업데이트 예정)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  관계 유형              주입 레이어     주입 방법
  ──────────────────────────────────────────────────
  방향성 관계 (A6 후보)  L02-L05         A6 Per-head Rotation
  대칭/범주형 (T1 후보)  L01-L18         T1 Additive Steering
  KV-side (A8)           전 레이어       KV Cache 단일 주입

  이론적 근거:
    T1 (Additive): 노름 변화 허용 → "어떤 도구가 맞는가" (크기 shift)
    A6 (Rotation): 노름 보존 SO(d_head) → "A before B vs B before A" (방향만)
    A8 (KV):       증폭 없음 (1회 고정) → 조건부/범주형 제약에도 적용 가능

Go/No-Go: ✓ v2 OVERALL PASS → Phase 1 진행 가능
           v3 결과 대기 중 → 42종 전체 검증 후 Phase 2 조건 확정

산출물:
  ✅ v2: 레이어별 probing accuracy 곡선 (PNG)
  ✅ v2: 개입 레이어 잠정 확정 (L02-L05 / L01-L18)
  ✅ v2: A6 ablation 설계 완료 (§4.3 참조)
  ⏳ v3: 42종 관계 probing 결과 (probing_results_v3_telecom.json)
  ⏳ v3: A7 예측 검증 결과
  ⏳ v3: PCLI Intervention Map (intervention_map.json) ← Phase 2 설계도
         포함 항목: 관계별 (method, best_layer, alpha, pattern, go_nogo)
```

### Phase 1: Baseline 구현 및 측정 (2주)

```
목표: 비교 기준선 확립

실험:
  1. B0 (Vanilla LLM) on τ²-bench × 3 domains
  2. B1 (ReAct) on τ²-bench × 3 domains
  3. B2 (Text Serialization) on τ²-bench × 3 domains
     - 온톨로지 관계를 텍스트로 변환 → 시스템 프롬프트에 주입
  4. 모든 조건: Qwen2.5-7B + LLaMA-3.1-8B

측정:
  pass^1, pass^5, Constraint Violation Rate, Token Cost

산출물:
  - Baseline 성능 테이블
  - 현재 D0 결과(Channel A)와 비교
```

#### Phase 1 v1 B0 실측 결과 (2026-05-27, telecom base)

실행: 2026-05-27 02:15 ~ 07:42 KST (5h 27m), Qwen2.5-7B-Instruct (local vLLM port 8000, max-model-len 16384), τ²-bench telecom base N=114, **trials=4, max_steps=200, max_concurrency=8** (공식 leaderboard 프로토콜).
원본 데이터: `reports/facet_rft_2026/phase1_baseline/base_n114_v1_16k_partial/B0_telecom_base.json/results.json` (109 MB).
해석 보고서: `reports/facet_rft_2026/phase1_baseline/base_n114_v1_16k_partial/B0_analysis_report.md`.

**Headline**:

| 지표 | 값 | 95% CI |
|---|---|---|
| Total / evaluated | 456 / 421 | infra error 35 제외 |
| **pass^1 (avg reward)** | **0.0475** | [0.031, 0.072] (Wilson) |
| pass^2 (≥1 in 2) | 0.0702 | |
| pass^3 (≥1 in 3) | 0.0965 | |
| **pass^4 (best-of-4)** | **0.1316** | 15/114 tasks |
| pass-all (4/4) | **0** | 어떤 task도 4 trial 모두 통과 X |

**Termination 분포 — 가장 진단적인 신호**:

| Termination | n | 비중 | avg_reward | pass rate (≥1) |
|---|---|---|---|---|
| user_stop | 254 | 55.7% | 0.0787 | **7.87%** |
| **max_steps** | **153** | **33.6%** | **0.0000** | **0.00%** |
| infrastructure_error | 35 | 7.7% | 0.0 | 0.00% |
| too_many_errors | 14 | 3.1% | 0.0 | 0.00% |

→ **max_steps=200에 닿으면 100% reward=0**. `user_stop`까지 정상 종료된 254건만 통과 가능성 존재.

**Category 능력 프로파일** (Qwen2.5-7B vanilla 한계 노출):

| Category | n_sims | avg_reward | user_stop pass rate | 진단 |
|---|---|---|---|---|
| **service_issue** | 116 | **0.1379** | **19.8% (16/81)** | 가능성 있음 |
| mobile_data_issue | 144 | 0.0278 | 5.3% (4/76) | 거의 실패 |
| **mms_issue** | **196** | **0.0000** | **0.0% (0/97)** | **전체 실패** |

→ `mms_issue` 196 sims 전부, persona/trial 무관하게 reward=0. **Vanilla로는 풀 수 없는 영역**. Phase 2 steering 개입 lift 측정의 *clean baseline*.

**Persona 효과 (직관 반대)**:

| Persona | n_evaluated | avg_reward |
|---|---|---|
| **Hard** | 136 | **0.0735** |
| Easy | 140 | 0.0429 |
| None | 145 | 0.0276 |

→ Hard가 가장 높음. Hard persona는 구체적 정보를 일찍 주거나 종료 신호가 명확해 agent가 결판을 빨리 봄. None은 user simulator가 모호하게 행동해 agent 혼란.

**구조적 병목 (Phase 2 설계 함의)**:
1. **max_steps=200 도달 33.6%** — agent에 *명시적 종료 기준 없음*. B1 (ReAct) Thought/Action 강제가 종료 판단 개선 여부가 핵심 지표.
2. **ContextWindowExceededError 18건** — message count 평균 201(max_steps 도달 sims) × 16K token 한계. v2에서 max-model-len=32K로 해소.
3. **Stochasticity** — agent temperature=0.0인데도 0 tasks가 4/4 통과. tool-call routing의 stochastic noise + user simulator temp=0.5 영향.

**Leaderboard 위치 비교** (taubench.com 2026-05 telecom):

| 모델 | pass^1 |
|---|---|
| Claude Opus 4.6 / LongCat-Thinking-2601 | 0.993 |
| GPT-4o | 0.235 |
| Qwen3-Next-80B-A3B-Instruct | 0.132 |
| **Qwen2.5-7B-Instruct (우리 B0)** | **0.0475** |

→ 7B dense는 leaderboard 미보고 영역. B0 0.0475는 *novel data point*. Phase 2 개입으로 GPT-4o(0.235) 수준 추격이 noble bar.

**v1 → v2 전환**:
v1에선 B1/B2가 vLLM 사망으로 모두 connection error (전체 456 sims 무효). v2 (2026-05-27 09:56 시작)는 GPU1:9000, max-model-len=32K, B0+B1+B2 모두 재실행 — cross-baseline 비교를 위해 동일 조건. 예상 종료 16~18 KST.

#### Phase 1 v2 partial 결과 — Qwen2.5-7B-Instruct (2026-05-27 14:20, 76% 진행)

실행: 2026-05-27 09:56 시작, GPU1:9000, max-model-len=32K. 현 시점 (4h22m 경과) **B0 진행 중** (346/456 sims = 76% 완료, B1/B2 아직 시작 안 함).

| 지표 | v1 (16K, B0 완료 421/456) | **v2 partial (32K, 346/456)** |
|---|---|---|
| pass^1 | 0.0475 [0.031, 0.072] | **0.0260 [0.014, 0.049]** |
| pass^2 (best-of-2) | 0.0702 | 0.0439 |
| pass^4 | 0.1316 | 0.0175 (3 trials only avg) |
| max_steps 종료 | 33.6% | **52.9%** (★ v1보다 ↑) |
| infra error | 7.7% (35건) | **0% (32K 효과)** |
| user_stop 종료 | 55.7% | 46.0% |
| mms_issue pass | 0/97 user_stop | 0/146 (전체 fail 패턴 유지) |
| service_issue pass | 19.8% user_stop | 4.6% |
| mobile_data pass | 5.3% | 4.4% |
| Hard persona avg | 0.0735 | 0.0550 |

→ **32K가 ContextWindowExceededError 35→0건 해결**. 그러나 **max_steps 종료 비율 33.6%→52.9% 증가** — task가 *오래 끌릴 수 있게* 되니 hard task가 더 많이 max에 닿음.
→ pass^1 0.0475 → 0.0260 *감소* — v2가 *더 어려운* setting일 가능성, 또는 partial sampling 효과 (남은 110 sims가 더 어려운 mms task).
→ v2 종료 후 (예상 18~20 KST) 최종 분석.

#### Phase 1 Llama-3.1-8B-Instruct Cross-Model — **★ Catastrophic Failure 0/456** (2026-05-27 13:46 B0 완료)

실행: 2026-05-27 11:24 시작, GPU0:9001, max-model-len=32K. Llama-3.1-8B-Instruct + vLLM `--tool-call-parser llama3_json`. B0 N=456 trials=4 *완료*.

| 지표 | 값 |
|---|---|
| pass^1 | **0.0000 (0/456)** CI95 [0.0000, 0.0084] |
| pass^2/^4 | 0.0000 / 0.0000 |
| pass-all | 0 |
| Termination | **user_stop 100% (456/456)** ← max_steps 0건, infra error 0건 |
| **★ Tool calls per sim** | **mean=0.0, max=0, distribution: {0: 456}** |
| Mean conversation length | 20 messages, ~10 assistant turns |
| Hallucination retries | 0/456 |

**근본 원인 — Tool calling 작동 안 함**:

샘플 trajectory:
```
[assistant]: "Hi! How can I help?"
[user]:      "데이터 문제..."
[assistant]: "Let's troubleshoot..."  ← 텍스트만, tool_call 없음
[user]:      "I'll use check_status_bar()..."  ← user_sim도 같은 Llama, 
                                                  도구 호출을 텍스트로 "역할극"만
[assistant]: "Let's try..."          ← 여전히 tool_call 없음
[user]:      "I've used check_roaming_status()..."
[assistant]: "transfer_to_human_agent()"  ← 텍스트로만 명령어 작성
[user]:      "###TRANSFER###"        ← 시뮬레이션 종료, DB 변경 없음
```

원인 진단:
1. **vLLM `llama3_json` tool-call-parser 비호환** — Llama-3.1-8B가 vLLM의 parser 형식대로 tool_call JSON을 생성 못함
2. **Llama-3.1-8B-Instruct의 native function calling 능력 약함** — Qwen2.5-Instruct는 강한 native tool calling, Llama-3.1-8B는 사전 학습에서 tool calling 노출 적음
3. **User simulator 같은 모델 사용** — agent(Llama-8B)와 user_sim(Llama-8B)이 *같은 분포* → user_sim도 tool 안 부르고 *역할극*만 → 100% user_stop으로 빨리 종료

**Cross-model 실험 설계 결함 노출**:
- agent_llm = user_sim_llm 일 때 *모델 capability deficit*이 양쪽에 동일하게 발현
- → 진정한 cross-model 비교 위해 **user_simulator는 독립된 strong 모델** (GPT-4o API) 사용 권장
- 또는 Llama-3.3-70B / Llama-3.1-70B (큰 모델은 tool calling 학습 잘 됨)

**Llama 결과의 *positive value*** (negative result로서):
- "8B 클래스 모델은 vanilla function-calling 환경에서 enterprise tool task 사실상 0%" — 우리 main claim 강화
- Multi-relation ontology 개입 *없이는* 작은 모델로 풀 수 없음 입증
- 향후 Llama-3.3-70B / Qwen2.5-32B 등 *큰 모델 baseline* 추가 시 비교 강해짐
- **EXPERIMENT_DESIGN §6 Tier 1 권고**: Llama-3.1-8B는 *baseline reference만* (cross-model claim에는 부적합), Qwen2.5-7B를 primary로

#### Phase 1 Qwen + GPT-4o user_sim (Cross-Model Fair) — 2026-05-27 15:57 시작, v1.18 신규

Llama catastrophic 0/456 분석에서 **agent=user_sim self-play의 한계 노출**:
- agent (Llama-8B)가 tool call 못함
- user_sim (Llama-8B)도 같은 모델 → 단순히 "역할극" 하고 100% user_stop으로 빨리 종료
- DB 변경 없음 → reward 0

→ 두 가능성 분리 필요:
- (H1) Llama 모델 자체 한계 (vLLM parser 또는 native function calling 약함)
- (H2) self-play 한계 (agent=user_sim 같은 모델 시 양쪽 deficit 결합)
- (H3) 둘 다

**해결책: User simulator = GPT-4o 외부 API (tau2-bench leaderboard 표준)**.

실험 설계 — **Llama (agent) + GPT-4o (user_sim)**:
- 목적: Llama 0/456 catastrophic의 진짜 원인 분리
  - **(H1) Llama 자체 deficit**: vLLM `llama3_json` parser 비호환 또는 Llama-3.1-8B native function-calling 약함
  - **(H2) Self-play deficit**: agent=user_sim 같은 모델일 때 양쪽 deficit 결합
- 실험: 단 한 변수 변경 (user_sim = Llama → GPT-4o), Llama agent 동일
- 결과 시나리오:
  - **결과 ≈ 0% 유지** → H1 확정 (Llama 자체 한계, GPT-4o user_sim도 도움 안 됨)
  - **결과 ↑ (≥5%p lift)** → H2 dominant (self-play 한계 컸음, model capability는 충분)
  - **부분 lift** → 두 원인 모두

진행 중 setup:
- **Agent**: NousResearch/Meta-Llama-3.1-8B-Instruct (local vLLM GPU0:9001, max-model-len 32K, tool-call-parser=llama3_json)
- **User_sim**: openai/gpt-4o (OpenAI API endpoint, leaderboard standard)
- task_split=base N=114, trials=4, max_steps=200, max_concurrency=8
- API key: `/home/woori/.openai_key` (chmod 600), curl /v1/models verified HTTP 200
- 시작 2026-05-27 16:01:45 KST
- out-dir: `reports/facet_rft_2026/phase1_baseline/base_n114_llama_gpt4ouser/`

이 setup의 가치 — *원인 분리*:
1. **Llama 0% catastrophic 진짜 원인 확정**: model self vs self-play 두 가설 분리 가능
2. **Leaderboard 비교 가능**: tau2-bench 공식 leaderboard 대부분 GPT-4o user_sim 사용
3. **Negative result strengthening**: 만약 GPT-4o user_sim에서도 Llama 0% 유지면 *우리 main claim 강화* (8B vanilla = enterprise tool 불가능)
4. **Future Llama-70B 비교 setup 준비**: user_sim 동일 (GPT-4o)이면 큰 모델 추가 시 fair

#### Phase 1 setup 매트릭스

| Setup | Agent | User sim | 시작 | 디렉토리 | 의의 |
|---|---|---|---|---|---|
| v1 (16K, partial) | Qwen-7B | Qwen-7B | 2026-05-27 02:15 | `base_n114_v1_16k_partial/` | 초기 측정, B1/B2 vLLM 사망 |
| **v2** (32K, self-play) | Qwen-7B | Qwen-7B | 2026-05-27 09:56 | `base_n114/` | self-play 한계 측정 (진행 중) |
| ~Llama self-play~ (kill) | Llama-8B | Llama-8B | 11:24 → 14:23 kill | `base_n114_llama/` | 0/456 negative result 보존 |
| **Llama + GPT-4o** ★ | Llama-8B | **GPT-4o** | 2026-05-27 **16:01** | `base_n114_llama_gpt4ouser/` | **원인 분리 (H1 vs H2), leaderboard standard** |

#### Smoke3 (chain=1, max_steps=200) vs base v1 (chain 2-9) 비교

smoke3 (small split N=20, 14 sims 저장, killed)에서 partial pass^1 = 0.214 (3/14) — base v1의 0.044와 약 5× 격차. **task split 구조 차이로 정밀 분석 시 통계적 분리 확인**:

| 항목 | smoke3 (small) | base v1 |
|---|---|---|
| chain_len 분포 | **1.0** (전부 단일 이슈) | 2~9 (평균 ~4.3) |
| task_id 공유 (small ∩ base) | **0** | 0 |
| pass^1 | 0.2143 (3/14) | 0.0439 (20/456) |
| Wilson 95% CI | [0.076, 0.476] | [0.028, 0.067] |
| CI 중첩 | **분리됨** (smoke3 LB 0.076 > v1 UB 0.067) | |

→ smoke3의 0.22는 **chain=1 단일 이슈 task의 7B 능력**이고, base의 0.044는 **multi-issue chained planning의 7B 능력**. 두 측정은 동일 모집단 아님. 공식 leaderboard는 base 조건 (multi-issue) 이므로 0.044가 정확한 비교군.

**chain length × category 분해 (v1 base 456 sims)** — pass^1 표:

| chain | service_issue | mobile_data_issue | mms_issue |
|---|---|---|---|
| 2 | 0.139 (5/36) | 0.094 (3/32) | **0.000 (0/32)** |
| 3 | 0.056 (2/36) | 0.000 (0/32) | **0.000 (0/36)** |
| 4 | **0.222** (8/36) | 0.042 (1/24) | **0.000 (0/24)** |
| 5 | 0.125 (1/8) | 0.000 | **0.000 (0/20)** |
| 6+ | n/a | 0.000 | **0.000** |

핵심 관찰:
1. **Chain-length scaling**: 모든 카테고리에서 chain≥6 = 0%. Prompt-only 한계.
2. **MMS multi-step deficit**: chain=2 영역에서 non-mms = 11.8% (8/68) vs mms = 0% (0/32). chain=4 영역 non-mms = 15.0% (9/60) vs mms = 0% (0/24). **같은 chain 깊이에서도 mms만 풀리지 않음** — chain length만으론 설명 불가.
3. **Chain=1 mms는 풀린다** (smoke3 1/2 = 50%, n작음): `break_app_sms_permission [None]` 통과. 모델 weight에 mms 단일 도구 사용 능력은 *존재*.

#### "MMS = 추가 학습 필요"인지 — 결정은 Phase 2 후

세 가설 분기:

| 가설 | 함의 | 검증 신호 |
|---|---|---|
| **A. Pure weight gap** | mms domain SFT/RFT (Phase 4) 필수 | Phase 2 steering lift ≈ 0%p |
| **B. LRH 적용 가능** | mms 표현이 weight에 약하게라도 존재 (chain=1 50%가 증거) → steering이 *증폭* | Phase 2 lift ≥ +5%p in mms chain 2-4 |
| **C. Hybrid 필요** | Phase 2 부분 lift, Phase 3 cross-attn 또는 RFT가 완성 | Phase 2 +1~5%p, Phase 4 추가 lift |

**현 데이터는 가설 A를 약하게 반박** (chain=1 mms 50% 통과). 가설 B/C 중 어느 쪽인지는 Phase 2 결과로 결정.

#### Phase 2 측정 설계에 추가할 항목

- **Chain length stratified 분석**: chain {2,3,4,5,6+} 각각에서 B0 vs Tx pass^1 lift 별도 보고. 평균 만 보면 chain=2 lift가 chain=8 noise에 묻힘.
- **MMS-specific Go/No-Go**: Phase 2 T1 또는 A6가 **mms chain=2-4 영역에서 ≥+5%p** lift = 가설 B 확정. lift 0% = 가설 A → Phase 4 우선순위 상향.
- **Chain=1 single-issue 결과는 별도 표** (선례 없는 sub-benchmark; leaderboard와 무관하지만 capability 분리 측정용).

### Phase 2a: Training-Free Vector Steering on base (2주)

```
목표: T1 (Steering-Only) 검증 + A6 (Per-head Rotation) 비교

실험:
  1. 관계별 steering vector 추출
     - τ²-bench 텔레콤: precedes, requires, mutex 각각
     - Contrast pair: 관계 충족 예시 200개 vs 위반 예시 200개
     - Phase 0 확정 레이어에 차등 주입:
         first_tool / next_tool 벡터 → L01-L18
         precedes / mutex 벡터     → L02-L05

  2. T1 (Additive) on τ²-bench (retail/telecom/airline)
     - α (steering strength) 그리드 서치: {0.1, 0.3, 0.5, 1.0, 2.0}
     - 최적 α에서 pass^1 측정

  3. A6 (Per-head Rotation) — SEKA rotation 재활용
     - SEKA base rotation R_seka → warm-start 초기화
     - 탐색 공간: SEKA 효과적 (layer, head) ∩ {L02-L05}
     - relation별 Givens rotation으로 R_ontology 구성 (학습 불필요)
     - ORDERING 제약 관련 pass^1 세부 분석

  4. A7 (Method × Relation Cross) — 핵심 신규 실험
     - 관계 유형 {precedes, mutex, requires, parameter_feeds, ...} 각각에 대해
       T1 vs A6 CVR (Constraint Violation Rate) 측정
     - 이론 예측표(§4.3 A7) vs 실측 결과 비교
     - 성공 시: Method Router (관계 유형 → 방법 자동 선택) 추가 기여로 정식화

  5. Ablation: A1~A3 (관계 유형별 기여도), A4 (random vector), A5 (layer)

측정:
  pass^1 vs B0/B1/B2 비교 (모든 측정은 base 모델에서)
  Constraint Violation Rate (ordering 위반, mutex 위반 각각)

Go/No-Go: T1@base vs B2(텍스트 직렬화) +3%p 이상 → Phase 3 진행
          단, training-free 단독 contribution은 *어떤 lift든* 명제 C' 검증에 가치 — Phase 3 진행 결정에 무관하게 보고
```

**▶ Phase 2a/2c/2d steering 실측 결과 (2026-05-29, Qwen2.5-7B, telecom, gpt-4o-mini user_sim)**

| 조건 (α=0.5) | pass^1(all) | pass^1(svc) | transfer%(svc) |
|---|---|---|---|
| a0 (무steer) | 0.176 | 0.275 | 27.5% |
| T1 validates (peak) | 0.192 | 0.300 | 37.5% |
| gate decay / orth | 0.183 / 0.200 | 0.300 / 0.350 | 30% / 25% |
| relation error_fallback | 0.183 | 0.300 | 30% |
| relation retry_after_fail | 0.217 | 0.400 | 45% |

핵심 발견:
- **모든 steering 조건이 baseline noise band [0.176-0.217] 내** (N≤120, CI±0.10) — robust한 +3%p lift 없음.
- **표상-공간 facet 상보 구조(AXIS-1: retry↔error_fallback cos<0)가 인과 행동으로 이어지지 않음** — 예측 반전: retry_after_fail(EXEC 극)이 transfer(45%)·pass 최고. 『표상 상보 ≠ 인과 상보』.
- 효과 비특이적: 모든 steering이 transfer를 a0 대비 올리나 relation별 차이는 노이즈; pass^1 미반영(H2 capability ceiling 지지).
- **C5(actuator 강도) 부정 경향 / C3(relation→behavior 매핑) 미지지.**

**판단**: 상수 single-relation steering(=bias-1step-RFT, class-hierarchy 최약점)은 7B에서 weak·non-specific actuator → power test 생략, **LoRA-RFT(L0=PEFT-RFT)로 피벗** (사용자 결정 2026-05-29). 상세 framework·사다리: phase2_steering/STEERING_CONTROL_DESIGN.md

### Phase 3: Cross-Attention 주입 구현 (3주)

```
목표: T2 (Cross-Attention) 검증

구현:
  1. 온톨로지 노드 인코더
     - 각 도구 노드: 이름 + 설명 + 관계 → 임베딩
     - Semantic Graph Module (GMT 방식 참조)
  
  2. Cross-Attention 모듈
     - Qwen2.5-7B Transformer 레이어에 cross-attn 삽입
     - Query: LLM hidden state
     - Key/Value: 온톨로지 노드 임베딩
     - 학습: LoRA (r=8, α=16), τ²-bench 학습셋만 사용
  
  3. T2 on τ²-bench × 3 domains
  4. T2 on TPS-Bench (Easy + Hard)
  5. T2 on FlowBench (텍스트 직렬화 B2와 직접 비교)

측정:
  pass^1, pass^5, Constraint Violation Rate
  레이어별 attention weight 시각화 (해석 가능성)

Ablation: A4 (random vector vs 온톨로지 벡터), A5 (레이어별)
```

### Phase 2b: Steering on T2-tuned (Compositional A, 1주)

```
목표: T3 = T2 + T1@T2 검증 — Compositional hypothesis 검증 Stage 1
근거: T2 LoRA SFT 후 모델은 ontology routing이 weight에 명시적으로 학습됨
      → 그 모델에서 contrast pair 재실행 → v_relation_T2 추출
      → base의 v_relation보다 *분리도가 높을 것으로 기대*
      → 동일 α로 가산해도 lift 폭 ↑

실험:
  1. T2 모델 freeze
  2. contrast pair (관계 충족/위반 예시 200×42)에서 hidden state 재추출
     - T2 cross-attn 직후의 residual stream 기준
  3. v_relation_T2 ∈ ℝ^d 계산 (mean-of-differences)
  4. α 그리드 서치 {0.1, 0.3, 0.5, 1.0, 2.0}
  5. T3 = T2 + α·v_relation_T2 inference

비교 (4-way):
  - B0 (Phase 1)
  - T1@base (Phase 2a)
  - T2 alone (Phase 3)
  - T3 = T2 + T1@T2 (이 phase)

Go/No-Go: T3 vs T2 +3%p 이상 → Compositional hypothesis 확정
          mms_issue chain 2-4에서 별도 +5%p ≥ → 명제 B/C' 정량 증거

측정:
  pass^1 (overall), pass^1 by chain length, pass^1 by category
  v_relation_T2 separability metric (긍/부 예시 cosine 분리도)
  Comparison plot: ΔT1@base vs ΔT1@T2 by relation type
```

### Phase 4: 하이브리드 및 LATS 결합 + facet-RFT (3주)

```
목표: T4-LATS, T4-RFT 검증 — heavy training path

실험 path α (Tree Search):
  1. T4-LATS (Cross-Attn + LATS)
     - τ²-bench simulator를 LATS reward source로 연결
     - MCTS rollout 수: {4, 8, 16}
     - Facet Prior로 탐색 가중치 부여

실험 path β (Reinforcement Fine-Tuning, v1.11 신규):
  2. T4-RFT (facet-aware RFT)
     - Rollout: τ²-bench train 74 task × 16 trial = 1184 rollout
     - Verifier: τ²-bench simulator pass/fail + ontology violation 패널티
        precedes 위반 → −0.1
        requires 위반 → −0.2
        mutex 위반 → −0.3
     - Update: GRPO (DeepSeek-R1 방식) 또는 rejection-SFT
     - Cross-attn LoRA + self-attn LoRA 동시 학습 (또는 cross-attn freeze 옵션)
     - 라운드: 2-4 iteration

측정:
  pass^1, pass^5, Compute Budget (rollout 수 vs 성능)
  T4-LATS vs T4-RFT 비교 (training-free reward vs training)
```

### Phase 2c: Steering on T4-RFT (Compositional B, 조건부 1주)

```
목표: T5 = T4-RFT + T1@T4 검증 — Compositional hypothesis Stage 2 (사용자 thesis, 명제 C')
근거: RFT가 multi-step ontology routing을 weight 자체에 깊이 내재화
      → T2 LoRA(cross-attn만)보다 representation separability 더 강함
      → v_relation@T4가 가장 깨끗한 방향 따라감
      → ΔT1@T4가 ΔT1@base, ΔT1@T2보다 클 것으로 기대

진입 조건 (Go/No-Go):
  C1: T4-RFT (Phase 4) absolute pass^1 ≥ 0.144 (B0+10%p)
  C2: T4-RFT 후 모델에서 v_relation separability (긍/부 cosine) > base의 1.5×

  C1 ∧ C2 → Phase 2c 진입
  C1만 → T4-RFT 단독으로 main result
  둘 다 못 만족 → Phase 4 가설 A 우세 (pure weight gap, capability ceiling)

실험:
  1. T4-RFT 모델 freeze
  2. 동일 contrast pair에서 v_relation_T4 추출
  3. T5 = T4-RFT + α·v_relation_T4
  4. α 그리드 서치, 최적값에서 pass^1 측정

비교 (6-way 확장):
  - B0, T1@base, T2, T3 (Phase 2b), T4-RFT, T5

측정:
  pass^1, chain-stratified, category-stratified
  v_relation_T4 vs v_relation_T2 vs v_relation_base — separability quartet
  Inference-time cost (steering은 +0% latency / RFT는 학습 비용)

Phase 2c의 핵심 메시지:
  "Representation quality × steering 효율 곱셈 관계 검증"
  ΔT1@base : ΔT1@T2 : ΔT1@T4 = ? : ? : ?  (논문 main figure 후보)
```

### Phase 5: 도메인 일반화 + T6 Triple Composition (조건부 2주)

```
목표: H2 (Cross-Domain 일반화) 검증 + T6 (조건부) 측정

Part 1: Cross-Domain 일반화 (1주)
  학습: Retail 온톨로지만 사용
  평가: Telecom, Airline (새 도메인)
  
  비교:
    - T2 (온톨로지 교체만으로 새 도메인 적용)
    - T3 (T2 + T1@T2, Phase 2b 결과 반영)
    - T5 (T4-RFT + T1@T4, Phase 2c 결과 반영, 진입 시)
    - Routine fine-tuned (새 도메인에서 재학습 필요)
    - B2 (텍스트 직렬화, 새 도메인 텍스트 교체)

  측정:
    Cross-Domain Drop = (학습 도메인 pass^1) - (새 도메인 pass^1)
    목표: T2/T3/T5의 Drop < B4(Routine)의 Drop
    추가: T1 vector swap (도메인별 v_relation 재추출만으로 transfer 가능성)

Part 2: T6 Triple Composition (조건부, 1주, v1.11 신규)
  목표: T6 = T2 + T4-RFT + T1@(T2+T4) — Pareto upper bound 측정

  진입 조건 (모두 충족 시):
    C3: T2 단독 vs B0 +5%p 이상 (Phase 3 ✓)
    C4: T3 = T2+T1 vs T2 +3%p 이상 (Phase 2b compositional A ✓)
    C5: T5 = T4+T1 vs T4 +3%p 이상 (Phase 2c compositional B ✓)
    C6: T4-RFT 단독 vs B0 +10%p 이상 (Phase 4 ✓)
    
    C3 ∧ C4 ∧ C5 ∧ C6 → T6 측정
    하나라도 실패 → T6 skip, 단순 모델 보고
  
  실험:
    1. 학습 cascade: base → T2 LoRA SFT → T4-RFT (cross-attn freeze 옵션 검증)
    2. T2+T4 모델에서 v_relation_T2T4 재추출
    3. T6 inference: T2+T4 모델 + α·v_relation_T2T4
    4. α 그리드 서치
  
  비교 (8-way 완전 ablation):
    B0, T1@base, T2, T3, T4-RFT, T5, T2+T4 (steer 없음), T6
  
  Risk:
    RFT가 T2 cross-attn LoRA overwrite → 완화: cross-attn LoRA freeze 옵션
    Inference framework 복잡도 ↑ → custom vLLM plugin 또는 HF inference
    Reproducibility ↓ → 모든 seed/hyperparam 상세 보고
  
  측정:
    pass^1 (overall + chain-stratified + category-stratified)
    Pareto frontier plot: x=학습 비용, y=pass^1
    각 lever의 marginal lift 분해 (T6 - T2+T4, T6 - T5, T6 - T3)
```

---

## 8. 예상 결과 및 논문 클레임

### 8.1 예상 성능 테이블 (τ²-bench pass^1)

| 조건 | Retail | Telecom | Airline | 비고 |
|---|---|---|---|---|
| B0 Vanilla | ~0.30 | ~0.20 **→ 실측 0.0475** | ~0.15 | Phase 1 v1 (N=114, trials=4) |
| B1 ReAct | ~0.35 | ~0.25 (v2 측정 중) | ~0.20 | |
| B2 Text Serial | ~0.40 | ~0.30 (v2 측정 중) | ~0.22 | |
| B3 KnowAgent | ~0.42 | ~0.32 | ~0.23 | |
| **T1@base** (Phase 2a) | ~0.45 | ~0.38 | ~0.26 | training-free |
| **T2 Cross-Attn LoRA** (Phase 3) | ~0.50 | ~0.45 | ~0.30 | LoRA SFT |
| **T3 = T2 + T1@T2** (Phase 2b) | ~0.53 | ~0.48 | ~0.32 | Compositional A |
| **T4-LATS** | ~0.55 | ~0.50 | ~0.34 | Tree search |
| **T4-RFT** (Phase 4) | ~0.60 | ~0.55 | ~0.38 | facet-aware RFT |
| **T5 = T4-RFT + T1@T4** (Phase 2c) | **~0.65** | **~0.60** | **~0.42** | Compositional B (명제 C') |
| **T6 = T2 + T4-RFT + T1@(T2+T4)** (Phase 5) | **~0.68** | **~0.63** | **~0.45** | Triple (Pareto upper bound) |
| B4 Routine (FT) | ~0.75 | ~0.70 | ~0.55 | 상한선 |

*B0 Telecom 사전 추정 ~0.20은 D0 (Channel A) 기반. **실측 0.0475는 4배 낮음** — 7B 모델 vanilla 한계가 D0 예측보다 훨씬 강함. T1-T6 셀의 예측치도 같은 규모로 *과대평가* 가능성. v2 실측 후 모든 셀 재추정 필요.*

**Pareto frontier 예측** (학습 비용 × pass^1):
```
                       pass^1
                         │
           T6 (Triple) — █  ← upper bound
                         │  ╲
              T5 (RFT+St) █  ← compositional B
                         │  ╲
                 T4-RFT — █  ← heavy training
                         │  ╲
                T3 (Hyb) — █  ← compositional A
                         │  ╲
                T2 (LoRA)— █  ← light training
                         │  ╲
              T1 (Steer) — █  ← training-free contribution
                         │  ╲
                B0 — — — █
                         └────────────────── 학습 비용
                       0       LoRA    RFT
```

핵심 가설 (검증 대상):
- (a) ΔT1@base < ΔT1@T2 < ΔT1@T4 (steering lift가 representation quality와 함께 증가)
- (b) T6 > T5 > T3 > T2 > T1@base (각 lever가 additive 또는 multiplicative)
- (c) T6 − (T2+T4 no-steer) ≥ ΔT1@base (steering이 모델 단독보다 cheaper boost)

### 8.2 논문 핵심 클레임

```
Claim 1 (주장):
  "온톨로지 관계를 Cross-Attention으로 주입하면
   텍스트 직렬화 대비 pass^1 +X%p, 
   Constraint Violation Rate -Y% 향상"

Claim 2 (일반화):
  "Training-free 방법이 새 도메인에서
   domain-specific fine-tuning 대비 Z% 높은 성능 유지"

Claim 3 (효율):
  "TPS-Bench Hard에서 도구 의존성 그래프 주입이
   병렬화 효율을 W% 개선하면서 완료율 유지"

Claim 4 (Compositional, v1.11 신규 — 사용자 thesis):
  "Steering vector의 effectiveness는 underlying model의
   representation quality에 *곱셈적으로* 비례한다.
   ΔT1@base < ΔT1@T2 < ΔT1@T4 < ΔT1@(T2+T4)
   학습된 모델 + training-free steering 합성이
   학습 단독보다 cheaper boost를 제공."

Claim 5 (Pareto frontier, v1.11 신규):
  "8-cell ablation (T2 × T4 × T1) 매트릭스에서
   T6 (Triple composition)이 학습 비용 대비 pass^1
   Pareto frontier 최우상점. 각 lever의 marginal lift
   분해로 메커니즘 분리 가능."

Claim 7 (Enterprise Viability Metrics, v1.19 신규):
  "Pass^1 단독으로는 enterprise agent 평가 *부족*.
   Phase 1 v2 실측에서 모델별 *실패 모드*가 질적으로 다름:
     - Llama-3.1-8B: Quick-failure 91.7% (즉시 포기, enterprise 불가)
     - Qwen2.5-7B vanilla: Over-diagnosis 54-58% (200 step trap)
     - Hermes-3-Llama-8B (FC tune): Balanced (EVS 6.4× Qwen)
   
   따라서 우리 측정 protocol에 다음 추가:
     - ATR / ATR p95 (latency)
     - max_steps_rate (over-diagnosis trap)
     - quick_failure_rate (즉시 포기)
     - productive_resolution_rate (효율 성공)
     - EVS (Enterprise Viability Score)
   
   우리 ontology 개입의 *부가 가치*:
     - Phase 2-4가 pass^1 lift 외에 EVS lift
     - achieves_goal/plan_committed_to_goal steering이
       *종료 신호 강화* → max_steps_rate 감소
     - facet-RFT reward에 *짧은 해결 보너스* 포함 가능"

Claim 6 (Novelty 정확한 articulation, v1.14 → v1.15 정정 강화):
  "Process reward 자체는 Lightman 2023+에 prior 풍부.
   Dependency-aware process reward (단일 relation type)는 Jiayang
   2603.24709 (2026-04)에 1개월 prior. 단 Jiayang의 dependency graph는
   *parameter_feeds + precedes* 합쳐진 *단일 type edge*만 표현 가능.
   
   우리가 *처음*인 것은 다음 6개 동시 cover:
   
    (i) ★ **Multi-relation ontology** (42 distinct semantic types):
        Jiayang single-type vs 우리 16 directional + 3 symmetric 
        + 4 conditional + 19 categorical
        — mutex / guardrail / conditional_on / validates / 
        retry_after_fail / compensates / fan_out / backtrack_to 
        등 *Jiayang으로 표현 불가능한* 관계 27+개
        
    (ii) **Relation-type-aware reward weighting**:
         precedes −0.1 vs mutex −0.3 vs guardrail −0.5 차등
         Jiayang은 모든 edge uniform penalty
         
    (iii) **Geometry-aware intervention** (Phase 0 v3 검증):
          directional → A6 per-head rotation
          symmetric/categorical → T1 additive
          학습된 dependency graph (Jiayang) 또는 학습된 PRM (AgentPRM)으로 
          관계 유형별 다른 개입 불가능
          
    (iv) **Cross-domain pre-defined ontology**:
         Jiayang은 task-specific workflow template 사람 작성
         우리는 enterprise tool ontology 42-relation 통합 정의
         
    (v) **AFOD auto-discovery** from tool schema (zero-label transfer)
        — Jiayang은 manually curated
        
    (vi) **4-layer injection** (probing/steering/cross-attn/RFT reward)
         — 모든 prior는 reward 1-layer만
   
   ★ 핵심 differentiation: Jiayang은 *dataflow language* (data flow + 
   ordering의 단일 edge type), 우리는 *planning-theory semantic 
   predicates* (PDDL/HTN/GoalAct 통합한 42 relation 분류).
   
   '온톨로지를 process reward로 사용'은 우리가 처음 아님 (단일 relation).
   '*다종 relation* ontology를 process reward + multi-layer injection에 
   사용'은 우리가 처음."
```

---

## 9. 논문 포지셔닝 — 전체 관련 연구 지형

### 9.0 포지셔닝 구조 개요

```
관련 연구를 5개 계열로 분류:

  계열 1: GraphRAG — 텍스트 직렬화 주류
  계열 2: GMT 계열 — 그래프 구조 직접 주입 (우리의 직접 선조)
  계열 3: Steering 계열 — LLM 내부 표현 개입 (우리의 또 다른 선조)
  계열 4: Routine 계열 — 구조화 계획 + Fine-tuning
  계열 5: Agent Planning 계열 — 에이전트 아키텍처 + 계획 수립

우리의 위치:
  계열 2 (GMT)의 주입 메커니즘 +
  계열 3 (Steering)의 test-time 적용 +
  계열 4/5의 계획 수립 태스크
  = 세 계열의 교차점. 단독으로는 미개척.
```

---

### 9.1 계열 1: GraphRAG — 텍스트 직렬화 계열

> **공통 한계**: 그래프 → 텍스트 변환 시 위상 정보 손실, attention 희석, 소프트 제약

| 방법 | 연도 | 주입 방식 | 우리와의 차이 |
|---|---|---|---|
| **Microsoft GraphRAG** (Edge et al.) | 2024 | 커뮤니티 요약 텍스트 → 프롬프트 | 직렬화 의존, 위상 정보 손실 |
| **LightRAG** | 2024 | 엔티티+관계 텍스트 → 프롬프트 | 동일 한계 |
| **HippoRAG / HippoRAG 2** | 2024/2025 | 해마 구조 경로 텍스트화 | 메모리 검색 특화, 계획 아님 |
| **SubgraphRAG** | 2024 | 서브그래프 추출 → 텍스트 | 크기 제약 내 최대 커버리지 |
| **StructRAG** | 2024 | 태스크별 최적 구조 선택 → 텍스트 | 구조 유형 자동 선택, 여전히 직렬화 |
| **RAPTOR** | 2024 | 계층적 요약 트리 → 텍스트 | 장문서 RAG, 계획 아님 |
| **ToG / ToG 2.0** | 2023/2024 | 그래프 탐색 경로 텍스트화 | 추론 체인 명시적, 직렬화 |
| **KGP, SURGE** | 2023-2024 | 서브그래프 직렬화 | 개인화/대화 특화 |

```
우리의 차별점 vs 계열 1 전체:
  이 계열은 모두 "Graph → Text → Prompt"의 병목을 가짐
  우리는 이 병목을 우회하여 구조를 직접 LLM 내부에 주입
  → 텍스트 직렬화 대비 위상 정보 보존, attention 희석 없음
```

---

### 9.2 계열 2: GMT 계열 — 그래프 구조 직접 주입 (선조 계열)

> **공통 특징**: 텍스트 직렬화 없이 그래프 구조를 LLM에 직접 통합  
> **공통 한계**: KGC/QA 태스크 특화, 계획 수립 적용 없음, 대부분 학습 필요

#### 선구자 (2019-2022)

| 방법 | 연도 | 메커니즘 | 한계 |
|---|---|---|---|
| **ERNIE (Baidu)** | 2019 | 엔티티/관계 임베딩 → attention 레이어 통합 | 사전학습 필요, KG 보강 |
| **K-BERT** | 2020 | KG 트리플 → soft attention bias 직접 주입 `A = softmax(QKᵀ/√d + B_kg)` | BERT 특화, 동적 그래프 미지원 |
| **QA-GNN** | 2021 | KG 노드 → LLM attention weight에 영향 | GNN 학습 필요, QA 특화 |
| **GreaseLM** | 2022 | GNN ↔ LLM Transformer 모달리티 상호작용 레이어 | GNN+LLM 동시 학습, 대형 모델 부적합 |

#### 최신 (2024-2025)

| 방법 | 연도 | 메커니즘 | 한계 |
|---|---|---|---|
| **DualR** | 2024 | GNN이 attention 가중 경로 추출 → LLM에 증거 제공 | GNN 학습 필요, KG QA 특화 |
| **TEA-GLM** | 2024 | GNN 표현 → PCA+linear projector → LLM embedding space 정렬 | 학습 필요 |
| **FLAME** | 2024 | KGE 모델로 엔티티/관계 벡터화 → structured token으로 변환 → frozen LLM 입력에 직접 주입 (fine-tuning 없음) | KGC 특화, 계획 적용 없음 |
| **GMT** (Ruitong Liu et al.) | 2025 | 로컬 이웃 → Semantic Graph Module → 고정 수 Graph Memory Token → Transformer 다중 레이어 Cross-Attention으로 주입 (LoRA만 학습) | KGC 특화, 계획 적용 없음 |
| **Beyond Textual Context** | 2025 | 그래프 구조 → adaptive space alignment → LLM embedding 공간 정렬 | hallucination 감소 특화 |
| **GLANCE** | 2025 | GNN이 약한 노드만 LLM에 선택적 라우팅 | 노드 분류 특화 |

```
우리의 차별점 vs 계열 2:
  FLAME: KGC 특화, 계획 수립 미적용
  GMT:   KGC 특화 + LoRA 학습 필요, 계획 수립 미적용
  K-BERT: attention bias 주입 (우리와 유사) but 2020년, BERT 특화, 동적 미지원

  우리:  GMT/FLAME의 주입 메커니즘을
         기업 도구 온톨로지 + 다단계 계획 수립 태스크에 최초 적용
         + (T1 조건) training-free vector steering과 결합
```

---

### 9.3 계열 3: Vector Steering 계열 — LLM 내부 표현 개입

> **공통 특징**: LLM 가중치 수정 없이 inference time에 activation 직접 조작  
> **공통 한계**: 단순 속성 제어 특화, 구조적 지식(그래프/온톨로지) 활용 없음

#### 기반 연구 (2023)

| 방법 | 연도 | 대상 개념 | 한계 |
|---|---|---|---|
| **Representation Engineering (RepE)** (Zou et al.) | 2023 | 정직성, 감정, 도덕 원칙 | 단일 속성, 관계 구조 없음 |
| **CAA (Contrastive Activation Addition)** (Rimsky et al.) | 2023 | 안전/비안전 행동 | 이진 개념, 관계 구조 없음 |
| **ITI (Inference-Time Intervention)** (Li et al.) | 2023 | 진실성 (truthfulness) | Attention head 단위, QA 특화 |
| **Function Vectors** (Todd et al.) | 2023 | ICL task 함수 자체를 벡터로 추출 | 태스크 함수, 온톨로지 관계 아님 |
| **ActAdd** (Turner et al.) | 2023 | 감정/주제 편향 | 단순 속성 |

#### 계획/추론 특화 (2024-2025) ← 우리와 가장 가까운 계열

| 방법 | 연도 | 대상 | 결과 | 한계 |
|---|---|---|---|---|
| **Uncovering Latent CoT Vectors** | 2024 | CoT 행동 방향 벡터 | 단일 벡터로 CoT 없이 정확도 향상 | 일반 추론, 도구 계획 아님 |
| **Steering LLM Reasoning (Bias-Only)** | 2025 | 수학적 추론 | RL fine-tuning과 동등한 정확도 | 수학 도메인, 구조적 지식 없음 |
| **Understanding Reasoning via Steering** (ICLR 2025) | 2025 | DeepSeek-R1 추론 행동 (backtracking 등) | 추론 전략 제어 가능 | 추론 과정 제어, 도구 선택 아님 |
| **Feature Extraction & Steering for CoT** | 2025 | CoT 언어적/상징적 과정 분해 | 추론 벤치마크 향상 | 일반 CoT, 계획 구조 없음 |
| **Activation Steering for CoT Compression** | 2025 | CoT 압축 | 재학습 없이 CoT 압축 | 압축 특화 |
| **VSPO (Vector-Steered Policy Optimization)** | 2025 | RL 학습 중 행동 강화 | sparse reward 문제 해결 | Training-time (test-time 아님) |
| **Fine-Grained Activation Steering** | 2025 | 다중 속성 동시 제어 | 직교 서브스페이스 분리 | 속성 제어, 관계 구조 없음 |
| **Dynamically Scaled Activation Steering** | 2024 | 조건부 개입 | 입력 의존 강도 조절 | 조건부 제어, 계획 아님 |

#### KV Cache Steering — 우리와 가장 밀접한 신규 연구

| 방법 | 연도 | 대상 | 결과 | 개입 위치 |
|------|------|------|------|-----------|
| **KV Cache Steering** (Belitsky et al.) | 2025 (arXiv 2507.08799) | CoT 추론 유도 (frozen LLM) | Llama-70B: +4.6% GPQA, +7.4% MATH | **KV-side** (K,V cache 직접 수정) |

```
KV Cache Steering의 핵심 통찰:
  - 기존 Activation Steering: 매 생성 스텝 개입 → 레이어/토큰 방향 복리 누적 → 불안정
  - KV Cache Steering: prefill 후 1회 KV cache에 가산 → 증폭 없음 → 더 안정적

  스티어링 벡터 추출: Mean-of-Differences (gradient 불필요)
    S^k_l = (1/N) Σ [f_l(p⁺) − f_l(p⁻)]
    p⁺ = CoT 포함 프롬프트, p⁻ = 답만 있는 프롬프트
  적용:
    K'_l = K_l + c^k · S^k_l  (prefill 후 단 1회)

우리의 차별점 vs 계열 3 + KV Cache Steering:
  기존 steering  = 단일 속성/행동 제어 (CoT 여부, 정직성 등)
  KV steering    = CoT 스타일 제어 (여전히 단일 속성)
  우리           = 이항 관계(binary relation) 벡터화
                   (precedes(A,B), mutex(A,B), ..., plan_revised_to(obs,old,new))
                   → 42종 구조적 관계를 activation/KV에 인코딩
                   → 다단계 계획 정확도(pass^1) 향상이 목표

  추가 신규성:
    우리 A8 실험 = KV Cache Steering 방법론을 온톨로지 관계별로 확장
    contrast_pairs_v3.json의 (p⁺, p⁻) 쌍이 S^k, S^v 추출의 직접 재료
    = 계획 수립 도메인에 KV Cache Steering을 최초로 체계적으로 적용

  핵심 신규성:
    "단일 개념"이 아닌 "관계적 구조" (42종)를 steering source로 사용
    Q-side(T1/A6) ↔ KV-side(A8) 개입의 체계적 비교 = 이 방향의 연구 없음
```

---

### 9.4 계열 4: Routine 계열 — 구조화 계획 + Fine-tuning

> **공통 특징**: 계획을 구조화된 표현으로 명시 → 높은 정확도  
> **공통 한계**: Fine-tuning 또는 SFT+RL 필수, 새 도메인 적응 비용 높음

**Routine 정확한 정체 (v1.12 정정)**: arXiv 2507.14447 (2025-07) 직접 확인 결과:
- **Prompt template** (step 번호 + 설명 + 입출력 + tool) **+ LoRA SFT**
- **Reward 없음** (no process/outcome/rule reward, RL 안 함)
- **명시적 inter-tool ontology 없음** (variable memory로 dependency 처리)
- 학습: LoRA r=8, BUTTON 4,209 samples + scenario-specific 537
- 결과: GPT-4o 41.1%→96.3% (template만), Qwen3-14B 32.6%→83.3% (SFT)→95.5% (scenario-specific)
- 이전 메모리의 "SFT+RL 필수" 기술은 *오류*. SFT only.

| 방법 | 연도 | 구조화 방식 | 학습 | 성능 | 한계 |
|---|---|---|---|---|---|
| **Routine** (기업용) | 2025 | step template (#/name/input/output/tool) + variable memory | **LoRA SFT only** (RL 없음, reward 없음) | GPT-4o 41%→96% (template만), Qwen3-14B 32.6%→83.3%→95.5% | scenario별 routine 사람이 작성, 새 task당 재학습 |
| **GAP** (NeurIPS 2025) | 2025 | 서브태스크 의존성 DAG | SFT + RL | ReAct 대비 유의미한 향상 | MHQA 특화, 기업 도메인 아님 |
| **KnowAgent** (NAACL 2025) | 2024/2025 | 행동 지식 텍스트 직렬화 + CoT | Self-learning | HotpotQA/ALFWorld 향상 | 텍스트 직렬화 병목 |
| **FlowBench 기반 방법들** (EMNLP 2024) | 2024 | 워크플로우 텍스트/코드/플로우차트 | — | GPT-4o도 낮음 | 포맷 다양성, 성능 낮음 |
| **Graph-CoT** (ACL 2024) | 2024 | 그래프 위 CoT 추론 단계화 | — | 멀티홉 QA 향상 | QA 특화, 도구 계획 아님 |
| **From Experience to Strategy** (2025) | 2025 | 과거 궤적 → 그래프 메모리 → 전략 재사용 | RL 가중치 최적화 | RL 학습 중 일관 향상 | 학습 필요, 특정 도메인 |
| **DAMCS** (멀티에이전트 KG) | 2025 | 계층적 KG 메모리 공유 | — | 협력 계획 향상 | 멀티에이전트 특화 |
| **Ego-Graph Tool Retrieval** (IJCNLP 2025) | 2025 | 1-hop ego 그래프 앙상블 도구 검색 | — | 91.85% recall | 검색 단계만, 계획 전체 아님 |

```
우리의 차별점 vs 계열 4:
  Routine:   구조화 표현이 핵심이지만 fine-tuning으로 학습해야 함
  GAP:       그래프 계획 능력을 SFT+RL로 학습
  KnowAgent: 구조적 지식을 텍스트로 직렬화

  우리:  구조화 계획 표현(Routine의 통찰) +
         그래프 의존성(GAP의 통찰) +
         training-free activation 주입(계열 2/3의 메커니즘)
         = fine-tuning 없이 구조적 지식을 LLM 내부에 인코딩

  핵심 주장:
    "Routine의 96% 향상이 구조화 표현에서 온다면,
     그 구조를 fine-tuning 없이 직접 주입해도 동등한 효과가 가능한가?"
```

**Routine vs 우리 — 6가지 차별 (v1.12 갱신)**:

| 차원 | Routine | 우리 |
|---|---|---|
| Inter-tool 관계 | 없음 (variable memory만) | **42-relation ontology** (precedes/requires/mutex/...) |
| 학습 신호 | SFT only | **SFT + RFT** (process reward + outcome) |
| Reward | **없음** | outcome (DB match) + per-turn ontology violation penalty |
| 표현 layer | 단일 (prompt + weight via SFT) | **4 layer** (probing/steering/cross-attn/RFT reward) |
| Auto-discovery | 수동 routine 작성 (scenario마다 사람) | **AFOD**: tool schema에서 auto-extract |
| Training-free 가능? | ❌ (반드시 SFT) | **✅ Phase 2 (T1 steering) 단독 동작** |

→ **Routine은 우리의 직접 baseline이 아니라 *complementary direction***. Routine = "사람이 plan template 작성, 모델이 SFT로 그것 따르도록 학습". 우리 = "자동 추출된 ontology를 다층 representation에 주입, training-free 가능".

Routine의 96% lift는 *enterprise scenario에 사람이 정확한 routine template을 작성했을 때*. 새 scenario 마다 라벨링 비용 큼. 우리는 *자동* ontology에서 lift 측정 (사람 라벨 0).

---

### 9.4.5 계열 4½: Process Reward Models (PRM) — 우리 RFT reward 직접 선조 (v1.12 → v1.13 확장)

#### 9.4.5.1 PRM Foundations — Math/Code (Lightman lineage)

| 논문 | 연도 | Process reward 출처 | Task | 코드/Bench |
|---|---|---|---|---|
| **Lightman et al. "Let's Verify Step by Step"** | OpenAI 2023 | 학습된 PRM이 math step 채점 | MATH | PRM800K |
| **STaR** (Zelikman et al.) | Stanford 2022 | self-rationalizer, intermediate CoT | reasoning | – |
| **ReST** (Gulcehre et al.) | DeepMind 2023 | outcome reward + filter | language | – |
| **MathShepherd** (Wang et al.) | 2024 | Monte Carlo step reward | math | – |
| **CodeRL / RLEF** | 2022-24 | unit test per-step | code | HumanEval/MBPP |
| **ReST-MCTS*** (Zhang et al.) | 2024 | MCTS-based step value | reasoning | – |
| **Quiet-STaR** (Stanford) | 2024 | continuous thought tokens | reasoning | – |
| **PRL (Process Reward Learning)** | arXiv 2601.10201 | outcome → dense process via TD | reasoning | – |
| **Survey of PRMs** | arXiv 2510.08049 | meta-review | – | – |
| **Awesome-PRMs** | github.com/RyanLiu112 | comprehensive list | – | – |

#### 9.4.5.2 Agent PRM — Tool-Use 영역 (우리와 직접 인접, v1.13 신규)

| 논문 | 연도/arXiv | Process reward source | Task / Bench | 우리와의 차별 |
|---|---|---|---|---|
| **AgentPRM** | 2511.08325 (2025-11) | **TD + GAE 학습 PRM** (rule 아님) | general agent | rule-based vs learned, ontology 없음 |
| **ToolRM** | 2510.26167 (2025-10) | rule-based verifier (tool call 정확성) | general tool-use | tool-use *결과* 평가; relation ontology 없음 |
| **ToolPRMBench** | 2601.12294 (2026-01) | PRM benchmark | – | 메타 benchmark, 우리 reward 정의 평가 가능 |
| **Web-Shepherd** | (2025) | web-agent trajectory filter | web agents | web 특화, ontology 없음 |
| **AgentR / One Model to Critique** (OpenReview) | 2025 | critic via efficient reasoning | agent tool-use | critic model, 우리는 rule-based |
| **RLTR** (RL with Tool-use Rewards) | (2025) | tool-use completeness | tool-use | completeness ≠ ontology violation |

#### 9.4.5.3 Graph/Structure-grounded Agent RL (가장 인접, v1.13 → v1.14 갱신)

**가장 인접 prior** (v1.14 → v1.15 정정):

> **Graduated Rewards** (Jiayang et al., arXiv 2603.24709, 2026-04)
> "Training LLMs for Multi-Step Tool Orchestration with Constrained Data Synthesis and Graduated Rewards"
>
> 핵심 메커니즘: workflow template + **단일 type dependency graph**로 R_atomic + R_orch reward.
> Edge (j,i) ∈ E ≡ "θ_i depends on observation o_j" (data flow + ordering combined).
> Reward gating: `1[μ(j) < μ(i)]` multiplicative (uniform penalty per edge).
>
> **★ 우리와의 *질적* 차이 — 단일 vs 42 relation type** (v1.15 강조):
>
> | 측면 | Jiayang | 우리 |
> |---|---|---|
> | **Relation type 수** | **1개** (data flow + ordering 합쳐진 단일 edge) | **42개** (16 directional + 3 symmetric + 4 conditional + 19 categorical) |
> | mutex (동시 호출 불가) | ❌ 표현 불가 | ✅ |
> | guardrail (호출 금지) | ❌ 표현 불가 | ✅ |
> | validates (결과 검증) | ❌ (parameter_feeds로 환원 안 됨) | ✅ |
> | conditional_on (context 조건부) | ❌ | ✅ |
> | retry_after_fail | ❌ | ✅ |
> | compensates (효과 역전) | ❌ | ✅ |
> | fan_out / backtrack_to (GoT/ToT) | ❌ | ✅ |
> | Reward 가중치 차등 | 동일 페널티 per edge | precedes −0.1 / requires −0.2 / mutex −0.3 / guardrail −0.5 *관계별 차등* |
> | Geometry-aware 개입 | ❌ (단일 type) | ✅ directional→A6 rotation, symmetric→T1 additive (Phase 0 v3 검증) |
> | Layer 시그니처 | ❌ | ✅ L01-L18 (next_tool) vs L02-L05 (ordering) differential |
> | Naming | "dependency graph" / "workflow template" | "ontology" + relation 어휘 |
> | Discovery | manually curated per-task | AFOD auto-extract from schema |
> | Layer 적용 | RFT reward 1-layer | 4-layer (probing/steering/cross-attn/RFT) |
> | 출간 | 2026-04 (1개월 prior) | 2026-05+ |
>
> **결론**: Jiayang의 dependency graph는 우리 42-relation ontology의 **1-2개 relation type 정도의 부분집합** (parameter_feeds ∪ precedes 합쳐진 형태). Mutex / guardrail / conditional / validation / retry 등 모두 표현 불가. 즉 Jiayang은 **dataflow language**에 가깝고, 우리는 **planning-theory semantic predicates** 통합.

| 논문 | 연도/arXiv | Structure 사용 | Reward 신호 | τ²-bench? |
|---|---|---|---|---|
| ★★★ **Graduated Rewards** (Jiayang) | 2603.24709 (2026-04) | **workflow template + dependency graph** (가장 직접 prior) | R_atomic + R_orch (sequencing check) | – |
| **GAP** (Graph-Based Agent Planning) | 2510.25320 (2025-10) | 학습된 dependency graph per task | **outcome correctness only** (process 아님) | ❌ (MHQA) |
| **CM2** (Checklist Rewards multi-turn) | 2602.12268 (2026-02) | per-turn binary checklist (task별) | **checklist reward (per-turn)** | **✅ +8pt** |
| **StepTool** | 2410.07745 (2024-10, CIKM'25) | step-grained reward shaping | invocation success + task contribution | – |
| **PORTool** (Rewarded Tree) | 2510.26020 (2025-10) | tree-based reward | reward tree | – |
| **Rewarding Graph Reasoning** | 2503.00845 (2025-03) | graph reasoning step | process reward | – |
| **Agent-R1** (E2E RL agents) | 2511.14460 (2025-11) | end-to-end RL framework | – | – |
| **OPRL (Online PRM)** | 2509.19199 (2025-09) | implicit step reward from trajectory pref | online PRM | – |
| **TRM (Tool-call Reward Model)** | OpenReview LnBEASInVr (2025) | tool invocation step PRM | learned PRM | – |
| **Planner-R1** (Reward Shaping smaller LLM) | 2509.25779 (2025-09) | trajectory-level + constraint MDP | dense process + sparse outcome | – |
| **STEP-LLM** | – | step-level reward shaping | per-step | tool orchestration |
| **MemReward** (Graph-based experience memory) | 2603.19310 (2026-03) | graph relational reward | learned reward, limited labels | – |
| **DynaSearcher** | 2507.17365 (2025-07) | dynamic KG augmented | multi-reward RL | – |
| **Graph-RFT** | (2025) | 2-stage RFT, KG + web | RFT verifier | – |
| **Tool Graph Retriever** | 2508.05152 (2025-08) | dependency graph (retrieval) | retrieval-only | – |
| **Plan-RewardBench** | 2604.08178 | trajectory-level judge | reward model eval | – |
| **Plan Then Retrieve** | 2510.20691 (2025-10) | KG reasoning + RL | KG-grounded | – |
| **ToolRL / ToRL** | (2025) | systematic reward design study | reward design framework | – |
| **PRMP framework** | 2502.10325 (2025-02) | PRM for agents — practical directions | PRM survey | – |

#### 9.4.5.4 PDDL/Symbolic Reward Shaping (LLM + 형식 logic, v1.13 신규)

| 논문 | 방식 | 우리와의 차별 |
|---|---|---|
| **LLM-Guided Reward Shaping with PDDL** | LLM이 PDDL plan 생성 → potential-based shaping | PDDL = 사람 도메인 정의 / 우리 AFOD auto-extract |
| **VAL-integrated training** | PDDL plan validity check를 optimization loop에 포함 | validity check만, multi-layer 없음 |
| **"Encouraging Good Processes"** | arXiv 2508.19598 (2025-08) | process reward (정답 불요) | 우리와 비슷한 motivation, ontology는 없음 |
| **Generalization Gap in LLM Planning** | arXiv 2601.14456 | verifier-reward RL | math/code, ontology 없음 |

#### 9.4.5.5 Ontology-Driven RL — 비-LLM Lineage (v1.13 참고)

LLM 이전부터 ontology + RL은 robotics/scheduling에 존재. 우리 진영의 *원격* 선조:

| 논문 | 도메인 | Ontology 역할 |
|---|---|---|
| Du Plessis (2021) "Using ontology to guide RL agents in unseen situations" | RL transfer | ontology = action concept 일반화 |
| Sustainable Manufacturing (MDPI 2024) | job shop scheduling | Reward Machine + ontology (low→high concept mapping) |
| Personalized Student Support (arXiv 2407.10332) | edu MARL | ontology semantic organization + MARL |
| Ontology-Guided DRL for Robotic Tasks (HAL 05266672, 2025) | robotics | interpretable DRL + ontology |

이 계열은 **LLM 아님** — 그러나 "ontology를 reward source의 구조적 prior로 사용"하는 *철학*은 동일. 우리가 LLM 영역에 처음 적용.

#### 9.4.5.6 우리와의 *정확* 분리 — 4×4 차별 매트릭스 (v1.13 강화)

각 prior가 우리 4가지 novelty 중 *몇 개*를 cover하는가:

| Prior | (1) Inter-tool ontology 명시 | (2) Multi-layer injection (probing/steering/CA/RFT) | (3) Auto-discoverable (AFOD) | (4) τ²-bench multi-turn |
|---|:---:|:---:|:---:|:---:|
| Lightman 2023 (PRM) | ❌ | ❌ | ❌ | ❌ |
| MathShepherd | ❌ | ❌ | ❌ | ❌ |
| **GAP** | △ (learned graph) | ❌ | ❌ | ❌ |
| **CM2** | ❌ (checklist) | ❌ | ❌ | **✅** |
| **AgentPRM** | ❌ (TD-value) | ❌ | ❌ | ❌ |
| ToolRM | ❌ | ❌ | △ (rule-based) | △ |
| Planner-R1 | △ (constraint MDP) | ❌ | ❌ | ❌ |
| STEP-LLM | ❌ | ❌ | ❌ | △ |
| PDDL+RL | ✅ (PDDL = ontology) | ❌ | ❌ (사람 정의) | ❌ |
| **Routine** | ❌ (variable memory) | ❌ | ❌ | ❌ |
| Ontology-DRL (non-LLM) | ✅ | ❌ | ❌ | ❌ |
| **우리** | **✅ 42-relation** | **✅ 4 layer** | **✅ AFOD** | **✅** |

→ **개별 차원에 대해서는 prior 풍부**. 그러나 **4 차원 모두 cover하는 work 없음**. CM2는 (4)만, PDDL+RL은 (1)만, GAP는 (1) 부분만.

#### 9.4.5.7 우리 contribution 정확 재정의 (Claim 6 강화)

```
주장:
  "우리는 process reward를 발명하지 않았다 (Lightman 2023부터 다수 선례).
   우리는 ontology RL을 발명하지 않았다 (PDDL+RL, robotics에 prior).
   우리는 graph-based agent planning을 발명하지 않았다 (GAP 2510 이 5개월 먼저).
   
   우리가 *처음*인 것:
   (i) Multi-turn enterprise tool-use에서 4-layer ontology injection 
       (probing/steering/cross-attn/RFT reward) 일관 적용
   (ii) Auto-discoverable 42-relation ontology (AFOD) — Routine처럼 사람 작성 불요
   (iii) Compositional ablation matrix (T1 × T2 × T4, 8-cell) 정량 측정
   (iv) Training-free 단독 동작 옵션 (Phase 2a T1@base) — CM2/GAP/AgentPRM 모두 학습 필수"
```

→ Reviewer가 "X paper에서 했다" 비판 시 위 4 차원 어느 것을 cover하는지 매트릭스로 응답.

#### 9.4.5.8 즉시 baseline 추가 권고

v2 base 결과 이후 다음 baseline을 추가 측정 권장 (시간 무제한 가정):

| 우선순위 | Baseline | 진입 조건 |
|---|---|---|
| **B6 (신규)** | **CM2 (Checklist RL)** — 우리와 같은 τ²-bench, 가장 직접 경쟁 | 즉시 |
| **B7 (신규)** | **Jiayang Graduated Rewards** — single-relation dependency reward 직접 비교 | 즉시 (v1.15) |
| B8 (조건부) | GAP — graph-based + outcome reward | 6-week sprint 외 시간 있으면 |
| B9 (장기) | AgentPRM (learned PRM with TD/GAE) | Phase 4 후 비교용 |

이 baseline들이 *우리 main result*를 강화하지 못하면 contribution 약화. 강화하면 매우 strong.

#### 9.4.5.9 왜 이 영역이 *방금* 폭발하는가 — 학계 dynamics 분석 (v1.16 신규)

"Ontology를 process reward로 사용한다"는 *쉬운 아이디어*가 왜 2026까지 안 됐는지의 7가지 구조적 원인:

**(1) Community fragmentation** — 4 communities가 만나지 않음:
```
Math reasoning RFT (NeurIPS/ICLR ML)    : Lightman, MathShepherd, DeepSeek
Tool-use SFT (ACL/EMNLP NLP)            : ToolFormer, Gorilla, Routine
KG/Ontology + LLM (ISWC/WWW Semantic)   : K-BERT, GraphRAG, GMT
Planning AI (ICAPS/AAAI)                : PDDL, HTN, BPMN
```
각 community 컨퍼런스 별개. PRM 연구자가 PDDL 모르고, semantic web 연구자가 GRPO 모름. 우리 = **4 community synthesis**.

**(2) Benchmark immaturity** — Multi-turn tool-use benchmark가 *최근*:
| Benchmark | 출간 | 나이 (2026-05 기준) |
|---|---|---|
| MATH | 2021 | 6년 (RFT testbed로 성숙) |
| HumanEval | 2021 | 6년 |
| BFCL multi-turn | 2024-02 | 2년 |
| **τ²-bench** | 2024-12 | **18개월** |
| τ³-bench | 2025-11 | 6개월 |
| CM2 τ²-bench RL | 2026-02 | **3개월** |

→ Multi-turn tool-use RL은 *2025 후반-2026*에야 본격 시작. 1년 전엔 testbed가 없었음.

**(3) Verifier 자동화 어려움**:
| 영역 | Verifier | 난이도 |
|---|---|---|
| Math step | SymPy, Wolfram | trivial |
| Code | unit test | trivial |
| Reasoning CoT | learned PRM (Lightman) | medium |
| **Tool ontology violation** | ??? | **hard** |

τ²-bench는 *최종 DB state match*만 검증 — process는 안 봄. 자동 ontology 위반 verifier 작성이 비자명. 우리 AFOD가 *schema → ontology* 자동 추출 → enabling.

**(4) Tool ontology의 학계 부재**:
- OWL/RDF: 너무 일반화
- PDDL: 학계에 있지만 enterprise tool 적용 없음
- WSDL/OpenAPI: API spec — 관계 없음
- BPMN: 사람 작성 워크플로우
- 우리 42-relation = **9 prior 통합** (Routine+GAP+PDDL+BPMN+KnowAgent+GoT+ToT+HTN+GoalAct)

**(5) RFT compute가 *최근*에야 합리적**:
```
PPO (2017)          매우 비싸 (RM + Value)
RLHF + PPO (2022)   InstructGPT 비용 막대
DPO (2023-05)       offline, 비용 ↓
GRPO (DeepSeek 2024) value model 제거
RFT API (OpenAI 2024-12) 대중화
```
GRPO 이전엔 multi-turn RFT *현실적이지 않음*. 2025년에야 가능.

**(6) Cross-disciplinary 언어 장벽**:
같은 개념을 5개 다른 단어로 부름:
| Community | "관계" 명명 |
|---|---|
| Semantic Web | RDF triple, OWL property |
| PDDL planning | precondition, effect |
| BPMN process mining | flow, gateway |
| NLP KG | relation, edge type |
| AgentRL | dependency, action constraint |

→ 통합 표 작성한 첫 work = 우리.

**(7) Enabling conditions 2024-2026 동시 도착**:
```
2023 — PRM (Lightman), Steering Vector mainstream, LoRA/PEFT
2024 — GRPO, BFCL, τ²-bench, ToolFormer-based tool agents
2025 — Routine, StepTool, Planner-R1, GAP, OPRL, ToolRM, AgentPRM
2026 — CM2 τ²-bench RL, Jiayang dependency reward, ★ 우리
```
**2026 = 이 영역 폭발 시점**. 1년 전엔 building block 없었음.

#### 9.4.5.10 "왜 *우리 정확한 형태*가 안 됐는가" — 4가지 추가 이유

학계가 "쉬운 아이디어 = ontology + reward"는 시작했지만 *우리 정확한 형태* (42-relation × multi-layer × AFOD × compositional)는 안 한 이유:

**(a) Single-relation의 함정**
대부분 prior는 "dependency = 1 type" 가정 (Jiayang). 이유:
- KG community: triple (s,r,o) but r은 *학습된 embedding*, 명시적 분류 안 함
- Workflow community: edge에 type 없음 (flow만)
- 학계 인지: "관계 세분화 → sparsity 증가" practical 우려
→ 우리는 *세분화가 lift를 만든다*는 가설 검증 (Phase 0 v3 probing이 증거).

**(b) Probing → ontology 발견 pipeline 부재**
대부분 prior는 ontology를 *bottom-up RL 학습* (GAP) 또는 *top-down 사람 정의* (Routine, PDDL).
**우리 Phase 0 v3 = LLM 내부 representation에서 ontology *발견***.
이건 mechanistic interpretability + ontology engineering 결합 — 매우 희소한 skill set.

**(c) Multi-layer injection의 engineering 부담**
4 layer 각각 구현 필요:
- Probing: sklearn LogisticRegression on hidden states
- Steering: PyTorch hook + residual modification
- Cross-attn: 새 module + LoRA 학습
- RFT: GRPO + custom reward + vLLM rollout

이 4개를 모두 하려면 *PyTorch + HF + vLLM + RL framework + interp* 모두 능숙해야.
단일 community 연구자가 안 함.

**(d) Patent + 학계의 분리**
우리 OISA Patent (2026-04, 사용자 단독 발명)에 이미 ontology + multi-layer 개념 포함.
그러나 patent는 학계 review 안 받음. 학계는 patent cite하지만 baseline으로 안 함.
→ 학계 work가 우리 patent를 모르고 *각자 부분 영역*만 진행.
→ 학계 work 통합 → 우리 patent의 *부분 재발명*.
→ 우리 paper는 *통합본*으로서 contribution.

#### 9.4.5.11 Publication timing 위험 (긴급)

```
2026-05 현재
  우리 Phase 0 ✅, Phase 1 base 진행 중
  Jiayang 1개월 prior (single-relation dependency)
  CM2 3개월 prior (checklist τ²-bench)
  StepTool 19개월 prior (step-grained tool reward)
  
2026-08~2026-12 (예측):
  유사 work *추가 발표 가능성 매우 큼*
  Multi-relation ontology + multi-layer RFT 누군가 시도 가능
  Compositional ablation도 시간 문제 (8-cell matrix는 명시적 아이디어)
  
→ **6-12개월 내 publication critical.**
  우리 sprint 6주 → ICLR 2027 또는 NeurIPS 2026 submit 가능.
  Patent v5 (2026-04 작성)는 이미 prior date 확보.
```

**구체 대응**:
- Phase 1 v2 완료 (예상 2026-05-27 18 KST): cross-model baseline 확보
- Phase 2a 즉시 진입 (training-free contribution 최소 형태로 *먼저* 보고)
- ArXiv preprint를 NeurIPS 2026 deadline (6월말) 전에 *workshop track*에라도 우선 등록
- Patent v5 → preprint cross-reference 추가

#### 우리와의 정확한 비교

```
공통:
  - outcome reward (sparse, 마지막에 한 번)
  - + per-step/turn intermediate reward (dense)
  - SFT 모델 → reward shaping → RL update

차별:
  (1) 도메인:
      PRM 선례   = math step, code execution, reasoning CoT
      우리       = multi-turn tool-use agent (τ²-bench)
      
  (2) Reward source:
      PRM 선례   = (a) 학습된 PRM (Lightman, MathShepherd)
                  (b) execution verifier (CodeRL)
                  (c) MCTS rollout (ReST-MCTS*)
      우리       = ontology relation rule-based:
                    precedes 위반 → −0.1
                    requires 위반 → −0.2
                    mutex 위반    → −0.3
                  학습 PRM 불요, MCTS 불요, 단순 rule
                  
  (3) Reward 적용 위치:
      PRM 선례   = process reward 단독 (학습 신호로만)
      우리       = ontology가 4 layer에 동시 적용:
                    probing (Phase 0) → steering (Phase 2) →
                    cross-attn (Phase 3) → RFT reward (Phase 4)
                  process reward는 그 중 하나의 application
                  
  (4) Reward source의 origin:
      PRM 선례   = 사람 라벨 (math step verify) 또는 외부 verifier (test cases)
      우리       = tool schema에서 auto-discovered (AFOD)
                  → 새 enterprise domain에 *zero-label transfer*
```

#### Risk — 우리 contribution은 *PRM의 application*인가 *novel mechanism*인가?

Pessimistic reviewer 관점:
> "PRM은 Lightman 2023부터 알려진 표준. Routine/RL training에 rule-based reward 추가도 prior 있음. 너희 novelty는 단지 'ontology relation을 rule로 변환'한 것뿐?"

우리 응답 (Claim 6):
> "Yes, isolated process reward는 prior. 그러나 단일 ontology를 *동시에 4 layer (Q-side steering, KV-side cross-attn, weight RFT, output probing)* 에 적용하는 setup은 학계 prior 없음. PRM 선례는 모두 *학습 신호*에만 ontology를 사용. 우리는 *representation injection 도구*로도 사용 — 이게 §3.3.2의 3-lever orthogonality 가설을 검증 가능한 매트릭스로 만듦."

### 9.5 계열 5: Agent Planning 계열 — 에이전트 아키텍처

> **공통 특징**: Test-time 탐색/정제로 계획 품질 향상  
> **공통 한계**: 계획 탐색에 구조적 사전 지식(온톨로지) 활용 없음

| 방법 | 연도 | 탐색 전략 | 구조 지식 활용 | 한계 |
|---|---|---|---|---|
| **ReAct** (Yao et al.) | 2023 | Reason+Act 교차 선형 | ❌ | 오류 복구 없음, 순차 한계 |
| **Reflexion** (Shinn et al.) | 2023 | 실패 → verbal reflection → 재시도 | ❌ | 반복 시도 필요, 구조 없음 |
| **Tree of Thoughts (ToT)** (Yao et al.) | 2023 | BFS/DFS 다중 경로 | ❌ | LLM self-eval 불안정 |
| **LATS** (Zhou et al.) | 2023 | MCTS + 환경 피드백 + reflection | ❌ | 온톨로지 제약 미활용 |
| **RAP** (Hao et al.) | 2023 | MCTS + LLM-as-world-model | ❌ | world model 추정 오류 |
| **ToolLLM/DFSDT** (Qin et al.) | 2023 | DFS Decision Tree + 백트래킹 | ❌ | 16K API 대상, 구조 없음 |
| **Tree-Planner** | 2023 | 실행 전 plan tree 선생성 | ❌ | 정적 트리, 피드백 없음 |
| **KG-Agent** | 2025 | KG 기반 자율 추론 도구박스 | ✅ (부분) | KG 탐색, 온톨로지 제약 없음 |
| **Blocksworld with MCP** (B-13) | 2025 | 고전 AI 계획 + MCP 도구 | ✅ (명시적 순서) | 단순 도메인, 실세계 복잡도 낮음 |

```
우리의 차별점 vs 계열 5:
  LATS/MCTS: τ²-bench simulator를 reward로 사용 가능 (T4 조건과 결합)
             but 온톨로지 제약 없이 blind 탐색 → 비효율
  
  우리 T4 = LATS + 온톨로지 제약 Prior
    - precedes/mutex 관계 → 탐색 가지치기
    - workflow_role → 탐색 우선순위
    - 기존 LATS 대비 rollout 수 동일 조건에서 pass^1 향상 주장 가능

  Blocksworld MCP:
    - 순서 제약이 완전 명시적 → 우리 방법의 단위 테스트 환경
    - 간단한 도메인에서 먼저 검증 → τ²-bench로 확장하는 전략 유효
```

---

### 9.6 연구 공간 지형도 (2차원)

```
          [구조적 지식 활용도]
               낮음 ◄──────────────────────► 높음
               │                              │
높음  ┌─────────┼──────────────────────────────┼──────┐
│     │ Reflexion│                        GMT  │      │
학    │ LATS     │                       FLAME │      │
습    │ ToT      │                      K-BERT │      │
비    │          │                             │      │
용    ├─────────────────────────────────────────────── ─│
낮음  │  ReAct   │  KnowAgent(텍스트)           │  🎯  │
│     │  RepE    │  GraphRAG 계열               │  우리  │
│     │  CAA     │  Steering(추론)              │      │
└─────┴──────────┴──────────────────────────────┴──────┘
      [학습 없음]                               [학습 없음]

🎯 = 우리 연구의 목표 위치:
  - 구조적 지식 활용도 높음 (온톨로지 관계)
  - 학습 비용 낮음 (training-free)
  - 이 사분면에 존재하는 선행 연구 없음
```

---

### 9.7 논문에서의 Related Work 구성 제안

```
§2.1 Knowledge Graph Integration in LLMs
  → 계열 1 (GraphRAG 텍스트 직렬화) + 계열 2 (GMT 직접 주입)
  → 우리: "계열 2의 메커니즘을 계획 수립 태스크에 최초 적용"

§2.2 Activation Steering and Representation Engineering
  → 계열 3 (Steering 전체)
  → 우리: "단일 속성이 아닌 이항 관계 구조를 steering source로 사용"

§2.3 Structured Planning for Tool-Use Agents
  → 계열 4 (Routine/GAP/KnowAgent) + 계열 5 (ReAct/LATS 등)
  → 우리: "구조화 계획의 효과(Routine)를 training-free로 달성"

§2.4 Benchmarks
  → τ²-bench, TPS-Bench, FlowBench, Blocksworld MCP 등
  → 우리가 새로운 통합 평가 프로토콜 제안
```

---

### 9.8 논문 제목 후보 (확장)

```
Option A (방법 × 적용):
  "Ontology-Injected Cross-Attention for Training-Free
   Multi-Step Tool Planning in Enterprise LLM Agents"

Option B (결과 × 비교):
  "Structure Without Training: Direct Graph Ontology Injection
   Outperforms Text Serialization in Agent Tool Planning"

Option C (위치 선점):
  "Beyond Text Serialization: Ontology-Grounded Activation Injection
   for Constraint-Aware Enterprise Tool Planning"

Option D (간결):
  "OntoPlan: Training-Free Ontology Injection for
   Multi-Step Tool-Use Planning"

추천: Option C 또는 D
  - "Beyond Text Serialization": GraphRAG 계열 전체를 정면으로 공략
  - "Ontology-Grounded": 계열 2/3와의 차별점 명확
  - "Constraint-Aware": 계열 4/5와의 차별점 명확
```

---

## 10. Go/No-Go 기준

**전략적 gating (v1.20 강화)** — 우리 main contribution = *메커니즘*. Base model 확장은 *capability 시연용*. 작은 모델에서 메커니즘 작동 안 하면 큰 모델 의미 없음.

| 단계 | 기준 | 판단 |
|---|---|---|
| Phase 0 완료 후 | Probing accuracy ≥ 70% at ≥1 layer | 진행 / 방향 재검토 ✅ 통과 (2026-05-27) |
| Phase 1 v2 완료 후 | Cross-model fair baseline (gpt-4o-mini user_sim) 확보 | Phase 2a 진행 (in progress, ~11 KST 완료) |
| **★ Phase 2a 후 (v1.20 gating)** | T1 steering lift ≥ +3%p on Hermes-3/Qwen-7B | **No (2026-05-29): Qwen 상수 single-relation steering(validates/error_fallback/retry) + gate(decay/orth) 전부 noise band [0.176-0.217], N≤120 → 방법 재검토 → LoRA-RFT 피벗 (§12 v1.21)** |
| **★ Phase 2b 후** | **T3 (T2+T1) compositional lift ≥ +3%p over T2** | **Yes → Phase 4 RFT / No → T2 단독** |
| **★ Phase 4 후** | **T4-RFT vs B3: +10%p on τ²-bench** | **Yes → 32B/70B 확장 시작 (Strategy C/D)** |
| Phase 2a 완료 후 | T1@base vs B2: +3%p on τ²-bench telecom | 진행 / Steering 포기, Cross-Attn 집중 |
| Phase 3 완료 후 | T2 vs B2: +5%p on τ²-bench ≥2 domains | 논문 제출 / 추가 실험 |
| Phase 2b 완료 후 (v1.11 신규) | T3 = T2+T1@T2 vs T2: +3%p | Compositional A 확정 / Phase 4 단순 진입 |
| Phase 4 완료 후 | T4-RFT vs B3: +10%p on τ²-bench | NeurIPS 목표 유지 / EMNLP 하향 |
| Phase 2c 진입 (v1.11 신규) | C1: T4-RFT ≥ B0+10%p (=0.144); C2: v_T4 separability > base × 1.5 | 둘 다 충족 → Phase 2c 진입 |
| Phase 2c 완료 후 (v1.11 신규) | T5 = T4+T1@T4 vs T4: +3%p | Compositional B 확정 (명제 C' 정량) |
| Phase 5 T6 진입 (v1.11 신규) | C3-C6 모두 통과 (T2/T3/T5/T4-RFT 각 임계치) | Triple 측정 / 단순 모델 보고 |
| Phase 5 T6 완료 후 (v1.11 신규) | T6 vs max(T3, T5): +2%p | Pareto upper bound 정당화 / T6 효용 없음 보고 |

**MMS-specific Go/No-Go** (chain-stratified, v1.10 도입):
- Phase 2a: T1@base mms chain 2-4 ≥ +5%p → 가설 B (LRH 적용) 확정, 추가 학습 불필요
- Phase 2c: T5 mms chain 2-4 ≥ T4-RFT +5%p → steering이 RFT 위에서도 boost 가능
- Phase 4: T4-RFT mms 전체 = 0% 유지 → 가설 A 확정 (pure weight gap, 더 큰 모델 또는 RFT 데이터 부족)

---

## 11. 리소스 및 환경

```
서버:       woori@61.33.35.153 (mais1234)
GPU:        cuda:0
Python:     /home/woori/venvs/seka_env/bin/python3.12
Repo:       ~/workspace_common/boltzmann-attention-pi
Branch:     facet-rft-2026
τ²-bench:   ~/workspace_common/boltzmann-attention/external/tau2-bench
Output:     ~/workspace_common/boltzmann-attention-pi/reports/facet_rft_2026/

신규 디렉토리 (생성 필요):
  reports/facet_rft_2026/phase0_probing/
  reports/facet_rft_2026/phase1_baseline/
  reports/facet_rft_2026/phase2_steering/
  reports/facet_rft_2026/phase3_crossattn/
  reports/facet_rft_2026/phase4_hybrid/
  reports/facet_rft_2026/phase5_generalization/
```

---

## 12. 변경 이력

| 날짜 | 내용 |
|---|---|
| 2026-05-29 | v1.23: **Facet-guided distillation lever 추가**. 7B 블로커(H2 capability ceiling + self-GRPO sparse cold-start at 0.18)를 teacher(GPT-4o/Qwen-72B) 성공궤적 distillation으로 우회 → floor(LoRA-RFT) lift의 실질 enabler. 형태: teacher 궤적 → ontology-violation reward 필터/가중 → student LoRA-SFT (=T4-RFT rejection-SFT의 teacher 버전, Phase 4 β 변형). ★confound 격리 필수: plain distill(teacher 복제)≠온톨로지 기여 → unfiltered vs facet-filtered ablation으로 marginal value 격리. 2단계: distill(capability)→facet-RFT(온톨로지 정제). 경제: distill-once→4도메인 zero-shot(합성-북극성 실질 수단, RL보다 쌈). Tier-4 API 모델을 teacher로 재활용. 상세: phase2_steering/STEERING_CONTROL_DESIGN.md §11. |
| 2026-05-29 | v1.22: **Cross-domain 전이 축 격상 + 합성-온톨로지 학습 북극성**. 학습비용=도메인당×N → 온톨로지 관계가 도메인-일반이면 1회 학습 후 전이로 amortize(thesis 경제 정당성·novelty). Phase 5에 묻힌 도메인 일반화를 *중심 축*으로 격상. 4 도메인(telecom 2285/retail 114/airline 50/banking_knowledge 97) cross-domain 전이: B0_d / SYN→d(합성 1회학습 zero-shot, 최강주장) / TEL→d(전이) / d-RFT(상한). 전이 주장 3분리: 스키마(auto)·개입(prize)·reward(이미 rule-based 도메인일반=공짜). 일반성 gradient + de-risk floor(in-domain real RFT). 선행: retail/airline/banking B0·banking AFOD 추출·합성 agentic 학습원 설계(성패 핵심). Go/No-Go 강화: 재학습 없이 held-out +X%p. 상세: phase2_steering/STEERING_CONTROL_DESIGN.md §10. |
| 2026-05-29 | v1.21: **Phase 2a/2c/2d steering 실측 + LoRA-RFT 피벗**. Qwen-7B 상수 single-relation steering(validates +1.5%p / error_fallback / retry_after_fail) + context-gating(decay/orth) 전부 baseline noise band [0.176-0.217] 내 (N≤120). **표상-공간 facet 상보 구조(AXIS-1)가 인과 행동으로 이어지지 않음** — retry_after_fail(EXEC 극)이 transfer(45%)·pass 최고로 예측 반전(『표상 상보 ≠ 인과 상보』). 효과 비특이적, pass^1 미반영(H2 ceiling 지지). C5 actuator 약함·C3 미지지 → Phase 2a Go/No-Go 미통과(<+3%p). **결정: class-hierarchy(상수 steering=bias-1step-RFT 최약점) 처방대로 학습 끝으로 피벗 → LoRA-RFT(L0=PEFT-RFT) 직행** (power test 생략, 사용자 결정). 신규 companion 문서 phase2_steering/STEERING_CONTROL_DESIGN.md: steering↔RFT 수학적 동치(상수=bias-1step-RFT, 가변 LoRA/steering ⊇ RFT), 실시간 steering 고유장점(closed-loop), 온톨로지 활용 taxonomy(합성/Read=RFT보상/ablation/그래프구동/training-time), 상보-전환 규칙, C1-C5 검증링크, 실험 사다리 Rung1-7. |
| 2026-05-18 | D0 가정 수정 완료 (가)(나)(마), Telecom S1 확인 |
| 2026-05-24 | 문제 재정의: 도구 선택 F1 → 다단계 계획 pass^1 |
| 2026-05-24 | 벤치마크 재정의: τ²-bench + TPS-Bench + FlowBench |
| 2026-05-25 | 방법론 재정의: facet label 폐기 → 온톨로지 관계 구조 |
| 2026-05-26 | 실험 설계서 v1.0 작성 |
| 2026-05-26 | v1.1: §5 벤치마크 23개 전체 목록 + 우선순위, §6 모델 Tier 1-4 확장 |
| 2026-05-26 | v1.2: §9 포지셔닝 전면 확장 — 5개 선행 연구 계열 전체 (계열1~5, 40+ 논문) + 2차원 지형도 |
| 2026-05-26 | v1.3: Phase 0 v2 결과 반영 — OVERALL PASS 확정. v1 confound 기록. 개입 레이어 전략 확정 (L02-L05 / L01-L18 차등). A6 Per-head Rotation ablation 추가 (§4.3). SEKA rotation 재활용 전략 Phase 2에 반영. |
| 2026-05-26 | v1.4: 온톨로지 확장 (5종→12종) + T1 vs A6 실증 비교 설계. §3.2에 서베이 기반 신규 관계 7종 추가 (Routine/GAP/PDDL/BPMN). 관계별 기하학 분류 및 방법 배정 이론 수립. A7 (Method×Relation Cross) ablation 추가. Method Router 논문 기여 후보로 추가. |
| 2026-05-26 | v1.5: 코드 반영 완료. tau2_telecom_ontology.py v2 (12종 + RELATION_GEOMETRY + PREDICTED_METHOD). generate_contrast_pairs_v3.py (12종 템플릿, 방향성/대칭/범주형 분리). probe_ontology_v3.py (A7 예측 검증 포함). run_probe_v3.sh, check_results_v3.py |
| 2026-05-26 | v1.6: 온톨로지 종합 서베이 반영 — 12종→27종. Routine/GAP/PDDL/BPMN/KnowAgent/KG 전체 서베이. 신규 15종: CAUSAL_LINK, DIRECTLY_FOLLOWS, ERROR_FALLBACK, TOOL_SUBSUMES, AND_JOIN, STATE_TRANSITION, EXCLUSIVE_CHOICE, EFFECT_STATE, DOMAIN_CATEGORY, CHECKPOINT, IDEMPOTENT, REVERSIBLE, MANDATORY_IN_FLOW, OPTIONAL_IN_FLOW, LOOP_CAPABLE. 코드 전면 갱신(ontology v3, pairs v3). |
| 2026-05-27 | v1.8: §3.5 PCLI (Probing-Calibrated Layerwise Intervention) 신규 추가. 증폭-효과 trade-off 해소 원리 정식화: amplification_risk ∝ 1/τ. 계수 자동 교정 수식 (α = BASE_ALPHA × tau_factor / settling). 방법 선택 결정 트리 (early_peak+directional → A6_peak, flat → A8, mid_late → T1_qonly/A8). 레이어 충돌 분석 설계. check_results_v3.py 전면 재작성: curve_pattern 분류, α 교정, 방법 선택, 충돌 탐지, intervention_map.json 자동 저장. Phase 0 출력에 intervention_map.json 추가 (Phase 2 구현 설계도). |
| 2026-05-27 | v1.9: Phase 1 v1 B0 telecom 실측 반영. **pass^1 = 0.0475 (95% CI [0.031, 0.072])**, pass^4 = 0.1316, N=114 trials=4 max_steps=200. §7에 결과 박스(Termination/Category/Persona/병목/leaderboard 위치) 신규 추가. §8.1 예상 테이블에 실측 표시. 핵심 발견: max_steps 33.6% 도달 시 100% reward=0, mms_issue 0/97 user_stop pass, 0 tasks 4/4 통과, Hard persona가 Easy/None보다 높음. v2 (max-model-len=32K, B0+B1+B2 통일 재실행) 진행 중. |
| 2026-05-27 | v1.10: smoke3(chain=1) vs base v1(chain 2-9) 정밀 비교 추가. small ∩ base = 0 (task 공유 없음), 두 split은 본질적으로 다른 difficulty regime. Chain length × category × pass^1 표 추가: chain=2 non-mms 11.8% vs mms 0%, chain=4 non-mms 15.0% vs mms 0% → MMS multi-step deficit 별개 효과. 그러나 chain=1 mms는 smoke3에서 50% 통과 → 가설 A(pure weight gap) 약하게 반박. Phase 2 측정에 chain stratified 분석 + MMS-specific Go/No-Go (+5%p in chain 2-4 = 가설 B 확정, 0% = Phase 4 우선) 추가. |
| 2026-05-27 | v1.11: Compositional lever 합성 8-cell ablation matrix 정식화 (사용자 thesis = 명제 C' 정량). Phase 2 분기: **Phase 2a** (T1@base, training-free) / **Phase 2b** (T3 = T2 + T1@T2, Compositional A) / **Phase 2c 조건부** (T5 = T4-RFT + T1@T4, Compositional B, 명제 C') / **Phase 5 T6 조건부** (Triple = T2 + T4-RFT + T1@(T2+T4), Pareto upper bound). §3.3에 3-lever 직교성 매트릭스 + LRH 곱셈 가설. §4.2 T1 variants notation (T1@base/T2/T4/(T2+T4)). §4.2 T4 분기 (T4-LATS path α / T4-RFT path β). §7 Phase 4에 facet-RFT 세부 (GRPO + ontology violation penalty). §8.1 Pareto frontier ASCII plot. §8.2 Claim 4-5 신규. §10 Go/No-Go에 C1-C6 조건 명시. 학계 prior: Persona Distill, Task Vectors 인용. T6 setting 자체는 prior 없음 — 우리 novel contribution. |
| 2026-05-27 | v1.12: Process Reward Models (PRM) 계열 정직 articulation. §8.2 Claim 6 신규 — process reward 자체는 Lightman 2023부터 알려진 prior, 우리 novelty는 (i) 단일 42-relation ontology를 4 layer (probing/steering/cross-attn/RFT reward)에 일관 적용, (ii) tool schema에서 AFOD auto-discoverable. §9.4.5 PRM 계열 신규 subsection — Lightman/STaR/ReST/MathShepherd/CodeRL/ReST-MCTS*/Quiet-STaR 비교 표 + 우리와의 4가지 차별점 (domain/source/위치/origin). Pessimistic reviewer 선제 대응 framing. **Routine 논문 정정**: arXiv 2507.14447 직접 확인 결과 (a) reward 없음, (b) inter-tool ontology 없음, (c) SFT only (RL 안 함). 이전 메모리의 "SFT+RL 필수" 오류 정정. §9.4 Routine 표 + "6가지 차별" 비교 추가. Routine은 우리 baseline 아니라 complementary direction (사람 routine + SFT vs 자동 ontology + multi-layer injection). |
| 2026-05-27 | v1.13: Lightman 2023 이후 *ontology + process reward* 선행연구 깊은 탐색·정리. §9.4.5 전면 재편 — 5개 sub-subsection: (5.1) PRM foundations (math/code), (5.2) **Agent PRM 신규** (AgentPRM 2511.08325, ToolRM 2510.26167, ToolPRMBench, Web-Shepherd, AgentR, RLTR), (5.3) **Graph/structure agent RL 신규** (GAP 2510.25320 MHQA, **CM2 2602.12268 τ²-bench +8pt** 가장 직접 경쟁, Planner-R1 2509.25779, DynaSearcher 2507.17365, STEP-LLM, Tool Graph Retriever, Plan-RewardBench), (5.4) **PDDL/symbolic 신규** (LLM-Guided PDDL Shaping, VAL-integrated, arXiv 2508.19598 "Encouraging Good Processes", 2601.14456 Generalization Gap), (5.5) **Ontology-driven RL non-LLM 신규** (robotics, scheduling, edu MARL — 우리의 원격 선조). §9.4.5.6 4×4 차별 매트릭스 — 12 prior × 4 차원 (ontology/multi-layer/AFOD/τ²-bench). 결과: 어느 prior도 4 차원 모두 cover 안 함. CM2가 (4)만, PDDL+RL이 (1)만. §8.2 Claim 6 강화: "process reward 발명 안 함, ontology RL 발명 안 함, graph planning 발명 안 함" 정직 인정 + 4차원 동시 cover가 unique. §9.4.5.8 baseline 추가 권고: B6 CM2 (즉시), B7 GAP, B8 AgentPRM. |
| 2026-05-27 | v1.14: 추가 깊은 탐색 → ★★★ **Graduated Rewards (Jiayang et al., 2603.24709, 2026-04)** 발견 — *가장 직접 prior* (1개월 전). Workflow template + dependency graph로 R_atomic + R_orch (sequencing check via `1[μ(j)<μ(i)]` multiplicative gating) 사용. 우리와 차이: (a) "ontology" 명명 안 함 ("workflow template + dependency graph"), (b) manually curated per-task (우리 AFOD auto-extract), (c) reward 1-layer (우리 4-layer). §9.4.5.3 표에 ★★★로 강조 표시 + 추가 prior 9편 (StepTool, PORTool, Agent-R1, OPRL, TRM, MemReward, ToolRL, PRMP, Rewarding Graph Reasoning). §8.2 Claim 6 정직 강화: "온톨로지를 process reward로 사용한 것 우리가 처음 아님 (Jiayang 1개월 prior, dependency graph도 동일 메커니즘). 우리가 처음인 것은 5개 동시 cover: (i) named relation ontology (precedes/requires/...), (ii) cross-domain 42-relation pre-defined, (iii) AFOD auto-discovery, (iv) 4-layer injection, (v) compositional ablation matrix." |
| 2026-05-27 | v1.15: **Jiayang vs 우리 — single relation vs 42 relation 질적 차이 정정**. v1.14에서 Jiayang을 "가장 직접 prior"라고 격상했으나 재확인: Jiayang의 dependency graph는 *단일 edge type* (data flow + ordering combined) — 우리 42-relation ontology의 1-2 type 부분집합 (parameter_feeds ∪ precedes). Jiayang으로 표현 불가능한 관계 27+개: mutex (동시호출 불가), guardrail (호출 금지), conditional_on, validates (검증 ≠ data flow), retry_after_fail, compensates, fan_out / backtrack_to (GoT/ToT), workflow_role / idempotent / reversible 등 unary 속성. §9.4.5.3에 14-row 비교 표 (relation type 수, mutex/guardrail/conditional 등 표현 가능 여부, reward 차등, geometry-aware intervention, layer 시그니처, naming, discovery, layer 적용). §8.2 Claim 6 6-point novelty로 확장: (i) multi-relation ontology, (ii) relation-type-aware reward weighting, (iii) geometry-aware intervention (Phase 0 v3 검증), (iv) cross-domain pre-defined, (v) AFOD, (vi) 4-layer injection. 핵심 framing: "Jiayang = dataflow language, 우리 = planning-theory semantic predicates (PDDL/HTN/GoalAct 통합)". |
| 2026-05-27 | v1.18: **Cross-model fair setup 도입 — Qwen + GPT-4o user_sim 시작**. Llama 0/456 분석에서 *agent=user_sim self-play* 한계 노출 (양쪽 같은 모델 deficit → 100% user_stop 빨리 종료). 해결책: user_sim = GPT-4o 외부 API (tau2-bench leaderboard 표준). `phase1_runner.py`에 `--user-llm` 인자 분리, `_run_phase1_qwen_gpt4o_user.sh` wrapper 신규. Agent = Qwen2.5-7B (local vLLM GPU1:9000), User_sim = openai/gpt-4o (OpenAI API). OPENAI_API_KEY 안전 저장 (`/home/woori/.openai_key` chmod 600), curl verify HTTP 200. 시작 2026-05-27 15:57, 예상 종료 21-23 KST, 비용 ~$60-120. Llama runner+vLLM kill (어차피 tool call 0%이므로 negative result 보존). §7에 v2 vs v3 비교 매트릭스 추가. 가치: (1) self-play 한계 정량화, (2) leaderboard fair 비교, (3) MMS 0% 가설 — user_sim 강해도 안 풀리면 진짜 capability deficit. |
| 2026-05-28 | v1.20: **Strategic gating — Base model 확장 (32B/70B)은 Phase 2-4 lift positive 확인 후**. 핵심 통찰: 우리 main contribution은 *ontology 개입 메커니즘*이지 *base model 크기* 아님. 32B/70B 확장은 *capability 시연* 용 (Pareto frontier 상한선). 그러나 메커니즘 자체가 *작은 모델 (7-8B)에서 작동* 안 하면 큰 모델로 확장 의미 없음. 따라서 우선순위 재조정: (1) Hermes-3 (8B) + Qwen-7B baseline 완성 [in progress, ~11 KST], (2) **Phase 2 T1 steering** on Hermes-3/Qwen-7B with gpt-4o-mini user_sim (cross-model fair), (3) Lift > 0 (≥+3%p) 확인 시에만 Phase 3 (T2 cross-attn LoRA), (4) 누적 lift > 5%p 확인 시 Phase 4 (RFT), (5) **Phase 2-4 framework working evidence 확보 후** 32B/70B 확장 (Strategy C/D). Watchdog B/C (Qwen-32B, Llama-70B 자동 launch) **취소**. 다운로드된 모델은 *보존* — 향후 활용. **결정적 발견 (Phase 1 v2 partial)**: Qwen v2 self-play (Qwen user_sim) = 0.029 vs Qwen+OR cross-model (gpt-4o-mini user_sim) = **0.160 (5.6× lift)**. self-play 평가가 *agent + user_sim 둘 다 약함*에 묶임 — leaderboard 비교에 cross-model 필수. |
| 2026-05-27 | v1.19: **Enterprise Viability Metrics (EVS) 신규 도입**. §5.7.1-4 추가 — pass^1 외 *실용성* 평가 9개 metric: ATR(p95), max_steps_rate, quick_failure_rate, over_diagnosis_rate, productive_resolution_rate, tool_calls_per_success, infra_error_rate, EVS 종합 지표. Phase 1 v2 실측 6 baselines 적용 결과 *모델별 실패 모드 trichotomy* 발견: (a) Llama-8B = Quick-failure 91.7% (즉시 포기, BFCL 70% 약함), (b) Qwen-7B vanilla = Over-diagnosis 54-58% (200 step trap, 종료 인지 부족), (c) Hermes-3 = Balanced (EVS 0.116, Qwen 6.4×, Llama 45×). v1 (16K) vs v2 (32K) Qwen 비교: 32K가 *오히려 EVS 떨어뜨림* (0.029→0.018) — context 늘리면 *더 길게 시도하나 못 풀음*. §8.2 Claim 7 신규: "Pass^1 단독 부족, EVS 추가 측정 필수. 우리 ontology 개입 부가가치 = achieves_goal/plan_committed_to_goal steering이 종료 신호 강화 → max_steps_rate 감소 → enterprise viable agent." |
| 2026-05-27 | v1.18: **Cross-model fair setup — Llama-8B agent + GPT-4o user_sim 시작 (원인 분리 실험)**. Llama 0/456 catastrophic (v1.17) 의 진짜 원인을 (H1) Llama 자체 deficit (vLLM parser 또는 native function-calling 약함) 와 (H2) self-play deficit (agent=user_sim 같은 모델 양쪽 결합)로 분리. 단 한 변수 변경 (user_sim Llama→GPT-4o), Llama agent 동일. 결과 시나리오: 0% 유지 → H1 확정 (negative result strengthening: "8B vanilla = enterprise tool 불가능"), ≥5%p lift → H2 dominant (self-play 한계). `phase1_runner.py`에 `--user-llm` 인자 분리 구현. `_run_phase1_llama_gpt4o_user.sh` wrapper 신규. OPENAI_API_KEY 안전 저장 (`/home/woori/.openai_key` chmod 600), curl verify HTTP 200. Llama vLLM 재가동 (GPU0:9001 max-model-len 32K). 시작 2026-05-27 16:01, pid 3770079. out-dir: `base_n114_llama_gpt4ouser/`. §7에 Phase 1 setup 매트릭스 (v1/v2/Llama-self-play/Llama+GPT-4o 4 row). Qwen v2 (self-play 진행 중) + Llama+GPT-4o 병행. |
| 2026-05-27 | v1.17: **Phase 1 v2 partial (Qwen) + Llama cross-model (B0 완료) 실측 반영**. (1) Qwen v2 partial (346/456 = 76% 진행): pass^1 = 0.026 [0.014, 0.049] vs v1 0.0475 — v2가 약간 낮음 (남은 110 sims hard task일 가능성). max_steps 종료 33.6%→52.9% 증가 (32K로 long task 허용 효과 양면), infra error 0%로 해소. (2) **★ Llama-3.1-8B B0 catastrophic 0.000 (0/456)** — 모든 sim user_stop, **tool calls per sim = 0 (전체 456 sim에서 도구를 단 한 번도 호출 안 함)**. 원인: vLLM llama3_json parser 비호환 + Llama-3.1-8B native tool calling 약함 + user_sim 같은 모델 사용 시 양쪽 deficit 결합. (3) 함의: (a) Llama 결과는 negative result로서 가치 — "8B vanilla function-calling 환경에서 enterprise tool task 사실상 0%", multi-relation ontology 개입 필요성 강화. (b) Cross-model 실험 설계 결함 — user_simulator는 *독립된 strong 모델* (GPT-4o API) 필요. (c) Tier 1 추천 갱신: Qwen2.5-7B primary, Llama-3.1-8B는 reference baseline만, *큰 모델* (Llama-3.3-70B, Qwen2.5-32B) baseline 추가 권장. |
| 2026-05-27 | v1.16: **§9.4.5.9-11 신규 — 학계 dynamics + publication timing 분석**. "쉬운 아이디어가 왜 안 됐는가" 7가지 구조적 원인: (1) Community fragmentation (4 communities — math RFT, tool SFT, KG/ontology, planning AI — 만나지 않음), (2) Benchmark immaturity (τ²-bench 18개월, CM2 τ²-RL 3개월 됨), (3) Verifier 자동화 어려움 (math는 SymPy 단순, ontology violation은 비자명), (4) Tool ontology 학계 부재 (PDDL/OWL/BPMN 등 부분만), (5) RFT compute가 GRPO (2024-말)에야 합리적, (6) Cross-disciplinary 언어 장벽 ("관계" 명명 5개 다름), (7) Enabling conditions 2024-2026 동시 도착. "왜 우리 정확한 형태가 안 됐나" 4가지: (a) Single-relation의 함정 (관계 세분화 → sparsity 우려), (b) Probing → ontology 발견 pipeline 부재 (mech interp + ontology eng 결합 희소), (c) Multi-layer injection engineering 부담 (PyTorch + HF + vLLM + RL framework + interp 모두 필요), (d) Patent + 학계 분리 (OISA patent v4 2026-04 이미 통합본). **§9.4.5.11 publication timing 위험**: 6-12개월 내 publication critical, ICLR 2027 / NeurIPS 2026 workshop submit. ArXiv preprint 6월말 권장. |
| 2026-05-26 | v1.7: 42종 온톨로지 확장 완료 (27→42). Group G: GoT/ToT/Harness 6종 (FAN_OUT, PRUNED_BY, SCORED_PREFERENCE, BACKTRACK_TO, OBSERVATION_TRIGGERS, GUARDRAIL). Group H: HTN 4종 (DECOMPOSES_INTO, SUBTASK_OF, ACHIEVES_GOAL, REFINES). Group I: GoalAct 5종 (PLAN_STEP_PRECEDES, PLAN_STEP_SKILL, PLAN_REVISED_TO, STEP_REALIZES_TOOL, PLAN_COMMITTED_TO_GOAL). GoalAct 수정: 주기적 목표 환기가 아닌 G_t=π(Q|T|S_t) 연속적 플랜 재작성 + 4종 skill 계층. §3.4 수학적 프레임워크 신규 추가: Q-side(T1/A6) vs KV-side(A8) 개입 공간 분류, 프롬프트-동치 정리. A8 실험 신규 추가 (§4.3): KV Cache Steering (arXiv 2507.08799) 온톨로지 관계별 확장. §9.3 KV Cache Steering 논문 추가 및 우리 연구와의 차별점 정리. GOAL_VOCAB(16), PLAN_STEP_VOCAB(12) 어휘 확장 반영. |
