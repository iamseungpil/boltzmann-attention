# E-PLAN live 배선 설계 (plan/execute e2e) — 2026-07-11

> 로컬 편집·repo 커밋. 등대 §1.5 Q2(부하) · `SCAFFOLD_ENDGAME_PLAN §L4·CP5` · [[14-eplan-priority]] 파생.
> **상태: [D] 설계서 (step a). 리뷰(step b) 대기.** 유료 e2e 검증은 승인 후에만.
> 비용: 설계·구현·단위·격리검증 = **무료**(로컬 32B/오프라인). 표적 nt=1 = 소액(승인).

---

## 0. 목적 · 표적 · 불변 가드

**한 줄**: coverage/discovery-부하 class(t95형)를 닫는 유일 레버 = plan/execute 분리 + **discovery-read 강제** + 턴간 **완료-추적 ledger**. write 강제·값 주입은 **절대 금지**([[05]] Q3·§1.5).

**표적 (정본 census)**:
- `SCAFFOLD_ENDGAME §L4`: **MISSED + ZERO_NEV 47 sims (10.3pp)** — 최대 headroom class.
- `FIXABLE_FAIL_CENSUS §1`: **⋈-missed(멀티엔티티 미완) ~11건** = 결정론 fixable 최대 조각.
- **정본 대표 = t95** (`T5C_SILENT_REPAIR_DESIGN §13`·[M]·floor 대조): 2주문 exchange인데 둘째 주문 `#W2905754`를 *조회조차 안 함* → exchange write 자체가 부재. floor는 `get_user_details`→3주문 전수조회로 발견(3/4), 우리 arm은 미조회(0/4·COMP도 0/4).
- retail·**banking 공통**: banking REACH 결손(MISS_P_reach 24~48%·C52) + "완주-후-불일치 45%"(단계별 결과기록 부재·C52) 동일 처방.

**왜 silent-repair로 못 닫나** (사용자 질문·§13 정본):
- silent = write의 *인자값*을 제자리 치환. t95는 *없는 write* → 고칠 인자가 없음.
- 없는 write를 엔진이 생성 = 도메인 행동 수행 = autofetch류 = **[[05]] Q3 위반**(금지).
- ∴ 정답 = write 생성이 아니라 **빠진 주문을 *발견하게* 함** = read 강제(§1.5 **읽기만 강제, 쓰기 절대 금지** 허용).

**진단 등급**: 부하 [M] — C14(격리 계획선 정답·궤적 누락=reach부하) · `PLAN_PROBE_PHASE0_VERDICT §1`(t99 격리계획 2주문 정답, 실런 1주문 누락+날조). §1.5 Q2 yes 경로.

**[[05]] 3질문 (매번)**:
1. 고정=TBox weights+Scaffold 엔진 / 변경=ABox만? → ✅ **controller 로직=도메인일반**(엔진). retail 지식(어떤 도구가 enumerator인지)은 전부 A2/ACTION_SPEC(ABox).
2. 도메인-특화 scaffold 금지? → ✅ discovery-enforce는 A2의 `enumerator_spec`(도메인당 1줄)만 참조·controller는 리터럴 0.
3. 도메인-타깃 학습? → ✅ 학습 0. 순수 결정론 controller + 기존 plan-추출 프롬프트(도메인일반).

---

## 1. 아키텍처 — 3 컴포넌트

```
[CP0 PLAN-EXTRACT]  첫 사용자 요청 확정 후 1회
    plan-spec 생성(모델·도메인일반 프롬프트) → 정규화(기존 controller: batch/status/provenance)
        │  plan-ledger = {planned_writes: [(intent_class, entity_scope, ...)]}
        ▼
[DISCOVERY-ENFORCE]  plan이 참조하는 entity-scope가 미조회면
    read-only enumerator 강제(A2 enumerator_spec: retail=get_user_details)
    → 발견된 전 entity를 에이전트에 표면화(리마인더·값판단 0) → 에이전트 재-plan
        │  ★핵심: plan 완결성은 discovery에 의존. 안 읽으면 안 계획됨(t95 기전).
        ▼
[에이전트 자유 실행]  기존 루프(gated 인터셉터가 실행된 write를 ledger에 관측 기록)
        ▼
[CP5 COVERAGE-WALK]  종결(is_stop) 직전
    planned_writes ⊋ executed_writes 이면
    → 미실행 planned 항목을 리마인더로 재프롬프트(에이전트 자신의 plan·gold 아님·read/write 강제 0)
    → 재확인 후에도 미완이면 통과(강제 없음·harm 회피)
```

**설계 원칙 (FIXABLE §0 재프레이밍·★핵심)**: 32B fail 16 중 **14가 격리 계획선 이미 core_ok·controller 0발화**. ⇒ 이득은 batch/status 정규화가 아니라 **discovery + 완료-추적**에서 나온다. batch/status controller는 14B·부하 시만 발화(보조). **주기능 = walk-reminder + discovery-enforce.**

---

## 2. 배선점 (실제 코드·`tau2/orchestrator/orchestrator.py`)

루프: `run()` → `while not self.done: step(); _check_termination()`. 종결=USER_STOP/AGENT_STOP/max_steps/max_errors.
기존 scaffold = `apply()`가 `BaseOrchestrator.gated(tool_calls)` monkeypatch(CP3 tool-call 인터셉터).

E-PLAN 후킹 (별도 patch·`t2_eplan_patch.py`·기존 gate_patch와 독립 toggle):
| 컴포넌트 | 후크 | 방식 |
|---|---|---|
| CP0 plan-extract | 첫 agent step 직전 (`initialize()` 후·first user msg 확정 후) | orchestrator 인스턴스에 `_eplan_ledger` 부착·1회 plan 생성 |
| discovery-enforce | agent 응답 생성 후·tool 실행 전 (`gated` 진입점 재사용 or step 후크) | enumerator 미호출 ∧ plan-scope 미발견이면 read-only enumerator 주입 |
| ledger 관측 | `gated(tool_calls)` (기존 인터셉터 확장) | 실행된 write tool_call을 `executed_writes`에 기록 |
| CP5 coverage-walk | `is_stop`/`_check_termination` 직전 | 미완 planned 있으면 `self.done` 보류 + 리마인더 UserMessage 주입(1~2회 상한) |

**규칙0 준수**: 주입 메시지는 전부 리마인더(에이전트 자신의 plan 재진술 + "아직 처리 안 한 주문이 있다")·**DB 내용 주입 0**·enumerator는 에이전트가 부를 *도구를 강제*할 뿐 결과를 대신 읽지 않음. (present/autofetch와 차별 = C34 폐기선 안 밟음.)

---

## 3. controller / ledger 결정론 로직 ([[10]])

기존 `plan_execute_orch.py`의 `controller()`(batch-merge·status-fix·provenance-drop) 재사용 + 신규:

```
plan_ledger:
  planned  : [(intent_class, order_id|SCOPE_TOKEN, items)]   # CP0서 정규화된 plan
  executed : [(intent_class, order_id, items)]               # gated서 관측
  discovered_scope : set(order_id)                            # enumerator 결과서 파생

coverage_gap() -> list:
  # planned 중 executed에 매칭 없는 항목 (intent_class + order_id 기준)
  return [p for p in planned if not any(_covers(e, p) for e in executed)]

discovery_needed() -> bool:
  # plan이 "전 주문" 스코프(SCOPE_TOKEN=ALL_PENDING 등)인데 enumerator 미호출
  return has_scope_token(planned) and not enumerator_called
```

- **selector/verifier = 결정론**(controller). **생성기 = LLM**(plan-spec·재-plan). [[10]] 준수.
- `intent_class`·`enumerator_spec`·`SCOPE_TOKEN` 어휘 = 도메인일반. 매핑(retail: `get_user_details`가 enumerator, ALL_PENDING→status=pending 필터)만 A2.
- ★banking 겸용: enumerator_spec = 계좌/절차 목록 도구·SCOPE_TOKEN = 절차 단계 집합. coverage-walk = "gold 절차 median 8단계" 중 미완 단계 추적(C52 horizon/reach 처방).

---

## 4. 반대편 계측 (제1원리·Δ 필수)

| 부작용 | 계측 | GO 조건 |
|---|---|---|
| over-read (불필요 enumerator·턴 낭비) | `_eplan_reads_added` / sim · turn 예산 | Δtme ≤ 0 (too_many_errors 미증) |
| **over-action** (walk가 안 시킨 write 유도) | passing-spurious Δ (vs floor) | **Δspurious ≤ 0** |
| walk-reminder가 멀쩡한 종결 흔듦 (C53 p4형) | 짝 flip census (pass→fail) | robust 상실 ≤ 획득 |
| plan-extract 오염 (틀린 plan을 walk가 강화) | plan pre/post core_ok (오프라인) | plan 정확도 유지 |

**절대선**: coverage-walk는 **읽기만 강제**. 미완 항목을 "해라"가 아니라 "이 주문들 아직 안 봤다" 리마인더 → 에이전트가 판단. abstain→forced-act 전환 금지(§1.5: ⋈서 p≈0.44<0.5 ⇒ 기대-유해). walk가 강제하는 유일한 것 = read.

---

## 5. 테스트 계획 (단계·[[09]] 무료 先)

| 단계 | 내용 | 비용 | 게이트 |
|---|---|---|---|
| (c) 단위 | `test_eplan.py`: ledger/coverage_gap/discovery_needed 순수로직 (tau2-stub·오프라인) | 무료 | ALL PASS |
| (c) 오프라인 replay | 기존 `plan_execute_orch --replay`로 controller 정규화 무회귀 확인 | 무료 | pre/post 무변 |
| (d) 격리 검증 | t95 등 표적의 실 궤적에 discovery-enforce 격리 주입 → 둘째 주문 표면화되나 (32B 로컬·유료런과 GPU 경합 회피) | 무료 | 표적서 enumerator 발화 ∧ scope 발견 |
| (e) 표적 nt=1 | t95+coverage class 소수(≤13) × nt=1 사이클(§0b 프로토콜) | 소액(승인) | per-case 복구 ∧ Δspurious≤0 ∧ Δtme≤0 |
| full | 별도 456 (루프 아키텍처 변경·**합산 금지**·`§CP5`) | 유료(승인) | GO 조건(아래) |

**스모크 필수**([[30]]): full 전 `--num_tasks 10 --num_trials 1`로 3컴포넌트 라이브 발화 검증(마커 stderr). 단위PASS≠라이브발화(calc 31/342 선례).

---

## 6. GO 조건 · 도달 목표

- **GO**: 표적 class per-case 복구(t95형 discovery→2write) ∧ **Δspurious ≤ 0** ∧ Δtme ≤ 0(turn 예산) ∧ 위반0 유지 ∧ 짝 flip 순증.
- **도달 목표**(`ENDGAME §3`): retail 32B R2 후 0.66~0.70 중 E-PLAN 몫 = MISSED+ZERO_NEV 47 sims(10.3pp) headroom의 부분회복.
- 실패 시: 레버 개별 제거(§1.3 죽은레버 등재)·discovery만 살리고 walk 드롭 등 부분채택.

## 7. 미해결 · 리스크

- **R1 plan 오염**: 첫 plan이 틀리면 walk가 틀린 걸 강화. 완화 = plan은 리마인더용일 뿐 강제 아님·에이전트 재판단 여지.
- **R2 discovery 과잉**: enumerator가 무관 entity 대량 표면화 → 창 오염(C43 정박치환 재료 공급 위험!). 완화 = enumerator 결과를 controller가 scope 필터 후 *개수만* 리마인드(id 나열 최소화). **← C43과 직접 긴장·격리서 측정 필수.**
- **R3 종결 지연**: walk 리마인더 상한(1~2회) 없으면 max_steps 낭비. 상한 하드코딩.
- **R4 banking 전이**: enumerator_spec/SCOPE_TOKEN이 banking 절차-집합에 매핑되나 = Phase 3 실측(retail 확정 후).
- **소유권**: E-SPEC(오케스트레이터 재설계)와 CP5 좌석 공유 — E-PLAN은 coverage-walk만·E-SPEC은 전체 재배치. 중복 구현 금지.

---

## 8. 다음 액션 (구현 순서)
1. `t2_eplan_patch.py` 스켈레톤 (ledger 부착·gated 확장 관측·CP5 후크) — 무료.
2. `test_eplan.py` 단위 (coverage_gap·discovery_needed·ledger) — 무료.
3. A2에 `enumerator_spec`(retail) 1줄 추가 + SCOPE_TOKEN 파서.
4. 격리 검증 (t95 궤적 discovery 주입) — 무료·GPU 한가할 때.
5. → 리뷰(step b) 후 표적 nt=1 (승인).
