# E-PLAN live 배선 설계 (plan/execute e2e) — 2026-07-11

> 로컬 편집·repo 커밋. 등대 §1.5 Q2(부하) · `SCAFFOLD_ENDGAME_PLAN §L4·CP5` · [[14-eplan-priority]] 파생.
> **상태: [D] 설계서 v1.2 — 리뷰(step b) 비판적 재해석 완료 2026-07-11. 설계 확정.** 유료 e2e 검증은 승인 후에만.
> v1.1 변경: ①CP5·discovery 주입 채널을 **생성-레벨(히스토리 비커밋)**으로 확정(REPLAY_SAFE 교훈·`t2_gate_patch.py:997` — 합성 메시지 커밋=set_state replay 파손=infra_error 선례) ②§2규칙0↔§7R2 모순 해소 ③coverage_gap SCOPE_TOKEN 확장 로직 추가 ④USER_STOP-후 walk 검증항목 등재.
> **★v1.2 변경(리뷰 자체를 [[08]]·§7.6로 재검증)**: 리뷰 4건 중 ①(인용 코드 실재 직접 확인 ✓ `t2_gate_patch.py:997-1004`)·③·④ 수용 유지. **②(discovery=기존 게이트 precondition·"t95 기전에 정확히 맞물림")는 t95 8-trial 실측으로 부분 반증**:
> (a) `get_user_details`(list-enumerator)는 **8/8 trial 기호출** → "미호출 시 deny" 술어는 t95서 **0회 발화**했을 것.
> (b) COMP tr0/tr2는 **두 gold 주문의 get_order_details까지 다 읽고도 실패**(binding-gap) — 미조회-gap(tr1/tr3: #W2905754 미조회)과 정확히 반반.
> (c) 첫 사용자 발화 8중 7이 단수 "a laptop" — "실은 두 대"는 **대화 중반 계시** → CP0 단발 plan은 이 의무를 구조적으로 못 담음.
> ⇒ ②의 골자(기존 replay-safe deny+regen 재사용·신규 후크 불요)는 유지하되 **술어를 2-수준으로 재설계**(§1) + **CP5 근거를 stop-time 전문맥 재-plan으로 교체**(§1·C14 정합). t95 GO 판정은 **db_match 기준**(reward 단독 금지 — NL-calc 이중결손 병발·`RETAIL_FULL_FAIL_CENSUS §5`).
> 비용: 설계·구현·단위·격리검증 = **무료**(로컬 32B/오프라인). 표적 nt=1 = 소액(승인).

---

## 0. 목적 · 표적 · 불변 가드

**한 줄**: coverage/discovery-부하 class(t95형)를 닫는 유일 레버 = plan/execute 분리 + **discovery-read 강제** + 턴간 **완료-추적 ledger**. write 강제·값 주입은 **절대 금지**([[05]] Q3·§1.5).

**표적 (정본 census)**:
- `SCAFFOLD_ENDGAME §L4`: **MISSED + ZERO_NEV 47 sims (10.3pp)** — 최대 headroom class.
- `FIXABLE_FAIL_CENSUS §1`: **⋈-missed(멀티엔티티 미완) ~11건** = 결정론 fixable 최대 조각.
- **정본 대표 = t95** (`T5C_SILENT_REPAIR_DESIGN §13` + **v1.2 8-trial 전수 재실측**·[M]): 2주문 exchange 미완이나 **실패 형상은 두 가지가 반반** — ⓐ **미조회-gap**(COMP tr1/tr3: `#W2905754` details 미조회·둘째 laptop 미발견) ⓑ **binding-gap**(COMP tr0/tr2: **두 gold 주문 details 다 읽고도** 한 주문에 item_id 중복 시도 = 읽기≠조립). `get_user_details`(목록)는 **8/8 기호출**. floor는 3/4 통과. ⇒ 레버도 두 갈래: ⓐ=detail-read 강제, ⓑ=stop-time 재-plan walk(§1). **주의: t95는 NL-calc(총액) 이중결손**(v25e tr2: 두 write 성공·db=True·reward 0) → **E-PLAN GO 판정은 db_match로**(reward 단독 금지·`RETAIL_FULL_FAIL_CENSUS §5`).
- retail·**banking 공통**: banking REACH 결손(MISS_P_reach 24~48%·C52) + "완주-후-불일치 45%"(단계별 결과기록 부재·C52) 동일 처방.
- **(v1.2·C64 교차) D클래스 미조회-원천분도 L2 표적**: "주소는 내 주문 중 하나에 있다"(t86-tr2 "unable to locate"=정답 미조회 시사) = 미검토 sibling 신호 → L2 detail-read 강제가 관할. 문맥-실재분은 DISAMB-주소(`CENSUS_LEVERS_DESIGN §4`)·분해는 그쪽 V0가 확정.

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
[CP0 PLAN-SEED]  첫 사용자 요청 확정 후 1회 (역할 축소·v1.2)
    plan-spec 생성(모델·도메인일반 프롬프트) → 정규화(기존 controller)
        │  역할 = ledger seed + discovery-enforce 조기 신호.
        │  ★CP5의 근거가 아님(대화 중반 계시 의무["실은 두 대"]를 못 담음·t95 실측 7/8 단수 시작).
        ▼
[DISCOVERY-ENFORCE]  기존 replay-safe deny+regen 게이트에 precondition 추가 (신규 후크 불요·리뷰② 골자 유지)
    ★v1.2 술어 2-수준 (t95 실측: 목록 조회는 8/8 이미 함 → L1 단독은 t95서 0회 발화):
    L1(목록): 멀티엔티티 의도(SCOPE_TOKEN/수량≥2) ∧ list-enumerator(A2: get_user_details) 미호출
              ∧ 해당 intent-class 첫 write 시도 → deny + "목록 먼저" 피드백 (t81형·저비용 유지)
    L2(상세): 요구 수량 N(사용자 발화 누적·ledger) > 매칭된 distinct entity 수 M
              ∧ 미검토 sibling entity 존재(목록엔 있으나 detail-reader(A2: get_order_details) 미호출)
              → write 시도 시 deny + "미검토 주문 [ids]의 details 먼저" 피드백
              (ids = 에이전트 자신이 가져온 목록 출력에서 옴 = 규칙0 클린)  ← t95 ⓐ(tr1/tr3형)를 닫는 술어
        ▼
[에이전트 자유 실행]  기존 루프(gated 인터셉터가 실행된 write + 읽은 entity를 ledger에 관측 기록)
        ▼
[CP5 COVERAGE-WALK]  종결(is_stop) 직전 · ★v1.2: 근거 = stop-time 전문맥 재-plan
    1) 재-plan 1회(LLM·격리 plan-추출·전 대화 = C14 "격리 계획선 정답" 소환 — 이 시점 문맥엔
       "두 대" 계시·양 주문 details가 있음 = 정보-맞춤)          ← t95 ⓑ(binding-gap·tr0/tr2형)를 닫는 축
    2) 결정론 diff: replan_writes ∖ executed_writes(expand_scope 후 매칭)
    3) gap 있으면 self.done 보류 + 리마인더를 생성-레벨(히스토리 비커밋)로 주입
       (에이전트 자신의 재-plan 재진술·gold 아님·read/write 강제 0) → 에이전트가 사용자에 재확인 발화
    4) 그래도 미완이면 통과(강제 없음·harm 회피·상한 1회 + step-budget 가드[리뷰 추가항목])
```

**설계 원칙 (FIXABLE §0 재프레이밍·★핵심)**: 32B fail 16 중 **14가 격리 계획선 이미 core_ok·controller 0발화**. ⇒ 이득은 batch/status 정규화가 아니라 **discovery + 완료-추적**에서 나온다. batch/status controller는 14B·부하 시만 발화(보조). **주기능 = CP5 stop-time 재-plan walk + L2 detail-read 강제.** 두 컴포넌트 분담 = t95의 두 실패 형상(ⓐ읽기-gap/ⓑ조립-gap)에 1:1 대응 — census(C64 A클래스: t41·t76·t81·t100·t102·t103·t111)도 같은 이분으로 커버.

---

## 2. 배선점 (실제 코드·`tau2/orchestrator/orchestrator.py`)

루프: `run()` → `while not self.done: step(); _check_termination()`. 종결=USER_STOP/AGENT_STOP/max_steps/max_errors.
기존 scaffold = `apply()`가 `BaseOrchestrator.gated(tool_calls)` monkeypatch(CP3 tool-call 인터셉터).

E-PLAN 후킹 (별도 patch·`t2_eplan_patch.py`·기존 gate_patch와 독립 toggle):
| 컴포넌트 | 후크 | 방식 |
|---|---|---|
| CP0 plan-extract | 첫 agent step 직전 (`initialize()` 후·first user msg 확정 후) | orchestrator 인스턴스에 `_eplan_ledger` 부착·1회 plan 생성 |
| discovery-enforce | **기존 replay-safe deny+regen 게이트에 precondition 추가** (신규 후크 불필요) | plan-scope 미발견 ∧ intent-class 첫 write 시도 → 생성-레벨 deny + enumerator-선행 피드백 |
| ledger 관측 | `gated(tool_calls)` (기존 인터셉터 확장) | 실행된 write tool_call을 `executed_writes`에 기록·enumerator 호출/결과서 `discovered_scope` 갱신 |
| CP5 coverage-walk | `is_stop`/`_check_termination` 직전 | 미완 planned 있으면 `self.done` 보류 + **생성-레벨 리마인더**(히스토리 비커밋·1~2회 상한) |

**★채널 절대규칙 (REPLAY_SAFE 교훈·`t2_gate_patch.py:997-1004`)**: 합성 메시지를 committed 히스토리에 넣으면 tau2 평가 `set_state` replay(mutating tool 재실행·environment.py:389 assertion)가 깨져 infrastructure_error가 난다 — 이미 밟고 generation-level로 옮겨 해결한 함정. **E-PLAN의 모든 개입(discovery deny-피드백·CP5 리마인더)은 생성-레벨(작업버퍼)만 사용·히스토리 커밋 금지.** 커밋되는 것은 에이전트가 실제 수행한 호출·발화뿐.

**규칙0 준수**: 주입(생성-레벨) 내용은 전부 리마인더(에이전트 자신의 plan 재진술 + "아직 처리 안 한 항목 N건")·**DB 내용 주입 0**·discovery는 에이전트가 부를 *도구를 지시*할 뿐 엔진이 대신 실행·대신 읽지 않음(에이전트가 스스로 호출→정상 tool-result). (present/autofetch와 차별 = C34 폐기선 안 밟음. autofetch=엔진이 실행+결과 주입, E-PLAN=도구-선행 요구만.)

---

## 3. controller / ledger 결정론 로직 ([[10]])

기존 `plan_execute_orch.py`의 `controller()`(batch-merge·status-fix·provenance-drop) 재사용 + 신규:

```
plan_ledger:
  planned  : [(intent_class, order_id|SCOPE_TOKEN, items, qty)]  # CP0 seed·qty=사용자 언급 수량(누적 갱신)
  executed : [(intent_class, order_id, items)]                   # gated서 관측
  listed   : set(order_id)     # list-enumerator(get_user_details) 출력서 파생
  examined : set(order_id)     # detail-reader(get_order_details) 호출 기록  ← v1.2
  replan   : [(intent_class, order_id, items)]                   # CP5 stop-time 재-plan 산출  ← v1.2

expand_scope(writes, listed) -> list:
  # ★v1.1: SCOPE_TOKEN 확장 — (exchange, ALL_PENDING)이 listed={W1,W2}와 만나면
  # [(exchange,W1),(exchange,W2)]로 구체화. 확장 없인 토큰 vs 구체 id 매칭 불가.
  # discovery 전엔 토큰 그대로 두고 "미확장 토큰 존재"를 L1 신호로 씀.

discovery_L1() -> bool:   # 목록-수준 (t81형)
  return (has_scope_token(planned) or max_qty(planned) >= 2) and not listed

discovery_L2() -> list:   # ★v1.2 상세-수준 (t95 ⓐ tr1/tr3형)
  # 요구수량 N > 매칭 distinct entity M ∧ 목록엔 있으나 미검토 sibling 존재
  N = required_qty(planned); M = len(distinct_entities(executed, intent_class))
  return sorted(listed - examined) if N > M else []

coverage_gap() -> list:   # ★v1.2: CP5서 replan 기준으로 diff (CP0 planned 아님)
  return [p for p in expand_scope(replan, listed) if not any(_covers(e, p) for e in executed)]
```

- **selector/verifier = 결정론**(controller). **생성기 = LLM**(plan-spec·재-plan). [[10]] 준수.
- `intent_class`·`enumerator_spec`·`SCOPE_TOKEN` 어휘 = 도메인일반. 매핑(retail: `get_user_details`가 enumerator, ALL_PENDING→status=pending 필터)만 A2.
- ★banking 겸용: enumerator_spec = 계좌/절차 목록 도구·SCOPE_TOKEN = 절차 단계 집합. coverage-walk = "gold 절차 median 8단계" 중 미완 단계 추적(C52 horizon/reach 처방).

---

## 4. 반대편 계측 (제1원리·Δ 필수)

| 부작용 | 계측 | GO 조건 |
|---|---|---|
| over-read (불필요 enumerator·턴 낭비) | `_eplan_reads_added` / sim · turn 예산 (**CP0 plan-extract 추가 생성 1회/sim 포함 계상**) | Δtme ≤ 0 (too_many_errors 미증) |
| **over-action** (walk가 안 시킨 write 유도) | passing-spurious Δ (vs floor) | **Δspurious ≤ 0** |
| walk-reminder가 멀쩡한 종결 흔듦 (C53 p4형) | 짝 flip census (pass→fail) | robust 상실 ≤ 획득 |
| plan-extract 오염 (틀린 plan을 walk가 강화) | plan pre/post core_ok (오프라인) | plan 정확도 유지 |

**절대선**: coverage-walk는 **읽기만 강제**. 미완 항목을 "해라"가 아니라 "이 주문들 아직 안 봤다" 리마인더 → 에이전트가 판단. abstain→forced-act 전환 금지(§1.5: ⋈서 p≈0.44<0.5 ⇒ 기대-유해). walk가 강제하는 유일한 것 = read.

---

## 5. 테스트 계획 (단계·[[09]] 무료 先)

| 단계 | 내용 | 비용 | 게이트 |
|---|---|---|---|
| (c) 단위 | `test_eplan.py`: ledger/coverage_gap/**expand_scope**/discovery_**L1·L2** 순수로직 (tau2-stub·오프라인) | 무료 | ALL PASS |
| (c) 오프라인 replay | 기존 `plan_execute_orch --replay`로 controller 정규화 무회귀 확인 | 무료 | pre/post 무변 |
| (d) 격리 검증 | 표적 실 궤적에 격리 주입 (32B 로컬·유료런과 GPU 경합 회피). 확인항목: ①USER_STOP 후 walk 시 user-sim이 재관여하나(sim instruction에 둘째 주문 의도 有→기대 합리·실측 필요) ②잔여 step < walk 소요면 walk 스킵 가드 ③CP5 리마인더 문구 A/B(id 명시 vs 개수만 — C43 재료 여부) **④(v1.2) L2 술어가 t95 tr1/tr3형(미검토 sibling)서 발화하고 tr0/tr2형(전부 검토)선 침묵하는지 — 술어 특이도** **⑤(v1.2) stop-time 재-plan이 t95 전문맥(양 주문 details+‘두 대’ 계시 포함)서 2-exchange plan을 내는지 = C14 정보-맞춤 재검증** **⑥(C64-G 프로브·스코프 밖 사전조사) 재-plan이 write-의무 외 communicate-의무("답해야 할 질문")도 안정 추출하는지 — t3형 relay-gap(`CENSUS_LEVERS_DESIGN §2b`)의 CP5-확장 타당성 판단용·측정만** | 무료 | ④⑤ 통과 = 두 형상 각각의 레버 유효 |
| (e) 표적 nt=1 | C64 A클래스(t41·t76·t81·t95·t100·t102·t103·t111) × nt=1 사이클(§0b 프로토콜) | 소액(승인) | per-case 복구(**t95는 db_match 기준**·NL-calc 병발 분리) ∧ Δspurious≤0 ∧ Δtme≤0 |
| full | 별도 456 (루프 아키텍처 변경·**합산 금지**·`§CP5`) | 유료(승인) | GO 조건(아래) |

**스모크 필수**([[30]]): full 전 `--num_tasks 10 --num_trials 1`로 3컴포넌트 라이브 발화 검증(마커 stderr). 단위PASS≠라이브발화(calc 31/342 선례).

---

## 6. GO 조건 · 도달 목표

- **GO**: 표적 class per-case 복구(t95형 discovery→2write·**db_match 기준** — reward는 NL-calc 이중결손과 혼동) ∧ **Δspurious ≤ 0** ∧ Δtme ≤ 0(turn 예산·재-plan +1콜/sim 포함) ∧ 위반0 유지 ∧ 짝 flip 순증.
- **도달 목표**(`ENDGAME §3`): retail 32B R2 후 0.66~0.70 중 E-PLAN 몫 = MISSED+ZERO_NEV 47 sims(10.3pp) headroom의 부분회복.
- 실패 시: 레버 개별 제거(§1.3 죽은레버 등재)·discovery만 살리고 walk 드롭 등 부분채택.

## 7. 미해결 · 리스크

- **R1 plan 오염 (v1.2 축소)**: CP5 근거가 stop-time 재-plan으로 바뀌어 CP0 오염의 사정거리 축소(seed·L1 신호만). 재-plan 자체의 오류는 남음 — 완화 = 리마인더는 강제 아님·에이전트 재판단 여지 + C14가 정보-맞춘 격리 plan의 정확성을 지지(§5④⑤가 실측).
- **R1b 재-plan 비용 (v1.2 신규)**: stop-time LLM 1콜/sim 추가. gap=∅이면 리마인더 없이 즉시 종결(비용만 지불) — Δtme 계측에 포함·과대면 "write 있었던 sim만 재-plan" 조건부로 강등.
- **R2 discovery 과잉 (v1.1 재정식화)**: 에이전트가 *스스로* enumerator를 부르므로 전체 결과가 어차피 정상 tool-result로 창에 들어옴 — "controller가 결과를 필터해 개수만 노출"은 주경로에서 **불가능**(그러려면 controller가 결과를 대신 읽어야 = 규칙0 위반. v1.0의 내부모순·해소). C43 노출 재평가: retail enumerator=get_user_details=**본인 주문만** 표면화 → 무관-entity 오염은 구조적으로 제한적. "개수만 vs id 명시" 선택지는 **CP5 리마인더 문구에만** 적용(리마인더 원천=에이전트 자신의 plan·DB 아님) → §5(d)③ A/B로 격리 측정. **C43 잔여 긴장 = 본인 주문 중 무관 주문 id가 write 인자로 새는지 — Δspurious 계측이 그 센서.**
- **R3 종결 지연**: walk 리마인더 상한(1~2회) 없으면 max_steps 낭비. 상한 하드코딩.
- **R4 banking 전이**: enumerator_spec/SCOPE_TOKEN이 banking 절차-집합에 매핑되나 = Phase 3 실측(retail 확정 후).
- **소유권**: E-SPEC(오케스트레이터 재설계)와 CP5 좌석 공유 — E-PLAN은 coverage-walk만·E-SPEC은 전체 재배치. 중복 구현 금지.
- **★ledger 이중 소속 (v1.3·`CENSUS_LEVERS_DESIGN §3a` 교차)**: ledger를 두 층으로 분리 — **관측-전용 부품**(CP0 plan-추출+executed/listed/examined 기록만·에이전트 창 불변·개입 0·교란=plan 서브콜 1회/sim뿐) vs **개입 레버**(discovery L1/L2·CP5 walk). E-PLAN *arm*(합산 금지) 소속은 개입 레버만. 관측-전용 ledger는 CENSUS-레버 stage B 스택(EXCLUSIVITY가 planned 참조)에 동거 가능. toggle도 분리: `T2_EPLAN_LEDGER=1`(관측만) / `T2_EPLAN=1`(개입 포함·전자 함의).

---

## 8. 다음 액션 (구현 순서)
1. `t2_eplan_patch.py` 스켈레톤 (ledger 부착·gated 확장 관측[executed+listed+examined]·CP5 stop-time 재-plan 후크·discovery L1/L2=기존 deny+regen에 precondition 추가) — 무료.
2. `test_eplan.py` 단위 (coverage_gap·expand_scope·discovery_L1/L2·ledger·qty 누적) — 무료.
3. A2에 `enumerator_spec`(retail: {list: get_user_details, detail: get_order_details}) 추가 + SCOPE_TOKEN 파서.
4. 격리 검증 §5(d) ①~⑤ — 무료·GPU 한가할 때. **⑤(재-plan C14 재검증)가 최우선**(CP5 주축의 생사).
5. → 표적 nt=1 (승인 필요). *(step b 리뷰 + v1.2 비판적 재해석 = 2026-07-11 완료·설계 확정)*
