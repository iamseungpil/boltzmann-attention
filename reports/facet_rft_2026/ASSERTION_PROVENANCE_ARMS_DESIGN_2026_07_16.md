# 주장-출처(assertion-provenance) 확장 — 3-arm 설계 (2026-07-16)

> 파생: `RESEARCH_MASTER` §1.3(제1원리·상쇄) · §1.5(Q1/Q5) · 원장 C45/C46/C48(출처선언 레버) · C30/[[42]](prompt-only 무효)
> 규율: [[03b]] 구현속임 금지 · [[05]] 엔진 리터럴 0·A2만 변경 · [[10]] LLM=formalize/엔진=검증 · [[08]] 궤적정독 · [[09]] 무료 先

## 0. 한 줄
C45 출처선언 레버는 **write 인자**에만 걸려 있다. banking 잔여 병목은 write가 아니라
**사용자에게 하는 사실 주장(assertion)** 에서 샌다 — 에이전트가 producer 도구를 안 부르고 "리워드 맞다"고 판단한다.
확장 후보 3안을 정의하고, **어느 것도 공짜가 아니므로**(§1.3) 상쇄를 같이 잰다.

## 1. 포렌식 근거 (t019g 3 sim 전수·[[08]])
정본 궤적: `sim_results/bank_t019g_20260716.results.json.gz` (커밋 **`48785fcc`**·이번 세션 영속화.
선행 `6e906e77`은 compliance.json을 잘못 담아 `48785fcc`가 교체). 대조: `bank_t019d_20260716.results.json.gz`.
종료 3/3 `user_stop`(crash/infra 0).

| sim | 도구호출 | `get_reward_discrepancies` | 주장 지점 | 주장자 |
|---|---|---|---|---|
| 0 | 8 | **0** | msg[16][20][24][28] "rewards ... align with the expected rates" | 에이전트 |
| 1 | 11 | **0** | msg[38] "rewards earned seem to be calculated correctly" | 에이전트 |
| 2 | 9 | **0** | msg[31] user-sim이 먼저 결론 → msg[32] 에이전트 **추인** | user-sim→에이전트 |

- **★sim 0 msg[26]**: 사용자가 밀어붙이자 "다시 확인하겠다"고 선언 → **도구 호출 0** → msg[28]서 같은 눈대중 반복.
  ⇒ soft 압박(사용자 발화)으로는 producer 호출이 유도되지 않는다([[42]] 동형).
- **★sim 2**: 실패 형태가 다르다 — *에이전트가 판단*이 아니라 *user-sim의 틀린 판단을 추인*. arm 설계는 둘 다 덮어야.

### 1b. ★기존 레버(NLNUM)는 이 병목에 **오조준** (오프라인 실측·무료)
`T2_NLNUM_PROV`(t2_gate_patch.py:1014·2566)의 검증기(`_MONEY_RE`/`_unverified_amounts`)를 두 arm 궤적에 재현 실행:

| arm | sim | 발화 고유금액 | 맞는 산술 | 근거불명(참양성) |
|---|---|---|---|---|
| t019g | 0·1·2 | 2·2·2 | **2·2·2** | **0·0·0** |
| t019d | 0·1·2 | 0·0·2 | 0·0·**2** | **0·0·0** |

- **발화 8/8 전부 *맞는 산술***: KB `doc_credit_cards_credit_cards_(general)_006` 원문 = *"1 point = $0.01 ... 250 points equals $2.50"*.
  7021 points → `$70.21`, 4839 → `$48.39` (대응 points 값이 전부 문맥에 실재). ⇒ **참양성 0 · 전량 over-block.**
- t019d sim0/1 발화 0 = 카드 조회 도달 실패(1/3 도달)와 정합 — 교차 일관.
- **정작 진짜 실패는 못 잡는다**: "align with the expected rates" 문장엔 날조 숫자 0 (points·amount 전부 도구 출력 원문 복사).
- ⇒ **값-출처(value-provenance)와 판단-출처(assertion-provenance)는 다른 술어다.** C46(날조 vs ⋈ 분리)의 동형 정련.
- 현재 banking A2에 `calc_tool` 키 부재 = NLNUM **비활성**. (=이번 t019g에 실제로 걸리지 않았음. 위는 *반사실* 재현.)

## 2. 3-arm 정의 (공통 불변량: 엔진은 어시스턴트 **텍스트를 파싱하지 않는다**)
> 텍스트 정규식으로 "판단"을 탐지하면 = 엔진-formalize + 도메인 리터럴 = **[[03b]]/[[05]] 위반 = 실험무효.**
> 이번 세션 위반 3회가 전부 이 지점. 후크는 **구조 이벤트** 또는 **LLM 자기선언**만 본다.

### arm `discovery-required` (도구-이벤트 후크)
- **A2**(ABox·도메인사실): `analysis_producers: [{data_source: <tool>, producer: <tool>, subject: <str>}]`.
- **엔진**: 사임 시점(=tool_calls 없는 어시스턴트 발화 = 턴을 사용자에게 넘김)에
  `data_source 호출됨 ∧ producer 미호출` → regen 1회(상한 1/sim). **엔진이 보는 것 = 호출 이름 집합뿐.**
- 근거: 등대 §1.5 Q5 *"읽기만 강제, 쓰기는 절대 금지"* · [[14]] read(discovery)만 강제. producer=GET(읽기).
- **예상 역효과**: 리워드와 무관한 대화(단순 거래 나열 요청)서도 발화 → over-block. **계측 필수.**

### arm `self-declaration` (C45 4지선다의 assertion 확장)
- **A2**: `assertion_operands: {<operand>: <producer tool|null>}` (예: `expected_reward → get_reward_discrepancies`).
- **엔진**: 사임 시점에 sub-call 1회 — *"직전 답변이 주장하는 operand와 각각의 출처를 {GET·FIND·INFER·ASK}로 선언(JSON)"*.
  선언 **구조체**를 결정론 검증: `source=INFER ∧ A2에 호출가능 producer 존재` → 무효 → 피드백+regen.
- [[10]] 정석: LLM=formalize(자기 주장의 operand 추출)·엔진=검증만. 엔진은 JSON 필드만 읽는다.
- **예상 역효과**: 매 사임 turn마다 sub-call = 전-궤적 비용(§1.3 thinking과 동형: F5 매도 위험) · 에이전트가 선언 회피/거짓선언 가능.

### arm `value-provenance` (기존 NLNUM) — **❌ 드롭(2026-07-16·사용자 결정)**
- 사유: §1b 반사실이 **이미 답을 줌**(발화 8/8 전량 over-block·참양성 0). 라이브는 돈만 쓰고 같은 결론.
- 지위: **오프라인 실측으로 갈음** — "값-출처 레버는 이 병목에 오조준"이 결과([[09]] 무료검증 先의 정확한 사례).

## 2b. ★오프라인 검증 결과 (구현 後·라이브 前·무료)
재현: `scripts/distill/tau2/bank_assertion_arms_offline.py` (커밋 `3043c474`). 엔진 판정 로직만 replay.

| arm | 궤적 | 사임(발화기회) | **발화** | 발화 지점 | over-block |
|---|---|---|---|---|---|
| discovery-required | t019g sim0 | 7 | **1** | msg[20] = 첫 눈대중 주장 | 0 |
| discovery-required | t019g sim1 | 12 | **1** | msg[38] = 눈대중 주장 | 0 |
| discovery-required | t019g sim2 | 10 | **1** | msg[30] = **user-sim의 틀린 결론[31] 직전** | 0 |
| discovery-required | t019d sim0·1 | 7·8 | **0·0** | (거래 미도달) | 0 |
| value-provenance(NLNUM) | t019g·d | — | 8 | 맞는 산술 | **8/8** |

- **정밀**: 사임 7~12회 중 **1회만** 발화(남발 아님) — data_source 읽기 前 사임엔 안 걸림.
- **sim2 추인 형태도 선점**: msg[30] 발화가 user-sim의 [31] 오결론보다 앞선다.
- ⚠️ **이 task는 리워드-중심**이라 *무관 대화서의 over-block*은 여기서 시험되지 않음 → **라이브 계측 대상**([[05]] 가드 (2)).
- ⚠️ 오프라인이 재는 것 = **발화 여부/지점**뿐. regen 後 실제로 producer를 부르는지(=thrash 여부)는 **라이브만 판정**.

## 3. 자기감사 ([[03b]]·제시 前 필수)
| 점검 | 결과 |
|---|---|
| 엔진에 도메인 리터럴? | **0** — 카드명·rate·"reward" 문자열 전부 A2. 엔진은 `{tool 이름 집합}`·`{JSON 필드}`만 |
| 엔진이 formalize? | **아니오** — arm1=이벤트·arm2=LLM 선언·arm3=기존(값 정규식은 *검증*이지 operand 공급 아님) |
| hack-pass? | 없음 — gold 4건을 엔진이 알지 못함. producer op-DAG는 A2 |
| 강제(forcing)? | 없음 — 전부 regen 피드백. `tool_choice=required` **미사용**(지난 세션 폐기 유지). write 강제 0 |
| 도메인-타깃 학습? | 없음(학습 0) |
| **잔존 위험** | **soft regen thrash**(핸드오프 §4 실증: deny 시 32B=동일반복·표면변형·옆-날조). arm1/arm2 둘 다 soft → 무효 가능 |

## 4. 계측 (등대 §1.3: 게이트 자신의 역효과를 반드시 같이 잰다)
1. **발화율**: sim당 fire 수·발화 지점 (stderr 태그).
2. **★over-block**: 발화가 *맞는 주장*에 걸린 건수 (NLNUM $70.21형). 궤적 정독으로 판정.
3. **★Δspurious ≤ 0**: regen이 새 날조/over-action을 유발했는가 (C45 GO 조건).
4. **thrash**: regen 후 producer 실제 호출 여부 vs 동일-주장 재발화.
5. pass / 종료사유 분포.
6. **미발화도 결과다**([[30]] 교훈: calc 31/342 미발화를 천장으로 오인).

## 5. 실행 순서 ([[09]])
1. 오프라인: t019g 궤적 replay로 arm1/arm2 발화 지점·over-block 예측 (**무료**).
2. 구현 → 오프라인 단위검증 → **스모크**(gpt-4.1-mini·nt=1~3) = 라이브 발화 확인 (단위통과 ≠ 라이브발화·[[30]]).
3. 정본: gpt-5.2 user-sim·nt≥3 — **사용자 승인 필수**([[09]]).

## 7. ★★★스모크가 드러낸 것 — arm이 아니라 **우리 게이트가 task를 통과불가로 만들고 있었다** (2026-07-16)
> arm 판정 이전에 **선행 무효화 사유**. [[08]]이 요구한 궤적 포렌식이 잡았고, 집계(pass=0)만 봤으면 "에이전트가 못 한다"로
> 오귀속했을 것. **C16(RBW 격차=scaffold 아티팩트)의 재발.**

### 7.1 실측
`bank_t019h`(= t019g + `T2_DISCOVERY_REQUIRED=1`·정본 `sim_results/bank_t019h_20260716.{results.json,log}.gz`·커밋 `6687b6d6`):
- discovery-required **라이브 발화 3회 확인**(오프라인 예측대로) · 단 3회 모두 `regen tool_calls=[]` · producer 호출 0.
- **그러나 종료 = `user_stop` 2 / `infrastructure_error` 1** · Retry **6회** ⇒ **실효 n=2 · 판정 불가**.

### 7.2 근본 원인 (버그·지난 세션 `f53c7621`)
- `orchestrator.py:882` = `from_role in [AGENT, USER] and to_role == ENV` → `_execute_tool_calls`
  ⇒ **user-sim의 도구 호출도 같은 함수를 지난다.**
- 우리 `t2_scaffold_get.exec2`는 **requestor를 보지 않고** `agent.tools`(=`_t2_known_tools`)로만 검사 ⇒
  **사용자가 자기 도구를 부르면 "없는 도구다·지어내지 마라"고 거부**당한다.
- **task_019 gold의 4/6이 `requestor:"user"`**:
  `019_2~5 = call_discoverable_user_tool(submit_cash_back_dispute_0589, txn_...)` × 4 (+019_0 log_verification·019_1 give_discoverable_user_tool).
  ⇒ **T2_TOOLGATE=1이면 사용자의 gold 액션이 항상 차단 = task_019는 구조적으로 통과 불가 = reward 0 보장.**
- 부작용 2: 거부 ToolMessage가 `requestor="assistant"` 하드코딩 → user-sim 히스토리 flip서
  `user_simulator_base.py:102 ValueError` → Retry → `infrastructure_error`.
  **ValueError 발생 자체가 "차단된 호출이 사용자의 것"이라는 증거**(에이전트 호출은 user 히스토리에 안 들어감).
- t019g 로그에도 같은 ValueError **1회** 존재 ⇒ **버그는 t019g에도 있었다**(운좋게 3/3 생존).

### 7.3 ⇒ 철회/보류되는 주장
- ❌ **"t019g reward 0/3 = 에이전트가 reward 도구를 안 골라서"** — **보류**. 그 런들은 gold 4/6이 차단된 상태였다.
  `get_reward_discrepancies` 미선택(0/3)은 여전히 사실이나, **pass=0의 귀속은 무효**.
- ❌ **"에이전트가 `call_discoverable_user_tool`을 날조했다"(정박 치환)** — **철회**. **실재하는 gold 도구**다.
  우리 게이트가 잘못 거부한 것. (진짜 날조 예: `get_user_information_by_phone_number`.)
- ⚠️ 핸드오프 §0의 서사(= 잔여 병목 = reward 도구 미선택)는 **이 버그 위에서 관측됨** → 수정 後 재측정 필요.

### 7.4 수정 (커밋 `<이 커밋>`)
- `exec2`: **requestor 격리** — `tc.requestor != "assistant"`면 우리 경로 일절 미적용(원본 실행으로).
- 반환 ToolMessage의 `requestor`를 tau2 원본과 동형으로 **미러링**(`environment.get_response`: `requestor=message.requestor`).
- 회귀 테스트 `test_toolgate_requestor.py` 5/5 PASS (user gold 통과 · 에이전트 날조는 ASK 보존).
- **재측정 필요**: floor / t019g(게이트만) / t019h(+discovery-required) — 전부 수정 後 다시.

## 6. Caveat (정직)
- t019g = **n=3 × gpt-4.1-mini** = robust 측정 아님·**메커니즘 관측**. reward도구 0선택만 3/3 일관 + 원문 정독으로 견고.
- **sim 2형(user-sim이 틀린 결론을 먼저 줌)** 은 user-sim 품질 의존 — gpt-5.2선 다르게 나올 수 있음([[47]] 권장표준).
- arm1/arm2 모두 **soft** → 지난 세션 thrash 실증에 비춰 무효 가능. 무효면 그 자체가 [[13]] 경계 증거(scaffold 상한).
