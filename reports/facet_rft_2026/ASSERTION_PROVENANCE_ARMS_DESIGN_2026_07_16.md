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
- **재측정**: `bank_t019i`(대조=게이트만·8140) vs `bank_t019j`(+discovery-required·8141) — 병렬·단일 변수 차이.

### 7.5 ★버그픽스 라이브 검증 (t019i·t019j)
| 지표 | 수정 前 t019h | **수정 後 t019i·t019j** |
|---|---|---|
| `ValueError`(user 히스토리 오염) | 1+ (t019g도 1) | **0 · 0** |
| Retry | **6** | **0 · 0** |
| `call_discoverable_user_tool` 오거부 | 발생 | **0 · 0** |
| `infrastructure_error` | 1/3 sim | (완료 후 기입) |
⇒ **requestor 격리가 크래시 원천을 닫았다**(라이브 실증·단위테스트만이 아님).

## 8. ★도구 목록 = 권위본 (2026-07-16 확정·추측 금지)
`registry.get_env_constructor("banking_knowledge")()` 직독:
- **AGENT 15개**: `KB_search · call_discoverable_agent_tool · change_user_email ·
  get_credit_card_accounts_by_user · get_credit_card_transactions_by_user · get_current_time ·
  get_referrals_by_user · get_user_information_by_{email,id,name} · give_discoverable_user_tool ·
  list_discoverable_agent_tools · log_verification · transfer_to_human_agents · unlock_discoverable_agent_tool`
  (+ 우리 A2 주입 2: `get_reward_discrepancies` · `verify_identity`)
- **USER 6개**: `apply_for_credit_card · call_discoverable_user_tool · list_discoverable_user_tools ·
  request_human_agent_transfer · submit_referral · submit_transaction`
⇒ **사용자 조회 key = email/id/name 뿐** · `get_user_information_by_{phone,phone_number,date_of_birth}` = **날조 확정**.
⇒ `call_discoverable_user_tool`·`apply_for_credit_card` = **실재 USER 도구**(§7 버그가 막고 있던 것).
> **날조 판정은 이 목록으로만 한다.** 이번 세션에 목록 미확인 상태로 "정박 치환 날조"라 단정했다가 철회(§7.3).

### 8.1 날조 루프의 원인 = 정책 factor와 조회 key의 불일치 (t019j sim2 정독)
`msg[8]` 에이전트: "DOB·email·phone·address 중 **2개**를 달라"(정책) → `msg[9]` 사용자: **DOB+phone** →
`msg[10,12,16,20]` **`get_user_information_by_phone` 4~5회 반복 날조** → 전부 차단(실행 0).
- **정책이 인정하는 factor(phone·dob)로는 *조회*가 불가**하고, 조회 key는 email/id/name뿐.
  올바른 형식화 = "**name으로 조회 → 레코드의 phone/dob를 대조**"인데 에이전트는 "phone으로 **조회**"로 형식화한다.
- 우리 `verify_identity` 설명에 *"name/email/id로만 조회"* 를 이미 명시했는데도 발생 ⇒ **C30/[[42]] prompt 천장 재확인**.
- ⇒ 소속검사(=출처 조사)는 **날조 실행을 0으로 막지만**, *재선택*을 사지 못한다. 잔여 = 형식화 오류.

### 8.2 ⚠️user-sim 오염 (내 규율 위반)
- **[[30]] 확정 = 라이브 e2e user-sim은 gpt-5.2**(리더보드 comparability). 그런데 이 세션 스모크(t019g/h/i/j)는
  **gpt-4.1-mini**로 돌았다 = 위반(핸드오프 §5의 "스모크=mini" 표기를 메모리보다 위에 둠).
- 실측 피해: t019j sim2 `msg[25]`서 **user-sim 자신이 `call_discoverable_user_tool(discoverable_tool_name=
  "get_user_information_by_name")`를 날조** → 환경 `Unknown discoverable tool` → 다음 턴 "그 도구가 없다네요"로 혼선.
  ⇒ **실패의 일부가 user-sim 산물**. §1의 "sim2 = user-sim이 틀린 결론을 주고 에이전트가 추인" 관측도 **mini 아티팩트 혐의** → 보류.
- ⇒ 정본 재측정 = `bank_{ctl,dreq}_20260716_2140` (**gpt-5.2 · `--user_temp 0.0` · nt=5 · 고유 tag · 완료 즉시 영속화**).

## 9. ★★★gpt-5.2 재측정 — 병목은 reward가 아니라 **verify서 도구 선택 실패** (2026-07-16·⚠️잠정)
정본: `sim_results/bank_dreq_20260716_2140.*` (gpt-5.2·user_temp 0·nt=5).
> ⚠️**등급 [P](잠정) — 확정 금지**([[08]]): (a) **완료 2/5**·(b) **대조군(ctl) 미완 = 교차표 없음**·
> (c) 단 이 2 sim은 **거래 미도달 ⇒ DISCREQ 발화 0 ⇒ dreq ≡ ctl**이므로, 아래는 *arm 효과가 아니라
> **기본 스택**의 관측*이다. nt=5 완주 + ctl 도착 後 등급 재판정.

| sim | agent 도구 시퀀스 | 거래 도달 | reward |
|---|---|---|---|
| 0 | `by_phone_number`(날조) → `get_current_time` → `by_phone_number`(날조) | ✗ | 0.0 |
| 1 | `KB_search` → `by_phone_number`×4 (날조) → `KB_search` | ✗ | 0.0 |

⇒ **`get_reward_discrepancies` 이전에 거래 조회 자체에 도달 못 함.** §0의 "잔여 병목=reward 도구 미선택"은 **하류 현상**이었다.

### 9.1 sim0 전문 정독 (근인 확정)
`[2]` 에이전트: "dob·email·phone·address 중 **2개**" → `[3]` 사용자: **dob+phone**(페르소나가 가진 전부) →
`[4][8]` `get_user_information_by_phone_number` 날조·차단 → `[10]` "email이나 주소를 달라" → `[11]` 없다 →
`[12]` **"시스템은 목록에서 2개를 요구합니다"**(이미 2개 받았음) → `[14]` **"다음엔 준비하세요: 1. 성명 — 계정을 찾는 데 필요"** →
`[15]` 사용자 `###STOP###`.
- ★**에이전트는 "이름이 계정을 찾는 key"임을 *알면서*(msg[14]) 지금 이름을 묻지 않는다.**

### 9.1b sim1 정독 (2/2 동형·종료 user_stop)
`[12]` **user ID** 요청 → `[14]` email/주소 → `[16]` KB_search("verify by email") → `[18]` email → `[32]` "다음엔 email이나 주소를 준비".
- ★★**2/2 sim 모두 `name`을 단 한 번도 묻지 않는다** — `get_user_information_by_name`이 자기 도구 목록에 있는데도.
  (묻는 것은 id·email·address = **factor∩key 중 사용자가 못 가진 것들**뿐.)
- ★★**언어화한 지식이 행동을 통제하지 못한다**: `[20][24][26][30]` *"I do not have a direct tool to
  fetch user information by phone number"* 라고 **스스로 말해놓고** `[22][28]`서 **또 `by_phone_number` 호출**.
  ⇒ **[[42]] prior-override의 최청정 사례** · 프롬프트·피드백 무효(C30) 재확인. 차단은 되나 **재선택은 안 산다**.
- **근인 = 검증 factor(dob/email/phone/address) ↔ 조회 key(email/id/name) 혼동.**
  name은 factor 목록에 없어서 **후보로 떠오르지 않고**, factor 중 조회가능한 email만 묻다가 없다니까 포기.
- 우리 A2 `verify_identity`(설명에 "name/email/id로만 조회" 명시)도 **호출 0**.
- TOOLGATE 피드백("ASK the customer … then call an available tool")도 무효 ⇒ **C30/[[42]] 재확인**.

### 9.2 ★mini가 에이전트를 구하고 있었다 (핸드오프 §0 무효화)
**직접 관측된 사실**(궤적 원문·인과 아님):
- mini user-sim은 **묻지 않은 이름을 스스로 제공**한다 — t019j sim2 `msg[25]` *"My full name is Priya Sharma."*
  = **spoon-feed**([[16]] 금지 대상을 user-sim이 수행). 같은 sim서 mini는 **도구명까지 날조**(`msg[25]` 참조).
- gpt-5.2 user-sim은 주지 않는다 — sim0 `[11]` *"don't want to guess and give you the wrong info"* → `[15]` `###STOP###`.

**⚠️인과 귀속은 보류**: "t019g가 거래 3/3 도달한 것은 mini가 이름을 줘서"는 **격리되지 않았다** —
t019g→dreq 사이에 **user-sim(mini→5.2)** 과 **§7 requestor 버그픽스** 두 변수가 같이 바뀌었다.
말할 수 있는 것: *mini는 spoon-feed를 하고 5.2는 안 한다*(관측) · *5.2선 에이전트가 검증 벽서 정지한다*(관측).
⇒ **핸드오프 §0의 병목 지정은 mini 위에서 만들어졌으므로 재검이 필요**하다(무효 단정이 아니라 **보류**).

### 9.1c ★★★sim2 (검증 벽을 넘은 유일 sim) — arm은 **선택을 샀고**, 잔여는 **operand formalize**
도구 시퀀스: `by_phone_number`×2(날조·차단) → **`get_user_information_by_name`(회복)** → `log_verification`(**gold 019_0**)
→ `get_credit_card_transactions_by_user`(거래 도달·23건) → `get_user_information_by_name` → `give_discoverable_user_tool`×3(**gold 019_1**).

| msg | 사건 | 판정 |
|---|---|---|
| `[25]` | 거래 23건 읽음 | — |
| **`[26]`** | **DISCREQ 발화 → 에이전트: _"I see, thank you for pointing that out. … I need to use the `get_reward_discrepancies` tool"_** | ⚠️**arm이 산 것 = *언급*이지 *호출*이 아니다** — **producer 실호출 = 0/5**(dreq 완주·DISCREQ 3발화). 이전(눈대중 "이상 없음")과는 다르나 **선택을 샀다는 서술은 과장**(자기교정 2026-07-16) |
| `[26]` 이어서 | _"However, this tool **requires specific transaction IDs**. Could you please **provide the transaction IDs**?"_ | ☒**operand formalize 실패** — 방금 읽은 23건을 인자로 안 넘기고 **사용자에게 요구** |
| `[29]` | **user-sim(gpt-5.2)이 직접 불일치를 계산** — _"txn_f093f96e2001 (Thrive Market, $175, Green) — rewards show 175 points, but other Green…"_ | ☒**과제를 사용자가 수행**(페르소나 *"Don't diagnose the problem yourself"* 위반) |
| `[34-38]` | 사용자가 찾아준 **3건**으로 dispute 도구 제공 | 부분 |
| `[39-44]` | **사용자가 3건 제출 성공**(`dsp_…` 3개) | gold 019_2·3·5류 부분 달성 |
| — | **gold는 4건**(`txn_f093f96e2001·580773a8649e·d398545ca1a2·37b5b8e67a5e`) → **3/4** | ⇒ **reward 0 = coverage(F4) 미달** |

**⇒ 실패 사슬(정본)**: `DISCREQ 발화` → **에이전트가 도구를 *말로만* 지목** → `인자 formalize 실패` → `사용자에게 떠넘김`
→ `user-sim이 눈대중 3/4` → `coverage 미달` → **0점**.

### 9.1d ★★자기교정: "arm이 선택을 샀다"는 **과장** (dreq 완주 수치)
`bank_dreq_20260716_2140` 완주: **n=5 · reward 0/5 · DISCREQ fired 3 · `get_reward_discrepancies` 실호출 0**.
- ⇒ arm이 산 것은 **언급(verbalization)**뿐. **호출은 0.** §9.1c 초안의 "선택을 샀다"를 철회한다.
- ★**대칭 관측 — knowing ↔ doing 괴리가 양방향으로 나타난다**:
  - `by_phone`: **"그 도구는 없다"고 말하고 → 그 도구를 부른다**(sim1 `[20][24]`→`[22][28]`)
  - `get_reward_discrepancies`: **"이 도구를 써야 한다"고 말하고 → 안 부른다**(sim2 `[26]`)
  ⇒ **언어화된 지식이 행동을 통제하지 못한다**가 이 도메인 잔여의 통일 서술. 프롬프트·설명·피드백은 전부 *언어* 층이므로
  이 층에서 못 닫는다는 예측과 정합(C30/C47/[[42]]).
- ⇒ **새 잔여 = [[10]] 경계의 LLM 몫(원시 leaf→operand)** + **언어↔행동 괴리**. 둘의 분리는 §10 op/operand 프로브가 판정.

### 9.1e ★용어 정정 — "LLM이 도구를 호출한다"는 틀린 서술 (2026-07-16)
**LLM은 도구를 실행하지 않는다.** LLM은 `tool_calls`(이름+인자)를 **emit 할 뿐**이고, 실행은 오케스트레이터가 한다:
`LLMAgent._generate_next_message`(생성) → `BaseOrchestrator._execute_tool_calls`(`orchestrator.py:882`) →
`environment.get_response(tool_call)`(실행) → `ToolMessage`로 문맥 주입. **MCP도 동형** — 호스트가 서버에 연결·호출하고
모델은 tool-use를 emit 할 뿐이다. TOOLGATE도 "모델이 못 부르게" 하는 게 아니라 **emit된 요청을 실행 직전에 가로채는** 것.
- ⇒ 정확한 실패 서술 = **"호출을 안 한다"가 아니라 "tool_call을 emit 하지 않고 텍스트를 emit 한다"**(sim2 `[26]`).
- ⇒ **금지선**: "오케스트레이터가 대신 불러주면 되지 않나" = **NO**. producer를 자동 호출하려면 엔진이 인자(23건 레코드)를
  스스로 만들어야 하고 = tool 출력 **텍스트 파싱** = **[[03b]] 엔진-formalize**(이번 세션 §7-전에 제거한 그 cheating).
  선례: **C34**가 같은 이유로 `candidate_summary`·`autofetch` 폐기(에이전트가 안 부른 도구를 엔진이 대신 호출·주입).
  ⇒ **선택(emit)과 formalize는 정의상 LLM 몫**이며, scaffold가 대신하면 측정 대상 자체가 소멸한다.
- ⇒ 우리 A2 `params.transactions` 설명(*"A JSON array … that you read from the transaction records"*)이 명시하는데도
  "고객이 ID를 줘야 한다"로 오독 ⇒ **또 prompt 천장**(C30/[[42]]).
- ⚠️ n=1 sim · **[P]**. arm 효과 주장은 ctl 대조 도착 後 확정.

### 9.3 ⇒ thesis는 오히려 선명해진다
잔여 = **"제공된 유한집합에서 맞는 도구를 선택하지 못함"** 하나이고, **두 지점서 동형**으로 발현:
- verify: `get_user_information_by_name` / `verify_identity`를 안 고름 → 없는 `by_phone*`을 만듦
- reward: `get_reward_discrepancies`를 안 고름 → 머릿속 눈대중
소속검사(=출처 조사)는 **날조 실행을 0으로 막지만 재선택을 사지 못한다.** 이것이 [[00]] 명제의 시험대.

## 6. Caveat (정직)
- t019g = **n=3 × gpt-4.1-mini** = robust 측정 아님·**메커니즘 관측**. reward도구 0선택만 3/3 일관 + 원문 정독으로 견고.
- **sim 2형(user-sim이 틀린 결론을 먼저 줌)** 은 user-sim 품질 의존 — gpt-5.2선 다르게 나올 수 있음([[47]] 권장표준).
- arm1/arm2 모두 **soft** → 지난 세션 thrash 실증에 비춰 무효 가능. 무효면 그 자체가 [[13]] 경계 증거(scaffold 상한).
