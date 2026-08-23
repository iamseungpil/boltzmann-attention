# t7346 task_050 per-step 포렌식 **v2** — 2026-08-23

> 대상 런 `bank_t7346_halfB_20260822` (meta sha `ee18d797`) · 전부 로컬 gz 무접촉 읽기 · SSH 0 · git 0.
> v1(`tasks__20260822/TASK_050.md`)의 재검증 + **독립 재현 + 3건의 신규 사실**.
> 수리 실행 없음 · 코드 무수정 · gold(`reward_info`)는 진단용으로만([[23]]).

| | trial 0 | trial 1 |
|---|---|---|
| sim | `task_050#s626729` | `task_050#s373753` |
| seed / msgs / 종료 | 626729 / 38 / `user_stop` | 373753 / 55 / `user_stop` |
| reward | **0.0** | **1.0** |
| 로그 라인 | 168 | 258 |

---

## 0. 채점 축 — **DB**(ACTION 아님) · v1과 동일

```
trial 0: reward_basis=["DB"]  reward_breakdown={"DB":0.0}  db_check={"db_match":false,"db_reward":0.0}
trial 1: reward_basis=["DB"]  reward_breakdown={"DB":1.0}  db_check={"db_match":true, "db_reward":1.0}
env_assertions=[]  nl_assertions=null  communicate_checks=null
info.action=null   (= action 축은 채점에 안 들어간다)
```

⇒ `action_checks` 는 **진단 보조**다([[69]]). 이 태스크 안에 그 증거가 있다: `050_10`
(`get_payment_history_6183{…,"months":3}`)은 **두 trial 모두 `action_match=false`** 인데 trial 1 은
reward 1.0 이다 — `months` 는 상태를 안 바꿔 DB 해시에 남지 않는다(C583ⓖ).

---

## 1. 변이표 — 정본 `t2_forensic.mutation_diff` 만 사용(손 비교기 0 · C583ⓐ)

`mut = t2_forensic.mutating_tools()` = env 가 `mutates=True` 로 선언한 44종.
⚠ 이번 v2 는 **A-7⑵ 교정이 들어간 `t2_forensic`**(`ENV_FAIL_PREFIX = ("Error:", "Failed to ")`,
`t2_forensic.py:514`)로 다시 돌렸다 — `MASTER_DUP_CORRECTION_2026_08_23.md` 가 무효화한 계기 결함이
이 태스크의 표를 바꾸는지 확인하기 위해서다. **바뀌지 않았다**(050 궤적에 `Failed to …` 본문 0건).

### trial 0 (fail)

| 종류 | 수 | 항목 |
|---|---|---|
| gold(변이) | 3 | `log_verification` · `submit_credit_limit_increase_request_7392` · `approve_credit_limit_increase_5847` |
| matched | 2 | `log_verification{Sofia Papadopoulos, 584f9c5d00, 1456 Astoria Boulevard…, sofia.p@outlook.com, 347-555-0387, 01/25/1986, 2025-11-14 03:40:00 EST}`(msg 14) · `approve_credit_limit_increase_5847{cc_584f9c5d00_gold, 584f9c5d00, 7500}`(msg 34) |
| **MISSING** | **1** | **`submit_credit_limit_increase_request_7392{credit_card_account_id: cc_584f9c5d00_gold, user_id: 584f9c5d00, requested_increase_amount: 2500}`** |
| WRONGARG | 0 | — |
| DUP | 0 | — |
| EXTRA | 0 | — |
| BLOCKED | 0 | — |

**WRONGARG 필드별 대조**: 성공한 `approve_…_5847` 의 세 필드는 gold 와 전부 일치
(`credit_card_account_id` `cc_584f9c5d00_gold` / `user_id` `584f9c5d00` / `new_credit_limit` `7500`).
⇒ 인자 문제 아님. **호출 자체의 부재 1건이 유일한 실패 단위.**

### trial 1 (pass)

`clean=true` — missing/wrongarg/extra/dup/blocked 전부 0 · matched 3
(`log_verification`(14) · `submit_…_7392{…,2500}`(40) · `approve_…_5847{…,7500}`(50)).

### 3세대 대조 (전부 정본 워커 재실행)

| 런 | t0 / t1 reward | 변이 |
|---|---|---|
| `bank_t7328_halfB_20260819r2` | 0.0 / 0.0 | **DUP** `approve_…_5847` (양 trial) |
| `bank_t7336_halfB_20260821b` | 0.0 / 1.0 | **MISSING** `approve_…_5847` / clean |
| **`bank_t7346_halfB_20260822`** | **0.0 / 1.0** | **MISSING `submit_…_7392`** / clean |

t7328 의 DUP 은 A-7⑵ 교정 **후에도 살아남는다**(재실행 확인) ⇒ 그 칸은 계기 결함이 아니다.
메모리 [[69]] 의 *"050 은 승인 **중복**으로 실패"* 는 t7336 이후로 **낡았다**(v1 §8 과 동일 결론).

---

## 2. 궤적 대조 — 두 trial 은 msg 30 까지 **도구 시퀀스가 동일**하다

| # | trial 0 | trial 1 |
|---|---|---|
| 4 | `verify_identity` | `verify_identity` |
| 8 | `get_user_information_by_email` | `get_user_information_by_name` |
| 10~14 | `verify_identity` → `get_current_time` → `log_verification` | 동일 |
| 16 | `get_credit_card_accounts_by_user` | 동일 |
| 18 | `check_cli_eligibility` (READ-FIRST deny) | 동일 |
| 20~26 | unlock/call `get_credit_limit_increase_history_4829` · `get_payment_history_6183{months:12}` | 동일 |
| 28 | `check_cli_eligibility` → **ELIGIBLE** | 동일(문면 바이트 동일) |
| 30 | `KB_search_bm25{"maximum increase for Gold Rewards Card"}` | `KB_search_bm25{"submit credit limit increase request"}` |
| 32 | **`unlock(approve_…_5847)`** | (산문) 금액 되물음 |
| 34 | **`call(approve_…_5847)`** ← 실패 확정 | `KB_search_bm25{"approve credit limit increase"}` |
| 36~50 | — (user STOP) | unlock approve → **PROCEDURE deny** → unlock/call `submit_…_7392` → disputes/replacement → unlock+call approve |

---

## 3. step-by-step 결정 지점 추적 (축자)

### ① [4]~[15] 신원확인 — 결함 0 (gold `050_0` matched)

[5] 우리 scaffold GET 축자:
> `NOT_VERIFIED — the account record has not been fetched yet, so nothing can be matched. Look the customer up with get_user_information_by_name/by_email/by_id, then call this tool again.`

[7] user: *"Of course — my email address is **sofia.p@outlook.com**."* → [8] 조회 → [11] 축자:
> `VERIFIED — 3 identity values match the account record (date_of_birth, phone_number, email); you may now call log_verification. Its time_verified argument must be the CURRENT time: FIRST call get_current_time, wait for its result, then copy that exact timestamp into time_verified`

→ [12] `get_current_time` → [13] *"The current time is 2025-11-14 03:40:00 EST."* → [14] `log_verification` **matched**.
**우리 층 발화 → 즉시 순응 → gold 일치.** 양(+).

### ② [18]~[19] READ-FIRST — 발화·효과 있음(양)

[18] 모델이 `consecutive_on_time_payments: "24"` 를 **지어냈다**(원장 근거 0). [19] 축자:
> `Error: [READ-FIRST] before checking CLI eligibility you must first read the inputs from the records. Missing required reads: get_credit_limit_increase_history, get_payment_history. … Their exact callable forms are: unlock_discoverable_agent_tool(agent_tool_name="get_credit_limit_increase_history_4829") then call_discoverable_agent_tool with that name; unlock_discoverable_agent_tool(agent_tool_name="get_payment_history_6183") then call_discoverable_agent_tool with that name.`

로그: `[T2_SG_REQREADS] check_cli_eligibility denied: missing reads ['get_credit_limit_increase_history', 'get_payment_history']` (1회).
모델은 [20]~[27] 로 **그대로 순응**했고 [28] 재호출에서 실측값 `"6"` 을 복사했다.
⚠ 다만 `months:12` 로 불렀다 — 이 게이트는 값은 안 본다. DB 축엔 무영향.

### ③ [29] 우리 층 판정문 — **선행 절차를 한 글자도 말하지 않고 "결정하라"고 명령** ★

```
CLI eligibility for cc_584f9c5d00_gold (Gold Rewards Card): ELIGIBLE - all tier requirements
(account age, cooldown, utilization, payment history) are satisfied. Decide with a TOOL CALL:
approve_credit_limit_increase, choosing the new limit within the tier's maximum increase cap
from the CLI policy documents - or deny only if a different documented requirement fails.
```

같은 A2 파일(`a2/banking_knowledge.specific.json`)의 `procedures[0] credit_limit_increase` 는
`decision` 노드 앞에 `submit_request`·`disputes`·`pending_replacement`·`payment_history`·`cooldown`
을 요구한다. 판정문은 그중 **무엇도 말하지 않는다** — 같은 층의 두 선언이 서로 다른 말을 한다([[55]]).
두 trial 모두 이 문면을 받았고 **두 trial 모두 그 직후 approve 로 향했다**.
(문면이 [[70]] 의 어느 쪽인지는 미측정 ⇒ **UNPROVEN**.)

### ④ [30]~[31] 검색 한 수 — ★**신규 사실: 절차 문서 자체가 문맥에 못 들어왔다**

v1 은 이 자리를 *"표면 분기"* 로만 적었다. 실제로는 **정보 가용성의 분기**다.
두 궤적 전문(全文)에 대한 부분문자열 전수 검사:

| 문자열 | trial 0 등장 msg | trial 1 등장 msg |
|---|---|---|
| `submit_credit_limit_increase_request_7392` | **없음 (0회, 38 msgs 전체)** | 31, 35, 38, 39, 40, 41 |
| `MUST be followed in the exact order` | **없음** | 31, 35 |
| `Step 1: Submit the CLI Request` | **없음** | 31, 35 |
| `doc_…logistics_007`(Processing CLI Approvals and Denials) | **없음** | 31, 35 |
| `doc_…logistics_004`(How can I request a CLI?) | **없음** | 31, 35 |
| `doc_…logistics_006`(CLI Payment History and Approval Criteria) | 31 | 31, 35 |
| `get_user_dispute_history_7291` | **없음** | 44~47 |
| `get_pending_replacement_orders_5765` | **없음** | 42~43, 48~49 |

trial 1 이 [31] 에서 받은 축자(= A2 `procedures._source` 가 인용하는 바로 그 문서):
> `## Purpose  This document outlines the step-by-step workflow for agents to process CLI requests, including verification and final decision. **These steps MUST be followed in the exact order listed.**`
> `## Step 1: Submit the CLI Request  Once the customer has confirmed a valid increase amount within their tier's limits, submit the request on their behalf using the **submit_credit_limit_increase_request_7392** tool. This creates a formal record of the request before eligibility checks are performed. Eligibility checks are internal and not exposed to customers, so the submission must happen first.`

trial 0 이 [31] 에서 받은 1위 축자:
> `1. Platinum Reserve Account: Business Credit Card APY Bonuses  ID: doc_business_savings_accounts_platinum_reserve_account_010  Score: 14.2883 … Business Gold Rewards Card: adds 0.5% to your APY`

⇒ **trial 0 의 결정 시점 문맥에는 "제출이 먼저"라는 사실도, 그 도구 이름도, 인자 스키마도
존재하지 않았다.** 그것을 아는 주체는 이 시점에 **우리 층뿐**이었고, 우리 층이 그것을 말할 수 있던
채널은 정확히 둘이었다 — `T2_PROCEDURE`(⑥에서 **평가조차 안 됨**)와 `T2_FOLLOWUP` unlock-hint
(⑦에서 **계산해 놓고 억제**). 둘 다 닫혔다.

동시각 로그:
```
[T2_SEARCH_AGENT] group=credit_cards · 문서 110(뺀 것 4 …) · now=2025-11-14 turn=32
[T2_DOCDECIDE] → 'Bronze Rewards Card'            ← 카드는 Gold Rewards Card 다(오결정)
[T2_SEARCH_ON_PROCEED] deny 아님 · 재료 254자 배달
[T2_VERDICT_SURFACE] surface decision=approve_credit_limit_increase_5847,deny_credit_limit_increase_5848,apply_credit_card_account_flag_6147
[T2_LEVER] T2_GATE_REGEN sim=task_050#s626729 verdict_surface
```

### ⑤ [32] unlock approve — **이름의 출처가 우리 층임이 코드로 확정된다**

[32] `unlock_discoverable_agent_tool{agent_tool_name:"approve_credit_limit_increase_5847"}`.
그 이름은 [32] 이전 대화(role∈{tool,user})에 **접미사 없는 형태(`approve_credit_limit_increase`,
msg 29)로만** 존재한다(전수 검사). 접미사 `_5847` 의 유일한 출처는 위 `T2_VERDICT_SURFACE` 줄이다.

★v2 신규 검증: `T2_UNLOCK_PROV` 는 **`_ap_regen` 함수 본문 안에만** 존재한다
(`t2_gate_patch.py:11192-11235`, 파일 전체에서 유일한 발화 지점). 따라서
```
[T2_UNLOCK_PROV] registry-provenanced (allow) tool=unlock_discoverable_agent_tool val=approve_credit_limit_increase_5847
```
가 찍혔다는 사실 자체가 **이 unlock 이 재생성 버퍼에서 태어났다는 기계적 증거**다.
대조: trial 1 은 `T2_UNLOCK_PROV` **0 라인** — 그쪽 unlock([36])은 원본 am 에 있었다.

### ⑥ [33]→[34] ★★**결정 지점** — 사임-경로 regen 이 절차 게이트 **밖에서** 승인을 커밋

[33] 은 unlock 성공 결과. 그 다음 턴의 **원본 am 에는 tool_calls 가 없었다**(산문 사임).
근거: `T2_UNCALLED_UNLOCK` 의 진입 술어가 `_resign`(tool_calls 없음 ∧ content 비어있지 않음)을
요구하는데(`t2_gate_patch.py:11548`) 그것이 발화했다.

```
[T2_UNCALLED_UNLOCK] surface approve_credit_limit_increase_5847
[T2_LEVER] T2_GATE_REGEN sim=task_050#s626729 uncalled_unlock (이후 무음)
[T2_LEVER] T2_STACK sim=task_050#s626729 uncalled_unlock|T2_UNCALLED_UNLOCK|표면화 (이후 무음)
```
(`(이후 무음)` = `t2_lever_beat._CAP=3` 의 (레버,sim)별 인쇄 상한 표시이지 레버 정지가 아니다 —
`t2_lever_beat.py:207-232`.)

보낸 문구 축자(`t2_gate_patch.py:11560-11568`):
> `Error: [UNLOCKED-NOT-CALLED] you unlocked approve_credit_limit_increase_5847 in this conversation and never called it. Unlocking only makes a tool available — it performs nothing. **If that step is still required, call it now with its arguments**; if it is not required, say plainly why you are not calling it.`

재생성 결과가 [34] 다:
> *"Let's proceed with increasing your credit limit to $7,500. I will now process the request."*
> `call_discoverable_agent_tool{agent_tool_name:"approve_credit_limit_increase_5847", arguments:"{\"credit_card_account_id\":\"cc_584f9c5d00_gold\",\"user_id\":\"584f9c5d00\",\"new_credit_limit\":7500}"}`

**이 sim 의 로그 168 라인에 `[T2_PROCEDURE]` 는 0 줄이다** — `deny` 도, `would-fire but suppressed
by=…` 도, `error (no-op)` 도 없다.

#### ★v2 신규 ①: 구조(들여쓰기/스코프) 확정 — "왜 0 줄인가"

| 코드 | 현재 줄 | 둘러싼 스코프 |
|---|---|---|
| `_procs = ((a2 or {}).get("procedures") …` | `t2_gate_patch.py:7505` | `apply_unified_regen`(6500) → `unified`(6739) → **`while True:`(7157)** |
| `for c in (am.tool_calls or []):` (절차 검사 루프) | `:7518` | 동상 |
| `print("[T2_PROCEDURE] deny …")` | `:7564` | 동상 |
| `def _ap_regen(fbtxt, tag, …)` | `:10972` | `unified`(6739) — **while 루프 밖(형제)** |
| `T2_UNCALLED_UNLOCK` 블록 → `_ap_regen(..., "uncalled_unlock")` | `:11548-11575` | `unified` — **while 루프 밖** |

⇒ 절차 게이트는 **while 루프 안**에서 `am.tool_calls` 만 훑는다. `_ap_regen` 은 루프가 끝난 뒤의
직선 코드이고, 그 반환 `am` 은 루프로 되돌아가지 않는다. 런 sha 에서 `_ap_regen` 의 재검사 목록은
`gate`(`_denied_calls`) · `T2_UNLOCK_NAME` · `T2_UNLOCK_PROV` **뿐**이었다.
같은 파일이 이 구멍을 이미 자백해 놓고 WEV 에만 이식했다 — `★T2_WEV_ROUNDS` 주석 축자:
> *"deny 후 regen된 호출은 같은 턴서 무검사 커밋"*

#### ★v2 신규 ②: 술어 오프라인 재생 — **deny 가 나온다**(짝 trial 의 라이브 문자열과 축자 동일)

정본 `t2_procedure.decide` 에 [34] **직전 상태**(`messages[:34]`)를 그대로 먹였다:

```
verdict = deny
missing = ['submit_request', 'disputes', 'pending_replacement']
notes[0] = "[PROCEDURE] The policy requires steps before 'approve_credit_limit_increase_5847'
that have not been done yet: submit_request, disputes, pending_replacement. Where the procedure
stands (2 of 7 done): [?] amount_within_tier_cap  [ ] submit_request -> submit_credit_limit_increase_request_7392
[x] cooldown  [ ] disputes -> get_user_dispute_history_7291  [ ] pending_replacement ->
get_pending_replacement_orders_5765  [x] payment_history  [ ] decision ->
approve_credit_limit_increase_5847/deny_credit_limit_increase_5848  Ready now:
submit_credit_limit_increase_request_7392. The tool for that step is
submit_credit_limit_increase_request_7392 and it has not been unlocked in this conversation - …"
```

라이브 trial 1 의 축자와 대조:
```
[sim=task_050#s373753] [T2_PROCEDURE] deny approve_credit_limit_increase_5847 missing=submit_request,disputes,pending_replacement
```
**missing 집합 축자 동일.** 라이브 로그의 `[T2_PROC_ABSENT] … done=2 of 7 done` (trial 0, 168라인 중
107행)도 재생값 `(2 of 7 done)` 과 일치 ⇒ **술어는 정상, 배관만 조용했다**([[55]] 0단계 통과).

#### ★v2 신규 ③: 폭발 반경 재현 — 런 전체에서 **1건뿐**

t7346 halfA+halfB 전 sim 의 **커밋된 assistant tool_call 960건**을 같은 술어로 전수 재판정:

```
committed assistant tool_calls scanned: 960
procedure-deny hits: 1
   (halfB, task_050, trial 0, msg 34, approve_credit_limit_increase_5847,
    missing=['submit_request','disputes','pending_replacement'], reward 0.0)
```
⇒ 이 구멍이 실제로 물린 자리는 이 한 곳이다. A-1 의 폭발 반경도 **이 런 기준 1 호출**.

### ⑦ [35]~[37] — 정답을 계산해 놓고 보내지 않았다

[35] *"Credit limit increase approved! - Account: cc_584f9c5d00_gold - Previous Limit: $5000.00 -
New Limit: $7500.00 - Increase: $2500.00"* → [36] 손님에게 완결 통보 → [37] user `###STOP###`.

같은 구간 로그:
```
[T2_FOLLOWUP] unlock-hint submit_credit_limit_increase_request_7392,get_user_dispute_history_7291,get_pending_replacement_orders_5765
[T2_FOLLOWUP] chain suppressed(th=2 resigns=1) after=['submit_credit_limit_increase_request_7392', 'check_cli_eligibility']
```
`unlock-hint` 인쇄는 `t2_gate_patch.py:11774` — 임계 검사(`:11783`)보다 **먼저** 실행되고, 힌트는
`_fb1` 에 붙기만 한 뒤 `:11783-11792` 의 `self._t2_fu_resigns(1) < _th(2)` 에서 `break` 로
**통째로 폐기**된다. 즉 **정확한 세 이름을 다 계산해 놓고 한 글자도 보내지 않았다**([[64]] 계열).

**"필요한 값이 그 시점 문맥에 실재했는가"**: `submit_credit_limit_increase_request_7392` 는 이 sim
38 메시지 어디에도 없다(§③ 전수표). 실재한 곳은 위 억제된 로그 1줄(우리 층 내부)뿐.
⇒ **모델이 베낄 값은 없었다.** 도달 경로는 있었다(`KB_search_bm25{"submit credit limit increase request"}`
한 수 — trial 1 이 증명).

---

## 4. trial 1 — 같은 자리에서 무엇이 달랐나 · 분기점 확정

- [30] 검색어가 갈린다. 모델 자백 축자:
  > *"It seems there was an error because the tool name was not properly discovered. Let me search the knowledge base to find the correct tool name for submitting the credit limit increase request."*
  → `KB_search_bm25{"submit credit limit increase request"}` → [31] `doc_…logistics_004` + `_006` + **`_007`** 전문 배달(17,186자).
  (직전 로그는 trial 0 과 **동일한** `[T2_RESOLVE] deny tool=unlock_discoverable_agent_tool arg=agent_tool_name reason=operator-fab` → `[T2_RESOLVE_CAP] 리셋(실행): 새 실행 ['KB_search_bm25']` 이다 — 우리 층 발화는 같았다.)
- [32] 금액 되물음(산문) → [33] user: *"I'd like to **increase my limit by $2,500** if possible."* (오도 0)
- [34] `KB_search_bm25{"approve credit limit increase"}` → [35] `_006`/`_007` 재배달
- [36] `unlock(approve_…_5847)` — **원본 am**(그 sim 에 `T2_UNLOCK_PROV` 0 라인)
- 다음 턴 모델이 `call(approve_…_5847)` 시도 → **원본 am 이므로 절차 게이트가 정상 평가**:
```
[T2_PROCEDURE] deny approve_credit_limit_increase_5847 missing=submit_request,disputes,pending_replacement
[T2_STACK] audit route=[…] chose=[('proc','approve_credit_limit_increase')] differs=True suppressed=['T2_VERDICT_SURFACE']
[T2_PIN_READ] pinned call_discoverable_agent_tool(agent_tool_name=submit_credit_limit_increase_request_7392)
[T2_LEVER] T2_PIN_READ sim=task_050#s373753 submit_credit_limit_increase_request_7392
```
→ [38] `unlock(submit_…_7392)` → [40] `call(submit_…_7392{cc_584f9c5d00_gold, 584f9c5d00, 2500})`
**matched** → [42]~[49] disputes·pending_replacement 실행 → [50] `unlock+call approve` 한 턴 동시 발행 → **1.0**.

### 분기점 (2층)

| 층 | trial 0 | trial 1 |
|---|---|---|
| **(a) 정보** [30]→[31] | 검색이 APY 문서 회수 ⇒ **순서 규칙·도구 이름·인자 스키마가 문맥에 0** | `_004`+`_007` 회수 ⇒ *"MUST be followed in the exact order"* · `submit_…_7392` · 인자 스키마 **전부 문맥 안** |
| **(b) 배관** [33]→[34] vs [36]→[38] | approve 호출이 **`_ap_regen` 버퍼**에서 태어남 ⇒ `T2_PROCEDURE` **0 라인** | approve 호출이 **원본 am** ⇒ `deny` 1회 → `T2_PIN_READ` 5회 → 선행 이행 |

(b)가 **결정적**이다: (a)만으로도 trial 1 은 이름을 알았지만, 실제로 submit 을 **먼저** 하게 만든
것은 절차 deny + PIN 이다(그 전 [36] 에서 모델은 곧장 approve 로 갔다).
(a)는 **A-1 이후의 회복 가능성**에 관한 조건이다(§7 반사실).

---

## 5. 레버 발화표 (요청 목록 전수 · fail↔pass)

| 레버 | fail / pass 라인수 | 판정 | 근거(축자) |
|---|---|---|---|
| **`T2_PROCEDURE`** | **0** / 1 | **미발화(음·치명)** | pass 만 `deny approve_… missing=submit_request,disputes,pending_replacement`. fail 은 `would-fire but suppressed`·`error` 조차 0 ⇒ **평가 자체가 없었다**. 오프라인 재생 = deny(§3⑥) |
| **`T2_UNCALLED_UNLOCK`** | 2 / 1 | **오발화(음·치명)** | fail: `surface approve_credit_limit_increase_5847` → *"call it now with its arguments"* 를 **선행 3개 미충족 시점에** 발사, sim당 1회 예산(`self._t2_uncalled_fired`) 소진. pass: 모든 선행이 끝난 뒤(로그 221행) 발화해 무해 |
| **`T2_PIN_READ`** | 1(무효 `PIN_READ_STEPS`) / 7 | **미발화(음)** | fail 은 `[T2_PIN_READ_STEPS] no read-only target (reads 0 of 1 ready)` 뿐. pass 는 `pinned call_discoverable_agent_tool(agent_tool_name=submit_…_7392)` — **`T2_PROCEDURE` deny 의 종속물**이라 게이트가 죽으면 같이 죽는다 |
| **`T2_FOLLOWUP`** | 8 / 15 | **계산 성공·전달 실패(음)** | `unlock-hint submit_…_7392,get_user_dispute_history_7291,get_pending_replacement_orders_5765`(`:11774`) 직후 `chain suppressed(th=2 resigns=1)`(`:11791`). pass 는 뒤에 `chain fired(followup_decision)` |
| **`T2_SEARCH_AGENT`** | 6 / 6 | **오결정 + 축 소진 침묵(음)** | fail: `[T2_DOCDECIDE] → 'Business Bronze Rewards Card'` · `→ 'Bronze Rewards Card'` — **둘 다 Gold Rewards Card 가 아니다**. 이후 `요청 축 business_credit_cards,credit_cards 모두 처리됨 — 침묵` **2회**. 배달 재료(263자·254자)가 궤적에 안 남아 해악은 **미측정** |
| **`T2_SEARCH_REARM`** | **0** / 2 | **미발화(음)** | pass 만 `group=credit_cards 신규 대상 gold_rewards_card (기배달 bronze_rewards_card) — 소진 해제·문서 델타` + `델타 배달 6973자 (문서 5) turn=50`. fail 은 채널이 닫힌 채 종료 |
| **`T2_CLAIMPROV`** | 10 / 33 | 발화·무해 | fail 최종 `window hit(resign) claims=3 unbacked=0 pending=0 unb_p=0 []` — 승인이 실제로 성공했으므로 산문 날조 0 |
| **READ-FIRST(`T2_SG_REQREADS`)** | 1 / 1 | **발화·효과 있음(양)** | 접미사 이름 2개 지목 → 즉시 순응 · 실측 `6` 복사. `months:12` 는 못 막음 |
| **`T2_VERDICT_SURFACE`** | 2 / 3 | 발화·양면 | `surface decision=approve_…_5847,deny_…_5848,apply_…_6147` — 정답 이름을 줬으나 **선행 절차를 침묵**. pass 에서는 `[T2_STACK] … suppressed=['T2_VERDICT_SURFACE']` 로 절차에 밀린다 |
| **`T2_UNLOCK_PROV`**(A5/OL-01 수리) | 1 / 0 | **발화·정당(양)·그러나 입구** | `registry-provenanced (allow) … val=approve_credit_limit_increase_5847` — t7336 자기차단 재발 0 |
| **`T2_SG_DOCS`** | 0 / 0 | 미발화 | 168/258 라인에 0회 |
| **`T2_DEMANDED_STEP`** | 0 / 0 | 미발화 | 0회 |
| **`FAB_STRIP`** | 0 / 0 | 미발화 | 0회 |
| **`T2_ARG_PRODUCERS`** | 0 / 0 | 미발화 | 0회 |
| **`T2_REQUIRE_DOC_DELIVER`** | 0 / 0 | 미발화 | 0회 |
| `T2_GATE_REGEN`(참고) | 3 / 3 | 발화 | fail 3회 중 2회(`verdict_surface`·`uncalled_unlock`)가 §7-① 사슬. 인쇄 상한 3 이므로 **실제 횟수는 ≥3** |

**직전 런 이후 들어간 수리가 이 궤적에 개입했는가 · 개입하고도 왜 못 샀나**
개입했다. `T2_UNLOCK_PROV` registry-allow(A5/OL-01)가 **정확히 의도대로** 발화해 t7336 의 실재-이름
자기차단을 없앴다. 못 산 이유: 그 통과가 `unlock → (사임) → UNCALLED_UNLOCK regen → call` 사슬을
열었고, 그 사슬의 마지막 호출이 **절차 게이트의 스코프 밖**에서 커밋됐다. t7336 §8-1 이 [[70]]
로 예고한 *"판다"* 항목의 더 나쁜 판본 — 이름은 옳았고 **시점이 절차 위반**이었다.
⚠**A-1 은 이 런에 없다**(런 sha `ee18d797`; A-1 은 2026-08-23 커밋으로 그 뒤에 들어갔다).

---

## 6. 선행 판정과의 대조 — **같은 원인인가 달라졌는가**

| 선행 문서 | 그때 판정 | v2 판정 |
|---|---|---|
| `tasks__20260822/TASK_050.md` §7-① (`_ap_regen` 산출물이 `T2_PROCEDURE` 재평가를 안 받음) | our_layer CONFIRMED | **같은 원인. 재확인 + 강화** — v1 은 로그 부재 + 짝 대조로 추론했고, v2 는 ⑴ 스코프(while-루프 안/밖)를 구조로 확정 ⑵ 술어를 오프라인 재생해 `missing` 축자 동일 확인 ⑶ 960 호출 전수 재판정으로 반경 1건 확정 |
| 동 §7-② (`T2_UNCALLED_UNLOCK` 술어가 절차를 조회 안 함) | our_layer CONFIRMED(증폭 인자) | **같은 원인.** 현재 트리에서도 `_idle12 = sorted(_unl12 - _called12)`(`:11555`)뿐 — **미수리** |
| 동 §7 보조 (`T2_FOLLOWUP` 임계 억제) | our_layer(분기 원인 아님) | **같은 원인. 현재 트리 `:11774`(힌트 인쇄) / `:11783-11792`(억제) 그대로 — 미수리** |
| 동 §7 보조 (ELIGIBLE 문면이 선행 침묵) | UNPROVEN | **UNPROVEN 유지** |
| 동 §5-a (*"표면 분기 = 검색어 한 수"*) | 표면 | **부분 정정.** 검색어는 표면이 아니라 **정보 가용성 분기**다 — trial 0 문맥엔 순서 규칙·도구 이름·인자 스키마가 **0회** 등장(§3④ 전수표). 결정적 분기가 (b) 배관인 것은 v1 이 옳다 |
| 동 §7 model 항 (*"[33]→[34] 원본 응답에서 산문 사임"*) | model 보조 | **유지.** 다만 그 사임이 **옳은 행동이었을 수 있다** — 그 시점에 순서 규칙이 문맥에 없었고 금액도 미확정이었다. 원문 텍스트는 regen 이 대체해 남지 않았고 t7346 은 fb 사이드카·trace 가 없다 ⇒ **UNPROVEN** |
| `FAILURE_MASTER__20260822.md` §축 E · §6.3-1 · §7 A-1 | 050#0 = 축 E 최강 · A-1 우선 | **전면 동의 + 독립 재현**(오프라인 재생·960 전수 census 둘 다 같은 수) |
| 동 §8 *"승격 8건 중 reward 반사실이 증명된 것은 사실상 0건 · 050#0 조차 n=1 짝 대조"* | 자기 경고 | **v2 가 더 강하게 지지**(§7 반사실 참조) |
| `MASTER_DUP_CORRECTION_2026_08_23.md` (deny_kind `Failed to …`) | 계기 결함 · 표 재계산 | **050 무영향** — 궤적에 `Failed to …` 0건, 교정본으로 재실행해도 t7346/t7336/t7328 표 불변 |
| `ATTRIBUTION_CORRECTION_2026_08_23.md` | 094 귀속 정정 | **050 언급 없음 · 무관** |
| 메모리 [[69]] *"050 은 승인 중복으로 실패"* | DUP | **낡음.** t7328=DUP → t7336=MISSING(approve) → t7346=MISSING(submit) |

---

## 7. 원인 확정

### 주 원인 — **our_layer** (코드 경로 지목 · CONFIRMED)

**OL-1 (강·CONFIRMED) — `_ap_regen` 산출 호출이 `T2_PROCEDURE` 스코프 밖에서 커밋된다**
- 게이트: `t2_gate_patch.py:7505`(`_procs`) · `:7518`(`for c in (am.tool_calls or [])`) · `:7564`(deny 인쇄) — 전부 `unified()`(`:6739`) **내부 `while True:`(`:7157`)** 안.
- 생성: `t2_gate_patch.py:10972 def _ap_regen(...)` — `unified()` 본문의 **while 루프 형제**(밖). 런 sha 의 재검사 목록 = `gate`(`_denied_calls`) · `T2_UNLOCK_NAME` · `T2_UNLOCK_PROV`(`:11192-11235`)뿐.
- 호출부: `T2_UNCALLED_UNLOCK`(`:11548-11575`) → `_ap_regen(..., "uncalled_unlock")` → `am = _newS`.
- 실측: trial 0 로그 168라인에 `[T2_PROCEDURE]` **0 줄** ↔ trial 1 동일 호출에 `deny … missing=submit_request,disputes,pending_replacement`. 같은 sha·같은 A2·같은 태스크.
- 오프라인 재생: `t2_procedure.decide(messages[:34])` → `verdict=deny`, `missing=['submit_request','disputes','pending_replacement']` — **짝 trial 라이브 문자열과 축자 동일**.
- 반증 검토: ⒜ cap 소진? `_t2_proc_deny=0` 이고 cap 이면 `would-fire but suppressed by=cap` 이 찍힌다(`:7536`) — 0줄이므로 아님. ⒝ 선점(`_pblocker`)? 같은 자리에서 `by=<lever>` 로 찍힌다 — 아님. ⒞ 예외? `[T2_PROCEDURE] error (no-op)`(`:7571`) — 0줄. ⒟ `_procs` 가 비었나? 같은 sim 에서 `[T2_PROC_ABSENT] surface credit_limit_increase quiet>=3 done=2 of 7 done` 이 찍혔다(선언 로드 확인 · 인접 블록). ⇒ **CONFIRMED**.

**OL-2 (준강·CONFIRMED, 단독 충분원인 아님) — `T2_UNCALLED_UNLOCK` 술어가 `procedures` 를 조회하지 않는다**
- `t2_gate_patch.py:11548-11575`, 술어 전부 = `_idle12 = sorted(_unl12 - _called12)`(`:11555`). 절차 선언 참조 0.
- A2 는 그 시점에 approve 가 아직 불가함을 이미 선언한다: `a2/banking_knowledge.specific.json` `procedures[0] credit_limit_increase` 의 `decision` 노드가 `requires=[cooldown, disputes, pending_replacement, payment_history]`, `submit_request` 노드가 `requires=[amount_within_tier_cap]`, `enforce=true`, `_quote_order="These steps MUST be followed in the exact order listed."`
- 그래서 문구가 *"call it now with its arguments"* 로 나갔고 sim당 1회 예산을 **선행 미충족 시점**에 태웠다.
- 반증: 문구에 *"if it is not required, say plainly why you are not calling it"* 탈출구가 있다 ⇒ 모델이 안 부를 수도 있었다. 그러나 [[64]] 기준 이 문면은 "무엇을 하면 풀리나"를 **틀린 값**으로 채웠고, OL-1 이 없었다면 그 오유도조차 게이트가 되돌렸을 것이다 ⇒ **OL-1 의 증폭 인자**.

**OL-3 (확정·분기 원인 아님) — `T2_FOLLOWUP` 이 정답 3개를 계산해 놓고 임계로 폐기**
- `t2_gate_patch.py:11774` 에서 `unlock-hint submit_credit_limit_increase_request_7392,…` 를 인쇄하고 `_fb1` 에 붙인 직후, `:11783`(`_th=T2_FOLLOWUP_RESIGN_TH` 기본 2) / `:11791`(억제 인쇄) / 그 다음 `break` 로 **문장 전체를 폐기**. 두 trial 모두 억제됐으므로 분기 원인은 아니나, trial 0 의 **마지막 구제선**이었다.

**OL-4 (UNPROVEN) — `check_cli_eligibility` ELIGIBLE 문면이 선행 절차를 침묵**
같은 A2 의 `procedures` 가 요구하는 `submit_request` 를 한 글자도 말하지 않고 *"Decide with a TOOL
CALL: approve_credit_limit_increase"* 라고 명령한다([[55]] 문구 모순). 두 trial 공통이라 분기 원인은
아니고, 문면 수정의 ± 는 미측정 ⇒ **UNPROVEN**.

**OL-5 (UNPROVEN) — `T2_SEARCH_AGENT` 축 결정이 카드 등급을 틀린다**
`[T2_DOCDECIDE] → 'Business Bronze Rewards Card'` / `→ 'Bronze Rewards Card'` (카드는 Gold Rewards
Card). 배달 재료가 궤적에 남지 않아 해악 **미측정**. 두 trial 동일이라 분기 원인 아님.

### 보조 원인 — **model**

- [30] 검색어 선택 — trial 1 은 같은 자리에서 `"submit credit limit increase request"` 를 골라
  `_004`/`_007` 을 회수했다. 부하가 아니라 **선택**이다(같은 문맥·같은 온도 0).
- [18] `consecutive_on_time_payments:"24"` 지어냄 — 우리 게이트가 즉시 교정(무해).
- [26] `months:12` — 정책 허용값은 6·3. DB 축 무영향.
- 금액을 손님에게 **묻지 않고** 스스로 정함(trial 1 은 [32] 에서 물었다).
- [33]→[34] 원본 응답에서 도구를 안 부르고 산문 사임 — OL-1 의 **트리거**. 다만 그 사임이 부적절했다는
  증거는 없다(원문 소실 · fb 사이드카/trace 부재) ⇒ **UNPROVEN**.

### **env** — 결함 아님
BM25 가 `"maximum increase for Gold Rewards Card"` 에 APY 문서를 상위로 준 것은 검색기의 정상 거동이고,
정답 문서(`doc_…logistics_004/006/007`)는 코퍼스에 실재하며 다른 질의로 도달한다(trial 1 증명).
도구 실행·결과 전부 정상. **[[68]] 제외 대상 아님.**

### **user_sim** — 무관
[1] 요청 · [3]/[7] 식별자 제공 · [33](trial 1) 금액 제공 · [37] 정상 종료. 오도 0.
trial 0 에서 금액을 못 받은 것은 **에이전트가 묻지 않았기 때문**이다([[21]]: user-sim 요인으로 종결 금지).

---

## 8. ★반사실 경고 — "A-1 이면 이 sim 이 통과한다"는 **증명되지 않았다**

FAILURE_MASTER §8 의 자기 경고를 v2 가 더 강하게 지지한다.

A-1 이 켜졌다면 [34] 의 approve 는 `[T2_PROCEDURE] regen-deny (tag=uncalled_unlock)` 를 받고
`notes[0]` 이 `submit_credit_limit_increase_request_7392` 이름 + unlock 힌트를 배달했을 것이다(§3⑥ 재생).
그 다음에 **write 가 실제로 나오려면** 두 가지가 더 필요하다:

1. **`requested_increase_amount = 2500`** — trial 0 에서 이 값은 손님이 말한 적이 없고 문맥에 없다.
   모델은 `new_credit_limit 7500`(=5000+2500)을 스스로 정했으므로 2500 은 **파생 가능**하지만, 그것은
   모델의 한 수이지 우리 층이 보증하는 값이 아니다. (gold 는 `2500`.)
2. **`disputes` · `pending_replacement` 두 read** — 이름은 deny notes 가 준다(재생 확인).

⇒ A-1 은 **기전(게이트 우회)** 을 닫는 것이 CONFIRMED 이고, **이 sim 의 reward 회복은 n=1 짝 대조에
의한 추정**이다. 짝 trial 1 은 A-1 없이도 통과했지만 그쪽은 `_004`/`_007` 문서와 손님이 말한 2500 을
**둘 다** 갖고 있었다(§4 분기표 (a)). trial 0 은 둘 다 없었다.
**[[57]] 부정통제 의무**: `T2_PROC_REGEN=1↔0` × `T2_PROCEDURE=1↔0` 4칸 + `no-action` 계수.

---

## 9. 처방 후보 (제안만 · 코드 수정 없음 · [[70]] A/B 의무)

| # | 처방 | 근거 | 상태 | **무엇을 파는가** |
|---|---|---|---|---|
| **1** | `_ap_regen` 반환 호출에 절차 게이트 1회 재평가 | OL-1 | **이미 들어감**(A-1 · `t2_gate_patch.py:11088-11176`) — 이 런에는 없었다 | 사임-경로 regen 이 절차 deny 로 접히면 그 턴이 **빈손**(over-action↓ / no-action↑). cap 은 메인 경로와 공유(`_t2_proc_deny`) |
| **2** | `T2_UNCALLED_UNLOCK` 술어에 절차 필터 — `_idle12`(`:11555`)에서 *미충족 선행을 가진 노드의 도구* 제외 | OL-2 | **미적용** | 053형(선행 완료·미호출)은 살고, 선행 미충족 상태의 넛지 1발이 사라진다. 값은 전부 `procedures` 선언에서 나오므로 새 결정론 0([[62]]·[[22]] 닫힌 술어) |
| **3** | `T2_FOLLOWUP` 의 `unlock-hint`(사실 배달)를 체인 임계에서 분리 — 명령만 억제, 이름은 배달 | OL-3 | **미적용** | 힌트 발화 빈도↑ → 슬롯 경합·컨텍스트 압박. [[64]] 계열(이름 없는 억제 금지) |
| **4** | `check_cli_eligibility` ELIGIBLE 문면에 선행 절차 1문장 | OL-4 | 미적용 · **측정 먼저** | A2 문구 저작이므로 정책 축자 근거 필수([[23]]). 1·2 만으로 회복되는지 먼저 잴 것 |
| **5** | `T2_SEARCH_AGENT` 배달 재료를 사이드카에 기록(계측만) | OL-5 | 미적용 | 거동 0. 지금은 배달 내용이 궤적에 안 남아 `T2_DOCDECIDE` 오결정의 해악을 **잴 수가 없다** |

---

## 10. 한 줄 결론

**trial 0 은 `submit_credit_limit_increase_request_7392` 하나를 빠뜨려 DB 해시가 갈렸고, 그 누락은
우리 층이 만들었다.** 우리 층은 (ⓐ `T2_VERDICT_SURFACE` 로 접미사 이름을 주고 → ⓑ `T2_UNCALLED_UNLOCK`
로 *"지금 인자를 실어 부르라"* 고 명령해 승인 호출을 **만들어 내고** → ⓒ 그 호출이 `_ap_regen`
(`t2_gate_patch.py:10972`, `unified()` while-루프 **밖**)에서 태어난 탓에 `T2_PROCEDURE`
(`:7505-7564`, while-루프 **안**)의 **평가를 한 번도 받지 않게** 하고 → ⓓ 마지막으로 `T2_FOLLOWUP`
이 정확한 세 이름을 계산해 놓고 임계(`:11783`)로 폐기했다) — **네 채널이 전부 같은 방향으로 틀렸다.**
술어는 정상이었다: 같은 상태를 정본 `t2_procedure.decide` 에 오프라인으로 먹이면
`deny missing=['submit_request','disputes','pending_replacement']` 가 나오고, 그것은 짝 trial 1 이
**라이브로 받아 1.0 을 만든 문자열과 축자 동일**하다. 런 전체 960 커밋 호출 중 이 술어가 무는 자리는
**이 한 곳뿐**이다.
