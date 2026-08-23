# t7346 task_050 per-step 포렌식 — 2026-08-23

- 런: `bank_t7346_halfB_20260822` (results/log 전부 로컬 gz 무접촉 읽기).
- sim 2개: **trial 0 = `task_050#s626729` reward 0.0**(38 msgs · `user_stop`) / **trial 1 = `task_050#s373753` reward 1.0**(55 msgs · `user_stop`).
- 로그 전수: `[sim=task_050#s626729]` **168 라인** / `[sim=task_050#s373753]` **258 라인**.
- 변이 = 정본 `t2_forensic.mutation_diff` 만 사용(손 비교기 0 · C583ⓐ). 인용은 전부 축자.
- **수리 실행 없음**(코드 무수정 · 처방은 후보로만). gold(`reward_info`)는 진단용으로만 읽었다([[23]]).
- 대조군: `bank_t7328_halfB_20260819r2`(sha 상이) · 선행 보고서 `t7336_tasks/T7336_TASK_050.md`.

---

## 0. 채점 축 — **DB**(ACTION 아님)

```
trial 0: reward_basis=["DB"]  reward_breakdown={"DB":0.0}  db_check={"db_match":false,"db_reward":0.0}
trial 1: reward_basis=["DB"]  reward_breakdown={"DB":1.0}  db_check={"db_match":true, "db_reward":1.0}
nl_assertions=null · communicate_checks=null · env_assertions=[]
```

⇒ **DB-해시 축**이므로 `action_checks` 는 진단 보조일 뿐 성적이 아니다([[69]]).
그 증거가 이 태스크 안에 있다: **`050_10`(`get_payment_history_6183{…, months:3}`)은 두 trial 모두 `action_match=false` 인데 trial 1 은 reward 1.0** — `months` 는 상태를 안 바꿔 해시에 안 남기 때문이다. ACTION 축으로 읽으면 거짓 결론이 나온다(C583ⓖ).

| trial | action_checks unmatched | reward |
|---|---|---|
| 0 | `050_1`~`050_6`(submit unlock/call · disputes · replacement) · `050_10` | 0.0 |
| 1 | `050_10` 뿐 | 1.0 |

---

## 1. 변이표 (`t2_forensic.mutation_diff` · mutating tools 정본 44종)

### trial 0 (fail)

| 종류 | 수 | 항목 |
|---|---|---|
| matched | 2 | `log_verification{Sofia Papadopoulos, 584f9c5d00, …, 2025-11-14 03:40:00 EST}`(msg 14) · `approve_credit_limit_increase_5847{cc_584f9c5d00_gold, 584f9c5d00, 7500}`(msg 34) |
| **MISSING** | **1** | **`submit_credit_limit_increase_request_7392{credit_card_account_id: cc_584f9c5d00_gold, user_id: 584f9c5d00, requested_increase_amount: 2500}`** |
| WRONGARG | 0 | — (승인 write 는 세 필드 전부 gold 와 일치: `cc_584f9c5d00_gold` / `584f9c5d00` / `7500`) |
| DUP | 0 | — |
| EXTRA | 0 | — |
| BLOCKED | 0 | — |

### trial 1 (pass)

`clean = true` — missing/wrongarg/dup/extra/blocked 전부 0, matched 3 (`log_verification`(14) · `submit_…_7392`(40) · `approve_…_5847`(50)).

⇒ **단일 실패 단위 = MISSING 1건(요청 제출 write)**. WRONGARG 0 이므로 필드별 대조 대상 없음(호출 자체가 부재).
⚠**기전이 또 이동했다**: t7328 = DUP(approve 2회) → t7336 = MISSING(approve) → t7346 = **MISSING(submit)**. 승인은 이제 성공한다.

---

## 2. 대조군

| 런 | trial | reward | 변이 |
|---|---|---|---|
| `bank_t7328_halfB_20260819r2` | 0 / 1 | 0.0 / 0.0 | **DUP** `approve_credit_limit_increase_5847` (2/2) |
| `bank_t7336_halfB_20260821b` | 0 / 1 | 0.0 / 1.0 | **MISSING** `approve_…_5847` / clean |
| **`bank_t7346_halfB_20260822`** | **0 / 1** | **0.0 / 1.0** | **MISSING `submit_…_7392`** / clean |

t7336 §8 처방 1(**`T2_UNLOCK_PROV` 출처에 env 레지스트리 추가**, A5/OL-01)은 **적용됐고 실물로 발화했다**:

```
[sim=task_050#s626729] [T2_UNLOCK_PROV] registry-provenanced (allow) tool=unlock_discoverable_agent_tool val=approve_credit_limit_increase_5847
```

t7336 을 죽이던 자기차단(`deny unprovenanced name … val=approve_credit_limit_increase_5847`)은 **재발 0**. 그 수리는 샀다. 그러나 아래 §7 이 보여 주듯 **그 통과가 새 결손의 입구가 됐다**.

---

## 3. trial 0 step-by-step — 결정 지점 추적 (축자)

메시지 인덱스 = results.json `messages` 배열 인덱스.

#### ① [4]~[15] 신원확인 — 정상 (gold `050_0` matched)
[4] `verify_identity{DOB, phone}` → [5] 우리 scaffold GET deny 축자:
> *"NOT_VERIFIED — the account record has not been fetched yet, so nothing can be matched. Look the customer up with get_user_information_by_name/by_email/by_id, then call this tool again."*

[6] 에이전트가 이메일 요청 → [7] user *"my email address is **sofia.p@outlook.com**"* → [8] `get_user_information_by_email` → [11] 축자 *"VERIFIED — 3 identity values match … you may now call log_verification. Its time_verified argument must be the CURRENT time: FIRST call get_current_time"* → [12] `get_current_time` → [14] `log_verification` **matched**. 결함 0.

#### ② [18]~[19] READ-FIRST 게이트 — 발화·효과 있음(양)
[18] `check_cli_eligibility{…, consecutive_on_time_payments: "24", …}` (모델이 24 를 **지어냄**) → [19] 축자:
> `Error: [READ-FIRST] before checking CLI eligibility you must first read the inputs from the records. Missing required reads: get_credit_limit_increase_history, get_payment_history. … Their exact callable forms are: unlock_discoverable_agent_tool(agent_tool_name="get_credit_limit_increase_history_4829") then call_discoverable_agent_tool with that name; unlock_discoverable_agent_tool(agent_tool_name="get_payment_history_6183") then call_discoverable_agent_tool with that name.`

모델은 **그대로 따랐다**([20]~[27]) → [28] 재호출에서 `consecutive_on_time_payments: "6"` 실측값 복사.
로그: `[T2_SG_REQREADS] check_cli_eligibility denied: missing reads ['get_credit_limit_increase_history','get_payment_history']` 1회.
⚠단, `months` 는 **12** 로 호출했다([26]) — A2 `payment_history.param_requirement{months one_of [6,3]}`(`a2/banking_knowledge.specific.json:4341-4358`) 위반이지만 이 게이트는 그 값을 막지 않았다. DB 축엔 무영향(→ `050_10` unmatched 인데 trial 1 도 통과).

#### ③ [29] 우리 층 판정문 — **submit 을 건너뛰라고 읽히는 문면** ★
```
CLI eligibility for cc_584f9c5d00_gold (Gold Rewards Card): ELIGIBLE - all tier requirements
(account age, cooldown, utilization, payment history) are satisfied. Decide with a TOOL CALL:
approve_credit_limit_increase, choosing the new limit within the tier's maximum increase cap
from the CLI policy documents - or deny only if a different documented requirement fails.
```
같은 A2 파일의 `procedures.credit_limit_increase` 는 `decision`(approve/deny) 앞에 `submit_request`·`disputes`·`pending_replacement`·`payment_history`·`cooldown` 을 요구한다(`:4365-4374`). **판정문은 그 선행을 한 글자도 말하지 않고 곧장 "Decide with a TOOL CALL: approve_…" 라고 명령한다.** 두 trial 모두 이 문면을 받았고 **두 trial 모두 그 직후 approve 로 향했다** — 차이는 뒤에서 절차 게이트가 되돌렸는지 여부뿐이다(§5).

#### ④ [30]~[31] 검색 한 수 — BM25 가 엉뚱한 문서를 반환
[30] `KB_search_bm25{query:"maximum increase for Gold Rewards Card", k:10}` → [31] 1위 축자:
> *"Platinum Reserve Account: Business Credit Card APY Bonuses / ID: doc_business_savings_accounts_platinum_reserve_account_010 / Score: 14.2883 … Business Gold Rewards Card: adds 0.5% to your APY"*

**요청한 것(CLI 티어 상한)과 무관한 문서**다. trial 1 은 같은 자리에서 `query:"submit credit limit increase request"` 를 던져 `doc_credit_cards_credit_card_account_logistics_004`("How can I request a CLI?", Score 16.5153)를 1위로 받았다(§5). 검색어 한 수가 갈렸다 — 이것이 **표면상의** 분기점이다(진짜 분기는 §5-b).

로그 동시각:
```
[T2_SEARCH_AGENT] group=credit_cards · 문서 110(뺀 것 4 …) · now=2025-11-14 turn=32
[T2_DOCDECIDE] → 'Bronze Rewards Card'          ← 카드는 Gold Rewards Card 다(오결정)
[T2_SEARCH_ON_PROCEED] deny 아님 · 재료 254자 배달
[T2_VERDICT_SURFACE] surface decision=approve_credit_limit_increase_5847,deny_credit_limit_increase_5848,apply_credit_card_account_flag_6147
[T2_LEVER] T2_GATE_REGEN sim=task_050#s626729 verdict_surface (이후 무음)
[T2_UNLOCK_PROV] registry-provenanced (allow) tool=unlock_discoverable_agent_tool val=approve_credit_limit_increase_5847
```

#### ⑤ [32] unlock approve — **우리 층이 이름을 주고 우리 층이 통과시켰다**
[32] `unlock_discoverable_agent_tool{agent_tool_name:"approve_credit_limit_increase_5847"}`.
**이 이름은 [32] 이전 대화(role∈{tool,user})에 한 번도 없다** — [29] 는 접미사 없는 `approve_credit_limit_increase`, [31] 은 APY 문서. 출처는 우리 층의 `T2_VERDICT_SURFACE` 재생성 버퍼(위 로그)다. 그리고 `T2_UNLOCK_PROV` 가 **레지스트리 출처**로 허용했다(A5/OL-01 수리의 실물 동작).
⇒ **이 호출은 `_ap_regen` 재생성에서 태어났다.** `_ap_regen`(`t2_gate_patch.py:10762`)의 재검사는 `gate`(`_denied_calls`, :10863-10879)·`T2_UNLOCK_NAME`·`T2_UNLOCK_PROV`(:10884-10945) **뿐**이고, 절차 게이트는 없다.

#### ⑥ [33]→[34] ★**결정 지점** — 사임-경로 regen 이 절차 게이트 밖에서 approve 를 커밋
[33] 은 unlock 성공 결과. 그 다음 모델의 **원본 응답에는 tool_calls 가 없었다**(사임 산문). 그래서:
```
[T2_UNCALLED_UNLOCK] surface approve_credit_limit_increase_5847
[T2_LEVER] T2_GATE_REGEN sim=task_050#s626729 uncalled_unlock (이후 무음)
[T2_LEVER] T2_STACK sim=task_050#s626729 uncalled_unlock|T2_UNCALLED_UNLOCK|표면화 (이후 무음)
```
`T2_UNCALLED_UNLOCK`(`t2_gate_patch.py:11249-11275`)이 보낸 문구는 축자:
> *"Error: [UNLOCKED-NOT-CALLED] you unlocked approve_credit_limit_increase_5847 in this conversation and never called it. Unlocking only makes a tool available — it performs nothing. **If that step is still required, call it now with its arguments**; if it is not required, say plainly why you are not calling it."*

재생성 결과가 [34] 다:
> *"Let's proceed with increasing your credit limit to $7,500. I will now process the request."*
> `call_discoverable_agent_tool{agent_tool_name:"approve_credit_limit_increase_5847", arguments:"{\"credit_card_account_id\":\"cc_584f9c5d00_gold\",\"user_id\":\"584f9c5d00\",\"new_credit_limit\":7500}"}`

**이 sim 전체에서 `[T2_PROCEDURE]` 는 0 라인이다** — deny 도, `would-fire but suppressed by=…` 도, `error (no-op)` 도 없다. 코드가 그 이유를 설명한다: 절차 블록은 `for c in (am.tool_calls or [])`(`t2_gate_patch.py:7355`)로 **원본 am** 만 훑는데 그 턴 원본 am 의 tool_calls 는 **비어 있었고**, 호출은 그 뒤 `_ap_regen`(:10762)이 만들어 냈다. 절차 게이트(:7343-7401)는 `_ap_regen` 정의보다 **2000 줄 앞**에서 이미 지나갔다.

#### ⑦ [35]~[36] 승인 성공 → 완결 선언, submit 은 영영 없음
[35] *"Credit limit increase approved! - Account: cc_584f9c5d00_gold - Previous Limit: $5000.00 - New Limit: $7500.00"* → [36] 손님에게 완결 통보.
그 턴 로그에 **정답이 계산되어 있었는데 전달되지 않았다**:
```
[T2_FOLLOWUP] unlock-hint submit_credit_limit_increase_request_7392,get_user_dispute_history_7291,get_pending_replacement_orders_5765
[T2_FOLLOWUP] chain suppressed(th=2 resigns=1) after=['submit_credit_limit_increase_request_7392','check_cli_eligibility']
```
`T2_FOLLOWUP_RESIGN_TH` 기본 2(`t2_gate_patch.py:11483`) 미달로 억제. 마지막 구제 기회 소실. [37] user `###STOP###`.

**필요한 값이 문맥에 실재했는가**: `submit_credit_limit_increase_request_7392` 라는 이름은 [36] 시점까지 대화(role∈{tool,user})에 **한 번도 등장하지 않았다**. 등장한 곳은 위 로그 1줄(우리 층 내부 계산)뿐이고 그 줄은 억제됐다. ⇒ **모델이 베낄 수 있는 값은 없었다.** 다만 `KB_search_bm25{"submit credit limit increase request"}` 한 수로 도달 가능했다(trial 1 이 증명).

---

## 4. trial 1 step-by-step — 같은 자리에서 무엇이 달랐나

- [4]~[14] 동형(신원 값만 DOB+email · `get_user_information_by_name` 사용).
- [18]/[19] READ-FIRST 동일 발화 → [20]~[27] 동일 순응(`months:12` 동일) → [28] 동일 재호출 → **[29] ELIGIBLE 문면 바이트 동일**.
- [30] 여기서 갈린다. 모델 자백 축자:
  > *"It seems there was an error because the tool name was not properly discovered. Let me search the knowledge base to find the correct tool name for submitting the credit limit increase request."*
  → `KB_search_bm25{query:"submit credit limit increase request"}` → [31] 1위 `doc_credit_cards_credit_card_account_logistics_004` 전문(*"How to Submit a Request … 4. Confirm submission"*).
  (직전 로그 `[T2_RESOLVE] deny tool=unlock_discoverable_agent_tool arg=agent_tool_name reason=operator-fab` → `[T2_RESOLVE_CAP] 리셋(실행): 새 실행 ['KB_search_bm25']` — trial 0 과 **동일한 deny**를 받았다.)
- [32] 금액 되물음 → [33] user *"I'd like to **increase my limit by $2,500**"* (오도 0) → [34] `KB_search_bm25{"approve credit limit increase"}` → [35] `doc_…_logistics_006` 전문(티어별 상한).
- [36] `unlock_discoverable_agent_tool{approve_credit_limit_increase_5847}` — **원본 am 에 담긴 호출**(로그에 `T2_UNLOCK_PROV` 0회 = followup-regen 경로 미경유).
- 그 다음 모델이 **같은 채널로 call approve 를 시도했고**, 원본 am 이었으므로 절차 게이트가 정상 평가했다:
```
[T2_PROCEDURE] deny approve_credit_limit_increase_5847 missing=submit_request,disputes,pending_replacement
[T2_STACK] audit route=[…] chose=[('proc','approve_credit_limit_increase')] differs=True suppressed=['T2_VERDICT_SURFACE']
[T2_PIN_READ] pinned call_discoverable_agent_tool(agent_tool_name=submit_credit_limit_increase_request_7392)
[T2_LEVER] T2_PIN_READ sim=task_050#s373753 submit_credit_limit_increase_request_7392
[T2_TOOL_CHANNEL] pre-call regen: call_discoverable_agent_tool
```
→ [38] `unlock(submit_…_7392)` → [40] `call(submit_…_7392{cc_584f9c5d00_gold, 584f9c5d00, 2500})` **matched** → [42]~[49] `T2_READ_ROUTINE`/`T2_PIN_READ` 가 지목한 disputes·pending_replacement 실행 → [50] `unlock+call approve` 한 턴 동시 발행 → reward 1.0.

**즉 trial 1 을 통과시킨 것은 검색어가 아니라 `T2_PROCEDURE` deny 다.** 검색어는 그 deny 가 발화할 수 있는 형태(원본 am 안의 호출)를 만든 우연이다.

---

## 5. 분기점 확정

**a) 표면 분기 = [30] 검색어 한 수** — `"maximum increase for Gold Rewards Card"`(BM25 오회수) ↔ `"submit credit limit increase request"`(정문서 회수). 우리 층 deny 는 두 trial 동형(`operator-fab` / `T2_RESOLVE_CAP` 리셋).

**b) 진짜 분기 = approve 호출이 어느 버퍼에서 태어났는가**

| | trial 0 (fail) | trial 1 (pass) |
|---|---|---|
| unlock approve 출처 | `T2_VERDICT_SURFACE` **regen 버퍼** → `T2_UNLOCK_PROV` registry-allow | 모델 **원본 am**([36]) |
| call approve 출처 | `T2_UNCALLED_UNLOCK` **regen 버퍼**([34]) | 모델 **원본 am** |
| `[T2_PROCEDURE]` | **0 라인** (검사 대상 tool_calls 가 원본 am 에 없었다) | `deny … missing=submit_request,disputes,pending_replacement` **1회** |
| `T2_PIN_READ` | 0(실질) | 5 → submit 고정 |
| `T2_SEARCH_REARM` | 0 | 2 (`신규 대상 gold_rewards_card` 델타 배달) |
| 결말 | approve 만 실행 · **submit MISSING** | submit→reads→approve · clean |

---

## 6. 레버 발화표 (trial 0 · 168 라인 전수 · fail↔pass 대조)

| 레버 | fail / pass | 판정 | 근거(축자) |
|---|---|---|---|
| **READ-FIRST(`T2_SG_REQREADS`)** | 1 / 1 | **발화·효과 있음(양)** | 접미사 이름 2개 지목 → 즉시 순응 · 실측값 6 복사. 단 `months:12` 는 못 막음 |
| **`T2_PROCEDURE`** | **0** / 1 | **미발화(음·치명)** | pass 에서만 `deny approve_… missing=submit_request,disputes,pending_replacement`. fail 은 `would-fire but suppressed`·`error` 조차 0 = **평가 자체가 없었다** |
| **`T2_UNCALLED_UNLOCK`** | 2 / 1 | **오발화(음·치명)** | `surface approve_credit_limit_increase_5847` → *"call it now with its arguments"*. sim당 1회 예산을 **선행 미충족 시점**에 소진. pass 는 같은 레버가 모든 선행이 끝난 뒤(로그 221행) 발화해 무해 |
| **`T2_UNLOCK_PROV`(A5/OL-01 수리)** | 1 / 0 | **발화·정당(양)** | `registry-provenanced (allow) … val=approve_credit_limit_increase_5847` — t7336 자기차단 재발 0. 다만 이 통과가 §7-① 사슬의 입구 |
| **`T2_VERDICT_SURFACE`** | 1 / 1 | 발화·양면 | `surface decision=approve_…_5847,deny_…_5848,apply_…_6147` — 정답 이름을 줬으나 **선행 절차를 말하지 않는다** |
| **`T2_FOLLOWUP`** | 8 / 15 | **계산 성공·전달 실패(음)** | `unlock-hint submit_credit_limit_increase_request_7392,…` 1회 → 다음 줄 `chain suppressed(th=2 resigns=1)`. **정답을 알고도 안 보냈다** |
| **`T2_PIN_READ` / `T2_READ_ROUTINE`** | 0(실질) / 5+4 | **미발화(음)** | fail 은 `[T2_PIN_READ_STEPS] no read-only target (reads 0 of 1 ready)` 뿐. pass 는 `pinned call_discoverable_agent_tool(agent_tool_name=submit_…_7392)` — **PROCEDURE deny 의 종속물**이라 ①이 안 뜨면 같이 죽는다 |
| **`T2_SEARCH_AGENT`** | 6 / 6 | **오결정 + 축 소진 침묵(음)** | `[T2_DOCDECIDE] → 'Business Bronze Rewards Card'` · `→ 'Bronze Rewards Card'` — **둘 다 Gold Rewards Card 가 아니다**. 이후 `요청 축 … 모두 처리됨 — 침묵` 2회 |
| **`T2_SEARCH_REARM`** | **0** / 2 | **미발화(음)** | pass 만 `group=credit_cards 신규 대상 gold_rewards_card (기배달 bronze_rewards_card) — 소진 해제·문서 델타 6973자`. fail 은 채널이 닫힌 채 종료 |
| **`T2_CLAIMPROV`** | 10 / 33 | 발화·무해 | `window hit(resign) claims=3 unbacked=0 pending=0` — 이 궤적엔 산문 날조 없음(승인이 실제로 성공) |
| **`T2_SG_DOCS`** | 0 / 0 | 미발화 | — |
| **`T2_DEMANDED_STEP`** | 0 / 0 | 미발화 | — |
| **`T2_CLAIMPROV` 계열 `FAB_STRIP`** | 0 / 0 | 미발화(설계상 무관) | write **호출** 전용 |
| **`T2_ARG_PRODUCERS`** | 0 / 0 | 미발화 | 168 라인에 0회 |
| **`T2_REQUIRE_DOC_DELIVER`** | 0 / 0 | 미발화 | — |
| **`T2_GATE_REGEN`** | 3 / 3 | 발화 | fail 의 3회 중 2회(`verdict_surface`·`uncalled_unlock`)가 §7-① 사슬 |

**직전 런 이후 들어간 수리가 이 궤적에 개입했는가**: 예. `T2_UNLOCK_PROV` registry-allow(A5/OL-01)가 **정확히 의도대로** 발화해 t7336 의 자기차단을 없앴다. **왜 못 샀나**: 그 통과가 unlock→(사임)→UNCALLED_UNLOCK regen→call 사슬을 열었고, 그 사슬의 마지막 호출이 절차 게이트를 우회했다. t7336 §8-1 이 [[70]] 로 예고한 "판다" 항목(*"레지스트리에 실재하나 이 태스크에는 엉뚱한 이름의 unlock 이 통과한다"*)의 **더 나쁜 판본**이 실현된 것이다 — 이름은 엉뚱하지 않고 **시점이 엉뚱했다**.

---

## 7. 원인 확정

### 주 원인 — **our_layer** (코드 지목 2건 · CONFIRMED)

**① 사임-경로 regen 이 만든 tool_call 은 `T2_PROCEDURE` 재평가를 받지 않는다**
- 절차 게이트: `t2_gate_patch.py:7343-7401` — `_procs = (a2).get("procedures") …` / `for c in (am.tool_calls or [])`(:7355) / `[T2_PROCEDURE] deny`(:7399).
- 재생성 진입점: `t2_gate_patch.py:10762 def _ap_regen(fbtxt, tag, …)`. 반환 `_am2` 에 대한 재검사는 `gate`(`_denied_calls`, :10863-10879) · `T2_UNLOCK_NAME`(:10884-10893) · `T2_UNLOCK_PROV`(:10894-10945) **뿐**. `procedures` 재평가 없음.
- 같은 파일 주석이 이 구멍을 이미 자백한다(:7331 부근 · `★T2_WEV_ROUNDS (2026-07-24 C125)`): *"deny 후 regen된 호출은 같은 턴서 무검사 커밋"*, 그리고 :10883 *"반환 am은 while-루프의 un_fb 검사를 **우회**해 그대로 커밋됐다"*.
- 실측: trial 0 에서 `[T2_PROCEDURE]` **0 라인** ↔ trial 1 에서 동일 호출에 `deny … missing=submit_request,disputes,pending_replacement`. 두 sim 은 같은 sha·같은 태스크·같은 A2.
- **반증 검토**: (ⅰ) `_pcapped`(cap 6) 소진? — trial 0 deny 카운트 0 이므로 아님. (ⅱ) `_pblocker` 선점? — 선점이면 `would-fire but suppressed by=…` 가 찍히는데(:7374) 0 라인이므로 아님. (ⅲ) 순서 문제로 어차피 실패? — trial 1 이 정확히 이 deny 경로(approve 차단 → submit 먼저 → 나중 approve)로 **1.0** 을 받았으므로 아님. ⇒ **CONFIRMED**.

**② `T2_UNCALLED_UNLOCK` 의 술어가 절차 선언을 조회하지 않는다**
- `t2_gate_patch.py:11249-11275`. 술어 전체가 `_idle12 = sorted(_unlocked_names(...) - _called12)`(:11256-11258) — **`procedures` 를 보지 않는다**.
- A2 는 그 시점에 approve 가 아직 불가함을 이미 선언하고 있다: `a2/banking_knowledge.specific.json:4365-4374` `{"id":"decision","tool_any":["approve_credit_limit_increase_5847","deny_credit_limit_increase_5848"],"requires":["cooldown","disputes","pending_replacement","payment_history"]}` (그리고 `submit_request` :4311, `disputes` :4326, `pending_replacement` :4334). 동본 `a2/banking_knowledge.gate.json:4629/4683`.
- 그래서 문구가 *"call it now with its arguments"* 로 나갔다. sim당 1회 예산(`self._t2_uncalled_fired`)을 **선행 미충족 시점**에 태웠다.
- **반증 검토**: 문구는 *"if it is not required, say plainly why"* 라는 탈출구를 준다 ⇒ 모델이 안 부를 수도 있었다. 그러나 [[64]] 기준으로 이 문면은 "무엇을 하면 풀리나"를 **틀린 값**(=지금 부르라)으로 채웠고, ①이 없었다면 그 오유도조차 게이트에서 되돌려졌을 것이다. ⇒ ①의 **증폭 인자**로 CONFIRMED, 단독 충분원인은 아님.

### 보조 우리-층 결손 (분기 원인 아님 · 두 trial 공통)

- **`T2_FOLLOWUP` 임계 억제** — `t2_gate_patch.py:11483` `_th = int(os.environ.get("T2_FOLLOWUP_RESIGN_TH","2") or 2)` / 억제 인쇄 :11492. 정답 `submit_credit_limit_increase_request_7392` 를 :11475 에서 **이미 계산**해 놓고 보내지 않았다. 두 trial 모두 억제됐으므로 분기 원인은 아니나, trial 0 의 마지막 구제선이었다.
- **`check_cli_eligibility` ELIGIBLE 문면이 선행 절차를 침묵** — 판정문(`a2/banking_knowledge.specific.json` CLI 게이트 feedback)이 *"Decide with a TOOL CALL: approve_credit_limit_increase"* 만 말하고 `submit_request` 를 말하지 않는다. 같은 파일이 다른 키에서 그 선행을 선언하고 있는데도 그렇다. 두 trial 모두 이 문면 직후 approve 로 향했다 — **문구 모순**([[55]]). *(단, 이 문면이 "판다"의 어느 쪽인지는 미측정 ⇒ UNPROVEN 으로 남긴다.)*
- **`T2_SEARCH_AGENT` 오결정** — `[T2_DOCDECIDE] → 'Bronze Rewards Card'`(카드는 Gold). 두 trial 동일이라 분기 원인 아님. 배달 재료(254자)가 궤적에 안 남아 영향은 미측정.

### 보조 원인 — **model**

- [30] 검색어 선택(`"maximum increase for Gold Rewards Card"`) — trial 1 은 같은 자리에서 `"submit credit limit increase request"` 를 골랐다. 부하가 아니라 선택이다.
- [26] `months:12` — 정책이 허용하는 값은 6·3 뿐이고 그 문서는 [35](pass)에서 회수 가능했다. DB 축엔 무영향.
- [18] `consecutive_on_time_payments:"24"` 지어냄 — 우리 게이트가 즉시 교정.
- [33]→[34] 원본 응답에서 도구를 안 부르고 산문 사임 — 이것이 ①의 트리거였다.

### **env** — 결함 아님
BM25 가 `"maximum increase for Gold Rewards Card"` 에 APY 문서를 1위로 준 것은 검색기의 정상 거동이며, 정답 문서(`doc_…_logistics_004/006/007`)는 코퍼스에 실재하고 다른 질의로 도달한다(trial 1 증명). 도구 실행·결과 모두 정상. **[[68]] 제외 대상 아님.**

### **user_sim** — 무관
[1] 요청 · [3]/[7] 식별자 제공 · [37] 정상 종료. **금액을 먼저 말하지 않은 것**이 trial 0 과 1 의 차이지만(trial 1 은 [32] 에서 에이전트가 물어 [33] 에서 받음), trial 0 은 **묻지 않고** 2500 을 스스로 정해 approve 했다 — 즉 정보 부재가 아니라 에이전트가 묻지 않은 것이다([[21]]: user-sim 요인으로 종결 금지).

---

## 8. 선행 판정과의 대조

| 문서 | 그때 판정 | 이번 |
|---|---|---|
| `t7336_tasks/T7336_TASK_050.md` §7-① (`T2_UNLOCK_PROV` 가 실재 이름 차단) | our_layer 확정 · 처방 = 레지스트리를 출처에 추가 | **해소됨.** `registry-provenanced (allow)` 실물 발화 · 자기차단 재발 0. **같은 원인 아님** |
| 동 §7-② (deny 문면이 소진된 검색어 재지시) | our_layer | **재발 0**(그 deny 자체가 안 남) |
| 동 §7-③ (`t2_stack.admit` 지문중복이 CLAIMPROV 억제 → 산문 날조) | our_layer | **재발 0**(승인이 실제로 성공했으므로 날조 없음) |
| 동 §8-1 [[70]] "판다" 예고 | *"레지스트리에 실재하나 이 태스크에는 엉뚱한 이름의 unlock 이 통과"* | **더 나쁜 판본이 실현**: 이름은 옳고 **시점이 절차 위반**. 예고된 계기(`over-action` 계수)가 이 사슬을 못 잡았다 |
| 메모리 [[69]] *"050 은 승인 **중복**으로 실패"* | DUP | **재갱신 필요 — t7336 부터 MISSING, t7346 은 MISSING(submit)** |
| `T7336_FORENSIC_016_2026_08_21.md` (`T2_SEARCH_AGENT` 축-소진 봉쇄) | 검색 채널 구조적 폐쇄 | **동일 결함 재현**(fail 은 `T2_SEARCH_REARM` 0 · pass 는 2) |
| `t2_gate_patch.py:7331` 주석(C125) *"regen된 호출은 같은 턴서 무검사 커밋"* | 저자 인지 · WEV 에만 `T2_WEV_ROUNDS` 도입 | **`T2_PROCEDURE` 에는 이식되지 않았음이 실물로 확인** |

---

## 9. 처방 후보 (제안만 · 코드 수정 없음 · [[70]] A/B 의무 대상)

1. **`_ap_regen` 반환 호출에 절차 게이트 1회 재평가**(`t2_gate_patch.py:10862` 뒤, `gate` 재검사와 같은 자리). 술어는 기존 `t2_procedure.decide` 를 그대로 재호출 — 새 결정론 0([[62]]), 도메인 어휘 0, 선언은 이미 존재. **판다**: 사임-경로 regen 이 절차 deny 로 다시 접히면 그 턴이 빈손으로 끝날 수 있다(over-action↓ / no-action↑). **부정통제 의무**: `T2_PROCEDURE=1↔0` × 이 재검사 on/off 4칸 + `no-action` 계수([[57]]).
2. **`T2_UNCALLED_UNLOCK` 술어에 절차 필터**(`t2_gate_patch.py:11256-11258`): `_idle12` 에서 *현재 미충족 선행을 가진 노드의 도구*를 제외. 값은 전부 `procedures` 선언에서 나오고 엔진은 집합 차집합만 한다([[22]] 닫힌 술어). **판다**: 053 형(선행 다 끝났는데 안 부름)은 그대로 살고, 선행 미충족 상태의 넛지 1발이 사라진다.
3. **`T2_FOLLOWUP` 의 `unlock-hint` 를 체인 임계와 분리** — 힌트(사실 배달)는 `th` 억제 대상에서 빼고 체인 명령만 억제. [[64]]의 "이름 없는 억제 금지" 계열. **판다**: 힌트 발화 빈도↑ → 슬롯 경합. 계측 대상.
4. **(UNPROVEN·측정 먼저) `check_cli_eligibility` ELIGIBLE 문면에 선행 절차 문장 추가 여부** — 지금은 `submit_request` 를 침묵한다. 단 이 문면을 바꾸는 것은 A2 문구 저작이므로 정책 축자 근거를 먼저 대야 하고([[23]]), 1·2 만으로 이 sim 이 회복되는지 먼저 재야 한다.

---

## 10. 한 줄 결론

**trial 0 은 `submit_credit_limit_increase_request_7392` 를 빠뜨려 DB 해시가 갈렸고, 그 누락은 우리 층이 만들었다** — `T2_VERDICT_SURFACE`/`T2_UNCALLED_UNLOCK` 의 재생성 버퍼에서 태어난 approve 호출이 `_ap_regen`(`t2_gate_patch.py:10762`)의 재검사 목록에 절차 게이트가 없는 탓에 `T2_PROCEDURE`(`:7343-7401`)를 **평가조차 받지 않고** 커밋됐다. 같은 sha·같은 A2 의 trial 1 은 동일 호출이 원본 am 에 있었기에 `deny … missing=submit_request,disputes,pending_replacement` 로 되돌려져 reward 1.0 을 받았다. 직전 수리(A5/OL-01 레지스트리 출처)는 의도대로 작동했고, 그것이 연 문을 절차 게이트가 지키지 못한 것이 이번 실패다.
