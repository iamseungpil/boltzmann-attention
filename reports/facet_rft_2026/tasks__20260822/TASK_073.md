# t7346 · task_073 (ATM FEE DISPUTE RESOLUTION) 궤적 per-step 포렌식

- 대상: `bank_t7346_halfA_20260822.results.json.gz` / `bank_t7346_halfA_20260822.log.gz`
  (지시서에 적힌 이중밑줄 `..._halfA__20260822...` 파일은 없다 — 실파일은 밑줄 1개)
- 런: sha `ee18d797` · `T2_ARG_DOC_SUB=1 T2_VALUE_FORMULA=full T2_SG_DOCS=1` · nt=2 · agent `Qwen2.5-32B-Instruct-GPTQ-Int8` · user-sim `gpt-5.2`
- **결과: trial 0 = 1.0 · trial 1 = 0.0 (1/2)**. 참조 t7336 = **0/2** → **+1** 회복. 기준선 t7328 = 1/2(단 그 통과는 무효·§5.1)
- 로그 sim 태그 매핑(결과 파일 `seed` 로 확정): `s626729` = **trial 0(PASS)** · `s373753` = **trial 1(FAIL)**
- 두 trial 모두 `termination_reason = user_stop`(user-sim `###STOP###`)

---

## 0. 채점축 (C583ⓖ 선행 확인)

```
reward_basis      = ["DB"]
reward_breakdown  = {"DB": 1.0}  (trial 0)   /  {"DB": 0.0}  (trial 1)
db_check          = {"db_match": true}       /  {"db_match": false}
env_assertions = []   nl_assertions = null   communicate_checks = null
```

**DB-해시 축이다**(ACTION 축 아님) ⇒ `mutation_diff` 로 읽는다. `action_checks` 는 진단 보조일 뿐이다([[69]]).

⚠ 이 태스크에서 `action_checks` 는 **거짓말을 한다**: trial 0 은 `reward=1.0`(db_match=true)인데
`apply_checking_account_credit_5829` 3행 전부 `action_match=false` 로 찍혀 있다. 액션표만 보면
"3건 실패"로 읽힌다. 성적은 `reward` 다([[69]]).

gold 변이 4건 = `log_verification` ×1 + `apply_checking_account_credit_5829` ×3
(**계좌당 정확히 1건의 NET credit**: `_1`=9.5 · `_2`=9.0 · `_3`=1.5).

태스크 notes 축자: *"apply **NET credits** accounting for both overcharges AND missing fees … NET CREDITS: Blue = $9.50, Green = $9.00, Light Green = $1.50 ($3.00 overcharges minus $1.50 missing fee)."*

---

## 1. 변이표 (정본 `t2_forensic.mutation_diff` · 손 비교기 미사용)

### trial 0 (`s626729`) — `missing 0 · wrongarg 0 · extra 0 · dup 0 · blocked 0 · matched 4` → **clean=True**

| 종류 | 내용 |
|---|---|
| MATCHED | `log_verification{Kim Junho…}` · `{_1, 9.5, fee_refund}` · `{_2, 9.0, fee_refund}` · `{_3, 1.5, fee_refund}` |

### trial 1 (`s373753`) — `missing 2 · wrongarg 6 · extra 0 · dup 0 · blocked 0 · matched 2`

| 종류 | 내용 |
|---|---|
| MISSING | `{_1, **9.5**, fee_refund}` · `{_2, **9.0**, fee_refund}` |
| WRONGARG | `_1`: **3.0**(msg 47) · **5.0**(msg 49) · **1.5**(msg 51) — 3턴 분할<br>`_2`: **3.0**(msg 53) · **3.0**(msg 55) · **3.0**(msg 57) — 3턴 분할 |
| MATCHED | `log_verification`(msg 27) · `{_3, 1.5, fee_refund}`(msg 59) |

**WRONGARG 필드별 대조** (보낸 인자 ↔ gold 인자):

| 필드 | 6건 전부 |
|---|---|
| `account_id` | **일치**(`chk_kj93a7b2e1_1` ×3 · `_2` ×3) |
| `credit_type` | **일치**(`fee_refund`) |
| `amount` | **불일치 — 어긋난 필드는 이것 하나뿐** |

그리고 **합계는 정확하다**: `_1` 3.0+5.0+1.5 = **9.50** = gold · `_2` 3.0+3.0+3.0 = **9.00** = gold.
잔액도 gold 와 같은 종점에 도달했다(축자 tool msg 48~60): `5200→5203→5208→**5209.50**` ·
`12750→12753→12756→**12759.00**` · `890.50→**892.00**`.

⇒ **실패 단위는 금액 오산도 탐색 실패도 아니라 호출 granularity 다** — 계좌당 1행 NET ↔ 수수료 라인당 1행.
DB 해시는 잔액이 아니라 **credit 트랜잭션 행 집합**을 본다(3행 ↔ 7행).

`_3` 가 matched 인 것은 **두 오류의 우연한 상쇄**다(§3.5) — NET 계산 성공의 증거가 아니다.

---

## 2. trial 1 (FAIL) — step-by-step 결정 지점 추적 (축자)

### 2.0 필요한 값·필요한 형태는 **문맥에 실재했다** (세 겹으로)

**⑴ turn 3 · msg[3]** — 최초 `KB_search_bm25("retrieving bank account transaction history")` 결과 7번째 문서
`doc_bank_accounts_bank_accounts_(general)_017` 축자:

> "The apply_checking_account_credit_5829 tool may only be called **ONCE per checking account per customer interaction**. … Because only one credit call is allowed per account, if multiple corrections are needed for the same account …, **combine them into a single credit with the total amount** and use the credit_type that applies to the majority of the corrections."

**⑵ turn 34 / 36 / 38** — 우리 A3 comparator `get_atm_fee_discrepancies` 반환문이 **계좌마다 1회씩, 3회** 축자:

> "ATM fee lines whose charged amount does NOT match the documented fee schedule for this account level: btxn_kj07s5t6u7v9 (charged $3.00, documented fee $0.00, difference $3.00); btxn_kj05k7l8m9n1 (charged $20.00, documented fee $15.00, difference $5.00); btxn_kj03c9d0e1f3 (charged $4.50, documented fee $3.00, difference $1.50). … **If corrections are owed, the credit policy requires ONE fee_refund credit for the net correction of THIS account (do not credit the same lines twice).**"

**⑶ turn 46** — unlock 결과가 도구 계약(`amount`, `credit_type`)을 다시 제시.

⇒ *"라인별 차액 3개"* 와 *"계좌당 1건 NET"* 이 **둘 다** 문맥에 있었다. 남은 일은 덧셈 하나였다.

### 2.1 ★결정 지점 A — msg[39] : 우리 층이 유도한 **재-read** 가 감사결과를 밀어냈다

turn 39 에서 모델이 산문으로 사임했고 우리 층이 두 번 개입했다(로그 축자):

```
[T2_SEARCH_REARM] group=checking_accounts 신규 대상 light_green_account (기배달 blue_account) — 소진 해제·문서 델타
[T2_SEARCH_REARM] group=checking_accounts 델타 배달 10673자 (문서 13·뺀 것 0) turn=39
[T2_SEARCH_ON_PROCEED] deny 아님 · 재료 10673자 배달
[T2_COVERAGE_FU] fired tool=get_atm_fee_discrepancies
```

`T2_COVERAGE_FU`(= `[coverage] 3 of 6 rows were checked (3 could not be verified)` 재인용 regen)가 발화했고,
그 결과 msg[39] 는 **본문 0자 + 도구호출 1개**로 나왔다:

> `>>TOOLCALL name=call_discoverable_agent_tool args={"agent_tool_name": "get_bank_account_transactions_9173", "arguments": "{\"account_id\": \"chk_kj93a7b2e1_3\"}"}`

msg[40] = `_3` 계좌 **26 레코드 통짜 덤프 6257자**. 미판정 3행을 다시 comparator 에 넣는 후속은 **일어나지 않았다**
(`get_atm_fee_discrepancies` 재호출 0회) — 즉 `T2_COVERAGE_FU` 는 **발화했으나 목적 미달**이고, 남긴 것은
**결정 직전 6257자 부하**뿐이다. 이 덤프가 §4.1 의 근거-창 잠식을 만든다.

### 2.2 ★결정 지점 B — msg[41] : 모델이 **라인별 환불**로 국면을 고정 (분기의 원점)

msg[41] assistant 축자(발췌):

> "Based on the discrepancies found, I will correct the ATM fees for your accounts. Here are the corrections that will be made:
> - **Blue Account (chk_kj93a7b2e1_1):**
>   - Transaction ID: `btxn_kj07s5t6u7v9` - Corrected from $3.00 to $0.00. **Refund: $3.00**
>   - Transaction ID: `btxn_kj05k7l8m9n1` - Corrected from $20.00 to $15.00. **Refund: $5.00**
>   - Transaction ID: `btxn_kj03c9d0e1f3` - Corrected from $4.50 to $3.00. **Refund: $1.50**
> - **Green Account (chk_kj93a7b2e1_2):** … Refund: $3.00 / $3.00 / $3.00
> - **Light Green Account (chk_kj93a7b2e1_3):** … Refund: $1.50"

**계좌 총액은 한 번도 계산되지 않았고**, `[34]/[36]/[38]` 이 3회 말한 *"ONE fee_refund credit for the net correction"* 은
**한 글자도 재현되지 않았다**. 여기서 "환불 = 라인당 1건"이라는 중간표상이 굳는다.

### 2.3 결정 지점 C — msg[42]~[43] : user-sim 은 라인별을 요구하지 **않았다**

msg[42] user 축자 전문:

> "Oh wow, so there **ARE** errors? I knew something was off. What can be done to fix this?"

시나리오 지시 4번 그대로다. **granularity 를 지정하지 않았다** ⇒ [[21]] 상 user_sim 귀속 불가.
(대조: PASS trial 의 user 는 오히려 *"apply credits for the full differences … ($3.00, $5.00, and $1.50)"* 로
**더 라인별에 가깝게** 말했는데 그쪽 에이전트는 합산했다 — §3.2.)

msg[43] assistant 는 msg[41] 의 라인별 표를 그대로 반복하고 **존재하지 않는 배치 도구를 날조**한다:

> `>>TOOLCALL name=call_discoverable_agent_tool args={"agent_tool_name": "process_atm_fee_refunds_8473", "arguments": "{\"refunds\": [{…\"amount\": 3.0}, {…\"amount\": 5.0}, {…\"amount\": 1.5}, … 7건]}"}`

msg[44] tool 축자: `Error: Unknown agent tool 'process_atm_fee_refunds_8473'. This tool is not available.`

**이 날조 도구의 스키마(`refunds: [ {transaction_id, amount} … ]`)가 곧 모델의 중간표상**이다 — 라인당 1건.

### 2.4 ★결정 지점 D — msg[45]~[47] : 도구가 바뀌었는데 **표상은 안 바뀌었다**

msg[45] 축자:

> "I apologize for the oversight. Let's unlock the `apply_checking_account_credit_5829` tool first and then proceed with applying the credits. … **For the first credit: Account ID: `chk_kj93a7b2e1_1`, Amount: $3.00**, Description: Refund for overcharged ATM fee (btxn_kj07s5t6u7v9)"

msg[47] 축자 + 호출:

> "Starting with the Blue Account (chk_kj93a7b2e1_1):
> 1. Transaction ID: `btxn_kj07s5t6u7v9` - Refund: $3.00 … Let's apply the **first credit** for the Blue Account."
> `>>TOOLCALL … {"agent_tool_name": "apply_checking_account_credit_5829", "arguments": "{\"account_id\": \"chk_kj93a7b2e1_1\", \"amount\": 3.0, \"credit_type\": \"fee_refund\"}"}`

**여기가 실패 확정점이다.** 그리고 바로 이 턴 생성 중에 우리 `T2_WRITE_SUB` 가 pre-draft 를 배달했다(§4.1).

### 2.5 반복 — msg[49]/[51]/[53]/[55]/[57] : 6발 전부 성공, 아무도 막지 않음

`_1` 에 5.0 · 1.5, `_2` 에 3.0 ×3. `T2_STALE_STRIP` **0회 발화**(동일 인자 `{_2, 3.0}` 3발이 3턴에 걸쳐 통과).
`T2_RESOLVE operator-find 침묵`(*"이미 성공 실행 — 재지시는 중복 write 를 만든다"*)은 **6발이 끝난 뒤** 7회 발화했다.

### 2.6 ★산술 능력은 있었다 — msg[61]

7발이 끝난 직후 모델 자신이 축자:

> "- **Blue Account (chk_kj93a7b2e1_1):** - **Total Refund: $9.50** - New Balance: $5209.50
> - **Green Account (chk_kj93a7b2e1_2):** - **Total Refund: $9.00** … - **Light Green …: $1.50**"

⇒ gold 3값을 **정확히 산출할 수 있었다**. 결손은 산술도 탐색도 아니고 **"합쳐서 1건으로 발행한다"는 호출 형태**뿐이다
([[63]] 형: 더하기 지시는 안 듣고 형태를 못 바꾼다).

---

## 3. trial 0 (PASS) — 분기점 특정

### 3.1 turn 29~31 : 같은 comparator, 같은 판정
msg[30] tool 축자는 trial 1 의 msg[34] 와 **바이트 동일**(같은 3라인 + 같은 NET 문장 + `[coverage] 10 of 10`).

### 3.2 ★분기점 — msg[31]→[33] : **먼저 고객에게 보고**하고, 그 답 턴에서 **정책을 인용해 합산**

msg[31] 은 Blue 3건을 나열하고 **실행 대신 확인을 구했다**: *"Would you like me to proceed with correcting these fees…?"*
msg[32] user 축자:

> "Yes, please go ahead and **correct all of those incorrect charges** and apply credits for the full differences on my **Blue Account** (**$3.00, $5.00, and $1.50**). What can be done to fix this, and how soon will the credits show up?"

msg[33] assistant 축자 — **여기가 갈린 자리다**:

> "**According to our policy, the correction will involve crediting the net difference for each account. In this case, the total correction amount for your Blue Account is $9.50 ($3.00 + $5.00 + $1.50).**"

user 가 **라인별로 말했는데도** 에이전트가 정책 문면을 인용해 **NET 으로 접었다**.
이후 msg[43] 축자: *"### Blue Account … - **Total Correction Amount: $9.50** / ### Green … **$9.00** / ### Light Green … **$1.50**"* → 3발 정타.

### 3.3 두 trial 의 구조적 차이 요약

| | trial 0 (PASS) | trial 1 (FAIL) |
|---|---|---|
| 감사 3계좌 완료 시점 | msg[30]·[38]·[40] (Blue 는 먼저·둘은 나중) | msg[34]·[36]·[38] (연속 3회) |
| **중간표상** | *"Total Correction Amount"* (계좌당 총액) — msg[33]/[43] | *"Refund: $3.00 / $5.00 / $1.50"* (라인당) — msg[41]/[43]/[45]/[47] |
| 결정 전 부하 | comparator 결과 2건 + unlock (recent 2407자) | **`_3` 26레코드 재-덤프 6257자** + unlock (recent 679자) |
| 날조 도구 시도 | `get_interest_correction`(오도구·env 가 RESULT-SIGN 으로 반려) | `process_atm_fee_refunds_8473`(**라인 배열 스키마** 날조) |
| WRITE_SUB pre-draft | 배달됨(근거 2407자) — **무시하고 9.5 발행** | 배달됨(근거 679자) — **라인별로 실행** |

### 3.4 trial 0 도 완벽하진 않았다(무해)
msg[33] 에서 `get_interest_correction` 을 잘못 골랐고 우리 `T2_SG_RESULT_RANGE` 가 반려했다
(축자: *"[RESULT-SIGN] this correction computes to 0.0, which is not greater than 0…"*). 모델은 msg[35] 에서 회복했다.
`T2_PRESCRIPTION` 오발화(`deny tool=apply_statement_credit_8472`)도 **3발이 끝난 뒤** msg[58] 에서 일어나 무해했다.

### 3.5 `_3`(Light Green) matched 는 우연한 상쇄다 — 양 trial 공통
gold notes 는 `_3` 에 3오류(과청구 1.50 + 과청구 1.50 + **누락** 1.50)를 둔다. 우리 comparator 는 (2)만 리포트했고
로그 축자 `[T2_COMPUTE] select_discrepant: 3/6행 판정불가(operand가 숫자 아님) — under-action 위험` ·
반환문 `[coverage] 3 of 6 rows were checked (3 could not be verified)`.
미검출 +1.50 과 미차감 −1.50 이 상쇄돼 1.50 이 나왔다. **NET 계산이 성공한 증거가 아니다.**
(선언 `_note_` 축자: *"light_green/light_blue oon … 월 무료횟수 규정이 문서에 불완전(모호점 8·10)→기대 null=판정 보류"* — 설계상 판정 불가.)

---

## 4. 레버 발화표 (로그의 이 sim 줄만 · 발화/무시/미발화/오발화)

| 레버 | trial 0 PASS | trial 1 FAIL | 판정 |
|---|---|---|---|
| `T2_SG_DOCS` (본런 ON) | **0** | **0** | **미발화 — 선언 공백**. `scaffold_get_tools[8].isolate` 에 `docs` 키가 없어 `t2_scaffold_get.py:552` 경로에 진입조차 안 한다. 이 태스크에 **도달 0** |
| `T2_VALUE_FORMULA=full` (본런 ON) | 0 | 0 | 미발화(불일치 없음·`t2_scaffold_get.py:2181` 은 불일치시만 인쇄) |
| `T2_ARG_DOC_SUB` (본런 ON) | 6 | 5 | 발화·**축 무관**. 전부 `spend_category=None`(카드 요율 축) — 073 실패축과 무관, 해도 없음 |
| `T2_PIN_READ` | 0 | 0 | 미발화 |
| `T2_DEMANDED_STEP` | 0 | 0 | 미발화 |
| `T2_FOLLOWUP` | 0 | 0 | 미발화 |
| `T2_ARG_PRODUCERS` | 0 | 0 | 미발화 |
| `T2_REQUIRE_DOC_DELIVER` | 0 | 0 | 미발화 |
| READ-FIRST (`requires_reads`·`T2_SG_REQREADS`) | 0 | 0 | **정상 미발화** — 양 trial 모두 comparator 호출 전 계좌목록+3계좌 거래 read 완료(요건 충족) |
| `FAB_STRIP` | 0 | 0 | 미발화. trial 1 의 `process_atm_fee_refunds_8473` **날조는 걸러지지 않았다**(env 가 `Error: Unknown agent tool` 로 반려) |
| `T2_SEARCH_AGENT` | 10 | 10 | 발화 1회 + 이후 *"요청 축 checking_accounts 모두 처리됨 — 침묵"* 반복 |
| `T2_SEARCH_REARM` (t7336 OL-29 수리·이번 ON) | 2 (turn 31) | 2 (turn 39) | **발화·성공(축-소진 해제 실증)**. 다만 배달 10673자가 **결정 구역 부하**로 들어감([[70]] 매도측) |
| `T2_COVERAGE_FU` | 1 (msgs 49 — **write 이후**) | 1 (msg 39 — **write 이전**) | **발화·목적 미달**: 두 trial 모두 재-read 만 유발하고 comparator 재호출 0회. FAIL 쪽에서는 그 재-read 가 §4.1 의 근거-창 잠식을 만듦 |
| `T2_CLAIMPROV` / `T2_CLAIM_PROV` | 50 / 4 | 37 / 13 | 발화·전량 `kind-index rescued`, `unbacked=0`. 이 실패축 아님 |
| `T2_WRITE_SUB` | 28 (pre-draft 6회) | 34 (pre-draft 5회) | **발화 — §4.1(핵심)** |
| `T2_STALE_STRIP` | 0 | **0** | **미발화**. 동일 인자 `{_2,3.0}` 3발이 3턴에 걸쳐 전부 통과 — t7336 OL-17/P-D **미수리 확인** |
| `T2_PRESCRIPTION` | 1 (msg 58·**write 이후**) | 0 | trial 0 에서 오발화했으나 **무해**(이미 3발 정타 완료). t7336 OL-26 **미수리 확인** |
| `T2_RESOLVE operator-find 침묵` | 0 | 7 (전부 **6발 이후**) | 발화 시점이 늦어 효과 0 |
| `T2_FORCE_ACTION` | 8 | 7 | 발화(사임→도구강제). FAIL 의 msg[39] 재-read 를 밀어낸 경로 |
| `T2_SG_ISOLATE` / `T2_SG_TRACE` | 13 / 6 | 12 / 5 | 발화·정상(3계좌 fetch_formalize 성공). `operand-size … ⚠MISMATCH` 는 fee 라인만 추출한 정상 축소 |

### 4.1 ★`T2_WRITE_SUB` — **발화했고, 그 발화가 라인별을 밀었다** (이번 런의 신규 소견)

배선(코드 축자):

- `t2_gate_patch.py:6946-6949` — 트리거용 `_basis = _SCw.recent_tool_text(state.messages, basis_max_chars)` … **`scope` 미전달 → 기본 `"recent"`**
- `t2_resolve.py:637-639`(`sub_write_proposal`) — 실제 서브 근거 `basis = SC.recent_tool_text(msgs, spec["basis_max_chars"], scope=spec["basis_scope"])`
- `a2/banking_knowledge.specific.json → write_initiation` : `basis_max_chars = 4000` · `basis_scope = "all"`
- `t2_subcall.py:117` — `return txt[-int(cap):]` … **뒤에서 cap 자만 남긴다**
- `t2_subcall.py:150-179`(`grounded_calls`) — 제안의 **모든 잎 값**이 근거 코퍼스에 실재해야 통과

로그 축자(FAIL, 첫 오발행 turn):
```
[T2_WRITE_SUB] 제안 1건 → 근거검산 통과 1건
[T2_WRITE_SUB] pre-draft 전달(근거 679자·미실행 필터 2종)
[T2_MATERIAL_GATE] stop=resolve_cap(정체 3회) turn=47
```
로그 축자(PASS, 첫 정발행 turn):
```
[T2_WRITE_SUB] 제안 1건 → 근거검산 통과 1건
[T2_WRITE_SUB] pre-draft 전달(근거 2407자·미실행 필터 2종)
```

**재구성 검증**: 결과 파일의 `messages` 로 같은 함수를 재현하면 FAIL@msg47 의 `recent` 창 = **679자**,
PASS@msg43 = **2407자** — 로그 인쇄값과 **정확히 일치**한다. 재구성이 `state.messages` 와 동형임이 확인된다.

그 위에서 **서브가 실제로 받은 창**(`scope="all"`, `cap=4000`)을 재현하면:

| | FAIL trial 1 @ msg47 | PASS trial 0 @ msg43 |
|---|---|---|
| 창 안의 `"ONE fee_refund credit"` 발생 | **0회** | **3회** |
| 창 안의 comparator 결과(`difference $5.00` 등) | **없음** | 있음 |
| 창의 내용 | `_3` 26레코드 재-덤프(6257자·msg[40])의 꼬리 + unlock(679자) | comparator 결과 2건(940+786) + unlock |

즉 **§2.1 의 재-read 덤프가 감사결과 3건(940+940+786=2666자)을 4000자 tail 창 밖으로 밀어냈다.**
FAIL trial 의 격리 서브는 *"감사 결과 0건 · NET 지시 0회"* 인 근거로 write 제안을 만들었고, 그럼에도 1건이
`grounded_calls` 를 통과해 메인에 **권위 문구로** 배달됐다(delivery_template 축자: *"[ISOLATED-FORMALIZATION] this decision
was formalized in isolation and produced the following call(s) … Check that each call matches its stated basis"*).

**더 근본적인 구조 편향**(`val_grounded` 직접 실행으로 확정):

| 후보 amount | 4000자 창 | 전체 도구 코퍼스 |
|---|---|---|
| 3.0 / 5.0 / 1.5 (라인별) | — | **grounded = True** |
| **9.5 (gold NET)** | False | **grounded = False** |
| 9.0 (gold NET) | — | True(우연) |

`9.50` 은 어느 도구 출력에도 없다(유일 히트는 KB 검색의 `Score: 9.5098`). 2026-08-19 에 `= ${delta_total:.2f}` 를
반환문에서 제거한 이래([[23]] 준수·옳은 제거) **gold 정답 호출은 `grounded_calls` 를 원리상 통과할 수 없고,
통과 가능한 것은 라인별 금액뿐이다.** ⇒ 이 채널이 073 에서 배달할 수 있는 제안은 **언제나 라인별**이다.

PASS trial 은 같은 채널의 라인별 pre-draft 를 **무시하고** 9.5 를 냈고(9.5 는 그 채널이 제안할 수 없는 값이다),
FAIL trial 은 **그대로 실행**했다. 이 레버는 이 태스크에서 **한쪽 부호로만 작동한다**([[70]] 매도측).

---

## 5. 선행 판정과의 대조

지시서에 선행 보고서가 지정되지 않아 `reports/facet_rft_2026/` 에서 073 을 다룬 절을 전수 확인했다.

| 문서 | 그 판정 | t7346 에서 |
|---|---|---|
| `t7336_tasks/T7336_TASK_073.md` §7 trial 0 | *"our_layer(주)+model(부) … P5 가 정책 축자 `across all identified fee discrepancies` 를 제거 … 건별로 쪼갰다"* | **동일 실패 모드가 trial 1 에서 그대로 재현**(missing 2 · wrongarg 6 · 필드는 `amount` 하나 · 합계 정확). 문면은 **미수리**(§5.2) |
| 같은 문서 §7 trial 1 | *"credit 도구 탐색 read 0회 … `T2_PRESCRIPTION` 오발화 deny → `file_credit_card_transaction_dispute` 오유도"* | **소멸**. t7346 양 trial 모두 `apply_checking_account_credit_5829` 를 스스로 unlock·호출했고 PRESCRIPTION 은 write 이후에만 1회 발화(무해). **원인이 달라졌다** |
| 같은 문서 §8 P-A(최우선) | *"return_template 의 NET 절에 정책 축자 수식어 복원 + SCOPE 경고문을 NET 절 뒤로"* | ❌ **미이행**. `a2/banking_knowledge.specific.json` `scaffold_get_tools[8].return_template` 현재 축자 = *"…requires ONE fee_refund credit for the net correction **of THIS account** (do not credit the same lines twice)."* — 수식어 없음 |
| 같은 문서 §8 P-D | *"`T2_STALE_STRIP` ②의 `_wtools` 공백"* | ❌ **미이행**(§4 표: 발화 0) |
| 같은 문서 §8 P-B/P-C | `T2_PRESCRIPTION` `role="tool"` 코퍼스 제외 | ❌ **미이행**(이번엔 무해하게 발화) |
| `T7336_FAILURE_MASTER_2026_08_22.md` OL-27 | *"CONFIRMED — P5 개정이 정책 축자 합산-범위 수식어를 삭제"* · B8/`x474` 로 격리 확정 예정 | `x474` **미생성**(스크립트·결과 모두 부재). OL-27 미수리 상태로 본런 진입 |
| `T7336_FAILURE_MASTER` §OL-29 | `T2_SEARCH_REARM` OFF | ✅ **ON 됐고 발화 확인**(축-소진 해제 실증) |
| `T7335_NT1_FORENSIC_HALFA_2026_08_21.md` §073 | *"DUP … 1차 완료 시점엔 만점"* | DUP 0건 — 이미 t7336 에서 원인이 바뀌었고 t7346 도 동일 |
| **t7328 기준선**(sha 상이) | 073 = 1/2 (trial 0 = 1.0) | 그 통과는 **무효**였다 — §5.1 |

### 5.1 t7328 의 통과는 엔진이 채점 인자를 공급한 통과다 (재확인)
t7328 반환문 축자(`tool[47]`): *"…requires ONE fee_refund credit for the net correction across all identified fee
discrepancies of THIS account **= $9.50** (do not credit the same lines twice)."*
gold `amount` 자체가 문면으로 배달됐다 ⇒ [[23]]/[[62]] 위반. 제거는 옳았다.

**따라서 t7346 의 1/2 는 t7328 의 1/2 와 같은 수가 아니다.** t7346 trial 0 은 도구 코퍼스 어디에도 `9.50` 이
없는 상태에서 모델이 msg[33] 에서 *"$9.50 ($3.00 + $5.00 + $1.50)"* 를 **스스로 계산**해 낸 **정당한 통과**다.
계보 정정: t7328 **유효 0/2** → t7335 0/1 → t7336 0/2 → **t7346 1/2(첫 유효 통과)**.

### 5.2 P-A 미이행에도 +1 이 난 이유 (부호 분해)
문면은 그대로인데 결과가 0/2→1/2 로 올랐다. 이번 런에서 073 에 실제 도달한 변화는
`T2_SEARCH_REARM` ON(OL-29 수리) 하나뿐이고, 나머지 ON 3종은 미도달(`T2_SG_DOCS`·`T2_VALUE_FORMULA`)이거나
축 무관(`T2_ARG_DOC_SUB`)이다. **n=2 에서 문면 미변경 + 1 표 변동은 [M] 이상으로 올릴 수 없다** — OL-27 의
인과는 여전히 `x474` 격리로만 확정된다([[62]] ①).

---

## 6. 원인 확정 (4주체 귀속 · 궤적 축자 근거만)

### trial 0 (`s626729`, reward 1.0) — 실패 없음
`clean=True`. 우리 층 오발화 2건(`get_interest_correction` RESULT-SIGN 반려·`T2_PRESCRIPTION` msg[58])이 있었으나
**둘 다 정발행 이후/무해**. 정당한 통과다(§5.1).

### trial 1 (`s373753`, reward 0.0) — **model(주) + our_layer(부·2건)**

**model — 1차 원인 (결정 지점 = msg[41])**
필요한 값(3.00/5.00/1.50)과 필요한 형태(*"ONE fee_refund credit for the net correction of THIS account"*)가
**세 겹으로 문맥에 실재**했는데(§2.0), msg[41] 에서 **라인별 환불표**로 국면을 고정했고 그 표상을
msg[43](날조 배치도구 `refunds[]` 스키마) → msg[45] → msg[47] 까지 **도구가 바뀌어도 유지**했다.
msg[61] 에서 총액 9.50/9.00/1.50 을 **정확히 산출**했으므로 산술·탐색 결손이 아니다.
결손은 **호출 granularity 하나**([[63]] 형: 지시로는 형태가 안 바뀐다).

**our_layer ⓐ — `T2_WRITE_SUB` 근거-창 잠식 + 라인별-단독 통과 (CONFIRMED · §4.1)**
코드 경로: `t2_subcall.py:117`(`txt[-cap:]`) · `t2_subcall.py:150-179`(`grounded_calls` 전-잎 검산) ·
`t2_gate_patch.py:6946-6949`(트리거 basis 가 `scope` 미전달 → `"recent"`, 서브 basis 와 코퍼스 불일치) ·
선언 키 `a2/banking_knowledge.specific.json → write_initiation.basis_max_chars = 4000` / `basis_scope = "all"`.
실측: FAIL 의 4000자 창에 NET 지시 **0회**(PASS 3회) · gold `amount=9.5` 는 전체 코퍼스에서도 **grounded=False**.
⇒ 이 채널이 msg[47] 턴에 배달할 수 있었던 제안은 **라인별뿐**이었고, 실제로 배달됐다.

**our_layer ⓑ — `T2_COVERAGE_FU` 가 유발한 재-read 가 그 창을 채웠다 (CONFIRMED·기여)**
코드 경로: `t2_gate_patch.py:11193-11198`. 실측: FAIL msg[39] = 본문 0자 + `get_bank_account_transactions_9173(_3)`
→ msg[40] 6257자 덤프. comparator 재호출은 **0회**(목적 미달). PASS 에서는 같은 레버가 write **이후**에 발화해
비용이 0이었다. ⇒ 정발화지만 **결정점 앞에 놓이면 근거를 밀어내는 부작용**이 있다.

**our_layer ⓒ — OL-27(P5 문면) : 재확인은 되나 이번 런 인과는 UNPROVEN**
선언 키 `a2/banking_knowledge.specific.json → scaffold_get_tools[8].return_template`(사본 `gate.json` · `split/core.json`).
문면이 t7336 과 **바이트 동일**이고 실패 모드도 동일하다는 사실은 확인했으나, 같은 문면에서 trial 0 이 통과했으므로
**이 문면만으로 갈린다는 인과는 이 런이 증명하지 못한다**. `x474`(3세대 A/B/C 격리) 대기.

**env 아님**: 도구 반려는 `Unknown agent tool 'process_atm_fee_refunds_8473'` 1건뿐이고 정당하다. blocked 0건.
**user_sim 아님**([[21]]): msg[42] 는 granularity 를 지정하지 않았고, PASS 쪽 user 는 **더 라인별로** 말했는데도
에이전트가 합산했다(§3.2).

---

## 7. 처방 후보 (제안만 · 수리 미실행)

1. **[P-1 · `T2_WRITE_SUB` 근거 창 · 최우선]** `t2_subcall.recent_tool_text` 의 `txt[-cap:]` 은 **뒤에서 자르므로 큰 후속
   read 가 감사결과를 통째로 밀어낸다**. 닫힌 술어 대안 = *"cap 을 메시지 경계 단위로 채운다(최신 tool 메시지부터
   역순으로 통째 담되, 한 메시지가 cap 을 넘으면 그 메시지를 건너뛴다)"* — 도메인 어휘 0·판단 0·[[59]] 무결.
   ⚠[[70]] 파는 것: 거대 레코드 덤프가 근거에서 빠지면 그 덤프에서만 접지되던 값(계좌 id 등)이 기각될 수 있다 →
   **부정통제 필수**(050/072/074 계열에서 `제안 N건 → 통과 0건` 증가 여부 계측).
2. **[P-2 · `grounded_calls` 편향 공개]** gold NET 값은 원리상 접지 불가이므로 이 채널은 073 형(집계 write)에서
   **라인별만 제안할 수 있다**. 값을 만들어 채우는 수리는 [[23]]/[[62]] 위반이므로 **금지**. 대신
   ⑴ 집계-write 태스크에서 pre-draft **배달 자체를 억제**하거나(닫힌 술어: *"동일 도구·동일 계좌로 복수 제안이
   가능한 상태"* 는 판정 불가하므로 실질적으로는) ⑵ delivery_template 에 **granularity 경고를 넣지 말고**,
   대신 §7-3 처럼 반환문 쪽에서 형태를 고정하는 편이 싸다. ⚠[[57]] 부정통제: pre-draft 를 끄면 x309(8/8 실행)
   이득이 함께 죽는다 — **끄기 아니라 조건부**([[60]]).
3. **[P-3 · OL-27 복원(t7336 P-A 계승)]** `return_template` NET 절을 정책 축자
   *"the net correction **across all identified fee discrepancies** of THIS account"* 로 되돌리고
   SCOPE 경고문은 **NET 절 뒤로** 옮긴다. `= ${delta_total}` 은 복원 금지([[23]]).
   ⛔ **선행 조건**: `x474`(3세대 문면 A/B/C 격리·엔드포인트 = *계좌당 credit 호출 개수*) 를 **먼저** 돌린다([[62]] ①).
   격리에서 문면만으로 갈리면 레버는 **전달(문면)** 이고 결정론 추가는 불필요하다.
4. **[P-4 · `T2_COVERAGE_FU` 위치]** 재-read 만 유발하고 comparator 재호출로 이어지지 않는다(양 trial 2/2).
   [[64]] 기준으로 **무엇을 하면 풀리나**(= 같은 comparator 를 미판정 행만 담아 재호출)를 문면이 지목하는지 점검.
   ⚠ 이 레버를 결정점 **앞**에 놓는 것이 P-1 을 안 고치면 손해라는 점이 이번 실측이다.
5. **[P-5 · 미이행 부채 재등재]** t7336 OL-17/P-D(`T2_STALE_STRIP` `_wtools` 공백·발화 0) · OL-26/P-B(`T2_PRESCRIPTION`
   `role="tool"` 코퍼스) 둘 다 **이번 런에서도 미수리 확인**. 073 의 이번 실패 원인은 아니지만 부채로 남는다.
6. **[집계 규율]** t7328 대비는 **1/2 ↔ 1/2 가 아니라 0/2(유효) ↔ 1/2** 로 읽어야 한다(§5.1). 계보 정정 없이
   "회복"으로 적으면 t7328 의 무효 통과를 기준선으로 되살리는 셈이다.
