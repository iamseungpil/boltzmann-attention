# t7346 / task_072 포렌식 — ATM 수수료 분쟁 (양 trial 실패, reward 0/0)

- 대상 런: `bank_t7346_halfA_20260822` (tag `t7346`, sha `ee18d797`), sim `task_072#s626729`(trial 0) · `task_072#s373753`(trial 1)
- 대조: `bank_t7328_halfA_20260819r`(**동일 seed 쌍** 626729/373753) · 선행 보고서 `t7336_tasks/T7336_TASK_072.md`(t7336A, 동일 seed 쌍)
- 결론 한 줄: **t7336 의 "첫 마일 붕괴"는 수리됐다(2건 CONFIRMED 유효).** 실패는 t7335/t7328 이 서 있던 **마지막 마일(누락 rebate $2.00)** 로 되돌아왔고, 그 위에 **우리 층이 만든 새 손실 2건**(trial 0 = comparator 미도달 · trial 1 = 크레딧 **중복 실행**)이 얹혔다.

---

## 1. 채점축 (C583ⓖ 선행 확인)

`sim['reward_info']` 직접 확인:

| trial | seed | reward | `reward_basis` | `db_check` | `reward_breakdown` | 종료 |
|---|---|---|---|---|---|---|
| 0 | 626729 | 0.0 | **`["DB"]`** | `{"db_match": false, "db_reward": 0.0}` | `{"DB": 0.0}` | `user_stop` (77 msgs · 1071s) |
| 1 | 373753 | 0.0 | **`["DB"]`** | `{"db_match": false, "db_reward": 0.0}` | `{"DB": 0.0}` | `user_stop` (86 msgs · 1354s) |

⇒ **DB-해시 축**이다. `action_checks` 는 진단 보조표일 뿐이며 양 trial 모두 `072_0`~`072_6`(7건) 이 `action_match=true` 인데도 reward 는 0 이다([[69]]).

gold 변이 액션(진단용 참조 · `mutating_tools()` 기준):
- `072_0 log_verification{Liang Jinhai, lj82d4f1a9, …, time_verified "2025-11-14 03:40:00 EST"}`
- `072_7 apply_checking_account_credit_5829{chk_lj82d4f1a9, **14.00**, fee_refund}`
- `072_8 apply_checking_account_credit_5829{chk_538bfb9cba, **3.50**, fee_refund}`

---

## 2. 변이 집합 (정본 `t2_forensic.mutation_diff` 만 사용 · 손 비교기 0)

### trial 0 (s626729)
| 종류 | 내용 |
|---|---|
| MATCHED | `log_verification{…, time_verified="2025-11-14 03:40:00 EST"}` (msg 16 · ok) = gold `072_0` |
| **MISSING** | `apply_checking_account_credit_5829{chk_lj82d4f1a9, 14, fee_refund}` (gold `072_7`) — **Bluest 계좌를 아예 감사하지 않음** |
| **WRONGARG** | `apply_checking_account_credit_5829{chk_538bfb9cba, **3**, fee_refund}` (msg 66 · ok) ↔ gold `{chk_538bfb9cba, **3.5**, fee_refund}` |
| EXTRA / DUP / BLOCKED | 0 / 0 / 0 |

WRONGARG 필드별 대조: `account_id` 일치 · `credit_type` 일치 · **`amount` 3.00 vs 3.50 (Δ −0.50)**.

### trial 1 (s373753)
| 종류 | 내용 |
|---|---|
| MATCHED | `log_verification{…}` (msg 16) = gold `072_0` · `apply_checking_account_credit_5829{chk_538bfb9cba, **3.5**, fee_refund}` (msg 70) = gold `072_8` |
| **MISSING** | `apply_checking_account_credit_5829{chk_lj82d4f1a9, 14, fee_refund}` (gold `072_7`) |
| **WRONGARG** | `apply_checking_account_credit_5829{chk_lj82d4f1a9, **12**, fee_refund}` (msg 70 · ok) **및 msg 81 재실행** ↔ gold `amount=14` (Δ −2.00) |
| **DUP** | `apply_checking_account_credit_5829{chk_538bfb9cba, 3.5, fee_refund}` **msg 81 재실행**(msg 70 과 동일 인자) |
| BLOCKED | 4건 — msg 63 ×2 `credit_amount` 오인자(env `Invalid arguments … unexpected keyword argument 'credit_amount'`) · msg 67 ×2 `description` 오인자(동형) |
| EXTRA | 0 |

⚠**중복이 단독으로 치명적이다.** env 는 msg 81/82~83 에서 같은 크레딧을 **또 반영**했다:
`chk_lj82d4f1a9 127462.00 → 127474.00` · `chk_538bfb9cba 1851.00 → 1854.50`. gold 최종 잔액은
`127450+14 = 127464` / `1847.50+3.50 = 1851.00` 이므로, **금액이 옳았더라도 중복만으로 DB 해시는 깨진다.**

---

## 3. Step-by-step 결정 지점 추적

### 3.0 두 trial 공통 — 신원 확인은 완결 (t7336 회귀 해소)

t7336 trial 0 을 죽였던 GB1 Recovery 의 7-필드 되묻기(`name, user_id, address, email, phone_number, date_of_birth, time_verified`)가 **재발하지 않았다.** 이번 문면은 2필드로 좁혀져 있다:

> [2] assistant: `"before we can check your account transactions, we need to verify your identity. Could you please provide me with **two of the following** pieces of information: your date of birth, email, phone number, or address?"`
> `[Note: the tool call(s) above were blocked by a policy gate and were NOT executed. … ([GB1_VERIFY_BEFORE_ACCOUNT_ACCESS] blocked by policy gate: customer identity verified and logged...)]`

→ [12] `get_user_information_by_name` → [14] `verify_identity` → [15] `VERIFIED — 4 identity values match …` → [16] `log_verification{…}` → [17] `Verification logged successfully.` **양 trial 동일** (gold `072_0` MATCHED).

### 3.1 trial 1 (s373753) — 계좌 id 획득은 성공, 마지막 마일에서 실패 + 중복 실행

**결정 지점 ① [40] comparator 를 날조 id 로 호출 → [41] READ-FIRST 가 생산자를 이름으로 댐 (t7336 P-A 수리 발화·CONFIRMED 유효)**

> [40] assistant: `get_atm_fee_discrepancies{"account_id": "Bluest_Account_ID", "account_class": "Bluest Account", "transactions": "[{'transaction_id': 'txn1', …}]"}` (+ 같은 턴에 `"Light_Green_Account_ID"`)
> [41] tool(우리 층): `Error: [READ-FIRST] this audit only judges fee lines that were READ in this conversation, and these required reads are missing: **get_all_user_accounts_by_user_id**. get_all_user_accounts_by_user_id is the read that returns the customer's accounts ("Retrieve all accounts (checking, savings, credit cards) for a customer.") - it is where the checking account's id and its account class come from (an account's display class name like 'Purple Account' is NOT an account id). … Their exact callable forms are: unlock_discoverable_agent_tool(agent_tool_name="**get_all_user_accounts_by_user_id_3847**") then call_discoverable_agent_tool with that name.`

**즉시 이행됐다.**
> [43] `unlock_discoverable_agent_tool{get_all_user_accounts_by_user_id_3847}` (gold `072_1`) → [45] `call_discoverable_agent_tool{…, "{\"user_id\": \"lj82d4f1a9\"}"}` (gold `072_2`) → [46] `chk_lj82d4f1a9`(Bluest) / `chk_538bfb9cba`(Light Green) 실물 획득 → [47] 두 계좌 거래 read(gold `072_4`/`072_5`).

**결정 지점 ② [50]→[53] comparator 정상 통과 — 두 계좌 모두 감사됨**
[50] 은 `transactions` 를 파이썬-리터럴(작은따옴표) 문자열로 보내 우리 `T2_SG_ARGS` 가 반려했고([51]/[52] `Error: [ARGS-FORMAT] … Re-issue this exact call with 'transactions' as a VALID JSON a…`), [53] 에서 정상 JSON 으로 재송신되어 통과했다.

> [54] tool: `ATM fee lines whose charged amount does NOT match … btxn_fcd7ef3a24ed (charged $2.50, documented fee $2.00, difference $0.50); btxn_6a3453e0afd9 (charged $3.50, documented fee $0.00, difference $3.50); btxn_344585b826eb (charged $8.00, documented fee $0.00, difference $8.00). **SCOPE OF THIS CHECK - fee-line amounts only** … **This tool did NOT check whether any rebate is missing** - i.e. whether the account's documented rebate policy promises a fee_rebate line that the transaction history does not show; **check the account's rebate policy against the fee_rebate lines yourself before crediting.** If corrections are owed, the credit policy requires ONE fee_refund credit for the net correction of THIS account (do not credit the same lines twice).`
> `[coverage] 8 of 8 rows were checked (0 could not be verified).`
> [55] tool: `… btxn_49c0c0b3b8c1 (charged $4.00, documented fee $3.50, difference $0.50); btxn_e00b60651fca (charged $5.00, documented fee $2.00, difference $3.00). …` + `[coverage] 4 of 6 rows were checked (2 could not be verified).`

**★결정 지점 ③ — 마지막 마일. [58] 에서 모델은 fee-line 3건만 더해 Bluest = $12.00 으로 확정한다.**
> [58] assistant: `"For the **Bluest Account** (chk_lj82d4f1a9): - Refund $0.50 for transaction btxn_fcd7ef3a24ed. - Refund $3.50 for transaction btxn_6a3453e0afd9. - Refund $8.00 for transaction btxn_344585b826eb."`

**그 시점 문맥에 필요한 값은 실재했다.**
(a) `T2_SEARCH_REARM` 이 turn=40 에 `bluest_account`/`light_green_account` 문서 **23편·19,529자**를 배달했고(§4), Bluest 리베이트 조항은 그 안에 있다(`bluest_010/_003/_007` = *third-party ATM 수수료 월 $50 캡*).
(b) [48] 의 Bluest 거래 32건에 `ATM FEE REBATE +2.0` 라인이 **11/20·11/18·11/10·11/05·11/02 에는 있고 11/14 에는 없다**:
> `11. btxn_44579729e7c3 … 11/14/2025 ATM WITHDRAWAL - CITIBANK #9012 SKOKIE IL -100.0 atm_withdrawal`
> `12. btxn_63306834d5ba … 11/14/2025 NON-RHO ATM FEE -2.0 atm_fee`  ← **뒤따르는 `fee_rebate` 라인 없음**
(c) 우리 도구가 [54] 에서 **이 축을 명시적으로 지목**하고 스스로 하라고 말했다(위 축자).
⇒ **$14.00 − $12.00 = $2.00 = 11/14 누락 rebate.** 모델은 (a)(b)(c) 를 다 가진 채 하지 않았다.

**★결정 지점 ④ — [74] 이후 크레딧 중복 실행(우리 층 강제).**
[70]/[71]/[72] 에서 두 크레딧이 성공했고 [73] 에서 모델이 결과를 복창했다. 손님 [74] 는 **확인 질문**일 뿐이다.
> [74] user: `"Just to confirm, the total credited back was **$12.00** to my **Bluest Account** and **$3.50** to my **Light Green Account**, right? … Is there anything else I need to do on my end?"`

이 턴 이후 로그가 찍은 것:
```
[T2_ACTIONREQ] window=open pending_user=['apply_for_credit_card','call_discoverable_user_tool','submit_referral','submit_transaction']
               pending_agent=[...] formalized_target=call_discoverable_agent_tool
[T2_FORCE_ACTION] say-don't-do → tool_choice=required 재생성      ← ×3 (turn 75/79/81)
[T2_RESOLVE] action-required reason=discovery-required target=call_discoverable_agent_tool
[T2_MATERIAL_GATE] stop=resolve_cap(정체 3회) turn=81
```
모델은 할 일이 없는 채로 3회 강제 호출당했고([75] `KB_search_bm25` · [77] `unlock submit_interest_discrepancy_report_7294` · [79] `KB_search_bm25` 동일 질의), **[81] 에서 이미 실행한 크레딧 두 건을 그대로 재발행**했다.
> [81] `apply_checking_account_credit_5829{chk_lj82d4f1a9, 12.0, fee_refund}` + `{chk_538bfb9cba, 3.5, fee_refund}`
> [82]/[83] `Credit applied successfully! … Previous Balance: $127462.00 - New Balance: $127474.00` / `… $1851.00 → $1854.50`

`resolve_cap` 은 turn=81 **이후에야** 멈췄다 — 즉 캡이 **중복 write 를 한 건 통과시키고 나서** 닫힌다.

### 3.2 trial 0 (s626729) — comparator 를 Bluest 에 **한 번도** 쓰지 않음

**결정 지점 ① [8] 스키마 예시값 검증 시도 → 우리 층이 정상 처리**
> [8] `verify_identity{"provided": "{\"email\": \"john.doe@example.com\", \"address\": \"123 Main St, Anytown, USA\"}"}` → [9] `NOT_VERIFIED — the account record has not been fetched yet …`
이후 [12]~[17] 로 정상 복구(§3.0).

**★결정 지점 ② turn=36 — comparator 호출 3건이 `T2_PROV` regen 에 흡수되어 `T2_SG_REQREADS` 에 **도달하지 못함**.**
로그(축자):
```
[T2_MATERIAL_GATE] stop=other_lever(prov) turn=36
[T2_PROV] regen fired tool=get_atm_fee_discrepancies arg=account_id val=@last:get_bank_account_transactions_9173
[T2_PROV] regen fired tool=get_atm_fee_discrepancies arg=account_id val=@last:get_bank_account_transactions_9173
[T2_PROV] regen fired tool=get_atm_fee_discrepancies arg=account_id val=bluest_account_number
```
`T2_SG_REQREADS` 는 이 sim 에서 **0회**(trial 1 은 2회). PROV 는 호출을 **실행 전에** 되돌리므로 READ-FIRST 문면(= `get_all_user_accounts_by_user_id_3847` 를 **이름으로** 대는 유일한 채널)이 발화할 기회 자체가 사라졌다. PROV 가 대신 내보낸 문면은 이름이 없다(`REGEN_FEEDBACK`, `t2_gate_patch.py:1103`):
> `Error: [PROVENANCE] argument 'account_id'='…' was not provided by the user nor returned by any tool — it looks invented … **Do NOT use placeholder/example values and do NOT ask the user.** Instead call a lookup/getter tool that produces this value (e.g. a getter to retrieve the user's records, payment methods, or addresses) …`

그 턴의 실제 출력은 **정확히 그 문면이 금지한 행위**였다:
> [36] assistant: `"To help you locate these account numbers, here are the steps you can follow in your online banking portal: 1. Log in to your Rho-Bank online portal. … Once you have the account numbers, please provide them to me"`
> [37] user: `"I'm not able to access my online portal at the moment … Is there another way you can look up the two checking accounts?"`

모델은 [38] `KB_search_dense`, [40] `call_discoverable_agent_tool{get_bank_account_transactions_9173, "{\"account_id\": \"lj82d4f1a9_Bluest\"}"}`(디스패처 **내포** 인자라 PROV 무통과 — 선행 보고서 ⑤ 미수리) 를 거쳐 **[42] 스스로** `unlock get_all_user_accounts_by_user_id_3847` 에 도달했다. **손님 턴 2개(29·37)와 모델 턴 4개를 우리 층이 태웠다.**

**★결정 지점 ③ [48] — Bluest 를 comparator 없이 "육안"으로 무결 판정.**
[46]/[47] 에서 실물 id 와 Bluest 거래 32건을 손에 넣은 직후:
> [48] assistant: `"### Summary: - **Total Non-Rho ATM Fees**: $12.50 - **Total ATM Fee Rebates**: $10.00 - **Net ATM Fees Paid**: $2.50 … Based on the transactions, the ATM fees appear to be consistent with the Bluest Account benefits, which include a **$50 monthly ATM fee rebate**. The fees charged and rebates applied seem to align with the account's benefits."`

리베이트 정책 인용 자체는 **옳다**(`bluest_010` 월 $50 캡). 그런데 같은 요약이 **11/20 $2.50(정답 $2.00)·11/12 Tokyo $3.50(Bluest foreign = $0.00)·11/08 London $8.00(동)** 을 전부 "일치"로 넘겼고, **자기가 센 rebate 5건 ↔ fee 5건 중 11/14 만 rebate 없음**도 못 봤다. [50] 에서 Light Green 도 같은 방식으로 `"the ATM fees appear to be consistent with the Light Green Account's fee structure"` 로 종결한다.

**전 궤적에서 `get_atm_fee_discrepancies` 가 `chk_lj82d4f1a9` 로 호출된 적은 0회다**(호출 3건 전부 `chk_538bfb9cba`, msg 52/54/56).

**★결정 지점 ④ [51] — user-sim 이 오류 하나를 직접 지목한 뒤에야 comparator 사용.**
> [51] user: `"On the **11/13/2025** transaction (HSBC London UK), you listed the withdrawal as **$80.00**, but then the fee is shown as "Tier 3 fee for above $300 withdrawal" and charged **$5.00**. That doesn't make sense for an $80 withdrawal. Can you double-check that specific fee rule application …"`
> [52] `get_atm_fee_discrepancies{"account_id": "chk_538bfb9cba", "account_class": "Light Green Account", "transactions": "[{…btxn_e00b60651fca, 5.0, 80.0, foreign}]"}` — **손님이 짚은 한 행만** 전달
> [53] tool: `… btxn_49c0c0b3b8c1 (charged $4.00, documented fee $3.50, **difference $0.50**); btxn_e00b60651fca (charged $5.00, documented fee $2.00, **difference $3.00**). …` + `[coverage] 4 of 6 rows were checked (2 could not be verified).`

격리 서브가 원천 26행에서 6개 fee 라인을 스스로 추출했으므로(`[T2_SG_ISOLATE] operand-size … sub=6 rows · source=26 rows ⚠MISMATCH`), **모델이 한 행만 넣었는데도 두 건이 반환**됐다.

**★결정 지점 ⑤ [60]/[66] — 같은 문장 안에 있는 $0.50 을 버리고 $3.00 만 크레딧.**
> [60] assistant: `"The charged fee of $5.00 should have been $2.00 based on the documented fee schedule. I will process the refund of **$3.00** to your account."` → `call_discoverable_agent_tool{"refund_atm_fee_discrepancy_8473", …}` → [61] `Error: Unknown agent tool …`
> [66] `call_discoverable_agent_tool{apply_checking_account_credit_5829, "{\"account_id\": \"chk_538bfb9cba\", \"amount\": 3.0, \"credit_type\": \"fee_refund\"}"}` → [67] `Credit applied successfully! … Amount: $3.00`

도구 반환문은 같은 문장에서 **"ONE fee_refund credit for the net correction of THIS account"** 를 요구했고 두 라인의 차액을 모두 인쇄했다($0.50 + $3.00). 모델은 **손님이 언급한 라인만** 골랐다.
⚠[[70]] 관점의 사실 기록: `{delta_total}`(= 합계 $3.50)은 **2026-08-19 에 return_template 에서 제거**됐다(`_note_delta_total_removed_2026_08_19`, 사유 = 채점되는 인자 그 자체이므로 [[62]]/[[03b]]). 그 제거의 효과는 *"미측정"* 으로 남아 있었고, **이 궤적이 그 미측정 칸의 실물 관측이다**(합을 안 주니 모델이 부분합을 냈다). Light Green net = gold $3.50 과 정확히 같으므로, 합계만 보였다면 이 WRONGARG 는 나지 않았을 것이다 — 그러나 **되살리는 것은 [[23]] 위반 위험이 있으므로 처방으로 제안하지 않는다**(§7 참조).

**결정 지점 ⑥ [69]/[75] — 확인 요구에도 Bluest 재감사 없음.**
> [69] user: `"was anything credited to the **Bluest Account** as well, or was that one fully correct?"`
> [75] assistant: `"**No additional credits were applied** to the Bluest Account. The ATM fees charged were consistent with the account's benefits … Therefore, no discrepancies were found for this account."`

`T2_COVERAGE_FU` 가 이 구간에서 1회 발화했으나(§4·§6-③) 지목 대상은 **Light Green 의 미판정 2행**이었고, 그 2행은 **선언상 판정 불가**라 이행 가능한 지시가 아니었다.

### 3.3 분기점 특정 (trial 0 ↔ trial 1)

**[40] 이 갈림이다.**
- trial 1 은 `account_id="Bluest_Account_ID"` 를 **`get_atm_fee_discrepancies` 직접 인자**로 실었다 → PROV 통과 → **`T2_SG_REQREADS` 발화 → 생산자 이름 획득 → 두 계좌 모두 감사**.
- trial 0 은 같은 종류의 날조를 turn=36 에 세 번 냈으나 **PROV 가 전부 먼저 잡아 regen** 했다 → READ-FIRST 미발화 → 이름 미획득 → 손님에게 되묻기 2왕복 → 자력 복구 후에는 comparator 노선을 **포기**하고 육안 감사로 전환.

즉 **우리 층의 두 게이트가 같은 결함(날조 `account_id`)에 대해 정반대 품질의 문면을 내고, 먼저 잡는 쪽이 이긴다.** 이름을 대는 쪽(SG_REQREADS)이 이기면 태스크가 살아나고, 이름을 안 대는 쪽(PROV)이 이기면 죽는다.

---

## 4. 레버 발화 대조 (이 sim 줄만 · `[TAG]` 카운트)

| 레버 | t0 | t1 | runA 전체 | 판정 |
|---|---|---|---|---|
| `T2_SG_DOCS` | 0 | 0 | 5 | **미발화**(이 태스크 도달 0) |
| `T2_PIN_READ` | 0 | 0 | 4 | **미발화** |
| `T2_DEMANDED_STEP` | 0 | 0 | 12 | **미발화** |
| `T2_CLAIMPROV` | 33 | 41 | 538 | 발화·**무해무익**. `window hit(resign) claims=5 unbacked=0`·`kind-index rescued: kind='search' tool='get_atm_fee_discrepancies' 원장에 있다` — 산문 주장 검산은 **행동 주장만** 보므로 [48] 의 *"fees appear to be consistent"*(오판정)는 원리상 대상 밖 |
| `T2_FOLLOWUP` | 0 | 0 | 15 | **미발화** |
| `T2_SEARCH_AGENT` | 20 | 16 | 155 | 발화. turn=2 `[T2_DOCDECIDE] → 'Blue Account'` = **오결정**(손님은 *Bluest*/*Light Green* 이라 말했다) → 이후 `요청 축 … 모두 처리됨 — 침묵` 8연속 |
| `T2_SEARCH_REARM` | 2 | 2 | 14 | **발화·정상·구제**. `신규 대상 bluest_account,light_green_account (기배달 blue_account) — 소진 해제·문서 델타` → `델타 배달 19529자 (문서 23)` turn=36/40. DOCDECIDE 오결정을 스스로 되돌렸다 |
| `FAB_STRIP` | 0 | 0 | **0** | **미발화 — 런 전체 0회**(선행 보고서와 동일). `arg_source_reads` 의 유일 소비 경로(`_fab_fix_note`)가 여기 걸려 있어 P4 는 이번에도 도달 0 |
| `T2_ARG_PRODUCERS` | 0 | 0 | **0** | **미발화 — 런 전체 0회** |
| READ-FIRST (`T2_SG_REQREADS`) | **0** | **2** | 4 | **t1 발화·CONFIRMED 유효**(§3.1①·§5). t0 은 PROV 선점으로 **미발화** |
| `T2_SG_ARGS` | 0 | 2 | 2 | t1 발화·정상(파이썬 리터럴 반려 → [53] 재송신 성공) |
| `T2_SG_ISOLATE` | 12 | 8 | 88 | 발화·정상. `operand-size … sub=6 rows · source=26 rows ⚠MISMATCH`(t0) — 모델이 1행만 줬는데 서브가 6행 전부 추출 |
| `T2_COVERAGE_FU` | 1 | 1 | 4 | **발화·오발화**(§6-③). 지목 대상이 **선언상 판정 불가**한 행이라 이행 불가능한 지시. cap=1/sim 이라 이걸로 소진 |
| `T2_PROV` | **11** | 4 | 65 | **발화·t0 에서 유해**(§6-②). t0 내역: `get_atm_fee_discrepancies.account_id` **4회**(`Bluest-Account-12345`·`@last:get_bank_account_transactions_9173`×2·`bluest_account_number`) + `email=john.doe@example.com` 1 + 이름-인자 3 |
| `T2_FORCE_ACTION` | 18 | 14 | 103 | **발화·t1 에서 유해**(§6-①). [74] 이후 3회 강제 → [81] 중복 write |
| `T2_UNCALLED_UNLOCK` | 1 | 1 | 10 | 발화(`surface get_bank_account_transactions_9173`) |
| `T2_WRITE_SUB` | 23 | 16 | — | 발화. 근거검산 **통과 8건(t0)·6건(t1)** — 死배선 아님. 다만 통과분이 금액 오류를 바꾸지 못함 |
| `T2_REQUIRE_DOC_DELIVER` | 0 | 0 | 17 | **미발화** |
| `T2_WRITE_ARG_GROUND` / `T2_OWNERSHIP_FIX` / `T2_VALUE_ACQUIRE` | 0/0/0 | 0/0/2 | 4/5/6 | t1 `T2_VALUE_ACQUIRE` 2회 — 무해 |
| `T2_COMPUTE` | 3 | 1 | — | `select_discrepant: 2/6행 판정불가(operand가 숫자 아님) — under-action 위험` (Light Green 만) |

### 직전 수리(t7336 마스터)의 이 궤적 개입 여부

| 수리 | 개입 | 결과 |
|---|---|---|
| **A6①/OL-37 `requires_reads += get_all_user_accounts_by_user_id`** (`a2/banking_knowledge.specific.json` `scaffold_get_tools[8].requires_reads`) | **○ (t1)** | **CONFIRMED 유효**. `[41]` 이 도구 이름을 축자로 댔고 `[43]/[45]` 가 즉시 이행 → **t7336/t7328 t1 이 못 넘던 첫 마일 통과**. gold `072_1`~`072_5` 5건이 새로 `action_match=true` 가 됐다. ⚠t0 에는 **도달 0**(PROV 선점) |
| **GB1 Recovery 문면 축소**(t7336 §7-①) | **○ (양 trial)** | **CONFIRMED 유효**. 7-필드 되묻기 소멸 → t7336 t0 의 `too_many_errors` 루프 **재발 0**. `log_verification` 이 양 trial 모두 [16] 에서 성공 |
| P5 rebate-scope 문면(2026-08-21) | **○ (t1 [54]/[55] · t0 [53])** | **발화했으나 무시**. *"check the account's rebate policy against the fee_rebate lines yourself before crediting"* 이 축자로 도달했고 모델은 rebate 검사를 **0회** 했다 |
| P-B PROV 폴백 → `arg_source_reads` | **×** | **미착수**. `default_specs=0 · regen_resolver_specs=0`(banking) → `_resolver_directive` 항상 `None` → `REGEN_FEEDBACK`(무명). t0 의 사인 |
| ⑤ 디스패처 내포 인자 PROV 전개 | **×** | **미착수**. t0 [40] `call_discoverable_agent_tool{…,"{\"account_id\": \"lj82d4f1a9_Bluest\"}"}` 가 그대로 통과 |
| P4 `FAB_STRIP` 해소-read 지목 | **×** | 런 전체 0회 — 도달 0(3연속 런 동일) |

---

## 5. 선행 판정과의 대조

| 런 | seed 626729 (t0) | seed 373753 (t1) |
|---|---|---|
| t7328A | comparator 통과·크레딧 2건 → **WRONGARG $12↔$14** (LG $3.50 MATCHED) | 계좌목록 read 성공했으나 **MISSING ×2** |
| t7335A | comparator 통과 → **WRONGARG $12↔$14**(누락 rebate $2.00) | — |
| t7336A | **첫 마일 붕괴**(GB1 되묻기 루프·`too_many_errors`·측정 불가) | **첫 마일 붕괴**(계좌 id 날조·이관·MISSING ×2) |
| **t7346A** | **MISSING(Bluest 전체) + WRONGARG $3↔$3.50** — comparator 를 Bluest 에 0회 사용 | **WRONGARG $12↔$14 + DUP ×2** — 두 계좌 감사 성공, 마지막 마일 실패 |

**원인 판정 변화 (명시):**
- **t7336 대비 = 원인이 달라졌다.** t7336 의 주 원인 2건(GB1 Recovery 문면 · 계좌목록 생산자 무명)은 **수리로 해소됐다** — 재발 0, 첫 마일 통과. 이번 실패는 t7336 판정과 **다른 자리**다.
- **t7335 대비(seed 626729 · t1 축) = 같은 원인이 살아 있다.** t7335 판정 *"우리 층 — `get_atm_fee_discrepancies` 의 누락-rebate 비커버(마지막 마일 $2.00)"* 가 **t7346 t1 에서 축자로 재현**($12 vs $14). P5 문면 승격만으로는 닫히지 않았다.
- **t7328 대비(seed 626729) = 하드 회귀.** t7328 t0 은 두 계좌를 감사하고 LG 를 **정답 $3.50** 로 맞췄는데, t7346 t0 은 **Bluest 감사 0회 · LG $3.00**. 즉 이 seed 에서 t7346 은 t7328/t7335 보다 **뒤로 갔다**.
- **t7346 의 새 손실:** trial 1 의 **크레딧 중복 실행**은 t7328/t7335/t7336 어디에도 없던 변이다(`dup` 0 → 2).

---

## 6. 원인 확정

### ① trial 1 주 원인 A — `our_layer` [CONFIRMED] · 종결 후 강제-행동이 크레딧을 중복 실행
`t2_gate_patch.py:9105`(`T2_FORCE_ACTION` → `force_required=True`)가 **모든 write 가 이미 완료되고 손님이 확인 질문만 한 턴**에서 3회 연속 발화해 `tool_choice=required` 를 강제했고, 그 마지막 강제가 [81] 의 중복 크레딧을 만들었다.
그 전제인 "행동이 미완"이라는 판정은 `t2_gate_patch.py:8236-8251` 이 만든다:
```python
_effall = {_eff_tool_name(tc) for m2 in state.messages for tc in (getattr(m2,"tool_calls",None) or [])}
_upending = sorted(_uacts - _effall)
if _upending or (_acts and not (_called & _acts)):
```
`_uacts`(손님-측 액션 도구)가 이 태스크와 무관한 `apply_for_credit_card` / `call_discoverable_user_tool` / `submit_referral` / `submit_transaction` 4종이고 **이 태스크에서는 영원히 호출되지 않는다** ⇒ `_upending` 이 **sim 내내 비지 않는다** ⇒ `window=open` 이 종결 뒤에도 닫히지 않는다. 로그가 그대로 보여준다: `pending_user=['apply_for_credit_card','call_discoverable_user_tool','submit_referral','submit_transaction']` 가 **86 msgs 내내 동일**.
`T2_RESOLVE_CAP`(`stop=resolve_cap(정체 3회) turn=81`)은 **중복 write 가 나간 뒤** 닫혔다.
- 근거: [73] 완료 복창 → [74] 손님 확인 질문 → 로그 `T2_FORCE_ACTION` ×3(turn 75/79/81) → [81] 동일 인자 재발행 → [82]/[83] 잔액 **재차 증가**.
- 부: `model` — 강제받은 자리에서 "이미 했다"고 답하는 대신 같은 write 를 골랐다.

### ② trial 0 주 원인 — `our_layer` [CONFIRMED] · PROV 가 READ-FIRST 를 선점하고 이름을 못 댐
`t2_gate_patch.py:9716-9733` 의 PROV regen 이 `get_atm_fee_discrepancies.account_id` 날조를 **호출 실행 전에** 되돌리므로, 같은 결함을 **이름으로 고쳐주는** `T2_SG_REQREADS`(scaffold_get 실행 경로)가 발화할 기회가 사라진다. 그리고 PROV 가 내보내는 문면은 생산자 이름이 없다 — `_resolver_directive`(`t2_gate_patch.py:1768`)가 `a2["default_specs"] + a2["regen_resolver_specs"]` 를 읽는데 banking 은 **둘 다 0건**이라 항상 `None` → `REGEN_FEEDBACK`(`t2_gate_patch.py:1103`) 폴백.
정답 이름은 **이미 우리 선언에 있다**: `a2/banking_knowledge.specific.json` → `arg_source_reads["account_id"] = ["get_all_user_accounts_by_user_id_3847"]`. 그러나 이 키의 유일 소비자는 `_fab_fix_note`(`t2_gate_patch.py:1890`)이고 그 호출처는 `FAB_STRIP` 하나뿐이며 **FAB_STRIP 은 런 전체 0회**다.
- 근거: 로그 `[T2_PROV] regen fired tool=get_atm_fee_discrepancies arg=account_id val=@last:… / bluest_account_number` ×3 (turn=36) · `T2_SG_REQREADS` **t0 0회 ↔ t1 2회** · t1 은 같은 결함이 SG_REQREADS 로 가서 [41]→[43]→[45] 로 **복구됨** · t0 은 [36] 에서 *"here are the steps you can follow in your online banking portal"* 로 손님에게 되묻기(= REGEN_FEEDBACK 이 명시적으로 금지한 행위).
- 부: `model` — [48]/[50] 육안 감사, [40] 내포-인자 날조.

### ③ 부수 원인 — `our_layer` [CONFIRMED] · `T2_COVERAGE_FU` 오발화(이행 불가능한 지시로 cap 소진)
`t2_gate_patch.py:11193-11209` 는 `[coverage] … (N could not be verified)` 만 보고
> `Error: [COVERAGE-FOLLOWUP] … reported rows it could not verify … **Read the missing value(s) from the records that contain them** and call '%s' again with the completed input for those rows`
를 낸다. 그런데 이 궤적의 `skipped=2` 는 **operand 결손이 아니라 우리 선언의 기대값 부재**다 — `scaffold_get_tools[8].op.steps.oon.cases["Light Green Account"] = null`(선언 `_note_`: *"light_green/light_blue oon … 월 무료횟수 규정이 문서에 불완전(모호점 8·10)→기대 null=판정 보류"*). `t2_compute.py:934` 의 `if en is None or act is None:` 이 **엔진-측 `en is None`** 과 **입력-측 `act is None`** 을 한 칸에 합치고, `_missing`(입력 필드 계상)은 이 경우 **비어 있다**(실제 도구 출력이 `[coverage]` 줄에서 끝나고 필드 지목 문장이 없다는 사실이 이를 확인한다).
같은 자리의 계기도 거짓을 말한다: `[T2_COMPUTE] select_discrepant: 2/6행 판정불가(**operand가 숫자 아님**)` — operand 3필드는 모두 정상 숫자였고 없던 것은 **우리 요율**이다([[25]] 위반).
결과: cap 1/sim 인 이 레버가 **닫을 수 없는 구멍**에 소진돼, 정작 이행 가능한 두 구멍(t0 = Bluest 미감사 / t1 = 누락 rebate)에는 영영 못 갔다.
- 근거: t0 `[T2_COVERAGE_FU] fired tool=get_atm_fee_discrepancies` 1회(turn 58~60 구간) · t1 1회([55] 직후 → [56] 은 같은 계좌 거래 **재-read**로 소모) · 두 경우 모두 미판정 행 = Light Green non_rho 2건.

### ④ 잔여 원인 — `model` [CONFIRMED] · 마지막 마일(누락 rebate + 부분합)
- t1 [58]: 도구가 *"This tool did NOT check whether any rebate is missing … check the account's rebate policy against the fee_rebate lines yourself before crediting"* 라고 **축자로 지시**했고, 정책 문서 23편이 turn=40 에 배달됐으며, [48] 거래 32건에 11/14 rebate 부재가 **가시적**이었다. 그럼에도 rebate 검사 0회 → $12.00.
- t0 [60]/[66]: 같은 문장 안의 두 차액($0.50 + $3.00)을 합하지 않고 **손님이 언급한 라인만** 크레딧 → $3.00.
- ⚠[[70]] 매매 기록: t0 의 부분합은 **2026-08-19 `{delta_total}` 제거**의 부작용이 실측된 첫 사례다(그 `_note_` 는 *"효과는 미측정"*). 다만 그 값은 **채점되는 인자 그 자체**이므로 되살리기는 [[23]]/[[62]] 위반 — §7 은 다른 경로를 제안한다.

### ⑤ env / user_sim
- `env` — 결함 없음. `Account 'X' not found` · `Unknown agent tool` · `unexpected keyword argument 'credit_amount'/'description'` 전부 정확한 거부. **단 [82]/[83] 에서 동일 `transaction_id` 로 잔액을 재차 증액**한 것은 env 의 멱등성 부재이며, 우리 층이 중복을 만들지 않으면 노출되지 않는다(면책 아님).
- `user_sim` — 시나리오 준수. t0 [51] 은 오히려 오류 하나를 **직접 지목**해 comparator 사용을 유도했다(도움). [21] 원칙상 면책 사유로 쓰지 않는다: 손님이 짚어주지 않으면 감사를 안 하는 것이 결함이다.

---

## 7. 처방 후보 (제안만 · 실행·코드 수정 없음)

1. **[P-1 · 최우선 · 중복 write 차단]** `T2_FORCE_ACTION`/`T2_ACTIONREQ` 의 `_upending` 에서 **이 대화에서 한 번도 요구되지 않은 손님-측 액션 도구**를 제외하거나, `pending` 판정에 **"직전 손님 발화가 확인 질문이고 새 요구가 없다"** 는 닫힌 술어를 추가한다. 최소 변경안: `T2_RESOLVE_CAP` 의 정체 카운트를 **write 성공 이력이 있는 sim 에서는 1**로 낮춰 중복 write **전에** 닫히게 한다. ⚠[[70]] 무엇을 파나 = 진짜 미완 write 가 있는 태스크에서 강제 1회를 잃는다 ⇒ **태스크별 부호표 필수**. 이 레버는 전 태스크 공유 상류 노드다([[66]]).
2. **[P-2 · t7336 P-B 재상신]** PROV 폴백 순서를 `_resolver_directive → arg_source_reads → REGEN_FEEDBACK` 로. `arg_source_reads["account_id"]` 가 이미 정답을 갖고 있고 소비 경로만 없다(신규 저작 0·레버 신설 0). t0 의 4회 무명 거부가 4회 유명 거부가 된다. **선행 3연속 런에서 미착수** — 우선순위를 올릴 것.
3. **[P-3 · COVERAGE_FU 오발화 제거]** `_sg_stats` 에 `skipped` 를 **`skipped_operand`(입력 결손·재-read 로 해소 가능) / `skipped_expected`(선언 abstention·해소 불가)** 로 분리(엔진 자기 집계의 분해일 뿐·판단 0). ⑴`T2_COVERAGE_FU` 는 전자에만 발화 ⑵후자는 [[64]] 대로 *"우리 스케줄에 이 계좌 등급의 out-of-network 요율이 없어 이 2행은 판정하지 않았다"* 로 **정직 표기**. 부수로 `[T2_COMPUTE] … (operand가 숫자 아님)` 의 거짓 문면도 교정된다([[25]]).
4. **[P-4 · 누락 rebate 축]** `_note_rebate_field` 가 보류한 `rebate_field` 를 되살리는 대신, **환급 라인 부재를 "판정"이 아니라 "표면화"로** 낸다: comparator 반환문에 *엔진이 이미 넘겨받은 행*에서 도출되는 사실만 병기 — `fee lines: N, fee_rebate lines: M`(단순 계수). t1 [54] 는 `fee 5 / rebate 4` 가 되어 모델이 볼 수 있다. **차액·정답 금액은 만들지 않는다**([[62]]·[[23]]). 효과는 격리 프로브로 먼저 잴 것([[18]]).
5. **[P-5]** 디스패처 내포 `arguments` JSON 을 PROV 스캔에 전개(t7336 ⑤·미착수) — t0 [40] 형 우회를 닫는다.
6. **[측정 의무]** `FAB_STRIP` · `T2_ARG_PRODUCERS` 가 **3연속 런 전체 0회**다. 수리 전에 `t2_liveness` 0단계로 배선 생존부터 확인할 것([[67]]).

> ⚠ 위 처방은 전부 **제안**이다. gold(`reward_info`)는 진단 목적으로만 읽었고 A2 내용 저작 근거로 쓰지 않았다([[23]]). §6-④ 의 `{delta_total}` 은 **되살리기를 제안하지 않는다**(채점 인자 위조 = [[62]]/[[03b]]).
