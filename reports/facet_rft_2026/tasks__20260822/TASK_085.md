# t7346 halfB task_085 per-step 포렌식 (2026-08-22 런 · 작성 2026-08-23)

- 런: `bank_t7346_halfB_20260822` (sha `ee18d797` · single all-on composed stack · `T2_ARG_DOC_SUB=1
  T2_VALUE_FORMULA=full T2_SG_DOCS=1` · nt=2). **trial-0 reward 0.0 · trial-1 reward 0.0 = 0/2.**
- 데이터: `sim_results/bank_t7346_halfB_20260822.results.json.gz` (`task_id='task_085'` — 접두 `task_`
  포함이 실제 키) · 로그 `bank_t7346_halfB_20260822.log.gz` 의 `[sim=task_085#s626729]` 473줄 /
  `[sim=task_085#s373753]` 445줄 전수.
- 방법: 변이 = 정본 `t2_forensic.mutation_diff`(손 비교기 0). 궤적 = 메시지 131/96 전수 정독. 인용 전부 축자.
- 대조: 선행 `T7335_NT1_FORENSIC_085_2026_08_21.md` · 기준선 `bank_t7328_halfB_20260819r2`(sha 상이).

---

## 0. 채점 축 확인 (방법 ①)

`sim['reward_info']` 직독:

```
reward_basis   : ['DB']
reward_breakdown: {"DB": 0.0}
db_check       : {"db_match": false, "db_reward": 0.0}
nl_assertions  : null      communicate_checks: null
info.action    : null
```

**DB-해시 축이다.** `action_checks` 는 12~13행이 실리고 대부분 `action_match=true` 로 보이지만
`reward_basis` 에 `ACTION` 이 없어 **성적에 0 기여**한다 — 진단용으로만 읽는다(C583ⓖ). 실제로
trial-0 은 `action_match` 가 10/13 참인데 reward 는 0.0 이다. 표를 ACTION 축으로 읽었으면
"거의 성공"이라는 거짓 결론이 나온다.

---

## 1. 변이표 (`t2_forensic.mutation_diff` · 정본)

| | gold | matched | MISSING | WRONGARG | EXTRA | DUP | BLOCKED | done |
|---|---|---|---|---|---|---|---|---|
| **trial-0** (seed 626729 · 131 msg · user_stop) | 4 | 1 | **3** | **4** | 0 | 0 | **18** | 5 |
| **trial-1** (seed 373753 · 96 msg · user_stop) | 4 | 1 | **3** | 0 | 0 | 0 | 6 | 1 |

`matched` 1건은 양 trial 동일 — `log_verification{Jordan Williams, f7d3a82c91, …, 2025-11-14 03:40:00 EST}` ✓.
MISSING 3건도 양 trial 동일 = **gold 의 dispute 3건 전부**.

### gold 가 요구한 3건 (`file_debit_card_transaction_dispute_6281`)

| aid | transaction_id | account | category | txn_date | disc_date | amount | txn_type | prov_credit |
|---|---|---|---|---|---|---|---|---|
| 085_6 | `btxn_a1b2c3d4e501` | chk_b4d92f7c28 | `atm_cash_discrepancy` | 11/05/2025 | **11/05/2025** | 100 | atm_withdrawal | **true** |
| 085_7 | `btxn_b2c3d4e5f602` | chk_b4d92f7c28 | `duplicate_charge` | 11/06/2025 | **11/08/2025** | 89.99 | **recurring_payment** | **true** |
| 085_9 | `btxn_f6g7h8i9j006` | chk_e8a31c9d47 | `recurring_charge_after_cancellation` | 11/09/2025 | **11/14/2025** | **14.99** | **recurring_payment** | **false** |

`btxn_e5f6g7h8i905`(Green 계좌 ATM $200)에 대한 gold 행은 **없다** — 이 건은 **안 걸어야** 한다.

### trial-0 WRONGARG 4건 — 필드별 대조

**① msg[87] `btxn_a1b2c3d4e501`** (gold 085_6 과 같은 거래·인자 3개 불일치)

| 필드 | 보낸 값 | gold |
|---|---|---|
| `dispute_category` | `incorrect_amount` | `atm_cash_discrepancy` |
| `discovery_date` | `11/06/2025` | `11/05/2025` |
| `provisional_credit_eligible` | `False` | `True` |

**② msg[95] `btxn_c3d4e5f6g703`** — **gold 에 대응 행이 없다**(gold 는 `btxn_b2c3d4e5f602`).
CityFit 중복 결제 2행 중 **틀린 쪽**을 골랐다. 부수로 `transaction_type=signature_purchase`(gold
`recurring_payment`) · `discovery_date=11/07/2025`(gold 11/08) · `provisional_credit_eligible=False`(gold True).

**③ msg[119] `btxn_e5f6g7h8i905`** — **gold 에 대응 행이 없다.** 걸지 말았어야 할 건을 걸었다
(**과행동**). `mutation_diff` 는 도구명이 gold 집합 안이라 `wrongarg` 버킷에 넣지만 실질은 EXTRA 다.
env 는 실행하며 `Provisional Credit: ISSUED - $200.00 credited within 10 business days` 를 냈다 —
**존재하지 않는 손실에 가지급을 냈다.**

**④ msg[127] `btxn_f6g7h8i9j006`** (gold 085_9 과 같은 거래·인자 4개 불일치)

| 필드 | 보낸 값 | gold |
|---|---|---|
| `disputed_amount` | **`49.99`** | **`14.99`** |
| `discovery_date` | `11/10/2025` | `11/14/2025` |
| `transaction_type` | `signature_purchase` | `recurring_payment` |
| `provisional_credit_eligible` | `True` | `False` |

### trial-1 BLOCKED 6건 — 동일 인자 5연속 재제출

```
msg[76] Error: Invalid arguments: … missing 5 required positional arguments:
        'account_id','card_id','user_id','transaction_date','discovery_date'
msg[78] Error: Missing required parameters.
msg[80] Error: Missing required parameters.   ← msg[78] 과 인자 완전 동일
msg[82] Error: Missing required parameters.   ← 동일
msg[84] Error: Missing required parameters.   ← 동일
msg[86] Error: Missing required parameters.   ← 동일
```

`mutation_diff` 로 인자 델타를 뽑으면 msg[78]→[86] 이 **delta {} (완전 동일)** 이다. 제출 인자는
`pin_compromised: false`(bool) · `discovery_date: "2025-11-14 03:40:00 EST"`(MM/DD/YYYY 아님).
trial-0 에서 같은 문구로 막힌 msg[75] 도 `pin_compromised: False` 였고, 성공한 msg[87] 은
`pin_compromised: "no"` 였다 — env 의 술어는 **falsy 값을 "missing" 으로 보고**하고 어느 인자인지
말하지 않는다.

---

## 2. step-by-step 결정 지점 추적 (trial-0 · 축자)

### (a) 계좌 식별 — **t7335 의 치명 결손이 닫혔다**

- [1] user: *"Hi, I need help with some disputes on my **debit cards**."*
- [11]~[19] `verify_identity` → *"VERIFIED — 4 identity values match the account record"* → `log_verification` **matched**.
- [25]~[33] `get_debit_cards_by_account_id_7823{account_id:"f7d3a82c91"}` ×2 → *"Error: Account 'f7d3a82c91'
  not found."* (user_id 를 account_id 로) · `shell grep -r 'Jordan Williams' .` ×2 → *"No matches found."*
- **[39]~[43] 자력 복구**: `unlock get_all_user_accounts_by_user_id_3847` → `call{user_id:"f7d3a82c91"}` →
  *"User accounts retrieved successfully."* → [43] *"you have two checking accounts"* →
  `get_debit_cards_by_account_id_7823{account_id:"chk_b4d92f7c28"}` **성공**.

t7335 에서 81 메시지를 통째로 태운 지점(3847 을 문서로 두 번 쥐고도 0회 호출)이 **이번엔 통과했다**.
`action_checks` 의 085_1/085_1a/085_2/085_2a/085_2b/085_3/085_4 가 전부 match 인 것이 그 증거다.
`arg_source_reads` 선언(`account_id → get_all_user_accounts_by_user_id_3847` · `card_id →
get_all_user_accounts_by_user_id_3847, get_debit_cards_by_account_id_7823`)이 실린 뒤의 첫 관측이다.

### (b) 필요한 값·규칙이 **전부 문맥에 실재했다** (부하 결손 아님)

- **[5] 도구 unlock 출력에 전체 시그니처가 실렸다**(2975자·turn 3):
  > `- pin_compromised: string (required) - … Must be one of: 'yes_shared', 'yes_observed', 'no', 'unknown'`
  > `- discovery_date: string (required) - Date when the user first noticed the issue, format MM/DD/YYYY`
  > `- transaction_type: string (required) - … 'recurring_payment', …`
- **[52]·[84] 필링 정본 문서 회수**(모델 자신의 `KB_search_bm25`/`KB_search_dense`):
  `doc_bank_accounts_bank_accounts_(general)_031 | Internal: Filing a Debit Card Transaction Dispute`,
  `_032 | Internal: Debit Card Provisional Credit Guidelines`,
  `_034 | Internal: Recurring Payment Disputes vs. Stop Future Payments`.
  doc 031 본문 축자:
  > *"**Dispute the earliest (first) transaction when multiple duplicates exist.**"*
  > *"'atm_cash_discrepancy': ATM dispensed wrong amount or no cash"*
  > *"7. **discovery_date** (string, MM/DD/YYYY) - Date customer first noticed the issue."*
  > *"'recurring_payment': Subscription or automatic payment"*

⇒ **[[62]] ① 결손 격리 관점에서 "전달 부족"이 아니다.** 카테고리·유형·중복선택·질문 의무가 전부
축자로 문맥에 있었고, 그 뒤에 벌어진 것은 전사 실패다.

### (c) dispute 1 — 8회 좌초 뒤 카테고리 오선택

- [56] user: *"On **November 5th**, I used **Rho-Bank ATM #4827** to withdraw **$300**, but the machine only
  dispensed **$200**. I'm missing **$100**. Category would be **ATM cash withdrawal discrepancy /
  incorrect amount dispensed**."*
- [68] `get_bank_account_transactions_9173{chk_b4d92f7c28}` → `btxn_a1b2c3d4e501 … RHO-BANK ATM #4827
  WITHDRAWAL · amount: -300.0 · type: atm_withdrawal` — **손님 주장과 일치**(정상 건).
- [69]~[81] 필링 8연속 실패. [75]→[81] 은 **인자 델타 {} 로 4회 동일 재제출**, env 는 매번
  *"Error: Missing required parameters."* 만 냈다(무명 거부·[[64]] 동형).
- [83] 모델이 `KB_search_dense{"file_debit_card_transaction_dispute_6281 required parameters"}` → [84] doc 031 도착.
- [85] 5개 인자를 한꺼번에 교정(`pin_compromised: false→"no"` · `card_action: null→"keep_active"` ·
  `dispute_category: "incorrect_amount_dispensed"→"incorrect_amount"`) → env 가 `customer_max_liability_amount`
  하나를 **이름 지어** 거부 → [87] 성공.
- **결정 지점**: [85] 에서 `incorrect_amount` 를 골랐다. 같은 메시지 [84] 안에 doc 031 의
  *"'atm_cash_discrepancy': ATM dispensed wrong amount or no cash"* 가 축자로 있었고 손님도 [56] 에서
  *"ATM cash withdrawal discrepancy"* 를 먼저 말했다. **문맥·손님 발화 둘 다 반대편을 가리키는데
  모델이 다른 enum 을 골랐다.** env 는 두 값 모두 유효 enum 이라 통과시켰다.
- `discovery_date` 는 [93] 형태의 *"(assuming you discovered it the next day)"* 로 만들었다 — 손님에게
  **묻지 않았다**. doc 031 은 *"Date customer first noticed the issue"*, user 시나리오는 *"If asked about
  discovery: 'I noticed right away on November 5th'"* 였다. **묻기만 하면 gold 값을 받는다.**

### (d) dispute 2 — 정책문이 문맥에 있는데 반대로 골랐다

- [90] user: *"On **November 6th**, **CityFit Gym** charged me **$89.99 twice** on the same day."*
- [92] tool 출력 (축자·순서 그대로):
  ```
  1. Record ID: btxn_c3d4e5f6g703 … date: 11/06/2025 … CITYFIT GYM MONTHLY DENVER … -89.99
  2. Record ID: btxn_b2c3d4e5f602 … date: 11/06/2025 … CITYFIT GYM MONTHLY DENVER … -89.99
  ```
- [93] assistant: *"The duplicate charges … 1. `btxn_c3d4e5f6g703` … 2. `btxn_b2c3d4e5f602` … Now, let's file
  the dispute for these duplicate charges."* → `transaction_id: "btxn_c3d4e5f6g703"`.
- **결정 지점**: doc 031 의 *"Dispute the earliest (first) transaction when multiple duplicates exist."* 가
  [84] 에 실려 있었다. 두 행의 날짜가 같아 "earliest" 를 뷰만으로 정하기 어렵다는 점은 사실이고
  (env 는 시각 필드를 안 주고 **역순으로 나열**한다 — …703 이 먼저), 그래서 이 한 건은
  **env 표현이 판정을 가린 정황**이 있다. 다만 레코드 id 자체가 `…e501/…f602/…g703/…h804` 순번을
  담고 있어 결정 가능하며, 무엇보다 **모델은 이 규칙을 인용하지도 적용하지도 않았다**(선택 이유를
  적지 않고 첫 줄을 집었다).
- 부수: `transaction_type` 을 [93] 에서 `debit_card_purchase`(레코드의 `type` 값 그대로)로 넣었다가
  [94] env 가 enum 을 이름 지어 거부하자 [95] 에서 `signature_purchase` 로 바꿨다. doc 031 의
  *"'recurring_payment': Subscription or automatic payment"* + 레코드 설명 *"CITYFIT GYM **MONTHLY**"* 가
  둘 다 문맥에 있었는데 반영 안 됐다.

### (e) dispute 3 — **이 태스크의 핵심 시험을 통째로 놓쳤다** (과행동)

- [98] user: *"On **November 8th**, I used **Rho-Bank ATM #5921** to withdraw **$500**, but I only got **$300**
  in cash. That's **$200 missing**."*
- [102] tool 출력 축자:
  ```
  2. Record ID: btxn_e5f6g7h8i905 … date: 11/08/2025 … RHO-BANK ATM #5921 WITHDRAWAL … amount: -200.0
  ```
  → **DB 의 인출액은 $200 이다. 손님이 말한 $500 인출은 존재하지 않는다.** 즉 "$300 만 나왔다"는
  기억 착오이고 분쟁 대상 자체가 없다.
- [103] assistant (축자):
  > *"The ATM withdrawal transaction on November 8th, 2025, is as follows: - **Amount**: -$200.00 …
  > Now, let's file the dispute for this ATM cash withdrawal discrepancy. … - **Disputed Amount**: `$200`"*

  **레코드 총액 $200 을 그대로 "분쟁 금액 $200" 으로 옮겼다.** 손님 주장($500 인출 / $300 수령)과의
  대조가 한 줄도 없다. 우연히 숫자가 겹쳐 모순이 은폐됐다.
- [103]~[117] 8회 좌초(`incorrect_amount_dispensed` 무효 enum·`description`/`dispute_description` 미존재
  인자) 뒤 [118] env 가 enum 을 이름 지어 거부 → [119] `atm_cash_discrepancy` 로 제출 → 실행됨.
- [121] assistant: *"**Provisional Credit**: ISSUED - $200.00 credited within 10 business days per Regulation E."*
- [122] user: *"Great, thanks."* — 시나리오의 *"If agent files it: 'Great, thanks.'"* 분기. **user-sim 정상**.
  시나리오의 교정 분기(*"If agent says the transaction shows a different amount … Let's skip this one."*)는
  **에이전트가 대조를 안 해서 열리지 않았다.**

### (f) dispute 4 — 직전 메시지의 레코드 값을 손님 주장으로 덮어썼다 (결정적)

- [122] user (축자): *"there's a charge from **Prime Streaming Service** on **November 9th** for **$49.99**. …
  I **just noticed it today** when reviewing my statements."*
- [124] tool 출력 (축자·직전 메시지):
  ```
  1. Record ID: btxn_f6g7h8i9j006 … date: 11/09/2025 … PRIME STREAMING SERVICE … amount: -14.99
  ```
- [125] assistant (축자·**바로 다음 메시지**):
  > *"The transaction for the Prime Streaming Service charge on November 9th, 2025, is as follows:
  > - **Transaction ID**: `btxn_f6g7h8i9j006` - **Date**: 11/09/2025 - **Description**: PRIME STREAMING SERVICE
  > - **Amount**: **-$49.99** - **Type**: debit_card_purchase"*

  **직전 도구 출력이 `-14.99` 인데 `-$49.99` 로 적었다.** "레코드는 이렇다"고 선언한 자리에 손님이
  말한 숫자를 넣었다. 이어서 `disputed_amount: 49.99` 로 제출([127]).
- 같은 [125] 에서 `discovery_date: 11/10/2025 (**assuming you discovered it the next day**)` — [122] 의
  *"I just noticed it today"* 와 **정면 모순**이고, [16] 에서 `get_current_time → "2025-11-14 03:40:00 EST"`
  를 이미 받아 두었다(gold 11/14/2025).
- 기계적 확인(substring): `49.99` 는 **role=user 메시지 [122] 에만** 실재하고 role=tool 출력에는 0건 ·
  `14.99` 는 **role=tool [102],[124] 에만** 실재하고 user 발화에는 0건. **두 값의 출처가 완전히 갈린다.**

### (g) 종료

[130] user: *"No, that covers everything. The cards are both fine—these aren't card security issues."* → user_stop.
`request_human_agent_transfer` 없음. 즉 trial-0 은 **끝까지 갔고 4건을 다 걸었는데 4건 다 틀렸다.**

---

## 3. trial-1 분기점 (어느 턴에서 갈렸나)

두 trial 은 [0]~[24] 가 거의 동형이다(같은 unlock 순서·같은 verify·같은 `account_id="f7d3a82c91"` 오투입).
갈린 자리는 **모델 자신의 KB 질의 선택**이다.

| | trial-0 | trial-1 |
|---|---|---|
| 1차 KB 질의 | [51] `KB_search_bm25{"file debit card dispute"}` → **doc 031 도착**([52]) | [45] `KB_search_bm25{"retrieve debit card last 4 digits"}` → doc 028/009 (필링 정본 아님) |
| 2차 KB 질의 | [83] `KB_search_dense{"…6281 required parameters"}` → **doc 031 재도착**([84]) | [68] `KB_search_bm25{"get transactions by account id"}` → 거래조회 문서 |
| doc 031 문맥 도달 | **[52], [84] 2회** | **0회**(전 궤적에서 `doc_bank_accounts_bank_accounts_(general)_031` substring 0건) |
| 결과 | 8회 좌초 후 [85] 에서 5개 인자 일괄 교정 → 필링 성사 | [78]~[86] **동일 인자 5연속** → 포기 |

- 분기 확정: **trial-1 [45] 턴**. 여기서 필링 요건이 아니라 카드 끝4자리를 물은 것이 doc 031 미도달로,
  그것이 `pin_compromised` bool 오형과 무명 거부 루프로 이어졌다.
- [88] assistant: *"It seems there is a persistent issue with the parameters …"* → [89] user 가
  `request_human_agent_transfer` → [93]/[95] `###TRANSFER###` → user_stop.
- **주의**: 시그니처는 trial-1 도 [5] 에 실려 있었다(`pin_compromised: string … 'no'`). doc 031 부재가
  결정적이었던 것이 아니라, **문맥에 있는 시그니처를 5회 연속 안 봤다**는 것이 결손이다.

---

## 4. 레버 발화표 (로그 축자 · 이 sim 줄만)

| 레버 | trial-0 | trial-1 | 판정 |
|---|---|---|---|
| `T2_SG_DOCS` | 0 | 0 | **미발화**(ON 인데 기회 0 — SG 격리 경로 미진입) |
| `T2_PIN_READ` | 0 | 0 | **미발화** |
| `T2_DEMANDED_STEP` | 0 | 0 | **미발화** |
| `T2_CLAIMPROV` | 51 | 15 | **발화·정상**. `kind='dispute_file' tool='call_discoverable_agent_tool' 원장에 있다` · `ledger narrowed: 10 failed call(s) excluded` · `window hit(resign) claims=8 unbacked=0`. **t7328 의 DUP 5 재발 0** |
| `T2_FOLLOWUP` | 0 | 0 | **미발화** |
| `T2_SEARCH_AGENT` | 12 | 15 | **발화·살아있음**. `group=checking_accounts · 문서 113 … 일반 자리 배달 1회째` / `2회째` / `3회째`. 침묵 라인은 전부 `요청 축 … 모두 처리됨 — 침묵`(설계된 중복 억제) — [[67]] 死배선 아님 |
| `FAB_STRIP`(`T2_FAB_STRIP=1`) | 0 | 0 | **미발화**. 트리거가 `EXHAUSTION→FAIL` 국면 한정(`t2_gate_patch.py:9572-9576`)이라 정상 write 경로엔 안 걸린다 |
| `T2_ARG_PRODUCERS` | **0** | **0** | **미발화 = 2026-08-21 수리 성립.** t7335 에서 KB 본문에 2회 오발화해 credit 축을 주입했던 F8 이 이번엔 0회 — 에러-형상 게이트가 잡았다 |
| READ-FIRST(P3 comparator) | 0 | 0 | 미발화(comparator 국면 미도달) |
| `T2_REQUIRE_DOC_DELIVER` | 0 | 0 | 미발화 |
| `T2_SEARCH_REARM` | **2** | 0 | **발화했으나 너무 늦다**: `group=checking_accounts 신규 대상 green_fee-free_account (기배달 blue_account) — 소진 해제·문서 델타` → `델타 배달 7413자 (문서 7) turn=65`. **Green 계좌 필링은 assistant turn 58·62 에 이미 끝났다**(총 63턴) |
| `T2_DECIDE_BEFORE_WRITE` | 1 | 1 | **발화했으나 같은 이유로 무효**: `write 1턴 유예 tool=file_debit_card_transaction_dispute (재료 7413자)` — turn=65, 마지막 필링(turn 62) 이후 |
| `T2_WRITE_ARG_GROUND` | 1 | 2 | **오적용 범위**: `deny tool=log_verification`(t0) · `deny tool=give_discoverable_user_tool` ×2(t1). **분쟁 필링에는 단 1회도 안 걸렸다** → §5-A |
| `T2_REF_VERIFY`(=1) | 0 | 0 | **미발화** — 선언이 credit 전용 → §5-A |
| `T2_ARG_DOC_SUB`(이번 런 신규 ON) | 2 | 1 | **개입 없음**: `spend_category=None 없음 — 넘긴 문서 12편에 근거가 없다` — 리워드-요율 인자 전용. 085 의 결정 인자에 무관 |
| `T2_VALUE_ACQUIRE` | 6 | 6 | 발화·부작용: `consumers card_last_4_digits=1` → `give-surfacing → nudge (regen)` ×3. [43]~[53] 에서 debit 문의에 credit 도구 `get_card_last_4_digits` 를 밀어 5턴 소모(env: *"Error: Missing required parameter: credit_card_account_id"*). DB 무영향 |
| `T2_MATERIAL_GATE` | 10 | 20 | `stop=resolve_cap(정체 3회)` 가 t1 의 사망 루프 구간(turn 58·62·64·66·68)에서 5회. **정체를 탐지하고도 재료를 끊는 쪽으로 동작** |
| `T2_EPLAN_LISTED_IS_READ` | 31 | 9 | 발화·무효: `배달된 5건을 검토됨으로 보정(btxn_a1b2c3d4e501,btxn_b2c3d4e5f602,btxn_c3d4e5f6g703,…)` — **두 CityFit 행을 둘 다 "검토됨"으로 부기**할 뿐 어느 쪽을 걸지 말하지 않는다 |
| `T2_GROUND` | 2 | 2 | 발화: `substituted arg=agent_tool_name val=get_checking_account_transactions_by_user -> JORDAN WILLIAMS` — 날조 도구명 치환. t1 [66] 에서 그 결과가 `Error: Unknown agent tool 'JORDAN WILLIAMS'` 로 나갔다(치환값이 도구명 자리에 부적합) |

**직전 런 이후 수리가 이 궤적에 개입했는가**: ⓐ `T2_ARG_PRODUCERS` 에러-형상 게이트 — **개입했고
성공**(오발화 0). ⓑ `arg_source_reads`(account_id/card_id → 3847/7823) — **개입했고 성공**(계좌 해결 통과).
ⓒ `T2_SEARCH_REARM`(t7336 016 처방) — **개입했으나 늦어서 못 샀다**. ⓓ 이번 런 신규 `T2_ARG_DOC_SUB` —
**해당 없음**.

---

## 5. 원인 확정 (4주체 귀속)

### A. our_layer — CONFIRMED ①: **debit 분쟁 필링이 인자-거버넌스 선언에서 통째로 빠져 있다**

`a2/banking_knowledge.specific.json` 를 계열별로 스캔한 결과:

| A2 계열 | `file_credit_card_transaction_dispute` | `file_debit_card_transaction_dispute` |
|---|---|---|
| `write_arg_grounding` | **있음** (`applies_when.prefix`, `grounded_args:['card_last_4_digits','transaction_id']`) | **없음** |
| `ref_verify` | **있음** (`record_field: merchant_name`) | **없음** |
| `ref_iso` | **있음** | **없음** |
| `write_evidence_specs` | **있음** | **없음** |
| `have_value_reask` / `value_acquisition` | **있음** | **없음** |
| `tool_signatures` | — (`give_discoverable_user_tool` 만) | **없음** |

- **코드 경로**: `scripts/distill/tau2/t2_gate_patch.py:1406` `_write_arg_ground_deny` 의
  `if name != sp.get("applies_to"): continue` 와 이어지는 `applies_when.prefix` 필터.
  선언이 없으므로 `file_debit_card_transaction_dispute_6281` 호출은 **술어에 도달조차 하지 않는다.**
- **선언 키**: `a2/banking_knowledge.specific.json` → `write_arg_grounding[0].applies_when.prefix =
  "file_credit_card_transaction_dispute"` (동일 사본이 `a2/banking_knowledge.gate.json`,
  `a2/split/banking_knowledge.core.json`).
- **런에서의 확증**: `T2_WRITE_ARG_GROUND=1`(`go_stack.sh:106`, `:261`)·`T2_REF_VERIFY=1`(`go_stack.sh:107`)
  둘 다 ON 인데 로그의 발화 대상은 `log_verification`·`give_discoverable_user_tool` 뿐이다.
- **얼마나 샀을 값인가(기계 검산)**: 제출 인자별 substring 근거 실재를 필링 시점 기준으로 재현하면

  | 필링 | `discovery_date` | `disputed_amount` |
  |---|---|---|
  | msg[87] | `11/06/2025` → tool[68] 존재(다른 거래 날짜에 **우연 통과**) | `100` → tool·user 존재 |
  | msg[95] | `11/07/2025` → **tool 0 · user 0 = 무근거** | `89.99` → 존재 |
  | msg[119] | `11/09/2025` → tool[102] 우연 통과 | `200` → 존재 |
  | msg[127] | `11/10/2025` → **tool 0 · user 0 = 무근거** | `49.99` → **user[122] 에만**, tool 0 |

  ⇒ `grounded_args` 에 `discovery_date` 만 넣어도 **msg[95]·msg[127] 2건이 기본 코퍼스로 차단**되고,
  `corpus_roles:{"disputed_amount":["tool"]}`(기전 실재: `t2_gate_patch.py` 의 `corpus_roles` 분기,
  선례 = `give_discoverable_user_tool.card_name`)을 더하면 **msg[127] 의 49.99 가 추가로 차단**된다.
- **⚠자기 반증(반드시 같이 기록)**: gold 의 `discovery_date=11/14/2025` 는 궤적 어디에도 축자로 없다
  (시계 출력은 `"2025-11-14 03:40:00 EST"` 로 **형식이 다르다**). 따라서 위 선언을 그대로 넣으면
  **정답도 false-block** 한다 — 054 `time_verified` 가 `arg_corpus_marker` 를 만든 것과 동형 문제다.
  형식 정규화 또는 marker 없이는 출시 불가. [[70]] 전체 reward 짝 A/B + 태스크별 부호표 + [[57]]
  부정통제 전에는 처방일 뿐이다.
- **⚠범위 정직성**: 이 선언은 ①카테고리 오선택(`incorrect_amount` vs `atm_cash_discrepancy`)
  ②CityFit `…703` vs `…602` ③걸지 말았어야 할 `…905` **셋 다 못 산다**. 세 건 모두 값이 원장에
  실재하므로 substring 술어를 통과한다. `ref_verify` 도 못 산다(merchant 이름이 손님 발화에 실재).

### B. our_layer — CONFIRMED ②: **`T2_MATERIAL_BYPASS` 가 구현·회귀검정까지 있는데 스택에 미등재**

- **코드 경로**: `t2_gate_patch.py:8069` `if (_mgate_kind == "resolve_cap" and
  os.environ.get("T2_MATERIAL_BYPASS") == "1" …)`. 회귀 `test_material_bypass.py` 존재.
- **미등재 확증**: `go_stack.sh` 에 `T2_MATERIAL_BYPASS` export 0건(레포 전수 grep 결과 검정 파일 외
  참조 없음) ⇒ 이 런에서 **OFF**.
- **이 궤적과의 접점**: trial-1 의 사망 루프 구간에서 `[T2_MATERIAL_GATE] stop=resolve_cap(정체 3회)` 가
  turn 58·62·64·66·68 로 5회 찍혔다. 그 자리가 바로 이 레버가 열도록 설계된 자리다
  (주석 축자: *"`_resolve_cap_ok` 는 제자리걸음을 억제하려고 만든 상한인데, 검색 에이전트의 **재료
  배달**이 같은 관문 안에 갇혀 있어 함께 멎는다"*).
- **⚠불확실**: trial-1 시점 `_t2_searchagent_fired`=2(<3)라 게이트 조건은 만족하나, 배달됐을 재료가
  doc 031 을 포함했을지는 이 로그로 확정 못 한다. trial-0 은 이미 3회 소진이라 **켜져 있었어도
  안 열린다**. 따라서 "켰으면 이겼다"는 주장은 **하지 않는다** — 등재 누락 자체가 [[60]] 관점의 결함이다.

### C. our_layer — 보조(부호 미확정, 코드 지목 가능)

- **`T2_SEARCH_REARM` 의 재무장 시점이 write 를 놓친다.** 술어부(`t2_gate_patch.py:3084` 이하)는
  ⑵ 재수요를 *"배달 시점 이후 user/assistant 발화에 계열 표시명이 축자 등장"* 으로 잡는데,
  손님이 *"Green Fee-Free Account"* 를 말한 것은 [98](turn≈50)이고 실제 재무장은 **turn 65** 에
  일어났다. 사이에 Green 계좌 필링 2건(turn 58·62)이 지나갔다. 원인은 술어가 아니라 **진입 시점** —
  이 경로는 모델이 KB 검색을 하는 턴에만 열리는데 [83] 이후 모델이 검색을 안 했다. 즉 우리 문서
  채널이 **모델 주도성에 종속**돼 있다. 코드 지목은 되지만 "이걸 고치면 산다"는 근거는 없다
  (§2-b 대로 doc 031 은 이미 [52]·[84] 로 두 번 도착해 있었다) ⇒ **UNPROVEN**.
- **`arg_producers = {"card_last_4_digits": {"user_tool": "get_card_last_4_digits"}}`**
  (`a2/banking_knowledge.specific.json`)가 조건 없이 선언돼 있어 debit 문의에서도 credit 도구를
  민다([43]~[53] 5턴 소모). DB 무영향이므로 이번 실패의 원인은 아니다. [[70]] 의 "조건은 도메인 일반
  닫힌 술어" 대상 후보.

### D. model — **주 원인**

문맥에 축자로 실재한 값·규칙을 write 인자로 옮기는 단계에서 6종이 실패했다.

1. **레코드 값을 손님 주장으로 덮어씀**: [124] `amount: -14.99` → [125] *"Amount: -$49.99"* →
   `disputed_amount: 49.99`. **직전 메시지**다. 이 태스크의 명시 시험(*"Agent should NOT blindly trust
   user claims"*)이 정확히 이 지점이다.
2. **손님 주장 대 레코드 대조 부재**: [98] *"withdraw $500 … only got $300"* 대 [102] `-200.0` 의
   모순을 [103] 이 언급조차 안 하고 걸었다(과행동 1건·가지급 $200 발생).
3. **묻지 않고 만듦**: `discovery_date` 4건 전부 *"(assuming you discovered it the next day)"*.
   doc 031 은 *"Date customer first noticed the issue"*, 시나리오는 물으면 답하는 구조였다.
   [122] 는 *"I just noticed it **today**"* 라고 **같은 메시지에서 말했는데** 11/10 을 넣었다.
4. **문맥의 정책문 미적용**: *"Dispute the earliest (first) transaction when multiple duplicates exist"*
   가 [84] 에 있는데 [93] 은 목록 첫 줄(`…703`)을 집었다.
5. **enum 매핑 오류**: `atm_cash_discrepancy` 정의문이 [84] 에 있는데 `incorrect_amount` ·
   `recurring_payment` 정의문이 있는데 `signature_purchase`. env 가 이름 지어 거부한 뒤에만 고쳤다
   ([94]→[95], [118]→[119], [126]→[127]) — **거부가 이름을 대면 고치고, 문서가 말하면 안 고친다.**
6. **동일 호출 반복**: trial-0 msg[75]→[81] 4회 · trial-1 msg[78]→[86] 5회, 인자 델타 0.
   시그니처는 [5] 에 있었다.

**이 서명은 sha 를 건너 재현된다** — 기준선 `bank_t7328_halfB_20260819r2` trial-1 의 필링 4건이
`btxn_c3d4e5f6g703`(같은 오선택) · `btxn_e5f6g7h8i905`(같은 과행동) · `btxn_f6g7h8i9j006`
`disputed_amount 49.99`(gold 14.99·**같은 덮어쓰기**) · `transaction_type signature_purchase`(같은 오매핑)
으로 **t7346 trial-0 과 거의 자리까지 일치**한다. 2런·상이 sha·상이 seed 에서 3회 중 3회 재현.

### E. env — 보조 (채점 무관하나 루프의 직접 원인)

- *"Error: Missing required parameters."* 는 **어느 인자인지 말하지 않는다.** 실제 원인은
  `pin_compromised` 를 bool `false` 로 보낸 것(falsy → missing 판정)이고 `card_action: null` 도 같다.
  같은 env 가 `customer_max_liability_amount`·`transaction_type`·`dispute_category` 는 **이름을 대고**
  거부했고, 모델은 이름을 댄 거부는 **1턴 만에 전부 고쳤다**. [[64]] 의 실물 대조군이 한 궤적 안에 있다.
  trial-1 은 이 무명 거부만 6회 받고 죽었다.
- 중복 거래 뷰가 시각 필드 없이 **역순 나열**(…703 먼저)이라 doc 031 의 "earliest" 판정을 가린다([[25]]:
  env 는 외부 주장·검증 의무는 agent 측).

### F. user_sim — 정상

시나리오 대로 반응했다. [56]·[90]·[98]·[122] 에서 요구 사항을 전량 제공했고, 교정 분기
(*"If agent says the transaction shows a different amount …"*)는 **에이전트가 대조를 안 해서** 열리지
않았다. trial-1 의 `request_human_agent_transfer` 는 8회 좌초 후의 스펙 내 행동이다. [[21]] 대로
면책 사유로 쓰지 않는다.

### reward 관점 ([[69]])

- trial-0: DB 해시가 어긋난 이유는 **①gold 3행 부재 ②gold 밖 4행 존재** 양쪽이다. 우리 게이트의
  차단이 gold 행을 막은 사례는 **0건**(BLOCKED 18건 전부 env deny, `deny='env'`). 손실은 전부
  **인자 내용**과 **과행동**.
- trial-1: 손실 전부 도달 실패(BLOCKED 6·전부 env). 우리 층 차단 기여 0.

---

## 6. 선행 판정과의 대조

| 항목 | t7335 halfB2 (2026-08-21 보고서) | t7328 halfB2 (기준선) | **t7346 (이번)** |
|---|---|---|---|
| 국면 도달 | filing **0회 시도** — 계좌 해결서 좌초 | t1 은 필링 4건 도달 | **양 trial 필링 국면 도달** |
| `get_all_user_accounts_by_user_id_3847` | **0회 호출**(문서 2회 회수하고도) | — | **호출 성공**(t0 [41], t1 [37]) |
| `T2_ARG_PRODUCERS` F8 | **오발화 ×2**(KB 본문에 발화·credit 축 주입) | — | **0회 — 수리 성립** |
| `T2_CLAIMPROV` DUP | 0(t7328 대비 수리) | **DUP 5** | **0 — 유지** |
| 변이 | MISSING 3 · WRONGARG 0 | MISSING 3 · WRONGARG 4(t1) | MISSING 3 · **WRONGARG 4**(t0) |
| 원인 | 모델의 **ID-해결 read 생략** + 우리 F8 오발화 | 인자 내용 오류 | **모델의 인자 전사/검증 실패** + 우리 debit 선언 부재 |

**결론: 원인이 달라졌다.** t7335 의 진단(도달 실패·F8 오발화·해소-read 무지목)은 **모두 해소됐다**.
남은 실패는 **한 단계 아래**로 이동했다 — 도구·계좌·레코드를 다 확보한 뒤 **인자에 무엇을 쓸지**에서
진다. 동시에 t7335 보고서의 무수리 판정 (a)`account_id→3847 해소-read 지목` 은 `arg_source_reads`
저작으로 해결됐음이 이 궤적으로 실증됐다.

한편 t7328 t1 과 t7346 t0 이 **거의 동일한 4건 오답**을 냈다는 사실은, 이 잔여가 **레버 조합에 둔감한
모델 결손**임을 시사한다(2런·상이 sha).

---

## 7. 처방 후보 (⛔실행·코드 수정 없음)

**P1 (CONFIRMED 우리-층 · 최우선)** — `write_arg_grounding` 에 debit 쌍둥이 항목 저작.
`a2/banking_knowledge.specific.json` + `.gate.json` + `split/*.core.json` 3곳 바이트 동일 동기([[24]]).
초안: `applies_when:{arg:"agent_tool_name", prefix:"file_debit_card_transaction_dispute"}`,
`grounded_args:["transaction_id","disputed_amount","transaction_date","discovery_date"]`,
`corpus_roles:{"disputed_amount":["tool"],"transaction_date":["tool"],"transaction_id":["tool"]}`.
출처는 도구 독스트링 축자([5])와 doc 031 축자 — **gold 무참조**([[23]]).
⚠**선행 필수**: `discovery_date` 는 gold 값(11/14/2025)이 시계 출력과 형식이 달라 무근거 판정된다.
`arg_corpus_marker`(054 선례) 또는 날짜 정규화 없이 넣으면 **정답을 막는다**. 격리 프로브([[62]] ①)로
false-block 율부터 재고, [[70]] 전체 reward 짝 A/B + 태스크별 부호표, [[57]] 부정통제 없이 출시 금지.

**P2 (CONFIRMED 우리-층)** — `T2_MATERIAL_BYPASS` 등재 여부 판정. 구현·검정 완비인데 `go_stack.sh`
미등재라 매 런 OFF. [[60]]("끄지 마라") 대 [[73]]("새 레버는 격리 프로브 후만")이 충돌하는 자리이므로
**등재 전 격리 프로브 + A/B** 를 먼저 건다. 이 태스크만으로 이득을 주장할 근거는 없다(§5-B ⚠).

**P3 (조사 후보 · 우리 층 아닐 수 있음)** — env 의 무명 거부 *"Missing required parameters."* 에
**우리 층이 이름을 붙여 줄 수 있는가**. 한 궤적 안에 대조군이 있다: 이름 있는 거부는 1턴에 교정,
무명 거부는 4~5회 동일 재제출. 닫힌 술어로 가능한 형태 = *"직전 동일 인자 재제출 N회"* 탐지 +
**이미 문맥에 있는 도구 시그니처의 해당 필드 재표면화**(값 선택 0·[[62]] ③). ⚠[[64]] 는 조언 채널
규칙이고 도구-결과 deny 는 fail-closed 유지 — 본문을 지우지 않고 **덧붙이는** 형태여야 한다.

**무수리 판정 (사유 박제)**

- **(a) "손님 주장 ↔ 레코드 대조" 게이트**: 어느 필드를 대조할지·불일치를 어떻게 처리할지는
  **열린 해석**이고 국면 자기평가 부류([[66]]). `disputed_amount` 한 축만은 P1 의 `corpus_roles` 로
  닫히므로 그쪽으로 흡수한다. 케이스 열거 금지.
- **(b) enum 매핑 결정기**(`atm_cash_discrepancy` vs `incorrect_amount`, `recurring_payment` vs
  `signature_purchase`): 정의문이 문맥에 있고 정답이 하나로 정해지므로 결정기를 지으면 **gold 프로그램
  재작성**([[62]] ⛔). `write_arg_enum` 의 axis 방식(LLM 이 답하고 엔진은 맞대기만)이 유일한 합법 형태이나
  **선행 격리 측정 없이 금지**.
- **(c) "earliest duplicate" 선택기**: 정책문이 닫혀 보이지만 뷰에 순서 근거가 없어 엔진이 고르려면
  레코드 id 순번을 해석해야 한다 — 도메인 리터럴 패턴매칭([[59]] 위반). 무수리.
- **(d) `T2_SEARCH_REARM` 시점**: §5-C 대로 UNPROVEN. doc 031 은 이미 두 번 도착했으므로 전달을
  더 사도 이 태스크는 안 산다.

---

*작성: t7346 halfB task_085 per-step 포렌식 · 2026-08-23 · 근거 전부
`bank_t7346_halfB_20260822.results.json.gz` 메시지 축자와 `bank_t7346_halfB_20260822.log.gz` 의
`[sim=task_085#s626729]`/`[sim=task_085#s373753]` 레버 라인 · 변이는 `t2_forensic.mutation_diff` 정본 ·
gold 는 진단용 대조로만 사용([[23]]).*
