# t7346 · task_094 (GOLD SAVINGS APY INVESTIGATION) 궤적 per-step 포렌식

- 대상: `sim_results/bank_t7346_halfA_20260822.results.json.gz` / `.log.gz`
  (⚠지시서의 `bank_t7346_halfA__20260822.*`(밑줄 2개)는 실재하지 않는다 — 실파일은 밑줄 1개. 094 는 **halfA** 에 있다.)
- 결과: **trial 0 = 0.0 · trial 1 = 0.0** (0/2 · 양 trial 실패) · 둘 다 `termination_reason=user_stop`
- 로그 sim 태그 매핑(`sim['seed']` 직독): `s626729` = **trial 0** · `s373753` = **trial 1**
- 대조: 선행 `t7336_tasks/T7336_TASK_094.md` (같은 seed 쌍) · 기준선 `bank_t7328_halfA_20260819r`(sha 상이·비엄밀)

> **한 줄 결론**: 선행(t7336)의 P1 처방 — `get_correct_savings_apy.requires_reads` — 은 **이번에 실제로 발화했고**
> 모델을 계좌 조회로 몰아넣는 데 성공했다. 그런데 그렇게 조회된 **전 계좌 레코드(1629자·`Green Account` 와 `EcoCard` 를
> 모두 담은 바로 그 출력)가 서브에게 전달되지 못했다.** 같은 날 들어간 R3 레버(`ref_from_outputs`)가
> `producer_contains=["accounts"]` 로 **바깥 래퍼 이름**을 훑는데, 그 레코드의 생산자 이름은
> `call_discoverable_agent_tool` 이라 걸리지 않고, 대신 `get_credit_card_accounts_by_user`(1023자·카드만)가
> `customer_products` 자리에 실렸다(로그 축자 `1024자` = 1023 + 접합 공백 1 · 산술 일치).
> ⇒ REFERENCE 에서 **Green checking 이 사라져** checking +0.75 가 통째로 누락 → `-> 6.1`(gold 6.85) → `amount 8`(gold 140).
> trial 1 은 needle 이 **아무것도 못 맞혀** REFRAW 자체가 침묵했고, 에이전트의 손요약 `"Personal Checking Account"` 가
> 그대로 REFERENCE 가 되어 카드까지 통째로 빠졌다 → `-> 5.5` → `amount 32`.
> **결손을 만든 것은 모델의 산술이 아니라 우리 층의 생산자-이름 매칭이다.**

---

## 0. 채점축 (C583ⓖ · `sim['reward_info']` 직독)

```
reward_basis      = ["DB"]
reward_breakdown  = {"DB": 0.0}
db_check          = {"db_match": false, "db_reward": 0.0}
env_assertions=[] · nl_assertions=null · communicate_checks=null · info.action=null
```

**DB-해시 축**이다. `action_checks` 는 진단 보조로만 읽는다([[69]]).

gold 변이 3건(`t2_forensic.gold_mutations` · unlock/give 는 DB 무변이라 제외):

| aid | 도구 | 인자 |
|---|---|---|
| 094_0 | `log_verification` | Wei-Ting Lin / wl94k7m3p8 / … / `2025-11-14 03:40:00 EST` |
| 094_6 | `apply_savings_account_credit_6831` | `{sav_wl94k7m3p8_gold, **140**, interest_correction}` |
| 094_8 | `submit_interest_discrepancy_report_7294` | `{sav_wl94k7m3p8_gold, wl94k7m3p8, expected_apy **6.85**, actual_apy **5.1**, amount_difference **140**}` |

정답 산식(태스크 `notes` 축자 · 진단용):
*"the actual correct APY is 6.85% (base 5.5% + Green checking boost 0.75% + EcoCard 0.6%), but only 5.1% was applied … calculate the real discrepancy ($140.00)"*
⇒ 6.85 = 5.5(Gold base) + 0.75(Green checking→Gold) + 0.6(카드 max1 = EcoCard) · 5.1 = 408×12/96000 · 차액 = 96000×1.75%/12 = **140.00**.

고객 오정보(시나리오 축자): base 5.0 · checking 1.0 ⇒ 6.0. 그리고 *"If agent corrects your rate assumptions, follow the agent's lead as long as they have proper citation. You are not 100% sure of your rates."*

---

## 1. 변이표 (정본 `t2_forensic.mutation_diff` · 손 비교기 미사용 · C583ⓐ)

### trial 0 (`s626729`) — `missing 2 · wrongarg 2 · extra 0 · dup 0 · blocked 0 · matched 1`

| 종류 | 내용 |
|---|---|
| MATCHED | `log_verification{Wei-Ting Lin, wl94k7m3p8, …, 2025-11-14 03:40:00 EST}` (msg 20) |
| MISSING | `apply_savings_account_credit_6831{sav_wl94k7m3p8_gold, **140**, interest_correction}` (aid 094_6) |
| MISSING | `submit_interest_discrepancy_report_7294{…, **6.85 / 5.1 / 140**}` (aid 094_8) |
| WRONGARG | `submit_interest_discrepancy_report_7294{sav_wl94k7m3p8_gold, wl94k7m3p8, **6.1 / 6.0 / 8**}` (msg 32 · ok=True) |
| WRONGARG | `apply_savings_account_credit_6831{sav_wl94k7m3p8_gold, **8**, interest_correction}` (msg 58 · ok=True) |

### trial 1 (`s373753`) — `missing 2 · wrongarg 2 · extra 0 · dup 0 · blocked 0 · matched 1`

| 종류 | 내용 |
|---|---|
| MATCHED | `log_verification{…}` (msg 12) |
| MISSING | `apply_savings_account_credit_6831{…, **140**, …}` (aid 094_6) |
| MISSING | `submit_interest_discrepancy_report_7294{…, **6.85 / 5.1 / 140**}` (aid 094_8) |
| WRONGARG | `submit_interest_discrepancy_report_7294{sav_wl94k7m3p8_gold, wl94k7m3p8, **5.5 / 5.1 / 32**}` (msg 48 · ok=True) |
| WRONGARG | `apply_savings_account_credit_6831{sav_wl94k7m3p8_gold, **32**, interest_correction}` (msg 70 · ok=True) |

**WRONGARG 필드별 대조** (보낸 인자 ↔ gold)

`submit_interest_discrepancy_report_7294`

| 필드 | trial 0 | trial 1 | gold | 판정 |
|---|---|---|---|---|
| `account_id` | `sav_wl94k7m3p8_gold` | `sav_wl94k7m3p8_gold` | 동일 | ✅ ✅ |
| `user_id` | `wl94k7m3p8` | `wl94k7m3p8` | 동일 | ✅ ✅ |
| `expected_apy` | **6.1** | **5.5** | **6.85** | ❌ −0.75 (checking 누락) / ❌ −1.35 (checking+card 누락) |
| `actual_apy` | **6.0** | **5.1** | **5.1** | ❌ (고객 주장 복제) / ✅ |
| `amount_difference` | **8** | **32** | **140** | ❌ / ❌ (상류 오차의 결정론적 귀결) |

`apply_savings_account_credit_6831`

| 필드 | trial 0 | trial 1 | gold | 판정 |
|---|---|---|---|---|
| `account_id` · `credit_type` | 동일 | 동일 | 동일 | ✅ |
| `amount` | **8** | **32** | **140** | ❌ |

⇒ **식별자·범주 필드는 6/6 일치**. 어긋난 것은 **수치 3필드**뿐이고, 그 3필드는 전부 상류 값
`get_correct_savings_apy -> {6.1, 5.5}` 에서 파생된다(`get_interest_correction` 이 그 값을 그대로 받아 곱한다).

**결손 분해**

| 성분 | gold | trial 0 반영 | trial 1 반영 | 근거 문서 배달 여부 |
|---|---|---|---|---|
| base (Gold Account) | 5.5 | **5.5** ✅ | **5.5** ✅ | `by_class['gold_account']` 8편 |
| checking (Green→Gold) | +0.75 | **0** ❌ | **0** ❌ | `doc_…_046` = `docs.always` — **배달됨**(축자 존재) |
| card (EcoCard · max1) | +0.6 | **0.6** ✅ | **0** ❌ | trial 0 만 `by_class['credit_cards_ecocard']` 배달 |
| relationship/tier | 0 | 0 ✅ | 0 ✅ | — |
| **합** | **6.85** | **6.1** | **5.5** | |

(로그 축자 `[T2_SG_ISOLATE] operand-size … components: sub=2 rows`(t0) / `sub=1 rows`(t1) 와
`[T2_SCAFFOLD_GET] get_correct_savings_apy -> 6.1 / -> 5.5` 의 산술이 이 분해와 일치한다. 서브의 원시 JSON 은
로그에 없으므로 성분의 개별 값은 **산술 정합에 의한 추정**으로 표기한다.)

---

## 2. 결정 지점 추적 (step-by-step · 축자 인용)

### 2.0 공통 [0]~[1] — 재료는 첫 턴에 전부 실재한다

```
[1] user: "I have a Gold savings account with about $96,000 and I only received $408 in interest this month.
     The base rate is 5.0% and my checking account gives me a 1% boost, so I should be at 6.0% APY.
     … And if there are any other boosts that I'm missing, I want that addressed now too!"
```

`$408` · `$96,000` 이 [1] 에 있다 ⇒ `actual_apy = 408×12/96000 = 5.1` 은 **닫힌 산술**이다.

### 2.1 [4]~[21] trial 0 / [4]~[13] trial 1 — 신원 확인 (양 trial 정상)

우리 층 게이트가 [[64]] 형태로 다음 단계를 이름으로 지목하고, 모델이 그대로 따른다:

```
[11] tool: NOT_VERIFIED — the account record has not been fetched yet, so nothing can be matched.
     Look the customer up with get_user_information_by_name/by_email/by_id, then call this tool again.
[17] tool: VERIFIED — 2 identity values match the account record (date_of_birth, email); you may now call
     log_verification. Its time_verified argument must be the CURRENT time: FIRST call get_current_time …
```

→ `get_current_time` → `log_verification` = **gold 094_0 완전 일치(MATCHED)**. 이 구간 결손 0.

### 2.2 ★공통 — READ-FIRST 게이트(선행 P1 처방)가 **발화했고 먹혔다**

trial 0 [22] 는 계좌 레코드 0회 read 상태에서 APY 계산을 시도한다(같은 턴에 카드 조회 1개를 병렬로 낸다):

```
[22] CALL get_credit_card_accounts_by_user {"user_id":"wl94k7m3p8"}
[22] CALL get_correct_savings_apy {
       "components":"[{\"kind\":\"base\",\"value\":5.0,\"source\":\"Base APY for Gold Savings Account is 5.0%\"},
                      {\"kind\":\"checking\",\"value\":1.0,\"source\":\"Checking account provides a 1.0% boost to savings APY\"}]",
       "savings_account_type":"Gold Savings Account",
       "customer_products":"Has a checking account that provides a 1.0% boost to savings APY",
       "current_balance":"96000" }
[23] tool: Error: [READ-FIRST] this calculation depends on records you have not read yet in this conversation.
     Missing required reads (BASE names): get_all_user_accounts_by_user_id. … Their exact callable forms are:
     unlock_discoverable_agent_tool(agent_tool_name="get_all_user_accounts_by_user_id_3847") then
     call_discoverable_agent_tool with that name.
```

로그: `[T2_SG_REQREADS] get_correct_savings_apy denied: missing reads ['get_all_user_accounts_by_user_id']` (양 trial 1회씩).
모델은 **정확히 지시대로** 복구한다:

```
[25] ASST: "It looks like there was an issue with fetching the account details. Let me try again to unlock and
     fetch the necessary information."
[25] CALL unlock_discoverable_agent_tool {"agent_tool_name":"get_all_user_accounts_by_user_id_3847"}
[25] CALL call_discoverable_agent_tool {"agent_tool_name":"get_all_user_accounts_by_user_id_3847","arguments":"{\"user_id\":\"wl94k7m3p8\"}"}
[27] tool(1629자): "… Bank Accounts: … 1. Record ID: sav_wl94k7m3p8_gold … level: Gold Account …
     2. Record ID: chk_wl94k7m3p8_green … class: checking … level: **Green Account** … current_holdings: 4215.80
     Credit Card Accounts: … Platinum Rewards Card … Gold Rewards Card … **EcoCard** …"
```

**선행 P1 처방은 여기서 정확히 목표를 달성했다** — gold 094_1/094_2(`unlock` + `call get_all_user_accounts_by_user_id_3847`)가 실행됐고 `action_match=true` 다.
그리고 이 시점 **문맥에는 `Green Account` 와 `EcoCard` 가 축자로 실재한다.**

### 2.3 ★★[28] trial 0 — **실패 결정 지점 1 · 우리 층이 그 레코드를 버렸다**

```
[28] ASST: "Now that we have your account details, let's calculate the correct APY … your Gold Savings Account
     has a base APY of 5.0%, and your checking account provides a 1.0% boost … Therefore, the total APY should be 6.0%."
[28] CALL get_correct_savings_apy { (인자는 [22] 와 **바이트 동일**) }
```

같은 턴의 우리 층 로그(축자 · `s626729`):

```
[T2_SG_TRACE]  get_correct_savings_apy: iso=True mode=fetch_formalize ctx=['components','current_balance','customer_products','savings_account_type']
[T2_SG_REFRAW] get_correct_savings_apy.customer_products ← 도구 출력 원문 1편(**1024자**·에이전트 전사 대체)
[T2_SG_DOCS]   get_correct_savings_apy: 클래스 ['gold_account','gold_saver_account','platinum_account','credit_cards_ecocard'] · 문서 17편 · 16557자 전달(검색 0)
[T2_SG_ISOLATE] operand-size get_correct_savings_apy.components: sub=2 rows · source=0 rows
[T2_SCAFFOLD_GET] get_correct_savings_apy -> **6.1**
```

**1024자가 어느 출력인가 — 산술로 확정된다.** 이 시점 `__tool_outputs_raw` 후보는 둘뿐이다:

| 생산자(=`tool_call.name`) | 길이 | Green Account 포함 | EcoCard 포함 | needle `"accounts"` 일치 |
|---|---|---|---|---|
| `get_credit_card_accounts_by_user` | **1023** | ❌ | ✅ | ✅ (이름에 `accounts`) |
| `call_discoverable_agent_tool` (= `get_all_user_accounts_by_user_id_3847` 실행분) | **1629** | ✅ | ✅ | ❌ (래퍼 이름에 `accounts` 없음) |

`_evidence_ctx` 는 접합 시 앞에 공백 1자를 붙인다(`outs[nm] = (outs.get(nm,"") + " " + s)` · `t2_scaffold_get.py:1768`)
⇒ 1023 + 1 = **1024**. 로그 수치와 정확히 일치한다. **선택된 것은 카드-전용 덤프다.**

결과: REFERENCE 의 `customer_products` 에 **checking 계좌가 없다.** 클래스 선택 서브는
`platinum_account`·`credit_cards_ecocard` 는 잡았지만 `green_account_(checking)` 은 **잡을 근거가 없었다**.
서브 지시문은 적용성을 REFERENCE 에 못박는다(A3 축자):

```
"find the APY components for the savings account named in REFERENCE: the account type's base APY, and
 EVERY applicable boost **given the customer's products listed in REFERENCE** - checking-account pairing
 boosts, credit card APY bonuses, relationship bonuses, account tier bonuses."
```

**재료 자체는 도달해 있었다.** `docs.always` 3편은 클래스와 무관하게 항상 실리고, 그중 `doc_…_046` 이 필요한 수치를 축자로 담는다
(같은 궤적 [41] 의 KB 검색 출력으로 확인 · env 축자):

```
doc_bank_accounts_bank_accounts_(general)_046:
  "### Example  If a customer with a Gold Savings Account holds:
   - Green Account (checking): **+0.75% APY boost**
   - Purple Account (checking): +0.1% APY boost
   The customer receives only the Green Account boost of +0.75% (the highest) …"
```

⇒ **문서는 갔고, "이 손님이 Green checking 을 갖고 있다"는 사실만 안 갔다.**
그리고 같은 궤적이 그 기전을 **양성 대조로 스스로 증명한다** — REFERENCE 에 EcoCard 가 들어간 순간
서브는 card +0.6 을 산출했다(`sub=2 rows` · 6.1 = 5.5 + 0.6). 결손은 서브의 판단력이 아니라 **REFERENCE 의 내용**이다.

**그리고 반환문은 그 결손을 감췄다**(축자):

```
[29] tool: Correct savings APY under the stacking policy (base + highest checking boost + highest card bonus
     + all relationship/tier bonuses): 6.1%. Compare this to the APY the system actually applied; …
```

성분은 2개(base·card)인데 문장은 **네 성분 전부**가 반영된 것처럼 선언한다. `op.required_groups=["base"]` 라
checking 부재에 abstain 도 플래그도 없다. → 선행 ⓑ·ⓒ **미수리·재발**.

### 2.4 [30] trial 0 — 실패 결정 지점 2 (`actual_apy=6.0`)

```
[30] ASST: "The correct APY for your Gold Savings Account is 6.1%. However, you mentioned that you received
     $408 in interest this month, which corresponds to an APY of approximately 6.0%."
[30] CALL get_interest_correction {"expected_apy":"6.1","actual_apy":"6.0","principal":"96000",
       "period_start":"10/01/2025","period_end":"10/31/2025","account_id":"sav_wl94k7m3p8_gold"}
[31] tool: [GROUNDING WARNING] 2 input value(s) … dropped: period_start=10/01/2025 …; period_end=10/31/2025 …
     Interest correction … applied(actual) APY=6.0%, correct(expected) APY=6.1%. Correction amount = 8.0.
     Use this as the credit amount and the amount_difference, and use these APY values in the discrepancy report.
```

*"$408 … corresponds to an APY of approximately 6.0%"* 는 **산술적으로 거짓**이다(408×12/96000 = 5.1). 재료는 [1] 에 실재했고
같은 모델이 trial 1 [22] 에서는 같은 식을 **정확히** 세운다 ⇒ 이 오류는 **model**.
우리 층 기여: `ground.scalar_fields[actual_apy].corpus=["ledger"]` 는 **문맥 실재**만 본다. 고객이 *"6.0% APY"* 라고 말했으므로
`6.0` 은 통과한다(`_corpus_texts` 가 `__user_text` 를 포함 · `t2_scaffold_get.py:227-231`). **틀린 값을 막을 술어가 없다.**

### 2.5 [32]~[60] trial 0 — 8이 확정된 뒤의 궤적 (우리 반환문이 만든 5턴 파생 논쟁)

```
[32] CALL call_discoverable_agent_tool {…submit_interest_discrepancy_report_7294 … "expected_apy":6.1,"actual_apy":6.0,"amount_difference":8.0}   → 성공 (WRONGARG 확정)
[35] user: "1) You're saying my **expected APY is 6.1%** (not 6.0%) — what is the **extra 0.1% boost** coming
     from? I want the exact eligibility/criteria."
```

우리 도구가 **성분 내역을 돌려주지 않았으므로** 모델은 6.1 의 출처를 설명할 수 없다. 이후 다섯 턴이 존재하지 않는
"0.1% relationship/tier bonus" 를 KB 에서 찾는 데 소모된다:

```
[36] CALL KB_search_bm25 {"query":"relationship or tier bonus for Gold Savings Account"}
[40] CALL KB_search_bm25 {"query":"additional 0.1% boost for Gold Savings Account"}
[44]/[48]/[52] CALL KB_search_bm25 {"query":"criteria for 0.1% relationship/tier bonus"}   (3회 동일 · [48]·[52] 는 사실상 재시도)
[54] ASST: "I understand that the knowledge base has not provided specific details on the 0.1% relationship/tier bonus."
```

아이러니: [42] 에서 모델은 **정답 성분을 문면에 쓴다** — *"Your Green Account (checking) provides a 0.75% APY boost to your
Gold Savings Account."* 그러나 `get_correct_savings_apy` 를 **재호출하지 않았다**(우리 반환문 6.1 이 이미 권위로 굳었다).
user-sim 은 [55] 에서 오히려 **하향 압력**을 넣는다: *"I don't want my 'expected APY' listed as 6.1% … reflects only boosts you
can substantiate."* 보고서는 [32] 에서 이미 제출된 뒤라 추가 손해는 없다.

```
[58] CALL call_discoverable_agent_tool {…apply_savings_account_credit_6831 … "amount":8.0,"credit_type":"interest_correction"}  → 성공 (WRONGARG 확정)
```

### 2.6 ★★[14]~[20] trial 1 — **분기점** · REFRAW 가 아예 침묵했다

trial 1 은 [3] 에서 user-sim 이 검증정보 5종을 **선제 제공**해 신원 구간이 8메시지 짧다. 그 결과 [14] 시점의 도구-출력 집합에
`get_credit_card_accounts_by_user` 가 **없다**(trial 1 은 그 도구를 한 번도 부르지 않는다 — 대신 `get_credit_card_transactions_by_user` 를 6회 부른다).

```
[14] CALL get_correct_savings_apy {"savings_account_type":"Gold Savings Account",
       "customer_products":"Gold Savings Account, Personal Checking Account","current_balance":"96000"}
[15] tool: Error: [READ-FIRST] … Missing required reads (BASE names): get_all_user_accounts_by_user_id …
[16] CALL unlock_discoverable_agent_tool {"agent_tool_name":"get_all_user_accounts_by_user_id_3847"}
[18] CALL call_discoverable_agent_tool {…}
[19] tool(1629자): "… level: Gold Account … level: **Green Account** … **EcoCard** …"
[20] CALL get_correct_savings_apy {"components":"[base 5.0 / checking 1.0 / relationship 0.5]",
       "savings_account_type":"Gold Savings Account","customer_products":"Gold Savings Account, Personal Checking Account", …}
```

우리 층 로그(축자 · `s373753`) — **`T2_SG_REFRAW` 줄이 없다**:

```
[T2_SG_TRACE]  get_correct_savings_apy: iso=True mode=fetch_formalize ctx=['components','current_balance','customer_products','savings_account_type']
[T2_SG_DOCS]   get_correct_savings_apy: 클래스 ['gold_account','gold_saver_account'] · 문서 12편 · 12655자 전달(검색 0)
[T2_SG_ISOLATE] operand-size get_correct_savings_apy.components: sub=1 rows · source=0 rows
[T2_SCAFFOLD_GET] get_correct_savings_apy -> **5.5**
```

`producer_contains=["accounts"]` 가 **어느 생산자 이름과도 일치하지 않았다**(후보: `verify_identity`,
`get_user_information_by_email`, `get_current_time`, `log_verification`, `unlock_discoverable_agent_tool`,
`call_discoverable_agent_tool`, `get_correct_savings_apy`). `_hit` 가 비면 코드는 **fail-open** 으로 에이전트 인자를 그대로 쓴다
(`t2_scaffold_get.py:720-724`) ⇒ REFERENCE = 에이전트의 손요약 `"Gold Savings Account, Personal Checking Account"`.
`Green` 도 `Eco` 도 없으므로 클래스는 `['gold_account','gold_saver_account']` 12편뿐이고, 서브는 base 하나만 낸다.

**즉 R3 레버는 trial 0 에서 오발화(잘못된 생산자), trial 1 에서 미발화(생산자 0)** — 같은 한 줄의 술어가 두 방식으로 다 빗나갔다.
그리고 두 trial 모두 **바로 직전 턴에 1629자짜리 정답 레코드를 이미 받아 놓고 있었다.**

### 2.7 [22]~[47] trial 1 — 실패 결정 지점 3 · **우리 grounding 이 gold 값 5.1 을 5회 반려했다**

모델은 여기서 **정확히 옳은 유도**를 한다(축자):

```
[22] ASST: "First, I need to derive the actual APY from the interest paid. The formula … \[ \text{Actual APY} =
     \left( \frac{408}{96000} \times 12 \right) \times 100 \approx 5.1\% \]"
[22] CALL get_interest_correction {"expected_apy":"5.5","actual_apy":"5.1","principal":"96000", …}
[23] tool: [GROUNDING WARNING] 3 input value(s) could not be verified against the account records / knowledge base
     and were dropped: **actual_apy=5.1 (not found in the records — re-read the exact value)**; period_start=…; period_end=…
     Interest correction … applied(actual) APY={actual_apy}%, … Correction amount = … **(could not compute — check your arguments)**.
```

`5.1` 은 **gold `action_checks[094_8].arguments.actual_apy` 와 동일한 값**이다. 우리 층이 그것을 5회 반려했다
(로그 실측: `s373753` 의 `T2_SG_GROUND` 7건 중 **5건이 `actual_apy=5.1` 드롭** · msgs 23·25·39·45·47).
기전: `ground.scalar_fields[actual_apy].corpus=["ledger"]` 는 **문자열 실재**만 본다(`_val_grounded` · `t2_scaffold_get.py:438-443`).
`5.1` 은 **유도값**이라 레코드에도 손님 발화에도 없다 ⇒ 드롭. **파생값과 리터럴을 구분하는 술어가 없다.**

그 반려가 유발한 낭비가 궤적에 그대로 남는다 — 모델은 존재하지 않는 저축 거래 read 를 6회 시도한다:

```
[26] ASST: "I will use the `get_credit_card_transactions_by_user` tool to fetch the transactions."
[27]/[29]/[31] tool: No records found in 'credit_card_transaction_history'.
[28]/[30] ASST: "let's directly fetch the savings account transactions. I will use the
     `get_credit_card_transactions_by_user` tool again but this time for the savings account."   ← 같은 도구·같은 인자
```

그리고 여기서도 우리 REFRAW 가 **같은 종류로 빗나간다**:
`[T2_SG_REFRAW] get_interest_correction.transactions_raw ← 도구 출력 원문 1편(**165자**·에이전트 전사 대체)`
— 165 = 54×3 + 공백 3 = `"No records found in 'credit_card_transaction_history'."` **3회 누적분**이다.
needle `["transactions"]` 가 **카드 거래**에 걸리고, gold 가 요구한 `get_bank_account_transactions_9173`(094_3/094_4)는
래퍼 이름 뒤에 숨어 영원히 안 걸린다. 결국 그 read 는 **양 trial 모두 0회**(`action_match=false`).

**자기-그라운딩으로 뒤늦게 통과한다.** [48] 에서 모델이 보고서를 제출하자 그 출력이 원장에 남고:

```
[49] tool: Interest Discrepancy Report Submitted Successfully! … Expected APY: 5.5% - **Actual APY: 5.1%** - …
[61] tool: [GROUNDING WARNING] **2** input value(s) … dropped: period_start=…; period_end=…    ← actual_apy 통과
     … Correction amount = … **32.0**.
```

`_corpus_texts(...,"ledger")` 가 도구 출력을 포함하므로(`t2_scaffold_get.py:227-229`) **에이전트 자신의 write 가 그 값의 근거가 됐다**
— C203 이 코드 주석에 이미 적어 둔 자기-그라운딩 그대로다. 즉 우리 게이트는 **옳은 값을 계산 전에는 막고, 이미 DB 에 쓴 뒤에 허락했다.**

### 2.8 [63]~[71] trial 1 — 신규 증거·명시 요구 후에도 재계산 0

```
[63] user: "Yes, please apply that **$32.00 credit**. And can you also clarify **why the expected APY is 5.5%
     (not 6.0%)**? I have a **Gold savings** with around **$96,000** and a **Green checking** account …
     If there are eligibility rules, caps, or any additional boosts I'm missing … I want that spelled out"
[70]/[72] CALL apply_savings_account_credit_6831 {"amount":32.0}  → 성공 (WRONGARG 확정)
[72] ASST: "Regarding the expected APY of 5.5%, here is the breakdown: - **Base A…"
```

필요한 모든 재료(계좌 레코드 [19]·[55]·[57] · Green checking 명시 [63] · stacking 정책 배달분)가 문맥에 실재하는데
`get_correct_savings_apy` 재호출은 **0회**. 이 부분은 **model** 몫이다(부: 반환문이 성분 내역을 안 줘 검산 불가 — ⓑ).

---

## 3. 레버 발화표 (이 sim 줄만 · 발화/무시/미발화/오발화)

| 레버 | trial 0 | trial 1 | 판정 |
|---|---|---|---|
| `T2_SG_REQREADS`(READ-FIRST) | **1회 발화·먹힘** | **1회 발화·먹힘** | ★**신규 수리 성공**. 두 trial 다 gold 094_1/094_2 를 실행시켰다. 그러나 읽어 온 레코드가 서브에 안 갔다(아래) |
| `T2_SG_REFRAW`(R3·x481) | 2회 발화 — `customer_products←카드덤프(1024자)` · `account_records←카드덤프(1024자)` | `customer_products` **미발화** · `transactions_raw←카드거래(165/581/691자)` 6회 | ★**오발화 + 미발화**. 두 자리 모두 **가장 정확한 생산자(`call_discoverable_agent_tool`)를 구조적으로 못 본다** |
| `T2_SG_DOCS` | 1회 (`17편·16557자`·클래스 4) | 2회 (`12편·12655자`·클래스 2) | **발화·배선 정상**. `always` 3편에 0.75 축자가 실려 도달. 재료는 갔고 **적용성 판단 근거(REFERENCE)** 가 빠졌다 |
| `T2_SG_GROUND` | 1회 (기간 2건 드롭 · `actual_apy=6.0` **통과**) | 7회 (**5회가 gold 값 `actual_apy=5.1` 드롭**) | ★**부호 반전 오발화**. 고객의 틀린 6.0 은 통과시키고 유도된 정답 5.1 은 5회 반려 |
| `T2_SG_ISOLATE` | `sub=2 rows` | `sub=1 rows` ×2 | 발화. 성분 결손이 그대로 통과(`required_groups=["base"]`) |
| `T2_PIN_READ` | 0 | 0 | **미발화**(배선 생존 — halfA 전체 8회) |
| `T2_DEMANDED_STEP` | 0 | 0 | **미발화**(halfA 전체 12회) |
| `T2_CLAIMPROV` | 다수 (`unbacked=0` 유지) | 다수 | 발화하나 이 실패와 무관(양 trial 다 write 를 **실제로** 실행했다) |
| `T2_FOLLOWUP` | 0 | 0 | **미발화**(halfA 전체 27회) |
| `T2_SEARCH_AGENT` | 12회 중 6회 `모두 처리됨 — 침묵` | 9회 중 5회 침묵 | 대체로 **침묵**. 실질 발화는 `savings_accounts`·`checking_accounts`·`business_savings_accounts` 축뿐 |
| `FAB_STRIP` | 0 | 0 | **halfA 전체 0회** — 이 런에서 미발화(대상 없음/비활성) |
| `T2_ARG_PRODUCERS` | 0 | 0 | **halfA 전체 0회** (`T2_ARG_DOC_SUB` 15회는 무관 인자) |
| `T2_REQUIRE_DOC_DELIVER` | 0 | 0 | **미발화**(halfA 전체 17회) |
| `T2_SEARCH_REARM` | 0 | 0 | **미발화**(halfA 전체 14회) |
| `T2_PROV` | 3회 (`SAV123456`·`John Doe`·`provide_statement_details` regen 차단) | — | **정발화**. 날조 인자 3건을 최종 궤적 이전에 제거 |
| `T2_RESOLVE` operator-scope | **0회** | **0회** | ★선행 ⓓ(t7336 trial0 write 0 의 직접 원인) **미재발** |

### 직전 수리가 이 궤적에 개입했나 · 개입하고도 왜 못 샀나

| 수리(2026-08-22) | 개입 | 결과 |
|---|---|---|
| **OL-39** `get_correct_savings_apy.requires_reads=[get_all_user_accounts_by_user_id]` | **개입함**(양 trial 1회 deny) | 목표 행동(gold 094_1/094_2)을 **샀다**. 그러나 **그 read 의 산출물이 소비되는 경로가 없다** — REFRAW 는 래퍼 이름 때문에 못 보고, 서브는 REFERENCE 만 본다. **읽게 만들었으나 읽은 것을 쓰지 못했다** |
| **R3** `ref_from_outputs`(x481·093 의 `Green Checking Account` 전사 손상 처방) | **개입함**(t0) / **침묵**(t1) | 처방의 표적(093 형 전사 손상)과 **동일한 손상이 094 에서 그대로 재현**됐다. 이유는 needle 이 discoverable 래퍼를 못 뚫기 때문 |
| ⓑ 반환문 성분 내역 | **미수리** | trial 0 의 5턴 "0.1% 보너스" 논쟁을 직접 유발 |
| ⓒ `required_groups` | **미수리** | checking 부재에 신호 0 |
| P5 `actual_apy` 파생 검산 | **미수리 · 부호가 반대로 실현** | 검산을 넣기는커녕, **파생된 정답을 5회 반려**했다 |
| ⓓ operator-scope | **해소됨(추정)** | 이번 런 0회 — trial 0 write 가 0 → 2 로 회복 |

---

## 4. 우리 층 결손 (코드 경로·선언 키 지목)

**OL-A · `producer_contains` 가 discoverable 래퍼를 구조적으로 못 본다 (이 실패의 1차 원인)** [CONFIRMED]

- 코드: `scripts/distill/tau2/t2_scaffold_get.py:1758` — `id2name[getattr(tc,"id",None)] = getattr(tc,"name",None)`
  (**바깥 도구 이름**만 기록 · `_eff_tool_name` 미적용) → `:1766-1768` `outs[nm] = outs.get(nm,"") + " " + s` → `:1777` `"__tool_outputs_raw": dict(outs)`
- 코드: `t2_scaffold_get.py:715-724` — `_needles`/`_hit` 가 그 **바깥 이름**에 substring 매칭
- 선언 키: `a2/banking_knowledge.specific.json` → `scaffold_get_tools[get_correct_savings_apy].isolate.ref_from_outputs.customer_products.producer_contains = ["accounts"]`
  (동형: 같은 파일 `…[get_interest_correction].isolate.ref_from_outputs.{account_records,transactions_raw}`)
- **비대칭의 증거**: 같은 파일의 READ-FIRST 게이트는 `t2_scaffold_get.py:1913` 에서 `_g._eff_tool_name(_t2)` 로 래퍼를 **푼다**
  (`t2_gate_patch.py:2491` — `if nm.startswith("call_"): return re.sub(r"_\d+$","",inner)`).
  ⇒ **한쪽은 `get_all_user_accounts_by_user_id` 로 보고, 다른 쪽은 `call_discoverable_agent_tool` 로 본다.** 같은 호출인데 이름이 둘이다.
- 실측 귀결(축자·산술): t0 `1024자` = 카드덤프 1023 + 접합공백 1 · t1 `_hit` 0 → fail-open.
  **양 trial 모두 직전 턴에 1629자 정답 레코드를 갖고 있었다.**

**OL-B · `actual_apy` 의 ledger-존재 grounding 이 파생값을 반려한다 (부호 반전)** [CONFIRMED]

- 코드: `t2_scaffold_get.py:438-443` (`_val_grounded` 실패 시 `flags.append(... not found in the records ...)` · `ctx[param]=None`)
  · 코퍼스 정의 `t2_scaffold_get.py:227-231` (`ledger` = 도구 출력 + `__user_text`)
- 선언 키: `a2/banking_knowledge.specific.json` → `scaffold_get_tools[get_interest_correction].ground.scalar_fields[{"param":"actual_apy","corpus":["ledger"],"on_fail":"drop"}]`
- 실측: trial 1 에서 **gold 값 `5.1` 을 5회 드롭**(`T2_SG_GROUND` 7건 중 5건) · trial 0 에서 **고객의 틀린 `6.0` 은 통과**
  (손님 발화 *"so I should be at 6.0% APY"* 가 `__user_text` 에 있으므로).
  그리고 [49] 의 자기 write 출력이 원장에 남은 뒤에야 5.1 이 통과했다(자기-그라운딩 · C203 기지 결함).

**OL-C · 반환문이 반영되지 않은 성분을 반영했다고 선언한다 (선행 ⓑ 재발)** [CONFIRMED]

- 선언 키: `a2/banking_knowledge.specific.json` → `scaffold_get_tools[get_correct_savings_apy].result_template`
  = `"Correct savings APY under the stacking policy (base + highest checking boost + highest card bonus + all relationship/tier bonuses): {result}%."`
- 실측: t0 성분 2개(`sub=2 rows`) · t1 성분 1개(`sub=1 rows`) 인데 문장은 네 성분 전부를 선언.
  직접 귀결 = trial 0 [35]~[54] 의 5턴 "존재하지 않는 0.1% 보너스" 탐색(KB 검색 5회 · 그중 2회는 동일 질의 재시도).

**OL-D · `op.required_groups=["base"]` — 성분 결손에 abstain 도 플래그도 없다 (선행 ⓒ 재발)** [CONFIRMED]

- 선언 키: `…[get_correct_savings_apy].op = {"required_groups":["base"], "unknown_policy":"flag"}`.
  `unknown_policy:"flag"` 는 **미상 kind** 만 잡고 checking/card 그룹의 **부재**는 무신호.

**OL-E · gold read `get_bank_account_transactions_9173`(094_3/094_4)가 양 trial 0회** [기여 · 부분 우리 층]

- `requires_reads` 에 거래 read 가 **의도적으로 빠져 있다**(A2 `_note_requires_reads` 축자: *"거래 read 는 넣지 않았다 — … 최소 선언"*).
  그 판단 자체는 [[62]] 정합이나, 결과적으로 `actual_apy` 의 유일한 1차 근거(월 이자 크레딧 408 레코드)가 원장에 들어오지 않았고
  OL-B 의 드롭과 겹쳐 trial 1 이 6회 헛 read 로 돈다. **model 부**: 저축 거래를 카드 거래 도구로 6회 시도.

---

## 5. 선행 판정과의 대조 (`t7336_tasks/T7336_TASK_094.md`)

| 선행(t7336) 판정 | t7346 재현 여부 |
|---|---|
| *"`get_correct_savings_apy` 가 양 trial 모두 `5.5` 반환"* | **부분 재현**. t1 = 5.5 동일 · **t0 = 6.1 로 개선**(card +0.6 획득). READ-FIRST 처방이 카드 클래스를 살렸다 |
| ⓐ *"`requires_reads` 미선언 — 게이트가 죽어 있다"* | **수리 확인**. 선언됐고 양 trial 발화·행동 변화 확인(gold 094_1/094_2 실행) |
| ⓑ *"반환문이 네 성분 반영을 선언"* | **완전 재현**(OL-C) — 미수리 |
| ⓒ *"`required_groups=[\"base\"]`"* | **완전 재현**(OL-D) — 미수리 |
| ⓓ *"`t2_resolve.py:221` operator-scope deny ×3 → trial 0 write 0"* | **미재발**(로그 `operator-scope` 0건). trial 0 write 2건 실행 ⇒ 실패가 MISSING→WRONGARG 로 **완화** |
| ⓔ *"`_evs` 가 도구 결과를 안 본다"* | **이 궤적에선 무관**(양 trial 다 write 성공) |
| ⓕ *"`ref_params` 에 보유 상품이 담길 결정론 경로 없음"* [UNPROVEN] | **CONFIRMED 로 승격 · 원인 정밀화**. 경로는 생겼다(R3) — 그런데 **생산자 이름 대조가 래퍼를 못 뚫어** 못 쓴다(OL-A) |
| P5 *"`actual_apy` 파생 검산 미이행"* | **미이행 + 역효과 관측**. 이번엔 파생된 gold 값 5.1 이 **5회 반려**됐다(OL-B) |
| 선행 결론 *"reward 를 죽인 값은 모델이 아니라 우리 도구가 만들었다"* | **같은 결론이 유지된다.** 다만 **원인 지점이 한 칸 앞으로 이동**했다: t7336 = "read 를 안 시켰다" → t7346 = "read 는 시켰는데 **그 산출물을 서브에 못 넘겼다**" |

**기준선 t7328(sha 상이·비엄밀·[[70]] 부호표용)**

| 런 | expected_apy | actual_apy | amount | write | gold 거리 |
|---|---|---|---|---|---|
| t7328 t0 | 6.5 / 6.525 | 4.25 | 180 | 예 | −0.35pp |
| t7328 t1 | 6.5 | 5.0 | 120 | 예 | −0.35pp |
| t7335 A | 5.5 | 5.25 | 20 | 예 | −1.35pp |
| t7336 t0 | — | — | — | **아니오** | write 0 |
| t7336 t1 | 5.5 | 5.0 | 40 | 예 | −1.35pp |
| **t7346 t0** | **6.1** | 6.0 | 8 | 예 | **−0.75pp** |
| **t7346 t1** | 5.5 | **5.1 ✅** | 32 | 예 | −1.35pp |
| gold | 6.85 | 5.1 | 140 | | |

`T2_SG_DOCS` 계열 안에서는 t7346 t0 이 최선(−0.75pp)이나, **t7328 기준선(−0.35pp)에는 아직 못 미친다.**
sha 가 달라 인과 단정은 하지 않되 이 태스크의 부호는 여전히 **음(−)** 으로 기재해야 한다([[70]]).

---

## 6. 원인 확정 (4주체 · [[08]] 궤적 축자 근거만)

| 변이 | 결정 지점 | 주체 | 근거 |
|---|---|---|---|
| **t0** `expected_apy 6.1`(gold 6.85) → `amount 8` | **[28]** | **our_layer** (부: model) | `[T2_SG_REFRAW] customer_products ← 1024자` = 카드전용 덤프(1023+1) · `[T2_SG_DOCS] 클래스 [… **green 없음**]` · `sub=2 rows` → `-> 6.1`. 직전 [27] 에 `level: Green Account` 축자 실재. 코드 `t2_scaffold_get.py:1758·715-724` · 키 `ref_from_outputs.customer_products.producer_contains=["accounts"]`. model 부: [28] 문면이 *"total APY should be 6.0%"* 로 고객 주장 반복 |
| **t0** `actual_apy 6.0`(gold 5.1) | **[30]** | **model** (부: our_layer) | *"$408 … corresponds to an APY of approximately 6.0%"* — 408×12/96000=5.1 은 닫힌 산술이고 재료는 [1] 에 실재. 같은 모델이 t1 [22] 에서 정확히 유도. our_layer 부: `ground[actual_apy].corpus=["ledger"]` 가 고객 발화 6.0 을 통과시킴(`t2_scaffold_get.py:227-231·438-443`) |
| **t0** 5턴 "0.1% 보너스" 탐색 낭비 | [35]~[54] | **our_layer** | `result_template` 이 성분 내역 없이 네 성분 반영을 선언(OL-C). user-sim 은 오도하지 않았다 — 우리 문장이 만든 질문에 답을 요구했을 뿐 |
| **t1** `expected_apy 5.5`(gold 6.85) → `amount 32` | **[20]** | **our_layer** (부: model) | `T2_SG_REFRAW` **미발화**(생산자 0) → fail-open → REFERENCE = 손요약 `"Personal Checking Account"` → 클래스 2개·12편 → `sub=1 rows` → `-> 5.5`. 직전 [19] 에 정답 레코드 1629자 실재 |
| **t1** `actual_apy` 5회 반려 · 6턴 헛 read | [23]~[47] | **our_layer** | `T2_SG_GROUND` 5건이 **gold 값 5.1** 드롭 · `(could not compute — check your arguments)` · 이후 `get_credit_card_transactions_by_user` 6회. [49] 자기 write 뒤에야 통과(자기-그라운딩) |
| **t1** [63] 명시 요구 후 재계산 0 | [63]~[72] | **model** (부: our_layer) | Green checking 을 손님이 [63] 에 축자로 말했는데 `get_correct_savings_apy` 재호출 0. our_layer 부: OL-C 로 검산 불가 |
| gold read 094_3/094_4 미실행 | 전 구간 | **our_layer**(선언 범위) + model | `requires_reads` 가 거래 read 를 의도적으로 제외(A2 `_note_requires_reads` 축자) · model 은 저축 거래를 카드 도구로 6회 시도 |

**cause_primary = `our_layer` (양 trial).**
- t0: 종단 3필드를 전부 정한 값 `-> 6.1` 이 **우리가 고른 잘못된 참조**에서 나왔다.
- t1: 같은 자리에서 참조 자체가 **fail-open** 으로 손요약이 됐다.

`env` 귀속 0 (BLOCKED 0건 · 모든 도구 응답 정상).
`user_sim` 귀속 0 — 오히려 t1 [33] 에서 연도 오류를 교정해 주고 [63] 에서 Green checking·보너스 재확인을 명시 요구했다.
t0 [55] 의 하향 압력(*"I don't want my expected APY listed as 6.1%"*)은 보고서 제출([32]) **이후**라 수치에 영향 없음.

---

## 7. 처방 후보 (제안까지 · 코드 수정 없음 · [[23]] 준수)

1. **[P1 · 최우선 · 우리 층 버그픽스] `_evidence_ctx` 의 생산자 키를 `_eff_tool_name` 으로 정규화한다.**
   `t2_scaffold_get.py:1758`. 같은 파일 `:1913` 이 이미 `_g._eff_tool_name` 을 쓰고 있으므로 **새 술어가 아니라 기존 술어의 일관 적용**이다
   (도메인 리터럴 0 · [[59]] 정합). 기존 두 키(`__tool_outputs`·소문자본)는 grounding 소비자가 여럿이므로([[67]])
   **`__tool_outputs_raw` 만** 이중 등재(래퍼 이름 + 유효 이름)하는 최소 형태를 권한다.
   이 하나가 t0·t1 양쪽의 1차 원인을 동시에 닫는다: `producer_contains=["accounts"]` 가
   `get_all_user_accounts_by_user_id` 에 걸려 1629자 레코드(Green Account + EcoCard 동봉)가 REFERENCE 가 된다.
   ⚠[[70]] 무엇을 파는가: REFERENCE 바이트가 1024 → 1629 로 증가(+59%)하고, 카드 3장이 들어가 클래스 과포함이 늘 수 있다.
   셀 것 = `[T2_SG_REFRAW] … N자` 값 · `[T2_SG_DOCS] 클래스` 목록에 `green_account_(checking)` 등장 여부 · 093/094 의 `expected_apy` · 전체 reward 짝.
   ⚠부정통제 필수([[57]]): 같은 sha 에서 이 한 줄만 바꾼 A/B.

2. **[P2 · 우리 층 버그픽스 · 선행 ⓑ 재제출] 반환문이 실제 반영된 성분만 말하게 한다.**
   선언 키 `…[get_correct_savings_apy].result_template`. 최소 형태 = `op.reducers` 의 그룹 키와 실제 입력 그룹의 **집합 차**를 인쇄
   (*"components applied: base, card. No checking / relationship / tier component was supplied or verified."*).
   엔진은 집합 연산만 한다(도메인 판단 0 · [[59]]) · [[64]]: 무엇이 빠졌는지 이름으로 말한다.
   t0 의 5턴 "0.1%" 논쟁이 정확히 이 결손의 비용이다.

3. **[P3 · 우리 층 버그픽스] `actual_apy` grounding 을 "리터럴 실재" 에서 "파생 검산" 으로 바꾼다.**
   현행은 **정답을 막고 오답을 통과시켰다**(OL-B · 실측 5회 반려). 원장에 (월 interest credit 금액, `current_holdings`)가 있으면
   `credit×12/principal` 은 닫힌 산술이므로 **드롭 대신 검산**이 가능하다. 값을 추출하는 게 아니라 모델 기입값과 대조만 한다([[03b]] 정합).
   최소 대안(더 좁음): `on_fail` 을 `drop` → `flag` 로 낮춰 **계산은 진행하되 경고만** — 자기-그라운딩 경로(write 후 통과)를 없앤다.
   ⚠[[70]]: 날조 차단력을 일부 판다. 셀 것 = `T2_SG_GROUND` 드롭 건수 · `actual_apy` 가 gold 와 일치한 sim 수.

4. **[P4 · 선언] `producer_contains` needle 을 래퍼-정규화 이후의 **정확 이름**으로 좁힌다.**
   P1 이 들어가면 `["accounts"]` 같은 느슨한 substring 은 여전히 `get_credit_card_accounts_by_user` 를 함께 잡는다.
   `get_all_user_accounts_by_user_id` 처럼 **선언된 도구 이름 전체**로 못박는 편이 안전하다(`requires_reads` 와 동일 문자열 재사용 = 저작 0).
   `get_interest_correction.transactions_raw` 의 `["transactions"]` 도 같은 병(카드 거래 165자를 저축 거래로 실었다).

5. **[P5 · 계측] `T2_SG_ISOLATE operand-size` 에 **kind 목록**을 함께 인쇄한다.**
   현행 `sub=2 rows` 로는 어느 성분이 빠졌는지 로그만으로 알 수 없어 이번 분석도 산술 역산에 의존했다.
   포렌식 비용을 줄이는 계측 전용 변경(거동 무변).

---

## 부록 · 참고 파일 경로

- 결과/로그: `C:\workspace\ba-frft\reports\facet_rft_2026\sim_results\bank_t7346_halfA_20260822.results.json.gz` / `.log.gz`
- 기준선: `…\sim_results\bank_t7328_halfA_20260819r.results.json.gz`
- 선언: `C:\workspace\ba-frft\scripts\distill\tau2\a2\banking_knowledge.specific.json`
  → `scaffold_get_tools[get_correct_savings_apy]` (`isolate.ref_from_outputs` · `result_template` · `op.required_groups` · `requires_reads`)
  → `scaffold_get_tools[get_interest_correction]` (`ground.scalar_fields[actual_apy]` · `isolate.ref_from_outputs`)
- 코드: `C:\workspace\ba-frft\scripts\distill\tau2\t2_scaffold_get.py` (227-231 · 438-443 · 552-680 · 715-724 · 1749-1777 · 1906-1970)
  · `C:\workspace\ba-frft\scripts\distill\tau2\t2_gate_patch.py` (2491 `_eff_tool_name`)
- 선행: `C:\workspace\ba-frft\reports\facet_rft_2026\t7336_tasks\T7336_TASK_094.md` · 같은 폴더 `T7336_TASK_093.md`
