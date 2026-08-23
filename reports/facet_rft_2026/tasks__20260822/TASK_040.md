# task_040 — bank_t7346 halfB 궤적 per-step 포렌식 (2026-08-22 런 · 보고 2026-08-23)

> 자료 = 전부 로컬. `sim_results/bank_t7346_halfB_20260822.results.json.gz` · 같은 tag `.log.gz`
> (줄 접두 `[sim=task_040#s626729]` = trial 0, `[sim=task_040#s373753]` = trial 1).
> 대조 = `bank_t7328_halfB_20260819r2.results.json.gz` + 그 `.fb.jsonl.gz` 사이드카(sha 상이).
> gold(`reward_info`)는 **진단용으로만** 썼다([[23]]). 수리·코드 수정 0.

---

## 0. 한 줄 요약

- **trial 0 (s626729)**: dispute **0건 실행**. 원인은 모델이 게으른 게 아니라 **우리 층이
  `give_discoverable_user_tool` 을 5회 거부**했고(WAG 2 + SIGNATURE 3) 모델이 그 뒤로 도구를
  넘기는 대신 **산문으로만 안내**하는 상태에 고착했다. gold 040_3/040_4/040_5(카드 last-4 획득
  경로)가 통째로 못 열려 8건 전부 MISSING.
- **trial 1 (s373753)**: **열거는 완전 성공** — gold 8개 transaction_id 를 전부 찾아 8건 전부 실행.
  실패는 오직 **필드 값 4종**: `issue_noticed_date` 8/8 오류 · `address` 7/8 빈 문자열 ·
  `eligible_for_provisional_credit` 6/8 오류 · 040_14 `resolution_requested`/`partial_refund_amount`.
- 우리-층 확정 결손 2건: ⓐ **`T2_ARG_EMPTY` 가 discoverable 디스패처 write 에 구조적으로 죽어 있다**
  (오프라인 재현 확인) → 빈 `address` 7건이 무경고 통과. ⓑ **A2 자체가 give 에 대해 모순된 두
  선언**을 갖고 있고(서명은 `arguments` 금지 / WAG 피드백은 `arguments` 를 채워 다시 넘기라고 지시)
  체인 순서상 **모순 쪽이 이긴다**.

---

## 1. 채점 축 (먼저 확인)

```
reward_info.reward        = 0.0   (trial 0, trial 1 모두)
reward_info.reward_basis  = ["DB"]
reward_info.reward_breakdown = {"DB": 0.0}
reward_info.db_check      = {"db_match": false, "db_reward": 0.0}
reward_info.info.action   = null      ← ACTION 축 아님
```

⇒ **DB-해시 축**이다. `action_checks` 는 진단 보조로만 읽는다(C583ⓖ).
gold 변이(=DB 를 바꾸는 호출)는 `log_verification` 1건 + `file_credit_card_transaction_dispute_4829`
**8건**. 나머지 gold action(040_1~040_6: unlock / give / 손님 실행)은 `tool_type=generic` 이거나
DB 를 안 바꾸므로 **점수의 직접 단위가 아니다** — 단 8건의 인자를 만드는 **전제 경로**다.

---

## 2. 변이 집합 (`t2_forensic.mutation_diff` 정본 · 손 비교기 0)

### trial 0 (seed 626729 · term=`user_stop` · 61 msg · 1,474s)

| 축 | 수 | 내용 |
|---|---|---|
| matched | 1 | `log_verification` (완전 일치) |
| **MISSING** | **8** | gold dispute 8건 전부 (`txn_25e23705f61f`, `txn_fd4c3871654e`, `txn_a1b2c3d4e503`, `txn_a1b2c3d4e510`, `txn_a1b2c3d4e508`, `txn_a1b2c3d4e513`, `txn_bc21d98cc4e4`, `txn_3ef1a3e9bf56`) |
| WRONGARG | 0 | — |
| EXTRA / DUP | 0 / 0 | — |
| BLOCKED | 2 | 둘 다 `deny=env` — msg41 `unexpected keyword argument 'date_noticed'`, msg43 `missing 12 required positional arguments` |

### trial 1 (seed 373753 · term=`user_stop` · 119 msg · 3,816s)

| 축 | 수 | 내용 |
|---|---|---|
| matched | 1 | `log_verification` |
| **MISSING** | **8** | gold 8건 (인자가 달라 어느 것도 gold 키와 일치하지 않음) |
| **WRONGARG** | **8** | 실행된 dispute 8건 전부 — **transaction_id 는 8/8 정답** |
| EXTRA / DUP / BLOCKED | 0 / 0 / 0 | env 거부 0 |

**WRONGARG 필드별 대조** (gold ↔ 보낸 값. 일치 필드는 생략)

| msg | transaction_id | gold aid | 어긋난 필드 |
|---|---|---|---|
| 49 | `txn_a1b2c3d4e503` | 040_9 | `issue_noticed_date` `11/14/2025`↔`noticed today (date unknown)` · `eligible_for_provisional_credit` False↔True |
| 57 | `txn_a1b2c3d4e510` | 040_10 | `address` `4532 Magnolia Lane…`↔`""` · `issue_noticed_date` · `eligible…` |
| 61 | `txn_a1b2c3d4e508` | 040_11 | `address` `""` · `issue_noticed_date` · `eligible…` |
| 63 | `txn_a1b2c3d4e513` | 040_12 | `address` `""` · `issue_noticed_date` · `eligible…` |
| 99 | `txn_fd4c3871654e` | 040_8 | `address` `""` · `issue_noticed_date` |
| 103 | `txn_25e23705f61f` | 040_7 | `address` `""` · `issue_noticed_date` |
| 105 | `txn_bc21d98cc4e4` | 040_13 | `address` `""` · `issue_noticed_date` · `eligible…` |
| 115 | `txn_3ef1a3e9bf56` | 040_14 | `address` `""` · `issue_noticed_date` · `eligible…` · `resolution_requested` full↔partial · `partial_refund_amount` None↔`189.99` |

정답이었던 것: `transaction_id` 8/8 · `dispute_reason` 8/8 · `purchase_date` 8/8 ·
`card_last_4_digits` 8/8 · `contacted_merchant` 8/8 · `card_action` 8/8 · `user_id`/`full_name`/
`phone`/`email` 8/8. ⇒ **열거·매칭·정책-enum 매핑은 다 됐고 값 4칸에서 죽었다.**

---

## 3. step-by-step 결정 지점 (축자 인용)

### 3-1. 두 trial 의 **분기점 = msg [13]**

msg [0]~[12] 은 두 trial 이 동형이다(bm25 검색 → unlock → verify_identity 오호출 → 이름 요청 →
`get_user_information_by_name` 로 레코드 회수). 갈린 자리는 **[12] 직후 첫 assistant 턴**:

- trial 1 [15]: 산문 뒤에 실제로 호출한다 —
  `TOOLCALL give_discoverable_user_tool {"discoverable_tool_name": "get_card_last_4_digits", "arguments": "{\"credit_card_account_id\": \"<your_personal_gold_account_id>\"}"}`
  → [16] `Tool given to user: get_card_last_4_digits` (**성공**).
- trial 0 [23]: **호출이 사라지고 산문만 남는다** —
  > "To get the last four digits of your **Personal Gold Card**, please use the following tool:
  > ```call_discoverable_tool("get_card_last_4_digits", {"credit_card_account_id": "your_personal_gold_card_account_id"})```"

  (`call_discoverable_tool` 은 **존재하지 않는 이름**이다.)

로그가 그 사라짐의 이유를 직접 적는다(trial 0, turn 23):

```
[T2_LEVER] T2_WRITE_EVIDENCE sim=task_040#s626729
[T2_WRITE_ARG_GROUND] deny tool=give_discoverable_user_tool inner=
[T2_MATERIAL_GATE] stop=other_lever(wev) turn=23
[T2_TOOL_SIGNATURE] would-deny tool=give_discoverable_user_tool but preempted-by=wev
   … (같은 턴 1회 더 반복) …
[T2_TOOL_SIGNATURE] deny tool=give_discoverable_user_tool          ← turn 23
```

즉 **모델은 give 를 실제로 emit 했고, 우리가 세 번 막았다.** 이후 turn 29·43 에서도 같은 deny:

```
[T2_MATERIAL_GATE] stop=resolve_cap(정체 3회) turn=29
[T2_TOOL_SIGNATURE] deny tool=give_discoverable_user_tool
[T2_STACK] window folded fb tag=signature (same fingerprint (seen=15)) — deny stays, body kept (R9)
…
[T2_MATERIAL_GATE] stop=resolve_cap(정체 3회) turn=43
[T2_TOOL_SIGNATURE] deny tool=give_discoverable_user_tool
```

trial 0 의 `T2_RESOLVE_CAP 리셋(실행)` 목록에 `give_discoverable_user_tool` 은 **끝까지 안 나온다**
(나온 것 = KB_search_bm25, unlock, get_user_information_by_name, verify_identity,
get_credit_card_accounts_by_user, get_credit_card_transactions_by_user) ⇒ **give 는 한 번도 실행되지
않았다**. trial 1 은 반대로 다음 줄이 있다:

```
[T2_TOOL_SIGNATURE] final-word deny tool=give_discoverable_user_tool (try 1)
[T2_RESOLVE_CAP] 리셋(실행): 새 실행 ['give_discoverable_user_tool'] (정체 1회 → 0)
[T2_GIVE_EXEC] nudge idle=['get_card_last_4_digits']
```

`final-word` 경로(`t2_gate_patch.py:12288-12312`, `while _sf_tries < 2` … `_new9 is None → break`)에서
재생성이 실패하자 **위반 호출이 그대로 커밋**되어 give 가 통과했다. trial 1 이 8건을 실행할 수 있었던
것은 이 **우연한 탈출구** 덕이다(그 호출은 `arguments` 를 실은 채였고 그래서 gold action 040_3 은
`action_match=false`).

**trial 0 의 그 후 20턴**(msg 29~60)은 이 고착의 결과다. 손님이 두 번 직접 실행을 시도하고 실패를
그대로 보고한다:

> [30] user `TOOLCALL call_discoverable_user_tool {"discoverable_tool_name":"get_card_last_4_digits", …}`
> [31] tool "Error: Tool 'get_card_last_4_digits' **has not been given to you by the agent**. The agent must first use `give_discoverable_user_tool`…"
> [34] user "Can you enable/share that tool on your side…?"
> [46] user "I'm still not able to run that tool on my end… Can you either: 1) enable/share that tool properly, …"

에이전트는 [45]·[47]·[51]·[55] 에서 **같은 산문 안내를 4회 반복**하고 끝내 [57] 에서 사람 이관을 제안,
[58]/[60] `###TRANSFER###` 로 종료. 그 사이 dispute 는 msg41/43 에서 두 번 시도됐으나 **인자 3개짜리**
호출이라 env 가 거절:

> [41] `TOOLCALL call_discoverable_agent_tool {"agent_tool_name":"file_credit_card_transaction_dispute_4829","arguments":"{\"transaction_id\": \"txn_a1b2c3d4e503\", \"dispute_reason\": \"Fraud/unauthorized\", \"date_noticed\": \"11/14/2025\"}"}`
> [42] tool "Error: Invalid arguments: … got an unexpected keyword argument 'date_noticed'"

여기서 **필요한 정보는 그 시점 문맥에 전부 실재했다**: 도구 파라미터 명세 전문이 [4] 와 [6] 에 두 번
인쇄되어 있었고(`dispute_reason: … Must be one of: 'unauthorized_fraudulent_charge', …`), 사용자
레코드([12])·카드 계좌([26])·57건 거래([28],[40])도 이미 읽혀 있었다. 즉 **읽기 결손이 아니라
스키마 준수 결손**이다(모델).

### 3-2. trial 1 결손 ① `issue_noticed_date` (8/8) — 값은 있었는데 **user-sim 요구에 항복**

값의 실재는 두 겹으로 확인된다.

- 도구 출력: `[26] tool  The current time is 2025-11-14 03:40:00 EST.` — 그리고 에이전트는 그것을
  `[27] log_verification(... time_verified="2025-11-14 03:40:00 EST")` 로 **직접 사용**했다.
- 우리 층 로그: `[T2_SEARCH_AGENT] … now=2025-11-14 turn=2 (스펙 있음 · 원값 '11/14/2025' · 대화텍스트 1)`
  — 우리가 이미 `11/14/2025` 를 대화에 배달했다.

에이전트는 [41] 에서 **처음엔 옳게** 판단했다:

> [41] assistant "Since you mentioned that you noticed the issue today, I will use the current date for
> the `issue_noticed_date`. … **Issue noticed date**: Today's date (2025-11-14)"

user-sim 이 두 턴에 걸쳐 밀어붙인다:

> [42] user "I **don't want you to use today's actual date** (I genuinely don't remember it)…"
> [44] user "for the **issue noticed date**, please keep it as **"noticed today (date unknown)"** on my end
> — I don't want it recorded as a specific date I didn't provide."

에이전트가 항복한다:

> [45] assistant "Absolutely, Kwame. We can keep the "issue noticed date" as **"noticed today (date unknown)"**
> for your records."

이후 8건 전부 그 문자열이 실린다. 도구 스키마는 [6] 에서 `issue_noticed_date: string (required) -
Date when the user noticed the issue, **format MM/DD/YYYY**` 라고 두 번 인쇄돼 있었다.
**결정 지점 = msg [45]** (그 이전 [41] 에는 정답이 있었다).

⚠ 태스크 대본은 손님에게 *"just mention that it was noticed today and you forget today's date"* 까지만
시킨다 — *"이 문자열을 그대로 기록해라"* 는 user-sim 의 **증폭**이다. 그러나 [[21]] 에 따라 면책이
아니다: 형식이 선언된 필드에서 손님의 표현을 그대로 기입한 것은 에이전트 측 흡수 실패다.

### 3-3. trial 1 결손 ② `address` = `""` (7/8) — **원장에 있는 값을 지웠다**

첫 건([49])은 정상이었다: `"address": "4532 Magnolia Lane, Philadelphia, PA 19103"`.
값의 출처는 [12] `get_user_information_by_name` 출력의 축자 `address: 4532 Magnolia Lane, Philadelphia, PA 19103`.

전환점은 손님의 **조건부** 요구다:

> [56] user "Two important corrections before you submit:
> - **Address:** I didn't provide an address, so please **don't use "4532 Magnolia Lane…"**
>   **unless you're seeing that on file from your system and can confirm it's what you have on record.**"

조건(`on file from your system`)은 **충족돼 있었다**(그리고 같은 값으로 이미 `log_verification` 을
통과시켰다). 그럼에도 다음 호출부터 필드가 빈 문자열이 된다:

> [57] `… "email": "midnight_runner_42@gmail.com", **"address": ""**, "contacted_merchant": true, …`

이후 msg 61·63·99·103·105·115 까지 **7건 연속** 빈 문자열. [[25]] 기준으로 손님 발화는 외부 주장이고
레코드가 원장인데, 에이전트가 원장을 버리고 주장을 따랐다.

**우리 층이 여기서 아무 말도 안 했다** — 아래 §4 ⓐ.

### 3-4. trial 1 결손 ③ `eligible_for_provisional_credit` (6/8) — 미읽은 상한

에이전트는 8건 전부 `true` 로 채웠다. 이 값을 결정하는 두 재료가 **문맥에 있었다**:

- [3] KB 배달 문서 `Provisional Credit Eligibility Guidelines (Internal)` 축자:
  > "4. Previous Disputes: The customer has not filed **more than 2 disputes in the past 12 months**"
  > "## NOT Eligible Scenarios … Dispute reason is 'incorrect_amount', 'goods_services_not_as_described',
  > 'canceled_subscription_still_charging', or 'refund_never_processed'"
- 같은 [3] 배달문 `Checking User Dispute History (Internal)` 축자:
  > "To retrieve a user's credit card dispute history, use the **get_user_dispute_history_7291** tool."

그런데 `get_user_dispute_history_7291` 은 **두 trial 어디에서도 호출되지 않았다**(문자열은 KB 본문
2~4회 등장, 호출 0). 초기 DB 에는 이미 `provisional_credit_given: true` 인 dispute 1건
(`dsp_ed3ab3dce038`)이 있고 gold 는 그래서 상위 2건만 True 로 둔다. 에이전트는 그 존재를 확인할
읽기를 하지 않았고, 이유-부적격 4종(`incorrect_amount`·`goods_services_not_as_described`·
`canceled_subscription_still_charging`·`refund_never_processed`)에도 `true` 를 실었다 — 즉 **읽은
문서의 닫힌 규칙조차 적용하지 않았다**([[63]] 빼기 실패의 전형).

### 3-5. trial 1 결손 ④ msg 115 Comcast — user-sim 의 오도를 그대로 실었다

gold 040_14 = `refund_never_processed` / `full_refund` / `partial_refund_amount` 없음.
에이전트는 `partial_refund` + `189.99` 를 실었다. 손님 대본 자체가 "promised a **partial refund**"
라고 말하므로 표면어에 끌린 매핑이다. (`incorrect_amount` 인 PECO 는 gold 도 `partial_refund` 라
에이전트가 맞혔다 — 즉 규칙이 아니라 단어를 봤다.)

---

## 4. 레버 발화표 (이 sim 줄만 · `log.gz` grep)

| 레버 | trial 0 | trial 1 | 판정 |
|---|---|---|---|
| `T2_SG_DOCS` | 0 | 0 | **미발화** (런 메타 `on: … T2_SG_DOCS=1` 인데 이 태스크엔 침묵) |
| `T2_PIN_READ` | 0 | 0 | 미발화 |
| `T2_DEMANDED_STEP` | 0 | 0 | 미발화 |
| `T2_CLAIMPROV` | 30 | 23 | 발화 — 산문 주장↔실행 대조는 정상 동작(`kind='give' … 원장 밖 — 강등`), 그러나 give 를 실행시키진 못함 |
| `T2_FOLLOWUP` | 0 | 0 | 미발화 |
| `T2_SEARCH_AGENT` | 16 | 18 | 발화 — credit_cards / business_credit_cards 두 축 배달 완료 후 침묵. **필요 문서(provisional credit 규정·dispute history 도구명)를 실제로 배달했다** ⇒ 배달은 성공, 소비가 실패 |
| `FAB_STRIP` | 0 | 0 | 미발화 |
| `T2_ARG_PRODUCERS` | 0 | 0 | 미발화 |
| READ-FIRST / `T2_WRITE_EVIDENCE` | 18 | 7 | 발화 — dispute 의 `card_last_4_digits` 선행-read 강제. trial 1 에선 last-4 8/8 정답이라 **성공**. trial 0 에선 give 가 막혀 충족 불가 |
| `T2_REQUIRE_DOC_DELIVER` | 0 | 0 | 미발화 |
| `T2_SEARCH_REARM` | 2 | 0 | 발화(t0 turn 37 delta 배달 6,973자) — 효과 없음 |
| `T2_WRITE_ARG_GROUND` | **2 (deny)** | **2 (deny)** | **오발화(§5 ⓑ)** — 표적이 `give_discoverable_user_tool` 이고, 그 피드백이 SIGNATURE 와 **반대 방향**을 지시 |
| `T2_TOOL_SIGNATURE` | **5** (would-deny 2 + deny 3) | **6** (deny 1 + would-deny 2 + final 1) | 발화 — 내용은 정당(정책 축자 근거)이나 t0 에선 회복 불가 고착을 만들었다 |
| `T2_VALUE_ACQUIRE` | 6 | 6 | 발화 — "give 를 넘겨라" 넛지. **자기 층의 deny 와 상쇄** |
| `T2_ARG_EMPTY` | **0** | **0** | **미발화 (구조적 死)** — §5 ⓐ |
| `T2_HAVE_VALUE` | 0 | 0 | 미발화(producer 성공 출력이 없어 정당한 침묵) |
| `T2_REF_VERIFY` / `T2_UNKNOWN_BOOL` | 0 | 0 | 미발화 |
| `T2_MATERIAL_GATE` stop | 35 | 41 | `resolve_cap(정체 3회)` 래치 — t0 turn 23~53, t1 turn 73~ (C537 이 이미 "손해=시간·문맥" 으로 판정) |
| `T2_TERM_GRANT(_USERDEMAND)` | 1+1 | 1+1 | 발화 — 손님 요구 이관 승인 |
| `T2_TRANSFER_LEAVES_STEPS` | `qty=10 executed=1` | `qty=10 executed=4` | 발화 — 미완 잔여를 정확히 셌다(진단 정확·행동 변화는 없음) |

> `[T2_STACK] window folded fb tag=signature (…) — **deny stays, body kept (R9)**` 가 turn 29·43 에
> 찍혀 있다 ⇒ t7313 보고서 §1 이 지적한 **이름 없는 거부 본문**(`_FB_GENERIC`) 문제는 이 궤적에선
> 재현되지 않았다(R9 로 본문이 보존됨). 그 항목은 **수리된 것으로 관측된다**.

---

## 5. 우리-층 주장 (코드 경로 지목)

### ⓐ CONFIRMED — `T2_ARG_EMPTY` 는 discoverable 디스패처 write 에 **구조적으로 죽어 있다**

- 코드 경로:
  - `t2_gate_patch.py:1513` `_arg_empty_deny(agent, tc, a2, applies_to)` →
    `t2_gate_patch.py:1517` `req = _schema_required(agent, name)` · `:1519 if not req: return None`
  - `name` = `t2_gate_patch.py:2491` `_eff_tool_name(tc)` → `call_` 디스패처를 unwrap 한 뒤
    **`re.sub(r"_\d+$", "", inner)`** ⇒ `file_credit_card_transaction_dispute` (접미 `_4829` 제거)
  - `t2_gate_patch.py:1495` `_schema_required` 캐시 키 = `agent.tools` 의 `openai_schema` **원본 이름**
    (= `_4829` 포함) ⇒ **키 불일치로 항상 캐시 미스**. 게다가 discoverable 도구는 애초에
    `agent.tools` 에 등록되지 않고 디스패처만 등록된다.
- 오프라인 재현(로컬 확인, 부작용 0):

  | 케이스 | `_arg_empty_deny` 반환 |
  |---|---|
  | 스키마가 **정확한 env 이름**(`…_4829`)으로 등록 | `None` (미발화) |
  | 스키마가 **접미 제거 이름**으로 등록 | `[ARG-EMPTY] … 'address' …` (발화) |
  | 디스패처 스키마만 등록(=실제 환경) | `None` (미발화) |

- 라이브 증거: 이 런 halfB 로그 전체에서 `T2_ARG_EMPTY` **0회**, halfA 에서 **2회 — 둘 다 공개 도구
  `get_reward_discrepancies`**. 040 은 `address:""` 를 **7회** 보냈는데 한 줄도 안 찍혔다.
- 손해 범위: `address` 7건. **단독으로는 pass 를 못 산다**(같은 7건이 `issue_noticed_date` 로도
  틀렸다). ⇒ 필요조건이지 충분조건 아님을 명시한다([[70]]).

### ⓑ CONFIRMED(모순 자체) / PLAUSIBLE(인과 비중) — A2 가 `give_discoverable_user_tool` 에 대해 **서로 반대되는 두 지시**를 싣고, 체인 순서가 모순 쪽을 이기게 한다

- 선언 키 1 — `a2/banking_knowledge.specific.json` `tool_signatures`:
  `{"give_discoverable_user_tool": ["discoverable_tool_name"]}`
  → 피드백(`t2_signature.py:25` `FEEDBACK`): *"takes only `discoverable_tool_name` … you also passed
  `arguments`. **Re-issue the call with the declared argument(s) only.**"*
- 선언 키 2 — 같은 파일 `write_arg_grounding` 3번째 항목(`applies_to: "give_discoverable_user_tool"`,
  `grounded_args: [transaction_id, credit_card_account_id, card_last_4_digits, user_id]`):
  → 피드백 축자: *"the value '{val}' you passed for {arg} **in this hand-off** … **hand the tool over
  again with the actual value.**"* ⇒ `arguments` 를 **채워서 다시 넘기라**는 지시.
- 순서를 정하는 코드: `t2_gate_patch.py:9287-9291` `_chain = [("gate",…),("prov",…),…,("wev", wev_fb),…]`
  → `_blocker = next((n for n,v in _chain if v), None)` → `_blocker` 가 있으면 SIGNATURE 는
  `would-deny … preempted-by=%s` 로 침묵. 로그가 그대로 찍는다:
  `[T2_TOOL_SIGNATURE] would-deny tool=give_discoverable_user_tool but preempted-by=wev` (양 trial).
- 여기에 `T2_VALUE_ACQUIRE`(6회 "give 를 넘겨라")와 WEV 피드백 본문
  (*"call give_discoverable_user_tool to give the customer the get_card_last_4_digits tool, **ask them
  to run it with their credit_card_account_id**"*)이 겹친다 ⇒ 우리 층이 한 턴에 **"넘겨라" + "값을 채워
  넘겨라" + "인자를 빼고 넘겨라"** 세 지시를 동시에 낸다([[55]] 문구 모순).
- 인과 비중을 `PLAUSIBLE` 로 남기는 이유: t0 에서 give 가 0회 실행된 것은 사실이고 deny 5회도 사실이나,
  "deny 가 없었으면 give 했을 것"은 격리 프로브 없이는 반증 불가다([[18]]·[[62]]). 다만 **동일 seed
  626729 가 t7328 에서도 SIGNATURE 10회 루프로 같은 방식으로 죽었고, 같은 런의 seed 373753 은 두 런
  모두 SIGNATURE 루프 없이 dispute 를 실행**했다 ⇒ 재현성은 2런 2sha 에서 확인된다.

### 우리-층이 **아닌** 것 (UNPROVEN / model)

- `issue_noticed_date` — 우리 층에 이 필드를 보는 규칙 자체가 없다(WAG `grounded_args` 에 미포함, 있어도
  손님 발화가 코퍼스라 통과). 값은 두 경로로 문맥에 실재했고 [41] 에서 스스로 맞혔다가 [45] 에서
  뒤집었다 ⇒ **model** (트리거는 `user_sim`).
- `address` **값의 선택** — 원장 값이 있는데 지운 것은 model. 다만 그 결과를 **아무도 안 잡은 것**은 ⓐ.
- `eligible_for_provisional_credit` — 근거 문서를 우리가 배달했고 도구 이름도 배달했다. 읽기 미실행·
  닫힌 규칙 미적용 ⇒ **model**. (레버 부재 = 결손이지 결함 아님.)

---

## 6. 선행 판정과의 대조

| 선행 | 판정 | 이번(t7346) |
|---|---|---|
| `T7313_040_WINDOW_LOOP_DIAGNOSIS_2026_08_18.md` §0 — 사이드카 최빈 문면 `[6x] [SIGNATURE] give… you also passed 'arguments'` · `[4x] [WRITE-GROUNDING] the value 'your_credit_card_account_id' …` · `[3x] [VALUE-ACQUIRE]` | 창 순환의 핵심 문면으로 기록 | **동일 원인 지속.** t7346 t0 = SIGNATURE deny 3 + WAG deny 2 + VALUE_ACQUIRE 6, give 실행 0. **바뀐 것은 규모뿐**(t7313 turn 104 → t7346 turn 53, `GO_MAX_STEPS` 계열 억제 효과) |
| 같은 문서 §1 ⒜ — `t2_gate_patch.py:8770` `_FB_GENERIC` 이름 없는 거부 | [[64]] 위반으로 지목 | **재현 안 됨.** 이 궤적의 fold 는 `deny stays, **body kept (R9)**` 로 찍힌다 ⇒ 수리 확인 |
| 같은 문서 §3-1 — resolve_cap 래치, "손해는 pass 가 아니라 시간·문맥" | 조기 종료 기각 | **동일**. t0 turn 23~53 · t1 turn 73~ 래치, 그 뒤로도 새 실행은 계속 생겼다 |
| `a2/banking_knowledge.specific.json` `write_arg_grounding[2]._note` (2026-08-02 P9/AX32) — *"040 = placeholder '[Enter your credit card account ID here]' 가 그대로 통과"* | 그래서 give 를 WAG 표적에 **추가**했다 | **그 추가가 지금 SIGNATURE 를 선점하는 당사자다.** 즉 P9 수리가 산 것(placeholder 차단)과 판 것(give 실행 자체의 봉쇄·서명 지시 무력화)이 같은 자리에 있다([[70]] 절충 대상) |
| 대조 런 `t7328_halfB` (sha 상이) | 040 t0 reward 0 / t1 reward 0 | **t7328 t1 은 gold dispute 2건(040_7·040_8)을 정확히 일치**시켰고 나머지 5건은 `eligible_for_provisional_credit` **한 칸만** 틀렸다. t7346 t1 은 같은 5건이 `address`+`issue_noticed_date`까지 추가로 틀렸다 ⇒ **필드 축에서는 t7346 이 후퇴**. 단 seed 는 같아도 user-sim 발화가 다르다(t7328 t1 손님은 *"I'd rather not share my full address here"* 로 그쳤고 에이전트는 원장 값을 유지했다) ⇒ 결정론적 회귀로 단정 불가, **관측으로만 기록** |

---

## 7. 처방 후보 (제안까지만 · 실행 0)

1. **ⓐ 수리 (우리-층·CONFIRMED)** — `_arg_empty_deny` 의 이름 해석을 `_eff_tool_name`(접미 제거) 대신
   `_exact_tool_name`(환경의 원래 이름) 계열로 맞추고, discoverable 도구의 `parameters.required` 를
   어디서 얻을지 결정한다. 이 궤적에는 **재료가 이미 대화에 있다** — unlock 출력 [4]/[6] 이
   `- address: string (required) - …` 를 축자로 인쇄한다. 도메인 리터럴 0 으로 가능.
   측정 의무: `address:""` 7건 차단 → **reward 는 여전히 0**(§5ⓐ) 이므로 단독 승격 금지, 2·3과 묶어야 함.
2. **ⓑ 절충 (레버 ±)** — `write_arg_grounding` 의 give 항목과 `tool_signatures` 를 **한 축으로 병합**한다
   ([[72]] 동의어 병합): 서명이 `arguments` 를 금지한 도구에 대해서는 WAG 의 give 표적을 조건부로 끄고
   (조건 = "이 도구의 서명이 그 키를 선언하지 않았다" = 도메인-일반 닫힌 술어, [[70]] 준수),
   placeholder 차단은 **SIGNATURE 피드백 본문 안에서** 함께 말한다(*"인자는 답변 본문에 적어라"* 는
   이미 `t2_signature.FEEDBACK` 에 있다). 부작용 계측 = give 성공률 ↔ placeholder 통과 수.
3. **날짜/형식 축 (미보유 레버)** — `issue_noticed_date` 처럼 **env 스키마가 형식을 선언한 필드**에서
   선언 형식과 불일치하는 값을 거부하는 도메인-일반 술어. 재료는 unlock 출력의 `format MM/DD/YYYY`
   축자이고, 엔진은 형식 검사만·값 생성 0([[62]] 최소 결정론). ⛔ 격리 프로브 선행 필수 — 격리에서
   모델이 스스로 닫는지부터 재야 한다(현재 [41] 에서 스스로 맞혔다가 뒤집은 기록이 있어 **부하 축일
   가능성**이 높다 ⇒ 레버는 전달/거부이지 계산이 아니다).
4. **provisional credit 상한 축** — `get_user_dispute_history_7291` 미호출이 병목. `T2_WRITE_EVIDENCE`
   의 dispute 스펙에 dispute-history 선행-read 요구를 **추가**하는 형태가 자연스럽지만, 근거는
   반드시 정책 축자(`Provisional Credit Eligibility Guidelines (Internal)` §4)여야 하고 gold 경유
   금지([[23]]). 역시 격리 선행.

---

## 8. 결론 (원인 확정)

| trial | 결정 지점 | 1차 귀속 | 2차 |
|---|---|---|---|
| 0 | turn 23 — 모델이 emit 한 `give_discoverable_user_tool` 을 우리 층이 WAG→SIGNATURE 로 차단, 이후 turn 29·43 재차단, give 실행 0 | **our_layer** (`t2_gate_patch.py:9287-9291` 체인 선점 + A2 `write_arg_grounding[2]` ↔ `tool_signatures` 모순) | model (deny 를 받고도 `arguments` 를 빼고 재발행하지 못함 · 스키마 전문을 두 번 보고도 3-인자 dispute 호출) |
| 1 | msg [45] (`issue_noticed_date` 항복) · msg [57] (`address` 삭제) · [3]→[49] 사이 dispute-history 미읽음 | **model** | user_sim (두 결정 모두 손님의 명시적 요구가 트리거) + our_layer (`T2_ARG_EMPTY` 死배선이 빈 `address` 7건을 무경고 통과시킴) |
