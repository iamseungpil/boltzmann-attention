# task_040 — bank_t7346 halfB per-step 포렌식 **v2 재검** (런 2026-08-22 · 보고 2026-08-23)

> 자료 = 전부 로컬. `sim_results/bank_t7346_halfB_20260822.results.json.gz` + 같은 tag `.log.gz`
> (`[sim=task_040#s626729]` = trial 0, `[sim=task_040#s373753]` = trial 1).
> 대조 = `bank_t7328_halfB_20260819r2`(+ `.fb.jsonl.gz` 사이드카) · `bank_t7336_halfB_20260821b`.
> **코드 인용은 전부 런 sha `ee18d797` 프리즈본**(`git show ee18d797:<path>`) — 워킹트리는
> 2026-08-23 14:11 에 다시 편집됐고 마스터 §8 함정 ①이 그 오프셋 오류를 박제했다.
> 변이 집합은 `t2_forensic.mutation_diff` 정본(커밋 `73efa6f7` 이후 `deny_kind` 수정 포함)만 썼다.
> gold(`reward_info`)는 **진단용으로만**([[23]]). 수리·코드 수정 **0**.

---

## 0. 한 줄 요약 (v1 대비 무엇이 바뀌었나)

| | v1(`tasks__20260822/TASK_040.md`) | **v2(이 문서)** |
|---|---|---|
| t0 1차 귀속 | our_layer (모순 지시가 give 를 0회로) | **our_layer(기전) · 인과는 PLAUSIBLE** — 마스터의 REFUTED 를 수용하되 **왜 t7328 은 회복했고 t7346 은 못 했는지**를 기전으로 특정했다(§6ⓑ) |
| 축 D 진술 | "두 선언이 모순" | **모순 + 순서 + *발화 조건*까지 확정**: WAG-on-give 는 handoff 값이 **접지 실패일 때만** 뜨고, 그때 SIGNATURE 를 선점한다 ⇒ 모델은 *"인자를 채워 다시 넘겨라"* 를 먼저 2회 받고 *"인자를 빼라"* 를 나중에 받는다. t7328(같은 seed)은 값이 접지돼 WAG 가 **0회**였고 SIGNATURE 만 깨끗이 3~5회 받아 **5/5 회복**했다 |
| 축 E | "final-word 재생성 실패로 위반 호출 커밋" | **확정 + 부호 반전**: 그 탈출구가 t1 의 give 를 통과시켰고, 그것이 **8건 실행 전체의 전제**였다. 런 전체에서 `arguments` 를 실은 채 커밋된 give 2건은 **둘 다 final-word 경유**다(040#1·063#0) |
| `T2_ARG_EMPTY` 死배선 | CONFIRMED | **CONFIRMED 유지** — 프리즈본 함수 원문을 잘라 재실행해 3/3 케이스 재현(§6ⓐ). 라이브 부정통제도 붙였다(런 전체 발화 2회, 둘 다 **최상위** 도구) |
| 레버 발화표 | 수치 다수 오기 | **전수 재계수**(WRITE_EVIDENCE t0 18→**7**, t1 7→**1**; SIGNATURE t1 6→**4**) — 마스터 §8 함정 ⑤ 동형 |
| 중재 정지 41/95 | (미측정) | **가름 완료**: 41건 중 우리 deny 와 **같은 턴**에 있던 것은 **12건뿐**(t0 10 · t1 2). 나머지 29건은 산문 루프의 **증상**. §5 |

**결론 한 줄** — t0 는 *값-획득 경로(give)가 우리 5회 deny 로 끝내 안 열려* 8건 전량 MISSING,
t1 는 *그 경로가 우리 버그(final-word 재생성 실패) 덕에 열려* 8건 전량 실행됐으나 **필드 4축**에서 죽었다.
⇒ **이 태스크에서 give 축을 고쳐도 reward 는 0 이다**(t1 이 그 반사실을 이미 보여준다).

---

## 1. 채점 축 (먼저 확인 · C583ⓖ)

```
reward_info.reward           = 0.0   (양 trial)
reward_info.reward_basis     = ["DB"]
reward_info.reward_breakdown = {"DB": 0.0}
reward_info.db_check         = {"db_match": false, "db_reward": 0.0}
reward_info.info["action"]   = None          ← ACTION 축 아님
```

⇒ **DB-해시 축**. `action_checks` 15행은 진단 보조로만 읽는다([[69]]).
`t2_forensic.gold_mutations` 는 `unlock`/`give`(=GRANTS)를 **DB 무변**으로 제외하므로,
gold 변이 집합 = `log_verification` 1 + `file_credit_card_transaction_dispute_4829` **8** = 9건.

> ★이 사실 하나가 §6ⓑ 의 [[70]] 판정을 결정한다: `give_discoverable_user_tool` 의 `arguments`
> 여분 키는 **점수에 닿지 않는다**(gold action 040_3 `action_match=false` 는 성적이 아니다).
> 그런데 그 여분 키를 막는 레버의 부작용(= give 미실행)은 **점수 전체에 닿는다**.

---

## 2. 변이 집합 (`t2_forensic.mutation_diff` 정본 · 손 비교기 0)

### trial 0 — seed 626729 · `user_stop` · 61 msg · 1,474s

| 축 | 수 | 내용 |
|---|---|---|
| matched | 1 | `log_verification` (완전 일치 · `address` 포함) |
| **MISSING** | **8** | gold dispute 전량 (`txn_25e23705f61f` `txn_fd4c3871654e` `txn_a1b2c3d4e503` `txn_a1b2c3d4e510` `txn_a1b2c3d4e508` `txn_a1b2c3d4e513` `txn_bc21d98cc4e4` `txn_3ef1a3e9bf56`) |
| WRONGARG / EXTRA / DUP | 0 / 0 / 0 | — |
| BLOCKED | 2 | 둘 다 `deny=env` — msg41 `unexpected keyword argument 'date_noticed'` · msg43 `missing 12 required positional arguments` |

### trial 1 — seed 373753 · `user_stop` · 119 msg · 3,816s

| 축 | 수 | 내용 |
|---|---|---|
| matched | 1 | `log_verification` |
| **MISSING** | **8** | gold 8건(인자 불일치로 어느 키도 안 맞음) |
| **WRONGARG** | **8** | 실행된 dispute 8건 전부 — `transaction_id` 는 **8/8 정답** |
| EXTRA / DUP / BLOCKED | 0 / 0 / 0 | env 거부 0 |

**WRONGARG 필드별 대조**(gold ↔ 보낸 값 · 일치 필드 생략)

| msg | transaction_id | gold aid | 어긋난 필드 |
|---|---|---|---|
| 49 | `txn_a1b2c3d4e503` (AA) | 040_9 | `issue_noticed_date` `11/14/2025`↔`noticed today (date unknown)` · `eligible…` False↔True |
| 57 | `txn_a1b2c3d4e510` (Best Buy 개인) | 040_10 | `address` `4532 Magnolia Lane…`↔`""` · `issue_noticed_date` · `eligible…` |
| 61 | `txn_a1b2c3d4e508` (PECO) | 040_11 | `address ""` · `issue_noticed_date` · `eligible…` |
| 63 | `txn_a1b2c3d4e513` (Spotify) | 040_12 | `address ""` · `issue_noticed_date` · `eligible…` |
| 99 | `txn_fd4c3871654e` (Uline) | 040_8 | `address ""` · `issue_noticed_date` |
| 103 | `txn_25e23705f61f` (Grainger) | 040_7 | `address ""` · `issue_noticed_date` |
| 105 | `txn_bc21d98cc4e4` (Best Buy 사업) | 040_13 | `address ""` · `issue_noticed_date` · `eligible…` |
| 115 | `txn_3ef1a3e9bf56` (Comcast) | 040_14 | `address ""` · `issue_noticed_date` · `eligible…` · `resolution_requested` full↔partial · `partial_refund_amount` 없음↔`189.99` |

**정답이었던 것**: `transaction_id` 8/8 · `dispute_reason` 8/8 · `purchase_date` 8/8 ·
`card_last_4_digits` **8/8**(개인 1652 / 사업 0581 를 정확히 갈랐다 — 손님이 msg84 에서
*"Business Gold Rewards card (last 4 **1234**)"* 라고 **틀린 값을 주장했는데도** 도구 출력을 따랐다.
[[25]] 원장-우선의 **성공** 사례다) · `contacted_merchant` 8/8 · `card_action` 8/8 · 신원 4필드 8/8.

⇒ **열거·매칭·enum 매핑·원장 우선순위는 다 됐고, 값 4칸에서 죽었다.**

⚠ **단독 수리의 매수 = 0**: `address` 7건을 고쳐도 같은 7행이 `issue_noticed_date` 로 독립 오답이다.
DB 해시는 8행 전부가 맞아야 바뀐다.

---

## 3. step-by-step — 결정 지점 (축자)

### 3-0. 분기점 = **msg [13]** (msg 0–12 은 두 trial 이 문면까지 동일)

msg[10] 의 user-sim 표현만 다르고(굵은 글씨 유무) 도구 궤적은 [12] 까지 같다
(`KB_search_bm25`+`unlock` → `verify_identity` 오호출 → 이름 요청 → `get_user_information_by_name`).
갈린 자리는 그 직후 assistant 턴이다.

- **t1 [13]** — 검증을 미루고 **손님 도구 안내**로 간다:
  > "I need you to run a tool to get the last 4 digits … ```call_discoverable_tool("get_card_last_4_digits", {"credit_card_account_id": "your_account_id"})```"

  → [15] 에서 실제로 호출: `TOOLCALL give_discoverable_user_tool {"discoverable_tool_name": "get_card_last_4_digits", "arguments": "{\"credit_card_account_id\": \"<your_personal_gold_account_id>\"}"}`
  → [16] `Tool given to user: get_card_last_4_digits` **성공**.
- **t0 [13]** — 정석대로 `verify_identity` 를 계속한다(→[18] VERIFIED →[21] `log_verification`).
  give 는 **turn 23** 에서야 시도되고, 그때는 아직 `get_credit_card_accounts_by_user` 를 안 했다.

⇒ **분기의 실질은 "give 를 계좌-read 전에 던졌나 후에 던졌나"** 이고, 그것이 §6ⓑ 의 발화 조건을 결정한다.

### 3-1. t0 turn 23 — 우리가 **강제로 행동시키고, 그 행동을 우리가 막았다**

로그(같은 turn 안의 3회 재생성 · 축자):

```
[T2_VALUE_ACQUIRE] give-surfacing → nudge (regen)
[T2_FORCE_ACTION] say-don't-do → tool_choice=required 재생성
[T2_LEVER] T2_WRITE_EVIDENCE sim=task_040#s626729
[T2_WRITE_ARG_GROUND] deny tool=give_discoverable_user_tool inner=
[T2_MATERIAL_GATE] stop=other_lever(wev) turn=23
[T2_TOOL_SIGNATURE] would-deny tool=give_discoverable_user_tool but preempted-by=wev
  … (동일 3줄 1회 더) …
[T2_FORCE_ACTION] say-don't-do → tool_choice=required 재생성
[T2_MATERIAL_GATE] stop=resolve_cap(정체 3회) turn=23
[T2_TOOL_SIGNATURE] deny tool=give_discoverable_user_tool
[T2_CLAIMPROV] regen tool_calls=[]
```

읽는 법:
1. 모델의 **자발 출력은 산문**이었다(`say-don't-do`). 우리 `T2_FORCE_ACTION`
   (`t2_gate_patch.py:9083-9090`)이 `tool_choice=required` 로 **행동을 강제**했다.
2. 강제된 행동 = `give(…, arguments={"credit_card_account_id": "your_personal_gold_card_account_id"})`.
   그 시점 계좌 ID 는 **아직 문맥에 없다**(`get_credit_card_accounts_by_user` 는 msg[25]).
   ⇒ WAG deny 는 **선언대로 정당**하다(placeholder 차단 = P9 의 목적).
3. 그 정당한 deny 가 SIGNATURE 를 **선점**한다(`preempted-by=wev`).
4. 3회째에 `resolve_cap` 이 래치되고 SIGNATURE 가 처음으로 발화, 마지막 `CLAIMPROV` 재생성이
   `tool_calls=[]` 를 내놓아 **커밋된 msg[23] 은 산문**이 된다.

커밋된 msg[23] 축자(존재하지 않는 함수명을 손님에게 안내한다):
> "To get the last four digits of your **Personal Gold Card**, please use the following tool:
> ```call_discoverable_tool("get_card_last_4_digits", {"credit_card_account_id": "your_personal_gold_card_account_id"})```"

(`call_discoverable_tool` 은 없는 이름이다. env 가 [31] 에서 정확한 이름을 인쇄한다:
*"The agent must first use `give_discoverable_user_tool` …"*. 우리 `discoverable_name_check`/
`T2_UNKNOWN_NAME_BL` 은 **손님에게 말한 이름**을 안 본다 — 다만 손님이 알아서 옳은 이름
`call_discoverable_user_tool` 로 실행했으므로(msg 30·32·48) **이건 인과가 아니다**.)

### 3-2. t0 turn 29~53 — give 5회 시도 · **5회 전부 차단** · dispute 6회 시도 · **6회 전부 차단**

계좌 ID 를 읽은 뒤([26])에도 같은 일이 반복된다. 프리즈본 로그의 deny 전수(정규식 계수):

| trial | 표적 | 레버 | 횟수 |
|---|---|---|---|
| t0 | `give_discoverable_user_tool` | `T2_WRITE_ARG_GROUND` deny | **2** |
| t0 | `give_discoverable_user_tool` | `T2_TOOL_SIGNATURE` deny | **3** (+ would-deny 2 = 위 2건과 같은 호출) |
| t0 | `call_discoverable_agent_tool`(inner=`file_credit_card_transaction_dispute_4829`) | `T2_WRITE_EVIDENCE` deny | **6** (turn 29×2·41·43·45·49, 그 뒤 `deny cap 8 reached`) |
| t1 | `give_discoverable_user_tool` | WAG 2 + SIGNATURE 1 + **final-word 1** | 4 |
| t1 | dispute | `T2_WRITE_EVIDENCE` deny | 1 |

**t0 에서 give 는 0회 실행**됐다(`T2_RESOLVE_CAP 리셋(실행)` 목록에 끝까지 안 나온다:
`KB_search_bm25`·`unlock`·`get_user_information_by_name`·`verify_identity`·
`get_credit_card_accounts_by_user`·`get_credit_card_transactions_by_user` 뿐).

⇒ **last-4 가 영영 안 들어왔고**, 그래서 WEV 스펙 9(`id_key: card_last_4_digits`)가
dispute 를 6회 정당하게 막았다. env 까지 새어 나간 2건은 라운드 cap 을 넘어간 **불완전 호출**이다:

> [41] `TOOLCALL call_discoverable_agent_tool {"agent_tool_name":"file_credit_card_transaction_dispute_4829","arguments":"{\"transaction_id\": \"txn_a1b2c3d4e503\", \"dispute_reason\": \"Fraud/unauthorized\", \"date_noticed\": \"11/14/2025\"}"}`
> [42] tool `Error: Invalid arguments: … got an unexpected keyword argument 'date_noticed'`
> [43] (3→2 인자로 재시도) → [44] `missing 12 required positional arguments: 'card_action', 'card_last_4_digits', …`

여기서 **필요한 정보는 전부 문맥에 실재했다** — 파라미터 명세 전문이 [4]·[6] 에 두 번,
`- issue_noticed_date: string (required) - … format MM/DD/YYYY` 포함. 즉 읽기 결손이 아니라
**스키마 준수 결손**(model).

이후 [45]·[47]·[51]·[55] 에서 **같은 산문 안내 4회 반복**, 손님이 세 번 실패를 보고
([34] *"Can you enable/share that tool on your side…?"* · [46] *"I'm still not able to run that tool on my end"* ·
[50] *"it doesn't look like it's actually enabled for me yet"*), [57] 이관 제안 → [58]/[60] `###TRANSFER###`.
`[T2_TRANSFER_LEAVES_STEPS] surface ledger gap qty=10 executed=1` — 잔여를 정확히 셌다(진단 정확·행동 무변).

### 3-3. t1 — give 가 통과한 **유일한 이유**: final-word 재생성 실패(축 E)

t1 turn 15 로그(축자·순서 그대로):

```
[T2_TOOL_SIGNATURE] deny tool=give_discoverable_user_tool
[T2_USER_TOOL_NOTE] pre-give note: get_card_last_4_digits
[T2_VALUE_ACQUIRE] give-surfacing → nudge (regen)
[T2_WRITE_ARG_GROUND] deny tool=give_discoverable_user_tool inner=
[T2_MATERIAL_GATE] stop=other_lever(wev) turn=15
[T2_TOOL_SIGNATURE] would-deny tool=give_discoverable_user_tool but preempted-by=wev
  … (WAG deny 1회 더) …
[T2_CLAIMPROV] regen tool_calls=[]
[T2_UNINSTRUCTABLE] regen: instruction with nothing given
[T2_LEVER] T2_TOOL_SIGNATURE sim=task_040#s373753 final
[T2_TOOL_SIGNATURE] final-word deny tool=give_discoverable_user_tool (try 1)
[T2_RESOLVE_CAP] 리셋(실행): 새 실행 ['give_discoverable_user_tool'] (정체 1회 → 0)
[T2_GIVE_EXEC] nudge idle=['get_card_last_4_digits']
```

`final-word` 는 `t2_gate_patch.py:12281-12315` 의 `while _sf_tries < 2` 루프다.
`_new9 = _ap_regen(…)`(12309) 가 `None` 을 돌려주면 `break`(12310) 하고 **`am` 이 그대로 반환**된다
⇒ **위반 호출이 커밋**된다. t1 이 8건을 실행할 수 있었던 것은 오직 이 탈출구 덕이다.

그리고 **env 는 `arguments` 를 받아들이고 되돌려준다** — [16] 축자:
> "Tool given to user: get_card_last_4_digits … **Arguments:** { "credit_card_account_id": "<your_personal_gold_account_id>" }
> The user can now execute this action by calling `call_discoverable_user_tool` … and the same arguments."

게다가 손님은 그 placeholder 를 **무시하고 실제 값으로 실행**해 성공한다:
> [~45] user `TOOLCALL call_discoverable_user_tool {"discoverable_tool_name":"get_card_last_4_digits","arguments":"{\"credit_card_account_id\":\"cc_01f21c9970_gold\"}"}`
> tool `Executed: get_card_last_4_digits … Last 4 digits of card: 1652`

⇒ **placeholder handoff 는 실무적으로 무해했다.** WAG-on-give(P9)가 사려던 것의 실효가
이 궤적에서는 0 이고, 그 대가(give 봉쇄)는 t0 에서 전부였다. §6ⓑ 의 [[70]] 절충 근거.

### 3-4. t1 결손 ① `issue_noticed_date` 8/8 — 값과 **형식 명세**가 다 있는데 손님 표현에 항복

값의 실재는 세 겹:
- 우리 배달: `[T2_SEARCH_AGENT] … now=2025-11-14 turn=2 (스펙 있음 · 원값 '11/14/2025' · 대화텍스트 1)`
- 도구 출력: `[26] tool The current time is 2025-11-14 03:40:00 EST` → [27] 에서 그대로 `log_verification` 에 사용
- 거래 원장: `11/14/2025` 가 msg 48·54·60·82·90·114 에 실재 · `format MM/DD/YYYY` 는 msg 4·6·34·36 에 4회 인쇄

에이전트는 [41] 에서 **스스로 옳게 판단했다**:
> "Since you mentioned that you noticed the issue today, I will use the current date for the
> `issue_noticed_date`. … **Issue noticed date**: Today's date (2025-11-14)"

손님이 두 턴에 걸쳐 밀어붙인다:
> [42] "I **don't want you to use today's actual date** (I genuinely don't remember it)"
> [44] "for the **issue noticed date**, please keep it as **"noticed today (date unknown)"** on my end
> — I don't want it recorded as a specific date I didn't provide."

**결정 지점 = msg [45]**:
> [45] "Absolutely, Kwame. We can keep the "issue noticed date" as **"noticed today (date unknown)"** for your records."

이후 8/8 에 그 문자열이 실린다. 태스크 대본은 손님에게 *"오늘 알았고 오늘 날짜를 잊었다"* 까지만
시킨다 — *"이 문자열 그대로 기록하라"* 는 user-sim 의 증폭이다. [[21]] 에 따라 면책 아님:
**env 스키마가 형식을 선언한 필드**에 손님 문면을 그대로 기입한 것은 에이전트 측 흡수 실패.

### 3-5. t1 결손 ② `address = ""` 7/8 — **원장에 있고 이미 통과시킨 값**을 지웠다

첫 건 [49] 은 정상: `"address": "4532 Magnolia Lane, Philadelphia, PA 19103"`.
출처는 [12] `get_user_information_by_name` 출력 축자이고, 같은 값으로 `log_verification` 이
**MATCHED** 됐다(§2).

전환점은 손님의 **조건부** 요구다:
> [56] "**Address:** I didn't provide an address, so please **don't use "4532 Magnolia Lane…"**
> **unless you're seeing that on file from your system and can confirm it's what you have on record.**"

조건(`on file from your system`)은 **충족돼 있었다**. 그런데 바로 다음 호출에서 빈 문자열이 된다
(같은 메시지의 산문에는 address 언급 자체가 사라진다):
> [57] `… "email": "midnight_runner_42@gmail.com", **"address": ""**, "contacted_merchant": true, …`

이후 msg 61·63·99·103·105·115 까지 **7건 연속**. **결정 지점 = msg [57]**.
우리 층은 여기서 한마디도 안 했다 — §6ⓐ.

### 3-6. t1 결손 ③ `eligible_for_provisional_credit` 6/8 — **닫힌 규칙을 두 번 배달받고 0회 적용**

정책 문서 `Provisional Credit Eligibility Guidelines (Internal)`
(`doc_credit_cards_credit_cards_(general)_015`)가 **두 번** 문맥에 들어왔다 —
msg[3](우리 `T2_SEARCH_AGENT` 가 credit_cards 110편 배달) · msg[34](모델 자신의 `KB_search_bm25`).
축자:
> "## NOT Eligible Scenarios … Dispute reason is 'incorrect_amount', 'goods_services_not_as_described',
> 'canceled_subscription_still_charging', or 'refund_never_processed'"
> "4. Previous Disputes: The customer has not filed more than 2 disputes in the past 12 months"

그리고 같은 배달에 도구 이름도 있다 — `Checking User Dispute History (Internal)`:
> "To retrieve a user's credit card dispute history, use the **get_user_dispute_history_7291** tool."

**`get_user_dispute_history_7291` 호출 = 0회**(양 trial · 문자열은 t0 2회 / t1 4회 문맥 실재).

에이전트는 8/8 을 `true` 로 채웠다. 그런데 **추가 read 없이도 닫히는 부분이 4건**이다 —
PECO(`incorrect_amount`) · Spotify(`canceled_subscription_still_charging`) ·
Best Buy 사업(`goods_services_not_as_described`) · Comcast(`refund_never_processed`)는
문서의 `NOT Eligible` 열거에 **낱말 그대로** 있다. 즉 읽은 문서의 **enum 배제조차 안 했다**
([[63]] 빼기 실패의 전형). 나머지 2건(AA·Best Buy 개인)의 False 는 §4 상한(직전 dispute 수)이
필요하고, 그건 `get_user_dispute_history_7291` 없이는 못 닫는다.

### 3-7. t1 결손 ④ msg 115 Comcast — 손님이 **불러준 값**을 그대로 실었다

gold 040_14 = `refund_never_processed` / **`full_refund`** / `partial_refund_amount` 없음.
손님이 직접 지시한다:
> [108] "For **Comcast** … Use: … **Resolution:** partial_refund"
> [110] "**Resolution requested:** partial_refund (for the expected/owed amount you find)"

에이전트는 [115] 에서 `partial_refund` + `189.99` 를 실었다. **그 189.99 는 거래 총액 그대로다**
(`Comcast $189.99 10/30/2025`) — *전액을 부분환불로 청구*하는 자기모순이고, 이건 gold 없이도
닫히는 술어다(§9-4). 대조: PECO 는 gold 도 `partial_refund`/`24.56`(=124.56−100) 인데 맞혔다
⇒ 규칙이 아니라 손님의 낱말을 봤다.

### 3-8. t1 msg 66~113 — **20턴 산문 확인 루프**(정지 41건의 대부분이 여기서 나온다)

손님이 *"filed successfully + case ID"* 확인을 6회 요구하고([68][72][74][76][88][92][98][112]),
에이전트는 [71][73][75][77][83][87][91][93][97][109][111] 에서 **세부만 되풀이**한다.
[78] 에서 손님이 `###TRANSFER###` 를 던졌는데도 이어졌다. 이 구간에 우리 deny 는 **1건뿐**이다(§5).

---

## 4. 레버 발화표 (이 sim 줄만 · **전수 재계수** · v1 표 정정)

| 레버 | t0 | t1 | 판정 |
|---|---|---|---|
| `T2_SG_DOCS` | 0 | 0 | **미발화**(런 ON 이지만 이 태스크에 apy/isolate 표적 없음 — 정당한 침묵) |
| `T2_PIN_READ` | 0 | 0 | 미발화 |
| `T2_DEMANDED_STEP` | 0 | 0 | 미발화 |
| `T2_CLAIMPROV` | **30** | **23** | 발화 — `kind='give' … 원장 밖 — 강등` 정상. 단 t0 turn 23·29 의 마지막 재생성이 `regen tool_calls=[]` 로 **행동턴을 산문턴으로 확정**했다 |
| `T2_FOLLOWUP` | 0 | 0 | 미발화 |
| `T2_SEARCH_AGENT` | **16** | **18** | 발화 — `credit_cards` 110편 + `business_credit_cards` 82편 배달 후 *"모두 처리됨 — 침묵"*. **필요 문서(provisional 규정·dispute-history 도구명·현재 날짜)를 전부 배달했다** ⇒ 배달 성공·소비 실패 |
| `FAB_STRIP` | 0 | 0 | 미발화 |
| `T2_ARG_PRODUCERS` | 0 | 0 | **미발화** (`arg_producers` 는 `card_last_4_digits→get_card_last_4_digits` 를 선언하고 있는데 이 궤적의 핵심 병목이 바로 그것이다 — 술어 미성립 사유 미확인 ⇒ UNPROVEN) |
| READ-FIRST(`T2_WRITE_EVIDENCE`) | **7** (deny 6) | **1** (deny 1) | 발화 — v1 의 18/7 은 오기. t0 의 6 deny 는 last-4 부재를 정확히 잡았다(정당) |
| `T2_REQUIRE_DOC_DELIVER` | 0 | 0 | 미발화 |
| `T2_SEARCH_REARM` | 2 | 0 | 발화(t0 turn 37 델타 6,973자) — 효과 0 |
| `T2_WRITE_ARG_GROUND` | **2 (deny)** | **2 (deny)** | 발화 — **선언대로는 정당**(placeholder 차단), 그러나 SIGNATURE 를 선점하고 **반대 방향**을 지시(§6ⓑ) |
| `T2_TOOL_SIGNATURE` | **5** (deny 3 + would 2) | **4** (deny 1 + would 2 + **final 1**) | 발화 — 정책 축자 근거 있음. t1 의 final 만이 통과를 만들었다 |
| `T2_VALUE_ACQUIRE` | 6 | 6 | 발화 — *"give 를 넘겨라"*. **자기 층의 give deny 와 정면 상쇄** |
| `T2_FORCE_ACTION` | **11** | **9** | 발화 — turn 23 의 give 3회 emit 을 **강제한 당사자** |
| `T2_USER_TOOL_NOTE` | 1 | 1 | 발화(pre-give note) |
| `T2_GIVE_EXEC` | 0 | 1 | t1 만(give 가 실행됐을 때만 뜬다) |
| `T2_ARG_EMPTY` | **0** | **0** | **미발화(구조적 死)** — §6ⓐ |
| `T2_WRITE_ARG_ENUM` | 0 | 0 | 미발화 — 선언이 `open_bank_account.account_class` **하나뿐**이라 dispute enum 3종은 사각지대 |
| `T2_REF_VERIFY` | 0 | 0 | 미발화 — **정당**(transaction_id 8/8 정답) |
| `T2_HAVE_VALUE` / `T2_UNKNOWN_BOOL` | 0 | 0 | 미발화 |
| `T2_MATERIAL_GATE` stop | **35** | **41** | `resolve_cap(정체 3회)` 래치 + `other_lever` — §5 |
| `T2_TERM_GRANT(_USERDEMAND)` | 1+1 | 1+1 | 발화(`[T2_LEVER]` 형태) |
| `T2_TRANSFER_LEAVES_STEPS` | `qty=10 executed=1` | `qty=10 executed=4` | 발화 — 진단 정확·행동 무변 |

> **[[55]] 0단계(마크≠전달) 확인**: t0 turn 29·43 의 SIGNATURE 는
> `[T2_STACK] window folded fb tag=signature (…) — deny stays, **body kept (R9)**` 로 찍혔고,
> `T2_KEEP_DENY_BODY=1` 이 PIN 에 있다(`t2_gate_patch.py:10016-10021`) ⇒ **원본 본문이 실제로 나갔다**.
> 즉 모델은 *"`arguments` 를 빼고 다시 발행하라"* 는 문면을 **3회** 받았다.
> ⚠한계: 재생성 버퍼는 비커밋(`state.messages` 밖)이라 t7346 에는 `fb` 사이드카가 없다 —
> 문면 축자는 t7328 사이드카와 A2 선언에서만 확인했다.

---

## 5. 중재 정지 41/95 — **원인인가 증상인가** (오늘 실측에 대한 답)

`x492_arbitration_ledger.py`(정본 술어 = `t2_liveness.arbitration`)의 혐의 버킷
= *"MATERIAL_GATE 정지가 **산문턴**(tool_calls 없음)에 떨어졌고 그 뒤로 배달이 없었다"*.
런 전체 95건 중 **040 이 41건**(t1 22 · t0 19) = **43%**.

정본 스크립트가 이미 적은 대조:
```
전체         PASS 1.54/sim ↔ FAIL 2.78/sim = 1.81x
040 제외 →   PASS 1.54/sim ↔ FAIL 1.36/sim = 0.88x
분기-이전 창 msg<=12   0.58x · msg<=20   0.39x · msg<=30   0.72x
task_040 창100F = 28.03  (2위 073 15.38/7.14 · 3위 016 7.14)
```
⇒ **런 수준의 "정지→실패" 신호는 040 하나가 만든 것**이고, 040 을 빼면 신호가 없다(0.88x).

그래서 040 안에서 다시 갈랐다. **deny 는 코드 순서상 같은 생성의 `MATERIAL_GATE stop` 바로 앞
줄이므로**(로그 실측), 각 deny 의 turn = 그 다음 stop 줄의 turn 으로 확정할 수 있다(±윈도 추정 금지):

| trial | 혐의 정지 | 그중 **같은 턴에 우리 deny 가 있던 것** | 위치 |
|---|---|---|---|
| t0 626729 | 19 | **10** | turn 23(4·WAG+SIG) · turn 29(4·WEV+SIG) · turn 45(2·WEV) |
| t1 373753 | 22 | **2** | turn 91(2·WEV) |
| 합 | **41** | **12** | |

나머지 29건은 turn 9·37·47·51·53(t0) / turn 37~111 의 산문 확인 루프(t1) 로,
**우리가 아무것도 안 막은 자리**다.

**답**: 41건 중 **12건만 "우리가 만든 산문턴"** 이고, 그중에서도 reward 에 닿을 수 있었던 것은
t0 turn 23·29 의 **give deny 5건**뿐이다(WEV 의 dispute deny 는 last-4 부재를 잡은 정당한 차단이고,
막지 않았어도 그 호출들은 `card_last_4_digits` 를 못 채운다). **나머지 29건은 증상**이다 —
040 은 이 런에서 가장 긴 궤적 쌍(61+119 msg)이고, t1 의 20턴 산문 루프가 정지를 쌓았다.

⇒ *"중재 정지가 040 을 죽였다"* 는 **성립하지 않는다**. 성립하는 것은
*"정지 41건 중 5건(give deny)이 값-획득 경로를 닫았고, 그 5건이 t0 의 8건 MISSING 의 필요조건이다"* 이다.

---

## 6. 우리-층 주장 (프리즈본 `ee18d797` 코드 경로 지목 · 등급 명시)

### ⓐ **CONFIRMED** — `T2_ARG_EMPTY` 는 디스패처 write 에 **구조적으로 죽어 있다**

- 코드 경로(프리즈본 줄번호):
  - `scripts/distill/tau2/t2_gate_patch.py:7832` `wd = _arg_empty_deny(self, c, a2, ae_tools)`
  - `:1513` `def _arg_empty_deny(agent, tc, a2=None, applies_to=None)` → `name = _eff_tool_name(tc)`
  - `:2491` `_eff_tool_name` — `call_` 디스패처를 unwrap 한 뒤 **`re.sub(r"_\d+$", "", inner)`**
    ⇒ `file_credit_card_transaction_dispute` (접미 `_4829` 제거)
  - `:1494` `_schema_required(agent, name)` — 캐시 키가 `agent.tools` 의 `openai_schema` **원본 이름**
    ⇒ ⑴이름 불일치(`…_4829` ↔ 접미 제거) ⑵애초에 discoverable 도구는 `agent.tools` 에 없다
    (unlock 출력 축자: *"You can now use this tool by calling `call_discoverable_agent_tool` …"*)
  - `:1519` `if not req: return None` ⇒ **항상 침묵**
- **오프라인 재현**(프리즈본 함수 원문을 그대로 잘라 실행 · 부작용 0 · 재작성 0):

  | 등록 형태 | `_arg_empty_deny` 반환 |
  |---|---|
  | A 디스패처만 등록(**실제 환경**) | `None` |
  | B env 실이름(`…_4829`)으로 등록 | `None` |
  | C 접미 제거 이름으로 등록 | `[ARG-EMPTY] … 'address' …` |

- **라이브 부정통제**: `T2_ARG_EMPTY=1` 이 PIN 에 있고(`run_t7346_…sh:84`), 런 전체(40 sim) 발화는
  **2회뿐이며 둘 다 `get_reward_discrepancies`**(= 최상위 등록 도구). 040 은 `address:""` 를 **7회**
  보냈는데 한 줄도 안 찍혔다.
- **[[70]] 판 것 / 산 것**: 이 수리는 `address` 7건을 막을 수 있지만 **reward 는 여전히 0**
  (같은 7행이 `issue_noticed_date` 로 독립 오답). **단독 승격 금지.**

### ⓑ **CONFIRMED(기전) / PLAUSIBLE(인과)** — give 에 대한 **두 선언의 모순 + 체인 순서 + 발화 조건**

- 선언 1 — `a2/banking_knowledge.specific.json` `tool_signatures`:
  `{"give_discoverable_user_tool": ["discoverable_tool_name"]}`
  → 피드백(`t2_signature.py:25` `FEEDBACK`): *"takes only `discoverable_tool_name` … you also passed
  `arguments`. **Re-issue the call with the declared argument(s) only.**"*
  **출처 정당**([[23]]): 이 런의 `sim['policy']` 축자 —
  *"Use the `give_discoverable_user_tool(discoverable_tool_name)` function"*.
- 선언 2 — 같은 파일 `write_arg_grounding[2]`(`applies_to: give_discoverable_user_tool`,
  `grounded_args: [transaction_id, credit_card_account_id, card_last_4_digits, user_id]`):
  → 피드백 축자: *"… an invented, transcribed-by-hand, or placeholder value (e.g. '[Enter your ...]')
  will fail for them. Look the value up in the records already retrieved (or ask the customer),
  then **hand the tool over again with the actual value.**"*
- 순서를 정하는 코드: `t2_gate_patch.py:9284-9288`
  `_chain = [("gate",…),("prov",…),("eplan",…),("cons",…),("resolve_action",…),("te",…),("wev", wev_fb),…]`
  → `_blocker = next((n, v) …)` → blocker 가 있으면 SIGNATURE 는 `would-deny … preempted-by=%s`(:9305) 로 침묵.
  로그가 그대로 찍는다: `[T2_TOOL_SIGNATURE] would-deny tool=give_discoverable_user_tool but preempted-by=wev`(양 trial).
- **발화 조건이 결정적이다** — WAG 는 `grounded_args` 값이 **비어 있지 않고 접지에 실패할 때만** 뜬다
  (`_write_arg_ground_deny`, `:1449 if not gs: continue`). 즉:
  - handoff 값이 **접지된 실제 ID** → WAG 침묵 → 모델은 **깨끗한 SIGNATURE 문면만** 받는다.
  - handoff 값이 **placeholder** → WAG 가 먼저 2회, *"채워서 다시 넘겨라"* → 그 다음에야 SIGNATURE.
- **[[57]] 부정통제 (같은 런 · 같은 레버)**:

  | sim | WAG-on-give | SIGNATURE deny | final-word | give 커밋 | reward |
  |---|---|---|---|---|---|
  | **task_017#s373753** | 0 | **5** | 0 | **2** | **1.0** |
  | task_055#s373753 | 0 | 3 | 0 | 3 | 0.0 |
  | **task_063#s626729** | **2** | 1 | **2** | **2** | 0.0 |
  | **task_040#s626729** | **2** | **3** | 0 | **0** | 0.0 |
  | task_040#s373753 | 2 | 1 | **1** | 1 | 0.0 |

  ⇒ **deny 는 회복 가능하다**(017 은 5회 deny 후 give 2회 + reward 1.0; 063 은 WAG 2회 뒤에도 통과).
  마스터 §5.3 의 REFUTED 판정은 **옳다**. 040#0 은 이 런에서 give 커밋이 0 인 **유일한** 사례다.
- **교차-런 대조(같은 seed)** — `t7328_halfB` `task_040#s626729`:
  give 를 **5회 emit·5회 커밋**했고 **5건 전부 `{"discoverable_tool_name": "get_card_last_4_digits"}`
  준수형**이며 SIGNATURE deny 도 5회 있었다 ⇒ **5/5 회복**. 그 sim 의 WAG-on-give 는 **0회**다.
  t7336 도 3건 중 2건 준수형. 런 전체(t7346 40 sim)에서 커밋된 give 10건 중 `arguments` 를 실은 것은
  **2건뿐**이고 **둘 다 placeholder**(`<your_personal_gold_account_id>` · `[YOUR_FULL_NAME]`)이며
  **둘 다 final-word 경유**다.
  ⇒ *"모델이 arguments 를 못 뗀다"*(=[[42]] prior-override)는 **과장**이다. 정확한 진술은
  **"handoff 값이 접지 실패인 턴에서, 모델은 반대 지시를 먼저 두 번 받은 뒤 SIGNATURE 를 받는다"**.
- 인과를 PLAUSIBLE 로 남기는 이유: 위 표가 반증 가능성을 열어 둔다(063#0 은 같은 배치에서 통과).
  라이브 A/B 도 격리 프로브도 없다([[18]]·[[62]]).

### ⓒ **CONFIRMED(기전) / 부호 반전** — `final-word` 재생성 실패가 **위반 호출을 커밋**시킨다(축 E)

- 코드 경로: `t2_gate_patch.py:12281-12315`. `while _sf_tries < 2` (:12292) →
  `_new9 = _ap_regen("Error: " + _hit[1], "signature")` (:12309) → `if _new9 is None: break` (:12310)
  → 루프 탈출 후 `return am` — **원래(위반) `am` 이 그대로 나간다**.
- 실물: t1 turn 15 `final-word deny … (try 1)` 직후 `[T2_RESOLVE_CAP] 리셋(실행): 새 실행
  ['give_discoverable_user_tool']` ⇒ 커밋.
- **⚠[[70]] 부호**: 이것은 "버그"이지만, **이 태스크에서는 유일하게 이득을 낸 경로**다.
  t1 의 8건 실행 전체가 이 탈출구 위에 서 있다. **막으면 t1 도 t0 가 된다**(8 WRONGARG → 8 MISSING).
  reward 는 어느 쪽이든 0 이지만, 부호표 없이 이 자리를 "수리"하면 안 된다.

### ⓓ **CONFIRMED(선언 공백)** — dispute 인자축에 우리 선언이 **없다**

- `write_arg_enum` 은 항목 **1개**뿐이고 표적이 `open_bank_account.account_class` 다
  ⇒ `dispute_reason`·`resolution_requested`·`eligible_for_provisional_credit` 은 **선언 밖**.
- `write_evidence_specs` 12항 중 dispute 표적은 **1항**(`id_key: card_last_4_digits`)뿐
  ⇒ provisional-credit 규정도, `get_user_dispute_history_7291` 선행-read 도 **미선언**.
- `arg_empty` 는 A2 3층(`specific`/`gate`/`settings`) 어디에도 **키 자체가 없다**
  ⇒ `ae_tools = None`(`t2_gate_patch.py:6741`) — 필터 없이 도는데도 ⓐ 때문에 침묵한다.
- 이것은 **결손(레버 부재)** 이지 결함이 아니다. 다만 [[72]] 기준으로는 *"1회 오프라인 저작이
  매 런 발견보다 싸다"* 에 해당하고, 근거 축자가 이미 KB 에 있다(§3-6).

### 우리-층이 **아닌** 것

- `issue_noticed_date` — 우리 층에 이 필드를 보는 규칙이 없고(WAG `grounded_args` 미포함·
  있어도 손님 발화가 코퍼스라 통과), 값·형식이 모두 문맥에 있었고 [41] 에서 스스로 맞혔다가
  [45] 에서 뒤집었다 ⇒ **model** (트리거 `user_sim`).
- `address` **값 선택** — 원장 값이 있는데 지운 것은 **model**. 결과를 아무도 안 잡은 것은 ⓐ.
- `eligible_for_provisional_credit` — 문서 2회 배달·도구명 4회 배달·enum 배제는 read 없이 닫힌다
  ⇒ **model**([[63]]).
- 040_14 `partial_refund` — 손님이 불러줬고 값이 거래 총액과 동일해 자기모순인데 실었다
  ⇒ **model**(트리거 `user_sim`).
- t0 의 `call_discoverable_tool` 오명명 6회 — 우리 이름 검사 밖이지만 **손님이 옳은 이름으로 실행**
  했으므로 **인과 아님**(UNPROVEN 아님 · 무해 확정).

---

## 7. 선행 판정과의 대조

| 선행 | 그때 | **이번(v2)** |
|---|---|---|
| `tasks__20260822/TASK_040.md` §0 — t0 1차 = our_layer | 확정 진술 | **부분 유지·격하**. 기전은 그대로(give 5회 차단·0 실행)지만 인과는 PLAUSIBLE. 대신 **왜 t7328 은 회복했나**를 발화 조건으로 특정(§6ⓑ) |
| 같은 문서 §4 레버 발화표 | WEV t0 18 / t1 7 · SIGNATURE t1 6 · TERM_GRANT 1+1 | **오기 정정**: WEV **7/1**(deny 6/1) · SIGNATURE t1 **4** · TERM_GRANT 는 `[T2_LEVER]` 형태 2건. 마스터 §8 함정 ⑤ 동형 |
| 같은 문서 §5ⓐ `T2_ARG_EMPTY` 死 | CONFIRMED | **CONFIRMED 유지**(프리즈본 재현 3/3 + 런 전체 부정통제) |
| 같은 문서 §5ⓑ "모순 쪽이 이긴다" | CONFIRMED(모순)/PLAUSIBLE(인과) | **유지 + 조건 추가**: 모순은 *"접지 실패 턴"* 에서만 발화한다 ⇒ 절충의 조건이 여기서 나온다(§9-2) |
| `FAILURE_MASTER__20260822.md` §5.3 — 040 "우리 층 모순 지시가 give 를 0회로" = **REFUTED** | 반증 근거 = t7328 같은 seed give 5회 실행 · 017#s373753 deny 5회 후 reward 1.0 | **수용**. 이번에 t7328 원자료로 직접 확인했다(5건 전부 준수형 · WAG-on-give 0회). 그리고 **같은 런 안의 부정통제표**(§6ⓑ)를 새로 붙였다 |
| 마스터 §6.3 — `040#0` = **경계(모델 결손)** *"클린 SIGNATURE 3회 받고도 5회 arguments 재발행"* | 경계 7건 중 하나 | **부분 반박**. *"클린 SIGNATURE 3회"* 는 맞다(R9 body kept 확인). 그러나 **5회 중 앞 2회는 클린 SIGNATURE 를 받은 적이 없다**(WAG 가 선점) ⇒ *"3회 받고 3회 불응"* 이 정확한 진술이다. 그리고 같은 모델·같은 seed 가 t7328 에서 **5/5 준수**했으므로 **[[42]] prior-override 라벨은 과하다**. ⇒ 040#0 은 경계 명부에 **그대로 두되 근거 문장을 교체**해야 한다 |
| 마스터 §1 표 — 040 DB 축·t0 8 MISSING·t1 필드 4축 | — | **전부 재현**(§1·§2) |
| 마스터 §3.1/`MASTER_DUP_CORRECTION` — DUP/MATCHED 계기 결함 | t7346 DUP 1건(040 아님) | 040 은 신판에서 **DUP 0 · matched 1 · blocked 2/0** — 정정 전후 동일 |
| 마스터 §2.2 축 E(게이트 우회 채널)에 040#1 등재 | 3건 중 하나 | **유지 + 부호 명시**: 이 우회가 t1 의 8건을 낳았다(§6ⓒ) |
| 마스터 §6.2 `B-4`(비가역 write 열린-enum 게이트) 표적에 040 | 미구현 | **근거 강화** — §3-6 의 `NOT Eligible` 열거는 read 0 으로 닫히는 enum 배제다 |
| 대조 런 t7328 t1(seed 373753) | gold 2건 정확 + 나머지 5건 `eligible` 한 칸만 오답 | **재현**(missing 6·wrong 5·matched 3). t7346 t1 은 같은 자리에 `address`+`issue_noticed_date` 가 추가 ⇒ **필드 축 후퇴**. 단 user-sim 발화가 다르므로(§3-5 의 조건부 요구는 t7328 에 없다) **결정론적 회귀로 단정 불가 — 관측으로만 기록** |
| `STATE_OF_PLAY_2026_08_23.md` §2.2/§3.3 · `ATTRIBUTION_CORRECTION_2026_08_23.md` §4 | 경계 7건에 `040#0` 포함 | **명부 유지**(위 근거 문장 교체 조건부). ⓐ·ⓑ·ⓒ·ⓓ 는 040#0 을 경계에서 빼지 **못한다** — give 가 열렸어도 t1 이 보여주듯 reward 는 0 이다 |

---

## 8. 원인 확정

| trial | 결정 지점 | 1차 귀속 | 2차 | 등급 |
|---|---|---|---|---|
| **0** (626729) | **turn 23** — `T2_FORCE_ACTION` 이 강제한 give 를 `T2_WRITE_ARG_GROUND` 가 2회, `T2_TOOL_SIGNATURE` 가 1회 차단(이후 turn 29·43 재차단·give 실행 0) → last-4 미획득 → WEV 가 dispute 6회 정당 차단 → 8건 전량 MISSING | **our_layer** — `t2_gate_patch.py:9284-9288`(체인 선점) + A2 `write_arg_grounding[2]` ↔ `tool_signatures` 모순 + `:12281-12315` 탈출구 부재 | **model** — 클린 SIGNATURE 3회에 불응 · 스키마 전문 2회 보고도 3·2 인자 dispute 호출 | 기전 CONFIRMED · **reward 인과 PLAUSIBLE**(t1 반사실이 reward 0 이므로 **매수 0**) |
| **1** (373753) | ① **msg [45]** `issue_noticed_date` 항복 ② **msg [57]** `address` 삭제 ③ **[3]/[34] → 전 구간** provisional 규정 미적용·`get_user_dispute_history_7291` 0회 ④ **msg [115]** Comcast partial | **model** | **user_sim**(①②④의 트리거) + **our_layer**(ⓐ `T2_ARG_EMPTY` 死배선이 `address:""` 7건 무경고 통과 · ⓓ 세 enum 축 선언 공백) | ⓐ CONFIRMED · 나머지 model CONFIRMED |

**두 trial 을 합친 진술** — 이 태스크의 실패는 **두 층에 각각 하나씩** 있다.
우리 층은 *값 획득 경로(give)* 를 닫았고(t0), 모델은 *값 4칸* 을 틀렸다(t1).
**어느 한쪽만 고쳐도 reward 는 0 이다** — 이것이 t0/t1 쌍이 주는 유일하고 확실한 정보다.

---

## 9. 처방 후보 (제안까지 · 실행 0 · 전부 [[62]] ①격리 선행 대상)

1. **ⓐ 계기 수리(우리 층·CONFIRMED·격리 불필요)** — `_arg_empty_deny` 의 이름 해석을
   `_eff_tool_name`(접미 제거) 대신 **호출이 실제로 실행하는 이름**으로 맞추고, discoverable 도구의
   `required` 를 어디서 얻을지 정한다. 재료는 대화에 있다 — unlock 출력 [4]/[6] 이
   `- address: string (required) - …` 를 축자 인쇄한다(도메인 리터럴 0).
   **[[70]] 계측 의무**: `address:""` 차단 수 ↔ 정당 write 오차단 수. **단독 승격 금지**(매수 0).
2. **ⓑ 절충(레버 ± · 조건은 도메인-일반 닫힌 술어)** — *끄지 않는다*([[60]]).
   조건 = **"이 도구의 서명 선언이 그 키를 선언하지 않았다"** 일 때 WAG 의 give-표적을 건너뛰고
   SIGNATURE 가 먼저 말하게 한다. placeholder 경고는 **SIGNATURE 피드백 본문 안에서** 함께 말한다
   (*"인자 값은 답변 본문에 적어라"* 는 이미 `t2_signature.FEEDBACK` 에 있다 ⇒ [[64]] 충족).
   **부작용 계측**: give 성공률 ↔ placeholder 통과 수 ↔ **§6ⓒ 부호**(final-word 탈출구를 함께 막으면
   t1 형 궤적이 t0 형으로 떨어진다).
   ⛔조건을 태스크 id·종류로 걸면 [[05]]/[[70]] 위반이다.
3. **형식 축(미보유 레버)** — `issue_noticed_date` 처럼 **env 스키마가 형식을 선언한 필드**에서
   선언 형식과 불일치하는 값을 거부하는 도메인-일반 술어(재료 = `format MM/DD/YYYY` 축자·
   엔진은 형식 검사만·값 생성 0). ⛔격리 선행 필수 — [41] 에서 스스로 맞혔다가 [45] 에서 뒤집은
   기록이 있어 **부하 축일 가능성**이 높다 ⇒ 레버는 전달/거부이지 계산이 아니다([[62]]).
4. **자기모순 술어(닫힘·gold 0)** — `resolution_requested == 'partial_refund'` 인데
   `partial_refund_amount == 원장의 transaction_amount` 이면 거부. 재료는 거래 원장뿐이고 판단 0.
   040_14 를 정확히 겨냥하되 도메인-일반이다. 격리 선행.
5. **provisional 상한 축** — 두 조각으로 나눈다. ⒜ **read 0 으로 닫히는 부분**:
   `NOT Eligible Scenarios` 의 reason enum 4종(축자) → `write_arg_enum` 확장(마스터 `B-4` 와 같은 축).
   ⒝ **read 가 필요한 부분**(직전 dispute 수): `write_evidence_specs` 에
   `get_user_dispute_history_7291` 선행-read 요구 추가. 근거는 반드시 정책 축자
   (`doc_credit_cards_credit_cards_(general)_015` §4 · `Checking User Dispute History (Internal)`),
   **gold 경유 금지**([[23]]). 둘 다 격리 선행.
6. **⛔ 하지 말 것** — §6ⓒ 의 final-word 탈출구를 **단독으로** 막는 것. 이 런에서 그 탈출구는
   040#1·063#0 의 give 를 통과시킨 유일한 경로였고, 부호표 없이 닫으면 **잰 적 없는 것을 판다**.

---

## 10. 이 문서가 **못 사는 것**(정직 절)

1. **reward 반사실 0건.** ⓐ·ⓑ·ⓒ 어느 것도 A/B 로 재지 않았다. t0↔t1 대조는 n=1 이고
   user-sim 발화가 다르다.
2. **재생성 버퍼는 비커밋**이라 t7346 에는 `fb` 사이드카가 없다 — *무엇을 실제로 보냈는가* 는
   로그 마크와 A2 선언, 그리고 t7328 사이드카로만 확인했다(마스터 §7.4ⓑ 와 같은 한계).
3. **격리 프로브 0건.** §9 의 어떤 처방도 [[62]] ①을 거치지 않았다.
4. **§6ⓑ 의 "발화 조건" 설명은 t7328 의 give 인자를 직접 못 본다** — 커밋된 5건이 준수형이고
   WAG-on-give 가 0회였다는 **두 관측**에서 역추론한 것이다.
5. **`T2_ARG_PRODUCERS` 침묵의 사유는 미확인**(선언은 `card_last_4_digits` 를 갖고 있는데
   이 궤적의 병목이 바로 그것이다) — UNPROVEN 으로 남긴다.
