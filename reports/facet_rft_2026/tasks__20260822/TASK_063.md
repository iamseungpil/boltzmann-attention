# t7346 task_063 per-step 포렌식 — 2026-08-23

- 런: `bank_t7346_halfB_20260822`(results/log 전부 로컬 gz · SSH 0 · git 무접촉 · 커밋 0).
- sim 2개: **trial 0 = `task_063#s626729` reward 0.0**(67 msgs · `user_stop` · 1538.7s) / **trial 1 = `task_063#s373753` reward 0.0**(62 msgs · `user_stop` · 963.4s). **양 trial 실패**.
- 로그 전수: `[sim=task_063#s626729]` **457 라인** / `[sim=task_063#s373753]` **367 라인**.
- 변이 = 정본 `t2_forensic.mutation_diff` 만 사용(손 비교기 0 · C583ⓐ). 인용은 전부 축자.
- 엔진: `git_commit=fc0055dc4e0a3162…` · agent `Qwen2.5-32B-Instruct-GPTQ-Int8` · user-sim `openrouter/openai/gpt-5.2`(temp 0 · reasoning low).
- 러너 = `run_t7346_overnight_stage1_20260822.sh` (PIN + `ON="T2_ARG_DOC_SUB=1 T2_VALUE_FORMULA=full T2_SG_DOCS=1"`).
- 대조: 기준선 `bank_t7328_halfB_20260819r2`(sha 상이) · 중간 `bank_t7336_halfB_20260821b` · 선행 보고서 `t7336_tasks/T7336_TASK_063.md`.
- **수리 실행 없음**(코드 무수정 · 처방은 후보로만). gold(`reward_info`)는 진단용으로만 읽었다([[23]]).

---

## 0. 채점 축 — **DB**(ACTION 아님)

```
trial 0: reward_basis=["DB"]  reward=0.0  db_check={"db_match":false,"db_reward":0.0}
trial 1: reward_basis=["DB"]  reward=0.0  db_check={"db_match":false,"db_reward":0.0}
env_assertions=[]  ·  info.nl=null  ·  info.communicate={"note":"No communicate_info to evaluate"}
```

⇒ **DB-해시 축**이다. `action_checks` 4행은 진단 보조일 뿐 성적이 아니다([[69]]).
판정 단위는 *gold 변이 집합의 정확한 재현* 하나이므로 **틀린 값의 write 는 미실행과 동등하게 치명적**이고, 실제로 trial 0 은 `Gold Account` 를 실제로 열어 DB 를 gold 와 다른 상태로 만들었다.

`action_checks` 표(진단용 · `action_match`):

| action_id | gold 행 | trial 0 | trial 1 |
|---|---|---|---|
| 063_0 | `log_verification{Taylor Brooks, tb58a3c9d2, …, 2025-11-14 03:40:00 EST}` | ✓ | ✓ |
| 063_1 | `apply_for_credit_card{**Silver Rewards Card**, Taylor Brooks, 85000, false}` (requestor=**user**) | ✗ | ✗ |
| 063_2 | `unlock_discoverable_agent_tool{open_bank_account_4821}` | ✓ | ✗ |
| 063_3 | `call_discoverable_agent_tool{open_bank_account_4821, savings, **"Silver Plus Account"**}` | ✗ | ✗ |

태스크 성격(`description.notes` 축자 요약): 신용점수 700 ⇒ Platinum(750)·Gold(720) 탈락 · "신용을 실제로 심사하는 카드만" ⇒ EcoCard(min_score 0) 탈락 · paper statement 필수 ⇒ Green Account 탈락 · $8,000 ⇒ Gold Account($10k) 탈락. **유일해 = Silver Rewards Card + Silver Plus Account (3.0% + 0.15% = 3.15%)**.

---

## 1. 변이표 (`t2_forensic.mutation_diff` 정본)

gold 변이(write) 3칸(양 trial 공통):

| aid | tool | args |
|---|---|---|
| 063_0 | `log_verification` | Taylor Brooks / tb58a3c9d2 / … / `2025-11-14 03:40:00 EST` |
| 063_1 | `apply_for_credit_card` | `{card_type:"Silver Rewards Card", annual_income:85000, rho_bank_subscription:false}` |
| 063_3 | `open_bank_account_4821` | `{user_id:tb58a3c9d2, account_type:"savings", account_class:"Silver Plus Account"}` |

### trial 0 (`s626729`)

| 종류 | 항목 |
|---|---|
| MATCHED | `log_verification{…, 2025-11-14 03:40:00 EST}` (msg **[43]**) |
| WRONGARG | `apply_for_credit_card{card_type:**Platinum Rewards Card**, Taylor Brooks, 85000, false}` (msg **[18]**, requestor=user) |
| WRONGARG | `open_bank_account_4821{savings, account_class:**Gold Account**}` (msg **[45]** → 성공 · Account ID `d200673364c5d7f4`) |
| WRONGARG | 같은 호출 재시도 (msg **[53]** → `Failed to open account: Account ID 'd200673364c5d7f4' may already exist.`) |
| MISSING | `apply_for_credit_card{Silver Rewards Card}` · `open_bank_account_4821{savings, Silver Plus Account}` |
| BLOCKED | `transfer_funds_between_bank_accounts_7291{chk_tb58a3c9d2 → d200673364c5d7f4, 8000}` — env deny `Error: Insufficient funds. Source account balance is $2500.00` |
| EXTRA / DUP | 0 / 0 |

### trial 1 (`s373753`)

| 종류 | 항목 |
|---|---|
| MATCHED | `log_verification{…}` (msg **[17]**) |
| WRONGARG | `apply_for_credit_card{card_type:**Platinum Rewards Card**, …}` (msg **[30]**, requestor=user) |
| MISSING | `apply_for_credit_card{Silver Rewards Card}` · `open_bank_account_4821{savings, Silver Plus Account}` — **저축 계좌는 시도 자체가 0회** |
| DUP | `log_verification` 재호출 (msg **[56]** → `Failed to log verification: Record may already exist.`) |
| BLOCKED / EXTRA | 0 / 0 |

### WRONGARG 필드별 대조

| 도구 | 필드 | 보낸 값 | gold | 판정 |
|---|---|---|---|---|
| `apply_for_credit_card` | card_type | `Platinum Rewards Card` (t0·t1 동일) | `Silver Rewards Card` | **불일치(치명)** |
| | customer_name | `Taylor Brooks` | `Taylor Brooks` | 일치 |
| | annual_income | `85000` | `85000` | 일치 |
| | rho_bank_subscription | `false` | `false` | 일치 |
| `open_bank_account_4821` | user_id | `tb58a3c9d2` | `tb58a3c9d2` | 일치 |
| | account_type | `savings` | `savings` | 일치 |
| | account_class | `Gold Account` (t0) / *미제출* (t1) | `Silver Plus Account` | **불일치(치명)** |

⇒ **두 축 모두 단 한 필드**가 틀렸고, 그 한 필드가 각각 *카드 티어*와 *저축 클래스*다. 즉 정보 부족이 아니라 **배제 실패**([[63]]).

---

## 2. 궤적 추적 — trial 0 (`s626729`)

### ⓐ msg [2]~[6] — 첫 결정점. 모델이 신용점수를 **날조**했고 우리 접지가 **옳게** 드롭했다

```
[2] assistant >>CALL KB_search_dense {"query":"best savings account for maximizing interest","k":10}
[2] assistant >>CALL KB_search_dense {"query":"credit card with highest cashback","k":10}
[5] assistant >>CALL check_card_application_fit {"min_cashback":"10","credit_score":"700"}
```

손님 발화는 msg [1] 하나뿐이고 거기에 `credit score` 도 `700` 도 없다(**trial 0 user 발화 전수 grep: `700`·`credit score`·`creditworth`·`no credit check` 모두 0건 — 끝까지 나오지 않는다**). 엔진 로그:

```
[T2_SG_GROUND] check_card_application_fit: 2 ungrounded operand 드롭 -> min_cashback=10 (the customer never
mentioned this kind of requirement — do not add limits they did not state); credit_score=700 (the customer
never mentioned this kind of requirement — do not add limits they did not state)
```

[6] 도구 출력 축자(머리):

```
[GROUNDING WARNING] 2 input value(s) could not be verified against the account records / knowledge base and
were dropped: min_cashback=10 (…); credit_score=700 (…).
{'eligible': [{'card': 'Platinum Rewards Card', 'facts': {'annual_fee':200.0, 'cashback':10.0,
   'min_score': 750, …}}, {'card':'Gold Rewards Card', … 'min_score':720 …},
   {'card':'Silver Rewards Card', … 'min_score':680 …}, {'card':'Bronze Rewards Card', … 'min_score':640 …},
   {'card':'EcoCard', … 'min_score':0 …}, {'card':'Crypto-Cash Back', … 'min_score':660 …}],
 'excluded': [{'card':'Diamond Elite Card','reason':'invitation-only'}],
 'note': "[checks] applied: (none) | not applied (no input given): annual_fee, fx_fee, min_payment_pct,
          cashback, limit_max, virtual_card, purchase_protection, **min_score** | …"}
```

**드롭 자체는 옳다** — 모델이 700을 어디서도 못 읽고 지어냈다. 결함은 세 가지다:
1. 문면이 `credit_score` 를 *"손님이 내건 제한"* 범주로 말한다(`do not add limits they did not state`). 실제로는 **손님 속성**이고, 다음 행동은 *"손님에게 신용점수를 물어라"* 다. [[64]] 형상 위반.
2. `min_score` 미적용 상태의 목록에 **`eligible` 이라는 표제어**를 붙여 돌려준다. `note` 의 `not applied … min_score` 는 그 표제어를 이기지 못했다.
3. 그 결과 손님이 **받을 수 없는 카드**(min_score 750, 손님 700)가 목록 1위로 올라왔다.

### ⓑ msg [7] — 카드 축 확정. 그 뒤로 되돌아오지 않는다

```
[7] assistant :: "Based on your goal to maximize the interest you'll earn over one year and your interest in a
     credit card with high cashback, I recommend the **Platinum Rewards Card**."
```

같은 도구 출력 안에 `'min_score': 750` 이 **facts 로 그대로 실려 있었다**. 그러나 그 시점 문맥에 손님의 점수는 **존재하지 않았고**(모델의 날조는 우리가 지웠다), 모델은 손님에게 묻지도 않았다.

### ⓒ turn 5~11 — 검색 서브가 **3축 모두 오답을 배달**했다

```
turn=5  [T2_DOCGROUP] raw='checking_accounts credit_cards savings_accounts' → 3군
        [T2_SEARCH_AGENT] group=checking_accounts · 문서 113 · now=2025-11-14 turn=5
        [T2_DOCDECIDE] → 'Blue Account'                    → [T2_SEARCH_ON_PROCEED] 재료 247자 배달
turn=9  [T2_SEARCH_AGENT] group=credit_cards · 문서 110
        [T2_DOCDECIDE] → 'Bronze Rewards Card'
        [T2_CP2_CLOBBER] SEARCH_ON_PROCEED 가 미소비 배달물 247자를 버리고 254자로 덮어씀
turn=11 [T2_SEARCH_AGENT] group=savings_accounts · 문서 92
        [T2_DOCDECIDE] → 'Gold Account'
        [T2_CP2_CLOBBER] SEARCH_ON_PROCEED 가 미소비 배달물 254자를 버리고 247자로 덮어씀
        [T2_SEARCH_ON_PROCEED] deny 아님 · 재료 247자 배달
```

배달 문면(A2 `policy_ontology.decided_by_docs_text` 축자·렌더 길이 검산 247자 = 템플릿+`Gold Account`(12자), 254자 = +`Bronze Rewards Card`(19자) — 로그 숫자와 **정확히 일치**):

> *"A separate check was run on the policy documents on record, with this request and nothing else from this conversation in front of it. Documents whose stated period does not include today were excluded before it answered. **It answers: Gold Account.**"*

turn 15 에 그 결정문이 재생성 버퍼에 붙었고(`[T2_DECISION_CARRY] 이 턴 재생성 버퍼에 부착 (247자)` · `[T2_MATERIAL_GATE] stop=other_lever(prov) turn=15`), **바로 그 turn 의 산출인 msg [15] 에서 계좌 이름이 바뀐다**:

```
[11] assistant :: "…open a **Diamond Elite Savings Account**…"
[13] assistant :: "### Opening a **Diamond Elite Savings Account**…"
[15] assistant :: "Let's proceed with the correct steps to apply for the Platinum Rewards Card and open the
      **Gold Savings Account**."          ← 배달 직후 최초 등장
```

`Gold Account` 는 최소예치 $10,000 이고 손님은 $8,000 이다. 즉 **우리 층이 하드 제약 위반 후보를 권위 문면으로 배달했고 모델이 채택했다.**

### ⓓ msg [17]~[19] — 우리 층이 그 상태로 **write 를 밀었다**

```
turn=11 [T2_GIVE_EXEC] nudge idle=['apply_for_credit_card']
turn=15 [T2_RESOLVE] user-action instruct target=apply_for_credit_card     (×2)
        [T2_STACK] guidance tag=resolve_write passed (new (seen=10))
```

```
[17] assistant :: "…Select the '**Platinum Rewards Card**'. Fill out the application form…"
[18] user  >>CALL apply_for_credit_card {"card_type":"Platinum Rewards Card","customer_name":"Taylor Brooks",
                                          "annual_income":85000,"rho_bank_subscription":false}
[19] TOOL  :: "Credit card application submitted…"
```

⇒ **WRONGARG(카드) 확정 지점 = msg [18]**, 결정된 지점은 msg [7], 원인이 만들어진 지점은 msg [5]→[6]이다. 손님 시나리오 6항이 *"card_type: Use whatever card the agent recommended"* 이므로 **카드 값은 전적으로 에이전트 추천이 결정한다**(user_sim 면책 사유 아님·[[21]]).

`T2_DEMANDED_STEP head=reads:check_card_application_fit → check_card_application_fit` 는 turn 3 에 1회 발화해 **비교기 선행을 실제로 만들었다**. 그러나 그 비교기가 `min_score` 미적용으로 돌았기 때문에 **요구가 공허하게 충족**됐다.

### ⓔ msg [21]~[39] — 발견 축 표류. 우리 층 오지목 1건 + 재료 채널 폐쇄

```
[T2_OUR_NAMES] 등재 name=apply_savings_account_credit_6831 (출처=T2_DISCOVERY_STEP2 지목)
[T2_DISCOVERY_STEP2] deny name=apply_savings_account_credit_6831 (레지스트리 폴백·미unlock·formalize 정합)
[T2_ACTION_INDEX] 1회 표면화 4536자
[T2_FORCE_ACTION] say-don't-do → tool_choice=required 재생성
```
[25] 모델이 정확히 그 이름을 unlock → [26] 설명을 읽고 [27] 스스로 되돌아옴. **2턴 손실**(t7336 과 동형 재현).

같은 구간 `T2_PROV` 가 모델의 날조 도구명 4건(`open_savings_account_7894`·`open_savings_account_6832`·`create_savings_account_6832`·`transfer_funds_between_accounts`)을 레지스트리 메시지로 되돌렸다 — **정확·유효**. 그리고 `[T2_MATERIAL_GATE] stop=resolve_cap(정체 3회) turn=23/25/27/29` 로 재료 채널이 한동안 닫혔다.

### ⓕ msg [39]~[45] — `T2_WRITE_ARG_ENUM` 이 오답을 **정본 이름으로 다듬어** 통과시켰다

```
turn=39 [T2_ARG_AXIS] formalize → ['savings'] (raw='savings')          ← 축은 정확
        [T2_WRITE_ARG_ENUM] 저장된 축 결정 동봉 group=savings_accounts (247자)
        [T2_WRITE_ARG_ENUM] deny val='Gold Savings Account' group=savings_accounts (후보 9)
turn=43 (log_verification 직후) 같은 두 줄 반복
```

A2 `write_arg_enum.feedback` 축자:

> *"Error: [OFFICIAL-NAME] '{val}' is not one of the official account_class names on file. The names on file for savings_accounts are: {candidates}. Choose the one you meant, verbatim, and call the tool again."*

여기에 `t2_gate_patch.py:9583-9589` 가 **저장된 축 결정**(= 위 247자 `It answers: Gold Account.`)을 **덧붙인다**. 후보 9개에는 정답 `Silver Plus Account` 가 들어 있었으나, 모델은 붙어 온 결정문을 골랐다:

```
[45] assistant :: "It seems there was a mistake in the account class name. The correct name for the Gold
      Savings Account is `**Gold Account**`."
[45] assistant >>CALL call_discoverable_agent_tool {"agent_tool_name":"open_bank_account_4821",
      "arguments":"{\"user_id\":\"tb58a3c9d2\",\"account_type\":\"savings\",\"account_class\":\"Gold Account\"}"}
[46] TOOL :: "Bank account opened successfully! - Account ID: d200673364c5d7f4 - Account Class: Gold Account"
```

⇒ **WRONGARG(저축) 확정 지점 = msg [45]**. 열거 게이트는 *닫힌 술어로서는 정확히 작동*(존재하지 않는 이름 차단)했고, **손해는 동봉된 결정문에서 왔다**.

`T2_VERDICT_GATE`(이름은 집합 안이지만 **손님 요구와 충돌**하는 값을 되돌리는 레버, `t2_gate_patch.py:9553-9570`)는 이 자리가 정확히 그 자리인데 **`T2_VERDICT_GATE=0` 으로 꺼져 있어 발화 0회**다.

### ⓖ msg [47]~[59] — 확인은 했으나 정정은 없었다

[50] `get_all_user_accounts_by_user_id_3847` 결과에 `chk_tb58a3c9d2 · Light Blue Account · current_holdings: **2500.00**` 이 **축자로** 실렸다. 손님이 $8,000 을 넣겠다고 했는데 잔액은 $2,500 이다. 모델은 그 값을 읽고도 계좌 클래스를 재검토하지 않았고 [59] 에서 $8,000 이체를 시도해 env deny 를 받았다.

### ⓗ turn 49 — 우리 층이 **태스크와 무관한 하위목표**를 3회 밀어 넣었다

```
turn=49 [T2_VALUE_ACQUIRE] consumers card_last_4_digits=1
        [T2_VALUE_ACQUIRE] give-surfacing → nudge (regen)      ×3
```
그 turn 의 산출 msg [51] 축자:

> *"### Retrieving the Last Four Digits of Your Credit Card … To give you the `get_card_last_4_digits` tool, I will use the `give_discoverable_user_tool` function."*

카드 마지막 4자리는 이 태스크의 gold 어디에도 없다. 계좌 클래스를 고칠 마지막 구간에서 **문맥 예산을 우리가 소비**했다.

### ⓘ 종료

[62]~[65] 손님이 $5,500 추가 입금을 제안하고 [66] `###OUT-OF-SCOPE###` 로 종료. **저축 클래스는 끝까지 `Gold Account`.**

---

## 3. 궤적 추적 — trial 1 (`s373753`)

### ⓐ 분기점 = msg [5] (trial 0 과 동일 지점, **다른 형태의 같은 결손**)

```
[5] assistant >>CALL check_card_application_fit {"max_annual_fee":null, …, "min_cashback":"10",
     "credit_score":**null**, "business":"false","invited":"false", …}
[T2_SG_GROUND] check_card_application_fit: 1 ungrounded operand 드롭 -> min_cashback=10 (…)
```

trial 0 은 700을 **날조**해 드롭당했고, trial 1 은 아예 **null 로 비웠다**. 결과는 같다 — `min_score` 미적용 · `eligible` 1위 `Platinum Rewards Card`. **양 trial 모두 손님에게 신용점수를 묻지 않았다.**

### ⓑ msg [7]~[18] — 검증 축은 성공

```
[10] verify_identity → "NOT_VERIFIED — the account record has not been fetched yet … Look the customer up with
      get_user_information_by_name/by_email/by_id, then call this tool again. Use a DIFFERENT identifier each time"
[12] get_user_information_by_email → tb58a3c9d2 …
[14] verify_identity → "VERIFIED — 2 identity values match … you may now call log_verification. Its time_verified
      argument must be the CURRENT time: FIRST call get_current_time, wait for its result, then copy that exact
      timestamp … do not make one up."
[16] get_current_time → 2025-11-14 03:40:00 EST
[17] log_verification{…, "2025-11-14 03:40:00 EST"}  → gold 063_0 MATCHED
```
**[[64]] 를 지킨 발화(무엇이 틀렸나 + 무엇을 하면 풀리나)가 정확히 그 칸을 샀다.** `T2_PHASE_PRECEDE` 10회 발화(t0 은 16회) — 양 trial 모두 검증 축은 통과했다(t7336 trial 0 의 `log_verification` MISSING 은 이번에 재발하지 않았다).

### ⓒ msg [19]~[25] — 저축 계산기가 **3회 전부 계산 불능**

```
[19] >>CALL get_correct_savings_apy {"savings_account_type":"Light Blue Savings Account", …}
[20] TOOL :: "Error: [READ-FIRST] this calculation depends on records you have not read yet in this conversation.
      Missing required reads (BASE names): get_all_user_accounts_by_user_id. …"
      ← 로그 [T2_SG_REQREADS] get_correct_savings_apy denied: missing reads ['get_all_user_accounts_by_user_id']
[21] 모델이 unlock+call get_all_user_accounts_by_user_id_3847  → [23] chk_tb58a3c9d2 · 2500.00
[24] >>CALL get_correct_savings_apy {"savings_account_type":"Light Blue Savings Account", …}
[25] TOOL :: "Correct savings APY under the stacking policy (…): **(could not compute — check your arguments)**%."
```

로그(3회 전부 동일):
```
[T2_SG_TRACE]   get_correct_savings_apy: iso=True mode=fetch_formalize
[T2_SG_DOCS]    클래스 ['light_blue_account','platinum_account'] · 문서 10편 · 10434자 전달(검색 0)
[T2_SG_ISOLATE] operand-size get_correct_savings_apy.components: **sub=0 rows · source=0 rows**
[T2_SCAFFOLD_GET] get_correct_savings_apy -> **None**
```

모델이 준 `savings_account_type` 이 실재하지 않는 이름(`Light Blue Savings Account`·`Platinum Savings Account`·`Bluest Savings Account`)이라 클래스 매핑이 **체킹 상품**으로 떨어졌고 격리 서브가 정직하게 0행을 냈다. **abstain 자체는 옳다.** 결함은 반환 문면이다 — `get_correct_savings_apy` 선언에 `missing_hint` 가 **없어서** 엔진 기본 문자열 `(could not compute — check your arguments)`(`t2_scaffold_get.py:281`)이 나갔다. *무엇이 틀렸는지*(그 이름은 파일에 있는 savings 클래스가 아니다)도 *무엇을 하면 풀리는지*(9개 공식 명단)도 말하지 않는다.

⚠ 이것은 t7336 대비 **거동 변화**다: t7336 trial 1 에서는 같은 경로가 `6.5` 를 계산해 Platinum 선택을 **추인**했다. 이번엔 계산 불능 3회.

### ⓓ msg [29]~[31] — 카드 축 확정

```
[29] assistant :: "1. **Platinum Savings Account**: Base APY 6.5% / Boost 0.65% / Bonus 0.25% / **Total 7.4%**
      2. **Platinum Rewards Card**: Cashback 10% …"      ← 도구 산출 0 · 순수 날조
[30] user >>CALL apply_for_credit_card {"card_type":"Platinum Rewards Card", …}
```
같은 turn(=29)에 저축 축 결정문이 배달됐다:
```
turn=29 [T2_SEARCH_AGENT] group=savings_accounts · 문서 92
        [T2_DOCDECIDE] → 'Gold Account'
        [T2_SEARCH_ON_PROCEED] deny 아님 · 재료 247자 배달
```
**모델은 그것을 채택하지 않았다**(trial 1 전 궤적에 `Gold` 0건). 즉 이 배달은 *오답이었고 무시됐다* — 이번 trial 에서는 실피해 0, 대신 기여도 0.

### ⓔ msg [32] — 손님이 드디어 신용점수를 말한다. **재검토 0**

```
[32] user :: "…Also, my **credit score is around 700**—just reiterating that I only want products that actually
      verify creditworthiness."
[33] assistant :: "Great to hear that you've submitted the application for the Platinum Rewards Card. …
      Additionally, I will ensure that the account meets your credit score requirements."
```

이 시점부터 `check_card_application_fit` 을 다시 부르면 `credit_score=700` 이 접지를 통과하고 Platinum(750)·Gold(720) 이 `excluded` 로 떨어진다 — t7328 trial 1 에서 실제로 그렇게 됐다(§5). **모델은 재호출하지 않았고, 우리 층에도 재호출을 요구하는 발화가 없다**(`T2_DEMANDED_STEP` 은 turn 3/5 에만 2회, `T2_SEARCH_REARM` 0회).

### ⓕ msg [37]~[48] — 존재하지 않는 상품 + 날조 APY

```
[36] TOOL :: "Platinum Account at a glance … APY: 6.5% | **Minimum balance to maintain account: $50,000**"
[37] assistant :: "…the **Bluest Savings Account** would be the best fit…"      ← savings 상품 아님(체킹)
[48] assistant :: "Base APY 6.5% + Boost 0.7% + Bonus 0.65% = **Total APY 7.85%** … Paper Statements: available,
      **no additional fee**."                                                   ← 도구 0회 · 순수 날조
```
$50,000 최소잔액을 **축자로 읽고도** $8,000 손님에게 그 계열을 계속 권했다.

### ⓖ msg [50]~[61] — write 를 끝내 하지 않고 포털 안내로 접었다

```
[58] assistant :: "It seems there was an error in providing the tool to open the Bluest Savings Account.
      Instead, I will guide you through the steps to open the account … manually."
[60] assistant :: "**Menu Path**: Settings → Communication Preferences → Statements → Paper Statements …"
[61] user :: "…Thanks for finding the best combination … ###STOP###"
```
`open_bank_account_4821` 은 **unlock 도 call 도 0회**(action_check 063_2 ✗). `T2_ACTION_INDEX`·`T2_OWNERSHIP_FIX` 는 trial 1 에서 각 0회/1회로, trial 0 처럼 정답 이름을 반복 표면화하지 못했다.

---

## 4. 레버 발화표 (이 sim 두 줄만 grep · 457/367 라인 전수)

| 레버 | t0 | t1 | 판정 |
|---|---|---|---|
| `T2_SG_GROUND` (=요구 접지·[GROUNDING WARNING]) | 1 | 1 | **발화·판정 정확(드롭은 옳다)·처방 부재** → `min_score` 미적용 `eligible` 목록이 Platinum 을 1위로 올렸다. **양 trial WRONGARG 의 필요조건** |
| `T2_SG_DOCS` (이번 런 ON) | **0** | 3 | t1 만 발화. 10~12편 배달했으나 클래스 매핑이 체킹 상품이라 `components` 0행 → 계산 불능 3회 |
| `T2_SG_ISOLATE` | 0 | 9 | t1 `sub=0 rows · source=0 rows` 3회 — 정직한 abstain, 그러나 반환 문면이 무정보 |
| `T2_SG_REQREADS` (C587) | 0 | 1 | **발화·수용·효과 있음**(t1 [21] 선행 read 실행) |
| `T2_SEARCH_AGENT` / `T2_DOCDECIDE` | 6 / 3 | 6 / 3 | **발화·3축 전부 오답**: `Blue Account`(손님이 체킹을 명시 거부) · `Bronze Rewards Card`(gold=Silver Rewards) · `Gold Account`($10k 최소·손님 $8,000). t0 은 **채택**(msg [15]·[45]) · t1 은 무시 |
| `T2_SEARCH_ON_PROCEED` / `T2_DECISION_CARRY` | 3 / 3 | 3 / 3 | 배달 배관은 살아 있다. 배달물의 품질이 문제 |
| `T2_CP2_CLOBBER` | 2 | 0 | **배관 결함 실재**: `Blue Account`(247자)와 `Bronze Rewards Card`(254자) 배달물이 미소비 폐기됐다. 이번엔 폐기된 것도 오답이라 실피해 0 |
| `T2_WRITE_ARG_ENUM` | 4 | 0 | **닫힌 술어는 정확**(`'Gold Savings Account'` 차단·후보 9 제시)·**동봉된 결정문이 손해**(→ `Gold Account` 확정) |
| `T2_ARG_AXIS` | 4 | 0 | `formalize → ['savings']` — t7336 의 `business_savings` 오지목 **재발 없음**(개선) |
| `T2_VERDICT_GATE` | **0** | **0** | 미발화(`T2_VERDICT_GATE=0`) — 이름은 집합 안이지만 요구와 충돌하는 값을 되돌릴 **바로 그 자리**가 비어 있었다 |
| `T2_SUB_REQUIREMENT` | **0** | **0** | **미발화(런 전체 0회)** — `run_t7346…sh:88` 이 `T2_SUB_REQUIREMENT=0`, `go_stack.sh` 는 이 이름을 **선언조차 하지 않는다** |
| `T2_DEMANDED_STEP` | 1 | 2 | **발화·수용**(`reads:check_card_application_fit`). 그러나 비교기가 공허하게 충족돼 효과 0 |
| `T2_CLAIMPROV` | 44 | 60 | **발화했으나 무해무익** — `kind-index rescued`(kind='search') 다수가 `unbacked=0` 을 만들어 t1 [48] `7.85%` 날조·[60] paper 절차 날조를 **못 잡았다**(kind 만 보고 value 를 안 본다) |
| `T2_PHASE_PRECEDE` | 16 | 10 | **양 trial 발화·효과** — `log_verification` 양쪽 MATCHED (t7336 t0 결손 해소) |
| `T2_PREKB` (`require_before`) | 6 | 0 | t0 `fam=open_bank_account (missing get_all_user_accounts_by_user_id)` 1회 — 수용됨 |
| `T2_DISCOVERY_STEP2` | 1 | 0 | **오지목 1건**(`apply_savings_account_credit_6831`) + `T2_FORCE_ACTION` 강제 → 2턴 손실 |
| `T2_PROV` | 10 | 2 | **정확**: 모델 날조 도구명 4건을 레지스트리 메시지로 반려 |
| `T2_VALUE_ACQUIRE` | 6 | 0 | **오발화**: turn 49 에 `card_last_4_digits` 하위목표 3회 주입 — gold 무관·문맥 예산 소모 |
| `T2_MATERIAL_GATE stop=…` | 14 | 11 | t0 turn 23~29 `resolve_cap(정체 3회)` 로 재료 채널 폐쇄 구간 존재 |
| `T2_PIN_READ` | 0 | 0 | 미발화(이 태스크에 해당 절차 없음) |
| `T2_FOLLOWUP` | 0 | 0 | 미발화 |
| `FAB_STRIP` | 0 | 0 | 미발화 — t0 [51]/t1 [48]·[60] 날조에 무반응 |
| `T2_ARG_PRODUCERS` (F8) | 0 | 0 | 미발화 — 오발화 0이지만 양성 기회도 0 |
| `READ-FIRST` (`T2_SG_REQREADS` 문면) | 0 | 1 | t1 [20] 발화·수용. **`check_card_application_fit` 에는 걸리지 않는다** — `credit_score` 공백을 막을 자리가 비어 있다 |
| `T2_REQUIRE_DOC_DELIVER` | 0 | 0 | 미발화 |
| `T2_SEARCH_REARM` | 0 | 0 | **미발화** — 손님이 t1 [32] 에서 새 요구(신용점수 700)를 냈는데 축이 이미 소진돼 재무장 없음 |

### 이번 런의 처치 3종(`ON`) 개입 여부

| 처치 | 개입? | 결과 |
|---|---|---|
| `T2_ARG_DOC_SUB=1` | 발화(t0 3 · t1 8) | `spend_category` 미사용 안내만. 이 태스크의 두 오답 축과 무관 |
| `T2_VALUE_FORMULA=full` | **미개입** | `spend_amount`/`spend_category` 가 없어 `documented_return_for_stated_spend` 계산 자체가 안 돎 |
| `T2_SG_DOCS=1` | t1 만 3회 | 문서는 배달됐으나 `components` 0행 → 계산 불능. t7336 대비 **6.5 추인이 사라진 것은 개선**이나 대신 무정보 abstain |

⇒ **이번 런에서 새로 켠 3종 중 어느 것도 이 태스크의 두 결정 축에 닿지 않았다.**

---

## 5. 선행 판정과의 대조

| 문서/런 | 그때의 판정 | 지금(t7346) |
|---|---|---|
| `t7336_tasks/T7336_TASK_063.md` §8 | trial 0 1차 원인 = our_layer(검증 게이트 진입 술어 → `log_verification` MISSING) · trial 1 1차 원인 = our_layer(`credit_score` 드롭 처방 부재 → Platinum) | **⑴ 검증 축 결손은 해소**(양 trial `log_verification` MATCHED · `T2_PHASE_PRECEDE` 16/10회) ⑵ **`credit_score` 축은 악화** — t7336 은 2 trial 중 1개만 Platinum, t7346 은 **2/2 Platinum** ⑶ `T2_ARG_AXIS` 의 `business_savings` 오지목은 **재발 없음** ⑷ `T2_DISCOVERY_STEP2` 오지목·`T2_CP2_CLOBBER`·`T2_CLAIMPROV` kind-폴백 **동일 재현** |
| `T7336_TASK_063` §9 처방 3 (`T2_SUB_REQUIREMENT` 재점화) | *"`_reqs` 가 `T2_SUB_REQUIREMENT=0` 이라 항상 빈 리스트"* — 미실행 | **여전히 미실행**. t7346 도 `T2_SUB_REQUIREMENT=0`. `[T2_DOCDECIDE] → 'Gold Account'` 가 **또** 나왔다 |
| `T7336_TASK_063` §9 처방 1·2(드롭 문면·`eligible` 표제어) | 미실행 | **여전히 미실행**. 문면 바이트 동일 |
| `T7336_TASK_063` §9 처방 6(저축 클래스 비교 GET) | 미실행 | **여전히 미실행**. 저축 축에 결정론 비교기가 없어 t0 `Blue→Gold`, t1 `Light Blue→Platinum→Bluest` 로 헤맸다 |
| 기준선 `bank_t7328_halfB_20260819r2` | 양 trial **카드 축 MATCHED**(`Silver Rewards Card`) · 저축 축만 실패(`MISSING` / `WRONGARG Silver Account`) · reward 0.0 | **카드 축 회귀 확정** |

### 카드 축 3런 추이 (같은 태스크·같은 두 seed)

| 런 | t0 `card_type` | t1 `card_type` | 카드 축 MATCHED |
|---|---|---|---|
| t7328 (08-19) | `Silver Rewards Card` ✓ | `Silver Rewards Card` ✓ | **2/2** |
| t7336 (08-21) | `Silver Rewards Card` ✓ | `Platinum Rewards Card` ✗ | 1/2 |
| **t7346 (08-22)** | `Platinum Rewards Card` ✗ | `Platinum Rewards Card` ✗ | **0/2** |

t7328 에서 성공한 기전은 **손님이 신용점수를 먼저 말했다**는 것 하나다(축자):

```
t7328 t0 [9]  user :: "Also, my credit score is around **700**, and I only want a real credit card that
                       actually checks creditworthiness…"      → [10] 모델이 Platinum(750)·Gold(720) 배제
t7328 t1 [5]  user :: "My credit score is around **700**…"      → [6] check_card_application_fit{credit_score:"700"}
                                                                 → [7] eligible = Silver Rewards Card 단독
```
t7328 두 trial 모두 모델이 **먼저 물었기 때문에** 손님이 말했다(t1 [4] *"Could you please provide me with … 1. Your credit score."*). t7346 은 **양 trial 모두 묻지 않았고**, user-sim 전역 지침(`global_simulation_guidelines` 축자: *"Disclose information progressively. Wait for the agent to ask for specific information before providing it."*)이 시나리오의 *"mention proactively"* 를 이겨 t0 에서는 끝까지, t1 에서는 카드 신청 **후에야** 나왔다.

⇒ **user-sim 변동은 원인이 아니라 노출기다([[21]])** — 흡수 지점은 *"점수를 모르면 물어라"* 이고, 그 지점은 우리 `T2_SG_GROUND` 문면이다.

### 저축 축 `T2_DOCDECIDE` 결정론적 재현 (6/6)

| 런 · sim | checking | credit_cards | savings |
|---|---|---|---|
| t7328 t0 / t1 | `Blue Account` | `Bronze Rewards Card` | **`Gold Account`** |
| t7336 t0 / t1 | `Blue Account` | `Bronze Rewards Card` | **`Gold Account`** |
| t7346 t0 / t1 | `Blue Account` | `Bronze Rewards Card` | **`Gold Account`** |

**3런 · 6 sim · 2 seed 에서 단 한 번도 `Silver Plus Account` 가 나온 적이 없다.** 이는 격리 프로브 `x343`(n=24·`x343_sub_requirement_iso.py` 머리말 축자)의 A_REF 셀과 정확히 같은 값이다:

> *"격리 서브 · 문서 O · 후보 O · **요구 X**(라이브) → savings **Gold 8/8** … 손님 요구 메시지를 축자로 받으면 **`Silver Plus` 24/24 정답**, 무관한 요구를 주면 **0/24**(부정통제 통과)"*

---

## 6. 원인 확정

### trial 0 (`s626729`) — WRONGARG 2 · MISSING 2

| 결손 | 주체 | 근거(코드 경로·축자) |
|---|---|---|
| `card_type=Platinum Rewards Card` (msg [18]) | **our_layer** 1차 | `t2_scaffold_get.py:459-462` + A2 `scaffold_get_tools[check_card_application_fit].ground.intent_fields[param="credit_score"]`(`cue_any=["credit score","fico","my score"]`·`on_fail="drop"`). 드롭은 옳으나 문면이 *"do not add limits they did not state"* 로 **범주를 틀리게** 말하고 *"손님에게 물어라"* 를 **안 댄다**([[64]]). 그 결과 `note`= *"not applied (no input given): … min_score"* 인 목록이 `eligible` 표제어로 나가 Platinum(min_score 750)이 1위 |
| " | **model** 2차 | 같은 도구 출력에 `'min_score': 750` 이 실려 있었고, 손님 점수가 미상임을 알 수 있었는데 **묻지 않았다** |
| `account_class=Gold Account` (msg [45]) | **our_layer** 1차 | `t2_gate_patch.py:3413-3418`(`_reqs = []` · `if (os.environ.get("T2_SUB_REQUIREMENT")=="1" or …VERDICT_CARRY…) and _po.get("requirement_prompt")`) — 런이 `T2_SUB_REQUIREMENT=0`(`run_t7346_overnight_stage1_20260822.sh:88`)이고 `go_stack.sh` 에 선언 자체가 없어 **항상 빈 리스트** ⇒ `t2_search.decide_from_docs`(`t2_search.py:720-742`)가 `decide_candidates_text`(이름 9개)+문서 92편만 보고 `Gold Account` 를 냈다. A3 `policy_ontology.requirement_prompt` 는 **양 정본 층에 이미 선언돼 있다**(`a2/banking_knowledge.specific.json:9545` · `.gate.json:10189`) |
| " | **our_layer** 1차(증폭) | `t2_gate_patch.py:9583-9589` — `T2_WRITE_ARG_ENUM` deny 가 후보 9개와 함께 **저장된 결정문**(`It answers: Gold Account.`)을 동봉해 모델이 `'Gold Savings Account'` → `'Gold Account'` 로 수렴(msg [45] 축자 *"The correct name for the Gold Savings Account is `Gold Account`"*) |
| " | **our_layer** 2차 | `T2_VERDICT_GATE=0`(`go_stack.sh:428`) — 집합 안이지만 요구와 충돌하는 값을 되돌릴 자리가 비어 있었다 |
| 발견 2턴 손실 | **our_layer** | `T2_DISCOVERY_STEP2` 가 `apply_savings_account_credit_6831` 을 요청-정합으로 오지목 + `T2_FORCE_ACTION` 으로 호출 강제 |
| 문맥 낭비(turn 49) | **our_layer** | `T2_VALUE_ACQUIRE` 가 gold 무관 `card_last_4_digits` 하위목표 3회 주입 → msg [51] 한 섹션 소비 |
| 이체 BLOCKED | **env** | `Error: Insufficient funds. Source account balance is $2500.00` — 정당. gold 변이 아님(DB 축 무기여) |

**trial 0 1차 원인 = our_layer** (요구 없는 결정 서브 → `Gold Account` → enum deny 동봉으로 확정).

### trial 1 (`s373753`) — WRONGARG 1 · MISSING 2 · DUP 1

| 결손 | 주체 | 근거 |
|---|---|---|
| `card_type=Platinum Rewards Card` (msg [30]) | **our_layer** 1차 | 위와 동일. `credit_score:null` 로 들어와 드롭조차 필요 없었고, `eligible` 목록이 min_score 미적용으로 나갔다. 손님이 [32] 에서 700을 말한 뒤에도 **재호출을 요구하는 발화가 우리 층에 없다**(`T2_SEARCH_REARM` 0 · `T2_DEMANDED_STEP` 은 turn 3/5 에 소진) |
| " | **model** 2차 | [32] 이후 `check_card_application_fit` 재호출 0회. [33] 에서 *"I will ensure that the account meets your credit score requirements"* 라고만 말하고 실행 0 |
| `open_bank_account_4821` 시도 0회 | **model** 1차 | [58]/[60] 에서 포털 수동 절차로 접었다. 같은 도구를 t0 은 실제로 열었으므로 능력 문제가 아니다 |
| " | **our_layer** 2차 | `get_correct_savings_apy` 무정보 abstain 3회(`t2_scaffold_get.py:281` 기본 문자열 · A2 선언에 `missing_hint` **부재**)가 *"어느 이름이 실재하는가"* 를 끝내 알려주지 않아 모델이 `Light Blue→Platinum→Bluest` 로 표류했다. `T2_ACTION_INDEX` 도 t1 에서는 0회 |
| APY 7.85% · paper 무료 날조 | **model** | 도구 호출 0회. `T2_CLAIMPROV` 가 `kind='search'` 로 `unbacked=0` 처리해 통과시켰다(우리 층 무해무익) |
| `log_verification` DUP | **model** | [56] 재호출 → `Record may already exist.` DB 축 무기여 |

**trial 1 1차 원인 = our_layer** (카드 축) · 저축 축은 **model**(시도 0)이 1차, our_layer(무정보 abstain)가 2차.

### 4주체 총평

- **our_layer**: 확정 3건 — ⑴ 요구 없는 격리 결정 서브(`T2_SUB_REQUIREMENT=0`) ⑵ `credit_score` 드롭 문면의 [[64]] 위반 + `eligible` 표제어 ⑶ enum deny 의 결정문 동봉. 부수 3건 — `T2_DISCOVERY_STEP2` 오지목 · `T2_VALUE_ACQUIRE` 오발화 · `get_correct_savings_apy` 무정보 abstain.
- **model**: 신용점수 미질의(양 trial) · 도구가 보여준 `min_score:750`·`최소잔액 $50,000`·`current_holdings 2500.00` 을 읽고도 대조 0 · 날조 3건(t0 [51] 절차·t1 [48] 7.85%·t1 [60] 메뉴 경로) · t1 write 유기.
- **env**: `Insufficient funds` · `Record may already exist` · `Unknown discoverable tool` 전부 정당. **결함 아님**.
- **user_sim**: **면책 아님·유해 아님**. 전역 지침의 *"progressive disclosure"* 가 시나리오의 *"mention proactively"* 를 이겨 t0 에서 점수를 끝까지 말하지 않았지만, t7328 두 trial 이 보여주듯 **에이전트가 물으면 100% 나온다**. 흡수 지점은 agent 측이다([[21]]).

---

## 7. 처방 후보 (제안까지 · 실행 금지 · [[70]] 절충 의무)

1. **`decide_from_docs` 에 손님 요구를 다시 싣는다(`T2_SUB_REQUIREMENT`).** 격리 근거는 이미 있다(x343 n=24: 요구 없으면 `Gold` 24/24 오답 · 요구 축자면 `Silver Plus` 24/24 정답 · D_NEG 0/24 부정통제 통과). 라이브에서 `Gold Account` 가 **3런 6/6 결정론적으로** 재현됐다. 재료(`requirement_prompt`)는 **이미 두 정본 층에 선언돼 있고 엔진 경로도 살아 있다** — 켜지 않은 것뿐이다. ⚠[[70]]: 전체 reward 짝 A/B + 태스크별 부호표 없이 켜지 말 것(t7305 가 왜 0 으로 남았는지 원장 확인 선행).
2. **`intent_fields` 드롭 문면에 "다음 행동"을 붙인다([[64]]).** `credit_score` 는 *제약*이 아니라 *손님 속성*이다. A2 에 `kind: "attribute"|"constraint"` 를 **선언**하고 엔진은 그 값으로 문면만 고른다(도메인 어휘 0·[[59]] 준수). attribute 계열 문면 = *"the customer has not stated this — ask them for it, then call this tool again."*
3. **필터 미적용 축이 있으면 `eligible` 이라는 낱말을 쓰지 않는다.** `note` 의 `not applied (no input given): min_score` 가 표제어에 졌다. A2 `return_template` 층에서 약한 표제어로 바꾸고 미적용 축을 각 행에 붙인다(엔진 리터럴 아님).
4. **`T2_WRITE_ARG_ENUM` 의 결정문 동봉을 조건화한다.** 현행(`t2_gate_patch.py:9583-9589`)은 **저장된 결정이 있으면 무조건** 붙인다. 그 결정이 요구 없이 만들어진 것이면(=처방 1 미적용 상태) **후보 명단만** 주는 편이 낫다. ⚠처방 1 이 먼저다 — 배달물의 품질을 고치는 것이 배관을 좁히는 것보다 앞선다.
5. **`get_correct_savings_apy` 에 `missing_hint` 를 선언한다.** 현행 기본 문자열 `(could not compute — check your arguments)` 은 [[64]] 위반이다. 대안 문면의 출처는 A3 `doc_index['savings_accounts']` 키(9개 공식 명칭)의 기계 전개뿐(gold 무참조).
6. **`T2_VERDICT_GATE` 를 이 자리에서 재검정한다.** 이름이 집합 안(`Gold Account`)인데 손님 요구($8,000·paper)와 충돌하는 값이 통과하는 것이 정확히 이 게이트의 대상이다. 현재 `go_stack.sh:428` 에서 0.
7. **저축 클래스 비교 GET(t7335·t7336 처방의 3회째 재제출).** `check_card_application_fit` 동형으로 `savings_account_class × {base_apy_tier1, min_opening_deposit, ongoing_min_balance, paperless_required, card_bonus}` 를 **닫힌 술어로 빼기**([[63]]). 카드 축은 그 GET 이 있어서 t7328 2/2·t7336 1/2 를 샀고, 저축 축은 GET 이 없어 **3런 6/6 전패**다. 출처는 정책·KB 문서뿐(`Silver Plus Account` 를 gold 에서 읽으면 실험 무효·[[23]]).
8. **(관측용) `T2_VALUE_ACQUIRE` 의 소비자 술어 점검.** turn 49 에 `card_last_4_digits` 를 3회 밀었다 — 이 태스크에는 소비자가 없다. 오발화 여부를 다른 태스크 로그와 함께 세어볼 것.

---

### 부록 — 재현 커맨드

```
cd C:\workspace\ba-frft\scripts\distill\tau2
PYTHONIOENCODING=utf-8 py -3 -c "import gzip,json,sys; sys.path.insert(0,'.'); import t2_forensic as F; \
 d=json.load(gzip.open(r'..\..\..\reports\facet_rft_2026\sim_results\bank_t7346_halfB_20260822.results.json.gz','rt',encoding='utf-8')); \
 [print(s['trial'], F.mutation_diff(s, F.mutating_tools())) for s in d['simulations'] if s['task_id']=='task_063']"
```
로그 분리: `[sim=task_063#s626729]`=trial 0 · `[sim=task_063#s373753]`=trial 1.
(태스크 id 는 결과 파일에서 **`task_063`** 이다 — `'063'` 으로 필터하면 0건이 나온다.)
