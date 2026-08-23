# t7346 task_055 per-step 포렌식 — 2026-08-23

- 런: `bank_t7346_halfA_20260822`(results/log 전부 로컬 gz · SSH 0 · git 무접촉).
- sim 2개: **trial 0 = `task_055#s626729` reward 0.0**(75 msgs · `user_stop` · 935s) / **trial 1 = `task_055#s373753` reward 0.0**(83 msgs · `user_stop` · 1095s).
- 로그 전수: `[sim=task_055#s626729]` **362 라인** / `[sim=task_055#s373753]` **382 라인**.
- 변이 = 정본 `t2_forensic.mutation_diff` 만 사용(손 비교기 0 · C583ⓐ). 인용은 전부 축자.
- **수리 실행 없음**(코드 무수정 · 처방은 후보로만). gold(`reward_info`)는 진단용으로만 읽었다([[23]]).
- 대조: 기준선 `bank_t7328_halfA_20260819r`(sha 상이) · 선행 보고서 `t7336_tasks/T7336_TASK_055.md`.
- 엔진: `git_commit=fc0055dc…` · agent `Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8` · user-sim `openrouter/openai/gpt-5.2`(temp 0·reasoning low).

---

## 0. 채점 축 — **DB**(ACTION 아님)

```
trial 0: reward_basis=["DB"]  reward_breakdown={"DB":0.0}  db_check={"db_match":false,"db_reward":0.0}
trial 1: reward_basis=["DB"]  reward_breakdown={"DB":0.0}  db_check={"db_match":false,"db_reward":0.0}
env_assertions=[]  ·  nl_assertions=null  ·  communicate_checks=null
```

⇒ **DB-해시 축**이다. `action_checks` 는 진단 보조일 뿐 성적이 아니다([[69]]).
따라서 판정 단위는 *gold 변이 집합의 정확한 재현* 하나이고, **여분 write(EXTRA)도 누락과 동일하게 치명적**이다 — trial 0 의 `close_bank_account_7392` 가 그 예다.

`action_checks` 표(진단용):

| trial | unmatched action_id | matched |
|---|---|---|
| 0 | `055_4`(checking Purple) · `055_5`(savings Silver Plus) · `055_7`(deposit) | `055_0` `055_1` `055_2` `055_3` `055_6` |
| 1 | `055_5`(savings Silver Plus) · `055_7`(deposit) | `055_0` `055_1` `055_2` `055_3` **`055_4`** `055_6` |

---

## 1. 변이표 (`t2_forensic.mutation_diff` 정본)

gold 변이(write) 4칸:

| aid | tool | args |
|---|---|---|
| 055_0 | `log_verification` | Yuki Nakamura / 224959b99e / … / `2025-11-14 03:40:00 EST` |
| 055_4 | `open_bank_account_4821` | `{checking, "Purple Account"}` |
| 055_5 | `open_bank_account_4821` | `{savings, "Silver Plus Account"}` |
| 055_7 | `deposit_check_3847`(user) | `{account_id:"7e48bf3b0589cfad", check_amount:1500}` |

### trial 0 (`s626729`)

| 종류 | 내용 |
|---|---|
| MATCHED | `log_verification`(msg 23) |
| **MISSING** | 055_4 · 055_5 · 055_7 |
| **WRONGARG** | `open_bank_account_4821{checking,"Green Fee-Free Account"}`(msg 25·ok) · `open_bank_account_4821{savings,"Green Account (savings)"}`(msg 31·ok) · `open_bank_account_4821{savings,"Gold Account"}`(msg 53·ok) · `deposit_check_3847{account_id:"1bc7064aea2ca2d3",1500}`(msg 62·ok) |
| **EXTRA** | **`close_bank_account_7392{account_id:"0de0aa560c1cc942", reason:"Changing to a different account", waive_early_closure_fee:false}`**(msg 51·ok) |
| BLOCKED | `deposit_check_3847{1bc7064aea2ca2d3,1500}`(msg 56·user) → `Error: Tool 'deposit_check_3847' has not been given to you by the agent.` |
| DUP | 없음 |

### trial 1 (`s373753`)

| 종류 | 내용 |
|---|---|
| MATCHED | `log_verification`(msg 19) · **`open_bank_account_4821{checking,"Purple Account"}`(msg 33)** |
| **MISSING** | 055_5 · 055_7 |
| **WRONGARG** | `open_bank_account_4821{savings,"Platinum Plus Account"}`(msg 38·ok) · `deposit_check_3847{account_id:"58bfe7ce8f9bf643",1500}`(msg 72·ok) |
| EXTRA | 없음 |
| BLOCKED | `open_bank_account_4821{user_id,account_class}`(msg 23) → `Error: Invalid arguments: … missing 1 required positional argument: 'account_type'` · `deposit_check_3847`(msg 56·60·user) → `… has not been given to you by the agent.` |
| DUP | 없음 |

### WRONGARG 필드별 대조

| 칸 | 보낸 값 | gold | 상이 필드 |
|---|---|---|---|
| 055_4 (t0) | user_id ✓ / `checking` ✓ / **`Green Fee-Free Account`** | `Purple Account` | **account_class** |
| 055_5 (t0) | user_id ✓ / `savings` ✓ / **`Green Account (savings)`** → **`Gold Account`** | `Silver Plus Account` | **account_class** |
| 055_5 (t1) | user_id ✓ / `savings` ✓ / **`Platinum Plus Account`** | `Silver Plus Account` | **account_class** |
| 055_7 (t0) | check_amount `1500` ✓ / account_id **`1bc7064aea2ca2d3`** | `7e48bf3b0589cfad` | account_id |
| 055_7 (t1) | check_amount `1500` ✓ / account_id **`58bfe7ce8f9bf643`** | `7e48bf3b0589cfad` | account_id |

**`account_id` 는 클래스의 함수다**(선행 보고서 T7336 §1 관찰 그대로 재현). 두 trial 모두 수표는 정상 입금됐고(잔액 $1,500 확인) **넣을 통만 틀렸다** ⇒ 055_7 의 MISSING 은 055_5 의 순수 downstream 이다.

⇒ **양 trial 공통 단일 결손 = `open_bank_account_4821.account_class` 오선택**, 그중 **savings 축은 2/2 실패**. 이 한 필드가 DB 해시를 결정한다.

---

## 2. step-by-step 결정 지점 추적

### 2.1 trial 0 (`s626729`) — checking·savings 둘 다 상실 + EXTRA close

**[msg 3] 손님이 checking 요구 7항을 발화**(축자):
> "I travel internationally pretty often … I end up using ATMs abroad constantly and those fees really eat into my budget. Is there any way to get those **reimbursed**? Also I hate those **foreign transaction fees** … I can keep maybe three to four thousand … I sometimes get paid in **yen or euros**. It'd be nice to **hold those currencies** … any "**green**" options? Also I've heard about **premium** accounts…"

**[결정 지점 ① · msg 4→7] checking 클래스 확정.** 도구 2개 병렬 호출 뒤 곧장 추천이다.
msg 6 = 우리 A2 comparator `get_checking_atm_fee_totals` 반환(축자 발췌):
```
- Green Fee-Free Account: out-of-network ATM total $0.00 | foreign ATM total $0.00
- Purple Account:         out-of-network ATM total $120.00 | foreign ATM total $0.00
…
This tool does not pick a class: compare the totals yourself, verify the remaining candidate's
eligibility and non-ATM terms in its cited source docs, and confirm with the customer.
```
msg 7 축자:
> "for your travel needs, the **Green Fee-Free Account** is still a strong option as it has no out-of-network or foreign ATM fees … For the savings account, the **Evergreen Account** remains a good choice due to its eco-friendly initiatives and competitive interest rates."

- 필요한 값이 그 시점 문맥에 **실재했는가**: **아니다**. 이 궤적 전체에서 `"Purple"` 이 나온 곳은 **위 comparator 표 한 줄뿐**이고(전수 grep 1건), `doc_checking_accounts_purple_account_001`(멀티통화 지갑 Yes·FX 0%·글로벌 ATM 환급 $30/월·라운지 6회)은 **한 번도 회수되지 않았다**.
- 어떤 read 를 했나: comparator 1회 + `check_card_application_fit` 1회. **계좌 문서 read 0회.**
- comparator 표가 보여주는 **유일한 축(ATM 비용)에서 gold Purple 은 Green Fee-Free 에 지배당한다**($120/$0 vs $0/$0). 손님이 말한 나머지 축(멀티통화·FX·환급·premium)은 이 표에 **없다**.
- READ-FIRST 문면(*"verify … in its cited source docs, and confirm with the customer"*)은 **발화했고 무시됐다**.
- savings(`Evergreen Account`)는 **손님이 savings 요구를 한 마디도 하기 전에**(요구는 msg 34) 근거 0으로 지어졌다.

**[msg 8] user-sim 은 오도하지 않았다** — 에이전트가 준 이름을 그대로 되받았을 뿐이다:
> "Okay, if you think the **Green Fee-Free Account** is the best fit for my travel needs, let's do it."

**[msg 19~24] 검증 정상.** `verify_identity` → VERIFIED(4값) → `get_current_time` → `log_verification` = **gold 055_0 일치**.

**[결정 지점 ② · msg 25] checking write.** `call_discoverable_agent_tool{open_bank_account_4821, {checking,"Green Fee-Free Account"}}` → 성공. **gold 055_4 상실 확정.**
직전 우리 층 로그: `[T2_PREKB] require_before post-check (replay-safe) fam=open_bank_account (missing get_all_user_accounts_by_user_id)` — **막지 못하고 사후 통지**만 했다(gold 055_1/055_2 는 이 시점 미실행).

**[결정 지점 ③ · msg 29~31] savings write.** 우리 층이 먼저 개입한다(로그 축자):
```
[T2_WRITE_ARG_ENUM] 저장된 축 결정 동봉 group=savings_accounts (247자)
[T2_WRITE_ARG_ENUM] deny val='Evergreen Account' group=savings_accounts (후보 9)
```
동봉된 247자 = A2 `decided_by_docs_text` 축자:
> "A separate check was run on the policy documents on record, with this request and nothing else from this conversation in front of it. … **It answers: Gold Account.**"

모델의 반응(msg 31 축자):
> "It appears there was an error because the account class name for the savings account was not recognized. The correct name for the eco-friendly savings account is the **Green Account (savings)**."
→ `{savings,"Green Account (savings)"}` 개설. **우리 deny 는 *"목록 밖"* 만 말했고 *"요구를 만족하는가"* 는 아무도 묻지 않았다** ⇒ 모델은 목록 안의 이웃 값으로 갈아탔다([[64]] 형태 위반: 무엇이 틀렸나만 있고 무엇을 하면 풀리나가 없다).

**[msg 34] 손님이 savings 요구 7항을 처음 발화** — **이미 개설된 뒤**다(축자):
> "I tap into my savings **3–4 times a week** … is there any **relationship bonus rate** … I can keep **$5k–$7k max** … I **need daily compounding** (non-negotiable) … I use **out-of-network ATMs a lot** … at least **3% APY** … anything **green** or a little more **premium**? A friend mentioned **Gold and Green** accounts"

**[결정 지점 ④ · msg 35~39] 재선택.** `KB_search_bm25`(msg 35) 회수 top-1 = `doc_savings_accounts_gold_account_013`("Gold Rewards Card holder benefits" · score 25.3). **개인 savings 스펙 문서는 이 회수집합에 Green(savings) 두 편뿐이고 `Silver Plus` 스펙은 없다.**
그 다음 우리 층(로그):
```
[T2_WRITE_ARG_ENUM] 저장된 축 결정 동봉 group=savings_accounts (247자)   ← "It answers: Gold Account."
[T2_WRITE_ARG_ENUM] deny val='Gold Account (savings)' group=savings_accounts (후보 9)
```
msg 39 축자:
> "1. **Close the Green Account (savings)** … 2. **Open the Gold Account (savings)** … 3. Deposit the check"

⇒ **EXTRA `close_bank_account_7392` 의 발생 지점**이다. 이름 날조를 우리가 3회 교정했고(`[T2_PROV] name-arg → registry message … val=close_bank_account_4822 / close_bank_account_4821`) msg 51 에서 close 성공, msg 53 에서 `{savings,"Gold Account"}` 개설. **gold 055_5 상실 확정 + EXTRA 1건 추가.**

**[msg 55~63] 수표 입금.** msg 55 에서 도구명·인자를 정확히 안내 → msg 56 손님 호출이 `has not been given` 로 실패 → msg 59 `give_discoverable_user_tool(deposit_check_3847)` 실행(**gold 055_6 일치**) → msg 62 손님 입금 성공. **다만 통이 `1bc7064aea2ca2d3`(Gold)** 라 055_7 은 상실.
직전 우리 층: `[T2_USER_TOOL_NOTE] pre-give note: deposit_check_3847`(정방향) · 그 뒤 `[T2_GIVE_EXEC] nudge idle=['deposit_check_3847']`(이미 실행된 도구를 미실행으로 판정 — 선행 §4-F 와 동일 형상, 손실 0).

### 2.2 trial 1 (`s373753`) — checking 은 샀고 savings 만 상실

**[분기점 = msg 4]** t0 과 갈리는 **유일한 지점**이다. 병렬 호출 중 하나가 `KB_search_dense{"travel checking account options", k:5}` 다.
msg 6 회수 5위 = `doc_checking_accounts_purple_account_001` 축자:
```
| Foreign transaction fee | 0% |
| Foreign ATM withdrawal fee | $0.00 |
| Global ATM fee rebates | Up to $30 per month |
| Multi-currency wallet | Yes |
| Supported wallet currencies | 30 |
| Complimentary airport lounge visits | 6 per year |
| APY boost with Platinum Plus Savings | +0.3% |
```
**msg 7 축자**: "…so we can open the **Purple Account** and the **Platinum Plus Savings** account for you."

- checking 정답은 **문서가 문맥에 도달했기 때문**이다. 같은 comparator 표(msg 5)는 t0 과 **바이트 동일**하고 여전히 Green Fee-Free 를 argmin 으로 보여주지만, 실제 문서가 함께 있으면 모델은 문서를 골랐다.
- **동시에 savings 오답의 씨앗도 이 문서다** — `"APY boost with Platinum Plus Savings"` 한 줄이 savings 클래스 후보로 전사됐다. 이 시점 손님은 savings 요구를 **한 마디도** 하지 않았다(요구는 msg 37).

**[msg 9~20] 검증.** msg 9 는 DOB 를 빈 문자열로 보내 `NOT_VERIFIED`(레코드 미조회) → msg 13 `get_user_information_by_email` → msg 15 재시도 VERIFIED(3값) → msg 19 `log_verification` **gold 055_0 일치**.

**[msg 23] env 오류 1건**: `open_bank_account_4821` 을 `account_type` 없이 호출 → `Error: Invalid arguments: … missing 1 required positional argument: 'account_type'`. 모델이 msg 33 에서 자력 교정.

**[결정 지점 ① · msg 33] checking write.** `{checking,"Purple Account"}` 성공 = **gold 055_4 일치**(t7336 t1 에서 잃었던 칸을 이번엔 샀다).

**[msg 37] 손님이 savings 요구 7항 발화**(축자):
> "I tap into my savings **3–4 times a week** … any **bonus rate** for having multiple accounts? … about **$5–6k**, maybe **$7k max** … I **need daily compounding** — that's non-negotiable … **out-of-network ATMs** … at least **3% APY** … A friend mentioned **Gold and Green** accounts … If **Platinum Plus Savings** is the *one* you recommend for all that, then yes—please open it."

**[결정 지점 ② · msg 38] savings write — 요구를 받은 바로 다음 턴인데 read 0회다.**
우리 층(로그):
```
[T2_WRITE_ARG_ENUM] 저장된 축 결정 동봉 group=savings_accounts (247자)   ← "It answers: Gold Account."
[T2_WRITE_ARG_ENUM] deny val='Platinum Plus Savings' group=savings_accounts (후보 9)
```
msg 38 축자:
> "It appears there was an error due to the incorrect account class name. The correct name for the Platinum Plus Savings account is "**Platinum Plus Account**"."
→ `{savings,"Platinum Plus Account"}` 개설. **gold 055_5 상실 확정.**

- 필요한 값이 문맥에 실재했는가: **아니다.** 이 궤적 전체에서 문자열 `"Silver Plus"` 는 **0건**이다(전수 grep). 개인 savings 스펙 문서도 **0편** 회수됐다.
- 우리 deny 는 여기서도 **철자 교정기로만 작동**했다: 요구 7항 중 어느 것도 대조되지 않았고, 모델은 `Platinum Plus Savings` → `Platinum Plus Account` 로 이름만 바꿔 통과했다.
- msg 41 은 근거 0의 6항 충족 선언(날조): *"It offers no fees for frequent withdrawals. There is a bonus rate … It supports daily compounding … at least a 3% APY … It aligns with your eco-conscious preferences."*

**[msg 42~72] 수표 입금 — 30턴 소모.** msg 43/45/47/49 는 포털·앱 절차 날조("since we're dealing with hypothetical scenarios and specific account names like "Purple Account" … aren't standard offerings"). msg 50 손님이 `mobile_check_deposit` 호출 → `Unknown discoverable tool`. msg 53 `KB_search_dense{"mobile check deposit tool"}` → top-1 이 정답을 축자로 준다:
> "To help a customer deposit a check, **give them the deposit_check_3847 tool**."

그럼에도 msg 55·59 는 give 없이 안내만. msg 63/65/69 에서 give 3회 실행(**gold 055_6 일치**), msg 72 손님 입금 성공 — 통은 `58bfe7ce8f9bf643`(Platinum Plus)라 055_7 상실.
msg 77 에는 `shell{"command": "grep -r '58bfe7ce8f9bf643' ."}` 라는 무의미한 호출이 1건 있다(`No matches found`).

### 2.3 분기점 요약

| | trial 0 | trial 1 |
|---|---|---|
| msg 4 병렬 호출 | `check_card_application_fit` + comparator | comparator + **`KB_search_dense`** |
| checking 근거 | comparator 표뿐(문서 0) | **purple_account_001 문서 도달** |
| checking 결과 | Green Fee-Free ✗ | **Purple ✓** |
| savings 근거 | Gold Rewards Card 문서(스펙 아님) | **문서 0**(Purple 문서의 boost 줄에서 전사) |
| savings 결과 | Green ✗ → close → Gold ✗ (+EXTRA) | Platinum Plus ✗ |
| `Silver Plus` 문맥 등장 | 3건(전부 스펙 아님) | **0건** |
| give(`deposit_check_3847`) | 실행 ✓ | 실행 ✓ |
| reward | 0.0 | 0.0 |

**갈린 곳은 msg 4 한 수**(dense 검색 병렬 여부)이고, **savings 는 두 궤적 동일 기전으로 실패**했다 ⇒ 어느 쪽에서도 DB 일치는 도달 불가였다.

---

## 3. 레버 발화표 (해당 sim 줄만 · t0 362라인 / t1 382라인 전수)

| 레버 | t0 | t1 | 판정 |
|---|---|---|---|
| `T2_SEARCH_AGENT` | 6줄(turn 2·4·7) | 5줄(turn 2·7·21) | **발화·결정 전량 오답** |
| `T2_DOCDECIDE` | `'Blue Account'`(checking·turn2) · `'Gold Account'`(savings·turn4) · `'Sky Blue'`(business_checking·turn7) | `'Blue Account'`(turn2) · `'Gold Account'`(turn7) | **오발화 2/2 축** (gold = Purple / Silver Plus) |
| `T2_SEARCH_ON_PROCEED` | 3회(247·247·243자) | 3회(247·247·**13998**자) | 발화 · **예산 3 전량 소진** |
| `T2_CP2_CLOBBER` | 2회(247→243 · 243→4536) | 0 | **결함**: savings 결정문이 미소비 상태로 폐기 |
| `T2_SEARCH_REARM` | 0 | **1회**(turn 21·checking·purple_account·13998자) | **신규 발화** — 이미 맞은 축에 마지막 예산 소모(§4-B) |
| `T2_WRITE_ARG_ENUM` | 동봉 3회 + deny 2회(`'Evergreen Account'`·`'Gold Account (savings)'`×2) | 동봉 1회 + deny 1회(`'Platinum Plus Savings'`) | **발화·소속만 검사** → 철자 교정기로 작동 |
| `T2_SUB_REQUIREMENT` | 0 | 0 | **미발화**(런 전체 halfA/halfB 각 0줄 · 기본 OFF) — §4-A 핵심 |
| `T2_VERDICT_CARRY` / `T2_VERDICT_GATE` | 0 | 0 | **미발화**(런 전체 0줄 · 기본 OFF) |
| `T2_OWNERSHIP_FIX` | 0 | **1회 `suppressed(user-side)`** | **수리 성공**(§5) |
| `T2_GIVE_QUOTE` | 0 | 1회 `retract=0 (give_present_after_reask=1)` | **오발화이나 무해화**(t7336 은 `retract=1`) |
| `T2_USER_TOOL_NOTE` | 1회(정확) | 1회(정확) | 정방향 |
| `T2_GIVE_EXEC` | 1회 `idle=['deposit_check_3847']` | 1회 동일 | **오발화 의심**(이미 실행된 도구) · 손실 0 |
| `T2_TOOL_SIGNATURE` | deny 1 | deny 3 | give 채널 지연 |
| `T2_PROV` + registry message | 4회(close 이름 날조 교정 3) | 2회 | 정방향(이름) · **선택 정확성과 무관** |
| `T2_PREKB require_before` | 1회 post-check | 1회 post-check | **발화하나 write 이후** — 못 막음 |
| `T2_FORCE_ACTION` | 6회 | 4회 | 실행은 밀었으나 **선택 레버 아님** |
| `T2_ARG_DOC_SUB` | 3회(`spend_category→'travel'`) | 3회 | 발화·중립(카드 축) |
| `T2_VALUE_FORMULA` | 1회 드롭 | 0 | 중립 |
| `T2_CLAIMPROV` | 37줄(`kind-index rescued` 4) | 다수 | 발화·중립 |
| READ-FIRST(comparator return_template) | 발화·**무시** | 발화·**무시**(문서가 따로 도달해 결과만 정답) | §4-C |
| `T2_SG_DOCS` | 0 | 0 | **구조적 미발화** — `isolate.docs` 는 `get_correct_savings_apy` 등 scaffold-GET 전용이고 `open_bank_account_4821` 은 env discoverable 이라 관할 밖(런 전체 5줄은 093/094 뿐) |
| `T2_REQUIRE_DOC_DELIVER` | 0 | 0 | **구조적 미발화** — A2 `require_doc_before.tools` = 이관 도구 4종뿐(§4-D) |
| `T2_PIN_READ` · `T2_DEMANDED_STEP` · `T2_FOLLOWUP` | 0 | 0 | 미발화(런 전체로는 8·12·22줄 — 이 태스크 관할 밖) |
| `FAB_STRIP` · `T2_ARG_PRODUCERS` · `T2_ACT_DEMAND` | 0 | 0 | 런 전체 0줄(OFF) |

---

## 4. 원인 확정 (4주체 귀속)

### A. savings 클래스 오선택 — **our_layer(1차) + model(2차)** · 2/2 궤적 · **CONFIRMED**

**A-1. 축이 요구 발화 전에 소진된다.** 로그 축자:
```
t0: [T2_SEARCH_AGENT] group=checking_accounts … turn=2 → [T2_DOCDECIDE] → 'Blue Account'
    [T2_SEARCH_AGENT] group=savings_accounts  … turn=4 → [T2_DOCDECIDE] → 'Gold Account'
t1: 동일 값 · turn=2 / turn=7
```
`turn = len(messages)`. 손님의 savings 요구는 **t0 msg 34 · t1 msg 37** 이다.
코드: `t2_gate_patch.py:3218` `_g = next((g for g in _gs if g not in _done and g not in _degen), None)` — 군마다 **1회·영구 소진**이고 술어에 *"그 군의 요구가 발화됐는가"* 가 없다.

**A-2. 격리 서브에 손님 요구가 실리지 않았다 — 이미 격리로 측정된 결손이다.**
`t2_gate_patch.py:3413-3414`:
```python
_reqs = []
if (os.environ.get("T2_SUB_REQUIREMENT") == "1"
        or os.environ.get("T2_VERDICT_CARRY") == "1") and _po.get("requirement_prompt"):
```
두 플래그 모두 **런 전체 로그 0줄** ⇒ `_reqs` 는 빈 리스트 ⇒ `decide_from_docs`(`t2_search.py:717`)에 들어간 `ask` 는 `decide_candidates_text` 한 줄뿐이다.
같은 파일 `t2_gate_patch.py:3402-3404` 주석 축자:
> "x343(n=24=8×3·블록 편차 0): 이 서브가 문서+후보줄만 받으면 `Gold Account` **24/24 오답**, 손님 요구 메시지를 축자로 받으면 `Silver Plus` **24/24 정답**, 무관한 요구를 주면 **0/24**(부정통제 통과)."

**t7346 의 `[T2_DOCDECIDE] → 'Gold Account'`(양 trial)는 그 오답 조건의 지문과 축자 일치한다.**

**A-3. 그 오답이 write 결정점에 권위 문장으로 재제시된다.**
`t2_gate_patch.py:3511` `_out = _po["decided_by_docs_text"].format(choice=_choice)` · A2(`a2/banking_knowledge.specific.json`·`.gate.json` 동일) 축자:
> "A separate check was run on the policy documents on record, **with this request and nothing else from this conversation in front of it**. … It answers: {choice}."

로그 `[T2_WRITE_ARG_ENUM] 저장된 축 결정 동봉 group=savings_accounts (247자)` — t0 **3회** · t1 **1회**.
문면은 *"이 요청을 앞에 놓고 점검했다"* 고 말하지만 **실제로 서브가 받은 것은 후보 목록뿐**이다 ⇒ [[25]] *우리 도구는 100% 정답 의무* 위반이고, 유일한 근거원을 오염시킨다.
t7336 에서는 모델이 이 값을 따르지 않았으나 **t7346 t0 에서는 따랐다** — msg 39 `"Close the Green Account (savings)… Open the Gold Account (savings)"` → msg 51 EXTRA close + msg 53 Gold 개설. (모델 자신의 msg 35 검색 top-1 도 Gold 문서였으므로 **단독 원인은 아니다 → 기여 CONFIRMED · 단독 인과는 UNPROVEN**.)

**A-4. 미소비 배달물 폐기(t0).** `t2_gate_patch.py:4468 _cp2_assign` 단일 슬롯:
```
[T2_CP2_CLOBBER] SEARCH_ON_PROCEED 가 미소비 배달물 247자를 버리고 243자로 덮어씀
[T2_CP2_CLOBBER] VIEW_FB 가 미소비 배달물 243자를 버리고 4536자로 덮어씀
```
t0 의 savings 결정문은 **한 번도 소비되지 않은 채** business_checking 결정문에 덮였다.

**A-5. 배달 예산 3이 요구 이전에 전량 소진된다.** `t2_gate_patch.py:8859` `getattr(self, "_t2_searchagent_fired", 0) < 3`(같은 카운터를 6676·8056·8971 이 공유).
- t0: turn 2(checking) · turn 4(savings) · turn 7(**business_checking_accounts** — 손님이 요청한 적 없는 군). 세 번째 단위는 `[T2_DOCGROUP] raw='checking_accounts savings_accounts business_checking_accounts credit_cards'` 로 군이 4개로 부풀면서 **낭비**됐다.
- t1: turn 2 · turn 7 · turn 21(rearm·checking).
⇒ 손님이 savings 요구를 말한 시점(t0 msg 34 / t1 msg 37)에 이 채널은 **예산 0**이다. 로그상 이후 `T2_SEARCH_AGENT` 0줄.

**A-6. 모델 몫(2차).** 요구를 듣기 전에 write 로 갔고(t0 msg 25·31 / t1 msg 33), 요구를 들은 뒤에도 savings 스펙 문서를 **한 편도 읽지 않고**(t1 은 read 0회) 근거 없는 6항 충족을 선언했다(t0 msg 41 형·t1 msg 41 축자).

### B. `T2_SEARCH_REARM`(t7336 처방 3) — 발화했으나 못 샀다 · **our_layer(설계) · CONFIRMED**

t1 로그 축자:
```
[T2_SEARCH_REARM] group=checking_accounts 신규 대상 purple_account (기배달 blue_account) — 소진 해제·문서 델타
[T2_SEARCH_REARM] group=checking_accounts 델타 배달 13998자 (문서 12·뺀 것 0) turn=21
[T2_DECISION_CARRY] 이 턴 재생성 버퍼에 부착 (13998자)
```
- 재무장 대상 술어는 **모델이 이미 말한 계열**(`purple_account`)을 따라간다 ⇒ **이미 맞힌 축**에 발화했다. turn=21 은 checking write(msg 33)보다도 **앞**이라 새 정보를 사지 못했다.
- 그리고 이것이 **세 번째이자 마지막 예산 단위**였다(`_t2_searchagent_fired` 3/3). 손님의 savings 요구가 온 msg 37 시점에 savings 축 재무장은 **예산이 없어 구조적으로 불가**했다.
- t0 에서는 **0줄** — 두 궤적 합쳐 **savings 축 재무장 0회**.

### C. checking 오선택(t0) — **model(1차) · our_layer 기여 PLAUSIBLE**

우리 comparator 표의 두 열 argmin 은 Green Fee-Free($0/$0)이고 gold Purple($120/$0)은 그 축에서 지배당한다. t0 은 표 출력 직후 첫 문장에서 Green Fee-Free 를 채택했고 **문서 확인 0 · 손님 확인 0** 이었다. return_template 의 READ-FIRST 문면은 발화했고 무시됐다.
**교차 런 대조**(comparator 호출 sim): t7328 t0=Blue · t7328 t1=Green Fee-Free · t7336 t1=Green Fee-Free · t7346 t0=Green Fee-Free · t7346 t1=**Purple**(dense 문서 동반) · t7335 t0=Purple · t7336 t0=Purple(comparator 미호출·dense).
⇒ **문서가 도달한 sim 은 Purple, 표만 있는 sim 은 Green Fee-Free/Blue** 라는 방향이 n=7 에서 일관되지만, 표 자체의 해악은 격리로 안 쟀다 ⇒ **UNPROVEN 유지**([[18]] 격리 프로브 필요).

### D. 문서 본문 배달 레버의 관할 공백 — **our_layer(선언 미완결) · CONFIRMED**

이 런에서 **실제 문서 본문**을 결정점에 싣는 유일한 레버는 `T2_REQUIRE_DOC_DELIVER` 다(halfA 17줄 · 예: `deliver tool=transfer_to_human_agents docs=6 chars=16498`). 그러나 A2 선언(`a2/banking_knowledge.specific.json` · `.gate.json` 의 `require_doc_before.tools`)은
```json
["transfer_to_human_agents", "initial_transfer_to_human_agent_0218",
 "initial_transfer_to_human_agent_1822", "emergency_credit_bureau_incident_transfer_1114"]
```
**이관 도구 4종뿐**이다 ⇒ 055 의 전 결손이 걸린 `open_bank_account_4821` 은 관할 밖이고, 이 태스크에서 **구조적으로 0줄**이다. 같은 이유로 `T2_SG_DOCS`(scaffold-GET 전용)도 0줄이다.
결과: 055 의 결정점에 도달한 KB 재료는 **247자짜리 오답 결정문**뿐이었다.

### E. 수표 입금 채널 — **t7336 대비 수리됨 · our_layer 잔여만 남음**

t1 로그 축자: `[T2_OWNERSHIP_FIX] suppressed(user-side): give-name=mobile_deposit_check customer-side candidate(s) ['deposit_check_3847']`
⇒ t7336 §4-C 가 지목한 **거짓 부정-존재 단언**(*"there is no customer-side tool by that name on file"*)이 **손님-측 레지스트리 조회로 억제**됐다. t7336 t1 은 이 자리에서 give 를 영영 잃었는데 **t7346 은 양 trial 모두 gold 055_6 을 샀다**.
잔여: `T2_GIVE_QUOTE` 는 손님이 도구명을 축자로 요청했음에도 여전히 *"no verbatim customer span"* 으로 재질의를 냈고(`retract=0` 로 무해화), `T2_TOOL_SIGNATURE deny` 3회와 겹쳐 t1 의 give 착지가 **msg 53→63, 10턴** 걸렸다.

### F. env / user_sim

- **env**: t1 msg 25 `missing 1 required positional argument: 'account_type'`(모델 인자 누락) · `has not been given to you by the agent`(정확한 안내) — 오도 없음. `close_bank_account_7392` 는 요청대로 동작. **실패 원인 아님.**
- **user_sim**: 오도 없다. t0 msg 8 은 에이전트가 준 이름을 되받았을 뿐이고, t0 msg 34 · t1 msg 37 에서 요구 7항을 **명시적으로** 냈으며 *"If the Green Account (savings) you opened is truly the one best fit … then okay"* / *"If Platinum Plus Savings is the one you recommend for all that"* 로 **재검토 기회를 두 번 열어줬다**. t1 msg 52·58·62·68 은 도구 오류를 축자로 전달해 복구를 도왔다. **면책 사유 아님**([[21]]).

---

## 5. 선행 판정과 대조 (`t7336_tasks/T7336_TASK_055.md`)

| 항목 | t7328(기준선) | t7336 | **t7346** | 판정 |
|---|---|---|---|---|
| checking | t0 **Blue** ✗ / t1 Green Fee-Free ✗ | t0 Purple ✓ / t1 Green Fee-Free ✗ | t0 Green Fee-Free ✗ / t1 **Purple ✓** | **불안정(문맥-민감 픽) 유지** — 두 팔이 t7336 과 **뒤바뀜** |
| savings | Gold ✗ / Gold Plus ✗ | Green ✗ / Green+Gold ✗ | Green→Gold ✗ / Platinum Plus ✗ | **동일 결손·값만 이동 · 3런 6/6 실패** |
| `T2_DOCDECIDE` | (동일 계열) | `Blue`/`Gold` | **`Blue`/`Gold` 동일** | 상수 오답 재현 |
| give(`deposit_check_3847`) | t1 blocked | t0 지연 실행 / **t1 미실행** | **t0·t1 모두 실행 ✓** | **개선(§E 수리 효과)** |
| EXTRA write | 0 | 0 | **t0 `close_bank_account_7392` 1건** | **신규 해악** |
| reward | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 불변 |

**같은 원인인가?** — **savings 축은 같다.** t7336 §4-A 가 확정한 3요소(①요구 발화 전 축 소진 ②`T2_SUB_REQUIREMENT` OFF 로 x343 오답 조건 재현 ③오답의 권위형 재제시)가 t7346 에서 **한 글자도 바뀌지 않고 재현**됐다. 추가된 것은 **④예산 3이 요구 이전에 전량 소진**(A-5)과 **⑤ t0 에서 그 오답을 모델이 실제로 따라 EXTRA close 를 만들었다**(A-3)는 두 사실이다.

**달라진 것**: 수표 입금 축의 귀속이 **our_layer → 해소**로 이동했다(§E). t7336 처방 4가 실물로 들어갔고 작동했다.

### 직전 런 이후 수리·레버의 개입 여부와 "왜 못 샀나"

| t7336 처방 | 이 런에 들어갔나 | 이 궤적에 개입했나 | 왜 못 샀나 |
|---|---|---|---|
| ① 축 소진 술어에 요구 발화 조건 | **미적용**(`t2_gate_patch.py:3218` 불변) | — | savings 축이 여전히 turn 4/7 에 소진 |
| ② `T2_SUB_REQUIREMENT` ON | **미적용**(런 전체 0줄) | — | 서브가 요구 없이 결정 → `Gold Account` 상수 오답 |
| ③ `T2_SEARCH_REARM` A/B | **적용·ON** | **개입(t1 1회)** | **이미 맞은 checking 축**에 마지막 예산을 썼다(§4-B) |
| ④ `feedback_user_tool_is_agents` 부정단언 제거 | **적용** | **개입(t1 `suppressed(user-side)`)** | **샀다** — 055_6 을 두 trial 모두 회수. 단 DB 축은 클래스가 결정하므로 reward 불변 |
| ⑤ `T2_VERDICT_GATE`/`CARRY` A/B | **미적용**(런 전체 0줄) | — | 요구충돌 값을 되돌릴 유일 게이트가 없었다 |
| ⑥ `arg_producers` 코퍼스 완결 | 미적용 | — | 이 런에서는 해당 형상 미발생 |
| ⑦ `T2_GIVE_QUOTE` 술어 조정 | 미적용 | **개입(t1)** | 여전히 오발화 · `retract=0` 로 손실만 면함 |

**핵심 답**: 이 궤적에 실제로 닿은 신규 수리는 ③④⑦ 셋이고, **④만 gold 칸을 샀다(055_6)**. 그러나 채점 축이 DB 해시라 `account_class` 두 칸을 못 사면 reward 는 0 이다. **savings 축을 살 수 있었던 스위치 ②⑤와 술어 수정 ①은 전부 미적용**이었고, ③은 적용됐으나 **예산을 잘못된 축에 썼다**.

---

## 6. 처방 후보 (제안만 · 구현 금지)

우선순위는 **삭제가 아니라 조정**([[70]])이고 **새 결정론이 아니라 전달 복구**([[62]] ②)다. 이 태스크는 격리에서 **된다**(x343 24/24) ⇒ 레버는 전달뿐이다.

1. **[P0·전달·짝] ①+② 를 한 짝으로 적용한다.**
   - `t2_gate_patch.py:3218` 축 소진 술어에 *"이 군에 대해 원문 검산을 통과한 손님 인용 수 ≥ 1"* 을 AND 로 건다(fail-open: 대화 끝까지 인용이 0이면 종전대로 소진). ⛔조건은 **도메인 일반 닫힌 술어**여야 한다 — 태스크 id·군 이름으로 켜면 [[05]] 위반.
   - `T2_SUB_REQUIREMENT=1`. **②만 켜면 여전히 요구 없는 turn 4/7 에 소진되어 이득 0** 이다(이 런이 그 반증이다: ③만 켰더니 예산만 탔다).

2. **[P0·계기] `decided_by_docs_text` 의 거짓 서술을 제거한다.**
   현행 문면 *"with this request and nothing else from this conversation in front of it"* 는 `_reqs` 가 비었을 때 **거짓**이다(`t2_gate_patch.py:3511`·A2 두 파일). 요구가 실리지 않은 결정은 **그 사실을 문면에 적거나, 아예 동봉하지 않는다**([[25]] 우리 계기 100% 정답 의무). t7346 t0 의 EXTRA close 는 이 문장을 모델이 믿은 결과다.

3. **[P0·예산] 배달 예산 3의 **사용처**를 요구-발화 이후로 예약한다.**
   `t2_gate_patch.py:8859`. 총량은 그대로 두고(새 판단 기구 0), *"해당 군의 검산 통과 인용이 아직 0 인 결정점"* 에서는 예산을 **쓰지 않는다**. t0 의 세 번째 단위가 손님이 요청한 적 없는 `business_checking_accounts` 에 나간 것도 이 술어로 함께 닫힌다.

4. **[P1·선언 완결] `require_doc_before.tools` 에 `open_bank_account_4821` 계열을 넣는다**([[72]] 1회 저작·코퍼스 단위).
   문서 본문을 결정점에 싣는 유일한 레버가 이관 4종만 관할한다(§4-D). deny 는 하지 말고 **배달만**(x93 근거 그대로). ±는 문맥 +16k자/회·지연이므로 `T2_REQUIRE_DOC_DELIVER_CAP` 과 함께 A/B.

5. **[P1·게이트] `T2_VERDICT_CARRY`/`T2_VERDICT_GATE` A/B.**
   `T2_WRITE_ARG_ENUM` 이 소속만 보고 통과시킨 값(`Green Account (savings)`·`Gold Account`·`Platinum Plus Account`)을 손님 요구와 대조해 되돌릴 유일한 자리다. 단 `verdict_lines`(`t2_search.py:660`)는 `req_block` 이 비면 `skip="no-template-or-req"` 로 침묵하므로 **②와 짝**이다.

6. **[P1·조정] `T2_SEARCH_REARM` 의 대상 선택 술어.**
   현행은 *모델이 이미 말한 계열*을 따라가 이미 맞은 축을 재배달했다. *"검산 통과 인용이 새로 생긴 군"* 을 우선순위로 두면 t1 msg 37 이후 savings 축이 대상이 된다. **끄지 말고 조정**([[70]]).

7. **[P2·절충] `T2_GIVE_QUOTE` 술어를 *손님 발화에 그 도구명이 축자로 있는가*(C45 동형·닫힌 술어)로 바꾼다.** t7336 처방 7 그대로 유효 — 이 런에서도 오발화했다(손실은 0).

8. **[P2·격리 선행] comparator(`get_checking_atm_fee_totals`) 표의 ± 를 격리로 잰다**([[18]]).
   n=7 교차 런에서 *"표만 있으면 Green Fee-Free / 문서가 있으면 Purple"* 방향이 보이지만 인과는 UNPROVEN 이다. 표를 끄는 것이 아니라 **표 옆에 후보 문서를 같이 싣는 조건**을 A/B 해야 한다([[70]] ③ 분해).
