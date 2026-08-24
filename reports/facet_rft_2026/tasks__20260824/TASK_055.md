# t7348 task_055 per-step 포렌식 — 2026-08-24

- 런: `bank_t7348_halfA_20260824`(results/log 전부 로컬 gz · SSH 0 · git 무접촉 · 커밋 0).
- sim 2개: **trial 0 = `task_055#s626729`**(seed 626729 · 83 msgs · `user_stop` · 1061s · reward **0.0**) /
  **trial 1 = `task_055#s373753`**(seed 373753 · 54 msgs · `user_stop` · 600s · reward **0.0**).
- 로그 전수 = `[sim=task_055#…]` 794 라인(s626729 / s373753 두 태그).
- 변이 = 정본 `t2_forensic.mutation_diff` 만 사용(손 비교기 0 · C583ⓐ). 인용은 전부 축자.
- gold(`reward_info`)는 **진단용으로만** 읽었다([[23]]). **수리 실행 없음** — 처방은 후보로만.
- 엔진: agent `openai/Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8` · user-sim/judge `openrouter/openai/gpt-5.2` ·
  `retrieval_config=alltools` · GUIDED/UNIFIED/E-PLAN/SCAFFOLD-GET/PRE-ACTION-KB ON.
- 대조: 직전 런 `bank_t7346_halfA_20260822`(같은 계열) · 선행 보고서
  `tasks__20260822/TASK_055.md` · `t7336_tasks/T7336_TASK_055.md` · `FAILURE_MASTER__20260822.md` ·
  `STATE_OF_PLAY_2026_08_23.md`.

---

## 0. 채점 축 — **DB**(ACTION 아님)

```
trial 0: reward_basis=["DB"]  reward_breakdown={"DB":0.0}  db_check={"db_match":false,"db_reward":0.0}
trial 1: reward_basis=["DB"]  reward_breakdown={"DB":0.0}  db_check={"db_match":false,"db_reward":0.0}
env_assertions=[]  ·  nl_assertions=null  ·  communicate_checks=null
```

⇒ **DB-해시 축**이다. `action_checks` 는 진단 보조일 뿐 성적이 아니다([[69]]).
판정 단위 = *gold 변이 집합의 정확한 재현* 하나이며 EXTRA 도 누락과 동등하게 치명적이다.

`action_checks`(진단용):

| trial | unmatched action_id | matched |
|---|---|---|
| 0 | `055_4`(checking Purple) · `055_5`(savings Silver Plus) · `055_7`(deposit) | `055_0` `055_1` `055_2` `055_3` `055_6` |
| 1 | `055_4` · `055_5` · **`055_6`**(give 미실행) · `055_7` | `055_0` `055_1` `055_2` `055_3` |

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
| MATCHED | `log_verification`(msg 20) |
| **MISSING** | 055_4 · 055_5 · 055_7 |
| **WRONGARG** | `open_bank_account_4821{checking,"Green Fee-Free Account"}`(msg 24·ok) · `open_bank_account_4821{savings,"Green Account (savings)"}`(msg 39·ok) · `deposit_check_3847{account_id:"0de0aa560c1cc942",1500}`(msg 63·ok) |
| EXTRA | **없음**(t7346 t0 의 `close_bank_account_7392` EXTRA 는 재발 안 함) |
| BLOCKED | `submit_transaction{0de0aa560c1cc942,1500,check_deposit}`(msg 45·user) → `Error: Unknown discoverable tool 'submit_transaction'.` · `deposit_check_3847{0de0aa560c1cc942,1500}`(msg 53·user) → `Error: Tool 'deposit_check_3847' has not been given to you by the agent.` |
| DUP | 없음 |

### trial 1 (`s373753`)

| 종류 | 내용 |
|---|---|
| MATCHED | `log_verification`(msg 16) |
| **MISSING** | 055_4 · 055_5 · 055_7 |
| **WRONGARG** | `open_bank_account_4821{checking,"Evergreen Account"}`(msg 30·ok) |
| EXTRA | 없음 |
| BLOCKED | `open_bank_account_4821{checking,"Evergreen Account"}` **×3**(msg 34·36·38) → `Failed to open account: Account ID 'e1241970db2a3277' may already exist.` |
| DUP | 없음(`mutation_diff` 는 blocked 로 분류 — 성공 1 + 차단 3) |

### WRONGARG 필드별 대조

| 칸 | 보낸 값 | gold | 상이 필드 |
|---|---|---|---|
| 055_4 (t0) | `user_id` ✓ / `checking` ✓ / **`Green Fee-Free Account`** | `Purple Account` | **`account_class`** |
| 055_5 (t0) | `user_id` ✓ / `savings` ✓ / **`Green Account (savings)`** | `Silver Plus Account` | **`account_class`** |
| 055_7 (t0) | `check_amount` 1500 ✓ / `account_id` **`0de0aa560c1cc942`** | `7e48bf3b0589cfad` | `account_id` |
| 055_4 (t1) | `user_id` ✓ / `checking` ✓ / **`Evergreen Account`** | `Purple Account` | **`account_class`** |
| 055_5 (t1) | — (호출 자체 없음) | `Silver Plus Account` | 전부 |

`account_id` 는 클래스의 함수다 ⇒ **055_7 의 MISSING 은 055_5 의 순수 downstream**.
⇒ **양 trial 공통 단일 결손 = `open_bank_account_4821.account_class` 오선택**, 이번 런은
**checking 축도 savings 축도 2/2 실패**(t7346 t1 이 샀던 `Purple Account` 를 **되팔았다**).

---

## 2. ★스텝 표 — trial 0 (`s626729`)

`step` = messages 인덱스(= `turn_idx`). `[LOG]` 접두는 궤적에 안 남는 우리-층 stderr 이벤트로,
직후 생성 턴에 귀속시켰다.

| step | role | what | actor | kind | code_path | evidence(축자) | consequence | reward_causal |
|---|---|---|---|---|---|---|---|---|
| 1 | user | 두 계좌(checking+savings) 개설 의사만 밝힘 · **요건 0개** | user_sim | 요구 개시 | — | `"I'm looking to open a second checking account, plus a savings account."` | 이 한 줄이 아래 ②의 유일한 입력이 된다 | 무관 |
| ①<br>[LOG]<br>turn=2 | our_layer | checking 축 **결정 서브 실행** — 손님 요건이 나오기 **전**에, 그리고 `_dask` 가 손님 발화를 **후보 명단으로 치환**한 채로 | our_layer | 격리 서브 오재료 | `t2_gate_patch.py:3787-3789`(`_reqs` 가드 = `T2_SUB_REQUIREMENT`/`T2_VERDICT_CARRY` 필요·둘 다 0) + `:3814-3817`(`_dask = _ctpl.format(candidates=…)`) + A3 `policy_ontology.decide_candidates_text`(`{ask}` 자리 없음) | `[T2_SEARCH_AGENT] group=checking_accounts · 문서 113 … turn=2 (… 대화텍스트 1)` → `[T2_DOCDECIDE] → 'Blue Account'` · A3 축자 `"The full official names on file for this group are: {candidates}. Answer with exactly one name from that list, verbatim, and nothing else."` | 서브가 낼 수 있는 답이 gold(`Purple Account`)와 무관해진다. 코드 주석 자신이 이 결손을 이미 격리로 쟀다: `x343(n=24) … 문서+후보줄만 받으면 'Gold Account' 24/24 오답, 손님 요구를 축자로 받으면 'Silver Plus' 24/24 정답` | 중 |
| ①′<br>[LOG] | our_layer | 같은 자리에서 **축을 영구 소진**(`_t2_search_done`)하고 247자 결정문을 CP2 슬롯에 적재 | our_layer | 예산·순서 | `t2_gate_patch.py:3878-3882`(`_done.add(_g)`) · `:9416-9448`(`T2_SEARCH_ON_PROCEED`) | `[T2_SEARCH_AGENT] 축 처리 완료: checking_accounts (남은 축 savings_accounts)` · `[T2_SEARCH_ON_PROCEED] deny 아님 · 재료 247자 배달` | 손님이 msg 3 에서 요건을 말한 **뒤**에는 checking 축 재결정 경로가 없다 | 준강 |
| 3 | user | checking 요건 7항 발화 | user_sim | 요구 확정 | — | `"I end up using ATMs abroad constantly … Any way to get those reimbursed? Also I hate those foreign transaction fees … I sometimes get paid in yen or euros. It'd be nice to hold those currencies … any "green" options? And I've heard about premium accounts…"` | gold `Purple Account` 를 유일하게 지목하는 축(멀티통화 지갑·FX 0%·글로벌 ATM 환급·라운지)이 여기 다 있다 | 무관(정상) |
| 4 | assistant | 도구 **1개만** 호출 — 계좌 클래스 문서 read **0회** | model | 근거 미확보 | — | `get_checking_atm_fee_totals({"months":"12","withdrawals_per_month":"4","withdrawal_amount":"100"})` | 결정 재료가 ATM 비용 한 축으로 축소 | 준강 |
| 5 | tool | 우리 A2 comparator 가 10개 클래스 × **ATM 비용 2열**만 반환 + READ-FIRST 문면 | our_layer | 재료 형태(축 축소) | A2 `scaffold_get_tools[get_checking_atm_fee_totals]`(A3 `isolate.docs` 미선언 → `[T2_SG_TRACE] get_checking_atm_fee_totals: iso=False mode=None`) | `- Green Fee-Free Account: out-of-network ATM total $0.00 \| foreign ATM total $0.00` / `- Purple Account: out-of-network ATM total $120.00 \| foreign ATM total $0.00` / `This tool does not pick a class: compare the totals yourself, verify the remaining candidate's eligibility and non-ATM terms in its cited source docs, and confirm with the customer.` | **이 표의 유일한 축에서 gold Purple 은 Green Fee-Free 에 지배당한다**($120/$0 vs $0/$0). 손님이 말한 멀티통화·환급·premium 은 표에 없다 | 준강 |
| **6** | **assistant** | **checking 클래스 확정 — `Green Fee-Free Account`** | **model** | **오선택(단일 축 닫기)** | — | `"the **Green Fee-Free Account** seems to be the best fit … **Currency Holding**: While the account doesn't directly support holding foreign currencies, you can manage your budget effectively by minimizing conversion costs"` | **gold 055_4 상실 결정 지점.** 모델은 손님 요건 중 *멀티통화 미충족* 을 **자인하고도** 그 후보를 유지했다. READ-FIRST 문면(*"verify … in its cited source docs"*)은 **발화했고 무시**됐다 | **강** |
| 7 | user | 에이전트가 준 이름을 그대로 되받음 | user_sim | — | — | `"if you think that's the one that fits all my needs, let's do it!"` | user-sim 오도 **없음**([[21]] 면책 불가) | 무관 |
| 10–21 | assistant/tool | `verify_identity`(NOT_VERIFIED ×2 → 조회 → VERIFIED) → `get_current_time` → `log_verification` | model+our_layer | 정상 | `T2_SCAFFOLD_GET verify_identity` | `VERIFIED — 2 identity values match the account record (date_of_birth, email)` | **gold 055_0 일치** | 무관(양) |
| ②<br>[LOG]<br>turn=6 | our_layer | savings 축 결정 서브 실행 — **손님이 savings 요건을 한 마디도 하기 전**(요건은 msg 32) | our_layer | 순서·예산 | 위 ①과 동일 경로 | `[T2_SEARCH_AGENT] group=savings_accounts · 문서 92 … turn=6 (… 대화텍스트 3)` → `[T2_DOCDECIDE] → 'Gold Account'` · `[T2_SEARCH_AGENT] 축 처리 완료: savings_accounts (남은 축 없음)` | savings 축도 **msg 6 시점에 영구 소진**. gold `Silver Plus Account` 와 무관한 답이 저장됨 | 준강 |
| ②′<br>[LOG] | our_layer | CP2 단일 슬롯이 미소비 checking 결정문을 폐기 | our_layer | 근거창 파괴 | `t2_gate_patch.py:5014`(`_cp2_assign`) · `go_stack.sh:387 export T2_CP2_QUEUE=0` | `[T2_CP2_CLOBBER] SEARCH_ON_PROCEED 가 미소비 배달물 247자를 버리고 247자로 덮어씀` | **버린 내용물이 오답('Blue Account')이라 이 sim 에서 실제 손실은 0** — 계기 결함이지 인과는 아니다 | 무관 |
| ③<br>[LOG]<br>turn=8 | our_layer | `T2_SEARCH_REARM` 이 **모델이 방금 스스로 고른 오답 계열**의 문서 7편(7,413자)을 배달하고, 미소비 savings 결정문을 덮어씀 | our_layer | 자기확증 배달 | `t2_gate_patch.py:3457-3503`(`_rearm_subjects` 술어 ⑵ = *계열 표시명이 이후 user/assistant 발화에 축자 등장*) · `:3616-3620` | `[T2_SEARCH_REARM] group=checking_accounts 신규 대상 green_fee-free_account (기배달 blue_account) — 소진 해제·문서 델타` · `[T2_SEARCH_REARM] … 델타 배달 7413자 (문서 7)` · `[T2_CP2_CLOBBER] … 247자를 버리고 7413자로 덮어씀` · `[T2_DECISION_CARRY] 이 턴 재생성 버퍼에 부착 (7413자)` | **결정 직전에 오답 클래스의 근거를 7,413자 실어 확증**했다. 재수요 술어가 *모델 자신의 발화*를 보므로 미언급 정답(`purple_account`)은 구조적으로 배달될 수 없다 | 준강 |
| 22·24 | assistant | `unlock_discoverable_agent_tool(open_bank_account_4821)` **중복 2회** 후 write | model | 실행 | — | msg 24 `call_discoverable_agent_tool({"agent_tool_name":"open_bank_account_4821","arguments":"{\"user_id\":\"224959b99e\",\"account_type\":\"checking\",\"account_class\":\"Green Fee-Free Account\"}"})` | **gold 055_4 상실 확정.** 우리 층은 사후 통지만: `[T2_PREKB] require_before post-check (replay-safe) fam=open_bank_account (missing get_all_user_accounts_by_user_id)` | 강 |
| 24 | our_layer | `T2_WRITE_ARG_ENUM` **미발화** — 보낸 값이 공식명 집합 **안**이라 술어가 통과 | our_layer | 게이트 상한 | A2 `write_arg_enum[0].feedback`(철자 프레이밍) · `t2_gate_patch.py:10032-10190` | 로그에 `[T2_WRITE_ARG_ENUM]` **0줄**(t0 전체) | 이 게이트는 *공식명인가* 만 묻고 *요건을 만족하는가* 는 못 묻는다 ⇒ **오답이 공식명이면 통과** | 준강 |
| 27–30 | assistant/tool | 도구명 찾으려 `KB_search_bm25` 2회 → 우리 층이 두 번째를 스텁 | our_layer | 중복차단 | `[T2_READ_DEDUP] stub tool=KB_search_bm25` | `[DUPLICATE-READ] This exact call … was already executed earlier` | 정상 억제(손실 0) | 무관 |
| 32 | user | savings 요건 7항 발화 — **축은 이미 turn 6 에 소진됨** | user_sim | 요구 확정 | — | `"I need daily compounding — that's non-negotiable … I want to earn at least 3% APY … is there something "green"? And I'd prefer something that feels a little premium … Can you just tell me which one is best for me"` | gold `Silver Plus Account`(FAQ: Tier1 3.0% / Tier2 4.5%)를 지목하는 축이 여기 다 있다. 그러나 `[T2_SEARCH_AGENT]` 는 이 이후 savings 에 대해 **영원히 침묵** | 무관(정상) |
| 33–34 | assistant/tool | bm25 1회 — 회수 1위 `doc_business_checking_accounts_world_blue_007`(**사업자 체킹**), 2위 `doc_bank_accounts_bank_accounts_(general)_046`(APY boost 정책) | model+env | 회수 실패 | — | 개인 savings **스펙 문서 0편** 회수 | 결정점에 gold 스펙이 **없다** — t7346 t0 과 갈리는 지점(그 런은 `silver_plus_account_002` 가 결정점에 도달했었다) | 준강 |
| 35 | assistant | 근거 0의 **조합 추천** 날조 | model | 날조 | — | `"the **Evergreen Account** linked with a **Green Account (savings)** seems to be the best fit … Linking a Green Account (savings) with the Evergreen Account provides a **+0.55% APY boost**"` | +0.55% 수치의 출처가 회수 문서에 없다 | 중 |
| 36–37 | user→assistant | 손님이 *"the ONE savings account"* 를 재요구 → 모델이 `Green Account (savings)` 로 축소, 6항 충족 선언 | model | 날조 | — | msg 37 `"**Daily Compounding:** The Green Account (savings) offers daily compounding … **Premium Feel:** … giving it a premium feel."` | 3% APY 요건은 아예 언급 안 함 | 강 |
| **39** | **assistant** | **savings write — `{savings, "Green Account (savings)"}`** | **model** | **오선택** | — | `call_discoverable_agent_tool({"agent_tool_name":"open_bank_account_4821","arguments":"{\"user_id\":\"224959b99e\",\"account_type\":\"savings\",\"account_class\":\"Green Account (savings)\"}"})` → `Account ID: 0de0aa560c1cc942` | **gold 055_5 상실 확정 + 055_7 통 확정.** 여기서도 `WRITE_ARG_ENUM` 미발화(공식명) | **강** |
| 44 | assistant | 존재하지 않는 모바일앱 절차 + 도구명 `submit_transaction` 안내 | model+our_layer | 오지목 | A2 `action_tools` 에 `submit_transaction` 등재(`a2/banking_knowledge.gate.json` · `a2/banking_knowledge.settings.json`) → `t2_gate_patch.py:8716-8718`(`_uacts`) | `[T2_RESOLVE] user-action instruct target=submit_transaction` ×4 · msg 44 `"you will need to use the \`submit_transaction\` tool in the mobile app"` | msg 45 손님 호출 → **env** `Error: Unknown discoverable tool 'submit_transaction'.` ⇒ 5턴 소모 | 약 |
| 50–52 | assistant/tool | bm25 `"deposit check into savings account"` → `doc_bank_accounts_bank_accounts_(general)_011` 회수 → 도구명 교정 | model | 자력 회복 | — | msg 52 `"you will need to use the \`deposit_check_3847\` tool right here in this conversation"` | 정답 경로 복귀 | 무관 |
| 53–59 | user/assistant | 손님 호출 차단(미부여) → `give_discoverable_user_tool` **2회** 실행 | model | 실행(중복) | — | `Error: Tool 'deposit_check_3847' has not been given to you by the agent.` → msg 56·58 give | **gold 055_6 일치** | 무관(양) |
| 63–64 | user/tool | 손님이 **`0de0aa560c1cc942`(Green savings)** 로 $1,500 입금 성공 | model(통 지정) | downstream | — | `Check deposit processed! - Account: 0de0aa560c1cc942 - Check Amount: $1500.00` | **gold 055_7 상실** — 금액·도구는 맞고 **통만 틀렸다** | 강(=055_5 종속) |
| 71 | tool | **gold 스펙이 뒤늦게 도달** — `doc_savings_accounts_silver_plus_account_005`(`Tier 1: 3.0% / Tier 2: 4.5%`) + 페어링 문서(`4. Blue Account (checking) + Silver Plus Account (savings)`) | env | 지연 회수 | — | 위 축자 | **write(msg 39) 보다 32 메시지 늦다** ⇒ 이번 런의 055#0 은 *"정답이 앞에 있는데 안 썼다"* 가 **아니라** *"정답이 결정 후에 왔다"* | 무관(사후) |
| 78–81 | assistant | gold 055_1/055_2(unlock+call `get_all_user_accounts_by_user_id_3847`) 를 **맨 끝에** 실행하고 성공 선언 | model | 순서 | — | msg 81 `"it confirms that the $1,500 check has been deposited into your Green A…"` | action_checks 상 matched 이지만 DB 축 기여 0 | 무관 |

---

## 3. ★스텝 표 — trial 1 (`s373753`)

| step | role | what | actor | kind | code_path | evidence(축자) | consequence | reward_causal |
|---|---|---|---|---|---|---|---|---|
| 1 | user | t0 과 **바이트 동일**한 개시 발화 | user_sim | 요구 개시 | — | `"I'm looking to open a second checking account, plus a savings account."` | — | 무관 |
| ①<br>[LOG]<br>turn=2 | our_layer | checking 결정 서브 — t0 과 동일하게 **요건 전·후보명단 치환** 상태 | our_layer | 격리 서브 오재료 | `t2_gate_patch.py:3787-3789` · `:3814-3817` | `[T2_SEARCH_AGENT] group=checking_accounts · 문서 113 … turn=2 (… 대화텍스트 1)` → `[T2_DOCDECIDE] → 'Blue Account'` | 양 trial **동일 오답**(재현성 2/2) | 중 |
| 3 | user | checking 요건 7항(t0 과 같은 내용) | user_sim | 요구 확정 | — | `"I sometimes get paid in yen or euros. It'd be nice to be able to hold those currencies …"` | — | 무관 |
| **②**<br>**[LOG]**<br>**turn=4** | **our_layer** | **모델 초안이 부르려던 `get_checking_atm_fee_totals` 를 우리 재생성 스택이 지웠다 — 착지한 msg 4 는 도구 0개 산문** | **our_layer** | **재생성 파괴** | `t2_gate_patch.py:8592`(`T2_MATERIAL_GATE`) · `T2_CLAIM_PROV` regen 경로(`go_stack.sh:162 export T2_CLAIM_PROV=1`) | `[T2_MATERIAL_GATE] stop=other_lever(gate) turn=4 calls=check_card_application_fit,get_checking_atm_fee_totals,unlock_discoverable_agent_tool pending=247 axes=8 prose=False` · `[T2_CLAIMPROV] regen tool_calls=[]` · `[T2_STACK] audit … chose=[('fb','check_card_application_fit')] differs=True` | **이 trial 에서 comparator 는 끝내 실행되지 않는다**(`[T2_SCAFFOLD_GET]` 전수 = `verify_identity` ×2 뿐). 착지본 msg 4 = `"Let's start by verifying your identity … Could you please provide me with your full name, user ID, address, email, phone number, and date of birth?"` | **준강** |
| ③<br>[LOG]<br>turn=4 | our_layer | savings 축까지 **turn 4 에 소진**(요건은 끝내 발화조차 안 됨) | our_layer | 순서·예산 | `t2_gate_patch.py:3878-3882` | `[T2_SEARCH_AGENT] group=savings_accounts … turn=4 (… 대화텍스트 2)` → `[T2_DOCDECIDE] → 'Gold Account'` · 이후 전 구간 `[T2_SEARCH_AGENT] 요청 축 checking_accounts,savings_accounts 모두 처리됨 — 침묵` | savings KB 채널이 msg 4 에 구조적으로 닫힌다 | 준강 |
| 5–17 | user/assistant/tool | 손님이 5값 제시 → `verify_identity` NOT_VERIFIED → `get_user_information_by_name` → VERIFIED(4값) → `get_current_time` → `log_verification` | model | 정상 | — | `VERIFIED — 4 identity values match the account record (date_of_birth, address, phone_number, email)` | **gold 055_0 일치** | 무관(양) |
| **18–19** | **assistant/tool** | **이 sim 의 유일한 KB 접근 = `shell grep -ri 'travel' .`** — 문서 본문이 아니라 **grep 라인 조각**이 돌아오고, 상위 결과는 전부 `business_checking`/`business_credit_cards` | **model** | **회수 형태 오선택** | — | `shell({"command":"grep -ri 'travel' ."})` → `./doc_business_checking_accounts_world_blue_005.md:# World Blue: Traveling Abroad with Your Business Account` … | 개인 checking 스펙 **0편**. `purple_account` 는 이 출력에 **없다**(궤적 전체 `Purple` 등장 = msg 19 의 1건뿐) | **강** |
| 20–23 | assistant/tool | `unlock`+`call get_all_user_accounts_by_user_id_3847` | model | 정상 | — | `class: checking / level: Green Account / current_holdings: 2850.00` | **gold 055_1·055_2 일치** | 무관(양) |
| **24** | **assistant** | **존재하지 않는 상품명 2개를 날조하고 스펙 5줄까지 붙임** | **model** | **날조** | — | `"I recommend considering the **EcoTravel Checking Account** and the **High Yield Eco Savings Account**. The EcoTravel Checking Account … offers: - No foreign transaction fees - Reimbursement for foreign ATM fees - The ability to hold multiple currencies (including yen and euros)"` | 서술된 4가지 속성은 **실제로는 `Purple Account` 의 스펙**이다 — 요건 매칭은 맞고 **이름만 지어냈다**. `T2_FAB_STRIP` 은 write 호출만 보므로 산문 날조에 미발화 | **강** |
| 25 | user | 날조된 이름을 그대로 승인 | user_sim | — | — | `"yes, I want to open the **EcoTravel Checking Account** first."` | user-sim 오도 없음(에이전트 발화 반사) | 무관 |
| 26 | assistant | 쓸 계획 없는 `apply_checking_account_credit_5829` unlock(정책 위반) | model | 과잉 | — | `unlock_discoverable_agent_tool({"agent_tool_name":"apply_checking_account_credit_5829"})` | 정책 축자 `"do not unlock tools you do not plan to use"` 위반 · 손실 0 | 무관 |
| **④**<br>**[LOG]** | **our_layer** | **열거 게이트가 오답을 잡고 공식명 10개 + 서브 결정문을 동봉했는데, 문면이 *철자 교정* 프레이밍이다** | **our_layer** | **[[64]] 형태 결함** | A2 `write_arg_enum[0].feedback` · `t2_gate_patch.py:10162-10187`(`_dsav` 동봉) | `[T2_WRITE_ARG_ENUM] 저장된 축 결정 동봉 group=checking_accounts (247자)` · `[T2_WRITE_ARG_ENUM] deny val='EcoTravel Checking Account' group=checking_accounts (후보 10 · 원장 1)` · 문면 축자 `"Error: [OFFICIAL-NAME] '{val}' is not one of the official {arg} names on file. The names on file for {group} are: {candidates}. **Choose the one you meant, verbatim**, and call the tool again."` · 동봉물 축자 `"It answers: Blue Account."` | *"the one you meant"* 는 **요건 재대조가 아니라 최근접 이름 치환**을 지시한다. 후보 10개에는 **`Purple Account` 가 들어 있었다** | **준강** |
| **30** | **assistant** | **`EcoTravel` → `Evergreen Account` 로 이름만 갈아끼워 write** | **model** | **오선택(최근접 치환)** | — | `"It seems there was a mistake in the account class name. The correct account class for the EcoTravel Checking Account is the **Evergreen Account**."` → `{checking, "Evergreen Account"}` 성공(`e1241970db2a3277`) | **gold 055_4 상실 확정.** 우리가 준 10개 공식명도, 우리 서브의 답(`Blue Account`)도 따르지 않았다 ⇒ *전달로 안 닫힌 잔여*가 이 스텝에는 실재한다 | **강** |
| 32 | assistant | 방금 개설을 손님에게 알리지 않고 savings 로 넘어가려 함 | model | 보고 누락 | — | `"Let's proceed with opening the High Yield Eco Savings Account next."` | 손님이 성공 사실을 모른 채 재요청(msg 33) | 중 |
| 33 | user | *"still want to open the EcoTravel Checking Account first"* | user_sim | 재요구 | — | `"I think we got a bit crossed up — I still want to open the **EcoTravel Checking Account first**"` | 손님은 오도하지 않았다 — 에이전트가 결과를 안 알려서 생긴 재요구다([[21]] 흡수 대상) | 중 |
| 34·36·38 | assistant/tool | **동일 인자 3회 재호출** → env 가 3회 동일 거절 | model / env | 반복+env 문면 | — | 모델: `{checking,"Evergreen Account"}` ×3 · env: `Failed to open account: Account ID 'e1241970db2a3277' may already exist.` ×3 | env 문면이 **원인을 틀리게 말한다**(실제 원인 = 이미 개설 성공 · 문면 = *account ID 충돌*) ⇒ 모델이 "충돌을 풀면 된다"로 오독 | 준강 |
| **40–51** | **assistant** | **savings 를 시작조차 못 하고 human transfer 로 종료** | model | 사임 | — | msg 40 `"Let me escalate this to a human agent"` → msg 50 `transfer_to_human_agents({... "reason": account_ownership_dispute})` → `Transfer successful (reason: account_ownership_dispute).` | **gold 055_5·055_6·055_7 전부 MISSING**. 사임 사유(`account_ownership_dispute`)도 사실과 무관 | **강** |
| ⑤<br>[LOG]<br>turn=42·44 | our_layer | **이 sim 유일한 문서 본문 배달(16,498자·6편)이 transfer 도구에만 발화** — 정작 오답이 난 `open_bank_account_4821` 은 관할 밖 | our_layer | 표적 미도달 | A3 `require_doc_before.tools = ["transfer_to_human_agents","initial_transfer_to_human_agent_0218","initial_transfer_to_human_agent_1822","emergency_credit_bureau_incident_transfer_1114"]` · `t2_gate_patch.py:3289-3371` | `[T2_REQUIRE_DOC_DELIVER] deliver tool=transfer_to_human_agents docs=6 chars=16498 turn=42 fired=1/3` · `[T2_REQUIRE_DOC_DELIVER] 이 턴 재생성 버퍼에 부착 (16498자)` | 결정 write 에는 0자, 실패 뒤 사임 write 에는 16,498자 — **예산이 정확히 반대로 쓰였다** | 준강 |

### 3.1 분기점

| | trial 0 | trial 1 |
|---|---|---|
| msg 4 초안 | comparator 1건 → **착지** | `check_card_application_fit`+**comparator**+`unlock` → **우리 regen 이 도구 0개 산문으로 대체** |
| checking 근거 | comparator 표(ATM 2열) | **아무것도 없음**(grep 라인 조각뿐) |
| checking 결과 | `Green Fee-Free Account` ✗ | `Evergreen Account` ✗ (t7346 t1 은 `Purple` ✓ 였다 = **회귀**) |
| savings 진행 | 개설(오답) + 입금(오통) | **미개설** — transfer 종료 |
| gold 스펙 도달 | msg 71(write 32턴 **후**) | **0건**(`Silver Plus` 전 궤적 0회) |
| our_layer 문서 배달 | 7,413자(**오답 계열**·write 직전) | 16,498자(**transfer 도구**·실패 후) |

⇒ **분기 지점은 msg 4 하나**이고, 그 자리에서 갈린 것은 모델이 아니라 **우리 재생성 스택**이다.

---

## 4. 레버 발화표 (이 sim 2개 한정 · 794 로그 라인 전수)

| 레버 | t0 | t1 | 판정 | 근거(축자) |
|---|---|---|---|---|
| `T2_SEARCH_AGENT` / `T2_DOCDECIDE` | 5 / 2 | 8 / 3 | **오발화** — 요건 전 실행 + 손님 발화를 후보명단으로 치환 | `turn=2 (… 대화텍스트 1)` → `→ 'Blue Account'` (양 trial 동일) · savings `→ 'Gold Account'` |
| `T2_SEARCH_ON_PROCEED` | 3 | 3 | **발화·객체 오류** — 247자 **결정문**만 배달(문서 본문 아님) | `deny 아님 · 재료 247자 배달` ×3 |
| `T2_PROCEED_DOCBODY` | — | — | **미발화(미등재)** — `go_stack.sh` 에 export 없음 | 코드 주석 축자: `t7303 로그 직독: 이 자리에 배달되던 서브 결정 자체가 오답이었다(055 양팔 DOCDECIDE → 'Blue Account'·gold Purple)` (`t2_gate_patch.py:9433-9437`) |
| `T2_SUB_REQUIREMENT` / `T2_VERDICT_CARRY` | — | — | **미발화(플래그 0)** — `_reqs` 빈 채로 `_dask` 치환 | `go_stack.sh:439 export T2_VERDICT_CARRY=0` · `T2_SUB_REQUIREMENT` export 없음. **단 [[FAILURE_MASTER §3.4]] 가 "재론 금지"(C508 라이브 A/B 0/8↔0/8)로 확정한 항목** |
| `T2_SEARCH_REARM` | 2 | 0 | **오발화(자기확증)** — 모델이 방금 고른 오답 계열을 배달 | `신규 대상 green_fee-free_account (기배달 blue_account)` · `델타 배달 7413자` |
| `T2_CP2_CLOBBER` / `T2_CP2_QUEUE` | 2 | 0 | **발화(계기)·손실 0** — 버린 페이로드가 둘 다 오답 | `미소비 배달물 247자를 버리고 247자로 덮어씀` / `… 7413자로 덮어씀` · `go_stack.sh:387 T2_CP2_QUEUE=0` |
| `T2_WRITE_ARG_ENUM` | **0회** | 6회(3 deny) | t0 **미발화(정상 술어)** — 오답이 공식명이라 통과 / t1 **발화·무시** | t1 `deny val='EcoTravel Checking Account' (후보 10 · 원장 1)` → 모델 `Evergreen Account` |
| `T2_REQUIRE_DOC_DELIVER` | **0줄** | 9줄(전부 transfer) | **표적 미도달** — `require_doc_before.tools` = 이관 4종뿐 | `deliver tool=transfer_to_human_agents docs=6 chars=16498` · **t7346 판정 그대로 재현** |
| `T2_SG_DOCS`(ON) | 0줄 | 0줄 | **死선언** — comparator 에 `isolate.docs` 미선언 | `[T2_SG_TRACE] get_checking_atm_fee_totals: iso=False mode=None ctx=['months','withdrawal_amount','withdrawals_per_month']` |
| `T2_ARG_DOC_SUB`(ON) | 4 | 2 | **축 무관 발화** | `spend_category: None -> 'travel'` — `check_card_application_fit`(신용카드 축)용 |
| `T2_PIN_READ` / `T2_PIN_READ_STEPS` | 0 | 0 | **미발화** | `[T2_PIN_READ]` 0줄 |
| `T2_DEMANDED_STEP` | 0 | 0 | **미발화** | 0줄 |
| `T2_ACT_DEMAND` | 0 | 0 | **미발화**(go_stack 미등재) | 0줄 |
| `T2_CLAIM_PROV` | 58 | 42 | **발화 · t1 에서 음(−)** | `regen tool_calls=[]` — comparator 초안을 산문으로 대체 |
| `T2_FOLLOWUP_*`(ON) | 0 | 0 | **미발화** | `[T2_FOLLOWUP]` 0줄 |
| `T2_ARG_PRODUCERS`(ON) | 0 | 0 | **미발화** | 0줄 |
| `T2_FAB_STRIP`(ON) | 0 | 0 | **범위 밖** — write 호출만 검사(`t2_gate_patch.py:11012`), 산문 상품명 날조 미포착 | t1 msg 24 `EcoTravel Checking Account` 통과 |
| READ-FIRST 문면(comparator 반환문) | 발화 | **미도달**(comparator 미실행) | t0 **발화·무시** | `verify the remaining candidate's eligibility and non-ATM terms in its cited source docs` → msg 6 이 곧장 확정 |
| `T2_MATERIAL_GATE` | 6(resolve_cap) | 2(other_lever) | **발화 · t1 turn 4 가 치명** | `stop=other_lever(gate) turn=4 calls=…,get_checking_atm_fee_totals,…` |
| `T2_RESOLVE` user-action | 4 | 0 | **오지목** — 존재하지 않는 `submit_transaction` 안내 | `user-action instruct target=submit_transaction` → env `Unknown discoverable tool` |
| `T2_PREKB` | 4 | 0 | **사후 통지만** | `require_before post-check (replay-safe) fam=open_bank_account (missing get_all_user_accounts_by_user_id)` |

---

## 5. 선행 판정과 대조 — **같은 원인인가 달라졌는가**

| 선행 진술(출처) | t7348 재현 여부 |
|---|---|
| `FAILURE_MASTER §2 표` — 055 = `account_class` 및 그 downstream `account_id` 단일 결손 · DB 축 | **그대로 재현.** 단 이번엔 **checking 축까지 2/2 실패**(t7346 t1 이 샀던 `Purple Account` 상실) ⇒ **action-cell 회귀** |
| `FAILURE_MASTER §3.5 경계 목록` — **055#0** *"정답 스펙 문서(`silver_plus_account_002`) + 공식명 9개가 동시에 앞에 있는데 'Silver Plus' 언급 0건"* | **반쪽만 재현·핵심은 달라졌다.** t7348 t0 에서 `Silver Plus` 스펙 문서는 **msg 71**(write msg 39 보다 **32 메시지 뒤**)에 처음 도달했고, 결정점(msg 33–37)의 회수집합에는 개인 savings 스펙이 **0편**이었다. 공식명 9개도 t0 에서는 `WRITE_ARG_ENUM` 이 **0회** 발화라 전달되지 않았다 ⇒ **t7348 t0 은 "경계"가 아니라 전달 결손**이다. *"정답이 앞에 있는데 안 쓴다"* 가 실제로 성립하는 곳은 **t1 msg 30 하나**(후보 10개 + `Blue Account` 동봉을 받고도 `Evergreen`) |
| `FAILURE_MASTER §반증표` — *"055 CP2 클로버로 savings 결정문이 한 번도 소비되지 않았다" = **REFUTED**(같은 247자가 `WRITE_ARG_ENUM` 경로로 t0 에서 3회 전달)* | **이 반증은 t7348 에 이월되지 않는다** — t7348 t0 의 `[T2_WRITE_ARG_ENUM]` 은 **0줄**(모델의 두 값이 모두 공식명이라 게이트 미발화)이므로 savings 결정문은 실제로 **한 번도 전달되지 않았다**. **다만 그 결정문이 오답(`Gold Account`)이라 reward 손실은 0** ⇒ 결론(클로버는 인과 아님)은 다른 경로로 유지 |
| `FAILURE_MASTER §3.2` — `T2_SEARCH_REARM` 057: *"모델 자신의 오답 발화를 재수요로 읽어 오답 계열 배달"* | **055 에서도 동일 형상 최초 확인.** t7348 t0: 기배달 `blue_account` → 신규 대상 `green_fee-free_account`(=모델이 msg 6 에 고른 오답) · 7,413자. **처방 B-10(화자 축 분리·부정문 배제)의 표적이 055 로 확장** |
| `FAILURE_MASTER §3.3` — `T2_REQUIRE_DOC_DELIVER` 055 0줄 · 사유 `require_doc_before.tools` = 이관 4종뿐 | **그대로 재현**(t0 0줄). t1 은 **사임 write 에만** 9줄 발화해 사유를 실물로 확증 |
| `FAILURE_MASTER §3.3` — `T2_SG_DOCS` 死선언(`isolate.docs` 미선언) | **055 comparator 에서도 재현** — `[T2_SG_TRACE] … iso=False mode=None` |
| `FAILURE_MASTER §3.4` — `T2_SUB_REQUIREMENT=0`/`T2_VERDICT_CARRY=0` **재론 금지**(C508 라이브 A/B 0/8↔0/8) | 기전 자체는 **이 런에서 코드로 확정**(`_dask` 치환 + `_reqs` 가드) 하지만, **승격은 금지 항목이므로 처방으로 올리지 않는다**. 여기서는 *"우리 서브의 오답은 배선상 필연"* 이라는 **진단**으로만 기록 |
| `FAILURE_MASTER §3.4` — `T2_PENDING_DISCOVERED` | 2026-08-23 R8 로 **제거 완료**(死레버). t7348 의 `submit_transaction` 오지목은 그 블록이 아니라 **정적 `action_tools` 선언**에서 나온다 ⇒ 원인 지점이 **이동** |
| `t7336_tasks/T7336_TASK_055` / `tasks__20260822/TASK_055` — *"comparator 표의 유일한 축에서 gold Purple 은 지배당한다"* | **t0 에서 바이트 동일 재현**(표 텍스트 동일·모델 결론 동일) |
| **신규**(선행에 없음) | ① **t1 turn 4 에서 우리 regen 이 comparator 호출을 삭제**(`regen tool_calls=[]`) — t7346 t1 에는 없던 개입이고, t7346 t1 이 `Purple` 을 산 경로(msg 4 병렬 호출)를 **정확히 그 자리에서 끊었다**. ② t1 모델이 **없는 상품명을 날조**(`EcoTravel Checking Account`)하는 형태는 055 에서 처음 관측 |

---

## 6. 원인 확정

**채점 단위**(①) = DB 해시. **변이 집합**(②) = 양 trial 공통 `open_bank_account_4821.account_class` MISSING/WRONGARG(+ 그 downstream `deposit_check_3847.account_id`).
**값의 KB 출처**(③) = `doc_checking_accounts_purple_account_*`(멀티통화 지갑·FX 0%·글로벌 ATM 환급 $30/월) 및 `doc_savings_accounts_silver_plus_account_005`(Tier1 3.0%/Tier2 4.5%) — 둘 다 A3 `doc_index` 에 **선언돼 있다**(checking 11계열 · savings 9계열에 `purple_account` · `silver_plus_account` 실재).
**우리 배선 발화**(④) = 아래 표.

| 순위 | 원인 | 주체 | 등급 | 근거 |
|---|---|---|---|---|
| 1 | **결정 서브가 손님 요건을 못 받는다** — `_dask` 가 후보 명단으로 치환되고 `_reqs` 는 플래그 0으로 빈다 ⇒ `DOCDECIDE` 4/4 오답(`Blue`×2·`Gold`×2·`Sky Blue`) | our_layer | **CONFIRMED** | `t2_gate_patch.py:3787-3789`·`:3814-3817` · A3 `decide_candidates_text`(`{ask}` 없음) · 코드 주석의 격리 실측 `x343 24/24` |
| 2 | **축 소진 술어에 요건 조건이 없다** — checking 은 turn 2, savings 는 turn 4~6 에 영구 소진. 손님 요건은 msg 3 / msg 32 에 온다 | our_layer | **CONFIRMED** | `t2_gate_patch.py:3878-3882` · `[T2_SEARCH_AGENT] 요청 축 … 모두 처리됨 — 침묵` |
| 3 | **t1: 우리 재생성이 유일한 근거 확보 호출을 삭제** | our_layer | **CONFIRMED** | `[T2_MATERIAL_GATE] … turn=4 calls=…get_checking_atm_fee_totals…` + `[T2_CLAIMPROV] regen tool_calls=[]` + 착지 msg 4 도구 0개 + `[T2_SCAFFOLD_GET]` 전수 `verify_identity` ×2 |
| 4 | **문서 배달 예산이 반대로 쓰인다** — 결정 write 0자 / 오답 계열 7,413자 / 사임 write 16,498자 | our_layer | **CONFIRMED** | `require_doc_before.tools`(이관 4종) · `_rearm_subjects` ⑵ |
| 5 | **모델이 단일 축으로 닫고, 미충족을 자인하고도 확정** (t0 msg 6) · **없는 상품명 날조**(t1 msg 24) · **10개 공식명을 앞에 두고 최근접 치환**(t1 msg 30) | model | **CONFIRMED** | 각 축자 인용 |
| 6 | env 문면이 원인을 틀리게 말한다 — `Account ID '…' may already exist` (실제 = 이미 개설 성공) · `Unknown discoverable tool 'submit_transaction'` | env | 관측 | msg 35/37/39 · msg 46 |
| 7 | user_sim | — | **면책 사유 없음** | 두 trial 모두 요건을 명료히 반복 발화했고 오도 0건. 에이전트 발화를 반사한 것뿐 ([[21]]) |

**한 문장**: `account_class` 를 고르는 자리에서 **요건과 재료가 한 번도 같은 창에 있지 못했다** — 우리 층이 요건이 오기 전에 축을 소진하고(2) 요건을 뺀 채 서브를 돌려(1) 오답을 저장했으며, t1 에서는 근거를 가지러 가던 호출까지 지웠고(3), 문서 예산은 결정점이 아닌 곳에 쓰였다(4). 그 위에서 모델은 한 축으로 닫거나 이름을 지어냈다(5).

---

## 7. 처방 후보 (제안뿐 · 구현·수리 실행 0)

> ⚠[[62]] 순서 준수: 아래 A·B·C 는 전부 **전달·순서 조정**이며 새 결정론기 0. D 는 계기.
> ⚠`T2_SUB_REQUIREMENT`/`T2_VERDICT_CARRY` 승격은 **FAILURE_MASTER §3.4 재론 금지 항목**이라 제외했다.

| id | 처방 | 표적 | 무엇을 파나([[70]]) |
|---|---|---|---|
| **A-1** | `_search_material` 축 소진(`_done.add`)을 **결정문 생성 시점이 아니라 배달-소비 시점**으로 옮긴다. 소비 전이면 축은 여전히 열린 것으로 둔다 | 원인 2 · 055 양 trial · 016·057 동형 | 재배달 증가 → 문맥 부피 |
| **A-2** | `require_doc_before.tools` 에 `open_bank_account_4821` 추가(선언 1줄 · 새 배선 0) — 이미 구현된 16,498자 배달기를 **결정 write** 로 돌린다 | 원인 4 · **055·079 공통 미도달** | 결정 턴 문맥 +16k자 · 지연 |
| **A-3** | `_rearm_subjects` 재수요 술어를 **user 발화로 한정**(assistant 자기발화 제외) = 기존 처방 **B-10** 의 최소판 | 원인 4(자기확증) · 057·016·055 | 재무장 빈도 감소(050#1·073#0 에서 양(+)이던 발화도 함께 준다 ⇒ ±필수) |
| **A-4** | `write_arg_enum.feedback` 에서 *"Choose the one you meant, verbatim"* 을 **요건 재대조 지시**로 교체([[64]] 형태: 무엇이 틀렸나 + 무엇을 하면 풀리나). 사례 열거 0·도메인 낱말 0 | 원인 5(t1 msg 30 최근접 치환) | 재시도 턴 증가 · abstain↑ |
| **A-5** | `get_checking_atm_fee_totals` 에 A3 `isolate.docs` 선언(死선언 `T2_SG_DOCS` 해소) — 표가 **ATM 2열만**인 상태로 결정점에 서지 않게 한다 | 원인 5(단일 축 닫기)·t0 msg 6 | comparator 반환 부피↑ |
| **B-1** | `T2_CLAIM_PROV` regen 이 **초안의 tool_calls 를 비우는 경우 반려**(도구 0개 재생성 금지 술어) | 원인 3 · t1 msg 4 | claim-날조 억제력 일부 상실 ⇒ 반드시 ± |
| **C-1** | A2 `action_tools` 에서 이 env 에 없는 `submit_transaction` 정리(또는 `_uacts` 를 레지스트리 ∩ 로 좁힘) | t0 msg 44–47 5턴 소모 | 손실 낮음(reward 헤드룸 0으로 기록됨) |
| **D-1** | `T2_DOCDECIDE` 로그에 **서브가 받은 `_dask` 앞 200자**를 함께 인쇄 — 지금은 *요건이 실렸는지* 를 로그만으로 알 수 없다(이번 건은 소스를 읽어야 판정됐다) | 계기 | 로그 부피 |
| **D-2** | 247자 결정문 배달에 **축 이름**을 찍는다 — `Blue Account`/`Gold Account` 가 **둘 다 정확히 247자**라 길이로 식별 불가(선행 §계기 함정 2 그대로 재현) | 계기 | 없음 |

---

## 8. 격리 프로브 제안(유료 런 전 · 무료)

- **x-055a**: `decide_from_docs` 를 ⓐ현행(`_dask`=후보명단) ⓑ`손님 요건 축자 + 후보명단` 두 팔로,
  checking/savings 각 n=8. **x343 이 이미 같은 구성을 쟀으므로 재실행이 아니라 축자 재사용**이 우선 —
  `reports/facet_rft_2026/` 에서 x343 산출물부터 grep 할 것([[74]] 검색 우선).
- **x-055b**: A-2(`require_doc_before` 에 `open_bank_account_4821` 추가) 의 **부정통제** =
  같은 16,498자를 **무관 문서**로 배달했을 때 매수 0 인지.
- 두 프로브 모두 **DB 축 reward 쌍 A/B + 태스크별 부호표**를 함께 낼 것([[70]] 판정 의무 3종).
