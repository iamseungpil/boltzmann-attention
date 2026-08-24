# TASK_016 — t7348 halfB(partial14) per-step 포렌식 (0/2 · 직전 런 t7346 도 0/2)

- 런: `bank_t7348_halfBpartial14_20260824`(results + log 전부 로컬 gz) · agent = `Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8`(temp 0) · user-sim = `openrouter/openai/gpt-5.2`.
- 대조(직전 런·같은 계열·**같은 seed 쌍**): `bank_t7346_halfB_20260822` — trial0=`s626729`, trial1=`s373753` 로 **양 런 동일**.
- 성적: t7346 **0.0 / 0.0** → t7348 **0.0 / 0.0**. reward 는 안 변했지만 **궤적은 크게 나빠졌다**(§6).
- 종료: 양 trial `user_stop`. trial0 = 74 msgs / 845s · trial1 = 60 msgs / 675s.
- ⚠**줄번호 주의**: 런 sha 는 리모트이고 아래 `code_path` 는 **로컬 워킹트리** 기준이다. 드리프트 대비로 **앵커 문자열을 함께** 적었다(`FAILURE_MASTER__20260822.md` §2.4-5 경고 반영).
- 방법: 변이 = 정본 `t2_forensic.mutation_diff`(손 비교기 0·[[67]]·C583ⓐ). 궤적 = `results.json.gz` 전 메시지 정독. 레버 = `.log.gz` 의 `[sim=task_016#s626729]`/`[sim=task_016#s373753]` **883라인 전수**. 인용 전부 축자.
- ⛔수리·코드 수정 실행 없음. gold(`reward_info`)는 진단용으로만([[23]]).

---

## §1 채점 축 — **DB** (ACTION 축 아님 · C583ⓖ)

`sim['reward_info']` 직독(양 trial 동일):

```
reward 0.0 · reward_basis ["DB"] · db_check {"db_match": false, "db_reward": 0.0}
reward_breakdown {"DB": 0.0} · env_assertions [] · nl_assertions null · communicate_checks null
```

**DB-해시 축**이다. `action_checks` 는 진단 보조로만 읽는다:

| action_id | requestor | name | action_match |
|---|---|---|---|
| 016_0 | assistant | `log_verification{Oluwaseun Adeyemi, 86e92f639e, …, 2025-11-14 03:40:00 EST}` | **true** (양 trial) |
| 016_1 | **user** | `submit_transaction{friend_user_5839, "Silver Rewards Card", "Best Buy", 750, "Shopping"}` | **false** (양 trial) |

★이 태스크의 구조: DB 델타의 마지막 한 칸은 **손님이 찍는다**. user-sim 스펙 축자:

> *"**Tool usage:** You have access to `submit_transaction`. ONLY use this tool if the agent has: - **Actually looked up your referrals and identified which one you're asking about** - Explained that your friend hasn't met a spending requirement yet - **Told you a specific dollar amount** your friend needs to spend"*

⇒ 에이전트가 **① 15행 중 최신 행(Silver Rewards Card · IN_PROGRESS · 11/13/2025)을 지목**하고 **② $750 을 말해야** 손님이 찍는다. 이 둘이 채점의 전부다.

---

## §2 변이표 (`t2_forensic.mutation_diff` 정본 · trial 별)

| 종류 | trial 0 (s626729) | trial 1 (s373753) |
|---|---|---|
| matched | 1 — `log_verification` ✓ (msg 20) | 1 — `log_verification` ✓ (msg 14) |
| **MISSING** | **1 — `submit_transaction{friend_user_5839, Silver Rewards Card, Best Buy, 750, Shopping}`** | **동일 1건** |
| WRONGARG | **0** (보낸 인자 자체가 0건 ⇒ 필드별 대조 대상 없음) | **0** (동상) |
| EXTRA | 0 | 0 |
| DUP | 0 | 0 |
| BLOCKED | 1 — `log_verification` (msg 48 · `Failed to log verification: Record may already exist.`) | 1 — `log_verification` (msg 38 · 동일 문면) |

BLOCKED 1건은 실패 반환이라 DB 를 안 바꾼다 ⇒ **양 trial 모두 유일한 채점 손실은 MISSING `submit_transaction` 하나**. 변이 집합은 t7346 과 **완전 동일**.

### 필요 값 두 개가 언제 문맥에 실재했는가

| 필요 값 | 출처 | trial 0 | trial 1 |
|---|---|---|---|
| ⓐ **Silver Rewards Card · IN_PROGRESS · 11/13/2025**(15행 중 최신) | `get_referrals_by_user` 출력 | msg **23** | msg **17** |
| ⓑ **"spend at least $750 within 60 days"**(`doc_credit_cards_silver_rewards_card_011`) | KB 검색 결과 | msg **3**(bm25 10위) · msg **33**(dense 재도착) | msg **45**(dense **1위**) |

**두 값 다 문맥에 도착했다.** 양 trial 어느 assistant 발화에도 `"Silver"` **0회** · `"750"` **0회**(전수 검색). 결손은 도착이 아니라 **선택·전달**이다.

---

## §3 ★스텝 표 — trial 0 (s626729 · 74 msgs)

| step | role | what | actor | kind | code_path | evidence (축자) | consequence | reward_causal |
|---|---|---|---|---|---|---|---|---|
| **3** | tool | `KB_search_bm25{"query":"referral bonus","k":10}` 결과 10위에 **정답 문서**가 실려 도착(11,214자) | env | 재료 도착(+) | — | `10. Silver Rewards Card: Referral Program … ID: doc_credit_cards_silver_rewards_card_011 … - The referred person must be approved and spend at least $750 within 60 days of account opening` | ⓑ 확보 | 무관(이득) |
| **3′** | (view) | 그 11,214자 메시지를 **다음 생성 턴부터** head/tail 다이제스트로 압축 | **our_layer** | 근거창 파괴 | `t2_gate_patch.py` — `T2_VIEW_MSG_CAP`(`go_stack.sh:156` `export T2_VIEW_MSG_CAP=8000`) · 앵커 `[T2_VIEW_COMPACT] active:` | 로그: `[T2_VIEW_COMPACT] active: 1 tool output(s) digested in view` (이 sim 총 **4회** 누적: `active: 1/2/3/4`) | ⓑ($750)가 모델 시야에서 사라진다 — t7346 부검이 실측한 다이제스트에 `'750'` 0회 | **준강** |
| **4** | assistant | `verify_identity{"email":"john.doe@example.com","phone_number":"123-456-7890"}` — 손님이 준 적 없는 **플레이스홀더 날조** | model | 날조 인자 | — | `{"provided": "{\"email\": \"john.doe@example.com\", \"phone_number\": \"123-456-7890\"}"}` | 우리 `T2_SCAFFOLD_GET` 이 교정해 회복 | 약 |
| **5** | tool | 우리 층 교정 문면이 다음 수를 **이름으로** 지정 | our_layer | 교정(+) | `t2_scaffold_get.py` 경유 · 앵커 `[T2_SCAFFOLD_GET] verify_identity ->` | `NOT_VERIFIED — the account record has not been fetched yet … Look the customer up with get_user_information_by_name/by_email/by_id, then call this tool again.` | msg 8 lookup 유도 | 무관(이득) |
| **20-21** | assistant | `log_verification{…, time_verified:"2025-11-14 03:40:00 EST"}` **성공** | model + our_layer(`T2_DEMANDED_STEP`·`GB1_VERIFY_BEFORE_ACCOUNT_ACCESS`) | gold 이행 | 앵커 `[T2_DEMANDED_STEP] head=GB1_VERIFY_BEFORE_ACCOUNT_ACCESS → verify_identity` | `Verification logged successfully. - User: Oluwaseun Adeyemi (ID: 86e92f639e)` | **gold 016_0 ✓** | 무관(이미 확보) |
| **23** | tool | `get_referrals_by_user{86e92f639e}` → **15행**. 최신 = `f2a3b4c5d6789012 · Silver Rewards Card · IN_PROGRESS · 11/13/2025` | env | 재료 도착(+) | — | `2. Record ID: f2a3b4c5d6789012 … referred_account_type: Silver Rewards Card / referral_status: IN_PROGRESS / date: 11/13/2025` | ⓐ 확보 | 무관(이득) |
| **23′** | (view) | ★**우리 층 격리 서브가 "미지급 행"을 골라 권위 문구로 배달** — 답이 **Platinum(ERROR·10/05/2025)** | **our_layer** | **오지목** | `t2_ledger.py` `diagnose_choice()` / `onto_context()` (앵커 `[T2_DIAG] raw=%r → %s`) · 호출부 `t2_gate_patch.py` 앵커 `_LG2.diagnose_choice(agent, _la9, _UM9, _sp2, _blk, _e2["rows"])` · 선언 `a2/banking_knowledge.gate.json` `ledger_metrics[0].diagnose_prompt` / `.diagnosed_text` | 로그: `[T2_DIAG] raw='Platinum Rewards Card — an error has occurred throughout the process.' → Platinum Rewards Card` · 배달 문면(선언 축자): `A separate check was run on the records and the policy definitions, with nothing else from this conversation in front of it. It answers: {answer}` · 프롬프트 축자: `One of these records did not pay out. Reply with that record's account type exactly as written above` | 최신행(Silver) 대신 **가장 오래된 이상행**을 권위 있게 지목. 격리 문맥에는 **대화가 한 글자도 없어서**(설계) 손님의 *"most recent"* 제약이 서브에 **원리상 전달 불가** | **중**(t0) / **강**(t1 · §4) |
| **24-27** | assistant | `unlock_discoverable_agent_tool{get_all_user_accounts_by_user_id_3847}` + `call_…` → *"No bank accounts found. / No credit card accounts found."* | **our_layer** | 선행-읽기 강제 | `a2/banking_knowledge.gate.json` `require_tool_before.submit_referral = ["get_all_user_accounts_by_user_id"]` · 중재 `t2_gate_patch.py` 앵커 `print("[T2_ARBITRATE] push dominated target=%s reqs=%s "` | 로그: `[T2_ARBITRATE] push dominated target=submit_referral reqs=GB1_VERIFY_BEFORE_ACCOUNT_ACCESS,GB3_ELIGIBILITY_BEFORE_REFERRAL_ADVICE,reads:get_all_user_accounts_by_user_id unsourced=0` | 4메시지 소각 + *"계좌가 하나도 없다"* 라는 **오도 서사**를 문맥에 심는다(msg 38 이 이걸 근거로 삼는다) | **중** |
| **30** | assistant | ★**결정점 A** — Bronze/COMPLETE 로 오진하고, 손님에게 **이 태스크에 존재하지 않는 `submit_referral` 실행을 지시** | **our_layer**(표적 공급) + model(행 선택) | **비존재 도구 오지목** | ⑴ 대기집합 `t2_gate_patch.py` 앵커 `_uacts = {t for t in ((a2 or {}).get("action_tools") or []) if _exec_side(t) == "user"}` — **태스크별 `user_tools` 가용성 필터 없음** ⑵ 선언 `a2/banking_knowledge.gate.json` `action_tools` 에 `"submit_referral"` 등재(line ≈185) ⑶ 발화 저작 `t2_gate_patch.py` 앵커 `print("[T2_ACTION_SUB] 발화를 격리에서 지음 …` 의 `_own` 표 | 로그: `[T2_RESOLVE] user-action instruct target=submit_referral`(이 sim **12회**) · `[T2_ACTIONREQ] window=open pending_user=['apply_for_credit_card', 'call_discoverable_user_tool', 'submit_referral', 'submit_transaction'] … formalized_target=submit_referral`(**36회**) · `[T2_ACTION_SUB] 발화를 격리에서 지음 (손님 발화 3건 · 값 2067자 · 표기 O)` · msg 30 축자: `you need to submit the referral manually. Please use the following tool to submit the referral: {"name": "submit_referral", "arguments": {"user_id": "86e92f639e", "account_type": "Bronze Rewards Card"}}` | gold 016_1(`submit_transaction`) 대신 **손님이 갖고 있지도 않은 도구**를 지목. `_own` 표는 이것을 *지시가 아니라 사실* 로 세운다 — 태스크 `user_tools` 는 `["submit_transaction","apply_for_credit_card"]` 뿐 | **강** |
| **31** | user | 손님이 우리 층을 **반증** | user_sim | 반증(계측) | — | `I don't have any "submit_referral" tool on my side, so I can't manually submit anything myself.` | 우리 오지목이 실물로 확인됨 | 무관(계측) |
| **32-33** | assistant/tool | `KB_search_dense{"query":"referral bonus processing"}` → **정답 문서 재도착** | model + env | 재료 재도착(+) | — | msg 33 안: `ID: doc_credit_cards_silver_rewards_card_011 … spend at least $750 within 60 days` | ⓑ 두 번째 기회 | 무관(이득) |
| **34-41** | assistant | `get_referrals_by_user` **동일 인자 재호출 ×2** → 우리 dedup stub | model + our_layer | 정체 | 앵커 `[T2_READ_DEDUP] stub tool=` | `[DUPLICATE-READ] This exact call (same tool, same arguments) was already executed earlier … Refer to that output instead of re-reading.` · 3회째: `You have now issued this IDENTICAL call 3 times … Do not issue this same call again.` | 두 번째 기회 소각. dedup 자체는 옳으나 **무엇을 하면 풀리는지**는 여전히 안 말한다([[64]] 부분 이행) | 중 |
| **38** | assistant | *"there are no credit card accounts associated with your user ID, which might explain why the referral bonus has not appeared … we may need to create a new account for you"* | model | 오귀인 | — | 축자 상동 | step 24-27 이 심은 오도 서사가 여기서 **원인 설명으로 승격** | 중 |
| **42-49** | assistant | 이미 VERIFIED·로그된 신원을 **재검증** → `log_verification` 중복 → env 차단 | model | 되돌이 | — | msg 49: `Failed to log verification: Record may already exist.` | 6메시지 소각(DB 무변 · BLOCKED 1건) | 약 |
| **52** | assistant | 행동 대신 *"Could you please confirm if I should proceed with this action?"* — 허락 요청 | model + **our_layer** | 정체 + 재료 채널 폐쇄 | `t2_gate_patch.py` 앵커 `print("[T2_MATERIAL_GATE] stop=%s turn=%d calls=%s …` · `T2_RESOLVE_CAP` | `[T2_MATERIAL_GATE] stop=resolve_cap(정체 3회) turn=52 calls=- pending=0 axes=7 prose=True` (turn 52·54·56·58·60·62 **연속**) | 손님이 결정 단서를 주기 **직전**부터 우리 재료 배달이 전부 막힌다 | **준강** |
| **53** | user | 손님이 **정답 좌표**를 준다 | user_sim | 단서(+) | — | `I'm specifically asking about the most recent referral — my friend told me their user ID is **friend_user_5839**.` | 최신행 판정에 필요한 전부가 갖춰짐 | 무관(이득) |
| **54-57** | assistant | 3번째 중복 read → `get_referrals_by_user{friend_user_5839}` → *"No records found"* | model | 오조회 | — | msg 57: `No records found in 'referrals'.` (`referrer_id` 축인데 피추천인 id 로 조회) | ⓐ 재확인의 **마지막 기회 소각** | **준강** |
| **58-59** | assistant/user | *"there are no referral records associated with … friend_user_5839 … I recommend escalating"* → 손님 동의 | model | 사임 | — | 축자 상동 | 진단 포기 확정 | **강** |
| **62-69** | assistant | 이관 프로토콜 사슬(`initial_transfer_to_human_agent_0218` unlock/call ×2) | **our_layer**(t7348 신규) | 표적 전환 | `t2_resolve.py` `T2_DISCOVERY_STEP2` · `t2_gate_patch.py` 앵커 `[T2_FORCE_ACTION] say-don't-do → tool_choice=required 재생성` | `[T2_OUR_NAMES] 등재 name=initial_transfer_to_human_agent_0218 (출처=T2_DISCOVERY_STEP2 지목)` · `[T2_DISCOVERY_STEP2] deny name=… (이미 회수·미unlock·formalize 정합)` · `[T2_FORCE_ACTION] say-don't-do → tool_choice=required 재생성` | 남은 턴을 **033 의 gold 사슬**로 소진 — 016 에서 이관은 gold 가 **아니다**(§7) | 중(이미 손실 후) |
| **70-71** | assistant | `transfer_to_human_agents{reason:"account_ownership_dispute"}` | model + our_layer | 이관 | `t2_gate_patch.py` 앵커 `print("[T2_TRANSFER_TIER] chosen=%s(tier %s) -> higher applicable=%s(tier %s) "` | `[T2_TRANSFER_TIER] chosen=kb_search_unsuccessful_customer_requests_transfer(tier 2) -> higher applicable=account_ownership_dispute(tier 1) evidence='NOT_VERIFIED'` — 손님은 msg 45 에서 **VERIFIED** 인데 옛 문자열을 근거로 삼았다([[25]] 위반) | 성적 무관(비-변이 도구) · **계기 결함**으로 기록 | 무관 |
| **73** | user | `###STOP###` | user_sim | 종료 | — | — | **MISSING `submit_transaction` 확정** | — |

---

## §4 ★스텝 표 — trial 1 (s373753 · 60 msgs)

| step | role | what | actor | kind | code_path | evidence (축자) | consequence | reward_causal |
|---|---|---|---|---|---|---|---|---|
| **2** | assistant | ★**분기점** — trial 0 은 첫 수가 `KB_search_bm25("referral bonus")` 였는데 trial 1 은 **검색을 건너뛰고** 바로 신원 요구 | model | 순서 분기 | — | msg 2: `To help us verify your identity, could you please provide me with your email address and phone number?` | ⓑ($750)가 **msg 45 까지 문맥에 없다** — t0 대비 42 메시지 지연 | **준강** |
| **4-11** | assistant/tool | `verify_identity` ×3(동일 인자 2회 포함) → 우리 교정 → lookup → VERIFIED | model + our_layer(+) | 교정 | 앵커 `[T2_SCAFFOLD_GET] verify_identity ->` | `NOT_VERIFIED — the account record has not been fetched yet … Use a DIFFERENT identifier each time` | 회복 | 무관 |
| **14-15** | assistant | `log_verification` **성공** | model | gold 이행 | — | `Verification logged successfully.` | **gold 016_0 ✓** | 무관 |
| **17** | tool | `get_referrals_by_user` → 15행(최신 = Silver·IN_PROGRESS·11/13) | env | 재료 도착(+) | — | 상동 | ⓐ 확보 | 무관(이득) |
| **17′** | (view) | ★**T2_DIAG 가 다시 Platinum(ERROR·10/05) 을 권위 배달** — t7346·t7348 · 양 trial **4/4 동일 문자열** | **our_layer** | **오지목** | `t2_ledger.py` `diagnose_choice()`/`onto_context()` · `a2/banking_knowledge.gate.json` `ledger_metrics[0].diagnose_prompt` | `[T2_DIAG] raw='Platinum Rewards Card — an error has occurred throughout the process.' → Platinum Rewards Card` | 다음 assistant 발화가 **즉시** 이 답을 복창한다(step 22) | **강** |
| **18-21** | assistant | `unlock`+`call get_all_user_accounts_by_user_id_3847` → *"No accounts found"* | **our_layer** | 선행-읽기 강제 | `require_tool_before.submit_referral` (§3 step 24-27 동일) | `[T2_ARBITRATE] push dominated target=submit_referral reqs=…reads:get_all_user_accounts_by_user_id` | 4메시지 소각 | 중 |
| **22** | assistant | ★**결정점 A′** — *"there seems to be an issue with the referral process for the **Platinum Rewards Card**, as indicated by the '**ERROR**' status"* | **our_layer**(주) + model | 오진 고착 | 상동(T2_DIAG) | 축자 상동. 해당 행은 **#12 · 10/05/2025** = 15행 중 **오래된 쪽**이고 손님은 *"most recent"* 를 묻고 있다 | 이후 **6개 assistant 메시지**(22·26·30·40·42·46·48)가 전부 Platinum 축에 묶인다 | **강** |
| **24-29** | assistant | `get_referrals_by_user` 동일 인자 ×2 → dedup stub · `get_credit_card_accounts_by_user` → none | model + our_layer | 정체 | `[T2_READ_DEDUP] stub tool=get_referrals_by_user` | `[DUPLICATE-READ] …` | 재확인 기회 소각 | 중 |
| **30** | assistant | Platinum ERROR 를 원인으로 확정하고 **escalate 제안** | model | 사임 예고 | — | `Given that the referral for the Platinum Rewards Card has an "ERROR" status, I will escalate this issue to our referral team` | 진단 종료 선언 | **준강** |
| **31** | user | 손님이 `friend_user_5839` 제공 + escalate 동의 | user_sim | 단서(+) | — | 축자 상동 | — | 무관 |
| **32** | assistant | ★**결정점 B** — 갑자기 *"I need you to run a specific tool to get the **last four digits** of your credit card … Please use the `get_card_last_4_digits` tool by providing your **`credit_card_account_id`**"* | **our_layer**(증폭·문면 제공) + model(초안 발단) | **무관 표적 표면화** | ⑴ `t2_gate_patch.py` `_value_acquire_fb(am, messages, specs, …)` — 조건은 ①값 미실재 ②give 미실행 ③**assistant 산문에 `reask_signals` 등장** 뿐이고 *"이 태스크가 그 write 로 가는가"* 는 **의도적으로 빼 놓았다**(주석 축자: `★C4 철회(2026-08-05) … 그건 열린 술어라 우리 몫이 아니다([[22]])`) ⑵ 선언 `a2/banking_knowledge.gate.json` `value_acquisition[0]` = `{"write": "file_credit_card_transaction_dispute_4829", "arg": "card_last_4_digits", …}` — **이 태스크와 무관한 write** ⑶ `t2_gate_patch.py` 앵커 `print("[T2_USER_TOOL_NOTE] pre-give note: %s"` | 로그: `[T2_VALUE_ACQUIRE] consumers card_last_4_digits=1` · `[T2_VALUE_ACQUIRE] give-surfacing → nudge (regen)` ×3 · `[T2_LEVER] T2_USER_TOOL_NOTE sim=task_016#s373753 usertool_note` · `[T2_USER_TOOL_NOTE] pre-give note: get_card_last_4_digits` · 선언 피드백 축자: `Use give_discoverable_user_tool to give get_card_last_4_digits to the customer NOW, then have them run it with their **credit_card_account_id**` ↔ msg 32 축자 `Provide your **credit_card_account_id** as an argument` | 손님이 정답 좌표를 준 **바로 다음 턴**을 통째로 소각. 이 sim 이전에 `credit_card_account_id` 는 대화 어디에도 없다 ⇒ 문면의 출처는 **우리 선언** | **준강** |
| **33** | user | *"I don't have my credit card account ID handy, and I don't see a way to run that tool on my end."* | user_sim | 반증 | — | 축자 상동 | 우리 오표적 실물 확인(트라이얼0 의 `submit_referral` 과 **동형**) | 무관(계측) |
| **34-39** | assistant | 재검증 → `log_verification` 중복 → 차단 | model | 되돌이 | — | msg 39: `Failed to log verification: Record may already exist.` | 6메시지 소각 | 약 |
| **41** | user | *"I'm asking about my **most recent referral** — the friend who was approved and has already started using the card."* | user_sim | 단서(강) | — | 축자 상동 | 최신행 제약을 **두 번째로** 명시 | 무관(이득) |
| **42** | assistant | 그럼에도 *"Given the '**ERROR**' status on your **most recent** referral for the **Platinum Rewards Card**"* | model | **고착** | (T2_DIAG 상류) | 축자 상동 — 손님 낱말 *"most recent"* 를 **우리가 준 Platinum 에 붙여** 재확인 | 최신행 제약이 **오답을 강화**하는 데 쓰였다 | **강** |
| **43** | user | ★손님이 gold 를 여는 질문을 **직접** 한다 | user_sim | 단서(최강) | — | `can you confirm whether there's **any spending requirement** my friend has to meet … (and if so, how much and within what timeframe)? If the requirement hasn't been met yet, I'd rather just help them complete it` | 남은 것은 *"Silver · $750 · 60일"* 한 문장 | 무관(이득) |
| **44-45** | assistant/tool | `KB_search_dense{"query":"Platinum Rewards Card referral bonus spending requirement"}` → **1위가 정답 문서** | model(질의어는 우리 오지목 Platinum) + env(+) | 재료 도착(+) | — | `1. Silver Rewards Card: Referral Program … ID: doc_credit_cards_silver_rewards_card_011 … - The referred person must be approved and spend at least **$750 within 60 days** of account opening` | ⓑ 도착 — **필요한 두 값이 처음으로 동시에 문맥에 있다** | 무관(이득) |
| **46** | assistant | ★★**결정점 C — 최종 손실 지점**. 1위 문서를 앞에 두고 *"the referral bonus for the **Platinum Rewards Card** requires your friend to spend at least **$1,500 within 90 days**"* | **model** | **날조(fabrication)** | — (우리 층 **차단 실패**: `T2_SOURCE claims=1 unsourced=0` · `[T2_CLAIMPROV] window hit(resign) claims=4 unbacked=0` · `FAB_STRIP` **0 발화**) | 축자 상동. `$1,500`/`90 days` 는 **이 대화의 어느 도구 출력에도 없다**(전수 검색) | user-sim 조건 *"identified which one you're asking about"* + *"specific dollar amount"* 를 **틀린 값으로** 채움 ⇒ `submit_transaction` 미발화 | **강** |
| **47** | tool | `get_credit_card_transactions_by_user{friend_user_5839}` → *"No records found"* | env | — | — | 축자 상동 | 자기 오진을 반증할 재료였으나 무시 | 약 |
| **48-49** | assistant/user | 에이전트가 오히려 손님에게 **카드 종류를 되묻는다** → 손님: *"I'm not actually sure which specific card type … Can you check on your side"* | model | 역전(부하 전가) | — | msg 48: `could you provide the specific type of credit card your friend signed up for?` | 손님이 알 수 없는 것을 요구 ⇒ 조건 ① 명시적 미달 | **강** |
| **50-51** | assistant | `get_credit_card_accounts_by_user{friend_user_5839}` → none | model | — | — | — | — | 약 |
| **52-55** | assistant | ★**우리 층이 마지막 두 턴을 이관 사슬로 돌린다** — `unlock{initial_transfer_to_human_agent_0218}` → `unlock{…_1822}` | **our_layer**(t7348 신규) | **표적 전환** | `t2_resolve.py` `T2_DISCOVERY_STEP2` · `t2_gate_patch.py` 앵커 `[T2_FORCE_ACTION] say-don't-do → tool_choice=required 재생성` | `[T2_OUR_NAMES] 등재 name=initial_transfer_to_human_agent_0218 (출처=T2_DISCOVERY_STEP2 지목)` · `[T2_DISCOVERY_STEP2] deny name=initial_transfer_to_human_agent_0218 (레지스트리 폴백·미unlock·formalize 정합)` · `[T2_FORCE_ACTION] say-don't-do → tool_choice=required 재생성` **×2** · `[T2_UNCALLED_UNLOCK] surface initial_transfer_to_human_agent_0218,initial_transfer_to_human_agent_1822` | 손님이 *"tell me the exact card type, the exact spend requirement, the deadline"* 를 물은 **직후 두 턴**이 도구 unlock 으로 소각. 016 에서 이관은 gold 가 아니다 | **준강** |
| **52·54·58** | (view) | `T2_REQUIRE_DOC_DELIVER` 가 이관 사전문서 16,498자/4,663자 부착 | our_layer | 결정점 **이후** | 앵커 `[T2_REQUIRE_DOC_DELIVER] deliver tool=` | `deliver tool=transfer_to_human_agents docs=6 chars=16498 turn=52 fired=1/3 unread=doc_bank_accounts_…_037,…_042,doc_checking_accounts_…_009,doc_credit_cards_credit_cards_(general)_010,_011,_012` | 배달 문서 6종 중 `doc_credit_cards_silver_rewards_card_011` **없음** | 무관 |
| **59** | user | `###TRANSFER###` | user_sim | 종료 | — | — | **MISSING `submit_transaction` 확정** | — |

---

## §5 레버 발화표 (지시 목록 · 이 sim 줄만 전수 계수)

| 레버 | t0 | t1 | 판정 | 근거 |
|---|---|---|---|---|
| `T2_SG_DOCS` | **0** | **0** | **미발화** (PIN 에 `T2_SG_DOCS=1` 있는데 이 태스크 축에 닿지 않음) | grep 0줄 |
| `T2_PIN_READ` | 4 | 0 | **발화했으나 무관** | `[T2_PIN_READ] pinned get_referrals_by_user(None=None)` · `pinned unlock_discoverable_agent_tool(agent_tool_name=get_all_user_accounts_by_user_id_3847)` — 핀 대상이 gold 와 무관 |
| `T2_DEMANDED_STEP` | 13 | 0 | **발화·이행(+)** | `head=GB1_VERIFY_BEFORE_ACCOUNT_ACCESS → verify_identity` ×13 → gold 016_0 확보 |
| `T2_CLAIMPROV` | 94(양 sim 합) | — | **발화했으나 결정적 거짓 통과** | msg 46 날조 턴에 `window hit(resign) claims=4 unbacked=0` — **수치 주장은 검사 범위 밖**(도구-kind 만 본다) |
| `T2_FOLLOWUP` | **0** | **0** | **미발화** | grep 0줄 |
| `T2_SEARCH_AGENT` | 6 | 5 | **발화·오축** | `group=business_credit_cards · 문서 82` / `group=credit_cards · 문서 110(뺀 것 4: … doc_credit_cards_silver_rewards_card_010 …)` → 이후 `요청 축 business_credit_cards,credit_cards 모두 처리됨 — 침묵` ×2. 축 선택이 `T2_DOCDECIDE → 'Bronze Rewards Card'` 를 따라가 **Silver 축이 서지 않았다** |
| `FAB_STRIP` | **0** | **0** | **미발화** — t1 msg 46 의 `$1,500 within 90 days` 날조를 **놓쳤다** | grep 0줄 |
| `T2_ARG_PRODUCERS` | **0** | **0** | **미발화** | grep 0줄 |
| READ-FIRST | **0** | **0** | **미발화**(이 이름의 마커 없음) | grep 0줄 |
| `T2_REQUIRE_DOC_DELIVER` | 6 | 5 | **발화했으나 결정점 이후** | t0 turn 60·62 / t1 turn 52·54·58 — 전부 이관 사슬용. 배달 6문서에 `silver_rewards_card_011` 없음 |
| `T2_SEARCH_REARM` | **0** | 2 | **발화·오표적** | `[T2_SEARCH_REARM] group=credit_cards 신규 대상 platinum_rewards_card (기배달 bronze_rewards_card)` → `델타 배달 11240자 turn=28`. gold 는 **Silver** — t7346 부검(`FAILURE_MASTER` §레버표)이 이미 *"016: 신규 대상이 platinum(gold 는 Silver) → 0 매수"* 로 기록 |

### 목록 밖이지만 이 궤적을 실제로 움직인 것 (전수 계수에서 상위)

| 레버 | t0 | t1 | 판정 |
|---|---|---|---|
| `T2_ACTIONREQ` `formalized_target=submit_referral` | 36회(양 sim 합) | | **오발화** — 태스크 `user_tools` 에 없는 도구를 대기집합에 유지 |
| `T2_ACTION_SUB` | 6 | 5 | **오발화(증폭)** — 위 오표적을 *"Tools the CUSTOMER runs in this chat"* 사실 표로 승격 |
| `T2_DIAG` | 1 | 1 | **오발화** — 4/4 전부 `Platinum Rewards Card` |
| `T2_VALUE_ACQUIRE` / `T2_USER_TOOL_NOTE` | 0 | 6 / 1 | **오발화(t7348 신규 노출)** — dispute 전용 spec 이 referral 태스크에서 발화 |
| `T2_DISCOVERY_STEP2` / `T2_FORCE_ACTION` | 2 / 1 | 2 / 2 | **오발화(t7348 신규)** — 마지막 턴을 033 의 gold 사슬로 전환 |
| `T2_MATERIAL_GATE` `stop=resolve_cap(정체 3회)` | 6 | 4 | **차단** — t0 turn 52~62 · t1 turn 40~58 연속. 결정점 구간이 통째로 재료 0 |
| `T2_VIEW_COMPACT` | 4 | 1 | **근거창 파괴** — ⓑ 를 담은 11,214자를 다이제스트 |

---

## §6 선행 판정과 대조 — **같은 원인 + 새 원인 3종**

### 그대로인 것
| 선행 판정 | 출처 | t7348 |
|---|---|---|
| 변이 = MISSING `submit_transaction` 하나 · 축 = DB | `FAILURE_MASTER__20260822.md:43` | **동일** |
| `T2_DIAG` 가 Platinum 을 오지목(카테고리 **B** 우리-층 오지목) | `FAILURE_MASTER__20260822.md:101` (당시 UNPROVEN) | **재현 · 4/4 동일 문자열**. t1 에서 6개 assistant 메시지를 지배해 **인과가 훨씬 강해졌다** |
| `T2_VIEW_MSG_CAP=8000` 이 ⓑ 를 담은 메시지를 다이제스트(카테고리 **A** 근거창 파괴) | `FAILURE_MASTER__20260822.md:102` | **재현**(`[T2_VIEW_COMPACT] active: 1~4`) |
| `T2_SEARCH_REARM` 이 platinum 으로 재무장(음 −) | `FAILURE_MASTER__20260822.md:167` | **재현**(t1 turn 28 · 11,240자) |
| `T2_MATERIAL_GATE resolve_cap` 이 결정점 구간을 먼저 막는다 | `FAILURE_MASTER__20260822.md:173` (016 반증자 ★) | **재현** — t0 turn 52~62 · t1 turn 40~58 |
| *"재무장이 마지막 검색 예산을 훔쳐 Silver 를 못 샀다"* = **REFUTED** | `FAILURE_MASTER__20260822.md:264` | **유지** — t7348 도 예산 문제가 아니다. t1 은 msg 45 에서 **정답 문서를 1위로 받고도** 날조했다 |

### 달라진 것 (t7346 → t7348 · **같은 seed · 같은 PIN/ON** · `diff` 로 러너 바이트 동일 확인)
| 변화 | t7346 | t7348 | 비고 |
|---|---|---|---|
| assistant 발화에 `"Silver"` | t0 msg 40 에 **1회** | **0회 (양 trial)** | 진단 축 **회귀**. reward 는 둘 다 0 이라 집계로는 안 보인다([[08]]) |
| `submit_referral` 을 손님에게 지시 | **없음** | **t0 msg 30 에 신규 출현** | `T2_ACTION_SUB` 6→**11** · `formalized_target=submit_referral` 23→**36** |
| `get_card_last_4_digits` 표면화 | **없음** | **t1 msg 32 에 신규 출현** | `T2_VALUE_ACQUIRE` 0→**6** · `T2_USER_TOOL_NOTE` 0→**1** |
| 이관 사슬 강제 | 없음 | `T2_DISCOVERY_STEP2` 0→**4** · `T2_FORCE_ACTION` 4→3(문맥 전환) | 033 부검(`x504`)이 *"t7348 신규"* 로 확정한 그 레버 |
| `T2_TRANSFER_TIER` | 2 | 1 | 근거가 `evidence='NOT_VERIFIED'` 로 **양 런 동일하게 틀렸다** |

⚠**변인 귀속**: 결과 JSON 의 `git_commit` 은 양 런 모두 `fc0055dc…` 인데 이는 **tau2-bench 하네스** 커밋이다. 우리 엔진은 런 스크립트 헤더 축자대로 *"t7346 sha `ee18d797` 뒤 **9 커밋**"* 이 들어갔고, `PIN`/`ON` 은 `diff` 로 **바이트 동일**임을 확인했다 ⇒ 위 5종 변화의 유일한 후보 변인은 **9-커밋 엔진 묶음**이다. **묶음 Δ 로 개별 커밋을 지목하지 않는다**(C594).

### 033 과의 길항 (지시가 지목한 *"서로 해를 끼치는 이득 관계"*)
`x504_TASK_033_t7348_perstep.md:176` 축자: *"**`T2_DISCOVERY_STEP2`** … ★**t7348 신규**. action-required 를 세워 `T2_FORCE_ACTION` 을 켜는 **트리거**"*. 033 의 gold 는 `initial_transfer_to_human_agent_1822`/`_0218` **사슬 자체**라서 이 레버가 표적이다. **016 의 gold 에는 이관이 한 칸도 없다** — 같은 레버가 016 에서는 손님의 결정적 질문(t1 msg 49) 직후 두 턴을 unlock 으로 돌려 버렸다. **부호가 태스크로 갈리는 전형**([[70]]) — 끄지 말고 **조건**을 붙일 자리다. 단 016 에서는 이미 msg 46 에서 손실이 확정된 뒤라 **인과 순위는 준강 이하**다.

---

## §7 원인 확정

### 채점 손실의 **직접** 원인 (스텝 단위)
- **trial 1 = msg 46 · `model` · 날조.** 정답 문서(`$750 · 60 days · Silver`)가 **직전 메시지 1위**에 있는 상태에서 `$1,500 within 90 days · Platinum` 을 지어냈다. 우리 층 검사 3종(`T2_SOURCE`·`T2_CLAIMPROV`·`FAB_STRIP`)이 전부 통과시켰다.
- **trial 0 = msg 30·54-58 · `our_layer` + `model`.** 우리가 **비존재 표적(`submit_referral`)** 을 사실 표로 승격했고(msg 30), 모델은 최신행 대신 Bronze/COMPLETE 를 골랐으며, 손님이 좌표를 준 뒤(msg 53)에도 `referrer_id` 축을 `friend_user_5839` 로 조회해(msg 56) 마지막 기회를 소각했다.

### 공통 **상류** 원인 — 우리 층
1. **`T2_DIAG` 오지목**(양 trial · 양 런 4/4). 격리 문맥에 대화가 0글자라 *"most recent"* 제약이 **원리상 서브에 도달 불가**한데, 프롬프트는 *"One of these records did not pay out"* 이라는 **거짓 유일성 전제**를 깔고(15행 중 REJECTED 3 · ERROR 1 · IN_PROGRESS 1 = 미지급 5행), 답은 *"A separate check was run …"* 이라는 **권위 문면**으로 배달된다. `agent._t2_diag` 메모이제이션 때문에 **sim 내내 재평가되지 않는다**.
2. **비존재 표적 유지 — `submit_referral`.** `_uacts` 는 A2 `action_tools` 를 실행자 축으로만 거르고 **태스크 `user_tools` 가용성**을 보지 않는다. 손님이 msg 31 에서 직접 반증했다.
3. **무관 표적 표면화 — `get_card_last_4_digits`.** `value_acquisition[0].write` 는 `file_credit_card_transaction_dispute_4829` 인데 이 태스크는 그 write 로 가지 않는다. 코드 주석이 그 조건을 **의도적으로 뺐다**고 자인한다(`★C4 철회`).
4. **근거창 파괴 — `T2_VIEW_MSG_CAP=8000`.** ⓑ 를 담은 11,214자가 다이제스트된다.
5. **결정점 구간 폐쇄 — `MATERIAL_GATE resolve_cap(정체 3회)`.** t0 turn 52~62 · t1 turn 40~58 연속.

### 주체 배분 (스텝 가중)
| 주체 | trial 0 | trial 1 |
|---|---|---|
| `our_layer` | **주** (msg 24-27·30·52 · T2_DIAG) | **준주** (T2_DIAG 강 · msg 32 준강 · msg 52-55 준강) |
| `model` | 준주 (Bronze 오선택 · msg 56 오조회 · 되돌이) | **주** (msg 46 날조 · msg 42 고착 · msg 48 부하 전가) |
| `env` | 무관(재료는 다 줬다 — ⓐ·ⓑ 모두 도착) | 동상 |
| `user_sim` | **면책 불가 아님** — 손님은 msg 53 에서 좌표를, msg 43 에서 질문을 **직접** 줬다([[21]]) | 동상 |

⇒ **원인 확정: 단일 원인 아님. `our_layer` 상류 오지목 3종(T2_DIAG · submit_referral · card_last_4_digits)이 두 trial 의 탐색 방향을 각각 다른 오답으로 고정했고, 그 위에서 model 이 마지막 칸(행 선택 / 금액 날조)을 틀렸다.** trial 1 의 최종 손실 지점(msg 46)만 떼면 `model` 이지만, **그 턴의 질의어(`Platinum Rewards Card …`)와 결론 축을 정한 것은 우리 층의 T2_DIAG 답**이다.

---

## §8 처방 후보 (⛔실행 없음 · 격리 프로브 전 승격 금지 [[62]])

| # | 후보 | 표적 | 근거 | 위험 |
|---|---|---|---|---|
| **P1** | `diagnose_choice` 의 격리 문맥에 **손님 질문의 형식화된 제약 하나**(예: recency)를 실어 준다 — 대화 원문이 아니라 **선언된 필드 축**(`date_field`)으로. 또는 프롬프트의 거짓 유일성(*"One of these records"*)을 **미지급 행이 여럿일 수 있음**으로 고친다 | T2_DIAG 4/4 오지목 | §4 step 17′·22·42 | 격리 문맥 오염([[65]]) — **격리 프로브(x46N) 필수** |
| **P2** | `_uacts` 대기집합에 **태스크 가용성 필터**(env 가 이 sim 에 실제로 준 user tool 집합)를 건다 — 도메인 어휘 0·닫힌 술어 | `formalized_target=submit_referral` 36회 | §3 step 30·31 | `T2_ACTION_SUB` 표가 비면 소유권 발화 자체가 죽을 수 있다(x228 부정통제 재확인 필요) |
| **P3** | `_value_acquire_fb` 에 **닫힌 조건** 추가: `spec["write"]` 가 이 sim 의 **호출 가능 도구 집합**에 없으면 침묵 | `T2_VALUE_ACQUIRE` 0→6 신규 오발화 | §4 step 32·33 | 주석의 `C4 철회` 사유(053 형 조기 발화)를 되살릴 위험 — **오프라인 전수(x104 §C 동형) 재검 필수** |
| **P4** | `T2_DISCOVERY_STEP2` 를 **손님이 이관을 요청했거나 정책 게이트가 이관을 지시한 뒤**로 조건화 | 033(+) ↔ 016(−) 길항 | §6 · `x504:176` | ⛔**끄지 말 것**([[60]]) · 조건은 **도메인 일반 닫힌 술어**여야 한다([[70]]·[[05]]) |
| **P5** | `T2_VIEW_MSG_CAP` 다이제스트에서 **`required_documents` 로 선언된 문서 블록**은 잘라내지 않는다(또는 tail 대신 문서 경계로 자른다) | ⓑ 소실 | §3 step 3′ | 창 폭발 — 상류 공유 노드라 폭발 반경 큼 |
| **P6** | `T2_TRANSFER_TIER` 의 `evidence` 수집이 **최신 verdict** 를 보게 한다(현재 `'NOT_VERIFIED'` 를 VERIFIED 이후에도 사용) | 계기 결함([[25]]) | §3 step 70-71 | 성적 무관 — 우선순위 낮음 |

⚠**우선순위**: 이 태스크에서 **가장 많은 스텝을 지배한 것은 P1(T2_DIAG)** 이다. P2·P3 는 t7348 신규 회귀라 **되돌리는 쪽**이 싸다. P4 는 033 과의 절충이므로 **부호표 없이 손대지 말 것**([[70]]).
