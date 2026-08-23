# t7346 halfB task_016 per-step 포렌식 (2026-08-22 런 · 작성 2026-08-23)

- 런: `bank_t7346_halfB_20260822`(sha `ee18d797` · stage1 20 tasks × nt=2 · PIN/ON = `run_t7346_overnight_stage1_20260822.sh:84-91`).
- 대상: `task_016` **2 trial 전부 reward 0.0 · term=`user_stop`** (trial0 = sim `s626729` · 70 msgs / trial1 = sim `s373753` · 62 msgs).
- 방법: 변이 = 정본 `t2_forensic.mutation_diff`(손 비교기 0·[[67]]). 궤적 = `results.json.gz` 전 메시지 정독. 레버 = `bank_t7346_halfB_20260822.log.gz` 의 `[sim=task_016#s626729]` **439라인** / `[sim=task_016#s373753]` **349라인** 전수. 인용 전부 축자.
- 대조: 선행 정본 `T7336_FORENSIC_016_2026_08_21.md` · 기준선 `bank_t7328_halfB_20260819r2`(sha 상이).
- ⛔수리·코드 수정 실행 없음. gold 는 진단용으로만([[23]]).

---

## 1. 채점축

`sim['reward_info']` 직독:

```
reward 0.0 · reward_basis ["DB"] · db_check {db_match:false, db_reward:0.0}
reward_breakdown {"DB": 0.0} · env_assertions [] · nl_assertions null · communicate_checks null
```

**DB-해시 축**이다(ACTION 축 아님). `action_checks` 는 진단 보조로만 읽는다:

| action_id | requestor | name | action_match |
|---|---|---|---|
| 016_0 | assistant | `log_verification{Oluwaseun Adeyemi, 86e92f639e, …, 2025-11-14 03:40:00 EST}` | **true** |
| 016_1 | **user** | `submit_transaction{friend_user_5839, "Silver Rewards Card", "Best Buy", 750, "Shopping"}` | **false** |

016_1 은 **user 도구**다(`user_tools: ["submit_transaction","apply_for_credit_card"]`). 즉 DB 델타를 만드는 마지막 한 칸은 **에이전트가 손님에게 구체 금액을 말해 줘야** 손님이 찍는다. user-sim 스펙 축자:

> *"**Tool usage:** You have access to `submit_transaction`. ONLY use this tool if the agent has: - Actually looked up your referrals and identified which one you're asking about - Explained that your friend hasn't met a spending requirement yet - **Told you a specific dollar amount** your friend needs to spend"*

## 2. 변이표 (`t2_forensic.mutation_diff` · trial 별)

| 종류 | trial 0 (s626729) | trial 1 (s373753) |
|---|---|---|
| matched | 1 — `log_verification` ✓ (msg 18) | 1 — `log_verification` ✓ (msg 16) |
| **MISSING** | **1 — `submit_transaction{friend_user_5839, Silver Rewards Card, Best Buy, 750, Shopping}`** | **동일 1건** |
| WRONGARG | 0 | 0 |
| EXTRA | 0 | 0 |
| **DUP** | 0 | **2 — `log_verification` ×2 (msg 30·38 · 둘 다 `Failed to log verification: Record may already exist.`)** |
| BLOCKED | 0 | 0 |

DUP 2건은 실패 반환이라 DB 를 안 바꾼다 ⇒ **양 trial 모두 유일한 채점 손실은 MISSING `submit_transaction` 하나**. 변이 집합은 t7336·t7328 과 **완전 동일**.

WRONGARG 가 0 이므로 필드별 대조는 대상 없음(보낸 인자 자체가 0건).

---

## 3. step-by-step — 결정 지점 축자 추적

### 3.0 두 trial 공통 전제 — **필요한 값 두 개가 언제 문맥에 실재했는가**

| 필요 값 | 어디서 왔나 | 언제 |
|---|---|---|
| ⓐ 문의 대상 = **Silver Rewards Card · IN_PROGRESS · 11/13/2025**(15건 중 최신) | `get_referrals_by_user` 출력 | trial0 msg[21] / trial1 msg[19] |
| ⓑ 요건 = **"spend at least $750 within 60 days"** (`doc_credit_cards_silver_rewards_card_011`) | `KB_search_bm25(query="referral bonus", k=10)` 출력의 **10번째 항목** | **양 trial 모두 msg[3]** |

즉 t7336 판정(*"요건 문장이 끝까지 문맥 미등장"*)과 달리 **t7346 에서는 ⓑ가 3번째 메시지에 도착했다**. 축자(msg[3] 말미):

> `10. Silver Rewards Card: Referral Program - Earn Rewards for Referring Friends`
> `   ID: doc_credit_cards_silver_rewards_card_011`
> `## What Your Referral Must Do`
> `- The referred person must be approved and spend at least $750 within 60 days of account opening`

**그런데 이 메시지는 11,214자다.** 우리 층의 뷰-압축이 `T2_VIEW_MSG_CAP=8000`(`go_stack.sh:152`)으로 **다음 생성 턴부터** 이것을 head 300 + tail 150 으로 잘라 낸다(`t2_gate_patch.py:6224-6231`). 실측 다이제스트:

```
HEAD300: '1. Beige Account Referral Program\n   ID: doc_business_checking_accounts_beige_012 … $500 for each qualify'
TAIL150: 'within the 7 limit\n\n\n[Timing: retrieval=1ms, …]\n[axis] matches: 36 documents contain all of these words; 32 shown (4 not shown)'
→ 다이제스트 안에 '750' 0회
```

배치 규칙(`_compact_view`: 마지막 assistant 이후의 tool 출력 = `batch` 는 면제)상 msg[3] 이 원문으로 보이는 생성은 **msg[4] 하나뿐**이다. msg[4] 시점에는 ⓐ(referral 목록)가 아직 없다.

⇒ **ⓐ와 ⓑ가 모델의 생성-뷰에 동시에 존재한 턴이 trial 0 에는 0회다.** 이 sim 의 KB 출력은 전부 8,000자 초과라(trial0: 11,214 / 16,105 / 16,233 / 16,188 · trial1: 11,214 / 17,820 / 29,238) 도착 다음 턴부터 예외 없이 잘린다.

`[T2_VIEW_COMPACT] active: 1 tool output(s) digested in view` 는 **`_t2_vc_logged` 로 sim 당 1회만 인쇄**된다(`t2_gate_patch.py:6765-6769`) — 로그 1줄은 "한 번 일어난 일"이 아니라 "처음 일어난 일"이다.

### 3.1 trial 0 (s626729 · 70 msgs)

- **[2]** 모델이 스스로 KB 를 친다: `KB_search_bm25{"query":"referral bonus","k":10}` → **[3]** 에 silver_011($750) 포함 10편 도착(위 ⓑ).
- **[4]~[19]** 신원확인. [9] 우리 scaffold deny 축자: *"NOT_VERIFIED — only 1 of the required 2 values … Ask the customer for ONE more of: date_of_birth, address, phone_number"* → [13] user 가 전화번호 제공 → [15] `VERIFIED` → [17] `get_current_time` → **[18] `log_verification` matched**. (소결함: [10] 에서 동일 인자 `get_user_information_by_name` 1회 낭비.)
- **[20]~[21]** `get_referrals_by_user` → 15행. 2번 행 축자: `referral_id: f2a3b4c5d6789012 · referred_account_type: Silver Rewards Card · referral_status: IN_PROGRESS · date: 11/13/2025` (= ⓐ, 최신).
- **[21] 직후 — 우리 층의 첫 개입(치명 seed)**: 로그 축자 순서
  `[T2_LEDGER] get_referrals_by_user rows=15 queued to view (non-committed)` → `[T2_FB_VIEW] 1 queued feedback item(s) injected in view` → … → **`[T2_DIAG] raw='Platinum Rewards Card — an error has occurred throughout the process.' → Platinum Rewards Card`**.
  이 답은 `t2_gate_patch.py:4030` 의 `_emit(_sp2["diagnosed_text"].format(answer=_dg[1]), is_answer=True)` 로 **메인 문맥**에 나간다. 선언 문구(`a2/banking_knowledge.settings.json:472`) 축자: *"A separate check was run on the records and the policy definitions, with nothing else from this conversation in front of it. It answers: {answer}"*.
- **[22]~[25]** 무관 read — 우리 레버가 실행시킴. 축자: `[T2_ARBITRATE] push dominated target=submit_referral reqs=…,reads:get_all_user_accounts_by_user_id` (`t2_gate_patch.py:8494`) + `[T2_PIN_READ] pinned unlock_discoverable_agent_tool(agent_tool_name=get_all_user_accounts_by_user_id_3847)` → [25] 결과 *"Bank Accounts: No bank accounts found. Credit Card Accounts: No credit card accounts found."*
- **[30] 1차 오진(우리 DIAG 와 동일 초점)**: 축자 *"it seems there might be an issue with the referral process for the **Platinum Rewards Card**, as indicated by the "ERROR" status."* + 실행하지 않은 도구를 코드블록으로 인쇄(`get_referrals_by_user(user_id="oadeyemi@gmail.com")`).
- **[31] user 정정 축자**: *"I'm a bit confused — I haven't mentioned a Platinum Rewards Card specifically, and I don't know what "ERROR" you're seeing. … And to be clear, I'm asking about my **most recent referral**"*.
- **[32] 시점 — 재무장 레버가 발화하고 **틀린 계열**을 산다**: `[T2_SEARCH_REARM] group=credit_cards 신규 대상 **platinum_rewards_card** (기배달 bronze_rewards_card) — 소진 해제·문서 델타` → `[T2_SEARCH_AGENT] group=credit_cards · 문서 10(뺀 것 0) turn=32` → `[T2_SEARCH_REARM] group=credit_cards 델타 배달 11240자` → `[T2_SEARCH_ON_PROCEED] deny 아님 · 재료 11240자 배달`. 이것이 **검색 예산 3의 세 번째**다(turn 4 / 26 / 32).
- **[40] 모델이 드디어 ⓐ를 옳게 읽는다**: 축자 *"2. **Silver Rewards Card**: - **Referral ID**: f2a3b4c5d6789012 - **Status**: IN_PROGRESS - **Date**: 11/13/2025 … Given that your friend has been approved and is actively using the card"*. 그런데 이 시점에 ⓑ(=$750)는 **뷰에서 이미 잘려 있다**. 그리고 로그상 turn 32 이후 `T2_DOCGROUP`/`T2_GROUPORDER`/`T2_SEARCH_AGENT` 진입이 **0회**다 — `t2_gate_patch.py:8859` 의 `getattr(self,"_t2_searchagent_fired",0) < 3` 가 채널을 닫았다. 즉 **주제가 Silver 로 확정된 바로 그 턴에 검색 채널이 소진 상태였다.**
- **[41] user 가 정확히 그 수치를 묻는다** 축자: *"1) What's the **required spend amount** (and in what time window)? 2) How much has my friend already spent so far, and **how much is remaining**?"*
- **[42]~[43]** 모델의 마지막 KB 시도: `KB_search_bm25{"query":"credit card referral bonus criteria","k":10}` → 반환 16,233자에 **silver_011 없음**(trial0 전체에서 `silver_rewards_card_011` 등장 **1회**=msg[3]뿐). 대신 Bronze $500·Dark Green·Beige 가 실림.
- **[44] 최종 결정 지점(치명)** 축자: *"1. **Required Spend Amount**: The referred individual must spend a certain amount within a specific time window after opening the account. **The exact amount and time window depend on the specific credit card type.**"* — 카드 종류(Silver)는 [40] 에서 스스로 확정해 놓고, 그 종류에 대응하는 수치를 못 댄다. user-sim 발동 조건 ③(구체 금액) **미충족**.
- **[46]~[51]** `get_user_information_by_id{"user_id":"friend_user_5839"}` **3회 반복** → 전부 *"No records found in 'users'."* (env 정상 — 친구는 users 테이블에 없다).
- **[52]~[68]** transfer 국면. `[T2_REQUIRE_DOC_DELIVER] deliver tool=transfer_to_human_agents docs=6 chars=16498 turn=52 … unread=doc_bank_accounts_…_037,…_042,doc_checking_accounts_…_009,doc_credit_cards_(general)_010,_011,_012` — **silver_011 은 이 6편에 없다**. [55] 에서 무관한 *"Internal: Backend Incident 11/13"*(결제 미반영 전용 프로토콜)을 물어와 [56]~[65] 에서 discoverable transfer 도구 2개를 헛돌린다. [69] `###STOP###`.

### 3.2 trial 1 (s373753 · 62 msgs)

- **[2]~[3]** 동일하게 `KB_search_bm25("referral bonus", k=10)` → ⓑ 도착(11,214자 · 동일 바이트).
- **[6]~[9]** `verify_identity` **동일 인자 2회 연속** — 우리 deny 축자: *"NOT_VERIFIED — the account record has not been fetched yet … Use a DIFFERENT identifier each time"* → [10] lookup → [13] `VERIFIED` → **[16] `log_verification` matched**.
- **[18]~[19]** `get_referrals_by_user` → 15행(ⓐ 실재).
- **[19] 직후 — 동일 seed**: `[T2_LEDGER] … rows=15 queued to view` → `[T2_FB_VIEW] 1 queued feedback item(s) injected in view` → **`[T2_DIAG] raw='Platinum Rewards Card — an error has occurred throughout the process.' → Platinum Rewards Card`**.
- **[24] 모델의 첫 산문이 그 답을 그대로 복창한다** 축자: *"it seems there might be an issue with your referral bonus for the **Platinum Rewards Card** due to an error in the process."* + **이미 끝난 신원확인을 1단계로 재계획**.
- **[24]~[44] 국면 붕괴**: `verify_identity` 4회 재호출([26]/[34]/[42] + [12]) · `log_verification` **DUP 2회**([30]/[38] → *"Failed to log verification: Record may already exist."*) — 10턴 소모. 이 구간 로그는 `[T2_PHASE_PRECEDE] cands=2 picked=call_discoverable_agent_tool reqs=['GB1_VERIFY_BEFORE_ACCOUNT_ACCESS']` + `suppressed (same fingerprint)` 6회로, 우리 층이 **이미 충족된 GB1** 을 계속 선행요건으로 들고 있었다.
- **[41] user 정정 축자**: *"I'm asking about my most recent referral (my friend was approved recently and has already been using the card)."*
- **[46] 모델이 그 정정을 정면으로 틀리게 해석** 축자: *"the **most recent** referral related to the Platinum Rewards Card has a status of "ERROR". … - **Referral ID**: 7890abcdef012345 - **Date**: 10/05/2025"* — 목록에 11/13/2025 행이 있는데 10/05 를 "most recent" 로 부른다(날짜 비교 실패).
- **[46]~[47] 이 sim 의 유일한 회복 기회**: `KB_search_bm25{"query":"resolve referral error","k":10}` → 17,820자 반환의 **8번 항목이 silver_011**이고 *"spend at least $750 within 60 days"* 를 담고 있다. msg[47] 은 직전 assistant 이후의 배치라 **[48] 생성에서 원문으로 보였다**.
- **[48] 그 한 턴에서 모델은 $750 을 쓰지 않는다** 축자: *"the referral status "ERROR" indicates a system error occurred during the referral process. To resolve this issue, we can try the following steps: 1. **Retry the Referral Process** … 2. **Escalate Internally**"* — 이후 msg[47] 도 8,000자 캡으로 다이제스트되어 두 번째 기회는 없다.
- **[49] user 축자**: *"I'd rather not "retry" the whole referral … Can you escalate it internally and also confirm which referral you're talking about (the most recent one)?"* → [51] `friend_user_5839` 제공 → **[52] TRANSFER NOTICE** → [53] `###TRANSFER###` → [61] `###STOP###`.
- 이 trial 은 `[T2_SEARCH_REARM] 신규 대상 platinum_rewards_card … 델타 배달 11240자 turn=32` 로 **trial 0 과 동일하게 세 번째 예산을 Platinum 에 썼다**.

### 3.3 분기점

둘 다 실패라 "성공/실패 분기"는 없다. **오진 회복 여부의 분기점 = msg[31] 의 user 정정 유무**다.
- trial 0: user 가 *"I haven't mentioned a Platinum Rewards Card specifically"* 라고 **명시 반박** → [40] 에서 Silver/IN_PROGRESS 로 복귀. 남은 결손은 ⓑ 하나.
- trial 1: user 는 *"most recent referral"* 이라고만 말하고 Platinum 을 부정하지 않음 → [46] 에서 모델이 "most recent = Platinum 10/05" 로 오독하며 **끝까지 복귀 못 함**. 대신 verify 루프로 10턴을 태움.

---

## 4. 레버 발화표 (요청 목록 기준 · 이 sim 만)

| 레버 | s626729 | s373753 | 판정 | 근거(축자) |
|---|---|---|---|---|
| **T2_VIEW_COMPACT**(요청 밖이나 이번 판 핵심) | 1줄(=최초 1회만 인쇄) | 1줄 | **발화·역효과** | `[T2_VIEW_COMPACT] active: 1 tool output(s) digested in view` · msg[3](11,214자) 다이제스트에 `750` 0회 |
| `T2_SEARCH_AGENT` | 11 | 11 | **발화·주제 전부 빗나감** | `[T2_DOCDECIDE] → 'Business Bronze Rewards Card'` / `→ 'Bronze Rewards Card'` · axis2 축자 `문서 110(뺀 것 4: …**doc_credit_cards_silver_rewards_card_010**…)` · 이후 `요청 축 … 모두 처리됨 — 침묵` |
| `T2_SEARCH_REARM` | 4 | 4 | **발화했으나 틀린 계열을 사고 마지막 예산 소진** | `신규 대상 **platinum_rewards_card** (기배달 bronze_rewards_card)` · `델타 배달 11240자 turn=32` · turn 32 이후 `T2_DOCGROUP` 진입 **0회** |
| `T2_PIN_READ` | 7 | 5 | 발화·무관 read 실행 | `pinned unlock_discoverable_agent_tool(agent_tool_name=get_all_user_accounts_by_user_id_3847)` → [25] *"No bank accounts found… No credit card accounts found."* |
| `T2_DEMANDED_STEP` | 7 | 5 | 발화·오형식화 대상 | `head=GB1_VERIFY_BEFORE_ACCOUNT_ACCESS → verify_identity` ×4 · `head=reads:get_all_user_accounts_by_user_id → …` ×2 |
| `T2_CLAIMPROV` | 61 | 51 | **발화·정상** | `unbacked=0` 전 window · `kind-index rescued … 원장에 있다` ×9 · `regen tool_calls=[]` — 날조 0 |
| `T2_REQUIRE_DOC_DELIVER` | 5 | 5 | **발화·너무 늦고 대상이 틀림** | `deliver tool=transfer_to_human_agents docs=6 chars=16498 **turn=52**` · unread 6편에 `silver_rewards_card_011` **없음** |
| `T2_SG_DOCS` | **0** | **0** | 미발화 | ON 에는 있으나(`run_…:91`) 이 태스크 축에서 0회 |
| `T2_FOLLOWUP` | **0** | **0** | 미발화 | halfB 전체 25회 중 이 sim 0회 |
| `FAB_STRIP` | **0** | **0** | 미발화 | halfB 전체에서도 0회 |
| `T2_ARG_PRODUCERS` | **0** | **0** | 미발화(오발화 0·양성 기회 0) | halfB 전체 0회 |
| READ-FIRST | **0** | **0** | 미발화 | 마커 자체가 halfB 0회 |
| (참고) `T2_ARBITRATE`/`T2_RESOLVE` | 15 / 17 | 11 / 10 | **오형식화 재발** | `push dominated target=**submit_referral**` ×6 · `user-action instruct target=**submit_referral**` ×11 — gold user 도구는 `submit_transaction` |
| (참고) `T2_DIAG` | 1 | 1 | **발화·틀린 행을 지목** | `raw='Platinum Rewards Card — an error has occurred throughout the process.'` |
| (참고) `T2_MATERIAL_GATE` | 12 | 11 | 발화·차단 | turn 38 이후 `stop=resolve_cap(정체 3회)` 반복 · `T2_MATERIAL_BYPASS` 는 halfB 전체 **0회**(PIN 밖=OFF) |

---

## 5. 선행 판정과의 대조 (`T7336_FORENSIC_016_2026_08_21.md`)

| 축 | t7336 (016) | **t7346 (016)** | 판정 |
|---|---|---|---|
| 변이 | MISSING 1 (`submit_transaction`) | 동일 (양 trial) | **동일** |
| 요건 문서 도달 | *"$750/60 days 0회 실측 — 요건 문장이 끝까지 문맥 미등장"* · 모델 KB 검색 **0회** | **모델이 msg[2] 에서 스스로 KB 를 쳤고 msg[3] 에 `silver_rewards_card_011` + `spend at least $750 within 60 days` 가 도착했다**(양 trial) | **상이 — 결손이 "미도달"에서 "도달 후 우리 층이 뷰에서 삭제"로 이동** |
| 봉쇄 기전 | 축-소진 영구 잠금(`모두 처리됨 — 침묵`) | ⑴`T2_VIEW_MSG_CAP=8000` 다이제스트 ⑵검색 예산 `<3` 소진(재무장이 3번째를 씀) | **상이(새 기전 2종)** |
| 처방 1(`T2_SEARCH_REARM`) | 후보로만 제시 | **구현·등재·발화 확인**(`go_stack.sh:95`) — 그러나 신규 대상이 `platinum_rewards_card` 라 **못 삼**, 게다가 마지막 예산을 소비 | **개입했으나 0 매수** |
| `T2_DIAG` Platinum 오지목 | 로그로 포착(*"이 초점을 포착"*) | **동일 문자열 재발**, 그리고 trial 1 에서는 모델의 첫 산문이 그것을 그대로 복창 | **동일(재발)·인과 지위 상승** |
| ACTIONREQ `submit_referral` 오형식화 | 재발 | **재발**(양 trial 전 window) | **동일(재발)** |
| 진단 깊이 | 요건의 존재까지 추론 후 수치 없이 transfer | trial0 동일 지점 도달(msg[44] *"The exact amount … depend on the specific credit card type"*) / trial1 은 Platinum 에 갇혀 더 얕음 | trial0 동일·trial1 퇴행 |
| 기준선 `t7328_halfB` | — | 양 trial 실패 · MISSING `submit_transaction` 동일 · `silver_rewards_card_011` 등장 **1회**로 동일 | **3런 연속 동형** |

---

## 6. 원인 확정 (4주체 귀속)

### CONFIRMED — our_layer (코드 경로 지목)

1. **뷰-압축이 요건 문장을 결정점 전에 삭제했다.** `t2_gate_patch.py:6224-6231`(`msg_cap` per-메시지 다이제스트) + `go_stack.sh:152 T2_VIEW_MSG_CAP=8000` + `go_stack.sh:115 T2_VIEW_COMPACT=1`.
   실측: msg[3] = 11,214자 > 8,000 ⇒ 도착 다음 생성부터 head 300 + tail 150 으로 치환, 그 다이제스트에 `750` **0회**. `_compact_view` 의 `batch` 면제 규칙상 원문 노출 턴은 **msg[4] 하나**뿐이고, 그 시점에 referral 목록(ⓐ)은 아직 없다.
   ⇒ **trial 0 에서 ⓐ(Silver·IN_PROGRESS)와 ⓑ($750)가 모델 생성-뷰에 동시에 있었던 턴은 0회.** 이 sim 의 KB 출력은 전부 8,000자 초과라 예외 없이 같은 처분을 받는다.
   ⚠반증 시도: 다이제스트 문구는 *"re-call the same tool if you need the details again"* 라는 탈출구를 준다(`T2_NO_DIGEST_REEXEC` 미설정 = 면제 살아 있음, `t2_gate_patch.py:5723,5732`). 그러나 다이제스트가 보여 주는 것은 Beige Account 와 `[axis] matches: 36 documents…` 뿐이라 **거기에 Silver $750 이 있었다는 단서가 남지 않는다**. 탈출구는 열려 있으나 표지가 없다.

2. **격리 진단 서브가 틀린 행을 지목하고, 그 답이 "별도 검사"라는 권위 문구로 메인에 실렸다.** `t2_gate_patch.py:4025-4030`(`_emit(_sp2["diagnosed_text"].format(answer=_dg[1]), is_answer=True)`) + `t2_ledger.py:1055-1099`(`diagnose_choice`) + 선언 키 `a2/banking_knowledge.settings.json:471-472`(`diagnose_prompt` / `diagnosed_text`).
   실측: 양 trial 모두 `[T2_DIAG] raw='Platinum Rewards Card — an error has occurred throughout the process.'` 가 `get_referrals_by_user rows=15 queued to view` **직후** 발화. trial 1 의 모델 첫 산문 [24] 는 *"an issue with your referral bonus for the Platinum Rewards Card due to an error in the process"* 로 이를 그대로 복창한다.
   구조적 이유: `diagnose_prompt` 축자는 *"One of these records did not pay out."* 인데 15행 중 미지급 후보가 **7행**(IN_PROGRESS 1 · REJECTED 4 · ERROR 1 · …)이라 **답이 유일하지 않다**. 유일하게 만드는 술어(*"손님이 묻는 것은 최신 건"*)는 손님 발화에만 있는데, 이 서브는 설계상 대화를 한 글자도 못 받는다(`t2_ledger.py:1071` 축자 *"문맥에는 **대화가 한 글자도 안 들어간다**"*). 손님 요구를 서브에 주는 노브(`T2_SUB_REQUIREMENT`, `t2_gate_patch.py:3414`)는 **검색 서브 전용**이고 진단 서브에는 경로가 없다. 게다가 답은 `agent._t2_diag` 로 **메모이즈**(`t2_ledger.py:1077-1079,1098`)되어 user 정정([31] *"I haven't mentioned a Platinum Rewards Card specifically"*) 이후에도 갱신되지 않는다.
   ⚠반증 시도: t7336 에서도 같은 문자열이 나왔고 그때 모델은 user 정정으로 회복했다 ⇒ **결정적이지는 않다**. 발화·배달·복창은 CONFIRMED, "이것을 빼면 통과한다"는 **UNPROVEN**(격리 A/B 필요).

3. **재무장 레버가 마지막 검색 예산을 틀린 계열에 썼고, 주제가 Silver 로 확정된 턴에는 채널이 닫혀 있었다.** `t2_gate_patch.py:3242-3249`(`T2_SEARCH_REARM` 진입) + `t2_gate_patch.py:3344-3390`(델타 배달) + **예산 상한 `t2_gate_patch.py:8859` `getattr(self,"_t2_searchagent_fired",0) < 3`** + `go_stack.sh:95 T2_SEARCH_REARM=1`.
   실측: 예산 3 = turn 4(business_credit_cards → *Business Bronze*) · turn 26(credit_cards → *Bronze*) · **turn 32(재무장 → platinum_rewards_card · 11,240자)**. trial 0 이 Silver 를 확정한 것은 **msg[40]** 이고, turn 32 이후 로그에 `T2_DOCGROUP`/`T2_GROUPORDER`/`T2_SEARCH_AGENT` 진입이 **0회**다(예산 가드가 블록 자체를 건너뛰므로 침묵 로그조차 안 남는다).
   왜 Platinum 이 걸렸나: `_rearm_subjects`(`t2_gate_patch.py:3083-3128`) 의 수요 술어 ⑵는 **user/assistant 발화만** 본다(축자 *"⚠도구 출력은 안 본다 — 레코드 덤프는 전 계열명을 담을 수 있어 수요가 아니다"*, `t2_gate_patch.py:3096-3098`). 재무장 시점(turn 32)에 "Silver Rewards Card" 는 **도구 출력(msg[21])에만** 있었고 산문에는 Platinum 뿐이었다 ⇒ **재무장 술어는 모델의 오진 하류에 있고, 오진이 가리키는 계열을 강화한다.**
   ⚠반증 시도: 예산 3은 재무장 도입 전에도 있던 상한이므로 "재무장이 예산을 훔쳤다"는 주장은 turn 32 배달이 없었다면 msg[40] 이후 재무장이 Silver 로 발화했을 것이라는 가정에 의존한다. 그 반사실은 **UNPROVEN**. 확정된 것은 ⓐ 재무장이 발화했고 ⓑ 대상이 gold 와 무관한 Platinum 이었고 ⓒ Silver 확정 이후 채널 진입이 0회였다는 사실뿐이다.

4. **`submit_referral` 오형식화 재발 + 이미 충족된 선행요건 반복.** `t2_gate_patch.py:8494`(`[T2_ARBITRATE] push dominated target=submit_referral`) · `t2_gate_patch.py:8670`(`[T2_RESOLVE] user-action instruct target=submit_referral`).
   실측: gold user 도구는 `submit_transaction` 인데 전 window 에서 `submit_referral` 을 지목했고, 그 요건 `reads:get_all_user_accounts_by_user_id` 를 `T2_PIN_READ` 가 실행시켜 [22]~[25] 2턴을 태웠으며 결과는 *"No bank accounts found… No credit card accounts found"*(이 빈 결과가 이후 transfer 사유 진술에 재사용된다). trial 1 에서는 `T2_PHASE_PRECEDE … reqs=['GB1_VERIFY_BEFORE_ACCOUNT_ACCESS']` 가 [16] 에서 이미 끝난 신원확인을 6회 더 선행요건으로 들어 [24]~[44] 의 verify 루프(DUP 2건 포함)를 만들었다.
   [[66]] 무수리 판정 유지 항목이므로 새 레버 대상은 아니나, **이번 판에서도 대가가 실측됐다**(무관 read 2턴 + trial1 10턴).

### model

- **join 실패(F3/⋈)**: trial 0 [40] 에서 *"Silver Rewards Card … IN_PROGRESS"* 를 스스로 확정하고도 [44] 에서 *"The exact amount and time window depend on the specific credit card type"* 로 헤지 — 카드 종류를 아는 상태에서 그 종류의 수치를 대지 못했다. (단 그 시점 ⓑ는 우리 층이 이미 뷰에서 지운 뒤다 ⇒ **부하 귀속 전에 our_layer 가 먼저 물린다**([[55]] 0단계).)
- **재검색 표적화 실패**: 다이제스트 탈출구가 열려 있었는데 `KB_search_bm25` 재호출 3회의 질의가 *"check referral bonus status"* / *"credit card referral bonus criteria"* / *"transfer protocol and reason-code documents"* 로, **한 번도 "Silver Rewards Card" 를 질의에 넣지 않았다**([[63]] 빼기 불가와 동형: 후보를 좁히는 닫힌 술어를 스스로 못 건다).
- **trial 1 날짜 비교 실패**: 목록에 `11/13/2025` 가 있는데 [46] 에서 `10/05/2025` 를 *"the most recent referral"* 이라 진술. 이것은 문맥에 실재하는 값의 오독이므로 **순수 모델 결손**.
- **trial 1 단일 회복 기회 낭비**: msg[47] 원문(17,820자·8번 항목 silver_011)이 보이는 유일한 생성 [48] 에서 $750 을 쓰지 않고 *"Retry the Referral Process / Escalate Internally"* 로 갔다.

### env — 정상

- `get_user_information_by_id("friend_user_5839")` → *"No records found in 'users'."* (친구는 users 테이블에 없음 = 스펙).
- `get_credit_card_transactions_by_user("86e92f639e")` → *"No records found"*(본인 카드 거래 없음 = 스펙).
- `[DUPLICATE-READ]` 스텁·`Failed to log verification: Record may already exist.` 전부 스펙 내.

### user_sim — 정상 (면책 아님·[[21]])

스펙 축자를 정확히 지켰다: [31] 최신-건 정정, [33]·[45]·[51] `friend_user_5839` 제공, [41] 요구 금액을 **명시적으로 두 번 질문**, [53] *"If the agent … offers to transfer you to a human, accept that"* 대로 수용. 발동 조건 ③(구체 금액)이 끝내 충족되지 않아 `submit_transaction` 을 안 쓴 것이므로 **에이전트-측 흡수 대상**이다.

### 한 줄 결론

> **016 은 "모델이 정책을 안 찾아서" 실패한 것이 아니다. 모델은 msg[2] 에 스스로 찾았고 msg[3] 에 `$750 within 60 days` 가 도착했다. 우리 층의 `T2_VIEW_MSG_CAP=8000` 다이제스트가 그것을 다음 턴에 지웠고, 그 뒤 그 값을 다시 실어 줄 수 있었던 유일한 채널(검색 예산 3)의 마지막 한 칸을 `T2_SEARCH_REARM` 이 모델의 오진(Platinum)에 썼으며, 그 오진의 씨앗은 우리 진단 서브의 `diagnosed_text` 였다.**

---

## 7. 처방 후보 (수리 실행 없음 · [[62]] 4문 + [[70]] A/B 의무)

각 후보는 **격리 프로브로 결손을 먼저 재고**(x46N·N_neg) 그 다음에만 등재한다.

1. **다이제스트에 표지를 남긴다(최우선·전달 축소만)**: `_compact_view` 의 다이제스트가 head/tail 순수 절단이라 *"거기에 무엇이 있었는지"* 가 0이다. 닫힌 술어 후보 = 잘린 tool 출력이 KB 결과일 때 **문서 id 목록만**(본문 0자·선언 id 축자 그대로) 다이제스트에 남기기. 엔진의 내용 판정 0([[59]]), 선택 0([[62]]③) — 모델이 재호출할지 여부는 그대로 모델 몫. 격리 질문: *"문서 id 목록만 보이면 모델이 그 doc 을 다시 부르는가"*.
2. **재무장 수요 술어의 관측 창을 도구 출력까지 넓힐지 여부를 격리로 잰다**: 현재 ⑵는 user/assistant 발화만 본다(`t2_gate_patch.py:3096`). 016 에서 필요한 계열명은 **레코드 덤프에만** 있었다. 다만 주석이 경고하는 대로 덤프는 전 계열명을 담아 과폭발한다 ⇒ 후보는 "전체 덤프" 가 아니라 **A2 가 지목한 ledger spec 의 `group_field` 값**(닫힌 집합·`get_referrals_by_user` 행의 `referred_account_type`)만. 부작용 계측 = 배달 부피·지연·over-action.
3. **검색 예산(`<3`)과 재무장을 분리 계측**: 재무장 배달이 예산 칸을 소비하는 현재 배선(`8859`)에서, 재무장 전용 칸을 별도로 둘 때의 ± 를 A/B 로. ⛔예산을 그냥 늘리는 것은 [[57]] 반대편 실패(무제한=반복)이므로 **칸 분리**만 후보.
4. **`diagnose_prompt` 의 미결정성 제거**: 현재 축자 *"One of these records did not pay out."* 는 15행 중 7행이 후보라 답이 유일하지 않다. 후보 = ⑴프롬프트를 *"미지급 행이 여럿이면 전부 나열하라"* 로 바꿔 **단일 지목을 없애기**(엔진이 고르지 않게 하는 방향 = [[62]]③ 준수) 또는 ⑵`T2_SUB_REQUIREMENT` 동형으로 **손님 요구 인용문(축자 검산 통과분)** 을 진단 서브에도 주기. ⚠⑵는 격리 계약(*"대화가 한 글자도 안 들어간다"*)을 건드리므로 x269 형 A/B 없이 켜지 말 것.
5. **`diagnosed_text` 의 권위 문구 조정(끄기 아님·[[70]] 절충)**: *"A separate check was run … It answers: {answer}"* 는 단일 답을 최고 권위로 제시한다. 후보 = 답이 미결정일 때(후보 행 ≥2) 문구를 **후보 나열형**으로 분기. 태스크 id 로 켜지 말 것([[05]]) — 조건은 "그룹 축 미지급 후보 수 ≥ 2" 라는 도메인-일반 닫힌 술어.
6. **`T2_REQUIRE_DOC_DELIVER` 의 배치 전진**: 현재 turn 52(transfer 국면)에서만 미독 문서를 표면화한다. t7336 처방 2 와 동일한 미해결 항목 — **진단 국면(레코드 특정 직후·결론 발화 전)** 으로 전진 배치하는 변형. 016 의 `required_documents` 에 `doc_credit_cards_silver_rewards_card_011` 이 있는데 이 레버가 실제로 실은 6편에는 없다는 점도 함께 볼 것.
7. **ACTIONREQ `submit_referral` 오형식화는 재론 없음**([[66]] 무수리 유지). 다만 이번 판의 대가(무관 read 2턴 · trial1 verify 루프 10턴 · DUP 2건)를 부호표에 병기한다.

---

*작성 근거: `bank_t7346_halfB_20260822.results.json.gz`(task_016 2 sims 전 메시지 정독) · `bank_t7346_halfB_20260822.log.gz` `[sim=task_016#s626729]` 439라인 / `[sim=task_016#s373753]` 349라인 전수 · 변이는 `t2_forensic.mutation_diff` 정본 · 대조 `bank_t7328_halfB_20260819r2` · 선행 `T7336_FORENSIC_016_2026_08_21.md`. 로컬 전용·SSH 0·git 변경 0·코드 수정 0.*
