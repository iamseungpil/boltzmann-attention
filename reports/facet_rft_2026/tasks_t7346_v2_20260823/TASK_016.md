# t7346 halfB `task_016` per-step 포렌식 v2 (재조사 · 2026-08-23)

- 대상 런: `bank_t7346_halfB_20260822` · 런 sha **`ee18d797`** (`sim_results/bank_t7346.meta.json`).
- 대상: `task_016` **2 trial 전부 reward 0.0** · trial0 = `s626729`(seed 626729 · 70 msgs · term `user_stop` · 755.8s) / trial1 = `s373753`(seed 373753 · 62 msgs · term `user_stop` · 688.0s).
- 방법: 채점축 = `sim['reward_info']` 직독 · 변이 = 정본 `t2_forensic.mutation_diff`(손 비교기 0·C583ⓐ) · 궤적 = 전 메시지 정독 · 레버 = `bank_t7346_halfB_20260822.log.gz` 의 `[sim=task_016#…]` 788라인 전수(정규식 재계수) · 대조 = `bank_t7336_halfB_20260821b` · `bank_t7328_halfB_20260819r2`.
- ⛔**줄번호 규율**: 인용 줄번호는 전부 **런 sha `ee18d797` 프리즈본**(`git show ee18d797:…`)이다. 워킹트리(`4ea50a09`)와 오프셋이 있다(FAILURE_MASTER 함정 1). 워킹트리 줄번호를 쓴 곳은 명시한다.
- ⛔수리·코드 수정 0 · git 변경 0 · SSH 0. gold(`reward_info`)는 진단용으로만([[23]]).

---

## 1. 채점축 — **DB-해시**

```
reward 0.0 · reward_basis ["DB"] · reward_breakdown {"DB": 0.0}
db_check {"db_match": false, "db_reward": 0.0}
env_assertions [] · nl_assertions null · communicate_checks null
```

**ACTION 축이 아니다.** `action_checks` 는 진단 보조로만 읽는다(양 trial 동일):

| action_id | requestor | name / arguments | action_match |
|---|---|---|---|
| `016_0` | assistant | `log_verification{name:"Oluwaseun Adeyemi", user_id:"86e92f639e", address:…, email:…, phone_number:"404-555-0672", date_of_birth:"02/14/1988", time_verified:"2025-11-14 03:40:00 EST"}` | **true** |
| `016_1` | **user** | `submit_transaction{user_id:"friend_user_5839", credit_card_type:"Silver Rewards Card", merchant_name:"Best Buy", amount:750, category:"Shopping"}` | **false** |

`016_1` 의 requestor 는 **user** 이고 `task.user_tools == ["submit_transaction","apply_for_credit_card"]` 다 — **에이전트는 이 write 를 직접 칠 수 없다.** DB 델타를 만드는 유일한 경로는 *에이전트가 손님에게 구체 금액을 말해 손님이 찍게 하는 것*이다. user-sim 스펙 축자:

> *"**Tool usage:** You have access to `submit_transaction`. ONLY use this tool if the agent has: - Actually looked up your referrals and identified which one you're asking about - Explained that your friend hasn't met a spending requirement yet - **Told you a specific dollar amount** your friend needs to spend"*
> *"submit a transaction … the **credit card type the agent mentioned**, and an amount **exactly the same as the requirement that the agent specified**"*

⇒ 채점 손실의 필요조건은 하나로 압축된다: **"Silver Rewards Card 는 $750" 을 에이전트가 발화했는가.**
task 선언의 `required_documents` 에도 그 문서가 있다: `doc_credit_cards_credit_cards_(general)_001` · `_002` · **`doc_credit_cards_silver_rewards_card_011`**.

---

## 2. 변이표 (`t2_forensic.mutation_diff` · 정본)

| 종류 | trial 0 (`s626729`) | trial 1 (`s373753`) |
|---|---|---|
| **matched** | 1 — `log_verification` (msg 18 · `ok=True`) | 1 — `log_verification` (msg 16 · `ok=True`) |
| **MISSING** | **1 — `submit_transaction{friend_user_5839, Silver Rewards Card, Best Buy, 750, Shopping}`** | **동일 1건** |
| WRONGARG | 0 | 0 |
| EXTRA | 0 | 0 |
| **DUP** | 0 | **0** |
| **BLOCKED** | 0 | **2 — `log_verification` (msg 30·38 · `deny="env"` · marker `"Failed to log verification: Record may already exist."`)** |

⚠**선행 표 정정 확인**: `FAILURE_MASTER__20260822.md` §1 은 trial1 을 *"DUP 2"* 로 적었다. 정본 재실행 결과는 **DUP 0 · BLOCKED 2** 이고, 이는 `MASTER_DUP_CORRECTION_2026_08_23.md` §47 의 정정(*"DUP 0. matched 3→1 · blocked 0→2"*)과 **일치**한다. 본 v2 는 정정본을 따른다.

WRONGARG 0 ⇒ 필드별 대조는 대상 없음(보낸 인자 자체가 0건). BLOCKED 2 는 env 실패 반환이라 DB 를 안 바꾼다.
⇒ **양 trial 모두 채점 손실은 MISSING `submit_transaction` 단 하나.**

---

## 3. step-by-step — 결정 지점 축자 추적

### 3.0 필요한 두 값이 문맥에 실재했는가 (양 trial 공통)

| 값 | 출처 | 도착 |
|---|---|---|
| **ⓐ 대상 = Silver Rewards Card · IN_PROGRESS · 11/13/2025** (15건 중 최신) | `get_referrals_by_user` 출력 축자 `2. Record ID: f2a3b4c5d6789012 / referred_account_type: Silver Rewards Card / referral_status: IN_PROGRESS / date: 11/13/2025` | t0 msg[21] · t1 msg[19] (2,844자 · 캡 미만 → **끝까지 전문 생존**) |
| **ⓑ 요건 = `The referred person must be approved and spend at least $750 within 60 days of account opening`** (`doc_credit_cards_silver_rewards_card_011`) | 모델 **자신의** `KB_search_bm25{"query":"referral bonus","k":10}` 결과의 **10번째**(score 8.3350) | **양 trial 모두 msg[3]** (11,214자) |

**ⓑ는 도착했다.** t7336 판정(*"요건 문장이 끝까지 문맥 미등장"*)은 이 런에 해당하지 않는다.

#### 3.0.1 우리 층의 뷰-압축 재현 (오프라인·프리즈본·[S])

`_compact_view`(프리즈본 `t2_gate_patch.py:6193`)를 **런 sha 함수 그대로 꺼내** 실제 메시지 열에 재적용했다.
검증: `frozen(ee18d797)._compact_view == worktree(4ea50a09)._compact_view` → **True**(바이트 동일 ⇒ 재현은 런 sha 유효).

파라미터는 프리즈본 `go_stack.sh` 축자 그대로:
`:115 T2_VIEW_COMPACT=1` · `:151 T2_VIEW_COMPACT_MINTOTAL=60000` · `:152 T2_VIEW_MSG_CAP=8000`.

| | assistant 생성 수 | **`$750` 이 생성-뷰에 보인 턴 (실제 런 · CAP=8000)** | 반사실 (CAP=0) |
|---|---|---|---|
| trial 0 | 35 | **1회 — gen@msg4 뿐** | **26회** (msg 4,6,…,54) |
| trial 1 | 31 | **2회 — gen@msg4, gen@msg48** | **30회** (msg 4,…,62) |

기전: `t2_gate_patch.py:6224`(프리즈본) `if msg_cap and len(c) > int(msg_cap): _hit = True` → `:6229` 다이제스트 = **head 300 + tail 150 순수 절단**.
실측 다이제스트 내용:

```
HEAD300 : "1. Beige Account Referral Program\n   ID: doc_business_checking_accounts_beige_012 … $500 for each qualifying referral"
TAIL150 : "… within the 7 limit\n\n\n[Timing: retrieval=1ms, reranking=0ms, total=1ms]\n[axis] matches: 36 documents contain all of these words; 32 shown (4 not shown)"
→ 다이제스트 안에 '750' 0회 · 'Silver' 0회 · 문서 id 0개
```

`batch` 면제 규칙(마지막 assistant 이후의 tool 출력만 전문)상 msg[3] 이 원문으로 보이는 생성은 **msg[4] 하나**뿐이고, 그 시점에는 ⓐ(referral 목록)가 아직 존재하지 않는다.

이 sim 의 KB 출력은 **전부 8,000자 초과**라 예외가 없다:
t0 = 11,214 / 16,105 / 16,233 / 16,188 · t1 = 11,214 / 17,820 / 29,238.

**⚠계기 결함(성적 무관)**: `[T2_VIEW_COMPACT] active: 1 tool output(s) digested in view` 는 프리즈본 `:6767` `if _dg and not getattr(self,"_t2_vc_logged",False)` 로 **sim 당 1회만** 인쇄된다. 최종 턴 실제 다이제스트 수는 **t0 5개 · t1 4개**다. 로그 1줄을 "한 번 일어난 일"로 읽으면 오독한다([[55]] 0단계).

### 3.1 trial 0 (`s626729`) — 결정 지점 5개

**[2]→[3] ⓑ 도착.** `KB_search_bm25{"query":"referral bonus","k":10}` → 10번째가 `doc_credit_cards_silver_rewards_card_011`, 축자 *"- The referred person must be approved and spend at least $750 within 60 days of account opening"*. **다음 생성부터 뷰에서 삭제**(§3.0.1).

**[4]~[19] 신원확인 — 정상 통과.** [9] 우리 deny 축자: *"NOT_VERIFIED — only 1 of the required 2 values the customer gave match the account record (matched: email). Ask the customer for ONE more of: date_of_birth, address, phone_number"* → [13] user 전화번호 → [15] `VERIFIED` → [17] `get_current_time` → **[18] `log_verification` matched**. (소결함: [10] 동일 인자 `get_user_information_by_name` 1회 낭비.)

**[20]~[21] ⓐ 도착.** 15행. 최신 = 2번 행(Silver / IN_PROGRESS / 11/13/2025).

**[21] 직후 — 결정 지점 ①: 우리 격리 진단 서브가 틀린 행을 지목.** 로그 축자 순서:
```
[T2_LEDGER] get_referrals_by_user rows=15 queued to view (non-committed)
…
[T2_DIAG] raw='Platinum Rewards Card — an error has occurred throughout the process.' → Platinum Rewards Card
```
이 답은 프리즈본 `t2_gate_patch.py:4030` `_add += _emit(_sp2["diagnosed_text"].format(answer=_dg[1]), is_answer=True)` 로 메인 뷰에 나간다. 선언 문구 축자(프리즈본 `a2/banking_knowledge.settings.json:472`):
> *"A separate check was run on the records and the policy definitions, with nothing else from this conversation in front of it. It answers: {answer}"*

**[22]~[25] 무관 read 2턴.** `[T2_ARBITRATE] push dominated target=submit_referral reqs=GB1_…,GB3_…,reads:get_all_user_accounts_by_user_id` → `[T2_PIN_READ] pinned unlock_discoverable_agent_tool(agent_tool_name=get_all_user_accounts_by_user_id_3847)` → [25] 결과 축자 *"Bank Accounts: No bank accounts found. Credit Card Accounts: No credit card accounts found."*

**[30] 결정 지점 ②: 모델의 1차 진단이 우리 DIAG 와 같은 초점.** 축자:
> *"it seems there might be an issue with the referral process for the **Platinum Rewards Card**, as indicated by the "ERROR" status."*

그리고 실행하지 않은 도구를 산문에 인쇄: `` `get_referrals_by_user(user_id="oadeyemi@gmail.com")` ``.

**[31] user 정정** 축자:
> *"I'm a bit confused — I haven't mentioned a Platinum Rewards Card specifically, and I don't know what "ERROR" you're seeing. … And to be clear, I'm asking about my **most recent referral** — my friend was recently approved and has already started making purchases on the card"*

**[32] 시점 — 결정 지점 ③: 재무장 레버가 손님의 *부정문*을 수요로 읽고 Platinum 을 산다.**
```
[T2_SEARCH_REARM] group=credit_cards 신규 대상 platinum_rewards_card (기배달 bronze_rewards_card) — 소진 해제·문서 델타
[T2_SEARCH_AGENT] group=credit_cards · 문서 10(뺀 것 0: ) · now=2025-11-14 turn=32
[T2_SEARCH_REARM] group=credit_cards 델타 배달 11240자 (문서 10·뺀 것 0) turn=32
[T2_SEARCH_ON_PROCEED] deny 아님 · 재료 11240자 배달
```
술어부(프리즈본 `:3083 _rearm_subjects`) ⑵는 `:3119` `if getattr(m,"role",None) in ("user","assistant")` — **user/assistant 산문만** 본다(도구 출력 제외는 주석에 명시). turn 32 시점 산문에 `Silver Rewards Card` 는 **0회**, `Platinum Rewards Card` 는 [30](모델 오진)·[31](손님의 **부정문**)·[32] 에 있다. ⇒ **재무장은 오진의 하류에 있고 오진이 가리키는 계열을 강화한다.**

**[40] 모델이 ⓐ를 옳게 읽는다 — 그런데 ⓑ는 이미 없다.** 축자:
> *"2. **Silver Rewards Card**: - **Referral ID**: f2a3b4c5d6789012 - **Status**: IN_PROGRESS - **Date**: 11/13/2025 … Given that your friend has been approved and is actively using the card"*

이 턴은 궤적 전체에서 assistant 산문이 `Silver Rewards Card` 를 부른 **유일한** 턴이다(§3.3 표). §3.0.1 대로 이 시점 뷰에 `$750` 은 없다.

**[41] 손님이 정확히 그 수치를 묻는다** 축자:
> *"1) What's the **required spend amount** (and in what time window)? 2) How much has my friend already spent so far, and **how much is remaining**? 3) Once they hit the requirement, how long does it usually take for the **referral bonus to post**"*

**[42]~[43] 모델의 마지막 KB 시도 — 질의에 'Silver' 가 없다.** `KB_search_bm25{"query":"credit card referral bonus criteria","k":10}` → 16,233자 반환에 `silver_rewards_card_011` **부재**(대신 Bronze $500 · Dark Green · Beige · Business Platinum 등). 궤적 전체에서 `doc_credit_cards_silver_rewards_card_011` 등장 = **1회(msg[3]) 뿐**.

**[44] 결정 지점 ④(치명): 카드는 알고 수치는 못 댄다.** 축자:
> *"1. **Required Spend Amount**: The referred individual must spend a certain amount within a specific time window after opening the account. **The exact amount and time window depend on the specific credit card type.**"*

⇒ user-sim 발동 조건 ③(*"Told you a specific dollar amount"*) **미충족** ⇒ `submit_transaction` 영구 미발생 ⇒ MISSING 확정.

**[46]~[51]** `get_user_information_by_id{"user_id":"friend_user_5839"}` **3회** → 전부 *"No records found in 'users'."* (env 스펙대로 — 친구는 users 테이블에 없다). `T2_READ_DEDUP` loop-break 는 `T2_READ_DEDUP_LOOP_K` 기본 **3**(프리즈본 `:5721`)이라 3회째까지는 스텁이 안 걸린다.

**[52]~[69]** transfer 국면. `[T2_REQUIRE_DOC_DELIVER] deliver tool=transfer_to_human_agents docs=6 chars=16498 turn=52` — 실린 6편은 `bank_accounts_(general)_037/_042`·`checking_accounts_(general)_009`·`credit_cards_(general)_010/_011/_012` 로 **`silver_rewards_card_011` 없음**. [55] 에서 무관한 *"Internal: Backend Incident 11/13"* 를 물어 [56]~[68] 에 discoverable transfer 도구 2개를 헛돌린다. [69] `###STOP###`.

### 3.2 trial 1 (`s373753`)

**[2]→[3]** 동일 바이트로 ⓑ 도착(11,214자).

**[6]~[9]** `verify_identity` **동일 인자 2회 연속** — 우리 deny 축자: *"NOT_VERIFIED — the account record has not been fetched yet…"* → [10] lookup → [13] `VERIFIED` → **[16] `log_verification` matched**.

**[18]~[19] ⓐ 도착** → 직후 **동일 seed**: `[T2_LEDGER] … rows=15 queued to view` → `[T2_DIAG] raw='Platinum Rewards Card — an error has occurred throughout the process.' → Platinum Rewards Card`.

**[24] 결정 지점 ①': 모델의 첫 산문이 그 답을 복창한다.** 축자:
> *"it seems there might be an issue with your referral bonus for the **Platinum Rewards Card** due to an error in the process."*

⚠**출처 검산**: 문자열 *"an error has occurred throughout the process"*(=`doc_credit_cards_credit_cards_(general)_001` 의 ERROR 정의문)는 이 시점 **커밋 전사(압축 뷰)에 0회**다 — 오프라인 재현으로 gen@msg20·22·24·26·28·30·32 전부 `False` 확인(그 정의문의 유일한 전사 출처인 msg[3] 이 다이제스트됨). 반면 상태 **토큰** `ERROR` 는 원장 덤프에 있어 `True`. ⇒ *"ERROR 라는 상태가 있다"* 는 모델이 원장에서 얻을 수 있었고, *"그 정의는 이것이며 문제 행은 Platinum 이다"* 라는 **지목·정의 문장의 유일한 경로는 우리 층 주입**(`diagnosed_text` + `status_meanings_text`)이다.

**[24]~[44] 국면 붕괴 10턴.** `verify_identity` 4회 재호출 · `log_verification` **BLOCKED 2회**([30]/[38] → *"Failed to log verification: Record may already exist."*). 이 구간 로그에 `[T2_PHASE_PRECEDE] … reqs=['GB1_VERIFY_BEFORE_ACCOUNT_ACCESS']` 가 반복 — 우리 층이 **[16]에서 이미 충족된** GB1 을 계속 선행요건으로 들고 있었다.

**[41] user 정정** 축자: *"I'm asking about my **most recent referral** (my friend was approved recently and has already been using the card)."*

**[46] 결정 지점 ②': 모델이 그 정정을 정면으로 오독.** 축자:
> *"the **most recent** referral related to the Platinum Rewards Card has a status of "ERROR". … - **Referral ID**: 7890abcdef012345 - **Date**: 10/05/2025"*

목록에 `11/13/2025` 행이 있는데 `10/05/2025` 를 "most recent" 라 부른다 — **문맥에 실재하는 값의 날짜 비교 실패 = 순수 model 결손**.

**[46]~[48] 이 sim 의 유일한 회복 기회 — 그리고 놓친다.** `KB_search_bm25{"query":"resolve referral error","k":10}` → **17,820자 반환의 8번 항목이 `doc_credit_cards_silver_rewards_card_011`** 이고 *"spend at least $750 within 60 days"* 를 담고 있다. msg[47] 은 직전 assistant 이후 배치라 **[48] 생성에서 원문 노출**(오프라인 재현: gen@msg48 `$750=True`). 그 턴의 모델 답 축자:
> *"the referral status "ERROR" indicates a system error occurred during the referral process. To resolve this issue, we can try the following steps: 1. **Retry the Referral Process** … 2. **Escalate Internally**"*

$750 을 한 번도 언급하지 않는다. **이 턴은 우리 층 주장 ①의 반증 자료다**(§6 ⚠). 참고로 같은 턴은 `[T2_MATERIAL_GATE] stop=resolve_cap(정체 3회) turn=48` 로 우리 재료 채널도 닫혀 있었다.

**[49]~[61]** [49] *"I'd rather not "retry" the whole referral … Can you escalate it internally and also confirm which referral you're talking about (the most recent one)?"* → [51] `friend_user_5839` 제공 → [52] TRANSFER NOTICE → [53] `###TRANSFER###` → [58] `transfer_to_human_agents{summary:"… for a **Platinum Rewards Card** referral …", reason:"account_ownership_dispute"}` → [61] `###STOP###`. **끝까지 Platinum.**

### 3.3 분기점 — "회복 여부"의 분기이지 성패 분기는 아니다

둘 다 실패라 성패 분기는 없다. 갈린 것은 **오진 회복 여부**이고, 분기 지점은 **[31] 손님이 Platinum 을 명시 부정했는가**다.

| | Silver 를 산문에서 부른 턴 | Platinum 을 산문에서 부른 턴 | `$750` 이 뷰에 있던 턴 | **교집합** |
|---|---|---|---|---|
| t7346 t0 | **[40]** | [30],[32],[40] | [4] | **∅** |
| t7346 t1 | **없음** | [24],[32],[40],[44],[46],[52],[58],[60] | [4],[48] | **∅** |
| t7328 t0 | 없음 | [24] | [4] | ∅ |
| t7328 t1 | 없음 | [24],[30] | [4] | ∅ |

⇒ **ⓐ의 확정(=Silver 를 대상이라 부름)과 ⓑ의 가시성이 동시에 성립한 생성은 4 sim 전부 0회다.**
반사실(CAP=0)에서는 t0 26회 · t1 30회의 생성에서 ⓑ가 보이며, 그 안에 t0 [40](Silver 확정 턴)·[42]·[44](치명 답변)가 **전부 포함**된다.

---

## 4. 레버 발화표 (정규식 전수 재계수 · 선행 표 정정)

계수 규약: 줄 **선두 마커**만 센다(`[T2_LEVER] T2_PIN_READ …` 는 `T2_LEVER` 로 계수).

| 레버 | t0 `s626729` | t1 `s373753` | 판정 | 근거(축자) |
|---|---|---|---|---|
| `T2_VIEW_COMPACT` | 1 (실 다이제스트 **5**) | 1 (실 **4**) | **발화·역효과 + 계기결함** | `active: 1 tool output(s) digested in view` · 다이제스트에 `750` 0회 |
| `T2_DIAG` | 1 | 1 | **발화·틀린 행 지목** | `raw='Platinum Rewards Card — an error has occurred throughout the process.'` |
| `T2_SEARCH_AGENT` | **6** | **5** | **발화·주제 전부 빗나감** | `[T2_DOCDECIDE] → 'Business Bronze Rewards Card'` / `→ 'Bronze Rewards Card'` |
| `T2_DOCDECIDE` | 2 | 2 | **오지목** | 두 축 모두 Bronze — Silver 0회 |
| `T2_SEARCH_REARM` | **2** | **2** | **발화·틀린 계열 매수** | `신규 대상 platinum_rewards_card (기배달 bronze_rewards_card)` · `델타 배달 11240자 turn=32` |
| `T2_SEARCH_ON_PROCEED` | 3 | 3 | 발화 (예산 3 소진) | `deny 아님 · 재료 263자 / 254자 / 11240자 배달` |
| `T2_MATERIAL_GATE` | **10** | **11** | **발화·결정점 차단** | t0 `stop=resolve_cap(정체 3회) turn=38,40,42,44,46` · t1 `turn=40,42,44,46,48,50` |
| `T2_MATERIAL_BYPASS` | 0 | 0 | 미발화(플래그 OFF) | halfB 전체 **0회** |
| `T2_PIN_READ` | **5** | **2** | 발화·무관 read 실행 | `pinned unlock_discoverable_agent_tool(agent_tool_name=get_all_user_accounts_by_user_id_3847)` |
| `T2_DEMANDED_STEP` | 7 | 5 | 발화·표적 오형식화 하류 | `head=GB1_VERIFY_BEFORE_ACCOUNT_ACCESS → verify_identity` · `head=reads:get_all_user_accounts_by_user_id → …` |
| `T2_CLAIMPROV` | 61 | 51 | **발화·정상(날조 0)** | 전 window `unbacked=0` · `kind-index rescued … 원장에 있다` · `regen tool_calls=[]` |
| `T2_REQUIRE_DOC_DELIVER` | 5 | 5 | **발화·늦고 대상 무관** | `deliver tool=transfer_to_human_agents docs=6 chars=16498 turn=52` · 6편에 `silver_rewards_card_011` 없음 |
| `T2_READ_DEDUP` | 1 | 1 | 발화 | `stub tool=get_referrals_by_user` (t0 msg[37]·t1 msg[45]) |
| `T2_SG_DOCS` | **0** | **0** | **미발화** | ON 목록에 있으나(`run_…sh` `ON=`) halfB 전체 3회·이 sim 0회 |
| `T2_FOLLOWUP` | **0** | **0** | 미발화 | halfB 전체 25회 중 이 sim 0회 |
| `FAB_STRIP` | **0** | **0** | 미발화 | halfB 전체 **0회** |
| `T2_ARG_PRODUCERS` | **0** | **0** | 미발화 | halfB 전체 **0회** |
| READ-FIRST | **0** | **0** | 미발화 | 마커 자체가 halfB **0회** |
| (참고) `T2_ARBITRATE` / `T2_RESOLVE` | 15 / 17 | 11 / 10 | **표적 오형식화** | `formalized_target=submit_referral` **23/23** · `submit_transaction` 은 매번 `pending_user` 후보에 있었다 |

⚠**선행 표 정정**: `tasks__20260822/TASK_016.md` §4 의 수치 중 `T2_SEARCH_AGENT 11/11`(→ **6/5**) · `T2_PIN_READ 7/5`(→ **5/2**) · `T2_SEARCH_REARM 4/4`(→ **2/2**) · `T2_MATERIAL_GATE 12/11`(→ **10/11**) 가 어긋난다. `FAILURE_MASTER__20260822.md` §264-265 의 REFUTED 판정(*"레버 발화표 수치가 로그와 일치 → REFUTED"*)과 일치한다. **이 표(v2)를 쓰라.**

---

## 5. 선행 판정과의 대조

### 5.1 `tasks__20260822/TASK_016.md`(v1) 대비

| 항목 | v1 | **v2 재조사** | 판정 |
|---|---|---|---|
| 채점축 | DB | DB | **동일** |
| 변이 | *"t1 DUP 2"* | **DUP 0 · BLOCKED 2** | **정정**(v1 오류 · `MASTER_DUP_CORRECTION` 과 일치) |
| ⓑ 도달 | msg[3] 도착 후 뷰에서 삭제 | **동일 + 정량 재현**: `$750` 가시 생성 t0 **1/35** · t1 **2/31**, 반사실 26/30 | **동일·증거 강화([S])** |
| 처방3 "재무장이 마지막 검색 예산을 훔쳤다" | CONFIRMED 로 서술 | **REFUTED 재확인** — turn 40·42·44 는 `_resolve_cap_ok` 가 **먼저** 닫아 예산 가드(`:8859`)에 **도달 불가**(구조: `:8859` 는 `:8075` 의 `_resolve_cap_ok` 블록 **안**) | **v1 폐기 · FAILURE_MASTER §264 지지** |
| 레버 발화 수치 | 11/11 등 | 6/5 등 | **v1 폐기** |
| `T2_DIAG` Platinum | 재발 포착 | **3런 × 2 sim = 6/6 동일 문자열**(t7328·t7336·t7346) — 완전 결정론 | **동일·재현성 확정** |
| `submit_referral` 오형식화 | 재발 | **23/23** | 동일 |

### 5.2 `FAILURE_MASTER__20260822.md` / `ATTRIBUTION_CORRECTION` / `STATE_OF_PLAY` 대비

- FM §102 축 **A**(근거창 파괴 — `T2_VIEW_MSG_CAP=8000`) : **CONFIRMED 유지·정량 강화**(1/35·2/31 vs 26/30).
- FM §101 축 **B**(우리 층 오지목 — `T2_DIAG` Platinum) : **CONFIRMED 유지 + 신규 검산** — ERROR **정의문**이 압축 뷰 전사에 0회임을 재현으로 확인 ⇒ 지목·정의의 유일 경로가 우리 주입.
- FM §103 축 **F**(검색 예산 3) : **이 태스크에서는 REFUTED**. 실제 차단자는 `_resolve_cap_ok`(FM §173·§312 `B-1` 과 동일 결론).
- FM §105 축 **H**(계기 — `_t2_vc_logged` sim당 1회) : **재확인**(실 5/4).
- `ATTRIBUTION_CORRECTION` §64 *"016 미상 · 축 A/B/F 동시 등재"* : **A·B 유지 · F 는 내린다**. 대신 **`T2_MATERIAL_GATE resolve_cap`(축 F의 다른 항목)** 으로 교체한다.
- `STATE_OF_PLAY` §132 *"스스로 Silver 확정 후 헤지 · KB 재검색 3회에 'Silver' 0회"* : **축자 확인**(질의 = `referral bonus` / `check referral bonus status` / `credit card referral bonus criteria` / `transfer protocol and reason-code documents`) — 그러나 *"스스로 Silver 확정"* 은 **trial 0 msg[40] 한 턴뿐**이고 trial 1 은 **0회**다. 태스크 요약으로 쓸 때 이 비대칭을 함께 적어야 한다.

### 5.3 3런 대조 (동일 seed 쌍)

| 런 | trial | reward | MISSING | `silver_011` 전사 등장 | 모델 KB 질의 | `T2_DIAG` |
|---|---|---|---|---|---|---|
| **t7346** | 0 / 1 | 0.0 / 0.0 | `submit_transaction` ×2 | 1회 / 2회 | 4개 / 3개 | Platinum / Platinum |
| t7336 | 0 / 1 | 0.0 / 0.0 | 동일 | **0회 / 0회** | **0개** / `transfer to human agent`×2 | Platinum / Platinum |
| t7328 | 0 / 1 | 0.0 / 0.0 | 동일 | 1회 / 1회 | 1개 / 3개 | Platinum / Platinum |

**3런 6 sim 전부 동일 변이.** 문서가 전사에 **도달한 4 sim** 에서도 전부 실패했다 — 즉 *"도달시키면 산다"* 는 이 태스크에서 **아직 증명되지 않았다**(§6 ⚠).

---

## 6. 원인 확정 (4주체 귀속)

### CONFIRMED — our_layer (코드 경로 지목)

**OL-1. 뷰-압축이 요건 문장을 결정점 전에 삭제했다.**
경로: `t2_gate_patch.py:6193 _compact_view` · `:6224` per-메시지 캡 분기 · `:6229` head300+tail150 · 호출부 `:6765 T2_VIEW_MSG_CAP` (전부 프리즈본 `ee18d797`) + 선언 `go_stack.sh:115,152`.
증거: 오프라인 재현([S]) — msg[3]=11,214자 > 8,000 ⇒ 도착 다음 생성부터 다이제스트, 다이제스트에 `750` 0회. `$750` 가시 생성 **t0 1/35 · t1 2/31**, 반사실 CAP=0 이면 **26/30**(치명 턴 [40],[42],[44] 포함).
⚠**반증 시도(성공)**: t7346 t1 gen@msg48 은 `$750` 이 **보였는데도** 모델이 안 썼다. ⇒ 뷰 복원은 **필요조건이나 충분조건이 아니다**. reward 반사실은 **UNPROVEN**(격리 A/B 필요).
⚠부수 반증: 다이제스트 문구가 *"re-call the same tool if you need the details again"* 라는 탈출구를 남기지만, 다이제스트가 보여 주는 것은 Beige Account 와 `[axis] matches: 36 documents…` 뿐이라 **무엇이 잘렸는지 표지가 0** 이다.

**OL-2. 격리 진단 서브가 미결정 질문에 단일 답을 내고, 그 답이 "별도 검사" 권위 문구로 메인에 실렸다.**
경로: `t2_ledger.py:1055 diagnose_choice`(프리즈본) · 메모이즈 `:1077,1098` · 문맥 생성 `:1045 onto_context` · 방출 `t2_gate_patch.py:4030` · 선언 키 `a2/banking_knowledge.settings.json:471 diagnose_prompt` / `:472 diagnosed_text`.
증거: **3런 × 2 sim = 6/6** 동일 문자열 `'Platinum Rewards Card — an error has occurred throughout the process.'`. trial1 [24] 가 그것을 복창하고 [58] 의 transfer summary 까지 Platinum 으로 끝난다. ERROR **정의문**은 그 시점 압축 뷰 전사에 **0회**(재현 확인) ⇒ 정의·지목의 유일 경로가 우리 주입.
구조적 이유: `diagnose_prompt` 축자 *"**One of these** records did not pay out."* 인데 15행 중 미지급 후보가 **7행**(IN_PROGRESS 1 · REJECTED 4 · ERROR 1 · …)이라 답이 **유일하지 않다**. 유일하게 만드는 술어(*"손님이 묻는 것은 최신 건"*)는 손님 발화에만 있는데, 이 서브는 설계상 대화를 받지 못한다(`t2_ledger.py:1071` 축자 *"문맥에는 **대화가 한 글자도 안 들어간다**"*). 손님 요구를 서브에 싣는 노브 `T2_SUB_REQUIREMENT` 는 **이 런에서 `=0`**(`run_t7346_overnight_stage1_20260822.sh` PIN)이고, 켜도 **검색 서브 전용**이라 진단 서브에 경로가 없다. 게다가 `agent._t2_diag` 메모이즈로 [31] 의 손님 정정 이후에도 **갱신되지 않는다**.
⚠**반증 시도**: t7336 에서도 같은 문자열이 나왔고 trial0 은 [40]에서 회복했다 ⇒ 결정적이지 않다. **발화·배달·복창·비갱신은 CONFIRMED**, *"이것을 빼면 통과"* 는 **UNPROVEN**.

**OL-3. 재무장 술어가 손님의 *부정문*과 모델의 오진을 수요로 읽어 틀린 계열을 배달했다.**
경로: `t2_gate_patch.py:3083 _rearm_subjects` · 수요 술어 `:3119`(`role in ("user","assistant")`) · 진입 `:3246` · 델타 배달 `:3383` + `go_stack.sh:95 T2_SEARCH_REARM=1`.
증거: 양 trial `신규 대상 platinum_rewards_card` · `델타 배달 11240자 turn=32`. turn 32 시점 산문의 `Platinum Rewards Card` 출처는 [30](모델 오진)·[31](손님 축자 *"I **haven't mentioned** a Platinum Rewards Card specifically"* = **부정문**)·[32]. `Silver Rewards Card` 는 그 시점 산문에 0회(도구 출력에만 존재하고 술어가 도구 출력을 배제).
[[70]] 무엇을 팔았나: 11,240자 배달 = 지연 · 오초점 강화. **이 태스크에서 부호는 음(−)**. 단 FM §167 대로 050#1·073#0 에서는 양(+)이므로 **끄지 말 것**([[60]]).

**OL-4. 결정점 턴에 재료 채널이 `resolve_cap` 으로 닫혀 있었다(축 F 의 실제 차단자).**
경로: `t2_gate_patch.py:4279 _resolve_cap_ok` · 판정/로그 `:8023,8027` · **바깥 가드 `:8075`** — 검색 재료 배달 자리(`:8859` 의 `_t2_searchagent_fired < 3`)는 `:8075` 블록 **안**이라 `_resolve_cap_ok` 가 False 면 **도달조차 못 한다**.
증거: t0 `stop=resolve_cap(정체 3회) turn=38,40,42,44,46` — **Silver 확정 [40] · 손님 질문 직후 [42] · 치명 답변 [44] 전부 포함**. t1 `turn=40,42,44,46,48,50` — **$750 이 유일하게 보였던 [48] 포함**.
⇒ v1 의 처방 3(예산 칸 분리)은 **결정점을 못 연다**. 실제 상류 노드는 `_resolve_cap_ok` 다(FM `B-1`).

**OL-5. 표적 오형식화의 대가.** `formalized_target=submit_referral` **23/23**(gold user 도구는 `submit_transaction`, 매번 후보에 실재). 그 요건 `reads:get_all_user_accounts_by_user_id` 를 `T2_PIN_READ` 가 실행시켜 t0 [22]~[25] 2턴을 태웠고, 결과는 축자 *"Bank Accounts: No bank accounts found. Credit Card Accounts: No credit card accounts found."* 로 이 태스크의 어떤 결정에도 쓰이지 않았다(순수 낭비). t1 에서는 이미 충족된 `GB1_VERIFY_BEFORE_ACCOUNT_ACCESS` 반복이 [24]~[44] 10턴 verify 루프(BLOCKED 2 포함)를 만들었다.
⚠표적 **선택 자체**는 LLM 서브(`t2_resolve.formalize_intent_tool`)이므로 귀속은 `model`([[66]] 무수리 유지). **우리 층 몫은 그 선택의 하류 배선**(요건 사슬 → PIN_READ 강제 read)이다.

### model

- **ⓐ⋈ⓑ join 실패**: t0 [40] 에서 Silver/IN_PROGRESS 를 스스로 확정하고 [44] 에서 *"The exact amount and time window depend on the specific credit card type"* 로 헤지. 단 그 시점 ⓑ는 우리가 이미 지웠다 ⇒ **부하 귀속 전에 our_layer 가 먼저 물린다**([[55]] 0단계).
- **재검색 표적화 실패**: KB 재질의 3회에 `Silver` **0회**([[63]] 동형 — 후보를 좁히는 닫힌 술어를 스스로 못 건다). ⚠*"Silver 를 질의에 넣었으면 나왔을 것"* 은 로컬에 tau2 리트리버가 없어 **UNPROVEN**.
- **가시 재료 미사용**: t1 [48] 은 `$750` 이 보이는 유일한 회복 턴인데 *"Retry the Referral Process / Escalate Internally"* 로 갔다. **우리 층 OL-1 의 반증 자료이자 순수 model 결손.**
- **날짜 비교 실패**: t1 [46] 이 `10/05/2025` 를 *"the most recent referral"* 이라 진술(목록에 `11/13/2025` 실재).

### env — 정상

`get_user_information_by_id("friend_user_5839")` → *"No records found in 'users'."* · `get_credit_card_transactions_by_user("86e92f639e")` → *"No records found"* · `log_verification` 재시도 → *"Failed to log verification: Record may already exist."* — 전부 스펙 내. **면책 사유 아님**(친구의 지출 진척은 원래 조회 불가이고, gold 가 요구하는 것은 **요건 금액 발화**뿐이다).

### user_sim — 정상 (면책 아님·[[21]])

스펙을 정확히 지켰다: [31] 최신-건 정정 + Platinum 부정 · [33]/[45]/[51] `friend_user_5839` 제공 · [41] 요구 금액을 **두 번 명시 질문** · [53] *"If the agent … offers to transfer you to a human, accept that"* 대로 수용. 발동 조건 ③이 끝내 충족되지 않아 `submit_transaction` 을 안 쓴 것이므로 **에이전트-측 흡수 대상**이다.

### 한 줄 결론

> **016 의 실패는 "정책을 못 찾아서"가 아니다.** 모델은 msg[2]에 스스로 KB 를 쳤고 msg[3]에 `$750 within 60 days` 가 도착했다. 우리 층은 ⑴그 값을 다음 턴에 뷰에서 지웠고(`T2_VIEW_MSG_CAP=8000` · 가시 1/35·2/31 ↔ 반사실 26/30), ⑵ 미결정 질문에 단일 답을 강제하는 격리 진단 서브가 **6/6 결정론적으로** 대상을 Platinum 으로 지목했으며, ⑶재무장 술어가 손님의 **부정문**을 수요로 읽어 그 오초점에 11,240자를 배달했고, ⑷결정점 턴([40][42][44]/[48])에는 `resolve_cap` 이 재료 채널을 닫아 두었다. **그 결과 "대상=Silver"와 "요건=$750"이 한 생성 안에 동시에 존재한 턴이 4 sim 전부 0회다.** 다만 t1 [48](값이 보인 유일한 턴)에서 모델이 그것을 안 쓴 사실이 남아 있어, **reward 반사실은 미증명(UNPROVEN)** 이다.

---

## 7. 처방 후보 (수리 실행 없음 · [[62]] 4문 + [[70]] 부호표 의무)

우선순위는 **격리 프로브로 결손을 먼저 재고**(x46N · N_neg) 그 다음에만 등재.

1. **[최우선·전달 축소만] 다이제스트에 표지를 남긴다.** 현재 head/tail 순수 절단이라 *"거기 무엇이 있었나"* 가 0이다. 닫힌 술어 후보 = 잘린 tool 출력에서 **선언 색인에 있는 문서 id 목록만** 축자로 남기기(본문 0자·엔진 내용 판정 0·[[59]]·선택 0·[[62]]③). 격리 질문: *"문서 id 목록만 보이면 모델이 그 doc 을 다시 부르는가."* ⚠부작용 계측 = 다이제스트 길이 증가.
2. **[상류·폭발 반경 큼] `T2_MATERIAL_GATE` 의 `resolve_cap` 완화 조건.** OL-4 로 이것이 결정점의 실제 차단자다(FM `B-1` 과 동일). ⛔**공유 상류 노드**라 016·072·085 동시 영향([[66]] ②) — 부호표 없이 켜지 말 것. 엔드포인트 = *결정점 턴에 채널이 열리는가*, 부작용 = over-action.
3. **`diagnose_prompt` 의 미결정성 제거.** 축자 *"One of these records did not pay out."* 이 7후보 중 1을 강제한다. 후보 = ⑴*"미지급 후보가 여럿이면 전부 나열"* 로 **단일 지목을 없애기**(엔진이 고르지 않게 하는 방향 = [[62]]③ 준수) · ⑵`diagnosed_text` 를 후보 ≥2 일 때 **나열형으로 분기**([[70]] 절충 — 끄기 아님). 조건은 *"그룹 축 미지급 후보 수 ≥ 2"* 라는 **도메인-일반 닫힌 술어**여야 한다(태스크 id 금지·[[05]]).
4. **`_t2_diag` 메모이즈 해제 술어.** 손님 정정 이후에도 답이 고정된다. 닫힌 술어 후보 = *"진단 대상 원장의 행 집합이 바뀌었을 때만 재계산"* 은 016 을 못 구하므로, **후보 나열형(3⑴)** 과 묶어 재는 편이 낫다. 단독 등재 금지.
5. **`T2_SEARCH_REARM` 수요 술어의 화자 축 분리 + 부정문 배제**(FM `B-10`). 실물: 손님의 **부정문**이 수요로 읽혔다. ⛔050#1·073#0 에서 양(+)이므로 **끄지 말 것**([[60]]) — 조건 조정만.
6. **계기 수리(성적 무관·즉시 가능)**: `_t2_vc_logged` 를 **다이제스트 집합이 바뀔 때마다** 인쇄로. (워킹트리 `4ea50a09` 에는 이미 `_t2_vc_logged_sig` 로 반영돼 있다 — **런 이후 변경**이라 t7346 로그에는 없다.)
7. **`T2_REQUIRE_DOC_DELIVER` 배치 전진** — 현재 turn 52(transfer 국면)에서만 발화하고 실린 6편에 `silver_rewards_card_011` 이 없다. task 선언의 `required_documents` 와 이 레버가 보는 집합이 **다른 축**이라는 사실을 먼저 문서화할 것.
8. ⛔**폐기**: v1 처방 3(검색 예산 칸 분리) — OL-4 로 **결정점을 못 연다**(REFUTED).

---

*근거 파일: `sim_results/bank_t7346_halfB_20260822.results.json.gz`(task_016 2 sims 전 메시지 정독) · 동 `.log.gz` `[sim=task_016#…]` 788라인 전수 · 변이 = `t2_forensic.mutation_diff` · 뷰 재현 = 프리즈본 `git show ee18d797:scripts/distill/tau2/t2_gate_patch.py` 의 `_compact_view` 원본 실행(워킹트리와 바이트 동일 확인) · 대조 `bank_t7336_halfB_20260821b` · `bank_t7328_halfB_20260819r2` · 선행 `tasks__20260822/TASK_016.md` · `FAILURE_MASTER__20260822.md` · `MASTER_DUP_CORRECTION_2026_08_23.md` · `ATTRIBUTION_CORRECTION_2026_08_23.md` · `STATE_OF_PLAY_2026_08_23.md`. 로컬 전용 · SSH 0 · git 커밋/변경 0 · 코드 수정 0.*
