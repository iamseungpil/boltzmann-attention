# t7346 · task_079 per-step 포렌식 (halfB · 양 trial 실패)

- 원본: `reports/facet_rft_2026/sim_results/bank_t7346_halfB_20260822.results.json.gz`
  · 로그 `reports/facet_rft_2026/sim_results/bank_t7346_halfB_20260822.log.gz`
- 대조(기준선·sha 상이): `bank_t7328_halfB_20260819r2.results.json.gz`
- 선행 보고서: `reports/facet_rft_2026/t7336_tasks/T7336_TASK_079.md` (t7336 · 같은 러너 계보)
- 런 meta: `sha ee18d797` · `ON="T2_ARG_DOC_SUB=1 T2_VALUE_FORMULA=full T2_SG_DOCS=1"` ·
  PIN 에 `T2_PROCEED_DOCBODY=0 T2_DOCS_AT_WRITE=0` (문서-본문 배달 레버 OFF)
- sim 매핑: **trial 0 = seed 626729**(`user_stop` · 141 msgs · 1548s) /
  **trial 1 = seed 373753**(`user_stop` · 68 msgs · 706s)
- 태스크: 지갑 도난 → 개인 checking 3계좌(Evergreen=PREMIUM / Light Blue=ENTRY / Green=MID)
  데빗카드 **freeze → unfreeze → close(stolen) → 교체 주문**.
  gold 최종 상태 = Evergreen ORDERED(RUSH·$35·PREMIUM·design_fee 0) · Light Blue **CLOSED 이되 주문 없음**(48시간 대기) · Green ORDERED(STANDARD·CLASSIC).

---

## 1. 채점 축 (먼저 확인 · C583ⓖ)

```
trial 0  reward_info.reward_basis      = ['DB']
         reward_info.reward_breakdown  = {"DB": 0.0}
         reward_info.db_check          = {"db_match": false, "db_reward": 0.0}
         env_assertions=[] · nl_assertions=null · communicate_checks=null
         info.action = null            ← ACTION 은 채점에 안 들어감
trial 1  (동일) reward_basis=['DB'] · reward_breakdown={"DB":0.0} · db_match=false
```

**⇒ DB-해시 축이다.** `action_checks` 25건은 양 trial 모두 존재하지만 `reward_breakdown` 에
ACTION 키가 없으므로 **진단용**이다. 아래 표는 전부 그 전제에서 읽는다.

참고(진단): action_match 집계 = trial 0 **20/25**, trial 1 **5/25**.

---

## 2. 변이표 (정본 `t2_forensic.mutation_diff` · 손 비교기 0)

```python
sys.path.insert(0,'.'); import t2_forensic as F
mut = F.mutating_tools(); m = F.mutation_diff(sim, mut)
```

### trial 0 (seed 626729) — MATCHED 11 · **MISSING 2 · WRONGARG 3** · EXTRA 0 · DUP 1 · BLOCKED 27

| 종류 | 도구 | 인자 (요약) |
|---|---|---|
| MATCHED | `log_verification` | 6필드 + `time_verified=2025-11-14 03:40:00 EST` |
| MATCHED ×3 | `freeze_debit_card_3892` | `_ev` / `_lb` / `_green` |
| MATCHED ×3 | `unfreeze_debit_card_3893` | `_ev` / `_lb` / `_green` |
| MATCHED ×3 | `close_debit_card_4721` | `_ev` / `_lb` / `_green` · `reason=stolen` |
| **MISSING** | `order_debit_card_5739` | `chk_…_1` · **RUSH · delivery_fee 35 · PREMIUM · design_fee 0** |
| **MISSING** | `order_debit_card_5739` | `chk_…_3` · **STANDARD** · 0 · CLASSIC · 0 |
| **WRONGARG** | `order_debit_card_5739` | `chk_…_1` · `standard` · `0` · `CLASSIC` · `0` |
| **WRONGARG** | `order_debit_card_5739` | `chk_…_2` · `standard` · `0` · `CLASSIC` · `0` |
| **WRONGARG** | `order_debit_card_5739` | `chk_…_3` · `standard` · `0` · `CLASSIC` · `0` |
| DUP | `log_verification` | 동일 인자 재호출 |
| BLOCKED 27 | (전부 `deny=env`) | freeze/unfreeze/close 재시도 + order 재시도 |

**WRONGARG 필드별 대조 (보낸 인자 ↔ gold 인자)**

| aid | 필드 | 보냄 | gold | 일치 |
|---|---|---|---|---|
| 079_17 (`chk_…_1`) | `delivery_option` | `"standard"` | `"RUSH"` | ✗ |
| | `delivery_fee` | `"0"` | `35` | ✗ |
| | `card_design` | `"CLASSIC"` | `"PREMIUM"` | ✗ |
| | `design_fee` | `"0"` | `0` | ✓ |
| | `account_id`·`user_id`·`shipping_address` | 동일 | 동일 | ✓ |
| 079_18 (`chk_…_3`) | `delivery_option` | `"standard"` | `"STANDARD"` | **대소문자만 상이** |
| | 나머지 5필드 | 동일 | 동일 | ✓ |
| (gold 없음) `chk_…_2` | — | 주문 실행됨 | **gold 는 chk_2 에 order 없음**(48h 대기) | ✗ |

⚠**계기 주의 2건**(판정에 반영):
1. `chk_…_3` 의 유일한 차이는 `standard` ↔ `STANDARD` 인데 env 는 대문자로 정규화해 기록한다
   (msg `[076]` 축자 `Delivery Option: STANDARD`). `t2_forensic.norm_args` 는 대소문자를 접지
   않으므로 이 한 건은 **표기 아티팩트**이고 DB 상 Green 주문은 gold 와 같을 개연이 높다.
   ⇒ **DB 해시를 실제로 깬 것은 ⑴Evergreen 주문의 3필드 오값 ⑵Light Blue 의 불필요 주문** 둘이다.
2. DUP 로 잡힌 두 번째 `log_verification` 은 **실패했다** — msg `[107]` 축자
   `Failed to log verification: Record may already exist.` `t2_forensic.deny_kind`
   (`t2_forensic.py:542`)는 `Error:` 로 시작하는 본문만 거절로 보므로(`OURS_DENY` 는
   `t2_forensic.py:509`) 이 실패가 `ok=True` 로 접혀 DUP 1건이 됐다. **DB 무변경 · 위양성**이다.

### trial 1 (seed 373753) — MATCHED 1 · **MISSING 11** · WRONGARG 0 · EXTRA 0 · DUP 0 · BLOCKED 0

| 종류 | 도구 | 인자 |
|---|---|---|
| MATCHED | `log_verification` | (msg `[034]` 성공) |
| MISSING ×3 | `freeze_debit_card_3892` | `_ev` / `_lb` / `_green` |
| MISSING ×3 | `unfreeze_debit_card_3893` | 같은 3장 |
| MISSING ×3 | `close_debit_card_4721` | 같은 3장 · `reason=stolen` |
| MISSING | `order_debit_card_5739` | `chk_…_1` RUSH·35·PREMIUM |
| MISSING | `order_debit_card_5739` | `chk_…_3` STANDARD·0·CLASSIC |

**BLOCKED 0** — 즉 이 trial 은 *막힌 것*이 아니라 **변이를 한 번도 시도해 성공/거절까지 가지 못했다**.
이 대화가 DB 에 남긴 상태변화는 `log_verification` 1건뿐이다.

---

## 3. step-by-step 결정 지점 추적 (축자)

### 3-A. 공통 초동 (turn 0 ~ 27) — 두 trial 바이트급으로 유사

`[002]` `unlock freeze_debit_card_3892` → `[004]` `KB_search_bm25{"query":"freeze debit card"}` →
`[012]` `verify_identity{"date_of_birth":"01/15/1985","email":"johndoe@example.com"}`
(**모델이 상수 페르소나를 넣음** → `[013]` `NOT_VERIFIED`) → `[015]~[021]` 실제 값으로 VERIFIED →
`[022]` `get_current_time` → `[023]` `2025-11-14 03:40:00 EST`.
우리 층 로그도 두 trial 이 line 000~118 까지 **같은 순서·같은 마크**다.

---

### 3-B. trial 0 — 분기(살아난 쪽): turn 33 에 **우리 층이 gold read 이름을 댔다**

로그 축자 (`log 221–222`):

```
[T2_OUR_NAMES] 등재 name=get_all_user_accounts_by_user_id_3847 (출처=T2_DISCOVERY_STEP2 지목)
[T2_DISCOVERY_STEP2] deny name=get_all_user_accounts_by_user_id_3847 (레지스트리 폴백·미unlock·formalize 정합)
```

궤적 대응:

```
[030] assistant  shell {"command":"grep -r 'debit card' ."}
[031] tool       ./doc_bank_accounts_bank_accounts__general__029.md:When a customer requests a
                 replacement debit card … the replacement policies vary based on the customer's
                 checking account tier.            ← 문서 **포인터만**, 본문 아님
[032] assistant  unlock get_debit_cards_by_account_id_7823   → [033] Tool unlocked   ← ★통과
[034] assistant  unlock get_all_user_accounts_by_user_id_3847 → [035] Tool unlocked
[036] call get_all_user_accounts_by_user_id_3847{"user_id":"cr89a2b3c4"} → [037] chk_…_1/_2/_3
[038] call get_debit_cards_by_account_id_7823 × 3 → [039][040][041] 카드 4장
[042] freeze × 3 → [043][044][045] 전부 성공
```

**주목**: trial 0 은 `get_debit_cards_by_account_id_7823` 의 unlock 을 **turn 32 에 미리** 통과시켰다.
같은 unlock 이 trial 1 에서는 turn 47~63 에 4회 거부된다(§3-E) — 이것이 두 trial 의 실질 분기다.

`[040]` 축자 — **Light Blue 교체 이력이 이 시점에 문맥에 실재한다**:

```json
[{"card_id":"dbc_cr89a2b3c4_lb", "status":"ACTIVE",  "issue_reason":"lost",
  "card_number_last_4":"5291","date_issued":"08/10/2025"},
 {"card_id":"dbc_cr89a2b3c4_lb_old1","status":"CLOSED","issue_reason":"lost",
  "card_number_last_4":"2847","date_issued":"04/05/2025"}]
```

`[047]` user: *"Yes — these cards are definitely gone. … I need to **cancel all three cards** and **get replacement cards**."*
`[050]` close ×3 → `[051]~[053]` `Error: … Current status: FROZEN. Only ACTIVE or PENDING cards can be closed.`
`[057]` unfreeze ×3 → `[061]` close ×3 성공. **여기까지 gold 변이 10/11 정확.**

---

### 3-C. trial 0 — **치명 결정점 = turn 65 → 69 → 73** (비가역 주문 3건, 묻지도 읽지도 않음)

```
[065] assistant  "I understand that you want to order new debit cards for your accounts. Let's
                 proceed with placing the orders … I will now order the new debit cards for you."
      ->CALL order_debit_card_5739 {"account_id":"chk_cr89a2b3c4_1"}
      ->CALL order_debit_card_5739 {"account_id":"chk_cr89a2b3c4_2"}
      ->CALL order_debit_card_5739 {"account_id":"chk_cr89a2b3c4_3"}
[066] tool  Error: Invalid arguments: KnowledgeTools.order_debit_card_5739() missing 6 required
            positional arguments: 'user_id','delivery_option','delivery_fee','card_design',
            'design_fee', and 'shipping_address'
[069] assistant  "I apologize for the oversight. … **We will use default options for the delivery
                 and design to simplify the process.**
                 Here are the default options:
                 - Delivery option: Standard  - Delivery fee: $0
                 - Card design: Basic         - Design fee: $0"
      ->CALL order_debit_card_5739 × 3 (card_design="basic")
[070] tool  Error: Invalid card_design. Must be one of: ['CLASSIC','PREMIUM','CUSTOM']
[073] assistant  "I apologize for the mistake. Let's use the 'CLASSIC' card design …"
      ->CALL order_debit_card_5739 {chk_…_1, standard, 0, CLASSIC, 0, addr}
      ->CALL order_debit_card_5739 {chk_…_2, standard, 0, CLASSIC, 0, addr}
      ->CALL order_debit_card_5739 {chk_…_3, standard, 0, CLASSIC, 0, addr}
[074][075][076] Debit Card Order Confirmed  ← ★비가역 커밋 3건. reward 는 여기서 잃었다.
```

**그 시점 문맥에 무엇이 실재했나 (닫힌 검사)**

| 필요한 값 | turn 73 시점 문맥에 있었나 | 근거 |
|---|---|---|
| Light Blue 교체 이력(`issue_reason:"lost"` ×2) | **있었다** | msg `[040]` 축자(§3-B) |
| 손님의 배송/디자인 선호 | **없었다** | 손님이 그 요구를 처음 말한 것은 `[078]` — **주문 5턴 뒤** |
| 등급별 교체 한도·48h·RUSH $35·PREMIUM 무료 표 | **없었다** | `doc_…(general)_029` **본문**은 `[080]` 에 처음 도착 |
| "배송·디자인을 물어라" 절차 | **없었다(포인터만)** | `[031]` grep 한 줄만 · `_023` 본문 미도착 |

⇒ 손님 선호는 **아직 존재하지 않는 값**이었고(묻지 않았으므로), 등급 표는 **아직 안 읽은 값**이었다.
모델은 두 미확정 값을 **자기 기본값으로 메꿔** 비가역 write 를 냈다(`[069]` 축자 *"use default options … to simplify the process"*).

**한 턴 뒤 손님이 정확히 그 값을 요구한다** (`[078]` 축자):

```
- For the **Evergreen** replacement, I need it **as fast as possible** — what's the quickest shipping option?
- Also, I'd like the **metallic design**, but **only if it's free**. If it costs extra, keep the free/classic option.
- For the **Green** replacement, **standard shipping is fine**.
```

`[079]` 에서 모델이 그제서야 KB 를 읽고(`[080]` = `doc_credit_cards_credit_card_replacements_002` +
**`doc_bank_accounts_bank_accounts_(general)_029`** 전문 + `_023` 전문 도착 — RUSH `delivery_fee:$35`·
PREMIUM tier `PREMIUM design design_fee $0`·ENTRY `48-hour waiting period`·`Maximum 2 replacement cards`가
**전부 여기 있다**) 재주문을 시도하나:

```
[082] order chk_…_1 {expedited, PREMIUM} · order chk_…_3 {standard, CLASSIC}
[083] tool  Error: There is already a pending debit card order for account 'chk_cr89a2b3c4_1'.
[084] tool  Error: There is already a pending debit card order for account 'chk_cr89a2b3c4_3'.
```

`[098]`·`[103]` 에서 4회 더 시도하나 전부 같은 env 잠금. **gold 079_17 은 turn 73 이후 도달 불가능**이다.

**⇒ 결정 지점은 msg `[065]`(첫 order 발행), 손실 확정은 msg `[073]`(성공 커밋).**
필요했던 행위 = *주문 전에 ⑴등급 정책 본문 read ⑵손님에게 배송·디자인 1회 질의 ⑶Light Blue 배제*.

### 3-D. trial 0 — 손실 이후 구간(비용): `T2_GROUND` 오치환 15회

`[098]` 궤적 축자 — 실행된 인자가 **고객 이름**이다:

```
[098] ->CALL call_discoverable_agent_tool {"agent_tool_name":"CARLOS RODRIGUEZ","arguments":"{\"order_id\":\"dcord_5bf4581e9838\"}"}
[099] tool  Error: Unknown agent tool 'CARLOS RODRIGUEZ'. This tool is not available.
```

우리 층 로그 축자(`log 353`·`428`·`433`·`435`·`442`·`443`·`444`·`448`·`463`·`464`·`473`·`474`·`487` … **총 15행**):

```
[T2_GROUND] substituted arg=agent_tool_name val=update_debit_card_order_9834 -> CARLOS RODRIGUEZ
[T2_GROUND] substituted arg=agent_tool_name val=cancel_debit_card_order_7829 -> CARLOS RODRIGUEZ
[T2_GROUND] substituted arg=agent_tool_name val=cancel_pending_debit_card_order_9281 -> CARLOS RODRIGUEZ
…  (15/15 전부 `-> CARLOS RODRIGUEZ`)
[T2_RESOLVE] deny tool=unlock_discoverable_agent_tool arg=agent_tool_name reason=operator-fab
```

= 선행 보고서 §7-1 이 지목한 결함이 **그대로 살아 있다**(§5 참조). 다만 turn ≥ 65 = **손실 이후**이므로
성적의 원인이 아니라 **비용**(turn 소모·이관 앞당김)이다.

---

### 3-E. trial 1 — **치명 결정점 = turn 63 → 64. 우리 층이 gold read 를 4회 거부했다**

정상 진행: `[034]` `log_verification` 성공. 그런데 `[036]` 에서 모델이 **날조 card_id 로 freeze 3건**을
내자 `T2_FAB_STRIP` 이 잡아 드롭했다(`log 235` 축자 `[T2_FAB_STRIP] dropped 3 ungrounded write call(s) (exhaustion->abstain)`).
그 자리에 **우리가 붙인 해소-지목 노트**가 커밋 메시지에 그대로 실렸다 — `[036]` 축자:

```
[Note: items whose supporting records could not be verified were not processed. freeze_debit_card
 was NOT executed: card_id='cr89a2b3c4-Evergreen' does not appear in any record read in this
 conversation; **to fix this, first read the real value with get_all_user_accounts_by_user_id_3847,
 then get_debit_cards_by_account_id_7823**, then re-issue the call with a value copied from that
 tool's output; …]
```

(= 어제 P4 `_fab_fix_note`·`t2_gate_patch.py:1888-1908` 가 A2 `arg_source_reads.card_id` 를 읽어 만든 문장.
그 선언의 `_note_card_id` 축자: *"079 실패 경로가 정확히 이 두 read 부재였다"*.)

모델은 그 지시를 따랐다. `[060]` 에서 첫 read 를 unlock 하고 `[062]`→`[063]` 에서 계좌 3건을 받는다:

```
[060] ->CALL unlock_discoverable_agent_tool {"agent_tool_name":"get_all_user_accounts_by_user_id_3847"}
[061] tool  Tool unlocked: get_all_user_accounts_by_user_id_3847
[062] ->CALL call_discoverable_agent_tool {…"user_id":"cr89a2b3c4"}
[063] tool  chk_cr89a2b3c4_1 (Evergreen) / chk_cr89a2b3c4_2 (Light Blue) / chk_cr89a2b3c4_3 (Green)
```

**남은 것은 두 번째 read 하나뿐이다.** 그런데 그 자리에서 우리 층 로그(`log 311–320` 축자):

```
[T2_RESOLVE] operator-scope: 지목 대신 범위 표면화 (get_debit_cards_by_account_id_7823, freeze_debit_card_3892)
[T2_RESOLVE] deny tool=unlock_discoverable_agent_tool arg=agent_tool_name reason=operator-scope
[T2_STACK]   window folded fb tag=resolve_write (same fingerprint (seen=14)) — deny stays, body kept (R9)
[T2_RESOLVE] operator-scope: 지목 대신 범위 표면화 (get_debit_cards_by_account_id_7823, freeze_debit_card_3892)
[T2_RESOLVE] deny tool=unlock_discoverable_agent_tool arg=agent_tool_name reason=operator-scope
[T2_STACK]   window folded fb tag=resolve_write (same fingerprint (seen=14)) — deny stays, body kept (R9)
[T2_RESOLVE] operator-scope: 지목 대신 범위 표면화 (get_debit_cards_by_account_id_7823, freeze_debit_card_3892)
[T2_RESOLVE] deny tool=call_discoverable_agent_tool arg=agent_tool_name reason=operator-scope
[T2_STACK]   window folded fb tag=resolve_write (same fingerprint (seen=14)) — deny stays, body kept (R9)
[T2_MATERIAL_GATE] stop=resolve_cap(정체 3회) turn=64
```

`turn=64` 는 정확히 다음 커밋 메시지다. `[064]` 축자:

> **"It seems there is a persistent issue with unlocking and using the tool to retrieve debit card
> information. Let's try a different approach to resolve this issue.
> Given the current limitations, I will transfer you to a human agent …"**

`[065]` user `###TRANSFER###` → `[066]`/`[067]` 종료.

**그리고 그 "unlocking issue" 는 env 가 만든 것이 아니다.** 궤적 전수 검사:

| 검사 | 결과 |
|---|---|
| trial 1 전체 tool 메시지 중 `Error:` 로 시작하는 것 | **0건** (KB 검색 결과 9건이 전부·전부 정상 출력) |
| trial 1 전체에서 커밋된 `unlock_discoverable_agent_tool` 호출 | 4건 — `freeze_debit_card_3892` ×3, `get_all_user_accounts_by_user_id_3847` ×1 · **전부 성공** |
| `get_debit_cards_by_account_id_7823` unlock 이 궤적에 있는가 | **0건** (전부 우리 층에서 소거) |

즉 모델이 반응한 "unlock 에러"는 **전부 우리 deny 의 반향**이다. 같은 형태가 앞서 두 번 더 있다:

```
log 238 / 245  [T2_RESOLVE] deny tool=unlock_discoverable_agent_tool … reason=operator-fab
   → [040] assistant "It seems there was an error because **the tool name was not found in the
                      previous searches**."      ← OPERATOR_PROVENANCE_FB 문면의 축자 반향
log 278 / 279  operator-scope (get_debit_cards_by_account_id_7823, freeze_debit_card_3892)
   → [050] assistant "It seems there was **an error in unlocking the tool**."
```

**기전 (코드)**

`t2_resolve.py:187-226` `resolve_operator` 의 FIND 분기:
후보가 2개 이상이면 격리 서브 `formalize_intent_tool`(`t2_resolve.py:761-788`)에게
*"which ONE of these tools must the agent CALL to fulfill the request?"* 를 묻고, 답(`want`)이
모델의 선택(`chosen`)과 다르면 deny 한다. 손님 발화는 *"freeze all three cards"* 이므로 서브의 답은
**당연히 `freeze_debit_card_3892`** 이고, 모델이 지금 부르려는 것은 그 **선행 read** 다.
즉 이 레버의 전제 *"한 요청 = 한 도구"* 가 다단 절차에서 깨진다(코드 자신이 `C10` 주석에서
051 로 같은 병을 기록해 두었다).

우회로는 `t2_resolve.py:185` `if str(chosen) in (declared_required or set()): return ok` 하나뿐인데,
그 집합은 `t2_resolve.py:1118-1131` 에서 A2 `follow_up_chains` + `procedures[].nodes[].tool` 로만
만들어진다. 실물 계산:

```
declared_required n=18 =
 ['apply_credit_card_account_flag_6147','approve_credit_limit_increase_5847',
  'close_credit_card_account_7834','deny_credit_limit_increase_5848',
  'emergency_credit_bureau_incident_transfer_1114','get_all_user_accounts_by_user_id_3847',
  'get_closure_reason_history_8293','get_credit_limit_increase_history_4829',
  'get_payment_history_6183','get_pending_replacement_orders_5765','get_user_dispute_history_7291',
  'initial_transfer_to_human_agent_0218','initial_transfer_to_human_agent_1822',
  'log_credit_card_closure_reason_4521','pay_credit_card_from_checking_9182',
  'submit_credit_limit_increase_request_7392','transfer_to_human_agents',
  'update_transaction_rewards_3847']
'get_all_user_accounts_by_user_id_3847' in req  → True    ← 그래서 [060] 은 통과했다
'get_debit_cards_by_account_id_7823'   in req  → False    ← 그래서 그 다음이 4회 막혔다
```

**같은 A2 파일 안에서 두 선언이 모순한다**: `arg_source_reads.card_id` 는
`["get_all_user_accounts_by_user_id_3847","get_debit_cards_by_account_id_7823"]` 를 **필수 원천 read**로
선언하고 `FAB_STRIP` 이 그것을 모델에게 축자로 지시했는데, `declared_required`(procedures 유래)에는
두 번째가 **없어서** `operator-scope` 가 그 호출을 거부한다. 데빗카드 절차는 `procedures` 에
등재돼 있지 않다(등재된 18개는 전부 신용카드·이관 계열).

---

## 4. 레버 발화표 (이 sim 줄만 · 로그 축자)

| 레버 | trial 0 | trial 1 | 판정 |
|---|---|---|---|
| `T2_SG_DOCS` | **0** | **0** | **미발화(구조적)**. 런 meta `ON` 에 `T2_SG_DOCS=1` 이고 스모크 게이트가 死배선을 배제했으므로 배선은 살아 있다 — 다만 발화 술어(`isolate.docs` 선언 도구)가 079 궤적에 없다. t7336 과 동일. |
| `T2_PIN_READ` | 0 | 0 | 미발화 |
| `T2_DEMANDED_STEP` | 0 | 0 | 미발화 |
| `T2_CLAIMPROV` | 32 | 22 | **발화·무해·무익**. 전 window `unbacked=0` → deny 0. 이 실패에 접점 없음 |
| `T2_FOLLOWUP` | 0 | 0 | 미발화 |
| `T2_SEARCH_AGENT` | 13행 | 10행 | **발화하나 이 태스크의 정책군을 구조적으로 못 나른다** — §5 아래 |
| `T2_SEARCH_REARM` | **2** (turn 34 · `checking_accounts` 신규 계열 `evergreen_account,light_blue_account` 델타 **14,851자**) | 0 | **발화·이 태스크에 무익**. 나른 것은 상품 문서(APY·ATM 수수료)이고 결정에 필요한 등급 교체표는 **다른 군**에 있다 |
| `T2_DEGENERATE_AXIS` | 6 | 6 | **발화 = 배달 차단**. `group=bank_accounts_bank_accounts` 를 6회 결정 경로에서 제외 → 그 축은 그대로 `요청 축 … 모두 처리됨 — 침묵` 으로 접힘 |
| `FAB_STRIP` | 0 | **1**(write 3콜 드롭) | **발화·옳음**. 해소-지목 노트가 커밋 메시지 `[036]` 에 **실제로 실렸다**(t7336 대비 개선) — 그런데 그 지목이 가리킨 read 를 `operator-scope` 가 막았다(§3-E) |
| `T2_ARG_PRODUCERS` | 0 | 0 | **미발화**. `[066]` 의 `missing 6 required positional arguments` 는 전형적 arg-결핍 형상인데도 안 열렸다(에러-형상 게이트 `t2_prekb_patch.py:596-641` 미적중) |
| READ-FIRST / `requires_reads`(C587) | 0 | 0 | **구조적 미발화** — 079 대상 도구에 선언 없음 |
| `T2_REQUIRE_DOC_DELIVER` | 0 | **1** | **발화 후 자기-스킵**: `skipped: est 218969+16498 chars > cap tool=transfer_to_human_agents turn=66`. 표적이 `transfer_to_human_agents` 뿐이고(A2 `require_doc_before.tools` = 이관 4종) **order_debit_card_5739 는 대상이 아니다** |
| `T2_RESOLVE` deny | 3 (`operator-scope` 1 · `operator-fab` 2) | **7** (`operator-scope` **5** · `operator-fab` 2) | **오발화(치명·trial 1)** — §3-E |
| `T2_GROUND` | **15회 치환·15/15 `-> CARLOS RODRIGUEZ`** | 0 | **오발화**. trial 0 은 전부 turn ≥ 65 = 손실 이후 → 비용 |
| `T2_WRITE_ARG_ENUM` | 0 | 0 | 미발화 — 선언이 `applies_when.prefix="open_bank_account"` 하나뿐이라 `order_debit_card_5739` 는 사정권 밖 |
| `T2_WRITE_EVIDENCE` | 0 | 0 | 미발화 |
| `T2_DISCOVERY_STEP2` | 13 | 6 | **발화·trial 0 에서 결정적 이득**: `log 221` 레지스트리 폴백이 gold read `get_all_user_accounts_by_user_id_3847` 를 지목 → `[034]` 에서 모델이 그대로 unlock. t7336 trial 0(2회 후 영구 침묵)과 대비되는 **개선** |
| `T2_FORCE_ACTION` / `T2_MATERIAL_GATE` | 13 / 25 | 6 / 12 | 발화. trial 1 의 `stop=resolve_cap(정체 3회) turn=64` 가 이관 직전 마지막 마크 |

**직전 런 이후 들어간 수리·레버가 이 궤적에 개입했는가 — 핵심 답**

| 수리/레버 (t7336→t7346) | 개입 | 결과 |
|---|---|---|
| A14 `_degenerate_axes`(`t2_gate_patch.py:3026-3042`) | **○ 6회/trial** | 085 의 `General` 날조는 막았으나, 079 에서는 이 태스크의 **정책군 전체**(`bank_accounts_bank_accounts`)를 결정 경로에서 제거했다. 단 그 경로는 `decide=True` 라 원래도 **본문을 안 날랐으므로**(`T2_PROCEED_DOCBODY=0`) **손실 원인은 아니다 — 이득도 손해도 0** |
| P4 `_fab_fix_note`(`t2_gate_patch.py:1888-1908`) | **○ trial 1 1회** | **문면이 커밋에 남았다**(t7336 은 0). 모델이 지시대로 첫 read 를 실행 `[060]~[063]`. **그런데 두 번째 read 가 `operator-scope` 에 막혀 이득이 상쇄됐다** |
| `T2_SEARCH_REARM`(t7336 016 처방 1) | ○ trial 0 2회 | 배달 성공(14,851자)하나 나른 재료가 이 결정과 무관 |
| `T2_DISCOVERY_STEP2` 레지스트리 폴백 | ○ trial 0 | **gold read 이름을 댔다 = 첫 벽 돌파**. 그 결과 trial 0 은 t7336 의 실패 지점을 넘어 **두 번째 벽(주문 단계)** 까지 갔다 |
| 선행 R1(`_grounded_candidates` operator 인자 분리) | **✕ 미적용** | `t2_gate_patch.py:2321` 축자 `if any(t in kl or kl in t for t in toks)` 그대로 · `_key_tokens('agent_tool_name') = {'agent','name','tool'}` 실측 · 치환 15회 재현 |
| 선행 R5(비가역 write 열린-enum 게이트) | **✕ 미적용** | `order_debit_card_5739` 앞에 어떤 우리 게이트도 서지 않는다(위 표 `require_doc_before`·`write_arg_enum`) — trial 0 의 손실이 정확히 그 자리 |

---

## 5. `T2_SEARCH_AGENT` 의 구조적 미배달 (이 태스크 전용 소견)

A3 `policy_ontology.doc_index` 실측:

```
bank_accounts_bank_accounts -> ['_general_']            (n=1)   ← 퇴화 축
checking_accounts           -> ['blue_account', …, '_general_', 'evergreen_account', …] (n=11)
```

이 태스크가 요구하는 정책 — **등급별 교체 한도·48시간 대기·RUSH $35·PREMIUM 디자인 무료·
주문 절차 4)5) "배송/디자인을 물어라"** — 는 전부 `doc_bank_accounts_bank_accounts_(general)_023 / _029 / _005`
에 있고, 이 셋은 모두 `bank_accounts_bank_accounts` 군이다.
`_degenerate_axes` 판정으로 그 군은 결정 경로에서 제외되고(`log 006/032/050/069/090/110`),
남은 축이 없으면 `_search_material` 은 `요청 축 … 모두 처리됨 — 침묵` 으로 빈 문자열을 돌려준다
(`t2_gate_patch.py:3216-3260` 부근). 이 런은 `T2_PROCEED_DOCBODY=0`·`T2_DOCS_AT_WRITE=0` 이라
**본문 배달 경로 자체가 꺼져 있다**.

⇒ **우리 층의 KB 채널은 이 태스크의 정책을 한 글자도 나르지 않는다.** 모델이 스스로
`KB_search_bm25` 를 쳐야 하는데, trial 0 은 그것을 **비가역 write 뒤(`[079]`)에** 했다.
이것은 *능동적 해악*이 아니라 **채널 부재**이므로 원인 귀속은 `model`(쓰기 전에 안 읽음)이고,
우리 층 몫은 *"레버가 살 자리인데 비어 있다"* 는 처방 근거로만 쓴다(§7 P1).

---

## 6. 선행 판정과의 대조 (`T7336_TASK_079.md`)

| 축 | t7336 | t7346 | 판정 |
|---|---|---|---|
| trial 0 (s626729) | ID-해결 실패 · MISSING 11 · `get_card_last_4_digits` give-루프 11턴 · 188K자 봉투 붕괴 | **ID-해결 성공** · MATCHED 11 · **주문 단계에서 손실** | **원인이 이동했다.** t7336 의 1차 결손(계좌/카드 ID read 부재)이 `T2_DISCOVERY_STEP2` 폴백으로 해소됨 |
| trial 1 (s373753) | ID-해결 성공 → **주문 단계 조기 커밋**(WRONGARG+EXTRA) → CWE | **ID-해결 도중 붕괴** · MISSING 11 · 이관 | **원인이 이동했다.** 새 원인 = 우리 `operator-scope` deny 4회 |
| 태스크 전체 | 0/2 | 0/2 | **점수 불변 · 두 seed 의 실패 모드가 서로 자리를 바꿨다** |
| `T2_GROUND -> CARLOS RODRIGUEZ` | trial 1 14회 | **trial 0 15회** | **같은 결함·미수리**(선행 R1 미적용) |
| 주문 단계 벽 | t7336 t1 · t7328 t1 에서 관측 | **t7346 t0 에서 재현** | **3개 sha·3개 sim 에서 같은 모양** = 태스크-특정이 아니라 **레버 공백** |
| `T2_VALUE_ACQUIRE` 오발화(선행 §7-2) | trial 0 3회 | **0회** | 이번엔 미발화(궤적이 `last 4` 문구를 안 냄) |
| 한국어 노트 손님 노출(선행 §7-3) | trial 0 msg51 | **0건** | 이번 궤적엔 재현 안 됨 |
| 봉투 붕괴 / CWE | 양쪽 1건씩 | **0건** | 재현 안 됨 |

**t7328 기준선 대조(sha 상이·비-A/B)**: `task_079` = 0/2. 그중 trial 1 의 변이표가
**MISSING 2 (`order_debit_card` RUSH·STANDARD) · WRONGARG 3 (`order_debit_card` standard/CLASSIC ×3)** 으로
**t7346 trial 0 과 완전히 같은 서명**이다. ⇒ 주문 단계 벽은 sha 를 가로질러 안정적으로 재현된다.

---

## 7. 원인 확정

### trial 0 (seed 626729) — **cause_primary = model** · 보조 = our_layer(비용)

- **1차(model)**: 결정점 `[065]`→`[073]`. 손님에게 배송·디자인을 **묻지 않고**, 등급 정책 본문을
  **읽지 않고**, 문맥에 실재하던 Light Blue 교체 이력(`[040]` `issue_reason:"lost"` ×2)을
  **빼지 않고**, 비가역 주문 3건을 자기 기본값으로 커밋했다(`[069]` 축자 *"use default options …
  to simplify the process"*). env 는 이후 재주문을 영구 차단(`[083]/[084]`) → gold 079_17 도달 불가.
  [[63]] 의 *"모델은 스스로 배제를 못한다"* 가 chk_2 에서 그대로 나왔다.
- **2차(our_layer·비용만)**: `T2_GROUND` 15회 오치환(`t2_gate_patch.py:2321`)으로
  `[098]`·`[103]` 구간이 `CARLOS RODRIGUEZ` unlock 시도–거부 루프가 됐다. **turn ≥ 65 = 손실 이후**
  이므로 성적의 원인이 아니다.
- **레버 공백(our_layer·구조)**: `order_debit_card_5739` 앞에 read-first / enum-질의 게이트가 없다
  (`require_doc_before.tools` = 이관 4종 · `write_arg_enum.applies_when.prefix` = `open_bank_account`).
- env: 정상(거부 메시지 정확·정보 충분). user_sim: **정상**([[21]] 면책 없음 — `[078]` 에서
  요구를 명확히 냈고 오도 0. 다만 그 요구가 **주문 뒤**에 온 것은 시나리오 설계상 정상이며,
  정책 절차가 *"먼저 물어라"* 를 명시하므로 agent 측 흡수 대상이다).

### trial 1 (seed 373753) — **cause_primary = our_layer** (CONFIRMED)

- **1차(our_layer)**: `t2_resolve.py:187-226` 의 `operator-scope` 가
  `get_debit_cards_by_account_id_7823` 의 unlock/call 을 **4회 거부**(`log 278·311·314·317`).
  그 도구는 카드 9변이(freeze/unfreeze/close) 전부의 유일한 `card_id` 원천이고,
  **우리 자신의 `FAB_STRIP` 노트(`[036]`)가 축자로 지시한 read** 다.
  우회 화이트리스트 `declared_required`(`t2_resolve.py:1118-1131`, A2 `procedures`/`follow_up_chains` 유래)
  18개에 그 이름이 없다(같은 A2 의 `arg_source_reads.card_id` 에는 있다) — **선언 간 모순**.
  결과: 모델은 `[064]` 에서 *"persistent issue with unlocking and using the tool to retrieve debit
  card information"* 이라 말하고 이관을 제안했다. **궤적 전체에 env 발 unlock 에러는 0건**이므로
  그 문장의 출처는 우리 deny 뿐이다([[55]] 0단계: 배선 → 문구 모순 → 계기 → 모델).
- **2차(model)**: `[012]` 의 상수 페르소나(`johndoe@example.com`), `[028]/[032]` 의 `shell grep` 오용,
  `[036]` 의 날조 card_id(=FAB_STRIP 이 옳게 막음). 다만 이들은 **회복 가능**했고,
  실제로 모델은 `[060]~[063]` 에서 지시대로 복구 궤도에 올라섰다 — **그 복구를 우리가 끊었다**.
- env: 정상(에러 0). user_sim: 정상(`[037]` 에서 정확히 재요구).

### 분기점 (두 trial 이 갈린 지점)

**turn 32.** trial 0 은 `unlock get_debit_cards_by_account_id_7823` 을 turn 32 에 발행해 **통과**했고
(`[033] Tool unlocked`), trial 1 은 같은 unlock 을 turn 47~63 에 발행해 **4회 전부 거부**당했다.
`operator-scope` 는 후보 ≥2 + 의도-서브 불일치일 때만 발화하므로, **같은 호출의 통과/거부가
발행 시점(후보 집합 크기·직전 손님 발화)에 좌우된다** — 우리 층이 만든 비결정성이다.

---

## 8. 처방 후보 (제안만 · 실행 금지 · gold 무참조)

- **P1 (최우선·trial 1 원인 직결)** `operator-scope`/`operator-find` 의 우회 술어를 넓힌다:
  `chosen` 이 A2 `arg_source_reads.*` 에 **원천 read 로 선언된 이름**이면 deny 하지 않는다.
  근거 = 우리가 이미 그 이름을 *"이걸 먼저 읽어라"* 로 모델에게 발화하고 있다(`_fab_fix_note`).
  닫힌 술어(선언 집합 소속 판정·도메인 리터럴 0·[[22]]). 부작용 계측([[70]]):
  `operator-scope` deny 수 감소분 ↔ 오도구 선택 증가분을 태스크별 부호표로.
- **P2 (동치·대안)** `declared_required` 의 재료에 `arg_source_reads` 를 합류시킨다
  (`t2_resolve.py:1118-1131` 한 줄). P1 과 같은 효과를 **한 자리**에서 얻는다.
- **P3 (trial 0 원인·선행 R5 재제출)** **비가역 write 의 열린-enum 인자** 게이트.
  술어 3항 모두 닫힘: ⑴그 write 의 스키마가 enum 파라미터를 선언하고
  ⑵그 enum 값 중 어느 것도 **손님 발화에 축자로 없고** ⑶그 도구가 `pending`/재실행 거부 부류이면,
  실행 전 1회 손님 선택을 요구한다. 태스크 id 조건 0([[70]] 금지선 준수).
  t7328·t7336·t7346 3개 sha 에서 같은 모양으로 재현되므로 태스크-특정이 아니다.
- **P4 (선행 R1 재제출·미적용분)** `_grounded_candidates`(`t2_gate_patch.py:2307-2325`)의
  후보 매칭을 **operator 인자에서 분리** — `arg` 가 A2 `operator_resolution=="discoverable"` 이면
  후보 집합을 `registry_names(agent)` 로 닫는다. 실측 재현: `_key_tokens('agent_tool_name')={'agent','name','tool'}`
  → `'name' in 'cardholder_name'` → 유일 후보 `CARLOS RODRIGUEZ`. trial 0 15/15 일치.
- **P5 (계기 수리·성적 무관)** `t2_forensic.deny_kind`(`t2_forensic.py:542`)가
  `Failed to …` 형상을 거절로 못 본다 → 실패한 write 가 `ok=True` 로 접혀 **DUP 위양성**을 만든다
  (본 건 `log_verification` 1건). [[25]] 우리 도구 100% 정답 의무.
- **P6 (구조·중기)** `bank_accounts_bank_accounts` 가 퇴화 축이라 우리 KB 채널이 이 도메인의
  **절차 문서 전부를 못 나른다**(§5). `_general_` 군에 대해서는 *결정*이 아니라
  **문서 델타 배달**(`T2_SEARCH_REARM` 과 같은 doc-only 경로)로 라우팅하는 안.
  ⚠[[62]] 4문 선행 필수 — 격리 프로브로 *"본문이 닿으면 모델이 쓰는가"* 를 먼저 재고,
  되면 레버는 **전달뿐**이어야 한다.

---

*작성 2026-08-23. 근거는 전부 `bank_t7346_halfB_20260822.results.json.gz` 궤적 축자와
`bank_t7346_halfB_20260822.log.gz` 의 `[sim=task_079#s626729|s373753]` 라인.
변이 판정은 `t2_forensic.mutation_diff` 정본(손 비교기 0). gold(`reward_info`)는 진단 대조에만 사용([[23]]).
수리 실행·코드 수정 0.*
