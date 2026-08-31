# TASK_22 — `t7391_reg12` (retail·ABox 스왑 1a) per-step 포렌식

- **런**: `bank_t7391_retail_20260829` 계열 회귀 12태스크 재런 · 결과 파일
  `reports/facet_rft_2026/sim_results/t7391_reg12.results.json.gz`
  (`info.git_commit = fc0055dc4e0a316c3f83133267fbd6faaa770992` · `num_trials=1` · `max_steps=200`)
- **도메인**: **retail**(banking 아님) · 에이전트 `Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8` T=0 ·
  user-sim `openrouter/openai/gpt-5.2`(reasoning low) · `mode=half_duplex` · `seed=626729`
- **대조군(PASS)**: `sim_results/hist_gpt52_reg12_PASS.results.json.gz` — **같은 12태스크·같은
  에이전트 모델·같은 user-sim**, task 22 는 `reward 1.0`. 이 문서의 핵심 대조축이다.
- **격리 프로브**: `reports/facet_rft_2026/x612_t7391_task22_ground_iso.py` (모델 0·env 0·재현 통과)
- **한 줄**: `reward 0.0 = DB 0.0 × NL_ASSERTION 1.0`. 최종 DB 가 gold 와 다른 칸은 **딱 하나** —
  주문 `#W9911714` 의 `address.address2` 가 gold `""` 인데 우리는 `"Suite 865"`(손님의 **옛 주소**
  부속칸)를 남겼다. 그 값을 대화에 처음 들여놓은 것은 모델이 아니라 **우리 층의 `T2_GROUND`
  제자리 치환**이다(msg[10]·격리 재현 바이트 동일).

---

## §0. 계기(instrument) 한계 — 먼저 적는다 ([[30]] · [[55]])

1. **이 런의 `.log.gz` 가 로컬에 없다.** 검색한 경로:
   `ls reports/facet_rft_2026/sim_results/ | grep -i 't7391'` → `t7391_reg12.results.json.gz`
   **한 파일뿐**. `fb_*`·`trace_*` 사이드카도 t7391 접두는 0건.
   ⇒ **stderr 로만 인쇄되는 `[T2_*]` 마커는 이 문서에서 직접 판정할 수 없다.** "미발화"와
   "발화했으나 미회수"를 로그로 가를 수 없다([[30]] — 쓰인 것과 회수된 것은 다르다).
   그래서 §4 레버표의 판정 근거는 ⑴ 런처가 export 한 플래그(`run_t7391_retail.sh` ·
   `go_stack.sh`) ⑵ **A2 선언에 재료가 있는가** ⑶ **궤적에 실제로 남은 문면·인자 변형**
   세 가지뿐이고, 그중 ⑶ 은 축자로 확정된다.
2. 런 sha `fc0055dc…` 는 로컬 트리에 없다. 대신 **동일성 증거**를 쓴다 —
   msg[10] 의 라이브 치환이 로컬 `t2_gate_patch.py` 함수 3개(`_first_fab_call` ·
   `_grounded_candidates` · `_subst_arg_value`)의 오프라인 출력과 **바이트 동일**하다
   (x612 A_repro). 즉 이 런은 로컬 트리의 그 경로를 그대로 썼다.
3. `num_trials=1` ⇒ trial 은 하나(trial 0). 분기점 분석 대상 없음.

---

## §1. 채점축 — `reward_info` 직독 (C583ⓖ: 축을 틀리면 표가 거짓말을 한다)

```
reward            = 0.0
reward_basis      = ["DB", "NL_ASSERTION"]
reward_breakdown  = {"DB": 0.0, "NL_ASSERTION": 1.0}
db_check          = {"db_match": false, "db_reward": 0.0}
nl_assertions     = []           # info.nl = "No nl_assertions to evaluate"
env_assertions    = []
```

- **점수축은 DB 해시 하나다.** `action_checks` 7칸이 붙어 있지만 `reward_basis` 에 `ACTION` 이
  없으므로 **진단용**이다.
- 진단용 action 표(gold 7칸 · match 2/7):
  | aid | 도구 | 인자 요지 | match |
  |---|---|---|---|
  | 22_0 | `find_user_id_by_name_zip` | Ethan/Garcia/80280 | ✔ |
  | 22_1 | `modify_user_address` | 101 Highway · **address2 `""`** · NY 10001 | ✘ |
  | 22_2 | `get_order_details` | `#W4967593` | ✘ |
  | 22_3 | `get_order_details` | `#W9911714` | ✘ |
  | 22_4 | `get_order_details` | `#W5733668` | ✘ |
  | 22_5 | `modify_pending_order_address` | `#W9911714` · **address2 `""`** | ✘ |
  | 22_6 | `modify_user_address` | 667 Highland Drive · Suite 865 · Denver | ✔ |
- **read 3칸(22_2~22_4)은 점수축이 아니다.** 에이전트는 `get_order_details` 를 **한 번도 안
  불렀다**(궤적 전수 확인 — messages 25개 중 `get_order_details` 0건). 그 대가는 DB 가 아니라
  msg[15]·msg[16] 의 env 오류 2건이다(§3-⑥).

**최종 DB 차이 = 한 칸.** 두 궤적을 끝까지 접으면
`user.address` 는 양쪽 다 `667 Highland Drive / Suite 865 / Denver CO 80280`(원상복구)로 같고,
`#W4967593`·`#W5733668` 은 양쪽 다 미변경이다. 남는 것은
`#W9911714.address.address2` = gold `""` ↔ 우리 `"Suite 865"` **하나뿐**이다.

---

## §2. 변이 집합 — 정본 `t2_forensic.mutation_diff` (trial 0)

⚠**정본 호출의 함정**(TASK_12 이 이미 적은 것과 동일): `F.mutating_tools()` 의 기본 인자는
`domain="banking_knowledge"` 다(`t2_forensic.py:1069`). 그대로 부르면 retail 도구가 하나도 안
잡혀 `clean=True` 라는 **거짓 표**가 나온다. `F.mutating_tools("retail")`(7종)로 부른다.

| 칸 | n | 내용 |
|---|---|---|
| gold | 3 | `modify_user_address`(NY·addr2 `""`) · `modify_pending_order_address`(#W9911714·addr2 `""`) · `modify_user_address`(Denver 복원) |
| **matched** | **1** | `modify_user_address`(Denver 복원·msg[21]) |
| **wrongarg** | **2** | `modify_user_address`(msg[10]) · `modify_pending_order_address`(msg[12]) — **둘 다 `address2` 한 필드만 틀림** |
| missing | 2 | 위 두 건의 gold 짝(같은 도구·같은 나머지 인자) |
| blocked | 2 | `modify_pending_order_address(#W4967593)` · `(#W5733668)` — `deny="env"` · `"Error: Non-pending order cannot be modified"`(msg[15]/[16]) |
| extra | 0 | — |
| dup | 0 | — |

### WRONGARG 필드별 대조 (보낸 인자 ↔ gold 인자)

| 인자 | msg[10] 보낸 값 | gold 22_1 | msg[12] 보낸 값 | gold 22_5 |
|---|---|---|---|---|
| `user_id`/`order_id` | `ethan_garcia_1261` | 동일 | `#W9911714` | 동일 |
| `address1` | `101 Highway` | 동일 | `101 Highway` | 동일 |
| **`address2`** | **`Suite 865`** | **`""`** | **`Suite 865`** | **`""`** |
| `city`/`state`/`country`/`zip` | New York/NY/USA/10001 | 동일 | New York/NY/USA/10001 | 동일 |

**변이 실패는 단일 필드 `address2` 다.** 그리고 msg[10] 은 gold 로 복구되므로(22_6) 점수에
남는 것은 **msg[12] 한 건**이다.

---

## §3. 결정 지점 step-by-step (축자 인용 · `sim#msg` 표기)

### ⓵ msg[1] — 손님이 새 주소를 축자로 준다 (address2 는 **말하지 않는다**)
> user: *"Hi—I'd like to update my shipping address on my account to **101 Highway, New York, NY
> 10001**, and also update **any existing orders** to use that address as well."*

부속칸(Suite/Apt)은 발화에 **없다**. gold 가 `address2: ""` 인 근거가 여기다.

### ⓶ msg[6] — 모델은 처음부터 **gold 인자를 냈다**. 게이트가 그것을 버렸다
`raw_data` 축자(완성 원본, 호출 2건):
```
modify_user_address {"user_id":"ethan_garcia_1261","address1":"101 Highway",
                     "address2":"","city":"New York","state":"NY","country":"USA","zip":"10001"}
get_user_details    {"user_id":"ethan_garcia_1261"}
```
커밋된 `tool_calls` 에는 `get_user_details` 하나만 남았고 본문에 우리 노트가 붙었다:
> assistant: *"I haven't made any changes yet, Ethan. … I need your explicit confirmation. …
> **[Note: the tool call(s) above were blocked by a policy gate and were NOT executed.**
> Satisfy the gate requirement (authenticate / get explicit user confirmation / check the record's
> status / fix the operation) before attempting the action again.] ([G2_CONFIRM_WRITE] blocked by
> policy gate: explicit user confirmation (yes) of the...)"*

- 차단은 **정책상 옳다**. retail 정책 축자: *"Before taking any action that updates the database
  (cancel, modify, return, exchange), you must list the action details and obtain explicit user
  confirmation (yes) to proceed."*(`sim['policy']`).
- 그러나 **버려진 인자가 gold 였다**. 차단된 호출은 메모되지 않고 폐기된다
  (`t2_gate_patch.py:6694` `_BLOCK_NOTE` 경로 — 확인 후 **재제출** 기전은 없다.
  검색한 곳: `grep -n "재제출\|replay\|pending_write\|_pending_confirm" t2_gate_patch.py`
  → 재제출 관련 히트는 tau2 평가 replay 얘기뿐이고 차단-write 메모는 0건).
- msg[7] 에서 `get_user_details` 가 돌아 **옛 주소가 문맥에 들어왔다**:
  `"address": {"address1":"667 Highland Drive","address2":"Suite 865","city":"Denver",…}`.
  ★대조군 PASS 는 이 read 를 **한 번도 하지 않았다** — 즉 PASS 문맥에는 `Suite 865` 가 아예
  없었다(§5).

### ⓷ msg[8] — 모델이 **gold 5칸을 통째로** 다시 냈고, 게이트가 4건을 또 버렸다
`raw_data` 축자(호출 4건 · 전부 `"address2": ""`):
```
modify_user_address          … "address2":"" …
modify_pending_order_address {"order_id":"#W4967593", … "address2":"" …}
modify_pending_order_address {"order_id":"#W9911714", … "address2":"" …}   ← gold 22_5 와 인자 동일
modify_pending_order_address {"order_id":"#W5733668", … "address2":"" …}
```
커밋된 `tool_calls` = **null**(4건 전부 제거) · 본문에 `[G2_CONFIRM_WRITE]` 4회 연접.
⇒ **이 시점까지 모델이 낸 새-주소 write 5건은 5/5 가 `address2:""`(gold)** 였다.

### ⓸ msg[9] — 손님 확인
> user: *"Yes."*

### ⓹ **msg[10] — 결정 지점.** 모델이 날조했고, **우리 층이 그것을 옛 값으로 바꿔 실행했다**
`raw_data`(모델 원본) ↔ 커밋된 `tool_calls`(실행분), **tool_call id 동일**
(`chatcmpl-tool-1dbf84f9d330407294e6325c4c459543`):
```
모델 : {"user_id":"ethan_garcia_1261","address1":"101 Highway","address2":"Apt 1", …}
실행 : {"user_id":"ethan_garcia_1261","address1":"101 Highway","address2":"Suite 865", …}
```
- 모델의 `"Apt 1"` 은 대화 어디에도 없다 ⇒ `_first_fab_call` 이 fab 로 잡는다
  (`address2` 는 `DEFAULT_ARG_HINTS`(`t2_gate_patch.py:69-70`)의 `"address"` 로 스캔 대상).
- `_grounded_candidates("address2","Apt 1", msgs, lenient=True)` → **`['Suite 865']`**
  (유일 후보 = msg[7] 프로필의 옛 부속칸).
- `t2_gate_patch.py:8435-8445`:
  ```
  if fab is not None and ground and subs < 8:
      gcands = _grounded_candidates(gk, gs, state.messages, lenient=True)
      if len(gcands) == 1 and gcands[0] != gs and _subst_arg_value(gtc, gk, gs, gcands[0]):
          … print("[T2_GROUND] substituted arg=%s val=%s -> %s" …)
          continue   # ← 거절·재생성 경로를 **건너뛴다**
  ```
- **격리 재현**(x612 A_repro): 같은 msgs[0..9] + 같은 모델 원본 인자 → 산출이 라이브 실행분과
  **바이트 동일**(`⇒ 바이트 동일: True`). 우리 층 귀속의 코드 경로가 확정된다.
- msg[11] 도구 결과가 그 값을 **문맥에 되돌려 준다**:
  `{"user_id":"ethan_garcia_1261", …, "address":{"address1":"101 Highway","address2":"Suite 865", …}}`

### ⓺ **msg[12] — 점수를 결정한 write.** 모델이 msg[11] 의 에코를 그대로 베낀다
`raw_data` == 커밋 `tool_calls`(**이 턴엔 엔진 개입 0**):
```
modify_pending_order_address {"order_id":"#W9911714","address1":"101 Highway",
                              "address2":"Suite 865","city":"New York","state":"NY",
                              "country":"USA","zip":"10001"}
```
msg[13] 결과가 성공을 확인한다 — `"address":{…,"address2":"Suite 865",…},"status":"pending"`.
**이 한 칸이 `db_match=false` 의 전부다.**

### ⓻ msg[14] — read 없이 나머지 두 주문에 write, env 가 거절
호출 2건(한 턴에 2개 = retail 정책 *"at most one tool call at a time"* 위반이나 우리 게이트는
이를 검사하지 않는다) → msg[15]/[16] 둘 다
> tool: *"Error: Non-pending order cannot be modified"*

gold 는 여기서 `get_order_details` 3건(22_2~22_4)으로 상태를 먼저 읽는다. 우리 층의 상태-선행
게이트 `G5_STATUS_PRECONDITION` 의 `applies_to` 에 **`modify_pending_order_address` 가 없다**
(`a2/retail.gate.json` — 선언 축자: `["modify_pending_order_items","cancel_pending_order",
"exchange_delivered_order_items","return_delivered_order_items"]`). DB 피해는 0(둘 다 blocked).

### ⓼ msg[17]~[24] — 원상복구는 정확히 됐다
> assistant(msg[17]): *"Your default address has been successfully updated … the other orders
> could not be updated because they are not in a pending state."*
> user(msg[18]): *"Actually, I regret that change. Please change my **default address on my
> account back to the original address in Denver, CO 80280**."*

msg[19] 확인 요청 → msg[20] *"Yes."* → msg[21] `modify_user_address(667 Highland Drive / Suite
865 / Denver CO 80280)` = **gold 22_6 와 인자 동일**(matched). msg[24] `###STOP###`.
⇒ 손님이 되돌려 달라고 한 것은 **기본 주소뿐**이고, 주문 주소는 NY 로 남는 것이 gold 다.
우리 궤적도 그렇게 남겼다 — 값 한 칸만 빼고.

---

## §4. 레버 발화표 (§0-1 의 계기 한계 아래에서)

| 레버 | 플래그 | 판정 | 근거 |
|---|---|---|---|
| `G2_CONFIRM_WRITE` | A2 게이트 | **발화·5건 차단** | msg[6] 1건 · msg[8] 4건. 본문에 문면 축자 잔존. 정책상 옳은 차단이나 **폐기된 5건 중 2건이 gold 인자**였다(§3⓶⓷) |
| `T2_GROUND` | `go_stack.sh:36` `T2_GROUND=1` | **발화·오작동** | msg[10] `address2` `Apt 1`→`Suite 865` 치환. 격리 바이트 동일 재현(x612) |
| `T2_KEEP_DENY_BODY` | `run_t7391_retail.sh` env_retail | 발화 | 거절 사유가 본문에 남아 있다([[64]] 준수 — 이름 없는 억제가 아니다) |
| `T2_FAB_STRIP` | `go_stack.sh:217` | **미발화(선점됨)** | `t2_gate_patch.py:12499` 는 **재생성 소진 시**의 abstain 분기다. msg[10] 은 8435 의 치환이 `continue` 로 앞질러 소진 자체가 없었다 |
| PROVENANCE 거절/재생성 | `T2_PROV_REGEN=1`(`go_stack.sh:33`) | **미발화(선점됨)** | 같은 이유. ★x612 C_feedback: 만약 갔더라도 문면은 *"grounded address2 value(s) … are: **Suite 865**. Use ONLY one of these"*(`GROUND_FEEDBACK`·`t2_gate_patch.py:3080`) — **같은 오값을 지시**했을 것이다 |
| `T2_PIN_READ` / `T2_PIN_READ_STEPS` | `go_stack.sh:454` ON | **미발화(재료 부재)** | `_read_routine_pin`(`t2_gate_patch.py:3695-3697`)은 `a2["procedures"]` 가 비면 즉시 `None`. `a2/retail.*.json` 3파일 전부 `procedures` 키 **0건**(파이썬으로 키 소속 확인) |
| `T2_DEMANDED_STEP` | 플래그 자체 없음 | 미발화 | `grep -n "export T2_DEMANDED_STEP" go_stack.sh` → 0건 · 코드는 `t2_gate_patch.py:10189` 에 있으나 절차 노드 의존 |
| READ-FIRST / `T2_SG_REQREADS` | `run_t7391_retail.sh` env_retail | **미발화(재료 부재)** | 요구-읽기 선언(`isolate`/`requires_reads`)이 retail A2 에 0건 |
| `T2_SG_DOCS` | `go_stack.sh:75` · env_retail | **미발화(재료 부재)** | `isolate.docs` 선언이 retail A2 에 0건 |
| `T2_SEARCH_AGENT` / `T2_SEARCH_REARM` | env_retail · `go_stack.sh` | **침묵** | retail KB 문서 코퍼스 없음 — 검색 재료 0 |
| `T2_REQUIRE_DOC_DELIVER` | `go_stack.sh:497` | **침묵** | 같은 이유(`t2_gate_patch.py:3950` 은 도출 0편이면 침묵으로 빠진다) |
| `T2_CLAIMPROV` | `go_stack.sh:280` 은 CAP 만 | 미발화 | 궤적에 `[T2_CLAIMPROV]` 계열 문면 0 |
| `T2_FOLLOWUP_*` | `go_stack.sh:229-230` | 미발화 | msg[17]·msg[23] 완료 보고에 후속 요구 문면 0 |
| `T2_ARG_PRODUCERS` | `go_stack.sh:276` | **판정 불가** | retail `producers` 는 `authenticated_user_record → get_user_details` 하나뿐이고 msg[6] 에서 실제로 그 도구가 호출됐다. 넛지가 그 호출을 유발했는지는 **로그 없이 가릴 수 없다**(§0-1) |
| `G5_STATUS_PRECONDITION` | A2 게이트 | **오선언(미적용)** | `applies_to` 에 `modify_pending_order_address` 부재 → msg[14] 무-read write 2건 통과(§3⓻) |
| `regen_resolver_specs`(A2) | `a2/retail.gate.json` | 미발화(잠복 위험) | `modify_pending_order_address` 의 `address1/address2/city/state/zip` 원천을 **`get_order_details` 의 그 주문 현재 주소**로 선언해 놨다. 이 태스크에서 발화했다면 **또 옛 주소를 지시**했을 것이다 |

**직전 런 이후 들어간 수리가 이 궤적에 개입했는가**: 개입한 것은 `T2_GROUND`(구 레버) 하나이고
**샀다기보다 팔았다**. 신규 전달 레버(`T2_SG_DOCS`·`T2_SEARCH_*`·`T2_REQUIRE_DOC_DELIVER`·
`T2_PIN_READ_STEPS`)는 **retail A2 에 재료가 없어 구조적으로 침묵**한다 — retail 은 저작 증분 0
에서 돈 런이므로([[run_t7391_retail.sh]] 주석 축자: *"retail A2 는 개발된 적이 없다"*) 이 표의
"미발화" 대부분은 레버의 실패가 아니라 **선언 부재**다.

---

## §5. 대조군 PASS 와의 대조 — 같은 모델이 같은 태스크에서 8/8 로 `""` 를 낸다

`hist_gpt52_reg12_PASS.results.json.gz` task 22 = `reward 1.0`(`db_match: true`). 동일 에이전트
모델·동일 user-sim. **우리 층 문면·인자 개입 0**(전 턴 `raw_data` == 커밋 `tool_calls`).

PASS 궤적 요지:
- msg[8] `modify_user_address … "address2":""` (확인은 msg[7] 손님 발화 *"Yes, that's correct—please
  update both …"* 로 이미 받아 둔 상태)
- msg[12] `#W4967593` 시도 → `"Error: Non-pending order cannot be modified"`
- msg[16] **`get_order_details(#W9911714)`** → msg[18] `modify_pending_order_address … "address2":""`
- msg[22] `get_order_details(#W5733668)` → `"status": "processed"` 확인 후 **쓰지 않음**
- msg[28] `modify_user_address(667 Highland Drive / Suite 865 / …)` 복원
- ★PASS 는 `get_user_details` 를 **한 번도 부르지 않았다** ⇒ `Suite 865` 가 문맥에 들어온 적이
  없다. msg[17]/[23] 의 `get_order_details` 출력에는 있지만, 그때 모델은 이미 `""` 를 쓰고 있었다.

### x612 B_prior — 모델 **원본** `address2` 전수 (두 런·task 22·n=14)

| 런 | msg | 도구 | address1 | 모델 원본 address2 |
|---|---|---|---|---|
| t7391 | 6 | `modify_user_address` | 101 Highway | `''` |
| t7391 | 8 | `modify_user_address` | 101 Highway | `''` |
| t7391 | 8 | `modify_pending_order_address` ×3 | 101 Highway | `''` `''` `''` |
| PASS | 8 | `modify_user_address` | 101 Highway | `''` |
| PASS | 12 | `modify_pending_order_address` | 101 Highway | `''` |
| PASS | 18 | `modify_pending_order_address` | 101 Highway | `''` |
| **t7391** | **10** | `modify_user_address` | 101 Highway | **`'Apt 1'`** ← 모델 날조 |
| **t7391** | **12** | `modify_pending_order_address` | 101 Highway | **`'Suite 865'`** ← 우리 값 에코 후 |
| **t7391** | **14** | `modify_pending_order_address` ×2 | 101 Highway | **`'Suite 865'` ×2** |
| t7391 | 21 | `modify_user_address` | 667 Highland Drive | `'Suite 865'`(gold·복원) |

**새 주소(`address1="101 Highway"`) write 11건**을 우리 값의 문맥 진입(msg[11]) 기준으로 가르면:
- 진입 **전** 8건 → **8/8 `""`**(gold)
- 진입 **후** 3건 → **0/3 `""`**, 3/3 `"Suite 865"`
- 그 사이 1건(msg[10]) = 모델 날조 `"Apt 1"`

---

## §6. 선행 판정과의 대조 — **같은 계열이 이미 2026-07-11 에 실측됐다**

검색한 경로: `ls reports/facet_rft_2026/ | grep -i retail` ·
`grep -rn "address2" reports/facet_rft_2026/*.md` ·
`grep -rn "ethan_garcia\|W9911714" reports/facet_rft_2026/*.md` ·
`ls reports/facet_rft_2026/tasks_reg12/ tasks__20260829/`.

1. `CENSUS_LEVERS_DESIGN_2026_07_11.md:29` 축자:
   > *"fuzzy 치환 표적 = 양 arm 0건(fix 0·break 0). **empty-치환 = break 실재**(t59: gold
   > address2가 정당한 빈 값인데 'Suite 165' 채움) → 설계의 empty-게이트가 기각 확정.
   > ⇒ **GROUND-VERBATIM 폐기(죽은 레버·§1.3).**"*
   ⇒ *"gold address2 가 정당한 빈 값인데 우리가 옛 부속칸을 채운다"* 는 **동일 현상**이 이미
   측정됐고, 그때 폐기된 것은 **신설 레버(GROUND-VERBATIM)의 empty-게이트**였다.
   **오늘 뚫린 문은 다른 문이다** — 값이 빈 문자열이 아니라 **날조**(`"Apt 1"`)여서
   기존 P-A `T2_GROUND` 의 일반 경로(`:8435` |C|=1 치환)로 들어왔다. 같은 결과, 다른 입구.
2. 같은 문서 §1 스코프 한계 축자:
   > *"C 원천 = tool 출력만(`_grounded_candidates` 구조 동일). **사용자 발화-원천 값의
   > 오복사**… 는 이 레버 밖"*
   ⇒ 후보 원천이 도구 출력뿐이라는 **구조적 한계가 그때 이미 문서화**돼 있었다. task 22 의
   `address2` 정답 원천은 **손님 발화의 부재**이므로 이 레버의 사정거리 밖이다.
3. `RETAIL_FULL_FAIL_CENSUS_2026_07_11.md`: task 22 는 당시 COMP 456 sim 에서 **FLAKY(2/4)** 였고
   버킷에 `WRONG_ADDRESS 9` 가 있다. 같은 문서 §2-D 는 *"t86 … Dallas 주소 오복사 · t102 …
   Seattle 주소 오복사 · t109 … 구주소 오복사 3/4"* 로 **구주소 오복사 계열**을 이미 이름
   붙였다. 오늘 task 22 는 그 계열의 **필드 단위(address2) 판본**이다.
4. `tasks_reg12/` · `tasks__20260829/` 의 선행 문서(TASK_1/3/4/9/12)에는 task 22 절이 없다
   (`ls` 로 확인 — 파일 목록에 22 없음). 이 문서가 이 런 task 22 의 1차 자료다.

**⇒ 원인은 달라지지 않았다. 계열이 같고, 2026-07-11 에 한 번 닫으려다 "신설 레버"만 폐기하고
기존 치환 경로는 열어 둔 자리가 재발했다.**

---

## §7. 원인 확정 ([[77]] 4칸 · 귀속 4주체)

### 주장 A — **model** (1차)
**①주장+양화**: t7391 task 22 sim#msg[10] 에서 모델이 `address2` 에 문맥 부재값 `"Apt 1"` 를
날조했다. 같은 태스크에서 이 모델의 새-주소 `address2` 사전분포는 **8/8 = `""`**(우리 값 문맥
진입 전·두 런 합산·x612 B_prior).
**②근거**: `messages[10].raw_data.choices[0].message.tool_calls[0].function.arguments` 축자
`"address2": "Apt 1"` · 대조 `messages[6]`·`messages[8]` raw 축자 `"address2": ""`.
**③반증 조건**: msg[10] 의 raw 완성에 `"Apt 1"` 이 없거나(=우리 층이 만든 값이거나),
`"Apt 1"` 이 대화 문맥 어딘가에 실재하면(=날조 아님) 거짓. → 전자는 raw 축자로,
후자는 `_ctx_has("Apt 1", ctx)=False`(x612 A_repro 가 fab 로 잡음)로 각각 반증 실패.
**④선행 확인**: `grep -rn "address2" reports/facet_rft_2026/*.md` ·
`RETAIL_FULL_FAIL_CENSUS_2026_07_11.md` §2-F(값 충실도) — 자유텍스트 주소 날조는 기지 계열.

### 주장 B — **our_layer** (2차·CONFIRMED 코드 경로)
**①주장+양화**: `T2_GROUND` 제자리 치환(`t2_gate_patch.py:8435-8445`)이 sim#msg[10] 에서
`address2` 를 `"Apt 1"` → **`"Suite 865"`**(손님의 옛 부속칸)로 바꿔 실행했고, 그 값이 msg[11]
도구 결과로 문맥에 되돌아온 뒤 모델의 새-주소 `address2` 는 **3/3 이 `"Suite 865"`**가 됐다
(msg[12]·msg[14]×2). 그중 msg[12] 한 건이 **점수축 DB 의 유일한 불일치 칸**이다.
**②근거**: ⑴ 코드 축자 `if len(gcands) == 1 and gcands[0] != gs and _subst_arg_value(...)` ·
`continue  # 치환값은 문맥-실재 …`(`t2_gate_patch.py:8438`·`:8445`) ⑵ 격리 재현
`x612_t7391_task22_ground_iso.py` A_repro — 오프라인 산출이 라이브 커밋 인자와 **바이트 동일**
(`⇒ 바이트 동일: True`) ⑶ msg[11] 도구 결과 축자 `"address2": "Suite 865"` ⑷ msg[12] raw 축자
`"address2": "Suite 865"`(이 턴 엔진 개입 0 — raw == 커밋).
**③반증 조건**: ㉠ msg[10] 의 커밋 인자가 모델 raw 와 같았다면(치환 없음) 거짓 — 반증 실패.
㉡ **치환을 끄고 같은 문맥으로 재생성했을 때 모델이 다시 `"Apt 1"` 류를 내고 msg[12] 가
`"Suite 865"` 가 아닌 다른 오값이 된다면**, "우리 값이 앵커였다"는 부분은 거짓이 된다 —
**이 팔은 아직 안 쟀다**(모델 호출이 필요). ㉢ 반대로 `GROUND_FEEDBACK` 재생성 경로도
`"Suite 865"` 를 지시하므로(x612 C_feedback 축자) *"거절로 갔으면 살았다"* 는 주장은
**지금 근거로는 성립하지 않는다** — 그래서 처방은 §8-①(슬롯 범위)이지 "치환 대신 거절"이 아니다.
**④선행 확인**: `CENSUS_LEVERS_DESIGN_2026_07_11.md:29`(empty-치환 break 실측·t59) ·
같은 문서 §1 스코프 한계(후보 원천=도구 출력뿐) · `t2_gate_patch.py:2534` 주석(같은 함수가
뱅킹에서 도구-선택자 슬롯에 371/371 오작동한 **선례**).

### 주장 C — **our_layer** (3차·비용은 0 이나 기록)
`G5_STATUS_PRECONDITION.applies_to`(`a2/retail.gate.json`)에
`modify_pending_order_address` 가 빠져 있어 msg[14] 의 무-read write 2건이 통과했다.
DB 피해 0(env 가 둘 다 거절) · gold read 3칸(22_2~22_4) 미수행의 우리-층 대응물.
**반증 조건**: 선언에 그 도구가 실재하면 거짓 — 파일 축자로 반증 실패.

### env / user_sim
- **env**: 정상. `"Error: Non-pending order cannot be modified"` 는 옳은 거절이고(대조군 PASS
  에서도 동일 문면), 오도 0.
- **user_sim**: 정상. msg[1] 에서 주소를 축자로 주고 부속칸을 말하지 않았다(gold 와 일치),
  msg[9]/[20] 에서 확인을 줬고 msg[18] 에서 철회 범위를 *"default address on my account"* 로
  명확히 한정했다. 오도 0. ([[21]] — user-sim 요인으로 종결하지 않는다.)

---

## §8. 처방 후보 (제안만 · 구현 금지)

1. **`T2_GROUND` 의 슬롯 범위를 좁힌다 — 1순위.**
   `DEFAULT_ARG_HINTS`(`t2_gate_patch.py:69-70`)의 `"address"` 는 **식별자 힌트 목록**에 얹혀
   있어서, *"덮어쓸 새 값"* 을 담는 free-text 슬롯이 *"기존 레코드를 가리키는 식별자"* 와 같은
   검사를 받는다. 두 슬롯은 **진리 원천이 다르다** — 식별자는 도구 출력, 덮어쓰기 값은 손님
   발화다. 후보 필터가 `len(s) < 4`(`:3068`)라 **빈 값은 원리상 후보가 못 되므로**, 이 슬롯에서
   `T2_GROUND` 는 *"손님이 비워 달라고 한 칸"* 을 항상 옛 값으로 되살린다.
   ⇒ 닫힌 술어 후보: *환경 스키마상 그 도구의 **자기 레코드 필드**를 통째로 덮어쓰는 인자
   집합*(retail `modify_*_address` 의 address/city/state/zip)은 fab-스캔 대상에서 제외.
   ⚠[[70]] 무엇을 파는가: 주소 자유텍스트 날조(선행 t17 계열)의 검출을 잃는다 ⇒ **부호표
   필수**(retail 전수에서 address 슬롯 치환 건수를 fix/break 로 갈라 센다).
2. **`_grounded_candidates` 에 "부재도 후보" 를 넣지 않는다 — 반대 방향 금지.**
   `""` 를 후보로 넣는 설계는 2026-07-11 에 **이미 기각**됐다(`CENSUS_LEVERS_DESIGN` §1 empty-게이트).
   재론 금지.
3. **차단된 write 의 인자를 확인 직후 되살린다(replay-after-confirm).**
   이 sim 은 gold 인자를 **두 번**(msg[6]·msg[8]) 내고 두 번 다 게이트가 버렸고, 확인 뒤
   재생성에서 무너졌다. 확인이 떨어진 턴에 *직전에 차단된 동일 도구 호출의 인자*를 그대로
   되제시(또는 본문에 축자 재게시)하면 재생성 손실이 닫힌다.
   ⚠[[62]] 자기점검: 결손을 먼저 쟀는가 → **쟀다**(이 sim 8/8 → 0/3). 다만 이는 1 sim 이므로
   **런 전수 센서스**(차단 write 의 인자가 확인 후 재생성에서 바뀌는 비율)가 선행 조건.
4. **`G5_STATUS_PRECONDITION.applies_to` 에 `modify_pending_order_address` 추가**
   (`a2/retail.gate.json` + `retail.settings.json` **양쪽 동기화**·[[24]]). 근거 = retail 정책
   축자 *"cancel or modify **pending** orders"* — gold 도 22_2~22_4 로 상태를 먼저 읽는다.
   기대: msg[14] 형 무-read write 제거 + gold read 3칸 유도. DB 이득은 이 태스크에선 0.
5. **`a2/retail.gate.json` 의 `regen_resolver_specs`(`modify_pending_order_address` 의
   `address1/address2/city/state/zip` → `get_order_details` 의 현재 주소) 재검토.**
   *"주문 주소를 바꿔 달라"* 가 이 도구의 존재 이유인데 원천을 **그 주문의 현재 주소**로
   선언해 놓았다. 이번 sim 에서는 발화하지 않았으나(재생성 자체가 없었다) 발화하면 §7-B 와
   **같은 오값을 지시**한다. ⇒ 잠복 위험으로 등재.
