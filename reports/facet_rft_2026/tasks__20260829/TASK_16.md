# TASK_16 — `t7391_reg12` (retail · ABox-swap 1a) per-step 포렌식

작성 2026-08-29 · 전부 로컬 · 모델 호출 0 · 수리 실행 0([[23]] gold=진단 전용)
근거 파일 = `C:\workspace\ba-frft\reports\facet_rft_2026\sim_results\t7391_reg12.results.json.gz`
대조군 = `C:\workspace\ba-frft\reports\facet_rft_2026\sim_results\hist_gpt52_reg12_PASS.results.json.gz`
런 스크립트 = `C:\workspace\ba-frft\scripts\distill\tau2\run_t7391_retail.sh`
격리 재현 = 이 문서 §6-ⓐ (정본 술어 `t2_gate_patch.membership_violation` 을 **이 sim 자신의 문맥**으로 호출 · 프롬프트 저작 0 · 부정통제 포함)

> ⚠**요청 경로 변경**: 지시 경로는 `tasks_reg12/TASK_16.md` 였으나 `C:\workspace\.claude\hooks\scaffold_guard.py:200-201`
> 이 `reports/` 아래 신설 .md 를 차단하고 예외 정규식이 `^TASK_\d+[a-z]?\.md$` ∧ `/tasks_+\d{8}/` 다.
> `tasks_reg12/` 는 `\d{8}` 이 아니라 **차단된다**. 같은 런의 형제 보고서가 이미
> `tasks__20260829/TASK_{1,3,4,9}.md` 로 서 있어 **정본 명명을 따랐다**.

> ⚠**요청 데이터 경로도 변경**: 지시된 `bank_t7391_retail_20260829_undefined_reg12.{results,log}.json.gz` ·
> 기준선 `undefined.results.json.gz` 는 **로컬에 없다**. 검색 경로 =
> `find C:\workspace\ba-frft -iname "*7391*"` → results 1 + 러너 1 + 격리프로브 3 ·
> `ls reports/facet_rft_2026/sim_results | grep -iE "7391|reg12"` → 2건. 실물은 `t7391_reg12.results.json.gz` 하나다.
> **`*.log.gz`·`fb_*`·`trace_*` 0건** ⇒ stderr 레버 마커(`[T2_CONS]`·`[T2_RESOLVE]` …)로는
> **아무것도 판정하지 않았다**([[30]] *"쓰이는 것과 회수되는 것이 다르다"*). 아래는 **궤적 축자 +
> 선언/코드 + 오프라인 재현**만으로 세운다.

- **런**: `git_commit=fc0055dc4e0a…` · agent `Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8`(로컬 8141) ·
  user-sim `openrouter/openai/gpt-5.2`(temp 0 · reasoning low) · `num_trials=1` ⇒ **trial 0 하나뿐**(분기점 절 불요)
- **결과**: reward **0.0** · `termination_reason=user_stop` · 26 메시지 · 277.5s · seed 626729

---

## §1. 채점축 — DB ∧ NL_ASSERTION **둘 다** 떨어졌다

```
reward            0.0
reward_basis      ['DB', 'NL_ASSERTION']
reward_breakdown  {'DB': 0.0, 'NL_ASSERTION': 0.0}
db_check          {'db_match': False, 'db_reward': 0.0}
env_assertions    []
communicate_checks[0]  {'info': '8276.23', 'met': False, 'justification': "Information '8276.23' not communicated."}
```

⇒ ACTION 축이 아니다. `action_checks` 9행은 **진단용으로만** 읽는다(C583ⓖ). read 5행 중 4행 match,
`calculate` 1행 unmatch, write 3행 중 **2 match · 1 unmatch**:

| aid | tool | action_match | 비고 |
|---|---|---|---|
| 16_0..16_4 | `find_user_id_by_name_zip` · `get_user_details` · `get_order_details`×3 | **True** ×5 | 읽기는 전부 옳았다 |
| 16_5 | `calculate("3131.1 + 4777.75 + 367.38")` | False | 궤적에 `calculate` 호출 **0회** |
| 16_6 / 16_7 | `cancel_pending_order(#W5199551 / #W8665881, "no longer needed")` | **True** | |
| **16_8** | `return_delivered_order_items(#W9389413, ["2554056026"], paypal_5364164)` | **False** | ★유일한 DB 손실 |

`nl_assertions[0].justification` 축자:
> *"The agent told the user the total refund from the two cancellations was $7,908.85 and stated the
> watch return could not be processed, so they never provided (or confirmed) a total refund amount of $8,276.23."*

⇒ **NL 축은 DB 축의 하류다.** 8,276.23 = 3,131.10 + 4,777.75 + **367.38**(= 반품 실패한 그 품목의 가격).
반품이 성사됐으면 나올 수 있었던 수다. 별개 결손이 아니다(§6-ⓔ 에서 정량).

---

## §2. 변이 집합 — 정본 `t2_forensic` (⛔손 비교기 0)

### 2-a. ★계기 함정 먼저 — `mutating_tools()` 의 기본값이 **banking** 이다

```python
import t2_forensic as F
F.mutation_diff(sim, F.mutating_tools())        # → missing/wrongarg/extra/dup/blocked 전부 0 · clean=True
```

`t2_forensic.py:1069` `def mutating_tools(domain="banking_knowledge")`. **retail 런에 기본값을 쓰면
표가 "무결"이라고 거짓말한다.** 이 sim 은 실제로 gold 변이 3개 중 1개를 놓쳤다.
⇒ **retail/airline 포렌식은 `F.mutating_tools('retail')` 를 명시해야 한다.**([[67]] *"이름을 믿지 마라"* 동형)

### 2-b. 올바른 표 (`F.mutating_tools('retail')` · trial 0 · 단일 trial)

| 칸 | n | 내용 |
|---|---|---|
| gold | 3 | `cancel_pending_order(#W5199551)` · `cancel_pending_order(#W8665881)` · `return_delivered_order_items(#W9389413, ["2554056026"], paypal_5364164)` |
| matched | 2 | cancel ×2 (인자까지 축자 일치) |
| **missing** | **1** | `return_delivered_order_items(#W9389413, **["2554056026"]**, paypal_5364164)` |
| **blocked** | **1** | `return_delivered_order_items(#W9389413, **["1994478369"]**, paypal_5364164)` → `deny='env'` · `marker="Error: Some item not found"` |
| wrongarg | 0 | ※`mut_key` 가 인자까지 접어 키를 만들므로 **같은 도구·다른 item_ids** 는 `done` 에 안 들어가 WRONGARG 칸에 안 잡힌다. **실질은 WRONGARG 1건**이다 |
| extra / dup | 0 / 0 | |
| sidecar | `unknown` | 사이드카 미회수 ⇒ 재생성으로 지워진 우리-층 반려는 **모른다**([[25]]) |

**필드별 인자 대조 (16_8)**

| 필드 | 보낸 값 | gold | 판정 |
|---|---|---|---|
| `order_id` | `#W9389413` | `#W9389413` | ✅ **일치** |
| `payment_method_id` | `paypal_5364164` | `paypal_5364164` | ✅ 일치 |
| **`item_ids`** | **`["1994478369"]`** | **`["2554056026"]`** | ⛔**불일치 — 이 한 칸이 전부다** |

`1994478369` = **`#W5199551`(pending)** 의 *Wristwatch* $2,025.51
`2554056026` = **`#W9389413`(delivered)** 의 *Smart Watch* $367.38
⇒ **주문은 맞고 품목만 다른 주문에서 왔다.** 전형적 cross-order item 바인딩 오류.

---

## §3. 궤적 per-step — 결정된 **정확한 지점**은 msg 12 다

### 3-a. 문맥에 필요한 값이 실재했는가 → **전부 실재했다**

`[8]` 에서 모델이 **스스로** 3주문을 병렬로 읽었다(gold read 3건 전량 충족):

```
[8] assistant  TC get_order_details {"order_id":"#W5199551"}
               TC get_order_details {"order_id":"#W8665881"}
               TC get_order_details {"order_id":"#W9389413"}
```

`[11]` `#W9389413` 축자(발췌):
```
"items": [ … {"name": "Smart Watch", "product_id":"6945232052", "item_id":"2554056026",
              "price":367.38, "options":{"color":"gold","band material":"metal","display":"AMOLED"}}, … ],
"status": "delivered",
"fulfillments": [{"tracking_id":["967556659983"],
                  "item_ids":["4127323219","2554056026","6342039236","5047954489"]}]
```
`[9]` `#W5199551` 축자(발췌): `{"name":"Wristwatch", "item_id":"1994478369", "price":2025.51}` · `"status":"pending"`

⇒ **손님이 "받은" 시계는 `status=delivered` 인 `#W9389413` 의 Smart Watch 하나뿐이다.**
판별에 필요한 것(이름·소속 주문·status·fulfillment 목록)이 **전부 turn 12 문맥에 있었다.**
읽기 결손 0 · discovery 결손 0 · 정보 부족 0.

### 3-b. ★**결정점 = msg 12** — 사용자 발화 이전에 이미 틀렸다

msg 12 의 `raw_data.choices[0].message.tool_calls` 축자(= 재생성 **전** 원본 모델 출력):

```json
{"name":"cancel_pending_order","arguments":"{\"order_id\": \"#W5199551\", \"reason\": \"no longer needed\"}"}
{"name":"cancel_pending_order","arguments":"{\"order_id\": \"#W8665881\", \"reason\": \"no longer needed\"}"}
{"name":"return_delivered_order_items",
 "arguments":"{\"order_id\": \"#W9389413\", \"item_ids\": [\"1994478369\"], \"payment_method_id\": \"paypal_5364164\"}"}
```

★**이것이 확정 사실이다**: 오답 `1994478369` 은 **turn 12 의 첫 write 시도**에 이미 들어 있었다.
user-sim 이 그 id 를 말한 것은 **turn 13**, 즉 **한 턴 뒤**다. ⇒ **user_sim 유도가 원인이 아니다**([[21]]).
모델이 손님 발화의 *"watch"* 를 **이름 표면**(`Wristwatch`)으로 매칭해 `#W5199551` 에서 집어 왔고,
`order_id` 만 delivered 주문으로 갈아 끼웠다.

### 3-c. G2 가 그 오답을 **손님에게 공표**로 바꿨다

turn 12 의 세 호출은 `G2_CONFIRM_WRITE` 로 전부 차단됐고, 재생성이 도구호출을 지우고 산문을 남겼다.
영속된 `[12]` 꼬리 축자:
```
[Note: the tool call(s) above were blocked by a policy gate and were NOT executed. …]
([G2_CONFIRM_WRITE] blocked by policy gate: explicit user confirmation (yes) of the…; ×3)
```
그 산문 본문 축자:
> *"3. **Return Delivered Order Item #W9389413**: … You wish to return the wristwatch with item ID
> `1994478369`. …  Please confirm if you would like to proceed with these actions by saying "yes"."*

⇒ 게이트 자체는 정책에 충실하다(retail policy 축자: *"you must list the action details and obtain
explicit user confirmation (yes) to proceed"*). 문제는 **공표되는 details 가 어떤 레코드에도 근거하지
않았는데 아무도 검사하지 않았다**는 것이다. 이 확인 루프가 모델의 추측을 **손님의 지시**로 세탁했다.

### 3-d. user-sim 이 그 오답을 되돌려줬다 (원인 아님 · 증폭기)

`[13]` 축자:
> *"For the delivered order, I only want to **return the wristwatch item (ID 1994478369)** and keep everything else."*

시나리오 `known_info` 는 *"You are Fatima Johnson in zipcode 78712"* 뿐이고 item id 는 없다.
⇒ user-sim 은 **에이전트가 방금 준 id 를 앵무새로 돌려준 것**이다. [[21]] 규율대로 면책 사유가 아니다 —
근인은 12 턴의 자기-정박이고, user-sim 은 그것을 **되돌릴 수 없게 굳혔다**.

### 3-e. `[14]` 의 read 는 판별에 **쓸모가 없는 read** 였다

```
[14] TC get_item_details {"item_id":"1994478369"}
[15] {"item_id":"1994478369","options":{"strap material":"silicone","dial color":"black"},
      "available":true,"price":2025.51}
```
`get_item_details` 출력에는 **소속 주문 필드가 없다.** 모델은 "확인했다"는 느낌만 얻고 오귀속을 강화했다.
(가격 2,025.51 은 delivered 주문 총액 918.95 보다 **크다** — 이 모순도 넘어갔다.)

### 3-f. ★`G6_SELECT_CONFIRM` 이 **발화했고 · 모델이 명시적으로 뒤집었다**

`[16]` 첫 문장 축자:
> *"**It appears there was a policy gate disambiguation check for the order IDs you provided.**
> **However, based on your previous confirmation**, the orders to be cancelled are #W5199551 and
> #W8665881, and the wristwatch item ID to be returned is `1994478369` from order #W9389413."*

**"disambiguation" 문자열은 retail 선언 전체에서 딱 한 곳에만 있다** —
`a2/retail.gate.json:170` = `a2/retail.settings.json:150` = `G6_SELECT_CONFIRM.message`
(엔진 폴백은 `gate_interpreter.py:329`). ⇒ **모델이 G6 문면을 보고 그것을 서술한 것**이다.
형제 보고서 `TASK_9.md §4-b` 가 같은 가설을 **PLAUSIBLE** 로 남긴 자리인데, sim 16 은
**모델 자신의 축자 서술**이라는 궤적-내 증거가 있어 **CONFIRMED** 로 올라간다.

**토큰 회계 보강** (`usage.prompt_tokens`):

| turn | prompt_tokens | Δ | 커밋된 증분 |
|---|---|---|---|
| 14 | 7,700 | — | — |
| 16 | **9,614** | **+1,914** | `[14]` 호출 31tok + `[15]` 결과 ≈45tok = **≈76** |

잔차 **≈1,838**. 이 sim 의 G6 문면을 `_present_candidates`(`gate_interpreter.py:306-331`) 로
재구성하면 **3,290자**(3주문 × `present_fields=[status,address,items]` 전문). `TASK_9.md` 가 같은 런에서
잰 비율(2,631자 ≈ 1,184tok = 2.22자/tok)로 환산하면 **≈1,482tok**, 나머지 ≈356 은 차단된
어시스턴트 메시지(동일 3호출 · turn 12 에서 166tok)와 deny 주석 · 도구 stub 로 설명된다. **크기가 맞는다.**

★그리고 그 3,290자 안에는 `#W9389413` 의 `items` 가 **전문으로** 들어 있었다 —
`item_id` 4개(`4127323219`·`2554056026`·`6342039236`·`5047954489`) 중 `1994478369` 은 **없다**.
**정답 재료가 눈앞에 배달됐는데 모델이 "However, based on your previous confirmation" 으로 밀어냈다.**

### 3-g. ★G6 는 **틀린 것을 묻지 않았다** — 범위 불일치

G6 문면 축자(`a2/retail.gate.json:170`):
> *"…verify **the order_id** you are about to use is the one they mean (disambiguate by shipping
> address, item contents/count, status). If it is wrong, call the write again with **the correct order_id**."*

에이전트의 `order_id` 는 **이미 옳았다**. 틀린 칸은 `item_ids` 였다.
⇒ G6 는 **이미 맞은 필드를 재확인하라 하고, 틀린 필드에 대해서는 한 마디도 하지 않았다.**
선언 키 `present_label:"order"` · `detail_id_arg:"order_id"`(`a2/retail.gate.json:161-172`)가
그 범위를 고정한다. 모델이 "확인은 이미 받았다"로 응수한 것은 **문면 그대로 읽으면 논리적으로 옳다.**

### 3-h. env 거부 → 회복 없이 포기

```
[19] tool  Error: Some item not found          (error=True)
[20] assistant  "…there was an issue as the item was not found. … the total refund … is
                $3131.10 + $4777.75 = $7908.85. … I will transfer you to a human agent…"
                [Note: … blocked by policy gate …] ([G4_TRANSFER_MSG] …)
[22] TC transfer_to_human_agents(summary=…)   → [23] "Transfer successful"
[25] user  ###STOP###
```
`Error: Some item not found` 는 *"이 id 는 이 주문에 없다"* 를 사실상 말해 주는 신호였다.
그러나 회복 힌트가 붙지 않았고(§4 `T2_TOOLERR`), 모델은 **재조회 0회**로 포기했다.
`[20]` 산술 `3131.10 + 4777.75 = 7908.85` 는 **정확**하다 — 산수 능력은 병목이 아니었다.

---

## §4. 레버 발화표 (이 sim)

⛔**관측 한계 선언**: 로그 · 사이드카 미회수 ⇒ **stderr 마커는 한 건도 못 봤다.** 아래 판정 근거는
⑴궤적 커밋 문자열 ⑵`go_stack.sh`/`run_t7391_retail.sh` 의 export 유무 ⑶선언 키 유무 ⑷오프라인 재현, 넷뿐이다.
검색 경로: `grep -oE "\[[A-Z][A-Z0-9_ -]{2,40}\]"` 전 sim · `grep -n "<FLAG>" go_stack.sh run_t7391_retail.sh` ·
`GI.load_domain_a2('retail').keys()`.

### 4-a. 궤적에서 실측된 마커 (전 12 sim)

```
task 16 : {'[G2_CONFIRM_WRITE]': 3, '[G4_TRANSFER_MSG]': 1}
런 전체 : G2_CONFIRM_WRITE 24 · G4_TRANSFER_MSG 2 · DUPLICATE-READ 3 · 그 외 0
런 전체 : 'DISAMBIGUATION NOTE' 0/12 · 'COMPUTED FACTS' 0/12   (대조군은 10/12 · 10/12)
```

### 4-b. 지시받은 레버 목록의 판정

| 레버 | 판정 | 근거 |
|---|---|---|
| `T2_SG_DOCS` | **미발화 — 재료 부재** | `go_stack.sh:100` ON. 그러나 술어가 요구하는 A3 `isolate.docs`/`require_doc_before` 가 **retail A2 에 없다**(retail 키 37 ↔ banking 118) |
| `T2_PIN_READ` | **미발화(무해)** | `go_stack.sh:409` ON. 이 sim 은 read 결손 0 — 고정할 잔여 read 가 없다 |
| `T2_DEMANDED_STEP`/`T2_PROCEDURE_LEFT` | **미발화 — 재료 부재** | `a2["procedures"]` 가 retail 에 **없다**(`t2_gate_patch.py:8779` 진입 조건) |
| `T2_CLAIMPROV` | **미발화 — 재료 부재** | `a2["claim_prov"]` 부재(`t2_gate_patch.py:13939`) |
| `T2_FOLLOWUP_*` | **미발화 — 재료 부재** | `a2["follow_up_chains"]` 부재(`t2_gate_patch.py:12878`) |
| `T2_SEARCH_AGENT` / `T2_SEARCH_REARM` | **미발화(정상)** | retail 에 KB 문서 표면 자체가 없다. `T2_SEARCH_REARM` 은 `_search_material` 하위 스위치 |
| `T2_FAB_STRIP` | **미발화(무해)** | `go_stack.sh:217` ON. 이 sim 에 날조 문면 없음 |
| `T2_ARG_PRODUCERS` | **미발화 — 재료 부재** | `go_stack.sh:276` ON. `a2["arg_producers"]` 는 retail 에 **없다**(`t2_prekb_patch.py:277`) |
| READ-FIRST 계열 | **미발화(무해)** | 모델이 스스로 3주문을 다 읽었다 |
| `T2_REQUIRE_DOC_DELIVER` | **미발화 — 재료 부재** | `go_stack.sh:497` ON. `a2["require_doc_before"]` 부재(`t2_gate_patch.py:3915,9091`) |
| **`G2_CONFIRM_WRITE`** | **발화 3회 · 중립→유해(증폭)** | §3-c. 정책 충실하나 **미근거 details 를 공표로 승격**시켰다 |
| **`G6_SELECT_CONFIRM`** | **발화 1회 · 무시됨(범위 불일치)** | §3-f/§3-g. 모델 축자 서술 + 토큰 회계 +1,914 |
| `G4_TRANSFER_MSG` | **발화 1회 · 무해** | notice 순서 강제. 재발행으로 통과(`[22]`), 채점 무관 |
| `G1/G3/G5/G7/G_EXHAUST` | **미발화(정상)** | 인증 완료 · 소유자 일치 · status 정합(delivered→return) · G7 은 `return_*` 미포함 |

### 4-c. ★**꺼져 있던** 레버 — 이 실패에 직접 걸리는 것들

| 플래그 | 엔진 실재 | `go_stack.sh` | 이 sim 에 걸리는가 |
|---|---|---|---|
| **`T2_CONSISTENCY`** (L10 멤버십) | `t2_gate_patch.py:8605` | **미export** | ★**예 — §6-ⓐ 에서 재현** |
| `T2_TOOLERR` | `t2_gate_patch.py:8329` | **미export** | 예(부차 · §6-ⓒ) |
| `T2_PRESENT_READS` | `t2_gate_patch.py:1096,7345` | **미export** | 예(단 폐기됨 · §5) |
| `T2_CALC` | `t2_gate_patch.py:1262` | **미export** | 부차(NL 축) |
| `T2_PRESENT_NESTED` | `t2_gate_patch.py:1253` | **미export** | 부차 |
| `T2_NLNUM_PROV` | `t2_gate_patch.py:12844` | **미export** | 무효(7,908.85 는 이미 정확) |

---

## §5. 선행 판정과의 대조

| 선행 | 이 sim 과의 관계 |
|---|---|
| `RETAIL_FULL_FAIL_CENSUS_2026_07_11.md §1` — task 16 은 **FLAKY 2/4** · 버킷 `WRONG_ITEMS 27` · *"disamb-도달(문맥에 gold·오답 공존) 63"* | **같은 클래스다.** 이번 실패도 정확히 *문맥에 gold 와 오답이 공존하는데 오답을 집은* 형태 |
| 같은 문서 **§4 t95.0** 축자 — *"둘째 laptop이 **다른 주문**임을 미발견 → 같은 주문에 재시도 → env 거부 → 포기"* → *"deny↔transfer 100% 상관 = **impasse 표지**"* | **동형이 그대로 재현됐다.** 우리 sim: watch 가 다른 주문 것임을 미발견 → env 거부(`Some item not found`) → 포기 → transfer. 선행의 결론(*"게이트는 옳게 차단 · 진짜 레버는 상류"*)도 **유지된다** — 단 §6-ⓑ 는 **G6 가 상류가 아니라 "틀린 것을 묻는 하류"** 라는 새 칸을 더한다 |
| 같은 문서 §3 레버표 **행 4 calc** 후속 정정 — *"실패는 **relay-gap**"* | 이 sim 의 NL 실패는 relay 도 calc 도 아니다. **DB 실패의 하류**다(§6-ⓔ) |
| `LEVER_ROSTER_CANONICAL_2026_08_19.md:191` 축자 — *"`T2_CONSISTENCY`… 414 로그 전수 0. 켠 .sh 3개(generalized_stack_v4/5/6)뿐이고 **go_stack 에 없음**"* | **이미 카탈로그된 결손이다.** 이 보고서는 그 결손의 **reward 로 환산된 첫 실물 사례**를 제공한다 |
| `DROPPED_LEVER_PER_TASK_MAP_2026_08_19.md:234-238` — `T2_L10` 멤버십 판정 **UNATTRIBUTABLE**, *"로그 미영속이라 `T2_CONS`·`membership` 문자열이 궤적에서 0건이고 사후 귀속이 원천 불가능"* | 이 sim 은 **로그 없이도** 귀속이 된다 — 술어가 순수함수라 **궤적만으로 오프라인 재현**되기 때문이다(§6-ⓐ) |
| `LEVER_ROSTER_CANONICAL_2026_08_19.md:131` — `T2_PRESENT_READS` 폐기 사유 축자: *"엔진이 대신 `detail_producer` 호출 = **규칙 0 위반**"* · *"present 는 frontier 격차의 83%를 차지하는 실패를 스스로 제조한다"* | **따른다.** §7 처방에서 `T2_PRESENT_READS` 부활은 **제안하지 않는다**. 대신 §6-ⓑ 가 지적하듯 **G6 의 `_present_candidates` 가 같은 대리 호출을 write 시점에 하고 있다** — 폐기 사유가 그대로 살아 있다 |
| 형제 `tasks__20260829/TASK_1.md §4` · `TASK_4.md ⓑ` · `TASK_9.md §4-b` | 세 형제는 **`T2_PRESENT_READS` 부재 / G6 축**으로 수렴했다. **본 보고서는 그와 다른 축**을 낸다 — 멤버십 술어(§6-ⓐ)는 **대리 호출이 0** 이라 규칙 0 반론에 걸리지 않는다. `TASK_9 §4-b` 의 G6-발화 가설은 sim 16 에서 **CONFIRMED 로 승급**된다(§3-f) |

---

## §6. 원인 확정 ([[77]] 4칸)

### ⓐ **CONFIRMED · our_layer (주)** — 멤버십 술어가 `return_delivered_order_items` 를 **덮지 않는다**

**⑴주장+양화**: sim 16 trial 0, 결정점 msg 12(및 재시도 msg 16). n=1/1 trial. 런 12 sim 전수 write 스캔에서
이 술어가 걸리는 호출은 **정확히 1건**(= 이 실패). 전칭 아님.

**⑵근거 — 코드 경로 두 갈래가 둘 다 닫혀 있다**

- **갈래 A (현행 정본 경로 · ON)** — `T2_RESOLVE=1`(`go_stack.sh:67`) → `t2_gate_patch.py:9607`
  `_rz.resolve_write(...)` → `t2_resolve.py:1288` 축자:
  ```python
  ops = ((a2 or {}).get("operands") or {}).get(tool) or {}
  ```
  retail `operands` 선언 키 = `a2/retail.settings.json:229`(정본 · [[24]]) / `a2/retail.gate.json:436`(미러).
  **선언된 도구 3개** = `modify_pending_order_items` · `exchange_delivered_order_items` ·
  `modify_pending_order_address`. ⇒ `return_delivered_order_items` 는 **`ops = {}`** 이고
  루프가 한 번도 안 돌아 `{"status":"ok"}` 로 통과한다. **item_ids 를 가진 write 3종 중 이 하나만 빠졌다.**
- **갈래 B (구 경로 · OFF)** — `t2_gate_patch.py:8605` 축자 `if os.environ.get("T2_CONSISTENCY") == "1"`.
  이 경로는 `a2["eplan"]`(`a2/retail.gate.json:396`) + `_confirm_write_tools(a2)`(`t2_gate_patch.py:3121`)
  를 쓰므로 **`return_delivered_order_items` 를 덮는다**(실측: 반환 7종에 포함).
  그러나 `go_stack.sh` 에 **export 0건**(`grep -c "^export .*T2_CONSISTENCY=" go_stack.sh` → 0 ·
  `run_t7391_retail.sh` 도 0 · 켜는 곳은 `generalized_stack_v{4,5,6}.sh:31` 셋뿐).
- **왜 좁아졌나 — 통합이 술어를 좁혔다**: `t2_gate_patch.py:9498-9500` 주석 축자 —
  *"★T2_RESOLVE (통일 인터프리터…): deny-kind(operator/**membership**/provenance) 통합 = L10+L3+operator
  한 경로. **개별 플래그(T2_CONSISTENCY/T2_PROV_ORIGIN) 대체용**"*.
  ⇒ 대체본(`operands` 열거)이 원본(`eplan` + confirm-write 전량)보다 **좁은데 그대로 대체했다.**
  [[67]] *"사본을 접을 때 좁히지 마라"* 의 실물.

**⑶오프라인 재현 (정본 술어 · 이 sim 자신의 문맥 · 프롬프트 저작 0)**

```python
import gate_interpreter as GI, t2_gate_patch as G
a2 = GI.load_domain_a2('retail')
sp = {'entity_key':'order_id', 'items_key':'item_ids', 'items_id_path':['items','item_id']}   # = a2['eplan']
msgs = sim['messages'][:12]                      # ← turn 12 시점 문맥 그대로
G.membership_violation({'order_id':'#W9389413','item_ids':['1994478369'],
                        'payment_method_id':'paypal_5364164'}, sp, msgs)
```
```
cut 12 -> (['1994478369'], '#W9389413', '#W5199551')
cut 16 -> (['1994478369'], '#W9389413', '#W5199551')
cut 20 -> (['1994478369'], '#W9389413', '#W5199551')
gold args(['2554056026']) -> None                      ← ★부정통제 통과(오차단 0)
```
발화됐을 문면(`t2_gate_patch.py:2743` `CONS_MEMBER_FEEDBACK` 축자 포맷):
> *"[CONSISTENCY] item(s) **1994478369** do not belong to order_id='**#W9389413**' according to its
> latest fetched details. **They appear in order_id='#W5199551'.** Re-check which record actually
> contains the item(s) the user means, then re-emit a corrected call."*

⇒ **틀린 칸을 정확히 지목하고, 어디를 보면 풀리는지까지 말한다**([[64]] 이행). 정답은 말하지 않는다
(선택은 여전히 모델 몫 ⇒ [[62]] 위반 아님).
그리고 `_record_for`(`t2_gate_patch.py:2644-2652`)는 **`state.messages` 만** 읽는다 —
**엔진 대리 도구 호출 0** ⇒ C34 규칙 0/[[03b]] 반론에 걸리지 않는다(§5 의 `T2_PRESENT_READS` 와 결정적 차이).

**⑷반증 조건 / refut**: `fc0055dc` 에서 **이 한 칸만** 되살린 팔(`operands` 에
`return_delivered_order_items` 추가, 또는 `T2_CONSISTENCY=1`)에서 sim 16 이 **여전히 `1994478369` 로
커밋되면 이 주장은 거짓이다.** 또한 그 팔에서 **다른 태스크의 정상 write 가 오차단되면** 순매수 주장이
무너진다(현재 런 12 sim 전수 스캔 결과 오차단 후보 **0건**).

**⑸선행 확인(grep 경로)**: `grep -rn "T2_CONSISTENCY" --include=*.py --include=*.sh --include=*.json .` ·
`grep -n "T2_CONSISTENCY\|T2_L10\|멤버십" reports/facet_rft_2026/LEVER_ROSTER_CANONICAL_2026_08_19.md
reports/facet_rft_2026/DROPPED_LEVER_PER_TASK_MAP_2026_08_19.md` ·
`grep -rn "membership_violation\|resolve_write\|_confirm_write_tools" --include=*.py .` ·
`ls reports/facet_rft_2026/tasks_*` + 형제 4편 정독 ⇒ **이 축(멤버십)을 다룬 선행 보고서는 검색 범위
(`tasks__20260829/*.md` 4편 · `reports/facet_rft_2026/*.md` grep 2회)에서 0건**(형제는 전부 present/G6 축).

---

### ⓑ **CONFIRMED · our_layer (부차)** — `G6_SELECT_CONFIRM` 이 **틀린 필드를 묻고 · 늦게 물었다**

**⑴주장+양화**: sim 16, msg 15~16 사이 **1회** 발화. 사 온 것 **0**, 판 것 **≈1,482 tok 문맥**.

**⑵근거**: ⓐ모델 축자 *"It appears there was a policy gate **disambiguation** check for **the order IDs**
you provided. **However, based on your previous confirmation** …"*(`[16]`) —
`disambiguation` 은 retail 선언 전체에서 `a2/retail.gate.json:170` 하나뿐(`grep -in "disambiguat" a2/retail.*.json`
→ 2건이고 둘은 정본/미러 동일 문자열). ⓑ토큰 회계 +1,914 ↔ 커밋 증분 ≈76.
ⓒ선언 범위: `detail_id_arg:"order_id"` · `present_label:"order"` · 문면이 **order_id 만** 재확인 요구
(`a2/retail.gate.json:161-172` = `a2/retail.settings.json:130-152`). 에이전트의 order_id 는 **옳았다**.
ⓓ우선순위 `_KIND_PRIORITY`(`gate_interpreter.py:21-22`) `confirm=3 < select_confirm=5` ⇒ turn 12 에서는
G2 가 선점해 G6 가 **평가조차 안 됐고**, 확인을 받은 turn 16 에야 나왔다. `state.presented_select`
(`gate_interpreter.py:233,435`)로 **sim 당 1회** ⇒ 재시도 기회도 없다.

**⑶반증 조건 / refut**: 회수된 `bank_t7391_retail_20260829.log` 의 `[sim=16#…]` 줄에서 turn 15~16 게이트
id 가 `G6_SELECT_CONFIRM` 이 **아니면 거짓**. 또는 G6 문면을 `item_ids` 까지 포함하도록 넓힌 팔에서
sim 16 이 **여전히 실패하면** "범위 불일치가 병목"이라는 부분이 **거짓이 된다**(그때 남는 원인은 ⓓ).

**⑷선행 확인**: `tasks__20260829/TASK_9.md:231-264`(같은 가설 PLAUSIBLE) · `TASK_1.md:150,183-185` ·
`grep -rn "select_confirm\|presented_select" gate_interpreter.py t2_gate_patch.py`.

**⑸부수 관측(처방 아님)**: `test_assembled_run.py:83-95` 가 박아 둔 config 불변식 —
*"present 는 read-증강(candidate_summary)만 · **select_confirm deny-경로는 replay 깸([[06]] 폐기)**"* ·
검사 항목 *"드라이버 `T2_GATE_KINDS` 에 select_confirm 없음"* — 이 검사는 **`reexp_assembled.sh` 에만**
걸려 있고(`if os.path.exists(_drv)`), **정본 런처 `go_stack.sh` 는 검사 대상이 아니다**
(`grep -n "T2_GATE_KINDS" go_stack.sh` → 0건). ⇒ 하드 제약이 **틀린 파일에 걸려 있다**([[07]] 동형).
형제 `TASK_1.md §5` 가 이미 지적한 축과 같다.

---

### ⓒ **CONFIRMED · our_layer (경미)** — `tool_error_specs` 도 같은 도구를 빠뜨렸다

**⑴주장+양화**: sim 16 msg 19, 1건.
**⑵근거**: `a2/retail.specific.json:164`(정본) / `a2/retail.gate.json:407`(미러) 축자:
```json
{"applies_to": ["exchange_delivered_order_items", "modify_pending_order_items"],
 "match": "not found or available|item not found|variant not found",
 "class": "recover",
 "hint": "The item/variant id does not exist for this product/order. Re-read the correct id from
          get_product_details / get_order_details and use it."}
```
`[19] "Error: Some item not found"` 는 이 정규식에 **매칭된다**. 그런데 `applies_to` 에
`return_delivered_order_items` 가 **없다** — ⓐ와 **정확히 같은 누락 패턴**(형제 두 도구는 있고 이것만 없다).
게다가 소비자 `T2_TOOLERR`(`t2_gate_patch.py:8329`)도 `go_stack.sh` 에 **미export** ⇒ **이중으로 죽었다.**
**⑶반증 조건 / refut**: `T2_TOOLERR=1` + `applies_to` 확장 팔에서 `[19]` 뒤에 힌트가 붙어도 모델이 재조회
없이 transfer 하면 **이 칸의 기여는 0 = 주장 거짓**.
**⑷선행 확인**: `grep -rn "tool_error_specs\|T2_TOOLERR" --include=*.py --include=*.json .` · `test_toolerr.py`.

---

### ⓓ **CONFIRMED · model** — 근인은 이름-표면 매칭에 의한 cross-order 바인딩

**⑴주장+양화**: sim 16 turn 12, 오답 1건. **정보 부족 아님**(§3-a: 판별 재료 전부 문맥 실재 · 읽기 결손 0).
**⑵근거**: 손님 발화는 *"a watch I already received"* — 술어는 *received(=delivered)* 다.
모델은 `status` 를 무시하고 이름 *"Wristwatch"* 로 `#W5199551`(pending)에서 집은 뒤 `order_id` 만
`#W9389413` 으로 바꿔 **한 호출 안에서 두 레코드를 섞었다**. `[16]` 축자 *"the wristwatch item ID to be
returned is `1994478369` **from order #W9389413**"* 는 문맥 `[9]`/`[11]` 과 **직접 모순**한다.
[[63]] 관점: *닫힌 술어로 후보 제거*(= `item ∈ order.items`)를 스스로 하지 못했다.
**⑶반증 조건 / refut**: ⓐ의 멤버십 문면을 받고도 같은 id 를 재발행하면, 결손은 "지적을 못 받아서"가 아니라
더 깊은 곳(모델의 선택 자체)이므로 **"지적만 있으면 산다"는 함의가 거짓이 된다** ⇒ ⓐ의 처방 가치도 떨어진다.
**⑷선행 확인**: `RETAIL_FULL_FAIL_CENSUS_2026_07_11.md` §2-D/§2-E · `NEXT_DET_LEVERS_DESIGN_2026_06_27.md:71-80`
(형제 `TASK_9.md:319` 가 인용한 축자 — *"후보를 다 보여줘도 모델이 잘못 고른다"*).

---

### ⓔ **하류 · 독립 결손 아님** — NL/COMMUNICATE 축

**⑴주장+양화**: sim 16 NL 축 1건, DB 축의 하류.
**⑵근거**: `8276.23 = 7908.85 + 367.38`. `[20]` 에서 모델은 `3131.10 + 4777.75 = 7908.85` 를 **정확히** 냈다
(2항 산술 성공). 대조군도 `calculate` 를 **한 번도 부르지 않고** 머릿속으로 8,276.23 을 냈다
(refPASS `[18]` 축자: *"the total refund amount … is $8276.23"*). ⇒ **산술 능력이 병목이 아니다.**
367.38 이 빠진 유일한 이유는 그 품목의 반품이 실패했기 때문이다.
**⑶반증 조건 / refut**: ⓐ 처방 팔에서 DB 가 통과했는데도 NL 이 8,276.23 을 못 내면 **별개 결손이 존재**하고
이 "하류" 판정은 거짓이 된다.
**⑷선행 확인**: `RETAIL_FULL_FAIL_CENSUS_2026_07_11.md` §2-G(NL_ONLY 13 sims) — 그 클래스와는 **다르다**
(그쪽은 db 통과 후 NL 실패). gold `16_5` `calculate` 미호출은 `reward_basis` 에 ACTION 이 없어 **채점과 무관**.

---

### ⓕ **user_sim** — 증폭기이지 원인 아님

**⑴주장+양화**: sim 16 msg 13, 1건.
**⑵근거**: `[13]` 이 `1994478369` 을 지목했으나 그 id 는 **에이전트가 `[12]` 에서 먼저 발명한 것**이고
시나리오 `known_info`(*"You are Fatima Johnson in zipcode 78712"*)에는 없다(task 정의 축자 확인).
[[21]] 규율상 종결 카테고리 금지.
**⑶반증 조건 / refut**: `[12]` 이전 어느 메시지에서든 `1994478369` 이 손님 측에서 먼저 나오면 이 판정은 거짓
(전수 확인: 궤적 첫 등장은 msg 12 assistant `raw_data` 다).
**⑷선행 확인**: [[21]] 메모리 · `RETAIL_FULL_FAIL_CENSUS_2026_07_11.md §4`(사용자 요청 transfer 사례 t8.0 대조).
**부수 관측**: 이 앵무새 응답이 `[16]` 의 *"based on your previous confirmation"* 이라는 **G6 무시의 명분**을
만들었다 — 우리 층이 미근거 details 를 공표시키지 않았다면 이 명분도 없었다(§3-c).

---

### ⓖ **env** — 무죄

`Error: Some item not found` 는 **정확한 거부**다. `F.deny_kind` → `'env'`. env 오도 0건.
**⑶반증 조건 / refut**: `#W9389413` 에 `1994478369` 이 실재하는데 env 가 거부했다면 거짓 —
`[11]` 의 `items[].item_id` 4개에 없음이 축자 확인된다.

---

## §7. 처방 후보 (제안만 · 실행/코드 수정 0)

| # | 처방 | 층 | 근거 | ±([[70]]) · 위험 |
|---|---|---|---|---|
| **P1** | `a2/retail.settings.json` `operands` 에 `return_delivered_order_items.item_ids = {kind:"membership", entity_key:"order_id", items_key:"item_ids", items_id_path:["items","item_id"]}` 추가 → `a2/retail.gate.json` 에 **주석까지 바이트 동일** 미러([[24]]) | our_layer | §6-ⓐ 오프라인 재현 + 부정통제 통과 | **엔진 수정 0**(형제 두 도구가 이미 쓰는 술어) · 대리 호출 0 ⇒ 규칙 0 무관 · 런 12 sim 전수 오차단 후보 **0건** · **산 것 1 / 판 것 0** |
| **P2** | `tool_error_specs[0].applies_to` 에 `return_delivered_order_items` 추가 + `T2_TOOLERR` 를 `go_stack.sh` 에 등재 | our_layer | §6-ⓒ | 저비용 · 단 P1 이 서면 `[19]` 자체가 안 일어나므로 **후순위** |
| **P3** | G6 문면/범위를 `order_id` **너머**로: `present_label`·`detail_id_arg` 가 고정한 축 하나만 묻는 구조를 재검토 | our_layer | §6-ⓑ | ⚠**대리 호출을 함께 없애지 않으면** `T2_PRESENT_READS` 폐기 사유(C34 규칙 0)를 그대로 계승한다. P1 이 서면 이 자리는 **불필요**해질 수 있으므로 P1 측정 후 판단 |
| **P4** | `test_assembled_run.py:83-95` 의 config 불변식 대상에 **`go_stack.sh` 추가** | our_layer | §6-ⓑ⑸ · [[07]] | 검사 대상이 6월 드라이버 하나뿐 ⇒ 정본 런처가 불변식 밖에 있다. **형제 `TASK_1.md` 와 중복 처방**이므로 통합 등재 |
| **P5** | `t2_forensic.mutating_tools()` 기본값 `banking_knowledge` 를 호출부 필수 인자로 승격하거나, 결과 파일의 `environment_info.domain_name` 에서 자동 선택 | our_layer(계기) | §2-a | ⚠**계기 결함이다** — retail 런 포렌식이 조용히 `clean=True` 를 낸다. [[25]] *"우리 도구는 100% 정답 의무"* |

**측정 설계(P1 단독 팔)**: 같은 sha `fc0055dc` · `run_t7391_retail.sh` 그대로 · **A2 한 칸만** 토글.
종점 = **reward**([[69]]). 태스크별 부호표 의무 — retail 114 중 `return_delivered_order_items` 가
gold 인 태스크 전량 + 오차단 감시로 `modify_pending_order_items`/`exchange_delivered_order_items`
태스크 전량. 부정통제 = 같은 길이의 무내용 문면([[57]]).

---

## §8. 검색 경로 기록 ([[74]] / [[77]]⑷)

```
find C:\workspace\ba-frft -iname "*7391*"
ls reports/facet_rft_2026/sim_results | grep -iE "7391|reg12|log|^(fb|trace)_"
ls reports/facet_rft_2026 | grep -iE "retail|t739|reg12|tasks_"
grep -rln "W9389413|fatima_johnson_7581|2554056026|1994478369" reports/facet_rft_2026
grep -rln "reg12" --include=*.md reports/facet_rft_2026
grep -n "PRESENT_READS|DISAMBIGUATION|T2_CONSISTENCY|CONS_MEMBER|membership|G6_SELECT" tasks__20260829/TASK_{1,3,4,9}.md
grep -rn "T2_CONSISTENCY|T2_TOOLERR|T2_DISAMB|T2_CALC|T2_NLNUM_PROV|T2_PRESENT_NESTED" --include=*.py --include=*.sh --include=*.json scripts/distill/tau2
grep -n "membership|member_of|disjoint|equal_len|_confirm_write_tools|resolve_write|candidate_summary" scripts/distill/tau2/{t2_gate_patch,t2_resolve,gate_interpreter}.py
grep -n "T2_GATE_KINDS|PRESENT_READS|CONSISTENCY|TOOLERR" scripts/distill/tau2/{go_stack.sh,run_t7391_retail.sh,reexp_assembled.sh}
grep -n "T2_PRESENT_READS|T2_CONSISTENCY|T2_L10" reports/facet_rft_2026/{LEVER_ROSTER_CANONICAL_2026_08_19,DROPPED_LEVER_PER_TASK_MAP_2026_08_19,RETAIL_FULL_FAIL_CENSUS_2026_07_11}.md
```

**⛔이 보고서가 하지 않은 것**: 리모트 접속 0 · git 커밋/push 0 · 코드 수정 0 · A2 수정 0 · 모델 호출 0.
**⛔말할 수 없는 것**: 로그 · 사이드카 미회수 ⇒ 재생성으로 지워진 우리-층 반려(`regen_blocked`)는
`sidecar='unknown'` = **모른다**([[25]]). §6-ⓑ 의 반증 조건이 회수된 로그를 지목하는 이유다.
