# TASK_4 — `t7391_reg12` (retail · ABox-swap 1a) per-step forensics

작성 2026-08-29 · 전부 로컬 · **모델 호출 0** · 수리 실행 0 · 커밋 0 · gold 는 진단 전용([[23]])
근거 = `C:\workspace\ba-frft\reports\facet_rft_2026\sim_results\t7391_reg12.results.json.gz`
대조군 = `C:\workspace\ba-frft\reports\facet_rft_2026\sim_results\hist_gpt52_reg12_PASS.results.json.gz`
런 스크립트 = `C:\workspace\ba-frft\scripts\distill\tau2\run_t7391_retail.sh`
대조 드라이버 = `C:\workspace\ba-frft\scripts\distill\tau2\reexp_assembled.sh`

> ⚠**산출 경로 변경**: 지시받은 `tasks_reg12/TASK_4.md` 는 `.claude/hooks/scaffold_guard.py:201`
> 의 예외 술어가 `r"/tasks_+\d{8}/"` 라서 exit 2 로 막힌다. 훅을 우회하지 않고([[07]]) 같은 런의
> 형제 보고서(`tasks__20260829/TASK_1.md`·`TASK_3.md`)와 **같은 정본 명명**을 따랐다.
> `tasks_reg12/` 에는 포인터만 둔다.

---

## §0. 재료 실사 — 지시문 경로 3개 중 2개가 실재하지 않는다

| 지시문이 준 이름 | 실사 |
|---|---|
| `bank_t7391_retail_20260829_undefined_reg12.results.json.gz` | 부재. 실재 = **`t7391_reg12.results.json.gz`** |
| 같은 이름 `.log.gz` | **회수 안 됨**. `find C:\workspace\ba-frft -iname "*t7391*"` 전수 = results gz 1 + 런 스크립트 1 + 형제 프로브 1. `fb_*`/`trace_*` 사이드카 **0건** |
| 대조 `undefined.results.json.gz` | 부재. 실제 대조군은 `hist_gpt52_reg12_PASS.results.json.gz` |

검색 경로([[77]]④ · prior-check paths): `find C:\workspace\ba-frft -iname "*t7391*"` ·
`ls sim_results | grep -iE "7391|reg12|undefined"` · `grep -rln "hist_gpt52_reg12_PASS" C:\workspace\ba-frft`.

⛔귀결 — **stderr `[T2_*]` 마커로 레버 발화표를 만들 수 없다.** `F.mutation_diff(..., tag=...)` 의
`sidecar` 칸도 축자 `absent` 다. 아래 §4 는 전부 ⒜궤적 본문 마커 ⒝런 스크립트 export 축자
⒞엔진 코드 축자 ⒟동일-seed 대조군 궤적, 네 가지 결정론 증거로만 세웠다([[30]]).

**런 메타**: `git_commit=fc0055dc4e0a…` · agent `Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8`(로컬 8141,
temp 0, max_tokens 8192) · user-sim `openrouter/openai/gpt-5.2`(temp 0·reasoning low) ·
domain **retail** · `num_trials=1` ⇒ **trial 0 하나뿐** ⇒ 지시문 §6(둘 다 추적·분기점) **비해당**.
seed **626729** · 28 메시지 · 278.3s · 종료 `user_stop`.

---

## §1. 채점축 먼저 ([[69]] · C583ⓖ)

`sim['reward_info']` 축자:

```
reward           = 0.0
reward_basis     = ["DB", "NL_ASSERTION"]
reward_breakdown = {"DB": 1.0, "NL_ASSERTION": 0.0}
db_check         = {"db_match": true, "db_reward": 1.0}
```

★**`ACTION` 은 basis 에 없다.** `action_checks` 13건 중 `action_match:false` 는 1건
(`4_1 get_product_details(product_id="6086499569")`)뿐이고 **점수와 무관**하다.
(부기: `6086499569` 는 이 궤적의 `list_all_product_types` 출력 49종 어디에도 없다 —
gold 주석자의 탐색 흔적으로 보이며 어느 팔도 재현할 수 없다. 실패 원인으로 세면 표가 거짓말을 한다.)

**실패를 만든 칸은 정확히 하나, 숫자 한 개다:**

```json
nl_assertions:      [{"nl_assertion": "Agent should tell the user that there are 10 t-shirt options available.",
                      "met": false,
                      "justification": "The agent told the user there are 12 t-shirt options available.
                                        The provided product data shows 12 variants total, but only 10 are
                                        marked available=true; the agent did not report 10 as expected."}]
communicate_checks: [{"info": "10", "met": false, "justification": "Information '10' not communicated."}]
```

이 런 12 sim 중 `DB=1.0` 인 것은 **task 3 과 4 뿐**이다. task_4 는 **write 축을 전부 사고
NL 한 칸에서 떨어졌다** — 이 subset 에서 pass 에 가장 가까운 두 궤적 중 하나다.

⚠**벤치 결함 표시 확인**: 이 태스크에는 미해결 이슈가 달려 있다 —
`issues[0].title` 축자: *"Task looks for number 10, but user does not ask for a number.
It is happy with 'various' or a description."* (status `open`, author `keshav@sierra.ai`).
**그러나 이 궤적에는 그 면책이 적용되지 않는다** — 손님이 msg 1 과 msg 25 에서 **두 번**
`"exactly how many"` 로 숫자를 명시 요구했다(§3ⓐ·ⓕ 축자). 이슈의 전제가 이 sim 에서 성립하지 않으므로
[[68]] 의 제외 대상으로 승격하지 않는다.

---

## §2. 변이 집합 — 정본 `t2_forensic` 으로만 ([[67]] · C583ⓐ)

```python
sys.path.insert(0,'.'); import t2_forensic as F
mut = F.mutating_tools("retail")     # ★도메인 인자 필수
m   = F.mutation_diff(sim, mut, tag="t7391_reg12")
```

⛔**계기 함정(내가 처음에 밟았다)**: `t2_forensic.py:1069` 축자
`def mutating_tools(domain="banking_knowledge")`. 인자 없이 부르면 retail 변이 도구가
0개로 잡혀 **여섯 칸이 전부 빈 채 `clean=True`** 가 나온다. 그 빈 표를 그대로 실으면
"변이 결손 없음"이 **모르는 것을 아는 것으로** 둔갑한다. 형제 보고서 TASK_3 §2 도 같은 함정을 적었다.

| 칸 | 값 |
|---|---|
| gold | 2 (`modify_pending_order_items` ×2) |
| **matched** | **2** |
| missing / wrongarg / extra / dup / blocked | **0 / 0 / 0 / 0 / 0** |
| clean | **true** |
| sidecar | `absent` · regen_join `None` · regen_blocked 0 ⇒ **BLOCKED 칸은 "모른다"** |

```
matched[0] modify_pending_order_items(order_id="#W6247578", item_ids=["3799046073"],
           new_item_ids=["9647292434"], payment_method_id="credit_card_9513926")  @msg_i=20 ok=true deny=""
matched[1] modify_pending_order_items(order_id="#W4776164", item_ids=["8349118980"],
           new_item_ids=["9647292434"], payment_method_id="credit_card_9513926")  @msg_i=22 ok=true deny=""
```

WRONGARG 필드별 대조: 두 호출 모두 **4/4 필드가 gold 와 축자 동일**. `db_match=true` 와 일치.
⇒ **이 실패는 write 축이 아니라 발화 축(숫자 한 개)이다.**

---

## §3. step-by-step 결정 지점 추적 (축자)

전체 28 메시지. 실패가 확정된 지점은 **msg 26** 이고, 답을 흘린 지점은 그보다 **16 메시지 앞의 msg 10** 이다.

### ⓐ msg 1 — 손님이 **숫자**를 명시 요구 (요구 2건)

> "Hi—I'd like to know **exactly how many t-shirt options are available** in your online store right now.
>
> Also, I have some pending t-shirt orders and I want to change them all to **purple**, **size S**, keep the **same v-neck**, and I'd prefer **polyester** if that's an option."

시나리오 `reason_for_call` 축자와 일치: *"You want to know exactly how many tshirt options are **available**
in the online store right now."* ⇒ **user_sim 오도 없음.**

### ⓑ msg 8~9 — 필요한 값이 문맥에 **실재했다**

msg 8 (assistant) 축자:
> "Thank you for your patience, Yusuf. I've located your account. **Let's start by checking the t-shirt options
> available in our store.** After that, we'll look into modifying your pending orders."
> `CALL get_product_details {"product_id": "9523456873"}`

msg 9 (tool) — **2,029자, 12 variant 전문**. 본 보고서에서 오프라인 재계산(모델 0):

```
total variants     = 12
available == true  = 10        ← gold 의 "10"
available == false = 2 :  5047954489 (blue / S / polyester / v-neck)
                          3542102174 (red  / S / cotton    / crew neck)
```

⇒ **정보 결손 없음.** `available` 불리언이 12행 전부에 붙은 채 창에 들어갔다.
필요한 연산은 **닫힌 술어로 2행을 빼는 것** 하나뿐이다([[63]] 빼기).

### ★ⓒ msg 10 — **답을 흘린 지점**(deferral). 대조군은 바로 여기서 샀다

t7391 msg 10 (assistant) 축자 — **본문이 빈 문자열이다**:

```python
s['messages'][10]['content'] == ''        # 재현 확인
s['messages'][10]['tool_calls'] == [get_user_details(user_id="yusuf_rossi_9620")]
```

직전 턴에 자기가 *"Let's start by checking the t-shirt options"* 라고 선언해 놓고, 그 답을 낼 자리에서
**한 글자도 쓰지 않고 다음 도구로 넘어갔다.** 이 시점 이후 개수 질문은 msg 26 까지 **16 메시지 동안 미해결**로 남는다.

**대조군 같은 자리**(§3-b) — 같은 모델·같은 seed·같은 태스크의 msg 10 축자:
> "**Currently, there are 10 available options for T-shirts in our store.**
> Regarding your second request, I will now locate your pending T-shirt orders …"

### ⓓ msg 12~17 — 5 병렬 read 가 창을 밀어낸다

msg 12 = `get_order_details` **5 병렬**(#W6247578·#W9711842·#W4776164·#W6679257·#W2378156)
→ msg 13~17 = 합계 **4,947자**. `available` 플래그가 실린 msg 9 과 최종 발화(msg 26) 사이에
이 5개 레코드가 통째로 끼어든다.

★대조군은 `get_order_details` 를 **한 번도 부르지 않았다**(호출 계수 재현: ref = `find_user_id_by_name_zip 1 ·
list_all_product_types 1 · get_product_details 1 · get_user_details 1 · modify_pending_order_items 2`).
같은 정보가 `get_user_details` **응답 꼬리 주석**으로 왔기 때문이다 — §3-b.

### ⓔ msg 18~23 — G2 게이트 발화(정당) → write 성공

msg 18 (assistant) 축자 꼬리:
> "[Note: the tool call(s) above were blocked by a policy gate and were NOT executed. Satisfy the gate
> requirement (authenticate / get explicit user confirmation / check the record's status / fix the operation)
> before attempting the action again.] (**[G2_CONFIRM_WRITE]** blocked by policy gate: explicit user
> confirmation (yes) of the…; [G2_CONFIRM_WRITE] …)"

msg 19 (user) 축자:
> "**Yes** — please proceed and change all my pending t-shirt orders to purple, size S, same v-neck,
> preferably polyester. And yes, you can use the Mastercard ending in 2478 to cover any price difference."

격리 재현(모델 0 · `gate_interpreter.CONFIRM_RE`, `gate_interpreter.py:16-18`):

| user 턴 | `CONFIRM_RE.search` |
|---|---|
| msg 1 | `None` |
| msg 5 | `None` |
| **msg 19** | **`Yes` @ span (0,3)** |
| msg 25 | `None` |

⇒ **task_4 에서 G2 는 양쪽 다 옳았다** — msg 18 에서 확인 없이 막았고, msg 19 이후 통과시켰다.
(형제 보고서 TASK_3 §5-b 가 보고한 *"`confirm` 단어에 오통과"* 는 **task_3 에서만** 일어났다.
task_4 에는 그 함정이 없다. 두 태스크를 하나로 묶어 적으면 안 된다.)

msg 20/22 write 2건 실행 → gold 인자와 축자 동일 → `db_match=true`.

### ⓕ msg 25 — 손님이 **두 번째로 숫자를 명시 요구**

> "Thanks for confirming those changes.
> One more thing: can you tell me **exactly how many t-shirt options** are available in the online store
> **right now**? That's all I need."

**오도 없음 · 회피 없음.** 모델에게 두 번째 기회가 명시적으로 주어졌다.

### ★ⓖ msg 26 — **실패가 확정된 지점**

msg 26 (assistant) 축자 첫 문장:
> "**Based on the product details I fetched earlier**, there are currently **12** different t-shirt options
> available in our online store. These include various combinations of colors, sizes, materials, and styles."

세 가지가 동시에 관측된다.
1. **재read 를 하지 않았다** — `get_product_details` 호출 계수 = **1**(msg 8 뿐). gold 는 이 도구를
   3회(`4_1`·`4_3`·`4_10`) 부른다. 모델은 16 메시지 전 기억에서 답했다.
2. **`available` 이라는 단어를 자기 문장에 쓰면서 `available` 필드로 거르지 않았다.** `len(variants)` 를 옮겼다.
3. 이 한 문장이 `nl_assertions[0].met=false` 와 `communicate_checks[0].met=false` 를 **동시에** 만든다 = reward 0.

msg 27 (user) 축자: > "No, that's perfect — thanks. ###STOP###"
⇒ **회복 기회 0.** 손님은 우리 층 결손을 알 수 없으므로 만족하고 끝냈다([[21]] — user_sim 요인으로 종결 금지).

---

## §3-b. 대조군 — **같은 seed·같은 모델·같은 태스크**에서 무엇이 달랐나 (핵심 증거)

`hist_gpt52_reg12_PASS` task 4: reward **1.0** · trial 0 · **seed 626729(t7391 과 동일)** ·
agent `Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8` 동일 · user-sim `gpt-5.2` 동일 · sha `5ebebbe827b4…`(상이).

**msg 9(`get_product_details` 응답)의 꼬리 축자 — 대조군에만 있다:**

```
[COMPUTED FACTS — deterministic; when you report any of these, use these EXACT values]
- number of CURRENTLY AVAILABLE variants of this product: 10
```

**msg 11(`get_user_details` 응답)의 꼬리 축자 — 대조군에만 있다:**

```
[DISAMBIGUATION NOTE — this customer's full order list]
- #W6247578: {"status": "pending", …  "item_id": "3799046073", … "options": {"color":"black","size":"XXL",…}}
- #W9711842: {"status": "cancelled", … }
- #W4776164: {"status": "pending",  … }
…
```

| 축 | 대조군 `hist_…PASS` (sha 5ebebbe8) | **t7391 (sha fc0055dc)** |
|---|---|---|
| msg 9 길이 | **4,270자**(원문 + item_id 복사지시 + COMPUTED FACTS) | **2,029자**(원문뿐) |
| `COMPUTED FACTS` | **1회** (`: 10`) | **0회** — 런 12 sim 전수 0 |
| `DISAMBIGUATION NOTE` | 있음(주문 전문 인라인) | **0회** — 런 전수 0 |
| `get_order_details` 호출 | **0회**(주석으로 이미 받음) | **5회 병렬 · 4,947자** |
| 개수 발화 시점 | **msg 10 즉시** "there are 10 available options" | msg 10 **침묵** → msg 26 에서 **12** |
| reward | **1.0** | 0.0 |

두 팔의 드라이버 축자:

```bash
# reexp_assembled.sh:39-40   (retail 정본 드라이버)
  T2_GATE_KINDS=auth,confirm,ownership,notice,preconditions,constraints \
  T2_PRESENT_READS=1 T2_PRESENT_NESTED=1 T2_CALC=1

# run_t7391_retail.sh:50-59  env_retail()  ← 위 세 플래그가 하나도 없다
  export T2_ACTION_SUB=1 T2_KEEP_DENY_BODY=1 T2_CALL_FORM=1 T2_ARG_EMPTY=1 T2_SEARCH_AGENT=1
  export T2_SG_DOCS=1 T2_SG_PROMPT_V2=1 T2_SPEC_AT_WRITE=1 T2_WRITE_ARG_TYPE=1
  export T2_RULE_AT_WRITE=1 T2_DUP_WRITE=1
  export T2_ACTIONREQ_GROUNDED=1 T2_SG_ROW_COUNT=1 T2_SG_CLOSE_SELF=1
  export T2_SG_REQREADS=1 T2_SG_REQREADS_CANON=1
  export T2_PROMPT_DUMP=1 T2_PROMPT_DUMP_MAX=80000
  export GO_MAX_STEPS=200 GO_CONCURRENCY=1
  export GO_DOMAIN=retail
  export GO_RETRIEVAL=
```

⚠**단일변수가 아니다**: 두 팔은 sha 도 다르고(5ebebbe8 ↔ fc0055dc) 대조군은 `go_stack.sh` 를
source 하지 않는다(레버 149 export 차이). 따라서 **"플래그 3개가 Δreward 를 만들었다"고 말하면 안 된다.**
말할 수 있는 것은 **기전 수준의 축자 사실**뿐이다 — *gold 가 요구한 문자열 `10` 이 대조군에서는
도구 출력에 실려 모델이 그것을 그대로 옮겼고, t7391 에서는 그 문자열이 생성되지 않았다.*

---

## §4. 레버 발화표 (⛔로그 미회수 → 코드·선언·궤적·대조군 축자로만)

| 레버 | 판정 | 축자 근거 |
|---|---|---|
| **`T2_CALC`** (`calc_specs` 게이트) | ⛔**미발화 — 플래그가 아예 없다.** 점수축을 겨눈 **유일한** 기구 | `grep -c T2_CALC go_stack.sh` = **0** · `run_t7391_retail.sh` 에도 없음. `t2_gate_patch.py:7348` 축자 `calc_specs = (a2.get("calc_specs") or []) if os.environ.get("T2_CALC")=="1" else []` → 빈 리스트 → 주입 지점 `:7471-7477` 무발화. 궤적·런 전수 `COMPUTED FACTS` **0회** |
| **`T2_PRESENT_NESTED`** (`present_specs`) | ⛔**미발화 — 플래그 부재** | 같은 파일 `:7352`. 대조군 msg 9 꼬리의 `item_id` 복사지시가 t7391 에 없다 |
| **`T2_PRESENT_READS`** (`candidate_summary`) | ⛔**미발화 — 플래그 부재** | `t2_gate_patch.py:7347` `present_on = os.environ.get("T2_PRESENT_READS")=="1"`. 런 전수 `DISAMBIGUATION` **0회**(대조군 26회). ⇒ msg 12 의 5 병렬 read 4,947자가 이 침묵의 대가다 |
| `T2_COMPUTE` (`go_stack.sh:67`) | ⛔**유령 export — 아무도 안 읽는다** | `grep -rn "environ.*T2_COMPUTE\|getenv.*T2_COMPUTE" --include=*.py .` = **0건**. 선행 정본 `LEVER_ROSTER_CANONICAL_2026_08_19.md:65` 축자: *"**`T2_COMPUTE` 는 존재하지 않는 이름이다** — `grep environ.*T2_COMPUTE` 0건. 실제 게이트는 `T2_CALC`"* ⇒ **정본 스택이 "계산 이관 켜 둠"이라 자칭하는 자리가 死배선이다** |
| `G2_CONFIRM_WRITE` | **발화 1회 · 정당 · 점수 영향 0** | msg 18 차단 축자 + `CONFIRM_RE` 격리 재현(msg 19 `Yes`). 런 전수 24회 |
| `[DUPLICATE-READ]` (`t2_gate_patch.py:7244`, `T2_READ_DEDUP=1` @ `go_stack.sh:236`) | **이 sim 미발화** | 모델이 재read 를 **시도하지 않았다**(`get_product_details` 계수 1). 런 전수 3회는 전부 다른 sim. **이 실패의 원인이 아니다** |
| `G6_SELECT_CONFIRM` · `G1/G3/G5/G7/G_EXHAUST` | **미발화** | `retail.gate.json` 8종 중 이 sim 궤적의 마커 전수 = `['G2_CONFIRM_WRITE']`. 형제 TASK_1 이 보고한 G6 유해 발화는 **task_1 한정** |
| `T2_SG_DOCS` · `T2_SG_ROW_COUNT` · `T2_SG_CLOSE_SELF` · `T2_SG_REQREADS` · `T2_SG_PROMPT_V2` | **구조적 침묵**(무시 아님) | `a2/retail.gate.json` 에 `scaffold_get_tools` 키 **없음**(재현 확인 `False`) → `t2_scaffold_get.py` 조기 return. 런 스크립트 헤더가 *"retail A2 는 gates 8 · `scaffold_get_tools` 0"* 로 사전 고지 |
| `T2_SEARCH_AGENT` / `T2_SEARCH_REARM` | **비해당** | `GO_RETRIEVAL=` 공란 · retail 에 KB 없음 |
| `T2_PIN_READ` · `T2_DEMANDED_STEP` · `T2_CLAIMPROV` · `T2_FOLLOWUP` · `FAB_STRIP` · `T2_ARG_PRODUCERS` · READ-FIRST · `T2_REQUIRE_DOC_DELIVER` | **UNPROVEN** — 로그·사이드카 미회수 | 궤적 본문 `[T2_` 계수 = **0**. 단 `T2_FAB_STRIP=1`(`go_stack.sh:217`)·`T2_PROV_REGEN=1`(`:33`)·`T2_GATE_REGEN=1`(`:26`) 은 **켜져 있었다** ⇒ msg 10 의 빈 본문이 모델의 침묵인지 재생성 교체인지 **이 코퍼스로는 가릴 수 없다**(§6 ⓔ) |

**지시문의 핵심 질문에 대한 답** — *"직전 런 이후 들어간 수리·레버가 이 궤적에 개입했는가"*:
개입한 우리 레버는 **`G2_CONFIRM_WRITE` 하나뿐**이고 그것은 **정당했으며 점수축이 아니다**.
**점수를 정한 축(NL/communicate)에 우리 레버는 한 개도 닿지 않았다.**
그 축을 겨눈 유일한 기구(`calc_specs.count_where`)는 **선언은 살아 있는데 런에서 플래그가 없었다.**

---

## §5. 선행 판정과 대조

| 선행 | 이 궤적과의 관계 |
|---|---|
| `CENSUS_LEVERS_DESIGN_2026_07_11.md:72` — *"t3 실측: calc `count_where` 가 **4/4 정확 발화**… 실패는 에이전트가 그 수를 **끝내 안 말함**(relay-gap)"* | **원인이 달라졌다.** 여기서는 relay 가 아니라 **주입 자체가 0회**다. 그리고 t7391 의 모델은 숫자를 **말하기는 했다**(틀린 12를). relay-gap 재현 안 됨 |
| `CALC_LEVER_PASS_PROVENANCE_2026_08_19.md §1-1 ④` — T2_CALC(retail) OFF `prov_e2e_retail_t4` 263/456 ↔ ON `comp_retail_t4` 289/456 = +26, **다만 묶음 arm 이라 calc 귀속 불가** | 그 문서의 유보가 여기서도 유효하다. 본 보고서는 **Δreward 를 주장하지 않고 기전만 주장한다**. 같은 문서 §1-1 ⑤ 의 **잡음 바닥 ±1** 도 n=1 짝에는 적용할 수 없다 |
| `LEVER_ROSTER_CANONICAL_2026_08_19.md:65` — `T2_COMPUTE` 는 존재하지 않는 이름 | **재확인**(§4). 이름 오류가 8개 런 스크립트와 `go_stack.sh` 에 그대로 살아 있다 |
| 형제 `tasks__20260829/TASK_3.md` §5-a | **동일 결론·독립 재현.** task_3 도 `12 vs 10`. 단 **task_3 은 재read 를 2회 했고 `[DUPLICATE-READ]` 를 받았으며 G2 가 `confirm` 단어에 오통과**했다. **task_4 는 재read 0회 · DUPLICATE-READ 0 · G2 정상.** ⇒ 같은 우리-층 원인, **다른 모델 국면**. 묶어서 하나로 적으면 안 된다 |
| 형제 `tasks__20260829/TASK_1.md` §4·§5 | 같은 런의 **다른 플래그 축**(`T2_PRESENT_READS` 부재로 G6 가 주석 대신 deny 로 나감)에서 같은 진단. 세 보고서가 **같은 한 줄**(`env_retail()` 이 retail 3플래그를 빠뜨림)로 수렴한다 |
| 벤치 이슈 `issue_2b74ee61` | §1 참조. **이 sim 에는 적용 불가**(손님이 두 번 숫자를 물었다) ⇒ [[68]] 제외 승격 안 함 |

---

## §6. 원인 확정 — 4주체 귀속 ([[77]] 4칸 · [[55]] 우리부터)

### ⓐ **CONFIRMED · our_layer** — 점수를 정한 그 숫자를 만드는 기구가 런에서 배선되지 않았다

**⑴주장+양화**: `t7391_reg12` sim `task_id=4`(n=1). 축 = `NL_ASSERTION` / `communicate_checks` 단 한 칸(`"10"`).
같은 런 `task_id=3` 이 동일 양식(n=2/2 · 형제 보고서 독립 확인).

**⑵근거(축자 + 파일:줄)**

선언(출처 = env 필드 `available`, gold 무관 · [[23]] 클린) — `a2/retail.specific.json` `calc_specs[0]`:
```json
{"trigger_tool": "get_product_details", "op": "count_where",
 "nested_field": "variants", "cond_field": "available", "cond_value": true,
 "label": "number of CURRENTLY AVAILABLE variants of this product"}
```
게이트 — `t2_gate_patch.py:7348`(및 사본 `:1103`):
```python
calc_specs = (a2.get("calc_specs") or []) if os.environ.get("T2_CALC") == "1" else []
```
주입 — `t2_gate_patch.py:7471-7477` (`facts = compute_facts(_rec, cs)` → `out.content += facts`).
플래그 부재 — `run_t7391_retail.sh:50-59` export 목록에 `T2_CALC` 없음 · `grep -c T2_CALC go_stack.sh` = 0.

**격리 검산(모델 0 · 이 궤적의 실물 레코드 그대로 · [[78]])** — 본 보고서에서 실행:
```python
from gate_interpreter import compute_facts            # gate_interpreter.py:557
compute_facts(json.loads(sim['messages'][9]['content']), retail_calc_specs_for_get_product_details)
→ "\n\n[COMPUTED FACTS — deterministic; when you report any of these, use these EXACT values]
   - number of CURRENTLY AVAILABLE variants of this product: 10
   - most expensive available variant: item_id=9647292434 (price=53.48)
   - cheapest available variant: item_id=3234800602 (price=46.66)"
```
그리고 **그 문자열이 실제로 붙은 팔이 실재한다** — 대조군 msg 9 꼬리 축자(§3-b)와
**같은 형식**이며, 그 팔의 모델은 다음 턴에 `"there are 10 available options"` 로 **그대로 옮겼다**.

**⑶반증 조건 / refutation condition (주장과 동시에 적는다)**
- ⒜ `T2_CALC=1` 로 돈 retail 런의 task_4 궤적에 `[COMPUTED FACTS] … : 10` 이 실려 있는데도
  모델이 12 를 발화하면 이 귀속은 **model 로 이동**한다(선행이 관측한 *relay-gap*).
  ⚠**대조군은 이 refut 를 완전히 닫지 못한다** — `hist_gpt52_reg12_PASS` 는 **pass 만 골라 모은
  큐레이션 집합**이라 선택 편향이 있다. 닫으려면 `T2_CALC=1` 로 돈 **비큐레이션** retail 런의
  task_4 전수가 필요하다.
- ⒝ `T2_CALC=1` 이어도 retail 에서 주입 구간이 통째로 건너뛰어지면(예: `if dedup_on:` 소속) 이 주장은
  **거짓이 된다**. **이 refut 는 부분적으로 닫혔다**: 라이브 실행 훅은 `_install_regen_exec()`
  (`t2_gate_patch.py:7097`, 설치 `:7482 BaseOrchestrator._execute_tool_calls = exec_augment`)이고
  그 안의 calc 주입 `:7471-7477` 은 `dedup_on` 블록 **밖**의 최상위 `for tc in tool_calls:` 안에 있다.
  (`t2_gate_patch.py:7350` 주석이 자백하는 병은 `apply()` 쪽 사본 `:1262` 에 해당한다.)
  ⛔완전히 닫으려면 라이브 프롬프트 덤프가 필요한데 `T2_PROMPT_DUMP=1` 산출물이 **회수되지 않았다**.

**⑷선행 확인(grep 한 경로)**: `grep -rln "T2_CALC" reports/facet_rft_2026/*.md`(10건 열람) ·
`LEVER_ROSTER_CANONICAL_2026_08_19.md:65` · `CALC_LEVER_PASS_PROVENANCE_2026_08_19.md §0,§1-1` ·
`CENSUS_LEVERS_DESIGN_2026_07_11.md:60,72` · `ls reports/facet_rft_2026/tasks_*` ·
`grep -rln "hist_gpt52_reg12_PASS" C:\workspace\ba-frft` · `grep -rn "environ.*T2_COMPUTE" --include=*.py .`

### ⓑ **CONFIRMED · our_layer (부차)** — `T2_PRESENT_READS` 부재가 창을 5개 레코드만큼 밀었다

**⑴주장+양화**: 같은 sim, msg 12 한 지점(n=1). 축 = 부하(점수축 아님·기여만).

**⑵근거**: `t2_gate_patch.py:7347` `present_on = os.environ.get("T2_PRESENT_READS")=="1"` →
`:7453-7460` `candidate_summary`(`gate_interpreter.py:493`) 무발화. 대조군은 이 주석 덕에
`get_order_details` **0회**, t7391 은 **5회·4,947자**. 그 결과 `available` 플래그와 최종 발화 사이 거리가
대조군 **0 메시지** ↔ t7391 **16 메시지**.

**⑶반증 조건 / refut**: `T2_PRESENT_READS=1` 인 retail 런에서도 모델이 `get_order_details` 를 5회 부르면
이 주장은 **거짓이 된다**. 또한 이 칸은 점수축이 아니므로 ⓐ가 닫히면 자동으로 부차로 남는다.

**⑷선행 확인**: 형제 `TASK_1.md:137,153` 이 같은 코드 경로를 독립 지목 ·
`grep -rn "T2_PRESENT_READS" --include=*.py --include=*.sh .`

⚠**점수 원인으로 승격하지 않는다** — ⓐ만으로 실패가 설명되고, 이것은 가중 요인이다([[70]] ± 공개).

### ⓒ **model** — 닫힌 술어 빼기 실패 + 답변 지연

**⑴주장+양화**: 같은 sim, msg 10(지연)·msg 26(오답) 두 지점(n=1).

**⑵근거**: `available` 12행이 전부 창에 있었고(§3ⓑ) 손님이 두 번 명시 요구했는데(§3ⓐ·ⓕ)
모델은 ⑴ msg 10 에서 본문 `''` 로 답을 **미뤘고** ⑵ msg 26 에서 **재read 없이**
`"Based on the product details I fetched earlier … 12"` 를 냈다.
[[63]] 의 *"모델은 스스로 배제(빼기)를 못한다"* 의 교과서적 재현이다.

**⑶반증 조건 / refut**: 같은 모델이 주입 없이도 이 레코드에서 10 을 내면 거짓이 된다.
⛔**단, 이것을 단독 원인으로 적으면 거짓이다** — 같은 모델·같은 seed 가 주입만 있으면 **10 을 옮겼다**(§3-b).
정확한 진술: **모델은 빼기를 못하고, 우리 층은 빼 준 값을 이번 런에서 전달하지 않았다.**

**⑷선행 확인**: [[63]] `63-subtraction-principle.md` · `CENSUS_LEVERS_DESIGN_2026_07_11.md:72`

### ⓓ **env** — 무관(정보 완전 제공). **user_sim** — 무관(두 번 명시 요구·오도 0·[[21]])

### ⓔ **UNPROVEN — 모른다로 적는다**
- msg 10 의 빈 본문이 모델 침묵인지 `T2_PROV_REGEN`/`T2_FAB_STRIP` 교체 결과인지 **가릴 수 없다**
  (사이드카 `absent` · 로그 미회수). 셋 다 `go_stack.sh:26,33,217` 로 **켜져 있었다**.
  가릴 방법 = 사이드카 회수 후 `regen_blocked` 조회.
- msg 4 의 뜬금없는 `"I'm sorry for the inconvenience."` 는 **우리 문구가 아니다**
  (`grep -rn "sorry for the inconvenience" --include=*.py --include=*.json --include=*.sh .` = **0건**) ⇒ 모델 산출.

---

## §7. 처방 후보 (제안만 · 실행 0 · 코드 수정 0)

1. **런 드라이버의 도메인별 플래그 누락을 배터리로 잡는다.** 현행 `test_assembled_run.py:93-95` 는
   *"드라이버 `T2_CALC=1`"*·*"드라이버 `T2_PRESENT_READS=1`"* 을 검사하지만 대상이
   **`reexp_assembled.sh` 한 파일로 고정**돼 있고, `run_t7391_retail.sh` 의 배터리 25종에는
   이 테스트가 **포함돼 있지도 않다**. ⇒ 검사 대상을 *"이번 런이 실제로 쓰는 드라이버"* 로 바꾸고
   배터리에 넣는 안.
2. **`T2_COMPUTE` 유령 이름 제거 · `T2_CALC` 로 통일**(`go_stack.sh:67` + 런 스크립트 8종).
   [[60]] 은 *"레버는 전부 항상 켠다"* 인데, **읽히지 않는 이름은 켠 것이 아니다.**
3. ⛔**"플래그만 켜면 산다"고 아직 말하면 안 된다** — §6ⓐ 의 refut ⒜(큐레이션 편향)와
   ⒝(프롬프트 덤프 미회수)가 열려 있다. 먼저 `T2_CALC=1` 단일변수 retail 짝(같은 sha·같은 seed·nt≥2)을
   돌리고, `[COMPUTED FACTS]` 실물 발화와 모델 발화를 **짝지어** 세는 것이 순서다([[62]] ①→②→③).
4. **로그·사이드카 회수를 런 절차의 일부로 강제**한다. 이번에도 `T2_PROMPT_DUMP=1` 을 켜고 산출물을
   회수하지 않아 §6ⓔ 두 칸이 UNPROVEN 으로 남았다([[30]] *"쓰이는 것과 회수되는 것이 다르다"*).
