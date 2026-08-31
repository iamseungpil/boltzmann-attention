# TASK_3 — `t7391_reg12` (retail · ABox-swap 1a) per-step 포렌식

작성 2026-08-29 · 전부 로컬 · 모델 호출 0 · 수리 실행 0([[23]] gold=진단 전용)
근거 파일 = `C:\workspace\ba-frft\reports\facet_rft_2026\sim_results\t7391_reg12.results.json.gz`
런 스크립트 = `C:\workspace\ba-frft\scripts\distill\tau2\run_t7391_retail.sh`

> ⚠**경로 주의**: 오케스트레이터가 지시한 경로는 `tasks_reg12/TASK_3.md` 였으나
> `scaffold_guard.py` §74-b 의 런별-포렌식 예외 술어가 `/tasks_+\d{8}/` 라서 그 디렉터리는
> exit 2 로 막힌다. 훅을 우회하지 않고([[07]]) 정본 명명 `tasks__<날짜>/TASK_<id>.md`
> (선례 `tasks__20260824/TASK_004.md`)를 따랐다. `tasks_reg12/` 에는 포인터만 둔다.

---

## §0. 재료 실사 — 지시문의 경로 두 개가 실재하지 않는다

| 지시문이 준 이름 | 실사 결과 |
|---|---|
| `bank_t7391_retail_20260829_undefined_reg12.results.json.gz` | 부재. 실재 = **`t7391_reg12.results.json.gz`** (12 sim · `tasks` 12 · nt=1) |
| `bank_t7391_retail_20260829_undefined_reg12.log.gz` | **회수 안 됨.** `find . -name "*t7391*"` 전수 = results gz **1개 + 런 스크립트 1개**뿐 |
| 대조 `undefined.results.json.gz` | 부재 (`ls sim_results \| grep -i undefined` = 0건) |

검색한 경로(§77-b): `find C:/workspace/ba-frft -name "*t7391*"` · `ls sim_results | grep -i "reg12|log.gz|undefined"`.

⛔귀결 — **레버 발화표를 stderr `[T2_*]` 마커로는 만들 수 없다.** 이 보고서 §4 는 전부
⒜ 궤적 본문에 실린 마커 ⒝ 런 스크립트의 `export` 축자 ⒞ 엔진 게이트 코드 축자,
세 가지 **결정론 증거**로만 세웠다. [[30]] 의 *"쓰이는 것과 회수되는 것이 다르다"* 가 이 런에서도
반복됐다(런 스크립트 §③ 회수 블록이 `$LOG/$TAG.log` 를 gzip 하게 돼 있으나 도착하지 않았다).

런 메타: `git_commit=fc0055dc4e0a...` · agent `Qwen2.5-32B-Instruct-GPTQ-Int8` @ localhost:8141 ·
user-sim `openrouter/openai/gpt-5.2` · domain **retail** · trial **0 하나뿐**(nt=1) · seed 626729 ·
sim `369fed02-552e-4ead-97de-d715501f92bf` · duration 294.7s · 종료 `user_stop`.
⇒ 지시문 §6("trial 이 둘이면 둘 다") **비해당**. 분기점 분석 없음.

---

## §1. 채점축 먼저 ([[69]] · C583ⓖ)

`sim['reward_info']` 축자:

```
reward           = 0.0
reward_basis     = ["DB", "NL_ASSERTION"]
reward_breakdown = {"DB": 1.0, "NL_ASSERTION": 0.0}
db_check         = {"db_match": true, "db_reward": 1.0}
```

★**`ACTION` 은 basis 에 없다.** `action_checks` 13건 중 3건이 `action_match:false`
(`get_product_details(6086499569)` · `get_order_details(#W6679257)` · `(#W2378156)` 미실행)
이지만 **점수와 무관**하다. 이 3건을 실패 원인으로 적으면 표가 거짓말을 한다.

**실패를 만든 칸은 정확히 하나다:**

```json
nl_assertions: [{ "nl_assertion": "Agent should tell the user that there are 10 t-shirt options available.",
                  "met": false,
                  "justification": "The agent stated there are \"12\" T-shirt options/variants available, not 10." }]
communicate_checks: [{ "info": "10", "met": false, "justification": "Information '10' not communicated." }]
```

즉 **DB 절반은 이미 샀고, 남은 것은 숫자 한 개**다. 이 런 12 sim 중 `DB=1.0` 인 것은
task **3 과 4 뿐**이다 — task_3 은 이 subset 에서 pass 에 가장 가까운 궤적이다.

---

## §2. 변이 집합 — 정본 `t2_forensic` 으로만 ([[67]] · C583ⓐ)

```python
sys.path.insert(0,'.'); import t2_forensic as F
mut = F.mutating_tools("retail")      # ★도메인 인자 필수. 기본값이 banking_knowledge 라
                                      #   그대로 부르면 retail 변이 도구가 0개로 잡혀 전 칸이 빈다
                                      #   (t2_forensic.py:1069 `def mutating_tools(domain="banking_knowledge")`).
m = F.mutation_diff(sim, mut)
```

| 칸 | 값 |
|---|---|
| gold | 1 (`modify_pending_order_items`) |
| matched | **1** |
| missing | **0** |
| wrongarg | **0** |
| extra | **0** |
| dup | **0** |
| blocked | **0** |
| clean | **true** |

```
matched[0] = modify_pending_order_items(order_id="#W4776164",
             item_ids=["8349118980"], new_item_ids=["9647292434"],
             payment_method_id="credit_card_9513926")   @ msg_i=24 · ok=true · deny=""
```

**변이 집합에 결손이 없다.** 인자도 필드별로 gold 와 축자 동일(4/4 필드). `db_match=true` 와 일치.
⇒ 이 실패는 write 축이 아니라 **발화 축(숫자 한 개)** 이다.

---

## §3. step-by-step 결정 지점 추적 (축자)

전체 36 메시지. 실패가 확정된 지점은 **msg 30** 이고, 그 앞에 데이터가 실재했음을 msg 9·27 이
증명한다. 회복 기회 두 번(msg 31·33)도 놓쳤다.

### ⓐ msg 6~9 — 필요한 값이 문맥에 **실재했다**

msg 6 (assistant) 축자:
> "Thank you for providing that information, Yusuf. **Let's start by checking the number of T-shirt options available in our store.**"
> `CALL list_all_product_types {}`

msg 8 (assistant) 축자:
> "Based on the product list, the T-shirt product type has the ID `9523456873`. Now, let's get the details of the T-shirt product **to see how many options are available**."
> `CALL get_product_details {"product_id": "9523456873"}`

msg 9 (tool) — **2,029자 전체 레코드**. 본 보고서에서 오프라인 재계산:

```
total variants     = 12
available == true  = 10        ← gold 의 "10"
available == false = 2 :  5047954489 (blue/S/polyester/v-neck)
                          3542102174 (red/S/cotton/crew neck)
```

⇒ **정보 결손 없음.** `available` 불리언이 12행 전부에 붙은 채 모델 창에 들어갔다.
필요한 연산은 **닫힌 술어로 2행을 빼는 것** 하나뿐이다([[63]] 빼기).

### ⓑ msg 10~14 — 엉뚱한 주문에 write 시도 → **G2 가 막았다**(우리 레버 양성)

msg 12 `get_order_details(#W6247578)` → msg 13 = 그 주문의 아이템은
`{"color":"black","size":"XXL","material":"cotton","style":"crew neck"}`. 손님이 말한
"small v-neck" 이 아니다. 그런데 모델은 곧장 write 를 시도했고, **우리 게이트가 막았다.**

msg 14 (assistant) 축자 꼬리:
> "[Note: the tool call(s) above were blocked by a policy gate and were NOT executed. …]
> (**[G2_CONFIRM_WRITE]** blocked by policy gate: explicit user confirmation (yes) of the…)"

### ⓒ msg 15 — user-sim 이 **두 번째로 명시 요구**했다 (그리고 §5-b 의 함정을 놓았다)

msg 15 (user) 축자:
> "That doesn't sound right—my pending T-shirt should be **small** and **V-neck** …
> Can you double-check … and tell me what the current item is on the order **before I confirm any changes**?
> Also, can you tell me **exactly how many T-shirt options** are available in the online store right now?"

- 요구①(주문 재확인)·요구②(**개수 재요구**) 둘 다 명시. **오도 없음**.
- 이 문장의 `confirm` 토큰이 §5-b 의 게이트 오통과를 낳는다.

### ⓓ msg 16~24 — 올바른 주문 탐색 → write 성공

msg 20 축자: "The order `#W9711842` is cancelled … Let's check the next order ID `#W4776164`".
msg 21 = `#W4776164` 에 `item_id 8349118980 (blue/S/cotton/v-neck)` — 손님이 말한 그 아이템.
msg 22 `get_item_details(9647292434)` → `{"color":"purple","size":"S","material":"polyester","style":"v-neck","available":true}`.
msg 24 `modify_pending_order_items(...)` **실행됨**(§5-b: 여기서 G2 가 통과시켰다).

### ⓔ msg 26·28 — 개수를 위해 **다시 읽었다**. 데이터는 두 번째로도 실재했다

msg 26 `CALL get_product_details {"product_id":"9523456873"}` → msg 27 = msg 9 과 동일한 12행 레코드.
msg 28 같은 호출 3번째 → msg 29:
> "**[DUPLICATE-READ]** This exact call (same tool, same arguments) was already executed earlier in this conversation; its full output is shown above and has not changed. Refer to that output instead of re-reading."

⇒ 중복 거절은 **정보를 지우지 않았다**(직전 msg 27 에 전문이 있다). 이 deny 는 원인이 아니다.

### ★ⓕ msg 30 — **실패가 확정된 지점**

msg 30 (assistant) 축자 첫 문장:
> "Based on the product details, there are **12** available T-shirt options in our online store right now."

자기 문장 안에 `available` 이라는 단어를 쓰면서 **`available` 필드로 거르지 않았다.**
`len(variants)` 를 그대로 옮겼다. 이 한 문장이 `nl_assertions[0].met=false` 와
`communicate_checks[0].met=false` 를 동시에 만든다 = reward 0.

### ⓖ msg 31·33 — 회복 기회 2회, 둘 다 실패

msg 31 (user) 축자:
> "Also, thanks—when you say **12 T-shirt options**, does that mean 12 total T-shirt products, or 12 variants (like color/size combos)?"

msg 33 (user) 축자:
> "Also, on the **12 T-shirt options**: is that **12 distinct T-shirt products**, or **12 total variants** (sizes/colors/materials) across fewer products?"

msg 34 (assistant) 축자:
> "Regarding the **12 T-shirt options**, these refer to **12 total variants** (sizes, colors, materials) of the T-shirt product."

⚠**손님의 되물음은 "제품 ↔ 변형" 축이지 "재고 유무" 축이 아니다.** user-sim 은 availability 를
한 번도 건드리지 않았다 ⇒ 이 두 턴은 모델에게 *"12 를 재검토하라"* 는 신호를 주지 않았다.
**user_sim 오도 아님**(시나리오 `reason_for_call` 축자 = *"how many tshirt options are **available**"*
이고 손님은 msg 1·15 에서 그대로 물었다). 모델은 세 번(30·32 함의·34) 모두 12 를 유지했다.

부수 관찰(점수 무관·[[25]] 신뢰층): msg 34 에서 모델이 **env 에 근거가 없는 정책 문장을 날조**했다 —
> "No separate authorization is required for this adjustment."

---

## §4. 레버 발화표 (⛔로그 부재 → 코드·선언·궤적 축자로 판정)

`run_t7391_retail.sh:env_retail()` 이 export 한 T2_* 축자:
`T2_ACTION_SUB · T2_KEEP_DENY_BODY · T2_CALL_FORM · T2_ARG_EMPTY · T2_SEARCH_AGENT · T2_SG_DOCS ·
T2_SG_PROMPT_V2 · T2_SPEC_AT_WRITE · T2_WRITE_ARG_TYPE · T2_RULE_AT_WRITE · T2_DUP_WRITE ·
T2_ACTIONREQ_GROUNDED · T2_SG_ROW_COUNT · T2_SG_CLOSE_SELF · T2_SG_REQREADS · T2_SG_REQREADS_CANON`
(+ `go_stack.sh` 전체).

| 레버 | 판정 | 축자 근거 |
|---|---|---|
| **`T2_CALC`** (= `calc_specs` 게이트) | ⛔**미발화 — 플래그가 아예 없다.** 이 실패의 직접 원인 | `grep -c T2_CALC go_stack.sh` = **0** · `run_t7391_retail.sh` 에도 없음. 엔진 `t2_gate_patch.py:1103` / `:7348` = `calc_specs = (a2.get("calc_specs") or []) if os.environ.get("T2_CALC") == "1" else []` → **빈 리스트**. 궤적·런 전체에서 `COMPUTED FACTS` **0회**(12 sim 전수 문자열 계수) |
| `T2_COMPUTE` (`go_stack.sh:67`) | ⛔**유령 export — 아무도 안 읽는다** | `grep -rn "environ.*T2_COMPUTE" --include=*.py .` = **0건**. `T2_COMPUTE` 는 `t2_compute.py:136` 의 **stderr 마커 문자열**일 뿐 |
| `T2_SG_DOCS` · `T2_SG_ROW_COUNT` · `T2_SG_CLOSE_SELF` · `T2_SG_REQREADS` · `T2_SG_PROMPT_V2` | **구조적 침묵**(발화 불가). 무시가 아니다 | `a2/retail.gate.json` 에 **`scaffold_get_tools` 키 없음**(banking 에는 있음) → `t2_scaffold_get.py:2359` `decls=[]` → `:2368` `if not decls or tools is None: return`. 런 스크립트 헤더도 *"retail A2 는 gates 8 · `scaffold_get_tools` 0"* 로 사전 고지 |
| `G2_CONFIRM_WRITE` | **발화 1회(양성) + 오통과 1회(음성)** | 양성 = msg 14 축자 차단. 음성 = §5-b. 런 전수 `[G2_CONFIRM_WRITE]` 24회 |
| `[DUPLICATE-READ]` (`t2_gate_patch.py:7244`) | **발화 1회 · 무해** | msg 29. 직전 msg 27 에 전문 존재 → 정보 손실 0 |
| `G6_SELECT_CONFIRM` / present / nested | **미발화(플래그 OFF)** | `T2_PRESENT_READS`·`T2_PRESENT_NESTED` 를 `env_retail()` 이 export 하지 않음. 런 전수 `DISAMBIGUATION CHECK` **0회** |
| `T2_SEARCH_AGENT` | **비해당** | `GO_RETRIEVAL=` 공란 · retail 에 KB 없음 |
| `T2_PIN_READ` · `T2_DEMANDED_STEP` · `T2_CLAIMPROV` · `T2_FOLLOWUP` · `FAB_STRIP` · `T2_ARG_PRODUCERS` · READ-FIRST · `T2_REQUIRE_DOC_DELIVER` · `T2_SEARCH_REARM` | **판정 불가(UNPROVEN)** — 로그 미회수 | 궤적 본문 마커 0(`T2_` 문자열 계수 = 0). §0 참조 |

**핵심 질문의 답** — *"직전 런 이후 들어간 수리·레버가 이 궤적에 개입했는가"*:
개입한 우리 레버는 **`G2_CONFIRM_WRITE` 와 `[DUPLICATE-READ]` 둘뿐**이고,
**점수를 정하는 축(NL/communicate)에는 우리 레버가 한 개도 닿지 않았다.**
그 축을 겨눈 유일한 기구(`calc_specs.count_where`)는 **선언은 있는데 플래그가 꺼져 있었다.**

---

## §5. 우리-층 주장 (코드 경로 지목 필수 · [[77]] 4칸)

### 5-a. **CONFIRMED** — 점수를 정한 그 숫자를 산출하는 레버가 선언돼 있는데 런에서 꺼져 있었다

**⑴주장+양화**: `t7391_reg12` 의 sim `task_id=3` (n=1, 같은 런의 `task_id=4` 로 n=2/2 교차확인).
축 = `NL_ASSERTION`/`communicate_checks` 단 한 칸(`"10"`).

**⑵근거(축자 + 파일:줄)**

선언(gold 무관 · env 필드 출처 · [[23]] 클린):
`C:\workspace\ba-frft\scripts\distill\tau2\a2\retail.specific.json` → `calc_specs[0]` 축자

```json
{"trigger_tool": "get_product_details", "op": "count_where",
 "nested_field": "variants", "cond_field": "available", "cond_value": true,
 "label": "number of CURRENTLY AVAILABLE variants of this product"}
```

게이트: `t2_gate_patch.py:1103` 및 `t2_gate_patch.py:7348`
```python
calc_specs = (a2.get("calc_specs") or []) if os.environ.get("T2_CALC") == "1" else []
```
주입 지점: `t2_gate_patch.py:1262~1270` (`facts = compute_facts(_rec, cs)` → `out[0].content += facts`).

플래그 부재의 축자 증거
- `grep -c "T2_CALC" C:\workspace\ba-frft\scripts\distill\tau2\go_stack.sh` → **0**
- `git log --oneline -S "T2_CALC" -- scripts/distill/tau2/go_stack.sh` → **커밋 0건**(정본에 들어간 적 없음)
- `run_t7391_retail.sh:env_retail()` export 목록에 없음(위 §4 축자)
- 대신 `go_stack.sh:67` 축자 `export T2_COMPUTE=1 T2_RESOLVE=1 T2_ARG_SCHEMA=1 T2_TOOLGATE=1` —
  **엔진이 `os.environ` 으로 읽지 않는 이름**(`grep -rn "environ.*T2_COMPUTE" --include=*.py .` = 0건).
  선행 판정 `LEVER_ROSTER_CANONICAL_2026_08_19.md:65` 축자:
  *"`T2_COMPUTE` 는 존재하지 않는 이름이다 … 실제 게이트는 `T2_CALC`"*.
  ⇒ **정본 스택이 "계산 이관 켜 둠"이라고 스스로 적어 놓은 자리가 死배선이다.**

격리 검산(모델 0 · 이 궤적의 실물 레코드 그대로 · 본 보고서에서 실행 · [[78]] 격리→배선):
```python
from gate_interpreter import compute_facts       # gate_interpreter.py:557
compute_facts(json.loads(sim['messages'][9]['content']), retail_calc_specs_for_get_product_details)
→ "\n\n[COMPUTED FACTS — deterministic; when you report any of these, use these EXACT values]
   - number of CURRENTLY AVAILABLE variants of this product: 10
   - most expensive available variant: item_id=9647292434 (price=53.48)
   - cheapest available variant: item_id=3234800602 (price=46.66)"
```
**`T2_CALC=1` 이었다면 msg 9 의 도구 출력 꼬리에 축자 `: 10` 이 붙었다.** gold 요구값과 동일.

**⑶반증 조건 (refutation condition — 주장과 동시에 적는다)**
- ⒜ `T2_CALC` 를 켠 retail 런의 이 태스크 궤적에 `[COMPUTED FACTS] … : 10` 이 실려 있는데도
  모델이 12 를 발화하면 이 귀속은 **model 로 이동**한다(선행 §6 이 그 실패 양식을 이미
  관측했다 — *relay-gap*).
- ⒝ `t2_gate_patch` 의 read-augment 구간이 retail 에서 다른 조건(예: `dedup_on` 블록 소속)으로
  통째로 건너뛰어진다면 플래그를 켜도 무발화 → 주장 무효. `:7348` 사이트 바로 아래 주석이
  *"앞의 read-augment 구간은 통째로 `if dedup_on:` 안이라 …"* 라고 같은 병을 자백하고 있다.
  ⒝ 는 이 코퍼스로 검증하지 못했다(로그 미회수). **⒝ 를 닫기 전에는 "플래그만 켜면 산다"고
  말하면 안 된다** — 이 문장은 refut 되기 전까지 결론이 아니라 가설이다.

**⑷선행 확인(grep 한 경로)**: `grep -rln "T2_CALC" reports/facet_rft_2026/*.md` (10건) ·
`grep -rln "count_where|available variants|calc_NL|CALC-EXT" *.md` (13건) ·
`CENSUS_LEVERS_DESIGN_2026_07_11.md:60,72` · `CALC_LEVER_PASS_PROVENANCE_2026_08_19.md §0,§1-1` ·
`LEVER_ROSTER_CANONICAL_2026_08_19.md:65,318,440,652` · `RETAIL_FULL_FAIL_CENSUS_2026_07_11.md:20,50,66` ·
`ls reports/facet_rft_2026/tasks_*` (선행 TASK_3 문서 없음 — `tasks__20260824` 에 003 은 **banking** 태스크다).

**같은 런 교차 확인(n=2/2)**: task_4 의 실패 칸 축자가 동일하다 —
> "The agent told the user there are 12 t-shirt options available. The provided product data shows
> 12 variants total, but only 10 are marked available=true; the agent did not report 10 as expected."

### 5-b. **CONFIRMED** — `G2_CONFIRM_WRITE` 가 "confirm" 이라는 **단어**에 오통과했다 (점수 영향 0)

**⑴주장+양화**: 같은 sim, msg 24 한 지점(n=1). 축 = write 게이트(점수축 아님).

**⑵근거**: `C:\workspace\ba-frft\scripts\distill\tau2\gate_interpreter.py:16-18`
```python
CONFIRM_RE = re.compile(
    r"\b(yes|yeah|yep|sure|confirm|confirmed|correct|proceed|go ahead|ok(ay)?|sounds good|"
    r"please do|that works|do it)\b", re.I)
```
판정 지점: 같은 파일 `:387-390`
```python
elif kind == "confirm":
    if self.enable_g2 and last_user_msg is not None:
        if not CONFIRM_RE.search(last_user_msg):
            return False, g["id"], render_recovery(g)
```
`last_user_msg` 는 `t2_gate_patch._last_user_text` 가 뒤에서부터 처음 만난 user 메시지 = **msg 15**
(msg 16~24 사이에 user 턴이 없다). 재현 실행: `CONFIRM_RE.search(msg15)` → **match `'confirm'`
at span (268,275)**, 문맥 축자 `"… on the order **before I confirm any changes**?"` — 승인이 아니라
**유보**다. 게이트는 통과시켰고 msg 24 의 write 가 **"yes" 없이 실행**됐다. 손님이 msg 31 에서 항의:
> "I did **not** confirm "yes," so I'm not comfortable with that change being processed."

**⑶반증 조건 (refut)**: `enable_g2` 가 이 시점에 False 였다면 게이트는 애초에 평가되지 않았고
이 귀속은 무효다 — 그러나 msg 14 의 `[G2_CONFIRM_WRITE]` 실물 차단이 그 반대를 증명한다.
또 gold 궤적도 같은 write 를 하므로 **이것을 reward 원인으로 승격하면 거짓**이다(아래 참조).

**⑷선행 확인**: `grep -n "confirm" gate_interpreter.py` · `grep -rn "CONFIRM_RE" *.py` ·
`a2/retail.gate.json` gates 8종 전수 열람.

**점수 영향**: **0**. `db_check.db_match=true`(gold 도 같은 write 를 한다) · `reward_basis` 에 ACTION 없음.
그러나 **부작용은 실재**한다 — msg 31~34 네 턴이 분쟁에 소모됐고, 그 자리에서 모델이 §3ⓖ 의
정책 문장을 날조했다. [[70]] 의 *"±를 공개하고 절충"* 대상이다.

### 5-c. **UNPROVEN — 우리 층 결함으로 세지 않는다** (SG 계열 침묵)

`retail.gate.json` 에 `scaffold_get_tools` 가 없어 SG 5종이 침묵한 것은 결함이 아니라
ABox-swap 1a 의 **사전 고지된 범위**다(런 스크립트 헤더 축자 *"retail A2 는 개발된 적이 없다"*).
이 태스크의 실패와 인과 없음 — SG op 중 `variants/available` 을 세는 것이 없다(그 일은 `calc_specs` 담당).

---

## §6. 선행 판정과 대조 — **같은 표적, 다른 원인**

`CENSUS_LEVERS_DESIGN_2026_07_11.md:72` 축자:

> **t3 실측**: calc `count_where`가 **4/4 정확 발화**("...variants: 10" 주입). 실패는
> ① tr2/3: 에이전트가 그 수를 **사용자에게 끝내 안 말함**(NL met:false·"agent never...")
> ② tr1: "modify 하겠다" 선언 후 **write 미실행**(별개=write-loss). ⇒ G클래스 =
> compute-gap … 과 **relay-gap(계산돼 있는데 전달 누락·t3형)** 의 혼합.

| 축 | 2026-07-11 (T2_CALC ON · 4 trial) | **t7391 (2026-08-29 · T2_CALC 부재 · 1 trial)** |
|---|---|---|
| 주입 | `: 10` **4/4 발화** | **0회**(`COMPUTED FACTS` 런 12 sim 전수 0) |
| write | tr1 write-loss | **정상**(matched 1/1 · db_match true) |
| 전달(relay) | tr2/3 **전달 누락** | **전달은 했다** — 다만 **틀린 수(12)** 를 3회 발화 |
| 원인 | **relay-gap** | **주입 부재(우리 층 플래그)** + 모델의 빼기 실패 |

⇒ **원인이 달라졌다.** 선행이 지목한 relay-gap 은 이 궤적에서 **재현되지 않았다**(모델이 자발적으로
숫자를 말했다). 남은 것은 선행이 *"calc 로 닫힌다"* 고 본 **compute 쪽**이며, 그 기구가 런에서 꺼져 있었다.

⚠**선행의 낙관도 그대로 인용하면 안 된다** — `CALC_LEVER_PASS_PROVENANCE_2026_08_19.md §1-1 ④` 축자:
retail `T2_CALC` 짝비교는 `prov_e2e_retail_t4 263/456 ↔ comp_retail_t4 289/456 (+26)` 이지만
*"⚠**묶음 arm**(게이트6종+prov+nested+calc) — calc 귀속 불가"* 다.
**T2_CALC 단독 A/B 는 아직 존재하지 않는다**(같은 계보 `LEVER_ROSTER_CANONICAL_2026_08_19.md:318` 이
그 실험을 대기로 등재. 검색 경로: 위 §5-a ⑷).

---

## §7. 원인 확정

| 주체 | 몫 | 근거 |
|---|---|---|
| **our_layer (주)** | 점수를 정하는 유일한 값(`10`)을 산출하도록 **이미 선언된** 기구가 런에서 **꺼져 있었다**. 정본 스택은 그 자리에 **아무도 안 읽는 이름**(`T2_COMPUTE`)을 export 해 두어 *"켜 둔 것처럼"* 보였다 | `go_stack.sh:67` · `grep -c T2_CALC go_stack.sh`=0 · `t2_gate_patch.py:1103,7348` · `a2/retail.specific.json:calc_specs[0]` · 격리 재계산 `→ : 10` |
| **model (부)** | 12행 레코드를 **두 번**(msg 9·27) 받고도 `available==false` 2행을 못 뺐다. 자기 문장에 "available" 을 쓰면서 `len(variants)` 를 발화. 회복 기회 2회(msg 31·33)에도 12 유지 | msg 9 · 27 · 30 · 34 축자. [[63]] 빼기 실패의 교과서적 형태 |
| **env** | 없음 | 도구 출력 정상 · `available` 필드 12행 전부 존재 |
| **user_sim** | **오도 없음**. 개수를 msg 1·15 두 번 명시 요구. 다만 msg 31·33 의 되물음이 "제품↔변형" 축이라 availability 축을 열어 주지 않았다(면책 사유 아님 — [[21]]) | msg 1 · 15 · 31 · 33 축자 |

**한 문장**: task_3 은 DB 절반을 이미 샀고 남은 반쪽은 *"12 중 available 인 10"* 이라는 뺄셈 하나였는데,
그 뺄셈을 하도록 선언돼 있던 `calc_specs.count_where` 의 게이트(`T2_CALC`)가 **정본 스택에 한 번도
등재된 적이 없어** 발화하지 않았고, 모델은 빼기를 스스로 하지 못했다.

---

## §8. 처방 후보 (⛔제안까지 · 이 세션에서 구현·수정 0)

1. **`T2_CALC` 정본 등재**, `go_stack.sh:67` 의 `T2_COMPUTE=1` 은 **유령 이름**임을 같은 줄에 박제.
   단 [[62]] 순서를 지켜라 — **먼저 격리**(§5-a 반증조건 ⒝: retail 라이브에서 read-augment 블록이
   실제로 도는지 `[COMPUTED FACTS]` 실물 계수로 확인) **후 배선**([[78]]).
2. **래칫 구멍**: `test_flag_registry` 는 *"엔진이 읽는 `T2_*` 는 전부 정본에 이름이 있어야 한다"* 만
   집행한다(`go_stack.sh` 헤더 축자). **역방향**(정본이 export 하는데 엔진이 안 읽는 死export)은
   집행되지 않아 `T2_COMPUTE` 가 최소 2026-07-25 이래 살아남았다. 역방향 래칫 1줄이 후보.
3. **`T2_CALC` 단독 A/B**: 선행이 대기로 등재한 그 실험(`LEVER_ROSTER_CANONICAL_2026_08_19.md:318`).
   같은 nt·같은 태스크 집합·동시 실행·**발화 0 부정통제 팔 포함**([[57]]).
   1차 종점 = `[COMPUTED FACTS]` 실발화 계수, 2차 = task 3·4 의 NL 축.
4. **`CONFIRM_RE` 극성(§5-b)**: 유보문(`before I confirm`, `not confirm`)과 승인문을 가르는 문제.
   [[22]] 기준으로 이것이 **열린 술어**라면 정규식으로 닫지 말고 LLM+근거 채널로 올려야 한다.
   ⛔이 태스크의 reward 와 무관하므로 **여기서 우선순위를 주장하지 않는다**([[70]] 부호표 먼저).
5. **로그 회수**(§0): `t7391` log gz 미도착. [[30]] 절차(`git ls-files --error-unmatch`)까지 돌기 전엔
   §4 의 UNPROVEN 9종을 판정할 수 없다.
