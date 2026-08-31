# TASK_60 — t7391_reg12 (retail) per-step 포렌식

⚠**데이터 파일명 정정**: 태스크 지시문의 `bank_t7391_retail_20260829_undefined_reg12.results.json.gz`
와 `.log.gz` 는 **로컬에 없다**(검색 경로: `find /c/workspace/ba-frft -name "*t7391*"` = 결과 1건 ·
`ls reports/facet_rft_2026/sim_results/ | grep -i "t7391\|undefined"`).
실제 = `sim_results/t7391_reg12.results.json.gz` (`git_commit fc0055dc`·12 sim·`num_trials 1`).
**로그·사이드카·trace 는 미회수** ⇒ `[T2_*]` stderr 마커 계수·`t2_liveness` 는 이 태스크에서 **불가**
(검색 경로: `ls reports/facet_rft_2026/sim_results/*.log.gz`·`find … -name "*t7391*"`).
레버 발화는 **커밋된 메시지 본문에 남은 표지**로만 셌다(§5 · 한계 명시).

⚠**sha 핀 불가 — 인용 규율 명시**([[77]] *"런 인용은 sha 로 고정"*). 런 sha `fc0055dc4e0a…` 는
**로컬 repo 에 없다**(`git cat-file -t fc0055dc4e0a316c3f83133267fbd6faaa770992` → `could not get
object info` · 로컬 HEAD = `0b612169`). ⇒ `git show <런sha>:파일` 로 A2·코드를 못 고정한다.
차선으로 **worktree ↔ HEAD** 를 대조했다: `a2/retail.gate.json` 은 워크트리가 수정 상태지만
(`git status` = `M` · 다른 에이전트 동시 작업) **`gates` 블록과 `present_specs` 는 HEAD 와
바이트 동일**하고 워크트리 변경은 `failure_markers` 등 **추가분뿐**이다. `gate_interpreter.py` ·
`t2_gate_patch.py` · `t2_forensic.py` 는 워크트리 수정 **0**(`git status` 목록에 없음).
⇒ §4·§8-a 의 코드·선언 인용은 **HEAD 기준으로는 확실**하고, 런 sha 기준으로는 **UNPROVEN**.
대조군으로 지시된 `undefined.results.json.gz` 도 같은 검색에서 나오지 않았다. 대신 로컬에 있는
`hist_gpt52_reg12_PASS.results.json.gz`(`5ebebbe8`·**task 60 이 reward 1.0**·같은 32B 에이전트·
**같은 seed 626729**·trial 0)를 **참조**로 썼다. ⚠sha 상이 ⇒ 통제 실험이 아니라 참조다.

---

## 한 줄 요약

`reward 0.0 = DB 0.0 × NL_ASSERTION 0.0` — **두 축이 같은 한 지점에서 함께 죽었다**(독립 실패 아님).

변이 집합 = **MISSING 1 · WRONGARG 1 · EXTRA 0 · DUP 0 · BLOCKED 0**. 4개 인자 중 **`new_item_ids`
한 필드만** 틀렸다 — 보낸 값 `8555936349`($226.49) ↔ gold `6077640618`($242.92). NL 축이 요구한
`$242.92` 도 같은 선택의 그림자다(같은 변이의 두 얼굴).

**결정점 = msg[8] 하나.** 모델이 *"I am now proceeding to make the change."* 라는 **본문과 write
도구호출을 한 메시지에 함께** 내면서 손님 턴을 건너뛰었고, 그 write 가 **게이트를 하나도 안 맞고**
실행됐다. 시나리오 축자 *"**If and only if the agent provides several options**, you want the option
without water resistance."* ⇒ 정답을 고를 **기준 자체가 손님에게서 유도되어야 하는 값**인데,
모델이 선택지를 제시하지 않아 손님은 그 말을 할 기회를 못 얻었다.

**우리 층 결함(재현 100%)**: `G2_CONFIRM_WRITE` 가 이 write 를 **통과**시켰다. 통과시킨 토큰은
손님 **최초 요청**(msg[1])의 `sure` — 축자 *"Please **make sure** the price is the same or lower"* 의
관용구다. 같은 문장은 *"**confirm** that explicitly **before making the change**"* 로 **행동 전
확인을 요구**하고 있었다. **확인을 요구하는 문장 자체가 확인 게이트를 열었다.**
코드 = `gate_interpreter.py:16-18`(`CONFIRM_RE`) + `gate_interpreter.py:387-390`(confirm 분기가
`args` 를 **한 번도 읽지 않는다**) + `t2_gate_patch.py:6938-6944`(`_regen_last_user`).

**⚠단독 충분하지 않다**: 게이트가 막았어도 모델이 **선택지 하나만** 나열했다면 손님은 *"several
options"* 조건이 불성립이라 선호를 발화하지 않는다 ⇒ 여전히 실패. 이 태스크는 **①확인 게이트**와
**②후보 열거** 둘 다 있어야 산다(§8-c 의 반증조건 / refutation 참조).

---

## §1. 채점축 — 먼저 확인했다 (C583ⓖ)

```python
sim['reward_info']['reward_basis']      # ['DB', 'NL_ASSERTION']
sim['reward_info']['reward_breakdown']  # {'DB': 0.0, 'NL_ASSERTION': 0.0}
sim['reward_info']['db_check']          # {'db_match': False, 'db_reward': 0.0}
```

⇒ **ACTION 축이 아니다.** `action_checks` 는 **진단용 보조**일 뿐 점수에 안 들어간다([[69]]).
그래도 두 축이 같은 곳을 가리킨다:

| 축 | 값 | 축자 |
|---|---|---|
| **DB** | 0.0 | `db_match=false` — 주문의 earbuds item 과 refund 금액이 gold DB 와 다르다 |
| **NL_ASSERTION** | 0.0 | 판정문 축자: *"The agent repeatedly stated the blue earbuds price was $226.49 (and processed the modification/refund based on that). The agent never quoted $242.92 as the new earbuds price."* |
| (참고·축 밖) `communicate_checks` | met=false | *"Information '242.92' not communicated."* |
| (참고·축 밖) `action_checks` | 3/4 | `60_0`·`60_1`·`60_2` match=true · **`60_3` match=false** |

종료 = `user_stop` · 메시지 22 · `duration 95.7s` · `hallucination_retries_used 0` · 예외 0.

---

## §2. 변이 집합 — 정본으로만 (`t2_forensic.mutation_diff` · ⛔손 비교기 0)

```python
sys.path.insert(0,'.'); import t2_forensic as F
mut = F.mutating_tools('retail')   # ★도메인 인자 필수 — 기본값 banking 으로 부르면 전부 빈 표가 나온다
m   = F.mutation_diff(sim, mut)
```

⚠**함정 기록**: `F.mutating_tools()` 의 기본값은 `banking_knowledge` 다(`t2_forensic.py:1069`).
그대로 부르면 retail write 가 하나도 변이로 안 잡혀 `clean=True` 라는 **거짓 표**가 나온다.
`'retail'` 을 넘겨야 `{cancel_pending_order, exchange_delivered_order_items,
modify_pending_order_address, modify_pending_order_items, modify_pending_order_payment,
modify_user_address, return_delivered_order_items}` 7종이 나온다.

| 칸 | n | 내용 |
|---|---|---|
| **MISSING** | **1** | `modify_pending_order_items(order_id=#W5061109, item_ids=['3694871183'], `**`new_item_ids=['6077640618']`**`, payment_method_id=paypal_3742148)` (aid `60_3`) |
| **WRONGARG** | **1** | 같은 도구·`msg_i=8`·`ok=true`·`deny=''` — **`new_item_ids=['8555936349']`** |
| EXTRA | 0 | — |
| DUP | 0 | — |
| BLOCKED | 0 | **게이트 차단 0건** |
| MATCHED | 0 | — |

**WRONGARG 필드별 대조**

| 인자 | 보낸 값 | gold | 판정 |
|---|---|---|---|
| `order_id` | `#W5061109` | `#W5061109` | ✅ |
| `item_ids` | `['3694871183']` | `['3694871183']` | ✅ |
| **`new_item_ids`** | **`['8555936349']`** | **`['6077640618']`** | ⛔**유일 오차** |
| `payment_method_id` | `paypal_3742148` | `paypal_3742148` | ✅ |

두 후보의 env 사실(msg[7] 축자):
`8555936349` = `{"color":"blue","battery life":"8 hours","water resistance":"IPX4"}` · avail · **$226.49**
`6077640618` = `{"color":"blue","battery life":"8 hours","water resistance":"not resistant"}` · avail · **$242.92**
⇒ 환불액도 갈린다: 256.67−226.49 = **30.18**(실제) ↔ 256.67−242.92 = **13.75**(gold). DB 해시 불일치.

---

## §3. 궤적 step-by-step — 결정점과 그 직전 몇 턴 (축자)

trial 은 **1개**(`num_trials=1`) ⇒ 두 trial 분기점 항목은 해당 없음. 대신 §6 에서 **대조군과의 분기점**을 잡았다.

### [1] user — 요구가 처음부터 전부 나온다 (축자)

> Hi—I'd like to change the wireless earbuds in my order **W5061109** to a **blue** color instead.
> Please **make sure** the **price is the same or lower** than what I already paid and **confirm that
> explicitly** before making the change.
> My details: **Chen Johnson**, **Houston, TX 77004**. I don't remember the email address used on the order.
> Can you help get this updated **as soon as possible**?

⚠이 문장이 §4 에서 게이트를 여는 문장이다. 의미론적으로는 **행동 전 확인을 요구**하는데
(`before making the change`), 우리 정규식은 `make `**`sure`** 의 `sure` 와 **`confirm`**` that explicitly`
의 `confirm` 을 **승인 토큰**으로 읽는다.

시나리오 원문(`user_scenario.instructions.reason_for_call`) 축자:
> "You want the price to be the same or lower, which you want the agent to verify explicitly.
> **If and only if the agent provides several options, you want the option without water resistance.**"

⇒ 손님은 **선택지를 받기 전에는 그 선호를 말하지 않는다**. 이것이 이 태스크의 재료 구조다.

### [2]–[5] 인증·주문 읽기 — 전부 정확 (gold `60_0`·`60_1` match)

`find_user_id_by_name_zip(Chen, Johnson, 77004)` → `chen_johnson_4204` ·
`get_order_details(#W5061109)` → status `pending` · earbuds `item_id 3694871183` / white / 8h / IPX4 / **$256.67** ·
결제 `paypal_3742148` / 총 $1319.43.

### [6] assistant — **본문 + 도구호출을 한 메시지에** (1차 위반)

> Your order **#W5061109** is currently pending, so we can modify the items. … Let me check if there is
> a blue variant … **Please confirm if you want to keep the same battery life and water resistance
> options for the new Wireless Earbuds.**
> `TOOLCALL get_product_details(product_id=9924732112)`

**물음을 던지고 같은 메시지에서 도구를 불렀다** ⇒ 손님 턴이 열리지 않는다. 이 질문은 **아무도
못 받았다**. 정책 축자(`sim['policy']`):
> "You should at most make one tool call at a time, and **if you take a tool call, you should not
> respond to the user at the same time.** If you respond to the user, you should not make a tool call
> at the same time."

### [7] tool — **필요한 값이 이 시점에 전부 실재했다** (재료 결손 아님)

12개 variant 전문. 손님 조건(blue ∧ available ∧ price ≤ $256.67)으로 **닫힌 뺄셈**을 하면 후보 3:

| item_id | color | battery | water resistance | price | available | |
|---|---|---|---|---|---|---|
| 1646531091 | blue | 6 hours | IPX4 | 232.49 | true | |
| **8555936349** | blue | 8 hours | IPX4 | **226.49** | true | ← 모델이 고른 것 |
| **6077640618** | blue | 8 hours | **not resistant** | **242.92** | true | ← gold |
| (6452271382) | blue | 4 hours | IPX4 | 258.84 | true | ← 원가 초과로 탈락 |
| (2757705742) | blue | 4 hours | IPX7 | 258.97 | **false** | ← 탈락 |

`x615` 실측: `blue ∧ available ∧ price<=256.67 → 3 후보 ['1646531091','6077640618','8555936349']` ·
그중 `water resistance=='not resistant'` → **`['6077640618']` 유일**.
⇒ **마지막 술어 하나(`not resistant`)만 있으면 유일해**다([[63]] 빼기). 그런데 그 술어는 문맥
어디에도 없다 — 손님이 *"선택지를 여러 개 받은 뒤에만"* 발화하는 값이다.
**부족한 것은 정보가 아니라 턴이다.**

### ★[8] assistant — **결정점**. 본문 + write 를 한 메시지에 (2차 위반·여기서 점수가 죽는다)

> I have verified the details of your order **#W5061109**. … the Wireless Earbuds will be changed to a
> blue variant **with the same battery life and water resistance**, priced at **$226.49**, which is
> lower than the original price of $256.67. You will receive a refund of $30.18 …
> **I am now proceeding to make the change.**
> `TOOLCALL modify_pending_order_items(order_id=#W5061109, item_ids=['3694871183'], new_item_ids=['8555936349'], payment_method_id='paypal_3742148')`

관측 사실 셋:
1. **손님의 확인이 궤적 어디에도 없다.** msg[8] 이전 user 턴은 **msg[1] 하나뿐**이고 그것은 요청이다.
2. **선택지를 하나도 제시하지 않았다.** *"the same battery life and water resistance"* 라는 자기 기준으로
   단독 선택했다 — 그 기준은 손님이 준 적 없다(손님은 **색만** 바꿔달라 했다).
3. **게이트가 하나도 안 걸렸다.** msg[9] 는 성공 결과이고, 형제 sim 들에서 보이는
   `[Note: the tool call(s) above were blocked by a policy gate and were NOT executed …]` 꼬리가
   msg[8] 에 **없다**(검색 경로: 22 메시지 전문 substring 검색 0건). `mutation_diff.blocked = []` 와 일치.

정책 축자(`sim['policy']`):
> "**Before taking any action that updates the database (cancel, modify, return, exchange), you must
> list the action details and obtain explicit user confirmation (yes) to proceed.**"

### [9] tool — 되돌릴 수 없는 상태 변경

`status: "pending (item modified)"` · `payment_history` 에 `refund 30.18` 추가. **여기서 DB 축은 끝났다.**

### [10]–[17] — 같은 read 4연타, 4번째만 stub

`get_user_details(chen_johnson_4204)` **4회**(msgs 10·12·14·16). 출력 491자 <
`T2_READ_DEDUP_MIN`(기본 2000) 이라 캐시 dedup 대상 밖 ⇒ 앞의 3회는 그대로 실행됐고,
`T2_READ_DEDUP_LOOP_K`(기본 3) 에 걸린 **4번째**만 stub:

> `[DUPLICATE-READ] This exact call (same tool, same arguments) was already executed earlier in this
> conversation; its full output is shown above and has not changed. Refer to that output instead of re-reading.`

**이것이 이 궤적에서 발화한 유일한 우리-층 개입이다** — 그리고 **결정점 9턴 뒤**다. 선언대로 동작했다
(`t2_gate_patch.py:7126`·`:7321`) ⇒ 결함 아님. 점수 영향 0.

### [18]–[21] — 잘못된 값을 두 번 더 못 박는다

[18] *"the price is $226.49 … refund of $30.18"* → [19] 손님이 되묻는다 *"just to confirm, the updated
blue earbuds are **$226.49** …, correct?"* → [20] *"indeed priced at $226.49"* → [21] `###STOP###`.
NL 판정문이 지적한 *"repeatedly stated"* 가 이 두 턴이다.

⚠**손님은 오도하지 않았다**([[21]]). msg[19] 는 에이전트가 준 숫자를 되읽은 것이고, 시나리오가 준
정보(선호)는 **조건 불성립으로 발화되지 않았을 뿐**이다.

---

## §4. 격리 — `x615_t7391_task60_confirm_iso.py` (모델 0 · 프롬프트 저작 0 · gold 는 채점에만)

```
① A_LIVE  check(modify_pending_order_items, last_user=msg[1]) -> (True, None, None)
   CONFIRM_RE 매치: [('sure',''), ('confirm','')]
   _regen_last_user 가 고른 발화 == msg[1] ? True
② N_NEG1 (토큰 2개만 치환: "make sure the"→"ensure the", "confirm that explicitly"→"state that plainly")
   매치 [] -> (False, 'G2_CONFIRM_WRITE')
   N_NEG2 (무관 발화 "My earbuds are white.") -> (False, 'G2_CONFIRM_WRITE')
```

⇒ **라이브 100% 재현**이고, **낱말 두 개가 통과의 전부**다([[57]] 부정통제 포함). `G2_CONFIRM_WRITE`
는 `applies_to` 에 `modify_pending_order_items` 를 **명시적으로 포함**한다(`a2/retail.gate.json`)
— 적용 대상 밖이라 조용했던 것이 아니라, **적용되고 통과시켰다**.

코드 축자 3줄:

```python
# gate_interpreter.py:16-18
CONFIRM_RE = re.compile(
    r"\b(yes|yeah|yep|sure|confirm|confirmed|correct|proceed|go ahead|ok(ay)?|sounds good|"
    r"please do|that works|do it)\b", re.I)

# gate_interpreter.py:387-390
elif kind == "confirm":
    if self.enable_g2 and last_user_msg is not None:
        if not CONFIRM_RE.search(last_user_msg):
            return False, g["id"], render_recovery(g)
```

**선언 ↔ 구현의 간극**([[22]]): 게이트 **선언 술어**는
`"explicit user confirmation (yes) of **the action details** in the latest user message"` 인데,
구현은 `args` 를 **한 번도 참조하지 않는다**. *"the action details"* 에 대응하는 항이 코드에 없다.

### 팔 A/B/C/D — 두 런 실행 write 전수 · **오차단 비용 공개** ([[70]])

| 팔 | 정의 | TREAT 통과 | TREAT gold 오차단 | CTRL 통과 | CTRL gold 오차단 |
|---|---|---|---|---|---|
| **A** 현행 | `CONFIRM_RE.search(last_user)` | **22/22** | 0 | **19/19** | 0 |
| **B** = `TASK_12 P1` 의 값싼 조작화 | A ∧ (확인 발화 직전 assistant 가 텍스트-전용) | **22/22** | 0 | **19/19** | 0 |
| **C** 신선도 | A ∧ (확인 index > 인자 산출 tool msg index) | 12/22 | **3** (3#24·4#22·54#37) | 15/19 | **4** |
| **D** = C 정제 | 생산자를 **read 결과**로만 한정 | 14/22 | **2** (3#24·54#37) | **18/19** | **1** (22#18) |

**task 60 은 A·B 통과 / C·D 차단.**

★**이 표의 첫 소득은 처방이 아니라 반증(refutation)이다** — **B 가 A 와 완전히 같다**(41 write 전수 동일).
task 60 의 `prevTxt=True` 는 **msg[0] 인사말** *"Hi! How can I help you today?"* 때문이다.
⇒ 형제 보고서 `TASK_12.md §9 P1` 이 제안한 *"직전 assistant 메시지가 있었는가"* 는
**이 값싼 조작화로는 아무것도 막지 않는다**. P1 을 채택하려면 *"그 assistant 텍스트가 **이 write 의
인자를 발화했는가**"* 까지 요구해야 한다(그 강화판이 D 에 가깝고, D 는 CTRL 에서 gold 1건을 판다).

⚠**팔 선택에 gold 를 쓰지 않았다**([[23]]): A~D 는 **정책 축자**(*"list the action details and obtain
explicit user confirmation"*)와 **게이트 선언 술어**에서 유도했고, gold 는 **오차단 비용을 인쇄**하는
데에만 썼다. 그리고 [[69]] 대로 **팔의 진짜 채점은 `reward` 짝 A/B 런**이지 이 표가 아니다 —
여기 표는 *"오차단이 0 이 아니다"* 를 보이는 데까지만 유효하다.

---

## §5. 레버 발화표 — 이 궤적 (⚠로그 미회수 ⇒ **커밋 메시지 본문 표지**로만 관측)

**한계 선언**: stderr 마커(`[T2_*]`)와 `fb` 사이드카가 없다. 형제 `TASK_1.md:189` 가 확립한 대로
**게이트 deny 문구 중 일부는 비커밋 `fb` 버퍼로만 가고 `state.messages` 에 안 남는다** ⇒
아래 **0 은 "미발화"가 아니라 "관측 불가"** 인 칸이 있다. 그 칸은 그렇게 표기했다.

| 레버 | 이 sim | 런 전수(12 sim) | 판정 | 근거 |
|---|---|---|---|---|
| **`G2_CONFIRM_WRITE`** | **0** | **24** | ⛔**발화했어야 하는데 통과** | §4 격리 100% 재현 · `x611b` 표 |
| `[DUPLICATE-READ]`(`T2_READ_DEDUP`) | **1**(msg 17) | 3 | ✅정상(선언대로·K=3) | §3[10]-[17] |
| `G1_AUTH_FIRST` | 0 | 0 | ✅정당(msg[2] 에서 인증 선행) | msg[3] `chen_johnson_4204` |
| `G4_TRANSFER_MSG` | 0 | 2 | 해당 없음(transfer 0회) | — |
| `G3`·`G5`·`G6`·`G7` | 0 | 0 | **관측 불가** — 리졸버 사망 가설(`TASK_12 §5-b`) ↔ 비커밋 fb 가설(`TASK_1:189`) **미해소** | 아래 ⓐ |
| `[OPERAND DISAMBIGUATION`(`T2_PRESENT_NESTED`) | **0** | **0** | **미발화 — 플래그 미수출** | ⓑ |
| `[DISAMBIGUATION NOTE`(`T2_PRESENT_READS`) | 0 | 0 | 미발화 — 플래그 미수출 | ⓑ |
| `[COMPUTED FACTS`(`T2_CALC`) | **0** | **0** | 미발화 — 플래그 미수출 | ⓑ |
| `T2_SG_DOCS`·`T2_REQUIRE_DOC_DELIVER` | 0 | 0 | **해당 없음** — retail 도메인에 KB 문서가 없다 | `T2_KB_DOCS_DIR` = banking 경로(`go_stack.sh:423`) |
| `T2_SEARCH_AGENT`·`T2_SEARCH_REARM` | 0 | 0 | 해당 없음(retail 도구 16종에 search 없음) | `a2/env_surface.json` retail tools |
| `T2_PIN_READ`·`T2_DEMANDED_STEP`·`T2_CLAIMPROV`·`T2_FOLLOWUP`·`FAB_STRIP`·`T2_ARG_PRODUCERS`·READ-FIRST | 0 | — | **관측 불가**(주입 텍스트 0 · stderr 없음) | 22 메시지 전문 정독: 도구 출력 외 삽입 텍스트는 msg[17] 하나뿐 |

**ⓐ `G3/G5/G6/G7` 침묵의 두 가설 — 이 궤적으로는 못 가른다.**
`T2_GATE_KINDS` 는 이 런에서 **미설정**(검색 경로: `grep -n T2_GATE_KINDS go_stack.sh run_t7391_retail.sh` = 0건)
⇒ `t2_gate_patch.py:7777-7781` 에 따라 **전 kind 활성**이다. 그러면 `G6_SELECT_CONFIRM` 은
msg[8] 에서 `presented_select=False` 이고 손님 주문이 **3건**(msg[11] 축자
`["#W5797164","#W5061109","#W3973757"]`)이므로 `_present_candidates` 가 **비어 있지 않아야** 한다
⇒ deny 가 나와야 한다. **그런데 msg[9] 는 성공 결과**다. ⇒ 리졸버가 죽어 있었다는 `TASK_12 §5-b` 와
정합적이다. 다만 `TASK_1` 의 형제-충돌 절이 *"fb 로만 가서 안 보인다"* 를 주장하므로
**UNPROVEN 으로 남긴다**(라이브 env 없이는 못 가른다 · 리모트 접속 금지).
★단, **task 60 에 한해 이 물음은 점수와 무관**하다: G6 는 **주문(order) 축** 후보를 제시하는 게이트인데
이 태스크의 모호성은 **variant 축**이다. G6 가 살아 있었어도 *"세 주문 중 어느 것이냐"* 를 물었을 뿐
`8555936349 ↔ 6077640618` 는 못 가른다.

**ⓑ 읽기-증강 3종 침묵은 신규 발견이 아니다** — 형제 **5편이 이미 지목**했다(§7). 여기서는
**이 태스크에서의 크기**만 새로 잰다: 대조군 task 60 은 msg[5]·[9]·[13] 에 `[OPERAND DISAMBIGUATION`
**3회** + `[COMPUTED FACTS` **3회**를 달았고, 치료군은 **0회**다(런 전수 16↔0 · 16↔0).
대조군 msg[9] 꼬리 축자:
> `[OPERAND DISAMBIGUATION — every purchasable variant of this product with its item_id]`
> `- item_id=6077640618: {"options": {"color": "blue", "battery life": "8 hours", "water resistance": "not resistant"}, "available": true, "price": 242.92}` …
> `When the action needs a item_id, copy the EXACT item_id above … Never guess, invent, or carry an item_id from a different …`

⚠**인과는 UNPROVEN**: 같은 12 variant 가 **raw JSON 에 이미 전부** 있었다(치료군 msg[7] 축자).
증강 블록은 **중복 제시**이지 새 재료가 아니다. 따라서 *"이 레버를 켰으면 샀다"* 는 말할 수 없다.
반증(refut) 조건: 치료군 msg[7] 에 `6077640618`/`242.92` 가 없었다면 이 문단은 거짓 —
실측 축자로 **둘 다 있다**.

---

## §6. 대조군 분기점 — `hist_gpt52_reg12_PASS` task 60 (reward **1.0** · 같은 seed 626729 · 같은 32B)

| | 치료(t7391_reg12) | 대조(hist_gpt52) |
|---|---|---|
| 첫 손님 발화 | msg[1] — 요청 + *"make sure"* + *"confirm … before making the change"* | msg[1] — 요청 + *"confirm explicitly that the price will be the same or lower"* |
| **첫 갈림** | **msg[6] 본문+도구호출 동시** ⇒ 손님 턴 안 열림 | **msg[6] 텍스트 전용** ⇒ msg[7] 손님 응답 |
| 후보 제시 | **0회** | **msg[10] 텍스트 전용으로 2개 나열** |
| 손님 선호 발화 | **없음** | **msg[11]** 축자: *"I'd like the **blue option with no water resistance** (i.e., **not IPX-rated**)."* |
| 재조회 | — | msg[12] `get_product_details` 재호출 |
| 최종 제시 | — | msg[14] 텍스트 전용: *"Item ID: 6077640618 … Price: **$242.92**"* |
| 손님 승인 | **없음** | msg[15] *"**Yes, please proceed with Item ID 6077640618** …"* |
| write | msg[8] `new_item_ids=['8555936349']` | msg[16] `new_item_ids=['6077640618']` |
| 본문+도구호출 동시 | **2회**(msg 6·8) | **0회** |

⇒ **분기점은 msg[6]** 이다. 대조군은 거기서 **턴을 양보**했고, 그 결과로 msg[10] 의 *"several options"*
가 성립해 손님이 msg[11] 에서 **선호를 유출**했다. 치료군은 msg[6]·[8] 두 번 다 양보하지 않았다.

**⚠이 갈림을 우리 층에 못 붙인다.** 본문+도구호출 동시 발화는 **두 런 모두에 있다**:
치료 21/169 assistant 메시지 · 대조 13/141. **비율 차가 태스크를 설명하지 않는다** ⇒ `model` 이다.
sha 도 다르다(`fc0055dc` ↔ `5ebebbe8`) ⇒ **통제 실험이 아니다**([[77]]② 참조로만 인용).

---

## §7. 선행 대조 — **같은 원인인가, 달라졌는가**

검색 경로: `grep -rn "CONFIRM_RE\|G2_CONFIRM_WRITE" reports/facet_rft_2026/tasks__20260829/*.md`
= **10/10 파일 hit** · `grep -rn "task 60\|TASK_60" tasks__20260829/*.md` = `TASK_12.md:257` 1건 ·
`ls reports/facet_rft_2026/tasks_*/` · `ls reports/facet_rft_2026/*.md | grep -i retail`.

| 선행 | 무엇을 확정했나 | task 60 과의 관계 |
|---|---|---|
| **`TASK_12.md:257-259`** | `x611b` 센서스 표에 **task 60 행이 이미 있다** — *"60 / msg 8 / msg 1 / `sure`"*, 그리고 축자 *"**task 60 이 이 결함의 순수형이다 — 확인을 요구하는 문장 자체가 확인 게이트를 연다.**"* | **완전 동일 원인.** 이 보고서는 **새 원인을 주장하지 않는다** — 격리 재현·N_NEG·팔 비용을 **보강**한다 |
| `TASK_12.md:247` | *"실행된 write 22건 전부가 `CONFIRM_RE` 매치를 통과했다"* | `x615` 로 **재검산 22/22 일치** |
| `TASK_24.md:277-282` · `TASK_3.md:298` · `TASK_28.md:357` | 같은 게이트·같은 코드 줄을 독립 지목 | 4편째 재확인 ⇒ **신규성 0** |
| `TASK_12.md §9 P1` | 처방: *"직전 assistant 메시지가 이 write 의 세부를 발화했고 그 뒤 user 발화가 매치"* | ★**task 60 이 이 처방의 값싼 조작화를 반증(refute)한다** — 팔 B = 팔 A(41/41 동일). 인사말 msg[0] 이 `prevTxt` 를 만족시킨다(§4). **이 보고서의 신규 소득 ①** |
| `TASK_1.md:153` · `TASK_12.md:347` · `TASK_16.md:242,271-272` · `TASK_24.md:297-298,476-484` · `TASK_28.md:286` | `T2_PRESENT_READS`/`T2_PRESENT_NESTED`/`T2_CALC` 미수출 | **5편이 이미 지목** ⇒ 재보고 금지([[74]]). 여기서는 **task 60 크기**(3↔0·3↔0)만 추가 |
| `TASK_1.md:183-189` | `T2_PRESENT_*` 는 G6 **차단 경로의 술어가 아니다** · deny 는 비커밋 fb 로 갈 수 있다 | 그 정정을 **그대로 따랐다**(§5ⓐ 에서 G6 침묵을 UNPROVEN 으로 남김) |
| `REPLAY_SAFE_GATE_DESIGN_2026_07_06.md:355-362` (TASK_12 인용) | 설계 시점 오탐 분석은 **과잉차단 방향만** 봤다 | task 60 은 **과소차단**의 가장 순수한 사례 |

**이 보고서의 신규 소득(2건)**
① **팔 B 반증(refutation)** — `TASK_12 P1` 의 `prevTxt` 조작화는 **0 개를 막는다**(41 write 실측).
② **이 태스크는 게이트 단독으로 못 산다** — gold 경로가 *"several options"* 라는 **손님-조건부 유도**를
요구한다(시나리오 축자). 형제 12편 중 이 기전을 다룬 절은 **0**
(검색 경로: `grep -rn "If and only if\|여러 선택지\|후보 제시\|elicit" tasks__20260829/*.md` = 이 파일 외 0건).

---

## §8. 원인 확정 — [[77]] 네 칸

### 8-a. **CONFIRMED · our_layer (주)** — `G2_CONFIRM_WRITE` 가 확인 없는 write 를 통과시켰다

**①주장+양화** — `t7391_reg12` / task 60 / trial 0 / 결정점 **1개(msg 8)** / 잃은 변이 **1 of 1**.
전칭 아님: 이 sim 의 write 는 1건이고 그 1건이 통과했다.

**②근거(축자 + 파일:줄)**
- 궤적: msg[8] 이전 user 턴 = msg[1] 하나. 축자 *"Please **make sure** the price is the same or lower …
  and **confirm that explicitly before making the change**."* · msg[9] = 성공 결과(deny 꼬리 없음) ·
  `mutation_diff.blocked = []`.
- 코드: `gate_interpreter.py:16-18`(`CONFIRM_RE` 에 `sure`·`confirm` 이 **단독 대안**으로 들어 있다) ·
  `gate_interpreter.py:387-390`(confirm 분기 3줄 — `args` 미참조) · `t2_gate_patch.py:6938-6944`
  (`_regen_last_user` = 마지막 user content 그대로).
- 선언: `a2/retail.gate.json` `G2_CONFIRM_WRITE.applies_to` 에 `modify_pending_order_items` **포함**.
- 격리: `x615` ① `(True, None, None)` = 라이브 재현 · ② N_NEG1/N_NEG2 둘 다 `(False,'G2_CONFIRM_WRITE')`.

**③반증 조건 (refut)** — ⒜ `CONFIRM_RE.search(msg[1])` 이 `None` 이면 이 주장은 거짓
(실측 `('sure',(109,113))` · `('confirm',…)`). ⒝ msg[8] 에 `[Note: … blocked by a policy gate …]`
꼬리가 있으면 거짓(전문 substring 검색 0건). ⒞ `G2_CONFIRM_WRITE.applies_to` 에
`modify_pending_order_items` 가 없으면 거짓(선언 축자로 확인). ⒟ **⛔이 칸만으로 pass 를 주장하면
거짓이다** — 8-c 참조.

**④선행 확인(grep 한 경로)** — `grep -rn "CONFIRM_RE|G2_CONFIRM_WRITE" reports/facet_rft_2026/tasks__20260829/*.md`
(10/10 hit) · `grep -rn "task 60|TASK_60" tasks__20260829/*.md`(`TASK_12:257` 만) ·
`grep -n "confirm" scripts/distill/tau2/gate_interpreter.py` · `ls reports/facet_rft_2026/tasks_*/`.
⇒ **이미 있다**(`TASK_12 §5-a`). 재발명이 아니라 **보강**으로 적는다([[74]]).

### 8-b. **CONFIRMED · model (부)** — 본문과 도구호출을 한 메시지에 내 손님 턴을 두 번 삼켰다

**①주장+양화** — task 60 / trial 0 / msg[6]·msg[8] **2건** / 런 전수 21건 of 169 assistant 메시지.
**②근거** — 정책 축자 *"if you take a tool call, you should not respond to the user at the same time"* ·
msg[6]·msg[8] 이 `content` 와 `tool_calls` 를 동시에 갖는다(전수 계수).
**③반증 조건 (refut)** — 대조군 task 60 에서도 이 동시발화가 있었으면 model 귀속이 약해진다 →
실측 **0회**(§6). 또 이 동시발화가 치료군에만 있는 현상이면 우리 층 프롬프트 변화를 의심해야 하나,
**두 런 모두에 있다**(21/169 ↔ 13/141) ⇒ 런-특이 회귀가 아니다.
**④선행 확인** — `grep -rn "본문+도구호출\|content and tool_calls\|동시 발화" tasks__20260829/*.md`.
**우리 층 처방 없음** — 계측만(`content and tool_calls` 동시 발생을 sim 단위로 집계).
⚠프롬프트 diff 로 확증하려면 `T2_PROMPT_DUMP` 로그가 필요한데 **미회수**다([[78]] — 코드로 추정 금지).

### 8-c. **CONFIRMED(구조) · 이 태스크는 게이트 단독으로 못 산다** — 선호가 *유도되어야 하는 값*이다

**①주장+양화** — task 60 / 결정점 msg[8] / 필요 조건 **2개**(확인 게이트 + 후보 열거).
**②근거** — 시나리오 축자 *"**If and only if the agent provides several options**, you want the option
without water resistance."* · `x615` 실측 후보 3(§3[7]) · 대조군은 **msg[10] 에 2개를 나열한 뒤에야**
msg[11] 에서 선호를 발화했다(§6).
**③반증 조건 (refut)** — G2 가 msg[8] 을 막은 뒤 모델이 **선택지 1개만** 제시했는데도 pass 가 나오면
이 주장은 거짓. (형제 `TASK_1.md §3[16]` 축자가 보이듯 G2 deny 뒤 모델은 *자기가 고른 하나*를
나열하는 경향이 있다 — *"Here are the details for the Smart Thermostat exchange: - **Current Item** …
- **New Item** …"* = **1안 제시**. ⇒ 이 반증은 **실제로 일어날 법하다**.)
**④선행 확인** — `grep -rn "If and only if|여러 선택지|후보 제시|elicit" tasks__20260829/*.md`
→ 이 기전을 다룬 절 **0건**.

### 8-d. **무죄** — `user_sim` · `env`

- `user_sim`: 시나리오를 축자 그대로 연기했다. 오도 0. msg[19] 의 *"$226.49 … correct?"* 는
  **에이전트가 준 숫자를 되읽은 것**이다. 선호를 말하지 않은 것은 **시나리오의 조건절**이 불성립이라서다.
  반증 조건 (refut): 손님이 시나리오에 없는 정보를 발화했거나 선호를 준 적이 있으면 거짓 —
  user 턴 4건 전수 정독으로 0건 확인.
  ⛔[[21]] — 손님 발화를 면책으로 쓰지 않는다. 흡수는 에이전트 측 몫이고 그 자리는 msg[6]/msg[8] 이다.
- `env`: 필요한 값을 **전부, 제때** 냈다 — msg[5] 에 원가 $256.67 과 `item_id`, msg[7] 에 12 variant
  전문(gold `6077640618` / `$242.92` / `available:true` 포함). 오류 0 · 거절 0.
  반증 조건 (refut): 필요한 값 하나라도 문맥 부재였으면 거짓 — 축자로 전부 실재.

---

## §9. 처방 후보 (⛔코드·A2 **미수정** — 후보 제시까지 · [[70]] 부호 공개)

| # | 처방 | 층 | 근거 | **무엇을 파는가 (실측)** |
|---|---|---|---|---|
| **P0** | ⛔**`TASK_12 P1` 을 `prevTxt` 로 구현하지 마라** — 41 write 전수에서 **A 와 완전 동일**(0개 차단). 인사말이 술어를 만족시킨다 | our_layer | §4 팔 B · `x615` | (반증이라 파는 것 없음 — 헛수고를 막는다) |
| **P1′** | confirm 술어에 **인자 결속**: *"마지막 user 발화 뒤가 아니라, 이 write 의 **인자를 발화한 assistant 텍스트 뒤**에 온 user 발화만 확인으로 센다"*(팔 D 계열) | our_layer | §4 표 D · 선언 술어 *"of the action details"* | **CTRL gold 1건 오차단**(22#18) · TREAT gold 2건(3#24·54#37). ⚠[[69]] — 진짜 부호표는 **retail 114 전수 `reward` A/B** 로만 |
| **P2** | `CONFIRM_RE` 에서 관용구 `make sure` 를 승인에서 제외(닫힌 술어: 선행 토큰이 `make`/`makes`/`making` 이면 `sure` 는 승인 아님) | our_layer | §4 ① 매치 `('sure',(109,113))` | 좁고 싸다. 단 `confirm` 매치는 남으므로 **이 sim 은 여전히 통과**한다(§4 로 검산) ⇒ **단독으로는 부족** |
| **P3** | `CONFIRM_RE` 의 **극성**: 요구문(`before making the change`·`before you switch`)과 승인문을 가른다 | our_layer | msg[1] 축자 · 선행 `TASK_3.md:375` 가 이미 같은 항목을 냈다 | [[66]] 케이스-열거 위험 — **어휘 열거 금지**, 닫힌 술어(선행 부정/시간 스코프)로만 |
| **P4** | **후보 열거를 write 전에**: 이 write 의 variant operand(`a2/retail.gate.json` `variant_operand=["new_item_ids"]`)가 **>1 후보**를 가질 때 그 후보 집합을 손님에게 **나열했는지**를 확인 술어에 넣는다 | our_layer(A2+게이트) | §8-c · 시나리오 축자 · 대조군 msg[10]→[11] | 후보 나열 강제 = **턴 1개 추가**. 후보가 1개뿐인 태스크에서 순수 지연. **retail 114 부호표 필수** |
| **P5** | retail A2 에 `write_rules` 저작 — 정책 축자 *"you must **list the action details** and obtain explicit user confirmation (yes)"* 를 write 결정점에 재제시 | our_layer(A2) | 정책 산문 축자 · [[72]] 1회 오프라인 저작 | A2 저작 비용. 출처는 **정책 축자뿐**([[23]]). ⚠형제 `TASK_54`·`TASK_12 P5` 가 같은 항목을 이미 냈다 — **중복 등재 말 것** |
| — | 본문+도구호출 동시 발화(msg 6·8) | **model** | §8-b | **우리 층 처방 없음.** 계측만 |
| — | `T2_PRESENT_NESTED`/`T2_PRESENT_READS`/`T2_CALC` 수출 | our_layer | §5ⓑ | **형제 5편이 이미 등재** ⇒ **여기서 다시 제안하지 않는다**([[74]]·[[31]] 규칙 ③) |

⛔**gold 를 보고 고르지 않았다** — P0·P1′·P2·P3 은 게이트 **선언 술어**와 `CONFIRM_RE` 축자에서,
P4·P5 는 **정책 산문**과 **시나리오 지시문**에서 나왔다. gold(`6077640618`/`242.92`)는 §2 변이표와
§4 오차단 계수에만 등장한다([[23]]).

---

## §10. 재현 명령 (전부 로컬 · SSH 0 · git 커밋 0)

```powershell
cd C:\workspace\ba-frft\reports\facet_rft_2026
$env:PYTHONIOENCODING="utf-8"
py -3 x615_t7391_task60_confirm_iso.py      # (1) 라이브 재현 (2) N_NEG (3) 팔 A/B/C/D 전수
py -3 x611b_t7391_confirm_census.py         # 런 전수 확인-토큰 센서스(선행·task 60 행 포함)
```

변이표 재생:

```powershell
cd C:\workspace\ba-frft\scripts\distill\tau2
py -3 -c "import sys,gzip,json; sys.path.insert(0,'.'); import t2_forensic as F; d=json.load(gzip.open(r'C:\workspace\ba-frft\reports\facet_rft_2026\sim_results\t7391_reg12.results.json.gz','rt',encoding='utf-8')); s=[x for x in d['simulations'] if x['task_id']=='60'][0]; print(json.dumps(F.mutation_diff(s, F.mutating_tools('retail')), ensure_ascii=False, indent=1))"
```

⚠`F.mutating_tools()` 를 **인자 없이** 부르면 banking 집합이라 빈 표가 나온다(§2 함정).
