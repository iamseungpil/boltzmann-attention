# x737 — 다음 97 실행 계획서 + 설계서 (2026-09-04)

> **자리**: 이 문서는 정본이 아니다. 정본 작업 순서는 `x509_axis_queue_2026_08_24.json` 이고,
> 여기서 제안하는 두 수리는 그 큐의 **새 단계 후보**다. 프레임 LOCK 은 `RESEARCH_MASTER.md` §1.
> **범위**: 진행 중인 banking 97 캠페인이 끝난 **뒤**에 무엇을 고치고, 그 수리를 어떻게 97 태스크
> A/B 로 재는가. 캠페인 중에는 아무것도 배선하지 않는다([[54]]).

---

## 0. 선행 확인 — 어디를 찾아봤나 ([[74]] · §77 (4))

```
grep -rl "readloop|retention_offer" reports/facet_rft_2026/
  -> DAY7_PRESCRIPTIONS_DESIGN_2026_07_28.md · N97B_FIX_LEDGER_2026_08_05.md
     PAPER_TRACKC_DRAFT_v0_2026_07_24.md · RATE_SUBAGENT_DESIGN_2026_07_18.md
     RESEARCH_MASTER.md · STAGE2_GATE_DESIGN_2026_07_26.md · x540_spec_derivation_2026_08_25.json
grep -rn "readloop|resignation" scripts/distill/tau2/*.py
  -> t2_gate_patch.py:5477 · :13756 · :13759 · :13760 · :14550
     test_c207_envelope.py:99,151 · x60_l4_predicate.py:12
read  reports/facet_rft_2026/x509_axis_queue_2026_08_24.json (axis_table · steps S0~S2 · status_2026_08_29)
read  scripts/distill/tau2/a2/banking_knowledge.specific.json:4742-4850
```

**이 자리는 이미 두 번 다뤄졌다. 재발명하지 마라.**

| 선행 | 무엇을 확정했나 |
|---|---|
| `N97B_FIX_LEDGER_2026_08_05.md:323` | `close.requires` 에서 `retention_offer`·`log_reason` 을 **일부러 뺐다**. 정책 축자 *"If records are found within the past year … skip retention offers and proceed directly to processing the closure"* ⇒ 조건부 단계는 **표면화만** 한다 |
| `banking_knowledge.specific.json` `_note_nodes` | 구판이 `close` 에 `retention_offer` 를 필수로 걸어 **gold 경로를 6 태스크·47곳에서 막았다(x91)** |
| RESEARCH_MASTER `C157`·`C158` | 같은 절차의 premature-close 를 오프라인 사다리(n=12/16)로 측정. ①순수 reasoning 은 억제 실패 ②억제는 *"닫기 외 대안"* 빈 슬롯이 산다 ③올바른 회상은 A2 범주명(`retention_offers`)이 있을 때만. C158 축자: *"유일 고장=close 순서"* |

⇒ **`close.requires` 를 되돌리는 처방은 이미 폐기된 것이다. 이 문서는 그것을 제안하지 않는다.**

---

## 1. 원인 진술 ([[77]] 4칸)

### (1) 주장 + 양화

`bank_049ctl2_20260904_0534` 의 sim **#s626729** 에서, `close_credit_card_account_7834` 가 성공
실행된 **뒤에도** 절차 `credit_card_closure_retention` 의 잔여 노드 `retention_offer` 가 계속
"다음에 할 일"로 표면화되어, 체크리스트가 **56턴 중 34턴** 동일 상태
(`done=5 left=['retention_offer']`)로 머물고 KB 검색 루프가 이어졌다. 자매 sim **#s373753** 은
48턴 중 13턴. 두 sim 은 각각 **23회·18회** `readloop-turn` 으로 집계됐고, agent 컨텍스트가
**1,552 tok/턴**(#s626729 최근 15턴 실측)으로 자라 131,072 천장까지 **약 26턴**을 남긴 상태에서
마감으로 중단됐다. n=2 sim.

### (2) 근거 — 축자 인용 + 파일:줄

```
로그  bank_049ctl2_20260904_0534.log
  [sim=task_049#s626729] [T2_PROCEDURE] checklist proc=credit_card_closure_retention
                          nodes=6 done=5 left=['retention_offer']            <- 34회 동일
  [sim=task_049#s626729] [T2_FOLLOWUP] readloop-turn counted as resignation  <- 23회
  전이 3->4 직전:
  [sim=task_049#s626729] [T2_PROCEDURE] unmet tool=close_credit_card_account_7834 missing=[] unobservable=[]
  [sim=task_049#s626729] [T2_TOOL_OBS] Credit card account closed successfully.

표면화 문구  a2/banking_knowledge.specific.json
             procedures.credit_card_closure_retention.feedback.absent 축자:
  "[PROCEDURE] You are inside {procedure} and its next step has not been taken
   ({done} of {total} done): {checklist}  NEXT: {next}. {unlock_hint}Do that step before continuing."

설계 의도    같은 파일 _note_nodes 축자:
  "보유-제안은 조건이 도구 출력 내용에 달려 있어 우리에겐 닫히지 않으므로 표면화만 한다."

계기 구현    scripts/distill/tau2/t2_gate_patch.py:13759   self._t2_fu_readloop_turn = True
             scripts/distill/tau2/t2_gate_patch.py:13760   print("[T2_FOLLOWUP] readloop-turn ...")
             scripts/distill/tau2/t2_gate_patch.py:14550   _fu_genuine = not ..._fu_readloop_turn
```

⇒ **선언은 "표면화만", 실제 문구는 명령형("Do that step before continuing")** 이다. 그리고 readloop
플래그는 **chain 예비-예산 소비를 막는 데만** 쓰이고(`:14550`), 모델에게는 아무 말도 하지 않는다.

### (3) 반증 조건 / refutation conditions — 무엇이 관측되면 이 주장이 거짓이 되는가

주장과 **동시에** 적는다. 아래 셋 중 하나라도 관측되면 위 원인 진술은 무너진다.

- **R1 (refute by isolation)**: 종결 노드 실행 뒤 표면화를 끈 격리 조건에서도 KB 읽기 루프가 같은
  빈도로 나오면, 원인은 표면화가 아니다(모델의 절차 종료 판단 결손).
- **R2 (refute the premise)**: `feedback.absent` 가 종결 뒤 실제로는 발화하지 않았고 로그의
  `checklist` 줄이 표면화가 아니라 **계기 전용 출력**이라면, 이 원인 진술 전체가 거짓이 된다.
  ⇒ **P0 에서 먼저 확인한다**.
- **R3 (refute by history)**: 같은 태스크의 과거 `context_window_exceeded` 2건
  (`n97_gpu0_main_20260805` 109분 · `n97_gpu0_main_20260806b` 82분) 궤적에 종결-후 표면화가
  **없었는데도** 같은 루프가 있었다면, 표면화는 충분조건이 아니다.

### (4) 선행 확인

§0 의 grep 경로 목록 그대로. `close.requires` 완화는 **이미 판정된 사안**이며 되돌리지 않는다.

> ⚠ **이 진술은 아직 가설이다.** refutation(R1~R3)을 거치기 전에는 처방의 근거로 쓰지 않는다([[77]]).

---

## 1b. 캠페인 실패 20건 전수 per-step 포렌식 (2026-09-04 08:00)

> ⛔⛔ **이 절 전체에 붙는 경고 — `action_checks` 는 채점 단위가 아니다** ([[69]])
> 이 캠페인의 실패 20건 중 **19건이 `reward_basis: ['DB']`** 다. 그 태스크의 reward 는
> `db_check.db_match`(궤적 재실행 후 **DB 해시 비교**)에서 나오고, `action_checks` 는 **진단용**이다.
> 실증(n=1 sim · task_051): `051_6` 이 `match=False` 인데 그 호출은 **실제로 실행됐고 DB 를 바꿨다** —
> `msg60` 축자 *"Payment processed successfully! - Payment Amount: $3000.00 - New Checking Balance:
> $2000.00"*. 불일치의 정체는 중첩 payload 문자열 비교(`3000` ↔ `3000.00`)뿐이었다
> (`tau2-bench/src/tau2/data_model/tasks.py:195` `return tool_args == action_args`).
> ⇒ **아래의 칸 단위 수치(81칸·45/32/4·34칸 값 대조)는 "어디를 볼지"의 지도이지 실패 귀속이 아니다.**
> 귀속은 **변이 집합(MISSING/WRONGARG/EXTRA)** 으로 다시 세야 한다([[69]]).
> ⚠같은 이유로 내가 한때 낸 *"81칸 중 12칸이 직렬화 문제 · base 는 345칸 중 204칸"* 은 **철회한다** —
> `action_checks` 를 실패 단위로 놓은 데서 나온 수치다.



채점 55/97 · pass 35 · **fail 20**. 20건 **전부 `user_stop`** 종료다(크래시 0 · 컨텍스트 소진 0).

### 실패 칸의 기전 분해 (81 칸)

각 실패 칸의 도구 이름이 궤적에 ①아예 안 나왔나 ②나왔는데 안 불렀나 ③불렀는데 불일치인가:

```
③ 불렀는데 인자/값 불일치   45 칸 (55.6%)
② 이름은 나왔는데 미호출     32 칸 (39.5%)
① 이름이 안 나옴(미발견)      4 칸 ( 4.9%)   <- 검색은 병목이 아니다([[79]] 와 정합)
```

첫 실패 칸(연쇄의 머리)의 requestor = **assistant 11 · user 8**. 도구는
`call_discoverable_agent_tool` 6 · `call_discoverable_user_tool` 5 · `unlock_discoverable_agent_tool` 4.
전체 실패 칸의 45/81 이 `assistant/call_discoverable_agent_tool` 다.

### 우리 층 거절 130건의 절반이 **부수 차단**이다

```
tool-deny 130 = 원발 65 + [BLOCKED] 부수 65 (50%)
원발 문구: resolve-the-flagged-call 22 · [SIGNATURE] 16 · [OPERATOR-SCOPE] 6
           [POLICY GATE GB2_NOTICE_BEFORE_TRANSFER] 5 · [DUPLICATE-WRITE] 4
           [WRITE-EVIDENCE] 4 · [REFERENCE] 3 · [E-PLAN] 2 · [PROCEDURE] 1
```

`task_041` 한 건이 130 중 **67**을 차지하고, **한 턴에 최대 21건**이 동반 차단됐다(turn 44).

### §1b-refute — 내가 CONFIRMED 라 적었다가 **스스로 반증한 것** (2026-09-04)

**주장(원래)**: `task_041` 의 dispute **8 칸**을 우리 `[REFERENCE]` 게이트가 죽였다. n=8 칸 · sim 1개.

**반증 근거 — 축자 + 위치(sim#turn / action_id)**
```
같은 sim 의 같은 게이트 아래에서 dispute 6 칸이 **통과**했다:
  041_9  match=True  txn_645286a3dd13 digits=0652     041_19 match=True txn_dd095dee227f
  041_10 match=True  txn_1b4cc30a928e digits=3081     041_20~041_24 match=True (4칸 더)
연쇄의 머리는 따로 있다 — 손님 액션이 안 났다:
  041_3 requestor=assistant give_discoverable_user_tool(get_card_last_4_digits)  match=True
  041_4~041_7 requestor=user  call_discoverable_user_tool(get_card_last_4_digits) **match=False ×4**
그리고 우리 게이트는 그 결손을 **정확히 지적**했다 (task_041 turn 42·44 축자):
  "[WRITE-EVIDENCE] no tool output in this conversation shows the card's last 4 digits (2716).
   Do NOT guess or fabricate the digits ... call give_discoverable_user_tool to give the customer
   the get_card_last_4_digits ..."
```
⇒ **도구를 넘기는 것(041_3)은 우리가 했고, 손님이 그것을 호출하지 않았다(041_4~7).** 자릿수 없이
분쟁을 걸려는 시도를 우리 층이 막은 것은 **정당하다**. 같은 게이트 아래 6 칸이 통과했으므로
*"게이트가 8 칸을 죽였다"* 는 인과는 **성립하지 않는다**.

**남는 것(축소된 주장)**: `[REFERENCE]` 문면이 *"does not appear in any record returned by the
tools"* 라고 말하는데 그 8개 id 는 msg 17(role=tool)에 있었다 ⇒ **문면이 거짓**이다. 이것만이
확정이고, 처방은 D3 의 **문면·술어 일치**로 한정된다.

**반증 조건 / refutation (남은 주장에 대해)**: `apply_op` 가 집합을 돌려주는데 호출부가 스칼라로
비교하는 것이면 수리 위치가 `t2_compute` 다. 그리고 041 의 8개 id 중 criteria 부합이 1개뿐이라면
게이트 술어는 옳고 문면만 고치면 된다 ⇒ P3a·P3b.

**선행 확인**: `grep -rn "does not appear in any record" scripts/distill/tau2/`(`t2_gate_patch.py:2851`
`:9410`) · `grep -n "def resolve_reference_filter" -A 90 t2_resolve.py` · 회수된
`fb_bank_g97151p11_viewmax2_20260903_1924.jsonl` · 해당 sim 의 `action_checks` 전문.

### 원래 적었던 관찰 (위 반증을 붙여 읽어라) — `[REFERENCE]` 게이트의 오판

**주장 + 양화 (n=8 칸 · sim 1개)**: gold 요구 `file_credit_card_transaction_dispute_4829`
8 칸(041_11~041_18)이 실패했고 같은 턴에서 우리 게이트가 이들을 거부·동반차단했다.

**근거 (축자 + 위치)**
```
원발 deny 축자:
  "Error: [REFERENCE] the transaction_id you named does not appear in any record
   returned by the tools in this conversation."
부수 deny 축자 (같은 턴 나머지 전부):
  "Error: [BLOCKED] this call was not run because another call in the same turn was
   blocked: 'call_discoverable_agent_tool(file_credit_card_transaction_dispute_4829)'"

그런데 그 8개 transaction_id 는 모두 이미 대화에 있었다:
  txn_107c4fa829bd · txn_3880720b4409 · txn_816986054539 · txn_4f6e48543e07
  txn_b4f90f6ee392 · txn_5e6ad271fefb · txn_a42ce2e4156d · txn_c7a1c5fad26b
  최초 등장 = 전부 **메시지 17 (role=tool)**
  dispute 호출이 나간 메시지 = **23 · 64**   => 차단 시점에 이미 6~47 메시지 전부터 존재
```

**반증 / refutation**: 그 id 들이 msg 17 이 아니라 차단 **이후**에 처음 나왔다면 게이트가 옳고
이 귀속은 무너진다. 위 index 측정이 그 반증을 이미 쳤다(8/8 이 msg 17 · 호출은 23·64).

**선행 확인**: §0 의 grep 경로 + `fb_bank_g97151p11_*.jsonl`(회수분) + 해당 sim 의 messages 전문.

**⛔내가 처음 세운 가설은 틀렸다 — 코드를 읽어 반증했다.** *"게이트가 view 창만 보느라 msg 17 을
못 봤다"* 고 적었으나, `t2_gate_patch.py:9394` 는 `state.messages` **전사**를 넘긴다. 창은 무관하다.
**진짜 버그는 술어 자체**이고 §2 의 D3 에 적었다 — 게이트는 *"지목한 id 가 기록에 있는가"* 가 아니라
*"내가 계산한 단 하나의 id 와 같은가"* 를 검사하며, 손님이 8건을 분쟁하는 이 태스크에서는 정의상
7개가 거부된다.

---

## 2. 설계 — 수리 후보 다섯(D1~D4 · D6) + 조사 하나(L1) · 파생값·오선택은 측정만

### [[05]] 3질문 (설계서 상설 의무 · [[17]])

1. **무엇이 고정인가**: TBox weights + Scaffold 엔진. 두 수리 모두 **엔진 층**이고 도메인 상수를
   담지 않는다.
2. **무엇이 변경인가**: 없음. A2/ABox 는 **손대지 않는다** — `close.requires` 도 노드 목록도 그대로.
3. **도메인-특화가 섞이는가**: 아니다. 아래 술어는 banking 도구 이름도 태스크 id 도 담지 않는다
   ([[58]] · [[05]]). `credit_card_closure_retention` 은 **관측 대상**이지 조건이 아니다.

### D1 — 절차가 종결되면 잔여 노드 표면화를 멈춘다

```
종결 노드(terminal) := 그 절차의 노드 중
                        (a) 다른 어떤 노드의 requires 에 등장하지 않고
                        (b) mutating write 인 노드
규칙 : 종결 노드의 도구가 성공 실행되면 그 절차 인스턴스를 closed 로 표시하고,
       이후 feedback.absent / absent_many 를 그 인스턴스에 대해 발화하지 않는다.
불변 : feedback.unmet (다른 도구의 선행 차단) 은 그대로 둔다 — 그건 표면화가 아니라 게이트다.
```

- **도출 가능**: (a)(b) 둘 다 A2 에 이미 있는 필드(`requires` · `mutates`)로 닫힌다. 새 선언 0.
- **x91 재발 없음**: `close.requires` 를 건드리지 않으므로 gold 경로를 다시 막지 않는다.
- **위험**: 종결 뒤에도 정당한 후속 단계가 있는 절차에서는 표면화가 사라진다. ⇒ 스모크에서
  전 절차의 종결 노드 목록을 덤프해 눈으로 확인한다(§4 (2)).

### D2 — 읽기 루프에 이름과 출구를 준다 ([[64]])

현재 `_t2_fu_readloop_turn`(`t2_gate_patch.py:13759`)은 예비-예산 보호에만 쓰인다. 여기에 발화를
붙인다.

```
조건 : 같은 절차 인스턴스에서 readloop 턴이 연속 K회를 넘고 그동안 체크리스트 상태가 불변
발화 : (1) 무엇이 틀렸나 — 최근 K턴 동안 절차가 한 칸도 전진하지 않았다 (관측 사실만)
       (2) 무엇을 하면 풀리나 — 이미 있는 unlock_hint 축자를 이 문맥에도 붙인다:
           "Do not search the knowledge base for it: the name above is complete, and each
            search returns a large amount of text that will crowd out the conversation."
       (3) 남은 노드가 조건부(우회 가능)이면 그 사실을 서술형으로 알린다 (명령형 금지)
```

- **K 는 상수다. gold 로 고르지 마라([[23]]).** 격리(P2)에서 정하고 그 근거를 이 문서에 적는다.
- (3)의 "조건부" 는 A2 의 `requires` 구조로 닫힌다 — 내용 해석 없음([[59]]).

---

### D3 — reference-filter 의 문면과 술어를 **일치**시킨다 (§1b-refute 로 축소된 주장)

**주장 + 양화 (n=8 칸 · sim 1개)**: `task_041` 의 `file_credit_card_transaction_dispute` 8 칸이
이 게이트에 막혔다. **창(view) 때문이 아니다** — 게이트는 `state.messages` 전사를 받는다.

**근거 — 축자 + 파일:줄**
```
t2_gate_patch.py:9394   _rz_rf.resolve_reference_filter(am, state.messages, a2, ...)
                                                            ^^^^^^^^^^^^^ 전사 (창 가설 반증됨)
t2_resolve.py:1258-1264
  correct = _c.apply_op({"op":"filter","over":"records","return":keyf,"match":match, ...})
  if correct and str(correct) != str(chosen):
      return {"status":"deny", ...}
```
검사는 *"지목한 id 가 기록에 있는가"* 가 **아니라** *"내가 계산한 단 하나의 id 와 같은가"* 다.
`formalize_reference_criteria` 는 criteria **하나**를 뽑고 `apply_op(filter)` 는 id **하나**를
돌려준다. 그런데 041 의 손님은 거래 8건을 분쟁한다 ⇒ 8개 중 7개는 정의상 `chosen != correct`
가 되어 거부된다. ⚠단 §1b-refute 대로 **이것이 041 실패의 원인이라는 인과는 성립하지 않는다**
(같은 게이트 아래 dispute 6 칸이 통과했다). 확정된 결함은 아래 ②(문면 거짓)이고, ①은 P3b 가
criteria 부합 수를 세기 전까지 **가설**이다.

**두 번째 결함 — 거부 문면이 검사한 내용과 다르다** (`t2_gate_patch.py:9410` 하드코딩)
```
"[REFERENCE] the %s you named does not appear in any record returned by the tools in this
 conversation. Re-read the records you already fetched and name a %s that appears in one of them"
```
`t2_resolve.py` 가 만든 `REF_FILTER_FB` 를 **쓰지 않고** 이 문장으로 대체한다(2026-08-19 의
*"치환 폐기"* 결정의 부산물). 그래서 문면이 사실과 다르다 — 041 의 8개 id 는 모두 msg 17 의
도구 출력에 있었다. 모델은 이미 시킨 대로 하고 있었으므로 msg 64 에서 **같은 배치를 재발행**했고
또 막혔다. [[64]] 위반이다 — 억제가 이름을 대되 **그 이름이 틀린 처방**이다.

```
규칙 : deny 조건을 "계산한 하나와 다르다" 에서
       **"지목한 id 가 criteria 에 부합하는 레코드 집합에 없다"** 로 바꾼다
       (집합 소속 검사 — 여전히 닫힌 술어이고 옳은 값을 흘리지 않는다 · [[59]] · [[23]])
문면 : 검사한 것과 같은 말을 한다. 집합에 없을 때만 "기록에 없다" 고 말한다.
```

**반증 / refutation**: `apply_op` 가 실제로는 집합을 돌려주는데 호출부가 스칼라로 비교하는
것이라면 수리 위치가 다르다(`t2_compute` 쪽) ⇒ P3 에서 반환형을 먼저 찍는다. 그리고 041 의
8개 id 중 criteria 부합이 실제로 1개뿐이라면 이 귀속은 무너진다.

**선행 확인**: `grep -rn "does not appear in any record" scripts/distill/tau2/` →
`t2_gate_patch.py:2851 · :9410` · `grep -n "def resolve_reference_filter" -A 90 t2_resolve.py` ·
`a2/banking_knowledge.specific.json` 의 `reference_filter` 2 스펙(debit · credit).

### D4 — 턴 동반 차단(`[BLOCKED]`)을 **의존 호출로만** 좁힌다

**주장 + 양화 (n=130 deny · sim 20개)**: `tool-deny` 130 중 **65(50%)** 가 부수 차단이고,
`task_041` 은 한 턴에 21건(turn 44) · 17건(turn 40)이 함께 죽었다. 그 대상이 gold 요구 8 칸이다.

**근거 — 축자**
```
"Error: [BLOCKED] this call was not run because another call in the same turn was blocked:
 'call_discoverable_agent_tool(file_credit_card_transaction_dispute_4829)'"
```
```
규칙 : 플래그된 호출만 막고 나머지는 실행한다.
예외 : 막힌 호출의 **출력에 의존하는** 호출만 함께 막는다 — 그 의존은 A2 `arg_source_reads`
       로 이미 닫혀 있다(새 선언 0).
```
⚠ [[70]] 무엇을 파는가: 부분 실행은 한 턴의 일부만 반영된 상태를 만든다(원자성 상실). 그 대가를
**태스크별 부호표**로 세지 않으면 판정 불가다.

**반증 / refutation**: D3 를 고치면 원발 거부가 사라져 부수 차단도 함께 사라질 수 있다. 그러면
D4 는 불필요하다 ⇒ **D3 만 켠 팔을 먼저** 보고 D4 단독 효과를 판단한다.

**선행 확인**: `grep -rn "BLOCKED" t2_gate_patch.py` · 회수된
`fb_bank_g97151p11_viewmax2_20260903_1924.jsonl` 의 turn 별 deny 집계.

### D6 — `[DUPLICATE-WRITE]` 의 중복 창을 **상태 변화로 리셋**한다 ★오늘 가장 확실한 우리-층 결함

**주장 + 양화 (n=1 sim · gold 칸 2개 직접 사망)**: `task_051`(`bank_k8143med1_20260904_0135`)에서
gold 가 **같은 write 를 두 번** 요구하는데 우리 게이트가 두 번째를 막았고, 그 하류 2 칸이 죽었다.

**근거 — 축자 + 위치**
```
gold 051_2 {"agent_tool_name":"submit_credit_limit_increase_request_7392",
            "arguments":"{\"credit_card_account_id\":\"cc_5e4c1a83b0_bronze\",
                          \"user_id\":\"5e4c1a83b0\",\"requested_increase_amount\":1000}"}
gold 051_7 위와 **바이트 단위로 동일**            <- 같은 호출을 두 번 요구한다
궤적의 실제 submit 호출: msg23 **한 번뿐**

우리 게이트 축자 (task_051 turn 61·63):
  "[DUPLICATE-WRITE] This exact call (same tool, same arguments) already succeeded earlier in this
   conversation, so this call was REMOVED and not run ... It ran at message 23 and returned:
   Credit limit increase request submitted successfully ... Request ID: cli_e33db0778663 ...
   That change is already done. **Do NOT attempt this change again and do not do anything further
   about it.**"

에이전트가 손님에게 (msg65): "시스템이 방금 처리·거절된 동일 요청으로 인식해서 새로 제출하지 못하게
                              하고 있습니다"
손님 (msg66): "몇 분 기다릴게요. 시스템이 허용하면 $1,000 증액을 **다시 제출**해 주세요"
죽은 하류: 051_8 · 051_9 (approve_credit_limit_increase_5847) = match False
```
gold 흐름은 **신청 → 조회 → 거절 → 대금 완납 → 재신청 → 승인**이다. 완납이 상태를 바꿨으므로 두 번째
신청은 **다른 요청**인데, 우리 게이트는 "같은 도구·같은 인자"만 보고 동일하다고 판정한다.

```
규칙 : 직전 동일 write 이후 **그 대상에 대한 다른 mutating 호출이 있었으면** 중복이 아니다.
       (닫힌 술어 — 변이 이력의 집합 소속만 본다 · 도메인 리터럴 0 · [[59]])
선언 : A2 `write_once_keys` 가 이 게이트의 선언 자리다. 새 키 없이 조건만 넓힌다.
```

⚠ [[70]] 무엇을 파는가: 창을 열면 **진짜 중복 실행**(같은 변경 두 번 적용)이 돌아온다 — 이 게이트가
원래 막던 것이 그것이다. 태스크별 부호표 없이 배선 금지.

**반증 / refutation**: gold `051_7` 이 `051_2` 와 인자가 달랐다면 이 귀속은 무너진다 —
**동일하다**(위 축자). 재신청이 `DUPLICATE-WRITE` 아닌 다른 이유로 막혔다면 무너진다 —
deny 문면이 그 이름을 달고 있다. 그리고 D6 를 켠 격리에서 재신청이 통과해도 **승인까지 가지 못하면**
이 태스크는 여전히 안 산다.

**선행 확인**: `grep -rn "DUPLICATE-WRITE" scripts/distill/tau2/` · A2 `write_once_keys` ·
`_note_write_once_keys` · 회수된 `fb_bank_k8143med1_20260904_0135.jsonl` turn 61·63 ·
해당 sim 의 `messages` msg20·23·57·59·60·65·66.

---

### L1 — **꺼진 레버 조사** (D5 를 철회하고 이것으로 대체한다 · 2026-09-04)

> ⛔**D5 는 재발명이었다. 철회한다.** 아래 원문은 근거로 남겨 두되 **새 레버로 올리지 마라.**

**주장 + 양화 (n=3 런 · 발화 0)**: 내가 D5 로 제안한 열거값 검사는 **이미 존재한다**. A2
`write_arg_enum` 에 **9 개 선언**이 있고 그 **0번이 `open_bank_account.account_class`** 다.
그런데 이번 캠페인 3개 런에서 **발화 0회**다.

**근거 — 축자 + 파일:줄**
```
A2      banking_knowledge.specific.json  "write_arg_enum" (9 항목)
        [0] applies_to=call_discoverable_agent_tool · applies_when.prefix=open_bank_account
            arg=account_class · group_arg=account_type · group_map={...}
        [3] prefix=file_credit_card_transaction_dispute
            booleans=["contacted_merchant","eligible_for_provisional_credit"]   <- 파생값 10칸의 그 인자
엔진    t2_gate_patch.py:12004  _ens = (a2 or {}).get("write_arg_enum") or []
        t2_gate_patch.py:12005  if os.environ.get("T2_WRITE_ARG_ENUM") == "1" and _ens:
스위치  go_stack.sh 에 T2_WRITE_ARG_ENUM  **없음**
라이브  bank_k8141med1 · bank_g97151p11 · bank_re151med1  발화 각 **0**
과거 런 축자 (CAUSE_STEP_FORENSIC_RAW_2026_08_23.json:188):
  "[sim=task_055#s363271] [T2_WRITE_ARG_ENUM] deny val='Beige Savings Account'
   group=savings_accounts (후보 9)"
```
⇒ **레버는 있고, 예전엔 발화했고, 지금은 꺼져 있다**([[81]]). 할 일은 새 게이트를 짓는 것이 아니라
*"언제·왜 꺼졌나, 켜면 무엇이 달라지나"* 를 재는 것이다.

**⚠선행이 이미 경고한다 — 그냥 켜지 마라.** `refute_2026_08_23/refute_1.json` 축자:
*"⑵`T2_WRITE_ARG_ENUM_CAP` fail-open. 단 [[70]] 판정 의무 3종이 아직 안 채워졌다(레버 ON/OFF
reward 짝 없음·태스크별 부호표 없음) … **격리 프로브 없이 손대지 말 것**([[62]] ②③)."*
같은 문서가 이미 셋을 박제해 뒀다: ⓐgold 값 오거부(2026-08-13 FIX-6 로 수리됨) ⓑ**CAP(기본 3)
소진 후 fail-open** ⓒdeny 본문이 **영속 궤적에 안 남아** `messages` 만 보는 포렌식엔 안 보임.

**반증 / refutation**: `T2_WRITE_ARG_ENUM=1` 로 켠 팔에서 059·066·071 의 값이 그대로 통과하면
이 레버는 그 칸들을 사지 못한다 ⇒ L1 폐기. 그리고 CAP 3 이 sim 당 소진되면 fail-open 이 되어
**켠 것과 안 켠 것이 같아진다** — 그 경우도 폐기다.

**선행 확인**: `grep -rn "T2_WRITE_ARG_ENUM" scripts/distill/tau2/` · `go_stack.sh`(부재 확인) ·
`reports/facet_rft_2026/CAUSE_STEP_FORENSIC_RAW_2026_08_23.json`(:188 · :251 · :271) ·
`reports/facet_rft_2026/refute_2026_08_23/refute_1.json`(:7 · :31 · :55) ·
`reports/facet_rft_2026/lever_consolidation_map_2026_08_19.json`(:1661 · :1667).

---

### P5 — 파생값 17칸은 **수리가 아니라 측정**이다

**주장 + 양화 (n=17 칸 · sim 5개)**: 값만 틀린 34 칸 중 **17 칸**이 정책 파생값이다.
```
eligible_for_provisional_credit  10칸 (041×8 · 040×2)  GOLD False ↔ OURS True   (전부 한 방향·과다 인정)
customer_max_liability_amount     3칸 (085)            GOLD 50    ↔ 100.0 · 89.99 · 14.99
new_rewards_earned                2칸 (026)            GOLD 1020 · 1500 ↔ 6300
provisional_credit_eligible       1칸 (085_7)          GOLD True  ↔ False        (반대 방향)
expedited_shipping                1칸 (038_4)          GOLD True  ↔ False
=> 12/17 이 불리언, 그중 10칸이 같은 인자를 같은 방향으로 틀린다.
```

**⛔ 이 자리는 일부러 비워 둔 자리다 — 계산 레버를 되살리면 실험이 무효다.**
`a2/banking_knowledge.specific.json` 의 `compute_ops` 는 `{}` 이고 옆의
`_note_compute_ops_removed_2026_08_19` 축자:

> *"REMOVED (user decision 2026-08-19, plan A). Two ops were deleted because **the engine was
> producing values that the benchmark scores as gold arguments, which erases the very deficit we
> measure** ([[62]]), and because one constant was fitted to gold ([[23]]). (1)
> `file_debit_card_transaction_dispute.customer_max_liability_amount` used thr=30 days while the
> policy text says 'within 2 business days of statement'; the threshold was chosen by **gold
> reproduction rate** (T1=2 → 73.6% vs T1=30 → 89.4%) … Live evidence in run
> `bank_t7326_*_20260819q`: `'[T2_RESOLVE] compute silent-repair customer_max_liability_amount
> -1->50' fired 8 times in **task_085**."*

⇒ **085 의 그 3칸은 예전에 엔진이 채워 주던 바로 그 칸이다.** 다시 계산하면 [[23]]·[[62]] 위반이다.

**경계는 같은 노트가 그어 뒀다** — 축자:
> *"The policy tables themselves stay legal as **DELIVERED TEXT** (surface the doc_036/_031
> wording to the model); **what is forbidden is the engine writing the value into the call**."*

정책 조건 자체는 KB 에 있다: 책임 상한 `doc_036/_031` *"within 2 business days of statement→$50 /
within 60 days→$500 / after→전액"* · 구조 `min(disputed_amount, tier_cap)`; 임시 크레딧 `doc_032`
`ALL{timely ≤ 60일, category ∈ 5종, written_statement, account OPEN}`. 085 의 OURS(100.0·89.99·
14.99)는 **거래 금액 자체**로 보이고(티어 표 미적용), 041 의 10칸은 `ALL{}` 을 **평가하지 않고
True 로 넘긴** 모습이다.

### R1 정밀 분석 — **gold 없이 닫히는가? 부분적으로 그렇다** (2026-09-04)

*"파생값은 gold 를 볼 수밖에 없나"* 에 대한 답. **아니다.** 규칙은 KB 에 있다. 2026-08-19 의 위반은
**규칙을 몰라서가 아니라 상수를 gold 재현율로 고른 것**이었다. 파라미터별로 가른다 (n=17 칸).

```
[A] 규칙이 KB 축자에 있다 — gold 불필요
  eligible_for_provisional_credit (10칸)
    KB 축자 doc_credit_cards_credit_cards_(general)_015:
      "Previous Disputes: The customer has not filed more than 2 disputes in the past 12 months"
      (이 축자는 A2 `_note` 에 **이미 인용돼 있다** — 새 사실 0)
    => 규칙은 "직전 12개월 분쟁 2건 이하" + 카테고리 적격 + 60일 이내.
       분쟁 이력은 get_user_dispute_history_7291 로 **관측된다**. 계좌 OPEN 여부도 관측된다.
  customer_max_liability_amount (3칸)
    KB 축자 doc_036/_031: "within 2 business days of statement -> $50 / within 60 days -> $500 /
                           after -> 전액", 구조 min(disputed_amount, tier_cap)

[B] 관측이 아니라 **손님 발화 해석**이 정한다 — LLM 몫([[52]])
  8건 중 어느 3건에 임시 크레딧을 줄 것인가. 041 의 손님은 "가장 큰 금액들"이라고 말한다.
  => 순위·선택은 해석이다. 엔진이 고르면 [[62]] 위반.

[C] 상수가 정책 문면과 어긋난다 — **여기가 2026-08-19 의 죄**
  구판은 thr=30일을 썼는데 정책 축자는 "within 2 business days".
  그 30일은 **gold 재현율로 선택**됐다(T1=2 -> 73.6% vs T1=30 -> 89.4%) => [[23]] 위반.
  ⇒ 상수를 다시 고를 때 gold 를 보면 같은 죄를 반복한다. **정책 축자 외의 출처 금지.**
```

⇒ **결론: gold 는 필요 없다. 필요한 것은 ⓐKB 축자를 전달하고 ⓑ선택은 모델에게 남기는 것**이다.
같은 노트가 이미 그 경계를 그어 뒀다 — *"policy tables stay legal as DELIVERED TEXT … forbidden is
the engine writing the value into the call."*

⚠**주의 — 이 규칙을 `tasks.json` 에서 읽지 마라.** 벤치의 태스크 주석에 *"maximum 3 disputes can
receive provisional credit … only 3 can actually receive provisional credit"* 라는 **해설이 들어
있다**. 그것은 gold 주석이지 KB 가 아니다([[23]]). 위 [A]의 출처는 **`documents/` 아래 KB 문서**여야
한다.

### 격리로는 안 되는가 — **된다. 그리고 그것이 정확히 옳은 도구다**

이 물음은 [[62]] 2b 가 이미 정한 형태다: *격리에서 되면 결손은 전달(부하)이고, 격리에서도 안 되면
능력 경계다.* 무료이고 gold 를 안 본다.

```
P5-iso : 041 · 040 · 085 의 결정 시점에서 모델이 **실제로 받은 재료**만 주고
         + KB 축자(doc_015 · doc_032 · doc_036/_031)를 앞쪽에 두고
         => eligible_for_provisional_credit / customer_max_liability_amount 를 산출시킨다
부정통제 : 같은 길이의 무내용 문구를 넣은 팔([[57]])
exit    : 격리에서 닫히면 => 결손은 **전달**이고 합법 레버는 "표를 앞쪽에 전달"이다(값은 안 쓴다)
          격리에서도 안 닫히면 => **능력 경계**로 기록하고 이 17칸을 수리 대상에서 내린다
```
전례가 있다: `x511` 이 ①금액 축에서 같은 실험을 했고 *"B_policy(궤적과 같은 자리·앞쪽) **8/8** ·
C_policy_last(요구 직전) 8/8 합치되 산수가 깨진다"* 를 얻었다 — **표를 어디에 두느냐가 결과를 갈랐다**.
이 자리도 같은 설계를 쓴다.

**반증 / refutation**: 격리에서 닫히는데 라이브에서 안 닫히면 iso↔live 차이를 **프롬프트 두 개를 찍어
diff** 해야 한다([[78]]) — 추정 금지.

**선행 확인**: `_note_compute_ops`(PROVENANCE 축자 · doc_036/_031 · doc_032 인용) ·
`_note_compute_ops_removed_2026_08_19` · `x509_axis_queue` 의 `steps[S2].result.isolation_x511` ·
`grep -rn "provisional" tau2-bench/data/tau2/domains/banking_knowledge/documents/`.

**그래서 P5 는 측정만 한다 (무료·오프라인)**
```
P5a  041 · 040 · 085 의 궤적에 그 정책 표(doc_032 · doc_036/_031)가 **실제로 전달됐는가**
     전달됐는데 틀렸다 => 능력 경계(모델 몫) · 전달 안 됐다 => 전달 레버가 자리다(값은 안 쓴다)
P5b  L1 을 켠 팔에서 write_arg_enum[3].booleans 가 그 10칸에 발화하는가
P5c  CAP(기본 3) 소진 시점과 그 뒤 통과 여부 (fail-open 재현)
```
**exit**: P5a 가 "전달됨"이면 이 17칸은 **수리 대상에서 내린다**(측정값으로만 기록).

**반증 / refutation**: 표가 전달되지 않았음이 확인되면 *"모델 몫"* 이라는 접기는 거짓이 되고,
전달 레버가 정당한 후보가 된다. 반대로 전달됐는데도 틀렸다면 어떤 우리-층 처방도 이 칸을 못 산다.

**선행 확인**: `_note_compute_ops_removed_2026_08_19` · `_note_compute_ops`(PROVENANCE 축자) ·
`grep -rn "compute_ops" scripts/distill/tau2/a2/*.json`(specific:67 · gate:260 모두 `{}`) ·
`t2_resolve.py:1281 resolve_compute_params`(선언 없으면 no-op).

---

### 오선택 14칸 — **분류만 한다. 수리 대상이 아니다**

**주장 + 양화 (n=14 칸 · sim 5개)**: `credit_card_account_id`(041×4 · 040×2) ·
`card_id` `_green↔_blue`(092) · `_lb/_green↔_lg`(078) · `transaction_id`(026×4).
지목한 **레코드가 다르다** — 값 형식도 계산도 아니다.

**근거 — 축자(값 대조)**
```
041_4  GOLD cc_a6a7d745b2_gold   OURS cc_a6a7d745b2_crypto
041_5  GOLD cc_a6a7d745b2_crypto OURS cc_a6a7d745b2_gold      <- 서로 뒤바뀐 꼴
092_13 GOLD dbc_rw42b8d3e1_green OURS dbc_rw42b8d3e1_blue
078_3  GOLD dbc_mc78a5b9d2_lb    OURS dbc_mc78a5b9d2_lg
```

이 축은 x509 큐의 **②범주**이고 `x512`(경계 판정 철회) · `x513`(*"표를 줘도 057·063 은 0/6"*)이
**이미 판정한 자리**다. 여기서 새 처방을 만들지 마라([[74]]).

**필요한 것은 격리 프로브 하나**: *같은 종류의 카드·계좌가 여럿일 때 손님 발화의 지시체를
고르는가*. [[18]] 상 F3/경계 판정 전에는 **정보-맞춘 격리**가 선행이고, 이 문서는 그 프로브를
**기술만 하고 설계하지 않는다**(큐 밖 작업 금지 · §74-d).

**반증 / refutation**: 격리에서 지시체 선택이 닫히면 이것은 능력 경계가 아니라 전달 부하이고,
그때는 ②범주 축의 판정을 되돌려야 한다.

**선행 확인**: `x509_axis_queue_2026_08_24.json`(`axis_table.boundary_RETRACTED` · `status_2026_08_24_pm.②범주`) ·
`grep -rn "x512\|x513" reports/facet_rft_2026/`.

---

### (철회됨) D5 — `[ARG-ENUM]`: 선언된 값 집합에서만 오는 인자를 검사한다

**주장 + 양화 (n=1 칸 · sim 1개)**: `task_059`(`bank_k8141med1_20260903_2256`)는 gold 6 칸 중
**5 칸을 통과하고 059_4 한 칸**으로 떨어졌다. 그 한 칸의 차이는 문자열 하나다.

**근거 — 축자 + 파일:줄**
```
GOLD  059_4 : account_class = "Green Account"
OURS        : account_class = "Green Account (savings)"          <- 모델이 "(savings)" 를 덧붙였다

도구 문서 축자  tau2-bench/src/tau2/domains/banking_knowledge/tools.py:2384
  "account_class (string): The full official account class name"
정책 문서의 등급 열거 (tasks/정책 JSON 등장 횟수)
  Green Account 194 · Sky Blue Account 18 · Gold Account 16 · Silver Plus Account 16 · Bronze Account 6

우리 층은 이 호출을 건드린 적이 없다 — task_059 의 tool-deny 는 3건뿐이고 전부 다른 자리다:
  turn 49  "resolve the flagged call(s) first"
  turn 53  "[ACTION] 'apply_for_credit_card' is run by the CUSTOMER, not by you"
  turn 53  "[ARG-EMPTY] ... left the required argument(s) ... as an empty string"
```

⇒ **059 는 우리 스택의 부작용이 아니다**(같은 sim 의 다른 `open_bank_account` 칸 059_3 은 통과).
그러나 **우리가 잡을 수 있었던 결함**이다: `account_class` 는 자유 문자열이 아니라 **열거값**이고,
지목한 값이 그 집합에 속하는지는 **닫힌 술어**다([[22]] 변이 불변 · [[59]] 문자열 소속만).

```
규칙 : 인자의 허용값 집합이 선언돼 있으면(A2/정책 열거), 집합에 없는 값을 거부한다.
문면 : 무엇이 틀렸나(집합 밖) + 무엇을 하면 풀리나(선언된 값 중 하나를 쓰라).
       ⛔옳은 값을 골라 주지 않는다 — 고르는 순간 측정 대상이 사라진다([[62]] · [[23]]).
```
같은 계열이 **이미 둘 있다**: `[ARG-EMPTY]`(빈 문자열) · `[SIGNATURE]`(선언 안 된 인자).
D5 는 그 형제이고 엔진에 도메인 리터럴을 박지 않는다([[58]]) — 집합은 A2 에서 온다.

**반증 / refutation**: 그 열거가 A2·정책에서 **닫히지 않으면**(등급이 문서마다 다르거나 자유 서술이면)
이 술어는 성립하지 않고 D5 는 폐기다. 그리고 059 의 `"Green Account (savings)"` 를 거부했을 때
모델이 올바른 값으로 재발행하지 못하면 **레버가 사는 것은 0**이다 ⇒ P4 에서 격리로 확인한다.

**선행 확인**: `grep -rn "account_class" tau2-bench/data/tau2/domains/banking_knowledge/*.json` ·
`grep -rn -A3 "account_class" tau2-bench/src/.../tools.py`(:2377·:2384·:2394) ·
`grep -rn "ARG-EMPTY|SIGNATURE" scripts/distill/tau2/` (기존 형제 게이트 확인) ·
회수된 `fb_bank_k8141med1_20260903_2256.jsonl` 의 task_059 deny 3건 전문.

---

## 3. 격리 먼저, 배선은 그 다음 ([[62]] · [[78]])

### P0 (선결·무료) — R2 를 먼저 친다

`feedback.absent` 가 종결 뒤 실제로 발화했는지 확인한다. 회수해 둔
`sim_results/bank_049ctl2_20260904_0534.log.gz` 와 `fb_*.jsonl.gz` 에서
`[PROCEDURE] You are inside` 의 turn 별 출현을 센다.
**0건이면 §1 의 원인 진술을 폐기하고 이 문서를 여기서 멈춘다.**

### P1 — D1 격리

- 프로브는 프롬프트를 쓰지 않고 **엔진 빌더를 부른다**([[78]]). 팔은 선언 오버라이드 한 칸
  (`terminal_closes_procedure = on/off`).
- 재료: 그 sim 이 종결 시점에 실제로 받은 메시지 전량(축자 재생).
- **exit**: off 에서 읽기 루프 재현 ∧ on 에서 소멸 ⇒ D1 배선 자격. 둘 다 루프면 R1 성립 ⇒ D1 폐기.
- 부정통제 필수([[57]]): 같은 길이의 무내용 문구를 붙인 팔.

### P3 — D3/D4 격리 (2026-09-04 신설)

- **P3a**: `apply_op(filter)` 의 반환형을 찍는다(스칼라 vs 집합). 집합이면 수리 위치가 `t2_compute` 다.
- **P3b**: 041 의 8개 id 각각이 formalize 된 criteria 에 부합하는지 결정론으로 센다.
  8개 다 부합하면 D3 확정 · 1개만 부합하면 이 귀속은 무너진다.
- **P3c**: D3 만 켠 팔 vs D3+D4 팔을 같은 재료로 돌려 D4 의 단독 기여를 잰다([[57]] 부정통제 포함).
- **exit**: D3 가 041 의 8 칸을 통과시키는가 · D4 가 그 위에 무엇을 더 사는가.

### P4 — L1 격리 (꺼진 열거 레버)

- **P4a**: `T2_WRITE_ARG_ENUM` 이 **언제 꺼졌는지** git 이력으로 찾는다(`git log -S`). 의도적 OFF 면 그 이유를 인용한다.
- **P4b**: 켠 팔에서 059·066·071 의 값이 실제로 거부되는가, 그리고 모델이 **선언된 값으로 재발행하는가**.
- **P4c**: `T2_WRITE_ARG_ENUM_CAP`(기본 3) 소진 후 fail-open 재현 — 켠 것과 안 켠 것이 같아지는지.
- **exit**: 재발행 성공 ∧ CAP 소진 전 발화 ⇒ L1(켜기) 자격. 부정통제 필수([[57]]).
  ⛔[[70]] 판정 의무 3종(ON/OFF reward 짝 · 태스크별 부호표 · 무엇을 팔았나)을 채우기 전엔 켜지 마라.

### P2 — D2 격리 + K 결정

- K ∈ {2, 3, 5} 를 각각 재고, 루프가 끊기는 **가장 작은 K** 를 쓴다. gold 무참조.
- **exit**: 어떤 K 에서도 안 끊기면 D2 폐기(경로 없음으로 기록).

### 배선 조건

P1·P2 를 통과한 것만 배선한다. 통과 후 `go_stack.sh` 정본 런처에 **등재까지가 한 작업**이고,
첫 런에서 **실발화를 확인**한다([[81]] — 고쳐 놓고 켠 적 없는 레버가 실재한다).

---

## 4. 스모크 게이트 ([[73]])

full-run 전에 반드시 통과시킨다. 단위테스트 통과 ≠ 라이브 발화.

```
(1) --num_tasks 10 --num_trials 1  (~6분)   크래시 0
(2) 전 절차의 terminal 노드 목록 덤프 — 눈으로 확인 (D1 이 엉뚱한 노드를 종결로 잡지 않는가)
(3) D1 발화 카운트 > 0 · D2 발화 카운트 > 0   <- 0 이면 배선 경로가 틀렸다([[81]])
(4) 기존 배터리: test_a2_three_layer.py · test_c207_envelope.py · test_lever_reachable.py
(5) 등가 게이트: 정본 A2 만 고치고 gate.json 미동기화면 FAIL ([[24]])
```

---

## 5. 실험 — 97 태스크 A/B

### 팔

```
A (대조) : 현행 sha (수리 전)
B (처치) : A + D1 + D2   (P1/P2 를 통과한 것만)
```

### ⚠ 대조군을 재실행해야 하는가 — 판단과 대가

지금 도는 캠페인(2026-09-03~04)은 **비교 규격이 깨져 있다**:

- `max_concurrency` 가 런마다 **4 와 2 로 섞였다**
- 서버가 `.151` 과 `.153` **두 대**로 갈렸고 `.151` 은 2026-09-04 07:00 에 반납했다
- 일부 태스크는 죽은 런의 잔여를 다른 태그에서 이어 돌았다

**권고: 대조군을 재실행한다.** reward 자체는 conc 에 불변일 **가능성이 높지만**(축출은 재계산일
뿐 토큰을 바꾸지 않는다), 이 태스크군의 지배적 실패 모드가 **컨텍스트 소진**이라 배치 조건이 종료
시점에 개입할 여지가 있고, 그 여지를 남긴 채로 Δ 를 주장할 수 없다([[54]]).

**무료 대안**: 이번 캠페인을 대조군으로 재사용한다. 그 경우 *"대조군은 혼합 배치에서 수집됨"* 을
명시하고, Δ 판정 시 **컨텍스트 소진으로 끝난 sim 을 따로 센다**. 비용을 아끼려면 이쪽.

### 배치 설계 (`.151` 반납 반영) — ★2026-09-04 07:40 측정으로 규칙을 바꿨다

```
엔진 2개 (.153 GPU0=8141 · GPU1=8143)
레인 = 엔진당 1개 (kvlane.sh · nb() 는 HOST:PORT 로 센다 — 포트만으론 엔진이 식별되지 않는다)
MAXB=1

★배치 규칙: conc 숫자가 아니라 **비행 컨텍스트 합 <= kv_cache_size_tokens (171,749)**
```

**왜 conc 가 기준이 아닌가 — 실측(§5a)**. `conc 2` 는 태스크가 짧을 때만 맞는 근사다.
92k 짜리 태스크는 **혼자 돌려야** 예산 안이고, 40k 짜리 둘은 같이 돌려도 된다.

⇒ 큐를 짤 때 각 묶음의 **태스크별 최대 `agent_response` prompt 실측치**로 묶는다. 그 값은
회수된 런 로그에서 무료로 뽑힌다(`[T2_GEN_TRACE] call=agent_response ... prompt=<N>` 의 최댓값).

### §5a 근거 — prefix 캐시 붕괴 측정 (2026-09-04 07:4x · 4분 구간)

같은 sha · 같은 팔(`viewmax2`) · 같은 모델인데 두 엔진의 **prefix 적중률이 16배** 갈렸다.

```
포트 8141 (비행 4 sim)  질의 265,321 블록 · 적중   9,408 -> 구간 적중률  3.5%  · 축출 +8
포트 8143 (비행 3 sim)  질의 198,782 블록 · 적중 112,112 -> 구간 적중률 56.4%  · 축출 +2

비행 컨텍스트 합 (sim 별 agent_response 최대 prompt)
  8141   92k + 89k + 87k + 77k = 346,733  = 예산의 2.0배  -> 적중률  3.5%
  8143   99k + 52k + 37k       = 190,779  = 예산의 1.1배  -> 적중률 56.4%
```

**기전**: sim 이 user-sim(gpt-5.2) 응답을 기다리는 동안 엔진에서 내려가고, 그 사이 다른 sim 들이
그 sim 의 캐시된 prefix 블록을 밀어낸다. 다음 턴에 돌아오면 90k 를 처음부터 다시 계산한다.
**이 퇴출은 `num_preemptions_total` 에 한 줄도 안 남는다** — 그 계수기는 *비행 중* 요청의 선점만
센다. 위 구간의 축출이 +8/+2 로 거의 0인데 적중률이 3.5% 인 것이 그 증거다.

**반증 / refutation**: 비행 합을 예산 아래로 넣었는데도 적중률이 60% 를 밑돌면 이 설명은 부족하고,
남은 몫은 다른 곳(예: 재생성 채널)에 있다.

**재생성 채널(`_ap_regen`)은 주범이 아니다 — 반증됨.** 재생성 비율은 8141 **7.6%**(619 중 47) ·
8143 **11.1%**(162 중 18) 로, **재생성이 더 잦은 쪽의 적중률이 16배 높다**. 상관이 반대다.
다만 8143 의 56.4% 도 순수 append 대화의 기대치(90%대)에는 못 미치므로, **비행 합이 예산 아래로
완전히 들어간 상태에서 한 번 더 재서** 잔여를 귀속시킨다(무료).

### 비용·기간 (정직하게)

```
sim 수 : 97 × 2 arms = 194 (대조군 재실행 시) / 97 (재사용 시)
처리율 : 엔진 2개 · conc2 실측 sim 당 12~65분 (n=3 · bank_k8143med1)
         => 낙관 4 sim/h · 비관 2 sim/h
기간   : 194 sim -> 48~97시간 / 97 sim -> 24~48시간
비용   : user-sim = openrouter gpt-5.2 ([[30]] 권장표준) — [[09]] 사용자 승인 필요
```

⚠ 위 처리율은 **비행 합이 예산의 1.1배였던 구간**(적중률 56%)에서 잰 것이다. §5a 규칙대로 예산
아래로 묶으면 재-prefill 이 줄어 빨라질 여지가 있으나 **그 상태를 아직 재보지 않았다** — 추정에
반영하지 않았다. 첫 배치에서 재고 이 표를 갱신한다.

### 판정 기준

- 1차 지표는 **reward**(궤적 재실행 후 DB 해시 비교 · [[69]]). 집계 metric 에서 결론 직행 금지([[08]]).
- **Δ ≥ 10/97** 을 유의로 본다([[73]] 의 Δ≥4/40 관례를 97 로 환산).
- **태스크별 부호표 필수**([[70]]) — 무엇을 샀고 **무엇을 팔았나**. D1 이 표면화를 줄이므로 종결
  전 단계를 놓치는 태스크가 생길 수 있다. 그 손실을 세지 않으면 판정이 아니다.
- 우리 층 귀속은 per-step 포렌식 + 적대적 refutation 을 거친 것만 CONFIRMED([[73]]).

### ★비용 축 — reward 만 보면 "무엇을 팔았나"가 안 보인다 ([[70]])

**주장 + 양화 (n=2 sim · base 대조)**: 같은 태스크를 base 는 분 단위로 통과하는데 ours 는 시간
단위를 쓴다. 두 사례 모두 base 팔이 **pass** 한 태스크다.

**근거 — 축자 + 위치**
```
task_059   base(x644) reward=1.0 ·  15분 · msg 47      ours reward=0.0 · 291분 · msg 72
task_064   base(x644) reward=1.0 ·  20분 · msg 68      ours 4.7시간째 미완 (2026-09-04 08:15)

task_064 의 생성 호출 분해 (bank_k8141med1_20260903_2256.log · [T2_GEN_TRACE] call=... 집계)
  agent_response 29  ↔  부수 생성 30
  (intent_operator_formalize 5 · source_claim_formalize 5 · recommend_formalize 4 ·
   agent_response_unified_regen 6 · claimprov 6 · selfdecl 3 · sg_arg_docs 2 · 기타)
=> 실질 턴당 생성이 2배 이상이다.
```

그러므로 판정표에 **세 칸을 더 적는다**(태스크별 부호표와 같은 줄에):

```
① reward 짝 (A/B)                      <- 지금 유일하게 보고 있는 것
② sim 당 벽시계 분                       <- 우리가 파는 것
③ gold 대비 생성 호출 배수 (agent_response + 부수 생성) / base 의 turn 수
```
②③ 없이 Δ 를 보고하면 *"정확도를 샀다"* 만 남고 **대가가 장부에서 사라진다**. base 팔의 분/턴은
`x738_q38_base97_census_2026_09_04.md` 의 두 런에서 무료로 뽑힌다.

**반증 / refutation**: ②③ 의 격차가 **KV 경합만으로** 설명되면(예산 안에서 돌린 배치에서 격차가
사라지면) 이건 우리 레버의 대가가 아니라 배치 문제다 ⇒ §5a 규칙대로 묶은 첫 배치에서 다시 잰다.

**선행 확인**: `grep -rn "T2_GEN_TRACE" scripts/distill/tau2/` · 회수된 base 런
`bank_x644_q38base_bank78_20260830.results.json.gz`(duration·messages) · 이 캠페인 로그의 호출 집계.

---

## 6. 중단 조건

| 신호 | 조치 |
|---|---|
| P0 에서 `[PROCEDURE] You are inside` 0건 | 이 문서 폐기. 원인 진술이 틀렸다 |
| P1 on/off 둘 다 루프 (R1) | D1 폐기 · *"표면화는 원인이 아니다"* 로 기록 |
| P2 어떤 K 에서도 안 끊김 | D2 폐기 · 경로 없음으로 기록 |
| 스모크 (3) 발화 0 | 배선 경로가 틀렸다. 런 금지([[81]]) |
| 태스크별 부호표에서 손실 > 이득 | 배선 철회. 끄지 말고 조건을 조정한다([[19]] · [[70]]) |

---

## 7. 실행 순서 (체크리스트)

```
[ ] 0. 진행 중 97 캠페인 완주 · 전 태그 회수 (gz -> sim_results -> git add -f -> ls-files 확인)
[ ] 1. P0  종결-후 표면화 실재 확인            <- 여기서 폐기될 수 있다
[ ] 2. P1  D1 격리 (+ 부정통제)
[ ] 3. P2  D2 격리 · K 결정
[ ] 3b. P3 D3/D4 격리 (apply_op 반환형 · criteria 부합 수 · D4 단독 기여)
[ ] 3c. P4 L1 격리 (언제 꺼졌나 · 재발행하는가 · CAP fail-open)
[ ] 3d. P5 파생값 17칸 측정 (P5-iso: KB 축자를 앞쪽에 두고 격리 · 부정통제) — **수리 아님**
[ ] 3e. P6 D6 격리 (상태 변화 후 재-write 가 통과하는가 · 진짜 중복은 여전히 막히는가)
[ ] 4. 통과분만 배선 + go_stack.sh 등재 + 단위테스트
[ ] 5. 스모크 게이트 5칸
[ ] 6. x509 큐에 단계 등재 (정본 갱신 — 새 문서 만들지 마라)
[ ] 7. 97 A/B 런 (배치: 엔진당 레인 1 · MAXB 1 · **비행 컨텍스트 합 <= 171,749**)
       7a. 태스크별 최대 컨텍스트를 회수된 로그에서 뽑아 묶음을 짠다 (무료)
       7b. 첫 배치에서 구간 prefix 적중률을 재고 §5a 의 반증 조건을 친다
[ ] 8. per-step 포렌식 -> 적대적 refutation -> 태스크별 부호표 -> 판정
```

---

## 8. 이 캠페인의 기준선 (2026-09-04 06:07 시점 · 진행 중)

```
arm=viewmax2 · 2026-09-03 이후
  완료 sim 55 · 고유 태스크 53/97 · pass 33 (채점분의 62%)

관측된 배치 병리 (재발 방지 대상)
  포트당 비행 토큰이 예산의 4.4배 -> Waiting 5 · KV 94% · 생성 20.5 tok/s 고착
  과부하 런 하나를 빼자 같은 엔진이 KV 30% · Waiting 0 · 42.6 tok/s
  ★그리고 그 값의 정체는 prefix 캐시였다 — 비행 합 2.0배에서 구간 적중률 3.5%,
    1.1배에서 56.4% (§5a). conc 4 로 발사해도 Running 은 늘 2 였다 — 엔진이 못 돌린 게
    아니라 매 턴 90k 를 다시 계산하느라 못 나아갔다.
```
