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

### 1c. 새 실패 3건 정밀 포렌식 (2026-09-04 09:00 · 059 · 064 · 088)

> **채점 단위 선언 ([[69]] ①)** — 세 태스크 모두 `reward_basis: ['DB']` 이고 셋 다 `db_check.db_match=false` ·
> `nl_assertions=null` · `env_assertions=[]` 다. 따라서 **아래 서술의 실패 단위는 전부 DB 변이 집합
> (MISSING / WRONGARG / EXTRA / DUP / MATCHED)** 이고, 정본 `t2_forensic.mutation_diff(sim, tag=TAG)` 로
> 산출·재현했다. `action_checks`(059 1/6 · 064 2/4 · 088 3/17 실패)는 **어디를 볼지의 지도일 뿐 실패 귀속이
> 아니다** — §1b 머리말의 경고가 그대로 적용된다. 아래에서 action_checks 수치는 한 번도 실패 단위로 쓰지 않았다.
>
> **절차** — 태스크마다 per-step 포렌식 1회 + **적대적 반증 1회**를 돌렸다. 반증에서 무너진 주장은 **지우지
> 않고 "⛔철회" 로 남긴다**([[73]] · §1b-refute 와 같은 규율). 각 문장에 **[CONFIRMED] / [PLAUSIBLE] /
> [미판정]** 을 붙였다.
>
> **분모 주의** — 태그는 `bank_k8141med1_20260903_2256` (results.json 12 sim)이며 §1b 의 20건과 **합산하지
> 마라**. 두 집계가 겹치는지는 **[미판정]**.
>
> **라벨 충돌 주의** — 088 반증 문서가 자체 라벨 `D1~D3` 을 썼는데 이 계획서의 `D1~D6` 과 **다른 것**이다.
> 이 절에서는 그 항목들을 **D8·D9** 로 재명명해 인용한다.
>
> **시간 수치 주의 ([[83]] · [[54]])** — ours 는 `Concurrency: 4`(`bank_k8141med1_20260903_2256.log` 축자
> *"Save: bank_k8141med1_20260903_2256  Concurrency: 4"*), base x644 는 축자 *"규격 : alltools · seed 300 ·
> max-steps 200 · timeout 7200 · **concurrency 1** · port 8143"* 다. **벽시계 분은 배선 비용과 배치 조건이
> 섞인 값이므로 원인 진술에 쓰지 않는다.** 아래 분 수치는 §5 ② 축의 참고값으로만 적는다.

---

#### 1c-0. 세 건 공통 (교차 확인된 것)

| 사실 | 등급 | 근거 |
|---|---|---|
| 세 건 모두 **검색 결손이 아니다** — 필요한 KB 문서가 궤적에 배달됐는데 값·건수·전달이 어긋났다 | **CONFIRMED** | 059 msg32(두 절차 문서) · 064 msg31(`check_card_application_fit` 로스터) · 088 `doc_..._031` **4회 배달**. [[79]] *"Q38 의 잔여는 retrieval 이 아니다"* 와 정합 |
| **우리 층 거절이 gold mutating 호출을 막은 사례 0건** | **CONFIRMED** | 059 deny 3건 전문 · 064 deny 8건 전수 + 형제-통과 대조 · 088 deny 6건 전수(`[OPERATOR-SCOPE]` 는 **지연**시켰고 turn 67 축자 *"[T2_RESOLVE] operator-scope 상한 초과(2회) — 통과시킨다"* 로 통과) |
| 그러나 **"우리 층이 값을 저작하지 않았다"** 는 059·064 에서만 유지되고 **088 에서는 무너졌다** | **CONFIRMED** | 088 msg 66 = 우리 claimprov 재생성 산출물(1256B, 바이트 일치) — 1c-3 참조 |
| 세 건 모두 `max_tokens=8192` 상한 미충돌 ⇒ [[82]] 폭주 아님 | **CONFIRMED** | 059 최대 completion **4,998**(`…log:5714` `gen=4998 prompt=58499`) |
| 세 건 모두 **절차 체크리스트 정체·readloop 없음** (§1 의 task_049 병리가 재현되지 않음) | **PLAUSIBLE** | 059 msg62→66 절차 소진 · 088 KB 검색 15회 **질의 중복 0**. ⚠같은 포렌식이 다른 칸(재생성 카운트)에서 오류를 냈으므로 등급을 낮춘다 |

---

#### 1c-1. task_059 — `account_class` 한 인자 (sim `task_059#s626729` · n=1 · 72 msg · 291분)

**실패 단위 [CONFIRMED]** — `MISSING 1 · WRONGARG 1(같은 gold 행의 짝) · EXTRA 0 · DUP 0 · MATCHED 2 · BLOCKED 0`.

```
MATCHED  msg51 log_verification              -> "Verification logged successfully. - User: Casey Rivera (ID: cr59b4d8e3)"
MATCHED  msg56 apply_for_credit_card(card_type="Silver Rewards Card")
WRONGARG msg68 open_bank_account_4821 account_class = "Green Account (savings)"   (gold "Green Account")
```
그 호출은 **거절되지 않고 실행됐다** [CONFIRMED] — msg69 축자: *"Bank account opened successfully! - Account ID:
f9386249cd4ade09 - Account Type: savings - **Account Class: Green Account (savings)** - Status: OPEN"*.
⇒ 이 태스크의 DB 실패 전체 = 문자열 `" (savings)"` 6글자.

**연쇄의 머리 [수정됨]** — **msg 40** 이다(msg 68 아님). 28 메시지 앞에서 이미 같은 문자열을 쓰고 출처를 KB 문서
id 로 자인했다:
```
msg40 get_correct_savings_apy {"savings_account_type": "Green Account (savings)",
      "source": "Green Account (savings) FAQ: '... 4.0%' (doc_savings_accounts_green_account__savings__005)"}
```
그리고 그 표기는 **유추가 아니라 KB 의 상품명 축자**다 [CONFIRMED] — msg3 *"doc_savings_accounts_green_account__savings__001.md
- **Green Account (savings)** specifications and requirements"*, msg7 표제 *"# Green Account (savings) specifications
and requirements"*, msg4 *"3. Evergreen Account (checking) + **Green Account (savings)**"* (msgs 3·4·7·9·10·13).
`"Green Account"` 와 `"Green Account (savings)"` 가 **둘 다 KB 에 실재**해 KB 접지로는 가를 수 없다.
⇒ 남는 진술: **표기 직렬화 분산** [PLAUSIBLE]. 양쪽 런 어디에도 선택 이유가 남지 않았다 [CONFIRMED].

**우리 층 개입 [수정됨]** — *"손대지 않았다"* 는 **문장 그대로는 거짓**이다. 문제의 생성(trace turn 65) 직전·직후로
우리 문장이 뷰에 들어갔다:
```
trace turn=63 / 65 / 67   [T2_FB_VIEW] 1 queued feedback item(s) injected in view
t2_gate_patch.py:8939-8956  _t2_view_fb 를 UserMessage 로 작업버퍼에만 부착(비커밋)
큐잉 원천 = T2_LEDGER (trace turn 61 · T2_LEDGER_VIEW_KEEP=3 -> 63·65·67 정확히 3회)
```
그러나 **주입된 내용은 값과 무관하다** [CONFIRMED] — 정본 `t2_ledger.facts_text` 를 실제 rows 로 렌더한 전문에
`account_class`·상품명·괄호 언급이 **0회**다(*"[COMPUTED FACTS] Counted from the accounts above (arithmetic, not a
recommendation): 1 account(s). …This is elapsed time only. It is NOT a threshold."*). 그 턴의 우리 층 발화 전량
(`T2_SUBWIN` · `T2_SUBCALL cache hit` · `T2_SIBLING_PAREN`(print) · `T2_A2_VARIANT`×2 · `T2_FB_VIEW`)에 인자를
고치는 경로가 없다. `attempted_mutations` 상 `open_bank_account_4821` 시도는 **msg68 단 1회 · ok=True**.
⇒ **우리 층은 `account_class` 값을 저작·변경하지 않았다** [CONFIRMED · 근거 교체됨].

**우리 층 계기 결함 (원인 아님) [CONFIRMED]** — `T2_SIBLING_PAREN` 이 이 호출을 정확히 탐지하고 고칠 값까지
이름 붙였는데(런 전체 8,300여 줄에서 **유일한 발화** · `logs/bank_k8141med1_20260903_2256.log:7764`) **집행되지
않는다**:
```
t2_gate_patch.py:13306  "★§T-8 계기 ... **거동 변화 0**. ... 반려(`deny`)는 이 수를 보고 붙인다"
t2_gate_patch.py:13311  if os.environ.get("T2_SIBLING_PAREN") in ("log","deny") ...   <- 분기는 print 한 줄뿐
arms/{t8log,t2prime,t3prime,viewmax2}.env  전부 "=log"     repo 전체 "=deny" 0건
```
인용한 코드 = 실제로 돈 코드임을 확인했다 [CONFIRMED]: `provenance.json` 이 `engine_dirty: true` 인데
`git status --porcelain | grep -v '^??'` → **0줄**(전부 untracked reports/).

**base 대조 [강등]** — x644(`sim_results/bank_x644_q38base_bank78_20260830.results.json.gz`)에서 task_059 는
**reward 1.0 · db_match True · 47 msg · 14.7분** 이고 보낸 값이 `"Green Account"` 다. ours 는 `0.0 · 72 msg ·
291분`. **그러나 "회귀" 로 쓰지 마라** — 코퍼스 전량(정본 `F.iter_all_sims(want_tasks={"task_059"})` · **14 sim**)에서
gold 문자열은 **x644 단 1회**이고, 호출을 한 7 sim 중 6 sim 은 아예 다른 상품(Diamond Elite ×7 · Platinum Savings …)을
골랐다. 이번 런은 **상품을 처음으로 맞히고 표기만 틀린 최초 사례**다. ⇒ 판정: **n=1 분산** [PLAUSIBLE], 회귀
[미판정]. (08-04~08-06 sim 은 Q2.5 레거시라 [[79]] 상 직접 비교 불가.)

**⛔ 철회 목록 (059)**

| 철회한 주장 | 왜 |
|---|---|
| *"사이드카 turn 집합 = [0,29,…,57] · turn≥58 은 0건 ⇒ 우리 층 미개입"* | 계기 파손. 33행 중 **17행이 `turn=0`이고 전부 `kind='subcall'`** — turn 이 subcall 행에 채워지지 않는다. 그 중 4행은 시간상 turn 57 이후이고, 하나는 `{"tool": "open_bank_account_4821"}` 로 **msg67 unlock 이후에만 존재하는 이름**을 담고 있다 |
| *"turn 57 < turn 68 이므로 개입 없음"* | **turn 축 3종 혼용**(msg 색인 / 사이드카 turn / trace turn). 문제의 생성은 trace turn **65** 다 |
| *"우리 층은 그 호출에 아예 손을 대지 않았다"* | `T2_FB_VIEW` 주입 3회가 사이드카·영속 궤적 **어디에도 안 남는다**. 결론은 유지하되 근거를 D1 계열(내용 무관)로 교체 |
| msg66 assistant reasoning 축자(*"…but that doesn't end in 'Account.' Hmm. … By analogy…"*) | **궤적에 존재하지 않는다.** 우리 런 assistant 28개 전부 `reasoning_content` 0B. `'By analogy'`·`"doesn't end in"`·`'Hmm'` 모두 검색 결과 `[]` |
| base x644 msg41 reasoning 축자(*"…account_type disambiguates…"*) | 동일. base assistant 중 `reasoning_content` 보유 **0개**, base msg41 은 content 0B 의 맨 도구 호출 |
| *"checking 예시로부터 유추해 술어를 뒤집었다"* | KB 가 그 상품을 그렇게 부른다(위 msgs 3·4·7·9·10·13). 유추 아님 |
| *"머리 = msg68 말단 단일 인자 오류"* | 첫 발화는 **msg40** |
| *"우리 층이 개입한 가지는 정답이 됐다(apply_for_credit_card MATCHED)"* | 그 호출은 **`role=user`** 다 — user-sim 이 실행했고 `annual_income=55000` 도 user-sim 이 채웠다(agent 는 msg55 에서 *"I left this blank for you to fill in"*). base 도 동일(`msg25 role=user`, 인자 바이트 동일) |
| *"turn 49 deny 대상 = apply_for_credit_card / check_card_application_fit"* | turn 49 의 대상은 `call_discoverable_agent_tool` 이고 사유는 `[T2_PHASE_PRECEDE] … reqs=['GB1_VERIFY_BEFORE_ACCOUNT_ACCESS']` — **gold 호출과 같은 래퍼**다. ⚠§2 (철회됨)D5 의 *"우리 층은 이 호출을 건드린 적이 없다"* 줄도 이 좁은 표현으로 읽어라 |
| *"T2_SIBLING_PAREN 이 이 결함을 막을 수 있었던 **유일한** 결정론 장치"* | 거짓. `[OFFICIAL-NAME]`/`T2_WRITE_ARG_ENUM` 계열이 존재하고 과거 fb 파일 **56개**에서 발화했다(이번 런 0건 — L1 참조) |
| 최대 completion `4,613` | 실측 **4,998**. 결론(8192 미충돌)은 불변 |

**⛔[[74]] 위반 (059 포렌식의 방법 결함) [CONFIRMED]** — `prior_checked` 12칸이 전부 `scripts/`·`logs/`·`sim_results` 이고
**`reports/` 를 한 번도 grep 하지 않았다.** 결론의 절반이 3주 전 정본에 이미 있다:
```
FAILURE_AXIS_AND_FIX_ORDER_2026_08_15.md:2478-2481 §T-8
  gold : open_bank_account_4821{account_class:"Green Account", ...}
  done : open_bank_account_4821{account_class:"Green Account (savings)", ...}
:2490  "**KB 접지** — `Green Account` 와 `Green Account (savings)` **둘 다 KB 에 실재**해 못 가른다."
:2508  ⛔W-5 (블로커) "모델이 반려를 받고도 같은 값을 다시 보낸다 ... 한 sim 최다 반복 18회 ...
        회복되지 않으면 이 레버는 **오답을 스텝 소진으로 바꾼다**"
:2521  "⛔엔진이 값을 고치지 않는다. ... 지금은 **관측+부호표**이고, 효과는 켠 팔과 안 켠 팔로 재야 한다."
DEFECT_LEVER_COVERAGE_2026_08_23.md §ⓒ
  open_bank_account_4821 WRONGARG 160건 중 **159건이 단일 인자 account_class**
  (Green Account (savings) ↔ Silver Plus Account 25 ...)
```

---

#### 1c-2. task_064 — operand 2칸(`account_class` · `card_type`) (sim `c65c88f6…` = `task_064#s626729` · n=1 · 92 msg · 301분)

**실패 단위 [CONFIRMED · 독립 재현 일치]** — `gold mutating 3 · MATCHED 1 · WRONGARG 2 · MISSING 2(같은 gold 행의
미매칭 쌍) · EXTRA 0 · DUP 0 · BLOCKED 0 · regen_blocked 8`.
```
MATCHED  msg53 log_verification            -> "Verification logged successfully. - User: Jordan Lee (ID: jl72b4e9d1)"
WRONGARG msg84 open_bank_account_4821  account_class "Silver Account"   (gold "Gold Plus Account")
         msg85 축자 "Bank account opened successfully! ... **Account Class: Silver Account** - Status: OPEN"
WRONGARG msg87 apply_for_credit_card   card_type "EcoCard"              (gold "Gold Rewards Card") · requestor=user(gold 도 user)
         msg88 축자 "Credit card application submitted: Your application has been successfully submitted."
```
두 호출 모두 **실행됐다**. 나머지 인자(`user_id`·`account_type`·`customer_name`·`annual_income`·
`rho_bank_subscription`)는 gold 와 동일 ⇒ **operand 값 선택 오류 2칸**이고 열거·전달·실행은 성공했다.

**연쇄의 머리 [수정됨]** — msg 35 송출본(689B, 권고표 소실) → **msg 47** → msg 80 → 집행. 다만 msg 47 은
**모델의 자발적 전환이 아니라 우리가 만든 턴**이다 [CONFIRMED]:
```
fb idx=27 kind=reminder-user turn=47 channel=claimprov len=306 축자:
  "Note: [CLAIM-PROVENANCE] tool ownership — the following are in YOUR OWN tool list, not the
   customer's: None (tool: verify_identity); None (tool: log_verification); None (tool: give_discoverable_user_tool)."
msg 47 reasoning_content 축자:
  "**This appears to be a system reminder/note rather than a genuine customer message.**"
  "Since there's no actionable customer request, **I'll send a brief message keeping the ball in their court.**"
```
그 턴의 채움말이 *"lock in the **EcoCard + Silver Account** combination"* 이었다. **심의된 operand 결정이 아니다.**
그리고 msg 80 송출본(1503B)은 두 금액을 **나란히** 제시했다 — *"**Reply \"1\"** → I open the **Silver Account** now
(best score-independent combo: $1,817.50 …)"* / *"**Reply \"2\"** → I open the **Gold Plus Account** now (best overall
combo: $1,905.00 …, if your score is 720+)"* — 고객은 그것을 보고 1을 골랐다(msg 81). ⇒ 머리의 최소 진술:
**닫힌 목적함수(연이자−연회비 최대화)의 operand 2칸이 우리가 점유한 턴에서 굳었고, 금액 병치 뒤에도 열린 조건
(미지 신용점수) 회피가 이겼다** [PLAUSIBLE · n=1].

**우리 층 개입 = yes [CONFIRMED]**, 단 세 갈래로 나눠 적는다.

1. **[B] `get_correct_savings_apy` grounding 이 `파일명: '인용'` 접두를 못 읽는다 — CONFIRMED, 기전 확정**
```
t2_scaffold_get.py:724     src_ok = bool(ns) and any(ns in nc for nc in norm_corpus if nc)     <- 순수 substring
t2_scaffold_get.py:121-123 _norm_ground = re.sub(r"[^a-z0-9%]+", " ", str(s).lower())
msg59 source="doc_savings_accounts_silver_account_003.md: 'At or above threshold | At least $10,000 | 4.0%'"
  -> msg60 "[GROUNDING WARNING] 3 input value(s) could not be verified ... (could not compute — check your arguments)"
msg64 접두만 제거한 **같은 인용**
  -> msg65 "Correct savings APY computed from the components supplied in this call: 6.2250000000000005%"
계약 축자 a2/banking_knowledge.gate.json:1092  "source": "<verbatim quote from the KB doc stating this value>"
```
   반증조건 [B'](코퍼스 미도달) **소거됨**: msg 59↔64 사이 새 read **0건**. 비용 = 4콜 1라운드 전손, 유효 수치
   6개를 얻는 데 14콜. 거절 문면 *"source not found in the knowledge base"* 는 **무엇을 하면 풀리는지 말하지
   않는다** ⇒ [[64]] 위반 [CONFIRMED].
2. **[A] RECOMMEND-OFFER 는 원인이 아니라 추인 — 단 근거가 바뀌었다**. 우리 서브가 gold 와 다른 값을 확언했고
   (`recommend_formalize` → `{"applies": true, "card_type": "EcoCard"}`) 명령형으로 뷰에 넣었다. 그러나 그
   산출물은 **사용자에게 전달되지 않았다** [CONFIRMED]: 621B 문면이 궤적에 0건, `give_discoverable_user_tool`
   영속 호출 0건, `driver.log:8255 [T2_RESOLVE] recommendation-offer deny`, 송출 msg 86(634B)에 카드명 0회.
3. **[N2] 같은 턴에서 우리 층 두 문면이 시키고 막았다 — CONFIRMED · [[55]] 대상**
```
8227 [T2_RESOLVE] recommendation-offer deny  -> "Offer it now by calling 'give_discoverable_user_tool' … card_type='EcoCard'"
8235 agent_response_unified_regen … tool_calls=1                 <- 시킨 대로 호출
8247 [T2_TOOL_SIGNATURE] would-deny tool=give_discoverable_user_tool but preempted-by=resolve_write
8249 [T2_TOOL_OBS] err=True -> "Error: [ACTION] 'apply_for_credit_card' is run by the CUSTOMER, not by you."
8255 [T2_RESOLVE] recommendation-offer deny                       <- 또 같은 요구
8263 [T2_MATERIAL_GATE] stop=resolve_cap(정체 3회) turn=86
```
   루프는 판단이 아니라 **정체 캡으로만** 멈췄다.
   부수: **[N3]** `[ARG-EMPTY]` 거절(fb idx=6, turn=30)이 날조를 유도했고(msg30 reasoning 축자 *"I shouldn't
   fabricate constraints. **But the tool requires non-empty values.**"*, args `"credit_score":"850"`), 우리
   grounding 이 msg31 에서 다시 드롭했다(*"credit_score=850 (the customer never mentioned this kind of
   requirement — do not add limits they did not state)"*). [F](우리 도구가 Gold 우세 근거를 쥐어줬다)는 사실이나
   **거절→날조→드롭 1왕복 뒤**에 얻은 것이다.

**gold 호출 차단 여부 [CONFIRMED · 형제 통과 전수 대조]** — 0건. 같은 `[READ-FIRST]` 게이트가 msg40 4콜을
막았으나 msg59 4콜·msg64 4콜·msg77 2콜을 통과시켰고, `[ACTION]` 라우팅은 gold(`064_1 requestor=user`)와
**일치**하며 사용자가 msg87 에 실제 실행했다. turn 51 의 `resolve the flagged call(s) first` 는
`GB1_VERIFY_BEFORE_ACCOUNT_ACCESS` 선행이고 이후 msg53 MATCHED · msg84 실행 성공.

**base 대조 [CONFIRMED · 회귀]** — x644 `task_064` = **reward 1.0 · 68 msg · 20.25분(1215.13s) · 같은 시드
626729 · 첫 사용자 발화 동일**. base 결정 축자(msg 28): *"**My recommendation: Gold Plus Savings + Gold Rewards
Card**"* → user msg 29 *"…let's do it"*. ours 는 `0.0 · 92 msg`. **pass→fail 성립**.

**⛔ 철회 목록 (064)**

| 철회한 주장 | 왜 |
|---|---|
| *"검증 패스(`recommend_operand_verify`)도 EcoCard 를 통과시켰다"* | **독립 검증이 아니다.** 두 프롬프트(plen 5935 / 5900)는 35자만 다른 **같은 재료**이고, `Option details` 블록에 `annual_fee`·`min_score`·`cashback`·`"Gold Rewards"`·`"Gold Plus"` 가 **각 0회**다. 앞선 "none" 6회도 카드가 0회 등장하는 프롬프트에서 나온 **강제값**이다 ⇒ [[78]] 재료 결손 |
| *"그 값이 그대로 W2 가 됐다"* | 산출물 **미전달**(위 [A]). 결론('추인')은 유지, 근거 교체 |
| *"월클럭 14.9배"* | **배치 산물**(conc 4 ↔ 1). 배치 불변 지표로 다시 재면 `ours comp 52,800 tok / prompt 2,137,899` ↔ `base comp 21,267 / prompt 2,338,132` ⇒ **생성 2.48배, 프롬프트는 오히려 우리가 적다**. `msg 92↔68` 과 `reward 1.0→0.0` 만 유효 |
| 계기 수치 *"생성 45콜 · 재생성 31% · recommend_formalize ×7"* | 실측 **79콜 · 재생성 13/79 = 16.5% · recommend_formalize 실생성 5회**(사이드카 8행 중 캐시 3) |
| msg 80 축자(*"…$87.50 more…"*) | **폐기된 초안**(fb idx=39, len=3117)의 문장이고 송출 msg 80 에는 없다. 송출본은 두 금액 병치(위) — 결론은 오히려 강화 |
| *"msg 47 에서 에이전트가 수치 하나 없이 스스로 전환했다"* | 그 턴은 **우리 claimprov note 가 점유한 턴**이다(위 축자) |
| *"[T2_CLAIM_PROV] 오작동 1건"* | **3건**(idx 17 turn 35 · idx 27 turn 47 · idx 43 turn 80) |
| *"regen 이 정답 초안을 2회 폐기 · EcoCard 대체안은 regen 산물"* | 폐기는 **3회**(turn 30·35·80)이고, turn 35 **초안 자체**가 이미 *"(The EcoCard has no score requirement, which is why it's the fallback.)"* 를 담고 있었다 ⇒ 대체안 프레임은 regen 이 만든 게 아니다. [C] 는 **상관까지만** |

---

#### 1c-3. task_088 — EXTRA 1 + WRONGARG 2 (sim `task_088#s626729` · n=1 · 93 msg · 242.2분)

**실패 단위 [CONFIRMED · 독립 재현 일치]** — `GOLD 4 · DONE 5 · MATCHED 2 · MISSING 2 · WRONGARG 2 · EXTRA 1 ·
DUP 0 · BLOCKED 0`.
```
MATCHED  msg13 log_verification · msg74 close_debit_card_4721(reason=fraud_suspected)
WRONGARG msg70 file_debit_card_transaction_dispute_6281  (17 인자 중 15 일치)
           transaction_type              gold 'signature_purchase' ↔ 'pin_purchase'
           customer_max_liability_amount gold 50                   ↔ 500
           실행 증거 msg71 "Dispute ID: dsp_76b0f2bc26c3 … Provisional Credit: ISSUED - $347.99"
WRONGARG msg78 order_debit_card_5739
           delivery_option              gold 'STANDARD' ↔ 'EXPEDITED'
           excess_replacement_fee       gold 미선언   ↔ 0 명시           (DB 영향 [미판정])
           실행 증거 msg79 "Debit Card Order Confirmed … Delivery Option: EXPEDITED"
EXTRA    msg49 transfer_funds_between_bank_accounts_7291(blue->green, 100)  gold 17 action 어디에도 없음
           실행 증거 msg50 "Transfer completed successfully! - Amount: $100.00 - From: chk_..._blue
                            (new balance: $1150.00) - To: chk_..._green (new balance: $480.47)"
```
`transfer_to_human_agents`(088_16)는 **`mutating_tools`(44종) 밖**이라 DB 단위에 들어오지 않는다 [CONFIRMED ·
`F.mutating_tools()` 직접 확인].

**연쇄의 머리 [수정됨]** — 두 사슬이다.

- **사슬 A (EXTRA)**: msg 45 무청구 제안 → msg 46 승낙 → msg 49 실행. user 대본에 이 분기는 없다
  ([CONFIRMED] `task_088.json` notes 축자 *"informational resolution"*; msg 46 은 대본 §11 문장 *"Okay, I
  understand now. So basically I need to wait a couple more days…"* 과 즉흥 승낙을 **한 메시지에** 담고 있다) ⇒
  제안의 저자는 에이전트다([[21]]). 단 KB 가 그 제안을 허용한다 [CONFIRMED]: `doc_checking_accounts_…_006`
  CODE 51 §3 *"If they want to transfer, help them do so."*
  ⚠**그 턴의 실행 경로는 우리가 코치했다** — 아래 철회 참조.
- **사슬 B (`delivery_option`)**: **msg 66 은 모델의 작문이 아니라 우리 claimprov 재생성 산출물이다**
  [CONFIRMED · 바이트 일치].
```
trace turn 65
  [T2_GEN_TRACE] call=agent_response          -> gen=2059 reason=5476B **content=2325B** tool_calls=0
  [T2_CLAIMPROV] window hit(resign) claims=12 unbacked=0 pending=3 **unb_p=3 [None, None, None]**
  [T2_CLAIMPROV] owner split: agent=0 user=0 **unknown=3**
  [T2_GUIDED] guided applied (call=agent_response_claimprov tools=27)
  [T2_GEN_TRACE] call=agent_response_claimprov -> gen=966 reason=2932B **content=1256B** tool_calls=0
  [T2_CLAIMPROV] regen tool_calls=[]
궤적 msg 66 = **1256자**  (원본 2325B 는 폐기 · -46%)
```
  그 재생성본이 배달 메뉴에서 **STANDARD 를 통째로 빠뜨렸다**: *"3. **Delivery speed:** Free expedited (3–5
  business days) or Rush (1–2 business days, $35)?"* → 대본 §18 *"Just the standard option is fine."* 인 user 는
  준 메뉴에서 고를 수밖에 없었다(msg 67) → msg 78 `EXPEDITED`. 그리고 그 메시지의 "안 했다 + 번호 매긴 재질의"
  골격은 우리 문구가 축자로 요구한 것이다 — reminder-user turn 66: *"Do not end your involvement by describing the
  work as done or under way — either call the tool now, or **state explicitly that it has NOT been performed.**"*
  ⚠**폐기된 2325B 원본에도 STANDARD 가 없었는지는 원리적으로 확인 불가**(→ D9) ⇒ 이 자리는 **[미판정]** 으로
  남긴다. 원 포렌식이 *"모델의 열거 결손"* 으로 **닫은 것 자체가 잘못**이다.

**우리 층 개입 = yes [철회된 판정]** — 아래 참조. 다만 **거절이 gold 호출을 막지는 않았다** [CONFIRMED]:
`[OPERATOR-SCOPE]`(turn 68)의 대상은 gold `088_9` 의 unlock 이었으나 turn 67 축자 *"[T2_RESOLVE] operator-scope
상한 초과(2회) — 통과시킨다"* 로 한 턴 **지연 후 통과**했고(msg69 *"Tool unlocked: …"*), unlock 은 GRANTS 라 DB 를
바꾸지 않는다. `[POLICY GATE GB2_NOTICE_BEFORE_TRANSFER]`(turn 85)도 이전을 막지 않았다(msg90 *"Transfer
successful"*).

**값이 어긋난 자리의 KB 출처 (gold 역산 없음 · [[23]] 준수) [CONFIRMED]** — 필요 문서는 배달됐다
(`doc_bank_accounts_…_031` **4회**):
```
_031  "- Reported within 2 business days of statement: Maximum liability $50
       - Reported within 60 days of statement: Maximum liability $500"      -> 모델은 500 선택
      msg55 축자 "That charge was on **11/09/2025** — about **five days ago**."  <- 거래일 앵커
_031  "9. transaction_type ... 'pin_purchase': In-store purchase with PIN
                             'signature_purchase': In-store purchase with signature"
      같은 호출에 pin_compromised='no' 를 넣고 msg58 에서 "the classic signature of a **cloned (counterfeit) card**"
      라고 진단해 놓고 pin_purchase 를 골랐다 (자기 인자와 모순)
_029  "PREMIUM TIER: ... (delivery_fee: $0 for both STANDARD and EXPEDITED) - Rush shipping ... ($35)"
      => EXPEDITED $0 은 **정책상 허용**된다. 정책 위반이 아니라 메뉴 결손이다
```
*"pin_compromised='no' ⇒ transaction_type≠'pin_purchase'"* 를 세울 KB 축자는 **없다** ⇒ 닫힌 술어 후보로
올릴 수 없다 [미판정 · [[23]]].

**base 대조 [CONFIRMED · 회귀 아님]** — x644 `task_088#s626729`(같은 시드) = **reward 0.0 · db_match False ·
61 msg · 9.3분(556.49s)** 이고 변이집합은 `MISSING 3 · WRONGARG 0 · EXTRA 0 · MATCHED 1` — dispute·close·order 를
**한 건도 실행하지 않았다**. ⇒ *"base 는 됐는데 우리가 깨뜨렸다"* 는 성립하지 않는다. [[79]] 그대로 **행동 부재
(base) → 값·건수 어긋남(ours)** 으로 이동했고 reward 는 둘 다 0.0. 대가는 msg 61→93, 생성 호출 110회 ·
프롬프트 누적 4.07M 토큰 [PLAUSIBLE — 이 두 수치는 재검증하지 않았다].

**⛔ 철회 목록 (088)**

| 철회한 주장 | 왜 |
|---|---|
| **`our_layer_involved: "no"`** | 판정 자체가 무너졌다(아래 두 줄) |
| *"msg 66 의 불완전 열거 = 모델의 열거 결손([[49]])"* | msg 66 은 **우리 claimprov 재생성본**(2325B→1256B, 바이트 일치)이고, 메뉴 결손은 그 안에서만 존재한다. 원본 확인 불가 ⇒ **[미판정]** |
| *"msg 45–49 구간에 우리 층 개입이 0건"* | **거짓.** fb reminder-user turn=47 channel=`channel` 축자 *"Error: [TOOL-CHANNEL] `transfer_funds_between_bank_accounts_7291` has not been unlocked yet. Call `unlock_discoverable_agent_tool(...)` first…"* + trace turn 46 `[T2_TOOL_CHANNEL] pre-call regen` ⇒ msg 47 = **0자 · tool_calls=1(unlock)** 이 우리 재생성 산출이다 |
| *"재생성 6회(모두 unified_regen)"* | 실측 **12회 · 11턴 · 4채널**(unified_regen 7 · claimprov 3 · channel 2). 교체 수지 4건 전부 바이트 일치 확인 |
| *"EXTRA 하나만으로 db_match=False 가 확정된다"* | 두 계좌 잔액이 실제로 바뀐 것은 확정이나, 세 변이 각각의 해시 기여도는 **재실행하지 않았다** ⇒ **[미판정]** (강한 [PLAUSIBLE]) |
| *"우리 층이 EXTRA 를 만들었다"* 로 읽힐 여지 | 만들지 않았다 [CONFIRMED · 형제 통과]: `T2_TOOL_CHANNEL` 은 EXTRA transfer(turn 47)와 **gold `close_debit_card_4721`**(turn 72, 같은 문면)에 똑같이 발화했고 둘 다 통과했다. 우리가 한 것은 **실행 경로 코치**이지 변이 저작이 아니다. ⚠단 `T2_CLAIM_PROV` 에는 **형제가 없다** — acting 구간에서 발화 조건이 성립한 자리는 turn 65 하나뿐이고 그것이 정확히 어긋난 메시지 위에 떨어졌다 |
| turn 68 `[BLOCKED]` 2건 = unlock 이라는 판정 | 추론으로 강등. 근거는 여전히 강하다(content=205B *"Let me get all three tools ready and execute everything."* · `tool_calls=3` = 잠긴 3종 · unlocked 카운터 6→7→8 한 턴에 하나). 그러나 **폐기된 tool_calls 인자 원문이 존재하지 않으므로**(D9) 원 보고서의 반증조건 #1 은 **원리적으로 시험 불가** |

---

#### 1c-4. 수리 후보에 미치는 영향 (D1 · D2 · D3 · D4 · D6 · L1)

| 후보 | 이 3건의 효과 | 근거 | 등급 |
|---|---|---|---|
| **D1** (종결 후 표면화 중지) | **중립~약화** — 세 건 어디에도 §1 의 절차 정체·표면화 루프가 **재현되지 않았다**. D1 의 적용 폭은 캠페인 전체가 아니라 049 계열로 좁다 | 059 msg62→66 절차 소진 · 088 KB 질의 중복 0 · 세 건 모두 `readloop-turn` 집계 없음 | PLAUSIBLE |
| **D2** (읽기 루프에 이름과 출구) | **약화** — 064 의 `get_correct_savings_apy` 14콜 중 6콜만 유효한 낭비는 **읽기 루프가 아니라 grounding 접두 드롭**이 원인이다. D2 는 이 칸을 사지 못하고 **D7 이 산다** | `t2_scaffold_get.py:724` · msg59↔64 대조 | CONFIRMED |
| **D3** (reference-filter 문면·술어 일치) | **그 게이트 자체는 미발화(중립)**, 그러나 **상위 계열 주장이 강화**된다 — *"deny 문면이 검사한 것과 다르거나 처방을 못 준다"* 의 **새 독립 사례 2건**: ① 064 grounding *"source not found in the knowledge base"*(무엇을 하면 풀리는지 없음) ② 088/064 claimprov *"None: None"*(이름 자체가 없음) | 위 축자 · [[64]] | CONFIRMED (계열) / 미판정 (D3 본체) |
| **D4** ([BLOCKED] 을 의존 호출로만) | **약화** — 이번 3건의 동반 차단은 **gold 를 죽이지 않고 지연만** 시켰다. 088 turn 68(unlock 3콜 → 한 턴에 하나씩 3턴에 전부 성공) · 064 turn 40(4콜 → msg59/64/77 에 재발행 통과). ⇒ D4 가 사는 것은 **턴 수**이지 reward 가 아닐 수 있다. 게다가 **D4 의 효과 측정 자체가 D9 에 의존한다**(폐기된 인자 원문이 없으면 무엇이 막혔는지 사후 확인 불가) | 위 · trace turn 67·69·73·77 | PLAUSIBLE |
| **D6** (중복 억제를 선언된 write 로 한정) | **중립 + ⊖ 신호 1개.** `[DUPLICATE-WRITE]` 는 이 3건에 미발화다. 그러나 **088 의 EXTRA 1건이 DB 채점을 무너뜨릴 수 있음을 보였다**(잔액 2계좌 실제 변경). D6 는 deny 를 309→88(**221건 소멸**)로 여는 레버이므로, §D6 선행조건 3(태스크별 부호표)의 **필요성이 커졌다** — "보호 상실"의 실물 형태가 EXTRA 다 | mutation_diff(088) · §2 D6 반사실 재현 | PLAUSIBLE (기여도 미분리 · [미판정]) |
| **L1** (꺼진 열거 레버 조사) | **전제 강화 · 기대수익 약화.** 강화: 이 런에서도 `[OFFICIAL-NAME]`/`T2_WRITE_ARG_ENUM` 발화 **0건**이고 `provenance.json.levers_on` 195개에 이름이 **없다**(과거 fb 파일 **56개**에는 있다) ⇒ [[81]] 배선 회귀 **CONFIRMED**. 약화: `x509_axis_queue_2026_08_24.json:183` 축자 *"모델이 낸 `account_class` 69건 중 집합 안 68(98.6%) … 게이트가 겨누는 것은 1.4% 뿐"* 이고 `"Green Account (savings)"` 는 **KB 에 실재하는 이름**이므로 켜져 있었어도 059 를 통과시켰을 공산이 크다 | 위 · 059 msgs 3·4·7 | 전제 CONFIRMED / 059 무효 PLAUSIBLE (`write_arg_enum[0].group_map` 실물 미확인 = **[미판정]**) |

**⇒ P4(L1 격리)의 exit 를 한 칸 늘려라**: *"`group_map` 에 `"… (savings)"` 형 변형이 들어 있는가"* 를 먼저
찍는다. 들어 있으면 L1 은 059 를 못 산다(그 자체가 폐기 사유는 아니고 **기대수익 0** 기록).

---

#### 1c-5. 새 후보 — D7 · D8 · D9(계기) · L2(조사) · 4칸 계약

> ⛔[[62]]: 넷 다 **격리 프로브 전에는 배선하지 않는다**. D9 는 레버가 아니라 **원장**이고, 나머지 셋의 판정
> 가능성이 D9 에 걸려 있으므로 **먼저 한다**.

##### D7 — grounding 출처 검사가 `파일명: '인용'` 접두를 삼키지 못한다 (**CONFIRMED 우리-층**)

1. **주장 + 양화** — n=1 sim(`task_064#s626729`) · 4콜 1라운드 전손 · 성분 3개 드롭. 같은 대화·같은 도구·같은
   축자 인용에서 **접두 유무로만** 결과가 갈렸다. 형제 통과 10콜이 있으므로 표적은 **문자열 형식 하나**로 좁다.
2. **근거 (축자 + 파일:줄)** — `t2_scaffold_get.py:724` `src_ok = bool(ns) and any(ns in nc for nc in norm_corpus
   if nc)` · `:121-123` `_norm_ground = re.sub(r"[^a-z0-9%]+", " ", str(s).lower())` · msg59 → msg60 *"[GROUNDING
   WARNING] 3 input value(s) could not be verified … (could not compute — check your arguments)"* · msg64(접두만
   제거) → msg65 *"Correct savings APY computed … 6.2250000000000005%"* · 계약 `a2/banking_knowledge.gate.json:1092`
   *"source": "<verbatim quote from the KB doc stating this value>"*.
   **규칙(2단)**: ① **문면 수리(무조건)** — 거절문에 요구 형식을 축자로 넣는다(무엇이 틀렸나 + 무엇을 하면
   풀리나 · [[64]]). ② **술어 완화(격리 후)** — 검사 전에 선행 `"<파일명>: "` 접두를 제거하는 정규화. 도메인
   리터럴 0, 값 선택 0([[59]] · [[23]]).
3. **반증 조건** — 격리에서 (i)접두형·(ii)순수 인용을 같은 components 로 넣어 **둘 다 통과하면** 이 귀속은
   거짓이다(원인은 다른 요인). / msg59↔64 사이에 새 read 가 있었다면 "코퍼스 미도달" 가설로 돌아간다(**이미
   소거: 0건**). / 접두형을 권장하는 계약 문구가 실제로 없다면 ②는 폐기하고 ①만 남는다.
4. **선행 확인** — `t2_scaffold_get.py:121-123·724` · `a2/banking_knowledge.gate.json:1092` · 형제 통과 6콜
   (msg64 4 · msg77 2) · [[22]] 근거-우선 formalize · [[64]] · §1b-refute(문면-술어 불일치 계열).

##### D8 — claimprov 는 **식별 불가 항목으로 발화하지 않는다** (`None` 금지) (**CONFIRMED 우리-층 · 런 전역**)

1. **주장 + 양화** — 이 태그 전역 8 태스크에서 `T2_CLAIMPROV window hit` 64행 중 발화 후보(`unb_p≥1`) **22행이고
   22/22 전부** 항목 목록에 `None` 을 담는다(`owner split` 20행 중 `unknown≥1` 이 12행). 실제 전송 문면 6건
   확인(088 turn 5·66·82 · 064 idx 17·27·43).
2. **근거 (축자)** — trace turn 65 `[T2_CLAIMPROV] window hit(resign) claims=12 unbacked=0 pending=3 unb_p=3
   **[None, None, None]**` · `owner split: agent=0 user=0 **unknown=3**` / 사이드카 전송 문면 *"…the conversation
   ledger shows it was never actually executed: **None: None; None: None; None: None**."* / 064 fb idx=27 turn=47
   *"…are in YOUR OWN tool list, not the customer's: **None** (tool: verify_identity); …"*.
   **동반 관측(같은 항목의 2차 피해)**: 그 문면이 **user 롤을 점유**해 에이전트가 없는 사용자 턴을 상상하고
   허위를 발화했다 — 064 msg47 reasoning *"This appears to be a system reminder/note rather than a genuine
   customer message."*, msg86 reasoning *"the user's message is empty… This probably means the user executed…"* →
   송출 *"Thanks for running it."*(사용자는 그때 실행하지 않았다 — 실행은 msg87).
   **규칙**: 식별 가능한 항목이 0건이면 **침묵한다**(닫힌 술어: `owner=unknown` 또는 `kind=None` 만 남으면 발화
   금지). 목적 미달성도 함께 관측됐다 — 088 turn 65 `[T2_CLAIMPROV] regen tool_calls=[]` (0→0), 얻은 것은 **-46%
   압축**뿐이다.
3. **반증 조건** — `None` 이 로그 렌더링만이고 모델에 간 실제 문면엔 실명이 들어갔음을 보이면 거짓(**반대 증거**:
   사이드카에 **저장된 전송 문면 자체**가 `None: None` 이다). / 침묵시킨 팔에서 unbacked claim 이 늘고 reward 가
   내려가면 억제 자체는 필요했다는 뜻이므로 규칙을 "침묵"에서 **"이름 확보 후 발화"** 로 좁혀야 한다.
4. **선행 확인** — [[64]] 거부는 이름을 대야 한다 · `t2_gate_patch.py:9837`(WORK-INCOMPLETE 인접) ·
   `fb_bank_k8141med1_20260903_2256.jsonl` · `trace_bank_k8141med1_20260903_2256.jsonl` · §1b-refute 의 문면 계열.

##### D9 — 재생성 폐기 원문을 **모든 채널**에서 원장에 남긴다 (계기 · 배선 자격의 선행조건) (**CONFIRMED**)

1. **주장 + 양화** — 이 태그 전역 `reminder-assistant`(폐기 원문 보존) **53행 / 53행 전부 `channel=unified_regen`**
   인 반면 `reminder-user` 59행의 채널은 `unified_regen 22 · claimprov 18 · usertoolnote 3 · channel 3 ·
   selfdecl 2 …` 다 ⇒ **claimprov·channel 재생성은 폐기 원문을 한 줄도 남기지 않는다.**
2. **근거 (축자 + 위치)** — 088 trace turn 65 `content=2325B` → `agent_response_claimprov … content=1256B`
   (= 궤적 msg 66 **1256자**, 바이트 일치)인데 **2325B 의 내용은 어디에도 없다** / 088 turn 46 `content=44B
   tool_calls=1` → `agent_response_channel … content=0B tool_calls=1`(= msg 47) / 088 turn 68 `[BLOCKED]` 2건의
   폐기 인자 원문도 없다. **귀결**: 1c-3 의 두 자리가 **확증도 반증도 불가**가 됐고, 059 포렌식이 존재하지 않는
   reasoning 을 인용한 것도 같은 구멍(영속 궤적 `reasoning_content` 0B)에서 나왔다.
3. **반증 조건** — claimprov·channel 재생성의 폐기 원문이 다른 산출물(`driver.log` 의 본문 덤프 등)에 **전량**
   남아 있음을 보이면 이 결손은 없다.
4. **선행 확인** — [[76]](서브는 진리다 — 검증 가능해야 한다) · [[70]] 판정 의무 3종 · `x509_axis_queue…` §방법_교훈
   *"레버 원장 상설화"* · `fb_/trace_` 채널 분포 실측.

##### L2 — `recommend_formalize` 격리 (서브가 **확언으로 오답**을 낸 2건) (**조사 · 레버 아님**)

1. **주장 + 양화** — n=2 sim. 059 사이드카 row0·row3 → `{"applies": true, "card_type": "Gold Rewards Card"}`
   (gold = Silver Rewards Card)이고 turn 29 에 **명령형으로 뷰 주입**: *"…'card_type=Gold Rewards Card' is the
   match. **Offer it now** by calling 'give_discoverable_user_tool' …"*. 064 #31·#32 → `EcoCard`(gold = Gold
   Rewards Card). 064 프롬프트 실물(plen 5935)의 `Option details` 에 `annual_fee`·`min_score`·`cashback` **각 0회**.
2. **근거** — 위 축자 + `[T2_RESOLVE] recommendation-offer deny`(059 trace turn 27·41 · 064 driver.log 8227·8255).
   059 는 모델이 무시해서 DB 가 살았을 뿐이다 ⇒ **오답 자체가 수리 대상**([[76]]).
3. **반증 조건** — 다른 sim 의 recommend 프롬프트 `Option details` 에 `check_card_application_fit` 로스터가 실려
   있으면 "재료 결손"은 국소 사고다(그때는 판단 결손). / 재료를 채운 격리에서 **여전히** 오답이면 결손이 아니라
   판단이고, [[76]] 대로 서브를 고치거나 폐기한다. / 재료를 채웠더니 정답이면 **전달 경로**만 남는다.
4. **선행 확인** — ⚠**로스터 주입을 새 레버로 올리기 전에 반드시 인용할 판정**: `x509_axis_queue_2026_08_24.json`
   `status_2026_08_24_pm.⑦유도` 축자 *"x516(후보집합)·x517(질문 프레임) **둘 다 gold 0/39** ⇒ **경로 없음**"*.
   무엇이 다른지 대지 못하면 제안 금지([[40]] · [[74]]). 그 외 `Option details` 를 채우는 **코드 경로는 아직
   grep 하지 않았다 — [미판정]**.

##### ⛔ 새 후보로 올리지 **않는** 것 (재유도 금지)

- **`T2_SIBLING_PAREN` 의 deny 승격** — `FAILURE_AXIS_AND_FIX_ORDER_2026_08_15.md` §T-8(:2476-2586)이 **이미**
  결함·KB 접지 불가·처방 후보·블로커 W-5(*"모델이 반려를 받고도 같은 값을 다시 보낸다 … 한 sim 최다 18회 …
  오답을 스텝 소진으로 바꾼다"*)까지 확정해 두었다. 승격 여부는 **§T-8 이 정한 게이트**(반대 팔 A/B + 반려 후
  괄호 제거율 부호표)를 그대로 따른다.
- **`account_class` 열거 검사** — 이미 `D5(철회됨)` → `L1` 로 처리된 자리다.
- **088 의 `customer_max_liability_amount`(①금액) · `transaction_type`(②범주) 에 표를 더 주는 처방** —
  x509 축자 *"②범주: x512 경계 판정 철회 · x513 표를 줘도 057·063 0/6"* ⇒ **이미 측정돼 실패한 경로**다.
- **088 의 이체 제안을 ⑦유도 축으로 접기** — ⑦유도는 `requestor=user` 축인데 이 EXTRA 는 **에이전트 자신이**
  실행했다. 접으면 오분류.

---

#### 1c-6. §5(비용 축)에 반영할 정정

- **② `sim 당 벽시계 분` 은 이번 캠페인에서 교란돼 있다** [CONFIRMED] — ours conc **4** ↔ base x644 conc **1**
  (양쪽 축자 확보). 배선 비용을 재려면 **배치 불변 지표**를 함께 적어라: task_064 `생성 토큰 52,800 ↔ 21,267
  (2.48배)` · `프롬프트 2,137,899 ↔ 2,338,132 (ours 가 더 적다)`.
- **③ 생성 호출 배수** — §5 의 task_064 분해(`agent_response 29 ↔ 부수 30`)는 08:15 미완 시점 값이다. **완주 후
  실측 79콜**(agent_response 32 · unified_regen 10 · claimprov 3 · source_claim_formalize 6 ·
  recommend_formalize 5 · intent_operator_formalize 5 · selfdecl 5 · agent_claimprov 5 · 기타)로 갱신하라.
  재생성 비율은 **13/79 = 16.5%**.
- **§5 표의 `task_064 ours 4.7시간째 미완`** → **완료: reward 0.0 · 92 msg · 301.0분**.

---

#### 1c-7. 아직 모르는 것 (원인 진술에 쓰지 마라)

- 088 turn 65 에서 폐기된 **2325B 원본의 내용** — D9 때문에 **영구 복구 불가**.
- 088 세 변이(EXTRA · dispute · order) **각각의 `db_match` 기여도** — db_check 해시 내부 미개봉, 재실행 미수행.
- 088 `excess_replacement_fee="0"` 명시가 DB 행을 바꾸는지.
- 059 `write_arg_enum[0].group_map` 에 `"… (savings)"` 형 변형이 있는지(→ L1 기대수익).
- 064 `Option details (from lookups)` 를 **무엇이 채우는지** — 프롬프트 실물만 봤고 코드 경로 미확인.
- `_t2_view_fb` 큐잉 원천이 059 turn 61 `T2_LEDGER` 하나뿐인지(다른 마커가 큐잉하면 1c-1 의 D1 렌더 근거가 흔들린다).

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

### D6 의 술어 — 초안을 폐기하고 좁힌다 (2026-09-04 · 사용자 지적)

**초안(폐기)**: *"직전 동일 write 이후 다른 mutating 호출이 있었으면 중복이 아니다."*
⛔**너무 느슨하다 — 진짜 중복 실행까지 열어 준다.** 폐기한다.

**대안으로 검토했다가 기각한 것**: *"인자가 바뀐 것만 통과."* ⛔**051 을 못 산다** — gold `051_2` 와
`051_7` 은 **바이트 단위로 동일**하다(§위 축자). 인자 변화는 이 자리의 판별자가 아니다.

**실제 버그는 반대편에 있다 — 억제가 선언 없이 기본으로 걸린다.**
```
t2_gate_patch.py:6121   for k in (_mut_key_of(tc), _once_key_of(tc, a2w)):
t2_gate_patch.py:6042   _mut_key_of: "변이 하나의 동일성 = 실행 이름 + **인자 전체**(문자열 접기)"
t2_gate_patch.py:6052   _once_key_of: "A2 `write_once_keys` 가 선언한 **정책의 유일성 키** (없으면 None)"
A2 축자 `_note_write_once_keys`:
   "정책이 선언한 **유일성 키**. 엔진은 이 이름들의 값을 읽어 이어 붙일 뿐이고
    **무엇이 유일한지는 여기서만 정한다**([[05]])."
```
선언은 *"여기서만 정한다"* 인데, 엔진은 **선언과 무관하게 `_mut_key_of` 도 함께 등록**한다. 그 결과
*"어떤 write 도 같은 인자로 반복될 수 없다"* 는 **정책이 말한 적 없는 유일성 규칙**이 전역으로 걸린다.
`write_once_keys` 에는 현재 `apply_checking_account_credit` **한 건만** 선언돼 있는데,
`submit_credit_limit_increase_request` 는 선언이 **없는데도** 막혔다 — 051 이 그 실물이다.

```
규칙 : 중복 억제는 **`write_once_keys` 가 선언한 write 에만** 적용한다(`_once_key_of`).
       선언이 없는 write 에 `_mut_key_of` 로 억제하지 않는다.
```
- **진짜 중복은 그대로 막힌다**: `apply_checking_account_credit` 는 선언돼 있다 —
  도구 설명 축자 *"may only be called ONCE per checking account per customer interaction"*.
- **051 은 통과한다**: 그 도구엔 유일성 선언이 없다. 정책이 반복을 금지한 적이 없다.
- **[[05]] 정합**: 무엇이 유일한지는 **선언(A2)** 이 정하고 엔진은 집합 소속만 본다([[59]]).

⚠ [[70]] 무엇을 파는가: **아직 선언되지 않았지만 반복하면 해로운 write** 가 보호를 잃는다.
완화책은 창을 여는 게 아니라 **선언을 채우는 것**이다. ⇒ **P6a 로 실제 감사했다(아래).**

#### P6a 결과 (1차) — ⛔**아래 결론은 P6c 가 반증했다. 정정은 이 절 끝에 있다.**

**주장 + 양화 (WRITE 도구 n=42 · KB 문서 전수)**: banking 도메인의 write 도구 42개 중 KB 가
유일성을 말하는 것은 **`apply_checking_account_credit_5829` 하나뿐이고, 그것은 이미 선언돼 있다.**

**근거 — 축자 + 출처**
```
유일성 문면이 있는 KB 문서 5개 · 그중 write 유일성은 1건
  doc_bank_accounts_bank_accounts_(general)_017 축자:
    "The apply_checking_account_credit_5829 tool may only be called ONCE per checking account
     per customer interaction. After a credit is applied to a checking account, the system
     enforces a 14-day cooldown period before another credit can be applied to that same account."
  => A2 write_once_keys 에 **이미 선언됨**(keys=["agent_tool_name","account_id"])

같이 걸린 나머지 1건은 write 유일성이 **아니다** — read 로 확인하라는 선행 조건이다:
  doc_credit_cards_credit_card_account_logistics_007 축자:
    "Cooldown Period: Use the get_credit_limit_increase_history_4829 tool to check if the customer
     has submitted a request within the cooldown period for their card tier."
  => 이것은 **점검(read)** 지시이지 재제출 금지가 아니다. 051 이 정확히 이 경우다 —
     정책은 "확인하라"고 하고 우리는 "하지 마라"로 막았다.
```

**보호를 잃는 write 목록 (41개 · 정책 근거 없음)**: `submit_credit_limit_increase_request_7392` ·
`approve_credit_limit_increase_5847` · `pay_credit_card_from_checking_9182` · `open_bank_account_4821` ·
`close_bank_account_7392` · `order_debit_card_5739` · `freeze_debit_card_3892` · `close_debit_card_4721` ·
`file_credit_card_transaction_dispute_4829` · `file_debit_card_transaction_dispute_6281` ·
`log_verification` · `submit_referral` · `apply_for_credit_card` … (전 41개)
⇒ **정책이 반복을 금지한 적 없는 도구들**이다. 기본 억제는 이들에게 근거가 없다.

**반증 / refutation**: 내 검색어가 놓친 표현이 있으면 이 "0개" 는 거짓이 된다. 쓴 패턴:
`only be called ONCE` · `may only be called` · `only once` · `ONCE per` · `a second time/request` ·
`cannot be called/submitted/applied again|twice` · `cooldown period` · `one per` ·
`single request/submission per` · `duplicate request/submission`. 다른 표현이 나오면 **선언을 먼저
채우고** D6 를 켠다. (⚠파이썬 docstring 만 뒤진 1차 감사는 enum 값 `duplicate_charge` ·
`cooldown_period_active` 에 걸린 **오탐 3건**을 냈고 정작 017 문서를 놓쳤다 — 출처는 **KB 문서**다.)

**선행 확인**: `grep -rln "may only be called ONCE" tau2-bench/` →
`documents/doc_bank_accounts_bank_accounts_(general)_017.json` · A2 `write_once_keys`(1건) ·
`_note_write_once_keys` · `env_surface.json`(banking 엔 유일성 문면 없음 · retail 만 2건).

#### ⛔P6a 결론 철회 — P6c 가 반증했다 (2026-09-04 · 워크플로 `wf_63c350dd`)

> 위 *"채울 선언 0개 · D6 으로 잃는 보호는 없다"* 는 **거짓이다.** 지우지 않고 남겨 둔다 —
> 내가 쓴 반증 조건(*"검색어가 놓친 표현이 있으면 이 0개는 거짓"*)이 그대로 성립했다.

**주장 + 양화 (미선언 유일성 write n=3 · 노출 12 태스크)**: banking KB 에 유일성을 말하는 write
도구가 **최소 3개 더** 있고 전부 `write_once_keys` 에 없다. 선언된 1건의 노출이 3 태스크인데
미선언 3건의 노출은 **12 태스크**다.

**근거 — 축자 + 파일:줄**
```
request_temporary_debit_card_limit_increase_8374
  doc_bank_accounts_bank_accounts_(general)_040.json:13
    "- **Frequency**: Only one temporary increase per 24-hour period per card"
  tools.py:4124  "- Only one temporary increase is allowed per 24-hour period"
  env 자체 가드: **없음** (본문 3986-4185 에 재호출 차단 분기 0)      노출 1 태스크(089)

order_replacement_credit_card_7291
  doc_credit_cards_credit_card_replacements_004.json:2
    "- You cannot submit another replacement while an existing request is still being processed."
  env 가드 부분적 — tools.py:1468 "Error: Order may have already been placed ..."
    단 `reason` 만 바꾸면 우회                                        노출 8 태스크
      (036·037·038·039·054·077·080·081)

deposit_check_3847
  doc_bank_accounts_bank_accounts_(general)_011.json:40
    "- Duplicate deposit detected: Do not attempt to redeposit."
  env 가드 **0건** ⇒ `_mut_key_of` 가 **유일한 보호**였다               노출 3 태스크(055·057·061)
```
**내 1차 감사가 놓친 이유**: 정규식에 `only once` · `ONCE per` · `one per` 는 넣었으나
**`only one` · `cannot submit another` · `duplicate deposit`** 을 넣지 않았다.

**증거 등급의 역전(부수 발견)**: 유일하게 선언된 `apply_checking_account_credit` 의
`_note_source_` 는 출처를 *"도구 설명 축자(env 출력)"* 라 적었는데 `tools.py` 에
`ONCE per checking` 은 **0건**이고 그 축자는 KB `doc_..._017:48` 에만 있다. 반면 `_040` 건은
**KB + env 양쪽**에 있다 — 선언된 사례보다 근거가 하나 더 많은 write 가 미선언 상태다.

#### D6 은 공짜가 아니다 — 반사실 재현 (실 sim n=9,468 · 2026-08-29 이후)

```
총 deny   현행 309  →  D6 88        **221건(71.5%) 소멸**
  정당한 회복: submit_credit_limit_increase_request 24 (051 계열)
  그 외:      open_bank_account 45 · update_transaction_rewards 17 · order_debit_card 15 …
once-키만 잡는 것도 실재: apply_checking_account_credit 49건 중 **18건** (mut_hit=False)
retail 도 exchange_delivered_order_items 23 · modify_pending_order_items 12 가 once 전용
```

#### P6c 좁은 질문은 PASS (실측)

리모트에서 repo 모듈을 `importlib` 로 직접 불러 돌린 프로브(`/home/woori/scratch/p6crun/x_p6c_keys.py`
· 모듈 sha `0b7d703` · HEAD 와 `t2_gate_patch.py` diff 없음 · 모델 0 · GPU 0):
```
(1) 선언 케이스 A(amount 14.5) vs B(30.0) 의 once-key 동일?   True  -> 계속 막힌다
(2) 다른 계좌 C 의 once-key 는 다른가?                        True  -> 오차단 없음
(3) 미선언 write D1/D2 의 once-key 가 None 인가?              True  -> 051 통과
★ A.mut != B.mut  — 지금의 전체-인자 키는 "같은 계좌·다른 금액"을 **못 잡는다**.
  once-키만 잡는다 ⇒ D6 는 막던 것을 잃지 않는다.
```
관련 위치: `t2_gate_patch.py:6121`(등록) · `:12278`(조회) · `:12285-12286`(문면 분기) ·
`:6065` docstring *"선언이 없으면 None 을 돌려 종전 거동(인자 전체 키)을 그대로 둔다 = fail-open"*.
⛔`_mut_key_of` **함수 자체를 무력화하면 안 된다** — `:12101` 이 `T2_WRITE_ARG_TYPE` 의 sim-당 cap
키로 별개 사용 중이고, 무력화하면 2026-08-28 에 고친 t7376 task_040 회귀가 되돌아온다.

#### 반증에서 살아남지 못한 우려들 (= 기우였다)

중첩 JSON 파싱은 **된다**(실 궤적 `apply_checking_account_credit` 호출 325/325 가 `account_id`
담은 키 생성 · `:6080-6081`) · `_a2_of` 도달함(`unlock_`/`give_` deny 0건 · `:3637-3648`) ·
등록/조회 접두가 달라 충돌 불가 · **레버는 라이브에 켜져 있다**(`/proc/<pid>/environ` 에
`T2_DUP_WRITE=1`; 정본 `go_stack.sh:695` 는 `0` 인데 `run_ours_task.sh:128` 이 덮어쓴다).

#### ⇒ 판정: **PASS-with-precondition**

```
선행조건 1  미선언 3건을 write_once_keys 에 추가한다 (키 선정은 KB/env 축자로 따로 정당화 · [[23]])
            우선순위: deposit_check_3847 (env 가드 0)
선행조건 2  airline 은 선언 **0건**이다. run_t7390_airline.sh:58 이 T2_DUP_WRITE=1 로 돌리므로
            D6 이후 그 도메인에서 이 레버는 **전면 무발화**가 된다.
            airline KB 에 유일성 문면이 있는지 — **모른다. 확인 안 했다.**
선행조건 3  [[70]] 부호표: 221건 손실 중 무엇이 정당한 회복이고 무엇이 보호 상실인지 태스크별로.
```

**반증 / refutation**: 위 3건 외에 또 다른 표현이 나오면 이 목록도 여전히 불완전하다.
airline·telecom KB 를 같은 방식으로 훑기 전에는 *"banking 만 3건"* 이라고 말할 수 없다.

**선행 확인**: 워크플로 `wf_63c350dd` 저널 · `x901_census.py`/`x902_dump.py`/`x903_writexkb.py`/
`x707.py`(반사실) · `grep -rn "T2_DUP_WRITE" scripts/distill/tau2/`(go_stack.sh:695 · run_ours_task.sh:128
· run_t7389.sh:61 · run_t7390_airline.sh:58 · run_t7391_retail.sh:52).

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
[ ] 3e. P6 D6 격리 — 세 칸
       P6a  ⛔1차 결론 철회 — 미선언 유일성 write 최소 3건 발견(노출 12 태스크)
            request_temporary_debit_card_limit_increase_8374 · order_replacement_credit_card_7291
            · deposit_check_3847(env 가드 0 · _mut_key_of 가 유일 보호)
       P6a' 그 3건을 write_once_keys 에 선언한다 (키는 KB/env 축자로 정당화)
       P6a'' airline·telecom KB 도 같은 방식으로 훑는다 (airline 선언 0건 = D6 시 전면 무발화)
       P6b  선언-only 억제로 바꿨을 때 051 의 재신청이 통과하고 승인까지 가는가
       P6c  ✅ 2026-09-04 완료 — PASS(3/3). A.mut != B.mut 이라 D6 은 막던 것을 잃지 않는다.
            단 반사실 재현에서 deny 309→88(221건 소멸) — 부호표 필수
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
