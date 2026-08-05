# 절차 부재-구동 · pin sticky · 레지스트리 멤버십 — 설계 (2026-08-05)

> 지시(2026-08-05): 핸드오프 `HANDOFF_2026_08_05_AFTERNOON.md` §6의 **2·3번을 먼저 설계하라**.
> 상위 = 등대 `RESEARCH_MASTER.md` §1(프레임 LOCK·제1원리) · 원인 축 = `ROOTCAUSE_LEVER_ATTRIBUTION_2026_08_05.md` ·
> 추적 = `N97B_FIX_LEDGER_2026_08_05.md`. 구현은 이 문서의 §7 순서로만 하고, 게이트를 통과하지 못한 항목은 켜지 않는다.
>
> **이 설계는 처방 3개(D1·D2·D3)를 담되, 그 앞에 선결 결함 D0를 새로 세운다.** D0를 고치지 않으면
> D1은 **같은 자리에서 같은 이유로 침묵**한다(§1).

---

## 0. 한 문단

핸드오프 §6은 실패 4건(048·050·051·012)의 공통 결함을 *"절차 트리거가 호출 시점뿐이라 '진입해 놓고 아무 것도
안 부름'을 못 잡는다"* 로 요약했다. 궤적·로그·오프라인 술어 재생을 대조한 결과 그 진술은 **절반만 맞다**:
050·051은 그 형태가 맞지만, **048은 호출 시점 트리거가 발화했어야 하는데 발화하지 않았다**(술어는 오프라인에서
정상 deny·라이브 로그엔 0건). 즉 048의 근인은 "트리거 부재"가 아니라 **배관 침묵**이고, 이는 C257(V7 死경로)와
같은 실패 모드의 3번째 재발이다. 그래서 부재-구동(D1)을 얹기 전에 **선점 계측(D0)**이 선결이다.

---

## 0b. 🔒 [[05]] 3질문 ([[17]] 상설 의무)

| # | 질문 | D0 계측 | D1 부재-구동 | D2 pin sticky | D3 레지스트리 멤버십 |
|---|---|---|---|---|---|
| Q1 | scaffold/A2의 **도메인-특화가 순증**하나? | **No** — 로그 1줄(도메인 리터럴 0) | **No(엔진)** / **A2 +1키**(`feedback.absent`·절차당 1문장) | **No** — 기존 pin 재무장, 새 키 0 | **No** — 새 코드 0(기존 `T2_DISPATCH_ROLE_ENVSET` 등재) + A2 `discoverable_name_check` 확장 0키 |
| Q2 | 모델의 **유동적 판단을 결정론에 동결**하나? | No(관측 전용) | **부분 Yes** — "지금 무엇을 할 차례인가"를 선언이 말한다. 단 **표면화(문장)일 뿐 선택은 모델**이라 동결은 아니다. `verdict=deny`가 아니라 `fb`만 | **★Yes** — pin은 `tool_choice`로 **다음 행동을 엔진이 고른다**. ⇒ 기본 OFF·§6 게이트 통과 전 금지 | No — **금지 집합 대조**(할 수 있는 것을 늘리지도 줄이지도 않음·존재하지 않는 이름만 거부) |
| Q3 | scaffold가 **도메인 행동을 대신 수행**하나? | No | No — 호출하지 않는다 | **경계선** — 호출을 *강제*하지만 인자는 모델이 채운다 | No |

⇒ **D0·D3 = 기본 진행**(3질문 전부 No·비용 0). **D1 = Q2 부분 Yes → §6 계측 게이트 통과 조건부.**
**D2 = Q2 Yes → 사용자 지적대로 `Δover-action ≤ 0` 실측 전 점화 금지.**

## 0c. 🔒 [[22]] 술어 감사 (닫힘/열림)

| 처방 | 술어 | 판정 | 근거 |
|---|---|---|---|
| D1 트리거 | "활성 절차의 관측가능 노드가 미충족 ∧ 그 절차 노드 호출 없이 K턴 경과" | **닫힘** | 전부 **호출 이력**(구조 사실). 변이 불변 |
| D1 문구 | "다음 단계는 X다" | **닫힘** | X = 선언의 노드 순서에서 결정론적으로 나옴 |
| D2 | "핀 대상이 아직 호출되지 않았다" | **닫힘** | 호출 이력 대조 |
| D3-a | "give 대상이 env user-discoverable 집합에 있는가" | **닫힘** | env가 권위 집합을 노출(`user_tools.get_discoverable_tools()`) |
| D3-b | "unlock/call 대상이 env agent-discoverable 집합에 있는가" | **닫힘** | 동형(`tools.get_discoverable_tools()`) |
| (제외) | "지금 이관하는 것이 옳은가" | **열림** | 중단판단은 L5-a 소관·이 설계는 손대지 않는다 |

전부 닫힌 술어 ⇒ [[22]] 기준 scaffold 적격. **열린 술어를 끌어들인 항목 0.**

## 0d. 🔒 [[23]] A2 출처 선언 (gold 경유 0)

이 설계가 A2에 추가하는 것은 **`feedback.absent` 문장 1개(절차당)** 뿐이고, 그 내용은
**이미 선언돼 있는 `_quote_order`의 재진술**이다 — 새 사실이 0이므로 gold 경유가 구조적으로 불가능하다.

| 절차 | `_quote_order` (정책 축자·기존) | `absent` 문구가 말하는 것 |
|---|---|---|
| `credit_limit_increase` | *"These steps MUST be followed in the exact order listed."* | 그 순서의 **다음 미충족 단계 이름** |
| `credit_card_closure_retention` | *"Check these in order: 1. Pending disputes … 2. No pending replacement cards … 3. Minimum account age"* | 동일 |

**노드 이름·도구 이름은 전부 기존 선언에서 온다.** 새 키를 쓸 때마다 `_note_`에 정책 축자 출처를 적는다.
gold를 보고 알게 된 내용은 **한 글자도 넣지 않는다** — 넣으면 실험 무효.

---

## 1. ★D0 (선결) — 048에서 절차 엔진이 침묵했다

### 1.1 실측

`bank_smk_gpu1_20260805f` / `task_048` 궤적(전량 확인):

```
 18 assistant check_card_closure_eligibility {...}      ← 절차 진입 도구 = 활성화 트리거이자 노드
 20 assistant shell {"command": "grep -r 'close credit card account' ."}
 24~28 KB_search_bm25 "close credit card account" ×3
 30·34 KB_search_bm25 {"query":"get_user_dispute_history"} … 도구명 질의 6종
 43 assistant TEXT  TRANSFER NOTICE
 45 assistant transfer_to_human_agents
```

- **절차 노드 호출은 msg18 이후 0회.** gold가 요구하는 `log_credit_card_closure_reason`·보유 제안·`close_…`
  전부 미실행.
- 같은 런의 로그 `[T2_PROCEDURE]` **4건 전부 `credit_limit_increase`**(050·053) — **closure는 0건**.
- 그런데 **오프라인 재생은 deny를 낸다**(선언·엔진 현행 HEAD 그대로):

```
decide(procs, "check_card_closure_eligibility", {}, executed=[verify_identity, get_user_information_by_name,
       get_current_time, log_verification, get_credit_card_accounts_by_user])
→ {"procedure":"credit_card_closure_retention","missing":["disputes","pending_replacement"],"verdict":"deny"}
```

⇒ **술어는 정상, 라이브에서 그 분기가 평가되지 않았다.**

### 1.2 원인 가설 (배타적으로 좁혀짐)

`t2_gate_patch.py:4587`의 진입 조건이

```python
if (_procs and not do_gate and not do_prov and ep_fb is None and cons_fb is None
        and ra_fb is None and te_fb is None and _t2_proc_deny < cap):
```

이라, **앞선 레버가 그 턴의 피드백을 이미 잡으면 절차는 아예 평가되지 않는다.** 같은 런에서
`T2_EPLAN` 8회·`T2_PREKB` 11회·`T2_CLAIMPROV` 29회가 떴다. 이는 `T2_TOOL_SIGNATURE`가 2026-07-31에
겪은 것과 **같은 사슬-선점**이고, 그때의 해법이 이미 코드에 있다(`t2_gate_patch.py:5089` `_chain`/`_blocker`).

### 1.3 D0 처방 — 2단(측정 먼저)

- **D0-a (관측 전용·거동 0)**: 절차 술어를 **항상** 평가하고(순수함수·비용 0), 못 뜬 턴엔
  `[T2_PROCEDURE] would-fire but suppressed by=<lever>`를 남긴다. `T2_TOOL_SIGNATURE_OBSERVE` 선례를
  그대로 복사한다 — 새 메커니즘 0.
- **D0-b (D0-a 결과가 선점을 확증할 때만)**: 절차 블록을 사슬에서 **앞으로** 옮긴다.
  근거: 절차 deny는 **정책 MUST 위반**이라 소프트 피드백 중 우선순위가 최상이다(`enforce`+`_quote_order`가
  선언으로 그것을 licence한다). 이동은 **순서만** 바꾸고 술어·문구는 불변.

> ⚠**D0-a 없이 D0-b를 하지 않는다.** "선점일 것이다"는 아직 [?]다. 로그 한 줄이 [M]으로 올린다([[08]]).

### 1.4 D0-c — 절차 피드백이 **한국어**다

`feedback.unmet`이 한국어인데 대화 전체가 영어다(다른 레버 문구는 전부 영어). 051에서 deny가 단계를
이름으로 지목하고도 이행되지 않은 것의 **경합 가설**이며, 교정 비용은 A2 문자열 1개다.
⇒ **영어로 통일**하고, D1/D2 실험 전에 처리한다(교란 제거).

---

## 2. D1 — 부재-구동 표면화

### 2.1 표적 (실측)

| sim | 진입 | 그 뒤 | 현행 엔진 |
|---|---|---|---|
| **050** | msg36 `check_cli_eligibility` | `submit_credit_limit_increase_request` **미호출**, msg44·46 이관 2회 | 호출이 없어 침묵 |
| **048** | msg18 `check_card_closure_eligibility` | 노드 호출 **0회**, msg45 이관 | D0(침묵) + 부재 |
| **051** | 진입·deny 발화 | 지목된 `submit_request` **끝내 미호출** | deny 뒤 부재 |

**공통형 = 절차가 켜져 있고 미충족 노드가 남았는데, 그 절차 쪽으로 아무 호출도 오지 않는다.**
현행 트리거는 `notes_for_call(...)` — **호출을 인자로 받는다**. 호출이 없으면 구조적으로 못 본다.

### 2.2 엔진 (도메인-일반·리터럴 0)

`t2_procedure.py`에 순수함수 2개를 **추가**한다(기존 함수 수정 0):

```python
def active_procedures(procs, executed):
    """진입 트리거가 이미 실행된 절차들. 호출을 인자로 받지 않는 유일한 진입점."""

def pending_step(proc, executed):
    """선언 순서에서 아직 충족되지 않은 **관측가능** 노드 중 첫 번째. 없으면 None.
    관측 불가 노드(도구를 이름하지 않는 bound)는 부재의 근거가 될 수 없다 — 기존 `_satisfied`의
    None 규약을 그대로 따른다."""
```

`pending_step`은 `unmet_nodes`와 달리 **특정 호출의 requires 사슬이 아니라 절차 전체**를 본다.
정렬 = 선언 순서(엔진이 순서를 만들지 않는다).

### 2.3 트리거·해제·상한

| 항목 | 규칙 | 이유 |
|---|---|---|
| 발화 조건 | `active_procedures` 비어있지 않음 ∧ `pending_step` 있음 ∧ **그 절차의 어떤 노드 도구도 최근 K assistant 턴 동안 실행되지 않음** | 048/050 실측형. KB 검색·shell은 "절차 쪽 호출"이 아니다 |
| K | `T2_PROC_ABSENT_K`(기본 **3**) | 048은 진입 후 13턴·050은 4턴 — 3이면 둘 다 잡고 정상 흐름의 연속 read는 안 건드린다 |
| 채널 | **비커밋 `fb` 1건**(deny 아님·`verdict` 불변) | 표면화만. 차단은 정책이 licence한 호출-시점에서만 |
| 턴당 | **1회** | 재생성 루프 무한 발화 방지(`wev_rounds` 선례) |
| sim당 상한 | `T2_PROC_ABSENT_CAP`(기본 **2**) | 불응 시 조용히 소진 — 기존 cap 규약 |
| 해제 | 해당 노드 실행 / 절차가 더 이상 활성 아님 / 상한 소진 | |

### 2.4 문구 — **이름만 주면 048이 재현된다**

048은 지목된 이름을 **그대로 BM25 질의로 넣어 0점을 6회** 받고 포기했다. 그러므로 부재 문구는
**미충족 노드의 도구가 discoverable인데 아직 unlock되지 않았으면 자연어 질의를 함께 준다**:

- 파생 규칙은 이미 엔진에 있다 — `re.sub(pattern,"",name).replace("_"," ")` (`t2_gate_patch.py:5142` `{name_words}`).
- 새 코드가 아니라 **기존 파생의 재사용**이고 도메인 리터럴 0이다.
- A2 `feedback.absent`가 `{missing}`·`{tool}`·`{name_words}` 슬롯을 갖는다(기존 `_fill` 규약 그대로).

### 2.5 D1이 **하지 않는 것**

- 호출하지 않는다(엔진이 도메인 행동을 수행 = Q3 위반).
- 이관을 막지 않는다(중단판단 = L5-a 소관·열린 술어).
- `verdict`를 바꾸지 않는다(차단 권한은 `enforce`+`_quote_order`에만).

---

## 3. D2 — pin sticky

### 3.1 현행 결함 (코드 실측)

- 설정: `t2_gate_patch.py:4616` — deny 시 `self._t2_proc_pin = (dispatcher, arg, tool)`
- 소비: `t2_gate_patch.py:5404` — **읽는 즉시 `None`**. 그 재생성 1회만 고정되고, 같은 턴의 후속
  재생성이나 다음 턴은 무장 해제 상태다.
- 051 실측: 고정했는데 표적 호출 0회.

### 3.2 설계

```
재무장: 표적이 실행될 때까지 매 재생성마다 다시 세운다.
해제  : (a) 표적 도구가 실행됨  (b) 절차가 더 이상 활성 아님  (c) 재무장 상한 소진
상한  : T2_PROC_PIN_REARM (기본 0 = **현행 거동 그대로**)
```

**기본값 0**이 핵심이다 — 코드는 들어가되 **측정 게이트(§6)를 통과하기 전에는 아무 것도 바뀌지 않는다.**

### 3.3 ⚠왜 기본 OFF인가 (사용자 지적의 형식화)

pin은 `tool_choice`를 단일값으로 좁힌다 = **"엔진이 다음 행동을 고른다"에 가장 근접한 레버**이고,
등대 §1.3의 자기-역효과 법칙("게이트 자신도 over-action을 판다")이 정확히 겨냥하는 형태다.
⇒ §6의 `Δover-action ≤ 0` 실측 전에는 켜지 않는다. **이 문장이 점화 조건이다.**

---

## 4. D3 — 레지스트리 멤버십 (닫힌 술어인데 새는 두 구멍)

### 4.1 표적 실측 — `task_012`

```
 16 assistant give_discoverable_user_tool {"discoverable_tool_name": "navigate_to_travel_notification"}
```

이 이름은 **존재하지 않는다**. banking env의 user-discoverable 집합은
`{deposit_check_3847, get_card_last_4_digits, get_referral_link, submit_cash_back_dispute_0589}` 4종이다.
그런데 두 검사 모두 이 호출을 놓친다:

| 검사 | 왜 놓쳤나 | 확인 |
|---|---|---|
| `T2_UNLOCK_NAME`(접미사) | A2 `discoverable_name_check.tools` = **`unlock_discoverable_agent_tool` 하나뿐** — give는 대상이 아님 | A2 직독 |
| `T2_DISPATCH_ROLE`(멤버십) | `T2_DISPATCH_ROLE_ENVSET` **미설정** ⇒ 구판 분기(`self.tools` 소속일 때만 deny)를 타서, **존재하지 않는 이름은 통과** | `grep T2_DISPATCH_ROLE_ENVSET go_stack.sh` = 0건 |

### 4.2 D3-a — `T2_DISPATCH_ROLE_ENVSET=1` **등재** (신규 코드 0)

구현·근거는 이미 있다(C257: Y1 give 89회 중 **18회가 env 집합 밖**, 그 우회가 `unlock`→`call` 미호출
**55건 = 전체 실패의 27%**로 이어짐). **등재된 적이 없을 뿐이다** — [[24]]가 경고한 死코드 패턴의
**4번째 사례**이고, 오늘 F1·F2에 이은 3·4번째 회수다.

### 4.3 D3-b — 접미사 패턴을 give로 **확장하지 않는다** (설계 판정)

직관적 확장은 틀렸다: 정당한 user-discoverable 중 `get_card_last_4_digits`·`get_referral_link`는
**접미사가 없다**. 접미사 규칙을 give에 적용하면 **정당한 give 2종을 상시 과차단**한다.
⇒ give의 술어는 **집합 소속**이어야 하고, 그것이 정확히 D3-a다. (패턴 규칙이 조용한 오탐을 낳는다는
C279 교훈과 동형.)

### 4.4 D3-c — agent 측 동형 확장

`unlock_discoverable_agent_tool` / `call_discoverable_agent_tool`의 이름도 같은 종류의 닫힌 술어로 잴 수 있다:

```python
def _agent_discoverable(env):
    return set(getattr(env, "tools").get_discoverable_tools())   # user 측과 동형·리터럴 0
```

env 구조 확인 완료(`environment.py:58-59` `self.tools`/`self.user_tools`, `toolkit.py:186`).
이것이 핸드오프 §6-3의 **접미사 날조**(`..._7894` vs 실제 `..._5847`)를 덮는다 — 접미사 규칙이 아니라
집합 대조라서 오탐이 구조적으로 불가능하다.

> ⚠**단 D3-c는 잠금 의미론 확인이 선결**: `get_discoverable_tools()`가 *잠긴* 도구도 포함하는지
> (=unlock 전 이름이 집합에 있는지) 확인해야 한다. 포함하지 않으면 이 검사는 **정당한 첫 unlock을
> 전부 차단**한다. 확인 전 구현 금지.

---

## 5. 이 설계가 **덮지 않는 것** (정직)

| 실패 | 왜 여기 없나 |
|---|---|
| 048의 **도구명 0점 검색 6회** | 근본기능 = **조립(도구 발견)** = F9. D1 문구가 완화할 뿐 닫지 않는다. 담당 = `T2_MATCH_COUNT`·`T2_CALLABLE_HINT` |
| 048·050의 **이관** | 중단판단 = **열린 술어** = L5-a. 이 설계는 손대지 않는다 |
| 051의 **2라운드 gold**(요청→거부→상환→재요청→승인) | 선언(DAG)에 분기 추가 = A2 작업. 별건 |
| 012의 **부재 종결** | 부재판정 = ⒟ 탐색소진. D3는 *날조 give*만 막는다 — 012가 통과로 뒤집힌다고 예측하지 않는다 |

---

## 6. 계측·사전등록 (전부 무료·194 sim 오프라인)

> 등대 §1.3: **모든 게이트에 반대편 계측을 단다.** 아래를 통과하지 못한 항목은 점화하지 않는다.

| 계기 | 무엇을 재나 | 점화 조건 |
|---|---|---|
| **x86** `procedure_absence_census` | ① D1이 발화했을 sim·턴 수 ② 그때 지목했을 노드 도구가 **그 sim의 gold에 있는가** | gold-밖 지목 비율 = **Δover-action 대리** |
| **x80 확장** | 현행 x80은 *과차단*만 본다. **"원하지 않은 호출을 밀어넣나"** 열을 추가 | D1·D2 공통 게이트 |
| **x87** `pin_rearm_replay` | 재무장이 있었다면 몇 턴 더 고정됐을지·표적이 gold인 비율 | **D2 점화의 유일 근거** |
| **x88** `give_membership_census` | 194 sim의 give 전수를 env 집합과 대조: 밖 N건 / 그 중 **gold가 요구한 give 0건인가** | D3-a 점화(오차단 0 확인) |

**사전등록 (판정은 발화·경로·표적 실측으로 한다 — pass 아님·[[08]])**

1. **D0-a**: `suppressed by=` 분포에서 048형(closure 절차)의 선점자가 특정된다. 특정 실패 시 D0-b 보류.
2. **D1**: 048·050에서 절차 활성 후 **K=3 내에 발화 ≥1**. gold-밖 지목 비율이 **x86에서 30% 초과면 K 상향** 후 재계량.
3. **D2**: x87의 표적-gold 비율이 **x86의 그것보다 높을 때만** 재무장을 켠다(핀은 부재보다 강한 개입이므로 근거도 더 강해야 한다).
4. **D3-a**: x88에서 **gold가 요구한 give 중 집합 밖 = 0건**이면 등재. 1건이라도 있으면 **등재하지 않는다**.
5. **D3-c**: 잠금 의미론 확인(§4.4 ⚠) 후에만 착수.

---

## 7. 순서 (이 순서로만)

```
D0-c  절차 피드백 영어화                        (A2 문자열·비용 0·교란 제거)
D0-a  선점 계측 로그                            (거동 0)
x88 → D3-a  ENVSET 등재                         (신규 코드 0·오차단 0 확인 후)
D0-b  절차 블록 이동                            (D0-a가 선점을 확증할 때만)
x86 → D1   부재-구동 표면화                     (엔진 순수함수 2 + A2 문구 1)
x87 → D2   pin sticky                           (기본 OFF로 구현 → 게이트 통과 시에만 REARM>0)
D3-c  agent 측 멤버십                           (잠금 의미론 확인 후)
```

각 단계는 **오프라인 단위검정 + 계기 산출**까지가 완료 조건이고, 라이브 발화 확인은
다음 스모크에서 한 번에 받는다([[30]]·이 아크에서 침묵 4회를 겪은 규율).

## 8. 원장 반영

- `N97B_FIX_LEDGER_2026_08_05.md` §2에 **F15(D0 배관 침묵)**·**F16(D1 부재-구동)**·**F17(D3-a 등재)** 추가.
- `F14`(권한 밖 행동)는 D1·D2가 아니라 **선언 분기**(051 2라운드) 소관임을 명시 — 혼동 방지.
