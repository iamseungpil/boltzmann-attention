# 절차 부재-구동 · pin sticky · 레지스트리 멤버십 — 설계 (2026-08-05)

> 지시(2026-08-05): 핸드오프 `HANDOFF_2026_08_05_AFTERNOON.md` §6의 **2·3번을 먼저 설계하라**.
> 상위 = 등대 `RESEARCH_MASTER.md` §1(프레임 LOCK·제1원리) · 원인 축 = `ROOTCAUSE_LEVER_ATTRIBUTION_2026_08_05.md` ·
> 추적 = `N97B_FIX_LEDGER_2026_08_05.md`. 구현은 이 문서의 §7 순서로만 하고, 게이트를 통과하지 못한 항목은 켜지 않는다.
>
> **이 설계는 처방 3개(D1′·D2·D3)를 담되, 그 앞에 선결 결함 D0를 새로 세운다.** D0를 고치지 않으면
> D1′은 **같은 자리에서 같은 이유로 침묵**한다(§1).
>
> **개정 2026-08-05(사용자 제안)**: D1을 *표면화*에서 **실행 체크리스트 + 조건부 다음-단계 지정**으로
> 바꿨다(§2). 계기는 §2.0의 정정 — 051이 받은 `missing=submit_request`는 **노드 id**였지 호출 가능한
> 이름이 아니었고, 그래서 *"알려줘도 안 한다"* 는 아직 증명되지 않았다.

---

## 0. 한 문단

핸드오프 §6은 실패 4건(048·050·051·012)의 공통 결함을 *"절차 트리거가 호출 시점뿐이라 '진입해 놓고 아무 것도
안 부름'을 못 잡는다"* 로 요약했다. 궤적·로그·오프라인 술어 재생을 대조한 결과 그 진술은 **절반만 맞다**:
050·051은 그 형태가 맞지만, **048은 호출 시점 트리거가 발화했어야 하는데 발화하지 않았다**(술어는 오프라인에서
정상 deny·라이브 로그엔 0건). 즉 048의 근인은 "트리거 부재"가 아니라 **배관 침묵**이고, 이는 C257(V7 死경로)와
같은 실패 모드의 3번째 재발이다. 그래서 부재-구동(D1′)을 얹기 전에 **선점 계측(D0)**이 선결이다.

그리고 남은 절반(050·051의 "진입해 놓고 안 부름")에 대해서도 처방이 바뀌었다. 초판은 *"미충족 단계가
있다"* 를 알리는 표면화였는데, **모델이 못 받고 있는 것은 규칙이 아니라 상태**다 — 정책 산문은 순서를
말하고, 어느 문서도 *지금 어디까지 왔는지*는 말하지 않는다. 엔진은 그 재료(실행 이력·DAG 위상·노드→도구
매핑·unlock 상태)를 이미 전부 갖고 있으면서 `missing=<노드 id>` 한 조각으로만 뱉고 있었다. ⇒ **D1′ =
실행 체크리스트 + 조건부 다음-단계 지정**(§2).

---

## 0b. 🔒 [[05]] 3질문 ([[17]] 상설 의무)

| # | 질문 | D0 계측 | **D1′ 체크리스트** | D2 pin sticky | D3 레지스트리 멤버십 |
|---|---|---|---|---|---|
| Q1 | scaffold/A2의 **도메인-특화가 순증**하나? | **No** — 로그 1줄(도메인 리터럴 0) | **No(엔진)** / **A2 +1키**(`feedback.absent`·절차당 1문장). ✓/☐·도구명·unlock·검색어는 **전부 기존 재료의 파생** | **No** — 기존 pin 재무장, 새 키 0 | **No** — 새 코드 0(기존 `T2_DISPATCH_ROLE_ENVSET` 등재) + A2 `discoverable_name_check` 확장 0키 |
| Q2 | 모델의 **유동적 판단을 결정론에 동결**하나? | No(관측 전용) | **체크리스트 본문 = No**(호출 이력의 **사실 진술**). **▶NEXT = 부분 Yes** → §2.3으로 봉함: **유일할 때만** 지목·동렬이면 목록·`enforce` 없는 절차엔 미부착 | **★Yes** — pin은 `tool_choice`로 **다음 행동을 엔진이 고른다**. ⇒ 기본 OFF·§6 게이트 통과 전 금지 | No — **금지 집합 대조**(할 수 있는 것을 늘리지도 줄이지도 않음·존재하지 않는 이름만 거부) |
| Q3 | scaffold가 **도메인 행동을 대신 수행**하나? | No | No — 호출하지 않는다 | **경계선** — 호출을 *강제*하지만 인자는 모델이 채운다 | No |

⇒ **D0·D3 = 기본 진행**(3질문 전부 No·비용 0). **D1′ = ▶NEXT만 Q2 부분 Yes → §2.3 봉함 + §6 계측 게이트.**
**D2 = Q2 Yes → 사용자 지적대로 `Δover-action ≤ 0` 실측 전 점화 금지.**

## 0c. 🔒 [[22]] 술어 감사 (닫힘/열림)

| 처방 | 술어 | 판정 | 근거 |
|---|---|---|---|
| D1′ 트리거 | "활성 절차의 관측가능 노드가 미충족 ∧ 그 절차 노드 호출 없이 K턴 경과" | **닫힘** | 전부 **호출 이력**(구조 사실). 변이 불변 |
| D1′ 체크리스트 | "노드 N은 실행됐다 / 안 됐다" | **닫힘** | 호출 이력 대조. 관측 불가 노드는 **판정하지 않는다**(done=None) |
| D1′ 도구명·unlock | "노드 N이 부를 도구는 T이고 아직 unlock되지 않았다" | **닫힘** | 선언 + env 레지스트리 — [[22]] 기준 D3와 같은 종류 |
| D1′ **▶NEXT** | "다음에 할 것은 X 하나다" | **조건부 닫힘** | 선행이 충족된 미충족 노드가 **유일할 때만** 닫힘. 동렬이면 열림 ⇒ 지목 금지·목록만(§2.3) |
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

### 1.5 ★★★스모크 g가 찾은 것 — 절차 deny는 **모델에게 간 적이 없다**

D0-a를 켜고 돌린 스모크 g(4 sim·concurrency 1)의 1차 지표는 **선점 0건**이었다. 즉 §1.2의 선점 가설은
**지지되지 않았다**. 대신 훨씬 큰 것이 나왔다.

`t2_gate_patch.py`의 재생성 루프는 *"세워진 피드백이 하나도 없으면 끝낸다"* 는 가드로 끝난다.
그 가드가 보는 피드백 변수는 14종이었고, **루프가 세우는 것은 15종이었다.** 빠진 하나가 `proc_fb`다
(AST 전수 대조로 확인·`test_regen_break_guard.py`).

```
while True:                       # 재생성 루프
    ...  proc_fb = (c, notes[0])  # 절차 deny 성립 → 여기까지는 정상
    if (... 14종 전부 None):       # ← proc_fb 미포함
        break                     # ★절차만 떴으면 여기서 끊긴다
    ...
    self._t2_proc_deny += 1       # (a) 카운터  — 도달 못 함
    ...  content = proc_fb[1]     # (b) 피드백 조립 — 도달 못 함
```

**결과 두 가지, 둘 다 실측으로 확인된다:**

| 잃은 것 | 실측 |
|---|---|
| **(a) sim당 cap이 한 번도 물리지 않았다** | 스모크 g `task_048`에서 절차 deny **11회**(cap=6). retries=0이므로 재시도로도 설명 안 됨 |
| **(b) deny 문구가 모델에게 전달된 적이 없다** | 048은 같은 deny(`log_credit_card_closure_reason_4521 missing=prior_attempts`)를 **10회** 받고도 행동이 한 번도 바뀌지 않았다 — 받은 적이 없으니 당연하다 |

로그의 `[T2_PROCEDURE] deny …`는 **차단이 아니라 인쇄**였다. 이 파일에서 이번 아크에만 3번째로 나온
같은 실패 모드다(C257 V7 死경로 · F1/F2 미등재 · 이번).

#### 1.5b 048의 livelock — 우리 엔진이 한쪽 다리였다

스모크 g의 048(150 msg·`user_stop`)은 msg 70~111에서 **8회전 폐루프**를 돈다:

```
70  call get_closure_reason_history_8293      → Error: "has not been unlocked"
72  unlock+call log_credit_card_closure_reason_4521   → 성공 (하지만 절차상 아직 이르다)
75  call get_closure_reason_history_8293      → 같은 Error
77  unlock+call log_credit_card_closure_reason_4521   → 성공
…  ×8회전, 그동안 [T2_PROCEDURE] deny missing=prior_attempts 10회 인쇄
```

- 모델은 **지목받은 단계를 하려고 했다** — `get_closure_reason_history_8293`을 8번 불렀다.
- 실패 이유는 딱 하나: **그 도구를 unlock하지 않았다.** 엉뚱하게 `log_…_4521`을 반복 unlock한다.
- 엔진은 에러 결과를 정확히 "미실행"으로 판정한다(`_executed_tool_names`의 error 규약은 옳다)
  ⇒ `prior_attempts` 영원히 미충족 ⇒ 같은 deny 무한 재생산.

⇒ **D1′이 정확히 이 구멍을 메운다**: `missing=prior_attempts`(노드 id) 대신
`▶ prior_attempts → get_closure_reason_history_8293 (아직 unlock 안 됨 — 먼저 unlock하라)`.
모델에게 없던 정보는 **오직 "그 도구를 unlock해야 한다"** 하나였고, unlock 의식 자체는 알고 있었다.

#### 1.5c 이 발견으로 **철회되는 판정**

| 철회 | 이유 |
|---|---|
| 스모크 d §2d *"`T2_PROCEDURE` deny 2회 — ✅ 표적대로 발화"* | 발화가 아니라 **인쇄**였다. 전달 0 |
| 스모크 d §2d *"051 = 차단은 됐는데 이행되지 않았다"* | **차단된 적이 없다.** C286③을 절차 레버에 적용한 것은 무효 |
| §2.0의 *"노드 id라서 못 알아들었다"* | **부분 철회** — 애초에 못 받았다. id 문제는 배관을 고친 **뒤에야** 검증 가능 |
| D2(pin sticky)의 근거 *"deny해도 이행 안 하니 고정이 필요하다"* | **근거 소멸.** pin의 필요성은 배관 교정 후 다시 재야 한다 |

#### 1.5d 처방 (적용 완료)

1. **가드에 `proc_fb` 추가** — 술어·문구·차단 조건 불변, **전달 경로만** 잇는다.
2. **`test_regen_break_guard.py`** — AST로 *루프가 세우는 모든 `*_fb`가 가드에 있는가*를 강제한다.
   주석·관례로는 재발을 못 막는다([[07]]): 다음 레버도 같은 방식으로 추가될 것이다.
   **부정 통제 확인**: 교정을 되돌린 사본에서 `FAIL(exit 1)`, 교정본에서 `PASS`.
3. **호출-레벨 선점도 관측**: 구판은 `denied_by_objid`를 술어 **앞에서** 건너뛰어, 다른 레버가 그 호출을
   이미 막은 턴은 로그조차 없었다 — **스모크 f의 048 침묵의 남은 후보**가 그것이다.
   거동은 그대로 두고 `suppressed by=call_denied`를 남긴다.

> ⚠**이제 거동이 실제로 바뀐다**(전달된 적 없던 deny가 전달되기 시작). 다음 스모크의 1차 지표는
> **① 전달 확인**(같은 deny의 반복이 줄어드는가) · **② cap 6이 무는가** · **③ Δover-block**이다.

### 1.4 D0-c — 절차 피드백이 **한국어**다

`feedback.unmet`이 한국어인데 대화 전체가 영어다(다른 레버 문구는 전부 영어). 051에서 deny가 단계를
이름으로 지목하고도 이행되지 않은 것의 **경합 가설**이며, 교정 비용은 A2 문자열 1개다.
⇒ **영어로 통일**하고, D1/D2 실험 전에 처리한다(교란 제거).

---

## 2. D1′ — 실행 체크리스트 + 조건부 다음-단계 지정

> **2026-08-05 개정(사용자 제안).** 초판 D1은 *"미충족 단계가 있다"* 를 알리는 **표면화**였다.
> 사용자 지적: *"실행 기억을 못한다는 DAG에 이미 실행된 것을 체크리스트로 체크하고 결정론이
> 다음 단계를 알려주게 하면 안 되나?"* — 초판보다 강하고, **아래 §2.1의 정정이 그 손을 들어준다.**

### 2.0 ★정정 — "알고도 안 한다"는 아직 증명되지 않았다

앞선 판정에서 나는 *"051은 deny가 누락 단계를 **이름으로 지목**했는데도 호출하지 않았다"* 고 적었다.
로그를 다시 읽으면 지목된 것은 **DAG 노드 id**다:

```
[T2_PROCEDURE] deny get_credit_limit_increase_history_4829 missing=submit_request
```

`submit_request`는 **호출 가능한 이름이 아니다**. 실제 도구는 `submit_credit_limit_increase_request_7392`이고
**unlock이 필요한 discoverable**이다. 즉 모델이 받은 것은 내부 식별자였고, 거기서 호출까지 가려면
접미사를 KB에서 찾아야 했다 — 048이 도구명을 질의로 넣어 **0점 6회**를 받은 그 벽이다.

⇒ **C286③("차단은 복원이 아니다")을 "모델이 알고도 거부한다"로 읽으면 안 된다.** 현 상태는
"어디까지 했는지도, 무엇을 부를지도 받지 못했다"에 가깝다. `missing=<node id>`는 **결함**이고
D1′이 그것을 함께 고친다(호출-시점 deny 문구에도 동일 적용).

### 2.1 표적 (실측)

| sim | 진입 | 그 뒤 | 현행 엔진 |
|---|---|---|---|
| **050** | msg36 `check_cli_eligibility` | `submit_credit_limit_increase_request` **미호출**, msg44·46 이관 2회 | 호출이 없어 침묵 |
| **048** | msg18 `check_card_closure_eligibility` | 노드 호출 **0회**, 도구명 0점 검색 6회, msg45 이관 | D0(침묵) + 부재 |
| **051** | 진입·deny 발화 | **노드 id만 받고** 미호출 | §2.0 |

**공통형 = 절차가 켜져 있고 미충족 노드가 남았는데, 그 절차 쪽으로 아무 호출도 오지 않는다.**
현행 트리거는 `notes_for_call(...)` — **호출을 인자로 받는다**. 호출이 없으면 구조적으로 못 본다.

### 2.2 무엇을 주는가 — 규칙이 아니라 **상태**

정책 산문은 *순서*를 말한다. 체크리스트는 ***지금 어디인가***를 말한다. 후자는 어느 문서에도 없고,
모델이 반복해서 틀리는 것이다(048은 하지 않은 단계를 한 것처럼 진행했다). 엔진은 그 재료를 이미
전부 갖고 있다 — 실행 이력·DAG 위상·노드→도구 매핑·env의 unlock 상태.

```
[PROCEDURE] credit_limit_increase — 2/7 done
  ✓ amount_within_tier_cap   ✓ submit_request
  ☐ cooldown            → get_credit_limit_increase_history_4829   (unlocked)
  ▶ NEXT: disputes      → get_user_dispute_history_7291
        not unlocked yet — unlock it first; search the knowledge base
        for "user dispute history" (a bare tool-name query matches nothing)
  ☐ decision (requires all four checks)
```

**한 줄도 새 사실이 아니다**: ✓/☐=호출 이력, 도구명=선언, unlock 상태=env, 검색어=이름의 순수 문자열 파생.

### 2.3 ⚠▶NEXT의 경계 — 여기가 유일한 위험 지점

- **다음 후보가 유일할 때만** `▶`로 지목한다. 병렬이면(CLI의 4개 검사는 전부 `submit_request`만 요구한다)
  **목록으로만** 준다. 엔진이 동렬에서 하나를 고르면 그것은 [[10]] 위반(선택기=결정론이 아니라 **생성기 대행**)이다.
- **`enforce`가 없는 절차에는 `▶`를 붙이지 않는다.** 정책이 순서를 명령하지 않은 곳에서 순서를 만드는 셈이다
  (`cash_back_dispute`가 그 경우).
- 체크리스트 본문(✓/☐)은 **사실 진술**이라 이 제약을 받지 않는다.

### 2.4 엔진 (도메인-일반·리터럴 0)

`t2_procedure.py`에 순수함수 3개를 **추가**한다(기존 함수 수정 0):

```python
def active_procedures(procs, executed):
    """진입 트리거가 이미 실행된 절차들. 호출을 인자로 받지 않는 유일한 진입점."""

def checklist(proc, executed):
    """[(node_id, tools, done)] — 선언 순서 그대로. 관측 불가 노드는 done=None으로 남긴다
    (도구를 이름하지 않는 bound는 이력으로 판정할 수 없다·기존 `_satisfied` 규약)."""

def next_step(proc, executed):
    """(node, unique: bool) — 선행이 전부 충족된 미충족 노드들. 하나뿐이면 unique=True.
    엔진은 동렬에서 고르지 않는다: 여럿이면 전부 돌려주고 호출자가 목록으로 표시한다."""
```

노드 id → 도구명 → unlock 여부 → 자연어 질의의 파생은 **전부 기존 재료**다.
`{name_words}` 규칙은 이미 엔진에 있다(`t2_gate_patch.py:5142` — `re.sub(pattern,"",n).replace("_"," ")`).

### 2.5 트리거·해제·상한

| 항목 | 규칙 | 이유 |
|---|---|---|
| 발화 조건 | `active_procedures` 비어있지 않음 ∧ 미충족 노드 있음 ∧ **그 절차의 어떤 노드 도구도 최근 K assistant 턴 동안 실행되지 않음** | 048/050 실측형. KB 검색·shell은 "절차 쪽 호출"이 아니다 |
| K | `T2_PROC_ABSENT_K`(기본 **3**) | 048은 진입 후 13턴·050은 4턴 — 3이면 둘 다 잡고 정상 흐름의 연속 read는 안 건드린다 |
| 채널 | **비커밋 `fb` 1건**(deny 아님·`verdict` 불변) | 차단은 정책이 licence한 호출-시점에서만 |
| 턴당 | **1회** | 재생성 루프 무한 발화 방지(`wev_rounds` 선례) |
| sim당 상한 | `T2_PROC_ABSENT_CAP`(기본 **2**) | 불응 시 조용히 소진 — 기존 cap 규약 |
| 해제 | 노드 실행 / 절차 비활성 / 상한 소진 | |
| **호출-시점 재사용** | 기존 deny 문구의 `{missing}`(노드 id)을 **같은 체크리스트로 교체** | 051이 받은 그 메시지가 고쳐진다 |

### 2.6 왜 초판보다 강한가 — 세 진단 중 무엇이 맞아도 산다

| 진단 | 초판 D1(표면화) | **D1′(체크리스트)** |
|---|---|---|
| 절차를 **모른다** | ○ | ○ |
| 알지만 **어디까지 했는지** 모른다 | ✗ | **○** |
| 알고 위치도 아는데 **이름을 못 찾는다**(048) | ✗ | **○** |

⇒ x89(절차 진술 vs DAG 프로브)는 **D1′의 전제 조건이 아니다.** 어느 결과가 나와도 D1′은 정당화된다.
x89는 *잔여가 어디 남는지*를 재는 용도로 순위를 내린다.

### 2.7 D1′이 **하지 않는 것**

- 호출하지 않는다(엔진이 도메인 행동을 수행 = Q3 위반). 호출 강제는 D2이고 별도 게이트다.
- 이관을 막지 않는다(중단판단 = L5-a 소관·열린 술어).
- `verdict`를 바꾸지 않는다(차단 권한은 `enforce`+`_quote_order`에만).
- 동렬 노드 중 하나를 고르지 않는다(§2.3).

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
| 048의 **도구명 0점 검색 6회** | 근본기능 = **조립(도구 발견)** = F9. D1′은 *절차 노드의 도구*에 한해 이름·unlock·검색어를 주므로 **그 구간은 덮지만**, 절차 밖 탐색은 닫지 않는다. 담당 = `T2_MATCH_COUNT`·`T2_CALLABLE_HINT` |
| 048·050의 **이관** | 중단판단 = **열린 술어** = L5-a. 이 설계는 손대지 않는다 |
| 051의 **2라운드 gold**(요청→거부→상환→재요청→승인) | 선언(DAG)에 분기 추가 = A2 작업. 별건 |
| 012의 **부재 종결** | 부재판정 = ⒟ 탐색소진. D3는 *날조 give*만 막는다 — 012가 통과로 뒤집힌다고 예측하지 않는다 |

---

## 6. 계측·사전등록 (전부 무료·194 sim 오프라인)

> 등대 §1.3: **모든 게이트에 반대편 계측을 단다.** 아래를 통과하지 못한 항목은 점화하지 않는다.

| 계기 | 무엇을 재나 | 점화 조건 |
|---|---|---|
| **x86** `procedure_absence_census` | ① D1′이 발화했을 sim·턴 수 ② 그때 **▶NEXT가 유일했던 비율** ③ 지목했을 노드 도구가 **그 sim의 gold에 있는가** | gold-밖 지목 비율 = **Δover-action 대리** |
| **x80 확장** | 현행 x80은 *과차단*만 본다. **"원하지 않은 호출을 밀어넣나"** 열을 추가 | D1′·D2 공통 게이트 |
| **x87** `pin_rearm_replay` | 재무장이 있었다면 몇 턴 더 고정됐을지·표적이 gold인 비율 | **D2 점화의 유일 근거** |
| **x88** `give_membership_census` ✅ | 194 sim의 give 전수를 env 집합과 대조: 밖 N건 / 그 중 **gold가 요구한 give 0건인가** | D3-a 점화(오차단 0 확인) — **통과**(41건 중 0) |
| **x89** `procedure_recall_probe` | 진입 시점 문맥을 재생해 **모델에게 절차를 진술시키고 DAG와 대조**. 진술 정확 / 위치 오인 / 이름 미해결로 3분 | **D1′의 전제가 아니다**(§2.6) — 잔여 귀속용 |

### 6.1 ✅x86 실측 (2026-08-05·N97B 194 sim·무료·`x86_procedure_absence_census.py`)

계기는 엔진이 쓸 **바로 그 순수함수**(`active_procedures`·`checklist`·`next_step`)를 호출한다 —
술어가 다르면 계량이 무의미하다.

| K | 발화 | sim | ▶유일 | gold에 있음 | **gold-밖의 write** | 미-unlock |
|---|---|---|---|---|---|---|
| 2 | 57회 | 30 (15%) | 98.2% | 70.2% | **0** (read 17) | 98.2% |
| **3**(기본) | 54회 | 29 (15%) | **98.1%** | 72.2% | **0** (read 15) | **100%** |
| 5 | 42회 | 25 (13%) | 100% | 78.6% | **0** (read 9) | 100% |

**판정 3개**

1. **over-action 위험이 구조적으로 0이다.** gold-밖 지목은 K와 무관하게 **전부 read**다(write 0/54).
   등대 §1.3이 겨냥하는 상쇄(DB-state를 바꾸는 여분 행동)는 이 레버에 없다 ⇒ 사전등록 게이트 통과.
   비용은 토큰이지 DB가 아니다.
2. **▶NEXT는 98% 유일하다.** 동렬은 CLI가 `submit_request` 직후 4개 검사를 동시에 여는 순간뿐이고,
   설계는 그 경우 **목록만** 준다(§2.3). 즉 "엔진이 고른다"는 위험 표면이 실측상 2%다.
   ⚠이 수치는 `next_step`의 **전이적** 선행 판정에 의존한다 — 직접 선행만 보던 초판은 유일 비율이
   **35%**였다(048에서 `eligibility`가 순서를 어기고 먼저 실행돼 뒤 단계가 "준비됨"으로 보였다).
   `unmet_nodes`와 같은 그래프 해석을 쓰도록 고쳐서 98%가 됐다(`_blocked_by`).
3. **지목 도구는 100%가 미-unlock이다.** 체크리스트의 결정적 내용은 *단계 이름*이 아니라
   **"그 도구를 먼저 unlock하라"** 라는 것이다 — §1.5b의 048 livelock과 정확히 같은 진단이다.

**정직한 한계**

- `gold-밖`은 "gold의 `action_checks`에 없다"는 뜻이고, **정책이 명령한 read를 gold가 채점하지 않는 경우**를
  구분하지 못한다(gold-밖 사례는 049·061·036이 전부 `disputes`=정책 Step 1을 가리킨다). ⇒ 27.8%는
  *상한*이지 오지목률이 아니다.
- 이 계량은 **레버가 없던 궤적** 위에서 잰 것이다. 켜면 궤적이 바뀐다(Lucas critique) —
  라이브 발화·Δ는 다음 스모크에서만 확정된다.

**사전등록 (판정은 발화·경로·표적 실측으로 한다 — pass 아님·[[08]])**

1. **D0-a**: `suppressed by=` 분포에서 048형(closure 절차)의 선점자가 특정된다. 특정 실패 시 D0-b 보류.
2. **D1′**: 048·050에서 절차 활성 후 **K=3 내에 발화 ≥1**. gold-밖 지목 비율이 **x86에서 30% 초과면 K 상향** 후 재계량.
   추가 게이트: **▶NEXT가 동렬인데 하나를 고른 사례 = 0**(§2.3 위반 검출·x86이 센다).
3. **D2**: x87의 표적-gold 비율이 **x86의 그것보다 높을 때만** 재무장을 켠다(핀은 부재보다 강한 개입이므로 근거도 더 강해야 한다).
4. **D3-a**: x88에서 **gold가 요구한 give 중 집합 밖 = 0건**이면 등재. 1건이라도 있으면 **등재하지 않는다**.
5. **D3-c**: 잠금 의미론 확인(§4.4 ⚠) 후에만 착수.

---

## 7. 순서 (이 순서로만)

```
✅D0-c  절차 피드백 영어화                        (A2 문자열·비용 0·교란 제거)
✅D0-a  선점 계측 로그                            (거동 0)
✅x88 → D3-a  ENVSET 등재                         (신규 코드 0·오차단 0 확인 후)
  D0-b  절차 블록 이동                            (D0-a가 선점을 확증할 때만 — 스모크 대기)
  x86 → D1′  실행 체크리스트 + ▶NEXT              (엔진 순수함수 3 + A2 문구 1 · 호출-시점 문구도 교체)
  x87 → D2   pin sticky                           (기본 OFF로 구현 → 게이트 통과 시에만 REARM>0)
  x89        절차-진술 프로브                      (D1′ 뒤 · 잔여 귀속용 · 무료·로컬 vLLM)
  D3-c  agent 측 멤버십                           (잠금 의미론 확인 후)
```

### 7.1 진척 (2026-08-05 오후)

| 단계 | 한 일 | 검정 |
|---|---|---|
| **D0-c** ✅ | `procedures[*].feedback` 7문장 영어화. **specific + gate 2층 동기**([[24]]) | `procedures` 두 층 **동일** · `test_t2_procedure.py` ALL PASS · `test_a2_three_layer.py` ALL PASS · 한글 잔존 0 |
| **D0-a** ✅ | 절차 술어를 **항상 평가**하고 선점 시 `would-fire but suppressed by=<lever>` 기록. 선점 체인 = gate·prov·eplan·cons·resolve_action·te·cap | **거동 불변**(선점·소진 턴은 종전대로 `proc_fb=None`) · 컴파일 OK |
| **x88** ✅ | 194 sim 전수. **gold give 41건 중 집합 밖 0** ⇒ 게이트 통과 | §6 사전등록 ④ 충족 |
| **D3-a** ✅ | `go_stack.sh`에 `T2_DISPATCH_ROLE_ENVSET=1` 등재 + 근거·금지사항 주석 | 라이브 발화는 다음 스모크에서 확인 |
| **F19~F21** ✅ | 스모크 g 포렌식 → break 가드 `proc_fb` 교정 · AST 회귀검정 · 호출-레벨 선점 관측 | §1.5 |
| **x86** ✅ | 194 sim 전수·K 스윕 2/3/5 | §6.1 — **gold-밖 write 0** |
| **D1′** ✅ **배선 완료** | 엔진 순수함수 5(`active_procedures`·`checklist`·`_blocked_by`·`next_step`·`render_state`/`absent_note`) + 패치 헬퍼 2(`_unlocked_names`·`_quiet_turns`) + A2 문구 3키 + `T2_PROC_ABSENT` 등재. 호출-시점 `unmet`도 같은 체크리스트로 교체 | 아래 §7.2 |

### 7.2 D1′ 배선 상세 (2026-08-05)

| 층 | 무엇 | 확인 |
|---|---|---|
| 엔진 `t2_procedure.py` | `render_state`(관측 슬롯만 채움) · `absent_note`(A2 문장에 주입) · `_hint`(잠금 절) · `_words`(자연어 질의·패턴은 A2에서) | 도메인 리터럴 0 |
| 엔진 `t2_gate_patch.py` | `_unlocked_names`(A2 `dispatcher_role_check`로 수집) · `_quiet_turns`(**이력에서** 계산 — 재생성 루프가 부풀리지 못한다) · 부재 블록 · `abs_fb` 가드·카운터·비커밋 전달 | AST 검정 fb **16/16** |
| A2 (2층 동기) | `absent` · `absent_many` · `unlock_hint` 신규 + `unmet` 교체 + `_note_absent`(출처 선언) | diff **10/2줄**·두 층 `procedures` 동일 |
| 드라이버 | `T2_PROC_ABSENT=1` · `_K=3` · `_CAP=2` | x77 죽은 플래그 미증가 |

**실궤적 검정** `test_proc_absent_wiring.py` — 스모크 g의 `task_048` 원본 궤적으로:
절차-무호출 **최대 18턴**(K=3 성립) · 문구가 **노드 id가 아니라 `get_closure_reason_history_8293`**을 주고
**"has not been unlocked"** 와 자연어 질의 `"get closure reason history"` 를 함께 준다 · CLI 4개 동렬에서는
**NEXT를 붙이지 않고** 목록만 준다. **ALL PASS.**

⚠**아직 라이브 발화는 확인 안 됨** — 이 아크에서 침묵을 4회 겪었으므로([[30]]) 다음 스모크의
1차 지표는 `[T2_PROC_ABSENT] surface` 실재 + 048의 반복 deny 감소 + Δover-block이다.

**남은 것의 선결 = D0-a의 로그**: D0-b(블록 이동)도 D1′(체크리스트)도 **같은 사슬 자리**에 얹히므로,
선점자가 특정되기 전에 그 위에 얹으면 048에서 겪은 침묵을 그대로 물려받는다.

각 단계는 **오프라인 단위검정 + 계기 산출**까지가 완료 조건이고, 라이브 발화 확인은
다음 스모크에서 한 번에 받는다([[30]]·이 아크에서 침묵 4회를 겪은 규율).

## 8. 원장 반영

- `N97B_FIX_LEDGER_2026_08_05.md` §2e에 **F15(D0 배관 침묵)**·**F16(D1′ 체크리스트)**·**F17(D3-a 등재)**·
  **F18(문구 영어화)** 등재. §2f = 스모크 g 사전등록.
- **F16의 정의가 개정됐다**(2026-08-05 오후): *부재 표면화* → **실행 체크리스트 + 조건부 ▶NEXT**.
  여기에 `missing=<노드 id>` → **호출 가능한 도구명 + unlock 상태 + 자연어 질의** 교체가 포함되며,
  이는 **호출-시점 deny 문구에도 같이 적용**된다(051이 받은 그 메시지).
- `F14`(권한 밖 행동)는 D1′·D2가 아니라 **선언 분기**(051 2라운드) 소관임을 명시 — 혼동 방지.

## 9. 이 개정으로 **철회된 내 주장** (정직·[[03b]])

| 철회 | 대체 |
|---|---|
| *"051은 deny가 누락 단계를 **이름으로** 지목했는데도 호출하지 않았다"* | 지목된 것은 **노드 id**(`submit_request`)이지 호출 가능한 이름이 아니었다(§2.0) |
| 그로부터 도출한 *"알고도 안 한다 = 표면화는 약하다"* | **미확정.** 현 증거는 "위치도 이름도 못 받았다"와 구분되지 않는다. 구분은 x89가 한다 |
| 초판 D1의 근거였던 *"차단보다 약한 개입을 먼저"* | 순서 규율일 뿐 **효과의 증거가 아니었다.** D1′은 세 진단 중 무엇이 맞아도 사는 형태로 다시 세웠다(§2.6) |
