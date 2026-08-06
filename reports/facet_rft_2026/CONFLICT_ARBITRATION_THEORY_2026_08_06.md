# 레버 충돌의 일반 이론 — 전수 census · 마스터 규칙 · 남는 기구 (2026-08-06)

> 사용자 지시(2026-08-06·재론 금지): **"무조건 근거를 확보한 쪽이 우세하다."**
> 이 문서는 그 지시를 마스터 규칙으로 놓고, **우리가 겪은 충돌 전수**가 그 규칙으로 몇 종이나
> 덮이는지를 가른다. 덮이지 않는 종에는 별도 기구가 필요하고, 그것을 여기서 정한다.
>
> 선행 = `LEVER_ARBITRATION_ADJUSTMENT_DESIGN_2026_08_06`(계열 A/B/C·처방 보류 상태) ·
> `LEVER_ARBITRATION_PHASE_CONDITION_DESIGN_2026_08_06`(C17 단계 소유권) ·
> `N97_TASK_ROOT_CAUSE_2026_08_06`(태스크별 확정 원인).
> 계기 = `x112_same_tool_conflict.py`(완주 전수·우리 문구 1,949행) · `x111_silent_failure_census.py`.

---

## §0 왜 이론이 필요한가 [M]

문구를 고치는 방식은 **네 번 실패했다**. 고칠 때마다 다음 런에서 **다른 쌍**이 충돌했다
(핸드오프 2026-08-06 §0: *"레버들이 서로를 밀어낸다"*). 쌍마다 패치하면 N² 이고, 레버가
80개를 넘은 지금 그 방식은 닫히지 않는다.

이론의 요건은 넷이다:
1. **정의** — 무엇을 충돌이라 부르는가
2. **분류** — 종류가 몇 개이고 서로 다른 기구를 요구하는가
3. **중재** — 누가 이기는가
4. **잔여 방지** — 진 쪽을 지웠을 때 *아무도 말하지 않는* 상태를 만들지 않는가

---

## §1 ★우리 충돌 전수 census

증거 등급: **[M]** 실측 · **[D]** 단일 관측(n=1) · **[?]** 미측정 추정.

| # | 사건 | 두 레버 | 무엇을 두고 다투나 | 등급 | 출처 |
|---|---|---|---|---|---|
| 1 | **028** | follow-up 체인 ↔ pre-close deny | **공유 예산**(`T2_EPLAN_DENY_CAP`) | [M] | go_stack C173-corr |
| 2 | **012** | C8 레지스트리 문구 ↔ `usertoolnote` | **발화 슬롯**(먼저 응답한 쪽이 다른 쪽을 영영 막음) | [M] | HANDOFF 08-06 §5a |
| 3 | **050/051** | C13 반복 억제 ↔ 반복이 만들던 이행 | **다른 레버의 작동 기제**(반복 자체) | [M] | HANDOFF 08-06 §7-5 |
| 4 | **050** | GB1 "검증 먼저" ↔ 발견 체인 "지금 돌려라" | **행동 순서** | [M] | C17 설계서 |
| 5 | **032** | 발견 문구 "이관 마라" ↔ notice 게이트가 이관 지목 | **행동 선택** | [M] | C17 설계서 |
| 6 | **계열 A** (25턴·K2) | `[ACTION]` "손님 도구다" ↔ `unified_regen` "네 도구다" | **같은 명제의 진리값** | [M] | 조정설계 §1 |
| 7 | **계열 B** | `[PROCEDURE]`가 이름을 댐 ↔ `OPERATOR-PROVENANCE`가 그 이름 차단 | **이름의 출처 자격** | [M] | 조정설계 §2 |
| 8 | **계열 C** (109/198 sim) | — (양쪽 다 침묵) | **발화 창** | [M] | x111 |
| 9 | **102/101** | GB1 `applies_to ∋ submit_referral` ↔ `[ACTION]` push | **행동 순서 + 실행자 분리** | [M] | 본 세션 §2 |
| 10 | **004** | 종결 직전 개입 ↔ 터미널 턴 보장 | **마지막 턴** | [M] | PREKB 면제 도입 사유 |
| 11 | **005** | PREKB 면제 ↔ 면제가 전제를 검증 안 함 | **면제의 범위** | [M] | 부검 task_005 |
| 12 | **035** | C17 침묵 ↔ escalation 도달 | **행동 순서** | **[?]** | HANDOFF 08-06 §5b(귀속 미확정) |

### §1b 여섯 종으로 접힌다

| 종 | 다투는 대상 | 해당 사건 | 근거 등급 규칙으로 풀리나 |
|---|---|---|---|
| **T1 사실 모순** | 같은 명제의 진리값 | 6 | **아니오 — 중재 대상이 아니다** |
| **T2 순서 지배** | 행동의 선후 | 4·5·9·(12) | **예** |
| **T3 출처 오분류** | 근거 등급 자체의 오판 | 7 | **예**(규칙의 적용 오류) |
| **T4 자원 경합** | 유한 예산·발화 슬롯 | 1·2 | 아니오 |
| **T5 기제 억제** | 다른 레버의 작동 조건 | 3 | 아니오 |
| **T6 창 부재** | 아무도 말하지 않음 | 8·10·11 | 아니오(쌍대 문제) |

**⇒ 마스터 규칙은 12건 중 5건(T2+T3)을 덮는다.** 나머지 셋은 다른 기구가 필요하다.
이것이 이 문서의 핵심 결론이고, "근거 우세 하나로 끝난다"는 과잉 일반화를 막는 지점이다.

---

## §2 마스터 규칙 — 근거 등급 (T2·T3)

**규칙**: 두 레버가 같은 턴에 같은 표적에 대해 반대 방향을 말하면, **술어의 근거 등급이 높은 쪽이
말하고 낮은 쪽은 명령을 잃는다.**

| 등급 | 근거원 | 예 |
|---|---|---|
| **E1** | 우리 실행 원장 — 무엇이 실행됐고 무엇이 반환됐나 | GB1의 "`log_verification` 실행됨" |
| **E2** | 선언·정책 축자 | `procedures._quote_order` · `gates[].applies_to` |
| **E3** | env 레코드 출력 | [[25]]: env는 **외부 주장** |
| **E4** | 회수 KB 산문 | 검색 결과 문서 |
| **E5** | 모델 formalize · 손님 주장 | `[ACTION]` push의 표적 추정 |

### §2a "전제 지배"는 별도 규칙이 아니라 따름정리

102에 대입: GB1 술어 = **E1**(원장에 `log_verification` 없음) · `[ACTION]` push 술어 = **E5**
(formalize 서브콜이 산문에서 뽑은 표적) ⇒ 게이트가 이긴다. *"전제니까"* 가 아니라 *"근거가 단단하니까"* 다.
⇒ **규칙을 둘로 세우지 않는다.**

### §2b 이 규칙은 이미 한 곳에서 시행 중이다 (발명이 아님)

`a2/banking_knowledge.specific.json` `procedures[]._note_enforce` 축자:

> 정책 축자가 "MUST be followed in the exact order listed"이므로 순서 강제의 허가가 있다.
> 이 필드가 없거나 `_quote_order`가 없으면 엔진은 **표면화만 하고 차단하지 않는다**

**근거 등급이 있으면 차단, 없으면 표면화** — 이미 하고 있다. 일반론은 이 패턴을 **모든 레버로 확장**하는 것이다.

### §2c 따라오는 세 귀결

1. **명령은 하나, 사실은 합집합** — 진 레버의 *명령*만 버리고, 그 레버만 아는 *사실*은 평서문으로 남긴다.
   완전 침묵은 T6을 만든다(012에서 실측).
2. **레버 구성 규칙** — 레버는 자기 근거 등급을 **선언**해야 한다. 못 대면 **deny 자격 없음·표면화만**.
   ⇒ [[22]]의 닫힌/열린 술어 경계가 중재 층에서 자동 집행된다.
3. **경계** — 이 규칙은 **누가 말하는가**만 정한다. **누가 옳은가**는 보장하지 않는다.
   020은 E1 근거를 갖고도 우리 산식이 틀려 없는 불일치를 만들었다. 정확성 의무는 [[25]]가 진다.

### §2d T3(출처 오분류)이 같은 규칙으로 풀리는 이유

계열 B에서 `OPERATOR-PROVENANCE`는 우리 절차 선언이 방금 말해 준 이름(**E2**)을
모델 날조(**E5**)로 취급했다. 새 규칙이 아니라 **등급 판정에 우리 층을 넣는 것**이다([[25]]).

---

## §3 마스터 규칙이 덮지 않는 세 종

### T1 사실 모순 — **중재하지 말고 제거하라**

계열 A(`[ACTION]` "손님 도구" ↔ `unified_regen` "네 도구")는 우선순위 문제가 **아니다**.
둘 중 하나가 **틀렸고**, 우선순위를 정하면 틀린 문장이 살아남는 경우가 생긴다.

**기구**: 같은 명제를 말하는 모든 문구는 **하나의 술어 함수**에서 답을 받는다. 판정 불가면
**그 문장을 뺀다**. (조정설계 §1 처방 A와 동일. 도구 소속은 `_user_discoverable` /
`env.user_tools.tools` / `self.tools` 세 집합의 **3갈래 판정**으로 닫힌다.)

> **불변식 I1**: 우리 층이 같은 명제에 대해 서로 다른 진리값을 말하는 턴 수 = **0**.

### T4 자원 경합 — **분리하라 (선례 있음)**

028은 우선순위로 풀지 않았다. **전용 예비 예산 분리**(`T2_PRECLOSE_CAP`)로 풀었다.
012는 발화 슬롯 경합인데 아직 안 풀렸다.

**기구**: 유한 자원을 공유하는 레버들은 ⓐ**분리 예산**을 갖거나 ⓑ**우선순위 큐**를 갖는다.
슬롯 경합(012)의 규칙은 §2c-1과 같다 — **하나만 명령하고 나머지는 사실로 합류**하므로
"먼저 응답한 쪽이 다른 쪽을 영영 막는" 구조 자체가 사라진다.

### T5 기제 억제 — **부작용을 선언하게 하라**

C13(반복 억제)은 050/051에서 **이행을 만들던 반복 자체**를 지웠다. 근거 등급과 무관하다 —
C13은 다른 레버와 *같은 명제*를 다투지 않고, 다른 레버의 **작동 조건**을 없앴다.

**기구**: 억제형 레버는 자기가 **무엇을 지우는지 선언**해야 한다. 지워지는 대상이 다른 레버의
발화 조건이면 **금지**하거나 그 레버를 면제한다.

> **불변식 I2**: 억제 레버의 대상 집합 ∩ 다른 레버의 발화 조건 = **∅**(또는 명시 면제).

### T6 창 부재 — **중재의 쌍대 문제**

실패 198 sim 중 **109(55%)** 가 커밋된 우리 개입 0건이다. 충돌이 아니라 **양쪽 다 침묵**이다.
원인은 술어가 아니라 **창** — push 레버군이 사임 턴(도구 없는 산문)에서만 발화하는데
모델은 **행동(이관·write)으로 나간다**.

**기구**: 발화 창 = `사임 턴` ∪ `종결성 행동을 실행하려는 턴`(조정설계 §3 처방 C).
⚠ 004형 *"마지막 턴 소각"* 재현 위험이 있으므로 **차단 없이 표면화만**으로 시작한다.

---

## §4 조건 그래프 — 순서 지배(T2)의 자료구조

T2를 쌍별 패치가 아니라 **도달가능성 질의**로 바꾸려면 순서 선언이 하나의 그래프여야 한다.
**그 그래프는 이미 존재하는데 다섯 키로 조각나 있다** — 그리고 레버 다섯 개가 각자 자기 것만 읽는다.

| A2 키 | 간선 | 읽는 레버 |
|---|---|---|
| `gates[].satisfiers / applies_to` | `log_verification ≺ {submit_referral, …}` | 정책 게이트 |
| `require_tool_before` | `get_all_user_accounts_by_user_id ≺ submit_referral` | PIN-READ / PREKB |
| `scaffold_get_tools[].requires_reads` | 동종 | 동일 레버(뒤늦게 합류) |
| `procedures[].steps` (+`enforce`) | 절차 내부 순서 | PROCEDURE |
| `follow_up_chains[].after / requires` | 사후 필수 체인 | FOLLOW-UP |
| `require_doc_before` | `문서 열람 ≺ transfer` | PROTOCOL |

**★결정적 관측 [M]**: `submit_referral`에는 **간선이 이미 둘** 선언돼 있다(GB1 · `require_tool_before`).
**둘 다 발화하지 않았다.** 이유가 같다 — 두 레버 모두 *에이전트가 그 도구를 부르는 순간*에 붙어 있고,
`submit_referral`은 에이전트가 부르지 않는다. `t2_pin_read.py:104` 축자:
*"그 read를 요구하는 의존 도구를 **이미 시도했다** — 호출 이력"*.

⇒ **한 레버의 버그가 아니라 아키텍처의 성질이다.** 선언은 *행동*을 덮는데 집행은 *채널*에 붙어 있다.

### §4a 필요한 형태 — 순수 DAG가 아니다

1. **가드된 간선**: GB1 축자 *"Transfer/incident tools do not need verification."* 면제가 실재하고,
   그 면제가 **사이클 차단기**다(검증하려면 계좌 읽기 필요 ↔ 계좌 읽기가 검증 요구).
   노드를 read/write로 **타이핑**하고 간선에 가드를 달아야 평가 후 비순환이 된다.
2. **실행자는 간선에 넣지 않는다**: 넣는 순간 지금 버그가 재발한다. 간선은 **행동**에만 걸고,
   실행자에 따라 **집행 원시연산만** 바꾼다 —
   에이전트 실행 → *호출 거부* / 손님 실행 → **발화 교체**(그 행동을 지시하는 문장을 요건 문장으로).
3. **비교 불가 노드**: 경로가 없으면 그래프는 침묵한다 → §2 근거 등급이 타이브레이크.

---

## §5 선행연구 대비 위치 (딥리서치 1/3 회신분)

**반드시 인용하고 차별화해야 할 근접 선행 — WIRE (arXiv:2605.27784, 2026-05)**
문제 진술이 우리 문장과 같다: *"individually reasonable standing rules can interact in uninspected ways."*
6개 공개 정책 276 규칙 → SAT hard-collision 170쌍 → witness 실현. **joint compliance 35.4%**.
**한계 넷이 우리 자리**: ⓐ정적 정책 텍스트만(런타임 발화 아님) ⓑ**arbitration 없음**(저자가 진단
파이프라인이라 명시) ⓒ**hard require/forbid 쌍만** — soft push는 모순으로 세지도 않음
ⓓ**precondition/gate 계층 없음**.

**동급 충돌에 대한 문헌의 유일한 명시 규칙 = 위치 규칙** — ManyIH(arXiv:2604.09443):
*"follow the one that appears later."* **우리 사례에서 이 규칙은 오답을 낸다**(push가 뒤에 오면
push가 이기고, 그것이 실측된 실패다). ⇒ **근거 등급 규칙의 직접 foil**이고, 반례 실험이 방어의 핵심.

**런타임 집행 계열은 중재 의미론이 없다** — AgentSpec(ICSE 2026·arXiv:2503.18666)은 우리와 동형
구조(trigger/predicate/enforcement)인데 다중 규칙 의미론이 "활성 규칙 순차 적용"이 전부.
FORGE(arXiv:2602.16708)의 **deny-wins**만이 부분적 답이고 allow/deny 이항 격자에 한정된다 —
우리 push는 deny가 아니라 obligation 형태라 그 격자에 들어가지 않는다.

**★경쟁자이자 baseline — PolicyGuard (arXiv:2606.29225)**: τ²-bench 위에서 **우리 Lever A를 이미
구현**(identity verification·prerequisite read·확인·절차 순서 검사 + 결손 지목 remediation).
pass⁴ **+12.0pp**(GPT-5.4). 그런데 그 게이트도 **mutating tool call에만** 걸리므로,
에이전트가 *"고객님이 직접 실행해 주세요"라고 말하는 경로*는 **구조적으로 통과**한다 = 우리가 찾은 구멍.

**완전 공백 — enforcement point ≠ executor**: 이름조차 없다. τ²-bench(arXiv:2506.07982)가
dual-control을 만들었지만 오류 축이 *reasoning vs communication*이라 이 실패 양식을 명명하지 않았다.
HITL 문헌은 전부 반대 방향(사람이 에이전트를 승인)만 다룬다. 게다가 이 경로는 **비가역성이 더 나쁘다**.

**끄는 것이 답이 아니라는 실증**: Roig(arXiv:2512.07497) — 명시적 절차 지시 추가로 52.92%→87.50%.
[[19]]("끄기가 아니라 조정")이 문헌으로 뒷받침된다.

⚠**인용 금지(primary source 미확인)**: tool-count 스케일링 수치 · reminder 누적 충돌 수치 ·
Safeguarding 서베이 conflicting-requirements 절 · τ²-bench dual-control 하락 수치.

⏳ 미회신: 접근제어 policy-combining / 생산규칙 conflict-set / 런타임 집행 원시연산.

---

## §5b ★규범 충돌 문헌 — **고전 3원칙이 우리 사례에서 전부 실패한다** (직접 정독분)

정본 = Santos, Zahn, Silvestre, Silva, **Vasconcelos**. *Detection and Resolution of Normative
Conflicts in Multi-agent Systems: A Literature Survey.* **AAMAS 2018 (JAAMAS Track), pp.1306–1309**
(PDF 직독 · `ifaamas.org/Proceedings/aamas2018/pdfs/p1306.pdf`).

**충돌의 정의**(축자): *"A conflict between norms is a situation in which the fulfillment of a norm
causes a violation of another one."* — 우리 사례가 정확히 이것이다. push를 이행하면(= 손님에게 지금
실행하라고 말하면) 게이트가 위반된다.

**분류**: **직접**(같은 행위·같은 에이전트·반대 deontic 양상) vs **간접**(요소가 같지 않고 *관련*됨 —
`Oq`와 `Fp`이고 `q → p`인 경우, **또는 같은 양상인데 동시에 수행 불가한 두 의무**).
⇒ **우리 T2는 간접 충돌**이다(둘 다 의무인데 순서 때문에 동시 이행 불가).

**해결의 두 계열**(축자 요약): **norm prioritization**(하나가 다른 하나를 override) vs
**norm adjustment**(충돌하는 norm 중 하나를 *변경* — 제약 추가·annotation으로 영향 범위 축소).

### ★핵심 — 고전 3원칙과 우리 사례

문헌이 인정하는 우선순위 원칙은 **셋뿐이다**: **lex posterior**(가장 최근) · **lex specialis**(가장
구체적) · **lex superior**(가장 권위 있는 발령자).

| 원칙 | 우리 102 사례에 적용하면 | 판정 |
|---|---|---|
| **lex posterior** | 나중에 온 문구가 이긴다 = ManyIH의 위치 규칙 | **오답** — push가 뒤에 오면 push가 이기고, 그것이 실측된 실패다 |
| **lex specialis** | 더 구체적인 쪽이 이긴다 → push가 **도구 이름과 인자까지** 지목하고, 게이트는 *account actions* 라는 **클래스**를 말한다 | **오답** — 더 구체적인 쪽이 틀린 쪽이다 |
| **lex superior** | 발령 권위가 높은 쪽 → **둘 다 우리 scaffold가 발령**했다 | **무승부** — 답을 주지 못한다 |

⇒ **세 원칙 중 둘은 오답, 하나는 무승부.** 사용자 지시(근거 우세)는 이 셋 중 어느 것도 아니다 —
형태는 lex superior에 가깝지만 권위가 **제도적 서열**이 아니라 **술어 근거의 인식적 질**이다.
**제안 명칭: lex probationis(증거 우선) = 근거 등급 원칙**, 이 분류표의 **네 번째 원칙**.

서베이 자신의 결론(축자): *"there is no single detection/resolution method that is best"* —
새 원칙을 제안하는 것이 이 분야에서 이례적이지 않음을 뒷받침한다.

### §5b-2 왜 push가 애초에 발화하면 안 되는지 — **탈착(detachment)** 프레임

SEP *Deontic Logic*의 **Factual Detachment**: `p ∧ OB(q|p) ⊢ OB q`.
push 레버는 조건부 의무 `OB(손님에게 실행 안내 | 전제 충족)`에서 **`p`(전제 충족)를 확인하지 않고**
`OB q`를 떼어낸다. 즉 우리 push 레버군의 결함은 **전제 미확인 탈착**으로 정확히 명명된다.

또한 SDL의 **NC 원리** `¬(OB p ∧ OB ¬p)`는 의무 충돌 자체를 금지하고, 허용하면 **deontic explosion**
(모든 것이 의무가 됨)이 따른다. 우리 스캐폴드는 실제로 그 상태에 있었다 — 같은 턴에 서로를 위반하는
두 의무를 내보냈고, 모델은 PRIME이 기술한 대로 **조용히 한쪽을 골랐다**.

### §5b-4 ★행위이론 — **집행 지점 비대칭의 정확한 이름이 이미 있다**

**⚠정정**: 앞 절(§5)에서 *"치환(replacement)은 whitespace"* 라고 썼다. **부분적으로 틀렸다** —
원리는 **Cohen & Levesque 1991**이 선점했다(아래 ③). 우리 몫은 원리가 아니라 **귀속 규칙**이다.

**① 우리 게이트의 정확한 분류 — regimentation vs regulation** (Jones & Sergot 1993;
정독본 = Noriega, Chopra, Fornara, Lopes Cardoso, Singh, *Regulated MAS: Social Perspective*,
Dagstuhl Follow-Ups Vol.4, 2013, pp.93–133 · DOI 10.4230/DFU.Vol4.12111.93):

> *"regimentation arises in a system that forces or precludes certain actions whereas regulation
> arises in a system that neither forces nor precludes"*
> *"Since the parties are autonomous, regimentation is out of the question"*

⇒ **한 문장 진단**: 우리 GB1은 **regimented perimeter 바깥의 행동을 겨냥한 regimentation 기구**다.
`submit_referral`은 자율적 타 행위자가 실행하므로 원리적으로 regiment 불가이고, 사용 가능한 유일한
도구는 **regulation**(규범 + 감시 + 소통)이다. 우리가 "부착점이 없다"고 부른 것의 문헌 명칭이 이것이다.

**② 실행자가 바뀌면 태도의 *종류*가 바뀐다 — 선점됨** (Grosz & Kraus 1996,
*Collaborative plans for complex group action*, **AIJ 86(2):269–357**):

> *"an agent only has intentions-to toward acts for which it is the agent; intentions-that represent
> its responsibilities with respect to the actions of other agents"*
> *"An agent can only adopt an intention-to toward an action for which it is the agent."*

그리고 Int.Th의 **해소 방식은 소통**이다: *"agents … are required to provide information about their
progress to each other."* contracting-out 구성(§FIP)이 정확히 우리 구조다 —
G는 자기가 할 수 있는 행위에 **Int.To**, 타자가 할 행위에 **Int.Th**.

⇒ **양보해야 할 것**: *"실행자가 바뀌면 태도 종류가 바뀐다"*는 **우리 발견이 아니다.** 인용 의무.
⇒ **우리 몫**: 이것의 **집행-층 유비**(deny → 발화-수준 요건으로 **type-demotion**)와, 그것이
다른 레버와 **같은 턴에 경합할 때**의 규칙. G&K는 중재를 다루지 않는다.

**③ "아무도 말하지 않음"의 원리도 선점됨** (Cohen & Levesque 1991, *Teamwork*, **Noûs 25(4):487–512**):
**WAG**(Weak Achievement Goal) — 목표가 이행 불가임을 알게 된 에이전트는

> *"should be left with a goal to make this fact known to the team as a whole"*

⇒ **이행 불가가 된 커밋먼트는 소멸하지 않고 *공표 의무로 전환된다*.** 우리 §2c-1(명령은 하나·사실은
합집합)과 §3-T6 처방의 **원리**가 이것이다. **양보하고 인용한다.**
⇒ **우리 몫**: C&L은 의무가 존재함을 말하지만 **N개 레버 중 누가 그 의무를 물려받는가**는 말하지 않는다.
**귀속 규칙(inheritance rule)이 우리 발명분**이다.

**④ 고전 계획 이론은 실행자 슬롯이 아예 없다** (검증된 부정 결과):
- Fikes & Nilsson 1971 STRIPS: `ROBOT`이 전제 논리식 안의 **상수**다 — 실행자를 추상화하지 않는다.
- PDDL 1.2 매뉴얼 전문에서 문자열 `agent` **0회**. PDDL 2.1 Def.10: *"a is applicable in a state s
  if the Pre_a is satisfied in s."*
- Lin, *Situation Calculus*(Handbook of KR, 2008) p.663 — **최고의 foil 한 문장**:
  > *"Thus whether A(x₁,…,xₙ) can be performed in a situation s depends entirely on s."*
- **MA-PDDL**(Kovacs 2012)은 실행자를 말하려고 **`:agent`라는 새 문법 슬롯을 추가해야 했다.**
  ⇒ "누가 실행하는가"는 **알려진 표현적 공백**이라는 실증.

**⑤ 선언 범위가 통제 불가 사건에 닿는 것을 *탐지*하는 선례** — STNU **dynamic controllability**
(Morris·Muscettola·Vidal IJCAI-01; 정의 정독 = Hunsberger et al. arXiv:1212.2005):
contingent link `(A,x,y,C)`에서 *"the execution of C is out of the agent's control."*
⇒ 우리 결함의 형식적 유비지만, **답이 "명세를 기각한다"**이지 런타임 중재가 아니다.

**⑥ ⚠우리 중재가 재현할 수 있는 실패** (Thielscher 2001, *AIJ* 131(1–2):1–37):
`Ab` 원자의 순진한 최소화는 **하류 실패를 피하려고 상류 실패를 발명하는** anomalous model을 낳는다 —
> *"there exists a minimal but anomalous model where DisableEmission is qualified in the first place.
> This without any actual reason at all"*

⇒ 우리 지배 규칙도 같은 형태를 만들 수 있다: 모델이 **push를 지배당하지 않게 하려고 상류 게이트를
회피**하는 경로. 사전 등록 계량에 넣어야 한다.
또한 그의 **prioritized default theory `(D, W, <)`** 는 우리 근거 등급의 **형식적 기반**으로 쓸 수 있다
(부분 순서 + extension 선택 의미론).

**⑦ 완전 미청구 — 단일 출력 채널 경합**: 어떤 행위이론도 지시들이 **하나의 발화 슬롯을 두고 경합**하는
것을 모델링하지 않는다. 우리 T4·T5가 바로 그 자리이고, **선행 없음**.

⏳ 미조사(다음 우선순위): **STIT**(Horty/Belnap — 행위성이 원시 개념인 유일 형식체계) ·
**reactive synthesis realizability**(Pnueli & Rosner — *"통제 불가 입력을 제약하는 명세는 실현 불가"*
= 우리 Lever A 결함의 가장 타이트한 형식 진술 후보) · Governatori & Rotolo defeasible deontic ·
Castelfranchi & Falcone 위임 이론(유료·미확인).

### §5c ★우리 규칙의 형식적 정체 = **순수 사전식 순서**, 그리고 그 알려진 실패가 T6이다

**근거 등급 E1≻E2≻E3≻E4≻E5는 형식적으로 lexicographic order다.** 이 계열은 다중목적 최적화에서
잘 연구돼 있고, **두 실패 모드가 정리로 알려져 있다.**

**① 가중합(scalarization)은 완전 보상적이라 안전이 매수당한다** — 우리가 "레버에 가중치를 주자"로
가지 않은 것이 옳았다는 근거:
- Arman, Arman, Hadi-Vencheh, *Comput. Intell. Neurosci.* 2022:8629986 (§1·PMC9420562 원문 검증):
  *"compensatory models consider the trade-offs between attributes … the strength of an attribute can
  compensate for the weakness of the other attribute."* — 가중합(WSM)을 **보상적**으로 분류.
- Shakerinava, Ravanbakhsh, Oberman, arXiv:2505.12049 App.B: 벌점형 `min C + λR`은
  *"results in a path that spends a nonzero amount of time in the unsafe region to reach the goal faster."*
- **형식적 분리**: Wray, Zilberstein, Mouaddib, **AAAI-15** Prop.6 — *"The optimal policy of an LMDP π
  may not exist in the space of solutions captured by its corresponding scalarized MOMDP's policy π_w."*
  ⇒ **사전식 최적은 어떤 가중치로도 재현 불가.** 레버 우선순위를 점수화하려는 시도는 원리적으로 틀린다.

**② ★순수 사전식은 *굶긴다* — 이것이 정확히 우리 T6이다**
- Wray et al. AAAI-15 §1: *"lexicographic order of objectives can be too rigid, not allowing any
  trade-offs between objectives."*
- Tercan & Prabhu, **ECAI 2024**(arXiv:2408.13493) §1 — 기전 축자:
  > *"Given two candidate solutions, a lower ranked objective is used to rank the two solutions only if
  > all higher order objectives have the same values."*

  상위 목적이 **정확히 같을 때만** 하위가 발언한다 ⇒ 하위는 사실상 영영 말하지 못한다.
  **내가 §3-T6에서 "마스터 규칙이 T6를 악화시킨다"고 쓴 것이 이 정리의 특수 사례다.**

**③ 그리고 표준 해법이 있다 — 문턱(threshold)/여유(slack)**
- **TLO**(Gábor, Kalmár, Szepesvári, **ICML 1998**): 상위 목적을 임계에서 **자른다** —
  `CQ ← min(Q, C_j)`. 저자 축자: *"Since the evaluations are cut at R_crit we may expect that
  v_{π1,1}(x) and v_{π2,2}(x) will be equal in a large number of cases."*
  ⇒ **자르기가 동률을 인위적으로 만들어 하위 목적이 비로소 작동한다.**
  정본 알고리즘 = Vamplew, Dazeley, Berry, Issabekov, Dekker, *Machine Learning* **84**(1–2):51–80, 2011 §3.2.3.
  요약 = Roijers, Vamplew, Whiteson, Dazeley, *JAIR* **48**:67–113 (2013) §6.1:
  *"State-action values for each objective that exceed the corresponding threshold are clamped to that
  threshold value prior to applying the lexicographic ordering."*
- 대안 = **여유**(Wray 2015 δ · Skalse, Hammond, Griffin, Abate **IJCAI-22** ε).
  ⚠Tercan & Prabhu §2: Skalse의 ε는 매우 작아야 하므로 **satisficing 의미론을 주지 않는다** — 우리에겐 문턱 쪽이 맞다.

### §5c-2 ⇒ 우리 설계에 대한 직접 귀결 (T6 미해결 → **부분 해결**로 승격)

**우리의 "문턱"은 이미 설계 안에 있다 — 다만 이름이 없었다.** §2 지배 규칙은 무조건이 아니라
**표적-조건부**다: 게이트는 *자기 `applies_to`가 실제로 덮는 표적*의 push만 지배한다. 그 밖에서는
아무도 침묵시키지 않는다. 이것이 TLO의 clipping과 같은 역할을 한다 — **지배의 범위를 잘라 하위 레버가
말할 수 있는 구간을 남긴다.**

⇒ 규칙의 정확한 서술을 이렇게 고친다(무조건 → 범위-제한):

> **근거 등급이 높은 레버는, 자기 선언 범위가 덮는 표적에 한해서만, 낮은 레버의 명령을 박탈한다.**
> 범위 밖에서는 등급 차이가 있어도 박탈하지 않는다.

이건 사용자 지시(*"무조건 근거 확보한 쪽이 우세"*)를 **약화**하는 것이 아니다 — 지시는 *충돌할 때
누가 이기는가*를 정하고, 여기서 자르는 것은 *애초에 충돌인가*의 판정이다. 범위가 겹치지 않으면
그건 충돌이 아니다(§5b 축자 정의: *"the fulfillment of a norm causes a violation of another one"*).

**⚠남는 위험(문헌이 경고)**: 문턱 근처의 추정 잡음이 결정을 뒤집는다 —
Vamplew et al. arXiv:2402.06266: *"Even a minimal level of overestimation is sufficient to cause a
substantial number of incorrect decisions."*
⇒ **근거 등급은 추론하지 말고 레버가 선언해야 한다**(§2c-2). 등급을 술어에서 추정하면 이 절벽에 걸린다.

### §5b-3 우리 기구가 문헌의 어디에 대응하는가

| 우리 기구 | 문헌 대응 | 선점 여부 |
|---|---|---|
| §2 근거 등급 | (없음) — 3원칙 밖 | **신규**(lex probationis) |
| §3 T1 제거(단일 술어) | — (사실 모순은 규범 충돌이 아님) | 분야 밖 |
| §3 T4 예산 분리 | **norm adjustment** — 제약 추가로 영향 범위 축소 | **선점**(Vasconcelos 계열) → 인용 |
| §2c-1 치환(명령은 하나·사실은 합집합) | norm adjustment에 **부분적으로** 가까움(단 문헌의 adjustment는 *적용 범위* 변경이지 *메시지 내용* 치환이 아님) | **부분 선점** — 정직하게 양보하고 차이를 명시 |
| §3 T5 부작용 선언 | (없음) | 신규 |
| §4 조건 그래프 | 제약/ontology 기반 탐지 계열과 인접 | 부분 선점 |

---

## §6 [[05]] 3질문 (설계서 상설·[[17]])

1. **무엇이 고정인가**: 모델 · scaffold 엔진 구조 · A2 층 구조. 중재는 **기존 레버의 발화 자격**을
   정하는 층이고 새 도메인 지식을 넣지 않는다.
2. **무엇이 변하는가**: A2 **순증 0** — 간선(`gates`/`require_tool_before`/`procedures`/
   `follow_up_chains`/`require_doc_before`)은 이미 선언돼 있고 어휘만 통일한다.
   엔진은 ⓐ소속 판정 단일 함수 경유 ⓑ근거 등급 선언 필드 ⓒ도달가능성 질의 한 곳.
3. **전이되는가**: 도메인 리터럴 0 — 레지스트리(프레임워크 API) · A2 선언 키 · 실행 원장만 쓴다.
   **부채 상환**: `t2_phase.py:79`의 엔진 리터럴 `{"verify_identity"}`가 도달가능성 질의로 대체돼 사라진다.

---

## §7 구현 순서와 사전 등록 계량

| 순서 | 처방 | 종 | 통과 조건(사전 등록) |
|---|---|---|---|
| 1 | **T1 제거** — 도구 소속 3갈래 단일 술어 | T1 | 소속 주장 중 레지스트리와 어긋난 건수 **0** |
| 2 | **근거 등급 선언 + 순서 지배** — push 표적이 미충족 간선의 하류면 명령 상실·요건으로 치환 | T2 | ⓐ치환 발화 수 ⓑ그 중 **gold이 그 행동을 요구한 sim = 오발화** ⓒ침묵(=치환 실패) **0** |
| 3 | **T3 출처 집합에 우리 층 포함** | T3 | 레지스트리 밖 이름 통과 **0** |
| 4 | **T6 창 확장**(차단 없이 표면화) | T6 | 새 발화 sim 중 gold이 그 종결을 요구한 비율 |
| 5 | T4 슬롯 규칙 · T5 부작용 선언 | T4·T5 | 불변식 I1·I2 오프라인 검정 |

⚠**[[09]]**: 1~3은 오프라인 전수(224 sim) 계량을 먼저 낸다. 라이브는 **누적 스택 한 번**으로 확인한다
(사용자 지시: 별도 코드 실험 금지·스택 누적).
⚠**[[19]]**: 간섭은 합성 런에서만 드러난다 — 처방을 하나씩 격리 발사하지 않는다.

---

## §8 아직 측정되지 않은 것 (추정으로 메우지 않는다)

- **035의 C17 귀속** [?] — 격리 arm 없이는 확정 불가(핸드오프 §5b).
- **T4 슬롯 경합의 규모** — 012 한 건 말고 전수에서 몇 건이 "먼저 응답한 쪽이 막았는가"는 미계량.
- **치환(replacement)의 순응률** — 요건 문장으로 바꿨을 때 모델이 따르는 비율은 라이브 전 미지.
- **근거 등급의 총 순서성** — E1~E5가 모든 레버 쌍에 대해 비교 가능한지는 전수 선언 후에 확인된다.
