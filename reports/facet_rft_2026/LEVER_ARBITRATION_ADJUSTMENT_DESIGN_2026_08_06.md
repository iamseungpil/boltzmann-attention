# 레버 중재 조정 설계 — 같은 턴에 반대로 말하는 문구들 (2026-08-06)

> 근거 = `x112_same_tool_conflict.py`(완주 전수 `20260806b`·우리 문구 1,949행) ·
> `x111_silent_failure_census.py`(실패 198 sim) · 태스크 부검 `N97_TASK_ROOT_CAUSE_2026_08_06`.
> 선행 = `LEVER_ARBITRATION_PHASE_CONDITION_DESIGN_2026_08_06`(§7.2e-정정의 86턴 수치).

## §0 무엇이 실제로 있는가 [M]

| | 완주 전수 |
|---|---|
| 우리 문구가 나간 턴 | 896 |
| 같은 도구에 DO·DONT가 함께 온 턴 | **81** (sim 52) |
| ├ **K3 DO 쪽이 push 레버가 아님** | **57** |
| └ **K2 push ↔ 다른 사실 주장** | **25** |
| K1 push ↔ 정책 게이트 | **0** |

**★선행 수치의 정정**: 설계서 §7.2e-정정은 이 현상을 *"조정 문제가 86턴 규모"* 로 요약했다.
완주분에서 성격을 갈라 보면 **그 대부분이 지시 모순이 아니다** — 최대 쌍인
`PROTOCOL ↔ unified_regen` **34턴**은 축자로 이렇다:

```
[PROTOCOL]  You are about to use transfer_to_human_agents, but nothing you retrieved in this
            conversation is the document that defines it.
[POLICY GATE GB2_NOTICE_BEFORE_TRANSFER] blocked by policy gate: the pre-transfer notice
            (ask-first) has been communicated to the user not established
```
둘 다 **같은 방향(제지)** 이다. PROTOCOL은 "너는 지금 X를 하려 한다"는 **서술**이고 권유가 아니다 —
`use`라는 단어 때문에 DO로 잡혔을 뿐이다.

⇒ **조정 대상은 81턴이 아니라 그보다 작다.** 남는 것은 두 계열이고, 처방이 서로 다르다.

---

## §1 계열 A — **채널에 대해 서로 다른 사실을 말한다** (K2의 핵심)

축자(같은 턴·같은 도구 `submit_transaction`):
```
[ACTION]        'submit_transaction' is run by the CUSTOMER, not by you, and needs no
                agent-side KB procedure - STOP searching. In your next message tell them…
[unified_regen] 'submit_transaction' is one of YOUR OWN agent tools - it cannot be given to
                the customer, and the environment will reject it.
```
**두 문구 중 하나는 틀렸다.** 그리고 이건 우선순위 문제가 아니라 **정본 오염**이다([[25]]):
모델은 우리 말을 근거로 삼는데, 우리가 같은 도구의 소속을 두 가지로 말한다.

### 처방 A — 채널 주장의 단일 출처화 (엔진 위생·플래그 없음)

- 도구의 **소속**(에이전트 도구 / 손님-실행 도구 / discoverable)을 말하는 모든 문구는
  **하나의 함수**에서 답을 받아 쓴다. 그 함수는 이미 있다 — `t2_prekb_patch._is_agent_regular(env, name)`
  와 `_agent_discoverable(env)`가 **라이브 레지스트리**를 읽는다(C182c에서 같은 이유로 도입됐다).
- 문구 생성 시점에 그 함수의 답을 넣고, **함수가 판정 불가면 소속을 말하지 않는다**(문장을 뺀다).
- ⇒ 도메인 리터럴 0·A2 순증 0·gold 미참조. 새 레버가 아니라 **기존 문구의 출처 통일**이다.
- 계량(사전 등록): 224 sim 오프라인에서 *"우리 문구가 소속을 주장한 건수"* 와 그 중
  **레지스트리와 어긋난 건수**. 후자가 0이 되는 것이 통과 조건.

---

## §2 계열 B — **우리 절차가 이름을 댄 도구를 우리 출처-가드가 막는다**

축자(같은 턴·`submit_credit_limit_increase_request_7392`):
```
[PROCEDURE]             You are inside credit_limit_increase and its next step has not been
                        taken (2 of 7 done): … [ ] submit_request ->
[OPERATOR-PROVENANCE]   tool name 'submit_credit_limit_increase_request_7392' was not
                        discovered from any prior search/listing result — do NOT invent…
```
출처 가드의 취지는 **모델이 이름을 지어내는 것**을 막는 것이다. 그런데 이 이름은 모델이 지어낸 게
아니라 **우리 절차 선언이 방금 말해 준 것**이다. 즉 가드의 "출처" 집합에 **우리 자신이 빠져 있다**.

### 처방 B — 출처 집합에 우리 층을 포함 (엔진 위생·플래그 없음)

- `OPERATOR-PROVENANCE`/`UNLOCKED-NOT-CALLED` 계열의 출처 판정은 지금 **검색·목록 결과**만 본다.
  거기에 **우리가 이 대화에서 이미 이름을 말한 문구**(절차 체크리스트·발견 문구·CALLABLE_HINT)를 더한다.
- 근거: [[25]] — 우리 scaffold가 정본이다. 정본이 말한 이름을 정본이 "출처 없음"으로 막으면
  **모델에게 남는 선택지가 없다**(그 턴에 할 수 있는 행동이 0이 된다).
- ⚠ 이 확장이 날조 차단을 약화시키지 않는 이유: 추가되는 것은 **우리 문구의 축자 토큰**뿐이고,
  우리 문구는 레지스트리에서 파생된다(처방 A와 같은 출처).
- 계량: 전수에서 ⓐ이 확장으로 통과하게 되는 건수 ⓑ그 중 **레지스트리에 없는 이름이 통과하는 건수**(=0이어야 함).

---

## §3 계열 C — 발화 창: **사임 대신 행동으로 나가는 경로** (후보 L)

`x111` 실측: 실패 198 sim 중 **109(55%)** 가 궤적에 커밋된 우리 개입이 0건이고, 사이드카가 켜진 arm의
48건은 **전부 reminder만 나갔다**. 원인은 술어가 아니라 **창**이다 — `GIVE_EXEC`·`SEARCH_EXHAUST`·
`UNCALLED_UNLOCK`·A2 `resign_th` 사슬이 모두 **사임 턴(도구 없는 산문)** 에서만 발화한다.
부검이 그 대가를 셋 보여 준다: 017(분쟁 없이 보상 갱신) · 019(확정 행 목록 미갱신) · 022 t0(판정 직후 이관).

### 처방 C — 창을 "종결성 행동"까지 확장

- 발화 창 = `사임 턴` ∪ **`종결성 행동을 실행하려는 턴`**. 종결성 행동 =
  ⓐ프레임워크 종결 도구(`transfer_to_human_agents`) ⓑA2 `eplan.finalize_writes` ⓒ선언된 절차의 종단 write.
  세 집합 전부 **이미 선언돼 있다**(엔진 리터럴 0).
- 그 턴에서는 **차단이 아니라 1회 표면화**(기존 문구 그대로)로 시작한다 — 순응 여부를 먼저 측정한다.
- 상한: 기존 cap 규약 유지(레버별 sim당 1회). 창만 넓히고 총량은 늘리지 않는다.
- 계량(사전 등록): ⓐ확장으로 새로 발화하는 sim 수 ⓑ그 중 **gold이 그 종결을 요구한 sim**(=오발화)
  ⓒ발화 후 행동이 바뀐 비율. ⓑ가 크면 창 대신 **표적 조건**을 좁힌다.

---

## §4 [[05]] 3질문 (설계서 상설 항목·[[17]])

1. **무엇이 고정인가**: 모델·scaffold 엔진·A2 층 구조. 처방 A·B는 **기존 문구가 사실을 어디서 얻는지**를
   통일하는 위생 수정이고, C는 **기존 레버의 발화 창**만 넓힌다. 새 도메인 지식 0.
2. **무엇이 변하는가**: A2는 **순증 0**(선언 재사용: `eplan.finalize_writes`·`procedures`·
   `scaffold_get_tools`). 엔진에서 바뀌는 것은 ⓐ소속 판정의 단일 함수 경유 ⓑ출처 집합에 우리 문구 추가
   ⓒ발화 창 술어 한 줄.
3. **전이되는가**: 세 처방 모두 **도메인 리터럴 0**이다 — 레지스트리(프레임워크 API), A2 선언 키,
   우리 문구의 축자 토큰만 쓴다. ABox를 갈아끼우면 그대로 따라간다.

## §5 위험과 사전 등록

- **A**: 소속을 말하지 못하는 경우(판정 불가)에 문장을 빼면 안내가 얇아진다 → 얇아진 문구가
  손님-도구 안내 실패를 늘리는지 `usertoolnote` 발화 대비 성공률로 관측.
- **B**: 출처 확장이 날조를 통과시키면 즉시 철회 — 통과 조건은 **레지스트리 밖 이름 통과 0건**.
- **C**: 종결 직전 개입은 004형 *"마지막 턴 소각"* 을 재현할 수 있다(그래서 PREKB에 면제가 생겼다).
  그러므로 **차단 없이 표면화만**으로 시작하고, `T2_TERM_GRANT`(터미널 턴 보장)와의 상호작용을
  같은 런에서 마크로 확인한다([[19]] 간섭은 합성 런에서만 드러난다).
- 세 처방 모두 **구현 전에 오프라인 전수 계량**을 먼저 낸다(설계서 관례·[[09]] 유료 런 금지).
