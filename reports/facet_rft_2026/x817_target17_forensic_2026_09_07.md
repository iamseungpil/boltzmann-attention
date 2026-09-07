# x817 — 표적 17건 우리-층 결함 포렌식 (2026-09-07)

> **왜 이 문서인가** (사용자 지시 2026-09-07 축자): *"먼저 17건과 이전 궤적 비교해서 우리쪽
> 결함을 확정하고, 수리한후 돌려라. 무조건 돌린다고 수리되는건 아니지 않나?"* ·
> *"이전 설계서들 모두 확인하고, 정독하고 미진하면 새로 포렌식해서 원인 규명하고 수리하고 나서 실험하라"*
>
> 선행 정독: `x737 §9`(수리 큐 P1~P12) · `x808`(004·023·024 per-step) · `x811`(F3 설계) ·
> `x807`(A2 KB 감사) · `x810` · `x814 rev2`. **이 문서는 그 셋이 안 덮은 13건을 채운다.**

---

## 1. 표적 선정 — 기준은 **base 능력**이다

사용자 지시 축자: *"base 4/4 3/4 등이 우리 실패하는 태스크에대해서 nt=4로 우리 스택이 회복하는지"* ·
*"base 도 회복해야 하고, viewmax2도 회복해야 한다. 두가지 다를 실험하라"*.

```
표적 = { base ≥3/4  ∧  ours(viewmax2 또는 현재스택) fail }        ← 11건
     ∪ { viewmax2 pass ∧ 현재스택 fail }                          ←  6건 추가
     = 17건  (024 는 F1 수리로 4/4 확정 · 별도)
```

| 층 | 태스크 | 근거 |
|---|---|---|
| **T1** base **4/4** ∧ ours F | 004 007 015 023 047 | 모델은 4/4 로 한다. 실패는 **우리 것**이다 |
| **T2** base **3/4** ∧ ours F | 014 031 048 049 051 056 | 모델 능력 있음 |
| **T3** vm2 P ∧ 현재 F (base <3/4·미측정) | 003 036 062 070 081 095 | 우리 스택 내 회귀 |

⚠ base nt=4 는 **선정기**다 — 이 축이 없으면 «모델 한계 ↔ 우리 결함» 이 안 갈린다.
현재 base nt=4 = 61/97 측정. 잔여 우선군 10 이 돌면 T1/T2 가 더 늘어난다.

---

## 2. 원인 확정표

| 태스크 | 층 | 확정 원인 | 출처 | 수리 |
|---|---|---|---|---|
| 024 | — | `[FOLLOW-UP]` 거짓 전제(*"you found reward discrepancies"*·도구는 0건 반환) | x808 §7-1 | **F1 ✅적용·nt=4 4/4 검증** |
| 023 | T1 | `[RECOMMEND-OFFER]` 가 «배포 도구로 넘겨라» 고정 + `[ARG-EMPTY]` 10칸 | x808 §7-2 | F2 + P12 |
| 004 | T1 | `[CLAIM-PROVENANCE]`·`[ACTION-REQUIRED]` 가 gold 이관을 **두 번 금지** + 위조 VERIFIED | x808 §7-3 | F3(x811) + P11 + P9 |
| 003 | T3 | A2 `optional` 13/13 ↔ 스키마 `required` 13/13 모순 | x737 §9c | P12 |
| **047** | **T1** | ★`[RECOMMEND-OFFER]` — 사용자는 *"close my Silver Zoom Card"* 인데 우리 분류기가 *"do they want apply_for_credit_card?"* 로 단정하고 turn 76 에 `Business Platinum Rewards Card` 를 **제안하라**고 명령. gold 는 전부 호출됐고(MISSING 0) reward_basis=DB → **EXTRA/WRONGARG**. 68 호출(base ~9) | **x817 신규** | F2 |
| **048** | **T2** | ★같은 분류기. 사용자는 *"Where is my direct deposit?!"*. 144 호출·shell 51 | **x817 신규** | F2 |
| **051** | **T2** | ★같은 분류기. 사용자는 *"request a credit limit increase"*. + `[CLAIM-PROVENANCE]` 45건. MISSING=`approve_credit_limit_increase_5847` | **x817 신규** | F2 + F3 |
| **056** | **T2** | ★⛔**틀린 절차 귀속**. `[PROCEDURE] You are inside credit_limit_increase … NEXT: approve_credit_limit_increase_5847` — 그러나 gold 는 `open_bank_account_4821`·`transfer_funds_between_bank_accounts_7291`(계좌 개설·이체). 사이드카 847줄 | **x817 신규** | **F4(신규)** |
| **014** | **T2** | ★⛔`[POLICY GATE GB2_NOTICE_BEFORE_TRANSFER]` 가 turn 43 에 gold 를 **차단**. gold = `transfer_to_human_agents` **단 하나** 이고 MISSING. `[SEARCH-EXHAUST]` 도 turn 69 에 발화 | **x817 신규** | **F5(신규)** |
| **049** | **T2** | ★`GB1`+`GB2` 게이트 115건이 gold(`log_verification`·`transfer_to_human_agents`)를 금지 문맥에 놓음. MISSING=`apply_statement_credit_8472` | **x817 신규** | F5 |
| **031** | **T2** | `[CLAIM-PROVENANCE]` 가 gold `file_credit_card_transaction_dispute_4829` 를 «네 것이다» 문맥에 13건. MISSING 0 → WRONGARG/EXTRA | **x817 신규** | F3 |
| 007 015 036 062 070 081 095 | | **미확정** — 2차 포렌식 필요 | | |

---

## 3. ★공통 근인 — **일반 원리** (x808 §7-4 의 확장 · 특허 축)

> **우리 층은 «지금 X 를 하라 / Y 는 하지 마라» 를 단정한다. 그 전제는 우리 분류기가 만든다.
> 분류기가 틀리면 그 위에 쌓인 모든 지시가 거짓이 되고, 모델은 [[63]] 대로 금지·명령형에 닫힌다.**

전제 생산자 3종이 실측으로 드러났다:

| 전제 생산자 | 무엇을 단정하나 | 오발 실물 |
|---|---|---|
| `apply_for_credit_card` 의도 분류기 (turn 0) | *"사용자가 카드 신청을 원한다"* | **047**(해지 요청) · **048**(급여 문의) · **051**(한도 증액) · **056**(계좌 업무) |
| `[PROCEDURE]` 절차 귀속 | *"너는 credit_limit_increase 안에 있다"* | **056**(gold 는 계좌 개설·이체) |
| `[FOLLOW-UP]` 결과 단정 | *"너는 불일치를 찾았다"* | **024**(도구가 0건 반환) — F1 로 수리됨 |

그리고 **게이트가 gold 를 막는** 별개 축:

| 게이트 | 막은 gold | 실물 |
|---|---|---|
| `GB2_NOTICE_BEFORE_TRANSFER` | `transfer_to_human_agents` | **014**(gold 가 그것 하나뿐인데 MISSING) · 049 |
| `GB1_VERIFY_BEFORE_ACCOUNT_ACCESS` | `log_verification` 외 | 031 047 049 056 |

⚠ base 는 이 문면을 **하나도 받지 않고** 9~10 호출로 통과한다. 우리 호출 수는 43~144.

[[66]] 정합: **의도 분류는 엔진이 하면 안 된다.** 047/048/051/056 은 그 금지선을 실측으로 확인한 것이다.

---

## 4. 수리 목록 (적용 대기)

| # | 위치 | 변경 | [[70]] 파는 것 | 상태 |
|---|---|---|---|---|
| **F1** | `t2_gate_patch.py:14866` | follow_up 장전을 «호출됨»→«비지 않은 결과» | 거의 없음 | ✅**적용·검증 4/4** |
| **F2** | `t2_resolve.py:1023` `RECOMMEND_OFFER_FB` | ①`{offer}` 를 «에이전트가 부를 수 있으면 그 도구 자체»로 ②★**의도 분류기가 «아니오/불확실»이면 발화 자체를 막는다** | 배포가 정답인 태스크 | 미적용 |
| **F3** | `[CLAIM-PROVENANCE]`+`[ACTION-REQUIRED]` | *"or transfer"* 무조건절 한정 (설계 = **x811**) | D12 가 겨눈 떠넘기기 | 미적용 |
| **F4** | `[PROCEDURE]` 절차 귀속 | 귀속 근거를 **호출된 도구**로 한정 — 추정 금지 | 미측정 | **신규·설계 필요** |
| **F5** | `GB1`/`GB2` 게이트 | gold 를 막는 게 아니라 **선행 조건을 만들어 주고** 통과시켜야. 014 는 gold 가 그 하나뿐 | 정책 준수 | **신규·설계 필요** |
| P9 P11 P12 P1 P2 P6 | x737 §9b | 전부 «근거 확정 · 적용 대기». P9·P12·P1·P2 는 **[[70]] 파는 것 없음** | | 미적용 |

---

## 5. 다음 수 (순서 고정)

1. **2차 포렌식** — 미확정 7건(007 015 036 062 070 081 095)
2. **수리** — P9·P12·P1·P2·P6(절충 없음) → F2·F3(x811) → F4·F5(설계 후)
3. **[[70]] 부호표** — F2·F3·F4·F5 는 파는 쪽이 실재. 부호표 없이 출시 금지
4. **스모크 게이트** ([[73]]) — `task_050` 포함(x737 §9b 지정)
5. **17건 × nt=4 = 68 sim** — 판정선 C548 `Δ≥4/40` 을 크게 넘는다

## 6. 반증 조건 ([[77]])

- §3 의 «분류기 오발» 주장은 **turn 0 사이드카 문면**에 있다. 그 문면이 없거나 사용자 발화와
  실제로 부합하면 반증된다. 확인 경로 = `fb_<run>.jsonl` 의 `turn=0` 항목 축자.
- §2 의 «base 4/4» 는 `iso_tau3/.../bank_x806_base_nt4_task_*/results.json` 4 시행 전부 1.0.
- F1 의 반증 조건은 x808 §7-5 가 지정한 대로였고 **`f1chk_024` = 4/4 로 통과**했다.

---

# §7 — [[70]] 부호표: **짝으로 나빠질 태스크를 런 전에 명단화** (2026-09-07)

> 사용자 지시 축자: *"P11 P12 등을 보류한 이유가 트레이드오프로 영향권의 다른 태스크가 나빠질거라서
> 였던거 같다. 하나 좋아지면 다른게 나빠지지 않게 짝으로 나빠질 걸로 보이는 부분을 같이 실험해서
> 확인하라"* — 맞다. x737 §9b 는 **P6 에만** 부호표를 붙였고 나머지는 «미측정»이었다.

## 7-1. 방법 — 사이드카 `simtag` 가 seed 를 담는다

```
사이드카 레코드  {"kind":"tool-deny","simtag":"task_003#s361454","turn":"5",
                 "channel":"unified_regen","call_name":"check_card_application_fit","text":"Error: [ARG-EMPTY] …"}
results.json     {"task_id":"task_003","seed":361454,"trial":2,"reward_info":{...}}
⇒ (task_id, seed) 로 **sim 단위 정확 짝맞춤**. 런 단위 귀속은 부호표로 못 쓴다([[25]]).
```

## 7-2. P12 부호표 (회수분 전수 · GPU 0)

| 칸 | sim | 태스크 |
|---|---|---|
| 통과 · 이 반려를 맞음 = **파는 쪽 후보** | **45** | **001 002 003 006 008 023 024 025 044 047 058 063 064 067 068** (15) |
| 실패 · 이 반려를 맞음 = **사는 쪽** | **39** | 003 007 023 024 047 048 057 059 063 064 068 069 071 (13) |
| 반려 없이 통과 (이 레버 무관) | 185 | — |
| 반려 없이 실패 | 349 | — |

★**023 · 047 · 063 · 064 · 068 은 양쪽에 다 있다** — 같은 태스크가 sim 에 따라 사기도 팔기도 한다.
이것이 사용자가 지목한 절충의 실물이다.

## 7-3. ⛔ 이 부호표가 **답할 수 없는 것** (정직 · [[77]])

*"통과한 45 sim 은 반려 **덕분에** 채운 것인가, 원래 채우던 것인가"* — **오프라인으로는 판정 불가**다.
이유: `_ap_regen` 이 반려당한 호출을 **교체**하므로 궤적에는 «수락된 호출»만 남는다. 그래서
«첫 호출이 빈칸이었다»는 **원리상 관측될 수 없다**(실측: 45/45 전부 «첫 호출부터 채움»으로 나오는데,
그건 사실이 아니라 **관측 불가의 그림자**다).

⇒ 이 칸은 **런으로만** 닫힌다. 그래서 부호표의 역할은 «파는 쪽 명단을 검증 런에 넣는 것»이다.

## 7-4. ⇒ 검증 런 태스크 집합 (확정)

```
표적 17        004 007 015 023 047 014 031 048 049 051 056 003 036 062 070 081 095
P12 파는 쪽 15  001 002 003 006 008 023 024 025 044 047 058 063 064 067 068
────────────────────────────────────────────────────────────────────────────
합집합 29      001 002 003 004 006 007 008 014 015 023 024 025 031 036 044
               047 048 049 051 056 058 062 063 064 067 068 070 081 095
29 × nt=4 = 116 sim
```
판정: **표적은 오르고 파는 쪽은 안 떨어져야** 통과. 태스크별 부호표를 런 후 다시 낸다([[70]] 판정의무 3종).

## 7-5. 나머지 수리의 부호표 상태

| 수리 | 파는 쪽 | 근거 |
|---|---|---|
| **P2** | **없음(증명됨)** | 미선언 도메인 = 빈 집합 ⇒ 그 가지 미발화. airline/retail 은 `verify_identity` 자체가 없다. 단위테스트 6/6 이 항등성을 강제 |
| **P9** | **없음** | 우리 보호를 없애도 **서버가 같은 검사를 더 정확히** 한다(`tools.py:533-534`). 기존 `test_p9_signature.py` 부정통제 통과 |
| **P1** | ⚠**미측정** | 표지 4종이 더해지면 «성공으로 세던 것»이 실패가 된다 ⇒ 게이트가 더 오래 열려 있는다. **부호표 필요** |
| **P12** | §7-2 (파는 쪽 15 태스크) | 위 |
| F2 F3 F4 F5 | ⚠**미측정** | x808 §7-5 가 *"F2·F3 는 파는 쪽이 실재 · 부호표 없이 출시 금지"* 라 못박음 |

---

# §8 — F2 부호표 · 그리고 **§3 의 자기정정** (2026-09-07)

## 8-1. ⛔ §3 의 「분류기 오발」 주장은 `047`·`048` 에서 **틀렸다**

§3 에서 나는 *"`apply_for_credit_card` 의도 분류기가 무관한 요청에 발화한다"* 며
`047`(*"close my Silver Zoom Card"*)·`048`(*"Where is my direct deposit?!"*)을 증거로 들었다.
gold 를 직접 읽으니 **둘 다 gold 에 `apply_for_credit_card` 가 있다**:

| | gold | 대본(축자 요지) |
|---|---|---|
| `047` | `apply_for_credit_card` 포함 4종 | *"경쟁사 Global Business Travel Plus 카드를 찾았다"* → **리텐션**(해지 방어 → 대체 카드 신청) |
| `048` | `apply_for_credit_card` 포함 4종 | *"카드 4장을 해지하고 싶다"* → 같은 리텐션 구조 |
| `008` | `transfer_to_human_agents` 하나 | 축자 *"you are NOT a business owner and do NOT qualify for any business credit cards"* |

나는 **사용자 발화의 첫 문장만 보고** 단정했고, 대본 전체는 리텐션 시나리오였다.
이것은 §3 이 비판한 죄 — **부정확한 전제 위의 단정** — 를 내가 저지른 것이다.
⇒ §3 의 「전제 생산자 3종」 표에서 **`apply_for_credit_card` 분류기 행의 실물은 `008` 하나**로 줄인다.
   `[PROCEDURE]` 오귀속(`056`)과 `[FOLLOW-UP]` 거짓 전제(`024`)는 그대로 유효하다.

## 8-2. F2 부호표 (귀속 = `(run, task, seed)`)

> ⚠1차 산출은 `(task, seed)` 로만 키를 잡아 **런이 다르면 seed 가 겹치는 것**을 같은 sim 으로 셌다
> (발화 26 → 행 634). 런을 넣어 다시 쟀다([[25]]).

```
사이드카 발화 (run,task,seed) 70 · 결과와 짝지어진 sim 62 · 고유 태스크 20
분류기가 지목한 action = apply_for_credit_card 62/62

갈래                          통과  실패  태스크
ⓐ gold 이 **직접 호출**을 요구   24    36    19    001 002 003 006 007 023 024 025 044
                                                  047 048 058 059 063 064 066 067 068 069
ⓑ gold 이 **배포(give)**를 요구   0     0     0    ★
ⓒ gold 에 아예 없음               2     0     1    008 (그 sim 은 통과)
```

## 8-3. 판정

**★F2a — 절충 없이 낸다.** 문면이 `{offer}`(= 배포 도구)를 **고정**해 강제하는데, 이 레버가 발화한
**20 태스크 전부에서 gold 는 직접 호출을 요구**한다. x808 §7-5 가 경고한 *"배포가 정답인 태스크를
되살릴 수 있다"* 는 위험이 **회수분 전수에서 0건**이다.

**⛔F2b(분류기 억제) — 하지 않는다.** 근거가 `008` 하나이고 **그 sim 은 통과**했다.
사는 쪽 0 · 파는 쪽 2 sim ⇒ [[70]] 상 낼 수 없다.

## 8-4. 구현 (`t2_resolve.py`)

```
+ RECOMMEND_OFFER_DIRECT_FB   "'{action}' is a tool you already have — call it directly with
                               {operand}='{correct}'. Do not hand it to the user and do not
                               search the knowledge base for it."
+ _agent_holds(agent, name)   `agent.tools` 만 본다 (닫힌 술어·도메인 리터럴 0)
```
⚠권위는 `agent.tools` 뿐이다 — `registry_names` 를 쓰면 아직 발견 안 된 discoverable 까지 합쳐
*"직접 불러라"* 가 **또 다른 거짓말**이 된다(`t2_resolve.py:186` 선례와 같은 이유).
⚠직접 호출을 이미 했으면 침묵한다([[64]] 처방이 소음이 되지 않게).
⚠미보유 도구는 **구판 문면 그대로** — 거동 변화 0.

`test_f2_offer_direct.py` **6/6 PASS**(도메인 리터럴 0 검정 포함).

## 8-5. ⚠ 출시 시점

`rep1`(8143) 이 **F2 없이** 이미 돌고 있다. 지금 넣으면 한 런 안에서 스택이 갈린다([[54]]).
⇒ **F2 는 2차 파동**이다. rep1 이 끝난 뒤 F3·F4·F5 와 함께 낸다.

---

# §9 — F3 부호표 「이관 금지 무조건절」 (2026-09-07)

대상 문면 둘(x808 §7-3 축자):
```
[CLAIM-PROVENANCE] "You are about to end your involvement (resign or transfer) with these
                    promises unfulfilled — that abandons the customer's request. Do … NOW"
[ACTION-REQUIRED]  "do NOT just explain how to do it, advise self-service, **or transfer**"
```

## 9-1. 1차 — 문면을 받은 sim 전수

```
발화 조합 865 · 짝지어진 sim 808 · 고유 태스크 89
채널: CLAIM-PROVENANCE 1,156 · other 250 · ACTION-REQUIRED 121

ⓐ gold 이 이관을 요구        통과  30 / 실패  42   · 12 태스크
ⓑ gold 은 이관을 요구 안 함   통과 116 / 실패 620   · 77 태스크
```
⚠이 표는 **인과가 아니다** — 「문면을 받았다」와 「문면이 행동을 바꿨다」는 다르다.

## 9-2. 2차 — **이관 호출이 실제로 막힌** sim 만 (사이드카 `tool-deny`)

우리 반려는 재생성이 원 메시지를 교체하므로 궤적에 안 남는다([[30]]) — 사이드카 `kind=tool-deny`
+ `call_name` 이 유일한 증거다.

```
금지절을 받은 808 sim 중 **이관 호출이 실제로 막힌 것 = 21 (2.6%)**

★사는 쪽  gold 이관 요구 · 막힘 · 실패    5 sim ·  4 태스크   049 081 088 092
★파는 쪽  gold 이관 불요 · 막힘 · 통과    1 sim ·  1 태스크   023
  중립    gold 이관 요구 · 막힘 · 통과    5 sim ·  5 태스크   005 008 012 033 035
  중립    gold 이관 불요 · 막힘 · 실패   10 sim · 10 태스크
```
⇒ 직접 증거로는 **사는 쪽 5 : 파는 쪽 1**.

## 9-3. ⛔ 이 부호표가 **놓치는 것** — 그리고 그게 가장 큰 몫이다

`task_004` 는 「막힘」 목록에 **없다**. 우리 문면이 너무 잘 들어서 모델이 **이관을 시도조차 안 했기
때문**이다(x808 §7-3: `transfer_to_human_agents` **0회**). 시도가 없으면 deny 기록도 없다.

⇒ **선제적 억제는 오프라인으로 관측 불가**다. [[63]] 이 말한 그대로 — 모델은 더하기 지시엔 둔하고
**금지엔 닫힌다**. 그래서 이 레버의 진짜 비용은 2.6% 가 아니라 그보다 크고, **부호는 사는 쪽으로
더 기운다**(놓친 몫이 전부 「막지 말았어야 할 이관」 쪽이므로).

## 9-4. ⛔ x811 §5 의 부호 반대편 지정을 정정한다

x811 §5 는 부호 반대편을 *"D12 가 겨눈 태스크(**033 계열** · 이관 떠넘기기)"* 라고 적었다. gold 실측:

```
task_033  이관도구 = transfer_to_human_agents   basis=['ACTION']   ← gold 이 **이관을 요구**한다
task_004  이관도구 = transfer_to_human_agents   basis=['ACTION']   ← 같은 모양
task_023  이관도구 = 없음                        basis=['DB']       ← 유일한 파는 쪽 실물
```
**`033` 은 사는 쪽이지 반대편이 아니다.** 반대편의 실물은 **`023` 하나**이고, 그것도 1 sim 이다.

## 9-5. 판정

| | |
|---|---|
| 직접 증거 | 사는 5 : 파는 1 |
| 관측 불가분 | 선제적 억제(시도 없음) — 전부 사는 쪽으로 기움 |
| 결론 | **수리 쪽이 유리하나 오프라인으로는 확정 불가.** x811 §5 가 이미 *"이 절충은 A/B 로 판정한다"* 라고 못박았다 |

⇒ **F3 는 2차 파동에서 A/B 로 낸다.** 대조 표적:
- 회복 표적 `004 033 049 081 088 092` (gold 이 이관을 요구하는데 막힌 것)
- 부호 반대편 `023` (유일한 파는 쪽 — 반드시 같은 런에 넣는다)

---

# §10 — F4 부호표 「`[PROCEDURE]` 절차 오귀속」 → **기각** (2026-09-07)

## 10-1. 격리 프로브가 사이드카 추정을 정정했다 ([[78]])

`_executed_tool_names` → `active_procedures` 를 **엔진 함수 그대로** 불러 회수분 전수에 먹였다
(사본 0·[[67]] · GPU 0).

```
궤적 있는 sim 4,575 · 활성 절차 판정 815

진입 도구별 (옳음 = gold 이 그 절차의 도구를 요구)
  옳음   cash_back_dispute             ← get_reward_discrepancies          502
  옳음   credit_limit_increase         ← check_cli_eligibility             135
  옳음   credit_card_closure_retention ← check_card_closure_eligibility    115
  오귀속  credit_card_closure_retention ← check_card_closure_eligibility     37
  오귀속  cash_back_dispute             ← get_reward_discrepancies           18
  오귀속  credit_limit_increase         ← check_cli_eligibility               1
  ── **행동 도구**를 통한 진입은 전부 합쳐 7건
```

⇒ **오귀속률 = 56/815 = 6.9%.** 사이드카 기반 1차 추정(44.5%)은 **사이드카가 있는 런만** 봐서
편향돼 있었다. §2 에서 `056`(`credit_limit_increase` 오귀속)을 주요 결함으로 든 것도 과대평가다 —
**815 중 1건**이고, 오귀속의 본체는 closure 절차의 37건이다.

## 10-2. 처방 후보 전부 실격

| 후보 (전부 런타임 관측 가능) | 오귀속 감도 | 옳음 오살 | |
|---|---|---|---|
| 진입은 **행동 도구로만** | — | — | ⛔레버를 끔: 815 → 7 ([[60]] 위반) |
| `[GROUNDING WARNING]` | 41% (15/37) | 8% (9/115) | 감도 낮음 |
| `CLOSURE_BLOCKED` | 38% | 34% | 무차별 |
| 절차 도구가 진입 하나뿐 | 100% | 88% | 무차별 |
| 진입 호출 1회 | 51% | 50% | 무차별 |

## 10-3. ⛔ 일반화 검정 — 층화하면 사라지는 오즈비

```
                 오귀속  옳음        절차별
GW 있음            15     9          closure  62%(15/24) ↔ 17%(22/130)   오즈비 8.2
GW 없음            41   750          cash_back_dispute      GW 발생 **0건** — 검정 불가
전체 오즈비 30.5                      credit_limit_increase  GW 발생 **0건** — 검정 불가
```
`[GROUNDING WARNING]` 은 **closure 절차에서만 발생**한다. 전체 오즈비 30.5 는 Simpson 형 **가짜**다.
하마터면 일반 원리로 보고할 뻔했다.

## 10-4. 판정 — 기각. 다만 부산물 하나

*"이 고객이 해지하려 하는가"* 는 **열린 술어**라([[22]]/[[66]]) 닫힌 술어로 근사되지 않는다. ⇒ **F4 불가.**

★부산물: `[GROUNDING WARNING]` 은 **우리 자신의 근거검증층**이 낸 문면이다 —
*"input value(s) could not be verified … and **were dropped**"*. 그런데 `active_procedures` 는 그
호출을 **「성공한 실행」으로 세어** 절차 상태를 켠다. **우리가 스스로 «근거 없다»고 표시한 호출 위에
상태를 쌓는다.** 순효과 +6(오귀속 −15 · 옳음 −9)으로 작아 단독 출시는 못 하나 **원리적으로 옳고**,
P1 의 `verdict_markers` 와 같은 계열(우리 층이 무엇을 «수행됨»으로 세는가)이다. 2차 파동에 묶는다.

---

# §11 — F5 부호표 「`GB2` 가 gold 이관을 차단」 → **기각** (2026-09-07)

충족 술어는 정본 `gate_interpreter.notice_sent_in` 을 그대로 호출(정규화 후 앞 48자 부분문자열 —
032 의 `", Sofia"` 개인화 사고 뒤 완화된 판정).

```
GB2 가 막은 sim 483
③ gold요구 · 이관 성공             45 통과 /  32 실패  13 태스크   ← 게이트 정상 작동
① gold요구 · 고지 보냄 · 이관 0      0 /  21           3 태스크 (004 033 088)
② gold요구 · 고지 안 보냄 · 이관 0    0 /   3           3 태스크
④a gold **불요** · 고지 보냄        15 통과 / 293 실패  38 태스크   ← ★파는 쪽
④b gold 불요 · 고지 안 보냄          9 /  65           21 태스크
```

## 11-1. 충족불가가 아니다 — **금지 문면의 억제**다

칸 ① 의 차단 횟수 분포 `{1: 19, 2: 2}` — **19건이 단 한 번 막히고 끝**났다. 게이트가 안 열린 게
아니라 모델이 **다시 시도하지 않았다**. 게이트 문면 축자:
```
Recovery: (1) do NOT retry this tool now; (2) the transfer is blocked only because …
```
`now` 가 붙어도 모델은 **종결 금지로 읽는다**([[63]]). §3 이 이름 붙인 형태 그대로다.

## 11-2. 그런데 처방의 부호가 **21 : 308** 으로 불리하다

*"고지가 기록에 올랐으니 이제 이관하라"* 를 넣으면 사는 쪽 21 sim(3 태스크) ↔ 파는 쪽
**308 sim(38 태스크·그중 15는 현재 통과)**. gold 이 요구하지 않는 이관을 유발한다.

⇒ **GB2 + «재시도 금지» 는 실제로 정책을 지키고 있다.** 게이트 자신의 문면이
*"do this only if you absolutely have to"* 라고 적은 그 일을 293번 해내고 있다.

## 11-3. ⛔ §3 의 F5 분류를 정정한다

§3 은 F5 를 *"게이트가 gold 를 직접 차단"* 으로 적고 `014`·`049` 를 근거로 들었다.
**21 sim(3 태스크)에서만 맞고 308 sim(38 태스크)에서는 게이트가 옳게 막고 있다.**
표본 2건으로 전체를 말한 것이었다.

## 11-4. 남는 여지 — 중립적 최소 수정 (A/B 필요)

`do NOT retry this tool now` 가 만드는 **거짓 종결감**만 없애고 **밀지는 않는다**:
> *"Once that notice is on the record this gate no longer blocks the transfer. Whether to transfer
> at all remains your judgment under the policy."*

금지를 풀되 유도하지 않으므로 파는 쪽이 훨씬 작다. **A/B 없이는 확정 불가.**

---

# §12 — 종합 (2026-09-07)

| 수리 | 판정 | 근거 |
|---|---|---|
| **F1 P12 P2 P9 P1 P6 P11** + E-PLAN 회귀 정정 | ✅ **출시** (rep1 검증 중) | 부호표 유리 또는 절충 0 · 전부 단위테스트 |
| **F2** | ✅ **2차 파동 출시** | 파는 쪽 실측 **0** (20 태스크 전부 gold 이 직접 호출 요구) |
| **F3** | ⚠ **A/B 필요** | 직접증거 5:1 유리 · 선제 억제는 오프라인 관측 불가 |
| **F4** | ⛔ **기각** | 닫힌 술어 없음 · 오귀속 6.9% |
| **F5** | ⛔ **기각** | 21:308 불리 · 게이트는 옳게 작동 중 |

★**부호표가 5건 중 2건을 기각했다.** 이 다섯을 다 «수리」라고 냈으면 둘은 순손실이었다 —
[[70]] 규율의 값어치가 여기서 실측으로 확인된다.

★**자기정정 3건**(§8-1 `047`/`048` 분류기 · §9-4 x811 부호 반대편 · §11-3 F5 귀속).
셋 다 **표본 소수로 전체를 말한** 같은 형태였고, 전수 측정이 정정했다.

---

# §13 — 정정 원장 · 그리고 **정정들이 전부 같은 형태였다** (2026-09-07)

> 이 문서를 쓰는 동안 **다섯 번** 스스로를 정정했다. 다섯이 모두 같은 죄다 —
> **한 층만 보고 전체를 말했다.** 그것은 x808 §7-4 가 **우리 층**에 대해 이름 붙인 결함
> (*"부정확한 전제 위의 단정"*)과 같은 형태이고, 이번엔 **분석자**가 저질렀다.

| # | 내가 말한 것 | 실제 | 무엇을 안 봤나 | 정정 위치 |
|---|---|---|---|---|
| **①** | *"`apply_for_credit_card` 분류기가 무관한 요청에 오발한다 (047·048)"* | 두 태스크 **gold 이 그 도구를 요구**한다(리텐션 시나리오) | 사용자 발화 첫 문장만 보고 **대본 전체·gold** 를 안 봄 | §8-1 |
| **②** | x811 §5 인용 *"부호 반대편 = 033 계열"* | `033` gold 은 **이관을 요구**한다 = 사는 쪽 | 정본 문서를 **검산 없이 인용** | §9-4 |
| **③** | *"F5 = 게이트가 gold 를 차단"* | 21 sim 에서만 맞고 **308 sim 에서는 게이트가 옳다** | `014`·`049` **2건**으로 38 태스크를 말함 | §11-3 |
| **④** | *"`check_card_application_fit` 인자 13 중 9개를 op 가 무시한다"* | **엔진 `catalog_filter`(t2_compute:351)가 구현**한다 | **A2 JSON 만** grep 하고 엔진을 안 봄 | 아래 |
| **⑤** | *"eligible 평균 6.7 · P12 가 후보를 두 배 넓힌다"* | eligible **4.15** · P12 후 **4.27**(변화 없음) | `'card':` 를 **출력 전체**에서 세어 `excluded`·`unverified` 버킷까지 합침 | 아래 |

## 13-1. ④⑤ 정정 후의 사실 (이것이 정본이다)

```
check_card_application_fit  파싱된 산출 565건
  eligible 4.15 · excluded 2.36 · unverified 0.20
  gold 이 eligible 에 포함 : 504/565 = 89%

채운 인자 ↔ eligible :  2개→5.03 · 5개→3.27 · 7개→2.32 · 9개→2.38
  ⇒ 필터는 **정상 작동**한다. 인자를 채울수록 좁아진다.

P12 전후:  과거 전수 eligible 4.15  ↔  rep1(P12 후) 4.27
  ⇒ **P12 는 후보를 넓히지 않았다. 파는 쪽 없음.**
```

남는 결손은 **선택**이다: `eligible ≥ 3` 인 호출이 **459/565 = 81%** 이고, 그 안에서
모델 정확도 **55%**(626 직접 호출 중 344), LLM 검증기 정확도 **62%**(89건 중 55).

## 13-2. ⇒ **F7′** (방향 정정: 필터가 아니라 **표면화**)

> eligible 이 2개 이상 남으면 **그 후보들을 가르는 표의 칸**(`annual_fee`·`cashback_scope`·
> `min_score`·`limit_max` 등)을 함께 낸다. 지금은 후보 목록만 주고 «무엇이 다른가»는 말하지 않는다.

- **[[70]] 파는 것 = 구성적 0** — 표면화(R4)는 x814 rev2 §2c 상 판정 면제. 막지도 밀지도 않는다
- **[[23]] 정합** — 출처는 A2 `table` 뿐
- 영향권 = `eligible ≥ 2` 인 호출 **479/565 (85%)**

⛔ **F7(«필터 조건 추가»)은 철회한다** — 조건은 이미 엔진에 있다(④).
⛔ **F2c(«검증기를 직접 호출에도 적용»)도 철회한다** — 검증기 62% vs 모델 55% = **+7%p**,
   순 기대이득이 얇아 [[70]] 판정선에 못 미친다(§F2c 측정).

## 13-3. 방법론 — 이 다섯이 논문 논거다

다섯 정정의 공통 형태:

```
한 층만 본다  →  그 층의 사실을 전체의 사실로 말한다  →  다른 층이 반증한다
  ① 발화 1문장 ↔ 대본·gold        ② 문서 인용 ↔ 원자료
  ③ 표본 2건 ↔ 전수 38            ④ 선언(A2) ↔ 구현(엔진)
  ⑤ 출력 전체 ↔ 버킷 구조
```

★**우리 층이 모델에게 저지르는 죄와 같다** — x808 §7-4: *"우리 층이 «지금 X 를 하라»를
부정확한 전제 위에서 단정하고, 모델이 순응했다."* 분석자도 같은 죄를 짓는다.
⇒ **처방도 같다: 전수·다층 검증을 규율로 강제한다.** 부호표 의무([[70]])와 격리 프로브([[78]])가
그 규율이고, 이 문서가 그 값어치의 실측이다 — **부호표가 5건 중 2건을 기각했고(§12),
다층 검증이 내 주장 5건을 기각했다.**

---

# §14 — `015` 포렌식: **우리가 gold 의 `give` 를 «엉뚱한 도구 이름»으로 막았다** (2026-09-07)

`015` 는 rep1 에서 잔여 격차가 가장 크다(**base 4/4 ↔ rep1 1/4**).

```
gold   give_discoverable_user_tool(discoverable_tool_name="get_referral_link")
       call_discoverable_user_tool(… card_name="Platinum Rewards Card")

통과 1 (373753)  give 성공 · [GIVE-EXEC] 받음 · MISSING = get_referral_link(고객 실행분)만
실패 3           MISSING = give · call_discoverable_user_tool · get_referral_link  (전부)
                 ★셋 다 `give_discoverable_user_tool` **반려** 1~2회
★실패에만 있는 마커: WORK-INCOMPLETE×2   ★통과에만: GIVE-EXEC×1
```

## 14-1. 반려 문면 축자 — 세 가지가 동시에 틀렸다

```
Error: [ACTION] 'submit_referral' is run by the CUSTOMER, not by you. There is no
agent-side procedure to look up for running it, so do not search for one and do not
transfer the conversation in order to get submit_referral run.
```

| | 무엇이 틀렸나 |
|---|---|
| ① | **엉뚱한 도구를 지목**한다. 모델은 `get_referral_link` 를 주려 했고 gold 도 그것이다 |
| ② | **gold 액션(`give`) 자체를 반려**한다 |
| ③ | **자기모순** — *"고객이 실행한다"* 면서 **고객에게 주는 행위**를 막는다. 받아야 실행한다 |

연쇄: `give 반려 → 모델 포기 → transfer 시도 → [GB2] 반려 → [WORK-INCOMPLETE]("7 asked, 0 acted") → 아무것도 못 함`

## 14-2. 기전 — LLM 이 고른 표적 위에 금지를 쌓는다 ([[66]] 위반)

```python
_upending = _uacts - _effall                          # 아직 안 돌린 user-side 도구
_utgt     = _rz.formalize_intent_tool(...)            # ★LLM 서브콜이 그중 하나를 고른다
if _utgt in _upending:  → "'{_utgt}' is run by the CUSTOMER" 로 반려
```
`015` 에서 분류기가 `get_referral_link` 대신 **`submit_referral`** 을 골랐다.
지목된 도구 분포(전수): `submit_referral 836 · apply_for_credit_card 591 · submit_transaction 382`.

★**선행 프로브가 이미 있다** — `x117_tool_channel_truth.py`(2026-08-06 · task_102) 축자:
*"같은 sim 안에서 우리 두 문구가 `submit_referral` 을 두고 **반대로** 말했다 … `_exec_side` 는
`self.tools` 를, give-가드는 `env.tools` 를 본다."* `015` 는 그 결함군의 **새 발현**이다.

## 14-3. F8 부호표 — **1 : 4 불리 · 결정적 칸은 관측 불가**

```
[ACTION] 발화 sim 572 (49 태스크) · 그중 **give 를 막은** sim 5 (1%)
  ⓐ gold 이 give 요구 · give 막힘   1 sim (통과 0)  task_015
  ⓑ gold 은 give 불요 · give 막힘   4 sim (통과 4)  058 063 064 098
```

결정적 칸 = *"문면이 지목한 도구 == 모델이 주려던 도구인가"* — 같으면 막을 이유가 없고(주는 것이
곧 고객이 실행하게 하는 길), 다르면 오지목이다. **5건 중 4건에서 그 칸을 못 채웠다**:
반려당한 give 호출을 재생성이 **교체**해 궤적에 인자가 남지 않는다([[30]]).

⇒ **F8 미출시.** 부호가 불리하고 인과가 미확정이다.

## 14-4. ⇒ 계측 제안 (수리가 아니라 **볼 수 있게** 하는 것)

사이드카 `tool-deny` 레코드에 **`call_args`** 를 남긴다(`call_name` 은 이미 있다).
그러면 §14-3 의 결정적 칸이 관측 가능해지고, 같은 한계에 걸린 **P12 §7-3 · F3 §9-3 · F8 §14-3**
세 부호표가 한꺼번에 닫힌다. 계기 추가라 [[70]] 파는 것 0.

## 14-5. ⇒ **선언 결손은 `015` 하나가 아니다** — 그리고 F8′ 부호표

환경 `KnowledgeUserTools` = **11개**. A2 `action_tools` 가 그중 **4개**만 담았다.

```
★미선언인데 gold 이 요구하는 도구
   get_card_last_4_digits           7 태스크  031 037 038 039 040 041 053
   submit_cash_back_dispute_0589   10 태스크  017 018 019 020 021 022 026 027 028 029
   request_human_agent_transfer     3 태스크  034 049 081
   get_referral_link                1 태스크  015
   ────────────────────────────────────────────────────────────
   합집합 **24 태스크**
```

★`get_card_last_4_digits` 는 **A2 자신이 `value_acquisition.acquire_tool` 로 선언한 도구**인데
`action_tools` 에는 없다 — **같은 A2 안의 모순**이고 [[24]] 양방향 검사가 잡았어야 할 종류다.

### F8′ 부호표 — 파는 쪽의 주 축이 **실측 0**

```
`_upending` 이 비지 않은 sim :  현재(4개) 4575/4575 (100%)  →  넓힘(9개) 4575/4575 (100%)
```
**창 개방 빈도 변화 0.** 현재도 4개 중 하나는 거의 항상 미실행이라(apply 13% · submit_referral 11% ·
change_email 0.8% · submit_transaction 0.3%) 창은 이미 항상 열려 있다.
⇒ 넓혀도 «[ACTION] 이 더 자주 나간다»는 우려는 **실현되지 않는다**.

남는 변화는 **서브의 후보 집합**뿐이다(4 → 9):

| | |
|---|---|
| 사는 쪽 | **24 태스크** — 서브가 정답 도구를 **고를 수 있게** 된다. 지금은 정답이 후보에 없으면 **반드시** 틀린다(015 실물) |
| 파는 쪽 | 후보가 늘어 서브가 더 틀릴 여지. **오프라인 측정 불가**(서브 재실행 필요) |

⇒ **F8′ 는 낼 만하다.** 단 A/B 로 판정하고, 부호 반대편으로 «미선언 도구를 gold 이 요구하지
않는 태스크»를 같은 런에 넣는다.

### 처방의 형태 — 손목록을 버린다

> `_uacts` 를 A2 `action_tools` 손목록이 아니라 **환경 `user_tools` 레지스트리**에서 유도하고,
> A2 는 «행동이 아닌 것»(`query_database` · `list_discoverable_user_tools`)만 **제외 목록**으로 선언한다.

- **[[05]]** 엔진은 이름을 모르고 환경에 묻는다(`registry_names`·`_decl_tool_collections` 선례)
- **[[63]]** 포함이 아니라 **빼기** — 빠뜨리면 «과잉 포함»이라 **관측된다**(지금은 «누락»이라 안 보였다)
- 이 결함 부류가 **구조적으로 불가능**해진다

## 14-6. ⛔ F8′ 는 **선행 판정이 반증**한다 — 그리고 살아남은 처방 F8‴

`_uacts` 확대는 2026-08-23 `R8-pending-disc-dead` 가 이미 기각했다. 근거 넷 중 둘이 결정적이다:

> **⑵ 종료 술어가 없다** — *"`_effall` 은 `state.messages` 의 호출만 본다. **손님이 실행한 도구는
> 거기 없다** … discoverable 을 더하면 손님이 이미 실행한 뒤에도 **영원히 pending** 이고 넛지가
> 끝나지 않는다"*
> **⑷ 병목은 이미 닿아 있다** — *"막혀 있는 행동은 `give_discoverable_user_tool` 이고 그 이름은
> 정적 `action_tools` **안에 있다** — `formalized_target=give…` 이 **291회**"*

★내 §14-5 부호표는 **⑵의 축(종료 술어)을 안 쟀다** — 「창 개방 빈도」만 보고 「넛지가 끝나는가」를
빠뜨렸다. §13 의 다섯과 같은 형태다(**한 층만 보고 전체를 말했다**). [[74]] 가 이것을 막았다.

### 살아남은 처방 — 사용자 제안 (2026-09-07): **호출 형식과 도구를 쌍으로**

> ⑶ 이 이미 지적했다: *"`user_action_feedback` 은 «tell the customer to run {tool} themselves»
> 인데 도메인 정책 축자는 «you must use the `give_discoverable_user_tool(discoverable_tool_name)`
> function» 이다 … 우리 층이 **무엇을 하면 풀리는지를 틀리게** 말하게 된다"*

호출 형식은 **환경 상태의 닫힌 함수**다(해석 0):
```
채널 = _exec_side(tool)                       ← t2_role.executor_of (환경 유도)
  assistant            →  tool(...)  직접 호출
  user · 미건네짐       →  give(discoverable_tool_name=tool)      ← 정책 축자
  user · 건네짐         →  "손님에게 실행하라고 안내"              ← 구판 문면
```

### 부호표 — 현 문면이 **91%** 틀린 처방을 말한다

```
[ACTION] 발화 1,824회
  ★미건네짐 — 현 문면 **틀림**    통과  550 / 실패 1,103   **49 태스크**
    건네짐 — 현 문면 맞음          통과  109 / 실패    62      8 태스크
  ⇒ 틀린 처방 비율 = 1,653/1,824 = **91%**
```
**파는 쪽 = 0**: 「건네짐」 8 태스크는 **구판 문면 그대로**다. 뒤집는 게 아니라 **조건을 나눈다**.
⚠ 남는 미측정 축: 새 문면이 «give 를 하라»고 밀므로, gold 이 give 를 원치 않는 태스크에서
   해로울 수 있다. **A/B 로 판정한다.**

### 구현

```
_gtool  = A2 value_acquisition[0].give_tool  또는  dispatcher_role_check.give_tool   ← 엔진 리터럴 0
_handed = _tool_given(state.messages, _gtool, _utgt)                                  ← 기존 헬퍼
미건네짐 → "…they cannot run it until you hand it over. Explaining is not enough.
            Call '{give}' with discoverable_tool_name='{tool}' now, then tell them to run it…"
건네짐   → 구판 문면 그대로 (거동 변화 0)
계기      [T2_ACTIONREQ] handover: …                                                  ← [[81]]
```
`test_f8_handover_form.py` **7/7 PASS**(도메인 리터럴 0 검정 · 구판 폴백 검정 포함).

---

## §15 — 야간(2026-09-07 밤 ~ 09-08 새벽) 확정분

### 15-1 클라우드 파동 전량 오염 · 아키텍처 전환

클라우드 하네스 시기(09-07 05:33~14:43)의 **shell 호출 101건 중 101건**이 동일한 211자
(`sandbox-runtime (srt) binary disappeared…`)를 돌려줬다. `TAU2_SANDBOX_FALLBACK` 은
끝내 발효된 적이 없다. 발각은 계기가 아니라 사용자의 육감이었다([[87]]).

오염 단위는 run 이 아니라 **sim** 이다 — shell 을 한 번도 안 부른 sim 은 무오염.
완주 99 sim = 무오염 25 · 오염 74. 무오염 pass 16/25(64.0%) ↔ 오염 40/74(54.1%) 로
결함이 두 집단을 갈라 주지도 않으므로 **어느 쪽 수치도 근거로 쓰지 않는다.**
매니페스트 = `x818_cloud_srt_contamination_manifest.json`(sim 별 calls/failed/contaminated).

⇒ 하네스를 사내(.153)로 옮기고 클라우드는 vLLM 만 남겼다(역터널). 그 뒤 shell 실패 0.
⚠**「rep1 타깃 6/6 회복」 중간 판독은 이 오염분 위에 서 있었다 — 잠정으로 내린다.**

### 15-2 base nt=4 통합(무오염분) — 09-08 05:35

완결 nt=4 **50 태스크 / 200 sim** · 4/4(A셀) **21** · 0/4 **8** · flip 21 ·
sim 단위 pass 134/200 = 67.0%. ⛔이 표본은 001~058 앞머리라 **모집단 값이 아니다.**
분모는 [[68]] 대로 96(102 제외).

### 15-3 task_048 — CONFIRMED 우리-층 결함 (base 3/4 → 1차 0/4)

**① 주장**: A2 `prescription_redirect[0]` 이 «카드 해지 + 유지제안» 대화를 «분쟁 청구»로
분류해 `apply_statement_credit` 을 12회 반려했고, 연쇄 `[BLOCKED]` 와 `[PROCEDURE]`
(closure 전에 disputes 해소 요구)로 4 sim 전부 `db_match=False`(reward_basis=['DB']).

**② 축자 + 위치**: `t2_gate_patch.py:11714-11715`
```python
_conv = " ".join(str(getattr(m2, "content", "") or "") for m2 in state.messages
                 if getattr(m2, "role", None) in ("user", "tool")).lower()
```
신호 스캔 범위에 **`"tool"` 이 들어 있다** = 손님 발화가 아니라 **우리가 검색해 온 KB 문서**에서
의도 신호를 찾는다. 선언은 `a2/banking_knowledge.gate.json:4731` +
`a2/banking_knowledge.specific.json:4487`([[24]] 양 층).

**③ 반증 조건**: 손님이 실제로 분쟁·미승인 결제를 말했다면 분류가 옳다.
**실측 = 아니다.** 손님 발화 11턴 전부 해지·유지제안·신규신청이고 신호어 **0회**.
신호 17회는 **전부 `role=tool`**, 걸린 문장은
*"## Reasons for Closing a Debit Card — Lost card · Stolen card · Suspected
fraud/unauthorized transactions …"* — **해지 태스크면 반드시 읽게 되는 해지 정책 문서**다.
따라서 이 오탐은 우연이 아니라 **해지 계열에서 구조적으로 재현**된다.

**④ 선행확인 경로**: `grep -rn "disputed or unauthorized charge" a2/*.json t2_*.py` ·
`grep -rn "prescription_redirect" *.py`.

**후보 수리**: 스캔 범위를 `("user",)` 로 좁힌다 — 닫힌 술어·도메인 일반·태스크 리터럴 0([[05]][[58]]).
⛔**아직 구현하지 않는다**: `repo_rep1`(rep1 33건)과 `repo_rep2`(스모크)가 도는 중이라
건드리면 조건이 바뀐다([[54]][[86]]). 선행 의무 = [[70]] 부호표 — 이 레버가 원래 잡던
038 계열(신호가 **손님 발화**에 있던 사례)이 `("user",)` 로도 보존되는지 회수분에서 먼저 센다.

### 15-4 A3 doc_index 는 검색에 연결된 적이 없다

11군·71계열이 전부 **상품 계열 축**이다. `Internal:` 절차 문서 47건 중 **36건(77%)** 이
계열 `_general_` 뿐인 군에 떨어진다 = 엔진 스스로 *"고를 것이 없는 축"* 이라 부르는 자리
(`t2_gate_patch.py:4719`). `doc_index` 를 질의로 바꾸는 코드는 `x318_query_formation_iso.py`
**하나뿐이고 그것은 격리 프로브**다 — 엔진 미import · 정본 런처 미등재([[81]]).
격리 검정 = `x829`(팔 4개 · 표적은 env `TransferReasonLiteral` 19코드를 2개 이상 담은
문서 = 698 중 정확히 1건 · gold 불참조).

### 15-5 task_049 — CONFIRMED 우리-층 결함 (base 3/4 → 1차 0/4) · 048 과 **다른** 레버

⚠먼저 정정: 048 의 원인(`prescription_redirect` 오탐)이 계열 전체를 무너뜨린다고 적으려 했으나
**049 에서는 PRESCRIPTION 이 4회뿐**이다(048 은 24회). 049 의 주도 레버는 **E-PLAN(48) · PROCEDURE(23)**.
공통된 것은 «dispute 신호가 전부 `role=tool` 에서만 나온다»(048 74건 · 049 93건 · user 0건)는 사실뿐이다.

**① 주장**: 우리 레버 둘이 **모든 sim 에서 정반대를 지시**했다.
  - `[PROCEDURE]` (계정별로 계산된 6단 체크리스트): *"NEXT: retention_offer -> apply_statement_credit_8472"*
    발화 s1567 3 · s361454 6 · s373753 3 · s626729 7
  - `[E-PLAN]` (A2 `eplan/intent_chains[0].phrase` 의 **고정 산문**):
    *"if it already contains a record, SKIP both the closure-reason logging and the retention offer"*
    발화 4 · 8 · 4 · 4
  모델은 skip 쪽을 택했고 `apply_statement_credit_8472` 를 **한 번도 부르지 않았다**.
  base 는 그것을 **1회 부르고 통과**(3/4). 우리는 0/4.

**② 축자 + 위치**: 선언 = `a2/banking_knowledge.gate.json:161` + `settings.json:150`
  (`/eplan/intent_chains[0]`). 이 조건은 **엔진이 계산하지 않는다** — `phrase` 라는 고정 문자열로
  모델에게 넘어가고, 판정은 모델의 계정별 기억에 맡겨진다. 반면 PROCEDURE 체크리스트는
  계정별로 계산된다. **한쪽은 계산, 한쪽은 보일러플레이트 — 그래서 어긋난다.**

**③ 반증 조건**: crypto 계정의 closure-reason history 에 레코드가 있었다면 skip 이 옳다.
  **실측 = 없다.** 같은 sim 의 반환값:
  `_green` **Found 1 record** · `_eco` **No closure reason record** · `_gold` 없음 · `_crypto` 없음.
  미달 gold 는 `_crypto` 의 `log_credit_card_closure_reason_4521` 과 `apply_statement_credit_8472`
  (+ `transfer_to_human_agents`, 이쪽은 `GB2_NOTICE_BEFORE_TRANSFER` 가 4회 차단).
  ⇒ skip 조건은 **green 에서만 참**인데 대화 전체에 적용됐다.

**④ 선행확인 경로**: `grep -rn "SKIP both the closure-reason logging" a2/*.json *.py` ·
  `grep -rn "closure_reason_history" t2_eplan_patch.py t2_gate_patch.py` (엔진 계산 **없음**).

**경위 (선언이 스스로 적어 둔 것)**: `_note_conditional_2026_09_01` —
*"종전 문구는 로깅을 무조건 지시했고 … 모델이 green 에 simplifying_finances 를 폐쇄 뒤에 또 찍었다"*.
09-01 에 **green 의 중복 로깅**을 막으려 문구를 조건부로 바꾼 것이, 카드 4장 대화에서
**나머지 세 장의 retention offer 와 closure 로깅을 죽였다.** [[70]] 이 요구하는 «무엇을 팔았나»가
그때 계상되지 않았다.

**후보 수리(둘 중 하나, 부호표 후)**:
  (a) skip 조건을 **엔진이 계정별로 계산**해 PROCEDURE 체크리스트와 같은 층에서 판정한다
      (닫힌 술어: 해당 계정의 closure-reason history 반환에 레코드가 있는가) — [[10]] 정합.
  (b) PROCEDURE 가 그 계정에 대해 `retention_offer` 를 NEXT 로 지목하는 동안에는 E-PLAN 의
      skip 문구를 **발화하지 않는다**.
⛔구현 보류 — `repo_rep1`·`repo_rep2` 가 도는 중이다([[54]][[86]]).
