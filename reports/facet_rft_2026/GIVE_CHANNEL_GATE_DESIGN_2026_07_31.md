# give 채널 게이트 설계 — 넘길 수 없는 것을 넘기는가 (2026-07-31 · **설계만 · 미구현**)

> 발단 = `Y2A_FAILURE_FORENSIC_2026_07_31.md` §9-2(채널 오분류 55건·27%).
> 상위 = [[22]] 닫힌 술어 · [[23]] A2 출처 · [[05]] 경계 · [[19]] 합성 · [[08]] 포렌식 선행.
> **미구현.** Y2-B가 도는 중이며, 구현·반영은 완주 후.
> ⛔선행 교훈: 같은 자리에서 나온 **직전 설계(`DISC_NAME_EXACT`)는 모집단 0건으로 폐기**됐다.
> 이번에는 **규모·전환·over-block을 설계 안에서 먼저 답한다**.

---

## 1. 표적과 규모 (Y1 전수 실측 — 사전 확인 완료)

**채널 오분류**: KB로 도구를 찾아놓고 agent-side `unlock`→`call` 대신 user-side `give`로 넘긴다.

| 실측 | 값 |
|---|---|
| `give` 총 호출 | **89** |
| 대상이 env user-discoverable(정당) | **71 (80%)** |
| **대상이 집합 밖** | **18 (20%)** · 15 sim |
| 그로 인한 gold action 실패(dispatcher 미호출) | **55건 = 전체 실패의 27%** · 8 sim에 집중 |

집합 밖 대상 내역: `apply_for_credit_card` 6 · `submit_transaction` 3 ·
**`submit_cash_back_dispute` 3** · `submit_referral` 2 · `setup_travel_notification` 2 ·
`claim_annual_fee_rebate` 1

**두 종류가 섞여 있다**:
- **(a) 일반 user 도구를 give** — 고객이 **이미 자기 도구로 부를 수 있는 것**(`apply_for_credit_card`
  등)을 굳이 넘긴다. give 자체가 무의미하다.
- **(b) 접미사 누락 표기** — `submit_cash_back_dispute`는 실재 discoverable
  (`submit_cash_back_dispute_0589`)의 잘못된 이름이다.

⇒ 하나의 술어가 둘 다 잡는다. **폐기된 직전 설계와의 차이**: 그건 "접미사 패턴"을 봤고 표적이
0건이었다. 이건 **집합 소속**을 보고 표적이 **18건**이다.

## 2. 술어 (닫힘)

```
D = env user-side discoverable 집합 (= _user_discoverable(env)·엔진에 이미 있음)
give_discoverable_user_tool(discoverable_tool_name = X) 호출 시
    X ∈ D  → 통과
    X ∉ D  → deny + "그 도구는 고객에게 넘길 수 있는 것이 아니다.
                     KB에서 확인하고, agent 도구라면 unlock 후 직접 호출하라"
```

- 집합 소속만 본다 — 산문 판정 0([[22]]).
- **오차단 구조적 불가**: 정당한 give 71건은 전부 D에 있으므로 통과. 038 자해(접미사 패턴으로
  정당 give 6건 차단)와 **형태가 다르다**.
- 도구 이름을 엔진에 박지 않는다(인자 키 `discoverable_tool_name`으로 찾는다).

### 2-1. 기존 레버와의 관계 — **구멍의 정확한 위치**
`dispatcher_role_check`에 *"give 대상 = agent-도구면 deny"*가 **이미 있다**. 그런데 판정을
**`self.tools` 소속**으로 한다. **잠긴 agent-discoverable 도구는 `self.tools`에 없어서** 그 검사를
빠져나간다 — 55건이 새는 구멍이 정확히 거기다.
⇒ **새 레버가 아니라 기존 레버의 판정 집합 교체**(`self.tools` → `env user-discoverable`).
[[19]] 간섭 없음(같은 레버 안에서 판정만 정확해진다).

## 3. ★"막으면 갈아타는가" (사전 확인 — **약한 양성**)

deny가 대안 행동을 만들지 못하면 **transfer만 앞당긴다**. 그래서 자연 전환 증거를 먼저 셌다:

| 잘못된 give가 있던 sim | 15 |
|---|---|
| 끝내 `unlock`을 안 함 | **12 (80%)** |
| 같은 sim에서 `unlock`도 함 | 3 |
| **give 뒤 `unlock`으로 전환한 적 있음** | **3 (20%)** |

**⇒ 전환 능력은 있으나 자발률은 20%다.** 이것은 "deny하면 갈아탄다"의 **약한 양성**이지 보장이
아니다. 따라서 §5의 판정 규칙은 **pass가 아니라 전환율**을 1차 지표로 놓는다.

## 4. [[23]] 출처 · [[05]] 3질문

**출처**: 정책 축자 — *"The unlock step is required before calling — you cannot call a tool that
hasn't been unlocked"* · *"Do not invent or guess user discoverable tools"* · *"Only use tool names
and arguments discovered in the knowledge base"*. 집합 D = **env 레지스트리**(배포 시점 보유).
**A2 신규 키 0**(피드백은 기존 `dispatcher_role_check.give_agent_tool_feedback` 재사용).

1. **도메인-특화 순증?** — **아니다.** A2 키 추가 0·엔진 도메인 리터럴 0·집합은 env 도출.
2. **유동 판단 동결?** — **아니다.** *어떤 도구가 필요한지*는 모델이 정한다. 엔진은 "그것이 넘길 수
   있는 물건인가"만 본다 = **인터페이스 사실**.
3. **scaffold가 도메인 행동 수행?** — **아니다.** unlock을 대신 해주지 않는다. deny + 안내뿐.

★**스푼피딩 금지**: 피드백은 D를 열거하지 않고 "정답 도구 이름"도 주지 않는다([[03b]]).

## 5. 계측·판정 (사전등록)

| 지표 | 1차/2차 | 판정 |
|---|---|---|
| **전환율** = deny 후 그 sim에서 `unlock`→`call`로 간 비율 | **1차** | 자연율 **20%** 대비 상승해야 의미 |
| **over-block** = D에 있는 give가 차단된 수 | **1차** | **0이어야 한다**(구조상 불가·실측 확인) |
| transfer 조기화 | **1차** | deny 후 transfer가 늘면 **역효과** — 중단 사유 |
| Δspurious · 턴 수 | 2차 | 부작용 없는 레버는 없다(§1.3) |
| pass | **2차** | 55건이 8 sim에 몰려 있어 **최대 8 sim**만 움직인다 |

**중단 조건**: over-block ≥ 1 또는 transfer 조기화가 전환보다 크면 즉시 OFF.

## 6. 배선 위치

**`unified`에만**(=`dispatcher_role_check`가 이미 있는 자리). `gated`는 이 스택에서 설치되지
않는다(V7 사고·`Y2_DESIGN` §12-1). `test_lever_reachable.py`가 자동 검사한다.
플래그 = 기존 `T2_DISPATCH_ROLE` 하위(별도 플래그 불요 — 판정 집합만 바꾸므로).
⚠**단 판정 집합 교체는 기존 레버의 동작을 바꾼다** ⇒ 롤백용으로
`T2_DISPATCH_ROLE_ENVSET`(기본 OFF)를 두고 스모크 후 승격.

## 7. 한계

- 55건이 **8 sim에 몰려** 있다. 태스크 다양성이 낮아 **한 태스크 유형의 특성일 수 있다.**
- (b)형(접미사 누락)은 이 게이트가 잡지만, 모델이 **정확한 이름을 모르면** deny가 KB 재검색을
  유도해야 한다 — 그 경로는 `prekb` 계열과 겹친다([[19]] 간섭 감시점).
- **자연 전환율 20%**가 상한을 시사한다. 8 sim 중 실제로 갈아탈 수 있는 것은 소수일 수 있다.
