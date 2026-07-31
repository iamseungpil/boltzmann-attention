# user-side discoverable 이름 정확성 게이트 설계 (2026-07-31 · **설계만 · 미구현**)

> 발단 = `Y2A_FAILURE_FORENSIC_2026_07_31.md` §3. 상위 = [[22]] 닫힌/열린 술어 · [[23]] A2 출처 ·
> [[05]] 고정-변경 경계 · [[19]] 합성-우선.
> **구현·발사 안 함.** Y2-B는 그대로 돈다(사용자 지시 2026-07-31).

---

## 0. 표적 (실측)

task_020 per-step 포렌식, 인자 수준:

```
gold: call_discoverable_user_tool(discoverable_tool_name="submit_cash_back_dispute_0589", …)
pred: call_discoverable_user_tool(discoverable_tool_name="submit_cash_back_dispute", …)   ← 접미사 _0589 누락
pred: call_discoverable_user_tool(discoverable_tool_name="file_reward_dispute", …)        ← 없는 이름
```

## 1. ★선행 음성 증거 — 순진한 처방은 **이미 자해했다**

A2 `discoverable_name_check`에는 이 표적을 겨냥한 `pattern: "_[0-9]+$"`가 **이미 있다**. 그런데 그
키의 주석이 축자로 기록한다:

> *"★2026-07-22 §2bm 교정: give(user-도구)는 접미사 無가 정상(e2e10 bare-give 성공 실증) —
> rall8 038서 **정당 give 6회 오차단**(give-flow 봉쇄→dispute 0)의 **자해 실측** → give를 검사에서
> 제거(unlock만 잔류)."*

**원인이 분명하다**: env의 user-side discoverable 집합은
`{deposit_check_3847, get_card_last_4_digits, get_referral_link, submit_cash_back_dispute_0589}` —
**접미사가 있는 것과 없는 것이 섞여 있다**. "접미사 필수"는 `get_card_last_4_digits`·
`get_referral_link`를 매번 오차단한다.

⇒ **술어를 바꾸지 않고 적용 범위만 넓히면 038이 재발한다.** 이 설계의 핵심은 범위가 아니라 술어다.

## 2. 술어 — 패턴이 아니라 **집합 소속**

```
D = env user-side discoverable 집합           (= _user_discoverable(env) · 이미 엔진에 있음)
base(x) = x에서 _[0-9]+$ 제거

호출 인자 n = discoverable_tool_name 일 때
  n ∈ D                       → 통과 (접미사 유무 무관 — 038 재발 불가)
  n ∉ D ∧ base(n) ∈ base(D)   → ★deny: "기저는 맞고 **접미사가 틀렸다**"
  n ∉ D ∧ base(n) ∉ base(D)   → 이 게이트는 **관여하지 않는다**(기존 unknown-name 계열 담당)
```

- **닫힌 술어**([[22]]): 집합 소속 + 접미사 정규화. 산문 판정 0.
- **적용 도구**: `discoverable_tool_name` 인자를 갖는 호출 전부(`give_…`·`call_discoverable_user_tool`).
  도구 이름을 엔진에 박지 않는다 — **인자 키 이름**으로 찾는다(도메인 리터럴 0).

### 2-1. 왜 세 번째 분기를 비워두나
`file_reward_dispute`처럼 **기저조차 없는 이름**은 "접미사 실수"가 아니라 **발명**이다. 그건
`T2_UNKNOWN_NAME_BL`·`T2_UNKNOWN_REPEAT_GUARD`(둘 다 ON)의 표적이고, 여기서 겹쳐 잡으면
[[19]] 간섭(이중 개입)이 된다. **한 결함에 한 레버.**

## 3. 왜 오차단이 원리적으로 불가능한가

deny 조건이 `n ∉ D`를 포함한다. `get_card_last_4_digits`는 **D에 그대로 있으므로** 어떤 경우에도
1번 분기에서 통과한다. 038의 자해 형태(정당 give 차단)는 **구조적으로 발생할 수 없다.**

⚠단 **Δspurious는 여전히 계측한다**(RESEARCH_MASTER §1.3 — 부작용 없는 레버는 없다). deny가
재발행을 유도하면서 다른 행동을 밀어낼 수 있다.

## 4. [[23]] 출처

- 정책 축자: *"Provide the exact tool name **as specified in the knowledge base**"* ·
  *"Do not invent or guess user discoverable tools"* · *"Only use tool names and arguments
  discovered in the knowledge base"*
- 집합 D의 출처 = **env 레지스트리**(배포 시점에 가진다) ⇒ gold 불참조.
- **A2 신규 키 0** — 피드백 문구는 기존 `discoverable_name_check.feedback`(`{name}` 자리표시자)을
  재사용한다. 즉 **opex 증가 0**.

★**스푼피딩 금지**([[03b]]): 피드백은 **D를 열거하지 않는다**. "그 이름은 등록된 것이 아니다 ·
KB를 검색해 정확한 이름을 찾아라"까지만. 정답 이름을 주면 그건 gold 이식이다.

## 5. [[05]] 결정 3질문 ([[17]] 상설 의무)

1. **scaffold/A2의 도메인-특화를 순증시키나?** — **아니다.** A2 키 추가 0(기존 피드백 재사용),
   집합은 env에서 도출. 엔진에 도메인 이름 0.
2. **모델이 할 수 있는 유동 판단을 결정론에 동결하나?** — **아니다.** *어느 도구를 쓸지*는 여전히
   모델이 정한다. 엔진은 "그 이름이 등록돼 있나"만 본다 — **인터페이스 사실**이지 판단이 아니다.
3. **scaffold가 모델 대신 도메인 행동을 수행하나?** — **아니다.** 이름을 고쳐주지 않는다.
   deny + 재검색 요구뿐(C151 패턴).

## 6. 배선 위치 — ★`unified`에만 (V7 사고 교훈)

**`gated`에 넣으면 안 된다.** go_stack은 `T2_GATE_REGEN=1`이라 `t2_gate_patch.apply()`가 호출되지
않고 `gated`는 설치조차 되지 않는다(`Y2_DESIGN` §12-1). V7이 그래서 죽어 있었다.

- 위치 = `unified`의 deny 사슬(다른 `*_fb`와 같은 자리)
- 우선순위 = 기존 사슬 규약대로 앞선 fb가 전부 None일 때만
- cap = `T2_DISC_NAME_CAP`(기본 6·다른 name 레버와 동형)
- 플래그 = `T2_DISC_NAME_EXACT`(**기본 OFF**)
- 회귀 = `test_lever_reachable.py`가 자동으로 `gated` 전용 여부를 잡는다

### 6-1. 구현 선결 조건 (미확인)
`unified` 스코프에서 **env 객체 접근이 확인되지 않았다**(`self.environment` 유무 미검증).
`_user_discoverable(env)`가 env를 요구하므로, 구현 전 **접근 경로부터 확인**해야 한다. 없으면
`self.tools`+대화 이력에서 집합을 만드는 대체안이 필요하고, 그 경우 **집합이 달라져 술어의 성질이
바뀌므로 재설계**다.

## 7. 계측·판정 (사전등록)

| 지표 | 판정 |
|---|---|
| deny 발화 수 · 재발행률 | deny 후 **정확한 접미사로 재호출**한 비율 |
| **over-block** | D에 있는 이름이 차단된 횟수 = **0이어야 한다**(구조상 불가·실측으로 확인) |
| Δspurious | 다른 write가 밀려났나 |
| 표적 태스크 | 020류(`NAME_ABSENT`/`TOP_VALUE` on `call_discoverable_user_tool`)의 변화 |

**스모크 없이 full-run 금지**([[30]]): 표적 태스크 2건(020·022)으로 먼저 발화·재발행을 본다.

## 8. 한계 (반박 가능하게)

- 표본이 **15 sim·trial 1개**다. 020 하나에서 나온 표적이고, 전수에서 규모가 확인되지 않았다.
  **Y2-B 완주 후 같은 포렌식으로 규모를 먼저 재고** 나서 구현 여부를 정하는 것이 순서다.
- 접미사 실수가 **원인**인지 **증상**인지 미확정이다. 모델이 KB 검색을 안 해서 이름을 모르는
  것이라면, 진짜 표적은 이름 검사가 아니라 **검색 유도**다(기존 `prekb` 계열).
- 020은 이 게이트로 닫아도 **coverage 실패**(같은 txn 4회 반복·§4)가 남아 pass가 안 될 수 있다.
  즉 이 처방의 상한은 "이름 오류 제거"이지 "태스크 통과"가 아니다.
