# 인계(hand-off) 레버 설계 — **찾았으면 줘라 · 전부 줘라** (2026-07-18)

> 계기: 사용자 *"실패 원인 per step 포렌식"* → **pass의 병목이 producer가 아니라 `give`+coverage**임이 실측됨
> (`FAB_PROBES §5.2-C`). 목적 = **τ² pass 올리기**(논문 신규성 아님).
> ⚠️**§8 하드룰 1**: **랭크 前에 반증 예측을 먼저 적는다**(§4). 오늘 프레임 3개가 이걸 안 해서 죽었다.

## 0. 근거 (nt=20 완주·N=20/arm·[[08]] 전수)
| 실패 원인 | dreq2 | ctl2 | 표적 |
|---|---|---|---|
| PASS | 11 | 9 | |
| producer 미호출 | 4 | 5 | (a1)이 78% 커버(§5.2-A/C) |
| ★**`give` 없음** | **4** | 1 | **본 설계 §2** |
| ★**coverage < 4** | 0 | **3** | **본 설계 §3** |
| 기타 | 1 | 2 | |
⇒ **`give`+coverage(8건) ≈ producer(9건)**. 그런데 **레버가 없다**(있는 줄 알았다 — §1).

## 1. ★진단 — 레버는 이미 있었고, **발화하는데도 진다**
`a2 get_reward_discrepancies.follow_up` = `{tool: give_discoverable_user_tool, feedback: "…call
'give_discoverable_user_tool' … **once per discrepant transaction** …"}` · 런 스크립트 `T2_FOLLOWUP_REQUIRED=1`.
**즉 "찾았으면 줘라 + 건마다"가 이미 문구에 있다.** 그런데:

| | `[T2_FOLLOWUP] fired` | regen → `give` | regen → **`[]`(빈손)** |
|---|---|---|---|
| dreq2 | **14** | 8 | **6 (43%)** |
| ctl2 | **14** | 5(+2건은 give×2) | **7 (50%)** |

★**regen의 43~50%가 도구 호출을 안 낸다 = 텍스트로 답한다.** ⇒ **미발화가 아니라 *형식*에서 진다**
(사용자가 제기한 **"선택 vs emit-형식 분리"** 그대로).
⚠️`completion_guard`는 **0회 발화**([S] grep: `claims_completion` 0) — 런 스크립트에 **`T2_WRITE_PROV`가 없다** = 불활성.

### 1.1 ★★두 표면의 뿌리는 **하나**다 — 인계 지점의 **산문 퇴화**
- **`give` 없음**: FOLLOWUP regen이 **도구 대신 텍스트**를 낸다.
- **coverage < 4**: 준 뒤의 **메시지가 구체 값 대신 사용법**을 준다. **축자 대조**(같은 `give` 1회·같은 도구):
  - **PASS**(t16·사용자 4회): *"…for the Thrive Market transaction (**`txn_f093f96e2001`**). You can execute
    this action by calling `call_discoverable_user_tool` with the following parameters: ```json {…}```"*
  - **FAIL**(t15·사용자 3회): *"Here's how you can use it: 1. **Transaction ID:** **You will need** the
    specific transaction ID for each purchase… 2. Use the tool with your user ID and the transaction ID."*
  ⇒ **실패 에이전트는 ID를 *주지* 않고 "ID가 필요하다"고 *설명*한다. 사용자는 모르는 ID를 실행할 수 없다.**
⇒ ***모델은 인계 경계에서 실행 가능한 인공물 대신 설명으로 퇴화한다.*** 두 레버는 그 한 병을 두 곳에서 친다.

## 2. 레버 A — **필수 follow-up regen에서 도구 채널 강제** (`tool_choice=required`)
- **무엇**: FOLLOWUP이 발화해 regen할 때만 API `tool_choice="required"`를 실어 **도구 호출로 답하게** 한다.
- **왜 정당한가**([[16]] 경계): **어느 도구를 부를지는 여전히 모델이 고른다.** 우리는 **채널(형식)**만 강제한다.
  그리고 이 상태는 **ASK가 정당한 출구가 아니다** — 에이전트는 **이미 데이터를 손에 쥐었고**(producer가 반환함)
  남은 일은 인계뿐이다. (ASK 출구가 살아 있어야 하는 일반 상태와 다르다.)
- **기존 자산**: 프로브 `discreq_arm_forced`(`tool_choice="required"`)가 **이미 구현돼 있으나 미실행**
  ([S] `FAB_PROBES`에 결과 0건). ⇒ **먼저 그걸 돌린다**(무료).
- ⚠️**Δspurious 위험**: 강제가 **엉뚱한 도구**를 낳을 수 있다(형식은 샀는데 선택을 판다). **§4 (F1)로 계측.**

## 3. 레버 B — **coverage: 우리가 센다** (엔진이 이미 답을 갖고 있다)
- ★**핵심**: 불일치 목록 **S = {id₁…id_N}은 우리 엔진이 계산한 것**이다(`t2_scaffold_get.exec2`의
  `_res = apply_op(select_discrepant, …)` → 리스트 반환). **모델에게 물을 필요가 없다.**
- **규칙**: producer가 S를 반환한 뒤, **S의 각 id마다 `give` 1회**가 있을 때까지 follow-up을 미충족으로 본다.
  미충족 시 피드백이 **빠진 id를 이름으로** 알린다(A2 템플릿).
- **[[03b]] 안전**: 산문을 파싱하지 않는다. `give` 호출의 **인자(구조)**에서 id를 읽어 S와 대조할 뿐이다.
  (인자 = LLM이 formalize해 넘긴 값 · 엔진은 집합 대조만.)
- **[[05]] 안전**: 엔진 = 집합 대조·카운트. A2 = follow-up 도구명·id 필드명·문구. **도메인 리터럴 0.**
- **구현**: `exec2`가 producer 결과 `_res`를 `self._t2_required_ids[…]`에 적재 → FOLLOWUP 검사기가
  `given_ids ⊇ S`를 요구.

## 4. ★반증 예측 — **사전등록** (결과 보기 前 · §8 하드룰 1)
> **레버 A가 옳다면**: `[T2_FOLLOWUP] regen tool_calls=[]`가 **43~50% → ~0**으로 떨어지고,
> **`give` 없음 실패(4+1)가 줄어야** 한다.
> **레버 B가 옳다면**: **coverage<4 실패(0+3)가 줄어야** 한다.
>
> **반증조건(하나라도 걸리면 그 레버 폐기 · 기함 교체 금지)**:
> - **(F1) 강제가 *엉뚱한 도구*를 낳는다** — regen=[]는 0이 됐는데 `give` 없음이 안 줄면, **형식만 사고 선택을
>   팔았다**(Δspurious>0). ⇒ 레버 A 폐기.
> - **(F2) `give`를 N회 했는데 사용자 실행이 여전히 <N** — 그러면 병목은 **에이전트의 give가 아니라 user-sim의
>   읽기**다. ⇒ 레버 B는 **우리 통제 밖**이고 폐기(§5 참조 — t16은 give 1회로 사용자 4회를 얻었다!).
> - **(F3) pass가 안 오른다** — 두 레버가 각자 표면을 닫아도 pass가 그대로면, **실패가 또 다른 표면으로 이동**한
>   것이다(깔때기·§2b). ⇒ 이동처를 먼저 규명. **"표면 닫음"을 성과로 보고하지 말 것.**

## 5. ⚠️정직한 위험 — **레버 B의 전제가 흔들린다**
**t16(PASS)은 `give`를 딱 1회 했는데 사용자가 4회 실행했다.** t15(FAIL)도 `give` 1회인데 3회.
⇒ **사용자 실행 횟수를 정하는 건 `give` 횟수가 아니라 *에이전트가 말로 준 구체 값*일 수 있다**(§1.1 축자 대조).
- 그렇다면 레버 B(=give를 N회 강제)는 **엉뚱한 것을 강제**하는 셈이다.
- **대안 B′**: `give`의 **인자에 S의 id가 다 실렸는지**를 보는 대신, **S의 각 id마다 give**를 요구하면
  **부수적으로** 각 give가 구체 id를 담게 되고 → 사용자가 그걸 읽는다. **간접이지만 구조적**이다.
- ⚠️**산문을 검사하는 길은 택하지 않는다** — 그건 텍스트 파싱([[03b]])이고, NabaOS가 자인한 한계
  (*"cannot verify every claim about tool output content"*)와 같은 함정이다.
- ⇒ **(F2)가 이 전제를 시험한다. 걸리면 레버 B 폐기하고, 병목이 user-sim이라고 정직 보고.**

## 6. 순서 (전부 무료 먼저)
1. **프로브 `discreq_arm_forced`**(이미 구현·미실행) → 레버 A의 (F1) 사전 판정. **무료.**
2. 레버 B 구현(엔진 집합 대조) + `T2_HANDOFF_COVER=1` **기본 OFF**.
3. **단일변수 라이브 arm**: `(a1)+A+B` vs 대조 — 단 **한 번에 하나씩**이 원칙이나, 세 레버가 **다른 표면**을
   치므로 조합 arm 1개 + 각 단독 arm은 [[09]] 비용상 보류. **조합이 지면 분해.**
4. 판정 = `bank_paired_arms.py`(페어·재시도 공변량) + **§4 반증조건**.
