# Task-Instance Constraint 설계서 — should_T 병목(과잉 게이팅 + full-tool 부담) 해소

> 상태: **리뷰 + zero-train 게이트 + §8.1 binding 진단 완료 (2026-06-02). 진짜 레버 규명 → 다음=검증-게더 args-aware 수정.**
> 작성 2026-06-02. 권위본 `WORKFLOW_ONTOLOGY_DESIGN.md §11`의 보강. 리뷰=`TASK_CONSTRAINT_DESIGN_REVIEW.md`.
> 선행 근거: `SOPBENCH_EXPERIMENT_RESULTS.md` Exp-4a v2 + 본 문서 §2(코드 증명)·§7(zero-train)·§8.1(binding 진단).
> ★요약 결과: mechanism A는 라이브 작동(login -59%)이나 should_T 불변. **§8.1: should_T 실패의 84%(37/44)가 "필수 CHECK 미호출"(login 0). root cause=GT teacher가 검증 체크를 name-dedup·establishable 제외로 불완전 생성(`build_tbox_planner_sft.py:108`). 진짜 레버=검증-게더 args-aware·완전화(게이팅 아님).**

---

## 0. 한 줄 요약

arm-4a v2의 should_T 병목은 **인자 바인딩이 아니라 "task별 실제 제약을 무시하고 domain-default(최대) 제약으로 무겁게 게이팅"** 이다.
planner 프롬프트(`build_v2_prompt`)가 goal의 precondition을 **domain-default 온톨로지**로 렌더해서, 실제로는 login이 불필요한 task에도
"login 먼저 하라"고 지시 → 7B가 불필요한 login/auth를 **환각 자격증명으로 호출** → 스스로 dirgraph 위반. 동시에 **full-tool 모드**(도구 ~20개)가
7B의 도구선택 부담을 키운다. 두 병목의 **공통 뿌리 = "WHAT(어떤 제약이 적용되나)"을 task별이 아니라 domain-default로 다룸**.
해법: **task-instance 제약을 per-task ABox로 사용**해 (A) precondition status를 정합 렌더 + (B) 도구목록을 task-관련으로 프루닝.

---

## 1. 정정된 사실관계 (이전 분석 오류 포함)

리뷰어가 먼저 알아야 할 정정 사항:

1. **should_T 천장 = 40/48, NOT 24.** strict-oracle 직접검사(`evidence_a_probe.py`, 2026-06-02 재실행)로 **오라클-불가 = 8개뿐**
   (cancel_credit_card×6 `goal_return=False`, pay_bill_with_credit_card×2 `KeyError:'credit_limit'`; 원인 §11.13 credit_cards list/dict). 나머지 40은 오라클 통과.
2. **이전 "자격증명 부재 16개 불가" 주장은 bucketing 오류였음.** census가 자격증명 요구를 **domain-default precondition**으로 판정해
   통과 가능한 task를 "불가"로 과대분류. 52개 공개 모델 대조 결과: 그 16개 중 **8개는 19~30모델이 통과**(명백히 가능),
   6개(39/44 get_loan·56 pay_bill·76/89 set_safety_box·120 transfer_funds)는 전원 미통과지만 **오라클은 통과 = 극難(자격증명이 에이전트엔 부재)·결함 아님**, 2개(66/67) 경계.
3. **leaderboard >50%는 정당.** 상위권은 거의 전부 **oracle 도구모드**(gpt-4o oracle 62% vs full 35%). 대부분 task는 login 불요(username 체크만),
   login 필요 시 자격증명이 user_known에 있음(task 78), fact는 호출가능 도구(`internal_get_credit_score`, `get_account_balance`)로 검증.

→ 본 설계는 위 정정 위에서, **full-mode·7B에서 40 천장에 접근**하기 위한 것.

---

## 2. 병목의 코드-증명 (root cause)

### 2.1 oracle vs full (메커니즘)
`run_simulation.py:152-157`:
- `tool_list=oracle`: `included_functions = [n[0] for n in task["directed_action_graph"]["nodes"]]` → **이 task에 필요한 도구만** 제공(~3-5개).
- `tool_list=full`: `included_functions=None` → **도메인 전체 도구**(~20개), 에이전트가 선택.

### 2.2 핵심 버그 — 프롬프트와 정답의 제약 출처 불일치
`build_tbox_planner_sft.py`(SFT 데이터 생성):
- **GT 정답 시퀀스**: `de.process(goal, **slots)` (L122-124). `de`는 `task_dep[goal]=task["constraints"]`(L86, **가벼운 task 제약**) 기반.
  → task 111 transfer_funds면 username 체크만 통과 시 곧장 `goal` 반환 = **login 안 부름(정답)**.
- **프롬프트**: `build_v2_prompt(ont, ...)` (L151) — goal status를 **`ont`=induced 온톨로지(default `dep_full`, 무거움)** 로 렌더.

실측 대조(rr.ps1):
```
induced transfer_funds precond (default):
  AND(internal_check_username_exist(dest), logged_in_user, authenticated_admin_password,
      CHAIN(internal_check_username_exist(user), sufficient_account_balance))   # 무거움
task 111 실제 constraints_original:
  AND(internal_check_username_exist(user), internal_check_username_exist(dest)) # 가벼움 (login/auth 없음)
assistant_prompt 정책(에이전트가 보는 것): "transfer_funds: username 존재 + destination 존재" # 가벼움 = task 제약과 일치
```

→ **메커니즘 = "비단사(non-injective) 프롬프트"** (2026-06-02 SFT 데이터 실측, 리뷰 P1 반영 정정).
이전 초안의 "일관된 모순쌍" 표현은 부정확. 실제 v2 SFT(login-default goal들)의 (프롬프트 goal-status)×(GT 타깃):

| 프롬프트 status | 타깃 | 건수 |
|---|---|--:|
| BLOCKED-login | login_user | **76** |
| BLOCKED-login | GOAL(login 건너뜀) | **24** |
| BLOCKED-login | other(internal_check 등) | 89 |
| READY | STOP(fact 거부) | 28 |

즉 **똑같은 "BLOCKED-login" status가 무거운 task엔 login(76), 가벼운 task엔 skip(24)이라는 다른 정답에 매핑** —
status가 task 제약을 무시하고 default로 렌더되어 **두 클래스를 구분 못 하는 비단사 입력**. policy 텍스트는 task 제약을
담지만 explicit status 줄이 misleading. 모델은 구분 불가 입력에 모순 지도를 받아 blend → 추론 시 login 과/오호출
(자격증명도 없어 환각). → **mechanism A(status를 task 제약으로 렌더 → 입력을 단사로)가 정당화됨.** (full 모드라 그 login 도구가 목록에 노출된 것도 가중.)

### 2.3 궤적 증거
- 우리 v2 task 111: `internal_check → login_user=F(환각 'password') → transfer_funds → authenticate=F` → dirgraph_satisfied=False.
- gpt-4o oracle task 111: `internal_check(user) → internal_check(dest) → transfer_funds` → **통과**(login 없음).
- 차이 = 정책(가벼운 task 제약)을 따랐는가 vs default(무거운) 게이팅에 끌렸는가.

### 2.4 진단 게이트 결과 (무료, 리뷰 P2·P3·P6 응답; 2026-06-02 실측 `_gates_p2p3.py`)
**P2 — A 수혜/위험 분포 (should_T 48, n≈1 함정 해소)**: A_HELPS(default-login인데 task-light)=**14** · task가 genuinely login 필요(A가 올바르게 무겁게 렌더, **망치지 않음**)=31 · neither=3. → **A의 addressable=14**(미니멀이 아님). A는 task가 login 필요한 31엔 무손해 → **should_T에서 A 순손해 위험 낮음.**

**P3 — should_F(거부축) 회귀 위험 (통과 31 분해)**: PRINCIPLED(fact-False 거부, 안전)=**16** · STOP/기타=9 · **ACCIDENTAL(auth 실패로 우연 거부)=5 [A-위험]** · goal-호출-통과=1. → 리뷰 P3가 옳음: **5/31(set_safety_box)이 fragile**(auth=F로 우연 통과 → 경로 변하면 뒤집힐 수 있음). 단 16 principled는 A가 손대지 않음(fact 게이팅 보존). **재학습/배포 전 이 5건 회귀 모니터 필수.**

**P6 — reconciled 분모 (단일 등록)**: **48 should_T = 8 결함(오라클불가) + 6 극難(오라클통과·52모델 미통과) + 2 경계(1-2모델) + 32 통상가능.**
- 보고: **주 지표 = should_T/48**(leaderboard 직접비교) · **보조 = /40**(오라클천장 정규화, 결함8 제외). "현실 상한 ~32~34"는 본문 주석으로만, **헤드라인 분모는 /48·/40 둘로 고정**(/34 등 혼용 금지).

---

## 3. 설계 원리 — WHAT vs HOW 분리

현재는 **"무엇을 요구하나(WHAT)"** 와 **"각 요구를 어떻게 충족하나(HOW)"** 를 domain-default precondition 하나에 뭉쳐서 항상 무겁다.

| 축 | 내용 | 성격 | 출처 |
|---|---|---|---|
| **WHAT** (요구 제약 집합) | 이 task에 적용되는 precondition 술어들 | **task별·가변** (light/heavy) | task 정책 / `task["constraints"]` |
| **HOW** (각 술어 충족법) | 술어→establishing 도구, 의존구조, fact 검사도구 | **domain-불변** | induced 온톨로지(means-ends 지식) |

**분리 후**: planner는 task 정책에서 WHAT을 읽어 **그 task에 실제로 필요한 술어만** 게이팅하고, HOW는 불변 온톨로지에서 가져온다.
default `dep_full`는 "타입 공간(모든 가능한 제약의 superset)"이지 "이 task의 ABox"가 아니다. **올바른 ABox granularity = per-task 인스턴스.**

> TBox/ABox 정합성: 이는 분리계약(§11.1)을 약화가 아니라 **강화**한다. ABox를 도메인뿐 아니라 task별로도 swap하는 더 미세한 형태. TBox(means-ends 추론)는 불변.

---

## 4. 메커니즘 — 제약-마스킹 ABox

domain ABox(HOW 지식)를 **task 제약으로 마스킹**해 task-active 술어만 "required"로 활성화.

- **(A) precondition status 정합 렌더**: `build_v2_prompt`가 goal의 precondition을 **task 제약**으로 렌더.
  - task 제약에 없는 술어(예: login)는 status에서 제외 → goal이 "READY"로 표시 → 과잉 login 호출 제거.
  - task 제약에 있는 fact는 그대로 VERIFY FIRST/STOP 게이팅(거부축 보존).
  - **효과**: 병목2(과잉 게이팅) 해소. **GT 정답은 이미 task 제약 기반(L86)이라 불변 → 프롬프트만 정합화.**
- **(B) 도구목록 프루닝**: planner에 보이는 도구를 **task 제약 관련 함수 + establishable + goal**로 한정.
  - 이 셋 = 사실상 oracle 셋(directed_action_graph)이나, **숨은 오라클이 아니라 에이전트가 보는 정책에서 유도** → 공정.
  - **효과**: 병목1(full-tool 부담) 해소 — full을 "정책-유도 oracle-유사"로 축소.

### light/heavy 조절이 자동으로 되는 이유
마스크 = `domain_ABox ∩ task_constraint`. task 제약이 username 체크뿐이면 가볍게, balance/credit-score fact를 포함하면 그만큼 무겁게 — **task별로 자연히 조절**. planner는 "항상 최대"가 아니라 "이 task가 명시한 만큼"만 게이팅.

---

## 5. 마스크 출처 — 3안 (공정성·강건성 trade-off)

| 안 | 출처 | 장점 | 단점 | 전이성 |
|---|---|---|---|---|
| **출처1 (구조)** | `task["constraints"]` 직접 | 정확·단순. oracle의 DAG와 동일정보=공정 | 추론 시 client에 task 제약 plumbing 필요 | 중 |
| 출처2 (NL파싱) | assistant_prompt 정책 파싱→구조화 | 가장 공정(에이전트가 보는 것만) | NL 파싱 brittle | 중 |
| **출처3 (정책 직독)** | planner가 정책 텍스트를 직접 읽고 WHAT 판단; ABox는 HOW만 | 가장 강건·일반·전이↑ | 학습 난이도↑ | 高 |

**1차 추천 = 출처1**(A+B를 가장 직접 검증, oracle-격차 두 요인 동시 제거). **후속 = 출처3**(전이·일반성 위해 정책 직독으로 WHAT 내재화).

---

## 6. 구현 스펙 (변경 파일·함수)

### 6.1 `two_stage_client.py`
- `build_v2_prompt(abox, op_names, ..., goal_constraint=None)`: 인자 추가. `goal_constraint`가 주어지면 **goal 연산자의 precondition을
  `op["precondition"]` 대신 `goal_constraint`로 렌더**. 비-goal 연산자는 기존 온톨로지 precond 유지(HOW 표시용).
- `TwoStageClient.__init__(..., task_constraints=None, prune_tools=False)`: 추론 시 task 제약 주입.
  - `_plan_v2`에서 `build_v2_prompt(..., goal_constraint=self.task_constraints)`.
  - `prune_tools=True`면 `tools`를 {goal} ∪ {task 제약 함수} ∪ {establishable by-actions}로 필터해 planner 프롬프트의 `op_names` 구성.
    (resolver에 넘기는 실제 tool spec은 full 유지 — env가 full 도구목록을 기대하므로 표시만 프루닝.)
- **plumbing**: `apply_two_stage_patch.py`가 클론의 simulation 루프에서 `task["constraints"]`를 client에 전달하도록 패치(이미 task 객체 접근 가능 지점 존재; §6.3).

### 6.2 `build_tbox_planner_sft.py`
- L151 `build_v2_prompt(ont, shown, ...)` → `build_v2_prompt(ont, shown, ..., goal_constraint=task["constraints"])`.
- `prune_tools`면 `shown`(tool order)도 task-관련으로 한정해 **train/test 프롬프트 일치**(§11.4 셔플은 프루닝 후 집합 내에서).
- GT 정답 로직(`next_decision`)은 불변(이미 task 제약 기반).

### 6.3 `apply_two_stage_patch.py`
- 멱등 패치에 "task 제약을 client.task_constraints로 set" 1줄 추가(턴 시작 시 `client.reset(task_constraints=task["constraints"])` 형태).
- `--prune_tools` 플래그 전파.

### 6.4 train/test 일치 불변식 (§11.4 준수)
프롬프트를 만드는 **모든 경로(SFT 생성·추론)가 동일 `build_v2_prompt(goal_constraint=..., 프루닝 동일)`** 를 거쳐야 함. 불일치 시 학습 무효.

---

## 7. ★ zero-train 게이트 결과 (2026-06-02 실측, `_lighten_compare.py`) — 재학습 보류

리뷰 P7대로 **재학습 전 zero-train 진단**을 먼저 실행(env `SOPBENCH_LIGHTEN`, 추론 시 goal status를 task 제약으로 렌더, 재학습 0).

| 지표 | baseline v2 | LIGHTEN | 게이트 |
|---|--:|--:|:--:|
| login/auth 호출 (A_HELPS 14) | 17 | **7 (-59%)** | (i) ✅ |
| login/auth 호출 (전체) | 183 | 151 | — |
| should_T | 4/48 | **4/48** | (ii) ❌ |
| should_F | 31/86 | 31/86 | — |
| should_F fragile 회귀 | — | 1 (task 100) | (iii) ✅ |

- **R1 반증·A 작동 확인**: login 호출이 실제 감소(no-op이면 불가) + 유닛테스트(task 111 OFF=BLOCKED-login / ON=VERIFY-internal_check, DIFFERENT=True). **mechanism A는 라이브로 정확히 작동.**
- **게이트 (i)✅(ii)❌(iii)✅ → 미통과 → 재학습 보류.** A가 불필요 login을 제거했으나 **should_T 불변** = login 과잉호출은 실재했으나 **should_T의 binding constraint가 아님**. **정밀 분해는 §8.1**: 실패 44건 중 **37(84%)이 "필수 CHECK 미호출"**, login 단독 binding=**0**.
- **R2 비대칭**: 어댑터가 무거운 프롬프트 학습 → lighten은 OOD. should_T null은 "A 무효"가 아니라 **"login만으론 should_T 안 움직임"**. zero-train은 positive-only 확증 도구 — null은 비결론. 깨끗한 A 검정은 재학습 필요하나 **기대값 낮음**.

### 재학습 분석 (참고, 게이트 통과 시에만)
- (A) 렌더 수정·(B) 도구 프루닝 모두 입력분포 변화 → SFT 재생성+재학습 1회(LODO holdout=bank, `lora_train_chat_toolcall.py --system-mode none --max-seq-len 8192`). GT 정답·파이프라인 불변.

---

## 8. 실험 계획 / 다음 (Exp-4c, zero-train 후 갱신)

분모 = **/48 주 · /40 보조**(reconciled §2.4 P6; "7/40" 등 카운트-분모 혼용 폐기).

**현 상태**: zero-train(§7)에서 게이트 미통과 → **재학습 아님.** binding 진단(§8.1)으로 진짜 레버를 규명함.

### 8.1 ★ binding-constraint 진단 결과 (2026-06-02 실측, `binding_diag.py`, args-aware)

should_T 실패 **44건**(48 중 4통과)을 **task 제약 기준 + args-aware**로 재census(구 `census_shouldT`의 default-precond 버그 + name-dedup 맹점 동시 수정). LIGHTEN run 기준:

| binding | 건수 | 내용 |
|---|--:|---|
| **필수 CHECK 미호출 (under-verification)** | **37 / 44 (84%)** | dirgraph가 요구하는 검증 도구를 planner가 안 부름 |
| checks_ok_other (순서/잉여호출/goal-skip) | 7 / 44 | 검증은 다 했으나 순서·잉여 login·goal 미호출 |
| login/auth 실패가 *단독* binding | **0** | (login은 어떤 should_T 실패의 단독 원인도 아님 — §7 결론 확증) |

**미호출 CHECK의 종류**(37건 분해):
- **transfer_funds — 두 번째(destination) username 체크 누락** (~8): task 제약=`AND(check(username), check(destination_username))`인데 planner는 `check(source)` 한 번만 부르고 goal 호출. **task 111 LITE 궤적**: `check(john_doe) → transfer_funds` (login 제거됨) → dirgraph F (`check(alice_smith)` 영구 누락). **R8 종결: 111은 login이 아니라 dest-check 누락으로 실패.**
- **set_safety_box — 자격 fact 체크 누락** (~12): `minimal_eligible_credit_score`, `safety_box_eligible` 미호출.
- **cancel/pay_bill_cc — 카드 fact 체크 누락**: `internal_check_credit_card_exist`, `no_credit_card_balance_on_card`, `not_over_credit_limit`.
- **get_loan/pay_loan — 대출 제약 체크 누락**: `get_loan_owed_balance_restr`, `pay_loan_account_balance_restr`, `pay_loan_amount_restr`.
- **balance/exchange**: `sufficient_account_balance`, `maximum_exchange_amount`.

**★ root cause (코드 확정, `build_tbox_planner_sft.py:108`)** — GT means-ends teacher 자체가 불완전한 검증 시퀀스를 생성:
```python
goal_fact_checkable = [p for p in dict.fromkeys(gleaves)        # (1) NAME으로 dedup → 동명 2회 체크(dest) 소멸
                       if p not in gest and p in tool_names]    # (2) establishable 제외 (3) 동명 callable만
```
→ (1) transfer_funds의 source/dest 동명 체크가 하나로 붕괴, (2)·(3) 자격/잔액/한도 등 establishable·비-동명 fact 체크가 누락. **모델은 충실히 "덜 검증하기"를 학습** → dirgraph 위반. **즉 should_T 천장은 login 게이팅도, 모델 용량도 아닌 SFT 데이터 생성 결함.**

→ **진짜 레버 = mechanism A(게이팅 경량화)가 아니라 "검증-게더를 args-aware·완전하게"**: `goal_fact_checkable`을 (name,args) 키로 + establishable/compound fact 포함하도록 재정의. 프롬프트(`build_v2_prompt`)의 `observed`/`facts`도 동일하게 args-aware. **이것이 다음 구현 1순위.**

### should_F gross churn (R6/R7 종결)
net 31→31은 **2-task churn을 은폐**: GAIN `[87] set_safety_box`(fail→pass) + LOSS `[100] set_safety_box`(pass→fail), 둘 다 auth=F 처리. 문서가 보고한 "fragile 1 회귀(100)"는 5-scope monitor가 100은 잡았으나 **87 gain은 놓침(R7 확증)**. → should_F는 **net 아닌 gross로 보고**.

### 다음 (binding 규명 후)
1. **★검증-게더 args-aware·완전화** (위 root cause 수정) → SFT 재생성 + 재학습 1회. **이것이 should_T 천장의 진짜 레버.**
2. mechanism A(게이팅)는 **부차**(login 호출 위생엔 유효하나 should_T 비-binding) — A를 1과 함께 흡수(프롬프트 정합).
3. should_F는 **gross gain/loss 모니터**(net 금지), 14-scope.
4. 개선 확인 후 6 LODO 회전 → 출처3 전이.

---

## 9. 위험 / 리뷰 포인트 (zero-train 후 갱신)

1. **공정성 — A와 B 분리(P4)**:
   - **A(status를 task 제약으로 렌더)는 공정**: 에이전트가 보는 정책 NL과 동일 제약의 구조적 표현(새 정보 아님). 헤드라인 가능.
   - **B(도구 프루닝)만 semi-oracle**(≈directed_action_graph) → **E-AB는 별도 조건 보고**, "full 천장근접" 헤드라인은 A 또는 출처3로만.
2. **거부축 보존(P3) — 회귀 모니터 = 14건(R3)**: should_F 통과 31 = 확인-안전 16(fact-STOP) + 확인-위험 5(auth-fail 우연거부) + **미정 9(STOP/other, principled 미확인)** + 1. A 변경 시 **5+9=14건**을 회귀 모니터(zero-train서 task 100 1건 이미 회귀). 사전등록 기준: 14건 중 회귀 ≤2.
3. **B의 표시-프루닝**: planner에 **보이는 목록만** 프루닝, resolver/env엔 full 유지(env 오류 방지).
4. **마스크 정의(P5 해소)**: zero-train서 task 제약 단독으로 login이 깨끗이 제거됨(A_HELPS 17→7) → **마스크 = `task_constraint` 단독으로 충분**(login 필요 task는 그 login이 task 제약에 포함). `∪ innate_dep`는 불필요로 판정(이전 §9.4 스테일, 정정).
5. **극難 6개**: 자격증명 부재로 실제 미통과(분모는 /48·/40 고정, "~34"는 주석으로만).

---

## 10. 리뷰어 결정 질문 (zero-train 후 — 대부분 해소)

- **Q1(마스크 출처)**: ~~미정~~ → 헤드라인엔 출처1 부적합(P4), 출처1=진단상한·보고는 출처3. **단 재학습 자체가 보류(§7)** 라 당면 무의미.
- **Q2(마스크 정의)**: ~~미정~~ → **`task_constraint` 단독 확정**(§9.4, zero-train 실측).
- **Q3(범위 A vs A+B)**: ~~미정~~ → **둘 다 보류**(게이트 미통과). 먼저 §8.1 binding 진단.
- **Q4(프루닝 강도)**: B 착수 시 task 제약 함수 + 1-hop establishable(회복경로 보존). 보류 중.

**현 결정 질문(신규)**: §8.1 binding-constraint 진단부터 갈지(권고), 아니면 should_F 회귀를 무릅쓰고 E-A 재학습을 강행할지.

---

## 부록. 관련 아티팩트
- 전수조사: `census_shouldT.py`(bucketing은 default precond 기반=수정 대상), `breakdown_should_tf.py`, `_leaderboard_bankcheck.py`(52모델 task별 통과수), `evidence_a_probe.py`(오라클 8 불가 확정).
- 결과본: `SOPBENCH_EXPERIMENT_RESULTS.md` Exp-4a v2 + Exp-4c.
- 코드: `two_stage_client.py`(build_v2_prompt L70-128), `build_tbox_planner_sft.py`(L84-171), `apply_two_stage_patch.py`, `induce_ontology_zekun.py`(L125-131 args, default precond).
