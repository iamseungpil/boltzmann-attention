# Task-Instance Constraint 설계서 — should_T 병목(과잉 게이팅 + full-tool 부담) 해소

> 상태: **리뷰 대기 (구현 전)**. 작성 2026-06-02. 권위본 `WORKFLOW_ONTOLOGY_DESIGN.md §11`의 보강.
> 대상 독자: 리뷰어(사용자). 리뷰 후 구현 착수.
> 선행 근거: `SOPBENCH_EXPERIMENT_RESULTS.md` Exp-4a v2 전수조사 + 본 문서 §2(코드 증명).

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

→ SFT 학습쌍이 **모순**: 입력(프롬프트)="transfer_funds BLOCKED, login_user 먼저" / 정답="transfer_funds 바로 호출".
7B는 혼란 정책을 학습하고, 추론 시 무거운 status에 끌려 **불필요한 login/auth를 환각 자격증명으로 호출** → dirgraph 위반(자해).
(full 모드라 그 login 도구가 목록에 노출된 것도 가중.)

### 2.3 궤적 증거
- 우리 v2 task 111: `internal_check → login_user=F(환각 'password') → transfer_funds → authenticate=F` → dirgraph_satisfied=False.
- gpt-4o oracle task 111: `internal_check(user) → internal_check(dest) → transfer_funds` → **통과**(login 없음).
- 차이 = 정책(가벼운 task 제약)을 따랐는가 vs default(무거운) 게이팅에 끌렸는가.

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

## 7. 재학습 분석

- **(A) precondition 렌더 수정**: 입력(프롬프트) 분포가 바뀌므로(무거운→가벼운 status) **SFT 재생성 + 재학습 1회 필요**. 정답·파이프라인 불변, 어댑터만.
- **(B) 도구 프루닝**: 프롬프트의 op_names 집합이 바뀌므로 이 역시 SFT 프롬프트 동반 수정 → A와 함께 1회 재학습으로 흡수.
- 따라서 **A 또는 A+B 모두 재학습 1회**(LODO holdout=bank, 기존 레시피 `lora_train_chat_toolcall.py --system-mode none --max-seq-len 8192`).
- 추론-전용 band-aid(재학습 0)는 가능하나(추론 시 login 선택을 task 제약으로 사후 필터) **band-aid → 비추천**(train/test 불일치 잔존).

---

## 8. 실험 계획 (ablation, Exp-4c 갱신)

분모 = **40**(정직; 결함8 제외). 서빙=GPU1 단일 lora.

1. **E-A (A만)**: precondition을 task 제약으로 정합 렌더 + 재학습 → bank LODO eval.
   - 측정: 과잉 login/auth 호출률↓, should_T(분모40)↑, dirgraph 위반↓. census로 goal_not_reached/dirgraph_value_mismatch 변화.
2. **E-AB (A+B)**: A + 도구 프루닝 + 재학습 → eval.
   - 측정: full-mode가 oracle-유사로 좁혀지며 추가 상승분 = 도구선택 부담 기여 분리.
3. **분리증명(§11.7 연계)**: 빈/틀린 task 제약 주입 → should_T 붕괴(WHAT이 task 제약에서 옴을 입증). 거부축(should_F)은 STOP 보존 확인.
4. **전이 검증**: A(+B)로 bank 개선 확인 후 6 LODO 회전(출처1) → 출처3(정책 직독)으로 전이 강건성.

**성공 기준(사전등록 제안)**: E-A에서 should_T ≥ L0 regression(7/48≈7/40) 수준 회복 AND 과잉 login 호출 ≥50%↓; E-AB에서 추가 상승.

---

## 9. 위험 / 리뷰 포인트

1. **공정성**: `task["constraints"]` 사용이 "치팅" 아닌가? → oracle 도구모드가 directed_action_graph(동일정보)를 쓰고, 정책설명으로 에이전트에 이미 제공됨 → **공정**. 단 논문 보고 시 "정책에서 유도" 프레이밍 명확화. 출처3(정책 직독)이 가장 방어적.
2. **거부축 보존(분리계약)**: A가 fact 게이팅/STOP을 약화하면 안 됨 — task 제약에 **있는** fact는 그대로 VERIFY/STOP. should_F 회복(31/86)이 유지되는지 필수 확인.
3. **B의 표시-프루닝 vs 실제 도구셋**: env는 full 도구목록을 기대 → planner에 **보이는 목록만** 프루닝하고 resolver/env엔 full 유지. 잘못하면 env 오류.
4. **task 제약과 innate-dep 구분**: 일부 goal은 innate-dep(login)이 task 제약과 별개로 dirgraph에 영향. task 111은 login 불요였으나, login이 innate인 goal에선 마스크가 login을 빠뜨리면 안 됨 → 마스크 = `task_constraint ∪ innate_dep(goal)`로 정의(검증 필요, §리뷰질문 Q2).
5. **극難 6개**: 자격증명 부재로 어차피 불가 → 분모40에서도 이 6은 못 풀 수 있음(천장 40이지만 현실 상한 ~34). 보고 시 명시.

---

## 10. 리뷰어 결정 질문

- **Q1. 마스크 출처**: 1차로 출처1(구조적 task 제약)로 갈지, 처음부터 출처3(정책 직독, 전이↑·난이도↑)로 갈지.
- **Q2. 마스크 정의**: `task_constraint` 단독인지 `task_constraint ∪ innate_dep(goal)`인지 (§9.4). login이 innate인 goal의 dirgraph 요구를 먼저 실측 확인 필요.
- **Q3. 범위**: A만 먼저(병목2 격리) vs A+B 동시(둘 다). 각각 재학습 1회.
- **Q4. 도구 프루닝 강도**: 정확히 task 제약 함수만 vs task 제약 + 1-hop establishable(여유). 너무 좁히면 회복경로 차단 위험.

---

## 부록. 관련 아티팩트
- 전수조사: `census_shouldT.py`(bucketing은 default precond 기반=수정 대상), `breakdown_should_tf.py`, `_leaderboard_bankcheck.py`(52모델 task별 통과수), `evidence_a_probe.py`(오라클 8 불가 확정).
- 결과본: `SOPBENCH_EXPERIMENT_RESULTS.md` Exp-4a v2 + Exp-4c.
- 코드: `two_stage_client.py`(build_v2_prompt L70-128), `build_tbox_planner_sft.py`(L84-171), `apply_two_stage_patch.py`, `induce_ontology_zekun.py`(L125-131 args, default precond).
