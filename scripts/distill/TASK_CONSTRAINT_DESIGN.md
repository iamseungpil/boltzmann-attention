# Task-Instance Constraint 설계서 — should_T 병목(과잉 게이팅 + full-tool 부담) 해소

> 상태: **(b) 설계 thesis 재정립 (2026-06-02) — §8.5.★가 최우선 권위본. 리뷰 `TASK_CONSTRAINT_IMPL_REVIEW.md`(C1-C6) 반영.**
> 작성 2026-06-02. 권위본 `WORKFLOW_ONTOLOGY_DESIGN.md §11`. 리뷰=`TASK_CONSTRAINT_DESIGN_REVIEW.md`·`TASK_CONSTRAINT_IMPL_REVIEW.md`. 결과=`SOPBENCH_EXPERIMENT_RESULTS.md` Exp-4c. related-work=`EXPERIMENT_DESIGN_v1_7 §9`.
> ★근본목표(재확정): **NL 멀티턴 요청 → 도메인 온톨로지(ABox)로 재해석 → 내부 dirgraph(절차) 추론·실행하는 agentic planner를 weight(TBox)에 학습 → held-out은 ABox swap만으로 재학습0 전이.** injection/steering 라인 폐기(null). FM SO.P·CAP-CPT(weight-baking)와 달리 분리·전이.
> ★최우선 설계(§8.5.★): **① 도구 이름 ALIAS 마스킹(1급, anti-cheat·전이 핵심)** ② 출처3(NL정책, precond 정답 미렌더)·출처1/3 강등 ③ 멀티턴 user_sim 평가(leaderboard 동일세팅) ④ 헤드라인=LODO 전이+ablation(빈ABox붕괴·L0 vs L2·alias on/off). binding=검증 도구 선택; 레버=완전 게더+act/STOP(resolver 계산, LLM 무계산). genuine-impossible=14(8 PartA+6 PartB).

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

**★ root cause = GT means-ends teacher가 불완전한 검증 시퀀스를 생성.** 두 결함이 합쳐짐:
```python
# build_tbox_planner_sft.py:108
goal_fact_checkable = [p for p in dict.fromkeys(gleaves)        # (1) NAME dedup → 동명 2회 체크 소멸
                       if p not in gest and p in tool_names]    # (2) establishable 제외 (3) 동명 callable만
```
→ (1) transfer_funds source/dest 동명 체크 붕괴, (2)(3) condition fact 체크 누락. **모델은 "덜 검증"을 학습** → dirgraph 위반. **천장은 login 게이팅·모델 용량이 아닌 SFT 데이터 생성 결함.**

### ★ 레버 분해 — 오프라인 정적 분석(`lever_decomp.py`, 재학습0). 44 실패의 제약 leaf 173개:
| leaf class | 건수 | 처리 |
|---|--:|---|
| A_callable_check (`internal_check_username_exist` 등) | 88 | **레버 A: args-aware** ((name,args) 키) |
| **B_condition** (`minimal_eligible_credit_score`·`sufficient_account_balance`·`no_credit_card_balance_on_card`·… **`kind:condition, by:null`**) | **47** | **레버 B: condition→getter 매핑** |
| handled_est (login/auth) | 38 | 기존 establishable로 처리됨 |

**task 단위(44 실패)**: **A-only=11**(transfer_funds dest 등 콜러블 체크만) · **needs-B=33**(set_safety_box×10·get_loan×4·pay_loan×4·cancel_cc×4·transfer×4·…, condition 1개 이상).

- **레버 B가 dominant.** root cause는 args-aware보다 **온톨로지 induction 결함**: `induce_ontology_zekun.py`가 모든 condition 술어를 **`by:null`** 로 남겨 → teacher가 자격/잔액/한도 fact를 **검증할 HOW 지식이 없음**(콜러블도 아님). dirgraph는 이를 getter(`internal_get_credit_score`·`get_account_balance`·`get_credit_card_info`…)로 검증.
- **B는 inducible(비-oracle)**: condition→getter는 도메인 상수(co-occurrence로 깨끗이 도출됨, `lever_decomp.py` 출력). 즉 ABox HOW 지식으로 induce 가능 — oracle 누수 아님.
- **⚠️ 정정 (free 검증 `_verify_ab`, 2026-06-02)**: 이전의 "A+B → 40/48 확정"은 **철회**. free 검증이 **A+B 불완전**을 발견:
  - **C: innate-dep 누락** — dirgraph가 `login_user`(일부 `get_account_owed_balance`)를 요구하나 task constraint에 없는 task ~8(get_loan·pay_bill·pay_loan·set_safety_box·transfer 120). **P5 "task_constraint 단독 충분" 반증** → 게더 = **task_constraint ∪ innate_dep ∪ condition→getter (A+B+C)**. (단 dirgraph가 flat OR이라 login이 strict 필수인지 구조적 확인 필요.)
  - **B 테이블 미완**: `maximum_deposit_limit`(deposit_funds) unmapped → 매핑 추가.
  - ~~천장 미재현~~ → **§8.2에서 harness sim으로 재현·확정.**

### 8.2 ★ harness sim 확정 (`run_scripted.py`, 결정론 scripted-gather + 실제 evaluator)
hand-replay 2회(15·9) 실패 원인 규명·수정(content `"True"`→bool 복원, 호출순서 check→login→auth→getter→goal) 후 신뢰 sim 확보. **oracle plan sanity = should_T 37/48**(천장 재현).

| mode (full, 결정론) | should_T |
|---|--:|
| oracle plan (sanity) | **37/48** |
| **abc (A+B+C)** | **37/48 = oracle** |
| ab (A+B only) | 24/48 |
| 실제 7B baseline | 4/48 |

- **under-verification = binding 확정**: 완전 게더(A+B+C)로 4→**37 = oracle 천장**.
- **★C(login/auth establishment) 필요·+13 기여** (ab 24 → abc 37). 출처 정정: C는 **goal의 induced 온톨로지 precondition의 establishable**(login/auth)에서 도출. (이전 "C 불필요(abc==ab)"는 C를 `dep_innate`(=null)에서 찾던 버그였음.) **evaluator의 dirgraph는 task 제약이 아니라 default deps(=induced precond)를 따르므로 login 필수.**
- **마스크 정정**: ~~task_constraint + getter~~ → **goal의 induced precondition을 완전 충족**(A: args-aware 콜러블 체크 + B: condition→getter + C: establishable login/auth). = "induced precond를 끝까지 establish+verify".
- **gap-13 root cause**: login 누락(전수 동일). **gap-3(37 vs 40)** = 값-반환 goal(exchange_foreign_currency×2·get_account_owed_balance×1) — **`run_scripted` 아티팩트, 벤치 결함 아님**: `evidence_a_probe`(저자 오라클, 8개 확정 기준) 재실행 시 이 3개는 **통과**(oracle-fail은 여전히 8개=cancel_credit_card×6+pay_bill_with_credit_card×2). ⇒ **결함 보고 목록은 8개 유지, 3개 추가 금지**(같은 기준으로 통과하므로). scripted oracle 37 = 실질 오라클 40.
- **full 모드 모든 도구 가용** → 배포 실현 가능. should_F는 scripted 미검증(항상 goal 호출)→재학습 시 거부축 별도 보존(P3, 14-scope).
- ⚠️ 원 §2 "login 과잉호출"과 정합: 7B는 login을 **틀린 자격증명으로 호출해 실패**(login=F)→dirgraph 위반.

### 8.3 ★★ login은 task-conditional + credential-conditional — 조건화 필수 (realistic 검증)
abc=37은 **account 레코드 자격증명 증강(cheating)** 에 의존. 에이전트는 그 creds에 접근 불가 → `--realistic`(user_known creds만):

| abc (A+B+C) | should_T |
|---|--:|
| account-cred 증강 | 37/48 |
| **realistic (user_known만)** | **21/48** |

**16개 통과가 cred-cheating이었음. 정직한 에이전트 천장 = 21/48**(37/40은 oracle/DB-cred 천장 — evidence_a_probe·메모리의 "40"도 동일 증강 사용).

**login 분류(should_T 48)**: dirgraph_login=True 41 / False 7(task-optional 맞음). 그중 **login 필요 ∧ user_known creds 부재 = ~16**(사용자가 자격증명 미제공 → 정직한 에이전트 완료 불가 ≈ 극難 일반화).

**⇒ per-task 조건화 필수** (login은 task-conditional + credential-conditional).

### 8.4 ★★★ "16 cred-absent"의 정확한 분해 (leaderboard 59 released files 재확인) — §8.3의 realistic=21 정정
이전 §8.3 "realistic=21 / 16 전부 cred-부재→refusal"은 **16을 lump한 오류, 철회.** 저자 released 모델 통과수로 16은 둘로 갈림:

| 그룹 | tasks (idx) | full-mode 모델 통과 | 성격 |
|---|---|--:|---|
| **8 도구선택** | apply_cc(0,2)·deposit(28)·set_safety_box(78,98)·transfer(111,115,124) | 10~17개 ✅ | **정보 가용 → 레버로 극복** (성공궤적=getter 호출) |
| **8 자격증명-부재** | get_loan(39,44)·pay_bill(56)·pay_loan(66,67)·set_safety_box(76,89)·transfer(120) | **0~1개** (oracle 포함) | identification(+admin_pw) user_known 부재 → 정직 극복 불가 |

- **사용자 지적 수용**: 16 중 **8은 도구선택**(oracle/leaderboard 통과 = 정보 있음, getter 빠뜨린 게 문제). 레버(A+B+C, getter 포함)로 극복.
- **단 8은 진짜 cred-부재**: 0~1/42 모델만 통과(oracle 도구 줘도). evidence_a_probe 통과는 account creds 읽는 cheat. 정직 에이전트는 불가 → 거부/can't-do.
  - **admin DB-read 검증(`_admincheck.py`)**: creds는 DB에 있고 `internal_get_database()`가 반환(identification·admin_password). 8개 dirgraph에 그 노드 있음. **그러나 internal_get_database는 에이전트에 미노출(full·oracle 둘 다); 노출 internal_*는 check/score뿐.** ⇒ admin으로 읽으려면 벤치가 그 도구를 노출해야 함=셋업변경+impersonation 보안문제. 현 도구셋 내 정직 극복 불가 확정.
- **정직한 천장 = ~32/48** (48 − 8 결함 − 8 cred-부재), NOT 21. 레버 타깃 32(baseline 4).
- **조건부 login**: cred 가용 시 호출(8 도구선택 포함 32), cred 부재 시 환각 금지(8). 이게 §2 7B 실패(cred-부재 task에서 비번 환각→login=F)와 정합.

→ **구현 1순위 = B(condition→getter 온톨로지 induction) + A(args-aware 게더)**. mechanism A(게이팅 경량화, §7)는 should_T 비-binding이므로 부차(프롬프트 정합으로 흡수).

### should_F gross churn (R6/R7 종결)
net 31→31은 **2-task churn을 은폐**: GAIN `[87] set_safety_box`(fail→pass) + LOSS `[100] set_safety_box`(pass→fail), 둘 다 auth=F 처리. 문서가 보고한 "fragile 1 회귀(100)"는 5-scope monitor가 100은 잡았으나 **87 gain은 놓침(R7 확증)**. → should_F는 **net 아닌 gross로 보고**.

### 다음 (binding 규명·오프라인 상한 확정 후)
1. **★레버 B = condition→getter 온톨로지 induction** (`induce_ontology_zekun.py`의 `by:null` 정정 → condition 술어를 getter에 연결) — needs-B 33 task 커버. **dominant.**
2. **레버 A = args-aware 게더** (`build_tbox_planner_sft.py:108` (name,args) 키 + `build_v2_prompt`의 `observed`/`facts` args-aware) — A-only 11 task(transfer dest 등).
3. A+B 반영 → **SFT 재생성 + 재학습 1회**. (오프라인 상한 40/48 확정됐으므로 재학습 가치 있음.)
4. mechanism A(게이팅, §7)는 should_T 비-binding → 부차(프롬프트 정합으로 흡수).
5. should_F는 **gross gain/loss 모니터**(net 금지, 14-scope). 개선 확인 후 6 LODO 회전 → 출처3 전이.

> 추가 오프라인 검증(선택, 재학습 전): A+B로 고친 teacher가 44 실패 task에서 dirgraph 노드를 전부 재구성하는지 replay로 확인 → 상한 40/48 직접 재현.

---

## 8.5 ★★★ (b) 구현 설계 — 완전 검증 게더 (should_T + should_F 동시), 2026-06-02 확정

> 이 절이 **구현 권위본**. §2~§8의 진단·검증 결론을 (b) 코드 설계로 종합. 모든 수치는 실측(`run_scripted`·`binding_diag`·`lever_decomp`·leaderboard union·`_shouldF2`·`_verifyF`).

### 8.5.★ 최우선 설계 원칙 (2026-06-02 thesis 재정립 — 이 블록이 §8.5 전체를 지배)

**근본 목표(재확정)**: 자연어 멀티턴 요청을, **도메인별 구조화 온톨로지(ABox)로 재해석**해 **내부적으로 절차(dirgraph)를 추론·실행**하는 agentic planner를, 작은 모델 weight(TBox)에 학습시키고, **본 적 없는 도메인은 ABox 교체만으로 재학습 0 전이**. (injection/steering 라인은 폐기=실증 null; FM SO.P·CAP-CPT의 weight-baking과 달리 분리·전이.)
- **TBox(weight, 학습·전이)** = "NL 요청 + ABox 어휘 → dirgraph(절차) 도출 + 절차 실행" 스킬. **dirgraph는 모델이 만들어내는 출력**(입력 컨닝 아님). L0는 NL→dirgraph 불가 → 이 매핑이 비자명·대체불가 기여.
- **ABox(in-context, swap)** = 도메인 도구 affordance(능력·설명) + NL 정책. goal의 precondition '정답 구조'는 안 떠먹임.

**① [최우선] 도구 이름 ALIAS 마스킹** (현재 `--alias` stub=미구현 → **1급 구현**):
- 프롬프트의 도구를 **per-task 추상 alias**로 제시(`tool_A: 사용자 존재 확인`), 타깃도 alias, **resolver가 alias→실제 도구 매핑**.
- 효과: (a) 이름 암기 차단(within-domain entanglement 봉쇄; 순서shuffle로 못 막던 부분), (b) **NL 조건 → 도구 설명 의미매칭 강제**(lexical 지름길 제거=학습 비자명, C1 정면 응답), (c) 전이 깨끗(held-out=새 alias+새 설명, 같은 스킬), (d) **"답지 컨닝" 원천 차단**(precond를 렌더해도 도구가 `tool_3`이라 무의미). → **출처1/3 논쟁보다 근본적인 anti-cheat.**
- 트레이드오프: 7B 난이도↑(이름 힌트 없이 설명만으로). β(전이 논제)엔 정답인 선택.

**② 출처1/3 강등**: 둘 다 NL→goal+args 매핑은 모델이 하므로 thesis는 둘 다 성립. **출처3(NL 정책 블록, precond '정답목록' 미렌더)을 기본**으로 하되, alias가 진짜 방어선이므로 1급 결정 아님. (G0 확인: NL 정책이 절차를 담음. 단 `build_v2_prompt`가 정책을 600자 truncate → **goal 관련 블록 전달로 수정**.)

**③ 평가 = 멀티턴 user_sim, leaderboard 동일 세팅 비교**: 현재 정적-user(`default_response` 1회 덤프) → **멀티턴 user_sim 전환**. 함의: 멀티턴이면 에이전트가 부족정보 되묻기 가능 → PartB(cred-부재) 일부 해소 가능(단 **user_sim이 자격증명을 보유·제공하는지 선확인 필요**, 천장 재산정).

**④ 헤드라인 실험(thesis 입증)**: ① **LODO 전이**(6→held-out, ABox swap, 재학습0) ② **ablation**: 빈/틀린 ABox→붕괴(온톨로지 실제 사용) · in-context(arm-3v2 2/48) vs 학습(arm-4a) vs **L0(arm-2)** · alias on/off. ③ 멀티턴 user_sim pass@1. **SOTA 절대수치보다 "재학습0 전이"가 1급 결과.**

> 아래 8.5.0~8.5.6은 **메커니즘 디테일**(완전 게더·resolver·should_F 교정)로 위 ★원칙 하에서 유효. resolver 계산 오프로딩·양축 게더는 그대로, 단 **planner 입력은 alias+NL정책(출처3)·precond 정답목록 미렌더**로 구현.

### 8.5.0 결론 한 줄
should_T·should_F 공통 binding = **검증 도구 선택**(어떤 체크/establish 도구를 호출하는가). 레버 = **완전 검증 게더 + act/STOP 게이트**, 단 **LLM은 도구 선택만 하고 정확한 계산은 결정론 resolver(ABox executor)가** 수행(사용자 설계: "모든 조건/계산을 도구호출로 환원").

### 8.5.1 메커니즘 (two-stage, LLM 무계산)
- **planner(LLM, TBox)**: 추상 검증액션 선택(예 "sufficient_account_balance 확인") + `observed` **bool**만 보고 act/STOP 결정. 산술·비교 안 함.
- **resolver(결정론, ABox)**: 추상 검증액션 → **실제 raw getter 호출**(`get_account_balance` 등 → func_call 로그 → dirgraph 노드 충족) + **내부 결정론 비교**(balance ≥ amount, score ≥ 임계 등) → bool을 `observed`에 기록.
- **게이트**: `observed` 전부 True → **goal 호출**(should_T); 하나라도 False → **STOP**(should_F 거부).
- **검증완료**: (i) raw getter 호출이 dirgraph 충족(`run_scripted` abc=37). (ii) should_F 거부는 goal 미호출로 success=True(`_verifyF`; dirgraph 불요). (iii) 새 wrapper 도구명 불필요 → dirgraph 안 깨짐(resolver가 raw getter 호출).

### 8.5.2 should_T·should_F 대칭 + should_F 샘플 교정 (핵심)
현재 teacher는 should_F 샘플을 **생성하나 불량**: condition-위반 should_F가 위반 조건을 게더하지 않고 fallback STOP → SFT가 "이유 없는 STOP"(예 apply_credit_card should_F: `[check_username=True, STOP]`, 위반인 credit_score 미게더) → 판별 무의미·과잉거부 위험(현 should_F 31/86).
**(b) 교정**: 완전 게더로 위반 조건을 게더·관찰 후 STOP → `[check_username=True, check_credit_score=False, STOP]` = **이유기반 거부 학습**. should_T(전부 True→act)·should_F(하나 False→STOP) 양축 SFT 동시 정합.

### 8.5.3 게더 구성요소 (A+B+C)
- **A — args-aware callable check**: `internal_check_username_exist` 등은 (name,args) 키로(동명 2회=source+dest 둘 다). `dict.fromkeys` dedup 제거.
- **B — condition→getter+compare (resolver)**: condition 술어(`by:null`)를 결정론 체크로. 매핑(leaderboard co-occurrence 확정):
  `minimal_elgibile_credit_score`→`internal_get_credit_score`≥`minimum_credit_score`; `sufficient_account_balance`→`get_account_balance`≥amount; `no_credit_card_balance_on_card`/`not_over_credit_limit`→`get_credit_card_info`; `safety_box_eligible`→`get_account_balance`≥`minimum_account_balance_safety_box`; `*_owed_balance_restr`/`pay_loan_*_restr`→`get_account_owed_balance`(+`get_account_balance`); `maximum_exchange_amount`/`maximum_deposit_limit`→내부 임계.
- **C — 조건부 login/auth**: goal의 induced precond에 establishable(login/auth)이 있고 **자격증명이 user_known에 가용**할 때만 호출. cred 부재 시 **환각 금지**(호출 안 함; §2 7B 실패 모드 회피). `run_scripted` 검증: unconditional→realistic 21, conditional→realistic 29.

### 8.5.4 변경 파일
1. **`two_stage_client.py`**:
   - resolver: condition 술어 처리 추가 — 추상 체크 선택 시 raw getter 호출 + 결정론 비교 → `observed[pred]=bool`. (기존 fact-visibility `observed` 확장: 동명 callable뿐 아니라 condition도.)
   - `build_v2_prompt`: condition 술어를 `observed` bool과 함께 표시(VERIFY FIRST/READY/BLOCKED-by-FACT). 게이트 룰 동일(observed-False→STOP).
2. **`build_tbox_planner_sft.py`**: `goal_fact_checkable`을 **(name,args)-aware + condition(→getter경유) + 조건부 login/auth** 로 교체. `next_decision`: 위반 condition 게더 후 STOP(이유), 전부 True면 establish→goal. should_T·should_F 양축 GT 재생성.
3. **train/test 일치(§6.4)**: 위 둘이 동일 게더·동일 `build_v2_prompt` 거치도록.

### 8.5.5 목표·평가
- 평가는 **실제 파이프라인**(`run_simulation`→`run_evaluation`); 값-반환 goal(exchange/owed_balance) scoring은 실제 evaluator가 정상 처리(run_scripted 아티팩트는 무관).
- 타깃: should_T **4→~34**(정직 천장; 14 불가 제외=8 PartA 코드결함+6 PartB cred-부재), should_F **31→다수**(86 중 ~83이 도구-탐지 거부트리거: 38 auth-fail+37 condition+8 callable).
- 134 환산: 우리 35 → 천장 union 120(=34 sT+86 sF). 단일모델 SOTA full 103/oracle 107. 레버는 sT·sF 양축 동시 상승 노림.
- 모니터: should_F **gross** gain/loss(net 금지). LODO holdout=bank.

### 8.5.6 검증 완료 / 미해결
- ✅ 검증완료: 게더 작동(abc 37=oracle), C 조건화 필요(21→29→37), should_F 거부=success, raw getter→dirgraph, 14 genuine-impossible(8+6), should_F 트리거 83/86 도구탐지.
- ⬜ 미해결(구현 중 확인): resolver의 condition→getter 매핑이 **모든 도메인**에서 일반화되는지(bank 외 6도메인 condition 술어 매핑); 7B가 condition `observed` bool로 게이팅을 실제 학습하는지(재학습 후 측정).

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
