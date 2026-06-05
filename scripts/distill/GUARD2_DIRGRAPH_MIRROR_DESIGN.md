# 설계서 — Guard-2: dirgraph 재구성 == evaluator 단위검증 (Cause-1 선행, 리뷰 후 구현)

> 상태: **✅ PASS (구현·실행 완료, 2-리뷰어 4 refinement 반영).** **결과 (`guard2_dirgraph_unitcheck.py`, 전 48 should_T)**: A1 재구성 = `dfsgather_invfunccalldirgraph(task["constraints_original"], cl,cp, default_dep(opt=**full**), action_params, goal_node)` → **OVER=0 ∧ UNDER=0 (exact match 48/48)**. INPUT AUDIT 통과(directed_action_graph 안 읽음). SAFETY PASS(BOTH 26 OVER-0=over-deny 회귀 불가) + OPTIMALITY PASS(UNDER-0=+8 상한 실현가능). ⇒ **"정책(constraints_original)+도메인 규칙이 cascade 완전결정·oracle 불요" 증명** = A1 비순환·배포-정당. (대조: required/none=UNDER48 subset, constraints[expanded]+full=OVER44.) **⇒ Cause-1(SOPBENCH_DGGATE) 구현 cleared.**
>
> (이하 DRAFT 설계 — 위 PASS로 검증됨.) Cause-1(게이트를 full dirgraph로)의 **유일한 catastrophic 위험 = 재구성 drift(over-deny)**. 회귀는 개념상 불가(현-BOTH는 자기 cascade 충족→permit)지만, 재구성된 graph가 evaluator와 leaf 다르면 현-BOTH를 over-deny. ⇒ **구현 전 재구성==evaluator를 leaf-동일로 단위검증(BLOCKING).** 본 문서 = 그 검증 방법론 + Cause-1 재구성 방식 확정.

## 1. evaluator `dirgraph_satisfied` 메커니즘 (확정, env/evaluator.py)
- L229: `ifcg = deepcopy(task["directed_action_graph"])` — **생성시 빌드된 task-specific graph**(user_known 값 plug-in). `nodes_task/connections_task/inv_nodes_task`.
- 각 func call마다(L256-): `node_ind=inv_nodes[func]` → `node_inds_to_check=connections[node_ind]`(선행) → `dfscheck_called_functions`(L154)가 **선행 함수가 인자-매칭으로 먼저 호출됐는지** DFS 확인 → 아니면 `dirgraph_satisfied=False`(L273).
- 성공 호출 시 innate dep도 satisfied 처리(L266: `dfsins_cl_cd_aid`).
- 빌더: `dfsgather_ifg_func`(helpers.py:986); evaluator도 graph 밖 func엔 이걸 사용(L248).

⇒ **mirror 대상 = task의 goal prerequisite graph(nodes+connections)** = "goal 전에 어떤 establishing/check가 인자-매칭으로 호출돼야 하는가".

## 2. ⚠️ 핵심 난점 — task-specific vs full
`task["directed_action_graph"]`는 **task의 sampled `constraints`로** 빌드됨(generation.py:1313 `inv_func_call_graph`). `dfsgather_ifg_func(domain, goal, default_constraint_option="full")`는 **full** graph → task-sampled와 **다를 수 있음**(full ⊋ sampled). 
- **Option A 정당성 조건**: oracle(`task["directed_action_graph"]` read) 없이, **`task["constraints"]`(배포-가용 정책) + 도메인 dep**로 task-specific graph를 재구성해야 함.
- **미해결 질문(Guard-2가 답할 것)**: generation의 graph 빌드 경로가 sampled constraints를 입력으로 받는가? 받으면 그 경로를 `task["constraints"]`로 호출해 재구성(Option A 성립). full만 되면 task-specific 재구성 불가 → 아래 §5 대안.

## 3. 재구성 방식 (Option A — A1 확정, 입력 audit BLOCKING)
**A1 빌더 확정** (generation.py:1287 = evaluator graph 생성 경로): `dfsgather_invfunccalldirgraph(dep_orig, cl, cp, action_default_dep_orig, action_parameters, user_goal_node)` where **`dep_orig = task["constraints_original"]`**(task의 sampled dep = 배포-가용 정책). cl/cp/default_dep/action_parameters = 도메인 dep 규칙(ABox). **task["directed_action_graph"] 안 읽음.**
- **⚠️Refinement-1 (BLOCKING) 입력 audit**: 재구성 함수의 실제 입력 ⊆ **{domain, goal, `task["constraints_original"]`, 도메인 dep 규칙(cl/cp/default_dep/action_parameters)}** 임을 코드로 assert. `task["directed_action_graph"]` 또는 constraints 밖 sampled 필드 read 시 **FAIL→A2 강등**. **audit 통과 시 A1 exact-match는 순환 아니라 최강결과** = "정책(task constraints)이 cascade 완전결정·숨은 oracle 불요"의 증명(정당성: settled "정책=per-task constraints"=배포가용, 정답 아님). ⇒ A1 채택 = audit 통과 조건부.
- **A2 (fallback + 삼각측량)**: `dfsgather_ifg_func(...,"full")` 후 constraints 밖 정책 leaf prune. A1==A2==taskgraph 삼각측량으로도 유용.
- **금지(B)**: `task["directed_action_graph"]` read = oracle 누출. **상한-probe 한정.**

## 4. 단위검증 방법론 (BLOCKING — OVER/UNDER 비대칭, 커버리지 분리)
**⚠️Refinement-2 (PASS 기준 변경) — 방향 비대칭**:
- **OVER**(재구성이 evaluator보다 leaf 더 요구) → 현-BOTH **over-deny → catastrophic 회귀**.
- **UNDER**(leaf 누락) → premature서 그 establishing 미구동 → **이득만 못 봄, 회귀 0(safe)**.
- ⇒ **안전 게이트(ship 가능) = 현-BOTH에서 OVER 0**. **최적성 게이트 = premature서 evaluator 일치(UNDER 0).** 0/0 exact-match는 *최적성* 목표이지 안전 관문 아님. benign UNDER에 FAIL 금지(불필요 축소판 후퇴 방지).
**⚠️Refinement-3 (커버리지)**:
- **안전축 = 현-BOTH 26 전수** OVER-0 확인(위험 모집단=BOTH 전부, zero-cost니 전수). + 각 BOTH 궤적 dfscheck로 dirgraph_satisfied=True 재현(회귀 0 사전증명).
- **이득축 = premature ≥7 + transfer 1**서 UNDER leaf 식별(어떤 establishing이 빠졌나 = 이득 상한).
**절차**: A1 재구성 vs `task["directed_action_graph"]` 노드+연결+param_mapping 비교(plug-in 값 정규화, 구조만).
**판정**: 안전 PASS = BOTH 26 OVER-0 ∧ dfscheck 재현 → Cause-1 ship 가능. 최적 PASS = premature UNDER-0 → +8 상한 실현.
**FAIL(BOTH OVER>0)**: A1 drift → A2 시도 → 그래도 OVER→ §5 establishing-only 축소판.

## 5. PASS/FAIL 분기 후 Cause-1 진입
- **PASS(안전)**: 게이트가 재구성 graph의 선행을 require, active-H3가 미충족 establishing을 **user_known creds로 결정론 구동**(Guard-4 PASSED=8 전부 creds-OK→`_force_call`이 hallucination 없이 구동=loop-free 근거). flag `SOPBENCH_DGGATE`, A/B vs 26, **BOTH 26 비회귀 필수**. 예상 +최대 8(천장 34).
- **FAIL(BOTH OVER>0)**: 축소판 = 게이트에 **establishing leaves만 추가**. **⚠️Refinement-4: establishing 분류 = ABox predicate kind**(`_evidence_tools`의 `info.get("kind")=="establishable"` = state-pred=login_user/authenticate_admin_password) — **하드코딩 리스트 금지**(drift원·bank특정). balance는 establishable 아니라 **computed-getter**라 별개 취급(코드로 분류 못박기). A2 prune·FAIL-축소판 둘 다 이 predicate-kind 기준 재사용 = 도메인-일반(§6.5 cross-domain 대비).

## 6. 산출물 (구현 시)
- `guard2_dirgraph_unitcheck.py`: 재구성 vs task graph 비교 + 현-BOTH dfscheck 재현. PASS/FAIL 출력.
- 결과 권위본 기록 후에만 Cause-1 구현 진입.

## 메타
- 본 검증은 zero-cost(GPU 불요, eval JSON + 벤치 함수 import).
- Cause-1은 A축(결정론 scaffold) 천장-확립; B축 내재화는 §6.5 별도.
- 강한 주장("재구성=evaluator") = 3/3 leaf-동일 통과 후에만 박제.
