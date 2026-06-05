# 설계서 — Guard-2: dirgraph 재구성 == evaluator 단위검증 (Cause-1 선행, 리뷰 후 구현)

> 상태: **DRAFT, 리뷰 대기.** Cause-1(게이트를 full dirgraph로)의 **유일한 catastrophic 위험 = 재구성 drift(over-deny)**. 회귀는 개념상 불가(현-BOTH는 자기 cascade 충족→permit)지만, 재구성된 graph가 evaluator와 leaf 다르면 현-BOTH를 over-deny. ⇒ **구현 전 재구성==evaluator를 leaf-동일로 단위검증(BLOCKING).** 본 문서 = 그 검증 방법론 + Cause-1 재구성 방식 확정.

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

## 3. 재구성 방식 (Option A 후보, 우선순위)
- **A1**: generation의 task-graph 빌드 함수를 `task["constraints"]`로 직접 호출(가능하면 1급). 입력=domain+goal+task constraints, 출력=graph. evaluator가 쓰는 것과 동일 함수면 정의상 일치.
- **A2**: `dfsgather_ifg_func(...,"full")` 후 task constraints에 없는 정책 leaf를 **prune**(sampled로 축소). establishing(login/admin/balance)은 유지.
- **금지(B)**: `task["directed_action_graph"]` 직접 read = per-task 정답 graph = oracle 누출, 배포 부정당. **상한-probe로만.**

## 4. 단위검증 방법론 (BLOCKING 기준)
**대상 ≥3 task, 반드시 포함**: ①현-BOTH 1+(over-deny 회귀 관문) ②premature 1+(set_safety_box, login+admin) ③transfer 1(dual-username+admin+balance).
**절차** (zero-cost, eval JSON + 벤치 import):
1. 각 task: 재구성 graph(A1/A2) vs `task["directed_action_graph"]`를 **노드 집합 + 엣지(연결) + param_mapping** 수준에서 비교(user_known plug-in 값은 정규화, 구조만).
2. **PASS 기준 = 3/3 task에서 nodes∪connections leaf-동일**(추가/누락 0). 특히 현-BOTH에서 재구성이 evaluator보다 **더 요구(over)하면 FAIL**(그게 over-deny 회귀 원인).
3. 추가 확인: 재구성 graph로 현-BOTH 궤적을 dfscheck하면 dirgraph_satisfied=True 재현(회귀 0 사전증명).
**FAIL 시**: Cause-1 Option-A 중단(drift catastrophic). §5 대안 또는 B는 상한-probe 한정.

## 5. PASS/FAIL 분기 후 Cause-1 진입
- **PASS**: 게이트가 재구성 graph의 선행(establishing+check)을 require, active-H3가 미충족 establishing(login/admin/balance)을 user_known creds로 구동. flag `SOPBENCH_DGGATE`, A/B vs 26, 회귀 0 확인. 예상 +최대 8(천장 34).
- **FAIL(재구성 drift)**: 축소판 = 게이트에 **establishing leaves만 추가**(login_user/authenticate_admin_password + balance-getter, dep_full에서 state-pred 필터 = 정책 superset 회피). graph 순서검증은 포기하되 establishing 충족만 강제 = premature의 주원인(선행 미수립) 부분 해소. 단위검증 = 이 establishing-filter가 evaluator innate/establishing과 일치하는지.

## 6. 산출물 (구현 시)
- `guard2_dirgraph_unitcheck.py`: 재구성 vs task graph 비교 + 현-BOTH dfscheck 재현. PASS/FAIL 출력.
- 결과 권위본 기록 후에만 Cause-1 구현 진입.

## 메타
- 본 검증은 zero-cost(GPU 불요, eval JSON + 벤치 함수 import).
- Cause-1은 A축(결정론 scaffold) 천장-확립; B축 내재화는 §6.5 별도.
- 강한 주장("재구성=evaluator") = 3/3 leaf-동일 통과 후에만 박제.
