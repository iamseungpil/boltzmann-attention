# 설계서 — 잔여 11 (BOTH 23→천장 34) 원인별 수정 계획 (리뷰 후 구현)

> 상태: **DRAFT, 리뷰 대기.** 현재 active-H3+ARGFIX+VALFIX = **BOTH 23/34 (68%), 회귀 0**. 잔여 fixable = 11 (= 34 − 23). 정직천장 34 = 48 − Part A 8 − Part B 6 (변동 없음; 아래 Cause 2 검증으로 exchange/get_account_owed는 defect 아님 확정 = 천장 34 유지). 모든 수치는 `eval_t1c_argvalfix` + `offload_log_argvalfix.jsonl` (task_sig 조인) + 실제 evaluator + released output 대조로 확정.

## 잔여 11 = 3 cause
| Cause | 수 | goal | 진단(확정) | 일반성 |
|---|---|---|---|---|
| **1. gate ⊊ dirgraph** | 7 | set_safety_box 3·pay_loan 2·pay_bill 1·transfer 1 | goal-call 정확(acc=True)인데 dirgraph_satisfied=False·cnv=False. 게이트가 sampled `task["constraints"]`로 permit → evaluator는 full `directed_action_graph`(login/admin/balance/chain cascade) 요구 → 선행 미수립 ACT | 일반 |
| **2. tuple-return success 인식 실패** | 3 | exchange_foreign_currency 2·get_account_owed_balance 1 | `swarm/core.py:167-168` `raw_result=raw_result[1]`이 tuple success-bool 폐기 → goal-call content="93.0"(값만) → evaluator `action_successfully_called`(evaluator.py:251-254, tuple[0] 검사)가 success 인식 못함. **defect 아님**: released는 content `'(True,93.0)'` 보존 → asc 44/86·22/42 | 일반 |
| **3. active-drive 미완(step cap)** | 1 | transfer_funds | ungathered `sufficient_account_balance`(getter=get_account_balance). active-H3가 turn당 1개씩 구동 → step cap 전 미도달 | 일반 |

---

## Cause 1 — gate를 full directed_action_graph로 (premature 7)
**진단(확정)**: premature 7 전부 `acc=True, cnv=False, dirgraph_satisfied=False`. 게이트 `_check_permitted`는 `cons=task["constraints"]`(sampled)만 평가(`two_stage_client.py:626/666`)하나, evaluator의 `dirgraph_satisfied`는 full `directed_action_graph`(login·admin auth·balance·chain cascade 포함, `evaluator.py:240-273`)를 순회. 게이트가 더 관대 → 선행 establishing 미수립 상태로 ACT → premature. credential은 user_known에 있음(set_safety_box admin_password 포함) ⇒ **fixable**(Part B처럼 불가능 아님).

**수정안**: 게이트·active-H3를 sampled cons가 아니라 **task의 full dependency**(= evaluator가 채점하는 것)로 평가/구동.
- 옵션 A (권장): 게이트가 `cons`에 벤치 `dfsins_cl_cd_aid` cascade를 적용한 결과를 평가 → evaluator.py:267과 동일 구조. active-H3가 그 cascade의 누락 establishing(login_user/authenticate_admin_password/get_account_balance)을 user_known creds로 구동(ARGFIX `_force_call` 재사용).
- 옵션 B: `task["directed_action_graph"]`(nodes/connections)를 dependency-tuple로 변환해 평가.
- **일반성**: cascade는 ABox-derived(도메인 무관 규칙, content는 task graph). dep_full superset 아님 = over-deny 위험 낮음(task-specific graph).
- **회귀 위험**: 中 — 게이트가 더 엄격해져 기존 BOTH/ACT 일부가 STOP 될 수 있음. **flag `SOPBENCH_DGGATE` 뒤 A/B 필수**, 기존 BOTH 23 비회귀 확인.
- **예상**: premature 7 중 다수 → BOTH (establishing 구동 가능분).
- **선행 검증**: directed_action_graph→dependency 변환/dfsins 적용이 evaluator와 leaf-동일한지 1개 task로 단위확인 후 전체.

## Cause 2 — tuple-return goal success 인식 (exchange 2 + get_account_owed 1)
**진단(확정·released 대조)**: `swarm/core.py:167-168`:
```python
raw_result = func(**args)
if isinstance(raw_result, tuple): raw_result = raw_result[1]   # success-bool 폐기
```
→ exchange goal-call content="93.0"(값만). evaluator(`evaluator.py:251-254`)는 `func_resp[0]`(tuple)로 success 판정 → 값만 있으면 인식 불가 → asc=False(우리 6런 전부 0/2, 0/1). **released는 content `'(True, 93.0)'` 보존 → asc 44/86·22/42 = solvable, defect 아님, 천장 34 유지.**

**수정안**: goal-call 결과의 **full tuple 보존**(success-bool 안 버림).
- **선행 확인(필수)**: 이 strip이 `apply_two_stage_patch.py`(우리 패치) 산물인지 원본 SOPBench core.py인지 확인 → 우리면 patch에서 해당 줄 제외(goal 한정), 원본이면 released 정합 위해 동일 처리.
- 최소 변경: goal action(=`task["user_goal"]`) 결과는 tuple 그대로 기록(다른 도구는 영향 최소화 위해 현행 유지 가능) — 단 evaluator는 모든 func_call content를 보므로 **전 도구 tuple 보존이 released와 정합**(권장).
- **일반성**: 완전 일반(도메인 무관·모든 tuple-return goal).
- **회귀 위험**: 低 — content 표현만 변경(model이 "(True,93.0)" 봄, released와 동일). flag `SOPBENCH_KEEPTUPLE` 뒤 A/B.
- **예상**: exchange 2 + get_account_owed 1 = **+3** (dirgraph 이미 True라 asc만 켜지면 BOTH).

## Cause 3 — active-drive 완전성 (transfer 1)
**진단**: transfer ungathered `sufficient_account_balance`(getter get_account_balance). active-H3는 turn당 1 evidence만 구동(`two_stage_client.py` active block) → step cap(10) 전 모든 누락 evidence 미도달.
**수정안**: active-H3가 한 turn에 **다중 evidence 구동** 또는 누락 우선순위·step budget↑. 단순히는 ungathered/argmismatch 큐를 소진할 때까지.
- **일반성**: 일반(driving 완전성).
- **회귀 위험**: 低. flag로 A/B. **예상**: transfer 1 (+ 다른 step-cap 한계 태스크 일부).

---

## 구현·검증 순서 (각 flag 독립 A/B, 회귀 감시)
1. **Cause 2** (가장 싸고 확실, +3 예상, 회귀 低): `SOPBENCH_KEEPTUPLE`. strip 출처 확인 → 수정 → A/B vs 23.
2. **Cause 3** (싸고 低위험): `SOPBENCH_DRIVEALL`. A/B.
3. **Cause 1** (가장 큰 변경·中위험·최대 잠재 +7): `SOPBENCH_DGGATE`. 단위검증(dfsins=evaluator) 후 A/B, BOTH 비회귀 필수.
- 각 단계 task_sig per-task 전이 추적 + 회귀 0 확인 후에만 박제. 누적 목표 BOTH→34.
- **메타규칙 준수**: 강한 주장(각 fix 효과·천장)은 A/B reliable test 후 박제. dead-end 변종 금지. should_T/F·identity 코드 강제.

## 열린 검증 (구현 전 확정할 것)
- Cause 2: strip 출처(patch vs 원본) — `git log -p swarm/core.py` or apply_two_stage_patch 내용 확인.
- Cause 1: directed_action_graph/dfsins가 evaluator와 leaf-동일 산출하는지 1-task 단위검증.
- 전체: 천장 34 불변 재확인(Cause 2가 exchange/get_account_owed를 defect 아님으로 이미 확정).

## 일반화 ToDo 연계 (별도, `HANDOFF_2026_06_05 §6.5`)
본 3 fix는 모두 하버스 offload(A축)·도메인-일반 규칙. weight-내재화(B축 DPO/RFT)·param_mapping inducer-소싱·cross-domain A/B는 §6.5 ToDo로 분리(다음 실험).
