# 개선 가능 fail 전수 census + 수정 레버 (2026-07-06)

> 지시: fail 원인 전수 조사 → **개선 가능한 fail을 실제로 개선**. 정본 입력 = `ASSEMBLED_FAILURE_FORENSIC_2026_06_27`(per-case 51건) + 궤적 정독(t41/t62/t66/t107) + controller 측정(`plan_execute_orch` 32B fail-set).
> 비용 gpt-4.1 0. 불변 [[08]]/[[13]]/[[05]].

## 0. controller 측정이 준 재프레이밍 (★[[08]])
32B fail-set 16개에 `plan_execute_orch`(plan 격리 + batch/status/provenance controller) 실행:
- **14/16이 격리 계획선 이미 core_ok**·controller 수정 **0회 발화**. ⇒ 32B fail은 *구조(planning) 오류가 아니라* **operand/실행-부하 오류**. 구조 controller만으론 안 고쳐짐(고칠 구조가 없음).
- ⇒ 수정 레버 재정의: (a) ⋈-missed=**plan/execute 분리**(격리 plan이 전 주문 포함→결정론 실행), (b) 불가능op/payment=**operand feasibility 게이트**, (c) order-total=**calc**. batch/status(controller)는 14B·부하시만 발화.

## 1. 51 clean fail → 수정 레버 · fixability (양 scale)
| 클래스 | 수정 레버 | fixability | 태스크(32B / 14B) | ~수 |
|---|---|---|---|---|
| **⋈-missed**(멀티엔티티 미완) | **C1 plan/execute 분리**(격리plan→결정론실행) | ✓ 결정론(실행하네스 필요) | t41·t98·t107·t76 / t1·t76·t83·t98·t99·t102·t111 | ~11 |
| **feasibility**(불가능 op) | **operand 게이트**(precondition block→fallback) | ✓ decidable(operand gate) | t10·t34·t57 / t34·t66 | ~5 |
| **payment**(환불카드) | **A2 refund-rule 게이트**(원결제∪gift·DB 5/5) | ✓ decidable | t63 / t8·t14·t51·t53 | ~5 |
| **status-action** | status-fix(controller·완성) | ✓ 결정론 | (t109·t85) | ~2 |
| **batching** | batch-merge(controller·완성) | ✓ 결정론 | (t71 orch) | ~1 |
| **calc**(order-total) | calc offload | ✓ decidable | t67·t68 | ~2 |
| **criterion**(변형) | present 전옵션 + debias | ◐ present+소량 scale | t13·t20·t100 / t8·t20·t27·t58·t110 | ~8 |
| **over-action**(valid-order scope) | — (LLM-scope) | ✗ scale | t33·t62 / t22·t31·t45 | ~5 |
| **loop**(idempotence) | dedup 게이트(저ROI) | ◐ | t69 / t19·t39·t69·t51·t53 | ~5 |
| **artifact**(tracking#·reason enum·format) | — (모델오류 아님) | ✗ | t17·t40·t104 / t30·t38·t40·t103·t104 | ~8 |

**결정론 fixable(✓) 합 = ⋈-missed 11 + feasibility 5 + payment 5 + status/batch 3 + calc 2 = ~26/51(51%).** criterion(◐8)은 present로 대부분. artifact/over-action/loop(~18)=scale/비모델.

## 2. 실제 개선 빌드 (우선순위·[[13]] 결정론 먼저)
1. **operand feasibility+payment 게이트**(feasibility 5 + payment 5 = 10건·가장 decidable·즉시 빌드+오프라인 검증 가능) — 불가능 op(부분취소·product-swap·타주문결제) precondition block, refund=원결제∪gift(DB검증). `feasibility_gate.py`.
2. **C1 plan/execute 실행 하네스**(⋈-missed 11건·최대·격리plan 검증됨 14/16) — plan 1회 + 결정론 walk(전 주문 실행). 빌드 큼·end-to-end 유료 확인.
3. **calc order-total**(2건) — 소량.
- **검증**: 로직=오프라인 단위테스트(무료). PASS-회복=end-to-end 유료 smoke(fail-subset만·[[09]]).

## 3. 남는 것(정직)
over-action(valid-order scope)·criterion 잔여·loop·artifact = ~18/51 = 결정론 불가(scale/LLM-scope/벤치아티팩트). 이게 진짜 scale이 사는 소수 + 비-개선대상.
