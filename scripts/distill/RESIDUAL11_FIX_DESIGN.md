# 설계서 — 잔여 11 (BOTH 23→천장 34) 원인별 수정 계획 (리뷰 후 구현)

> 상태: **REVIEWED (2-리뷰어 엔도스, 4 가드 반영), 구현 진입.** 현재 active-H3+ARGFIX+VALFIX = **BOTH 23/34 (68%), 회귀 0**. 잔여 fixable = 11 (= 34 − 23). 정직천장 34 = 48 − Part A 8 − Part B 6 (변동 없음; 아래 Cause 2 검증으로 exchange/get_account_owed는 defect 아님 확정 = 천장 34 유지). 모든 수치는 `eval_t1c_argvalfix` + `offload_log_argvalfix.jsonl` (task_sig 조인) + 실제 evaluator + released output 대조로 확정.
>
> **★census exhaustiveness 증명**: **23(BOTH) + 7(Cause1) + 3(Cause2) + 1(Cause3) = 34 = honest 천장 정확 분할.** 잔여가 정직분모를 빈틈없이 partition = **숨은 잔여 없음**의 내부정합 증거. (Part A 8 + Part B 6은 별도 14, 비-BOTH에 분산, 천장 밖.)
>
> **★메타 프레이밍 (헤드라인 정직성)**: 본 3 fix는 전부 **A축 = 결정론 scaffold 천장-확립**(하버스 offload, 도메인-일반 규칙)이지 **모델 학습 아님**. "결정론 scaffold가 ~33/34 천장 도달"은 **A축 주장으로만** 기술; "모델이 내재화"와 절대 혼합 금지. 진짜 과학 deliverable = **B축(§6.5 DPO/RFT 내재화)**. 게이트가 깨끗해질수록 A−B 격차가 곧 내재화 과제의 크기.

## 잔여 11 = 3 cause
| Cause | 수 | goal | 진단(확정) | 일반성 |
|---|---|---|---|---|
| **1. gate ⊊ dirgraph** | 7 | set_safety_box 3·pay_loan 2·pay_bill 1·transfer 1 | goal-call 정확(acc=True)인데 dirgraph_satisfied=False·cnv=False. 게이트가 sampled `task["constraints"]`로 permit → evaluator는 full `directed_action_graph`(login/admin/balance/chain cascade) 요구 → 선행 미수립 ACT | 일반 |
| **2. tuple-return success 인식 실패** | 3 | exchange_foreign_currency 2·get_account_owed_balance 1 | `swarm/core.py:167-168` `raw_result=raw_result[1]`이 tuple success-bool 폐기 → goal-call content="93.0"(값만) → evaluator `action_successfully_called`(evaluator.py:251-254, tuple[0] 검사)가 success 인식 못함. **defect 아님**: released는 content `'(True,93.0)'` 보존 → asc 44/86·22/42 | 일반 |
| **3. active-drive 미완(step cap)** | 1 | transfer_funds | ungathered `sufficient_account_balance`(getter=get_account_balance). active-H3가 turn당 1개씩 구동 → step cap 전 미도달 | 일반 |

---

## Cause 1 — gate를 full directed_action_graph로 (premature 7)
**진단(확정)**: premature 7 전부 `acc=True, cnv=False, dirgraph_satisfied=False`. 게이트 `_check_permitted`는 `cons=task["constraints"]`(sampled)만 평가(`two_stage_client.py:626/666`)하나, evaluator의 `dirgraph_satisfied`는 full `directed_action_graph`(login·admin auth·balance·chain cascade 포함, `evaluator.py:240-273`)를 순회. 게이트가 더 관대 → 선행 establishing 미수립 상태로 ACT → premature. credential은 user_known에 있음(set_safety_box admin_password 포함) ⇒ **fixable**(Part B처럼 불가능 아님).

**수정안 = Option A 확정** (B는 금지, 아래 정당성): 게이트가 `cons`에 벤치 `dfsins_cl_cd_aid` cascade를 적용한 결과를 평가 → `evaluator.py:267`과 동일 구조. active-H3가 그 cascade의 누락 establishing(login_user/authenticate_admin_password/get_account_balance)을 user_known creds로 구동(ARGFIX `_force_call` 재사용).
- **★Option A vs B = 정당성(legitimacy) 문제** (리뷰 지적, 핵심): **B(`task["directed_action_graph"]` read) = oracle 누출** — per-task 정답 그래프는 실배포에 없음(라벨). **A(도메인규칙 dfsins on dep + task constraints로 cascade 재구성) = 배포-현실적.** ⇒ A만 정당, **B는 상한-probe로만·배포 주장 금지.**
- **★dep_full 재발 아님** (settled 화해): 철회된 `dep_full` = 태스크-무관 정책 superset(**constraint축**)→비-sample 정책 over-deny. 본 cascade = task-specific 검증 순서(**dirgraph축**). **다른 축**이므로 "철회한 거 다시 걷기" 아님.
- **★login settled 화해**: "login=BOTH-레버 아님(T1T2 4/4 실패)"은 **uniform 모델-강제 login** 실패. Cause 1 = **task-specific 결정론 게이트-구동**(active-H3, creds 사용) = **다른 메커니즘**. 모순 아님.
- **★회귀는 개념상 불가** (리뷰 reframing): 기존 BOTH 23은 이미 `dirgraph_satisfied=True`=자기 cascade 충족 → dirgraph-게이트가 여전히 permit. 회귀는 **오직 dfsins 재구성 ≠ evaluator leaf(over-deny 버그)일 때만**. ⇒ 中위험 아니라 "단위검증 통과 시 低위험 / dfsins drift 시 catastrophic."
- **⚠️BLOCKING 1 — 단위검증 ≥3 task (현-BOTH 포함)**: dfsins cascade가 evaluator와 **leaf-동일** 산출하는지 확인. **현-BOTH 태스크에서 over-deny가 안 나는지가 회귀의 유일 관문.** 1개 아니라 ≥3개(BOTH 포함) 통과 전 전체 런 금지.
- **⚠️BLOCKING 2 — 7개 creds 가용성**: "+7"은 7개 각각의 establishing leaf(login/admin_password) creds가 user_known에 **실재할 때만** 실현. 없으면 dirgraph-게이트는 premature(acted)→**DENY-never-acts(루프)+환각-비번 병리 재발**, BOTH 이득 0. → set_safety_box 3(admin_password 있다 주장=확인)·pay_loan 2·pay_bill 1·transfer 1 per-task creds 먼저 확인.
- **예상**: creds 가용분만큼 premature 7 → BOTH.
- **flag `SOPBENCH_DGGATE`**, BLOCKING 1·2 통과 후 A/B, BOTH 23 비회귀.

## Cause 2 — tuple-return goal success 인식 (exchange 2 + get_account_owed 1)
**진단(확정·released 대조)**: `swarm/core.py:167-168`:
```python
raw_result = func(**args)
if isinstance(raw_result, tuple): raw_result = raw_result[1]   # success-bool 폐기
```
→ exchange goal-call content="93.0"(값만). evaluator(`evaluator.py:251-254`)는 `func_resp[0]`(tuple)로 success 판정 → 값만 있으면 인식 불가 → asc=False(우리 6런 전부 0/2, 0/1). **released는 content `'(True, 93.0)'` 보존 → asc 44/86·22/42 = solvable, defect 아님, 천장 34 유지.**

**수정안**: goal-call 결과의 **full tuple 보존**(success-bool 안 버림).
- **선행 확인(필수)**: 이 strip이 `apply_two_stage_patch.py`(우리 패치) 산물인지 원본 SOPBench core.py인지 확인 → 우리면 patch에서 해당 줄 제외(goal 한정), 원본이면 released 정합 위해 동일 처리.
- **전 도구 tuple 보존**(released 정합, 권장) — evaluator는 모든 func_call content를 봄.
- **provenance (리뷰 로컬검증)**: strip은 `apply_two_stage_patch.py`에 **없음**(identification dict 패치만) → **클론 `swarm/core.py` 자체**. 어느 버전이든 tuple 보존이 released 정합. (확인: `git log -p swarm/core.py`로 upstream vs 클론-로컬.)
- **일반성**: 완전 일반(도메인 무관·모든 tuple-return goal).
- **회귀 위험**: 低 — content 표현만 변경(model이 "(True,93.0)" 봄, released와 동일). flag `SOPBENCH_KEEPTUPLE` 뒤 A/B.
- **⚠️BLOCKING — 전 48 재census**: strip이 asc를 **ALL 태스크에서** 억눌렀다면 현 baseline 23이 **undercount**일 수 있음(현 23은 strip에도 asc 인식된 케이스). 고친 뒤 **23 불변 ∧ 정확히 +3**인지 확인 — 아니면 census 전체 이동. **이게 Cause 2를 먼저 두는 진짜 이유 = 측정축 먼저 고정.**
- **예상**: exchange 2 + get_account_owed 1 = **+3** (dirgraph 이미 True라 asc만 켜지면 BOTH) — 단 재census로 검증.

## ✅ 진행 현황 (2026-06-05 PM)
- **Cause 2 = DONE**: KEEPTUPLE → BOTH 23→26 (+3 isolated, 회귀 0, BLOCKING 재census 통과). 누적 **26/34 (76%)**, 잔여 8.
- **Guard-4 (creds 가용성) PASSED**: fixable 8(premature 7 + transfer DENY 1) **전부 creds-OK**(login id 항상·admin_password는 set_safety_box 3+transfer 2에 존재) → **+8 잠재 = 천장 34 도달가능**.
- **Cause 3 재진단 (folded into Cause 1)**: transfer DENY는 "drive-all"이 아니라 **false-leaf stall + step-cap**(offload trace: turn3 admin auth 구동 후 turn4-9 reason="false"로 active-H3가 구동 못 함→모델 루프, turn10 balance 도달했으나 cap). false leaf = establishing action(admin/login)이 게더됐으나 False. ⇒ **Cause-1의 full-dirgraph establishment 재확립으로 흡수**(별도 drive-all 런 불필요).

## Cause 1 (정밀화) — gate = sampled policy ∪ establishing leaves (NOT policy superset)
**리뷰 반영 + Cause-3 흡수**: 게이트가 평가할 것 = **sampled `task["constraints"]`(정책축, 현행 유지) ∪ task가 요구하는 establishing/state leaves**(login_user→logged_in_user · authenticate_admin_password→authenticated_admin_password · balance-getter). establishing leaves는 `get_default_dep_full[goal]`에서 **establishable/state-pred만 필터**(정책조건 credit_score 등은 제외 = dep_full 정책 superset over-deny 회피). active-H3가 그 establishing을 user_known creds로 구동(false-stall 해소: false establishable이면 올바른 creds로 재확립).
- **⚠️Guard-2 (BLOCKING, 미완)**: establishing-filter가 evaluator dirgraph와 leaf-동일한지 ≥3 task(현-BOTH 포함) 단위검증 — 구현 전 필수. evaluator dirgraph_satisfied 정확 메커니즘(graph traversal order-check) 확인 후 mirror.
- Option A(dep+constraints서 재구성), B(directed_action_graph read=oracle) 금지.

## Cause 3 (구舊) — active-drive 완전성 (transfer 1) — Cause 1에 흡수됨
**진단**: transfer ungathered `sufficient_account_balance`(getter get_account_balance). active-H3는 turn당 1 evidence만 구동(`two_stage_client.py` active block) → step cap(10) 전 모든 누락 evidence 미도달.
**수정안**: active-H3가 한 turn에 **다중 evidence 구동**(또는 큐 소진까지). **⚠️dirgraph-required leaf로 한정**(게이트 deny 분해의 `ungathered`/`argmismatch`만) → **over-call 병리 방지**(리뷰 caveat). 현 설계(게이트 산출 큐만 구동)면 안전.
- **일반성**: 일반(driving 완전성).
- **회귀 위험**: 低. flag `SOPBENCH_DRIVEALL` A/B. **예상**: transfer 1 (+ 다른 step-cap 한계 태스크 일부).

---

## 구현·검증 순서 (각 flag 독립 A/B, 회귀 감시)
1. **Cause 2** (가장 싸고 확실, +3 예상, 회귀 低): `SOPBENCH_KEEPTUPLE`. strip 출처 확인 → 수정 → A/B vs 23.
2. **Cause 3** (싸고 低위험): `SOPBENCH_DRIVEALL`. A/B.
3. **Cause 1** (가장 큰 변경·中위험·최대 잠재 +7): `SOPBENCH_DGGATE`. 단위검증(dfsins=evaluator) 후 A/B, BOTH 비회귀 필수.
- 각 단계 task_sig per-task 전이 추적 + 회귀 0 확인 후에만 박제. 누적 목표 BOTH→34.
- **메타규칙 준수**: 강한 주장(각 fix 효과·천장)은 A/B reliable test 후 박제. dead-end 변종 금지. should_T/F·identity 코드 강제.

## 4 BLOCKING 가드 (리뷰 확정 — 전부 통과해야 박제)
1. **Cause 2 후 전 48 재census**: 23 불변 ∧ 정확히 +3 확인(isolated 아닐 수 있음 → census 전체 이동 감시). 측정축 먼저 고정.
2. **Cause 1 단위검증 ≥3 task(현-BOTH 포함)**: dfsins=evaluator leaf-동일, 현-BOTH over-deny 0. = 회귀의 유일 관문(개념상 회귀불가, dfsins drift만 위험).
3. **Cause 1 Option A만**: B는 oracle 누출(per-task 정답 그래프 read)→배포 부정당. A=배포-현실적.
4. **Cause 1 7개 creds 가용성**: per-task login/admin_password ∈ user_known 확인(없으면 +0 + 환각병리).
- Cause 2 strip 출처 확인(`git log -p swarm/core.py`); 천장 34 불변(Cause 2가 defect 아님 확정).

## 일반화 ToDo 연계 (별도, `HANDOFF_2026_06_05 §6.5`)
본 3 fix는 모두 하버스 offload(A축)·도메인-일반 규칙. weight-내재화(B축 DPO/RFT)·param_mapping inducer-소싱·cross-domain A/B는 §6.5 ToDo로 분리(다음 실험).
