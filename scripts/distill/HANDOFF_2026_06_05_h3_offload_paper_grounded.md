# HANDOFF 2026-06-05 — H3 decision-offload LIVE + 논문-근거 정책 결론 (다음 세션 진입점)

> **진입점 = 이 문서 + `RUNG1_REDESIGN_2026_06_04.md` §10.3** (전체 설계·전수 census·논문 인용). 권위본 결과 = `reports/facet_rft_2026/SOPBENCH_EXPERIMENT_RESULTS.md`(★★Gate-A / passive-H3 / active-H3 행). HEAD push=4d4764b(branch facet-rft-2026).

---

## §0 첫 행동 (다음 세션)
1. `cd $REPO && git pull` (리모트 behind 잦음).
2. **§3 처방 구현**: `check_permitted`를 `task["constraints"]`(sampled) → **`dep_full`(full SOP)** 로 + `_plan_v2:505` 정책 `[:600]` truncation 수정 → 재런 → **BOTH 천장 34 향함** 확인.
3. ⚠️ launch 전 §4 인프라 함정(SSH timeout=setsid detach, 폴링 분리) 숙지.

---

## §1 현 상태 / 헤드라인 (라이브 확증)
**H3 decision-offload = 모델 emit 게이트 대신 결정론 `check_permitted`(게더된 실제 도구결과로 ACT/STOP). 구현·라이브 검증 완료.**

| 런 | BOTH (honest/40) | ACT | 비고 |
|---|---|---|---|
| T1c-emit (대조) | **1** | ~3 | 모델이 게이트 emit → cold-bias 환각 (gathered_then_REFUSE) |
| C-none (대조) | 3 | — | |
| **passive-H3** | **6** | 19 | offload가 decision축 relieve(ACT 3→19 lift) 라이브 확증·누출 없음 |
| **active-H3** | **15** | 30 | 게이트가 누락 getter 직접 구동(무재학습) → BOTH 6→15 |

**핵심: offload는 결정축을 풀었다(ACT 3→19). BOTH가 낮은 건 *결정*이 아니라 *게더*(모델이 full SOP를 안 따름=login 안 함).**

---

## §2 SETTLED — 재유도/재census 금지 (논문·정책·저자궤적 정독으로 확정)
1. **BOTH=6 진짜 뿌리 = 정책 vs dirgraph** (전수 census, 실제 evaluator):
   - **정책(verbalize) = per-task `constraints`** (sampled subset). **dirgraph는 *항상* login_user 포함(⊋정책).**
   - 우리 7B는 정책(constraints)을 따라 login 생략 → dirgraph 실패. 빅모델(Claude)은 **항상 login(35/35), set_safety_box 성공 8/10**(자격증명 user_known 가용).
2. **★논문(SOPBench/AgentOrca, ar5iv 2503.08669) 권위**: 에이전트는 *full NL SOP* 받음 · *모든 선행단계(login/auth) 필수*(cascading) · success=*full directed action graph* · constraints=*complete operational dependency*(sampled 아님). ⇒ **"정책만(sampled constraints)"이 아니라 full SOP(항상 login)를 따라야 함이 논문 의도. dirgraph login=over-require 아니라 *의도된 선행조건*.**
3. **internal_get_database = 결함 아니라 OR-대안** (`bank_assistant.py:372` `OR(get_account_balance, internal_get_database)`; 미노출 시스템 도구). **에이전트는 노출 `get_account_balance`로 충족**(set_safety_box dirgraph 10/10 exposed getter). 진짜 게더 타깃=get_account_balance. (§8.1/§10.2의 "internal_get_database 필요" 프레임 폐기.)
4. **정직분모 = 34/48** (Part A 8 credit_card + Part B 6 login∧cred-불가용). **Part B 6만 진짜 defect**(login필수 ∧ credential 불가용 교집합). 나머지 solvable. ([[reference-sopbench-bench-defects-settled]])
5. **우리는 정책조차 안 읽음**: `_plan_v2:505` `policy=...[:600]` → 전체 10,578자 중 일반 서문 600자만, per-action SOP 전부 잘림.

---

## §3 다음 처방 (논문-근거 BOTH solver) — 구현 대상
1. **check_permitted 기준 = `dep_full`(full SOP, 항상 login·auth) — `task["constraints"]`(sampled) 폐기.** `two_stage_client.py:_check_permitted`의 `cons = self._task_constraints` → goal의 full dependency(operator precond / dep_full)로. (reset에 dep_full 또는 goal precond 전달, 혹은 abox operators[goal][precondition] 사용 — 이미 login+admin 포함.)
2. **정책 truncation 수정**: `_plan_v2:505` `[:600]` → full SOP 제시(또는 우리 induced precond rendering이 이미 login+admin 포함하는지 확인 후 충분하면 유지).
3. **재런** (passive + active 둘 다, dep_full 기준): BOTH가 6→천장 34로 가는지. 빅모델처럼 항상-login.
4. **대조 유지**: T1c-emit(1)/C-none(3) vs passive/active-H3.
5. ⚠️ 단 이는 "정책 충실"이 아니라 **"full SOP 충실"**(빅모델이 하는 것). paper-framing: active-H3=gate-driven(배포/LLM-Modulo), axis-A(모델 자율학습)와 구분.

---

## §4 인프라 (실행 레시피) — 함정 포함
- **어댑터**: `$REPO/reports/facet_rft_2026/phase4_distill/sft_runs/qwen7b_tbox_t1c_lodo_bank` (T1c, alias_s3 treeval slot-fixed).
- **드라이버 패턴** (`/home/woori/scratch/sft_alias_run/offload_active.sh` 참고): git pull → `apply_two_stage_patch.py $CL`(클론 재패치) → vllm serve(tau2_vllm_env, port 8351, GPU0, --enable-lora tbox_v2=$ADAPTER) → SERVER_READY 폴링 → run_simulation → SIM_DONE → run_evaluation → EVAL_DONE → kill.
- **env 토글**: `SOPBENCH_ALIAS=1 SOPBENCH_GATE=1 SOPBENCH_SCRATCHPAD=1 SOPBENCH_SOURCE=1 SOPBENCH_PLAN_MAXTOK=1024 SOPBENCH_OFFLOAD=1 [SOPBENCH_OFFLOAD_ACTIVE=1] SOPBENCH_AUGMENT_CRED=1 SOPBENCH_OFFLOAD_LOG=<path> SOPBENCH_VLLM_BASE_URL=http://localhost:8351/v1`.
- **fresh output dir 필수**: run_simulation이 기존 결과를 "완료"로 skip → 재런마다 새 `--output_dir`.
- **★SSH 함정**: rr.ps1로 `setsid bash driver.sh < /dev/null &` 띄우면 **SSH 채널이 잡혀 PipeTimeout(exit 255/timeout)** 나지만 **드라이버는 detached로 정상 기동**. → launch 후 **별도 rr.ps1 호출로 폴링**(pgrep/markers). harness 알림 없음.
- **eval JSON**: `/home/woori/scratch/sft_alias_run/eval_<name>/bank/*full*shuffle_False.json`. BOTH = `dirgraph_satisfied ∧ action_successfully_called` (should_T만). offload 결정 census = `SOPBENCH_OFFLOAD_LOG` jsonl(decision/reason/n_ungathered/n_argmismatch).
- **seka python**: `/home/woori/venvs/seka_env/bin/python` (py3.12); 분석 스크립트는 `PYTHONPATH=/home/woori/scratch/SOPBench:$REPO/scripts/distill/sopbench`. py3.9(기본 python3)는 match문 import 실패.

---

## §5 코드 상태 (구현됨·push)
- `two_stage_client.py`: **check_permitted**(벤치 `Dependency_Evaluator` subclass·evidence-gated bench-compute·5 정확성 잠금: 게더-only·벤치평가기재사용·모델goal호출·unknown→deny분해·credential-augment입력) + **active-H3**(env `SOPBENCH_OFFLOAD_ACTIVE`, 누락 getter 구동·loop guard). reset이 `task_db/constraint_params/domain/user_known` 받음(없으면 args_unresolvable 버그).
- `apply_two_stage_patch.py`: **Edit D credential-augment**(env `SOPBENCH_AUGMENT_CRED`, identification만 user_known 주입·누출 0 검증됨) + reset 호출에 task 필드 전달.
- ⚠️ **active-H3 part-B(internal_get_database 구동)는 미작동·삭제 대상**(미노출 도구). part-A(condition getter 구동)는 유효.

---

## §6 열린 항목
1. **§3 처방(dep_full 게이트)** — 다음 1순위.
2. **non-alias gather-skip 체크 (fold-in, zero-cost)**: §10.1 "게더도 SFT-positive 저항(model-skip)" 통합 finding의 alias caveat 해소(alias-transfer vs 진짜 SFT-limit). **이거 풀기 전 "둘 다 RFT 필요" 박지 말 것.**
3. **decision-axis A/B/B' (§9)**: C(자기-emit·환각)/A(결정론 offload=상한)/B(verifier-교정 DPO·RFT=충실성 내재화)/B'(grounded-copy) 비교 = 소형모델 환각제거 논문 축.
4. **gather-타겟팅(model-skip)**: 모델이 get_account_balance·login을 자율 게더하게 = SFT-positive로는 안 됨(teacher 시범하는데 스킵) → active-H3(gate-driven, 무재학습) 또는 RFT. axis-A 주장하려면 후자.

---

## §7 reliability 교훈 (이번 세션 4함정, 전부 자가/사용자 교정)
①observed-called를 available-tool로 착각 ×2 ②string-search aliasing 아티팩트(0/4189) ③"internal_get_database 필요=bench defect" 점프(빅모델 풂이 반증) ④"정책=dirgraph" 가정(실제: 정책=constraints, dirgraph가 over-require). **전부 코드·정책 원문·저자 궤적·논문을 *직접 읽어* 정정.** ⇒ **벤치 의미론(정책/success 기준)은 induced 구조 말고 논문·정책 원문·저자 궤적이 권위.** 강한 주장(헤드룸·결함·LOCK수정)은 reliable test(실제 evaluator) + 원문 후 박제. [[feedback-check-authority-before-rederive]].
