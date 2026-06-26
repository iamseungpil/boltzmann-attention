# HANDOFF 2026-06-05 — H3 decision-offload LIVE + 논문-근거 정책 결론 (다음 세션 진입점)

> **진입점 = 이 문서 + `RUNG1_REDESIGN_2026_06_04.md` §10.3** (전체 설계·전수 census·논문 인용). 권위본 결과 = `reports/facet_rft_2026/SOPBENCH_EXPERIMENT_RESULTS.md`(★★Gate-A / passive-H3 / active-H3 행). HEAD push=4d4764b(branch facet-rft-2026).

---

## §0 첫 행동 (다음 세션)
1. `cd $REPO && git pull` (리모트 behind 잦음).
2. ⚠️**§3 처방 정정(2026-06-05 리뷰): `check_permitted`→`dep_full`은 RETRACTED**(dep_full=태스크-무관 superset→비-sample 정책조건 over-deny=active-H3 15 재붕괴; evaluator는 goal dep를 `task["constraints"]`로 override "match evaluator"=권위). **게이트=task constraints 유지; ★진짜 레버=gather완성(active-H3→15)+credential**(login 유무 아님). `_plan_v2:505 [:600]` truncation 수정(full policy 노출)만 유효. login=dirgraph-required지만 BOTH-레버 아님(T1T2 login-uniform BOTH4/4 실패·특별취급 금지).
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

**핵심: offload는 결정축을 풀었다(ACT 3→19). BOTH가 낮은 건 *결정*이 아니라 *게더*(모델이 downstream getter[balance]·login 미게더). 레버=gather완성(active-H3→15)+credential, login *강제* 아님(T1T2서 실패).**

---

## §2 SETTLED — 재유도/재census 금지 (논문·정책·저자궤적 정독으로 확정)
1. **BOTH=6 진짜 뿌리 = 정책 vs dirgraph** (전수 census, 실제 evaluator):
   - **정책(verbalize) = per-task `constraints`** (sampled subset).
   - **login은 ⚠️innate 아님**(bank `action_innate_dependencies` 0/28, code-verified 2026-06-05 B)·`logged_in_user` 술어로 `get_default_dep_full`(required deps, account 액션 = `AND(internal_check_username_exist, logged_in_user, …)`)에 인코딩 → **태스크-특정**(그 태스크 그래프가 login-gated read를 거칠 때만 필요, leave-one-out 12/17 · "항상/보편" 아님).
   - 우리 7B는 정책(constraints)을 따라 login-gated downstream(balance read 등) 생략 → dirgraph 실패. 빅모델(Claude)은 방어적으로 **거의 항상 login(35/35), set_safety_box 성공 8/10**(자격증명 user_known 가용) — 단 이는 빅모델 *전략*이지 벤치가 보편 강제해서가 아님.
2. ⚠️**(2026-06-05 정정 — 이 항목 "SETTLED" 격하, 철회)**: paper-fetch 2개 상충(1차 "full SOP/not sampled" vs 재-fetch "task-specific/permuting constraint subsets") → **paper-요약 권위 불가, 권위=코드(evaluator)**. evaluator의 constraint축은 goal dep를 `task["constraints"]`로 override("match evaluator" `build:219`·`check_permitted:666`)=**task-specific, dep_full(태스크-무관 superset) 아님**. ⇒ "full SOP/항상 login 강제"·"dirgraph login=의도된 보편 선행조건" **철회**. login은 dirgraph-required(leave-one-out 12/17 사실)이나 강제는 BOTH-레버 아님(T1T2 BOTH4/4·특별취급 금지). success=그 태스크의 directed action graph(task constraints+cascade), 모든 도메인 제약 아님.
3. **internal_get_database = 결함 아니라 OR-대안** (`bank_assistant.py:372` `OR(get_account_balance, internal_get_database)`; 미노출 시스템 도구). **에이전트는 노출 `get_account_balance`로 충족**(set_safety_box dirgraph 10/10 exposed getter). 진짜 게더 타깃=get_account_balance. (§8.1/§10.2의 "internal_get_database 필요" 프레임 폐기.)
4. **정직분모 = 34/48** (Part A 8 credit_card + Part B 6 login∧cred-불가용). **Part B 6만 진짜 defect**(login필수 ∧ credential 불가용 교집합). 나머지 solvable. ([[reference-sopbench-bench-defects-settled]])
5. **우리는 정책조차 안 읽음**: `_plan_v2:505` `policy=...[:600]` → 전체 10,578자 중 일반 서문 600자만, per-action SOP 전부 잘림.

---

## §3 다음 처방 (정정 2026-06-05) — dep_full 폐기, 레버=gather+credential
⚠️**"check_permitted→dep_full(full SOP)"는 RETRACTED.** dep_full=`get_default_dep_full`=태스크-무관 전체 default dep superset → 그 태스크가 sample 안 한 정책조건까지 요구 = **over-deny → active-H3 15 재붕괴**. evaluator의 constraint축은 goal dep를 `task["constraints"]`로 override("match evaluator" `build:219`·`check_permitted:666`)=권위. paper-fetch 2개 상충→코드(evaluator)가 권위.
1. **게이트 = `task["constraints"]` 유지**(현행). dep_full로 바꾸지 말 것.
2. **정책 truncation 수정**(`_plan_v2:505 [:600]`→full policy): 모델 *입력*은 full(success-요구와 무관하게 유효).
3. **★진짜 레버 = gather 완성(active-H3 gate-driven → BOTH 15) + credential 가용**(login=False ≤9). login *유무* 아니라 creds·downstream getter(balance)가 binding. → 다음 = active-H3 라이브로 상승폭 확정 + credential-augment 잔여.
4. **대조 유지**: T1c-emit(1)/C-none(3) vs passive(6)/active-H3(15).
5. **login**: dirgraph-required(사실)이나 강제 안 함 — 실패(T1T2 login-uniform BOTH4/4)·금지(특별취급=전이파괴)·moot(모델이 creds 가용 시 자연 login). 모델 gather(active-H3)가 충족.
6. **dep object 정체 (B-체크 2026-06-05, code-verified — OPEN 해소):** evaluator는 goal을 `task["constraints"]`(sampled)로 채점(`evaluator.py:110`) + 호출별 dirgraph로 cascade 적용(`dfsins_cl_cd_aid`, `:267`). login = `logged_in_user` 술어로 `get_default_dep_full`(=ard+acd+cd; `task.py:88`)에 있음 — **`action_innate_dependencies`(=`dep_innate`)엔 0/28**. `get_default_dep_full`은 innate 제외 + 정책 superset → **dep_full 게이트는 login도 안 더하고 policy over-require만 함**(§3 폐기의 2번째 이유). 우리 게이트 `check_permitted:666` `task_dep[goal]=cons`는 `evaluator.py:110`과 동일=이미 mirror. **게이트가 active-H3로 *구동*해야 할 gather-target(login_user+노출 getter)은 evaluator dirgraph(dep_full+dfsins)에서 끌어옴 — accept 기준은 task constraints 유지**(둘은 별개: target=무엇을 게더, accept=무엇을 통과).

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
1. ⚠️**§3 dep_full 게이트 = RETRACTED**(over-deny). 다음 1순위 = **active-H3 라이브로 gather-driver 상승폭 확정**(BOTH 15가 현 천장) + credential-augment 잔여(≤9). 게이트는 task constraints 유지.
2. **non-alias gather-skip 체크 (fold-in, zero-cost)**: §10.1 "게더도 SFT-positive 저항(model-skip)" 통합 finding의 alias caveat 해소(alias-transfer vs 진짜 SFT-limit). **이거 풀기 전 "둘 다 RFT 필요" 박지 말 것.**
3. **decision-axis A/B/B' (§9)**: C(자기-emit·환각)/A(결정론 offload=상한)/B(verifier-교정 DPO·RFT=충실성 내재화)/B'(grounded-copy) 비교 = 소형모델 환각제거 논문 축.
4. **gather-타겟팅(model-skip)**: 모델이 get_account_balance·login을 자율 게더하게 = SFT-positive로는 안 됨(teacher 시범하는데 스킵) → active-H3(gate-driven, 무재학습) 또는 RFT. axis-A 주장하려면 후자.

---

## §6.5 다음-실험 ToDo — ARGFIX 규칙 R의 완전 일반화 (도메인 특화 제거)
**배경**: ARGFIX(env `SOPBENCH_ARGFIX`, push 62b5c6c)가 BOTH 15→21(+6, 회귀0) 달성. 코드는 이미 도메인-일반(bank 분기 無)이나 "일반"은 bank LODO 1도메인서만 검증·하버스 offload(A축). 일반 규칙:
> **R: 도구 인자는 충족하려는 dependency leaf의 `param_mapping`을 요청 파라미터(user_known)에 바인딩한다 — 이름 같다고 단일 default 슬롯에 묶지 않는다.**

**ToDo (다음 세션, 우선순위순):**
1. **wart 제거 → R 완전 일반화 (무재학습)**: 기존 active-H3의 `"internal_get_database" in tool_names` **리터럴 제거** → 일반 "dirgraph-required 미게더 evidence leaf 구동". (ARGFIX 아님, 기존 active 코드 `two_stage_client.py` ~566.)
2. **param_mapping 출처를 벤치→inducer로**: 게이트가 지금 `task["constraints"]`(벤치 ground-truth) 소비 → 배포-일반이려면 `induce_ontology_zekun` 출력(NL정책→구조) 소비로. inducer가 param_mapping 정확 생산하는지 검증.
3. **cross-domain A/B (R 일반성 reliable test)**: 동일 어댑터+ARGFIX를 다른 도메인(online_market/healthcare 등 multi-slot 연산 보유)서 A/B → delta 재현되면 "R=도메인-일반 arg-binding offload" 박제. 무재학습 ~30분 eval.
4. **B축 내재화 (thesis 본선, weight-전이)**: R을 모델이 체득하게 **DPO/RFT, positive=게이트 fp(R 정답 인자)·negative=오바인딩(source-only/값오염)**. 신호가 param_mapping 기반=도메인무관 → held-out 도메인 weight-전이. = 메모리 A(offload상한)↔B(내재화) 축의 B.
**주의**: 1·2·3은 무재학습(빠름), 4는 학습(4h). 강한 주장("R 일반")은 3 통과 후 박제. ARGFIX=A축 offload이지 weight-내재화 아님(혼동 금지).

## §7 reliability 교훈 (이번 세션 4함정, 전부 자가/사용자 교정)
①observed-called를 available-tool로 착각 ×2 ②string-search aliasing 아티팩트(0/4189) ③"internal_get_database 필요=bench defect" 점프(빅모델 풂이 반증) ④"정책=dirgraph" 가정(실제: 정책=constraints, dirgraph가 over-require) ⑤**(2026-06-05 PM) paper fast-fetch 요약 과신** — 한 fetch="full SOP/항상 login 강제", 재-fetch="task-specific/sampled" 상충 → 양쪽 다 추론 박지 말 것. **권위=벤치 소스코드**(`evaluator.py:110/267`·`variables.py action_innate_dependencies`)가 tiebreaker: login은 innate 아님·`logged_in_user` required-precond·태스크-특정. **전부 코드·정책 원문·저자 궤적을 *직접 읽어* 정정.** ⇒ **벤치 의미론(정책/success 기준)은 induced 구조·paper 요약 단독 말고 소스코드·정책 원문·저자 궤적이 권위.** 강한 의미론 주장은 (원문 직접인용 ∧ 빅모델 실측 ∧ **소스코드**) 교차 후 박제. [[feedback-check-authority-before-rederive]].
