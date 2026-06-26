# HANDOFF 2026-06-05 PM — ARGFIX→VALFIX→KEEPTUPLE→DGGATE 사다리 (BOTH 15→29/34, 다음 세션 진입점)
> 📌 **구조 안내**: 모든 설계·실험 문서의 단일 마스터 = repo `scripts/distill/EXPERIMENT_DESIGN.md` (**§7 문서지도**에서 각 문서의 역할·상태 확인; 목표·순서 변경은 마스터 §0-§4에서만). 처음 읽는다면 마스터부터.

> **진입점 = 이 문서.** 권위본 결과 = `reports/facet_rft_2026/SOPBENCH_EXPERIMENT_RESULTS.md`(active-H3 census~DGGATE 행). 설계 = `RESIDUAL11_FIX_DESIGN.md`(2-리뷰어 엔도스·4 BLOCKING 가드) + `GUARD2_DIRGRAPH_MIRROR_DESIGN.md`(PASS). 직전 진입점 `HANDOFF_2026_06_05_h3_offload_paper_grounded.md`는 SUPERSEDED(단 §6.5 일반화 ToDo·§7 reliability는 유효).

---

## §0 첫 행동 (다음 세션)
1. `cd $REPO && git pull` (리모트 behind 잦음).
2. **잔여 fixable ~5 zero-cost 진단**: set_safety_box 3·pay_loan 2·pay_bill 1 premature(goal-call `acc=True`인데 `constraint_not_violated=False`) — DGGATE(establishing-순서)로 안 풀림 = **다른 blocker**. 원인 = goal-call이 실제 정책조건 위반(잔액 임계/safety_box 값?). `eval_t1c_dggate` + `offload_log_dggate.jsonl`(task_sig) 조인으로 건별 cnv-위반 leaf 식별. **GPU 전 zero-cost.**
3. 진단 후 처방 설계 → 리뷰 → 구현(메타규칙: 강한주장 reliable test 후 박제, dead-end변종 금지, should_T/F·identity 코드강제).

---

## §1 헤드라인 — 사다리 (전부 무재학습·실제 evaluator·회귀 0·박제)
```
active-H3 15 → ARGFIX 21(+6) → VALFIX 23(+2) → KEEPTUPLE 26(+3) → DGGATE 29(+3)  /  정직천장 34 (85%)
```
- **천장 34** = 48 − Part A 8(credit_card 코드버그) − Part B 6(login∧cred-불가용). 고정([[reference-sopbench-bench-defects-settled]]). exchange/get_account_owed는 defect 아님(released asc 44/86·22/42, KEEPTUPLE로 확인).
- **census exhaustive**: BOTH 29 + 잔여 5 = 34 정확분할(숨은 잔여 없음).
- 모든 fix = **A축(결정론 scaffold) 천장-확립**, 모델 학습 아님. B축 내재화는 §5 ToDo.

---

## §2 4개 개입 (env flag·무엇을 풂·재현)
| flag | Δ | 원인→처방 | push |
|---|---|---|---|
| `SOPBENCH_ARGFIX` | +6 | arg/slot-binding: ①active-H3가 argmismatch leaf를 게이트 fp로 강제구동(transfer dual-username) ②`_resolve`가 required args를 user_known서 결정론충족(goal-call 값오염 차단) | 62b5c6c |
| `SOPBENCH_VALFIX` | +2 | value-restriction 조건(maximum_deposit_limit 등=amount vs 상수, DB 안읽음) getter_map missing→no_evidence_route 과잉deny → getter route 없고 params 있으면 직접 compute | 5abe3c0 |
| `SOPBENCH_KEEPTUPLE` | +3 | `swarm/core.py:167`이 tuple-반환 success-bool 폐기(`raw_result[1]`)→evaluator asc 인식불가(exchange/get_account_owed). full tuple 보존=공식측정 복원(released 정합) | 8a0d8c3 (apply_two_stage_patch #6) |
| `SOPBENCH_DGGATE` | +3 | gate를 sampled constraints가 아니라 **full task directed_action_graph**(Guard-2 재구성=`dfsgather_invfunccalldirgraph(constraints_original,...,opt=full)`==evaluator OVER0/UNDER0)로 → 모델이 login→balance→admin 순서 establish 후 ACT. active-H3가 미충족 prereq를 user_known creds로 deepest-first 구동 | d5c64fc, 81d1b73 |
- **`task_sig`**(`reset`서 md5(goal+constraints+user_known), push a5e8b4c): offload-log↔eval-JSON **identity 조인** 키. per-task flip 추적 필수.
- **전체 flag 스택**(DGGATE 드라이버): `SOPBENCH_ALIAS=1 GATE=1 SCRATCHPAD=1 SOURCE=1 PLAN_MAXTOK=1024 OFFLOAD=1 OFFLOAD_ACTIVE=1 ARGFIX=1 VALFIX=1 KEEPTUPLE=1 DGGATE=1 AUGMENT_CRED=1`.

---

## §3 잔여 5 (genuine fixable, 다음 타깃)
- **set_safety_box 3 · pay_loan 2 · pay_bill 1 premature**: `acc=True`(goal-call 정확)·`dirgraph_satisfied=False`·`cnv=False`. DGGATE(establishing-순서)로 미해결 → **establishing 순서 아닌 goal-call의 실제 정책조건 위반**이 원인 후보(잔액 임계/safety_box 값). 
- ⚠️ offline 검증(dfscheck)이 "premature=순서문제"로 본 게 **transfer엔 맞고 나머진 부분적**(정직 정정). 그래서 §0.2 zero-cost 진단으로 cnv-위반 leaf 정체부터.
- Part A 8(DENY)·Part B 6(premature 5+DENY 1)는 defect=수정 불가(천장 밖).

---

## §4 인프라 (실행 레시피·함정)
- **어댑터**: `$REPO/reports/facet_rft_2026/phase4_distill/sft_runs/qwen7b_tbox_t1c_lodo_bank`.
- **드라이버 패턴**(`offload_dggate.sh` 최신·표준): git pull REPO → **`cd $CL && git checkout -- swarm/ run_simulation.py`(전체 reset 필수!)** → `apply_two_stage_patch.py $CL` → CORE_PATCHED_OK·RESET_PATCHED_OK 마커 확인 → vllm serve(tau2_vllm_env, port 8351, GPU0) → SERVER_READY 폴링 → run_simulation → run_evaluation → kill.
- **★swarm 전체 reset 함정**: apply_two_stage_patch는 이미-패치 파일서 AssertionError로 **중단**(constants.py anchor). KEEPTUPLE(#6 core.py)·reset constraints_original 패치가 적용되려면 **`git checkout -- swarm/ run_simulation.py`로 먼저 upstream 복원** 후 fresh 적용. (구 드라이버는 run_simulation.py만 reset→core 패치 누락.)
- **fresh OUT/OFFLOG 필수**(run_simulation skip 회피): 재런마다 새 `--output_dir` + 새 OFFLOG.
- **★SSH 함정**: `setsid bash driver.sh </dev/null &`로 띄우면 SSH PipeTimeout(exit 255)이나 **드라이버는 detached 정상기동**. → 별도 rr.ps1 폴링(harness 알림無). 긴 폴링은 PowerShell run_in_background로(완료 알림 옴).
- **eval JSON**: `/home/woori/scratch/sft_alias_run/eval_t1c_<name>/bank/*full*shuffle_False.json`. BOTH = `dirgraph_satisfied ∧ action_successfully_called`(should_T만). offload-log = `offload_log_<name>.jsonl`(task_sig·decision·reason·ungathered/argmismatch/false).
- **seka python**: `/home/woori/venvs/seka_env/bin/python`(py3.12). 분석 PYTHONPATH=`/home/woori/scratch/SOPBench:$REPO/scripts/distill/sopbench`. py3.9는 match문 실패.
- **GPU**: GPU0 eval(vllm serve). GPU1 ollama(타인 21GB 상주, **kill 금지**). 런 끝 kill $SPID로 GPU0 반납.
- **census 스크립트 패턴**(권위·재사용): task_sig 조인 A/B + per-task 전이 + 회귀체크(BOTH→not). Part A=goal∈{cancel_credit_card,pay_bill_with_credit_card}, Part B=augment-invariant sig 매칭(`data/bank_tasks.json` dirgraph-needs-auth∧creds-absent). Guard-2 단위검증=`guard2_dirgraph_unitcheck.py`.

---

## §5 SETTLED (재유도 금지·인용만)
1. **정직천장 34** = 48−PartA8−PartB6. exchange/get_account_owed=defect 아님(KEEPTUPLE). 헤드라인=**BOTH/34** (BOTH/48 금지).
2. **잔여 지배원인 = arg/slot-binding**(ARGFIX가 +6)·tuple-측정버그(KEEPTUPLE +3)·establishing-순서(DGGATE +3, transfer). 잔여 5=goal-call cnv 위반(미진단).
3. **Guard-2 PASS** (`guard2_dirgraph_unitcheck.py`): A1 재구성=`dfsgather_invfunccalldirgraph(constraints_original, cl,cp, default_dep(full), ap, goal_node)` → 전48 OVER=0∧UNDER=0 exact=evaluator graph. **정책+도메인규칙이 cascade 완전결정·oracle 불요** 증명. directed_action_graph read=oracle(B)금지.
4. **evaluator dirgraph_satisfied** = task graph(generation서 constraints_original로 빌드) graph-traversal 선행체크(`dfscheck_called_functions`, nested=import불가→복제). constraints_original+full만 exact(constraints[expanded]=OVER44, required/none=UNDER48).
5. **DGGATE 회귀 불가**(개념·실측): 현-BOTH는 자기 cascade 충족→permit. Guard-2 OVER=0 + 실측 REGRESSION=0.

---

## §6 일반화 ToDo (별도, A축→B축·도메인일반·다음 실험)
직전 핸드오프 §6.5 유지. 본 4 fix는 코드-일반(param_mapping/predicate-kind/tool spec, bank분기 無)이나 **bank LODO 1도메인·하버스 offload(A축)** 검증뿐:
1. **wart 제거**: 기존 active-H3 `internal_get_database` 리터럴 → 일반 "dirgraph-required 미게더 leaf 구동".
2. **param_mapping을 inducer 출력서**(현 벤치 task["constraints_original"] 소비 → `induce_ontology_zekun` NL정책→구조 검증).
3. **cross-domain A/B**: 동일 어댑터+flag를 다른 도메인(online_market 등)서 → R 일반성 reliable test.
4. **★B축 내재화(thesis 본선)**: 게이트 fp positive로 **DPO/RFT** → 모델이 R(arg/slot-binding·establishing-순서) 체득 → weight-전이. ARGFIX/DGGATE는 A축 상한·scaffold이지 내재화 아님(혼동 금지). novelty=A↔B 전이.

---

## §7 reliability 교훈 (이번 세션)
- **확정 전 reliable test**: 모든 +Δ는 task_sig 조인 A/B + 회귀체크 후 박제. confound 2건(admin-19 should_F혼입·게이트 값평가) 철회. census /48 위반→/34 교정(리뷰어).
- **GPU 전 zero-cost 진단**: Guard-2 단위검증·offline dfscheck로 DGGATE 안전 사전증명 후 런(catastrophic over-deny 회피).
- **paper fast-fetch fallible** → 소스코드가 tiebreaker(KEEPTUPLE provenance·login-innate 정정). 강한 의미론주장=(원문∧빅모델∧소스코드) 교차후 박제.
- **offline 검증 한계**: dfscheck offline이 "premature=순서"로 본 게 transfer만 맞음(DGGATE +3≠+8) → offline 예측도 reliable run으로 확정.
- [[feedback-check-authority-before-rederive]] [[feedback_results_master_doc]].
