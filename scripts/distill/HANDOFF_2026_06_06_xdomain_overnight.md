# HANDOFF 2026-06-06 — cross-domain transfer (held-out) 밤샘 학습+eval 진행 중

> 진입점 보강: `CROSS_DOMAIN_TRANSFER_DESIGN.md`(§2.0 held-out 교정) + `EXPERIMENT_DESIGN.md` §2.

## 첫 행동 (아침)
1. `cd $REPO && git pull`.
2. **밤샘 결과 확인**: `/home/woori/scratch/sft_alias_run/xdomain_overnight_eval.log`에 `OVERNIGHT_EVAL_DONE` 있으면 완료. held-out eval 출력 = `xho_<tag>_<domain>_{stack,adapteronly}`. 각 도메인 공식 success = `eval_tasks.py <json>` 또는 `diag_leaderboard.py`(경로 추가).
3. **transfer 표 작성**: held-out 도메인별 base(리더보드 Qwen7B) / adapter-only / stack / 리더보드-max. scaffold Δ = stack−adapter-only.

## 진행 상태 (취침 시점 ~06:00 KST)
- **학습(held-out 어댑터, t1c 레시피 LoRA r16 3ep)**:
  - ✅ `t1c_lodo_library` (train=7−library) — 완료(05:47).
  - 🔄 `t1c_lodo_healthcare` (train=7−healthcare) — GPU0.
  - 🔄 `t1c_train1_bank` (train=bank만) — GPU1.
  - 체인 = `xtrain_orchestrate.sh`(GPU 비는 대로 자동 launch).
- **밤샘 held-out eval** = `xdomain_overnight_eval.sh`: GPU0 free 대기 → 각 어댑터 serve(GPU0) → held-out 타깃서 adapter-only + stack eval(`eval_tasks.py`). 매핑: lodo_library→library, lodo_healthcare→healthcare, train1_bank→{dmv,healthcare,hotel,library,online_market,university}.

## ★핵심 교정 2건 (이 세션)
1. **타당성(사용자 지적)**: transfer는 **held-out 도메인에서만**. lodo_bank 어댑터로 학습한 6도메인 테스트 = in-domain(무효). ⇒ held-out 어댑터 재학습(위). 유일 기존 transfer 점 = **bank 43.28%(honest)**.
2. **bench 크래시 버그**: `run_evaluation.py` goal-statistics가 ZeroDivisionError(total_interactions=0)로 **저장 전 크래시** → healthcare/hotel/online_market evaluations 누락=0%(가짜). **`eval_tasks.py`(robust per-task eval, goal-stats 우회)로 해결.** 재평가 실값: healthcare 92.7%·hotel 90.8%·online_market 73.8%(전부 in-domain).

## in-domain 상한 참조 (transfer 아님, 학습 도메인)
bank 43.3%(held-out) · dmv 80.4 · healthcare 92.7 · hotel 90.8 · library 77.3 · online_market 73.8 · university 100. (login 일반화 작동: library/market/university `password` 구동 확인. scaffold Δ 거대: adapter-only 0~21% → stack 73~100%.)

## 헤드라인 (현재, honest)
- bank held-out 공식 success **43.28%(58/134)** = 오픈소스 SOTA(Llama70B 42.54%)와 동률. should_T 강(32 honest)·should_F 약(25/86)=주 레버. LOGINCALL(quirk) 드롭됨.
- 밤샘 결과로 **held-out transfer 점 3개 추가**(library·healthcare·train1) → A축 "scaffold ABox-swap 재학습0 전이" 다점 입증.

## 메타
- login 특별취급 제거(arg=도메인 시그니처 derive·dummy-login quirk 폐기, 사용자 통찰).
- quirk 측정 규율 [[feedback-crossmodel-strict-metric-discipline]]. 리모트=git만 [[feedback-remote-file-transfer-git]].
- coworker = 32B(`COWORKER_EXPERIMENT_PLAN.md` v1.42: #0 leaderboard 재현 sanity → #1 32B+scaffold). 이 LODO 학습들도 coworker 4×A100서 병렬 가능.
