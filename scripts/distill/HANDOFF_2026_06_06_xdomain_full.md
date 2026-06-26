# HANDOFF 2026-06-06 — cross-domain transfer (held-out) 전수 학습+eval 진행 중
> 📌 **구조 안내**: 모든 설계·실험 문서의 단일 마스터 = repo `scripts/distill/EXPERIMENT_DESIGN.md` (**§7 문서지도**에서 각 문서의 역할·상태 확인; 목표·순서 변경은 마스터 §0-§4에서만). 처음 읽는다면 마스터부터.

> **진입점 = 이 문서.** 환경 = [[reference-remote-server-environment]] (메모리, 세션 시작 필독). 설계 = `CROSS_DOMAIN_TRANSFER_DESIGN.md`(§2.0 held-out 교정). 마스터 = `EXPERIMENT_DESIGN.md`. 결과 권위본 = `reports/facet_rft_2026/SOPBENCH_EXPERIMENT_RESULTS.md`.

## §0 첫 행동 (다음 세션)
1. `cd /c/workspace` (ssh_run.py 위치). `cd /c/workspace/ba-frft && git pull`. 리모트 `cd $REPO && git pull`.
2. **상태 점검** (실 프로세스 = `nvidia-smi --query-compute-apps`; pgrep self-match 주의): 학습 큐(`xdomain_train_queue.log` `TRAIN_QUEUE_DONE`?) + train1_bank(`finish_train1bank.log` `FINISH_TRAIN1BANK_DONE`?).
3. **어댑터 14개 READY 확인**: `qwen7b_tbox_t1c_{lodo_,train1_}{bank,dmv,healthcare,hotel,library,online_market,university}` (lodo 7 + train1 7).
4. **학습 완료면 → eval 배치**: `setsid bash $SB/xdomain_eval_heldout.sh </dev/null >/dev/null 2>&1 &` (GPU0 serve, eval_tasks; 이미 done인 (tag,domain)은 skip). 완료 후 `diag_heldout_summary.py`(+ train1 전체 스캔하도록 확장 필요)로 전체 transfer 매트릭스.

## §1 ★확정 결과 (held-out transfer, 재학습0·honest·quirk-free)
| held-out | STACK 공식success | 리더보드-max | 판정 |
|---|---|---|---|
| bank (LODO) | **43.3%**(58/134) | o4-mini 76.9% | 동률권(bank 난도↑·PartA8) |
| **library (LODO)** | **75.8%**(50/66) | GPT-5 66.7% | **추월** |
| **healthcare (LODO)** | **95.2%**(118/124, sT 44/44) | o4-mini 92.7% | **추월** |
- **scaffold가 전부 들어올림**(adapter-only ~0% → stack 75~95%), 어댑터가 안 본 도메인에서, **재학습0(ABox-swap만)·quirk0**. transfer gap≈0(held-out≈in-domain). = A축 "도메인-일반 scaffold + 재학습0 전이" 강한 직접 증거.

## §1.5 ★train-1 확정 (2026-06-06 10:05 KST 완료) — bank 한 도메인만 학습 → 6 held-out 전이
| held-out | STACK success | should_T | LB-max | vs LB |
|---|---|---|---|---|
| dmv | 71.1%(69/97) | 35/36 | 86.7 | below |
| healthcare | 64.5%(80/124) | 44/44 | 92.7 | below |
| **hotel** | **83.6%(163/195)** | 58/67 | 69.7 | **추월** |
| **library** | **71.4%(40/56)** | 14/21 | 66.7 | **추월** |
| online_market | 73.8%(127/172) | 53/59 | 89.5 | below |
| **university** | **97.6%(41/42)** | 6/6 | 95.2 | **추월** |
| **avg** | **77.3%** | | | **3/6 LB-MAX 추월** |
- 단일도메인(bank)만 학습·재학습0·honest(LOGINCALL off). should_T 거의 천장 → 낮은 도메인 should_F-bound. 결과 박제 = `SOPBENCH_EXPERIMENT_RESULTS.md` Exp-5a, 설계 §11.1. 문서 4종(results·EXPERIMENT_DESIGN§2·CROSS_DOMAIN§11·COWORKER v1.43) 업데이트 완료.

## §2 진행 중 (2026-06-06 10:05 KST 갱신)
- **train1_bank eval = ✅DONE** (`FINISH_TRAIN1BANK_DONE`, §1.5 표). GPU1 vllm 정리됨.
- **학습 큐 `xdomain_train_queue.sh`** (GPU-aware 2병렬): LODO 4(dmv·hotel·online_market·university) + train-1 6(dmv·healthcare·hotel·library·online_market·university) = **10 어댑터**. 현재 **2병렬 가동**: lodo_dmv(GPU1) + lodo_hotel(GPU0, train1_bank eval 종료 후 기동). 나머지 8 큐 대기.
- READY 어댑터: lodo_{bank,dmv(학습중),library,healthcare}, train1_bank.

## §3 다음 단계 (순서·충돌 회피)
1. 학습 큐 + train1_bank 완료 대기.
2. **`xdomain_eval_heldout.sh`** 1회 (10 어댑터 held-out eval; LODO→stack+adapteronly, train1→stack on 6).
3. 전체 매트릭스 집계 → **LODO-7**(도메인별 held-out transfer) + **train-1 7×6**(1도메인 학습→타 6 전이, train-diversity 효과). `diag_heldout_summary.py`를 train1 전 어댑터 스캔하도록 확장.
4. 분석: ① LODO transfer가 다도메인서 리더보드 추월 재현되나 ② train-1(저자원) vs LODO(혼합) 격차 = 학습다양성 효과.

## §4 메서드/지표 (엄수)
- **지표 = 공식 success(`evaluator.py:277`, 134/도메인, tool_full)**. BOTH(dg∧acc) 헤드라인 금지(success 과대계상). honest = should_T quirk 제외(`diag_quirk_rescore.py`, should_T/F 분리·OR-분기 비-필수). **LOGINCALL 드롭**(quirk).
- **transfer = held-out only**: lodo_X는 X를 학습서 제외→X 테스트; train1_X는 X만 학습→나머지 6 테스트. (학습 도메인 테스트=in-domain=무효.)
- **eval = `eval_tasks.py`** (run_evaluation는 goal-stats ZeroDivisionError 크래시 → 저장전 사망, healthcare/hotel/online_market 가짜0%). sim 정상이니 사후 eval_tasks로 복구.
- **login 일반화**(이번 세션): LOGINFIRST가 credential arg를 `action_parameters[login_user]−{username}`서 derive(하드코딩 identification 제거)→library/market/university `password` 자동. dummy-login(LOGINCALL quirk) 폐기=사용자 통찰.

## §5 7B 헤드라인 (bank, honest)
공식 success **43.28%(LOGINCALL off live)** ≈ 오픈소스 SOTA(Llama70B 42.54%) 동률. should_T 강(honest 32/48)·should_F 약(25/86)=주 레버. (50.75%는 quirk 포함 폐기.)

## §6 coworker (이승필, 4×A100)
`COWORKER_EXPERIMENT_PLAN.md` v1.42: **#0 leaderboard 재현 sanity**(32B vanilla→README 40.30% 확인) → **#1 32B+scaffold**(4열, 공식success). 가정 32B>7B(should_F base 강). 이 LODO/train-1 학습들도 coworker서 병렬 가능.

## 메타
- 환경 시행착오 금지 = [[reference-remote-server-environment]] (git전송·ssh_run·pgrep self-match·연결불안정·eval_tasks·GPU충돌).
- 강한주장 reliable test 후 박제 [[feedback-check-authority-before-rederive]]. quirk지표 규율 [[feedback-crossmodel-strict-metric-discipline]].
