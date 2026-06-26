# HANDOFF 2026-06-11 — TaskBench 측정 2일차 종료 (census-기제 확립 → 처방 단계 진입점)
> 📌 **구조 안내**: 모든 설계·실험 문서의 단일 마스터 = repo `scripts/distill/EXPERIMENT_DESIGN.md` (**§7 문서지도**에서 각 문서의 역할·상태 확인; 목표·순서 변경은 마스터 §0-§4에서만). 처음 읽는다면 마스터부터.

> **다음 세션 진입점.** 권위: 결과 = `reports/facet_rft_2026/TASKBENCH_EXPERIMENT_RESULTS.md` **§8(3축 기제)·§9(실행 큐·분기 사전등록)·§9.5(snap)·§9.6(DPO)**. thesis = `FIELD_GAP_LLM_VALUE_DESIGN.md` §17.9(동결)·§18. 구 핸드오프 `HANDOFF_2026_06_10_taskbench_learning.md`(인프라 §2·converter §5는 여전히 유효 참조).
> 리모트 규칙은 memory `reference-remote-server-environment` 그대로 (**ssh_run은 반드시 `/c/workspace`에서** — cd 후 호출이 이번 세션 최다 반복 실수).

## 0. ★첫 행동 (순서대로)
1. **DPO 학습 확인** (세션 종료 시점 GPU1에서 진행 중, ~16시 완료 예상):
   `py -3 ssh_run.py --timeout 45 --cmd 'grep -E "ep[01] step|\[done\]" /home/woori/scratch/tb_dpo_train6.log | tail -3; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader'`
   - `[done] DPO adapter` 있으면 → 2로. 죽었으면 로그 전체 확인(아래 §4 DPO 4-fix 참조).
2. **DPO 평가 투입** (어댑터 `sft_runs/qwen7b_tb_dpo_mm`, TAG=tb_dpo_mm):
   `cd /home/woori/workspace_common/boltzmann-attention-pi && setsid bash scripts/distill/taskbench/tb_eval_adapter.sh dpo_mm data_multimedia "data_huggingface data_dailylifeapis" 0 0.85 </dev/null >/dev/null 2>&1 &`
   (로그 `/home/woori/scratch/tb_evaladapter_dpo_mm.log`, sentinel `ADAPTER_EVAL_DONE_dpo_mm`, ~75분)
3. **판정 (L2 표적 지표 = 조기종결)**: rft2 대비 ①node-deficit(+0.23)·short율(1028/5546)이 줄었나 — 본 세션의 P/R-결손 inline census(결과문서 §8 보강 참조) 재사용 ②edge/held-out — `tb_census.py --dir_a ..._evalfull_tb_rft2_mm --dir_b ..._evalfull_tb_dpo_mm`. **+ snap 겹쳐서**(`tb_name_snap.py`) 패키지 수치도.
4. 판정 후 분기는 §2 큐.

## 1. 오늘(06-11) 확정 결과 (전부 push됨, 상세는 결과문서)
- **★3축 기제 (§8, 전수조사·해석 권위)**: held-out 효과 = (+)참조-인덱싱 규율 전이(이득∝base 자기참조율) ⊕ (−)도구-어휘 간섭(−4~−8pp) ⊕ (0)누락-불변(결손 +0.23·short 21%, SFT/RFT/스케일 무반응). ~~형식-간섭~~·~~14B=용량~~·~~전이0~~ 철회/정정.
- **전이-vs-용량 U-커브 (§5)**: 1.5B +8.7 / 7B −0.8 / 14B +4.3 (sub500 동일-id).
- **RFT r1/r2 (§6)**: in-domain 레버 확정 — daily 75.9→85.2(r1), HF 47.8→51.6(r2, 일부 관례-수렴 ⚠️분리보고). held-out은 **재추첨**(±450 거울상 trade)=보상-side 한계 확정. r1↔r2 rollout 보상통계 비교금지(정의 다름).
- **★grounded-copy v0 (§9.5)**: name-snap 후처리만으로 **RFT2+snap=52.5 = held-out 첫 base(50.0) 추월**. v0/v1 경계 실측: daily 미스냅 689건=의미적 패러프레이즈("install software"→software_management) → **v1=제약 디코딩 필요**.
- **L3 probe**: type-gap 탐지 47%/유일수리 1.7% → "탐지→flag/재샘플" 게이트로 채택(자동삽입 아님).
- **Qwen3 곡선 완성 (§7)**: 4B≈Qwen2.5-7B(2x 효율), 8B full이 7B를 edge에서 +7.6~+11.1 추월(세대이득=구조축), 곡선 모양 family-불변.
- **DPO 채굴**: 318쌍(`/home/woori/scratch/tb_rft/dpo_earlyclose.jsonl`; no_short 2528 = in-domain 샘플링에선 조기종결 희소 — 누락질량은 greedy/held-out 측).

## 2. 실행 큐 (DPO 판정 후 분기 — 결과문서 §9가 권위, 요약)
- **DPO가 결손/short를 줄였으면**: best-stack 확정(rft2+dpo+snap) → 전 도메인 held-out 재측정으로 패키지 헤드라인.
- **못 줄였으면**: 누락축은 L3 게이트(flag→재샘플)로 이관 — GPU0에서 "type-gap 검출 케이스만 K=4 재샘플" 실험(probe 코드 = `tb_typeclosure_probe.py` 확장).
- **다음 본명 = grounded-copy v1**(제약/guided decoding — daily 의미-변형 축): vllm guided_choice/outlines 조사 → 도구명 슬롯만 valid-set 제약. 성공 시 TaskBench가 propose+gate 패키지의 2번째 실증으로 격상.
- coworker P0a/P0b/P1 대기 (`COWORKER_REQUEST_TB_SCALE.md` **v3** — P1 step-0=32B base census 선행·Δ 사전예측).
- 그 외: alias arm(P3 위생)·RFT r3(수확체감으로 비권장)·E2 복귀(P1 집필 전, FIELD_GAP §18.3).

## 3. 어댑터·산출물 지도 (리모트)
- 어댑터: `$REPO/reports/facet_rft_2026/phase4_distill/sft_runs/` — `qwen7b_tb_{lodo_mm,lodo_hf,lodo_daily,rft_mm,rft2_mm,dpo_mm}`·`qwen14b_tb_lodo_mm_14b`·`qwen15b_tb_lodo_mm_15b`.
- 평가 dir: `$TB/{dom}_evalfull_<tag>`·`{dom}_sub500_eval_<tag>`·`{dom}_sub500x_eval_<tag>`(동일-id 정합)·`{dom}_snapeval_<tag>`. census 보고서: `/home/woori/scratch/census_*.md`, probe `/home/woori/scratch/probe_tc_*.md`, RFT `/home/woori/scratch/tb_rft/`.
- 스크립트(전부 repo `scripts/distill/taskbench/`): build_sft/build_eval/census/typeclosure_probe/name_snap/dpo_mine/rft_rollout/rft_round/train_lodo/eval_adapter/scale_curve(+_qwen3)/night_batch*.

## 4. 인프라 gotchas (오늘 추가분 — 재발견 금지)
1. **dpo_train.py OOM 4-fix (전부 적용·push됨)**: ①단일 base+이중 어댑터(policy="default" trainable/"ref" frozen, set_adapter 스위칭) ②lm_head는 completion-span에만 ③**`.train()` 필수**(transformers grad-ckpt는 training=True에서만 발동 — ref pass는 `.eval()`로 결정론 유지) ④grad-ckpt enable. 결과 47GB OOM→18.4GB. SOPBench B축 DPO에도 그대로 이득.
2. **vllm 잔여 정리**: tb_eval_adapter.sh는 종료 시 vllm을 안 죽임 — 다음 GPU 작업 전 `nvidia-smi --query-compute-apps`로 확인 후 vllm PID만 kill(트레이너 보호).
3. **hf download 락 경합**: 같은 모델을 두 프로세스가 받으면 한쪽이 .lock에 무한대기 — stale 프로세스 kill로 즉시 해소.
4. **ssh_run cwd 함정**(재강조): 모든 `py -3 ssh_run.py`는 `/c/workspace`에서. compound 명령에 `cd /c/workspace/ba-frft && git ...; py -3 ssh_run.py ...` 패턴 금지(이번 세션 6회 반복).
5. 폴러는 `run_in_background` ssh wait-loop(~70분 윈도우, 만료 시 갱신); PipeTimeout/EXIT-1은 실행 여부를 **검증**(원격 잡은 보통 살아 있음).

## 5. 메타 (오늘 규율 수확)
- **궤적 전수조사가 해석을 2회 뒤집음**(형식→어휘 / 용량→오류율) — "집계 후 즉시 census" 를 표준 단계로.
- 사전등록 판정 기준(ⓐⓑ)·보상-정의 변경 시 round 간 통계 비교금지·관례-수렴 분리보고 — 전부 결과문서에 박제됨.
