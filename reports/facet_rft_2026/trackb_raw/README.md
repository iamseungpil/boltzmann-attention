# trackb_raw — Track-B raw 궤적 (요청서 v3 §3.5 이행, 2026-06-11)

생성: H100×4 노드 2대 (`tb-h100-0611-eval`/`-train`, 80G4-H100 standard). serving = vllm 0.10.2 / bf16 / TP2 (235B·72B는 TP4), `--max-model-len 8192 --gpu-memory-utilization 0.90`, temp/top_p = inference.py 기본값. Qwen3 계열은 non-thinking 고정(`tb_patch_inference.py`의 chat_template_kwargs payload 패치 + serve-시 `<think>` 누출 프로브 통과). 전체 상태 스토어 = HF dataset `iamseungpil/sopbench-trackb-h200` (10분 sync 원본).

## 내용물
- `preds/<dir>/<tag>.json` — sub500 raw predictions (P0 4모델 ×3도메인 = eval 노드, `tb_lodo_mm_32b` ×3 = P1 SFT, train 노드).
  - ⚠️ `data_multimedia_sub500/qwen25_32b.json` (eval 노드, P0a)와 `qwen25_32b.p1census_trainnode.json` (train 노드, **P1 step-0 census/prereg의 입력**)은 같은 모델·같은 sub500의 독립 2-run — census 재검증은 후자를 쓸 것.
- `metrics/<evaldir>/<tag>.json` — 공식 evaluate.py 산출(tb_build_eval 경유) 전부 (P0 12런 + P1 sub500 3런 + MM full 어댑터/베이스 + census 입력 run).
- `p1_census_prereg.json` — **step-0 사전등록 동결본** (2026-06-11T09:38Z HF 커밋이 원본 박제; nself=0.0 → Δpred −5.0).
- `p1_census_32b_base.md` — step-0 census 원문 (base vs base).
- `p1_census_base_vs_sft.md` — 학습 후 base vs SFT census (valid_frac 1.000→0.952, n=498).

## §3.5 item 5 (inference 로그) 대체 요약
원시 `<tag>.log`는 노드 로컬(`/scratch/logs/`)이라 sync 미포함 — pred 행수로 드롭 집계 (500 − n = 영구 실패/드롭 id):

| dir | q25_32b | q3_32b | q25_72b | q3_235b_int4 | tb_lodo_mm_32b |
|---|---|---|---|---|---|
| HF_sub500 | 500 | 500 | **496** | 500 | 500 |
| MM_sub500 | 498(+499 train) | 499 | 499 | 500 | 500 |
| daily_sub500 | 500 | 500 | **497** | 497 | 500 |

(72B의 HF 4건·daily 3건 드롭이 최대 — reformat 소진 영구 실패. 나머지는 0–2건.)

## 비고
- P1 판정·기제 시그니처는 `TASKBENCH_EXPERIMENT_RESULTS.md` §8.5 (Δ실측 −5.4 vs 예측 −5.0).
- 72B/235B full-eval dir(`*_sub500_eval_*`)의 pred/data 사본은 용량상 제외 — preds/ + metrics/로 재구성 가능 (`tb_build_eval.py --pred_file`).
