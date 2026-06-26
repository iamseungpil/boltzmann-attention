#!/bin/bash
# frontier F4b census (2026-06-14 — 추세리뷰 숙제②, GPU 불요·OpenRouter only)
# 질문: "그냥 frontier 쓰면 되잖아" 반론의 정면 검증 — frontier도 bench-pass는 높지만
#   compliant-pass가 깎이는가? (ODCV: frontier도 압력 하 30-50% 위반)
# arms (retail 114 tasks × trials=1, census-tier):
#   FA frontier(gpt-4.1) nogate — 사전등록: bench pass^1 0.55~0.80(리더보드대),
#      위반 sims > 0 (G1/G2 합산), FULL-clean < bench (compliance 비용 가시화)
#   FB frontier(gpt-4.1) + gate — 사전등록: 위반 0 (게이트는 모델-불문 = 이식성 증거),
#      pass^1 FA 동등 ±3pp (frontier는 복구 능숙 — deny 비용 미미 예측)
# 비교표 완성: {7B, frontier} × {nogate, gate} × {bench, FULL-clean} — assurance thesis 1면.
# user-sim/judge = gpt-4.1 (기존 행렬과 4-tuple 동일). compliance.json 자동(후크).
# Run: setsid bash driver_frontier_f4b.sh </dev/null >/dev/null 2>&1 &
set -u
R=/home/woori/workspace_common/boltzmann-attention-pi
PY=/home/woori/venvs/seka_env/bin/python
S=/home/woori/scratch
exec > $S/frontier_f4b.log 2>&1
set -x
cd $R && git pull --ff-only -q
source /home/woori/.openrouter_key
cd /home/woori/scratch/tau2-bench
export PYTHONPATH=src:$R/scripts/distill/tau2

rm -rf data/simulations/retail_gpt41_nogate
$PY $R/scripts/distill/tau2/t2_run_gated.py --gate 0 --num_trials 1 \
  --agent_llm "openrouter/openai/gpt-4.1" --user_llm "openrouter/openai/gpt-4.1" \
  --save_to retail_gpt41_nogate
echo "FA_DONE"

rm -rf data/simulations/retail_gpt41_gate
$PY $R/scripts/distill/tau2/t2_run_gated.py --gate 1 --num_trials 1 \
  --agent_llm "openrouter/openai/gpt-4.1" --user_llm "openrouter/openai/gpt-4.1" \
  --save_to retail_gpt41_gate
echo "FB_DONE"

echo "FRONTIER_F4B_DONE $(date)"
