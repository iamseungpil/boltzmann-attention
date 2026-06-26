#!/bin/bash
# 프레임 천장 측정 (2026-06-14 사용자): frontier(Fable-5)로 만든 A2 GATE_SPEC를 결정론 검증기로,
# frontier 생성기(gpt-4.1)로 retail 계획 K=4 생성 → compliant pass^k 천장.
# 질문: "결정론 A2(검증기)+다양생성기" 프레임이 원하는 compliant-pass 천장에 닿나.
#   compliance는 gate가 model-agnostic 보장(F4b 실증 위반0) → 천장 = 생성기의 gate-하 pass^k.
# arms: gpt-4.1 + gate(Fable-5 A2) × num_trials=4 × num_tasks=40. user-sim=gpt-4.1(4-tuple 동일).
# Run: setsid bash driver_frontier_ceiling.sh </dev/null >/dev/null 2>&1 &
set -u
R=/home/woori/workspace_common/boltzmann-attention-pi
PY=/home/woori/venvs/seka_env/bin/python
S=/home/woori/scratch
exec > $S/frontier_ceiling.log 2>&1
set +x; source /home/woori/.openrouter_key; set -x   # 키 로그노출 방지(day5 gotcha)
cd $R && git pull --ff-only -q
cd /home/woori/scratch/tau2-bench
export PYTHONPATH=src:$R/scripts/distill/tau2

rm -rf data/simulations/retail_gpt41_gate_k4
# t2_run_gated가 t2_compliance(pass^k + 위반) eval-후크 자동 실행
$PY $R/scripts/distill/tau2/t2_run_gated.py --gate 1 --num_trials 4 --num_tasks 40 \
  --agent_llm "openrouter/openai/gpt-4.1" --user_llm "openrouter/openai/gpt-4.1" \
  --user_temp 0.0 --save_to retail_gpt41_gate_k4
echo "FRONTIER_CEILING_DONE $(date)"
