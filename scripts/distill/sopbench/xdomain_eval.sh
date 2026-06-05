#!/bin/bash
# Cross-domain T-A: per domain, ADAPTER-ONLY (scaffold off) vs STACK (full scaffold, LOGINCALL off,
# augment off). One server (SFT adapter). scaffold Δ = stack - adapter-only (A-axis claim, BLOCKING-2).
# base(raw 7B) per domain = cite leaderboard Qwen2.5-7B. Metric = official success (134-eq, tool_full).
# DOMAINS env overrides (default = smoke: library). Login-arg now domain-derived (cross-domain fix).
exec > /home/woori/scratch/sft_alias_run/xdomain_eval.log 2>&1
set -x
DOMAINS=${DOMAINS:-"library"}
REPO=/home/woori/workspace_common/boltzmann-attention-pi
CL=/home/woori/scratch/SOPBench
SEKA=/home/woori/venvs/seka_env/bin/python
VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
ADAPTER=$REPO/reports/facet_rft_2026/phase4_distill/sft_runs/qwen7b_tbox_t1c_lodo_bank
PORT=8351
cd $REPO && git pull --ff-only
cd $CL && git checkout -- swarm/ run_simulation.py 2>/dev/null || true
$SEKA $REPO/scripts/distill/sopbench/apply_two_stage_patch.py $CL
$SEKA -c "import py_compile; py_compile.compile('$CL/scripts/two_stage_client.py', doraise=True); print('PYCOMPILE_OK')" || { echo "PYCOMPILE_FAIL"; exit 1; }
cd $CL
CUDA_VISIBLE_DEVICES=0 nohup $VLLM serve Qwen/Qwen2.5-7B-Instruct --port $PORT --enable-lora --lora-modules tbox_v2=$ADAPTER --enable-auto-tool-choice --tool-call-parser hermes --max-model-len 8192 --gpu-memory-utilization 0.85 --dtype bfloat16 --trust-remote-code > /home/woori/scratch/sft_alias_run/serve_xdomain.log 2>&1 &
SPID=$!; echo "SERVE_PID=$SPID"
for i in $(seq 1 90); do
  if curl -s http://localhost:$PORT/v1/models 2>/dev/null | grep -q tbox_v2; then echo "SERVER_READY"; break; fi
  sleep 10
done
STACK="SOPBENCH_ALIAS=1 SOPBENCH_GATE=1 SOPBENCH_SCRATCHPAD=1 SOPBENCH_SOURCE=1 SOPBENCH_PLAN_MAXTOK=1024 SOPBENCH_OFFLOAD=1 SOPBENCH_OFFLOAD_ACTIVE=1 SOPBENCH_ARGFIX=1 SOPBENCH_VALFIX=1 SOPBENCH_KEEPTUPLE=1 SOPBENCH_DGGATE=1 SOPBENCH_LOGINFIRST=1 SOPBENCH_STOPSUCCESS=1"
for D in $DOMAINS; do
  echo "===== DOMAIN $D : ADAPTER-ONLY (scaffold off) ====="
  OUT_A=/home/woori/scratch/sft_alias_run/xdom_${D}_adapteronly
  env SOPBENCH_VLLM_BASE_URL=http://localhost:$PORT/v1 $SEKA run_simulation.py --domain $D --assistant_model tbox_v2 --tool_call_mode fc --tool_list full --two_stage --two_stage_v2 --ont_dir $CL/induced --output_dir $OUT_A --env_mode prompt
  $SEKA run_evaluation.py --domain $D --assistant_model tbox_v2 --tool_call_mode fc --tool_list full --output_dir $OUT_A
  echo "ADAPTERONLY_DONE_$D"
  echo "===== DOMAIN $D : STACK (full scaffold, LOGINCALL off) ====="
  OUT_S=/home/woori/scratch/sft_alias_run/xdom_${D}_stack
  OFFLOG=/home/woori/scratch/sft_alias_run/xdom_${D}_stack.jsonl; rm -f "$OFFLOG"
  env $STACK SOPBENCH_OFFLOAD_LOG=$OFFLOG SOPBENCH_VLLM_BASE_URL=http://localhost:$PORT/v1 $SEKA run_simulation.py --domain $D --assistant_model tbox_v2 --tool_call_mode fc --tool_list full --two_stage --two_stage_v2 --ont_dir $CL/induced --output_dir $OUT_S --env_mode prompt
  $SEKA run_evaluation.py --domain $D --assistant_model tbox_v2 --tool_call_mode fc --tool_list full --output_dir $OUT_S
  echo "STACK_DONE_$D"
done
kill $SPID 2>/dev/null
echo "DONE_DRIVER"
