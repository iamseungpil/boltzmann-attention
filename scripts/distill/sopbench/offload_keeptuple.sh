#!/bin/bash
# Cause-2: active-H3 + ARGFIX + VALFIX + KEEPTUPLE (preserve tuple-return success-bool).
# A/B vs argvalfix (BOTH 23). BLOCKING: re-census all 48 (23 unchanged + exactly +3).
exec > /home/woori/scratch/sft_alias_run/offload_keeptuple.log 2>&1
set -x
REPO=/home/woori/workspace_common/boltzmann-attention-pi
CL=/home/woori/scratch/SOPBench
SEKA=/home/woori/venvs/seka_env/bin/python
VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
ADAPTER=$REPO/reports/facet_rft_2026/phase4_distill/sft_runs/qwen7b_tbox_t1c_lodo_bank
PORT=8351
OUT=/home/woori/scratch/sft_alias_run/eval_t1c_keeptuple
OFFLOG=/home/woori/scratch/sft_alias_run/offload_log_keeptuple.jsonl
rm -f "$OFFLOG"
cd $REPO && git pull --ff-only
cd $CL && git checkout -- swarm/ run_simulation.py 2>/dev/null || true   # full reset so patch #6 (core.py) applies fresh
$SEKA $REPO/scripts/distill/sopbench/apply_two_stage_patch.py $CL
grep -q SOPBENCH_KEEPTUPLE $CL/swarm/core.py && echo "CORE_PATCHED_OK" || echo "CORE_PATCH_MISSING"
cd $CL
CUDA_VISIBLE_DEVICES=0 nohup $VLLM serve Qwen/Qwen2.5-7B-Instruct --port $PORT --enable-lora --lora-modules tbox_v2=$ADAPTER --enable-auto-tool-choice --tool-call-parser hermes --max-model-len 8192 --gpu-memory-utilization 0.85 --dtype bfloat16 --trust-remote-code > /home/woori/scratch/sft_alias_run/serve_keeptuple.log 2>&1 &
SPID=$!; echo "SERVE_PID=$SPID"
for i in $(seq 1 90); do
  if curl -s http://localhost:$PORT/v1/models 2>/dev/null | grep -q tbox_v2; then echo "SERVER_READY"; break; fi
  sleep 10
done
cd $CL
env SOPBENCH_ALIAS=1 SOPBENCH_GATE=1 SOPBENCH_SCRATCHPAD=1 SOPBENCH_SOURCE=1 SOPBENCH_PLAN_MAXTOK=1024 SOPBENCH_OFFLOAD=1 SOPBENCH_OFFLOAD_ACTIVE=1 SOPBENCH_ARGFIX=1 SOPBENCH_VALFIX=1 SOPBENCH_KEEPTUPLE=1 SOPBENCH_AUGMENT_CRED=1 SOPBENCH_OFFLOAD_LOG=$OFFLOG SOPBENCH_VLLM_BASE_URL=http://localhost:$PORT/v1 $SEKA run_simulation.py --domain bank --assistant_model tbox_v2 --tool_call_mode fc --tool_list full --two_stage --two_stage_v2 --ont_dir $CL/induced --output_dir $OUT --env_mode prompt
echo "SIM_DONE"
$SEKA run_evaluation.py --domain bank --assistant_model tbox_v2 --tool_call_mode fc --tool_list full --output_dir $OUT
echo "EVAL_DONE"
kill $SPID 2>/dev/null
echo "DONE_DRIVER"
