#!/bin/bash
# Fix-1 LOGINFIRST + AUGMENT_CRED OFF (user decision). A/B, ONE server, two sequential sim+eval:
#   BASE = DGGATE ladder, augment OFF, loginfirst OFF  (expect ~28: transfer 047d augment-pass drops)
#   FIX1 = same + SOPBENCH_LOGINFIRST=1                (expect BASE + cred-present 4 login-order fixes)
# pay_loan 2 (no-login routing = Fix 2) NOT addressed here. PartB 6 = defects, untouched.
exec > /home/woori/scratch/sft_alias_run/offload_loginfirst.log 2>&1
set -x
REPO=/home/woori/workspace_common/boltzmann-attention-pi
CL=/home/woori/scratch/SOPBench
SEKA=/home/woori/venvs/seka_env/bin/python
VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
ADAPTER=$REPO/reports/facet_rft_2026/phase4_distill/sft_runs/qwen7b_tbox_t1c_lodo_bank
PORT=8351
OUT_BASE=/home/woori/scratch/sft_alias_run/eval_t1c_base_noaug
OUT_FIX=/home/woori/scratch/sft_alias_run/eval_t1c_loginfirst
OFFLOG_BASE=/home/woori/scratch/sft_alias_run/offload_log_base_noaug.jsonl
OFFLOG_FIX=/home/woori/scratch/sft_alias_run/offload_log_loginfirst.jsonl
rm -f "$OFFLOG_BASE" "$OFFLOG_FIX"
cd $REPO && git pull --ff-only
cd $CL && git checkout -- swarm/ run_simulation.py 2>/dev/null || true   # full reset so patches apply fresh
$SEKA $REPO/scripts/distill/sopbench/apply_two_stage_patch.py $CL
grep -q SOPBENCH_KEEPTUPLE $CL/swarm/core.py && echo "CORE_PATCHED_OK" || echo "CORE_PATCH_MISSING"
grep -q constraints_original $CL/run_simulation.py && echo "RESET_PATCHED_OK" || echo "RESET_PATCH_MISSING"
grep -q SOPBENCH_LOGINFIRST $CL/scripts/two_stage_client.py && echo "LOGINFIRST_PATCHED_OK" || echo "LOGINFIRST_MISSING"
$SEKA -c "import py_compile,sys; py_compile.compile('$CL/scripts/two_stage_client.py', doraise=True); print('PYCOMPILE_OK')" || { echo "PYCOMPILE_FAIL"; exit 1; }
cd $CL
CUDA_VISIBLE_DEVICES=0 nohup $VLLM serve Qwen/Qwen2.5-7B-Instruct --port $PORT --enable-lora --lora-modules tbox_v2=$ADAPTER --enable-auto-tool-choice --tool-call-parser hermes --max-model-len 8192 --gpu-memory-utilization 0.85 --dtype bfloat16 --trust-remote-code > /home/woori/scratch/sft_alias_run/serve_loginfirst.log 2>&1 &
SPID=$!; echo "SERVE_PID=$SPID"
for i in $(seq 1 90); do
  if curl -s http://localhost:$PORT/v1/models 2>/dev/null | grep -q tbox_v2; then echo "SERVER_READY"; break; fi
  sleep 10
done
# common flags (NOTE: SOPBENCH_AUGMENT_CRED removed = OFF)
COMMON="SOPBENCH_ALIAS=1 SOPBENCH_GATE=1 SOPBENCH_SCRATCHPAD=1 SOPBENCH_SOURCE=1 SOPBENCH_PLAN_MAXTOK=1024 SOPBENCH_OFFLOAD=1 SOPBENCH_OFFLOAD_ACTIVE=1 SOPBENCH_ARGFIX=1 SOPBENCH_VALFIX=1 SOPBENCH_KEEPTUPLE=1 SOPBENCH_DGGATE=1 SOPBENCH_VLLM_BASE_URL=http://localhost:$PORT/v1"

echo "=== RUN BASE (augment OFF, loginfirst OFF) ==="
env $COMMON SOPBENCH_OFFLOAD_LOG=$OFFLOG_BASE $SEKA run_simulation.py --domain bank --assistant_model tbox_v2 --tool_call_mode fc --tool_list full --two_stage --two_stage_v2 --ont_dir $CL/induced --output_dir $OUT_BASE --env_mode prompt
$SEKA run_evaluation.py --domain bank --assistant_model tbox_v2 --tool_call_mode fc --tool_list full --output_dir $OUT_BASE
echo "BASE_DONE"

echo "=== RUN FIX1 (augment OFF, loginfirst ON) ==="
env $COMMON SOPBENCH_LOGINFIRST=1 SOPBENCH_OFFLOAD_LOG=$OFFLOG_FIX $SEKA run_simulation.py --domain bank --assistant_model tbox_v2 --tool_call_mode fc --tool_list full --two_stage --two_stage_v2 --ont_dir $CL/induced --output_dir $OUT_FIX --env_mode prompt
$SEKA run_evaluation.py --domain bank --assistant_model tbox_v2 --tool_call_mode fc --tool_list full --output_dir $OUT_FIX
echo "FIX1_DONE"
kill $SPID 2>/dev/null
echo "DONE_DRIVER"
