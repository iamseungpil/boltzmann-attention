#!/bin/bash
# Fix-3 STOPSUCCESS A/B (augment OFF, FULL stack incl DGGATE = final shipping config, Refinement 2).
# B-3 PASSED 12/12 (diag_fix3_offline). Metric = official success (pass@1), reported end-to-end.
#   S0 = full stack, STOPSUCCESS off  (= logincall L1C reproduction, official success 28)
#   S1 = full stack + SOPBENCH_STOPSUCCESS=1  (expect +up to 12 -> ~40 full success)
exec > /home/woori/scratch/sft_alias_run/offload_stopsuccess.log 2>&1
set -x
REPO=/home/woori/workspace_common/boltzmann-attention-pi
CL=/home/woori/scratch/SOPBench
SEKA=/home/woori/venvs/seka_env/bin/python
VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
ADAPTER=$REPO/reports/facet_rft_2026/phase4_distill/sft_runs/qwen7b_tbox_t1c_lodo_bank
PORT=8351
OUT_S0=/home/woori/scratch/sft_alias_run/eval_t1c_s0
OUT_S1=/home/woori/scratch/sft_alias_run/eval_t1c_stopsuccess
OFFLOG_S0=/home/woori/scratch/sft_alias_run/offload_log_s0.jsonl
OFFLOG_S1=/home/woori/scratch/sft_alias_run/offload_log_stopsuccess.jsonl
rm -f "$OFFLOG_S0" "$OFFLOG_S1"
cd $REPO && git pull --ff-only
cd $CL && git checkout -- swarm/ run_simulation.py 2>/dev/null || true
$SEKA $REPO/scripts/distill/sopbench/apply_two_stage_patch.py $CL
grep -q SOPBENCH_STOPSUCCESS $CL/scripts/two_stage_client.py && echo "STOPSUCCESS_PATCHED_OK" || echo "STOPSUCCESS_MISSING"
$SEKA -c "import py_compile; py_compile.compile('$CL/scripts/two_stage_client.py', doraise=True); print('PYCOMPILE_OK')" || { echo "PYCOMPILE_FAIL"; exit 1; }
cd $CL
CUDA_VISIBLE_DEVICES=0 nohup $VLLM serve Qwen/Qwen2.5-7B-Instruct --port $PORT --enable-lora --lora-modules tbox_v2=$ADAPTER --enable-auto-tool-choice --tool-call-parser hermes --max-model-len 8192 --gpu-memory-utilization 0.85 --dtype bfloat16 --trust-remote-code > /home/woori/scratch/sft_alias_run/serve_stopsuccess.log 2>&1 &
SPID=$!; echo "SERVE_PID=$SPID"
for i in $(seq 1 90); do
  if curl -s http://localhost:$PORT/v1/models 2>/dev/null | grep -q tbox_v2; then echo "SERVER_READY"; break; fi
  sleep 10
done
# FULL stack (Refinement 2: DGGATE included; AUGMENT_CRED absent = OFF)
COMMON="SOPBENCH_ALIAS=1 SOPBENCH_GATE=1 SOPBENCH_SCRATCHPAD=1 SOPBENCH_SOURCE=1 SOPBENCH_PLAN_MAXTOK=1024 SOPBENCH_OFFLOAD=1 SOPBENCH_OFFLOAD_ACTIVE=1 SOPBENCH_ARGFIX=1 SOPBENCH_VALFIX=1 SOPBENCH_KEEPTUPLE=1 SOPBENCH_DGGATE=1 SOPBENCH_LOGINFIRST=1 SOPBENCH_LOGINCALL=1 SOPBENCH_VLLM_BASE_URL=http://localhost:$PORT/v1"

echo "=== RUN S0 (full stack, STOPSUCCESS off) ==="
env $COMMON SOPBENCH_OFFLOAD_LOG=$OFFLOG_S0 $SEKA run_simulation.py --domain bank --assistant_model tbox_v2 --tool_call_mode fc --tool_list full --two_stage --two_stage_v2 --ont_dir $CL/induced --output_dir $OUT_S0 --env_mode prompt
$SEKA run_evaluation.py --domain bank --assistant_model tbox_v2 --tool_call_mode fc --tool_list full --output_dir $OUT_S0
echo "S0_DONE"

echo "=== RUN S1 (full stack + STOPSUCCESS) ==="
env $COMMON SOPBENCH_STOPSUCCESS=1 SOPBENCH_OFFLOAD_LOG=$OFFLOG_S1 $SEKA run_simulation.py --domain bank --assistant_model tbox_v2 --tool_call_mode fc --tool_list full --two_stage --two_stage_v2 --ont_dir $CL/induced --output_dir $OUT_S1 --env_mode prompt
$SEKA run_evaluation.py --domain bank --assistant_model tbox_v2 --tool_call_mode fc --tool_list full --output_dir $OUT_S1
echo "S1_DONE"
kill $SPID 2>/dev/null
echo "DONE_DRIVER"
