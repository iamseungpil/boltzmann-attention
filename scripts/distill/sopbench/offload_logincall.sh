#!/bin/bash
# Fix-2 LOGINCALL A/B (augment OFF). ONE server, two sequential sim+eval:
#   L1   = LOGINFIRST on, LOGINCALL off            (re-measure Fix-1 baseline = 33)
#   L1C  = LOGINFIRST on + SOPBENCH_LOGINCALL=1     (expect +pay_loan x2: drive login CALL for cred-absent)
exec > /home/woori/scratch/sft_alias_run/offload_logincall.log 2>&1
set -x
REPO=/home/woori/workspace_common/boltzmann-attention-pi
CL=/home/woori/scratch/SOPBench
SEKA=/home/woori/venvs/seka_env/bin/python
VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
ADAPTER=$REPO/reports/facet_rft_2026/phase4_distill/sft_runs/qwen7b_tbox_t1c_lodo_bank
PORT=8351
OUT_L1=/home/woori/scratch/sft_alias_run/eval_t1c_l1
OUT_L1C=/home/woori/scratch/sft_alias_run/eval_t1c_logincall
OFFLOG_L1=/home/woori/scratch/sft_alias_run/offload_log_l1.jsonl
OFFLOG_L1C=/home/woori/scratch/sft_alias_run/offload_log_logincall.jsonl
rm -f "$OFFLOG_L1" "$OFFLOG_L1C"
cd $REPO && git pull --ff-only
cd $CL && git checkout -- swarm/ run_simulation.py 2>/dev/null || true
$SEKA $REPO/scripts/distill/sopbench/apply_two_stage_patch.py $CL
grep -q SOPBENCH_LOGINCALL $CL/scripts/two_stage_client.py && echo "LOGINCALL_PATCHED_OK" || echo "LOGINCALL_MISSING"
$SEKA -c "import py_compile; py_compile.compile('$CL/scripts/two_stage_client.py', doraise=True); print('PYCOMPILE_OK')" || { echo "PYCOMPILE_FAIL"; exit 1; }
cd $CL
CUDA_VISIBLE_DEVICES=0 nohup $VLLM serve Qwen/Qwen2.5-7B-Instruct --port $PORT --enable-lora --lora-modules tbox_v2=$ADAPTER --enable-auto-tool-choice --tool-call-parser hermes --max-model-len 8192 --gpu-memory-utilization 0.85 --dtype bfloat16 --trust-remote-code > /home/woori/scratch/sft_alias_run/serve_logincall.log 2>&1 &
SPID=$!; echo "SERVE_PID=$SPID"
for i in $(seq 1 90); do
  if curl -s http://localhost:$PORT/v1/models 2>/dev/null | grep -q tbox_v2; then echo "SERVER_READY"; break; fi
  sleep 10
done
COMMON="SOPBENCH_ALIAS=1 SOPBENCH_GATE=1 SOPBENCH_SCRATCHPAD=1 SOPBENCH_SOURCE=1 SOPBENCH_PLAN_MAXTOK=1024 SOPBENCH_OFFLOAD=1 SOPBENCH_OFFLOAD_ACTIVE=1 SOPBENCH_ARGFIX=1 SOPBENCH_VALFIX=1 SOPBENCH_KEEPTUPLE=1 SOPBENCH_DGGATE=1 SOPBENCH_LOGINFIRST=1 SOPBENCH_VLLM_BASE_URL=http://localhost:$PORT/v1"

echo "=== RUN L1 (loginfirst, NO logincall) ==="
env $COMMON SOPBENCH_OFFLOAD_LOG=$OFFLOG_L1 $SEKA run_simulation.py --domain bank --assistant_model tbox_v2 --tool_call_mode fc --tool_list full --two_stage --two_stage_v2 --ont_dir $CL/induced --output_dir $OUT_L1 --env_mode prompt
$SEKA run_evaluation.py --domain bank --assistant_model tbox_v2 --tool_call_mode fc --tool_list full --output_dir $OUT_L1
echo "L1_DONE"

echo "=== RUN L1C (loginfirst + logincall) ==="
env $COMMON SOPBENCH_LOGINCALL=1 SOPBENCH_OFFLOAD_LOG=$OFFLOG_L1C $SEKA run_simulation.py --domain bank --assistant_model tbox_v2 --tool_call_mode fc --tool_list full --two_stage --two_stage_v2 --ont_dir $CL/induced --output_dir $OUT_L1C --env_mode prompt
$SEKA run_evaluation.py --domain bank --assistant_model tbox_v2 --tool_call_mode fc --tool_list full --output_dir $OUT_L1C
echo "L1C_DONE"
kill $SPID 2>/dev/null
echo "DONE_DRIVER"
