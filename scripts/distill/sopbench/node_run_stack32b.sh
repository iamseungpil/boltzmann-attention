#!/bin/bash
# node_run_stack32b.sh — COWORKER plan v1.42 #1 eval: 32B 4-column rows
#   (a) 32B+scaffold STACK  : integrated flags, LOGINCALL OFF, augment OFF  (headline)
#   (b) adapter-only-32B    : SFT planner, scaffold ladder OFF (t1c regime flags only)
# Port of Track-A offload_headline.sh to /scratch + Qwen2.5-32B + lora tbox_32b.
# Report = OFFICIAL success via run_evaluation (134, tool_full); quirk check via diag_quirk_rescore.
set -x
export HF_HUB_CACHE=/scratch/hf_cache
PYT=/scratch/venvs/sop_env/bin/python
VLLM=/scratch/venvs/sop_env/bin/vllm
REPO=/scratch/boltzmann-attention
CL=/scratch/SOPBench
ADAPTER=${ADAPTER:-/scratch/sft_runs/qwen32b_tbox_t1c_lodo_bank}
PORT=${PORT:-8351}
cd $CL

if ! curl -s http://localhost:$PORT/v1/models | grep -q tbox_32b; then
  # TP2 (default GPU0,1 — freed after #1 SFT): 32B bf16+LoRA tight on one 80GB H100
  CUDA_VISIBLE_DEVICES=${SERVE_GPUS:-0,1} nohup $VLLM serve Qwen/Qwen2.5-32B-Instruct \
    --port $PORT --tensor-parallel-size 2 \
    --enable-lora --lora-modules tbox_32b=$ADAPTER \
    --enable-auto-tool-choice --tool-call-parser hermes --max-model-len 8192 \
    --gpu-memory-utilization 0.90 --dtype bfloat16 --trust-remote-code \
    > /scratch/logs/serve_stack32b.log 2>&1 &
  for i in $(seq 1 120); do
    curl -s http://localhost:$PORT/v1/models | grep -q tbox_32b && { echo SERVER_READY; break; }
    sleep 15
  done
fi

STACK_FLAGS="SOPBENCH_ALIAS=1 SOPBENCH_GATE=1 SOPBENCH_SCRATCHPAD=1 SOPBENCH_SOURCE=1 SOPBENCH_PLAN_MAXTOK=1024 SOPBENCH_OFFLOAD=1 SOPBENCH_OFFLOAD_ACTIVE=1 SOPBENCH_ARGFIX=1 SOPBENCH_VALFIX=1 SOPBENCH_KEEPTUPLE=1 SOPBENCH_DGGATE=1 SOPBENCH_LOGINFIRST=1 SOPBENCH_STOPSUCCESS=1"
ADONLY_FLAGS="SOPBENCH_ALIAS=1 SOPBENCH_GATE=1 SOPBENCH_SCRATCHPAD=1 SOPBENCH_SOURCE=1 SOPBENCH_PLAN_MAXTOK=1024"

run_cell () {  # $1=cell-name  $2=env-flags
  local OUT=/scratch/sopbench_runs/$1
  local OFFLOG=/scratch/sopbench_runs/$1_offload.jsonl
  rm -f "$OFFLOG"
  env $2 SOPBENCH_OFFLOAD_LOG=$OFFLOG SOPBENCH_VLLM_BASE_URL=http://localhost:$PORT/v1 \
    $PYT run_simulation.py --domain bank --assistant_model tbox_32b \
    --tool_call_mode fc --tool_list full --two_stage --two_stage_v2 \
    --ont_dir $CL/induced --output_dir $OUT --env_mode prompt \
    > /scratch/logs/$1.simlog 2>&1
  $PYT run_evaluation.py --domain bank --assistant_model tbox_32b \
    --tool_call_mode fc --tool_list full --output_dir $OUT \
    > /scratch/logs/$1.evallog 2>&1
  grep -E "Mean Pass Rate" /scratch/logs/$1.evallog | tail -2
}

run_cell stack32b_bank   "$STACK_FLAGS"
run_cell adonly32b_bank  "$ADONLY_FLAGS"
echo "STACK_EVAL_DONE"
