#!/bin/bash
# Overnight: wait for the 3 held-out adapters, then eval each on its HELD-OUT target(s) with
# STACK (full scaffold, LOGINCALL off, augment off) + ADAPTER-ONLY (scaffold off). Official success.
# Adapters (trained by xtrain_orchestrate): t1c_lodo_library, t1c_lodo_healthcare, t1c_train1_bank.
exec > /home/woori/scratch/sft_alias_run/xdomain_overnight_eval.log 2>&1
set -x
REPO=/home/woori/workspace_common/boltzmann-attention-pi
CL=/home/woori/scratch/SOPBench
SEKA=/home/woori/venvs/seka_env/bin/python
VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
RUNS=$REPO/reports/facet_rft_2026/phase4_distill/sft_runs
OUT=/home/woori/scratch/sft_alias_run
PORT=8351
STACK="SOPBENCH_ALIAS=1 SOPBENCH_GATE=1 SOPBENCH_SCRATCHPAD=1 SOPBENCH_SOURCE=1 SOPBENCH_PLAN_MAXTOK=1024 SOPBENCH_OFFLOAD=1 SOPBENCH_OFFLOAD_ACTIVE=1 SOPBENCH_ARGFIX=1 SOPBENCH_VALFIX=1 SOPBENCH_KEEPTUPLE=1 SOPBENCH_DGGATE=1 SOPBENCH_LOGINFIRST=1 SOPBENCH_STOPSUCCESS=1"

wait_adapter () { for i in $(seq 1 360); do [ -f $RUNS/qwen7b_tbox_$1/adapter_model.safetensors ] && return 0; sleep 60; done; return 1; }

eval_adapter_on () {  # $1=adapter_name  $2=tag  $3..=held-out target domains
  local ADNAME=$1; local AD=$RUNS/qwen7b_tbox_$1; local TAG=$2; shift 2; local DOMS="$@"
  wait_adapter $ADNAME || { echo "ADAPTER_MISSING $ADNAME"; return; }
  cd $CL && git checkout -- swarm/ run_simulation.py 2>/dev/null || true
  $SEKA $REPO/scripts/distill/sopbench/apply_two_stage_patch.py $CL
  rm -f /dev/shm/vllm* 2>/dev/null
  CUDA_VISIBLE_DEVICES=0 nohup $VLLM serve Qwen/Qwen2.5-7B-Instruct --port $PORT --enable-lora --lora-modules tbox_v2=$AD --enable-auto-tool-choice --tool-call-parser hermes --max-model-len 8192 --gpu-memory-utilization 0.85 --dtype bfloat16 --trust-remote-code > $OUT/serve_overnight_$TAG.log 2>&1 &
  local SPID=$!
  for i in $(seq 1 90); do curl -s http://localhost:$PORT/v1/models 2>/dev/null | grep -q tbox_v2 && break; sleep 10; done
  for D in $DOMS; do
    echo "=== $TAG : $D : adapter-only ==="
    $SEKA run_simulation.py --domain $D --assistant_model tbox_v2 --tool_call_mode fc --tool_list full --two_stage --two_stage_v2 --ont_dir $CL/induced --output_dir $OUT/xho_${TAG}_${D}_adapteronly --env_mode prompt
    $SEKA run_evaluation.py --domain $D --assistant_model tbox_v2 --tool_call_mode fc --tool_list full --output_dir $OUT/xho_${TAG}_${D}_adapteronly
    echo "=== $TAG : $D : stack ==="
    env $STACK SOPBENCH_VLLM_BASE_URL=http://localhost:$PORT/v1 $SEKA run_simulation.py --domain $D --assistant_model tbox_v2 --tool_call_mode fc --tool_list full --two_stage --two_stage_v2 --ont_dir $CL/induced --output_dir $OUT/xho_${TAG}_${D}_stack --env_mode prompt
    $SEKA run_evaluation.py --domain $D --assistant_model tbox_v2 --tool_call_mode fc --tool_list full --output_dir $OUT/xho_${TAG}_${D}_stack
    echo "DONE $TAG $D"
  done
  kill $SPID 2>/dev/null; sleep 5; rm -f /dev/shm/vllm* 2>/dev/null
}

# wait for GPU0 free (in-domain xdomain run + lodo_healthcare train both finish on GPU0) before serving.
for i in $(seq 1 480); do
  busy0=$(nvidia-smi --id=0 --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
  [ "$busy0" -eq 0 ] && break; sleep 60
done
echo "GPU0 free $(date) -> start held-out evals"
eval_adapter_on t1c_lodo_library    lodo_library    library
eval_adapter_on t1c_lodo_healthcare lodo_healthcare healthcare
eval_adapter_on t1c_train1_bank     train1bank      dmv healthcare hotel library online_market university
echo "OVERNIGHT_EVAL_DONE $(date)"
