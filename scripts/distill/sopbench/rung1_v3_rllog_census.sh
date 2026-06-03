#!/bin/bash
# rung1_v3_rllog_census.sh — MECHANISM-LAYER census for the v3 A/B (2026-06-03).
# Re-serves the already-trained nt/treeval adapters and re-runs bank eval with SOPBENCH_RLLOG set,
# capturing every planner (prompt, raw scratchpad output). Purpose: confirm WHY treeval fails to
# converge (behavioral census showed 35/48 should_T loop to the max_steps=10 cap). Hypothesis:
# the verbose grounded terminal `ready=true; gate = <expr> = <val>; <ACT|STOP>` exceeds the planner
# decode budget (max_tokens=24 in two_stage_client) -> truncated -> no terminal token parsed -> the
# harness re-gathers -> loop. RLLOG lets us classify each terminal: grounded-gate vs legacy-permitted
# vs truncated/no-terminal vs gather.  Read-only on adapters; writes rllog_{arm}.jsonl + sim/eval.
set +e
PY=/home/woori/venvs/seka_env/bin/python
VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
REPO=/home/woori/workspace_common/boltzmann-attention-pi
SB=$REPO/scripts/distill/sopbench
CLONE=/home/woori/scratch/SOPBench
RUNS=$REPO/reports/facet_rft_2026/phase4_distill/sft_runs
OUT=/home/woori/scratch/sft_alias_run
SUM=$OUT/RLLOG_CENSUS.txt
echo "=== v3 RLLOG mechanism census start $(date) ===" > $SUM
rm -f /dev/shm/vllm* /dev/shm/nccl* 2>/dev/null
$PY $SB/apply_two_stage_patch.py $CLONE >> $SUM 2>&1

kill_gpu () { for p in $(nvidia-smi --id=$1 --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 $p 2>/dev/null; done; sleep 4; rm -f /dev/shm/vllm* /dev/shm/nccl* 2>/dev/null; }
eval_one () {  # arm gpu port
  local arm=$1 gpu=$2 port=$3
  local AD=$RUNS/qwen7b_tbox_alias_s3_${arm}_lodo_bank
  echo "[$arm] gpu=$gpu $(date)" >> $SUM
  [ -f "$AD/adapter_model.safetensors" ] || { echo "[$arm] NO adapter" >> $SUM; return; }
  rm -rf $OUT/eval_rllog_${arm}; rm -f $OUT/rllog_${arm}.jsonl
  local iport=$((8500 + gpu*200))
  CUDA_VISIBLE_DEVICES=$gpu VLLM_PORT=$iport VLLM_DP_MASTER_PORT=$((iport+50)) nohup $VLLM serve Qwen/Qwen2.5-7B-Instruct --enable-lora --max-lora-rank 16 \
    --lora-modules tbox_v2=$AD --port $port --dtype bfloat16 --gpu-memory-utilization 0.85 \
    --max-model-len 8192 --enable-auto-tool-choice --tool-call-parser hermes --trust-remote-code \
    > $OUT/serve_rllog_${arm}.log 2>&1 &
  for i in $(seq 1 120); do curl -s -m 3 http://localhost:$port/v1/models 2>/dev/null | grep -q tbox_v2 && break; sleep 4; done
  curl -s -m 3 http://localhost:$port/v1/models 2>/dev/null | grep -q tbox_v2 || { echo "[$arm] SERVE FAILED" >> $SUM; kill_gpu $gpu; return; }
  cd $CLONE
  env SOPBENCH_GATE=1 SOPBENCH_SCRATCHPAD=1 SOPBENCH_ALIAS=1 SOPBENCH_SOURCE=3 \
      SOPBENCH_RLLOG=$OUT/rllog_${arm}.jsonl SOPBENCH_VLLM_BASE_URL=http://localhost:$port/v1 $PY run_simulation.py \
    --domain bank --assistant_model tbox_v2 --tool_call_mode fc --tool_list full \
    --two_stage --two_stage_v2 --ont_dir $CLONE/induced --output_dir $OUT/eval_rllog_${arm} --env_mode prompt \
    > $OUT/sim_rllog_${arm}.log 2>&1
  echo "[$arm] rllog lines=$(wc -l < $OUT/rllog_${arm}.jsonl 2>/dev/null)" >> $SUM
  kill_gpu $gpu
}
eval_one treeval 0 9501 &
eval_one nt      1 9502 &
wait
echo "=== v3 RLLOG census DONE $(date) ===" >> $SUM
cat $SUM
