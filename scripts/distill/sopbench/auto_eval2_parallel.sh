#!/bin/bash
# auto_eval2_parallel.sh — evaluate s3 + alias_s3 IN PARALLEL on GPU0 + GPU1 on bank.
# (s1 already done = 0.6045.) Each regime is isolated: serves on its own GPU+port and tears
# down ONLY that GPU's processes (nvidia-smi --id=<gpu> pids), so the two jobs never kill each
# other's serve. Avoids pkill-all (which collided before). Assumes both GPUs start free.
set +e
PY=/home/woori/venvs/seka_env/bin/python
VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
CLONE=/home/woori/scratch/SOPBench
RUNS=/home/woori/workspace_common/boltzmann-attention-pi/reports/facet_rft_2026/phase4_distill/sft_runs
OUT=/home/woori/scratch/sft_alias_run
SUM=$OUT/AUTO_EVAL2_RESULTS.txt
echo "=== auto_eval2 (parallel) started $(date) ===" > $SUM
rm -f /dev/shm/vllm* /dev/shm/nccl* 2>/dev/null

kill_gpu () {   # kill only the compute procs on GPU $1 (isolates the two jobs)
  for p in $(nvidia-smi --id=$1 --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    kill -9 $p 2>/dev/null
  done
}

eval_regime () {
  local r=$1 gpu=$2 port=$3; shift 3; local envv="$*"
  local AD=$RUNS/qwen7b_tbox_${r}_lodo_bank
  local LP=$OUT/p_$r.log
  echo "[$r] gpu=$gpu port=$port env=[$envv] start $(date)" >> $SUM
  [ -f "$AD/adapter_model.safetensors" ] || { echo "[$r] NO adapter, skip" >> $SUM; return; }
  CUDA_VISIBLE_DEVICES=$gpu nohup $VLLM serve Qwen/Qwen2.5-7B-Instruct --enable-lora --max-lora-rank 16 \
    --lora-modules tbox_v2=$AD --port $port --dtype bfloat16 --gpu-memory-utilization 0.85 \
    --max-model-len 8192 --enable-auto-tool-choice --tool-call-parser hermes --trust-remote-code \
    > $OUT/serve_$r.log 2>&1 &
  for i in $(seq 1 120); do
    curl -s -m 3 http://localhost:$port/v1/models 2>/dev/null | grep -q tbox_v2 && break
    sleep 4
  done
  if ! curl -s -m 3 http://localhost:$port/v1/models 2>/dev/null | grep -q tbox_v2; then
    echo "[$r] SERVE FAILED (serve_$r.log)" >> $SUM; kill_gpu $gpu; return; fi
  echo "[$r] endpoint up, simulating $(date)" >> $LP
  cd $CLONE
  env $envv SOPBENCH_VLLM_BASE_URL=http://localhost:$port/v1 $PY run_simulation.py \
    --domain bank --assistant_model tbox_v2 --tool_call_mode fc --tool_list full \
    --two_stage --two_stage_v2 --ont_dir $CLONE/induced --output_dir $OUT/eval_$r --env_mode prompt \
    > $OUT/sim_$r.log 2>&1
  $PY run_evaluation.py --domain bank --assistant_model tbox_v2 --tool_call_mode fc --tool_list full \
    --output_dir $OUT/eval_$r > $OUT/evalout_$r.txt 2>&1
  echo "[$r] $(grep -E 'Mean Pass Rate' $OUT/evalout_$r.txt | tail -1) ($(date))" >> $SUM
  kill_gpu $gpu
}

# refresh clone client to alias-aware version (idempotent; constants assert may warn — harmless)
$PY /home/woori/workspace_common/boltzmann-attention-pi/scripts/distill/sopbench/apply_two_stage_patch.py $CLONE >> $SUM 2>&1
echo "clone make_alias_map=$(grep -c make_alias_map $CLONE/scripts/two_stage_client.py)" >> $SUM

eval_regime s3       0 9001 SOPBENCH_SOURCE=3 &
eval_regime alias_s3 1 9002 SOPBENCH_ALIAS=1 SOPBENCH_SOURCE=3 &
wait
rm -f /dev/shm/vllm* /dev/shm/nccl* 2>/dev/null

echo "=== auto_eval2 DONE $(date) ===" >> $SUM
echo "--- SUMMARY /134 (arm-4a v2=0.2610, s1=0.6045) ---" >> $SUM
for r in s3 alias_s3; do echo "$r: $(grep -E 'Mean Pass Rate' $OUT/evalout_$r.txt 2>/dev/null | tail -1)" >> $SUM; done
