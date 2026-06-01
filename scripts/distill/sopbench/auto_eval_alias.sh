#!/bin/bash
# auto_eval_alias.sh — wait for the 3 alias-regime SFTs to finish, then serve+eval each on bank.
# Launched overnight (nohup) so the trained s1/s3/alias_s3 adapters get a real bank pass-rate
# (run_simulation --two_stage_v2 -> run_evaluation) with the matching SOPBENCH_ALIAS/SOURCE toggle.
# Results -> $OUT/AUTO_EVAL_RESULTS.txt. Idempotent-ish; reuses lora-module name tbox_v2 (FC-proven).
set +e
PY=/home/woori/venvs/seka_env/bin/python
VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
REPO=/home/woori/workspace_common/boltzmann-attention-pi/scripts/distill/sopbench
CLONE=/home/woori/scratch/SOPBench
RUNS=/home/woori/workspace_common/boltzmann-attention-pi/reports/facet_rft_2026/phase4_distill/sft_runs
OUT=/home/woori/scratch/sft_alias_run
SUM=$OUT/AUTO_EVAL_RESULTS.txt
echo "=== auto_eval started $(date) ===" > $SUM

# 1) wait for all 3 trainings to finish (train_meta.json) or trainers to die
while true; do
  n=0
  for r in s1 s3 alias_s3; do [ -f "$RUNS/qwen7b_tbox_${r}_lodo_bank/train_meta.json" ] && n=$((n+1)); done
  alive=$(pgrep -fc lora_train_chat_toolcall)
  echo "$(date) meta=$n/3 trainers_alive=$alive" >> $SUM
  [ "$n" -ge 3 ] && break
  [ "$alive" -eq 0 ] && { echo "trainers gone, meta=$n -> proceed with whats saved" >> $SUM; break; }
  sleep 120
done

# 2) refresh clone client to alias-aware version (sanctioned deploy tool, idempotent)
cd $REPO && $PY apply_two_stage_patch.py $CLONE >> $SUM 2>&1
echo "clone client make_alias_map=$(grep -c make_alias_map $CLONE/scripts/two_stage_client.py)" >> $SUM

# serve on GPU1 (GPU0 may hold a wedged/engine-init-failed vLLM that pkill cannot reap).
SERVE_GPU=1
teardown () {
  pkill -9 -f "vllm serve Qwen/Qwen2.5-7B" 2>/dev/null
  pkill -9 -f "tau2_vllm_env/bin/python" 2>/dev/null
  # poll until the SERVE_GPU memory is released (the reapable serve we just started)
  for i in $(seq 1 40); do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits --id=$SERVE_GPU)
    [ "${u:-9999}" -lt 2000 ] && break
    sleep 4
  done
  sleep 5
}

run_one () {
  local r=$1; shift; local envv="$*"
  local AD=$RUNS/qwen7b_tbox_${r}_lodo_bank
  echo "=== [$r] env=[$envv] $(date) ===" >> $SUM
  [ -f "$AD/adapter_model.safetensors" ] || { echo "[$r] NO adapter, skip" >> $SUM; return; }
  if grep -q "Mean Pass Rate" "$OUT/evalout_$r.txt" 2>/dev/null; then
    echo "[$r] already evaluated, skip" >> $SUM; return; fi
  teardown
  CUDA_VISIBLE_DEVICES=$SERVE_GPU nohup $VLLM serve Qwen/Qwen2.5-7B-Instruct --enable-lora --max-lora-rank 16 \
    --lora-modules tbox_v2=$AD --port 9000 --dtype bfloat16 --gpu-memory-utilization 0.85 \
    --max-model-len 8192 --enable-auto-tool-choice --tool-call-parser hermes --trust-remote-code \
    > $OUT/serve_$r.log 2>&1 &
  for i in $(seq 1 100); do
    curl -s -m 3 http://localhost:9000/v1/models 2>/dev/null | grep -q tbox_v2 && break
    sleep 4
  done
  curl -s -m 3 http://localhost:9000/v1/models 2>/dev/null | grep -q tbox_v2 || {
    echo "[$r] SERVE FAILED (see serve_$r.log)" >> $SUM; teardown; return; }
  echo "[$r] endpoint up, simulating $(date)" >> $SUM
  cd $CLONE
  env $envv SOPBENCH_VLLM_BASE_URL=http://localhost:9000/v1 $PY run_simulation.py \
    --domain bank --assistant_model tbox_v2 --tool_call_mode fc --tool_list full \
    --two_stage --two_stage_v2 --ont_dir $CLONE/induced --output_dir $OUT/eval_$r --env_mode prompt \
    > $OUT/sim_$r.log 2>&1
  $PY run_evaluation.py --domain bank --assistant_model tbox_v2 --tool_call_mode fc --tool_list full \
    --output_dir $OUT/eval_$r > $OUT/evalout_$r.txt 2>&1
  echo "[$r] $(grep -E 'Mean Pass Rate' $OUT/evalout_$r.txt | tail -1)" >> $SUM
  teardown
}

run_one s1
run_one s3 SOPBENCH_SOURCE=3
run_one alias_s3 SOPBENCH_ALIAS=1 SOPBENCH_SOURCE=3

echo "=== auto_eval DONE $(date) ===" >> $SUM
echo "--- SUMMARY (Mean Pass Rate /134; arm-4a v2 baseline=0.2610) ---" >> $SUM
for r in s1 s3 alias_s3; do
  echo "$r: $(grep -E 'Mean Pass Rate' $OUT/evalout_$r.txt 2>/dev/null | tail -1)" >> $SUM
done
