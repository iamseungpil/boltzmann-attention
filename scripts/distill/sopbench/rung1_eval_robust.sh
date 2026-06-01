#!/bin/bash
# rung1_eval_robust.sh <regime> <gpu> <port> [extra env KEY=VAL ...]
# Robust single-regime bank eval on a SAVED Rung1 adapter. Designed for the recurring
# post-training vLLM "engine core initialization failed" (GPU not yet freed after training,
# parallel-serve contention). Unlike rung1_train_eval.sh's auto-eval (parallel, no GPU-free
# wait), this: (1) WAITS until the target GPU is free, (2) cleans /dev/shm, (3) serves with
# up to 3 RETRIES, (4) runs bank sim+eval, (5) prints BOTH. Re-runnable on saved adapters,
# so a flaky auto-eval never wastes the ~2h training (adapters persist on disk).
#   s1:        bash rung1_eval_robust.sh s1       0 9001
#   alias_s3:  bash rung1_eval_robust.sh alias_s3 1 9002 SOPBENCH_ALIAS=1 SOPBENCH_SOURCE=3
set +e
PY=/home/woori/venvs/seka_env/bin/python
VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
REPO=/home/woori/workspace_common/boltzmann-attention-pi
SB=$REPO/scripts/distill/sopbench
CLONE=/home/woori/scratch/SOPBench
RUNS=$REPO/reports/facet_rft_2026/phase4_distill/sft_runs
OUT=/home/woori/scratch/sft_alias_run
r=$1; gpu=$2; port=$3; shift 3; envv="$*"
AD=$RUNS/qwen7b_tbox_${r}_scratch_lodo_bank
[ -f "$AD/adapter_model.safetensors" ] || { echo "[$r] NO adapter at $AD (train not done?)"; exit 1; }

# clone client must be 2-token aware (copies two_stage_client.py; constants assert is non-fatal)
$PY $SB/apply_two_stage_patch.py $CLONE 2>&1 | tail -1

clean () { for p in $(nvidia-smi --id=$gpu --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 $p 2>/dev/null; done; sleep 3; rm -f /dev/shm/vllm* /dev/shm/nccl* 2>/dev/null; }
wait_gpu_free () {
  for i in $(seq 1 60); do
    m=$(nvidia-smi --id=$gpu --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
    [ "${m:-99999}" -lt 2000 ] && { echo "[$r] gpu$gpu free (${m}MiB)"; return 0; }
    sleep 10
  done
  echo "[$r] WARN gpu$gpu still ${m}MiB after 600s — proceeding anyway"; return 0
}
serve_ready () { for i in $(seq 1 90); do curl -s -m 3 http://localhost:$port/v1/models 2>/dev/null | grep -q tbox_v2 && return 0; sleep 4; done; return 1; }

wait_gpu_free
ok=0
for attempt in 1 2 3; do
  clean
  echo "[$r] serve attempt $attempt (gpu$gpu port$port) $(date)"
  CUDA_VISIBLE_DEVICES=$gpu nohup $VLLM serve Qwen/Qwen2.5-7B-Instruct --enable-lora --max-lora-rank 16 \
    --lora-modules tbox_v2=$AD --port $port --dtype bfloat16 --gpu-memory-utilization 0.85 \
    --max-model-len 8192 --enable-auto-tool-choice --tool-call-parser hermes --trust-remote-code \
    > $OUT/serve_${r}_scratch_re.log 2>&1 &
  spid=$!
  serve_ready && { ok=1; break; }
  echo "[$r] attempt $attempt FAILED (engine-init); killing+retry"
  kill -9 $spid 2>/dev/null
done
[ "$ok" -ne 1 ] && { echo "[$r] SERVE FAILED after 3 retries"; clean; exit 1; }

cd $CLONE
env SOPBENCH_GATE=1 SOPBENCH_SCRATCHPAD=1 $envv SOPBENCH_VLLM_BASE_URL=http://localhost:$port/v1 $PY run_simulation.py \
  --domain bank --assistant_model tbox_v2 --tool_call_mode fc --tool_list full \
  --two_stage --two_stage_v2 --ont_dir $CLONE/induced --output_dir $OUT/eval_${r}_scratch_re --env_mode prompt \
  > $OUT/sim_${r}_scratch_re.log 2>&1
$PY run_evaluation.py --domain bank --assistant_model tbox_v2 --tool_call_mode fc --tool_list full \
  --output_dir $OUT/eval_${r}_scratch_re > $OUT/evalout_${r}_scratch_re.txt 2>&1
clean
$PY - "$r" <<'PYEOF'
import json, glob, sys
r = sys.argv[1]
def e0(x):
    e = x.get("evaluations"); return (e[0] if e else {}) if isinstance(e, list) else (e or {})
fs = glob.glob(f'/home/woori/scratch/sft_alias_run/eval_{r}_scratch_re/bank/*.json')
if not fs:
    print(f"{r}: NO DATA"); sys.exit()
d = json.load(open(fs[0]))
T = [x for x in d if e0(x).get("action_should_succeed")]
F = [x for x in d if not e0(x).get("action_should_succeed")]
both = sum(1 for x in T if e0(x).get("dirgraph_satisfied") and e0(x).get("action_called_correctly"))
print(f"[RESULT] {r}_scratch: should_T succ={sum(bool(e0(x).get('success')) for x in T)}/48 "
      f"dirgraph={sum(bool(e0(x).get('dirgraph_satisfied')) for x in T)} "
      f"goal={sum(bool(e0(x).get('action_called_correctly')) for x in T)} "
      f"BOTH={both} || should_F={sum(bool(e0(x).get('success')) for x in F)}/86")
print("(baseline gate-token: s1_gate BOTH=2, alias_s3_gate BOTH=1)")
PYEOF
