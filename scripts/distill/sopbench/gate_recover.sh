#!/bin/bash
# gate_recover.sh — recover Exp-4d: s3_gate OOM'd (2 jobs/GPU), eval failed (serve at train->eval
# transition). GPUs are free now. Phase A: eval the 2 READY adapters (s1_gate, alias_s3_gate) in
# parallel (the key headline). Phase B: retrain s3_gate SOLO (no OOM) then eval it. Headline =
# should_T dirgraph+ AND goal+.
set +e
PY=/home/woori/venvs/seka_env/bin/python
VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
REPO=/home/woori/workspace_common/boltzmann-attention-pi
SB=$REPO/scripts/distill/sopbench
TR=$REPO/scripts/distill/lora_train_chat_toolcall.py
CLONE=/home/woori/scratch/SOPBench
RUNS=$REPO/reports/facet_rft_2026/phase4_distill/sft_runs
OUT=/home/woori/scratch/sft_alias_run
SUM=$OUT/GATE_RESULTS2.txt
echo "=== gate_recover start $(date) ===" > $SUM
$PY $SB/apply_two_stage_patch.py $CLONE >> $SUM 2>&1
rm -f /dev/shm/vllm* /dev/shm/nccl* 2>/dev/null

kill_gpu () { for p in $(nvidia-smi --id=$1 --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 $p 2>/dev/null; done; sleep 4; rm -f /dev/shm/vllm* /dev/shm/nccl* 2>/dev/null; }
eval_gate () {
  local r=$1 gpu=$2 port=$3; shift 3; local envv="$*"
  local AD=$RUNS/qwen7b_tbox_${r}_gate_lodo_bank
  echo "[$r] gpu=$gpu port=$port env=[SOPBENCH_GATE=1 $envv] $(date)" >> $SUM
  [ -f "$AD/adapter_model.safetensors" ] || { echo "[$r] NO adapter" >> $SUM; return; }
  CUDA_VISIBLE_DEVICES=$gpu nohup $VLLM serve Qwen/Qwen2.5-7B-Instruct --enable-lora --max-lora-rank 16 \
    --lora-modules tbox_v2=$AD --port $port --dtype bfloat16 --gpu-memory-utilization 0.85 \
    --max-model-len 8192 --enable-auto-tool-choice --tool-call-parser hermes --trust-remote-code \
    > $OUT/serve_${r}_gate.log 2>&1 &
  for i in $(seq 1 120); do curl -s -m 3 http://localhost:$port/v1/models 2>/dev/null | grep -q tbox_v2 && break; sleep 4; done
  curl -s -m 3 http://localhost:$port/v1/models 2>/dev/null | grep -q tbox_v2 || { echo "[$r] SERVE FAILED" >> $SUM; kill_gpu $gpu; return; }
  cd $CLONE
  env SOPBENCH_GATE=1 $envv SOPBENCH_VLLM_BASE_URL=http://localhost:$port/v1 $PY run_simulation.py \
    --domain bank --assistant_model tbox_v2 --tool_call_mode fc --tool_list full \
    --two_stage --two_stage_v2 --ont_dir $CLONE/induced --output_dir $OUT/eval_${r}_gate --env_mode prompt \
    > $OUT/sim_${r}_gate.log 2>&1
  $PY run_evaluation.py --domain bank --assistant_model tbox_v2 --tool_call_mode fc --tool_list full \
    --output_dir $OUT/eval_${r}_gate > $OUT/evalout_${r}_gate.txt 2>&1
  echo "[$r] $(grep -E 'Mean Pass Rate' $OUT/evalout_${r}_gate.txt | tail -1)" >> $SUM
  kill_gpu $gpu
}
breakdown () {
  $PY - "$@" >> $SUM 2>&1 <<'PYEOF'
import json,glob,sys
def e0(x):
    e=x.get("evaluations"); return (e[0] if e else {}) if isinstance(e,list) else (e or {})
for r in sys.argv[1:]:
    fs=glob.glob(f'/home/woori/scratch/sft_alias_run/eval_{r}_gate/bank/*.json')
    if not fs: print(f"{r}_gate: NO DATA"); continue
    d=json.load(open(fs[0])); T=[x for x in d if e0(x).get("action_should_succeed")]; F=[x for x in d if not e0(x).get("action_should_succeed")]
    succ=sum(bool(e0(x).get("success")) for x in T); dg=sum(bool(e0(x).get("dirgraph_satisfied")) for x in T)
    ac=sum(bool(e0(x).get("action_called_correctly")) for x in T)
    both=sum(1 for x in T if e0(x).get("dirgraph_satisfied") and e0(x).get("action_called_correctly"))
    sf=sum(bool(e0(x).get("success")) for x in F)
    print(f"{r}_gate: should_T succ={succ}/48 dirgraph={dg} goal={ac} BOTH(headline)={both} || should_F={sf}/86")
PYEOF
}

# Phase A: eval the 2 ready adapters in parallel (GPU0: s1, GPU1: alias_s3)
echo "=== Phase A: eval ready adapters $(date) ===" >> $SUM
eval_gate s1       0 9001 &
eval_gate alias_s3 1 9002 SOPBENCH_ALIAS=1 SOPBENCH_SOURCE=3 &
wait
echo "--- Phase A headline ---" >> $SUM
breakdown s1 alias_s3

# Phase B: retrain s3_gate SOLO on GPU0 (no OOM), then eval
echo "=== Phase B: retrain s3_gate solo $(date) ===" >> $SUM
rm -f /dev/shm/vllm* /dev/shm/nccl* 2>/dev/null
cd $REPO/scripts/distill
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY $TR \
  --base-model Qwen/Qwen2.5-7B-Instruct --device cuda:0 --max-seq-len 2048 --epochs 3 --lora-r 16 \
  --val-frac 0.05 --skip-overlong --train-jsonl $OUT/lodo_train_s3_gate.jsonl \
  --out-dir $RUNS/qwen7b_tbox_s3_gate_lodo_bank > $OUT/train_s3_gate.log 2>&1
echo "s3_gate retrain done $(date), meta=$(ls $RUNS/qwen7b_tbox_s3_gate_lodo_bank/train_meta.json 2>/dev/null | wc -l)" >> $SUM
eval_gate s3 0 9001 SOPBENCH_SOURCE=3

echo "=== FINAL HEADLINE (should_T dirgraph+ INT goal+) $(date) ===" >> $SUM
breakdown s1 s3 alias_s3
echo "=== gate_recover DONE $(date) ===" >> $SUM
