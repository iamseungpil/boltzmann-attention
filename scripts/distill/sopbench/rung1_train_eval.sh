#!/bin/bash
# rung1_train_eval.sh — §3 Rung1 ①: per-step readiness-gate SFT, train + bank-eval.
# 2 regimes in parallel (each SOLO on its own GPU = no OOM): s1_scratch (GPU0, source1+gate+scratch),
# alias_s3_scratch (GPU1, alias+source3+gate+scratch). Headline = should_T BOTH(dirgraph∩goal),
# compared to gate-token baseline (s1_gate BOTH 2, alias_s3_gate BOTH 1).
set +e
PY=/home/woori/venvs/seka_env/bin/python
VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
REPO=/home/woori/workspace_common/boltzmann-attention-pi
SB=$REPO/scripts/distill/sopbench
TR=$REPO/scripts/distill/lora_train_chat_toolcall.py
CLONE=/home/woori/scratch/SOPBench
RUNS=$REPO/reports/facet_rft_2026/phase4_distill/sft_runs
OUT=/home/woori/scratch/sft_alias_run
SUM=$OUT/RUNG1_RESULTS.txt
NB="dmv healthcare hotel library online_market university"
echo "=== rung1 (readiness-gate) train+eval start $(date) ===" > $SUM
rm -f /dev/shm/vllm* /dev/shm/nccl* 2>/dev/null

# 0) build readiness-gate LODO data (idempotent)
cd $SB
PYTHONPATH=$CLONE $PY build_tbox_planner_sft.py --out $OUT --data_dir $CLONE/data --ont_dir $CLONE/induced --scratchpad >> $SUM 2>&1
PYTHONPATH=$CLONE $PY build_tbox_planner_sft.py --out $OUT --data_dir $CLONE/data --ont_dir $CLONE/induced --alias --source 3 --scratchpad >> $SUM 2>&1
: > $OUT/lodo_train_s1_scratch.jsonl;        for d in $NB; do cat $OUT/sft_tbox_${d}_gate_scratch.jsonl          >> $OUT/lodo_train_s1_scratch.jsonl 2>/dev/null; done
: > $OUT/lodo_train_alias_s3_scratch.jsonl;  for d in $NB; do cat $OUT/sft_tbox_${d}_alias_s3_gate_scratch.jsonl >> $OUT/lodo_train_alias_s3_scratch.jsonl 2>/dev/null; done
wc -l $OUT/lodo_train_s1_scratch.jsonl $OUT/lodo_train_alias_s3_scratch.jsonl >> $SUM

# 1) train 2 regimes SOLO (GPU0 s1_scratch, GPU1 alias_s3_scratch) — no OOM
cd $REPO/scripts/distill
COMMON="--base-model Qwen/Qwen2.5-7B-Instruct --device cuda:0 --max-seq-len 2048 --epochs 3 --lora-r 16 --val-frac 0.05 --skip-overlong"
echo "=== train $(date) ===" >> $SUM
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup $PY $TR $COMMON \
  --train-jsonl $OUT/lodo_train_s1_scratch.jsonl --out-dir $RUNS/qwen7b_tbox_s1_scratch_lodo_bank > $OUT/train_s1_scratch.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup $PY $TR $COMMON \
  --train-jsonl $OUT/lodo_train_alias_s3_scratch.jsonl --out-dir $RUNS/qwen7b_tbox_alias_s3_scratch_lodo_bank > $OUT/train_alias_s3_scratch.log 2>&1 &
while true; do
  n=0; for r in s1 alias_s3; do [ -f $RUNS/qwen7b_tbox_${r}_scratch_lodo_bank/train_meta.json ] && n=$((n+1)); done
  echo "$(date) train_meta=$n/2 alive=$(pgrep -fc lora_train_chat_toolcall)" >> $SUM
  [ "$n" -ge 2 ] && break
  [ "$(pgrep -fc lora_train_chat_toolcall)" -eq 0 ] && { echo "trainers gone meta=$n" >> $SUM; break; }
  sleep 120
done

# 2) refresh clone client (readiness-gate aware), clean shm
$PY $SB/apply_two_stage_patch.py $CLONE >> $SUM 2>&1
rm -f /dev/shm/vllm* /dev/shm/nccl* 2>/dev/null

kill_gpu () { for p in $(nvidia-smi --id=$1 --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 $p 2>/dev/null; done; sleep 4; rm -f /dev/shm/vllm* /dev/shm/nccl* 2>/dev/null; }
eval_one () {
  local r=$1 gpu=$2 port=$3; shift 3; local envv="$*"
  local AD=$RUNS/qwen7b_tbox_${r}_scratch_lodo_bank
  echo "[$r] gpu=$gpu env=[GATE=1 SCRATCHPAD=1 $envv] $(date)" >> $SUM
  [ -f "$AD/adapter_model.safetensors" ] || { echo "[$r] NO adapter" >> $SUM; return; }
  CUDA_VISIBLE_DEVICES=$gpu nohup $VLLM serve Qwen/Qwen2.5-7B-Instruct --enable-lora --max-lora-rank 16 \
    --lora-modules tbox_v2=$AD --port $port --dtype bfloat16 --gpu-memory-utilization 0.85 \
    --max-model-len 8192 --enable-auto-tool-choice --tool-call-parser hermes --trust-remote-code \
    > $OUT/serve_${r}_scratch.log 2>&1 &
  for i in $(seq 1 120); do curl -s -m 3 http://localhost:$port/v1/models 2>/dev/null | grep -q tbox_v2 && break; sleep 4; done
  curl -s -m 3 http://localhost:$port/v1/models 2>/dev/null | grep -q tbox_v2 || { echo "[$r] SERVE FAILED" >> $SUM; kill_gpu $gpu; return; }
  cd $CLONE
  env SOPBENCH_GATE=1 SOPBENCH_SCRATCHPAD=1 $envv SOPBENCH_VLLM_BASE_URL=http://localhost:$port/v1 $PY run_simulation.py \
    --domain bank --assistant_model tbox_v2 --tool_call_mode fc --tool_list full \
    --two_stage --two_stage_v2 --ont_dir $CLONE/induced --output_dir $OUT/eval_${r}_scratch --env_mode prompt \
    > $OUT/sim_${r}_scratch.log 2>&1
  $PY run_evaluation.py --domain bank --assistant_model tbox_v2 --tool_call_mode fc --tool_list full \
    --output_dir $OUT/eval_${r}_scratch > $OUT/evalout_${r}_scratch.txt 2>&1
  echo "[$r] $(grep -E 'Mean Pass Rate' $OUT/evalout_${r}_scratch.txt | tail -1)" >> $SUM
  kill_gpu $gpu
}
eval_one s1       0 9001 &
eval_one alias_s3 1 9002 SOPBENCH_ALIAS=1 SOPBENCH_SOURCE=3 &
wait

echo "=== RUNG1 HEADLINE (should_T dirgraph+ INT goal+) $(date) ===" >> $SUM
$PY - >> $SUM 2>&1 <<'PYEOF'
import json,glob
def e0(x):
    e=x.get("evaluations"); return (e[0] if e else {}) if isinstance(e,list) else (e or {})
for r in ["s1","alias_s3"]:
    fs=glob.glob(f'/home/woori/scratch/sft_alias_run/eval_{r}_scratch/bank/*.json')
    if not fs: print(f"{r}_scratch: NO DATA"); continue
    d=json.load(open(fs[0])); T=[x for x in d if e0(x).get("action_should_succeed")]; F=[x for x in d if not e0(x).get("action_should_succeed")]
    both=sum(1 for x in T if e0(x).get("dirgraph_satisfied") and e0(x).get("action_called_correctly"))
    print(f"{r}_scratch: should_T succ={sum(bool(e0(x).get('success')) for x in T)}/48 "
          f"dirgraph={sum(bool(e0(x).get('dirgraph_satisfied')) for x in T)} "
          f"goal={sum(bool(e0(x).get('action_called_correctly')) for x in T)} "
          f"BOTH={both} || should_F={sum(bool(e0(x).get('success')) for x in F)}/86")
print("(baseline gate-token: s1_gate BOTH=2, alias_s3_gate BOTH=1)")
PYEOF
echo "=== rung1 DONE $(date) ===" >> $SUM
