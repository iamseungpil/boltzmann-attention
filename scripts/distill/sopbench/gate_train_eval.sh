#!/bin/bash
# gate_train_eval.sh — §8.6 Exp-4d: train the 3 gate-token regimes on 2 GPUs, then bank-eval each.
# Regimes: s1_gate (STATUS+gate), s3_gate (NL+gate), alias_s3_gate (alias+NL+gate).
# Terminal target = constant ACT/STOP (not the varying goal name) -> tests the act/STOP gate fix.
# Headline metric = should_T with dirgraph+ AND goal+ (co-occurrence), NOT total (total inflates via refusal).
set +e
PY=/home/woori/venvs/seka_env/bin/python
VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
REPO=/home/woori/workspace_common/boltzmann-attention-pi
SB=$REPO/scripts/distill/sopbench
TR=$REPO/scripts/distill/lora_train_chat_toolcall.py
CLONE=/home/woori/scratch/SOPBench
RUNS=$REPO/reports/facet_rft_2026/phase4_distill/sft_runs
OUT=/home/woori/scratch/sft_alias_run
SUM=$OUT/GATE_RESULTS.txt
NB="dmv healthcare hotel library online_market university"
echo "=== gate_train_eval start $(date) ===" > $SUM

# 0) build gate-token LODO data for all 3 regimes (idempotent)
cd $SB
build_one () {  # $1=tag(file suffix)  $2..=build flags
  local tag=$1; shift
  [ -f $OUT/lodo_train_${tag}.jsonl ] && { echo "data ${tag} exists" >> $SUM; return; }
  PYTHONPATH=$CLONE $PY build_tbox_planner_sft.py --out $OUT --data_dir $CLONE/data \
    --ont_dir $CLONE/induced "$@" >> $SUM 2>&1
}
build_one s1_gate       --gate-token
build_one s3_gate       --source 3 --gate-token
build_one alias_s3_gate --alias --source 3 --gate-token
# concat the 6 non-bank into LODO train files (per regime file suffix on each domain)
cat_lodo () { local out=$1 suf=$2; : > $OUT/$out; for d in $NB; do cat $OUT/sft_tbox_${d}${suf}.jsonl >> $OUT/$out 2>/dev/null; done; }
cat_lodo lodo_train_s1_gate.jsonl       _gate
cat_lodo lodo_train_s3_gate.jsonl       _s3_gate
cat_lodo lodo_train_alias_s3_gate.jsonl _alias_s3_gate
wc -l $OUT/lodo_train_s1_gate.jsonl $OUT/lodo_train_s3_gate.jsonl $OUT/lodo_train_alias_s3_gate.jsonl >> $SUM

# 1) train 3 gate regimes (GPU0: s1+s3 shared, GPU1: alias_s3 solo)
cd $REPO/scripts/distill
COMMON="--base-model Qwen/Qwen2.5-7B-Instruct --device cuda:0 --max-seq-len 2048 --epochs 3 --lora-r 16 --val-frac 0.05 --skip-overlong"
echo "=== train 3 gate regimes $(date) ===" >> $SUM
CUDA_VISIBLE_DEVICES=0 nohup $PY $TR $COMMON --train-jsonl $OUT/lodo_train_s1_gate.jsonl       --out-dir $RUNS/qwen7b_tbox_s1_gate_lodo_bank       > $OUT/train_s1_gate.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup $PY $TR $COMMON --train-jsonl $OUT/lodo_train_s3_gate.jsonl       --out-dir $RUNS/qwen7b_tbox_s3_gate_lodo_bank       > $OUT/train_s3_gate.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup $PY $TR $COMMON --train-jsonl $OUT/lodo_train_alias_s3_gate.jsonl --out-dir $RUNS/qwen7b_tbox_alias_s3_gate_lodo_bank > $OUT/train_alias_s3_gate.log 2>&1 &
# wait for all 3 train_meta.json (or trainers to die)
while true; do
  n=0; for r in s1 s3 alias_s3; do [ -f $RUNS/qwen7b_tbox_${r}_gate_lodo_bank/train_meta.json ] && n=$((n+1)); done
  alive=$($PY -c "import subprocess" 2>/dev/null; pgrep -fc lora_train_chat_toolcall)
  echo "$(date) train_meta=$n/3 alive=$alive" >> $SUM
  [ "$n" -ge 3 ] && break
  [ "${alive:-0}" -eq 0 ] && { echo "trainers gone meta=$n" >> $SUM; break; }
  sleep 120
done

# 2) refresh clone client (alias+gate aware), clean shm
$PY $SB/apply_two_stage_patch.py $CLONE >> $SUM 2>&1
rm -f /dev/shm/vllm* /dev/shm/nccl* 2>/dev/null

# 3) eval each gate regime (gate-aware env). Isolated: serve on own GPU+port, kill only that GPU.
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
# two in parallel on the 2 GPUs, then the third
eval_gate s1       0 9001 &
eval_gate alias_s3 1 9002 SOPBENCH_ALIAS=1 SOPBENCH_SOURCE=3 &
wait
eval_gate s3       0 9001 SOPBENCH_SOURCE=3

# 4) HEADLINE breakdown: should_T dirgraph+ AND goal+ (co-occurrence), per regime
echo "=== HEADLINE: should_T dirgraph+ INT goal+ (the real metric) $(date) ===" >> $SUM
$PY - >> $SUM 2>&1 <<'PYEOF'
import json,glob
def e0(x):
    e=x.get("evaluations"); return (e[0] if e else {}) if isinstance(e,list) else (e or {})
for r in ["s1","s3","alias_s3"]:
    fs=glob.glob(f'/home/woori/scratch/sft_alias_run/eval_{r}_gate/bank/*.json')
    if not fs: print(f"{r}_gate: NO DATA"); continue
    d=json.load(open(fs[0]))
    T=[x for x in d if e0(x).get("action_should_succeed")]; F=[x for x in d if not e0(x).get("action_should_succeed")]
    succ=sum(bool(e0(x).get("success")) for x in T)
    dg=sum(bool(e0(x).get("dirgraph_satisfied")) for x in T)
    ac=sum(bool(e0(x).get("action_called_correctly")) for x in T)
    both=sum(1 for x in T if e0(x).get("dirgraph_satisfied") and e0(x).get("action_called_correctly"))
    sf=sum(bool(e0(x).get("success")) for x in F)
    print(f"{r}_gate: should_T succ={succ}/48 | dirgraph={dg} goal={ac} BOTH(headline)={both} || should_F={sf}/86")
PYEOF
echo "=== gate_train_eval DONE $(date) ===" >> $SUM
