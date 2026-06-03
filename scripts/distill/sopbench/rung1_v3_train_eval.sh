#!/bin/bash
# rung1_v3_train_eval.sh — v3 grounded tree-eval derivation (--treeval) on top of T1/T2.
# 2 regimes SOLO: s1_treeval (GPU0, source1+scratch+treeval), alias_s3_treeval (GPU1, alias+s3+scratch+treeval).
# Headline = should_T BOTH(dirgraph∩goal) vs T1/T2 baseline (s1 4 / alias_s3 4). Tests whether grounding
# permitted as an AND/OR tree-eval over gathered truths lifts BOTH (over-refusal fix).
set +e
PY=/home/woori/venvs/seka_env/bin/python
VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
REPO=/home/woori/workspace_common/boltzmann-attention-pi
SB=$REPO/scripts/distill/sopbench
TR=$REPO/scripts/distill/lora_train_chat_toolcall.py
CLONE=/home/woori/scratch/SOPBench
RUNS=$REPO/reports/facet_rft_2026/phase4_distill/sft_runs
OUT=/home/woori/scratch/sft_alias_run
SUM=$OUT/RUNG1_V3_RESULTS.txt
NB="dmv healthcare hotel library online_market university"
echo "=== rung1 v3 (treeval) train+eval start $(date) ===" > $SUM
rm -f /dev/shm/vllm* /dev/shm/nccl* 2>/dev/null

# 0) build v3 LODO data (treeval). idempotent.
cd $SB
PYTHONPATH=$CLONE $PY build_tbox_planner_sft.py --out $OUT --data_dir $CLONE/data --ont_dir $CLONE/induced --treeval >> $SUM 2>&1
PYTHONPATH=$CLONE $PY build_tbox_planner_sft.py --out $OUT --data_dir $CLONE/data --ont_dir $CLONE/induced --alias --source 3 --treeval >> $SUM 2>&1
: > $OUT/lodo_train_s1_treeval.jsonl;        for d in $NB; do cat $OUT/sft_tbox_${d}_gate_scratch_treeval.jsonl          >> $OUT/lodo_train_s1_treeval.jsonl 2>/dev/null; done
: > $OUT/lodo_train_alias_s3_treeval.jsonl;  for d in $NB; do cat $OUT/sft_tbox_${d}_alias_s3_gate_scratch_treeval.jsonl >> $OUT/lodo_train_alias_s3_treeval.jsonl 2>/dev/null; done
wc -l $OUT/lodo_train_s1_treeval.jsonl $OUT/lodo_train_alias_s3_treeval.jsonl >> $SUM

# 1) train 2 regimes SOLO
cd $REPO/scripts/distill
COMMON="--base-model Qwen/Qwen2.5-7B-Instruct --device cuda:0 --max-seq-len 2048 --epochs 3 --lora-r 16 --val-frac 0.05 --skip-overlong"
rm -f $RUNS/qwen7b_tbox_s1_treeval_lodo_bank/train_meta.json $RUNS/qwen7b_tbox_alias_s3_treeval_lodo_bank/train_meta.json
echo "=== train $(date) ===" >> $SUM
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup $PY $TR $COMMON \
  --train-jsonl $OUT/lodo_train_s1_treeval.jsonl --out-dir $RUNS/qwen7b_tbox_s1_treeval_lodo_bank > $OUT/train_s1_treeval.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup $PY $TR $COMMON \
  --train-jsonl $OUT/lodo_train_alias_s3_treeval.jsonl --out-dir $RUNS/qwen7b_tbox_alias_s3_treeval_lodo_bank > $OUT/train_alias_s3_treeval.log 2>&1 &
while true; do
  n=0; for r in s1 alias_s3; do [ -f $RUNS/qwen7b_tbox_${r}_treeval_lodo_bank/train_meta.json ] && n=$((n+1)); done
  echo "$(date) train_meta=$n/2 alive=$(pgrep -fc lora_train_chat_toolcall)" >> $SUM
  [ "$n" -ge 2 ] && break
  [ "$(pgrep -fc lora_train_chat_toolcall)" -eq 0 ] && { echo "trainers gone meta=$n" >> $SUM; break; }
  sleep 120
done

# 2) refresh clone client, clean shm
$PY $SB/apply_two_stage_patch.py $CLONE >> $SUM 2>&1
rm -f /dev/shm/vllm* /dev/shm/nccl* 2>/dev/null

kill_gpu () { for p in $(nvidia-smi --id=$1 --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 $p 2>/dev/null; done; sleep 4; rm -f /dev/shm/vllm* /dev/shm/nccl* 2>/dev/null; }
eval_one () {
  local r=$1 gpu=$2 port=$3; shift 3; local envv="$*"
  local AD=$RUNS/qwen7b_tbox_${r}_treeval_lodo_bank
  echo "[$r] gpu=$gpu env=[GATE=1 SCRATCHPAD=1 $envv] $(date)" >> $SUM
  [ -f "$AD/adapter_model.safetensors" ] || { echo "[$r] NO adapter" >> $SUM; return; }
  rm -rf $OUT/eval_${r}_treeval
  local iport=$((8100 + gpu*200))
  CUDA_VISIBLE_DEVICES=$gpu VLLM_PORT=$iport VLLM_DP_MASTER_PORT=$((iport+50)) nohup $VLLM serve Qwen/Qwen2.5-7B-Instruct --enable-lora --max-lora-rank 16 \
    --lora-modules tbox_v2=$AD --port $port --dtype bfloat16 --gpu-memory-utilization 0.85 \
    --max-model-len 8192 --enable-auto-tool-choice --tool-call-parser hermes --trust-remote-code \
    > $OUT/serve_${r}_treeval.log 2>&1 &
  for i in $(seq 1 120); do curl -s -m 3 http://localhost:$port/v1/models 2>/dev/null | grep -q tbox_v2 && break; sleep 4; done
  curl -s -m 3 http://localhost:$port/v1/models 2>/dev/null | grep -q tbox_v2 || { echo "[$r] SERVE FAILED" >> $SUM; kill_gpu $gpu; return; }
  cd $CLONE
  env SOPBENCH_GATE=1 SOPBENCH_SCRATCHPAD=1 $envv SOPBENCH_VLLM_BASE_URL=http://localhost:$port/v1 $PY run_simulation.py \
    --domain bank --assistant_model tbox_v2 --tool_call_mode fc --tool_list full \
    --two_stage --two_stage_v2 --ont_dir $CLONE/induced --output_dir $OUT/eval_${r}_treeval --env_mode prompt \
    > $OUT/sim_${r}_treeval.log 2>&1
  $PY run_evaluation.py --domain bank --assistant_model tbox_v2 --tool_call_mode fc --tool_list full \
    --output_dir $OUT/eval_${r}_treeval > $OUT/evalout_${r}_treeval.txt 2>&1
  kill_gpu $gpu
}
eval_one s1       0 9001 &
eval_one alias_s3 1 9002 SOPBENCH_ALIAS=1 SOPBENCH_SOURCE=3 &
wait

echo "=== RUNG1 v3 HEADLINE (should_T dirgraph+ ∩ goal+) $(date) ===" >> $SUM
$PY - >> $SUM 2>&1 <<'PYEOF'
import json,glob,os,time
def e0(x):
    e=x.get("evaluations"); return (e[0] if e else {}) if isinstance(e,list) else (e or {})
RUNS="/home/woori/workspace_common/boltzmann-attention-pi/reports/facet_rft_2026/phase4_distill/sft_runs"
def seq(x):
    s=[]
    for it in (x.get("interactions") or []):
        for st in (it.get("interaction") or []):
            if st.get("tool_name"): s.append(st["tool_name"])
    return s
for r in ["s1","alias_s3"]:
    fs=glob.glob(f'/home/woori/scratch/sft_alias_run/eval_{r}_treeval/bank/*.json')
    if not fs: print(f"{r}_treeval: NO DATA"); continue
    ad=f"{RUNS}/qwen7b_tbox_{r}_treeval_lodo_bank/adapter_model.safetensors"
    if os.path.exists(ad) and os.path.getmtime(fs[0])<=os.path.getmtime(ad):
        print(f"{r}_treeval: STALE!! REJECT"); continue
    d=json.load(open(fs[0])); T=[x for x in d if e0(x).get("action_should_succeed")]; F=[x for x in d if not e0(x).get("action_should_succeed")]
    dg=sum(1 for x in T if e0(x).get("dirgraph_satisfied")); gc=sum(1 for x in T if e0(x).get("action_called_correctly"))
    both=sum(1 for x in T if e0(x).get("dirgraph_satisfied") and e0(x).get("action_called_correctly"))
    acted=sum(1 for x in T if e0(x).get("action_successfully_called"))
    stopF=sum(1 for x in F if e0(x).get("success"))
    print(f"{r}_treeval: should_T dirgraph={dg} acted={acted} goal={gc} BOTH={both}/48 || should_F STOP={stopF}/86")
print("(baseline T1/T2: s1 BOTH=4, alias_s3 BOTH=4)")
PYEOF
echo "=== rung1 v3 DONE $(date) ===" >> $SUM
cat $SUM
