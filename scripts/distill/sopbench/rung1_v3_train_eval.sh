#!/bin/bash
# rung1_v3_train_eval.sh — CONTROLLED A/B on headline regime (alias_s3, source=3), both key-fixed:
#   CONTROL = non-treeval (corrected T1/T2 teacher, should_succeed terminal)  -> GPU0
#   V3      = treeval (grounded AND/OR tree-eval derivation terminal)          -> GPU1
# Isolates the v3 (grounded-permitted derivation) effect from T1/T2. Both trained on the SAME
# key-fixed teacher data, differing ONLY in --treeval. Metrics = ACT-recall|gather + STOP-recall
# separated (Mean Pass Rate alone forbidden). LODO holdout = bank.
set +e
PY=/home/woori/venvs/seka_env/bin/python
VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
REPO=/home/woori/workspace_common/boltzmann-attention-pi
SB=$REPO/scripts/distill/sopbench
TR=$REPO/scripts/distill/lora_train_chat_toolcall.py
CLONE=/home/woori/scratch/SOPBench
RUNS=$REPO/reports/facet_rft_2026/phase4_distill/sft_runs
OUT=/home/woori/scratch/sft_alias_run
SUM=$OUT/RUNG1_V3_AB_RESULTS.txt
NB="dmv healthcare hotel library online_market university"
echo "=== rung1 v3 A/B (alias_s3, key-fixed) start $(date) ===" > $SUM
rm -f /dev/shm/vllm* /dev/shm/nccl* 2>/dev/null

# 0) build both teachers (alias_s3). idempotent. control=non-treeval, v3=treeval.
cd $SB
PYTHONPATH=$CLONE $PY build_tbox_planner_sft.py --out $OUT --data_dir $CLONE/data --ont_dir $CLONE/induced --alias --source 3 >> $SUM 2>&1
PYTHONPATH=$CLONE $PY build_tbox_planner_sft.py --out $OUT --data_dir $CLONE/data --ont_dir $CLONE/induced --alias --source 3 --treeval >> $SUM 2>&1
: > $OUT/lodo_train_alias_s3_nt.jsonl;      for d in $NB; do cat $OUT/sft_tbox_${d}_alias_s3_gate_scratch.jsonl          >> $OUT/lodo_train_alias_s3_nt.jsonl 2>/dev/null; done
: > $OUT/lodo_train_alias_s3_treeval.jsonl; for d in $NB; do cat $OUT/sft_tbox_${d}_alias_s3_gate_scratch_treeval.jsonl >> $OUT/lodo_train_alias_s3_treeval.jsonl 2>/dev/null; done
wc -l $OUT/lodo_train_alias_s3_nt.jsonl $OUT/lodo_train_alias_s3_treeval.jsonl >> $SUM
# grounded vs fallback ratio in v3 train data (sanity)
echo "v3 grounded(gate =) vs fallback(emit permitted= in ASSISTANT): " >> $SUM
$PY - >> $SUM 2>&1 <<'PYC'
import json,glob
g=f=0
for fn in glob.glob('/home/woori/scratch/sft_alias_run/sft_tbox_*_alias_s3_gate_scratch_treeval.jsonl'):
    for L in open(fn):
        a=json.loads(L)["messages"][1]["content"]
        if a.startswith("ready=true; gate ="): g+=1
        elif a.startswith("ready=true; preconds_verified"): f+=1
print(f"  grounded={g} fallback={f} ({100*g/max(1,g+f):.1f}% grounded)")
PYC

# 1) train control(GPU0) + v3(GPU1) SOLO
cd $REPO/scripts/distill
COMMON="--base-model Qwen/Qwen2.5-7B-Instruct --device cuda:0 --max-seq-len 2048 --epochs 3 --lora-r 16 --val-frac 0.05 --skip-overlong"
rm -f $RUNS/qwen7b_tbox_alias_s3_nt_lodo_bank/train_meta.json $RUNS/qwen7b_tbox_alias_s3_treeval_lodo_bank/train_meta.json
echo "=== train $(date) ===" >> $SUM
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup $PY $TR $COMMON \
  --train-jsonl $OUT/lodo_train_alias_s3_nt.jsonl --out-dir $RUNS/qwen7b_tbox_alias_s3_nt_lodo_bank > $OUT/train_alias_s3_nt.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup $PY $TR $COMMON \
  --train-jsonl $OUT/lodo_train_alias_s3_treeval.jsonl --out-dir $RUNS/qwen7b_tbox_alias_s3_treeval_lodo_bank > $OUT/train_alias_s3_treeval.log 2>&1 &
while true; do
  n=0; for r in nt treeval; do [ -f $RUNS/qwen7b_tbox_alias_s3_${r}_lodo_bank/train_meta.json ] && n=$((n+1)); done
  echo "$(date) train_meta=$n/2 alive=$(pgrep -fc lora_train_chat_toolcall)" >> $SUM
  [ "$n" -ge 2 ] && break
  [ "$(pgrep -fc lora_train_chat_toolcall)" -eq 0 ] && { echo "trainers gone meta=$n" >> $SUM; break; }
  sleep 120
done

# 2) refresh clone client, clean shm
$PY $SB/apply_two_stage_patch.py $CLONE >> $SUM 2>&1
rm -f /dev/shm/vllm* /dev/shm/nccl* 2>/dev/null

kill_gpu () { for p in $(nvidia-smi --id=$1 --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 $p 2>/dev/null; done; sleep 4; rm -f /dev/shm/vllm* /dev/shm/nccl* 2>/dev/null; }
eval_one () {  # arm gpu port
  local arm=$1 gpu=$2 port=$3
  local AD=$RUNS/qwen7b_tbox_alias_s3_${arm}_lodo_bank
  echo "[$arm] gpu=$gpu $(date)" >> $SUM
  [ -f "$AD/adapter_model.safetensors" ] || { echo "[$arm] NO adapter" >> $SUM; return; }
  rm -rf $OUT/eval_alias_s3_${arm}
  local iport=$((8100 + gpu*200))
  CUDA_VISIBLE_DEVICES=$gpu VLLM_PORT=$iport VLLM_DP_MASTER_PORT=$((iport+50)) nohup $VLLM serve Qwen/Qwen2.5-7B-Instruct --enable-lora --max-lora-rank 16 \
    --lora-modules tbox_v2=$AD --port $port --dtype bfloat16 --gpu-memory-utilization 0.85 \
    --max-model-len 8192 --enable-auto-tool-choice --tool-call-parser hermes --trust-remote-code \
    > $OUT/serve_alias_s3_${arm}.log 2>&1 &
  for i in $(seq 1 120); do curl -s -m 3 http://localhost:$port/v1/models 2>/dev/null | grep -q tbox_v2 && break; sleep 4; done
  curl -s -m 3 http://localhost:$port/v1/models 2>/dev/null | grep -q tbox_v2 || { echo "[$arm] SERVE FAILED" >> $SUM; kill_gpu $gpu; return; }
  cd $CLONE
  env SOPBENCH_GATE=1 SOPBENCH_SCRATCHPAD=1 SOPBENCH_ALIAS=1 SOPBENCH_SOURCE=3 SOPBENCH_VLLM_BASE_URL=http://localhost:$port/v1 $PY run_simulation.py \
    --domain bank --assistant_model tbox_v2 --tool_call_mode fc --tool_list full \
    --two_stage --two_stage_v2 --ont_dir $CLONE/induced --output_dir $OUT/eval_alias_s3_${arm} --env_mode prompt \
    > $OUT/sim_alias_s3_${arm}.log 2>&1
  $PY run_evaluation.py --domain bank --assistant_model tbox_v2 --tool_call_mode fc --tool_list full \
    --output_dir $OUT/eval_alias_s3_${arm} > $OUT/evalout_alias_s3_${arm}.txt 2>&1
  kill_gpu $gpu
}
eval_one nt      0 9001 &
eval_one treeval 1 9002 &
wait

echo "=== RUNG1 v3 A/B HEADLINE (alias_s3, 분리 지표) $(date) ===" >> $SUM
$PY - >> $SUM 2>&1 <<'PYEOF'
import json,glob,os
def e0(x):
    e=x.get("evaluations"); return (e[0] if e else {}) if isinstance(e,list) else (e or {})
def seq(x):
    s=[]
    for it in (x.get("interactions") or []):
        for st in (it.get("interaction") or []):
            if st.get("tool_name"): s.append(st["tool_name"])
    return s
for arm in ["nt","treeval"]:
    fs=glob.glob(f'/home/woori/scratch/sft_alias_run/eval_alias_s3_{arm}/bank/*.json')
    if not fs: print(f"{arm}: NO DATA"); continue
    d=json.load(open(fs[0])); T=[x for x in d if e0(x).get("action_should_succeed")]; F=[x for x in d if not e0(x).get("action_should_succeed")]
    dg=sum(1 for x in T if e0(x).get("dirgraph_satisfied")); gc=sum(1 for x in T if e0(x).get("action_called_correctly"))
    both=sum(1 for x in T if e0(x).get("dirgraph_satisfied") and e0(x).get("action_called_correctly"))
    acted=sum(1 for x in T if e0(x).get("action_successfully_called"))
    # ACT-recall | gather: among should_T that ACTED, fraction with dirgraph too = BOTH/acted; and over-refuse = not acted
    noact=sum(1 for x in T if not e0(x).get("action_successfully_called") and e0(x).get("user_goal") not in seq(x))
    prem=acted-both if acted>=both else 0
    stopF=sum(1 for x in F if e0(x).get("success"))
    actrecall = both/dg if dg else 0.0
    print(f"== alias_s3_{arm} ==")
    print(f"  should_T(48): dirgraph={dg} acted={acted} goal={gc} BOTH={both} | ACT-recall|gather={actrecall:.2f} over-refuse(noact)={noact} premature={prem}")
    print(f"  should_F(86): STOP-recall={stopF} ({100*stopF/86:.0f}%)")
print("(baseline T1/T2 alias_s3: BOTH=4, over-refuse~30, STOP~46)")
PYEOF
echo "=== rung1 v3 A/B DONE $(date) ===" >> $SUM
cat $SUM
