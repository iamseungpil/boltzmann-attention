#!/bin/bash
# rung1_agent2_upperbound.sh — Agent2 UPPER-BOUND (source-effect) measurement, 2026-06-04.
# Design: RUNG1_SOURCE_LADDER_DESIGN.md §11-12 (Agent2@oracle = rung C). 1st gate of the whole thesis:
#   does providing the per-task STRUCTURE (source=1) lift BOTH toward the run_scripted oracle (37/48)?
# Two FRESH arms, BOTH tree-emit OFF (the tree-emit line is closed), BOTH on the bug-fixed client
# (_resolve no longer 400s when goal absent from tools):
#   A = rung A baseline   : --alias --source 3 --scratchpad  (T1T2 regime: structure INFERRED)
#   C = rung C / Agent2@oracle : --alias --source 1 --scratchpad (structure PROVIDED, anonymized dirgraph)
# Same two-gate target (ready=true; preconds_verified; permitted=should_succeed; ACT|STOP), no fabrication
# (no tree to emit). LODO holdout=bank, ep3, r16, SOLO, maxtok=1024. Metrics computed AFTER run_evaluation
# (freshness-guarded) to avoid the headline race that zeroed nt last run.
set +e
PLAN_MAXTOK=${PLAN_MAXTOK:-1024}
PY=/home/woori/venvs/seka_env/bin/python
VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
REPO=/home/woori/workspace_common/boltzmann-attention-pi
SB=$REPO/scripts/distill/sopbench
TR=$REPO/scripts/distill/lora_train_chat_toolcall.py
CLONE=/home/woori/scratch/SOPBench
RUNS=$REPO/reports/facet_rft_2026/phase4_distill/sft_runs
OUT=/home/woori/scratch/sft_alias_run
SUM=$OUT/RUNG1_UPPERBOUND_RESULTS.txt
NB="dmv healthcare hotel library online_market university"
echo "=== rung1 Agent2 upper-bound (A=s3 / C=s1) start $(date) ===" > $SUM
rm -f /dev/shm/vllm* /dev/shm/nccl* 2>/dev/null

# 0) build teachers (no treeval). C: source=1 -> tag has NO _s suffix (=_alias_gate_scratch). A: _alias_s3_gate_scratch.
cd $SB
PYTHONPATH=$CLONE $PY build_tbox_planner_sft.py --out $OUT --data_dir $CLONE/data --ont_dir $CLONE/induced --alias --source 1 --scratchpad >> $SUM 2>&1
PYTHONPATH=$CLONE $PY build_tbox_planner_sft.py --out $OUT --data_dir $CLONE/data --ont_dir $CLONE/induced --alias --source 3 --scratchpad >> $SUM 2>&1
: > $OUT/lodo_train_alias_s1_ub.jsonl;  for d in $NB; do cat $OUT/sft_tbox_${d}_alias_gate_scratch.jsonl    >> $OUT/lodo_train_alias_s1_ub.jsonl 2>/dev/null; done
: > $OUT/lodo_train_alias_s3_ub.jsonl;  for d in $NB; do cat $OUT/sft_tbox_${d}_alias_s3_gate_scratch.jsonl >> $OUT/lodo_train_alias_s3_ub.jsonl 2>/dev/null; done
wc -l $OUT/lodo_train_alias_s1_ub.jsonl $OUT/lodo_train_alias_s3_ub.jsonl >> $SUM

# 1) train C(s1, GPU0) + A(s3, GPU1) SOLO
cd $REPO/scripts/distill
COMMON="--base-model Qwen/Qwen2.5-7B-Instruct --device cuda:0 --max-seq-len 2048 --epochs 3 --lora-r 16 --val-frac 0.05 --skip-overlong"
rm -f $RUNS/qwen7b_tbox_ub_s1_lodo_bank/train_meta.json $RUNS/qwen7b_tbox_ub_s3_lodo_bank/train_meta.json
echo "=== train $(date) ===" >> $SUM
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup $PY $TR $COMMON \
  --train-jsonl $OUT/lodo_train_alias_s1_ub.jsonl --out-dir $RUNS/qwen7b_tbox_ub_s1_lodo_bank > $OUT/train_ub_s1.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup $PY $TR $COMMON \
  --train-jsonl $OUT/lodo_train_alias_s3_ub.jsonl --out-dir $RUNS/qwen7b_tbox_ub_s3_lodo_bank > $OUT/train_ub_s3.log 2>&1 &
while true; do
  n=0; for r in s1 s3; do [ -f $RUNS/qwen7b_tbox_ub_${r}_lodo_bank/train_meta.json ] && n=$((n+1)); done
  echo "$(date) train_meta=$n/2 alive=$(pgrep -fc lora_train_chat_toolcall)" >> $SUM
  [ "$n" -ge 2 ] && break
  [ "$(pgrep -fc lora_train_chat_toolcall)" -eq 0 ] && { echo "trainers gone meta=$n" >> $SUM; break; }
  sleep 180
done

# 2) eval: C at SOPBENCH_SOURCE=1, A at SOPBENCH_SOURCE=3. maxtok=1024. RLLOG. (bug-fixed client.)
cd $SB; $PY $SB/apply_two_stage_patch.py $CLONE >> $SUM 2>&1
rm -f /dev/shm/vllm* /dev/shm/nccl* 2>/dev/null
kill_gpu () { for p in $(nvidia-smi --id=$1 --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 $p 2>/dev/null; done; sleep 4; rm -f /dev/shm/vllm* /dev/shm/nccl* 2>/dev/null; }
eval_one () {  # tag adapter gpu port source
  local tag=$1 ad=$2 gpu=$3 port=$4 src=$5
  local AD=$RUNS/$ad
  echo "[$tag] adapter=$ad gpu=$gpu src=$src maxtok=$PLAN_MAXTOK $(date)" >> $SUM
  [ -f "$AD/adapter_model.safetensors" ] || { echo "[$tag] NO adapter" >> $SUM; return; }
  rm -rf $OUT/eval_ub_${tag}; rm -f $OUT/rllog_ub_${tag}.jsonl
  local iport=$((8300 + gpu*200))
  CUDA_VISIBLE_DEVICES=$gpu VLLM_PORT=$iport VLLM_DP_MASTER_PORT=$((iport+50)) nohup $VLLM serve Qwen/Qwen2.5-7B-Instruct --enable-lora --max-lora-rank 16 \
    --lora-modules tbox_v2=$AD --port $port --dtype bfloat16 --gpu-memory-utilization 0.85 \
    --max-model-len 8192 --enable-auto-tool-choice --tool-call-parser hermes --trust-remote-code \
    > $OUT/serve_ub_${tag}.log 2>&1 &
  for i in $(seq 1 120); do curl -s -m 3 http://localhost:$port/v1/models 2>/dev/null | grep -q tbox_v2 && break; sleep 4; done
  curl -s -m 3 http://localhost:$port/v1/models 2>/dev/null | grep -q tbox_v2 || { echo "[$tag] SERVE FAILED" >> $SUM; kill_gpu $gpu; return; }
  cd $CLONE
  env SOPBENCH_GATE=1 SOPBENCH_SCRATCHPAD=1 SOPBENCH_ALIAS=1 SOPBENCH_SOURCE=$src SOPBENCH_PLAN_MAXTOK=$PLAN_MAXTOK \
      SOPBENCH_RLLOG=$OUT/rllog_ub_${tag}.jsonl SOPBENCH_VLLM_BASE_URL=http://localhost:$port/v1 $PY run_simulation.py \
    --domain bank --assistant_model tbox_v2 --tool_call_mode fc --tool_list full \
    --two_stage --two_stage_v2 --ont_dir $CLONE/induced --output_dir $OUT/eval_ub_${tag} --env_mode prompt \
    > $OUT/sim_ub_${tag}.log 2>&1
  $PY run_evaluation.py --domain bank --assistant_model tbox_v2 --tool_call_mode fc --tool_list full \
    --output_dir $OUT/eval_ub_${tag} > $OUT/evalout_ub_${tag}.txt 2>&1
  kill_gpu $gpu
}
eval_one C qwen7b_tbox_ub_s1_lodo_bank 0 8301 1 &
eval_one A qwen7b_tbox_ub_s3_lodo_bank 1 8302 3 &
wait

echo "=== RUNG1 UPPER-BOUND HEADLINE (maxtok=$PLAN_MAXTOK) $(date) ===" >> $SUM
$PY - >> $SUM 2>&1 <<'PYEOF'
import json,glob,collections,re
def e0(x):
    e=x.get("evaluations"); return (e[0] if e else {}) if isinstance(e,list) else (e or {})
def seq(x):
    s=[]
    for it in (x.get("interactions") or []):
        for stp in (it.get("interaction") or []):
            tc=stp.get("tool_calls")
            if tc:
                for c in tc:
                    fn=(c.get("function") or {}).get("name")
                    if fn: s.append(fn)
    return s
def nleaf(t):
    if not t or not isinstance(t,(list,tuple)) or not t: return 0
    if t[0]=="single": return 1
    if t[0] in ("and","or","chain","gate"): return sum(nleaf(s) for s in t[1])
    return 0
for tag in ["A","C"]:
    fs=glob.glob(f'/home/woori/scratch/sft_alias_run/eval_ub_{tag}/bank/*.json')
    if not fs: print(f"{tag}: NO DATA"); continue
    d=json.load(open(fs[0]))
    T=[x for x in d if e0(x).get("action_should_succeed")]; F=[x for x in d if not e0(x).get("action_should_succeed")]
    if not T: print(f"{tag}: evaluations EMPTY (race?) n={len(d)}"); continue
    dg=sum(1 for x in T if e0(x).get("dirgraph_satisfied")); gc=sum(1 for x in T if e0(x).get("action_called_correctly"))
    both=sum(1 for x in T if e0(x).get("dirgraph_satisfied") and e0(x).get("action_called_correctly"))
    acted=sum(1 for x in T if e0(x).get("action_successfully_called"))
    noact=sum(1 for x in T if not e0(x).get("action_successfully_called") and e0(x).get("user_goal") not in seq(x))
    prem=sum(1 for x in T if e0(x).get("action_successfully_called") and not e0(x).get("dirgraph_satisfied"))
    loop=sum(1 for x in T if (max(collections.Counter(seq(x)).values()) if seq(x) else 0)>=3)
    stepmed=sorted(len(seq(x)) for x in T)[len(T)//2]
    stopF=sum(1 for x in F if e0(x).get("success"))
    actrec=both/dg if dg else 0
    by=collections.defaultdict(lambda:[0,0])
    for x in T:
        n=nleaf((x.get("task") or {}).get("constraints")); b=1 if (e0(x).get("dirgraph_satisfied") and e0(x).get("action_called_correctly")) else 0
        by[n][0]+=b; by[n][1]+=1
    cc=" ".join(f"{n}c:{v[0]}/{v[1]}" for n,v in sorted(by.items()))
    print(f"== {tag} (n_T={len(T)} n_F={len(F)}) ==")
    print(f"  should_T48: dirgraph={dg} acted={acted} goal={gc} BOTH={both} | ACT-recall|gather={actrec:.2f} over-refuse={noact} premature={prem} loop>=3={loop} step_med={stepmed}")
    print(f"  should_F86: STOP-recall={stopF} ({100*stopF/max(1,len(F)):.0f}%)")
    print(f"  cond-count BOTH: {cc}")
print("(baseline: T1T2 BOTH 4 | maxtok-retest nt/treeval BOTH 5 STOP 42% | run_scripted oracle 37/48)")
PYEOF
echo "=== rung1 upper-bound DONE $(date) ===" >> $SUM
cat $SUM
