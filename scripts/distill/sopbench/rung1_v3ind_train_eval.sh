#!/bin/bash
# rung1_v3ind_train_eval.sh — v3-INDUCTIVE A/B (litreview §4 prescription, 2026-06-03).
# Exp-4-rung1-v3-AB retest (maxtok=1024) found the single-step grounded treeval CONVERGES but gives
# BOTH==control (5) = single-step whole-expression globality limit (Abbe/Feng/Kim&Suzuki predict this).
# This trains the INDUCTIVE teacher (bottom-up reduction chain: each internal AND/OR node folded as a
# supervised step) and evals it vs the EXISTING single-step treeval adapter, BOTH at PLAN_MAXTOK=1024
# (no truncation). Hypothesis: inductive derivation lifts BOTH above the single-step ceiling.
# control = existing qwen7b_tbox_alias_s3_treeval_lodo_bank (single-step, byte-identical teacher data
# since the single-step path was unchanged by this edit). Only the inductive arm is (re)trained.
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
SUM=$OUT/RUNG1_V3IND_RESULTS.txt
NB="dmv healthcare hotel library online_market university"
echo "=== rung1 v3-inductive (alias_s3) start $(date) ===" > $SUM
rm -f /dev/shm/vllm* /dev/shm/nccl* 2>/dev/null

# 0) build inductive teacher (alias_s3). idempotent. files: sft_tbox_<d>_alias_s3_gate_scratch_treevalind.jsonl
cd $SB
PYTHONPATH=$CLONE $PY build_tbox_planner_sft.py --out $OUT --data_dir $CLONE/data --ont_dir $CLONE/induced --alias --source 3 --treeval_inductive >> $SUM 2>&1
: > $OUT/lodo_train_alias_s3_treevalind.jsonl
for d in $NB; do cat $OUT/sft_tbox_${d}_alias_s3_gate_scratch_treevalind.jsonl >> $OUT/lodo_train_alias_s3_treevalind.jsonl 2>/dev/null; done
wc -l $OUT/lodo_train_alias_s3_treevalind.jsonl >> $SUM
# sanity: inductive grounded(reduction chain "t1=") vs fallback(permitted=) ratio + a few samples
$PY - >> $SUM 2>&1 <<'PYC'
import json,glob
g=f=0; samp=[]
for fn in glob.glob('/home/woori/scratch/sft_alias_run/sft_tbox_*_alias_s3_gate_scratch_treevalind.jsonl'):
    for L in open(fn):
        a=json.loads(L)["messages"][1]["content"]
        if not a.startswith("ready=true"): continue
        if a.startswith("ready=true; preconds_verified"): f+=1            # fallback (non-grounded permitted)
        elif a.startswith("ready=true; done"): pass                       # T2 post-success exit (not a gate)
        elif "gate=" in a: g+=1; (samp.append(a) if len(samp)<6 else None) # grounded chain incl. single-leaf
print(f"  inductive grounded={g} fallback={f} ({100*g/max(1,g+f):.1f}% grounded)  [single-leaf groundeds now counted]")
for s in samp: print("   SAMPLE:", s[:220])
PYC

# 1) train inductive adapter (GPU0 solo). control(single-step treeval) already trained at this data state.
cd $REPO/scripts/distill
COMMON="--base-model Qwen/Qwen2.5-7B-Instruct --device cuda:0 --max-seq-len 2048 --epochs 3 --lora-r 16 --val-frac 0.05 --skip-overlong"
rm -f $RUNS/qwen7b_tbox_alias_s3_treevalind_lodo_bank/train_meta.json
echo "=== train $(date) ===" >> $SUM
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup $PY $TR $COMMON \
  --train-jsonl $OUT/lodo_train_alias_s3_treevalind.jsonl --out-dir $RUNS/qwen7b_tbox_alias_s3_treevalind_lodo_bank > $OUT/train_alias_s3_treevalind.log 2>&1 &
while true; do
  [ -f $RUNS/qwen7b_tbox_alias_s3_treevalind_lodo_bank/train_meta.json ] && { echo "$(date) train DONE" >> $SUM; break; }
  [ "$(pgrep -fc lora_train_chat_toolcall)" -eq 0 ] && { echo "trainer gone, no meta" >> $SUM; break; }
  echo "$(date) training... alive=$(pgrep -fc lora_train_chat_toolcall) $(tail -1 $OUT/train_alias_s3_treevalind.log 2>/dev/null)" >> $SUM
  sleep 180
done

# 2) refresh clone client, eval treeval(existing single-step) + treevalind(new) at PLAN_MAXTOK
cd $SB; $PY $SB/apply_two_stage_patch.py $CLONE >> $SUM 2>&1
rm -f /dev/shm/vllm* /dev/shm/nccl* 2>/dev/null
kill_gpu () { for p in $(nvidia-smi --id=$1 --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 $p 2>/dev/null; done; sleep 4; rm -f /dev/shm/vllm* /dev/shm/nccl* 2>/dev/null; }
eval_one () {  # arm gpu port
  local arm=$1 gpu=$2 port=$3
  local AD=$RUNS/qwen7b_tbox_alias_s3_${arm}_lodo_bank
  echo "[$arm] gpu=$gpu maxtok=$PLAN_MAXTOK $(date)" >> $SUM
  [ -f "$AD/adapter_model.safetensors" ] || { echo "[$arm] NO adapter" >> $SUM; return; }
  rm -rf $OUT/eval_ind_${arm}; rm -f $OUT/rllog_ind_${arm}.jsonl
  local iport=$((8900 + gpu*200))
  CUDA_VISIBLE_DEVICES=$gpu VLLM_PORT=$iport VLLM_DP_MASTER_PORT=$((iport+50)) nohup $VLLM serve Qwen/Qwen2.5-7B-Instruct --enable-lora --max-lora-rank 16 \
    --lora-modules tbox_v2=$AD --port $port --dtype bfloat16 --gpu-memory-utilization 0.85 \
    --max-model-len 8192 --enable-auto-tool-choice --tool-call-parser hermes --trust-remote-code \
    > $OUT/serve_ind_${arm}.log 2>&1 &
  for i in $(seq 1 120); do curl -s -m 3 http://localhost:$port/v1/models 2>/dev/null | grep -q tbox_v2 && break; sleep 4; done
  curl -s -m 3 http://localhost:$port/v1/models 2>/dev/null | grep -q tbox_v2 || { echo "[$arm] SERVE FAILED" >> $SUM; kill_gpu $gpu; return; }
  cd $CLONE
  env SOPBENCH_GATE=1 SOPBENCH_SCRATCHPAD=1 SOPBENCH_ALIAS=1 SOPBENCH_SOURCE=3 SOPBENCH_PLAN_MAXTOK=$PLAN_MAXTOK \
      SOPBENCH_RLLOG=$OUT/rllog_ind_${arm}.jsonl SOPBENCH_VLLM_BASE_URL=http://localhost:$port/v1 $PY run_simulation.py \
    --domain bank --assistant_model tbox_v2 --tool_call_mode fc --tool_list full \
    --two_stage --two_stage_v2 --ont_dir $CLONE/induced --output_dir $OUT/eval_ind_${arm} --env_mode prompt \
    > $OUT/sim_ind_${arm}.log 2>&1
  $PY run_evaluation.py --domain bank --assistant_model tbox_v2 --tool_call_mode fc --tool_list full \
    --output_dir $OUT/eval_ind_${arm} > $OUT/evalout_ind_${arm}.txt 2>&1
  kill_gpu $gpu
}
# 3-way, all FRESH this run (reviewer #3: don't carry nt as a hardcoded constant — re-eval to
# control env drift). Round 1 (2 GPUs): inductive + single-step treeval. Round 2: non-treeval control.
eval_one treevalind 0 8901 &
eval_one treeval    1 8902 &
wait
eval_one nt 0 8901
wait

echo "=== RUNG1 v3-INDUCTIVE HEADLINE (maxtok=$PLAN_MAXTOK) $(date) ===" >> $SUM
$PY - >> $SUM 2>&1 <<'PYEOF'
import json,glob,re,collections
def e0(x):
    e=x.get("evaluations"); return (e[0] if e else {}) if isinstance(e,list) else (e or {})
def seq(x):
    s=[]
    for it in (x.get("interactions") or []):
        for st in (it.get("interaction") or []):
            tc=st.get("tool_calls")
            if tc:
                for c in tc:
                    fn=(c.get("function") or {}).get("name")
                    if fn: s.append(fn)
    return s
def chain_last_val(o):
    # final fold/leaf value emitted just before 'gate=' in a model chain output
    segs=[s.strip() for s in o.split(";")]
    gi=[i for i,s in enumerate(segs) if s.startswith("gate=")]
    if not gi: return None
    m=re.search(r"=(true|false|unknown)\s*$", segs[gi[-1]-1]) if gi[-1]>0 else None
    return m.group(1) if m else None
for arm in ["nt","treeval","treevalind"]:
    fs=glob.glob(f'/home/woori/scratch/sft_alias_run/eval_ind_{arm}/bank/*.json')
    if not fs: print(f"{arm}: NO DATA"); continue
    d=json.load(open(fs[0])); T=[x for x in d if e0(x).get("action_should_succeed")]; F=[x for x in d if not e0(x).get("action_should_succeed")]
    dg=sum(1 for x in T if e0(x).get("dirgraph_satisfied")); gc=sum(1 for x in T if e0(x).get("action_called_correctly"))
    both=sum(1 for x in T if e0(x).get("dirgraph_satisfied") and e0(x).get("action_called_correctly"))
    acted=sum(1 for x in T if e0(x).get("action_successfully_called"))
    noact=sum(1 for x in T if not e0(x).get("action_successfully_called") and e0(x).get("user_goal") not in seq(x))
    loop=sum(1 for x in T if (max(collections.Counter(seq(x)).values()) if seq(x) else 0)>=3)
    prem=sum(1 for x in T if e0(x).get("action_successfully_called") and not e0(x).get("dirgraph_satisfied"))  # acted AND NOT dirgraph (direct)
    stopF=sum(1 for x in F if e0(x).get("success"))
    actrec = both/dg if dg else 0.0
    print(f"== {arm} ==")
    print(f"  should_T(48): dirgraph={dg} acted={acted} goal={gc} BOTH={both} | ACT-recall|gather={actrec:.2f} over-refuse(noact)={noact} premature(acted&!dg)={prem} loop>=3={loop}")
    print(f"  should_F(86): STOP-recall={stopF} ({100*stopF/86:.0f}%)")
    # RLLOG MODEL-OUTPUT census (reviewer #4): format mixing + chain final-value vs decision consistency
    try:
        rows=[json.loads(l) for l in open(f'/home/woori/scratch/sft_alias_run/rllog_ind_{arm}.jsonl')]
        ready=[r["output"].strip() for r in rows if "ready=true" in r["output"].lower()]
        reached=sum(1 for o in ready if re.search(r'\b(ACT|STOP)\s*$',o))
        chain=[o for o in ready if "gate=" in o and "preconds_verified" not in o]
        perm =[o for o in ready if "preconds_verified" in o]
        inc=0
        for o in chain:
            cv=chain_last_val(o); dec=("ACT" if re.search(r'\bACT\s*$',o) else ("STOP" if re.search(r'\bSTOP\s*$',o) else "?"))
            if cv is None or dec=="?": continue
            if not ((cv=="true" and dec=="ACT") or (cv=="false" and dec=="STOP")): inc+=1
        print(f"  [rllog] terminal-attempts={len(ready)} reached={reached} ({100*reached/len(ready) if ready else 0:.0f}%) | chain-fmt={len(chain)} permitted-fmt={len(perm)} | chain final!=decision={inc}")
    except Exception as e: print(f"  [rllog] err {e}")
print("(@maxtok1024 prior, FRESH baselines re-run this run: nt STOP-recall=42% is the STOP bar; single-step treeval BOTH~5)")
PYEOF
echo "=== rung1 v3-inductive DONE $(date) ===" >> $SUM
cat $SUM
