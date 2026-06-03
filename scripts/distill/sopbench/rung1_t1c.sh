#!/bin/bash
# rung1_t1c.sh — T1c = grounded-permitted @ source=1 (treeval@s1), 2026-06-04.
# Design: RUNG1_T1C_DESIGN.md. Motivation: upperbound census = gathered_then_REFUSE 29 (permitted
# cold-collapse after complete gather). T1c = run the EXISTING treeval grounded-gate at source=1
# (structure provided -> blocks the s3 fabrication that sank v3). Only NEW training = T1c arm.
# 2x2 {none, treeval} x {s3, s1}: T1c trains; C-none(ub_s1)/A(ub_s3) eval REUSED (no re-eval);
# treeval@s3 RE-EVAL on bug-fixed client (old BOTH4 was n_T=45 under tool_choice bug). Pre-check PASSED
# (ceiling ~34). Pre-registered: success BOTH>=12, strong>=20, partial 6-11, null<=5.
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
SUM=$OUT/RUNG1_T1C_RESULTS.txt
NB="dmv healthcare hotel library online_market university"
echo "=== rung1 T1c (treeval@s1) start $(date) ===" > $SUM
rm -f /dev/shm/vllm* /dev/shm/nccl* 2>/dev/null

# 0) build T1c teacher = --source 1 --treeval (NO inductive). tag = _alias_gate_scratch_treeval (source1 -> no _s).
cd $SB
PYTHONPATH=$CLONE $PY build_tbox_planner_sft.py --out $OUT --data_dir $CLONE/data --ont_dir $CLONE/induced --alias --source 1 --treeval >> $SUM 2>&1
: > $OUT/lodo_train_t1c.jsonl; for d in $NB; do cat $OUT/sft_tbox_${d}_alias_gate_scratch_treeval.jsonl >> $OUT/lodo_train_t1c.jsonl 2>/dev/null; done
wc -l $OUT/lodo_train_t1c.jsonl >> $SUM

# 1) train T1c (GPU0 solo). C-none(ub_s1)/A(ub_s3) already trained+evaled (reuse).
cd $REPO/scripts/distill
COMMON="--base-model Qwen/Qwen2.5-7B-Instruct --device cuda:0 --max-seq-len 2048 --epochs 3 --lora-r 16 --val-frac 0.05 --skip-overlong"
rm -f $RUNS/qwen7b_tbox_t1c_lodo_bank/train_meta.json
echo "=== train $(date) ===" >> $SUM
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup $PY $TR $COMMON \
  --train-jsonl $OUT/lodo_train_t1c.jsonl --out-dir $RUNS/qwen7b_tbox_t1c_lodo_bank > $OUT/train_t1c.log 2>&1 &
while true; do
  [ -f $RUNS/qwen7b_tbox_t1c_lodo_bank/train_meta.json ] && { echo "$(date) train DONE" >> $SUM; break; }
  [ "$(pgrep -fc lora_train_chat_toolcall)" -eq 0 ] && { echo "trainer gone, no meta" >> $SUM; break; }
  echo "$(date) training... $(tail -1 $OUT/train_t1c.log 2>/dev/null)" >> $SUM; sleep 180
done

# 2) eval: T1c (src=1, GPU0) + treeval@s3 RE-EVAL (src=3, GPU1) in parallel. bug-fixed client.
cd $SB; $PY $SB/apply_two_stage_patch.py $CLONE >> $SUM 2>&1
rm -f /dev/shm/vllm* /dev/shm/nccl* 2>/dev/null
kill_gpu () { for p in $(nvidia-smi --id=$1 --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 $p 2>/dev/null; done; sleep 4; rm -f /dev/shm/vllm* /dev/shm/nccl* 2>/dev/null; }
eval_one () {  # tag adapter gpu port source
  local tag=$1 ad=$2 gpu=$3 port=$4 src=$5
  local AD=$RUNS/$ad
  echo "[$tag] adapter=$ad gpu=$gpu src=$src maxtok=$PLAN_MAXTOK $(date)" >> $SUM
  [ -f "$AD/adapter_model.safetensors" ] || { echo "[$tag] NO adapter $ad" >> $SUM; return; }
  rm -rf $OUT/eval_${tag}; rm -f $OUT/rllog_${tag}.jsonl
  local iport=$((8350 + gpu*200))
  CUDA_VISIBLE_DEVICES=$gpu VLLM_PORT=$iport VLLM_DP_MASTER_PORT=$((iport+50)) nohup $VLLM serve Qwen/Qwen2.5-7B-Instruct --enable-lora --max-lora-rank 16 \
    --lora-modules tbox_v2=$AD --port $port --dtype bfloat16 --gpu-memory-utilization 0.85 \
    --max-model-len 8192 --enable-auto-tool-choice --tool-call-parser hermes --trust-remote-code \
    > $OUT/serve_${tag}.log 2>&1 &
  for i in $(seq 1 120); do curl -s -m 3 http://localhost:$port/v1/models 2>/dev/null | grep -q tbox_v2 && break; sleep 4; done
  curl -s -m 3 http://localhost:$port/v1/models 2>/dev/null | grep -q tbox_v2 || { echo "[$tag] SERVE FAILED" >> $SUM; kill_gpu $gpu; return; }
  cd $CLONE
  env SOPBENCH_GATE=1 SOPBENCH_SCRATCHPAD=1 SOPBENCH_ALIAS=1 SOPBENCH_SOURCE=$src SOPBENCH_PLAN_MAXTOK=$PLAN_MAXTOK \
      SOPBENCH_RLLOG=$OUT/rllog_${tag}.jsonl SOPBENCH_VLLM_BASE_URL=http://localhost:$port/v1 $PY run_simulation.py \
    --domain bank --assistant_model tbox_v2 --tool_call_mode fc --tool_list full \
    --two_stage --two_stage_v2 --ont_dir $CLONE/induced --output_dir $OUT/eval_${tag} --env_mode prompt \
    > $OUT/sim_${tag}.log 2>&1
  $PY run_evaluation.py --domain bank --assistant_model tbox_v2 --tool_call_mode fc --tool_list full \
    --output_dir $OUT/eval_${tag} > $OUT/evalout_${tag}.txt 2>&1
  kill_gpu $gpu
}
eval_one t1c        qwen7b_tbox_t1c_lodo_bank            0 8351 1 &
eval_one treevals3  qwen7b_tbox_alias_s3_treeval_lodo_bank 1 8352 3 &
wait

echo "=== RUNG1 T1c 2x2 HEADLINE (maxtok=$PLAN_MAXTOK) $(date) ===" >> $SUM
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
# 2x2: cell -> eval dir
cells={"s1_none(C)":"eval_ub_C","s3_none(A)":"eval_ub_A","s1_treeval(T1c)":"eval_t1c","s3_treeval":"eval_treevals3"}
rl  ={"s1_treeval(T1c)":"rllog_t1c","s3_treeval":"rllog_treevals3"}
for name,ed in cells.items():
    fs=glob.glob(f'/home/woori/scratch/sft_alias_run/{ed}/bank/*.json')
    if not fs: print(f"{name}: NO DATA ({ed})"); continue
    d=json.load(open(fs[0])); T=[x for x in d if e0(x).get("action_should_succeed")]; F=[x for x in d if not e0(x).get("action_should_succeed")]
    if not T: print(f"{name}: evaluations EMPTY"); continue
    dg=sum(1 for x in T if e0(x).get("dirgraph_satisfied")); both=sum(1 for x in T if e0(x).get("dirgraph_satisfied") and e0(x).get("action_called_correctly"))
    acted=sum(1 for x in T if e0(x).get("action_successfully_called"))
    gref=sum(1 for x in T if e0(x).get("dirgraph_satisfied") and e0(x).get("user_goal") not in seq(x))
    prem=sum(1 for x in T if e0(x).get("action_successfully_called") and not e0(x).get("dirgraph_satisfied"))
    stepmed=sorted(len(seq(x)) for x in T)[len(T)//2]
    stopF=sum(1 for x in F if e0(x).get("success"))
    print(f"== {name} (n_T={len(T)}) == BOTH={both} dirgraph={dg} acted={acted} gathered_then_REFUSE={gref} premature={prem} step_med={stepmed} | should_F STOP={stopF}({100*stopF/max(1,len(F)):.0f}%)")
    # fabrication + format-mixing guard from rllog
    if name in rl:
        try:
            rows=[json.loads(l) for l in open(f'/home/woori/scratch/sft_alias_run/{rl[name]}.jsonl')]
            gnd=[o for o in (r["output"] for r in rows) if "gate" in o and "ready=true" in o.lower()]
            fb =[o for o in (r["output"] for r in rows) if "preconds_verified" in o]
            ops=[len(set(re.findall(r'op_\d+',o))) for o in gnd]
            print(f"   [rllog] grounded-gate={len(gnd)} fallback={len(fb)} | gate distinct-op med={sorted(ops)[len(ops)//2] if ops else 0} max={max(ops) if ops else 0} (fabrication guard: med~#leaf=ok)")
        except Exception as e: print("   rllog err",e)
print("(pre-check ceiling=34 grounded should_T | threshold: success BOTH>=12, strong>=20, null<=5 | baseline C-none/A BOTH 3)")
PYEOF
echo "=== rung1 T1c DONE $(date) ===" >> $SUM
cat $SUM
