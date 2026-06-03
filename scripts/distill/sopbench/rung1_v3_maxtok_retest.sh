#!/bin/bash
# rung1_v3_maxtok_retest.sh — RETEST the v3 A/B after fixing the planner decode-budget artifact.
# RLLOG census proved treeval's verbose grounded terminal `ready=true; gate = AND(..nested..) = <v>; ACT`
# is TRUNCATED at max_tokens=24 (0/29 terminals reached a decision) -> non-convergence loop. This
# re-serves the SAME already-trained adapters and re-runs bank eval with SOPBENCH_PLAN_MAXTOK=160 so
# the grounded gate expression can complete. If treeval now converges and BOTH jumps, the v3 grounding
# hypothesis was never actually tested by the headline; if it converges but BOTH stays low, v3 semantics
# genuinely fail. nt is the unchanged control (160 is harmless: its terminal already fits in <12 tok).
# Also captures RLLOG to classify completed terminals (grounded gate val vs should_succeed agreement).
set +e
PLAN_MAXTOK=${PLAN_MAXTOK:-160}
PY=/home/woori/venvs/seka_env/bin/python
VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
REPO=/home/woori/workspace_common/boltzmann-attention-pi
SB=$REPO/scripts/distill/sopbench
CLONE=/home/woori/scratch/SOPBench
RUNS=$REPO/reports/facet_rft_2026/phase4_distill/sft_runs
OUT=/home/woori/scratch/sft_alias_run
SUM=$OUT/RUNG1_V3_MAXTOK_RETEST.txt
echo "=== v3 max_tokens=$PLAN_MAXTOK retest start $(date) ===" > $SUM
rm -f /dev/shm/vllm* /dev/shm/nccl* 2>/dev/null
$PY $SB/apply_two_stage_patch.py $CLONE >> $SUM 2>&1

kill_gpu () { for p in $(nvidia-smi --id=$1 --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 $p 2>/dev/null; done; sleep 4; rm -f /dev/shm/vllm* /dev/shm/nccl* 2>/dev/null; }
eval_one () {  # arm gpu port
  local arm=$1 gpu=$2 port=$3
  local AD=$RUNS/qwen7b_tbox_alias_s3_${arm}_lodo_bank
  echo "[$arm] gpu=$gpu maxtok=$PLAN_MAXTOK $(date)" >> $SUM
  [ -f "$AD/adapter_model.safetensors" ] || { echo "[$arm] NO adapter" >> $SUM; return; }
  rm -rf $OUT/eval_mt_${arm}; rm -f $OUT/rllog_mt_${arm}.jsonl
  local iport=$((8700 + gpu*200))
  CUDA_VISIBLE_DEVICES=$gpu VLLM_PORT=$iport VLLM_DP_MASTER_PORT=$((iport+50)) nohup $VLLM serve Qwen/Qwen2.5-7B-Instruct --enable-lora --max-lora-rank 16 \
    --lora-modules tbox_v2=$AD --port $port --dtype bfloat16 --gpu-memory-utilization 0.85 \
    --max-model-len 8192 --enable-auto-tool-choice --tool-call-parser hermes --trust-remote-code \
    > $OUT/serve_mt_${arm}.log 2>&1 &
  for i in $(seq 1 120); do curl -s -m 3 http://localhost:$port/v1/models 2>/dev/null | grep -q tbox_v2 && break; sleep 4; done
  curl -s -m 3 http://localhost:$port/v1/models 2>/dev/null | grep -q tbox_v2 || { echo "[$arm] SERVE FAILED" >> $SUM; kill_gpu $gpu; return; }
  cd $CLONE
  env SOPBENCH_GATE=1 SOPBENCH_SCRATCHPAD=1 SOPBENCH_ALIAS=1 SOPBENCH_SOURCE=3 SOPBENCH_PLAN_MAXTOK=$PLAN_MAXTOK \
      SOPBENCH_RLLOG=$OUT/rllog_mt_${arm}.jsonl SOPBENCH_VLLM_BASE_URL=http://localhost:$port/v1 $PY run_simulation.py \
    --domain bank --assistant_model tbox_v2 --tool_call_mode fc --tool_list full \
    --two_stage --two_stage_v2 --ont_dir $CLONE/induced --output_dir $OUT/eval_mt_${arm} --env_mode prompt \
    > $OUT/sim_mt_${arm}.log 2>&1
  $PY run_evaluation.py --domain bank --assistant_model tbox_v2 --tool_call_mode fc --tool_list full \
    --output_dir $OUT/eval_mt_${arm} > $OUT/evalout_mt_${arm}.txt 2>&1
  kill_gpu $gpu
}
eval_one treeval 0 9701 &
eval_one nt      1 9702 &
wait

echo "=== RUNG1 v3 max_tokens=$PLAN_MAXTOK RETEST HEADLINE $(date) ===" >> $SUM
$PY - >> $SUM 2>&1 <<'PYEOF'
import json,glob,re
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
for arm in ["nt","treeval"]:
    fs=glob.glob(f'/home/woori/scratch/sft_alias_run/eval_mt_{arm}/bank/*.json')
    if not fs: print(f"{arm}: NO DATA"); continue
    d=json.load(open(fs[0])); T=[x for x in d if e0(x).get("action_should_succeed")]; F=[x for x in d if not e0(x).get("action_should_succeed")]
    dg=sum(1 for x in T if e0(x).get("dirgraph_satisfied")); gc=sum(1 for x in T if e0(x).get("action_called_correctly"))
    both=sum(1 for x in T if e0(x).get("dirgraph_satisfied") and e0(x).get("action_called_correctly"))
    acted=sum(1 for x in T if e0(x).get("action_successfully_called"))
    noact=sum(1 for x in T if not e0(x).get("action_successfully_called") and e0(x).get("user_goal") not in seq(x))
    import collections
    loop=sum(1 for x in T if (max(collections.Counter(seq(x)).values()) if seq(x) else 0)>=3)
    stopF=sum(1 for x in F if e0(x).get("success"))
    print(f"== {arm} (maxtok retest) ==")
    print(f"  should_T(48): dirgraph={dg} acted={acted} goal={gc} BOTH={both} | over-refuse(noact)={noact} loop>=3={loop}")
    print(f"  should_F(86): STOP-recall={stopF} ({100*stopF/86:.0f}%)")
# RLLOG terminal-reach + grounded agreement
for arm in ["nt","treeval"]:
    try: rows=[json.loads(l) for l in open(f'/home/woori/scratch/sft_alias_run/rllog_mt_{arm}.jsonl')]
    except Exception: print(f"{arm}: no rllog"); continue
    ready=[r["output"] for r in rows if "ready=true" in r["output"].lower()]
    reached=sum(1 for o in ready if re.search(r'\b(ACT|STOP)\s*$',o.strip()))
    print(f"  [{arm}] terminal-attempts={len(ready)} reached_decision={reached} ({100*reached/len(ready) if ready else 0:.0f}%)")
print("(prior @maxtok24: nt BOTH=5 STOP=42% reach=100% | treeval BOTH=2 STOP=20% reach=0% loop=35)")
PYEOF
echo "=== v3 maxtok retest DONE $(date) ===" >> $SUM
cat $SUM
