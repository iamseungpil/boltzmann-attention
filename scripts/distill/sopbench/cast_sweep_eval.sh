#!/bin/bash
# cast_sweep_eval.sh — CAST probe step 2: alpha-sweep the ACT behavior vector via the
# (always-on/orth) gated steering server on bank eval; measure should_T ACT-recall|gather
# and should_F STOP-recall per alpha. alpha=0 = control (server disables steering => plain vLLM).
#
# HONEST SCOPE (EXPERIMENT_DESIGN Rung3 재검토): this is the ALWAYS-ON/position-gated family
# (decode-time anchor gating is unimplemented in _steering_vllm_server_gated.py). It tests whether
# steering toward ACT/permitted reduces the over-refusal PRIOR and at what should_F cost. It does
# NOT test grounded tree-eval. ~1-layer offset between extraction-layer index (hidden_states[L])
# and server decoder-layer index is accepted for a coarse band probe.
#
# RUN (remote): bash scripts/distill/sopbench/cast_sweep_eval.sh
set +e
PY=/home/woori/venvs/seka_env/bin/python
VLLM_PY=/home/woori/venvs/tau2_vllm_env/bin/python   # vLLM lives here (seka has no vllm); gated server needs it
REPO=/home/woori/workspace_common/boltzmann-attention-pi
SB=$REPO/scripts/distill/sopbench
CLONE=/home/woori/scratch/SOPBench
RUNS=$REPO/reports/facet_rft_2026/phase4_distill/sft_runs
OUT=/home/woori/scratch/sft_alias_run
AD=$RUNS/qwen7b_tbox_alias_s3_scratch_lodo_bank
VEC=$OUT/cast_actvec_alias_s3.pt
GATED=$REPO/_steering_vllm_server_gated.py
LAYERS="14,15,16,17,18,19,20"
ALPHAS="${ALPHAS:-0 8 16}"
SUM=$OUT/CAST_SWEEP_RESULTS.txt
echo "=== CAST alpha-sweep $(date) layers=$LAYERS alphas=$ALPHAS ===" > $SUM
[ -f "$VEC" ] || { echo "NO VECTOR $VEC" | tee -a $SUM; exit 1; }
$PY $SB/apply_two_stage_patch.py $CLONE >> $SUM 2>&1

kill_gpu () { for p in $(nvidia-smi --id=0 --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 $p 2>/dev/null; done; sleep 4; rm -f /dev/shm/vllm* /dev/shm/nccl* 2>/dev/null; }

for A in $ALPHAS; do
  kill_gpu
  echo "[alpha=$A] serving $(date)" >> $SUM
  EVDIR=$OUT/cast_eval_a${A}
  rm -rf $EVDIR
  CUDA_VISIBLE_DEVICES=0 VLLM_PORT=8100 VLLM_DP_MASTER_PORT=8150 nohup $VLLM_PY $GATED \
    --steering-vectors $VEC --relation actvec --alpha $A --layers $LAYERS --gate-mode orth --target-layer-class qwen2 \
    -- --model Qwen/Qwen2.5-7B-Instruct --enable-lora --max-lora-rank 16 --lora-modules tbox_v2=$AD \
       --port 9001 --dtype bfloat16 --gpu-memory-utilization 0.85 --max-model-len 8192 \
       --enable-auto-tool-choice --tool-call-parser hermes --trust-remote-code \
    > $OUT/cast_serve_a${A}.log 2>&1 &
  for i in $(seq 1 120); do curl -s -m3 http://localhost:9001/v1/models 2>/dev/null | grep -q tbox_v2 && break; sleep 4; done
  curl -s -m3 http://localhost:9001/v1/models 2>/dev/null | grep -q tbox_v2 || { echo "[alpha=$A] SERVE FAILED" >> $SUM; tail -6 $OUT/cast_serve_a${A}.log >> $SUM; kill_gpu; continue; }
  grep -m1 "steer-vllm.*ACTIVE\|steering DISABLED" $OUT/cast_serve_a${A}.log >> $SUM
  cd $CLONE
  env SOPBENCH_GATE=1 SOPBENCH_SCRATCHPAD=1 SOPBENCH_ALIAS=1 SOPBENCH_SOURCE=3 \
      SOPBENCH_VLLM_BASE_URL=http://localhost:9001/v1 $PY run_simulation.py \
      --domain bank --assistant_model tbox_v2 --tool_call_mode fc --tool_list full \
      --two_stage --two_stage_v2 --ont_dir $CLONE/induced --output_dir $EVDIR --env_mode prompt \
      > $OUT/cast_sim_a${A}.log 2>&1
  $PY run_evaluation.py --domain bank --assistant_model tbox_v2 --tool_call_mode fc --tool_list full \
      --output_dir $EVDIR > $OUT/cast_evalout_a${A}.txt 2>&1
  kill_gpu
  EVDIR=$EVDIR $PY - >> $SUM 2>&1 <<'PYEOF'
import json, glob, os
def e0(x):
    e=x.get("evaluations"); return (e[0] if e else {}) if isinstance(e,list) else (e or {})
def seq(x):
    s=[]
    for it in (x.get("interactions") or []):
        for st in (it.get("interaction") or []):
            if st.get("tool_name"): s.append(st["tool_name"])
    return s
fs=glob.glob(os.environ["EVDIR"]+"/bank/*.json")
if not fs: print("  NO EVAL JSON"); raise SystemExit
d=json.load(open(fs[0]))
T=[x for x in d if e0(x).get("action_should_succeed")]; F=[x for x in d if not e0(x).get("action_should_succeed")]
dg=sum(1 for x in T if e0(x).get("dirgraph_satisfied")); gc=sum(1 for x in T if e0(x).get("action_called_correctly"))
both=sum(1 for x in T if e0(x).get("dirgraph_satisfied") and e0(x).get("action_called_correctly"))
acted=sum(1 for x in T if e0(x).get("action_successfully_called"))
stopF=sum(1 for x in F if e0(x).get("success"))
print(f"  should_T(48): dirgraph={dg} acted={acted} goal={gc} BOTH={both} | should_F STOP-recall={stopF}/86")
PYEOF
done
echo "=== CAST sweep DONE $(date) ===" >> $SUM
cat $SUM
