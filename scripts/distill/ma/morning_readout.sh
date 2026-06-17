#!/bin/bash
# One-command morning readout of the overnight experiments. Each section independent (failure -> next).
PY=/home/woori/venvs/seka_env/bin/python
MA=/home/woori/workspace_common/boltzmann-attention-pi/scripts/distill/ma
S=/home/woori/scratch/depth/c8
echo "############ 1. WIDTH x SCALE  (S*(width): where does multi-attr extraction break?) ############"
$PY $MA/width_summary.py --dir $S/width 2>&1 || echo "(width_summary failed)"
echo; echo "############ 2. DIVERSITY (K-sweep, now all 7-op) ############"
$PY $MA/ksweep_summary.py 2>&1 | tail -40 || echo "(ksweep_summary failed/absent)"
echo; echo "############ 3. WIDEN-RETRAIN (does covering B-width in training fix component B?) ############"
for f in $S/multidomain/results/MD_widesubst_w*__*_g0.json; do
  [ -f "$f" ] || continue
  $PY - "$f" <<'PYEOF'
import json,sys
d=json.load(open(sys.argv[1])); o=d["overall"]
print(f"  {sys.argv[1].split('/')[-1]}: acc={o[0]}/{o[1]}({o[0]/max(o[1],1):.2f}) recog={d.get('recognition')} op_dist={d.get('op_dist')}")
PYEOF
done
echo "  -- widen-retrain width curve --"
$PY $MA/width_summary.py --dir $S/width 2>&1 | grep -i widesubst || true
echo; echo "############ 3b. §17 CLOSURE 3-way (base vs 5-op-C8 vs 7-op-C8, identical cases) ############"
for f in $S/multidomain/results/base7b_*_g0.json $S/multidomain/results/5op_*__*_g0.json $S/multidomain/results/MD_route_ep1__*_g0.json; do
  [ -f "$f" ] || continue
  $PY - "$f" <<'PYEOF'
import json,sys
d=json.load(open(sys.argv[1])); o=d["overall"]
print(f"  {sys.argv[1].split('/')[-1]:42s} acc={o[0]}/{o[1]}({o[0]/max(o[1],1):.2f}) recog={d.get('recognition')} op_dist={d.get('op_dist')}")
PYEOF
done
echo; echo "############ 4. JOB STATUS ############"
for p in width_eval.py widen_retrain.sh width_ladder_local width_overnight_or kshot_sweep; do
  echo "  $p: $(pgrep -fc "[${p:0:1}]${p:1}" 2>/dev/null) running"
done
echo "  GPUs:"; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | sed 's/^/    /'
echo "  logs: $S/width/{overnight_or,ladder_local}.log  $S/multidomain/logs/MD_widesubst_w4.log"