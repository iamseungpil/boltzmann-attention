#!/bin/bash
# Auto-chain: wait for sanity_equiv to finish, then launch α grid.
#
# Detection signal: outer sanity_equiv bash wrapper PID exits AND
# sanity_equiv_done.txt is touched.
#
# After sanity completes:
#   1. Verify both pass^1 land within Wilson 95% CI of baseline subset
#      (Qwen [0.124, 0.262], Hermes [0.077, 0.196])
#   2. Auto-launch α grid for α=0.1, 0.3, 0.5, 1.0, 2.0 (skip α=0 — reuse sanity)
#
# Logs to phase2_steering/_auto_chain.log
set -uo pipefail
cd /home/woori/workspace_common/boltzmann-attention-pi

PHASE2="reports/facet_rft_2026/phase2_steering"
mkdir -p "$PHASE2"
LOG="$PHASE2/_auto_chain_$(date +%Y%m%d_%H%M%S).log"
exec >>"$LOG" 2>&1

echo "[auto-chain] start $(date)"

# Find the sanity outer wrapper PID (passed via env or auto-detect)
SANITY_PID="${SANITY_PID:-}"
if [ -z "$SANITY_PID" ]; then
  # auto-detect: oldest still-alive bash _phase2a_sanity_equiv.sh
  SANITY_PID=$(pgrep -of "bash _phase2a_sanity_equiv.sh" 2>/dev/null | head -n1)
fi
if [ -z "$SANITY_PID" ] || ! ps -p "$SANITY_PID" >/dev/null 2>&1; then
  echo "[auto-chain] WARN no live sanity wrapper found; checking done marker"
  if [ ! -f "$PHASE2/sanity_equiv_done.txt" ]; then
    echo "[auto-chain] ERROR sanity not running and not done; abort"; exit 1
  fi
fi

echo "[auto-chain] watching sanity wrapper pid=$SANITY_PID"

while ps -p "$SANITY_PID" >/dev/null 2>&1; do
  # log progress every ~5 min
  for d in "$PHASE2"/sanity_equiv_qwen7b_*_235432 "$PHASE2"/sanity_equiv_hermes3_*_235432; do
    if [ -d "$d" ] && [ -f "$d/run.log" ]; then
      status=$(grep -oE 'Status: [0-9]+/[0-9]+ complete\. Avg reward: [0-9.]+' "$d/run.log" 2>/dev/null | tail -1)
      [ -n "$status" ] && echo "[auto-chain] $(date '+%H:%M:%S') $(basename $d): $status"
    fi
  done
  sleep 300
done
echo "[auto-chain] sanity wrapper exited at $(date)"

# Wait for done marker (final write may lag pid exit by seconds)
for i in $(seq 1 30); do
  [ -f "$PHASE2/sanity_equiv_done.txt" ] && break
  sleep 5
done

# Verify pass^1 in expected Wilson CI
echo "[auto-chain] checking final pass^1..."
python3 << 'PYEOF'
import json, os, sys
from math import sqrt
def wilson(p, n, z=1.96):
    if n == 0: return (0, 1)
    denom = 1 + z*z/n
    center = (p + z*z/(2*n)) / denom
    margin = z * sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
    return (max(0, center-margin), min(1, center+margin))

baseline = {'qwen7b': 0.1833, 'hermes3': 0.1250}
ci = {'qwen7b': (0.124, 0.262), 'hermes3': (0.077, 0.196)}

ok = True
for tag, hf_dir_glob in [('qwen7b', 'sanity_equiv_qwen7b_*_235432'),
                          ('hermes3', 'sanity_equiv_hermes3_*_235432')]:
    import glob
    P='/home/woori/workspace_common/boltzmann-attention/external/tau2-bench/data/simulations/reports/facet_rft_2026/phase2_steering/'
    dirs=sorted(glob.glob(P+hf_dir_glob))
    if not dirs:
        print(f'[auto-chain]   {tag}: NO RESULTS DIR'); ok=False; continue
    p=f'{dirs[-1]}/B0_telecom_base.json/results.json'
    if not os.path.exists(p):
        print(f'[auto-chain]   {tag}: NO results.json'); ok=False; continue
    d=json.load(open(p))
    sims=d['simulations']
    done=[s for s in sims if s.get('end_time')]
    rewards=[(s.get('reward_info') or {}).get('reward') for s in done]
    rewards=[r for r in rewards if r is not None]
    n_pass=sum(1 for r in rewards if r>=1.0)
    p1=n_pass/max(len(rewards),1)
    lo, hi = ci[tag]
    in_ci = lo <= p1 <= hi
    print(f'[auto-chain]   {tag}: n={len(rewards)} pass^1={p1:.4f} baseline={baseline[tag]:.4f} CI=[{lo}, {hi}] in_CI={in_ci}')
    if not in_ci: ok=False
if not ok:
    print('[auto-chain]   one or more checks failed — α grid will still launch (researcher decision)')
sys.exit(0)
PYEOF

# Launch α grid (skip α=0, trials=4)
echo "[auto-chain] launching α grid (α=0.1, 0.3, 0.5, 1.0, 2.0, trials=4)"
ALPHAS="0.1 0.3 0.5 1.0 2.0" TRIALS=4 \
  nohup bash _phase2a_vllm_orchestrator.sh \
  >> "$PHASE2/_orch_vllm.nohup.log" 2>&1 &
ORCH_PID=$!
echo "$ORCH_PID" > "$PHASE2/_orch_vllm.pid"
echo "[auto-chain] orchestrator launched pid=$ORCH_PID"

# wait briefly to confirm orch is alive
sleep 30
if ps -p "$ORCH_PID" >/dev/null 2>&1; then
  echo "[auto-chain] orchestrator confirmed alive after 30s"
else
  echo "[auto-chain] WARN orchestrator died within 30s; check logs"
fi

date > "$PHASE2/auto_chain_done.txt"
echo "[auto-chain] DONE $(date)"
