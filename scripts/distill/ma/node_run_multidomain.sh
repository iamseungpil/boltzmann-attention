#!/bin/bash
# node_run_multidomain.sh — multidomain content-routing AT SCALE (coworker lane, inference-only).
#   Runs woori's committed multidomain_scale_eval.sh (retail+airline x gloss{0,1}) on a big base model.
#   ★multidomain_scale_eval.sh supports ENV-OVERRIDE (REPO/PY/VLLM/SCRATCH) so NO /home/woori symlink
#   dance is needed — we just point it at /scratch. Cases come from the repo (no tau2-bench dep).
#   Args: $1=HF_MODEL  $2=TAG (32B/72B/235B_awq).  Preempt-safe: per-file json + HF sync.
set -u
# ★COST-INCIDENT defense: local vLLM only; clear inference keys (no shared OpenRouter billing).
unset OPENROUTER_API_KEY ANTHROPIC_API_KEY OPENAI_API_KEY 2>/dev/null || true
MODEL="${1:?hf model}"; TAG="${2:?tag}"
GPUS="${3:-0,1,2,3}"; PORT="${4:-8055}"
REPO=/scratch/boltzmann-attention
PY=/scratch/venvs/sop_env/bin/python
VLLM=/scratch/venvs/sop_env/bin/vllm
HF=/scratch/venvs/sop_env/bin/hf
PIP=/scratch/venvs/sop_env/bin/pip
HFREPO=iamseungpil/sopbench-trackb-h200
SCRATCH=/scratch/woori_scratch
RES=$SCRATCH/depth/c8/multidomain/results; mkdir -p $RES $SCRATCH/logs
set -x

# 0. deps: pin fastapi/starlette (vllm-0.10.2 serve crash '_IncludedRouter' otherwise).
$PIP install -q requests "fastapi==0.136.3" "starlette==1.2.1" >> $SCRATCH/logs/pip.log 2>&1 || true

# 0b. restore prior results from HF (preempt-safe) + background sync
$HF download $HFREPO --repo-type dataset --include "multidomain_scale/*" --local-dir /scratch/hf_md 2>>$SCRATCH/logs/restore.log || true
[ -d /scratch/hf_md/multidomain_scale ] && cp -rn /scratch/hf_md/multidomain_scale/* $RES/ 2>/dev/null || true
( while true; do $HF upload $HFREPO $RES multidomain_scale --repo-type dataset \
    --commit-message "multidomain_scale sync $(date -u +%FT%TZ)" >> $SCRATCH/logs/sync.log 2>&1; sleep 600; done ) &
SYNC=$!

# 0c. model meta (L / d_model / params / MoE-active) — design §2 "꼭 기록".
$HF download "$MODEL" --include "config.json" --local-dir $SCRATCH/cfg_$TAG >> $SCRATCH/logs/cfg.log 2>&1 || true
cp $SCRATCH/cfg_$TAG/config.json $RES/meta_${TAG}.json 2>/dev/null || true

# 1. ★run woori's eval UNMODIFIED via env-override (4 evals: retail/airline x gloss 0/1).
#    skip if the last expected file (airline g1) already present from a prior preempted run.
if [ -f $RES/${TAG}__airline_g1.json ]; then
  echo "ALREADY DONE ($TAG): airline_g1 present — skip"
else
  REPO=$REPO PY=$PY VLLM=$VLLM SCRATCH=$SCRATCH \
    bash $REPO/scripts/distill/ma/multidomain_scale_eval.sh "$MODEL" "$TAG" "$GPUS" "$PORT"
fi
echo "=== $TAG mdscale tail ==="; tail -6 $SCRATCH/mdscale_${TAG}.log 2>/dev/null
ls -la $RES/

# 2. final sync
kill $SYNC 2>/dev/null || true
$HF upload $HFREPO $RES multidomain_scale --repo-type dataset \
  --commit-message "multidomain_scale $TAG FINAL $(date -u +%FT%TZ)" >> $SCRATCH/logs/sync.log 2>&1 || true
echo "MULTIDOMAIN_${TAG}_NODE_DONE $(date)"
