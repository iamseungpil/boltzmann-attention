#!/bin/bash
# node_run_depth_scale.sh — B(절차예산) scale: serve a big dense model bf16 TP4 + run woori's
#   committed depth_scale_batch.sh BYTE-UNMODIFIED (frozen synth data/arms/prompts across sizes,
#   B_BUDGET_SCALE_DESIGN §3 "편집 금지") via /home/woori symlinks. Coworker half = 32B/72B/235B.
#   in-head (arm A) + op-IR+engine (arm B) + oracle (arm D) over synth_depth N∈{5,10,20,50}.
#   Args: $1=HF_MODEL  $2=TAG (e.g. 32B / 72B / 235B_awq).  Preempt-safe: per-N json + HF sync.
set -u
# ★COST-INCIDENT defense (2026-06-16): local vLLM only; clear inference keys so this node can
# never bill the shared OpenRouter key even on an accidental agentic call.
unset OPENROUTER_API_KEY ANTHROPIC_API_KEY OPENAI_API_KEY 2>/dev/null || true
MODEL="${1:?hf model}"; TAG="${2:?tag}"
EXTRA="${3:---tensor-parallel-size 4}"   # 32B+ needs TP; default TP4 (one H100x4 node)
REPO=/scratch/boltzmann-attention
PIP=/scratch/venvs/sop_env/bin/pip
HF=/scratch/venvs/sop_env/bin/hf
HFREPO=iamseungpil/sopbench-trackb-h200
WS=/scratch/woori_scratch; mkdir -p $WS/logs $WS/depth
set -x

# 0. deps: pin fastapi/starlette to vllm-0.10.2-compatible (node_setup's fresh pip pulls a too-new
#    fastapi -> vllm serve crashes '_IncludedRouter' has no attribute 'path' = SERVE_FAIL).
$PIP install -q requests "fastapi==0.136.3" "starlette==1.2.1" >> $WS/logs/pip.log 2>&1 || true

# 1. ★symlink woori's hardcoded tree -> node paths so depth_scale_batch runs UNMODIFIED.
#    JOB runs as aiscuser (NOT root) -> plain mkdir /home/woori DENIED; use sudo then FAIL LOUD.
SUDO=""; command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null && SUDO="sudo -n"
$SUDO mkdir -p /home/woori/workspace_common /home/woori/venvs 2>/dev/null || mkdir -p /home/woori/workspace_common /home/woori/venvs 2>/dev/null || true
$SUDO chmod -R 777 /home/woori 2>/dev/null || true
ln -sfn $REPO            /home/woori/workspace_common/boltzmann-attention-pi 2>/dev/null
ln -sfn $WS             /home/woori/scratch 2>/dev/null
ln -sfn /scratch/venvs/sop_env /home/woori/venvs/seka_env 2>/dev/null
ln -sfn /scratch/venvs/sop_env /home/woori/venvs/tau2_vllm_env 2>/dev/null
[ -d /home/woori/scratch ] && [ -d /home/woori/workspace_common/boltzmann-attention-pi ] || {
  echo "FATAL: /home/woori symlinks failed (job not root + no sudo) — batch would die on cd. ABORT."; exit 1; }
echo "[symlinks OK]"

# 1b. restore prior depth results from HF (preempt-safe) + background sync
$HF download $HFREPO --repo-type dataset --include "depth_scale/*" --local-dir /scratch/hf_depth 2>>$WS/logs/restore.log || true
[ -d /scratch/hf_depth/depth_scale ] && cp -rn /scratch/hf_depth/depth_scale/* $WS/depth/ 2>/dev/null || true
( while true; do $HF upload $HFREPO $WS/depth depth_scale --repo-type dataset \
    --commit-message "depth_scale sync $(date -u +%FT%TZ)" >> $WS/logs/sync.log 2>&1; sleep 600; done ) &
SYNC=$!

# 1c. ★model meta (L / d_model / params / MoE-active) for S* ∝ L vs params (design §2 "꼭 기록").
$HF download "$MODEL" --include "config.json" --local-dir $WS/cfg_$TAG >> $WS/logs/cfg.log 2>&1 || true
cp $WS/cfg_$TAG/config.json $WS/depth/meta_${TAG}.json 2>/dev/null || true

# 2. ★run woori's batch UNMODIFIED: 4 GPUs TP4, single served model, all N inside.
#    (skip if final N=50 json already present from a prior preempted run)
if [ -f $WS/depth/depth_${TAG}_N50.json ]; then
  echo "ALREADY DONE ($TAG): N50 json present — skip"
else
  bash $REPO/scripts/distill/ma/depth_scale_batch.sh "$MODEL" "$TAG" "0,1,2,3" 8025 "$EXTRA"
fi
echo "=== $TAG tail ==="; tail -6 $WS/depth_${TAG}_g0,1,2,3.log 2>/dev/null
ls -la $WS/depth/

# 3. final sync
kill $SYNC 2>/dev/null || true
$HF upload $HFREPO $WS/depth depth_scale --repo-type dataset \
  --commit-message "depth_scale $TAG FINAL $(date -u +%FT%TZ)" >> $WS/logs/sync.log 2>&1 || true
echo "DEPTH_SCALE_${TAG}_NODE_DONE $(date)"
