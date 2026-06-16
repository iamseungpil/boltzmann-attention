#!/bin/bash
# node_run_factorial_iso.sh — M-sigma v4 factorial ISO=ON half {A-iso,FULL,C-in,C-ip} on H100x4.
#   ★KEY: runs woori's committed ma_factorial_batch.sh BYTE-UNMODIFIED (factorial validity, §4
#   "편집 금지") by SYMLINKING the hardcoded /home/woori tree to node /scratch (root in container).
#   4 arms in PARALLEL (one GPU each) -> ~1.5h vs ~5h sequential. NO tau2 in training (transfer target).
#   Preemption-safe: per-arm FACTORIAL_<arm>_DONE markers + HF sync to factorial_iso/.
set -u
# ★COST-INCIDENT defense (2026-06-16): this node must NEVER bill the shared OpenRouter key.
# It uses local vLLM only; clear any inference keys so even an accidental agentic call can't drain.
unset OPENROUTER_API_KEY ANTHROPIC_API_KEY OPENAI_API_KEY 2>/dev/null || true
REPO=/scratch/boltzmann-attention
PIP=/scratch/venvs/sop_env/bin/pip
HF=/scratch/venvs/sop_env/bin/hf
HFREPO=iamseungpil/sopbench-trackb-h200
WS=/scratch/woori_scratch; mkdir -p $WS/logs
set -x

# 0. env: sop_env has vllm+transformers (node_setup_h200); ADD peft for LoRA SFT (lora_train needs it)
$PIP install -q peft >> $WS/logs/pip.log 2>&1 || true

# 1. ★symlink woori's hardcoded tree -> node paths (so the committed batch runs UNMODIFIED).
#    ★FIX (2026-06-16): the JOB runs as aiscuser (NOT root) so plain `mkdir /home/woori` is DENIED
#    -> symlinks silently absent -> arms die on `cd /home/woori/...` (0 results, GPUs idle 1h).
#    Use sudo (container allows it); then FAIL LOUD if symlinks didn't materialize.
SUDO=""; command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null && SUDO="sudo -n"
$SUDO mkdir -p /home/woori/workspace_common /home/woori/venvs 2>/dev/null || mkdir -p /home/woori/workspace_common /home/woori/venvs 2>/dev/null || true
$SUDO chmod -R 777 /home/woori 2>/dev/null || true
ln -sfn $REPO            /home/woori/workspace_common/boltzmann-attention-pi 2>/dev/null
ln -sfn $WS             /home/woori/scratch 2>/dev/null
ln -sfn /scratch/venvs/sop_env /home/woori/venvs/seka_env 2>/dev/null
ln -sfn /scratch/venvs/sop_env /home/woori/venvs/tau2_vllm_env 2>/dev/null
[ -d /home/woori/scratch ] && [ -d /home/woori/workspace_common/boltzmann-attention-pi ] || {
  echo "FATAL: /home/woori symlinks failed (job not root + no sudo) — woori batch would die on cd. ABORT (no silent idle)."; exit 1; }
echo "[symlinks OK]"

# 2. tau2-bench retail data at woori's expected path (gold_extract default = /home/woori/scratch/tau2-bench)
[ -f $WS/tau2-bench/data/tau2/domains/retail/tasks.json ] || \
  git clone --depth 1 https://github.com/sierra-research/tau2-bench.git $WS/tau2-bench >> $WS/logs/clone.log 2>&1

# 2b. restore prior arm results from HF (preempt-safe)
$HF download $HFREPO --repo-type dataset --include "factorial_iso/*" --local-dir /scratch/hf_fi 2>>$WS/logs/restore.log || true
[ -d /scratch/hf_fi/factorial_iso ] && cp -rn /scratch/hf_fi/factorial_iso/* $WS/ 2>/dev/null || true

# 2c. background HF sync of results
( while true; do $HF upload $HFREPO $WS/factorial factorial_iso --repo-type dataset \
    --commit-message "factorial_iso sync $(date -u +%FT%TZ)" >> $WS/logs/sync.log 2>&1; sleep 600; done ) &
SYNC=$!

# 3. ★4 ISO=ON arms in PARALLEL — woori's batch UNMODIFIED, one GPU + port each.
#    (A-iso, FULL = headline ISO/full; C-in, C-ip = combos.) Skip an arm if its DONE marker exists.
BATCH=$REPO/scripts/distill/ma/ma_factorial_batch.sh
declare -A G=( [A-iso]=0 [FULL]=1 [C-in]=2 [C-ip]=3 )
declare -A P=( [A-iso]=8041 [FULL]=8042 [C-in]=8043 [C-ip]=8044 )
pids=()
for ARM in A-iso FULL C-in C-ip; do
  if grep -q "FACTORIAL_${ARM}_DONE" $WS/ma_factorial_${ARM}_g${G[$ARM]}.log 2>/dev/null; then
    echo "skip $ARM (done)"; continue; fi
  ( bash $BATCH $ARM ${G[$ARM]} ${P[$ARM]} ) &
  pids+=($!)
  sleep 20   # stagger model-load / hf-cache contention
done
echo "launched ${#pids[@]} arms in parallel"; wait
echo "=== all arms returned ==="
for ARM in A-iso FULL C-in C-ip; do
  echo "--- $ARM tail ---"; tail -3 $WS/ma_factorial_${ARM}_g${G[$ARM]}.log 2>/dev/null
done

# 4. final sync
kill $SYNC 2>/dev/null || true
$HF upload $HFREPO $WS/factorial factorial_iso --repo-type dataset \
  --commit-message "factorial_iso FINAL $(date -u +%FT%TZ)" >> $WS/logs/sync.log 2>&1 || true
echo "FACTORIAL_ISO_NODE_DONE $(date)"
