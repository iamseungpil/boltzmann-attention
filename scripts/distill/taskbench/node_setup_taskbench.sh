#!/bin/bash
# node_setup_taskbench.sh — TaskBench (COWORKER_REQUEST_TB_SCALE v3) bootstrap on a node.
# Idempotent; run after node_setup_h200.sh (needs /scratch/boltzmann-attention + sop_env).
#   - JARVIS sparse clone (taskbench only, Apache-2.0)
#   - tb_env venv: pinned datasets/pyarrow for evaluate.py (kept apart from sop_env/vllm)
#   - inference.py patches (results-init + Qwen3 non-thinking payload)
#   - 500-sub per domain + data.json copy (census gold) + predictions symlinked into
#     /scratch/taskbench_runs/preds so the HF sync loop captures resume state
set -ex
export HF_HUB_CACHE=/scratch/hf_cache
REPO=/scratch/boltzmann-attention
mkdir -p /scratch/logs /scratch/venvs /scratch/taskbench_runs/preds

# 1. JARVIS sparse clone
if [ ! -d /scratch/JARVIS/taskbench ]; then
  git clone --depth 1 --filter=blob:none --sparse https://github.com/microsoft/JARVIS /scratch/JARVIS
  cd /scratch/JARVIS && git sparse-checkout set taskbench && cd /scratch
fi
TB=/scratch/JARVIS/taskbench

# 2. tb venv (separate: TaskBench eval wants datasets==2.14.5/pyarrow==12 which would
#    fight sop_env's training/vllm stack)
[ -d /scratch/venvs/tb_env ] || python3 -m venv /scratch/venvs/tb_env
TPIP=/scratch/venvs/tb_env/bin/pip
$TPIP install -q --upgrade pip
$TPIP install -q numpy scikit-learn networkx python-Levenshtein "datasets==2.14.5" \
  "pyarrow==12.0.0" rouge_score aiohttp emoji click requests

# 3. inference.py patches (idempotent)
/scratch/venvs/tb_env/bin/python $REPO/scripts/distill/taskbench/tb_patch_inference.py $TB

# 4. sub500 dirs + census gold + predictions->taskbench_runs symlinks (sync/resume state)
for d in data_huggingface data_multimedia data_dailylifeapis; do
  s=$TB/${d}_sub500
  if [ ! -f $s/user_requests.json ]; then
    mkdir -p $s
    head -500 $TB/$d/user_requests.json > $s/user_requests.json
    cp $TB/$d/tool_desc.json $TB/$d/graph_desc.json $s/
  fi
  [ -f $s/data.json ] || cp $TB/$d/data.json $s/
  for dir in $s $TB/$d; do
    base=$(basename $dir)
    mkdir -p /scratch/taskbench_runs/preds/$base
    [ -e $dir/predictions ] || ln -s /scratch/taskbench_runs/preds/$base $dir/predictions
  done
done
echo "TB_SETUP_DONE"
