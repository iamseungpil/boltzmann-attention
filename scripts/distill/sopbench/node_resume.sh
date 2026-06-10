#!/bin/bash
# node_resume.sh — pull prior state from the HF dataset and (re)launch this node's work.
# Safe to run on a FRESH node after preemption AND on a node mid-work (idempotent):
#   - restores /scratch/{sopbench_runs,sft_runs} from HF (no overwrite of newer local files)
#   - starts the 10-min HF sync loop if not running
#   - dispatches the role's workload in the background:
#       eval  -> node_run_sanity32b.sh   (run_simulation skips already-completed tasks)
#       train -> node_run_sft32b.sh      (trainer --resume picks up ckpt_state.pt)
# usage: bash node_resume.sh train|eval
set -x
ROLE=$1
[ -z "$ROLE" ] && { echo "usage: node_resume.sh train|eval"; exit 1; }
REPO=/scratch/boltzmann-attention
HFREPO=iamseungpil/sopbench-trackb-h200
HF=/scratch/venvs/sop_env/bin/hf
mkdir -p /scratch/logs /scratch/sopbench_runs /scratch/sft_runs

# 1. restore state (ok if the dataset is still empty)
$HF download $HFREPO --repo-type dataset --local-dir /scratch/hf_state >> /scratch/logs/hf_sync.log 2>&1 || true
for d in sopbench_runs sft_runs; do
  if [ -d /scratch/hf_state/$ROLE/$d ]; then
    cp -rn /scratch/hf_state/$ROLE/$d/* /scratch/$d/ 2>/dev/null || true
  fi
done

# 2. sync loop (single instance)
pgrep -f "node_sync_hf.sh $ROLE" > /dev/null || \
  nohup bash $REPO/scripts/distill/sopbench/node_sync_hf.sh $ROLE > /dev/null 2>&1 &

# 3. dispatch workload (single instance)
case $ROLE in
  eval)
    pgrep -f node_run_sanity32b.sh > /dev/null || \
      nohup bash $REPO/scripts/distill/sopbench/node_run_sanity32b.sh \
        > /scratch/logs/run_sanity_driver.log 2>&1 &
    ;;
  train)
    pgrep -f node_run_sft32b.sh > /dev/null || \
      nohup bash $REPO/scripts/distill/sopbench/node_run_sft32b.sh \
        > /scratch/logs/run_sft_driver.log 2>&1 &
    ;;
esac
echo "RESUME_DISPATCHED role=$ROLE"
