#!/bin/bash
# Collect the 7B same-substrate width point (apples-to-apples with frontier gpt-4.1, M_A_RESULTS §22):
# wait for the K1/K2 redo sweep to finish, then run width_scale_batch for 7B on a freed GPU. Detached.
set -u
REPO=/home/woori/workspace_common/boltzmann-attention-pi
MA=$REPO/scripts/distill/ma
LOG=/home/woori/scratch/depth/c8/width/width7b_wait.log; mkdir -p $(dirname $LOG)
exec > $LOG 2>&1; set -x; date
while pgrep -f "[k]shot_sweep.sh" >/dev/null; do echo "redo ksweep running"; date; sleep 120; done
echo "redo done"; date; sleep 30
# pick a free GPU (prefer GPU1)
PICK=""
for g in 1 0; do
  n=$(nvidia-smi --id=$g --query-compute-apps=pid --format=csv,noheader | grep -c . || true)
  echo "GPU$g apps=$n"; [ "$n" = "0" ] && PICK=$g && break
done
[ -z "$PICK" ] && { echo "NO_FREE_GPU"; nvidia-smi; exit 1; }
cd $REPO && git pull --ff-only 2>&1 | tail -1
bash $MA/width_scale_batch.sh "Qwen/Qwen2.5-7B-Instruct" 7B "$PICK" 8068
echo "WIDTH_7B_WAIT_DONE"; date
