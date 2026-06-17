#!/bin/bash
# S*(width) critical-scale ladder (local vLLM): where does multi-attr set-extraction start to break?
# Same Qwen2.5 family across {0.5,1.5,3,7,14}B (architecture fixed, only scale varies). Waits for the
# K1/K2 redo to finish, then runs width_scale_batch per size sequentially on a freed GPU. Detached.
set -u
REPO=/home/woori/workspace_common/boltzmann-attention-pi
MA=$REPO/scripts/distill/ma
LOG=/home/woori/scratch/depth/c8/width/ladder_local.log; mkdir -p $(dirname $LOG)
exec > $LOG 2>&1; set -x; date
while pgrep -f "[k]shot_sweep.sh" >/dev/null; do echo "redo running"; date; sleep 120; done
echo "redo done"; date; sleep 30
PICK=""
for g in 1 0; do
  n=$(nvidia-smi --id=$g --query-compute-apps=pid --format=csv,noheader | grep -c . || true)
  echo "GPU$g apps=$n"; [ "$n" = "0" ] && PICK=$g && break
done
[ -z "$PICK" ] && { echo NO_FREE_GPU; nvidia-smi; exit 1; }
cd $REPO && git pull --ff-only 2>&1 | tail -1
for MS in 0.5B:Qwen/Qwen2.5-0.5B-Instruct 1.5B:Qwen/Qwen2.5-1.5B-Instruct 3B:Qwen/Qwen2.5-3B-Instruct \
          7B:Qwen/Qwen2.5-7B-Instruct 14B:Qwen/Qwen2.5-14B-Instruct; do
  TAG=${MS%%:*}; MODEL=${MS#*:}
  echo "##### LADDER $TAG ($MODEL) on GPU$PICK #####"; date
  bash $MA/width_scale_batch.sh "$MODEL" "$TAG" "$PICK" 8068 || echo "FAIL $TAG"
done
echo "WIDTH_LADDER_LOCAL_DONE"; date
