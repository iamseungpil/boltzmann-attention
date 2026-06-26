#!/bin/bash
# Queue the multi-domain routing transfer (§20) WITHOUT preempting the K-sweep: wait until the K-sweep
# wrapper processes exit, then launch multidomain_route.sh on a freed GPU. Detached (setsid nohup) so it
# survives ssh disconnect. Self-contained log; safe to fire-and-forget.
set -u
REPO=/home/woori/workspace_common/boltzmann-attention-pi
MA=$REPO/scripts/distill/ma
LOG=/home/woori/scratch/depth/c8/multidomain/logs; mkdir -p $LOG
exec > $LOG/wait_launch.log 2>&1; set -x; date

# 1. wait for the K-sweep wrapper(s) to finish (bracket trick avoids pgrep self-match)
while pgrep -f "[k]shot_sweep.sh" >/dev/null; do echo "ksweep still running"; date; sleep 120; done
echo "ksweep finished"; date; sleep 30

# 2. pick a GPU with zero compute apps (prefer GPU1)
PICK=""
for g in 1 0; do
  n=$(nvidia-smi --id=$g --query-compute-apps=pid --format=csv,noheader | grep -c . || true)
  echo "GPU$g apps=$n"
  if [ "$n" = "0" ]; then PICK=$g; break; fi
done
if [ -z "$PICK" ]; then echo "NO_FREE_GPU"; nvidia-smi; exit 1; fi

# 3. launch the transfer driver (1 epoch, 6000 synth examples)
echo "launching multidomain_route on GPU $PICK"
cd $REPO && git pull --ff-only 2>&1 | tail -1
bash $MA/multidomain_route.sh $PICK 8027 1 6000
echo "WAIT_LAUNCH_DONE"; date
