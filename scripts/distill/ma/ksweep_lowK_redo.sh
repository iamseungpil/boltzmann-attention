#!/bin/bash
# Re-run the LOW-K end (K1,K2) of the diversity sweep with the CURRENT 7-op data builder, to remove the
# mid-sweep confound: K1/K2 were trained on 5-op data (before substitute/create were added); K4..K32 on
# 7-op. This regenerates K1/K2 as 7-op (overwriting the confounded files) so the whole curve shares one
# op-set. Waits for BOTH the original sweep AND the §20 multidomain run to finish (no GPU contention),
# then runs random(GPU0)+kcenter(GPU1) in parallel. Detached; fire-and-forget.
set -u
REPO=/home/woori/workspace_common/boltzmann-attention-pi
MA=$REPO/scripts/distill/ma
LOG=/home/woori/scratch/depth/c8/ksweep/redo_wait.log; mkdir -p $(dirname $LOG)
exec > $LOG 2>&1; set -x; date

# 1. wait for the original K-sweep wrappers (by PID) to exit
for pid in 4172136 4172137; do
  while kill -0 $pid 2>/dev/null; do echo "orig ksweep $pid still running"; date; sleep 120; done
done
echo "orig ksweep finished"; date; sleep 60

# 2. wait for the §20 multidomain_route (launched by md_wait_launch) to finish, if it is running
while pgrep -f "[m]ultidomain_route.sh" >/dev/null; do echo "multidomain_route still running"; date; sleep 120; done
echo "multidomain_route clear"; date; sleep 20

# 3. both GPUs now free -> re-run K1,K2 (7-op) on both methods in parallel
cd $REPO && git pull --ff-only 2>&1 | tail -1
setsid nohup bash $MA/kshot_sweep.sh 0 8025 random  "1 2" </dev/null >/dev/null 2>&1 &
setsid nohup bash $MA/kshot_sweep.sh 1 8026 kcenter "1 2" </dev/null >/dev/null 2>&1 &
wait
echo "KSWEEP_LOWK_REDO_DONE"; date
