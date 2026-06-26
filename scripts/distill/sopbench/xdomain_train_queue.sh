#!/bin/bash
# Train the 10 remaining held-out adapters (4 LODO + 6 train-1), GPU-aware: up to 2 concurrent,
# on whichever GPU has NO compute process (so it runs alongside an eval on the other GPU).
exec > /home/woori/scratch/sft_alias_run/xdomain_train_queue.log 2>&1
set -x
REPO=/home/woori/workspace_common/boltzmann-attention-pi
RUNS=$REPO/reports/facet_rft_2026/phase4_distill/sft_runs
SB=$REPO/scripts/distill/sopbench
ALL="bank dmv healthcare hotel library online_market university"
minus () { for d in $ALL; do [ "$d" != "$1" ] && printf "%s " "$d"; done; }
ntrain () { pgrep -f "[l]ora_train_chat_toolcall.py" | wc -l; }   # bracket trick: no self-match
gpu_free () { [ "$(nvidia-smi --id=$1 --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)" -eq 0 ]; }

NEED="lodo_dmv lodo_hotel lodo_online_market lodo_university train1_dmv train1_healthcare train1_hotel train1_library train1_online_market train1_university"
for cfg in $NEED; do
  [ -f $RUNS/qwen7b_tbox_t1c_${cfg}/adapter_model.safetensors ] && { echo "SKIP $cfg exists"; continue; }
  case $cfg in
    lodo_*)   X=${cfg#lodo_};   DOMS=$(minus $X);;
    train1_*) X=${cfg#train1_}; DOMS=$X;;
  esac
  # wait for <2 trainers AND a free GPU
  G=""
  while true; do
    if [ "$(ntrain)" -lt 2 ]; then
      for g in 1 0; do if gpu_free $g; then G=$g; break; fi; done
      [ -n "$G" ] && break
    fi
    sleep 30
  done
  echo "TRAIN $cfg GPU$G doms=[$DOMS] $(date)"
  setsid bash $SB/xdomain_train.sh t1c_$cfg $G $DOMS </dev/null >/dev/null 2>&1 &
  sleep 75   # let the new trainer claim the GPU before re-checking
done
while [ "$(ntrain)" -ge 1 ]; do sleep 30; done
echo "TRAIN_QUEUE_DONE $(date)"
