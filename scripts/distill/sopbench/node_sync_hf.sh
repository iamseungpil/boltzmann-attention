#!/bin/bash
# node_sync_hf.sh — periodic result sync to HF dataset (preemption-safe state store).
# usage: nohup bash node_sync_hf.sh <role> &      role = train | eval
# Uploads are content-hash deduped by HF, so unchanged files cost nothing.
ROLE=${1:-misc}
HFREPO=iamseungpil/sopbench-trackb-h200
HF=/scratch/venvs/sop_env/bin/hf
INTERVAL=${SYNC_INTERVAL:-600}
while true; do
  for d in sopbench_runs sft_runs logs; do
    [ -d /scratch/$d ] && [ -n "$(ls -A /scratch/$d 2>/dev/null)" ] && \
      $HF upload $HFREPO /scratch/$d $ROLE/$d --repo-type dataset \
        --commit-message "sync $ROLE/$d $(date -u +%FT%TZ)" \
        >> /scratch/logs/hf_sync.log 2>&1
  done
  date -u +"%FT%TZ sync cycle done" >> /scratch/logs/hf_sync.log
  sleep $INTERVAL
done
