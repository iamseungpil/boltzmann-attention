#!/bin/bash
# Batch robust-eval all held-out outputs (orchestrator used crashing run_evaluation -> missing evals).
SEKA=/home/woori/venvs/seka_env/bin/python
SB=/home/woori/workspace_common/boltzmann-attention-pi/scripts/distill/sopbench
for f in /home/woori/scratch/sft_alias_run/xho_*/*/ast_tbox_v2-mode_fc-dep_full-fmt_structured-tool_full-shuffle_False.json; do
  [ -f "$f" ] && $SEKA $SB/eval_tasks.py "$f" 2>&1 | tail -1
done
