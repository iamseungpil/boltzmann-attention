#!/bin/bash
# Watcher: wait until the dist trainer finishes, then run the fixed full-catalog eval
# on the FINAL adapter (step~22800). Detached fire-and-forget; result lands in metrics.
# Trainer signature: lora_train_chat_toolcall.py ... qwen7b_nfc_lodo_daily_dist
set -u
S=/home/woori/scratch
LOG=$S/nfc_dist_autofinal_eval.log
exec > $LOG 2>&1; set -x; date
SIG="[l]ora_train_chat_toolcall.py.*daily_dist"
# Wait for trainer to exit (cap ~10h). bracket-trick avoids pgrep self-match.
for i in $(seq 1 1200); do
  pgrep -f "$SIG" >/dev/null || { echo "TRAINER_GONE after ${i} ticks"; break; }
  sleep 30
done
echo "=== trainer state ==="; pgrep -af "$SIG" || echo "confirmed gone"
date
# Run the fixed eval driver (serves final resume_adapter on GPU0, evals on evalfull dir).
bash /home/woori/scratch/nfc_tb_eval_dist.sh
echo "=== FINAL metrics ==="
EVALDIR=$S/JARVIS_tb/taskbench/data_dailylifeapis_evalfull_qwen7b
/home/woori/venvs/seka_env/bin/python -c "import json
d=json.load(open('$EVALDIR/metrics/nfcdist_fc.json'))['overall_overall']
print('FINAL DIST node_f1=%.3f prec=%.3f rec=%.3f samples=%d'%(d['node_micro_f1_no_matching'],d['node_micro_precision_no_matching'],d['node_micro_recall_no_matching'],d['all_samples']))"
echo NFC_DIST_AUTOFINAL_DONE; date
