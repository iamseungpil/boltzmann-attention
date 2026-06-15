#!/bin/bash
# Interim full-catalog TaskBench-native eval of the DISTRACTOR-augmented dist adapter
# (qwen7b_nfc_lodo_daily_dist) at its current resume_adapter step, on GPU0 ONLY.
# Question: does distractor-training recover the full-catalog collapse (node 0.918->0.448)?
# Baseline ref: base_fc node_micro_f1=0.918 (harness-verified, full 40-tool catalog).
# SAFETY: GPU0 only. NEVER touch GPU1 (training runs there). Reuses existing base_fc.json.
set -u
DISTROOT=/home/woori/scratch/sft_runs/qwen7b_nfc_lodo_daily_dist
SNAP=$DISTROOT/eval_snapshot   # frozen copy so live training writes don't race the serve
NAME=nfcdist; GPU=0; PORT=8013; N=500
DATA=/home/woori/scratch/JARVIS_tb/taskbench/data_dailylifeapis
PY=/home/woori/venvs/seka_env/bin/python
VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
S=/home/woori/scratch
LOG=$S/nfc_tb_eval_dist.log
exec > $LOG 2>&1; set -x; date

# Freeze a snapshot of the current adapter (resume_adapter is overwritten every save-50).
rm -rf $SNAP; mkdir -p $SNAP; cp -a $DISTROOT/resume_adapter/. $SNAP/
ls -la $SNAP

# Kill ONLY GPU0 compute procs (training is on GPU1 and must be left alone).
for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 4

CUDA_VISIBLE_DEVICES=$GPU setsid nohup $VLLM serve Qwen/Qwen2.5-7B-Instruct --port $PORT \
  --enable-auto-tool-choice --tool-call-parser hermes --max-model-len 16384 \
  --enable-lora --lora-modules ${NAME}=$SNAP --max-lora-rank 32 > $S/vllm_nfctb_dist.log 2>&1 &
ok=0; for i in $(seq 1 60); do curl -s localhost:$PORT/v1/models 2>/dev/null | grep -q "$NAME" && ok=1 && break; sleep 10; done
[ $ok = 1 ] || { echo SERVE_FAIL; tail -30 $S/vllm_nfctb_dist.log; exit 1; }
B=http://localhost:$PORT/v1

echo '===== GEN dist adapter ====='
$PY $S/nfc_tb_gen.py --data $DATA --base $B --model $NAME --out $DATA/predictions/nfcdist_fc.json --n $N

# Reuse existing base_fc.json (adapter-independent). Regenerate only if missing.
if [ ! -f $DATA/predictions/base_fc.json ]; then
  echo '===== GEN base (missing) ====='
  $PY $S/nfc_tb_gen.py --data $DATA --base $B --model Qwen/Qwen2.5-7B-Instruct --out $DATA/predictions/base_fc.json --n $N
fi

for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done

echo '===== EVAL dist adapter (nfcdist_fc) ====='
$PY $S/tb_evaluate.py --data_dir $DATA --llm nfcdist_fc -m f1 -m link -m argument -s all -t temporal 2>&1 | grep -iE 'node_micro_f1|link_binary_f1|argument_task_argname_binary|argument_task_argname_value' | head
echo '===== EVAL base (base_fc) ref ====='
$PY $S/tb_evaluate.py --data_dir $DATA --llm base_fc -m f1 -m link -m argument -s all -t temporal 2>&1 | grep -iE 'node_micro_f1|link_binary_f1|argument_task_argname_binary|argument_task_argname_value' | head
echo NFC_TB_EVAL_DIST_DONE; date
