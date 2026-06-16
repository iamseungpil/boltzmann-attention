#!/bin/bash
# node_run_ma_72b.sh — M_A scale_floor Q2 (72B-bf16 ceiling) on an amlt H100x4 node.
#   Clean answer to Q2 that the local A100 80GB could only do at AWQ-Int4 (confounded).
#   Same committed ma/ pipeline (arms A,Bfair,L0,L1,L2a,L2b,L3, base inference, NO SFT),
#   72B-bf16 served TP4. Preemption-safe: idempotent markers + HF result sync.
#   Pre-flight GATEs mirror the local run (29 cases + resolver) so we never fire a blind sweep.
set -u
REPO=/scratch/boltzmann-attention
MA=$REPO/scripts/distill/ma
PY=/scratch/venvs/sop_env/bin/python
VLLM=/scratch/venvs/sop_env/bin/vllm
HF=/scratch/venvs/sop_env/bin/hf
PIP=/scratch/venvs/sop_env/bin/pip
OUT=/scratch/ma_scale; mkdir -p $OUT/logs
TAU=/scratch/tau2-bench/data/tau2/domains/retail
HFREPO=iamseungpil/sopbench-trackb-h200
M=Qwen/Qwen2.5-72B-Instruct
PORT=8033; TAG=Qwen2_5_72B_Instruct_bf16
export PYTHONPATH=$MA
set -x

# 0. restore prior result from HF (preempt may wipe /scratch)
$HF download $HFREPO --repo-type dataset --include "ma_scale/*" --local-dir /scratch/hf_ma 2>>$OUT/logs/restore.log || true
[ -d /scratch/hf_ma/ma_scale ] && cp -rn /scratch/hf_ma/ma_scale/* $OUT/ 2>/dev/null || true
[ -f $OUT/ma_72b_done ] && { echo "ALREADY DONE"; cat $OUT/eval_72b.log | grep -A12 SUMMARY; exit 0; }

# 0b. deps the eval needs beyond node_setup (vllm brings xgrammar; ensure requests)
$PIP install -q requests xgrammar >> $OUT/logs/pip.log 2>&1 || true

# 0c. background HF sync so a preempt keeps partial logs/result
( while true; do $HF upload $HFREPO $OUT ma_scale --repo-type dataset \
    --commit-message "ma_scale 72b sync $(date -u +%FT%TZ)" >> $OUT/logs/sync.log 2>&1; sleep 600; done ) &
SYNC=$!

# 1. tau2-bench retail data (clone once)
if [ ! -f $TAU/tasks.json ]; then
  git clone --depth 1 https://github.com/sierra-research/tau2-bench.git /scratch/tau2-bench >> $OUT/logs/clone.log 2>&1
fi

# 2. GATE: cases==29 + resolver self-test (ODCV-prevention; abort if pipeline is blind)
$PY $MA/ma_gold_extract.py --tasks $TAU/tasks.json --db $TAU/db.json --out $OUT/ma_eval_cases.jsonl >> $OUT/logs/gate.log 2>&1
N=$(wc -l < $OUT/ma_eval_cases.jsonl)
$PY $MA/ma_resolver.py --db $TAU/db.json --test >> $OUT/logs/gate.log 2>&1 || { echo "GATE_RESOLVER_FAIL"; kill $SYNC; exit 1; }
[ "$N" = 29 ] || echo "WARN cases=$N (expected 29; cost denom assumes 29)"

# 3. download + serve 72B-bf16 TP4
$HF download $M >> $OUT/logs/hfdl.log 2>&1 || { echo "DL_FAIL"; kill $SYNC; exit 1; }
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 4
CUDA_VISIBLE_DEVICES=0,1,2,3 setsid nohup $VLLM serve $M --port $PORT --dtype bfloat16 \
  --tensor-parallel-size 4 --max-model-len 8192 --gpu-memory-utilization 0.90 > $OUT/logs/vllm_72b.log 2>&1 &
ok=0; for i in $(seq 1 120); do curl -s localhost:$PORT/v1/models 2>/dev/null | grep -q "72B" && ok=1 && break; sleep 10; done
[ $ok = 1 ] || { echo "SERVE_FAIL"; tail -25 $OUT/logs/vllm_72b.log; kill $SYNC; exit 1; }

# 4. eval 29x7 (same arms as local floor sweep)
$PY $MA/ma_eval.py --cases $OUT/ma_eval_cases.jsonl --base http://localhost:$PORT/v1 \
  --model $M --arms A,Bfair,L0,L1,L2a,L2b,L3 --out $OUT/ma_eval_${TAG}.jsonl > $OUT/eval_72b.log 2>&1
grep -A12 "=== SUMMARY ===" $OUT/eval_72b.log | tail -12
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done
touch $OUT/ma_72b_done

# 5. final sync
kill $SYNC 2>/dev/null || true
$HF upload $HFREPO $OUT ma_scale --repo-type dataset --commit-message "ma_scale 72b FINAL $(date -u +%FT%TZ)" >> $OUT/logs/sync.log 2>&1 || true
echo "MA_72B_NODE_DONE $(date)"
