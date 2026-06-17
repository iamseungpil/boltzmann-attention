#!/bin/bash
# Airtight closure of §17 (original C8 2nd-gate negative): re-eval the OLD 5-op routing adapters
# (trained before substitute/create existed) on the CURRENT retail+airline cases + grounded resolver,
# identical footing to §21's 7-op MD_route. Expect: 5-op CANNOT emit substitute -> over-comparative ->
# low acc, vs 7-op MD_route 0.44 + base 0.28. Eval-only (no training). Waits for widen_retrain to free
# GPU0, then serves each adapter there. Detached.
set -u
REPO=/home/woori/workspace_common/boltzmann-attention-pi
MA=$REPO/scripts/distill/ma; S=/home/woori/scratch/depth/c8; CASES=$MA/cases
PY=/home/woori/venvs/seka_env/bin/python; VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
BASE=Qwen/Qwen2.5-7B-Instruct; GPU=0; PORT=8070; OUT=$S/multidomain/results
LOG=$S/multidomain/logs/c8_5op_reeval.log; mkdir -p $(dirname $LOG) $OUT; exec > $LOG 2>&1; set -x; date
# wait for widen_retrain (GPU0) to finish
while pgrep -f "[w]iden_retrain.sh" >/dev/null; do echo "widen_retrain running"; date; sleep 120; done
echo "widen done"; date; sleep 30
kg(){ for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 4; }
cd $REPO && git pull --ff-only 2>&1 | tail -1
for A in DIV_ep3 A2_ep3; do
  [ -d $S/adapters/$A ] || { echo "NO_ADAPTER $A"; continue; }
  echo "##### 5-op reeval $A #####"; date
  kg
  CUDA_VISIBLE_DEVICES=$GPU setsid nohup $VLLM serve $BASE --port $PORT --max-model-len 16384 \
    --enable-lora --max-lora-rank 64 --lora-modules ${A}=$S/adapters/$A > $S/multidomain/logs/serve_5op_$A.log 2>&1 &
  ok=0; for i in $(seq 1 100); do curl -s localhost:$PORT/v1/models 2>/dev/null | grep -q '"id"' && ok=1 && break; sleep 6; done
  [ $ok = 1 ] || { echo "SERVE_FAIL $A"; tail -15 $S/multidomain/logs/serve_5op_$A.log; continue; }
  for D in retail airline; do
    $PY $MA/tau2_op_eval.py --cases $CASES/tau2_${D}_cases.jsonl --base http://localhost:$PORT/v1 \
      --model $A --gloss 0 --out $OUT/5op_${A}__${D}_g0.json 2>&1 | grep -E "overall new_item|recognition|op dist"
  done
  kg
done
echo "C8_5OP_REEVAL_DONE"; date