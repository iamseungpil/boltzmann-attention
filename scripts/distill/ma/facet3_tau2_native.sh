#!/bin/bash
# facet (3) gate① — native op-naming의 진짜 τ²(retail+airline) 전이.
# facet3_native_ep1(synth-only LoRA·resolve_selection)이 *실 τ² NL*서 op 명명+new_item_id 맞히나.
# = §21(op-IR로 retail+airline 0.44)을 native 형식서 재현/개선하나. base floor 포함.
# Usage: facet3_tau2_native.sh <gpu> <port>
set -u
GPU=${1:?gpu}; PORT=${2:?port}
R=/home/woori/workspace_common/boltzmann-attention-pi
MA=$R/scripts/distill/ma
PY=/home/woori/venvs/seka_env/bin/python
VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
BASE=Qwen/Qwen2.5-7B-Instruct
S=/home/woori/scratch
OUT=$S/depth/c8/facet3; mkdir -p $OUT/results $OUT/logs
TAG=facet3_native_ep1
LOG=$OUT/logs/tau2_native_g${GPU}.log
exec > $LOG 2>&1; set -x; date
export PYTHONPATH=$MA

serve(){ for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 4
  if [ -n "$2" ]; then
    CUDA_VISIBLE_DEVICES=$GPU setsid nohup $VLLM serve $BASE --port $PORT --max-model-len 8192 \
      --enable-auto-tool-choice --tool-call-parser hermes --enable-lora --max-lora-rank 64 \
      --lora-modules ${1}=${2} > $OUT/logs/serve_t2_${1}.log 2>&1 &
  else
    CUDA_VISIBLE_DEVICES=$GPU setsid nohup $VLLM serve $BASE --port $PORT --max-model-len 8192 \
      --enable-auto-tool-choice --tool-call-parser hermes > $OUT/logs/serve_t2_base.log 2>&1 &
  fi
  for i in $(seq 1 70); do curl -s localhost:$PORT/v1/models 2>/dev/null | grep -qE "${1}|Qwen2.5-7B" && return 0; sleep 10; done
  echo "SERVE_FAIL $1"; tail -20 $OUT/logs/serve_t2_${1}.log; return 1; }

# τ² 케이스 (retail exchange + airline cabin) 재생성
$PY $MA/ma_gold_extract.py --domain retail  --out $S/ma_eval_cases.jsonl
$PY $MA/ma_gold_extract.py --domain airline --out $S/ma_eval_cases_airline.jsonl
RETAIL=$S/ma_eval_cases.jsonl; AIR=$S/ma_eval_cases_airline.jsonl

evalarm(){ # $1=served-id $2=domain-tag $3=cases
  $PY $MA/tau2_op_eval.py --native 1 --cases "$3" --base http://localhost:$PORT/v1 --model "$1" \
    --out $OUT/results/t2native_${1}__$2.json 2>&1 | sed "s/^/[$1 $2] /" || echo "EVAL_FAIL $1 $2"; }

# arm 1 = base (no lora·served name=BASE)
if serve base ""; then
  evalarm "$BASE" retail "$RETAIL"; evalarm "$BASE" airline "$AIR"
fi
# arm 2 = trained (lora·served name=TAG)
if serve "$TAG" "$S/adapters/$TAG"; then
  evalarm "$TAG" retail "$RETAIL"; evalarm "$TAG" airline "$AIR"
fi

for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done
echo "=== FACET3 τ² NATIVE RESULT (op recognition + new_item_id·§21 op-IR=0.44 대비) ==="
for f in $OUT/results/t2native_*.json; do echo "--- $(basename $f) ---"; $PY -c "import json;d=json.load(open('$f'));o=d['overall'];print('new_item_id acc=%d/%d(%.2f)'%(o[0],o[1],o[0]/max(o[1],1)));r=d['recognition'];print('recognition=%d/%d(%.2f)'%(r[0],r[1],r[0]/max(r[1],1)) if r[1] else 'recog NA');print('op_dist',d['op_dist'])" 2>/dev/null || echo NA; done
date; echo FACET3_TAU2_DONE
