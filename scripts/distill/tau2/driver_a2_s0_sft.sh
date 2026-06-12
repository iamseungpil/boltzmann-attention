#!/bin/bash
# S0 스모크 드라이버 (HANDOFF day4 §0.2-②):
# build(135쌍, holdout 15) -> encode sanity -> 7B LoRA SFT(GPU0) ->
# serve base+adapter -> airline 재census + holdout 평가 -> GPU 해제.
# Run: setsid bash driver_a2_s0_sft.sh </dev/null >/dev/null 2>&1 &
set -u
REPO=/home/woori/workspace_common/boltzmann-attention-pi
T2=$REPO/scripts/distill/tau2
OUT=/home/woori/scratch/a2_s0
RUNS=$REPO/reports/facet_rft_2026/phase4_distill/sft_runs
PY=/home/woori/venvs/seka_env/bin/python
ADAPTER=$RUNS/qwen7b_a2_s0
mkdir -p $OUT
exec > $OUT/s0_driver.log 2>&1
cd $REPO && git pull --ff-only

# GPU0 비어있는지 확인 (충돌 금지)
if nvidia-smi --id=0 --query-compute-apps=pid --format=csv,noheader | grep -q .; then
  echo S0_ABORT_GPU0_BUSY
  exit 1
fi

# 1. build
$PY $T2/t2_a2_s0_build_sft.py --dataset $T2/specs/a2_s0_dataset_v3.jsonl \
  --out $OUT/sft_a2_s0.jsonl --holdout 15 || { echo S0_BUILD_FAIL; exit 1; }

# 2. encode sanity (CPU)
$PY $REPO/scripts/distill/lora_train_chat_toolcall.py --train-jsonl $OUT/sft_a2_s0.jsonl \
  --out-dir $OUT/encode_smoke --encode-only --max-seq-len 8192 \
  || { echo S0_ENCODE_FAIL; exit 1; }

# 3. train (GPU0)
rm -f $ADAPTER/train_meta.json
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY \
  $REPO/scripts/distill/lora_train_chat_toolcall.py \
  --base-model Qwen/Qwen2.5-7B-Instruct --device cuda:0 --max-seq-len 8192 \
  --epochs 3 --lora-r 16 --val-frac 0.08 --skip-overlong \
  --train-jsonl $OUT/sft_a2_s0.jsonl --out-dir $ADAPTER \
  || { echo S0_TRAIN_FAIL; exit 1; }
echo S0_TRAIN_DONE

# 4. adapter 저장 size-stability 가드 (vllm serve<->save 레이스, day-4 gotcha #1)
for i in $(seq 1 30); do
  s1=$(stat -c %s $ADAPTER/adapter_model.safetensors 2>/dev/null || echo 0)
  sleep 5
  s2=$(stat -c %s $ADAPTER/adapter_model.safetensors 2>/dev/null || echo -1)
  [ "$s1" = "$s2" ] && [ "$s1" != "0" ] && break
done

# 5. serve base+adapter 동시 (한 serve로 paired 비교)
CUDA_VISIBLE_DEVICES=0 /home/woori/venvs/tau2_vllm_env/bin/vllm serve \
  Qwen/Qwen2.5-7B-Instruct --port 8000 --max-model-len 16384 \
  --enable-lora --lora-modules a2s0=$ADAPTER > $OUT/vllm_s0.log 2>&1 &
PS=$!
up=0
for i in $(seq 1 120); do
  curl -s http://localhost:8000/v1/models | grep -q '"data"' && { up=1; break; }
  sleep 5
done
if [ "$up" != "1" ]; then echo S0_SERVE_FAIL; kill $PS 2>/dev/null; exit 1; fi

# 6a. airline 재census (baseline 비교축)
$PY $T2/t2_a2_size_census.py --target airline --ref_dir $T2/specs \
  --policy $T2/specs/airline_policy.md --catalog $T2/specs/airline_tool_catalog.json \
  --model qwen7b_base:http://localhost:8000/v1:Qwen/Qwen2.5-7B-Instruct \
  --model qwen7b_s0:http://localhost:8000/v1:a2s0 \
  --out $OUT/post_s0_census
echo S0_RECENSUS_DONE

# 6b. 합성 holdout 15쌍 평가 (EM·gate_recall·applies_F1 — 민감 신호)
$PY $T2/t2_a2_s0_eval.py --holdout $OUT/sft_a2_s0_holdout.jsonl \
  --endpoint http://localhost:8000/v1 \
  --model base:Qwen/Qwen2.5-7B-Instruct --model s0:a2s0 \
  --out $OUT/holdout_eval
echo S0_HOLDOUT_DONE

kill $PS 2>/dev/null
sleep 15
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
  kill -9 "$pid" 2>/dev/null
done
echo S0_DRIVER_DONE
