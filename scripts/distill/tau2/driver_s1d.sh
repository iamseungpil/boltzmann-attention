#!/bin/bash
# S1-diag (S1 기각 후속 — 2026-06-13 설계·사전등록 동결, DAY13_DONE 게이트 후 GPU0)
# 가설 분리 3-arm:
#  H4 구체성 갭: 합성 db_check가 플레이스홀더("<...>") ↔ 실 spec은 구체 조건문 — 모델이
#     구체 술어 작성 미학습. 처방 = 구체-술어 합성 60쌍(t2_a2_concrete_gen, 전과정 프로그램)
#     을 v6에 합쳐 v7 → 재학습(s1d). 사전등록: airline applies_F1 > 0.528 ∧ n_gates ≥ 3.
#  H3 형식 불일치: census는 retail 1-shot 포함 ↔ 학습은 zero-shot. 사전등록: --no_oneshot
#     평가에서 어댑터 점수 ↑ (방향). base는 1-shot 제거 시 ↓ 예상 (예시 의존).
#  둘 다 기각 시: 남는 가설 = 도메인 다양성(P5 대기, dose-response 그대로) — 가설 공간 축소.
# arms: census {base, s0v2, s1d} × {1shot, 0shot} (multi-LoRA 1-serve).
# log: /home/woori/scratch/a2_s1d/s1d.log, sentinel S1D_DONE (day13_summary에 결과 append)
set -u
R=/home/woori/workspace_common/boltzmann-attention-pi
T2=$R/scripts/distill/tau2
RUNS=$R/reports/facet_rft_2026/phase4_distill/sft_runs
PY=/home/woori/venvs/seka_env/bin/python
VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
S=/home/woori/scratch
OUT=$S/a2_s1d
mkdir -p $OUT
exec > $OUT/s1d.log 2>&1
set -x
cd $R && git pull --ff-only -q

# DAY13_DONE 게이트 (최대 9h; GPU0 빈 것도 확인)
for i in $(seq 1 540); do
  grep -q "DAY13_DONE" $S/day13.log 2>/dev/null && break
  sleep 60
done
for i in $(seq 1 60); do
  nvidia-smi --id=0 --query-compute-apps=pid --format=csv,noheader | grep -q . || break
  sleep 30
done

kill_gpu0() {
  for p in $(nvidia-smi --id=0 --query-compute-apps=pid --format=csv,noheader); do
    kill -9 $p 2>/dev/null; done; sleep 8
}

# 1. 구체-술어 합성 60 + v7 병합 + 실 2도메인(oversample 4)
$PY $T2/t2_a2_concrete_gen.py --n 60 --seed 8 --out $T2/specs/a2_s1d_concrete.jsonl
cat $T2/specs/a2_s0_dataset_v6.jsonl $T2/specs/a2_s1d_concrete.jsonl \
  > $T2/specs/a2_s0_dataset_v7.jsonl
$PY $T2/t2_a2_s1_build.py --synth $T2/specs/a2_s0_dataset_v7.jsonl \
  --real retail:$T2/specs/retail_policy.md:$S/tau2_adapter/retail_tool_catalog.json:$T2/specs/retail_gate_spec_fable5.json \
  --real telecom:$T2/specs/s1_inputs/telecom_policy.md:$T2/specs/s1_inputs/telecom_tool_catalog.json:$T2/specs/s1_inputs/telecom_gate_spec_fable5.json \
  --oversample 4 --out $T2/specs/a2_s1d_dataset.jsonl
$PY $T2/t2_a2_s0_build_sft.py --dataset $T2/specs/a2_s1d_dataset.jsonl \
  --out $OUT/sft_s1d.jsonl --holdout 0 || { echo S1D_BUILD_FAIL; exit 1; }

# 2. 학습 (GPU0)
kill_gpu0
ADAPTER=$RUNS/qwen7b_a2_s1d
rm -f $ADAPTER/train_meta.json
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY \
  $R/scripts/distill/lora_train_chat_toolcall.py \
  --base-model Qwen/Qwen2.5-7B-Instruct --device cuda:0 --max-seq-len 8192 \
  --epochs 3 --lora-r 16 --val-frac 0.06 --skip-overlong \
  --train-jsonl $OUT/sft_s1d.jsonl --out-dir $ADAPTER || { echo S1D_TRAIN_FAIL; exit 1; }
echo S1D_TRAIN_DONE
for i in $(seq 1 30); do
  s1=$(stat -c %s $ADAPTER/adapter_model.safetensors 2>/dev/null || echo 0); sleep 5
  s2=$(stat -c %s $ADAPTER/adapter_model.safetensors 2>/dev/null || echo -1)
  [ "$s1" = "$s2" ] && [ "$s1" != "0" ] && break
done

# 3. census: {base, s0v2, s1d} x {1shot, 0shot}
CUDA_VISIBLE_DEVICES=0 VLLM_PORT=8120 setsid nohup $VLLM serve Qwen/Qwen2.5-7B-Instruct \
  --port 8000 --max-model-len 16384 --enable-lora --max-loras 2 \
  --lora-modules a2s0v2=$RUNS/qwen7b_a2_s0v2 a2s1d=$ADAPTER \
  > $S/vllm_s1d.log 2>&1 &
ok=0
for i in $(seq 1 90); do
  curl -s localhost:8000/v1/models | grep -q '"data"' && ok=1 && break; sleep 10
done
if [ $ok = 1 ]; then
  MODELS="--model base:http://localhost:8000/v1:Qwen/Qwen2.5-7B-Instruct \
          --model s0v2:http://localhost:8000/v1:a2s0v2 \
          --model s1d:http://localhost:8000/v1:a2s1d"
  echo "== 1-shot (기존 프로토콜)" | tee $OUT/s1d_census.txt
  $PY $T2/t2_a2_size_census.py --target airline --ref_dir $T2/specs \
    --policy $T2/specs/airline_policy.md --catalog $T2/specs/airline_tool_catalog.json \
    $MODELS --out $OUT/cen_1shot 2>&1 | tee -a $OUT/s1d_census.txt
  echo "== 0-shot (H3 — 학습 형식 일치)" | tee -a $OUT/s1d_census.txt
  $PY $T2/t2_a2_size_census.py --target airline --ref_dir $T2/specs --no_oneshot \
    --policy $T2/specs/airline_policy.md --catalog $T2/specs/airline_tool_catalog.json \
    $MODELS --out $OUT/cen_0shot 2>&1 | tee -a $OUT/s1d_census.txt
  { echo "== [S1-diag] H4/H3 census"; cat $OUT/s1d_census.txt; } >> $S/day13_summary.txt
else
  echo S1D_SERVE_FAIL
fi
kill_gpu0
echo "S1D_DONE $(date)" | tee -a $S/day13_summary.txt
