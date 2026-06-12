#!/bin/bash
# driver_s1_sel4.sh — 야간 후속(NIGHT14_DONE 게이트 후 GPU 2개 이어받기).
# S1 (GPU0): 실-도메인 verified-distill 스모크 — 합성200 + 실(retail+telecom, airline held-out)
#   LoRA SFT → airline census (S0-v2 0.528 대비 = 합성 과적합 교정 판정).
# SEL-4 (GPU1): 7B reverse-likelihood Reviewer 합성 — v3g 풀 위 MBR+Reviewer 재선별
#   (NB가 tb_v3g_mmk0-7 생성했을 때만; 없으면 dpo2g 풀로 폴백).
# 사전등록: S1 airline applies_F1 > 0.528(S0-v2) = 실-spec 도입이 분포갭 교정 / SEL-4 공식
#   link F1 ≥ SEL-1 단독(소수-정답 구제로 +). 둘 다 미달도 정직 기록.
# Run: setsid bash driver_s1_sel4.sh </dev/null >/dev/null 2>&1 &
set -u
R=/home/woori/workspace_common/boltzmann-attention-pi
T2=$R/scripts/distill/tau2
SC=$R/scripts/distill/taskbench
TB=/home/woori/scratch/JARVIS_tb/taskbench
TBPRED=$TB/data_multimedia_sub500/predictions
RUNS=$R/reports/facet_rft_2026/phase4_distill/sft_runs
PY=/home/woori/venvs/seka_env/bin/python
IP=/home/woori/scratch/tbeval_venv/bin/python
VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
S=/home/woori/scratch
OUT=$S/a2_s1
mkdir -p $OUT
exec > $OUT/s1_sel4_driver.log 2>&1
set -x
cd $R && git pull --ff-only -q

# ---- NIGHT14_DONE 게이트 (최대 6h 대기; 없으면 GPU 비는지 직접 확인) ----
for i in $(seq 1 360); do
  grep -q NIGHT14_DONE $S/tb_night14.log 2>/dev/null && break
  if ! nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; then
    echo "GPUS_FREE_no_sentinel"; break; fi
  sleep 60
done
sleep 20  # kill_gpu 여파 정리

kill_gpu() {
  for p in $(nvidia-smi --id=$1 --query-compute-apps=pid --format=csv,noheader); do
    kill -9 $p 2>/dev/null; done; sleep 8
}

# ====================== SEL-4 (GPU1, 백그라운드) ======================
(
kill_gpu 1
AR_TAG=tb_dpo2g_mmk; AR_GRP=dpo2g
[ -f $TBPRED/tb_v3g_mmk7.json ] && { AR_TAG=tb_v3g_mmk; AR_GRP=v3g; }
CUDA_VISIBLE_DEVICES=1 VLLM_PORT=8201 setsid nohup $VLLM serve Qwen/Qwen2.5-7B-Instruct \
  --port 8001 --served-model-name base_model --max-model-len 8192 \
  --gpu-memory-utilization 0.85 > $S/vllm_sel4.log 2>&1 &
ok=0
for i in $(seq 1 90); do
  curl -s localhost:8001/v1/models | grep -q base_model && ok=1 && break; sleep 10
done
if [ $ok = 1 ]; then
  $IP $SC/tb_reviewer_select.py --tb_dir $TB --ar_tag $AR_TAG --ar_group $AR_GRP \
    --endpoint http://localhost:8001/v1 --served base_model --lam 1.0 \
    --out $TBPRED/tb_sel4_${AR_GRP}.json
  $IP $SC/tb_build_eval.py --tb_dir $TB --domain data_multimedia \
    --pred_file $TBPRED/tb_sel4_${AR_GRP}.json \
    --dst $TB/data_multimedia_sub500_eval_tb_sel4_${AR_GRP} --llm tb_sel4_${AR_GRP} \
    > $OUT/sel4_${AR_GRP}.txt 2>&1
  grep -hE "link_binary_f1|node_micro_f1_no" $OUT/sel4_${AR_GRP}.txt
  echo "SEL4_DONE ($AR_GRP)"
else
  echo "SEL4_SERVE_FAIL"
fi
kill_gpu 1
) &
SEL4_PID=$!

# ====================== S1 (GPU0) ======================
$PY $T2/t2_a2_s1_build.py --synth $T2/specs/a2_s0_dataset_v6.jsonl \
  --real retail:$T2/specs/retail_policy.md:$S/tau2_adapter/retail_tool_catalog.json:$T2/specs/retail_gate_spec_fable5.json \
  --real telecom:$T2/specs/s1_inputs/telecom_policy.md:$T2/specs/s1_inputs/telecom_tool_catalog.json:$T2/specs/s1_inputs/telecom_gate_spec_fable5.json \
  --oversample 8 --out $T2/specs/a2_s1_dataset.jsonl || { echo S1_BUILD_FAIL; }

# build SFT (holdout 0 — airline census가 held-out 평가)
$PY $T2/t2_a2_s0_build_sft.py --dataset $T2/specs/a2_s1_dataset.jsonl \
  --out $OUT/sft_a2_s1.jsonl --holdout 0 || { echo S1_SFTBUILD_FAIL; }

kill_gpu 0
ADAPTER=$RUNS/qwen7b_a2_s1
rm -f $ADAPTER/train_meta.json
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY \
  $R/scripts/distill/lora_train_chat_toolcall.py \
  --base-model Qwen/Qwen2.5-7B-Instruct --device cuda:0 --max-seq-len 8192 \
  --epochs 3 --lora-r 16 --val-frac 0.06 --skip-overlong \
  --train-jsonl $OUT/sft_a2_s1.jsonl --out-dir $ADAPTER || { echo S1_TRAIN_FAIL; }
echo S1_TRAIN_DONE

# adapter size-stability 가드
for i in $(seq 1 30); do
  s1=$(stat -c %s $ADAPTER/adapter_model.safetensors 2>/dev/null || echo 0); sleep 5
  s2=$(stat -c %s $ADAPTER/adapter_model.safetensors 2>/dev/null || echo -1)
  [ "$s1" = "$s2" ] && [ "$s1" != "0" ] && break
done

# airline held-out census (base vs S1)  — S0-v2와 동일 평가축
CUDA_VISIBLE_DEVICES=0 VLLM_PORT=8101 setsid nohup $VLLM serve Qwen/Qwen2.5-7B-Instruct \
  --port 8000 --max-model-len 16384 --enable-lora --lora-modules a2s1=$ADAPTER \
  > $S/vllm_s1.log 2>&1 &
ok=0
for i in $(seq 1 90); do
  curl -s localhost:8000/v1/models | grep -q '"data"' && ok=1 && break; sleep 10
done
if [ $ok = 1 ]; then
  $PY $T2/t2_a2_size_census.py --target airline --ref_dir $T2/specs \
    --policy $T2/specs/airline_policy.md --catalog $T2/specs/airline_tool_catalog.json \
    --model qwen7b_base:http://localhost:8000/v1:Qwen/Qwen2.5-7B-Instruct \
    --model qwen7b_s1:http://localhost:8000/v1:a2s1 \
    --out $OUT/s1_airline_census | tee $OUT/s1_census.txt
  echo S1_CENSUS_DONE
else
  echo S1_SERVE_FAIL
fi
kill_gpu 0

wait $SEL4_PID || true
echo "S1_SEL4_ALL_DONE $(date)"
