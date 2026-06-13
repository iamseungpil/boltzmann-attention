#!/bin/bash
# P-lora 재발사(순차 서빙 — multi-LoRA 버그 우회) + P-D0 재발사(버그수정분)
# day13 [B] 실패(dl_0~6 빈 파일·multi-LoRA 라우팅) → 어댑터 1개씩 serve→infer→kill.
# P-lora = 이종-목적 어댑터 8종 각 1샘플 (같은 7B base·목적-다양 = H6와 구분되는 다양성원).
# 사전등록: SELECTOR_DESIGN ⑸ — P-lora 다양성 > P-temp(0.0114) ∧ ≈H6(0.024)면 "목적-다양이
#   이종-어댑터 다양성의 본체" / P-D0: 형식 이중관문(파싱≥0.5 ∧ snap-valid≥0.8).
# log: /home/woori/scratch/plora_pd0.log, sentinel PLORA_PD0_DONE
set -u
R=/home/woori/workspace_common/boltzmann-attention-pi
SC=$R/scripts/distill/taskbench
TB=/home/woori/scratch/JARVIS_tb/taskbench
TBPRED=$TB/data_multimedia_sub500/predictions
RUNS=$R/reports/facet_rft_2026/phase4_distill/sft_runs
VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
IP=/home/woori/scratch/tbeval_venv/bin/python
PY=/home/woori/venvs/seka_env/bin/python
S=/home/woori/scratch
exec > $S/plora_pd0.log 2>&1
set -x
cd $R && git pull --ff-only -q
[ -f $S/tb_guided_mm_schema.json ] || $IP $SC/tb_guided_schema.py \
  --tool_desc $TB/data_multimedia/tool_desc.json --dep resource --out $S/tb_guided_mm_schema.json
$IP $SC/tb_guided_patch.py $TB/inference.py || true

kill_gpu() { for p in $(nvidia-smi --id=$1 --query-compute-apps=pid --format=csv,noheader); do
  kill -9 $p 2>/dev/null; done; sleep 8; }

# ---------- P-D0 (GPU1, 백그라운드) ----------
(
kill_gpu 1
mkdir -p $S/dream_p_d0
CUDA_VISIBLE_DEVICES=1 timeout 10000 $PY $SC/tb_diffusion_sample.py \
  --data_dir $TB/data_multimedia --out_prefix $S/dream_p_d0/dream --n 50 --k 4 \
  > $S/plora_pd0_pd0.txt 2>&1 || echo "PD0_FAIL(or timeout)" >> $S/plora_pd0_pd0.txt
tail -15 $S/plora_pd0_pd0.txt
kill_gpu 1
echo "PD0_DONE"
) &
PD0_PID=$!

# ---------- P-lora (GPU0, 순차 서빙) ----------
declare -a ADS=("dpo2=$RUNS/qwen7b_tb_dpo2_mm" "v3mix=$RUNS/qwen7b_tb_dpo_v3mix" \
  "struct=$RUNS/qwen7b_tb_dpo_struct" "cost=$RUNS/qwen7b_tb_dpo_cost" \
  "lodomm=$RUNS/qwen7b_tb_lodo_mm" "daily=$RUNS/qwen7b_tb_lodo_daily" \
  "rft2=$RUNS/qwen7b_tb_rft2_mm" "rft=$RUNS/qwen7b_tb_rft_mm")
n=0
for ad in "${ADS[@]}"; do
  NAME=${ad%%=*}; PATH_=${ad#*=}
  kill_gpu 0
  CUDA_VISIBLE_DEVICES=0 VLLM_PORT=8130 setsid nohup $VLLM serve Qwen/Qwen2.5-7B-Instruct \
    --port 8000 --served-model-name base_model --enable-lora \
    --lora-modules m=$PATH_ --max-model-len 8192 --gpu-memory-utilization 0.85 \
    > $S/vllm_plora_$NAME.log 2>&1 &
  ok=0
  for i in $(seq 1 90); do curl -s localhost:8000/v1/models | grep -q '"m"' && ok=1 && break; sleep 10; done
  if [ $ok = 1 ]; then
    rm -f $TBPRED/m.json
    (cd $TB && TB_GUIDED=1 TB_GUIDED_SCHEMA=$S/tb_guided_mm_schema.json \
      $IP inference.py --data_dir data_multimedia_sub500 --api_addr localhost \
      --api_port 8000 --api_key dummy --llm m --multiworker 8 \
      --dependency_type resource --temperature 0.8)
    mv $TBPRED/m.json $TBPRED/tb_dl_${n}.json
    echo "PLORA_DONE $NAME -> tb_dl_$n ($(wc -l < $TBPRED/tb_dl_${n}.json) lines)"
  else
    echo "PLORA_SERVE_FAIL $NAME"
  fi
  n=$((n+1))
done
kill_gpu 0
echo "PLORA_ALL_DONE"

# ---------- 분석 (P-lora 합류) ----------
$PY $SC/tb_divgen_analyze.py --tb_dir $TB \
  --policy "P-temp=$TBPRED/tb_dt_*.json" \
  --policy "P-unguided=$TBPRED/tb_du_*.json" \
  --policy "P-lora=$TBPRED/tb_dl_*.json" \
  --policy "REF-dpo2g=$TBPRED/tb_dpo2g_mmk*.json" \
  --out_prefix $TBPRED/tb_divsel2 > $S/plora_divgen.txt 2>&1 || true
grep -aE "\[P-|\[REF|회귀|이득" $S/plora_divgen.txt

wait $PD0_PID || true
echo "PLORA_PD0_DONE $(date)"
