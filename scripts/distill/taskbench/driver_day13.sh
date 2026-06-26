#!/bin/bash
# ★주간 배치 2026-06-13 09:30→17:00 (무인 — 검토된 전 방향 자동 실행, 사전등록 동결)
# GPU0 체인: [A] 다양성-생성 샘플링 P-temp(8: temp 0.5/0.8/1.1/1.4×2)+P-unguided(8: guided OFF)
#            → [D] P-D0 diffusion 형식게이트 스모크 (Dream-7B, 50×K4, timeout 격리)
# GPU1 체인: [substrate_hf 완료 대기] → [B] P-lora (이종-목적 어댑터 8종 × 1샘플)
#            → [F] p1 분산통제 τ² arm (gate r3-구성, user-sim temp 0.0 ×4 trials)
# CPU: [E] V-1/V-2 MAV-식 집계 (즉시) → [C] 다양성 분석+회귀+공식 eval (A·B 합류 후)
# ⚠️P-prompt arm 제외 (inference.py에 system 주입점 부재 — 설계서에 사유 기록)
# 사전등록: SELECTOR_DESIGN 큐 ⑸(P-unguided 다양성>P-temp·P-lora 최상·회귀 SIG) §6(V-2 ⓥ1-ⓥ3)
#           TB_DIFFUSION §3b(P-D0 이중관문) PORTFOLIO §3.7d(p1: ut0에서 flaky 감소=분산원 확정)
# log: /home/woori/scratch/day13.log, 요약: /home/woori/scratch/day13_summary.txt, sentinel DAY13_DONE
R=/home/woori/workspace_common/boltzmann-attention-pi
SC=$R/scripts/distill/taskbench
TB=/home/woori/scratch/JARVIS_tb/taskbench
TBPRED=$TB/data_multimedia_sub500/predictions
RUNS=$R/reports/facet_rft_2026/phase4_distill/sft_runs
VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
IP=/home/woori/scratch/tbeval_venv/bin/python
PY=/home/woori/venvs/seka_env/bin/python
S=/home/woori/scratch
SUM=$S/day13_summary.txt
exec > $S/day13.log 2>&1
set -x
cd $R && git pull --ff-only -q
: > $SUM

kill_gpu() {
  for p in $(nvidia-smi --id=$1 --query-compute-apps=pid --format=csv,noheader); do
    kill -9 $p 2>/dev/null; done; sleep 8
}
serve_lora() { # gpu port vllm_port "name=path name=path..."
  local GPU=$1 PORT=$2 VP=$3 MODS=""
  shift 3
  for m in "$@"; do MODS="$MODS --lora-modules $m"; done
  CUDA_VISIBLE_DEVICES=$GPU VLLM_PORT=$VP setsid nohup $VLLM serve Qwen/Qwen2.5-7B-Instruct \
    --port $PORT --served-model-name base_model --enable-lora $MODS \
    --max-model-len 8192 --gpu-memory-utilization 0.85 --max-loras 8 \
    > $S/vllm_day13_g$GPU.log 2>&1 &
  for i in $(seq 1 90); do
    curl -s localhost:$PORT/v1/models | grep -q base_model && return 0; sleep 10
  done
  return 1
}
infer() { # port llm temp guided(1/0) outfile
  local P=$1 L=$2 T=$3 G=$4 O=$5
  rm -f $TBPRED/$L.json
  if [ "$G" = "1" ]; then
    (cd $TB && TB_GUIDED=1 TB_GUIDED_SCHEMA=$S/tb_guided_mm_schema.json \
      $IP inference.py --data_dir data_multimedia_sub500 --api_addr localhost \
      --api_port $P --api_key dummy --llm $L --multiworker 8 \
      --dependency_type resource --temperature $T)
  else
    (cd $TB && $IP inference.py --data_dir data_multimedia_sub500 --api_addr localhost \
      --api_port $P --api_key dummy --llm $L --multiworker 8 \
      --dependency_type resource --temperature $T)
  fi
  mv $TBPRED/$L.json $O
}

# ---------- [E] V-1/V-2 (CPU, 즉시) ----------
(cd $SC && $PY tb_mav_select.py --tb_dir $TB > $S/day13_v2.txt 2>&1) || true
{ echo "== [E] V-1/V-2 MAV"; cat $S/day13_v2.txt; } >> $SUM

# ---------- GPU1 체인 (백그라운드) ----------
(
for i in $(seq 1 240); do
  grep -q "SUBSTRATE_HF_DONE" $S/substrate_hf.log 2>/dev/null && break; sleep 60
done
kill_gpu 1
# [B] P-lora: 이종-목적 어댑터 8종
MODS="dl_dpo2=$RUNS/qwen7b_tb_dpo2_mm dl_v3mix=$RUNS/qwen7b_tb_dpo_v3mix \
dl_struct=$RUNS/qwen7b_tb_dpo_struct dl_cost=$RUNS/qwen7b_tb_dpo_cost \
dl_lodomm=$RUNS/qwen7b_tb_lodo_mm dl_daily=$RUNS/qwen7b_tb_lodo_daily \
dl_rft2=$RUNS/qwen7b_tb_rft2_mm dl_rft=$RUNS/qwen7b_tb_rft_mm"
if serve_lora 1 8001 8211 $MODS; then
  n=0
  for m in $MODS; do
    NAME=${m%%=*}
    infer 8001 $NAME 0.8 1 $TBPRED/tb_dl_${n}.json || true
    echo "B_DONE $NAME -> tb_dl_$n"; n=$((n+1))
  done
fi
kill_gpu 1
echo "B_ALL_DONE"
# [F] p1 분산통제 arm (GPU1, tau2)
source /home/woori/.openrouter_key
CUDA_VISIBLE_DEVICES=1 VLLM_PORT=8212 setsid nohup $VLLM serve Qwen/Qwen2.5-7B-Instruct \
  --port 8351 --enable-auto-tool-choice --tool-call-parser hermes --max-model-len 16384 \
  > $S/vllm_day13_t2.log 2>&1 &
ok=0
for i in $(seq 1 90); do
  curl -s localhost:8351/v1/models | grep -q Qwen && ok=1 && break; sleep 10
done
if [ $ok = 1 ]; then
  rm -rf /home/woori/scratch/tau2-bench/data/simulations/retail_7b_gate_r3_ut0
  cd /home/woori/scratch/tau2-bench
  PYTHONPATH=src:$R/scripts/distill/tau2 $PY $R/scripts/distill/tau2/t2_run_gated.py \
    --gate 1 --num_trials 4 --user_llm "openrouter/openai/gpt-4.1" --user_temp 0.0 \
    --save_to retail_7b_gate_r3_ut0 || true
  cd $R
fi
kill_gpu 1
echo "F_DONE"
) &
G1_PID=$!

# ---------- GPU0 체인: [A] P-temp + P-unguided ----------
kill_gpu 0
[ -f $S/tb_guided_mm_schema.json ] || $IP $SC/tb_guided_schema.py \
  --tool_desc $TB/data_multimedia/tool_desc.json --dep resource \
  --out $S/tb_guided_mm_schema.json
$IP $SC/tb_guided_patch.py $TB/inference.py || true
if serve_lora 0 8000 8210 "dt=$RUNS/qwen7b_tb_dpo2_mm"; then
  n=0
  for T in 0.5 0.5 0.8 0.8 1.1 1.1 1.4 1.4; do
    infer 8000 dt $T 1 $TBPRED/tb_dt_${n}.json || true
    echo "A_TEMP_DONE $n (T=$T)"; n=$((n+1))
  done
  for n2 in 0 1 2 3 4 5 6 7; do
    infer 8000 dt 0.8 0 $TBPRED/tb_du_${n2}.json || true
    echo "A_UNG_DONE $n2"
  done
fi
kill_gpu 0
echo "A_DONE"

# ---------- [D] P-D0 diffusion 스모크 (GPU0, timeout 격리) ----------
CUDA_VISIBLE_DEVICES=0 timeout 9000 $PY $SC/tb_diffusion_sample.py \
  --data_dir $TB/data_multimedia --out_prefix $S/dream_p_d0/dream --n 50 --k 4 \
  > $S/day13_pd0.txt 2>&1 || echo "PD0_FAIL(or timeout)" >> $S/day13_pd0.txt
{ echo "== [D] P-D0"; tail -20 $S/day13_pd0.txt; } >> $SUM
kill_gpu 0

# ---------- [C] 분석 (A·B 합류 후) ----------
wait $G1_PID || true
(cd $SC && $PY tb_divgen_analyze.py --tb_dir $TB \
  --policy "P-temp=$TBPRED/tb_dt_*.json" \
  --policy "P-unguided=$TBPRED/tb_du_*.json" \
  --policy "P-lora=$TBPRED/tb_dl_*.json" \
  --policy "REF-dpo2g=$TBPRED/tb_dpo2g_mmk*.json" \
  --out_prefix $TBPRED/tb_divsel > $S/day13_divgen.txt 2>&1) || true
{ echo "== [C] 다양성-생성 분석"; cat $S/day13_divgen.txt; } >> $SUM
for pol in P-temp P-unguided P-lora; do
  [ -f $TBPRED/tb_divsel_${pol}.json ] || continue
  $IP $SC/tb_build_eval.py --tb_dir $TB --domain data_multimedia \
    --pred_file $TBPRED/tb_divsel_${pol}.json \
    --dst $TB/data_multimedia_sub500_eval_divsel_${pol} --llm divsel_${pol} \
    > $S/day13_off_${pol}.txt 2>&1 || true
  { echo "== [C] 공식 $pol"; grep -hE "link_binary_f1|node_micro" $S/day13_off_${pol}.txt; } >> $SUM
done
{ echo "== [F] p1 ut0 compliance"
  cat /home/woori/scratch/tau2-bench/data/simulations/retail_7b_gate_r3_ut0/compliance.json 2>/dev/null
  echo "== substrate_hf"
  grep -hE "link_binary_f1" $S/hf_c0.txt $S/hf_sel1.txt $S/hf_sel4.txt 2>/dev/null
} >> $SUM
echo "DAY13_DONE $(date)" | tee -a $SUM
