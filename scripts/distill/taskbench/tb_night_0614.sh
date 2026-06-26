#!/bin/bash
# ★야간 배치 2026-06-13→14 — GPU 2개 병렬 (사전등록 동결)
# NA (GPU0+OpenRouter): τ² retail gate r3 = G4 deny-게이트 + 중립 템플릿 검증 (PORTFOLIO §3.8 사전등록)
#    통제 r2: pass^1 0.1908·G1/G2/G3/G4 위반 0/0/0/0(G4는 운) — 예측: ①G4 위반 0(이번엔 게이트 집행)
#    ②G4 deny 1~3건 발생 ③pass^1 0.184~0.20(r2 동등) ④G1-G3 위반 0 유지. compliance.json 자동(후크).
# NB (GPU1): v3mix+guided K=8 temp0.8 샘플링 sub500 MM → tb_v3g_mmk0-7 (선별기 합성 1단)
# NC (CPU, NB 후): 선별 3-구성 공식 채점 — 통제 C0 = v3g k0 단일
#    C1 = v3g-AR8 + H6 (prior_beta 2) — 예측: link F1 ≥ 68 (dpo2g-기반 67.22 초과 = best-stack 합성 이득)
#    C2 = C1 + Track-B 6 proposer (32b/72b/235b ±guided 등) — 예측: ≥ C1 (+1~2, 풀 확장·대형 prior)
# ND (CPU, 선행): 기존 dpo2g-풀 + Track-B 확장만 (NB 무관 0원 — 풀 확장 단독 효과 분리)
#    예측: 67.22 → ≥68.
# log: /home/woori/scratch/tb_night14.log, sentinel NIGHT14_DONE (NA/NB/NC/ND 각 *_DONE)
R=/home/woori/workspace_common/boltzmann-attention-pi
TB=/home/woori/scratch/JARVIS_tb/taskbench
RUNS=$R/reports/facet_rft_2026/phase4_distill/sft_runs
VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
IP=/home/woori/scratch/tbeval_venv/bin/python
PY=/home/woori/venvs/seka_env/bin/python
S=/home/woori/scratch
TBPRED=$TB/data_multimedia_sub500/predictions
TRACKB=$R/reports/facet_rft_2026/trackb_raw/preds/data_multimedia_sub500
exec > $S/tb_night14.log 2>&1
set -x
cd $R && git pull --ff-only -q

kill_gpu() {
  for p in $(nvidia-smi --id=$1 --query-compute-apps=pid --format=csv,noheader); do
    kill -9 $p 2>/dev/null; done
  sleep 8
}

EXTRA=""
for f in qwen25_32b qwen25_32b_guided qwen25_72b qwen25_72b_guided qwen3_235b_a22b_int4_guided qwen3_32b; do
  [ -f $TRACKB/$f.json ] && EXTRA="$EXTRA --extra $f=$TRACKB/$f.json"
done

# ---------- ND (CPU 선행): 기존 풀 + Track-B 확장 ----------
(cd $R/scripts/distill/taskbench && $IP tb_select_official.py --tb_dir $TB \
  --prior_beta 2.0 $EXTRA --out $TBPRED/tb_sel1_b2_xpool.json)
$IP $R/scripts/distill/taskbench/tb_build_eval.py --tb_dir $TB --domain data_multimedia \
  --pred_file $TBPRED/tb_sel1_b2_xpool.json \
  --dst $TB/data_multimedia_sub500_eval_tb_sel1_b2_xpool --llm tb_sel1_b2_xpool \
  > $S/nd_xpool.txt 2>&1
grep -hE "link_binary_f1|node_micro_f1_no" $S/nd_xpool.txt | head -2
echo "ND_DONE"

# ---------- NB (GPU1): v3mix+guided K=8 샘플링 ----------
(
kill_gpu 1
VTAG=tb_v3g_night
$IP $R/scripts/distill/taskbench/tb_guided_patch.py $TB/inference.py || true
[ -f $S/tb_guided_mm_schema.json ] || $IP $R/scripts/distill/taskbench/tb_guided_schema.py \
  --tool_desc $TB/data_multimedia/tool_desc.json --dep resource --out $S/tb_guided_mm_schema.json
CUDA_VISIBLE_DEVICES=1 VLLM_PORT=8200 setsid nohup $VLLM serve Qwen/Qwen2.5-7B-Instruct \
  --port 8001 --served-model-name base_model --enable-lora \
  --lora-modules ${VTAG}=$RUNS/qwen7b_tb_dpo_v3mix \
  --max-model-len 8192 --gpu-memory-utilization 0.85 > $S/vllm_${VTAG}.log 2>&1 &
ok=0
for i in $(seq 1 90); do
  curl -s localhost:8001/v1/models | grep -q "\"$VTAG\"" && ok=1 && break; sleep 10
done
if [ $ok != 1 ]; then echo NB_SERVE_FAIL; exit 0; fi
for k in 0 1 2 3 4 5 6 7; do
  rm -f $TBPRED/$VTAG.json
  (cd $TB && TB_GUIDED=1 TB_GUIDED_SCHEMA=$S/tb_guided_mm_schema.json \
    $IP inference.py --data_dir data_multimedia_sub500 --api_addr localhost --api_port 8001 \
    --api_key dummy --llm $VTAG --multiworker 8 --dependency_type resource --temperature 0.8)
  mv $TBPRED/$VTAG.json $TBPRED/tb_v3g_mmk${k}.json
  echo "NB_K_DONE $k"
done
kill_gpu 1
echo "NB_DONE"

# ---------- NC (CPU): v3g-풀 선별 합성 ----------
$IP $R/scripts/distill/taskbench/tb_build_eval.py --tb_dir $TB --domain data_multimedia \
  --pred_file $TBPRED/tb_v3g_mmk0.json \
  --dst $TB/data_multimedia_sub500_eval_tb_v3g_k0 --llm tb_v3g_mmk0 > $S/nc_c0.txt 2>&1
(cd $R/scripts/distill/taskbench && $IP tb_select_official.py --tb_dir $TB \
  --ar_tag tb_v3g_mmk --ar_group v3g --prior_beta 2.0 \
  --out $TBPRED/tb_sel_v3g.json)
$IP $R/scripts/distill/taskbench/tb_build_eval.py --tb_dir $TB --domain data_multimedia \
  --pred_file $TBPRED/tb_sel_v3g.json \
  --dst $TB/data_multimedia_sub500_eval_tb_sel_v3g --llm tb_sel_v3g > $S/nc_c1.txt 2>&1
(cd $R/scripts/distill/taskbench && $IP tb_select_official.py --tb_dir $TB \
  --ar_tag tb_v3g_mmk --ar_group v3g --prior_beta 2.0 $EXTRA \
  --out $TBPRED/tb_sel_v3g_xpool.json)
$IP $R/scripts/distill/taskbench/tb_build_eval.py --tb_dir $TB --domain data_multimedia \
  --pred_file $TBPRED/tb_sel_v3g_xpool.json \
  --dst $TB/data_multimedia_sub500_eval_tb_sel_v3g_xpool --llm tb_sel_v3g_xpool > $S/nc_c2.txt 2>&1
grep -hE "link_binary_f1|node_micro_f1_no" $S/nc_c0.txt $S/nc_c1.txt $S/nc_c2.txt
echo "NC_DONE"
) &
NB_PID=$!

# ---------- NA (GPU0 + OpenRouter): τ² gate r3 ----------
source /home/woori/.openrouter_key
kill_gpu 0
CUDA_VISIBLE_DEVICES=0 VLLM_PORT=8100 setsid nohup $VLLM serve Qwen/Qwen2.5-7B-Instruct \
  --port 8351 --enable-auto-tool-choice --tool-call-parser hermes --max-model-len 16384 \
  > $S/vllm_t2_agent_r3.log 2>&1 &
ok=0
for i in $(seq 1 90); do
  curl -s localhost:8351/v1/models | grep -q Qwen && ok=1 && break; sleep 10
done
if [ $ok = 1 ]; then
  rm -rf /home/woori/scratch/tau2-bench/data/simulations/retail_7b_gate_r3
  cd /home/woori/scratch/tau2-bench
  export PYTHONPATH=src:$R/scripts/distill/tau2
  $PY $R/scripts/distill/tau2/t2_run_gated.py --gate 1 --num_trials 4 \
    --user_llm "openrouter/openai/gpt-4.1" --save_to retail_7b_gate_r3
  echo "NA_DONE"
else
  echo "NA_SERVE_FAIL"
fi
kill_gpu 0

wait $NB_PID || true
echo "NIGHT14_DONE $(date)"
