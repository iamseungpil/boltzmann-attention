#!/bin/bash
# ★야간 배치 2026-06-12→13 — day-4 후속 3실험 (사전등록 동결, 딥리서치 비의존)
# N1 v3mix+guided MM full: 통제 v3 raw 56.46·dpo2+guided 57.22 — 예측: ≥58 (guided 가산 +1~1.5)
# N2 선별기 공식-척도: filter+propMBR 선별(AR8+H6) vs k0 단일 — 예측: 공식 link F1 ≥ +2
#    (census 0.671→0.753 = +8.2의 공식-척도 보수 하한; 미달=census-공식 괴리 1급 기록)
# N3 τ² G1 deny-복구 메시지 주입 gate arm 재실행: 통제 run7 gate pass^1 0.147·deny→fail 92%
#    — 예측: deny→fail <50% ∧ pass^1 ≥0.184(게이트 비용 상쇄) ∧ write-차단 ~98% 유지
# log: /home/woori/scratch/tb_night13.log, sentinel NIGHT13_DONE
R=/home/woori/workspace_common/boltzmann-attention-pi
TB=/home/woori/scratch/JARVIS_tb/taskbench
RUNS=$R/reports/facet_rft_2026/phase4_distill/sft_runs
VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
IP=/home/woori/scratch/tbeval_venv/bin/python
PY=/home/woori/venvs/seka_env/bin/python
S=/home/woori/scratch
exec > $S/tb_night13.log 2>&1
set -x
cd $R && git pull --ff-only -q

# ---------- N2 (CPU, 선행 — GPU 무관) ----------
(cd $R/scripts/distill/taskbench && $IP tb_select_official.py --tb_dir $TB \
  --out $TB/data_multimedia_sub500/predictions/tb_sel_fpmbr.json)
$IP $R/scripts/distill/taskbench/tb_build_eval.py --tb_dir $TB --domain data_multimedia \
  --pred_file $TB/data_multimedia_sub500/predictions/tb_sel_fpmbr.json \
  --dst $TB/data_multimedia_sub500_eval_tb_sel_fpmbr --llm tb_sel_fpmbr > $S/n2_selected.txt 2>&1
$IP $R/scripts/distill/taskbench/tb_build_eval.py --tb_dir $TB --domain data_multimedia \
  --pred_file $TB/data_multimedia_sub500/predictions/tb_dpo2g_mmk0.json \
  --dst $TB/data_multimedia_sub500_eval_tb_dpo2g_k0ctl --llm tb_dpo2g_mmk0 > $S/n2_control.txt 2>&1
grep -hE "link_binary_f1|node_micro_f1_no" $S/n2_selected.txt $S/n2_control.txt | head -4

# ---------- N1 (GPU0): v3mix + guided, MM full ----------
TAG=tb_dpo_v3mix_guided
for p in $(nvidia-smi --id=0 --query-compute-apps=pid --format=csv,noheader); do
  kill -9 $p 2>/dev/null; done
sleep 10
$IP $R/scripts/distill/taskbench/tb_guided_patch.py $TB/inference.py || exit 1
$IP $R/scripts/distill/taskbench/tb_guided_schema.py \
  --tool_desc $TB/data_multimedia/tool_desc.json --dep resource \
  --out $S/tb_guided_mm_schema.json || exit 1
CUDA_VISIBLE_DEVICES=0 setsid nohup $VLLM serve Qwen/Qwen2.5-7B-Instruct \
  --port 8000 --served-model-name base_model --enable-lora \
  --lora-modules ${TAG}=$RUNS/qwen7b_tb_dpo_v3mix \
  --max-model-len 8192 --gpu-memory-utilization 0.85 \
  > $S/vllm_${TAG}.log 2>&1 &
ok=0
for i in $(seq 1 90); do
  curl -s localhost:8000/v1/models | grep -q "\"$TAG\"" && ok=1 && break
  sleep 10
done
[ $ok = 1 ] || { echo SERVE_FAIL_$TAG; exit 1; }
(cd $TB && TB_GUIDED=1 TB_GUIDED_SCHEMA=$S/tb_guided_mm_schema.json \
  $IP inference.py --data_dir data_multimedia --api_addr localhost --api_port 8000 \
  --api_key dummy --llm $TAG --multiworker 8 --dependency_type resource)
$IP $R/scripts/distill/taskbench/tb_build_eval.py --tb_dir $TB --domain data_multimedia \
  --llm $TAG --dst $TB/data_multimedia_evalfull_${TAG} > $S/n1_v3guided.txt 2>&1
grep -hE "link_binary_f1|node_micro_f1_no" $S/n1_v3guided.txt | head -2
for p in $(nvidia-smi --id=0 --query-compute-apps=pid --format=csv,noheader); do
  kill -9 $p 2>/dev/null; done
sleep 5

# ---------- N3 (GPU0 + OpenRouter): τ² gate arm 재실행 (deny-복구 메시지) ----------
source /home/woori/.openrouter_key
CUDA_VISIBLE_DEVICES=0 setsid nohup $VLLM serve Qwen/Qwen2.5-7B-Instruct --port 8351 \
  --enable-auto-tool-choice --tool-call-parser hermes --max-model-len 16384 \
  > $S/vllm_t2_agent.log 2>&1 &
ok=0
for i in $(seq 1 90); do
  curl -s localhost:8351/v1/models | grep -q Qwen && ok=1 && break
  sleep 10
done
[ $ok = 1 ] || { echo SERVE_FAIL_8351; echo "NIGHT13_DONE $(date)"; exit 1; }
cd /home/woori/scratch/tau2-bench
export PYTHONPATH=src:$R/scripts/distill/tau2
$PY $R/scripts/distill/tau2/t2_run_gated.py --gate 1 --num_trials 4 \
  --user_llm "openrouter/openai/gpt-4.1" --save_to retail_7b_gate_r2
$PY $R/scripts/distill/tau2/t2_passk_census.py --simdir /home/woori/scratch/tau2-bench/data/simulations || true

echo "NIGHT13_DONE $(date)"
