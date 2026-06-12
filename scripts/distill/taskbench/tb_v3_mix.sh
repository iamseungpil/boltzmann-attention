#!/bin/bash
# 혼합쌍 v3: 균형(714) + 구조(1017) = 1731쌍 DPO on rft2 (TB §8.10 D1 채택 분기).
# 사전등록 예측: MM full ≥ max(dpo2 55.95, dpo_struct 53.5) — 두 축 직교라 합집합 기대.
#   감시: D2-부검 체크리스트 (short/deficit/R 비악화 = 길이-prior 부재 확인).
# log: /home/woori/scratch/tb_v3mix.log, sentinel V3MIX_DONE
R=/home/woori/workspace_common/boltzmann-attention-pi
TB=/home/woori/scratch/JARVIS_tb/taskbench
RUNS=$R/reports/facet_rft_2026/phase4_distill/sft_runs
IP=/home/woori/scratch/tbeval_venv/bin/python
PY=/home/woori/venvs/seka_env/bin/python
SC=$R/scripts/distill/taskbench
S=/home/woori/scratch
exec > $S/tb_v3mix.log 2>&1
set -x
cd $R && git pull --ff-only -q

# 쌍 합본 (seed-42 셔플, 결정론)
$PY - <<'EOF'
import json, random
a = [l for l in open('/home/woori/scratch/tb_rft/dpo_balance.jsonl')]
b = [l for l in open('/home/woori/scratch/tb_rft/dpo_structure.jsonl')]
rows = a + b
random.seed(42); random.shuffle(rows)
open('/home/woori/scratch/tb_rft/dpo_v3mix.jsonl', 'w').writelines(rows)
print(f'[v3mix] {len(a)}+{len(b)}={len(rows)} pairs')
EOF

# 사전-kill (gotcha: 잔여 vllm/EngineCore)
for g in 0 1; do
  for p in $(nvidia-smi --id=$g --query-compute-apps=pid --format=csv,noheader); do
    kill -9 $p 2>/dev/null; done; done
sleep 10

CUDA_VISIBLE_DEVICES=1 $PY $R/scripts/distill/sopbench/dpo_train.py \
  --base Qwen/Qwen2.5-7B-Instruct --sft-adapter $RUNS/qwen7b_tb_rft2_mm \
  --pairs $S/tb_rft/dpo_v3mix.jsonl --out-dir $RUNS/qwen7b_tb_dpo_v3mix \
  --max-seq-len 6144 --epochs 2 --beta 0.1 --lr 5e-6 \
  > $S/tb_train_qwen7b_tb_dpo_v3mix.log 2>&1
grep -q "\[done\]" $S/tb_train_qwen7b_tb_dpo_v3mix.log || { echo V3MIX_TRAIN_FAIL; exit 1; }

# ★저장↔serve 레이스 가드 (2026-06-12 사고 재발 방지): safetensors 크기 안정 확인
for i in $(seq 1 30); do
  s1=$(stat -c%s $RUNS/qwen7b_tb_dpo_v3mix/adapter_model.safetensors 2>/dev/null || echo 0)
  sleep 5
  s2=$(stat -c%s $RUNS/qwen7b_tb_dpo_v3mix/adapter_model.safetensors 2>/dev/null || echo 0)
  [ "$s1" = "$s2" ] && [ "$s1" != "0" ] && break
done

bash $SC/tb_eval_adapter.sh dpo_v3mix data_multimedia "data_huggingface data_dailylifeapis" 0 0.85

# 판정 배터리
$IP $SC/tb_pr_census.py \
  --set $TB/data_multimedia_evalfull_tb_rft2_mm:tb_rft2_mm \
  --set $TB/data_multimedia_evalfull_tb_dpo2_mm:tb_dpo2_mm \
  --set $TB/data_multimedia_evalfull_tb_dpo_v3mix:tb_dpo_v3mix > $S/v3mix_pr.txt
$IP $SC/tb_census.py --dir_a $TB/data_multimedia_evalfull_tb_dpo2_mm --llm_a tb_dpo2_mm \
  --dir_b $TB/data_multimedia_evalfull_tb_dpo_v3mix --llm_b tb_dpo_v3mix \
  --tool_desc $TB/data_multimedia/tool_desc.json --dep resource \
  --out $S/census_dpo2_to_v3mix.md
cd $SC && $IP tb_d2_autopsy.py --dir_a $TB/data_multimedia_evalfull_tb_rft2_mm --llm_a tb_rft2_mm \
  --dir_b $TB/data_multimedia_evalfull_tb_dpo_v3mix --llm_b tb_dpo_v3mix \
  --tool_desc $TB/data_multimedia/tool_desc.json --pairs $S/tb_rft/dpo_v3mix.jsonl \
  --out $S/v3mix_autopsy.md

echo "V3MIX_DONE $(date)"
