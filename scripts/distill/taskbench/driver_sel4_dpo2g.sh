#!/bin/bash
# SEL-4를 최적 풀(dpo2g-AR8+H6, 67.22)에 적용 (TB §8.9d "다음 1수" — 사전등록:
# 공식 link F1 > 0.6722 [SEL-1 동풀 최고] — Reviewer 직교 신호 +0.73pp가 풀 불변로 재현되는가).
# Run: setsid bash driver_sel4_dpo2g.sh </dev/null >/dev/null 2>&1 &
set -u
R=/home/woori/workspace_common/boltzmann-attention-pi
TB=/home/woori/scratch/JARVIS_tb/taskbench
TBPRED=$TB/data_multimedia_sub500/predictions
IP=/home/woori/scratch/tbeval_venv/bin/python
VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
S=/home/woori/scratch
exec > $S/sel4_dpo2g.log 2>&1
set -x
cd $R && git pull --ff-only -q

for p in $(nvidia-smi --id=1 --query-compute-apps=pid --format=csv,noheader); do
  kill -9 $p 2>/dev/null; done
sleep 8
CUDA_VISIBLE_DEVICES=1 VLLM_PORT=8202 setsid nohup $VLLM serve Qwen/Qwen2.5-7B-Instruct \
  --port 8001 --served-model-name base_model --max-model-len 8192 \
  --gpu-memory-utilization 0.85 > $S/vllm_sel4b.log 2>&1 &
ok=0
for i in $(seq 1 90); do
  curl -s localhost:8001/v1/models | grep -q base_model && ok=1 && break; sleep 10
done
[ $ok = 1 ] || { echo SEL4B_SERVE_FAIL; exit 1; }

$IP $R/scripts/distill/taskbench/tb_reviewer_select.py --tb_dir $TB \
  --ar_tag tb_dpo2g_mmk --ar_group dpo2g \
  --endpoint http://localhost:8001/v1 --served base_model --lam 1.0 \
  --out $TBPRED/tb_sel4_dpo2g.json
$IP $R/scripts/distill/taskbench/tb_build_eval.py --tb_dir $TB --domain data_multimedia \
  --pred_file $TBPRED/tb_sel4_dpo2g.json \
  --dst $TB/data_multimedia_sub500_eval_tb_sel4_dpo2g --llm tb_sel4_dpo2g \
  > $S/sel4_dpo2g_eval.txt 2>&1
grep -hE "link_binary_f1|node_micro_f1_no" $S/sel4_dpo2g_eval.txt

for p in $(nvidia-smi --id=1 --query-compute-apps=pid --format=csv,noheader); do
  kill -9 $p 2>/dev/null; done
echo "SEL4_DPO2G_DONE $(date)"
