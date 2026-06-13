#!/bin/bash
# B1: self-certainty 선별 (SELECTOR §7 후속 — C 진단 selectable~50% 검증). GPU1.
# 사전등록: combined(MBR+selfcert) link F1 vs SEL-1 0.6722 / SEL-4 0.6803. selectable 갭을
#   forward-confidence가 구제하면 > SEL-4. pure-selfcert는 신호 자체 유효성(>k0 0.577?) 진단.
# Run: setsid bash driver_selfcert.sh </dev/null >/dev/null 2>&1 &
set -u
R=/home/woori/workspace_common/boltzmann-attention-pi
TB=/home/woori/scratch/JARVIS_tb/taskbench
TBPRED=$TB/data_multimedia_sub500/predictions
IP=/home/woori/scratch/tbeval_venv/bin/python
VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
S=/home/woori/scratch
exec > $S/selfcert_dpo2g.log 2>&1
set -x
cd $R && git pull --ff-only -q

for p in $(nvidia-smi --id=1 --query-compute-apps=pid --format=csv,noheader); do
  kill -9 $p 2>/dev/null; done
sleep 8
CUDA_VISIBLE_DEVICES=1 setsid nohup $VLLM serve Qwen/Qwen2.5-7B-Instruct \
  --port 8001 --served-model-name base_model --max-model-len 8192 \
  --gpu-memory-utilization 0.85 > $S/vllm_selfcert.log 2>&1 &
ok=0
for i in $(seq 1 90); do
  curl -s localhost:8001/v1/models | grep -q base_model && ok=1 && break; sleep 10
done
[ $ok = 1 ] || { echo SELFCERT_SERVE_FAIL; exit 1; }

$IP $R/scripts/distill/taskbench/tb_selfcert_select.py --tb_dir $TB \
  --ar_tag tb_dpo2g_mmk --ar_group dpo2g \
  --endpoint http://localhost:8001/v1 --served base_model --lam 1.0 \
  --out $TBPRED/tb_selfcert_dpo2g.json --out_pure $TBPRED/tb_selfcertpure_dpo2g.json

for tag in tb_selfcert tb_selfcertpure; do
  $IP $R/scripts/distill/taskbench/tb_build_eval.py --tb_dir $TB --domain data_multimedia \
    --pred_file $TBPRED/${tag}_dpo2g.json \
    --dst $TB/data_multimedia_sub500_eval_${tag} --llm ${tag} > $S/${tag}_eval.txt 2>&1
  echo "=== $tag ==="
  grep -hE "link_binary_f1|node_micro_f1_no" $S/${tag}_eval.txt
done

for p in $(nvidia-smi --id=1 --query-compute-apps=pid --format=csv,noheader); do
  kill -9 $p 2>/dev/null; done
echo "SELFCERT_DONE $(date)"
