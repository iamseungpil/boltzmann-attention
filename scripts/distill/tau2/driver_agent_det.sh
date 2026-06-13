#!/bin/bash
# ⓟ1 agent 격리 결정론 테스트 (2026-06-14 사용자): det serve 설정 그대로 7B 띄우고
# 고정 context N=40회 반복 생성 → 동일성. user-sim 완전 배제 = agent vLLM 단독.
# Run: setsid bash driver_agent_det.sh </dev/null >/dev/null 2>&1 &
set -u
R=/home/woori/workspace_common/boltzmann-attention-pi
VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
IP=/home/woori/scratch/tbeval_venv/bin/python
S=/home/woori/scratch
exec > $S/agent_det.log 2>&1
set -x
cd $R && git pull --ff-only -q
for p in $(nvidia-smi --id=0 --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done
sleep 5
# det 런과 동일 serve 설정
CUDA_VISIBLE_DEVICES=0 setsid nohup $VLLM serve Qwen/Qwen2.5-7B-Instruct \
  --port 8351 --enable-auto-tool-choice --tool-call-parser hermes --max-model-len 16384 \
  --enforce-eager --max-num-seqs 1 --seed 0 > $S/vllm_agentdet.log 2>&1 &
ok=0
for i in $(seq 1 90); do curl -s localhost:8351/v1/models | grep -q Qwen && ok=1 && break; sleep 10; done
[ $ok = 1 ] || { echo SERVE_FAIL; exit 1; }

$IP $R/scripts/distill/tau2/t2_agent_determinism.py \
  --endpoint http://localhost:8351/v1 \
  --simdir /home/woori/scratch/tau2-bench/data/simulations/retail_7b_gate_det \
  --n 40 --nctx 10

for p in $(nvidia-smi --id=0 --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done
echo "AGENT_DET_DONE $(date)"
