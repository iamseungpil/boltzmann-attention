#!/bin/bash
# 결정론 재현 실험 (ⓟ1 재개 선결 — 2026-06-13 사용자 발주, PLORA_PD0_DONE 게이트 후 GPU0)
# 가설: temp0인데 4-trial 0/111 동일했던 원인 = 배칭 비결정성(continuous batching+CUDA graph).
# 처방 시험: serve에 --enforce-eager(CUDA graph off) + --max-num-seqs 1(배칭 제거) + seed,
#   client max_concurrency=1(순차) → 같은 입력 결정론화 시도.
# 사전등록(SELECTOR/PORTFOLIO §3.7d 정정):
#   ⓓ1 det-arm 4-trial 인자포함-seq 동일률 0%→ **>=70%** = 배칭이 비결정 주범 확정
#        ∧ ⓟ1 재측정 경로 개방 (gate det vs nogate det로 Δpass^4 재판정 가능)
#   ⓓ2 여전히 <70% = vLLM enforce-eager로도 잔존 비결정 → batch-invariant 커널 필요(차기)
# 비용 절감: num_tasks 40(앞 40, 4 trials) — 동일성 측정엔 충분. gate ON(r3 구성).
# log: /home/woori/scratch/determinism.log, sentinel DET_DONE
set -u
R=/home/woori/workspace_common/boltzmann-attention-pi
T2=$R/scripts/distill/tau2
PY=/home/woori/venvs/seka_env/bin/python
VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
S=/home/woori/scratch
exec > $S/determinism.log 2>&1
set -x
cd $R && git pull --ff-only -q

# 게이트: 앞선 배치 종료 대기 (최대 6h)
for i in $(seq 1 360); do
  grep -q "PLORA_PD0_DONE" $S/plora_pd0.log 2>/dev/null && break
  sleep 60
done
for i in $(seq 1 60); do
  nvidia-smi --id=0 --query-compute-apps=pid --format=csv,noheader | grep -q . || break
  sleep 30
done

kill_gpu0() { for p in $(nvidia-smi --id=0 --query-compute-apps=pid --format=csv,noheader); do
  kill -9 $p 2>/dev/null; done; sleep 8; }

source /home/woori/.openrouter_key
kill_gpu0
# ★결정론 serve: enforce-eager(CUDA graph off) + max-num-seqs 1(배칭 제거) + seed 0
CUDA_VISIBLE_DEVICES=0 VLLM_PORT=8140 setsid nohup $VLLM serve Qwen/Qwen2.5-7B-Instruct \
  --port 8351 --enable-auto-tool-choice --tool-call-parser hermes --max-model-len 16384 \
  --enforce-eager --max-num-seqs 1 --seed 0 \
  > $S/vllm_det.log 2>&1 &
ok=0
for i in $(seq 1 90); do
  curl -s localhost:8351/v1/models | grep -q Qwen && ok=1 && break; sleep 10
done
if [ $ok != 1 ]; then echo DET_SERVE_FAIL; exit 1; fi

cd /home/woori/scratch/tau2-bench
export PYTHONPATH=src:$R/scripts/distill/tau2
# gate ON, agent seed 고정, user-sim temp0(분산원 전부 통제), 순차(concurrency 1)
rm -rf data/simulations/retail_7b_gate_det
$PY $R/scripts/distill/tau2/t2_run_gated.py --gate 1 --num_trials 4 --num_tasks 40 \
  --max_concurrency 1 --agent_seed 0 \
  --user_llm "openrouter/openai/gpt-4.1" --user_temp 0.0 \
  --save_to retail_7b_gate_det || echo DET_RUN_FAIL
cd $R
kill_gpu0

# 동일성 재측정 (det arm 단독 — 0/40 였던 게 올라가나)
$PY $T2/t2_p1_autopsy.py --simdir /home/woori/scratch/tau2-bench/data/simulations \
  --arms retail_7b_gate_det 2>&1 | tee $S/det_autopsy.txt
{ echo "== [결정론 실험] det-arm 4-trial 동일성"; cat $S/det_autopsy.txt; } >> $S/day13_summary.txt
echo "DET_DONE $(date)" | tee -a $S/day13_summary.txt
