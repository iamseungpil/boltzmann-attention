#!/bin/bash
# 자율 파이프라인 (2026-06-14 야간, 사용자 "자동으로 전이테스트까지"):
#   qwen7b_fc_tbox_v1 학습완료 대기 → GPU0 해제 → 어댑터 serve(native FC·hermes)
#   → τ² retail 전이테스트(gate·40태스크·4trial, agent=fctbox) → compliant-pass 기록.
# 핵심: per-traj 랜덤 alias 학습이라 모델은 tools=컨텍스트서 이름을 *복사*(R1) → τ² 실도구명에 전이 기대.
# 비교축: base-7B compliant-pass^1=0.17 / frontier(gpt-4.1)=0.81 (PORTFOLIO §3.7f).
# log=/home/woori/scratch/fc_v1_pipeline.log, sentinel=FC_V1_TRANSFER_DONE
set -u
R=/home/woori/workspace_common/boltzmann-attention-pi
T2=$R/scripts/distill/tau2
PY=/home/woori/venvs/seka_env/bin/python
VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
S=/home/woori/scratch
O=$S/sft_runs/qwen7b_fc_tbox_v1
exec > $S/fc_v1_pipeline.log 2>&1
set -x
date

kill_gpu0() { for p in $(nvidia-smi --id=0 --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 8; }

# 1) 학습 완료 대기 (프로세스 종료 + 어댑터 저장), 최대 8h
for i in $(seq 1 480); do
  pgrep -f "[l]ora_train_chat_toolcall.py.*fc_tbox_v1" >/dev/null || break
  sleep 60
done
if [ ! -f $O/adapter_config.json ]; then echo "FC_V1_TRAIN_FAIL: no adapter_config"; tail -20 $O/train.log; exit 1; fi
echo "=== TRAIN DONE ==="; ls -la $O | tail -5; tail -6 $O/train.log

# 2) GPU0 해제
kill_gpu0

# 3) 어댑터 serve (native FC, multi-LoRA)
source /home/woori/.openrouter_key
export SSL_CERT_FILE=$($PY -c "import certifi;print(certifi.where())")
CUDA_VISIBLE_DEVICES=0 setsid nohup $VLLM serve Qwen/Qwen2.5-7B-Instruct \
  --port 8351 --enable-auto-tool-choice --tool-call-parser hermes --max-model-len 16384 \
  --enable-lora --lora-modules fctbox=$O --max-lora-rank 32 \
  > $S/vllm_fc_v1.log 2>&1 &
ok=0; for i in $(seq 1 90); do curl -s localhost:8351/v1/models 2>/dev/null | grep -q fctbox && ok=1 && break; sleep 10; done
if [ $ok != 1 ]; then echo "FC_V1_SERVE_FAIL"; tail -25 $S/vllm_fc_v1.log; exit 1; fi
echo "=== SERVE OK ==="

# 4) τ² retail 전이테스트 (gate ON, agent=fctbox)
cd $S/tau2-bench
export PYTHONPATH=src:$T2
rm -rf data/simulations/retail_fctbox_v1_gate
$PY $T2/t2_run_gated.py --gate 1 --num_trials 4 --num_tasks 40 \
  --agent_model fctbox --agent_base http://localhost:8351/v1 \
  --user_llm "openrouter/openai/gpt-4.1" --user_temp 0.0 \
  --save_to retail_fctbox_v1_gate || echo "TAU2_RUN_FAIL"
cd $R
kill_gpu0

# 5) 결과 보고 (compliance.json = pass^k + 위반, t2_run_gated 자동산출)
echo "=== TAU2 TRANSFER RESULT (fctbox v1) ==="
cat $S/tau2-bench/data/simulations/retail_fctbox_v1_gate/compliance.json 2>/dev/null || echo "no compliance.json"
echo "(비교: base-7B pass^1=0.17 / frontier=0.81)"
date
echo "FC_V1_TRANSFER_DONE"
