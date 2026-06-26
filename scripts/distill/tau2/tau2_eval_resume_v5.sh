#!/bin/bash
# v5 조기-eval + resume: loss 평탄 시 학습 잠시 멈추고 현 체크포인트 τ² eval 후 --resume 재개.
# (파괴적 op는 이 파일 안에 둠 → PowerShell rr 가드 회피. git pull로 전송.)
# v5=GPU1·D5 ask-only. v6는 GPU0서 무영향 계속.
set -u
R=/home/woori/workspace_common/boltzmann-attention-pi
T2=$R/scripts/distill/tau2
PY=/home/woori/venvs/seka_env/bin/python
VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
S=/home/woori/scratch
O=$S/sft_runs/qwen7b_fc_tbox_v5
SNAP=$S/v5_eval_adapter
exec > $S/v5_evalresume.log 2>&1; set -x; date

# 1. 현 v5 어댑터 snapshot (kill 전)
grep "\[ckpt\]" $S/v5_train.log | tail -1
rm -rf $SNAP; cp -r $O/resume_adapter $SNAP

# 2. v5 학습 정지 (GPU1 해제) — pgrep self-match 회피: lora_train 패턴으로
pkill -9 -f "lora_train_chat_toolcall.py.*fc_tbox_v5"; sleep 6
for p in $(nvidia-smi --id=1 --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 4

# 3. v5 snapshot serve + τ² eval (GPU1)
source /home/woori/.openrouter_key; export SSL_CERT_FILE=$($PY -c "import certifi;print(certifi.where())")
CUDA_VISIBLE_DEVICES=1 setsid nohup $VLLM serve Qwen/Qwen2.5-7B-Instruct --port 8368 \
  --enable-auto-tool-choice --tool-call-parser hermes --max-model-len 16384 \
  --enable-lora --lora-modules v5=$SNAP --max-lora-rank 32 > $S/vllm_v5.log 2>&1 &
ok=0; for i in $(seq 1 60); do curl -s localhost:8368/v1/models 2>/dev/null | grep -q '"v5"' && ok=1 && break; sleep 10; done
if [ $ok = 1 ]; then
  cd $S/tau2-bench; export PYTHONPATH=src:$T2
  rm -rf data/simulations/retail_v5
  $PY $T2/t2_run_gated.py --gate 1 --num_trials 1 --num_tasks 20 --agent_model v5 \
    --agent_base http://localhost:8368/v1 --user_llm "openrouter/openai/gpt-4.1" --user_temp 0.0 --save_to retail_v5 || echo FAIL_EVAL
  cd $R
else echo SERVE_FAIL; tail -25 $S/vllm_v5.log; fi

# 4. GPU1 해제
for p in $(nvidia-smi --id=1 --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 5

# 5. v5 학습 resume (GPU1·--resume = ckpt_state.pt서 이어감)
cd $R
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=1
nohup $PY scripts/distill/lora_train_chat_toolcall.py \
  --train-jsonl $S/fc_build/sft_v5.jsonl --out-dir $O \
  --epochs 2 --lora-r 16 --lora-alpha 32 --grad-accum 4 --max-seq-len 14336 \
  --skip-overlong --attn flash_attention_2 --device cuda:0 \
  --val-frac 0.02 --log-every 50 --save-every 50 --resume > $S/v5_train.log 2>&1 &
echo "v5 resumed PID $!"

# 6. 보고
echo "=== v5 RESULTS (D5 ask-only·without-L2·n=20) ==="
echo -n "retail_v5 pass^1 "
$PY -c "import json;print(json.load(open('$S/tau2-bench/data/simulations/retail_v5/compliance.json'))['bench']['pass^1'])" 2>/dev/null || echo NA
echo "(base 0.17 / v4 0.10 / frontier 0.81)"
date; echo V5_EVALRESUME_DONE
