#!/bin/bash
# RFT round-1 chain (RAFT): rollout from served SFT policy -> warm-start train on winners
# -> full adapter eval. vLLM with the source adapter must already be serving on PORT.
# Usage: tb_rft_round.sh rft_mm 0 lodo_mm data_multimedia "data_huggingface data_dailylifeapis"
NAME=${1:-rft_mm}; GPU=${2:-0}; SRC=${3:-lodo_mm}
HOLDOUT=${4:-data_multimedia}; TRAIN_DOMS=${5:-"data_huggingface data_dailylifeapis"}
PORT=$((8000+GPU))
# env overrides: PROMPTS (rollout prompt jsonl), RWARGS (reward weights etc.)
PROMPTS=${PROMPTS:-/home/woori/scratch/tb_sft/train_lodo_mm.jsonl}
RWARGS=${RWARGS:-}
R=/home/woori/workspace_common/boltzmann-attention-pi
RUNS=$R/reports/facet_rft_2026/phase4_distill/sft_runs
EP=/home/woori/scratch/tbeval_venv/bin/python
PY=/home/woori/venvs/seka_env/bin/python
RFT=/home/woori/scratch/tb_rft
LOG=$RFT/round_${NAME}.log
mkdir -p $RFT
exec > $LOG 2>&1
set -x

# 1) rollout (K=8; reward weights via RWARGS, default = round-1 0.3node+0.7edge)
$EP $R/scripts/distill/taskbench/tb_rft_rollout.py \
  --sft_jsonl $PROMPTS \
  --api http://localhost:$PORT/v1/chat/completions --model tb_${SRC} \
  --k 8 --temp 1.0 --min_reward 0.8 --concurrency 16 $RWARGS \
  --out $RFT/winners_${NAME}.jsonl || exit 1
wc -l $RFT/winners_${NAME}.jsonl

# 2) free the GPU (kill vllm only), then RAFT warm-start train
for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid,process_name --format=csv,noheader | grep -i vllm | cut -d, -f1); do kill -9 $p; done
sleep 15
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY \
  $R/scripts/distill/lora_train_chat_toolcall.py \
  --base-model Qwen/Qwen2.5-7B-Instruct --device cuda:0 --max-seq-len 6144 \
  --epochs 2 --lr 5e-5 --lora-r 16 --val-frac 0.02 --skip-overlong \
  --init-adapter $RUNS/qwen7b_tb_${SRC} \
  --train-jsonl $RFT/winners_${NAME}.jsonl --out-dir $RUNS/qwen7b_tb_${NAME}
echo "RFT_TRAIN_DONE_${NAME} $(date)"

# 3) eval (serves fresh vllm with the new adapter on this GPU)
bash $R/scripts/distill/taskbench/tb_eval_adapter.sh $NAME $HOLDOUT "$TRAIN_DOMS" $GPU 0.85
echo "RFT_ROUND_DONE_${NAME} $(date)"
