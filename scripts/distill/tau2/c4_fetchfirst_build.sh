#!/bin/bash
# C4-learn fetch-first arm (RULE_LEVER_COST_EFFICIENCY_PROGRAM·C4_LEARN_FETCHFIRST_CROSSOVER §3)
# = cfbsynth(추상 P2b/P4 fetch-first SHAPE·Synth stratum) + tbnfc replay(general·no-collapse) 1:1.
# ★격리: merged solo(synth+soptb+cfbsynth) 아님 = fetch-first 귀속 깨끗. tau2 학습 금지(grep0 게이트).
# Usage: c4_fetchfirst_build.sh <gpu> [n_cfb] [n_replay] [r] [seq]
set -u
GPU=${1:?gpu}; N_CFB=${2:-2000}; N_REPLAY=${3:-2000}; R=${4:-16}; SEQ=${5:-4096}
REPO=/home/woori/workspace_common/boltzmann-attention-pi
DIST=$REPO/scripts/distill; MA=$DIST/ma; CV=$DIST/taskbench
PY=/home/woori/venvs/seka_env/bin/python
S=/home/woori/scratch; FC=$S/fc_build; B=$S/c4_ff
TAG=qwen7b_c4_fetchfirst
mkdir -p $B
LOG=$S/c4_fetchfirst_build.log
exec > $LOG 2>&1; set -x; date
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 1. cfbsynth 추상 P2b/P4 생성 (per-traj 익명·randomized id·리스트선택)
$PY $MA/synth_fetch_nativefc.py --out $B/cfbsynth_native.jsonl --n $N_CFB --seed 0 --max_hops 3 --max_list 5
# 2. QC + 포맷 정규화 (--no_alias: 이미 per-traj 익명)
$PY $CV/fc_build_sft.py --inputs $B/cfbsynth_native.jsonl --out $B/cfbsynth_part.jsonl --no_alias --seed 42
# 3. replay subsample (general tool-use·TaskBench native-FC·tau2 무관)
cat $FC/tbnfc_huggingface_fc.jsonl $FC/tbnfc_multimedia_fc.jsonl | shuf --random-source=<(yes 42) | head -n $N_REPLAY > $B/replay_sub.jsonl
# 4. mix + shuffle
cat $B/cfbsynth_part.jsonl $B/replay_sub.jsonl | shuf --random-source=<(yes 7) > $B/c4_ff_train.jsonl
echo "=== MIX ==="; wc -l $B/cfbsynth_part.jsonl $B/replay_sub.jsonl $B/c4_ff_train.jsonl

# 5. ★tau2 학습 금지 게이트: retail/airline 도구명이 학습데이터에 있으면 ABORT ([[05]]·[[11]])
TAU2_TOOLS='get_user_details|get_order_details|cancel_pending_order|exchange_delivered_order_items|modify_pending_order|modify_user_address|return_delivered_order_items|find_user_id_by|book_reservation|cancel_reservation|get_reservation_details|update_reservation_|send_certificate|transfer_to_human_agents'
HIT=$(grep -cE "$TAU2_TOOLS" $B/c4_ff_train.jsonl || true)
echo "tau2-tool grep in train = $HIT (expect 0)"
[ "$HIT" != "0" ] && { echo "TAU2_CONTAMINATION_ABORT"; exit 2; }

# 6. small-rank LoRA + replay (진행률 가시: -u·log-every·save-every·ckpt-at)
for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 4
rm -rf $S/adapters/$TAG
CUDA_VISIBLE_DEVICES=$GPU $PY -u $DIST/lora_train_chat_toolcall.py \
  --base-model Qwen/Qwen2.5-7B-Instruct --train-jsonl $B/c4_ff_train.jsonl \
  --out-dir $S/adapters/$TAG --device cuda:0 \
  --epochs 1 --lr 5e-5 --lora-r $R --lora-alpha $((R*2)) --grad-accum 16 --max-seq-len $SEQ \
  --log-every 10 --save-every 50 --ckpt-at 50,100,200,400 --resume \
  --skip-overlong --attn flash_attention_2
[ -d $S/adapters/$TAG ] && echo C4_FETCHFIRST_DONE || echo TRAIN_FAIL
echo "=== snapshots ==="; ls -d $S/adapters/$TAG/step* 2>/dev/null | sort -V
date
