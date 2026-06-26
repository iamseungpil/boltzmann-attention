#!/bin/bash
# 외출 배치 — GPU0 학습 큐. cfb_mid(r16·이미 진행중) 완료 대기 → rank 스윕(r8·r32) 연속.
# install-rank=내재차원 프록시(CLOSURE_PAYOFF §61): 더 낮은 rank서도 P2b 설치되나(저차원 증거)+forgetting↓?
set -u
G=0
R=/home/woori/workspace_common/boltzmann-attention-pi
T2=$R/scripts/distill/tau2
S=/home/woori/scratch; A=$S/adapters
LOG=$S/batch_gpu0.log
exec > $LOG 2>&1; set -x; date
echo "=== GPU0 BATCH START (cfb_mid r16 진행중·완료 대기) ==="

# 1. cfb_mid(r16) 완료 대기 (train_meta.json = epoch 끝)
for i in $(seq 1 500); do [ -f $A/qwen7b_solo_cfb_mid/train_meta.json ] && break; sleep 30; done
echo "=== cfb_mid r16 done -> rank sweep r8 ==="; date
# 2. cfb_mid r8 (mid-layer·cfbsynth·r8)
bash $T2/build_solo_train_cfb.sh 0 8-19 5e-5 8 16
echo "=== r8 done -> r32 ==="; date
# 3. cfb_mid r32
bash $T2/build_solo_train_cfb.sh 0 8-19 5e-5 32 64
echo "=== BATCH_GPU0_DONE ==="; date
