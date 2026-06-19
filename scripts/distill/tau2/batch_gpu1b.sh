#!/bin/bash
# 외출 배치 GPU1 꼬리 — batch_gpu1 완료 후 cfb_mid 최선후보(step10·step40) full-40 확정(S0 mid arm 연장).
set -u
G=1; PORT=8372
R=/home/woori/workspace_common/boltzmann-attention-pi
T2=$R/scripts/distill/tau2
S=/home/woori/scratch; A=$S/adapters
LOG=$S/batch_gpu1b.log
exec > $LOG 2>&1; set -x; date

# batch_gpu1 끝 + cfb_mid 학습 끝 대기
for i in $(seq 1 600); do
  grep -q BATCH_GPU1_DONE $S/batch_gpu1.log 2>/dev/null && [ -f $A/qwen7b_solo_cfb_mid/train_meta.json ] && break
  sleep 30
done
echo "=== GPU1b START (cfb_mid full-40) ==="; date

conf(){ bash $T2/real_e2e_solo.sh $G $PORT 40 2 "$1" "$2"; set +x; echo "### RESULT $2:"; grep -E "\[$2\] n=" $S/real_e2e_solo_$2.log 2>/dev/null | tail -1; date; set -x; }
[ -d $A/qwen7b_solo_cfb_mid/step10 ] && conf $A/qwen7b_solo_cfb_mid/step10 cfbmid_s10
[ -d $A/qwen7b_solo_cfb_mid/step40 ] && conf $A/qwen7b_solo_cfb_mid/step40 cfbmid_s40
echo "=== BATCH_GPU1B_DONE ==="; date
