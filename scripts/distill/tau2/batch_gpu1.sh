#!/bin/bash
# 외출 배치 — GPU1 eval/confirm 큐 (연속). 각 단계 = serve GPU1 + 실행 + kill GPU1(자식이 처리).
# 1-3: full-40 confirm(pass·ε-unmask grounding ok/fail·FAB) / 4: cfb_mid 학습 끝나면 FAB 곡선.
set -u
G=1; PORT=8371
R=/home/woori/workspace_common/boltzmann-attention-pi
T2=$R/scripts/distill/tau2
S=/home/woori/scratch; A=$S/adapters
LOG=$S/batch_gpu1.log
exec > $LOG 2>&1; set -x; date
echo "=== GPU1 BATCH START ==="

conf(){ # $1=adapterdir $2=tag
  bash $T2/real_e2e_solo.sh $G $PORT 40 2 "$1" "$2"
  set +x; echo "### RESULT $2:"; grep -E "\[$2\] n=" $S/real_e2e_solo_$2.log 2>/dev/null | tail -1; date; set -x
}

# 1. cfb 전층 step10 (현 best cfb checkpoint) — 결정적 cfb 확정 + ε-unmask
conf $A/qwen7b_solo_cfb/step10 cfb_s10
# 2. baseline lite step10 — A/B 기준선
conf $A/qwen7b_solo_lite/step10 lite_s10
# 3. cons step5
conf $A/qwen7b_solo_cons/step5 cons_s5
# 4. cfb_mid 학습 완료 대기 → FAB 곡선
for i in $(seq 1 300); do [ -f $A/qwen7b_solo_cfb_mid/train_meta.json ] && break; sleep 30; done
bash $T2/eval_solo_ckpts.sh $G $PORT 12 $A/qwen7b_solo_cfb_mid
set +x; echo "### cfb_mid FAB curve:"; grep -E "CKPT=|pass1=|FAB|sims_fab" $S/eval_solo_ckpts_qwen7b_solo_cfb_mid.log 2>/dev/null
echo "=== BATCH_GPU1_DONE ==="; date
