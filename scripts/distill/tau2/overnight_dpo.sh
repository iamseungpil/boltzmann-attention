#!/bin/bash
# DPO 체인 (무인): build(pairs+fresh LoRA+DPO) → retail eval → airline 전이 eval → 요약.
# GPU0 단독 순차. 핵심 판정: DPO penalty가 schema_copy → ~0 인가(SFT 52 vs).
set -u
GPU=${1:-0}
REPO=/home/woori/workspace_common/boltzmann-attention-pi; T2=$REPO/scripts/distill/tau2
S=/home/woori/scratch
LOG=$S/overnight_dpo.log; exec > $LOG 2>&1; set -x; date

# 1. DPO 학습 (build)
bash $T2/c4_dpo_build.sh $GPU 2000 0.1 2 5e-6
grep -q C4_DPO_DONE $S/c4_dpo_build.log || { echo "DPO_BUILD_FAILED — abort"; tail -20 $S/c4_dpo_build.log; exit 1; }

# 2. retail eval
bash $T2/c4_dpo_eval.sh $GPU 8380 retail 114

# 3. airline 전이 eval
bash $T2/c4_dpo_eval.sh $GPU 8380 airline 50

# 4. 요약
echo "================ OVERNIGHT DPO SUMMARY ================"
echo "--- DPO train tail ---"; tail -6 $S/c4_dpo_build.log
echo "--- retail (base/dpo-pure/dpo-deny/dpo-perform · schema_copy) ---"; grep -h "C4DPO_ROW retail" $S/c4_dpo_eval_retail.log 2>/dev/null
echo "--- airline 전이 ---"; grep -h "C4DPO_ROW airline" $S/c4_dpo_eval_airline.log 2>/dev/null
echo "--- 대조: SFT schema_copy=52·base=40·autofetch=32·fewshot-retail=0 ---"
date; echo OVERNIGHT_DPO_DONE
