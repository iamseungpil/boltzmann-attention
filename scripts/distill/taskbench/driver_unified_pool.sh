#!/bin/bash
# 통합 풀 선별 (TB §8.9g "다음 1수" — 2026-06-13, GPU0 유휴 활용)
# 풀 = dpo2g-AR8(품질·강한합의) + P-lora8(목적-다양) + H6(크기/세대/도메인-이종) = 22종.
# 가설: sel≈mean+회수×(oracle-mean) — dpo2g가 mean↑, P-lora/H6가 oracle↑ → 합산이 둘 다.
# 사전등록: 통합 SEL-1+SEL-4 공식 link F1 > best-stack 0.6803 (품질+다양성 합산이 단일축 초과).
#   미달 시 = "다양성↑이 품질↓로 상쇄"(P-lora 어댑터가 평균 끌어내림) → dpo2g+H6가 최적 유지.
# 구성: dpo2g 8개=1그룹(같은정책 K샘플) / P-lora 8개=각 독립그룹(--extra) / H6 6개=각 그룹(자동).
# GPU0 전용. log: /home/woori/scratch/unified_pool.log, sentinel UNIFIED_DONE
set -u
R=/home/woori/workspace_common/boltzmann-attention-pi
SC=$R/scripts/distill/taskbench
TB=/home/woori/scratch/JARVIS_tb/taskbench
TBPRED=$TB/data_multimedia_sub500/predictions
VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
IP=/home/woori/scratch/tbeval_venv/bin/python
S=/home/woori/scratch
exec > $S/unified_pool.log 2>&1
set -x
cd $R && git pull --ff-only -q
if nvidia-smi --id=0 --query-compute-apps=pid --format=csv,noheader | grep -q .; then
  echo "UNIFIED_ABORT_GPU0_BUSY"; exit 1
fi

# P-lora 8종을 extra로 (각 독립 그룹). H6는 hm_list 자동 포함.
EXTRA=""
for k in 0 1 2 3 4 5 6 7; do EXTRA="$EXTRA --extra plora$k=$TBPRED/tb_dl_$k.json"; done

eval_one() {
  $IP $SC/tb_build_eval.py --tb_dir $TB --domain data_multimedia \
    --pred_file $1 --dst $TB/data_multimedia_sub500_eval_$2 --llm $2 > $S/up_$2.txt 2>&1
  echo -n "  $2: "; grep -hE "link_binary_f1" $S/up_$2.txt | head -1
}

# SEL-1: 통합 22종 prior-MBR (dpo2g=1그룹 + P-lora8 + H6)
(cd $SC && $IP tb_select_official.py --tb_dir $TB --domain data_multimedia \
  --ar_tag tb_dpo2g_mmk --ar_group dpo2g --prior_beta 2.0 $EXTRA \
  --out $TBPRED/tb_unified_sel1.json)
eval_one $TBPRED/tb_unified_sel1.json unified_sel1

# SEL-4: 통합 22종 + 7B Reviewer (GPU0)
kill_gpu0() { for p in $(nvidia-smi --id=0 --query-compute-apps=pid --format=csv,noheader); do
  kill -9 $p 2>/dev/null; done; sleep 8; }
CUDA_VISIBLE_DEVICES=0 VLLM_PORT=8160 setsid nohup $VLLM serve Qwen/Qwen2.5-7B-Instruct \
  --port 8003 --served-model-name base_model --max-model-len 8192 \
  --gpu-memory-utilization 0.85 > $S/vllm_unified.log 2>&1 &
ok=0
for i in $(seq 1 90); do
  curl -s localhost:8003/v1/models | grep -q base_model && ok=1 && break; sleep 10
done
if [ $ok = 1 ]; then
  (cd $SC && $IP tb_reviewer_select.py --tb_dir $TB --domain data_multimedia \
    --ar_tag tb_dpo2g_mmk --ar_group dpo2g $EXTRA \
    --endpoint http://localhost:8003/v1 --served base_model --lam 1.0 \
    --out $TBPRED/tb_unified_sel4.json)
  eval_one $TBPRED/tb_unified_sel4.json unified_sel4
else
  echo "UNIFIED_SEL4_SERVE_FAIL"
fi
kill_gpu0

{ echo "== [통합 22-풀] SEL-1 / SEL-4 (vs best-stack 0.6803)"
  grep -hE "link_binary_f1" $S/up_unified_sel1.txt $S/up_unified_sel4.txt 2>/dev/null
} >> $S/day13_summary.txt
echo "UNIFIED_DONE $(date)" | tee -a $S/day13_summary.txt
