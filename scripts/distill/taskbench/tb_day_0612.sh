#!/bin/bash
# ★주간 배치 2026-06-12 — D1 구조-표적 DPO ∥ D2 비용-표적 DPO (병렬 학습→병렬 평가→판정→보고서 push)
# 사전예측(동결): D1 in-domain edge +1~3·held-out은 본 실험의 질문(방향 미커밋)·nself/dangle ↓
#                D2 평균 노드수 ↓·P ↑·edge ±1 내·⚠️short/deficit 악화 감시(v1 거울상 위험)
# log: /home/woori/scratch/tb_day.log, sentinel DAY_DONE
R=/home/woori/workspace_common/boltzmann-attention-pi
TB=/home/woori/scratch/JARVIS_tb/taskbench
RUNS=$R/reports/facet_rft_2026/phase4_distill/sft_runs
IP=/home/woori/scratch/tbeval_venv/bin/python
PY=/home/woori/venvs/seka_env/bin/python
SC=$R/scripts/distill/taskbench
S=/home/woori/scratch
REPORT=$R/reports/facet_rft_2026/DAY_REPORT_2026_06_12.md
exec > $S/tb_day.log 2>&1
set -x
cd $R && git pull --ff-only -q

train() { # gpu pairs outname
  CUDA_VISIBLE_DEVICES=$1 $PY $R/scripts/distill/sopbench/dpo_train.py \
    --base Qwen/Qwen2.5-7B-Instruct --sft-adapter $RUNS/qwen7b_tb_rft2_mm \
    --pairs $2 --out-dir $RUNS/$3 --max-seq-len 6144 --epochs 2 --beta 0.1 --lr 5e-6 \
    > $S/tb_train_$3.log 2>&1
}
train 1 $S/tb_rft/dpo_structure.jsonl qwen7b_tb_dpo_struct &
T1=$!
train 0 $S/tb_rft/dpo_cost.jsonl qwen7b_tb_dpo_cost &
T2=$!
wait $T1 || echo TRAIN_STRUCT_FAIL
wait $T2 || echo TRAIN_COST_FAIL

# 병렬 평가 (struct=GPU0/port8000, cost=GPU1/port8001 — tb_eval_adapter가 PORT=8000+GPU)
bash $SC/tb_eval_adapter.sh dpo_struct data_multimedia "data_huggingface data_dailylifeapis" 0 0.85 &
E1=$!
bash $SC/tb_eval_adapter.sh dpo_cost data_multimedia "data_huggingface data_dailylifeapis" 1 0.85 &
E2=$!
wait $E1; wait $E2

# 판정 배터리: P/R census (rft2·dpo2 대조) + 궤적 census(rft2→각각)
$IP $SC/tb_pr_census.py \
  --set $TB/data_multimedia_evalfull_tb_rft2_mm:tb_rft2_mm \
  --set $TB/data_multimedia_evalfull_tb_dpo2_mm:tb_dpo2_mm \
  --set $TB/data_multimedia_evalfull_tb_dpo_struct:tb_dpo_struct \
  --set $TB/data_multimedia_evalfull_tb_dpo_cost:tb_dpo_cost > $S/day_pr.txt
for T in dpo_struct dpo_cost; do
  $IP $SC/tb_census.py --dir_a $TB/data_multimedia_evalfull_tb_rft2_mm --llm_a tb_rft2_mm \
    --dir_b $TB/data_multimedia_evalfull_tb_${T} --llm_b tb_${T} \
    --tool_desc $TB/data_multimedia/tool_desc.json --dep resource \
    --out $S/census_rft2_to_${T}.md
done

# 보고서
{
echo "# DAY REPORT 2026-06-12 — D1 구조-표적 / D2 비용-표적 DPO (자동 생성)"
echo "> 쌍: structure 1017 / cost 376 (rft2 rollout, 어휘청정·통제 필터). base=rft2. 통제: rft2 MM full 49.0 / dpo2 55.95 / in-domain HF 51.6·daily 85.0(sub500, rft2)."
for T in dpo_struct dpo_cost; do
  echo; echo "## $T 공식 수치 (MM full / HF sub500 / daily sub500 순)"
  grep -E "  link_binary_f1|  node_micro_f1_no" $S/tb_evaladapter_${T}.log | head -6
done
echo; echo "## P/R·결손 (MM full)"; cat $S/day_pr.txt
for T in dpo_struct dpo_cost; do
  echo; echo "## census rft2->${T} aggregate"; sed -n 3,6p $S/census_rft2_to_${T}.md
  grep -E "^## (improved|worsened|same):" $S/census_rft2_to_${T}.md
done
} > $REPORT
cd $R && git stash -q 2>/dev/null; git pull --rebase -q; git stash pop -q 2>/dev/null
git add reports/facet_rft_2026/DAY_REPORT_2026_06_12.md && \
  git commit -q -m "DAY_REPORT 2026-06-12: structure/cost-targeted DPO auto results" && git push -q || echo PUSH_FAIL
echo "DAY_DONE $(date)"
