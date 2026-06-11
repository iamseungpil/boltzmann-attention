#!/bin/bash
# DPO(dpo_mm) 판정 배터리 (HANDOFF_2026_06_11 §0.3): ①P/R-결손 ②census rft2->dpo ③snap 겹침
# 산출: /home/woori/scratch/dpo_judgment.log (sentinel JUDGMENT_DONE)
R=/home/woori/workspace_common/boltzmann-attention-pi
TB=/home/woori/scratch/JARVIS_tb/taskbench
IP=/home/woori/scratch/tbeval_venv/bin/python
S=/home/woori/scratch
cd $R
{
echo "== 0. official blocks (label -> scores, from eval log) =="
grep -E "tb_build_eval.py|link_binary_f1 =|node_micro_f1" $S/tb_evaladapter_dpo_mm.log | grep -v "^++"

echo "== 1. PR census (MM full: rft2 vs dpo) =="
$IP scripts/distill/taskbench/tb_pr_census.py \
  --set $TB/data_multimedia_evalfull_tb_rft2_mm:tb_rft2_mm \
  --set $TB/data_multimedia_evalfull_tb_dpo_mm:tb_dpo_mm

echo "== 2. tb_census rft2 -> dpo (MM full) =="
$IP scripts/distill/taskbench/tb_census.py \
  --dir_a $TB/data_multimedia_evalfull_tb_rft2_mm --llm_a tb_rft2_mm \
  --dir_b $TB/data_multimedia_evalfull_tb_dpo_mm --llm_b tb_dpo_mm \
  --tool_desc $TB/data_multimedia/tool_desc.json --dep resource \
  --out $S/census_rft2_to_dpo_mm.md
sed -n 1,10p $S/census_rft2_to_dpo_mm.md

echo "== 3. snap overlay (dpo MM full) =="
$IP scripts/distill/taskbench/tb_name_snap.py \
  --pred $TB/data_multimedia_evalfull_tb_dpo_mm/predictions/tb_dpo_mm.json \
  --tool_desc $TB/data_multimedia/tool_desc.json \
  --out $S/tb_dpo_mm_snapped.json
$IP scripts/distill/taskbench/tb_build_eval.py --tb_dir $TB --domain data_multimedia \
  --pred_file $S/tb_dpo_mm_snapped.json \
  --dst $TB/data_multimedia_snapeval_tb_dpo_mm --llm tb_dpo_mm

echo JUDGMENT_DONE
} > $S/dpo_judgment.log 2>&1
