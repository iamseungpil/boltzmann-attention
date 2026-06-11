#!/bin/bash
# edge-snap v0 측정 배터리 (zero-GPU): MM(resource) 5 타깃 — 7B base 통제·7B best-stack·
# 32B base·32B-SFT·72B base. 산출 log: /home/woori/scratch/tb_edgesnap.log
R=/home/woori/workspace_common/boltzmann-attention-pi
TB=/home/woori/scratch/JARVIS_tb/taskbench
IP=/home/woori/scratch/tbeval_venv/bin/python
RAW=$R/reports/facet_rft_2026/trackb_raw/preds/data_multimedia_sub500
S=/home/woori/scratch
TD=$TB/data_multimedia/tool_desc.json
exec > $S/tb_edgesnap.log 2>&1
set -x
cd $R && git pull --ff-only -q

run() { # tag pred_file
  $IP scripts/distill/taskbench/tb_edge_snap.py --pred $2 --tool_desc $TD \
    --out $S/esnap_${1}.json
  $IP scripts/distill/taskbench/tb_build_eval.py --tb_dir $TB --domain data_multimedia \
    --pred_file $S/esnap_${1}.json --dst $TB/mm_esnap_eval_${1} --llm ${1} 2>/dev/null \
    | grep -E "link_binary_f1|node_micro_f1_no"
  echo "ARM_DONE $1"
}

run qwen7b        $TB/data_multimedia_evalfull_qwen7b/predictions/qwen7b.json
run tb_dpo2_mm_guided $TB/data_multimedia/predictions/tb_dpo2_mm_guided.json
run qwen25_32b    $RAW/qwen25_32b.json
run tb_lodo_mm_32b $RAW/tb_lodo_mm_32b.json
run qwen25_72b    $RAW/qwen25_72b.json

# worsened 검사 (32B base: snap 전후 census — 과잉-재작성 감시)
$IP scripts/distill/taskbench/tb_census.py \
  --dir_a $TB/mm_sub500_verify_qwen25_32b --llm_a qwen25_32b \
  --dir_b $TB/mm_esnap_eval_qwen25_32b --llm_b qwen25_32b \
  --tool_desc $TD --dep resource --examples 3 \
  --out $S/census_esnap_q32b.md
grep -E "^## (improved|worsened|same):" $S/census_esnap_q32b.md
echo "EDGESNAP_DONE $(date)"
