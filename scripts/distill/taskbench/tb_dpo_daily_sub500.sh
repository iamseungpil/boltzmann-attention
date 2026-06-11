#!/bin/bash
# dpo_mm in-domain daily sub500 보완 (eval 누락분 — PS argv 따옴표 절단으로 TRAIN_DOMS서 탈락).
# 전제: vllm tb_dpo_mm이 port 8000에 서빙 중. 산출 log: /home/woori/scratch/tb_dpo_daily.log
R=/home/woori/workspace_common/boltzmann-attention-pi
TB=/home/woori/scratch/JARVIS_tb/taskbench
IP=/home/woori/scratch/tbeval_venv/bin/python
{
set -x
cd $TB
$IP inference.py --data_dir data_dailylifeapis_sub500 --api_addr localhost --api_port 8000 \
  --api_key dummy --llm tb_dpo_mm --multiworker 8 --dependency_type temporal
$IP $R/scripts/distill/taskbench/tb_build_eval.py --tb_dir $TB --domain data_dailylifeapis \
  --pred_file $TB/data_dailylifeapis_sub500/predictions/tb_dpo_mm.json \
  --dst $TB/data_dailylifeapis_sub500_eval_tb_dpo_mm --llm tb_dpo_mm
echo DAILY_SUB500_DONE
} > /home/woori/scratch/tb_dpo_daily.log 2>&1
