#!/bin/bash
# SEL-5 (큐 ⑶): MBR-shortlist + 7B pairwise judge 토너먼트를 최적 풀(dpo2g-AR8+H6)에 적용.
# 사전등록: 공식 link F1 vs SEL-1 0.6722 / SEL-4 0.6803 — pairwise가 잔여 oracle 갭(~0.176)을
#   더 회수하는가. shortlist 3·5 두 변형(민감도). 동점=MBR 서열이라 하한=MBR-top(SEL-1)과 동등.
# Run: setsid bash driver_sel5.sh </dev/null >/dev/null 2>&1 &
set -u
R=/home/woori/workspace_common/boltzmann-attention-pi
TB=/home/woori/scratch/JARVIS_tb/taskbench
TBPRED=$TB/data_multimedia_sub500/predictions
IP=/home/woori/scratch/tbeval_venv/bin/python
VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
S=/home/woori/scratch
exec > $S/sel5_dpo2g.log 2>&1
set -x
cd $R && git pull --ff-only -q

# GPU1 정리 후 base 7B 서빙 (SEL-4 패턴)
for p in $(nvidia-smi --id=1 --query-compute-apps=pid --format=csv,noheader); do
  kill -9 $p 2>/dev/null; done
sleep 8
CUDA_VISIBLE_DEVICES=1 setsid nohup $VLLM serve Qwen/Qwen2.5-7B-Instruct \
  --port 8001 --served-model-name base_model --max-model-len 8192 \
  --gpu-memory-utilization 0.85 > $S/vllm_sel5.log 2>&1 &
ok=0
for i in $(seq 1 90); do
  curl -s localhost:8001/v1/models | grep -q base_model && ok=1 && break; sleep 10
done
[ $ok = 1 ] || { echo SEL5_SERVE_FAIL; exit 1; }

for K in 3 5; do
  $IP $R/scripts/distill/taskbench/tb_pairwise_select.py --tb_dir $TB \
    --ar_tag tb_dpo2g_mmk --ar_group dpo2g \
    --endpoint http://localhost:8001/v1 --served base_model --shortlist $K \
    --out $TBPRED/tb_sel5k${K}_dpo2g.json
  $IP $R/scripts/distill/taskbench/tb_build_eval.py --tb_dir $TB --domain data_multimedia \
    --pred_file $TBPRED/tb_sel5k${K}_dpo2g.json \
    --dst $TB/data_multimedia_sub500_eval_tb_sel5k${K} --llm tb_sel5k${K} \
    > $S/sel5k${K}_eval.txt 2>&1
  echo "=== SEL-5 shortlist=$K ==="
  grep -hE "link_binary_f1|node_micro_f1_no" $S/sel5k${K}_eval.txt
done

for p in $(nvidia-smi --id=1 --query-compute-apps=pid --format=csv,noheader); do
  kill -9 $p 2>/dev/null; done
echo "SEL5_DONE $(date)"
