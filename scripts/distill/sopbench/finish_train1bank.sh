#!/bin/bash
# Finish train1_bank (train=bank only) eval on its held-out 6 domains. 4 done (dmv/healthcare/hotel/
# library), finish online_market+university, then eval_tasks ALL 6 (run_evaluation crashed earlier).
exec > /home/woori/scratch/sft_alias_run/finish_train1bank.log 2>&1
set -x
REPO=/home/woori/workspace_common/boltzmann-attention-pi
CL=/home/woori/scratch/SOPBench
SEKA=/home/woori/venvs/seka_env/bin/python
VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
SB=$REPO/scripts/distill/sopbench
OUT=/home/woori/scratch/sft_alias_run
AD=$REPO/reports/facet_rft_2026/phase4_distill/sft_runs/qwen7b_tbox_t1c_train1_bank
PORT=8351
STACK="SOPBENCH_ALIAS=1 SOPBENCH_GATE=1 SOPBENCH_SCRATCHPAD=1 SOPBENCH_SOURCE=1 SOPBENCH_PLAN_MAXTOK=1024 SOPBENCH_OFFLOAD=1 SOPBENCH_OFFLOAD_ACTIVE=1 SOPBENCH_ARGFIX=1 SOPBENCH_VALFIX=1 SOPBENCH_KEEPTUPLE=1 SOPBENCH_DGGATE=1 SOPBENCH_LOGINFIRST=1 SOPBENCH_STOPSUCCESS=1"
cd $CL && git checkout -- swarm/ run_simulation.py 2>/dev/null || true
$SEKA $SB/apply_two_stage_patch.py $CL
rm -f /dev/shm/vllm* 2>/dev/null
CUDA_VISIBLE_DEVICES=0 nohup $VLLM serve Qwen/Qwen2.5-7B-Instruct --port $PORT --enable-lora --lora-modules tbox_v2=$AD --enable-auto-tool-choice --tool-call-parser hermes --max-model-len 8192 --gpu-memory-utilization 0.85 --dtype bfloat16 --trust-remote-code > $OUT/serve_finish_t1bank.log 2>&1 &
SPID=$!
for i in $(seq 1 90); do curl -s http://localhost:$PORT/v1/models 2>/dev/null | grep -q tbox_v2 && break; sleep 10; done
for D in online_market university; do
  od=$OUT/xho_train1bank_${D}_stack
  env $STACK SOPBENCH_VLLM_BASE_URL=http://localhost:$PORT/v1 $SEKA run_simulation.py --domain $D --assistant_model tbox_v2 --tool_call_mode fc --tool_list full --two_stage --two_stage_v2 --ont_dir $CL/induced --output_dir $od --env_mode prompt
  echo "SIM_DONE_$D"
done
kill $SPID 2>/dev/null; sleep 4; rm -f /dev/shm/vllm* 2>/dev/null
echo "=== eval_tasks all 6 train1_bank stack + summary ==="
for D in dmv healthcare hotel library online_market university; do
  f=$OUT/xho_train1bank_${D}_stack/${D}/ast_tbox_v2-mode_fc-dep_full-fmt_structured-tool_full-shuffle_False.json
  [ -f "$f" ] && $SEKA $SB/eval_tasks.py "$f"
done
echo "FINISH_TRAIN1BANK_DONE $(date)"
