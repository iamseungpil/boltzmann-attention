#!/bin/bash
# EVAL-ONLY (run AFTER xdomain_train_queue done): eval each held-out adapter on its target(s).
# LODO_X -> test X (stack + adapter-only). train1_X -> test the other 6 (stack). eval_tasks (robust).
# Idempotent-ish: skips a (tag,domain) whose json already has 'evaluations'. Serve on GPU0.
exec > /home/woori/scratch/sft_alias_run/xdomain_eval_heldout.log 2>&1
set -x
REPO=/home/woori/workspace_common/boltzmann-attention-pi
CL=/home/woori/scratch/SOPBench
SEKA=/home/woori/venvs/seka_env/bin/python
VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
RUNS=$REPO/reports/facet_rft_2026/phase4_distill/sft_runs
SB=$REPO/scripts/distill/sopbench
OUT=/home/woori/scratch/sft_alias_run
PORT=8351
ALL="bank dmv healthcare hotel library online_market university"
STACK="SOPBENCH_ALIAS=1 SOPBENCH_GATE=1 SOPBENCH_SCRATCHPAD=1 SOPBENCH_SOURCE=1 SOPBENCH_PLAN_MAXTOK=1024 SOPBENCH_OFFLOAD=1 SOPBENCH_OFFLOAD_ACTIVE=1 SOPBENCH_ARGFIX=1 SOPBENCH_VALFIX=1 SOPBENCH_KEEPTUPLE=1 SOPBENCH_DGGATE=1 SOPBENCH_LOGINFIRST=1 SOPBENCH_STOPSUCCESS=1"
minus () { for d in $ALL; do [ "$d" != "$1" ] && printf "%s " "$d"; done; }
cd $CL && git checkout -- swarm/ run_simulation.py 2>/dev/null || true
$SEKA $SB/apply_two_stage_patch.py $CL
serve () { rm -f /dev/shm/vllm* 2>/dev/null; CUDA_VISIBLE_DEVICES=0 nohup $VLLM serve Qwen/Qwen2.5-7B-Instruct --port $PORT --enable-lora --lora-modules tbox_v2=$1 --enable-auto-tool-choice --tool-call-parser hermes --max-model-len 8192 --gpu-memory-utilization 0.85 --dtype bfloat16 --trust-remote-code > $OUT/serve_evalho.log 2>&1 & SPID=$!; for i in $(seq 1 90); do curl -s http://localhost:$PORT/v1/models 2>/dev/null | grep -q tbox_v2 && break; sleep 10; done; }
do_eval () { # tag domain flags outdir
  local od=$4 D=$2; local f=$od/$D/ast_tbox_v2-mode_fc-dep_full-fmt_structured-tool_full-shuffle_False.json
  if [ -f "$f" ] && $SEKA -c "import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if (d and 'evaluations' in d[0]) else 1)" "$f" 2>/dev/null; then echo "SKIP $1 $D (done)"; return; fi
  env $3 SOPBENCH_VLLM_BASE_URL=http://localhost:$PORT/v1 $SEKA run_simulation.py --domain $D --assistant_model tbox_v2 --tool_call_mode fc --tool_list full --two_stage --two_stage_v2 --ont_dir $CL/induced --output_dir $od --env_mode prompt
  $SEKA $SB/eval_tasks.py "$f"
}
# LODO held-out: dmv hotel online_market university  (bank/library/healthcare already done)
for X in dmv hotel online_market university; do
  AD=$RUNS/qwen7b_tbox_t1c_lodo_$X; [ -f $AD/adapter_model.safetensors ] || { echo "MISSING lodo_$X"; continue; }
  serve $AD
  do_eval lodo_$X $X "$STACK" $OUT/xho_lodo_${X}_${X}_stack
  do_eval lodo_$X $X ""       $OUT/xho_lodo_${X}_${X}_adapteronly
  kill $SPID 2>/dev/null; sleep 5
done
# train-1: dmv healthcare hotel library online_market university (bank done via finish_train1bank)
for X in dmv healthcare hotel library online_market university; do
  AD=$RUNS/qwen7b_tbox_t1c_train1_$X; [ -f $AD/adapter_model.safetensors ] || { echo "MISSING train1_$X"; continue; }
  serve $AD
  for D in $(minus $X); do do_eval train1_$X $D "$STACK" $OUT/xho_train1${X}_${D}_stack; done
  kill $SPID 2>/dev/null; sleep 5
done
echo "XDOMAIN_EVAL_HELDOUT_DONE $(date)"
