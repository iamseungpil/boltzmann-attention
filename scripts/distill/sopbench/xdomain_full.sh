#!/bin/bash
# Full cross-domain transfer: train ALL 7 LODO adapters + ALL 7 train-1 adapters, eval each on its
# HELD-OUT target(s) with STACK (LOGINCALL off, augment off) + adapter-only(LODO only). eval_tasks
# (robust; run_evaluation crashes). bank LODO(43.3%) + library/healthcare done earlier; this adds the rest.
exec > /home/woori/scratch/sft_alias_run/xdomain_full.log 2>&1
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

# ---- 1) TRAIN missing adapters (2 parallel on GPU0/GPU1) ----
# LODO remaining: dmv hotel online_market university  | train-1 remaining: dmv healthcare hotel library online_market university
NEED=""
for X in dmv hotel online_market university; do NEED="$NEED lodo_$X"; done
for X in dmv healthcare hotel library online_market university; do NEED="$NEED train1_$X"; done
train_one () { # name gpu domains...
  local name=$1 gpu=$2; shift 2
  setsid bash $SB/xdomain_train.sh t1c_$name $gpu "$@" </dev/null >/dev/null 2>&1 &
}
gpu=0
for cfg in $NEED; do
  [ -f $RUNS/qwen7b_tbox_t1c_${cfg}/adapter_model.safetensors ] && { echo "SKIP $cfg (exists)"; continue; }
  case $cfg in
    lodo_*)   X=${cfg#lodo_};   DOMS=$(minus $X);;
    train1_*) X=${cfg#train1_}; DOMS=$X;;
  esac
  # wait for a free GPU slot
  while [ "$(pgrep -fc lora_train_chat_toolcall)" -ge 2 ]; do sleep 30; done
  # pick the idle gpu
  if [ "$(nvidia-smi --id=0 --query-compute-apps=pid --format=csv,noheader | wc -l)" -eq 0 ]; then gpu=0; else gpu=1; fi
  echo "TRAIN $cfg on GPU$gpu doms=$DOMS"
  train_one $cfg $gpu $DOMS
  sleep 45
done
while [ "$(pgrep -fc lora_train_chat_toolcall)" -ge 1 ]; do sleep 30; done
echo "ALL_TRAIN_DONE $(date)"

# ---- 2) EVAL each adapter on held-out target(s); serve on GPU0, eval_tasks ----
serve () { rm -f /dev/shm/vllm* 2>/dev/null; CUDA_VISIBLE_DEVICES=0 nohup $VLLM serve Qwen/Qwen2.5-7B-Instruct --port $PORT --enable-lora --lora-modules tbox_v2=$1 --enable-auto-tool-choice --tool-call-parser hermes --max-model-len 8192 --gpu-memory-utilization 0.85 --dtype bfloat16 --trust-remote-code > $OUT/serve_full.log 2>&1 & SPID=$!; for i in $(seq 1 90); do curl -s http://localhost:$PORT/v1/models 2>/dev/null | grep -q tbox_v2 && break; sleep 10; done; }
sim_eval () { # tag domain flags outdir
  local tag=$1 D=$2 flags=$3 od=$4
  env $flags SOPBENCH_VLLM_BASE_URL=http://localhost:$PORT/v1 $SEKA run_simulation.py --domain $D --assistant_model tbox_v2 --tool_call_mode fc --tool_list full --two_stage --two_stage_v2 --ont_dir $CL/induced --output_dir $od --env_mode prompt
  $SEKA $SB/eval_tasks.py $od/$D/ast_tbox_v2-mode_fc-dep_full-fmt_structured-tool_full-shuffle_False.json
}
cd $CL && git checkout -- swarm/ run_simulation.py 2>/dev/null || true
$SEKA $SB/apply_two_stage_patch.py $CL
# LODO (remaining 4): stack + adapter-only on the held-out domain
for X in dmv hotel online_market university; do
  AD=$RUNS/qwen7b_tbox_t1c_lodo_$X; [ -f $AD/adapter_model.safetensors ] || continue
  serve $AD
  sim_eval lodo_$X $X "$STACK" $OUT/xho_lodo_${X}_${X}_stack
  sim_eval lodo_$X $X "" $OUT/xho_lodo_${X}_${X}_adapteronly
  kill $SPID 2>/dev/null; sleep 5
done
# train-1 (ALL 7, incl bank for clean eval): STACK on the other 6 domains
for X in bank dmv healthcare hotel library online_market university; do
  AD=$RUNS/qwen7b_tbox_t1c_train1_$X; [ -f $AD/adapter_model.safetensors ] || continue
  serve $AD
  for D in $(minus $X); do sim_eval train1_$X $D "$STACK" $OUT/xho_train1${X}_${D}_stack; done
  kill $SPID 2>/dev/null; sleep 5
done
echo "XDOMAIN_FULL_DONE $(date)"
