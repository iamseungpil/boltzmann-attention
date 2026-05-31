#!/usr/bin/env bash
# ============================================================================
# run_arm3_sweep.sh — SOPBench arm-1 (baseline) + arm-3 (L1 TwoStage) sweep.
#
#   *** TO RUN A NEW MODEL, CHANGE ONLY `MODEL` (and serve that model). ***
#
# Everything else (domains, eval, scoring) is the SOPBench author pipeline, so
# every number is directly comparable to the leaderboard / Track-A 7B numbers.
#
# PREREQUISITES (once per clone):
#   1. git clone the SOPBench repo into $CLONE (https://github.com/zli12321/SOPBench
#      or your existing clone). cd into it.
#   2. Deploy the arm-3 patch (idempotent, makes .bak backups):
#        python /path/to/bap-pi/scripts/distill/sopbench/apply_two_stage_patch.py "$CLONE"
#   3. Serve the model on a vLLM OpenAI endpoint with the SHORT canonical name
#      (see SERVING section at the bottom). The served-model-name MUST equal $MODEL.
#
# USAGE:
#   MODEL=qwen2.5-32b-instruct SOPBENCH_VLLM_BASE_URL=http://localhost:9100/v1 \
#     PY=/home/woori/venvs/seka_env/bin/python bash run_arm3_sweep.sh
#
#   # 72B, served on a second endpoint, arm-3 only, bank+dmv first:
#   MODEL=qwen2.5-72b-instruct SOPBENCH_VLLM_BASE_URL=http://localhost:9000/v1 \
#     ARMS="arm3" DOMAINS="bank dmv" bash run_arm3_sweep.sh
# ============================================================================
set -u

# ---- the ONLY thing you normally change -----------------------------------
MODEL="${MODEL:-qwen2.5-32b-instruct}"     # served-model-name (short id, must be in
                                           # swarm/constants.py FUNCTION_CALLING_MODELS["vllm"]
                                           # — qwen2.5-{3b,7b,14b,32b,72b}-instruct already are)
# ---------------------------------------------------------------------------

PY="${PY:-/home/woori/venvs/seka_env/bin/python}"
export SOPBENCH_VLLM_BASE_URL="${SOPBENCH_VLLM_BASE_URL:-http://localhost:9100/v1}"
DOMAINS="${DOMAINS:-bank dmv healthcare hotel library online_market university}"
TOOL_LIST="${TOOL_LIST:-full}"             # full = leaderboard-standard; oracle = diagnostic
MODE="${MODE:-fc}"                         # arm-3 resolver is fc-native; keep fc for both arms
                                           #   so arm3 vs arm1 is same-mode (apples-to-apples).
ARMS="${ARMS:-arm1 arm3}"                  # subset of: arm1 arm3 arm3det arm1react
NUM_TASKS="${NUM_TASKS:-}"                 # empty = all tasks in the domain
OUT="${OUT:-./output_coworker}"
MAXTURNS="${MAXTURNS:-20}"
MAXACTIONS="${MAXACTIONS:-10}"

SUMMARY="${OUT}/summary_${MODEL//\//_}.tsv"
mkdir -p "$OUT"
printf "arm\tmode\ttool\tdomain\tpass_rate\taction_called\n" > "$SUMMARY"

echo "============================================================"
echo " SOPBench sweep  MODEL=$MODEL  endpoint=$SOPBENCH_VLLM_BASE_URL"
echo " arms=[$ARMS]  domains=[$DOMAINS]  tool=$TOOL_LIST  mode=$MODE"
echo "============================================================"

nt_flag=""; [ -n "$NUM_TASKS" ] && nt_flag="--num_tasks $NUM_TASKS"

run_one () {   # $1=arm  $2=mode  $3=extra run_simulation flags
  local arm="$1" mode="$2" extra="$3"
  local odir="${OUT}/${arm}"
  for d in $DOMAINS; do
    echo "--- [$arm] $d ($mode/$TOOL_LIST) ---"
    $PY run_simulation.py \
      --domain "$d" --assistant_model "$MODEL" \
      --tool_call_mode "$mode" --tool_list "$TOOL_LIST" $extra \
      --env_mode prompt --max_num_turns "$MAXTURNS" --max_num_actions "$MAXACTIONS" \
      --output_dir "$odir" $nt_flag > "${odir}/${d}.simlog" 2>&1
    # score with the UNCHANGED author evaluator
    local ev
    ev=$($PY run_evaluation.py \
          --domain "$d" --assistant_model "$MODEL" \
          --tool_call_mode "$mode" --tool_list "$TOOL_LIST" \
          --output_dir "$odir" 2>/dev/null)
    local pr ac
    pr=$(printf "%s" "$ev" | grep -oE "Mean Pass Rate: [0-9.]+" | grep -oE "[0-9.]+$" | head -1)
    ac=$(printf "%s" "$ev" | grep -oE "percentage_action_successfully_called[ ]+[0-9.]+" | grep -oE "[0-9.]+$" | head -1)
    printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$arm" "$mode" "$TOOL_LIST" "$d" "${pr:-NA}" "${ac:-NA}" | tee -a "$SUMMARY"
  done
}

for arm in $ARMS; do
  case "$arm" in
    arm1)     run_one arm1     "$MODE"  "" ;;                       # LLM-alone baseline (same mode)
    arm3)     run_one arm3     "$MODE"  "--two_stage" ;;            # L1 planner + LLM resolver
    arm3det)  run_one arm3det  "$MODE"  "--two_stage --two_stage_det" ;;  # + deterministic shortcut
    arm1react) run_one arm1react "react" "" ;;                     # leaderboard-anchor (OSS=ReAct)
    *) echo "unknown arm: $arm (skip)";;
  esac
done

echo; echo "================= SUMMARY  ($MODEL) ================="
column -t -s $'\t' "$SUMMARY"
echo "raw: $SUMMARY"
echo
echo "Compare arm3 vs arm1 (same $MODE/$TOOL_LIST mode) per domain for the L1-structure delta."
echo "Track-A 7B reference (fc/full): bank 3.7 dmv 11.3 health 8.1 hotel 0.5 library 13.6 market 9.3 univ 4.8 (avg 7.3%)."
echo "Leaderboard react/full: 32B avg 39.8 (bank 40.3) / 72B avg 36.4 (bank 35.1)."
