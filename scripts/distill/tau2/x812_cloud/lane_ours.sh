#!/usr/bin/env bash
# R2 — 회귀 태스크 우리-팔 레인 워커. 환경은 f1chk_024(4/4 확인분)와 동일.
# 사용: lane_ours.sh <레인> <포트> <큐파일> <태그접두>
set -u
LANE="$1"; PORT="$2"; QUEUE="$3"; PREFIX="$4"
LOCK="${QUEUE}.lock"; OUT=~/out; mkdir -p "$OUT" ~/logs
cd /home/woori/workspace_common/boltzmann-attention-pi/scripts/distill/tau2 || exit 1
source ./go_stack.sh >/dev/null 2>&1
source ./arms/viewmax2.env >/dev/null 2>&1
source ~/.openrouter_key; [ -f ~/.openai_key ] && source ~/.openai_key
export HF_HOME=/workspace/.hf_home T2_MAX_MODEL_LEN=131072 T2_FB_SIDECAR_TEXT=1
export PYTHONPATH=src:/root/t2/tau2
M="Qwen/Qwen3.8-27B-FP8"
GOT=$(curl -s -m 10 "http://localhost:$PORT/v1/models" | grep -oE "\"id\":\"[^\"]+\"" | head -1 | cut -d\" -f4)
[ "$GOT" = "$M" ] || { echo "[$LANE] 중단 - 서빙 모델 $GOT"; exit 1; }
echo "[$LANE] ours 워커 시작 :$PORT 서빙=$GOT 큐=$QUEUE"
cd /root/tau2-bench || exit 1
pop(){ local t; exec 9>"$LOCK"; flock 9
  t=$(head -1 "$QUEUE" 2>/dev/null); [ -n "$t" ] && sed -i "1d" "$QUEUE"
  flock -u 9; exec 9>&-; echo "$t"; }
while true; do
  T=$(pop); [ -z "$T" ] && { echo "[$LANE] 큐 소진 - 종료"; break; }
  TAG="${PREFIX}_${T}"
  echo "[$LANE $(date +%m-%d\ %H:%M)] -> $T (잔여 $(wc -l < "$QUEUE"))"
  export T2_FB_SIDECAR=/root/logs/fb_${TAG}.jsonl
  rm -rf /root/tau2-bench/data/simulations/$TAG
  /root/tau2-bench/venv/bin/python -u /root/t2/tau2/t2_run_gated.py \
    --domain banking_knowledge --gate 1 --retrieval_config alltools \
    --agent_model "$M" --agent_base "http://localhost:$PORT/v1" \
    --user_llm openrouter/openai/gpt-5.2 --user_temp 0.0 --user_reasoning_effort low \
    --task_ids "$T" --num_trials 4 --max_concurrency 4 --max_steps 200 \
    --save_to "$TAG" > /root/logs/${TAG}_drv.log 2>&1 || echo "  [$LANE] FAIL $T"
  d=/root/tau2-bench/data/simulations/$TAG
  [ -f "$d/results.json" ] && gzip -c "$d/results.json" > "$OUT/$TAG.results.json.gz"
  [ -f /root/logs/${TAG}_drv.log ] && gzip -c /root/logs/${TAG}_drv.log > "$OUT/${TAG}_drv.log.gz"
  [ -f /root/logs/fb_${TAG}.jsonl ] && gzip -c /root/logs/fb_${TAG}.jsonl > "$OUT/fb_${TAG}.jsonl.gz"
  echo "  [$LANE] 영속 $TAG"
done
date; echo "${LANE}_OURS_DONE"
