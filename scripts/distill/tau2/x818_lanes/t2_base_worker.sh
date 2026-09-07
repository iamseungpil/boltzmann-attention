#!/usr/bin/env bash
# t2_base_worker — base(--gate 0) nt=4 큐 워커. 여러 레인이 한 큐를 flock 으로 공유.
# 사용: t2_base_worker.sh <레인> <HOST> <PORT> <큐파일>
#   ⛔git pull 없음 · ⛔bracket grep · 발사 전 /v1/models id 대조([[30]])
set -u
LANE="$1"; AHOST="$2"; PORT="$3"; QUEUE="$4"
T2=/home/woori/workspace_common/boltzmann-attention-pi/scripts/distill/tau2
PY=/home/woori/venvs/seka_env/bin/python
TB=/home/woori/iso_tau3/tau2-bench
M="Qwen/Qwen3.8-27B-FP8"
LOCK="${QUEUE}.lock"
GOT=$(curl -s -m 10 "http://$AHOST:$PORT/v1/models" | grep -oE '"id":"[^"]+"' | head -1 | cut -d'"' -f4)
case "$GOT" in
  *Qwen3.8*) echo "[$LANE] 서빙 모델 = $GOT ($AHOST:$PORT)" ;;
  *) echo "[$LANE] 중단 — Q3.8 이 아니다: $GOT"; exit 1 ;;
esac
source /home/woori/.openrouter_key
[ -f /home/woori/.openai_key ] && source /home/woori/.openai_key
export PYTHONPATH=src:$T2
export T2_MAX_MODEL_LEN=131072
cd "$TB" || exit 1
pop() { local t; exec 9>"$LOCK"; flock 9
  t=$(head -1 "$QUEUE" 2>/dev/null); [ -n "$t" ] && sed -i '1d' "$QUEUE"
  flock -u 9; exec 9>&-; echo "$t"; }
while true; do
  T=$(pop); [ -z "$T" ] && { echo "[$LANE $(date '+%H:%M')] 큐 소진 — 종료"; break; }
  TAG="bank_x806_base_nt4_$T"
  echo "[$LANE $(date '+%H:%M')] → $T (잔여 $(wc -l < "$QUEUE"))"
  rm -rf "$TB/data/simulations/$TAG"
  $PY $T2/t2_run_gated.py --gate 0 --domain banking_knowledge \
    --retrieval_config alltools --agent_model "$M" --agent_base "http://$AHOST:$PORT/v1" \
    --user_llm openrouter/openai/gpt-5.2 --user_temp 0.0 --user_reasoning_effort low \
    --task_ids "$T" --num_trials 4 --max_concurrency 4 --max_steps 200 \
    --save_to "$TAG" > /home/woori/scratch/logs/${TAG}_drv.log 2>&1 || echo "[$LANE] FAIL $T"
done
date; echo ${LANE}_DONE
