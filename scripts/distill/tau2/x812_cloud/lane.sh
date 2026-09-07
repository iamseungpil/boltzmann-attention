#!/usr/bin/env bash
# x812 클라우드 레인 워커 — 큐 공유(flock) · 태스크마다 영속.
# 사용: lane.sh <레인> <포트> <큐파일>
set -u
LANE="$1"; PORT="$2"; QUEUE="$3"
LOCK="${QUEUE}.lock"; OUT=~/out; mkdir -p "$OUT" ~/logs
source ~/.openrouter_key; [ -f ~/.openai_key ] && source ~/.openai_key
export TAU2_SANDBOX_FALLBACK=1  # 2026-09-07 비특권 컨테이너: bwrap 불가 → sh -c 폴백(등가 검증 11/12)
export PYTHONPATH="src:$HOME/t2/tau2" T2_MAX_MODEL_LEN=131072 HF_HOME=/workspace/.hf_home
M="Qwen/Qwen3.8-27B-FP8"
GOT=$(curl -s -m 10 "http://localhost:$PORT/v1/models" | grep -oE "\"id\":\"[^\"]+\"" | head -1 | cut -d\" -f4)
[ "$GOT" = "$M" ] || { echo "[$LANE] 중단 — 서빙 모델 $GOT"; exit 1; }
echo "[$LANE] 서빙 = $GOT (:$PORT)"
cd ~/tau2-bench || exit 1
pop(){ local t; exec 9>"$LOCK"; flock 9
  t=$(head -1 "$QUEUE" 2>/dev/null); [ -n "$t" ] && sed -i "1d" "$QUEUE"
  flock -u 9; exec 9>&-; echo "$t"; }
persist(){ local tag="$1"
  local d=~/tau2-bench/data/simulations/$tag
  [ -f "$d/results.json" ] && gzip -c "$d/results.json" > "$OUT/$tag.results.json.gz"
  [ -f ~/logs/${tag}_drv.log ] && gzip -c ~/logs/${tag}_drv.log > "$OUT/${tag}_drv.log.gz"
  [ -f ~/logs/fb_${tag}.jsonl ] && gzip -c ~/logs/fb_${tag}.jsonl > "$OUT/fb_${tag}.jsonl.gz"
  echo "  [$LANE] 영속 $tag ($(ls -la "$OUT/$tag.results.json.gz" 2>/dev/null | awk "{print \$5}")B)"
}
while true; do
  T=$(pop); [ -z "$T" ] && { echo "[$LANE $(date +%m-%d\ %H:%M)] 큐 소진 — 종료"; break; }
  TAG="bank_x806_base_nt4_$T"
  echo "[$LANE $(date +%m-%d\ %H:%M)] → $T (잔여 $(wc -l < "$QUEUE"))"
  rm -rf ~/tau2-bench/data/simulations/$TAG
  ~/tau2-bench/venv/bin/python ~/t2/tau2/t2_run_gated.py --gate 0 --domain banking_knowledge \
    --retrieval_config alltools --agent_model "$M" --agent_base "http://localhost:$PORT/v1" \
    --user_llm openrouter/openai/gpt-5.2 --user_temp 0.0 --user_reasoning_effort low \
    --task_ids "$T" --num_trials 4 --max_concurrency 4 --max_steps 200 \
    --save_to "$TAG" > ~/logs/${TAG}_drv.log 2>&1 || echo "  [$LANE] FAIL $T"
  persist "$TAG"
done
date; echo "${LANE}_DONE"
