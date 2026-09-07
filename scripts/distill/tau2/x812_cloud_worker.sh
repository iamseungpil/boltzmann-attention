#!/usr/bin/env bash
# x812_cloud_worker — base(--gate 0) nt=4 큐 워커 · **태스크마다 영속**.
#   정본 `t2_base_worker.sh` 에서 파생. 차이는 하나 — ⛔인스턴스가 언제든 죽으므로
#   태스크가 끝날 때마다 결과·로그·사이드카를 gzip → git add -f → push → tracked 확인.
#   근거: [[30]] 실측 사고("sim 결과는 gitignored → 복구불가" · "24 sim 이 전부 untracked").
# 사용: x812_cloud_worker.sh <레인> <PORT> <큐파일>
set -u
LANE="$1"; PORT="$2"; QUEUE="$3"
BASE="$HOME"; TB="$BASE/tau2-bench"; REPO="$BASE/boltzmann-attention-pi"
T2="$REPO/scripts/distill/tau2"; PY="$TB/venv/bin/python"; M="Qwen/Qwen3.8-27B-FP8"
LOCK="${QUEUE}.lock"; SIMR="$REPO/reports/facet_rft_2026/sim_results"
mkdir -p "$SIMR" "$BASE/logs"

GOT=$(curl -s -m 10 "http://localhost:$PORT/v1/models" | grep -oE '"id":"[^"]+"' | head -1 | cut -d'"' -f4)
[ "$GOT" = "$M" ] || { echo "[$LANE] 중단 — 서빙 모델 $GOT"; exit 1; }
echo "[$LANE] 서빙 = $GOT"
source "$BASE/.openrouter_key"
export PYTHONPATH="src:$T2" T2_MAX_MODEL_LEN=131072
cd "$TB" || exit 1

pop(){ local t; exec 9>"$LOCK"; flock 9
  t=$(head -1 "$QUEUE" 2>/dev/null); [ -n "$t" ] && sed -i '1d' "$QUEUE"
  flock -u 9; exec 9>&-; echo "$t"; }

persist(){                      # $1=TAG
  local tag="$1" gz="$SIMR/$1.results.json.gz" n=0
  [ -f "$TB/data/simulations/$tag/results.json" ] || { echo "  ⛔결과 없음 $tag"; return 1; }
  gzip -c "$TB/data/simulations/$tag/results.json" > "$gz"
  cp -f "$BASE/logs/${tag}_drv.log" "$SIMR/${tag}_drv.log" 2>/dev/null || true
  cp -f "$BASE/logs/fb_${tag}.jsonl" "$SIMR/fb_${tag}.jsonl" 2>/dev/null || true
  cd "$REPO" || return 1
  git add -f "$gz" "$SIMR/${tag}_drv.log" "$SIMR/fb_${tag}.jsonl" 2>/dev/null
  git -c user.email=cloud@local -c user.name=cloud commit -q -m "results: $tag" 2>/dev/null
  while [ $n -lt 3 ]; do git push -q 2>/dev/null && break; n=$((n+1)); sleep 20
    git -c user.email=cloud@local -c user.name=cloud pull -q --rebase 2>/dev/null; done
  git ls-files --error-unmatch "$gz" >/dev/null 2>&1 \
    && echo "  ✅영속 $tag" || echo "  ⛔영속 실패 $tag (tracked 아님)"
  cd "$TB"
}

while true; do
  T=$(pop); [ -z "$T" ] && { echo "[$LANE $(date '+%m-%d %H:%M')] 큐 소진 — 종료"; break; }
  TAG="bank_x806_base_nt4_$T"
  echo "[$LANE $(date '+%m-%d %H:%M')] → $T (잔여 $(wc -l < "$QUEUE"))"
  rm -rf "$TB/data/simulations/$TAG"
  $PY "$T2/t2_run_gated.py" --gate 0 --domain banking_knowledge \
    --retrieval_config alltools --agent_model "$M" --agent_base "http://localhost:$PORT/v1" \
    --user_llm openrouter/openai/gpt-5.2 --user_temp 0.0 --user_reasoning_effort low \
    --task_ids "$T" --num_trials 4 --max_concurrency 4 --max_steps 200 \
    --save_to "$TAG" > "$BASE/logs/${TAG}_drv.log" 2>&1 || echo "[$LANE] FAIL $T"
  persist "$TAG"
done
date; echo "${LANE}_DONE"
