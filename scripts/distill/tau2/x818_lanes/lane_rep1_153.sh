#!/usr/bin/env bash
# rep1 (2026-09-07 전환판) — 하네스는 **사내**, vLLM 은 역터널 너머 **클라우드**.
#   수리본 트리 = /home/woori/scratch/repo_rep1 (1차 파동: F1 P12 P2 P9 P1 P6 P11 + E-PLAN 정정)
#   ⚠사내에서 도니 srt/shell 이 정상이다 — sandbox 폴백 불필요([[54]] 팔 동일).
set -u
PORT="${1:-9143}"; QUEUE=/home/woori/scratch/x768/q_crep1.txt; LOCK="$QUEUE.lock"
R=/home/woori/scratch/repo_rep1
T2="$R/scripts/distill/tau2"; GO=/home/woori/iso_tau3/tau2-bench
OUT=/home/woori/scratch/x768/out_rep1; mkdir -p "$OUT" /home/woori/scratch/logs
M="Qwen/Qwen3.8-27B-FP8"
GOT=$(curl -s -m 20 "http://localhost:$PORT/v1/models" | grep -oE '"id":"[^"]+"' | head -1 | cut -d'"' -f4)
[ "$GOT" = "$M" ] || { echo "[rep1] 중단 - 서빙 $GOT"; exit 1; }
echo "[rep1 $(date '+%m-%d %H:%M')] 시작 :$PORT 서빙=$GOT 큐=$(wc -l < $QUEUE) REPO=$R"
pop(){ local t; exec 9>"$LOCK"; flock 9
  t=$(head -1 "$QUEUE" 2>/dev/null); [ -n "$t" ] && sed -i '1d' "$QUEUE"
  flock -u 9; exec 9>&-; echo "$t"; }
cd "$T2" || exit 1
while true; do
  T=$(pop); [ -z "$T" ] && { echo "[rep1] 큐 소진"; break; }
  TAG="rep1_$T"; T0=$(date +%s)
  echo "[rep1 $(date '+%m-%d %H:%M')] -> $T (잔여 $(wc -l < "$QUEUE"))"
  rm -rf "$GO/data/simulations/$TAG"
  REPO="$R" GO_TAU2="$GO" T2_AGENT_HOST=localhost \
    T2_FB_SIDECAR="/home/woori/scratch/logs/fb_${TAG}.jsonl" T2_FB_SIDECAR_TEXT=1 \
    bash ./run_ours_task.sh --arm viewmax2 --trials 4 --concurrency 4 "$TAG" "$PORT" "$T" \
    > "/home/woori/scratch/logs/${TAG}_drv.log" 2>&1 || echo "  [rep1] FAIL $T"
  EL=$(( $(date +%s) - T0 ))
  if [ "$EL" -lt 120 ]; then
    exec 9>"$LOCK"; flock 9; { echo "$T"; cat "$QUEUE"; } > "$QUEUE.tmp" && mv "$QUEUE.tmp" "$QUEUE"
    flock -u 9; exec 9>&-
    echo "  ⛔[rep1] $T 이 ${EL}초 만에 끝났다 = 하네스 고장. 큐 복구 후 정지."
    tail -5 "/home/woori/scratch/logs/${TAG}_drv.log"; exit 3
  fi
  d="$GO/data/simulations/$TAG"
  [ -f "$d/results.json" ] && gzip -c "$d/results.json" > "$OUT/$TAG.results.json.gz"
  [ -f "/home/woori/scratch/logs/${TAG}_drv.log" ] && gzip -c "/home/woori/scratch/logs/${TAG}_drv.log" > "$OUT/${TAG}_drv.log.gz"
  [ -f "/home/woori/scratch/logs/fb_${TAG}.jsonl" ] && gzip -c "/home/woori/scratch/logs/fb_${TAG}.jsonl" > "$OUT/fb_${TAG}.jsonl.gz"
  echo "  [rep1] 영속 $TAG"
done
date; echo rep1_DONE
