#!/usr/bin/env bash
# rep1 — 수리검증 레인(2026-09-07). **정본 `run_ours_task.sh --arm viewmax2` 를 그대로 부른다**
#   ([[67]] 사본 짜지 마라). 패스1(현재스택 기준선)이 쓴 것과 같은 팔이고, 차이는 nt=4 와
#   **오늘 적용한 수리 8건**뿐이다: F1 · P12 · P2 · P9 · P1 · P6 · P11 · E-PLAN 회귀 정정.
set -u
PORT="${1:-8143}"; QUEUE=/root/q_rep1.txt; LOCK="$QUEUE.lock"
R=/home/woori/workspace_common/boltzmann-attention-pi
T2="$R/scripts/distill/tau2"; GO=/home/woori/scratch/tau2-bench
OUT=/root/out; mkdir -p "$OUT" /root/logs
M="Qwen/Qwen3.8-27B-FP8"
GOT=$(curl -s -m 10 "http://localhost:$PORT/v1/models" | grep -oE '"id":"[^"]+"' | head -1 | cut -d'"' -f4)
[ "$GOT" = "$M" ] || { echo "[rep1] 중단 - 서빙 모델 $GOT"; exit 1; }
echo "[rep1 $(date '+%m-%d %H:%M')] 시작 :$PORT 서빙=$GOT 큐=$(wc -l < $QUEUE)"
pop(){ local t; exec 9>"$LOCK"; flock 9
  t=$(head -1 "$QUEUE" 2>/dev/null); [ -n "$t" ] && sed -i '1d' "$QUEUE"
  flock -u 9; exec 9>&-; echo "$t"; }
cd "$T2" || exit 1
while true; do
  T=$(pop); [ -z "$T" ] && { echo "[rep1] 큐 소진 - 종료"; break; }
  TAG="rep1_$T"
  echo "[rep1 $(date '+%m-%d %H:%M')] -> $T (잔여 $(wc -l < "$QUEUE"))"
  T0=$(date +%s)
  rm -rf "$GO/data/simulations/$TAG"
  T2_SANDBOX_FB_NOTE=1 TAU2_SANDBOX_FALLBACK=1 T2_FB_SIDECAR="/root/logs/fb_${TAG}.jsonl" T2_FB_SIDECAR_TEXT=1 \
    bash ./run_ours_task.sh --arm viewmax2 --trials 4 --concurrency 4 "$TAG" "$PORT" "$T" \
    > "/root/logs/${TAG}_drv.log" 2>&1 || echo "  [rep1] FAIL $T"
  # ★[[86]] 즉시-실패 가드 (2026-09-07 사고 재발 방지): 한 태스크가 120초 안에 끝나면
  #   그것은 결과가 아니라 **하네스 고장**이다(오늘 CRLF 로 35건이 수초 만에 드레인됐다).
  #   큐를 더 갉아먹지 않도록 **그 태스크를 되돌리고 레인을 세운다.**
  EL=$(( $(date +%s) - T0 ))
  if [ "$EL" -lt 120 ]; then
    exec 9>"$LOCK"; flock 9; { echo "$T"; cat "$QUEUE"; } > "$QUEUE.tmp" && mv "$QUEUE.tmp" "$QUEUE"
    flock -u 9; exec 9>&-
    echo "  ⛔[rep1] $T 이 ${EL}초 만에 끝났다 = 하네스 고장. 큐에 되돌리고 레인 정지."
    tail -5 "/root/logs/${TAG}_drv.log"
    exit 3
  fi
  d="$GO/data/simulations/$TAG"
  [ -f "$d/results.json" ] && gzip -c "$d/results.json" > "$OUT/$TAG.results.json.gz"
  [ -f "/root/logs/${TAG}_drv.log" ] && gzip -c "/root/logs/${TAG}_drv.log" > "$OUT/${TAG}_drv.log.gz"
  [ -f "/root/logs/fb_${TAG}.jsonl" ] && gzip -c "/root/logs/fb_${TAG}.jsonl" > "$OUT/fb_${TAG}.jsonl.gz"
  echo "  [rep1] 영속 $TAG ($(ls -la "$OUT/$TAG.results.json.gz" 2>/dev/null | awk '{print $5}')B)"
done
date; echo rep1_DONE
