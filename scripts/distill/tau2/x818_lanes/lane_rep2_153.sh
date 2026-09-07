#!/usr/bin/env bash
# rep2 스모크 게이트 — 2차 파동(F2 t2_resolve.py + F8''' t2_gate_patch.py) 단 둘.
#   수리본 트리 = /home/woori/scratch/repo_rep2 (커밋 8f1fde7a)
#   1차와 바이트 다른 파일은 t2_gate_patch.py · t2_resolve.py 뿐 — 나머지 3종 동일.
#   엔진 = 사내 8141(49GB). conc 1 (사용자 지시 2026-09-08) — 이 엔진에서 conc 4 는 capacity 대기 3 · preemption 3,419 로 직렬화된다.
set -u
PORT="${1:-8141}"; QUEUE=/home/woori/scratch/x768/q_rep2.txt; LOCK="$QUEUE.lock"
R=/home/woori/scratch/repo_rep2
T2="$R/scripts/distill/tau2"; GO=/home/woori/iso_tau3/tau2-bench
OUT=/home/woori/scratch/x768/out_rep2; mkdir -p "$OUT" /home/woori/scratch/logs
M="Qwen/Qwen3.8-27B-FP8"
GOT=$(curl -s -m 20 "http://localhost:$PORT/v1/models" | grep -oE '"id":"[^"]+"' | head -1 | cut -d'"' -f4)
[ "$GOT" = "$M" ] || { echo "[rep2] 중단 - 서빙 $GOT"; exit 1; }
echo "[rep2 $(date '+%m-%d %H:%M')] 시작 :$PORT 서빙=$GOT 큐=$(wc -l < $QUEUE) REPO=$R"
pop(){ local t; exec 9>"$LOCK"; flock 9
  t=$(head -1 "$QUEUE" 2>/dev/null); [ -n "$t" ] && sed -i '1d' "$QUEUE"
  flock -u 9; exec 9>&-; echo "$t"; }
cd "$T2" || exit 1
while true; do
  T=$(pop); [ -z "$T" ] && { echo "[rep2] 큐 소진"; break; }
  TAG="rep2_$T"; T0=$(date +%s)
  echo "[rep2 $(date '+%m-%d %H:%M')] -> $T (잔여 $(wc -l < "$QUEUE"))"
  rm -rf "$GO/data/simulations/$TAG" /home/woori/scratch/tau2-bench/data/simulations/"$TAG"
  REPO="$R" GO_TAU2="$GO" T2_AGENT_HOST=localhost \
    T2_FB_SIDECAR="/home/woori/scratch/logs/fb_${TAG}.jsonl" T2_FB_SIDECAR_TEXT=1 \
    bash ./run_ours_task.sh --arm viewmax2 --trials 4 --concurrency 1 "$TAG" "$PORT" "$T" \
    > "/home/woori/scratch/logs/${TAG}_drv.log" 2>&1 || echo "  [rep2] FAIL $T"
  EL=$(( $(date +%s) - T0 ))
  if [ "$EL" -lt 120 ]; then
    exec 9>"$LOCK"; flock 9; { echo "$T"; cat "$QUEUE"; } > "$QUEUE.tmp" && mv "$QUEUE.tmp" "$QUEUE"
    flock -u 9; exec 9>&-
    echo "  ⛔[rep2] $T 이 ${EL}초 만에 끝났다 = 하네스 고장. 큐 복구 후 정지."
    tail -5 "/home/woori/scratch/logs/${TAG}_drv.log"; exit 3
  fi
  d="$GO/data/simulations/$TAG"
  [ -f "$d/results.json" ] && gzip -c "$d/results.json" > "$OUT/$TAG.results.json.gz"
  [ -f "/home/woori/scratch/logs/${TAG}_drv.log" ] && gzip -c "/home/woori/scratch/logs/${TAG}_drv.log" > "$OUT/${TAG}_drv.log.gz"
  [ -f "/home/woori/scratch/logs/fb_${TAG}.jsonl" ] && gzip -c "/home/woori/scratch/logs/fb_${TAG}.jsonl" > "$OUT/fb_${TAG}.jsonl.gz"
  echo "  [rep2] 영속 $TAG"
done
date; echo rep2_DONE
