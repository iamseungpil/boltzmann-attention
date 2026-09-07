#!/usr/bin/env bash
# rep3 = rep2(2차 파동) + **LB7 재료 배달 하나만** (2026-09-08 · 사용자 승인)
#   repo_rep3 는 repo_rep2 의 복제 + t2_gate_patch.py 한 군데(33줄) 삽입 = 단일 변수.
#   근거: x829 격리 — 재료 없음 0/8 · 우리 군 이름 0/8 · 그 군의 문서 제목 47건 **8/8**.
#   ⚠T2_KB_DOCS_DIR 은 어디에도 설정돼 있지 않았다(그래서 T2_REQUIRE_DOC_DELIVER 도 rep2 에서 0회).
#     여기서 명시한다 — 이것도 rep2 와의 차이이므로 판독 시 함께 계상한다.
#   ⛔rep1(클라우드 9143·repo_rep1)은 건드리지 않는다. 트리·큐·출력·엔진 전부 별개다.
set -u
PORT="${1:-8141}"; QUEUE=/home/woori/scratch/x768/q_rep3.txt; LOCK="$QUEUE.lock"
R=/home/woori/scratch/repo_rep3
T2="$R/scripts/distill/tau2"; GO=/home/woori/scratch/tau2-bench
OUT=/home/woori/scratch/x768/out_rep3; mkdir -p "$OUT" /home/woori/scratch/logs
M="Qwen/Qwen3.8-27B-FP8"
export T2_KB_DOCS_DIR=/home/woori/scratch/tau2-bench/data/tau2/domains/banking_knowledge/documents
GOT=$(curl -s -m 20 "http://localhost:$PORT/v1/models" | grep -oE '"id":"[^"]+"' | head -1 | cut -d'"' -f4)
[ "$GOT" = "$M" ] || { echo "[rep3] 중단 - 서빙 $GOT"; exit 1; }
echo "[rep3 $(date '+%m-%d %H:%M')] 시작 :$PORT 서빙=$GOT 큐=$(wc -l < $QUEUE) REPO=$R DOCS=$T2_KB_DOCS_DIR"
pop(){ local t; exec 9>"$LOCK"; flock 9
  t=$(head -1 "$QUEUE" 2>/dev/null); [ -n "$t" ] && sed -i '1d' "$QUEUE"
  flock -u 9; exec 9>&-; echo "$t"; }
cd "$T2" || exit 1
while true; do
  T=$(pop); [ -z "$T" ] && { echo "[rep3] 큐 소진"; break; }
  TAG="rep3_$T"; T0=$(date +%s)
  echo "[rep3 $(date '+%m-%d %H:%M')] -> $T (잔여 $(wc -l < "$QUEUE"))"
  rm -rf "$GO/data/simulations/$TAG" /home/woori/iso_tau3/tau2-bench/data/simulations/"$TAG"
  REPO="$R" GO_TAU2="$GO" T2_AGENT_HOST=localhost \
    T2_KB_DOCS_DIR="$T2_KB_DOCS_DIR" \
    T2_FB_SIDECAR="/home/woori/scratch/logs/fb_${TAG}.jsonl" T2_FB_SIDECAR_TEXT=1 \
    bash ./run_ours_task.sh --arm viewmax2 --trials 4 --concurrency 1 "$TAG" "$PORT" "$T" \
    > "/home/woori/scratch/logs/${TAG}_drv.log" 2>&1 || echo "  [rep3] FAIL $T"
  EL=$(( $(date +%s) - T0 ))
  if [ "$EL" -lt 120 ]; then
    exec 9>"$LOCK"; flock 9; { echo "$T"; cat "$QUEUE"; } > "$QUEUE.tmp" && mv "$QUEUE.tmp" "$QUEUE"
    flock -u 9; exec 9>&-
    echo "  ⛔[rep3] $T 이 ${EL}초 만에 끝났다 = 하네스 고장. 큐 복구 후 정지."
    tail -6 "/home/woori/scratch/logs/${TAG}_drv.log"; exit 3
  fi
  echo "  [rep3] LB7 발화 $(grep -c 'T2_DEGEN_TITLES' "/home/woori/scratch/logs/${TAG}_drv.log" 2>/dev/null)회"
  [ -f "/home/woori/scratch/logs/fb_${TAG}.jsonl" ] && gzip -c "/home/woori/scratch/logs/fb_${TAG}.jsonl" > "$OUT/fb_${TAG}.jsonl.gz"
  [ -f "/home/woori/scratch/logs/${TAG}_drv.log" ] && gzip -c "/home/woori/scratch/logs/${TAG}_drv.log" > "$OUT/${TAG}_drv.log.gz"
  echo "  [rep3] 영속 $TAG"
done
date; echo rep3_DONE
