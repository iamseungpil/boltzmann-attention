#!/usr/bin/env bash
# x829 가 끝나면 rep2 를 A셀 6건으로 연장한다 (사용자 지시 2026-09-08 "8141 에도 아침까지 갈거").
#   순서 보장: 스모크 4건 -> x829 -> 이 연장분. q_rep2 는 x829 발사 뒤에만 건드린다.
#   고른 6건은 전부 base 4/4(A셀)이고 rep1(1차)도 도는 집합이라 1차<->2차가 짝이 맞는다.
#   task_004 를 맨 앞에: rep1 이 1 sim 잃었고 우리 [ACTION-REQUIRED] 가 gold 인 transfer 를
#   축자로 금지하는 자리다(t2_resolve.py:338).
set -u
Q=/home/woori/scratch/x768/q_rep2.txt; LOCK="$Q.lock"
OUT=/home/woori/scratch/logs/x829_out.log
echo "[chain2 $(date '+%m-%d %H:%M')] x829 완료 대기"
while true; do
  if [ -f "$OUT" ] && grep -q "== 결과 ==" "$OUT" 2>/dev/null; then break; fi
  if ps -eo args | grep -q "[c]hain_x829.sh" || ps -eo args | grep -q "[l]ane_rep2_153.sh"; then sleep 60; continue; fi
  # 체인도 레인도 없는데 결과 표식이 없다 = x829 가 실패했거나 안 돌았다
  echo "[chain2] ⛔x829 결과 표식 없음 — 연장 보류. 기록만 남긴다."; tail -5 "$OUT" 2>/dev/null; exit 3
done
echo "[chain2 $(date '+%m-%d %H:%M')] x829 완료 확인"
got=$(curl -s -m 20 http://localhost:8141/v1/models | grep -oE '"id":"[^"]+"' | head -1 | cut -d'"' -f4)
[ "$got" = "Qwen/Qwen3.8-27B-FP8" ] || { echo "[chain2] 중단 — 8141 서빙 '$got'"; exit 1; }
exec 9>"$LOCK"; flock 9
for t in task_004 task_024 task_023 task_002 task_006 task_008; do
  grep -qx "$t" "$Q" 2>/dev/null || echo "$t" >> "$Q"
done
flock -u 9; exec 9>&-
echo "[chain2] q_rep2 연장 -> $(tr '\n' ' ' < $Q)"
cd /home/woori/scratch
setsid nohup bash /home/woori/scratch/lane_rep2_153.sh 8141 >> /home/woori/scratch/logs/laneRep2.log 2>&1 < /dev/null &
sleep 20
echo "[chain2 $(date '+%m-%d %H:%M')] rep2 연장 레인 발사"
tail -3 /home/woori/scratch/logs/laneRep2.log
