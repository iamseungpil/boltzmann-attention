#!/usr/bin/env bash
# rep2 스모크가 끝나면 x829 격리 프로브를 8141 에 붙인다 (사용자 지시 2026-09-08).
set -u
LANE_PAT="lane_rep2_153.sh"
echo "[chain $(date '+%m-%d %H:%M')] rep2 종료 대기 시작"
while true; do
  pid=$(ps -eo pid,args | grep -F "$LANE_PAT" | grep -v grep | awk '{print $1}' | head -1)
  [ -z "$pid" ] && break
  sleep 60
done
echo "[chain $(date '+%m-%d %H:%M')] rep2 종료 확인 — 큐 잔여 $(wc -l < /home/woori/scratch/x768/q_rep2.txt)"
got=$(curl -s -m 20 http://localhost:8141/v1/models | grep -oE '"id":"[^"]+"' | head -1 | cut -d'"' -f4)
if [ "$got" != "Qwen/Qwen3.8-27B-FP8" ]; then
  echo "[chain] 중단 — 8141 서빙 '$got' ([[30]] 모델 id 대조 실패)"; exit 1
fi
echo "[chain] 8141 서빙=$got — x829 발사"
X829_BASE=http://localhost:8141/v1 X829_N=8 \
  /home/woori/venvs/seka_env/bin/python -u /home/woori/scratch/x829_probe.py \
  > /home/woori/scratch/logs/x829_out.log 2>&1
echo "[chain $(date '+%m-%d %H:%M')] x829 종료 rc=$?"
tail -12 /home/woori/scratch/logs/x829_out.log
