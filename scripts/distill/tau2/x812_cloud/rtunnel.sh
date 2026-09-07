#!/usr/bin/env bash
# 역터널 (2026-09-07): 하네스는 사내(.153)에서 돌고 vLLM 만 여기(클라우드)다.
#   .153 의 localhost:9141 → 여기 8141 (GPU0) · 9143 → 8143 (GPU1)
#   ⚠방향이 역인 이유: vast 가 /root/.ssh/authorized_keys 를 계정 키로 주기적으로 덮어써서
#     .153→클라우드 방향 키 등록이 유지되지 않는다(실측: 추가 직후 사라짐).
K=/root/.ssh/id_to153
while true; do
  ssh -N -i "$K" -o StrictHostKeyChecking=no -o ExitOnForwardFailure=yes \
      -o ServerAliveInterval=20 -o ServerAliveCountMax=3 \
      -R 9141:localhost:8141 -R 9143:localhost:8143 \
      woori@61.33.35.153
  echo "[rtunnel $(date "+%m-%d %H:%M")] 끊김 — 5초 후 재연결"
  sleep 5
done
