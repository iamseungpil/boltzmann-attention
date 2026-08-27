#!/usr/bin/env bash
# 밤샘 큐 — 한 GPU 에 런을 **순서대로** 걸고, 매 단계 진행을 남긴다 (2026-08-28).
#
# ## 왜 있나
#   GPU 가 둘이 되면서 각 장치가 자기 줄을 갖는다. 앞 런의 PID 가 죽기를 기다렸다 다음을 쏜다.
#   ⚠[[30]] *"진행률 가시 의무"* — 오늘 배터리 조사에서 집계를 루프 **끝에만** 찍어 진행이 하나도
#     안 보였다. 여기서는 **매 항목 시작·끝에 한 줄씩** 즉시 남긴다(`tail -f` 로 읽힌다).
#
# ## 사용
#   WAIT_PID=<앞 런 PID · 없으면 0> QUEUE_NAME=gpu0 bash night_queue.sh <cmd1> [cmd2 ...]
#   각 cmd 는 `bash -c` 로 실행되는 한 줄 문자열이다.
#
# ## 중단 규칙
#   항목이 0 이 아닌 코드로 끝나도 **다음 항목으로 간다** — 밤을 통째로 버리지 않기 위해서다.
#   다만 **연속 두 항목이 결과 없이** 끝나면 멈춘다(배터리가 붉으면 뒤도 전부 붉다).
set -o pipefail
Q="${QUEUE_NAME:-queue}"
W="${WAIT_PID:-0}"
say() { echo "[$Q $(date +%H:%M:%S)] $*"; }

if [ "$W" -gt 0 ] 2>/dev/null; then
  say "앞 런(PID $W)이 끝나기를 기다린다"
  n=0
  while kill -0 "$W" 2>/dev/null; do
    sleep 60; n=$((n+1))
    [ $((n % 10)) -eq 0 ] && say "  … 대기 ${n}분"
  done
  say "앞 런 종료 확인 — 큐 시작"
else
  say "대기 없음 — 즉시 시작"
fi

i=0; dry=0
for cmd in "$@"; do
  i=$((i+1))
  say "▶ [$i/$#] 시작: $cmd"
  bash -c "$cmd"
  rc=$?
  say "◀ [$i/$#] 종료 exit=$rc"
  # 결과가 실제로 내려왔나 — 태그를 명령줄에서 뽑아 확인한다(판단 0·존재 확인뿐).
  tg=$(echo "$cmd" | grep -o "TAG=[A-Za-z0-9_]*" | head -1 | cut -d= -f2)
  got=0
  if [ -n "$tg" ]; then
    [ -s "/home/woori/scratch/tau2-bench/data/simulations/$tg/results.json" ] && got=1
    say "   결과 파일: $([ $got -eq 1 ] && echo 있음 || echo 없음) ($tg)"
  else
    got=1                      # 태그 없는 항목(프로브 등)은 이 규칙에서 뺀다
  fi
  if [ $got -eq 0 ]; then
    dry=$((dry+1))
    if [ $dry -ge 2 ]; then say "⛔연속 2 항목이 결과 없이 끝났다 — 큐를 멈춘다"; break; fi
  else
    dry=0
  fi
done
say "큐 종료 ($i 항목)"
