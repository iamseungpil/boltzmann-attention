#!/usr/bin/env bash
# t2_persist_logs — 라이브 런의 **로그 영속** 상설 스위퍼 (2026-08-19·E-MFIX 항목)
#
# 왜: 런처마다 `results.json.gz` 만 영속하고 `$LOG/<tag>.log` 는 리모트에만 남았다.
#     그래서 **stderr 가 유일 신호인 레버**(T2_ARG_SCHEMA·T2_GROUND·T2_TOOLGATE …)의
#     "0 발화" 를 사후에 확인할 수 없었다 — 원장 C549ⓔ · t7326 실물.
#     `t2_launch` 안에서 gzip 하면 **자기 로그를 자기가 압축**하는 꼴이라(호출자 리다이렉트가
#     아직 열려 있다) 꼬리가 잘린다. ⇒ 런 종료 후 도는 **별도 스위퍼**로 둔다.
#
# 규칙: results.json 이 있는데 log.gz 가 없는 태그만 회수한다(멱등·덮어쓰지 않는다).
# 사용: bash t2_persist_logs.sh            # 회수
#       bash t2_persist_logs.sh --check    # 미영속 목록만 인쇄(회수 안 함)
set -e
R=/home/woori/workspace_common/boltzmann-attention-pi
L=/home/woori/scratch/logs
S=/home/woori/scratch/tau2-bench/data/simulations
P=$R/reports/facet_rft_2026/sim_results
CHECK=0; [ "$1" = "--check" ] && CHECK=1

n=0; miss=0
for d in "$S"/bank_*; do
  t=$(basename "$d")
  [ -f "$d/results.json" ] || continue
  [ -f "$P/$t.log.gz" ] && continue
  if [ ! -f "$L/$t.log" ]; then
    echo "  ⚠로그 자체가 없다: $t"; miss=$((miss+1)); continue
  fi
  if [ "$CHECK" = 1 ]; then
    echo "  미영속: $t ($(stat -c%s "$L/$t.log") bytes)"
  else
    gzip -c "$L/$t.log" > "$P/$t.log.gz"
    echo "  회수: $t"
  fi
  n=$((n+1))
done
echo "[t2_persist_logs] 대상 $n · 로그 부재 $miss · check=$CHECK"
