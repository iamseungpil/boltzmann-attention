#!/usr/bin/env bash
# ── 밤샘 전수 커버리지 (2026-09-01 밤 → 09-02 아침 회수)
#
# 사용자 지시: "A축에 너무 많이 몰려있다. A·B축 남은 태스크들 gpu 1,2 에 내일 아침에 같이
#   결과 볼 수 있게 배치하라."
#
# 무엇을 담나 (총 32):
#   · A큐 잔여 21 (한 번도 안 돈 것)
#   · B큐 잔여 8  (061·080·066·083 / 022·091·033·092)
#   · **수리 검증 재런 3** (046·048·049) — 오늘 넣은 `skip_when_tokens`(정책 Step 2 스킵)이
#     실제로 그 세 태스크의 과잉 로깅을 없애는지 본다. 부호표는 ⊖0 · ⊕37 이었다.
#
# 균형: 장시간 두 개를 **서로 다른 GPU** 에 갈라 놓는다(074 ≈6h · 092 ≈2.9h). 나머지는 16/16.
# 팔: 두 GPU 모두 **t3prime 단일 팔** — 커버리지 런이므로 팔을 섞지 않는다([[54]]).
#   ⚠`viewmax2` 는 여기 넣지 않는다. 그건 **짝 A/B 전용**이고 섞으면 둘 다 못 읽는다.
# 프로필: 오늘 올린 TRUNC 수리가 적용된다(PROBE 2048 · JUDGE 16384) ⇒ **TRUNC 0 이 게이트 하나**다.
#
# 사용: bash run_night_20260902.sh [1|2]     (인자 없으면 둘 다 — 2번은 8143 이 비면 시작)
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
TS="$(date +%Y%m%d_%H%M)"

G1_TASKS="task_074,task_034,task_060,task_099,task_071,task_037,task_020,task_008,task_077,task_078,task_056,task_097,task_046,task_048,task_049,task_016"
G2_TASKS="task_092,task_027,task_075,task_096,task_041,task_087,task_090,task_086,task_085,task_061,task_080,task_066,task_083,task_022,task_091,task_033"

launch () {                       # $1=port  $2=tag  $3=tasks
  local port="$1" tag="$2" tasks="$3"
  echo "[night] $tag  port=$port  tasks=$(echo "$tasks" | tr ',' '\n' | wc -l)"
  cd "$HERE" || exit 1
  nohup bash ./run_ours_task.sh --trials 1 --concurrency 4 --arm t3prime \
        "$tag" "$port" "$tasks" > "/home/woori/scratch/logs/${tag}_driver.log" 2>&1 &
  sleep 5
}

wait_port_free () {               # $1=port — 그 포트를 쓰는 t2_run_gated 가 사라질 때까지
  local port="$1" n=0
  while ps -eo args | grep -q "[t]2_run_gated.*localhost:${port}"; do
    n=$((n+1)); [ $((n % 30)) -eq 1 ] && echo "[night] 8${port#8} 대기중 ... ${n}0s"
    sleep 10
  done
}

WHICH="${1:-both}"

if [ "$WHICH" = "1" ] || [ "$WHICH" = "both" ]; then
  launch 8141 "bank_night1_t3prime_${TS}" "$G1_TASKS"
fi

if [ "$WHICH" = "2" ] || [ "$WHICH" = "both" ]; then
  ( wait_port_free 8143
    launch 8143 "bank_night2_t3prime_${TS}" "$G2_TASKS" ) &
fi

sleep 3
echo "[night] 발사 완료 — 로그: /home/woori/scratch/logs/bank_night*_${TS}*.log"
