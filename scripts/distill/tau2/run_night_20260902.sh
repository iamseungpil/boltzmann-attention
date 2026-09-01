#!/usr/bin/env bash
# ── 밤샘 전수 커버리지 (2026-09-01 밤 → 09-02 아침 회수) · nt=1
#
# 사용자 지시:
#   · "A축에 너무 많이 몰려있다. A·B축 남은 태스크들 gpu 1,2 에 내일 아침에 같이 결과 볼 수 있게 배치"
#   · "048 계열은 밤샘에서 모두 pass 하는지 넣어라"
#   · "pass 새로 재야 하는 태스크 모두 넣고 nt=1"
#   · "074 092 등은 **마지막에** 배치하라"
#
# ⚠tau2 는 `--task_ids` **순서를 무시한다**(`runner/helpers.py` 가 도메인 파일 순서로 멤버십 필터).
#   그래서 "마지막에"는 **별도 후속 런**으로만 실현된다 ⇒ GPU 당 2단(단거리 → 장거리).
#
# 담는 것 (총 41):
#   ⓐ A큐 잔여 21 · ⓑ B큐 잔여 8
#   ⓒ **048 계열 3**(046·048·049) — 오늘 넣은 `skip_when_tokens`(정책 Step 2 스킵) 검증.
#      코퍼스 부호표는 ⊖0 · ⊕37 이었고, 이 셋이 라이브에서 pass 로 가는지가 판정이다.
#   ⓓ **재측정 9**(010·062·063·065·067·068·093·094·095) — 오늘 **다른 팔**에서 통과한 것들이다.
#      단일 팔 수치가 없으면 pass율을 인용할 수 없다([[54]]) ⇒ t3prime 으로 다시 잰다.
#
# 팔: 두 GPU 모두 **t3prime 단일 팔**. ⚠`viewmax2` 는 넣지 않는다 — 짝 A/B 전용이라 섞으면 못 읽는다.
# 프로필: 오늘 올린 TRUNC 수리 적용(PROBE 2048 · JUDGE 16384) ⇒ **TRUNC 0 이 게이트 하나**다.
#
# 사용: bash run_night_20260902.sh [1|2]   (인자 없으면 둘 다 · 2번은 8143 이 비면 시작)
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
TS="$(date +%Y%m%d_%H%M)"
LOGD=/home/woori/scratch/logs

# 1단 = 단거리. 2단 = 장거리(마지막).
G1_SHORT="task_046,task_048,task_049,task_010,task_062,task_063,task_065,task_034,task_060,task_099,task_071,task_037,task_020,task_008,task_077,task_078,task_056,task_097,task_027,task_075"
G1_LONG="task_074"
G2_SHORT="task_067,task_068,task_093,task_094,task_095,task_096,task_041,task_087,task_090,task_086,task_085,task_016,task_061,task_080,task_066,task_083,task_022,task_091,task_033"
G2_LONG="task_092"

run_one () {                      # $1=port $2=tag $3=tasks  (동기 실행)
  local port="$1" tag="$2" tasks="$3"
  echo "[night] START $tag port=$port n=$(echo "$tasks" | tr ',' '\n' | wc -l) $(date +%H:%M)"
  cd "$HERE" || exit 1
  bash ./run_ours_task.sh --trials 1 --concurrency 4 --arm t3prime \
       "$tag" "$port" "$tasks" > "${LOGD}/${tag}_driver.log" 2>&1
  echo "[night] DONE  $tag rc=$? $(date +%H:%M)"
}

wait_port_free () {               # 그 포트를 쓰는 t2_run_gated 가 사라질 때까지
  local port="$1" n=0
  while ps -eo args | grep -q "[t]2_run_gated.*localhost:${port}"; do
    n=$((n+1)); [ $((n % 30)) -eq 1 ] && echo "[night] port ${port} 대기 ${n}0s"
    sleep 10
  done
}

gpu_lane () {                     # $1=port $2=이름 $3=단거리 $4=장거리
  wait_port_free "$1"
  run_one "$1" "bank_night$2a_t3prime_${TS}" "$3"
  run_one "$1" "bank_night$2b_long_${TS}"    "$4"
}

WHICH="${1:-both}"
[ "$WHICH" = "1" ] || [ "$WHICH" = "both" ] && ( gpu_lane 8141 1 "$G1_SHORT" "$G1_LONG" ) &
[ "$WHICH" = "2" ] || [ "$WHICH" = "both" ] && ( gpu_lane 8143 2 "$G2_SHORT" "$G2_LONG" ) &

sleep 3
echo "[night] 배치 완료 — 1단 20+19, 2단 074/092(마지막). 로그: ${LOGD}/bank_night*_${TS}*"
wait
