#!/usr/bin/env bash
# ★티어 순차 · 2 GPU 동시 대조 — 2026-09-01
#
# 사용: run_tiered_20260901.sh <A|B> [arm]
#   예) A viewscale   (GPU0/8141 = 새 문턱)   ·   B ctl   (GPU1/8143 = 종전 상수)
#
# 왜 이 모양인가
#  ⑴ **순서는 인자로 통제되지 않는다** — `tau2/runner/helpers.py:76-82` 가 도메인 파일 순서를
#     순회하며 멤버십으로만 거른다(실측: `task_093,…` 로 줘도 010·036·039 부터 돌았다).
#     ⇒ 우선순위는 **런을 쪼개는 것**으로만 보장된다(벤치 소스는 안 고친다·[[54]]).
#  ⑵ 티어 = 우선순위이고 **티어 경계가 안전한 정지점**이다(앞 티어가 끝나야 다음이 돈다).
#  ⑶ 두 GPU 는 **같은 태스크를 서로 다른 팔로** 돈다. 종전판은 태스크를 반씩 갈랐는데 그러면
#     두 GPU 의 결과가 다른 태스크라 **팔의 Δ 를 못 잰다**([[57]] 부정통제 부재).
#  ⑷ 모델에 매인 값은 전부 `model_profiles/<모델 id>.env` 가 갖는다 — 포트가 어느 모델을
#     서빙하든 런처가 **그 모델의 프로필을 골라** 싣는다. Q2.5 와 Q3.8 을 동시에 돌릴 수 있다.
set -o pipefail
SIDE=${1:?A 또는 B}
ARM=${2:-}
STAMP=${STAMP:-20260901}
case "$SIDE" in
  A) PORT=${PORT_A:-8141} ;;
  B) PORT=${PORT_B:-8143} ;;
  *) echo "REFUSING: side=$SIDE (A|B)"; exit 1 ;;
esac

#  T1 수리 표적 — §T-1a(격리 덮어쓰기·093) · 회귀감시(094) · §S-1 `_json`(062·065)
#  T2 스텝 소진 — §T-6 뷰-압축 문턱의 표적(base 51~81 메시지 ↔ ours 209~293 · shell 0~13 ↔ 88~163)
#  T3 전손·후보 — §S-2 전손 재시작(039·040·095) · 후보집합(010)
#  T4 장시간   — 074(6.0h) · 092(2.9h)
#  T5 나머지
T1="task_093,task_094,task_062,task_065"
T2="task_067,task_069,task_063,task_068,task_036,task_082"
T3="task_039,task_040,task_095,task_010"
T4="task_074,task_092"
T5="task_046,task_048,task_060,task_061,task_066,task_078,task_080,task_084,task_085,task_096,task_097,task_099,task_101,task_020,task_026,task_027,task_029,task_041,task_083"

HERE="$(cd "$(dirname "$0")" && pwd)"
LOG=${LOG:-/home/woori/scratch/logs}
SUF="${ARM:-noarm}"
ARMOPT=""
[ -n "$ARM" ] && ARMOPT="--arm $ARM"

for TIER in 1 2 3 4 5; do
  eval IDS=\$T$TIER
  TAG="bank_x72${TIER}_t${TIER}${SIDE}_${SUF}_${STAMP}"
  echo "=================================================================="
  echo "[tiered] $(date '+%F %T') side=$SIDE port=$PORT arm=${ARM:-없음} tier=$TIER tag=$TAG"
  echo "[tiered] ids=$IDS"
  # shellcheck disable=SC2086
  bash "$HERE/run_ours_task.sh" --trials 1 --concurrency 4 $ARMOPT "$TAG" "$PORT" "$IDS" \
    >> "$LOG/tiered_${SIDE}_${SUF}_${STAMP}.log" 2>&1
  rc=$?
  echo "[tiered] tier=$TIER rc=$rc"
  # ⛔티어가 실패하면 멈춘다 — 다음 티어로 넘어가면 무엇이 원인인지 못 가른다([[08]]).
  [ "$rc" = "0" ] || { echo "[tiered] REFUSING to continue (tier $TIER rc=$rc)"; exit "$rc"; }
done
echo "[tiered] $(date '+%F %T') side=$SIDE arm=${ARM:-없음} 전 티어 완료"
