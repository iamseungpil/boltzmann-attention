#!/usr/bin/env bash
# ★티어 순차 · 2 GPU 균등 — 2026-09-01 (사용자 지시: *"순서도 고려해 중간에 멈추게 하되,
#   GPU 2개에 고루 배분해 동시에 끝나게 하라"*).
#
# 왜 이 모양인가
#   ⑴ **순서는 인자로 통제되지 않는다** — `tau2/runner/helpers.py:76-82` 가 도메인 파일 순서를
#      순회하며 멤버십으로만 거른다(실측: `task_093,…` 로 줘도 010·036·039 부터 돌았다).
#      ⇒ 우선순위는 **런을 쪼개는 것**으로만 보장된다(벤치 소스는 안 고친다·[[54]]).
#   ⑵ 그래서 티어를 **순차 런**으로 돌린다. 티어 경계가 곧 **안전한 정지점**이다.
#   ⑶ 두 포트(8141=GPU0 · 8143=GPU1)는 **각 티어를 반씩** 나눠 가진다 ⇒ 한쪽만 먼저 끝나
#      노는 일이 없다(직전 4:24 배분의 실제 결함).
#
# 사용: run_tiered_20260901.sh A   (포트 8141)  ·  run_tiered_20260901.sh B   (포트 8143)
set -o pipefail
SIDE=${1:?A 또는 B}
STAMP=${STAMP:-20260901}
case "$SIDE" in
  A) PORT=8141 ;;
  B) PORT=8143 ;;
  *) echo "REFUSING: side=$SIDE (A|B)"; exit 1 ;;
esac

# 티어 = 우선순위. 앞 티어가 끝나야 다음 티어가 돈다.
#  T1 표적 판정   — §T-1a(격리 덮어쓰기·093) · 회귀감시(094) · §S-1 `_json`(062·065)
#  T2 전손·후보   — §S-2 전손 재시작(039·040·095) · 후보집합(010)
#  T3 스텝 소진   — max_steps 6건(036·067·068·063·069·082)
#  T4 장시간      — 074(6.0h)·092(2.9h) 를 **서로 다른 GPU** 로 가른다
#  T5 나머지
if [ "$SIDE" = "A" ]; then
  T1="task_093,task_094"
  T2="task_039,task_095"
  T3="task_036,task_068,task_082"
  T4="task_092,task_101"
  T5="task_046,task_060,task_066,task_080,task_084,task_096,task_099,task_026,task_041"
else
  T1="task_062,task_065"
  T2="task_040,task_010"
  T3="task_067,task_069,task_063"
  T4="task_074"
  T5="task_048,task_061,task_078,task_085,task_097,task_020,task_027,task_029,task_083"
fi

HERE="$(cd "$(dirname "$0")" && pwd)"
LOG=${LOG:-/home/woori/scratch/logs}
for TIER in 1 2 3 4 5; do
  eval IDS=\$T$TIER
  TAG="bank_x72${TIER}_t${TIER}${SIDE}_${STAMP}"
  echo "=================================================================="
  echo "[tiered] $(date '+%F %T') side=$SIDE port=$PORT tier=$TIER tag=$TAG ids=$IDS"
  bash "$HERE/run_ours_task.sh" --trials 1 --concurrency 4 "$TAG" "$PORT" "$IDS" \
    >> "$LOG/tiered_${SIDE}_${STAMP}.log" 2>&1
  rc=$?
  echo "[tiered] tier=$TIER rc=$rc"
  # ⛔티어가 실패하면 멈춘다 — 다음 티어로 넘어가면 무엇이 원인인지 못 가른다([[08]]).
  [ "$rc" = "0" ] || { echo "[tiered] REFUSING to continue (tier $TIER rc=$rc)"; exit "$rc"; }
done
echo "[tiered] $(date '+%F %T') side=$SIDE 전 티어 완료"
