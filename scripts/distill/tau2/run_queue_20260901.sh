#!/usr/bin/env bash
# ★소요순 큐 — base 실패 48 전수 커버 (2026-09-01·사용자 선택 ⓑ)
#
# 사용: run_queue_20260901.sh <A|B> [arm]
#
# 왜 이 모양인가
#  ⑴ **커버리지 목적**이다. 종전 35개 집합은 *수리 검증용*이라 base 실패 48 중 30만 겹쳤다
#     (실측: base 착지 78 · 실패 48 · 교집합 30 · 미포함 18). 그 18을 넣어야 *"base 실패 N개
#     회복"* 을 말할 수 있다([[54]] 분모).
#  ⑵ **빠른 것부터**(사용자 지시). 순서는 base 실측 소요 오름차순이고, 두 GPU 에 **번갈아** 배분해
#     양쪽이 비슷하게 끝난다. 074(360분)·092(174분)는 **맨 뒤**에 둔다 — 밤에 도는 큐이므로
#     긴 것을 버릴 이유가 없고, 앞의 짧은 것들이 먼저 착지해 아침에 볼 것이 남는다.
#  ⑶ tau2 는 `--task_ids` 순서를 무시하므로(helpers.py:76-82) **4개씩 끊어 순차 런**으로만
#     순서를 보장한다. 배치 경계가 정지점이다.
#  ⑷ 실패한 배치에서 멈춘다 — 원인을 못 가르게 되므로([[08]]).
set -o pipefail
SIDE=${1:?A 또는 B}
ARM=${2:-t3prime}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M)}
case "$SIDE" in
  A) PORT=${PORT_A:-8141} ;;
  B) PORT=${PORT_B:-8143} ;;
  *) echo "REFUSING: side=$SIDE"; exit 1 ;;
esac

# base 실측 소요 오름차순 → 홀수번째=A · 짝수번째=B (085·091 은 base 데이터 없음 → 맨 뒤)
A_LIST="task_034,task_060,task_099,task_071,task_037,task_020,task_008,task_077,task_078,task_056,task_097,task_027,task_075,task_096,task_041,task_087,task_090,task_086,task_085,task_074"
B_LIST="task_046,task_012,task_088,task_048,task_053,task_101,task_049,task_084,task_026,task_102,task_081,task_029,task_061,task_080,task_066,task_083,task_022,task_091,task_092"

if [ "$SIDE" = "A" ]; then LIST="$A_LIST"; else LIST="$B_LIST"; fi
HERE="$(cd "$(dirname "$0")" && pwd)"
LOG=${LOG:-/home/woori/scratch/logs}

i=0; batch=""; n=0
for t in $(echo "$LIST" | tr "," " "); do
  batch="${batch:+$batch,}$t"; n=$((n+1))
  if [ "$n" = "4" ]; then
    i=$((i+1)); TAG="bank_x73${i}_q${SIDE}_${ARM}_${STAMP}"
    echo "=================================================================="
    echo "[queue] $(date '+%F %T') side=$SIDE port=$PORT arm=$ARM batch=$i tag=$TAG ids=$batch"
    bash "$HERE/run_ours_task.sh" --trials 1 --concurrency 4 --arm "$ARM" "$TAG" "$PORT" "$batch" \
      >> "$LOG/queue_${SIDE}_${STAMP}.log" 2>&1
    rc=$?; echo "[queue] batch=$i rc=$rc"
    [ "$rc" = "0" ] || { echo "[queue] REFUSING to continue (batch $i rc=$rc)"; exit "$rc"; }
    batch=""; n=0
  fi
done
if [ -n "$batch" ]; then
  i=$((i+1)); TAG="bank_x73${i}_q${SIDE}_${ARM}_${STAMP}"
  echo "[queue] $(date '+%F %T') side=$SIDE port=$PORT arm=$ARM batch=$i(마지막) tag=$TAG ids=$batch"
  bash "$HERE/run_ours_task.sh" --trials 1 --concurrency 4 --arm "$ARM" "$TAG" "$PORT" "$batch" \
    >> "$LOG/queue_${SIDE}_${STAMP}.log" 2>&1
  echo "[queue] batch=$i rc=$?"
fi
echo "[queue] $(date '+%F %T') side=$SIDE 전 배치 완료"
