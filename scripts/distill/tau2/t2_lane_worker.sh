#!/usr/bin/env bash
# t2_lane_worker — 큐에서 태스크를 하나씩 꺼내 도는 레인 워커 (2026-09-05)
#
# 사용: t2_lane_worker.sh <레인> <AGENT_HOST> <PORT> <큐파일> <SEED> <태그접두>
#
# ── 왜 워커인가 (사용자 제약) ──────────────────────────────────────────────
#   축자: *"월요일 8시쯤에 153서버의 gpu 0 하나만 사용하게 해야 한다. 주말동안은 3개 gpu
#   쓰지만, 일과중에는 1개만 쓴다. 월~금 오전 8시 저녁 7시까지 1개 gpu 쓰는걸 가정해서
#   최대한 효율적으로 실험해야 한다."*
#   ⇒ 시각에 따라 레인 수가 바뀌어야 하는데, **도는 sim 을 죽이면 그 sim 이 통째로 버려진다**.
#     그래서 «죽이기»가 아니라 «다음 것을 안 집기»로 구현한다 — 워커는 태스크 **하나를 끝내고**
#     다음을 집기 전에 시계를 본다. 자연히 배수(drain)되고 잃는 일이 0이다.
#   ⛔[[30]] pkill -f 금지 — 이 설계는 애초에 kill 이 필요 없다.
#
# ── 스케줄 ────────────────────────────────────────────────────────────────
#   평일(월~금) 08:00~18:59  →  lane1 만 실행. lane2/lane3 은 대기.
#   그 외(주말·평일 야간)    →  전 레인 실행.
#
# ── 큐 ────────────────────────────────────────────────────────────────────
#   한 줄에 태스크 하나. `flock` 으로 원자적 pop 하므로 워커 여럿이 같은 큐를 공유해도
#   중복 실행이 없다. 큐가 비면 워커가 종료한다.
LANE="$1"; AHOST="$2"; PORT="$3"; QUEUE="$4"; SEED="$5"; PREFIX="$6"
[ -z "$PREFIX" ] && { echo "사용: $0 <레인> <HOST> <PORT> <큐> <SEED> <태그접두>"; exit 1; }

REPO=/home/woori/workspace_common/boltzmann-attention-pi
LOG=/home/woori/scratch/logs
LOCK="${QUEUE}.lock"
cd "$REPO/scripts/distill/tau2" || exit 1

allowed() {
  local d h
  d=$(date +%u); h=$(date +%H); h=${h#0}; [ -z "$h" ] && h=0
  if [ "$d" -ge 1 ] && [ "$d" -le 5 ] && [ "$h" -ge 8 ] && [ "$h" -lt 19 ]; then
    [ "$LANE" = "lane1" ] && return 0 || return 1     # 평일 주간 = lane1 만
  fi
  return 0                                            # 주말·야간 = 전부
}

pop() {
  local t
  exec 9>"$LOCK"
  flock 9
  t=$(head -1 "$QUEUE" 2>/dev/null)
  if [ -n "$t" ]; then sed -i '1d' "$QUEUE"; fi
  flock -u 9; exec 9>&-
  echo "$t"
}

echo "[$LANE $(date '+%m-%d %H:%M')] 워커 시작 host=$AHOST port=$PORT seed=$SEED 큐=$QUEUE"
# ★[[30]] 포트만으로 엔진을 식별하지 마라 — 발사 전 id 대조.
GOT=$(curl -s -m 10 "http://$AHOST:$PORT/v1/models" | grep -oE '"id":"[^"]+"' | head -1 | cut -d'"' -f4)
case "$GOT" in
  *Qwen3.8*) echo "[$LANE] 서빙 모델 = $GOT" ;;
  *) echo "[$LANE] 중단 — Q3.8 이 아니다([[79]]): $GOT"; exit 1 ;;
esac

WAITED=0
while true; do
  if ! allowed; then
    [ "$WAITED" = "0" ] && echo "[$LANE $(date '+%m-%d %H:%M')] 일과 시간 — 대기(다음 것을 집지 않는다)"
    WAITED=1; sleep 300; continue
  fi
  [ "$WAITED" = "1" ] && echo "[$LANE $(date '+%m-%d %H:%M')] 일과 종료 — 재개"
  WAITED=0
  T=$(pop)
  [ -z "$T" ] && { echo "[$LANE $(date '+%m-%d %H:%M')] 큐 소진 — 종료"; break; }
  TAG="${PREFIX}_${T}"
  echo "[$LANE $(date '+%m-%d %H:%M')] → $T (tag=$TAG)"
  T2_AGENT_HOST="$AHOST" bash ./run_ours_task.sh --arm viewmax2 --concurrency 1 --trials 1 \
      --seed "$SEED" "$TAG" "$PORT" "$T" > "$LOG/${TAG}_driver.log" 2>&1
  RC=$?
  RW=$(/home/woori/iso_tau3/venv/bin/python -c "
import json,sys
try:
    r=json.load(open('/home/woori/scratch/tau2-bench/data/simulations/$TAG/results.json'))
    ss=r.get('simulations') or []
    print('%.1f' % ((ss[0].get('reward_info') or {}).get('reward') or 0.0) if ss else 'nosim')
except Exception as e: print('?')
" 2>/dev/null)
  echo "[$LANE $(date '+%m-%d %H:%M')] ← $T rc=$RC reward=$RW · 큐잔여 $(wc -l < "$QUEUE" 2>/dev/null)"
done
