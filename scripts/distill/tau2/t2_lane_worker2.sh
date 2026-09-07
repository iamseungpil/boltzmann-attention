#!/usr/bin/env bash
# t2_lane_worker — 큐에서 태스크를 하나씩 꺼내 도는 레인 워커 (2026-09-05)
#
# 사용: t2_lane_worker2.sh <레인> <AGENT_HOST> <PORT> <큐파일[:큐파일2...]> <SEED> <태그접두>
#
# ── v2 변경(2026-09-06) ───────────────────────────────────────────────────
#   ① 큐 여러 개를 `:` 로 받아 **앞의 것부터** 소진한다. 이걸로 «conc2 먼저, 끝나면 conc1»
#     2단계 배치를 워커 재발사 없이 구현한다(사용자 축자: *"eviction 을 최소화하게 2개
#     돌릴거는 먼저 concurrency 2로 먼저 배치하고, 끝나면 concurrency 1 개로 돌릴 걸 돌려라"*).
#   ② 평일 주간 게이트를 `lane1` 정확일치 → `lane1*` 접두일치로 넓힌다. 한 GPU 에 워커 둘을
#     붙여도 **GPU 는 하나**이므로 사용자 제약("일과중에는 1개만")을 어기지 않는다.
#     이유: 한 태스크가 막혀도 그 GPU 가 놀지 않게([[83]] Σ컨텍스트 ≤ kv_cache_size_tokens).
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
LANE="$1"; AHOST="$2"; PORT="$3"; QUEUES="$4"; SEED="$5"; PREFIX="$6"
[ -z "$PREFIX" ] && { echo "사용: $0 <레인> <HOST> <PORT> <큐> <SEED> <태그접두>"; exit 1; }

REPO=/home/woori/workspace_common/boltzmann-attention-pi
LOG=/home/woori/scratch/logs
# ★락은 **첫 큐 파일 기준**으로 잡는다. v1 워커가 `${QUEUE}.lock` 을 쓰므로 같은 큐를
#   공유할 때 v1/v2 가 **같은 락**을 잡아야 한다. 다른 이름을 쓰면 두 워커가 같은 줄을
#   동시에 집는다(원자적 pop 이 깨진다).
LOCK="${QUEUES%%:*}.lock"
cd "$REPO/scripts/distill/tau2" || exit 1

allowed() {
  local d h
  d=$(date +%u); h=$(date +%H); h=${h#0}; [ -z "$h" ] && h=0
  if [ "$d" -ge 1 ] && [ "$d" -le 5 ] && [ "$h" -ge 8 ] && [ "$h" -lt 19 ]; then
    case "$LANE" in lane1*) return 0 ;; *) return 1 ;; esac   # 평일 주간 = lane1* 만(GPU 0)
  fi
  return 0                                            # 주말·야간 = 전부
}

# 큐를 앞에서부터 훑어 **첫 비지 않은 큐**에서 원자적으로 pop 한다.
# 반환은 "큐파일<TAB>태스크". 전부 비면 빈 문자열.
pop() {
  local q t
  exec 9>"$LOCK"
  flock 9
  for q in $(echo "$QUEUES" | tr ":" " "); do
    t=$(head -1 "$q" 2>/dev/null)
    if [ -n "$t" ]; then sed -i "1d" "$q"; printf "%s	%s" "$q" "$t"; break; fi
  done
  flock -u 9; exec 9>&-
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
  POPPED=$(pop)
  [ -z "$POPPED" ] && { echo "[$LANE $(date '+%m-%d %H:%M')] 전 큐 소진 — 종료"; break; }
  CURQ="${POPPED%%	*}"; T="${POPPED##*	}"
  TAG="${PREFIX}_${T}"
  echo "[$LANE $(date '+%m-%d %H:%M')] → $T (tag=$TAG · 큐=$(basename "$CURQ"))"
  # ★SEED=DEFAULT 면 `--seed` 를 **안 넘긴다**. tau2 는 기저 seed 로 시행별 seed 를 파생한다
  #   (`runner/batch.py:517-518` 축자: `random.seed(config.seed)` / `seeds = [random.randint(0,1000000) ...]`).
  #   기본 300 이 첫 시행에 626729 를 준다 — 이미 돈 49 sim 이 그 값이므로 패스1 의 잔여는
  #   기본을 그대로 써야 **같은 패스**가 된다. `--seed 626729` 를 주면 파생이 달라져 다른 패스가 된다.
  SEEDARG=""
  [ "$SEED" != "DEFAULT" ] && SEEDARG="--seed $SEED"
  T2_AGENT_HOST="$AHOST" bash ./run_ours_task.sh --arm viewmax2 --concurrency 1 --trials 1 \
      $SEEDARG "$TAG" "$PORT" "$T" > "$LOG/${TAG}_driver.log" 2>&1
  RC=$?
  RW=$(/home/woori/iso_tau3/venv/bin/python -c "
import json,sys
try:
    r=json.load(open('/home/woori/scratch/tau2-bench/data/simulations/$TAG/results.json'))
    ss=r.get('simulations') or []
    print('%.1f' % ((ss[0].get('reward_info') or {}).get('reward') or 0.0) if ss else 'nosim')
except Exception as e: print('?')
" 2>/dev/null)
  REM=0
  for q in $(echo "$QUEUES" | tr ":" " "); do REM=$((REM + $(wc -l < "$q" 2>/dev/null || echo 0))); done
  echo "[$LANE $(date '+%m-%d %H:%M')] ← $T rc=$RC reward=$RW · 전체 큐잔여 $REM"
done
