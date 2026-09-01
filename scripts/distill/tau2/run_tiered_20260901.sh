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
# ★시작 티어 (2026-09-01): 앞 티어를 이미 돌렸으면 거기서부터 잇는다 — 티어 경계가 정지점이자
#   **재개점**이다. 예) `START_TIER=2 run_tiered_20260901.sh A viewscale_max`
START_TIER=${START_TIER:-1}
# ★SPLIT=1 (2026-09-01): 두 GPU 가 **같은 팔로 일감을 나눈다**(대조가 아니라 처리량).
#   ⚠파는 것: 같은 sha 반대 팔이 없어지므로 이후 Δ 는 **인과가 아니라 관측**이다([[57]]·U-5 서식).
#   기준선은 밤샘런(ctl 설정·다른 sha)과 T1 의 ctl 착지분으로만 남는다.
SPLIT=${SPLIT:-0}
# ★태그는 **발사마다 유일**해야 한다 (2026-09-01 사고): 같은 태그의 `results.json` 이 있으면
#   tau2 가 *"resume? (y/n)"* 을 **대화형으로 묻고**, nohup 은 stdin 이 없어 `EOFError` 로 즉사한다.
#   그리고 그 죽음이 rc=0 으로 보이면 드라이버가 다음 티어로 넘어간다(위 두 사고가 겹쳤다).
STAMP=${STAMP:-$(date +%Y%m%d_%H%M)}
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
  [ "$TIER" -lt "$START_TIER" ] && continue
  eval IDS=\$T$TIER
  if [ "$SPLIT" = "1" ]; then
    # 티어 목록을 짝/홀로 갈라 A 는 짝번째, B 는 홀번째를 맡는다(티어 순서·우선순위는 보존).
    _n=0; _mine=""
    for _t in $(echo "$IDS" | tr "," " "); do
      _n=$((_n+1)); _p=$((_n % 2))
      if { [ "$SIDE" = "A" ] && [ "$_p" = "1" ]; } || { [ "$SIDE" = "B" ] && [ "$_p" = "0" ]; }; then
        _mine="${_mine:+$_mine,}$_t"
      fi
    done
    IDS="$_mine"
    [ -n "$IDS" ] || { echo "[tiered] tier=$TIER 이 쪽 몫 없음 — 건너뜀"; continue; }
  fi
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
