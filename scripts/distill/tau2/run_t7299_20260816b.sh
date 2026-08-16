#!/bin/bash
# t7299 — **`T2_MATERIAL_RESERVE` 좁힌 A/B**(2026-08-16·C498 수리의 1차 종점 검정).
#
# 무엇을 재나. C498: 배달은 그 턴의 **재생성 버퍼**에만 붙고(비커밋·C298) `state.messages` 에
#   남지 않는다. t7298 의 055 는 sim 당 예산 3회를 **`대화텍스트 1`**(손님이 요구를 말하기도 전)
#   부터 소진했고, 궤적 재료 표지 **0건**, 선택 **0/4** ↔ 같은 재료로 격리는 **24/24**.
#   ⇒ 처치 = 초반 자리 배달을 **1회로 묶고** 남은 예산을 결정 자리에 남긴다(총량 3 불변).
#
# ★**1차 종점은 성적이 아니다**([[62]] 규칙4·오늘 신설). 사전 고정 순서:
#   ⓐ배선   treat 에서 `[T2_SEARCH_AGENT] 일반 자리 배달 …(예약 on)` 발화 > 0 · ctl 0
#   ⓑ**1차** **결정 자리 배달 수** — `[T2_SEARCH_ON_PROCEED] … 재료 배달` 이 sim 당 몇 번인가.
#            ctl 은 예산을 초반에 다 써서 여기서 0 에 가까워야 한다. 이것이 안 오르면 **중단**.
#   ⓒ선택   gold 계좌 클래스를 실제로 연 sim(=t7298 의 ⓑ 지표·055 는 t7298 에서 0/4)
#   ⓓ성적   `reward`/`db_match` 로만(C486)
#   ⓔ부작용 over-action(`ONLY-PRED`)·지연(재료가 결정점에 실리면 토큰이 는다·t7296 은 1.8× 였다)
#            · **098 불변**(t7295 3/3 · t7297 5/5 · t7298 4/4)
#   ⚠nt=4 · 잡음 바닥 ±4(C483) ⇒ ⓒⓓ 차이는 인용 금지. 이 런은 **ⓑ 를 사러 간다**.
#
# 편성: `055`(격리 24/24 ↔ 라이브 0/4 인 바로 그 자리) + `098`(거동 불변 대조) × nt=4 × 2팔 = 16 sim.
#   070·071 은 여기 없다 — ⓑ 가 열리면 그때 전체 편성(t7300)으로 간다. 열리지 않으면 **안 간다**.

set -e
REPO=/home/woori/workspace_common/boltzmann-attention-pi
cd "$REPO/scripts/distill/tau2"

NT=4
TASKS=task_055,task_098
LOG=/home/woori/scratch/logs
SIMS=/home/woori/scratch/tau2-bench/data/simulations
mkdir -p "$LOG"

SHA=$(cd "$REPO" && git rev-parse --short HEAD)
DIRTY=$(cd "$REPO" && git status --porcelain -- \
  scripts/distill/tau2/t2_gate_patch.py scripts/distill/tau2/t2_search.py \
  scripts/distill/tau2/t2_resolve.py scripts/distill/tau2/t2_compute.py \
  scripts/distill/tau2/t2_scaffold_get.py scripts/distill/tau2/a2/ | grep -cv '^??' || true)
if [ "$DIRTY" != "0" ]; then
  echo "[t7299] REFUSING: 엔진 경로 커밋 안 된 변경 $DIRTY 개." >&2; exit 1
fi

for t in test_material_reserve.py test_material_bypass.py test_probe_canonical.py \
         test_now_selfcall.py test_provenance_nested.py test_no_undefined_names.py \
         test_decision_carry.py test_subcall_return_type.py test_a2_three_layer.py \
         test_operator_find.py test_route_trace.py; do
  PYTHONPATH=/home/woori/scratch/tau2-bench/src /home/woori/venvs/seka_env/bin/python "$t" \
    >/dev/null 2>&1 || { echo "[t7299] REFUSING: $t FAIL" >&2; exit 1; }
done
echo "[t7299] VERIFY OK"

if pgrep -f "[t]2_gap.py" >/dev/null || pgrep -f "[x]33[0-9]_" >/dev/null; then
  echo "[t7299] REFUSING: 무료 프로브가 8141 을 쓰는 중(팔을 동시에 못 띄운다)" >&2; exit 1
fi

launch () {
  NAME="$1"; PORT="$2"; RESERVE="$3"
  TAG="bank_t7299_${NAME}_20260816b"
  if [ -e "$LOG/${TAG}.log" ]; then echo "[t7299] SKIP: ${TAG}.log 존재" >&2; return 0; fi
  if [ -e "$SIMS/${TAG}" ]; then echo "[t7299] REFUSING: $SIMS/${TAG} 잔존" >&2; return 1; fi
  if ps -eo cmd | grep -v grep | grep "t2_run_gated.py" | grep -q "localhost:${PORT}/"; then
    echo "[t7299] REFUSING: 포트 ${PORT} 사용 중" >&2; return 1
  fi
  echo "{\"tag\":\"$TAG\",\"scaffold_sha\":\"$SHA\",\"tasks\":\"$TASKS\",\"port\":$PORT,\"nt\":$NT,\"reserve\":\"$RESERVE\",\"why\":\"C498 primary endpoint = material present at decision\"}" \
    | tee "$LOG/${TAG}.meta.json"
  setsid bash -c "cd '$REPO/scripts/distill/tau2' && source ./go_stack.sh >/dev/null 2>&1 && \
    export T2_ACTION_SUB=1 T2_KEEP_DENY_BODY=1 T2_CALL_FORM=1 T2_ARG_EMPTY=1 \
           T2_SEARCH_AGENT=1 T2_DECIDE_ANY=1 \
           T2_WRITE_ARG_ENUM=1 T2_DECIDE_BEFORE_WRITE=1 T2_DECISION_CARRY=1 \
           T2_DISCOVERY_STEP2=1 T2_ARG_AXIS=1 T2_WRITE_SUB=3 T2_ACTION_INDEX=1 \
           T2_NOW_SELFCALL=1 T2_SEARCH_ON_PROCEED=1 && \
    export T2_ACT_DEMAND=0 T2_MATERIAL_RESERVE=$RESERVE && \
    t2_launch $TAG $PORT '$TASKS' $NT" </dev/null >"$LOG/${TAG}.log" 2>&1 &
  echo "[t7299] $NAME(reserve=$RESERVE) → PID=$! port=$PORT"
}

launch ctl   8140 0
launch treat 8141 1
echo "[t7299] 기동 완료 · sha=$SHA · nt=$NT · tasks=$TASKS"
