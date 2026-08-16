#!/bin/bash
# t7303 — **커밋-이전 전달 인과 실험**(2026-08-16·사용자 지시 "1,2 순"·특허 B §B2-6 실시예).
#
# 가설. 재료가 **모델이 첫 확정을 말하기 전**에 도착하면 선택 적중이 오른다.
#   근거(상관·기전, 인과 아님): 055 첫 지목 msg 7~15 뒤 배달 3회인데 이름 불변 · 024 첫 지목 msg 4 ·
#   gold 문서가 msg 7 검색 1위로 왔는데 불변 · 019 격리 24/24 ↔ 라이브 0 · 재료는 **한 턴만 산다**
#   (재생성 버퍼·비커밋·C498). 선행은 조기 확정을 **탐지**만 하고(2606.22936) 주입 시점을 제어하지 않는다.
#
# 처치 = `T2_DELIVER_PRECOMMIT=1` 하나. sim 당 1회, 문서군이 형식화되는 **가장 이른 턴**에 배달.
#   ★2026-08-16 수리: 선-배달은 **문서 본문**(`decide=False`)을 나른다. 그 전 판(tag g)은 서브의
#   **결정**(243~263자)을 날랐는데, 격리에서 24/24 를 만든 객체는 문서 51k자였다 — 다른 것이다.
#   결정을 나르면 [[62]]③(엔진이 답을 주면 측정 대상이 사라진다) 자리에 선다.
#   예산 총량(3)·군 선택(LLM)·최종 선택(모델) **전부 불변** — 옮긴 것은 **시점 하나**.
#
# ★판정(사전 고정·이 순서로·[[62]] 규칙4)
#   ⓐ배선   treat 에서 `[T2_DELIVER_PRECOMMIT] 선-배달` > 0 · ctl **0**. 실패면 성적 안 읽는다.
#   ⓑ**1차 종점 = 첫 지목 이전 도달 sim 비율** — `t2_forensic.first_named()` 로 첫 지목 index,
#            로그 `SEARCH_AGENT … turn=N` 으로 배달 턴. **이 지표가 안 오르면 처치 실패**로 종결하고
#            성적을 인용하지 않는다(오늘 t7299 가 이 규율로 t7300 을 막았다).
#   ⓒ선택   gold 계좌/카드 클래스를 실제로 연 sim(055·024)
#   ⓓ성적   `reward`/`db_match` 만(C486)
#   ⓔ부작용 **군 오선택**(요구를 다 듣기 전에 배달 = 이 처치의 대가) · 지연 · over-action ·
#            **098 불변**(t7295 3/3·t7297 5/5·t7298 4/4·t7299 3~4/4)
#   ⚠nt=4·팔당 12 sim·잡음 바닥 ±4 ⇒ ⓒⓓ 차이 인용 금지. 이 런은 **ⓑ 를 사러 간다**.
#
# 편성: 055(첫 지목이 늦다) + 024(첫 지목이 **msg 4**로 가장 이르다·gold 액션 1개) + 098(불변 대조).

set -e
REPO=/home/woori/workspace_common/boltzmann-attention-pi
cd "$REPO/scripts/distill/tau2"

NT=4
TASKS=task_055,task_024,task_098
LOG=/home/woori/scratch/logs
SIMS=/home/woori/scratch/tau2-bench/data/simulations
mkdir -p "$LOG"

SHA=$(cd "$REPO" && git rev-parse --short HEAD)
DIRTY=$(cd "$REPO" && git status --porcelain -- \
  scripts/distill/tau2/t2_gate_patch.py scripts/distill/tau2/t2_search.py \
  scripts/distill/tau2/t2_resolve.py scripts/distill/tau2/a2/ | grep -cv '^??' || true)
if [ "$DIRTY" != "0" ]; then
  echo "[t7303] REFUSING: 엔진 경로 커밋 안 된 변경 $DIRTY 개." >&2; exit 1
fi

for t in test_no_unbound_a2.py test_deliver_precommit.py test_material_reserve.py test_material_bypass.py \
         test_probe_canonical.py test_log_join.py test_now_selfcall.py \
         test_no_undefined_names.py test_decision_carry.py test_subcall_return_type.py \
         test_a2_three_layer.py test_operator_find.py test_route_trace.py; do
  PYTHONPATH=/home/woori/scratch/tau2-bench/src /home/woori/venvs/seka_env/bin/python "$t" \
    >/dev/null 2>&1 || { echo "[t7303] REFUSING: $t FAIL" >&2; exit 1; }
done
echo "[t7303] VERIFY OK"

if pgrep -f "[t]2_gap\.py" >/dev/null || pgrep -f "[x]33[0-9]_" >/dev/null; then
  echo "[t7303] REFUSING: 무료 프로브가 8141 을 쓰는 중" >&2; exit 1
fi

launch () {
  NAME="$1"; PORT="$2"; PRE="$3"
  TAG="bank_t7303_${NAME}_20260816h"
  if [ -e "$LOG/${TAG}.log" ]; then echo "[t7303] SKIP: ${TAG}.log 존재" >&2; return 0; fi
  if [ -e "$SIMS/${TAG}" ]; then echo "[t7303] REFUSING: $SIMS/${TAG} 잔존" >&2; return 1; fi
  if ps -eo cmd | grep -v grep | grep "t2_run_gated.py" | grep -q "localhost:${PORT}/"; then
    echo "[t7303] REFUSING: 포트 ${PORT} 사용 중" >&2; return 1
  fi
  echo "{\"tag\":\"$TAG\",\"scaffold_sha\":\"$SHA\",\"tasks\":\"$TASKS\",\"port\":$PORT,\"nt\":$NT,\"precommit\":\"$PRE\",\"why\":\"causal test of commit-before-delivery (patent B B2-6 embodiment)\"}" \
    | tee "$LOG/${TAG}.meta.json"
  setsid bash -c "cd '$REPO/scripts/distill/tau2' && source ./go_stack.sh >/dev/null 2>&1 && \
    export T2_ACTION_SUB=1 T2_KEEP_DENY_BODY=1 T2_CALL_FORM=1 T2_ARG_EMPTY=1 \
           T2_SEARCH_AGENT=1 T2_DECIDE_ANY=1 \
           T2_WRITE_ARG_ENUM=1 T2_DECIDE_BEFORE_WRITE=1 T2_DECISION_CARRY=1 \
           T2_DISCOVERY_STEP2=1 T2_ARG_AXIS=1 T2_WRITE_SUB=3 T2_ACTION_INDEX=1 \
           T2_NOW_SELFCALL=1 T2_SEARCH_ON_PROCEED=1 && \
    export T2_ACT_DEMAND=0 T2_DELIVER_PRECOMMIT=$PRE && \
    t2_launch $TAG $PORT '$TASKS' $NT" </dev/null >"$LOG/${TAG}.log" 2>&1 &
  echo "[t7303] $NAME(precommit=$PRE) → PID=$! port=$PORT"
}

launch ctl   8140 0
launch treat 8141 1
echo "[t7303] 기동 완료 · sha=$SHA · nt=$NT · tasks=$TASKS"
