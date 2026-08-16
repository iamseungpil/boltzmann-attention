#!/bin/bash
# t7304 (tag j) — **배달 자리를 결정 자리로** (2026-08-16·S1 재설계 리뷰 반영판).
#   설계서 = reports/facet_rft_2026/S1_REDESIGN_T7304_2026_08_16.md (§0b 리뷰 검증·§2 처치·§4 판정)
#
# ⚠tag i 는 **사용자 판정 "발사 보류"로 중단**했다(30분·부분 데이터). 분석 금지.
#
# 왜 자리를 옮기나(t7303 전수 실측·8/8):
#   기존 배달 자리(SEARCH_ON_PROCEED)는 turn 2·6 인데 손님이 요구를 진술하는 것은 msg 3(checking)·
#   msg 23~69(savings) — **배달이 요구보다 먼저 끝난다**(간격 중앙 29.5). 재료는 한 턴만 사니
#   결정 순간엔 없다. 반면 *선택을 담은 write 시도*는 요구가 다 나온 뒤이고 모델이 값을 쓰겠다고
#   나선 순간이다. 그 자리는 A2 가 이미 선언한다(choice_grounding.tool · recommendation_verify.action_tool).
#
# 처치 = `T2_DOCS_AT_WRITE=1` 하나. 예산(3) 불변 — 이른 자리를 비우고 write 자리로 **옮긴다**.
#   객체는 문서 본문(중앙 스위치 decide=False). 선택은 끝까지 모델(write 1턴 유예 후 모델이 다시 냄).
#
# ★판정(사전 고정)  ⓐ배선(부착·hold 발화) → ⓑ축별 gold 일치(055 합산 0~16·**GO=+5**) →
#   ⓒ기전 분류(부착 생성이 tool_call-only 인가) → ⓓreward → ⓔ부작용(CWE·skip·098 불변·오군 교차표)
#   → ⓕGO 시에만 wave-2(무내용 통제).
#
# 편성: 055 nt=8(축 종점 검정력) + 024·098 nt=4 = 16 sim/팔 = 32 sim.

set -e
REPO=/home/woori/workspace_common/boltzmann-attention-pi
cd "$REPO/scripts/distill/tau2"

LOG=/home/woori/scratch/logs
SIMS=/home/woori/scratch/tau2-bench/data/simulations
mkdir -p "$LOG"

SHA=$(cd "$REPO" && git rev-parse --short HEAD)
DIRTY=$(cd "$REPO" && git status --porcelain -- \
  scripts/distill/tau2/t2_gate_patch.py scripts/distill/tau2/t2_search.py \
  scripts/distill/tau2/t2_resolve.py scripts/distill/tau2/a2/ | grep -cv '^??' || true)
if [ "$DIRTY" != "0" ]; then
  echo "[t7304j] REFUSING: 엔진 경로 커밋 안 된 변경 $DIRTY 개." >&2; exit 1
fi

for t in test_docs_at_write.py test_proceed_docbody.py test_cp2_clobber.py test_no_unbound_a2.py \
         test_deliver_precommit.py test_material_reserve.py test_material_bypass.py \
         test_probe_canonical.py test_log_join.py test_now_selfcall.py \
         test_no_undefined_names.py test_decision_carry.py test_subcall_return_type.py \
         test_a2_three_layer.py test_operator_find.py test_route_trace.py; do
  PYTHONPATH=/home/woori/scratch/tau2-bench/src /home/woori/venvs/seka_env/bin/python "$t" \
    >/dev/null 2>&1 || { echo "[t7304j] REFUSING: $t FAIL" >&2; exit 1; }
done
echo "[t7304j] VERIFY OK (16검)"

# ★양성 통제: 판정 계기가 t7303 기지값을 재현해야 발사한다(계기 결함을 런 뒤에 발견하는 형 방지).
/home/woori/venvs/seka_env/bin/python x341_docbody_verdict.py --selftest > /tmp/x341_st.txt 2>&1 \
  || { echo "[t7304j] REFUSING: x341 selftest 크래시" >&2; exit 1; }
grep -q "ctl   big부착 sim 0/12" /tmp/x341_st.txt \
  || { echo "[t7304j] REFUSING: x341 양성통제 불일치(ctl big부착≠0)" >&2; exit 1; }
grep -q "ctl   task_024  2/4" /tmp/x341_st.txt \
  || { echo "[t7304j] REFUSING: x341 양성통제 불일치(ctl 024 write gold≠2/4)" >&2; exit 1; }
echo "[t7304j] x341 양성통제 OK (ctl big부착 0/12 · 024 2/4)"

# ⚠자기-매치 방지: 브래킷 트릭. 좀비 대기루프(while pgrep …)는 프로브가 아니므로 제외한다.
if pgrep -f "[t]2_gap\.py" >/dev/null || pgrep -f "[x]3[0-9][0-9]_.*\.py" >/dev/null; then
  echo "[t7304j] REFUSING: 무료 프로브 실행 중(양 포트 필요)" >&2; exit 1
fi

launch () {
  NAME="$1"; PORT="$2"; DOCW="$3"; TASKS="$4"; NT="$5"; SUF="$6"
  TAG="bank_t7304_${NAME}${SUF}_20260816j"
  if [ -e "$LOG/${TAG}.log" ]; then echo "[t7304j] SKIP: ${TAG}.log 존재" >&2; return 0; fi
  if [ -e "$SIMS/${TAG}" ]; then echo "[t7304j] REFUSING: $SIMS/${TAG} 잔존" >&2; return 1; fi
  echo "{\"tag\":\"$TAG\",\"scaffold_sha\":\"$SHA\",\"tasks\":\"$TASKS\",\"port\":$PORT,\"nt\":$NT,\"docs_at_write\":\"$DOCW\",\"why\":\"S1 redesign: docs at the choice-encoding write site\"}" \
    | tee "$LOG/${TAG}.meta.json"
  setsid bash -c "cd '$REPO/scripts/distill/tau2' && source ./go_stack.sh >/dev/null 2>&1 && \
    export T2_ACTION_SUB=1 T2_KEEP_DENY_BODY=1 T2_CALL_FORM=1 T2_ARG_EMPTY=1 \
           T2_SEARCH_AGENT=1 T2_DECIDE_ANY=1 \
           T2_WRITE_ARG_ENUM=1 T2_DECIDE_BEFORE_WRITE=1 T2_DECISION_CARRY=1 \
           T2_DISCOVERY_STEP2=1 T2_ARG_AXIS=1 T2_WRITE_SUB=3 T2_ACTION_INDEX=1 \
           T2_NOW_SELFCALL=1 T2_SEARCH_ON_PROCEED=1 && \
    export T2_ACT_DEMAND=0 T2_DELIVER_PRECOMMIT=0 T2_PROCEED_DOCBODY=0 T2_DOCS_AT_WRITE=$DOCW && \
    t2_launch $TAG $PORT '$TASKS' $NT && \
    export T2_ACT_DEMAND=0 T2_DELIVER_PRECOMMIT=0 T2_PROCEED_DOCBODY=0 T2_DOCS_AT_WRITE=$DOCW && \
    t2_launch bank_t7304_${NAME}aux_20260816j $PORT 'task_024,task_098' 4" \
    </dev/null >"$LOG/${TAG}.log" 2>&1 &
  echo "[t7304j] $NAME(docs_at_write=$DOCW) → PID=$! port=$PORT · 055 nt=8 후 024/098 nt=4"
}

# 각 포트에서 055(nt=8) 를 먼저 돌고, 끝나면 같은 프로세스가 024/098(nt=4) 를 이어 돈다.
launch ctl   8140 0 task_055 8 ""
launch treat 8141 1 task_055 8 ""
echo "[t7304j] 기동 완료 · sha=$SHA · 팔당 16 sim(055×8 + 024×4 + 098×4)"
