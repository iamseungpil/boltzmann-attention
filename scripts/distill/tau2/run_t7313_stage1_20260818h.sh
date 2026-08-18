#!/bin/bash
# t7313 — **1단계**(staged census · 16 태스크). 명부 = `reports/facet_rft_2026/S3_STAGED_ROSTERS_2026_08_18.json`.
#
# ## 위치
#
#   스모크 4(t7312·**통과**) → **1단계 16(여기)** → 2단계 30 → 3단계 45  = **95** (겹침 0·재실행 0)
#   · t7312 는 기능 종점을 **전부 통과**했고 새 결함이 없어 스택을 안 바꿨다 ⇒ 그 4 sim/팔은
#     **1단계 몫으로 인정**한다(명부 규칙). 그래서 여기는 나머지 **16**만 돈다.
#   · ⛔**스택 동결 시작점 = t7312 의 `sha 022b086c`**. 여기서 3단계 종료까지 엔진·A2·플래그
#     **불변**이라야 네 런을 합쳐 셀 수 있다. 각 런의 `scaffold_sha` 를 meta 에 남긴다 —
#     판정 때 **네 값이 같은지 먼저 대조**하고, 다르면 합산하지 않는다.
#
# ## ★1단계는 검문소다 (사용자 지시 2026-08-18)
#
#   1단계에서 문제가 많이 나오면 **고친 뒤 1단계를 처음부터 다시** 돈다(그때 스모크 4개도
#   되돌려 포함). 2·3단계는 1단계가 통과한 뒤에만 간다.
#
# ## 1단계가 무엇을 대표하는가 (명부 `stage1_coverage`)
#
#   · **W그룹**(재현 가능한 축·`TASK_GROUPS_W_2026_08_18.json`): 20 그룹 중 11 커버 —
#     **3개 이상인 9개 그룹은 전부** 포함(C466 의 10그룹은 데이터로 재현되지 않아 대표 라벨로만 병기)
#   · **레버·원인축**: L1 전달(055·024) · L2 제거(024) · **L3 검증/BYREF(074·072)** ·
#     L4 끝맺음(072·073·079) · L5 완결(093·094) · L6 계산(072·073) ·
#     선택축 VERDICT+ELIG 도달(055·057·063·024) · **어느 레버도 안 닿음 = 음성 대조(004·016)**
#   · ⚠**074 가 여기 있는 이유**: t7310·t7312 에서 `[T2_SG_BYREF]` 가 **양팔 0회** 였다.
#     C531 수리는 두 런 연속 **미시험**이고 074 가 마지막 기회다.
#
# ## ⚠사전 고정 판정 — **단계마다 전수 포렌식**(사용자 지시)
#
#   ⓐ**성적**: `reward` 만(C486). W그룹별·레버축별로 갠다. 이번엔 census 이므로 **읽는다**.
#   ⓑ**레버 발화율**: ELIG·VERDICT 를 **각자의 트리거 자리 대비**로. 이웃 마커를 분모로 쓰지 마라(C532⒝).
#   ⓒ**오염**: ctl 에서 신규 마커 0.
#   ⓓ**전수 포렌식**(집계에서 결론 직행 금지·[[08]]): 종료사유 분포 · 단계별 실패 분류 ·
#      **W그룹 × 레버 교차표** · 궤적 2~3건 정독. sim·turn 단위는 `t2_forensic.trace`/`turns_of`
#      (2026-08-18 정본화 — trace 가 turn 을 99.3% 갖고 있다)와 `sidecar_rows` 로 읽는다.
#   ⓔ**074 BYREF**: `[T2_SG_BYREF]` 발화 여부 — 0 이면 세 런 연속 미시험으로 확정 기록.
#   ⓕ**부작용**: 크래시 0 · CWE · 지연.
#
# ## 구성 — 네 런 **바이트 동일**
#
#   ctl / treat = +`T2_VERDICT_CARRY` +`T2_ELIG_LINE` · nt=1 · `GO_MAX_STEPS=150` ·
#   **`GO_CONCURRENCY=1`**([[30]] §동시성).

set -e
REPO=/home/woori/workspace_common/boltzmann-attention-pi
cd "$REPO/scripts/distill/tau2"

LOG=/home/woori/scratch/logs
SIMS=/home/woori/scratch/tau2-bench/data/simulations
mkdir -p "$LOG"

SHA=$(cd "$REPO" && git rev-parse --short HEAD)
DIRTY=$(cd "$REPO" && git status --porcelain -- \
  scripts/distill/tau2/t2_gate_patch.py scripts/distill/tau2/t2_search.py \
  scripts/distill/tau2/t2_resolve.py scripts/distill/tau2/t2_scaffold_get.py \
  scripts/distill/tau2/a2/ | grep -cv '^??' || true)
if [ "$DIRTY" != "0" ]; then
  echo "[t7313] REFUSING: 엔진 경로 커밋 안 된 변경 $DIRTY 개." >&2; exit 1
fi

for t in test_no_prose_regex.py test_sub_requirement.py test_docs_at_write.py \
         test_proceed_docbody.py test_cp2_clobber.py test_no_unbound_a2.py \
         test_deliver_precommit.py test_material_reserve.py test_material_bypass.py \
         test_probe_canonical.py test_log_join.py test_now_selfcall.py \
         test_no_undefined_names.py test_decision_carry.py test_subcall_return_type.py \
         test_a2_three_layer.py test_operator_find.py test_route_trace.py \
         test_group_parse.py test_verdict_carry.py test_pending_discovered.py \
         test_probe_scoring.py test_quote_in.py test_elig_handoff.py test_args_equal.py \
         test_flag_registry.py test_resolve_cap_marker.py test_byref_repairs.py; do
  [ -f "$t" ] || continue
  PYTHONPATH=/home/woori/scratch/tau2-bench/src timeout 60 \
    /home/woori/venvs/seka_env/bin/python "$t" >/dev/null 2>&1 \
    || { echo "[t7313] REFUSING: $t FAIL" >&2; exit 1; }
done
echo "[t7313] VERIFY OK"

if pgrep -f "[t]2_gap\.py" >/dev/null || pgrep -f "[x]3[0-9][0-9]_.*\.py" >/dev/null; then
  echo "[t7313] REFUSING: 무료 프로브 실행 중(양 포트 필요)" >&2; exit 1
fi

PIN="T2_ACTION_SUB=1 T2_KEEP_DENY_BODY=1 T2_CALL_FORM=1 T2_ARG_EMPTY=1 T2_SEARCH_AGENT=1 \
T2_DECIDE_ANY=1 T2_WRITE_ARG_ENUM=1 T2_DECIDE_BEFORE_WRITE=1 T2_DECISION_CARRY=1 \
T2_DISCOVERY_STEP2=1 T2_ARG_AXIS=1 T2_WRITE_SUB=3 T2_ACTION_INDEX=1 T2_NOW_SELFCALL=1 \
T2_SEARCH_ON_PROCEED=1 T2_ACT_DEMAND=0 T2_DELIVER_PRECOMMIT=0 T2_PROCEED_DOCBODY=0 \
T2_DOCS_AT_WRITE=0 T2_SUB_REQUIREMENT=0 T2_HANDOFF_PREDICATE=0 T2_PENDING_DISCOVERED=0"
TASKS='task_003,task_004,task_016,task_017,task_033,task_040,task_050,task_057,task_063,task_073,task_074,task_079,task_093,task_094,task_098,task_100'

launch () {
  NAME="$1"; PORT="$2"; VC="$3"; EL="$4"
  TAG="bank_t7313_${NAME}_20260818h"
  if [ -e "$LOG/${TAG}.log" ]; then echo "[t7313] SKIP: ${TAG}.log 존재" >&2; return 0; fi
  if [ -e "$SIMS/${TAG}" ]; then echo "[t7313] REFUSING: $SIMS/${TAG} 잔존" >&2; return 1; fi
  echo "{\"tag\":\"$TAG\",\"scaffold_sha\":\"$SHA\",\"port\":$PORT,\"tasks\":\"$TASKS\",\"nt\":1,\"verdict_carry\":\"$VC\",\"elig_line\":\"$EL\",\"max_steps\":150,\"concurrency\":1,\"pin\":\"$PIN\",\"why\":\"staged census stage 1 (16 tasks); stack frozen at t7312 sha\"}" \
    | tee "$LOG/${TAG}.meta.json"
  setsid bash -c "cd '$REPO/scripts/distill/tau2' && source ./go_stack.sh >/dev/null 2>&1 && \
    export $PIN && export T2_VERDICT_CARRY=$VC T2_ELIG_LINE=$EL && \
    export GO_MAX_STEPS=150 GO_CONCURRENCY=1 && \
    t2_launch $TAG $PORT '$TASKS' 1" \
    </dev/null >"$LOG/${TAG}.log" 2>&1 &
  echo "[t7313] $NAME(VC=$VC ELIG=$EL) → PID=$! port=$PORT"
}

launch ctl   8140 0 0
launch treat 8141 1 1
echo "[t7313] 기동 완료 · sha=$SHA · 팔당 12 sim · max_steps=150 · 1차 종점 = 레버별 발화율(pass 아님)"
