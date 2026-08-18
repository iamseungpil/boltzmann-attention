#!/bin/bash
# t7312 — **기능 스모크**(사용자 지시 2026-08-18 축자: *"스모크는 1단계중에서 4개 정도만 빼서
#         레버들이 동작하는지만 기능만 확인하라"*).
#
# ## 위치 — 이것은 3단계 census 의 **앞머리**다
#
# 사용자 지시로 실효 **95** 를 3단계로 **정확히 한 번씩** 나눠 돈다(겹침 0·재실행 0). 사전 고정
# 명부 = `reports/facet_rft_2026/S3_STAGED_ROSTERS_2026_08_18.json`.
#   · 스모크 **4**(여기) → 1단계 **16** → 2단계 **30** → 3단계 **45**  = 4+16+30+45 = **95**
#   · 스모크가 **스택을 안 바꾸고** 끝나면 이 4 sim/팔은 **1단계 몫으로 인정**한다(재실행 없음).
#     결함이 나와 엔진·A2 를 고치면 이 4개는 1단계에 **다시 포함**한다(스택이 달라졌으므로).
#   · ★**1단계는 검문소다**(사용자 지시): 1단계에서 문제가 많이 나오면 **고친 뒤 1단계를
#     처음부터 다시** 돈다. 2·3단계는 1단계가 통과한 뒤에만 간다.
#   · ★**단계마다 전수 포렌식**(사용자 지시): 종료사유 분포·실패 분류·**그룹×레버 교차표**·
#     궤적 정독을 매 단계 수행한다 — 집계에서 결론 직행 금지([[08]]).
#   · 1단계는 **그룹 대표 ∪ 레버·원인별 대표**(074=L3/BYREF 포함)로 짜여 있어, 1단계만으로
#     *어느 그룹에서 어느 레버가 통했나* 를 읽을 수 있다(명부 `stage1_coverage`).
#   · ⛔**스택 동결**: 스모크 통과 시점부터 3단계 종료까지 엔진·A2·플래그 **불변**. 바꾸면 단계를
#     합쳐 셀 수 없다 — 각 단계의 `scaffold_sha` 를 meta 에 남기고 판정 때 대조한다.
#
# ## 스모크 4개를 왜 이것으로 골랐나 (사전 고정)
#
#   · `task_055` 선택축(L-V) 표적 — t7310 에서 요구 인용 3/3·6/6 통과 실적(양성 기준선)
#   · `task_024` 자격축(ELIG) 표적 — 사업자 · t7310 에서 **2차 호출 3/3 기각**이 났던 자리
#   · `task_072` BYREF·계산 축 — 074 수리(C531)가 라이브에서 시험될 유일한 기회
#   · `task_085` G1 대표 — 실패 질량 최대 그룹(64/101)
#
# ## ⚠사전 고정 판정 — **기능만 본다**(사용자 지시)
#
#   ⓐ**레버가 도는가**: treat 에서 `[T2_ELIG]`·`[T2_VERDICT]` 가 **각자의 트리거 자리 대비** 발화.
#       ⚠분모를 이웃 마커에서 뽑지 마라(C532⒝ 에서 그 실수로 82% 를 잘못 셌다).
#   ⓑ**오염**: ctl 에서 그 두 마커 **0**.
#   ⓒ**요구-인용 기각의 성질**(C533⒡): `[T2_SUB_REQUIREMENT] … 기각 K: <축자>` 를 원문과 대조 —
#       원문에 **없으면** 옳은 거부 · **있으면** 남은 검산 결함(고치고 이 4개는 1단계로 되돌린다).
#   ⓓ**따옴표 수리 확인**: 감싼 인용(`"…"`)이 기각 목록에 **안 나와야** 한다.
#   ⓔ**074 BYREF**: `[T2_SG_BYREF]` 발화 여부. 0 이면 **미시험**으로 기록(死배선 아님·분모 0).
#   ⓕ**크래시 0**.
#   ⛔**pass 로 레버를 판정하지 않는다**(n=4). 다만 census 의 일부이므로 **성적은 보존**한다.
#
# ## 구성 — 3단계 전체와 **바이트 동일**해야 한다
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
  echo "[t7312] REFUSING: 엔진 경로 커밋 안 된 변경 $DIRTY 개." >&2; exit 1
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
    || { echo "[t7312] REFUSING: $t FAIL" >&2; exit 1; }
done
echo "[t7312] VERIFY OK"

if pgrep -f "[t]2_gap\.py" >/dev/null || pgrep -f "[x]3[0-9][0-9]_.*\.py" >/dev/null; then
  echo "[t7312] REFUSING: 무료 프로브 실행 중(양 포트 필요)" >&2; exit 1
fi

PIN="T2_ACTION_SUB=1 T2_KEEP_DENY_BODY=1 T2_CALL_FORM=1 T2_ARG_EMPTY=1 T2_SEARCH_AGENT=1 \
T2_DECIDE_ANY=1 T2_WRITE_ARG_ENUM=1 T2_DECIDE_BEFORE_WRITE=1 T2_DECISION_CARRY=1 \
T2_DISCOVERY_STEP2=1 T2_ARG_AXIS=1 T2_WRITE_SUB=3 T2_ACTION_INDEX=1 T2_NOW_SELFCALL=1 \
T2_SEARCH_ON_PROCEED=1 T2_ACT_DEMAND=0 T2_DELIVER_PRECOMMIT=0 T2_PROCEED_DOCBODY=0 \
T2_DOCS_AT_WRITE=0 T2_SUB_REQUIREMENT=0 T2_HANDOFF_PREDICATE=0 T2_PENDING_DISCOVERED=0"
TASKS='task_055,task_024,task_072,task_085'

launch () {
  NAME="$1"; PORT="$2"; VC="$3"; EL="$4"
  TAG="bank_t7312_${NAME}_20260818g"
  if [ -e "$LOG/${TAG}.log" ]; then echo "[t7312] SKIP: ${TAG}.log 존재" >&2; return 0; fi
  if [ -e "$SIMS/${TAG}" ]; then echo "[t7312] REFUSING: $SIMS/${TAG} 잔존" >&2; return 1; fi
  echo "{\"tag\":\"$TAG\",\"scaffold_sha\":\"$SHA\",\"port\":$PORT,\"tasks\":\"$TASKS\",\"nt\":1,\"verdict_carry\":\"$VC\",\"elig_line\":\"$EL\",\"max_steps\":150,\"concurrency\":1,\"pin\":\"$PIN\",\"why\":\"function smoke, head of the staged census; 4 tasks drawn from stage 1\"}" \
    | tee "$LOG/${TAG}.meta.json"
  setsid bash -c "cd '$REPO/scripts/distill/tau2' && source ./go_stack.sh >/dev/null 2>&1 && \
    export $PIN && export T2_VERDICT_CARRY=$VC T2_ELIG_LINE=$EL && \
    export GO_MAX_STEPS=150 GO_CONCURRENCY=1 && \
    t2_launch $TAG $PORT '$TASKS' 1" \
    </dev/null >"$LOG/${TAG}.log" 2>&1 &
  echo "[t7312] $NAME(VC=$VC ELIG=$EL) → PID=$! port=$PORT"
}

launch ctl   8140 0 0
launch treat 8141 1 1
echo "[t7312] 기동 완료 · sha=$SHA · 팔당 12 sim · max_steps=150 · 1차 종점 = 레버별 발화율(pass 아님)"
