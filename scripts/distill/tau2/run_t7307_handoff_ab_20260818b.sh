#!/bin/bash
# t7307 — **B안: 019 가족 인계-술어 A/B** (사용자 승인 2026-08-18 *"B안 진행하라"*).
#
# ## 왜 이 런인가 (오늘의 격리가 정한 것)
#
# `x370 v6b`(C525·계기 유효: P_HINT 7/8 · D_NEG 0/8)가 give 단계의 결손 이름을 바꿨다 —
# **실행 능력은 있다**(답을 알려 주면 7/8 이 정확한 도구·인자로 건넨다). 정보-맞춘 문맥에서 없는 것은
# *"지금이 손님에게 넘길 자리다"* 라는 **판단**뿐이다(`B_NOLEAK` 1/8).
# ⇒ 판단은 엔진이 대신할 수 없다([[66]]·⛔0④). 허용 형태는 **표면화**뿐이고 그게 이 treat 이다.
#
# ## 처치 둘 (합성·[[19]])
#
#   ⑴ `T2_HANDOFF_PREDICATE=1` — 손님-측 도구 이름을 **말한 그 턴**에 *"아직 안 건넸다 + 건네려면
#      이 도구를 불러라"*. 기존 `GIVE_EXEC_NUDGE`·`UNCALLED_UNLOCK` 은 `_resign` 창에서만 발화해
#      **너무 늦다**. 엔진은 모델이 방금 말한 이름을 되읽을 뿐 고르지 않는다(여럿이면 전부 적는다).
#   ⑵ **W4 수리**(양팔 공통이 아니라 **treat 만**? — 아니다. A2 는 런처가 못 가르므로 **양팔 공통**이다.
#      ⇒ 이 런은 W4 수리가 **들어간 스택 위에서** ⑴만 A/B 한다. W4 자체의 인과는 이 런으로 못 산다.
#      대신 ⓑ로 **발화 사실**만 확인한다. W4 인과는 별도 런이 필요하다 — 이 한계를 먼저 적는다.)
#
# ## ★판정 (사전 고정 · 결과보다 먼저 · 이 순서로)
#
#   ⓐ**1차 종점 = 술어 발화율**(pass 가 아니다·C525⒠): treat 에서
#       `[T2_HANDOFF] named-but-not-given` 발화 sim / **손님-측 이름을 발화한 sim**.
#       ≥50% → 술어가 닿는다 · <50% → **사각지대 확정**(모델이 이름조차 안 꺼낸다) ⇒ 조건을
#       *국면*으로 넓히는 설계로 이동(그 판정은 LLM 몫·[[66]]).
#       ctl 에서 이 마커는 **0** 이어야 한다(안 그러면 배선 오염).
#   ⓑ**W4 발화 확인**: `[T2_ACTIONREQ] pending_user` 에 `call_discoverable_user_tool` 이 등장하는가.
#       양팔 공통이므로 **양팔 모두 등장**해야 정상(등장 0 이면 W4 수리가 死코드).
#   ⓒ**절차 지표**: `give_discoverable_user_tool` 호출 sim 수 · **손님 호출**(requestor=user) sim 수.
#   ⓓ**2차 종점 = pass**(`reward` 만·C486). ⚠**n=12/팔이라 검정력이 없다** — 방향만 본다.
#       pass 로 승격/폐기를 결정하지 않는다(그 결정은 S2/S3 몫).
#   ⓔ**부작용**: 지연 배수 · CWE · over-action · 크래시 0.
#
# 편성: 019 가족 6 태스크(019·020·022·027·028·029) × nt=2 × 2팔 = **24 sim** (≈2.7h).
# 구성 핀 = t7305 `ctl` 축자 + 오늘 확정 레버(기본 OFF 유지) · treat 만 `T2_HANDOFF_PREDICATE=1`.

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
  echo "[t7307] REFUSING: 엔진 경로 커밋 안 된 변경 $DIRTY 개." >&2; exit 1
fi

for t in test_no_prose_regex.py test_sub_requirement.py test_docs_at_write.py \
         test_proceed_docbody.py test_cp2_clobber.py test_no_unbound_a2.py \
         test_deliver_precommit.py test_material_reserve.py test_material_bypass.py \
         test_probe_canonical.py test_log_join.py test_now_selfcall.py \
         test_no_undefined_names.py test_decision_carry.py test_subcall_return_type.py \
         test_a2_three_layer.py test_operator_find.py test_route_trace.py \
         test_group_parse.py test_verdict_carry.py test_pending_discovered.py \
         test_probe_scoring.py test_quote_in.py test_elig_handoff.py test_args_equal.py; do
  [ -f "$t" ] || continue
  PYTHONPATH=/home/woori/scratch/tau2-bench/src /home/woori/venvs/seka_env/bin/python "$t" \
    >/dev/null 2>&1 || { echo "[t7307] REFUSING: $t FAIL" >&2; exit 1; }
done
echo "[t7307] VERIFY OK"

if pgrep -f "[t]2_gap\.py" >/dev/null || pgrep -f "[x]3[0-9][0-9]_.*\.py" >/dev/null; then
  echo "[t7307] REFUSING: 무료 프로브 실행 중(양 포트 필요)" >&2; exit 1
fi

PIN="T2_ACTION_SUB=1 T2_KEEP_DENY_BODY=1 T2_CALL_FORM=1 T2_ARG_EMPTY=1 T2_SEARCH_AGENT=1 \
T2_DECIDE_ANY=1 T2_WRITE_ARG_ENUM=1 T2_DECIDE_BEFORE_WRITE=1 T2_DECISION_CARRY=1 \
T2_DISCOVERY_STEP2=1 T2_ARG_AXIS=1 T2_WRITE_SUB=3 T2_ACTION_INDEX=1 T2_NOW_SELFCALL=1 \
T2_SEARCH_ON_PROCEED=1 T2_ACT_DEMAND=0 T2_DELIVER_PRECOMMIT=0 T2_PROCEED_DOCBODY=0 \
T2_DOCS_AT_WRITE=0 T2_SUB_REQUIREMENT=0 T2_VERDICT_CARRY=0 T2_ELIG_LINE=0 \
T2_PENDING_DISCOVERED=0"
TASKS='task_019,task_020,task_022,task_027,task_028,task_029'

launch () {
  NAME="$1"; PORT="$2"; HP="$3"
  TAG="bank_t7307_${NAME}_20260818b"
  if [ -e "$LOG/${TAG}.log" ]; then echo "[t7307] SKIP: ${TAG}.log 존재" >&2; return 0; fi
  if [ -e "$SIMS/${TAG}" ]; then echo "[t7307] REFUSING: $SIMS/${TAG} 잔존" >&2; return 1; fi
  echo "{\"tag\":\"$TAG\",\"scaffold_sha\":\"$SHA\",\"port\":$PORT,\"tasks\":\"$TASKS\",\"nt\":2,\"handoff\":\"$HP\",\"pin\":\"$PIN\",\"why\":\"handoff predicate A/B; primary endpoint is predicate firing rate, not pass (C525)\"}" \
    | tee "$LOG/${TAG}.meta.json"
  setsid bash -c "cd '$REPO/scripts/distill/tau2' && source ./go_stack.sh >/dev/null 2>&1 && \
    export $PIN && export T2_HANDOFF_PREDICATE=$HP && \
    t2_launch $TAG $PORT '$TASKS' 2" \
    </dev/null >"$LOG/${TAG}.log" 2>&1 &
  echo "[t7307] $NAME(handoff=$HP) → PID=$! port=$PORT"
}

launch ctl   8140 0
launch treat 8141 1
echo "[t7307] 기동 완료 · sha=$SHA · 팔당 12 sim(6 태스크 × nt=2) · 1차 종점 = 술어 발화율"
