#!/bin/bash
# t7311 — **그룹 횡단 확인 런**(S3 발사 전 보험 · 사용자 지시 2026-08-18 축자:
#         *"12 태스크로 늘려라. 대표적인 태스크들 다 넣어라 각 그룹별"*).
#
# ## 무엇을 사는가 (S3 15시간을 걸기 전에 닫아야 하는 것)
#
#   ⑴ **C533⒡ 종결** — 요구-인용 기각이 *옳은 거부*인지 *과한 검산*인지. 오프라인 재현(x373)은
#      양성통제가 무너져 **무효**였다. 이제 로그가 **기각된 인용을 축자로** 남기므로 라이브가 답한다.
#   ⑵ **따옴표 수리(C533⒠) 라이브 확인** — 감싼 인용을 벗기는 수리가 실제로 통과를 만드는가.
#   ⑶ **074 BYREF 수리 라이브 시험**(C531⒠) — t7310 로스터는 `@last:` 를 한 번도 안 썼다(양팔 0회).
#      로스터를 넓히면 시험될 가능성이 커진다. 안 뜨면 그대로 *미시험*으로 기록한다(분모 0).
#   ⑷ **레버가 다른 그룹에서도 닿는가** — S2 는 선택축이 몰린 로스터였다. 그룹 횡단에서도
#      ELIG/VERDICT 발화율이 유지되는지 본다.
#
# ## 편성 — **C466⒟ 그룹 대표**(10 그룹) + 최대 2그룹 보강 = 12
#
#   G1 085·081 / G2 055·061 / G3 050 / G4 040 / G5 094 / G6 098 / G7 017 / G8 033 / G9 003 / G10 072
#   · 각 그룹의 **1순위 대표**를 먼저 넣고, 실패 질량이 가장 큰 두 그룹(G1 64/101 · G2 39/75)만
#     2순위 대표(081·061)를 더해 12를 채웠다. **사후 선택 금지** — 이 규칙으로 고정.
#   · ⛔제외 명부([[68]]) 반영: **005**(G8 3순위·gold 전 필드 센티널) · **102**(G6 3순위·env 출처 부재)
#     는 분모 제외라 안 넣고, **069**(G2 3순위)는 분모엔 남지만 **표적 금지**라 안 넣는다.
#
# ## ⚠사전 고정 판정 (결과보다 먼저 · 이 순서로)
#
#   ⓐ**1차 = 요구-인용 기각의 성질**(C533⒡): `[T2_SUB_REQUIREMENT] … 기각 K: <축자>` 를 전수로 읽어
#       기각분이 **원문에 있는가**를 오프라인에서 대조한다.
#       · 기각 0 또는 기각분이 원문에 **없다** → **옳은 거부** 확정(모델이 축자를 안 지킴)
#       · 기각분이 원문에 **있다** → 아직 남은 **검산 결함** — 그 형태를 보고 다시 고친다
#   ⓑ**따옴표 수리 확인**: 감싼 인용(`"…"`)이 기각 목록에 **안 나와야** 한다(수리가 먹었다면).
#   ⓒ**074 BYREF**: `[T2_SG_BYREF]` 발화 sim 수 · `resolved by reference` ↔ `no committed non-error
#       output` 비. **0 이면 미시험**으로 기록(死배선 아님·분모 0).
#   ⓓ**레버 발화율(그룹 횡단)**: ELIG·VERDICT 를 **각자의 트리거 자리 대비**로 잰다.
#       ⚠분모를 이웃 마커에서 뽑지 마라 — C532⒝ 에서 그 실수로 82% 를 잘못 셌다.
#   ⓔ**오염**: ctl 에서 신규 마커 0. **부작용**: 크래시 0 · 지연 배수.
#   ⛔**pass 는 레버 판정에 쓰지 않는다**(12 태스크 nt=1·검정력 없음). 다만 그룹 대표 로스터라
#     **census 관측치**로는 기록한다 — 옛 census(2026-08-10)가 낡았다는 것이 이미 실측됐다(C-08-17).
#
# ## 구성
#
#   t7310 과 **바이트 동일**(ctl / treat = +`T2_VERDICT_CARRY` +`T2_ELIG_LINE`) · nt=1 ·
#   `GO_MAX_STEPS=150` · **`GO_CONCURRENCY=1`**([[30]] §동시성).
#   엔진은 t7310 이후 **세 곳** 바뀌었다(전부 배터리 28/28 통과):
#     · `_resolve_ref_output` 래퍼 언랩 + isolate 선점 해소(C531)
#     · `quote_in` 감싼 따옴표 벗김(C533)
#     · 기각 인용 로깅 · resolve-cap 리셋 마커(관측·거동 0)

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
  echo "[t7311] REFUSING: 엔진 경로 커밋 안 된 변경 $DIRTY 개." >&2; exit 1
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
    || { echo "[t7311] REFUSING: $t FAIL" >&2; exit 1; }
done
echo "[t7311] VERIFY OK"

if pgrep -f "[t]2_gap\.py" >/dev/null || pgrep -f "[x]3[0-9][0-9]_.*\.py" >/dev/null; then
  echo "[t7311] REFUSING: 무료 프로브 실행 중(양 포트 필요)" >&2; exit 1
fi

PIN="T2_ACTION_SUB=1 T2_KEEP_DENY_BODY=1 T2_CALL_FORM=1 T2_ARG_EMPTY=1 T2_SEARCH_AGENT=1 \
T2_DECIDE_ANY=1 T2_WRITE_ARG_ENUM=1 T2_DECIDE_BEFORE_WRITE=1 T2_DECISION_CARRY=1 \
T2_DISCOVERY_STEP2=1 T2_ARG_AXIS=1 T2_WRITE_SUB=3 T2_ACTION_INDEX=1 T2_NOW_SELFCALL=1 \
T2_SEARCH_ON_PROCEED=1 T2_ACT_DEMAND=0 T2_DELIVER_PRECOMMIT=0 T2_PROCEED_DOCBODY=0 \
T2_DOCS_AT_WRITE=0 T2_SUB_REQUIREMENT=0 T2_HANDOFF_PREDICATE=0 T2_PENDING_DISCOVERED=0"
TASKS='task_085,task_081,task_055,task_061,task_050,task_040,task_094,task_098,task_017,task_033,task_003,task_072'

launch () {
  NAME="$1"; PORT="$2"; VC="$3"; EL="$4"
  TAG="bank_t7311_${NAME}_20260818f"
  if [ -e "$LOG/${TAG}.log" ]; then echo "[t7311] SKIP: ${TAG}.log 존재" >&2; return 0; fi
  if [ -e "$SIMS/${TAG}" ]; then echo "[t7311] REFUSING: $SIMS/${TAG} 잔존" >&2; return 1; fi
  echo "{\"tag\":\"$TAG\",\"scaffold_sha\":\"$SHA\",\"port\":$PORT,\"tasks\":\"$TASKS\",\"nt\":1,\"verdict_carry\":\"$VC\",\"elig_line\":\"$EL\",\"max_steps\":150,\"concurrency\":1,\"pin\":\"$PIN\",\"why\":\"S2 wiring smoke: primary endpoint is per-lever firing rate, pass is not judged\"}" \
    | tee "$LOG/${TAG}.meta.json"
  setsid bash -c "cd '$REPO/scripts/distill/tau2' && source ./go_stack.sh >/dev/null 2>&1 && \
    export $PIN && export T2_VERDICT_CARRY=$VC T2_ELIG_LINE=$EL && \
    export GO_MAX_STEPS=150 GO_CONCURRENCY=1 && \
    t2_launch $TAG $PORT '$TASKS' 1" \
    </dev/null >"$LOG/${TAG}.log" 2>&1 &
  echo "[t7311] $NAME(VC=$VC ELIG=$EL) → PID=$! port=$PORT"
}

launch ctl   8140 0 0
launch treat 8141 1 1
echo "[t7311] 기동 완료 · sha=$SHA · 팔당 12 sim · max_steps=150 · 1차 종점 = 레버별 발화율(pass 아님)"
