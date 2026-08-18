#!/bin/bash
# t7314 — **수리 스모크**(사용자 지시 2026-08-18: *"4번 수리후 20개 태스크 재실행시 4개의 태스크를
#         수리 스모크로 다시 선정하라"*).
#
# ## 무엇을 확인하는 런인가
#
# 오늘 네 곳을 고쳤다. 이 스모크는 **그 넷이 라이브에서 실제로 도는지**만 본다.
#   ⒜ C538 복원 — `_resolve_cap_ok` 두 경로에서 **대입을 print 앞으로** + `_sys`→`sys`.
#      (오늘 09:19 `a627a18b` 가 `NameError` 를 `except: pass` 로 삼켜 **리셋 대입을 통째로 건너뛰게**
#       만들었고, 그것이 캡을 **영구 래치**로 바꿨다 — x381 줄-추적으로 확정)
#   ⒝ C536ⓑ — fb 조립의 마지막 `else`(이름 없는 `_FB_GENERIC`)를 `_sibling_wait`(막힌 호출 **이름**
#      + 다음 한 수)로. 회계·접힘은 불변, 나가는 문자열만 바뀐다. OFF 면 바이트 동일.
#   ⒞ C534⒢ — `quote_in` 이 **대시류를 동치**로 본다(098 의 `—` ↔ `--` 거짓 기각).
#   ⒟ C538 가족 — `_cp2_assign` 잠복 크래시 · `unified()` 의 `_ts` 미정의(死배선) 수리.
#
# ## 태스크 4개를 왜 이것으로 골랐나 (★측정으로 골랐다 · 사전 고정)
#
#   · `task_098` — **회귀 최강 신호**. seed 626729 에서 08-13~08-17 **14/15** → 08-18 **0/4**(C540).
#                  대시 거짓 기각(⒞)도 이 태스크에서 났다.
#   · `task_100` — 회귀 2호. fail 궤적이 gold `get_all_user_accounts_by_user_id_3847` 를 **아예 안
#                  부르고** 곧장 `submit_referral` 했다(조회 0회).
#   · `task_073` — 회귀 3호. 계좌 3개 중 **Blue 하나만** 처리(커버리지 절단).
#   · `task_050` — ⒝ **최대 노출**: t7313 두 팔 합쳐 이름 없는 거부 본문 **199회**(2위 074 가 13회).
#
# ## ⚠사전 고정 판정 — **종점은 원인 지표이고 pass 는 부수 관측**이다(런북 STEP 4 규약)
#
#   ⓐ **리셋이 산다**: 어느 sim 에서든 `[T2_RESOLVE_CAP] 리셋` **≥1**.
#      (영속 6런 927 stop 동안 **0회**였다 — 그것이 죽어 있었다는 증거였다)
#   ⓑ **래치가 풀린다**: 098·073 의 `stop=resolve_cap` 이 t7313 대비 감소(098 6 · 073 25 기준).
#   ⓒ **조회가 돌아온다**: 100 에서 `get_all_user_accounts_by_user_id_3847` **호출 ≥1** ·
#      073 에서 `get_atm_fee_discrepancies` 가 **계좌 3종**(Blue·Green·Light Green) 전부.
#   ⓓ **이름 없는 본문이 사라진다**: 050 사이드카에서 `resolve the flagged call(s) first` **0** ∧
#      `another call in the same turn was blocked` **≥1**.
#   ⓔ **대시 수리**: 098 요구-인용 `기각 K:` **0**.
#   ⓕ **크래시 0 · CWE 0**.
#   ⛔pass 로 레버를 판정하지 않는다(n=4). 다만 회귀 3건이 되살아나면 그것은 **C538 귀속의 확증**이다.
#
# ## 구성 — 20 태스크 재실행과 **바이트 동일**해야 한다
#
#   ctl / treat = +`T2_VERDICT_CARRY` +`T2_ELIG_LINE` · nt=1 · `GO_MAX_STEPS=150` · `GO_CONCURRENCY=1`.
#   `T2_NOREC_BRANCH` 는 **끈 채로 둔다** — x379 가 사전 기준을 충족하지 못했고(C541ⓐ), 이 런의
#   목적은 **수리 귀속**이라 다른 변수를 같이 넣지 않는다.

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
  echo "[t7314] REFUSING: 엔진 경로 커밋 안 된 변경 $DIRTY 개." >&2; exit 1
fi

for t in test_no_prose_regex.py test_sub_requirement.py test_docs_at_write.py \
         test_proceed_docbody.py test_cp2_clobber.py test_no_unbound_a2.py \
         test_deliver_precommit.py test_material_reserve.py test_material_bypass.py \
         test_probe_canonical.py test_log_join.py test_now_selfcall.py \
         test_no_undefined_names.py test_decision_carry.py test_subcall_return_type.py \
         test_a2_three_layer.py test_operator_find.py test_route_trace.py \
         test_group_parse.py test_verdict_carry.py test_pending_discovered.py \
         test_probe_scoring.py test_quote_in.py test_elig_handoff.py test_args_equal.py \
         test_flag_registry.py test_resolve_cap_marker.py test_byref_repairs.py \
         test_resolve_cap_runtime.py; do
  [ -f "$t" ] || continue
  PYTHONPATH=/home/woori/scratch/tau2-bench/src timeout 90 \
    /home/woori/venvs/seka_env/bin/python "$t" >/dev/null 2>&1 \
    || { echo "[t7314] REFUSING: $t FAIL" >&2; exit 1; }
done
echo "[t7314] VERIFY OK (배터리 29)"

if pgrep -f "[x]3[0-9][0-9]_.*\.py" >/dev/null; then
  echo "[t7314] REFUSING: 무료 프로브 실행 중(양 포트 필요)" >&2; exit 1
fi

PIN="T2_ACTION_SUB=1 T2_KEEP_DENY_BODY=1 T2_CALL_FORM=1 T2_ARG_EMPTY=1 T2_SEARCH_AGENT=1 \
T2_DECIDE_ANY=1 T2_WRITE_ARG_ENUM=1 T2_DECIDE_BEFORE_WRITE=1 T2_DECISION_CARRY=1 \
T2_DISCOVERY_STEP2=1 T2_ARG_AXIS=1 T2_WRITE_SUB=3 T2_ACTION_INDEX=1 T2_NOW_SELFCALL=1 \
T2_SEARCH_ON_PROCEED=1 T2_ACT_DEMAND=0 T2_DELIVER_PRECOMMIT=0 T2_PROCEED_DOCBODY=0 \
T2_DOCS_AT_WRITE=0 T2_SUB_REQUIREMENT=0 T2_HANDOFF_PREDICATE=0 T2_PENDING_DISCOVERED=0"
TASKS='task_098,task_100,task_073,task_050'

launch () {
  NAME="$1"; PORT="$2"; VC="$3"; EL="$4"
  TAG="bank_t7314_${NAME}_20260818j"
  if [ -e "$LOG/${TAG}.log" ]; then echo "[t7314] SKIP: ${TAG}.log 존재" >&2; return 0; fi
  if [ -e "$SIMS/${TAG}" ]; then echo "[t7314] REFUSING: $SIMS/${TAG} 잔존" >&2; return 1; fi
  echo "{\"tag\":\"$TAG\",\"scaffold_sha\":\"$SHA\",\"port\":$PORT,\"tasks\":\"$TASKS\",\"nt\":1,\"verdict_carry\":\"$VC\",\"elig_line\":\"$EL\",\"max_steps\":150,\"concurrency\":1,\"pin\":\"$PIN\",\"why\":\"repair smoke: C538 restore + C536b named deny body + quote_in dash; endpoint = cause indicators, not pass\"}" \
    | tee "$LOG/${TAG}.meta.json"
  setsid bash -c "cd '$REPO/scripts/distill/tau2' && source ./go_stack.sh >/dev/null 2>&1 && \
    export $PIN && export T2_VERDICT_CARRY=$VC T2_ELIG_LINE=$EL && \
    export GO_MAX_STEPS=150 GO_CONCURRENCY=1 && \
    t2_launch $TAG $PORT '$TASKS' 1" \
    </dev/null >"$LOG/${TAG}.log" 2>&1 &
  echo "[t7314] $NAME(VC=$VC ELIG=$EL) → PID=$! port=$PORT"
}

launch ctl   8140 0 0
launch treat 8141 1 1
echo "[t7314] 기동 · sha=$SHA · 팔당 4 sim · 종점 = 원인 지표(ⓐ~ⓕ) · pass 는 부수"
