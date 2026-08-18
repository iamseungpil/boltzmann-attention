#!/bin/bash
# t7324 (구 t7323 재발사) — **선언-우선 가이드 A/B** (사용자 지시 2026-08-19: *"A/B 설계 유료런 하라"*).
#
# ## 무엇을 판정하는 런인가
#
# `T2_DECLFIRST_GUIDE_FIX` 는 **되살리기 스위치**다. `unified` 경로의 declfirst 가이드 주입이
# `a2` 미바인딩(`UnboundLocalError`)으로 죽어 있었고 `except: pass` 가 그것을 삼켰다 —
# 그 경로에서 가이드는 **한 번도 주입된 적이 없다**(2026-08-16 발견). 주석이 못 박아 둔 조건:
# *"조용히 살리지 않는다: 살리면 모든 과거 런과 베이스라인이 달라진다. `T2_DECLFIRST_GUIDE_FIX=1`
# 일 때만 살리고 효과는 **별도 A/B** 로 잰다([[57]])."* 이 런이 그 A/B 다.
#
# ★재발사 (t7323 중단·2026-08-19): 첫 sim 에서 **가이드와 읽기 루틴의 간섭**이 드러났다 —
#   declfirst 팔 050 이 1.0→0.0, 루틴 발화 4→1, `viol=R2_CALL_WITHOUT_ACT,R4_NEXT_ACTION_MISMATCH`.
#   조정(끄기 아님·[[19]]): 가이드 **rev2** = *"이 턴의 도구가 하나로 제한돼 있으면 그 호출을 먼저
#   하고 봉투는 다음 메시지에"*. 그래서 이 런은 **rev2 를 재는 것**이고 X13 의 31.8%(rev1)와 직접
#   비교하지 않는다. 추가로 이관 자리에 E-PLAN 원장 근거가 붙었다(073 표적).
#
# ⚠상위 `T2_DECLFIRST` 도 死배선이라 **둘을 함께** 켜야 가이드가 나간다. 문면 실재 확인 완료
#   (A2 L1 `declaration.guide` **716자** · 축자: *"state a declaration envelope for every turn …
#   **Write it FIRST, before any tool call** … In done_report, `tool` MUST be the exact name of the
#   tool that performed what you claim"*).
#
# ## 사전 고정 판정 (런 전에 못 박는다)
#
#   ⓐ **배선이 산다** — treat 에서 `[T2_DECLFIRST]`/봉투 산출이 **>0**. 0 이면 死배선 재발이고
#      그 실행은 **무효**로 선언한다(오늘 규칙: 신호가 0 으로 붙으면 무효).
#   ⓑ **1차 종점 = 봉투 산출율**. 선행 X13(A_PROMPT) 은 *가이드만으로 턴의 **31.8%*** 에서 봉투를
#      냈다. 그 자릿수가 재현되는지가 이 레버의 생사다.
#   ⓒ **반대편 계측(§1.3)** — Δspurious = gold 밖 write 건수. 늘면 손해다(촉구 `T2_ACT_DEMAND` 가
#      라이브에서 정확히 그렇게 죽었다: pass null · over-action 2→8).
#   ⓓ 지연·steps: 가이드는 시스템 프롬프트에 716자를 얹는다. steps 합·sim 당 초를 ctl 과 나란히.
#   ⛔pass 로 판정하지 않는다 — 4 태스크 × nt=2 는 그 해상도가 없다(오늘 073 이 같은 시드에서
#     1.0 ↔ 0.0 을 냈다). pass 는 부수 관측이고, 판정은 ⓐ~ⓓ다.
#
# ## 이 런의 두 번째 쓸모 — ctl 팔 = **4 태스크 스모크**
#
# ctl 은 오늘 켠 네 레버(`T2_PROV_OURS`·`T2_NOREC_BRANCH`·`T2_GROUND_HDR`·`T2_RETURN_EMPTY`)를
# 포함한 **새 스택의 기준선**이다. 사용자 조건(*"4개 태스크는 모두 pass 해야 20 태스크 실행"*)의
# 재확인을 여기서 겸한다 — 050 은 t7322 에서 1.0 을 냈고, 나머지 셋을 같은 스택에서 다시 본다.
#
# ## 구성
#   4 태스크(098·100·073·050) × **nt=2** × 2팔 = **16 sim** · `GO_MAX_STEPS=150` · concurrency 1.
#   nt=2 인 이유: 오늘 073 이 단일 시드에서 뒤집혔다(C543ⓐ 의 귀속도 그만큼 약해졌다).

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
  echo "[t7324] REFUSING: 엔진 경로 커밋 안 된 변경 $DIRTY 개." >&2; exit 1
fi

for t in test_a2_three_layer.py test_flag_registry.py test_claim_verify.py \
         test_claim_tool_index.py test_read_routine.py test_proc_read_connect.py \
         test_verdict_gate.py test_verdict_carry.py test_no_undefined_names.py \
         test_no_unbound_a2.py test_quote_in.py test_args_equal.py test_t2_procedure.py \
         test_proc_absent_wiring.py test_pin_read_replay.py test_eplan.py \
         test_decision_carry.py test_route_trace.py test_group_parse.py \
         test_resolve_cap_runtime.py test_byref_repairs.py test_no_prose_regex.py; do
  [ -f "$t" ] || continue
  PYTHONPATH=/home/woori/scratch/tau2-bench/src timeout 90 \
    /home/woori/venvs/seka_env/bin/python "$t" >/dev/null 2>&1 \
    || { echo "[t7324] REFUSING: $t FAIL" >&2; exit 1; }
done
echo "[t7324] VERIFY OK (배터리 22)"

if pgrep -f "[x]3[0-9][0-9]_.*\.py" >/dev/null; then
  echo "[t7324] REFUSING: 무료 프로브 실행 중(양 포트 필요)" >&2; exit 1
fi
if pgrep -f "[t]2_launch" >/dev/null; then
  echo "[t7324] REFUSING: 다른 라이브 런이 돌고 있다" >&2; exit 1
fi

PIN="T2_ACTION_SUB=1 T2_KEEP_DENY_BODY=1 T2_CALL_FORM=1 T2_ARG_EMPTY=1 T2_SEARCH_AGENT=1 \
T2_DECIDE_ANY=1 T2_WRITE_ARG_ENUM=1 T2_DECIDE_BEFORE_WRITE=1 T2_DECISION_CARRY=1 \
T2_DISCOVERY_STEP2=1 T2_ARG_AXIS=1 T2_WRITE_SUB=3 T2_ACTION_INDEX=1 T2_NOW_SELFCALL=1 \
T2_SEARCH_ON_PROCEED=1 T2_ACT_DEMAND=0 T2_DELIVER_PRECOMMIT=0 T2_PROCEED_DOCBODY=0 \
T2_DOCS_AT_WRITE=0 T2_SUB_REQUIREMENT=0 T2_HANDOFF_PREDICATE=0 T2_PENDING_DISCOVERED=0 \
T2_VERDICT_CARRY=0 T2_ELIG_LINE=0 T2_VERDICT_GATE=0 T2_CLAIM_VERIFY=0"
TASKS='task_098,task_100,task_073,task_050'
NT=2

launch () {
  NAME="$1"; PORT="$2"; DF="$3"
  TAG="bank_t7324_${NAME}_20260819b"
  if [ -e "$LOG/${TAG}.log" ]; then echo "[t7324] SKIP: ${TAG}.log 존재" >&2; return 0; fi
  if [ -e "$SIMS/${TAG}" ]; then echo "[t7324] REFUSING: $SIMS/${TAG} 잔존" >&2; return 1; fi
  echo "{\"tag\":\"$TAG\",\"scaffold_sha\":\"$SHA\",\"port\":$PORT,\"tasks\":\"$TASKS\",\"nt\":$NT,\"declfirst\":\"$DF\",\"max_steps\":150,\"concurrency\":1,\"pin\":\"$PIN\",\"why\":\"declfirst guide revival A/B; endpoint = envelope rate + delta-spurious, not pass\"}" \
    | tee "$LOG/${TAG}.meta.json"
  setsid bash -c "cd '$REPO/scripts/distill/tau2' && source ./go_stack.sh >/dev/null 2>&1 && \
    export $PIN && export T2_DECLFIRST=$DF T2_DECLFIRST_GUIDE_FIX=$DF && \
    export GO_MAX_STEPS=150 GO_CONCURRENCY=1 && \
    t2_launch $TAG $PORT '$TASKS' $NT" \
    </dev/null >"$LOG/${TAG}.log" 2>&1 &
  echo "[t7324] $NAME(declfirst=$DF) → PID=$! port=$PORT"
}

launch ctl      8140 0
sleep 2
launch declfirst 8141 1
echo "[t7324] 기동 · sha=$SHA · 팔당 8 sim · 1차 종점 = 봉투 산출율 · pass 는 부수"
