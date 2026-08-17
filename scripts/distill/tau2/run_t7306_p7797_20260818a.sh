#!/bin/bash
# t7306 — **077~097 블록 파일럿**(사용자 제안 2026-08-18: *"77~97까지의 태스크 중 선별한 몇 개를
# 먼저 실험하여 증거를 확보하는 건 어떤가"*).
#
# ## 왜 이 8개인가 (두 구멍을 한 런으로 메운다)
#
# ⑴**측정 부재**: 077~097 21개는 census(2026-08-10) 전부 0.00 인데 그 수치는 **v1.0.1 이전**이고,
#   권위본이 *"077–086은 1.0.1 에서 수정됨(#402 agent-realizable)"* 이라고 적어 놨다 ⇒ 분모의
#   **22%가 사실상 미측정**이다. 표적을 모르면 어떤 레버도 기대값을 못 낸다.
# ⑵**배선 가설 검정**: 이 블록은 전부 **도구-지급/해제 프로토콜** 가족이고, 그 가족을 표적으로 한
#   유도 레버 5종(`T2_USER_TOOL_NOTE`·`T2_UNINSTRUCTABLE`·`T2_GIVE_EXEC_NUDGE`·
#   `T2_GIVE_RELEVANCE_NUDGE`·`T2_DUP_REPRESENT`)과 `T2_COVERAGE_FOLLOWUP` 이 **이미 go_stack 에
#   켜져 있다**. 그런데 라이브 발화가 0 이라는 관측이 있다([[55]] 0단계). 옛 로그로 추론하지 말고
#   **그 가족의 새 로그로** 직접 본다.
#
# ## 편성 (사전 고정 · 층화 표본 · 사후 선택 금지)
#
#   077·079  debit card activate/close/freeze + get_credit_card_accounts_by_user
#   081      + credit card dispute · human transfer
#   084·087  debit card transaction dispute 가족
#   090      bank account transactions 가족
#   093·096  apply_savings_account_credit 가족
#   = 8 태스크 × nt=1 × **1팔** = 8 sim (두 포트로 4/4 · ≈30~60분)
#
# **1팔인 이유**: 이 런은 A/B 가 아니라 **측정**이다(현재 수치는 얼마인가·우리 레버가 발화하는가).
# 구성은 t7305 `ctl` 팔과 **축자로 동일**하게 핀한다 ⇒ 통과하면 이 8 sim 은 B안 ctl 팔의
# **선불 8칸으로 재사용**할 수 있다(같은 env·같은 sha).
#
# ## 판정 (사전 고정 · 결과보다 먼저 읽는다)
#
#   ⓐ census  pass>0 인 태스크가 **하나라도** 있으면 → census 0.00 은 낡았다 **확정** ⇒ 표적 재도출
#             전부 0 이면 → 그 블록은 진짜 0 이고 표적 우선순위가 올라간다
#   ⓑ 배선    유도 레버 5종 + COVERAGE_FOLLOWUP 의 **발화 수**.
#             전부 0  → 배선 가설 **[S] 승격**(발화 0 의 원인을 그 자리에서 지목)
#             발화>0 인데 `give_discoverable_user_tool` 호출 0 → **배선 아님** ⇒ 문구·모델로 이동
#   ⓒ 절차    gold 도구(unlock/give/call) 실제 호출 수 · 손님 호출 수(requestor=user)
#   ⓓ 부작용  크래시 0 · 지연 · CWE · 종료 사유 분포
#
# ⚠[[09]] 비용: 8 sim · user-sim gpt-5.2. 탐색용 full-run 금지 규율 아래, 이것은 **표적 확정용
#   최소 scope** 이고 사용자 제안·승인으로 발사한다.
# ⚠[[30]] 종료 즉시 gzip → sim_results → `git add -f` → **tracked 확인**까지가 절차다.

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
  echo "[t7306] REFUSING: 엔진 경로 커밋 안 된 변경 $DIRTY 개." >&2; exit 1
fi

for t in test_no_prose_regex.py test_sub_requirement.py test_docs_at_write.py \
         test_proceed_docbody.py test_cp2_clobber.py test_no_unbound_a2.py \
         test_deliver_precommit.py test_material_reserve.py test_material_bypass.py \
         test_probe_canonical.py test_log_join.py test_now_selfcall.py \
         test_no_undefined_names.py test_decision_carry.py test_subcall_return_type.py \
         test_a2_three_layer.py test_operator_find.py test_route_trace.py \
         test_group_parse.py test_verdict_carry.py test_pending_discovered.py \
         test_probe_scoring.py test_quote_in.py test_elig_handoff.py; do
  [ -f "$t" ] || continue
  PYTHONPATH=/home/woori/scratch/tau2-bench/src /home/woori/venvs/seka_env/bin/python "$t" \
    >/dev/null 2>&1 || { echo "[t7306] REFUSING: $t FAIL" >&2; exit 1; }
done
echo "[t7306] VERIFY OK"

if pgrep -f "[t]2_gap\.py" >/dev/null || pgrep -f "[x]3[0-9][0-9]_.*\.py" >/dev/null; then
  echo "[t7306] REFUSING: 무료 프로브 실행 중(양 포트 필요)" >&2; exit 1
fi

# ★구성 핀 — t7305 ctl 팔의 export 목록 **축자**. 런처가 얹는 플래그(go_stack.sh 밖)도 여기 다 있다.
#   meta.json 에 그대로 적어 남긴다(이게 없으면 이 8 sim 은 baseline 구실을 못 한다).
PIN="T2_ACTION_SUB=1 T2_KEEP_DENY_BODY=1 T2_CALL_FORM=1 T2_ARG_EMPTY=1 T2_SEARCH_AGENT=1 \
T2_DECIDE_ANY=1 T2_WRITE_ARG_ENUM=1 T2_DECIDE_BEFORE_WRITE=1 T2_DECISION_CARRY=1 \
T2_DISCOVERY_STEP2=1 T2_ARG_AXIS=1 T2_WRITE_SUB=3 T2_ACTION_INDEX=1 T2_NOW_SELFCALL=1 \
T2_SEARCH_ON_PROCEED=1 T2_ACT_DEMAND=0 T2_DELIVER_PRECOMMIT=0 T2_PROCEED_DOCBODY=0 \
T2_DOCS_AT_WRITE=0 T2_SUB_REQUIREMENT=0 T2_VERDICT_CARRY=0 T2_PENDING_DISCOVERED=0"

launch () {
  NAME="$1"; PORT="$2"; TASKS="$3"
  TAG="bank_t7306_${NAME}_20260818a"
  if [ -e "$LOG/${TAG}.log" ]; then echo "[t7306] SKIP: ${TAG}.log 존재" >&2; return 0; fi
  if [ -e "$SIMS/${TAG}" ]; then echo "[t7306] REFUSING: $SIMS/${TAG} 잔존" >&2; return 1; fi
  echo "{\"tag\":\"$TAG\",\"scaffold_sha\":\"$SHA\",\"port\":$PORT,\"tasks\":\"$TASKS\",\"nt\":1,\"arms\":1,\"pin\":\"$PIN\",\"why\":\"077-097 block pilot: is the census stale, and do the elicitation levers fire\"}" \
    | tee "$LOG/${TAG}.meta.json"
  setsid bash -c "cd '$REPO/scripts/distill/tau2' && source ./go_stack.sh >/dev/null 2>&1 && \
    export $PIN && t2_launch $TAG $PORT '$TASKS' 1" \
    </dev/null >"$LOG/${TAG}.log" 2>&1 &
  echo "[t7306] $NAME → PID=$! port=$PORT tasks=$TASKS"
}

launch a 8140 'task_077,task_079,task_081,task_084'
launch b 8141 'task_087,task_090,task_093,task_096'
echo "[t7306] 기동 완료 · sha=$SHA · 8 sim(8 태스크 × nt=1 × 1팔) · 층화 표본 사전 고정"
