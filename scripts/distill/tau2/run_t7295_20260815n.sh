#!/bin/bash
# ★★**밤샘런 t7295** (2026-08-15·사용자 지시) — **22 태스크 × nt=4 = 88 sim**.
#
# 편성: G대표 12(003·036·048·049·050·055·061·063·069·081·085·087) + fee 가족 4(072~075)
#       + 지정 6(010·098·099·100·070·071).  a/b 각 11 태스크(무거운 것 분산: 081→b · 048→a).
#
# 실린 것 — 오늘 **측정으로 정당화된** 레버 둘이 라이브 첫 판정:
#   ⑴ **지목 → 범위 표면화**(x322·n=24·블록 8·8·8): A_REF **24/24** ↔ 옛 지목 **0/24** ↔
#      범위 표시 **24/24**. 우리 개입이 모델이 맞히던 선택을 파괴하고 있었고, 073 의 중복 적립
#      (C485·같은 계좌 $9.50 두 번 → `db_match=False`)이 그 손실의 라이브 착지였다.
#      이제 엔진은 **고르지 않는다** — 후보의 선언된 범위만 인쇄한다.
#   ⑵ **환급 차감**(x323 + 072 원장 검산): 부과 24.00 − 환급 10.00 = **14.00 = gold**.
#      환급 축자 제공 0/24 · 정책 문면까지 0/24 · 엔진이 뺀 값 24/24 ⇒ 뺄셈 한 칸만 정당.
#   (+ 오늘 오전분: `sub_generate` 반환형 8곳 수리 · 중첩 계약 검산 · ACTION-INDEX 43줄 1회 표면화)
#
# 검색 = **alltools**(bm25+dense+shell). t7293 의 shell 단독은 기각(0/8·075 2/2→0/2·C487) —
#   빼기는 기계적으로 작동했으나 **도구 채택 ≠ 검색 성능**이었다. 통과한 sim 들은 alltools 에서
#   shell 을 **골라** 썼다 ⇒ 둘 다 주고 바꿔 쓰게 한다.
#
# 판정(사전 고정·아침에 이 순서로):
#   ⓐ 배선  `t2_liveness` **3축**(로그=발화 · 사이드카 `arrived`=도달 · 궤적=가시).
#           **도달 위험 0** 이 아니면 성적을 판정에 쓰지 않는다(오늘 이걸 안 봐서 하루를 태웠다).
#   ⓑ 레버  `operator-scope` 발화>0 ∧ `operator-find`(지목) **0** ∧ 중복 write 0
#   ⓒ 산술  072 가 **14.00** 을 크레딧하는가
#   ⓓ 성적  **reward / db_match 로만** 판정한다 — `action_match`(gold N/M)는 소수점 표기로
#           무너진다(C486: 통과한 런에서도 크레딧 3건이 전부 False 였다).
#   ⚠nt=4 = 판정 등급(C467: nt=1·2 는 판정 도구가 아니다) · 잡음 바닥 ±4(C483) — 작은 차이를 신호로 읽지 말 것.
#
# 종료 후: 결과 **즉시 영속화**(gzip → sim_results → `git add -f` → **tracked 확인**·[[30]]).

set -e
REPO=/home/woori/workspace_common/boltzmann-attention-pi
cd "$REPO/scripts/distill/tau2"

NT=4
LOG=/home/woori/scratch/logs
SIMS=/home/woori/scratch/tau2-bench/data/simulations
mkdir -p "$LOG"

SHA=$(cd "$REPO" && git rev-parse --short HEAD)
DIRTY=$(cd "$REPO" && git status --porcelain -- \
  scripts/distill/tau2/t2_gate_patch.py scripts/distill/tau2/t2_search.py \
  scripts/distill/tau2/t2_resolve.py scripts/distill/tau2/t2_compute.py \
  scripts/distill/tau2/t2_scaffold_get.py scripts/distill/tau2/a2/ | grep -cv '^??' || true)
if [ "$DIRTY" != "0" ]; then
  echo "[t7295] REFUSING: 엔진 경로 커밋 안 된 변경 $DIRTY 개." >&2; exit 1
fi

/home/woori/venvs/seka_env/bin/python - <<'PY' || exit 1
import os, subprocess, sys
d = "/home/woori/workspace_common/boltzmann-attention-pi/scripts/distill/tau2"
bad = []
for t in ("test_regen_break_guard.py", "test_no_undefined_names.py", "test_discovery_step2.py",
          "test_write_arg_enum.py", "test_decide_before_write.py", "test_route_trace.py",
          "test_a2_three_layer.py", "test_decision_carry.py", "test_decision_isolate.py",
          "test_axis_levers.py", "test_operator_find.py", "test_action_operator.py",
          "test_atm_fee_op.py", "test_checking_fee_totals.py", "test_c197_inputholes.py",
          "test_slug_disp.py", "test_ownership_fix.py",
          "test_claim_owner_recovery.py",
          "test_dispatch_history_guard.py", "test_resolve_cap_name_reset.py",
          "test_write_initiation_sub.py", "test_subcall_canonical.py",
          "test_write_sub_predraft.py", "test_subcall_return_type.py", "test_action_index.py",
          "test_grounded_calls_nested.py", "test_operator_find_executed.py",
          "test_rebate_netting.py", "test_atm_fee_op.py",
          "test_forensic_canonical.py", "test_omitted_rows_note.py", "test_bailout_axes.py"):
    if not os.path.exists(os.path.join(d, t)):
        continue
    r = subprocess.run(["/home/woori/venvs/seka_env/bin/python", t], cwd=d,
                       capture_output=True, text=True)
    if r.returncode != 0:
        bad.append("%s: %s" % (t, (r.stdout or "")[-140:]))
print("VERIFY " + ("FAIL: " + " · ".join(bad) if bad else "OK"))
sys.exit(1 if bad else 0)
PY

launch () {
  NAME="$1"; TASKS="$2"; PORT="$3"
  TAG="bank_t7295_${NAME}_20260815n"
  if [ -e "$LOG/${TAG}.log" ]; then
    echo "[t7295] SKIP: $LOG/${TAG}.log 가 이미 있다." >&2; return 0
  fi
  if [ -e "$SIMS/${TAG}" ]; then
    echo "[t7295] REFUSING: $SIMS/${TAG} 잔존." >&2; return 1
  fi
  if ps -eo cmd | grep -v grep | grep "t2_run_gated.py" | grep -q "localhost:${PORT}/"; then
    echo "[t7295] REFUSING: 포트 ${PORT} 사용 중." >&2; return 1
  fi
  echo "{\"tag\":\"$TAG\",\"scaffold_sha\":\"$SHA\",\"dirty_files\":$DIRTY,\"tasks\":\"$TASKS\",\"port\":$PORT,\"nt\":$NT,\"arm\":\"on\",\"frozen\":\"t7295_20260813\"}" \
    | tee "$LOG/${TAG}.meta.json"
  setsid bash -c "cd '$REPO/scripts/distill/tau2' && source ./go_stack.sh >/dev/null 2>&1 && \
    export T2_ACTION_SUB=1 T2_KEEP_DENY_BODY=1 T2_CALL_FORM=1 T2_ARG_EMPTY=1 \
           T2_SEARCH_AGENT=1 T2_DECIDE_ANY=1 \
           T2_WRITE_ARG_ENUM=1 T2_DECIDE_BEFORE_WRITE=1 T2_DECISION_CARRY=1 \
           T2_DISCOVERY_STEP2=1 T2_ARG_AXIS=1 T2_WRITE_SUB=3 T2_ACTION_INDEX=1 && \
    t2_launch $TAG $PORT '$TASKS' $NT" </dev/null >"$LOG/${TAG}.log" 2>&1 &
  echo "[t7295] $TASKS → PID=$! port=$PORT log=$LOG/${TAG}.log"
}

launch a task_072,task_074,task_003,task_048,task_050,task_061,task_069,task_085,task_010,task_099,task_070 8140
launch b task_073,task_075,task_036,task_049,task_055,task_063,task_081,task_087,task_098,task_100,task_071 8141
echo "[t7295] 기동 완료 · sha=$SHA · nt=$NT · a=11태스크→8140 · b=11태스크→8141"
