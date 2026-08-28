#!/bin/bash
# **6태스크 판정 런** (저녁·사용자 지시 2026-08-12) — nt=4 · ON 팔만 · **동결**.
# 사전등록 = C438 top-up 규칙 그대로. 대조군은 돌리지 않는다 — 기준선 =
# `bank_dbw_off_20260812`(070/071 nt=4) + `bank_batch4_20260812`(010/098/099/100 nt=4).
#
# 이 런이 판정하는 것:
#   · 070/071: 오늘 스택(ENUM/AXIS/STEP2/CARRY/DBW)의 nt=4 성적 — j런(nt=2)은 0/2였으나
#     실패 모양이 이동(이름 날조 닫힘·잔여=중복 개설/None-인자/오진).
#   · 010/099: j런 개선(1/2·2/2)이 nt=4 에서 유지되는가.
#   · 098/100: **무회귀 검사** — batch4 4/4·3/4 를 오늘 레버가 깨지 않는가(over-block·Δspurious).
#
# ★★밤샘 재런 (2026-08-13·사용자 지시 "010 070 071 포렌식하고 수정하고 재런하라").
#   델타 = 010/070/071 재포렌식이 찾은 **우리 층 결함 5건 수리**(커밋 72c3f8e6). 전부
#   기존 배관·문구 버그이고 신규 레버 0:
#     [1] _ufb 덮어쓰기로 결정 턴에 [ACTION] 소유권 문장이 소멸(010 3/4 실패의 단일 기전)
#     [2] 센티널 "omit"을 실명으로 받아 "그 도구는 존재하지 않는다" 거짓 발화(071 t0 26턴 날조)
#     [3] record_update 매핑이 디스패처 실행을 못 봐 실행한 개설을 "없다"고 단정(070 t3 중복 write)
#     [4] 축별 결정 단일 슬롯 덮어쓰기로 Sky Blue 가 재제시되지 않음(071 t1 True Blue)
#     [5] ENUM deny 가 후보만 주고 저장된 결정을 안 실음(오답을 매끄럽게 확정)
#   읽을 것: 기준선 11/24 · judge6 5/24 · 재판정 11/24 와 4열 대조. 특히 010/070/071 이동과
#     098/099/100 무회귀(over-block·Δspurious 0).
#
# ★j런 포렌식 블로킹 7건 수리 반영 (원장 예정·커밋 참조):
#   [1] STEP2 스코프(formalize NONE·후보-정합 LLM 선택·이관-실행 후 침묵)
#   [2] OPERATOR-SELECT 완료-검사(실행 성공한 want 재지시 침묵)
#   [3] UNAVAILABLE-CAPABILITY 잠김-분기(레지스트리 실재 시 '없다' 금지·feedback_unavailable_locked)
#   [4] 잘못-접미사 deny(feedback_wrong_suffix — 'there is none' 거짓 금지문 대체·소비처 2곳)
#   [5] GB2 recovery 자기모순 제거(3파일 byte-동일 동기·split/core 기존 비동기 치유)
#   [6] ARG-AXIS 창(첫 발화 포함)·[7] [SOURCE] 다중행 레코드 블록 대조(하드 2앵커)
#   추가 관측: [T2_DISCOVERY_STEP2] "정합 없음(none)" · [T2_UNAVAIL] locked 분리 집계.
#
# 읽을 것: ①태스크별 pass ②ENUM deny→집합內 재작성 교정률 ③over-block(집합內 이름 거부) 0
#   ④[T2_ACTIONREQ] 침묵-사유 분포 ⑤Δspurious(_FB_GENERIC·거부 수·gold 밖 쓰기) ⑥건너뜀 0.
#
# usage: run_judge6q_20260813q.sh
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
  scripts/distill/tau2/a2/ | grep -cv '^??' || true)
if [ "$DIRTY" != "0" ]; then
  echo "[judge6] REFUSING: 엔진 경로 커밋 안 된 변경 $DIRTY 개 — 판정 런은 동결 필수." >&2; exit 1
fi

/home/woori/venvs/seka_env/bin/python - <<'PY' || exit 1
import os, subprocess, sys
d = "/home/woori/workspace_common/boltzmann-attention-pi/scripts/distill/tau2"
bad = []
for t in ("test_regen_break_guard.py",
          "test_no_undefined_names.py",
          "test_discovery_step2.py",
          "test_write_arg_enum.py", "test_decide_before_write.py", "test_route_trace.py",
          "test_a2_three_layer.py", "test_decision_carry.py", "test_decision_isolate.py",
          "test_axis_levers.py",
          "test_operator_find.py", "test_action_operator.py"):
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
  TAG="bank_judge6q_${NAME}_20260813q"
  if [ -e "$LOG/${TAG}.log" ]; then
    echo "[judge6] SKIP: $LOG/${TAG}.log 가 이미 있다." >&2; return 0
  fi
  if [ -e "$SIMS/${TAG}" ]; then
    echo "[judge6] REFUSING: $SIMS/${TAG} 가 이미 있다 — 지우고 다시 걸어라." >&2; return 1
  fi
  if ps -eo cmd | grep -v grep | grep "t2_run_gated.py" | grep -q "localhost:${PORT}/"; then
    echo "[judge6] REFUSING: 포트 ${PORT} 사용 중." >&2; return 1
  fi
  echo "{\"tag\":\"$TAG\",\"scaffold_sha\":\"$SHA\",\"dirty_files\":$DIRTY,\"tasks\":\"$TASKS\",\"port\":$PORT,\"nt\":$NT,\"arm\":\"on\",\"frozen\":\"judge6q_20260813\"}" \
    | tee "$LOG/${TAG}.meta.json"
  setsid bash -c "cd '$REPO/scripts/distill/tau2' && source ./go_stack.sh >/dev/null 2>&1 && \
    export T2_ACTION_SUB=1 T2_KEEP_DENY_BODY=1 T2_CALL_FORM=1 T2_ARG_EMPTY=1 \
           T2_SEARCH_AGENT=1 T2_DECIDE_ANY=1 \
           T2_WRITE_ARG_ENUM=1 T2_DECIDE_BEFORE_WRITE=1 T2_DECISION_CARRY=1 \
           T2_DISCOVERY_STEP2=1 T2_ARG_AXIS=1 && \
    t2_launch $TAG $PORT '$TASKS' $NT" </dev/null >"$LOG/${TAG}.log" 2>&1 &
  echo "[judge6] $TASKS → PID=$! port=$PORT log=$LOG/${TAG}.log"
}

launch a task_070,task_071,task_098 8140
launch b task_010,task_099,task_100 8141
echo "[judge6] 기동 완료 · sha=$SHA · nt=$NT · 동결(frozen=judge6q_20260813)"
