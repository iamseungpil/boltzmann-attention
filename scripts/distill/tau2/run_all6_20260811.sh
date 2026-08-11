#!/bin/bash
# **현황 런** — 6 태스크(010·070·071·098·099·100) × nt=1 을 **한 플래그 집합**으로 훑는다.
#
# 왜 (사용자 지시 2026-08-11 야간: *"두 세션 작업을 합쳐 6 태스크를 nt=1 로 GPU 0·1 모두 써서"*)
#   두 세션이 각각 다른 태스크·다른 플래그로 달려왔다(010·099 ↔ 070·071). 지금 필요한 것은
#   **한 바탕 위의 현황 스냅샷**이다 — 어느 칸이 서 있고 어느 기구가 우는지.
#
# ⚠**이 런으로 성적을 주장하지 않는다.** nt=1 은 통계가 아니다(C403·C415 가 3 sim 으로도 못
#   가른다고 네 번 적었다). 읽을 것은 **P0(기구가 우는가)** 와 **칸 분포**뿐이다.
#   handoff §6-1 축자: *"P0 가 0 이면 '레버 무효'가 아니라 '기구 미발화'다."*
#
# ★기록 (오늘 확인된 결손·C423): 결과 파일의 `git_commit` 은 **tau2-bench 의 SHA** 이고
#   시간당 바뀌는 **우리 스캐폴드 SHA 는 어디에도 안 남는다**. 그래서 여기서 태그 옆에 박는다.
#   이게 없으면 다음 비교도 오늘과 같은 운명이다(플래그 델타 + 코드 17커밋 = 귀속 불가).
#
# 플래그 — 근거 있는 것 전부 ON([[60]] 합성 우선), 근거 없는 둘은 OFF:
#   ON  T2_ACTION_SUB      C406 (넘김 값)
#       T2_KEEP_DENY_BODY  C415 (라이브락 계기를 실제로 산 첫 레버·이번에 0회까지 감소)
#       T2_CALL_FORM       C419 (격리 32/32·닿은 곳에서 2/3 호출)
#       T2_ARG_EMPTY       C420 (표적 매수 확인: 빈 인자 2/12 → 0/9)
#       T2_SEARCH_AGENT    C417/C418 (070·071 · 오프라인 두 축 8/8)
#       T2_DECIDE_ANY      C418 (결정점 진입)
#   OFF T2_UNLOCK_QUIET       C408 판정 = 정당화되지 않음
#       T2_MAIN_ANSWERS_ONLY  라이브 미측정 · 구성이 바뀌므로 별도 팔([[65]])
#
# usage: run_all6_20260811.sh <TASKS> <PORT> <TAG>
set -e
REPO=/home/woori/workspace_common/boltzmann-attention-pi
cd "$REPO/scripts/distill/tau2"

TASKS="${1:?tasks}"
PORT="${2:?port}"
TAG="${3:?tag}"
LOG=/home/woori/scratch/logs
mkdir -p "$LOG"

if ps -eo cmd | grep -v grep | grep "t2_run_gated.py" | grep -q "localhost:${PORT}/"; then
  echo "[run] REFUSING: 포트 ${PORT} 에 이미 런이 있다." >&2; exit 1
fi
if [ -e "$LOG/${TAG}.log" ]; then
  echo "[run] REFUSING: $LOG/${TAG}.log 가 이미 있다." >&2; exit 1
fi

SHA=$(cd "$REPO" && git rev-parse --short HEAD)
DIRTY=$(cd "$REPO" && git status --porcelain -- scripts/distill/tau2 | wc -l)
echo "{\"tag\":\"$TAG\",\"scaffold_sha\":\"$SHA\",\"dirty_files\":$DIRTY,\"tasks\":\"$TASKS\",\"port\":$PORT}" \
  | tee "$LOG/${TAG}.meta.json"
if [ "$DIRTY" != "0" ]; then
  echo "[run] WARNING: 커밋되지 않은 변경 $DIRTY 개 — 이 런은 어떤 SHA 로도 재현되지 않는다." >&2
fi

/home/woori/venvs/seka_env/bin/python - <<'PY' || exit 1
import os, subprocess, sys
d = "/home/woori/workspace_common/boltzmann-attention-pi/scripts/distill/tau2"
bad = []
for t in ("test_call_form_and_arg_empty.py", "test_keep_deny_body.py",
          "test_search_agent_wiring.py", "test_decision_isolate.py"):
    if not os.path.exists(os.path.join(d, t)):
        continue
    r = subprocess.run(["/home/woori/venvs/seka_env/bin/python", t], cwd=d,
                       capture_output=True, text=True)
    if r.returncode != 0:
        bad.append("%s 실패: %s" % (t, (r.stdout or "")[-160:]))
print("VERIFY " + ("FAIL: " + " · ".join(bad) if bad else "OK"))
sys.exit(1 if bad else 0)
PY

setsid bash -c "cd '$REPO/scripts/distill/tau2' && source ./go_stack.sh >/dev/null 2>&1 && \
  export T2_ACTION_SUB=1 T2_KEEP_DENY_BODY=1 T2_CALL_FORM=1 T2_ARG_EMPTY=1 \
         T2_SEARCH_AGENT=1 T2_DECIDE_ANY=1 && t2_launch $TAG $PORT '$TASKS' 1" \
  </dev/null >"$LOG/${TAG}.log" 2>&1 &
echo "PID=$! · sha=$SHA · tasks=$TASKS · port=$PORT · log=$LOG/${TAG}.log"
