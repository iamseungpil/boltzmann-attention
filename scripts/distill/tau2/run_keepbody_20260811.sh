#!/bin/bash
# 유료 런 — **R9(접힘이 deny 본문을 지우지 않는다)** 를 표적 두 태스크에서 잰다.
#
# 왜 (사용자 지시 2026-08-11: *"R9 켜고 다시 099 010 돌려라"*)
#   C413: `resolve the flagged call` 이 한 sim 에 3회 이상 나온 6건은 **6/6 전부 실패**(3 런·30 sim).
#   C414: 인과 확정 — 같은 문맥에서 그 문구 3회 = 정체 3/8 · 1회 = 2/8 · **원본 본문 0/8** ·
#         **아무것도 안 냄 0/8**. 반복 횟수가 아니라 **문구 자체**가 정체를 만든다.
#   ⇒ 접힐 때 본문을 갈아 끼우지 않는다(도구는 여전히 막힌다·fail-closed).
#
# 귀속 (★플래그 **한 개** 차이)
#   직전 런 `bank_uq_20260811` = T2_ACTION_SUB + T2_UNLOCK_QUIET  → 3/6 (010 2/3 · 099 1/3)
#   이 런              = 위 + **T2_KEEP_DENY_BODY**              → 델타 = R9 하나
#
# 사전 등록 (보기 전에 적는다)
#   P0 팔 오염   `body kept (R9)` 마크가 이 런에만 있는가 · `_FB_GENERIC` 발화 수가 **줄었는가**
#   P1 성적      태스크별 pass (기준 010 2/3 · 099 1/3)
#   P2 표적 계기 **`resolve the flagged call` 발화 수**(C413 의 그 수). 3회↑ sim 이 사라지는가 —
#                이것이 이 레버가 직접 겨눈 유일한 수다. 성적보다 **이 수를 먼저** 읽는다.
#   P3 넘김 발화 도구 이름+값이 한 메시지에 있는 턴(C412) · 그 다음 턴에 손님이 실행하는가
#   P4 Δspurious 게이트 거부 수 · gold 밖 쓰기 호출 (본문을 살리면 위반이 늘 수 있다·§1.3)
#
# ⚠3 sim×2 태스크는 총점을 **못 가른다**(C403·C406·C408). 쓸 수 있는 것은 P2·P3 의 계수다.
# ⚠태그는 새 것을 쓴다([[30]]).
#
# usage: run_keepbody_20260811.sh [TASKS] [NT] [TAG]
set -e
REPO=/home/woori/workspace_common/boltzmann-attention-pi
cd "$REPO/scripts/distill/tau2"

TASKS="${1:-task_099,task_010}"
NT="${2:-3}"
TAG="${3:-bank_kb_20260811}"
LOG=/home/woori/scratch/logs
mkdir -p "$LOG"

if ps -eo cmd | grep -v grep | grep -q "t2_run_gated.py"; then
  echo "[run] REFUSING: t2_run_gated 가 이미 돌고 있다." >&2; exit 1
fi
if [ -e "$LOG/${TAG}.log" ]; then
  echo "[run] REFUSING: $LOG/${TAG}.log 가 이미 있다. 다른 TAG 를 쓰라." >&2; exit 1
fi

/home/woori/venvs/seka_env/bin/python - <<'PY' || exit 1
import os, subprocess, sys
d = "/home/woori/workspace_common/boltzmann-attention-pi/scripts/distill/tau2"
sys.path.insert(0, d)
bad = []
src = open(os.path.join(d, "t2_gate_patch.py"), encoding="utf-8").read()
if 'os.environ.get("T2_KEEP_DENY_BODY") == "1"' not in src:
    bad.append("R9 코드가 없다")
if "kept (R9)" not in src:
    bad.append("R9 마크가 없다(팔 오염 검사를 못 한다)")
if '_FB_GENERIC = "Error: resolve the flagged call' not in src:
    bad.append("OFF 경로 문구가 사라졌다(되돌리기 불가)")
for t in ("test_keep_deny_body.py", "test_unlock_quiet.py", "test_decision_isolate.py"):
    r = subprocess.run(["/home/woori/venvs/seka_env/bin/python", t], cwd=d,
                       capture_output=True, text=True)
    if r.returncode != 0:
        bad.append("%s 실패: %s" % (t, (r.stdout or "")[-200:]))
if os.environ.get("T2_KEEP_DENY_BODY") == "1":
    bad.append("검증 프로세스에 플래그가 켜져 있다(런처가 켜야 한다)")
print("VERIFY " + ("FAIL: " + " · ".join(bad) if bad else "OK"))
sys.exit(1 if bad else 0)
PY

setsid bash -c "cd '$REPO/scripts/distill/tau2' && source ./go_stack.sh >/dev/null 2>&1 && \
  export T2_ACTION_SUB=1 T2_UNLOCK_QUIET=1 T2_KEEP_DENY_BODY=1 && t2_launch $TAG 8140 '$TASKS' $NT" \
  </dev/null >"$LOG/${TAG}.log" 2>&1 &
echo "PID=$!"
sleep 12
echo "--- 발사 직후 ---"
head -12 "$LOG/${TAG}.log" 2>/dev/null || true
echo "launched · tasks=$TASKS nt=$NT · log: $LOG/${TAG}.log"
echo "  sidecar: $LOG/fb_${TAG}.jsonl · trace: $LOG/trace_${TAG}.jsonl"
