#!/bin/bash
# 유료 런 — **R8c(잠금-미호출 침묵)**을 표적 두 태스크에서 잰다. 그리고 같은 런이 010 의
# 침묵 원인을 **계기로** 잡는다(`[T2_MATERIAL_GATE]`).
#
# 왜 이 둘만인가 (사용자 지시 2026-08-11: *"099와 010 원인 확정되었으면 수정하고 2개만 nt=3"*)
#   · 099 = 원인 확정. x241(n=8): 궤적만 주면 gold 도구 호출 **8/8**(`A_FREE`) 인데, 우리 층이
#     실제로 넣었던 문장을 되돌리면 **1/8**(`H_LIVE_TRUE`) 이고 문장 하나만 얹어도 4/8(`B_TELL`).
#     격리로 옮기는 길은 먼저 재고 접었다(`E_ISO` 2/8 · 사전상태를 되돌린 `G_ISO_STATE` 6/8 < 8/8).
#   · 010 = **미확정**. 재료는 생산됐고(원장 4행 전사 4/4 · `emitted` 19회) 모델에 닿지 않았다.
#     이 런의 `[T2_MATERIAL_GATE] stop=…` 이 어느 칸에서 멈추는지 턴마다 인쇄한다.
#
# 무엇을 보는가 (사전 등록 · 보기 전에 적는다)
#   P0 팔 오염   `[T2_UNLOCK_QUIET] 억제` 가 이 런에만 있는가 (기준선 런에는 0)
#   P1 성적      099·010 각 3 sim · 기준선 = asubON(099 2/3 · 010 1/3) · isoOFF(3/3 · 0/3)
#   P2 gold 칸   `099_2`(발견 호출)가 실제로 채워지는가 — 이 처방의 표적은 그 칸 하나다
#   P3 Δspurious 침묵이 게이트 거부까지 미루므로 **위반이 늘 수 있다**(§1.3). 거부 수·gold 밖
#                액션 수를 함께 센다. 늘면 이득과 상쇄해 적는다.
#   P4 010 진단  `[T2_MATERIAL_GATE]` 의 stop 사유 분포 (이 런의 **주 산출물**)
#
# ⚠성적 주장 규율: 3 sim×2 태스크는 **못 가른다**. 쓸 수 있는 것은 ⑴`099_2` 가 채워졌는가
#   ⑵Δspurious 가 늘었는가 ⑶010 이 어디서 멈추는가 세 가지고, 총점은 [?] 로 남긴다(C403·C406).
# ⚠태그는 새 것을 쓴다([[30]]: 같은 tag 재런 = 덮어쓰기 + resume 프롬프트에서 죽음).
#
# usage: run_unlockquiet_20260811.sh [TASKS] [NT] [TAG]
set -e
REPO=/home/woori/workspace_common/boltzmann-attention-pi
cd "$REPO/scripts/distill/tau2"

TASKS="${1:-task_099,task_010}"
NT="${2:-3}"
TAG="${3:-bank_uq_20260811}"
LOG=/home/woori/scratch/logs
mkdir -p "$LOG"

if ps -eo cmd | grep -v grep | grep -q "t2_run_gated.py"; then
  echo "[run] REFUSING: t2_run_gated 가 이미 돌고 있다." >&2; exit 1
fi
if [ -e "$LOG/${TAG}.log" ]; then
  echo "[run] REFUSING: $LOG/${TAG}.log 가 이미 있다. 다른 TAG 를 쓰라." >&2; exit 1
fi

# 선-점검: 이 런이 의존하는 것이 **실제로 코드에 있는가** (발사 전 VERIFY OK)
/home/woori/venvs/seka_env/bin/python - <<'PY' || exit 1
import os, subprocess, sys
d = "/home/woori/workspace_common/boltzmann-attention-pi/scripts/distill/tau2"
sys.path.insert(0, d)
bad = []
src = open(os.path.join(d, "t2_gate_patch.py"), encoding="utf-8").read()
if 'os.environ.get("T2_UNLOCK_QUIET") == "1"' not in src:
    bad.append("R8c 코드가 없다")
if "[T2_MATERIAL_GATE]" not in src:
    bad.append("재료 정지-사유 계기가 없다")
beat = open(os.path.join(d, "t2_lever_beat.py"), encoding="utf-8").read()
if "def set_turn" not in beat or '"turn"' not in beat:
    bad.append("턴 계기가 없다")
for t in ("test_unlock_quiet.py", "test_decision_isolate.py", "test_sim_tag_thread_local.py"):
    r = subprocess.run(["/home/woori/venvs/seka_env/bin/python", t], cwd=d,
                       capture_output=True, text=True)
    if r.returncode != 0:
        bad.append("%s 실패: %s" % (t, (r.stdout or "")[-200:]))
if os.environ.get("T2_UNLOCK_QUIET") == "1":
    bad.append("검증 프로세스에 플래그가 켜져 있다(런처가 켜야 한다)")
print("VERIFY " + ("FAIL: " + " · ".join(bad) if bad else "OK"))
sys.exit(1 if bad else 0)
PY

setsid bash -c "cd '$REPO/scripts/distill/tau2' && source ./go_stack.sh >/dev/null 2>&1 && \
  export T2_ACTION_SUB=1 T2_UNLOCK_QUIET=1 && t2_launch $TAG 8140 '$TASKS' $NT" \
  </dev/null >"$LOG/${TAG}.log" 2>&1 &
echo "PID=$!"
sleep 12
echo "--- 발사 직후 ---"
head -12 "$LOG/${TAG}.log" 2>/dev/null || true
echo "launched · tasks=$TASKS nt=$NT · log: $LOG/${TAG}.log"
echo "  sidecar: $LOG/fb_${TAG}.jsonl · trace: $LOG/trace_${TAG}.jsonl"
