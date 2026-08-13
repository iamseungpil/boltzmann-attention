#!/bin/bash
# **밤샘 체인 2026-08-14** (사용자 지시: "4개 태스크 런 끝나고 밤샘 런 할 실험 돌려라").
#
# 단계:
#   0) 현재 러너(t7284) 완주 대기 → 영속화 (결과 소실 방지·[[30]])
#   1) **t7285 nt=4** — 072~075 확정런. 이번 세션 레버 6종((B2)·FIX-5/6/7/8/9/10/11/12)의
#      robust 판정. nt=1 은 판정 도구가 아님이 실증됨(074 9/13→0/13).
#   2) **t7286 nt=1** — G별 대표 12태스크 관측런(C466 확정 대표):
#      G1 085/081/087 · G3 050/049/048 · G2 055/061/069 · G9 003/036/063.
#      pass 주장용이 아니라 **현 스택에서 어디서 무너지는지** 첫 자료(다음 세션 표적 선정).
#   3) 각 단계마다 즉시 영속화+push · 요약을 `_OVERNIGHT_SUMMARY.txt` 에 기록.
#
# 안전: 단계마다 VERIFY(검정 18종) · 포트 점유 확인 · 잔존 디렉터리 거부 · setsid 백그라운드.
# usage: setsid bash run_overnight_20260814.sh </dev/null >$LOG/overnight.log 2>&1 &
set -u
REPO=/home/woori/workspace_common/boltzmann-attention-pi
SIMS=/home/woori/scratch/tau2-bench/data/simulations
LOG=/home/woori/scratch/logs
DST=$REPO/reports/facet_rft_2026/sim_results
SUM=$REPO/reports/facet_rft_2026/_OVERNIGHT_SUMMARY.txt
PY=/home/woori/venvs/seka_env/bin/python
cd "$REPO/scripts/distill/tau2"

say () { echo "[$(date +%H:%M:%S)] $*" | tee -a "$SUM"; }

wait_idle () {
  for i in $(seq 1 480); do
    if ! ps -eo cmd | grep -v grep | grep -q t2_run_gated.py; then return 0; fi
    sleep 30
  done
  say "WARN: 러너가 4시간 넘게 안 끝남 — 다음 단계 강행하지 않고 종료"; exit 1
}

persist () {   # $1.. = tag 들
  cd "$REPO"
  for T in "$@"; do
    [ -f "$SIMS/$T/results.json" ] || { say "persist skip(없음): $T"; continue; }
    gzip -c "$SIMS/$T/results.json" > "$DST/${T}_results.json.gz"
    [ -f "$LOG/$T.log" ] && gzip -c "$LOG/$T.log" > "$DST/${T}.log.gz"
    for S in fb trace route; do
      [ -f "$LOG/${S}_${T}.jsonl" ] && gzip -c "$LOG/${S}_${T}.jsonl" > "$DST/${S}_${T}.jsonl.gz"
    done
    git add -f "$DST"/*"${T}"* 2>/dev/null
  done
  git pull --rebase -q origin facet-rft-2026 2>/dev/null
  git commit -q -m "Persist overnight: $*" >/dev/null 2>&1 && git push -q origin facet-rft-2026 \
    && say "persisted+pushed: $*" || say "persist: nothing to commit ($*)"
  # tracked 확인까지가 절차([[30]])
  for T in "$@"; do
    git ls-files --error-unmatch "$DST/${T}_results.json.gz" >/dev/null 2>&1 \
      && say "  tracked OK: $T" || say "  ⚠TRACKED 실패: $T"
  done
  cd "$REPO/scripts/distill/tau2"
}

judge () {     # $1.. = tag 들
  $PY - "$@" <<'PYEOF' | tee -a "$SUM"
import json, io, sys
sys.stdout.reconfigure(encoding="utf-8")
for tag in sys.argv[1:]:
    try:
        d = json.load(io.open("/home/woori/scratch/tau2-bench/data/simulations/%s/results.json" % tag,
                              encoding="utf-8"))
    except Exception as e:
        print("  %s READ FAIL %s" % (tag, e)); continue
    for sim in (d.get("simulations") or d.get("results") or d):
        ri = sim.get("reward_info") or {}
        ac = ri.get("action_checks") or []
        g = "".join("1" if a.get("action_match") else "0" for a in ac)
        print("  %-9s t%-2s reward=%.1f gold=%-14s match=%2d/%-2d msgs=%3d term=%s" % (
            sim.get("task_id"), sim.get("trial"), ri.get("reward") or 0, g,
            g.count("1"), len(g), len(sim.get("messages") or []),
            sim.get("termination_reason")))
PYEOF
}

verify () {
  $PY - <<'PYEOF'
import os, subprocess, sys
d = "/home/woori/workspace_common/boltzmann-attention-pi/scripts/distill/tau2"
bad = []
for t in ("test_regen_break_guard.py", "test_no_undefined_names.py", "test_discovery_step2.py",
          "test_write_arg_enum.py", "test_decide_before_write.py", "test_route_trace.py",
          "test_a2_three_layer.py", "test_decision_carry.py", "test_decision_isolate.py",
          "test_axis_levers.py", "test_operator_find.py", "test_action_operator.py",
          "test_atm_fee_op.py", "test_checking_fee_totals.py", "test_c197_inputholes.py",
          "test_slug_disp.py", "test_ownership_fix.py", "test_claim_owner_recovery.py"):
    if not os.path.exists(os.path.join(d, t)):
        continue
    r = subprocess.run(["/home/woori/venvs/seka_env/bin/python", t], cwd=d,
                       capture_output=True, text=True)
    if r.returncode != 0:
        bad.append(t)
print("VERIFY " + ("FAIL: " + " ".join(bad) if bad else "OK"))
sys.exit(1 if bad else 0)
PYEOF
}

launch () {    # $1=tag $2=tasks $3=port $4=nt
  local TAG="$1" TASKS="$2" PORT="$3" NT="$4"
  if [ -e "$SIMS/$TAG" ]; then say "REFUSING(잔존): $TAG"; return 1; fi
  if ps -eo cmd | grep -v grep | grep "t2_run_gated.py" | grep -q "localhost:${PORT}/"; then
    say "REFUSING(포트 사용중 $PORT): $TAG"; return 1
  fi
  local SHA; SHA=$(cd "$REPO" && git rev-parse --short HEAD)
  echo "{\"tag\":\"$TAG\",\"scaffold_sha\":\"$SHA\",\"tasks\":\"$TASKS\",\"port\":$PORT,\"nt\":$NT,\"arm\":\"on\"}" \
    > "$LOG/${TAG}.meta.json"
  setsid bash -c "cd '$REPO/scripts/distill/tau2' && source ./go_stack.sh >/dev/null 2>&1 && \
    export T2_ACTION_SUB=1 T2_KEEP_DENY_BODY=1 T2_CALL_FORM=1 T2_ARG_EMPTY=1 \
           T2_SEARCH_AGENT=1 T2_DECIDE_ANY=1 \
           T2_WRITE_ARG_ENUM=1 T2_DECIDE_BEFORE_WRITE=1 T2_DECISION_CARRY=1 \
           T2_DISCOVERY_STEP2=1 T2_ARG_AXIS=1 && \
    t2_launch $TAG $PORT '$TASKS' $NT" </dev/null >"$LOG/${TAG}.log" 2>&1 &
  say "launched $TAG · tasks=$TASKS · port=$PORT · nt=$NT · sha=$SHA"
}

# ── 0) 현재 런(t7284) 완주 대기 → 영속화·판정 ─────────────────────────────
say "=== 밤샘 체인 시작 · HEAD=$(cd $REPO && git rev-parse --short HEAD) ==="
wait_idle
say "--- [0] t7284 완주 · 판정 ---"
persist bank_t7284_a_20260814f bank_t7284_b_20260814f
judge   bank_t7284_a_20260814f bank_t7284_b_20260814f

# ── 1) t7285 nt=4 확정런 (072~075) ────────────────────────────────────────
verify || { say "VERIFY 실패 — 체인 중단"; exit 1; }
say "--- [1] t7285 nt=4 (072~075 확정런) ---"
launch bank_t7285_a_20260814g "task_072,task_074" 8140 4
launch bank_t7285_b_20260814g "task_073,task_075" 8141 4
sleep 60
wait_idle
persist bank_t7285_a_20260814g bank_t7285_b_20260814g
judge   bank_t7285_a_20260814g bank_t7285_b_20260814g

# ── 2) t7286 nt=1 G별 대표 관측런 ─────────────────────────────────────────
verify || { say "VERIFY 실패 — 2단계 중단"; exit 1; }
say "--- [2] t7286 nt=1 (G별 대표 12태스크 · 관측용) ---"
launch bank_t7286_a_20260814h "task_085,task_081,task_087,task_050,task_049,task_048" 8140 1
launch bank_t7286_b_20260814h "task_055,task_061,task_069,task_003,task_036,task_063" 8141 1
sleep 60
wait_idle
persist bank_t7286_a_20260814h bank_t7286_b_20260814h
judge   bank_t7286_a_20260814h bank_t7286_b_20260814h

say "=== 밤샘 체인 완료 ==="
