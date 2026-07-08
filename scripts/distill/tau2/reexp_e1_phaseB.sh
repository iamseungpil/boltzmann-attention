#!/bin/bash
# E1 Phase B (2026-07-08) — 완결/persistence 게이트 mode=inspect의 CLOSED 측정.
# 무료: user-sim+judge = 로컬 14B(8140). agent = QwQ+reasoning-parser(8141).
# ★짝 설계: 양 arm 모두 auth 게이트 ON(=auth_user 확립 필요). 차이 = exhaust_before_escalate 하나뿐.
#   arm-OFF: T2_GATE_KINDS=auth        arm-ON: T2_GATE_KINDS=auth,exhaust_before_escalate
# 태스크: fail-set{13,30} + risk{50=gold transfer} + control{10,12,25,46,57,67,68}
# 측정: 건별 복구 · Δspurious(비요청 write) · turn 예산 · t50 재-transfer 복귀 · over-block(closed)
set -u
REPO=/home/woori/workspace_common/boltzmann-attention-pi
T2=$REPO/scripts/distill/tau2; PY=/home/woori/venvs/seka_env/bin/python
S=/home/woori/scratch; TB=$S/tau2-bench
AGENT=Qwen/QwQ-32B-AWQ; ABASE=http://localhost:8141/v1
USIM=Qwen/Qwen2.5-14B-Instruct; UBASE=http://localhost:8140/v1
TASKS=13,30,50,10,12,25,46,57,67,68
LOG=$S/e1_phaseB.log
exec > $LOG 2>&1; set -x; date
cd $REPO && git pull --ff-only 2>&1 | tail -1
export PYTHONPATH=src:$T2
export SSL_CERT_FILE=$($PY -c "import certifi;print(certifi.where())")
curl -s localhost:8141/v1/models | grep -q QwQ || { echo AGENT_DOWN; exit 1; }
curl -s localhost:8140/v1/models | grep -q 14B || { echo USIM_DOWN; exit 1; }
echo "SERVERS_OK (gpt-4.1 = 0)"; date

run_arm () {  # $1=tag  $2=T2_GATE_KINDS
  local TAG=$1 KINDS=$2 SAVE=e1pb_$1
  cd $TB; rm -rf "$TB/data/simulations/$SAVE"
  T2_GATE_REGEN=1 T2_GATE_REGEN_K=2 T2_GATE_KINDS="$KINDS" \
  $PY $T2/t2_run_gated.py --gate 1 --domain retail \
    --agent_model "$AGENT" --agent_base "$ABASE" \
    --user_model "$USIM" --user_base "$UBASE" \
    --num_trials 1 --task_ids "$TASKS" --max_concurrency 2 --save_to "$SAVE" \
    || echo "ARM_FAIL $TAG"
  echo "ARM_DONE $TAG"; date
}
run_arm off "auth"
run_arm on  "auth,exhaust_before_escalate"

# ── 분석: Δspurious · 복구 · turns · transfer 결과 ──
$PY - <<'PYEOF'
import json
WRITE={"return_delivered_order_items","exchange_delivered_order_items","cancel_pending_order",
       "modify_pending_order_items","modify_pending_order_address","modify_pending_order_payment",
       "modify_user_address","place_order"}
def load(tag):
    p=f"/home/woori/scratch/tau2-bench/data/simulations/e1pb_{tag}/results.json"
    return json.load(open(p))["simulations"]
def calls(s):
    return [(tc.get("name"),tc.get("arguments") or {}) for m in s.get("messages",[])
            if m.get("role")=="assistant" for tc in (m.get("tool_calls") or [])]
def spurious(s):
    ri=s.get("reward_info") or {}
    gold={(a.get("action") or {}).get("name") for a in (ri.get("action_checks") or [])}
    return sum(1 for n,_ in calls(s) if n in WRITE and n not in gold)
def turns(s): return sum(1 for m in s.get("messages",[]) if m.get("role")=="assistant")
def xfer(s): return any(n=="transfer_to_human_agents" for n,_ in calls(s))
def rew(s): return (s.get("reward_info") or {}).get("reward")
try:
    OFF={str(s["task_id"]):s for s in load("off")}; ON={str(s["task_id"]):s for s in load("on")}
except Exception as e:
    print("ANALYZE_FAIL",e); raise SystemExit
FAIL={'13','30'}; RISK={'50'}
print("\n===== E1 PHASE B (closed) =====")
print(f"{'task':>5} {'set':<8} {'rew off→on':<12} {'spur off→on':<12} {'turns off→on':<13} {'xfer off→on'}")
dsp=0; rec=0; ob=0
for t in sorted(OFF, key=int):
    if t not in ON: continue
    o,n=OFF[t],ON[t]
    grp = "FAIL" if t in FAIL else ("RISK" if t in RISK else "control")
    so,sn=spurious(o),spurious(n); dsp+=sn-so
    if grp=="FAIL" and rew(o)!=1 and rew(n)==1: rec+=1
    if grp not in ("FAIL",) and rew(o)==1 and rew(n)!=1: ob+=1
    print(f"{t:>5} {grp:<8} {str(rew(o))+'→'+str(rew(n)):<12} {str(so)+'→'+str(sn):<12} "
          f"{str(turns(o))+'→'+str(turns(n)):<13} {str(xfer(o))+'→'+str(xfer(n))}")
print(f"\nRESULT_E1PB  d_spurious={dsp}  recovered(FAIL)={rec}/2  over_block(closed)={ob}")
print(f"GO?  d_spurious<=0: {dsp<=0} · over_block==0: {ob==0} · recovered>0: {rec>0}")
PYEOF
touch $S/e1pb_end; echo "E1_PHASEB_DONE"; date
