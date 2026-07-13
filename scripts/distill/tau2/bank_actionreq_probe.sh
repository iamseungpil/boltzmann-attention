#!/bin/bash
# ★banking action-required 리마인더 채널 라이브 probe (2026-07-13 LATE·[[09]] 승인·최소 scope).
#   목적: 순수-조언 회피 재생성(offline 상한 36%·14/14)이 banking live pass로 이어지나 = make-or-break.
#   ★unified 래퍼(action-required 코드 소재)는 do_prov가 항상 활성 → prov를 격리 못 함.
#   ⇒ prov를 *양 arm 공통*으로 두고 T2_RESOLVE만 차등 = action-required 순수 marginal:
#     g  = unified: auth 게이트 + prov-rescue          (T2_RESOLVE OFF)
#     gr = g + T2_RESOLVE=1                              (action-required + 도구명-날조 검출)
#   marginal(gr−g)@표적 = action-required 인과(prov는 양쪽·상쇄). floor(--gate 0)=all-fail(표적 정의).
#   ※ T2_PROV_REGEN=1이 unified 라우팅 트리거(t2_run_gated _unified 조건)·양 arm 필수.
# 용법: bash bank_actionreq_probe.sh <g|gr> <TAG> <TASKS|ALL> <NT> <PORT>
set -u
MODE="${1:?g|gr}"; TAG="${2:?tag}"; TASKS="${3:?tasks|ALL}"; NT="${4:-1}"; PORT="${5:-8141}"
REPO=/home/woori/workspace_common/boltzmann-attention-pi
T2=$REPO/scripts/distill/tau2; PY=/home/woori/venvs/seka_env/bin/python
S=/home/woori/scratch; TB=$S/tau2-bench
M="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8"; DOM=banking_knowledge
LOG=$S/bankar_${TAG}.log; exec > $LOG 2>&1; set -x; date
cd $REPO && git pull --rebase -q origin facet-rft-2026 2>/dev/null
source /home/woori/.openrouter_key
export SSL_CERT_FILE=$($PY -c "import certifi;print(certifi.where())")
export PYTHONPATH=src:$T2
curl -s --max-time 5 localhost:$PORT/v1/models | grep -q "$M" || { echo SERVE_MISSING_$PORT; exit 1; }
# ── 격리: 표적 외 레버 전부 OFF (disamb/calc/eplan/readall/cons 등) ──
unset T2_CALC T2_EPLAN T2_EPLAN_EXAMINED_SAFE T2_EPLAN_READS_ONLY \
      T2_WRITE_CAP T2_DISAMB T2_READALL T2_CONSISTENCY T2_COV T2_TOOLERR \
      T2_PRESENT_NESTED T2_PRESENT_READS T2_GROUND T2_PRINCIPLE_DEFAULT T2_AUTOFETCH \
      T2_PROV_ADDR_FULL T2_PROV_ORIGIN T2_L4_MODE T2_RESOLVE
# ── 양 arm 공통: auth 게이트 + prov-rescue (prov=unified 라우팅 트리거·양쪽 상쇄) ──
export T2_GATE_REGEN=1 T2_GATE_REGEN_K=1 T2_GATE_KINDS=auth
export T2_PROV_REGEN=1 T2_PROV_REGEN_K=4 T2_PROV_MODE=rescue
[ "$MODE" = "gr" ] && export T2_RESOLVE=1     # ★처치 arm만 action-required/operator-fab
echo "ARM=$MODE  T2_RESOLVE=${T2_RESOLVE:-unset}  PROV=$T2_PROV_MODE  GATE_KINDS=$T2_GATE_KINDS  DOMAIN=$DOM"
cd $TB
TIDARG=""; [ "$TASKS" != "ALL" ] && TIDARG="--task_ids ${TASKS// /,}"
rm -rf "$TB/data/simulations/bankar_$TAG"
timeout 14400 $PY $T2/t2_run_gated.py --gate 1 --domain $DOM --agent_model "$M" \
  --agent_base http://localhost:$PORT/v1 --user_llm openrouter/openai/gpt-4.1 --user_temp 0.0 \
  --num_trials $NT --max_concurrency 8 --save_to "bankar_$TAG" $TIDARG || echo "ARM_FAIL $TAG"
date; echo RUN_DONE
RES="$TB/data/simulations/bankar_$TAG/results.json"
if [ -f "$RES" ]; then
  gzip -c "$RES" > $REPO/reports/facet_rft_2026/sim_results/bankar_$TAG.results.json.gz
  cd $REPO && git add -f reports/facet_rft_2026/sim_results/bankar_$TAG.results.json.gz && \
    git commit -q -m "persist bankar $TAG (auto)" && git pull --rebase -q origin facet-rft-2026 && \
    git push -q origin facet-rft-2026 && echo PERSISTED
fi
echo "== audit =="
echo "action-required fire: $(grep -aE 'T2_RESOLVE\] action-required' $LOG | grep -avc '^+') | resolve deny(operator-fab): $(grep -aE 'T2_RESOLVE\] deny' $LOG | grep -avc '^+')"
$PY - "$RES" <<'PYEOF'
import json, sys, os
from collections import Counter
p = sys.argv[1]
if not os.path.exists(p): print("no results"); raise SystemExit
d = json.load(open(p)); s = d["simulations"]
tasks = {str(t.get("id")): t for t in (d.get("tasks") or [])}
try:
    A2 = json.load(open(os.path.join(os.path.dirname(p), "..", "..", "..", "..",
        "workspace_common", "boltzmann-attention-pi", "scripts", "distill", "tau2",
        "a2", "banking_knowledge.gate.json"), encoding="utf-8"))
except Exception:
    A2 = None
ACTION = set(A2["action_tools"]) if A2 else set()
XFER = {"transfer_to_human_agents", "request_human_agent_transfer"}
inf = sum(1 for x in s if (x.get("reward_info") or {}).get("reward") is None)
term = Counter(x.get("termination_reason") for x in s)
npass = ndeflect_final = nact_final = 0
per = {}
for x in s:
    tid = str(x["task_id"])
    dbm = ((x.get("reward_info") or {}).get("db_check") or {}).get("db_match")
    msgs = x.get("messages") or []
    la = [m for m in msgs if m.get("role") == "assistant"][-1:] or [{}]
    lc = {tc.get("name") for tc in (la[0].get("tool_calls") or [])}
    called = {tc.get("name") for m in msgs if m.get("role") == "assistant" and m.get("tool_calls")
              for tc in m["tool_calls"]}
    final_act = bool(called & ACTION)          # 어느 시점이든 action_tool 실행 시도
    final_deflect = (not lc) or (lc <= XFER)    # 마지막 턴 회피
    npass += 1 if dbm else 0
    nact_final += 1 if final_act else 0
    ndeflect_final += 1 if final_deflect else 0
    per.setdefault(tid, []).append(1 if dbm else 0)
print("SUMMARY n=%d infra=%d pass=%d (%.1f%%) called_action=%d final_deflect=%d term=%s"
      % (len(s), inf, npass, 100*npass/max(len(s),1), nact_final, ndeflect_final, dict(term)))
print("passing tasks:", ",".join(sorted(t for t, r in per.items() if any(r))))
PYEOF
echo AUDIT_DONE
