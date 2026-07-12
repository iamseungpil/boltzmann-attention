#!/bin/bash
# S1a-1 write-cap SMOKE — v25c 스택 + T2_WRITE_CAP=1. t0(무회귀)+t102(cap 발화) 2 sims·유료최소.
# GO = [T2_WRITE_CAP] 마커 발화(t102) ∧ 크래시/infra 0 ∧ t0 무회귀. ([[30]] full 전 필수 스모크)
set -u
REPO=/home/woori/workspace_common/boltzmann-attention-pi
T2=$REPO/scripts/distill/tau2; PY=/home/woori/venvs/seka_env/bin/python
S=/home/woori/scratch; TB=/home/woori/scratch/tau2-bench
M="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8"; PORT=8140
exec > $S/t5c_capsmoke.log 2>&1; set -x; date
cd $REPO && git pull --rebase -q origin facet-rft-2026 2>/dev/null
source /home/woori/.openrouter_key
export SSL_CERT_FILE=$($PY -c "import certifi;print(certifi.where())")
export PYTHONPATH=src:$T2
curl -s --max-time 5 localhost:$PORT/v1/models | grep -q "$M" || { echo SERVE_MISSING; exit 1; }
# COMP+D-v2 스택 (v25c 동일) + ★write-cap
export T2_GATE_REGEN=1 T2_GATE_REGEN_K=1 T2_GATE_KINDS=auth,confirm,ownership,notice,preconditions,constraints
export T2_PRESENT_NESTED=1 T2_CALC=1
export T2_PROV_REGEN=1 T2_PROV_REGEN_K=4 T2_PROV_MODE=rescue T2_GROUND=1
export T2_DISAMB=1 T2_DISAMB_MODE=subcall
export T2_WRITE_CAP=1 T2_WRITE_CAP_K=2   # ★S1a-1
unset T2_PRESENT_READS T2_AUTOFETCH T2_PROV_BADWORDS
TASKS="0,102"   # t0=무회귀·t102=19× 루프(cap 발화 기대)
cd $TB; rm -rf "$TB/data/simulations/t5c_capsmoke"
$PY $T2/t2_run_gated.py --gate 1 --domain retail \
  --agent_model "$M" --agent_base http://localhost:$PORT/v1 \
  --user_llm openrouter/openai/gpt-4.1 --user_temp 0.0 \
  --num_trials 1 --max_concurrency 2 --num_tasks 2 --save_to t5c_capsmoke --task_ids "$TASKS" || echo "ARM_FAIL smoke"
echo "SMOKE_RUN_DONE"; date
echo "== [T2_WRITE_CAP] 발화 =="; grep -c "T2_WRITE_CAP. capped" $S/t5c_capsmoke.log || true
grep "T2_WRITE_CAP. capped" $S/t5c_capsmoke.log | head -3 || true
$PY - "$TB/data/simulations/t5c_capsmoke/results.json" <<'PYEOF'
import json,sys
d=json.load(open(sys.argv[1])); s=d["simulations"]
WR=("modify","exchange","return","cancel")
inf=sum(1 for x in s if (x.get("reward_info") or {}).get("reward") is None)
for x in sorted(s,key=lambda z:int(z["task_id"])):
    ri=x.get("reward_info") or {}; dc=ri.get("db_check")
    nw=sum(1 for m in x.get("messages") or [] for tc in (m.get("tool_calls") or []) if any(w in (tc.get("name") or "") for w in WR))
    print("SMOKE t%-4s r=%s db=%s nwrite=%d term=%s"%(x["task_id"],ri.get("reward"),(dc.get("db_match") if isinstance(dc,dict) else None),nw,x.get("termination_reason")))
print("SMOKE_TOTAL n=%d infra=%d"%(len(s),inf))
PYEOF
cd $REPO
gzip -c "$TB/data/simulations/t5c_capsmoke/results.json" > $REPO/reports/facet_rft_2026/sim_results/t5c_capsmoke.results.json.gz 2>/dev/null
git add -f reports/facet_rft_2026/sim_results/t5c_capsmoke.results.json.gz 2>/dev/null
git commit -q -m "persist: t5c_capsmoke (write-cap smoke·auto)" 2>/dev/null
git pull --rebase -q origin facet-rft-2026 2>/dev/null; git push -q origin facet-rft-2026 && echo "PERSISTED_capsmoke"
echo "SMOKE_ALLDONE"; date
