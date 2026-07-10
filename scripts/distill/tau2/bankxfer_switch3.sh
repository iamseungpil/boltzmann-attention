#!/bin/bash
# v3: floor trial2 완료(trial 3 시작 감지·최대 5h) -> 중단 -> trial0+1 salvage(=floor nt2) -> 미니스모크 -> full-stack arm nt=1
set -u
REPO=/home/woori/workspace_common/boltzmann-attention-pi
T2=$REPO/scripts/distill/tau2; PY=/home/woori/venvs/seka_env/bin/python
S=/home/woori/scratch; TB=/home/woori/scratch/tau2-bench
M="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8"; PORT=8142
exec > $S/bankxfer_switch3.log 2>&1; set -x; date
for i in $(seq 1 300); do
  grep -q "(trial 3/4)" $S/bankxfer_full.log 2>/dev/null && break
  sleep 60
done
grep -q "(trial 3/4)" $S/bankxfer_full.log || { echo "TRIAL2_NOT_DONE_5H — abort"; exit 1; }
echo "TRIAL2_DONE"; date
RUN_PID=$(pgrep -f "t2_run_gated.py --gate 0 --domain banking_knowledge" | head -1)
DRV_PID=$(pgrep -f "bankxfer_full_run.sh" | grep -v $$ | head -1)
[ -n "$RUN_PID" ] && kill "$RUN_PID"; sleep 5
[ -n "$RUN_PID" ] && kill -9 "$RUN_PID" 2>/dev/null
[ -n "$DRV_PID" ] && kill "$DRV_PID" 2>/dev/null
echo "KILLED run=${RUN_PID:-none} drv=${DRV_PID:-none}"
mkdir -p $TB/data/simulations/bankxfer_floor_bank_t2
for try in 1 2 3; do
  $PY - <<'PYEOF' && break || sleep 10
import json
from collections import Counter
d = json.load(open("/home/woori/scratch/tau2-bench/data/simulations/bankxfer_floor_bank_t4/results.json"))
sims = [s for s in d.get("simulations", []) if s.get("trial") in (0, 1)]
per = Counter(s.get("trial") for s in sims)
d2 = dict(d); d2["simulations"] = sims
json.dump(d2, open("/home/woori/scratch/tau2-bench/data/simulations/bankxfer_floor_bank_t2/results.json", "w"))
rs = [(s.get("reward_info") or {}).get("reward") for s in sims]
rs2 = [r for r in rs if r is not None]
print("FLOOR_T2_SALVAGED n=%d per_trial=%s mean_r=%.4f pass1=%d infra=%d" % (
    len(sims), dict(per), sum(rs2)/max(len(rs2),1), sum(1 for r in rs2 if r >= 1),
    sum(1 for r in rs if r is None)))
PYEOF
done
gzip -c $TB/data/simulations/bankxfer_floor_bank_t2/results.json > $REPO/reports/facet_rft_2026/sim_results/bankxfer_floor_bank_t2.results.json.gz
source /home/woori/.openrouter_key
[ -f /home/woori/.openai_key ] && source /home/woori/.openai_key
export SSL_CERT_FILE=$($PY -c "import certifi;print(certifi.where())")
export PYTHONPATH=src:$T2
curl -s --max-time 5 localhost:$PORT/v1/models | grep -q "$M" || { echo SERVE_MISSING; exit 1; }
export T2_GATE_REGEN=1 T2_GATE_REGEN_K=1 T2_GATE_KINDS=auth
export T2_PROV_REGEN=1 T2_PROV_REGEN_K=4 T2_DISAMB=1 T2_PRESENT_NESTED=1 T2_CALC=1
unset T2_PRESENT_READS T2_AUTOFETCH T2_PROV_GROUND T2_PROV_BADWORDS
cd $TB; rm -rf "$TB/data/simulations/bankx_uni_smoke"
$PY $T2/t2_run_gated.py --gate 1 --domain banking_knowledge \
  --agent_model "$M" --agent_base http://localhost:$PORT/v1 \
  --user_llm openrouter/openai/gpt-4.1 --user_temp 0.0 \
  --num_trials 1 --max_concurrency 3 --save_to bankx_uni_smoke \
  --task_ids task_004,task_035,task_050 || echo SMOKE_RUN_FAIL
SMOKE_OK=$($PY - <<'PYEOF'
import json
try:
    s = json.load(open("/home/woori/scratch/tau2-bench/data/simulations/bankx_uni_smoke/results.json"))["simulations"]
    inf = sum(1 for x in s if (x.get("reward_info") or {}).get("reward") is None)
    print("OK" if (len(s) == 3 and inf == 0) else "BAD")
except Exception:
    print("BAD")
PYEOF
)
echo "UNI_SMOKE=$SMOKE_OK"
if [ "$SMOKE_OK" != "OK" ]; then
  echo "FALLBACK_GATE_ONLY"
  unset T2_PROV_REGEN T2_DISAMB T2_PRESENT_NESTED T2_CALC
fi
cd $TB; rm -rf "$TB/data/simulations/bankxfer_gate_bank_t1"
$PY $T2/t2_run_gated.py --gate 1 --domain banking_knowledge \
  --agent_model "$M" --agent_base http://localhost:$PORT/v1 \
  --user_llm openrouter/openai/gpt-4.1 --user_temp 0.0 \
  --num_trials 1 --max_concurrency 8 --save_to bankxfer_gate_bank_t1 || echo "ARM_FAIL gate_t1"
echo "GATE_T1_DONE"; date
RES=$TB/data/simulations/bankxfer_gate_bank_t1/results.json
$PY - "$RES" <<'PYEOF'
import json, sys
from collections import Counter
s = json.load(open(sys.argv[1]))["simulations"]
rs = [(x.get("reward_info") or {}).get("reward") for x in s]
rs2 = [r for r in rs if r is not None]
fire = lv = 0
for x in s:
    for m in x.get("messages") or []:
        c = m.get("content")
        if isinstance(c, str) and "blocked by a policy gate" in c: fire += 1
        for tc in (m.get("tool_calls") or []):
            fn = tc.get("function", tc) if isinstance(tc, dict) else tc
            if (fn.get("name") if isinstance(fn, dict) else None) == "log_verification": lv += 1
print("GATE_T1_SUMMARY n=%d mean_r=%.4f pass1=%d infra=%d term=%s strip_notes=%d log_verif=%d" % (
    len(s), sum(rs2)/max(len(rs2),1), sum(1 for r in rs2 if r >= 1),
    sum(1 for r in rs if r is None), dict(Counter(x.get("termination_reason") for x in s)), fire, lv))
PYEOF
echo "== lever census =="
grep -c "T2_PROV. regen fired" $S/bankxfer_switch3.log || true
grep -c "T2_DISAMB. fired" $S/bankxfer_switch3.log || true
grep -c "R8 strip" $S/bankxfer_switch3.log || true
gzip -c "$RES" > $REPO/reports/facet_rft_2026/sim_results/bankxfer_gate_bank_t1.results.json.gz
cd $REPO
git add -f reports/facet_rft_2026/sim_results/bankxfer_floor_bank_t2.results.json.gz reports/facet_rft_2026/sim_results/bankxfer_gate_bank_t1.results.json.gz
git commit -q -m "persist sim results: bankxfer floor nt2 salvage + gate(full-stack) nt1 (auto)" 2>/dev/null
git pull --rebase -q origin facet-rft-2026 2>/dev/null
git push -q origin facet-rft-2026 && echo "PERSISTED_bank_v3"
echo "BANK_V3_ALLDONE"; date