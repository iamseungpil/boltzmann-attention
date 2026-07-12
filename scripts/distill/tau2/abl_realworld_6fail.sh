#!/bin/bash
# 실세계 분포측정: 6-fail{1,6,7,23,75,106} × 3스택 × nt=4 = 72 sim. 비결정성 커버(rate+CI).
# stacks: comp(base) / full(=b78c+cap·개입레버 전부) / guard(=comp+prov-rescue+cap·가드만·개입레버 0).
# 목표: full이 trivial-fail을 기대값서 회귀시키나 ∧ guard-only는 COMP만큼 안전한가.
set -u
REPO=/home/woori/workspace_common/boltzmann-attention-pi
T2=$REPO/scripts/distill/tau2; PY=/home/woori/venvs/seka_env/bin/python
S=/home/woori/scratch; TB=$S/tau2-bench
M="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8"; PORT=8140
TASKS="1,6,7,23,75,106"; NT=4
LOG=$S/abl_realworld_6fail.log; exec > $LOG 2>&1; set -x; date
cd $REPO && git pull --rebase -q origin facet-rft-2026 2>/dev/null
source /home/woori/.openrouter_key
export SSL_CERT_FILE=$($PY -c "import certifi;print(certifi.where())")
export PYTHONPATH=src:$T2
curl -s --max-time 5 localhost:$PORT/v1/models | grep -q "$M" || { echo SERVE_MISSING; exit 1; }
cd $TB

run_stack(){
  local tag="$1"; local extra="$2"
  (
    export T2_GATE_REGEN=1 T2_GATE_REGEN_K=1 T2_GATE_KINDS=auth,confirm,ownership,notice,preconditions,constraints
    export T2_PROV_REGEN=1 T2_PROV_REGEN_K=4 T2_PRESENT_NESTED=1 T2_CALC=1
    unset T2_PROV_MODE T2_GROUND T2_DISAMB T2_DISAMB_MODE T2_PRINCIPLE_DEFAULT T2_EPLAN T2_EPLAN_WALK T2_WRITE_CAP T2_WRITE_CAP_K T2_PRESENT_READS T2_AUTOFETCH T2_PROV_BADWORDS
    eval "$extra"
    rm -rf "$TB/data/simulations/bstack_$tag"
    $PY $T2/t2_run_gated.py --gate 1 --domain retail --agent_model "$M" --agent_base http://localhost:$PORT/v1 \
      --user_llm openrouter/openai/gpt-4.1 --user_temp 0.0 --num_trials $NT --max_concurrency 8 \
      --save_to "bstack_$tag" --task_ids "$TASKS" && echo "STACK_DONE $tag"
  ) &
}

run_stack comp  ":"
run_stack full  "export T2_PROV_MODE=rescue T2_GROUND=1 T2_DISAMB=1 T2_DISAMB_MODE=subcall T2_PRINCIPLE_DEFAULT=1 T2_EPLAN=1 T2_EPLAN_WALK=1 T2_WRITE_CAP=1 T2_WRITE_CAP_K=2"
run_stack guard "export T2_PROV_MODE=rescue T2_WRITE_CAP=1 T2_WRITE_CAP_K=2"
wait
date; echo ALL_STACK_DONE

# persist (한 번에·순차 커밋·[[30]])
for tag in comp full guard; do
  src="$TB/data/simulations/bstack_$tag/results.json"
  [ -f "$src" ] && gzip -c "$src" > $REPO/reports/facet_rft_2026/sim_results/bstack_$tag.results.json.gz
done
cd $REPO && git add -f reports/facet_rft_2026/sim_results/bstack_*.results.json.gz 2>/dev/null
git commit -q -m "persist: 실세계 6-fail 3스택 nt4 (comp/full/guard·auto)" 2>/dev/null
git pull --rebase -q origin facet-rft-2026 2>/dev/null; git push -q origin facet-rft-2026 && echo PERSISTED

# summary: per-stack per-task db pass-rate over nt trials
$PY - <<'PYEOF'
import json,os
from collections import defaultdict
TB="/home/woori/scratch/tau2-bench/data/simulations"
tasks=["1","6","7","23","75","106"]
res={}
for tag in ["comp","full","guard"]:
    p=f"{TB}/bstack_{tag}/results.json"
    if not os.path.exists(p): print(tag,"MISSING"); continue
    d=json.load(open(p)); sims=d["simulations"]
    per=defaultdict(list)
    for s in sims:
        ri=s.get("reward_info") or {}
        db=1 if (ri.get("db_check") or {}).get("db_match") else 0
        per[str(s["task_id"])].append(db)
    res[tag]=per
print("=== db pass-rate per task (n=%d trials) ==="%4)
print("%-6s | %-14s | %-14s | %-14s"%("task","comp","full","guard"))
for t in tasks:
    row=[]
    for tag in ["comp","full","guard"]:
        v=res.get(tag,{}).get(t,[])
        row.append("%d/%d=%.2f"%(sum(v),len(v),(sum(v)/len(v) if v else 0)))
    print("%-6s | %-14s | %-14s | %-14s"%(t,row[0],row[1],row[2]))
print("--- overall (36 sims/stack) ---")
for tag in ["comp","full","guard"]:
    allv=[x for t in tasks for x in res.get(tag,{}).get(t,[])]
    print("  %-6s db-pass %d/%d = %.3f"%(tag,sum(allv),len(allv),(sum(allv)/len(allv) if allv else 0)))
print("\nGO if: full < comp (개입레버 회귀 실재) AND guard ~= comp (가드-only 안전)")
PYEOF
echo ALLDONE; date
