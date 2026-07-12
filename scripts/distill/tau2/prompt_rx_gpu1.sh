#!/bin/bash
# prompt-rx arm (HOW=advise·WHEN=always·WHAT=all-5): COMP엔진 + 5처방을 rules_prompt로.
# D-v2 엔진레버 OFF. GPU1(port 8141)서 B(GPU0)와 병렬. 6-fail nt4. 비교=comp/full(B).
# 판독: prompt-rx가 trivial 무회귀면 부작용원=override 기전(advise가 해소).
set -u
REPO=/home/woori/workspace_common/boltzmann-attention-pi
T2=$REPO/scripts/distill/tau2; PY=/home/woori/venvs/seka_env/bin/python
S=/home/woori/scratch; TB=$S/tau2-bench
M="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8"; PORT=8141
RULES=$T2/a2/RULES_PROMPT_DV2.txt
TASKS="1,6,7,23,75,106"; NT=4
LOG=$S/prompt_rx.log; exec > $LOG 2>&1; set -x; date
cd $REPO && git pull --rebase -q origin facet-rft-2026 2>/dev/null
source /home/woori/.openrouter_key
export SSL_CERT_FILE=$($PY -c "import certifi;print(certifi.where())")
export PYTHONPATH=src:$T2
curl -s --max-time 5 localhost:$PORT/v1/models | grep -q "$M" || { echo SERVE_8141_MISSING; exit 1; }
cd $TB
# COMP base only · D-v2 엔진 OFF · rules_prompt로 5처방 주입
export T2_GATE_REGEN=1 T2_GATE_REGEN_K=1 T2_GATE_KINDS=auth,confirm,ownership,notice,preconditions,constraints
export T2_PROV_REGEN=1 T2_PROV_REGEN_K=4 T2_PRESENT_NESTED=1 T2_CALC=1
unset T2_PROV_MODE T2_GROUND T2_DISAMB T2_DISAMB_MODE T2_PRINCIPLE_DEFAULT T2_EPLAN T2_EPLAN_WALK T2_WRITE_CAP T2_WRITE_CAP_K T2_PRESENT_READS T2_AUTOFETCH T2_PROV_BADWORDS
rm -rf "$TB/data/simulations/prompt_rx"
$PY $T2/t2_run_gated.py --gate 1 --domain retail --agent_model "$M" --agent_base http://localhost:$PORT/v1 \
  --user_llm openrouter/openai/gpt-4.1 --user_temp 0.0 --num_trials $NT --max_concurrency 8 \
  --rules_prompt "$RULES" --save_to prompt_rx --task_ids "$TASKS" && echo "PROMPT_RX_DONE"
date; echo RUN_DONE
# persist
gzip -c "$TB/data/simulations/prompt_rx/results.json" > $REPO/reports/facet_rft_2026/sim_results/prompt_rx.results.json.gz 2>/dev/null
cd $REPO && git add -f reports/facet_rft_2026/sim_results/prompt_rx.results.json.gz 2>/dev/null
git commit -q -m "persist: prompt-rx (COMP+5rules advise·6-fail nt4·auto)" 2>/dev/null
git pull --rebase -q origin facet-rft-2026 2>/dev/null; git push -q origin facet-rft-2026 && echo PERSISTED
# summary
$PY - <<'PYEOF'
import json,os
from collections import defaultdict
p="/home/woori/scratch/tau2-bench/data/simulations/prompt_rx/results.json"
if not os.path.exists(p): print("MISSING"); raise SystemExit
d=json.load(open(p)); per=defaultdict(list)
for s in d["simulations"]:
    ri=s.get("reward_info") or {}
    per[str(s["task_id"])].append(1 if (ri.get("db_check") or {}).get("db_match") else 0)
print("=== prompt-rx db pass-rate (nt4) ===")
for t in ["1","6","7","23","75","106"]:
    v=per.get(t,[]); print("  task %4s: %d/%d=%.2f"%(t,sum(v),len(v),(sum(v)/len(v) if v else 0)))
allv=[x for t in per for x in per[t]]
print("  overall %d/%d=%.3f"%(sum(allv),len(allv),sum(allv)/len(allv) if allv else 0))
print("  비교: comp/full = B(bstack_*)·판독 trivial 무회귀면 advise가 override 부작용 해소")
PYEOF
echo ALLDONE; date
