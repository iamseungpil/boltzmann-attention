#!/bin/bash
# Additive 절단: COMP(pass) + D-v2 레버 하나씩 → task 106(COMP=4/4, b78c=fail) 을 뒤집는 오염원 특정.
# temp=0(agent+user) 결정론. nt=2로 vLLM 배칭 비결정성 체크. 무료우선 진단([[08]]).
# 확장: 오염원 확정 후 --task_ids 를 6-fail(1,6,7,23,75,106)로.
set -u
REPO=/home/woori/workspace_common/boltzmann-attention-pi
T2=$REPO/scripts/distill/tau2; PY=/home/woori/venvs/seka_env/bin/python
S=/home/woori/scratch; TB=$S/tau2-bench
M="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8"; PORT=8140
TASKS="${1:-106}"; NT="${2:-2}"
LOG=$S/abl_sysprompt_106.log; exec > $LOG 2>&1; set -x; date
cd $REPO && git pull --rebase -q origin facet-rft-2026 2>/dev/null
source /home/woori/.openrouter_key
export SSL_CERT_FILE=$($PY -c "import certifi;print(certifi.where())")
export PYTHONPATH=src:$T2
curl -s --max-time 5 localhost:$PORT/v1/models | grep -q "$M" || { echo SERVE_MISSING; exit 1; }
cd $TB

run_cfg(){
  local tag="$1"; local extra="$2"
  (
    export T2_GATE_REGEN=1 T2_GATE_REGEN_K=1 T2_GATE_KINDS=auth,confirm,ownership,notice,preconditions,constraints
    export T2_PROV_REGEN=1 T2_PROV_REGEN_K=4 T2_PRESENT_NESTED=1 T2_CALC=1
    unset T2_PROV_MODE T2_GROUND T2_DISAMB T2_DISAMB_MODE T2_PRINCIPLE_DEFAULT T2_EPLAN T2_EPLAN_WALK T2_WRITE_CAP T2_WRITE_CAP_K T2_PRESENT_READS T2_AUTOFETCH T2_PROV_BADWORDS
    eval "$extra"
    rm -rf "$TB/data/simulations/abl106_$tag"
    $PY $T2/t2_run_gated.py --gate 1 --domain retail --agent_model "$M" --agent_base http://localhost:$PORT/v1 \
      --user_llm openrouter/openai/gpt-4.1 --user_temp 0.0 --num_trials $NT --max_concurrency 2 \
      --save_to "abl106_$tag" --task_ids "$TASKS" && echo "CFG_DONE $tag"
  ) &
}

run_cfg c0 ":"
run_cfg c1 "export T2_PROV_MODE=rescue"
run_cfg c2 "export T2_GROUND=1"
run_cfg c3 "export T2_DISAMB=1 T2_DISAMB_MODE=subcall"
run_cfg c4 "export T2_PRINCIPLE_DEFAULT=1"
run_cfg c5 "export T2_EPLAN=1 T2_EPLAN_WALK=1"
run_cfg cf "export T2_PROV_MODE=rescue T2_GROUND=1 T2_DISAMB=1 T2_DISAMB_MODE=subcall T2_PRINCIPLE_DEFAULT=1 T2_EPLAN=1 T2_EPLAN_WALK=1 T2_WRITE_CAP=1 T2_WRITE_CAP_K=2"
wait
date; echo ALL_CFG_DONE

$PY - <<'PYEOF'
import json,os
TB="/home/woori/scratch/tau2-bench/data/simulations"
for tag in ["c0","c1","c2","c3","c4","c5","cf"]:
    p=f"{TB}/abl106_{tag}/results.json"
    if not os.path.exists(p): print(tag,"MISSING"); continue
    d=json.load(open(p)); sims=d["simulations"]
    parts=[]
    for s in sims:
        ri=s.get("reward_info") or {}
        db=(ri.get("db_check") or {}).get("db_match")
        wr=None
        for m in s.get("messages",[]):
            if m.get("role")=="assistant":
                for tc in (m.get("tool_calls") or []):
                    fn=tc.get("function",{}) if "function" in tc else tc
                    nm=fn.get("name") or tc.get("name") or ""
                    if "exchange" in nm or "modify" in nm or "return" in nm:
                        try: wr=json.loads(fn.get("arguments") or tc.get("arguments") or "{}").get("new_item_ids")
                        except: wr="?"
        parts.append("t%s:db=%s new=%s(%s)"%(s.get("trial"),db,wr,s.get("termination_reason")))
    print("%-3s | %s"%(tag," | ".join(parts)))
print("TASK106 gold new_item_ids=['2060066974']  (COMP passes 4/4)")
PYEOF
echo ALLDONE; date
