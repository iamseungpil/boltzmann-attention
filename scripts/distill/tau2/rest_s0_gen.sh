#!/bin/bash
# ReST S0 생성잡 — 32B-int8 teacher로 τ² retail 검증-정답 궤적 생성 (증류 데이터)
# 동기: coverage 진단(2026-06-20) = base 7B+pg는 14/50(쉬운것 11)·HOLE 29(32B만 통과).
#   ⇒ 32B-teacher가 구멍 29 메움(정의상 32B 통과). 검증기=DB-match로 필터(다음 단계).
# 설계=REST_INTERNALIZE_DESIGN §8 S0. proven pg config(gate1+T2_PROVENANCE+NOFAB·resolve0).
# Usage: rest_s0_gen.sh <gpu> <port> [num_trials] [num_tasks]
set -u
GPU=${1:-0}; PORT=${2:-8361}; TR=${3:-5}; NT=${4:-114}
MODEL="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8"
R=/home/woori/workspace_common/boltzmann-attention-pi; T2=$R/scripts/distill/tau2
PY=/home/woori/venvs/seka_env/bin/python; VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
S=/home/woori/scratch; NOFAB=$T2/a2/NOFAB_PROMPT.txt
SAVE=rest_s0_gen32b
LOG=$S/rest_s0_gen32b.log; exec > $LOG 2>&1; set -x; date; echo "MODEL=$MODEL TR=$TR NT=$NT"

# clean GPU
for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 12
CUDA_VISIBLE_DEVICES=$GPU setsid nohup $VLLM serve $MODEL --port $PORT \
  --enable-auto-tool-choice --tool-call-parser hermes --max-model-len 16384 --enforce-eager \
  > $S/vllm_rest_s0.log 2>&1 &
# healthcheck (32B-int8 load ~ a few min)
ok=0; for i in $(seq 1 150); do curl -s localhost:$PORT/v1/models 2>/dev/null | grep -q "$MODEL" && ok=1 && break; sleep 10; done
[ $ok = 1 ] || { echo SERVE_FAIL; tail -40 $S/vllm_rest_s0.log; exit 1; }
hc=$(curl -s localhost:$PORT/v1/chat/completions -H "Content-Type: application/json" \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":3}" 2>/dev/null)
echo "$hc" | grep -q '"content"' || { echo "HC_FAIL: $hc"; exit 1; }; echo "[hc OK]"

set +x; source /home/woori/.openrouter_key; export SSL_CERT_FILE=$($PY -c "import certifi;print(certifi.where())"); set -x
cd $S/tau2-bench; export PYTHONPATH=src:$T2
export T2_PROVENANCE=1   # gate (proven pg config)

$PY -u $T2/t2_run_gated.py --gate 1 --resolve 0 --domain retail \
  --num_trials $TR --num_tasks $NT --rules_prompt $NOFAB \
  --agent_model "$MODEL" --agent_base http://localhost:$PORT/v1 \
  --user_llm "openrouter/openai/gpt-4.1" --user_temp 0.7 \
  --max_concurrency 8 --save_to $SAVE || echo RUN_FAIL

# yield report: how many sims db_match
$PY - <<'PYEOF'
import json,os,collections
p="data/simulations/rest_s0_gen32b/results.json"
if os.path.exists(p):
    d=json.load(open(p)); sims=d.get("simulations") or []
    byt=collections.defaultdict(list)
    for s in sims:
        ri=s.get("reward_info") or {}; db=(ri.get("db_check") or {})
        ok=bool(db.get("db_match")) or (ri.get("reward",0) or 0)>=0.999
        byt[str(s.get("task_id"))].append(ok)
    keep_sims=sum(1 for s in sims if (lambda ri:(bool((ri.get('db_check') or {}).get('db_match')) or (ri.get('reward',0) or 0)>=0.999))(s.get('reward_info') or {}))
    tasks_any=sum(1 for t,v in byt.items() if any(v))
    print(f"GEN_YIELD: keep_sims={keep_sims}/{len(sims)}  tasks_with_>=1_correct={tasks_any}/{len(byt)}")
PYEOF

for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done
date; echo REST_S0_GEN_DONE
