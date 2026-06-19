#!/bin/bash
# 실 τ² user-sim e2e — 단독 통합 LoRA (qwen7b_solo_sts) arm. base floor(real_e2e_base.sh)와 동일 조건
# (retail·gate=1·gpt-4.1 user-sim·temp0·num_tasks 40) + resolve_selection wiring ON + num_trials 2(분산 축소).
# 판정 3축: (a) resolve_selection 실제 호출되나 (C0 NO-GO 해소) (b) base 0.205 대비 Δ≥3-4 pass인가
#           (c) operand 명명 합본학습 도움/해 (§23D 실 e2e 판정).
# Usage: real_e2e_solo.sh <gpu> <port> [num_tasks] [num_trials] [adapter_dir] [tag]
set -u
GPU=${1:?gpu}; PORT=${2:?port}; NT=${3:-40}; TR=${4:-2}
R=/home/woori/workspace_common/boltzmann-attention-pi
T2=$R/scripts/distill/tau2
PY=/home/woori/venvs/seka_env/bin/python
VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
BASE=Qwen/Qwen2.5-7B-Instruct
LORA=${5:-/home/woori/scratch/adapters/qwen7b_solo_sts}
TAG=${6:-qwen7b_solo_sts}
S=/home/woori/scratch
LOG=$S/real_e2e_solo_${TAG}.log
exec > $LOG 2>&1; set -x; date

[ -d "$LORA" ] || { echo "ADAPTER_MISSING $LORA"; exit 1; }
for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 4
CUDA_VISIBLE_DEVICES=$GPU setsid nohup $VLLM serve $BASE --port $PORT \
  --enable-auto-tool-choice --tool-call-parser hermes --max-model-len 16384 \
  --enable-lora --max-lora-rank 64 --lora-modules ${TAG}=${LORA} > $S/vllm_real_e2e_solo.log 2>&1 &
ok=0; for i in $(seq 1 80); do curl -s localhost:$PORT/v1/models 2>/dev/null | grep -q "$TAG" && ok=1 && break; sleep 10; done
[ $ok = 1 ] || { echo SERVE_FAIL; tail -30 $S/vllm_real_e2e_solo.log; exit 1; }

set +x; source /home/woori/.openrouter_key; export SSL_CERT_FILE=$($PY -c "import certifi;print(certifi.where())"); set -x
cd $S/tau2-bench; export PYTHONPATH=src:$T2

rm -rf data/simulations/real_solo_${TAG}_retail
$PY $T2/t2_run_gated.py --gate 1 --resolve 1 --domain retail --num_trials $TR --num_tasks $NT \
  --agent_model $TAG --agent_base http://localhost:$PORT/v1 \
  --user_llm "openrouter/openai/gpt-4.1" --user_temp 0.0 --save_to real_solo_${TAG}_retail \
  > $S/t2_real_solo_${TAG}_retail.log 2>&1 || echo "FAIL retail"

set +x
echo "=== RESULT (RESULT/pass1 line) ==="; grep -E "RESULT|pass1|resolve_selection wiring" $S/t2_real_solo_${TAG}_retail.log | tail -5
echo "=== [$TAG] full metrics (pass·resolve grounding ok/fail=ε·FAB) ==="
$PY - "$S/tau2-bench/data/simulations/real_solo_${TAG}_retail/results.json" "$TAG" <<'PY'
import json,sys
try: d=json.load(open(sys.argv[1]))
except Exception as e: print("NA",e); raise SystemExit
sims=d.get("simulations",[]); rew=[(s.get("reward_info") or {}).get("reward") for s in sims]; rew=[r for r in rew if r is not None]
rcall=gok=gfail=fab=0
for s in sims:
  for m in s.get("messages",[]):
    if m.get("role")=="assistant":
      for tc in (m.get("tool_calls") or []):
        if tc.get("name")=="resolve_selection": rcall+=1
    elif m.get("role")=="tool":
      c=str(m.get("content") or "")
      if '"tool": "resolve_selection"' in c or '"tool":"resolve_selection"' in c: gok+=1
      elif "resolve_selection could not ground" in c: gfail+=1
      if "not found" in c.lower(): fab+=1
print(f"[{sys.argv[2]}] n={len(rew)} pass1={sum(r>=1 for r in rew)}/{len(rew)} ({sum(r>=1 for r in rew)/max(len(rew),1):.3f})  "
      f"resolve_calls={rcall} ground_OK={gok} ground_FAIL={gfail} (ε-unmask: OK>0 = P2b 살아 도달)  FAB(not-found)={fab}")
PY

for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done
echo "=== REAL E2E SOLO DONE (qwen7b_solo_sts·retail·gate+resolve·trials=$TR) ==="; date; echo REAL_E2E_SOLO_DONE
