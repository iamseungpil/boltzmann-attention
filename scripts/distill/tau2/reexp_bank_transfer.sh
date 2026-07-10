#!/bin/bash
# ★Phase 3 도메인 전이 실측 — banking_knowledge (DOMAIN_TRANSFER_STATUS_AND_PLAN §3.2 Phase 3).
#   arm = 32B + 도메인-일반 게이트엔진(T2_GATE_REGEN·replay-safe) + banking A2(GB1 verify 게이트만).
#   floor = 동일 32B·--gate 0. ★기존 ours_n32int8_floor_bank_t3는 16384 serve로 infra 31/291
#   (ContextWindowExceeded) = 방법-결함 → 32768로 재런(이 파일 floor 모드).
#   present/autofetch = 영구 폐기(C34 규칙0)·prov는 gate_regen과 상호배타 → 이 arm은 게이트-단독.
# Usage: reexp_bank_transfer.sh <GPU> <PORT> <MODE: smoke|floor|arm>
set -u
GPU=$1; PORT=$2; MODE=$3
REPO=/home/woori/workspace_common/boltzmann-attention-pi
T2=$REPO/scripts/distill/tau2; PY=/home/woori/venvs/seka_env/bin/python
VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm; S=/home/woori/scratch; TB=/home/woori/scratch/tau2-bench
M="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8"
DOM=banking_knowledge
LOG=$S/bankxfer_${MODE}.log
exec > $LOG 2>&1; set -x; date
cd $REPO && git pull --ff-only
source /home/woori/.openrouter_key
[ -f /home/woori/.openai_key ] && source /home/woori/.openai_key
export SSL_CERT_FILE=$($PY -c "import certifi;print(certifi.where())")
export PYTHONPATH=src:$T2

# serve (32768 — banking 장기 태스크의 16k ContextWindow 크래시 방지)
for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 4
CUDA_VISIBLE_DEVICES=$GPU setsid nohup $VLLM serve "$M" --port $PORT --enable-auto-tool-choice \
  --tool-call-parser hermes --max-model-len 32768 --enforce-eager --gpu-memory-utilization 0.92 \
  > $S/vllm_bankxfer_${MODE}.log 2>&1 &
ok=0; for i in $(seq 1 180); do curl -s localhost:$PORT/v1/models 2>/dev/null | grep -q "$M" && ok=1 && break; sleep 10; done
[ $ok = 1 ] || { echo "SERVE_FAIL"; tail -40 $S/vllm_bankxfer_${MODE}.log; exit 1; }
echo "SERVE_OK"; date

ARM_ENV=(T2_GATE_REGEN=1 T2_GATE_REGEN_K=1 T2_GATE_KINDS=auth)

run () {  # $1=save $2=gate(0/1) $3=nt $4=extra_args...
  local save=$1 gate=$2 nt=$3; shift 3
  cd $TB; rm -rf "$TB/data/simulations/$save"
  if [ "$gate" = 1 ]; then
    env "${ARM_ENV[@]}" $PY $T2/t2_run_gated.py --gate 1 --domain $DOM \
      --agent_model "$M" --agent_base http://localhost:$PORT/v1 \
      --user_llm openrouter/openai/gpt-4.1 --user_temp 0.0 \
      --num_trials $nt --max_concurrency 8 --save_to "$save" "$@" || echo "ARM_FAIL $save"
  else
    $PY $T2/t2_run_gated.py --gate 0 --domain $DOM \
      --agent_model "$M" --agent_base http://localhost:$PORT/v1 \
      --user_llm openrouter/openai/gpt-4.1 --user_temp 0.0 \
      --num_trials $nt --max_concurrency 8 --save_to "$save" "$@" || echo "ARM_FAIL $save"
  fi
  echo "ARM_DONE $save"; date
  $PY - "$TB/data/simulations/$save/results.json" <<'PYEOF'
import json, sys
from collections import Counter
p = sys.argv[1]
s = json.load(open(p))["simulations"]
inf = sum(1 for x in s if (x.get("reward_info") or {}).get("reward") is None)
rs = [(x.get("reward_info") or {}).get("reward") for x in s]
rs = [r for r in rs if r is not None]
term = Counter(x.get("termination_reason") for x in s)
gate_fire = lv = 0
for x in s:
    for m in x.get("messages") or []:
        c = m.get("content")
        if isinstance(c, str) and "blocked by policy gate" in c:
            gate_fire += 1
        for tc in (m.get("tool_calls") or []):
            fn = tc.get("function", tc) if isinstance(tc, dict) else tc
            if (fn.get("name") if isinstance(fn, dict) else None) == "log_verification":
                lv += 1
print("SUMMARY n=%d infra=%d mean_r=%.4f term=%s gate_fire_msgs=%d log_verification_calls=%d"
      % (len(s), inf, sum(rs)/max(len(rs),1), dict(term), gate_fire, lv))
PYEOF
}

persist () {  # $1=save
  local save=$1
  local RES=$TB/data/simulations/$save/results.json
  [ -f "$RES" ] || { echo "PERSIST_SKIP $save"; return; }
  gzip -c "$RES" > $REPO/reports/facet_rft_2026/sim_results/${save}.results.json.gz
  cd $REPO && git pull --rebase -q origin facet-rft-2026 2>/dev/null
  git add -f reports/facet_rft_2026/sim_results/${save}.results.json.gz
  git commit -q -m "persist sim results: ${save} (banking transfer Phase3·auto)" 2>/dev/null
  for try in 1 2 3; do git pull --rebase -q origin facet-rft-2026 2>/dev/null; git push -q origin facet-rft-2026 && { echo "PERSISTED_${save}"; break; }; sleep 5; done
}

case $MODE in
  smoke)
    # 8태스크: 검증-요구 5(gate 실발화 기대) + 면제 3(015 give·032/035 transfer=false-block 0 검증)
    run bankxfer_smoke 1 1 --task_ids task_001,task_002,task_003,task_004,task_005,task_015,task_032,task_035
    persist bankxfer_smoke
    echo "BANKXFER_SMOKE_DONE" ;;
  floor)
    run bankxfer_floor_bank_t4 0 4
    persist bankxfer_floor_bank_t4
    echo "BANKXFER_FLOOR_DONE" ;;
  arm)
    run bankxfer_gate_bank_t4 1 4
    persist bankxfer_gate_bank_t4
    echo "BANKXFER_ARM_DONE" ;;
esac
for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done
echo "BANKXFER_${MODE}_ALLDONE"; date
