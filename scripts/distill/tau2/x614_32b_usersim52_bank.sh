#!/usr/bin/env bash
# x614 — user-sim 교락 제거 대조.
#   질문: banking 002~005 의 32B(0/4) ↔ Qwen3.8(4/4) 격차가 세대 때문인가, user-sim 때문인가.
#   설계: 32B floor 를 **user-sim 만 gpt-4.1 → gpt-5.2 로 바꿔** 재실행. 그 외 전부 원런과 동일.
#   원런 = ours_n32int8_floor_bank (user gpt-4.1 · seed 300 · alltools · gate 0 · nt1 · max_steps 200)
#   비교 대상 = bank_x599_q38base_banking_20260829 (user gpt-5.2 reasoning low)
#   ⛔floor 이므로 go_stack 을 source 하지 않는다 = T2_* 레버 0 (x607 rev1 결함의 교훈).
set -u
REPO=/home/woori/workspace_common/boltzmann-attention-pi
LOG=/home/woori/scratch/logs
SR=$REPO/reports/facet_rft_2026/sim_results
TAG=bank_x614_32b_us52_bank_20260830
PY=/home/woori/venvs/seka_env/bin/python

curl -s -m 10 http://localhost:8141/v1/models -o /dev/null || { echo "REFUSING: 8141 not serving"; exit 1; }
echo "[x614] engine_sha=$(git -C $REPO rev-parse --short HEAD) tasks=task_002..005 nt=1 user=gpt-5.2"

cd $REPO/scripts/distill/tau2
env -u T2_ACTION_SUB -u T2_KEEP_DENY_BODY -u T2_CALL_FORM -u T2_ARG_EMPTY \
    -u T2_SEARCH_AGENT -u T2_SG_DOCS -u T2_SG_PROMPT_V2 -u T2_SUPPRESS_AUTH -u T2_RESOLVE \
    T2_LLM_TIMEOUT=900 \
  $PY -u "$REPO/scripts/distill/tau2/t2_run_gated.py" \
    --domain banking_knowledge --gate 0 --retrieval_config alltools \
    --agent_model Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8 \
    --agent_base http://localhost:8141/v1 \
    --user_llm openrouter/openai/gpt-5.2 --user_temp 0.0 --user_reasoning_effort low \
    --task_ids task_002,task_003,task_004,task_005 \
    --num_trials 1 --max_concurrency 1 --max_steps 200 \
    --save_to $TAG 2>&1 | tee $LOG/$TAG.log

# ---- 영속화 ([[30]]: tracked 확인까지가 절차) ----
TB=/home/woori/scratch/tau2-bench/data/simulations
[ -f "$TB/$TAG/results.json" ]    && gzip -c "$TB/$TAG/results.json"    > "$SR/$TAG.results.json.gz"
[ -f "$TB/$TAG/compliance.json" ] && gzip -c "$TB/$TAG/compliance.json" > "$SR/$TAG.compliance.json.gz"
[ -f "$LOG/$TAG.log" ]            && gzip -c "$LOG/$TAG.log"            > "$SR/$TAG.log.gz"
for sc in fb trace; do
  [ -f "$LOG/${sc}_$TAG.jsonl" ] && gzip -c "$LOG/${sc}_$TAG.jsonl" > "$SR/${sc}_$TAG.jsonl.gz"
done
cd $REPO
git add -f reports/facet_rft_2026/sim_results/*x614* 2>/dev/null
git commit -q -m "persist: x614 32B floor banking 002-005 with gpt-5.2 user-sim (confound control)" 2>/dev/null
git pull --rebase -q origin facet-rft-2026 2>/dev/null; git push -q origin facet-rft-2026 && echo PUSHED
for f in $SR/*x614*; do
  git ls-files --error-unmatch "$f" >/dev/null 2>&1 && echo "OK   $(basename $f)" || echo "FAIL $(basename $f)"
done
echo "[x614 $(date +%H:%M:%S)] 종료"
