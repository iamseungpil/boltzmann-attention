#!/bin/bash
# Assembled-Deterministic 한-런 (2026-06-25·convergence checkpoint·learn 질문 종결).
# 정본 설계 = ASSEMBLED_DETERMINISTIC_RUN_DESIGN_2026_06_25.md. 모든 깨끗한 결정론 max:
#   present(order-list) + nested(item/variant) + 전체게이트(auth/confirm/ownership/notice/preconditions)
#   + constraints(disjoint=new≠old) + calc(available-count·order-total·report-conversion)
# → gold-diff 잔여 census로 operand가 present로 닫히나(결정론) / present-but-wrong(learn-or-capability) 격리.
# ★구현 게이트1-3+A2 PASS(test_assembled_run.py)·게이트4(71/74/101/102 present발화+주소열거) 확인됨.
# 측정 = pass^all(robust)·결정론 census·pass^1 단독 금지. baseline=present+nest+g15(*_presentnest_g15_retail_t3).
# 사용법: reexp_assembled.sh <GPU> <PORT> <MODEL> <TAG>
set -u
GPU=$1; PORT=$2; M=$3; TAG=$4
REPO=/home/woori/workspace_common/boltzmann-attention-pi
T2=$REPO/scripts/distill/tau2; PY=/home/woori/venvs/seka_env/bin/python
VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm; S=/home/woori/scratch; TB=/home/woori/scratch/tau2-bench
LOG=$S/reexp_asm_$TAG.log
exec > $LOG 2>&1; set -x; date
cd $REPO && git pull --ff-only
# ★구현 게이트(단위) — 통과 못하면 런 중단([[05]] 드리프트 차단)
$PY $T2/test_assembled_run.py || { echo "GATE_FAIL — abort"; exit 1; }
source /home/woori/.openrouter_key
export SSL_CERT_FILE=$($PY -c "import certifi;print(certifi.where())")
export PYTHONPATH=src:$T2
run () { local save=$1; shift
  echo "######## RUN $save env=$* ########"; date
  cd $TB; rm -rf "$TB/data/simulations/$save"
  env "$@" PYTHONPATH=src:$T2 $PY $T2/t2_run_gated.py --gate 1 --domain retail \
    --agent_model "$M" --agent_base http://localhost:$PORT/v1 \
    --user_llm openrouter/openai/gpt-4.1 --user_temp 0.0 \
    --num_trials 3 --max_concurrency 8 --save_to "$save" || echo "ARM_FAIL $save"
  echo "ARM_DONE $save"; date; }
for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 4
CUDA_VISIBLE_DEVICES=$GPU setsid nohup $VLLM serve "$M" --port $PORT --enable-auto-tool-choice \
  --tool-call-parser hermes --max-model-len 16384 --enforce-eager --gpu-memory-utilization 0.92 \
  > $S/vllm_asm_$TAG.log 2>&1 &
ok=0; for i in $(seq 1 150); do curl -s localhost:$PORT/v1/models 2>/dev/null | grep -q "$M" && ok=1 && break; sleep 10; done
[ $ok = 1 ] || { echo "SERVE_FAIL"; tail -40 $S/vllm_asm_$TAG.log; exit 1; }
echo "SERVE_OK"; date
run ${TAG}_assembled_retail_t3 \
  T2_GATE_KINDS=auth,confirm,ownership,notice,preconditions,constraints \
  T2_PRESENT_READS=1 T2_PRESENT_NESTED=1 T2_CALC=1 ${XFLAGS:-T2_RETRY_CONTROLLER=1}
for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done
# ★결과 영속화 (gitignore 우회·소실방지·재사용): gzip → repo tracked path → commit+push.
RES=$TB/data/simulations/${TAG}_assembled_retail_t3/results.json
PERSIST=$REPO/reports/facet_rft_2026/sim_results; mkdir -p $PERSIST
if [ -f "$RES" ]; then
  gzip -c "$RES" > $PERSIST/${TAG}_assembled_retail_t3.results.json.gz
  cd $REPO && git pull --rebase -q origin facet-rft-2026 2>/dev/null
  git add -f $PERSIST/${TAG}_assembled_retail_t3.results.json.gz
  git commit -q -m "persist sim results: ${TAG}_assembled_retail_t3 (auto·소실방지)" 2>/dev/null
  for try in 1 2 3; do git pull --rebase -q origin facet-rft-2026 2>/dev/null; git push -q origin facet-rft-2026 && { echo "PERSISTED_${TAG}"; break; }; sleep 5; done
else
  echo "PERSIST_SKIP_NO_RESULTS_${TAG}"
fi
echo "REEXP_ASM_${TAG}_DONE"; date
