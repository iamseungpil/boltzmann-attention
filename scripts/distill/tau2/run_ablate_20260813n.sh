#!/bin/bash
# **절제(ablation) 런 — 스택 수준 부정통제** (2026-08-13·사용자 지시 "삭감 실험"·[[57]]).
#
# ★왜 이 런이 필요한가 (확인된 공백)
#   banking 로그 전수에서 `[t2_run RESULT] gate=` 102건이 **전부 gate=1** 이고 gate=0 은 **0건**.
#   `bank_dbw_off`·`bank_lever_off` 의 "off" 는 **레버 하나만** 끈 arm 이고 스택은 켜져 있었다.
#   ⇒ *"우리 스캐폴드가 이 6태스크에서 무엇을 사는가"* 는 **한 번도 측정된 적이 없다**.
#   재판정 런이 기준선과 정확히 같은 11/24 를 냈으므로(원장 C458 예정) 이 질문이 지금 결정적이다.
#
# ★arm 정의: `--gate 0` = `t2_gate_patch` 를 **아예 임포트하지 않는다**(t2_run_gated.py:199).
#   게이트·레버·A2 구동 재생성 전부 없음 = **순수 base 에이전트**. T2_* 환경변수는 소비자가
#   없으므로 무해하지만, 누수를 원천 차단하려고 이 런처는 go_stack.sh 를 **source 하지 않고**
#   필요한 경로·키만 직접 세운다.
#
# ★비교 정합성(이것이 틀리면 절제가 무효다): 모델·user-sim·retrieval·온도·effort·max_steps·
#   concurrency·nt·태스크·저장형식을 스택 arm(go_stack.sh t2_launch)과 **문자 그대로 동일**하게
#   맞춘다. 유일한 델타 = `--gate 0`.
#
# 읽을 것: 태스크별 pass 를 기준선(batch4+dbw_on 11/24)·재판정(11/24)과 3열 대조.
#   base 가 비슷하면 ⇒ 스택은 이 6태스크에서 값을 못 산다(대폭 삭감 근거).
#   base 가 낮으면  ⇒ 스택이 사는 것이 있고, 다음 절제는 조각별로 좁힌다.
#
# usage: run_ablate_20260813n.sh
set -e
REPO=/home/woori/workspace_common/boltzmann-attention-pi
TAU2=/home/woori/scratch/tau2-bench
PY=/home/woori/venvs/seka_env/bin/python
LOG=/home/woori/scratch/logs
SIMS=$TAU2/data/simulations
NT=4
mkdir -p "$LOG"

SHA=$(cd "$REPO" && git rev-parse --short HEAD)

# dense KB(alltools)는 키가 필요하다 — 없으면 스택 arm 과 retrieval 조건이 달라져 절제가 무효.
[ -f /home/woori/.openai_key ] && . /home/woori/.openai_key
if [ -z "$OPENAI_API_KEY" ]; then
  echo "[ablate] REFUSING: alltools(dense KB) needs OPENAI_API_KEY — 스택 arm 과 조건 불일치." >&2
  exit 1
fi

launch () {
  NAME="$1"; TASKS="$2"; PORT="$3"
  TAG="bank_base_${NAME}_20260813n"
  if [ -e "$LOG/${TAG}.log" ]; then
    echo "[ablate] SKIP: $LOG/${TAG}.log 가 이미 있다." >&2; return 0
  fi
  if [ -e "$SIMS/${TAG}" ]; then
    echo "[ablate] REFUSING: $SIMS/${TAG} 가 이미 있다 — 지우고 다시 걸어라." >&2; return 1
  fi
  if ps -eo cmd | grep -v grep | grep "t2_run_gated.py" | grep -q "localhost:${PORT}/"; then
    echo "[ablate] REFUSING: 포트 ${PORT} 사용 중." >&2; return 1
  fi
  echo "{\"tag\":\"$TAG\",\"scaffold_sha\":\"$SHA\",\"tasks\":\"$TASKS\",\"port\":$PORT,\"nt\":$NT,\"arm\":\"base(gate=0)\",\"note\":\"stack-level negative control\"}" \
    | tee "$LOG/${TAG}.meta.json"
  setsid bash -c "cd '$TAU2' && PYTHONPATH=src:$REPO/scripts/distill/tau2 OPENAI_API_KEY='$OPENAI_API_KEY' \
    $PY -u '$REPO/scripts/distill/tau2/t2_run_gated.py' \
      --domain banking_knowledge --retrieval_config alltools \
      --gate 0 \
      --agent_model Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8 \
      --agent_base 'http://localhost:${PORT}/v1' \
      --user_llm openrouter/openai/gpt-5.2 --user_temp 0.0 \
      --user_reasoning_effort low \
      --task_ids '$TASKS' --num_trials $NT --max_concurrency 4 \
      --max_steps 200 --save_to '$TAG'" </dev/null >"$LOG/${TAG}.log" 2>&1 &
  echo "[ablate] $TASKS → PID=$! port=$PORT log=$LOG/${TAG}.log"
}

launch a task_070,task_071,task_098 8140
launch b task_010,task_099,task_100 8141
echo "[ablate] 기동 완료 · sha=$SHA · nt=$NT · arm=base(gate=0) · 유일 델타=스캐폴드 부재"
