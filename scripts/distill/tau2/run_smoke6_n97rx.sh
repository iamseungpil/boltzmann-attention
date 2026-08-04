#!/bin/bash
# Smoke for the four N97 prescriptions — does each one actually fire in a live run?
#
# Unit tests pass on all four and that has never been the question ([[30]]): T2_FN_ISOLATE
# was implemented, tested, and fired zero times in production (x43). So the six tasks here
# are not a sample — each is the trajectory whose read-through produced one of the levers,
# so a lever that stays silent here is silent because it does not work, not because the
# situation never arose.
#
#   task_051 · task_052   [READ-FIRST] with no callable suffix → shell dead end   P3
#   task_046              a true value dropped by the grounding guard, 11 turns    P5
#   task_084              the same call 71 times → max_steps, ungraded             P4
#   task_087 · task_092   user_id fed to an account_id parameter, gate never read  P1
#
#   usage:  run_smoke6_n97rx.sh [port] [tag]
set -u
PORT="${1:-8140}"; TAG="${2:-bank_smoke6_n97rx_$(date +%Y%m%d_%H%M)}"
R=/home/woori/workspace_common/boltzmann-attention-pi
LOGD=/home/woori/scratch/logs; mkdir -p "$LOGD"
TASKS="task_046,task_051,task_052,task_084,task_087,task_092"

[ -f /home/woori/.openai_key ] && . /home/woori/.openai_key
[ -f /home/woori/.openrouter_key ] && . /home/woori/.openrouter_key
. $R/scripts/distill/tau2/go_stack.sh

echo "== env (the flags this run believes it has) =="
env | grep -E '^T2_(CALLABLE_HINT|QUOTE_HINT|PIN_READ|REPEAT_CAP)=' | sort

t2_launch "$TAG" "$PORT" "$TASKS" 1 > "$LOGD/$TAG.log" 2>&1
echo "exit $? · log $LOGD/$TAG.log"

echo
echo "== 라이브 발화 (0이면 그 레버는 죽어 있다) =="
for f in T2_CALLABLE_HINT T2_QUOTE_HINT T2_PIN_READ T2_REPEAT_CAP; do
  printf '%-20s %s\n' "$f" "$(grep -c "\[T2_LEVER\] $f" "$LOGD/$TAG.log")"
done
printf '%-20s %s\n' "[T2_PIN_READ] pinned" "$(grep -c '\[T2_PIN_READ\] pinned' "$LOGD/$TAG.log")"
printf '%-20s %s\n' "[T2_PIN_READ] skip" "$(grep -c '\[T2_PIN_READ\] skipped' "$LOGD/$TAG.log")"

echo
echo "== 부작용 감시 (발화가 무언가를 깨뜨렸나) =="
printf '%-20s %s\n' "replay ValueError" "$(grep -c 'ValueError' "$LOGD/$TAG.log")"
printf '%-20s %s\n' "400/schema 오류" "$(grep -ciE 'BadRequest|invalid_request|__log_extra_fields__' "$LOGD/$TAG.log")"
grep -E '\[T2_LEVER\] (T2_CALLABLE_HINT|T2_QUOTE_HINT)|\[T2_PIN_READ\]|REPEAT-CAP' "$LOGD/$TAG.log" | head -12
