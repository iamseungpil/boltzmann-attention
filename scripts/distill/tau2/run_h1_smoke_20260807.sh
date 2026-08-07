#!/usr/bin/env bash
# h1 스모크 — 예산 폐지 + 원장 산수(T2_LEDGER)를 101·102에 함께 건다.
#
# 무엇을 보려는가 (`DAG_SCHEDULER_DESIGN_2026_08_07` §6):
#   1) 발화: `[ORDER]`가 turn 6 이후에도 살아 있고, NOW가 `get_referrals_by_user`로 **전진**하는가.
#      직전 라이브에서 ORDER는 turn 4·6·8 세 번으로 끝났고(cap 3), 원장 조회 명령은 0회였다.
#   2) 발화: `[T2_LEDGER]`가 실제로 뜨고, 궤적에 창_잔여가 들어가는가.
#   3) 회귀: 102 DB(직전 3/22)가 무너지지 않는가 — 우리의 유일한 0 아닌 신호라 해가 있으면 여기서 보인다.
#   4) Δspurious: 개입이 늘어 gold 밖 호출·컨텍스트 초과가 늘지 않는가(등대 제1원리).
#
# arm = 직전 f1 구성(T2_ARBITRATE·T2_SOURCE 실발화 확인) + T2_LEDGER=1.
# ⚠유료(gpt-5.2 user-sim) — 2 태스크 × 2 trial = 4 sim. 스모크 목적이므로 이 이상 키우지 않는다([[09]]).
set -u
REPO=/home/woori/workspace_common/boltzmann-attention-pi
cd "$REPO/scripts/distill/tau2" || exit 1
# shellcheck disable=SC1091
source ./go_stack.sh 2>/dev/null || true
[ -f /home/woori/.openai_key ] && source /home/woori/.openai_key
[ -f /home/woori/.openrouter_key ] && source /home/woori/.openrouter_key

export T2_ARBITRATE=1 T2_SOURCE=1
export T2_LEDGER=1                # 신규: 원장이 돌아오면 창_잔여·유형별 누적을 낸다(전사=모델·산수=엔진)
unset T2_ACTION_DENY_CAP          # 예산 폐지(2026-08-07 사용자 지시) — 미설정=무제한

LOG=/home/woori/scratch/logs
mkdir -p "$LOG"
# 101 = gpu1(8141) · 102 = gpu0(8140) — 직전 arm과 같은 배치라 GPU 요인이 섞이지 않는다.
setsid bash -c "cd '$REPO/scripts/distill/tau2' && source ./go_stack.sh >/dev/null 2>&1; \
  [ -f /home/woori/.openai_key ] && source /home/woori/.openai_key; \
  [ -f /home/woori/.openrouter_key ] && source /home/woori/.openrouter_key; \
  export T2_ARBITRATE=1 T2_SOURCE=1 T2_LEDGER=1; unset T2_ACTION_DENY_CAP; \
  t2_launch bank_h1_gpu1 8141 task_101 2" </dev/null >"$LOG/h1_gpu1.log" 2>&1 &
setsid bash -c "cd '$REPO/scripts/distill/tau2' && source ./go_stack.sh >/dev/null 2>&1; \
  [ -f /home/woori/.openai_key ] && source /home/woori/.openai_key; \
  [ -f /home/woori/.openrouter_key ] && source /home/woori/.openrouter_key; \
  export T2_ARBITRATE=1 T2_SOURCE=1 T2_LEDGER=1; unset T2_ACTION_DENY_CAP; \
  t2_launch bank_h1_gpu0 8140 task_102 2" </dev/null >"$LOG/h1_gpu0.log" 2>&1 &
sleep 2
echo "launched. logs: $LOG/h1_gpu0.log $LOG/h1_gpu1.log"
