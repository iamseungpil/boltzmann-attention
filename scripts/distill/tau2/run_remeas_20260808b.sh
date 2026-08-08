#!/bin/bash
# 재측정 런 — C324/C325/C326/C327 이후의 스택을 **성한 표적**에서 본다.
#
# 표적 선정(원장 C328): 추천 계열 5개 중 **102·101은 gold 결함으로 제외**한다.
#   · 102 = 상류 #432(열림·설립일 출처 부재)
#   · 101 = C328(gold가 문서 코퍼스와 모순·상류 미보고). 원문 축자로 확인:
#           TechFlow(2년·예치 $12,000)에 Lime Green은 예치 미달, Hunter Green $175가
#           자격 만족이고 Sky Blue $150보다 높은데 gold는 Sky Blue를 요구한다.
#   · 099·100 = gold 수치가 코퍼스와 **정확히 일치**하고, 둘 다 *"손님이 말한 요구가
#           11월 판촉보다 우선"* 이라는 **하나의 읽기**로 성립한다(101만 그 읽기를 깬다).
#
# ⚠이 런은 지난 `bank_stage1b_20260808`의 **깨끗한 대조가 아니다**: C326으로 dense KB가
#   살아나 **환경 자체가 다르다**(지난 런은 두 sim 모두 첫 호출이 Missing credentials였다).
#   ⇒ 레버 귀속 런이 아니라 *"고쳐진 스택이 무엇을 하는가"* 를 보는 런이다. 원장에 그렇게 적는다.
#
# ★스택은 정본 go_stack 그대로([[60]] 전부 켠다). 플래그를 깎으면 재는 대상이 실제 것이 아니다.
# ★태그는 지난 런과 **분리**한다([[30]]: 같은 tag 재런 = 이전 데이터 덮어씀).
#
# usage: run_remeas_20260808b.sh [TASKS] [NT] [TAG]
set -e
REPO=/home/woori/workspace_common/boltzmann-attention-pi
cd "$REPO/scripts/distill/tau2"

TASKS="${1:-task_099,task_100}"
NT="${2:-1}"
TAG="${3:-bank_remeas_20260808b}"
LOG=/home/woori/scratch/logs
mkdir -p "$LOG"

# 선-점검 ①: 이미 도는 런이 있으면 멈춘다([[30]] 중복 실행·GPU 경합).
if ps -eo cmd | grep -v grep | grep -q "t2_run_gated.py"; then
  echo "[run] REFUSING: t2_run_gated 가 이미 돌고 있다. ps 로 확인하고 PID 지정 kill 후 재시도." >&2
  exit 1
fi
# 선-점검 ②: 같은 태그의 산출물이 있으면 멈춘다(덮어쓰기 방지).
if [ -e "$LOG/${TAG}.log" ]; then
  echo "[run] REFUSING: $LOG/${TAG}.log 가 이미 있다. 다른 TAG 를 쓰거나 먼저 영속화하라." >&2
  exit 1
fi

# 서브셸이 go_stack 을 **스스로 source** 한다 — `export -f` 로 함수를 실어 나르면
# 새로 생긴 `t2_require_key`(C326) 같은 의존 함수가 빠져 조용히 깨진다.
setsid bash -c "cd '$REPO/scripts/distill/tau2' && source ./go_stack.sh >/dev/null 2>&1 && \
  T2_FB_SIDECAR='$LOG/fb_${TAG}.jsonl' T2_FB_SIDECAR_TEXT=1 \
  t2_launch $TAG 8140 '$TASKS' $NT" \
  </dev/null >"$LOG/${TAG}.log" 2>&1 &
echo "PID=$!"
sleep 8
echo "--- 발사 직후 로그(키 로드·sim 태거 확인) ---"
head -25 "$LOG/${TAG}.log" 2>/dev/null || true
echo
echo "launched · tasks=$TASKS nt=$NT · log: $LOG/${TAG}.log · sidecar: $LOG/fb_${TAG}.jsonl"
