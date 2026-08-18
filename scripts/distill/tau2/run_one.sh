#!/bin/bash
# run_one.sh — **한 태스크 · 한 팔** 실험 런처 (사용자 지시 2026-08-18:
#   *"gpu 0,1 에 73 과 50을 별개로 돌리고 별개로 원인 귀속하여 수정하라"*).
#
# 왜 따로 도는가: 두 수리를 한 런에 얹으면 **귀속이 섞인다**. 태스크마다 다른 결함이고
# 지표도 다르므로(050 = `T2_SG_REQREADS` 발화 · 073 = 환급 3건 실행) 한 번에 하나만 바꾼다.
#
# 사용: bash run_one.sh <task> <arm:ctl|treat> <port> <tag>
#   예: bash run_one.sh task_050 treat 8140 bank_t7315_050treat_20260818k
#
# 구성은 스모크·census 와 **바이트 동일**: PIN 동일 · nt=1 · GO_MAX_STEPS=150 · GO_CONCURRENCY=1.
# ⚠배터리는 여기서 안 돌린다(이미 통과한 상태에서 단발 실험용). 스택 변경 후 첫 런은
#   `run_t7314_fixsmoke_*.sh` 처럼 배터리를 도는 런처를 쓸 것.

set -e
TASK="$1"; ARM="$2"; PORT="$3"; TAG="$4"
[ -n "$TAG" ] || { echo "usage: run_one.sh <task> <ctl|treat> <port> <tag>" >&2; exit 1; }

REPO=/home/woori/workspace_common/boltzmann-attention-pi
LOG=/home/woori/scratch/logs
SIMS=/home/woori/scratch/tau2-bench/data/simulations
cd "$REPO/scripts/distill/tau2"

SHA=$(cd "$REPO" && git rev-parse --short HEAD)
DIRTY=$(cd "$REPO" && git status --porcelain -- \
  scripts/distill/tau2/t2_gate_patch.py scripts/distill/tau2/t2_search.py \
  scripts/distill/tau2/t2_resolve.py scripts/distill/tau2/t2_scaffold_get.py \
  scripts/distill/tau2/a2/ | grep -cv '^??' || true)
[ "$DIRTY" = "0" ] || { echo "[run_one] REFUSING: 엔진 경로 미커밋 변경 $DIRTY 개" >&2; exit 1; }
[ -e "$LOG/${TAG}.log" ] && { echo "[run_one] REFUSING: ${TAG}.log 존재" >&2; exit 1; }
[ -e "$SIMS/${TAG}" ] && { echo "[run_one] REFUSING: $SIMS/${TAG} 잔존" >&2; exit 1; }

# 팔 = 노브의 조합. **귀속을 가르려면 하나씩** 켜야 한다(사용자 지시 2026-08-18).
#   treat  = VC1 EL1 (합성)   ctl = VC0 EL0 (기준선)
#   vconly = VC1 EL0          elonly = VC0 EL1
# ★VG = VC **호출-트리거**(2026-08-18·C543ⓓ·`T2_VERDICT_GATE`). push 형 VC 는 결정점에 닿기만
#   하면 발화해 073 에서 음수였다(ctl 1.0 ↔ vconly 0.0). VG 는 후보를 먹는 호출이 실제로 나올
#   때만 판정한다 ⇒ 비-선택 태스크엔 **트리거 자체가 없다**.
#   ⚠VC 와 VG 를 **같이 켜는 팔은 없다** — 같은 판정을 두 번 사면 귀속이 섞인다.
#   vgate = VG1 EL0 (VC 자리 대체)   vgate_el = VG1 EL1 (t7314 treat 와 짝)
CV=0                     # 완료-주장 격리 검증(`cverify` 팔에서만 1)
case "$ARM" in
  treat)    VC=1; EL=1; VG=0 ;;
  ctl)      VC=0; EL=0; VG=0 ;;
  vconly)   VC=1; EL=0; VG=0 ;;
  elonly)   VC=0; EL=1; VG=0 ;;
  vgate)    VC=0; EL=0; VG=1 ;;
  vgate_el) VC=0; EL=1; VG=1 ;;
  cverify)  VC=0; EL=0; VG=0; CV=1 ;;   # 완료-주장 격리 검증만
  *) echo "[run_one] REFUSING: arm=$ARM (ctl|treat|vconly|elonly|vgate|vgate_el)" >&2; exit 1 ;;
esac

PIN="T2_ACTION_SUB=1 T2_KEEP_DENY_BODY=1 T2_CALL_FORM=1 T2_ARG_EMPTY=1 T2_SEARCH_AGENT=1 \
T2_DECIDE_ANY=1 T2_WRITE_ARG_ENUM=1 T2_DECIDE_BEFORE_WRITE=1 T2_DECISION_CARRY=1 \
T2_DISCOVERY_STEP2=1 T2_ARG_AXIS=1 T2_WRITE_SUB=3 T2_ACTION_INDEX=1 T2_NOW_SELFCALL=1 \
T2_SEARCH_ON_PROCEED=1 T2_ACT_DEMAND=0 T2_DELIVER_PRECOMMIT=0 T2_PROCEED_DOCBODY=0 \
T2_DOCS_AT_WRITE=0 T2_SUB_REQUIREMENT=0 T2_HANDOFF_PREDICATE=0 T2_PENDING_DISCOVERED=0"

echo "{\"tag\":\"$TAG\",\"scaffold_sha\":\"$SHA\",\"port\":$PORT,\"tasks\":\"$TASK\",\"arm\":\"$ARM\",\"nt\":1,\"verdict_carry\":\"$VC\",\"elig_line\":\"$EL\",\"verdict_gate\":\"$VG\",\"claim_verify\":\"$CV\",\"max_steps\":150,\"concurrency\":1,\"why\":\"single-task attribution run\"}" \
  | tee "$LOG/${TAG}.meta.json"

setsid bash -c "cd '$REPO/scripts/distill/tau2' && source ./go_stack.sh >/dev/null 2>&1 && \
  export $PIN && export T2_VERDICT_CARRY=$VC T2_ELIG_LINE=$EL T2_VERDICT_GATE=$VG T2_CLAIM_VERIFY=$CV && \
  export GO_MAX_STEPS=150 GO_CONCURRENCY=1 && \
  t2_launch $TAG $PORT '$TASK' 1" \
  </dev/null >"$LOG/${TAG}.log" 2>&1 &
echo "[run_one] $TASK/$ARM → PID=$! port=$PORT tag=$TAG sha=$SHA"
