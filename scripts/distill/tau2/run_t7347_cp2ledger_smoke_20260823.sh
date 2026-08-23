#!/bin/bash
# t7347 — **CP2 배달 원장 스모크** (Stage 1 · 2026-08-23 · 4 sim · 유료 소량).
#
# ## 무엇을 확인하나 — 성적이 아니라 **계기**다
#   R4 로 넣은 배달 생애 원장(`_cp2_open`/`_cp2_close`)이 라이브에서 실제로 돌고, 사이드카가
#   회수되고, **닫힌 분할 검산식이 서는지**만 본다:
#       대입 = attached + clobbered + ctx_skip + 잔존
#   ⛔`T2_CP2_QUEUE` 는 **켜지 않는다**(기본 0). 이 런은 큐의 효과를 재지 않는다 — 계기만 본다.
#      큐 A/B(Stage 2)는 이 검산식이 선 뒤에만 돈다(감사 `CP2_QUEUE_AUDIT_2026_08_23.md` §5).
#
# ## 왜 이게 선결인가 (원장 C502)
#   t7303 A/B 가 무효가 된 이유는 결손이 아니라 계기였다 — 1차 종점이 `[T2_CP2_APPEND] … (queue)`
#   였는데 그 줄은 **꺼진 팔에서는 존재할 수 없다**. 그래서 "0/8 → 8/8" 은 처치 배정의 재인쇄였다.
#   그리고 지금 쓰던 계기도 같은 병이었다(실측): 보관 사이드카 14파일 전수에서 `decision_carry`
#   행 `arrived` 가 **100% True**(303행·False 0)이고 그 행 수는 도달 수가 아니라 **VIEW_FB 대입
#   수와 1:1** 이다 — 다섯 배달 자리 중 하나만 등재하고 있었다([[25]]).
#
# ## ★사이드카 회수 — 이 런이 고치는 배선
#   `go_stack.sh:222` 가 `T2_FB_SIDECAR=$LOG/fb_${TAG}.jsonl` 을 **이미** 쓰고 있는데 러너가
#   results/log 만 회수했다. 그래서 t7336·t7346 의 `fb_*.jsonl` 은 `sim_results/` 에 **없다**
#   (보관 마지막이 t7328). 아래에서 함께 회수한다.
#
# ## 표적 4 태스크 — cp2 트래픽이 실제로 있는 것들 (t7346 로그 실측)
#   098·057·063 = 그 런에서 CLOBBER 를 맞은 셋 · 093 = 배달 자리가 여럿 겹치는 태스크
#   ⚠성적은 이 런의 종점이 **아니다**. reward 는 인쇄만 하고 판정에 쓰지 않는다([[69]]).
set -e
REPO=/home/woori/workspace_common/boltzmann-attention-pi
LOG=/home/woori/scratch/logs
SIMS=/home/woori/scratch/tau2-bench/data/simulations
TAG=bank_t7347_cp2ledger_20260823
mkdir -p "$LOG"
cd "$REPO/scripts/distill/tau2"
SHA=$(cd "$REPO" && git rev-parse --short HEAD)

DIRTY=$(cd "$REPO" && git status --porcelain -- \
  scripts/distill/tau2/t2_gate_patch.py scripts/distill/tau2/t2_lever_beat.py \
  scripts/distill/tau2/t2_resolve.py scripts/distill/tau2/a2/ | grep -cv '^??' || true)
if [ "$DIRTY" != "0" ]; then
  echo "[t7347] REFUSING: 엔진 경로 커밋 안 된 변경 $DIRTY 개." >&2; exit 1
fi

for t in test_cp2_ledger.py test_cp2_queue_behavior.py test_cp2_clobber.py \
         test_proceed_docbody.py test_route_trace.py test_flag_registry.py \
         test_regen_break_guard.py test_unified_regen.py test_no_undefined_names.py \
         test_no_unbound_a2.py test_a2_three_layer.py \
         test_forensic_sidecar_authority.py; do
  [ -f "$t" ] || continue
  PYTHONPATH=/home/woori/scratch/tau2-bench/src timeout 90 \
    /home/woori/venvs/seka_env/bin/python "$t" >/dev/null 2>&1 \
    || { echo "[t7347] REFUSING: $t FAIL" >&2; exit 1; }
done
echo "[t7347] VERIFY OK (원장 검정 포함)"

if pgrep -f "[t]2_lau""nch" >/dev/null; then
  echo "[t7347] REFUSING: 다른 라이브 런이 돌고 있다" >&2; exit 1
fi
for f in "$LOG"/${TAG}*.log "$SIMS"/${TAG}*; do
  [ -e "$f" ] && { echo "[t7347] REFUSING: $f 잔존" >&2; exit 1; }
done

source ./go_stack.sh >/dev/null 2>&1
export T2_ACTION_SUB=1 T2_KEEP_DENY_BODY=1 T2_CALL_FORM=1 T2_ARG_EMPTY=1 T2_SEARCH_AGENT=1
export T2_DECIDE_ANY=1 T2_WRITE_ARG_ENUM=1 T2_DECIDE_BEFORE_WRITE=1 T2_DECISION_CARRY=1
export T2_DISCOVERY_STEP2=1 T2_ARG_AXIS=1 T2_WRITE_SUB=3 T2_ACTION_INDEX=1 T2_NOW_SELFCALL=1
export T2_SEARCH_ON_PROCEED=1 T2_ARG_DOC_SUB=1 T2_VALUE_FORMULA=full T2_SG_DOCS=1
export T2_CP2_QUEUE=0                     # ★이 런은 큐를 켜지 않는다 — 계기만 본다
export GO_MAX_STEPS=150 GO_CONCURRENCY=1

echo "{\"tag\":\"t7347\",\"sha\":\"$SHA\",\"design\":\"instrument smoke for the cp2 delivery ledger (R4); queue stays off\",\"tasks\":\"098,057 on 8140 and 063,093 on 8141, nt=1 = 4 sims\",\"endpoint\":\"ledger balance and sidecar recovery, not reward\"}" \
  | tee "$LOG/${TAG}.meta.json"

run_half() {
  NAME=$1; PORT=$2; TL=$3
  T=${TAG}_${NAME}
  t2_launch "$T" "$PORT" "$TL" 1 2>&1 | tee "$LOG/$T.log"
}
( run_half a 8140 'task_098,task_057' ) > "$LOG/${TAG}_a_chain.log" 2>&1 &
P1=$!
( run_half b 8141 'task_063,task_093' ) > "$LOG/${TAG}_b_chain.log" 2>&1 &
P2=$!
wait $P1 $P2

# ── 회수: results · log · ★사이드카 ────────────────────────────────────────────
cd "$REPO"
mkdir -p reports/facet_rft_2026/sim_results
for N in a b; do
  T=${TAG}_${N}
  gzip -c "$SIMS/$T/results.json" > "reports/facet_rft_2026/sim_results/$T.results.json.gz" || true
  gzip -c "$LOG/$T.log"           > "reports/facet_rft_2026/sim_results/$T.log.gz" || true
  # ★사이드카·계기 — 지금까지 아무 러너도 안 가져왔다. 둘 다 go_stack(222·231)이 만든다.
  #   왜 둘 다인가(2026-08-23 R2): 우리 층 거절은 재생성 채널로 나가고 `_ap_regen` 이 원
  #   어시스턴트 메시지를 **교체**하므로 영속 궤적에 안 남는다 — *무엇을* 말했나는 `fb_`,
  #   *어느 기구가* 말했나는 `trace_` 에만 있다. 없으면 **없다고 인쇄한다**(침묵≠부재·[[25]]).
  for _S in fb trace; do
    _F="$LOG/${_S}_$T.jsonl"
    if [ -s "$_F" ]; then
      gzip -c "$_F" > "reports/facet_rft_2026/sim_results/${_S}_$T.jsonl.gz"
      echo "[t7347] ${_S} 회수 $T ($(wc -l < "$_F") 행)"
    else
      echo "[t7347] ⚠${_S} 미회수: $_F 없음/빈 파일 — 이 런의 우리-층 귀속은 판정 불가([[25]]·[[55]])"
    fi
  done
done

# ── 검산 게이트 ───────────────────────────────────────────────────────────────
PYTHONIOENCODING=utf-8 /home/woori/venvs/seka_env/bin/python \
  scripts/distill/tau2/x490_cp2_ledger_audit.py \
  --tags ${TAG}_a,${TAG}_b --json ${TAG}_ledger.json | tee "$LOG/${TAG}_audit.log"

git add -f reports/facet_rft_2026/sim_results/${TAG}_*.gz \
           reports/facet_rft_2026/sim_results/fb_${TAG}_*.jsonl.gz \
           reports/facet_rft_2026/sim_results/trace_${TAG}_*.jsonl.gz \
           reports/facet_rft_2026/${TAG}_ledger.json 2>/dev/null || true
git -c user.name=ghlee -c user.email=beingrelative@gmail.com commit -q -m "t7347 cp2 ledger smoke" || true
git pull -q --rebase origin facet-rft-2026 || true
git push -q origin facet-rft-2026 || true
git ls-files --error-unmatch reports/facet_rft_2026/sim_results/${TAG}_a.results.json.gz >/dev/null 2>&1 \
  && echo "[t7347] persisted+tracked OK"
echo "[t7347] ALL DONE"
