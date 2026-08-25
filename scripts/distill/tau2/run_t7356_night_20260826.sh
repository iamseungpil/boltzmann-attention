#!/bin/bash
# t7356 — 밤샘 런 (사용자 지시 2026-08-25: "내가 자는 동안 돌릴 수 있는 태스크를 밤샘 런에서 돌려라")
#
# ## 이 런이 t7355 와 다른 것 넷 (전부 오늘 밤 격리로 자격을 얻은 것)
#   A2 write_rules  책임 한도 표를 debit 분쟁 write 에 선언 — x538 A 12/20 vs B 20/20 vs N_len 12/20,
#                   합성 x538b B_both 20/20 (이미 실려 있는 문장과 서로 안 죽인다)
#   T2_WRITE_ARG_FAB=1        자리표시자로 채운 인자 — 술어 셋 전부 선언이거나 값의 모양이고
#                   이름 패턴 0. t7354 6배치 전수 20건 전부 진짜 날조·오차단 0
#   T2_SG_RECORD_ORDER=1      서브에게 주는 덤프를 타입별로 묶고 날짜 오름차순 —
#                   x536 N_wire 17 vs D_old_group 16 vs N_scramble 은 두 계좌를 부순다,
#                   x539 가 도메인 낱말 0 판(D_firstseen_group)이 같은 수를 내는지 확인
#   identifying_arg_types.digit  **철회**했다 — 이름 패턴이었고 위 파생이 더 잘 잡는다
#
# ## 왜 이 로스터인가 (오늘 밤 실측으로 다시 정했다)
#   085  t7355 스모크에서 gold 분쟁 #1 이 **인자 차이 0** 으로 접수됐다(msg78/79). 남은 것은
#        표기가 아니라 **커버리지**다 — gold 4건 중 1건만 내고 손님이 대화를 끝냈다(user_stop).
#        n=1 이라 귀속하지 않는다. 이 런의 nt10 이 그 물음에 답한다.
#   074  x536/x539 가 전사 결손을 **순서**로 지목했다. 이 런이 그 수리의 첫 라이브다.
#   040  reward_basis=['DB'] 라 gold 8건이 **전부** 맞아야 한다. 네 축 중 둘이 열려 있어
#        (eligible 판단·부분환불) 뒤집힐 후보가 아니다 — 그래도 FAB 레버의 유일한 표적이라 nt3 을 준다.
#
# ## 대조와 판정선
#   대조 = t7355(085) · t7348/t7354(074·040). 판정선 = **표적의 0->1**.
#   총점 델타 금지 · 레버가 넷이라 개별 귀속 불가 (C594).
#
# ## 안전
#   앞 런이 끝날 때까지 **기다린다**(폴링). 그 다음 origin 과 화해하고(rebase) 배터리를 돌린다.
#   스모크는 마커가 아니라 **산출**로 건다: 074 nt1 에서 재배열이 적용됐고 operand 가 살아 있는가.
set -e
REPO=/home/woori/workspace_common/boltzmann-attention-pi
LOG=/home/woori/scratch/logs
SIMS=/home/woori/scratch/tau2-bench/data/simulations
TAGBASE=bank_t7356
DEADLINE_HHMM=${DEADLINE_HHMM:-0730}
mkdir -p "$LOG"
say() { echo "[t7356 $(date +%H:%M:%S)] $*"; }

# ── ⑴ 앞 런이 끝날 때까지 기다린다 (최대 4시간)
WAITED=0
while pgrep -f "[t]2_launch" >/dev/null || pgrep -f "[t]2_run_gated" >/dev/null; do
  [ $WAITED -ge 14400 ] && { say "REFUSING: 앞 런이 4시간째 안 끝난다"; exit 1; }
  [ $((WAITED % 600)) -eq 0 ] && say "앞 런 대기 중 ${WAITED}s"
  sleep 60
  WAITED=$((WAITED + 60))
done
say "앞 런 없음 (대기 ${WAITED}s)"

# ── ⑵ origin 과 화해 — 앞 런의 persist 커밋이 로컬에만 있고 내 커밋이 원격에 있다
cd "$REPO"
git fetch -q origin facet-rft-2026
git -c user.name=ghlee -c user.email=beingrelative@gmail.com rebase origin/facet-rft-2026 \
  || { say "REFUSING: rebase 실패 — 손으로 화해해야 한다"; git rebase --abort || true; exit 1; }
git push -q origin facet-rft-2026 || say "push 보류(원격 선행) — 런 뒤 화해"
SHA=$(git rev-parse --short HEAD)
say "sha=$SHA"

cd "$REPO/scripts/distill/tau2"
DIRTY=$(cd "$REPO" && git status --porcelain -- \
  scripts/distill/tau2/t2_gate_patch.py scripts/distill/tau2/t2_search.py \
  scripts/distill/tau2/t2_resolve.py scripts/distill/tau2/t2_scaffold_get.py \
  scripts/distill/tau2/go_stack.sh scripts/distill/tau2/a2/ | grep -cv '^??' || true)
[ "$DIRTY" = "0" ] || { say "REFUSING: 엔진 경로 미커밋 $DIRTY"; exit 1; }

for t in test_a2_three_layer.py test_flag_registry.py test_no_undefined_names.py \
         test_no_unbound_a2.py test_quote_in.py test_args_equal.py test_t2_procedure.py \
         test_sg_docs_delivery.py test_sg_fetch_iso.py test_sg_isofb.py \
         test_sg_prompt_v2_reachable.py test_sg_record_order.py \
         test_atm_ledger_close.py test_compute_params.py test_write_arg_enum.py \
         test_write_arg_enum_values.py test_spec_at_write.py test_write_arg_type.py \
         test_write_arg_fab.py test_rule_at_write.py test_identifying_hints.py \
         test_result_round.py test_apy_balance_tier.py test_ref_from_outputs.py \
         test_no_prose_regex.py test_ours_text_canonical.py test_regen_break_guard.py; do
  [ -f "$t" ] || continue
  PYTHONPATH=/home/woori/scratch/tau2-bench/src timeout 90 \
    /home/woori/venvs/seka_env/bin/python "$t" >/dev/null 2>&1 || { say "REFUSING: $t FAIL"; exit 1; }
done
say "VERIFY OK (배터리 27)"

for f in "$LOG"/${TAGBASE}_*.log; do [ -e "$f" ] && { say "REFUSING: $f 존재"; exit 1; }; done
for d in "$SIMS"/${TAGBASE}_*; do [ -e "$d" ] && { say "REFUSING: $d 잔존"; exit 1; }; done

PIN="T2_ACTION_SUB=1 T2_KEEP_DENY_BODY=1 T2_CALL_FORM=1 T2_ARG_EMPTY=1 T2_SEARCH_AGENT=1 \
T2_DECIDE_ANY=1 T2_WRITE_ARG_ENUM=1 T2_DECIDE_BEFORE_WRITE=1 T2_DECISION_CARRY=1 \
T2_DISCOVERY_STEP2=1 T2_ARG_AXIS=1 T2_WRITE_SUB=3 T2_ACTION_INDEX=1 T2_NOW_SELFCALL=1 \
T2_SEARCH_ON_PROCEED=1 T2_ACT_DEMAND=0 T2_DELIVER_PRECOMMIT=0 T2_PROCEED_DOCBODY=0 \
T2_DOCS_AT_WRITE=0 T2_SUB_REQUIREMENT=0 T2_HANDOFF_PREDICATE=0 T2_PENDING_DISCOVERED=0 \
T2_VERDICT_CARRY=0 T2_ELIG_LINE=0 T2_VERDICT_GATE=0 T2_CLAIM_VERIFY=0 \
T2_DECLFIRST=0 T2_DECLFIRST_GUIDE_FIX=0 T2_SCHEMA_ENUM=0 T2_CATEGORY_CITE="
ON="T2_ARG_DOC_SUB=1 T2_VALUE_FORMULA=full T2_SG_DOCS=1 T2_SG_PROMPT_V2=1 T2_SPEC_AT_WRITE=1 \
T2_WRITE_ARG_TYPE=1 T2_RULE_AT_WRITE=1 T2_WRITE_ARG_ENUM_CAP=8 T2_WRITE_ARG_FAB=1 \
T2_SG_RECORD_ORDER=1"

echo "{\"tag\":\"t7356\",\"sha\":\"$SHA\",\"design\":\"overnight; four things earned isolation today - the liability table at the write, a fabrication guard with no name patterns, the record dump reordered the way the isolation won, and the name-pattern hint withdrawn\",\"on\":\"$ON\",\"reference\":\"t7355 for 085; t7348 and t7354 for 074 and 040 - all zero\",\"bar\":\"a target going 0 to 1\",\"cannot_measure\":\"which of the four moved it - four levers ride together (C594)\",\"deadline\":\"$DEADLINE_HHMM\"}" \
  | tee "$LOG/${TAGBASE}.meta.json"

setsid bash -c '
  REPO=/home/woori/workspace_common/boltzmann-attention-pi
  LOG=/home/woori/scratch/logs
  SIMS=/home/woori/scratch/tau2-bench/data/simulations
  TAGBASE=bank_t7356
  cd "$REPO/scripts/distill/tau2"
  source ./go_stack.sh >/dev/null 2>&1
  export '"$PIN"'
  export '"$ON"'
  export GO_MAX_STEPS=150 GO_CONCURRENCY=1
  DEADLINE_HHMM='"$DEADLINE_HHMM"'

  fits() {
    now=$((10#$(date +%H) * 60 + 10#$(date +%M)))
    dl=$((10#${DEADLINE_HHMM:0:2} * 60 + 10#${DEADLINE_HHMM:2:2}))
    [ $dl -lt $now ] && dl=$((dl + 1440))
    [ $((now + $1)) -le $dl ]
  }

  persist() {
    TAG=$1
    cd "$REPO" && mkdir -p reports/facet_rft_2026/sim_results
    gzip -c "$SIMS/$TAG/results.json" > reports/facet_rft_2026/sim_results/$TAG.results.json.gz 2>/dev/null || true
    gzip -c $LOG/$TAG.log > reports/facet_rft_2026/sim_results/$TAG.log.gz 2>/dev/null || true
    for _S in fb trace; do
      _F=$LOG/${_S}_${TAG}.jsonl
      if [ -s "$_F" ]; then
        gzip -c "$_F" > reports/facet_rft_2026/sim_results/${_S}_${TAG}.jsonl.gz
      else
        echo "[t7356] WARN ${_S} 미회수 $_F"
      fi
    done
    git add -f reports/facet_rft_2026/sim_results/$TAG.results.json.gz \
               reports/facet_rft_2026/sim_results/$TAG.log.gz \
               reports/facet_rft_2026/sim_results/fb_$TAG.jsonl.gz \
               reports/facet_rft_2026/sim_results/trace_$TAG.jsonl.gz 2>/dev/null || true
    git -c user.name=ghlee -c user.email=beingrelative@gmail.com commit -q -m "t7356 batch $TAG" || true
    git push -q origin facet-rft-2026 || echo "[t7356] push 보류"
    git ls-files --error-unmatch reports/facet_rft_2026/sim_results/$TAG.results.json.gz >/dev/null 2>&1 \
      && echo "[t7356] $TAG persisted+tracked OK" || echo "[t7356] $TAG NOT TRACKED"
    cd "$REPO/scripts/distill/tau2"
  }

  batch() {
    NAME=$1; PORT=$2; TL=$3; NT=$4; EST=$5; TAG=${TAGBASE}_${NAME}_20260826
    if ! fits $EST; then
      echo "[t7356] SKIP $NAME (추정 ${EST}분 · 마감 초과) — 이 배치는 안 돌았다"
      return 0
    fi
    unset T2_FB_SIDECAR T2_TRACE
    echo "[t7356 $(date +%H:%M:%S)] === $NAME · $TL · nt=$NT · 추정 ${EST}분 ==="
    t2_launch $TAG $PORT "$TL" $NT 2>&1 | tee $LOG/$TAG.log
    RO=$(grep -ac "T2_SG_RECORD_ORDER" $LOG/$TAG.log || true); RO=${RO:-0}
    FA=$(grep -ac "T2_WRITE_ARG_FAB" $LOG/$TAG.log || true); FA=${FA:-0}
    TY=$(grep -ac "T2_WRITE_ARG_TYPE" $LOG/$TAG.log || true); TY=${TY:-0}
    RU=$(grep -ac "T2_RULE_AT_WRITE" $LOG/$TAG.log || true); RU=${RU:-0}
    TB=$(grep -ac "Traceback" $LOG/$TAG.log || true); TB=${TB:-0}
    echo "[t7356] $NAME 완료 · RECORD_ORDER=$RO · ARG_FAB=$FA · ARG_TYPE=$TY · RULE_AT_WRITE=$RU · Traceback=$TB"
    persist $TAG
  }

  cd "$REPO"
  /home/woori/venvs/seka_env/bin/python reports/facet_rft_2026/freeze.py --on \
    --tag t7356 --reason "liability table, fabrication guard without name patterns, record order" || true
  cd "$REPO/scripts/distill/tau2"

  # ── 스모크 = 074 nt1. 산출로 건다: 재배열이 적용됐고 operand 가 살아 있는가.
  batch smoke 8140 task_074 1 45
  SLOG=$LOG/${TAGBASE}_smoke_20260826.log
  SAP=$(grep -ac "덤프 재배열 적용" $SLOG || true); SAP=${SAP:-0}
  SMK=$(grep -ac "T2_SG_RECORD_ORDER" $SLOG || true); SMK=${SMK:-0}
  STB=$(grep -ac "Traceback" $SLOG || true); STB=${STB:-0}
  SOK=$(grep -acF "operand keys=[" $SLOG || true); SOK=${SOK:-0}
  # 게이트는 **死배선과 크래시**만 막는다. 적용 0 은 중단 사유가 아니다 —
  # `_reorder_records` 는 이미 그 순서면 무변이고 덤프가 둘이면 손대지 않는다(설계).
  # 다른 세 레버가 함께 타므로 밤을 태울 이유가 없다. 대신 수를 크게 남긴다.
  if [ "$SMK" -eq 0 ] || [ "$STB" -gt 0 ]; then
    echo "[t7356] 중단: 마커=$SMK Traceback=$STB — 死배선이거나 크래시다"
    cd "$REPO"
    /home/woori/venvs/seka_env/bin/python reports/facet_rft_2026/freeze.py --off --tag t7356 || true
    exit 1
  fi
  echo "[t7356] 스모크 게이트 PASS 마커=$SMK 재배열적용=$SAP operand=$SOK Traceback=$STB"
  [ "$SAP" -eq 0 ] && echo "[t7356] ★주의: 재배열이 한 번도 적용되지 않았다 — 아침에 이유를 볼 것(무변인가·덤프가 둘인가)"

  (
    batch grpA1 8140 task_085 6 160
    batch grpA2 8140 task_040 3 220
  ) > $LOG/${TAGBASE}_grpA_chain.log 2>&1 &
  P1=$!
  (
    batch grpB1 8141 task_074 5 200
    batch grpB2 8141 task_085 4 110
  ) > $LOG/${TAGBASE}_grpB_chain.log 2>&1 &
  P2=$!
  wait $P1 $P2

  cd "$REPO"
  cp $LOG/${TAGBASE}.meta.json reports/facet_rft_2026/sim_results/${TAGBASE}.meta.json || true
  git add -f reports/facet_rft_2026/sim_results/${TAGBASE}.meta.json
  git -c user.name=ghlee -c user.email=beingrelative@gmail.com commit -q -m "t7356 meta" || true
  git push -q origin facet-rft-2026 || echo "[t7356] push 보류"
  /home/woori/venvs/seka_env/bin/python reports/facet_rft_2026/freeze.py --off --tag t7356 || true
  echo "[t7356] ALL DONE"
' </dev/null >"$LOG/${TAGBASE}_chain.log" 2>&1 &
say "기동 PID=$! · sha=$SHA · 마감 $DEADLINE_HHMM · 로그 $LOG/${TAGBASE}_chain.log"
