#!/bin/bash
# t7355 — 085 표적 단일 배치 (사용자 지시 2026-08-25 밤: "비용은 많이 써도 된다. 시간이 제일 아깝다")
#
# ## 왜 085 인가 — t7354 실측이 gold 13 중 10 일치를 보여준다
#   정본 t2_forensic.action_diff 로 gold 행과 궤적 호출을 자연키로 짝지은 결과(grpA1 t0),
#   미달은 file_debit_card_transaction_dispute_6281 한 행뿐이고 그 행의 차이가 전부 표기다:
#     불리언 5  card_in_possession / contacted_merchant / police_report_filed /
#               written_statement_provided / provisional_credit_eligible = 전부 문자열 Yes|No
#     열거   4  transaction_type  'ATM Withdrawal'  vs gold 'atm_withdrawal'
#               dispute_category  산문             vs gold 'atm_cash_discrepancy'
#               pin_compromised   'No'             vs gold 'no'   (소문자)
#               card_action       'None'           vs gold 'keep_active'
#     판단   1  customer_max_liability_amount '0'  vs gold '50'   <- 이 런은 이것을 못 산다
#   => 이 런이 사는 것은 앞의 아홉이고, 열 번째가 남으면 0 그대로다. 그것이 반증조건이다.
#
# ## 이 런에 처음 실리는 것 셋 (전부 라이브 미측정)
#   T2_WRITE_ARG_TYPE=1   선언된 불리언에 문자열을 보내면 되돌려준다 (변환 0 / 선택 0)
#   T2_RULE_AT_WRITE=1    선언된 절차 문장을 write 결정점에 (격리 x537 A 0/12 vs B 12/12 vs N 0/12)
#   A2 write_arg_enum 값 목록 6칸 신설 — 출처는 도구 사용법 문서 축자뿐 ([[23]])
#   상한을 올린다: T2_WRITE_ARG_ENUM_CAP=8. 이유 = 한 턴에 한 칸만 되돌려주는데(en_fb 는 첫
#   위반에서 break) 085 의 한 호출에 어긋난 열거가 넷이라 기본 3 으로는 마지막 칸을 못 본다.
#   [[70]] 무엇을 파나 = 재시도 턴이 늘고 문맥이 커진다.
#
# ## 왜 단일 배치인가 — 자정에 두 GPU 로 갈아탄다
#   8141 은 이 시간 동안 무료 프로브가 쓴다(포트 분리 [[30]]). 이 배치가 끝나면 그때 새 코드를
#   pull 해 t7356 을 양쪽 GPU 에 건다. 런 도중 pull 하면 뒤 배치가 다른 엔진으로 돌아 균일성이
#   깨지므로 배치를 하나로 둔다.
#
# ## 스모크 = 마커가 아니라 산출로 (핸드오프 2026-08-25 밤 §6 의 교훈)
#   085 nt1 을 먼저 돌려 T2_WRITE_ARG_TYPE 과 T2_WRITE_ARG_ENUM 이 둘 다 발화했는지 본다.
#   0 이면 중단한다 — 오늘 아침 V2 가 격리 통과 후 라이브에서 죽은 전례가 있다.
#   그 sim 은 버리지 않고 실 trial 로 센다.
#
# ## 대조와 판정선
#   대조 = t7354(sha 730abb63) 085 0/4 · t7348 085 0/2. 판정선 = 085 의 0->1.
#   총점 델타 금지 · 이 런은 세 레버가 함께라 개별 귀속 불가 (C594).
set -e
REPO=/home/woori/workspace_common/boltzmann-attention-pi
LOG=/home/woori/scratch/logs
SIMS=/home/woori/scratch/tau2-bench/data/simulations
TAGBASE=bank_t7355
mkdir -p "$LOG"
cd "$REPO/scripts/distill/tau2"
SHA=$(cd "$REPO" && git rev-parse --short HEAD)
say() { echo "[t7355 $(date +%H:%M:%S)] $*"; }

DIRTY=$(cd "$REPO" && git status --porcelain -- \
  scripts/distill/tau2/t2_gate_patch.py scripts/distill/tau2/t2_search.py \
  scripts/distill/tau2/t2_resolve.py scripts/distill/tau2/t2_scaffold_get.py \
  scripts/distill/tau2/go_stack.sh scripts/distill/tau2/a2/ | grep -cv '^??' || true)
[ "$DIRTY" = "0" ] || { say "REFUSING: 엔진 경로 미커밋 $DIRTY"; exit 1; }

for t in test_a2_three_layer.py test_flag_registry.py test_no_undefined_names.py \
         test_no_unbound_a2.py test_quote_in.py test_args_equal.py test_t2_procedure.py \
         test_sg_docs_delivery.py test_sg_fetch_iso.py test_sg_isofb.py \
         test_sg_prompt_v2_reachable.py \
         test_atm_ledger_close.py test_compute_params.py test_write_arg_enum.py \
         test_write_arg_enum_values.py test_spec_at_write.py test_write_arg_type.py \
         test_result_round.py test_apy_balance_tier.py test_ref_from_outputs.py \
         test_no_prose_regex.py test_ours_text_canonical.py test_regen_break_guard.py; do
  [ -f "$t" ] || continue
  PYTHONPATH=/home/woori/scratch/tau2-bench/src timeout 90 \
    /home/woori/venvs/seka_env/bin/python "$t" >/dev/null 2>&1 || { say "REFUSING: $t FAIL"; exit 1; }
done
say "VERIFY OK (배터리 23)"

pgrep -f "[t]2_launch" >/dev/null && { say "REFUSING: 다른 라이브 런"; exit 1; } || true
pgrep -f "[t]2_run_gated" >/dev/null && { say "REFUSING: 잔존 sim 프로세스"; exit 1; } || true
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
T2_WRITE_ARG_TYPE=1 T2_RULE_AT_WRITE=1 T2_WRITE_ARG_ENUM_CAP=8"

echo "{\"tag\":\"t7355\",\"sha\":\"$SHA\",\"design\":\"085 only; the one missing gold row differs from gold in argument NOTATION alone - five booleans sent as Yes/No strings and four enum values sent in prose or wrong case\",\"on\":\"$ON\",\"reference\":\"t7354 sha 730abb63 - 085 was 0/4; t7348 085 0/2\",\"bar\":\"085 going 0 to 1\",\"cannot_measure\":\"customer_max_liability_amount 0 vs 50 - that is a judgment and nothing here supplies it\"}" \
  | tee "$LOG/${TAGBASE}.meta.json"

setsid bash -c '
  REPO=/home/woori/workspace_common/boltzmann-attention-pi
  LOG=/home/woori/scratch/logs
  SIMS=/home/woori/scratch/tau2-bench/data/simulations
  TAGBASE=bank_t7355
  cd "$REPO/scripts/distill/tau2"
  source ./go_stack.sh >/dev/null 2>&1
  export '"$PIN"'
  export '"$ON"'
  export GO_MAX_STEPS=150 GO_CONCURRENCY=1

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
        echo "[t7355] WARN ${_S} 미회수 $_F — 우리-층 귀속 판정 불가"
      fi
    done
    git add -f reports/facet_rft_2026/sim_results/$TAG.results.json.gz \
               reports/facet_rft_2026/sim_results/$TAG.log.gz \
               reports/facet_rft_2026/sim_results/fb_$TAG.jsonl.gz \
               reports/facet_rft_2026/sim_results/trace_$TAG.jsonl.gz 2>/dev/null || true
    git -c user.name=ghlee -c user.email=beingrelative@gmail.com commit -q -m "t7355 batch $TAG" || true
    git push -q origin facet-rft-2026 || echo "[t7355] push 보류(원격 선행)"
    git ls-files --error-unmatch reports/facet_rft_2026/sim_results/$TAG.results.json.gz >/dev/null 2>&1 \
      && echo "[t7355] $TAG persisted+tracked OK" || echo "[t7355] $TAG NOT TRACKED"
    cd "$REPO/scripts/distill/tau2"
  }

  batch() {
    NAME=$1; PORT=$2; TL=$3; NT=$4; TAG=${TAGBASE}_${NAME}_20260825
    unset T2_FB_SIDECAR T2_TRACE
    echo "[t7355 $(date +%H:%M:%S)] === $NAME · $TL · nt=$NT ==="
    t2_launch $TAG $PORT "$TL" $NT 2>&1 | tee $LOG/$TAG.log
    TY=$(grep -ac T2_WRITE_ARG_TYPE $LOG/$TAG.log || true); TY=${TY:-0}
    EN=$(grep -ac T2_WRITE_ARG_ENUM $LOG/$TAG.log || true); EN=${EN:-0}
    TB=$(grep -ac Traceback $LOG/$TAG.log || true); TB=${TB:-0}
    echo "[t7355] $NAME 완료 · WRITE_ARG_TYPE=$TY · WRITE_ARG_ENUM=$EN · Traceback=$TB"
    persist $TAG
  }

  cd "$REPO"
  /home/woori/venvs/seka_env/bin/python reports/facet_rft_2026/freeze.py --on \
    --tag t7355 --reason "boolean type and declared enum values reach the debit dispute write" || true
  cd "$REPO/scripts/distill/tau2"

  batch smoke 8140 task_085 1
  SLOG=$LOG/${TAGBASE}_smoke_20260825.log
  STY=$(grep -ac T2_WRITE_ARG_TYPE $SLOG || true); STY=${STY:-0}
  SEN=$(grep -ac T2_WRITE_ARG_ENUM $SLOG || true); SEN=${SEN:-0}
  if [ "$STY" -eq 0 ] || [ "$SEN" -eq 0 ]; then
    echo "[t7355] 중단: 스모크에서 레버가 발화하지 않았다 TYPE=$STY ENUM=$SEN — 死배선"
    cd "$REPO"
    /home/woori/venvs/seka_env/bin/python reports/facet_rft_2026/freeze.py --off --tag t7355 || true
    exit 1
  fi
  echo "[t7355] 스모크 게이트 PASS TYPE=$STY ENUM=$SEN"

  batch main 8140 task_085 5

  cd "$REPO"
  cp $LOG/${TAGBASE}.meta.json reports/facet_rft_2026/sim_results/${TAGBASE}.meta.json || true
  git add -f reports/facet_rft_2026/sim_results/${TAGBASE}.meta.json
  git -c user.name=ghlee -c user.email=beingrelative@gmail.com commit -q -m "t7355 meta" || true
  git push -q origin facet-rft-2026 || echo "[t7355] push 보류"
  /home/woori/venvs/seka_env/bin/python reports/facet_rft_2026/freeze.py --off --tag t7355 || true
  echo "[t7355] ALL DONE"
' </dev/null >"$LOG/${TAGBASE}_chain.log" 2>&1 &
say "기동 PID=$! · sha=$SHA · 로그 $LOG/${TAGBASE}_chain.log"
