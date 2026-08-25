#!/bin/bash
# t7357 — 오늘 아침 확정된 두 수리의 첫 라이브 (사용자 지시 2026-08-26: "돌려라")
#
# ## 무엇이 새로 실리나 — 둘 다 오늘 아침 측정으로 자격을 얻었다
#
# ⑴ A2 `{delta_total}` (플래그 없음·선언이라 항상 켜진다)
#    엔진은 2026-08-13 부터 delta 들의 **부호 합**을 템플릿 인자로 내놓고 있었는데
#    ATM 수수료 도구의 return_template 이 그것을 **안 썼다** = 死설정.
#    t7356 실측이 대가를 보여준다 - 모델이 부호를 버리고 절댓값을 더한다:
#      lb 부호합 14.50 ↔ 제출 19.50 · dg 4.75 ↔ 10.25 · ev 3.70 ↔ 9.30
#      purple 은 **음수가 하나도 없어** 27.00 으로 유일하게 정확했다(자연 실험)
#      072 도 동형: 3.50 ↔ 6.50
#    x542(창 6·팔 5·n4, 계기 공정 = A_asis 가 라이브 제출값 재현):
#      A_asis 0/24 · B_fmt(우리 렌더링 교정) 0/24 · C_sign(부호 명시 문장) 0/24 ·
#      D_both 0/24 · N_len 0/24   ⇒ **전달로는 못 산다**([[62]]③ 그 단계에만 결정론)
#
# ⑵ 호출 형식 3단계 (T2_GIVE_REQUIRED=1 · T2_CALL_FORM_FIX=1)
#    사용자 확정: 내용(도구·인자)=LLM · **형식(어느 래퍼로 부르나)=엔진**.
#    ② 정확한 호출을 지목해 재생성 채널로 되돌린다 → ③ 그래도 안 되면 엔진이 래퍼를 바꿔 부른다.
#    표적(t7356 저장 궤적에 술어를 되돌린 would-fire): 017 2/2 · 040 1/1 · 055 1/2 · 085 1/3.
#    ⚠057 은 발화하지 않는다 - 거기선 손님이 그 도구를 **부르지도 않는다**(call 0·give 0).
#
# ## 로스터와 대조
#   grpA(8140) 074 → 072    = {delta_total} 표적
#   grpB(8141) 017 → 055 → 085 = 형식 3단계 표적
#   grpA 꼬리 040 nt1       = 형식 표적이면서 T2_WRITE_ARG_FAB 의 유일한 표적(계기)
#   대조 = **t7356**(같은 태스크·같은 로스터·어제 sha). 판정선 = 표적의 0->1.
#   ⛔두 수리가 서로 다른 태스크를 표적하므로 이번엔 **부분 귀속이 가능하다**:
#     074/072 가 움직이면 {delta_total} · 017/055/085 가 움직이면 형식 3단계.
set -e
REPO=/home/woori/workspace_common/boltzmann-attention-pi
LOG=/home/woori/scratch/logs
SIMS=/home/woori/scratch/tau2-bench/data/simulations
TAGBASE=bank_t7357
DEADLINE_HHMM=${DEADLINE_HHMM:-1600}
mkdir -p "$LOG"
say() { echo "[t7357 $(date +%H:%M:%S)] $*"; }

cd "$REPO"
git fetch -q origin facet-rft-2026
git -c user.name=ghlee -c user.email=beingrelative@gmail.com rebase --autostash origin/facet-rft-2026 \
  || { say "REFUSING: rebase 실패"; git rebase --abort || true; exit 1; }
git push -q origin facet-rft-2026 || say "push 보류(원격 선행)"
SHA=$(git rev-parse --short HEAD)
say "sha=$SHA"

cd "$REPO/scripts/distill/tau2"
DIRTY=$(cd "$REPO" && git status --porcelain -- \
  scripts/distill/tau2/t2_gate_patch.py scripts/distill/tau2/t2_search.py \
  scripts/distill/tau2/t2_resolve.py scripts/distill/tau2/t2_scaffold_get.py \
  scripts/distill/tau2/t2_compute.py \
  scripts/distill/tau2/go_stack.sh scripts/distill/tau2/a2/ | grep -cv '^??' || true)
[ "$DIRTY" = "0" ] || { say "REFUSING: 엔진 경로 미커밋 $DIRTY"; exit 1; }

for t in test_a2_three_layer.py test_flag_registry.py test_no_undefined_names.py \
         test_no_unbound_a2.py test_quote_in.py test_args_equal.py test_t2_procedure.py \
         test_sg_docs_delivery.py test_sg_fetch_iso.py test_sg_isofb.py \
         test_sg_prompt_v2_reachable.py test_sg_record_order.py \
         test_atm_ledger_close.py test_compute_params.py test_write_arg_enum.py \
         test_write_arg_enum_values.py test_spec_at_write.py test_write_arg_type.py \
         test_write_arg_fab.py test_rule_at_write.py test_identifying_hints.py \
         test_spec_arg_facts.py test_arg_policy_join.py \
         test_give_required.py test_delta_total_used.py \
         test_result_round.py test_apy_balance_tier.py test_ref_from_outputs.py \
         test_no_prose_regex.py test_ours_text_canonical.py test_regen_break_guard.py; do
  [ -f "$t" ] || continue
  PYTHONPATH=/home/woori/scratch/tau2-bench/src timeout 90 \
    /home/woori/venvs/seka_env/bin/python "$t" >/dev/null 2>&1 || { say "REFUSING: $t FAIL"; exit 1; }
done
say "VERIFY OK (배터리 31)"

for f in "$LOG"/${TAGBASE}_*.log; do [ -e "$f" ] && { say "REFUSING: $f 존재"; exit 1; }; done
for d in "$SIMS"/${TAGBASE}_*; do [ -e "$d" ] && { say "REFUSING: $d 잔존"; exit 1; }; done

PIN="T2_ACTION_SUB=1 T2_KEEP_DENY_BODY=1 T2_CALL_FORM=1 T2_ARG_EMPTY=1 T2_SEARCH_AGENT=1 \
T2_DECIDE_ANY=1 T2_WRITE_ARG_ENUM=1 T2_DECIDE_BEFORE_WRITE=1 T2_DECISION_CARRY=1 \
T2_DISCOVERY_STEP2=1 T2_ARG_AXIS=1 T2_WRITE_SUB=3 T2_ACTION_INDEX=1 T2_NOW_SELFCALL=1 \
T2_SEARCH_ON_PROCEED=1 T2_ACT_DEMAND=0 T2_DELIVER_PRECOMMIT=0 T2_PROCEED_DOCBODY=0 \
T2_DOCS_AT_WRITE=0 T2_SUB_REQUIREMENT=0 T2_HANDOFF_PREDICATE=0 T2_PENDING_DISCOVERED=0 \
T2_VERDICT_CARRY=0 T2_ELIG_LINE=0 T2_VERDICT_GATE=0 T2_CLAIM_VERIFY=0 \
T2_DECLFIRST=0 T2_DECLFIRST_GUIDE_FIX=0 T2_SCHEMA_ENUM=0 T2_ARG_POLICY_AT_WRITE=0 \
T2_CATEGORY_CITE="
ON="T2_ARG_DOC_SUB=1 T2_VALUE_FORMULA=full T2_SG_DOCS=1 T2_SG_PROMPT_V2=1 T2_SPEC_AT_WRITE=1 \
T2_WRITE_ARG_TYPE=1 T2_RULE_AT_WRITE=1 T2_WRITE_ARG_ENUM_CAP=8 T2_WRITE_ARG_FAB=1 \
T2_SG_RECORD_ORDER=1 T2_SPEC_ARG_FACTS=1 T2_GIVE_REQUIRED=1 T2_CALL_FORM_FIX=1"

echo "{\"tag\":\"t7357\",\"sha\":\"$SHA\",\"design\":\"two repairs that earned their measurement this morning - the signed total the tool had already computed but never printed, and a three-stage call-form correction where the engine may finally re-issue the same call in the form the environment accepts\",\"on\":\"$ON\",\"reference\":\"t7356 - same tasks, all zero\",\"bar\":\"a target going 0 to 1; 074/072 attribute to delta_total, 017/055/085 to the call form\",\"deadline\":\"$DEADLINE_HHMM\"}" \
  | tee "$LOG/${TAGBASE}.meta.json"

setsid bash -c '
  REPO=/home/woori/workspace_common/boltzmann-attention-pi
  LOG=/home/woori/scratch/logs
  SIMS=/home/woori/scratch/tau2-bench/data/simulations
  TAGBASE=bank_t7357
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
        echo "[t7357] WARN ${_S} 미회수 $_F"
      fi
    done
    git add -f reports/facet_rft_2026/sim_results/$TAG.results.json.gz \
               reports/facet_rft_2026/sim_results/$TAG.log.gz \
               reports/facet_rft_2026/sim_results/fb_$TAG.jsonl.gz \
               reports/facet_rft_2026/sim_results/trace_$TAG.jsonl.gz 2>/dev/null || true
    git -c user.name=ghlee -c user.email=beingrelative@gmail.com commit -q -m "t7357 batch $TAG" \
      -- reports/facet_rft_2026/sim_results/ || true
    git push -q origin facet-rft-2026 || echo "[t7357] push 보류"
    git ls-files --error-unmatch reports/facet_rft_2026/sim_results/$TAG.results.json.gz >/dev/null 2>&1 \
      && echo "[t7357] $TAG persisted+tracked OK" || echo "[t7357] $TAG NOT TRACKED"
    cd "$REPO/scripts/distill/tau2"
  }

  batch() {
    NAME=$1; PORT=$2; TL=$3; NT=$4; EST=$5; TAG=${TAGBASE}_${NAME}_20260826
    if ! fits $EST; then
      echo "[t7357] SKIP $NAME (추정 ${EST}분 · 마감 초과) — 이 배치는 안 돌았다"
      return 0
    fi
    unset T2_FB_SIDECAR T2_TRACE
    echo "[t7357 $(date +%H:%M:%S)] === $NAME · $TL · nt=$NT · 추정 ${EST}분 ==="
    t2_launch $TAG $PORT "$TL" $NT 2>&1 | tee $LOG/$TAG.log
    GR=$(grep -ac "T2_GIVE_REQUIRED" $LOG/$TAG.log || true); GR=${GR:-0}
    CF=$(grep -ac "T2_CALL_FORM_FIX" $LOG/$TAG.log || true); CF=${CF:-0}
    TB=$(grep -ac "Traceback" $LOG/$TAG.log || true); TB=${TB:-0}
    echo "[t7357] $NAME 완료 · GIVE_REQUIRED=$GR · CALL_FORM_FIX=$CF · Traceback=$TB"
    persist $TAG
  }

  cd "$REPO"
  /home/woori/venvs/seka_env/bin/python reports/facet_rft_2026/freeze.py --on \
    --tag t7357 --reason "the signed total is printed, and the call form is corrected in three stages" || true
  cd "$REPO/scripts/distill/tau2"

  # ── grpB(8141) = 형식 3단계 표적. 8140 의 스모크와 병렬로 먼저 띄운다.
  (
    batch grpB1 8141 task_017 3  80
    batch grpB2 8141 task_055 3  95
    batch grpB3 8141 task_085 3 125
  ) > $LOG/${TAGBASE}_grpB_chain.log 2>&1 &
  P2=$!

  # ── 스모크 = 074 nt1. **산출로 건다**: 도구 출력에 부호 합 문장이 실제로 실렸는가.
  batch smoke 8140 task_074 1 45
  SGN=$(/home/woori/venvs/seka_env/bin/python - <<PY
import sys, gzip, json
sys.path.insert(0, ".")
import t2_forensic as F
n = 0
try:
    for s in F.sims("bank_t7357_smoke_20260826", ".results.json.gz"):
        for m in (s.get("messages") or []):
            if "signed total of the differences" in str(m.get("content") or ""):
                n += 1
except Exception:
    n = -1
print(n)
PY
)
  echo "[t7357] 스모크 산출 게이트: 부호합 문장이 실린 도구 메시지 = $SGN"
  GATE=1
  if [ "${SGN:-0}" -le 0 ]; then
    GATE=0
    echo "[t7357] ⛔게이트 FAIL — {delta_total} 이 라이브 출력에 도달하지 않았다. grpA 건너뜀(grpB 는 계속)"
  fi

  (
    if [ "$GATE" = "1" ]; then
      batch grpA1 8140 task_074 3 130
      batch grpA2 8140 task_072 4  85
      batch grpA3 8140 task_040 1  95
    else
      echo "[t7357] grpA 건너뜀 — 스모크 산출 게이트 FAIL"
    fi
  ) > $LOG/${TAGBASE}_grpA_chain.log 2>&1 &
  P1=$!
  wait $P1 $P2

  cd "$REPO"
  cp $LOG/${TAGBASE}.meta.json reports/facet_rft_2026/sim_results/${TAGBASE}.meta.json || true
  git add -f reports/facet_rft_2026/sim_results/${TAGBASE}.meta.json
  git -c user.name=ghlee -c user.email=beingrelative@gmail.com commit -q -m "t7357 meta" \
    -- reports/facet_rft_2026/sim_results/ || true
  git push -q origin facet-rft-2026 || echo "[t7357] push 보류"
  /home/woori/venvs/seka_env/bin/python reports/facet_rft_2026/freeze.py --off --tag t7357 || true
  echo "[t7357] ALL DONE"
' </dev/null >"$LOG/${TAGBASE}_chain.log" 2>&1 &
say "기동 PID=$! · sha=$SHA · 마감 $DEADLINE_HHMM · 로그 $LOG/${TAGBASE}_chain.log"
