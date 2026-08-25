#!/bin/bash
# t7358 — **수리가 됐는지만 본다** (사용자 지시 2026-08-26: "1개 태스크씩 문제 해결됐는지 먼저
# 확인하고 nt=1 1개 태스크로 하라")
#
# 넓게 돌리기 전에 짧게 확인한다. 두 수리가 **서로 다른 태스크**를 표적하므로 GPU 하나에 하나씩:
#
#   8140  task_074 nt1   ← A2 `{delta_total}`  (엔진이 이미 계산해 둔 부호 합을 인쇄)
#   8141  task_017 nt1   ← 호출 형식 3단계 (T2_GIVE_REQUIRED → T2_CALL_FORM_FIX)
#
# ## 무엇을 보면 "됐다" 인가 — 산출로 정한다(마커만으로는 안 된다)
#
#   074 : ⓐ 도구 출력에 "signed total of the differences" 가 실렸는가 (배달)
#         ⓑ 제출한 크레딧 `amount` 가 그 **부호 합**과 같은가 (수용)
#         t7356 대조: lb 14.50↔19.50 · dg 4.75↔10.25 · ev 3.70↔9.30 (절댓값 합을 냈다)
#   017 : ⓐ [T2_GIVE_REQUIRED] 가 발화했는가 (②단계 도달)
#         ⓑ `give_discoverable_user_tool` 이 실제로 불렸는가 (모델이 따랐거나 ③이 부른 것)
#         ⓒ [T2_CALL_FORM_FIX] 가 발화했는가 (②로 안 돼서 ③까지 갔는가)
#         t7356 대조: trial0 은 give 0회 · 손님 호출 4회가 전부 거절 · user_stop
#
# 각 배치 끝에서 위를 **자동으로 세어** 로그에 찍는다. 통과하면 그때 넓힌다.
set -e
REPO=/home/woori/workspace_common/boltzmann-attention-pi
LOG=/home/woori/scratch/logs
SIMS=/home/woori/scratch/tau2-bench/data/simulations
TAGBASE=bank_t7358
mkdir -p "$LOG"
say() { echo "[t7358 $(date +%H:%M:%S)] $*"; }

cd "$REPO"
git fetch -q origin facet-rft-2026
git -c user.name=ghlee -c user.email=beingrelative@gmail.com rebase --autostash origin/facet-rft-2026 \
  || { say "REFUSING: rebase 실패"; git rebase --abort || true; exit 1; }
git push -q origin facet-rft-2026 || say "push 보류"
SHA=$(git rev-parse --short HEAD)
say "sha=$SHA"

cd "$REPO/scripts/distill/tau2"
DIRTY=$(cd "$REPO" && git status --porcelain -- \
  scripts/distill/tau2/t2_gate_patch.py scripts/distill/tau2/t2_scaffold_get.py \
  scripts/distill/tau2/t2_compute.py scripts/distill/tau2/go_stack.sh \
  scripts/distill/tau2/a2/ | grep -cv '^??' || true)
[ "$DIRTY" = "0" ] || { say "REFUSING: 엔진 경로 미커밋 $DIRTY"; exit 1; }

for t in test_a2_three_layer.py test_flag_registry.py test_no_undefined_names.py \
         test_no_unbound_a2.py test_give_required.py test_delta_total_used.py \
         test_sg_record_order.py test_spec_arg_facts.py test_write_arg_fab.py \
         test_rule_at_write.py test_identifying_hints.py test_arg_policy_join.py; do
  [ -f "$t" ] || continue
  PYTHONPATH=/home/woori/scratch/tau2-bench/src timeout 90 \
    /home/woori/venvs/seka_env/bin/python "$t" >/dev/null 2>&1 || { say "REFUSING: $t FAIL"; exit 1; }
done
say "VERIFY OK (배터리 12)"

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

echo "{\"tag\":\"t7358\",\"sha\":\"$SHA\",\"design\":\"one task each, one trial each - does the repair actually land before anything is scaled up\",\"on\":\"$ON\",\"reference\":\"t7356\",\"bar\":\"074 - the submitted credit equals the signed total; 017 - the handover call is actually made\"}" \
  | tee "$LOG/${TAGBASE}.meta.json"

setsid bash -c '
  REPO=/home/woori/workspace_common/boltzmann-attention-pi
  LOG=/home/woori/scratch/logs
  SIMS=/home/woori/scratch/tau2-bench/data/simulations
  TAGBASE=bank_t7358
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
      [ -s "$_F" ] && gzip -c "$_F" > reports/facet_rft_2026/sim_results/${_S}_${TAG}.jsonl.gz
    done
    git add -f reports/facet_rft_2026/sim_results/$TAG.results.json.gz \
               reports/facet_rft_2026/sim_results/$TAG.log.gz \
               reports/facet_rft_2026/sim_results/fb_$TAG.jsonl.gz \
               reports/facet_rft_2026/sim_results/trace_$TAG.jsonl.gz 2>/dev/null || true
    git -c user.name=ghlee -c user.email=beingrelative@gmail.com commit -q -m "t7358 batch $TAG" \
      -- reports/facet_rft_2026/sim_results/ || true
    git push -q origin facet-rft-2026 || echo "[t7358] push 보류"
    cd "$REPO/scripts/distill/tau2"
  }

  verdict() {   # $1 태그 · $2 태스크
    /home/woori/venvs/seka_env/bin/python - "$1" "$2" <<PY
import sys, json
sys.path.insert(0, ".")
import t2_forensic as F
tag, task = sys.argv[1], sys.argv[2]
try:
    sims = F.sims(tag, ".results.json.gz")
except Exception as e:
    print("[verdict] 결과 없음: %r" % (e,)); raise SystemExit
for s in sims:
    rw = (s.get("reward_info") or {}).get("reward")
    ms = s.get("messages") or []
    print("[verdict] %s reward=%s term=%s msgs=%d" % (task, rw, s.get("termination_reason"), len(ms)))
    if task == "task_074":
        import re
        blocks = [str(m.get("content") or "") for m in ms
                  if "whose net charge does NOT match" in str(m.get("content") or "")]
        printed = sum(1 for b in blocks if "signed total of the differences" in b)
        print("[verdict] 074 (a) 부호합 문장이 실린 도구 출력 = %d / %d" % (printed, len(blocks)))
        want = {}
        for b in blocks:
            vals = [float(x) for x in re.findall(r"difference \\\$(-?[\\d.]+)", b)]
            t = re.search(r"is (-?[\\d.]+) \\(a negative", b)
            want[round(sum(vals), 2)] = (round(sum(vals), 2), float(t.group(1)) if t else None)
        subs = []
        for m in ms:
            for tc in (m.get("tool_calls") or []):
                a = F.argsof(tc) or {}
                inner = a.get("arguments")
                if isinstance(inner, str):
                    try: inner = json.loads(inner)
                    except Exception: inner = {}
                if not isinstance(inner, dict): inner = {}
                nm = str(a.get("agent_tool_name") or "")
                if "apply_checking_account_credit" in nm:
                    try: subs.append(round(float(str(inner.get("amount")).replace("\\$","")), 2))
                    except Exception: pass
        print("[verdict] 074 (b) 제출 amount = %s · 블록 부호합 = %s" % (subs, sorted(want)))
        print("[verdict] 074 (b) 일치 = %d / %d" % (len([x for x in subs if x in want]), len(subs)))
    if task == "task_017":
        gave = [1 for m in ms for tc in (m.get("tool_calls") or [])
                if str(tc.get("name")) == "give_discoverable_user_tool"]
        print("[verdict] 017 (b) give 호출 = %d 회  (t7356 trial0 = 0)" % len(gave))
PY
  }

  batch() {
    NAME=$1; PORT=$2; TL=$3; TAG=${TAGBASE}_${NAME}_20260826
    unset T2_FB_SIDECAR T2_TRACE
    echo "[t7358 $(date +%H:%M:%S)] === $NAME · $TL · nt=1 ==="
    t2_launch $TAG $PORT "$TL" 1 2>&1 | tee $LOG/$TAG.log
    GR=$(grep -ac "T2_GIVE_REQUIRED" $LOG/$TAG.log || true); GR=${GR:-0}
    CF=$(grep -ac "T2_CALL_FORM_FIX" $LOG/$TAG.log || true); CF=${CF:-0}
    TB=$(grep -ac "Traceback" $LOG/$TAG.log || true); TB=${TB:-0}
    echo "[t7358] $NAME 완료 · GIVE_REQUIRED=$GR · CALL_FORM_FIX=$CF · Traceback=$TB"
    persist $TAG
    verdict $TAG $TL
  }

  ( batch d074 8140 task_074 ) > $LOG/${TAGBASE}_A.log 2>&1 &
  P1=$!
  ( batch f017 8141 task_017 ) > $LOG/${TAGBASE}_B.log 2>&1 &
  P2=$!
  wait $P1 $P2
  echo "[t7358] ALL DONE"
' </dev/null >"$LOG/${TAGBASE}_chain.log" 2>&1 &
say "기동 PID=$! · sha=$SHA · 로그 $LOG/${TAGBASE}_chain.log · A=$LOG/${TAGBASE}_A.log B=$LOG/${TAGBASE}_B.log"
