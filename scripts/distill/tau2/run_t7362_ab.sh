#!/usr/bin/env bash
# t7362 A/B — 오늘 수리 중 **플래그가 달린 셋**을 가른다.
#
# ⛔`set -u` 금지 (t7360 교훈: go_stack 의 `t2_require_key` 가 미설정을 스스로 처리하는데
#   `set -u` 아래선 그 표현식이 먼저 죽어 t2_launch 가 한 줄도 안 돈다).
#
# ── 무엇을 가르나 ──────────────────────────────────────────────────────────
#   A_ctl   지금 기본값 그대로                      (대조)
#   B_say   PROCEDURE_LEFT=1 · EPLAN_ENUM_SUBTRACT=1  (**옳은 말을 하기**)
#   C_scope A_ctl + T2_SCOPE_ALL=1                  (**범위 표면화 되돌리기**)
#
# 왜 셋인가: 085 에 후보가 **둘**이고 방향이 반대다 — `SCOPE_ALL=1` 은 문면을 **늘리고**
#   `EPLAN_ENUM_SUBTRACT=1` 은 문면을 **고친다**. 한 팔에 묶으면 어느 것이 샀는지 못 가른다.
#   B 와 C 를 A 에 각각 대면 두 물음이 따로 답해진다([[70]] 태스크별 부호표).
#
# 항상 켜져 있는 오늘 수리 둘(`OPERATOR-DIRECT` · 연쇄 대상 명명)은 **세 팔 공통**이다 —
#   플래그가 없고, 허위/모호한 문면을 참말로 바꾼 것이라 되돌릴 팔을 두지 않는다.
#
# ── 표적과 대조 ────────────────────────────────────────────────────────────
#   050  PROCEDURE_LEFT 의 표적 (`left=['decision']` 을 알면서 통과시킨 자리)
#   085  SCOPE·EPLAN 둘 다의 표적 (t7360 done=5 ↔ t7361 done=2)
#   079  SCOPE 의 두 번째 표적 (반려 26건으로 085 다음)
#   074  세 팔 공통 대조 — 오늘 상시 수리 둘이 여기 걸린다
#   ⚠040 은 뺐다: 79분·turn 98 루프였고, 그 원인(ARG_POLICY 과부하)은 이미 껐다. 남은 축
#     `eligible_for_provisional_credit` 은 이 셋 중 무엇도 표적하지 않는다(C583 ⓑ 가 기록).
#   ⚠050·098·100 은 불안정 8 이다([[75]] 규칙 ③) — 050 은 표적이라 넣되 **단독 Δ 로
#     판정하지 마라**. 100 은 이번엔 뺐다(팔이 셋이라 sim 예산이 는다).
#
# ── 조건 ──────────────────────────────────────────────────────────────────
#   같은 sha · 같은 서버(8140·8141 은 **다른 모델**이라 못 쓴다) · 팔마다 GO_CONCURRENCY=1
#   ⇒ 서버 부하는 3 동시. **세 팔이 같은 부하를 받으므로 대조는 유효**하다(절대 지연만 는다).
set -o pipefail
REPO=/home/woori/workspace_common/boltzmann-attention-pi
LOG=/home/woori/scratch/logs
SIMS=/home/woori/scratch/tau2-bench/data/simulations
STAMP=20260826
TASKS="task_050,task_074,task_085,task_079"
cd "$REPO/scripts/distill/tau2"

python "$REPO/reports/facet_rft_2026/freeze.py" --on --tag "bank_t7362_ab_$STAMP" \
  --reason "t7362 A/B: procedure-left + eplan-subtract + scope-revert"

run_arm() {
  ARM="$1"; shift
  TAG="bank_t7362_${ARM}_${STAMP}"
  (
    source ./go_stack.sh >/dev/null 2>&1
    # t7361 과 **같은 레버 집합** — 오늘 판정과 이어 붙기 위해 노브를 더 바꾸지 않는다.
    export T2_ACTION_SUB=1 T2_KEEP_DENY_BODY=1 T2_CALL_FORM=1 T2_ARG_EMPTY=1 T2_SEARCH_AGENT=1 T2_DECIDE_ANY=1 T2_WRITE_ARG_ENUM=1 T2_DECIDE_BEFORE_WRITE=1 T2_DECISION_CARRY=1 T2_DISCOVERY_STEP2=1 T2_ARG_AXIS=1 T2_WRITE_SUB=3 T2_ACTION_INDEX=1 T2_NOW_SELFCALL=1 T2_SEARCH_ON_PROCEED=1
    export T2_ARG_DOC_SUB=1 T2_VALUE_FORMULA=full T2_SG_DOCS=1 T2_SG_PROMPT_V2=1 T2_SPEC_AT_WRITE=1 T2_WRITE_ARG_TYPE=1 T2_RULE_AT_WRITE=1 T2_WRITE_ARG_ENUM_CAP=8 T2_WRITE_ARG_FAB=1 T2_SG_RECORD_ORDER=1 T2_SPEC_ARG_FACTS=1 T2_GIVE_REQUIRED=1 T2_CALL_FORM_FIX=1
    export T2_DUP_WRITE=1
    export GO_MAX_STEPS=150 GO_CONCURRENCY=1
    # 팔 고유 노브 (이것만 다르다)
    for kv in "$@"; do export "$kv"; done
    echo "[$ARM $(date +%H:%M:%S)] start  $* "
    t2_launch "$TAG" 8140 "$TASKS" 1 2>&1 | tee "$LOG/$TAG.log"
    echo "[$ARM $(date +%H:%M:%S)] done"
  ) > "$LOG/${TAG}_driver.log" 2>&1
}

run_arm A_ctl   T2_PROCEDURE_LEFT=0 T2_EPLAN_ENUM_SUBTRACT=0 T2_SCOPE_ALL=0 &
P1=$!
run_arm B_say   T2_PROCEDURE_LEFT=1 T2_EPLAN_ENUM_SUBTRACT=1 T2_SCOPE_ALL=0 &
P2=$!
run_arm C_scope T2_PROCEDURE_LEFT=0 T2_EPLAN_ENUM_SUBTRACT=0 T2_SCOPE_ALL=1 &
P3=$!
echo "[t7362] launched A=$P1 B=$P2 C=$P3"
wait $P1 $P2 $P3
echo "[t7362] all arms finished at $(date +%H:%M:%S)"

python "$REPO/reports/facet_rft_2026/freeze.py" --off

# ── 회수 ([[30]]: gzip 만으론 영속이 아니다 — tracked 확인까지가 절차) ──
cd "$REPO" && mkdir -p reports/facet_rft_2026/sim_results
for ARM in A_ctl B_say C_scope; do
  TAG="bank_t7362_${ARM}_${STAMP}"
  gzip -c "$SIMS/$TAG/results.json" > reports/facet_rft_2026/sim_results/$TAG.results.json.gz 2>/dev/null || true
  gzip -c "$LOG/$TAG.log" > reports/facet_rft_2026/sim_results/$TAG.log.gz 2>/dev/null || true
  for _S in fb trace; do _F=$LOG/${_S}_${TAG}.jsonl; [ -s "$_F" ] && gzip -c "$_F" > reports/facet_rft_2026/sim_results/${_S}_${TAG}.jsonl.gz; done
done
git add -f reports/facet_rft_2026/sim_results/bank_t7362_* 2>/dev/null || true
git -c user.name=ghlee -c user.email=beingrelative@gmail.com commit -q -m "t7362 A/B: procedure-left, eplan-subtract, scope-revert" -- reports/facet_rft_2026/sim_results/ || true
git push -q origin facet-rft-2026 || echo "[t7362] push held"
echo "=== TRACKED ==="
for ARM in A_ctl B_say C_scope; do
  git ls-files --error-unmatch reports/facet_rft_2026/sim_results/bank_t7362_${ARM}_${STAMP}.results.json.gz 2>&1 | tail -1
done

echo "=== 팔별 성적 ==="
python - <<'PY'
import json, io, os
SIMS = "/home/woori/scratch/tau2-bench/data/simulations"
for arm in ("A_ctl", "B_say", "C_scope"):
    p = os.path.join(SIMS, "bank_t7362_%s_20260826" % arm, "results.json")
    try:
        d = json.load(io.open(p, encoding="utf-8"))
    except Exception as e:
        print("  %-8s 못 읽음: %r" % (arm, e)); continue
    sims = d.get("simulations") or d.get("results") or []
    row = {str(s.get("task_id")): (s.get("reward_info") or {}).get("reward") for s in sims}
    tot = sum(1 for v in row.values() if v == 1.0)
    print("  %-8s %d/%d  %s" % (arm, tot, len(sims), row))
PY

echo "=== 팔별 마커 ==="
for ARM in A_ctl B_say C_scope; do
  TAG="bank_t7362_${ARM}_${STAMP}"
  printf "%-8s " $ARM
  for m in T2_PROCEDURE_LEFT T2_OPERATOR_DIRECT T2_EPLAN Traceback; do
    printf "%s=%s " $m $(grep -ac "\[$m" "$LOG/$TAG.log" 2>/dev/null || echo 0)
  done
  printf "scope침묵=%s " $(grep -ac "operator-scope 침묵" "$LOG/$TAG.log" 2>/dev/null || echo 0)
  printf "L1released=%s\n" $(grep -ac "L1 released" "$LOG/$TAG.log" 2>/dev/null || echo 0)
done
echo "[t7362] AB DONE"
