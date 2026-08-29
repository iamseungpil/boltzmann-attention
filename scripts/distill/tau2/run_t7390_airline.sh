#!/usr/bin/env bash
# t7390 — **ABox 스왑 1a단계 · airline 전수 nt1 · treat-only** (사용자 지시 2026-08-29)
#
# ── 무엇을 재나 ──────────────────────────────────────
#   특허 B 정본 §3.4(c) 는 *"도메인 리터럴 0 의 도메인-일반 엔진 ⇒ 신규 배포 시 재작성 불요"* 를
#   주장한다. 이 런은 그 주장의 **유일한 실측 방어선**이다.
#
#   1차 종점 = **엔진을 한 줄도 안 고치고 도는가**
#       · Traceback / 예외 / 크래시 sim 수
#       · 우리 레버가 발화하는가(도메인 선언이 얇아도 gates 3 은 있다)
#       · 도메인-특화 수리가 필요해지는 지점의 **목록**
#   2차 종점 = pass (control 팔은 1b 에서 따로 — 이 런은 treat-only)
#
#   ⛔**엔진을 고치고 싶어지면 고치지 말고 적어라.** 한 줄이라도 고치면 그 사실 자체가
#     §3.4(c) 에 대한 반증이고, 애든덤 §5 주장 금지선이 그것을 그대로 기재하라고 요구한다.
#
# ── 이 런이 답하지 못하는 것 (미리 적는다) ───────────
#   · airline 채점축은 **50/50 이 `('DB','COMMUNICATE')`** 이고 banking 은 순수 `DB` 다.
#     ⇒ **pass 를 banking 과 나란히 놓지 마라.** 축이 다르다.
#   · airline gold 행동은 중앙값 1 (banking 8) · 7 태스크는 **0 행동**이다.
#     ⇒ 상태변경 표면이 훨씬 얇아 우리 스캐폴드가 걸릴 자리가 적다. 효과 작음이 기본 기대다.
#   · airline A2 는 **개발된 적이 없다**(`scaffold_get_tools` 0 · gates 3). 이 런은
#     *"저작 증분 0 에서 무엇이 되나"* 이지 *"airline 을 잘하게 만들었나"* 가 아니다.
#
# ── 배선 ─────────────────────────────────────────────
#   `GO_DOMAIN=airline` · `GO_RETRIEVAL=`(빈 문자열 → `--retrieval_config` 미전달).
#   `t2_launch` 는 그 두 줄만 파라미터화됐고 기본값이 종전과 같다(banking 거동 보존).
#
# ⛔`set -u` 금지 · 줄 이음 금지
set -o pipefail
REPO=/home/woori/workspace_common/boltzmann-attention-pi
LOG=/home/woori/scratch/logs
SIMS=/home/woori/scratch/tau2-bench/data/simulations
STAMP=20260829
cd "$REPO/scripts/distill/tau2"

echo "[t7390 $(date +%H:%M:%S)] sha=$(cd $REPO && git rev-parse --short HEAD)"

# ── 발사 전: banking 거동이 안 바뀌었는지부터 (진입점 패치의 회귀 검사) ────
echo "[t7390] === 배터리 (banking 거동 보존 확인 포함) ==="
BAD=0
for t in test_sg_dedup test_write_once_key test_write_amount_evidence test_sg_close_self \
         test_sg_row_count test_freeze_multihold test_atm_fee_op test_atm_ledger_close \
         test_rebate_netting test_delta_total_used test_actionreq_grounded test_procedure_left \
         test_operator_find_executed test_decision_point_load test_diag_unambiguous \
         test_read_per_entity test_flag_registry test_arg_label test_a2_three_layer \
         test_card_docs test_apy_balance_tier test_banking_gate \
         test_derived_grounding test_groups_used_note test_over_rows_membership; do
  PYTHONPATH=. PYTHONIOENCODING=utf-8 /home/woori/venvs/seka_env/bin/python $t.py >/dev/null 2>&1
  rc=$?; echo "  $t exit=$rc"; [ $rc -ne 0 ] && BAD=1
done
if [ $BAD -ne 0 ]; then echo "[t7390] ⛔배터리 붉음 — 발사하지 않는다"; exit 1; fi

env_airline() {
  source ./go_stack.sh >/dev/null 2>&1
  export T2_ACTION_SUB=1 T2_KEEP_DENY_BODY=1 T2_CALL_FORM=1 T2_ARG_EMPTY=1 T2_SEARCH_AGENT=1
  export T2_SG_DOCS=1 T2_SG_PROMPT_V2=1 T2_SPEC_AT_WRITE=1 T2_WRITE_ARG_TYPE=1
  export T2_RULE_AT_WRITE=1 T2_DUP_WRITE=1
  export T2_ACTIONREQ_GROUNDED=1 T2_SG_ROW_COUNT=1 T2_SG_CLOSE_SELF=1
  export T2_SG_REQREADS=1 T2_SG_REQREADS_CANON=1
  export T2_PROMPT_DUMP=1 T2_PROMPT_DUMP_MAX=80000
  export GO_MAX_STEPS=200 GO_CONCURRENCY=1
  # ★이 런의 전부 — 도메인만 바꾼다.
  export GO_DOMAIN=airline
  export GO_RETRIEVAL=
}

# ── ① 스모크 — 3 태스크. 도는가·크래시 없나만 본다 ────────────────────────
STAG="bank_t7390_airline_smoke_${STAMP}"
rm -rf "$SIMS/$STAG"
(
  env_airline
  echo "[smoke $(date +%H:%M:%S)] airline task 0,1,2 nt=1"
  t2_launch "$STAG" 8140 "0,1,2" 1 2>&1 | tee "$LOG/$STAG.log"
) > "$LOG/${STAG}_driver.log" 2>&1

TB=$(grep -ac "Traceback" "$LOG/$STAG.log" 2>/dev/null); TB=${TB:-0}
DONE=$(grep -ac "Status:" "$LOG/$STAG.log" 2>/dev/null); DONE=${DONE:-0}
LEV=$(grep -aoE "\[T2_[A-Z_]+\]" "$LOG/$STAG.log" 2>/dev/null | sort -u | wc -l)
echo "[t7390] 스모크: Traceback=$TB · 완료표시=$DONE · 서로 다른 레버 마커=$LEV"
grep -aoE "\[T2_[A-Z_]+\]" "$LOG/$STAG.log" 2>/dev/null | sort | uniq -c | sort -rn | head -20
if [ "$TB" -ne 0 ]; then
  echo "[t7390] ⛔airline 에서 Traceback $TB — **고치지 말고 기록한다.** 본런 중단."
  grep -a -A12 "Traceback" "$LOG/$STAG.log" | head -40
  exit 1
fi
if [ "$DONE" -eq 0 ]; then echo "[t7390] ⛔sim 이 한 건도 안 끝났다 — 중단."; exit 1; fi

# ── ② 본런 — airline 전수 50 태스크 nt=1 (task_ids 미지정 = 전수) ─────────
cd "$REPO"
/home/woori/venvs/seka_env/bin/python reports/facet_rft_2026/freeze.py --on \
  --tag "bank_t7390_$STAMP" --reason "t7390 airline ABox swap phase 1a" || true
cd "$REPO/scripts/distill/tau2"

TAG="bank_t7390_airline_${STAMP}"
(
  env_airline
  echo "[main $(date +%H:%M:%S)] airline 전수 nt=1"
  t2_launch "$TAG" 8140 "" 1 2>&1 | tee "$LOG/$TAG.log"
  echo "[main $(date +%H:%M:%S)] done"
) > "$LOG/${TAG}_driver.log" 2>&1

cd "$REPO"
/home/woori/venvs/seka_env/bin/python reports/facet_rft_2026/freeze.py --off

# ── ③ 회수 ───────────────────────────────────────────────────────────────
mkdir -p reports/facet_rft_2026/sim_results
for T in "$STAG" "$TAG"; do
  gzip -c "$SIMS/$T/results.json" > reports/facet_rft_2026/sim_results/$T.results.json.gz 2>/dev/null || true
  gzip -c "$LOG/$T.log" > reports/facet_rft_2026/sim_results/$T.log.gz 2>/dev/null || true
  for S in fb trace; do
    F="$LOG/${S}_${T}.jsonl"; [ -s "$F" ] && gzip -c "$F" > reports/facet_rft_2026/sim_results/${S}_${T}.jsonl.gz
  done
done
git add -f reports/facet_rft_2026/sim_results/*bank_t7390_* 2>/dev/null || true
git -c user.name=ghlee -c user.email=beingrelative@gmail.com \
  commit -q -m "bank_t7390 airline: ABox swap phase 1a, engine unchanged" \
  -- reports/facet_rft_2026/sim_results || true
if ! git push -q origin facet-rft-2026; then
  echo "[t7390] push 실패 — rebase 후 재시도"
  git -c user.name=ghlee -c user.email=beingrelative@gmail.com pull --rebase -q origin facet-rft-2026 || true
  git push -q origin facet-rft-2026 || echo "[t7390] ⛔재시도도 실패 — 손으로 회수하라."
fi
echo "=== TRACKED ==="
for F in reports/facet_rft_2026/sim_results/*bank_t7390_*; do
  git ls-files --error-unmatch "$F" >/dev/null 2>&1 && echo "  ok   $(basename $F)" || echo "  ⛔UNTRACKED $(basename $F)"
done

# ── ④ 1차 종점 표 ────────────────────────────────────────────────────────
echo "=== 1차 종점 — 엔진 수정 없이 도는가 ==="
echo "  Traceback  : $(grep -ac Traceback $LOG/$TAG.log)"
echo "  예외 마커  : $(grep -acE '예외|Exception' $LOG/$TAG.log)"
echo "  완료 sim   : $(grep -ac 'Status:' $LOG/$TAG.log)"
echo "  레버 마커 종류: $(grep -aoE '\[T2_[A-Z_]+\]' $LOG/$TAG.log | sort -u | wc -l)"
grep -aoE "\[T2_[A-Z_]+\]" "$LOG/$TAG.log" | sort | uniq -c | sort -rn | head -25

echo "=== 2차 (참고) — pass ==="
cd "$REPO"
TAG="$TAG" /home/woori/venvs/seka_env/bin/python - <<'PY'
import os, sys
sys.path.insert(0, "scripts/distill/tau2")
sys.stdout.reconfigure(encoding="utf-8")
import t2_forensic as F
tag = os.environ["TAG"]
try:
    sims = F.sims(tag)
except Exception as e:
    print("  결과 로드 실패: %r" % (e,)); raise SystemExit(0)
ok = sum(1 for s in sims if (s.get("reward_info") or {}).get("reward") == 1.0)
print("  pass %d/%d" % (ok, len(sims)))
import collections
c = collections.Counter(F.term_reason(s) for s in sims)
print("  종료사유:", dict(c))
print("  ⚠airline 채점축은 ('DB','COMMUNICATE') 다 — banking 수와 나란히 두지 마라.")
PY
echo "[t7390 $(date +%H:%M:%S)] 끝"
