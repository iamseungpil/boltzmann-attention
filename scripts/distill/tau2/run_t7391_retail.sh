#!/usr/bin/env bash
# t7391 — **ABox 스왑 1a단계 · retail 전수 nt1 · treat-only** (사용자 지시 2026-08-29)
#
# t7390(airline)의 쌍이다. 같은 sha·같은 엔진·같은 레버 집합에서 **도메인만** 바꾼다.
# GPU0 = airline(8140) · GPU1 = retail(8141) 로 **병렬**(사용자 지시 *"gpu 0,1 두개다 사용하라"*).
#
# ── 1차 종점 (pass 아니다) ───────────────────────────
#   · Traceback / 예외 / 크래시 sim 수 — **엔진을 한 줄도 안 고치고 도는가**
#   · 우리 레버가 발화하는가(retail A2 는 gates 8 · `scaffold_get_tools` 0)
#   ⛔고치고 싶어지면 고치지 말고 적어라. 한 줄이라도 고치면 특허 §3.4(c) 에 대한 반증이다.
#
# ── 이 런이 답하지 못하는 것 ─────────────────────────
#   · retail 채점축은 **112/114 가 `('DB','NL_ASSERTION')`** — LLM 판정 축이 섞여 있다.
#     banking(순수 DB)·airline(DB+COMMUNICATE)과 **pass 를 나란히 놓지 마라**. 셋 다 축이 다르다.
#   · retail A2 는 **개발된 적이 없다**. 이 런은 *"저작 증분 0 에서 무엇이 되나"* 다.
#
# ── ⚠프리즈를 부르지 않는다 ──────────────────────────
#   t7390 이 같은 sha(`cccbf7dd`)로 이미 홀드 중이고, 다중 홀드 해제는 **미수리 버그**다
#   (핸드오프 2026-08-29 §5-6: 다른 런 종료 후 `freeze --status` 가 `not frozen` 이 됐다).
#   여기서 `--off` 를 부르면 airline 의 홀드를 푼다. ⇒ 두 런이 도는 동안 **엔진 커밋 금지**가
#   규율이고, 프리즈는 t7390 것 하나만 쓴다.
#
# ⛔`set -u` 금지 · 줄 이음 금지
set -o pipefail
REPO=/home/woori/workspace_common/boltzmann-attention-pi
LOG=/home/woori/scratch/logs
SIMS=/home/woori/scratch/tau2-bench/data/simulations
STAMP=20260829
cd "$REPO/scripts/distill/tau2"

echo "[t7391 $(date +%H:%M:%S)] sha=$(cd $REPO && git rev-parse --short HEAD)"

echo "[t7391] === 배터리 ==="
BAD=0
for t in test_sg_dedup test_write_once_key test_write_amount_evidence test_sg_close_self \
         test_sg_row_count test_freeze_multihold test_atm_fee_op test_atm_ledger_close \
         test_rebate_netting test_delta_total_used test_actionreq_grounded test_procedure_left \
         test_operator_find_executed test_decision_point_load test_diag_unambiguous \
         test_read_per_entity test_flag_registry test_arg_label test_a2_three_layer \
         test_card_docs test_apy_balance_tier test_banking_gate \
         test_derived_grounding test_groups_used_note test_over_rows_membership; do
  PYTHONPATH=. PYTHONIOENCODING=utf-8 /home/woori/venvs/seka_env/bin/python $t.py >/dev/null 2>&1
  rc=$?; [ $rc -ne 0 ] && { echo "  $t exit=$rc"; BAD=1; }
done
if [ $BAD -ne 0 ]; then echo "[t7391] ⛔배터리 붉음 — 발사하지 않는다"; exit 1; fi
echo "  25/25 초록"

env_retail() {
  source ./go_stack.sh >/dev/null 2>&1
  export T2_ACTION_SUB=1 T2_KEEP_DENY_BODY=1 T2_CALL_FORM=1 T2_ARG_EMPTY=1 T2_SEARCH_AGENT=1
  export T2_SG_DOCS=1 T2_SG_PROMPT_V2=1 T2_SPEC_AT_WRITE=1 T2_WRITE_ARG_TYPE=1
  export T2_RULE_AT_WRITE=1 T2_DUP_WRITE=1
  export T2_ACTIONREQ_GROUNDED=1 T2_SG_ROW_COUNT=1 T2_SG_CLOSE_SELF=1
  export T2_SG_REQREADS=1 T2_SG_REQREADS_CANON=1
  export T2_PROMPT_DUMP=1 T2_PROMPT_DUMP_MAX=80000
  export GO_MAX_STEPS=200 GO_CONCURRENCY=1
  # ★이 런의 전부 — 도메인만 바꾼다.
  export GO_DOMAIN=retail
  export GO_RETRIEVAL=
}

# ── ① 스모크 — 3 태스크 ──────────────────────────────────────────────────
STAG="bank_t7391_retail_smoke_${STAMP}"
rm -rf "$SIMS/$STAG"
(
  env_retail
  echo "[smoke $(date +%H:%M:%S)] retail task 0,1,2 nt=1 · port 8141"
  t2_launch "$STAG" 8141 "0,1,2" 1 2>&1 | tee "$LOG/$STAG.log"
) > "$LOG/${STAG}_driver.log" 2>&1

TB=$(grep -ac "Traceback" "$LOG/$STAG.log" 2>/dev/null); TB=${TB:-0}
DONE=$(grep -ac "Status:" "$LOG/$STAG.log" 2>/dev/null); DONE=${DONE:-0}
echo "[t7391] 스모크: Traceback=$TB · 완료표시=$DONE"
grep -aoE "\[T2_[A-Z_]+\]" "$LOG/$STAG.log" 2>/dev/null | sort | uniq -c | sort -rn | head -20
if [ "$TB" -ne 0 ]; then
  echo "[t7391] ⛔retail 에서 Traceback $TB — **고치지 말고 기록한다.** 본런 중단."
  grep -a -A12 "Traceback" "$LOG/$STAG.log" | head -40
  exit 1
fi
if [ "$DONE" -eq 0 ]; then echo "[t7391] ⛔sim 이 한 건도 안 끝났다 — 중단."; exit 1; fi

# ── ② 본런 — retail 전수 114 태스크 nt=1 ─────────────────────────────────
TAG="bank_t7391_retail_${STAMP}"
(
  env_retail
  echo "[main $(date +%H:%M:%S)] retail 전수 nt=1 · port 8141"
  t2_launch "$TAG" 8141 "" 1 2>&1 | tee "$LOG/$TAG.log"
  echo "[main $(date +%H:%M:%S)] done"
) > "$LOG/${TAG}_driver.log" 2>&1

# ── ③ 회수 ([[30]]) — rebase 전에 프리즈 파일을 되돌린다 ────────────────
#   t7389 에서 push 가 실패한 원인이 `FREEZE.json` 수정으로 rebase 가 막힌 것이었다.
cd "$REPO"
mkdir -p reports/facet_rft_2026/sim_results
for T in "$STAG" "$TAG"; do
  gzip -c "$SIMS/$T/results.json" > reports/facet_rft_2026/sim_results/$T.results.json.gz 2>/dev/null || true
  gzip -c "$LOG/$T.log" > reports/facet_rft_2026/sim_results/$T.log.gz 2>/dev/null || true
  for S in fb trace; do
    F="$LOG/${S}_${T}.jsonl"; [ -s "$F" ] && gzip -c "$F" > reports/facet_rft_2026/sim_results/${S}_${T}.jsonl.gz
  done
done
git add -f reports/facet_rft_2026/sim_results/*bank_t7391_* 2>/dev/null || true
git -c user.name=ghlee -c user.email=beingrelative@gmail.com \
  commit -q -m "bank_t7391 retail: ABox swap phase 1a, engine unchanged" \
  -- reports/facet_rft_2026/sim_results || true
if ! git push -q origin facet-rft-2026; then
  echo "[t7391] push 실패 — FREEZE.json 되돌리고 rebase 후 재시도"
  git checkout -- reports/facet_rft_2026/FREEZE.json 2>/dev/null
  git -c user.name=ghlee -c user.email=beingrelative@gmail.com pull --rebase -q origin facet-rft-2026 || true
  git push -q origin facet-rft-2026 || echo "[t7391] ⛔재시도도 실패 — 손으로 회수하라."
fi
echo "=== TRACKED ==="
for F in reports/facet_rft_2026/sim_results/*bank_t7391_*; do
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
TAG="$TAG" /home/woori/venvs/seka_env/bin/python - <<'PY'
import os, sys, collections
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
print("  종료사유:", dict(collections.Counter(F.term_reason(s) for s in sims)))
print("  ⚠retail 채점축은 ('DB','NL_ASSERTION') 다 — banking·airline 수와 나란히 두지 마라.")
PY
echo "[t7391 $(date +%H:%M:%S)] 끝"
