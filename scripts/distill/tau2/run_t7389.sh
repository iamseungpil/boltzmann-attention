#!/usr/bin/env bash
# t7389 — 094 단독 × nt2 × 두 팔 · `T2_SG_REQREADS_CANON` 라이브 첫 측정 (2026-08-29)
#
# ── 무엇을 재나 ──────────────────────────────────────
#   094 의 두 오퍼랜드가 **각 sim 에서 하나씩만** 맞았다(t7385~t7388 · x592):
#       expected_apy  6.85 정답 1회 ↔ 6.275/6.1 오답 4회   (우리 반환문의 완결 단언)
#       actual_apy    5.1  정답 1회 ↔ 6.0/5.0/5.99 오답    (우리 접지의 파생 드롭)
#   둘 다 우리 층이고 둘 다 이 sha 에서 수리됐다(`207a6df9` · `f59bc3af`).
#   그런데 파생 접지는 **거래 read 가 있어야만** 살아난다 — 함의 리터럴(월 이자 크레딧)이
#   도구 출력에 있어야 대조가 되기 때문이다. 그 read 를 요구하는 선언은 A3 정본 인덱스에
#   이미 있고(`get_interest_correction -> [계좌목록, 거래이력]`), 그것을 읽는 스위치가
#   `T2_SG_REQREADS_CANON` 인데 **기본 OFF** 라 한 번도 안 걸렸다.
#   ⇒ 이 런의 유일한 차이는 그 스위치 하나다([[70]] 귀속).
#
# ── 반경은 미리 쟀다 ─────────────────────────────────
#   스캐폴드 10 중 자신의 키와 정본 입구가 다른 것은 `get_interest_correction` **하나**다
#   (나머지 9 는 이미 같다 = 거동 불변). 파는 것 = 계좌·거래 read 를 아직 안 한 궤적에서
#   계산이 한 턴 밀린다.
#
# ── 판정 ─────────────────────────────────────────────
#   1차 = **채널이 서는가**(pass 는 2차다):
#       ⓐ 거래 read 호출 0 → n
#       ⓑ `파생-검산 통과` 0 → n            (옳은 파생이 살아남기 시작하는가)
#       ⓒ `get_interest_correction -> None` 이 줄어드는가
#       ⓓ `[components]` 가 뜬 뒤 재호출이 오는가 (완결 문면의 효과)
#   ⚠pass 가 안 나와도 실패가 아니다 — 두 오퍼랜드가 **같은 sim 에서 동시에** 맞아야
#     140 이 나오고, 이 런은 그 둘이 각각 살아나는지를 먼저 본다.
#   ⛔묶음 Δ 로 개별 수리를 주장하지 마라(C594). 대조는 같은 sha 의 control 팔이다.
#
# ⛔`set -u` 금지 · 스모크 디렉터리만 치운다 · 줄 이음 금지
set -o pipefail
REPO=/home/woori/workspace_common/boltzmann-attention-pi
LOG=/home/woori/scratch/logs
SIMS=/home/woori/scratch/tau2-bench/data/simulations
STAMP=20260829
TASKS="task_094"
SMOKE_TASKS="task_094"
cd "$REPO/scripts/distill/tau2"

echo "[t7389 $(date +%H:%M:%S)] sha=$(cd $REPO && git rev-parse --short HEAD)"

echo "[t7389] === 발사 전 배터리 ==="
BAD=0
for t in test_sg_dedup test_write_once_key test_write_amount_evidence test_sg_close_self \
         test_sg_row_count test_freeze_multihold test_atm_fee_op test_atm_ledger_close \
         test_rebate_netting test_delta_total_used test_actionreq_grounded test_procedure_left \
         test_operator_find_executed test_decision_point_load test_diag_unambiguous \
         test_read_per_entity test_flag_registry test_arg_label test_a2_three_layer \
         test_card_docs test_apy_balance_tier test_banking_gate \
         test_derived_grounding test_groups_used_note; do
  PYTHONPATH=. PYTHONIOENCODING=utf-8 /home/woori/venvs/seka_env/bin/python $t.py >/dev/null 2>&1
  rc=$?; echo "  $t exit=$rc"; [ $rc -ne 0 ] && BAD=1
done
if [ $BAD -ne 0 ]; then echo "[t7389] ⛔배터리 붉음 — 발사하지 않는다"; exit 1; fi

# ── 두 팔은 이 한 줄에서만 다르다 ──────────────────────────────────────────
env_arm() {
  source ./go_stack.sh >/dev/null 2>&1
  export T2_ACTION_SUB=1 T2_KEEP_DENY_BODY=1 T2_CALL_FORM=1 T2_ARG_EMPTY=1 T2_SEARCH_AGENT=1
  export T2_SG_DOCS=1 T2_SG_PROMPT_V2=1 T2_SPEC_AT_WRITE=1 T2_WRITE_ARG_TYPE=1
  export T2_RULE_AT_WRITE=1 T2_DUP_WRITE=1
  export T2_READ_PER_ENTITY=0
  export T2_ARG_LABEL=0 T2_DIAG_UNAMBIGUOUS=0
  export T2_CARD_DOCS=1
  export T2_REARM_USER_ONLY=0 T2_PROCEDURE_LEFT=0 T2_EPLAN_ENUM_SUBTRACT=0 T2_SCOPE_ALL=0
  export T2_ACTIONREQ_GROUNDED=1 T2_SG_ROW_COUNT=1 T2_SG_CLOSE_SELF=1
  export T2_PROMPT_DUMP=1 T2_PROMPT_DUMP_MAX=80000
  export GO_MAX_STEPS=200
  # ★이 런이 재는 **유일한 차이**.
  export T2_SG_REQREADS_CANON="$ARM_CANON"
}

# ── ① 스모크 — 자리가 살아 있는지만 본다(성적 아님) ───────────────────────
STAG="bank_t7389_smoke_${STAMP}"
rm -rf "$SIMS/$STAG"
(
  ARM_CANON=1; env_arm; export GO_CONCURRENCY=1
  echo "[smoke $(date +%H:%M:%S)] $SMOKE_TASKS nt=1 · CANON=1"
  t2_launch "$STAG" 8140 "$SMOKE_TASKS" 1 2>&1 | tee "$LOG/$STAG.log"
) > "$LOG/${STAG}_driver.log" 2>&1

CAN=$(grep -ac "T2_SG_REQREADS\] 정본 입구 채택" "$LOG/$STAG.log" 2>/dev/null); CAN=${CAN:-0}
SKP=$(grep -ac "정본 입구 건너뜀" "$LOG/$STAG.log" 2>/dev/null); SKP=${SKP:-0}
DRV=$(grep -ac "파생-검산" "$LOG/$STAG.log" 2>/dev/null); DRV=${DRV:-0}
CMP=$(grep -ac "\[components\]" "$LOG/$STAG.log" 2>/dev/null); CMP=${CMP:-0}
EXC=$(grep -ac "파생-검산 예외" "$LOG/$STAG.log" 2>/dev/null); EXC=${EXC:-0}
TB=$(grep -ac "Traceback" "$LOG/$STAG.log" 2>/dev/null); TB=${TB:-0}
echo "[t7389] 스모크: 정본입구채택=$CAN · 건너뜀=$SKP · 파생-검산=$DRV · components=$CMP · 예외=$EXC · Traceback=$TB"
if [ "$EXC" -ne 0 ]; then echo "[t7389] ⛔파생 술어가 예외로 죽었다 — 중단."; exit 1; fi
if [ "$SKP" -ne 0 ]; then echo "[t7389] ⛔정본 입구를 못 읽었다 — 중단."; exit 1; fi
if [ "$CAN" -eq 0 ]; then echo "[t7389] ⛔스위치가 한 번도 안 걸렸다 — 중단."; exit 1; fi
if [ "$TB" -ne 0 ]; then echo "[t7389] ⛔Traceback $TB — 중단."; exit 1; fi

# ── ② 본런 — control(0) 먼저, treat(1) 나중 ───────────────────────────────
cd "$REPO"
/home/woori/venvs/seka_env/bin/python reports/facet_rft_2026/freeze.py --on \
  --tag "bank_t7389_$STAMP" --reason "t7389 094: canonical requires_reads, same-sha A/B" || true
cd "$REPO/scripts/distill/tau2"

for A in 0 1; do
  NAME=control; [ "$A" = "1" ] && NAME=treat
  TAG="bank_t7389_${NAME}_${STAMP}"
  (
    ARM_CANON=$A; env_arm; export GO_CONCURRENCY=1
    echo "[$NAME $(date +%H:%M:%S)] $TASKS nt=2 · CANON=$A"
    t2_launch "$TAG" 8140 "$TASKS" 2 2>&1 | tee "$LOG/$TAG.log"
    echo "[$NAME $(date +%H:%M:%S)] done"
  ) > "$LOG/${TAG}_driver.log" 2>&1
done

cd "$REPO"
/home/woori/venvs/seka_env/bin/python reports/facet_rft_2026/freeze.py --off

# ── ③ 회수 ([[30]]: gzip 만으론 영속이 아니다) ────────────────────────────
mkdir -p reports/facet_rft_2026/sim_results
for T in "$STAG" "bank_t7389_control_${STAMP}" "bank_t7389_treat_${STAMP}"; do
  gzip -c "$SIMS/$T/results.json" > reports/facet_rft_2026/sim_results/$T.results.json.gz 2>/dev/null || true
  gzip -c "$LOG/$T.log" > reports/facet_rft_2026/sim_results/$T.log.gz 2>/dev/null || true
  for S in fb trace; do
    F="$LOG/${S}_${T}.jsonl"; [ -s "$F" ] && gzip -c "$F" > reports/facet_rft_2026/sim_results/${S}_${T}.jsonl.gz
  done
done
git add -f reports/facet_rft_2026/sim_results/*bank_t7389_* 2>/dev/null || true
git -c user.name=ghlee -c user.email=beingrelative@gmail.com \
  commit -q -m "bank_t7389 094: canonical requires_reads, same-sha A/B" \
  -- reports/facet_rft_2026/sim_results || true
if ! git push -q origin facet-rft-2026; then
  echo "[t7389] push 실패 — rebase 후 재시도"
  git -c user.name=ghlee -c user.email=beingrelative@gmail.com pull --rebase -q origin facet-rft-2026 || true
  if ! git push -q origin facet-rft-2026; then
    echo "[t7389] ⛔재시도도 실패 — 결과가 이 디스크에만 있다. 손으로 회수하라."
  fi
fi
echo "=== TRACKED ==="
for F in reports/facet_rft_2026/sim_results/*bank_t7389_*; do
  git ls-files --error-unmatch "$F" >/dev/null 2>&1 && echo "  ok   $(basename $F)" || echo "  ⛔UNTRACKED $(basename $F)"
done

# ── ④ 판정표 — 채널이 섰나(1차) · pass(2차) ───────────────────────────────
echo "=== 채널 (1차 판정) ==="
for A in control treat; do
  T="$LOG/bank_t7389_${A}_${STAMP}.log"
  printf '  %-8s 거래read=%-4s 파생검산통과=%-4s 검산불성립=%-4s ->None=%-4s components=%-4s\n' \
    "$A" \
    "$(grep -ao 'get_bank_account_transactions' $T 2>/dev/null | wc -l)" \
    "$(grep -ac '파생-검산 통과' $T 2>/dev/null)" \
    "$(grep -ac '파생-검산 불성립' $T 2>/dev/null)" \
    "$(grep -ac 'get_interest_correction -> None' $T 2>/dev/null)" \
    "$(grep -ac '\[components\]' $T 2>/dev/null)"
done

echo "=== 성적 (2차) ==="
cd "$REPO"
STAMP="$STAMP" /home/woori/venvs/seka_env/bin/python - <<'PY'
# -*- coding: utf-8 -*-
import os, sys
sys.path.insert(0, "scripts/distill/tau2")
sys.stdout.reconfigure(encoding="utf-8")
import t2_forensic as F
stamp = os.environ["STAMP"]
mut = F.mutating_tools()
for arm in ("control", "treat"):
    tag = "bank_t7389_%s_%s" % (arm, stamp)
    try:
        sims = F.sims(tag)
    except Exception as e:
        print("  %-8s 결과 로드 실패: %r" % (arm, e)); continue
    print("  [%s] pass %d/%d" % (arm, sum(1 for s in sims
          if (s.get("reward_info") or {}).get("reward") == 1.0), len(sims)))
    for s in sims:
        d = F.mutation_diff(s, mut, tag=tag) or {}
        gap = sum(len(d.get(k) or ()) for k in ("missing", "wrongarg", "extra", "dup"))
        print("    %-20s reward=%-5s gap=%-3d %s"
              % (F.simtag(s), (s.get("reward_info") or {}).get("reward"), gap, F.term_reason(s)))
        for k in ("missing", "wrongarg"):
            for x in (d.get(k) or ()):
                print("        %-9s %s" % (k, str(x.get("key"))[:150]))
PY
echo "[t7389 $(date +%H:%M:%S)] 끝"
