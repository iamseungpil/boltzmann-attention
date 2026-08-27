#!/usr/bin/env bash
# t7370 — 비교기 **반경 전체** × nt2 · `T2_SG_ROW_COUNT` 첫 라이브 측정 (사용자 지시 2026-08-28)
#
# ── 무엇을 재나 ──────────────────────────────────────
#   전사 서브가 **선언된 종류의 레코드보다 적게** 넘겼을 때 총액을 단언하지 않고
#   `return_template_short`(총액 문장 없음 + 재공급 지시)로 나간다.
#   실측 근거(t7368 `task_072#s626729`):
#       msg[25] Bluest      레코드 32 · `type: atm_withdrawal`  9 → 서브 9 → delta_total 14.0 = gold
#       msg[35] Light Green 레코드 26 · `type: atm_withdrawal` **10** → 서브 **9** → **5.0 ≠ gold 3.5**
#   빠진 행 = `btxn_8c58b19a3628 (charged $0.00, documented fee $1.50, difference $-1.50)`.
#   그런데 반환문은 `[coverage] 9 of 9 rows were checked (0 could not be verified)` 였다 —
#   분모가 *넘어온 행 수*라 자기 자신을 잰다 ⇒ 우리가 **틀린 총액을 권위 문면으로** 건넸다([[25]]).
#
# ── ★왜 표적이 아니라 반경 위에서 재나 (사용자 지시) ─
#   *"태스크 하나 수리하면 이전에 돌던 태스크가 안 돌게 되는 문제"* 가 바로 이 도구에서 났다:
#   `T2_SG_PROMPT_V2` 가 같은 결손을 **프롬프트 모양**으로 고쳐 074 chk_2 를 13~15/16 → 16/16 으로
#   사고 072 Light Green 을 10/10 → 9/10 으로 팔았다(t7348 ↔ t7363·t7368). 그 부호표는 **074 로만**
#   재졌다. ⇒ 이 런은 이 도구를 부르는 **세 태스크 전부**를 싣는다.
#   반경 실측(t7363·t7368 로그 `operand-size get_atm_fee_discrepancies`): **072 · 074 · 085**.
#
# ── 대조·판정 ────────────────────────────────────────
#   대조 = **t7368**(072 0/2 · 074 0/2 · 085 0/2). 노브는 **`T2_SG_ROW_COUNT=1` 하나만** 다르다.
#   ⛔`T2_ACTIONREQ_GROUNDED` 는 싣지 않는다 — 그것은 t7369 가 072 에서 따로 재는 중이고,
#     둘을 한 런에 섞으면 어느 쪽이 무엇을 했는지 못 가른다([[70]] 귀속). 둘 다 양성이면
#     t7371 이 합성한다([[19]]).
#   판정 = **태스크별 0→1 과 1→0 을 함께** 본다(총점 Δ 금지·C594).
#   1차 = 술어가 옳게 서는가(072 Light Green 에서 서고 Bluest 에서 안 선다).
#   2차 = 짧은 반환문을 받은 모델이 **다시 읽어 10 행을 넘기는가**(그러면 총액이 3.5 가 된다).
#   ⚠pass 는 3차다. 이 레버가 직접 사는 것은 *틀린 총액을 안 내보내는 것*이다.
#
# ── 발사 게이트 ──────────────────────────────────────
#   배터리 exit 0(새 래칫 `test_sg_row_count` 포함) · 스모크에서 **예외 0**.
#   ⚠발화 자체는 게이트가 아니다 — 서브가 전부 옳게 넘기면 0 이 정상이다. 성공 경로만 세면
#     *"안 났다"* 와 *"나다 죽었다"* 가 안 갈린다(핸드오프 §5) ⇒ 줄 수·예외 수를 함께 센다.
# ⛔`set -u` 금지 · 스모크 디렉터리만 치운다 · 줄 이음 금지
set -o pipefail
REPO=/home/woori/workspace_common/boltzmann-attention-pi
LOG=/home/woori/scratch/logs
SIMS=/home/woori/scratch/tau2-bench/data/simulations
STAMP=20260828
TASKS="task_072,task_074,task_085"
SMOKE_TASKS="task_072"
cd "$REPO/scripts/distill/tau2"

echo "[t7370 $(date +%H:%M:%S)] === 발사 전 배터리 ==="
BAD=0
for t in test_sg_row_count test_atm_ledger_close test_rebate_netting test_delta_total_used test_procedure_left test_actionreq_grounded test_operator_find_executed test_decision_point_load test_diag_unambiguous test_read_per_entity test_flag_registry test_arg_label test_a2_three_layer test_card_docs; do
  PYTHONPATH=. PYTHONIOENCODING=utf-8 /home/woori/venvs/seka_env/bin/python $t.py >/dev/null 2>&1
  rc=$?; echo "  $t exit=$rc"; [ $rc -ne 0 ] && BAD=1
done
if [ $BAD -ne 0 ]; then echo "[t7370] ⛔배터리 붉음 — 발사하지 않는다"; exit 1; fi

env_arm() {
  source ./go_stack.sh >/dev/null 2>&1
  # t7368 과 **같은 레버 집합**.
  export T2_ACTION_SUB=1 T2_KEEP_DENY_BODY=1 T2_CALL_FORM=1 T2_ARG_EMPTY=1 T2_SEARCH_AGENT=1
  export T2_DECIDE_ANY=1 T2_WRITE_ARG_ENUM=1 T2_ARG_DOC_SUB=1 T2_VALUE_FORMULA=full
  export T2_SG_DOCS=1 T2_SG_PROMPT_V2=1 T2_SPEC_AT_WRITE=1 T2_WRITE_ARG_TYPE=1
  export T2_RULE_AT_WRITE=1 T2_DUP_WRITE=1
  export T2_READ_PER_ENTITY=0
  export T2_ARG_LABEL=0 T2_DIAG_UNAMBIGUOUS=0
  export T2_CARD_DOCS=1
  export T2_REARM_USER_ONLY=0 T2_PROCEDURE_LEFT=0 T2_EPLAN_ENUM_SUBTRACT=0 T2_SCOPE_ALL=0
  # ⛔t7369 가 재는 중이라 여기서는 **끈다**(한 런에 한 노브).
  export T2_ACTIONREQ_GROUNDED=0
  # ★이 런이 재는 **유일한 차이**.
  export T2_SG_ROW_COUNT=1
  export GO_MAX_STEPS=150
}

# ── ① 스모크 — 자리가 살아 있고 예외가 없는지만 본다 ──────────────────────
STAG="bank_t7370_smoke_${STAMP}"
rm -rf "$SIMS/$STAG"
(
  env_arm; export GO_CONCURRENCY=1
  echo "[smoke $(date +%H:%M:%S)] $SMOKE_TASKS nt=1"
  t2_launch "$STAG" 8140 "$SMOKE_TASKS" 1 2>&1 | tee "$LOG/$STAG.log"
) > "$LOG/${STAG}_driver.log" 2>&1

SZ=$(grep -ac "operand-size get_atm_fee_discrepancies" "$LOG/$STAG.log" 2>/dev/null); SZ=${SZ:-0}
KIND=$(grep -ac "atm_withdrawal=" "$LOG/$STAG.log" 2>/dev/null); KIND=${KIND:-0}
FIRE=$(grep -ac "\[T2_SG_ROW_COUNT\]" "$LOG/$STAG.log" 2>/dev/null); FIRE=${FIRE:-0}
NODECL=$(grep -ac "대체 템플릿 미선언" "$LOG/$STAG.log" 2>/dev/null); NODECL=${NODECL:-0}
SHORT=$(grep -ac "This audit is INCOMPLETE" "$LOG/$STAG.log" 2>/dev/null); SHORT=${SHORT:-0}
TB=$(grep -ac "Traceback" "$LOG/$STAG.log" 2>/dev/null); TB=${TB:-0}
echo "[t7370] 스모크: operand-size=$SZ · 종류계수 인쇄=$KIND · 발화=$FIRE · 미선언=$NODECL · short 문면=$SHORT · Traceback=$TB"
grep -ao "operand-size get_atm_fee_discrepancies[^|]\{0,110\}" "$LOG/$STAG.log" 2>/dev/null | head -4
if [ "$TB" -ne 0 ]; then echo "[t7370] ⛔Traceback $TB — 중단."; exit 1; fi
if [ "$SZ" -eq 0 ]; then echo "[t7370] ⛔비교기에 한 번도 안 닿았다 — 중단."; exit 1; fi
if [ "$KIND" -eq 0 ]; then echo "[t7370] ⛔종류 계수가 한 번도 안 찍혔다 = row_kind 미도달. 중단."; exit 1; fi
if [ "$NODECL" -ne 0 ]; then echo "[t7370] ⛔술어는 섰는데 대체 템플릿이 미선언이다 — 중단."; exit 1; fi

# ── ② 본런 ────────────────────────────────────────────────────────────────
cd "$REPO"
/home/woori/venvs/seka_env/bin/python reports/facet_rft_2026/freeze.py --on \
  --tag "bank_t7370_$STAMP" --reason "t7370 comparator radius: count the rows the sub owed" || true
cd "$REPO/scripts/distill/tau2"

TAG="bank_t7370_radius_${STAMP}"
(
  env_arm; export GO_CONCURRENCY=2
  echo "[main $(date +%H:%M:%S)] $TASKS nt=2"
  t2_launch "$TAG" 8140 "$TASKS" 2 2>&1 | tee "$LOG/$TAG.log"
  echo "[main $(date +%H:%M:%S)] done"
) > "$LOG/${TAG}_driver.log" 2>&1

cd "$REPO"
/home/woori/venvs/seka_env/bin/python reports/facet_rft_2026/freeze.py --off

# ── ③ 회수 ([[30]]: gzip 만으론 영속이 아니다) ────────────────────────────
mkdir -p reports/facet_rft_2026/sim_results
for T in "$STAG" "$TAG"; do
  gzip -c "$SIMS/$T/results.json" > reports/facet_rft_2026/sim_results/$T.results.json.gz 2>/dev/null || true
  gzip -c "$LOG/$T.log" > reports/facet_rft_2026/sim_results/$T.log.gz 2>/dev/null || true
  for S in fb trace; do
    F="$LOG/${S}_${T}.jsonl"; [ -s "$F" ] && gzip -c "$F" > reports/facet_rft_2026/sim_results/${S}_${T}.jsonl.gz
  done
done
git add -f reports/facet_rft_2026/sim_results/*bank_t7370_* 2>/dev/null || true
git -c user.name=ghlee -c user.email=beingrelative@gmail.com \
  commit -q -m "t7370 comparator radius: the row-count check, measured against t7368" \
  -- reports/facet_rft_2026/sim_results || true
if ! git push -q origin facet-rft-2026; then
  echo "[t7370] push 실패 — rebase 후 재시도"
  git -c user.name=ghlee -c user.email=beingrelative@gmail.com pull --rebase -q origin facet-rft-2026 || true
  if ! git push -q origin facet-rft-2026; then
    echo "[t7370] ⛔재시도도 실패 — 결과가 이 디스크에만 있다. 손으로 회수하라."
    PUSH_FAIL=1
  fi
fi
echo "=== TRACKED ==="
for F in reports/facet_rft_2026/sim_results/*bank_t7370_*; do
  git ls-files --error-unmatch "$F" >/dev/null 2>&1 && echo "  ok   $(basename $F)" || echo "  ⛔UNTRACKED $(basename $F)"
done

echo "=== 성적 · 부호표 ==="
TAG="$TAG" /home/woori/venvs/seka_env/bin/python - <<'PY'
# -*- coding: utf-8 -*-
import os, sys, re, collections
sys.path.insert(0, "scripts/distill/tau2")
sys.stdout.reconfigure(encoding="utf-8")
import t2_forensic as F
tag = os.environ["TAG"]
try:
    sims = F.sims(tag)
except Exception as e:
    print("결과 로드 실패: %r" % (e,)); raise SystemExit(0)
mut = F.mutating_tools()
per = collections.defaultdict(list)
for s in sims:
    per[F.task_id(s)].append((s.get("reward_info") or {}).get("reward"))
print("  pass %d/%d" % (sum(1 for v in per.values() for r in v if r == 1.0), len(sims)))
print("  ★부호표 (대조 t7368: 072 [0,0] · 074 [0,0] · 085 [0,0])")
for t in sorted(per):
    print("    %-10s %s" % (t, per[t]))
for s in sims:
    d = F.mutation_diff(s, mut, tag=tag) or {}
    gap = sum(len(d.get(k) or ()) for k in ("missing", "wrongarg", "extra", "dup"))
    print("    %-20s reward=%-5s gap=%-3d %s"
          % (F.simtag(s), (s.get("reward_info") or {}).get("reward"), gap, F.term_reason(s)))
print("  종료사유: %s" % dict(collections.Counter(F.term_reason(s) for s in sims)))
PY

echo "=== 레버 발화 · 술어가 옳게 섰나 ==="
for M in "\[T2_SG_ROW_COUNT\]" "This audit is INCOMPLETE" "대체 템플릿 미선언" "⚠SHORT" "Traceback"; do
  C=$(grep -ac "$M" "$LOG/$TAG.log" 2>/dev/null); echo "  $M = ${C:-0}"
done
echo "--- operand-size 전량 (sub ↔ 종류 계수) ---"
grep -ao "sim=task_[0-9]*\]\|operand-size get_atm_fee_discrepancies[^|]\{0,110\}" "$LOG/$TAG.log" 2>/dev/null | paste - - 2>/dev/null | head -20
echo "--- 도구가 건넨 총액 ---"
grep -ao "computed by this tool, is [-0-9.]*" "$LOG/$TAG.log" 2>/dev/null | sort | uniq -c
echo "[t7370] DONE $(date +%H:%M:%S)"
[ -n "$PUSH_FAIL" ] && exit 1
exit 0
