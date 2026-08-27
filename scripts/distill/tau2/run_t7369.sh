#!/usr/bin/env bash
# t7369 — 072 단독 × nt4 · `T2_ACTIONREQ_GROUNDED` 첫 라이브 측정 (사용자 지시 2026-08-28)
#
# ── 왜 이 레버, 왜 이 태스크 ─────────────────────────
#   `TASK_072.md` §7-2 가 처방 P-A 로 적었고 `3053e5d3` 이 구현했으나 **어느 드라이버에도
#   실린 적이 없다** — `go_stack` 기본 0 이고 t7360~t7368 전부 미탑재. 커밋 메시지의
#   *"default so the run measures it"* 이 아직 안 지켜졌다.
#   결함: `formalize_intent_tool` 이 **이 대화에 한 번도 안 나온** 손님-측 도구를 지목하면
#   `[ACTION]` 이 *"'X' 는 손님이 실행한다"* 고 말한다 — 참이지만 이 대화와 무관하고,
#   072 t0 에서 그 한 줄이 마지막 가용 턴을 사과로 태웠다(`TASK_072.md` §6-②).
#
# ── 표적이 맞는지 먼저 쟀다 (x576 · 2026-08-28) ──────
#   현 세대 072 네 sim 전부 궤적에 `submit_transaction` **축자 0회** ⇒ 술어가 전부 겨냥한다.
#   그리고 그 지목이 **뜬 두 sim 은 크레딧 호출 0건**이다:
#       t7368 s373753  표적_submit_transaction 9회 · 크레딧 0
#       t7363 s373753  표적_submit_transaction 11회 · 크레딧 0
#       t7368 s626729  지목 없음 · 크레딧 0
#       t7363 s626729  지목 없음 · 크레딧 3 (Bluest `14` **MATCHED** · LG `5` WRONGARG)
#   래칫 `test_actionreq_grounded` 는 실데이터 1,059 발화 표본에서 PASS 다.
#
# ── 대조·판정 ────────────────────────────────────────
#   대조 = **t7368**(072 0/2) + **t7363**(072 0/2) = 0/4. 이 런은 같은 레버 집합에
#   `T2_ACTIONREQ_GROUNDED=1` **하나만** 더한다([[70]] 귀속).
#   1차 판정 = **크레딧 호출이 서는가**(0건 → n건). pass 는 2차다 — Light Green 의
#   부호합($3.50)은 `x542` 가 5팔×24셀 전부 0 으로 잰 자리이고 이 레버는 그것을 안 산다.
#   ⇒ **이 런에서 pass 가 안 나와도 실패가 아니다.** 재는 것은 write 착수 채널이다.
#
# ── 발사 게이트 ──────────────────────────────────────
#   배터리 exit 0 · 스모크에서 `[T2_ACTIONREQ]` 자리가 **살아 있고**(줄>0) **예외 0**.
#   ⚠침묵 자체는 게이트가 아니다 — 모델 거동에 달린 조건부라 nt=1 스모크에서 0 일 수 있다.
#     성공 경로만 세면 *"안 났다"* 와 *"나다 죽었다"* 가 안 갈린다(핸드오프 §5).
# ⛔`set -u` 금지 · 스모크 디렉터리만 치운다 · 줄 이음 금지
set -o pipefail
REPO=/home/woori/workspace_common/boltzmann-attention-pi
LOG=/home/woori/scratch/logs
SIMS=/home/woori/scratch/tau2-bench/data/simulations
STAMP=20260828
TASKS="task_072"
SMOKE_TASKS="task_072"
cd "$REPO/scripts/distill/tau2"

echo "[t7369 $(date +%H:%M:%S)] === 발사 전 배터리 ==="
BAD=0
for t in test_procedure_left test_actionreq_grounded test_operator_find_executed test_decision_point_load test_diag_unambiguous test_read_per_entity test_flag_registry test_arg_label test_a2_three_layer test_card_docs; do
  PYTHONPATH=. PYTHONIOENCODING=utf-8 /home/woori/venvs/seka_env/bin/python $t.py >/dev/null 2>&1
  rc=$?; echo "  $t exit=$rc"; [ $rc -ne 0 ] && BAD=1
done
if [ $BAD -ne 0 ]; then echo "[t7369] ⛔배터리 붉음 — 발사하지 않는다"; exit 1; fi

env_arm() {
  source ./go_stack.sh >/dev/null 2>&1
  # t7368 과 **같은 레버 집합** — 판정을 이어 붙이려고 다른 노브는 건드리지 않는다.
  export T2_ACTION_SUB=1 T2_KEEP_DENY_BODY=1 T2_CALL_FORM=1 T2_ARG_EMPTY=1 T2_SEARCH_AGENT=1
  export T2_DECIDE_ANY=1 T2_WRITE_ARG_ENUM=1 T2_ARG_DOC_SUB=1 T2_VALUE_FORMULA=full
  export T2_SG_DOCS=1 T2_SG_PROMPT_V2=1 T2_SPEC_AT_WRITE=1 T2_WRITE_ARG_TYPE=1
  export T2_RULE_AT_WRITE=1 T2_DUP_WRITE=1
  export T2_READ_PER_ENTITY=0
  export T2_ARG_LABEL=0 T2_DIAG_UNAMBIGUOUS=0
  export T2_CARD_DOCS=1
  export T2_REARM_USER_ONLY=0 T2_PROCEDURE_LEFT=0 T2_EPLAN_ENUM_SUBTRACT=0 T2_SCOPE_ALL=0
  # ★이 런이 재는 **유일한 차이**.
  export T2_ACTIONREQ_GROUNDED=1
  export GO_MAX_STEPS=150
}

# ── ① 스모크 — 자리가 살아 있는지만 본다(성적 아님) ───────────────────────
STAG="bank_t7369_smoke_${STAMP}"
# ★스모크 디렉터리는 자기 것만 치운다. 본런 디렉터리는 건드리지 않는다([[30]]) —
#   남아 있으면 tau2 가 stdin 으로 resume 을 묻고 `</dev/null` 이라 EOFError 로 죽는다.
rm -rf "$SIMS/$STAG"
(
  env_arm; export GO_CONCURRENCY=1
  echo "[smoke $(date +%H:%M:%S)] $SMOKE_TASKS nt=1"
  t2_launch "$STAG" 8140 "$SMOKE_TASKS" 1 2>&1 | tee "$LOG/$STAG.log"
) > "$LOG/${STAG}_driver.log" 2>&1

REQ=$(grep -ac "\[T2_ACTIONREQ\]" "$LOG/$STAG.log" 2>/dev/null); REQ=${REQ:-0}
SIL=$(grep -ac "T2_ACTIONREQ\] 침묵" "$LOG/$STAG.log" 2>/dev/null); SIL=${SIL:-0}
EXC=$(grep -ac "grounded 검사 건너뜀" "$LOG/$STAG.log" 2>/dev/null); EXC=${EXC:-0}
TGT=$(grep -ao "formalized_target=[a-z_]*" "$LOG/$STAG.log" 2>/dev/null | sort | uniq -c | tr '\n' ' ')
TB=$(grep -ac "Traceback" "$LOG/$STAG.log" 2>/dev/null); TB=${TB:-0}
echo "[t7369] 스모크: ACTIONREQ 줄=$REQ · 침묵=$SIL · **예외=$EXC** · Traceback=$TB"
echo "  표적 분포: $TGT"
if [ "$EXC" -ne 0 ]; then echo "[t7369] ⛔술어가 예외로 죽었다 — 중단."; exit 1; fi
if [ "$REQ" -eq 0 ]; then echo "[t7369] ⛔ACTIONREQ 자리에 한 번도 안 닿았다 — 중단."; exit 1; fi
if [ "$TB" -ne 0 ]; then echo "[t7369] ⛔Traceback $TB — 중단."; exit 1; fi

# ── ② 본런 ────────────────────────────────────────────────────────────────
cd "$REPO"
/home/woori/venvs/seka_env/bin/python reports/facet_rft_2026/freeze.py --on \
  --tag "bank_t7369_$STAMP" --reason "t7369 072: silence the ungrounded customer-tool pointer" || true
cd "$REPO/scripts/distill/tau2"

TAG="bank_t7369_072_${STAMP}"
(
  env_arm; export GO_CONCURRENCY=2
  echo "[main $(date +%H:%M:%S)] $TASKS nt=4"
  t2_launch "$TAG" 8140 "$TASKS" 4 2>&1 | tee "$LOG/$TAG.log"
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
git add -f reports/facet_rft_2026/sim_results/*bank_t7369_* 2>/dev/null || true
git -c user.name=ghlee -c user.email=beingrelative@gmail.com \
  commit -q -m "t7369 072: the ungrounded pointer silenced, measured against t7368" \
  -- reports/facet_rft_2026/sim_results || true
# ★push 실패를 삼키지 않는다([[30]] 1순위 사고 경로). 그 사이 로컬 푸시가 있으면
#   non-fast-forward 가 나므로 **rebase 후 한 번 더** 시도한다(t7368 에서 손으로 했다).
if ! git push -q origin facet-rft-2026; then
  echo "[t7369] push 실패 — rebase 후 재시도"
  git -c user.name=ghlee -c user.email=beingrelative@gmail.com pull --rebase -q origin facet-rft-2026 || true
  if ! git push -q origin facet-rft-2026; then
    echo "[t7369] ⛔재시도도 실패 — 결과가 이 디스크에만 있다. 손으로 회수하라."
    PUSH_FAIL=1
  fi
fi
echo "=== TRACKED ==="
for F in reports/facet_rft_2026/sim_results/*bank_t7369_*; do
  git ls-files --error-unmatch "$F" >/dev/null 2>&1 && echo "  ok   $(basename $F)" || echo "  ⛔UNTRACKED $(basename $F)"
done

echo "=== 성적 ==="
TAG="$TAG" /home/woori/venvs/seka_env/bin/python - <<'PY'
# -*- coding: utf-8 -*-
import os, sys, json, collections
sys.path.insert(0, "scripts/distill/tau2")
sys.stdout.reconfigure(encoding="utf-8")
import t2_forensic as F
tag = os.environ["TAG"]
try:
    sims = F.sims(tag)
except Exception as e:
    print("결과 로드 실패: %r" % (e,)); raise SystemExit(0)
mut = F.mutating_tools()
print("  pass %d/%d" % (sum(1 for s in sims
                            if (s.get("reward_info") or {}).get("reward") == 1.0), len(sims)))
for s in sims:
    d = F.mutation_diff(s, mut, tag=tag) or {}
    gap = sum(len(d.get(k) or ()) for k in ("missing", "wrongarg", "extra", "dup"))
    cred = [c for c in F.calls(s)
            if "apply_checking_account_credit" in str(F.inner_name(F.argsof(c) or {})
                                                      or F.nameof(c) or "")]
    print("    %-20s reward=%-5s gap=%-3d 크레딧호출=%-3d %s"
          % (F.simtag(s), (s.get("reward_info") or {}).get("reward"), gap, len(cred),
             F.term_reason(s)))
    for k in ("matched", "missing", "wrongarg", "extra", "dup"):
        for x in (d.get(k) or ()):
            print("        %-9s %s" % (k, x.get("key")))
PY

echo "=== 레버 발화 ==="
for M in "T2_ACTIONREQ] 침묵" "formalized_target=submit_transaction" "formalized_target=None" "T2_FORCE_ACTION" "grounded 검사 건너뜀" "Traceback"; do
  C=$(grep -ac "$M" "$LOG/$TAG.log" 2>/dev/null); echo "  $M = ${C:-0}"
done
echo "[t7369] DONE $(date +%H:%M:%S)"
[ -n "$PUSH_FAIL" ] && exit 1
exit 0
