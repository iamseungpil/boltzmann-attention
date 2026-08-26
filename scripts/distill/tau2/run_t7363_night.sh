#!/usr/bin/env bash
# t7363 — hard-0 밤샘 런 (사용자 승인 2026-08-27)
#
# ⛔`set -u` 금지 (t7360 교훈: go_stack 의 `t2_require_key` 가 미설정을 스스로 처리하는데
#   `set -u` 아래선 그 표현식이 먼저 죽어 t2_launch 가 한 줄도 안 돈다).
#
# ── 무엇이 t7356 과 다른가 ────────────────────────────
#   ⑴ **`T2_DIAG_UNAMBIGUOUS=1`** (신규·go_stack 등재) — 진단 서브의 답이 **이름** 단위인데
#      원장은 **행** 단위다. 한 이름이 상태를 여럿 이면 *"어느 record 가 미지급인가"* 가 그
#      문맥에서 정해지지 않는데 구판은 *"A separate check was run … It answers: X"* 로 단언한다.
#      부호표(발화 119/34 태그): 016 22 제거(전부 reward 0) · 010 12 불변 · 098·099 이미 침묵.
#   ⑵ 앞 세션의 상시 수리 둘 — `[OPERATOR-DIRECT]` 날조 교정 · 연쇄 문면이 대상 도구까지 댐
#      (커밋 3053e5d3·5424988e). **t7362 세 팔은 이것 없이 돌았다** — 이 런이 처음 싣는다.
#
# ── ⛔무엇을 **뺐는가** ───────────────────────────────
#   `T2_REARM_USER_ONLY`(A-3′) 는 **끈다**. go_stack 에는 등재돼 있지만 이 런에서는 0 이다.
#     이유: 그 래칫(`test_rearm_user_only.py`)이 **리모트 코퍼스에서 붉다** — 로컬에서는
#     통과 sim 에서 죽는 발화 3·반증 잔존 1 인데, 리모트에는 영속 안 된 로그가 더 있어
#     같은 술어가 **6·2** 를 낸다. 부호표가 잰 것보다 나쁘면 그것은 *우리가 잰 것과 다른 것을
#     켜는 것*이다([[76]]). 코퍼스를 마저 회수해 다시 재기 전엔 안 싣는다.
#   `T2_PROCEDURE_LEFT`·`T2_EPLAN_ENUM_SUBTRACT`·`T2_SCOPE_ALL` 도 0 —
#     t7362 판정: SCOPE_ALL 은 **음성 확정**(085 에서 반려 48↔20·turn 만 태움), 나머지 둘은
#     **발화 0회라 미측정**이다(*"껐다"* 가 아니라 *"못 쟀다"*).
#
# ── 표적·대조 ────────────────────────────────────────
#   hard-0 10 (x509 axis_table `per_task_required` 인용) × nt2 = 20 sim · 대조 = t7356
#   판정 = **표적의 0→1**. ⛔총점 Δ 로 판정하지 마라(C594).
#   특히 016 — `T2_DIAG_UNAMBIGUOUS` 가 유일하게 겨냥하는 태스크다.
#
# ── 발사 게이트 ──────────────────────────────────────
#   ① 배터리 6종 exit 0  ② 스모크에서 **새 레버가 실제로 발화**(자국 0 이면 중단)
#   이 런의 교훈: t7362 는 세 레버 중 둘이 **한 번도 안 나서** 아무것도 못 쟀다([[30]]).
set -o pipefail
REPO=/home/woori/workspace_common/boltzmann-attention-pi
LOG=/home/woori/scratch/logs
SIMS=/home/woori/scratch/tau2-bench/data/simulations
STAMP=20260827
TASKS="task_016,task_040,task_055,task_057,task_063,task_072,task_074,task_079,task_085,task_094"
SMOKE_TASKS="task_016,task_055"
cd "$REPO/scripts/distill/tau2"

echo "[t7363 $(date +%H:%M:%S)] === 발사 전 배터리 ==="
BAD=0
for t in test_procedure_left test_actionreq_grounded test_operator_find_executed \
         test_decision_point_load test_diag_unambiguous; do
  PYTHONPATH=. PYTHONIOENCODING=utf-8 /home/woori/venvs/seka_env/bin/python $t.py >/dev/null 2>&1
  rc=$?; echo "  $t exit=$rc"; [ $rc -ne 0 ] && BAD=1
done
if [ $BAD -ne 0 ]; then echo "[t7363] ⛔배터리 붉음 — 발사하지 않는다"; exit 1; fi

env_arm() {
  source ./go_stack.sh >/dev/null 2>&1
  # t7361/t7362 와 **같은 레버 집합** — 판정을 이어 붙이려고 노브를 더 바꾸지 않는다.
  export T2_ACTION_SUB=1 T2_KEEP_DENY_BODY=1 T2_CALL_FORM=1 T2_ARG_EMPTY=1 T2_SEARCH_AGENT=1
  export T2_DECIDE_ANY=1 T2_WRITE_ARG_ENUM=1 T2_ARG_DOC_SUB=1 T2_VALUE_FORMULA=full
  export T2_SG_DOCS=1 T2_SG_PROMPT_V2=1 T2_SPEC_AT_WRITE=1 T2_WRITE_ARG_TYPE=1
  export T2_RULE_AT_WRITE=1 T2_DUP_WRITE=1
  # ⛔이 런이 **싣지 않는** 것들 (위 주석 참조)
  export T2_REARM_USER_ONLY=0 T2_PROCEDURE_LEFT=0 T2_EPLAN_ENUM_SUBTRACT=0 T2_SCOPE_ALL=0
  export GO_MAX_STEPS=150
}

# ── ① 스모크 — **레버가 나는지**만 본다(성적 아님) ────────────────────────
STAG="bank_t7363_smoke_${STAMP}"
(
  env_arm; export GO_CONCURRENCY=2
  echo "[smoke $(date +%H:%M:%S)] $SMOKE_TASKS nt=1"
  t2_launch "$STAG" 8140 "$SMOKE_TASKS" 1 2>&1 | tee "$LOG/$STAG.log"
) > "$LOG/${STAG}_driver.log" 2>&1

DIAG=$(grep -ac "T2_DIAG] 모호" "$LOG/$STAG.log" 2>/dev/null); DIAG=${DIAG:-0}
TB=$(grep -ac "Traceback" "$LOG/$STAG.log" 2>/dev/null); TB=${TB:-0}
echo "[t7363] 스모크: T2_DIAG 모호-침묵 발화=$DIAG · Traceback=$TB"
if [ "$DIAG" -eq 0 ]; then
  echo "[t7363] ⛔새 레버가 **한 번도 안 났다** — t7362 와 같은 실수를 반복하지 않는다. 중단."
  exit 1
fi
if [ "$TB" -ne 0 ]; then echo "[t7363] ⛔Traceback $TB — 중단."; exit 1; fi

# ── ② 본런 ────────────────────────────────────────────────────────────────
cd "$REPO"
/home/woori/venvs/seka_env/bin/python reports/facet_rft_2026/freeze.py --on \
  --tag "bank_t7363_$STAMP" --reason "t7363 hard-0 night: T2_DIAG_UNAMBIGUOUS + 상시 수리 둘" || true
cd "$REPO/scripts/distill/tau2"

TAG="bank_t7363_hard0_${STAMP}"
(
  env_arm; export GO_CONCURRENCY=3   # t7362 의 3-동시와 같은 서버 부하 ⇒ 지연 조건 비교 가능
  echo "[main $(date +%H:%M:%S)] $TASKS nt=2"
  t2_launch "$TAG" 8140 "$TASKS" 2 2>&1 | tee "$LOG/$TAG.log"
  echo "[main $(date +%H:%M:%S)] done"
) > "$LOG/${TAG}_driver.log" 2>&1

cd "$REPO"
/home/woori/venvs/seka_env/bin/python reports/facet_rft_2026/freeze.py --off

# ── ③ 회수 ([[30]]: gzip 만으론 영속이 아니다) ────────────────────────────
#   ⚠t7362 의 실물 결함 둘을 여기서 고친다:
#     ⒜ `git add bank_t7363_*` 는 **`fb_`·`trace_` 접두를 못 잡는다** — 사이드카 6 파일이
#        그래서 add 조차 안 됐다. 글롭을 `*bank_t7363_*` 로 바꾼다.
#     ⒝ push 실패를 `|| echo` 로 삼켜 결과가 디스크 한 곳에만 남았다 — **exit 1** 로 죽는다.
mkdir -p reports/facet_rft_2026/sim_results
for T in "$STAG" "$TAG"; do
  gzip -c "$SIMS/$T/results.json" > reports/facet_rft_2026/sim_results/$T.results.json.gz 2>/dev/null || true
  gzip -c "$LOG/$T.log" > reports/facet_rft_2026/sim_results/$T.log.gz 2>/dev/null || true
  for S in fb trace; do
    F="$LOG/${S}_${T}.jsonl"; [ -s "$F" ] && gzip -c "$F" > reports/facet_rft_2026/sim_results/${S}_${T}.jsonl.gz
  done
done
git add -f reports/facet_rft_2026/sim_results/*bank_t7363_* 2>/dev/null || true
git -c user.name=ghlee -c user.email=beingrelative@gmail.com \
  commit -q -m "t7363 hard-0 night: diag-unambiguous + the two standing repairs" \
  -- reports/facet_rft_2026/sim_results || true
if ! git push -q origin facet-rft-2026; then
  echo "[t7363] ⛔push 실패 — 결과가 이 디스크에만 있다. 손으로 회수하라([[30]] 1순위 사고)."
  PUSH_FAIL=1
fi
echo "=== TRACKED ==="
for F in reports/facet_rft_2026/sim_results/*bank_t7363_*; do
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
per = collections.defaultdict(list)
for s in sims:
    per[F.task_id(s)].append((s.get("reward_info") or {}).get("reward"))
tot = sum(1 for v in per.values() for r in v if r == 1.0)
n = sum(len(v) for v in per.values())
print("  pass %d/%d" % (tot, n))
for t in sorted(per):
    print("    %-10s %s" % (t, per[t]))
print("  종료사유: %s" % dict(collections.Counter(F.term_reason(s) for s in sims)))
PY

echo "=== 레버 발화 ==="
for M in "T2_DIAG] 모호" "T2_SEARCH_REARM] group=" "OPERATOR-DIRECT" "Traceback"; do
  C=$(grep -ac "$M" "$LOG/$TAG.log" 2>/dev/null); echo "  $M = ${C:-0}"
done
echo "[t7363] DONE $(date +%H:%M:%S)"
[ -n "$PUSH_FAIL" ] && exit 1
exit 0
