#!/usr/bin/env bash
# t7368 — hard-0 전체 · `T2_CARD_DOCS` 부작용 측정 (사용자 지시 2026-08-27)
#
# t7367 이 016 을 **2/4** 로 통과시켰다(대조 t7365 0/4 · 이 레버만 차이).
# 이제 물을 것은 [[70]] ①전체 reward 짝 ②태스크별 부호표 ③무엇을 팔았나 이다 —
# 016 밖에서 이 서브가 서는가, 서면 해로운가.
#
# ── 표적·대조·판정 ───────────────────────────────────
#   hard-0 10 × nt2 = 20 sim · 대조 = **t7363**(0/20 · 같은 10 태스크).
#   판정 = **태스크별 0→1 과 1→0 을 함께** 본다(총점 Δ 금지·C594).
#   ⚠016 은 이미 샀다 — 이 런이 재는 것은 **나머지 아홉의 부호**다.
#
# ── 실린 것 ──────────────────────────────────────────
#   `T2_CARD_DOCS=1` + t7365/t7367 과 같은 레버 집합. ARG_LABEL·READ_PER_ENTITY 는 0.
#
# ── 발사 게이트 ──────────────────────────────────────
#   배터리 exit 0 · 스모크에서 CARD_DOCS **예외 0**(줄 수·인용 수·예외 수를 함께 센다)
# ⛔`set -u` 금지 · 스모크 디렉터리만 치운다 · 줄 이음 금지
set -o pipefail
REPO=/home/woori/workspace_common/boltzmann-attention-pi
LOG=/home/woori/scratch/logs
SIMS=/home/woori/scratch/tau2-bench/data/simulations
STAMP=20260827
TASKS="task_016,task_040,task_055,task_057,task_063,task_072,task_074,task_079,task_085,task_094"
SMOKE_TASKS="task_016,task_072"
cd "$REPO/scripts/distill/tau2"

echo "[t7368 $(date +%H:%M:%S)] === 발사 전 배터리 ==="
BAD=0
for t in test_procedure_left test_actionreq_grounded test_operator_find_executed test_decision_point_load test_diag_unambiguous test_read_per_entity test_flag_registry test_arg_label test_a2_three_layer test_card_docs; do
#
##
  PYTHONPATH=. PYTHONIOENCODING=utf-8 /home/woori/venvs/seka_env/bin/python $t.py >/dev/null 2>&1
  rc=$?; echo "  $t exit=$rc"; [ $rc -ne 0 ] && BAD=1
done
if [ $BAD -ne 0 ]; then echo "[t7368] ⛔배터리 붉음 — 발사하지 않는다"; exit 1; fi

env_arm() {
  source ./go_stack.sh >/dev/null 2>&1
  # t7361/t7362 와 **같은 레버 집합** — 판정을 이어 붙이려고 노브를 더 바꾸지 않는다.
  export T2_ACTION_SUB=1 T2_KEEP_DENY_BODY=1 T2_CALL_FORM=1 T2_ARG_EMPTY=1 T2_SEARCH_AGENT=1
  export T2_DECIDE_ANY=1 T2_WRITE_ARG_ENUM=1 T2_ARG_DOC_SUB=1 T2_VALUE_FORMULA=full
  export T2_SG_DOCS=1 T2_SG_PROMPT_V2=1 T2_SPEC_AT_WRITE=1 T2_WRITE_ARG_TYPE=1
  export T2_RULE_AT_WRITE=1 T2_DUP_WRITE=1
  # ★이 런이 재는 것. go_stack 이 이미 =1 로 내보내지만 런 기록에 남기려고 명시한다.
  export T2_READ_PER_ENTITY=0
  # ⛔이 런이 **싣지 않는** 것들 (위 주석 참조)
  export T2_ARG_LABEL=0 T2_DIAG_UNAMBIGUOUS=0
  # ★이 런이 재는 것.
  export T2_CARD_DOCS=1
  export T2_REARM_USER_ONLY=0 T2_PROCEDURE_LEFT=0 T2_EPLAN_ENUM_SUBTRACT=0 T2_SCOPE_ALL=0
  export GO_MAX_STEPS=150
}

# ── ① 스모크 — **레버가 나는지**만 본다(성적 아님) ────────────────────────
STAG="bank_t7368_smoke_${STAMP}"
# ★스모크 디렉터리는 **자기 것만** 치운다. 남아 있으면 tau2 가 stdin 으로
#   *"Do you want to resume the run? (y/n)"* 를 묻고 `</dev/null` 이라 **EOFError** 로
#   죽는다(2026-08-27 실측 — 그 크래시가 "레버 미발화" 처럼 보였다).
#   ⛔본런 디렉터리는 건드리지 않는다([[30]]: 같은 태그 재런이 앞 데이터를 덮은 사고).
rm -rf "$SIMS/$STAG"
(
  env_arm; export GO_CONCURRENCY=2
  echo "[smoke $(date +%H:%M:%S)] $SMOKE_TASKS nt=1"
  t2_launch "$STAG" 8140 "$SMOKE_TASKS" 1 2>&1 | tee "$LOG/$STAG.log"
) > "$LOG/${STAG}_driver.log" 2>&1

# ★마커는 **stderr 자국**이어야 한다. 1차 발사에서 나는 배달 문구(*"A separate check was
#   run …"*)를 셌는데 그것은 **생성-뷰 채널**로 나가므로 드라이버 로그에 안 남는다 —
#   레버가 옳게 발화하고 `Silver Rewards Card` 를 답했는데도 게이트가 런을 세웠다.
#   오늘 같은 계열의 실수를 세 번 했다(x560 주체·x564 생산자·이 마커) — 전부 *"어디에
#   있었나"* 와 *"무엇으로 있었나"* 를 섞은 것이다.
# ⚠마커는 **성공 경로**만 세면 안 된다 — 1차에서 `건너뜀` 18회를 발화 0 으로 읽었다.
DIAG=$(grep -ac "T2_CARD_DOCS" "$LOG/$STAG.log" 2>/dev/null); DIAG=${DIAG:-0}
SKIP=$(grep -ac "T2_CARD_DOCS] 건너뜀" "$LOG/$STAG.log" 2>/dev/null); SKIP=${SKIP:-0}
CITE=$(grep -a "T2_CARD_DOCS] subject=" "$LOG/$STAG.log" 2>/dev/null | grep -ac "인용 doc"); CITE=${CITE:-0}
CRIT=$(grep -ao "\[T2_DIAG\] raw=.\{0,60\}" "$LOG/$STAG.log" 2>/dev/null | head -2)
TB=$(grep -ac "Traceback" "$LOG/$STAG.log" 2>/dev/null); TB=${TB:-0}
echo "[t7368] 스모크: CARD_DOCS 줄=$DIAG · 문서 인용=$CITE · **예외=$SKIP** · Traceback=$TB"
if [ "$SKIP" -ne 0 ]; then echo "[t7368] ⛔술어가 예외로 죽었다 — 중단."; exit 1; fi
echo "  답: $CRIT"
if [ "$DIAG" -eq 0 ]; then
  echo "[t7368] ⛔CARD_DOCS 가 한 번도 안 섰다 — 배선이 죽었다. 중단."
  exit 1
fi
if [ "$TB" -ne 0 ]; then echo "[t7368] ⛔Traceback $TB — 중단."; exit 1; fi

# ── ② 본런 ────────────────────────────────────────────────────────────────
cd "$REPO"
/home/woori/venvs/seka_env/bin/python reports/facet_rft_2026/freeze.py --on \
  --tag "bank_t7368_$STAMP" --reason "t7368 hard-0 ten: side effects of the card documents lever" || true
cd "$REPO/scripts/distill/tau2"

TAG="bank_t7368_hard0_${STAMP}"
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
#     ⒜ `git add bank_t7368_*` 는 **`fb_`·`trace_` 접두를 못 잡는다** — 사이드카 6 파일이
#        그래서 add 조차 안 됐다. 글롭을 `*bank_t7368_*` 로 바꾼다.
#     ⒝ push 실패를 `|| echo` 로 삼켜 결과가 디스크 한 곳에만 남았다 — **exit 1** 로 죽는다.
mkdir -p reports/facet_rft_2026/sim_results
for T in "$STAG" "$TAG"; do
  gzip -c "$SIMS/$T/results.json" > reports/facet_rft_2026/sim_results/$T.results.json.gz 2>/dev/null || true
  gzip -c "$LOG/$T.log" > reports/facet_rft_2026/sim_results/$T.log.gz 2>/dev/null || true
  for S in fb trace; do
    F="$LOG/${S}_${T}.jsonl"; [ -s "$F" ] && gzip -c "$F" > reports/facet_rft_2026/sim_results/${S}_${T}.jsonl.gz
  done
done
git add -f reports/facet_rft_2026/sim_results/*bank_t7368_* 2>/dev/null || true
git -c user.name=ghlee -c user.email=beingrelative@gmail.com \
  commit -q -m "t7368 hard-0: the per-subject read requirement, measured against t7363" \
  -- reports/facet_rft_2026/sim_results || true
if ! git push -q origin facet-rft-2026; then
  echo "[t7368] ⛔push 실패 — 결과가 이 디스크에만 있다. 손으로 회수하라([[30]] 1순위 사고)."
  PUSH_FAIL=1
fi
echo "=== TRACKED ==="
for F in reports/facet_rft_2026/sim_results/*bank_t7368_*; do
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
for M in "T2_CARD_DOCS] subject=" "인용 doc" "750" "Traceback"; do
  C=$(grep -ac "$M" "$LOG/$TAG.log" 2>/dev/null); echo "  $M = ${C:-0}"
done
echo "[t7368] DONE $(date +%H:%M:%S)"
[ -n "$PUSH_FAIL" ] && exit 1
exit 0
