#!/usr/bin/env bash
# t7366 — 016 단독 · **프롬프트 회수 런** (사용자 승인 2026-08-27)
#
# 목적은 성적이 아니라 **계기**다. 오늘 두 번 라이브 실패가 같은 접두 위 격리에서 재현되지
# 않았다(x562 B_live 4/4 · x571 A_asis 가 옳게 답함). 라이브 프롬프트가 어디에도 기록되지
# 않기 때문이다 — 영속 궤적(커밋분) + 비커밋 `work` 주입 + 뷰 압축의 **합**이 실제 프롬프트인데
# 그 합은 안 남는다. `T2_PROMPT_DUMP=1` 이 `_gen` 에서 그것을 사이드카에 남긴다.
#
# 회수 대상 = 016 의 **기록 주장 발화 직전** 프롬프트(t7365 s1567 의 msg[42] 자리).
# 그 프롬프트와 내 격리 프롬프트를 축자로 diff 하는 것이 다음 작업이다([[78]]).
#
# 레버 집합은 t7365 와 **동일**하다 — 계기만 켠다. 성적은 부차이고 대조는 t7365(0/4).
# ⛔`set -u` 금지 · 스모크 디렉터리는 드라이버가 치운다(tau2 resume 프롬프트 회피).
set -o pipefail
REPO=/home/woori/workspace_common/boltzmann-attention-pi
LOG=/home/woori/scratch/logs
SIMS=/home/woori/scratch/tau2-bench/data/simulations
STAMP=20260827
TASKS="task_016"
SMOKE_TASKS="task_016"
cd "$REPO/scripts/distill/tau2"

echo "[t7366 $(date +%H:%M:%S)] === 발사 전 배터리 ==="
BAD=0
for t in test_procedure_left test_actionreq_grounded test_operator_find_executed \
         test_decision_point_load test_diag_unambiguous \
         test_read_per_entity test_flag_registry \
         test_arg_label test_a2_three_layer; do
  PYTHONPATH=. PYTHONIOENCODING=utf-8 /home/woori/venvs/seka_env/bin/python $t.py >/dev/null 2>&1
  rc=$?; echo "  $t exit=$rc"; [ $rc -ne 0 ] && BAD=1
done
if [ $BAD -ne 0 ]; then echo "[t7366] ⛔배터리 붉음 — 발사하지 않는다"; exit 1; fi

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
  # ★이 런의 목적. 태스크로 좁히고 상한을 둔다(턴당 30~40k자).
  export T2_PROMPT_DUMP=1 T2_PROMPT_DUMP_TASKS=task_016 T2_PROMPT_DUMP_MAX=60000
  export T2_REARM_USER_ONLY=0 T2_PROCEDURE_LEFT=0 T2_EPLAN_ENUM_SUBTRACT=0 T2_SCOPE_ALL=0
  export GO_MAX_STEPS=150
}

# ── ① 스모크 — **레버가 나는지**만 본다(성적 아님) ────────────────────────
STAG="bank_t7366_smoke_${STAMP}"
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
# 게이트는 **산출물**을 본다 — 자국이 아니라 사이드카에 실제로 실린 prompt 레코드 수.
SC="$LOG/fb_${STAG}.jsonl"
DIAG=$(grep -ac '"kind": "prompt"' "$SC" 2>/dev/null); DIAG=${DIAG:-0}
SKIP=$(grep -ac "PROMPT_DUMP] skipped" "$LOG/$STAG.log" 2>/dev/null); SKIP=${SKIP:-0}
CRIT=$(grep -ao "\[T2_DIAG\] raw=.\{0,60\}" "$LOG/$STAG.log" 2>/dev/null | head -2)
TB=$(grep -ac "Traceback" "$LOG/$STAG.log" 2>/dev/null); TB=${TB:-0}
echo "[t7366] 스모크: prompt 레코드=$DIAG · 기록기 예외=$SKIP · Traceback=$TB"
echo "  답: $CRIT"
if [ "$DIAG" -eq 0 ]; then
  echo "[t7366] ⛔프롬프트 기록기가 예외를 냈다 — 중단."
  exit 1
fi
if [ "$TB" -ne 0 ]; then echo "[t7366] ⛔Traceback $TB — 중단."; exit 1; fi

# ── ② 본런 ────────────────────────────────────────────────────────────────
cd "$REPO"
/home/woori/venvs/seka_env/bin/python reports/facet_rft_2026/freeze.py --on \
  --tag "bank_t7366_$STAMP" --reason "t7366 016 alone: capture the live prompt" || true
cd "$REPO/scripts/distill/tau2"

TAG="bank_t7366_hard0_${STAMP}"
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
#     ⒜ `git add bank_t7366_*` 는 **`fb_`·`trace_` 접두를 못 잡는다** — 사이드카 6 파일이
#        그래서 add 조차 안 됐다. 글롭을 `*bank_t7366_*` 로 바꾼다.
#     ⒝ push 실패를 `|| echo` 로 삼켜 결과가 디스크 한 곳에만 남았다 — **exit 1** 로 죽는다.
mkdir -p reports/facet_rft_2026/sim_results
for T in "$STAG" "$TAG"; do
  gzip -c "$SIMS/$T/results.json" > reports/facet_rft_2026/sim_results/$T.results.json.gz 2>/dev/null || true
  gzip -c "$LOG/$T.log" > reports/facet_rft_2026/sim_results/$T.log.gz 2>/dev/null || true
  for S in fb trace; do
    F="$LOG/${S}_${T}.jsonl"; [ -s "$F" ] && gzip -c "$F" > reports/facet_rft_2026/sim_results/${S}_${T}.jsonl.gz
  done
done
git add -f reports/facet_rft_2026/sim_results/*bank_t7366_* 2>/dev/null || true
git -c user.name=ghlee -c user.email=beingrelative@gmail.com \
  commit -q -m "t7366 hard-0: the per-subject read requirement, measured against t7363" \
  -- reports/facet_rft_2026/sim_results || true
if ! git push -q origin facet-rft-2026; then
  echo "[t7366] ⛔push 실패 — 결과가 이 디스크에만 있다. 손으로 회수하라([[30]] 1순위 사고)."
  PUSH_FAIL=1
fi
echo "=== TRACKED ==="
for F in reports/facet_rft_2026/sim_results/*bank_t7366_*; do
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
for M in "separate check was run" "Silver Rewards Card" "750" "Traceback"; do
  C=$(grep -ac "$M" "$LOG/$TAG.log" 2>/dev/null); echo "  $M = ${C:-0}"
done
echo "[t7366] DONE $(date +%H:%M:%S)"
[ -n "$PUSH_FAIL" ] && exit 1
exit 0
