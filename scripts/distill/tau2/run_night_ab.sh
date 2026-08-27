#!/usr/bin/env bash
# 밤샘 A/B — **같은 sha 위**에서 처치/대조를 두 GPU 에 하나씩 (사용자 지시 2026-08-28 밤)
#
# ── 왜 매개변수 드라이버인가 ─────────────────────────
#   처치와 대조가 **레버 두 칸 말고는 한 글자도 달라선 안 된다**. 파일을 둘로 복사하면
#   그 보장이 사람의 주의력에 걸린다(오늘 하루에만 사본 어긋남을 두 번 봤다 — `a2/split` 6/10 ·
#   `test_atm_fee_op` 계약 모순). 그래서 **한 파일 · 팔은 env 한 칸**이다.
#
# ── 무엇을 재나 ──────────────────────────────────────
#   오늘 지은 레버 둘의 **합성** 효과([[19]]) — 그리고 그 판정을 위한 **같은 sha 대조군**([[73]]).
#     `T2_ACTIONREQ_GROUNDED`  대화에 없는 손님-측 도구 지목을 침묵 (t7369 가 072 에서 단독 측정)
#     `T2_SG_ROW_COUNT`        전사 서브가 선언된 종류보다 적게 넘기면 총액을 단언하지 않음
#                              (t7370 이 072·074·085 에서 단독 측정)
#   ⚠t7368(대조로 써 오던 런)은 sha `82012e6e` 다. 그 뒤로 a2·go_stack·t2_scaffold_get 이
#     바뀌었으므로 t7368 과의 Δ 는 **오늘 바뀐 전부와 교락**한다. 이 A/B 가 그것을 끊는다.
#
# ── 사용 ─────────────────────────────────────────────
#   ARM=treat|control TAG=bank_t7371_... AB_PORT=8140 [TASKS=..] [NT=2] [CONC=3] bash run_night_ab.sh
#
# ⛔**`PORT` 라는 이름을 쓰지 마라 — 이 호스트에서는 통하지 않는다** (2026-08-28 01:50 실측):
#     로그인 셸        PORT=[]
#     bash -c 안       PORT=[8100]        ← 새 bash 마다 비대화 시작 파일이 넣는다
#     PORT=8140 bash   PORT=[8100]        ← **접두 할당을 덮어쓴다**(시작 파일이 나중에 돈다)
#     PORT=8140 env    PORT=8140          ← env 는 시작 파일을 안 타므로 여기선 8140 로 보인다
#   그래서 `PORT=8140` 으로 넘긴 t7371 이 **8100(node 프로세스)** 에 붙어 죽었다. 스크립트 안에서
#   직접 대입하는 경우는 무사하다(본문이 시작 파일보다 나중이라 — run_t7370 이 8141 로 잘 돌았다).
#
# ⛔`set -u` 금지 · 스모크 디렉터리만 치운다 · 줄 이음 금지
set -o pipefail
REPO=/home/woori/workspace_common/boltzmann-attention-pi
LOG=/home/woori/scratch/logs
SIMS=/home/woori/scratch/tau2-bench/data/simulations
ARM="${ARM:-treat}"
AB_PORT="${AB_PORT:-8140}"
NT="${NT:-2}"
CONC="${CONC:-3}"
TASKS="${TASKS:-task_016,task_040,task_055,task_057,task_063,task_072,task_074,task_079,task_085,task_094}"
SMOKE_TASKS="${SMOKE_TASKS:-task_072}"
if [ -z "$TAG" ]; then echo "⛔TAG 가 없다"; exit 1; fi
case "$ARM" in treat|control) ;; *) echo "⛔ARM 은 treat|control"; exit 1;; esac
cd "$REPO/scripts/distill/tau2"

echo "[$TAG $(date +%H:%M:%S)] === $ARM · port=$AB_PORT · nt=$NT · conc=$CONC ==="
echo "[$TAG] sha=$(cd $REPO && git rev-parse --short HEAD)"

echo "[$TAG] === 발사 전 배터리 ==="
BAD=0
# ★`test_atm_fee_op` 을 **넣는다**. 오늘까지 이 파일은 어느 배터리에도 없었고 그래서 08-26 부터
#   붉은 채 아무도 못 봤다(계약 두 개가 정면 모순인 상태로 이틀). 오늘 29/29 로 고쳤으니 싣는다.
for t in test_sg_row_count test_freeze_multihold test_atm_fee_op test_atm_ledger_close \
         test_rebate_netting test_delta_total_used test_actionreq_grounded test_procedure_left \
         test_operator_find_executed test_decision_point_load test_diag_unambiguous \
         test_read_per_entity test_flag_registry test_arg_label test_a2_three_layer test_card_docs; do
  PYTHONPATH=. PYTHONIOENCODING=utf-8 /home/woori/venvs/seka_env/bin/python $t.py >/dev/null 2>&1
  rc=$?; echo "  $t exit=$rc"; [ $rc -ne 0 ] && BAD=1
done
if [ $BAD -ne 0 ]; then echo "[$TAG] ⛔배터리 붉음 — 발사하지 않는다"; exit 1; fi

env_arm() {
  source ./go_stack.sh >/dev/null 2>&1
  # ── 공통(= t7368 의 레버 집합). 두 팔이 여기서 한 글자도 안 다르다.
  export T2_ACTION_SUB=1 T2_KEEP_DENY_BODY=1 T2_CALL_FORM=1 T2_ARG_EMPTY=1 T2_SEARCH_AGENT=1
  export T2_DECIDE_ANY=1 T2_WRITE_ARG_ENUM=1 T2_ARG_DOC_SUB=1 T2_VALUE_FORMULA=full
  export T2_SG_DOCS=1 T2_SG_PROMPT_V2=1 T2_SPEC_AT_WRITE=1 T2_WRITE_ARG_TYPE=1
  export T2_RULE_AT_WRITE=1 T2_DUP_WRITE=1
  export T2_READ_PER_ENTITY=0 T2_ARG_LABEL=0 T2_DIAG_UNAMBIGUOUS=0 T2_CARD_DOCS=1
  export T2_REARM_USER_ONLY=0 T2_PROCEDURE_LEFT=0 T2_EPLAN_ENUM_SUBTRACT=0 T2_SCOPE_ALL=0
  export GO_MAX_STEPS=150
  # ── 팔 (여기만 다르다)
  if [ "$ARM" = "treat" ]; then
    export T2_ACTIONREQ_GROUNDED=1 T2_SG_ROW_COUNT=1
  else
    export T2_ACTIONREQ_GROUNDED=0 T2_SG_ROW_COUNT=0
  fi
}

# ── ① 스모크 — 팔에 맞는 것만 본다 ───────────────────────────────────────
STAG="${TAG}_smoke"
rm -rf "$SIMS/$STAG"
(
  env_arm; export GO_CONCURRENCY=1
  echo "[smoke $(date +%H:%M:%S)] $SMOKE_TASKS nt=1 arm=$ARM"
  t2_launch "$STAG" "$AB_PORT" "$SMOKE_TASKS" 1 2>&1 | tee "$LOG/$STAG.log"
) > "$LOG/${STAG}_driver.log" 2>&1

SZ=$(grep -ac "operand-size " "$LOG/$STAG.log" 2>/dev/null); SZ=${SZ:-0}
KIND=$(grep -ac "atm_withdrawal=" "$LOG/$STAG.log" 2>/dev/null); KIND=${KIND:-0}
RC=$(grep -ac "\[T2_SG_ROW_COUNT\]" "$LOG/$STAG.log" 2>/dev/null); RC=${RC:-0}
SIL=$(grep -ac "T2_ACTIONREQ\] 침묵" "$LOG/$STAG.log" 2>/dev/null); SIL=${SIL:-0}
EXC=$(grep -ac "grounded 검사 건너뜀\|대체 템플릿 미선언" "$LOG/$STAG.log" 2>/dev/null); EXC=${EXC:-0}
TB=$(grep -ac "Traceback" "$LOG/$STAG.log" 2>/dev/null); TB=${TB:-0}
echo "[$TAG] 스모크($ARM): 비교기 도달=$SZ · 종류계수=$KIND · ROW_COUNT 발화=$RC · 침묵=$SIL · 예외=$EXC · Traceback=$TB"
if [ "$TB" -ne 0 ]; then echo "[$TAG] ⛔Traceback — 중단."; exit 1; fi
if [ "$EXC" -ne 0 ]; then echo "[$TAG] ⛔술어가 예외로 죽었다 — 중단."; exit 1; fi
# ★2026-08-28 01:46 수리 — 1차 게이트가 `atm_withdrawal=` 를 **무조건** 요구해 t7371 을 세웠다.
#   그 인쇄는 **비교기에 닿아야만** 난다 = 모델 거동에 달린 조건부인데 나는 그것을 배선 검정으로
#   썼다. 이 파일 머리에 *"발화 자체는 게이트가 아니다"* 라고 적어 놓고 같은 실수를 했다.
#   ⇒ **닿았는데 안 찍혔을 때만** 배선 결함이다. 안 닿았으면 이 스모크로는 못 재는 것이고 그건
#     중단 사유가 아니다 — 배선은 `test_sg_row_count`(배터리)가 오프라인으로 이미 증명한다.
if [ "$SZ" -gt 0 ] && [ "$KIND" -eq 0 ]; then
  echo "[$TAG] ⛔비교기에 닿았는데 종류 계수가 안 찍혔다 = 배선 결함. 중단."; exit 1
fi
if [ "$SZ" -eq 0 ]; then
  echo "[$TAG] (스모크가 비교기에 안 닿았다 — 이 계기는 못 쟀다. 배선 증명은 배터리가 한다.)"
fi
# ★대조 팔의 게이트는 **꺼져 있음의 증명**이다 — 켜진 팔에서만 서는 것이 안 서야 한다.
if [ "$ARM" = "control" ] && [ "$RC" -ne 0 ]; then
  echo "[$TAG] ⛔대조인데 ROW_COUNT 가 발화했다 = 팔이 안 갈렸다. 중단."; exit 1
fi
if [ "$ARM" = "control" ] && [ "$SIL" -ne 0 ]; then
  echo "[$TAG] ⛔대조인데 ACTIONREQ 침묵이 발화했다 = 팔이 안 갈렸다. 중단."; exit 1
fi

# ── ② 본런 ────────────────────────────────────────────────────────────────
cd "$REPO"
/home/woori/venvs/seka_env/bin/python reports/facet_rft_2026/freeze.py --on \
  --tag "$TAG" --reason "night A/B $ARM: grounded pointer + row count" || true
cd "$REPO/scripts/distill/tau2"
(
  env_arm; export GO_CONCURRENCY="$CONC"
  echo "[main $(date +%H:%M:%S)] $TASKS nt=$NT arm=$ARM"
  t2_launch "$TAG" "$AB_PORT" "$TASKS" "$NT" 2>&1 | tee "$LOG/$TAG.log"
  echo "[main $(date +%H:%M:%S)] done"
) > "$LOG/${TAG}_driver.log" 2>&1
cd "$REPO"
/home/woori/venvs/seka_env/bin/python reports/facet_rft_2026/freeze.py --off --tag "$TAG"

# ── ③ 회수 ([[30]]: gzip 만으론 영속이 아니다) ────────────────────────────
mkdir -p reports/facet_rft_2026/sim_results
for T in "$STAG" "$TAG"; do
  gzip -c "$SIMS/$T/results.json" > reports/facet_rft_2026/sim_results/$T.results.json.gz 2>/dev/null || true
  gzip -c "$LOG/$T.log" > reports/facet_rft_2026/sim_results/$T.log.gz 2>/dev/null || true
  for S in fb trace; do
    F="$LOG/${S}_${T}.jsonl"; [ -s "$F" ] && gzip -c "$F" > reports/facet_rft_2026/sim_results/${S}_${T}.jsonl.gz
  done
done
git add -f reports/facet_rft_2026/sim_results/*${TAG}* 2>/dev/null || true
git -c user.name=ghlee -c user.email=beingrelative@gmail.com \
  commit -q -m "$TAG ($ARM): grounded pointer and row count, same-sha A/B" \
  -- reports/facet_rft_2026/sim_results || true
# ★두 팔이 **동시에** 밀므로 non-fast-forward 가 정상 경로다. rebase 후 재시도를 여러 번 한다.
PUSHED=0
for i in 1 2 3 4 5; do
  if git push -q origin facet-rft-2026 2>/dev/null; then PUSHED=1; break; fi
  echo "[$TAG] push 실패 $i — rebase 후 재시도"
  git -c user.name=ghlee -c user.email=beingrelative@gmail.com pull --rebase -q origin facet-rft-2026 || true
  sleep 20
done
[ "$PUSHED" -eq 0 ] && echo "[$TAG] ⛔push 5회 실패 — 결과가 이 디스크에만 있다."
echo "=== TRACKED ==="
for F in reports/facet_rft_2026/sim_results/*${TAG}*; do
  git ls-files --error-unmatch "$F" >/dev/null 2>&1 && echo "  ok   $(basename $F)" || echo "  ⛔UNTRACKED $(basename $F)"
done

echo "=== 성적 · 부호표 ($ARM) ==="
TAG="$TAG" ARM="$ARM" /home/woori/venvs/seka_env/bin/python - <<'PY'
# -*- coding: utf-8 -*-
import os, sys, collections
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
print("  [%s] pass %d/%d" % (os.environ["ARM"],
                             sum(1 for v in per.values() for r in v if r == 1.0), len(sims)))
for t in sorted(per):
    print("    %-10s %s" % (t, per[t]))
for s in sims:
    d = F.mutation_diff(s, mut, tag=tag) or {}
    gap = sum(len(d.get(k) or ()) for k in ("missing", "wrongarg", "extra", "dup"))
    print("    %-20s reward=%-5s gap=%-3d %s"
          % (F.simtag(s), (s.get("reward_info") or {}).get("reward"), gap, F.term_reason(s)))
print("  종료사유: %s" % dict(collections.Counter(F.term_reason(s) for s in sims)))
PY

echo "=== 레버 자국 ==="
for M in "\[T2_SG_ROW_COUNT\]" "This audit is INCOMPLETE" "T2_ACTIONREQ\] 침묵" "atm_withdrawal=" "Traceback"; do
  C=$(grep -ac "$M" "$LOG/$TAG.log" 2>/dev/null); echo "  $M = ${C:-0}"
done
echo "--- 도구가 건넨 총액 ---"
grep -ao "computed by this tool, is [-0-9.]*" "$LOG/$TAG.log" 2>/dev/null | sort | uniq -c
echo "[$TAG] DONE $(date +%H:%M:%S)"
exit 0
