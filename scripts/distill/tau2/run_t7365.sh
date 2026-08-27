#!/usr/bin/env bash
# t7365 — 016 단독 (사용자 지시 2026-08-27: "pass 닫는데 집중하라. 한 태스크씩 닫아라. 016부터")
#
# ⛔`set -u` 금지 (t7360 교훈).
#
# ── 무엇이 t7364 와 다른가 — **한 칸뿐이다** ───────────
#   `ledger_metrics[].diagnose_prompt` 의 물음. 구판은 *"One of these records did not pay out"*
#   이었는데 그 문장은 **같은 블록에 실려 나가는 정의와 겹친다**:
#     `COMPLETE — the referred person has … met the criteria to get the referral bonus`
#   그 정의 아래에서 *'못 받은 것'* 은 COMPLETE 로도 읽힌다 ⇒ 서브가 Platinum/Bronze 를 냈고,
#   그 답이 *"A separate check was run … It answers: X"* 로 나가 뒤따르는 발화를 그 축에 묶었다.
#   격리 `x566`(8140·4팔×5×2태그):
#       구판 그대로        Platinum 3/5   (라이브 오지목 재현)
#       행 단위로만 바꿈   Platinum 5/5   ⇒ 단위는 원인이 아니었다
#       정의의 어휘로 물음 **Silver 10/10**  ← 새 선언
#       부정통제          무응답 5/5
#   함께: `T2_DIAG_UNAMBIGUOUS=0`. 그 침묵의 근거(*"어느 이름도 단일 상태가 아니다"*)는
#   원장의 성질이 아니라 **구판 물음이 만든 모호**였다. 이 코퍼스에서 원장이 있는 sim 은 016
#   뿐(9개)이고 침묵은 그 9개 전부에서 발동했다 ⇒ 끄는 폭발 반경도 016 뿐이다.
#
# ── ⛔싣지 않는 것 ───────────────────────────────────
#   `T2_ARG_LABEL=0` — 오늘 등재했지만 **016 에서는 발화하지 않는다**(016 의 값은 `user_id`
#     자리에 `user_id` 로 들어간다·`ctx-user` 분류). 단일 변수를 지키려고 뺀다.
#   `T2_REARM_USER_ONLY`·`PROCEDURE_LEFT`·`EPLAN_ENUM_SUBTRACT`·`SCOPE_ALL`·`READ_PER_ENTITY` = 0
#
# ── 표적·대조·판정 ───────────────────────────────────
#   task_016 × nt4 = 4 sim · 대조 = t7364(016 **0/4**) · t7363(016 0/2).
#   gap 은 **1**(`MISSING submit_transaction`) — hard-0 에서 제일 얕다.
#   판정 = **016 의 0→1**. ⛔총점 Δ 금지(C594).
#   ⚠남은 사슬: 서브가 Silver 를 내도 에이전트가 그 이름에 **$750** 을 붙여 말해야 손님이 찍는다.
#     라이브는 그 금액을 이미 말한다(t7363 msg[44]) — 다만 Bronze 에 붙여서. 그 붙임이 옮겨가는지가
#     이 런이 재는 것이다.
#
# ── 발사 게이트 ──────────────────────────────────────
#   ① 배터리 8종 exit 0
#   ② 스모크에서 **새 물음이 실제로 나가고**(`[T2_DIAG]` 배달 자국) Traceback 0
set -o pipefail
REPO=/home/woori/workspace_common/boltzmann-attention-pi
LOG=/home/woori/scratch/logs
SIMS=/home/woori/scratch/tau2-bench/data/simulations
STAMP=20260827
TASKS="task_016"
SMOKE_TASKS="task_016"
cd "$REPO/scripts/distill/tau2"

echo "[t7365 $(date +%H:%M:%S)] === 발사 전 배터리 ==="
BAD=0
for t in test_procedure_left test_actionreq_grounded test_operator_find_executed \
         test_decision_point_load test_diag_unambiguous \
         test_read_per_entity test_flag_registry \
         test_arg_label test_a2_three_layer; do
  PYTHONPATH=. PYTHONIOENCODING=utf-8 /home/woori/venvs/seka_env/bin/python $t.py >/dev/null 2>&1
  rc=$?; echo "  $t exit=$rc"; [ $rc -ne 0 ] && BAD=1
done
if [ $BAD -ne 0 ]; then echo "[t7365] ⛔배터리 붉음 — 발사하지 않는다"; exit 1; fi

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
  export T2_REARM_USER_ONLY=0 T2_PROCEDURE_LEFT=0 T2_EPLAN_ENUM_SUBTRACT=0 T2_SCOPE_ALL=0
  export GO_MAX_STEPS=150
}

# ── ① 스모크 — **레버가 나는지**만 본다(성적 아님) ────────────────────────
STAG="bank_t7365_smoke_${STAMP}"
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
DIAG=$(grep -ac "\[T2_DIAG\] raw=" "$LOG/$STAG.log" 2>/dev/null); DIAG=${DIAG:-0}
CRIT=$(grep -ao "\[T2_DIAG\] raw=.\{0,60\}" "$LOG/$STAG.log" 2>/dev/null | head -2)
TB=$(grep -ac "Traceback" "$LOG/$STAG.log" 2>/dev/null); TB=${TB:-0}
echo "[t7365] 스모크: 진단 발화=$DIAG · Traceback=$TB"
echo "  답: $CRIT"
if [ "$DIAG" -eq 0 ]; then
  echo "[t7365] ⛔진단이 한 번도 배달되지 않았다 — 침묵이 안 풀렸거나 배선이 죽었다. 중단."
  exit 1
fi
if [ "$TB" -ne 0 ]; then echo "[t7365] ⛔Traceback $TB — 중단."; exit 1; fi

# ── ② 본런 ────────────────────────────────────────────────────────────────
cd "$REPO"
/home/woori/venvs/seka_env/bin/python reports/facet_rft_2026/freeze.py --on \
  --tag "bank_t7365_$STAMP" --reason "t7365 016 alone: the diagnosis question, reworded, against t7364" || true
cd "$REPO/scripts/distill/tau2"

TAG="bank_t7365_hard0_${STAMP}"
(
  env_arm; export GO_CONCURRENCY=3   # t7362 의 3-동시와 같은 서버 부하 ⇒ 지연 조건 비교 가능
  echo "[main $(date +%H:%M:%S)] $TASKS nt=4"
  t2_launch "$TAG" 8140 "$TASKS" 4 2>&1 | tee "$LOG/$TAG.log"
  echo "[main $(date +%H:%M:%S)] done"
) > "$LOG/${TAG}_driver.log" 2>&1

cd "$REPO"
/home/woori/venvs/seka_env/bin/python reports/facet_rft_2026/freeze.py --off

# ── ③ 회수 ([[30]]: gzip 만으론 영속이 아니다) ────────────────────────────
#   ⚠t7362 의 실물 결함 둘을 여기서 고친다:
#     ⒜ `git add bank_t7365_*` 는 **`fb_`·`trace_` 접두를 못 잡는다** — 사이드카 6 파일이
#        그래서 add 조차 안 됐다. 글롭을 `*bank_t7365_*` 로 바꾼다.
#     ⒝ push 실패를 `|| echo` 로 삼켜 결과가 디스크 한 곳에만 남았다 — **exit 1** 로 죽는다.
mkdir -p reports/facet_rft_2026/sim_results
for T in "$STAG" "$TAG"; do
  gzip -c "$SIMS/$T/results.json" > reports/facet_rft_2026/sim_results/$T.results.json.gz 2>/dev/null || true
  gzip -c "$LOG/$T.log" > reports/facet_rft_2026/sim_results/$T.log.gz 2>/dev/null || true
  for S in fb trace; do
    F="$LOG/${S}_${T}.jsonl"; [ -s "$F" ] && gzip -c "$F" > reports/facet_rft_2026/sim_results/${S}_${T}.jsonl.gz
  done
done
git add -f reports/facet_rft_2026/sim_results/*bank_t7365_* 2>/dev/null || true
git -c user.name=ghlee -c user.email=beingrelative@gmail.com \
  commit -q -m "t7365 hard-0: the per-subject read requirement, measured against t7363" \
  -- reports/facet_rft_2026/sim_results || true
if ! git push -q origin facet-rft-2026; then
  echo "[t7365] ⛔push 실패 — 결과가 이 디스크에만 있다. 손으로 회수하라([[30]] 1순위 사고)."
  PUSH_FAIL=1
fi
echo "=== TRACKED ==="
for F in reports/facet_rft_2026/sim_results/*bank_t7365_*; do
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
echo "[t7365] DONE $(date +%H:%M:%S)"
[ -n "$PUSH_FAIL" ] && exit 1
exit 0
