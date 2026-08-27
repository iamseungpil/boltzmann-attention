#!/usr/bin/env bash
# t7364 — hard-0 런 (사용자 승인 2026-08-27)
#
# ⛔`set -u` 금지 (t7360 교훈: go_stack 의 `t2_require_key` 가 미설정을 스스로 처리하는데
#   `set -u` 아래선 그 표현식이 먼저 죽어 t2_launch 가 한 줄도 안 돈다).
#
# ── 무엇이 t7363 과 다른가 ────────────────────────────
#   **`T2_READ_PER_ENTITY=1` 하나뿐이다.** 나머지 레버 집합은 t7363 과 **바이트 동일**이라
#   차이가 이 술어에 귀속된다(대조 = t7363 0/20 · 같은 10 태스크 · 같은 서버 부하).
#
#   결함: 선행 read 요건의 충족 판정이 **도구 이름만** 봤다 — 다른 주체로 돈 read 가 요건을
#   영구히 닫는다. 016 실측(t7363·t7356 두 세대): 계좌 read 는 손님 자신으로만 돌았고 손님이
#   묻는 **친구**로는 끝내 안 돌았다. 원장 15행 중 어느 행이 그 친구 것인지 아는 유일한 경로다.
#   ⇒ 충족을 **주체별로** 본다. 인자 키·값만 비교한다([[59]]·[[22]]).
#
#   격리(`x561`·8140·3팔×4): A_asis **0/4**(라이브 축자 재현 — 거래 read 로 샌다) ·
#     B_demand **4/4**(`get_all_user_accounts_by_user_id_3847{user_id: friend_…}`) · N_len **0/4**.
#   부호표(`x560`·t7363·t7356 채점 33 sim): 발화 **7(21%)** — 016 2·072 1·074 1·085 3,
#     **전부 reward 0** ⇒ 이 코퍼스에서 손실 불가. 판 것 = 발화 sim 당 read 한 턴.
#
# ── ⛔무엇을 **뺐는가** (t7363 과 동일) ───────────────
#   `T2_REARM_USER_ONLY`(A-3′) 0 — 래칫이 리모트 코퍼스에서 붉다(미영속 로그 회수 전엔 안 싣는다).
#   `T2_PROCEDURE_LEFT`·`T2_EPLAN_ENUM_SUBTRACT`·`T2_SCOPE_ALL` 0 — 각각 미측정·미측정·음성 확정.
#
# ── 표적·대조 ────────────────────────────────────────
#   hard-0 10 × nt2 = 20 sim · 대조 = **t7363**(0/20).
#   판정 = **표적의 0→1**. ⛔총점 Δ 로 판정하지 마라(C594).
#   ⚠016 의 gap 은 1 이지만 이 레버가 사는 것은 **친구 카드 종류를 얻는 한 단계**다 —
#     그 뒤 Silver·IN_PROGRESS 행과 $750 을 잇는 것은 모델 몫이라 **필요조건이지 충분조건이 아니다**.
#
# ── 발사 게이트 ──────────────────────────────────────
#   ① 배터리 7종 exit 0
#   ② 스모크에서 **술어가 실제로 평가된다**(`checked=` 자국 ≥1). ⛔`gap=` 이 비는 것으로는
#      중단하지 않는다 — 술어는 모델이 그 주체를 인자에 넣은 sim 에서만 서고 그것은 5 중 2 다.
#      자국이 **0** 이면 배선이 죽은 것이므로 그때는 중단한다(t7362 교훈).
set -o pipefail
REPO=/home/woori/workspace_common/boltzmann-attention-pi
LOG=/home/woori/scratch/logs
SIMS=/home/woori/scratch/tau2-bench/data/simulations
STAMP=20260827
TASKS="task_016,task_040,task_055,task_057,task_063,task_072,task_074,task_079,task_085,task_094"
SMOKE_TASKS="task_016,task_072"
cd "$REPO/scripts/distill/tau2"

echo "[t7364 $(date +%H:%M:%S)] === 발사 전 배터리 ==="
BAD=0
for t in test_procedure_left test_actionreq_grounded test_operator_find_executed \
         test_decision_point_load test_diag_unambiguous \
         test_read_per_entity test_flag_registry; do
  PYTHONPATH=. PYTHONIOENCODING=utf-8 /home/woori/venvs/seka_env/bin/python $t.py >/dev/null 2>&1
  rc=$?; echo "  $t exit=$rc"; [ $rc -ne 0 ] && BAD=1
done
if [ $BAD -ne 0 ]; then echo "[t7364] ⛔배터리 붉음 — 발사하지 않는다"; exit 1; fi

env_arm() {
  source ./go_stack.sh >/dev/null 2>&1
  # t7361/t7362 와 **같은 레버 집합** — 판정을 이어 붙이려고 노브를 더 바꾸지 않는다.
  export T2_ACTION_SUB=1 T2_KEEP_DENY_BODY=1 T2_CALL_FORM=1 T2_ARG_EMPTY=1 T2_SEARCH_AGENT=1
  export T2_DECIDE_ANY=1 T2_WRITE_ARG_ENUM=1 T2_ARG_DOC_SUB=1 T2_VALUE_FORMULA=full
  export T2_SG_DOCS=1 T2_SG_PROMPT_V2=1 T2_SPEC_AT_WRITE=1 T2_WRITE_ARG_TYPE=1
  export T2_RULE_AT_WRITE=1 T2_DUP_WRITE=1
  # ★이 런이 재는 것. go_stack 이 이미 =1 로 내보내지만 런 기록에 남기려고 명시한다.
  export T2_READ_PER_ENTITY=1
  # ⛔이 런이 **싣지 않는** 것들 (위 주석 참조)
  export T2_REARM_USER_ONLY=0 T2_PROCEDURE_LEFT=0 T2_EPLAN_ENUM_SUBTRACT=0 T2_SCOPE_ALL=0
  export GO_MAX_STEPS=150
}

# ── ① 스모크 — **레버가 나는지**만 본다(성적 아님) ────────────────────────
STAG="bank_t7364_smoke_${STAMP}"
(
  env_arm; export GO_CONCURRENCY=2
  echo "[smoke $(date +%H:%M:%S)] $SMOKE_TASKS nt=1"
  t2_launch "$STAG" 8140 "$SMOKE_TASKS" 1 2>&1 | tee "$LOG/$STAG.log"
) > "$LOG/${STAG}_driver.log" 2>&1

CHK=$(grep -ac "T2_READ_PER_ENTITY] checked=" "$LOG/$STAG.log" 2>/dev/null); CHK=${CHK:-0}
RAISED=$(grep -a "T2_READ_PER_ENTITY] checked=" "$LOG/$STAG.log" 2>/dev/null | grep -avc "gap=none"); RAISED=${RAISED:-0}
DIAG=$(grep -ac "T2_DIAG] 모호" "$LOG/$STAG.log" 2>/dev/null); DIAG=${DIAG:-0}
TB=$(grep -ac "Traceback" "$LOG/$STAG.log" 2>/dev/null); TB=${TB:-0}
echo "[t7364] 스모크: READ_PER_ENTITY 평가=$CHK (그중 주체 지목=$RAISED) · T2_DIAG=$DIAG · Traceback=$TB"
if [ "$CHK" -eq 0 ]; then
  echo "[t7364] ⛔술어가 **한 번도 평가되지 않았다** — 배선이 죽었다. 중단."
  exit 1
fi
[ "$RAISED" -eq 0 ] && echo "[t7364] ⓘ스모크에서 주체 지목 0 — 정상 범위다(5 중 2 형상). 본런으로 간다."
if [ "$TB" -ne 0 ]; then echo "[t7364] ⛔Traceback $TB — 중단."; exit 1; fi

# ── ② 본런 ────────────────────────────────────────────────────────────────
cd "$REPO"
/home/woori/venvs/seka_env/bin/python reports/facet_rft_2026/freeze.py --on \
  --tag "bank_t7364_$STAMP" --reason "t7364 hard-0: T2_READ_PER_ENTITY only, against t7363" || true
cd "$REPO/scripts/distill/tau2"

TAG="bank_t7364_hard0_${STAMP}"
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
#     ⒜ `git add bank_t7364_*` 는 **`fb_`·`trace_` 접두를 못 잡는다** — 사이드카 6 파일이
#        그래서 add 조차 안 됐다. 글롭을 `*bank_t7364_*` 로 바꾼다.
#     ⒝ push 실패를 `|| echo` 로 삼켜 결과가 디스크 한 곳에만 남았다 — **exit 1** 로 죽는다.
mkdir -p reports/facet_rft_2026/sim_results
for T in "$STAG" "$TAG"; do
  gzip -c "$SIMS/$T/results.json" > reports/facet_rft_2026/sim_results/$T.results.json.gz 2>/dev/null || true
  gzip -c "$LOG/$T.log" > reports/facet_rft_2026/sim_results/$T.log.gz 2>/dev/null || true
  for S in fb trace; do
    F="$LOG/${S}_${T}.jsonl"; [ -s "$F" ] && gzip -c "$F" > reports/facet_rft_2026/sim_results/${S}_${T}.jsonl.gz
  done
done
git add -f reports/facet_rft_2026/sim_results/*bank_t7364_* 2>/dev/null || true
git -c user.name=ghlee -c user.email=beingrelative@gmail.com \
  commit -q -m "t7364 hard-0: the per-subject read requirement, measured against t7363" \
  -- reports/facet_rft_2026/sim_results || true
if ! git push -q origin facet-rft-2026; then
  echo "[t7364] ⛔push 실패 — 결과가 이 디스크에만 있다. 손으로 회수하라([[30]] 1순위 사고)."
  PUSH_FAIL=1
fi
echo "=== TRACKED ==="
for F in reports/facet_rft_2026/sim_results/*bank_t7364_*; do
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
for M in "T2_READ_PER_ENTITY] checked=" "T2_DIAG] 모호" "OPERATOR-DIRECT" "Traceback"; do
  C=$(grep -ac "$M" "$LOG/$TAG.log" 2>/dev/null); echo "  $M = ${C:-0}"
done
echo "[t7364] DONE $(date +%H:%M:%S)"
[ -n "$PUSH_FAIL" ] && exit 1
exit 0
