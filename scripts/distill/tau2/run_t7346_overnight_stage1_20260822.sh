#!/bin/bash
# t7346 — ★**밤샘 본런**(사용자 지시 2026-08-22 축자: *"c 밤샘런 가라"*). 20 태스크 × nt2 = 40 sim.
#
# ## t7345 와 무엇이 다른가 — **엔진은 같다. 이건 t7345 의 완주판이다.**
#   t7345 는 2/20 에서 멈췄다(그 세션 여덟 번 발사·여덟 번 중단). 그 뒤 오늘 커밋된 것은
#   프로브 스크립트(x484·x485)와 A3 선언 `enum_priority` 하나뿐이고, **그 선언은 어떤 엔진
#   코드도 읽지 않는다**(grep 확인: 참조처는 x485 프로브뿐·A2 를 통째로 프롬프트에 덤프하는
#   자리 없음) ⇒ 라이브 거동 불변. 엔진 경로 diff = **0**.
#
# ## 오늘의 포렌식이 이 런에 붙이는 단서 (원장 C597~C599)
#   ★**004 는 회귀가 아니다 — 기저율 22/71 ≈ 31% 의 불안정 태스크**(C599·부정통제 n=6:
#     docoff2 2/6 ↔ base2 3/6). 어제 지목한 `T2_REQUIRE_DOC_DELIVER` 는 **무죄**다.
#     ⇒ 이 런에서 004 가 0 이든 1 이든 **그것만으로 회귀·수리를 주장하지 말 것**.
#   ★**004 의 성적은 `reason` enum 한 칸이 정한다**(C597·71 sim 전수 분리 예외 0):
#     `account_ownership_dispute` → PASS 22/22 · 다른 값 → FAIL 30/30 · 미실행 → FAIL 19.
#     경로(검색 횟수·순서)는 채점되지 않는다.
#   ★**그 칸은 전달로 안 닫힌다 — 경계로 기록됐다**(C598·격리 192 재생): 정의 표를 줘도 0/24,
#     티어 규칙을 더해도 0/24. 움직인 것은 답을 이름으로 대는 사례 열거뿐이고 그마저 2~5/24.
#     ⇒ **이 런의 004 실패를 새 레버로 고치려 들지 말 것**([[23]] 레버 없음).
#   ★**미실행(이관을 말만 하고 안 함)은 부하다**(C598ⓗ·격리 192/192 이 호출을 냈다·
#     라이브가 호출 0 이던 원천 포함) — 처방 축은 전달·부하 축소이지 학습·scale 이 아니다.
#
# ## 대조군 — t7336 **13/40**
#   로스터 20 · nt=2 · PIN · ON 이 t7336 러너와 **바이트 동일**하고 sha 만 다르다.
#   판정선 Δ ≥ 4/40. ⚠엄밀 A/B 아님([M]) — 묶음이 크다(수리 14 + 잔여 3 + 093 수리 5 = 22).
#   **묶음 Δ 로 개별 수리를 주장하지 않는다**(C594 실증). 귀속은 per-task 포렌식([[08]]).
#   채점은 **reward 뿐**([[69]]).
#
# ## 이번에 특히 볼 자리
#   ★093·094    apy 4.275 · interest 33.0 · WEV deny 0 이 **본런에서도** 재현되나
#   ★[T2_SG_REFRAW] 발화 수 — 원문 전달이 실제로 몇 번 살았나(0 이면 폴백만 이긴 것)
#   ★[T2_SG_ROUND] 발화 ↔ WEV deny 수(t7341 기준선 10)
#   073·050·085·040#1  수리 묶음 표적 · F8 부활(t7336 = 0)
#   004  reason 값을 로그에서 **그대로** 기록만 할 것(C597 축과 대조·수리 금지)
#
# ## 스모크 게이트 ([[30]] 死배선에 돈 금지) — t7345 와 동일, 하나도 빼지 않았다
set -e
REPO=/home/woori/workspace_common/boltzmann-attention-pi
LOG=/home/woori/scratch/logs
SIMS=/home/woori/scratch/tau2-bench/data/simulations
mkdir -p "$LOG"
cd "$REPO/scripts/distill/tau2"
SHA=$(cd "$REPO" && git rev-parse --short HEAD)

DIRTY=$(cd "$REPO" && git status --porcelain -- \
  scripts/distill/tau2/t2_gate_patch.py scripts/distill/tau2/t2_search.py \
  scripts/distill/tau2/t2_resolve.py scripts/distill/tau2/t2_scaffold_get.py \
  scripts/distill/tau2/a2/ | grep -cv '^??' || true)
if [ "$DIRTY" != "0" ]; then
  echo "[t7346] REFUSING: 엔진 경로 커밋 안 된 변경 $DIRTY 개." >&2; exit 1
fi

for t in test_a2_three_layer.py test_flag_registry.py test_claim_verify.py \
         test_claim_tool_index.py test_read_routine.py test_proc_read_connect.py \
         test_verdict_gate.py test_verdict_carry.py test_no_undefined_names.py \
         test_no_unbound_a2.py test_quote_in.py test_args_equal.py test_t2_procedure.py \
         test_proc_absent_wiring.py test_pin_read_replay.py test_eplan.py \
         test_decision_carry.py test_route_trace.py test_group_parse.py \
         test_resolve_cap_runtime.py test_byref_repairs.py test_no_prose_regex.py \
         test_compute_params.py test_sg_docs_delivery.py test_sg_src0_axis.py \
         test_sg_fetch_iso.py test_sg_isofb.py \
         test_t7337_residual_debt.py test_t7336_g1_our_layer.py \
         test_t7336_g2_gate_axis.py test_write_arg_enum.py \
         test_a2_answer_format_placeholder.py test_result_round.py \
         test_apy_balance_tier.py test_ref_from_outputs.py; do
  [ -f "$t" ] || continue
  PYTHONPATH=/home/woori/scratch/tau2-bench/src timeout 90 \
    /home/woori/venvs/seka_env/bin/python "$t" >/dev/null 2>&1 \
    || { echo "[t7346] REFUSING: $t FAIL" >&2; exit 1; }
done
echo "[t7346] VERIFY OK (배터리 31 - 오늘 수리 검정 5 등재·문법 포함)"

if pgrep -f "[t]2_launch" >/dev/null; then
  echo "[t7346] REFUSING: 다른 라이브 런이 돌고 있다" >&2; exit 1
fi
for f in "$LOG"/bank_t7346_*.log; do
  [ -e "$f" ] && { echo "[t7346] REFUSING: $f 존재" >&2; exit 1; }
done
for d in "$SIMS"/bank_t7346_*; do
  [ -e "$d" ] && { echo "[t7346] REFUSING: $d 잔존" >&2; exit 1; }
done

# t7328/t7333 과 같은 PIN — 0 항목 = 측정이 기각/보류한 노브(위 머리말)
PIN="T2_ACTION_SUB=1 T2_KEEP_DENY_BODY=1 T2_CALL_FORM=1 T2_ARG_EMPTY=1 T2_SEARCH_AGENT=1 \
T2_DECIDE_ANY=1 T2_WRITE_ARG_ENUM=1 T2_DECIDE_BEFORE_WRITE=1 T2_DECISION_CARRY=1 \
T2_DISCOVERY_STEP2=1 T2_ARG_AXIS=1 T2_WRITE_SUB=3 T2_ACTION_INDEX=1 T2_NOW_SELFCALL=1 \
T2_SEARCH_ON_PROCEED=1 T2_ACT_DEMAND=0 T2_DELIVER_PRECOMMIT=0 T2_PROCEED_DOCBODY=0 \
T2_DOCS_AT_WRITE=0 T2_SUB_REQUIREMENT=0 T2_HANDOFF_PREDICATE=0 T2_PENDING_DISCOVERED=0 \
T2_VERDICT_CARRY=0 T2_ELIG_LINE=0 T2_VERDICT_GATE=0 T2_CLAIM_VERIFY=0 \
T2_DECLFIRST=0 T2_DECLFIRST_GUIDE_FIX=0 T2_CATEGORY_CITE="
ON="T2_ARG_DOC_SUB=1 T2_VALUE_FORMULA=full T2_SG_DOCS=1"

HALF_A='task_003,task_004,task_017,task_024,task_055,task_072,task_073,task_093,task_094,task_100'
HALF_B='task_016,task_033,task_040,task_050,task_057,task_063,task_074,task_079,task_085,task_098'
NT=2
M_VAL='documented_return_for_stated_spend'
M_DOCS='T2_SG_DOCS'
APYTOOL='get_correct_savings_apy'

echo "{\"tag\":\"t7346\",\"sha\":\"$SHA\",\"design\":\"overnight completion of t7345 - single all-on composed stack, no arms; engine diff vs t7345 is zero (only probe scripts and one inert A3 declaration landed since)\",\"on\":\"$ON\",\"tasks\":\"stage1 20 x nt=$NT = 40 sims\",\"endpoint\":\"reward only\",\"reference\":\"t7336 13/40 - same runner lineage, roster, PIN and ON; engine sha differs by 22 items (14 repair bundle + 3 residual debts + 5 fixes measured in isolation)\",\"bar\":\"delta >= 4/40 vs reference; per-task sign table\",\"caveat\":\"task_004 is a 31% coin flip (22/71) and its reason-enum axis is recorded as a boundary (C597-C599) - do not read a single 004 outcome as a regression or repair it with a new lever\"}" \
  | tee "$LOG/bank_t7346.meta.json"

setsid bash -c "
  cd '$REPO/scripts/distill/tau2'
  source ./go_stack.sh >/dev/null 2>&1
  export $PIN
  export $ON
  export GO_MAX_STEPS=150 GO_CONCURRENCY=1

  # ── 스모크 (2 sim · **두 GPU 병렬**) ───────────────────────────────────
  # ★2026-08-22 사용자 지시(축자): 스모크에도 2개 gpu 다 사용하라. 각각 1개 gpu 사용하라. 시간줄여라.
  #   구판은 8141 하나로 두 태스크를 **순차** 실행해 스모크가 본런만큼 길어졌다(093 34분 + 024 13분).
  #   두 팔은 GPU 가 다르므로 병렬이어도 vLLM prefix 캐시를 서로 밀어내지 않는다 — 본런의
  #   halfA/halfB 와 **같은 패턴**이고, 러너 안 동시성(GO_CONCURRENCY=1)은 그대로다([[30]] 동시성 지시).
  #   ⇒ 스모크 벽시계 = max(093, 024) ≈ 절반 이하.
  #   게이트 문구는 **합본 로그**에서 종전 그대로 평가한다(판정 로직 불변).
  SMK=bank_t7346_smoke_20260822
  SMKA=\${SMK}_a
  SMKB=\${SMK}_b
  echo '[t7346] === 스모크(093→8140 · 024→8141 · nt=1 · 병렬) ==='
  ( t2_launch \$SMKA 8140 task_093 1 ) > $LOG/\$SMKA.log 2>&1 &
  SPA=\$!
  ( t2_launch \$SMKB 8141 task_024 1 ) > $LOG/\$SMKB.log 2>&1 &
  SPB=\$!
  wait \$SPA \$SPB
  cat $LOG/\$SMKA.log $LOG/\$SMKB.log > $LOG/\$SMK.log
  NV=\$(grep -c '$M_VAL' $LOG/\$SMK.log 2>/dev/null); NV=\${NV:-0}
  ND=\$(grep -c '$M_DOCS' $LOG/\$SMK.log 2>/dev/null); ND=\${ND:-0}
  NC=\$(grep '$APYTOOL' $LOG/\$SMK.log 2>/dev/null | grep -v 'injected' | wc -l)
  echo \"[t7346] 스모크 발화 — 값주석=\$NV · docs전달=\$ND · apy도구 언급=\$NC\"
  grep '$M_DOCS' $LOG/\$SMK.log 2>/dev/null | head -4 || true
  cd '$REPO' && mkdir -p reports/facet_rft_2026/sim_results
  for _P in \$SMKA \$SMKB; do
    gzip -c '$SIMS/'\$_P'/results.json' > reports/facet_rft_2026/sim_results/\$_P.results.json.gz
  done
  gzip -c $LOG/\$SMK.log > reports/facet_rft_2026/sim_results/\$SMK.log.gz
  git add -f reports/facet_rft_2026/sim_results/\$SMK*.gz
  git -c user.name=ghlee -c user.email=beingrelative@gmail.com commit -q -m 't7346 smoke' || true
  git push -q origin facet-rft-2026 || true
  cd '$REPO/scripts/distill/tau2'
  if [ \"\$NV\" -eq 0 ]; then
    echo '[t7346] ⛔값 주석 발화 0 — 본런을 돌리지 않는다'
    exit 1
  fi
  if [ \"\$NC\" -gt 0 ] && [ \"\$ND\" -eq 0 ]; then
    echo '[t7346] ⛔apy 도구가 불렸는데 T2_SG_DOCS 발화 0 = 死배선 — 본런을 돌리지 않는다'
    exit 1
  fi
  [ \"\$NC\" -eq 0 ] && echo '[t7346] ⚠apy 도구 자체가 안 불림 — docs 게이트 판단 불가(관측으로 남긴다)'

  # ── 오늘 수리분: 관측 계수(게이트 아님) ────────────────────────────────
  NF8=\$(grep -c 'T2_ARG_PRODUCERS. fired' $LOG/\$SMK.log 2>/dev/null); NF8=\${NF8:-0}
  NSN=\$(grep -c 'T2_STALE_NOTE' $LOG/\$SMK.log 2>/dev/null); NSN=\${NSN:-0}
  NEN=\$(grep -c 'T2_WRITE_ARG_ENUM. deny' $LOG/\$SMK.log 2>/dev/null); NEN=\${NEN:-0}
  echo \"[t7346] 수리분 관측 - F8 발화=\$NF8 · STALE_NOTE=\$NSN · ENUM deny=\$NEN\"

  # ── ⛔死배선 게이트: 오늘 수리분이 예외로 조용히 죽으면 F8 이 또 침묵한다 ──
  NDEAD=\$(grep -cE 'arg-producer skipped|Traceback' $LOG/\$SMK.log 2>/dev/null); NDEAD=\${NDEAD:-0}
  if [ \"\$NDEAD\" -gt 0 ]; then
    echo '[t7346] ⛔오늘 수리분이 예외로 죽었다(arg-producer skipped / Traceback) - 본런을 돌리지 않는다'
    grep -E 'arg-producer skipped|Traceback' $LOG/\$SMK.log | head -5
    exit 1
  fi

  NTIER=\$(grep -c 'current_balance' $LOG/\$SMK.log 2>/dev/null); NTIER=\${NTIER:-0}
  echo \"[t7346] 잔액인자 발화=\$NTIER (0 이면 에이전트가 안 채운 것 - 종전 거동 폴백)\"
  NRD=\$(grep -c 'T2_SG_ROUND' $LOG/\$SMK.log 2>/dev/null); NRD=\${NRD:-0}
  NWEV=\$(grep -c 'T2_WRITE_EVIDENCE. deny' $LOG/\$SMK.log 2>/dev/null); NWEV=\${NWEV:-0}
  NM1=\$(grep -c 'principal=-1' $LOG/\$SMK.log 2>/dev/null); NM1=\${NM1:-0}
  echo \"[t7346] 반올림=\$NRD · WEV deny=\$NWEV (t7341=10) · 서브 -1 폐기=\$NM1 (미해결 표적)\"

  # ── ⛔자리표시자 수리 게이트: 서브가 예시 0.0 을 또 복사하면 중단 ──────────
  N00=\$(grep -c '부재(principal=0.0' $LOG/\$SMK.log 2>/dev/null); N00=\${N00:-0}
  echo \"[t7346] 격리 서브 0.0-복사 폐기 = \$N00 (0 이어야 수리가 먹은 것)\"
  if [ \"\$N00\" -gt 0 ]; then
    echo '[t7346] ⛔격리 서브가 answer_format 예시값(0.0)을 또 복사했다 - 자리표시자 수리 미적용 - 본런을 돌리지 않는다'
    grep '부재(principal=0.0' $LOG/\$SMK.log | head -3
    exit 1
  fi

  # ── ⛔문법 死배선 게이트: 격리 서브가 돌았는데 문법이 안 걸렸으면 중단 ──────
  NISO=\$(grep -c 'SG_ISOLATE. fetch' $LOG/\$SMK.log 2>/dev/null); NISO=\${NISO:-0}
  NSCH=\$(grep -c 'T2_SG_SCHEMA' $LOG/\$SMK.log 2>/dev/null); NSCH=\${NSCH:-0}
  echo \"[t7346] 격리 서브 fetch=\$NISO · 문법 적용=\$NSCH\"
  if [ \"\$NISO\" -gt 0 ] && [ \"\$NSCH\" -eq 0 ]; then
    echo '[t7346] ⛔격리 서브가 돌았는데 T2_SG_SCHEMA 발화 0 = 문법 死배선 - 본런을 돌리지 않는다'
    exit 1
  fi

  # ── ⛔누수 재발 게이트: 후보 명단에 General 이 실리면 중단 ────────────────
  if grep -q ', General ,' $LOG/\$SMK.log 2>/dev/null; then
    echo '[t7346] ⛔WRITE_ARG_ENUM 후보 명단에 General 재출현 - 본런을 돌리지 않는다'
    exit 1
  fi

  # ── 동결 ([[07]]·스모크 뒤) ────────────────────────────────────────────
  cd '$REPO'
  /home/woori/venvs/seka_env/bin/python reports/facet_rft_2026/freeze.py --on \\
    --tag t7346 --reason 'all-on composed stack, stage1 20 x nt2' || true
  cd '$REPO/scripts/distill/tau2'

  run_half() {
    NAME=\$1; PORT=\$2; TL=\$3
    TAG=bank_t7346_\${NAME}_20260822
    t2_launch \$TAG \$PORT \"\$TL\" $NT 2>&1 | tee $LOG/\$TAG.log
    echo \"[t7346] \$NAME 완료 · docs발화=\$(grep -c '$M_DOCS' $LOG/\$TAG.log 2>/dev/null || echo 0) · 값=\$(grep -c '$M_VAL' $LOG/\$TAG.log 2>/dev/null || echo 0)\"
    cd '$REPO' && mkdir -p reports/facet_rft_2026/sim_results
    gzip -c '$SIMS/'\$TAG'/results.json' > reports/facet_rft_2026/sim_results/\$TAG.results.json.gz
    gzip -c $LOG/\$TAG.log > reports/facet_rft_2026/sim_results/\$TAG.log.gz
    cd '$REPO/scripts/distill/tau2'
  }
  ( run_half halfA 8140 '$HALF_A' ) > $LOG/bank_t7346_halfA_chain.log 2>&1 &
  P1=\$!
  ( run_half halfB 8141 '$HALF_B' ) > $LOG/bank_t7346_halfB_chain.log 2>&1 &
  P2=\$!
  wait \$P1 \$P2

  # ── 영속 + 동결 해제 ([[30]] tracked 확인까지) ─────────────────────────
  cd '$REPO'
  cp $LOG/bank_t7346.meta.json reports/facet_rft_2026/sim_results/bank_t7346.meta.json || true
  git add -f reports/facet_rft_2026/sim_results/bank_t7346_*.gz reports/facet_rft_2026/sim_results/bank_t7346.meta.json
  git -c user.name=ghlee -c user.email=beingrelative@gmail.com commit -q -m 't7346 all-on stage1 results' || true
  git push -q origin facet-rft-2026 || true
  git ls-files --error-unmatch reports/facet_rft_2026/sim_results/bank_t7346_halfA_20260822.results.json.gz \\
    && echo '[t7346] persisted+tracked OK'
  /home/woori/venvs/seka_env/bin/python reports/facet_rft_2026/freeze.py --off --tag t7346 || true
  echo '[t7346] ALL DONE'
" </dev/null >"$LOG/bank_t7346_chain.log" 2>&1 &
echo "[t7346] 기동 PID=$! · sha=$SHA · 스모크 2 → 본런 40 sim (halfA 8140 · halfB 8141) · 로그 $LOG/bank_t7346_chain.log"
