#!/bin/bash
# t7343 — t7342 를 본런 초기(2/20)에 중단하고 **093 근인 수리까지 실어** 재발사한 런.
#   중단 이유: 사용자 지시로 093 을 정밀 포렌식했고(x480) 근인이 나왔다. 그것은 오늘 고친
#   셋(0.0 복사·문법·반올림)의 **상류**였다 — 그래서 증상만 바뀌고 reward 는 0.0 에 머물렀다.
#
# ## 근인 (x480 전수 대조·네 런 공통)
#   093 은 네 런 전부 같은 두 변이가 MISSING 이고 갈린 것은 **base APY 하나**다:
#     산출 expected_apy 2.775 ↔ 정답 4.275 (차이 1.5)
#   정책 문서 doc_savings_accounts_silver_account_003 축자:
#     | Below threshold       | Less than $10,000  | 2.5% |
#     | At or above threshold | At least $10,000   | 4.0% |
#   계좌 잔액 **144,000** ⇒ 문서상 base = 4.0. 검산도 맞는다:
#     144000 x (4.275 - 4.0)/100 / 12 = **33.0** = gold amount
#   그런데 서브의 REFERENCE 에는 **잔액이 없었고** 지시문은 base 가 하나인 것처럼 말했다
#   ("the account type's base APY") - 두 숫자 중 어느 쪽인지 판정할 근거가 **원리상 없었다**.
#
# ## 수리 전에 **격리로 쟀다** (x481 - [[62]] 0순위 - 사용자 지시: 프로브부터 재라)
#     A_asis      0/4    기준선
#     N_neg       0/4    부정통제([[57]]) - 반복해도 안 나온다
#     B_bal       0/4    ★잔액만 주면 **전혀** 안 된다
#     C_bal_hint  4/4    ★잔액 + 지시 한 문장이면 100%
#   ⇒ 결정적인 것은 재료가 아니라 **지시**였다. 프로브를 건너뛰고 잔액 전달만 했다면
#     아무 효과 없이 8시간을 태웠을 것이다. 이 프로브 하나가 그것을 막았다.
#
# ## 무엇이 실렸나 = **21건 한 묶음**
#   e7dcb97d  A1~A8·A10·A12~A16 (14건)
#   4373e7db  잔여 부채 3건 · d4a38ead 자리표시자 · 07c4c2f0 문법 · 079c1d93 반올림
#   fa181438  **잔액-티어 지시**(x481 C_bal_hint 그대로) - ref_params/params/ground + 지시 한 문장.
#             지시문에 임계·값 리터럴 **0**(문서의 표를 읽으라고만 한다) - 엔진은 고르지 않고
#             다른 계좌로도 전이된다([[05]]·[[62]]). gold 미참조([[23]]).
#
# ## 대조군 — t7336 **13/40** (변함없음)
#   로스터 20 태스크 · nt=2 · PIN · ON 이 t7336 러너와 **바이트 동일**하고 sha 만 다르다.
#   판정선 Δ ≥ 4/40. ⚠21건 묶음이므로 **묶음 Δ 로 개별 수리를 주장하지 않는다**(C594 실증).
#   귀속은 per-task 포렌식이 한다([[08]]). 채점은 reward 뿐([[69]]).
#
# ## 이번에 특히 볼 자리
#   ★093·094    expected_apy 가 **4.275** 로 나오나(격리 4/4 가 라이브에서 재현되나) ·
#               current_balance 인자가 실린 호출 수(에이전트가 안 채우면 종전 거동으로 폴백)
#   ★반올림     [T2_SG_ROUND] 발화 ↔ WEV deny 수(t7341 기준선 10) - t7342 스모크는 그 경로에
#               도달조차 안 해 **미검증**으로 남았다
#   ★서브 -1    부재(principal=-1 - 미수리 표적(원인 미확정이라 레버를 짓지 않았다·[[62]])
#   073·050·085·040#1  수리 묶음 표적
#
# ## 스모크 게이트 ([[30]] 死배선에 돈 금지)
#   task_093(격리·계산·티어 표적·8140) + task_024(값·배달 표적·8141) x nt=1 = 2 sim **병렬**.
#   ⛔중단: (1)값 주석 발화 0 (2)apy 불렸는데 SG_DOCS 0 (3)수리분 예외사망 (4)0.0 복사 재출현
#   ⛔      (5)격리 서브가 돌았는데 T2_SG_SCHEMA 발화 0(문법 死배선)
#   ⚠경고만: F8/STALE_NOTE/ROUND 발화 0 · 티어 지시가 실렸는데 base 가 안 바뀌는 경우
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
  echo "[t7343] REFUSING: 엔진 경로 커밋 안 된 변경 $DIRTY 개." >&2; exit 1
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
         test_apy_balance_tier.py; do
  [ -f "$t" ] || continue
  PYTHONPATH=/home/woori/scratch/tau2-bench/src timeout 90 \
    /home/woori/venvs/seka_env/bin/python "$t" >/dev/null 2>&1 \
    || { echo "[t7343] REFUSING: $t FAIL" >&2; exit 1; }
done
echo "[t7343] VERIFY OK (배터리 31 - 오늘 수리 검정 5 등재·문법 포함)"

if pgrep -f "[t]2_launch" >/dev/null; then
  echo "[t7343] REFUSING: 다른 라이브 런이 돌고 있다" >&2; exit 1
fi
for f in "$LOG"/bank_t7343_*.log; do
  [ -e "$f" ] && { echo "[t7343] REFUSING: $f 존재" >&2; exit 1; }
done
for d in "$SIMS"/bank_t7343_*; do
  [ -e "$d" ] && { echo "[t7343] REFUSING: $d 잔존" >&2; exit 1; }
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

echo "{\"tag\":\"t7343\",\"sha\":\"$SHA\",\"design\":\"single all-on composed stack (user directive), no arms; t7342 halted at 2/20 and relaunched with the balance-tier fix for 093 included\",\"on\":\"$ON\",\"tasks\":\"stage1 20 x nt=$NT = 40 sims\",\"endpoint\":\"reward only\",\"reference\":\"t7336 13/40 - same runner lineage, same roster, same PIN/ON; engine sha differs by the repair bundle (14), the residual debts (3), the answer_format placeholder fix (1), the isolate output grammar (1), the currency rounding fix (1) and the balance-tier instruction (1) = 21 items\",\"bar\":\"delta >= 4/40 vs reference; per-task sign table; reads/fabs/over-action logged\"}" \
  | tee "$LOG/bank_t7343.meta.json"

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
  SMK=bank_t7343_smoke_20260822
  SMKA=\${SMK}_a
  SMKB=\${SMK}_b
  echo '[t7343] === 스모크(093→8140 · 024→8141 · nt=1 · 병렬) ==='
  ( t2_launch \$SMKA 8140 task_093 1 ) > $LOG/\$SMKA.log 2>&1 &
  SPA=\$!
  ( t2_launch \$SMKB 8141 task_024 1 ) > $LOG/\$SMKB.log 2>&1 &
  SPB=\$!
  wait \$SPA \$SPB
  cat $LOG/\$SMKA.log $LOG/\$SMKB.log > $LOG/\$SMK.log
  NV=\$(grep -c '$M_VAL' $LOG/\$SMK.log 2>/dev/null); NV=\${NV:-0}
  ND=\$(grep -c '$M_DOCS' $LOG/\$SMK.log 2>/dev/null); ND=\${ND:-0}
  NC=\$(grep '$APYTOOL' $LOG/\$SMK.log 2>/dev/null | grep -v 'injected' | wc -l)
  echo \"[t7343] 스모크 발화 — 값주석=\$NV · docs전달=\$ND · apy도구 언급=\$NC\"
  grep '$M_DOCS' $LOG/\$SMK.log 2>/dev/null | head -4 || true
  cd '$REPO' && mkdir -p reports/facet_rft_2026/sim_results
  for _P in \$SMKA \$SMKB; do
    gzip -c '$SIMS/'\$_P'/results.json' > reports/facet_rft_2026/sim_results/\$_P.results.json.gz
  done
  gzip -c $LOG/\$SMK.log > reports/facet_rft_2026/sim_results/\$SMK.log.gz
  git add -f reports/facet_rft_2026/sim_results/\$SMK*.gz
  git -c user.name=ghlee -c user.email=beingrelative@gmail.com commit -q -m 't7343 smoke' || true
  git push -q origin facet-rft-2026 || true
  cd '$REPO/scripts/distill/tau2'
  if [ \"\$NV\" -eq 0 ]; then
    echo '[t7343] ⛔값 주석 발화 0 — 본런을 돌리지 않는다'
    exit 1
  fi
  if [ \"\$NC\" -gt 0 ] && [ \"\$ND\" -eq 0 ]; then
    echo '[t7343] ⛔apy 도구가 불렸는데 T2_SG_DOCS 발화 0 = 死배선 — 본런을 돌리지 않는다'
    exit 1
  fi
  [ \"\$NC\" -eq 0 ] && echo '[t7343] ⚠apy 도구 자체가 안 불림 — docs 게이트 판단 불가(관측으로 남긴다)'

  # ── 오늘 수리분: 관측 계수(게이트 아님) ────────────────────────────────
  NF8=\$(grep -c 'T2_ARG_PRODUCERS. fired' $LOG/\$SMK.log 2>/dev/null); NF8=\${NF8:-0}
  NSN=\$(grep -c 'T2_STALE_NOTE' $LOG/\$SMK.log 2>/dev/null); NSN=\${NSN:-0}
  NEN=\$(grep -c 'T2_WRITE_ARG_ENUM. deny' $LOG/\$SMK.log 2>/dev/null); NEN=\${NEN:-0}
  echo \"[t7343] 수리분 관측 - F8 발화=\$NF8 · STALE_NOTE=\$NSN · ENUM deny=\$NEN\"

  # ── ⛔死배선 게이트: 오늘 수리분이 예외로 조용히 죽으면 F8 이 또 침묵한다 ──
  NDEAD=\$(grep -cE 'arg-producer skipped|Traceback' $LOG/\$SMK.log 2>/dev/null); NDEAD=\${NDEAD:-0}
  if [ \"\$NDEAD\" -gt 0 ]; then
    echo '[t7343] ⛔오늘 수리분이 예외로 죽었다(arg-producer skipped / Traceback) - 본런을 돌리지 않는다'
    grep -E 'arg-producer skipped|Traceback' $LOG/\$SMK.log | head -5
    exit 1
  fi

  NTIER=\$(grep -c 'current_balance' $LOG/\$SMK.log 2>/dev/null); NTIER=\${NTIER:-0}
  echo \"[t7343] 잔액인자 발화=\$NTIER (0 이면 에이전트가 안 채운 것 - 종전 거동 폴백)\"
  NRD=\$(grep -c 'T2_SG_ROUND' $LOG/\$SMK.log 2>/dev/null); NRD=\${NRD:-0}
  NWEV=\$(grep -c 'T2_WRITE_EVIDENCE. deny' $LOG/\$SMK.log 2>/dev/null); NWEV=\${NWEV:-0}
  NM1=\$(grep -c 'principal=-1' $LOG/\$SMK.log 2>/dev/null); NM1=\${NM1:-0}
  echo \"[t7343] 반올림=\$NRD · WEV deny=\$NWEV (t7341=10) · 서브 -1 폐기=\$NM1 (미해결 표적)\"

  # ── ⛔자리표시자 수리 게이트: 서브가 예시 0.0 을 또 복사하면 중단 ──────────
  N00=\$(grep -c '부재(principal=0.0' $LOG/\$SMK.log 2>/dev/null); N00=\${N00:-0}
  echo \"[t7343] 격리 서브 0.0-복사 폐기 = \$N00 (0 이어야 수리가 먹은 것)\"
  if [ \"\$N00\" -gt 0 ]; then
    echo '[t7343] ⛔격리 서브가 answer_format 예시값(0.0)을 또 복사했다 - 자리표시자 수리 미적용 - 본런을 돌리지 않는다'
    grep '부재(principal=0.0' $LOG/\$SMK.log | head -3
    exit 1
  fi

  # ── ⛔문법 死배선 게이트: 격리 서브가 돌았는데 문법이 안 걸렸으면 중단 ──────
  NISO=\$(grep -c 'SG_ISOLATE. fetch' $LOG/\$SMK.log 2>/dev/null); NISO=\${NISO:-0}
  NSCH=\$(grep -c 'T2_SG_SCHEMA' $LOG/\$SMK.log 2>/dev/null); NSCH=\${NSCH:-0}
  echo \"[t7343] 격리 서브 fetch=\$NISO · 문법 적용=\$NSCH\"
  if [ \"\$NISO\" -gt 0 ] && [ \"\$NSCH\" -eq 0 ]; then
    echo '[t7343] ⛔격리 서브가 돌았는데 T2_SG_SCHEMA 발화 0 = 문법 死배선 - 본런을 돌리지 않는다'
    exit 1
  fi

  # ── ⛔누수 재발 게이트: 후보 명단에 General 이 실리면 중단 ────────────────
  if grep -q ', General ,' $LOG/\$SMK.log 2>/dev/null; then
    echo '[t7343] ⛔WRITE_ARG_ENUM 후보 명단에 General 재출현 - 본런을 돌리지 않는다'
    exit 1
  fi

  # ── 동결 ([[07]]·스모크 뒤) ────────────────────────────────────────────
  cd '$REPO'
  /home/woori/venvs/seka_env/bin/python reports/facet_rft_2026/freeze.py --on \\
    --tag t7343 --reason 'all-on composed stack, stage1 20 x nt2' || true
  cd '$REPO/scripts/distill/tau2'

  run_half() {
    NAME=\$1; PORT=\$2; TL=\$3
    TAG=bank_t7343_\${NAME}_20260822
    t2_launch \$TAG \$PORT \"\$TL\" $NT 2>&1 | tee $LOG/\$TAG.log
    echo \"[t7343] \$NAME 완료 · docs발화=\$(grep -c '$M_DOCS' $LOG/\$TAG.log 2>/dev/null || echo 0) · 값=\$(grep -c '$M_VAL' $LOG/\$TAG.log 2>/dev/null || echo 0)\"
    cd '$REPO' && mkdir -p reports/facet_rft_2026/sim_results
    gzip -c '$SIMS/'\$TAG'/results.json' > reports/facet_rft_2026/sim_results/\$TAG.results.json.gz
    gzip -c $LOG/\$TAG.log > reports/facet_rft_2026/sim_results/\$TAG.log.gz
    cd '$REPO/scripts/distill/tau2'
  }
  ( run_half halfA 8140 '$HALF_A' ) > $LOG/bank_t7343_halfA_chain.log 2>&1 &
  P1=\$!
  ( run_half halfB 8141 '$HALF_B' ) > $LOG/bank_t7343_halfB_chain.log 2>&1 &
  P2=\$!
  wait \$P1 \$P2

  # ── 영속 + 동결 해제 ([[30]] tracked 확인까지) ─────────────────────────
  cd '$REPO'
  cp $LOG/bank_t7343.meta.json reports/facet_rft_2026/sim_results/bank_t7343.meta.json || true
  git add -f reports/facet_rft_2026/sim_results/bank_t7343_*.gz reports/facet_rft_2026/sim_results/bank_t7343.meta.json
  git -c user.name=ghlee -c user.email=beingrelative@gmail.com commit -q -m 't7343 all-on stage1 results' || true
  git push -q origin facet-rft-2026 || true
  git ls-files --error-unmatch reports/facet_rft_2026/sim_results/bank_t7343_halfA_20260822.results.json.gz \\
    && echo '[t7343] persisted+tracked OK'
  /home/woori/venvs/seka_env/bin/python reports/facet_rft_2026/freeze.py --off --tag t7343 || true
  echo '[t7343] ALL DONE'
" </dev/null >"$LOG/bank_t7343_chain.log" 2>&1 &
echo "[t7343] 기동 PID=$! · sha=$SHA · 스모크 2 → 본런 40 sim (halfA 8140 · halfB 8141) · 로그 $LOG/bank_t7343_chain.log"
