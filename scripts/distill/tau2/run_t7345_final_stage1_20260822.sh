#!/bin/bash
# t7345 — ★**093 이 통과한 뒤** 처음 도는 20 태스크 본런 (사용자 선제 조건 충족).
#   조건: 20 태스크 런은 093·024 가 **둘 다 pass** 해야 돈다(사용자 지시 2026-08-22).
#     024  reward 1.0 (여러 런에서 안정)
#     093  reward **1.0** — 단독 스모크 `bank_x093_check` 에서 확인. 네 런 연속 0.0 이던 것이다.
#          축자: apy 4.275 · [T2_SG_ROUND] 33.00000000000004 → 33.0 · interest 33.0 · WEV deny 0
#
# ## 093 을 통과시킨 수리 사슬 (전부 **우리 층**·전부 격리 측정 뒤에 수리)
#   ① 표기 오염     에이전트가 레코드 계좌명을 전사하며 바꿔 checking boost 누락
#                   → 엔진이 도구 출력 **원문**을 REFERENCE 에 전달(ref_from_outputs)
#                   x481: 에이전트 요약 0/4 · 요약+지시 4/4 · **원문 4/4**(지시 없이)
#   ② 잔액 티어     base APY 가 잔액-조건부인데 서브가 잔액을 못 받아 낮은 쪽을 집음
#                   → REFERENCE 에 잔액 + "문서의 티어 표를 읽으라" 한 문장
#                   x481: A 0/4 · N_neg 0/4 · B(잔액만) **0/4** · C(잔액+지시) **4/4**
#   ③ 통화 반올림   우리 도구가 32.999999999999986 을 내고 모델이 옳게 33.0 으로 접자
#                   **우리 WEV 가 10회 반려**했다([[25]] 우리 오차가 우리 게이트를 막음)
#   ④ 0.0 포이즈닝  형식 예시 `{"principal": 0.0}` 의 값을 서브가 그대로 베낌(두 런 재현)
#                   → 자리표시자화 + 마감 라운드 문법 강제(T2_SG_SCHEMA)
#   ⑤ interest 전달 `account_id` 만 주고 getter 로 읽으라던 경로가 **격리에서 죽어 있다**
#                   x482: A_asis 답반환 **0/3** · N_neg 0/3 · **B_raw 3/3**
#
# ## ⚠정직하게 남길 것
#   · ⑤ 의 `ref_from_outputs` 는 093 단독 스모크에서 **결정적이지 않았다** — 서브는 여전히
#     `principal=-1` 로 폐기되고 폴백이 이겼다. 첫 갈래를 고치자 메인 인자가 옳아졌기 때문이다.
#     그 수리는 격리 0/3→3/3 로 옳지만 이번 pass 의 직접 원인은 ①+③ 이다.
#   · `[T2_SG_REFRAW]` 발화가 그 로그에 없었다 — 도구 출력이 궤적에 없는 시점의 호출은
#     fail-open 으로 종전 경로를 탄다. **이 런에서 발화 수를 세는 것이 그 확인이다.**
#
# ## 대조군 — t7336 **13/40** (변함없음)
#   로스터 20 태스크 · nt=2 · PIN · ON 이 t7336 러너와 **바이트 동일**하고 sha 만 다르다.
#   판정선 Δ ≥ 4/40. ⚠묶음이 크므로(수리 14 + 잔여 3 + 오늘 5) **묶음 Δ 로 개별 수리를
#   주장하지 않는다**(C594 실증). 귀속은 per-task 포렌식([[08]]). 채점은 reward 뿐([[69]]).
#
# ## 이번에 특히 볼 자리
#   ★093·094    apy 4.275 · interest 33.0 · WEV deny 0 이 **본런에서도** 재현되나
#   ★[T2_SG_REFRAW] 발화 수 — 원문 전달이 실제로 몇 번 살았나(0 이면 폴백만 이긴 것)
#   ★[T2_SG_ROUND] 발화 ↔ WEV deny 수(t7341 기준선 10)
#   073·050·085·040#1  수리 묶음 표적 · F8 부활(t7336 = 0)
#
# ## 스모크 게이트 ([[30]] 死배선에 돈 금지)
#   093 이 단독으로 이미 통과했지만 스모크는 **유지**한다 — 게이트가 오늘 여러 번 死배선을
#   잡았고(문법·0.0 복사), 두 GPU 병렬이라 벽시계 비용이 작다.
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
  echo "[t7345] REFUSING: 엔진 경로 커밋 안 된 변경 $DIRTY 개." >&2; exit 1
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
    || { echo "[t7345] REFUSING: $t FAIL" >&2; exit 1; }
done
echo "[t7345] VERIFY OK (배터리 31 - 오늘 수리 검정 5 등재·문법 포함)"

if pgrep -f "[t]2_launch" >/dev/null; then
  echo "[t7345] REFUSING: 다른 라이브 런이 돌고 있다" >&2; exit 1
fi
for f in "$LOG"/bank_t7345_*.log; do
  [ -e "$f" ] && { echo "[t7345] REFUSING: $f 존재" >&2; exit 1; }
done
for d in "$SIMS"/bank_t7345_*; do
  [ -e "$d" ] && { echo "[t7345] REFUSING: $d 잔존" >&2; exit 1; }
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

echo "{\"tag\":\"t7345\",\"sha\":\"$SHA\",\"design\":\"single all-on composed stack (user directive), no arms; first run after 093 passes solo (reward 1.0), the precondition for the 20-task run\",\"on\":\"$ON\",\"tasks\":\"stage1 20 x nt=$NT = 40 sims\",\"endpoint\":\"reward only\",\"reference\":\"t7336 13/40 - same runner lineage, same roster, same PIN/ON; engine sha differs by the repair bundle (14), the residual debts (3) and five fixes measured in isolation today (answer_format placeholder, isolate grammar, currency rounding, balance-tier instruction, ref_from_outputs) = 22 items; 093 passes solo before this run\",\"bar\":\"delta >= 4/40 vs reference; per-task sign table; reads/fabs/over-action logged\"}" \
  | tee "$LOG/bank_t7345.meta.json"

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
  SMK=bank_t7345_smoke_20260822
  SMKA=\${SMK}_a
  SMKB=\${SMK}_b
  echo '[t7345] === 스모크(093→8140 · 024→8141 · nt=1 · 병렬) ==='
  ( t2_launch \$SMKA 8140 task_093 1 ) > $LOG/\$SMKA.log 2>&1 &
  SPA=\$!
  ( t2_launch \$SMKB 8141 task_024 1 ) > $LOG/\$SMKB.log 2>&1 &
  SPB=\$!
  wait \$SPA \$SPB
  cat $LOG/\$SMKA.log $LOG/\$SMKB.log > $LOG/\$SMK.log
  NV=\$(grep -c '$M_VAL' $LOG/\$SMK.log 2>/dev/null); NV=\${NV:-0}
  ND=\$(grep -c '$M_DOCS' $LOG/\$SMK.log 2>/dev/null); ND=\${ND:-0}
  NC=\$(grep '$APYTOOL' $LOG/\$SMK.log 2>/dev/null | grep -v 'injected' | wc -l)
  echo \"[t7345] 스모크 발화 — 값주석=\$NV · docs전달=\$ND · apy도구 언급=\$NC\"
  grep '$M_DOCS' $LOG/\$SMK.log 2>/dev/null | head -4 || true
  cd '$REPO' && mkdir -p reports/facet_rft_2026/sim_results
  for _P in \$SMKA \$SMKB; do
    gzip -c '$SIMS/'\$_P'/results.json' > reports/facet_rft_2026/sim_results/\$_P.results.json.gz
  done
  gzip -c $LOG/\$SMK.log > reports/facet_rft_2026/sim_results/\$SMK.log.gz
  git add -f reports/facet_rft_2026/sim_results/\$SMK*.gz
  git -c user.name=ghlee -c user.email=beingrelative@gmail.com commit -q -m 't7345 smoke' || true
  git push -q origin facet-rft-2026 || true
  cd '$REPO/scripts/distill/tau2'
  if [ \"\$NV\" -eq 0 ]; then
    echo '[t7345] ⛔값 주석 발화 0 — 본런을 돌리지 않는다'
    exit 1
  fi
  if [ \"\$NC\" -gt 0 ] && [ \"\$ND\" -eq 0 ]; then
    echo '[t7345] ⛔apy 도구가 불렸는데 T2_SG_DOCS 발화 0 = 死배선 — 본런을 돌리지 않는다'
    exit 1
  fi
  [ \"\$NC\" -eq 0 ] && echo '[t7345] ⚠apy 도구 자체가 안 불림 — docs 게이트 판단 불가(관측으로 남긴다)'

  # ── 오늘 수리분: 관측 계수(게이트 아님) ────────────────────────────────
  NF8=\$(grep -c 'T2_ARG_PRODUCERS. fired' $LOG/\$SMK.log 2>/dev/null); NF8=\${NF8:-0}
  NSN=\$(grep -c 'T2_STALE_NOTE' $LOG/\$SMK.log 2>/dev/null); NSN=\${NSN:-0}
  NEN=\$(grep -c 'T2_WRITE_ARG_ENUM. deny' $LOG/\$SMK.log 2>/dev/null); NEN=\${NEN:-0}
  echo \"[t7345] 수리분 관측 - F8 발화=\$NF8 · STALE_NOTE=\$NSN · ENUM deny=\$NEN\"

  # ── ⛔死배선 게이트: 오늘 수리분이 예외로 조용히 죽으면 F8 이 또 침묵한다 ──
  NDEAD=\$(grep -cE 'arg-producer skipped|Traceback' $LOG/\$SMK.log 2>/dev/null); NDEAD=\${NDEAD:-0}
  if [ \"\$NDEAD\" -gt 0 ]; then
    echo '[t7345] ⛔오늘 수리분이 예외로 죽었다(arg-producer skipped / Traceback) - 본런을 돌리지 않는다'
    grep -E 'arg-producer skipped|Traceback' $LOG/\$SMK.log | head -5
    exit 1
  fi

  NTIER=\$(grep -c 'current_balance' $LOG/\$SMK.log 2>/dev/null); NTIER=\${NTIER:-0}
  echo \"[t7345] 잔액인자 발화=\$NTIER (0 이면 에이전트가 안 채운 것 - 종전 거동 폴백)\"
  NRD=\$(grep -c 'T2_SG_ROUND' $LOG/\$SMK.log 2>/dev/null); NRD=\${NRD:-0}
  NWEV=\$(grep -c 'T2_WRITE_EVIDENCE. deny' $LOG/\$SMK.log 2>/dev/null); NWEV=\${NWEV:-0}
  NM1=\$(grep -c 'principal=-1' $LOG/\$SMK.log 2>/dev/null); NM1=\${NM1:-0}
  echo \"[t7345] 반올림=\$NRD · WEV deny=\$NWEV (t7341=10) · 서브 -1 폐기=\$NM1 (미해결 표적)\"

  # ── ⛔자리표시자 수리 게이트: 서브가 예시 0.0 을 또 복사하면 중단 ──────────
  N00=\$(grep -c '부재(principal=0.0' $LOG/\$SMK.log 2>/dev/null); N00=\${N00:-0}
  echo \"[t7345] 격리 서브 0.0-복사 폐기 = \$N00 (0 이어야 수리가 먹은 것)\"
  if [ \"\$N00\" -gt 0 ]; then
    echo '[t7345] ⛔격리 서브가 answer_format 예시값(0.0)을 또 복사했다 - 자리표시자 수리 미적용 - 본런을 돌리지 않는다'
    grep '부재(principal=0.0' $LOG/\$SMK.log | head -3
    exit 1
  fi

  # ── ⛔문법 死배선 게이트: 격리 서브가 돌았는데 문법이 안 걸렸으면 중단 ──────
  NISO=\$(grep -c 'SG_ISOLATE. fetch' $LOG/\$SMK.log 2>/dev/null); NISO=\${NISO:-0}
  NSCH=\$(grep -c 'T2_SG_SCHEMA' $LOG/\$SMK.log 2>/dev/null); NSCH=\${NSCH:-0}
  echo \"[t7345] 격리 서브 fetch=\$NISO · 문법 적용=\$NSCH\"
  if [ \"\$NISO\" -gt 0 ] && [ \"\$NSCH\" -eq 0 ]; then
    echo '[t7345] ⛔격리 서브가 돌았는데 T2_SG_SCHEMA 발화 0 = 문법 死배선 - 본런을 돌리지 않는다'
    exit 1
  fi

  # ── ⛔누수 재발 게이트: 후보 명단에 General 이 실리면 중단 ────────────────
  if grep -q ', General ,' $LOG/\$SMK.log 2>/dev/null; then
    echo '[t7345] ⛔WRITE_ARG_ENUM 후보 명단에 General 재출현 - 본런을 돌리지 않는다'
    exit 1
  fi

  # ── 동결 ([[07]]·스모크 뒤) ────────────────────────────────────────────
  cd '$REPO'
  /home/woori/venvs/seka_env/bin/python reports/facet_rft_2026/freeze.py --on \\
    --tag t7345 --reason 'all-on composed stack, stage1 20 x nt2' || true
  cd '$REPO/scripts/distill/tau2'

  run_half() {
    NAME=\$1; PORT=\$2; TL=\$3
    TAG=bank_t7345_\${NAME}_20260822
    t2_launch \$TAG \$PORT \"\$TL\" $NT 2>&1 | tee $LOG/\$TAG.log
    echo \"[t7345] \$NAME 완료 · docs발화=\$(grep -c '$M_DOCS' $LOG/\$TAG.log 2>/dev/null || echo 0) · 값=\$(grep -c '$M_VAL' $LOG/\$TAG.log 2>/dev/null || echo 0)\"
    cd '$REPO' && mkdir -p reports/facet_rft_2026/sim_results
    gzip -c '$SIMS/'\$TAG'/results.json' > reports/facet_rft_2026/sim_results/\$TAG.results.json.gz
    gzip -c $LOG/\$TAG.log > reports/facet_rft_2026/sim_results/\$TAG.log.gz
    cd '$REPO/scripts/distill/tau2'
  }
  ( run_half halfA 8140 '$HALF_A' ) > $LOG/bank_t7345_halfA_chain.log 2>&1 &
  P1=\$!
  ( run_half halfB 8141 '$HALF_B' ) > $LOG/bank_t7345_halfB_chain.log 2>&1 &
  P2=\$!
  wait \$P1 \$P2

  # ── 영속 + 동결 해제 ([[30]] tracked 확인까지) ─────────────────────────
  cd '$REPO'
  cp $LOG/bank_t7345.meta.json reports/facet_rft_2026/sim_results/bank_t7345.meta.json || true
  git add -f reports/facet_rft_2026/sim_results/bank_t7345_*.gz reports/facet_rft_2026/sim_results/bank_t7345.meta.json
  git -c user.name=ghlee -c user.email=beingrelative@gmail.com commit -q -m 't7345 all-on stage1 results' || true
  git push -q origin facet-rft-2026 || true
  git ls-files --error-unmatch reports/facet_rft_2026/sim_results/bank_t7345_halfA_20260822.results.json.gz \\
    && echo '[t7345] persisted+tracked OK'
  /home/woori/venvs/seka_env/bin/python reports/facet_rft_2026/freeze.py --off --tag t7345 || true
  echo '[t7345] ALL DONE'
" </dev/null >"$LOG/bank_t7345_chain.log" 2>&1 &
echo "[t7345] 기동 PID=$! · sha=$SHA · 스모크 2 → 본런 40 sim (halfA 8140 · halfB 8141) · 로그 $LOG/bank_t7345_chain.log"
