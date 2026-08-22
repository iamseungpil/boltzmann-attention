#!/bin/bash
# t7342 — t7341 을 본런 시작 직후(0/20) 중단하고 **금액 반올림 수리까지 실어** 재발사한 런.
#   중단 이유(2026-08-22 · t7341 스모크 포렌식): 문법 수리는 먹었는데(0.0 복사 0 · 문법 9/9 ·
#   `get_interest_correction` 이 None → 값) **다음 결함이 드러났고 그것도 우리 것**이었다 —
#   도구가 `32.999999999999986` 을 냈고 모델은 통화로 **옳게** 33.0 으로 접어 write 했는데
#   `T2_WRITE_EVIDENCE` 가 *"the amount_difference (33.0) does not appear in any
#   get_interest_correction tool output"* 로 **10회 반려**했다. 모델은 아무것도 틀리지 않았다.
#   우리 표현 오차가 우리 게이트를 막았고, 그 출력이 게이트의 **유일 근거원**이라 오염이 곧
#   차단이 됐다([[25]]). 계산 도구를 쓰는 태스크가 로스터에 여럿이라 그대로 40 sim 을 태우면
#   전부 같은 벽에 부딪힌다 ⇒ 0/20 에서 멈추는 것이 싸다.
#
# ## 무엇이 실렸나 = **20건 한 묶음**
#   e7dcb97d  A1~A8·A10·A12~A16 (14건)
#   4373e7db  잔여 부채 3건 (A9 호출부 · OL-55 형제 · WRITE_ARG_ENUM 누수)
#   d4a38ead  answer_format 자리표시자화
#   07c4c2f0  T2_SG_SCHEMA — 격리 서브 형식을 문법(guided_json)으로 강제 · 도구 없는 라운드만
#   079c1d93  **A2 `result_round` — 금액 결과를 통화 자릿수로 접는다**
#             op 의 `0.08333333333333333`(1/12 근사) 잔차를 표현에서 제거. 접기는 **범위
#             게이트보다 앞**이라 두 검사와 반환문이 같은 수를 본다. 자릿수는 A2 선언뿐
#             (엔진 리터럴 0·[[05]])이고 **금액 도구에만** — `get_correct_savings_apy` 는
#             APY(2.775)라 접으면 값이 바뀌므로 건드리지 않았다([[62]] 근거 없는 확대 금지).
#             gold 미참조([[23]]): 근거는 크레딧 도구의 인자 계약(달러 금액)이다.
#
# ## t7341 스모크가 남긴 확정 사실 (이 런의 기준선)
#   0.0 예시 복사 폐기 = **0**(t7337 1 · t7338 4) · 문법 적용 **9/9**(死배선 아님)
#   `get_interest_correction` **32.999999999999986**(t7338 None) · WEV deny **10**(전부 표현 불일치)
#   task_024 reward **1.0** · task_093 reward **0.0**
#   ⚠**미해결로 남긴 것**: 격리 서브 자체는 여전히 실패한다 — sentinel 이 `0.0` → **`-1`** 로
#     바뀌었을 뿐 7회 폐기됐다(값이 서브 자신의 getter 출력에 부재). 원인이 재료 부족인지
#     "모름" 표현의 부재인지 **아직 확정되지 않았으므로 레버를 짓지 않는다**([[62]] 0순위).
#     이번 런은 그 자리를 **폴백이 메꾼 채로** 측정된다 — 다음 포렌식의 1순위 표적.
#
# ## 대조군 — t7336 **13/40** (변함없음)
#   로스터 20 태스크 · nt=2 · PIN · ON 이 t7336 러너와 **바이트 동일**하고 sha 만 다르다.
#   판정선 Δ ≥ 4/40. ⚠20건 묶음이므로 **묶음 Δ 로 개별 수리를 주장하지 않는다**(C594 실증).
#   귀속은 per-task 포렌식이 한다([[08]]). 채점은 reward 뿐([[69]]).
#
# ## 이번에 특히 볼 자리
#   ★093·094    `[T2_SG_ROUND]` 발화 ↔ **WEV deny 수가 줄었나**(10 → ?) · reward 가 움직였나
#   ★서브 -1    `부재(principal=-1` 발화 수(미해결 표적·수리 안 했으므로 남아 있어야 정상)
#   F8 부활     `[T2_ARG_PRODUCERS] fired` (t7336 = 0)
#   073·050·085·040#1  수리 묶음 표적
#
# ## 스모크 게이트 ([[30]] 死배선에 돈 금지)
#   task_093(격리·계산 표적·8140) + task_024(값·배달 표적·8141) x nt=1 = 2 sim **병렬**.
#   ⛔중단: ⑴값 주석 발화 0 ⑵apy 불렸는데 SG_DOCS 0 ⑶수리분 예외사망 ⑷0.0 복사 재출현
#   ⛔      ⑸격리 서브가 돌았는데 T2_SG_SCHEMA 발화 0(문법 死배선)
#   ⚠경고만: F8/STALE_NOTE 발화 0 · `[T2_SG_ROUND]` 발화 0(그 도구가 안 불린 것일 수 있다)
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
  echo "[t7342] REFUSING: 엔진 경로 커밋 안 된 변경 $DIRTY 개." >&2; exit 1
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
         test_a2_answer_format_placeholder.py; do
  [ -f "$t" ] || continue
  PYTHONPATH=/home/woori/scratch/tau2-bench/src timeout 90 \
    /home/woori/venvs/seka_env/bin/python "$t" >/dev/null 2>&1 \
    || { echo "[t7342] REFUSING: $t FAIL" >&2; exit 1; }
done
echo "[t7342] VERIFY OK (배터리 31 - 오늘 수리 검정 5 등재·문법 포함)"

if pgrep -f "[t]2_launch" >/dev/null; then
  echo "[t7342] REFUSING: 다른 라이브 런이 돌고 있다" >&2; exit 1
fi
for f in "$LOG"/bank_t7342_*.log; do
  [ -e "$f" ] && { echo "[t7342] REFUSING: $f 존재" >&2; exit 1; }
done
for d in "$SIMS"/bank_t7342_*; do
  [ -e "$d" ] && { echo "[t7342] REFUSING: $d 잔존" >&2; exit 1; }
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

echo "{\"tag\":\"t7342\",\"sha\":\"$SHA\",\"design\":\"single all-on composed stack (user directive), no arms; t7341 halted at 0/20 and relaunched with the currency rounding fix included\",\"on\":\"$ON\",\"tasks\":\"stage1 20 x nt=$NT = 40 sims\",\"endpoint\":\"reward only\",\"reference\":\"t7336 13/40 - same runner lineage, same roster, same PIN/ON; engine sha differs by the repair bundle (14), the residual debts (3), the answer_format placeholder fix (1), the isolate output grammar (1) and the currency rounding fix (1) = 20 items\",\"bar\":\"delta >= 4/40 vs reference; per-task sign table; reads/fabs/over-action logged\"}" \
  | tee "$LOG/bank_t7342.meta.json"

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
  SMK=bank_t7342_smoke_20260822
  SMKA=\${SMK}_a
  SMKB=\${SMK}_b
  echo '[t7342] === 스모크(093→8140 · 024→8141 · nt=1 · 병렬) ==='
  ( t2_launch \$SMKA 8140 task_093 1 ) > $LOG/\$SMKA.log 2>&1 &
  SPA=\$!
  ( t2_launch \$SMKB 8141 task_024 1 ) > $LOG/\$SMKB.log 2>&1 &
  SPB=\$!
  wait \$SPA \$SPB
  cat $LOG/\$SMKA.log $LOG/\$SMKB.log > $LOG/\$SMK.log
  NV=\$(grep -c '$M_VAL' $LOG/\$SMK.log 2>/dev/null); NV=\${NV:-0}
  ND=\$(grep -c '$M_DOCS' $LOG/\$SMK.log 2>/dev/null); ND=\${ND:-0}
  NC=\$(grep '$APYTOOL' $LOG/\$SMK.log 2>/dev/null | grep -v 'injected' | wc -l)
  echo \"[t7342] 스모크 발화 — 값주석=\$NV · docs전달=\$ND · apy도구 언급=\$NC\"
  grep '$M_DOCS' $LOG/\$SMK.log 2>/dev/null | head -4 || true
  cd '$REPO' && mkdir -p reports/facet_rft_2026/sim_results
  for _P in \$SMKA \$SMKB; do
    gzip -c '$SIMS/'\$_P'/results.json' > reports/facet_rft_2026/sim_results/\$_P.results.json.gz
  done
  gzip -c $LOG/\$SMK.log > reports/facet_rft_2026/sim_results/\$SMK.log.gz
  git add -f reports/facet_rft_2026/sim_results/\$SMK*.gz
  git -c user.name=ghlee -c user.email=beingrelative@gmail.com commit -q -m 't7342 smoke' || true
  git push -q origin facet-rft-2026 || true
  cd '$REPO/scripts/distill/tau2'
  if [ \"\$NV\" -eq 0 ]; then
    echo '[t7342] ⛔값 주석 발화 0 — 본런을 돌리지 않는다'
    exit 1
  fi
  if [ \"\$NC\" -gt 0 ] && [ \"\$ND\" -eq 0 ]; then
    echo '[t7342] ⛔apy 도구가 불렸는데 T2_SG_DOCS 발화 0 = 死배선 — 본런을 돌리지 않는다'
    exit 1
  fi
  [ \"\$NC\" -eq 0 ] && echo '[t7342] ⚠apy 도구 자체가 안 불림 — docs 게이트 판단 불가(관측으로 남긴다)'

  # ── 오늘 수리분: 관측 계수(게이트 아님) ────────────────────────────────
  NF8=\$(grep -c 'T2_ARG_PRODUCERS. fired' $LOG/\$SMK.log 2>/dev/null); NF8=\${NF8:-0}
  NSN=\$(grep -c 'T2_STALE_NOTE' $LOG/\$SMK.log 2>/dev/null); NSN=\${NSN:-0}
  NEN=\$(grep -c 'T2_WRITE_ARG_ENUM. deny' $LOG/\$SMK.log 2>/dev/null); NEN=\${NEN:-0}
  echo \"[t7342] 수리분 관측 - F8 발화=\$NF8 · STALE_NOTE=\$NSN · ENUM deny=\$NEN\"

  # ── ⛔死배선 게이트: 오늘 수리분이 예외로 조용히 죽으면 F8 이 또 침묵한다 ──
  NDEAD=\$(grep -cE 'arg-producer skipped|Traceback' $LOG/\$SMK.log 2>/dev/null); NDEAD=\${NDEAD:-0}
  if [ \"\$NDEAD\" -gt 0 ]; then
    echo '[t7342] ⛔오늘 수리분이 예외로 죽었다(arg-producer skipped / Traceback) - 본런을 돌리지 않는다'
    grep -E 'arg-producer skipped|Traceback' $LOG/\$SMK.log | head -5
    exit 1
  fi

  NRD=\$(grep -c 'T2_SG_ROUND' $LOG/\$SMK.log 2>/dev/null); NRD=\${NRD:-0}
  NWEV=\$(grep -c 'T2_WRITE_EVIDENCE. deny' $LOG/\$SMK.log 2>/dev/null); NWEV=\${NWEV:-0}
  NM1=\$(grep -c 'principal=-1' $LOG/\$SMK.log 2>/dev/null); NM1=\${NM1:-0}
  echo \"[t7342] 반올림=\$NRD · WEV deny=\$NWEV (t7341=10) · 서브 -1 폐기=\$NM1 (미해결 표적)\"

  # ── ⛔자리표시자 수리 게이트: 서브가 예시 0.0 을 또 복사하면 중단 ──────────
  N00=\$(grep -c '부재(principal=0.0' $LOG/\$SMK.log 2>/dev/null); N00=\${N00:-0}
  echo \"[t7342] 격리 서브 0.0-복사 폐기 = \$N00 (0 이어야 수리가 먹은 것)\"
  if [ \"\$N00\" -gt 0 ]; then
    echo '[t7342] ⛔격리 서브가 answer_format 예시값(0.0)을 또 복사했다 - 자리표시자 수리 미적용 - 본런을 돌리지 않는다'
    grep '부재(principal=0.0' $LOG/\$SMK.log | head -3
    exit 1
  fi

  # ── ⛔문법 死배선 게이트: 격리 서브가 돌았는데 문법이 안 걸렸으면 중단 ──────
  NISO=\$(grep -c 'SG_ISOLATE. fetch' $LOG/\$SMK.log 2>/dev/null); NISO=\${NISO:-0}
  NSCH=\$(grep -c 'T2_SG_SCHEMA' $LOG/\$SMK.log 2>/dev/null); NSCH=\${NSCH:-0}
  echo \"[t7342] 격리 서브 fetch=\$NISO · 문법 적용=\$NSCH\"
  if [ \"\$NISO\" -gt 0 ] && [ \"\$NSCH\" -eq 0 ]; then
    echo '[t7342] ⛔격리 서브가 돌았는데 T2_SG_SCHEMA 발화 0 = 문법 死배선 - 본런을 돌리지 않는다'
    exit 1
  fi

  # ── ⛔누수 재발 게이트: 후보 명단에 General 이 실리면 중단 ────────────────
  if grep -q ', General ,' $LOG/\$SMK.log 2>/dev/null; then
    echo '[t7342] ⛔WRITE_ARG_ENUM 후보 명단에 General 재출현 - 본런을 돌리지 않는다'
    exit 1
  fi

  # ── 동결 ([[07]]·스모크 뒤) ────────────────────────────────────────────
  cd '$REPO'
  /home/woori/venvs/seka_env/bin/python reports/facet_rft_2026/freeze.py --on \\
    --tag t7342 --reason 'all-on composed stack, stage1 20 x nt2' || true
  cd '$REPO/scripts/distill/tau2'

  run_half() {
    NAME=\$1; PORT=\$2; TL=\$3
    TAG=bank_t7342_\${NAME}_20260822
    t2_launch \$TAG \$PORT \"\$TL\" $NT 2>&1 | tee $LOG/\$TAG.log
    echo \"[t7342] \$NAME 완료 · docs발화=\$(grep -c '$M_DOCS' $LOG/\$TAG.log 2>/dev/null || echo 0) · 값=\$(grep -c '$M_VAL' $LOG/\$TAG.log 2>/dev/null || echo 0)\"
    cd '$REPO' && mkdir -p reports/facet_rft_2026/sim_results
    gzip -c '$SIMS/'\$TAG'/results.json' > reports/facet_rft_2026/sim_results/\$TAG.results.json.gz
    gzip -c $LOG/\$TAG.log > reports/facet_rft_2026/sim_results/\$TAG.log.gz
    cd '$REPO/scripts/distill/tau2'
  }
  ( run_half halfA 8140 '$HALF_A' ) > $LOG/bank_t7342_halfA_chain.log 2>&1 &
  P1=\$!
  ( run_half halfB 8141 '$HALF_B' ) > $LOG/bank_t7342_halfB_chain.log 2>&1 &
  P2=\$!
  wait \$P1 \$P2

  # ── 영속 + 동결 해제 ([[30]] tracked 확인까지) ─────────────────────────
  cd '$REPO'
  cp $LOG/bank_t7342.meta.json reports/facet_rft_2026/sim_results/bank_t7342.meta.json || true
  git add -f reports/facet_rft_2026/sim_results/bank_t7342_*.gz reports/facet_rft_2026/sim_results/bank_t7342.meta.json
  git -c user.name=ghlee -c user.email=beingrelative@gmail.com commit -q -m 't7342 all-on stage1 results' || true
  git push -q origin facet-rft-2026 || true
  git ls-files --error-unmatch reports/facet_rft_2026/sim_results/bank_t7342_halfA_20260822.results.json.gz \\
    && echo '[t7342] persisted+tracked OK'
  /home/woori/venvs/seka_env/bin/python reports/facet_rft_2026/freeze.py --off --tag t7342 || true
  echo '[t7342] ALL DONE'
" </dev/null >"$LOG/bank_t7342_chain.log" 2>&1 &
echo "[t7342] 기동 PID=$! · sha=$SHA · 스모크 2 → 본런 40 sim (halfA 8140 · halfB 8141) · 로그 $LOG/bank_t7342_chain.log"
