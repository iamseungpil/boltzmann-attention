#!/bin/bash
# t7339 — t7338 을 스모크 직후 중단하고 **격리 서브 answer_format 수리까지 실어** 재발사한 런.
#   중단 이유(2026-08-22 · 093 실시간 포렌식): t7338 스모크의 task_093 이 34분을 태우던 중
#   그 원인을 우리 층에서 찾았다 — 격리 서브가 `principal=0.0; actual_apy=0.0` 을 내고
#   폐기되는 일이 **t7337·t7338 두 런 모두에서** 재현됐고(1회 ↔ 4회), 그 두 숫자는 우리가 준
#   형식 예시 `{"principal": 0.0, "actual_apy": 0.0}` 의 값과 **정확히 같았다**. 저축계좌 잔액이
#   0.0 일 수 없으므로 계산이 아니라 **예시 복사**다([[42]] copy=induction-head).
#   사슬: 서브 답 폐기 → 폴백에서 메인 추측 → grounding 드롭 → 도구 None → 모델이 amount 를
#   자기 계산해 write → `T2_WRITE_EVIDENCE` deny(정당) → **출구 없는 반복**.
#
# ## 무엇이 실렸나 = **18건 한 묶음**
#   e7dcb97d  A1~A8·A10·A12~A16 (14건)
#   4373e7db  잔여 부채 3건 (A9 호출부 · OL-55 형제 · WRITE_ARG_ENUM 누수)
#   d4a38ead  **isolate.answer_format 자리표시자화** — banking scaffold_get 4개 도구의 숫자
#             자리 6곳을 `<number>` 로. 같은 파일이 이미 `"source": "<verbatim quote>"` 로
#             올바른 관행을 쓰고 있었고 **숫자 자리만** 실제 값이었다. A2 3층 바이트 동일.
#             ⇒ 이것은 새 레버가 아니라 **우리가 떠먹이던 값의 제거**다([[62]]): 엔진은 여전히
#               고르지 않고 값은 서브가 낸다. gold 는 보지 않았다([[23]]).
#
# ## 대조군 — t7336 **13/40** (변함없음)
#   로스터 20 태스크 · nt=2 · PIN · ON 이 t7336 러너와 **바이트 동일**하고 sha 만 다르다.
#   판정선 Δ ≥ 4/40. ⚠18건 묶음이므로 **묶음 Δ 로 개별 수리를 주장하지 않는다**(C594 실증).
#   귀속은 per-task 포렌식이 한다([[08]]). 채점은 reward 뿐([[69]]).
#
# ## 이번에 특히 볼 자리
#   ★093        `마감-답 값이 서브 출력에 부재(principal=0.0` 이 **0 이 되는가**(수리의 직접 표적)
#               → 서브가 실값을 내면 grounding 드롭·WEV livelock 사슬 전체가 끊긴다
#   F8 부활     `[T2_ARG_PRODUCERS] fired` 수 (t7336 = 0)
#   OL-55 형제  `[T2_STALE_NOTE]` 수
#   073         P5 회귀(1/2→0/2)를 A-묶음이 되돌렸나
#   040#1       F8 오억제가 풀렸나(호출부까지 배선된 첫 런)
#   ⚠[[70]] 무엇을 파는가: 예시가 덜 구체적이라 서브가 **형식을 틀릴** 여지가 는다(JSON 아닌 답).
#     `<number>` 로 타입은 남겼다. 세는 것 = 0.0-폐기 수 ↔ 형식오류 폐기 수.
#
# ## 스모크 게이트 ([[30]] 死배선에 돈 금지)
#   task_093(격리 서브·SG_DOCS 표적) + task_024(값·배달 표적) x nt=1 = 2 sim.
#   ⛔중단: ⑴값 주석 발화 0  ⑵apy 도구가 불렸는데 [T2_SG_DOCS] 0
#   ⛔      ⑶오늘 수리분이 예외로 죽음(`arg-producer skipped` · `Traceback`)
#   ⛔**신규 ⑷**: `부재(principal=0.0` 재출현 = 자리표시자 수리가 **먹지 않았다**. 이 한 줄이
#     이번 수리의 유일한 라이브 판정이므로 게이트로 건다(0.0 복사는 재현되는 결정적 신호였다).
#   ⚠경고만: 도구 자체가 안 불림 · F8/STALE_NOTE 발화 0
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
  echo "[t7339] REFUSING: 엔진 경로 커밋 안 된 변경 $DIRTY 개." >&2; exit 1
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
    || { echo "[t7339] REFUSING: $t FAIL" >&2; exit 1; }
done
echo "[t7339] VERIFY OK (배터리 31 - 오늘 수리 검정 5 등재)"

if pgrep -f "[t]2_launch" >/dev/null; then
  echo "[t7339] REFUSING: 다른 라이브 런이 돌고 있다" >&2; exit 1
fi
for f in "$LOG"/bank_t7339_*.log; do
  [ -e "$f" ] && { echo "[t7339] REFUSING: $f 존재" >&2; exit 1; }
done
for d in "$SIMS"/bank_t7339_*; do
  [ -e "$d" ] && { echo "[t7339] REFUSING: $d 잔존" >&2; exit 1; }
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

echo "{\"tag\":\"t7339\",\"sha\":\"$SHA\",\"design\":\"single all-on composed stack (user directive), no arms; t7338 halted after smoke and relaunched with the isolate answer_format fix included\",\"on\":\"$ON\",\"tasks\":\"stage1 20 x nt=$NT = 40 sims\",\"endpoint\":\"reward only\",\"reference\":\"t7336 13/40 - same runner lineage, same roster, same PIN/ON; engine sha differs by the repair bundle (14), the residual debts (3) and the isolate answer_format placeholder fix (1) = 18 items\",\"bar\":\"delta >= 4/40 vs reference; per-task sign table; reads/fabs/over-action logged\"}" \
  | tee "$LOG/bank_t7339.meta.json"

setsid bash -c "
  cd '$REPO/scripts/distill/tau2'
  source ./go_stack.sh >/dev/null 2>&1
  export $PIN
  export $ON
  export GO_MAX_STEPS=150 GO_CONCURRENCY=1

  # ── 스모크 (2 sim · 8141) ──────────────────────────────────────────────
  SMK=bank_t7339_smoke_20260822
  echo '[t7339] === 스모크(task_093,task_024 x nt=1 · 8141) ==='
  t2_launch \$SMK 8141 task_093,task_024 1 2>&1 | tee $LOG/\$SMK.log
  NV=\$(grep -c '$M_VAL' $LOG/\$SMK.log 2>/dev/null); NV=\${NV:-0}
  ND=\$(grep -c '$M_DOCS' $LOG/\$SMK.log 2>/dev/null); ND=\${ND:-0}
  NC=\$(grep '$APYTOOL' $LOG/\$SMK.log 2>/dev/null | grep -v 'injected' | wc -l)
  echo \"[t7339] 스모크 발화 — 값주석=\$NV · docs전달=\$ND · apy도구 언급=\$NC\"
  grep '$M_DOCS' $LOG/\$SMK.log 2>/dev/null | head -4 || true
  cd '$REPO' && mkdir -p reports/facet_rft_2026/sim_results
  gzip -c '$SIMS/'\$SMK'/results.json' > reports/facet_rft_2026/sim_results/\$SMK.results.json.gz
  gzip -c $LOG/\$SMK.log > reports/facet_rft_2026/sim_results/\$SMK.log.gz
  git add -f reports/facet_rft_2026/sim_results/\$SMK.*.gz
  git -c user.name=ghlee -c user.email=beingrelative@gmail.com commit -q -m 't7339 smoke' || true
  git push -q origin facet-rft-2026 || true
  cd '$REPO/scripts/distill/tau2'
  if [ \"\$NV\" -eq 0 ]; then
    echo '[t7339] ⛔값 주석 발화 0 — 본런을 돌리지 않는다'
    exit 1
  fi
  if [ \"\$NC\" -gt 0 ] && [ \"\$ND\" -eq 0 ]; then
    echo '[t7339] ⛔apy 도구가 불렸는데 T2_SG_DOCS 발화 0 = 死배선 — 본런을 돌리지 않는다'
    exit 1
  fi
  [ \"\$NC\" -eq 0 ] && echo '[t7339] ⚠apy 도구 자체가 안 불림 — docs 게이트 판단 불가(관측으로 남긴다)'

  # ── 오늘 수리분: 관측 계수(게이트 아님) ────────────────────────────────
  NF8=\$(grep -c 'T2_ARG_PRODUCERS. fired' $LOG/\$SMK.log 2>/dev/null); NF8=\${NF8:-0}
  NSN=\$(grep -c 'T2_STALE_NOTE' $LOG/\$SMK.log 2>/dev/null); NSN=\${NSN:-0}
  NEN=\$(grep -c 'T2_WRITE_ARG_ENUM. deny' $LOG/\$SMK.log 2>/dev/null); NEN=\${NEN:-0}
  echo \"[t7339] 수리분 관측 - F8 발화=\$NF8 · STALE_NOTE=\$NSN · ENUM deny=\$NEN\"

  # ── ⛔死배선 게이트: 오늘 수리분이 예외로 조용히 죽으면 F8 이 또 침묵한다 ──
  NDEAD=\$(grep -cE 'arg-producer skipped|Traceback' $LOG/\$SMK.log 2>/dev/null); NDEAD=\${NDEAD:-0}
  if [ \"\$NDEAD\" -gt 0 ]; then
    echo '[t7339] ⛔오늘 수리분이 예외로 죽었다(arg-producer skipped / Traceback) - 본런을 돌리지 않는다'
    grep -E 'arg-producer skipped|Traceback' $LOG/\$SMK.log | head -5
    exit 1
  fi

  # ── ⛔자리표시자 수리 게이트: 서브가 예시 0.0 을 또 복사하면 중단 ──────────
  N00=\$(grep -c '부재(principal=0.0' $LOG/\$SMK.log 2>/dev/null); N00=\${N00:-0}
  echo \"[t7339] 격리 서브 0.0-복사 폐기 = \$N00 (0 이어야 수리가 먹은 것)\"
  if [ \"\$N00\" -gt 0 ]; then
    echo '[t7339] ⛔격리 서브가 answer_format 예시값(0.0)을 또 복사했다 - 자리표시자 수리 미적용 - 본런을 돌리지 않는다'
    grep '부재(principal=0.0' $LOG/\$SMK.log | head -3
    exit 1
  fi

  # ── ⛔누수 재발 게이트: 후보 명단에 General 이 실리면 중단 ────────────────
  if grep -q ', General ,' $LOG/\$SMK.log 2>/dev/null; then
    echo '[t7339] ⛔WRITE_ARG_ENUM 후보 명단에 General 재출현 - 본런을 돌리지 않는다'
    exit 1
  fi

  # ── 동결 ([[07]]·스모크 뒤) ────────────────────────────────────────────
  cd '$REPO'
  /home/woori/venvs/seka_env/bin/python reports/facet_rft_2026/freeze.py --on \\
    --tag t7339 --reason 'all-on composed stack, stage1 20 x nt2' || true
  cd '$REPO/scripts/distill/tau2'

  run_half() {
    NAME=\$1; PORT=\$2; TL=\$3
    TAG=bank_t7339_\${NAME}_20260822
    t2_launch \$TAG \$PORT \"\$TL\" $NT 2>&1 | tee $LOG/\$TAG.log
    echo \"[t7339] \$NAME 완료 · docs발화=\$(grep -c '$M_DOCS' $LOG/\$TAG.log 2>/dev/null || echo 0) · 값=\$(grep -c '$M_VAL' $LOG/\$TAG.log 2>/dev/null || echo 0)\"
    cd '$REPO' && mkdir -p reports/facet_rft_2026/sim_results
    gzip -c '$SIMS/'\$TAG'/results.json' > reports/facet_rft_2026/sim_results/\$TAG.results.json.gz
    gzip -c $LOG/\$TAG.log > reports/facet_rft_2026/sim_results/\$TAG.log.gz
    cd '$REPO/scripts/distill/tau2'
  }
  ( run_half halfA 8140 '$HALF_A' ) > $LOG/bank_t7339_halfA_chain.log 2>&1 &
  P1=\$!
  ( run_half halfB 8141 '$HALF_B' ) > $LOG/bank_t7339_halfB_chain.log 2>&1 &
  P2=\$!
  wait \$P1 \$P2

  # ── 영속 + 동결 해제 ([[30]] tracked 확인까지) ─────────────────────────
  cd '$REPO'
  cp $LOG/bank_t7339.meta.json reports/facet_rft_2026/sim_results/bank_t7339.meta.json || true
  git add -f reports/facet_rft_2026/sim_results/bank_t7339_*.gz reports/facet_rft_2026/sim_results/bank_t7339.meta.json
  git -c user.name=ghlee -c user.email=beingrelative@gmail.com commit -q -m 't7339 all-on stage1 results' || true
  git push -q origin facet-rft-2026 || true
  git ls-files --error-unmatch reports/facet_rft_2026/sim_results/bank_t7339_halfA_20260822.results.json.gz \\
    && echo '[t7339] persisted+tracked OK'
  /home/woori/venvs/seka_env/bin/python reports/facet_rft_2026/freeze.py --off --tag t7339 || true
  echo '[t7339] ALL DONE'
" </dev/null >"$LOG/bank_t7339_chain.log" 2>&1 &
echo "[t7339] 기동 PID=$! · sha=$SHA · 스모크 2 → 본런 40 sim (halfA 8140 · halfB 8141) · 로그 $LOG/bank_t7339_chain.log"
