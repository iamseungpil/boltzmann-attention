#!/bin/bash
# t7338 — t7337 을 51분(2/40 sim)만에 중단하고 **잔여 부채 3건까지 실어** 재발사한 런.
#   중단 이유(사용자 판단 2026-08-22): t7337 은 알려진 결함 3개를 실은 채 돌고 있었다 —
#   특히 A9 호출부 미배선은 **F8 정당 발화 전멸**을 뜻하는데, 그것은 C594 가 t7336 의
#   *매입(비용)* 으로 지목한 바로 그 항목이다. 즉 t7337 의 수치는 "수리된 스택" 을
#   대표하지 못한다 ⇒ 반쯤 낡은 측정에 7시간을 더 쓰지 않는다.
#   (t7337 부분 결과 2 sim 은 커밋 966ed1ea 에 영속돼 있다 — 소실 0.)
#
# ## 무엇이 실렸나 = **17건 한 묶음**
#   커밋 e7dcb97d  A1~A8·A10·A12~A16 (14건, t7337 이 재려던 것)
#   커밋 4373e7db  잔여 부채 3건 —
#     · A9 호출부   F8 억제 술어를 정본 `user_tool_value_ready` 로 재배선
#                   (구판은 *이름이 등장했나* 를 봤다: tool_call 이름 + 인자 JSON 의
#                    `[a-z0-9_]+` 토막 + 문자열 인자값 전부 ⇒ **건네기만 해도** 영구 침묵.
#                    t7328 7 · t7335 5 → t7336 0. 인자 토막 파싱은 [[59]] 위반이기도 했다)
#     · OL-55 형제  `T2_STALE_STRIP` 노트도 tool_calls 를 전부 지운 턴에 본문이 비면
#                   **손님 발화 전체**가 된다 → A15 와 같은 정본(`_commit_machine_note`)으로
#     · 누수        `T2_WRITE_ARG_ENUM` 후보 명단의 `' General '`(= `_general_` 슬러그) 제거
#                   + fail-open 술어를 `_subs` → `_names` 로(빈 후보로 deny 하지 않는다·[[64]])
#     (곁들여 `_general_` 을 아는 4자리를 정본 `_subject_keys` 하나로 통합 — 실물 A2 18군 전수 동치)
#
# ## 대조군 — t7336 **13/40**
#   **같은 러너 계보 · 같은 로스터 20 태스크 · 같은 nt=2 · 같은 PIN/ON** 이고 sha 만 수리분
#   차이다. 판정선 = C483/C548 잡음 바닥 **Δ ≥ 4/40**.
#   ⚠묶음이 17건이므로 **묶음 Δ 로 개별 수리를 주장하지 않는다** — C594 실증대로 묶음 Δ 는
#     귀속을 못 준다(t7336 의 +7 이 7건 어디에도 안 붙었다). 귀속은 per-task 포렌식이 한다([[08]]).
#   [[70]] 의무 3종: 전체 reward 짝 · **태스크별 부호표** · 무엇을 팔았나.
#   채점은 **reward 뿐**([[69]]).
#
# ## 이번에 특히 볼 자리
#   F8 부활       `[T2_ARG_PRODUCERS] fired` 수 (t7336 = 0 이었다) ↔ 그중 값이 이미 있던 수
#   OL-55 형제    `[T2_STALE_NOTE] regen ok|empty-body` 수 (손님 발화가 될 뻔한 턴 수)
#   073           P5 회귀(1/2→0/2)를 A-묶음이 되돌렸나
#   050·085       A1(에러-형상)·A2(원장 성공만)·A5(레지스트리 출처) 표적
#   040#1         F8 오억제가 풀렸나 — **이번에는 호출부까지 배선됐다**(t7337 은 아니었다)
#
# ## 스모크 게이트 ([[30]] 死배선에 돈 금지)
#   task_093(T2_SG_DOCS 표적) + task_024(값·배달 표적) x nt=1 = 2 sim.
#   ⛔중단: ⑴값 주석 발화 0  ⑵apy 도구가 불렸는데 [T2_SG_DOCS] 0
#   ⛔**신규**: 오늘 수리분이 **예외로 조용히 죽는가**. A9 호출부는 `except Exception` 안에
#     있어 예외가 나도 no-op 로 넘어간다 — 그러면 F8 이 또 침묵하고 우리는 그걸 모른 채
#     40 sim 을 태운다. `arg-producer skipped` · `Traceback` 중 하나라도 나오면 중단한다.
#   ⚠경고만: 도구 자체가 안 불림 · F8/STALE_NOTE 발화 0(발화 자리가 안 온 것일 수 있다 —
#     게이트로 걸면 정당한 본런을 막는다. 계수만 남긴다)
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
  echo "[t7338] REFUSING: 엔진 경로 커밋 안 된 변경 $DIRTY 개." >&2; exit 1
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
         test_t7336_g2_gate_axis.py test_write_arg_enum.py; do
  [ -f "$t" ] || continue
  PYTHONPATH=/home/woori/scratch/tau2-bench/src timeout 90 \
    /home/woori/venvs/seka_env/bin/python "$t" >/dev/null 2>&1 \
    || { echo "[t7338] REFUSING: $t FAIL" >&2; exit 1; }
done
echo "[t7338] VERIFY OK (배터리 30 - 오늘 수리 검정 4 등재)"

if pgrep -f "[t]2_launch" >/dev/null; then
  echo "[t7338] REFUSING: 다른 라이브 런이 돌고 있다" >&2; exit 1
fi
for f in "$LOG"/bank_t7338_*.log; do
  [ -e "$f" ] && { echo "[t7338] REFUSING: $f 존재" >&2; exit 1; }
done
for d in "$SIMS"/bank_t7338_*; do
  [ -e "$d" ] && { echo "[t7338] REFUSING: $d 잔존" >&2; exit 1; }
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

echo "{\"tag\":\"t7338\",\"sha\":\"$SHA\",\"design\":\"single all-on composed stack (user directive), no arms; t7337 halted at 2/40 and relaunched with the residual debts included\",\"on\":\"$ON\",\"tasks\":\"stage1 20 x nt=$NT = 40 sims\",\"endpoint\":\"reward only\",\"reference\":\"t7336 13/40 - same runner lineage, same roster, same PIN/ON; engine sha differs by the repair bundle (14) plus the residual debts (3) = 17 items\",\"bar\":\"delta >= 4/40 vs reference; per-task sign table; reads/fabs/over-action logged\"}" \
  | tee "$LOG/bank_t7338.meta.json"

setsid bash -c "
  cd '$REPO/scripts/distill/tau2'
  source ./go_stack.sh >/dev/null 2>&1
  export $PIN
  export $ON
  export GO_MAX_STEPS=150 GO_CONCURRENCY=1

  # ── 스모크 (2 sim · 8141) ──────────────────────────────────────────────
  SMK=bank_t7338_smoke_20260822
  echo '[t7338] === 스모크(task_093,task_024 x nt=1 · 8141) ==='
  t2_launch \$SMK 8141 task_093,task_024 1 2>&1 | tee $LOG/\$SMK.log
  NV=\$(grep -c '$M_VAL' $LOG/\$SMK.log 2>/dev/null); NV=\${NV:-0}
  ND=\$(grep -c '$M_DOCS' $LOG/\$SMK.log 2>/dev/null); ND=\${ND:-0}
  NC=\$(grep '$APYTOOL' $LOG/\$SMK.log 2>/dev/null | grep -v 'injected' | wc -l)
  echo \"[t7338] 스모크 발화 — 값주석=\$NV · docs전달=\$ND · apy도구 언급=\$NC\"
  grep '$M_DOCS' $LOG/\$SMK.log 2>/dev/null | head -4 || true
  cd '$REPO' && mkdir -p reports/facet_rft_2026/sim_results
  gzip -c '$SIMS/'\$SMK'/results.json' > reports/facet_rft_2026/sim_results/\$SMK.results.json.gz
  gzip -c $LOG/\$SMK.log > reports/facet_rft_2026/sim_results/\$SMK.log.gz
  git add -f reports/facet_rft_2026/sim_results/\$SMK.*.gz
  git -c user.name=ghlee -c user.email=beingrelative@gmail.com commit -q -m 't7338 smoke' || true
  git push -q origin facet-rft-2026 || true
  cd '$REPO/scripts/distill/tau2'
  if [ \"\$NV\" -eq 0 ]; then
    echo '[t7338] ⛔값 주석 발화 0 — 본런을 돌리지 않는다'
    exit 1
  fi
  if [ \"\$NC\" -gt 0 ] && [ \"\$ND\" -eq 0 ]; then
    echo '[t7338] ⛔apy 도구가 불렸는데 T2_SG_DOCS 발화 0 = 死배선 — 본런을 돌리지 않는다'
    exit 1
  fi
  [ \"\$NC\" -eq 0 ] && echo '[t7338] ⚠apy 도구 자체가 안 불림 — docs 게이트 판단 불가(관측으로 남긴다)'

  # ── 오늘 수리분: 관측 계수(게이트 아님) ────────────────────────────────
  NF8=\$(grep -c 'T2_ARG_PRODUCERS. fired' $LOG/\$SMK.log 2>/dev/null); NF8=\${NF8:-0}
  NSN=\$(grep -c 'T2_STALE_NOTE' $LOG/\$SMK.log 2>/dev/null); NSN=\${NSN:-0}
  NEN=\$(grep -c 'T2_WRITE_ARG_ENUM. deny' $LOG/\$SMK.log 2>/dev/null); NEN=\${NEN:-0}
  echo \"[t7338] 수리분 관측 - F8 발화=\$NF8 · STALE_NOTE=\$NSN · ENUM deny=\$NEN\"

  # ── ⛔死배선 게이트: 오늘 수리분이 예외로 조용히 죽으면 F8 이 또 침묵한다 ──
  NDEAD=\$(grep -cE 'arg-producer skipped|Traceback' $LOG/\$SMK.log 2>/dev/null); NDEAD=\${NDEAD:-0}
  if [ \"\$NDEAD\" -gt 0 ]; then
    echo '[t7338] ⛔오늘 수리분이 예외로 죽었다(arg-producer skipped / Traceback) - 본런을 돌리지 않는다'
    grep -E 'arg-producer skipped|Traceback' $LOG/\$SMK.log | head -5
    exit 1
  fi

  # ── ⛔누수 재발 게이트: 후보 명단에 General 이 실리면 중단 ────────────────
  if grep -q ', General ,' $LOG/\$SMK.log 2>/dev/null; then
    echo '[t7338] ⛔WRITE_ARG_ENUM 후보 명단에 General 재출현 - 본런을 돌리지 않는다'
    exit 1
  fi

  # ── 동결 ([[07]]·스모크 뒤) ────────────────────────────────────────────
  cd '$REPO'
  /home/woori/venvs/seka_env/bin/python reports/facet_rft_2026/freeze.py --on \\
    --tag t7338 --reason 'all-on composed stack, stage1 20 x nt2' || true
  cd '$REPO/scripts/distill/tau2'

  run_half() {
    NAME=\$1; PORT=\$2; TL=\$3
    TAG=bank_t7338_\${NAME}_20260822
    t2_launch \$TAG \$PORT \"\$TL\" $NT 2>&1 | tee $LOG/\$TAG.log
    echo \"[t7338] \$NAME 완료 · docs발화=\$(grep -c '$M_DOCS' $LOG/\$TAG.log 2>/dev/null || echo 0) · 값=\$(grep -c '$M_VAL' $LOG/\$TAG.log 2>/dev/null || echo 0)\"
    cd '$REPO' && mkdir -p reports/facet_rft_2026/sim_results
    gzip -c '$SIMS/'\$TAG'/results.json' > reports/facet_rft_2026/sim_results/\$TAG.results.json.gz
    gzip -c $LOG/\$TAG.log > reports/facet_rft_2026/sim_results/\$TAG.log.gz
    cd '$REPO/scripts/distill/tau2'
  }
  ( run_half halfA 8140 '$HALF_A' ) > $LOG/bank_t7338_halfA_chain.log 2>&1 &
  P1=\$!
  ( run_half halfB 8141 '$HALF_B' ) > $LOG/bank_t7338_halfB_chain.log 2>&1 &
  P2=\$!
  wait \$P1 \$P2

  # ── 영속 + 동결 해제 ([[30]] tracked 확인까지) ─────────────────────────
  cd '$REPO'
  cp $LOG/bank_t7338.meta.json reports/facet_rft_2026/sim_results/bank_t7338.meta.json || true
  git add -f reports/facet_rft_2026/sim_results/bank_t7338_*.gz reports/facet_rft_2026/sim_results/bank_t7338.meta.json
  git -c user.name=ghlee -c user.email=beingrelative@gmail.com commit -q -m 't7338 all-on stage1 results' || true
  git push -q origin facet-rft-2026 || true
  git ls-files --error-unmatch reports/facet_rft_2026/sim_results/bank_t7338_halfA_20260822.results.json.gz \\
    && echo '[t7338] persisted+tracked OK'
  /home/woori/venvs/seka_env/bin/python reports/facet_rft_2026/freeze.py --off --tag t7338 || true
  echo '[t7338] ALL DONE'
" </dev/null >"$LOG/bank_t7338_chain.log" 2>&1 &
echo "[t7338] 기동 PID=$! · sha=$SHA · 스모크 2 → 본런 40 sim (halfA 8140 · halfB 8141) · 로그 $LOG/bank_t7338_chain.log"
