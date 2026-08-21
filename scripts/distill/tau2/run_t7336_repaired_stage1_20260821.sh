#!/bin/bash
# t7336 — t7335 재런(수리 스택): P1~P5 + F8 게이트 + C587/C585 반영 후 1단계 20×nt2.
#   t7335 러너 게이트 결함 2건 수리(이중-0 카운터·스키마-주입줄 오계수 — 핸드오프 §0c).
#   참고 대조 = t7335 nt1 (동일 계열 스택·부분) · t7328 6/40(sha 상이). 판정선 Δ≥4/40.
# t7336 — **1단계 20 태스크 × nt=2 · 전-레버 합성 단일 스택** (사용자 지시 2026-08-21 저녁 축자:
#   *"24 태스크 A/B 비교할게 있나? 그냥 모든 레버 다 켜고 실행하는게 낫지 않나? 각 레버를 켜는
#    조합이 달라지면 성과도 달라지니 다켜고 pass 측정하는게 맞지 않나? nt=2 로."*)
#
# ## 구성 — A/B 아님 · [[19]] 합성-우선 · [[60]] 전부 켜기
#   go_stack.sh 정본 스택(SG 서브·관문·핀·읽기루틴 전부 상주) 위에
#     + t7333 승자 조합    T2_ARG_DOC_SUB=1 · T2_VALUE_FORMULA=full   (C580: 값+배달 합성 8/8)
#     + 이번 세션 신규     T2_SG_DOCS=1                                (C585: 격리 관문1 생존 45%→87.5%)
#     + requires_reads 4행(C587)은 A3 커밋에 실려 있어 플래그 불요(큐→핀이 읽는다)
#   PIN 의 0 항목은 **측정이 기각/보류한 것**(T2_ACT_DEMAND=C492 over-action 2→8 등) — 되켜지 않는다.
#   ⇒ "전부 켜기" = 승자 전 조합. 기각된 레버 부활은 [[70]] 절충 설계 후에만.
#
# ## 로스터·판정
#   1단계 정본 20 태스크 × nt=2 = 40 sim. 채점 = **reward 뿐**([[69]]).
#   ⚠참고 대조 = t7328 6/40 — 단 **엔진 sha 가 다르므로 엄밀 A/B 아님**(계획서 §4 P1 재사용 금지
#   조항 그대로). 판정선 = C483/C548 잡음 바닥 **Δ ≥ 4/40**. 의무 3종([[70]]): 전체 짝 ·
#   태스크별 부호표 · 무엇을 팔았나(조회·날조·over-action).
#
# ## 스모크 게이트 ([[30]] 死배선에 돈 금지)
#   task_093(T2_SG_DOCS 표적) + task_024(값·배달 표적) × nt=1 = 2 sim.
#   ⛔중단: ⑴값 주석 발화 0  ⑵get_correct_savings_apy 가 불렸는데 [T2_SG_DOCS] 0
#   ⚠경고만: 도구 자체가 안 불림(발화 자리 자체가 안 온 것 — user-sim 변동)
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
  echo "[t7336] REFUSING: 엔진 경로 커밋 안 된 변경 $DIRTY 개." >&2; exit 1
fi

for t in test_a2_three_layer.py test_flag_registry.py test_claim_verify.py \
         test_claim_tool_index.py test_read_routine.py test_proc_read_connect.py \
         test_verdict_gate.py test_verdict_carry.py test_no_undefined_names.py \
         test_no_unbound_a2.py test_quote_in.py test_args_equal.py test_t2_procedure.py \
         test_proc_absent_wiring.py test_pin_read_replay.py test_eplan.py \
         test_decision_carry.py test_route_trace.py test_group_parse.py \
         test_resolve_cap_runtime.py test_byref_repairs.py test_no_prose_regex.py \
         test_compute_params.py test_sg_docs_delivery.py test_sg_src0_axis.py \
         test_sg_fetch_iso.py test_sg_isofb.py; do
  [ -f "$t" ] || continue
  PYTHONPATH=/home/woori/scratch/tau2-bench/src timeout 90 \
    /home/woori/venvs/seka_env/bin/python "$t" >/dev/null 2>&1 \
    || { echo "[t7336] REFUSING: $t FAIL" >&2; exit 1; }
done
echo "[t7336] VERIFY OK (배터리 26)"

if pgrep -f "[t]2_launch" >/dev/null; then
  echo "[t7336] REFUSING: 다른 라이브 런이 돌고 있다" >&2; exit 1
fi
for f in "$LOG"/bank_t7336_*.log; do
  [ -e "$f" ] && { echo "[t7336] REFUSING: $f 존재" >&2; exit 1; }
done
for d in "$SIMS"/bank_t7336_*; do
  [ -e "$d" ] && { echo "[t7336] REFUSING: $d 잔존" >&2; exit 1; }
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

echo "{\"tag\":\"t7336\",\"sha\":\"$SHA\",\"design\":\"single all-on composed stack (user directive), no arms\",\"on\":\"$ON\",\"tasks\":\"stage1 20 x nt=$NT = 40 sims\",\"endpoint\":\"reward only\",\"reference\":\"t7328 6/40 - informal (different engine sha, not a strict A/B)\",\"bar\":\"delta >= 4/40 vs reference; per-task sign table; reads/fabs/over-action logged\"}" \
  | tee "$LOG/bank_t7336.meta.json"

setsid bash -c "
  cd '$REPO/scripts/distill/tau2'
  source ./go_stack.sh >/dev/null 2>&1
  export $PIN
  export $ON
  export GO_MAX_STEPS=150 GO_CONCURRENCY=1

  # ── 스모크 (2 sim · 8141) ──────────────────────────────────────────────
  SMK=bank_t7336_smoke_20260821b
  echo '[t7336] === 스모크(task_093,task_024 x nt=1 · 8141) ==='
  t2_launch \$SMK 8141 task_093,task_024 1 2>&1 | tee $LOG/\$SMK.log
  NV=\$(grep -c '$M_VAL' $LOG/\$SMK.log 2>/dev/null); NV=\${NV:-0}
  ND=\$(grep -c '$M_DOCS' $LOG/\$SMK.log 2>/dev/null); ND=\${ND:-0}
  NC=\$(grep '$APYTOOL' $LOG/\$SMK.log 2>/dev/null | grep -v 'injected' | wc -l)
  echo \"[t7336] 스모크 발화 — 값주석=\$NV · docs전달=\$ND · apy도구 언급=\$NC\"
  grep '$M_DOCS' $LOG/\$SMK.log 2>/dev/null | head -4 || true
  cd '$REPO' && mkdir -p reports/facet_rft_2026/sim_results
  gzip -c '$SIMS/'\$SMK'/results.json' > reports/facet_rft_2026/sim_results/\$SMK.results.json.gz
  gzip -c $LOG/\$SMK.log > reports/facet_rft_2026/sim_results/\$SMK.log.gz
  git add -f reports/facet_rft_2026/sim_results/\$SMK.*.gz
  git -c user.name=ghlee -c user.email=beingrelative@gmail.com commit -q -m 't7336 smoke' || true
  git push -q origin facet-rft-2026 || true
  cd '$REPO/scripts/distill/tau2'
  if [ \"\$NV\" -eq 0 ]; then
    echo '[t7336] ⛔값 주석 발화 0 — 본런을 돌리지 않는다'
    exit 1
  fi
  if [ \"\$NC\" -gt 0 ] && [ \"\$ND\" -eq 0 ]; then
    echo '[t7336] ⛔apy 도구가 불렸는데 T2_SG_DOCS 발화 0 = 死배선 — 본런을 돌리지 않는다'
    exit 1
  fi
  [ \"\$NC\" -eq 0 ] && echo '[t7336] ⚠apy 도구 자체가 안 불림 — docs 게이트 판단 불가(관측으로 남긴다)'

  # ── 동결 ([[07]]·스모크 뒤) ────────────────────────────────────────────
  cd '$REPO'
  /home/woori/venvs/seka_env/bin/python reports/facet_rft_2026/freeze.py --on \\
    --tag t7336 --reason 'all-on composed stack, stage1 20 x nt2' || true
  cd '$REPO/scripts/distill/tau2'

  run_half() {
    NAME=\$1; PORT=\$2; TL=\$3
    TAG=bank_t7336_\${NAME}_20260821b
    t2_launch \$TAG \$PORT \"\$TL\" $NT 2>&1 | tee $LOG/\$TAG.log
    echo \"[t7336] \$NAME 완료 · docs발화=\$(grep -c '$M_DOCS' $LOG/\$TAG.log 2>/dev/null || echo 0) · 값=\$(grep -c '$M_VAL' $LOG/\$TAG.log 2>/dev/null || echo 0)\"
    cd '$REPO' && mkdir -p reports/facet_rft_2026/sim_results
    gzip -c '$SIMS/'\$TAG'/results.json' > reports/facet_rft_2026/sim_results/\$TAG.results.json.gz
    gzip -c $LOG/\$TAG.log > reports/facet_rft_2026/sim_results/\$TAG.log.gz
    cd '$REPO/scripts/distill/tau2'
  }
  ( run_half halfA 8140 '$HALF_A' ) > $LOG/bank_t7336_halfA_chain.log 2>&1 &
  P1=\$!
  ( run_half halfB 8141 '$HALF_B' ) > $LOG/bank_t7336_halfB_chain.log 2>&1 &
  P2=\$!
  wait \$P1 \$P2

  # ── 영속 + 동결 해제 ([[30]] tracked 확인까지) ─────────────────────────
  cd '$REPO'
  cp $LOG/bank_t7336.meta.json reports/facet_rft_2026/sim_results/bank_t7336.meta.json || true
  git add -f reports/facet_rft_2026/sim_results/bank_t7336_*.gz reports/facet_rft_2026/sim_results/bank_t7336.meta.json
  git -c user.name=ghlee -c user.email=beingrelative@gmail.com commit -q -m 't7336 all-on stage1 results' || true
  git push -q origin facet-rft-2026 || true
  git ls-files --error-unmatch reports/facet_rft_2026/sim_results/bank_t7336_halfA_20260821b.results.json.gz \\
    && echo '[t7336] persisted+tracked OK'
  /home/woori/venvs/seka_env/bin/python reports/facet_rft_2026/freeze.py --off --tag t7336 || true
  echo '[t7336] ALL DONE'
" </dev/null >"$LOG/bank_t7336_chain.log" 2>&1 &
echo "[t7336] 기동 PID=$! · sha=$SHA · 스모크 2 → 본런 40 sim (halfA 8140 · halfB 8141) · 로그 $LOG/bank_t7336_chain.log"
