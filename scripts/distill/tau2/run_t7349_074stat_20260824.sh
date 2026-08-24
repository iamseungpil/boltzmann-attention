#!/bin/bash
# t7349 — **074 단독 통계 확보 런** (사용자 지시 2026-08-24: *"074 의 통계를 위해서 필요하면 nt 를
#   4 이상으로 필요한 만큼 늘려서 통계 확보하라"*).
#
# ## 왜 이 런인가
#   074 는 t7328·t7335·t7336·t7346·t7348 다섯 런에서 **0/9** 다. 그런데 그 9 sim 의 시나리오는
#   **두 종뿐**이다 — 배치 seed 300 에서 파생되는 trial 0(626729)·trial 1(373753) 을 반복해 왔다.
#   ⇒ *"같은 이유로 죽는가"* 를 시나리오 n=2 로는 답할 수 없다. 시나리오를 **12 종**으로 넓힌다.
#
# ## 설계
#   표적 = `task_074` **단독**(사용자 지시: 074 와 다른 태스크를 섞지 마라).
#   A: 배치 seed 300(역사와 같은 계열) x nt6 -> trial 0..5   포트 8140
#   B: 배치 seed 400(새 계열)          x nt6 -> trial 0..5   포트 8141
#   합 12 sim. 두 GPU 병렬 · 러너 안 동시성 2.
#
# ## 이 런이 **답하는 것**
#   (1) 현재 코드에서 074 pass 율(기저선) (2) **실패 칸의 분포** — 12 시나리오에서 같은 자리에
#   멈추는가 (3) 향후 수리 A/B 의 대조군(같은 seed 로 재발사하면 짝 비교 성립).
#   검정력: 0/12 vs 5/12 = Fisher 양측 p=0.037 · 0/12 vs 6/12 = p=0.014.
#
# ## 이 런이 **못 하는 것**
#   금지: 개별 수리 귀속. 코드는 t7348 이후 로컬 커밋 10개를 실은 **묶음**이다(C594).
#     라이브 델타 둘(3aa3bf03 A2 문면 · 2f1b8f36 색인 695->698)이 함께 실린다.
#     이 런의 074 수치를 그 둘의 효과로 읽지 마라. **기준선이지 A/B 가 아니다.**
#   금지: 총점 델타(단일 태스크). 판정 단위는 reward 뿐([[69]]).
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
  echo "[t7349] REFUSING: engine paths have uncommitted changes: $DIRTY" >&2; exit 1
fi

# 배터리 = t7348 것을 그대로 이어받는다([[67]] 사본 금지) + ATM 원장 폐쇄 검정 추가
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
         test_apy_balance_tier.py test_ref_from_outputs.py \
         test_forensic_sidecar_authority.py \
         test_proc_regen_recheck.py test_r5_arg_empty_dispatched.py \
         test_selector_guard.py test_enum_reject_ledger.py \
         test_return_template_imperative.py test_decide_first_axis_named.py \
         test_actionreq_waitset_evidence.py test_deny_kind_env_fail.py \
         test_ours_text_canonical.py test_atm_ledger_close.py; do
  [ -f "$t" ] || continue
  PYTHONPATH=/home/woori/scratch/tau2-bench/src timeout 90 \
    /home/woori/venvs/seka_env/bin/python "$t" >/dev/null 2>&1 \
    || { echo "[t7349] REFUSING: $t FAIL" >&2; exit 1; }
done
echo "[t7349] VERIFY OK"

if pgrep -f "[t]2_launch" >/dev/null; then
  echo "[t7349] REFUSING: another live run is going" >&2; exit 1
fi
for f in "$LOG"/bank_t7349_*.log; do
  [ -e "$f" ] && { echo "[t7349] REFUSING: $f exists" >&2; exit 1; }
done
for d in "$SIMS"/bank_t7349_*; do
  [ -e "$d" ] && { echo "[t7349] REFUSING: $d exists" >&2; exit 1; }
done

PIN="T2_ACTION_SUB=1 T2_KEEP_DENY_BODY=1 T2_CALL_FORM=1 T2_ARG_EMPTY=1 T2_SEARCH_AGENT=1 \
T2_DECIDE_ANY=1 T2_WRITE_ARG_ENUM=1 T2_DECIDE_BEFORE_WRITE=1 T2_DECISION_CARRY=1 \
T2_DISCOVERY_STEP2=1 T2_ARG_AXIS=1 T2_WRITE_SUB=3 T2_ACTION_INDEX=1 T2_NOW_SELFCALL=1 \
T2_SEARCH_ON_PROCEED=1 T2_ACT_DEMAND=0 T2_DELIVER_PRECOMMIT=0 T2_PROCEED_DOCBODY=0 \
T2_DOCS_AT_WRITE=0 T2_SUB_REQUIREMENT=0 T2_HANDOFF_PREDICATE=0 T2_PENDING_DISCOVERED=0 \
T2_VERDICT_CARRY=0 T2_ELIG_LINE=0 T2_VERDICT_GATE=0 T2_CLAIM_VERIFY=0 \
T2_DECLFIRST=0 T2_DECLFIRST_GUIDE_FIX=0 T2_CATEGORY_CITE="
ON="T2_ARG_DOC_SUB=1 T2_VALUE_FORMULA=full T2_SG_DOCS=1"

echo "{\"tag\":\"t7349\",\"sha\":\"$SHA\",\"design\":\"task_074 alone, twelve scenarios: batch seed 300 nt6 on 8140 plus batch seed 400 nt6 on 8141\",\"on\":\"$ON\",\"why\":\"five runs have only ever drawn two scenarios of 074 (trial seeds 626729 and 373753), so 0/9 is 0 out of two scenarios repeated\",\"answers\":\"baseline pass rate at current code, distribution of the failure cell across twelve scenarios, and a paired control for a future repair arm\",\"caveat\":\"bundle of ten local commits since t7348 including two live deltas (A2 wording 3aa3bf03, doc index 2f1b8f36) - this is a baseline, not an A/B, and no individual delta may be attributed from it (C594)\"}" \
  | tee "$LOG/bank_t7349.meta.json"

setsid bash -c "
  cd '$REPO/scripts/distill/tau2'
  source ./go_stack.sh >/dev/null 2>&1
  export $PIN
  export $ON
  export GO_MAX_STEPS=150 GO_CONCURRENCY=2

  harvest_instr() {
    local _T=\$1 _S _F
    mkdir -p '$REPO/reports/facet_rft_2026/sim_results'
    for _S in fb trace; do
      _F=$LOG/\${_S}_\${_T}.jsonl
      if [ -s \"\$_F\" ]; then
        gzip -c \"\$_F\" > '$REPO/reports/facet_rft_2026/sim_results/'\${_S}_\${_T}.jsonl.gz
        echo \"[t7349] \${_S} harvested \${_T} (\$(wc -l < \"\$_F\") lines)\"
      else
        echo \"[t7349] WARN \${_S} not harvested: \$_F missing/empty - our-layer attribution is undecidable ([[25]] [[55]])\"
      fi
    done
  }

  SMK=bank_t7349_smoke_20260824
  echo '[t7349] === smoke (074 nt=1 on 8140) ==='
  t2_launch \$SMK 8140 task_074 1 > $LOG/\$SMK.log 2>&1 || true
  NTB=\$(grep -c 'Traceback' $LOG/\$SMK.log 2>/dev/null); NTB=\${NTB:-0}
  NVAL=\$(grep -c 'documented_return_for_stated_spend' $LOG/\$SMK.log 2>/dev/null); NVAL=\${NVAL:-0}
  NATM=\$(grep -c 'atm_fee_discrepanc' $LOG/\$SMK.log 2>/dev/null); NATM=\${NATM:-0}
  NSUB=\$(grep -c subcall $LOG/\$SMK.log 2>/dev/null); NSUB=\${NSUB:-0}
  NCMP=\$(grep -c T2_COMPUTE $LOG/\$SMK.log 2>/dev/null); NCMP=\${NCMP:-0}
  echo \"[t7349] smoke markers - Traceback=\$NTB val=\$NVAL atm=\$NATM subcall=\$NSUB compute=\$NCMP\"
  cd '$REPO' && mkdir -p reports/facet_rft_2026/sim_results
  gzip -c '$SIMS/'\$SMK'/results.json' > reports/facet_rft_2026/sim_results/\$SMK.results.json.gz 2>/dev/null || true
  gzip -c $LOG/\$SMK.log > reports/facet_rft_2026/sim_results/\$SMK.log.gz
  harvest_instr \$SMK
  git add -f reports/facet_rft_2026/sim_results/\$SMK*.gz \\
             reports/facet_rft_2026/sim_results/fb_\$SMK*.jsonl.gz \\
             reports/facet_rft_2026/sim_results/trace_\$SMK*.jsonl.gz 2>/dev/null || true
  git -c user.name=ghlee -c user.email=beingrelative@gmail.com commit -q -m 't7349 smoke' || true
  git push -q origin facet-rft-2026 || true
  cd '$REPO/scripts/distill/tau2'
  if [ \"\$NTB\" -gt 0 ]; then
    echo '[t7349] ABORT: Traceback in smoke - not running the main arms'
    grep -A3 Traceback $LOG/\$SMK.log | head -20
    exit 1
  fi
  if [ \"\$NATM\" -eq 0 ]; then
    echo '[t7349] ABORT: 074 trajectory with zero ATM comparator mentions - target path not reached'
    exit 1
  fi

  cd '$REPO'
  /home/woori/venvs/seka_env/bin/python reports/facet_rft_2026/freeze.py --on \\
    --tag t7349 --reason 'task_074 alone, 12 scenarios for statistics' || true
  cd '$REPO/scripts/distill/tau2'

  run_arm() {
    NAME=\$1; PORT=\$2; SEED=\$3
    TAG=bank_t7349_\${NAME}_20260824
    t2_launch \$TAG \$PORT task_074 6 --seed \$SEED 2>&1 | tee $LOG/\$TAG.log
    echo \"[t7349] \$NAME done (batch seed \$SEED)\"
    cd '$REPO' && mkdir -p reports/facet_rft_2026/sim_results
    gzip -c '$SIMS/'\$TAG'/results.json' > reports/facet_rft_2026/sim_results/\$TAG.results.json.gz
    gzip -c $LOG/\$TAG.log > reports/facet_rft_2026/sim_results/\$TAG.log.gz
    harvest_instr \$TAG
    cd '$REPO/scripts/distill/tau2'
  }
  ( run_arm s300 8140 300 ) > $LOG/bank_t7349_s300_chain.log 2>&1 &
  P1=\$!
  ( run_arm s400 8141 400 ) > $LOG/bank_t7349_s400_chain.log 2>&1 &
  P2=\$!
  wait \$P1 \$P2

  cd '$REPO'
  cp $LOG/bank_t7349.meta.json reports/facet_rft_2026/sim_results/bank_t7349.meta.json || true
  git add -f reports/facet_rft_2026/sim_results/bank_t7349_*.gz reports/facet_rft_2026/sim_results/bank_t7349.meta.json
  git add -f reports/facet_rft_2026/sim_results/fb_bank_t7349_*.jsonl.gz \\
             reports/facet_rft_2026/sim_results/trace_bank_t7349_*.jsonl.gz 2>/dev/null || true
  git -c user.name=ghlee -c user.email=beingrelative@gmail.com commit -q -m 't7349 task_074 twelve scenarios' || true
  git push -q origin facet-rft-2026 || true
  git ls-files --error-unmatch reports/facet_rft_2026/sim_results/bank_t7349_s300_20260824.results.json.gz \\
    && echo '[t7349] persisted+tracked OK'
  /home/woori/venvs/seka_env/bin/python reports/facet_rft_2026/freeze.py --off --tag t7349 || true
  echo '[t7349] ALL DONE'
" </dev/null >"$LOG/bank_t7349_chain.log" 2>&1 &
echo "[t7349] launched PID=$! sha=$SHA - smoke 1 then 12 sims (s300@8140 s400@8141) - log $LOG/bank_t7349_chain.log"
