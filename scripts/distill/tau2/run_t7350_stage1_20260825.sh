#!/bin/bash
# t7350 — stage1 20 태스크 × nt2 = 40 sim (사용자 지시 2026-08-25:
#   *"20 태스크 nt=2 실험 준비하라. 수정한 모든 내용 들어가서 pass 최대한 많은 태스크에서 올려야 한다"* ·
#   *"이번 런에서 hard-0 문제들의 blocked 된 상태를 최대한 돌파할 수 있게 하라"*)
#
# ## 대조 = t7348 12/40 (sha aed30e20 · 같은 러너 계보·로스터·PIN)
#
# ## 이 런에 처음 실리는 것
#   ⑴ **T2_SG_PROMPT_V2** (074 전사 결손·오늘 격리로 확정)
#      x525 계열 7런·팔 13종·chk_2 기준 실측:
#        REFERENCE 를 JSON 블록으로 주면 행이 빠지고(13~15), 문장/평문이면 16/16
#        answer_format 이 재료보다 앞이면 유령 중복 +3, 뒤면 정확히 16
#      ⇒ 조립 순서만 바꾼다(선언 텍스트 그대로·도메인 리터럴 0). 검정 팔 `N_wire`.
#   ⑵ 밤새 커밋된 것들 — A2 문면 수리(3aa3bf03) · 색인 695→698(2f1b8f36) ·
#      계산기/체크리스트 성공 마커(bf0f7c59) · **서브 입출력 사이드카**(e5df320f)
#      ⇒ 이 런은 서브가 **무엇을 받고 무엇을 냈나**를 처음으로 기록한다([[76]] 진단 ①의 재료).
#
# ## 이 런이 **못 하는 것**
#   ⛔묶음이다 — 개별 수리 귀속 금지(C594). 판정은 **태스크별 부호표**로 하고 총점 Δ 는 보조다.
#   ⛔016·057·063 에는 이번에 새 레버가 **없다**(오늘 셋 다 음성/미확정). 그 셋이 안 열려도
#     그것은 이 런의 실패가 아니다.
#   ⛔004 는 31% 동전던지기다(C597~C599) — 한 결과로 회귀도 수리도 말하지 마라.
set -e
REPO=/home/woori/workspace_common/boltzmann-attention-pi
LOG=/home/woori/scratch/logs
SIMS=/home/woori/scratch/tau2-bench/data/simulations
REP=$REPO/reports/facet_rft_2026
mkdir -p "$LOG"
cd "$REPO/scripts/distill/tau2"
SHA=$(cd "$REPO" && git rev-parse --short HEAD)

# ── 게이트 0: 전사 배선 검정(N_wire)이 chk_2 를 닫았나 ────────────────────
say() { echo "[t7350 $(date +%H:%M:%S)] $*"; }
for i in $(seq 1 90); do
  [ -s "$REP/x525j_wire_check_2026_08_25.json" ] && break
  sleep 20
done
GATE=$(/home/woori/venvs/seka_env/bin/python - <<'PY'
import json, io, os
p = os.path.expanduser("/home/woori/workspace_common/boltzmann-attention-pi/reports/facet_rft_2026/x525j_wire_check_2026_08_25.json")
try:
    d = json.load(io.open(p, encoding="utf-8"))
except Exception:
    print("NOFILE"); raise SystemExit
rows = [r for r in d.get("rows", []) if r.get("arm") == "N_wire" and r.get("msg") == 38]
ok = [r for r in rows if r.get("cover") == r.get("withdrawals")]
print("PASS" if rows and len(ok) == len(rows) else "FAIL:%d/%d" % (len(ok), len(rows)))
PY
)
say "배선 검정 게이트 = $GATE"
if [ "$GATE" != "PASS" ]; then
  say "REFUSING: N_wire 가 chk_2 를 닫지 못했다 ($GATE) — 검증 안 된 수리로 런을 태우지 않는다"
  exit 1
fi

DIRTY=$(cd "$REPO" && git status --porcelain -- \
  scripts/distill/tau2/t2_gate_patch.py scripts/distill/tau2/t2_search.py \
  scripts/distill/tau2/t2_resolve.py scripts/distill/tau2/t2_scaffold_get.py \
  scripts/distill/tau2/a2/ | grep -cv '^??' || true)
if [ "$DIRTY" != "0" ]; then say "REFUSING: 엔진 경로 미커밋 $DIRTY"; exit 1; fi

for t in test_a2_three_layer.py test_flag_registry.py test_claim_verify.py \
         test_claim_tool_index.py test_read_routine.py test_proc_read_connect.py \
         test_verdict_gate.py test_verdict_carry.py test_no_undefined_names.py \
         test_no_unbound_a2.py test_quote_in.py test_args_equal.py test_t2_procedure.py \
         test_proc_absent_wiring.py test_pin_read_replay.py test_eplan.py \
         test_decision_carry.py test_route_trace.py test_group_parse.py \
         test_resolve_cap_runtime.py test_byref_repairs.py test_no_prose_regex.py \
         test_compute_params.py test_sg_docs_delivery.py test_sg_src0_axis.py \
         test_sg_fetch_iso.py test_sg_isofb.py test_atm_ledger_close.py \
         test_t7337_residual_debt.py test_t7336_g1_our_layer.py \
         test_t7336_g2_gate_axis.py test_write_arg_enum.py \
         test_a2_answer_format_placeholder.py test_result_round.py \
         test_apy_balance_tier.py test_ref_from_outputs.py \
         test_forensic_sidecar_authority.py \
         test_proc_regen_recheck.py test_r5_arg_empty_dispatched.py \
         test_selector_guard.py test_enum_reject_ledger.py \
         test_return_template_imperative.py test_decide_first_axis_named.py \
         test_actionreq_waitset_evidence.py test_deny_kind_env_fail.py \
         test_ours_text_canonical.py; do
  [ -f "$t" ] || continue
  PYTHONPATH=/home/woori/scratch/tau2-bench/src timeout 90 \
    /home/woori/venvs/seka_env/bin/python "$t" >/dev/null 2>&1 \
    || { say "REFUSING: $t FAIL"; exit 1; }
done
say "VERIFY OK"

if pgrep -f "[t]2_launch" >/dev/null; then say "REFUSING: 다른 라이브 런"; exit 1; fi
for f in "$LOG"/bank_t7350_*.log; do [ -e "$f" ] && { say "REFUSING: $f 존재"; exit 1; }; done
for d in "$SIMS"/bank_t7350_*; do [ -e "$d" ] && { say "REFUSING: $d 잔존"; exit 1; }; done

PIN="T2_ACTION_SUB=1 T2_KEEP_DENY_BODY=1 T2_CALL_FORM=1 T2_ARG_EMPTY=1 T2_SEARCH_AGENT=1 \
T2_DECIDE_ANY=1 T2_WRITE_ARG_ENUM=1 T2_DECIDE_BEFORE_WRITE=1 T2_DECISION_CARRY=1 \
T2_DISCOVERY_STEP2=1 T2_ARG_AXIS=1 T2_WRITE_SUB=3 T2_ACTION_INDEX=1 T2_NOW_SELFCALL=1 \
T2_SEARCH_ON_PROCEED=1 T2_ACT_DEMAND=0 T2_DELIVER_PRECOMMIT=0 T2_PROCEED_DOCBODY=0 \
T2_DOCS_AT_WRITE=0 T2_SUB_REQUIREMENT=0 T2_HANDOFF_PREDICATE=0 T2_PENDING_DISCOVERED=0 \
T2_VERDICT_CARRY=0 T2_ELIG_LINE=0 T2_VERDICT_GATE=0 T2_CLAIM_VERIFY=0 \
T2_DECLFIRST=0 T2_DECLFIRST_GUIDE_FIX=0 T2_CATEGORY_CITE="
ON="T2_ARG_DOC_SUB=1 T2_VALUE_FORMULA=full T2_SG_DOCS=1 T2_SG_PROMPT_V2=1"

HALF_A='task_003,task_004,task_017,task_024,task_055,task_072,task_073,task_093,task_094,task_100'
HALF_B='task_016,task_033,task_040,task_050,task_057,task_063,task_074,task_079,task_085,task_098'
NT=2

echo "{\"tag\":\"t7350\",\"sha\":\"$SHA\",\"design\":\"stage1 20 x nt2, same roster and PIN as t7348, with the transcription prompt order fixed\",\"on\":\"$ON\",\"new\":\"T2_SG_PROMPT_V2 (isolation-verified on 074: plain reference, field contract before the records, answer_format after) plus the overnight commits that were never measured live\",\"reference\":\"t7348 12/40 at sha aed30e20\",\"bar\":\"per-task sign table is mandatory; a bundle delta cannot attribute an individual repair (C594)\",\"caveat\":\"016, 057 and 063 have no new lever in this run - their staying at zero is not evidence against the run\"}" \
  | tee "$LOG/bank_t7350.meta.json"

setsid bash -c "
  cd '$REPO/scripts/distill/tau2'
  source ./go_stack.sh >/dev/null 2>&1
  export $PIN
  export $ON
  export GO_MAX_STEPS=150 GO_CONCURRENCY=1

  harvest_instr() {
    local _T=\$1 _S _F
    mkdir -p '$REP/sim_results'
    for _S in fb trace; do
      _F=$LOG/\${_S}_\${_T}.jsonl
      if [ -s \"\$_F\" ]; then
        gzip -c \"\$_F\" > '$REP/sim_results/'\${_S}_\${_T}.jsonl.gz
        echo \"[t7350] \${_S} 회수 \${_T} (\$(wc -l < \"\$_F\") 행)\"
      else
        echo \"[t7350] ⚠\${_S} 미회수 \$_F — 우리-층 귀속 판정 불가([[25]])\"
      fi
    done
  }

  SMK=bank_t7350_smoke_20260825
  SMKA=\${SMK}_a; SMKB=\${SMK}_b
  echo '[t7350] === 스모크(093→8140 · 024→8141 · nt1 · 병렬) ==='
  ( t2_launch \$SMKA 8140 task_093 1 ) > $LOG/\$SMKA.log 2>&1 &
  SPA=\$!
  ( t2_launch \$SMKB 8141 task_024 1 ) > $LOG/\$SMKB.log 2>&1 &
  SPB=\$!
  wait \$SPA \$SPB
  cat $LOG/\$SMKA.log $LOG/\$SMKB.log > $LOG/\$SMK.log
  NV=\$(grep -c 'documented_return_for_stated_spend' $LOG/\$SMK.log 2>/dev/null); NV=\${NV:-0}
  ND=\$(grep -c 'T2_SG_DOCS' $LOG/\$SMK.log 2>/dev/null); ND=\${ND:-0}
  NV2=\$(grep -c 'T2_SG_PROMPT_V2' $LOG/\$SMK.log 2>/dev/null); NV2=\${NV2:-0}
  NTB=\$(grep -c 'Traceback' $LOG/\$SMK.log 2>/dev/null); NTB=\${NTB:-0}
  NSC=\$(grep -c 'subcall' $LOG/\$SMK.log 2>/dev/null); NSC=\${NSC:-0}
  echo \"[t7350] 스모크 — 값주석=\$NV · docs=\$ND · **PROMPT_V2=\$NV2** · subcall=\$NSC · Traceback=\$NTB\"
  cd '$REPO' && mkdir -p reports/facet_rft_2026/sim_results
  for _P in \$SMKA \$SMKB; do
    gzip -c '$SIMS/'\$_P'/results.json' > reports/facet_rft_2026/sim_results/\$_P.results.json.gz 2>/dev/null || true
    harvest_instr \$_P
  done
  gzip -c $LOG/\$SMK.log > reports/facet_rft_2026/sim_results/\$SMK.log.gz
  git add -f reports/facet_rft_2026/sim_results/\$SMK*.gz \\
             reports/facet_rft_2026/sim_results/fb_\$SMK*.jsonl.gz \\
             reports/facet_rft_2026/sim_results/trace_\$SMK*.jsonl.gz 2>/dev/null || true
  git -c user.name=ghlee -c user.email=beingrelative@gmail.com commit -q -m 't7350 smoke' || true
  git push -q origin facet-rft-2026 || true
  cd '$REPO/scripts/distill/tau2'
  if [ \"\$NTB\" -gt 0 ]; then echo '[t7350] ⛔스모크 Traceback — 중단'; grep -A3 Traceback $LOG/\$SMK.log | head -20; exit 1; fi
  if [ \"\$NV\" -eq 0 ]; then echo '[t7350] ⛔값 주석 0 — 중단'; exit 1; fi
  if [ \"\$NV2\" -eq 0 ]; then
    echo '[t7350] ⚠PROMPT_V2 발화 0 — 093/024 는 그 서브를 안 쓸 수 있다(관측으로 남긴다)'
  fi

  cd '$REPO'
  /home/woori/venvs/seka_env/bin/python reports/facet_rft_2026/freeze.py --on \\
    --tag t7350 --reason 'stage1 20 x nt2 with transcription prompt order' || true
  cd '$REPO/scripts/distill/tau2'

  run_half() {
    NAME=\$1; PORT=\$2; TL=\$3
    TAG=bank_t7350_\${NAME}_20260825
    t2_launch \$TAG \$PORT \"\$TL\" $NT 2>&1 | tee $LOG/\$TAG.log
    echo \"[t7350] \$NAME 완료 · V2발화=\$(grep -c 'T2_SG_PROMPT_V2' $LOG/\$TAG.log 2>/dev/null || echo 0)\"
    cd '$REPO' && mkdir -p reports/facet_rft_2026/sim_results
    gzip -c '$SIMS/'\$TAG'/results.json' > reports/facet_rft_2026/sim_results/\$TAG.results.json.gz
    gzip -c $LOG/\$TAG.log > reports/facet_rft_2026/sim_results/\$TAG.log.gz
    harvest_instr \$TAG
    cd '$REPO/scripts/distill/tau2'
  }
  ( run_half halfA 8140 '$HALF_A' ) > $LOG/bank_t7350_halfA_chain.log 2>&1 &
  P1=\$!
  ( run_half halfB 8141 '$HALF_B' ) > $LOG/bank_t7350_halfB_chain.log 2>&1 &
  P2=\$!
  wait \$P1 \$P2

  cd '$REPO'
  cp $LOG/bank_t7350.meta.json reports/facet_rft_2026/sim_results/bank_t7350.meta.json || true
  git add -f reports/facet_rft_2026/sim_results/bank_t7350_*.gz reports/facet_rft_2026/sim_results/bank_t7350.meta.json
  git add -f reports/facet_rft_2026/sim_results/fb_bank_t7350_*.jsonl.gz \\
             reports/facet_rft_2026/sim_results/trace_bank_t7350_*.jsonl.gz 2>/dev/null || true
  git -c user.name=ghlee -c user.email=beingrelative@gmail.com commit -q -m 't7350 stage1 results' || true
  git push -q origin facet-rft-2026 || true
  git ls-files --error-unmatch reports/facet_rft_2026/sim_results/bank_t7350_halfA_20260825.results.json.gz \\
    && echo '[t7350] persisted+tracked OK'
  /home/woori/venvs/seka_env/bin/python reports/facet_rft_2026/freeze.py --off --tag t7350 || true
  echo '[t7350] ALL DONE'
" </dev/null >"$LOG/bank_t7350_chain.log" 2>&1 &
say "기동 PID=$! · sha=$SHA · 스모크 2 → 본런 40 sim · 로그 $LOG/bank_t7350_chain.log"
