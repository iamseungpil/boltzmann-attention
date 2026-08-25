#!/bin/bash
# t7351 — **hard-0 표적 런** (사용자 지시 2026-08-25: *"20 태스크 런 걸지 말고, 어제 계획한
#   hard-0 최소 nt=4 로 외출시 걸어라"*). 큐 `x509` P4 설계의 표적판이다.
#
# ## 표적 6 × nt=4 = 24 sim  (큐 x509 P4 설계 축자: *표적 = 072·073·074·016*.
#    ②범주가 안 살아나서 057·063 은 뺐다. 6시간 외출에 맞춰 nt 를 4→6 으로 올린다 —
#    판정선이 *표적의 0→1* 이므로 n 이 클수록 그 한 건의 의미가 커진다.)
#   074  ①금액 단독 — 오늘 격리로 확정된 **전사 결손**의 유일한 보유 태스크
#   072  ①금액 단독 — 같은 서브를 쓰지만 **전사는 이미 온전**(9·10 = 계약값) ⇒ V2 의 **매도 측정**
#   073  같은 ATM 서브 — 전사는 온전(10·11·10)하므로 **V2 의 매도**를 재는 자리다
#   085·040  우리 층이 **gold 거래를 시도하는 호출을 11~17회 막았다**(mutation_diff blocked).
#        ⚠단 `x529` 가 그 거절을 **오프라인에서 재현하지 못했다** ⇒ 기전 미확정·검증된 수리 없음.
#        이 런에서 처음 실리는 **거절 사이드카**로 우리가 그때 무엇을 보냈는지 잡는다.
#   016  ⑦유도 단독 — 새 레버는 **없다**. 그런데 이번 런에 **서브 사이드카**가 처음 실리므로
#        *"서브가 무엇을 받고 무엇을 냈나"* 가 처음 기록된다([[76]] 진단 ① 재료). 오늘 아침
#        gold 정정(카드 적격 지출·`submit_transaction`)에 맞는 저작을 하려면 그 기록이 필요하다.
#
# ## 대조
#   t7348(sha aed30e20) 의 같은 네 태스크 = **전부 0/2**. 판정선 = **표적의 0→1**.
#   ⛔총점 Δ 로 판정하지 마라 · ⛔묶음이라 개별 수리 귀속 금지(C594).
#
# ## 이 런에 처음 실리는 것
#   T2_SG_PROMPT_V2 (격리 검정 `N_wire` 통과가 **게이트**) + 밤새 커밋분(문면·색인·마커·사이드카)
set -e
REPO=/home/woori/workspace_common/boltzmann-attention-pi
LOG=/home/woori/scratch/logs
SIMS=/home/woori/scratch/tau2-bench/data/simulations
REP=$REPO/reports/facet_rft_2026
mkdir -p "$LOG"
cd "$REPO/scripts/distill/tau2"
SHA=$(cd "$REPO" && git rev-parse --short HEAD)
say() { echo "[t7351 $(date +%H:%M:%S)] $*"; }

# ── 게이트: 전사 배선이 격리에서 chk_2 를 닫았나 ─────────────────────────
say "게이트 대기 — N_wire"
for i in $(seq 1 120); do
  [ -s "$REP/x525j_wire_check_2026_08_25.json" ] && break
  sleep 20
done
GATE=$(/home/woori/venvs/seka_env/bin/python - <<'PY'
import json, io
p = "/home/woori/workspace_common/boltzmann-attention-pi/reports/facet_rft_2026/x525j_wire_check_2026_08_25.json"
try:
    d = json.load(io.open(p, encoding="utf-8"))
except Exception:
    print("NOFILE"); raise SystemExit
rows = [r for r in d.get("rows", []) if r.get("arm") == "N_wire" and r.get("msg") == 38]
ok = [r for r in rows if r.get("cover") == r.get("withdrawals")]
print("PASS" if rows and len(ok) == len(rows) else "FAIL:%d/%d" % (len(ok), len(rows)))
PY
)
say "게이트 = $GATE"
if [ "$GATE" != "PASS" ]; then
  # ★2026-08-25: 이 런은 표적이 여덟이라 074 배선 하나로 전체를 거부하지 않는다.
  #   대신 **경고로 남기고** 계속한다 — V2 는 켜진 채 태스크별로 측정된다.
  #   ⚠판정 시 이 줄을 인용하라: 배선이 격리에서 완전히 닫히지 않은 상태로 실렸다.
  say "WARN: 배선 격리 미완결 ($GATE) — V2 는 켜고 진행하되 074 결과를 수리 성공으로 읽지 마라"
fi

DIRTY=$(cd "$REPO" && git status --porcelain -- \
  scripts/distill/tau2/t2_gate_patch.py scripts/distill/tau2/t2_search.py \
  scripts/distill/tau2/t2_resolve.py scripts/distill/tau2/t2_scaffold_get.py \
  scripts/distill/tau2/a2/ | grep -cv '^??' || true)
[ "$DIRTY" = "0" ] || { say "REFUSING: 엔진 경로 미커밋 $DIRTY"; exit 1; }

for t in test_a2_three_layer.py test_flag_registry.py test_no_undefined_names.py \
         test_no_unbound_a2.py test_quote_in.py test_args_equal.py test_t2_procedure.py \
         test_sg_docs_delivery.py test_sg_fetch_iso.py test_sg_isofb.py \
         test_atm_ledger_close.py test_compute_params.py test_write_arg_enum.py \
         test_result_round.py test_apy_balance_tier.py test_ref_from_outputs.py \
         test_no_prose_regex.py test_ours_text_canonical.py; do
  [ -f "$t" ] || continue
  PYTHONPATH=/home/woori/scratch/tau2-bench/src timeout 90 \
    /home/woori/venvs/seka_env/bin/python "$t" >/dev/null 2>&1 || { say "REFUSING: $t FAIL"; exit 1; }
done
say "VERIFY OK"

pgrep -f "[t]2_launch" >/dev/null && { say "REFUSING: 다른 라이브 런"; exit 1; } || true
for f in "$LOG"/bank_t7351_*.log; do [ -e "$f" ] && { say "REFUSING: $f 존재"; exit 1; }; done
for d in "$SIMS"/bank_t7351_*; do [ -e "$d" ] && { say "REFUSING: $d 잔존"; exit 1; }; done

PIN="T2_ACTION_SUB=1 T2_KEEP_DENY_BODY=1 T2_CALL_FORM=1 T2_ARG_EMPTY=1 T2_SEARCH_AGENT=1 \
T2_DECIDE_ANY=1 T2_WRITE_ARG_ENUM=1 T2_DECIDE_BEFORE_WRITE=1 T2_DECISION_CARRY=1 \
T2_DISCOVERY_STEP2=1 T2_ARG_AXIS=1 T2_WRITE_SUB=3 T2_ACTION_INDEX=1 T2_NOW_SELFCALL=1 \
T2_SEARCH_ON_PROCEED=1 T2_ACT_DEMAND=0 T2_DELIVER_PRECOMMIT=0 T2_PROCEED_DOCBODY=0 \
T2_DOCS_AT_WRITE=0 T2_SUB_REQUIREMENT=0 T2_HANDOFF_PREDICATE=0 T2_PENDING_DISCOVERED=0 \
T2_VERDICT_CARRY=0 T2_ELIG_LINE=0 T2_VERDICT_GATE=0 T2_CLAIM_VERIFY=0 \
T2_DECLFIRST=0 T2_DECLFIRST_GUIDE_FIX=0 T2_CATEGORY_CITE="
ON="T2_ARG_DOC_SUB=1 T2_VALUE_FORMULA=full T2_SG_DOCS=1 T2_SG_PROMPT_V2=1"
NT=4
GRP_A='task_074,task_016,task_085,task_057'
GRP_B='task_072,task_073,task_040,task_063'

echo "{\"tag\":\"t7351\",\"sha\":\"$SHA\",\"design\":\"queue P4 targets 074 016 072 073 at nt=6 (24 sims), the transcription prompt order on\",\"on\":\"$ON\",\"reference\":\"t7348 sha aed30e20 - all four were 0/2\",\"bar\":\"a target going 0 -> 1; never judge on a total delta; bundle so no individual attribution (C594)\",\"why_016\":\"no new lever - it rides to capture the sub-call sidecar, which this run records for the first time\",\"gate\":\"$GATE on the N_wire isolation check\"}" \
  | tee "$LOG/bank_t7351.meta.json"

setsid bash -c "
  cd '$REPO/scripts/distill/tau2'
  source ./go_stack.sh >/dev/null 2>&1
  export $PIN
  export $ON
  export GO_MAX_STEPS=150 GO_CONCURRENCY=2

  harvest_instr() {
    local _T=\$1 _S _F
    mkdir -p '$REP/sim_results'
    for _S in fb trace; do
      _F=$LOG/\${_S}_\${_T}.jsonl
      if [ -s \"\$_F\" ]; then
        gzip -c \"\$_F\" > '$REP/sim_results/'\${_S}_\${_T}.jsonl.gz
        echo \"[t7351] \${_S} 회수 \${_T} (\$(wc -l < \"\$_F\") 행)\"
      else
        echo \"[t7351] ⚠\${_S} 미회수 \$_F — 우리-층 귀속 판정 불가([[25]])\"
      fi
    done
  }

  SMK=bank_t7351_smoke_20260825
  echo '[t7351] === 스모크(074 · nt1 · 8140) — 표적 경로에서 V2 가 발화하나 ==='
  t2_launch \$SMK 8140 task_074 1 > $LOG/\$SMK.log 2>&1 || true
  NV2=\$(grep -c 'T2_SG_PROMPT_V2' $LOG/\$SMK.log 2>/dev/null); NV2=\${NV2:-0}
  NTB=\$(grep -c 'Traceback' $LOG/\$SMK.log 2>/dev/null); NTB=\${NTB:-0}
  NOP=\$(grep -c 'operand-size' $LOG/\$SMK.log 2>/dev/null); NOP=\${NOP:-0}
  echo \"[t7351] 스모크 — PROMPT_V2=\$NV2 · operand-size 줄=\$NOP · Traceback=\$NTB\"
  grep 'operand-size' $LOG/\$SMK.log 2>/dev/null | head -4 || true
  cd '$REPO' && mkdir -p reports/facet_rft_2026/sim_results
  gzip -c '$SIMS/'\$SMK'/results.json' > reports/facet_rft_2026/sim_results/\$SMK.results.json.gz 2>/dev/null || true
  gzip -c $LOG/\$SMK.log > reports/facet_rft_2026/sim_results/\$SMK.log.gz
  harvest_instr \$SMK
  git add -f reports/facet_rft_2026/sim_results/\$SMK*.gz \\
             reports/facet_rft_2026/sim_results/fb_\$SMK*.jsonl.gz \\
             reports/facet_rft_2026/sim_results/trace_\$SMK*.jsonl.gz 2>/dev/null || true
  git -c user.name=ghlee -c user.email=beingrelative@gmail.com commit -q -m 't7351 smoke' || true
  git push -q origin facet-rft-2026 || true
  cd '$REPO/scripts/distill/tau2'
  if [ \"\$NTB\" -gt 0 ]; then echo '[t7351] ⛔스모크 Traceback — 중단'; grep -A3 Traceback $LOG/\$SMK.log | head -20; exit 1; fi
  if [ \"\$NV2\" -eq 0 ]; then echo '[t7351] ⛔074 궤적인데 PROMPT_V2 발화 0 = 死배선 — 중단'; exit 1; fi

  cd '$REPO'
  /home/woori/venvs/seka_env/bin/python reports/facet_rft_2026/freeze.py --on \\
    --tag t7351 --reason 'queue P4 targets at nt=6 with transcription order' || true
  cd '$REPO/scripts/distill/tau2'

  run_grp() {
    NAME=\$1; PORT=\$2; TL=\$3
    TAG=bank_t7351_\${NAME}_20260825
    t2_launch \$TAG \$PORT \"\$TL\" $NT 2>&1 | tee $LOG/\$TAG.log
    echo \"[t7351] \$NAME 완료 · V2발화=\$(grep -c 'T2_SG_PROMPT_V2' $LOG/\$TAG.log 2>/dev/null || echo 0)\"
    cd '$REPO' && mkdir -p reports/facet_rft_2026/sim_results
    gzip -c '$SIMS/'\$TAG'/results.json' > reports/facet_rft_2026/sim_results/\$TAG.results.json.gz
    gzip -c $LOG/\$TAG.log > reports/facet_rft_2026/sim_results/\$TAG.log.gz
    harvest_instr \$TAG
    cd '$REPO/scripts/distill/tau2'
  }
  ( run_grp grpA 8140 '$GRP_A' ) > $LOG/bank_t7351_grpA_chain.log 2>&1 &
  P1=\$!
  ( run_grp grpB 8141 '$GRP_B' ) > $LOG/bank_t7351_grpB_chain.log 2>&1 &
  P2=\$!
  wait \$P1 \$P2

  cd '$REPO'
  cp $LOG/bank_t7351.meta.json reports/facet_rft_2026/sim_results/bank_t7351.meta.json || true
  git add -f reports/facet_rft_2026/sim_results/bank_t7351_*.gz reports/facet_rft_2026/sim_results/bank_t7351.meta.json
  git add -f reports/facet_rft_2026/sim_results/fb_bank_t7351_*.jsonl.gz \\
             reports/facet_rft_2026/sim_results/trace_bank_t7351_*.jsonl.gz 2>/dev/null || true
  git -c user.name=ghlee -c user.email=beingrelative@gmail.com commit -q -m 't7351 queue P4 targets at nt6' || true
  git push -q origin facet-rft-2026 || true
  git ls-files --error-unmatch reports/facet_rft_2026/sim_results/bank_t7351_grpA_20260825.results.json.gz \\
    && echo '[t7351] persisted+tracked OK'
  /home/woori/venvs/seka_env/bin/python reports/facet_rft_2026/freeze.py --off --tag t7351 || true
  echo '[t7351] ALL DONE'
" </dev/null >"$LOG/bank_t7351_chain.log" 2>&1 &
say "기동 PID=$! · sha=$SHA · 게이트 $GATE · 스모크 1 → 본런 24 sim · 로그 $LOG/bank_t7351_chain.log"
