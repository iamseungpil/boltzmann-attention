#!/bin/bash
# t7348 — **수리 묶음 본런**. 20 태스크 × nt2 = 40 sim. 대조 = **t7348 13/40**.
#
# ## 무엇을 재나 — t7348 이후 라이브에 들어간 것 전부
#   t7348 은 sha `ee18d797` 에서 돌았다. 그 뒤 엔진·A2 에 **9 커밋**이 들어갔고 어느 것도
#   라이브에서 측정된 적이 없다:
#     `d93c389a` R1~R8 수리 묶음(자[尺]·사이드카 권위·GROUND selector·enum cap fail-open·
#                ARG_EMPTY 死배선·ratefix 문면·DECIDE-FIRST 축 명명·PENDING_DISC 제거)
#     `ad76fdf2` A-1 재생성 호출도 절차 게이트를 받는다 (`T2_PROC_REGEN`)
#     `73efa6f7` A-7 계기 6종
#     `e78ee2f3`·`3bff2409` ATM 비교기 — 옛 op 4/9 → 새 op 9/9 · ⚠**라이브 효과 미측정**
#   ⇒ 라이브 표면 diff = t2_gate_patch +1,006 / A2 3 파일 ×109. **묶음이다.**
#
# ## 그래서 이 런이 **못 하는 것**
#   ⛔**묶음 Δ 로 개별 수리를 주장하지 않는다**(C594 실증). 이 런이 답하는 것은 *"어제 묶음이
#     성적을 샀나"* 하나뿐이고, 귀속은 per-task 포렌식이 한다([[08]]·`per_task_forensics`).
#   ⛔채점은 **reward 뿐**([[69]]). gold 일치율·action_match 는 진단 보조지 성적이 아니다.
#   ⛔**004 는 회귀가 아니다** — 기저율 22/71 ≈ 31% 의 불안정 태스크이고 `reason` enum 한 칸이
#     성적을 정하며 그 칸은 전달로 안 닫힌다(C597~C599·격리 192 재생 0/24). 이 런의 004 결과
#     하나로 회귀도 수리도 주장하지 말고, `reason` 값만 로그에서 **그대로 기록**하라.
#   ⛔**005·102 는 분모에서 제외**된 벤치 결함이고 **069 는 표적 금지**다([[68]]).
#
# ## 판정선
#   대조 t7348 **13/40**(같은 러너 계보·로스터·PIN·ON — sha 만 다르다). Δ ≥ 4/40.
#   **태스크별 부호표 필수**([[70]]) — 합이 null 이어도 태스크별 부호는 갈린다.
#   ⚠엄밀 A/B 아님([M]): 묶음이 9 커밋이다.
#
# ## 이번에 특히 볼 자리 (수리분이 **라이브에서 발화하나** — 死배선에 돈 쓰지 않는다)
#   ★`[T2_PROCEDURE] regen-*`   A-1 이 실제로 재평가했나 (t7348 = 0 줄)
#   ★`[T2_ARG_EMPTY] deny`      R5 死배선 수리 후 discoverable write 에 닿나 (전 코퍼스 79 발화)
#   ★`[T2_WRITE_ARG_ENUM] deny` R4 cap 이 *처음 보는 값*만 세는가 (t7348 = 18)
#   ★`fb_`·`trace_` 회수         R2 — 2026-08-24 에 366 개가 리모트 디스크에만 있던 것을 건졌다.
#                                이 런은 **끝나자마자** 회수한다(harvest_instr).
#   073·050·085·040#1           수리 묶음 표적 · F8 부활(t7336 = 0)
#
# ## 스모크 게이트 ([[30]] 死배선에 돈 금지) — t7348 것을 하나도 빼지 않고 이어받았다
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
  echo "[t7348] REFUSING: 엔진 경로 커밋 안 된 변경 $DIRTY 개." >&2; exit 1
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
    || { echo "[t7348] REFUSING: $t FAIL" >&2; exit 1; }
done
echo "[t7348] VERIFY OK (배터리 40 - R1~R8·A-1 검정 9 등재)"

if pgrep -f "[t]2_launch" >/dev/null; then
  echo "[t7348] REFUSING: 다른 라이브 런이 돌고 있다" >&2; exit 1
fi
for f in "$LOG"/bank_t7348_*.log; do
  [ -e "$f" ] && { echo "[t7348] REFUSING: $f 존재" >&2; exit 1; }
done
for d in "$SIMS"/bank_t7348_*; do
  [ -e "$d" ] && { echo "[t7348] REFUSING: $d 잔존" >&2; exit 1; }
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

echo "{\"tag\":\"t7348\",\"sha\":\"$SHA\",\"design\":\"repair-bundle main run - same runner lineage, roster, PIN and ON as t7346; nine commits of engine and A2 landed since t7346 sha ee18d797 and none has been measured live\",\"on\":\"$ON\",\"tasks\":\"stage1 20 x nt=$NT = 40 sims\",\"endpoint\":\"reward only\",\"reference\":\"t7346 13/40 - byte-identical runner except tag and battery; engine differs by R1-R8 (d93c389a), A-1 proc-regen recheck (ad76fdf2), A-7 instruments (73efa6f7) and the ATM comparator (e78ee2f3, 3bff2409) whose live effect is unmeasured\",\"bar\":\"delta >= 4/40 vs reference; per-task sign table mandatory\",\"caveat\":\"bundle of nine - a bundle delta cannot attribute an individual repair (C594); task_004 is a 31 percent coin flip (22/71) with a boundary-recorded reason enum (C597-C599), so record its reason value and do not read one outcome as regression or repair\"}" \
  | tee "$LOG/bank_t7348.meta.json"

setsid bash -c "
  cd '$REPO/scripts/distill/tau2'
  source ./go_stack.sh >/dev/null 2>&1
  export $PIN
  export $ON
  export GO_MAX_STEPS=150 GO_CONCURRENCY=1

  # ── ★계기 회수 (2026-08-23 R2) — 한 자리에만 둔다 ────────────────────────
  #   go_stack:222/231 이 \`T2_FB_SIDECAR\`·\`T2_TRACE\` 를 **모든 라이브 런에** 켜는데,
  #   stage-1 러너 계보는 results/log 만 회수해 왔다. 그래서 사이드카 보관은 t7328 이
  #   마지막이고 t7336·t7348 은 **없다**. 그게 왜 치명적인가: 우리 층 거절은 재생성
  #   채널로 나가고 \`_ap_regen\` 이 원 어시스턴트 메시지를 **교체**하므로 막힌 호출은
  #   영속 궤적에도 \`mutation_diff\` 의 BLOCKED 칸에도 안 남는다. 남는 자리는 사이드카
  #   뿐이다 — 그것이 없으면 분석자가 그 공백을 *\"우리 표지가 없으니 env 가 했다\"* 로
  #   읽는다([[25]]). 실제로 그 아티팩트 위에 반증 셋이 세워졌다(refute_1⑷⑸·4⑵·6⑶).
  #   ⛔없으면 **없다고 인쇄한다** — 조용히 넘어가면 같은 오독이 재생산된다.
  #   스모크와 본런이 갈리면 또 한쪽만 회수하므로(2026-08-06 사이드카 사고와 같은 뿌리)
  #   함수 하나로 둔다([[67]]).
  harvest_instr() {
    local _T=\$1 _S _F
    mkdir -p '$REPO/reports/facet_rft_2026/sim_results'
    for _S in fb trace; do
      _F=$LOG/\${_S}_\${_T}.jsonl
      if [ -s \"\$_F\" ]; then
        gzip -c \"\$_F\" > '$REPO/reports/facet_rft_2026/sim_results/'\${_S}_\${_T}.jsonl.gz
        echo \"[t7348] \${_S} 회수 \${_T} (\$(wc -l < \"\$_F\") 행)\"
      else
        echo \"[t7348] ⚠\${_S} 미회수: \$_F 없음/빈 파일 — 이 런의 우리-층 귀속은 **판정 불가**다([[25]]·[[55]])\"
      fi
    done
  }

  # ── 스모크 (2 sim · **두 GPU 병렬**) ───────────────────────────────────
  # ★2026-08-22 사용자 지시(축자): 스모크에도 2개 gpu 다 사용하라. 각각 1개 gpu 사용하라. 시간줄여라.
  #   구판은 8141 하나로 두 태스크를 **순차** 실행해 스모크가 본런만큼 길어졌다(093 34분 + 024 13분).
  #   두 팔은 GPU 가 다르므로 병렬이어도 vLLM prefix 캐시를 서로 밀어내지 않는다 — 본런의
  #   halfA/halfB 와 **같은 패턴**이고, 러너 안 동시성(GO_CONCURRENCY=1)은 그대로다([[30]] 동시성 지시).
  #   ⇒ 스모크 벽시계 = max(093, 024) ≈ 절반 이하.
  #   게이트 문구는 **합본 로그**에서 종전 그대로 평가한다(판정 로직 불변).
  SMK=bank_t7348_smoke_20260824
  SMKA=\${SMK}_a
  SMKB=\${SMK}_b
  echo '[t7348] === 스모크(093→8140 · 024→8141 · nt=1 · 병렬) ==='
  ( t2_launch \$SMKA 8140 task_093 1 ) > $LOG/\$SMKA.log 2>&1 &
  SPA=\$!
  ( t2_launch \$SMKB 8141 task_024 1 ) > $LOG/\$SMKB.log 2>&1 &
  SPB=\$!
  wait \$SPA \$SPB
  cat $LOG/\$SMKA.log $LOG/\$SMKB.log > $LOG/\$SMK.log
  NV=\$(grep -c '$M_VAL' $LOG/\$SMK.log 2>/dev/null); NV=\${NV:-0}
  ND=\$(grep -c '$M_DOCS' $LOG/\$SMK.log 2>/dev/null); ND=\${ND:-0}
  NC=\$(grep '$APYTOOL' $LOG/\$SMK.log 2>/dev/null | grep -v 'injected' | wc -l)
  echo \"[t7348] 스모크 발화 — 값주석=\$NV · docs전달=\$ND · apy도구 언급=\$NC\"
  grep '$M_DOCS' $LOG/\$SMK.log 2>/dev/null | head -4 || true
  cd '$REPO' && mkdir -p reports/facet_rft_2026/sim_results
  for _P in \$SMKA \$SMKB; do
    gzip -c '$SIMS/'\$_P'/results.json' > reports/facet_rft_2026/sim_results/\$_P.results.json.gz
    harvest_instr \$_P
  done
  gzip -c $LOG/\$SMK.log > reports/facet_rft_2026/sim_results/\$SMK.log.gz
  git add -f reports/facet_rft_2026/sim_results/\$SMK*.gz \\
             reports/facet_rft_2026/sim_results/fb_\$SMK*.jsonl.gz \\
             reports/facet_rft_2026/sim_results/trace_\$SMK*.jsonl.gz 2>/dev/null || true
  git -c user.name=ghlee -c user.email=beingrelative@gmail.com commit -q -m 't7348 smoke' || true
  git push -q origin facet-rft-2026 || true
  cd '$REPO/scripts/distill/tau2'
  if [ \"\$NV\" -eq 0 ]; then
    echo '[t7348] ⛔값 주석 발화 0 — 본런을 돌리지 않는다'
    exit 1
  fi
  if [ \"\$NC\" -gt 0 ] && [ \"\$ND\" -eq 0 ]; then
    echo '[t7348] ⛔apy 도구가 불렸는데 T2_SG_DOCS 발화 0 = 死배선 — 본런을 돌리지 않는다'
    exit 1
  fi
  [ \"\$NC\" -eq 0 ] && echo '[t7348] ⚠apy 도구 자체가 안 불림 — docs 게이트 판단 불가(관측으로 남긴다)'

  # ── 오늘 수리분: 관측 계수(게이트 아님) ────────────────────────────────
  NF8=\$(grep -c 'T2_ARG_PRODUCERS. fired' $LOG/\$SMK.log 2>/dev/null); NF8=\${NF8:-0}
  NSN=\$(grep -c 'T2_STALE_NOTE' $LOG/\$SMK.log 2>/dev/null); NSN=\${NSN:-0}
  NEN=\$(grep -c 'T2_WRITE_ARG_ENUM. deny' $LOG/\$SMK.log 2>/dev/null); NEN=\${NEN:-0}
  echo \"[t7348] 수리분 관측 - F8 발화=\$NF8 · STALE_NOTE=\$NSN · ENUM deny=\$NEN\"

  # ── ⛔死배선 게이트: 오늘 수리분이 예외로 조용히 죽으면 F8 이 또 침묵한다 ──
  NDEAD=\$(grep -cE 'arg-producer skipped|Traceback' $LOG/\$SMK.log 2>/dev/null); NDEAD=\${NDEAD:-0}
  if [ \"\$NDEAD\" -gt 0 ]; then
    echo '[t7348] ⛔오늘 수리분이 예외로 죽었다(arg-producer skipped / Traceback) - 본런을 돌리지 않는다'
    grep -E 'arg-producer skipped|Traceback' $LOG/\$SMK.log | head -5
    exit 1
  fi

  NTIER=\$(grep -c 'current_balance' $LOG/\$SMK.log 2>/dev/null); NTIER=\${NTIER:-0}
  echo \"[t7348] 잔액인자 발화=\$NTIER (0 이면 에이전트가 안 채운 것 - 종전 거동 폴백)\"
  NRD=\$(grep -c 'T2_SG_ROUND' $LOG/\$SMK.log 2>/dev/null); NRD=\${NRD:-0}
  NWEV=\$(grep -c 'T2_WRITE_EVIDENCE. deny' $LOG/\$SMK.log 2>/dev/null); NWEV=\${NWEV:-0}
  NM1=\$(grep -c 'principal=-1' $LOG/\$SMK.log 2>/dev/null); NM1=\${NM1:-0}
  echo \"[t7348] 반올림=\$NRD · WEV deny=\$NWEV (t7341=10) · 서브 -1 폐기=\$NM1 (미해결 표적)\"

  # ── ⛔자리표시자 수리 게이트: 서브가 예시 0.0 을 또 복사하면 중단 ──────────
  N00=\$(grep -c '부재(principal=0.0' $LOG/\$SMK.log 2>/dev/null); N00=\${N00:-0}
  echo \"[t7348] 격리 서브 0.0-복사 폐기 = \$N00 (0 이어야 수리가 먹은 것)\"
  if [ \"\$N00\" -gt 0 ]; then
    echo '[t7348] ⛔격리 서브가 answer_format 예시값(0.0)을 또 복사했다 - 자리표시자 수리 미적용 - 본런을 돌리지 않는다'
    grep '부재(principal=0.0' $LOG/\$SMK.log | head -3
    exit 1
  fi

  # ── ⛔문법 死배선 게이트: 격리 서브가 돌았는데 문법이 안 걸렸으면 중단 ──────
  NISO=\$(grep -c 'SG_ISOLATE. fetch' $LOG/\$SMK.log 2>/dev/null); NISO=\${NISO:-0}
  NSCH=\$(grep -c 'T2_SG_SCHEMA' $LOG/\$SMK.log 2>/dev/null); NSCH=\${NSCH:-0}
  echo \"[t7348] 격리 서브 fetch=\$NISO · 문법 적용=\$NSCH\"
  if [ \"\$NISO\" -gt 0 ] && [ \"\$NSCH\" -eq 0 ]; then
    echo '[t7348] ⛔격리 서브가 돌았는데 T2_SG_SCHEMA 발화 0 = 문법 死배선 - 본런을 돌리지 않는다'
    exit 1
  fi

  # ── ⛔누수 재발 게이트: 후보 명단에 General 이 실리면 중단 ────────────────
  if grep -q ', General ,' $LOG/\$SMK.log 2>/dev/null; then
    echo '[t7348] ⛔WRITE_ARG_ENUM 후보 명단에 General 재출현 - 본런을 돌리지 않는다'
    exit 1
  fi

  # ── 동결 ([[07]]·스모크 뒤) ────────────────────────────────────────────
  cd '$REPO'
  /home/woori/venvs/seka_env/bin/python reports/facet_rft_2026/freeze.py --on \\
    --tag t7348 --reason 'all-on composed stack, stage1 20 x nt2' || true
  cd '$REPO/scripts/distill/tau2'

  run_half() {
    NAME=\$1; PORT=\$2; TL=\$3
    TAG=bank_t7348_\${NAME}_20260824
    t2_launch \$TAG \$PORT \"\$TL\" $NT 2>&1 | tee $LOG/\$TAG.log
    echo \"[t7348] \$NAME 완료 · docs발화=\$(grep -c '$M_DOCS' $LOG/\$TAG.log 2>/dev/null || echo 0) · 값=\$(grep -c '$M_VAL' $LOG/\$TAG.log 2>/dev/null || echo 0)\"
    cd '$REPO' && mkdir -p reports/facet_rft_2026/sim_results
    gzip -c '$SIMS/'\$TAG'/results.json' > reports/facet_rft_2026/sim_results/\$TAG.results.json.gz
    gzip -c $LOG/\$TAG.log > reports/facet_rft_2026/sim_results/\$TAG.log.gz
    harvest_instr \$TAG
    cd '$REPO/scripts/distill/tau2'
  }
  ( run_half halfA 8140 '$HALF_A' ) > $LOG/bank_t7348_halfA_chain.log 2>&1 &
  P1=\$!
  ( run_half halfB 8141 '$HALF_B' ) > $LOG/bank_t7348_halfB_chain.log 2>&1 &
  P2=\$!
  wait \$P1 \$P2

  # ── 영속 + 동결 해제 ([[30]] tracked 확인까지) ─────────────────────────
  cd '$REPO'
  cp $LOG/bank_t7348.meta.json reports/facet_rft_2026/sim_results/bank_t7348.meta.json || true
  git add -f reports/facet_rft_2026/sim_results/bank_t7348_*.gz reports/facet_rft_2026/sim_results/bank_t7348.meta.json
  # ★계기도 함께 추적 — 위 glob 은 \`fb_bank_…\`·\`trace_bank_…\` 를 안 잡는다(접두가 다르다)
  git add -f reports/facet_rft_2026/sim_results/fb_bank_t7348_*.jsonl.gz \\
             reports/facet_rft_2026/sim_results/trace_bank_t7348_*.jsonl.gz 2>/dev/null || true
  git -c user.name=ghlee -c user.email=beingrelative@gmail.com commit -q -m 't7348 all-on stage1 results' || true
  git push -q origin facet-rft-2026 || true
  git ls-files --error-unmatch reports/facet_rft_2026/sim_results/bank_t7348_halfA_20260824.results.json.gz \\
    && echo '[t7348] persisted+tracked OK'
  /home/woori/venvs/seka_env/bin/python reports/facet_rft_2026/freeze.py --off --tag t7348 || true
  echo '[t7348] ALL DONE'
" </dev/null >"$LOG/bank_t7348_chain.log" 2>&1 &
echo "[t7348] 기동 PID=$! · sha=$SHA · 스모크 2 → 본런 40 sim (halfA 8140 · halfB 8141) · 로그 $LOG/bank_t7348_chain.log"
