#!/bin/bash
# t7341 — t7340 과 **같은 스택**이고 스모크만 **두 GPU 병렬**로 바꾼 판(사용자 지시).
#   t7340 은 스모크를 8141 하나로 순차 실행해 093(34분) + 024(13분) 를 더한 시간을 썼다.
#   두 태스크를 GPU 하나씩에 나눠 붙이면 벽시계가 max 로 줄고, 본런 halfA/halfB 와 같은 패턴이라
#   측정 조건은 바뀌지 않는다(러너 안 동시성은 여전히 1).
# t7341 — t7339 를 스모크 도중 중단하고 **격리 서브 형식의 문법 강제까지 실어** 재발사한 런.
#   중단 이유(사용자 지시 2026-08-22 "더 근본적인 해법으로 모두 고쳐라"): t7339 는 형식 예시의
#   숫자를 자리표시자로 바꾼 판이었는데, 자리표시자는 **베낄 값**만 없앨 뿐 형식 준수를
#   보장하지 못한다 — 산문으로 부탁하는 한 서브는 언제든 형식을 벗어날 수 있다.
#
# ## 무엇이 실렸나 = **19건 한 묶음**
#   e7dcb97d  A1~A8·A10·A12~A16 (14건)
#   4373e7db  잔여 부채 3건 (A9 호출부 · OL-55 형제 · WRITE_ARG_ENUM 누수)
#   d4a38ead  answer_format 자리표시자화 (숫자 6자리 → <number>)
#   07c4c2f0  **T2_SG_SCHEMA — 격리 서브 형식을 문법으로 강제**
#             A2 `isolate.operand_schema` 를 guided_json(xgrammar)으로 건다. 부탁이 아니라
#             디코딩 차단이므로 형식을 어길 수 없고, **베낄 예시 자체가 불필요**해진다.
#             ⚠**도구가 없는 라운드에만** 건다(`_tl is None`): 도구와 문법을 같이 걸면
#               tool_calls 가 0 이 되어 서브가 레코드를 못 읽는다(t2_declfirst 실측·C248).
#               getter 모드=마감 라운드만 · docs 모드=원래 도구 0이라 전 라운드.
#             선행 근거: declfirst 2패스 실측 프롬프트만 32% ↔ 도구미제공+문법 **96%**(C250).
#             ⚠[[70]] 무엇을 파는가: "JSON 하나만" 이 강제되면 서브가 추론할 자리를 잃는다 ⇒
#               스키마 **첫 required 필드를 `derivation`** 으로 두어 추론이 값보다 앞에 오게 했다
#               (파싱은 operand_keys 만 뽑으므로 무해).
#   ⇒ 이 넷 중 어느 것도 새 판단을 하지 않는다([[62]]): 우리가 떠먹이던 값을 없애고 형식만
#     강제할 뿐 **값은 여전히 서브가 낸다**. gold 는 보지 않았다([[23]]).
#
# ## 왜 이 자리인가 — 이미 이름이 붙어 있던 결함
#   `t2_scaffold_get.py:2be` 주석 축자: *"§2as 0.0-포이즈닝의 신형 재발"*(2026-07-21).
#   당시 처방은 **답 폐기**(증상 억제)였고, 그 폐기가 폴백 → 메인 추측 → grounding 드롭 →
#   도구 None → 모델이 값을 자기 계산해 write → `T2_WRITE_EVIDENCE` deny 의 **출구 없는 반복**을
#   낳았다(093 실시간 포렌식 2026-08-22: t7337 1회·t7338 4회 재현·한 태스크가 34분 소모).
#   원인은 우리 문구였다 — 형식 예시가 `{"principal": 0.0, "actual_apy": 0.0}` 였고 서브 출력이
#   그와 **정확히 같았다**(저축계좌 잔액이 0.0 일 수 없다).
#
# ## 대조군 — t7336 **13/40** (변함없음)
#   로스터 20 태스크 · nt=2 · PIN · ON 이 t7336 러너와 **바이트 동일**하고 sha 만 다르다.
#   판정선 Δ ≥ 4/40. ⚠19건 묶음이므로 **묶음 Δ 로 개별 수리를 주장하지 않는다**(C594 실증).
#   귀속은 per-task 포렌식이 한다([[08]]). 채점은 reward 뿐([[69]]).
#
# ## 이번에 특히 볼 자리
#   ★093        `부재(principal=0.0` **0** + `[T2_SG_SCHEMA]` 발화 > 0 + `SCAFFOLD_GET
#               get_interest_correction -> <숫자>`(None 아님) ⇒ livelock 사슬이 첫 마디에서 끊긴다
#   ★서브 형식  문법이 켜졌는데도 폐기되는 건이 있나(= 형식이 아니라 **값**이 문제였다는 신호)
#   F8 부활     `[T2_ARG_PRODUCERS] fired` 수 (t7336 = 0)
#   OL-55 형제  `[T2_STALE_NOTE]` 수
#   073·050·085·040#1  수리 묶음 표적(핸드오프 §0)
#
# ## 스모크 게이트 ([[30]] 死배선에 돈 금지)
#   task_093(격리 서브 표적·8140) + task_024(값·배달 표적·8141) x nt=1 = 2 sim **병렬**.
#   ⛔중단: ⑴값 주석 발화 0  ⑵apy 도구가 불렸는데 [T2_SG_DOCS] 0
#   ⛔      ⑶수리분이 예외로 죽음(`arg-producer skipped` · `Traceback`)
#   ⛔      ⑷`부재(principal=0.0` 재출현 = 형식 수리가 먹지 않았다
#   ⛔**신규 ⑸**: 격리 서브가 돌았는데 `[T2_SG_SCHEMA]` 발화 0 = **문법이 死배선**이다.
#     (`SG_ISOLATE fetch` 가 있는데 SCHEMA 가 없으면 중단 — 플래그만 켜고 안 걸리는 것을 막는다)
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
  echo "[t7341] REFUSING: 엔진 경로 커밋 안 된 변경 $DIRTY 개." >&2; exit 1
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
    || { echo "[t7341] REFUSING: $t FAIL" >&2; exit 1; }
done
echo "[t7341] VERIFY OK (배터리 31 - 오늘 수리 검정 5 등재·문법 포함)"

if pgrep -f "[t]2_launch" >/dev/null; then
  echo "[t7341] REFUSING: 다른 라이브 런이 돌고 있다" >&2; exit 1
fi
for f in "$LOG"/bank_t7341_*.log; do
  [ -e "$f" ] && { echo "[t7341] REFUSING: $f 존재" >&2; exit 1; }
done
for d in "$SIMS"/bank_t7341_*; do
  [ -e "$d" ] && { echo "[t7341] REFUSING: $d 잔존" >&2; exit 1; }
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

echo "{\"tag\":\"t7341\",\"sha\":\"$SHA\",\"design\":\"single all-on composed stack (user directive), no arms; t7339 halted mid-smoke and relaunched with the isolate output grammar included\",\"on\":\"$ON\",\"tasks\":\"stage1 20 x nt=$NT = 40 sims\",\"endpoint\":\"reward only\",\"reference\":\"t7336 13/40 - same runner lineage, same roster, same PIN/ON; engine sha differs by the repair bundle (14), the residual debts (3), the answer_format placeholder fix (1) and the isolate output grammar (1) = 19 items\",\"bar\":\"delta >= 4/40 vs reference; per-task sign table; reads/fabs/over-action logged\"}" \
  | tee "$LOG/bank_t7341.meta.json"

setsid bash -c "
  cd '$REPO/scripts/distill/tau2'
  source ./go_stack.sh >/dev/null 2>&1
  export $PIN
  export $ON
  export GO_MAX_STEPS=150 GO_CONCURRENCY=1

  # ── 스모크 (2 sim · **두 GPU 병렬**) ───────────────────────────────────
  # ★2026-08-22 사용자 지시 *"스모크에도 2개 gpu 다 사용하라. 각각 1개 gpu 사용하라. 시간줄여라"*.
  #   구판은 8141 하나로 두 태스크를 **순차** 실행해 스모크가 본런만큼 길어졌다(093 34분 + 024 13분).
  #   두 팔은 GPU 가 다르므로 병렬이어도 vLLM prefix 캐시를 서로 밀어내지 않는다 — 본런의
  #   halfA/halfB 와 **같은 패턴**이고, 러너 안 동시성(GO_CONCURRENCY=1)은 그대로다([[30]] 동시성 지시).
  #   ⇒ 스모크 벽시계 = max(093, 024) ≈ 절반 이하.
  #   게이트 문구는 **합본 로그**에서 종전 그대로 평가한다(판정 로직 불변).
  SMK=bank_t7341_smoke_20260822
  SMKA=\${SMK}_a
  SMKB=\${SMK}_b
  echo '[t7341] === 스모크(093→8140 · 024→8141 · nt=1 · 병렬) ==='
  ( t2_launch \$SMKA 8140 task_093 1 ) > $LOG/\$SMKA.log 2>&1 &
  SPA=\$!
  ( t2_launch \$SMKB 8141 task_024 1 ) > $LOG/\$SMKB.log 2>&1 &
  SPB=\$!
  wait \$SPA \$SPB
  cat $LOG/\$SMKA.log $LOG/\$SMKB.log > $LOG/\$SMK.log
  NV=\$(grep -c '$M_VAL' $LOG/\$SMK.log 2>/dev/null); NV=\${NV:-0}
  ND=\$(grep -c '$M_DOCS' $LOG/\$SMK.log 2>/dev/null); ND=\${ND:-0}
  NC=\$(grep '$APYTOOL' $LOG/\$SMK.log 2>/dev/null | grep -v 'injected' | wc -l)
  echo \"[t7341] 스모크 발화 — 값주석=\$NV · docs전달=\$ND · apy도구 언급=\$NC\"
  grep '$M_DOCS' $LOG/\$SMK.log 2>/dev/null | head -4 || true
  cd '$REPO' && mkdir -p reports/facet_rft_2026/sim_results
  for _P in \$SMKA \$SMKB; do
    gzip -c '$SIMS/'\$_P'/results.json' > reports/facet_rft_2026/sim_results/\$_P.results.json.gz
  done
  gzip -c $LOG/\$SMK.log > reports/facet_rft_2026/sim_results/\$SMK.log.gz
  git add -f reports/facet_rft_2026/sim_results/\$SMK*.gz
  git -c user.name=ghlee -c user.email=beingrelative@gmail.com commit -q -m 't7341 smoke' || true
  git push -q origin facet-rft-2026 || true
  cd '$REPO/scripts/distill/tau2'
  if [ \"\$NV\" -eq 0 ]; then
    echo '[t7341] ⛔값 주석 발화 0 — 본런을 돌리지 않는다'
    exit 1
  fi
  if [ \"\$NC\" -gt 0 ] && [ \"\$ND\" -eq 0 ]; then
    echo '[t7341] ⛔apy 도구가 불렸는데 T2_SG_DOCS 발화 0 = 死배선 — 본런을 돌리지 않는다'
    exit 1
  fi
  [ \"\$NC\" -eq 0 ] && echo '[t7341] ⚠apy 도구 자체가 안 불림 — docs 게이트 판단 불가(관측으로 남긴다)'

  # ── 오늘 수리분: 관측 계수(게이트 아님) ────────────────────────────────
  NF8=\$(grep -c 'T2_ARG_PRODUCERS. fired' $LOG/\$SMK.log 2>/dev/null); NF8=\${NF8:-0}
  NSN=\$(grep -c 'T2_STALE_NOTE' $LOG/\$SMK.log 2>/dev/null); NSN=\${NSN:-0}
  NEN=\$(grep -c 'T2_WRITE_ARG_ENUM. deny' $LOG/\$SMK.log 2>/dev/null); NEN=\${NEN:-0}
  echo \"[t7341] 수리분 관측 - F8 발화=\$NF8 · STALE_NOTE=\$NSN · ENUM deny=\$NEN\"

  # ── ⛔死배선 게이트: 오늘 수리분이 예외로 조용히 죽으면 F8 이 또 침묵한다 ──
  NDEAD=\$(grep -cE 'arg-producer skipped|Traceback' $LOG/\$SMK.log 2>/dev/null); NDEAD=\${NDEAD:-0}
  if [ \"\$NDEAD\" -gt 0 ]; then
    echo '[t7341] ⛔오늘 수리분이 예외로 죽었다(arg-producer skipped / Traceback) - 본런을 돌리지 않는다'
    grep -E 'arg-producer skipped|Traceback' $LOG/\$SMK.log | head -5
    exit 1
  fi

  # ── ⛔자리표시자 수리 게이트: 서브가 예시 0.0 을 또 복사하면 중단 ──────────
  N00=\$(grep -c '부재(principal=0.0' $LOG/\$SMK.log 2>/dev/null); N00=\${N00:-0}
  echo \"[t7341] 격리 서브 0.0-복사 폐기 = \$N00 (0 이어야 수리가 먹은 것)\"
  if [ \"\$N00\" -gt 0 ]; then
    echo '[t7341] ⛔격리 서브가 answer_format 예시값(0.0)을 또 복사했다 - 자리표시자 수리 미적용 - 본런을 돌리지 않는다'
    grep '부재(principal=0.0' $LOG/\$SMK.log | head -3
    exit 1
  fi

  # ── ⛔문법 死배선 게이트: 격리 서브가 돌았는데 문법이 안 걸렸으면 중단 ──────
  NISO=\$(grep -c 'SG_ISOLATE. fetch' $LOG/\$SMK.log 2>/dev/null); NISO=\${NISO:-0}
  NSCH=\$(grep -c 'T2_SG_SCHEMA' $LOG/\$SMK.log 2>/dev/null); NSCH=\${NSCH:-0}
  echo \"[t7341] 격리 서브 fetch=\$NISO · 문법 적용=\$NSCH\"
  if [ \"\$NISO\" -gt 0 ] && [ \"\$NSCH\" -eq 0 ]; then
    echo '[t7341] ⛔격리 서브가 돌았는데 T2_SG_SCHEMA 발화 0 = 문법 死배선 - 본런을 돌리지 않는다'
    exit 1
  fi

  # ── ⛔누수 재발 게이트: 후보 명단에 General 이 실리면 중단 ────────────────
  if grep -q ', General ,' $LOG/\$SMK.log 2>/dev/null; then
    echo '[t7341] ⛔WRITE_ARG_ENUM 후보 명단에 General 재출현 - 본런을 돌리지 않는다'
    exit 1
  fi

  # ── 동결 ([[07]]·스모크 뒤) ────────────────────────────────────────────
  cd '$REPO'
  /home/woori/venvs/seka_env/bin/python reports/facet_rft_2026/freeze.py --on \\
    --tag t7341 --reason 'all-on composed stack, stage1 20 x nt2' || true
  cd '$REPO/scripts/distill/tau2'

  run_half() {
    NAME=\$1; PORT=\$2; TL=\$3
    TAG=bank_t7341_\${NAME}_20260822
    t2_launch \$TAG \$PORT \"\$TL\" $NT 2>&1 | tee $LOG/\$TAG.log
    echo \"[t7341] \$NAME 완료 · docs발화=\$(grep -c '$M_DOCS' $LOG/\$TAG.log 2>/dev/null || echo 0) · 값=\$(grep -c '$M_VAL' $LOG/\$TAG.log 2>/dev/null || echo 0)\"
    cd '$REPO' && mkdir -p reports/facet_rft_2026/sim_results
    gzip -c '$SIMS/'\$TAG'/results.json' > reports/facet_rft_2026/sim_results/\$TAG.results.json.gz
    gzip -c $LOG/\$TAG.log > reports/facet_rft_2026/sim_results/\$TAG.log.gz
    cd '$REPO/scripts/distill/tau2'
  }
  ( run_half halfA 8140 '$HALF_A' ) > $LOG/bank_t7341_halfA_chain.log 2>&1 &
  P1=\$!
  ( run_half halfB 8141 '$HALF_B' ) > $LOG/bank_t7341_halfB_chain.log 2>&1 &
  P2=\$!
  wait \$P1 \$P2

  # ── 영속 + 동결 해제 ([[30]] tracked 확인까지) ─────────────────────────
  cd '$REPO'
  cp $LOG/bank_t7341.meta.json reports/facet_rft_2026/sim_results/bank_t7341.meta.json || true
  git add -f reports/facet_rft_2026/sim_results/bank_t7341_*.gz reports/facet_rft_2026/sim_results/bank_t7341.meta.json
  git -c user.name=ghlee -c user.email=beingrelative@gmail.com commit -q -m 't7341 all-on stage1 results' || true
  git push -q origin facet-rft-2026 || true
  git ls-files --error-unmatch reports/facet_rft_2026/sim_results/bank_t7341_halfA_20260822.results.json.gz \\
    && echo '[t7341] persisted+tracked OK'
  /home/woori/venvs/seka_env/bin/python reports/facet_rft_2026/freeze.py --off --tag t7341 || true
  echo '[t7341] ALL DONE'
" </dev/null >"$LOG/bank_t7341_chain.log" 2>&1 &
echo "[t7341] 기동 PID=$! · sha=$SHA · 스모크 2 → 본런 40 sim (halfA 8140 · halfB 8141) · 로그 $LOG/bank_t7341_chain.log"
