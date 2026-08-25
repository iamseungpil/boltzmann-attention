#!/bin/bash
# t7352 — **hard-0 표적 런 · 시간상자판** (사용자 지시 2026-08-25: *"hard-0 최대한 수리해서
#   실험 런해야 한다"* + *"12시부터 6시간 정도 실험할 실험계획을 다시 세우라"*)
#
# ## 왜 t7351 을 그대로 쏘지 않았나 — **6시간에 안 들어간다**
# t7348 per-sim 실측(분·`duration` 필드 직독):
#   016 12.7 · 040 **87.1** · 057 27.0 · 063 15.5 · 072 19.9 · 073 20.4 · 074 39.3 · 085 37.0
# t7351 설계(8 태스크 × nt=4)를 그 표로 곱하면 grpA 464분(7.7h) · grpB 572분(9.5h) 이다.
# 그리고 결과는 **배치 끝에 한 번** 쓰인다(t7348 sim 디렉터리에 results.json 하나뿐)
# ⇒ 시간 초과로 끊으면 **그 그룹 전량 소실**. 그래서 셋을 바꾼다:
#   ① **배치 분할** — 태그가 다른 순차 배치. 배치마다 즉시 영속·커밋·push.
#   ② **마감시각 가드** — 다음 배치는 `지금 + 추정 <= DEADLINE` 일 때만 시작하고,
#      건너뛴 것은 **로그에 남긴다**([[73]] no silent caps).
#   ③ **`GO_CONCURRENCY=1`** — t7351 드라이버의 2 는 [[30]] 의 사용자 지시와 어긋난다.
#
# ## 이 런에 처음 실리는 것
#   `T2_SPEC_AT_WRITE=1` (085·040) — 격리 `x532`: A_asis 1/6 ↔ **B_spec 6/6** ↔ N_neg 2/6.
#        env 가 앞서 보낸 도구 명세를 write 결정점에 **그대로** 되붙인다(거리 46·58 을 0 으로).
#   `T2_SG_PROMPT_V2=1` (074) — 격리 `x525j` N_wire: cover==withdrawals 4/4 창.
#        ⚠rows 는 +1 이 남는다(초과 행 1·원인 미규명) ⇒ 074 결과를 **완전 수리로 읽지 마라**.
#   계기 `[T2_SPEC_DIST]`(모든 write 결정점의 재료 거리) · `[T2_SUBWIN]`(서브가 못 본 구간) ·
#        `[T2_AXIS_CLOBBER]`(같은 축의 결정문 덮어쓰기) — 전부 거동 변경 0·미수리분 진단용.
#
# ## 대조와 판정선
#   대조 = t7348(sha aed30e20) — 이 로스터 **전부 0/2**.
#   판정선 = **표적 태스크의 0→1**. hard-0 는 3런 0/6 이라 1건이 잡음(SD 2.65) 위다.
#   ⛔총점 Δ 로 판정하지 마라 · ⛔묶음이라 개별 귀속 금지(C594) ·
#   ⛔이 런은 `T2_SPEC_AT_WRITE` 가 **무엇을 파는지 못 잰다** — 로스터가 전부 hard-0 라
#     통과 중인 태스크가 하나도 없다([[70]] 매도 측정은 다음 런 몫).
set -e
REPO=/home/woori/workspace_common/boltzmann-attention-pi
LOG=/home/woori/scratch/logs
SIMS=/home/woori/scratch/tau2-bench/data/simulations
REP=$REPO/reports/facet_rft_2026
TAGBASE=bank_t7352
DEADLINE_HHMM=${DEADLINE_HHMM:-1830}
mkdir -p "$LOG"
cd "$REPO/scripts/distill/tau2"
SHA=$(cd "$REPO" && git rev-parse --short HEAD)
say() { echo "[t7352 $(date +%H:%M:%S)] $*"; }

DIRTY=$(cd "$REPO" && git status --porcelain -- \
  scripts/distill/tau2/t2_gate_patch.py scripts/distill/tau2/t2_search.py \
  scripts/distill/tau2/t2_resolve.py scripts/distill/tau2/t2_scaffold_get.py \
  scripts/distill/tau2/go_stack.sh scripts/distill/tau2/a2/ | grep -cv '^??' || true)
[ "$DIRTY" = "0" ] || { say "REFUSING: 엔진 경로 미커밋 $DIRTY"; exit 1; }

for t in test_a2_three_layer.py test_flag_registry.py test_no_undefined_names.py \
         test_no_unbound_a2.py test_quote_in.py test_args_equal.py test_t2_procedure.py \
         test_sg_docs_delivery.py test_sg_fetch_iso.py test_sg_isofb.py \
         test_atm_ledger_close.py test_compute_params.py test_write_arg_enum.py \
         test_write_arg_enum_values.py test_spec_at_write.py \
         test_result_round.py test_apy_balance_tier.py test_ref_from_outputs.py \
         test_no_prose_regex.py test_ours_text_canonical.py test_regen_break_guard.py; do
  [ -f "$t" ] || continue
  PYTHONPATH=/home/woori/scratch/tau2-bench/src timeout 90 \
    /home/woori/venvs/seka_env/bin/python "$t" >/dev/null 2>&1 || { say "REFUSING: $t FAIL"; exit 1; }
done
say "VERIFY OK (배터리 21 · 새 래칫 test_spec_at_write 포함)"

pgrep -f "[t]2_launch" >/dev/null && { say "REFUSING: 다른 라이브 런"; exit 1; } || true
for f in "$LOG"/${TAGBASE}_*.log; do [ -e "$f" ] && { say "REFUSING: $f 존재"; exit 1; }; done
for d in "$SIMS"/${TAGBASE}_*; do [ -e "$d" ] && { say "REFUSING: $d 잔존"; exit 1; }; done

PIN="T2_ACTION_SUB=1 T2_KEEP_DENY_BODY=1 T2_CALL_FORM=1 T2_ARG_EMPTY=1 T2_SEARCH_AGENT=1 \
T2_DECIDE_ANY=1 T2_WRITE_ARG_ENUM=1 T2_DECIDE_BEFORE_WRITE=1 T2_DECISION_CARRY=1 \
T2_DISCOVERY_STEP2=1 T2_ARG_AXIS=1 T2_WRITE_SUB=3 T2_ACTION_INDEX=1 T2_NOW_SELFCALL=1 \
T2_SEARCH_ON_PROCEED=1 T2_ACT_DEMAND=0 T2_DELIVER_PRECOMMIT=0 T2_PROCEED_DOCBODY=0 \
T2_DOCS_AT_WRITE=0 T2_SUB_REQUIREMENT=0 T2_HANDOFF_PREDICATE=0 T2_PENDING_DISCOVERED=0 \
T2_VERDICT_CARRY=0 T2_ELIG_LINE=0 T2_VERDICT_GATE=0 T2_CLAIM_VERIFY=0 \
T2_DECLFIRST=0 T2_DECLFIRST_GUIDE_FIX=0 T2_SCHEMA_ENUM=0 T2_CATEGORY_CITE="
ON="T2_ARG_DOC_SUB=1 T2_VALUE_FORMULA=full T2_SG_DOCS=1 T2_SG_PROMPT_V2=1 T2_SPEC_AT_WRITE=1"

echo "{\"tag\":\"t7352\",\"sha\":\"$SHA\",\"design\":\"time-boxed hard0 targets in deadline-guarded batches; per-sim minutes measured from t7348 durations\",\"on\":\"$ON\",\"reference\":\"t7348 sha aed30e20 - every task in this roster was 0/2\",\"bar\":\"a target going 0 -> 1; never judge on a total delta; bundle so no individual attribution (C594)\",\"cannot_measure\":\"what T2_SPEC_AT_WRITE sells - the roster is all hard-0, so no passing task is at risk in it\",\"deadline\":\"$DEADLINE_HHMM\"}" \
  | tee "$LOG/${TAGBASE}.meta.json"

setsid bash -c "
  cd '$REPO/scripts/distill/tau2'
  source ./go_stack.sh >/dev/null 2>&1
  export $PIN
  export $ON
  export GO_MAX_STEPS=150 GO_CONCURRENCY=1

  fits() {   # \$1 = 추정 분. 마감 전에 끝나면 0.
    local est=\$1 now dl
    now=\$((10#\$(date +%H) * 60 + 10#\$(date +%M)))
    dl=\$((10#\${DEADLINE_HHMM:0:2} * 60 + 10#\${DEADLINE_HHMM:2:2}))
    [ \$((now + est)) -le \$dl ]
  }

  persist() {
    local TAG=\$1 _S _F
    cd '$REPO' && mkdir -p reports/facet_rft_2026/sim_results
    gzip -c '$SIMS/'\$TAG'/results.json' > reports/facet_rft_2026/sim_results/\$TAG.results.json.gz 2>/dev/null || true
    gzip -c $LOG/\$TAG.log > reports/facet_rft_2026/sim_results/\$TAG.log.gz 2>/dev/null || true
    for _S in fb trace; do
      _F=$LOG/\${_S}_\${TAG}.jsonl
      if [ -s \"\$_F\" ]; then
        gzip -c \"\$_F\" > reports/facet_rft_2026/sim_results/\${_S}_\${TAG}.jsonl.gz
      else
        echo \"[t7352] WARN \${_S} 미회수 \$_F — 우리-층 귀속 판정 불가([[25]])\"
      fi
    done
    git add -f reports/facet_rft_2026/sim_results/\$TAG*.gz \\
               reports/facet_rft_2026/sim_results/fb_\$TAG*.jsonl.gz \\
               reports/facet_rft_2026/sim_results/trace_\$TAG*.jsonl.gz 2>/dev/null || true
    git -c user.name=ghlee -c user.email=beingrelative@gmail.com commit -q -m \"t7352 batch \$TAG\" || true
    git push -q origin facet-rft-2026 || true
    git ls-files --error-unmatch reports/facet_rft_2026/sim_results/\$TAG.results.json.gz >/dev/null 2>&1 \\
      && echo \"[t7352] \$TAG persisted+tracked OK\" || echo \"[t7352] ⛔\$TAG NOT TRACKED\"
    cd '$REPO/scripts/distill/tau2'
  }

  batch() {  # \$1 이름 · \$2 포트 · \$3 태스크목록 · \$4 nt · \$5 추정분
    local NAME=\$1 PORT=\$2 TL=\$3 NT=\$4 EST=\$5 TAG=${TAGBASE}_\$1_20260825
    if ! fits \$EST; then
      echo \"[t7352] SKIP \$NAME (추정 \${EST}분 · 마감 $DEADLINE_HHMM 초과) — 이 배치는 **안 돌았다**\"
      return 0
    fi
    echo \"[t7352 \$(date +%H:%M:%S)] === \$NAME · \$TL · nt=\$NT · 추정 \${EST}분 ===\"
    t2_launch \$TAG \$PORT \"\$TL\" \$NT 2>&1 | tee $LOG/\$TAG.log
    echo \"[t7352] \$NAME 완료 · SPEC_AT_WRITE=\$(grep -ac 'T2_SPEC_AT_WRITE' $LOG/\$TAG.log || true)\
 · SPEC_DIST=\$(grep -ac 'T2_SPEC_DIST' $LOG/\$TAG.log || true)\
 · SUBWIN=\$(grep -ac 'T2_SUBWIN' $LOG/\$TAG.log || true)\
 · CLOBBER=\$(grep -ac 'T2_AXIS_CLOBBER' $LOG/\$TAG.log || true)\
 · PROMPT_V2=\$(grep -ac 'T2_SG_PROMPT_V2' $LOG/\$TAG.log || true)\
 · Traceback=\$(grep -ac 'Traceback' $LOG/\$TAG.log || true)\"
    persist \$TAG
  }

  # ── 스모크 = 016 nt1. 싸고(실측 12.7분) 새 계기 두 개를 다 지나간다:
  #    016 은 \`formalize_intent_tool\` 이 sim 당 59회 도는 태스크라 [T2_SUBWIN] 이 반드시 찍힌다.
  #    ⚠[T2_SPEC_AT_WRITE] 는 여기서 안 찍힌다 — 그 갈래는 040·085 에만 있다(t7348 로그 실측:
  #      \`축 미상\` 040 54회·085 44회·나머지 0). 그 배선은 **오프라인 함수 검정**이 지킨다
  #      (test_spec_at_write.py [7] — 실제 호출로 본문·인덱스·거리·침묵 4건).
  SMK=${TAGBASE}_smoke_20260825
  echo '[t7352] === 스모크(016 · nt1 · 8140) ==='
  t2_launch \$SMK 8140 task_016 1 > $LOG/\$SMK.log 2>&1 || true
  # ⚠2026-08-25 수리: 구판은 \`|| echo 0\` 이었다. \`grep -c\` 는 매치가 0 일 때 **이미 '0' 을
  #   찍고** 종료코드 1 을 내므로 그 뒤 \`echo 0\` 이 한 줄을 더 붙여 값이 \"0\\n0\" 이 됐고,
  #   \`[ \"\$NTB\" -gt 0 ]\` 이 *정수 표현식 예상됨* 으로 죽어 **게이트가 통째로 무력화**됐다
  #   (t7352 실물 — 그 런은 사람이 수동 확인해서 유효했다). 계기가 거짓말하면 게이트도 거짓말한다([[25]]).
  NTB=\$(grep -ac 'Traceback' $LOG/\$SMK.log || true); NTB=\${NTB:-0}
  NSW=\$(grep -ac 'T2_SUBWIN' $LOG/\$SMK.log || true); NSW=\${NSW:-0}
  echo \"[t7352] 스모크 — SUBWIN=\$NSW · Traceback=\$NTB\"
  grep -a 'T2_SUBWIN' $LOG/\$SMK.log | head -3 || true
  persist \$SMK
  if [ \"\$NTB\" -gt 0 ]; then echo '[t7352] ⛔스모크 Traceback — 중단'; grep -a -A3 Traceback $LOG/\$SMK.log | head -20; exit 1; fi
  if [ \"\$NSW\" -eq 0 ]; then echo '[t7352] ⛔[T2_SUBWIN] 발화 0 = 死배선 — 중단'; exit 1; fi

  cd '$REPO'
  /home/woori/venvs/seka_env/bin/python reports/facet_rft_2026/freeze.py --on \\
    --tag t7352 --reason 'time-boxed hard0 batches with spec-at-write' || true
  cd '$REPO/scripts/distill/tau2'

  # ── 두 GPU 를 나란히. 배치는 순차이고 각자 마감 가드를 본다.
  #   순서 = **수리가 실린 것 먼저**. 뒤쪽 배치는 마감 가드가 알아서 자른다(건너뛰면 로그에 남는다).
  #   085 는 두 그룹에 걸쳐 3+2=**5 시행** — 오늘 유일하게 격리를 통과한 새 레버가 실린 자리라
  #   0→1 한 건의 의미가 가장 큰 표적이다. 040 은 3 시행이되 **nt2 + nt1 두 배치로 쪼갠다**
  #   (한 sim 이 128.6분까지 갔던 태스크라 통째로 두면 마감 초과 시 전량을 잃는다).
  (
    batch grpA1 8140 'task_085,task_074' 3 229
    batch grpA2 8140 'task_016'          3  38
    batch grpA3 8140 'task_057'          3  81
    batch grpA4 8140 'task_063'          3  47
  ) > $LOG/${TAGBASE}_grpA_chain.log 2>&1 &
  P1=\$!
  (
    batch grpB1 8141 'task_040'          2 174
    batch grpB2 8141 'task_085'          2  74
    batch grpB3 8141 'task_040'          1  87
    batch grpB4 8141 'task_072,task_073' 2  81
  ) > $LOG/${TAGBASE}_grpB_chain.log 2>&1 &
  P2=\$!
  wait \$P1 \$P2

  cd '$REPO'
  cp $LOG/${TAGBASE}.meta.json reports/facet_rft_2026/sim_results/${TAGBASE}.meta.json || true
  git add -f reports/facet_rft_2026/sim_results/${TAGBASE}.meta.json
  git -c user.name=ghlee -c user.email=beingrelative@gmail.com commit -q -m 't7352 meta' || true
  git push -q origin facet-rft-2026 || true
  /home/woori/venvs/seka_env/bin/python reports/facet_rft_2026/freeze.py --off --tag t7352 || true
  echo '[t7352] ALL DONE'
" </dev/null >"$LOG/${TAGBASE}_chain.log" 2>&1 &
say "기동 PID=$! · sha=$SHA · 마감 $DEADLINE_HHMM · 로그 $LOG/${TAGBASE}_chain.log"
