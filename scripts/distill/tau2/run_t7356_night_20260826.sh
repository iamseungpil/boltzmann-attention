#!/bin/bash
# t7356 — 밤샘 런 (사용자 지시 2026-08-25: "내가 자는 동안 돌릴 수 있는 태스크를 밤샘 런에서 돌려라")
#
# ## 이 런이 t7355 와 다른 것 넷 (전부 오늘 밤 격리로 자격을 얻은 것)
#   A2 write_rules  책임 한도 표를 debit 분쟁 write 에 선언 — x538 A 12/20 vs B 20/20 vs N_len 12/20,
#                   합성 x538b B_both 20/20 (이미 실려 있는 문장과 서로 안 죽인다)
#   T2_WRITE_ARG_FAB=1        자리표시자로 채운 인자 — 술어 셋 전부 선언이거나 값의 모양이고
#                   이름 패턴 0. t7354 6배치 전수 20건 전부 진짜 날조·오차단 0
#   T2_SG_RECORD_ORDER=1      서브에게 주는 덤프를 타입별로 묶고 날짜 오름차순 —
#                   x536 N_wire 17 vs D_old_group 16 vs N_scramble 은 두 계좌를 부순다,
#                   x539 가 도메인 낱말 0 판(D_firstseen_group)이 같은 수를 내는지 확인
#   identifying_arg_types.digit  **철회**했다 — 이름 패턴이었고 위 파생이 더 잘 잡는다
#
# ## 왜 이 로스터인가 (사용자 지시 2026-08-25 밤·축자)
#   *"내일 아침에 다양한 결과를 분석할 수 있게 밤샘런을 최대한 많이 돌려라. 자는 동안 최대한
#    많은 실험이 이루어져서 그걸 바탕으로 내일 아침에 최대한 많은 수정을 했으면 한다."*
#   ⇒ 목적함수가 *0→1 하나*에서 **아침에 고칠 거리의 양**으로 바뀌었다. 같은 태스크를 13회
#     돌리는 것은 그 목적에 나쁘다(진단 정보는 nt3 쯤에서 포화한다). 그래서 **넓힌다**.
#
#   ⑴ hard-0 **전 10 태스크**: 016·040·055·057·063·072·074·079·085·094
#      아침 `morning_review.py §4` 가 태스크마다 gold 행 ↔ 궤적 호출 **인자 차이표**를 낸다 —
#      오늘 085·040 의 축을 가른 바로 그 표다. 열 태스크분이 한꺼번에 생긴다.
#   ⑵ **회귀 대조 017·098**(지금 통과 중): t7354 드라이버가 스스로 적어 둔 한계가
#      *"로스터가 전부 hard-0 라 레버가 **무엇을 파는지 못 잰다**"* 였다. 017 은 오늘 새로
#      감시를 붙인 **선언 불리언·열거 도구를 실제로 부르는** 태스크다(t7348 실측: 불리언 2·열거 3).
#      떨어지면 그것이 [[70]] 이 요구하는 *무엇을 팔았나*의 첫 수치다.
#   ⑶ 가중: 074(nt4·스모크 포함)와 085(nt3)에 조금 더 — 오늘 수리가 닿는 유일한 둘이다.
#      040 은 nt1(계기)뿐 — reward_basis=['DB'] 에 gold 8건 전부라 뒤집힐 후보가 아니고
#      `T2_WRITE_ARG_FAB` 의 유일한 표적이라 **라이브 발화만** 본다.
#   ⚠072 는 **넣되 기대를 적어 둔다**: t7348 실측에서 `apply_checking_account_credit_5829` 를
#     한 번도 부르지 않는다(unlock 0). 오늘 레버는 write 도달 뒤에만 일하므로 이 태스크는
#     **표기가 아니라 도달**이 결손이다 — nt3 은 그 도달 실패를 아침에 표로 보기 위한 것이다.
#   ⚠마감 가드는 **뒤에서부터** 버린다. 그래서 회귀 대조를 grpB 맨 앞에 뒀다.
#
# ## 대조와 판정선
#   대조 = t7355(085) · t7348/t7354(074·040). 판정선 = **표적의 0->1**.
#   총점 델타 금지 · 레버가 넷이라 개별 귀속 불가 (C594).
#
# ## 안전 · 동시성
#   앞 런(t7355)은 **8140** 만 쓴다. 그래서 grpB(8141)는 **지금 바로** 시작하고 8140 쪽만
#   그 런의 저장 태그(`bank_t7355_main`)가 사라질 때까지 기다린다 — 기다리면 GPU 하나를
#   45~50분 버린다(사용자 지적 2026-08-25 밤).
#   ⚠술어를 `pgrep -f t2_run_gated` 로 넓게 잡으면 **우리 자신도 잡혀** 영원히 기다린다.
#   ⚠앞 런의 파이썬은 임포트를 이미 마쳤으므로 디스크의 엔진 갱신에 영향받지 않는다.
#   스모크는 마커가 아니라 **산출**로 보되, 실패해도 exit 하지 않는다(grpB 고아 방지·grpA 만 건너뜀).
#
# ## 총량
#   15 배치 · **35 sim** · **14 태스크** = hard-0 10 + 회귀 대조 4(017·098·100·050)
set -e
REPO=/home/woori/workspace_common/boltzmann-attention-pi
LOG=/home/woori/scratch/logs
SIMS=/home/woori/scratch/tau2-bench/data/simulations
TAGBASE=bank_t7356
DEADLINE_HHMM=${DEADLINE_HHMM:-0730}
mkdir -p "$LOG"
say() { echo "[t7356 $(date +%H:%M:%S)] $*"; }

# ── ⑴ **기다리지 않는다** (사용자 지적 2026-08-25 밤: *"gpu 2 개 다를 지금부터 사용하면 안되나?"*).
#   앞 런(t7355)은 **8140** 만 쓴다. 8141 은 비어 있고, 종료를 기다리면 그 GPU 를 45~50분 버린다.
#   그래서 grpB(8141)는 **지금 시작**하고, 8140 을 쓰는 쪽(스모크+grpA)만 아래에서 기다린다.
#   ⚠앞 런의 파이썬 프로세스는 임포트를 이미 마쳤으므로 디스크의 엔진 갱신에 영향받지 않는다.
say "8141 은 즉시 시작한다 · 8140 은 앞 런 종료를 기다린다"

# ── ⑵ origin 과 화해 — 앞 런의 persist 커밋이 로컬에만 있고 내 커밋이 원격에 있다
cd "$REPO"
git fetch -q origin facet-rft-2026
# --autostash: 앞 런이 남긴 `FREEZE.json` 수정으로 rebase 가 거부되는 것을 막는다(런 산출물).
git -c user.name=ghlee -c user.email=beingrelative@gmail.com rebase --autostash origin/facet-rft-2026 \
  || { say "REFUSING: rebase 실패 — 손으로 화해해야 한다"; git rebase --abort || true; exit 1; }
git push -q origin facet-rft-2026 || say "push 보류(원격 선행) — 런 뒤 화해"
SHA=$(git rev-parse --short HEAD)
say "sha=$SHA"

cd "$REPO/scripts/distill/tau2"
DIRTY=$(cd "$REPO" && git status --porcelain -- \
  scripts/distill/tau2/t2_gate_patch.py scripts/distill/tau2/t2_search.py \
  scripts/distill/tau2/t2_resolve.py scripts/distill/tau2/t2_scaffold_get.py \
  scripts/distill/tau2/go_stack.sh scripts/distill/tau2/a2/ | grep -cv '^??' || true)
[ "$DIRTY" = "0" ] || { say "REFUSING: 엔진 경로 미커밋 $DIRTY"; exit 1; }

for t in test_a2_three_layer.py test_flag_registry.py test_no_undefined_names.py \
         test_no_unbound_a2.py test_quote_in.py test_args_equal.py test_t2_procedure.py \
         test_sg_docs_delivery.py test_sg_fetch_iso.py test_sg_isofb.py \
         test_sg_prompt_v2_reachable.py test_sg_record_order.py \
         test_atm_ledger_close.py test_compute_params.py test_write_arg_enum.py \
         test_write_arg_enum_values.py test_spec_at_write.py test_write_arg_type.py \
         test_write_arg_fab.py test_rule_at_write.py test_identifying_hints.py \
         test_spec_arg_facts.py test_arg_policy_join.py \
         test_result_round.py test_apy_balance_tier.py test_ref_from_outputs.py \
         test_no_prose_regex.py test_ours_text_canonical.py test_regen_break_guard.py; do
  [ -f "$t" ] || continue
  PYTHONPATH=/home/woori/scratch/tau2-bench/src timeout 90 \
    /home/woori/venvs/seka_env/bin/python "$t" >/dev/null 2>&1 || { say "REFUSING: $t FAIL"; exit 1; }
done
say "VERIFY OK (배터리 29)"

for f in "$LOG"/${TAGBASE}_*.log; do [ -e "$f" ] && { say "REFUSING: $f 존재"; exit 1; }; done
for d in "$SIMS"/${TAGBASE}_*; do [ -e "$d" ] && { say "REFUSING: $d 잔존"; exit 1; }; done

PIN="T2_ACTION_SUB=1 T2_KEEP_DENY_BODY=1 T2_CALL_FORM=1 T2_ARG_EMPTY=1 T2_SEARCH_AGENT=1 \
T2_DECIDE_ANY=1 T2_WRITE_ARG_ENUM=1 T2_DECIDE_BEFORE_WRITE=1 T2_DECISION_CARRY=1 \
T2_DISCOVERY_STEP2=1 T2_ARG_AXIS=1 T2_WRITE_SUB=3 T2_ACTION_INDEX=1 T2_NOW_SELFCALL=1 \
T2_SEARCH_ON_PROCEED=1 T2_ACT_DEMAND=0 T2_DELIVER_PRECOMMIT=0 T2_PROCEED_DOCBODY=0 \
T2_DOCS_AT_WRITE=0 T2_SUB_REQUIREMENT=0 T2_HANDOFF_PREDICATE=0 T2_PENDING_DISCOVERED=0 \
T2_VERDICT_CARRY=0 T2_ELIG_LINE=0 T2_VERDICT_GATE=0 T2_CLAIM_VERIFY=0 \
T2_DECLFIRST=0 T2_DECLFIRST_GUIDE_FIX=0 T2_SCHEMA_ENUM=0 T2_CATEGORY_CITE="
# ⛔`T2_ARG_POLICY_AT_WRITE` 는 **일부러 뺐다** — x541 이 음성이다(A_asis {false4,true4} ↔
#   B_join {true8} ↔ N_axis {false4,true4}). 조인이 답을 전부 true 로 민다. 배선은 남기고 끈다.
ON="T2_ARG_DOC_SUB=1 T2_VALUE_FORMULA=full T2_SG_DOCS=1 T2_SG_PROMPT_V2=1 T2_SPEC_AT_WRITE=1 \
T2_WRITE_ARG_TYPE=1 T2_RULE_AT_WRITE=1 T2_WRITE_ARG_ENUM_CAP=8 T2_WRITE_ARG_FAB=1 \
T2_SG_RECORD_ORDER=1 T2_SPEC_ARG_FACTS=1"

echo "{\"tag\":\"t7356\",\"sha\":\"$SHA\",\"design\":\"breadth over depth - every hard-0 task plus two that currently pass, so the morning has an argument-diff table for each of them; the objective this time is how much there is to repair tomorrow, not one target flipping\",\"on\":\"$ON\",\"reference\":\"t7355 for 085; t7348 for the rest\",\"bar\":\"a target going 0 to 1; and separately, whether 017 or 098 drop - that is the first measurement of what the new levers sell\",\"cannot_measure\":\"which lever moved what - five ride together (C594)\",\"deadline\":\"$DEADLINE_HHMM\"}" \
  | tee "$LOG/${TAGBASE}.meta.json"

setsid bash -c '
  REPO=/home/woori/workspace_common/boltzmann-attention-pi
  LOG=/home/woori/scratch/logs
  SIMS=/home/woori/scratch/tau2-bench/data/simulations
  TAGBASE=bank_t7356
  cd "$REPO/scripts/distill/tau2"
  source ./go_stack.sh >/dev/null 2>&1
  export '"$PIN"'
  export '"$ON"'
  export GO_MAX_STEPS=150 GO_CONCURRENCY=1
  DEADLINE_HHMM='"$DEADLINE_HHMM"'

  fits() {
    now=$((10#$(date +%H) * 60 + 10#$(date +%M)))
    dl=$((10#${DEADLINE_HHMM:0:2} * 60 + 10#${DEADLINE_HHMM:2:2}))
    [ $dl -lt $now ] && dl=$((dl + 1440))
    [ $((now + $1)) -le $dl ]
  }

  persist() {
    TAG=$1
    cd "$REPO" && mkdir -p reports/facet_rft_2026/sim_results
    gzip -c "$SIMS/$TAG/results.json" > reports/facet_rft_2026/sim_results/$TAG.results.json.gz 2>/dev/null || true
    gzip -c $LOG/$TAG.log > reports/facet_rft_2026/sim_results/$TAG.log.gz 2>/dev/null || true
    for _S in fb trace; do
      _F=$LOG/${_S}_${TAG}.jsonl
      if [ -s "$_F" ]; then
        gzip -c "$_F" > reports/facet_rft_2026/sim_results/${_S}_${TAG}.jsonl.gz
      else
        echo "[t7356] WARN ${_S} 미회수 $_F"
      fi
    done
    git add -f reports/facet_rft_2026/sim_results/$TAG.results.json.gz \
               reports/facet_rft_2026/sim_results/$TAG.log.gz \
               reports/facet_rft_2026/sim_results/fb_$TAG.jsonl.gz \
               reports/facet_rft_2026/sim_results/trace_$TAG.jsonl.gz 2>/dev/null || true
    # ⚠**경로 한정 커밋**이어야 한다 (2026-08-25 사고): 인자 없는 `commit` 은 인덱스에 스테이지된
    #   것을 전부 쓸어 담는다. 그날 내가 `git checkout FETCH_HEAD -- <드라이버>` 로 스테이지해 둔
    #   스크립트가 런의 persist 커밋에 실렸고, 다음 rebase 가 add/add 충돌로 거부됐다.
    git -c user.name=ghlee -c user.email=beingrelative@gmail.com commit -q -m "t7356 batch $TAG" \
      -- reports/facet_rft_2026/sim_results/ || true
    git push -q origin facet-rft-2026 || echo "[t7356] push 보류"
    git ls-files --error-unmatch reports/facet_rft_2026/sim_results/$TAG.results.json.gz >/dev/null 2>&1 \
      && echo "[t7356] $TAG persisted+tracked OK" || echo "[t7356] $TAG NOT TRACKED"
    cd "$REPO/scripts/distill/tau2"
  }

  batch() {
    NAME=$1; PORT=$2; TL=$3; NT=$4; EST=$5; TAG=${TAGBASE}_${NAME}_20260826
    if ! fits $EST; then
      echo "[t7356] SKIP $NAME (추정 ${EST}분 · 마감 초과) — 이 배치는 안 돌았다"
      return 0
    fi
    unset T2_FB_SIDECAR T2_TRACE
    echo "[t7356 $(date +%H:%M:%S)] === $NAME · $TL · nt=$NT · 추정 ${EST}분 ==="
    t2_launch $TAG $PORT "$TL" $NT 2>&1 | tee $LOG/$TAG.log
    RO=$(grep -ac "T2_SG_RECORD_ORDER" $LOG/$TAG.log || true); RO=${RO:-0}
    FA=$(grep -ac "T2_WRITE_ARG_FAB" $LOG/$TAG.log || true); FA=${FA:-0}
    TY=$(grep -ac "T2_WRITE_ARG_TYPE" $LOG/$TAG.log || true); TY=${TY:-0}
    RU=$(grep -ac "T2_RULE_AT_WRITE" $LOG/$TAG.log || true); RU=${RU:-0}
    TB=$(grep -ac "Traceback" $LOG/$TAG.log || true); TB=${TB:-0}
    echo "[t7356] $NAME 완료 · RECORD_ORDER=$RO · ARG_FAB=$FA · ARG_TYPE=$TY · RULE_AT_WRITE=$RU · Traceback=$TB"
    persist $TAG
  }

  cd "$REPO"
  /home/woori/venvs/seka_env/bin/python reports/facet_rft_2026/freeze.py --on \
    --tag t7356 --reason "liability table, fabrication guard without name patterns, record order" || true
  cd "$REPO/scripts/distill/tau2"

  # ── grpB(8141)는 **스모크와 병렬로 먼저** 띄운다. 스모크가 가르는 것은 074 의 재배열이고
  #    grpB 에 074 가 없으므로 안전하다. 이렇게 해야 8141 이 45분을 놀지 않고, 마감 가드가
  #    뒤 배치를 잘라내지 않는다(폭이 이 런의 목적이다).
  #    앞쪽 = 회귀 대조(017·098) — 가장 못 재봤던 것을 먼저 확보한다([[70]]).
  (
    batch grpB1 8141 task_017 2  50
    batch grpB2 8141 task_098 2  40
    batch grpB3 8141 task_016 3  45
    batch grpB4 8141 task_063 3  54
    batch grpB5 8141 task_057 2  60
    batch grpB6 8141 task_055 2  60
    batch grpB7 8141 task_094 2  60
    batch grpB8 8141 task_079 2  60
    batch grpB9 8141 task_100 2  30
    batch grpB10 8141 task_050 2 50
  ) > $LOG/${TAGBASE}_grpB_chain.log 2>&1 &
  P2=$!

  # ── 8140 은 앞 런(t7355)이 그 GPU 를 놓을 때까지만 기다린다(최대 4h).
  #    술어는 **그 런의 저장 태그**다 — `pgrep -f t2_run_gated` 로 넓게 잡으면 우리 자신도 잡힌다.
  W=0
  while pgrep -f "[b]ank_t7355_main" >/dev/null; do
    [ $W -ge 14400 ] && { echo "[t7356] 8140 대기 4시간 초과 — grpA 포기"; break; }
    [ $((W % 600)) -eq 0 ] && echo "[t7356] 8140 대기 ${W}s (t7355 진행 중)"
    sleep 60
    W=$((W + 60))
  done
  echo "[t7356] 8140 해제 (대기 ${W}s)"

  # ── 스모크 = 074 nt1. 산출로 건다: 재배열이 적용됐고 operand 가 살아 있는가.
  batch smoke 8140 task_074 1 45
  SLOG=$LOG/${TAGBASE}_smoke_20260826.log
  SAP=$(grep -ac "덤프 재배열 적용" $SLOG || true); SAP=${SAP:-0}
  SMK=$(grep -ac "T2_SG_RECORD_ORDER" $SLOG || true); SMK=${SMK:-0}
  STB=$(grep -ac "Traceback" $SLOG || true); STB=${STB:-0}
  SOK=$(grep -acF "operand keys=[" $SLOG || true); SOK=${SOK:-0}
  # 게이트는 **死배선과 크래시**만 막는다. 적용 0 은 중단 사유가 아니다 —
  # `_reorder_records` 는 이미 그 순서면 무변이고 덤프가 둘이면 손대지 않는다(설계).
  # 다른 세 레버가 함께 타므로 밤을 태울 이유가 없다. 대신 수를 크게 남긴다.
  # ⚠게이트가 실패해도 **exit 하지 않는다** — grpB 가 이미 8141 에서 돌고 있어 고아가 된다.
  #   grpA(074·085·072·040)만 건너뛰고 grpB 는 끝까지 살린다. 폭이 이 런의 목적이므로
  #   074 하나 때문에 나머지 여덟 배치를 버리는 것이 더 나쁘다.
  GATE=1
  if [ "$SMK" -eq 0 ] || [ "$STB" -gt 0 ]; then
    GATE=0
    echo "[t7356] ⛔스모크 게이트 FAIL 마커=$SMK Traceback=$STB — grpA 를 건너뛴다(grpB 는 계속)"
  fi
  echo "[t7356] 스모크 게이트 마커=$SMK 재배열적용=$SAP operand=$SOK Traceback=$STB GATE=$GATE"
  [ "$SAP" -eq 0 ] && echo "[t7356] ★주의: 재배열이 한 번도 적용되지 않았다 — 아침에 이유를 볼 것(무변인가·덤프가 둘인가)"

  # ── 로스터 = **수리가 닿는 두 태스크에 몰아준다**(2026-08-25 밤 재설계).
  #   074 를 가장 무겁게 준다: 오늘 격리가 그 태스크의 **측정된 결손 전부**를 닫았다
  #   (x536/x539 4계좌 4/4 · rows=expect · cover 만점 · fee_paired 만점 · dup 0 ·
  #    부정통제는 부순다). 그리고 선행 포렌식이 *"전사가 맞으면 chk_3·chk_4 산수는 센트까지
  #    닫히고 chk_1 은 이미 정확"* 이라고 적었다 ⇒ 뒤집힐 확률이 가장 높다.
  #   085 는 표기가 닫혔으나 **분쟁 3건을 다 내야** 하고 3번째는 판단 두 칸이 남는다.
  #   040 은 DB 축에 gold 8건 전부라 뒤집힐 후보가 아니다 — `T2_WRITE_ARG_FAB` 의
  #   유일한 표적이라 **계기용 nt1** 만 맨 뒤에 둔다(마감 가드가 먼저 버린다).
  #   ⛔072 는 뺐다: t7348 실측에서 `apply_checking_account_credit_5829` 를 **한 번도 부르지
  #     않는다**(unlock 0). 오늘 레버는 write 에 닿은 뒤에만 일하므로 그 결손에 무력하다.
  # ── grpA(8140) = **수리가 닿는 자리**를 두껍게. 스모크의 074 도 실 trial 로 센다.
  (
    if [ "$GATE" = "1" ]; then
      batch grpA1 8140 task_074 3 126
      batch grpA2 8140 task_085 3 120
      batch grpA3 8140 task_072 3  66
      batch grpA4 8140 task_040 1  90
    else
      echo "[t7356] grpA 건너뜀 — 스모크 게이트 FAIL"
    fi
  ) > $LOG/${TAGBASE}_grpA_chain.log 2>&1 &
  P1=$!
  wait $P1 $P2

  cd "$REPO"
  cp $LOG/${TAGBASE}.meta.json reports/facet_rft_2026/sim_results/${TAGBASE}.meta.json || true
  git add -f reports/facet_rft_2026/sim_results/${TAGBASE}.meta.json
  git -c user.name=ghlee -c user.email=beingrelative@gmail.com commit -q -m "t7356 meta" || true
  git push -q origin facet-rft-2026 || echo "[t7356] push 보류"
  /home/woori/venvs/seka_env/bin/python reports/facet_rft_2026/freeze.py --off --tag t7356 || true
  echo "[t7356] ALL DONE"
' </dev/null >"$LOG/${TAGBASE}_chain.log" 2>&1 &
say "기동 PID=$! · sha=$SHA · 마감 $DEADLINE_HHMM · 로그 $LOG/${TAGBASE}_chain.log"
