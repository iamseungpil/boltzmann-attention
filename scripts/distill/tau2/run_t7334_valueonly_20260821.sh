#!/usr/bin/env bash
# t7334 — **값-단독 팔**: t7333 의 +4 를 배달과 값에 나눈다 (사용자 지시 2026-08-21 *"값 단독 팔 돌려라"*)
#
# ★사전등록 (런 시작 전에 못 박는다 · 결과 보고 바꾸지 않는다)
#   질문 : t7333 의 **+4(6/20 → 10/20)** 는 값 주석만으로도 나오는가, 아니면 **배달이 있어야** 나오는가
#   종점 : **reward 뿐**([[69]]).
#   설계 : **한 팔만** 새로 돈다 — 나머지 두 칸은 t7333 에 이미 있다(같은 sha·같은 태스크·같은 nt).
#            ctl(둘 다 unset)      = **6/20**   ← t7333 기존
#            val(이 런)            = ?          ← `T2_VALUE_FORMULA=full` 만 · `T2_ARG_DOC_SUB` unset
#            treat(둘 다 ON)       = **10/20**  ← t7333 기존
#   비교 가능성 : 엔진·A2·go_stack 이 t7333 sha `2266aa15` 와 **바이트 동일**함을 발사 전 확인했다.
#   표적/규모 : t7333 과 **완전히 같다** — task_024×6 + 나머지 7태스크×2 = **20 sim**(+스모크 1)
#   판정 : ⒜ `val < ctl(6)` 이면 **값 단독은 해롭고 배달이 구조한다** ⇒ 합성이 일하는 것
#          ⒝ `val ≈ treat(10)` 이면 **값이 다 한 것**이고 배달의 기여는 미확인
#          ⒞ 그 사이면 **둘 다 필요**하다는 뜻 — 크기는 이 규모에서 읽지 않는다
#          ⛔어느 경우든 |차| < 2 는 null 로 읽는다(C483 ±4/40).
#   동급 계측 : 조회 · 쓰기(over-action) · 날조(`_provenance_deny`) · **배달 발화 = 0 이어야 한다**
#
# ★예측(사전 등록·틀리면 그대로 기록한다)
#   024 에서 모델은 `spend_category='operations'` 를 **보낸다**(t7333 궤적 전수 확인). 배달이 없으면
#   그 범주가 살아남아 `Business Gold` 가 **2.5% × 40,000 − 200 = 800** 으로 1위가 된다(C566/C567 의
#   증폭 조건). ⇒ **val 은 ctl 보다 나쁘거나 같을 것**이고, 특히 024 는 2/6 이하로 예상한다.
#
# ⚠유료(user-sim = openrouter gpt-5.2). 두 부분을 다른 GPU 에 올려 병렬(hot=8140 · rest=8141).
set -e
REPO=/home/woori/workspace_common/boltzmann-attention-pi
LOG=/home/woori/scratch/logs
SIMS=/home/woori/scratch/tau2-bench/data/simulations
SHA=$(cd "$REPO" && git rev-parse --short HEAD)

PIN="T2_ACTION_SUB=1 T2_KEEP_DENY_BODY=1 T2_CALL_FORM=1 T2_ARG_EMPTY=1 T2_SEARCH_AGENT=1 \
T2_DECIDE_ANY=1 T2_WRITE_ARG_ENUM=1 T2_DECIDE_BEFORE_WRITE=1 T2_DECISION_CARRY=1 \
T2_DISCOVERY_STEP2=1 T2_ARG_AXIS=1 T2_WRITE_SUB=3 T2_ACTION_INDEX=1 T2_NOW_SELFCALL=1 \
T2_SEARCH_ON_PROCEED=1 T2_ACT_DEMAND=0 T2_DELIVER_PRECOMMIT=0 T2_PROCEED_DOCBODY=0 \
T2_DOCS_AT_WRITE=0 T2_SUB_REQUIREMENT=0 T2_HANDOFF_PREDICATE=0 T2_PENDING_DISCOVERED=0 \
T2_VERDICT_CARRY=0 T2_ELIG_LINE=0 T2_VERDICT_GATE=0 T2_CLAIM_VERIFY=0 \
T2_DECLFIRST=0 T2_DECLFIRST_GUIDE_FIX=0 T2_CATEGORY_CITE="

HOT='task_024'
HOT_NT=6
REST='task_003,task_025,task_001,task_070,task_055,task_047,task_063'
REST_NT=2
M_DOC='T2_ARG_DOC_SUB'
M_VAL='documented_return_for_stated_spend'

for f in "$LOG/bank_t7334_"*.log "$LOG/bank_t7334_chain.log"; do
  [ -e "$f" ] && { echo "[t7334] REFUSING: $f 존재" >&2; exit 1; }
done
for d in "$SIMS/bank_t7334_"*; do
  [ -d "$d" ] && { echo "[t7334] REFUSING: $d 존재 — 영속화하고 새 태그를 써라" >&2; exit 1; }
done

echo "{\"tag\":\"t7334\",\"sha\":\"$SHA\",\"arm\":\"val(T2_VALUE_FORMULA=full only)\",\"compare_to\":{\"ctl\":\"6/20\",\"treat\":\"10/20\",\"from\":\"t7333 sha 2266aa15 · engine byte-identical\"},\"hot\":\"$HOT x $HOT_NT\",\"rest\":\"$REST x $REST_NT\",\"n\":20,\"endpoint\":\"reward only\",\"bar\":\"|d| < 2 = null (C483 ±4/40)\",\"prediction\":\"val <= ctl; 024 <= 2/6 because the surviving operations category makes Business Gold 800\",\"must_be_zero\":\"T2_ARG_DOC_SUB firings\"}" \
  | tee "$LOG/bank_t7334.meta.json"

setsid bash -c "
  cd '$REPO/scripts/distill/tau2'
  source ./go_stack.sh >/dev/null 2>&1
  export $PIN
  export GO_MAX_STEPS=150 GO_CONCURRENCY=1
  export T2_VALUE_FORMULA=full
  unset T2_ARG_DOC_SUB

  # ── 0단계 스모크(1 sim): 값 주석이 발화하고 **배달은 0** 이어야 이 팔이 성립한다
  SMK=bank_t7334_smoke_20260821
  echo '[t7334] === 스모크(1 sim · task_024 · 8141) ==='
  t2_launch \$SMK 8141 $HOT 1 2>&1 | tee $LOG/\$SMK.log
  ND=\$(grep -c '$M_DOC' $LOG/\$SMK.log 2>/dev/null || echo 0)
  NV=\$(grep -c '$M_VAL' $LOG/\$SMK.log 2>/dev/null || echo 0)
  echo \"[t7334] 스모크 — 값주석=\$NV · 배달=\$ND (배달은 0 이어야 한다)\"
  cd '$REPO' && mkdir -p reports/facet_rft_2026/sim_results
  gzip -c '$SIMS/'\$SMK'/results.json' > reports/facet_rft_2026/sim_results/\$SMK.results.json.gz
  gzip -c $LOG/\$SMK.log > reports/facet_rft_2026/sim_results/\$SMK.log.gz
  cd '$REPO/scripts/distill/tau2'
  if [ \"\$NV\" -eq 0 ]; then echo '[t7334] ⛔값 주석 0 — 본런 중단'; exit 1; fi
  if [ \"\$ND\" -ne 0 ]; then echo '[t7334] ⛔배달이 발화했다 — 팔이 오염됐다. 중단'; exit 1; fi

  cd '$REPO'
  /home/woori/venvs/seka_env/bin/python reports/facet_rft_2026/freeze.py --on \\
    --tag t7334 --reason 'value-only arm for attribution' || true
  cd '$REPO/scripts/distill/tau2'

  run_part() {
    PART=\$1; PORT=\$2
    TAG=bank_t7334_val_\${PART}_20260821
    if [ \"\$PART\" = hot ]; then TL='$HOT'; NT=$HOT_NT; else TL='$REST'; NT=$REST_NT; fi
    echo \"[t7334] === val/\$PART · port=\$PORT ===\"
    t2_launch \$TAG \$PORT \"\$TL\" \$NT 2>&1 | tee $LOG/\$TAG.log
    echo \"[t7334] val/\$PART 완료 · 값=\$(grep -c '$M_VAL' $LOG/\$TAG.log 2>/dev/null || echo 0) · 배달=\$(grep -c '$M_DOC' $LOG/\$TAG.log 2>/dev/null || echo 0)\"
    cd '$REPO' && mkdir -p reports/facet_rft_2026/sim_results
    gzip -c '$SIMS/'\$TAG'/results.json' > reports/facet_rft_2026/sim_results/\$TAG.results.json.gz
    gzip -c $LOG/\$TAG.log > reports/facet_rft_2026/sim_results/\$TAG.log.gz
    cd '$REPO/scripts/distill/tau2'
  }
  ( run_part hot 8140 )  > $LOG/bank_t7334_hot_chain.log 2>&1 &
  P1=\$!
  ( run_part rest 8141 ) > $LOG/bank_t7334_rest_chain.log 2>&1 &
  P2=\$!
  wait \$P1 \$P2

  echo '[t7334] 두 부분 완료 — 영속화'
  cd '$REPO'
  git add -f reports/facet_rft_2026/sim_results/bank_t7334_*.gz
  git -c user.name=ghlee -c user.email=beingrelative@gmail.com commit -q -m 't7334 value-only arm' || true
  git push -q origin facet-rft-2026 || true
  for P in hot rest; do
    G=reports/facet_rft_2026/sim_results/bank_t7334_val_\${P}_20260821.results.json.gz
    git ls-files --error-unmatch \$G >/dev/null 2>&1 \\
      && echo \"[t7334] \$P 영속 확인 tracked\" || echo \"[t7334] ⚠\$P 영속 실패\"
  done
  /home/woori/venvs/seka_env/bin/python reports/facet_rft_2026/freeze.py --off || true
  echo '[t7334] 동결 해제 · 끝'
" </dev/null > "$LOG/bank_t7334_chain.log" 2>&1 &

echo "[t7334] 기동 PID=$! · sha=$SHA · 스모크 1 + 본런 20 sim · hot=8140 / rest=8141"
