#!/usr/bin/env bash
# t7333 — **⒡ 합성**(선언 문서 배달 + 후보별 값)의 첫 라이브 A/B (사용자 지시 2026-08-21 *"합성으로 돌려라"*)
#
# ★사전등록 (런 시작 전에 못 박는다 · 결과 보고 바꾸지 않는다)
#   질문 : A2 가 선언한 문서를 격리 서브에 배달해 **범주를 근거 위에 세우고**, 그 요율로 **후보별
#          값**을 붙이면 **reward 를 사는가**
#   종점 : **reward 뿐**([[69]]). 범주 정확도·gold 일치율은 진단이지 성적이 아니다.
#   설계 : 2팔. ctl = 둘 다 unset(종전 스택·바이트 동일) ↔ treat = 둘 다 ON
#            `T2_ARG_DOC_SUB=1`     선언 12편 → 격리 서브 → 값 + 인용, 엔진은 인용 실재만 검산
#            `T2_VALUE_FORMULA=full` 그 범주의 요율 × 손님이 말한 금액 − 연회비 (정렬·지목 0)
#          ⇒ C571ⓓ 의 ⒡ 그대로다: *"색인 문서로 범주 결정 + 인용 검산 후에만 범주 요율로 곱함"*.
#          단독으로 돌리지 않는 이유([[19]] 합성-우선 + 아래 근거): 배달만으로는 하류가 안 바뀐다.
#   표적 : **024 에 검정력을 몰고**(nt=6) 나머지 7 태스크는 부작용 관측(nt=2).
#          024 = 범주가 카드 선택을 뒤집는 것이 전수로 확인된 유일한 자리.
#          나머지 = fit 이 실제로 도는 태스크(003·025·001·070·055·047·063) — [[70]] 부호표용.
#   규모 : 팔당 6 + 14 = **20 sim** · 두 팔 40 · 스모크 2 = **총 42 sim**
#   판정 : reward 합의 차. **판정선 = C483 잡음 바닥 ±4/40** ⇒ 차 **2 미만이면 null**.
#          024 단독은 6 sim 이라 **부호만** 읽고 크기는 읽지 않는다.
#   의무 3종([[70]]) : ⒜ 전체 reward 짝 ⒝ **태스크별 부호표** ⒞ 무엇을 팔았나
#   동급 계측(사전 고정) : ⒜ 조회 수 ⒝ 날조(`_provenance_deny`) ⒞ over-action(쓰기) · Δspurious ≤ 0
#
# ★왜 이 조합인가 — 전수 포렌식이 배달 단독을 기각했다(2026-08-21·무료)
#   ⒜ 스모크에서 배달이 **발화하지 않았다** — 모델이 fit 을 부를 때 `spend_category` 를 안 단다
#      (전수 326 sim 중 그 인자를 낸 것은 131 = 40%). 배달은 **모델이 이미 낸 인자**에만 붙는다.
#   ⒝ 발화하는 자리에서도 reward 가 안 움직인다 — task_024(n=44·gold=Business Bronze):
#      범주 틀림(`operations`) 24 → reward=1 이 **0** · 범주 맞음(`None`) 20 → reward=1 이 **2**.
#      ⇒ 범주만 고쳐 옮겨도 기댓값 **+2.4/24**(판정선 언저리)이고, 더 중요한 것은 범주를 옳게
#      철회한 20 중 **8 이 여전히 Business Gold 를 신청**한다는 것 — **병목은 범주가 아니라 계산**이다.
#   ⒞ 그 계산 칸은 격리로 이미 특정돼 있다: 패턴 0.38 · 목적식 0.42 · **값을 주면 0.98**(C562).
#   ⇒ 배달은 그 계산에 들어갈 **요율을 옳게** 만드는 부품이고, 둘은 **같이 켜야** 경로가 완성된다.
#
# ★격리 근거와 그 한계(사전에 적어 둔다)
#   배달: 문서 안 주면 44/71 · **선언 문서 주면 71/71**(C576) · 검색은 선언 12편 중 11편을 0회(C577)
#   ⛔단, 선언 71/71 ↔ **아무 카드 문서 12편 70/71** — *"색인이 더 정확하다"* 는 아직 근거가 없다.
#   값  : 합성 격자 0.98 이고 **라이브 reward 로 검증된 적이 없다**(C562 ⛔).
#
# ⚠유료(user-sim = openrouter gpt-5.2). 두 팔을 다른 GPU 에 올려 병렬(ctl=8140 · treat=8141).
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

HOT='task_024'                                                   # 검정력을 모으는 자리
HOT_NT=6
REST='task_003,task_025,task_001,task_070,task_055,task_047,task_063'
REST_NT=2
M_DOC='T2_ARG_DOC_SUB'
M_VAL='documented_return_for_stated_spend'

for f in "$LOG/bank_t7333_"*.log "$LOG/bank_t7333_chain.log"; do
  [ -e "$f" ] && { echo "[t7333] REFUSING: $f 존재" >&2; exit 1; }
done
# ★중단된 런이 남긴 결과 디렉터리가 있으면 tau2 가 **대화형으로 resume 을 묻고** EOF 로 죽는다
#   (2026-08-21 실측). 지우지 않고 **거부**한다 — 과거에 rm 이 완주 런을 날린 적이 있다([[30]]).
for d in "$SIMS/bank_t7333_"*; do
  [ -d "$d" ] && { echo "[t7333] REFUSING: $d 존재 — 영속화하고 새 태그를 써라" >&2; exit 1; }
done

echo "{\"tag\":\"t7333\",\"sha\":\"$SHA\",\"arms\":[\"ctl(both unset)\",\"treat(T2_ARG_DOC_SUB=1 + T2_VALUE_FORMULA=full)\"],\"ports\":{\"ctl\":8140,\"treat\":8141},\"hot\":\"$HOT x $HOT_NT\",\"rest\":\"$REST x $REST_NT\",\"n_per_arm\":20,\"endpoint\":\"reward only\",\"bar\":\"|d| < 2 = null (C483 ±4/40)\",\"also\":[\"reads\",\"fabs\",\"over-action\",\"per-task sign table\",\"delivery fire rate\"],\"why_composed\":\"delivery alone rejected by census: 024 category-correct sims reward 2/20, 8/20 still apply for Gold\",\"iso_caveat\":\"sham 70/71; value grid never validated on live reward\"}" \
  | tee "$LOG/bank_t7333.meta.json"

setsid bash -c "
  cd '$REPO/scripts/distill/tau2'
  source ./go_stack.sh >/dev/null 2>&1
  export $PIN
  export GO_MAX_STEPS=150 GO_CONCURRENCY=1

  # ── 0단계 스모크(2 sim·treat): 두 기구가 **라이브에서 발화**하는지([[30]]·死배선에 돈 금지)
  export T2_ARG_DOC_SUB=1 T2_VALUE_FORMULA=full
  SMK=bank_t7333_smoke_20260821b
  echo '[t7333] === 스모크(2 sim · task_024 · treat · 8141) ==='
  t2_launch \$SMK 8141 $HOT 2 2>&1 | tee $LOG/\$SMK.log
  ND=\$(grep -c '$M_DOC' $LOG/\$SMK.log 2>/dev/null || echo 0)
  NV=\$(grep -c '$M_VAL' $LOG/\$SMK.log 2>/dev/null || echo 0)
  echo \"[t7333] 스모크 발화 — 배달=\$ND · 값주석=\$NV\"
  grep '$M_DOC' $LOG/\$SMK.log 2>/dev/null | head -4 || true
  cd '$REPO' && mkdir -p reports/facet_rft_2026/sim_results
  gzip -c '$SIMS/'\$SMK'/results.json' > reports/facet_rft_2026/sim_results/\$SMK.results.json.gz
  gzip -c $LOG/\$SMK.log > reports/facet_rft_2026/sim_results/\$SMK.log.gz
  git add -f reports/facet_rft_2026/sim_results/\$SMK.*.gz
  git -c user.name=ghlee -c user.email=beingrelative@gmail.com commit -q -m 't7333 smoke' || true
  git push -q origin facet-rft-2026 || true
  cd '$REPO/scripts/distill/tau2'
  # ⚠배달은 모델이 그 인자를 내야만 붙으므로 **0 이어도 죽은 배선이 아니다**(경고만).
  #   값 주석은 fit 이 돌면 항상 붙어야 한다 ⇒ **그것이 0 이면 중단**한다.
  if [ \"\$NV\" -eq 0 ]; then
    echo '[t7333] ⛔값 주석 발화 0 — 본런을 돌리지 않는다'
    exit 1
  fi
  [ \"\$ND\" -eq 0 ] && echo '[t7333] ⚠배달 발화 0 — 모델이 spend_category 를 안 냈다는 뜻(관측으로 남긴다)'

  # ── 코드 동결(C423ⓒ·[[07]]) — 스모크 뒤에 건다
  cd '$REPO'
  /home/woori/venvs/seka_env/bin/python reports/facet_rft_2026/freeze.py --on \\
    --tag t7333 --reason 'live A/B of composed delivery + value' || true
  cd '$REPO/scripts/distill/tau2'

  run_arm() {
    ARM=\$1; PORT=\$2
    if [ \"\$ARM\" = ctl ]; then
      unset T2_ARG_DOC_SUB T2_VALUE_FORMULA
    else
      export T2_ARG_DOC_SUB=1 T2_VALUE_FORMULA=full
    fi
    env | grep -E '^T2_(ARG_DOC_SUB|VALUE_FORMULA)' | sort > $LOG/env_t7333_\${ARM}.txt || true
    echo \"[t7333] === \$ARM · port=\$PORT · doc='\${T2_ARG_DOC_SUB:-(unset)}' val='\${T2_VALUE_FORMULA:-(unset)}' ===\"
    for PART in hot rest; do
      TAG=bank_t7333_\${ARM}_\${PART}_20260821b
      if [ \"\$PART\" = hot ]; then TL='$HOT'; NT=$HOT_NT; else TL='$REST'; NT=$REST_NT; fi
      t2_launch \$TAG \$PORT \"\$TL\" \$NT 2>&1 | tee $LOG/\$TAG.log
      echo \"[t7333] \$ARM/\$PART 완료 · 배달=\$(grep -c '$M_DOC' $LOG/\$TAG.log 2>/dev/null || echo 0) · 값=\$(grep -c '$M_VAL' $LOG/\$TAG.log 2>/dev/null || echo 0)\"
      cd '$REPO' && mkdir -p reports/facet_rft_2026/sim_results
      gzip -c '$SIMS/'\$TAG'/results.json' > reports/facet_rft_2026/sim_results/\$TAG.results.json.gz
      gzip -c $LOG/\$TAG.log > reports/facet_rft_2026/sim_results/\$TAG.log.gz
      cd '$REPO/scripts/distill/tau2'   # ★git 은 부모가 wait 뒤 한 번만(index lock 다툼 방지)
    done
  }
  ( run_arm treat 8141 ) > $LOG/bank_t7333_treat_chain.log 2>&1 &
  P1=\$!
  ( run_arm ctl 8140 )   > $LOG/bank_t7333_ctl_chain.log 2>&1 &
  P2=\$!
  wait \$P1 \$P2

  echo '[t7333] 네 런 완료 — 영속화'
  cd '$REPO'
  git add -f reports/facet_rft_2026/sim_results/bank_t7333_*.gz
  git -c user.name=ghlee -c user.email=beingrelative@gmail.com commit -q -m 't7333 arms' || true
  git push -q origin facet-rft-2026 || true
  for A in ctl treat; do for P in hot rest; do
    G=reports/facet_rft_2026/sim_results/bank_t7333_\${A}_\${P}_20260821b.results.json.gz
    git ls-files --error-unmatch \$G >/dev/null 2>&1 \\
      && echo \"[t7333] \$A/\$P 영속 확인 tracked\" || echo \"[t7333] ⚠\$A/\$P 영속 실패 — 리모트 디스크가 유일본\"
  done; done
  /home/woori/venvs/seka_env/bin/python reports/facet_rft_2026/freeze.py --off || true
  echo '[t7333] 동결 해제 · 끝'
" </dev/null > "$LOG/bank_t7333_chain.log" 2>&1 &

echo "[t7333] 기동 PID=$! · sha=$SHA · 스모크 2 + 본런 40 sim · ctl=8140 / treat=8141"
