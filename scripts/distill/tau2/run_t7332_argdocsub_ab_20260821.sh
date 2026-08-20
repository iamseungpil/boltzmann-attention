#!/usr/bin/env bash
# t7332 — **A2 선언 문서를 격리 서브에 배달**하는 첫 라이브 A/B (사용자 지시 2026-08-21 *"라이브로 A/B 돌려라"*)
#
# ★사전등록 (런 시작 전에 못 박는다 · 결과 보고 바꾸지 않는다)
#   질문 : A2 가 선언한 문서를 **격리 서브 하나**에 배달하면 **reward 를 사는가**
#   종점 : **reward 뿐**([[69]]). gold 일치율·범주 정확도는 성적이 아니라 진단이다.
#   설계 : 단일 변수 A/B. 변수 = `T2_ARG_DOC_SUB` 하나.
#          ctl   = unset (배달 0 = 종전 스택과 **바이트 동일**·`test_arg_doc_sub.py` ⒜ 로 고정)
#          treat = 1  (선언 12편을 서브에 넘기고, 서브가 값을 내고, 엔진은 인용 실재만 검산)
#   표적 : **레버가 발화할 수 있는 태스크만** — `check_card_application_fit` 이 `spend_category` 를
#          달고 실제로 불린 sim 이 있는 태스크 상위 8종(전수 131 중 117 = 89% 를 덮는다).
#          나머지 태스크는 이 블록에 **닿지 않아 바이트 동일**이므로 돈만 태운다.
#   규모 : 8 태스크 × nt=2 = 팔당 16 sim · 스모크 1 = **총 33 sim**
#   판정 : reward 합의 차. **판정선 = C483 잡음 바닥 ±4/40** ⇒ 16 sim 규모에서 **차 2 미만이면 null**
#   의무 3종([[70]]) : ⒜ 전체 reward 짝 ⒝ **태스크별 부호표** ⒞ 무엇을 팔았나
#   동급 계측(사전 고정) : ⒜ fit 이후 조회 수 ⒝ 날조(`_provenance_deny`) ⒞ over-action(쓰기)
#                          · Δspurious ≤ 0 (등대 §1.3)
#
# ★왜 이 레버인가 — [[62]] 사다리 (측정이 먼저 있었다)
#   격리 n=71 : 문서 **안 주면 44/71** · **선언 문서를 주면 71/71**(C576) ⇒ 격리에서 되므로
#               레버는 **전달뿐**이고 판단은 서브(=같은 모델)가 한다.
#   검색 불가 : bm25·dense 는 선언 12편 중 **11편을 71 사례에서 0회** 돌려준다(C577) —
#               이 자리의 판정 문서에 **검색으로는 닿지 않는다**.
#   ⛔격리에서 분리 안 된 자리도 사전에 적어 둔다: 선언 71/71 ↔ **아무 카드 문서 12편 70/71**.
#      즉 *"색인이 더 정확하다"* 는 아직 근거가 없다. 이 런이 사는 것은 **라이브 reward** 하나다.
#
# ⚠유료(user-sim = openrouter gpt-5.2). 두 팔을 **다른 GPU**에 올려 병렬(ctl=8140 · treat=8141).
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
T2_DECLFIRST=0 T2_DECLFIRST_GUIDE_FIX=0 T2_VALUE_FORMULA= T2_CATEGORY_CITE="

TASKS='task_003,task_024,task_025,task_001,task_070,task_055,task_047,task_063'
NT=2
MARK='T2_ARG_DOC_SUB'

for f in "$LOG/bank_t7332_"*.log; do [ -e "$f" ] && { echo "[t7332] REFUSING: $f 존재" >&2; exit 1; }; done

echo "{\"tag\":\"t7332\",\"sha\":\"$SHA\",\"tasks\":\"$TASKS\",\"nt\":$NT,\"arms\":[\"ctl(unset)\",\"treat(T2_ARG_DOC_SUB=1)\"],\"ports\":{\"ctl\":8140,\"treat\":8141},\"endpoint\":\"reward only\",\"bar\":\"|d| < 2 = null (C483 ±4/40)\",\"also\":[\"reads_after_fit\",\"fabs\",\"over-action\",\"per-task sign table\"],\"iso_basis\":[\"C576 44/71 -> 71/71\",\"C577 search misses 11/12\"],\"iso_caveat\":\"sham 70/71 = 색인 우위 미확인\"}" \
  | tee "$LOG/bank_t7332.meta.json"

setsid bash -c "
  cd '$REPO/scripts/distill/tau2'
  source ./go_stack.sh >/dev/null 2>&1
  export $PIN
  export GO_MAX_STEPS=150 GO_CONCURRENCY=1

  # ── 0단계 스모크: 배선이 라이브에서 **발화**하는지부터([[30]]·[[67]] 0단계·死배선 방지)
  export T2_ARG_DOC_SUB=1
  SMK=bank_t7332_smoke_20260821
  echo '[t7332] === 스모크(1 sim·task_024·treat·8141) ==='
  t2_launch \$SMK 8141 task_024 1
  N=\$(grep -c '$MARK' $LOG/\$SMK.log || echo 0)
  echo \"[t7332] 스모크 배달 발화 = \$N\"
  grep '$MARK' $LOG/\$SMK.log | head -5 || true
  cd '$REPO' && mkdir -p reports/facet_rft_2026/sim_results
  gzip -c '$SIMS/'\$SMK'/results.json' > reports/facet_rft_2026/sim_results/\$SMK.results.json.gz
  gzip -c $LOG/\$SMK.log > reports/facet_rft_2026/sim_results/\$SMK.log.gz
  git add -f reports/facet_rft_2026/sim_results/\$SMK.results.json.gz reports/facet_rft_2026/sim_results/\$SMK.log.gz
  git -c user.name=ghlee -c user.email=beingrelative@gmail.com commit -q -m 't7332 smoke' || true
  git push -q origin facet-rft-2026 || true
  cd '$REPO/scripts/distill/tau2'
  if [ \"\$N\" -eq 0 ]; then
    echo '[t7332] ⛔스모크 발화 0 — 본런을 돌리지 않는다(死배선에 돈을 쓰지 않는다)'
    exit 1
  fi

  # ── 코드 동결 (C423ⓒ·[[07]]): 런 도중 엔진이 바뀌면 그 런은 **어떤 SHA 로도 귀속되지 않는다**.
  #   스모크 **뒤**에 건다 — 그 전에 걸면 死배선 수리를 막는다.
  cd '$REPO'
  /home/woori/venvs/seka_env/bin/python reports/facet_rft_2026/freeze.py --on \
    --tag t7332 --reason 'live A/B of the declared-document delivery' || true
  cd '$REPO/scripts/distill/tau2'

  # ── 본런: 두 팔을 **다른 GPU**에 올려 병렬. 각 팔 안은 GO_CONCURRENCY=1(사용자 지시).
  run_arm() {
    ARM=\$1; PORT=\$2
    TAG=bank_t7332_\${ARM}_20260821
    if [ \"\$ARM\" = ctl ]; then unset T2_ARG_DOC_SUB; else export T2_ARG_DOC_SUB=1; fi
    env | grep -E '^T2_ARG_DOC_SUB' | sort > $LOG/env_\${TAG}.txt || true
    echo \"[t7332] === \$ARM 시작 · port=\$PORT · T2_ARG_DOC_SUB='\${T2_ARG_DOC_SUB:-(unset)}' ===\"
    t2_launch \$TAG \$PORT '$TASKS' $NT
    echo \"[t7332] \$ARM 완료 · 배달 발화=\$(grep -c '$MARK' $LOG/\$TAG.log || echo 0)\"
    cd '$REPO' && mkdir -p reports/facet_rft_2026/sim_results
    gzip -c '$SIMS/'\$TAG'/results.json' > reports/facet_rft_2026/sim_results/\$TAG.results.json.gz
    gzip -c $LOG/\$TAG.log > reports/facet_rft_2026/sim_results/\$TAG.log.gz
    cd '$REPO/scripts/distill/tau2'   # ★git 은 부모가 wait 뒤 한 번만(병렬 두 팔이 index lock 을 다툰다)
  }
  ( run_arm treat 8141 ) > $LOG/bank_t7332_treat_chain.log 2>&1 &
  P1=\$!
  ( run_arm ctl 8140 )   > $LOG/bank_t7332_ctl_chain.log 2>&1 &
  P2=\$!
  wait \$P1 \$P2
  echo '[t7332] 두 팔 완료 — 영속화'
  cd '$REPO'
  git add -f reports/facet_rft_2026/sim_results/bank_t7332_*.gz
  git -c user.name=ghlee -c user.email=beingrelative@gmail.com commit -q -m 't7332 arms' || true
  git push -q origin facet-rft-2026 || true
  for A in ctl treat; do
    G=reports/facet_rft_2026/sim_results/bank_t7332_\${A}_20260821.results.json.gz
    git ls-files --error-unmatch \$G >/dev/null 2>&1 \
      && echo \"[t7332] \$A 영속 확인 tracked\" || echo \"[t7332] ⚠\$A 영속 실패 — 리모트 디스크가 유일본\"
  done
  /home/woori/venvs/seka_env/bin/python reports/facet_rft_2026/freeze.py --off || true
  echo '[t7332] 동결 해제'
" </dev/null > "$LOG/bank_t7332_chain.log" 2>&1 &

echo "[t7332] 기동 PID=$! · sha=$SHA · 스모크 1 + 본런 32 sim · ctl=8140 / treat=8141"
