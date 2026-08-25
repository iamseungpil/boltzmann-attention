#!/bin/bash
# t7354 — **hard-0 표적 재런 · 074 제외판** (사용자 지시 2026-08-25: *"재발사하고 074 분석을 더하라"*)
#
# ## 왜 074 를 뺐나 — **0/5 이고 막는 결손에 출시할 수리가 없다**
#   t7348 0/2 · t7352 0/1 · t7353 0/1(87.1분). 배선 수리(47ef453c)는 성공했다:
#   `operand keys=['transactions']` 복귀 · `[T2_SG_PROMPT_V2]` 9회. 그런데 그 위에서
#   전사가 계약과 안 맞아(sub=17 ↔ 기대 16 · 날조 id `btxn_ar_lb_08f` 가 select_discrepant
#   출력에 실림) 재시도가 9회 돌고 **`context_window_exceeded`** 로 죽어 gold write 에 못 갔다.
#   x533 이 그 초과 행에 **출시할 수리가 없다**고 판정했으므로(팔마다 한 계좌씩만 고치고
#   합치면 상쇄) nt2 에 ~170분을 쓰는 것은 이미 아는 것을 다시 사는 일이다.
#
# ## 왜 t7352 를 죽였나 — **074 수리가 라이브에 도달하지 못했고 상류를 끊었다**
#   t7352 grpA1 축자: `fetch get_atm_fee_discrepancies: 2라운드·getter 1회·operand keys=**[]**`
#   · `'transactions' 인자 str 잔류(JSON 파싱실패)` · `select_discrepant: **9/17행 판정불가**`
#   대조 t7348(V2 off): `operand keys=**['transactions']**` · 판정불가 0
#   원인: V2 는 `answer_format` 을 머리에서 빼고 마감 메시지로 옮기는데 그 메시지가
#   `_tl is None`(**마지막** 라운드)에만 붙었다. 이 서브는 `max_rounds=3` 인데 **라운드 1에
#   답한다** ⇒ 형식 지시를 한 번도 못 본 채 답 ⇒ 파싱 실패. 수리 = 47ef453c.
#
# ## t7352 에서 배운 것 — 이 드라이버가 막는 결함 셋
#   ⑴ **V2 게이트 부재**: t7352 스모크를 016(싸다)으로 바꾸면서 t7351 의 `NV2 -eq 0 → 중단`
#      게이트를 잃었다. 016 은 V2 경로를 지나가지 않는다.
#      ⇒ 스모크를 **074 nt1** 로 되돌리고, 게이트를 마커가 아니라 **수리의 산출**로 건다:
#        `operand keys=['transactions']` 가 돌아왔는가. 그 sim 은 버리지 않고 **실 trial 로 센다**.
#   ⑵ **사이드카 경로 상속**: `t2_launch` 가 `: "${T2_FB_SIDECAR:=…}"` 로 **미설정일 때만** 잡는데
#      스모크가 먼저 export 해서 **뒤 배치가 스모크 파일에 같이 썼다**(t7352 실물 2.4MB·4 태스크
#      혼재). ⇒ 배치마다 `unset` 한다.
#   ⑶ **정수 비교**: `grep -ac … || echo 0` 이 매치 0 일 때 `"0\n0"` 을 만들어 게이트가 죽었다.
#      ⇒ `|| true` + `${VAR:-0}`.
#
# ## 시간상자 (t7348 per-sim 실측 분으로 산정 · 마감 가드가 안 맞는 배치를 건너뛰고 로그에 남긴다)
#   016 12.7 · 040 87.1 · 057 27.0 · 063 15.5 · 072 19.9 · 073 20.4 · 074 39.3 · 085 37.0
#   ★결과는 **증분으로** 쓰인다(t7352 실측: 배치 도중 results.json 이 갱신돼 완료 sim 3건을
#     그대로 회수했다). 그래도 배치를 쪼개는 이유는 **태그별 커밋**이라 회수가 자동이기 때문이다.
#
# ## 대조와 판정선
#   대조 = t7348(sha aed30e20) — 이 로스터 전부 0/2. 판정선 = **표적의 0→1**.
#   ⛔총점 Δ 금지 · ⛔묶음이라 개별 귀속 금지(C594).
#   ⛔이 런도 `T2_SPEC_AT_WRITE` 가 **무엇을 파는지 못 잰다**(로스터가 전부 hard-0).
set -e
REPO=/home/woori/workspace_common/boltzmann-attention-pi
LOG=/home/woori/scratch/logs
SIMS=/home/woori/scratch/tau2-bench/data/simulations
REP=$REPO/reports/facet_rft_2026
TAGBASE=bank_t7354
DEADLINE_HHMM=${DEADLINE_HHMM:-1830}
mkdir -p "$LOG"
cd "$REPO/scripts/distill/tau2"
SHA=$(cd "$REPO" && git rev-parse --short HEAD)
say() { echo "[t7354 $(date +%H:%M:%S)] $*"; }

DIRTY=$(cd "$REPO" && git status --porcelain -- \
  scripts/distill/tau2/t2_gate_patch.py scripts/distill/tau2/t2_search.py \
  scripts/distill/tau2/t2_resolve.py scripts/distill/tau2/t2_scaffold_get.py \
  scripts/distill/tau2/go_stack.sh scripts/distill/tau2/a2/ | grep -cv '^??' || true)
[ "$DIRTY" = "0" ] || { say "REFUSING: 엔진 경로 미커밋 $DIRTY"; exit 1; }

for t in test_a2_three_layer.py test_flag_registry.py test_no_undefined_names.py \
         test_no_unbound_a2.py test_quote_in.py test_args_equal.py test_t2_procedure.py \
         test_sg_docs_delivery.py test_sg_fetch_iso.py test_sg_isofb.py \
         test_sg_prompt_v2_reachable.py \
         test_atm_ledger_close.py test_compute_params.py test_write_arg_enum.py \
         test_write_arg_enum_values.py test_spec_at_write.py \
         test_result_round.py test_apy_balance_tier.py test_ref_from_outputs.py \
         test_no_prose_regex.py test_ours_text_canonical.py test_regen_break_guard.py; do
  [ -f "$t" ] || continue
  PYTHONPATH=/home/woori/scratch/tau2-bench/src timeout 90 \
    /home/woori/venvs/seka_env/bin/python "$t" >/dev/null 2>&1 || { say "REFUSING: $t FAIL"; exit 1; }
done
say "VERIFY OK (배터리 22 · 새 래칫 test_sg_prompt_v2_reachable 포함)"

pgrep -f "[t]2_launch" >/dev/null && { say "REFUSING: 다른 라이브 런"; exit 1; } || true
pgrep -f "[t]2_run_gated" >/dev/null && { say "REFUSING: 잔존 sim 프로세스"; exit 1; } || true
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

echo "{\"tag\":\"t7354\",\"sha\":\"$SHA\",\"design\":\"relaunch of t7352 after the transcription repair was found dead in the live wiring; smoke is a real 074 trial gated on the repair's own output\",\"on\":\"$ON\",\"reference\":\"t7348 sha aed30e20 - every task in this roster was 0/2; t7352 was stopped after 3 sims (074 t0, 085 t0, 040 t0, all reward 0)\",\"bar\":\"a target going 0 -> 1; never judge on a total delta; bundle so no individual attribution (C594)\",\"cannot_measure\":\"what T2_SPEC_AT_WRITE sells - the roster is all hard-0\",\"deadline\":\"$DEADLINE_HHMM\"}" \
  | tee "$LOG/${TAGBASE}.meta.json"

setsid bash -c "
  cd '$REPO/scripts/distill/tau2'
  source ./go_stack.sh >/dev/null 2>&1
  export $PIN
  export $ON
  export GO_MAX_STEPS=150 GO_CONCURRENCY=1

  fits() {
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
        echo \"[t7354] WARN \${_S} 미회수 \$_F — 우리-층 귀속 판정 불가([[25]])\"
      fi
    done
    git add -f reports/facet_rft_2026/sim_results/\$TAG*.gz \\
               reports/facet_rft_2026/sim_results/fb_\$TAG*.jsonl.gz \\
               reports/facet_rft_2026/sim_results/trace_\$TAG*.jsonl.gz 2>/dev/null || true
    git -c user.name=ghlee -c user.email=beingrelative@gmail.com commit -q -m \"t7354 batch \$TAG\" || true
    git push -q origin facet-rft-2026 || echo '[t7354] push 보류(원격 선행) — 런 종료 후 화해'
    git ls-files --error-unmatch reports/facet_rft_2026/sim_results/\$TAG.results.json.gz >/dev/null 2>&1 \\
      && echo \"[t7354] \$TAG persisted+tracked OK\" || echo \"[t7354] ⛔\$TAG NOT TRACKED\"
    cd '$REPO/scripts/distill/tau2'
  }

  # ★⑵ 사이드카 경로는 배치마다 **다시 잡는다** — t2_launch 의 \`:=\` 는 미설정일 때만 쓴다.
  fresh_sidecar() { unset T2_FB_SIDECAR T2_TRACE; }

  batch() {  # \$1 이름 · \$2 포트 · \$3 태스크목록 · \$4 nt · \$5 추정분
    local NAME=\$1 PORT=\$2 TL=\$3 NT=\$4 EST=\$5 TAG=${TAGBASE}_\$1_20260825
    if ! fits \$EST; then
      echo \"[t7354] SKIP \$NAME (추정 \${EST}분 · 마감 $DEADLINE_HHMM 초과) — 이 배치는 **안 돌았다**\"
      return 0
    fi
    fresh_sidecar
    echo \"[t7354 \$(date +%H:%M:%S)] === \$NAME · \$TL · nt=\$NT · 추정 \${EST}분 ===\"
    t2_launch \$TAG \$PORT \"\$TL\" \$NT 2>&1 | tee $LOG/\$TAG.log
    local V2 SD SW OK TB
    V2=\$(grep -ac 'T2_SG_PROMPT_V2' $LOG/\$TAG.log || true); V2=\${V2:-0}
    SD=\$(grep -ac 'T2_SPEC_AT_WRITE' $LOG/\$TAG.log || true); SD=\${SD:-0}
    SW=\$(grep -ac 'T2_SUBWIN' $LOG/\$TAG.log || true); SW=\${SW:-0}
    OK=\$(grep -ac \"operand keys=\['transactions'\]\" $LOG/\$TAG.log || true); OK=\${OK:-0}
    TB=\$(grep -ac 'Traceback' $LOG/\$TAG.log || true); TB=\${TB:-0}
    echo \"[t7354] \$NAME 완료 · PROMPT_V2=\$V2 · SPEC_AT_WRITE=\$SD · SUBWIN=\$SW · operandOK=\$OK · Traceback=\$TB\"
    persist \$TAG
  }

  # ── ★스모크 없음 — **같은 sha 가 방금 라이브로 돌았다**.
  #    t7354 스모크(074 nt1 · 87.1분 · Traceback 0)가 이 sha 의 스모크이고, 그 로그가
  #    수리 도달까지 확인했다: `operand keys=['transactions']` 복귀 · `[T2_SG_PROMPT_V2]`
  #    9회 발화. 엔진은 그 뒤로 한 글자도 안 변했다(freeze 검사 `동결 경로 무변` 통과).
  #    ⇒ 또 태우면 남은 창의 40분을 사는 일이라 생략한다([[09]] 무료 검증 우선).

  cd '$REPO'
  /home/woori/venvs/seka_env/bin/python reports/facet_rft_2026/freeze.py --on \\
    --tag t7354 --reason 'relaunch after the transcription repair reached the sub' || true
  cd '$REPO/scripts/distill/tau2'

  (
    batch grpA1 8140 'task_085'          2  50
    batch grpA2 8140 'task_057'          2  60
    batch grpA3 8140 'task_016'          2  28
    batch grpA4 8140 'task_063'          2  35
  ) > $LOG/${TAGBASE}_grpA_chain.log 2>&1 &
  P1=\$!
  (
    batch grpB1 8141 'task_040'          2 140
    batch grpB2 8141 'task_085'          2  50
    batch grpB3 8141 'task_072'          2  45
  ) > $LOG/${TAGBASE}_grpB_chain.log 2>&1 &
  P2=\$!
  wait \$P1 \$P2

  cd '$REPO'
  cp $LOG/${TAGBASE}.meta.json reports/facet_rft_2026/sim_results/${TAGBASE}.meta.json || true
  git add -f reports/facet_rft_2026/sim_results/${TAGBASE}.meta.json
  git -c user.name=ghlee -c user.email=beingrelative@gmail.com commit -q -m 't7354 meta' || true
  git push -q origin facet-rft-2026 || echo '[t7354] push 보류 — 화해 필요'
  /home/woori/venvs/seka_env/bin/python reports/facet_rft_2026/freeze.py --off --tag t7354 || true
  echo '[t7354] ALL DONE'
" </dev/null >"$LOG/${TAGBASE}_chain.log" 2>&1 &
say "기동 PID=$! · sha=$SHA · 마감 $DEADLINE_HHMM · 로그 $LOG/${TAGBASE}_chain.log"
