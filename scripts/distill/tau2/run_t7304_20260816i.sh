#!/bin/bash
# t7304 — **결정-자리 배달 객체 교체**(2026-08-16·S1 재설계·설계서 = S1_REDESIGN_T7304_2026_08_16.md).
#
# 가설 H. 라이브 055 의 선택 실패는 결정 자리에 놓인 **객체**가 격리와 다르기 때문이다.
#   실측: 그 자리에 배달되던 것은 서브의 **오답 결정문 247자**(055 양팔 `DOCDECIDE → 'Blue
#   Account'`·gold Purple)였고, 격리 24/24(x335b)를 만든 문서 본문은 도달한 적 없다(C502).
#
# 처치 = `T2_PROCEED_DOCBODY=1` 하나. SEARCH_ON_PROCEED 자리의 `_search_material` 에
#   `decide=False` — 같은 자리·같은 예산(3)·같은 슬롯·같은 축 소비, **객체만** 문서 본문.
#   ⚠t7303 의 `T2_DELIVER_PRECOMMIT` 은 양팔 다 **0**(그 처치는 쓰지 않는다).
#
# ★판정(사전 고정·설계서 §4·이 순서로만·실패 시 아래를 읽지 않는다)
#   ⓐ 배선   **부착 기준·양팔 동일 계기**: treat 055·024 각 sim ≥10k 부착 ≥1 · ctl ≥10k 부착 0
#            · 대용량 CLOBBER 0 · skip 이 055 4/4 면 실패 · infra 0. (`x341` ⓐ)
#   ⓑ 1차   **축별 최종 제출 클래스 gold 일치**(궤적에서 양팔 같은 코드로): 055 합산(0~8)이
#            ctl 대비 **+4 이상 = GO** · +1~3 = 미결. 참고: t7303 ctl = checking 1/4·savings 0/4.
#   ⓒ 기전   이동 sim 직독(부착 턴 ↔ 최종 클래스 첫 발화).
#   ⓓ 성적   `reward` 만(C486) · 차 ≥3 만 언급.
#   ⓔ 부작용 CWE 0(가드 skip 수 병기) · **098 3/4±1 불변** · duration/write · 군 오선택.
#   ⚠n=4/태스크·잡음 ±4 ⇒ ⓑ 문턱 밖 수치 인용 금지. 이 런은 **ⓑ 를 사러 간다**.
#
# 편성: 055(표적·2축) + 024(보조·1 write) + 098(부정통제) × nt=4 × 2팔 = 24 sim.

set -e
REPO=/home/woori/workspace_common/boltzmann-attention-pi
cd "$REPO/scripts/distill/tau2"

NT=4
TASKS=task_055,task_024,task_098
LOG=/home/woori/scratch/logs
SIMS=/home/woori/scratch/tau2-bench/data/simulations
mkdir -p "$LOG"

SHA=$(cd "$REPO" && git rev-parse --short HEAD)
DIRTY=$(cd "$REPO" && git status --porcelain -- \
  scripts/distill/tau2/t2_gate_patch.py scripts/distill/tau2/t2_search.py \
  scripts/distill/tau2/t2_resolve.py scripts/distill/tau2/a2/ | grep -cv '^??' || true)
if [ "$DIRTY" != "0" ]; then
  echo "[t7304] REFUSING: 엔진 경로 커밋 안 된 변경 $DIRTY 개." >&2; exit 1
fi

for t in test_proceed_docbody.py test_cp2_clobber.py test_no_unbound_a2.py \
         test_deliver_precommit.py test_material_reserve.py test_material_bypass.py \
         test_probe_canonical.py test_log_join.py test_now_selfcall.py \
         test_no_undefined_names.py test_decision_carry.py test_subcall_return_type.py \
         test_a2_three_layer.py test_operator_find.py test_route_trace.py; do
  PYTHONPATH=/home/woori/scratch/tau2-bench/src /home/woori/venvs/seka_env/bin/python "$t" \
    >/dev/null 2>&1 || { echo "[t7304] REFUSING: $t FAIL" >&2; exit 1; }
done
echo "[t7304] VERIFY OK (15검)"

# ★양성 통제: 판정 계기(x341)가 t7303 에서 기대값을 재현하는가 — 재현 실패면 발사 금지
#   (t7303 은 계기 결함을 런 뒤에 발견했다·같은 실수 반복 방지).
/home/woori/venvs/seka_env/bin/python x341_docbody_verdict.py --selftest > /tmp/x341_st.txt 2>&1 \
  || { echo "[t7304] REFUSING: x341 selftest 크래시" >&2; exit 1; }
grep -q "ctl   big부착 sim 0/12" /tmp/x341_st.txt \
  || { echo "[t7304] REFUSING: x341 양성통제 불일치(ctl big부착≠0)" >&2; exit 1; }
echo "[t7304] x341 양성통제 OK"

if pgrep -f "[t]2_gap\.py" >/dev/null || pgrep -f "[x]3[0-9][0-9]_" >/dev/null; then
  echo "[t7304] REFUSING: 무료 프로브 실행 중(양 포트 필요)" >&2; exit 1
fi

launch () {
  NAME="$1"; PORT="$2"; DOCB="$3"
  TAG="bank_t7304_${NAME}_20260816i"
  if [ -e "$LOG/${TAG}.log" ]; then echo "[t7304] SKIP: ${TAG}.log 존재" >&2; return 0; fi
  if [ -e "$SIMS/${TAG}" ]; then echo "[t7304] REFUSING: $SIMS/${TAG} 잔존" >&2; return 1; fi
  if ps -eo cmd | grep -v grep | grep "t2_run_gated.py" | grep -q "localhost:${PORT}/"; then
    echo "[t7304] REFUSING: 포트 ${PORT} 사용 중" >&2; return 1
  fi
  echo "{\"tag\":\"$TAG\",\"scaffold_sha\":\"$SHA\",\"tasks\":\"$TASKS\",\"port\":$PORT,\"nt\":$NT,\"docbody\":\"$DOCB\",\"why\":\"decision-site delivery OBJECT swap (S1 redesign after C502)\"}" \
    | tee "$LOG/${TAG}.meta.json"
  setsid bash -c "cd '$REPO/scripts/distill/tau2' && source ./go_stack.sh >/dev/null 2>&1 && \
    export T2_ACTION_SUB=1 T2_KEEP_DENY_BODY=1 T2_CALL_FORM=1 T2_ARG_EMPTY=1 \
           T2_SEARCH_AGENT=1 T2_DECIDE_ANY=1 \
           T2_WRITE_ARG_ENUM=1 T2_DECIDE_BEFORE_WRITE=1 T2_DECISION_CARRY=1 \
           T2_DISCOVERY_STEP2=1 T2_ARG_AXIS=1 T2_WRITE_SUB=3 T2_ACTION_INDEX=1 \
           T2_NOW_SELFCALL=1 T2_SEARCH_ON_PROCEED=1 && \
    export T2_ACT_DEMAND=0 T2_DELIVER_PRECOMMIT=0 T2_PROCEED_DOCBODY=$DOCB && \
    t2_launch $TAG $PORT '$TASKS' $NT" </dev/null >"$LOG/${TAG}.log" 2>&1 &
  echo "[t7304] $NAME(docbody=$DOCB) → PID=$! port=$PORT"
}

launch ctl   8140 0
launch treat 8141 1
echo "[t7304] 기동 완료 · sha=$SHA · nt=$NT · tasks=$TASKS"
