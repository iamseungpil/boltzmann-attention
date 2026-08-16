#!/bin/bash
# t7305 — **요구 주입 A/B**(2026-08-17·x343/x344v2 가 격리로 산 레버를 라이브에서 검정).
#
# 처치 = `T2_SUB_REQUIREMENT=1` 하나. 결정 서브의 `ask` 에 **손님 요구 인용**을 싣는다.
#   인용은 LLM 이 내고(A2 `requirement_prompt`·도메인 어휘 0) 엔진은 원문 **존재만** 확인한다
#   (`in`·정규식 0·C45 동형). 검증 통과 0 이면 아무것도 안 붙인다(fail-safe).
#
# 격리 근거(무료·det n=1·부정통제 통과):
#   savings: A_REF `Gold Account` ✗ → +요구 **`Silver Plus` ✓**(seed 2개 복제) → D_NEG 0
#   checking: A_REF `Blue` ✗ → +요구 **`Purple` ✓** → D_NEG(같은 축 다른 요구) `Blue` ✗
#   ⇒ 라이브 0/8 의 원인은 재료가 아니라 **요구가 서브에 없다**는 것(원장 C506·C507).
#
# ⚠양팔 공통으로 **정규식 이설**이 들어가 있다(2026-08-17): 두 검증기의 입력이
#   `parse_records`/`re.finditer` → `t2_search.sub_records`(LLM formalize + 존재확인).
#   라이브 미측정이므로 ⓐ에서 배선을 먼저 본다. 양팔 동일이라 비교는 성립한다.
#
# ★판정(사전 고정·이 순서로)
#   ⓐ배선  treat 에 `[T2_SUB_REQUIREMENT] 인용 … 통과` > 0 · ctl **0** · infra 0.
#          + `[T2_SUB_RECORDS]` 발화 수(양팔·이설 배선 생존 확인·0 이어도 종전과 같음).
#   ⓑ1차   **055 축별 최종 제출 gold 일치 합**(checking+savings·0~16) — `x341` 로 양팔 같은 코드.
#          **GO = ctl 대비 +5 이상**(전역 잡음 규칙). +1~4 미결.
#   ⓒ기전  `[T2_DOCDECIDE]` 값이 gold 클래스로 바뀌었는가(격리에서 본 이동이 라이브에 오는가).
#   ⓓ성적  `reward` 만(C486).
#   ⓔ부작용 지연(요구 서브 1회 추가) · **098 불변** · CWE · over-action.
#
# 편성: 055 nt=8 + 024·098 nt=4 = 16 sim/팔 = 32 sim (t7304j 와 동일·비교 가능).

set -e
REPO=/home/woori/workspace_common/boltzmann-attention-pi
cd "$REPO/scripts/distill/tau2"

LOG=/home/woori/scratch/logs
SIMS=/home/woori/scratch/tau2-bench/data/simulations
mkdir -p "$LOG"

SHA=$(cd "$REPO" && git rev-parse --short HEAD)
DIRTY=$(cd "$REPO" && git status --porcelain -- \
  scripts/distill/tau2/t2_gate_patch.py scripts/distill/tau2/t2_search.py \
  scripts/distill/tau2/t2_resolve.py scripts/distill/tau2/a2/ | grep -cv '^??' || true)
if [ "$DIRTY" != "0" ]; then
  echo "[t7305] REFUSING: 엔진 경로 커밋 안 된 변경 $DIRTY 개." >&2; exit 1
fi

for t in test_no_prose_regex.py test_sub_requirement.py test_docs_at_write.py \
         test_proceed_docbody.py test_cp2_clobber.py test_no_unbound_a2.py \
         test_deliver_precommit.py test_material_reserve.py test_material_bypass.py \
         test_probe_canonical.py test_log_join.py test_now_selfcall.py \
         test_no_undefined_names.py test_decision_carry.py test_subcall_return_type.py \
         test_a2_three_layer.py test_operator_find.py test_route_trace.py; do
  PYTHONPATH=/home/woori/scratch/tau2-bench/src /home/woori/venvs/seka_env/bin/python "$t" \
    >/dev/null 2>&1 || { echo "[t7305] REFUSING: $t FAIL" >&2; exit 1; }
done
echo "[t7305] VERIFY OK (18검)"

/home/woori/venvs/seka_env/bin/python x341_docbody_verdict.py --selftest > /tmp/x341_st.txt 2>&1 \
  || { echo "[t7305] REFUSING: x341 selftest 크래시" >&2; exit 1; }
grep -q "ctl   big부착 sim 0/12" /tmp/x341_st.txt \
  || { echo "[t7305] REFUSING: x341 양성통제 불일치" >&2; exit 1; }
echo "[t7305] x341 양성통제 OK"

if pgrep -f "[t]2_gap\.py" >/dev/null || pgrep -f "[x]3[0-9][0-9]_.*\.py" >/dev/null; then
  echo "[t7305] REFUSING: 무료 프로브 실행 중(양 포트 필요)" >&2; exit 1
fi

launch () {
  NAME="$1"; PORT="$2"; REQ="$3"
  TAG="bank_t7305_${NAME}_20260817a"
  if [ -e "$LOG/${TAG}.log" ]; then echo "[t7305] SKIP: ${TAG}.log 존재" >&2; return 0; fi
  if [ -e "$SIMS/${TAG}" ]; then echo "[t7305] REFUSING: $SIMS/${TAG} 잔존" >&2; return 1; fi
  echo "{\"tag\":\"$TAG\",\"scaffold_sha\":\"$SHA\",\"port\":$PORT,\"sub_requirement\":\"$REQ\",\"why\":\"requirement injection A/B (x343/x344v2 isolation -> live)\"}" \
    | tee "$LOG/${TAG}.meta.json"
  setsid bash -c "cd '$REPO/scripts/distill/tau2' && source ./go_stack.sh >/dev/null 2>&1 && \
    export T2_ACTION_SUB=1 T2_KEEP_DENY_BODY=1 T2_CALL_FORM=1 T2_ARG_EMPTY=1 \
           T2_SEARCH_AGENT=1 T2_DECIDE_ANY=1 \
           T2_WRITE_ARG_ENUM=1 T2_DECIDE_BEFORE_WRITE=1 T2_DECISION_CARRY=1 \
           T2_DISCOVERY_STEP2=1 T2_ARG_AXIS=1 T2_WRITE_SUB=3 T2_ACTION_INDEX=1 \
           T2_NOW_SELFCALL=1 T2_SEARCH_ON_PROCEED=1 && \
    export T2_ACT_DEMAND=0 T2_DELIVER_PRECOMMIT=0 T2_PROCEED_DOCBODY=0 T2_DOCS_AT_WRITE=0 \
           T2_SUB_REQUIREMENT=$REQ && \
    t2_launch $TAG $PORT 'task_055' 8 && \
    export T2_ACT_DEMAND=0 T2_DELIVER_PRECOMMIT=0 T2_PROCEED_DOCBODY=0 T2_DOCS_AT_WRITE=0 \
           T2_SUB_REQUIREMENT=$REQ && \
    t2_launch bank_t7305_${NAME}aux_20260817a $PORT 'task_024,task_098' 4" \
    </dev/null >"$LOG/${TAG}.log" 2>&1 &
  echo "[t7305] $NAME(sub_requirement=$REQ) → PID=$! port=$PORT"
}

launch ctl   8140 0
launch treat 8141 1
echo "[t7305] 기동 완료 · sha=$SHA · 팔당 16 sim(055×8 + 024×4 + 098×4)"
