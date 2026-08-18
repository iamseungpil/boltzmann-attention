#!/bin/bash
# t7310 — **S2 배선 스모크**(`PASS_PATH_V2_2026_08_17.md` §S2 · 2026-08-18 재설계판).
#
# ## 무엇을 사는가
#
# S3(최종 census)의 두 팔과 **바이트 동일한 구성**을 미리 돌려, 신규 레버가 **실제로 발화하는지**를
# 유료 15시간 런 전에 확인한다. 이것은 성적 실험이 아니다 — **배선 리허설**이다.
#
# ## ⚠사전 고정 판정 (결과보다 먼저 · 이 순서로)
#
#   ⓐ**1차 종점 = 레버별 실발화율**(pass 아님·C525→C529 의 교훈을 S2 전체로 확장):
#       `[T2_VERDICT]` · `[T2_ELIG]` 가 **트리거 자리 대비 ≥90%** 발화하는가.
#       ⚠분모(트리거 자리)는 **마커가 아닌 곳**에서 센다 — 마커로 분모를 만들면 t7303 ⓑ 형 순환이다.
#   ⓑ**오염 검사**: ctl 에서 신규 마커는 **0** 이어야 한다.
#   ⓒ**순서·소모 채널**(C517⒟ 가 *라이브에서만 잰다* 고 남긴 것): `[T2_GROUPORDER]` 로
#       **첫 지목 군**과 **소모한 결정점 수**를 기록. 격리에선 후보집합 채널 이득이 0 이었다(HIT 27/27).
#   ⓓ**부작용**: 크래시 0 · 지연 **≤1.5×** · treat 가 지는 태스크 **≤2**.
#   ⛔**pass 는 판정에 쓰지 않는다** — 12 태스크로는 검정력이 없다. pass 를 읽는 순간 이 런은
#     우리가 폐기한 *"레버당 유료 런"* 이 된다(2026-08-17 규율).
#
# ## 편성 (사전 고정 · 사후 선택 금지)
#
#   선택축(L-V + ELIG) 055·057·063·024 / 077~097 블록 079·093 / 계산·끝맺음 072·073 /
#   회귀 감시 098·100·004·016(뒤 둘 = **어느 레버도 안 닿는 자리**가 정말 불변인가)
#   ⚠019 가족 제외 — t7307/t7308 이 이미 그 가족을 따로 쟀다(중복 발사 금지).
#
# ⚠**문서의 산술 불일치를 여기서 확정한다**: §S2 표는 *"12 태스크 × nt1 × 2팔 = 24 sim"* 이고
#   편성 문단은 *"nt=2 × 2팔 = 24 sim"* 이라 적었다. 12×2×2 = **48** 이므로 후자가 오기다.
#   sim 수(24)와 비용 추정(≈2.7h)에 맞는 **nt=1** 로 간다. 1차 종점이 발화율이라 시행 반복이
#   필요 없고, [[09]] 비용 규율상 작은 쪽이 안전측이다.
#
# ## 구성
#
#   ctl   = t7308 축자 PIN(신규 레버 전부 OFF)
#   treat = ctl + `T2_VERDICT_CARRY=1` + `T2_ELIG_LINE=1`
#   · `T2_HANDOFF_PREDICATE` **미포함** — C529 가 폐기했다(닿지만 표적 부재·pass null·CWE 13건).
#   · `T2_PENDING_DISCOVERED` **미포함** — 보류 유지(§S2 F3·x370 형식 재검 전).
#   · W4 는 A2 라 **양팔 공통**이다(런처가 못 가른다).
#   · `GO_MAX_STEPS=150` — t7307 표류(두 sim × 7,800초) 재발 방지. 관측된 pass 최대 99 step 위로
#     여유를 두며, 1차 종점(발화)은 대화 초반에 정해지므로 이 상한에 영향받지 않는다.
#   · **`GO_CONCURRENCY=1`(사용자 지시 2026-08-18 축자: *"concurrency 쓰지 말라. KV caching 으로
#     인해 더 늦어지는거 같다. 한 태스크 씩 진행하라"*)** — 이 재런(t7310)이 t7309(동시 6)를
#     대체한다. 긴 문맥 sim 을 동시에 물리면 vLLM prefix 캐시가 서로를 밀어내 매 요청이
#     재-prefill 되므로, 동시성이 **처리량을 사고 지연을 판다**. 팔은 GPU 가 달라 여전히 병렬이다.
#     ⚠**측정 기회**: t7308(동시 6·sim 당 ctl 5,662s / treat 10,765s)과 **sim 당 지연**을 비교하면
#       이 지시의 근거를 실측으로 확인할 수 있다 — 판정 때 함께 기록한다.

set -e
REPO=/home/woori/workspace_common/boltzmann-attention-pi
cd "$REPO/scripts/distill/tau2"

LOG=/home/woori/scratch/logs
SIMS=/home/woori/scratch/tau2-bench/data/simulations
mkdir -p "$LOG"

SHA=$(cd "$REPO" && git rev-parse --short HEAD)
DIRTY=$(cd "$REPO" && git status --porcelain -- \
  scripts/distill/tau2/t2_gate_patch.py scripts/distill/tau2/t2_search.py \
  scripts/distill/tau2/t2_resolve.py scripts/distill/tau2/t2_scaffold_get.py \
  scripts/distill/tau2/a2/ | grep -cv '^??' || true)
if [ "$DIRTY" != "0" ]; then
  echo "[t7310] REFUSING: 엔진 경로 커밋 안 된 변경 $DIRTY 개." >&2; exit 1
fi

for t in test_no_prose_regex.py test_sub_requirement.py test_docs_at_write.py \
         test_proceed_docbody.py test_cp2_clobber.py test_no_unbound_a2.py \
         test_deliver_precommit.py test_material_reserve.py test_material_bypass.py \
         test_probe_canonical.py test_log_join.py test_now_selfcall.py \
         test_no_undefined_names.py test_decision_carry.py test_subcall_return_type.py \
         test_a2_three_layer.py test_operator_find.py test_route_trace.py \
         test_group_parse.py test_verdict_carry.py test_pending_discovered.py \
         test_probe_scoring.py test_quote_in.py test_elig_handoff.py test_args_equal.py \
         test_flag_registry.py test_resolve_cap_marker.py test_byref_repairs.py; do
  [ -f "$t" ] || continue
  PYTHONPATH=/home/woori/scratch/tau2-bench/src timeout 60 \
    /home/woori/venvs/seka_env/bin/python "$t" >/dev/null 2>&1 \
    || { echo "[t7310] REFUSING: $t FAIL" >&2; exit 1; }
done
echo "[t7310] VERIFY OK"

if pgrep -f "[t]2_gap\.py" >/dev/null || pgrep -f "[x]3[0-9][0-9]_.*\.py" >/dev/null; then
  echo "[t7310] REFUSING: 무료 프로브 실행 중(양 포트 필요)" >&2; exit 1
fi

PIN="T2_ACTION_SUB=1 T2_KEEP_DENY_BODY=1 T2_CALL_FORM=1 T2_ARG_EMPTY=1 T2_SEARCH_AGENT=1 \
T2_DECIDE_ANY=1 T2_WRITE_ARG_ENUM=1 T2_DECIDE_BEFORE_WRITE=1 T2_DECISION_CARRY=1 \
T2_DISCOVERY_STEP2=1 T2_ARG_AXIS=1 T2_WRITE_SUB=3 T2_ACTION_INDEX=1 T2_NOW_SELFCALL=1 \
T2_SEARCH_ON_PROCEED=1 T2_ACT_DEMAND=0 T2_DELIVER_PRECOMMIT=0 T2_PROCEED_DOCBODY=0 \
T2_DOCS_AT_WRITE=0 T2_SUB_REQUIREMENT=0 T2_HANDOFF_PREDICATE=0 T2_PENDING_DISCOVERED=0"
TASKS='task_055,task_057,task_063,task_024,task_079,task_093,task_072,task_073,task_098,task_100,task_004,task_016'

launch () {
  NAME="$1"; PORT="$2"; VC="$3"; EL="$4"
  TAG="bank_t7310_${NAME}_20260818e"
  if [ -e "$LOG/${TAG}.log" ]; then echo "[t7310] SKIP: ${TAG}.log 존재" >&2; return 0; fi
  if [ -e "$SIMS/${TAG}" ]; then echo "[t7310] REFUSING: $SIMS/${TAG} 잔존" >&2; return 1; fi
  echo "{\"tag\":\"$TAG\",\"scaffold_sha\":\"$SHA\",\"port\":$PORT,\"tasks\":\"$TASKS\",\"nt\":1,\"verdict_carry\":\"$VC\",\"elig_line\":\"$EL\",\"max_steps\":150,\"concurrency\":1,\"pin\":\"$PIN\",\"why\":\"S2 wiring smoke: primary endpoint is per-lever firing rate, pass is not judged\"}" \
    | tee "$LOG/${TAG}.meta.json"
  setsid bash -c "cd '$REPO/scripts/distill/tau2' && source ./go_stack.sh >/dev/null 2>&1 && \
    export $PIN && export T2_VERDICT_CARRY=$VC T2_ELIG_LINE=$EL && \
    export GO_MAX_STEPS=150 GO_CONCURRENCY=1 && \
    t2_launch $TAG $PORT '$TASKS' 1" \
    </dev/null >"$LOG/${TAG}.log" 2>&1 &
  echo "[t7310] $NAME(VC=$VC ELIG=$EL) → PID=$! port=$PORT"
}

launch ctl   8140 0 0
launch treat 8141 1 1
echo "[t7310] 기동 완료 · sha=$SHA · 팔당 12 sim · max_steps=150 · 1차 종점 = 레버별 발화율(pass 아님)"
