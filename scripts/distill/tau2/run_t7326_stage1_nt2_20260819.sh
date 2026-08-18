#!/bin/bash
# t7326 (구 t7325·중단 후 재발사) — **1단계 20 태스크 × nt=2 밤샘 런** (사용자 지시 2026-08-19: *"1단계 20태스크 하기로 한걸
#         nt=2 로 하라"* · *"밤샘 런 결과 아침에 보고 싶다"*).
#
# ## 로스터 (20) — C542 의 그 20 이다
#   t7313(1단계 16) ∪ t7312(4) = **정확히 20**. 제외 명부([[68]]·`TASK_LEVER_MAP_AND_EXCLUSIONS`)의
#   005·102 는 애초에 이 로스터에 없고, 069 도 없다.
#
# ## ★재발사 사유 (2026-08-19·사용자 지시 *"멈추고 050 문제 확인후 nt=2 로 재런하라"*)
#   t7325 를 120초 만에 중단하고 050 을 정밀 포렌식했다. **회귀가 아니었다** — 우리 핀과
#   후속 힌트를 **두 출처 가드**(`operator-fab` · `unlock_prov`)가 각각 막고 있었고, 통과한 런은
#   문구가 이름을 먼저 말해 둔 **순서**였을 뿐이다. 지목한 이름을 한 곳에 적어 두 가드가 함께
#   보게 고쳤다(커밋 `e7ce0258`). 이 런이 그 수리를 태스크 20개에서 확인한다.
#
# ## ★050 수리 확인 (2026-08-19·t7327 A/B 두 GPU)
#   `_t2_our_names` 연결(커밋 e7ce0258) 뒤 050 을 양 GPU 에서 각각 돌려 **2/2 통과**
#   (각 reward 1.0 · 매치 12/13 · 읽기 루틴 4회 · `T2_UNLOCK_PROV` deny **0**).
#   수리 전 같은 시드에서 1.0/1.0/0.0 으로 흔들리던 것이
#   두 샘플 모두 통과로 붙었다. 이 런이 그것을 20 태스크로 넓힌다.
#
# ## 왜 nt=2 인가
#   오늘 같은 시드에서 뒤집힌 것이 둘이다 — 073(1.0 ↔ 0.0·레버 발화 0) · 050(1.0 두 번 뒤 0.0).
#   nt=1 은 레버와 잡음을 못 가른다. nt=2 는 **태스크별 변동폭**을 처음으로 재게 해 준다.
#
# ## 구성 — A/B 아님, **단일 스택**
#   오늘 만든 스택 그대로(읽기 루틴 · `T2_PROV_OURS` · `T2_NOREC_BRANCH` · `T2_GROUND_HDR` ·
#   `T2_RETURN_EMPTY` · 이관-원장). 측정 대상 노브(VC·EL·VG·CV·declfirst)는 **전부 0** —
#   A/B 는 나중에 한다(사용자 지시).
#
# ## 두 GPU 분배 (과거 소요로 균형)
#   80 sim 을 두 반으로 가른다. 예상 ≈ **7.8시간/GPU**(과거 태스크별 평균 × 2).
#   ⚠아침에는 **절반쯤** 끝나 있다. results.json 은 sim 마다 갱신되므로 중간 판독이 가능하다.
#   ⚠`GO_CONCURRENCY=1` 고정([[30]] 사용자 지시) — 동시성으로 줄이지 않는다.
#
# ## 영속 ([[30]] 결과 소실 방지)
#   각 팔이 끝나면 gz 로 묶어 repo `sim_results/` 에 넣고 **`git add -f` 까지** 한다.
#   (커밋·푸시는 사람이 아침에 확인 후 — 밤중 자동 푸시는 하지 않는다.)

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
  echo "[t7326] REFUSING: 엔진 경로 커밋 안 된 변경 $DIRTY 개." >&2; exit 1
fi

for t in test_a2_three_layer.py test_flag_registry.py test_claim_verify.py \
         test_claim_tool_index.py test_read_routine.py test_proc_read_connect.py \
         test_verdict_gate.py test_verdict_carry.py test_no_undefined_names.py \
         test_no_unbound_a2.py test_quote_in.py test_args_equal.py test_t2_procedure.py \
         test_proc_absent_wiring.py test_pin_read_replay.py test_eplan.py \
         test_decision_carry.py test_route_trace.py test_group_parse.py \
         test_resolve_cap_runtime.py test_byref_repairs.py test_no_prose_regex.py; do
  [ -f "$t" ] || continue
  PYTHONPATH=/home/woori/scratch/tau2-bench/src timeout 90 \
    /home/woori/venvs/seka_env/bin/python "$t" >/dev/null 2>&1 \
    || { echo "[t7326] REFUSING: $t FAIL" >&2; exit 1; }
done
echo "[t7326] VERIFY OK (배터리 22)"

if pgrep -f "[t]2_launch" >/dev/null; then
  echo "[t7326] REFUSING: 다른 라이브 런이 돌고 있다" >&2; exit 1
fi

PIN="T2_ACTION_SUB=1 T2_KEEP_DENY_BODY=1 T2_CALL_FORM=1 T2_ARG_EMPTY=1 T2_SEARCH_AGENT=1 \
T2_DECIDE_ANY=1 T2_WRITE_ARG_ENUM=1 T2_DECIDE_BEFORE_WRITE=1 T2_DECISION_CARRY=1 \
T2_DISCOVERY_STEP2=1 T2_ARG_AXIS=1 T2_WRITE_SUB=3 T2_ACTION_INDEX=1 T2_NOW_SELFCALL=1 \
T2_SEARCH_ON_PROCEED=1 T2_ACT_DEMAND=0 T2_DELIVER_PRECOMMIT=0 T2_PROCEED_DOCBODY=0 \
T2_DOCS_AT_WRITE=0 T2_SUB_REQUIREMENT=0 T2_HANDOFF_PREDICATE=0 T2_PENDING_DISCOVERED=0 \
T2_VERDICT_CARRY=0 T2_ELIG_LINE=0 T2_VERDICT_GATE=0 T2_CLAIM_VERIFY=0 \
T2_DECLFIRST=0 T2_DECLFIRST_GUIDE_FIX=0"

# 과거 소요 평균으로 균형 분배 (긴 것부터 번갈아)
HALF_A='task_003,task_004,task_017,task_024,task_055,task_072,task_073,task_093,task_094,task_100'
HALF_B='task_016,task_033,task_040,task_050,task_057,task_063,task_074,task_079,task_085,task_098'
NT=2

launch () {
  NAME="$1"; PORT="$2"; TASKS="$3"
  TAG="bank_t7326_${NAME}_20260819q"
  if [ -e "$LOG/${TAG}.log" ]; then echo "[t7326] SKIP: ${TAG}.log 존재" >&2; return 0; fi
  if [ -e "$SIMS/${TAG}" ]; then echo "[t7326] REFUSING: $SIMS/${TAG} 잔존" >&2; return 1; fi
  echo "{\"tag\":\"$TAG\",\"scaffold_sha\":\"$SHA\",\"port\":$PORT,\"tasks\":\"$TASKS\",\"nt\":$NT,\"max_steps\":150,\"concurrency\":1,\"pin\":\"$PIN\",\"why\":\"stage-1 20 tasks x nt=2 overnight; single stack, no arm knobs; endpoint = per-task pass distribution + variance\"}" \
    | tee "$LOG/${TAG}.meta.json"
  setsid bash -c "cd '$REPO/scripts/distill/tau2' && source ./go_stack.sh >/dev/null 2>&1 && \
    export $PIN && export GO_MAX_STEPS=150 GO_CONCURRENCY=1 && \
    t2_launch $TAG $PORT '$TASKS' $NT ; \
    cd '$REPO' && mkdir -p reports/facet_rft_2026/sim_results && \
    gzip -c '$SIMS/$TAG/results.json' > reports/facet_rft_2026/sim_results/${TAG}.results.json.gz && \
    git add -f reports/facet_rft_2026/sim_results/${TAG}.results.json.gz && \
    echo '[t7326] persisted ${TAG}'" \
    </dev/null >"$LOG/${TAG}.log" 2>&1 &
  echo "[t7326] $NAME → PID=$! port=$PORT ($(echo $TASKS | tr ',' '\n' | wc -l) 태스크 × nt=$NT)"
}

launch halfA 8140 "$HALF_A"
sleep 2
launch halfB 8141 "$HALF_B"
echo "[t7326] 기동 · sha=$SHA · 총 40 sim · 예상 ≈7.8시간/GPU"
echo "[t7326] 종점 = 태스크별 pass 분포와 **변동폭**(오늘 050·073 이 같은 시드에서 뒤집혔다)"
