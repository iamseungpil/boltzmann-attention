#!/bin/bash
# t7308 — **t7307 재런**(019 가족 인계-술어 A/B). 사용자 지시 2026-08-17 밤 축자:
#   *"7307 결과 알려주고, 재런 해야 하면 수정해서 재런 하라."*
#
# ## 왜 재런인가 (C527 판정에서 곧바로 나온다)
#
# t7307 은 **창 순환으로 5/24 에서 중단**했고, 그 5 sim 은 *안 막힌 sim* 이라 표본이 편향됐다 —
# 완주 5 sim 은 **양팔 전부 `give` 를 실제로 호출**했으므로 이 레버의 표적 상태
# (*이름은 말했는데 끝내 안 건넴* · `x368` 51/82)는 표본에 **0건**이다. 잃어버린 8 sim 이
# 정확히 그 정보를 들고 있었다. ⇒ 판정을 못 한 것이 아니라 **표적을 못 본 것**이라 재런한다.
#
# 그리고 중단 전에 **마커와 다른 사이트에서 처치 효과**가 보였다(C527⒠):
#   `formalized_target=give_discoverable_user_tool` **ctl 5 → treat 17**(sim 당 0.83 → 2.43) ·
#   `[T2_RESOLVE] reason=action-required target=give` **3 → 9**. 노출 불균형 때문에 [D] 였고,
#   이 지표는 **라이브에서만** 잴 수 있어 격리로 대체 불가다. ⇒ 이 런이 사는 것은 그 확증이다.
#
# ## 무엇을 고쳤나 (엔진·A2·플래그는 **한 바이트도 안 건드린다**)
#
#   ⑴ `GO_MAX_STEPS=120` (신설 노브 · 기본값 200 = 종전 거동 불변).
#      근거: t7307 의 순환 sim 은 **turn 99 에서 7,800초**였고 상한 200 까지 두면 sim 당 4시간이다.
#      관측된 pass 는 **70·74 step**(t7307 완주) 이고 C521 의 풀링 pass 는 **≈50 msg** ⇒ 120 은
#      **관측된 모든 pass 위로 60% 여유**를 두고 낭비만 자른다.
#      ⚠**스택 변경으로 기록한다** — t7307 과 tail 이 다르다. ⓐ·ⓒ 는 턴-단위 관측이라 영향이
#        없고, 영향을 받을 수 있는 것은 ⓓ(pass)뿐인데 ⓓ 는 애초에 검정력이 없다.
#   ⑵ `GO_CONCURRENCY=6`(종전 4). 12 sim/팔을 2 파도로 — t7307 은 3 파도였고 파도마다 가장
#      느린 sim 을 기다렸다.
#   ⑶ 태그 신규(`..._20260818c`) — [[30]] 같은 태그 재사용 = 덮어쓰기 사고.
#
# ⚠**안 고친 것**: 창 순환 자체. 원인은 `_resolve_cap_ok` 의 *진행* 판정이 **새 도구를 하나라도
#   부르면 리셋**돼서, 모델이 새 read 를 계속 하는 한 상한이 무한히 되돌아간다는 것이다
#   (`stop=resolve_cap` 99회 = 상한은 살아 있고 계속 리셋됐다). 이건 설계 문제라 유료 런 직전에
#   손댈 자리가 아니다 — **무료 프로브로 따로 진단**하고, 그전까지는 ⑴로 손해만 자른다.
#
# ## ★판정 (사전 고정 · 결과보다 먼저 · t7307 헤더 + C527 이 배운 것)
#
#   ⓐ**1차 종점 = 술어 발화율**(pass 아님·C525⒠): treat 에서
#       `[T2_HANDOFF] named-but-not-given` 발화 sim / **손님-측 이름을 발화한 sim**.
#       ★**분모는 마커에서 뽑지 않는다**(t7303 ⓑ 순환 재발 방지) — `x371` 이 하듯 **궤적
#       어시스턴트 본문**에서 독립 검출하고, 이름 집합은 env `__discoverable__` 4종에서만 온다.
#       **≥50% → 닿는다** · <50% → 사각지대. ctl 은 **0** 이어야 한다(아니면 배선 오염).
#   ⓐ'**표적 모집단**(C527⒞ 가 못 잰 것): `give` 를 **끝내 안 부른** sim 수 / 전체 sim.
#       0 이면 이 가족에 표적이 없다는 뜻이고 **레버의 근거(`x368` 62%)가 이 가족엔 안 선다**.
#   ⓑ**W4 발화**: `pending_user` 에 `call_discoverable_user_tool` — **양팔 전 sim**(t7307 12/13).
#   ⓒ**절차 종점**(C527⒠ 확증·**이 런이 사는 것**): sim 당
#       `formalized_target=give_discoverable_user_tool` 횟수 · `reason=action-required target=give`
#       횟수 · `give` 호출 sim 수 · 손님(requestor=user) 호출 sim 수.
#       사전 바 = **treat 의 sim 당 give-지목이 ctl 의 2배 이상이고 노출(sim 수·turn 수)로
#       정규화한 뒤에도 유지**. ⚠n=12/팔이라 **유의성 주장 금지** — 방향 확증까지다.
#   ⓓ**2차 = pass**(`reward` 만·C486). ⚠**검정력 없음** — 승격/폐기에 쓰지 않는다.
#       짝지은 쌍(같은 task·같은 seed)만 본다(t7307 은 짝이 **1개**뿐이라 무정보였다).
#   ⓔ**부작용**: 지연 · CWE · 크래시 · **창 순환**(양팔 공통이던 것이 그대로인지 · `resign` 창 ·
#       동일지문 접힘 · `stop=resolve_cap`). ⑴이 이 축을 얼마나 잘랐는지도 여기서 읽는다.
#
# 편성: 019 가족 6 태스크 × nt=2 × 2팔 = **24 sim**. 구성 핀 = **t7307 축자 그대로**.

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
  echo "[t7308] REFUSING: 엔진 경로 커밋 안 된 변경 $DIRTY 개." >&2; exit 1
fi

for t in test_no_prose_regex.py test_sub_requirement.py test_docs_at_write.py \
         test_proceed_docbody.py test_cp2_clobber.py test_no_unbound_a2.py \
         test_deliver_precommit.py test_material_reserve.py test_material_bypass.py \
         test_probe_canonical.py test_log_join.py test_now_selfcall.py \
         test_no_undefined_names.py test_decision_carry.py test_subcall_return_type.py \
         test_a2_three_layer.py test_operator_find.py test_route_trace.py \
         test_group_parse.py test_verdict_carry.py test_pending_discovered.py \
         test_probe_scoring.py test_quote_in.py test_elig_handoff.py test_args_equal.py \
         test_flag_registry.py; do
  [ -f "$t" ] || continue
  PYTHONPATH=/home/woori/scratch/tau2-bench/src /home/woori/venvs/seka_env/bin/python "$t" \
    >/dev/null 2>&1 || { echo "[t7308] REFUSING: $t FAIL" >&2; exit 1; }
done
echo "[t7308] VERIFY OK"

if pgrep -f "[t]2_gap\.py" >/dev/null || pgrep -f "[x]3[0-9][0-9]_.*\.py" >/dev/null; then
  echo "[t7308] REFUSING: 무료 프로브 실행 중(양 포트 필요)" >&2; exit 1
fi

PIN="T2_ACTION_SUB=1 T2_KEEP_DENY_BODY=1 T2_CALL_FORM=1 T2_ARG_EMPTY=1 T2_SEARCH_AGENT=1 \
T2_DECIDE_ANY=1 T2_WRITE_ARG_ENUM=1 T2_DECIDE_BEFORE_WRITE=1 T2_DECISION_CARRY=1 \
T2_DISCOVERY_STEP2=1 T2_ARG_AXIS=1 T2_WRITE_SUB=3 T2_ACTION_INDEX=1 T2_NOW_SELFCALL=1 \
T2_SEARCH_ON_PROCEED=1 T2_ACT_DEMAND=0 T2_DELIVER_PRECOMMIT=0 T2_PROCEED_DOCBODY=0 \
T2_DOCS_AT_WRITE=0 T2_SUB_REQUIREMENT=0 T2_VERDICT_CARRY=0 T2_ELIG_LINE=0 \
T2_PENDING_DISCOVERED=0"
TASKS='task_019,task_020,task_022,task_027,task_028,task_029'

launch () {
  NAME="$1"; PORT="$2"; HP="$3"
  TAG="bank_t7308_${NAME}_20260818c"
  if [ -e "$LOG/${TAG}.log" ]; then echo "[t7308] SKIP: ${TAG}.log 존재" >&2; return 0; fi
  if [ -e "$SIMS/${TAG}" ]; then echo "[t7308] REFUSING: $SIMS/${TAG} 잔존" >&2; return 1; fi
  echo "{\"tag\":\"$TAG\",\"scaffold_sha\":\"$SHA\",\"port\":$PORT,\"tasks\":\"$TASKS\",\"nt\":2,\"handoff\":\"$HP\",\"max_steps\":120,\"concurrency\":6,\"pin\":\"$PIN\",\"why\":\"t7307 rerun after a window loop truncated it at 5/24; primary endpoint is predicate firing rate (C525), confirmatory endpoint is the give-target shift (C527e)\"}" \
    | tee "$LOG/${TAG}.meta.json"
  setsid bash -c "cd '$REPO/scripts/distill/tau2' && source ./go_stack.sh >/dev/null 2>&1 && \
    export $PIN && export T2_HANDOFF_PREDICATE=$HP && \
    export GO_MAX_STEPS=120 GO_CONCURRENCY=6 && \
    t2_launch $TAG $PORT '$TASKS' 2" \
    </dev/null >"$LOG/${TAG}.log" 2>&1 &
  echo "[t7308] $NAME(handoff=$HP) → PID=$! port=$PORT"
}

launch ctl   8140 0
launch treat 8141 1
echo "[t7308] 기동 완료 · sha=$SHA · 팔당 12 sim · max_steps=120 · 1차 종점 = 술어 발화율"
