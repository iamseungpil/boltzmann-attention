#!/usr/bin/env bash
# t7360 스모크 — `_a2_of` 배선 수리(x549) 후 **라이브 발화 + 회귀**를 함께 본다.
#
# 왜 이 판인가 (2026-08-26)
#   t7359 스모크가 첫 판에서 `[T2_DUP_WRITE] deny tool=unlock_discoverable_agent_tool` 을 냈다.
#   원인은 술어가 아니라 **재료 미전달**이었다 — `unified()` 는 에이전트에 설치되는데
#   `_a2_of(obj)` 가 `obj.environment`(오케스트레이터에만 있는 속성)만 봐서 unified 안 여섯
#   자리가 전부 `None` 을 받았다. 그래서 `unlock_`·`give_`·`call_` 이 실효 write 로 뒤집혔다.
#   측정 = `x549_a2_binding_leak.py` · 기록 = `x549_a2_binding_leak_2026_08_26.json`.
#
# ⚠이 수리는 **양날**이다([[70]] — ± 를 공개한다)
#   `T2_WRITE_PROV`·`T2_CLAIM_PROV` 는 상시 ON(go_stack:110·162)이고 둘 다 같은 술어로 갈린다.
#   x549 실측(최근 런 12개 23 sim): `_any_effective_write` 참 **100% → 34.8%**, 전-sim 판정
#   뒤집힘 **15/23**. 뒤집히는 15 중 **098×2·100 은 지금 reward=1.0** 이다.
#   ⇒ 그래서 이 스모크는 표적 넷에 **회귀 대조 098·100 을 더해** 여섯을 돈다.
#
# 무엇을 판정하나 (하나라도 어긋나면 유료 런 금지)
#   ① `T2_DUP_WRITE` 가 **절차적 래퍼에 발화하지 않는다** (t7359 의 결함이 사라졌나)
#   ② `T2_SPEC_DIST`·`T2_SPEC_AT_WRITE`·`T2_RULE_AT_WRITE` 가 **0 이 아니다**
#      (선언 배달 수리가 살아 있나 — 예측: 085×2·040 에서 명세+규칙이 함께 나간다)
#   ③ `T2_WRITEPROV`·`T2_CLAIMPROV` 가 **이제 발화한다**
#      (`LEVER_ROSTER_CANONICAL_2026_08_19.md:248` 의 *"마크 12,181 : 실발화 3"* 이 이 결함의
#       하류였는지 — 발화가 0 이면 원인이 다른 데 있다는 뜻이므로 그것도 정보다)
#   ④ **098·100 이 여전히 1.0** 이다 (수리가 회귀를 사지 않았나)
#   ⑤ Traceback 0
#
# ⚠t7359 와 다른 노브 하나: `GO_CONCURRENCY=2`(t7359 는 1). 태스크가 넷→여섯이라 벽시계를
#   맞추려는 것이고, sim 은 서로 독립이라 per-sim 판정에 교락이 아니다. 다만 서버 경합이
#   생기면 타임아웃으로 보일 수 있으니 **종료사유를 함께 읽어라**(§7-5 는 동시성 근거가
#   여전히 미실측이라고 적어 두었다).
# ⛔`set -u` 를 쓰지 마라 — `go_stack.sh` 의 `t2_require_key` 는 `[ -n "$OPENAI_API_KEY" ]` 로
#   **미설정을 스스로 처리**하는데, `set -u` 아래서는 그 표현식 자체가 죽는다. 2026-08-26 초판이
#   그래서 `t2_launch` 를 한 줄도 안 돌리고 8초 만에 "SMOKE DONE" 을 찍었다(마커 전부 0 =
#   *"배선이 죽었다"* 로 오독되기 딱 좋은 모양). 그 무음 실패를 아래 §발사 확인이 이제 잡는다.
REPO=/home/woori/workspace_common/boltzmann-attention-pi
LOG=/home/woori/scratch/logs
SIMS=/home/woori/scratch/tau2-bench/data/simulations
TAG=bank_t7360_smoke_20260826
cd "$REPO/scripts/distill/tau2"
source ./go_stack.sh >/dev/null 2>&1

# t7359 와 **같은 레버 집합** (수리 효과만 남기기 위해 노브를 더 바꾸지 않는다)
export T2_ACTION_SUB=1 T2_KEEP_DENY_BODY=1 T2_CALL_FORM=1 T2_ARG_EMPTY=1 T2_SEARCH_AGENT=1 T2_DECIDE_ANY=1 T2_WRITE_ARG_ENUM=1 T2_DECIDE_BEFORE_WRITE=1 T2_DECISION_CARRY=1 T2_DISCOVERY_STEP2=1 T2_ARG_AXIS=1 T2_WRITE_SUB=3 T2_ACTION_INDEX=1 T2_NOW_SELFCALL=1 T2_SEARCH_ON_PROCEED=1
export T2_ARG_DOC_SUB=1 T2_VALUE_FORMULA=full T2_SG_DOCS=1 T2_SG_PROMPT_V2=1 T2_SPEC_AT_WRITE=1 T2_WRITE_ARG_TYPE=1 T2_RULE_AT_WRITE=1 T2_WRITE_ARG_ENUM_CAP=8 T2_WRITE_ARG_FAB=1 T2_SG_RECORD_ORDER=1 T2_SPEC_ARG_FACTS=1 T2_GIVE_REQUIRED=1 T2_CALL_FORM_FIX=1
export T2_DUP_WRITE=1
export GO_MAX_STEPS=150 GO_CONCURRENCY=2

# ★동결은 런처가 건다(가드 §코드 동결·C423⒞) — 런 도중 엔진이 바뀌면 그 런은 귀속 불가다.
python "$REPO/reports/facet_rft_2026/freeze.py" --on --tag "$TAG" \
  --reason "t7360 smoke: _a2_of wiring repair (x549) - liveness + 098/100 regression"

echo "[t7360 $(date +%H:%M:%S)] smoke start · DUP_WRITE=$T2_DUP_WRITE · CONCURRENCY=$GO_CONCURRENCY"
t2_launch $TAG 8140 "task_074,task_050,task_085,task_040,task_098,task_100" 1 2>&1 | tee $LOG/$TAG.log

python "$REPO/reports/facet_rft_2026/freeze.py" --off

# ── ★발사 확인 — 마커 0 을 "배선이 죽었다" 로 오독하지 않기 위한 전제 ──
#   sim 이 하나도 안 돌았으면 마커가 0 인 것은 **당연**하고 판정 재료가 아니다.
_RES="$SIMS/$TAG/results.json"
_LINES=$(wc -l < "$LOG/$TAG.log" 2>/dev/null || echo 0)
if [ ! -s "$_RES" ] || [ "$_LINES" -lt 50 ]; then
  echo "⛔[t7360] 발사 실패 — sim 이 안 돌았다 (results=$( [ -s "$_RES" ] && echo 있음 || echo 없음) · 로그 $_LINES 줄)"
  echo "   아래 마커는 **판정 재료가 아니다**. 원인부터 봐라:"
  grep -aE "REFUSING|command not found|해제된 변수|unbound|Traceback" "$LOG/$TAG.log" | head -5
  exit 1
fi

# ── 회수 (★[[30]]: gzip 만으로는 영속이 아니다 — tracked 확인까지가 절차) ──
cd "$REPO" && mkdir -p reports/facet_rft_2026/sim_results
gzip -c "$SIMS/$TAG/results.json" > reports/facet_rft_2026/sim_results/$TAG.results.json.gz 2>/dev/null || true
gzip -c $LOG/$TAG.log > reports/facet_rft_2026/sim_results/$TAG.log.gz 2>/dev/null || true
for _S in fb trace; do _F=$LOG/${_S}_${TAG}.jsonl; [ -s "$_F" ] && gzip -c "$_F" > reports/facet_rft_2026/sim_results/${_S}_${TAG}.jsonl.gz; done
git add -f reports/facet_rft_2026/sim_results/${TAG}* reports/facet_rft_2026/sim_results/fb_${TAG}* reports/facet_rft_2026/sim_results/trace_${TAG}* 2>/dev/null || true
git -c user.name=ghlee -c user.email=beingrelative@gmail.com commit -q -m "t7360 smoke: post-x549 wiring repair" -- reports/facet_rft_2026/sim_results/ || true
git push -q origin facet-rft-2026 || echo "[t7360] push held"
echo "=== TRACKED ==="
git ls-files --error-unmatch reports/facet_rft_2026/sim_results/${TAG}.results.json.gz 2>&1 | tail -1

echo "=== MARKERS (0 이면 그 배선은 아직 죽어 있다) ==="
for m in T2_DUP_WRITE T2_SPEC_DIST T2_SPEC_AT_WRITE T2_RULE_AT_WRITE T2_SPEC_ARG_FACTS T2_WRITEPROV T2_CLAIMPROV Traceback; do
  printf "%-22s=" $m; grep -ac "\[$m\]" $LOG/$TAG.log
done
echo "=== ①DUP_WRITE 가 절차적 래퍼를 잡았나 (있으면 수리 실패) ==="
grep -a "\[T2_DUP_WRITE\] deny" $LOG/$TAG.log | grep -aE "unlock_|give_" || echo "  없음 ✓"
echo "=== ④회귀 대조 098·100 ==="
grep -aE "task_(098|100).*(reward|complete)" $LOG/$TAG.log | tail -8
echo "[t7360] SMOKE DONE"
