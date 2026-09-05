#!/usr/bin/env bash
# t7393 — 실패 47 전수 treat 팔 (2026-09-05) · 레인 하나를 돈다
#
# 사용: run_t7393_treat.sh <레인이름> <AGENT_HOST> <PORT> <task_a,task_b,...>
#
# ── 왜 이 런인가 (사용자 지시 축자) ────────────────────────────────────────
#   *"wrongarg로 실패한 태스크들에게 효과있는지 실험하면 안되나?"*
#   된다. 그리고 **오프라인 분류를 못 하기 때문에 오히려 이 길뿐이다** —
#   dbdiff 리플레이가 회수분 94건 중 **87건에서 REPLAY-FAIL**
#   ("Unknown tool 'KB_search_bm25'" · 리플레이 env 가 retrieval 없이 지어진다).
#   ⇒ 「어느 태스크가 WRONGARG 실패인가」를 재현할 수 없다. 표에서 로스터를 고르면
#     선택 편향이므로 **실패 전수**를 돌리고 reward 에게 묻는다([[69]] 채점단위).
#   ★A/B 런은 그 깨진 계기가 필요 없다 — reward 는 하버스가 낸다.
#
# ── 이 팔이 켜는 것 (전부 오늘 sha 에 처음 배선된 것) ──────────────────────
#   ① eplan.write_tools 3 -> 36   (_wc 사정권 · 라이브 확인됨: 050 이 _wc 를 통과했고 1.0 유지)
#   ② T2_REGEN_WRITE_GATES=1      D14 재생성 산출의 쓰기게이트 재진입
#   ③ T2_REGEN_KEEP_MUTATING=1    D11ⓐ
#   ④ T2_DUP_WRITE=1 (선언분 한정) ⑤ T2_SIBLING_PAREN=strip  ⑥ T2_SCOPE_AT_DISPATCH_ONLY=1
#   팔 = viewmax2 (캠페인 본 팔). run_ours_task.sh 가 프로필·preflight·:127-128 을 실어 준다.
#
# ── 판정 ([[85]] 를 만족시키는 최소 설계) ──────────────────────────────────
#   1단계(이 런) = **뒤집힌 태스크를 찾는다**. 0->1 목록.
#   2단계(뒤에) = 그 뒤집힌 것만 **같은 sha · 같은 seed(nt=1 -> 626729) · 5종 OFF** 로 대조.
#     ⇒ 짝 비교의 내용은 그대로면서 92 -> ~55 sim.
#   ⛔이 런 하나로 "레버가 샀다"고 말하지 마라. flip 바닥이 18.8~25% 다([[85]]).
#     이 런은 **후보를 뽑는 것**이고 2단계가 인과를 판정한다.
#
# ⛔`set -u` 금지 · pkill -f 금지([[30]]) · 도는 런의 조건 변경 금지([[54]])
LANE="$1"; AHOST="$2"; PORT="$3"; TASKS="$4"
[ -z "$TASKS" ] && { echo "사용: $0 <레인> <AGENT_HOST> <PORT> <tasks>"; exit 1; }

REPO=/home/woori/workspace_common/boltzmann-attention-pi
LOG=/home/woori/scratch/logs
SIMS=/home/woori/scratch/tau2-bench/data/simulations
STAMP=${STAMP:-$(date +%Y%m%d_%H%M)}
TAG="bank_t7393_${LANE}_${STAMP}"

cd "$REPO/scripts/distill/tau2" || exit 1
echo "[t7393/$LANE $(date +%H:%M:%S)] sha=$(cd $REPO && git rev-parse --short HEAD) host=$AHOST port=$PORT"
echo "[t7393/$LANE] tasks=$TASKS"

# ★[[30]] 함정 — 포트만으로 엔진을 식별하지 마라. 발사 전 /v1/models id 대조.
GOT=$(curl -s -m 10 "http://$AHOST:$PORT/v1/models" | grep -oE '"id":"[^"]+"' | head -1 | cut -d'"' -f4)
echo "[t7393/$LANE] 서빙 모델 = $GOT"
case "$GOT" in
  *Qwen3.8*) : ;;
  *) echo "[t7393/$LANE] 중단 — Q3.8 이 아니다([[79]] 프레임 위반): $GOT"; exit 1 ;;
esac

rm -rf "$SIMS/$TAG"
T2_AGENT_HOST="$AHOST" bash ./run_ours_task.sh --arm viewmax2 --concurrency 1 --trials 1 "$TAG" "$PORT" "$TASKS" > "$LOG/$TAG.log" 2>&1
echo "[t7393/$LANE $(date +%H:%M:%S)] 종료 exit=$?"

echo "[t7393/$LANE] === 새 레버 발화 ([[81]]) ==="
for m in "T2_REGEN_WGATE" "operator-scope" "T2_SIBLING_PAREN" "T2_DUP_WRITE" "T2_SPEC_AT_WRITE" "T2_RULE_AT_WRITE" "T2_WRITE_ARG_TYPE" "Traceback"; do
  N=$(grep -ac "$m" "$LOG/$TAG.log" 2>/dev/null); N=${N:-0}
  printf "  %-22s %s\n" "$m" "$N"
done

echo "[t7393/$LANE] === reward ==="
TAG="$TAG" SIMS="$SIMS" /home/woori/venvs/seka_env/bin/python - <<'PY'
import json,os
p=os.path.join(os.environ["SIMS"],os.environ["TAG"],"results.json")
try: r=json.load(open(p))
except Exception as e: print("  회수 실패 %r"%(e,)); raise SystemExit
sims=r.get("simulations") or []
flip=[]
for s in sims:
    rw=(s.get("reward_info") or {}).get("reward") or 0.0
    print("  %-10s %.1f  term=%s" % (s.get("task_id"), rw, s.get("termination_reason")))
    if rw >= 1.0: flip.append(s.get("task_id"))
print()
print("  sim %d · 1.0 도달 %d : %s" % (len(sims), len(flip), ",".join(sorted(flip)) or "(없음)"))
print("  ⚠전부 캠페인에서 0.0 이던 태스크다. 이 목록이 2단계 대조의 로스터다([[85]]).")
PY
