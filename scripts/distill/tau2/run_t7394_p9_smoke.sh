#!/usr/bin/env bash
# t7394 — P9 수리 스모크 (2026-09-06) · 패스2 발사 전 관문
#
# ── 무엇을 재나 ────────────────────────────────────────────────────────────
#   수리: `tool_signatures.give_discoverable_user_tool`
#         ["discoverable_tool_name"] → ["discoverable_tool_name", "arguments"]
#   근거: 서버 `tools.py:533-534` 가 `arguments: str = "{}"` 를 받고, 본체(:551-570)가 그것을
#         `json.loads` 한 뒤 **실제 사용자-도구 시그니처와 대조**해 `Error: Unexpected parameter:`
#         를 스스로 낸다 — 우리보다 정확하다.
#   ⚠선언의 출처(정책 인용)는 **실재한다**(`prompts/components/additional_instructions.md` 축자
#     *"Use the `give_discoverable_user_tool(discoverable_tool_name)` function"* ×2).
#     틀린 것은 관측이 아니라 **추론**이다 — 그 문장은 «설명 말고 실제로 불러라» 이지 인자 목록의
#     전수 규정이 아니다(task_005 센티널과 같은 형상).
#
# ── 로스터 (한 태스크 = 한 의무) ──────────────────────────────────────────
#   ★표적 2 — P9 가 실제로 죽인 것
#     task_015  deny 6회 · 통과 0. base 팔은 그 «금지된» 호출로 통과(1.0), 우리 팔은 deny 3회로
#               grant 를 msg35 까지 밀어 user 가 포기 → MISSING referrals.data → 0.0
#     task_022  deny 3회(로그 699·866·912) → `pending_user` 를 넘길 방법이 없어
#               `intent_operator_formalize` 21회 재호출 · **130 스텝 · 2시간 46분** 폭주(손으로 kill)
#   ★대조 4 — P9 반려를 맞고도 **전 sim 통과**하던 것 (회귀가 나면 여기서 보인다)
#     task_057 (2/2) · task_028 (2/2) · task_020 (2/2) · task_017 (1/1)
#   ⛔제외: task_031 — P9 무관이 확정됐다. 통과 판이 SIGNATURE 를 **3배 더** 맞고도 통과했고,
#     오늘 갈림은 `GB1_VERIFY_BEFORE_ACCOUNT_ACCESS` 다. 넣으면 신호가 흐려진다.
#
# ── 판정 ─────────────────────────────────────────────────────────────────
#   PASS = ⑴ `[SIGNATURE] give_discoverable_user_tool` 발화 **0**  (수리가 닿았다는 직접 증거)
#          ⑵ 대조 4 가 **전부 1.0 유지**                            ([[70]] 파는 것 0)
#          ⑶ task_022 가 **폭주하지 않는다**(스텝 < 130 · 90분 내 종료)
#   ⚠표적 2 의 reward 는 **2차**다. n=1 이고 [[85]] flip 바닥이 18.8~25% 라 0→1 하나는 증거가 아니다.
#     이 스모크가 확정하는 것은 «오반려가 사라졌나 · 통과하던 것을 깨지 않았나» 다.
#
# ⛔`set -u` 금지 · pkill -f 금지([[30]]) · 줄 이음 금지
REPO=/home/woori/workspace_common/boltzmann-attention-pi
LOG=/home/woori/scratch/logs
SIMS=/home/woori/scratch/tau2-bench/data/simulations
STAMP=$(date +%Y%m%d_%H%M)

cd "$REPO/scripts/distill/tau2" || exit 1
echo "[t7394 $(date '+%m-%d %H:%M')] sha=$(cd $REPO && git rev-parse --short HEAD)"

# ★수리가 실제로 들어왔는지 발사 전 확인 — 안 들어왔으면 스모크가 무의미하다([[81]])
GOT=$(/home/woori/iso_tau3/venv/bin/python -c "
import json
d=json.load(open('a2/banking_knowledge.gate.json'))
print(','.join((d.get('tool_signatures') or {}).get('give_discoverable_user_tool') or []))
")
echo "[t7394] 선언 = $GOT"
case "$GOT" in
  *arguments*) : ;;
  *) echo "[t7394] ⛔수리가 안 들어왔다 — 중단"; exit 1 ;;
esac

echo "[t7394] === 발사 전 배터리 ==="
BAD=0
for t in test_p9_signature test_a2_three_layer test_banking_gate test_lever_wiring; do
  PYTHONPATH=. PYTHONIOENCODING=utf-8 /home/woori/venvs/seka_env/bin/python $t.py >/dev/null 2>&1
  rc=$?; echo "  $t exit=$rc"; [ $rc -ne 0 ] && BAD=1
done
[ $BAD -ne 0 ] && { echo "[t7394] ⛔배터리 붉음 — 발사하지 않는다"; exit 1; }

PIDS=""
for A in 1 2 3; do
  case $A in
    1) NAME=lane1; AHOST=localhost;    PORT=8141; T1=task_015; T2=task_057 ;;
    2) NAME=lane2; AHOST=localhost;    PORT=8143; T1=task_022; T2=task_028 ;;
    3) NAME=lane3; AHOST=10.10.10.151; PORT=8141; T1=task_020; T2=task_017 ;;
  esac
  (
    for T in $T1 $T2; do
      TAG="bank_t7394_${T}_${STAMP}"
      rm -rf "$SIMS/$TAG"
      echo "[$NAME $(date '+%H:%M')] → $T"
      T2_AGENT_HOST="$AHOST" bash ./run_ours_task.sh --arm viewmax2 --concurrency 1 --trials 1 \
          "$TAG" "$PORT" "$T" > "$LOG/${TAG}_driver.log" 2>&1
      echo "[$NAME $(date '+%H:%M')] ← $T rc=$?"
    done
  ) > "$LOG/t7394_${NAME}_${STAMP}.log" 2>&1 &
  PIDS="$PIDS $!"
  echo "[t7394] $NAME 발사 pid=$! $AHOST:$PORT ($T1, $T2)"
  sleep 4
done
echo "[t7394] 세 레인 대기:$PIDS"
for P in $PIDS; do wait $P; done

echo "[t7394] ═══ 판정 ═══"
STAMP="$STAMP" SIMS="$SIMS" LOG="$LOG" /home/woori/iso_tau3/venv/bin/python - <<'PY'
import json, os, glob, re
st, sims, log = os.environ["STAMP"], os.environ["SIMS"], os.environ["LOG"]
TARGET = {"task_015", "task_022"}
CTRL = {"task_057", "task_028", "task_020", "task_017"}
rw, steps = {}, {}
for t in sorted(TARGET | CTRL):
    d = os.path.join(sims, "bank_t7394_%s_%s" % (t, st))
    try:
        r = json.load(open(os.path.join(d, "results.json")))
        ss = r.get("simulations") or []
        rw[t] = (ss[0].get("reward_info") or {}).get("reward") if ss else None
    except Exception:
        rw[t] = None
    f = os.path.join(log, "bank_t7394_%s_%s.log" % (t, st))
    try:
        b = open(f, "rb").read()
    except Exception:
        b = b""
    steps[t] = len(re.findall(rb"\[T2_GEN_TRACE\]", b))
sig = 0
for f in glob.glob(os.path.join(log, "bank_t7394_*_%s*.log" % st)):
    try:
        sig += len(re.findall(rb"Error: \[SIGNATURE\] `give_discoverable_user_tool`", open(f, "rb").read()))
    except Exception:
        pass
print("  %-11s %6s %6s  %s" % ("task", "reward", "steps", "역할"))
for t in sorted(TARGET | CTRL):
    print("  %-11s %6s %6d  %s" % (t, rw[t], steps[t], "표적" if t in TARGET else "대조"))
print()
g1 = (sig == 0)
g2 = all((rw.get(t) or 0) >= 1 for t in CTRL)
g3 = (steps.get("task_022", 0) < 130) and rw.get("task_022") is not None
print("  ⑴ SIGNATURE(give_…) 발화 = %d  →  %s" % (sig, "PASS" if g1 else "⛔FAIL — 수리가 안 닿았다"))
print("  ⑵ 대조 4 전부 1.0        →  %s" % ("PASS" if g2 else "⛔FAIL — 회귀"))
print("  ⑶ 022 폭주 없음(<130스텝) →  %s (steps=%d)" % ("PASS" if g3 else "⛔FAIL", steps.get("task_022", 0)))
print()
print("  표적 reward: 015=%s · 022=%s  ⚠n=1 이라 2차 지표다([[85]] flip 바닥 18.8~25%%)"
      % (rw.get("task_015"), rw.get("task_022")))
print("\n  ★게이트: %s" % ("PASS — 패스2 발사 가능" if (g1 and g2 and g3) else "⛔FAIL — 패스2 보류"))
PY
echo "[t7394 $(date '+%m-%d %H:%M')] 끝"
