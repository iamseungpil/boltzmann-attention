#!/usr/bin/env bash
# t7392 — sha 2f511ece 의 레버 5종 + write 사정권 확대(3→36) 스모크 (2026-09-05)
#
# ── 무엇을 재나 (성적이 아니라 **자리가 서는가**) ─────────────────────────
#   이 sha 는 한 번도 라이브를 돈 적이 없는 변경 여섯을 담고 있다:
#     ① eplan.write_tools 3 → 36        `_wrset` 이 dispute 3종 밖을 처음 본다
#     ② T2_REGEN_WRITE_GATES=1  (D14)   재생성 산출이 쓰기 게이트에 재진입
#     ③ T2_REGEN_KEEP_MUTATING=1 (D11ⓐ) 재생성이 잃은 env-변이 호출을 되붙임
#     ④ T2_DUP_WRITE 0→1                정본↔런처 불일치 해소 + 술어를 선언분으로 좁힘
#     ⑤ T2_SIBLING_PAREN log→strip
#     ⑥ T2_SCOPE_AT_DISPATCH_ONLY=1
#   ★①②③⑥ 은 **계기가 처음 도는 것**이라 [[81]] 대로 «첫 런에서 발화 0 이면 경로가 틀린 것».
#
# ── ⛔이 런의 존재 이유 = T2_DUP_WRITE 의 부정통제 ────────────────────────
#   go_stack.sh:725 축자: *"발사 전 스모크에 `task_050` 을 넣어라 — 회수분에서 미선언 deny 를
#   맞고도 1.0 인 유일한 banking sim 이고, 막힌 도구가 051 이 사려는 것과 **동일**하다.
#   거기서 1.0 -> 0.0 이면 이 레버의 순효과는 0 이거나 음이다([[70]] 절충 공개)."*
#   ⇒ 050 은 성적이 아니라 **판정 장치**다. 이 스모크 없이 97 을 태우면 그 절충을 못 잰다.
#
# ── 로스터 (한 태스크 = 한 의무) ──────────────────────────────────────────
#   050  ④ 부정통제 — 1.0 유지해야 한다
#   051  ④ 표적     — gold 재제출 4발을 지웠던 그 자리
#   065  ⑤ 표적     — 채점된 sim 의 action_match=False 가 그 괄호 칸 하나
#   015  ③ 표적     — pre-give 재생성이 give 호출을 떨구던 자리
#   078  ① 표적     — close 계열은 구 _wrset(도구 3종) **밖**이었다
#
# ── 판정 ─────────────────────────────────────────────────────────────────
#   PASS = Traceback 0 ∧ 050 이 1.0 ∧ ①②⑥ 계기가 각각 1회 이상 발화
#   ⚠pass 수로 레버를 주장하지 마라 — nt=1 이고 [[85]] flip 바닥이 18.8~25% 다.
#     이 런이 답하는 것은 **«걸리는가»** 뿐이고, 얼마나 사는가는 97 짝런이 답한다.
#
# ⛔`set -u` 금지 · 줄 이음 금지 · pkill -f 금지([[30]])
REPO=/home/woori/workspace_common/boltzmann-attention-pi
LOG=/home/woori/scratch/logs
SIMS=/home/woori/scratch/tau2-bench/data/simulations
STAMP=$(date +%Y%m%d_%H%M)
TASKS_A="task_050,task_051,task_065"
TASKS_B="task_015,task_078"

cd "$REPO/scripts/distill/tau2" || exit 1
echo "[t7392 $(date +%H:%M:%S)] sha=$(cd $REPO && git rev-parse --short HEAD)"

echo "[t7392] === 발사 전 배터리 ==="
BAD=0
for t in test_lever_wiring test_proc_regen_recheck test_sibling_paren test_dup_write \
         test_operator_find_executed test_a2_three_layer test_banking_gate \
         test_action_attach test_terse_schema test_arg_policy_join; do
  PYTHONPATH=. PYTHONIOENCODING=utf-8 /home/woori/venvs/seka_env/bin/python $t.py >/dev/null 2>&1
  rc=$?; echo "  $t exit=$rc"; [ $rc -ne 0 ] && BAD=1
done
[ $BAD -ne 0 ] && { echo "[t7392] ⛔배터리 붉음 — 발사하지 않는다"; exit 1; }

# ── ★경로 정정 (2026-09-05) — 첫 시도는 `t2_launch` 로 쐈고 즉사했다 ─────────
#   `t2_launch` 는 `t2_run_gated.py` 를 직접 부른다 ⇒ 모델 프로필을 안 실어서
#   `T2_MAX_MODEL_LEN` 이 비고, 그 자동탐지 블록이 `sys` 미임포트로 NameError 를 냈다
#   (`b5d049ed` 가 08-31 에 넣은 코드 · t7389 는 08-29 라 그 전이었다).
#   더 큰 것은 그 함수가 `--agent_model Qwen/Qwen2.5-32B...` 를 박아 두고 있었다는 것 —
#   서버는 Q3.8 을 서빙한다([[84]]/[[30]]). 둘 다 이 sha 에서 고쳤다.
#   ⇒ 이 스모크는 **캠페인이 실제로 탄 경로**로 간다([[54]] 비교 규격):
#     run_ours_task.sh = 프로필 자동선택 + x704 표면 preflight + :127-128 환경줄.
#   ⚠팔은 `viewmax2` — 캠페인 본 팔이고, 이 sha 에서 그 팔의 SIBLING_PAREN 을 strip 으로
#     올렸다(log 로 되돌리고 있었다·[[81]]). 부정통제 팔 셋은 log 그대로다.

PIDS=""
for A in 0 1; do
  NAME=lane1; ARM_PORT=8141; T="$TASKS_A"
  [ "$A" = "1" ] && { NAME=lane2; ARM_PORT=8143; T="$TASKS_B"; }
  TAG="bank_t7392_${NAME}_${STAMP}"
  rm -rf "$SIMS/$TAG"
  (
    echo "[$NAME $(date +%H:%M:%S)] $T nt=1 port=$ARM_PORT arm=viewmax2"
    bash ./run_ours_task.sh --arm viewmax2 --concurrency 1 --trials 1          "$TAG" "$ARM_PORT" "$T" 2>&1 | tee "$LOG/$TAG.log"
    echo "[$NAME $(date +%H:%M:%S)] done"
  ) > "$LOG/${TAG}_driver.log" 2>&1 &
  PIDS="$PIDS $!"
  echo "[t7392] $NAME 발사 pid=$! port=$ARM_PORT tasks=$T"
done
echo "[t7392] 두 레인 대기:$PIDS"
for P in $PIDS; do wait $P; echo "[t7392] pid $P 종료 exit=$?"; done

echo "[t7392] ═══ 계기 발화 ([[81]] — 0 이면 경로가 틀린 것) ═══"
CAT="$LOG/bank_t7392_lane1_${STAMP}.log $LOG/bank_t7392_lane2_${STAMP}.log"
for m in "T2_REGEN_WGATE" "operator-scope" "T2_SIBLING_PAREN" "T2_SPEC_AT_WRITE" \
         "T2_RULE_AT_WRITE" "T2_WRITE_ARG_TYPE" "T2_DUP_WRITE" "T2_REGEN_KEEP" "Traceback"; do
  N=$(cat $CAT 2>/dev/null | grep -ac "$m"); N=${N:-0}
  printf "  %-22s %s\n" "$m" "$N"
done
echo "[t7392] ═══ 태스크별 reward ═══"
STAMP="$STAMP" SIMS="$SIMS" /home/woori/venvs/seka_env/bin/python - <<'PY'
import json,glob,os,collections
st=os.environ["STAMP"]; sims=os.environ["SIMS"]
r=collections.defaultdict(list)
for f in glob.glob(os.path.join(sims,"bank_t7392_*_%s"%st,"*.json")):
    try: d=json.load(open(f))
    except Exception: continue
    for s in (d.get("simulations") or []):
        r[s.get("task_id")].append((s.get("reward_info") or {}).get("reward"))
for k in sorted(r):
    print("  %-10s %s" % (k, r[k]))
print()
z=r.get("task_050") or []
if z and all((x or 0) >= 1.0 for x in z):
    print("  ✅050 = %s 유지 — T2_DUP_WRITE 의 부정통제 통과" % z)
elif z:
    print("  ⛔050 = %s — go_stack.sh:725 축자대로 이 레버의 순효과는 0 이거나 음이다([[70]])" % z)
else:
    print("  ⚠050 sim 회수 실패 — 판정 불가")
PY
echo "[t7392 $(date +%H:%M:%S)] 끝"
