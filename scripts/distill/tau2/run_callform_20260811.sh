#!/bin/bash
# 유료 런 — **C419 `T2_CALL_FORM` + C420 `T2_ARG_EMPTY`** 를 표적 두 태스크에서 잰다.
#
# 왜 (C421: 하나의 진술로 묶인다 — *우리가 도구 이름을 댈 때는 그 타입의 호출 형식으로 댄다*)
#   C419: 우리 `[ORDER]` 가 발견형 도구를 **부를 수 없는 이름**으로 요구한다. 격리 인과
#         (x249·n=8·실패 궤적 2개·두 판): `A_LIVE` 8/32 ↔ **`B_CALLFORM` 32/32**,
#         출시본 문자열 그대로인 `B_ENGINE` **16/16**. 오답은 전부 접미사 이름 **직접 호출**.
#   C420: `log_verification` 이 `date_of_birth=""` 로 나갔는데 **아무 규칙도 안 본다**
#         (WAG 는 *"값 없음 = skip"*). x250: `A_LIVE` EMPTY 8/8 ↔ **`B_NAME`/`B_ENGINE` 8/8**,
#         이름 없는 거부 `C_GENERIC` **0/8**(부정 통제).
#   C421: 요구 채널 전수에서 **T2 결함 1406/1406 = 100%** — 099 는 그 1406 중 하나였다.
#         `[ORDER]` 채널이 1217/1309 이므로 이 처방이 **93%** 를 덮는다.
#
# 귀속 (직전 런에서 **플래그 두 개** 차이)
#   `bank_kb_20260811` = ACTION_SUB + UNLOCK_QUIET + KEEP_DENY_BODY → 2/6 (010 1/3 · 099 1/3)
#   이 런              = 위 + **CALL_FORM + ARG_EMPTY**
#   ⚠두 개를 함께 켜는 이유: 겨누는 자리가 **다른 태스크의 다른 칸**이라 서로의 표적을 가리지
#     않는다(099_2 ↔ 010_0). 그래도 총점 델타는 두 레버의 합이므로 **계기별로 따로 읽는다**.
#
# 사전 등록 (보기 전에 적는다)
#   P0 팔 오염   `[T2_CALL_FORM] named` · `[T2_ARG_EMPTY] deny` 마크가 **이 런에만** 있는가
#   P1 성적      태스크별 pass (기준 010 1/3 · 099 1/3 — nt=6 이므로 /6)
#   P2 표적 계기 ⑴099: `call_discoverable_agent_tool` 호출 수 · gold `099_2` 충족 시행 수
#                ⑵010: `log_verification` 의 빈 필수 인자 건수(기준 12 sim 중 2건)
#                **성적보다 이 수를 먼저** 읽는다(C415 와 같은 규율).
#   P3 넘김 발화 도구 이름+값이 한 메시지에 있는 턴(C412) · 손님 제출과의 대응 — **n 을 늘려**
#                C415⒠ 가 깬 대응을 다시 센다(충분조건인지 필요조건인지).
#   P4 Δspurious 게이트 거부 수 · gold 밖 쓰기 호출. **ARG_EMPTY 는 거부를 늘리는 레버**이므로
#                여기가 진짜 위험이다 — 정당한 write 를 막아 다른 칸을 깨뜨리는지 본다(§1.3).
#   P5 라이브락  `resolve the flagged call` 수(R9 가 산 계기가 유지되는가)
#
# ⚠nt=6 = **태스크당 6 시행**(총 12 sim). 3 sim×2 로는 총점을 못 가른다(C403·C406·C408·C415).
#   12/12 을 겨누는 런이므로 분모도 12 로 맞춘다.
# ⚠태그는 새 것을 쓴다([[30]]).
#
# usage: run_callform_20260811.sh [TASKS] [NT] [TAG]
set -e
REPO=/home/woori/workspace_common/boltzmann-attention-pi
cd "$REPO/scripts/distill/tau2"

TASKS="${1:-task_099,task_010}"
NT="${2:-6}"
TAG="${3:-bank_cf_20260811}"
LOG=/home/woori/scratch/logs
mkdir -p "$LOG"

# ★가드는 **같은 포트**일 때만 거부한다 (2026-08-11·사용자 지시 *"8140 GPU0 에서만 실행하라"*).
#   구판은 t2_run_gated 가 하나라도 있으면 거부했는데, 이 서버는 GPU 두 장에 vLLM 두 개(8140·8141)
#   라 **다른 GPU 의 런까지 막았다**. 겹치면 안 되는 것은 프로세스가 아니라 **모델 서버**다.
#   ⚠유료 예산은 user-sim 쪽에서 공유된다([[09]]) — 병렬은 사용자 승인이 있을 때만.
PORT=8140
if ps -eo cmd | grep -v grep | grep "t2_run_gated.py" | grep -q "localhost:${PORT}/"; then
  echo "[run] REFUSING: 포트 ${PORT} 에서 t2_run_gated 가 이미 돌고 있다." >&2; exit 1
fi
if ps -eo cmd | grep -v grep | grep -q "t2_run_gated.py"; then
  echo "[run] NOTE: 다른 포트에서 유료 런이 돌고 있다(승인된 병렬):" >&2
  ps -eo cmd | grep -v grep | grep "t2_run_gated.py" | grep -o "save_to [a-z0-9_]*" >&2
fi
if [ -e "$LOG/${TAG}.log" ]; then
  echo "[run] REFUSING: $LOG/${TAG}.log 가 이미 있다. 다른 TAG 를 쓰라." >&2; exit 1
fi

/home/woori/venvs/seka_env/bin/python - <<'PY' || exit 1
import os, subprocess, sys
d = "/home/woori/workspace_common/boltzmann-attention-pi/scripts/distill/tau2"
sys.path.insert(0, d)
bad = []
src = open(os.path.join(d, "t2_gate_patch.py"), encoding="utf-8").read()
for need, why in (
        ('os.environ.get("T2_CALL_FORM") == "1"', "C419 코드가 없다"),
        ('[T2_CALL_FORM] named', "C419 마크가 없다(팔 오염 검사를 못 한다)"),
        ('os.environ.get("T2_ARG_EMPTY") == "1"', "C420 코드가 없다"),
        ('ARG-EMPTY', "C420 문구가 없다"),
        ('_FB_GENERIC = "Error: resolve the flagged call', "R9 OFF 경로 문구가 사라졌다")):
    if need not in src:
        bad.append(why)
for t in ("test_call_form_and_arg_empty.py", "test_keep_deny_body.py",
          "test_unlock_quiet.py", "test_decision_isolate.py"):
    r = subprocess.run(["/home/woori/venvs/seka_env/bin/python", t], cwd=d,
                       capture_output=True, text=True)
    if r.returncode != 0:
        bad.append("%s 실패: %s" % (t, (r.stdout or "")[-200:]))
for f in ("T2_CALL_FORM", "T2_ARG_EMPTY"):
    if os.environ.get(f) == "1":
        bad.append("검증 프로세스에 %s 가 켜져 있다(런처가 켜야 한다)" % f)
print("VERIFY " + ("FAIL: " + " · ".join(bad) if bad else "OK"))
sys.exit(1 if bad else 0)
PY

setsid bash -c "cd '$REPO/scripts/distill/tau2' && source ./go_stack.sh >/dev/null 2>&1 && \
  export T2_ACTION_SUB=1 T2_UNLOCK_QUIET=1 T2_KEEP_DENY_BODY=1 \
         T2_CALL_FORM=1 T2_ARG_EMPTY=1 && t2_launch $TAG 8140 '$TASKS' $NT" \
  </dev/null >"$LOG/${TAG}.log" 2>&1 &
echo "PID=$!"
sleep 12
echo "--- 발사 직후 ---"
head -12 "$LOG/${TAG}.log" 2>/dev/null || true
echo "launched · tasks=$TASKS nt=$NT · log: $LOG/${TAG}.log"
echo "  sidecar: $LOG/fb_${TAG}.jsonl · trace: $LOG/trace_${TAG}.jsonl"
