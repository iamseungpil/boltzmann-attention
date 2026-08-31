#!/usr/bin/env bash
# 우리 전 스택으로 **한 태스크**를 돌린다 — 인자 구동·repo 버전관리본.
#
# 사용: run_ours_task.sh <TAG> <PORT> <TASK_IDS>
#   예: run_ours_task.sh bank_x696_t094_20260831 8141 task_094
#
# ★왜 repo 안인가 (2026-08-31·사용자 지시 *"모든 수정은 git 으로 하라"*):
#   런 스크립트가 리모트 홈(`/home/woori/run_x69*.sh`)에만 있어서 **정본과 갈렸다**. 실제 피해:
#   `export T2_AGENT_MAX_TOKENS=3072` 한 줄이 정본 `go_stack.sh` 의 8192 를 **근거 주석 0줄로**
#   덮었고, 그 캡이 `</think>` 이전에 걸려 생성 전량이 reasoning 으로 분류 → `content=None` ∧
#   `tool_calls` 부재 → tau2 `AssistantMessage must have either content or tool_calls` →
#   **태스크 전체 재시작**(x693 에서 1,590초 폐기). 런처가 repo 밖이면 이 종류의 이탈이 안 보인다.
#
# ⛔[[19]]/[[60]] 레버는 전부 켠다 — `go_stack.sh` 를 source 하고, 여기서는 **실험 축만** 얹는다.
# ⛔[[30]] 함정 2: 포트의 모델이 바뀌어 있을 수 있다 → 발사 직전 `/v1/models` 대조(아래 REFUSING).
# ⛔[[30]] 함정 6: 출력은 리모트 파일로 남긴다(채널 타임아웃 유실 금지).
set -o pipefail

# 옵션: --profile <이름|경로>  (미지정 = 서빙 중인 모델 id 로 자동 선택)
#       --no-preflight            (표면형 선발사 검산 생략 — 권하지 않는다)
PROFILE=""
PREFLIGHT=1
POS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --profile) PROFILE="$2"; shift 2 ;;
    --profile=*) PROFILE="${1#*=}"; shift ;;
    --no-preflight) PREFLIGHT=0; shift ;;
    -h|--help)
      echo "사용: run_ours_task.sh [--profile 이름|경로] [--no-preflight] <TAG> <PORT> <TASK_IDS>"
      echo "  프로필 = model_profiles/<모델 id 의 / 를 __ 로 바꾼 이름>.env"
      ls -1 "$(dirname "$0")/model_profiles" 2>/dev/null | sed 's|^|  가용: |'
      exit 0 ;;
    *) POS+=("$1"); shift ;;
  esac
done
set -- "${POS[@]}"

TAG=${1:?TAG 를 주어라 (예: bank_x696_t094_20260831)}
PORT=${2:?PORT 를 주어라 (8141 | 8143)}
TASKS=${3:?TASK_IDS 를 주어라 (예: task_094)}

REPO=${REPO:-/home/woori/workspace_common/boltzmann-attention-pi}
GO_TAU2=${GO_TAU2:-/home/woori/scratch/tau2-bench}
LOG=${LOG:-/home/woori/scratch/logs}
# 기대 모델 — 미지정이면 **서버가 서빙 중인 것**을 그대로 쓴다(모델을 바꿔도 런처는 그대로).
EXPECT=${EXPECT:-}

source /home/woori/.openrouter_key
# 임베딩 검색(KB_search_dense)은 OPENAI_API_KEY 없이 죽는다 — base 는 230회 쓰는데 우리 런은
# 첫 호출에서 Missing credentials 로 죽고 포기했다(2026-08-31 수리①).
source /home/woori/.openai_key
[ -n "${OPENROUTER_API_KEY:-}" ] || { echo "REFUSING: no OPENROUTER_API_KEY"; exit 1; }

S=$(curl -s -m 10 "http://localhost:$PORT/v1/models" | grep -oE '"id":"[^"]+"' | head -1 | cut -d'"' -f4)
[ -n "$S" ] || { echo "REFUSING: 포트 $PORT 에서 모델 id 를 못 읽었다"; exit 1; }
if [ -n "$EXPECT" ]; then
  echo "[run_ours] 서빙: $S (기대 $EXPECT)"
  [ "$S" = "$EXPECT" ] || { echo "REFUSING: 포트 $PORT 가 $S 를 서빙중 - 발사 중단"; exit 1; }
else
  EXPECT="$S"
  echo "[run_ours] 서빙: $S (기대 미지정 → 서빙 중인 모델을 쓴다)"
fi
BUSY=$(ps -eo args --no-headers | grep "localhost:$PORT" | grep -v "grep " | grep -c "tau2 run\|t2_run_gated" || true)
[ "$BUSY" -gt 0 ] && { echo "REFUSING: $PORT 사용중"; exit 1; }

cd "$REPO/scripts/distill/tau2" || exit 1
source ./go_stack.sh >/dev/null 2>&1

# ★모델 프로필 — 모델·서버에 매인 값만 담긴 config([[84]]). 순서: go_stack(정본 기본값)
#   → 프로필(모델별) → 아래 실험 축. 없으면 **발사를 거부**한다(조용한 불일치 금지).
if [ -z "$PROFILE" ]; then
  # `Qwen/Qwen3.8-27B-FP8` → `Qwen__Qwen3.8-27B-FP8.env`
  PROFILE="model_profiles/$(echo "$EXPECT" | sed 's|/|__|g').env"
elif [ -f "model_profiles/$PROFILE.env" ]; then
  PROFILE="model_profiles/$PROFILE.env"
fi
[ -f "$PROFILE" ] || {
  echo "REFUSING: 모델 프로필이 없다 → $PROFILE"
  echo "  이 모델로 처음 돌리는 것이라면 프로필을 **먼저** 만들어라(값마다 출처 한 줄·README 참조)."
  ls -1 model_profiles/*.env 2>/dev/null | sed 's|^|  가용: |'
  exit 1; }
echo "[run_ours] 프로필: $PROFILE"
# shellcheck disable=SC1090
source "$PROFILE"
export T2_ACTION_SUB=1 T2_KEEP_DENY_BODY=1 T2_CALL_FORM=1 T2_ARG_EMPTY=1 T2_SEARCH_AGENT=1
export T2_SG_DOCS=1 T2_SG_PROMPT_V2=1 T2_SPEC_AT_WRITE=1 T2_WRITE_ARG_TYPE=1
export T2_RULE_AT_WRITE=1 T2_DUP_WRITE=1
export T2_ACTIONREQ_GROUNDED=1 T2_SG_ROW_COUNT=1 T2_SG_CLOSE_SELF=1
export T2_SG_REQREADS=1 T2_SG_REQREADS_CANON=1
export T2_SEARCH_EXHAUST_MID=1
export T2_TOOL_OBS=1
export T2_GEN_TRACE=1 T2_NO_FORCE_TOOLCHOICE=1 T2_PROBE_TERSE=1 T2_TC_SALVAGE=1 T2_P2_REGEN=1
export T2_GUIDED_VERBOSE=1   # 어느 표면형으로 문법이 걸렸는지 로그에 남긴다([[81]] 발화 확인)
export T2_FAILDUMP=$LOG/faildump_${TAG}.jsonl
export T2_FB_SIDECAR=$LOG/fb_${TAG}.jsonl T2_FB_SIDECAR_TEXT=1
export T2_TRACE=$LOG/trace_${TAG}.jsonl

# ★상한은 정본(go_stack `T2_AGENT_MAX_TOKENS=8192`)을 **덮지 않는다**. 근거(2026-08-31 전수):
#   base 궤적 1,713콜의 completion_tokens 분포 p50=268 · p90=1,778 · **p99=7,408** · max=17,075
#   이고 `completion_tokens ≥ 3072` 가 **106/1713 = 6.19%** 다. 3072 캡은 그 6%를 전손시킨다
#   (우리 팔 대조: mt=8192 x681 전손 0/30 ↔ mt=3072 x687·688·689·692·693 각 1~3건).
#   ⚠[[82]]: 8192 는 Q2.5 기준값이다 — Q3.8 재측정 전까지는 **정본을 그대로 쓰고**, 바꾸려면
#     분포를 먼저 재고 근거를 여기 적어라.
echo "[run_ours] T2_AGENT_MAX_TOKENS=${T2_AGENT_MAX_TOKENS:-(미설정)} · T2_TOOL_SURFACE=${T2_TOOL_SURFACE:-(미선언)}"
[ -n "${T2_TOOL_SURFACE:-}" ] || { echo "REFUSING: T2_TOOL_SURFACE 미선언 — 문법 표면형이 서버 파서와 어긋나면 도구 파싱이 전량 죽는다(x703)"; exit 1; }
echo "[run_ours] 켜진 T2_/GO_ 변수 $(env | grep -cE '^(T2_|GO_)') 개"

# ★표면형 검산 — 선언을 믿지 않고 **한 번 쏴 본다**(요청 2개·max_tokens 128).
#   문법이 파서와 어긋나면 네이티브 도구 파싱이 전량 죽는데, 그것은 런이 끝나야 보인다([[84]]).
if [ "$PREFLIGHT" = "1" ]; then
  /home/woori/venvs/seka_env/bin/python "$REPO/scripts/distill/tau2/x704_surface_preflight.py"       "http://localhost:$PORT/v1" "$EXPECT"
  _pf=$?
  [ "$_pf" = "1" ] && { echo "REFUSING: 표면형 검산 실패 - 발사 중단"; exit 1; }
  [ "$_pf" = "2" ] && echo "[run_ours] ⚠표면형 검산 불가 — 그대로 발사한다"
fi

cd "$GO_TAU2" || exit 1
export PYTHONPATH=src:$REPO/scripts/distill/tau2
/home/woori/venvs/seka_env/bin/python -u "$REPO/scripts/distill/tau2/t2_run_gated.py" \
  --domain banking_knowledge --gate 1 --retrieval_config alltools \
  --agent_model "$EXPECT" --agent_base "http://localhost:$PORT/v1" \
  --user_llm openrouter/openai/gpt-5.2 --user_temp 0.0 --user_reasoning_effort low \
  --task_ids "$TASKS" --num_trials 1 --max_concurrency 1 --max_steps 200 \
  --save_to "$TAG" 2>&1 | tee "$LOG/$TAG.log"

echo "=================================================================="
echo "[run_ours] $(date '+%F %T') $TAG  pass1=$(grep -aoE 'pass1=[0-9]+/[0-9]+' "$LOG/$TAG.log" | tail -1)  TB=$(grep -ac Traceback "$LOG/$TAG.log")"
echo "[run_ours] 계기: SALVAGED=$(grep -c 'SALVAGED=' "$LOG/$TAG.log") · TRUNC=$(grep -c 'TRUNC' "$LOG/$TAG.log") · 빈메시지재시작=$(grep -ac 'must have either content or tool_calls\|Message must have content or tool calls' "$LOG/$TAG.log")"
