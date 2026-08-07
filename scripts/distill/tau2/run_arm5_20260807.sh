#!/usr/bin/env bash
# arm-5 — 계약 5개만 켜고 나머지 레버를 전부 끈다 (사용자 지시 2026-08-07: "레버는 5개만이다").
#
# 왜 (`LEVER_TO_CONTRACT_MAP_2026_08_07`):
#   "79 레버를 5개로"는 **A2 키** 이야기였고, 통합 커밋(`9eb0a946`)은 설계서 1개 파일만 바꿨다.
#   엔진 레버는 `go_stack.sh` 기준 **97개**가 그대로 살아 있다. 계약이 정말 일반규칙이라면
#   97개를 꺼도 바닥이 유지돼야 하고, 무너지면 **어느 계약이 무엇을 못 흡수했는지**가 드러난다.
#   어느 쪽이든 답이 나온다.
#
# ★자기 기록의 반증도 같은 방향이다 — go_stack.sh 헤더 축자:
#   "56-레버 시대 런(rall21~25)은 0/4였고 18-레버 go_stack이 12 pass를 냈다 —
#    즉 '많을수록 좋다'는 증거는 없다."
#
# 방식: go_stack을 source해 **정본 환경과 t2_launch를 얻은 뒤**, T2_* 를 allowlist만 남기고
#   전부 unset한다. 목록을 손으로 옮겨 적지 않으므로 go_stack이 바뀌어도 arm 정의가 안 썩는다.
#
# ⚠정직하게: 이 arm이 실제로 켜는 계약은 **C1(출처)·C3(중재, C2 선행 경로를 그 안에서 태움)** 이다.
#   C4는 설계상 무플래그인데 라이브 구현이 `DISPATCH_ROLE` 등 개별 플래그에 흩어져 있어 여기서 꺼진다.
#   C5는 `t2_offload.py`가 아직 없다. ⇒ 결과 해석 시 **C4·C5는 부재**로 읽어야 한다.
set -u
REPO=/home/woori/workspace_common/boltzmann-attention-pi
cd "$REPO/scripts/distill/tau2" || exit 1
# shellcheck disable=SC1091
source ./go_stack.sh >/dev/null 2>&1

# ── 남기는 것: 설정뿐 (레버 아님) ────────────────────────────────────────────
#   KB_DOCS_DIR   검색 코퍼스 경로 — 없으면 회수 자체가 안 된다
#   LLM_*·AGENT_MAX_TOKENS  런타임 한계
#   A2_VARIANT    **A2 데이터 선택**이지 레버가 아니다. `ledger` 변이는 verify_identity의 record
#                 슬롯을 없애 날조된 VERIFIED 경로를 구조적으로 막는다(task_004 사고). 끄면
#                 계약과 무관한 회귀가 섞인다.
#   FB_SIDECAR    비커밋 관측(거동 변화 0) — 이게 없으면 포렌식의 절반이 불가능하다
KEEP="T2_KB_DOCS_DIR T2_LLM_TIMEOUT T2_LLM_RETRIES T2_AGENT_MAX_TOKENS T2_A2_VARIANT"
KEEP="$KEEP T2_FB_SIDECAR T2_FB_SIDECAR_TEXT"
KEEP="$KEEP T2_SOURCE T2_ARBITRATE T2_LEDGER T2_RESOLVE T2_FORCE_ACTION"

off=0
for v in $(compgen -e | grep '^T2_' | sort); do
  case " $KEEP " in
    *" $v "*) : ;;
    *) unset "$v"; off=$((off+1)) ;;
  esac
done

export T2_SOURCE=1      # C1 출처 (근거 확보)
export T2_ARBITRATE=1   # C3 중재 (합병·등급) — C2 선행 그래프를 이 경로가 태운다
export T2_FORCE_ACTION=1 # C2 선행 집행: 미충족 표적을 향한 push를 잡는 트리거
export T2_RESOLVE=1     # C2 선행 집행: per-operand 해소 + user-action 탐지(_utgt)
export T2_LEDGER=1      # C5 이관: 원장 산수(전사=모델·산수=엔진)
# C4 역할 = 무플래그 위생(t2_role) · C6 창 = t2_window (배선 전이라 이 arm엔 미포함)

echo "arm-5: unset ${off} T2_* flags; contracts on = C1 SOURCE · C2 RESOLVE+FORCE_ACTION · C3 ARBITRATE · C5 LEDGER"
compgen -e | grep '^T2_' | sort | sed 's/^/  keep /'

TASKS="${1:-task_100,task_101,task_102}"
NT="${2:-1}"
LOG=/home/woori/scratch/logs
mkdir -p "$LOG"
export T2_FB_SIDECAR="$LOG/fb_arm5_gpu0.jsonl" T2_FB_SIDECAR_TEXT=1
# ⚠자식에서 go_stack을 다시 source하면 **끈 플래그가 전부 되살아난다**. 함수만 넘긴다.
export -f t2_launch
setsid bash -c "cd '$REPO/scripts/distill/tau2' && \
  T2_FB_SIDECAR='$LOG/fb_arm5_gpu0.jsonl' t2_launch bank_arm5_gpu0 8140 '$TASKS' $NT" \
</dev/null >"$LOG/arm5_gpu0.log" 2>&1 &
sleep 2
echo "launched. log: $LOG/arm5_gpu0.log"
