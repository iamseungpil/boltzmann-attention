#!/usr/bin/env bash
# ⛔**사용 금지 (2026-08-07 사용자 지시)**: "앞으로는 레버 끄는 건 없다. 꺼서도 안 된다."
#   이 드라이버는 레버를 끄는 arm을 만든다 = 금지된 구성이다. **기록용으로만 남긴다.**
#   하루에 세 번, 끄기가 arm의 능력을 조용히 없앴다(호스트·채널·보호). 분류는 로그·사이드카
#   분석과 논문 서술에서만 쓰는 렌즈이지 런 구성이 아니다. 정본 = memory [[60]].
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
KEEP="$KEEP T2_SOURCE T2_ARBITRATE T2_LEDGER T2_RESOLVE T2_FORCE_ACTION T2_WINDOW"
# ★호스트 래퍼(2026-08-07·arm-5b 실측): 계약 코드는 전부 `apply_unified_regen` 안에 있고,
#   그 설치 조건이 `T2_GATE_REGEN=1 ∧ (T2_PROV_REGEN|GROUND|badwords|disamb)`이다.
#   둘을 끄면 **계약 코드가 설치조차 되지 않는다** — arm-5·arm-5b에서 마커가 전부 0이었던 이유.
#   즉 이 둘은 레버가 아니라 **우리 층이 말할 수 있게 하는 채널**이다(생성-레벨 deny→재생성).
KEEP="$KEEP T2_GATE_REGEN T2_GATE_REGEN_K T2_PROV_REGEN T2_PROV_REGEN_K T2_PROV_MODE"
# ★C7 발견(2026-08-07 신설·설계서 §2-C7). 이 다섯이 30.2%→63.4%를 만든 축이고, arm-5는 이것을
#   **꺼 놓고** 바닥(59/93)과 비교하려 했다 — 성립할 수 없는 비교였다.
# ★'계약 밖 17개' 전수 재검토(2026-08-07·x125_flag_role_audit) — **전부 살린다**.
#   판정 기준을 '무엇을 결정하는가'로 바꾸면 이들 중 도메인 판정을 하는 것은 하나도 없다:
#     채널   READ_DEDUP  = exec_augment 본문 **216줄**을 감싼다(원장 산수·AXIS·REPEAT가 그 안)
#     보호   OVERFLOW_GUARD TRUNC_GUARD ENVELOPE_GUARD DYN_MT MAXPROMPT VIEW_COMPACT
#            VIEW_ANNOTATE STALE_STRIP PAIRCHECK PAIRFIX = 컨텍스트·토큰·쌍 무결성(023 사고 계열)
#     관측   FAILED_PERSIST = 사이드카 덤프(거동 0)
#     잔재   ACTION_PROGRESS_REFUND FOLLOWUP_PROGRESS_REFUND = 예산 환급(정체-과금으로 대체됨·무해)
#     설정   A2_VARIANT DUP_REPRESENT GUIDED
#   ⚠끄면 arm이 **조용히 능력을 잃는다** — 오늘 READ_DEDUP 하나로 런 6회를 태웠다.
KEEP="$KEEP T2_OVERFLOW_GUARD T2_TRUNC_GUARD T2_ENVELOPE_GUARD T2_ENVELOPE_CAP T2_DYN_MT"
KEEP="$KEEP T2_DYN_MT_MARGIN T2_MT_FLOOR T2_MAXPROMPT T2_VIEW_COMPACT T2_VIEW_COMPACT_MINTOTAL"
KEEP="$KEEP T2_VIEW_MSG_CAP T2_VIEW_ANNOTATE T2_STALE_STRIP T2_PAIRCHECK T2_PAIRFIX"
KEEP="$KEEP T2_READ_DEDUP T2_READ_DEDUP_MIN T2_FAILED_PERSIST T2_DUP_REPRESENT T2_GUIDED"
KEEP="$KEEP T2_ACTION_PROGRESS_REFUND T2_FOLLOWUP_PROGRESS_REFUND"
KEEP="$KEEP T2_DISCOVERY_NAMES T2_UNCALLED_UNLOCK T2_VERDICT_SURFACE T2_TRANSFER_LEAVES_STEPS T2_MATCH_COUNT"

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
export T2_GATE_REGEN=1 T2_GATE_REGEN_K=1          # 호스트: 생성-레벨 훅 설치
export T2_PROV_REGEN=1 T2_PROV_REGEN_K=4 T2_PROV_MODE=full  # C1 출처(+unified 라우팅 조건)
export T2_DISCOVERY_NAMES=1 T2_UNCALLED_UNLOCK=1 T2_VERDICT_SURFACE=1 \n       T2_TRANSFER_LEAVES_STEPS=1 T2_MATCH_COUNT=1   # C7 발견: 출처 안·미사용을 보여준다
export T2_WINDOW=1      # C6 창: 사임 ∪ 행동 ∪ **지시**(표적 이름이 답변에 등장)
# C4 역할 = 무플래그 위생 — t2_role.executor_of로 배선(항상 켜짐)

echo "arm-5: unset ${off} T2_* flags; contracts on = C1 · C2 · C3 · C4(무플래그) · C5 · C6 · C7(발견 5)"
compgen -e | grep '^T2_' | sort | sed 's/^/  keep /'

TASKS="${1:-task_100,task_101,task_102}"
NT="${2:-1}"
TAG="${3:-bank_arm5_gpu0}"
LOG=/home/woori/scratch/logs
mkdir -p "$LOG"
export T2_FB_SIDECAR="$LOG/fb_${TAG}.jsonl" T2_FB_SIDECAR_TEXT=1
# ⚠자식에서 go_stack을 다시 source하면 **끈 플래그가 전부 되살아난다**. 함수만 넘긴다.
export -f t2_launch
setsid bash -c "cd '$REPO/scripts/distill/tau2' && \
  T2_FB_SIDECAR='$LOG/fb_${TAG}.jsonl' t2_launch $TAG 8140 '$TASKS' $NT" \
</dev/null >"$LOG/${TAG}.log" 2>&1 &
sleep 2
echo "launched. log: $LOG/${TAG}.log"
