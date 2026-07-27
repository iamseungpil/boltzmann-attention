#!/bin/bash
# ★정본 GO-STACK 런처 (single source of truth · 2026-07-25 C167)
# 목적: 스택 조성이 세션/사람 기억에 의존해 플래그가 누락되는 사고 방지([[07]] hard-constraint).
#   모든 라이브 런은 이 파일을 source한 뒤 t2_launch로 띄운다. 실험 arm은 그 위에
#   T2_* 플래그를 추가/제거하고, 그 차이를 런 태그와 원장에 명시한다.
#
# ★합성-우선 원칙(사용자 지시 2026-07-25·C168): **성공한 레버는 전부 함께 켠다** —
#   간섭은 합성 런에서만 드러나고, 드러나야 레버간 조정이 가능하다(격리 검증만으로는
#   간섭이 영원히 미지·"최종 스택"이 미검증 상태로 남음). 실증: UNIFIED=gate+prov 합성이
#   드러낸 CONFLICT의 조정물·guided pre-gate 순서=합성 라이브가 드러낸 관통의 조정물.
#   ⇒ 기본 = 전부 ON. 간섭 관측 시 레버를 끄는 게 아니라 **조정**(통합·순서·창/캡)한다.
#   개별 격리 검증은 "귀속용 실험 arm"에서만(그때 명시적으로 끄고 태그에 기록).
#
# ── 환경(정본) ──────────────────────────────────────────────────────────────
export GO_REPO=/home/woori/workspace_common/boltzmann-attention-pi
export GO_TAU2=/home/woori/scratch/tau2-bench           # ★정본 tau2([[30]]·C166 사고 재발 방지)
export PYTHONPATH=src:$GO_REPO/scripts/distill/tau2
source /home/woori/.openrouter_key                       # export OPENROUTER_API_KEY=... 형식

# ── [GO] 기본 스택 (nt4 계보·E11-e2e GO·C146~C149 아크) ────────────────────
export T2_OVERFLOW_GUARD=1
export T2_GATE_REGEN=1          # UNIFIED regen의 gate 축 (단독 아님·아래 PROV와 통합 라우팅)
export T2_GATE_REGEN_K=1        # (구 C173의 K=2는 unified 경로서 미사용=no-op으로 판명·철회.
                                #  비-unified 분기 전용 knob이라 기본 1 유지.)
export T2_PRECLOSE_CAP=2        # ★C173-corr(2026-07-25): 진짜 원인=pre-close deny가 공유
                                #  T2_EPLAN_DENY_CAP(4)을 discovery deny와 나눠 써서 044서 소진→
                                #  2번째 close 통과(CLOSED·상태오염 3). 전용 예비 예산으로 분리
                                #  (t2_gate_patch 3386~·claimprov transfer-창 §2ao 선례 동일 패턴).
export T2_PROV_REGEN=1          # 출처선언/provenance (E11 GO·C45 67→0%)
export T2_PROV_REGEN_K=4
export T2_PROV_MODE=full
export T2_GROUND=1              # P-A GROUND (T5-C rev3)
export T2_EPLAN=1               # E-PLAN ledger+walk ([[14]])
export T2_EPLAN_WALK=1
export T2_BRANCH_REGROUND=1     # C146 make-or-break GO·C149 close-차단 인과 [S]
export T2_SCAFFOLD_GET=1        # A2 scaffold_get_tools (검증기 GET)
# ★C186(2026-07-25·W7 004 부검서 발견한 **런처 누락 회귀**): 아래 둘은 2026-07-18/20에 검증돼
#   당시 런 스크립트 10여 개(run_e2e9/e2e10/r095*/hv_*/eplan_smoke)가 켜던 레버인데, go_stack이
#   정본 런처가 되면서(C167/C168) **조용히 빠졌다** = [[19]] 합성-우선 위반.
#   · T2_A2_VARIANT=ledger → verify_identity의 **record 슬롯 삭제**(match_verdict_grounded).
#     근거 `VERIFY_IDENTITY_LEDGER_BINDING_DESIGN_2026_07_18`: record 날조 46%·grounded 0/24·
#     A2 설명 레버 0/24·라이브 PROV 미포착 ⇒ 슬롯 삭제만 남음. **누락의 대가 = task_004**:
#     조회 실패("No records found") 후 모델이 record를 날조(DOB 01/15/1985·"123 Main St")했고
#     우리 도구가 그것을 모델 자신의 provided와 대조해 **VERIFIED 발급**(가짜 검증).
#     변이 ON이면 record 인자가 아예 없어 이 경로가 구조적으로 불가.
#     (ratefix = get_reward_discrepancies rate 테이블 교정본·같은 시기 검증분)
#   · T2_SG_GROUND=1 → A2 `ground` 선언 도구(check_rebate/apy/interest/closure 4종)의 operand를
#     KB/원장 대조로 검증·미검증은 드롭→abstain(가짜 정밀도 차단).
export T2_A2_VARIANT=ledger,ratefix
export T2_SG_GROUND=1

# ★★C186 검증-레버 복원(2026-07-25·[[19]] 이행) ─────────────────────────────
#  발견: go_stack은 C167서 **13개로 새로 작성**됐고(tiers GO/VAL/TGT+promotion rule) C168이
#  "성공 레버 전부 ON"을 선언했지만 **승격이 실행되지 않아** 직전 검증 런(`run_rall25_20260724`
#  =56 레버)의 **43개가 스택에서 이탈**했다. 그런데 handoff/메모리는 go_stack을 "전 레버 ON"으로
#  기록 ⇒ 문서-실제 불일치. C185 34-fail 포렌식이 "미설계"로 분류한 표적 다수가 **이미 구현된
#  레버를 끈 상태**였다: ⑤KB반복=READ_DEDUP · ⑤컨텍스트=VIEW_COMPACT · ③가공도구=UNKNOWN_NAME_BL
#  · ①라우팅=DISPATCH_ROLE/UNLOCK_NAME · ⑨값=WRITE_EVIDENCE/REF_VERIFY · ②완료날조=FOLLOWUP_*.
#  ⚠**단서(정직)**: 56-레버 시대 런(rall21~25)은 0/4였고 18-레버 go_stack이 12 pass를 냈다 —
#  즉 "많을수록 좋다"는 증거는 없다. [[19]] 대응은 끄기가 아니라 조정이므로 **복원 후 기준셋
#  (032/033/035/043/058) 재측정으로 회귀를 확인**해야 하고, 그 전에 스모크 필수([[30]]).
#  ⚠**런타임**: SG_ISOLATE/FORCE 계열은 서브콜·토큰을 늘린다(20~60분/태스크 관측).
export T2_COMPUTE=1 T2_RESOLVE=1 T2_ARG_SCHEMA=1 T2_TOOLGATE=1
export T2_SG_TRUTH=1 T2_SG_ISOLATE=1 T2_SG_ISOFB=1 T2_SG_REQREADS=1 T2_SG_TRACE=1
export T2_FAB_STRIP=1 T2_UNKNOWN_NAME_BL=1 T2_UNLOCK_NAME=1 T2_UNLOCK_PROV=1
export T2_DISPATCH_ROLE=1 T2_TOOLLIST=1 T2_PRESCRIPTION=1
export T2_WRITE_EVIDENCE=1 T2_WEV_ROUNDS=2 T2_WRITE_ARG_GROUND=1 T2_WRITE_PROV=1
export T2_REF_VERIFY=1 T2_VALUE_ACQUIRE=1 T2_HAVE_VALUE=1 T2_HAVE_VALUE_FORCE=1
export T2_FOLLOWUP_REQUIRED=1 T2_FOLLOWUP_FORCE=1 T2_FOLLOWUP_READLOOP=1
export T2_FOLLOWUP_CAP=3 T2_FOLLOWUP_PROGRESS_REFUND=1
export T2_FORCE_ACTION=1 T2_ACTION_DENY_CAP=3 T2_ACTION_PROGRESS_REFUND=1
export T2_VERIFY_DENY_CAP=2 T2_PARAM_CAP=1 T2_PAIRCHECK=1 T2_PAIRFIX=1
export T2_STALE_STRIP=1 T2_READ_DEDUP=1 T2_VIEW_COMPACT=1 T2_VIEW_ANNOTATE=1
export T2_COV_MIDDRIVE=1 T2_COV_MIDDRIVE_K=4 T2_EPLAN_DRIVE_K=4
export T2_REGEN_BUDGET=12 T2_LLM_RETRIES=1 T2_LLM_TIMEOUT=480
# (제외 유지: dd_fb·retry·투표 = 실측 해로움·C154/C168)

# ── 신규 레버 (합성-우선 원칙에 따라 전부 기본 ON·C168) ─────────────────────
export T2_GUIDED=1              # C162 실증·C166 체인수정(pre-gate 순서=합성 조정물)
export T2_PREKB=1               # C165 행동-키 검색 게이트
# ★C204/D7(2026-07-27): 동일-인자 계산도구 반복=결정론 stub(022 ctx초과 10회·003 5회 실측 표적).
#   evidence_from(원장-의존)·fetch_formalize(env-가변)는 자동 제외 — 005형 정당 재호출 보호.
export T2_SG_DEDUP=1
export T2_CLAIM_PROV=1          # claim-날조 원장대조(사임/transfer 창·035 기전 표적)
export T2_CLAIMPROV_CAP=3       # cap=1은 빈손 regen 1회에 전소(코드 포렌식 실측)→스모크 권장 3

# ── 간섭 감시점(합성 런에서 로그 마크로 확인·관측 시 '조정'이 기본 대응) ────
#  W1 claim_prov × EPLAN drive: 둘 다 사임/user_stop 창에서 발화 → 같은 턴 이중 넛지
#     (over-steer) 여부. 마크: [T2_CLAIMPROV]·[T2_EPLAN] drive 동일 턴 공발화.
#  W2 claim_prov × PREKB: transfer 호출이 양쪽 창을 동시 트리거(생성-레벨 감사 +
#     실행-레벨 deny) → C152형 포기 유발 여부. 각각 캡 有(1/fam·1/sim)로 유계.
#  W3 guided × claim_prov 서브콜: 감사 서브콜은 tools=None → 문법 미주입(무간섭 확인됨).

# ── 폐기/실험전용 (기본 OFF 유지 — '성공한 레버'가 아님) ────────────────────
# export T2_DD_FB=1             # C154 폐기 권고(soft·교란)
# export T2_MAXPROMPT=1         # 프롬프트-한계 실험 전용

# ── 공통 런처 함수 ──────────────────────────────────────────────────────────
# 사용: t2_launch <TAG> <PORT> <TASK_IDS> <NUM_TRIALS> [EXTRA_ARGS...]
t2_launch() {
  local TAG="$1" PORT="$2" TASKS="$3" NT="$4"; shift 4
  cd "$GO_TAU2" || return 1
  /home/woori/venvs/seka_env/bin/python -u "$GO_REPO/scripts/distill/tau2/t2_run_gated.py" \
    --domain banking_knowledge --retrieval_config bm25 \
    --gate 1 \
    --agent_model Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8 \
    --agent_base "http://localhost:${PORT}/v1" \
    --user_llm openrouter/openai/gpt-5.2 --user_temp 0.0 \
    --task_ids "$TASKS" --num_trials "$NT" --max_concurrency 4 \
    --max_steps 200 \
    --save_to "$TAG" "$@"
}
