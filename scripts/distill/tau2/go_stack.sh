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
export T2_GATE_REGEN_K=1
export T2_PROV_REGEN=1          # 출처선언/provenance (E11 GO·C45 67→0%)
export T2_PROV_REGEN_K=4
export T2_PROV_MODE=full
export T2_GROUND=1              # P-A GROUND (T5-C rev3)
export T2_EPLAN=1               # E-PLAN ledger+walk ([[14]])
export T2_EPLAN_WALK=1
export T2_BRANCH_REGROUND=1     # C146 make-or-break GO·C149 close-차단 인과 [S]
export T2_SCAFFOLD_GET=1        # A2 scaffold_get_tools (검증기 GET)

# ── 신규 레버 (합성-우선 원칙에 따라 전부 기본 ON·C168) ─────────────────────
export T2_GUIDED=1              # C162 실증·C166 체인수정(pre-gate 순서=합성 조정물)
export T2_PREKB=1               # C165 행동-키 검색 게이트
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
