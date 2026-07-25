#!/bin/bash
# ★정본 GO-STACK 런처 (single source of truth · 2026-07-25 C167)
# 목적: 스택 조성이 세션/사람 기억에 의존해 플래그가 누락되는 사고 방지([[07]] hard-constraint).
#   모든 라이브 런은 이 파일을 source한 뒤 t2_launch로 띄운다. 실험 arm은 그 위에
#   T2_* 플래그를 추가/제거하고, 그 차이를 런 태그와 원장에 명시한다.
#
# 등급 규율(등대 §1.3 — 합성은 측정된 상쇄로만):
#   [GO]   = e2e GO 확정·기본 ON.
#   [VAL]  = 라이브 검증 중·검증 arm에서만 ON(여기선 주석으로 존재만 표시).
#   [TGT]  = 특정 실패-기전 표적·그 기전이 확인된 태스크 arm에서만 ON.
#   승격 절차: VAL/TGT → 표적 태스크 확인 + 참조셋 무퇴행(Δspurious≤0) → GO로 이동·커밋.
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

# ── [VAL] 검증 중 (2026-07-25 라이브 arm: gd4/cand2/cand3) ──────────────────
export T2_GUIDED=1              # C162 실증·C166 체인수정 — 라이브 무퇴행 확인 후 GO 승격
export T2_PREKB=1               # C165 — PREKB 발화→절차발견 인과 확인 후 GO 승격

# ── [TGT] 표적 레버 (해당 기전 태스크 arm에서만 켜고 태그에 명시) ───────────
# export T2_CLAIM_PROV=1        # claim-날조(사임/transfer 창·035 기전). GO 승격 전·cap 튜닝 흔적.
# export T2_DD_FB=1             # discovery-dispatch deny — C154 폐기 권고(soft)·참고용
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
