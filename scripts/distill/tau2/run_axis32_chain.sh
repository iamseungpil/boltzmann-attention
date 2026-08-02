#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# AXIS-32 — 6축 포렌식 처방 **전부 탑재** 실험 (2026-08-02)
#
# 스택 = qp32(QUOTE_PIN) + 축-레버 전량. 대조 = `bank_qp32p{1,2}_20260802`(QUOTE_PIN만) ⇒
#   **단일 변수 = 축-레버 묶음**. 나머지 env는 y2cp3 기준과 동일함이 이미 실측 대조됨.
#
# 탑재 처방(전부 플래그·기본 OFF를 여기서만 ON):
#   T2_TOOL_CHANNEL     채널 오분류 + 방출 불가 표면화(레지스트리=env DISCOVERABLE_ATTR 런타임 도출)
#   T2_TERMINAL_TURN    이관 동의 후 미실행(008·035형)
#   T2_FIT_DIFF         후보 미판별(003·007·023·024형) — 갈리는 필드만·추천 0
#   T2_SCALAR_ARRAY     배치화(029형)
#   T2_NO_DIGEST_REEXEC 뷰-압축↔dedup 상쇄로 인한 대형 테이블 전문 재유입(022형)
#   T2_REPEAT_CAP=8     반복 캡(x35 ① K곡선: K=8에서 과차단 하한 0) + 3층 누적 기록
#   T2_RETURN_EMPTY     D4 모순((none)인데 갱신 지시)
#   T2_GROUND_HDR       D3 헤더-상세 모순
#   T2_NOREC_BRANCH     D1 종료분기 + D1b 순서지시
#   T2_AXIS_NOTE_CAP=2  ★표면화 발화 상한 — 없으면 026에서 55회(오프라인 재생이 잡은 자기 결함)
#
# 사전 근거: 단위 19검정 + 회귀 11종 PASS · 오프라인 재생(x42) 35 notes/32 sim·실패 24 중 16 결정점 적중.
# 함정: 발사 전 pgrep · [cfg] 출력 · 완주 후 envcheck · pass마다 즉시 영속화([[30]]).
# ★trace는 **경로**다(불리언 아님 — qp32 러너의 결함 교정).
# 사용: bash run_axis32_chain.sh [smoke|nt2]
# ─────────────────────────────────────────────────────────────────────────────
set -u
R=/home/woori/workspace_common/boltzmann-attention-pi
D=20260802
G0=task_005,task_006,task_007,task_008,task_015,task_016,task_021,task_023,task_027,task_028,task_032,task_033,task_034,task_035,task_040,task_041
G1=task_001,task_002,task_003,task_004,task_010,task_012,task_014,task_017,task_018,task_019,task_020,task_022,task_024,task_025,task_026,task_029
SMOKE0=task_027,task_028
SMOKE1=task_019,task_029
log(){ echo "[axis32 $(date +%m-%d\ %H:%M)] $*"; }

if pgrep -f "[t]2_run_gated" >/dev/null; then
  log "❌ 중단 — 다른 드라이버가 이미 돈다"; exit 1
fi
for P in 8140 8141; do
  curl -s -m 5 http://localhost:$P/v1/models >/dev/null || { log "❌ 중단 — serve $P 무응답"; exit 1; }
done
log "선행 점검 통과"

one(){ # $1=tag $2=gpu $3=port $4=tasks
  cd /home/woori/scratch/tau2-bench
  rm -rf data/simulations/bank_$1_gpu$2_$D
  source $R/scripts/distill/tau2/go_stack.sh
  export T2_DECLFIRST=1 T2_DECLFIRST_GUIDE=0 T2_DECLFIRST_ENFORCE=0
  export T2_TOOL_SIGNATURE=0 T2_TOOL_SIGNATURE_OBSERVE=1
  export T2_QUOTE_PIN=1
  # ★축-레버 전량 ON
  export T2_TOOL_CHANNEL=1 T2_TERMINAL_TURN=1 T2_FIT_DIFF=1 T2_SCALAR_ARRAY=1
  export T2_NO_DIGEST_REEXEC=1 T2_REPEAT_CAP=8
  export T2_RETURN_EMPTY=1 T2_GROUND_HDR=1 T2_NOREC_BRANCH=1
  export T2_AXIS_NOTE_CAP=2
  export T2_SG_ISOLATE_TRACE=/home/woori/scratch/axis32run/trace_$1_gpu$2.jsonl
  export T2_FB_SIDECAR=/home/woori/scratch/axis32run/$1_gpu$2_sidecar.jsonl
  mkdir -p /home/woori/scratch/axis32run
  echo "[cfg gpu$2] CHANNEL=$T2_TOOL_CHANNEL TERM=$T2_TERMINAL_TURN FIT=$T2_FIT_DIFF ARR=$T2_SCALAR_ARRAY NODIGEST=$T2_NO_DIGEST_REEXEC CAP=$T2_REPEAT_CAP EMPTY=$T2_RETURN_EMPTY HDR=$T2_GROUND_HDR NOREC=$T2_NOREC_BRANCH NOTECAP=$T2_AXIS_NOTE_CAP QP=$T2_QUOTE_PIN MAXTOK=${T2_AGENT_MAX_TOKENS:-unset} TIMEOUT=${T2_LLM_TIMEOUT:-unset}"
  /home/woori/venvs/seka_env/bin/python -u $R/scripts/distill/tau2/t2_run_gated.py \
    --domain banking_knowledge --retrieval_config bm25 --gate 1 \
    --agent_model Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8 --agent_base http://localhost:$3/v1 \
    --user_llm openrouter/openai/gpt-5.2 --user_temp 0.0 \
    --task_ids $4 --num_trials 1 --max_concurrency 2 --max_steps 200 \
    --save_to bank_$1_gpu$2_$D > /home/woori/scratch/logs/$1_gpu$2.log 2>&1
}

persist(){
  cd $R && git pull -q --rebase origin facet-rft-2026 2>/dev/null
  for g in 0 1; do
    S=/home/woori/scratch/tau2-bench/data/simulations/bank_$1_gpu${g}_$D/results.json
    [ -f "$S" ] && gzip -c "$S" > $R/reports/facet_rft_2026/sim_results/bank_$1_gpu${g}_$D.results.json.gz
    L=/home/woori/scratch/logs/$1_gpu${g}.log
    [ -f "$L" ] && gzip -c "$L" > $R/reports/facet_rft_2026/sim_results/bank_$1_gpu${g}_$D.log.gz
    T=/home/woori/scratch/axis32run/trace_$1_gpu${g}.jsonl
    [ -f "$T" ] && gzip -c "$T" > $R/reports/facet_rft_2026/sim_results/bank_$1_gpu${g}_$D.trace.jsonl.gz
  done
  cd $R && git add -f reports/facet_rft_2026/sim_results/bank_$1_gpu*_$D.* \
    && git -c user.email=woori@local -c user.name=woori commit -q -m "Persist AXIS-32 $1 (all axis levers on)" \
    && git push -q origin facet-rft-2026
  log "$1 영속화+push 완료"
}

MODE=${1:-smoke}
if [ "$MODE" = "smoke" ]; then
  log "★스모크 — 4태스크(축 레버가 라이브에서 실제로 발화하는지·[[30]])"
  one axsmoke 0 8140 "$SMOKE0" &
  one axsmoke 1 8141 "$SMOKE1" &
  wait
  log "스모크 완주 — 레버 발화 계수"
  grep -ho "\[T2_AXIS\][^|]\{0,60\}" /home/woori/scratch/logs/axsmoke_gpu*.log | sort | uniq -c | sort -rn | head
  grep -hc "REPEAT-CAP" /home/woori/scratch/logs/axsmoke_gpu*.log
  persist axsmoke
else
  for TAG in ax32p1 ax32p2; do
    log "$TAG 발사"
    one $TAG 0 8140 "$G0" &
    one $TAG 1 8141 "$G1" &
    wait
    log "$TAG 완주"
    persist $TAG
  done
  log "체인 종료 — AXIS-32 nt=2(64 sim)"
fi
