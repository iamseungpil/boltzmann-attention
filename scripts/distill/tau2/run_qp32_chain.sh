#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# QP-32 밤샘 체인 — `T2_QUOTE_PIN=1` 승격 판정용 front32 × nt=2
#
# 설계: Y2-C(OFF arm)와 **단일 변수만 다르다**. 대조 기준 = `bank_y2cp3_*_20260801`
#   info 실측 대조 완료(2026-08-02): git 5ebebbe8 · max_steps 200 · max_errors 10 ·
#   user gpt-5.2/temp0/timeout2400 · agent max_tokens 8192/timeout 2400 — **전부 동일**.
#   유일 변경 = T2_QUOTE_PIN=1 (+ ISOLATE_TRACE=1 = 관측 전용 로깅·거동 무영향).
#
# 함정 대응(핸드오프 §7):
#   3. 런처마다 다른 env → 발사 전 [cfg] 출력 + 완주 후 info 대조를 강제한다.
#   6. 툴 타임아웃 ≠ 원격 종료 → 이 스크립트는 반드시 `setsid … &` + 로그로 띄운다.
#   [[30]] 결과 소실 방지 → pass마다 즉시 gzip → sim_results → commit → push.
# ─────────────────────────────────────────────────────────────────────────────
set -u
R=/home/woori/workspace_common/boltzmann-attention-pi
D=20260802
G0=task_005,task_006,task_007,task_008,task_015,task_016,task_021,task_023,task_027,task_028,task_032,task_033,task_034,task_035,task_040,task_041
G1=task_001,task_002,task_003,task_004,task_010,task_012,task_014,task_017,task_018,task_019,task_020,task_022,task_024,task_025,task_026,task_029
log(){ echo "[qp32 $(date +%m-%d\ %H:%M)] $*"; }

# ── 선행 점검 (발사 거부 조건) ────────────────────────────────────────────────
if pgrep -f "[t]2_run_gated" >/dev/null; then
  log "❌ 중단 — 다른 드라이버가 이미 돈다(중복 실행·GPU 경합 방지)"; exit 1
fi
for P in 8140 8141; do
  curl -s -m 5 http://localhost:$P/v1/models >/dev/null || { log "❌ 중단 — serve $P 무응답"; exit 1; }
done
log "선행 점검 통과 — 드라이버 유휴 · serve 8140/8141 응답"

one(){ # $1=tag $2=gpu(0/1) $3=port $4=tasks
  cd /home/woori/scratch/tau2-bench
  rm -rf data/simulations/bank_$1_gpu$2_$D
  source $R/scripts/distill/tau2/go_stack.sh
  export T2_DECLFIRST=1 T2_DECLFIRST_GUIDE=0 T2_DECLFIRST_ENFORCE=0
  export T2_TOOL_SIGNATURE=0 T2_TOOL_SIGNATURE_OBSERVE=1
  export T2_QUOTE_PIN=1                 # ★유일한 처치 변수
  export T2_SG_ISOLATE_TRACE=1          # 관측 전용(핀·kind·verdict 기록·거동 무영향)
  export T2_FB_SIDECAR=/home/woori/scratch/$1_gpu$2_sidecar.jsonl
  echo "[cfg gpu$2] QUOTE_PIN=$T2_QUOTE_PIN TRACE=$T2_SG_ISOLATE_TRACE TIMEOUT=${T2_LLM_TIMEOUT:-unset} MAXTOK=${T2_AGENT_MAX_TOKENS:-unset} SIG=$T2_TOOL_SIGNATURE OBS=$T2_TOOL_SIGNATURE_OBSERVE DECLFIRST=$T2_DECLFIRST"
  /home/woori/venvs/seka_env/bin/python -u $R/scripts/distill/tau2/t2_run_gated.py \
    --domain banking_knowledge --retrieval_config bm25 --gate 1 \
    --agent_model Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8 --agent_base http://localhost:$3/v1 \
    --user_llm openrouter/openai/gpt-5.2 --user_temp 0.0 \
    --task_ids $4 --num_trials 1 --max_concurrency 2 --max_steps 200 \
    --save_to bank_$1_gpu$2_$D > /home/woori/scratch/logs/$1_gpu$2.log 2>&1
}

persist(){ # $1=tag
  cd $R && git pull -q --rebase origin facet-rft-2026 2>/dev/null
  for g in 0 1; do
    S=/home/woori/scratch/tau2-bench/data/simulations/bank_$1_gpu${g}_$D/results.json
    [ -f "$S" ] && gzip -c "$S" > $R/reports/facet_rft_2026/sim_results/bank_$1_gpu${g}_$D.results.json.gz
    L=/home/woori/scratch/logs/$1_gpu${g}.log
    [ -f "$L" ] && gzip -c "$L" > $R/reports/facet_rft_2026/sim_results/bank_$1_gpu${g}_$D.log.gz
  done
  cd $R && git add -f reports/facet_rft_2026/sim_results/bank_$1_gpu*_$D.* \
    && git -c user.email=woori@local -c user.name=woori commit -q -m "Persist QP-32 $1 (overnight quote-pin arm)" \
    && git push -q origin facet-rft-2026
  log "$1 영속화+push 완료"
}

envcheck(){ # $1=tag — 완주 후 info를 OFF arm 기준과 대조(함정 3)
  /home/woori/venvs/seka_env/bin/python - "$1" "$D" <<'PY'
import json, sys
tag, D = sys.argv[1], sys.argv[2]
base = "/home/woori/workspace_common/boltzmann-attention-pi/reports/facet_rft_2026/sim_results/bank_y2cp3_gpu0_20260801.results.json.gz"
cur = "/home/woori/scratch/tau2-bench/data/simulations/bank_%s_gpu0_%s/results.json" % (tag, D)
import gzip
b = json.load(gzip.open(base, "rt", encoding="utf-8"))["info"]
c = json.load(open(cur, encoding="utf-8"))["info"]
keys = ["max_steps", "max_errors", "num_trials"]
bad = []
for k in keys:
    if b.get(k) != c.get(k):
        bad.append("%s: base=%r cur=%r" % (k, b.get(k), c.get(k)))
for side in ("user_info", "agent_info"):
    ba = (b.get(side) or {}).get("llm_args") or {}
    ca = (c.get(side) or {}).get("llm_args") or {}
    for k in set(ba) | set(ca):
        if k == "api_base":
            continue
        if ba.get(k) != ca.get(k):
            bad.append("%s.%s: base=%r cur=%r" % (side, k, ba.get(k), ca.get(k)))
print("[envcheck] " + ("OFF arm과 동일 ✓" if not bad else "⚠차이: " + " | ".join(bad)))
PY
}

log "체인 시작 — front32 × 2 pass · QUOTE_PIN=1"
for TAG in qp32p1 qp32p2; do
  log "$TAG 발사 (gpu0=G0 · gpu1=G1)"
  one $TAG 0 8140 "$G0" &
  one $TAG 1 8141 "$G1" &
  wait
  log "$TAG 완주"
  envcheck $TAG
  persist $TAG
done
log "체인 종료 — QP arm nt=2 확보(64 sim)"
