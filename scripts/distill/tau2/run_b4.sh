#!/bin/bash
# ★B4 런 — front32 × nt2 × alltools (2026-08-03)
#
# arm B4 = ① 이관 통행료 정정 + ④ 회수 경계 표면화.
# arm A 는 **다시 돌지 않는다** — 완주한 `bank_ax33n_gpu{0,1}_20260803g`(01:36~06:05)가 arm A 다.
# 그 시점 커밋(cff28b00)부터 지금까지 런타임에 닿는 변경은 5파일뿐이고 전부 ①·④ 다
# (a2 3파일 · t2_gate_patch.py +17줄 · t2_match_count.py 신규). go_stack.sh·gate_interpreter.py·
# run 드라이버는 미변경이라 레버 스택·회수 설정·user-sim 구성이 A와 동일하다.
#
# ⚠판정문에 반드시 명시할 단서:
#   ⓐ A와 **동시 실행이 아니다**(A=오늘 01:36~06:05). 위 diff가 사이 변경을 ①+④로 한정하지만
#     시간 분리 자체는 남는 사실이다.
#   ⓑ user-sim(gpt-5.2 temp0)은 완전 결정론이 아니다 ⇒ 1급 지표는 pass가 아니라
#     **prefix 미일치 수**(A 기준선 48/76·63%·22 sim = `x47_incantation_baseline_20260803.txt`).
#   ⓒ 짝은 **task id 기준**이라 GPU 배정과 무관하게 유지된다.
#
# 사전등록 = `TRANSFER_INSTRUCTION_FIDELITY_DESIGN_2026_08_03.md` §5.
set -u
R=/home/woori/workspace_common/boltzmann-attention-pi
D=20260803h
TAG=b4
# ★A와 동일 분할(짝비교 유지·과거 실측 bin-pack 그대로)
G0=task_001,task_002,task_018,task_022,task_025,task_021,task_026,task_035,task_008,task_016,task_014,task_017,task_007,task_012,task_033,task_034
G1=task_003,task_004,task_041,task_027,task_029,task_020,task_028,task_019,task_015,task_010,task_006,task_023,task_040,task_005,task_024,task_032
log(){ echo "[b4 $(date +%m-%d\ %H:%M)] $*"; }

# ★pgrep self-match 회피([[30]]): 이 스크립트 본문이 명령줄에 실리면 자기 패턴에 걸린다
PAT="t2_run_ga""ted.py"
if pgrep -f "$PAT" >/dev/null; then
  log "❌ 중단 — 다른 드라이버가 이미 돈다(중복 실행·GPU 경합 방지)"; exit 1
fi
for P in 8140 8141; do
  curl -s -m 5 http://localhost:$P/v1/models >/dev/null || { log "❌ 중단 — serve $P 무응답"; exit 1; }
done
mkdir -p /home/woori/scratch/logs
log "선행 점검 통과 — 드라이버 유휴 · serve 8140/8141 응답"

one(){ # $1=gpu $2=port $3=tasks
  cd /home/woori/scratch/tau2-bench
  rm -rf data/simulations/bank_${TAG}_gpu$1_$D
  source $R/scripts/distill/tau2/go_stack.sh      # ★정본 스택 — A와 동일(미변경)
  source /home/woori/.openai_key
  # ─── B4 추가분: go_stack.sh는 건드리지 않고 여기서만 얹는다(diff를 ①+④로 유지) ───
  export T2_MATCH_COUNT=1
  export T2_KB_DOCS_DIR=/home/woori/scratch/tau2-bench/data/tau2/domains/banking_knowledge/documents
  echo "[cfg gpu$1] retrieval=${GO_RETRIEVAL:-alltools} user_effort=${GO_USER_EFFORT:-low}" \
       "MATCH_COUNT=$T2_MATCH_COUNT DOCS=$(ls $T2_KB_DOCS_DIR | wc -l)" \
       "GIVE_QUOTE=$T2_GIVE_QUOTE DISPATCH=$T2_DISPATCH_LEDGER WINDOW=$T2_SG_WINDOW_ABSTAIN"
  /home/woori/venvs/seka_env/bin/python -u $R/scripts/distill/tau2/t2_run_gated.py \
    --domain banking_knowledge --retrieval_config "${GO_RETRIEVAL:-alltools}" --gate 1 \
    --agent_model Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8 --agent_base http://localhost:$2/v1 \
    --user_llm openrouter/openai/gpt-5.2 --user_temp 0.0 \
    --user_reasoning_effort "${GO_USER_EFFORT:-low}" \
    --task_ids $3 --num_trials 2 --max_concurrency 2 --max_steps 200 \
    --save_to bank_${TAG}_gpu$1_$D > /home/woori/scratch/logs/${TAG}_gpu$1.log 2>&1
}

persist(){   # ★결과 소실 방지([[30]]): sim 결과는 gitignored scratch에만 있다 — 즉시 영속화
  cd $R && git pull -q --rebase origin facet-rft-2026 2>/dev/null
  for g in 0 1; do
    S=/home/woori/scratch/tau2-bench/data/simulations/bank_${TAG}_gpu${g}_$D/results.json
    [ -f "$S" ] && gzip -c "$S" > $R/reports/facet_rft_2026/sim_results/bank_${TAG}_gpu${g}_$D.results.json.gz
    L=/home/woori/scratch/logs/${TAG}_gpu${g}.log
    [ -f "$L" ] && gzip -c "$L" > $R/reports/facet_rft_2026/sim_results/bank_${TAG}_gpu${g}_$D.log.gz
  done
  cd $R && git add -f reports/facet_rft_2026/sim_results/bank_${TAG}_gpu*_$D.*.gz 2>/dev/null
  git -C $R commit -q -m "Persist B4 run (transfer toll + retrieval boundary) ${TAG}_$D" 2>/dev/null
  git -C $R push -q origin facet-rft-2026 2>/dev/null && log "영속화·push 완료"
}

log "발사 — GPU0=16태스크 · GPU1=16태스크 · 각 nt2 · alltools · MATCH_COUNT=1"
one 0 8140 "$G0" &
P0=$!
one 1 8141 "$G1" &
P1=$!
log "PID gpu0=$P0 gpu1=$P1"
wait $P0; log "gpu0 종료(exit=$?)"
wait $P1; log "gpu1 종료(exit=$?)"
persist
log "체인 종료"
