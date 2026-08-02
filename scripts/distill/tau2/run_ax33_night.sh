#!/bin/bash
# ★AX33 밤샘 런 — front32 × nt2 × **alltools** (2026-08-03·사용자 지시)
#
# 목적 2개:
#  ⑴ AX33 처방 스택(P1·P2/P10·P5·P8·P9·P11·P12·P13 + §4-1/§4-2)의 라이브 효과 측정
#  ⑵ **검색 설정 귀속** — 같은 front32를 bm25(기존 ax32/qp32 원장)와 alltools로 비교
#     ⚠교락 2개를 판정문에 반드시 명시: ⓐ레버 스택도 달라졌다(AX33 신규 6종) ⓑ v1.0.1 환경 변경
#       (#403 거래 최신순·#388 문서 모순 제거 — dispute 가족 019/020/027/028 직격). 순수 검색-효과
#       귀속은 이 런만으로 성립하지 않는다(정직 표기 의무·[[08]]).
#
# 구성: front32를 GPU0/GPU1에 16+16으로 갈라 **동시** 실행(벽시계 반감) · 각 16 × nt2 = 32 sim/GPU.
# ★2026-08-03 재분배(사용자 지시): 과거 ax32 런의 **태스크별 실측 소요시간**으로 bin-pack →
#   두 GPU 종료 시각을 맞춘다(기존 분할은 405분 vs 473분 = 14.5% 불균형 → 0.2%).
#   짝비교는 **태스크 id 기준**이라 GPU 배정이 바뀌어도 유지된다(같은 32개 집합).
#   001·002=GPU0 / 003·004=GPU1은 사용자 지시로 고정하고 **맨 앞**에 둬 먼저 확인된다.
set -u
R=/home/woori/workspace_common/boltzmann-attention-pi
D=20260803b
TAG=ax33n
G0=task_001,task_002,task_018,task_022,task_025,task_021,task_026,task_035,task_008,task_016,task_014,task_017,task_007,task_012,task_033,task_034
G1=task_003,task_004,task_041,task_027,task_029,task_020,task_028,task_019,task_015,task_010,task_006,task_023,task_040,task_005,task_024,task_032
log(){ echo "[ax33n $(date +%m-%d\ %H:%M)] $*"; }

if pgrep -f "[t]2_run_gated" >/dev/null; then
  log "❌ 중단 — 다른 드라이버가 이미 돈다(중복 실행·GPU 경합 방지·[[30]] 함정)"; exit 1
fi
for P in 8140 8141; do
  curl -s -m 5 http://localhost:$P/v1/models >/dev/null || { log "❌ 중단 — serve $P 무응답"; exit 1; }
done
mkdir -p /home/woori/scratch/logs
log "선행 점검 통과 — 드라이버 유휴 · serve 8140/8141 응답"

one(){ # $1=gpu(0/1) $2=port $3=tasks
  cd /home/woori/scratch/tau2-bench
  rm -rf data/simulations/bank_${TAG}_gpu$1_$D
  source $R/scripts/distill/tau2/go_stack.sh      # ★정본 스택(alltools·user low·AX33 플래그 포함)
  source /home/woori/.openai_key                  # alltools = OpenAI 임베딩 필요
  echo "[cfg gpu$1] retrieval=${GO_RETRIEVAL:-alltools} user_effort=${GO_USER_EFFORT:-low}" \
       "GIVE_QUOTE=$T2_GIVE_QUOTE DISPATCH=$T2_DISPATCH_LEDGER WINDOW=$T2_SG_WINDOW_ABSTAIN" \
       "KBNOHIT=$T2_KB_NOHIT_SURFACE WAG=$T2_WRITE_ARG_GROUND"
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
  git -C $R commit -q -m "Persist AX33 night run (front32 x nt2, alltools) ${TAG}_$D" 2>/dev/null
  git -C $R push -q origin facet-rft-2026 2>/dev/null && log "영속화·push 완료"
}

log "발사 — GPU0=16태스크 · GPU1=16태스크 · 각 nt2 · alltools"
one 0 8140 "$G0" &
P0=$!
one 1 8141 "$G1" &
P1=$!
log "PID gpu0=$P0 gpu1=$P1"
wait $P0; log "gpu0 종료(exit=$?)"
wait $P1; log "gpu1 종료(exit=$?)"
persist
log "체인 종료"
