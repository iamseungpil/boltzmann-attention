#!/bin/bash
# 밤샘 무인 (2026-06-22): maxprompt(GPU1) 끝나면 GPU1서 airline 전이 eval 자동 시작.
# retail eval(GPU0)은 독립 진행. 끝나면 통합 요약 + OVERNIGHT_C4_DONE 마커. 함정=healthcheck·sleep폴.
# 선행조건: c4_maxprompt_sweep(GPU1)·c4_fetchfirst_eval retail(GPU0) 이미 setsid 가동 중.
set -u
REPO=/home/woori/workspace_common/boltzmann-attention-pi; T2=$REPO/scripts/distill/tau2
S=/home/woori/scratch
LOG=$S/overnight_c4.log; exec > $LOG 2>&1; set -x; date

# 1. maxprompt sweep 완료 대기 (GPU1 free) — 최대 ~75min
for i in $(seq 1 75); do
  grep -q C4_MAXPROMPT_DONE $S/c4_maxprompt_sweep.log 2>/dev/null && { echo "[ok] maxprompt done"; break; }
  sleep 60
done
# 잔여 GPU1 proc 정리(혹시 안 죽었으면)
for p in $(nvidia-smi --id=1 --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 8

# 2. airline 전이 eval on GPU1 (도메인-일반 adapter+prompts·ABox-swap·c4_fetchfirst_eval가 cells 인라인 실행→블록)
echo "[launch] airline transfer eval"
bash $T2/c4_fetchfirst_eval.sh 1 8379 airline 50
echo "[ok] airline eval returned (see c4_ff_eval_airline.log)"

# 3. retail eval(GPU0) 완료도 대기 — 최대 ~120min
for i in $(seq 1 120); do
  grep -q C4FF_EVAL_DONE $S/c4_ff_eval_retail.log 2>/dev/null && { echo "[ok] retail eval done"; break; }
  sleep 60
done

# 4. 통합 요약 (아침 수거)
echo "================ OVERNIGHT C4 SUMMARY ================"
echo "--- A) maxprompt (프롬프트 한계·schema_copy) ---"
grep -h C4MP_ROW $S/c4_maxprompt_sweep.log 2>/dev/null
echo "--- B) retail 곡선 (base/prompt/skill/fewshot-dg/fewshot-retail/scaffold-deny/scaffold-perform/learn/learn-pure) ---"
grep -hE "C4FFROW retail|C4FF_MECH retail" $S/c4_ff_eval_retail.log 2>/dev/null
echo "--- C) airline 전이 곡선 (동일 adapter/prompts·ABox-swap) ---"
grep -hE "C4FFROW airline|C4FF_MECH airline" $S/c4_ff_eval_airline.log 2>/dev/null
echo "--- 핵심 질문: learn/fewshot-dg 의 schema_copy → 0 인가? (prior 닫혔나) ---"
date; echo OVERNIGHT_C4_DONE
