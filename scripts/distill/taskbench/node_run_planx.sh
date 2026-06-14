#!/bin/bash
# ✅ 2026-06-14 UPDATE (R1b/D5 반영 — 학습부 보류 해제): 구 데이터(ask-user 무·값 memorize)는
#   base ask-user 파국적 망각 → 인자값 날조 → τ² compliant-pass 0.10/0.05 < base 0.17 이었음.
#   ⇒ 이제 §2b 값-randomization(memorize 차단) + §2c D5 대조 ask(fetch-우선 카탈로그 게이트)를
#   파이프라인에 반영. 모델은 "tools= 카탈로그 보고 분기"(getter 有→fetch / 無→ask) 학습 = always-ask 붕괴 방지.
#   woori 실측: v4(값랜덤+ask) τ² 날조 0-90%→0-5%. v5 = v4와 동일 레시피·ask만 D5 게이트(over_ask=0 보장).
#   상세 = COWORKER_REQUEST_TB_SCALE.md · R1B_PROVENANCE_DESIGN_2026_06_14.md §3a/D5 · TB결과 §10.5 R1b.
# node_run_planx.sh — CROSS_BENCH_TRANSFER_PLAN (plan X) P3 on an amlt node.
#   SOPBench(output/ fc-full success rollouts, all teachers×7 domains)
#   + TaskBench(3 domains tool-graph) -> native OpenAI function-calling 궤적
#   -> fc_build_sft unified (per-traj random global alias=R1, QC) -> Qwen2.5-7B TBox LoRA.
# Inputs are all in the public clones (SOPBench output/ ships rollouts; JARVIS taskbench) —
# fully reproducible on a fresh node. Idempotent + trainer --resume = preemption-safe.
set -x
export HF_HUB_CACHE=/scratch/hf_cache
REPO=/scratch/boltzmann-attention
SOP=/scratch/SOPBench
TB=/scratch/JARVIS/taskbench
PY=/scratch/venvs/sop_env/bin/python   # py3.10: torch/peft/transformers + SOPBench env (match)
CV=$REPO/scripts/distill/taskbench
OUT=/scratch/planx
HFREPO=iamseungpil/sopbench-trackb-h200
HF=/scratch/venvs/sop_env/bin/hf
mkdir -p $OUT/fc $OUT/logs

# 0. restore prior state (preemption wipes /scratch): pull planx/ from HF if present
$HF download $HFREPO --repo-type dataset --local-dir /scratch/hf_planx \
  --include "planx/*" >> $OUT/logs/restore.log 2>&1 || true
[ -d /scratch/hf_planx/planx ] && cp -rn /scratch/hf_planx/planx/* $OUT/ 2>/dev/null || true

# 0b. background sync loop: push planx (adapter ckpt + data + logs) every 10min so a
#     mid-training preemption resumes from the last checkpoint, not from scratch
( while true; do
    $HF upload $HFREPO $OUT planx --repo-type dataset \
      --commit-message "planx sync $(date -u +%FT%TZ)" >> $OUT/logs/sync.log 2>&1
    sleep 600
  done ) &
SYNC_PID=$!

# 1. SOPBench converters: every fc-full success rollout (all teachers), 7 domains
if [ ! -f $OUT/fc/.sop_done ]; then
  for d in bank dmv healthcare hotel library online_market university; do
    for f in $SOP/output/$d/ast_*mode_fc-dep_full*tool_full-shuffle_False.json; do
      [ -f "$f" ] || continue
      tag=$(basename "$f" .json | tr -c 'A-Za-z0-9' '_')
      $PY $CV/fc_convert_sopbench.py --rollout "$f" --sopbench_root $SOP --domain $d \
        --out $OUT/fc/sop_${d}__${tag}.jsonl >> $OUT/logs/conv_sop.log 2>&1
    done
  done
  touch $OUT/fc/.sop_done
fi

# 2. TaskBench converters: 3 domains
if [ ! -f $OUT/fc/.tb_done ]; then
  for d in data_huggingface data_multimedia data_dailylifeapis; do
    $PY $CV/fc_convert_taskbench.py --data $TB/$d/data.json --tool_desc $TB/$d/tool_desc.json \
      --out $OUT/fc/tb_${d}.jsonl >> $OUT/logs/conv_tb.log 2>&1
  done
  touch $OUT/fc/.tb_done
fi
wc -l $OUT/fc/sop_*.jsonl | tail -1; wc -l $OUT/fc/tb_*.jsonl | tail -1

# 2b. R1b 값-randomization: user-제공 식별값을 포맷-보존 랜덤토큰으로 일관치환 → memorize 차단,
#     컨텍스트서 복사 강제(R1b). 모든 SOPBench fc 궤적 concat 후 일괄 (seed 고정).
if [ ! -f $OUT/fc/.rand_done ]; then
  cat $OUT/fc/sop_*.jsonl > $OUT/fc/sop_all.jsonl
  $PY $CV/fc_value_randomize.py --in $OUT/fc/sop_all.jsonl --out $OUT/fc/sop_rand.jsonl --seed 42 \
    >> $OUT/logs/build.log 2>&1
  touch $OUT/fc/.rand_done
fi

# 2c. D5 대조 ask: ask/fetch 분기를 카탈로그-결정론 게이트로(provenance + getter_map). 휴리스틱 폐기.
#     getter_map = repo 박제본($CV/getter_map.json, 결정론 산물). frac 0.40 = v4 ask volume 일치(깨끗 A/B).
if [ ! -f $OUT/fc/.d5_done ]; then
  $PY $CV/fc_d5_contrastive.py --in $OUT/fc/sop_rand.jsonl --getter_map $CV/getter_map.json \
    --out_ask $OUT/fc/sop_d5_ask.jsonl --frac 0.40 --seed 42 >> $OUT/logs/build.log 2>&1
  touch $OUT/fc/.d5_done
fi

# 3. unified builder (alias=R1 on; balance benches). 입력 = sop_rand(fetch+upfront) + sop_d5_ask(게이트 ask) + tb.
#    fetch/upfront 예시 = sop_rand 자체(자연 fetch-then-use). ask 예시 = D5 게이트(over_ask=0). 대조는 데이터셋 레벨.
if [ ! -f $OUT/planx_sft.jsonl ]; then
  $PY $CV/fc_build_sft.py --inputs $OUT/fc/sop_rand.jsonl $OUT/fc/sop_d5_ask.jsonl $OUT/fc/tb_*.jsonl \
    --out $OUT/planx_sft.jsonl --max_per_bench 7000 --seed 42 >> $OUT/logs/build.log 2>&1
fi
cat $OUT/logs/build.log

# 4. Qwen2.5-7B TBox LoRA (native FC; trainer renders tools+tool_calls, assistant-only mask)
$HF download Qwen/Qwen2.5-7B-Instruct >> $OUT/logs/hfdl.log 2>&1
if [ ! -f $OUT/planx_train_done ]; then
  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY \
    $REPO/scripts/distill/lora_train_chat_toolcall.py \
    --base-model Qwen/Qwen2.5-7B-Instruct --device cuda:0 \
    --train-jsonl $OUT/planx_sft.jsonl --out-dir $OUT/planx_tbox_7b \
    --max-seq-len 6144 --epochs 2 --lora-r 16 --lora-alpha 32 --lr 1e-4 \
    --val-frac 0.02 --skip-overlong --save-every 100 --resume \
    > $OUT/logs/train.log 2>&1 && touch $OUT/planx_train_done
fi
tail -3 $OUT/logs/train.log

# 5. final sync (stop the loop, one authoritative push)
kill $SYNC_PID 2>/dev/null || true
$HF upload $HFREPO $OUT planx --repo-type dataset \
  --commit-message "planx FINAL $(date -u +%FT%TZ)" >> $OUT/logs/sync.log 2>&1 || true
echo "PLANX_DONE $(date)"
