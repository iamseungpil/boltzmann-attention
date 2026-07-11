#!/bin/bash
# eref_scale_ladder.sh — E-REF 참조-바인딩 scale 사다리 (밤샘·무료·GPU1 단독)
# 각 모델: GPU1:8142 서빙 → clean(P0/P1/P2) + 오염(C 부하·A distractor) 프로브 → kill → 다음.
# 논문 figure: x=scale, y=bind, 선=clean/loaded/distractor. gpt-4.1 user-sim 0(비용 0).
set -u
REPO=/home/woori/workspace_common/boltzmann-attention-pi
T2=$REPO/scripts/distill/tau2
GZ=$REPO/reports/facet_rft_2026/sim_results/comp_retail_t4.results.json.gz
OUT=$REPO/reports/facet_rft_2026/sim_results
VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
P=/home/woori/venvs/seka_env/bin/python
PORT=8142
LOG=/home/woori/scratch/eref_ladder.log
exec > $LOG 2>&1
set -x
date
cd $REPO && git pull --ff-only -q
# GPU1의 기존 8142 서버 종료(유휴 32B)
OLD=$(ps aux | grep "[v]llm serve.*$PORT" | awk '{print $2}' | head -1)
[ -n "$OLD" ] && kill $OLD && sleep 20
pkill -f "[v]llm serve.*$PORT" 2>/dev/null; sleep 10

# 사다리: 소형(full precision·GPU1 단독 적재) + 32B는 GPTQ
MODELS="Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-3B-Instruct Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-14B-Instruct Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8"

serve_wait() {  # $1=model
  CUDA_VISIBLE_DEVICES=1 setsid $VLLM serve "$1" --port $PORT \
    --enable-auto-tool-choice --tool-call-parser hermes --max-model-len 16384 \
    --enforce-eager --gpu-memory-utilization 0.90 </dev/null >/home/woori/scratch/vllm_ladder.log 2>&1 &
  for i in $(seq 1 90); do
    curl -s --max-time 3 localhost:$PORT/v1/models 2>/dev/null | grep -q "$1" && return 0
    sleep 10
  done
  return 1
}

for M in $MODELS; do
  tag=$(echo "$M" | sed 's#.*/##;s/Qwen2.5-//;s/-Instruct//;s/-GPTQ-Int8/-gptq/')
  echo "===== MODEL $M (tag=$tag) ====="; date
  serve_wait "$M" || { echo "SERVE_FAIL $M"; continue; }
  # clean 사다리 P0/P1/P2
  $P $T2/eref_probe.py --gz $GZ --n 36 --cells P0,P1,P2 --base http://localhost:$PORT/v1 \
     --model "$M" --workers 3 --out $OUT/eref_ladder_${tag}_clean.jsonl 2>&1 | tail -8
  # 오염 축 C(부하)·A(distractor)
  $P $T2/eref_probe.py --gz $GZ --n 36 --v2 C:8000,20000 --base http://localhost:$PORT/v1 \
     --model "$M" --workers 3 --out $OUT/eref_ladder_${tag}_loadC.jsonl 2>&1 | tail -6
  $P $T2/eref_probe.py --gz $GZ --n 36 --v2 A:5,10 --base http://localhost:$PORT/v1 \
     --model "$M" --workers 3 --out $OUT/eref_ladder_${tag}_distA.jsonl 2>&1 | tail -6
  # kill 서버
  K=$(ps aux | grep "[v]llm serve.*$PORT" | awk '{print $2}' | head -1)
  [ -n "$K" ] && kill $K; sleep 15
  pkill -f "[v]llm serve.*$PORT" 2>/dev/null; sleep 8
  echo "===== DONE $tag ====="; date
done

# 결과 영속
cd $REPO
git add -f $OUT/eref_ladder_*.jsonl 2>/dev/null
git commit -q -m "persist: E-REF scale ladder (0.5B~32B × clean/loadC/distA·밤샘·무료)" 2>/dev/null
git pull --rebase -q 2>/dev/null; git push -q origin facet-rft-2026 && echo PERSISTED
touch $OUT/../EREF_LADDER_DONE 2>/dev/null; touch /home/woori/scratch/EREF_LADDER_DONE
echo "===== EREF_LADDER_ALLDONE ====="; date
