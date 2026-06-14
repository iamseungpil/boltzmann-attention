#!/bin/bash
# SOPBench in-distribution eval (plan-X 어댑터가 학습벤치 자체서 점수 떨어졌나 = 사용자 진단).
# run_simulation.py + SOPBENCH_VLLM_BASE_URL(pre-served vLLM) 경로. native-FC(tool_call_mode=fc).
# 파괴적 op는 파일 안(rr 가드 회피). git pull로 전송.
#
# Usage: sopbench_indist_eval.sh <adapter_dir> <lora_name> <gpu> <port> <domain> <num_tasks>
#   예: bash .../sopbench_indist_eval.sh /home/woori/scratch/v6_eval_s2550 v6tbox 1 8370 online_market 10
set -u
ADAPTER=${1:?adapter}; NAME=${2:?lora name}; GPU=${3:?gpu}; PORT=${4:?port}; DOM=${5:?domain}; NT=${6:-10}
SOP=/home/woori/scratch/SOPBench
PY=/home/woori/venvs/seka_env/bin/python
VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
S=/home/woori/scratch
LOG=$S/sopindist_${NAME}.log
exec > $LOG 2>&1; set -x; date

# 0. OSS_MODELS에 lora 엔트리 idempotent 추가 (요청 모델명=lora 이름이 되도록)
$PY - <<PYEOF
import re
p="$SOP/swarm/constants.py"
s=open(p).read()
if '"$NAME"' not in s:
    s=s.replace("OSS_MODELS = {", 'OSS_MODELS = {\n    "$NAME": "$NAME",', 1)
    open(p,"w").write(s); print("[oss] added $NAME")
else: print("[oss] $NAME present")
PYEOF

# 1. serve base + lora on GPU
for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 4
CUDA_VISIBLE_DEVICES=$GPU setsid nohup $VLLM serve Qwen/Qwen2.5-7B-Instruct --port $PORT \
  --enable-auto-tool-choice --tool-call-parser hermes --max-model-len 16384 \
  --enable-lora --lora-modules ${NAME}=$ADAPTER --max-lora-rank 32 > $S/vllm_${NAME}.log 2>&1 &
ok=0; for i in $(seq 1 60); do curl -s localhost:$PORT/v1/models 2>/dev/null | grep -q "$NAME" && ok=1 && break; sleep 10; done
[ $ok = 1 ] || { echo SERVE_FAIL; tail -25 $S/vllm_${NAME}.log; exit 1; }
echo "[sanity] served models:"; curl -s localhost:$PORT/v1/models | $PY -c "import sys,json;print([m['id'] for m in json.load(sys.stdin)['data']])"

# 2. run_simulation (in-dist held-out, native FC, 정보-dump user=학습분포 일치)
export SOPBENCH_VLLM_BASE_URL=http://localhost:$PORT/v1
cd $SOP
$PY run_simulation.py --assistant_model $NAME --tool_call_mode fc --domain $DOM \
  --num_tasks $NT --max_num_turns 20 --max_num_actions 10 2>&1 | tail -30

# 3. free GPU
for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done
echo "=== output 위치 (파싱용) ==="
ls -t $SOP/output/$DOM/ast_${NAME}* 2>/dev/null | head -3
date; echo SOPINDIST_DONE_${NAME}
