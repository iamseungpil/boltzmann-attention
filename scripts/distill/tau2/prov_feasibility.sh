#!/bin/bash
# bad_words plumbing 최소 검증 (긴 eval 전 feasibility 게이트):
# litellm completion(extra_body={"bad_words":[...]}) → vLLM 가 실제로 #W0000000 생성을 막나.
# Usage: prov_feasibility.sh <adapter> <name> <gpu> <port>
set -u
ADAPTER=${1:?}; NAME=${2:?}; GPU=${3:?}; PORT=${4:?}
PY=/home/woori/venvs/seka_env/bin/python
VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
S=/home/woori/scratch
LOG=$S/feas.log
exec > $LOG 2>&1; set -x; date

for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 3
CUDA_VISIBLE_DEVICES=$GPU setsid nohup $VLLM serve Qwen/Qwen2.5-7B-Instruct --port $PORT \
  --enable-auto-tool-choice --tool-call-parser hermes --max-model-len 8192 \
  --enable-lora --lora-modules ${NAME}=$ADAPTER --max-lora-rank 32 > $S/vllm_feas.log 2>&1 &
ok=0; for i in $(seq 1 60); do curl -s localhost:$PORT/v1/models 2>/dev/null | grep -q "$NAME" && ok=1 && break; sleep 10; done
[ $ok = 1 ] || { echo SERVE_FAIL; tail -20 $S/vllm_feas.log; exit 1; }

set +x
$PY << PYEOF
import litellm
litellm.drop_params = True
base="http://localhost:$PORT/v1"
msgs=[{"role":"user","content":"Output exactly this order id and nothing else: #W0000000"}]
def run(extra):
    r=litellm.completion(model="openai/$NAME", api_base=base, api_key="dummy",
                         messages=msgs, temperature=0.0, max_tokens=20, **extra)
    return r.choices[0].message.content
a=run({})
print("=== WITHOUT bad_words ===")
print(repr(a))
print("contains #W0000000:", "#W0000000" in (a or ""))
try:
    b=run({"extra_body":{"bad_words":["#W0000000"]}})
    print("=== WITH bad_words ===")
    print(repr(b))
    print("contains #W0000000:", "#W0000000" in (b or ""))
    print("FEASIBLE:", ("#W0000000" in (a or "")) and ("#W0000000" not in (b or "")))
except Exception as e:
    print("WITH bad_words ERROR:", type(e).__name__, str(e)[:200])
PYEOF
for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done
echo FEAS_DONE; date
