#!/bin/bash
# qwen7b_solo_lite 의 중간 checkpoint들(stepN) + final 을 한 vLLM에 동시 로드 → 각각 빠른 retail e2e.
# 목적 = 어느 학습 시점이 base 범용 named-tool 호출을 보존하며 resolve_selection을 부르나(과적합 전 지점).
# 판정/ckpt = (pass^1) + (resolve_selection assistant 호출수) + (op/attr/set bleed수) + (캐논ID 환각).
# Usage: eval_solo_ckpts.sh <gpu> <port> [num_tasks] [adapter_dir]
set -u
GPU=${1:?gpu}; PORT=${2:?port}; NT=${3:-12}
REPO=/home/woori/workspace_common/boltzmann-attention-pi
T2=$REPO/scripts/distill/tau2
PY=/home/woori/venvs/seka_env/bin/python
VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
BASE=Qwen/Qwen2.5-7B-Instruct
AD=${4:-/home/woori/scratch/adapters/qwen7b_solo_lite}
ADTAG=$(basename $AD)
S=/home/woori/scratch
LOG=$S/eval_solo_ckpts_${ADTAG}.log
exec > $LOG 2>&1; set -x; date

# --lora-modules 목록 구성: 각 stepN snapshot + final(=AD 루트)
MODS=""; TAGS=""
for d in $(ls -d $AD/step* 2>/dev/null | sort -V); do
  nm=$(basename $d); MODS="$MODS ${nm}=${d}"; TAGS="$TAGS $nm"
done
[ -f $AD/adapter_model.safetensors ] && { MODS="$MODS final=${AD}"; TAGS="$TAGS final"; }
echo "TAGS=$TAGS"
[ -n "$TAGS" ] || { echo NO_CKPTS; exit 1; }

for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 4
CUDA_VISIBLE_DEVICES=$GPU setsid nohup $VLLM serve $BASE --port $PORT \
  --enable-auto-tool-choice --tool-call-parser hermes --max-model-len 16384 \
  --enable-lora --max-lora-rank 64 --max-loras 6 --lora-modules $MODS \
  > $S/vllm_eval_ckpts.log 2>&1 &
ok=0; for i in $(seq 1 80); do curl -s localhost:$PORT/v1/models 2>/dev/null | grep -q "final\|step" && ok=1 && break; sleep 10; done
[ $ok = 1 ] || { echo SERVE_FAIL; tail -30 $S/vllm_eval_ckpts.log; exit 1; }

set +x; source /home/woori/.openrouter_key; export SSL_CERT_FILE=$($PY -c "import certifi;print(certifi.where())"); set -x
cd $S/tau2-bench; export PYTHONPATH=src:$T2

for tag in $TAGS; do
  set +x
  echo "############ CKPT=${ADTAG}:$tag ############"
  rm -rf data/simulations/${ADTAG}_$tag
  $PY $T2/t2_run_gated.py --gate 1 --resolve 1 --domain retail --num_trials 1 --num_tasks $NT \
    --agent_model $tag --agent_base http://localhost:$PORT/v1 \
    --user_llm "openrouter/openai/gpt-4.1" --user_temp 0.0 --save_to ${ADTAG}_$tag \
    > $S/t2_${ADTAG}_$tag.log 2>&1 || echo "FAIL $tag"
  grep -E "RESULT|pass1" $S/t2_${ADTAG}_$tag.log | tail -1
  $PY - "$tag" "$ADTAG" <<'PY'
import json,sys,collections
tag=sys.argv[1]; adtag=sys.argv[2]
p=f"/home/woori/scratch/tau2-bench/data/simulations/{adtag}_{tag}/results.json"
try: d=json.load(open(p))
except Exception as e: print("  NA",e); raise SystemExit
sims=d.get("simulations",[]); rew=[(s.get("reward_info") or {}).get("reward") for s in sims]
rew=[r for r in rew if r is not None]
resolve_calls=0; bleed=0; canon=0; names=collections.Counter()
CANON={"sara_doe_496","#W0000000","6086499569"}
OPERAND={"op","attr","among","dir","set"}
for s in sims:
  for m in s.get("messages",[]):
    if m.get("role")!="assistant": continue
    for tc in (m.get("tool_calls") or []):
      nm=tc.get("name"); names[nm]+=1
      args=tc.get("arguments") or {}
      if isinstance(args,str):
        try: args=json.loads(args)
        except Exception: args={}
      if nm=="resolve_selection": resolve_calls+=1
      elif isinstance(args,dict) and (set(args)&OPERAND): bleed+=1
      if any(str(v) in CANON for v in (args.values() if isinstance(args,dict) else [])): canon+=1
print(f"  pass1={sum(r>=1 for r in rew)}/{len(rew)} resolve_calls={resolve_calls} "
      f"operand_bleed_nonresolve={bleed} canon_id_halluc={canon}")
PY
  set -x
done

for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done
echo "=== EVAL_CKPTS_DONE ==="; date
