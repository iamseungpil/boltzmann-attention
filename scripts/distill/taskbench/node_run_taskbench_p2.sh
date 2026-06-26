#!/bin/bash
# node_run_taskbench_p2.sh — 신판 P2 (요청서 v4 §7): 대형-모델 결정론-leg 2×2, 추론-only.
# arms (전부 MM sub500 동일 첫-500; 사전예측 §7 표에 동결, 2026-06-12):
#   P2a-1 lodo_mm_32b(SFT)+guided  TP2 GPU0,1  tag=tb_lodo_mm_32b_guided    예측 +3~5 (어휘분 회복)
#   P2a-2 base-32B+guided          TP2 GPU0,1  tag=qwen25_32b_guided        예측 ≈0 (통제)
#   P2c   base-32B+guided+slim1    같은 서버    tag=qwen25_32b_guided_slim1  desc 제거, 예측 손실<3.1
#   P2b-1 base-72B+guided          TP4         tag=qwen25_72b_guided        예측 +0~0.5
#   P2b-2 base-235B-INT4+guided    TP4         tag=qwen3_235b_a22b_int4_guided  음성 통제 ≈0
# guided = TB_GUIDED=1 + tb_guided_schema(enum tool-name slots); non-thinking은 기존
# tbpatch(chat_template_kwargs)가 항상 적용 — 두 패치는 앵커가 달라 공존.
# 멱등: per-arm done-marker + inference.py 자연 resume. preds는 symlink로 HF sync 포함.
set -x
export HF_HUB_CACHE=/scratch/hf_cache
TB=/scratch/JARVIS/taskbench
R=/scratch/boltzmann-attention
VLLM=/scratch/venvs/sop_env/bin/vllm
HF=/scratch/venvs/sop_env/bin/hf
IP=/scratch/venvs/tb_env/bin/python
PORT=${TB_PORT:-8500}
OUT=/scratch/taskbench_runs
# v2 markers: 1차(p2_done)는 vllm 0.10.2가 structured_outputs를 조용히 무시해 무효
# (사실상 unguided) — 새 이름으로 stale-marker 복원 차단
DONE=$OUT/p2v2_done
ADAPTER=$OUT/sft/qwen32b_tb_lodo_mm
SCHEMA=$OUT/p2_guided_mm_schema.json
mkdir -p $DONE /scratch/logs

kill_gpus() {
  for g in $(echo $1 | tr , ' '); do
    for p in $(nvidia-smi --id=$g --query-compute-apps=pid --format=csv,noheader); do
      kill -9 $p 2>/dev/null || true
    done
  done
  sleep 15
}

# 0. patches + schema + slim1 variant (desc 제거 tool_desc) + preds symlink
$IP $R/scripts/distill/taskbench/tb_guided_patch.py $TB/inference.py || exit 1
# vllm 0.10.2 fallback: per-request 키는 guided_json — structured_outputs(0.11+)는
# protocol.py가 "ignored" 경고만 내고 무시 (P2 1차 무효의 원인, vllm 로그로 확인)
$IP - <<'EOF'
import re
p = "/scratch/JARVIS/taskbench/inference.py"
s = open(p).read()
s2 = re.sub(
    r'\{"structured_outputs":\s*\{"json":\s*(_tb_json\.load\(open\(_tb_os\.environ\["TB_GUIDED_SCHEMA"\]\)\))\}\}',
    r'{"guided_json": \1}', s)
if s2 != s:
    open(p, "w").write(s2); print("[fallback] structured_outputs -> guided_json (vllm 0.10.x)")
elif '"guided_json"' in s:
    print("[fallback] already guided_json")
else:
    raise SystemExit("[fallback] FAIL: helper pattern not found")
EOF
[ $? -eq 0 ] || exit 1
$IP $R/scripts/distill/taskbench/tb_guided_schema.py \
  --tool_desc $TB/data_multimedia/tool_desc.json --dep resource --out $SCHEMA || exit 1
if [ ! -f $TB/data_multimedia_sub500_slim1/tool_desc.json ]; then
  $IP - <<'EOF'
import json, shutil, os
TB = "/scratch/JARVIS/taskbench"
src, dst = f"{TB}/data_multimedia_sub500", f"{TB}/data_multimedia_sub500_slim1"
td = json.load(open(f"{TB}/data_multimedia/tool_desc.json"))
KEEP = {"id", "input-type", "output-type", "parameters"}   # P2c = desc 제거 (Track-A slim1)
os.makedirs(dst, exist_ok=True)
for f in ["user_requests.json", "graph_desc.json"]:
    shutil.copy(f"{src}/{f}", f"{dst}/{f}")
json.dump({"nodes": [{k: v for k, v in t.items() if k in KEEP} for t in td["nodes"]]},
          open(f"{dst}/tool_desc.json", "w"))
full = sum(len(json.dumps(t)) for t in td["nodes"])
s1 = sum(len(json.dumps({k: v for k, v in t.items() if k in KEEP})) for t in td["nodes"])
print(f"[promptsize chars] full={full} slim1={s1} ({100*s1/full:.0f}%)")
EOF
fi
mkdir -p $OUT/preds/data_multimedia_sub500_slim1
[ -e $TB/data_multimedia_sub500_slim1/predictions ] || \
  ln -s $OUT/preds/data_multimedia_sub500_slim1 $TB/data_multimedia_sub500_slim1/predictions

serve() {  # serve GPUS TPN PROBE_NAME <vllm serve args...>
  local gpus=$1 tpn=$2 probe=$3; shift 3
  kill_gpus $gpus
  CUDA_VISIBLE_DEVICES=$gpus setsid nohup $VLLM serve "$@" \
    --port $PORT --tensor-parallel-size $tpn \
    --max-model-len 8192 --gpu-memory-utilization 0.90 \
    > /scratch/logs/vllm_p2_${probe}.log 2>&1 &
  for i in $(seq 1 180); do
    curl -s localhost:$PORT/v1/models | grep -q "\"$probe\"" && return 0; sleep 10
  done
  echo "SERVE_FAIL_$probe"; return 1
}

guided_run() {  # guided_run TAG DATA_DIR
  local tag=$1 ddir=$2
  # 바인딩 게이트: guided_json 1콜의 content가 JSON으로 강제되는지 확인 — 안 묶이면
  # 즉시 중단 (1차처럼 조용한 unguided 재실행을 결과로 오인하는 사고 방지)
  SAN=$(curl -s -m 300 localhost:$PORT/v1/chat/completions -H "Content-Type: application/json" -d "{
    \"model\": \"$tag\", \"max_tokens\": 128,
    \"messages\": [{\"role\": \"user\", \"content\": \"emit a tiny plan\"}],
    \"guided_json\": $(cat $SCHEMA)}")
  echo "$SAN" | head -c 200; echo
  echo "$SAN" | grep -q '"content":"{' || { echo "GUIDED_NOT_BINDING_$tag"; return 1; }
  rm -f $TB/$ddir/predictions/${tag}.json   # 1차 unguided 잔재 제거 (강제 재추론)
  (cd $TB && TB_GUIDED=1 TB_GUIDED_SCHEMA=$SCHEMA $IP inference.py --data_dir $ddir \
    --api_addr localhost --api_port $PORT --api_key dummy --llm $tag --multiworker 8 \
    --dependency_type resource) || return 1
  $IP $R/scripts/distill/taskbench/tb_build_eval.py --tb_dir $TB --domain data_multimedia \
    --pred_file $TB/$ddir/predictions/${tag}.json --dst $OUT/${ddir}_eval_${tag} --llm $tag
}

# ---- P2a-1: SFT 어댑터 + guided (어댑터는 node_resume의 HF 복원으로 존재해야 함) ----
if [ ! -f $DONE/p2a1 ]; then
  [ -f $ADAPTER/adapter_model.safetensors ] || { echo "ADAPTER_MISSING $ADAPTER (HF 복원 확인)"; exit 1; }
  $HF download Qwen/Qwen2.5-32B-Instruct >> /scratch/logs/hfdl_p2.log 2>&1
  serve 0,1 2 tb_lodo_mm_32b_guided Qwen/Qwen2.5-32B-Instruct \
    --served-model-name qwen25_32b_p2base --enable-lora --lora-modules tb_lodo_mm_32b_guided=$ADAPTER \
  && guided_run tb_lodo_mm_32b_guided data_multimedia_sub500 && touch $DONE/p2a1
fi

# ---- P2a-2 + P2c: base 32B 한 서버, 두 served-name (sub500 / slim1) ----
if [ ! -f $DONE/p2a2 ] || [ ! -f $DONE/p2c ]; then
  serve 0,1 2 qwen25_32b_guided Qwen/Qwen2.5-32B-Instruct \
    --served-model-name qwen25_32b_guided qwen25_32b_guided_slim1 || exit 1
  [ -f $DONE/p2a2 ] || { guided_run qwen25_32b_guided data_multimedia_sub500 && touch $DONE/p2a2; }
  [ -f $DONE/p2c ]  || { guided_run qwen25_32b_guided_slim1 data_multimedia_sub500_slim1 && touch $DONE/p2c; }
fi

# ---- P2b-1: 72B TP4 ----
if [ ! -f $DONE/p2b1 ]; then
  $HF download Qwen/Qwen2.5-72B-Instruct >> /scratch/logs/hfdl_p2.log 2>&1
  serve 0,1,2,3 4 qwen25_72b_guided Qwen/Qwen2.5-72B-Instruct \
    --served-model-name qwen25_72b_guided \
  && guided_run qwen25_72b_guided data_multimedia_sub500 && touch $DONE/p2b1
fi

# ---- P2b-2: 235B-INT4 TP4 (음성 통제) ----
if [ ! -f $DONE/p2b2 ]; then
  $HF download Qwen/Qwen3-235B-A22B-GPTQ-Int4 >> /scratch/logs/hfdl_p2.log 2>&1
  serve 0,1,2,3 4 qwen3_235b_a22b_int4_guided Qwen/Qwen3-235B-A22B-GPTQ-Int4 \
    --served-model-name qwen3_235b_a22b_int4_guided \
  && guided_run qwen3_235b_a22b_int4_guided data_multimedia_sub500 && touch $DONE/p2b2
fi

kill_gpus 0,1,2,3
ls $DONE
echo "P2_ALL_DONE $(date)"
