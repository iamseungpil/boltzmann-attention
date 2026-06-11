#!/bin/bash
# node_run_taskbench_p1.sh — TB_SCALE v3 P1 (TRAIN node, GPU2,3; SOPBench #1 SFT owns GPU0,1).
#   step 0 (v3 필수): 32B base census on MM sub500 -> Δedge PREREG frozen + pushed to HF
#                     BEFORE any training (tb_census.py + tb_p1_prereg.py)
#   step 1: gold-SFT jsonl = HF(400/1000/dag-all) + daily(400/1000/dag-all), seed 42 -> 3869
#   step 2: LoRA r16/a32 2ep seq6144 lr1e-4 val2% @ Qwen2.5-32B, --device auto over the
#           2 GPUs (bf16 32B > 80GB single H100), step-ckpt --save-every --resume
#   step 3: eval — held-out MM sub500 -> in-domain HF/daily sub500 -> MM full (보충)
# Preemption-safe: every step has a done-guard; trainer resumes from ckpt_state.pt;
# inference.py skips already-predicted ids.
set -x
export HF_HUB_CACHE=/scratch/hf_cache
TB=/scratch/JARVIS/taskbench
R=/scratch/boltzmann-attention
VLLM=/scratch/venvs/sop_env/bin/vllm
HF=/scratch/venvs/sop_env/bin/hf
IP=/scratch/venvs/tb_env/bin/python
PYT=/scratch/venvs/sop_env/bin/python
PORT=${TB_PORT:-8400}
GPUS=${P1_GPUS:-2,3}
OUT=/scratch/taskbench_runs
ADAPTER=$OUT/sft/qwen32b_tb_lodo_mm
HFREPO=iamseungpil/sopbench-trackb-h200
BASE=Qwen/Qwen2.5-32B-Instruct
mkdir -p $OUT/sft /scratch/logs

kill_port_vllm() {
  pkill -9 -f "vllm serve.*--port $PORT"
  for i in $(seq 1 24); do pgrep -f "vllm serve.*--port $PORT" >/dev/null || break; sleep 5; done
  sleep 10
}

serve_and_wait() {  # serve_and_wait NAME_GREP <vllm serve args...>
  local probe=$1; shift
  kill_port_vllm
  CUDA_VISIBLE_DEVICES=$GPUS setsid nohup $VLLM serve "$@" \
    > /scratch/logs/vllm_p1_${probe}.log 2>&1 &
  for i in $(seq 1 180); do
    curl -s localhost:$PORT/v1/models | grep -q "\"$probe\"" && return 0; sleep 10
  done
  echo "SERVE_FAIL_$probe"; return 1
}

infer_eval() {  # infer_eval DATA_DIR_BASENAME DOMAIN TAG DEP [EVAL_DST_SUFFIX]
  (cd $TB && $IP inference.py --data_dir $1 --api_addr localhost --api_port $PORT \
    --api_key dummy --llm $3 --multiworker 8 --dependency_type $4)
  $IP $R/scripts/distill/taskbench/tb_build_eval.py --tb_dir $TB --domain $2 \
    --pred_file $TB/$1/predictions/$3.json --dst $OUT/${1}_eval_$3 --llm $3
}

# ---------- step 0: base census + PREREG (학습 착수 전 박제, v3 필수) ----------
if [ ! -f $OUT/p1_census_prereg.json ]; then
  $HF download $BASE >> /scratch/logs/hfdl_qwen25_32b.log 2>&1
  serve_and_wait qwen25_32b $BASE --port $PORT --served-model-name qwen25_32b \
    --tensor-parallel-size 2 --max-model-len 8192 --gpu-memory-utilization 0.90 || exit 1
  infer_eval data_multimedia_sub500 data_multimedia qwen25_32b resource
  $IP $R/scripts/distill/taskbench/tb_census.py \
    --dir_a $TB/data_multimedia_sub500 --llm_a qwen25_32b \
    --dir_b $TB/data_multimedia_sub500 --llm_b qwen25_32b \
    --tool_desc $TB/data_multimedia/tool_desc.json --dep resource \
    --out $OUT/p1_census_32b_base.md
  $IP $R/scripts/distill/taskbench/tb_p1_prereg.py \
    $OUT/p1_census_32b_base.md $OUT/p1_census_prereg.json
  # freeze: push the prereg to HF immediately (sync loop would also catch it, but
  # the timestamped commit IS the 박제)
  $HF upload $HFREPO $OUT/p1_census_prereg.json train/taskbench_runs/p1_census_prereg.json \
    --repo-type dataset --commit-message "P1 step-0 PREREG (before training) $(date -u +%FT%TZ)"
  kill_port_vllm
fi

# ---------- step 1: SFT data (seed 42 -> byte-identical to Track A) ----------
TRAIN=$OUT/train_lodo_mm.jsonl
if [ ! -f $TRAIN ]; then
  $IP $R/scripts/distill/taskbench/tb_build_sft.py --tb_dir $TB --domain data_huggingface \
    --out $OUT/sft_data_huggingface.jsonl --n_single 400 --n_chain 1000 --n_dag 0 || exit 1
  $IP $R/scripts/distill/taskbench/tb_build_sft.py --tb_dir $TB --domain data_dailylifeapis \
    --out $OUT/sft_data_dailylifeapis.jsonl --n_single 400 --n_chain 1000 --n_dag 0 || exit 1
  cat $OUT/sft_data_huggingface.jsonl $OUT/sft_data_dailylifeapis.jsonl > $TRAIN
fi
wc -l $TRAIN   # expect 3869

# ---------- step 2: train (recipe locked to 7B control: r16/a32/2ep/seq6144/lr1e-4/val2%) ----------
if [ ! -f $OUT/p1_train_done ]; then
  kill_port_vllm   # free the pair for training
  CUDA_VISIBLE_DEVICES=$GPUS PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PYT \
    $R/scripts/distill/lora_train_chat_toolcall.py \
    --base-model $BASE --device auto --max-seq-len 6144 --epochs 2 \
    --lora-r 16 --lora-alpha 32 --lr 1e-4 --val-frac 0.02 --skip-overlong \
    --save-every 50 --resume \
    --train-jsonl $TRAIN --out-dir $ADAPTER \
    > /scratch/logs/p1_train.log 2>&1 || exit 1
  touch $OUT/p1_train_done
fi

# ---------- step 3: eval (4열 비교용; base-32B sub500은 P0a/step-0에서 기측정) ----------
TAG=tb_lodo_mm_32b
if [ ! -f $OUT/p1_eval_done ]; then
  serve_and_wait $TAG $BASE --port $PORT --served-model-name qwen25_32b \
    --enable-lora --lora-modules ${TAG}=$ADAPTER \
    --tensor-parallel-size 2 --max-model-len 8192 --gpu-memory-utilization 0.90 || exit 1
  infer_eval data_multimedia_sub500   data_multimedia   $TAG resource   # held-out (1차 판정)
  infer_eval data_huggingface_sub500  data_huggingface  $TAG resource   # in-domain
  infer_eval data_dailylifeapis_sub500 data_dailylifeapis $TAG temporal # in-domain
  infer_eval data_multimedia          data_multimedia   $TAG resource   # held-out FULL (보충)
  infer_eval data_multimedia          data_multimedia   qwen25_32b resource  # base FULL (보충)
  kill_port_vllm
  touch $OUT/p1_eval_done
fi

# adapter -> HF (sync loop covers $OUT already; explicit final push for safety)
$HF upload $HFREPO $ADAPTER train/taskbench_runs/sft/qwen32b_tb_lodo_mm \
  --repo-type dataset --commit-message "P1 adapter final $(date -u +%FT%TZ)" || true
echo "P1_ALL_DONE $(date)"
