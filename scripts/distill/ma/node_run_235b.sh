#!/bin/bash
# node_run_235b.sh — 235B EXTENSION POINT for BOTH coworker lanes on ONE served model (cost: serve once).
#   Qwen3-235B-A22B-Instruct-2507-FP8 (FP8 fits 4xH100 320GB; bf16=470GB would need 8 GPUs). MoE:
#   active 22B != total 235B (★star in aggregation). FP8 = quant confound (tag _fp8) — design-sanctioned
#   extension, NOT the load-bearing dense ladder (0.5-72B).
#   Runs the FROZEN eval scripts UNMODIFIED (same args as depth_scale_batch / multidomain_scale_eval),
#   just pointed at the already-running server so the giant model is loaded once.
#   Lane 1 (B-budget):   depth_eval arms A/B/D over synth_depth N{5,10,20,50}  -> depth_scale/
#   Lane 2 (multidomain): tau2_op_eval retail/airline x gloss{0,1}             -> multidomain_scale/
set -u
unset OPENROUTER_API_KEY ANTHROPIC_API_KEY OPENAI_API_KEY 2>/dev/null || true   # cost-incident defense
MODEL="Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"; TAG="235B_fp8"; PORT=8060
REPO=/scratch/boltzmann-attention; MA=$REPO/scripts/distill/ma; CASES=$MA/cases
PY=/scratch/venvs/sop_env/bin/python; VLLM=/scratch/venvs/sop_env/bin/vllm
HF=/scratch/venvs/sop_env/bin/hf; PIP=/scratch/venvs/sop_env/bin/pip
HFREPO=iamseungpil/sopbench-trackb-h200
SCRATCH=/scratch/woori_scratch
DRUN=$SCRATCH/depth; MRES=$SCRATCH/depth/c8/multidomain/results
mkdir -p $DRUN $MRES $SCRATCH/logs
export PYTHONPATH=$MA
set -x

# 0. deps (pin fastapi/starlette: vllm-0.10.2 serve crash '_IncludedRouter' otherwise).
$PIP install -q requests xgrammar "fastapi==0.136.3" "starlette==1.2.1" >> $SCRATCH/logs/pip.log 2>&1 || true

# 0b. restore prior 235B partials + background sync of BOTH lanes
$HF download $HFREPO --repo-type dataset --include "depth_scale/*" --include "multidomain_scale/*" --local-dir /scratch/hf_r 2>>$SCRATCH/logs/restore.log || true
[ -d /scratch/hf_r/depth_scale ] && cp -rn /scratch/hf_r/depth_scale/* $DRUN/ 2>/dev/null || true
[ -d /scratch/hf_r/multidomain_scale ] && cp -rn /scratch/hf_r/multidomain_scale/* $MRES/ 2>/dev/null || true
( while true; do
    $HF upload $HFREPO $DRUN depth_scale --repo-type dataset --commit-message "depth_scale 235B sync $(date -u +%FT%TZ)" >> $SCRATCH/logs/sync.log 2>&1
    $HF upload $HFREPO $MRES multidomain_scale --repo-type dataset --commit-message "multidomain_scale 235B sync $(date -u +%FT%TZ)" >> $SCRATCH/logs/sync.log 2>&1
    sleep 600; done ) &
SYNC=$!

# 0c. model meta (MoE: layers / d_model / experts / active) — design §2.
$HF download "$MODEL" --include "config.json" --local-dir $SCRATCH/cfg_$TAG >> $SCRATCH/logs/cfg.log 2>&1 || true
cp $SCRATCH/cfg_$TAG/config.json $DRUN/meta_${TAG}.json 2>/dev/null
cp $SCRATCH/cfg_$TAG/config.json $MRES/meta_${TAG}.json 2>/dev/null

# 1. download + serve ONCE (FP8 auto-detected from checkpoint; TP4).
$HF download "$MODEL" >> $SCRATCH/logs/hfdl.log 2>&1 || { echo "DL_FAIL"; kill $SYNC; exit 1; }
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 4
CUDA_VISIBLE_DEVICES=0,1,2,3 setsid nohup $VLLM serve "$MODEL" --port $PORT \
  --tensor-parallel-size 4 --max-model-len 16384 --gpu-memory-utilization 0.92 \
  > $SCRATCH/vllm_235b.log 2>&1 &
ok=0; for i in $(seq 1 240); do curl -s localhost:$PORT/v1/models 2>/dev/null | grep -q '"id"' && ok=1 && break; sleep 10; done
[ $ok = 1 ] || { echo "SERVE_FAIL"; tail -40 $SCRATCH/vllm_235b.log; kill $SYNC; exit 1; }
BASE=http://localhost:$PORT/v1

# 2. LANE 1 — B-budget depth (frozen synth seed 0; arms A/B/D). Skip per-N if json present.
for NN in 5 10 20 50; do
  [ -f $DRUN/synth_N${NN}.jsonl ] || $PY $MA/synth_depth.py --out $DRUN/synth_N${NN}.jsonl --n 250 --N $NN --iso 1 --seed 0
  [ -f $DRUN/depth_${TAG}_N${NN}.json ] && { echo "skip depth N$NN"; continue; }
  echo "===== DEPTH $TAG N=$NN ====="
  $PY $MA/depth_eval.py --data $DRUN/synth_N${NN}.jsonl --base $BASE --model "$MODEL" \
    --arms A,B,D --out $DRUN/depth_${TAG}_N${NN}.json
done

# 3. LANE 2 — multidomain content-routing (retail/airline x gloss 0/1). Skip if present.
for D in retail airline; do
  for G in 0 1; do
    [ -f $MRES/${TAG}__${D}_g${G}.json ] && { echo "skip md $D g$G"; continue; }
    echo "===== MD $TAG $D gloss=$G ====="
    $PY $MA/tau2_op_eval.py --cases $CASES/tau2_${D}_cases.jsonl --base $BASE \
      --model "$MODEL" --gloss $G --out $MRES/${TAG}__${D}_g${G}.json
  done
done

for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done
# 4. final sync of both lanes
kill $SYNC 2>/dev/null || true
$HF upload $HFREPO $DRUN depth_scale --repo-type dataset --commit-message "depth_scale 235B FINAL $(date -u +%FT%TZ)" >> $SCRATCH/logs/sync.log 2>&1 || true
$HF upload $HFREPO $MRES multidomain_scale --repo-type dataset --commit-message "multidomain_scale 235B FINAL $(date -u +%FT%TZ)" >> $SCRATCH/logs/sync.log 2>&1 || true
echo "BOTH_LANES_235B_NODE_DONE $(date)"
