#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:-/scratch/boltzmann/boltzmann-attention}"
VENV_DIR="${2:-/scratch/boltzmann/venvs/seka}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-4B-Base}"
SHORT_NAME="${SHORT_NAME:-Qwen3-4B-Base}"
RUN_TAG="${RUN_TAG:-phase1_robustness_qwen3_4b_500}"
EXAMPLE_SUBSET="${EXAMPLE_SUBSET:-0:500}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
BASELINE_GPU="${BASELINE_GPU:-0}"
SEKA_GPU="${SEKA_GPU:-1}"
ONTOLOGY_GPU="${ONTOLOGY_GPU:-2}"
SEKA_OUT_TAG="${SEKA_OUT_TAG:-counterfact-diff010}"
ONTOLOGY_OUT_TAG="${ONTOLOGY_OUT_TAG:-ontology-qwen3-4b-rank8}"
BOOTSTRAP_ITERS="${BOOTSTRAP_ITERS:-5000}"

if [ -d /home/azureuser/.cache/huggingface ]; then
  export HF_HOME="${HF_HOME:-/home/azureuser/.cache/huggingface}"
  export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/home/azureuser/.cache/huggingface}"
fi

source "$VENV_DIR/bin/activate"
export PYTHONPATH="$ROOT_DIR/external/SEKA:$ROOT_DIR"

RESULT_DIR="$ROOT_DIR/results/$RUN_TAG"
mkdir -p "$RESULT_DIR"

cd "$ROOT_DIR"

pids=()
cleanup() {
  local code=$?
  if [ "$code" -ne 0 ] && [ "${#pids[@]}" -gt 0 ]; then
    for pid in "${pids[@]}"; do
      kill "$pid" 2>/dev/null || true
    done
  fi
}
trap cleanup EXIT

python scripts/phase1_normalize_jsonl.py --input external/SEKA/data/pasta_bench/counterfact.jsonl
python scripts/phase1_seka_env_audit.py --json-out "$RESULT_DIR/env_audit.json"

python scripts/phase1_ontology_projection_rank8.py \
  --model-id "$MODEL_ID" \
  --short-name "$SHORT_NAME" \
  --device "cuda:${ONTOLOGY_GPU}" \
  --out-tag "$ONTOLOGY_OUT_TAG"

python external/SEKA/src/custom_builders/synthetic_qa_builder.py \
  --model "$MODEL_ID" \
  --data external/SEKA/data/synthetic/pair_qa_new.jsonl \
  --output_dir "external/SEKA/seka_projections/${SEKA_OUT_TAG}/${SHORT_NAME}" \
  --max_samples 200 \
  --min_diff 0.10 \
  --top_pct 0.90 \
  --layers last10

run_eval() {
  local gpu="$1"
  local name="$2"
  shift 2
  CUDA_VISIBLE_DEVICES="$gpu" python external/SEKA/benchmarks/eval_fact_gen.py \
    --model "$MODEL_ID" \
    --data_path external/SEKA/data/pasta_bench \
    --output_dir "$RESULT_DIR/$name" \
    --overwrite_output_dir \
    --benchmarks efficacy paraphrase \
    --add_unmediated_fact True \
    --batch_size "$BATCH_SIZE" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --example_subset "$EXAMPLE_SUBSET" \
    "$@"
}

run_eval "$BASELINE_GPU" baseline &
pid_baseline=$!
pids+=("$pid_baseline")

run_eval "$SEKA_GPU" seka \
  --seka \
  --pos "external/SEKA/seka_projections/${SEKA_OUT_TAG}/${SHORT_NAME}/${SHORT_NAME}_pos_proj.pt" \
  --neg "external/SEKA/seka_projections/${SEKA_OUT_TAG}/${SHORT_NAME}/${SHORT_NAME}_neg_proj.pt" \
  --amplify_pos 1.56 \
  --amplify_neg 0.0 \
  --layers last10 &
pid_seka=$!
pids+=("$pid_seka")

run_eval "$ONTOLOGY_GPU" ontology \
  --seka \
  --pos "external/SEKA/seka_projections/${ONTOLOGY_OUT_TAG}/${SHORT_NAME}_pos_proj.pt" \
  --neg "external/SEKA/seka_projections/${ONTOLOGY_OUT_TAG}/${SHORT_NAME}_neg_proj.pt" \
  --amplify_pos 1.56 \
  --amplify_neg 0.0 \
  --layers last10 &
pid_ontology=$!
pids+=("$pid_ontology")

wait_status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    wait_status=1
  fi
done
if [ "$wait_status" -ne 0 ]; then
  exit "$wait_status"
fi

for name in baseline seka ontology; do
  python scripts/phase1_export_counterfact_samples.py \
    --efficacy-json "$RESULT_DIR/$name/efficacy.json" \
    --paraphrase-json "$RESULT_DIR/$name/paraphrase.json" \
    --out-jsonl "$RESULT_DIR/$name/per_example.jsonl" \
    --out-csv "$RESULT_DIR/$name/per_example.csv"
done

python scripts/phase1_paired_bootstrap.py \
  --a "$RESULT_DIR/baseline/per_example.jsonl" \
  --b "$RESULT_DIR/seka/per_example.jsonl" \
  --label-a baseline \
  --label-b seka \
  --bootstrap-iters "$BOOTSTRAP_ITERS" \
  --json-out "$RESULT_DIR/baseline_vs_seka_bootstrap.json"

python scripts/phase1_paired_bootstrap.py \
  --a "$RESULT_DIR/baseline/per_example.jsonl" \
  --b "$RESULT_DIR/ontology/per_example.jsonl" \
  --label-a baseline \
  --label-b ontology \
  --bootstrap-iters "$BOOTSTRAP_ITERS" \
  --json-out "$RESULT_DIR/baseline_vs_ontology_bootstrap.json"

python scripts/phase1_paired_bootstrap.py \
  --a "$RESULT_DIR/seka/per_example.jsonl" \
  --b "$RESULT_DIR/ontology/per_example.jsonl" \
  --label-a seka \
  --label-b ontology \
  --bootstrap-iters "$BOOTSTRAP_ITERS" \
  --json-out "$RESULT_DIR/seka_vs_ontology_bootstrap.json"

python - <<'PY' "$RESULT_DIR"
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])

def load_metrics(run_dir, bench):
    metrics_path = run_dir / f'{bench}_metrics.json'
    if metrics_path.exists():
        return json.loads(metrics_path.read_text())
    full_path = run_dir / f'{bench}.json'
    return json.loads(full_path.read_text())['metrics']

summary = {}
for name in ['baseline', 'seka', 'ontology']:
    metrics = {}
    for bench in ['efficacy', 'paraphrase']:
        metrics[bench] = load_metrics(root / name, bench)
    summary[name] = metrics

for name in [
    'baseline_vs_seka_bootstrap',
    'baseline_vs_ontology_bootstrap',
    'seka_vs_ontology_bootstrap',
]:
    summary[name] = json.loads((root / f'{name}.json').read_text())

(root / 'summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
PY

echo "[phase1_e8_robustness] done -> $RESULT_DIR"
