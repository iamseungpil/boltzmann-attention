#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:-/scratch/boltzmann/boltzmann-attention}"
VENV_DIR="${2:-/scratch/boltzmann/venvs/seka}"
GPU="${GPU:-0}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-4B-Base}"
SHORT_NAME="${SHORT_NAME:-Qwen3-4B-Base}"
SMOKE_TAG="${SMOKE_TAG:-phase1_smoke_qwen3_4b}"

if [ -d /home/azureuser/.cache/huggingface ]; then
  export HF_HOME="${HF_HOME:-/home/azureuser/.cache/huggingface}"
  export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/home/azureuser/.cache/huggingface}"
fi

source "$VENV_DIR/bin/activate"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$ROOT_DIR/external/SEKA:$ROOT_DIR"

RESULT_DIR="$ROOT_DIR/results/$SMOKE_TAG"
mkdir -p "$RESULT_DIR"

cd "$ROOT_DIR"
python scripts/phase1_normalize_jsonl.py --input external/SEKA/data/pasta_bench/counterfact.jsonl
python scripts/phase1_seka_env_audit.py --json-out "$RESULT_DIR/env_audit.json"
python scripts/ontology_facet_basis.py --self-test > "$RESULT_DIR/ontology_selftest.txt"
python scripts/phase1_ontology_projection.py --self-test --require-external-seka > "$RESULT_DIR/projection_selftest.txt"
python scripts/phase1_ontology_projection_rank8.py --self-test --require-external-seka > "$RESULT_DIR/projection_rank8_selftest.txt"

python scripts/phase1_ontology_projection.py \
  --model-id "$MODEL_ID" \
  --short-name "$SHORT_NAME" \
  --device cuda:0 \
  --out-tag "${SMOKE_TAG}_ontology"

python external/SEKA/src/custom_builders/synthetic_qa_builder.py \
  --model "$MODEL_ID" \
  --data external/SEKA/data/synthetic/pair_qa_new.jsonl \
  --output_dir "external/SEKA/seka_projections/${SMOKE_TAG}_seka/${SHORT_NAME}" \
  --max_samples 20 \
  --min_diff 0.10 \
  --top_pct 0.90 \
  --layers last10

python external/SEKA/benchmarks/eval_fact_gen.py \
  --model "$MODEL_ID" \
  --data_path external/SEKA/data/pasta_bench \
  --output_dir "$RESULT_DIR/baseline" \
  --overwrite_output_dir \
  --benchmarks efficacy paraphrase \
  --add_unmediated_fact True \
  --batch_size 4 \
  --max_new_tokens 32 \
  --example_subset 0:10

python external/SEKA/benchmarks/eval_fact_gen.py \
  --model "$MODEL_ID" \
  --data_path external/SEKA/data/pasta_bench \
  --output_dir "$RESULT_DIR/seka" \
  --overwrite_output_dir \
  --benchmarks efficacy paraphrase \
  --add_unmediated_fact True \
  --batch_size 4 \
  --max_new_tokens 32 \
  --example_subset 0:10 \
  --seka \
  --pos "external/SEKA/seka_projections/${SMOKE_TAG}_seka/${SHORT_NAME}/${SHORT_NAME}_pos_proj.pt" \
  --neg "external/SEKA/seka_projections/${SMOKE_TAG}_seka/${SHORT_NAME}/${SHORT_NAME}_neg_proj.pt" \
  --amplify_pos 1.56 \
  --amplify_neg 0.0 \
  --layers last10

python external/SEKA/benchmarks/eval_fact_gen.py \
  --model "$MODEL_ID" \
  --data_path external/SEKA/data/pasta_bench \
  --output_dir "$RESULT_DIR/ontology" \
  --overwrite_output_dir \
  --benchmarks efficacy paraphrase \
  --add_unmediated_fact True \
  --batch_size 4 \
  --max_new_tokens 32 \
  --example_subset 0:10 \
  --seka \
  --pos "external/SEKA/seka_projections/${SMOKE_TAG}_ontology/${SHORT_NAME}_pos_proj.pt" \
  --neg "external/SEKA/seka_projections/${SMOKE_TAG}_ontology/${SHORT_NAME}_neg_proj.pt" \
  --amplify_pos 1.56 \
  --amplify_neg 0.0 \
  --layers last10

python - <<'PY' "$RESULT_DIR"
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
summary = {}
for name in ["baseline", "seka", "ontology"]:
    metrics = {}
    for bench in ["efficacy", "paraphrase"]:
        p = root / name / f"{bench}_metrics.json"
        if p.exists():
            metrics[bench] = json.loads(p.read_text())
    summary[name] = metrics
(root / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
PY

echo "[phase1_e8_smoke] done -> $RESULT_DIR"
