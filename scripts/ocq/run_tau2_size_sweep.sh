#!/usr/bin/env bash
# run_tau2_size_sweep.sh — Qwen size × method × domain sweep on tau2-bench.
#
# Execution plan M1 (tau2 retail, primary performance table) and M2 (tau2
# airline, cross-domain sign consistency) from
# reports/steering_paper/EXPERIMENT_PLAN_UNIFIED_2026_04_16_v1.md.
#
# Expectation: Layer-Adaptive K+Q (ours) beats stationary K, Q-only, SEKA,
# and a PCA-basis ablation on all model sizes except 1.5B (fragile regime),
# with the PCA row producing a neutral result vs no_steer.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="${ROOT}/reports/tau2_size_sweep_2026_04_17"
mkdir -p "${OUT_DIR}"

MODELS=(
  "Qwen/Qwen2.5-1.5B-Instruct:qwen25_1_5b"
  "Qwen/Qwen2.5-3B-Instruct:qwen25_3b"
  "Qwen/Qwen2.5-7B-Instruct:qwen25_7b"
  "Qwen/Qwen2.5-14B-Instruct:qwen25_14b"
)

DOMAINS=("retail" "airline")

METHODS=(
  "no_steer"
  "ocq_bias_a0.3"
  "ocq_qbias_b-0.03"
  "ocq_ladapt_k0.05_q-0.03"
)

MAX_SAMPLES=${MAX_SAMPLES:-60}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-512}

for model_spec in "${MODELS[@]}"; do
  model="${model_spec%%:*}"
  tag="${model_spec##*:}"

  for domain in "${DOMAINS[@]}"; do
    b_ont_path="${ROOT}/reports/tau2_ontology_bases/${tag}_${domain}_B_ont.pt"
    if [[ ! -f "${b_ont_path}" ]]; then
      echo "[warn] missing B_ont for ${tag}/${domain} at ${b_ont_path}"
      echo "[warn] build it with:"
      echo "       python scripts/ocq/build_tau2_ontology.py --model ${model} --domain ${domain} --out ${b_ont_path}"
      continue
    fi

    joined_methods="$(IFS=' '; echo "${METHODS[*]}")"
    out_path="${OUT_DIR}/${tag}_${domain}.json"
    echo ""
    echo "===== ${model} on tau2/${domain} -> ${out_path} ====="
    python "${ROOT}/scripts/ocq/eval_tau2_bench.py" \
      --model "${model}" \
      --device cuda:0 \
      --domain "${domain}" \
      --b-ont "${b_ont_path}" \
      --methods ${joined_methods} \
      --max-samples "${MAX_SAMPLES}" \
      --max-new-tokens "${MAX_NEW_TOKENS}" \
      --out "${out_path}"
  done
done

echo ""
echo "[size-sweep] complete. Summaries landed in ${OUT_DIR}."
