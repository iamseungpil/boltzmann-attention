#!/usr/bin/env bash
# H-cat falsifiability diagnostic: Mistral-Inst (reversed) vs Qwen-Inst (intact) vs Llama-Inst (large+)
# Sequential on GPU0 (free).
set -u
REPO=/home/woori/workspace_common/boltzmann-attention
cd "$REPO"
source /home/woori/workspace_common/CDP/poc/set.env
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LOG_DIR=logs/hcat_diagnostic_2026_04_15
OUT_DIR=reports/hcat_diagnostic_2026_04_15
mkdir -p "$LOG_DIR" "$OUT_DIR"

LAYERS="5 13 21"
N_PROMPTS=64
DEV=cuda:0

# 1. Mistral-Inst (the reversed model — primary test)
echo "[hcat] Mistral-Inst start $(date)" >> "$LOG_DIR/wave.log"
python scripts/ocq/measure_hcat_violation_mistral_2026_04_15.py \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --b-ont external/SEKA/seka_projections/ontology-mistral-7b-v03-metatool-skipL0-padmax/B_ont.pt \
    --device "$DEV" --layers $LAYERS --n-prompts $N_PROMPTS \
    --out "$OUT_DIR/mistral_inst.json" \
    > "$LOG_DIR/mistral_inst.log" 2>&1
echo "[hcat] Mistral-Inst done rc=$? $(date)" >> "$LOG_DIR/wave.log"

# 2. Qwen-Inst (specificity intact, baseline for comparison)
echo "[hcat] Qwen-Inst start $(date)" >> "$LOG_DIR/wave.log"
python scripts/ocq/measure_hcat_violation_mistral_2026_04_15.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --b-ont external/SEKA/seka_projections/ontology-qwen25-7b-metatool/B_ont.pt \
    --device "$DEV" --layers $LAYERS --n-prompts $N_PROMPTS \
    --out "$OUT_DIR/qwen_inst.json" \
    > "$LOG_DIR/qwen_inst.log" 2>&1
echo "[hcat] Qwen-Inst done rc=$? $(date)" >> "$LOG_DIR/wave.log"

# 3. Llama-Inst (large positive lift, upper-bound comparison)
echo "[hcat] Llama-Inst start $(date)" >> "$LOG_DIR/wave.log"
python scripts/ocq/measure_hcat_violation_mistral_2026_04_15.py \
    --model NousResearch/Meta-Llama-3.1-8B-Instruct \
    --b-ont external/SEKA/seka_projections/ontology-llama31-8b-metatool/B_ont.pt \
    --device "$DEV" --layers $LAYERS --n-prompts $N_PROMPTS \
    --out "$OUT_DIR/llama_inst.json" \
    > "$LOG_DIR/llama_inst.log" 2>&1
echo "[hcat] Llama-Inst done rc=$? $(date)" >> "$LOG_DIR/wave.log"

# Summary
echo "" >> "$LOG_DIR/wave.log"
echo "=== H-CAT FALSIFIABILITY VERDICT ===" >> "$LOG_DIR/wave.log"
python3 -c "
import json
for name, path in [
    ('Mistral-Inst', '$OUT_DIR/mistral_inst.json'),
    ('Qwen-Inst',    '$OUT_DIR/qwen_inst.json'),
    ('Llama-Inst',   '$OUT_DIR/llama_inst.json'),
]:
    try:
        d = json.load(open(path))
        print(f'\n{name}:')
        print(f'  pathological layers: {d[\"pathological_layers_count\"]}')
        for m in d['per_layer_hcat']:
            verdict = 'HOLDS' if m['gain_mean']>2.0 else 'WEAK' if m['gain_mean']>1.2 else 'VIOLATED'
            print(f'  L={m[\"layer\"]:2d} gain={m[\"gain_mean\"]:.2f}x baseline={m[\"random_baseline\"]:.3f} [{verdict}]')
    except Exception as e:
        print(f'{name}: {e}')
" >> "$LOG_DIR/wave.log" 2>&1

echo "[hcat] WAVE COMPLETE $(date)" >> "$LOG_DIR/wave.log"
