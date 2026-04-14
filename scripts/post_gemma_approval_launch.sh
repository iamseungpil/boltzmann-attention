#!/usr/bin/env bash
# Run this AFTER HuggingFace approves gemma-3-27b-it access AND HF_TOKEN is set.
#
# Sequence:
#   1. Verify access (tokenizer_config.json download)
#   2. Download model weights (54.9 GB — 30-60 min on 1 Gbps)
#   3. Build B_ont for Gemma-3-27b-it (pre-RoPE facet basis)
#   4. Queue E1 / E2 / E6 cells under Gemma after Wave 4

set -u

REPO=/home/woori/workspace_common/boltzmann-attention
cd "$REPO"
source /home/woori/workspace_common/CDP/poc/set.env

if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN not set. Run: export HF_TOKEN=hf_xxx or huggingface-cli login"
    exit 1
fi

LOG=logs/gemma_post_approval/launch.log
mkdir -p logs/gemma_post_approval reports/gemma

# Step 1: verify access
echo "[1/4] Verifying Gemma-3-27b-it access..."
python3 -c "
from huggingface_hub import HfApi
import os
api = HfApi(token=os.environ['HF_TOKEN'])
info = api.model_info('google/gemma-3-27b-it')
print(f'  access verified, gated={info.gated}, last_modified={info.last_modified}')
" 2>&1 | tee -a "$LOG"

# Step 2: download weights (heavy, 54.9 GB)
echo "[2/4] Downloading Gemma-3-27b-it weights (~55GB)..."
python3 -c "
from huggingface_hub import snapshot_download
p = snapshot_download(
    repo_id='google/gemma-3-27b-it',
    allow_patterns=['*.json','*.model','*.safetensors','tokenizer*'],
)
print(f'  cached to {p}')
" 2>&1 | tee -a "$LOG"

# Step 3: Build B_ont for Gemma
echo "[3/4] Building B_ont for Gemma-3-27b-it (pre-RoPE)..."
python3 scripts/ocq/build_qwen_metatool_b_ont.py \
    --model google/gemma-3-27b-it --device cuda:0 \
    --target-layers "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41" \
    --pad-to-max \
    --out external/SEKA/seka_projections/ontology-gemma3-27b-it-metatool/B_ont.pt \
    2>&1 | tee -a "$LOG"

# Generate random + featshuffle controls
python3 scripts/ocq/make_control_b_ont.py \
    --src external/SEKA/seka_projections/ontology-gemma3-27b-it-metatool/B_ont.pt \
    --out external/SEKA/seka_projections/ontology-gemma3-27b-it-metatool-random/B_ont.pt \
    --mode random_orthonormal --seed 0
python3 scripts/ocq/make_control_b_ont.py \
    --src external/SEKA/seka_projections/ontology-gemma3-27b-it-metatool/B_ont.pt \
    --out external/SEKA/seka_projections/ontology-gemma3-27b-it-metatool-featshuffle/B_ont.pt \
    --mode feature_shuffle --seed 0

# Step 4: placeholder — add Gemma run to E1/E2/E6 queue
echo "[4/4] Gemma setup complete. Ready to add to E1/E2/E6 runs."
echo ""
echo "Next: edit scripts/run_e1_e2_e6_primary.sh to include:"
echo "  MODEL=google/gemma-3-27b-it"
echo "  B_ONT=external/SEKA/seka_projections/ontology-gemma3-27b-it-metatool/B_ont.pt"
echo ""
echo "Expected runtime per cell (Gemma-3-27B on MetaTool full 995): ~2-3 hours with bf16 on a single A6000 (48GB)."
