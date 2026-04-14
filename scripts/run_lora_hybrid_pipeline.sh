#!/usr/bin/env bash
# L1→L2→L3 LoRA hybrid pipeline for Subtask4 rescue.
# Waits for non-uniform smoke supervisor 941974 to finish, then runs sequentially.
set -u

REPO=/home/woori/workspace_common/boltzmann-attention
cd "$REPO"
source /home/woori/workspace_common/CDP/poc/set.env

MODEL="Qwen/Qwen2.5-7B-Instruct"
BONT_BASE="external/SEKA/seka_projections/ontology-qwen25-7b-metatool/B_ont.pt"
LORA_DIR="lora_adapters/qwen25_7b_subtask1_r8"
BONT_LORA_DIR="external/SEKA/seka_projections/ontology-qwen25-7b-it-metatool-lora-r8"
LOG_DIR=logs/lora_hybrid
OUT_DIR=reports/lora_hybrid
mkdir -p "$LOG_DIR" "$OUT_DIR" "$LORA_DIR" "$BONT_LORA_DIR"

# --- Wait for non-uniform smoke supervisor to finish (frees GPU1) ---
WAIT_PID=941974
echo "[lora] waiting for non-uniform supervisor PID $WAIT_PID at $(date)" >> "$LOG_DIR/chain.log"
while kill -0 "$WAIT_PID" 2>/dev/null; do
  sleep 120
done
echo "[lora] non-uniform done at $(date)" >> "$LOG_DIR/chain.log"

# --- L1: LoRA train (GPU1, ~4 GPU-hr) ---
echo "[lora] L1 LoRA training at $(date)" >> "$LOG_DIR/chain.log"
python scripts/ocq/lora_train_metatool.py \
    --base-model "$MODEL" --device cuda:1 \
    --train-dataset /tmp/MetaTool/dataset/tmp_dataset/Task2-Subtask1.json \
    --train-size 500 --val-size 50 --epochs 3 \
    --lora-r 8 --lora-alpha 16 --batch-size 2 --lr 1e-4 \
    --lora-target q_proj k_proj v_proj \
    --out-dir "$LORA_DIR" \
    > "$LOG_DIR/l1_train.log" 2>&1
echo "[lora] L1 done at $(date)" >> "$LOG_DIR/chain.log"

# --- L2: Rebuild B_ont from LoRA-adapted K distribution ---
# We need to modify build_qwen_metatool_b_ont.py or create a wrapper that
# loads base + LoRA adapter, then extracts K. For now, use base B_ont
# as proxy (future: dedicated LoRA B_ont builder).
# Simple approach: use existing base B_ont (text-derived) but evaluate with
# LoRA-adapted model. This tests whether LoRA alone rescues Subtask4.
echo "[lora] L2 SKIPPED — using base B_ont with LoRA model for initial smoke" >> "$LOG_DIR/chain.log"

# --- L3: Subtask4 eval with LoRA-adapted model ---
echo "[lora] L3 Subtask4 N=20 smoke at $(date)" >> "$LOG_DIR/chain.log"

# L3a: LoRA alone (no K-bias) — baseline for LoRA contribution
python -c "
import json, torch, os, sys
os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')
sys.path.insert(0, 'scripts/ocq')
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = '$MODEL'
lora_dir = '$LORA_DIR'
device = 'cuda:1'

tok = AutoTokenizer.from_pretrained(base_model, use_fast=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(
    base_model, dtype=torch.bfloat16, device_map=device,
    attn_implementation='eager', low_cpu_mem_usage=True,
)
model = PeftModel.from_pretrained(model, lora_dir)
model.eval()
model = model.merge_and_unload()  # Merge LoRA into base for simpler hook access
print('[l3a] model loaded with LoRA merged', flush=True)

# Now run Subtask4 eval via the driver
from eval_metatool_subtask4 import run_method, build_fc_prompt, extract_tool_names, compute_metrics
import time

data = json.load(open('/tmp/MetaTool/dataset/tmp_dataset/Task2-Subtask4.json'))[:20]
print(f'[l3a] {len(data)} queries')

class Args: pass
args = Args()
args.device = device
args.max_new_tokens = 256
args.verbose = False

cfg = model.config
n_kv = cfg.num_key_value_heads
n_q = cfg.num_attention_heads
head_dim = getattr(cfg, 'head_dim', None) or (cfg.hidden_size // n_q)

# no_steer + LoRA only (no K-bias)
r = run_method(model, tok, data, 'no_steer', args, B_ont=None, n_kv=n_kv, head_dim=head_dim, facet_map=None)
print(f\"[l3a] no_steer+LoRA F1={r['macro']['F1']:.3f} rec={r['macro']['recall']:.3f} Exact={r['macro']['Exact']:.3f}\")
json.dump({'method': 'lora_no_steer', 'results': [r]}, open('$OUT_DIR/l3a_lora_no_steer.json','w'), indent=2)

# LoRA + K-bias a0.3 (using base B_ont)
B_ont = torch.load('$BONT_BASE', map_location='cpu', weights_only=False)
if isinstance(B_ont, dict): B_ont = B_ont['B_ont']
r2 = run_method(model, tok, data, 'ocq_bias_a0.3', args, B_ont=B_ont, n_kv=n_kv, head_dim=head_dim, facet_map=None)
print(f\"[l3a] LoRA+K-bias a0.3 F1={r2['macro']['F1']:.3f} rec={r2['macro']['recall']:.3f} Exact={r2['macro']['Exact']:.3f}\")
json.dump({'method': 'lora_plus_kbias_a0.3', 'results': [r2]}, open('$OUT_DIR/l3b_lora_plus_kbias.json','w'), indent=2)
" > "$LOG_DIR/l3_smoke.log" 2>&1
echo "[lora] L3 done at $(date)" >> "$LOG_DIR/chain.log"

# Summary
python3 -c "
import json, glob
print('=== LoRA Hybrid Pipeline Summary ===')
for p in sorted(glob.glob('$OUT_DIR/*.json')):
    d = json.load(open(p))
    for r in d.get('results', []):
        m = r.get('macro', r)
        if isinstance(m, dict) and 'F1' in m:
            print(f\"{p.split('/')[-1]:35s} F1={m['F1']:.3f} rec={m.get('recall',0):.3f} Exact={m.get('Exact',0):.3f}\")
" >> "$LOG_DIR/summary.log" 2>&1
