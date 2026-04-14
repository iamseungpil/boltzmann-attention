#!/usr/bin/env bash
# Rerun of LoRA hybrid pipeline with GPU isolation + alloc-conf fix.
# Original attempt 2026-04-15 02:32 KST OOM'd due to sibling procs on GPU1.
set -u

REPO=/home/woori/workspace_common/boltzmann-attention
cd "$REPO"
source /home/woori/workspace_common/CDP/poc/set.env

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL="Qwen/Qwen2.5-7B-Instruct"
BONT_BASE="external/SEKA/seka_projections/ontology-qwen25-7b-metatool/B_ont.pt"
LORA_DIR="lora_adapters/qwen25_7b_subtask1_r8"
LOG_DIR=logs/lora_hybrid_rerun
OUT_DIR=reports/lora_hybrid
DEVICE="cuda:1"
mkdir -p "$LOG_DIR" "$OUT_DIR" "$LORA_DIR"

echo "[lora-rerun] start $(date)" >> "$LOG_DIR/chain.log"

# L1 — reduced batch-size 1 to be safe with the logits.float() spike
echo "[lora-rerun] L1 train $(date)" >> "$LOG_DIR/chain.log"
python scripts/ocq/lora_train_metatool.py \
    --base-model "$MODEL" --device "$DEVICE" \
    --train-dataset /tmp/MetaTool/dataset/tmp_dataset/Task2-Subtask1.json \
    --train-size 500 --val-size 50 --epochs 3 \
    --lora-r 8 --lora-alpha 16 --batch-size 1 --lr 1e-4 \
    --lora-target q_proj k_proj v_proj \
    --out-dir "$LORA_DIR" \
    > "$LOG_DIR/l1_train.log" 2>&1
L1_RC=$?
echo "[lora-rerun] L1 done rc=$L1_RC $(date)" >> "$LOG_DIR/chain.log"

if [ $L1_RC -ne 0 ]; then
    echo "[lora-rerun] L1 FAILED, aborting" >> "$LOG_DIR/chain.log"
    exit 1
fi

# L3 — Subtask4 N=20 smoke with LoRA-merged model (L2 still using base B_ont)
echo "[lora-rerun] L3 smoke $(date)" >> "$LOG_DIR/chain.log"
python -c "
import json, torch, os, sys
os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')
sys.path.insert(0, 'scripts/ocq')
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = '$MODEL'
lora_dir = '$LORA_DIR'
device = '$DEVICE'

tok = AutoTokenizer.from_pretrained(base_model, use_fast=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(
    base_model, dtype=torch.bfloat16, device_map=device,
    attn_implementation='eager', low_cpu_mem_usage=True,
)
model = PeftModel.from_pretrained(model, lora_dir)
model.eval()
model = model.merge_and_unload()
print('[l3] model loaded, LoRA merged', flush=True)

from eval_metatool_subtask4 import run_method

data = json.load(open('/tmp/MetaTool/dataset/tmp_dataset/Task2-Subtask4.json'))[:20]
print(f'[l3] {len(data)} queries')

class Args: pass
args = Args()
args.device = device
args.max_new_tokens = 256
args.verbose = False

cfg = model.config
n_kv = cfg.num_key_value_heads
n_q = cfg.num_attention_heads
head_dim = getattr(cfg, 'head_dim', None) or (cfg.hidden_size // n_q)

r = run_method(model, tok, data, 'no_steer', args, B_ont=None, n_kv=n_kv, head_dim=head_dim, facet_map=None)
print(f\"[l3a] no_steer+LoRA F1={r['macro']['F1']:.3f} rec={r['macro']['recall']:.3f} Exact={r['macro']['Exact']:.3f}\", flush=True)
json.dump({'method': 'lora_no_steer', 'results': [r]}, open('$OUT_DIR/l3a_lora_no_steer.json','w'), indent=2)

B_ont = torch.load('$BONT_BASE', map_location='cpu', weights_only=False)
if isinstance(B_ont, dict): B_ont = B_ont['B_ont']
r2 = run_method(model, tok, data, 'ocq_bias_a0.3', args, B_ont=B_ont, n_kv=n_kv, head_dim=head_dim, facet_map=None)
print(f\"[l3b] LoRA+K-bias a0.3 F1={r2['macro']['F1']:.3f} rec={r2['macro']['recall']:.3f} Exact={r2['macro']['Exact']:.3f}\", flush=True)
json.dump({'method': 'lora_plus_kbias_a0.3', 'results': [r2]}, open('$OUT_DIR/l3b_lora_plus_kbias.json','w'), indent=2)
" > "$LOG_DIR/l3_smoke.log" 2>&1
echo "[lora-rerun] L3 done $(date)" >> "$LOG_DIR/chain.log"

python3 -c "
import json, glob
print('=== LoRA Hybrid Rerun Summary ===')
for p in sorted(glob.glob('$OUT_DIR/l3*.json')):
    d = json.load(open(p))
    for r in d.get('results', []):
        m = r.get('macro', r)
        if isinstance(m, dict) and 'F1' in m:
            print(f\"{p.split('/')[-1]:35s} F1={m['F1']:.3f} rec={m.get('recall',0):.3f}\")
" >> "$LOG_DIR/summary.log" 2>&1
echo "[lora-rerun] ALL DONE $(date)" >> "$LOG_DIR/chain.log"
