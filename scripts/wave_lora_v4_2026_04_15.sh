#!/usr/bin/env bash
# LoRA v4 — richer training: Subtask1+2+3 mixed (3,830 single-tool) + 50% synth 2-tool
# Diagnoses whether v3 0.333 → v4 closer to base 0.731 with more diverse training
set -u

REPO=/home/woori/workspace_common/boltzmann-attention
cd "$REPO"
source /home/woori/workspace_common/CDP/poc/set.env
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

QWEN="Qwen/Qwen2.5-7B-Instruct"
LORA_V4_DIR="lora_adapters/qwen25_7b_subtask123_synth_r16"
LOG_DIR=logs/lora_v4_2026_04_15
OUT_DIR=reports/lora_v4_2026_04_15
mkdir -p "$LOG_DIR" "$OUT_DIR"

# Wait for SEKA micro to finish (GPU0 sharing)
WAIT_PID=2535107
echo "[v4] wait SEKA $WAIT_PID at $(date)" >> "$LOG_DIR/chain.log"
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
echo "[v4] SEKA done $(date), launching" >> "$LOG_DIR/chain.log"

# Build merged training set: Subtask1 + Subtask2 + Subtask3
python3 -c "
import json, random
random.seed(42)
all_train = []
for n in ['Task2-Subtask1', 'Task2-Subtask2', 'Task2-Subtask3']:
    d = json.load(open(f'/tmp/MetaTool/dataset/tmp_dataset/{n}.json'))
    all_train.extend(d)
random.shuffle(all_train)
print(f'Merged: {len(all_train)} entries')
json.dump(all_train, open('/tmp/metatool_merged_st123.json', 'w'))
"

echo "[v4] LoRA v4 train start $(date)" >> "$LOG_DIR/chain.log"
python scripts/ocq/lora_train_metatool_v3.py \
    --base-model "$QWEN" --device cuda:0 \
    --train-dataset /tmp/metatool_merged_st123.json \
    --val-dataset   /tmp/MetaTool/dataset/tmp_dataset/Task2-Subtask4.json \
    --train-size 2400 --val-size 50 --epochs 4 \
    --synth-frac 0.5 \
    --lr 5e-5 --batch-size 2 --grad-accum 4 \
    --lora-r 16 --lora-alpha 32 \
    --lora-target q_proj k_proj v_proj o_proj up_proj down_proj \
    --early-stop-patience 2 \
    --out-dir "$LORA_V4_DIR" \
    > "$LOG_DIR/lora_v4_train.log" 2>&1
RC=$?
echo "[v4] LoRA v4 done rc=$RC $(date)" >> "$LOG_DIR/chain.log"

if [ $RC -eq 0 ] && [ -f "$LORA_V4_DIR/adapter_config.json" ]; then
  echo "[v4] L4 eval start $(date)" >> "$LOG_DIR/chain.log"
  python -c "
import json, torch, os, sys
os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')
sys.path.insert(0, 'scripts/ocq')
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = '$QWEN'
lora_dir = '$LORA_V4_DIR'
device = 'cuda:0'

tok = AutoTokenizer.from_pretrained(base_model, use_fast=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(
    base_model, dtype=torch.bfloat16, device_map=device,
    attn_implementation='eager', low_cpu_mem_usage=True,
)
model = PeftModel.from_pretrained(model, lora_dir).merge_and_unload()

from eval_metatool_subtask4 import run_method
class Args: pass
args = Args(); args.device=device; args.max_new_tokens=256; args.verbose=False
cfg = model.config
n_kv = cfg.num_key_value_heads
n_q = cfg.num_attention_heads
head_dim = getattr(cfg, 'head_dim', None) or (cfg.hidden_size // n_q)
B_ont = torch.load('external/SEKA/seka_projections/ontology-qwen25-7b-metatool/B_ont.pt',
                   map_location='cpu', weights_only=False)
if isinstance(B_ont, dict): B_ont = B_ont['B_ont']

data_full = json.load(open('/tmp/MetaTool/dataset/tmp_dataset/Task2-Subtask4.json'))
print(f'[L4 full] N={len(data_full)}')
for method in ['no_steer', 'ocq_qbias_b-0.1', 'ocq_bias_a0.3']:
    r = run_method(model, tok, data_full, method, args, B_ont, n_kv, head_dim, facet_map=None)
    m = r['macro']
    print(f\"[L4 full] {method:30s} F1={m['F1']:.3f} rec={m['recall']:.3f} Exact={m['Exact']:.3f}\")
    json.dump({'method': method, 'lora': lora_dir, 'n_queries': len(data_full),
               'results': [r]}, open(f'$OUT_DIR/l4_full_{method}.json','w'), indent=2)
" > "$LOG_DIR/l4_eval.log" 2>&1
  echo "[v4] L4 done $(date)" >> "$LOG_DIR/chain.log"
fi

echo "[v4] WAVE COMPLETE $(date)" >> "$LOG_DIR/chain.log"
