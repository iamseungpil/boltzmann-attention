#!/usr/bin/env bash
# GPU0 PM wave 2026-04-15 — runs after LoRA v2 (PID 1899307) completes
# Order: L3' Subtask4 smoke (LoRA + 5 variants) → L3' best on full 497
#        → Llama-Instruct full 497 K + Q-coverage → Subtask1 cross-model with Q
set -u

REPO=/home/woori/workspace_common/boltzmann-attention
cd "$REPO"
source /home/woori/workspace_common/CDP/poc/set.env
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DEV="cuda:0"
QWEN="Qwen/Qwen2.5-7B-Instruct"
LLAMA_INST="NousResearch/Meta-Llama-3.1-8B-Instruct"
LORA_DIR="lora_adapters/qwen25_7b_subtask4_chat_r16"
BONT_QWEN="external/SEKA/seka_projections/ontology-qwen25-7b-metatool/B_ont.pt"
BONT_LLAMA="external/SEKA/seka_projections/ontology-llama31-8b-metatool/B_ont.pt"

LOG=logs/wave_2026_04_15_pm/gpu0
OUT=reports/wave_2026_04_15_pm/gpu0
mkdir -p "$LOG" "$OUT"

WAIT_PID=1899307
echo "[g0] waiting for LoRA v2 PID $WAIT_PID at $(date)" >> "$LOG/chain.log"
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
echo "[g0] LoRA v2 done at $(date), starting wave" >> "$LOG/chain.log"

# === Phase A: L3' Subtask4 smoke with LoRA-merged model + bias variants ===
# Tests whether LoRA v2 (chat-format-trained) baseline + various biases
# beats base no_steer 0.731.
echo "[g0] L3' smoke start $(date)" >> "$LOG/chain.log"

python -c "
import json, torch, os, sys, gc
os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')
sys.path.insert(0, 'scripts/ocq')
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = '$QWEN'
lora_dir = '$LORA_DIR'
device = '$DEV'

tok = AutoTokenizer.from_pretrained(base_model, use_fast=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(
    base_model, dtype=torch.bfloat16, device_map=device,
    attn_implementation='eager', low_cpu_mem_usage=True,
)
model = PeftModel.from_pretrained(model, lora_dir)
model.eval()
model = model.merge_and_unload()
print('[L3 v2] LoRA-merged model loaded', flush=True)

from eval_metatool_subtask4 import run_method

class Args: pass
args = Args()
args.device = device
args.max_new_tokens = 256
args.verbose = False

cfg = model.config
n_kv = cfg.num_key_value_heads
n_q = cfg.num_attention_heads
head_dim = getattr(cfg, 'head_dim', None) or (cfg.hidden_size // n_q)

B_ont = torch.load('$BONT_QWEN', map_location='cpu', weights_only=False)
if isinstance(B_ont, dict): B_ont = B_ont['B_ont']

# Smoke N=20 across 5 variants
data_smoke = json.load(open('/tmp/MetaTool/dataset/tmp_dataset/Task2-Subtask4.json'))[:20]
print(f'[L3 v2] smoke N={len(data_smoke)}', flush=True)

for method in ['no_steer', 'ocq_bias_a0.3', 'ocq_qbias_b-0.1',
               'ocq_qkv_a0.3_v0.3_q-0.1', 'ocq_vbias_a0.3']:
    r = run_method(model, tok, data_smoke, method, args, B_ont,
                   n_kv, head_dim, facet_map=None)
    m = r['macro']
    print(f\"[L3 v2 smoke] {method:30s} F1={m['F1']:.3f} rec={m['recall']:.3f} Exact={m['Exact']:.3f}\", flush=True)
    json.dump({'method': method, 'lora': lora_dir, 'results': [r]},
              open(f\"$OUT/l3v2_smoke_{method}.json\", 'w'), indent=2)
" > "$LOG/l3v2_smoke.log" 2>&1
echo "[g0] L3' smoke done $(date)" >> "$LOG/chain.log"

# === Phase B: L3' full 497 with the best two smoke methods ===
# Decide best via simple grep; default to no_steer + qbias.
echo "[g0] L3' full 497 start $(date)" >> "$LOG/chain.log"
python -c "
import json, torch, os, sys
os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')
sys.path.insert(0, 'scripts/ocq')
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = '$QWEN'
lora_dir = '$LORA_DIR'
device = '$DEV'

tok = AutoTokenizer.from_pretrained(base_model, use_fast=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(
    base_model, dtype=torch.bfloat16, device_map=device,
    attn_implementation='eager', low_cpu_mem_usage=True,
)
model = PeftModel.from_pretrained(model, lora_dir)
model.eval()
model = model.merge_and_unload()

from eval_metatool_subtask4 import run_method

class Args: pass
args = Args()
args.device = device
args.max_new_tokens = 256
args.verbose = False

cfg = model.config
n_kv = cfg.num_key_value_heads
n_q = cfg.num_attention_heads
head_dim = getattr(cfg, 'head_dim', None) or (cfg.hidden_size // n_q)

B_ont = torch.load('$BONT_QWEN', map_location='cpu', weights_only=False)
if isinstance(B_ont, dict): B_ont = B_ont['B_ont']

data_full = json.load(open('/tmp/MetaTool/dataset/tmp_dataset/Task2-Subtask4.json'))
print(f'[L3 v2 full] N={len(data_full)}', flush=True)

for method in ['no_steer', 'ocq_qbias_b-0.1', 'ocq_bias_a0.3']:
    r = run_method(model, tok, data_full, method, args, B_ont,
                   n_kv, head_dim, facet_map=None)
    m = r['macro']
    print(f\"[L3 v2 full] {method:30s} F1={m['F1']:.3f} rec={m['recall']:.3f}\", flush=True)
    json.dump({'method': method, 'lora': lora_dir, 'n_queries': len(data_full),
               'results': [r]},
              open(f\"$OUT/l3v2_full_{method}.json\", 'w'), indent=2)
" > "$LOG/l3v2_full.log" 2>&1
echo "[g0] L3' full done $(date)" >> "$LOG/chain.log"

# === Phase C: Llama-3.1-8B-Instruct Subtask4 full 497 (cross-model verification) ===
# Tests stability + Q-coverage on second instruction-tuned family.
echo "[g0] llama st4 full 497 start $(date)" >> "$LOG/chain.log"
python scripts/ocq/eval_metatool_subtask4.py \
    --model "$LLAMA_INST" --device "$DEV" \
    --methods no_steer ocq_bias_a0.3 ocq_qbias_b-0.1 \
    --b-ont "$BONT_LLAMA" \
    --out "$OUT/llama_inst_st4_full497.json" \
    > "$LOG/llama_st4_full.log" 2>&1
echo "[g0] llama st4 done $(date)" >> "$LOG/chain.log"

# === Phase D: Subtask1 cross-model with Q-coverage ===
# Quick check whether Q-coverage helps single-tool too on Llama.
echo "[g0] llama st1 full start $(date)" >> "$LOG/chain.log"
python scripts/ocq/eval_metatool_subtask1.py \
    --model "$LLAMA_INST" --device "$DEV" \
    --methods no_steer ocq_qbias_b-0.1 \
    --b-ont "$BONT_LLAMA" \
    --max-new-tokens 32 \
    --out "$OUT/llama_st1_qbias_full995.json" \
    > "$LOG/llama_st1_qbias.log" 2>&1
echo "[g0] llama st1 done $(date)" >> "$LOG/chain.log"

# === Final summary ===
python3 -c "
import json, glob, os
print('=== GPU0 PM wave summary ===')
for p in sorted(glob.glob('$OUT/*.json')):
    try:
        d = json.load(open(p))
        for r in d.get('results', []):
            m = r.get('macro', r)
            if isinstance(m, dict) and 'F1' in m:
                print(f\"{os.path.basename(p):45s} {r['method']:30s} F1={m['F1']:.3f} rec={m.get('recall',0):.3f}\")
            elif 'top1_accuracy' in r:
                print(f\"{os.path.basename(p):45s} {r['method']:30s} top1={r['top1_accuracy']*100:.2f}%\")
    except Exception as e:
        print(f'{p}: ERROR {e}')
" >> "$LOG/SUMMARY.txt" 2>&1
echo "[g0] WAVE COMPLETE $(date)" >> "$LOG/chain.log"
