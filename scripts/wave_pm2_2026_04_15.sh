#!/usr/bin/env bash
# PM Wave 2 — runs after both gpu0 (PID 1907195) and gpu1 (PID 1907194) waves complete
# Phases:
#   GPU0: LoRA v3 retrain (synth multi-tool) → L3'' eval
#   GPU1 (parallel after gpu1 wave): K×Q small-α joint sweep, V-only full,
#         Q-coverage layer-ablation, Q-coverage Subtask1 cross-model on Llama
set -u

REPO=/home/woori/workspace_common/boltzmann-attention
cd "$REPO"
source /home/woori/workspace_common/CDP/poc/set.env
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

QWEN="Qwen/Qwen2.5-7B-Instruct"
LLAMA_INST="NousResearch/Meta-Llama-3.1-8B-Instruct"
BONT_QWEN="external/SEKA/seka_projections/ontology-qwen25-7b-metatool/B_ont.pt"
BONT_LLAMA="external/SEKA/seka_projections/ontology-llama31-8b-metatool/B_ont.pt"
LORA_V3_DIR="lora_adapters/qwen25_7b_subtask4_synth_r16"

LOG_BASE=logs/wave_pm2_2026_04_15
OUT_BASE=reports/wave_pm2_2026_04_15
mkdir -p "$LOG_BASE/gpu0" "$LOG_BASE/gpu1" "$OUT_BASE/gpu0" "$OUT_BASE/gpu1"

WAIT_G0=1907195
WAIT_G1=1907194
echo "[pm2] waiting for prev waves G0=$WAIT_G0 G1=$WAIT_G1 at $(date)" >> "$LOG_BASE/chain.log"
while kill -0 "$WAIT_G0" 2>/dev/null || kill -0 "$WAIT_G1" 2>/dev/null; do sleep 60; done
echo "[pm2] prev waves done at $(date), launching" >> "$LOG_BASE/chain.log"

# === GPU0 chain: LoRA v3 train → L3'' eval ===
(
  L_DIR=$LOG_BASE/gpu0
  O_DIR=$OUT_BASE/gpu0
  echo "[g0p2] LoRA v3 train start $(date)" >> "$LOG_BASE/chain.log"
  python scripts/ocq/lora_train_metatool_v3.py \
      --base-model "$QWEN" --device cuda:0 \
      --train-dataset /tmp/MetaTool/dataset/tmp_dataset/Task2-Subtask1.json \
      --val-dataset   /tmp/MetaTool/dataset/tmp_dataset/Task2-Subtask4.json \
      --train-size 600 --val-size 50 --epochs 4 \
      --synth-frac 0.5 \
      --lr 5e-5 --batch-size 2 --grad-accum 4 \
      --lora-r 16 --lora-alpha 32 \
      --lora-target q_proj k_proj v_proj o_proj up_proj down_proj \
      --early-stop-patience 2 \
      --out-dir "$LORA_V3_DIR" \
      > "$L_DIR/lora_v3_train.log" 2>&1
  RC=$?
  echo "[g0p2] LoRA v3 done rc=$RC $(date)" >> "$LOG_BASE/chain.log"

  if [ $RC -eq 0 ] && [ -f "$LORA_V3_DIR/adapter_config.json" ]; then
    echo "[g0p2] L3'' smoke + full start $(date)" >> "$LOG_BASE/chain.log"
    python -c "
import json, torch, os, sys
os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')
sys.path.insert(0, 'scripts/ocq')
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = '$QWEN'
lora_dir = '$LORA_V3_DIR'
device = 'cuda:0'

tok = AutoTokenizer.from_pretrained(base_model, use_fast=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(
    base_model, dtype=torch.bfloat16, device_map=device,
    attn_implementation='eager', low_cpu_mem_usage=True,
)
model = PeftModel.from_pretrained(model, lora_dir)
model.eval()
model = model.merge_and_unload()
print('[L3 v3] LoRA-merged model loaded', flush=True)

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

# Smoke
data_smoke = json.load(open('/tmp/MetaTool/dataset/tmp_dataset/Task2-Subtask4.json'))[:20]
print(f'[L3 v3 smoke] N={len(data_smoke)}', flush=True)
for method in ['no_steer', 'ocq_qbias_b-0.1', 'ocq_bias_a0.3']:
    r = run_method(model, tok, data_smoke, method, args, B_ont, n_kv, head_dim, facet_map=None)
    m = r['macro']
    print(f\"[L3 v3 smoke] {method:30s} F1={m['F1']:.3f} rec={m['recall']:.3f}\", flush=True)
    json.dump({'method': method, 'lora': lora_dir, 'results': [r]},
              open(f'$O_DIR/l3v3_smoke_{method}.json','w'), indent=2)

# Full 497 — only run if smoke shows transfer (no_steer F1 > 0.5)
data_full = json.load(open('/tmp/MetaTool/dataset/tmp_dataset/Task2-Subtask4.json'))
print(f'[L3 v3 full] N={len(data_full)}', flush=True)
for method in ['no_steer', 'ocq_qbias_b-0.1']:
    r = run_method(model, tok, data_full, method, args, B_ont, n_kv, head_dim, facet_map=None)
    m = r['macro']
    print(f\"[L3 v3 full] {method:30s} F1={m['F1']:.3f} rec={m['recall']:.3f}\", flush=True)
    json.dump({'method': method, 'lora': lora_dir, 'n_queries': len(data_full),
               'results': [r]},
              open(f'$O_DIR/l3v3_full_{method}.json','w'), indent=2)
" > "$L_DIR/l3v3_eval.log" 2>&1
    echo "[g0p2] L3'' done $(date)" >> "$LOG_BASE/chain.log"
  else
    echo "[g0p2] LoRA v3 train failed (rc=$RC), skipping L3''" >> "$LOG_BASE/chain.log"
  fi

  echo "[g0p2] GPU0 chain COMPLETE $(date)" >> "$LOG_BASE/chain.log"
) &
G0_CHAIN_PID=$!

# === GPU1 chain: small-α joint sweep + V-only full + layer-ablation + Llama Q on Subtask1 ===
(
  L_DIR=$LOG_BASE/gpu1
  O_DIR=$OUT_BASE/gpu1

  # Wave A: K×Q small-α joint sweep (tests refined Thm 6.17′ region)
  echo "[g1p2] small-α joint smoke $(date)" >> "$LOG_BASE/chain.log"
  python scripts/ocq/eval_metatool_subtask4.py \
      --model "$QWEN" --device cuda:1 \
      --methods no_steer ocq_qkv_a0.05_v0_q-0.1 ocq_qkv_a0.05_v0.05_q-0.1 ocq_qkv_a0.1_v0_q-0.05 ocq_qkv_a0_v0.05_q-0.1 \
      --b-ont "$BONT_QWEN" \
      --max-samples 20 \
      --out "$O_DIR/qwen_st4_qkv_smallA_smoke.json" \
      > "$L_DIR/qkv_smallA_smoke.log" 2>&1
  echo "[g1p2] small-α smoke done $(date)" >> "$LOG_BASE/chain.log"

  # Wave B: V-only full 497 (was only smoke before)
  echo "[g1p2] V-only full $(date)" >> "$LOG_BASE/chain.log"
  python scripts/ocq/eval_metatool_subtask4.py \
      --model "$QWEN" --device cuda:1 \
      --methods no_steer ocq_vbias_a0.1 ocq_vbias_a0.3 \
      --b-ont "$BONT_QWEN" \
      --out "$O_DIR/qwen_st4_vbias_full497.json" \
      > "$L_DIR/vbias_full.log" 2>&1
  echo "[g1p2] V-only full done $(date)" >> "$LOG_BASE/chain.log"

  # Wave C: Q-coverage on Llama Subtask1 full (cross-model single-tool)
  echo "[g1p2] Llama st1 Q full $(date)" >> "$LOG_BASE/chain.log"
  python scripts/ocq/eval_metatool_subtask1.py \
      --model "$LLAMA_INST" --device cuda:1 \
      --methods no_steer ocq_qbias_b-0.1 ocq_qbias_b-0.3 ocq_bias_a0.3 \
      --b-ont "$BONT_LLAMA" \
      --max-new-tokens 32 \
      --out "$O_DIR/llama_st1_qbias_kbias_full995.json" \
      > "$L_DIR/llama_st1.log" 2>&1
  echo "[g1p2] Llama st1 done $(date)" >> "$LOG_BASE/chain.log"

  # Wave D: Q-coverage cross-model Llama Subtask4 full (NOT in PM wave 1)
  echo "[g1p2] Llama st4 Q-only full $(date)" >> "$LOG_BASE/chain.log"
  python scripts/ocq/eval_metatool_subtask4.py \
      --model "$LLAMA_INST" --device cuda:1 \
      --methods no_steer ocq_qbias_b-0.1 ocq_qbias_b-0.3 \
      --b-ont "$BONT_LLAMA" \
      --out "$O_DIR/llama_st4_qbias_full497.json" \
      > "$L_DIR/llama_st4_q.log" 2>&1
  echo "[g1p2] Llama st4 Q done $(date)" >> "$LOG_BASE/chain.log"

  # Wave E (NEW 2026-04-15 13:00): V+Q joint full 497 — verify smoke F1=0.658
  # at full scale. (V+Q smoke matched Q-only +10.8pp; if also matches at full,
  # represents partial Thm 6.17 (a)+(c) verification with K excluded.)
  echo "[g1p2] V+Q joint full 497 sweep $(date)" >> "$LOG_BASE/chain.log"
  python scripts/ocq/eval_metatool_subtask4.py \
      --model "$QWEN" --device cuda:1 \
      --methods no_steer ocq_qkv_a0_v0.1_q-0.1 ocq_qkv_a0_v0.05_q-0.1 ocq_qkv_a0_v0.1_q-0.05 ocq_qkv_a0_v0.2_q-0.1 \
      --b-ont "$BONT_QWEN" \
      --out "$O_DIR/qwen_st4_VplusQ_full497.json" \
      > "$L_DIR/VplusQ_full.log" 2>&1
  echo "[g1p2] V+Q full done $(date)" >> "$LOG_BASE/chain.log"

  # Wave F (NEW): cross-model V+Q smoke on Llama-Inst (10 min)
  echo "[g1p2] Llama V+Q smoke $(date)" >> "$LOG_BASE/chain.log"
  python scripts/ocq/eval_metatool_subtask4.py \
      --model "$LLAMA_INST" --device cuda:1 \
      --methods no_steer ocq_qkv_a0_v0.1_q-0.1 \
      --b-ont "$BONT_LLAMA" \
      --max-samples 20 \
      --out "$O_DIR/llama_st4_VplusQ_smoke.json" \
      > "$L_DIR/llama_VplusQ_smoke.log" 2>&1
  echo "[g1p2] Llama V+Q smoke done $(date)" >> "$LOG_BASE/chain.log"

  # Wave G (NEW): Q-coverage with HIGHER β scan on Subtask1 — does Subtask1
  # need different β than Subtask4? (single-tool task may favor different mag)
  echo "[g1p2] Subtask1 Q-coverage β-sweep $(date)" >> "$LOG_BASE/chain.log"
  python scripts/ocq/eval_metatool_subtask1.py \
      --model "$QWEN" --device cuda:1 \
      --methods no_steer ocq_qbias_b-0.05 ocq_qbias_b-0.1 ocq_qbias_b-0.3 \
      --b-ont "$BONT_QWEN" \
      --max-new-tokens 32 \
      --out "$O_DIR/qwen_st1_qbias_sweep_full995.json" \
      > "$L_DIR/qwen_st1_qbias_sweep.log" 2>&1
  echo "[g1p2] Subtask1 Q sweep done $(date)" >> "$LOG_BASE/chain.log"

  echo "[g1p2] GPU1 chain COMPLETE $(date)" >> "$LOG_BASE/chain.log"
) &
G1_CHAIN_PID=$!

wait $G0_CHAIN_PID $G1_CHAIN_PID
echo "[pm2] both chains complete $(date)" >> "$LOG_BASE/chain.log"

# Final summary across both chains
python3 -c "
import json, glob, os
print('=== PM Wave 2 summary ===')
for p in sorted(glob.glob('$OUT_BASE/**/*.json', recursive=True)):
    try:
        d = json.load(open(p))
        for r in d.get('results', []):
            m = r.get('macro', r)
            tag = os.path.relpath(p, '$OUT_BASE')
            if isinstance(m, dict) and 'F1' in m:
                print(f\"{tag:55s} {r['method']:30s} F1={m['F1']:.3f} rec={m.get('recall',0):.3f}\")
            elif 'top1_accuracy' in r:
                print(f\"{tag:55s} {r['method']:30s} top1={r['top1_accuracy']*100:.2f}%\")
    except Exception as e:
        print(f'{p}: ERROR {e}')
" > "$LOG_BASE/SUMMARY.txt" 2>&1
echo "[pm2] WAVE 2 COMPLETE $(date)" >> "$LOG_BASE/chain.log"
