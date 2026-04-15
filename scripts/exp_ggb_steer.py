#!/usr/bin/env python3
"""
exp_ggb_steer — Golden Gate Bridge facet steering experiment.

Reproduces the spirit of Anthropic's "Golden Gate Claude" experiment using
facet-orthogonal steering on a frozen LLM (Mistral-7B). Instead of training
a multi-million-parameter SAE on residual streams, we use a tiny ontology
(6 landmarks × 8 sentences) and a single offline pass to construct a
direction v_GGB in K-space per (layer, kv_head). Then we steer by adding
β · v_GGB to query vectors at chosen layers.

Hypothesis: any prompt, regardless of topic, will be biased toward
Golden Gate Bridge content as β grows. Compositional steering across
multiple landmarks should be cleaner than the SAE version because the
facet basis is orthogonal by construction.

Pipeline:
  1. Build LANDMARK ontology, extract per-category mean K vectors per
     (layer, kv_head), build facet basis B^(ℓ,h).
  2. Identify the GGB direction v_GGB^(ℓ,h) within B^(ℓ,h).
  3. Install Q-side forward hooks: q' = q + β · v_GGB at chosen layers.
  4. Generate text on a fixed prompt set, with and without steering, at
     several β values. Record:
        - GGB keyword frequency in generation
        - sample generations
        - per-prompt change vs baseline
  5. Save results.
"""

import os
import json
import time
import re

os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

DTYPE = torch.bfloat16
MODEL_ID = 'mistralai/Mistral-7B-v0.3'
OUT = Path(
    '/home/woori/workspace_common/boltzmann-attention/reports/'
    'axis2_theoretical_verification/exp_ggb_steer.json'
)

# Steering parameters
STEER_LAYERS = [7, 15, 23]                  # mid layers (Anthropic-style)
BETA_VALUES = [0.0, 1.0, 3.0, 6.0, 12.0]    # steering strength sweep
MAX_NEW_TOKENS = 80
GEN_DO_SAMPLE = False                        # greedy for reproducibility


# =====================================================================
# 1. Landmark ontology
# =====================================================================

LANDMARK_ONTOLOGY = {
    'landmark': {
        'Golden Gate Bridge': [
            'The Golden Gate Bridge spans the entrance to San Francisco Bay.',
            'Tourists photograph the Golden Gate Bridge from Marin Headlands.',
            'The Golden Gate Bridge is painted in International Orange.',
            'Engineer Joseph Strauss led the Golden Gate Bridge construction.',
            'Cyclists ride across the Golden Gate Bridge on weekend mornings.',
            'The Golden Gate Bridge opened to traffic in May 1937.',
            'Fog often shrouds the Golden Gate Bridge in summer months.',
            'The Golden Gate Bridge connects San Francisco to Sausalito.',
        ],
        'Eiffel Tower': [
            'The Eiffel Tower stands in the Champ de Mars in Paris.',
            'Visitors climb the Eiffel Tower for views of the river Seine.',
            'Gustave Eiffel designed the Eiffel Tower for the 1889 exhibition.',
            'The Eiffel Tower is illuminated every evening in Paris.',
            'Tourists ride the elevators of the Eiffel Tower to the top.',
            'The Eiffel Tower is the tallest structure in central Paris.',
            'Wedding photos are often taken at the Eiffel Tower.',
            'The Eiffel Tower has three observation levels for visitors.',
        ],
        'Statue of Liberty': [
            'The Statue of Liberty stands on Liberty Island in New York.',
            'France gave the Statue of Liberty to the United States in 1886.',
            'Visitors take a ferry to see the Statue of Liberty up close.',
            'The Statue of Liberty holds a torch and a tablet.',
            'Frederic Auguste Bartholdi sculpted the Statue of Liberty.',
            'The Statue of Liberty welcomes ships entering New York Harbor.',
            'Tourists climb to the crown of the Statue of Liberty.',
            'The Statue of Liberty is a symbol of freedom and democracy.',
        ],
        'Mount Fuji': [
            'Mount Fuji is the highest mountain in Japan.',
            'Climbers hike to the summit of Mount Fuji every summer.',
            'Mount Fuji is a sacred site in Japanese culture.',
            'The snow-capped peak of Mount Fuji is visible from Tokyo.',
            'Artists have painted Mount Fuji for centuries in ukiyo-e prints.',
            'Mount Fuji last erupted in the early eighteenth century.',
            'The five lakes around Mount Fuji attract many tourists.',
            'Mount Fuji rises gracefully above the surrounding plains.',
        ],
        'Big Ben': [
            'Big Ben is the bell of the clock at the Palace of Westminster.',
            'The Big Ben clock tower is located in central London.',
            'Tourists hear Big Ben chime every hour in London.',
            'Big Ben is one of the most recognizable landmarks in London.',
            'The Big Ben clock has four faces facing each direction.',
            'Big Ben underwent renovation for several years recently.',
            'The Big Ben tower stands beside the river Thames.',
            'Visitors photograph Big Ben from Westminster Bridge.',
        ],
        'Pyramids of Giza': [
            'The Pyramids of Giza stand on the Giza plateau in Egypt.',
            'Ancient Egyptians built the Pyramids of Giza as royal tombs.',
            'The Great Pyramid of Giza is the largest of the three.',
            'Tourists explore the Pyramids of Giza near Cairo.',
            'The Pyramids of Giza are wonders of the ancient world.',
            'Pharaoh Khufu commissioned the Great Pyramid of Giza.',
            'Camels carry visitors around the Pyramids of Giza.',
            'The Sphinx sits beside the Pyramids of Giza.',
        ],
    }
}

GGB_KEYWORDS = [
    'golden gate', 'san francisco', 'bay area', 'marin', 'sausalito',
    'international orange', 'suspension bridge', 'fog', 'bridge',
]

TEST_PROMPTS = {
    # Unrelated to GGB — should baseline-answer normally,
    # then drift toward GGB as β grows
    'weather':    'What is the weather like in Tokyo today?',
    'recipe':     'Give me a simple recipe for chocolate chip cookies.',
    'math':       'Explain how addition works for small children.',
    'history':    'Who was the first president of the United States?',
    'science':    'How does photosynthesis work in plants?',
    # Related to GGB or California — should already mention something
    'travel_sf':  'I am visiting San Francisco for the first time. What should I see?',
    'control_paris': 'I am visiting Paris next month. What landmarks should I see?',
}


# =====================================================================
# 2. Build facet basis (reuse Day 1 logic, single facet)
# =====================================================================

@torch.no_grad()
def extract_landmark_K(model, tok, ontology, target_layers, n_kv, head_dim):
    n_layers = model.config.num_hidden_layers
    pl_buf = [None] * n_layers

    handles = []
    for li in target_layers:
        layer = model.model.layers[li]

        def make_hook(li_):
            def h(m, inp, out):
                pl_buf[li_] = out.detach().cpu().float().numpy()
            return h

        handles.append(layer.self_attn.k_proj.register_forward_hook(make_hook(li)))

    out = {}
    for facet_name, cats in ontology.items():
        for cat_name, sentences in cats.items():
            sentence_K = {(li, h): [] for li in target_layers for h in range(n_kv)}
            for sent in sentences:
                ids = tok(sent, return_tensors='pt')['input_ids'].to('cuda:0')
                model(ids, use_cache=False)
                for li in target_layers:
                    K_li = pl_buf[li]
                    if K_li is None:
                        continue
                    K_li = K_li[0]
                    K_li = K_li.reshape(K_li.shape[0], n_kv, head_dim)
                    if K_li.shape[0] > 1:
                        K_li = K_li[1:]
                    for h in range(n_kv):
                        sentence_K[(li, h)].append(K_li[:, h, :].mean(0))
            for (li, h), v_list in sentence_K.items():
                if v_list:
                    out[(facet_name, cat_name, li, h)] = (
                        np.stack(v_list).mean(0).astype(np.float64)
                    )

    for hd in handles:
        hd.remove()
    return out


def build_v_ggb_per_kv_head(cat_K, target_layers, n_kv, head_dim,
                            target_category='Golden Gate Bridge',
                            facet_name='landmark'):
    """For each (layer, kv_head), compute v_GGB:
       the mean K vector of GGB MINUS the mean of all *other* landmarks.

       This contrast vector points 'toward GGB and away from the average
       landmark', which is what we want for steering.
    """
    cats = list(LANDMARK_ONTOLOGY[facet_name].keys())
    other_cats = [c for c in cats if c != target_category]

    v_ggb = {}
    for li in target_layers:
        for h in range(n_kv):
            mu_ggb = cat_K.get((facet_name, target_category, li, h))
            if mu_ggb is None:
                continue
            mu_other = np.mean(
                [cat_K[(facet_name, c, li, h)] for c in other_cats], axis=0
            )
            v = mu_ggb - mu_other
            n = np.linalg.norm(v)
            if n > 1e-12:
                v = v / n
            v_ggb[(li, h)] = v.astype(np.float32)
    return v_ggb


# =====================================================================
# 3. Q-side steering hook
# =====================================================================

class QSteerHook:
    """Adds β · v_GGB to query at a chosen layer.

    GQA: n_q query heads share n_kv KV heads. Each Q head i belongs to
    KV group g(i) = i // (n_q // n_kv). We use v_GGB[(layer, g(i))] for
    Q head i. All heads in the same group get the same direction.
    """

    def __init__(self, layer_idx, n_q, n_kv, head_dim, v_ggb_layer, beta):
        self.layer_idx = layer_idx
        self.n_q = n_q
        self.n_kv = n_kv
        self.head_dim = head_dim
        self.beta = beta
        # Precompute (n_q, head_dim) bias matrix
        group_size = n_q // n_kv
        bias = np.zeros((n_q, head_dim), dtype=np.float32)
        for i in range(n_q):
            kv_head = i // group_size
            if kv_head in v_ggb_layer:
                bias[i] = v_ggb_layer[kv_head]
        self.bias = bias  # numpy
        self._bias_torch = None

    def __call__(self, module, inputs, output):
        x = output  # (B, T, n_q*head_dim)
        if self._bias_torch is None or self._bias_torch.device != x.device:
            self._bias_torch = torch.from_numpy(self.bias).to(
                x.device, x.dtype
            )
        B, T, D = x.shape
        x4 = x.reshape(B, T, self.n_q, self.head_dim)
        # Add β * v_GGB broadcast over (B, T)
        x4 = x4 + self.beta * self._bias_torch[None, None, :, :]
        return x4.reshape(B, T, D).contiguous()


def install_steer_hooks(model, steer_layers, v_ggb, beta, n_q, n_kv, head_dim):
    handles = []
    for li in steer_layers:
        v_layer = {h: v_ggb[(li, h)] for h in range(n_kv) if (li, h) in v_ggb}
        if not v_layer:
            continue
        hook = QSteerHook(li, n_q, n_kv, head_dim, v_layer, beta)
        handle = model.model.layers[li].self_attn.q_proj.register_forward_hook(hook)
        handles.append(handle)
    return handles


# =====================================================================
# 4. Generation + scoring
# =====================================================================

@torch.no_grad()
def generate_text(model, tok, prompt, max_new_tokens=MAX_NEW_TOKENS):
    ids = tok(prompt, return_tensors='pt')['input_ids'].to('cuda:0')
    out = model.generate(
        ids,
        max_new_tokens=max_new_tokens,
        do_sample=GEN_DO_SAMPLE,
        num_beams=1,
        pad_token_id=tok.eos_token_id,
        repetition_penalty=1.15,
    )
    full = tok.decode(out[0], skip_special_tokens=True)
    if full.startswith(prompt):
        return full[len(prompt):].strip()
    return full.strip()


def count_ggb_keywords(text):
    text_low = text.lower()
    counts = {}
    total = 0
    for kw in GGB_KEYWORDS:
        c = len(re.findall(re.escape(kw), text_low))
        counts[kw] = c
        total += c
    return total, counts


# =====================================================================
# 5. Main
# =====================================================================

def main():
    print("=" * 72)
    print("exp_ggb_steer — Golden Gate Bridge facet steering")
    print("=" * 72)
    print(f"  model        : {MODEL_ID}")
    print(f"  steer layers : {STEER_LAYERS}")
    print(f"  beta sweep   : {BETA_VALUES}")
    print(f"  prompts      : {len(TEST_PROMPTS)}")
    print()

    print("Loading model ...", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=DTYPE, device_map='cuda:0',
        attn_implementation='eager', low_cpu_mem_usage=True,
    )
    model.eval()
    cfg = model.config
    n_layers = cfg.num_hidden_layers
    n_q = cfg.num_attention_heads
    n_kv = cfg.num_key_value_heads
    head_dim = (
        getattr(cfg, 'head_dim', None)
        or (cfg.hidden_size // n_q)
    )
    print(f"  loaded in {time.time()-t0:.1f}s: "
          f"n_layers={n_layers}  n_q={n_q}  n_kv={n_kv}  head_dim={head_dim}",
          flush=True)

    # Need K vectors at all layers we'll possibly steer
    print("\n[1/3] Extracting per-landmark K vectors ...", flush=True)
    t0 = time.time()
    cat_K = extract_landmark_K(
        model, tok, LANDMARK_ONTOLOGY, STEER_LAYERS, n_kv, head_dim
    )
    print(f"  extracted {len(cat_K)} entries in {time.time()-t0:.1f}s",
          flush=True)

    print("\n[2/3] Building v_GGB per (layer, kv_head) ...", flush=True)
    v_ggb = build_v_ggb_per_kv_head(cat_K, STEER_LAYERS, n_kv, head_dim)
    # Sanity: norm of v_GGB difference vector
    norm_diffs = []
    for li in STEER_LAYERS:
        for h in range(n_kv):
            if (li, h) in v_ggb:
                norm_diffs.append(np.linalg.norm(
                    cat_K[('landmark', 'Golden Gate Bridge', li, h)]
                    - np.mean([
                        cat_K[('landmark', c, li, h)]
                        for c in list(LANDMARK_ONTOLOGY['landmark'].keys())
                        if c != 'Golden Gate Bridge'
                    ], axis=0)
                ))
    print(f"  raw GGB-vs-others contrast magnitude: "
          f"mean={np.mean(norm_diffs):.4f}  max={np.max(norm_diffs):.4f}  "
          f"min={np.min(norm_diffs):.4f}", flush=True)

    print("\n[3/3] Generating with steering sweep ...", flush=True)
    results = {}
    for beta in BETA_VALUES:
        print(f"\n  --- β = {beta} ---", flush=True)
        # Install hooks
        handles = install_steer_hooks(
            model, STEER_LAYERS, v_ggb, beta, n_q, n_kv, head_dim
        )
        beta_results = {}
        total_kw_count = 0
        for prompt_name, prompt in TEST_PROMPTS.items():
            try:
                gen = generate_text(model, tok, prompt)
            except Exception as e:
                gen = f"<GENERATION ERROR: {e}>"
            kw_total, kw_counts = count_ggb_keywords(gen)
            total_kw_count += kw_total
            beta_results[prompt_name] = {
                'prompt': prompt,
                'generation': gen,
                'ggb_keyword_count': kw_total,
                'kw_breakdown': {k: v for k, v in kw_counts.items() if v > 0},
            }
            preview = gen[:120].replace('\n', ' ')
            print(f"    [{prompt_name:>14}] kw={kw_total}  "
                  f"\"{preview}{'...' if len(gen) > 120 else ''}\"",
                  flush=True)
        for hdl in handles:
            hdl.remove()
        results[f'beta_{beta}'] = {
            'beta': beta,
            'total_ggb_keyword_count': total_kw_count,
            'per_prompt': beta_results,
        }
        print(f"  Σ keywords across prompts: {total_kw_count}", flush=True)

    # Per-prompt baseline diff
    baseline = results['beta_0.0']['per_prompt']
    deltas = {}
    for beta_key, br in results.items():
        if beta_key == 'beta_0.0':
            continue
        deltas[beta_key] = sum(
            br['per_prompt'][p]['ggb_keyword_count']
            - baseline[p]['ggb_keyword_count']
            for p in TEST_PROMPTS
        )
    print(f"\n  GGB keyword delta vs baseline:")
    for k, v in deltas.items():
        print(f"    {k}: +{v}")

    out = {
        'model': MODEL_ID,
        'steer_layers': STEER_LAYERS,
        'beta_values': BETA_VALUES,
        'max_new_tokens': MAX_NEW_TOKENS,
        'do_sample': GEN_DO_SAMPLE,
        'ggb_keywords': GGB_KEYWORDS,
        'landmarks': list(LANDMARK_ONTOLOGY['landmark'].keys()),
        'n_categories': sum(len(c) for c in LANDMARK_ONTOLOGY['landmark'].values()),
        'results_per_beta': results,
        'keyword_delta_vs_baseline': deltas,
    }
    OUT.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwrote {OUT}")


if __name__ == '__main__':
    main()
