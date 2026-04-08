#!/usr/bin/env python3
"""
exp_ggb_residual_steer — Golden Gate Bridge facet steering via RESIDUAL STREAM
injection (closer to SAE clamping than the Q-bias version).

Companion to exp_ggb_steer.py. The previous experiment showed that Q-side
attention bias cannot inject content into unrelated prompts because it can
only AMPLIFY existing context — there are no GGB-aligned tokens to attend
to in a math/recipe/weather prompt. SAE clamping works by INJECTING into
residual stream directly. This script reproduces that mechanism using a
facet-derived direction instead of a learned SAE feature.

Pipeline:
  1. Extract per-landmark mean RESIDUAL STREAM vector at chosen layers
     (hidden_size = 4096 for Mistral-7B, not head_dim = 128).
  2. Build v_GGB^resid = mean(GGB) - mean(other landmarks), unit-normalized.
  3. Forward-hook on each chosen layer's output to add β · v_GGB^resid to
     the residual stream at all token positions.
  4. Generate text on the same prompts as exp_ggb_steer.py and count GGB
     keywords. Direct comparison to Q-bias results.
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
    'axis2_theoretical_verification/exp_ggb_residual_steer.json'
)

STEER_LAYERS = [7, 15, 23]
BETA_VALUES = [0.0, 1.0, 3.0, 6.0, 12.0]
MAX_NEW_TOKENS = 80
GEN_DO_SAMPLE = False


# =====================================================================
# Same ontology + prompts as exp_ggb_steer.py for direct comparison
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
    'weather':       'What is the weather like in Tokyo today?',
    'recipe':        'Give me a simple recipe for chocolate chip cookies.',
    'math':          'Explain how addition works for small children.',
    'history':       'Who was the first president of the United States?',
    'science':       'How does photosynthesis work in plants?',
    'travel_sf':     'I am visiting San Francisco for the first time. What should I see?',
    'control_paris': 'I am visiting Paris next month. What landmarks should I see?',
}


# =====================================================================
# 1. Extract per-category residual stream vectors
# =====================================================================

def _unwrap(out):
    """Decoder layers may return tensor or tuple — return the hidden states."""
    if isinstance(out, tuple):
        return out[0]
    return out


@torch.no_grad()
def extract_landmark_residual(model, tok, ontology, target_layers, d_model):
    """Capture per-(layer, category) mean residual stream vectors.

    Hooks the FULL transformer block, captures its output (hidden states
    after the block, i.e. the updated residual stream), and averages over
    content tokens (excluding BOS) and across sentences for each category.
    """
    n_layers = model.config.num_hidden_layers
    pl_buf = [None] * n_layers

    handles = []
    for li in target_layers:
        layer = model.model.layers[li]

        def make_hook(li_):
            def h(m, inp, out):
                h_ = _unwrap(out)
                pl_buf[li_] = h_.detach().cpu().float().numpy()
            return h

        handles.append(layer.register_forward_hook(make_hook(li)))

    out = {}
    for facet_name, cats in ontology.items():
        for cat_name, sentences in cats.items():
            sentence_h = {li: [] for li in target_layers}
            for sent in sentences:
                ids = tok(sent, return_tensors='pt')['input_ids'].to('cuda:0')
                model(ids, use_cache=False)
                for li in target_layers:
                    h_li = pl_buf[li]
                    if h_li is None:
                        continue
                    h_li = h_li[0]  # (T, d_model)
                    if h_li.shape[0] > 1:
                        h_li = h_li[1:]  # exclude BOS
                    sentence_h[li].append(h_li.mean(0))
            for li, v_list in sentence_h.items():
                if v_list:
                    out[(facet_name, cat_name, li)] = (
                        np.stack(v_list).mean(0).astype(np.float64)
                    )

    for hd in handles:
        hd.remove()
    return out


def build_v_ggb_residual(cat_h, target_layers,
                          target_category='Golden Gate Bridge',
                          facet_name='landmark'):
    cats = list(LANDMARK_ONTOLOGY[facet_name].keys())
    other_cats = [c for c in cats if c != target_category]
    v_ggb = {}
    raw_norms = []
    for li in target_layers:
        mu_ggb = cat_h.get((facet_name, target_category, li))
        if mu_ggb is None:
            continue
        mu_other = np.mean(
            [cat_h[(facet_name, c, li)] for c in other_cats], axis=0
        )
        v = mu_ggb - mu_other
        n = np.linalg.norm(v)
        raw_norms.append(n)
        if n > 1e-12:
            v = v / n
        v_ggb[li] = v.astype(np.float32)
    return v_ggb, raw_norms


# =====================================================================
# 2. Residual stream steering hook
# =====================================================================

class ResidualSteerHook:
    """Adds β · v_GGB to the residual stream at all token positions."""

    def __init__(self, layer_idx, v_residual, beta):
        self.layer_idx = layer_idx
        self.v = v_residual.astype(np.float32)
        self.beta = beta
        self._v_torch = None

    def __call__(self, module, inputs, output):
        h = _unwrap(output)
        if self._v_torch is None or self._v_torch.device != h.device:
            self._v_torch = torch.from_numpy(self.v).to(h.device, h.dtype)
        h_new = h + self.beta * self._v_torch[None, None, :]
        if isinstance(output, tuple):
            return (h_new,) + output[1:]
        return h_new


def install_residual_steer(model, steer_layers, v_ggb, beta):
    handles = []
    for li in steer_layers:
        if li not in v_ggb:
            continue
        hook = ResidualSteerHook(li, v_ggb[li], beta)
        handle = model.model.layers[li].register_forward_hook(hook)
        handles.append(handle)
    return handles


# =====================================================================
# 3. Generation + scoring (same as Q-bias version)
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
# 4. Main
# =====================================================================

def main():
    print("=" * 72)
    print("exp_ggb_residual_steer — residual stream injection variant")
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
    d_model = cfg.hidden_size
    print(f"  loaded in {time.time()-t0:.1f}s: "
          f"n_layers={n_layers}  d_model={d_model}",
          flush=True)

    print("\n[1/3] Extracting per-landmark residual vectors ...", flush=True)
    t0 = time.time()
    cat_h = extract_landmark_residual(
        model, tok, LANDMARK_ONTOLOGY, STEER_LAYERS, d_model
    )
    print(f"  extracted {len(cat_h)} entries in {time.time()-t0:.1f}s",
          flush=True)

    print("\n[2/3] Building v_GGB^resid per layer ...", flush=True)
    v_ggb, raw_norms = build_v_ggb_residual(cat_h, STEER_LAYERS)
    for li, n in zip(STEER_LAYERS, raw_norms):
        print(f"  L{li}: raw GGB-vs-others contrast norm = {n:.4f}",
              flush=True)

    print("\n[3/3] Generating with residual steering sweep ...", flush=True)
    results = {}
    for beta in BETA_VALUES:
        print(f"\n  --- β = {beta} ---", flush=True)
        handles = install_residual_steer(model, STEER_LAYERS, v_ggb, beta)
        beta_results = {}
        total_kw_count = 0
        for prompt_name, prompt in TEST_PROMPTS.items():
            try:
                gen = generate_text(model, tok, prompt)
            except Exception as e:
                gen = f"<GEN ERROR: {e}>"
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

        # Save partial after each beta in case of crash
        partial = {
            'model': MODEL_ID,
            'steer_layers': STEER_LAYERS,
            'beta_values': BETA_VALUES,
            'partial': True,
            'results_per_beta': results,
        }
        OUT.write_text(json.dumps(partial, indent=2, default=float))

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
        print(f"    {k}: {v:+d}")

    out = {
        'model': MODEL_ID,
        'mechanism': 'residual_stream_injection',
        'steer_layers': STEER_LAYERS,
        'beta_values': BETA_VALUES,
        'max_new_tokens': MAX_NEW_TOKENS,
        'do_sample': GEN_DO_SAMPLE,
        'd_model': d_model,
        'ggb_keywords': GGB_KEYWORDS,
        'landmarks': list(LANDMARK_ONTOLOGY['landmark'].keys()),
        'raw_contrast_norms_per_layer': dict(zip(STEER_LAYERS, raw_norms)),
        'results_per_beta': results,
        'keyword_delta_vs_baseline': deltas,
    }
    OUT.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwrote {OUT}")


if __name__ == '__main__':
    main()
