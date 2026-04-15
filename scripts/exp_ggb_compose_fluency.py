#!/usr/bin/env python3
"""
exp_ggb_compose_fluency — Compositional facet steering + fluency degradation.

Two experiments in one script:

  (#3) Compositional steering: build v_GGB and v_Eiffel from the same landmark
       facet ontology, then test single vs joint steering. If facets are truly
       orthogonal in residual stream, joint = single ⊕ single with no
       interference (cleaner than SAE).

  (#5) Fluency degradation curve: for each steering config, measure PPL on a
       neutral held-out text (WikiText-2 test). The key trade-off curve is
       (steering strength) vs (quality loss).

Builds on exp_ggb_residual_steer.py — same residual injection mechanism,
different vector(s) and metrics.
"""

import os
import json
import time
import re

os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

DTYPE = torch.bfloat16
MODEL_ID = 'mistralai/Mistral-7B-v0.3'
OUT = Path(
    '/home/woori/workspace_common/boltzmann-attention/reports/'
    'axis2_theoretical_verification/exp_ggb_compose_fluency.json'
)

STEER_LAYERS = [7, 15, 23]
MAX_NEW_TOKENS = 80
PPL_LENGTH = 512  # Neutral text length for fluency measurement


# =====================================================================
# Same ontology
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
    'international orange', 'suspension bridge', 'fog',
]
EIFFEL_KEYWORDS = [
    'eiffel', 'paris', 'champ de mars', 'seine', 'gustave',
    'champ-de-mars', 'parisian',
]

NEUTRAL_PROMPTS = {
    'weather':       'What is the weather like in Tokyo today?',
    'recipe':        'Give me a simple recipe for chocolate chip cookies.',
    'math':          'Explain how addition works for small children.',
    'history':       'Who was the first president of the United States?',
    'control_paris': 'I am visiting Paris next month. What landmarks should I see?',
}


# =====================================================================
# Helpers (same shape as exp_ggb_residual_steer.py)
# =====================================================================

def _unwrap(out):
    if isinstance(out, tuple):
        return out[0]
    return out


@torch.no_grad()
def extract_landmark_residual(model, tok, target_layers):
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
    for facet_name, cats in LANDMARK_ONTOLOGY.items():
        for cat_name, sentences in cats.items():
            sentence_h = {li: [] for li in target_layers}
            for sent in sentences:
                ids = tok(sent, return_tensors='pt')['input_ids'].to('cuda:0')
                model(ids, use_cache=False)
                for li in target_layers:
                    h_li = pl_buf[li]
                    if h_li is None:
                        continue
                    h_li = h_li[0]
                    if h_li.shape[0] > 1:
                        h_li = h_li[1:]
                    sentence_h[li].append(h_li.mean(0))
            for li, v_list in sentence_h.items():
                if v_list:
                    out[(cat_name, li)] = (
                        np.stack(v_list).mean(0).astype(np.float64)
                    )
    for hd in handles:
        hd.remove()
    return out


def build_landmark_vector(cat_h, target_layers, target_category):
    """For each layer: v = mu_target - mean(other landmarks), unit-normalized."""
    cats = list(LANDMARK_ONTOLOGY['landmark'].keys())
    other_cats = [c for c in cats if c != target_category]
    v_per_layer = {}
    for li in target_layers:
        mu_t = cat_h.get((target_category, li))
        if mu_t is None:
            continue
        mu_other = np.mean([cat_h[(c, li)] for c in other_cats], axis=0)
        v = mu_t - mu_other
        n = np.linalg.norm(v)
        if n > 1e-12:
            v = v / n
        v_per_layer[li] = v.astype(np.float32)
    return v_per_layer


def cosine_similarity(v_a, v_b):
    """Cosine between two layer-keyed dicts of unit vectors, mean across layers."""
    sims = []
    for li in v_a:
        if li in v_b:
            sims.append(float(np.dot(v_a[li], v_b[li])))
    return float(np.mean(sims)), sims


# =====================================================================
# Steering hook supporting MULTIPLE direction sources
# =====================================================================

class MultiResidualSteerHook:
    """Adds Σ_i β_i · v_i to the residual stream at this layer."""

    def __init__(self, layer_idx, vectors_with_betas):
        """vectors_with_betas: list of (v_array, beta) pairs."""
        self.layer_idx = layer_idx
        self.vbs = vectors_with_betas
        # Pre-sum
        if vectors_with_betas:
            self.combined = sum(beta * v for v, beta in vectors_with_betas)
            self.combined = self.combined.astype(np.float32)
        else:
            self.combined = None
        self._t = None

    def __call__(self, module, inputs, output):
        if self.combined is None:
            return output
        h = _unwrap(output)
        if self._t is None or self._t.device != h.device:
            self._t = torch.from_numpy(self.combined).to(h.device, h.dtype)
        h_new = h + self._t[None, None, :]
        if isinstance(output, tuple):
            return (h_new,) + output[1:]
        return h_new


def install_multi(model, steer_layers, layer_to_vbs):
    handles = []
    for li in steer_layers:
        vbs = layer_to_vbs.get(li, [])
        if not vbs:
            continue
        hook = MultiResidualSteerHook(li, vbs)
        handle = model.model.layers[li].register_forward_hook(hook)
        handles.append(handle)
    return handles


# =====================================================================
# Generation + scoring
# =====================================================================

@torch.no_grad()
def generate_text(model, tok, prompt, max_new_tokens=MAX_NEW_TOKENS):
    ids = tok(prompt, return_tensors='pt')['input_ids'].to('cuda:0')
    out = model.generate(
        ids, max_new_tokens=max_new_tokens,
        do_sample=False, num_beams=1,
        pad_token_id=tok.eos_token_id,
        repetition_penalty=1.15,
    )
    full = tok.decode(out[0], skip_special_tokens=True)
    if full.startswith(prompt):
        return full[len(prompt):].strip()
    return full.strip()


def count_keywords(text, kws):
    text_low = text.lower()
    n = 0
    breakdown = {}
    for kw in kws:
        c = len(re.findall(re.escape(kw), text_low))
        if c > 0:
            breakdown[kw] = c
            n += c
    return n, breakdown


@torch.no_grad()
def compute_ppl(model, ids):
    out = model(ids, use_cache=False)
    logits = out.logits[:, :-1].contiguous()
    tgt = ids[:, 1:].contiguous()
    logits_flat = logits.reshape(-1, logits.size(-1)).float()
    tgt_flat = tgt.reshape(-1)
    loss = F.cross_entropy(logits_flat, tgt_flat, reduction='mean')
    return float(np.exp(float(loss.item())))


# =====================================================================
# Main
# =====================================================================

def main():
    print("=" * 72)
    print("exp_ggb_compose_fluency — composition + fluency degradation")
    print("=" * 72)
    print(f"  model        : {MODEL_ID}")
    print(f"  steer layers : {STEER_LAYERS}")
    print()

    print("Loading model ...", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=DTYPE, device_map='cuda:0',
        attn_implementation='eager', low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

    print("\n[1/5] Extract landmark residual vectors ...", flush=True)
    cat_h = extract_landmark_residual(model, tok, STEER_LAYERS)

    print("\n[2/5] Build v_GGB and v_Eiffel ...", flush=True)
    v_ggb = build_landmark_vector(cat_h, STEER_LAYERS, 'Golden Gate Bridge')
    v_eiffel = build_landmark_vector(cat_h, STEER_LAYERS, 'Eiffel Tower')
    cos_mean, cos_per_layer = cosine_similarity(v_ggb, v_eiffel)
    print(f"  ⟨v_GGB, v_Eiffel⟩ per layer: "
          f"{[f'{c:+.3f}' for c in cos_per_layer]}",
          flush=True)
    print(f"  mean cosine: {cos_mean:+.4f}  "
          f"(0 = perfectly orthogonal, ±1 = colinear)",
          flush=True)

    # Load neutral PPL text
    print("\n[3/5] Load WikiText-2 neutral text for PPL ...", flush=True)
    from datasets import load_dataset
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    test_text = '\n\n'.join(
        [t for t in ds['text'] if len(t.strip()) > 100][:30]
    )
    test_ids = tok(test_text, return_tensors='pt', truncation=True,
                   max_length=PPL_LENGTH)['input_ids'].to('cuda:0')
    print(f"  PPL text length: {test_ids.shape[1]} tokens", flush=True)

    # Define configurations
    print("\n[4/5] Steering configurations ...", flush=True)
    configs = {
        'baseline':           {},
        'ggb_b0.5':           {li: [(v_ggb[li], 0.5)] for li in STEER_LAYERS},
        'ggb_b1.0':           {li: [(v_ggb[li], 1.0)] for li in STEER_LAYERS},
        'ggb_b2.0':           {li: [(v_ggb[li], 2.0)] for li in STEER_LAYERS},
        'eiffel_b1.0':        {li: [(v_eiffel[li], 1.0)] for li in STEER_LAYERS},
        'eiffel_b2.0':        {li: [(v_eiffel[li], 2.0)] for li in STEER_LAYERS},
        'joint_b0.5_each':    {li: [(v_ggb[li], 0.5), (v_eiffel[li], 0.5)]
                               for li in STEER_LAYERS},
        'joint_b1.0_each':    {li: [(v_ggb[li], 1.0), (v_eiffel[li], 1.0)]
                               for li in STEER_LAYERS},
    }

    print("\n[5/5] Running configs ...", flush=True)
    results = {}
    for cfg_name, layer_to_vbs in configs.items():
        print(f"\n  --- {cfg_name} ---", flush=True)
        handles = install_multi(model, STEER_LAYERS, layer_to_vbs)

        # Fluency
        ppl = compute_ppl(model, test_ids)
        # Generations
        gens = {}
        ggb_total = 0
        eiffel_total = 0
        for pname, prompt in NEUTRAL_PROMPTS.items():
            try:
                gen = generate_text(model, tok, prompt)
            except Exception as e:
                gen = f"<ERR: {e}>"
            ggb_n, ggb_brk = count_keywords(gen, GGB_KEYWORDS)
            eif_n, eif_brk = count_keywords(gen, EIFFEL_KEYWORDS)
            gens[pname] = {
                'prompt': prompt,
                'generation': gen,
                'ggb_kw': ggb_n,
                'eiffel_kw': eif_n,
                'ggb_brk': ggb_brk,
                'eiffel_brk': eif_brk,
            }
            ggb_total += ggb_n
            eiffel_total += eif_n
        for hdl in handles:
            hdl.remove()

        results[cfg_name] = {
            'ppl_neutral': ppl,
            'ggb_total': ggb_total,
            'eiffel_total': eiffel_total,
            'generations': gens,
        }
        print(f"    PPL_neutral = {ppl:.3f}  "
              f"ΣGGB = {ggb_total}  ΣEiffel = {eiffel_total}",
              flush=True)
        for pname in NEUTRAL_PROMPTS:
            r = gens[pname]
            preview = r['generation'][:90].replace('\n', ' ')
            print(f"      [{pname:>14}] G={r['ggb_kw']} E={r['eiffel_kw']}  "
                  f"\"{preview}...\"",
                  flush=True)

        # Save partial after each config
        OUT.write_text(json.dumps({
            'partial': True,
            'configs_done': list(results.keys()),
            'results': results,
            'cosine_v_ggb_v_eiffel': {
                'mean': cos_mean,
                'per_layer': dict(zip(STEER_LAYERS, cos_per_layer)),
            },
        }, indent=2, default=float))

    # Summary
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"  ⟨v_GGB, v_Eiffel⟩ mean cosine: {cos_mean:+.4f}")
    print()
    print(f"  {'config':<22}{'PPL':>10}{'ΣGGB':>8}{'ΣEiffel':>10}")
    base_ppl = results['baseline']['ppl_neutral']
    for cfg_name, r in results.items():
        ppl = r['ppl_neutral']
        delta = ppl - base_ppl
        print(f"  {cfg_name:<22}{ppl:>10.3f}{r['ggb_total']:>8d}"
              f"{r['eiffel_total']:>10d}    (Δ={delta:+.3f})")

    out = {
        'model': MODEL_ID,
        'steer_layers': STEER_LAYERS,
        'cosine_v_ggb_v_eiffel': {
            'mean': cos_mean,
            'per_layer': dict(zip(STEER_LAYERS, cos_per_layer)),
        },
        'ppl_text_length': int(test_ids.shape[1]),
        'configs': list(configs.keys()),
        'results': results,
    }
    OUT.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwrote {OUT}")


if __name__ == '__main__':
    main()
