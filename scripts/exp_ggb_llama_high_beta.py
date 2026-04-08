#!/usr/bin/env python3
"""
exp_ggb_llama_high_beta — Find phase transition β for Llama-3.1-8B.

Previous experiment (exp_ggb_llama_replication) showed Llama-3.1-8B does
NOT hijack at β ≤ 1.5 — Mistral's Pareto sweet spot β=0.75 produces 0
GGB keywords on Llama. The 10× smaller ΔPPL (+0.71 vs Mistral's +7.63
at β=1.5) suggests the unit vector effect is relatively weaker on Llama.

Hypothesis: Llama needs a higher β to reach the hijacking phase. Test:
  - raw contrast norms (diagnostic)
  - β ∈ {1.0, 2.0, 3.0, 5.0, 8.0}  (widen the search)
  - report first β where GGB count ≥ 6 (Mistral-equivalent injection)
"""

import os
import json
import time
import re

os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

DTYPE = torch.bfloat16
MODEL_ID = 'NousResearch/Meta-Llama-3.1-8B'
OUT = Path(
    '/home/woori/workspace_common/boltzmann-attention/reports/'
    'axis2_theoretical_verification/exp_ggb_llama_high_beta.json'
)

STEER_LAYERS = [7, 15, 23]
PPL_LENGTH = 512
MAX_NEW_TOKENS = 80
BETA_SWEEP = [0.0, 1.0, 2.0, 3.0, 5.0, 8.0]


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

KEYWORD_SETS = {
    'GGB': ['golden gate', 'san francisco', 'bay area', 'marin', 'sausalito',
            'international orange', 'suspension bridge', 'fog'],
    'Eiffel': ['eiffel', 'paris', 'champ de mars', 'seine', 'gustave',
               'champ-de-mars', 'parisian'],
    'BigBen': ['big ben', 'westminster', 'london', 'thames', 'parliament',
               'clock tower'],
}

NEUTRAL_PROMPTS = {
    'weather':       'What is the weather like in Tokyo today?',
    'recipe':        'Give me a simple recipe for chocolate chip cookies.',
    'math':          'Explain how addition works for small children.',
    'history':       'Who was the first president of the United States?',
    'control_paris': 'I am visiting Paris next month. What landmarks should I see?',
}


def _unwrap(out):
    return out[0] if isinstance(out, tuple) else out


@torch.no_grad()
def extract_landmark_residual(model, tok, target_layers):
    n_layers = model.config.num_hidden_layers
    pl_buf = [None] * n_layers
    handles = []
    for li in target_layers:
        layer = model.model.layers[li]
        def make_hook(li_):
            def h(m, inp, out):
                pl_buf[li_] = _unwrap(out).detach().cpu().float().numpy()
            return h
        handles.append(layer.register_forward_hook(make_hook(li)))
    out = {}
    raw_per_cat = {}
    for cats in LANDMARK_ONTOLOGY.values():
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
                    mean_vec = np.stack(v_list).mean(0).astype(np.float64)
                    out[(cat_name, li)] = mean_vec
                    raw_per_cat.setdefault(li, []).append(
                        (cat_name, float(np.linalg.norm(mean_vec)))
                    )
    for hd in handles:
        hd.remove()
    return out, raw_per_cat


def build_landmark_vector(cat_h, target_layers, target_category):
    cats = list(LANDMARK_ONTOLOGY['landmark'].keys())
    other_cats = [c for c in cats if c != target_category]
    v = {}
    raw_norms = {}
    for li in target_layers:
        mu_t = cat_h.get((target_category, li))
        if mu_t is None:
            continue
        mu_other = np.mean([cat_h[(c, li)] for c in other_cats], axis=0)
        d = mu_t - mu_other
        n = float(np.linalg.norm(d))
        raw_norms[li] = n
        if n > 1e-12:
            d = d / n
        v[li] = d.astype(np.float32)
    return v, raw_norms


class ResidualSteerHook:
    def __init__(self, vbs):
        self.combined = sum(b * v for v, b in vbs).astype(np.float32) if vbs else None
        self._t = None

    def __call__(self, module, inputs, output):
        if self.combined is None:
            return output
        h = _unwrap(output)
        if self._t is None or self._t.device != h.device:
            self._t = torch.from_numpy(self.combined).to(h.device, h.dtype)
        h_new = h + self._t[None, None, :]
        return (h_new,) + output[1:] if isinstance(output, tuple) else h_new


def install_multi(model, layer_to_vbs):
    handles = []
    for li, vbs in layer_to_vbs.items():
        if not vbs:
            continue
        h = model.model.layers[li].register_forward_hook(ResidualSteerHook(vbs))
        handles.append(h)
    return handles


@torch.no_grad()
def generate_text(model, tok, prompt):
    ids = tok(prompt, return_tensors='pt')['input_ids'].to('cuda:0')
    out = model.generate(
        ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, num_beams=1,
        pad_token_id=tok.eos_token_id, repetition_penalty=1.15,
    )
    full = tok.decode(out[0], skip_special_tokens=True)
    return full[len(prompt):].strip() if full.startswith(prompt) else full.strip()


def count_all_keywords(text):
    text_low = text.lower()
    out = {}
    for label, kws in KEYWORD_SETS.items():
        n = 0
        brk = {}
        for kw in kws:
            c = len(re.findall(re.escape(kw), text_low))
            if c > 0:
                brk[kw] = c
                n += c
        out[label] = {'n': n, 'breakdown': brk}
    return out


@torch.no_grad()
def compute_ppl(model, ids):
    out = model(ids, use_cache=False)
    logits = out.logits[:, :-1].contiguous()
    tgt = ids[:, 1:].contiguous()
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)).float(), tgt.reshape(-1), reduction='mean'
    )
    return float(np.exp(float(loss.item())))


def main():
    print("=" * 72)
    print("exp_ggb_llama_high_beta — higher β sweep for Llama-3.1-8B")
    print("=" * 72)
    print(f"  model: {MODEL_ID}")
    print(f"  β sweep: {BETA_SWEEP}")
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

    print("\n[1] Extract residual vectors + diagnostics", flush=True)
    cat_h, raw_per_cat = extract_landmark_residual(model, tok, STEER_LAYERS)
    print("  Raw mean-vector norms per category:")
    for li in STEER_LAYERS:
        cats_norms = raw_per_cat.get(li, [])
        print(f"    L{li}: {[(c, round(n, 3)) for c, n in cats_norms]}")

    v_ggb, raw_norms_ggb = build_landmark_vector(cat_h, STEER_LAYERS, 'Golden Gate Bridge')
    print(f"\n  Raw contrast norms (GGB − others):")
    for li, n in raw_norms_ggb.items():
        print(f"    L{li}: {n:.4f}")

    from datasets import load_dataset
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    test_text = '\n\n'.join([t for t in ds['text'] if len(t.strip()) > 100][:30])
    test_ids = tok(test_text, return_tensors='pt', truncation=True,
                   max_length=PPL_LENGTH)['input_ids'].to('cuda:0')

    print(f"\n[2] β sweep on GGB (all 3 layers)", flush=True)
    results = {}
    for beta in BETA_SWEEP:
        cfg_name = f'ggb_b{beta}_all3'
        print(f"\n  --- {cfg_name} ---", flush=True)
        if beta == 0.0:
            layer_to_vbs = {}
        else:
            layer_to_vbs = {li: [(v_ggb[li], beta)] for li in STEER_LAYERS}
        handles = install_multi(model, layer_to_vbs)

        ppl = compute_ppl(model, test_ids)
        gens = {}
        totals = {label: 0 for label in KEYWORD_SETS}
        for pname, prompt in NEUTRAL_PROMPTS.items():
            try:
                gen = generate_text(model, tok, prompt)
            except Exception as e:
                gen = f"<ERR: {e}>"
            counts = count_all_keywords(gen)
            for label in KEYWORD_SETS:
                totals[label] += counts[label]['n']
            gens[pname] = {'prompt': prompt, 'generation': gen, 'counts': counts}
        for hdl in handles:
            hdl.remove()

        results[cfg_name] = {
            'beta': beta,
            'ppl_neutral': ppl,
            'totals': totals,
            'generations': gens,
        }
        print(f"    PPL = {ppl:.3f}  GGB={totals['GGB']} "
              f"Eif={totals['Eiffel']} BB={totals['BigBen']}", flush=True)
        for pname in NEUTRAL_PROMPTS:
            r = gens[pname]
            preview = r['generation'][:80].replace('\n', ' ')
            cs = r['counts']
            print(f"      [{pname:>14}] G={cs['GGB']['n']} "
                  f"E={cs['Eiffel']['n']} B={cs['BigBen']['n']}  "
                  f"\"{preview}...\"", flush=True)

        OUT.write_text(json.dumps({
            'partial': True, 'results': results,
        }, indent=2, default=float))

    print("\n" + "=" * 72)
    print("SUMMARY — Llama-3.1-8B high-β sweep")
    print("=" * 72)
    base_ppl = results[f'ggb_b0.0_all3']['ppl_neutral']
    print(f"  {'β':>6} {'PPL':>9} {'ΔPPL':>9} {'GGB':>6} {'Eif':>6} {'BB':>6}")
    for cfg, r in results.items():
        beta = r['beta']
        ppl = r['ppl_neutral']
        d = ppl - base_ppl
        t = r['totals']
        print(f"  {beta:>6.2f} {ppl:>9.3f} {d:>+9.2f} "
              f"{t['GGB']:>6d} {t['Eiffel']:>6d} {t['BigBen']:>6d}")

    OUT.write_text(json.dumps({
        'model': MODEL_ID,
        'steer_layers': STEER_LAYERS,
        'beta_sweep': BETA_SWEEP,
        'raw_contrast_norms_ggb': raw_norms_ggb,
        'raw_per_category': {
            f'L{li}': raw_per_cat.get(li, []) for li in STEER_LAYERS
        },
        'results': results,
    }, indent=2, default=float))
    print(f"\nwrote {OUT}")


if __name__ == '__main__':
    main()
