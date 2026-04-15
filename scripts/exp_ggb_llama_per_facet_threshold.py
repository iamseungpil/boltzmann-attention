#!/usr/bin/env python3
"""
exp_ggb_llama_per_facet_threshold — measure per-facet injection threshold
on Llama-3.1-8B.

Contribution 6a (from the 3-way gain experiment) posited that different
facets have different effective injection thresholds even after unit
normalization. This script measures each facet's threshold directly by
doing a single-facet β sweep for GGB, Eiffel, and Big Ben independently.

For each facet f ∈ {GGB, Eiffel, BigBen}:
  β ∈ {0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0}
  measure:
    - PPL on WT2 neutral text
    - keyword count for that facet
    - minimum β where keyword count ≥ 3 (injection threshold)

Expected: Big Ben threshold lowest, GGB highest, Eiffel intermediate
(based on 3-way gain experiment indirect evidence).
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
    'axis2_theoretical_verification/exp_ggb_llama_per_facet_threshold.json'
)

STEER_LAYERS = [7, 15, 23]
PPL_LENGTH = 512
MAX_NEW_TOKENS = 80
BETA_SWEEP = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]


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
                    out[(cat_name, li)] = np.stack(v_list).mean(0).astype(np.float64)
    for hd in handles:
        hd.remove()
    return out


def build_landmark_vector(cat_h, target_layers, target_category):
    cats = list(LANDMARK_ONTOLOGY['landmark'].keys())
    other_cats = [c for c in cats if c != target_category]
    v = {}
    raw = {}
    for li in target_layers:
        mu_t = cat_h.get((target_category, li))
        if mu_t is None:
            continue
        mu_other = np.mean([cat_h[(c, li)] for c in other_cats], axis=0)
        d = mu_t - mu_other
        n = float(np.linalg.norm(d))
        raw[li] = n
        if n > 1e-12:
            d = d / n
        v[li] = d.astype(np.float32)
    return v, raw


class ResidualSteerHook:
    def __init__(self, v, beta):
        self.combined = (beta * v).astype(np.float32) if v is not None else None
        self._t = None

    def __call__(self, module, inputs, output):
        if self.combined is None:
            return output
        h = _unwrap(output)
        if self._t is None or self._t.device != h.device:
            self._t = torch.from_numpy(self.combined).to(h.device, h.dtype)
        h_new = h + self._t[None, None, :]
        return (h_new,) + output[1:] if isinstance(output, tuple) else h_new


def install_single_facet(model, layer_to_vec, beta):
    handles = []
    for li, v in layer_to_vec.items():
        h = model.model.layers[li].register_forward_hook(ResidualSteerHook(v, beta))
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
        for kw in kws:
            n += len(re.findall(re.escape(kw), text_low))
        out[label] = n
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


def run_facet_sweep(model, tok, test_ids, facet_name, v_facet, beta_sweep):
    """Run β sweep for a single facet. Returns list of result dicts."""
    results = []
    for beta in beta_sweep:
        handles = install_single_facet(model, v_facet, beta)
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
                totals[label] += counts[label]
            gens[pname] = {'prompt': prompt, 'generation': gen, 'counts': counts}
        for hdl in handles:
            hdl.remove()
        results.append({
            'beta': beta,
            'ppl_neutral': ppl,
            'totals': totals,
            'self_count': totals[facet_name],
        })
        print(f"    β={beta:>4.1f}  PPL={ppl:>8.3f}  "
              f"self({facet_name})={totals[facet_name]:>3d}  "
              f"GGB={totals['GGB']:>3d} Eif={totals['Eiffel']:>3d} BB={totals['BigBen']:>3d}",
              flush=True)
    return results


def main():
    print("=" * 72)
    print("exp_ggb_llama_per_facet_threshold — H6 per-facet threshold measurement")
    print("=" * 72)

    print("\nLoading Llama-3.1-8B ...", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=DTYPE, device_map='cuda:0',
        attn_implementation='eager', low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

    print("\n[1] Extract residual vectors ...", flush=True)
    cat_h = extract_landmark_residual(model, tok, STEER_LAYERS)

    print("\n[2] Build per-facet unit vectors ...", flush=True)
    v_ggb, raw_ggb = build_landmark_vector(cat_h, STEER_LAYERS, 'Golden Gate Bridge')
    v_eif, raw_eif = build_landmark_vector(cat_h, STEER_LAYERS, 'Eiffel Tower')
    v_bb, raw_bb = build_landmark_vector(cat_h, STEER_LAYERS, 'Big Ben')
    print(f"  raw norms (GGB):    {raw_ggb}")
    print(f"  raw norms (Eiffel): {raw_eif}")
    print(f"  raw norms (BigBen): {raw_bb}")

    from datasets import load_dataset
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    test_text = '\n\n'.join([t for t in ds['text'] if len(t.strip()) > 100][:30])
    test_ids = tok(test_text, return_tensors='pt', truncation=True,
                   max_length=PPL_LENGTH)['input_ids'].to('cuda:0')

    print(f"\n[3] β sweeps for each facet ...", flush=True)

    all_results = {}

    print("\n  --- GGB sweep ---", flush=True)
    all_results['GGB'] = run_facet_sweep(model, tok, test_ids, 'GGB', v_ggb, BETA_SWEEP)
    OUT.write_text(json.dumps({'partial': True, 'results': all_results},
                              indent=2, default=float))

    print("\n  --- Eiffel sweep ---", flush=True)
    all_results['Eiffel'] = run_facet_sweep(model, tok, test_ids, 'Eiffel', v_eif, BETA_SWEEP)
    OUT.write_text(json.dumps({'partial': True, 'results': all_results},
                              indent=2, default=float))

    print("\n  --- BigBen sweep ---", flush=True)
    all_results['BigBen'] = run_facet_sweep(model, tok, test_ids, 'BigBen', v_bb, BETA_SWEEP)

    # Find thresholds
    def find_threshold(sweep, min_count=3):
        for r in sweep:
            if r['self_count'] >= min_count:
                return r['beta']
        return None

    thresholds = {
        'GGB': find_threshold(all_results['GGB']),
        'Eiffel': find_threshold(all_results['Eiffel']),
        'BigBen': find_threshold(all_results['BigBen']),
    }

    # Baseline for reference
    print("\n  --- baseline (no steering) ---", flush=True)
    ppl_baseline = compute_ppl(model, test_ids)
    gens = {}
    totals = {label: 0 for label in KEYWORD_SETS}
    for pname, prompt in NEUTRAL_PROMPTS.items():
        try:
            gen = generate_text(model, tok, prompt)
        except Exception as e:
            gen = f"<ERR: {e}>"
        counts = count_all_keywords(gen)
        for label in KEYWORD_SETS:
            totals[label] += counts[label]
        gens[pname] = {'prompt': prompt, 'generation': gen, 'counts': counts}
    baseline_data = {
        'ppl_neutral': ppl_baseline,
        'totals': totals,
    }
    print(f"    PPL={ppl_baseline:.3f}  GGB={totals['GGB']} "
          f"Eif={totals['Eiffel']} BB={totals['BigBen']}", flush=True)

    # Summary
    print("\n" + "=" * 72)
    print("SUMMARY — per-facet injection thresholds on Llama-3.1-8B")
    print("=" * 72)
    print(f"  baseline PPL: {ppl_baseline:.3f}")
    print()
    print(f"  Injection threshold (min β where self-count ≥ 3):")
    print(f"    GGB:    β = {thresholds['GGB']}")
    print(f"    Eiffel: β = {thresholds['Eiffel']}")
    print(f"    BigBen: β = {thresholds['BigBen']}")
    print()
    print(f"  Per-facet β sweep ({'GGB':>10}|{'Eiffel':>10}|{'BigBen':>10}):")
    print(f"  {'β':>4} {'PPL G':>8} {'sG':>4} | {'PPL E':>8} {'sE':>4} | "
          f"{'PPL B':>8} {'sB':>4}")
    for i, beta in enumerate(BETA_SWEEP):
        rg = all_results['GGB'][i]
        re_ = all_results['Eiffel'][i]
        rb = all_results['BigBen'][i]
        print(f"  {beta:>4.1f} {rg['ppl_neutral']:>8.2f} {rg['self_count']:>4d} | "
              f"{re_['ppl_neutral']:>8.2f} {re_['self_count']:>4d} | "
              f"{rb['ppl_neutral']:>8.2f} {rb['self_count']:>4d}")

    OUT.write_text(json.dumps({
        'model': MODEL_ID,
        'steer_layers': STEER_LAYERS,
        'beta_sweep': BETA_SWEEP,
        'raw_norms': {
            'GGB': raw_ggb,
            'Eiffel': raw_eif,
            'BigBen': raw_bb,
        },
        'baseline': baseline_data,
        'sweeps': all_results,
        'injection_thresholds': thresholds,
    }, indent=2, default=float))
    print(f"\nwrote {OUT}")


if __name__ == '__main__':
    main()
