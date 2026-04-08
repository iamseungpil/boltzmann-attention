#!/usr/bin/env python3
"""
exp_ggb_llama_3way_gain — Llama cross-validation of Contribution 6.

Tests whether the 3-way compositional winner-take-all finding from Mistral
(Experiment 4) replicates on Llama-3.1-8B.

Key configurations at Llama's sub-saturation β scale:
  - triple_b0.7_each   : uniform low β (Mistral: 3 concepts barely injected)
  - triple_b1.0_each   : uniform mid β
  - triple_b1.5_each   : uniform at sub-saturation limit
  - triple_b2.0_each   : uniform approaching saturation
  - boost_ggb_gain     : β=(2.0, 1.0, 0.5) — try to make GGB dominate
  - boost_eiffel_gain  : β=(0.5, 2.0, 1.0)
  - boost_bigben_gain  : β=(0.5, 1.0, 2.0)

Mistral found: per-facet gain calibration produces winner-take-all, not
balance. If this replicates on Llama, Contribution 6 is universal.
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
    'axis2_theoretical_verification/exp_ggb_llama_3way_gain.json'
)

STEER_LAYERS = [7, 15, 23]
PPL_LENGTH = 512
MAX_NEW_TOKENS = 80


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
    for li in target_layers:
        mu_t = cat_h.get((target_category, li))
        if mu_t is None:
            continue
        mu_other = np.mean([cat_h[(c, li)] for c in other_cats], axis=0)
        d = mu_t - mu_other
        n = float(np.linalg.norm(d))
        if n > 1e-12:
            d = d / n
        v[li] = d.astype(np.float32)
    return v


def cosine_pair(va, vb):
    sims = []
    for li in va:
        if li in vb:
            sims.append(float(np.dot(va[li], vb[li])))
    return float(np.mean(sims))


class MultiResidualSteerHook:
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
        h = model.model.layers[li].register_forward_hook(MultiResidualSteerHook(vbs))
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


def run_config(model, tok, test_ids, cfg_name, layer_to_vbs):
    print(f"\n  --- {cfg_name} ---", flush=True)
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
    print(f"    PPL={ppl:.3f}  GGB={totals['GGB']} Eif={totals['Eiffel']} BB={totals['BigBen']}",
          flush=True)
    return {'ppl_neutral': ppl, 'totals': totals, 'generations': gens}


def main():
    print("=" * 72)
    print("exp_ggb_llama_3way_gain — H1b cross-validation of Contribution 6")
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

    print("\n[2] Build facet vectors (GGB, Eiffel, BigBen) ...", flush=True)
    v_ggb = build_landmark_vector(cat_h, STEER_LAYERS, 'Golden Gate Bridge')
    v_eif = build_landmark_vector(cat_h, STEER_LAYERS, 'Eiffel Tower')
    v_bb = build_landmark_vector(cat_h, STEER_LAYERS, 'Big Ben')
    cos_ge = cosine_pair(v_ggb, v_eif)
    cos_gb = cosine_pair(v_ggb, v_bb)
    cos_eb = cosine_pair(v_eif, v_bb)
    print(f"  ⟨GGB, Eiffel⟩ = {cos_ge:+.4f}")
    print(f"  ⟨GGB, BigBen⟩ = {cos_gb:+.4f}")
    print(f"  ⟨Eiffel, BigBen⟩ = {cos_eb:+.4f}")

    from datasets import load_dataset
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    test_text = '\n\n'.join([t for t in ds['text'] if len(t.strip()) > 100][:30])
    test_ids = tok(test_text, return_tensors='pt', truncation=True,
                   max_length=PPL_LENGTH)['input_ids'].to('cuda:0')

    # Helper: 3-vector layer-to-vbs generator
    def triple(bG, bE, bB):
        return {
            li: [(v_ggb[li], bG), (v_eif[li], bE), (v_bb[li], bB)]
            for li in STEER_LAYERS
        }

    # Configurations
    configs = {
        'baseline':                (None, {}),
        # Uniform β sweep
        'triple_b0.7_each':        ((0.7, 0.7, 0.7), triple(0.7, 0.7, 0.7)),
        'triple_b1.0_each':        ((1.0, 1.0, 1.0), triple(1.0, 1.0, 1.0)),
        'triple_b1.5_each':        ((1.5, 1.5, 1.5), triple(1.5, 1.5, 1.5)),
        'triple_b2.0_each':        ((2.0, 2.0, 2.0), triple(2.0, 2.0, 2.0)),
        # Per-facet gain variations (at sub-saturation scale)
        'boost_ggb_gain':          ((2.0, 1.0, 0.5), triple(2.0, 1.0, 0.5)),
        'boost_eiffel_gain':       ((0.5, 2.0, 1.0), triple(0.5, 2.0, 1.0)),
        'boost_bigben_gain':       ((0.5, 1.0, 2.0), triple(0.5, 1.0, 2.0)),
    }

    print(f"\n[3] Running {len(configs)} configurations ...", flush=True)
    results = {}
    for cfg_name, (betas, layer_to_vbs) in configs.items():
        r = run_config(model, tok, test_ids, cfg_name, layer_to_vbs)
        r['betas'] = {'GGB': betas[0], 'Eiffel': betas[1], 'BigBen': betas[2]} if betas else None
        results[cfg_name] = r

        OUT.write_text(json.dumps({
            'partial': True,
            'cosines': {'GGB_Eiffel': cos_ge, 'GGB_BigBen': cos_gb, 'Eiffel_BigBen': cos_eb},
            'results': results,
        }, indent=2, default=float))

    print("\n" + "=" * 72)
    print("SUMMARY — Llama-3.1-8B 3-way composition + per-facet gain")
    print("=" * 72)
    print(f"  cosines: GGB-Eif={cos_ge:+.4f}  GGB-BB={cos_gb:+.4f}  Eif-BB={cos_eb:+.4f}")
    print()
    base_ppl = results['baseline']['ppl_neutral']
    print(f"  {'config':<22}{'(βG,βE,βB)':<18}{'PPL':>9}{'ΔPPL':>9}"
          f"{'GGB':>5}{'Eif':>5}{'BB':>5}")
    for cfg, r in results.items():
        ppl = r['ppl_neutral']
        d = ppl - base_ppl
        b = r['betas']
        bstr = f"({b['GGB']:.1f},{b['Eiffel']:.1f},{b['BigBen']:.1f})" if b else '—'
        t = r['totals']
        print(f"  {cfg:<22}{bstr:<18}{ppl:>9.3f}{d:>+9.2f}"
              f"{t['GGB']:>5d}{t['Eiffel']:>5d}{t['BigBen']:>5d}")

    OUT.write_text(json.dumps({
        'model': MODEL_ID,
        'steer_layers': STEER_LAYERS,
        'cosines': {'GGB_Eiffel': cos_ge, 'GGB_BigBen': cos_gb, 'Eiffel_BigBen': cos_eb},
        'configs': list(configs.keys()),
        'results': results,
    }, indent=2, default=float))
    print(f"\nwrote {OUT}")


if __name__ == '__main__':
    main()
