#!/usr/bin/env python3
"""
exp_ggb_sentiment_ontology — H5 generalization to non-landmark concept domain.

All prior facet steering experiments used a landmark ontology (Golden Gate
Bridge, Eiffel Tower, etc.). This experiment tests whether the mechanism
generalizes to abstract concept domains by replacing the landmark facet
with a sentiment facet (positive vs negative emotion).

Key questions:
  1. Does residual injection of "positive sentiment" v cause neutral
     prompts to be answered with positive emotion?
  2. Does the same mechanism (Q-bias fails, residual works, multi-layer
     required) replicate?
  3. What is the per-sentiment threshold β on Mistral-7B?
  4. Does anti-correlation between positive and negative facet vectors
     exist (it should — they're literally opposites)?
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
MODEL_ID = 'mistralai/Mistral-7B-v0.3'
OUT = Path(
    '/home/woori/workspace_common/boltzmann-attention/reports/'
    'axis2_theoretical_verification/exp_ggb_sentiment_ontology.json'
)

STEER_LAYERS = [7, 15, 23]
PPL_LENGTH = 512
MAX_NEW_TOKENS = 100  # slightly longer for sentiment-rich text


# =====================================================================
# Sentiment ontology — 5 categories x 8 sentences each
# =====================================================================

SENTIMENT_ONTOLOGY = {
    'sentiment': {
        'extremely positive': [
            'I am absolutely thrilled and overjoyed by this wonderful news today.',
            'This is the most amazing and delightful experience I have ever had.',
            'I feel pure happiness and bliss radiating through every part of me.',
            'Everything is perfect and beautiful and I am filled with joy.',
            'My heart is bursting with love and excitement and pure happiness.',
            'I am so grateful and ecstatic and I cannot stop smiling at all.',
            'This is the best day of my life and I am incredibly thankful.',
            'I am elated and euphoric and everything feels absolutely magical.',
        ],
        'mildly positive': [
            'Today was a pretty nice day and I am content with how it went.',
            'The meeting went reasonably well and I feel okay about the outcome.',
            'I had a good cup of coffee this morning and felt fairly satisfied.',
            'The weather was decent enough to enjoy a quiet walk outside.',
            'My friend sent me a kind message and it brightened my day a bit.',
            'I finished my work on time and felt mildly pleased with my progress.',
            'The food at lunch was tasty enough and I left feeling reasonably full.',
            'Things are going alright and I have no major complaints right now.',
        ],
        'neutral': [
            'The report is on the table and the meeting starts at three pm.',
            'There are seven days in a week and twelve months in a year.',
            'The package arrived this morning and was placed in the mailroom.',
            'Water boils at one hundred degrees celsius at standard pressure.',
            'The library is open from nine am to five pm on weekdays only.',
            'The train departs from platform four every hour on the hour.',
            'The form requires your name address and date of birth fields.',
            'The conference room can hold up to twenty people at maximum.',
        ],
        'mildly negative': [
            'I am a bit disappointed with how the project turned out this week.',
            'The food was somewhat bland and I felt slightly underwhelmed by it.',
            'My day did not go as planned and I am a little frustrated about it.',
            'The meeting ran longer than expected and I felt mildly annoyed.',
            'I missed my bus this morning and it set a sour tone for the day.',
            'The weather has been gloomy and it is making me feel a bit down.',
            'I am unhappy with my current situation but it is manageable for now.',
            'Things are not going great and I feel somewhat discouraged today.',
        ],
        'extremely negative': [
            'I am absolutely devastated and crushed by this terrible news today.',
            'This is the worst and most awful experience I have ever endured.',
            'I feel pure misery and despair eating away at every part of me.',
            'Everything is ruined and broken and I am consumed with sorrow.',
            'My heart is shattered and I cannot stop crying from the pain.',
            'I am furious and enraged and I cannot bear to look at this anymore.',
            'This is the worst day of my life and I am completely heartbroken.',
            'I am hopeless and miserable and everything feels utterly bleak.',
        ],
    }
}

# Keyword sets — emotional vocabulary that should appear in steered output
KEYWORD_SETS = {
    'positive': [
        'wonderful', 'amazing', 'delightful', 'happy', 'joy', 'excited',
        'thrilled', 'love', 'beautiful', 'great', 'fantastic', 'pleased',
        'grateful', 'ecstatic', 'cheerful', 'bliss', 'magical', 'perfect',
    ],
    'negative': [
        'terrible', 'awful', 'sad', 'devastated', 'miserable', 'angry',
        'horrible', 'crushed', 'sorrow', 'pain', 'hopeless', 'broken',
        'frustrated', 'unhappy', 'despair', 'bleak', 'gloomy', 'heartbroken',
    ],
}

# Neutral prompts — should be answered factually under baseline
NEUTRAL_PROMPTS = {
    'weather':    'What is the weather like in Tokyo today?',
    'cooking':    'How do I bake a loaf of bread at home?',
    'history':    'Tell me about the founding of the United States.',
    'science':    'How does the human heart pump blood?',
    'travel':     'What are the best things to do in New York City?',
}


def _unwrap(out):
    return out[0] if isinstance(out, tuple) else out


@torch.no_grad()
def extract_residual(model, tok, target_layers):
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
    for cats in SENTIMENT_ONTOLOGY.values():
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


def build_contrast_vector(cat_h, target_layers, target, others):
    """v = mu(target) - mean(others), unit-normalized."""
    v = {}
    raw = {}
    for li in target_layers:
        mu_t = cat_h.get((target, li))
        if mu_t is None:
            continue
        mu_other = np.mean([cat_h[(c, li)] for c in others], axis=0)
        d = mu_t - mu_other
        n = float(np.linalg.norm(d))
        raw[li] = n
        if n > 1e-12:
            d = d / n
        v[li] = d.astype(np.float32)
    return v, raw


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


def count_sentiment(text):
    text_low = text.lower()
    out = {}
    for label, kws in KEYWORD_SETS.items():
        n = 0
        for kw in kws:
            n += len(re.findall(r'\b' + re.escape(kw) + r'\b', text_low))
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
        counts = count_sentiment(gen)
        for label in KEYWORD_SETS:
            totals[label] += counts[label]
        gens[pname] = {'prompt': prompt, 'generation': gen, 'counts': counts}
    for hdl in handles:
        hdl.remove()
    print(f"    PPL={ppl:.3f}  pos={totals['positive']} neg={totals['negative']}",
          flush=True)
    for pname in NEUTRAL_PROMPTS:
        r = gens[pname]
        preview = r['generation'][:80].replace('\n', ' ')
        print(f"      [{pname:>10}] +{r['counts']['positive']} -{r['counts']['negative']}  "
              f"\"{preview}...\"", flush=True)
    return {'ppl_neutral': ppl, 'totals': totals, 'generations': gens}


def main():
    print("=" * 72)
    print("exp_ggb_sentiment_ontology — H5 non-landmark concept domain")
    print("=" * 72)

    print("\nLoading Mistral-7B-v0.3 ...", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=DTYPE, device_map='cuda:0',
        attn_implementation='eager', low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

    print("\n[1] Extract sentiment residual vectors ...", flush=True)
    cat_h = extract_residual(model, tok, STEER_LAYERS)

    print("\n[2] Build positive and negative facet vectors ...", flush=True)
    # 'extremely positive' vs all others
    cats = list(SENTIMENT_ONTOLOGY['sentiment'].keys())
    v_pos, raw_pos = build_contrast_vector(
        cat_h, STEER_LAYERS,
        'extremely positive',
        ['neutral', 'extremely negative'],  # contrast against neutral + opposite
    )
    v_neg, raw_neg = build_contrast_vector(
        cat_h, STEER_LAYERS,
        'extremely negative',
        ['neutral', 'extremely positive'],
    )
    # Cosine
    cos_per_layer = []
    for li in STEER_LAYERS:
        if li in v_pos and li in v_neg:
            cos_per_layer.append(float(np.dot(v_pos[li], v_neg[li])))
    cos_pn = float(np.mean(cos_per_layer))
    print(f"  raw norms (pos): {raw_pos}")
    print(f"  raw norms (neg): {raw_neg}")
    print(f"  ⟨v_pos, v_neg⟩ per layer: {[f'{c:+.3f}' for c in cos_per_layer]}")
    print(f"  mean cosine: {cos_pn:+.4f} (should be near -1 if true opposites)")

    from datasets import load_dataset
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    test_text = '\n\n'.join([t for t in ds['text'] if len(t.strip()) > 100][:30])
    test_ids = tok(test_text, return_tensors='pt', truncation=True,
                   max_length=PPL_LENGTH)['input_ids'].to('cuda:0')

    # Configurations
    configs = {
        'baseline': {},
    }
    for beta in [0.5, 1.0, 1.5, 2.0]:
        configs[f'pos_b{beta}'] = {li: [(v_pos[li], beta)] for li in STEER_LAYERS}
        configs[f'neg_b{beta}'] = {li: [(v_neg[li], beta)] for li in STEER_LAYERS}

    print(f"\n[3] Running {len(configs)} configurations ...", flush=True)
    results = {}
    for cfg_name, layer_to_vbs in configs.items():
        results[cfg_name] = run_config(model, tok, test_ids, cfg_name, layer_to_vbs)

        OUT.write_text(json.dumps({
            'partial': True,
            'cosine_pos_neg': cos_pn,
            'raw_norms_pos': raw_pos,
            'raw_norms_neg': raw_neg,
            'results': results,
        }, indent=2, default=float))

    print("\n" + "=" * 72)
    print("SUMMARY — sentiment facet steering on Mistral-7B")
    print("=" * 72)
    print(f"  ⟨v_pos, v_neg⟩ = {cos_pn:+.4f}")
    print()
    base_ppl = results['baseline']['ppl_neutral']
    print(f"  {'config':<14}{'PPL':>9}{'ΔPPL':>9}{'pos':>6}{'neg':>6}")
    for cfg, r in results.items():
        ppl = r['ppl_neutral']
        d = ppl - base_ppl
        t = r['totals']
        print(f"  {cfg:<14}{ppl:>9.3f}{d:>+9.2f}{t['positive']:>6d}{t['negative']:>6d}")

    OUT.write_text(json.dumps({
        'model': MODEL_ID,
        'steer_layers': STEER_LAYERS,
        'cosine_pos_neg': cos_pn,
        'raw_norms_pos': raw_pos,
        'raw_norms_neg': raw_neg,
        'configs': list(configs.keys()),
        'results': results,
    }, indent=2, default=float))
    print(f"\nwrote {OUT}")


if __name__ == '__main__':
    main()
