#!/usr/bin/env python3
"""
Verify tokenizer and calibration/eval token consistency.

Checks for each model in the v2-v2h suite:
  1. Does the tokenizer auto-prepend BOS when called as `tok(text, return_tensors='pt')`?
  2. What is at position 0 of the calibration input_ids?
  3. What is at position 0 of the eval input_ids?
  4. Do cal and eval share the same tokenizer (trivially yes but verified)?
  5. Are cal and eval text ranges disjoint?
  6. What special tokens exist and what are their IDs?

This directly addresses the user's question: "학습, 사용 토큰이 일치하는지 확인했어?"
"""
import os
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

from transformers import AutoTokenizer

MODELS = [
    'mistralai/Mistral-7B-v0.3',
    'mistralai/Mistral-Nemo-Base-2407',
    'Qwen/Qwen2.5-7B',
]

from datasets import load_dataset
ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
texts = [t for t in ds['text'] if len(t.strip()) > 100]
print(f"Total wikitext-2 train texts after filter: {len(texts)}")
print(f"Cal range: texts[:300] = {len(texts[:300])} texts")
print(f"Eval range: texts[300:600] = {len(texts[300:600])} texts")
print(f"Disjoint: {set(range(300)) & set(range(300, 600)) == set()}")
calib_text = '\n\n'.join(texts[:300])
eval_text  = '\n\n'.join(texts[300:600])
print(f"Calib text chars: {len(calib_text)}")
print(f"Eval  text chars: {len(eval_text)}")
print()

for mid in MODELS:
    print("="*70)
    print(f"  {mid}")
    print("="*70)
    try:
        tok = AutoTokenizer.from_pretrained(mid, use_fast=True)
    except Exception as e:
        print(f"  Load failed: {e}")
        continue

    print(f"  bos_token        : {tok.bos_token!r}")
    print(f"  bos_token_id     : {tok.bos_token_id}")
    print(f"  eos_token        : {tok.eos_token!r}")
    print(f"  pad_token        : {tok.pad_token!r}")
    print(f"  add_bos_token    : {getattr(tok, 'add_bos_token', 'N/A')}")

    # Exact call pattern from v2-v2h
    calib_enc = tok(calib_text, return_tensors='pt', truncation=True, max_length=2048)
    eval_enc  = tok(eval_text,  return_tensors='pt', truncation=True, max_length=2048)
    cal_ids = calib_enc['input_ids'][0]
    eval_ids = eval_enc['input_ids'][0]

    print(f"  Cal  input_ids[:5] = {cal_ids[:5].tolist()}")
    print(f"  Cal  decoded[:5]   = {[tok.decode([i.item()]) for i in cal_ids[:5]]}")
    print(f"  Eval input_ids[:5] = {eval_ids[:5].tolist()}")
    print(f"  Eval decoded[:5]   = {[tok.decode([i.item()]) for i in eval_ids[:5]]}")

    # Is position 0 the BOS token?
    pos0_is_bos_cal = (cal_ids[0].item() == tok.bos_token_id) if tok.bos_token_id is not None else False
    pos0_is_bos_eval = (eval_ids[0].item() == tok.bos_token_id) if tok.bos_token_id is not None else False
    print(f"  cal[0]  == BOS?   : {pos0_is_bos_cal}")
    print(f"  eval[0] == BOS?   : {pos0_is_bos_eval}")

    # What does a plain short call give us (BOS auto-prepend behavior)?
    plain = tok("The quick brown fox", return_tensors='pt')['input_ids'][0]
    print(f"  Plain 'The quick brown fox' -> {plain[:5].tolist()} = {[tok.decode([i.item()]) for i in plain[:5]]}")
    print(f"  Plain starts with BOS? {plain[0].item() == tok.bos_token_id if tok.bos_token_id is not None else False}")
    print()
