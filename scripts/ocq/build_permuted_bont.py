#!/usr/bin/env python3
"""Phase C — Catalog-permutation falsifier.

Takes an existing tau2/metatool ontology JSON, permutes (a) category → sentence
mapping within each facet, or (b) tool names across categories, or (c) both.
Writes a new permuted ontology JSON ready for ``build_qwen_metatool_b_ont.py``.

Usage:
    python build_permuted_bont.py \
      --catalog reports/axis2_theoretical_verification/tau2_telecom_ontology.json \
      --mode facet_values --seed 42 \
      --out reports/axis2_theoretical_verification/tau2_telecom_ontology_perm_seed42.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
from copy import deepcopy


def permute_facet_values(ontology_dict, rng):
    """Within each facet, shuffle the mapping category -> sentences.

    i.e. facet "function_action" has keys {get, send, transfer, other}; we take
    the list of sentences attached to each key and shuffle the assignment, so
    e.g. "get" gets the sentences that used to belong to "transfer".
    """
    out = deepcopy(ontology_dict)
    for facet, cat_dict in out.items():
        keys = list(cat_dict.keys())
        vals = list(cat_dict.values())
        perm = list(range(len(vals)))
        rng.shuffle(perm)
        new_cat = {keys[i]: vals[perm[i]] for i in range(len(keys))}
        out[facet] = new_cat
    return out


def permute_tool_names(ontology_dict, rng):
    """Within each facet, collect all example tool names (from sentences starting
    with "Example tool XXX:"), then permute the names and reinsert.
    """
    import re
    EX_RE = re.compile(r"Example tool ([A-Za-z0-9_\-]+):")

    out = deepcopy(ontology_dict)
    for facet, cat_dict in out.items():
        # collect name positions
        positions = []  # list of (cat, idx_in_list, name)
        names = []
        for cat, sents in cat_dict.items():
            for i, s in enumerate(sents):
                m = EX_RE.search(s)
                if m:
                    positions.append((cat, i, m.group(1)))
                    names.append(m.group(1))
        rng.shuffle(names)
        for (cat, i, _orig), new_name in zip(positions, names):
            old = cat_dict[cat][i]
            cat_dict[cat][i] = EX_RE.sub(f"Example tool {new_name}:", old, count=1)
    return out


def full_random_sentences(ontology_dict, rng, vocab=None):
    """Replace every sentence with a short random-gibberish sentence.
    This destroys catalog semantics entirely.
    """
    out = deepcopy(ontology_dict)
    vocab = vocab or [
        "alpha", "beta", "gamma", "delta", "epsilon", "foo", "bar", "baz",
        "quux", "plurp", "fizz", "buzz", "nonce", "xyz", "blank", "thing"
    ]
    for facet, cat_dict in out.items():
        for cat, sents in cat_dict.items():
            new_sents = []
            for s in sents:
                length = rng.randint(6, 12)
                new_sents.append(" ".join(rng.choice(vocab) for _ in range(length)) + ".")
            cat_dict[cat] = new_sents
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--catalog", required=True)
    p.add_argument("--mode", required=True,
                   choices=["facet_values", "tool_names", "full_random"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    rng = random.Random(args.seed)
    with open(args.catalog) as f:
        full = json.load(f)

    ontology = full["ontology"]
    if args.mode == "facet_values":
        permuted = permute_facet_values(ontology, rng)
    elif args.mode == "tool_names":
        permuted = permute_tool_names(ontology, rng)
    elif args.mode == "full_random":
        permuted = full_random_sentences(ontology, rng)
    else:
        raise ValueError(args.mode)

    out_obj = deepcopy(full)
    out_obj["ontology"] = permuted
    out_obj["_permutation"] = {"mode": args.mode, "seed": args.seed, "source": args.catalog}

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out_obj, f, indent=2, ensure_ascii=False)
    print(f"[permuted] mode={args.mode} seed={args.seed}")
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
