"""Phase 2a — T1 Steering Vector 추출.
contrast_pairs_v3.json의 1608 pairs (42 relations × ~38 pairs)를 사용해
각 relation의 residual stream steering vector를 mean-of-differences로 추출.

Output: steering_vectors_<model_name>.pt
  format: {
    "metadata": {...},
    "vectors": {
      <relation_name>: {
        <layer_idx>: torch.Tensor[d_model]  # mean(pos) - mean(neg)
      }
    }
  }

Usage:
  python _extract_steering_vectors.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --pairs reports/facet_rft_2026/phase0_probing/contrast_pairs_v3.json \
    --out reports/facet_rft_2026/phase2_steering/steering_vectors_qwen7b.pt \
    --device cuda:0 \
    --layers all  # 또는 0,1,2,...,28 specific
"""
import argparse
import json
import time
from pathlib import Path
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model id")
    ap.add_argument("--pairs", required=True, help="contrast_pairs_v3.json")
    ap.add_argument("--out", required=True, help="output .pt path")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--layers", default="all", help="comma-separated indices or 'all'")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=2048)
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    print(f"[t1-extract] Loading {args.model}...")
    t0 = time.time()
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map=args.device,
        output_hidden_states=True, attn_implementation="eager",
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size
    print(f"[t1-extract] n_layers={n_layers}, d_model={d_model}")
    print(f"[t1-extract] loaded in {time.time()-t0:.1f}s")

    if args.layers == "all":
        layer_idxs = list(range(n_layers + 1))  # 0=embedding, 1..n_layers=after each block
    else:
        layer_idxs = [int(x) for x in args.layers.split(",")]
    print(f"[t1-extract] capturing layers: {layer_idxs}")

    # Load pairs
    d = json.load(open(args.pairs))
    pairs = d["pairs"]
    print(f"[t1-extract] {len(pairs)} pairs loaded")

    # Group by relation × label
    grouped = defaultdict(lambda: {"pos": [], "neg": []})
    for p in pairs:
        rel = p["pair_type"]
        side = "pos" if p["label"] == 1 else "neg"
        grouped[rel][side].append(p["text"])

    # Accumulate sums of last-token hidden states per (relation, layer, label)
    sums = {rel: {l: {"pos": None, "neg": None} for l in layer_idxs} for rel in grouped}
    counts = {rel: {"pos": 0, "neg": 0} for rel in grouped}

    @torch.no_grad()
    def encode_batch(texts):
        enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                        max_length=args.max_length).to(args.device)
        out = model(**enc, output_hidden_states=True)
        # out.hidden_states: tuple len=(n_layers+1), each [B, T, d_model]
        # Use last non-pad token per sample
        attn = enc["attention_mask"]
        last_idx = attn.sum(dim=1) - 1   # [B]
        result = {}
        for l in layer_idxs:
            h = out.hidden_states[l]    # [B, T, d_model]
            # Extract h[b, last_idx[b], :]
            batch_idx = torch.arange(h.size(0), device=h.device)
            last_hidden = h[batch_idx, last_idx]  # [B, d_model]
            result[l] = last_hidden.detach().to(torch.float32).cpu()
        return result

    t_start = time.time()
    relations = sorted(grouped.keys())
    for ri, rel in enumerate(relations):
        for side in ("pos", "neg"):
            texts = grouped[rel][side]
            for i in range(0, len(texts), args.batch_size):
                batch_texts = texts[i:i+args.batch_size]
                hs = encode_batch(batch_texts)
                for l in layer_idxs:
                    h_sum = hs[l].sum(dim=0)
                    if sums[rel][l][side] is None:
                        sums[rel][l][side] = h_sum
                    else:
                        sums[rel][l][side] = sums[rel][l][side] + h_sum
                counts[rel][side] += len(batch_texts)
        elapsed = time.time() - t_start
        eta = elapsed / (ri + 1) * (len(relations) - ri - 1)
        print(f"[t1-extract] [{ri+1}/{len(relations)}] {rel} pos={counts[rel]['pos']} neg={counts[rel]['neg']}  elapsed={elapsed:.1f}s eta={eta:.1f}s")

    # Compute mean-of-differences
    vectors = {}
    for rel in relations:
        vectors[rel] = {}
        for l in layer_idxs:
            pos_mean = sums[rel][l]["pos"] / counts[rel]["pos"]
            neg_mean = sums[rel][l]["neg"] / counts[rel]["neg"]
            vectors[rel][l] = (pos_mean - neg_mean)

    print(f"[t1-extract] Saving to {args.out}")
    torch.save({
        "metadata": {
            "model": args.model,
            "n_layers": n_layers,
            "d_model": d_model,
            "layer_idxs": layer_idxs,
            "n_pairs": len(pairs),
            "by_relation_counts": {r: dict(counts[r]) for r in relations},
        },
        "vectors": vectors,
    }, args.out)
    print(f"[t1-extract] Done. Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
