"""measure_l01_cosine.py
L01에 peak를 갖는 관계들의 steering vector pairwise cosine similarity 측정.

steering vector = mean(pos_hidden@L01) - mean(neg_hidden@L01)
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer

# L01 peak 관계 (check_results_v3.py 결과 기준)
L01_RELS = [
    "validates", "causal_link", "enables", "backtrack_to",
    "requires", "tool_subsumes",          # A6_peak
    "scored_preference", "optional_in_flow", "loop_capable", "parallel_safe",  # T1_qonly
]
TARGET_LAYER = 1  # L01


def extract_at_layer(texts, model, tokenizer, device, layer_idx,
                     batch_size=8, max_length=300):
    model.eval()
    vecs = []
    for s in range(0, len(texts), batch_size):
        batch = texts[s: s + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True,
                        truncation=True, max_length=max_length).to(device)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True, return_dict=True)
        hs   = out.hidden_states[layer_idx]          # (B, seq, d)
        mask = enc["attention_mask"]
        last = mask.sum(1) - 1                       # (B,)
        idx  = last.view(-1, 1, 1).expand(-1, 1, hs.size(-1))
        vecs.append(hs.gather(1, idx).squeeze(1).cpu().float().numpy())
    return np.concatenate(vecs, axis=0)              # (N, d)


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", default=(
        "/home/woori/workspace_common/boltzmann-attention-pi/"
        "reports/facet_rft_2026/phase0_probing/contrast_pairs_v3.json"))
    p.add_argument("--model",  default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size",  type=int, default=8)
    p.add_argument("--max-length",  type=int, default=300)
    p.add_argument("--out", default=(
        "/home/woori/workspace_common/boltzmann-attention-pi/"
        "reports/facet_rft_2026/phase0_probing/l01_cosine.json"))
    args = p.parse_args()

    with open(args.pairs) as f:
        data = json.load(f)
    pairs = data["pairs"]

    # 관계별 텍스트/레이블 분류
    by_rel: dict = {}
    for pp in pairs:
        pt = pp["pair_type"].lower()
        if pt not in L01_RELS:
            continue
        by_rel.setdefault(pt, {"texts": [], "labels": []})
        by_rel[pt]["texts"].append(pp["text"])
        by_rel[pt]["labels"].append(pp["label"])

    missing = [r for r in L01_RELS if r not in by_rel]
    if missing:
        print(f"[warn] pairs에 없는 관계: {missing}")

    print(f"모델 로딩: {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16,
        device_map=args.device, trust_remote_code=True)
    mdl.eval()

    # steering vector 추출
    sv: dict[str, np.ndarray] = {}
    for rel in L01_RELS:
        if rel not in by_rel:
            continue
        info = by_rel[rel]
        texts  = info["texts"]
        labels = np.array(info["labels"])
        print(f"  [{rel}] {len(texts)} samples → L{TARGET_LAYER:02d}")
        h = extract_at_layer(texts, mdl, tok, args.device,
                             TARGET_LAYER, args.batch_size, args.max_length)
        scaler = StandardScaler()
        h = scaler.fit_transform(h)
        pos = h[labels == 1].mean(axis=0)
        neg = h[labels == 0].mean(axis=0)
        vec = pos - neg
        sv[rel] = vec / (np.linalg.norm(vec) + 1e-12)

    rels = [r for r in L01_RELS if r in sv]
    n = len(rels)

    # pairwise cosine matrix
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            mat[i, j] = cosine(sv[rels[i]], sv[rels[j]])

    print("\n" + "=" * 80)
    print(f"L{TARGET_LAYER:02d} Steering Vector — Pairwise Cosine Similarity  (n={n})")
    print("=" * 80)
    header = f"{'':24s}" + "".join(f"{r[:8]:>10s}" for r in rels)
    print(header)
    print("-" * len(header))
    for i, ri in enumerate(rels):
        row = f"{ri:24s}" + "".join(f"{mat[i,j]:>10.3f}" for j in range(n))
        print(row)

    # 상삼각 off-diagonal 통계
    vals = [mat[i, j] for i in range(n) for j in range(i + 1, n)]
    print(f"\n  off-diagonal cosine  mean={np.mean(vals):.3f}  "
          f"max={np.max(vals):.3f}  min={np.min(vals):.3f}")
    high = [(rels[i], rels[j], mat[i, j])
            for i in range(n) for j in range(i + 1, n) if mat[i, j] > 0.5]
    if high:
        print(f"\n  ⚠ cosine > 0.5 쌍 ({len(high)}개) — 직교화 필요:")
        for a, b, v in sorted(high, key=lambda x: -x[2]):
            print(f"    {a:24s} ↔ {b:24s}  {v:.3f}")
    else:
        print("\n  ✓ 모든 쌍 cosine ≤ 0.5 — 자연 직교, 간섭 위험 낮음")

    # JSON 저장
    out = {
        "layer": TARGET_LAYER,
        "relations": rels,
        "cosine_matrix": mat.tolist(),
        "stats": {
            "mean": float(np.mean(vals)),
            "max":  float(np.max(vals)),
            "min":  float(np.min(vals)),
        },
        "high_cosine_pairs": [
            {"a": a, "b": b, "cosine": round(v, 4)} for a, b, v in high
        ],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  → {args.out}")


if __name__ == "__main__":
    main()
