#!/usr/bin/env python3
"""Expression diversity D measurement (EXPRESSION_DIVERSITY_TRANSFER_DESIGN §3).
We measure SURFACE diversity (lexical/syntactic variation among paraphrases of the SAME op),
not semantic similarity — so we embed with char n-gram TF-IDF (sentence-embeddings would
collapse same-op paraphrases to high similarity and HIDE surface variation). D reported as
multiple indicators (no single-metric claim): effective rank (exp of singular-value entropy),
mean pairwise cosine distance. Also a greedy k-center selector for orthogonal-K sampling.
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances


def embed(texts):
    """Surface embedding: char_wb 2-4 grams + word unigrams. Dense array."""
    v = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)
    return v.fit_transform(texts).toarray()


def effective_rank(X):
    """exp(entropy of normalized singular values) = effective # of independent directions."""
    if len(X) < 2:
        return 1.0
    s = np.linalg.svd(X, compute_uv=False)
    s = s[s > 1e-9]
    if s.size == 0:
        return 1.0
    p = s / s.sum()
    H = -(p * np.log(p)).sum()
    return float(np.exp(H))


def mean_pairwise_dist(X):
    if len(X) < 2:
        return 0.0
    D = cosine_distances(X)
    iu = np.triu_indices(len(X), 1)
    return float(D[iu].mean())


def diversity(texts):
    """D indicators for a set of expression strings."""
    uniq = list(dict.fromkeys(texts))  # dedup, keep order
    X = embed(uniq)
    return {"eff_rank": effective_rank(X), "mean_dist": mean_pairwise_dist(X),
            "n_unique": len(uniq), "n_total": len(texts)}


def kcenter_indices(texts, K, seed=0):
    """Greedy max-min (k-center) on surface embeddings -> K indices spanning the set widely."""
    n = len(texts)
    if K >= n:
        return list(range(n))
    X = embed(texts)
    D = cosine_distances(X)
    start = seed % n
    sel = [start]
    while len(sel) < K:
        mind = D[:, sel].min(axis=1)
        for s in sel:
            mind[s] = -1.0
        sel.append(int(mind.argmax()))
    return sel


if __name__ == "__main__":
    import sys, json
    texts = [l.rstrip("\n") for l in sys.stdin if l.strip()]
    print(json.dumps(diversity(texts), indent=2))
