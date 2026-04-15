"""Diagnose V-K basis overlap for Thm 6.17 V-inert re-examination.

Compute per-(layer, head) cosine similarity between B_K and B_V columns.
If |cos(B_K col, B_V col)| ≈ 1 for most columns → V-inert is genuine degeneracy.
If |cos| ≈ 0 → K-basis-on-V is an artifact; V-inert result is a code bug.

Also: compute Frobenius-norm similarity of BB^T projectors.
"""
import sys, json, argparse
from pathlib import Path
import torch
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--b-ont-k", required=True)
p.add_argument("--b-ont-v", required=True)
p.add_argument("--out", required=True)
args = p.parse_args()

def load_bont(path):
    o = torch.load(path, map_location="cpu", weights_only=False)
    return o["B_ont"] if isinstance(o, dict) else o

B_k = load_bont(args.b_ont_k).float()  # (L, H, d, R_k)
B_v = load_bont(args.b_ont_v).float()  # (L, H, d, R_v)
print(f"B_k: {tuple(B_k.shape)}, B_v: {tuple(B_v.shape)}")

L, H, d, R_k = B_k.shape
_, _, _, R_v = B_v.shape

# Per-(L,h) diagnostics
proj_fro_sims = []       # Frobenius norm of P_K - P_V, normalized
proj_inner_sims = []     # <P_K, P_V>_F / (||P_K||_F · ||P_V||_F), captures directional overlap
max_col_cos = []         # max |cos(col_i of B_K, col_j of B_V)| per (L, h)
mean_col_cos = []        # mean over column pairs
subspace_principal_angles = []  # top singular value of B_K^T B_V (== cos of smallest principal angle)

for l in range(L):
    for h in range(H):
        Bk = B_k[l, h]  # (d, R_k)
        Bv = B_v[l, h]  # (d, R_v)
        # Skip zero-padded columns
        Bk_nz = Bk[:, Bk.norm(dim=0) > 1e-6]
        Bv_nz = Bv[:, Bv.norm(dim=0) > 1e-6]
        if Bk_nz.shape[1] == 0 or Bv_nz.shape[1] == 0:
            continue
        # Column-wise cos
        cos_mat = Bk_nz.T @ Bv_nz / (
            Bk_nz.norm(dim=0, keepdim=True).T * Bv_nz.norm(dim=0, keepdim=True) + 1e-12
        )  # (R_k_nz, R_v_nz)
        abs_cos = cos_mat.abs()
        max_col_cos.append(abs_cos.max().item())
        mean_col_cos.append(abs_cos.mean().item())
        # Principal angles via SVD of B_K^T B_V (both column-normalized → SVs are cos of angles)
        Bk_n = Bk_nz / Bk_nz.norm(dim=0, keepdim=True)
        Bv_n = Bv_nz / Bv_nz.norm(dim=0, keepdim=True)
        s = torch.linalg.svdvals(Bk_n.T @ Bv_n)
        subspace_principal_angles.append(s.tolist())
        # Projector diff
        P_k = Bk_nz @ torch.linalg.pinv(Bk_nz.T @ Bk_nz) @ Bk_nz.T
        P_v = Bv_nz @ torch.linalg.pinv(Bv_nz.T @ Bv_nz) @ Bv_nz.T
        proj_inner = (P_k * P_v).sum() / (P_k.norm() * P_v.norm() + 1e-12)
        proj_inner_sims.append(proj_inner.item())
        proj_fro_sims.append((P_k - P_v).norm().item() / (P_k.norm() + P_v.norm() + 1e-12))

max_col_cos = np.array(max_col_cos)
mean_col_cos = np.array(mean_col_cos)
proj_inner_sims = np.array(proj_inner_sims)
proj_fro_sims = np.array(proj_fro_sims)

# Top principal angle is cos of smallest angle between subspaces
top_sv = np.array([max(s) if s else 0.0 for s in subspace_principal_angles])

summary = {
    "B_k_shape": list(B_k.shape), "B_v_shape": list(B_v.shape),
    "n_heads_sampled": len(max_col_cos),
    "max_col_cos_mean": float(max_col_cos.mean()),
    "max_col_cos_median": float(np.median(max_col_cos)),
    "max_col_cos_p5": float(np.percentile(max_col_cos, 5)),
    "max_col_cos_p95": float(np.percentile(max_col_cos, 95)),
    "mean_col_cos_mean": float(mean_col_cos.mean()),
    "mean_col_cos_median": float(np.median(mean_col_cos)),
    "top_principal_sv_mean": float(top_sv.mean()),
    "top_principal_sv_median": float(np.median(top_sv)),
    "top_principal_sv_p5": float(np.percentile(top_sv, 5)),
    "top_principal_sv_p95": float(np.percentile(top_sv, 95)),
    "proj_inner_product_mean": float(proj_inner_sims.mean()),
    "proj_inner_product_median": float(np.median(proj_inner_sims)),
    "proj_fro_diff_mean": float(proj_fro_sims.mean()),
}

print("\n=== V-K basis overlap ===")
for k, v in summary.items(): print(f"  {k}: {v}")

# Interpretation
top_mean = summary["top_principal_sv_mean"]
mean_col = summary["mean_col_cos_mean"]
print("\n=== Interpretation ===")
if top_mean > 0.95:
    print("  top principal SV > 0.95 → K and V subspaces nearly identical")
    print("  → V-inert result (Thm 6.17) likely GENUINE (not artifact)")
elif top_mean > 0.7:
    print("  top principal SV ~ 0.7-0.95 → substantial overlap")
    print("  → V-inert result PARTIALLY driven by K-basis-on-V but subspace shares most of V-facet info")
elif top_mean > 0.3:
    print("  top principal SV ~ 0.3-0.7 → moderate overlap")
    print("  → V-inert result ambiguous — needs full-eval V-basis test (we have one running)")
else:
    print("  top principal SV < 0.3 → K and V subspaces mostly orthogonal")
    print("  → V-inert result LIKELY ARTIFACT of K-basis-on-V; paper Finding 1 confirmed")

Path(args.out).parent.mkdir(parents=True, exist_ok=True)
with open(args.out, "w") as f: json.dump(summary, f, indent=2)
print(f"wrote {args.out}")
