"""Tier 3 design-space ablation: build AdaSEKA expert variants for comparison.

Variants:
  A: current (facet-split + uniform SV) — already built via build_adaseka_experts_from_bont.py
  B: no split (single expert spanning all B_ont columns)
  D: random orthonormal (same shapes as A, random directions)

Output layout:
  external/SEKA/seka_projections/adaseka-qwen25-7b-tau2-{domain}-{variant}/
    facet_*_svd.pt   (+ expert_paths.json)

Usage:
  python build_adaseka_variants_tier3.py \
    --b-ont external/SEKA/seka_projections/ontology-qwen25-7b-tau2-telecom/B_ont.pt \
    --out-base external/SEKA/seka_projections/adaseka-qwen25-7b-tau2-telecom \
    --variant B \
    --r-per-facet 1 3 5 3 \
    --layers-spec last10
"""
import argparse
import json
from pathlib import Path

import torch


def parse_layers_spec(spec: str, n_layers: int):
    if spec == "all":
        return list(range(n_layers))
    if spec.startswith("last"):
        return list(range(n_layers - int(spec[4:]), n_layers))
    if spec.startswith("first"):
        return list(range(int(spec[5:])))
    return [int(x) for x in spec.split(",")]


def build_variant_B_nosplit(B, sel_layers, out_dir: Path, facet_order, seed: int = 0):
    """Single expert = all B_ont columns (no facet split)."""
    L_sel = len(sel_layers)
    H, d, R = B.shape[1], B.shape[2], B.shape[3]
    B_sel = B[sel_layers].float()
    # Pad to d
    U = torch.zeros(L_sel, H, d, d, dtype=torch.float32)
    U[..., :R] = B_sel
    S = torch.zeros(L_sel, H, d, dtype=torch.float32)
    S[..., :R] = 1.0

    out_dir.mkdir(parents=True, exist_ok=True)
    expert_file = out_dir / "facet_all_svd.pt"
    torch.save({"layers": sel_layers, "U_matrices": U, "singular_values": S}, expert_file)
    expert_paths = {"all": str(expert_file.relative_to(Path("external/SEKA")))}
    json.dump(expert_paths, open(out_dir / "expert_paths.json", "w"), indent=2)
    print(f"[variant B] single expert: U {tuple(U.shape)}, R={R}")
    return expert_paths


def build_variant_D_random(B_shape, sel_layers, r_per_facet, facet_order, out_dir: Path, seed: int = 0):
    """Random orthonormal experts with identical shapes to variant A."""
    torch.manual_seed(seed)
    L_sel = len(sel_layers)
    _, H, d, _ = B_shape

    out_dir.mkdir(parents=True, exist_ok=True)
    expert_paths = {}
    for f_name, r_f in zip(facet_order, r_per_facet):
        # Random orthonormal (L_sel, H, d, d)
        U = torch.zeros(L_sel, H, d, d, dtype=torch.float32)
        for li in range(L_sel):
            for h in range(H):
                # Generate random dxd matrix, orthonormalize
                A = torch.randn(d, d)
                Q, _ = torch.linalg.qr(A)
                U[li, h] = Q
        # Singular values uniform on first r_f (match variant A pattern)
        S = torch.zeros(L_sel, H, d, dtype=torch.float32)
        S[..., :r_f] = 1.0

        expert_file = out_dir / f"facet_{f_name}_svd.pt"
        torch.save({"layers": sel_layers, "U_matrices": U, "singular_values": S}, expert_file)
        expert_paths[f_name] = str(expert_file.relative_to(Path("external/SEKA")))
        print(f"[variant D] facet {f_name}: random orthonormal U {tuple(U.shape)}, r_f={r_f}")

    json.dump(expert_paths, open(out_dir / "expert_paths.json", "w"), indent=2)
    return expert_paths


def build_variant_C_unsup_sv(B, sel_layers, r_per_facet, facet_order, out_dir: Path,
                              model_id: str, domain: str, seed: int = 0):
    """Unsupervised real SVs: build facet centroids from actual K activations and SVD them.

    Approach:
      - For each facet f with r_f dimensions, the unsupervised SV derivation sets
        S_f[:r_f] = sorted singular values of the per-facet K centroid projection onto
        B_f subspace. This requires K activations from real data.

    Simplification (since we don't have access to K stats here):
      - Use B_f's existing geometry: compute per-(L,H) actual singular values of B_f
        when viewed as a projection matrix P_f = B_f @ B_f.T. The "strength" of each
        direction in U is sqrt(eigenvalue of P_f in that direction).
      - In our case B_f columns are already orthonormal (per build_qwen_metatool_b_ont
        construction), so all eigenvalues = 1 → degenerate.
      - Instead: approximate unsupervised real SV by the COLUMN NORMS of B_f BEFORE
        orthonormalization. But we don't have that.

    Therefore variant C is SKIPPED in this implementation — requires K activation
    extraction pipeline we don't have. Noted as "deferred" in Tier 3 table.
    """
    raise NotImplementedError(
        "Variant C (unsupervised real SV from K stats) deferred — requires K activation "
        "extraction pipeline not yet implemented."
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--b-ont", required=True)
    p.add_argument("--out-base", required=True, help="Base output dir (variant suffix auto-added)")
    p.add_argument("--variant", required=True, choices=["B", "D"])
    p.add_argument("--facet-order", nargs="+",
                   default=["function_action", "io_type", "domain", "tool_category"])
    p.add_argument("--r-per-facet", nargs="+", type=int, default=None)
    p.add_argument("--layers-spec", default="last10")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    bobj = torch.load(args.b_ont, map_location="cpu", weights_only=False)
    B = bobj["B_ont"] if isinstance(bobj, dict) else bobj
    L, H, d, R = B.shape
    print(f"[B_ont] L={L} H={H} d={d} R={R}")
    sel_layers = parse_layers_spec(args.layers_spec, L)
    print(f"[sel_layers] {sel_layers}")

    if args.r_per_facet is not None:
        r_per_facet = args.r_per_facet
    else:
        F = len(args.facet_order)
        base = R // F
        r_per_facet = [base] * F
        r_per_facet[-1] = R - base * (F - 1)
    print(f"[r_per_facet] {r_per_facet}")
    assert sum(r_per_facet) <= R

    out_dir = Path(f"{args.out_base}-variant{args.variant}")
    if args.variant == "B":
        build_variant_B_nosplit(B, sel_layers, out_dir, args.facet_order, args.seed)
    elif args.variant == "D":
        build_variant_D_random(B.shape, sel_layers, r_per_facet, args.facet_order, out_dir, args.seed)
    print(f"\n[saved variant {args.variant}] {out_dir}/expert_paths.json")


if __name__ == "__main__":
    main()
