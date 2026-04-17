"""Build AdaSEKA canonical expert files from our B_ont per-facet basis.

Takes:
- B_ont tensor (L, H, d, R) where R = sum_f r_f
- Per-facet rank allocation (r_per_facet: list of 4 ints summing to R)

Produces one expert SVD file per facet in AdaSEKA's expected format:
- U_matrices: (L_selected, H, d, d) — pad per-facet B_f columns with zeros to d×d
- singular_values: (L_selected, H, d) — [1..1, 0..0] (r_f ones then zeros)
- layers: list of selected layer indices

Then emits expert_paths.json mapping facet name → file path.

Usage:
  python build_adaseka_experts_from_bont.py \
    --b-ont external/SEKA/seka_projections/ontology-qwen25-7b-metatool/B_ont.pt \
    --diag reports/axis2_theoretical_verification/build_qwen_metatool_b_ont.json \
    --out-dir external/SEKA/seka_projections/adaseka-qwen25-7b-metatool/ \
    --layers-spec last10
"""
import argparse, json
from pathlib import Path
import torch


def parse_layers_spec(spec: str, n_layers: int):
    if spec == "all":
        return list(range(n_layers))
    if spec.startswith("last"):
        k = int(spec[4:])
        return list(range(n_layers - k, n_layers))
    if spec.startswith("first"):
        k = int(spec[5:])
        return list(range(k))
    return [int(x) for x in spec.split(",")]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--b-ont", required=True)
    p.add_argument("--diag", default=None, help="Build diagnostic JSON for per-facet rank allocation")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--layers-spec", default="last10")
    p.add_argument("--facet-order", nargs="+",
                   default=["function_action", "io_type", "domain", "tool_category"])
    p.add_argument("--r-per-facet", nargs="+", type=int, default=None,
                   help="Per-facet rank; if None, infer evenly from B_ont total rank R")
    args = p.parse_args()

    bobj = torch.load(args.b_ont, map_location="cpu", weights_only=False)
    B = bobj["B_ont"] if isinstance(bobj, dict) else bobj
    L, H, d, R = B.shape
    print(f"B_ont shape: L={L} H={H} d={d} R={R}")

    # Determine per-facet rank allocation
    if args.r_per_facet is not None:
        r_per_facet = args.r_per_facet
    elif args.diag is not None and Path(args.diag).exists():
        diag = json.load(open(args.diag))
        r_per_facet = diag.get("r_per_facet_median", None)
        if r_per_facet is None:
            # Fall back
            r_per_facet = None
    if 'r_per_facet' not in dir() or r_per_facet is None:
        # Heuristic: even split across facets
        F = len(args.facet_order)
        base = R // F
        r_per_facet = [base] * F
        r_per_facet[-1] = R - base * (F - 1)
    print(f"r_per_facet: {r_per_facet}")
    assert sum(r_per_facet) <= R, f"sum r_per_facet {sum(r_per_facet)} > R {R}"
    assert len(r_per_facet) == len(args.facet_order)

    sel_layers = parse_layers_spec(args.layers_spec, L)
    print(f"Selected layers: {sel_layers}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Slice B_ont to selected layers
    B_sel = B[sel_layers]  # (L_sel, H, d, R)
    L_sel = len(sel_layers)

    expert_paths = {}
    offset = 0
    for f_name, r_f in zip(args.facet_order, r_per_facet):
        # Extract per-facet block: (L_sel, H, d, r_f)
        B_f = B_sel[..., offset:offset + r_f].float()
        # Pad to (L_sel, H, d, d): [B_f | 0]
        U = torch.zeros(L_sel, H, d, d, dtype=torch.float32)
        U[..., :r_f] = B_f
        # Singular values: [1..1, 0..0]
        S = torch.zeros(L_sel, H, d, dtype=torch.float32)
        S[..., :r_f] = 1.0

        expert_file = out_dir / f"facet_{f_name}_svd.pt"
        torch.save({"layers": sel_layers, "U_matrices": U, "singular_values": S}, expert_file)
        # Use relative path (from external/SEKA/) as AdaSEKA's convention
        rel_path = str(expert_file.relative_to(Path("external/SEKA"))) if str(expert_file).startswith("external/SEKA") else str(expert_file)
        expert_paths[f_name] = rel_path
        print(f"  facet {f_name}: r_f={r_f}, saved → {expert_file}")
        offset += r_f

    exp_path_file = out_dir / "expert_paths.json"
    json.dump(expert_paths, open(exp_path_file, "w"), indent=2)
    print(f"\nexpert_paths.json → {exp_path_file}")
    print(json.dumps(expert_paths, indent=2))


if __name__ == "__main__":
    main()
