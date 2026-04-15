#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="Path to source B_ont payload")
    p.add_argument("--out", required=True, help="Path to output control payload")
    p.add_argument(
        "--mode",
        required=True,
        choices=["random_orthonormal", "feature_shuffle"],
        help=(
            "random_orthonormal = same shape/rank but fresh random basis per "
            "(layer,head); feature_shuffle = row-permuted version of the real "
            "basis, preserving rank/norm while breaking coordinate semantics."
        ),
    )
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def random_orthonormal_like(B: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    out = torch.zeros_like(B)
    L, H, d, r = B.shape
    for li in range(L):
        for hi in range(H):
            if torch.count_nonzero(B[li, hi]).item() == 0:
                continue
            M = torch.randn((d, r), generator=gen, dtype=torch.float32)
            Q, _ = torch.linalg.qr(M, mode="reduced")
            out[li, hi] = Q[:, :r]
    return out


def feature_shuffle_like(B: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    out = torch.zeros_like(B)
    L, H, d, _ = B.shape
    for li in range(L):
        for hi in range(H):
            if torch.count_nonzero(B[li, hi]).item() == 0:
                continue
            perm = torch.randperm(d, generator=gen)
            out[li, hi] = B[li, hi][perm]
    return out


def main() -> None:
    args = parse_args()
    payload = torch.load(args.src, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "B_ont" not in payload:
        raise ValueError("expected dict payload with 'B_ont'")
    B = payload["B_ont"].float()
    gen = torch.Generator().manual_seed(args.seed)

    if args.mode == "random_orthonormal":
        B_control = random_orthonormal_like(B, gen)
    elif args.mode == "feature_shuffle":
        B_control = feature_shuffle_like(B, gen)
    else:
        raise ValueError(args.mode)

    out_payload = dict(payload)
    out_payload["B_ont"] = B_control
    out_payload.pop("r_per_pair", None)
    out_payload.pop("facet_order", None)
    out_payload["control_mode"] = args.mode
    out_payload["control_seed"] = args.seed
    out_payload["control_src"] = str(args.src)
    out_payload["control_payload"] = True
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_payload, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
