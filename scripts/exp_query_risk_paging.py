#!/usr/bin/env python3
"""
Query-Dynamic Risk Paging diagnostic.

This script isolates the core question behind query-time selective refinement:
does a quantization-aware page-risk score recover the true winner's page more
reliably than score-only selection under the same page budget?

The implementation is intentionally small and synthetic-first. It is a kill
criterion runner, not a full benchmark harness.
"""

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

try:
    import torch
except ImportError as exc:
    raise SystemExit(
        "torch is required. Run this script with a torch-enabled env, e.g. "
        "`~/miniconda3/envs/vllm-server/bin/python scripts/exp_query_risk_paging.py ...`"
    ) from exc


def normal_sf(z: torch.Tensor) -> torch.Tensor:
    """Standard normal survival function."""
    return 0.5 * torch.erfc(z / math.sqrt(2.0))


def page_start_indices(num_tokens: int, page_size: int) -> List[int]:
    return list(range(0, num_tokens, page_size))


def reduce_pages(values: torch.Tensor, page_size: int, reduce: str) -> torch.Tensor:
    if values.dim() != 1:
        raise ValueError(f"Expected 1D tensor, got shape {tuple(values.shape)}")
    parts = []
    for start in page_start_indices(values.numel(), page_size):
        chunk = values[start:start + page_size]
        if reduce == "sum":
            parts.append(chunk.sum())
        elif reduce == "max":
            parts.append(chunk.max())
        elif reduce == "min":
            parts.append(chunk.min())
        else:
            raise ValueError(f"Unknown reduction: {reduce}")
    return torch.stack(parts)


def token_flip_risk(quant_scores: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """
    Approximate probability that each challenger can overtake the current
    quantized winner, under Gaussian score-noise.
    """
    winner = int(torch.argmax(quant_scores).item())
    margin = quant_scores[winner] - quant_scores
    pair_std = torch.sqrt((sigma[winner] ** 2 + sigma ** 2).clamp(min=1e-12))
    risk = normal_sf(margin / pair_std)
    risk[winner] = 0.0
    return risk


def select_pages(
    quant_scores: torch.Tensor,
    sigma: torch.Tensor,
    page_size: int,
    budget_pages: int,
    mode: str,
    true_scores: torch.Tensor | None = None,
) -> torch.Tensor:
    if budget_pages <= 0:
        raise ValueError("budget_pages must be positive")
    n_pages = math.ceil(quant_scores.numel() / page_size)
    budget_pages = min(budget_pages, n_pages)

    if mode == "score":
        page_metric = reduce_pages(quant_scores, page_size, reduce="max")
        return torch.topk(page_metric, k=budget_pages).indices
    elif mode == "margin":
        winner = int(torch.argmax(quant_scores).item())
        margin = quant_scores[winner] - quant_scores
        margin[winner] = 0.0
        page_metric = -reduce_pages(margin, page_size, reduce="min")
        return torch.topk(page_metric, k=budget_pages).indices
    elif mode == "risk":
        page_metric = reduce_pages(token_flip_risk(quant_scores, sigma), page_size, reduce="sum")
        return torch.topk(page_metric, k=budget_pages).indices
    elif mode == "hybrid":
        if budget_pages == 1:
            page_metric = reduce_pages(token_flip_risk(quant_scores, sigma), page_size, reduce="sum")
            return torch.topk(page_metric, k=1).indices

        score_metric = reduce_pages(quant_scores, page_size, reduce="max")
        risk_metric = reduce_pages(token_flip_risk(quant_scores, sigma), page_size, reduce="sum")
        first = int(torch.argmax(score_metric).item())
        chosen = [first]
        if budget_pages > 1:
            risk_metric = risk_metric.clone()
            risk_metric[first] = float("-inf")
            rest = torch.topk(risk_metric, k=budget_pages - 1).indices.tolist()
            chosen.extend(rest)
        return torch.tensor(chosen, dtype=torch.long)
    elif mode == "oracle":
        if true_scores is None:
            raise ValueError("oracle selection requires true_scores")
        winner_page = int(torch.argmax(true_scores).item()) // page_size
        page_metric = torch.zeros(n_pages, dtype=quant_scores.dtype)
        page_metric[winner_page] = 1.0
        return torch.topk(page_metric, k=budget_pages).indices
    else:
        raise ValueError(f"Unknown selection mode: {mode}")


@dataclass
class TrialResult:
    winner_page_hit: float
    recovered_top1: float


def evaluate_selection(
    true_scores: torch.Tensor,
    quant_scores: torch.Tensor,
    sigma: torch.Tensor,
    page_size: int,
    budget_pages: int,
    mode: str,
) -> TrialResult:
    selected = select_pages(
        quant_scores=quant_scores,
        sigma=sigma,
        page_size=page_size,
        budget_pages=budget_pages,
        mode=mode,
        true_scores=true_scores,
    )
    winner = int(torch.argmax(true_scores).item())
    winner_page = winner // page_size
    page_hit = float((selected == winner_page).any().item())

    refined = quant_scores.clone()
    for page_idx in selected.tolist():
        start = page_idx * page_size
        refined[start:start + page_size] = true_scores[start:start + page_size]
    recovered = float(int(torch.argmax(refined).item()) == winner)
    return TrialResult(winner_page_hit=page_hit, recovered_top1=recovered)


def make_synthetic_case(
    num_tokens: int,
    page_size: int,
    generator: torch.Generator,
) -> Dict[str, torch.Tensor]:
    """
    Generate a setting where score-only selection is often fooled by a wrong but
    low-variance page, while a high-variance near-winner page should have high
    risk.
    """
    n_pages = math.ceil(num_tokens / page_size)
    true_scores = torch.randn(num_tokens, generator=generator) * 0.12
    sigma = torch.full((num_tokens,), 0.03)

    true_winner_page = int(torch.randint(low=0, high=n_pages, size=(1,), generator=generator).item())
    distractor_page = (true_winner_page + 1 + int(torch.randint(low=0, high=max(1, n_pages - 1), size=(1,), generator=generator).item())) % n_pages

    winner_offset = int(torch.randint(low=0, high=page_size, size=(1,), generator=generator).item())
    distractor_offset = int(torch.randint(low=0, high=page_size, size=(1,), generator=generator).item())

    winner_idx = true_winner_page * page_size + winner_offset
    distractor_idx = distractor_page * page_size + distractor_offset
    winner_idx = min(winner_idx, num_tokens - 1)
    distractor_idx = min(distractor_idx, num_tokens - 1)

    true_scores[winner_idx] = 1.00 + torch.rand(1, generator=generator).item() * 0.08
    true_scores[distractor_idx] = 0.93 + torch.rand(1, generator=generator).item() * 0.04

    winner_start = true_winner_page * page_size
    winner_end = min(num_tokens, winner_start + page_size)
    sigma[winner_start:winner_end] = 0.20

    distractor_start = distractor_page * page_size
    distractor_end = min(num_tokens, distractor_start + page_size)
    sigma[distractor_start:distractor_end] = 0.02

    noise = torch.randn(num_tokens, generator=generator) * sigma
    quant_scores = true_scores + noise
    quant_scores[distractor_idx] += 0.06
    quant_scores[winner_idx] -= 0.06

    return {
        "true_scores": true_scores,
        "quant_scores": quant_scores,
        "sigma": sigma,
    }


def run_synthetic_experiment(
    trials: int,
    num_tokens: int,
    page_size: int,
    budget_pages: int,
    seed: int,
) -> Dict[str, Dict[str, float]]:
    generator = torch.Generator().manual_seed(seed)
    aggregate = {
        "score": {"page_hit": 0.0, "recover": 0.0},
        "margin": {"page_hit": 0.0, "recover": 0.0},
        "risk": {"page_hit": 0.0, "recover": 0.0},
        "hybrid": {"page_hit": 0.0, "recover": 0.0},
        "oracle": {"page_hit": 0.0, "recover": 0.0},
    }

    for _ in range(trials):
        case = make_synthetic_case(num_tokens=num_tokens, page_size=page_size, generator=generator)
        for mode in aggregate:
            result = evaluate_selection(
                true_scores=case["true_scores"],
                quant_scores=case["quant_scores"],
                sigma=case["sigma"],
                page_size=page_size,
                budget_pages=budget_pages,
                mode=mode,
            )
            aggregate[mode]["page_hit"] += result.winner_page_hit
            aggregate[mode]["recover"] += result.recovered_top1

    for mode in aggregate:
        aggregate[mode]["page_hit"] /= trials
        aggregate[mode]["recover"] /= trials

    aggregate["risk_vs_score_page_hit"] = {
        "delta": aggregate["risk"]["page_hit"] - aggregate["score"]["page_hit"]
    }
    aggregate["risk_vs_score_recover"] = {
        "delta": aggregate["risk"]["recover"] - aggregate["score"]["recover"]
    }
    aggregate["hybrid_vs_score_recover"] = {
        "delta": aggregate["hybrid"]["recover"] - aggregate["score"]["recover"]
    }
    return aggregate


def run_self_tests() -> None:
    scores = torch.tensor([1.0, 0.94, 0.80, 0.70])
    sigma = torch.tensor([0.02, 0.20, 0.03, 0.03])
    risk = token_flip_risk(scores, sigma)
    assert risk[1] > risk[2], "high-variance near challenger should carry larger flip risk"

    metrics = run_synthetic_experiment(
        trials=512,
        num_tokens=64,
        page_size=8,
        budget_pages=1,
        seed=0,
    )
    assert metrics["oracle"]["page_hit"] == 1.0
    assert metrics["risk"]["page_hit"] > metrics["score"]["page_hit"], metrics
    assert metrics["risk"]["recover"] >= metrics["score"]["recover"], metrics
    print("[PASS] exp_query_risk_paging self-tests")
    print(json.dumps(metrics, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["self_test", "synthetic"], default="self_test")
    parser.add_argument("--trials", type=int, default=2000)
    parser.add_argument("--num-tokens", type=int, default=256)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--budget-pages", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.mode == "self_test":
        run_self_tests()
        return

    metrics = run_synthetic_experiment(
        trials=args.trials,
        num_tokens=args.num_tokens,
        page_size=args.page_size,
        budget_pages=args.budget_pages,
        seed=args.seed,
    )
    payload = {
        "mode": args.mode,
        "trials": args.trials,
        "num_tokens": args.num_tokens,
        "page_size": args.page_size,
        "budget_pages": args.budget_pages,
        "seed": args.seed,
        "metrics": metrics,
    }
    text = json.dumps(payload, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
