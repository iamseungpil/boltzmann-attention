#!/usr/bin/env python3
"""
Tiny Recursive Innovation Cache diagnostic.

This is a synthetic-first runner for the depth-direction hypothesis:
can a tiny shared recursive predictor explain cross-layer KV evolution better
than copy-last or a shared linear map, making innovation coding plausible?
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as exc:
    raise SystemExit(
        "torch is required. Run this script with a torch-enabled env, e.g. "
        "`~/miniconda3/envs/vllm-server/bin/python scripts/exp_tric_recursive_predictor.py ...`"
    ) from exc

torch.set_num_threads(1)


def generate_depth_process(
    batch: int,
    layers: int,
    dim: int,
    seed: int,
) -> torch.Tensor:
    """
    Synthetic nonlinear depth process with a shared hidden mechanism.
    Shape: (batch, layers, dim)
    """
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(batch, dim, generator=g)
    seq = [base]
    W1 = torch.randn(dim, dim, generator=g) / dim**0.5
    W2 = torch.randn(dim, dim, generator=g) / dim**0.5
    for layer in range(layers - 1):
        x = seq[-1]
        nonlinear = torch.tanh(x @ W1.T)
        gated = torch.sigmoid(x @ W2.T)
        drift = 0.55 * x + 0.35 * (nonlinear * gated)
        layer_bias = 0.03 * torch.sin(torch.full_like(x, float(layer + 1)))
        noise = 0.02 * torch.randn(batch, dim, generator=g)
        seq.append(drift + layer_bias + noise)
    return torch.stack(seq, dim=1)


def flatten_pairs(seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    x = seq[:, :-1, :].reshape(-1, seq.size(-1))
    y = seq[:, 1:, :].reshape(-1, seq.size(-1))
    return x, y


def fit_shared_linear_predictor(x: torch.Tensor, y: torch.Tensor, l2: float = 1e-3) -> torch.Tensor:
    xtx = x.T @ x
    ridge = l2 * torch.eye(xtx.size(0), dtype=x.dtype, device=x.device)
    return torch.linalg.solve(xtx + ridge, x.T @ y)


class TinyRecursivePredictor(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.in_proj = nn.Linear(dim, hidden_dim)
        self.cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, dim)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """
        seq: (batch, layers, dim)
        returns predictions for layers 1..L-1 with shape (batch, layers-1, dim)
        """
        batch, layers, _ = seq.shape
        h = torch.zeros(batch, self.cell.hidden_size, dtype=seq.dtype, device=seq.device)
        outputs = []
        for layer in range(layers - 1):
            inp = torch.tanh(self.in_proj(seq[:, layer, :]))
            h = self.cell(inp, h)
            outputs.append(seq[:, layer, :] + self.out_proj(h))
        return torch.stack(outputs, dim=1)


@dataclass
class EvalMetrics:
    mse: float
    rel_mse: float


def evaluate_copy_last(seq: torch.Tensor) -> EvalMetrics:
    pred = seq[:, :-1, :]
    target = seq[:, 1:, :]
    mse = F.mse_loss(pred, target).item()
    rel = mse / target.pow(2).mean().item()
    return EvalMetrics(mse=mse, rel_mse=rel)


def evaluate_linear(seq_train: torch.Tensor, seq_eval: torch.Tensor) -> EvalMetrics:
    x_train, y_train = flatten_pairs(seq_train)
    W = fit_shared_linear_predictor(x_train, y_train)
    x_eval, y_eval = flatten_pairs(seq_eval)
    pred = x_eval @ W
    mse = F.mse_loss(pred, y_eval).item()
    rel = mse / y_eval.pow(2).mean().item()
    return EvalMetrics(mse=mse, rel_mse=rel)


def train_recursive_predictor(
    seq_train: torch.Tensor,
    seq_eval: torch.Tensor,
    hidden_dim: int,
    epochs: int,
    lr: float,
) -> EvalMetrics:
    model = TinyRecursivePredictor(dim=seq_train.size(-1), hidden_dim=hidden_dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    target_train = seq_train[:, 1:, :]
    target_eval = seq_eval[:, 1:, :]
    best_state = None
    best_eval = float("inf")

    for _ in range(epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        pred = model(seq_train)
        loss = F.mse_loss(pred, target_train)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            eval_loss = F.mse_loss(model(seq_eval), target_eval).item()
        if eval_loss < best_eval:
            best_eval = eval_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_eval = model(seq_eval)
        mse = F.mse_loss(pred_eval, target_eval).item()
        rel = mse / target_eval.pow(2).mean().item()
    return EvalMetrics(mse=mse, rel_mse=rel)


def run_synthetic_experiment(
    train_batch: int,
    eval_batch: int,
    layers: int,
    dim: int,
    hidden_dim: int,
    epochs: int,
    seed: int,
) -> Dict[str, float]:
    seq_train = generate_depth_process(batch=train_batch, layers=layers, dim=dim, seed=seed)
    seq_eval = generate_depth_process(batch=eval_batch, layers=layers, dim=dim, seed=seed + 1)

    copy_metrics = evaluate_copy_last(seq_eval)
    linear_metrics = evaluate_linear(seq_train, seq_eval)
    recursive_metrics = train_recursive_predictor(
        seq_train=seq_train,
        seq_eval=seq_eval,
        hidden_dim=hidden_dim,
        epochs=epochs,
        lr=5e-3,
    )

    return {
        "copy_mse": copy_metrics.mse,
        "linear_mse": linear_metrics.mse,
        "recursive_mse": recursive_metrics.mse,
        "copy_rel_mse": copy_metrics.rel_mse,
        "linear_rel_mse": linear_metrics.rel_mse,
        "recursive_rel_mse": recursive_metrics.rel_mse,
        "linear_gain_vs_copy": 1.0 - (linear_metrics.mse / copy_metrics.mse),
        "recursive_gain_vs_copy": 1.0 - (recursive_metrics.mse / copy_metrics.mse),
        "recursive_gain_vs_linear": 1.0 - (recursive_metrics.mse / linear_metrics.mse),
    }


def run_self_tests() -> None:
    metrics = run_synthetic_experiment(
        train_batch=64,
        eval_batch=32,
        layers=8,
        dim=16,
        hidden_dim=8,
        epochs=80,
        seed=0,
    )
    assert metrics["linear_mse"] < metrics["copy_mse"], metrics
    assert metrics["recursive_mse"] < metrics["copy_mse"], metrics
    assert metrics["recursive_gain_vs_copy"] > 0.10, metrics
    print("[PASS] exp_tric_recursive_predictor self-tests")
    print(json.dumps(metrics, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["self_test", "synthetic"], default="self_test")
    parser.add_argument("--train-batch", type=int, default=256)
    parser.add_argument("--eval-batch", type=int, default=128)
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.mode == "self_test":
        run_self_tests()
        return

    metrics = run_synthetic_experiment(
        train_batch=args.train_batch,
        eval_batch=args.eval_batch,
        layers=args.layers,
        dim=args.dim,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        seed=args.seed,
    )
    payload = {
        "mode": args.mode,
        "train_batch": args.train_batch,
        "eval_batch": args.eval_batch,
        "layers": args.layers,
        "dim": args.dim,
        "hidden_dim": args.hidden_dim,
        "epochs": args.epochs,
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
