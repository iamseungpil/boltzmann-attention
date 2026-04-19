#!/usr/bin/env python3
"""Phase F12 — FacetRot Q+K Coupled SO(2) Rotation (LoRA R1).

Implements the Hybrid FacetRot construction from Thm 6.14 at inference-time on
a facet-separated subspace P_fac (from F8d verb x domain NMI-orthogonal block).
Queries and keys are rotated by content-dependent SO(2) angles indexed by a
soft facet gate (Lemma 6.14.A). RoPE continues to act on the residual
subspace P_res (approximate commutativity regime — full-rotary Qwen2 has no
strictly RoPE-free channel pair; documented in NEW_THEOREM_TEST.md §7).

Math (forward, per (layer, head)):
    g_f(x)    = softmax_f(- ||x - anchor_{f,L,h}||^2 / tau)
    theta*(x) = sum_f g_f(x) * theta[f]
    x_fac     = P_fac x  (coeffs in R)
    x_rot     = reshape_pairs_then_rotate(x_fac, theta*(x))
    x'        = s * (B_fac @ x_rot) + (1-s) * (B_fac @ x_fac) + x_res

LoRA trainable: theta (F, rot_pairs) + log_tau scalars (F12 primary cell uses
shared rotation table; subspace + anchors frozen).

Usage:
  source /home/woori/venvs/seka_env/bin/activate
  CUDA_VISIBLE_DEVICES=0 python3 scripts/new_theorem_test/train_f12_facetrot_qk.py \\
    --subspace-path external/SEKA/seka_projections/f12-qwen25-7b-metatool-facet-subspace/facet_subspace.pt \\
    --out-dir external/SEKA/seka_projections/f12b-qwen25-7b-r32-uniform \\
    --rot-pairs 16 --schedule uniform --steered-layers 18,19,20,21,22,23,24,25,26,27 \\
    --epochs 5 --lr 1e-3 --batch-size 4 --max-train 350
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "scripts" / "ocq"))

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

# Reuse Subtask4 prompt / parsing helpers
from eval_metatool_subtask1 import parse_candidates, resolve_dtype           # noqa: E402
from eval_metatool_subtask4 import (                                         # noqa: E402
    build_fc_prompt,
    extract_tool_name_sequence,
)


# ============================================================================
# Config
# ============================================================================

@dataclass
class F12Config:
    """F12 uniform (default) and F13 FunnelRot shared config."""

    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    subspace_path: Path = Path(
        "external/SEKA/seka_projections/f12-qwen25-7b-metatool-facet-subspace/facet_subspace.pt"
    )
    rot_pairs: int = 16                 # R/2 pairs per head (F12: 16, F13: 2)
    steered_layers: tuple[int, ...] = tuple(range(18, 28))
    gate_tau_init: float = 1.0
    lr: float = 1e-3
    epochs: int = 5
    batch_size: int = 4
    max_train: int = 350
    max_seq_len: int = 1024
    lipschitz_penalty: float = 1e-2
    seed: int = 0

    # F13 FunnelRot schedule --------------------------------------------------
    schedule: str = "uniform"           # "uniform" | "ladapt"
    skip_layer_28: bool = True
    early_end: int = 5
    mid_end: int = 18
    mid_alpha_scale: float = 0.3
    alpha_k: float = 1.0
    beta_q: float = 1.0


def build_kq_schedule(cfg: F12Config, num_layers: int) -> tuple[list[float], list[float]]:
    """Per-layer (s_K, s_Q) strength. See F12Config docstring."""
    s_K = [0.0] * num_layers
    s_Q = [0.0] * num_layers
    steered = set(cfg.steered_layers)

    if cfg.schedule == "uniform":
        for ell in range(num_layers):
            if ell in steered:
                s_K[ell] = 1.0
                s_Q[ell] = 1.0
    elif cfg.schedule == "ladapt":
        for ell in range(num_layers):
            if ell <= cfg.early_end:
                s_K[ell] = cfg.alpha_k
                s_Q[ell] = 0.0
            elif ell <= cfg.mid_end:
                s_K[ell] = cfg.alpha_k * cfg.mid_alpha_scale
                s_Q[ell] = cfg.beta_q
            else:
                s_K[ell] = 0.0
                s_Q[ell] = cfg.beta_q
    else:
        raise ValueError(f"unknown schedule: {cfg.schedule!r}")

    if cfg.skip_layer_28 and 28 < num_layers:
        s_K[28] = 0.0
        s_Q[28] = 0.0

    return s_K, s_Q


# ============================================================================
# FacetRotSO2 — shared rotation table theta (F, rot_pairs)
# ============================================================================

class FacetRotSO2(nn.Module):
    """Per-facet SO(2)^{rot_pairs} rotations (shared across layers/heads).

    Parameters: theta (F, rot_pairs). At theta=0 the rotation is identity,
    matching the baseline operating point — training moves theta away.

    Soft-angle interpolation (Lemma 6.14.A Option A, Lipschitz linear-index).
    """

    def __init__(self, n_facets: int, rot_pairs: int) -> None:
        super().__init__()
        self.n_facets = n_facets
        self.rot_pairs = rot_pairs
        self.theta = nn.Parameter(torch.zeros(n_facets, rot_pairs))

    def soft_theta(self, gate: torch.Tensor) -> torch.Tensor:
        """gate: (..., F) -> (..., rot_pairs)."""
        return torch.einsum("...f,fi->...i", gate, self.theta)

    def apply_rotation(self, pairs: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """pairs: (..., rot_pairs, 2); gate: (..., F) -> rotated pairs (..., rot_pairs, 2)."""
        theta = self.soft_theta(gate).unsqueeze(-1)     # (..., rot_pairs, 1)
        c = torch.cos(theta)
        s = torch.sin(theta)
        x0 = pairs[..., 0:1]
        x1 = pairs[..., 1:2]
        out0 = c * x0 - s * x1
        out1 = s * x0 + c * x1
        return torch.cat([out0, out1], dim=-1)

    def lipschitz_bound(self) -> torch.Tensor:
        """Proxy: sum_{f,i} theta[f,i]^2 — encourages small angles."""
        return (self.theta ** 2).sum()


# ============================================================================
# FacetSubspace — per (layer, role) B_fac (H, d, R) + anchors (F, H, d)
# ============================================================================

class FacetSubspace(nn.Module):
    """Per-(layer, role) facet basis and anchor centroids.

    Buffers (frozen):
      B_fac   : (H, d, R)  — orthonormal columns, H matches head count for the
                              role (n_kv for K, n_q for Q after GQA repeat)
      anchors : (F, H, d)  — per-facet centroid for gate energy

    Trainable:
      log_tau : scalar temperature for the gate softmax
    """

    def __init__(self, B_fac: torch.Tensor, anchors: torch.Tensor, tau_init: float = 1.0) -> None:
        super().__init__()
        assert B_fac.dim() == 3 and anchors.dim() == 3, \
            f"B_fac must be (H,d,R); anchors must be (F,H,d); got {tuple(B_fac.shape)}, {tuple(anchors.shape)}"
        H_b, d_b, _ = B_fac.shape
        F_a, H_a, d_a = anchors.shape
        assert (H_b, d_b) == (H_a, d_a), "head/dim mismatch B_fac vs anchors"
        self.register_buffer("B_fac", B_fac.contiguous())
        self.register_buffer("anchors", anchors.contiguous())
        self.log_tau = nn.Parameter(torch.tensor(math.log(tau_init)))

    @property
    def tau(self) -> torch.Tensor:
        return torch.exp(self.log_tau).clamp(min=0.1)

    def split(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, T, H, d) -> (fac_coeffs (B,T,H,R), x_res (B,T,H,d))."""
        coeffs = torch.einsum("bthd,hdr->bthr", x, self.B_fac)
        fac_full = torch.einsum("bthr,hdr->bthd", coeffs, self.B_fac)
        return coeffs, x - fac_full

    def gate(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B,T,H,d) -> gate (B,T,H,F).

        Uses ||x - a||^2 = ||x||^2 - 2<x,a> + ||a||^2 to avoid the O(B*T*H*F*d)
        materialization of an explicit difference tensor, which OOMs on n_q=28.
        """
        x_sq = x.pow(2).sum(-1, keepdim=True)                    # (B,T,H,1)
        a_sq = self.anchors.pow(2).sum(-1)                       # (F, H)
        inner = torch.einsum("bthd,fhd->bthf", x, self.anchors)  # (B,T,H,F)
        sq = x_sq + a_sq.transpose(0, 1).unsqueeze(0).unsqueeze(0) - 2.0 * inner
        logits = -sq / self.tau
        return torch.softmax(logits, dim=-1)


# ============================================================================
# Hook — apply rotation on facet block after q_proj or k_proj
# ============================================================================

class FacetRotHook:
    """Forward hook on q_proj / k_proj output. Rotates facet-block of K/Q.

    Hook input output shape: (B, T, H*d) where H = n_kv for K, n_q for Q.
    At strength=0 returns identity. Strength is baked into the hook (per-layer
    schedule); rotation + subspace modules are shared per-layer at construction.
    """

    def __init__(
        self,
        subspace: FacetSubspace,
        rotation: FacetRotSO2,
        n_heads: int,
        head_dim: int,
        strength: float,
    ) -> None:
        self.subspace = subspace
        self.rotation = rotation
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.strength = strength
        assert rotation.rot_pairs * 2 == subspace.B_fac.shape[-1], \
            f"rot_pairs*2 ({rotation.rot_pairs*2}) != R ({subspace.B_fac.shape[-1]})"

    def __call__(self, module, inputs, output):
        if self.strength == 0.0:
            return output
        if output.dim() != 3:
            return output
        B, T, D = output.shape
        if D != self.n_heads * self.head_dim:
            return output
        x = output.view(B, T, self.n_heads, self.head_dim)
        orig_dtype = x.dtype
        x_f = x.to(torch.float32)

        fac_coeffs, x_res = self.subspace.split(x_f)          # (B,T,H,R), (B,T,H,d)
        gate = self.subspace.gate(x_f)                        # (B,T,H,F)
        R = fac_coeffs.shape[-1]
        pairs = fac_coeffs.view(B, T, self.n_heads, R // 2, 2)
        rotated = self.rotation.apply_rotation(pairs, gate)
        rotated_flat = rotated.view(B, T, self.n_heads, R)

        fac_rot_full = torch.einsum(
            "bthr,hdr->bthd", rotated_flat, self.subspace.B_fac
        )
        fac_id_full = torch.einsum(
            "bthr,hdr->bthd", fac_coeffs, self.subspace.B_fac
        )
        s = self.strength
        new_x = s * fac_rot_full + (1.0 - s) * fac_id_full + x_res
        return new_x.to(orig_dtype).reshape(B, T, D)


# ============================================================================
# Hook registration — per-layer (k_proj + q_proj)
# ============================================================================

def install_hooks(
    model,
    cfg: F12Config,
    k_subspaces: nn.ModuleDict,
    q_subspaces: nn.ModuleDict,
    rotation: FacetRotSO2,
    n_kv: int,
    n_q: int,
    head_dim: int,
    s_K_schedule: List[float],
    s_Q_schedule: List[float],
) -> List:
    handles: List = []
    for layer_idx, layer in enumerate(model.model.layers):
        key = str(layer_idx)
        sK = s_K_schedule[layer_idx]
        sQ = s_Q_schedule[layer_idx]
        if sK > 0.0 and key in k_subspaces:
            hook = FacetRotHook(k_subspaces[key], rotation, n_kv, head_dim, strength=sK)
            handles.append(layer.self_attn.k_proj.register_forward_hook(hook))
        if sQ > 0.0 and key in q_subspaces:
            hook = FacetRotHook(q_subspaces[key], rotation, n_q, head_dim, strength=sQ)
            handles.append(layer.self_attn.q_proj.register_forward_hook(hook))
    return handles


# ============================================================================
# MetaTool Subtask4 train loader
# ============================================================================

def _gt_completion(gt_tools: List[str]) -> str:
    parts = [
        '<tool_call>{"name": "' + name + '", "arguments": {}}</tool_call>'
        for name in gt_tools
    ]
    return "\n".join(parts)


def build_training_examples(
    dataset_path: Path,
    tokenizer,
    max_samples: int,
    max_seq_len: int,
) -> List[dict]:
    data = json.loads(dataset_path.read_text())
    if max_samples > 0:
        data = data[:max_samples]
    examples: List[dict] = []
    skipped = 0
    for i, entry in enumerate(data):
        action_prompt = entry["action_prompt"]
        gt = entry["tool"] if isinstance(entry["tool"], list) else [entry["tool"]]
        cands = parse_candidates(action_prompt)
        fc_prompt = build_fc_prompt(tokenizer, action_prompt, cands)
        completion = _gt_completion(gt)
        if tokenizer.eos_token:
            completion_full = completion + tokenizer.eos_token
        else:
            completion_full = completion

        prompt_ids = tokenizer(fc_prompt, add_special_tokens=False)["input_ids"]
        comp_ids = tokenizer(completion_full, add_special_tokens=False)["input_ids"]
        full_ids = prompt_ids + comp_ids
        if len(full_ids) > max_seq_len:
            skipped += 1
            continue
        labels = [-100] * len(prompt_ids) + list(comp_ids)
        examples.append({
            "index": i,
            "input_ids": full_ids,
            "labels": labels,
            "gt": gt,
        })
    print(f"[data] built {len(examples)} training examples ({skipped} skipped > max_seq_len)")
    return examples


def collate(batch: List[dict], pad_id: int) -> dict:
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    attn_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, b in enumerate(batch):
        L = len(b["input_ids"])
        input_ids[i, :L] = torch.tensor(b["input_ids"], dtype=torch.long)
        labels[i, :L] = torch.tensor(b["labels"], dtype=torch.long)
        attn_mask[i, :L] = 1
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attn_mask}


# ============================================================================
# train_f12
# ============================================================================

def _repeat_kv_to_q(B_fac_K: torch.Tensor, anchors_K: torch.Tensor, n_q: int, n_kv: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """GQA expansion: each K head is shared by n_q//n_kv Q heads."""
    assert n_q % n_kv == 0, f"n_q ({n_q}) must be divisible by n_kv ({n_kv})"
    rep = n_q // n_kv
    B_fac_Q = B_fac_K.repeat_interleave(rep, dim=0)                 # (n_q, d, R)
    anchors_Q = anchors_K.repeat_interleave(rep, dim=1)             # (F, n_q, d)
    return B_fac_Q, anchors_Q


def train_f12(cfg: F12Config, out_dir: Path) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(cfg.seed)

    # --- 1. Load frozen base model ---
    print(f"[load] {cfg.model_name}", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    ).to("cuda")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"[load] done in {time.time()-t0:.1f}s", flush=True)

    mcfg = model.config
    n_q = mcfg.num_attention_heads
    n_kv = mcfg.num_key_value_heads
    head_dim = getattr(mcfg, "head_dim", None) or (mcfg.hidden_size // n_q)
    num_layers = len(model.model.layers)
    print(f"[model] L={num_layers} n_q={n_q} n_kv={n_kv} d={head_dim}", flush=True)

    # --- 2. Load facet subspace ---
    sub_path = cfg.subspace_path if Path(cfg.subspace_path).is_absolute() else REPO / cfg.subspace_path
    assert Path(sub_path).exists(), f"facet subspace not found: {sub_path}"
    print(f"[subspace] {sub_path}", flush=True)
    blob = torch.load(sub_path, map_location="cpu", weights_only=False)
    B_fac_all: torch.Tensor = blob["B_fac"].float()      # (L, n_kv, d, R)
    anchors_all: torch.Tensor = blob["anchors"].float()  # (F, L, n_kv, d)
    n_facets = int(blob["n_facets"])
    R = int(blob["rank"])
    assert R == 2 * cfg.rot_pairs, f"rot_pairs={cfg.rot_pairs} mismatches R={R}"
    assert B_fac_all.shape == (num_layers, n_kv, head_dim, R)
    assert anchors_all.shape == (n_facets, num_layers, n_kv, head_dim)
    print(f"[subspace] F={n_facets} R={R}", flush=True)

    # --- 3. Compute (s_K, s_Q) schedule ---
    s_K_schedule, s_Q_schedule = build_kq_schedule(cfg, num_layers)
    print(
        f"[schedule] mode={cfg.schedule!r}, skip_L28={cfg.skip_layer_28}\n"
        f"  s_K = {[round(x, 2) for x in s_K_schedule]}\n"
        f"  s_Q = {[round(x, 2) for x in s_Q_schedule]}",
        flush=True,
    )

    # --- 4. Build per-layer subspace modules (K n_kv heads, Q n_q heads via repeat) ---
    device = next(model.parameters()).device
    rotation = FacetRotSO2(n_facets=n_facets, rot_pairs=cfg.rot_pairs).to(device=device, dtype=torch.float32)

    k_subspaces = nn.ModuleDict()
    q_subspaces = nn.ModuleDict()
    for ell in range(num_layers):
        if s_K_schedule[ell] == 0.0 and s_Q_schedule[ell] == 0.0:
            continue
        B_K = B_fac_all[ell]                # (n_kv, d, R)
        anchors_K = anchors_all[:, ell]     # (F, n_kv, d)
        if s_K_schedule[ell] > 0.0:
            k_subspaces[str(ell)] = FacetSubspace(B_K.clone(), anchors_K.clone(), cfg.gate_tau_init).to(device=device, dtype=torch.float32)
        if s_Q_schedule[ell] > 0.0:
            B_Q, anchors_Q = _repeat_kv_to_q(B_K, anchors_K, n_q, n_kv)
            q_subspaces[str(ell)] = FacetSubspace(B_Q.clone(), anchors_Q.clone(), cfg.gate_tau_init).to(device=device, dtype=torch.float32)

    n_k_hooked = len(k_subspaces)
    n_q_hooked = len(q_subspaces)
    print(f"[subspace modules] K-layers={n_k_hooked} Q-layers={n_q_hooked}", flush=True)

    # --- 5. Training loader ---
    dataset_path = Path("/tmp/MetaTool/dataset/tmp_dataset/Task2-Subtask4.json")
    examples = build_training_examples(
        dataset_path, tokenizer, cfg.max_train, cfg.max_seq_len
    )

    # --- 6. Register hooks ---
    handles = install_hooks(
        model, cfg, k_subspaces, q_subspaces, rotation,
        n_kv, n_q, head_dim, s_K_schedule, s_Q_schedule,
    )
    print(f"[hooks] installed {len(handles)} forward hooks", flush=True)

    # --- 7. Optimizer on trainable params only ---
    trainable: List[nn.Parameter] = [rotation.theta]
    for sub in list(k_subspaces.values()) + list(q_subspaces.values()):
        trainable.append(sub.log_tau)
    n_params = sum(p.numel() for p in trainable)
    print(f"[opt] AdamW on {len(trainable)} tensors, {n_params} scalars", flush=True)
    optim = torch.optim.AdamW(trainable, lr=cfg.lr)

    # --- 8. Training loop ---
    log_history: List[dict] = []
    t_train_start = time.time()
    for epoch in range(cfg.epochs):
        # shuffle each epoch
        perm = torch.randperm(len(examples), generator=torch.Generator().manual_seed(cfg.seed + epoch)).tolist()
        loss_sum_ce = 0.0
        loss_sum_lip = 0.0
        n_steps = 0
        for start in range(0, len(examples), cfg.batch_size):
            batch_ids = perm[start:start + cfg.batch_size]
            batch = [examples[i] for i in batch_ids]
            feed = collate(batch, pad_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
            input_ids = feed["input_ids"].to(device)
            labels = feed["labels"].to(device)
            attn_mask = feed["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
            loss_ce = outputs.loss
            loss_lip = cfg.lipschitz_penalty * rotation.lipschitz_bound()
            loss = loss_ce + loss_lip

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optim.step()

            loss_sum_ce += float(loss_ce.item())
            loss_sum_lip += float(loss_lip.item())
            n_steps += 1
        avg_ce = loss_sum_ce / max(n_steps, 1)
        avg_lip = loss_sum_lip / max(n_steps, 1)
        theta_norm = float(rotation.theta.detach().norm().item())
        theta_max_abs = float(rotation.theta.detach().abs().max().item())
        log_history.append({
            "epoch": epoch,
            "avg_ce_loss": avg_ce,
            "avg_lipschitz_penalty": avg_lip,
            "theta_fro": theta_norm,
            "theta_max_abs": theta_max_abs,
            "steps": n_steps,
        })
        print(
            f"[epoch {epoch}] ce={avg_ce:.4f} lip={avg_lip:.4f} "
            f"||theta||_F={theta_norm:.4f} |theta|_max={theta_max_abs:.4f} steps={n_steps}",
            flush=True,
        )

    train_runtime = time.time() - t_train_start

    for h in handles:
        h.remove()

    # --- 9. Save checkpoint ---
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "f12_checkpoint.pt"
    torch.save({
        "rotation_theta": rotation.theta.detach().cpu(),
        "k_log_tau": {k: float(v.log_tau.detach().cpu()) for k, v in k_subspaces.items()},
        "q_log_tau": {k: float(v.log_tau.detach().cpu()) for k, v in q_subspaces.items()},
        "config": {
            "model_name": cfg.model_name,
            "subspace_path": str(cfg.subspace_path),
            "rot_pairs": cfg.rot_pairs,
            "steered_layers": list(cfg.steered_layers),
            "schedule": cfg.schedule,
            "skip_layer_28": cfg.skip_layer_28,
            "early_end": cfg.early_end,
            "mid_end": cfg.mid_end,
            "mid_alpha_scale": cfg.mid_alpha_scale,
            "alpha_k": cfg.alpha_k,
            "beta_q": cfg.beta_q,
            "lipschitz_penalty": cfg.lipschitz_penalty,
            "n_facets": n_facets,
            "R": R,
            "n_layers": num_layers,
            "n_kv": n_kv,
            "n_q": n_q,
            "head_dim": head_dim,
            "epochs": cfg.epochs,
            "lr": cfg.lr,
            "batch_size": cfg.batch_size,
            "max_train": cfg.max_train,
            "seed": cfg.seed,
        },
        "s_K_schedule": s_K_schedule,
        "s_Q_schedule": s_Q_schedule,
        "log_history": log_history,
        "train_runtime_s": train_runtime,
    }, ckpt_path)
    print(f"[saved] {ckpt_path}", flush=True)

    meta = {
        "out_dir": str(out_dir),
        "train_runtime_s": train_runtime,
        "n_train_examples": len(examples),
        "final_epoch": log_history[-1] if log_history else None,
        "log_history": log_history,
    }
    (out_dir / "train_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


# ============================================================================
# CLI
# ============================================================================

def main() -> None:
    p = argparse.ArgumentParser(description="F12 FacetRot-QK LoRA R1 training.")
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--subspace-path", type=Path,
                   default=Path("external/SEKA/seka_projections/f12-qwen25-7b-metatool-facet-subspace/facet_subspace.pt"))
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--rot-pairs", type=int, default=16,
                   help="R/2 (F12 default 16 -> R=32; F13 2 -> R=4)")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-train", type=int, default=350)
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--lipschitz-penalty", type=float, default=1e-2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steered-layers", type=str, default="18,19,20,21,22,23,24,25,26,27")
    p.add_argument("--schedule", choices=["uniform", "ladapt"], default="uniform")
    p.add_argument("--skip-layer-28",
                   type=lambda s: s.lower() in {"1", "true", "yes", "y"},
                   default=True)
    p.add_argument("--early-end", type=int, default=5)
    p.add_argument("--mid-end", type=int, default=18)
    p.add_argument("--mid-alpha-scale", type=float, default=0.3)
    p.add_argument("--alpha-k", type=float, default=1.0)
    p.add_argument("--beta-q", type=float, default=1.0)
    args = p.parse_args()

    steered_arg = args.steered_layers.strip().lower()
    if steered_arg == "all":
        steered = tuple(range(0, 28))
    else:
        steered = tuple(int(x) for x in args.steered_layers.split(","))

    cfg = F12Config(
        model_name=args.model,
        subspace_path=args.subspace_path,
        rot_pairs=args.rot_pairs,
        steered_layers=steered,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_train=args.max_train,
        max_seq_len=args.max_seq_len,
        lipschitz_penalty=args.lipschitz_penalty,
        seed=args.seed,
        schedule=args.schedule,
        skip_layer_28=args.skip_layer_28,
        early_end=args.early_end,
        mid_end=args.mid_end,
        mid_alpha_scale=args.mid_alpha_scale,
        alpha_k=args.alpha_k,
        beta_q=args.beta_q,
    )
    train_f12(cfg, args.out_dir)


if __name__ == "__main__":
    main()
