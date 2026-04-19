"""Phase F12 — FacetRot Q+K Coupled Rotation (LoRA R1) skeleton.

Implements Hybrid FacetRot (Thm 6.14) at inference-time on a facet-separated subspace
P_fac (F8d verb x domain NMI-orthogonal block). Queries and keys are rotated by
content-dependent SO(2) angles indexed by a soft facet gate (Lemma 6.14.A). RoPE
continues to act on the residual subspace P_res.

Math (forward):
    pi_soft(x) = sum_f f * g_f(x) / sum_f g_f(x)
    g_f(x)     = exp(||P_f x||^2 / tau)
    R_{pi(x)}  = linear interpolation of {R_f}_{f=1..F} via pi(x)
    q_tilde    = R_{pi(q)} P_fac q + P_res q
    k_tilde    = R_{pi(k)} P_fac k + P_res k

LoRA trainable:
    theta[f, i] for f in 1..F facets, i in 1..R/2 SO(2) pairs  (F * R/2 scalars per head)
    tau (scalar gate temperature, optional)

Frozen:
    Base model (Qwen2.5-7B-Instruct) weights
    P_fac / P_res projectors (derived from F8d NMI verb x domain block pre-training)
    Facet assignment function (ontology labels)

Status: SKELETON — placeholder hooks, no GPU run yet. See NEW_THEOREM_TEST.md
Phase F12 for pre-registered decision tree.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------- Config --------

@dataclass
class F12Config:
    """Shared config for F12 uniform (default) and F13 FunnelRot variants.

    F12 uniform: schedule="uniform", steered_layers=[18..27], skip_layer_28=True (naturally).
    F13 FunnelRot: schedule="ladapt", steered_layers=[0..27], skip_layer_28=True (explicit),
                    early_end=5, mid_end=18, mid_alpha_scale=0.3, rot_pairs=2 (R=4 low-rank).
    """

    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    bont_dir: Path = Path("external/SEKA/seka_projections/f12-qwen25-7b-metatool-facet-subspace")
    n_facets: int = 16
    rot_pairs: int = 16  # R/2 — number of SO(2) pairs per head. F12: 16 (R=32). F13: 2 (R=4).
    steered_layers: tuple[int, ...] = tuple(range(18, 28))  # F12 default; F13 uses range(0, 28)
    gate_tau_init: float = 1.0
    lr: float = 1e-3
    epochs: int = 5
    batch_size: int = 4
    lipschitz_penalty: float = 1e-2  # Lemma 6.14.A regularity

    # F13 FunnelRot extension: ladapt schedule + explicit L28 skip ------------
    schedule: str = "uniform"          # "uniform" (F12) | "ladapt" (F13 FunnelRot)
    skip_layer_28: bool = True         # F13.Funnel: preserve natural amplifier
    early_end: int = 5                 # ladapt stage 1 upper bound (inclusive)
    mid_end: int = 18                  # ladapt stage 2 upper bound (inclusive)
    mid_alpha_scale: float = 0.3       # weak K in stage 2 (K scaling; Q stays at 1.0)
    alpha_k: float = 1.0               # global K strength multiplier (stage 1 base)
    beta_q: float = 1.0                # global Q strength multiplier (stages 2-3 base)


def build_kq_schedule(cfg: F12Config, num_layers: int) -> tuple[list[float], list[float]]:
    """Return per-layer (s_K, s_Q) schedule scalars.

    `schedule="uniform"`: s_K=s_Q=1.0 for layers in cfg.steered_layers, 0 elsewhere.
                           skip_layer_28=True forces s=0 at layer 28 regardless.
    `schedule="ladapt"` : three-stage ladapt schedule per F13 spec.
                           skip_layer_28=True forces s=0 at layer 28.

    Stage definitions (ladapt):
      Stage 1: [0, early_end]         -> (alpha_k, 0)
      Stage 2: (early_end, mid_end]   -> (alpha_k * mid_alpha_scale, beta_q)
      Stage 3: (mid_end, num_layers)  -> (0, beta_q)
      Layer 28 (if skip_layer_28)     -> (0, 0) hard-overridden
    """
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
        raise ValueError(f"unknown schedule: {cfg.schedule!r} (expected 'uniform' or 'ladapt')")

    if cfg.skip_layer_28 and 28 < num_layers:
        # F13.Funnel: L28 is the natural amplifier; do not perturb it.
        s_K[28] = 0.0
        s_Q[28] = 0.0

    return s_K, s_Q


# -------- SO(2) rotation module --------

class FacetRotSO2(nn.Module):
    """Per-facet SO(2)^(R/2) rotations.

    Parameters: theta of shape (n_facets, rot_pairs). Each pair i defines
    a 2x2 rotation R_{f,i} = [[cos theta, -sin theta], [sin theta, cos theta]].

    Soft rotation via weighted-angle interpolation (Lemma 6.14.A Option A):
        theta_soft[i] = sum_f g_f * theta[f, i] / sum_f g_f
        R_soft[i] = [[cos theta_soft[i], -sin theta_soft[i]], ...]

    This is Option A (linear-index interpolation, Lipschitz, cheap). Option C
    (Frechet / Lie-algebra mean, canonical-in-SO(R)) is TODO for ablation R5.
    """

    def __init__(self, n_facets: int, rot_pairs: int) -> None:
        super().__init__()
        self.n_facets = n_facets
        self.rot_pairs = rot_pairs
        # Initialize angles at zero — identity rotation, so training starts from
        # a structurally-equivalent-to-baseline operating point.
        self.theta = nn.Parameter(torch.zeros(n_facets, rot_pairs))

    def soft_theta(self, gate: torch.Tensor) -> torch.Tensor:
        """gate: (..., n_facets) softmax-normalized facet weights.
        Returns: (..., rot_pairs) interpolated angles.
        """
        # (..., rot_pairs)
        return torch.einsum("...f,fi->...i", gate, self.theta)

    def apply(self, x_block: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """Apply soft rotation to facet-block activations.

        x_block: (..., rot_pairs, 2) — reshape of P_fac x into SO(2) pairs
        gate:    (..., n_facets)     — facet softmax
        returns: (..., rot_pairs, 2) rotated
        """
        theta = self.soft_theta(gate)               # (..., rot_pairs)
        cos_t = torch.cos(theta).unsqueeze(-1)      # (..., rot_pairs, 1)
        sin_t = torch.sin(theta).unsqueeze(-1)
        x0 = x_block[..., 0:1]
        x1 = x_block[..., 1:2]
        out_0 = cos_t * x0 - sin_t * x1
        out_1 = sin_t * x0 + cos_t * x1
        return torch.cat([out_0, out_1], dim=-1)

    def lipschitz_bound(self, tau_min: float = 0.1) -> torch.Tensor:
        """Lemma 6.14.A: L_fac = 2 pi F L_g / min sum g_f.
        Here we only track the numerator component (2 pi F * theta-variation)
        which is the trainable signal; full bound is computed with gate data.
        """
        # Proxy: sum of squared angles per pair — discourages angle explosion
        return (self.theta ** 2).sum()


# -------- Facet subspace & gate --------

class FacetSubspace(nn.Module):
    """Holds P_fac, P_res projectors and facet anchor centroids for gate computation.

    Loaded from `build_f12_facet_subspace.py` output (pre-trained, frozen):
      - B_fac: (d_head, R) orthonormal basis of verb x domain NMI-orthogonal block
      - anchors: (n_facets, d_head) mean K per facet (for gate g_f)
    """

    def __init__(self, B_fac: torch.Tensor, anchors: torch.Tensor, tau: float = 1.0) -> None:
        super().__init__()
        # Frozen
        self.register_buffer("B_fac", B_fac)                # (d, R)
        self.register_buffer("anchors", anchors)            # (F, d)
        # Trainable (optional — can be held fixed)
        self.log_tau = nn.Parameter(torch.tensor(math.log(tau)))

    @property
    def tau(self) -> torch.Tensor:
        return torch.exp(self.log_tau).clamp(min=0.1)

    def split(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (..., d). Returns (x_fac coeffs (..., R), x_res (..., d))."""
        fac_coeffs = torch.einsum("...d,dr->...r", x, self.B_fac)
        x_fac_full = torch.einsum("...r,dr->...d", fac_coeffs, self.B_fac)
        x_res = x - x_fac_full
        return fac_coeffs, x_res

    def gate(self, x: torch.Tensor) -> torch.Tensor:
        """Soft facet gate g_f(x) = exp(-||x - anchor_f||^2 / tau) normalized over f.
        x: (..., d)  →  returns (..., n_facets)
        (Equivalent to the exp-energy form for centered anchors; this cosine-distance
        variant is numerically stabler.)
        """
        # (..., F) squared distances
        diff = x.unsqueeze(-2) - self.anchors                  # (..., F, d)
        sq = (diff ** 2).sum(-1)                               # (..., F)
        logits = -sq / self.tau
        return torch.softmax(logits, dim=-1)


# -------- Hook layer wrapping q_proj / k_proj --------

class FacetRotQKHook:
    """Per-layer forward-post-hook that modifies Q and K after q_proj / k_proj.

    Engineering note: HF Qwen2 uses grouped-query attention (4 KV heads), so we
    register the hook AFTER the projection and BEFORE RoPE + GQA expansion. This
    guarantees P_res is disjoint from the RoPE rotation channels (Thm 6.14
    Hybrid commuting-subgroup requirement).

    F13 addition: a `strength` scalar scales the facet-block rotation, so a
    schedule can gate Q vs K differently per layer (ladapt 3-stage). When
    strength=0 the facet block is passed through identically (no perturbation).

    TODO(F12): verify RoPE channel pair indices to construct P_res correctly.
    Qwen2 RoPE pairs are (0,1), (2,3), ..., (d-2, d-1); P_fac must select channels
    from the remaining subspace.
    """

    def __init__(
        self,
        subspace: FacetSubspace,
        rotation: FacetRotSO2,
        steered_heads: Iterable[int] | None = None,
        strength: float = 1.0,
    ) -> None:
        self.subspace = subspace
        self.rotation = rotation
        self.steered_heads = set(steered_heads) if steered_heads is not None else None
        self.strength = strength

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, n_heads, d_head) — rotate facet block (scaled by strength),
        keep residual. At strength=0 returns x unchanged."""
        if self.strength == 0.0:
            return x
        fac_coeffs, x_res = self.subspace.split(x)       # (..., R), (..., d)
        gate = self.subspace.gate(x)                     # (..., F)
        B, T, H, R = fac_coeffs.shape
        assert R % 2 == 0, "R (facet subspace dim) must be even for SO(2) pairing"
        fac_pairs = fac_coeffs.view(B, T, H, R // 2, 2)
        fac_rot = self.rotation.apply(fac_pairs, gate.unsqueeze(-2))
        fac_rot_flat = fac_rot.view(B, T, H, R)
        fac_rotated_full = torch.einsum("bthr,dr->bthd", fac_rot_flat, self.subspace.B_fac)
        # Blend: (1 - s) * identity on facet block + s * rotated block. At s=1
        # this equals F12's unconditional rotation; at s=0 it is identity.
        # Recover the un-rotated facet embedding from coeffs for the (1-s) term:
        fac_identity = torch.einsum("bthr,dr->bthd", fac_coeffs, self.subspace.B_fac)
        return self.strength * fac_rotated_full + (1.0 - self.strength) * fac_identity + x_res


# -------- LoRA-style training loop skeleton --------

def train_f12(cfg: F12Config, out_dir: Path) -> None:
    """Skeleton LoRA-style training.

    TODO(F12):
      - Load Qwen2.5-7B-Instruct via transformers (torch_dtype=torch.bfloat16)
      - Freeze all base params; only {FacetRotSO2.theta, FacetSubspace.log_tau} trainable
      - Register hooks on steered layers' q_proj and k_proj outputs
      - Loader: MetaTool Subtask4 train split, tokenize in chat template
      - Forward: compute CE loss on GT tool name sequence
      - Add Lipschitz penalty: cfg.lipschitz_penalty * rotation.lipschitz_bound()
      - Optimizer: AdamW on trainable params only
      - Save {rotation.state_dict, subspace.state_dict} + metadata each epoch
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # --- 1. Load frozen base model ---
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, torch_dtype=torch.bfloat16)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    # --- 2. Load pre-built facet subspace from F8d / F11 pipeline ---
    bont_pt = cfg.bont_dir / "facet_subspace.pt"
    assert bont_pt.exists(), (
        f"Run build_f12_facet_subspace.py first (missing {bont_pt}). "
        "It must produce keys: B_fac (d_head, R), anchors (F, d_head), facet_labels"
    )
    blob = torch.load(bont_pt, map_location="cpu")
    B_fac: torch.Tensor = blob["B_fac"]
    anchors: torch.Tensor = blob["anchors"]
    d_head = B_fac.shape[0]
    R = B_fac.shape[1]
    assert R == 2 * cfg.rot_pairs, f"rot_pairs={cfg.rot_pairs} mismatches R={R}"

    subspace = FacetSubspace(B_fac, anchors, tau=cfg.gate_tau_init).to(model.device)
    rotation = FacetRotSO2(cfg.n_facets, cfg.rot_pairs).to(model.device)

    # --- 3. Compute per-layer (s_K, s_Q) schedule ---
    #   F12 uniform  : s_K = s_Q = 1.0 on steered_layers, 0 elsewhere
    #   F13 FunnelRot: ladapt 3-stage (early K / mid K+Q / late Q) + explicit L28 skip
    num_layers = len(model.model.layers)
    s_K_schedule, s_Q_schedule = build_kq_schedule(cfg, num_layers)

    print(f"[f12/f13] schedule={cfg.schedule!r}, skip_layer_28={cfg.skip_layer_28}, "
          f"rot_pairs={cfg.rot_pairs} (R={2*cfg.rot_pairs}), n_facets={cfg.n_facets}")
    print(f"[f12/f13] s_K per layer: {[round(x, 3) for x in s_K_schedule]}")
    print(f"[f12/f13] s_Q per layer: {[round(x, 3) for x in s_Q_schedule]}")

    # --- 4. Register per-layer hooks with per-layer strength ---
    hook_handles = []
    for layer_idx in range(num_layers):
        s_K = s_K_schedule[layer_idx]
        s_Q = s_Q_schedule[layer_idx]
        if s_K == 0.0 and s_Q == 0.0:
            continue  # layer not steered (includes skip_layer_28)
        layer = model.model.layers[layer_idx]  # Qwen2DecoderLayer
        # K-proj hook (active when s_K > 0)
        if s_K > 0.0:
            k_hook = FacetRotQKHook(subspace, rotation, strength=s_K)
            # TODO(F12): register forward_post_hook on layer.self_attn.k_proj
            #   Reshape (B, T, n_kv * d_head) → (B, T, n_kv, d_head) before transform,
            #   flatten back after. Handle GQA: apply on n_kv heads before expansion.
            # hook_handles.append(layer.self_attn.k_proj.register_forward_hook(...))
        # Q-proj hook (active when s_Q > 0)
        if s_Q > 0.0:
            q_hook = FacetRotQKHook(subspace, rotation, strength=s_Q)
            # TODO(F12): register forward_post_hook on layer.self_attn.q_proj
            # hook_handles.append(layer.self_attn.q_proj.register_forward_hook(...))

    # --- 4. Loader ---
    # TODO(F12): build MetaTool Subtask4 train loader; tokenize with chat template;
    # label = GT tool name token ids

    # --- 5. Optimizer on trainable params only ---
    trainable = [p for p in rotation.parameters()] + [subspace.log_tau]
    optim = torch.optim.AdamW(trainable, lr=cfg.lr)

    # --- 6. Training loop ---
    for epoch in range(cfg.epochs):
        # TODO(F12): iterate batches, forward, CE loss on GT tool seq,
        # + cfg.lipschitz_penalty * rotation.lipschitz_bound()
        # optim.step(); optim.zero_grad()
        pass

    # --- 7. Save checkpoint ---
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "rotation": rotation.state_dict(),
            "subspace": subspace.state_dict(),
            "config": cfg.__dict__,
        },
        out_dir / "f12_checkpoint.pt",
    )
    (out_dir / "meta.json").write_text(json.dumps({"cfg": {k: str(v) for k, v in cfg.__dict__.items()}}, indent=2))

    for h in hook_handles:
        h.remove()


# -------- CLI --------

def main() -> None:
    p = argparse.ArgumentParser(description="F12 FacetRot-QK LoRA R1 (skeleton)")
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--bont-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--n-facets", type=int, default=16)
    p.add_argument("--rot-pairs", type=int, default=16, help="R/2 — SO(2) pairs per head")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument(
        "--steered-layers",
        type=str,
        default="18,19,20,21,22,23,24,25,26,27",
        help="comma-separated layer indices (Thm 6.14 R1: mid-late)",
    )
    args = p.parse_args()

    cfg = F12Config(
        model_name=args.model,
        bont_dir=args.bont_dir,
        n_facets=args.n_facets,
        rot_pairs=args.rot_pairs,
        steered_layers=tuple(int(x) for x in args.steered_layers.split(",")),
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    train_f12(cfg, args.out_dir)


if __name__ == "__main__":
    main()
