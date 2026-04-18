"""Canonical AdaSEKA engine (ports external/SEKA/src/model/adaptive_seka_llm.py hook
logic to hook-attach form, sharing the same model reference as eval_tau2_bench.py).

Key canonical invariants preserved:
  - K-side steering (k_norm if present, else k_proj).
  - Expert SVDs loaded from expert_paths.json (separately trained per-facet),
    NOT an equal-rank split of a single ontology matrix.
  - Routing uses the last prompt-token's per-head Q.
  - Coefficients are MAX-NORMALIZED over experts (not softmax).
  - Steer_mask gates which positions receive K-bias (marker-span only).
  - Alignment is q̂·U·S (signed, singular-value weighted).

The only deliberate difference from the original class is that we attach hooks
to the caller's already-loaded model instead of instantiating a fresh
AdaptiveSEKALLM object (which would double GPU memory).
"""
from __future__ import annotations

import json
import os
from contextlib import contextmanager
from typing import Dict, List, Optional

import torch


class CanonicalAdaSEKAEngine:
    def __init__(
        self,
        model,
        expert_paths_json: str,
        layers: str = "last10",
        topk: int = 3,
        temperature: float = 1.0,
        combo: str = "weighted_top_k",
        seka_root: str = "external/SEKA",
        n_kv: Optional[int] = None,
        n_q: Optional[int] = None,
        head_dim: Optional[int] = None,
    ):
        self.model = model
        self.topk = topk
        self.T = temperature
        self.combo = combo
        self.n_kv = n_kv
        self.n_q = n_q
        self.head_dim = head_dim

        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype

        with open(expert_paths_json) as f:
            paths = json.load(f)

        self.expert_names: List[str] = list(paths.keys())
        self.expert_U: Dict[str, torch.Tensor] = {}
        self.expert_S: Dict[str, torch.Tensor] = {}
        self.expert_layers: Dict[str, List[int]] = {}

        for name, path in paths.items():
            full = path if os.path.isabs(path) else os.path.join(seka_root, path)
            obj = torch.load(full, map_location=device)
            if isinstance(obj, dict) and "U_matrices" in obj:
                U = obj["U_matrices"].to(device, dtype=dtype)
                S = obj["singular_values"].to(device, dtype=dtype)
                self.expert_layers[name] = obj.get("layers", None)
            elif isinstance(obj, dict) and "proj" in obj:
                proj = obj["proj"].to(device, dtype=dtype)
                U, S = _svd_decompose(proj)
                self.expert_layers[name] = obj.get("layers", None)
            else:
                proj = obj.to(device, dtype=dtype)
                U, S = _svd_decompose(proj)
                self.expert_layers[name] = None

            if U.ndim == 2:
                U = U.unsqueeze(0).unsqueeze(0)
                S = S.unsqueeze(0).unsqueeze(0)
            elif U.ndim == 3:
                U = U.unsqueeze(1)
                S = S.unsqueeze(1)
            self.expert_U[name] = U
            self.expert_S[name] = S

        first = self.expert_names[0]
        L_sel, H_sel, d, _ = self.expert_U[first].shape
        assert head_dim is None or head_dim == d, f"head_dim mismatch {head_dim} vs {d}"
        assert n_kv is None or n_kv == H_sel, f"n_kv mismatch {n_kv} vs {H_sel}"

        self.n_kv = self.n_kv or H_sel
        self.head_dim = self.head_dim or d

        L_total = len(model.model.layers)
        self.sel_layers = _parse_layers(layers, L_total)
        assert len(self.sel_layers) == L_sel, (
            f"layers spec '{layers}' → {len(self.sel_layers)} layers, "
            f"expert SVDs have {L_sel}. Mismatch."
        )

    def _get_last_token_q_per_head(self, hidden_states: torch.Tensor, layer_idx: int) -> torch.Tensor:
        last_idx = hidden_states.shape[1] - 1
        tok_h = hidden_states[0, last_idx, :]
        attn = self.model.model.layers[layer_idx].self_attn
        ln = self.model.model.layers[layer_idx].input_layernorm
        norm_h = ln(tok_h.unsqueeze(0).unsqueeze(0))[0, 0, :]
        q = attn.q_proj(norm_h)
        q_view = q.view(self.n_q, self.head_dim)
        if hasattr(attn, "q_norm"):
            q_view = attn.q_norm(q_view)
        if self.n_q != self.n_kv:
            assert self.n_q % self.n_kv == 0, "n_q must divide n_kv for GQA"
            g = self.n_q // self.n_kv
            q_view = q_view.view(self.n_kv, g, self.head_dim).mean(dim=1)
        return q_view  # (n_kv, head_dim)

    def _routing_coefficients(self, q_head: torch.Tensor, i_layer: int, h: int) -> torch.Tensor:
        q_norm = q_head / (torch.norm(q_head) + 1e-8)
        coeffs = []
        for name in self.expert_names:
            U = self.expert_U[name][i_layer, h]
            S = self.expert_S[name][i_layer, h]
            if torch.isnan(U).any() or torch.isnan(S).any():
                coeffs.append(torch.tensor(float("-inf"), device=q_head.device, dtype=q_head.dtype))
                continue
            k = min(self.topk, U.shape[1]) if self.topk is not None else U.shape[1]
            if self.combo == "weighted_top_k":
                top_U = U[:, :k]
                top_S = S[:k]
                align = torch.matmul(q_norm, top_U)
                if self.T != 1.0:
                    align = align / self.T
                coef = (align * top_S).sum()
            elif self.combo == "all_weighted":
                align = torch.matmul(q_norm, U)
                if self.T != 1.0:
                    align = align / self.T
                coef = (align * S).sum()
            elif self.combo == "top_k_uniform":
                top_U = U[:, :k]
                align = torch.matmul(q_norm, top_U)
                if self.T != 1.0:
                    align = align / self.T
                coef = align.mean()
            else:
                raise ValueError(f"unknown combo {self.combo}")
            coeffs.append(coef)
        coeffs = torch.stack(coeffs)
        abs_max = torch.max(torch.abs(coeffs))
        coeffs = coeffs / (abs_max + 1e-8)
        if torch.isnan(coeffs).any() or torch.isinf(coeffs).any():
            coeffs = torch.ones_like(coeffs) / coeffs.numel()
        return coeffs

    def _dynamic_projection(self, coeffs: torch.Tensor, i_layer: int, h: int) -> torch.Tensor:
        first = self.expert_names[0]
        dtype = self.expert_U[first].dtype
        device = self.expert_U[first].device
        proj = torch.zeros(self.head_dim, self.head_dim, device=device, dtype=dtype)
        for e, name in enumerate(self.expert_names):
            U = self.expert_U[name][i_layer, h]
            if torch.isnan(U).any():
                continue
            k = min(self.topk, U.shape[1]) if self.topk is not None else U.shape[1]
            U_top = U[:, :k]
            expert_proj = U_top @ U_top.T
            proj = proj + coeffs[e] * expert_proj
        return proj

    @torch.no_grad()
    def precompute_projections(self, input_ids: torch.Tensor) -> Dict:
        input_ids = input_ids.to(next(self.model.parameters()).device)
        out = self.model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
        proj_table: Dict = {}
        for i, L in enumerate(self.sel_layers):
            h_states = out.hidden_states[L + 1]
            q_heads = self._get_last_token_q_per_head(h_states, L)
            for h in range(self.n_kv):
                coeffs = self._routing_coefficients(q_heads[h], i, h)
                proj_table[(i, h)] = self._dynamic_projection(coeffs, i, h)
        return proj_table

    @contextmanager
    def install_hooks(self, input_ids: torch.Tensor, steer_mask: torch.Tensor, amplify: float):
        """Canonical K-side hook: applies amp*P_dyn @ K to marker-span positions
        on the k_norm output (or k_proj if k_norm absent)."""
        proj_table = self.precompute_projections(input_ids)
        handles = []
        n_kv, head_dim = self.n_kv, self.head_dim
        sm = steer_mask
        try:
            for i, L in enumerate(self.sel_layers):
                attn = self.model.model.layers[L].self_attn
                module = attn.k_norm if hasattr(attn, "k_norm") else attn.k_proj

                def _make_hook(i_layer=i, sm_t=sm, amp=amplify):
                    def _hook(mod, inputs, output):
                        k_in = output
                        need_reshape = False
                        if k_in.dim() == 3:
                            B, T, D = k_in.shape
                            if D != n_kv * head_dim:
                                return k_in
                            k_view = k_in.view(B, T, n_kv, head_dim).clone()
                            need_reshape = True
                        elif k_in.dim() == 4:
                            B, T, H, d = k_in.shape
                            k_view = k_in.clone()
                        else:
                            return k_in

                        if sm_t is None or sm_t.sum() == 0:
                            return k_in

                        mask_b = sm_t[0].to(k_view.device)
                        if mask_b.shape[0] != T:
                            return k_in
                        if mask_b.sum() == 0:
                            return k_in

                        for h in range(n_kv):
                            dyn = proj_table.get((i_layer, h))
                            if dyn is None:
                                continue
                            if dyn.device != k_view.device:
                                dyn = dyn.to(k_view.device)
                            k_sel = k_view[0, mask_b, h, :]
                            delta = torch.matmul(dyn.to(k_sel.dtype), k_sel.T).T
                            k_view[0, mask_b, h, :] = k_sel + amp * delta

                        if need_reshape:
                            return k_view.reshape(B, T, n_kv * head_dim).to(k_in.dtype)
                        return k_view.to(k_in.dtype)
                    return _hook

                handles.append(module.register_forward_hook(_make_hook()))
            yield
        finally:
            for h in handles:
                h.remove()


def _svd_decompose(projections: torch.Tensor):
    L, H, d, _ = projections.shape
    U_out = torch.zeros(L, H, d, d, device=projections.device, dtype=projections.dtype)
    S_out = torch.zeros(L, H, d, device=projections.device, dtype=projections.dtype)
    for l in range(L):
        for h in range(H):
            try:
                U, S, _ = torch.svd(projections[l, h])
                U_out[l, h] = U
                S_out[l, h] = S
            except Exception:
                U_out[l, h] = torch.eye(d, device=projections.device, dtype=projections.dtype)
                S_out[l, h] = torch.ones(d, device=projections.device, dtype=projections.dtype)
    return U_out, S_out


def _parse_layers(spec: str, total: int) -> List[int]:
    if spec == "all":
        return list(range(total))
    if spec.startswith("last"):
        k = int(spec[4:])
        return list(range(total - k, total))
    if spec.startswith("first"):
        k = int(spec[5:])
        return list(range(k))
    return [int(i) for i in spec.split(",")]


def build_marker_steer_mask(
    prompt_text: str,
    tokenizer,
    marker: str = "**",
) -> (torch.Tensor, torch.Tensor):
    """Strip marker pairs from prompt_text, tokenize, return (ids, steer_mask).
    Prompt should contain matching `**...**` around the user message content."""
    import re
    pat = re.escape(marker) + r"(.*?)" + re.escape(marker)
    pieces, spans, last = [], [], 0
    for m in re.finditer(pat, prompt_text, flags=re.DOTALL):
        pieces.append(prompt_text[last:m.start()])
        start_plain = sum(len(p) for p in pieces)
        content = m.group(1)
        pieces.append(content)
        spans.append((start_plain, start_plain + len(content)))
        last = m.end()
    pieces.append(prompt_text[last:])
    stripped = "".join(pieces)

    enc = tokenizer(stripped, return_offsets_mapping=True, return_tensors="pt", add_special_tokens=False)
    ids = enc["input_ids"]
    mask = torch.zeros_like(ids, dtype=torch.bool)
    for s, e in spans:
        for tid, (cs, ce) in enumerate(enc["offset_mapping"][0]):
            if cs >= (s - 1) and ce <= e and not (cs == ce == 0):
                mask[0, tid] = True
    return ids, mask
