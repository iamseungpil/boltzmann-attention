#!/usr/bin/env python3
"""H-Path-E — Active LLM retrieval with residual ontology projection.

Path C (K·K and Q·K) achieved 0% top-1 — passive encoding doesn't carry
retrieval signal in Qwen2.5-7B. Path E measures the LLM's ACTUAL retrieval
behavior via active generation, then probes whether Park-style ontology
projection in the FINAL residual space (where W_U operates) preserves /
enhances / nullifies the signal.

Pipeline (per spec):
  1. Load MetaTool plugins (388 with desc) + Subtask1, subsample N=100.
  2. For each tool, capture residual at output of `model.model.norm`
     (final layer-norm, pre-LM-head) at prompt-end on raw description.
  3. Build 28 verb + 21 domain Park directions in residual space, SVD →
     U_verb, U_dom; combined → U_onto via re-orthonormalization.
     P_onto, P_perp, P_rand (49 Gaussian, orthonormalized).
  4. Park Thm 8 native check: ||P_verb P_dom||_F / sqrt(min(r_v,r_d)) < 0.3.
  5. Active retrieval prompt with all 388 tool names + the query,
     final residual h_q at last position.
  6. Tool-name first-token w_t = W_U[tok_t] (with leading space).
  7. 4-arm scoring: F / O / ⊥ / R.  Top-1 / Top-5 / MRR.
  8. Bootstrap 1000x CI.
  9. 5 pre-reg gates → joint outcome.
  10. JSON to reports/boltzmann_energy/path_e_active_retrieval.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "scripts" / "new_theorem_test"))
sys.path.insert(0, str(HERE))

from afod_probe_metatool_toolbench import extract_verb, extract_domain  # noqa: E402

# Reuse pure-math helpers from prior Path C script
from measure_h_onto_hopfield import (  # noqa: E402
    orthonormal_basis,
    retrieval_metrics,
    bootstrap_top1_ci,
)

METATOOL_INFO = "/tmp/MetaTool/dataset/plugin_info.json"
METATOOL_ST1 = "/tmp/MetaTool/dataset/tmp_dataset/Task2-Subtask1.json"
DEFAULT_OUT = REPO / "reports" / "boltzmann_energy" / "path_e_active_retrieval.json"


# ---------------------------------------------------------------
# Final residual capture (output of model.model.norm)
# ---------------------------------------------------------------

@torch.no_grad()
def capture_final_residual(model, tok, text: str,
                           max_len: int | None = None) -> Tuple[torch.Tensor, int] | Tuple[None, int]:
    """Capture residual at OUTPUT of `model.model.norm` at last position.

    This is the pre-LM-head representation that, when multiplied by W_U,
    produces the next-token logits.

    Returns:
        (h_last (d_model,) float CPU tensor, token_count int) on success;
        (None, token_count) on failure.
    """
    enc = tok(text, return_tensors="pt", add_special_tokens=True,
              truncation=(max_len is not None), max_length=max_len)
    ids = enc["input_ids"].to(next(model.parameters()).device)
    T = int(ids.shape[1])
    cap: dict = {}

    def _hook(mod, inputs, output):
        # output: (B, T, d_model) post-RMSNorm
        if isinstance(output, tuple):
            output = output[0]
        if output.dim() == 3:
            cap["h"] = output[0, -1, :].detach().float().cpu()

    h = model.model.norm.register_forward_hook(_hook)
    try:
        model(input_ids=ids, use_cache=False)
    finally:
        h.remove()

    return cap.get("h"), T


# ---------------------------------------------------------------
# Ortho-normalized random control directions (49 in d_model)
# ---------------------------------------------------------------

def random_unit_dirs_orthonormal(n: int, d: int, seed: int) -> np.ndarray:
    """49 Gaussian unit vectors → SVD orthonormalize → (d, r) basis (r ≤ n).

    Used as a random-projection control with comparable rank to U_onto.
    """
    rng = np.random.default_rng(seed)
    G = rng.standard_normal(size=(n, d)).astype(np.float32)
    G /= (np.linalg.norm(G, axis=1, keepdims=True) + 1e-12)
    # SVD on (d, n) — same procedure as U_onto
    U_r = orthonormal_basis(G.T)  # (d, r)
    return U_r


# ---------------------------------------------------------------
# Build active-retrieval prompt
# ---------------------------------------------------------------

def build_active_prompt(plugin_names: List[str], query: str) -> str:
    """Single prompt listing all tool names + query + retrieval cue."""
    return (
        "You have access to the following tools: "
        + ", ".join(plugin_names)
        + "\nUser query: " + query
        + "\nThe best tool to use is:"
    )


# ---------------------------------------------------------------
# Tool-name first-token lookup (with leading space)
# ---------------------------------------------------------------

def tool_first_token_id(tok, name: str) -> int:
    """Tokenize ' <name>' (mid-context style) and return the first BPE id."""
    ids = tok(" " + name, add_special_tokens=False)["input_ids"]
    if len(ids) == 0:
        # fallback: no leading space
        ids = tok(name, add_special_tokens=False)["input_ids"]
    return int(ids[0])


# ---------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-queries", type=int, default=100)
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-prompt-tokens", type=int, default=32000)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    t_start = time.time()

    partial: Dict = {
        "variant": "path_e_active_retrieval",
        "model": args.model,
        "dataset": "MetaTool-Subtask1",
        "seed": args.seed,
        "n_bootstrap": args.n_bootstrap,
    }

    try:
        rng = np.random.default_rng(args.seed)
        torch.manual_seed(args.seed)

        # -- Step 1: dataset loading --
        print("[Step 1] loading plugin_info + Subtask1 ...", flush=True)
        info = json.load(open(METATOOL_INFO))
        st1 = json.load(open(METATOOL_ST1))

        plugins: Dict[str, dict] = {}
        for p in info:
            n = p.get("name_for_model") or p.get("name_for_human")
            if not n:
                continue
            desc = (p.get("description_for_model") or "").strip()
            if not desc:
                continue
            plugins[n] = {
                "desc": desc[:400],
                "verb": extract_verb(desc),
                "domain": extract_domain(desc),
            }
        plugin_names = sorted(plugins.keys())
        plugin_idx = {n: i for i, n in enumerate(plugin_names)}
        n_tools = len(plugin_names)
        verb_labels = [plugins[n]["verb"] for n in plugin_names]
        domain_labels = [plugins[n]["domain"] for n in plugin_names]
        verb_uniq = sorted(set(verb_labels))
        domain_uniq = sorted(set(domain_labels))
        n_verb = len(verb_uniq)
        n_domain = len(domain_uniq)
        print(f"[catalog] n_tools={n_tools}  n_verb={n_verb}  n_domain={n_domain}", flush=True)

        pool = [e for e in st1 if e.get("tool") in plugins]
        print(f"[queries pool] {len(pool)} / {len(st1)} GT ∈ catalog", flush=True)
        assert len(pool) >= args.n_queries
        idx_perm = rng.permutation(len(pool))[:args.n_queries]
        queries = [pool[i] for i in idx_perm]
        n_q = len(queries)
        print(f"[queries] using N={n_q}", flush=True)

        partial.update({
            "n_tools": n_tools,
            "n_verb_directions": n_verb,
            "n_domain_directions": n_domain,
            "n_queries": n_q,
        })

        # -- Load model --
        print(f"[model] loading {args.model} fp16 on {args.device}", flush=True)
        tok = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float16,
        ).to(args.device)
        model.eval()
        cfg = model.config
        d_model = cfg.hidden_size
        L_total = len(model.model.layers)
        print(f"[config] d_model={d_model}  L={L_total}  vocab={cfg.vocab_size}", flush=True)
        partial["d_model"] = d_model

        # -- Verify model.model.norm and lm_head exist --
        assert hasattr(model.model, "norm"), "model.model.norm missing"
        assert hasattr(model, "lm_head"), "model.lm_head missing"

        # -- Step 2: residual extraction for each tool --
        print(f"[Step 2] extracting final residuals for {n_tools} tools (output of model.model.norm)",
              flush=True)
        H = np.zeros((n_tools, d_model), dtype=np.float32)
        t0 = time.time()
        for i, name in enumerate(plugin_names):
            h, T = capture_final_residual(model, tok, plugins[name]["desc"])
            if h is None:
                raise RuntimeError(f"residual capture failed for {name}")
            H[i] = h.numpy()
            if (i + 1) % 100 == 0 or i == 0:
                elapsed = time.time() - t0
                eta = elapsed / (i + 1) * (n_tools - i - 1)
                print(f"    [resid {i+1}/{n_tools}] T={T} t={elapsed:.1f}s eta={eta:.1f}s",
                      flush=True)
        h_norms = np.linalg.norm(H, axis=1)
        print(f"[Step 2] H norm: mean={h_norms.mean():.3f} min={h_norms.min():.3f} "
              f"max={h_norms.max():.3f}", flush=True)

        # -- Park directions in residual space --
        verb_arr = np.array(verb_labels)
        domain_arr = np.array(domain_labels)

        def compute_dirs(arr, uniq):
            D = np.zeros((len(uniq), d_model), dtype=np.float32)
            for ci, c in enumerate(uniq):
                pos = (arr == c)
                neg = ~pos
                if pos.sum() == 0 or neg.sum() == 0:
                    continue
                d = H[pos].mean(axis=0) - H[neg].mean(axis=0)
                nrm = np.linalg.norm(d)
                if nrm > 1e-8:
                    D[ci] = d / nrm
            return D

        U_verb_dirs = compute_dirs(verb_arr, verb_uniq)   # (n_verb, d_model)
        U_dom_dirs = compute_dirs(domain_arr, domain_uniq)  # (n_domain, d_model)

        # SANITY: direction norms (should be ~1.0 after normalization, or 0 if degenerate)
        verb_norms = np.linalg.norm(U_verb_dirs, axis=1)
        dom_norms = np.linalg.norm(U_dom_dirs, axis=1)
        n_verb_alive = int((verb_norms > 0.01).sum())
        n_dom_alive = int((dom_norms > 0.01).sum())
        print(f"[Step 2] verb dirs: {n_verb_alive}/{n_verb} alive (norm > 0.01)", flush=True)
        print(f"[Step 2] domain dirs: {n_dom_alive}/{n_domain} alive (norm > 0.01)", flush=True)
        if n_verb_alive == 0 or n_dom_alive == 0:
            print("[WARN] all directions collapsed — check label distribution", flush=True)

        # -- SVD orthonormalize per category, then combine --
        # Use (d_model, n_cat) layout for orthonormal_basis
        U_verb = orthonormal_basis(U_verb_dirs.T, tol=1e-5)   # (d_model, r_v)
        U_dom = orthonormal_basis(U_dom_dirs.T, tol=1e-5)     # (d_model, r_d)
        r_v = U_verb.shape[1]
        r_d = U_dom.shape[1]
        # Combined onto = SVD orthonormalize of [U_verb | U_dom]
        U_onto = orthonormal_basis(np.concatenate([U_verb, U_dom], axis=1), tol=1e-5)
        r_onto = U_onto.shape[1]
        print(f"[Step 2/SVD] r_verb={r_v}  r_dom={r_d}  r_onto={r_onto}  (max={d_model})",
              flush=True)

        # -- Projectors (built lazily for memory; full d_model x d_model = 3584^2 ~ 51 MB ok) --
        P_onto = (U_onto @ U_onto.T).astype(np.float32)
        P_perp = (np.eye(d_model, dtype=np.float32) - P_onto)

        # Random control: 49 Gaussian dirs → SVD orthonormalize
        U_rand = random_unit_dirs_orthonormal(n_verb + n_domain, d_model, args.seed)
        r_rand = U_rand.shape[1]
        P_rand = (U_rand @ U_rand.T).astype(np.float32)
        print(f"[Step 2/random] r_rand={r_rand}", flush=True)

        # -- Step 3: Park Thm 8 in FINAL residual space --
        P_verb = U_verb @ U_verb.T
        P_dom = U_dom @ U_dom.T
        cross_fro = float(np.linalg.norm(P_verb @ P_dom, ord="fro"))
        normalizer = float(np.sqrt(min(r_v, r_d))) if min(r_v, r_d) > 0 else 1.0
        verb_dom_fro = cross_fro / normalizer
        g_park_pass = bool(verb_dom_fro < 0.3)
        print(f"[Step 3] Park Thm8 final residual: ||P_verb P_dom||_F={cross_fro:.4f}  "
              f"normalized={verb_dom_fro:.4f}  pass={g_park_pass}", flush=True)

        # -- Step 5: Tool-name first-token IDs and W_U row extraction --
        print("[Step 5] tool-name first-token IDs + W_U rows", flush=True)
        tool_token_ids = np.zeros(n_tools, dtype=np.int64)
        for i, name in enumerate(plugin_names):
            tool_token_ids[i] = tool_first_token_id(tok, name)
        n_unique_tokens = len(set(tool_token_ids.tolist()))
        print(f"[Step 5] {n_unique_tokens} unique first-tokens / {n_tools} tools "
              f"(collisions: {n_tools - n_unique_tokens})", flush=True)

        # W_U = lm_head.weight (vocab, d_model). Extract just the rows we need.
        W_U_full = model.lm_head.weight.detach()  # fp16 on device
        # Index by tool_token_ids → (n_tools, d_model)
        w_t_torch = W_U_full[torch.from_numpy(tool_token_ids).to(W_U_full.device)]
        w_t = w_t_torch.float().cpu().numpy()  # (n_tools, d_model)
        print(f"[Step 5] w_t shape={w_t.shape}  norm_mean={np.linalg.norm(w_t, axis=1).mean():.3f}",
              flush=True)

        # SANITY: pairwise cosine of w_t (sample for speed, no need for full 388^2)
        sample_n = min(50, n_tools)
        sample_idx = np.linspace(0, n_tools - 1, sample_n, dtype=int)
        w_sample = w_t[sample_idx]
        w_sample_norm = w_sample / (np.linalg.norm(w_sample, axis=1, keepdims=True) + 1e-12)
        cos_mat = w_sample_norm @ w_sample_norm.T
        # Mask diagonal
        np.fill_diagonal(cos_mat, np.nan)
        with np.errstate(invalid="ignore"):
            cos_min = float(np.nanmin(cos_mat))
            cos_max = float(np.nanmax(cos_mat))
            cos_mean = float(np.nanmean(cos_mat))
        print(f"[Step 5/sanity] w_t pairwise cosine (sample {sample_n}): "
              f"min={cos_min:.3f} max={cos_max:.3f} mean={cos_mean:.3f}", flush=True)
        if cos_min > 0.99:
            print("[WARN] w_t vectors highly degenerate — first-tokens may be collapsed", flush=True)

        # -- Step 4: Active-retrieval prompt → query final residuals --
        # Length probe on the FIRST query first
        first_prompt = build_active_prompt(plugin_names, queries[0]["query"])
        first_enc = tok(first_prompt, return_tensors="pt", add_special_tokens=True)
        first_T = int(first_enc["input_ids"].shape[1])
        print(f"[Step 4] first prompt length: {first_T} tokens "
              f"(limit {args.max_prompt_tokens})", flush=True)
        if first_T > args.max_prompt_tokens:
            raise RuntimeError(f"prompt too long: {first_T} > {args.max_prompt_tokens}")

        # Batch capture for all queries
        print(f"[Step 4] capturing query final residuals (N={n_q})", flush=True)
        H_q = np.zeros((n_q, d_model), dtype=np.float32)
        gt_idx = np.zeros(n_q, dtype=np.int64)
        prompt_lens: List[int] = []
        h_tool_mean = H.mean(axis=0)  # for sanity

        # EARLY ABORT detection: track top-1 on the first 10 queries, baseline arm
        early_top1_correct = 0
        early_n = 0

        t0 = time.time()
        for qi, entry in enumerate(queries):
            prompt = build_active_prompt(plugin_names, entry["query"])
            h, T = capture_final_residual(model, tok, prompt,
                                          max_len=args.max_prompt_tokens)
            if h is None:
                raise RuntimeError(f"final residual capture failed at qi={qi}")
            H_q[qi] = h.numpy()
            prompt_lens.append(T)
            gt_idx[qi] = plugin_idx[entry["tool"]]

            if qi == 0:
                hq_np = h.numpy()
                hq_norm = float(np.linalg.norm(hq_np))
                hq_first5 = hq_np[:5].tolist()
                diff_norm = float(np.linalg.norm(hq_np - h_tool_mean))
                print(f"[Step 4/sanity qi=0] h_q norm={hq_norm:.3f}  "
                      f"first5={[round(x,3) for x in hq_first5]}  "
                      f"||h_q - h_tool_mean||={diff_norm:.3f}", flush=True)

            # On-the-fly baseline (Arm F) check on first 10 queries for early abort
            if qi < 10:
                # quick score: w_t @ h_q
                s = w_t @ H_q[qi]
                pred = int(np.argmax(s))
                if pred == gt_idx[qi]:
                    early_top1_correct += 1
                early_n += 1

            if (qi + 1) % 25 == 0 or qi == 0:
                elapsed = time.time() - t0
                eta = elapsed / (qi + 1) * (n_q - qi - 1)
                print(f"    [Q {qi+1}/{n_q}] T={T} t={elapsed:.1f}s eta={eta:.1f}s",
                      flush=True)

            if qi == 9:
                early_top1 = early_top1_correct / early_n
                print(f"[Step 4/early-check] top1(F) on first 10 queries = {early_top1:.3f}",
                      flush=True)
                if early_top1 < 0.05:
                    print("[Step 4/EARLY_ABORT] top1(F) < 0.05 on first 10 → "
                          "saving partial + exiting", flush=True)
                    partial["early_abort"] = "EARLY_ABORT_BASELINE_DEAD"
                    partial["early_top1_first10"] = early_top1
                    partial["n_queries_completed"] = early_n
                    partial["runtime_s"] = float(time.time() - t_start)
                    out_path = Path(args.out)
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(out_path, "w") as f:
                        json.dump(partial, f, indent=2)
                    print(f"[saved early-abort] {out_path}", flush=True)
                    return

        prompt_len_arr = np.array(prompt_lens)
        prompt_len_mean = int(prompt_len_arr.mean())
        prompt_len_max = int(prompt_len_arr.max())
        print(f"[Step 4] prompt lens: mean={prompt_len_mean} max={prompt_len_max}",
              flush=True)

        # Free GPU before scoring (W_U already extracted to CPU as w_t)
        del model, W_U_full, w_t_torch
        torch.cuda.empty_cache()

        # -- Step 6: 4-arm retrieval --
        print("[Step 6] 4-arm retrieval scoring", flush=True)

        def score_arm(P: np.ndarray | None) -> np.ndarray:
            """Return scores (n_q, n_tools) = w_t @ (P h_q) for projector P (or identity)."""
            if P is None:
                H_proj = H_q  # (n_q, d_model)
            else:
                # P is symmetric: H_q @ P^T  (n_q, d_model)
                H_proj = H_q @ P.T
            # scores[q, t] = w_t[t] · H_proj[q]
            return H_proj @ w_t.T  # (n_q, n_tools)

        scores_F = score_arm(None)
        scores_O = score_arm(P_onto)
        scores_perp = score_arm(P_perp)
        scores_R = score_arm(P_rand)

        m_F = retrieval_metrics(scores_F, gt_idx)
        m_O = retrieval_metrics(scores_O, gt_idx)
        m_perp = retrieval_metrics(scores_perp, gt_idx)
        m_R = retrieval_metrics(scores_R, gt_idx)

        # -- Step 7: bootstrap CIs --
        print(f"[Step 7] bootstrap CI (n={args.n_bootstrap})", flush=True)
        ci_F = bootstrap_top1_ci(m_F["_top1_correct"], args.n_bootstrap, args.seed)
        ci_O = bootstrap_top1_ci(m_O["_top1_correct"], args.n_bootstrap, args.seed + 1)
        ci_perp = bootstrap_top1_ci(m_perp["_top1_correct"], args.n_bootstrap, args.seed + 2)
        ci_R = bootstrap_top1_ci(m_R["_top1_correct"], args.n_bootstrap, args.seed + 3)

        def _fmt(name, m, ci):
            print(f"[arm {name:>5s}] top1={m['top1']:.4f}  top5={m['top5']:.4f}  "
                  f"mrr={m['mrr']:.4f}  CI95=[{ci[0]:.4f}, {ci[2]:.4f}]", flush=True)
        _fmt("F", m_F, ci_F)
        _fmt("O", m_O, ci_O)
        _fmt("⊥", m_perp, ci_perp)
        _fmt("R", m_R, ci_R)

        # -- Step 8: pre-reg gates --
        t1_F = m_F["top1"]
        t1_O = m_O["top1"]
        t1_perp = m_perp["top1"]
        t1_R = m_R["top1"]

        # G-E-1: top1(F) ≥ 0.15 AND CI_lo(F) ≥ 0.08
        g_e1 = bool((t1_F >= 0.15) and (ci_F[0] >= 0.08))

        eps = 1e-9
        if t1_F < eps:
            g_e2 = g_e3 = g_e4 = False
        else:
            # G-E-2: top1(O)/top1(F) ≥ 0.9 AND CI_lo(O)/top1(F) ≥ 0.80
            g_e2 = bool((t1_O / t1_F >= 0.9) and (ci_O[0] / t1_F >= 0.80))
            # G-E-3: top1(⊥)/top1(F) ≤ 0.5
            g_e3 = bool((t1_perp / t1_F) <= 0.5)
            # G-E-4: |top1(R) - top1(⊥)|/top1(F) ≤ 0.15
            g_e4 = bool(abs(t1_R - t1_perp) / t1_F <= 0.15)

        gates = {
            "G_E_1_baseline_pass": g_e1,
            "G_E_2_ontology_pass": g_e2,
            "G_E_3_ortho_pass": g_e3,
            "G_E_4_random_pass": g_e4,
            "G_Park_Thm8_Final_pass": g_park_pass,
        }
        print(f"[gates] {gates}", flush=True)

        # -- Step 9: joint outcome --
        if not g_e1:
            joint = "BASELINE_DEAD_PATH_E"
        elif g_e1 and g_e2 and g_e3 and g_e4 and g_park_pass:
            joint = "STRONG_POSITIVE"
        elif g_e1 and g_e2 and g_e3 and g_e4 and not g_park_pass:
            joint = "ONTOLOGY_HELPS"
        elif g_e1 and g_e2 and not g_e3:
            joint = "BASELINE_ONLY"
        elif g_e1 and not g_e2:
            joint = "PROJECTION_DETRIMENTAL"
        else:
            joint = "PARTIAL"
        print(f"[joint] → {joint}", flush=True)

        # -- Step 10: output --
        try:
            import subprocess
            sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=str(REPO)).decode().strip()
        except Exception:
            sha = "unknown"

        out = {
            "variant": "path_e_active_retrieval",
            "model": args.model,
            "dataset": "MetaTool-Subtask1",
            "n_queries": n_q,
            "n_tools": n_tools,
            "prompt_length_tokens_mean": prompt_len_mean,
            "prompt_length_tokens_max": prompt_len_max,
            "n_verb_directions": n_verb,
            "n_domain_directions": n_domain,
            "r_verb": r_v,
            "r_dom": r_d,
            "r_onto": r_onto,
            "r_rand": r_rand,
            "d_model": d_model,
            "seed": args.seed,
            "n_bootstrap": args.n_bootstrap,
            "n_unique_tool_first_tokens": n_unique_tokens,
            "w_t_pairwise_cosine_sample": {
                "min": cos_min, "max": cos_max, "mean": cos_mean,
                "n_sampled": sample_n,
            },
            "park_thm8_final": {
                "verb_dom_fro": cross_fro,
                "normalized_fro": verb_dom_fro,
                "pass": g_park_pass,
            },
            "arm_F": {
                "top1": m_F["top1"], "top5": m_F["top5"], "mrr": m_F["mrr"],
                "basin_depth_mean": m_F["basin_depth_mean"],
                "top1_ci_95": [ci_F[0], ci_F[2]], "top1_ci_mean": ci_F[1],
            },
            "arm_O": {
                "top1": m_O["top1"], "top5": m_O["top5"], "mrr": m_O["mrr"],
                "basin_depth_mean": m_O["basin_depth_mean"],
                "top1_ci_95": [ci_O[0], ci_O[2]], "top1_ci_mean": ci_O[1],
            },
            "arm_ortho": {
                "top1": m_perp["top1"], "top5": m_perp["top5"], "mrr": m_perp["mrr"],
                "basin_depth_mean": m_perp["basin_depth_mean"],
                "top1_ci_95": [ci_perp[0], ci_perp[2]], "top1_ci_mean": ci_perp[1],
            },
            "arm_R": {
                "top1": m_R["top1"], "top5": m_R["top5"], "mrr": m_R["mrr"],
                "basin_depth_mean": m_R["basin_depth_mean"],
                "top1_ci_95": [ci_R[0], ci_R[2]], "top1_ci_mean": ci_R[1],
            },
            "gates": gates,
            "joint_outcome": joint,
            "runtime_s": float(time.time() - t_start),
            "git_sha": sha,
        }

        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[saved] {out_path}", flush=True)
        print(f"[summary] F={t1_F:.3f}  O={t1_O:.3f}  ⊥={t1_perp:.3f}  R={t1_R:.3f}  "
              f"Park8={verb_dom_fro:.3f}  → {joint}  "
              f"runtime={out['runtime_s']:.1f}s", flush=True)

    except Exception as e:
        tb = traceback.format_exc()
        print(f"[ERROR] {e}\n{tb}", flush=True)
        partial["error"] = str(e)
        partial["traceback"] = tb
        partial["runtime_s"] = float(time.time() - t_start)
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(partial, f, indent=2)
        print(f"[saved partial] {out_path}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
