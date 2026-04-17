#!/usr/bin/env python3
"""build_placeholder_figures.py — Generate paper placeholder figures.

Every figure is rendered with a clear "placeholder" watermark overlay so that
reviewers can see the expected shape of each result without mistaking the
sketch for a measurement. When real result JSONs land, the corresponding
`build_fig_*_from_json.py` script (one per figure) will overwrite these PDFs.

Outputs land in `paper/neurips2026_steering_ko/figures/` and, after the
English mirror step, are copied to `paper/neurips2026_steering_v2/figures/`.

Usage:
    python build_placeholder_figures.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


HERE = Path(__file__).resolve().parent
OUT_KO = HERE.parent / "figures"
OUT_V2 = HERE.parent.parent / "neurips2026_steering_v2" / "figures"
OUT_KO.mkdir(parents=True, exist_ok=True)
OUT_V2.mkdir(parents=True, exist_ok=True)


def _watermark(ax):
    ax.text(
        0.5,
        0.5,
        "PLACEHOLDER",
        transform=ax.transAxes,
        fontsize=36,
        color="#c0c0c0",
        alpha=0.25,
        ha="center",
        va="center",
        rotation=30,
        zorder=0,
    )


def _save(fig, name):
    for out_dir in (OUT_KO, OUT_V2):
        fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
        fig.savefig(out_dir / f"{name}.png", dpi=160, bbox_inches="tight")


def fig1_concept():
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.2))
    panels = [
        ("SEKA (stationary K)", "#d73027", r"$k' = k + \alpha P k$" + "\n(every step, every layer)"),
        ("Q-coverage (ours)", "#1a9850", r"$q' = q + \beta P q,\ \beta<0$" + "\n(history-free, every layer)"),
        (
            "Layer-Adaptive K+Q (ours)",
            "#4575b4",
            r"K on $\ell<L/4$, Q on all $\ell$" + "\n(imprint + coverage)",
        ),
    ]
    for ax, (title, col, eq) in zip(axes, panels):
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.1, 0.2), 0.8, 0.6,
            boxstyle="round,pad=0.02",
            facecolor=col, alpha=0.15, edgecolor=col, linewidth=1.5,
        ))
        ax.text(0.5, 0.78, title, ha="center", va="center",
                fontsize=11.5, fontweight="bold", color=col)
        ax.text(0.5, 0.5, eq, ha="center", va="center", fontsize=10)
        ax.text(0.5, 0.25, "history-free", ha="center", va="center",
                fontsize=8, color="gray", style="italic")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
    fig.suptitle(
        "Figure 1. Three intervention families on the ontology subspace $\\mathrm{Im}(B_{\\mathrm{ont}})$.",
        fontsize=11,
    )
    fig.tight_layout()
    _save(fig, "fig1_concept")
    plt.close(fig)


def fig2_delta_vs_k():
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    k = np.array([1, 2, 3, 4, 5, 6, 8, 10, 13])
    seka = -2 - 0.3 * k + 0.1 * np.random.RandomState(0).randn(len(k))
    stat_k = np.where(k == 1, +10.5, -3 - 0.5 * (k - 1)) + 0.2 * np.random.RandomState(1).randn(len(k))
    q_only = 2.0 + 0.15 * np.log(k) + 0.1 * np.random.RandomState(2).randn(len(k))
    ladapt = 3.5 - 0.45 * (k - 2) + 0.15 * np.random.RandomState(3).randn(len(k))

    ax.axhline(0, color="gray", linewidth=0.8)
    ax.plot(k, seka, "o-", color="#d73027", label="SEKA amp=1.0", linewidth=2)
    ax.plot(k, stat_k, "s--", color="#fc8d59", label="Stationary K (ours, all layers)", linewidth=1.5)
    ax.plot(k, q_only, "^-", color="#1a9850", label="Q-coverage (ours)", linewidth=2)
    ax.plot(k, ladapt, "D-", color="#4575b4", label="Layer-Adaptive K+Q (ours)", linewidth=2)

    ax.set_xlabel("ground-truth tool count $k$", fontsize=11)
    ax.set_ylabel(r"$\Delta$ F1 vs no\_steer (pp)", fontsize=11)
    ax.set_title("Figure 2. Regime-split: $\\Delta$F1 as function of $k$", fontsize=11)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.95)
    ax.grid(alpha=0.25)
    _watermark(ax)
    fig.tight_layout()
    _save(fig, "fig2_delta_vs_k")
    plt.close(fig)


def fig3_stepwise():
    fig, ax = plt.subplots(figsize=(8.2, 4.3))
    categories = ["first_hit", "second_hit", "second_distinct", "repeated_first"]
    methods = [
        ("no_steer", [72, 54, 50, 12], "#999999"),
        ("SEKA amp=1.0", [70, 35, 28, 32], "#d73027"),
        ("Q-coverage (ours)", [72, 61, 58, 8], "#1a9850"),
        ("Layer-Adaptive K+Q (ours)", [76, 66, 62, 7], "#4575b4"),
    ]
    x = np.arange(len(categories))
    width = 0.2
    for i, (name, vals, col) in enumerate(methods):
        offset = (i - 1.5) * width
        ax.bar(x + offset, vals, width, label=name, color=col, alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in categories], fontsize=10)
    ax.set_ylabel("rate (%)", fontsize=11)
    ax.set_title(
        "Figure 3. Stepwise coverage on MetaTool Subtask4 N=497.\n"
        "SEKA raises repeated_first (Cor. 4.2); ours lowers it and raises second_distinct_hit.",
        fontsize=10,
    )
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    ax.grid(axis="y", alpha=0.25)
    _watermark(ax)
    fig.tight_layout()
    _save(fig, "fig3_stepwise")
    plt.close(fig)


def fig4_basis():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.8, 3.9))
    bases = ["Real $B_{ont}$", "Feature-shuffled", "Random orth.", "PCA-of-K"]
    q_f1 = [74.7, 72.5, 70.7, 72.0]  # PCA placeholder
    k_f1 = [68.5, 0.0, 0.0, 35.0]    # PCA placeholder
    colors = ["#4575b4", "#fc8d59", "#d73027", "#fee090"]
    ax1.barh(bases, q_f1, color=colors, edgecolor="black", linewidth=0.5)
    ax1.axvline(73.07, color="black", linestyle="--", linewidth=1, label="no_steer")
    ax1.set_xlabel("Q-coverage F1 (×100)")
    ax1.set_xlim(55, 80)
    ax1.set_title("(a) Q-coverage")
    ax1.legend(loc="lower right", fontsize=8)
    ax1.grid(axis="x", alpha=0.25)

    ax2.barh(bases, k_f1, color=colors, edgecolor="black", linewidth=0.5)
    ax2.axvline(73.07, color="black", linestyle="--", linewidth=1, label="no_steer")
    ax2.set_xlabel("Stationary K F1 (×100)")
    ax2.set_xlim(0, 80)
    ax2.set_title("(b) Stationary K")
    ax2.legend(loc="lower right", fontsize=8)
    ax2.grid(axis="x", alpha=0.25)

    fig.suptitle(
        "Figure 4. Basis specificity on Subtask4 (Qwen2.5-7B, N=497). "
        "Real basis only — random and shuffled bases break stationary K entirely.",
        fontsize=10,
    )
    _watermark(ax1)
    _watermark(ax2)
    fig.tight_layout()
    _save(fig, "fig4_basis")
    plt.close(fig)


def fig5_size_sweep():
    fig, ax = plt.subplots(figsize=(7.8, 4.0))
    sizes = np.array([1.5, 3.0, 7.0, 14.0])
    seka = np.array([-6.2, -8.5, -12.0, -9.5])
    q_only = np.array([1.1, 1.8, 2.3, 2.0])
    ladapt = np.array([0.6, 1.5, 2.1, 1.8])
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.plot(sizes, seka, "o-", color="#d73027", label="SEKA amp=1.0", linewidth=2)
    ax.plot(sizes, q_only, "^-", color="#1a9850", label="Q-coverage (ours)", linewidth=2)
    ax.plot(sizes, ladapt, "D-", color="#4575b4", label="Layer-Adaptive K+Q (ours)", linewidth=2)
    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels([f"{s:g}B" for s in sizes])
    ax.set_xlabel("Qwen2.5-Instruct size (log scale)", fontsize=11)
    ax.set_ylabel(r"$\Delta$ F1 vs no\_steer (pp)", fontsize=11)
    ax.set_title("Figure 5. Model-size robustness on MetaTool Subtask4.", fontsize=11)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    ax.grid(alpha=0.25)
    _watermark(ax)
    fig.tight_layout()
    _save(fig, "fig5_size_sweep")
    plt.close(fig)


if __name__ == "__main__":
    fig1_concept()
    fig2_delta_vs_k()
    fig3_stepwise()
    fig4_basis()
    fig5_size_sweep()
    print("[figures] wrote 5 placeholder figures to", OUT_KO, "and", OUT_V2)
