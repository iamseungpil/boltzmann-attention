#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
FIG_DIR = ROOT / "figures"


def load_json(path: str) -> dict:
    with open(REPO / path, "r", encoding="utf-8") as f:
        return json.load(f)


def macro(path: str, method: str) -> dict:
    obj = load_json(path)
    for row in obj["results"]:
        if row["method"] == method:
            return row["macro"]
    raise KeyError(f"{method} not found in {path}")


def save_fig(fig: plt.Figure, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{stem}.{ext}", bbox_inches="tight", dpi=220)
    plt.close(fig)


def fig_main_results() -> None:
    qwen_base = macro("reports/qkv_joint_2026_04_15/full497_qbias_sweep.json", "no_steer")
    qwen_q = macro("reports/qkv_joint_2026_04_15/full497_qbias_sweep.json", "ocq_qbias_b-0.1")
    qwen_rand = macro("reports/wave_2026_04_15_pm/gpu1/qwen_st4_qbias_b-0.1_random_bont.json", "ocq_qbias_b-0.1")
    qwen_fshuf = macro("reports/wave_2026_04_15_pm/gpu1/qwen_st4_qbias_b-0.1_featshuffle_bont.json", "ocq_qbias_b-0.1")
    qwen_k = macro("reports/subtask4_overnight/st4_real_N0.json", "ocq_bias_a0.3")
    qwen_k_rand = macro("reports/subtask4_overnight/st4_random_N0.json", "ocq_bias_a0.3")
    qwen_k_fshuf = macro("reports/subtask4_overnight/st4_featshuffle_N0.json", "ocq_bias_a0.3")

    llama_base = macro("reports/wave_pm2_2026_04_15/gpu1/llama_st4_qbias_full497.json", "no_steer")
    llama_q = macro("reports/wave_pm2_2026_04_15/gpu1/llama_st4_qbias_full497.json", "ocq_qbias_b-0.1")
    llama_k = macro("reports/wave_2026_04_15_pm/gpu0/llama_inst_st4_full497.json", "ocq_bias_a0.3")

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), sharey=True)
    colors = ["#b8c4d6", "#2b6cb0", "#7aa6d8", "#c1d7f0", "#d9534f", "#f2a6a2", "#f7cfc9"]

    q_labels = [
        "No steer",
        "Q-bias\n(real)",
        "Q-bias\n(featshuffle)",
        "Q-bias\n(random)",
        "K-bias\n(real)",
        "K-bias\n(featshuffle)",
        "K-bias\n(random)",
    ]
    q_vals = [
        qwen_base["F1"],
        qwen_q["F1"],
        qwen_fshuf["F1"],
        qwen_rand["F1"],
        qwen_k["F1"],
        qwen_k_fshuf["F1"],
        qwen_k_rand["F1"],
    ]
    axes[0].bar(np.arange(len(q_vals)), q_vals, color=colors, edgecolor="black", linewidth=0.4)
    axes[0].set_title("Qwen2.5-7B-Instruct, Subtask4")
    axes[0].set_xticks(np.arange(len(q_labels)))
    axes[0].set_xticklabels(q_labels, rotation=25, ha="right")
    axes[0].set_ylabel("Macro F1")
    axes[0].set_ylim(0, 0.82)
    axes[0].grid(axis="y", alpha=0.25, linestyle="--")

    l_labels = ["No steer", "Q-bias (real)", "K-bias (real)"]
    l_vals = [llama_base["F1"], llama_q["F1"], llama_k["F1"]]
    l_colors = ["#b8c4d6", "#2b6cb0", "#d9534f"]
    axes[1].bar(np.arange(len(l_vals)), l_vals, color=l_colors, edgecolor="black", linewidth=0.4)
    axes[1].set_title("Llama-3.1-8B-Instruct, Subtask4")
    axes[1].set_xticks(np.arange(len(l_labels)))
    axes[1].set_xticklabels(l_labels, rotation=20, ha="right")
    axes[1].grid(axis="y", alpha=0.25, linestyle="--")

    fig.suptitle("Q-side coverage bias improves multi-tool F1 while stationary K-bias collapses", fontsize=12)
    fig.tight_layout()
    save_fig(fig, "fig_main_subtask4_results")


def fig_ablations() -> None:
    q_sweep = load_json("reports/qkv_joint_2026_04_15/full497_qbias_sweep.json")
    beta_points = []
    for row in q_sweep["results"]:
        if row["method"] == "no_steer":
            beta_points.append((0.0, row["macro"]["F1"]))
        elif row["method"].startswith("ocq_qbias_b-"):
            beta = float(row["method"].split("b-")[1])
            beta_points.append((-beta, row["macro"]["F1"]))
    beta_points.sort(key=lambda x: x[0])

    alpha_sweep = load_json("reports/qkv_alpha_microsweep_2026_04_15/full497_alpha_microsweep.json")
    alpha_points = []
    for row in alpha_sweep["results"]:
        if row["method"] == "no_steer":
            alpha_points.append((0.0, row["macro"]["F1"]))
        elif row["method"].startswith("ocq_qkv_a"):
            alpha = float(row["method"].split("a")[1].split("_")[0])
            alpha_points.append((alpha, row["macro"]["F1"]))
    alpha_points.sort(key=lambda x: x[0])

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.0))

    xs, ys = zip(*beta_points)
    axes[0].plot(xs, ys, marker="o", color="#2b6cb0", linewidth=2)
    axes[0].axvline(-0.1, color="#2b6cb0", linestyle=":", alpha=0.7)
    axes[0].set_title("Q-bias sweep on Qwen Subtask4")
    axes[0].set_xlabel("beta")
    axes[0].set_ylabel("Macro F1")
    axes[0].grid(alpha=0.25, linestyle="--")

    xs, ys = zip(*alpha_points)
    axes[1].plot(xs, ys, marker="o", color="#b45309", linewidth=2)
    axes[1].axvline(0.025, color="#b45309", linestyle=":", alpha=0.7)
    axes[1].set_title("Small-alpha K augmentation on top of Q-bias")
    axes[1].set_xlabel("alpha_K")
    axes[1].set_ylabel("Macro F1")
    axes[1].grid(alpha=0.25, linestyle="--")

    fig.suptitle("The useful regime is narrow: beta near -0.1 and alpha_K near 0.025-0.05", fontsize=12)
    fig.tight_layout()
    save_fig(fig, "fig_qbias_ablations")


def fig_stability_and_bound() -> None:
    eff = load_json("reports/stability_effective_magnitude_2026_04_15/result.json")
    qwen_bound = load_json("reports/theory_verify_2026_04_14/thm61_qwen_L13_a0.3_N100.json")["summary"]
    llama_bound = load_json("reports/thm61_llama_2026_04_15/llama_L15_a0.3_N100.json")["summary"]

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0))

    labels = ["Real", "Random", "Featshuffle"]
    vals = [eff["real"]["mean"], eff["random"]["mean"], eff["featshuffle"]["mean"]]
    cols = ["#2f855a", "#d69e2e", "#718096"]
    axes[0].bar(labels, vals, color=cols, edgecolor="black", linewidth=0.4)
    axes[0].set_title("Effective attention perturbation magnitude")
    axes[0].set_ylabel("Mean ||ΔK q||")
    axes[0].grid(axis="y", alpha=0.25, linestyle="--")

    ratio_labels = ["Qwen L13", "Llama L15"]
    ratio_vals = [qwen_bound["median_ratio"], llama_bound["median_ratio"]]
    axes[1].bar(ratio_labels, ratio_vals, color=["#2b6cb0", "#805ad5"], edgecolor="black", linewidth=0.4)
    axes[1].set_yscale("log")
    axes[1].set_title("Theorem 6.1 empirical bound ratio")
    axes[1].set_ylabel("Median LHS / RHS (log scale)")
    axes[1].grid(axis="y", alpha=0.25, linestyle="--")

    fig.suptitle("Real ontology perturbations are larger, yet remain the only stable direction", fontsize=12)
    fig.tight_layout()
    save_fig(fig, "fig_stability_and_bound")


def main() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "figure.titlesize": 12,
        }
    )
    fig_main_results()
    fig_ablations()
    fig_stability_and_bound()
    print(f"Wrote figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
