import os

import matplotlib.pyplot as plt
import numpy as np


def build_readiness(out_dir: str) -> None:
    claims = [
        "Anisotropy observed",
        "PCA static axis",
        "Dynamic schedule exists",
        "Preliminary PPL signal",
        "Standard PPL",
        "External baselines",
        "Large-model GQA",
    ]
    scores = np.array([0.95, 0.90, 0.78, 0.62, 0.25, 0.20, 0.15])

    colors = []
    for s in scores:
        if s >= 0.75:
            colors.append("#2E7D32")
        elif s >= 0.45:
            colors.append("#F9A825")
        else:
            colors.append("#9E9E9E")

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    y = np.arange(len(claims))
    ax.barh(y, scores, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_yticks(y, claims, fontsize=10)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Claim readiness")
    ax.set_title("FOKVQ draft readiness map")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.invert_yaxis()
    for yi, s in zip(y, scores):
        ax.text(min(s + 0.02, 0.98), yi, f"{s:.2f}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "evidence_readiness.pdf"))
    fig.savefig(os.path.join(out_dir, "evidence_readiness.png"), dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(base, "figures")
    os.makedirs(out_dir, exist_ok=True)
    build_readiness(out_dir)
