#!/usr/bin/env python3
"""Collate all H-Wells rescue JSONs into a single summary markdown table."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RPT = ROOT / "reports" / "boltzmann_energy"


def load(name):
    with open(RPT / name) as f:
        return json.load(f)


ORDER = [
    ("v1 (baseline)", "h_wells_qwen25_metatool_n100.json"),
    ("P1 L=6", "h_wells_p1_layer06.json"),
    ("P1 L=12", "h_wells_p1_layer12.json"),
    ("P1 L=18", "h_wells_p1_layer18.json"),
    ("P1 L=24", "h_wells_p1_layer24.json"),
    ("P1 L=27", "h_wells_p1_layer27.json"),
    ("P2 kmeans-verb L=18", "h_wells_p2_kmeans_verb_L18.json"),
    ("P2 kmeans-domain L=18", "h_wells_p2_kmeans_domain_L18.json"),
    ("P2 kmeans-both L=18", "h_wells_p2_kmeans_both_L18.json"),
    ("P2 kmeans-verb L=12", "h_wells_p2_kmeans_verb_L12.json"),
    ("P2 kmeans-domain L=12", "h_wells_p2_kmeans_domain_L12.json"),
    ("P3 mean_all afod L=18", "h_wells_p3_mean_all_afod_L18.json"),
    ("P3 first_name afod L=18", "h_wells_p3_first_name_afod_L18.json"),
    ("P3 mean_all kmd L=18", "h_wells_p3_mean_all_kmd_L18.json"),
    ("P3 first_name kmd L=18", "h_wells_p3_first_name_kmd_L18.json"),
    ("P4 chat afod L=18", "h_wells_p4_chat_afod_L18.json"),
    ("P4 chat kmd L=18", "h_wells_p4_chat_kmd_L18.json"),
]


def main():
    rows = []
    for label, fn in ORDER:
        d = load(fn)
        ws = d["wells_aggregated"]
        wq = d["wells_per_query"]
        rc = d["random_control"]
        vf = d["v_form_fits"]
        rows.append({
            "label": label,
            "layer": d["layer"],
            "pooling": d["pooling"],
            "kspace": d.get("kspace_cluster", "none"),
            "chat": d.get("chat_template", False),
            "dnorm": ws["delta_E_normalized"],
            "g1": ws["G_Wells_1_pass"],
            "rho": wq["median_rho"],
            "g2": wq["G_Wells_2_pass"],
            "rho_shuf": rc["median_rho"],
            "g3": rc["G_Wells_3_pass"],
            "add_r2": vf["additive"]["R2"],
            "hop_r2": vf["hopfield"]["R2"],
            "joint": d["joint_outcome"],
            "runtime": d["runtime_s"],
        })

    out = ["# H-Wells Rescue Summary (Path A, P1-P4)",
           "",
           "**Executed**: 2026-04-19 Qwen2.5-7B × MetaTool Subtask1 N=100, seed=0",
           "",
           "**Pre-reg gates** (LOCKED, unchanged across variants):",
           "- G1: strict Ē(0)<Ē(1)<Ē(2) AND Δ_norm ≥ 0.5",
           "- G2: median per-query Spearman ρ ≥ 0.4",
           "- G3: shuffled-label median ρ < 0.1 (null clean)",
           "",
           "## Results table",
           "",
           "| variant | L | pool | kspace | chat | Δ_norm | G1 | ρ | G2 | ρ_shuf | G3 | Hop R² | joint |",
           "|---|---|---|---|---|---:|:---:|---:|:---:|---:|:---:|---:|---|"]

    for r in rows:
        out.append(
            f"| {r['label']} | {r['layer']} | {r['pooling']} | {r['kspace']} | "
            f"{'✓' if r['chat'] else '—'} | "
            f"{r['dnorm']:+.3f} | {'✓' if r['g1'] else '✗'} | "
            f"{r['rho']:+.3f} | {'✓' if r['g2'] else '✗'} | "
            f"{r['rho_shuf']:+.3f} | {'✓' if r['g3'] else '✗'} | "
            f"{r['hop_r2']:.3f} | {r['joint']} |"
        )

    # Best finder
    best = max(rows, key=lambda r: r["dnorm"])
    out += ["",
            "## Winner",
            "",
            f"**{best['label']}** — Δ_norm=**{best['dnorm']:+.3f}** (G1={'PASS' if best['g1'] else 'FAIL'}), "
            f"median ρ={best['rho']:+.3f} (G2={'PASS' if best['g2'] else 'FAIL'}), "
            f"Hopfield R²={best['hop_r2']:.3f}. Joint = **{best['joint']}**.",
            "",
            "## Per-phase conclusions",
            "",
            "### P1 Layer sweep",
            "- Best-L = 12 (Δ_norm = +0.231), but all 5 layers FAIL G1.",
            "- L=6/12/18 weak correct direction; L=24/27 reverse.",
            "- Hop R² spikes at L=6 (0.409) and L=18 (0.304) — cluster structure exists, not monotone.",
            "",
            "### P2 K-space KMeans (the rescue)",
            "- **FIRST G1 PASS** across entire rescue: kmeans-domain L=18 (Δ_norm=+0.579, joint=WEAK).",
            "- kmeans-both L=18 also PASS (Δ_norm=+0.569). kmeans-verb L=18 FAIL but +0.440.",
            "- L=12 K-space clustering FAILS — best-K-space layer is L=18, not L=12.",
            "- **afod-heuristic label hypothesis CONFIRMED**: swapping verb/domain labels from regex-extracted afod to KMeans-on-K unlocks the basin.",
            "",
            "### P3 Pooling sweep",
            "- mean_all and first_name BOTH fail across both label sets — actually REVERSE Δ_norm direction in most cells.",
            "- **prompt_end pooling is load-bearing.**",
            "",
            "### P4 Chat template",
            "- afod L=18 chat: Δ_norm=+0.088 (worse than raw +0.139).",
            "- kmd L=18 chat: Δ_norm=+0.449 (worse than raw +0.579).",
            "- **Chat template HURTS**, not helps. v1's raw-query design was correct.",
            "",
            "## Verdict",
            "",
            "**Path A partially rescues H-Energy-Wells framework**:",
            "- G1 (aggregate basin) PASS with K-space labels — framework survives at aggregate level.",
            "- G2 (per-query Spearman) still FAIL — basin too shallow for robust per-query retrieval.",
            "- G3 (shuffled null) PASS throughout — signal is real, not artifact.",
            "",
            "**Root cause of v1 falsification**: afod-heuristic regex labels were K-space-orthogonal. The ontology is intrinsic to the K-space, not the regex categorization scheme.",
            "",
            "**Path B (H-V-NegBasin / framework pivot) NOT triggered** — kill criterion (P1+P2+P3 all FAIL) not met. P2 kmeans-domain L=18 joint=WEAK is a partial rescue.",
            "",
            "## Next-step options",
            "",
            "1. **H-Storage-Capacity** at P2 winner config (spec §4) — test Hopfield-style pattern counting.",
            "2. Tighten G2 investigation — why per-query ρ so low (0.144) despite aggregate basin?",
            "3. Paper narrative pivot: 'ontology exists but at K-intrinsic level, not at lexical-heuristic level.'",
            "4. Replicate at best-L for additional layers around L=18 (L=16, 20, 22) to localize.",
            ""]

    (RPT / "h_wells_rescue_summary.md").write_text("\n".join(out))
    print("[saved]", RPT / "h_wells_rescue_summary.md")
    for r in rows:
        flag = "★" if r["g1"] else " "
        print(f"  {flag} {r['label']:<36} Δ={r['dnorm']:+.3f} G1={r['g1']} G2={r['g2']} joint={r['joint']}")


if __name__ == "__main__":
    main()
