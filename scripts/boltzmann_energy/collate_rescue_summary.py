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
    ("D.3 BERT-KM domain L=18", "h_wells_d3_bertkm_domain_L18.json"),
    ("D.3 BERT-KM verb L=18", "h_wells_d3_bertkm_verb_L18.json"),
    ("D.3 BERT-KM both L=18", "h_wells_d3_bertkm_both_L18.json"),
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
            "## Winner (highest Δ_norm — but see D.3 tautology check below)",
            "",
            f"**{best['label']}** — Δ_norm=**{best['dnorm']:+.3f}** (G1={'PASS' if best['g1'] else 'FAIL'}), "
            f"median ρ={best['rho']:+.3f} (G2={'PASS' if best['g2'] else 'FAIL'}), "
            f"Hopfield R²={best['hop_r2']:.3f}. Joint = **{best['joint']}**.",
            "",
            "## D.3 BERT-KM defense — TAUTOLOGY CHECK",
            "",
            "Pre-reg tier (locked before D.3 run):",
            "- Δ ≥ 0.30: cross-feature semantic basin (paper tier 5.0-6.0)",
            "- 0.15 ≤ Δ < 0.30: ambiguous (paper tier 4.0-5.0)",
            "- Δ < 0.15: tautology confirmed, pure negative (paper tier 3.5-4.5)",
            "",
            "BERT-KM Δ_norm at L=18 prompt_end (best of {verb, domain, both}):",
            "- domain: +0.143 (best, BARELY below 0.15 boundary)",
            "- verb: +0.128 (non-monotone)",
            "- both: +0.048",
            "",
            "**TAUTOLOGY CONFIRMED.** afod-domain (+0.139) ≈ BERT-KM-domain (+0.143) << Qwen-K-self-KMeans-domain (+0.579, 4× larger). Two independent semantic spaces (lexical regex, BERT embedding) BOTH produce identical near-zero basin in Qwen K-space; only K-self-derived labels lift the signal. The P2 'rescue' was self-similarity by construction, not semantic structure.",
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
            "## Verdict (post-D.3)",
            "",
            "**Path A FAILS to rescue H-Energy-Wells framework once D.3 tautology check is applied:**",
            "- P2 kmeans-domain L=18 G1 PASS = self-similarity artifact (Δ=+0.579 with K-self labels collapses to Δ=+0.143 with BERT-independent labels of same K).",
            "- afod (+0.139) ≈ BERT (+0.143) — two independent semantic spaces both fail. Real semantic basin in Qwen attention K-space at L=18 ≈ 0.14σ ≈ noise.",
            "- G2 (per-query Spearman) FAIL throughout — even the artifact-inflated P2 winner only reaches +0.144.",
            "- G3 (shuffled null) PASS throughout — what little signal exists IS structured, just very weak.",
            "",
            "**Triple negative**: afod fail + BERT fail + G2 fail. **Tautology confirmed via D.3.**",
            "",
            "**Kill criterion technically NOT triggered** (G1 PASS exists, even if artifact), but D.3 reveals it as construction artifact. Effective interpretation: H-Energy-Wells v1 framework is **dead at semantic-basin level**, regardless of methodology variant.",
            "",
            "## Next-step options",
            "",
            "1. **Paper Option B (pure negative)** — frame as falsification with mechanistic ablations: 'Tool-selection ontology does NOT exist as Hopfield basin in Qwen2.5 attention K-space; reported aggregates under self-derived labels are tautological. afod, BERT, and per-query retrieval all fail.'",
            "2. **Path B brainstorm RE-ACTIVATES**: H-V-NegBasin (anti-basin), FEP (Friston), Gärdenfors conceptual spaces, or abandon ontology axis (attention-only measurements).",
            "3. **Cross-layer + cross-model replication** before paper writing: confirm BERT-KM Δ ≈ afod Δ at L∈{6,12,24} too (if so, framework dead globally) and on Qwen2.5-1.5B / Llama-3-8B.",
            "4. **G2-first redesign**: rebuild framework around per-query metric (e.g., rank of GT in nearest-N) rather than aggregate basin.",
            ""]

    (RPT / "h_wells_rescue_summary.md").write_text("\n".join(out))
    print("[saved]", RPT / "h_wells_rescue_summary.md")
    for r in rows:
        flag = "★" if r["g1"] else " "
        print(f"  {flag} {r['label']:<36} Δ={r['dnorm']:+.3f} G1={r['g1']} G2={r['g2']} joint={r['joint']}")


if __name__ == "__main__":
    main()
