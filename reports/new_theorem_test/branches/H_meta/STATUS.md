# H-Meta — Meta-attention layer Status

**Status**: **dormant** (2026-04-19). Gate closed by F13b +3.85pp. Gate required F12<+3 AND F13<+3 — F13 exceeded. **Do not execute.** F13 FunnelRot now occupies the "ontology-conditioned attention modulation" slot; separate meta-layer architecture unjustified.
**Primary spec**: memory `user_5question_synthesis_f14_metafocus_2026_04_19.md` §5 H-Meta
**Concept**: frozen LLM + ontology-axis meta-attention layer (LoRA rank comparable). Ontology-typed Q/K routing.
**Null control**: identity rotation + identity gate = same-capacity LoRA with random ontology labels.
**Gate**: `F12 < +3pp` AND `F13 < +3pp` (otherwise F-series already wins; H-Meta cost not justified).
**Decision**: `meta > pure-LoRA baseline by ≥ +2pp` → qualitatively distinct routing verified (Q5 partial answer).
**Preemption risk** (3-agent audit):
  - HHGT (WSDM 2025) — ontology-typed attention, but trained-from-scratch vs frozen
  - LUKE (EMNLP 2020), K-BERT (AAAI 2020) — entity-typed attention; training-phase
  - Conceptors (NeurIPS 2024) — composable steering; linear projection only
  → Differentiation: **training-free + frozen-base + ontology-axis + multi-facet** combination is empty cell.
**Cost**: 10–15 GPU-hr (LoRA train + eval).
**Log**: (empty)
